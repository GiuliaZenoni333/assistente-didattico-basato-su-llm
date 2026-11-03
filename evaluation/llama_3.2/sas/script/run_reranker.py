"""
Esegue la valutazione del sistema RAG
con reranking dei risultati recuperati
"""

import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import chromadb
from sentence_transformers import CrossEncoder

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

# Aggiungo la root del progetto al percorso Python
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.llama_3_2.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K
)


QUESTIONS_FILE = (PROJECT_ROOT / "evaluation" / "questions" / "sas" / "questions_sas_final.jsonl")

RESULTS_FILE = (PROJECT_ROOT / "evaluation" / "llama_3.2" / "sas" / "results" / "reranker" / "llamaindex_chroma_reranker.jsonl")

CHROMA_DIR = (PROJECT_ROOT / "rag_system" / "llama_3_2" / "chroma_db" / "sas")


# EMBEDDING
Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)


# CHROMADB
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

chroma_collection = chroma_client.get_collection(
    name="tesi_rag"
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)

retriever = index.as_retriever(
    similarity_top_k=TOP_K
)


# RERANKER

reranker = CrossEncoder (
    "BAAI/bge-reranker-v2-m3",
    max_length=512
)



# LLM
llm = Ollama(
    model=LLM_MODEL,
    request_timeout=900.0,
    temperature=0
)


# CARTELLA RISULTATI
RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# VALUTAZIONE
# Recupera gli ID delle domande già completate
completed_ids = set()

if RESULTS_FILE.exists():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                completed_ids.add(data["id"])

print(f"Domande già completate: {len(completed_ids)}")

with open(
    QUESTIONS_FILE,
    "r",
    encoding="utf-8"
) as questions_file, open(
    RESULTS_FILE,
    "a",
    encoding="utf-8"
) as results_file:

    for line in questions_file:

        question_data = json.loads(line)

        # Salta le domande già completate
        if question_data["id"] in completed_ids:
            continue

        question = question_data["question"]
        question_type = question_data["question_type"]
        difficulty = question_data["difficulty"]

        print(
            f"\nDomanda {question_data['id']} "
            f"[{question_type}, {difficulty}]: "
            f"{question}"
        )


        # RETRIEVAL

        results = retriever.retrieve(question)


        # RERANKING

        pairs = []

        for result in results:

            pairs.append(
                [
                    question,
                    result.node.get_content()
                ]
            )

        rerank_scores = reranker.predict(
            pairs
        )


        reranked_results = sorted(
            zip(results, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )


        # FONTI RECUPERATE E RIORDINATE

        retrieved_sources = []

        for rank, (result, rerank_score) in enumerate(
            reranked_results,
            start=1
        ):

            metadata = result.node.metadata

            source = {
                "rank": rank,
                "file": metadata.get(
                    "file_name",
                    "Non disponibile"
                ),
                "page": metadata.get(
                    "source",
                    "Non disponibile"
                ),
                "retrieval_score": result.score,
                "rerank_score": float(rerank_score)
            }

            retrieved_sources.append(
                source
            )


        # COSTRUZIONE DEL CONTENUTO

        context_parts = []

        for position, (result, rerank_score) in enumerate(
            reranked_results,
            start=1
        ):

            source = retrieved_sources[position - 1]

            context_parts.append(
                f"Fonte {position} - "
                f"Documento: {source['file']}, "
                f"Pagina: {source['page']}\n"
                f"{result.node.get_content()}"
            )

        context = "\n\n".join(
            context_parts
        )


        # PROMPT

        prompt = f"""
Sei un assistente didattico universitario.

Devi rispondere alla DOMANDA utilizzando esclusivamente
le informazioni presenti nel CONTENUTO fornito.


REGOLE GENERALI:

- Cerca nel CONTENUTO tutte le informazioni pertinenti
  alla domanda.
- Se la risposta è presente nel CONTENUTO, devi rispondere.
- Puoi combinare informazioni provenienti da più fonti.
- Puoi riformulare le informazioni presenti nel CONTENUTO.
- Non è necessario che le parole della domanda coincidano
  con quelle utilizzate nei documenti.
- Se la domanda richiede un confronto tra concetti,
  collega le informazioni pertinenti presenti nel CONTENUTO.
- Non utilizzare conoscenze esterne al CONTENUTO.
- Non inventare informazioni.

DOMANDE CON PREMESSA ERRATA:

Se la domanda contiene una premessa non supportata
o contraddetta dal CONTENUTO, non assumere che la premessa
sia vera.

Indica chiaramente il problema e, quando possibile,
correggi la premessa utilizzando esclusivamente
le informazioni presenti nel CONTENUTO.

DOMANDE SENZA RISPOSTA NEL MATERIALE:

Se il CONTENUTO non contiene realmente le informazioni
necessarie per rispondere alla domanda, non inventare
una risposta.

In questo caso rispondi esattamente:

"Non trovo informazioni sufficienti nel materiale fornito."

IMPORTANTE:

Non dichiarare che mancano informazioni quando il CONTENUTO
contiene informazioni sufficienti per costruire la risposta.

CONTENUTO:
{context}

DOMANDA:
{question}

RISPOSTA:
"""


        # GENERAZIONE DELLA RISPOSTA

        response = llm.complete(prompt)

        response_text = str(
            response
        ).strip()


        # SALVATAGGIO

        output = {
            "id": question_data["id"],
            "question": question,
            "question_type": question_type,
            "difficulty": difficulty,
            "expected_answer": question_data["expected_answer"],
            "expected_sources": question_data["expected_sources"],
            "response": response_text,
            "retrieved_sources": retrieved_sources
        }

        results_file.write(
            json.dumps(
                output,
                ensure_ascii=False
            ) + "\n"
        )
        results_file.flush()

        print(
            "Risposta:",
            response_text
        )


print(
    "\nValutazione con reranker completata."
)

print(
    f"Risultati salvati in: {RESULTS_FILE}"
)