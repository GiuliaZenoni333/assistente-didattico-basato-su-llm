"""
Esegue la valutazione del sistema RAG
con reranking dei risultati recuperati
per il corso di Programmazione 1
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
    BASE_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K
)


QUESTIONS_FILE = (BASE_DIR / "evaluation" / "questions" / "progr1" / "questions_progr1_final.jsonl")

RESULTS_FILE = (BASE_DIR / "evaluation" / "llama_3.2" / "progr1" / "results" / "reranker" / "llamaindex_chroma_reranker_TEMP.jsonl")

CHROMA_DIR = (BASE_DIR / "rag_system" / "llama_3_2" / "chroma_db" / "progr1")

# Controllo che il dataset esista
if not QUESTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Dataset non trovato: {QUESTIONS_FILE}"
    )


# Controllo che il database ChromaDB esista
if not CHROMA_DIR.exists():
    raise FileNotFoundError(
        f"Database ChromaDB non trovato: {CHROMA_DIR}"
    )


print(f"\nDataset: {QUESTIONS_FILE}")
print(f"ChromaDB: {CHROMA_DIR}")


# Uso lo stesso modello di embedding utilizzato
# durante la costruzione dell'indice
Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)


# Apro il database ChromaDB di Programmazione 1
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# Recupero la collection contenente i chunk
chroma_collection = chroma_client.get_collection(
    name="tesi_rag"
)


# Collego ChromaDB a LlamaIndex
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)


# Ricostruisco l'indice a partire dai dati già salvati
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)


# Creo il retriever
retriever = index.as_retriever(
    similarity_top_k=TOP_K
)


# Configuro il reranker
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=512
)


# Configuro Llama 3.2 eseguito localmente tramite Ollama
llm = Ollama(
    model=LLM_MODEL,
    request_timeout=600.0,
    temperature=0
)


# Creo la cartella dei risultati se non esiste
RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# Eseguo la valutazione su tutte le domande
with open(
    QUESTIONS_FILE,
    "r",
    encoding="utf-8"
) as questions_file, open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as results_file:

    for line in questions_file:

        line = line.strip()

        if not line:
            continue

        question_data = json.loads(line)

        if question_data["id"] not in IDS_TO_RUN:
            continue


        question = question_data["question"]
        question_type = question_data["question_type"]
        difficulty = question_data["difficulty"]

        print(
            f"\nDomanda {question_data['id']} "
            f"[{question_type}, {difficulty}]: "
            f"{question}"
        )


        # Recupero iniziale dei chunk
        results = retriever.retrieve(question)


        # Costruisco le coppie domanda-chunk
        # utilizzate dal reranker
        pairs = []

        for result in results:

            pairs.append(
                [
                    question,
                    result.node.get_content()
                ]
            )


        # Calcolo il punteggio di rilevanza
        # assegnato dal reranker
        rerank_scores = reranker.predict(
            pairs
        )


        # Riordino i chunk in base al punteggio
        # assegnato dal reranker
        reranked_results = sorted(
            zip(results, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )


        # Salvo le fonti recuperate e riordinate
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


        # Costruisco il contesto utilizzando
        # l'ordine determinato dal reranker
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


        # Costruisco il prompt
        # Il modello NON riceve question_type, difficulty,
        # expected_answer o expected_sources
        prompt = f"""
Sei un assistente didattico universitario.

Devi rispondere alla DOMANDA utilizzando esclusivamente
le informazioni presenti nel CONTENUTO fornito.

REGOLE:

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

Se la domanda contiene una premessa errata,
non assumere che la premessa sia vera.

Se il CONTENUTO permette di correggere la premessa,
spiega chiaramente l'errore utilizzando esclusivamente
le informazioni presenti nel CONTENUTO.

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


        # Genero la risposta
        response = llm.complete(prompt)

        response_text = str(
            response
        ).strip()


        # Salvo il risultato
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