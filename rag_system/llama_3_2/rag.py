""" Esegue il sistema RAG completo

La domanda viene usata per recuperare da ChromaDB i chunk più pertinenti.
I chunk recuperati vengono poi forniti come contesto a LLama 3.2,
per generare la risposta finale

Adatta per entrambi i corsi
Uso:
    python rag.py sas
    python rag.py progr1
"""
import chromadb
import sys

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

from config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K
)


# Controllo che sia stato specificato il corso
if len(sys.argv) != 2:
    print("Uso: python rag.py [sas|progr1]")
    sys.exit(1)


# Recupero il corso dal comando
COURSE = sys.argv[1].lower()


# Controllo che il corso sia valido
if COURSE not in ["sas", "progr1"]:
    print("Errore: il corso deve essere 'sas' oppure 'progr1'.")
    sys.exit(1)


# Costruisco il percorso del database ChromaDB
# in base al corso selezionato
CHROMA_DIR = (BASE_DIR / "rag_system" / "llama_3_2" / "chroma_db" / COURSE)


print(f"\nCorso selezionato: {COURSE}")
print(f"ChromaDB: {CHROMA_DIR}")


# Uso lo stesso modello di embedding utilizzato
# durante la costruzione dell'indice
Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)


# Apro il database vettoriale del corso selezionato
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


# Creo il retriever che recupera i TOP_K chunk più pertinenti
retriever = index.as_retriever(
    similarity_top_k=TOP_K
)


# Configuro Llama 3.2 eseguito localmente tramite Ollama
llm = Ollama(
    model=LLM_MODEL,
    request_timeout=120.0
)


# Chiedo la domanda all'utente
query = input("\nInserisci una domanda: ")


# Recupero i chunk più pertinenti
results = retriever.retrieve(query)

# Unisco il testo dei chunk recuperati in un unico contesto
context_parts = []


for position, result in enumerate(results, start=1):

    metadata = result.node.metadata

    source = (
        f"Fonte {position} - "
        f"Documento: {metadata.get('file_name', 'Non disponibile')}, "
        f"Pagina: {metadata.get('source', 'Non disponibile')}"
    )

    context_parts.append(
        f"{source}\n{result.node.get_content()}"
    )


context = "\n\n".join(context_parts)


# Costruisco il prompt che verrà inviato a Llama
prompt = f"""
Sei un assistente didattico.

Rispondi alla domanda utilizzando esclusivamente le informazioni
contenute nel materiale fornito nel CONTENUTO.

Non aggiungere informazioni provenienti dalle tue conoscenze generali.

Se il contenuto non contiene informazioni sufficienti per rispondere,
rispondi esattamente:
"Non trovo informazioni sufficienti nel materiale fornito."

CONTENUTO:
{context}

DOMANDA:
{query}

RISPOSTA:
"""


# Invio il prompt a Llama tramite Ollama
response = llm.complete(prompt)


# Mostro la risposta finale
print("\n--- RISPOSTA RAG ---\n")
print(response)


# Mostro anche le fonti recuperate dal retrieval
print("\n--- FONTI RECUPERATE ---")


for position, result in enumerate(results, start=1):

    metadata = result.node.metadata

    print(
        f"{position}. "
        f"{metadata.get('file_name', 'Documento non disponibile')} - "
        f"pagina {metadata.get('source', 'N/D')} - "
        f"score {result.score:.4f}"
    )