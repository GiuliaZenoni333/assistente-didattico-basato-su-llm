"""
Interroga l'indice vettoriale creato con LlamaIndex e ChromaDB.

La domanda viene trasformata in embedding con lo stesso modello
BGE-M3 utilizzato durante l'indicizzazione. ChromaDB recupera
i TOP_K chunk più pertinenti, che vengono mostrati nel terminale.

Il file può essere utilizzato per entrambi i corsi.

Uso:
    python ask.py sas
    python ask.py progr1
"""

import sys
import chromadb

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    TOP_K
)


# Controllo che sia stato specificato il corso
if len(sys.argv) != 2:
    print("Uso: python ask.py [sas|progr1]")
    sys.exit(1)


# Recupero il corso dal comando
COURSE = sys.argv[1].lower()


# Controllo che il corso sia valido
if COURSE not in ["sas", "progr1"]:
    print("Errore: il corso deve essere 'sas' oppure 'progr1'.")
    sys.exit(1)


# Costruisco il percorso del database ChromaDB
# in base al corso selezionato
CHROMA_DIR = (
   BASE_DIR / "rag_system" / "llama_3_2" / "chroma_db" / COURSE
)

print(f"\nCorso selezionato: {COURSE}")
print(f"ChromaDB: {CHROMA_DIR}")


# Uso lo stesso modello di embedding utilizzato
# durante la costruzione dell'indice
Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)


# Apro il database ChromaDB del corso selezionato
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# Recupero la collection che contiene i chunk
# e i relativi embedding
chroma_collection = chroma_client.get_collection(
    name="tesi_rag"
)


# Collego ChromaDB a LlamaIndex
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)


# Ricostruisco l'indice usando i dati già presenti
# in ChromaDB senza rigenerare gli embedding
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)


# Creo il retriever e gli dico di restituire
# i TOP_K chunk più pertinenti
retriever = index.as_retriever(
    similarity_top_k=TOP_K
)


# Chiedo all'utente una domanda da terminale
query = input("\nInserisci una domanda: ")


# Eseguo il retrieval
results = retriever.retrieve(query)


# Mostro i chunk recuperati
for position, result in enumerate(results, start=1):

    metadata = result.node.metadata

    print(f"\n--- RISULTATO {position} ---")

    # Mostro la similarità tra domanda e chunk
    print(f"Score: {result.score}")

    # Mostro il documento di provenienza
    print(
        f"Documento: "
        f"{metadata.get('file_name', 'Non disponibile')}"
    )

    # Mostro la pagina di provenienza
    print(
        f"Pagina: "
        f"{metadata.get('source', 'Non disponibile')}"
    )

    # Mostro il testo del chunk recuperato
    print("\nTesto:")
    print(result.node.get_content())