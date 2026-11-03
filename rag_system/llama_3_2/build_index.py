""" Costruisce l'indiice vettoriale del sistema RAG

La pipeline è comune ai diversi corsi, mentre documenti e
database vettoriale vengono mantenuti separati

uso: python build_index.py sas / progr1
"""

import chromadb
import sys

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings
)
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# SELEZIONE DEL CORSO
if len(sys.argv) != 2:

    print(
        "Uso: python build_index.py [sas|progr1]"
    )

    sys.exit(1)


COURSE = sys.argv[1].lower()


if COURSE not in ["sas", "progr1"]:

    print(
        "Errore: il corso deve essere 'sas' oppure 'progr1'."
    )

    sys.exit(1)



# PERCORSI
DOCUMENTS_DIR = ( BASE_DIR / "data" / COURSE / "documents")

CHROMA_DIR = (BASE_DIR / "rag_system" / "llama_3_2" / "chroma_db" / COURSE)


print(f"\nCorso selezionato: {COURSE}")
print(f"Documenti: {DOCUMENTS_DIR}")
print(f"ChromaDB: {CHROMA_DIR}")


# Controllo che la cartella dei documenti esista
if not DOCUMENTS_DIR.exists():

    print(
        f"\nErrore: la cartella dei documenti non esiste:\n"
        f"{DOCUMENTS_DIR}"
    )

    sys.exit(1)


# EMBEDDING
Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)


# CHUNKING
Settings.text_splitter = TokenTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


# CARICAMENTO DOCUMENTI
pdf_reader = PyMuPDFReader()

documents = SimpleDirectoryReader(
    input_dir=str(DOCUMENTS_DIR),
    file_extractor={
        ".pdf": pdf_reader
    }
).load_data()


print(
    f"\nDocumenti caricati: {len(documents)}"
)


# CHROMADB
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# Se esiste già una collection precedente,
# la elimino per evitare di duplicare i chunk.

try:

    chroma_client.delete_collection(
        name="tesi_rag"
    )

    print(
        "Collection precedente eliminata."
    )

except Exception:

    pass


# Creo una nuova collection
chroma_collection = chroma_client.create_collection(
    name="tesi_rag"
)


# VECTOR STORE
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)


storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# CHUNKING
nodes = Settings.text_splitter.get_nodes_from_documents(
    documents
)

print(
    f"Chunk creati: {len(nodes)}"
)

# CREAZIONE INDICE
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=Settings.embed_model,
    show_progress=True
)


print(
    f"\nIndice {COURSE} creato e salvato in ChromaDB."
)

print(
    f"Percorso: {CHROMA_DIR}"
)