"""
Configurazione della pipeline RAG con Phi-4-mini.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "bge-m3"

LLM_MODEL = "phi4-mini"

TOP_K = 8

CHUNK_SIZE = 700

CHUNK_OVERLAP = 120

CHROMA_BASE_DIR = (BASE_DIR / "rag_system" / "llama_3_2" / "chroma_db")