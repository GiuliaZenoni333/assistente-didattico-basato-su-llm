""" Configurazione del sistema RAG 
La pipeline può essere utilizzata su più corsi mantenendo
separati i rispettivi documenti e indici vettoriali
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "llama3.2:3b"

TOP_K = 8
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120