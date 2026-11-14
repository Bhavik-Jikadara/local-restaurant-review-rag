"""Central settings. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
DB_DIR = Path(os.getenv("RAG_DB_DIR", BASE_DIR / "chroma_langchain_db"))
MANIFEST_PATH = DB_DIR / "manifest.json"
COLLECTION_NAME = "restaurant_reviews"

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3-coder:30b")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))
LLM_NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192"))

# Retrieval
TOP_K = int(os.getenv("RAG_TOP_K", "6"))            # reviews sent to the LLM
VECTOR_FETCH_K = int(os.getenv("RAG_FETCH_K", "20"))  # candidates for MMR
KEYWORD_K = int(os.getenv("RAG_KEYWORD_K", "8"))      # BM25 candidates
MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))  # 1 = relevance, 0 = diversity
RRF_CONSTANT = 60

# Ingestion
REQUIRED_COLUMNS = {"Title", "Review", "Rating", "Date"}
OPTIONAL_COLUMNS = {"Restaurant"}  # falls back to the CSV file name
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 64
# Bump to force a full re-index after changing how documents are built.
INDEX_VERSION = 2

# Conversation
HISTORY_TURNS = 4
