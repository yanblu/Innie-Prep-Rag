"""Shared constants — no third-party imports (safe for lightweight eval scripts)."""

EMBEDDING_MODEL = "text-embedding-3-small"
# Must match across ingest, loader, and any direct chromadb clients
CHROMA_LANGCHAIN_COLLECTION = "langchain"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
