"""Shared Chroma query helpers for eval scripts (no LangChain)."""

from __future__ import annotations

import os
from pathlib import Path

from embed_config import CHROMA_LANGCHAIN_COLLECTION, EMBEDDING_MODEL


def human_pages_to_meta(human_pages: list[int]) -> set[int]:
    return {int(p) - 1 for p in human_pages}


def query_chroma(
    persist_dir: Path,
    question: str,
    k: int,
) -> list[tuple[str, float, dict]]:
    """Return list of (document_text, distance, metadata) best-first."""
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (.env or environment).")

    ef = OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        col = client.get_collection(
            name=CHROMA_LANGCHAIN_COLLECTION,
            embedding_function=ef,
        )
    except Exception as e:
        names = [c.name for c in client.list_collections()]
        raise SystemExit(
            f"Could not open collection {CHROMA_LANGCHAIN_COLLECTION!r}: {e}\n"
            f"Collections present: {names or '(none)'}"
        ) from e

    r = col.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )
    docs = (r.get("documents") or [[]])[0]
    dists = (r.get("distances") or [[]])[0]
    metas = (r.get("metadatas") or [[]])[0]
    out: list[tuple[str, float, dict]] = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) and metas[i] is not None else {}
        dist = float(dists[i]) if i < len(dists) else 0.0
        out.append((docs[i], dist, dict(meta)))
    return out


def first_gold_rank(
    ranked: list[tuple[str, float, dict]],
    gold: set[int],
) -> int | None:
    for i, (_t, _d, md) in enumerate(ranked, start=1):
        page = md.get("page")
        if page is not None and int(page) in gold:
            return i
    return None
