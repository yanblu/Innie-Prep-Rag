"""Shared Chroma query helpers for eval scripts (no LangChain)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from book_coach.config import CHROMA_LANGCHAIN_COLLECTION, EMBEDDING_MODEL

SPARSE_FILENAME = "_sparse_chunks.jsonl"


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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _sparse_index_rows(persist_dir: Path) -> list[dict]:
    path = persist_dir / SPARSE_FILENAME
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def query_sparse(
    persist_dir: Path,
    question: str,
    k: int,
) -> list[tuple[str, float, dict]]:
    """Return sparse (BM25) ranking as (document_text, bm25_score, metadata)."""
    rows = _sparse_index_rows(persist_dir)
    if not rows:
        return []

    tokenized_corpus = [_tokenize(str(r.get("text", ""))) for r in rows]
    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens = _tokenize(question)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]

    out: list[tuple[str, float, dict]] = []
    for idx in ranked_idx:
        score = float(scores[idx])
        if score <= 0:
            continue
        row = rows[idx]
        md = {
            "source": row.get("source"),
            "page": row.get("page"),
            "chunk_id": row.get("chunk_id"),
        }
        out.append((str(row.get("text", "")), score, md))
    return out


def _dedupe_key(text: str, md: dict) -> str:
    source = str(md.get("source", ""))
    page = str(md.get("page", ""))
    head = text.strip()[:220]
    return f"{source}|{page}|{head}"


def query_hybrid_rrf(
    persist_dir: Path,
    question: str,
    k: int,
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float, dict]]:
    """Hybrid retrieval via RRF fusion over dense + sparse ranks."""
    kd = max(k, k * 2)
    ks = max(k, k * 2)
    dense = query_chroma(persist_dir, question, kd)
    sparse = query_sparse(persist_dir, question, ks)

    fused: dict[str, dict] = {}
    for rank, (text, _dist, md) in enumerate(dense, start=1):
        key = _dedupe_key(text, md)
        row = fused.setdefault(
            key,
            {
                "text": text,
                "md": dict(md),
                "score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            },
        )
        row["score"] += 1.0 / (rrf_k + rank)
        row["dense_rank"] = rank

    for rank, (text, _bm25, md) in enumerate(sparse, start=1):
        key = _dedupe_key(text, md)
        row = fused.setdefault(
            key,
            {
                "text": text,
                "md": dict(md),
                "score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            },
        )
        row["score"] += 1.0 / (rrf_k + rank)
        row["sparse_rank"] = rank

    ranked = sorted(fused.values(), key=lambda x: float(x["score"]), reverse=True)[:k]
    out: list[tuple[str, float, dict]] = []
    for row in ranked:
        md = dict(row["md"])
        md["dense_rank"] = row["dense_rank"]
        md["sparse_rank"] = row["sparse_rank"]
        out.append((str(row["text"]), float(row["score"]), md))
    return out


def query_ranked(
    persist_dir: Path,
    question: str,
    k: int,
    *,
    retrieval_mode: str = "dense",
) -> list[tuple[str, float, dict]]:
    """Unified retrieval query for eval runners."""
    mode = retrieval_mode.strip().lower()
    if mode == "hybrid":
        return query_hybrid_rrf(persist_dir, question, k)
    if mode == "dense":
        return query_chroma(persist_dir, question, k)
    raise SystemExit(f"Invalid retrieval mode: {retrieval_mode!r}. Use 'dense' or 'hybrid'.")


def first_gold_rank(
    ranked: list[tuple[str, float, dict]],
    gold: set[int],
) -> int | None:
    for i, (_t, _d, md) in enumerate(ranked, start=1):
        page = md.get("page")
        if page is not None and int(page) in gold:
            return i
    return None
