"""Sparse retrieval (BM25) + hybrid fusion (RRF)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

SPARSE_FILENAME = "_sparse_chunks.jsonl"


@dataclass
class HybridRankedItem:
    doc: Document
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_distance: float | None
    sparse_score: float | None


def sparse_index_path(persist_dir: str) -> Path:
    return Path(persist_dir).resolve() / SPARSE_FILENAME


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _dedupe_key(doc: Document) -> str:
    md = doc.metadata or {}
    source = str(md.get("source", ""))
    page = str(md.get("page", ""))
    content = doc.page_content.strip()
    head = content[:220]
    return f"{source}|{page}|{head}"


def rebuild_sparse_index_from_vectorstore(vectorstore: Chroma, persist_dir: str) -> int:
    """Rebuild sparse JSONL artifact from all current chunks in Chroma."""
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        raise RuntimeError("Could not access Chroma collection to build sparse index.")

    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    out_path = sparse_index_path(persist_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i in range(len(ids)):
        text = str(docs[i]) if i < len(docs) and docs[i] is not None else ""
        if not text.strip():
            continue
        md = metas[i] if i < len(metas) and metas[i] is not None else {}
        rows.append(
            {
                "chunk_id": str(ids[i]),
                "text": text,
                "source": md.get("source"),
                "page": md.get("page"),
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def sparse_search(query: str, persist_dir: str, k: int) -> list[tuple[Document, float]]:
    """Return sparse top-k as (Document, bm25_score)."""
    path = sparse_index_path(persist_dir)
    if not path.exists():
        return []

    rows: list[dict] = []
    tokenized_corpus: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            rows.append(row)
            tokenized_corpus.append(_tokenize(text))

    if not rows:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]

    out: list[tuple[Document, float]] = []
    for idx in ranked_idx:
        score = float(scores[idx])
        if score <= 0:
            continue
        row = rows[idx]
        doc = Document(
            page_content=str(row.get("text", "")),
            metadata={
                "source": row.get("source"),
                "page": row.get("page"),
                "chunk_id": row.get("chunk_id"),
            },
        )
        out.append((doc, score))
    return out


def hybrid_search(
    vectorstore: Chroma,
    query: str,
    *,
    k_final: int,
    persist_dir: str,
    k_dense: int | None = None,
    k_sparse: int | None = None,
    rrf_k: int = 60,
) -> list[HybridRankedItem]:
    """Fuse dense (Chroma) + sparse (BM25) rankings via Reciprocal Rank Fusion."""
    kd = max(k_final, k_dense or (k_final * 2))
    ks = max(k_final, k_sparse or (k_final * 2))
    dense = vectorstore.similarity_search_with_score(query, k=kd)
    sparse = sparse_search(query, persist_dir=persist_dir, k=ks)

    fused: dict[str, HybridRankedItem] = {}
    for rank, (doc, distance) in enumerate(dense, start=1):
        key = _dedupe_key(doc)
        score = 1.0 / (rrf_k + rank)
        item = fused.get(
            key,
            HybridRankedItem(
                doc=doc,
                rrf_score=0.0,
                dense_rank=None,
                sparse_rank=None,
                dense_distance=None,
                sparse_score=None,
            ),
        )
        item.rrf_score += score
        item.dense_rank = rank
        item.dense_distance = float(distance)
        fused[key] = item

    for rank, (doc, bm25_score) in enumerate(sparse, start=1):
        key = _dedupe_key(doc)
        score = 1.0 / (rrf_k + rank)
        item = fused.get(
            key,
            HybridRankedItem(
                doc=doc,
                rrf_score=0.0,
                dense_rank=None,
                sparse_rank=None,
                dense_distance=None,
                sparse_score=None,
            ),
        )
        item.rrf_score += score
        item.sparse_rank = rank
        item.sparse_score = float(bm25_score)
        fused[key] = item

    ranked = sorted(fused.values(), key=lambda x: x.rrf_score, reverse=True)
    return ranked[:k_final]

