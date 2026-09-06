"""Load PDFs, chunk, embed with OpenAI, persist to Chroma (append or full reset)."""

import book_coach.warn_filters  # noqa: F401

import os
import shutil
from uuid import uuid4
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from book_coach.chroma_lifecycle import ensure_chroma_tree_writable
from book_coach.config import (
    CHROMA_LANGCHAIN_COLLECTION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL,
)
from book_coach.hybrid_retrieval import rebuild_sparse_index_from_vectorstore

load_dotenv()


class AppendStats(TypedDict):
    """Summary of an append/reindex operation."""

    chunks_added: int
    files_processed: int
    files_replaced: int
    files_new: int


def chroma_persist_populated(persist_dir: str) -> bool:
    """True if persist directory exists and is non-empty (rough proxy for an index on disk)."""
    p = Path(persist_dir)
    if not p.is_dir():
        return False
    try:
        return any(p.iterdir())
    except OSError:
        return False


def _ensure_persist_parent_writable(persist_dir: Path) -> None:
    parent = persist_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not os.access(parent, os.W_OK):
        raise PermissionError(
            f"Folder not writable (Chroma/SQLite cannot save the index): {parent}\n"
            "Fix: chmod -R u+w .  |  move project off iCloud Desktop  |  run from a directory you own."
        )


def _rmtree_chroma(p: Path) -> None:
    if not p.exists():
        return
    try:
        shutil.rmtree(p)
    except OSError as e:
        raise PermissionError(
            f"Cannot delete index at {p} (close Streamlit, then: chmod -R u+w {p} or rm -rf chroma_db)."
        ) from e


def _load_split_one_pdf(
    pdf_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {pdf_path}")
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    source = str(path)
    for doc in splits:
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["source"] = source
    return splits


def reset_knowledge_base(persist_dir: str = "chroma_db") -> None:
    """Delete the entire vector index on disk. Safe if the directory does not exist."""
    p = Path(persist_dir).resolve()
    _ensure_persist_parent_writable(p)
    if p.exists():
        ensure_chroma_tree_writable(p)
    _rmtree_chroma(p)


def append_pdfs(
    pdf_paths: Sequence[str],
    persist_dir: str = "chroma_db",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> AppendStats:
    """Add chunks from one or more PDFs to the shared index. Creates a new index if none exists.

    Returns ingest stats (chunks and file replacement/new counts).

    Per-file dedupe behavior: if a source PDF already exists in the shared index, replacement
    chunks are added before the old chunks are removed. A failed insert therefore preserves the
    prior indexed source rather than deleting it first.
    """
    resolved_paths = [str(Path(p).expanduser().resolve()) for p in pdf_paths if str(p).strip()]
    # Keep first-seen order and avoid duplicate work within a single submit.
    paths = list(dict.fromkeys(resolved_paths))
    if not paths:
        raise ValueError("No PDF paths provided.")

    p = Path(persist_dir).resolve()
    _ensure_persist_parent_writable(p)
    if p.exists():
        ensure_chroma_tree_writable(p)

    all_splits: list[Document] = []
    for pdf_path in paths:
        all_splits.extend(_load_split_one_pdf(pdf_path, chunk_size, chunk_overlap))
    if not all_splits:
        raise ValueError("No extractable text was found in the supplied PDF(s).")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    collection = CHROMA_LANGCHAIN_COLLECTION

    files_replaced = 0
    if chroma_persist_populated(str(p)):
        store = Chroma(
            persist_directory=str(p),
            embedding_function=embeddings,
            collection_name=collection,
        )
        old_ids: list[str] = []
        for source in paths:
            existing = store.get(where={"source": source}, include=[])
            ids = existing.get("ids", []) if isinstance(existing, dict) else []
            if ids:
                old_ids.extend(str(item) for item in ids)
                files_replaced += 1
        new_ids = [str(uuid4()) for _ in all_splits]
        try:
            store.add_documents(all_splits, ids=new_ids)
        except Exception:
            # Best-effort cleanup for an implementation that partially wrote before raising.
            # These IDs are unique to this attempt, so existing chunks remain untouched.
            try:
                store.delete(ids=new_ids)
            except Exception:
                pass
            raise
        if old_ids:
            store.delete(ids=old_ids)
        rebuild_sparse_index_from_vectorstore(store, str(p))
    else:
        if p.exists():
            _rmtree_chroma(p)
        store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=str(p),
            collection_name=collection,
        )
        rebuild_sparse_index_from_vectorstore(store, str(p))

    return {
        "chunks_added": len(all_splits),
        "files_processed": len(paths),
        "files_replaced": files_replaced,
        "files_new": len(paths) - files_replaced,
    }


def ingest_pdf(
    pdf_path: str,
    persist_dir: str = "chroma_db",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Replace any existing index and build a new one from a single PDF. Returns chunk count."""
    reset_knowledge_base(persist_dir)
    stats = append_pdfs(
        [pdf_path],
        persist_dir=persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return int(stats["chunks_added"])
