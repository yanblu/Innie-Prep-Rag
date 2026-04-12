"""Load a PDF, chunk, embed with OpenAI, persist to Chroma."""

import warn_filters  # noqa: F401

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embed_config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, EMBEDDING_MODEL

load_dotenv()


def ingest_pdf(
    pdf_path: str,
    persist_dir: str = "chroma_db",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Replace any existing index and build a new one from the PDF. Returns chunk count."""
    p = Path(persist_dir).resolve()
    parent = p.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not os.access(parent, os.W_OK):
        raise PermissionError(
            f"Folder not writable (Chroma/SQLite cannot save the index): {parent}\n"
            "Fix: chmod -R u+w .  |  move project off iCloud Desktop  |  run from a directory you own."
        )
    if p.exists():
        try:
            shutil.rmtree(p)
        except OSError as e:
            raise PermissionError(
                f"Cannot delete old index at {p} (close Streamlit, then: chmod -R u+w {p} or rm -rf chroma_db)."
            ) from e

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(p),
    )
    return len(splits)
