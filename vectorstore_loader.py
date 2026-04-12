"""Open Chroma index created by LangChain ingest."""

import warn_filters  # noqa: F401

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from embed_config import EMBEDDING_MODEL

load_dotenv()


def load_vectorstore(persist_dir: str = "chroma_db") -> Chroma | None:
    if not Path(persist_dir).exists():
        return None
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
