"""Close Chroma clients and fix permissions so SQLite can write (esp. Streamlit + append)."""

from __future__ import annotations

import gc
import os
import stat
from pathlib import Path
from typing import Any


def close_langchain_chroma_client(vectorstore: Any | None) -> None:
    """Call chromadb ``Client.close()`` held by LangChain ``Chroma`` so the DB is not locked.

    Chroma documents that PersistentClient must be closed before another client writes the
    same path; otherwise SQLite may report readonly / locking errors.
    """
    if vectorstore is None:
        return
    client = getattr(vectorstore, "_client", None)
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass
    gc.collect()


def ensure_chroma_tree_writable(root: str | Path) -> None:
    """Best-effort user read+write on the persist tree (helps iCloud / odd umasks)."""
    r = Path(root)
    if not r.is_dir():
        return
    try:
        r.chmod(stat.S_IRWXU)
    except OSError:
        pass
    for dirpath, _, filenames in os.walk(r, topdown=False):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                fp.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
        try:
            os.chmod(dirpath, stat.S_IRWXU)
        except OSError:
            pass
