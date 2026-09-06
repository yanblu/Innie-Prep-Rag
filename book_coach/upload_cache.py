"""Stable local cache paths for PDFs uploaded through the Streamlit UI."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class UploadedFile(Protocol):
    """The small portion of Streamlit's upload object used by the cache."""

    name: str

    def getbuffer(self) -> memoryview: ...


def _cache_name(name: str, content: bytes, duplicate_names: Counter[str]) -> str:
    """Return a stable, safe filename; disambiguate same-name uploads by content."""
    original = Path(name).name or "upload.pdf"
    if duplicate_names[original] == 1:
        return original
    suffix = Path(original).suffix
    stem = Path(original).stem or "upload"
    digest = hashlib.sha256(content).hexdigest()[:12]
    return f"{stem}-{digest}{suffix}"


def cache_uploaded_pdfs(
    uploads: Sequence[UploadedFile],
    upload_dir: Path,
) -> list[str]:
    """Cache uploads under stable names and return their resolved paths.

    A single uploaded filename always maps to the same cache path, regardless of its
    position in a Streamlit multi-upload selection. Re-uploading changed bytes under
    that filename therefore replaces the cached file and triggers normal per-source
    re-indexing. Same-name files in one selection are disambiguated by a content hash.
    """
    materialized = [(upload.name, bytes(upload.getbuffer())) for upload in uploads]
    names = Counter(Path(name).name or "upload.pdf" for name, _content in materialized)
    upload_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for name, content in materialized:
        destination = upload_dir / _cache_name(name, content, names)
        if not destination.exists() or destination.read_bytes() != content:
            destination.write_bytes(content)
        paths.append(str(destination.resolve()))
    return paths
