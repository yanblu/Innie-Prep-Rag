"""Tests for stable Streamlit upload cache paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from book_coach.upload_cache import cache_uploaded_pdfs


class FakeUpload:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getbuffer(self) -> memoryview:
        return memoryview(self._content)


class TestUploadCache(unittest.TestCase):
    def test_paths_do_not_depend_on_upload_selection_order(self) -> None:
        first = FakeUpload("guide.pdf", b"first")
        second = FakeUpload("notes.pdf", b"second")
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            original = cache_uploaded_pdfs([first, second], cache_dir)
            reordered = cache_uploaded_pdfs([second, first], cache_dir)

            self.assertEqual(set(original), set(reordered))
            self.assertEqual((cache_dir / "guide.pdf").read_bytes(), b"first")
            self.assertEqual((cache_dir / "notes.pdf").read_bytes(), b"second")

    def test_same_name_uploads_are_disambiguated_by_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = cache_uploaded_pdfs(
                [FakeUpload("guide.pdf", b"first"), FakeUpload("guide.pdf", b"second")],
                Path(tmp),
            )

            self.assertEqual(len(set(paths)), 2)
            self.assertTrue(all(Path(path).name.startswith("guide-") for path in paths))
