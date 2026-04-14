"""Unit tests for append vs reset ingest (Chroma mocked)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestIngestModes(unittest.TestCase):
    def test_chroma_persist_populated_empty(self) -> None:
        from book_coach.ingest import chroma_persist_populated

        self.assertFalse(chroma_persist_populated("__does_not_exist_ingest_test__"))

    @patch("book_coach.ingest._ensure_persist_parent_writable")
    @patch("book_coach.ingest.shutil.rmtree")
    def test_reset_knowledge_base_removes_dir(
        self,
        mock_rmtree: MagicMock,
        _mock_ensure: MagicMock,
    ) -> None:
        from book_coach.ingest import reset_knowledge_base

        with patch("book_coach.ingest.Path.exists", return_value=True):
            reset_knowledge_base(persist_dir="__fake_chroma__")
        mock_rmtree.assert_called_once()

    @patch("book_coach.ingest._ensure_persist_parent_writable")
    @patch("book_coach.ingest.Chroma")
    @patch("book_coach.ingest.OpenAIEmbeddings")
    @patch("book_coach.ingest.chroma_persist_populated")
    @patch("book_coach.ingest._load_split_one_pdf")
    @patch("book_coach.ingest.rebuild_sparse_index_from_vectorstore")
    def test_append_creates_fresh_index_when_empty(
        self,
        _mock_sparse: MagicMock,
        mock_load: MagicMock,
        mock_populated: MagicMock,
        _mock_emb: MagicMock,
        mock_chroma: MagicMock,
        _mock_ensure: MagicMock,
    ) -> None:
        from book_coach.ingest import append_pdfs

        mock_populated.return_value = False
        mock_load.return_value = [
            Document(page_content="a", metadata={"source": "/x/1.pdf", "page": 0}),
        ]
        with patch("book_coach.ingest.Path.exists", return_value=False):
            stats = append_pdfs(["/fake/one.pdf"], persist_dir="chroma_test_fresh")
        self.assertEqual(stats["chunks_added"], 1)
        self.assertEqual(stats["files_processed"], 1)
        self.assertEqual(stats["files_replaced"], 0)
        self.assertEqual(stats["files_new"], 1)
        mock_chroma.from_documents.assert_called_once()
        mock_chroma.return_value.add_documents.assert_not_called()

    @patch("book_coach.ingest._ensure_persist_parent_writable")
    @patch("book_coach.ingest.Chroma")
    @patch("book_coach.ingest.OpenAIEmbeddings")
    @patch("book_coach.ingest.chroma_persist_populated")
    @patch("book_coach.ingest._load_split_one_pdf")
    @patch("book_coach.ingest.rebuild_sparse_index_from_vectorstore")
    def test_append_adds_to_existing_index(
        self,
        _mock_sparse: MagicMock,
        mock_load: MagicMock,
        mock_populated: MagicMock,
        _mock_emb: MagicMock,
        mock_chroma: MagicMock,
        _mock_ensure: MagicMock,
    ) -> None:
        from book_coach.ingest import append_pdfs

        mock_populated.return_value = True
        mock_load.return_value = [
            Document(page_content="b", metadata={"source": "/x/2.pdf", "page": 0}),
        ]
        store = MagicMock()
        mock_chroma.return_value = store
        store.get.return_value = {"ids": []}
        stats = append_pdfs(["/fake/two.pdf"], persist_dir="chroma_test_append")
        self.assertEqual(stats["chunks_added"], 1)
        self.assertEqual(stats["files_processed"], 1)
        self.assertEqual(stats["files_replaced"], 0)
        self.assertEqual(stats["files_new"], 1)
        mock_chroma.from_documents.assert_not_called()
        store.add_documents.assert_called_once()
        args, _kwargs = store.add_documents.call_args
        self.assertEqual(len(args[0]), 1)

    @patch("book_coach.ingest._ensure_persist_parent_writable")
    @patch("book_coach.ingest.Chroma")
    @patch("book_coach.ingest.OpenAIEmbeddings")
    @patch("book_coach.ingest.chroma_persist_populated")
    @patch("book_coach.ingest._load_split_one_pdf")
    @patch("book_coach.ingest.rebuild_sparse_index_from_vectorstore")
    def test_append_replaces_existing_chunks_for_same_source(
        self,
        _mock_sparse: MagicMock,
        mock_load: MagicMock,
        mock_populated: MagicMock,
        _mock_emb: MagicMock,
        mock_chroma: MagicMock,
        _mock_ensure: MagicMock,
    ) -> None:
        from book_coach.ingest import append_pdfs

        mock_populated.return_value = True
        mock_load.return_value = [
            Document(page_content="b", metadata={"source": "/x/2.pdf", "page": 0}),
        ]
        store = MagicMock()
        store.get.return_value = {"ids": ["id-1", "id-2"]}
        mock_chroma.return_value = store

        stats = append_pdfs(["/fake/two.pdf"], persist_dir="chroma_test_append")
        self.assertEqual(stats["chunks_added"], 1)
        self.assertEqual(stats["files_processed"], 1)
        self.assertEqual(stats["files_replaced"], 1)
        self.assertEqual(stats["files_new"], 0)
        store.get.assert_called_once()
        store.delete.assert_called_once_with(ids=["id-1", "id-2"])
        store.add_documents.assert_called_once()

    @patch("book_coach.ingest._ensure_persist_parent_writable")
    @patch("book_coach.ingest.Chroma")
    @patch("book_coach.ingest.OpenAIEmbeddings")
    @patch("book_coach.ingest.chroma_persist_populated")
    @patch("book_coach.ingest._load_split_one_pdf")
    @patch("book_coach.ingest.rebuild_sparse_index_from_vectorstore")
    def test_append_dedupes_duplicate_input_paths(
        self,
        _mock_sparse: MagicMock,
        mock_load: MagicMock,
        mock_populated: MagicMock,
        _mock_emb: MagicMock,
        _mock_chroma: MagicMock,
        _mock_ensure: MagicMock,
    ) -> None:
        from book_coach.ingest import append_pdfs

        mock_populated.return_value = False
        mock_load.return_value = [
            Document(page_content="a", metadata={"source": "/x/1.pdf", "page": 0}),
        ]
        with patch("book_coach.ingest.Path.exists", return_value=False):
            stats = append_pdfs(
                ["/fake/one.pdf", "/fake/one.pdf"],
                persist_dir="chroma_test_fresh",
            )

        self.assertEqual(stats["chunks_added"], 1)
        self.assertEqual(stats["files_processed"], 1)
        self.assertEqual(stats["files_replaced"], 0)
        self.assertEqual(stats["files_new"], 1)
        mock_load.assert_called_once()

    @patch("book_coach.ingest.append_pdfs")
    @patch("book_coach.ingest.reset_knowledge_base")
    def test_ingest_pdf_replaces_then_appends(
        self,
        mock_reset: MagicMock,
        mock_append: MagicMock,
    ) -> None:
        from book_coach.ingest import ingest_pdf

        mock_append.return_value = {
            "chunks_added": 42,
            "files_processed": 1,
            "files_replaced": 0,
            "files_new": 1,
        }
        n = ingest_pdf("/fake/book.pdf", persist_dir="/tmp/x")
        mock_reset.assert_called_once_with("/tmp/x")
        mock_append.assert_called_once()
        self.assertEqual(n, 42)


class TestLoadSplitMetadata(unittest.TestCase):
    def test_source_metadata_set_on_splits(self) -> None:
        """End-to-end load + split; minimal one-page PDF with extractable text."""
        from io import BytesIO

        from book_coach.ingest import _load_split_one_pdf
        from pypdf import PdfReader, PdfWriter

        _MIN_PDF_ONE_PAGE = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length 51>>stream\n"
            b"BT /F1 24 Tf 100 700 Td (Hello ingest test) Tj ET\n"
            b"endstream\nendobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseType/Helvetica>>endobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000266 00000 n \n"
            b"0000000367 00000 n \n"
            b"trailer<</Root 1 0 R/Size 6>>\n"
            b"startxref\n439\n%%EOF"
        )

        fd, raw = tempfile.mkstemp(suffix=".pdf", dir=Path(__file__).resolve().parent)
        os.close(fd)
        path = Path(raw)
        try:
            w = PdfWriter()
            w.add_page(PdfReader(BytesIO(_MIN_PDF_ONE_PAGE)).pages[0])
            with open(path, "wb") as f:
                w.write(f)

            splits = _load_split_one_pdf(str(path), chunk_size=80, chunk_overlap=10)
            self.assertGreaterEqual(len(splits), 1)
            want = str(path.resolve())
            for s in splits:
                self.assertEqual(s.metadata.get("source"), want)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
