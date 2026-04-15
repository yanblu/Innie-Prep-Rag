#!/usr/bin/env python3
"""
Retrieval eval: recall@k using human PDF page numbers (1-based in questions.json).

Uses Chroma's Python client + OpenAIEmbeddingFunction only (no LangChain import chain),
so startup is much faster than importing book_coach.rag / ChatOpenAI.

Schema for each item:
  "id": optional string
  "question": string (same wording you want to test at query time)
  "gold_pages": list of integers — page numbers as shown in your PDF viewer (1 = first page)

Chunk metadata from PyPDFLoader uses 0-based page index; this script converts: meta_page = human_page - 1.

Usage (from project root, OPENAI_API_KEY in .env, index built):
  source .venv/bin/activate && python eval/run_eval.py
  .venv/bin/python eval/run_eval.py
  .venv/bin/python eval/run_eval.py -k 8 --persist chroma_db

Python: 3.10+ including 3.14.x (see ../pyproject.toml). Use `.venv/bin/python` if deps
are not installed globally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print(
        "Missing `python-dotenv`. Use the project virtualenv (not system python3):\n"
        "  cd path/to/rag\\ test && source .venv/bin/activate && python eval/run_eval.py\n"
        "or:\n"
        "  .venv/bin/python eval/run_eval.py",
        file=sys.stderr,
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
for p in (EVAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

load_dotenv(ROOT / ".env")

from chroma_retrieval import first_gold_rank, human_pages_to_meta, query_ranked  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval recall@k vs gold PDF pages.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "eval" / "questions.json",
        help="JSON array of {id?, question, gold_pages}",
    )
    parser.add_argument(
        "--persist",
        default="chroma_db",
        help="Chroma persist directory (relative to project root or absolute).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Top-k chunks per question (same idea as app sidebar).",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "hybrid"],
        default="dense",
        help="Retrieval mode used for eval queries.",
    )
    args = parser.parse_args()

    persist = Path(args.persist)
    if not persist.is_absolute():
        persist = ROOT / persist

    if not persist.exists():
        raise SystemExit(f"No index at {persist}; index your PDF in the app first.")

    raw = args.questions.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit("questions.json must be a JSON array.")

    if not data:
        print("eval/questions.json is empty. Add objects like:")
        print(
            json.dumps(
                [
                    {
                        "id": "q1",
                        "question": "What is RICE prioritization?",
                        "gold_pages": [42],
                    }
                ],
                indent=2,
            )
        )
        raise SystemExit(0)

    k = max(1, min(args.k, 50))
    hits: list[bool] = []
    mrr_terms: list[float] = []

    for row in data:
        if not isinstance(row, dict):
            raise SystemExit("Each question must be a JSON object.")
        q = row.get("question")
        gh = row.get("gold_pages")
        if not q or not isinstance(gh, list) or not gh:
            raise SystemExit(f"Invalid row (need question + gold_pages): {row!r}")
        gold = human_pages_to_meta(gh)

        ranked = query_ranked(persist, str(q), k=k, retrieval_mode=args.retrieval_mode)
        first_rank = first_gold_rank(ranked, gold)

        hit = first_rank is not None
        hits.append(hit)
        mrr_terms.append(1.0 / first_rank if first_rank is not None else 0.0)

        rid = row.get("id", "?")
        status = "HIT" if hit else "MISS"
        rank_s = str(first_rank) if first_rank is not None else "-"
        q_short = (q[:70] + "…") if len(q) > 70 else q
        print(
            f"[{status}] id={rid} first_gold_rank={rank_s} "
            f"mode={args.retrieval_mode}  {q_short}"
        )

        if not hit and ranked:
            tops = []
            for _text, dist, md in ranked[: min(3, len(ranked))]:
                p = md.get("page")
                if p is None:
                    tops.append("?")
                else:
                    label = "rrf" if args.retrieval_mode == "hybrid" else "d"
                    tops.append(f"p~{int(p) + 1}({label}={float(dist):.3f})")
            print(f"        top chunks (human page ~): {', '.join(tops)}")

    n = len(hits)
    recall = sum(hits) / n
    print(f"\nRecall@{k}: {recall:.2%} ({sum(hits)}/{n})")
    mrr = sum(mrr_terms) / n
    print(f"MRR (mean reciprocal rank of first gold chunk in top-{k}; 0 if none): {mrr:.3f}")


if __name__ == "__main__":
    main()
