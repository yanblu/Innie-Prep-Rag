#!/usr/bin/env python3
"""
Multi-turn retrieval eval: compare baseline (last user message only) vs conversation-aware
rewrite (same logic as rag.answer).

Each JSON row:
  "id": string
  "gold_pages": human 1-based PDF pages (same as questions.json)
  "messages": list of {role: user|assistant, content: str}, must end with "user"
    (that final user line is scored; prior messages are chat history)

Usage (venv active, from project root):
  python eval/run_conversation_eval.py
  python eval/run_conversation_eval.py -k 8 --no-rewrite   # baseline only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Install python-dotenv and use .venv/bin/python.", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
for p in (EVAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

load_dotenv(ROOT / ".env")

from chroma_retrieval import first_gold_rank, human_pages_to_meta, query_chroma  # noqa: E402
from rag import build_retrieval_query  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=ROOT / "eval" / "conversation_eval.json",
    )
    parser.add_argument("--persist", default="chroma_db")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Skip rewrite path; only print baseline scores.",
    )
    args = parser.parse_args()

    persist = Path(args.persist)
    if not persist.is_absolute():
        persist = ROOT / persist
    if not persist.exists():
        raise SystemExit(f"No index at {persist}")

    data = json.loads(args.file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("conversation_eval.json must be a non-empty array.")

    k = max(1, min(args.k, 50))

    base_hits: list[bool] = []
    base_mrr: list[float] = []
    rew_hits: list[bool] = []
    rew_mrr: list[float] = []

    print("=== Multi-turn retrieval eval ===\n")
    for row in data:
        rid = row.get("id", "?")
        msgs = row.get("messages")
        gh = row.get("gold_pages")
        if not msgs or not isinstance(msgs, list) or not gh:
            raise SystemExit(f"Bad row {rid!r}: need messages + gold_pages")
        if msgs[-1].get("role") != "user":
            raise SystemExit(f"{rid}: last message must be role=user")
        latest = str(msgs[-1].get("content", "")).strip()
        history = msgs[:-1]
        gold = human_pages_to_meta(gh)

        ranked_base = query_chroma(persist, latest, k=k)
        rb = first_gold_rank(ranked_base, gold)
        base_hits.append(rb is not None)
        base_mrr.append(1.0 / rb if rb is not None else 0.0)

        print(f"id={rid}")
        print(f"  latest user: {latest[:80]}{'…' if len(latest) > 80 else ''}")

        if not args.no_rewrite:
            rq, used = build_retrieval_query(history, latest, use_query_rewrite=True)
            ranked_rew = query_chroma(persist, rq, k=k)
            rr = first_gold_rank(ranked_rew, gold)
            rew_hits.append(rr is not None)
            rew_mrr.append(1.0 / rr if rr is not None else 0.0)
            print(f"  baseline: rank={rb or '-'}  |  rewrite: rank={rr or '-'}")
            print(f"  rewrite query: {rq[:120]}{'…' if len(rq) > 120 else ''}")
        else:
            print(f"  baseline: rank={rb or '-'}")

        print()

    n = len(data)
    print(f"--- Summary (n={n}, k={k}) ---")
    print(
        f"Baseline Recall@{k}: {sum(base_hits) / n:.2%}  MRR: {sum(base_mrr) / n:.3f}"
    )
    if not args.no_rewrite:
        print(
            f"Rewrite  Recall@{k}: {sum(rew_hits) / n:.2%}  MRR: {sum(rew_mrr) / n:.3f}"
        )


if __name__ == "__main__":
    main()
