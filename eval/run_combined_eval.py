#!/usr/bin/env python3
"""Combined evaluation: retrieval metrics + optional LLM judge scores."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
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
from judge import JudgeResult, judge_answer  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from book_coach.rag import CHAT_MODEL, SYSTEM_PROMPT, build_retrieval_query  # noqa: E402


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _format_context_from_ranked(ranked: list[tuple[str, float, dict]]) -> str:
    parts: list[str] = []
    for i, (doc_text, _distance, md) in enumerate(ranked, start=1):
        page = md.get("page")
        src = md.get("source")
        pdf_name = Path(str(src)).name if src else f"document-{i}.pdf"
        if page is not None:
            header = f"[PDF: {pdf_name} — ~page {int(page) + 1}]"
        else:
            header = f"[PDF: {pdf_name}]"
        parts.append(f"{header}\n{doc_text}")
    return "\n\n---\n\n".join(parts)


def _answer_like_app(
    *,
    latest_user_message: str,
    chat_history: list[dict],
    context: str,
    answer_model: str,
) -> str:
    llm = ChatOpenAI(model=answer_model, temperature=0.3)
    messages: list[dict] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{context}"}
    ]
    for m in chat_history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": latest_user_message})
    r = llm.invoke(messages)
    return str(getattr(r, "content", "")).strip()


def _parse_eval_row(row: dict) -> tuple[str, str, list[dict], list[int]]:
    rid = str(row.get("id", "?"))
    gold_pages = row.get("gold_pages")
    if not isinstance(gold_pages, list) or not gold_pages:
        raise SystemExit(f"Bad row {rid!r}: need non-empty gold_pages")
    if "messages" in row:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"Bad row {rid!r}: messages must be a non-empty array")
        if messages[-1].get("role") != "user":
            raise SystemExit(f"Bad row {rid!r}: last message must have role=user")
        latest = str(messages[-1].get("content", "")).strip()
        history = list(messages[:-1])
        return rid, latest, history, [int(x) for x in gold_pages]
    question = row.get("question")
    if not question:
        raise SystemExit(f"Bad row {rid!r}: need either question or messages")
    return rid, str(question).strip(), [], [int(x) for x in gold_pages]


def _avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined RAG eval: retrieval metrics + LLM-as-a-judge."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=ROOT / "eval" / "conversation_eval.json",
        help="JSON array rows with either {question,...} or {messages,...}",
    )
    parser.add_argument("--persist", default="chroma_db")
    parser.add_argument("-k", type=int, default=5, help="Top-k retrieval chunks.")
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable conversation-aware retrieval rewrite.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Model used by LLM-as-a-judge.",
    )
    parser.add_argument(
        "--answer-model",
        default=CHAT_MODEL,
        help="Model used to generate candidate answers (app-like behavior).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Retrieval-only dry run (skip answer generation and judging).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limit rows processed (0 = all).",
    )
    args = parser.parse_args()

    persist = Path(args.persist)
    if not persist.is_absolute():
        persist = ROOT / persist
    if not persist.exists():
        raise SystemExit(f"No index at {persist}; build/index PDFs first.")

    data = json.loads(args.file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("Eval file must be a non-empty JSON array.")
    if args.max_rows and args.max_rows > 0:
        data = data[: args.max_rows]

    k = max(1, min(args.k, 50))
    use_rewrite = not args.no_rewrite

    row_results: list[dict] = []
    hits: list[bool] = []
    mrr_terms: list[float] = []
    first_ranks: list[int] = []
    judge_rows: list[JudgeResult] = []
    judge_rows_hit: list[JudgeResult] = []
    judge_rows_miss: list[JudgeResult] = []

    print("=== Combined Eval (retrieval + judge) ===")
    print(
        f"file={args.file} k={k} rewrite={'on' if use_rewrite else 'off'} "
        f"judge={'off' if args.skip_judge else args.judge_model}"
    )
    print()

    for row in data:
        if not isinstance(row, dict):
            raise SystemExit("Each eval row must be a JSON object.")
        rid, latest, history, gold_pages = _parse_eval_row(row)
        retrieval_query, rewrite_applied = build_retrieval_query(
            history,
            latest,
            use_query_rewrite=use_rewrite,
        )
        ranked = query_chroma(persist, retrieval_query, k=k)
        gold = human_pages_to_meta(gold_pages)
        first_rank = first_gold_rank(ranked, gold)
        hit = first_rank is not None
        hits.append(hit)
        mrr_terms.append(1.0 / first_rank if first_rank is not None else 0.0)
        if first_rank is not None:
            first_ranks.append(first_rank)

        context = _format_context_from_ranked(ranked)
        answer_text = ""
        judge_dict: dict | None = None
        if not args.skip_judge:
            answer_text = _answer_like_app(
                latest_user_message=latest,
                chat_history=history,
                context=context,
                answer_model=args.answer_model,
            )
            judged = judge_answer(
                question=latest,
                context=context,
                answer=answer_text,
                hit=hit,
                first_gold_rank=first_rank,
                gold_pages=gold_pages,
                model=args.judge_model,
            )
            judge_rows.append(judged)
            if hit:
                judge_rows_hit.append(judged)
            else:
                judge_rows_miss.append(judged)
            judge_dict = judged.as_dict()

        row_result = {
            "id": rid,
            "latest_user_message": latest,
            "query_type": "rewrite" if rewrite_applied else "baseline",
            "retrieval_query": retrieval_query,
            "gold_pages": gold_pages,
            "hit": hit,
            "first_gold_rank": first_rank,
            "retrieved_pages_human": [
                (int(md.get("page")) + 1) if md.get("page") is not None else None
                for _txt, _d, md in ranked
            ],
            "retrieved_top_distances": [float(dist) for _txt, dist, _md in ranked],
            "answer": answer_text,
            "judge": judge_dict,
        }
        row_results.append(row_result)

        status = "HIT" if hit else "MISS"
        rank_s = str(first_rank) if first_rank is not None else "-"
        line = (
            f"[{status}] id={rid} first_gold_rank={rank_s} "
            f"query_type={row_result['query_type']} "
            f"user='{_truncate(latest, 70)}'"
        )
        if judge_dict:
            line += (
                f" judge_overall={judge_dict['overall']}"
                f" judge_pass={judge_dict['pass']}"
            )
        print(line)
    print()

    n = len(row_results)
    recall = sum(hits) / n
    mrr = _avg(mrr_terms)
    rank_counts = dict(sorted(Counter(first_ranks).items()))

    print(f"--- Retrieval summary (n={n}, k={k}) ---")
    print(f"Recall@{k}: {recall:.2%} ({sum(hits)}/{n})")
    print(f"MRR: {mrr:.3f}")
    print(f"HIT/MISS: {sum(hits)}/{n - sum(hits)}")
    print(f"first_gold_rank distribution: {rank_counts or '{}'}")

    if not args.skip_judge:
        print("\n--- Judge summary ---")
        overall = [float(x.overall) for x in judge_rows]
        groundedness = [float(x.groundedness) for x in judge_rows]
        correctness = [float(x.correctness) for x in judge_rows]
        citation = [float(x.citation_faithfulness) for x in judge_rows]
        pass_rate = sum(1 for x in judge_rows if x.passed) / len(judge_rows)
        print(f"Avg overall: {_avg(overall):.3f}")
        print(f"Avg groundedness: {_avg(groundedness):.3f}")
        print(f"Avg correctness: {_avg(correctness):.3f}")
        print(f"Avg citation_faithfulness: {_avg(citation):.3f}")
        print(f"Pass rate: {pass_rate:.2%}")

        if judge_rows_hit:
            print(
                f"Avg overall on retrieval HIT rows: "
                f"{_avg([float(x.overall) for x in judge_rows_hit]):.3f}"
            )
        if judge_rows_miss:
            print(
                f"Avg overall on retrieval MISS rows: "
                f"{_avg([float(x.overall) for x in judge_rows_miss]):.3f}"
            )

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"combined_{ts}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in row_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved row results: {out_path}")


if __name__ == "__main__":
    main()

