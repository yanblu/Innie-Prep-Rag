"""LLM-as-a-judge helpers for combined evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_openai import ChatOpenAI


JUDGE_SYSTEM = """You are a strict evaluator for a retrieval-augmented assistant.

Score the candidate answer using this rubric (1-5 integers only):
- groundedness: How well the answer is supported by the provided retrieved context.
- correctness: How well the answer addresses the user question accurately.
- citation_faithfulness: Whether citations/source mentions align with provided context.
- overall: Holistic quality for this task.

Rules:
- Be conservative: if the context does not support a claim, lower groundedness/citation scores.
- Do not reward eloquence if unsupported by context.
- Keep reason to one short sentence.
- Output valid JSON only, matching the required schema exactly.
"""

JUDGE_USER = """Question:
{question}

Retrieved context:
{context}

Candidate answer:
{answer}

Retrieval signals:
- hit: {hit}
- first_gold_rank: {first_gold_rank}
- gold_pages (human numbering): {gold_pages}

Return JSON with keys:
groundedness, correctness, citation_faithfulness, overall, reason
"""

# Deterministic production pass rule (not model-defined):
# - overall >= 4
# - groundedness >= 4
# - citation_faithfulness >= 3
PASS_OVERALL_MIN = 4
PASS_GROUNDEDNESS_MIN = 4
PASS_CITATION_MIN = 3


@dataclass
class JudgeResult:
    groundedness: int
    correctness: int
    citation_faithfulness: int
    overall: int
    passed: bool
    reason: str
    model_pass: bool | None = None

    def as_dict(self) -> dict:
        return {
            "groundedness": self.groundedness,
            "correctness": self.correctness,
            "citation_faithfulness": self.citation_faithfulness,
            "overall": self.overall,
            "pass": self.passed,
            "reason": self.reason,
            "model_pass": self.model_pass,
        }


def _score_to_int(raw: object, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Judge field {name!r} must be an integer 1-5.") from e
    if value < 1 or value > 5:
        raise ValueError(f"Judge field {name!r} must be in range 1-5, got {value}.")
    return value


def parse_judge_json(text: str) -> JudgeResult:
    """Parse strict judge JSON and validate required fields."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Tolerate fenced output if model ignores instruction.
        cleaned = text.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        obj = json.loads(cleaned)

    if not isinstance(obj, dict):
        raise ValueError("Judge output must be a JSON object.")

    groundedness = _score_to_int(obj.get("groundedness"), "groundedness")
    correctness = _score_to_int(obj.get("correctness"), "correctness")
    citation_faithfulness = _score_to_int(obj.get("citation_faithfulness"), "citation_faithfulness")
    overall = _score_to_int(obj.get("overall"), "overall")
    reason = str(obj.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided."
    model_pass_raw = obj.get("pass")
    model_pass: bool | None
    if model_pass_raw is None:
        model_pass = None
    elif isinstance(model_pass_raw, bool):
        model_pass = model_pass_raw
    else:
        raise ValueError("Judge field 'pass' must be boolean when provided.")

    passed = (
        overall >= PASS_OVERALL_MIN
        and groundedness >= PASS_GROUNDEDNESS_MIN
        and citation_faithfulness >= PASS_CITATION_MIN
    )

    return JudgeResult(
        groundedness=groundedness,
        correctness=correctness,
        citation_faithfulness=citation_faithfulness,
        overall=overall,
        passed=passed,
        reason=reason,
        model_pass=model_pass,
    )


def judge_answer(
    *,
    question: str,
    context: str,
    answer: str,
    hit: bool,
    first_gold_rank: int | None,
    gold_pages: list[int],
    model: str = "gpt-4o-mini",
) -> JudgeResult:
    """Call an LLM judge and return normalized scores."""
    llm = ChatOpenAI(model=model, temperature=0.0)
    prompt = JUDGE_USER.format(
        question=question.strip(),
        context=context.strip(),
        answer=answer.strip(),
        hit=str(hit),
        first_gold_rank=str(first_gold_rank) if first_gold_rank is not None else "-",
        gold_pages=json.dumps(gold_pages),
    )
    r = llm.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    )
    content = (getattr(r, "content", None) or "").strip()
    return parse_judge_json(content)

