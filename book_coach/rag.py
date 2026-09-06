"""Load vector store and run retrieval + chat."""

import book_coach.warn_filters  # noqa: F401

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from book_coach.hybrid_retrieval import hybrid_search
from book_coach.vectorstore_loader import load_vectorstore

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
DEFAULT_RETRIEVAL_K = 5

GUARDRAIL_MESSAGE = (
    "I couldn’t find a close enough match in your indexed PDFs for this question—the best "
    "retrieval hit exceeded your **distance threshold**. Try rephrasing, adjusting the "
    "threshold in the sidebar, or re-indexing with different chunk settings. Open the "
    "retrieval trace below to see what was still retrieved."
)

SYSTEM_PROMPT = """You are an expert coach grounded in the book/material provided in CONTEXT below. Speak as the author or a senior industry expert who knows this material deeply. Use CONTEXT for specific frameworks, terminology, and examples from the book.

Rules:
- Prefer answers that clearly tie back to CONTEXT when it is relevant.
- When you cite where something came from, refer to the **PDF file name** and **approximate page** exactly as shown in each CONTEXT block’s header (e.g. “In *Cracking the PM Interview.pdf*, around page 14…” or “(see *guide.pdf*, ~p. 223)”). Do **not** say “chunk” or “chunk 1/2/3” in your reply—users care about the source document, not retrieval slices.
- Use only filenames and page numbers that appear in CONTEXT headers. Do not invent sources.
- If CONTEXT does not contain enough to answer, say so in one short sentence, then you may add brief general interview-prep advice and label it as general advice (not from the book).
- Be concise, practical, and encouraging."""

REWRITE_SYSTEM = """You write a single line search query for a vector database over the user’s indexed PDFs (interview prep, PM, careers).

Output rules:
- Reply with ONLY the search query text: no quotes, labels, or explanation.
- One line; resolve pronouns (it, that, this) using the prior turns.
- Include concrete keywords likely to appear in the relevant passage.
- If the latest user message is already a clear standalone question, keep or tighten it."""

REWRITE_USER = """Prior conversation:
{transcript}

Latest user message (the answer must still address this exactly): {latest}

Standalone search query for the book index:"""


def _history_to_transcript(history: list[dict], max_turns: int = 8, max_chars: int = 1200) -> str:
    lines: list[str] = []
    for m in history[-max_turns:]:
        role = m.get("role", "?")
        text = (m.get("content") or "").strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _rewrite_retrieval_query(
    history: list[dict],
    latest: str,
    llm: ChatOpenAI,
) -> str:
    transcript = _history_to_transcript(history)
    content = REWRITE_USER.format(transcript=transcript, latest=latest.strip())
    r = llm.invoke(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": content},
        ]
    )
    raw = (getattr(r, "content", None) or "").strip()
    line = raw.split("\n", 1)[0].strip().strip('"').strip("'")
    return line if len(line) > 1 else latest.strip()


def build_retrieval_query(
    chat_history: list[dict],
    user_message: str,
    *,
    use_query_rewrite: bool = True,
) -> tuple[str, bool]:
    """Build the string embedded for Chroma search (same rules as ``answer()``)."""
    if use_query_rewrite and chat_history:
        rewrite_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)
        q = _rewrite_retrieval_query(chat_history, user_message, rewrite_llm)
        return q, True
    return user_message.strip(), False


def _format_context_chunks(docs: list) -> str:
    """Build CONTEXT blocks with PDF+page headers so the model cites documents, not chunk ids."""
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata or {}
        page = md.get("page")
        src = md.get("source")
        pdf_name = Path(str(src)).name if src else f"document-{i}.pdf"
        if page is not None:
            header = f"[PDF: {pdf_name} — ~page {int(page) + 1}]"
        else:
            header = f"[PDF: {pdf_name}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer(
    vectorstore: Chroma,
    user_message: str,
    chat_history: list[dict],
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    guardrail_max_distance: float | None = None,
    use_query_rewrite: bool = True,
    retrieval_mode: str = "dense",
) -> tuple[str, dict, bool]:
    """Return (assistant_reply, retrieval_bundle, guardrail_skipped_llm).

    retrieval_bundle has: chunks (list of row dicts), search_query, latest_user_message,
    rewrite_applied (whether an LLM rewrite was used for embedding search).
    """
    k = max(1, min(retrieval_k, 50))
    retrieval_query, rewrite_applied = build_retrieval_query(
        chat_history,
        user_message,
        use_query_rewrite=use_query_rewrite,
    )

    is_hybrid = retrieval_mode == "hybrid"
    if is_hybrid:
        persist_dir = str(getattr(vectorstore, "_persist_directory", "chroma_db"))
        ranked_hybrid = hybrid_search(
            vectorstore,
            retrieval_query,
            k_final=k,
            persist_dir=persist_dir,
        )
        docs = [item.doc for item in ranked_hybrid]
    else:
        ranked = vectorstore.similarity_search_with_score(retrieval_query, k=k)
        docs = [doc for doc, _ in ranked]

    chunks: list[dict] = []
    if is_hybrid:
        for i, item in enumerate(ranked_hybrid, start=1):
            md = item.doc.metadata or {}
            chunks.append(
                {
                    "rank": i,
                    "rrf_score": float(item.rrf_score),
                    "dense_rank": item.dense_rank,
                    "sparse_rank": item.sparse_rank,
                    "dense_distance": item.dense_distance,
                    "sparse_score": item.sparse_score,
                    "preview": item.doc.page_content[:800] + ("…" if len(item.doc.page_content) > 800 else ""),
                    "full_content": item.doc.page_content,
                    "page": md.get("page"),
                    "source": md.get("source"),
                }
            )
    else:
        for i, (doc, distance) in enumerate(ranked, start=1):
            md = doc.metadata or {}
            chunks.append(
                {
                    "rank": i,
                    "distance": float(distance),
                    "preview": doc.page_content[:800] + ("…" if len(doc.page_content) > 800 else ""),
                    "full_content": doc.page_content,
                    "page": md.get("page"),
                    "source": md.get("source"),
                }
            )

    retrieval_bundle: dict = {
        "chunks": chunks,
        "search_query": retrieval_query,
        "latest_user_message": user_message,
        "rewrite_applied": rewrite_applied,
        "retrieval_mode": retrieval_mode,
    }

    if guardrail_max_distance is not None and chunks and not is_hybrid:
        if chunks[0]["distance"] > guardrail_max_distance:
            return GUARDRAIL_MESSAGE, retrieval_bundle, True

    context = _format_context_chunks(docs)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)
    messages: list[dict] = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{context}",
        }
    ]
    for m in chat_history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    response = llm.invoke(messages)
    return response.content, retrieval_bundle, False


__all__ = ["answer", "build_retrieval_query", "DEFAULT_RETRIEVAL_K", "CHAT_MODEL"]
