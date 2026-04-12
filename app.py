"""Streamlit UI: index one PDF, chat with RAG.

Heavy deps (LangChain, OpenAI client) load only when you index, open an existing
index, or send a chat message — not on every sidebar widget change.
"""

import warn_filters  # noqa: F401

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from embed_config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

load_dotenv()

st.set_page_config(page_title="Book Coach RAG", page_icon="📚")
st.title("Book Coach")
st.caption("Index a PDF, then chat as if the author is coaching you.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("Set `OPENAI_API_KEY` in a `.env` file in this folder (see `.env.example`).")
    st.stop()

persist_dir = "chroma_db"


def _chroma_populated(persist: str) -> bool:
    p = Path(persist)
    if not p.is_dir():
        return False
    try:
        return any(p.iterdir())
    except OSError:
        return False


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "_vectorstore_resolved" not in st.session_state:
    st.session_state._vectorstore_resolved = False


def get_vectorstore():
    """Load Chroma only once per session when an on-disk index exists."""
    if st.session_state.vectorstore is not None:
        return st.session_state.vectorstore
    if st.session_state._vectorstore_resolved:
        return None
    st.session_state._vectorstore_resolved = True
    if not _chroma_populated(persist_dir):
        return None
    from vectorstore_loader import load_vectorstore

    st.session_state.vectorstore = load_vectorstore(persist_dir)
    return st.session_state.vectorstore


with st.sidebar:
    st.subheader("Knowledge base")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    path_input = st.text_input("Or path to PDF on disk", placeholder="/path/to/book.pdf")

    st.caption("Chunking applies on **Index PDF** only.")
    chunk_size = st.number_input(
        "Chunk size (chars)",
        min_value=200,
        max_value=4000,
        value=DEFAULT_CHUNK_SIZE,
        step=100,
    )
    chunk_overlap = st.number_input(
        "Chunk overlap (chars)",
        min_value=0,
        max_value=800,
        value=DEFAULT_CHUNK_OVERLAP,
        step=50,
    )

    if st.button("Index PDF", type="primary"):
        from ingest import ingest_pdf
        from vectorstore_loader import load_vectorstore

        pdf_path: str | None = None
        if uploaded is not None:
            tmp = Path(".uploaded.pdf")
            tmp.write_bytes(uploaded.getvalue())
            pdf_path = str(tmp.resolve())
        elif path_input.strip():
            p = Path(path_input.strip()).expanduser()
            if p.is_file():
                pdf_path = str(p)
            else:
                st.error("File not found.")
        else:
            st.warning("Upload a file or enter a valid path.")

        if pdf_path:
            with st.spinner("Indexing…"):
                try:
                    n = ingest_pdf(
                        pdf_path,
                        persist_dir=persist_dir,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                    st.success(f"Indexed {n} chunks.")
                    st.session_state.vectorstore = load_vectorstore(persist_dir)
                    st.session_state._vectorstore_resolved = True
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    if Path(".uploaded.pdf").exists() and st.button("Clear uploaded temp file"):
        Path(".uploaded.pdf").unlink(missing_ok=True)
        st.rerun()

    st.divider()
    st.subheader("Retrieval tuning")
    retrieval_k = st.number_input(
        "Top-k chunks",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="How many chunks to concatenate into context (no re-index needed).",
    )
    use_query_rewrite = st.checkbox(
        "Conversation-aware retrieval",
        value=True,
        help="When there is prior chat, use a short LLM step to rewrite the **search query** "
        "(not your spoken question) so follow-ups retrieve better chunks.",
    )

    st.subheader("Match quality guardrail")
    guardrail_on = st.checkbox(
        "Enable distance guardrail",
        value=False,
        help="If the #1 hit’s distance is above the threshold, skip the LLM and show a warning. Tune using the retrieval trace on real questions.",
    )
    guardrail_max = st.number_input(
        "Max distance (rank #1)",
        min_value=0.0,
        max_value=10.0,
        value=1.2,
        step=0.05,
        disabled=not guardrail_on,
        help="Chroma/L2: lower = closer. Typical good hits are often well below 1–2; calibrate on your PDF.",
    )

    st.divider()
    st.caption("Models: `text-embedding-3-small` + `gpt-4o-mini`")
    show_trace = st.checkbox(
        "Show retrieval trace",
        value=True,
        help="Ranked chunks and scores used as context for each reply.",
    )
    st.session_state.show_retrieval_trace = show_trace

if "messages" not in st.session_state:
    st.session_state.messages = []

vs = get_vectorstore()
if vs is None:
    st.info("Index a PDF in the sidebar to start chatting.")
else:
    show_trace = st.session_state.get("show_retrieval_trace", True)

    def render_retrieval_trace(data: list | dict) -> None:
        if isinstance(data, dict):
            chunks = data.get("chunks") or []
            sq = data.get("search_query") or ""
            latest = data.get("latest_user_message") or ""
            rew = data.get("rewrite_applied", False)
        else:
            chunks = data
            sq = ""
            latest = ""
            rew = False

        n = len(chunks)
        if rew and sq:
            st.markdown("**Search query used for embedding** (conversation-aware rewrite)")
            st.code(sq, language=None)
            if latest and latest.strip() != sq.strip():
                st.caption("Latest user message (unchanged for the answer model)")
                st.text(latest)
        elif sq:
            st.markdown("**Search query used for embedding**")
            st.code(sq, language=None)
        st.caption(
            f"Top **{n}** chunks by **embedding similarity** to that search query "
            "(Chroma). Rank **#1** is closest; **distance** is Chroma’s score "
            "(lower usually means more similar for L2 distance)."
        )
        for row in chunks:
            label = f"#{row['rank']} — distance `{row['distance']:.4f}`"
            if row.get("page") is not None:
                label += f" — PDF page ~{int(row['page']) + 1}"
            st.markdown(f"**{label}**")
            st.text(row["preview"])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_trace and msg["role"] == "assistant" and msg.get("retrieval"):
                with st.expander("Retrieval trace (how the answer was grounded)"):
                    if msg.get("guardrail_skip"):
                        st.caption("LLM was **skipped** (guardrail): context was not sent to the model.")
                    render_retrieval_trace(msg["retrieval"])  # dict or legacy list

    if prompt := st.chat_input("Ask about the book or practice an interview answer…"):
        from rag import answer

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            history = st.session_state.messages[:-1]
            gmax = float(guardrail_max) if guardrail_on else None
            try:
                reply, trace, guard_skip = answer(
                    vs,
                    prompt,
                    history,
                    retrieval_k=int(retrieval_k),
                    guardrail_max_distance=gmax,
                    use_query_rewrite=use_query_rewrite,
                )
            except Exception as e:
                reply = f"Error: {e}"
                trace = {
                    "chunks": [],
                    "search_query": "",
                    "latest_user_message": prompt,
                    "rewrite_applied": False,
                }
                guard_skip = False
            if guard_skip:
                st.caption("Guardrail: LLM skipped (weak best match).")
            st.markdown(reply)
            _chunks = trace.get("chunks") if isinstance(trace, dict) else trace
            if show_trace and _chunks:
                with st.expander("Retrieval trace (how the answer was grounded)"):
                    if guard_skip:
                        st.caption("LLM was **skipped** (guardrail): context was not sent to the model.")
                    render_retrieval_trace(trace)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "retrieval": trace,
                "guardrail_skip": guard_skip,
            }
        )
