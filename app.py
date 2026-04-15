"""Streamlit UI: build a shared PDF index (append) or reset it, then chat with RAG.

Heavy deps (LangChain, OpenAI client) load only when you index, open an existing
index, or send a chat message — not on every sidebar widget change.
"""

import book_coach.warn_filters  # noqa: F401

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from book_coach.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from book_coach.ingest import chroma_persist_populated

load_dotenv()

st.set_page_config(page_title="Book Coach RAG", page_icon="📚")
st.title("Book Coach")
st.caption("Add PDFs to a shared knowledge base, then chat as if the author is coaching you.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("Set `OPENAI_API_KEY` in a `.env` file in this folder (see `.env.example`).")
    st.stop()

persist_dir = "chroma_db"
UPLOAD_DIR = Path(".uploaded_pdfs")


def _gather_pdf_paths(
    uploaded_list: list | None,
    path_lines: str,
) -> tuple[list[str], list[str]]:
    """Collect absolute paths from uploads and from one-path-per-line text. Returns (paths, errors)."""
    paths: list[str] = []
    errors: list[str] = []
    if uploaded_list:
        UPLOAD_DIR.mkdir(exist_ok=True)
        for i, uf in enumerate(uploaded_list):
            name = Path(uf.name).name
            dest = UPLOAD_DIR / f"{i}_{name}"
            dest.write_bytes(uf.getbuffer())
            paths.append(str(dest.resolve()))
    for line in path_lines.splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line).expanduser()
        if p.is_file():
            paths.append(str(p.resolve()))
        else:
            errors.append(line)
    return paths, errors


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "_vectorstore_resolved" not in st.session_state:
    st.session_state._vectorstore_resolved = False


def _release_chroma_for_disk_write() -> None:
    """Close the in-memory Chroma client so ingest can open the DB read-write (SQLite)."""
    import gc

    from book_coach.chroma_lifecycle import close_langchain_chroma_client

    close_langchain_chroma_client(st.session_state.get("vectorstore"))
    st.session_state.vectorstore = None
    st.session_state._vectorstore_resolved = False
    gc.collect()


def get_vectorstore():
    """Load Chroma only once per session when an on-disk index exists."""
    if st.session_state.vectorstore is not None:
        return st.session_state.vectorstore
    if st.session_state._vectorstore_resolved:
        return None
    st.session_state._vectorstore_resolved = True
    if not chroma_persist_populated(persist_dir):
        return None
    from book_coach.vectorstore_loader import load_vectorstore

    st.session_state.vectorstore = load_vectorstore(persist_dir)
    return st.session_state.vectorstore


with st.sidebar:
    st.subheader("Knowledge base")
    uploaded = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Each run of “Add to index” appends chunks. Re-adding the same file replaces that file's old chunks.",
    )
    path_input = st.text_area(
        "Or paths on disk (one per line)",
        placeholder="/path/to/first.pdf\n/path/to/second.pdf",
        height=88,
    )

    st.caption("Chunking applies only when you **add PDFs** (not on each chat).")
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

    if st.button("Add PDF(s) to index", type="primary"):
        from book_coach.ingest import append_pdfs
        from book_coach.vectorstore_loader import load_vectorstore

        ul = list(uploaded) if uploaded else []
        pdf_paths, missing = _gather_pdf_paths(ul if ul else None, path_input)

        for line in missing:
            st.error(f"File not found: {line}")

        if not pdf_paths:
            if not missing:
                st.warning("Upload at least one PDF or enter valid path(s).")
        else:
            _release_chroma_for_disk_write()
            with st.spinner("Indexing…"):
                try:
                    stats = append_pdfs(
                        pdf_paths,
                        persist_dir=persist_dir,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                    st.success(
                        "Indexed "
                        f"**{stats['files_processed']}** file(s): "
                        f"**{stats['files_new']}** new, "
                        f"**{stats['files_replaced']}** replaced; "
                        f"added **{stats['chunks_added']}** chunks."
                    )
                    st.session_state.vectorstore = load_vectorstore(persist_dir)
                    st.session_state._vectorstore_resolved = True
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    st.subheader("Reset knowledge base")
    st.caption("Removes **all** indexed PDFs from disk (`chroma_db/`). This cannot be undone.")
    reset_ok = st.checkbox("I understand this deletes the entire index", key="confirm_reset_kb")
    if st.button("Reset knowledge base", type="secondary", disabled=not reset_ok):
        from book_coach.ingest import reset_knowledge_base

        try:
            _release_chroma_for_disk_write()
            reset_knowledge_base(persist_dir)
            st.session_state.vectorstore = None
            st.session_state._vectorstore_resolved = False
            st.success("Knowledge base cleared. Add PDFs again to rebuild.")
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

    if UPLOAD_DIR.is_dir() and any(UPLOAD_DIR.iterdir()) and st.button("Clear cached uploads"):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
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
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=["dense", "hybrid"],
        index=0,
        help="Dense uses embeddings only. Hybrid uses BM25 (keyword) + dense with RRF fusion.",
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
    st.info("Add one or more PDFs in the sidebar to start chatting.")
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
        mode = "dense"
        if isinstance(data, dict):
            mode = str(data.get("retrieval_mode", "dense"))
        if mode == "hybrid":
            st.caption(
                f"Top **{n}** chunks by **hybrid fusion** (dense + BM25 via RRF). "
                "Higher RRF score is better."
            )
        else:
            st.caption(
                f"Top **{n}** chunks by **embedding similarity** to that search query "
                "(Chroma). Rank **#1** is closest; **distance** is Chroma’s score "
                "(lower usually means more similar for L2 distance)."
            )
        for row in chunks:
            if mode == "hybrid":
                label = f"#{row['rank']} — RRF `{float(row.get('rrf_score', 0.0)):.4f}`"
                if row.get("dense_rank") is not None:
                    label += f" — dense_rank `{int(row['dense_rank'])}`"
                if row.get("sparse_rank") is not None:
                    label += f" — sparse_rank `{int(row['sparse_rank'])}`"
            else:
                label = f"#{row['rank']} — distance `{row['distance']:.4f}`"
            src = row.get("source")
            if src:
                label += f" — `{Path(str(src)).name}`"
            if row.get("page") is not None:
                label += f" — page ~{int(row['page']) + 1}"
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
        from book_coach.rag import answer

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
                    retrieval_mode=str(retrieval_mode),
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
