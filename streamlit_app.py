import streamlit as st
from pathlib import Path

from pdf_processor import load_and_chunk_pdf
from vector_store import add_chunks
from retriever import retrieve
from generator import generate_answer, NO_ANSWER

UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PDF Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction.
# session_state persists values across reruns.

if "indexed_files" not in st.session_state:
    # Pick up PDFs already uploaded in a previous session
    st.session_state.indexed_files = sorted(f.name for f in UPLOADS_DIR.glob("*.pdf"))

if "last_result" not in st.session_state:
    st.session_state.last_result = None  # {"answer": ..., "chunks": ..., "question": ...}

# ── Sidebar: upload ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 PDF Q&A")
    st.caption("Upload a PDF, ask questions about it.")

    st.divider()
    st.subheader("Upload a PDF")

    uploaded_file = st.file_uploader("Choose a file", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        already_indexed = uploaded_file.name in st.session_state.indexed_files
        btn_label = "Re-index" if already_indexed else "Index PDF"

        if st.button(btn_label, use_container_width=True):
            save_path = UPLOADS_DIR / uploaded_file.name
            save_path.write_bytes(uploaded_file.getbuffer())

            with st.spinner(f"Indexing {uploaded_file.name}…"):
                chunks = load_and_chunk_pdf(str(save_path))
                add_chunks(chunks)

            if uploaded_file.name not in st.session_state.indexed_files:
                st.session_state.indexed_files.append(uploaded_file.name)
                st.session_state.indexed_files.sort()

            st.success(f"Done — {len(chunks)} chunks indexed.")

    st.divider()
    st.subheader("Indexed PDFs")

    if st.session_state.indexed_files:
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")
    else:
        st.caption("No PDFs indexed yet.")

# ── Main area: ask ────────────────────────────────────────────────────────────

st.header("Ask a question")

col_source, col_spacer = st.columns([2, 3])
with col_source:
    source_options = ["All PDFs"] + st.session_state.indexed_files
    source_label = st.selectbox("Search in", source_options, label_visibility="visible")

source = (
    None
    if source_label == "All PDFs"
    else str(UPLOADS_DIR / source_label)
)

question = st.text_input(
    "Question",
    placeholder="e.g. What is the attention mechanism?",
    label_visibility="collapsed",
)

ask_clicked = st.button("Ask", type="primary", use_container_width=False)

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.indexed_files:
        st.warning("Upload and index a PDF first.")
    else:
        with st.spinner("Retrieving and generating…"):
            chunks = retrieve(question, n_results=5, source=source)
            answer = generate_answer(question, chunks)
        st.session_state.last_result = {
            "question": question,
            "answer": answer,
            "chunks": chunks,
        }

# ── Results ───────────────────────────────────────────────────────────────────

result = st.session_state.last_result

if result:
    st.divider()

    # Answer
    st.subheader("Answer")
    if result["answer"] == NO_ANSWER:
        st.warning(result["answer"])
    else:
        st.success(result["answer"])

    # Retrieved chunks — collapsed by default, open them to see why the answer came out as it did
    chunks = result["chunks"]
    if chunks:
        with st.expander(f"Retrieved chunks ({len(chunks)}) — click to see what the LLM read"):
            for i, chunk in enumerate(chunks, 1):
                meta = chunk["metadata"]
                source_name = Path(meta.get("source", "?")).name
                page = meta.get("page", "?")

                if chunk.get("from_bm25") and chunk.get("distance") is None:
                    tag = "🔑 BM25 keyword match"
                    tag_color = "blue"
                elif chunk.get("from_bm25"):
                    tag = f"🔑 BM25 + semantic  dist={chunk.get('distance')}"
                    tag_color = "blue"
                else:
                    tag = f"🔍 Semantic  dist={chunk.get('distance')}"
                    tag_color = "green"

                st.markdown(
                    f"**Chunk {i}** &nbsp;·&nbsp; `{source_name}` &nbsp;·&nbsp; "
                    f"page {page} &nbsp;·&nbsp; :{tag_color}[{tag}]"
                )
                st.text(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))
                if i < len(chunks):
                    st.divider()
    else:
        st.info("No relevant chunks found — the answer is based on the guardrail, not retrieved content.")
