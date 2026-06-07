"""Streamlit UI for paper-chat.

Run with:  streamlit run app.py
Needs GROQ_API_KEY in the environment (or a .env file).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env BEFORE importing paper_chat modules: config reads env vars at import
# time, so the .env must be in os.environ first or PC_LLM_PROVIDER is missed.
load_dotenv()

from paper_chat import config, ingest  # noqa: E402
from paper_chat.chat import answer, make_llm  # noqa: E402
from paper_chat.store import SentenceTransformerEmbedder, VectorStore  # noqa: E402

st.set_page_config(page_title="paper-chat", page_icon="📄")
st.title("📄 paper-chat")
st.caption("Ask questions across your papers. Every answer cites its sources.")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(config.EMBED_MODEL)


if "store" not in st.session_state:
    st.session_state.store = VectorStore()
store: VectorStore = st.session_state.store

with st.sidebar:
    st.header("Library")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    arxiv_in = st.text_input("…or paste an arXiv id / link")
    if st.button("Ingest", use_container_width=True):
        embedder = get_embedder()
        added = 0
        with tempfile.TemporaryDirectory() as tmp:
            for up in uploads or []:
                try:
                    p = Path(tmp) / up.name
                    p.write_bytes(up.getvalue())
                    chunks = ingest.load_pdf(p, source=Path(up.name).stem)
                    store.add(chunks, embedder)
                    added += len(chunks)
                    if chunks:
                        st.write(f"• {up.name}: {len(chunks)} chunks")
                    else:
                        st.warning(f"• {up.name}: 0 chunks (no extractable text — scanned PDF?)")
                except Exception as e:  # noqa: BLE001 - surface per-file, don't abort batch
                    st.error(f"• {up.name}: failed — {e}")
            aid = ingest.parse_arxiv_id(arxiv_in or "")
            if aid:
                try:
                    pdf = ingest.fetch_arxiv(aid, Path(tmp))
                    chunks = ingest.load_pdf(pdf, source=f"arXiv:{aid}")
                    store.add(chunks, embedder)
                    added += len(chunks)
                    st.write(f"• arXiv:{aid}: {len(chunks)} chunks")
                except Exception as e:  # noqa: BLE001
                    st.error(f"• arXiv:{aid}: failed — {e}")
        st.success(f"Indexed {added} chunks. Library now holds {len(store)}.")
    st.metric("Chunks indexed", len(store))

question = st.chat_input("Ask about your papers…")
if question:
    if len(store) == 0:
        st.warning("Add at least one PDF first.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            result = answer(question, store, get_embedder(), make_llm())
            st.write(result.text)
            if result.sources:
                with st.expander(f"Sources ({len(result.cited)} cited)"):
                    for s in result.sources:
                        mark = "✅" if s.split("]")[0] + "]" in result.cited else "·"
                        st.write(f"{mark} {s}")
