"""Retrieval lab: compare what each search arm returns for one question."""

import time

import streamlit as st

from nexus import ui
from nexus.retrieve import get_retriever

ARMS = [
    ("Vector only", ":material/scatter_plot:", dict(use_vector=True, use_bm25=False, rerank=False)),
    ("Keyword only (BM25)", ":material/text_fields:", dict(use_vector=False, use_bm25=True, rerank=False)),
    ("Hybrid (RRF)", ":material/merge:", dict(use_vector=True, use_bm25=True, rerank=False)),
    ("Hybrid + rerank", ":material/sort:", dict(use_vector=True, use_bm25=True, rerank=True)),
]

st.markdown("#### Retrieval lab")
st.caption("Search only — no model runs here. See what each arm of the search finds for the same "
           "question; this is what `python -m nexus.evaluate` measures over the golden set.")

with st.form("lab", border=True):
    question = st.text_input("Question", placeholder="e.g. which embedding model is used?")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        k = st.slider("Results per arm", 1, 10, 5, width=320)
        st.form_submit_button("Search", icon=":material/search:", type="primary")

if question.strip():
    retriever = get_retriever()
    cols = st.columns(len(ARMS))
    for col, (name, icon, kwargs) in zip(cols, ARMS):
        with col:
            st.markdown(f"{icon} **{name}**")
            started = time.monotonic()
            try:
                hits = retriever.query(question, top_k=k, **kwargs)
            except Exception as exc:
                st.error(str(exc), icon=":material/error:")
                continue
            st.caption(f"{(time.monotonic() - started) * 1000:.0f} ms")
            if not hits:
                st.caption("No matches.")
            for rank, h in enumerate(hits, 1):
                with st.container(border=True, gap="small"):
                    st.markdown(f"**{rank}.** `{h['meta'].get('source', '?')}`")
                    st.caption(f"score {ui.fmt_score(h.get('score'))} · {ui.arm_ranks(h)}")
                    st.caption(ui.excerpt(h["text"], 160))
else:
    st.caption(":material/arrow_upward: Ask something to compare the four searches side by side.")
