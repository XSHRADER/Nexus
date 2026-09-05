"""
app.py
Streamlit UI for NEXUS.

The design goal is a glass box rather than a black box. NEXUS picks a model,
decides whether to consult your documents, and retrieves context -- all
automatically. This UI shows every one of those decisions, and lets you
override each of them, so "why did it answer like that?" is always answerable
without reading the logs.

Automatic stays the default everywhere. Every control has an Auto setting that
reproduces the untouched behaviour exactly.
"""

import io
import json
import time
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import streamlit as st

import providers
from engine import Options, answer, apply_pending

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "documents"
LOG_PATH = PROJECT_DIR / "router_logs.jsonl"

TASKS = ["general", "coding", "reasoning", "planning", "vision", "speech", "system_agent"]

st.set_page_config(
    page_title="NEXUS AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1400px; }
      .nx-badge {
          display: inline-block; padding: 2px 9px; margin-right: 6px;
          border-radius: 999px; font-size: 0.72rem; font-weight: 600;
          letter-spacing: .02em; border: 1px solid rgba(128,128,128,.35);
      }
      .nx-model   { background: rgba(56,139,253,.16); }
      .nx-task    { background: rgba(163,113,247,.16); }
      .nx-rag     { background: rgba(46,160,67,.18); }
      .nx-time    { background: rgba(128,128,128,.14); }
      .nx-pin     { background: rgba(219,109,40,.20); }
      .nx-src {
          border-left: 3px solid rgba(56,139,253,.55);
          padding: .45rem .7rem; margin-bottom: .55rem;
          background: rgba(128,128,128,.07); border-radius: 0 6px 6px 0;
          font-size: .82rem;
      }
      .nx-src-head { font-weight: 600; font-size: .78rem; margin-bottom: .25rem; }
      .nx-dim { opacity: .62; font-size: .76rem; }
      div[data-testid="stMetricValue"] { font-size: 1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _retriever():
    from retrieve import Retriever

    return Retriever()


@st.cache_data(ttl=5, show_spinner=False)
def _availability():
    return providers.availability()


def _index_summary():
    """Per-file chunk counts straight from the vector store."""
    try:
        retriever = _retriever()
        retriever._ensure_fresh()
        counts: dict[str, int] = {}
        for record in retriever._doc_by_id.values():
            src = record["meta"].get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return retriever._indexed_count, counts
    except Exception:
        return 0, {}


def _state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


_state("messages", [])
_state("pending_action", None)
_state("last_question", None)
_state("regenerate_with", None)


# ---------------------------------------------------------------------------
# Sidebar: status + every override
# ---------------------------------------------------------------------------

avail = _availability()
installed = avail.get("ollama", [])
loaded = avail.get("loaded", [])
chunk_total, per_file = _index_summary()

with st.sidebar:
    st.markdown("### NEXUS")

    if installed:
        st.success(f"Ollama · {len(installed)} models")
    else:
        st.error("Ollama unreachable — `ollama serve`")
    col_a, col_b = st.columns(2)
    col_a.metric("Chunks", chunk_total)
    col_b.metric("In VRAM", len(loaded))
    if loaded:
        st.caption("Resident: " + ", ".join(loaded))

    st.divider()
    st.markdown("#### Answer settings")

    model_choice = st.selectbox(
        "Model",
        ["Auto (recommended)"] + installed,
        help=(
            "Auto scores every installed model against the task, how hard the "
            "prompt looks, and what is already in VRAM. Pin one to override it."
        ),
    )
    task_choice = st.selectbox(
        "Task",
        ["Auto"] + TASKS,
        help="Override the intent classifier if it reads a question wrong.",
    )

    rag_choice = st.radio(
        "Your documents",
        ["Auto", "Always", "Never"],
        horizontal=True,
        help=(
            "Auto decides from keywords in your question, so a question whose "
            "answer *is* in your documents can still be missed. Always forces "
            "retrieval; Never skips it and answers from the model alone."
        ),
    )

    top_k = st.slider(
        "Retrieval depth (top_k)", 1, 20, 10,
        help=(
            "How many chunks to put in front of the model. Chunks are ~240 "
            "tokens, so 10 is about 2,200 tokens of context."
        ),
        disabled=(rag_choice == "Never"),
    )
    rerank = st.checkbox(
        "Cross-encoder reranking", value=True,
        help="Re-scores fused candidates against the question. Slower, sharper.",
        disabled=(rag_choice == "Never"),
    )
    temperature = st.slider(
        "Creativity (temperature)", 0.0, 1.5, 0.7, 0.1,
        help="0 is deterministic and factual; higher wanders more.",
    )

    st.divider()
    if st.button("New chat", width='stretch'):
        st.session_state.messages = []
        st.session_state.pending_action = None
        st.session_state.last_question = None
        st.rerun()

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"[{m['role']}] {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "Export transcript",
            transcript,
            file_name=f"nexus-chat-{datetime.now():%Y%m%d-%H%M}.txt",
            width='stretch',
        )

options = Options(
    force_model=None if model_choice.startswith("Auto") else model_choice,
    force_task=None if task_choice == "Auto" else task_choice,
    rag_mode=rag_choice.lower(),
    top_k=top_k,
    rerank=rerank,
    temperature=temperature,
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def fmt_score(score) -> str:
    """Format a relevance score at a precision that actually distinguishes rows.

    Cross-encoder logits span roughly -11..+11, but RRF scores sit near
    1/(60+rank) -- around 0.016 -- so two decimals collapsed every fused row
    to "+0.02".
    """
    if not isinstance(score, (int, float)):
        return "—"
    return f"{score:+.4f}" if abs(score) < 0.1 else f"{score:+.2f}"


def arm_ranks(hit: dict) -> str:
    """`v=`/`b=` ranks, which is the informative part for a fused result."""
    v, b = hit.get("vector_rank"), hit.get("bm25_rank")
    parts = [p for p in (
        f"v={v}" if v is not None else None,
        f"b={b}" if b is not None else None,
    ) if p]
    return " · ".join(parts) if parts else "—"


def render_badges(meta: dict) -> None:
    bits = [f"<span class='nx-badge nx-model'>{meta.get('model', '?')}</span>",
            f"<span class='nx-badge nx-task'>{meta.get('task', '?')}</span>"]
    if meta.get("forced_model"):
        bits.append("<span class='nx-badge nx-pin'>pinned</span>")
    if meta.get("needs_rag"):
        bits.append(
            f"<span class='nx-badge nx-rag'>{len(meta.get('sources', []))} sources</span>"
        )
    if meta.get("elapsed"):
        bits.append(f"<span class='nx-badge nx-time'>{meta['elapsed']:.1f}s</span>")
    st.markdown("".join(bits), unsafe_allow_html=True)


def render_why(meta: dict) -> None:
    with st.expander("Why this model"):
        st.markdown(
            f"**Task** `{meta.get('task')}`  ·  "
            f"**Difficulty** `{meta.get('complexity', 0):.2f}`  ·  "
            f"**Prompt** `{meta.get('prompt_chars', 0)} chars`"
        )
        if meta.get("auto_task") and meta.get("auto_task") != meta.get("task"):
            st.warning(
                f"You overrode the classifier: it read this as "
                f"`{meta['auto_task']}`."
            )
        chain = meta.get("chain") or []
        if chain:
            st.caption("Models considered, best first:")
            st.dataframe(
                [
                    {
                        "model": c["model"],
                        "score": round(c.get("score", 0), 3),
                        "why": c.get("reason", ""),
                    }
                    for c in chain
                ],
                width='stretch',
                hide_index=True,
            )
        attempts = meta.get("attempts") or []
        failed = [a for a in attempts if a.get("error")]
        if failed:
            st.caption("Skipped:")
            for a in failed:
                st.text(f"  {a['model']}: {a['error'][:140]}")


def render_sources(meta: dict) -> None:
    sources = meta.get("sources") or []
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        st.caption(
            "`v` = rank from vector search, `b` = rank from BM25 keyword search. "
            "A chunk found by both is what hybrid retrieval is for."
        )
        for i, s in enumerate(sources, 1):
            arms = arm_ranks(s)
            score_txt = fmt_score(s.get("score"))
            st.markdown(
                f"<div class='nx-src'><div class='nx-src-head'>"
                f"{i}. {s.get('source', '?')} "
                f"<span class='nx-dim'>· score {score_txt} · {arms}</span></div>"
                f"{s.get('text', '')[:400].replace('<', '&lt;')}…</div>",
                unsafe_allow_html=True,
            )


def run_turn(question: str, opts: Options) -> None:
    """Generate one answer, streaming it into the transcript."""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        buffer: list[str] = []

        def on_token(tok: str) -> None:
            buffer.append(tok)
            placeholder.markdown("".join(buffer) + "▌")

        try:
            result = answer(question, options=opts, on_token=on_token)
        except Exception as exc:
            placeholder.empty()
            st.error(f"NEXUS could not answer: {exc}")
            return

        placeholder.markdown(result.get("answer", ""))

        meta = {
            k: result.get(k)
            for k in (
                "model", "task", "needs_rag", "sources", "elapsed", "chain",
                "complexity", "attempts", "prompt_chars", "info", "auto_task",
            )
        }
        meta["forced_model"] = bool(opts.force_model)
        render_badges(meta)
        if result.get("info"):
            st.info(result["info"])
        render_why(meta)
        render_sources(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.get("answer", ""), "meta": meta}
    )
    if result.get("requires_confirmation"):
        st.session_state.pending_action = result["pending"]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_docs, tab_lab, tab_diag = st.tabs(
    ["Chat", "Documents", "Retrieval lab", "Diagnostics"]
)


# -- Chat -------------------------------------------------------------------
with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            "#### Ask anything\n"
            "NEXUS reads the question, picks the model that suits it, and pulls "
            "in your documents when they help. Every choice it makes is shown "
            "underneath the answer — and every one can be overridden in the "
            "sidebar."
        )
        cols = st.columns(3)
        starters = [
            "What does this project use to store embeddings?",
            "Compare hybrid retrieval against dense-only search.",
            "Sort my Downloads folder",
        ]
        for col, text in zip(cols, starters):
            if col.button(text, width='stretch'):
                st.session_state.last_question = text
                st.session_state.messages.append({"role": "user", "content": text})
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            meta = message.get("meta")
            if meta:
                render_badges(meta)
                render_why(meta)
                render_sources(meta)

    # A queued question (from a starter button or a regenerate click).
    queued = st.session_state.last_question
    if queued and (
        not st.session_state.messages
        or st.session_state.messages[-1]["role"] == "user"
    ):
        st.session_state.last_question = None
        turn_options = options
        alt = st.session_state.regenerate_with
        if alt:
            st.session_state.regenerate_with = None
            turn_options = replace(options, force_model=alt)
        run_turn(queued, turn_options)
        st.rerun()

    if st.session_state.pending_action:
        pending = st.session_state.pending_action
        with st.chat_message("assistant"):
            st.warning(
                f"Waiting for confirmation: **{pending['op']}** on `{pending['path']}`"
            )
            c1, c2 = st.columns(2)
            if c1.button("Apply changes", type="primary", width='stretch'):
                try:
                    msg = apply_pending(pending)["answer"]
                except Exception as exc:
                    msg = f"Action failed: {exc}"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.session_state.pending_action = None
                st.rerun()
            if c2.button("Cancel", width='stretch'):
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Cancelled — nothing on disk changed."}
                )
                st.session_state.pending_action = None
                st.rerun()

    # Offer a re-run on a different model, using the chain NEXUS already scored.
    last = st.session_state.messages[-1] if st.session_state.messages else None
    if last and last["role"] == "assistant" and last.get("meta"):
        chain = last["meta"].get("chain") or []
        used = last["meta"].get("model")
        alts = [c["model"] for c in chain if c["model"] != used][:3]
        if alts and st.session_state.messages:
            question = next(
                (m["content"] for m in reversed(st.session_state.messages)
                 if m["role"] == "user"),
                None,
            )
            if question:
                st.caption("Not convinced? Answer again with:")
                cols = st.columns(len(alts))
                for col, alt in zip(cols, alts):
                    if col.button(alt, key=f"regen_{alt}", width='stretch'):
                        st.session_state.messages.append(
                            {"role": "user", "content": question}
                        )
                        st.session_state.last_question = question
                        st.session_state.regenerate_with = alt
                        st.rerun()

    prompt = st.chat_input("Ask NEXUS AI...")
    if prompt:
        st.session_state.pending_action = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_question = prompt
        st.rerun()


# -- Documents --------------------------------------------------------------
with tab_docs:
    st.markdown("#### Knowledge base")
    st.caption(
        "Everything here is indexed locally and never leaves this machine. "
        "Re-indexing only re-embeds files whose contents changed."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Files", len(per_file))
    c2.metric("Chunks", chunk_total)
    c3.metric("Avg chunks/file", round(chunk_total / max(1, len(per_file)), 1))

    if per_file:
        st.dataframe(
            [{"file": k, "chunks": v} for k, v in sorted(per_file.items())],
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("Nothing indexed yet. Add documents below, then re-index.")

    st.divider()
    uploads = st.file_uploader(
        "Add documents",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploads:
        saved = []
        for f in uploads:
            target = DOCS_DIR / f.name
            target.write_bytes(f.getbuffer())
            saved.append(f.name)
        st.success(f"Saved {len(saved)} file(s): {', '.join(saved)} — now re-index.")

    c1, c2 = st.columns(2)
    do_incremental = c1.button("Re-index changed files", width='stretch')
    do_rebuild = c2.button("Full rebuild", width='stretch')

    if do_incremental or do_rebuild:
        import ingest

        with st.spinner("Indexing..."):
            log = io.StringIO()
            try:
                with redirect_stdout(log):
                    ingest.main(force_rebuild=do_rebuild)
                ok = True
            except Exception as exc:
                ok = False
                log.write(f"\nFAILED: {exc}")
        st.code(log.getvalue() or "(no output)", language="text")
        if ok:
            _retriever().refresh()
            st.cache_data.clear()
            st.success("Index updated.")


# -- Retrieval lab ----------------------------------------------------------
with tab_lab:
    st.markdown("#### Retrieval lab")
    st.caption(
        "Retrieval only — no model runs here. Compare what each arm returns for "
        "the same question. This is the same measurement `eval_rag.py` reports, "
        "one query at a time."
    )

    lab_q = st.text_input("Question", placeholder="e.g. which embedding model is used?")
    lab_k = st.slider("Results per arm", 1, 10, 5, key="lab_k")

    if lab_q:
        arms = [
            ("Dense only", dict(use_vector=True, use_bm25=False, rerank=False)),
            ("BM25 only", dict(use_vector=False, use_bm25=True, rerank=False)),
            ("Hybrid (RRF)", dict(use_vector=True, use_bm25=True, rerank=False)),
            ("Hybrid + rerank", dict(use_vector=True, use_bm25=True, rerank=True)),
        ]
        cols = st.columns(len(arms))
        retriever = _retriever()
        for col, (name, kwargs) in zip(cols, arms):
            with col:
                st.markdown(f"**{name}**")
                started = time.monotonic()
                try:
                    hits = retriever.query(lab_q, top_k=lab_k, **kwargs)
                except Exception as exc:
                    st.error(str(exc))
                    continue
                st.caption(f"{(time.monotonic() - started) * 1000:.0f} ms")
                if not hits:
                    st.caption("no matches")
                for rank, h in enumerate(hits, 1):
                    st.markdown(
                        f"<div class='nx-src'><div class='nx-src-head'>"
                        f"{rank}. {h['meta'].get('source', '?')}</div>"
                        f"<span class='nx-dim'>{fmt_score(h.get('score'))}"
                        f" · {arm_ranks(h)}</span><br>"
                        f"<span class='nx-dim'>{h['text'][:110].replace('<', '&lt;')}…"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )


# -- Diagnostics ------------------------------------------------------------
with tab_diag:
    st.markdown("#### Diagnostics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Models installed", len(installed))
    c2.metric("Loaded in VRAM", len(loaded))
    c3.metric("Indexed chunks", chunk_total)

    st.caption("Installed: " + (", ".join(installed) if installed else "none"))

    st.divider()
    st.markdown("##### Index configuration")
    try:
        import ingest as _ingest

        st.json(_ingest.index_config())
    except Exception as exc:
        st.caption(f"unavailable ({exc})")

    st.divider()
    st.markdown("##### Recent routing decisions")
    if LOG_PATH.exists():
        try:
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-15:]
            rows = []
            for line in reversed(lines):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    {
                        "task": d.get("task"),
                        "model": d.get("model"),
                        "difficulty": d.get("complexity"),
                        "rag": d.get("needs_rag"),
                        "chain": " → ".join(
                            c["model"] for c in (d.get("chain") or [])[:3]
                        ),
                    }
                )
            if rows:
                st.dataframe(rows, width='stretch', hide_index=True)
            else:
                st.caption("No decisions logged yet.")
        except Exception as exc:
            st.caption(f"could not read log ({exc})")
    else:
        st.caption("No routing log yet — ask something first.")
