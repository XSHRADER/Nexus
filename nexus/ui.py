"""
ui.py
Streamlit building blocks shared by the pages in app_pages/: session state,
cached status lookups, and how an answer's metadata is rendered.

Kept out of the engine on purpose -- this is the only NEXUS module that
imports Streamlit.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any

import streamlit as st

from nexus import config, providers, store
from nexus.engine import Options
from nexus.retrieve import get_retriever

logger = logging.getLogger("nexus.ui")

# Session keys and their starting values, in one place.
DEFAULTS: dict[str, Any] = {
    "messages": [],
    "pending_action": None,
    "last_question": None,
    "regenerate_with": None,
    "chat_id": None,
    "confirm_delete": None,
    # answer settings (widget keys)
    "opt_model": "Auto",
    "opt_task": "Auto",
    "opt_docs": "Auto",
    "opt_top_k": config.DEFAULT_TOP_K,
    "opt_rerank": True,
    "opt_temperature": 0.7,
}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else value


def reset_chat(chat_id: str | None = None, messages: list | None = None) -> None:
    st.session_state.messages = messages or []
    st.session_state.chat_id = chat_id
    st.session_state.pending_action = None
    st.session_state.last_question = None
    st.session_state.regenerate_with = None


def current_options() -> Options:
    ss = st.session_state
    return Options(
        force_model=None if ss.opt_model == "Auto" else ss.opt_model,
        force_task=None if ss.opt_task == "Auto" else ss.opt_task,
        rag_mode=ss.opt_docs.lower(),
        top_k=ss.opt_top_k,
        rerank=ss.opt_rerank,
        temperature=ss.opt_temperature,
    )


def overrides() -> list[str]:
    """Settings that differ from fully automatic, as short labels."""
    ss = st.session_state
    out = []
    if ss.opt_model != "Auto":
        out.append(f"model: {ss.opt_model}")
    if ss.opt_task != "Auto":
        out.append(f"task: {ss.opt_task}")
    if ss.opt_docs != "Auto":
        out.append(f"documents: {ss.opt_docs.lower()}")
    if ss.opt_top_k != config.DEFAULT_TOP_K:
        out.append(f"depth: {ss.opt_top_k}")
    if not ss.opt_rerank:
        out.append("no reranking")
    if ss.opt_temperature != DEFAULTS["opt_temperature"]:
        out.append(f"temperature: {ss.opt_temperature:.1f}")
    return out


# ---------------------------------------------------------------------------
# Status lookups
# ---------------------------------------------------------------------------


@st.cache_data(ttl=5, show_spinner=False)
def availability() -> dict[str, Any]:
    return providers.availability()


def index_summary() -> tuple[int, dict[str, int]]:
    """(chunks, {file: chunks}). Cheap: the retriever only reloads on change."""
    try:
        stats = get_retriever().stats()
        return stats["chunks"], stats["files"]
    except Exception as exc:
        logger.warning("index unavailable: %s", exc)
        return 0, {}


def render_status() -> None:
    """Compact health block for the sidebar: Ollama, models in memory, index."""
    avail = availability()
    chunks, files = index_summary()
    installed, loaded = avail.get("ollama", []), avail.get("loaded", [])
    with st.container(border=True, gap="small"):
        if installed:
            st.markdown(f":green-badge[:material/check_circle: Ollama · {len(installed)} models]")
        else:
            st.markdown(":red-badge[:material/error: Ollama offline]")
            st.caption("Start it with `ollama serve`.")
        if chunks:
            st.markdown(f":blue-badge[:material/description: {len(files)} files · {chunks} chunks]")
        else:
            st.markdown(":orange-badge[:material/warning: No documents indexed]")
        if loaded:
            st.caption("In memory: " + ", ".join(loaded))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~:$])")


def md_escape(text: str) -> str:
    """Show document text literally inside markdown."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def excerpt(text: str, limit: int) -> str:
    """One escaped line of document text. Joining lines first defuses block
    syntax that escaping can't (a line of `===` turns the one above it into
    a heading)."""
    flat = " ".join(text.split())
    return md_escape(flat[:limit]) + ("…" if len(flat) > limit else "")


def badge_text(text: str) -> str:
    """Badge labels end at `]`; keep model tags like `qwen2.5:7b` intact."""
    return str(text).replace("[", "(").replace("]", ")")


def fmt_score(score: Any) -> str:
    """Precision that distinguishes rows: RRF scores sit near 0.016, cross-
    encoder logits span roughly -11..+11."""
    if not isinstance(score, (int, float)):
        return "—"
    return f"{score:+.4f}" if abs(score) < 0.1 else f"{score:+.2f}"


def arm_ranks(hit: dict) -> str:
    """`vector #n · keyword #m`: which retrieval arm found the chunk."""
    v, b = hit.get("vector_rank"), hit.get("bm25_rank")
    parts = [f"vector #{v + 1}" if v is not None else None,
             f"keyword #{b + 1}" if b is not None else None]
    return " · ".join(p for p in parts if p) or "—"


def secs(ms: Any) -> float | None:
    return None if ms is None else round(ms / 1000, 2)


def ago(ts: float) -> str:
    minutes = int((time.time() - ts) // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 1440:
        return f"{minutes // 60} h ago"
    return datetime.fromtimestamp(ts).strftime("%d %b")


# ---------------------------------------------------------------------------
# Rendering an answer's metadata
# ---------------------------------------------------------------------------


def render_badges(meta: dict) -> None:
    if meta.get("stopped"):
        st.markdown(":orange-badge[:material/stop_circle: stopped]")
        return
    bits = [f":blue-badge[:material/smart_toy: {badge_text(meta.get('model') or '?')}]",
            f":violet-badge[{badge_text(meta.get('task') or '?')}]"]
    if meta.get("forced_model"):
        bits.append(":orange-badge[:material/push_pin: pinned]")
    if meta.get("sources"):
        bits.append(f":green-badge[:material/description: {len(meta['sources'])} sources]")
    if meta.get("truncated"):
        bits.append(":red-badge[:material/content_cut: prompt cut]")
    if meta.get("elapsed"):
        bits.append(f":gray-badge[:material/schedule: {meta['elapsed']:.1f}s]")
    st.markdown(" ".join(bits))


def render_reasoning(meta: dict) -> None:
    if meta.get("thinking"):
        with st.expander("Reasoning", icon=":material/psychology:"):
            st.markdown(meta["thinking"])


def render_details(meta: dict) -> None:
    """Why this model answered, what it cost, and what was skipped."""
    if meta.get("stopped"):
        return
    with st.expander("How this was answered", icon=":material/insights:"):
        m = meta.get("metrics") or {}
        rag = meta.get("rag_reason") or ("not used" if not meta.get("sources") else "used")
        st.markdown(
            f"**Task** `{meta.get('task')}` · **Difficulty** `{meta.get('complexity') or 0:.2f}` · "
            f"**Documents** {rag}"
        )
        cols = st.columns(4)
        cols[0].metric("First token", f"{secs(m.get('ttft_ms'))} s" if m.get("ttft_ms") else "—")
        cols[1].metric("Speed", f"{m['tokens_per_s']:.0f} tok/s" if m.get("tokens_per_s") else "—")
        cols[2].metric("Prompt", f"{m['prompt_tokens']} tok" if m.get("prompt_tokens") else "—")
        cols[3].metric("Model load", f"{secs(m.get('load_ms'))} s" if m.get("load_ms") else "—")
        if meta.get("auto_task") and meta.get("auto_task") != meta.get("task"):
            st.caption(f":material/info: You overrode the classifier, which read this as "
                       f"`{meta['auto_task']}`.")
        chain = meta.get("chain") or []
        if chain:
            st.caption("Models considered, best first")
            st.dataframe(
                [{"model": c["model"], "score": round(c.get("score", 0), 3), "why": c.get("reason", "")}
                 for c in chain],
                hide_index=True,
                column_config={"score": st.column_config.ProgressColumn(
                    "score", min_value=0.0, max_value=max(1.5, max(c.get("score", 0) for c in chain)),
                    format="%.3f")},
            )
        for a in (meta.get("attempts") or []):
            if a.get("error"):
                st.caption(f":material/skip_next: Skipped **{a['model']}**: {a['error'][:160]}")


def render_sources(meta: dict) -> None:
    sources = meta.get("sources") or []
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/menu_book:"):
        for i, s in enumerate(sources, 1):
            st.markdown(
                f"**[{i}]** `{s.get('source', '?')}` "
                f":gray[· relevance {fmt_score(s.get('score'))} · {arm_ranks(s)}]"
            )
            st.caption(excerpt(s.get("text", ""), 420))


def render_message(message: dict) -> None:
    """One stored message, with everything recorded about how it was made."""
    meta = message.get("meta")
    if meta:
        render_reasoning(meta)
    st.markdown(message["content"])
    if meta:
        render_badges(meta)
        if meta.get("info"):
            st.caption(f":material/info: {meta['info']}")
        render_details(meta)
        render_sources(meta)


def turn_row(turn: dict) -> dict:
    failed = sum(1 for a in turn.get("attempts") or [] if a.get("error"))
    flags = [name for name, on in (("truncated", turn.get("truncated")),
                                   ("stopped", turn.get("stopped")),
                                   ("error", turn.get("error"))) if on]
    return {
        "when": datetime.fromtimestamp(turn["ts"]).strftime("%d %b %H:%M"),
        "task": turn.get("task"),
        "model": turn.get("model") or "—",
        "total s": secs(turn.get("total_ms")),
        "first token s": secs(turn.get("ttft_ms")),
        "tok/s": turn.get("tokens_per_s"),
        "cold load": bool((turn.get("load_ms") or 0) > store.COLD_LOAD_MS),
        "fallbacks": failed,
        "flags": ", ".join(flags),
    }
