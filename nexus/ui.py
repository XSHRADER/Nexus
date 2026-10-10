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

from nexus import brain, cloud, config, feedback, providers, store, truth_check
from nexus.engine import Options
from nexus.retrieve import get_retriever

logger = logging.getLogger("nexus.ui")

# Cloud switch: label shown -> mode the engine understands.
CLOUD_CHOICES = {"Off": "off", "Hard questions": "hard", "Allowed": "allowed"}
# How a question is answered: one model, two blind, or several plus a judge.
MODES = ("One model", "Arena", "Council")
STATUS_ICON = {"ready": ":green[●]", "no_key": ":gray[○]", "invalid_key": ":red[●]",
               "cooling_down": ":orange[●]", "limit_reached": ":orange[●]"}
TRUTH_WORDS = {
    "supported": "supported by your files",
    "not_found": "not in your files",
    "contradicted": "your files say otherwise",
}
TRUTH_COLOURS = {"supported": "green", "not_found": "orange", "contradicted": "red"}


def _cloud_label(mode: str) -> str:
    return next((label for label, value in CLOUD_CHOICES.items() if value == mode), "Off")


_settings = config.get_settings()

# Session keys and their starting values, in one place.
DEFAULTS: dict[str, Any] = {
    "messages": [],
    "pending_action": None,
    "last_question": None,
    "regenerate_with": None,
    "chat_id": None,
    "confirm_delete": None,
    "battle": None,           # an Arena battle waiting for your vote
    "pending_images": [],     # images attached to the question being answered
    "card_revealed": False,   # study page: is the flashcard's answer showing?
    "audio_widget": 0,        # bumped to get a fresh recorder after each use
    # cloud and mode start from nexus.toml (where cloud is off by default)
    "opt_cloud": _cloud_label(_settings.cloud_mode),
    "opt_allow_docs": _settings.allow_docs_to_cloud,
    "opt_allow_paid": _settings.allow_paid,
    "opt_mode": MODES[0],
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
    st.session_state.battle = None
    st.session_state.pending_images = []


def cloud_mode() -> str:
    return CLOUD_CHOICES.get(st.session_state.opt_cloud, "off")


def current_options() -> Options:
    ss = st.session_state
    return Options(
        force_model=None if ss.opt_model == "Auto" else ss.opt_model,
        force_task=None if ss.opt_task == "Auto" else ss.opt_task,
        rag_mode=ss.opt_docs.lower(),
        top_k=ss.opt_top_k,
        rerank=ss.opt_rerank,
        temperature=ss.opt_temperature,
        cloud_mode=cloud_mode(),
        allow_docs=bool(ss.opt_allow_docs),
        allow_paid=bool(ss.opt_allow_paid),
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
    if ss.opt_cloud != DEFAULTS["opt_cloud"]:
        out.append(f"cloud: {ss.opt_cloud.lower()}")
    if cloud_mode() != "off":
        if ss.opt_allow_docs != DEFAULTS["opt_allow_docs"]:
            out.append("documents may go to cloud" if ss.opt_allow_docs else "documents stay local")
        if ss.opt_allow_paid != DEFAULTS["opt_allow_paid"]:
            out.append("paid models allowed" if ss.opt_allow_paid else "no paid models")
    if ss.opt_mode != MODES[0]:
        out.append(f"mode: {ss.opt_mode.lower()}")
    return out


# ---------------------------------------------------------------------------
# Status lookups
# ---------------------------------------------------------------------------


@st.cache_data(ttl=5, show_spinner=False)
def availability() -> dict[str, Any]:
    return providers.availability()


@st.cache_resource(show_spinner=False)
def background_brain() -> brain.Brain:
    """One background brain per app process, started once."""
    b = brain.get_brain()
    if config.get_settings().brain_enabled:
        b.start()
    return b


def ready_cloud_models(avail: dict[str, Any]) -> list[str]:
    """Cloud models that could be pinned right now (cloud on, provider ready)."""
    if cloud_mode() == "off":
        return []
    status = avail.get("cloud") or {}
    return [spec.name for spec in cloud.cloud_specs()
            if status.get(spec.provider, {}).get("status") == "ready"]


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
    bits = []
    # Where the answer was written. Older saved answers predate cloud and
    # carry no flag; they were all local.
    if meta.get("local") is False:
        label = cloud.PROVIDERS.get(meta.get("provider"), {}).get("label", meta.get("provider"))
        where = f"cloud · {badge_text(label)}" if label and label != "council" else "cloud involved"
        bits.append(f":violet-badge[:material/cloud: {where}]")
    elif "local" in meta:
        bits.append(":green-badge[:material/computer: this PC]")
    bits += [f":blue-badge[:material/smart_toy: {badge_text(meta.get('model') or '?')}]",
             f":violet-badge[{badge_text(meta.get('task') or '?')}]"]
    if meta.get("forced_model"):
        bits.append(":orange-badge[:material/push_pin: pinned]")
    if meta.get("sources"):
        bits.append(f":green-badge[:material/description: {len(meta['sources'])} sources]")
    if meta.get("history_used"):
        n = meta["history_used"]
        bits.append(f":gray-badge[:material/history: remembers {n} earlier "
                    f"message{'s' if n != 1 else ''}]")
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
            f"**Task** `{meta.get('task')}` · **Router** `{meta.get('router_method') or 'rules'}` · "
            f"**Difficulty** `{meta.get('complexity') or 0:.2f}` · **Documents** {rag}"
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


def highlighted(text: str, claims: list[dict]) -> str:
    """The answer as markdown, each checked sentence on its verdict's colour."""
    out, at = [], 0
    for claim in sorted(claims, key=lambda c: c["start"]):
        if claim["start"] < at:
            continue
        out.append(md_escape(text[at:claim["start"]]))
        sentence = " ".join(text[claim["start"]:claim["end"]].split())
        colour = TRUTH_COLOURS.get(claim["label"], "gray")
        out.append(f":{colour}-background[{md_escape(sentence)}]")
        at = claim["end"]
    out.append(md_escape(text[at:]))
    return "".join(out).replace("\n", "  \n")


def render_truth(meta: dict, content: str) -> None:
    """Which sentences of the answer your documents back up."""
    report = meta.get("truth")
    if not report:
        return
    if report.get("status") != "ok":
        st.caption(f":material/fact_check: Truth check: {report.get('status', '').replace('_', ' ')}")
        return
    counts = report["counts"]
    how = report["method"] + (" (approximate)" if report.get("approximate") else "")
    st.markdown(
        f":blue-badge[:material/fact_check: trust {round((report['trust'] or 0) * 100)}%] "
        f":green-badge[{counts['supported']} supported] "
        f":orange-badge[{counts['not_found']} not in your files] "
        f":red-badge[{counts['contradicted']} contradicted]"
    )
    with st.expander(f"Truth check — by {how}, against {report['evidence']}",
                     icon=":material/fact_check:"):
        st.markdown(highlighted(content, report["claims"]))
        st.divider()
        for claim in report["claims"]:
            colour = TRUTH_COLOURS.get(claim["label"], "gray")
            st.markdown(f":{colour}[**{TRUTH_WORDS.get(claim['label'], claim['label'])}**] · "
                        f"{md_escape(claim['text'])}")
            if claim.get("source"):
                st.caption(f"`{claim['source']}`: “{excerpt(claim.get('evidence') or '', 240)}”")


def run_truth(message: dict) -> None:
    """Check one saved answer against your documents and keep the result with it."""
    meta = dict(message.get("meta") or {})
    meta["truth"] = truth_check.check(message["content"], sources=meta.get("sources"))
    message["meta"] = meta
    if message.get("id"):
        try:
            store.update_meta(message["id"], {"truth": meta["truth"]})
        except Exception as exc:
            logger.warning("could not save the truth check: %s", exc)


def rateable(message: dict) -> bool:
    """A saved answer written by a model (not a file action or a notice)."""
    meta = message.get("meta") or {}
    return bool(message.get("id")) and meta.get("model") not in (None, "pc-toolkit")


def render_rating(message: dict) -> None:
    """Thumbs up / down under an answer, with a reason after a thumbs down."""
    if not rateable(message):
        return
    meta = message.get("meta") or {}
    mid, current = message["id"], meta.get("rating") or 0
    clicked = None
    with st.container(horizontal=True, gap="small"):
        if st.button(":material/thumb_up:", key=f"up_{mid}", help="Good answer",
                     type="primary" if current == 1 else "tertiary"):
            clicked = 0 if current == 1 else 1
        if st.button(":material/thumb_down:", key=f"down_{mid}", help="Bad answer",
                     type="primary" if current == -1 else "tertiary"):
            clicked = 0 if current == -1 else -1
    if clicked is not None:
        feedback.rate(mid, clicked)
        meta.update(rating=clicked or None, rating_reason=None)
        message["meta"] = meta
        st.rerun()
    if current == -1:
        reasons = list(feedback.REASONS[:4])
        reason = st.pills("What was wrong?", reasons, key=f"why_{mid}",
                          default=meta.get("rating_reason") if meta.get("rating_reason") in reasons
                          else None)
        if reason and reason != meta.get("rating_reason"):
            feedback.rate(mid, -1, reason)
            meta["rating_reason"] = reason


def render_council(meta: dict) -> None:
    """Who sat on the council, where they agreed, and where they did not."""
    c = meta.get("council")
    if not c:
        return
    used = [m for m in c["members"] if m["model"] in c["used"]]

    def where(member: dict) -> str:
        icon = ":material/cloud:" if member.get("local") is False else ":material/computer:"
        return f"{icon} {member['model']}"

    head = f":material/groups: Council of {len(c['used'])}"
    if c.get("auto"):
        head += " (convened automatically: hard question)"
    head += f" · judge {c['judge']}" if c.get("judged") else " · no judge"
    if c.get("agreement") is not None:
        head += f" · answers agree {round(c['agreement'] * 100)}% ({c['agreement_method']})"
    st.caption(head)
    if c["agreements"]:
        st.markdown("**They agree on**\n" + "\n".join(f"- {a}" for a in c["agreements"]))
    if c["disagreements"]:
        lines = ["**They disagree on**"]
        for d in c["disagreements"]:
            lines.append(f"- {d['point']}")
            for k, v in d["positions"].items():
                k = int(k)
                name = where(used[k - 1]) if 0 < k <= len(used) else f"Answer {k}"
                lines.append(f"    - *{name}:* {v}")
        st.markdown("\n".join(lines))
    with st.expander(f"The {len(c['members'])} individual answers", icon=":material/forum:"):
        for m in c["members"]:
            st.markdown(f"**{where(m)}**" + (f" — failed: {m['error']}" if m.get("error") else ""))
            st.markdown(m.get("answer") or "")


def render_message(message: dict) -> None:
    """One stored message, with everything recorded about how it was made."""
    meta = message.get("meta")
    if message["role"] == "user":
        st.markdown(message["content"])
        for image in message.get("image_bytes") or []:
            st.image(image, width=240)
        if meta and meta.get("voice"):
            st.caption(":material/mic: from a voice recording")
        if meta and meta.get("images") and not message.get("image_bytes"):
            st.caption(f":material/image: {meta['images']} image(s) attached")
        return
    if meta:
        render_reasoning(meta)
    st.markdown(message["content"])
    if meta:
        if meta.get("model"):
            render_badges(meta)
        if meta.get("info"):
            st.caption(f":material/info: {meta['info']}")
        render_council(meta)
        if meta.get("model") and not meta.get("council"):
            render_details(meta)
        render_sources(meta)
        render_truth(meta, message["content"])
        render_rating(message)


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
