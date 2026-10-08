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

import base64
import io
import json
import time
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import streamlit as st

import html
import brain
import cloud
import feedback
import providers
import truth_check
from config import get_settings
from engine import (Options, answer, apply_pending, arena, council, council_message_meta,
                    council_recommended, transcribe)
from store import get_store

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "documents"
LOG_PATH = PROJECT_DIR / "router_logs.jsonl"

TASKS = ["general", "coding", "reasoning", "planning", "vision", "speech", "system_agent"]

CLOUD_CHOICES = {"Off — this PC only": "off", "Hard questions only": "hard", "Allowed": "allowed"}
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "gif"]
AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "webm", "flac"]
STATUS_ICON = {"ready": "🟢", "no_key": "⚪", "invalid_key": "🔴",
               "cooling_down": "🟠", "limit_reached": "🟠"}

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
      .nx-mem     { background: rgba(236,72,153,.16); }
      .nx-cloud   { background: rgba(167,139,250,.22); }
      .nx-trust   { background: rgba(103,232,249,.18); }
      .nx-claim-supported    { background: rgba(46,160,67,.20); border-bottom: 2px solid #2ea043; }
      .nx-claim-not_found    { background: rgba(219,154,4,.18); border-bottom: 2px dashed #d29922; }
      .nx-claim-contradicted { background: rgba(248,81,73,.20); border-bottom: 2px solid #f85149; }
      .nx-hl { line-height: 1.75; font-size: .95rem; }
      .nx-local   { background: rgba(46,160,67,.12); }
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


@st.cache_resource(show_spinner=False)
def _brain():
    """One background brain per app process, started once."""
    b = brain.get_brain()
    if get_settings().brain_enabled:
        b.start()
    return b


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
_state("chat_id", None)
_state("pending_images", [])
_state("audio_widget", 0)
_state("battle", None)          # an Arena battle waiting for your vote
_state("card_revealed", False)


def add_message(role: str, content: str, meta: dict | None = None) -> None:
    """Append a turn to the screen and to the saved chat.

    The chat row is created on the first message, titled after it, so empty
    "New chat" clicks never leave blank chats behind.
    """
    store = get_store()
    if st.session_state.chat_id is None:
        st.session_state.chat_id = store.create_chat(content if role == "user" else "New chat")
    message_id = store.add_message(st.session_state.chat_id, role, content, meta)
    message = {"id": message_id, "role": role, "content": content}
    if meta:
        message["meta"] = meta
    st.session_state.messages.append(message)


def open_chat(chat_id: int) -> None:
    st.session_state.chat_id = chat_id
    st.session_state.messages = get_store().get_messages(chat_id)
    st.session_state.pending_action = None
    st.session_state.last_question = None


def history_for(question: str, regenerate: bool) -> list[dict]:
    """The conversation before `question`, which is the last message on screen.

    A regenerate re-asks the previous question on another model. That model
    should not see the first model's answer, or "answer again" just becomes
    "rephrase what the other model said", so the earlier Q/A pair is dropped.
    """
    earlier = st.session_state.messages[:-1]
    if regenerate and len(earlier) >= 2 and earlier[-2].get("content") == question:
        earlier = earlier[:-2]
    return earlier


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

    st.markdown("#### Cloud")
    settings = get_settings()
    default_label = next(k for k, v in CLOUD_CHOICES.items() if v == settings.cloud_mode)
    cloud_label = st.radio(
        "Cloud", list(CLOUD_CHOICES), index=list(CLOUD_CHOICES).index(default_label),
        label_visibility="collapsed",
        help=(
            "Off: nothing leaves this PC. Hard questions only: cloud models are used "
            "for hard prompts and for things no local model can do (like images). "
            "Allowed: cloud competes on every prompt, but easy ones still go local first."
        ),
    )
    cloud_mode = CLOUD_CHOICES[cloud_label]
    allow_docs = st.checkbox(
        "Send documents to cloud", value=settings.allow_docs_to_cloud,
        disabled=cloud_mode == "off",
        help="Off: questions that use your documents are always answered on this PC.",
    )
    allow_paid = st.checkbox(
        "Allow paid models", value=settings.allow_paid, disabled=cloud_mode == "off",
        help="OpenRouter auto is billed per token with no free tier.",
    )
    arena_on = st.toggle(
        "⚔️ Arena mode", value=False,
        help="Two models answer each question with their names hidden; you pick the "
             "better one. Your votes build the leaderboard NEXUS will learn from.",
    )
    council_on = st.toggle(
        "🏛️ Council mode", value=False,
        help="Several models answer, a judge lists where they agree and disagree, "
             "and writes one merged answer. Slower; best for hard questions.",
    )
    if arena_on and council_on:
        st.caption("Arena and Council are both on — Council is used.")
    cloud_status = avail.get("cloud") or {}
    with st.expander("Providers", expanded=False):
        for prov, info in cloud_status.items():
            label = cloud.PROVIDERS[prov]["label"]
            st.caption(f"{STATUS_ICON.get(info['status'], '⚪')} **{label}** · {info['detail']}")
    ready_cloud_models = [
        spec.name for spec in cloud.cloud_specs()
        if cloud_mode != "off" and cloud_status.get(spec.provider, {}).get("status") == "ready"
    ]

    st.divider()
    model_choice = st.selectbox(
        "Model",
        ["Auto (recommended)"] + installed + ready_cloud_models,
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
        st.session_state.chat_id = None
        st.rerun()

    st.markdown("#### Past chats")
    past_chats = get_store().list_chats(limit=30)
    if not past_chats:
        st.caption("Chats are saved here automatically.")
    for chat in past_chats:
        is_open = chat["id"] == st.session_state.chat_id
        label = ("▸ " if is_open else "") + chat["title"]
        if st.button(
            label, key=f"chat_{chat['id']}", width='stretch',
            help=f"{chat['message_count']} messages · "
                 f"{datetime.fromtimestamp(chat['updated_at']):%d %b %H:%M}",
            disabled=is_open,
        ):
            open_chat(chat["id"])
            st.rerun()
    if st.session_state.chat_id is not None:
        if st.button("Delete this chat", width='stretch'):
            get_store().delete_chat(st.session_state.chat_id)
            st.session_state.chat_id = None
            st.session_state.messages = []
            st.session_state.pending_action = None
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
    cloud_mode=cloud_mode,
    allow_docs=allow_docs,
    allow_paid=allow_paid,
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
    if meta.get("local") is False:
        label = cloud.PROVIDERS.get(meta.get("provider"), {}).get("label", meta.get("provider"))
        where = f"<span class='nx-badge nx-cloud'>🌐 cloud · {label}</span>"
    else:
        where = "<span class='nx-badge nx-local'>💻 this PC</span>"
    bits = [where,
            f"<span class='nx-badge nx-model'>{meta.get('model', '?')}</span>",
            f"<span class='nx-badge nx-task'>{meta.get('task', '?')}</span>"]
    if meta.get("forced_model"):
        bits.append("<span class='nx-badge nx-pin'>pinned</span>")
    if meta.get("needs_rag"):
        bits.append(
            f"<span class='nx-badge nx-rag'>{len(meta.get('sources', []))} sources</span>"
        )
    if meta.get("history_used"):
        n = meta["history_used"]
        bits.append(
            f"<span class='nx-badge nx-mem'>remembers {n} earlier "
            f"message{'s' if n != 1 else ''}</span>"
        )
    if meta.get("elapsed"):
        bits.append(f"<span class='nx-badge nx-time'>{meta['elapsed']:.1f}s</span>")
    st.markdown("".join(bits), unsafe_allow_html=True)


def render_why(meta: dict) -> None:
    with st.expander("Why this model"):
        st.markdown(
            f"**Task** `{meta.get('task')}`  ·  "
            f"**Router** `{meta.get('router_method') or 'rules'}`  ·  "
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


TRUTH_WORDS = {
    "supported": "🟢 supported by your files",
    "not_found": "🟡 not in your files",
    "contradicted": "🔴 your files say otherwise",
}


def highlighted_html(text: str, claims: list[dict]) -> str:
    out, at = [], 0
    for c in sorted(claims, key=lambda c: c["start"]):
        if c["start"] < at:
            continue
        out.append(html.escape(text[at:c["start"]]))
        tip = TRUTH_WORDS[c["label"]] + (f" — {c['source']}" if c.get("source") else "")
        out.append(f"<span class='nx-claim-{c['label']}' title='{html.escape(tip, quote=True)}'>"
                   f"{html.escape(text[c['start']:c['end']])}</span>")
        at = c["end"]
    out.append(html.escape(text[at:]))
    return "<div class='nx-hl'>" + "".join(out).replace("\n", "<br>") + "</div>"


def render_truth(meta: dict, content: str) -> None:
    report = meta.get("truth")
    if not report:
        return
    if report.get("status") != "ok":
        st.caption(f"Truth check: {report.get('status', '').replace('_', ' ')}")
        return
    c = report["counts"]
    how = report["method"] + (" (approximate)" if report.get("approximate") else "")
    st.markdown(
        f"<span class='nx-badge nx-trust'>Trust {round((report['trust'] or 0) * 100)}%</span>"
        f"<span class='nx-badge'>🟢 {c['supported']} supported</span>"
        f"<span class='nx-badge'>🟡 {c['not_found']} not in your files</span>"
        f"<span class='nx-badge'>🔴 {c['contradicted']} contradicted</span>",
        unsafe_allow_html=True,
    )
    with st.expander(f"Truth check — checked by {how} against {report['evidence']}"):
        st.markdown(highlighted_html(content, report["claims"]), unsafe_allow_html=True)
        st.divider()
        for claim in report["claims"]:
            line = f"{TRUTH_WORDS[claim['label']]} · **{claim['text']}**"
            if claim.get("source"):
                line += f"  \n<span class='nx-dim'>{html.escape(claim['source'])}: “" \
                        f"{html.escape((claim.get('evidence') or '')[:240])}”</span>"
            st.markdown(line, unsafe_allow_html=True)


def render_rating(message: dict) -> None:
    """👍 / 👎 under an answer, with a reason after a 👎."""
    meta = message.get("meta") or {}
    if not message.get("id") or meta.get("model") in (None, "pc-toolkit"):
        return
    current = meta.get("rating") or 0
    c1, c2, _ = st.columns([1, 1, 10])
    clicked = None
    if c1.button("👍", key=f"up_{message['id']}", type="primary" if current == 1 else "secondary"):
        clicked = 0 if current == 1 else 1
    if c2.button("👎", key=f"down_{message['id']}", type="primary" if current == -1 else "secondary"):
        clicked = 0 if current == -1 else -1
    if clicked is not None:
        feedback.rate(message["id"], clicked)
        meta.update(rating=clicked or None, rating_reason=None)
        message["meta"] = meta
        st.rerun()
    if current == -1:
        reasons = list(feedback.REASONS[:4])
        reason = st.radio("What was wrong?", reasons, horizontal=True, key=f"why_{message['id']}",
                          index=reasons.index(meta["rating_reason"])
                          if meta.get("rating_reason") in reasons else None)
        if reason and reason != meta.get("rating_reason"):
            feedback.rate(message["id"], -1, reason)
            meta["rating_reason"] = reason


def run_council(question: str, opts: Options, history: list[dict], images: list[dict],
                auto: bool = False) -> None:
    with st.spinner("🏛️ The council is answering (several models, then a judge)..."):
        out = council(question, options=opts, history=history, images=images)
    if out.get("error"):
        add_message("assistant", f"🏛️ Council: {out['error']}")
        return
    add_message("assistant", out["answer"], council_message_meta(out, auto=auto))


def render_council(meta: dict) -> None:
    c = meta.get("council")
    if not c:
        return
    used = [m for m in c["members"] if m["model"] in c["used"]]
    where = lambda m: ("🌐 " if m.get("local") is False else "💻 ") + m["model"]  # noqa: E731
    head = f"🏛️ Council of {len(c['used'])}"
    if c.get("auto"):
        head += " (convened automatically: hard question)"
    head += f" · judge {c['judge']}" if c.get("judged") else " · no judge"
    if c.get("agreement") is not None:
        head += f" · answers agree {round(c['agreement'] * 100)}% ({c['agreement_method']})"
    st.caption(head)
    if c["agreements"]:
        st.markdown("**✅ They agree on**\n" + "\n".join(f"- {a}" for a in c["agreements"]))
    if c["disagreements"]:
        lines = ["**⚠️ They disagree on**"]
        for d in c["disagreements"]:
            lines.append(f"- {d['point']}")
            for k, v in d["positions"].items():
                k = int(k)
                name = where(used[k - 1]) if 0 < k <= len(used) else f"Answer {k}"
                lines.append(f"    - *{name}:* {v}")
        st.markdown("\n".join(lines))
    with st.expander(f"Show the {len(c['members'])} individual answers"):
        for m in c["members"]:
            st.markdown(f"**{where(m)}**" + (f" — failed: {m['error']}" if m.get("error") else ""))
            st.markdown(m.get("answer") or "")


def run_arena(question: str, opts: Options, history: list[dict], images: list[dict]) -> None:
    with st.spinner("⚔️ Two models are answering..."):
        out = arena(question, options=opts, history=history, images=images,
                    chat_id=st.session_state.chat_id)
    if out.get("error"):
        add_message("assistant", f"⚔️ Arena: {out['error']}")
        return
    st.session_state.battle = out


def render_battle(battle: dict) -> None:
    """Blind A/B answers and the vote; names appear only after voting."""
    st.markdown("#### ⚔️ Arena — which answer is better?")
    if battle.get("note"):
        st.caption(battle["note"])
    cols = st.columns(2)
    for col, side in zip(cols, ("a", "b")):
        with col.container(border=True):
            st.markdown(f"**Answer {side.upper()}**")
            st.markdown(battle[side]["answer"])
    v1, v2, v3, v4 = st.columns(4)
    choice = None
    if v1.button("A is better", type="primary", width='stretch'):
        choice = "a"
    if v2.button("B is better", type="primary", width='stretch'):
        choice = "b"
    if v3.button("Tie", width='stretch'):
        choice = "tie"
    if v4.button("Both bad", width='stretch'):
        choice = "both_bad"
    if choice is None:
        return
    decided = feedback.vote(battle["battle_id"], choice)
    st.session_state.battle = None
    reveal = (f"A was **{decided['model_a']}**, B was **{decided['model_b']}**.")
    if choice == "both_bad":
        add_message("assistant", f"⚔️ You marked both answers as bad. {reveal}",
                    {"arena": {"battle_id": decided["id"], "winner": choice}})
    else:
        side = "b" if choice == "b" else "a"
        model = decided[f"model_{side}"]
        spec = providers.spec_by_name(model)
        add_message("assistant", battle[side]["answer"], {
            "model": model, "provider": decided[f"provider_{side}"],
            "local": spec.is_local if spec else True, "task": battle.get("task"),
            "sources": battle[side].get("sources"), "needs_rag": battle[side].get("needs_rag"),
            "info": f"⚔️ Arena: you picked {'a tie' if choice == 'tie' else 'answer ' + side.upper()}. {reveal}",
            "arena": {"battle_id": decided["id"], "winner": choice},
        })
    st.rerun()


def run_truth(message: dict) -> None:
    """Check one saved answer, keep the result with it, and show it."""
    meta = dict(message.get("meta") or {})
    with st.spinner("Checking each sentence against your documents..."):
        meta["truth"] = truth_check.check(message["content"], sources=meta.get("sources"))
    message["meta"] = meta
    if message.get("id"):
        get_store().update_meta(message["id"], {"truth": meta["truth"]})


def run_turn(question: str, opts: Options, history: list[dict],
             images: list[dict] | None = None) -> None:
    """Generate one answer, streaming it into the transcript."""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        buffer: list[str] = []

        def on_token(tok: str) -> None:
            buffer.append(tok)
            placeholder.markdown("".join(buffer) + "▌")

        try:
            result = answer(question, options=opts, on_token=on_token, history=history,
                            images=images)
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
                "history_used", "provider", "local", "router_method",
            )
        }
        meta["forced_model"] = bool(opts.force_model)
        render_badges(meta)
        if result.get("info"):
            st.info(result["info"])
        render_why(meta)
        render_sources(meta)

    add_message("assistant", result.get("answer", ""), meta)
    message = st.session_state.messages[-1]
    # Answers built from your documents are checked straight away; others
    # get a button, since most of what they say won't be in your files.
    if result.get("task") != "system_agent" and (meta.get("needs_rag") or meta.get("sources")):
        run_truth(message)
    if result.get("requires_confirmation"):
        st.session_state.pending_action = result["pending"]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

the_brain = _brain()
unread = the_brain.unread()
study = the_brain.study_stats()
tab_chat, tab_inbox, tab_board, tab_docs, tab_lab, tab_diag = st.tabs(
    ["Chat", f"Inbox ({unread}) · Study ({study['due']})", "Leaderboard", "Documents",
     "Retrieval lab", "Diagnostics"]
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
                add_message("user", text)
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for img in message.get("image_bytes") or []:
                st.image(img, width=240)
            meta = message.get("meta")
            if meta and message["role"] == "user":
                if meta.get("voice"):
                    st.caption("🎤 from a voice recording")
                if meta.get("images") and not message.get("image_bytes"):
                    st.caption(f"📎 {meta['images']} image(s) attached")
            elif meta:
                render_badges(meta)
                if (meta.get("arena") or meta.get("council")) and meta.get("info"):
                    st.info(meta["info"])
                render_council(meta)
                render_why(meta)
                render_sources(meta)
                render_truth(meta, message["content"])
                render_rating(message)

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
        images = st.session_state.pending_images
        st.session_state.pending_images = []
        if council_on and not alt:
            run_council(queued, turn_options, history_for(queued, regenerate=False), images)
        elif arena_on and not alt:
            run_arena(queued, turn_options, history_for(queued, regenerate=False), images)
        elif not alt and not images and council_recommended(queued, turn_options):
            run_council(queued, turn_options, history_for(queued, regenerate=False), images,
                        auto=True)
        else:
            run_turn(queued, turn_options, history_for(queued, regenerate=bool(alt)), images)
        st.rerun()

    if st.session_state.battle:
        render_battle(st.session_state.battle)

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
                add_message("assistant", msg)
                st.session_state.pending_action = None
                st.rerun()
            if c2.button("Cancel", width='stretch'):
                add_message("assistant", "Cancelled — nothing on disk changed.")
                st.session_state.pending_action = None
                st.rerun()

    # Offer a re-run on a different model, using the chain NEXUS already scored.
    last = st.session_state.messages[-1] if st.session_state.messages else None
    if (last and last["role"] == "assistant" and last.get("meta")
            and not last["meta"].get("truth") and last["meta"].get("model") not in (None, "pc-toolkit")):
        if st.button("🔎 Check this answer against my files"):
            run_truth(last)
            st.rerun()
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
                        # Asking another model is a quiet 👎 for this one.
                        feedback.record_signal("regenerate", question,
                                               task=last["meta"].get("task"), model=used,
                                               value=alt, message_id=last.get("id"))
                        add_message("user", question)
                        st.session_state.last_question = question
                        st.session_state.regenerate_with = alt
                        st.rerun()

    def submit_voice(audio_bytes: bytes, name: str, mime: str, typed: str = "") -> None:
        try:
            with st.spinner("Transcribing..."):
                heard = transcribe(audio_bytes, name, mime, options)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        text = f"{typed}\n{heard['text']}".strip() if typed else heard["text"]
        st.session_state.pending_action = None
        add_message("user", text, {"voice": True})
        st.session_state.last_question = text
        st.rerun()

    with st.expander("🎤 Ask by voice"):
        recording = st.audio_input("Record a question",
                                   key=f"audio_{st.session_state.audio_widget}")
        if recording is not None:
            st.session_state.audio_widget += 1  # fresh widget next run
            submit_voice(recording.getvalue(), "recording.wav", "audio/wav")

    submitted = st.chat_input(
        "Ask NEXUS AI... (attach an image or a voice recording with the clip)",
        accept_file=True, file_type=IMAGE_TYPES + AUDIO_TYPES,
    )
    if submitted:
        typed = (submitted.text or "").strip()
        files = list(submitted.files or [])
        audio = next((f for f in files if (f.type or "").startswith("audio/")), None)
        if audio is not None:
            submit_voice(audio.getvalue(), audio.name, audio.type or "audio/wav", typed)
        else:
            pics = [f for f in files if (f.type or "").startswith("image/")]
            question = typed or ("What is in this image?" if pics else "")
            if question:
                st.session_state.pending_action = None
                st.session_state.pending_images = [
                    {"data": base64.b64encode(f.getvalue()).decode("ascii"),
                     "mime": f.type or "image/png"}
                    for f in pics
                ]
                add_message("user", question, {"images": len(pics)} if pics else None)
                if pics:
                    st.session_state.messages[-1]["image_bytes"] = [f.getvalue() for f in pics]
                st.session_state.last_question = question
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


# -- Inbox & Study ------------------------------------------------------------
with tab_inbox:
    st.markdown("#### 📥 Inbox")
    st.caption("What NEXUS did on its own: indexing new files, flashcards from your notes and "
               "the weekly digest. Everything here was made on this PC.")
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Check for changes now", width='stretch'):
        with st.spinner("Looking for new and changed files..."):
            the_brain.tick()
        st.rerun()
    if a2.button("Write the digest now", width='stretch'):
        with st.spinner("Writing the digest..."):
            the_brain.make_digest()
        st.rerun()
    if a3.button("Flashcards from all my notes", width='stretch'):
        with st.spinner("Writing and checking flashcards..."):
            the_brain.study_all()
        st.rerun()
    if a4.button("Mark all read", width='stretch', disabled=not unread):
        the_brain.mark_read()
        st.rerun()
    icons = {"indexed": "📚", "digest": "🗞️", "cards": "🃏", "error": "⚠️"}
    items = the_brain.inbox()
    if not items:
        st.info("Nothing yet. Add or edit a file in documents/ and NEXUS will notice within a minute.")
    for item in items:
        with st.container(border=True):
            when = datetime.fromtimestamp(item["created_at"]).strftime("%d %b %H:%M")
            st.markdown(f"{'🔵 ' if not item['read_at'] else ''}{icons.get(item['kind'], '•')} "
                        f"**{item['title']}** · {when}")
            if item["body"]:
                st.markdown(item["body"])

    st.divider()
    st.markdown("#### 🎓 Study")
    st.caption(f"{study['total']} flashcards · {study['verified']} verified against their source · "
               f"{study['due']} due now · {study['mastered']} mastered. Know it and a card comes "
               "back later (1, 3, 7, 14 days); miss it and it comes back soon.")
    due = the_brain.due_cards(limit=1)
    if not due:
        st.success("Nothing due. New notes become flashcards automatically.")
    else:
        card = due[0]
        with st.container(border=True):
            st.caption(("✓ verified against " if card["check_label"] == "supported"
                        else "unverified — not found word-for-word in ") + card["source"]
                       + f" · box {card['box']} of 5")
            st.markdown(f"### {card['question']}")
            if not st.session_state.card_revealed:
                if st.button("Show answer"):
                    st.session_state.card_revealed = True
                    st.rerun()
            else:
                st.markdown(card["answer"])
                st.caption(f"From {card['source']}: “{(card['evidence'] or '')[:300]}”")
                k1, k2 = st.columns(2)
                for col, label, knew in ((k1, "I knew it", True), (k2, "I didn't", False)):
                    if col.button(label, width='stretch', type="primary" if knew else "secondary"):
                        the_brain.review(card["id"], knew)
                        st.session_state.card_revealed = False
                        st.rerun()


# -- Leaderboard ------------------------------------------------------------
with tab_board:
    st.markdown("#### 🏆 Your model leaderboard")
    counts = feedback.summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Arena votes", counts["battles"])
    c2.metric("👍/👎 ratings", counts["ratings"])
    c3.metric("Corrections", counts["signals"])
    st.caption(
        "Elo ratings from your Arena votes (everyone starts at 1000; beating a stronger "
        "model gains more), plus how often you gave each model a 👍. A rating marked "
        "*settling* has fewer than 5 decided votes and can still move a lot. This is the "
        "data the learning router will train on."
    )
    board_task = st.selectbox("Task", ["All tasks", "general", "coding", "reasoning",
                                       "planning", "vision"])
    rows = feedback.leaderboard(None if board_task == "All tasks" else board_task)
    if rows:
        st.dataframe(
            [
                {
                    "model": ("💻 " if r["provider"] == "ollama" else "🌐 ") + r["model"],
                    "Elo": round(r["elo"]),
                    "": "" if r["settled"] else "settling",
                    "won": r["wins"], "lost": r["losses"], "tied": r["ties"],
                    "both bad": r["both_bad"],
                    "👍 approval": (f"{round(r['approval'] * 100)}% of "
                                   f"{r['thumbs_up'] + r['thumbs_down']}"
                                   if r["approval"] is not None else "—"),
                }
                for r in rows
            ],
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("No votes yet. Switch on ⚔️ Arena mode in the sidebar and compare a few answers.")

    st.divider()
    st.markdown("#### 🧠 Router")
    import learned_router
    from engine import get_router

    router_meta = learned_router.current_meta()
    in_use = get_router().learned
    st.caption(
        f"Routing now with **{'learned ' + in_use.version if in_use else 'keyword rules'}**. "
        "Retraining uses the seed prompts plus your corrections and Arena votes, takes "
        "seconds, and only switches if the new router measures at least as well on held-out prompts."
    )
    if router_meta:
        g = router_meta["metrics"]["golden"]
        st.dataframe(
            [
                {"router": name, "task accuracy": f"{r['task_accuracy']:.0%}",
                 "docs precision": f"{r['docs_precision']:.0%}",
                 "docs recall": f"{r['docs_recall']:.0%}",
                 "docs false alarms": r["docs_false_alarms"]}
                for name, r in (("keyword rules", g["rules"]),
                                (f"learned {router_meta['version']}", g["new"]))
            ],
            width='stretch', hide_index=True,
        )
        strong = router_meta["metrics"].get("strong")
        if strong:
            st.caption(f"Strong-vs-weak head (when to use cloud): AUC {strong['auc']} on "
                       f"{strong['held_out']} held-out votes — "
                       f"{'in use' if strong['passed'] else 'not in use (no better than random)'}.")
    if st.button("Retrain the router now"):
        from train.train_router import main as train_main

        with st.spinner("Training..."):
            report = train_main([])
        get_router().reload_learned()
        if report["made_current"]:
            st.success(f"{report['version']} measured at least as well and is now in use.")
        else:
            st.warning(f"{report['version']} measured worse than the router in use, so it was kept aside.")


# -- Diagnostics ------------------------------------------------------------
with tab_diag:
    st.markdown("#### Diagnostics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Models installed", len(installed))
    c2.metric("Loaded in VRAM", len(loaded))
    c3.metric("Indexed chunks", chunk_total)

    st.caption("Installed: " + (", ".join(installed) if installed else "none"))

    st.divider()
    st.markdown("##### Cloud providers")
    st.caption(
        f"Cloud mode: **{cloud_label}**. Keys come from `.env`; limits and models from "
        "`nexus.toml` and `cloud_models.toml`. Nothing here makes a network call "
        "except the button."
    )
    st.dataframe(
        [
            {
                "provider": cloud.PROVIDERS[p]["label"],
                "status": f"{STATUS_ICON.get(i['status'], '')} {i['status']}",
                "today": f"{i['used']}/{i['limit'] or '∞'}",
                "detail": i["detail"],
                "models": ", ".join(s.name for s in cloud.cloud_specs() if s.provider == p),
            }
            for p, i in cloud_status.items()
        ],
        width='stretch',
        hide_index=True,
    )
    if st.button("Check model names with each provider"):
        with st.spinner("Asking providers which models they serve..."):
            report = cloud.verify_models()
        if not report:
            st.info("No provider keys are set, so there is nothing to check.")
        for prov, r in report.items():
            label = cloud.PROVIDERS[prov]["label"]
            if r["error"]:
                st.error(f"{label}: {r['error']}")
            elif r["missing"]:
                st.warning(f"{label}: not served any more → {', '.join(r['missing'])}. "
                           "Update cloud_models.toml or nexus.toml.")
            else:
                st.success(f"{label}: all {len(r['ok'])} configured model(s) found.")

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
