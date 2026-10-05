"""Chat: ask questions, see how each answer was made, confirm PC actions."""

import logging
from dataclasses import replace
from datetime import datetime

import streamlit as st

from nexus import config, store, ui
from nexus.engine import answer, apply_pending
from nexus.router import TaskRouter

logger = logging.getLogger("nexus.ui.chat")
ss = st.session_state

USER_AVATAR = ":material/person:"
NEXUS_AVATAR = ":material/hub:"
# question -> icon shown on its suggestion chip
SUGGESTIONS = {
    "What does this project use to store embeddings?": ":material/description:",
    "Compare hybrid retrieval with dense-only search": ":material/compare_arrows:",
    "Find large files in my Downloads folder": ":material/folder_open:",
}
META_KEYS = (
    "model", "task", "needs_rag", "rag_reason", "sources", "elapsed", "chain", "complexity",
    "attempts", "prompt_chars", "info", "auto_task", "thinking", "truncated", "chunks_dropped",
    "metrics",
)

avail = ui.availability()
installed = avail.get("ollama", [])


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------


def persist(role: str, content: str, meta: dict | None = None) -> None:
    """Add a message to the transcript and save it. Saving never blocks chatting."""
    message = {"role": role, "content": content}
    if meta is not None:
        message["meta"] = meta
    ss.messages.append(message)
    try:
        if ss.chat_id is None:
            ss.chat_id = store.create_chat(content)
        store.append_message(ss.chat_id, role, content, meta)
    except Exception as exc:
        logger.warning("could not save message: %s", exc)


def history_for(regenerate: bool) -> list[dict]:
    """The turns before the question being answered, without UI metadata.

    For a regenerate, the answer being redone and its original question are
    left out too, so the model never sees the answer it's asked to replace.
    """
    msgs = ss.messages
    prior = msgs[:-1] if msgs and msgs[-1]["role"] == "user" else list(msgs)
    if (regenerate and len(prior) >= 2
            and prior[-1]["role"] == "assistant" and prior[-2]["role"] == "user"):
        prior = prior[:-2]
    return [{"role": m["role"], "content": m["content"]} for m in prior]


def save_stopped(messages, chat_id, buffer, thoughts, forced_model) -> None:
    """Keep what streamed before Stop was clicked.

    Runs while Streamlit unwinds the interrupted script, and while a stop is
    pending every st.* call re-raises it — *including reading
    st.session_state*. So this only touches objects captured before streaming
    began: `messages` is the session's own transcript list, mutated in place.
    """
    partial = "".join(buffer).strip()
    if not partial and not thoughts:
        return
    content = (partial or "_(stopped before the answer began)_") + "\n\n_— stopped_"
    meta = {"stopped": True, "thinking": "".join(thoughts) or None, "forced_model": forced_model}
    messages.append({"role": "assistant", "content": content, "meta": meta})
    try:
        if chat_id:
            store.append_message(chat_id, "assistant", content, meta)
        store.record_turn({"chat_id": chat_id, "stopped": True})
    except Exception as exc:
        logger.warning("could not save the stopped answer: %s", exc)


def transcript_markdown() -> str:
    lines = [f"# NEXUS chat — {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in ss.messages:
        who = "You" if m["role"] == "user" else f"NEXUS ({(m.get('meta') or {}).get('model') or 'pc-toolkit'})"
        lines += [f"**{who}:**", "", m["content"], ""]
        for i, s in enumerate((m.get("meta") or {}).get("sources") or [], 1):
            lines.append(f"> [{i}] {s.get('source')}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One answer
# ---------------------------------------------------------------------------


def run_turn(question: str, opts, regenerate: bool = False) -> None:
    """Generate one answer, streaming it (and any reasoning) into the transcript."""
    history = history_for(regenerate)
    with st.chat_message("assistant", avatar=NEXUS_AVATAR):
        stop_slot = st.empty()
        stop_slot.button("Stop", key="stop_generation", icon=":material/stop_circle:",
                         help="Stop this answer. What's been written so far is kept.")
        status_slot = st.empty()
        think_slot = st.empty()
        placeholder = st.empty()
        buffer: list[str] = []
        thoughts: list[str] = []

        def on_status(text: str) -> None:
            if not buffer and not thoughts:
                with status_slot.container():
                    st.status(text, state="running")

        def on_thinking(text: str) -> None:
            thoughts.append(text)
            status_slot.empty()
            with think_slot.container(), st.expander("Reasoning…", expanded=True,
                                                     icon=":material/psychology:"):
                st.markdown("".join(thoughts))

        def on_token(tok: str) -> None:
            if not buffer:
                status_slot.empty()
            buffer.append(tok)
            placeholder.markdown("".join(buffer) + "▌")

        on_status("Reading your question…")
        # Captured now: once Stop is clicked, session_state can't be read.
        messages, chat_id = ss.messages, ss.chat_id
        finished = False
        try:
            result = answer(
                question, options=opts, on_token=on_token, on_thinking=on_thinking,
                history=history, chat_id=chat_id, on_status=on_status,
            )
            finished = True
        except Exception as exc:
            finished = True
            for slot in (stop_slot, status_slot, think_slot, placeholder):
                slot.empty()
            st.error(f"NEXUS could not answer: {exc}", icon=":material/error:")
            st.caption("Check that Ollama is running — the Diagnostics page shows what NEXUS can reach.")
            return
        finally:
            # Clicking Stop makes Streamlit raise its (BaseException) stop
            # signal inside a callback; that closes the Ollama stream on its
            # way out, and the next script run shows what was saved here.
            if not finished:
                save_stopped(messages, chat_id, buffer, thoughts, bool(opts.force_model))
        stop_slot.empty()
        status_slot.empty()

        meta = {k: result.get(k) for k in META_KEYS}
        meta["forced_model"] = bool(opts.force_model)
        think_slot.empty()
        placeholder.empty()
        ui.render_message({"content": result.get("answer", ""), "meta": meta})

    persist("assistant", result.get("answer", ""), meta)
    if result.get("requires_confirmation"):
        ss.pending_action = result["pending"]


# ---------------------------------------------------------------------------
# Sidebar: chats
# ---------------------------------------------------------------------------


def open_chat(chat_id: str) -> None:
    ui.reset_chat(chat_id, store.load_messages(chat_id))


def delete_chat(chat_id: str) -> None:
    store.delete_chat(chat_id)
    if ss.chat_id == chat_id:
        ui.reset_chat()
    ss.confirm_delete = None


with st.sidebar:
    st.button("New chat", icon=":material/add:", type="primary", width="stretch",
              on_click=ui.reset_chat, key="new_chat")
    search = st.text_input("Search chats", placeholder="Search chats", key="chat_search",
                           icon=":material/search:", label_visibility="collapsed")
    try:
        chats = store.list_chats(limit=30, search=search)
    except Exception as exc:
        chats = []
        st.caption(f"Saved chats unavailable ({exc})")
    if not chats:
        st.caption("No matching chats." if search else "Your chats are saved here.")
    for chat in chats:
        title = chat["title"] if len(chat["title"]) <= 34 else chat["title"][:33] + "…"
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.button(title, key=f"open_{chat['id']}", help=f"{chat['title']} · {ui.ago(chat['updated'])}",
                      type="secondary" if chat["id"] == ss.chat_id else "tertiary",
                      width="stretch", on_click=open_chat, args=(chat["id"],))
            st.button(":material/delete:", key=f"del_{chat['id']}", help="Delete this chat",
                      type="tertiary", on_click=lambda cid=chat["id"]: ss.update(confirm_delete=cid))
        if ss.confirm_delete == chat["id"]:
            with st.container(border=True):
                st.caption("Delete this chat? This can't be undone.")
                with st.container(horizontal=True, gap="small"):
                    st.button("Delete", key="confirm_delete_chat", type="primary",
                              on_click=delete_chat, args=(chat["id"],))
                    st.button("Keep", key="cancel_delete_chat",
                              on_click=lambda: ss.update(confirm_delete=None))


# ---------------------------------------------------------------------------
# Header: title, settings, export
# ---------------------------------------------------------------------------


def reset_options() -> None:
    for key, value in ui.DEFAULTS.items():
        if key.startswith("opt_"):
            ss[key] = value


if ss.opt_model != "Auto" and ss.opt_model not in installed:
    ss.opt_model = "Auto"  # the pinned model was removed or Ollama is down

title = next((c["title"] for c in chats if c["id"] == ss.chat_id), None) if ss.chat_id else None
with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown(f"#### {title or 'New chat'}", width="stretch")
    with st.popover("Answer settings", icon=":material/tune:"):
        st.caption("Auto everywhere reproduces NEXUS's own choices; each answer shows what it chose.")
        st.selectbox("Model", ["Auto", *installed], key="opt_model", persist_state="session",
                     help="Auto scores every installed model against the task, how hard the prompt "
                          "looks, and what is already in memory. Pin one to override it.")
        st.selectbox("Task", ["Auto", *TaskRouter.TASKS], key="opt_task", persist_state="session",
                     help="Override the intent classifier if it reads a question wrong.")
        st.segmented_control("Use your documents", ["Auto", "Always", "Never"], key="opt_docs",
                             required=True, persist_state="session",
                             help="Auto searches your documents and uses them when they look relevant. "
                                  "Always forces it; Never answers from the model alone.")
        st.slider("Retrieval depth", 1, 20, key="opt_top_k", persist_state="session",
                  disabled=ss.opt_docs == "Never",
                  help="How many ~240-token passages to put in front of the model.")
        st.toggle("Rerank with cross-encoder", key="opt_rerank", persist_state="session",
                  disabled=ss.opt_docs == "Never",
                  help="Re-scores candidates against the question. Slower, sharper — and "
                       "Auto needs it to judge relevance.")
        st.slider("Temperature", 0.0, 1.5, step=0.1, key="opt_temperature", persist_state="session",
                  help="0 is deterministic and factual; higher wanders more.")
        st.button("Reset to automatic", icon=":material/restart_alt:", on_click=reset_options,
                  disabled=not ui.overrides(), key="reset_options")
    if ss.messages:
        st.download_button("Export", transcript_markdown(), icon=":material/download:",
                           file_name=f"nexus-chat-{datetime.now():%Y%m%d-%H%M}.md",
                           mime="text/markdown", key="export_chat")

if ui.overrides():
    st.markdown(" ".join(f":orange-badge[{ui.badge_text(o)}]" for o in ui.overrides()))

options = ui.current_options()


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

if not ss.messages:
    with st.container(horizontal_alignment="center"):
        st.space("large")
        st.markdown("## What do you want to know?", text_alignment="center")
        st.caption("NEXUS picks the model that suits each question and pulls in your documents "
                   "when they help. Everything runs on this PC.", text_alignment="center")
        chunks, _ = ui.index_summary()
        if not installed:
            st.warning("Ollama isn't reachable, so only folder actions work right now. "
                       f"Start it with `ollama serve`, then `ollama pull {config.DEFAULT_PULL_MODEL}`.",
                       icon=":material/power_off:")
        elif not chunks:
            st.info("No documents are indexed yet — add some so NEXUS can answer from them.",
                    icon=":material/upload_file:")
            st.page_link("app_pages/documents.py", label="Add documents", icon=":material/folder_open:")
        picked = st.pills("Try asking", list(SUGGESTIONS), key="suggestion",
                          format_func=lambda q: f"{SUGGESTIONS[q]} {q}", label_visibility="collapsed")
        if picked:
            ss.last_question = picked
            persist("user", ss.last_question)
            del ss["suggestion"]
            st.rerun()

for message in ss.messages:
    avatar = USER_AVATAR if message["role"] == "user" else NEXUS_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        ui.render_message(message)

# A queued question (from a suggestion or a regenerate click).
queued = ss.last_question
if queued and (not ss.messages or ss.messages[-1]["role"] == "user"):
    ss.last_question = None
    turn_options = options
    alt = ss.regenerate_with
    if alt:
        ss.regenerate_with = None
        turn_options = replace(options, force_model=alt)
    run_turn(queued, turn_options, regenerate=bool(alt))
    st.rerun()

if ss.pending_action:
    pending = ss.pending_action
    with st.chat_message("assistant", avatar=NEXUS_AVATAR), st.container(border=True):
        st.markdown(f":material/pending_actions: **Waiting for your confirmation** — "
                    f"`{pending['op']}` in `{pending['path']}`")
        st.caption("Nothing has changed on disk yet.")
        with st.container(horizontal=True, gap="small"):
            if st.button("Apply changes", type="primary", icon=":material/check:", key="apply_pending"):
                try:
                    msg = apply_pending(pending)["answer"]
                except Exception as exc:
                    msg = f"⚠️ The action failed: {exc}"
                persist("assistant", msg)
                ss.pending_action = None
                st.rerun()
            if st.button("Cancel", icon=":material/close:", key="cancel_pending"):
                persist("assistant", "Cancelled — nothing on disk changed.")
                ss.pending_action = None
                st.rerun()

# Offer a re-run on a different model, using the chain NEXUS already scored.
last = ss.messages[-1] if ss.messages else None
if last and last["role"] == "assistant" and last.get("meta") and not ss.pending_action:
    used = last["meta"].get("model")
    alts = [c["model"] for c in last["meta"].get("chain") or [] if c["model"] != used][:3]
    question = next((m["content"] for m in reversed(ss.messages) if m["role"] == "user"), None)
    if alts and question:
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.caption("Answer again with", width="content")
            for alt in alts:
                if st.button(alt, key=f"regen_{alt}", icon=":material/refresh:", type="tertiary"):
                    persist("user", question)
                    ss.last_question = question
                    ss.regenerate_with = alt
                    st.rerun()

prompt = st.chat_input("Ask about your documents — or anything else")
if prompt:
    ss.pending_action = None
    persist("user", prompt)
    ss.last_question = prompt
    st.rerun()
