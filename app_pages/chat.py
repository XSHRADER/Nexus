"""Chat: ask questions, see how each answer was made, confirm PC actions.

Also where the ways of answering live: one model (the default), Arena (two
models, names hidden until you vote) and Council (several models and a judge).
"""

import base64
import logging
from dataclasses import replace
from datetime import datetime

import streamlit as st

from nexus import cloud, config, feedback, store, ui
from nexus.engine import (
    answer,
    apply_pending,
    arena,
    council,
    council_message_meta,
    council_recommended,
    message_meta,
    transcribe,
)
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
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "gif"]
AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "webm", "flac"]

avail = ui.availability()
installed = avail.get("ollama", [])
cloud_status = avail.get("cloud") or {}
cloud_models = ui.ready_cloud_models(avail)


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------


def persist(role: str, content: str, meta: dict | None = None) -> dict:
    """Add a message to the transcript and save it. Saving never blocks chatting.

    The saved message's id is kept on it: ratings and truth checks attach to it.
    """
    message = {"role": role, "content": content}
    if meta is not None:
        message["meta"] = meta
    ss.messages.append(message)
    try:
        if ss.chat_id is None:
            ss.chat_id = store.create_chat(content)
        message["id"] = store.append_message(ss.chat_id, role, content, meta)
    except Exception as exc:
        logger.warning("could not save message: %s", exc)
    return message


def history_for(regenerate: bool) -> list[dict]:
    """The turns before the question being answered, without UI metadata.

    For a regenerate, the answer being redone and its original question are
    left out too, so the model never sees the answer it's asked to replace.
    An answer that drew on your documents keeps that one fact with it: such
    turns are held back from cloud models unless you allow documents out.
    """
    msgs = ss.messages
    prior = msgs[:-1] if msgs and msgs[-1]["role"] == "user" else list(msgs)
    if (regenerate and len(prior) >= 2
            and prior[-1]["role"] == "assistant" and prior[-2]["role"] == "user"):
        prior = prior[:-2]
    history = []
    for m in prior:
        turn = {"role": m["role"], "content": m["content"]}
        meta = m.get("meta") or {}
        if m["role"] == "assistant" and (meta.get("needs_rag") or meta.get("sources")):
            turn["meta"] = {"needs_rag": True}
        history.append(turn)
    return history


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


def run_turn(question: str, opts, regenerate: bool = False, images: list | None = None) -> None:
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
        attachments = {"images": images} if images else {}
        finished = False
        try:
            result = answer(
                question, options=opts, on_token=on_token, on_thinking=on_thinking,
                history=history, chat_id=chat_id, on_status=on_status, **attachments,
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

        meta = message_meta(result)
        meta["forced_model"] = bool(opts.force_model)
        think_slot.empty()
        placeholder.empty()
        ui.render_message({"role": "assistant", "content": result.get("answer", ""), "meta": meta})

        message = persist("assistant", result.get("answer", ""), meta)
        # Answers built from your documents are checked straight away; others
        # get a button, since most of what they say won't be in your files.
        if result.get("task") != "system_agent" and (meta.get("needs_rag") or meta.get("sources")):
            try:
                with st.spinner("Checking each sentence against your documents…"):
                    ui.run_truth(message)
            except Exception as exc:
                logger.warning("truth check failed: %s", exc)
    if result.get("requires_confirmation"):
        ss.pending_action = result["pending"]


def run_council(question: str, opts, images: list, auto: bool = False) -> None:
    """Several models answer and a judge merges them into one saved answer."""
    with st.chat_message("assistant", avatar=NEXUS_AVATAR), \
            st.status("The council is answering — several models, then a judge…",
                      state="running") as box:
        out = council(question, options=opts, history=history_for(False), images=images,
                      chat_id=ss.chat_id, on_status=box.write)
    if out.get("error"):
        persist("assistant", f"Council: {out['error']}")
        return
    persist("assistant", out["answer"], council_message_meta(out, auto=auto))


def run_arena(question: str, opts, images: list) -> None:
    """Two models answer blind; the pair waits in `ss.battle` for your vote."""
    with st.chat_message("assistant", avatar=NEXUS_AVATAR), \
            st.status("Two models are answering, one after the other…", state="running") as box:
        out = arena(question, options=opts, history=history_for(False), images=images,
                    chat_id=ss.chat_id, on_status=box.write)
    if out.get("error"):
        persist("assistant", f"Arena: {out['error']}")
        return
    ss.battle = out


def render_battle(battle: dict) -> None:
    """Blind A/B answers and the vote; the model names appear only after voting."""
    with st.container(border=True):
        st.markdown(":material/swords: **Arena — which answer is better?**")
        if battle.get("note"):
            st.caption(battle["note"])
        for col, side in zip(st.columns(2), ("a", "b")):
            with col.container(border=True):
                st.markdown(f"**Answer {side.upper()}**")
                st.markdown(battle[side]["answer"])
        choice = None
        with st.container(horizontal=True, gap="small"):
            for label, value, kind in (("A is better", "a", "primary"), ("B is better", "b", "primary"),
                                       ("Tie", "tie", "secondary"), ("Both bad", "both_bad", "secondary")):
                if st.button(label, key=f"vote_{value}", type=kind):
                    choice = value
    if choice is None:
        return
    decided = feedback.vote(battle["battle_id"], choice)
    ss.battle = None
    text, meta = feedback.battle_message(decided)
    reveal = f"A was **{decided['model_a']}**, B was **{decided['model_b']}**."
    if choice == "both_bad":
        meta["info"] = f"Arena: you marked both answers as bad. {reveal}"
    else:
        side = "b" if choice == "b" else "a"
        picked = "a tie" if choice == "tie" else f"answer {side.upper()}"
        meta.update(sources=battle[side].get("sources"), needs_rag=battle[side].get("needs_rag"),
                    info=f"Arena: you picked {picked}. {reveal}")
    persist("assistant", text, meta)
    st.rerun()


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


if ss.opt_model != "Auto" and ss.opt_model not in [*installed, *cloud_models]:
    ss.opt_model = "Auto"  # the pinned model was removed, or its provider went away

title = next((c["title"] for c in chats if c["id"] == ss.chat_id), None) if ss.chat_id else None
with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown(f"#### {title or 'New chat'}", width="stretch")
    with st.popover("Answer settings", icon=":material/tune:"):
        st.caption("Auto everywhere reproduces NEXUS's own choices; each answer shows what it chose.")
        st.segmented_control("Answer with", list(ui.MODES), key="opt_mode", required=True,
                             persist_state="session",
                             help="One model: the usual way. Arena: two models answer with their "
                                  "names hidden and you pick the better one. Council: several "
                                  "models answer and a judge merges them (slower; for hard questions).")
        st.selectbox("Model", ["Auto", *installed, *cloud_models], key="opt_model",
                     persist_state="session",
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
        st.divider()
        st.segmented_control("Cloud models", list(ui.CLOUD_CHOICES), key="opt_cloud", required=True,
                             persist_state="session",
                             help="Off: nothing leaves this PC. Hard questions: cloud models are "
                                  "used for hard prompts and for things no local model can do. "
                                  "Allowed: cloud competes on every prompt, but easy ones still "
                                  "go local first.")
        cloud_off = ui.cloud_mode() == "off"
        st.toggle("Send documents to cloud", key="opt_allow_docs", persist_state="session",
                  disabled=cloud_off,
                  help="Off: questions that use your documents are always answered on this PC.")
        st.toggle("Allow paid models", key="opt_allow_paid", persist_state="session",
                  disabled=cloud_off, help="Some cloud models are billed per token with no free tier.")
        if not cloud_off:
            for prov, info in cloud_status.items():
                st.caption(f"{ui.STATUS_ICON.get(info['status'], ':gray[○]')} "
                           f"**{cloud.PROVIDERS[prov]['label']}** · {info['detail']}")
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
        where = ("Everything runs on this PC." if ui.cloud_mode() == "off"
                 else "Cloud models are switched on; your documents and PC actions stay on this "
                      "PC unless you allow otherwise.")
        st.caption("NEXUS picks the model that suits each question and pulls in your documents "
                   f"when they help. {where}", text_alignment="center")
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

# A queued question (from the chat box, a suggestion or a regenerate click).
queued = ss.last_question
if queued and (not ss.messages or ss.messages[-1]["role"] == "user"):
    ss.last_question = None
    turn_options = options
    alt = ss.regenerate_with
    if alt:
        ss.regenerate_with = None
        turn_options = replace(options, force_model=alt)
    images, ss.pending_images = ss.pending_images, []
    if ss.opt_mode == "Council" and not alt:
        run_council(queued, turn_options, images)
    elif ss.opt_mode == "Arena" and not alt:
        run_arena(queued, turn_options, images)
    elif not alt and not images and council_recommended(queued, turn_options):
        run_council(queued, turn_options, images, auto=True)
    else:
        run_turn(queued, turn_options, regenerate=bool(alt), images=images)
    st.rerun()

if ss.battle:
    render_battle(ss.battle)

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

last = ss.messages[-1] if ss.messages else None
answered = bool(last and last["role"] == "assistant" and last.get("meta")
                and not ss.pending_action and not ss.battle)

# Check an answer that didn't use your documents against them anyway.
if (answered and ui.rateable(last) and not last["meta"].get("truth")
        and st.button("Check this answer against my files", icon=":material/fact_check:",
                      type="tertiary", key="truth_last")):
    with st.spinner("Checking each sentence against your documents…"):
        ui.run_truth(last)
    st.rerun()

# Offer a re-run on a different model, using the chain NEXUS already scored.
if answered:
    used = last["meta"].get("model")
    alts = [c["model"] for c in last["meta"].get("chain") or [] if c["model"] != used][:3]
    question = next((m["content"] for m in reversed(ss.messages) if m["role"] == "user"), None)
    if alts and question:
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.caption("Answer again with", width="content")
            for alt in alts:
                if st.button(alt, key=f"regen_{alt}", icon=":material/refresh:", type="tertiary"):
                    try:  # asking another model is a quiet thumbs-down for this one
                        feedback.record_signal("regenerate", question, task=last["meta"].get("task"),
                                               model=used, value=alt, message_id=last.get("id"))
                    except Exception as exc:
                        logger.warning("could not record the regenerate signal: %s", exc)
                    persist("user", question)
                    ss.last_question = question
                    ss.regenerate_with = alt
                    st.rerun()


def submit_voice(audio_bytes: bytes, name: str, mime: str, typed: str = "") -> None:
    """Voice becomes text first; the transcript is then asked like any question."""
    try:
        with st.spinner("Transcribing…"):
            heard = transcribe(audio_bytes, name, mime, options)
    except RuntimeError as exc:
        st.error(str(exc), icon=":material/mic_off:")
        return
    text = f"{typed}\n{heard['text']}".strip() if typed else heard["text"]
    ss.pending_action = None
    persist("user", text, {"voice": True})
    ss.last_question = text
    st.rerun()


with st.expander("Ask by voice", icon=":material/mic:"):
    recording = st.audio_input("Record a question", key=f"audio_{ss.audio_widget}")
    if recording is not None:
        ss.audio_widget += 1  # a fresh recorder next run
        submit_voice(recording.getvalue(), "recording.wav", "audio/wav")

submitted = st.chat_input("Ask about your documents — or anything else",
                          accept_file=True, file_type=IMAGE_TYPES + AUDIO_TYPES)
if submitted:
    typed = (submitted.text or "").strip()
    files = list(submitted.files or [])
    audio = next((f for f in files if (f.type or "").startswith("audio/")), None)
    if audio is not None:
        submit_voice(audio.getvalue(), audio.name, audio.type or "audio/wav", typed)
    else:
        pics = [f for f in files if (f.type or "").startswith("image/")]
        prompt = typed or ("What is in this image?" if pics else "")
        if prompt:
            ss.pending_action = None
            ss.battle = None
            ss.pending_images = [
                {"data": base64.b64encode(f.getvalue()).decode("ascii"), "mime": f.type or "image/png"}
                for f in pics
            ]
            message = persist("user", prompt, {"images": len(pics)} if pics else None)
            if pics:
                message["image_bytes"] = [f.getvalue() for f in pics]
            ss.last_question = prompt
            st.rerun()
