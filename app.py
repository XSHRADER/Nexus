import os
from pathlib import Path

import streamlit as st

import providers
from engine import answer, apply_pending, get_router
from gemini_client import load_saved_key, save_key

PROJECT_DIR = Path(__file__).resolve().parent
LOG_PATH = PROJECT_DIR / "router_logs.jsonl"

st.set_page_config(page_title="NEXUS AI", page_icon="🧠", layout="wide")

router = get_router()

with st.sidebar:
    st.header("🤖 Automatic AI selection")
    st.write(
        "You don't pick a model. NEXUS reads each message, decides what kind of "
        "task it is, and sends it to whichever AI is strongest at that — local "
        "Ollama for quick chat, code, and anything using your own documents; "
        "**Gemini for planning and multi-step reasoning**. If the first choice "
        "is down it silently moves to the next one."
    )

    avail = providers.availability()
    st.markdown("### What's reachable now")
    if avail["ollama"]:
        st.success(f"Local (Ollama): {len(avail['ollama'])} model(s)")
        st.caption(", ".join(avail["ollama"]))
    else:
        st.warning("Local (Ollama): not running — start it with `ollama serve`")
    if avail["gemini"]:
        st.success("Cloud (Gemini): connected")
    else:
        st.warning("Cloud (Gemini): no API key — planning tasks will stay local")

    with st.expander("🔑 Gemini API key (saved once)"):
        saved = load_saved_key() or ""
        key_input = st.text_input(
            "Google Gemini API Key",
            value=saved,
            type="password",
            help="From Google AI Studio. Stored in .gemini_key so you only enter it once.",
        )
        if key_input != saved:
            save_key(key_input)
            os.environ["GEMINI_API_KEY"] = key_input
            st.rerun()
        if saved:
            os.environ["GEMINI_API_KEY"] = saved

    st.markdown("### Which AI handles what")
    st.table(
        [
            {"Task": "Planning / roadmaps", "Preferred": "Gemini 2.5 Pro"},
            {"Task": "Deep reasoning", "Preferred": "Gemini 2.5 Pro → deepseek-r1"},
            {"Task": "Coding", "Preferred": "qwen2.5-coder (local) → Gemini"},
            {"Task": "Everyday chat", "Preferred": "llama3.1 (local)"},
            {"Task": "Your documents", "Preferred": "local models (stays private)"},
            {"Task": "Images", "Preferred": "qwen2.5vl / Gemini"},
            {"Task": "PC folder actions", "Preferred": "built-in toolkit, no model"},
        ]
    )

    st.caption(
        "PC automation can **sort a folder** into category subfolders, remove empty folders, "
        "find duplicates/large files, or analyze usage. Try: _\"sort my Downloads folder\"_. "
        "File-moving actions show a preview and wait for you to confirm."
    )
    if LOG_PATH.exists():
        st.caption(f"Routing log: {LOG_PATH.name}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything — I'll pick the right AI for it myself. Coding, planning, your project documents, or PC folder cleanup."}
    ]
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# A local file operation is waiting for the user's go-ahead.
if st.session_state.pending_action:
    pending = st.session_state.pending_action
    with st.chat_message("assistant"):
        st.warning(f"Waiting for confirmation: **{pending['op']}** on `{pending['path']}`")
        col_apply, col_cancel = st.columns(2)
        if col_apply.button("✅ Apply changes", type="primary", use_container_width=True):
            try:
                res = apply_pending(pending)
                msg = res["answer"]
            except Exception as exc:
                msg = f"Action failed: {exc}"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.pending_action = None
            st.rerun()
        if col_cancel.button("❌ Cancel", use_container_width=True):
            st.session_state.messages.append({"role": "assistant", "content": "Cancelled — nothing on disk was changed."})
            st.session_state.pending_action = None
            st.rerun()

prompt = st.chat_input("Ask NEXUS AI...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # A new instruction supersedes any un-applied action.
    st.session_state.pending_action = None

    try:
        with st.spinner("Choosing the best AI for this..."):
            result = answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        badge = f"🤖 `{result['model']}` · {result['task']}"
        if result.get("needs_rag"):
            badge += " · used your documents"
        st.session_state.messages.append({"role": "assistant", "content": badge})

        if result.get("info"):
            st.session_state.messages.append({"role": "assistant", "content": f"ℹ️ {result['info']}"})
        if result.get("requires_confirmation"):
            st.session_state.pending_action = result["pending"]
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "content": f"NEXUS could not answer: {exc}"})

    st.rerun()
