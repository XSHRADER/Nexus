"""
app.py
Streamlit UI for NEXUS: `python run.py` (or `streamlit run app.py`).

The design goal is a glass box rather than a black box. NEXUS picks a model,
decides whether to consult your documents, and retrieves context -- all
automatically. The UI shows every one of those decisions under each answer,
and every one can be overridden from the chat's answer settings.

Each page lives in app_pages/ and only the open page runs, so chatting never
pays for the diagnostics or the retrieval lab.
"""

import streamlit as st

from nexus import log, ui

st.set_page_config(
    page_title="NEXUS",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)
log.setup()
ui.init_state()

# The background brain (folder watcher, digest, flashcards) starts with the
# app; its unread count rides on the Inbox entry so new items are noticed.
try:
    unread = ui.background_brain().unread()
except Exception:
    unread = 0

page = st.navigation(
    [
        st.Page("app_pages/chat.py", title="Chat", icon=":material/forum:", default=True),
        st.Page("app_pages/inbox.py", icon=":material/inbox:",
                title=f"Inbox & study ({unread})" if unread else "Inbox & study"),
        st.Page("app_pages/leaderboard.py", title="Leaderboard", icon=":material/leaderboard:"),
        st.Page("app_pages/documents.py", title="Documents", icon=":material/folder_open:"),
        st.Page("app_pages/lab.py", title="Retrieval lab", icon=":material/science:"),
        st.Page("app_pages/diagnostics.py", title="Diagnostics", icon=":material/monitoring:"),
    ]
)
page.run()
# After the page, so the page draws first; the status needs the index and
# Ollama, which can take a few seconds on the very first load.
with st.sidebar:
    ui.render_status()
