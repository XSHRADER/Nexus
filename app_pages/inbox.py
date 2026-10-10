"""Inbox and study: what NEXUS did on its own, and flashcards from your notes."""

from datetime import datetime

import streamlit as st

from nexus import config, ui

ss = st.session_state
ICONS = {"indexed": ":material/library_books:", "digest": ":material/newspaper:",
         "cards": ":material/style:", "error": ":material/warning:"}

the_brain = ui.background_brain()
unread = the_brain.unread()
study = the_brain.study_stats()


def run(label: str, action) -> None:
    """Run one piece of background work now, with the failure shown, not raised."""
    try:
        with st.spinner(label):
            action()
    except Exception as exc:
        st.error(f"That didn't work: {exc}", icon=":material/error:")
        return
    st.rerun()


st.markdown("#### Inbox")
watching = "watching your documents folder" if config.get_settings().brain_enabled else \
    "the background watcher is switched off in nexus.toml, so nothing runs by itself"
st.caption("What NEXUS did on its own: indexing new files, flashcards from your notes and the "
           f"weekly digest. Everything here was made on this PC. Right now: {watching}.")

with st.container(horizontal=True, gap="small"):
    if st.button("Check for changes now", icon=":material/sync:", key="brain_tick"):
        run("Looking for new and changed files…", the_brain.tick)
    if st.button("Write the digest now", icon=":material/newspaper:", key="brain_digest"):
        run("Writing the digest…", the_brain.make_digest)
    if st.button("Flashcards from all my notes", icon=":material/style:", key="brain_cards"):
        run("Writing and checking flashcards…", the_brain.study_all)
    if st.button("Mark all read", icon=":material/done_all:", key="brain_read", disabled=not unread):
        the_brain.mark_read()
        st.rerun()

items = the_brain.inbox()
if not items:
    st.info("Nothing yet. Add or edit a file in your documents folder and NEXUS will notice "
            "within a minute.", icon=":material/inbox:")
for item in items:
    with st.container(border=True):
        when = datetime.fromtimestamp(item["created_at"]).strftime("%d %b %H:%M")
        new = ":blue-badge[new] " if not item["read_at"] else ""
        st.markdown(f"{new}{ICONS.get(item['kind'], ':material/circle:')} **{item['title']}** "
                    f":gray[· {when}]")
        if item["body"]:
            st.markdown(item["body"])

st.divider()
st.markdown("#### Study")
with st.container(horizontal=True):
    st.metric("Flashcards", study["total"], border=True)
    st.metric("Verified", study["verified"], border=True,
              help="The answer was found in the note it came from.")
    st.metric("Due now", study["due"], border=True)
    st.metric("Mastered", study["mastered"], border=True)
st.caption("Know a card and it comes back later (after 1, 3, 7, then 14 days); miss it and it "
           "comes back soon.")

due = the_brain.due_cards(limit=1)
if not due:
    st.success("Nothing due. New notes become flashcards automatically.", icon=":material/check_circle:")
else:
    card = due[0]
    with st.container(border=True):
        checked = ("verified against " if card["check_label"] == "supported"
                   else "unverified — not found word-for-word in ")
        st.caption(f"{checked}`{card['source']}` · box {card['box']} of 5")
        st.markdown(f"### {card['question']}")
        if not ss.card_revealed:
            if st.button("Show answer", key="card_show", type="primary"):
                ss.card_revealed = True
                st.rerun()
        else:
            st.markdown(card["answer"])
            st.caption(f"From `{card['source']}`: “{ui.excerpt(card['evidence'] or '', 300)}”")
            with st.container(horizontal=True, gap="small"):
                for label, knew in (("I knew it", True), ("I didn't", False)):
                    if st.button(label, key=f"card_{'yes' if knew else 'no'}",
                                 type="primary" if knew else "secondary"):
                        the_brain.review(card["id"], knew)
                        ss.card_revealed = False
                        st.rerun()
