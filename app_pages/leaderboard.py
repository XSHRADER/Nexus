"""Leaderboard: which models you prefer, and the router that learns from it."""

import streamlit as st

from nexus import feedback, learned_router, providers
from nexus.engine import get_router

st.markdown("#### Your model leaderboard")
counts = feedback.summary()
with st.container(horizontal=True):
    st.metric("Arena votes", counts["battles"], border=True)
    st.metric("Ratings", counts["ratings"], border=True, help="Thumbs up or down on an answer.")
    st.metric("Corrections", counts["signals"], border=True,
              help="Times you asked another model, changed the task, or forced documents on or off.")
st.caption(
    "Elo ratings from your Arena votes (everyone starts at 1000; beating a stronger model gains "
    "more), plus how often you gave each model a thumbs up. A rating marked *settling* has fewer "
    f"than {feedback.SETTLED_AFTER} decided votes and can still move a lot."
)

task = st.selectbox("Task", ["All tasks", *providers.TEXT_TASKS, "vision"], key="board_task")
rows = feedback.leaderboard(None if task == "All tasks" else task)
if rows:
    st.dataframe(
        [
            {
                "model": r["model"],
                "runs on": "this PC" if r["provider"] == "ollama" else "cloud",
                "Elo": round(r["elo"]),
                "status": "" if r["settled"] else "settling",
                "won": r["wins"], "lost": r["losses"], "tied": r["ties"],
                "both bad": r["both_bad"],
                "thumbs up": (f"{round(r['approval'] * 100)}% of {r['thumbs_up'] + r['thumbs_down']}"
                              if r["approval"] is not None else "—"),
            }
            for r in rows
        ],
        hide_index=True,
    )
else:
    st.info("No votes yet. In Chat, open Answer settings, choose **Arena**, and compare a few answers.",
            icon=":material/swords:")

st.divider()
st.markdown("#### Router")
meta = learned_router.current_meta()
in_use = get_router().learned
st.caption(
    f"Questions are being routed by **{'the learned router ' + in_use.version if in_use else 'rules and examples'}**. "
    "Retraining uses the seed prompts plus your corrections and Arena votes, takes seconds, and "
    "the new router is only switched on if it measures at least as well on held-out prompts."
)
if meta:
    golden = meta["metrics"]["golden"]
    st.dataframe(
        [
            {"router": name, "task accuracy": f"{r['task_accuracy']:.0%}",
             "documents: precision": f"{r['docs_precision']:.0%}",
             "documents: recall": f"{r['docs_recall']:.0%}",
             "false alarms": r["docs_false_alarms"]}
            for name, r in (("rules and examples", golden["rules"]),
                            (f"learned {meta['version']}", golden["new"]))
        ],
        hide_index=True,
    )
    strong = meta["metrics"].get("strong")
    if strong:
        st.caption(f"Strong-or-weak head (when to use cloud): AUC {strong['auc']} on "
                   f"{strong['held_out']} held-out votes — "
                   f"{'in use' if strong['passed'] else 'not in use (no better than random)'}.")
else:
    st.caption("No router has been trained yet.")

if st.button("Retrain the router now", icon=":material/model_training:", key="train_router"):
    from train.train_router import main as train_main

    try:
        with st.spinner("Training…"):
            report = train_main([])
    except Exception as exc:
        st.error(f"Training failed: {exc}", icon=":material/error:")
    else:
        get_router().reload_learned()
        if report["made_current"]:
            st.success(f"{report['version']} measured at least as well and is now in use.",
                       icon=":material/check_circle:")
        else:
            st.warning(f"{report['version']} measured worse than the router in use, so it was set aside.",
                       icon=":material/warning:")
