"""Diagnostics: what NEXUS can reach, how answers have performed, and why."""

import json

import streamlit as st

from nexus import cloud, config, ingest, providers, store, ui

TAIL_BYTES = 64 * 1024


def tail_lines(path, n: int) -> list[str]:
    """Last `n` lines without reading a multi-MB log into memory."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - TAIL_BYTES))
        return fh.read().decode("utf-8", errors="replace").splitlines()[-n:]


with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("#### Diagnostics", width="stretch")
    if st.button("Refresh", icon=":material/refresh:", key="refresh_diag"):
        ui.availability.clear()

avail = ui.availability()
installed, loaded = avail.get("ollama", []), avail.get("loaded", [])
chunks, per_file = ui.index_summary()

with st.container(horizontal=True):
    st.metric("Ollama", "online" if installed else "offline", border=True,
              help=f"Reached at {config.OLLAMA_URL}")
    st.metric("Models installed", len(installed), border=True)
    st.metric("In memory", len(loaded), border=True, help=", ".join(loaded) or "none")
    st.metric("Indexed passages", chunks, border=True, help=f"{len(per_file)} files")

if not installed:
    st.warning(f"Ollama isn't answering at `{config.OLLAMA_URL}`. Start it with `ollama serve`, "
               "or set `NEXUS_OLLAMA_URL` if it runs elsewhere.", icon=":material/power_off:")

# -- cloud --------------------------------------------------------------------
st.markdown("##### Cloud providers")
cloud_status = avail.get("cloud") or {}
st.caption(
    f"Cloud is **{st.session_state.opt_cloud.lower()}** (change it in Chat → Answer settings). "
    "Keys come from `.env`; limits and extra models from `nexus.toml`; the model list from "
    "`cloud_models.toml`. Nothing on this page makes a network call except the button."
)
if cloud_status:
    st.dataframe(
        [
            {
                "provider": cloud.PROVIDERS[p]["label"],
                "status": info["status"].replace("_", " "),
                "today": f"{info['used']}/{info['limit'] or '∞'}",
                "detail": info["detail"],
                "models": ", ".join(spec.name for spec in cloud.cloud_specs() if spec.provider == p),
            }
            for p, info in cloud_status.items()
        ],
        hide_index=True,
    )
if st.button("Check model names with each provider", icon=":material/rule:", key="verify_cloud"):
    with st.spinner("Asking providers which models they serve…"):
        report = cloud.verify_models()
    if not report:
        st.info("No provider keys are set, so there is nothing to check.", icon=":material/key_off:")
    for prov, r in report.items():
        label = cloud.PROVIDERS[prov]["label"]
        if r["error"]:
            st.error(f"{label}: {r['error']}", icon=":material/error:")
        elif r["missing"]:
            st.warning(f"{label}: not served any more → {', '.join(r['missing'])}. "
                       "Update `cloud_models.toml` or `nexus.toml`.", icon=":material/warning:")
        else:
            st.success(f"{label}: all {len(r['ok'])} configured model(s) found.",
                       icon=":material/check_circle:")

# -- performance --------------------------------------------------------------
st.markdown("##### Models")
st.caption("Over the last 500 answers. Failure rate counts every attempt, including ones the next "
           "model in the chain recovered from.")
try:
    stats = store.model_stats()
except Exception as exc:
    stats = []
    st.caption(f"Metrics unavailable ({exc})")
if stats:
    st.dataframe(
        [{"model": s["model"], "answers": s["answers"], "median s": ui.secs(s["median_ms"]),
          "median tok/s": s["median_tokens_per_s"], "failure rate": s["failure_rate"],
          "cold loads": s["cold_loads"]} for s in stats],
        hide_index=True,
        column_config={"failure rate": st.column_config.ProgressColumn(
            "failure rate", min_value=0.0, max_value=1.0, format="percent")},
    )
else:
    st.caption("No answers recorded yet.")

st.markdown("##### Recent answers")
try:
    turns = store.recent_turns(25)
except Exception:
    turns = []
if turns:
    st.dataframe([ui.turn_row(t) for t in turns], hide_index=True)
else:
    st.caption("No answers recorded yet.")

st.markdown("##### Recent routing decisions")
if config.ROUTER_LOG.exists():
    rows = []
    for line in reversed(tail_lines(config.ROUTER_LOG, 20)):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({
            "task": d.get("task"), "model": d.get("model"), "difficulty": d.get("complexity"),
            "keyword says docs": d.get("needs_rag"),
            "chain": " → ".join(c["model"] for c in (d.get("chain") or [])[:3]),
        })
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.caption("No decisions logged yet.")
else:
    st.caption("No routing log yet — ask something first.")

# -- models ------------------------------------------------------------------
discovered = providers.discover(installed)
with st.expander(f"Models outside the catalogue ({len(discovered)})", icon=":material/travel_explore:"):
    if discovered:
        st.caption("Profiled from name, size and reported capabilities, then scaled to 90% so a "
                   "hand-tuned entry wins ties.")
        st.dataframe(
            [{"model": spec.name,
              "profile": ", ".join(f"{t} {f:.2f}" for t, f in sorted(spec.strengths.items(),
                                                                     key=lambda kv: -kv[1])),
              "capabilities": ", ".join(sorted(providers.capabilities(spec.name)["caps"])) or "unknown"}
             for spec in discovered],
            hide_index=True,
        )
    else:
        st.caption("Every installed model is in the catalogue.")

with st.expander("Configuration", icon=":material/settings:"):
    st.json({
        "ollama_url": config.OLLAMA_URL,
        "context_window": config.NUM_CTX,
        "documents": str(config.DOCS_DIR),
        "index": str(config.INDEX_DIR),
        "database": str(config.db_path()),
        "settings_file": str(config.TOML_FILE) + ("" if config.TOML_FILE.exists() else " (not present: defaults)"),
        "trained_routers": str(config.MODELS_DIR),
        "log_file": str(config.APP_LOG),
        "index_settings": ingest.index_config(),
    })
    st.caption("Override the paths with the NEXUS_* environment variables listed in "
               "`nexus/config.py`; behaviour (memory, cloud, truth check, router, council, "
               "background brain) lives in `nexus.toml` — copy `nexus.toml.example` to start.")
