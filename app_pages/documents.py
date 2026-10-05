"""Documents: what's indexed, adding files, and re-indexing."""

from datetime import datetime
from pathlib import Path

import streamlit as st

from nexus import config, ingest, ui
from nexus.retrieve import STAMP_NAME

st.markdown("#### Documents")
st.caption(f"Everything in `{config.DOCS_DIR}` is indexed locally and never leaves this PC. "
           "Re-indexing only re-reads files whose contents changed.")


def run_ingest(rebuild: bool = False) -> None:
    label = "Rebuilding the index…" if rebuild else "Indexing changes…"
    with st.status(label, expanded=False) as status:
        lines: list[str] = []
        try:
            report = ingest.run(force_rebuild=rebuild, echo=lines.append)
        except Exception as exc:
            st.code("\n".join(lines + [f"FAILED: {exc}"]), language=None)
            status.update(label="Indexing failed", state="error", expanded=True)
            return
        st.code("\n".join(lines) or "(no output)", language=None)
        if report.failed:
            status.update(label=f"Indexed, but {len(report.failed)} file(s) could not be read",
                          state="error", expanded=True)
        elif report.changed:
            status.update(label=f"Index updated — {report.total_chunks} chunks from "
                                f"{report.total_files} files", state="complete")
        else:
            status.update(label="Already up to date", state="complete")
    st.session_state.ingest_failed = report.failed


# -- summary ------------------------------------------------------------------
chunks, per_file = ui.index_summary()
on_disk = ingest.scan(config.DOCS_DIR) if config.DOCS_DIR.is_dir() else {}
stamp = config.INDEX_DIR / STAMP_NAME
last_indexed = datetime.fromtimestamp(stamp.stat().st_mtime).strftime("%d %b %H:%M") if stamp.exists() else "never"
pending = [rel for rel in on_disk if rel not in per_file]

with st.container(horizontal=True):
    st.metric("Files indexed", len(per_file), border=True)
    st.metric("Passages", chunks, border=True, help="~240-token chunks the search works over")
    st.metric("Waiting to index", len(pending), border=True)
    st.metric("Last indexed", last_indexed, border=True)

# -- add files ----------------------------------------------------------------
with st.form("add_documents", clear_on_submit=True, border=True):
    st.markdown("**Add documents**")
    uploads = st.file_uploader(
        "Files to add", type=sorted(ext.lstrip(".") for ext in ingest.SUPPORTED),
        accept_multiple_files=True, label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Add and index", icon=":material/upload:", type="primary")
if submitted:
    if not uploads:
        st.toast("Choose at least one file first.", icon=":material/info:")
    else:
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        replaced = []
        for f in uploads:
            target = config.DOCS_DIR / Path(f.name).name  # never a path from the browser
            if target.exists():
                replaced.append(target.name)
            target.write_bytes(f.getbuffer())
        st.toast(f"Saved {len(uploads)} file(s)" + (f", replacing {', '.join(replaced)}" if replaced else ""),
                 icon=":material/check:")
        run_ingest()

with st.container(horizontal=True, gap="small"):
    index_now = st.button("Index changes", icon=":material/sync:", key="index_changes",
                          help="Pick up files added, edited or deleted in the folder directly.")
    rebuild_now = st.button("Rebuild from scratch", icon=":material/restart_alt:", key="rebuild_index",
                            help="Re-read and re-embed every file. Only needed if the index looks wrong.")
if index_now or rebuild_now:
    run_ingest(rebuild=rebuild_now)

# -- table --------------------------------------------------------------------
chunks, per_file = ui.index_summary()
failed = st.session_state.get("ingest_failed") or {}
rows = []
for rel, path in sorted(on_disk.items()):
    try:
        stat = path.stat()
    except OSError:
        continue
    if rel in failed:
        state = "could not be read"
    elif rel in per_file:
        state = "indexed"
    else:
        state = "not indexed yet"
    rows.append({
        "file": rel,
        "type": path.suffix.lstrip(".").upper(),
        "passages": per_file.get(rel, 0),
        "size KB": round(stat.st_size / 1024, 1),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "status": state,
    })
for rel, count in per_file.items():
    if rel not in on_disk:  # deleted from the folder, still in the index
        rows.append({"file": rel, "type": "", "passages": count, "size KB": None,
                     "modified": None, "status": "deleted — index changes to remove"})

if rows:
    st.dataframe(
        rows, hide_index=True,
        column_config={
            "modified": st.column_config.DatetimeColumn("modified", format="D MMM YYYY, HH:mm"),
            "passages": st.column_config.NumberColumn("passages", help="Chunks in the index"),
        },
    )
else:
    st.info("No documents yet. Add .txt, .md, .pdf or .docx files above.", icon=":material/upload_file:")
if failed:
    with st.expander(f"{len(failed)} file(s) could not be read", icon=":material/error:"):
        for rel, err in failed.items():
            st.markdown(f"- `{rel}` — {err}")
