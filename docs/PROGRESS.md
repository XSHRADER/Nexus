# Nexus optimization — progress log

Branch: `nexus-optimization` (from `phase1-hardening`). Started 2026-10-06.
Scope: Nexus only. No code, data, API or dependency touches any other project.

Baseline: 133 tests passing (8 suites), ruff clean apart from style nits.

## Plan

**Phase 1 — core**
1. Central config (`config.py`): one place for paths, Ollama URL, model
   names, context size; env-var overrides. Today the same constants are
   duplicated across 6 modules.
2. One Ollama client module: today HTTP calls to Ollama live in 5 files.
3. Retrieval: one shared retriever (app.py and engine.py each built their
   own), refresh on index *content* change, survive a full rebuild.
4. Ingest: only read changed files, don't cache files that failed to load,
   rebuild through Chroma instead of deleting files under a live client.
5. Routing: stop sending text-only prompts to vision/speech models; dedupe
   the model chain; stricter tag matching.
6. PC tools: empty-dir deletion ignored its own skip list; undo could
   overwrite files; a mid-run failure lost the undo manifest.
7. Logging via `logging` instead of scattered `print(..., file=sys.stderr)`.

**Phase 2 — GUI**: dark theme, layout, status, loading/error states.

**Phase 3 — cleanup & pipeline**: package layout, dead files, pyproject,
lint + test in CI, one test command, docs.

## Log

(newest last)

### 1. Package layout (commit "Move the modules into a nexus package")
Pure move of the 13 top-level modules into `nexus/`; entry points `app.py`
and `run.py` stay at the root. One test command now:
`python -m unittest discover -s tests` (40 s vs ~95 s for 8 separate runs).

### 2. Phase 1 core
**Architecture**
- `nexus/config.py`: every path, URL and model name, with env overrides
  (`NEXUS_OLLAMA_URL`/`OLLAMA_HOST`, `NEXUS_NUM_CTX`, `NEXUS_DATA_DIR`, ...).
  Replaced constants duplicated across 6 modules.
- `nexus/ollama.py`: the only module that talks to Ollama (was 5). Listing
  results are cached, *including* "unreachable", so a stopped daemon costs
  one refused connection per TTL instead of one per UI rerun. HTTP errors now
  carry Ollama's own message ("model 'x' not found"), not just "HTTP 404".
- Default Ollama URL is `127.0.0.1`, not `localhost`: on Windows `localhost`
  can resolve to `::1` first while Ollama listens on IPv4 only.
- `nexus/log.py`: rotating log file at `data/nexus.log`; warnings to stderr.
  `print(..., file=sys.stderr)` calls replaced. Generated files
  (`router_logs.jsonl`, legacy eval index) moved under `data/`.
- `prompts.py` (was `rag_pipeline.py`) reduced to prompt building; its
  duplicate generation path (`/api/generate`, hard-coded model) is gone.

**Bugs fixed (each has a regression test that fails on the old code)**
- Two retrievers per Streamlit process (app's and engine's): double memory,
  and only one was refreshed after indexing. Now one shared `get_retriever()`.
- Retriever only reloaded when the *chunk count* changed: an edited document
  with the same count kept serving old text from BM25.
- "Full rebuild" deleted the collection under a running UI; every later
  query failed until restart. Rebuild now goes through Chroma, and the
  retriever re-opens the collection when the index stamp changes.
- Ingest read and chunked *every* document on every run, then discarded all
  but the changed ones. Now only changed files are read.
- A document that failed to load was recorded as indexed and never retried.
- Router sent text-only prompts to vision/speech models ("explain the chart
  in my notes" -> llava, which had no image). Speech models (`whisper:*`)
  aren't Ollama chat models at all. Text routing now only offers text models.
- Two catalogue entries could resolve to the same installed tag, so the
  fallback chain retried a failed model. Chain is now de-duplicated.
- `llama3.1:8b` matched an installed `llama3.1:70b` (different speed
  profile). Size-aware tag matching; quantised tags of the same size match.
- RAG keyword gate matched substrings ("profile" -> "file").
- `delete_empty_dirs` pruned ignored folders in a bottom-up walk, which has
  no effect: empty folders inside `.git`, venvs and `node_modules` were
  deleted. Its preview also under-reported nested empty folders.
- Undo of "organize" could overwrite a newer file with the same name
  (`shutil.move` replaces on Windows). It now refuses and keeps the entry.
- A failure part-way through "organize" lost the manifest, so the moves
  already made could not be undone. A second organize overwrote the first
  run's manifest. Manifest now records every run; undo pops the latest.
- A missing folder in a PC request crashed the chat turn; now a message.

**Improvements**
- Relevance probe: in Auto mode, questions that don't name the documents are
  still checked against the index; they are grounded only if the best
  cross-encoder score >= -3.0. Calibrated on the golden set: 15/18 document
  questions pass, 0/15 off-topic ones do (best off-topic -4.2).
- Sources are numbered `[1]..[n]` in the prompt so answers cite them inline
  and the UI can match citations to the sources list.
- DOCX tables are now indexed (were silently skipped); text files lose BOMs.
- Duplicate finder compares 64 KB prefixes before hashing whole files.
- PC results render as tables/lists instead of raw JSON dumps.
- `run.py`: the end-to-end generation test (~30 s cold) runs with `--check`
  only, so normal startup is faster. New `--test` runs lint + tests.
- BM25 top-n by `argpartition` instead of a full sort.
- `pyproject.toml` with ruff config; codebase lint-clean.

Eval on the real index after the changes (golden set, 18 questions):
hybrid + cross-encoder recall@5 1.000, MRR 0.724, grounded@5 0.944.

Tests: 133 -> 164, all passing. New: `test_ingest.py`, `test_ollama.py`.

### 3. Terminal chat
`python -m nexus` replaces the loop in the old `rag_pipeline.py`, which
skipped routing and always called one hard-coded model. Consoles print UTF-8
(Windows defaults to cp1252 and crashed on emoji).

### 4. Phase 2 — GUI
**Structure**: `app.py` is now a 40-line entry point using `st.navigation`;
pages live in `app_pages/` (Chat, Documents, Retrieval lab, Diagnostics) and
shared rendering in `nexus/ui.py`. Before, the four views were tabs, and
Streamlit computes every tab on every rerun -- each chat message also
re-ran the diagnostics queries and, if a lab question was typed, four
retrievals.

**Theme**: `.streamlit/config.toml` dark theme (indigo accent, ~5:1 contrast
for button text), system fonts only so the UI works offline, no custom CSS.
Server bound to 127.0.0.1 in config too, so a bare `streamlit run app.py`
can't expose the file tools to the network. The plain HTML UI uses the same
palette.

**UX changes**
- Live progress before the first token ("Searching your documents…",
  "Loading llama3.1:8b — the first answer from a model takes longer…") via a
  new `on_status` engine callback. Previously a 30 s cold load showed nothing.
- Answer settings moved from the sidebar into a popover; any non-automatic
  setting shows as a badge above the chat, with "Reset to automatic".
  Settings now survive switching pages (`persist_state="session"`).
- Sidebar: New chat, chat search (titles and message text), per-chat delete
  with inline confirm, status block (Ollama, index, models in memory).
- Per answer: native badges, "How this was answered" with first-token time,
  tokens/s, prompt size, model load time and the scored model chain; numbered
  sources matching the `[n]` citations.
- Empty state with suggestion chips and guidance when Ollama is down or
  nothing is indexed (links to Documents).
- Documents: metrics row, files table with per-file status (indexed / not
  yet / could not be read / deleted), "Add and index" in one step (the old
  uploader rewrote every file on every rerun), progress in `st.status`.
- Export as Markdown with sources.

**Bugs found while testing in the browser**
- Document excerpts rendered as Markdown: a line of `===` turned the line
  above into a page-wide heading. Excerpts are now one escaped line.
- Status block above the page delayed the whole page on first load (models
  loading); moved below the page content.

Tests: 171 (new app tests: every page renders, settings survive page switch,
suggestion chips, chat search, file actions wait for confirmation; store
search; engine progress callback).

### 5. Phase 3 — cleanup and pipeline (in progress)
Done:
- Removed `program_info/` (byte-identical copy of files in `documents/`),
  `install_streamlit.bat` (hard-coded a path that no longer exists) and
  `documents/requirements.txt` (a drifting mirror). `ACTION_LOG.md` moved to
  `docs/`. Stale root `__pycache__/` deleted; the old root routing log was
  kept as `data/router_logs.before-move.jsonl`.
- **Bug**: the test suite wrote routing decisions and logs into the user's
  real `data/` folder. `tests/isolate.py` now points every path at a temp
  dir, and a test guards it.
- `numpy` declared (imported directly); Python minimum is now 3.11 (numpy 2.4).
  `requirements-dev.txt` adds ruff.
- CI: separate lint job; tests run with one command on Ubuntu *and* Windows.

Remaining after 1.0.0:
- Cloud providers have only been exercised against `demos/mock_cloud.py`. Try
  each with a real key (version 1.1.0).
- `data/router_logs.jsonl` still contains decisions from earlier test runs
  (model names `a`/`b`); safe to delete.

### Router training data (2026-10-08)
- Exemplars moved out of code into `nexus/router_examples.json`: 17 -> 285
  labelled prompts across the five tasks. Add a line there to teach a phrasing.
- New held-out set `eval/router_set.json` (150 prompts, 30 per task) and
  `python -m nexus.evaluate_router`. Exemplars within 0.8 cosine of an eval
  prompt were reworded or dropped so the score isn't inflated.
- Semantic score is now the mean of the top 6 exemplar matches (was the single
  best), weighted 8x (was 4x); `general`'s +1.2 head start removed. Chosen by
  leave-one-out accuracy over the exemplars, not on the eval set.
- Held-out accuracy 62.0% -> 89.3% (reasoning 23% -> 87%, planning 43% -> 87%,
  coding 60% -> 90%). A test keeps it >= 85% and keeps the two sets disjoint.

### Feature merge, version 0.5.0 (2026-10-10)
`nexus-features` (Phases 0-6, written on the September flat layout) merged
into `main`. Not a textual merge: both sides had rewritten `engine`, `server`,
`store`, `config`, `router`, `providers` and the Streamlit app.

- **Kept from main**: the `nexus` package, the server's same-origin and
  pending-id checks, context-window fitting with numbered sources, the
  relevance probe, progress callbacks, per-answer metrics, the page-per-screen
  UI, test isolation.
- **Added from the feature branch**: `cloud`, `cloud_client`, `speech`,
  `truth_check`, `feedback`, `learned_router`, `council`, `brain`, the
  `train/` scripts, the mock Ollama and cloud servers the tests run against.
- **Reconciled**: one `config.py` (environment paths plus `nexus.toml`/`.env`
  settings); one `store.py` (schema v2 adds nine tables, v1 files upgrade in
  place; chat ids stay text); `providers.plan()` takes cloud policy, images
  and Arena bonuses as keywords; the engine walks one chain of local and
  cloud candidates with the privacy check at call time.
- **Dropped as superseded**: the feature branch's own chat store and message
  shapes (main's are newer), its `confirm` flag on file actions (replaced by
  main's preview-then-apply), `demos/phase0_demo.py`.
- **Bug found**: the learned router's gate counted 40 "vision"/"speech"
  prompts that text is never routed to. Fixed; rules 91.0%, learned 99.0% on
  the remaining 100.
- Tests 179 -> 328, lint clean. Checked in the browser against real Ollama:
  a document question (truth check 100%, NLI), every page, and the plain web UI.

### Version 1.0.0 (2026-10-10)
- `documents/` refreshed to describe the current project: the user guide,
  project info and file layout rewritten; stale lines in the model map and
  checklist corrected; later milestones appended to the action log.
- Test sets updated to match: q07 and q15 in `eval/golden_set.json`, seven
  claims in `eval/claims_golden.json`. Re-measured: hybrid + cross-encoder
  grounded@5 0.889 (q05, q13 miss), 1.000 at top_k=10; truth check 0.833
  given passage, 0.767 end to end.
- Routing log keeps the first 500 characters of the question.
- Arena and Council run live against real Ollama models (they had only been
  run in tests). Found on the way: with Ollama down they asked for "another
  model" instead of saying nothing was reachable. Fixed.
- Tests 328 -> 329.

### Version 1.1.0 (2026-10-10)
- Found on measuring the learned router on `eval/router_set.json` for the
  first time: 13 of its 150 prompts had a near-copy (cosine > 0.85) in
  `seed_tasks.jsonl`, several word for word. Removed 21 seed rows and 9
  examples that were within 0.85 of a prompt in either held-out set.
- Training now drops held-out prompts whatever their source (exact match after
  normalising), and a test checks shipped data for near-copies by embedding.
- Fed: `train/data/curated_tasks.jsonl` (173 rows, task + needs_docs) and the
  276 router examples (task only) as training rows.
- Gate extended to the 150-prompt set. Router v2: golden 1.000 task, docs
  P 0.929 / R 1.000; wide set 0.987 (rules 0.893; v1 0.973 with the leak,
  0.971 on the 137 clean prompts).
- UI: "Wrong task?" under the last answer records a task_override signal.
- Tests 329 -> 335.

