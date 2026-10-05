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
