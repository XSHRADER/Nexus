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
