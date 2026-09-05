# Foundation Action Log

This file records setup and foundation actions in chronological order.

1. Inspected the RAG module files and README.
2. Confirmed Python 3.13.1 is available.
3. Confirmed the pinned dependencies were not installed in the base interpreter.
4. Made project paths resolve from the `claude` module directory, so commands work from any current directory.
5. Made nested document paths unique in the hash cache, Chroma metadata, and chunk IDs.
6. Created the `documents/` and `vector_store/` runtime directories.
7. Created the isolated `nexus-env` virtual environment.
8. Installed the pinned dependencies from `requirements.txt` into `nexus-env`.
9. Validated the modules with Python bytecode compilation.
10. Found the original dependency pins incompatible with Python 3.13.1: `tiktoken==0.7.0` has no compatible wheel and the old Chroma stack requires a source NumPy build.
11. Updated Chroma, sentence-transformers, and tiktoken to Python 3.13-compatible releases while keeping the module APIs unchanged.
12. Confirmed all dependencies import successfully and downloaded the embedding model during the ingestion smoke test.
13. Fixed ingestion bookkeeping so placeholder files such as `.gitkeep` are ignored instead of being treated as documents.
14. Final validation passed: all modules compiled, dependencies imported, and empty ingestion exited with code 0.
15. Checked the optional generation runtime: Ollama is not installed, so `rag_pipeline.py` will remain unavailable until Ollama is installed and a model is pulled.
12. Final validation: all dependencies imported successfully; ingest.py smoke test exited with code 0.
16. Confirmed Ollama 0.33.1 is reachable and `llama3.1:8b` is installed.
17. Added a startup preflight to `rag_pipeline.py` for Ollama availability and exact model-tag validation.
18. Added `local_ai_models.txt`, mapping the embedding and generation models to their NEXUS tasks.

## Optimization pass (2026-09-05)

19. Removed the `langchain_text_splitters` dependency (it was imported by
    `loaders.py` but never installed, so `ingest.py` crashed on start).
    Replaced it with a dependency-free recursive character splitter with
    the same signature.
20. Added `embeddings.py`: `lru_cache`-backed loaders for the
    sentence-transformer and cross-encoder. `router.py`, `retrieve.py`,
    and `ingest.py` now share one in-memory copy of each model instead of
    loading their own.
21. `router.py` now scores queries against the exemplar vectors that were
    already pre-embedded in `__init__` (previously `_semantic_scores`
    re-embedded all 12 exemplars on every call and the cached vectors
    were unused). Cached the Ollama `/api/tags` result for 5s.
22. `retrieve.py` loads the cross-encoder lazily (only when there are
    candidates to re-rank).
23. `rag_pipeline.py` caches the `Retriever` (`get_retriever()`), so the
    Chroma client, models, and BM25 index are built once, not per query.
24. Added `engine.py`: single `answer()` dispatch shared by `app.py` and
    `server.py`, replacing ~40 lines of drifted copy-paste. Added a
    Gemini fallback to the RAG path (it previously just raised when
    Ollama was down).
25. `ingest.py` now removes Chroma chunks for documents that were deleted
    since the last run.
26. `pc_tools.py` scans now skip `nexus-env`, `vector_store`, `.git`,
    `__pycache__`, `node_modules`, etc. A "find duplicates" scan of the
    project dropped from ~250s to <0.01s.
27. Added root `requirements.txt` (README Phase 1 expects it there);
    dropped the unused `tiktoken` pin, added `streamlit`.
28. Fixed `.vscode/tasks.json` (interpreter path pointed at a
    non-existent `claude\` folder); added ingest + test tasks and
    `.vscode/settings.json` to auto-select `nexus-env` in VS Code.
29. `.gitignore`: added `.venv/`, `router_logs.jsonl`, `*.log`.
30. Validation: all modules compile, all 5 router tests pass, `ingest.py`
    runs clean, engine routes offline paths, retriever + model caches
    verified reused across calls.

## Local file-operation agent (2026-09-05)

31. `pc_tools.py`: `organize_folder` hardened -- refuses drive roots / home
    / system dirs, skips dotfiles, never overwrites (collision-safe
    rename), and writes `.nexus_organize_manifest.json` on apply. Added
    `undo_last_organize` (reverses via manifest) and `delete_empty_dirs`.
32. New `pc_agent.py`: parses a request into an operation
    (organize / undo / empty_dirs / duplicates / large / analyze) plus a
    target folder (known names like Downloads/Desktop, a quoted path, a
    drive path, or the project dir). Read-only ops run immediately;
    organize + empty-dir removal return a preview and a `pending` action.
33. `engine.answer(question, base_dir=None, confirm=False)` now routes the
    system agent through `pc_agent`; added `engine.apply_pending(pending)`
    to execute an approved action. Every `answer()` result now carries
    `requires_confirmation` and `pending`.
34. `app.py`: shows the plan, then **Apply changes** / **Cancel** buttons;
    a new prompt cancels any un-applied action. (Dropped the per-message
    "Routing details" expander -- decisions are still in `router_logs.jsonl`.)
35. `server.py` + `ui/index.html`: `/api/chat` accepts `confirm`; the
    browser UI renders an "Apply changes" button when one is needed.
36. `router.py`: `sort / tidy / arrange / declutter / undo / empty folders`
    now score as system-agent intents, with matching exemplars.
37. Added `tests/test_pc_agent.py` (13 tests: intent parsing, path
    resolution, preview -> apply -> undo, safety refusal). Full suite: 18 pass.

## Automatic model selection (no user choice)

38. New `providers.py`: capability catalogue (`ModelSpec` per model: task
    strengths 0-1, quality, speed, local/cloud) plus the selector.
    `estimate_complexity()` scores prompt difficulty from length, planning
    language, and multi-part structure. `plan()` returns an **ordered chain**
    of reachable models, scored as
    `fit + complexity*quality*0.6 + (1-complexity)*speed*0.35 + local bonus`.
39. Local models get a `LOCAL_BONUS` (free/private/fast) and an extra
    `RAG_LOCAL_BONUS` so personal documents never leave the PC by default --
    **except** for `planning` and `reasoning` (`CLOUD_FAVOURED_TASKS`), where
    the bonus is dropped so `gemini-2.5-pro` wins on merit. This is the
    "use Gemini when it's better at planning" requirement.
40. `router.py`: added a **`planning`** task (regex + exemplars), replaced
    `_select_model()` with `providers.plan()`, and `route()` now returns
    `chain`, `provider`, `complexity`, and `available` alongside `model`.
    `MODEL_MAP` is now derived from the catalogue (one place to add a model).
41. `router.py`: added a `general` prior + greeting rule -- "hi" was being
    handed to a vision model by embedding noise. Added a location rule
    (`desktop|downloads|my pc|...`) so "analyze my desktop" hits the toolkit
    while "what is a desktop environment in linux" stays general.
42. `engine.py`: walks the chain top-down, first model that answers wins;
    records `attempts`, reports auto-switches in `info`, and returns a plain
    setup message instead of raising when nothing is reachable. RAG context is
    built once via new `rag_pipeline.build_rag_prompt()` and reused for
    whichever provider answers (previously retrieval was Ollama-only).
43. `gemini_client.py`: key is saved to `.gemini_key` (gitignored) and loaded
    automatically -- entered once, not every run.
44. `pc_agent.py`: known folders now resolve through OneDrive
    (`~/OneDrive/Documents`) -- `~/Documents` doesn't exist on this machine, so
    "analyze my documents folder" was erroring.
45. `app.py` sidebar rewritten: no model picker anywhere. Shows what's
    reachable, the task -> AI mapping, and a per-answer badge of which AI was
    auto-picked. `ui/index.html` shows the same via new `GET /api/status`.
46. Tests: new `tests/test_providers.py` (13) + 6 new router tests covering
    planning->Gemini, fallback without a key, Ollama-down->cloud, greeting
    routing, and the empty chain. Full suite: **37 pass**.
47. VRAM-aware selection: `router.get_loaded_ollama_models()` reads
    `/api/ps`, and `providers.WARM_BONUS` (0.10) favours a model already
    resident. An 8GB card holds one 7B model, so a swap costs ~30s; the bonus
    breaks near-ties without ever displacing a specialist (verified: a warm
    `mistral` does not steal a coding task from `qwen2.5-coder`). Measured
    live: back-to-back coding queries went 22.7s -> 11.9s on reuse.
48. Live verification with Ollama running (11 models): coding ->
    `qwen2.5-coder:7b`, general+RAG -> `llama3.1:8b`, planning ->
    `deepseek-r1:7b` (correct, no Gemini key set). 40 tests pass.
