# Changelog

Every released version of NEXUS, newest first. Planned versions are in
[ROADMAP.md](ROADMAP.md). Each heading below is a git tag (`v0.4.0`) and a
GitHub release.

## [0.4.0] - 2026-10-10

Router accuracy and version labelling.

### Added
- Router training data: 285 labelled example prompts in
  `nexus/router_examples.json` (was 17 hard-coded). Add a line to teach the
  router a new phrasing.
- Held-out router test set `eval/router_set.json` (150 prompts) and
  `python -m nexus.evaluate_router` to score it.
- Version labelling: `nexus.__version__`, `python -m nexus --version`, this
  changelog, the roadmap, and a GitHub workflow that publishes a release when
  a version tag is pushed.

### Changed
- Router scoring uses the mean of the 6 closest examples instead of the
  single best match, and weighs examples more against keyword rules.
  Accuracy on unseen prompts went from 62.0% to 89.3% (reasoning 23% to 87%,
  planning 43% to 87%, coding 60% to 90%).

### Fixed
- A PC-tools test failed on Windows CI because it compared the short and
  long spelling of the same temp folder.

## [0.3.0] - 2026-10-06

Restructure, terminal chat and a new UI.

### Added
- Terminal chat: `python -m nexus`.
- Multipage dark-themed UI (Chat, Documents, Retrieval lab, Diagnostics) with
  live progress before the first word, per-answer details (model, speed,
  sources), chat search, and export as Markdown.
- Relevance check: questions that do not name your documents are still
  checked against the index and grounded when a good match exists.
- Numbered sources, so answers cite `[1]..[n]` and the UI links them.
- DOCX tables are indexed.
- CI runs tests on Ubuntu and Windows, plus a lint job.

### Changed
- All modules moved into a `nexus` package, with one config module (paths,
  URLs, model names, environment overrides) and one Ollama client.
- Indexing reads only files that changed, and retries files that failed.

### Fixed
- Two retrievers per UI process, one of which went stale after indexing.
- Text prompts could be routed to vision or speech models.
- PC tools: empty-folder cleanup entered `.git` and virtual environments;
  undo could overwrite a newer file; a failure part-way lost the undo record.
- The test suite wrote into the user's real `data/` folder.

## [0.2.0] - 2026-09-20

Hardening: security, chat history and streaming.

### Added
- Saved chats, messages and per-answer metrics in SQLite.
- Conversation history is sent to the model, with every prompt budgeted to
  the model's context window.
- Streamed answers with a Stop button that keeps the partial answer.
- Automatic model choice includes installed models outside the catalogue.
- The UI shows the model's reasoning and timing for each answer.
- The router log rotates at 5 MB.

### Fixed
- Security: another website open in the browser could trigger file
  operations through the local server.
- Oversized requests are drained before being rejected.

## [0.1.0] - 2026-09-06

First version: a fully local document assistant.

### Added
- Document search over PDF, DOCX, TXT and Markdown: Chroma vector search plus
  BM25 keyword search, fused and re-ranked with a cross-encoder.
- Task router that picks a local Ollama model per question.
- Streamlit UI that shows every automatic decision and lets you override it,
  plus a dependency-free HTTP UI.
- PC tools: organize a folder, find duplicates and large files, undo.
- Measured retrieval quality (`--eval`) on a fixed question set.
- One-command start (`run.py`, `start.bat`), CI and an MIT license.

### Removed
- The Gemini cloud provider. NEXUS runs entirely on this PC.
