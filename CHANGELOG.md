# Changelog

Every released version of NEXUS, newest first. Planned versions are in
[ROADMAP.md](ROADMAP.md). Each heading below is a git tag (`v0.4.0`) and a
GitHub release.

## [1.2.0] - 2026-10-11

Cloud models tried with real keys for the first time, on Groq, Gemini and
OpenRouter.

### Checked for this release
- Against the real services: chat and streaming on Gemini, three Groq models
  and OpenRouter; voice input transcribed by Groq; an image read by Gemini, by
  OpenRouter's free models and, with cloud off, by a local vision model.
- The rules, end to end: easy questions stay local; questions about your
  documents stay on this PC and say so; a pinned cloud model is refused for a
  document question; with "Send documents to cloud" on, it is used; a rejected
  key is skipped, recorded, and not retried.
- Seen in practice: Gemini's free tier answered "high demand" on about one
  request in five. NEXUS used the next model and said so each time.
- Not checked: Mistral and DeepSeek. Those are still tested only against a
  stand-in.

### Added
- `openrouter/free`: OpenRouter's router for free models, which uses whichever
  one is available. Free, so it needs no "Allow paid models".
- Groq's `qwen/qwen3.8-27b`.

### Fixed
- A rejected Gemini key crashed the error reader: Gemini wraps its errors in a
  list and reports a bad key as 400, not 401. The key was never marked bad, so
  Gemini was retried on every question. It is now recognised, recorded, and
  skipped until the key changes.
- When every candidate was held back by a rule and no local model was
  running, NEXUS raised an error. It now explains what was held back and why.
- The paid OpenRouter model was offered in the model list with paid models
  switched off.

## [1.1.0] - 2026-10-10

The router learns from more data and from your corrections, and its scores
are kept honest.

### Added
- *Wrong task?* under an answer: say what a question really was. The
  correction is saved and used the next time the router is trained.
- 173 new hand-labelled training prompts (`train/data/curated_tasks.jsonl`),
  aimed at phrasings the router got wrong, each with a "needs your documents"
  label.
- The learned router also trains on the 276 labelled examples the rules use.
- The Leaderboard page shows what the router in use was trained on and its
  score on both held-out sets.

### Changed
- The gate a new router must pass now checks both held-out sets (100 and 150
  prompts), so a gain on one cannot hide a loss on the other.
- Learned router retrained: 100% on the 100-prompt set (was 99.0%), 98.7% on
  the 150-prompt set (the rules and examples: 89.3%), and it now recognises
  every question that needs your documents (recall 84.6% to 100%).

### Fixed
- Test prompts had leaked into training. 21 starter prompts and 9 examples
  were near-copies of held-out prompts (13 of the 150-prompt set had one),
  which flattered the earlier scores. They are removed; held-out prompts are
  now dropped from training data at training time, including your own
  corrections; and a test fails if a shipped training prompt is a close
  rewording of a test prompt.

## [1.0.0] - 2026-10-10

The first version called complete: a private assistant that answers from your
files with local models, shows how each answer was made, and checks itself.
Cloud models remain optional, off by default, and experimental: they are
tested against a stand-in for the providers, not yet with real keys.

### Added
- The routing log (`data/router_logs.jsonl`, on this PC) keeps the first 500
  characters of each question, and Diagnostics shows it, so a misrouted
  question can be found and turned into a labelled example.

### Changed
- The demo documents in `documents/` describe the current project. The
  retrieval and truth-check test sets were updated to match and re-measured:
  the full pipeline's context contains the answer for 16 of 18 questions at
  five passages and all 18 at the default ten; the truth check scores 83.3%
  given the passage and 76.7% finding its own evidence.
- Arena and Council say that no model is reachable when that is the problem,
  instead of asking for a second model.

### Checked for this release
- Arena and Council were run against real local models, not only in tests:
  two models answered and the vote updated the leaderboard; three models
  answered and a judge merged them.

## [0.5.0] - 2026-10-10

Feature merge: everything built on the `nexus-features` branch now runs on
`main`, on the `nexus` package, with main's security checks and context-window
handling kept. Cloud is off by default; with it off, nothing leaves this PC.

### Added
- Settings file: `nexus.toml` for behaviour and `.env` for cloud keys
  (examples included). Neither is needed to run.
- Optional cloud models (Gemini, Groq, OpenRouter, DeepSeek, Mistral) with
  three modes: off, hard questions only, allowed. Questions that use your
  documents and all PC actions stay on this PC unless you allow otherwise;
  earlier document-based answers are held back from cloud models too. Daily
  limits per provider; a failed provider is skipped and the answer says why.
- Image and voice input. Only image-reading models are considered when an
  image is attached; voice is transcribed first.
- Truth check: each sentence of an answer is marked supported, not found or
  contradicted against your documents. 85% accurate on 60 labelled claims
  when given the passage, 75% finding its own evidence.
- Ratings (thumbs up/down with a reason), Arena (two models answer with their
  names hidden and you pick) and a personal leaderboard with Elo ratings.
- Learned router: a small classifier trained on seed prompts and your
  corrections. It replaces the rules only if it measures at least as well on
  held-out prompts: 99.0% against 91.0% for the rules and examples.
- Model council: several models answer, a judge lists where they agree and
  disagree and writes one merged answer.
- Background brain: watches your documents folder, indexes changes, writes a
  weekly digest and makes flashcards that are checked against their source.
- Two new pages in the Streamlit UI, *Inbox & study* and *Leaderboard*; cloud
  status in Diagnostics; the same features in the dependency-free web UI.

### Changed
- One database for everything: existing `data/nexus.db` files are upgraded in
  place and keep their chats.
- The web server accepts up to 25 MB on the three routes that can carry an
  image or a recording; every other route stays at 1 MB.
- Conversation memory has a budget (8 messages, 8,000 characters, set in
  `nexus.toml`) applied before fitting to the model's context window.
- README rewritten for the current app; retrieval numbers re-measured.

### Fixed
- The learned router's gate scored it on "vision" and "speech" prompts that a
  text prompt is never routed to, which no router could get right. Those are
  now left out, so the comparison with the rules is like for like.

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
