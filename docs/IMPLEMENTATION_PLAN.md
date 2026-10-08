# NEXUS Implementation Plan

Goal: turn NEXUS from "a front end for Ollama" into **a hybrid local + cloud AI
that checks its own answers and learns which model is best for you**.

Ollama stays the local engine, but it becomes one part among many. The parts
that make NEXUS unique all live in NEXUS's own code:

```
 question ──► learned router ──► model(s) ──► truth check ──► answer + 👍/👎
                 ▲   local or cloud?   │                            │
                 │                     └─ council on hard prompts   │
                 └──────────── feedback + public data train it ◄────┘
```

---

## Overview

| Phase | What it adds | Size | Depends on |
|---|---|---|---|
| 0 | Foundations: chat memory, SQLite store, config + `.env` | M | — |
| 1 | Cloud providers (Gemini, Groq, OpenRouter, DeepSeek, Mistral) + image and voice input | L | 0 |
| 2 | Truth check: every sentence colored by how well your files back it | M | 0 |
| 3 | Feedback + Arena: 👍/👎, blind A/B battles, personal leaderboard | M | 0, 1 |
| 4 | Learning router: trained on your feedback **and on public data** | L | 3 |
| 5 | Model council: several models agree on hard questions | M | 1 |
| 6 | Background brain: folder watcher, weekly digest, flashcards | M | 0 |

Sizes: S = a few hours, M = a few days, L = about a week or more.

Build in phase order. Each phase ships on its own, with tests, and leaves
NEXUS working.

---

## Phase 0: Foundations

Every later phase needs these three things.

### 0.1 Chat memory
**Problem:** `engine.py:272` sends the question alone (`prompt = question`) to
`/api/generate`, so the model forgets earlier messages.

**Steps**
1. Switch `_ollama_generate` to Ollama's `/api/chat`, which takes a `messages`
   list (`[{"role": "user", "content": ...}, ...]`).
2. Change `answer()` to accept `history: list[dict]` and send the last N turns
   (default 8, trimmed to fit a token budget).
3. When RAG is on, put the retrieved chunks in a system message for the
   **current** turn only (never stored in history), so old context doesn't
   pile up. A short follow-up is searched together with the previous
   question, so "why that one?" still finds the right documents.
4. Define one shared generation interface that every provider will use:
   ```python
   generate(provider, model, messages, temperature, on_token=None,
            images=None) -> str
   ```
5. Pass `st.session_state.messages` from `app.py` and the chat history from
   `server.py`.

**Tests:** a follow-up question includes the earlier turns; history is trimmed
to the budget; RAG context appears only in the last message.

### 0.2 Local store (`store.py`, SQLite file `nexus.db`)
Tables:
- `chats(id, title, created_at)`
- `messages(id, chat_id, role, content, meta_json, created_at)`
- `feedback(message_id, rating, reason, created_at)` (used in phase 3)
- `battles(id, prompt, task, model_a, model_b, winner, created_at)` (phase 3)
- `usage(provider, day, requests, tokens)` (phase 1)

Add a "Past chats" list to the sidebar. `nexus.db` goes in `.gitignore`.

### 0.3 Config
- `.env` for API keys (already in `.gitignore`), loaded with `python-dotenv`.
- `nexus.toml` for settings: cloud mode, daily limits, which models are
  enabled. A missing file means "fully local", so the app's current behaviour
  stays the default.
- `config.py` reads both and exposes one `Settings` object.

---

## Phase 1: Cloud providers

### 1.1 Design: one OpenAI-style client

Every provider accepts the same OpenAI-style `/chat/completions` request, so
**one file covers them all**.

| Provider | Base URL | Key variable | Client | Main use in NEXUS |
|---|---|---|---|---|
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | `cloud_client.py` | general use, long documents, **images** |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | `cloud_client.py` | very fast chat, **Whisper speech-to-text** |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `cloud_client.py` | many models with one key |
| DeepSeek | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | `cloud_client.py` | cheap deep reasoning |
| Mistral | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` | `cloud_client.py` | optional extra |

**Model names change often, so they never go in code.** They live in
`nexus.toml`. On startup NEXUS calls each provider's `GET /models` and the
Diagnostics tab flags any configured name that no longer exists.

### 1.2 `cloud_client.py`
- Uses `requests` (already a dependency), so no new SDK is needed.
- `chat(provider, model, messages, temperature, stream, on_token, images)`.
- Streaming: parse `data: {...}` server-sent-event lines until `data: [DONE]`.
- Images: OpenAI-style `image_url` content part holding a base64 data URI.
- `list_models(provider)` uses `GET {base}/models`.
- Typed errors that the engine understands:
  - `AuthError` (401/403): bad key → provider marked "key invalid".
  - `RateLimited` (429): skip to the next model, mark the provider "cooling
    down" for N minutes.
  - `Offline` (connection or timeout error): skip, mark the provider offline.

### 1.3 Catalogue changes (`providers.py`)
Extend `ModelSpec`:
```python
provider: str            # "ollama" | "gemini" | "groq" | "openrouter" | ...
local: bool              # False for every cloud model
cost_tier: str           # "free" | "cheap" | "paid"
modalities: frozenset    # {"text"}, {"text", "image"}, {"audio"}
context_window: int
```
Cloud `ModelSpec`s are built from `nexus.toml` at startup, so adding a model
means editing config, not code.

### 1.4 `availability()` and `plan()`
- `availability()` adds
  `cloud: {provider: "ready" | "no_key" | "invalid_key" | "cooling_down" | "offline"}`.
- `plan()`: remove the line that skips every non-Ollama model
  (`providers.py:285`) and replace it with policy checks:

**Privacy and cost rules**

| Request | Cloud: Off (default) | Cloud: Hard only | Cloud: Allowed |
|---|---|---|---|
| Question using your documents (RAG) | local | local | local, unless "Send documents to cloud" is ticked |
| PC actions (`system_agent`) | toolkit | toolkit | toolkit, **always** |
| Easy chat | local | local | local first; cloud is a fallback |
| Hard prompt (complexity ≥ 0.6) | local | cloud may win | cloud may win |
| Image, with no local vision model | message: "install one or enable cloud" | Gemini | Gemini |
| Voice input | local Whisper if installed | Groq Whisper | Groq Whisper |
| Paid model (OpenRouter auto) | never | only if `allow_paid = true` | only if `allow_paid = true` |
| Daily limit reached for a provider | — | provider skipped | provider skipped |

`LOCAL_BONUS` and `RAG_LOCAL_BONUS` already exist and start to matter as soon
as cloud models compete.

### 1.5 Engine (`engine.py`)
- `_generate()` gets a branch for `cloud_client`.
- On `RateLimited` or `Offline`, record the attempt and move to the next model
  in the chain. That loop already exists, so a dead internet connection just
  means a local model answers.
- After each cloud call, add one row to the `usage` table.
- Keys are never logged. `router_logs.jsonl` stores provider and model names
  only.

### 1.6 Image and voice input (fills the empty vision and speech tasks)
- **Images:** add an image upload to the Chat tab. Send the image to a local
  vision model (`/api/chat` with `images: [base64]`) or to Gemini, following
  the rules above.
- **Voice:** record or upload audio, transcribe it (Groq
  `/audio/transcriptions` with a Whisper model, or local `faster-whisper` when
  cloud is off), then treat the text as a normal question.
- Gemini audio through the OpenAI-style layer is not confirmed, so speech uses
  Whisper. Gemini audio can come later through Gemini's native API.

### 1.7 UI (`app.py`, `server.py`, `ui/index.html`)
- Sidebar: **Cloud: Off / Hard only / Allowed**, plus checkboxes for **Send
  documents to cloud** (default off) and **Allow paid models** (default off).
- 🌐 badge on any answer that left your PC, with the provider name.
- Diagnostics tab: one row per provider showing key status, today's usage
  against the daily limit, and models found through `/models`.

### 1.8 Tests (`tests/test_cloud.py`, every HTTP call mocked, so CI needs no keys)
1. No key → no cloud model in the chain.
2. A RAG question never includes a cloud model unless "Send documents to cloud"
   is on.
3. `system_agent` never routes to cloud, under any setting.
4. A 429 from the first cloud model → the next model answers, and the UI note
   says why.
5. Offline → a local model answers.
6. Cloud mode Off → the chain is identical to today's (regression guard).
7. Paid models are excluded unless `allow_paid` is on.
8. The daily limit is reached → that provider is skipped.
9. Streaming parser handles split lines, empty lines and `[DONE]`.

---

## Phase 2: Truth check

**Output:** each sentence of an answer is colored, and the answer gets a trust
score.

- 🟢 **Supported**: a source chunk backs it (the file is shown).
- 🟡 **Not found**: nothing in your files says this.
- 🔴 **Contradicted**: your files say the opposite.

**Steps**
1. Split the answer into sentences, skipping questions, greetings and code
   blocks.
2. For each sentence, compare it with the retrieved chunks. If it's a non-RAG
   answer, run a quick retrieval first.
3. Score each sentence/chunk pair with a small CPU NLI cross-encoder (for
   example `cross-encoder/nli-deberta-v3-small`). It runs the same way as the
   reranker you already use.
4. Label each sentence with its best result: entailment → 🟢,
   contradiction → 🔴, otherwise 🟡.
5. Trust score = share of factual sentences that are 🟢.
6. UI: colored highlights, and clicking a sentence shows its supporting chunk.
7. Measure it: build `eval/claims_golden.json` (about 60 labeled sentences)
   and report accuracy in `eval_truth.py`, the same way `eval_rag.py` works
   for retrieval.

**Budget:** about 20 sentences × 5 chunks = 100 pairs, which is fast on CPU.
Runs after the answer finishes streaming, so the chat never waits for it.

---

## Phase 3: Feedback and Arena (collects training data)

1. **👍/👎** under every answer, with an optional reason (wrong, too slow, too
   long, off-topic). Saved to `feedback`.
2. **Implicit signals:**
   - clicking "answer again with…" (`app.py:435`) = 👎 for the first model
   - a Task override = a corrected task label
   - forcing RAG to Always = the RAG gate was wrong
3. **Arena mode** (toggle): two models answer the same question with names
   hidden, and you pick A, B or tie. Saved to `battles`. Pairs can be local vs
   local or local vs cloud, which shows whether cloud is worth it for you.
4. **Leaderboard** in Diagnostics: an Elo score for each model and each task,
   from your battles.

---

## Phase 4: Learning router + training from other data

This is real machine learning that runs outside Ollama. It trains two small
models and measures both.

### 4.1 Task classifier (replaces the hand-written rules)
- Input: the MiniLM embedding the router already computes, plus a few cheap
  features (length, has code, has a path).
- Model: logistic regression (scikit-learn), small and fast.
- **Training data, from three sources:**
  1. **Your own data:** task overrides and 👍/👎 from phase 3.
  2. **Teacher labels:** a cloud model labels a few thousand varied prompts
     with a task (general, coding, reasoning, planning, vision, speech,
     system_agent). This is cheap, and it's the fastest way to get a large
     training set.
  3. **Public prompt datasets:** prompts from open chat datasets, labeled by
     the teacher in step 2.
- Safety net: if the classifier's confidence is low, fall back to today's
  rules.
- `eval_router.py` plus `eval/router_golden.json` (about 150 hand-labeled
  prompts) report accuracy and a confusion matrix: **rules vs trained model**.

### 4.2 Strong-vs-weak router (decides local or cloud)
This follows the RouteLLM approach (LMSYS, 2024): learn from human preference
data whether a prompt needs a strong model.
- **Bootstrap from public data:** `lmarena-ai/arena-human-preference-55k`
  (reported as Apache 2.0; confirm on the dataset card before using it). Put
  the models in each battle into "strong" and "weak" groups, and label a
  prompt "needs strong" when the strong model clearly won.
- Train: prompt embedding → P(needs a strong model).
- **Fine-tune on your Arena battles** (local vs cloud) from phase 3.
- This replaces the fixed `complexity ≥ 0.6` cut-off with a learned one.
- Measure: a **cost-quality curve**, plotting answer quality (from held-out
  battles) against the share of prompts sent to cloud. Pick the threshold
  where the curve flattens.

### 4.3 Personal preference bonus
- Your per-task Elo scores (phase 3) add a small bonus inside
  `score_model()`, so models you prefer rise over time.

### 4.4 Training pipeline (`train/`)
```
train/build_dataset.py   # merges own data + teacher labels + public data
train/train_router.py    # trains both models and writes metrics
models/router_vN.joblib  # versioned; ignored by git
models/router_vN.json    # accuracy, confusion matrix, data sizes
```
- `python run.py --train` retrains. A new model is used **only if it beats the
  current one** on the golden set, so a bad training run can't make routing
  worse.
- Training-only dependencies (`scikit-learn`, `datasets`) go in
  `requirements-train.txt`, so the app stays light.

### 4.5 Stretch goal: NEXUS's own model inside Ollama (distillation)
- Opt in to saving cloud answers to hard prompts.
- Fine-tune a small local model on them with QLoRA (for example, Unsloth in
  4-bit), export it to GGUF, and load it with
  `ollama create nexus-student -f Modelfile`.
- Result: a model **made by NEXUS** that runs in Ollama and is added to the
  catalogue like any other.
- Limits: the 8 GB RTX 4060 can handle QLoRA on roughly 3B–8B models, but
  slowly. Some providers' terms limit training on their outputs, so check
  each provider before including its answers.

---

## Phase 5: Model council

- Triggers when the learned router (or complexity score) marks a prompt as
  hard, or when you turn it on manually.
- Asks the top 3 models from the chain. Cloud calls run **in parallel**. Local
  models run **one after another**, because the 8 GB card holds one at a time.
- A judge model gets all the answers and returns:
  - points where all the models agree
  - points where they disagree, and which model said what
  - a final merged answer
- Phase 2's truth check runs on the merged answer.
- Limits: one council per prompt, counted against the daily limits.

---

## Phase 6: Background brain

1. **Watcher** (`watchdog`): new or changed files in `documents/` and chosen
   folders are indexed automatically (`ingest.py` already only re-embeds
   changed files).
2. **Scheduler:** a weekly digest of what changed, what you studied and what's
   new in your notes, written by a **local** model.
3. **Study mode:** flashcards and quiz questions from new notes, with answers
   checked against the source by the phase 2 truth check.
4. **Inbox tab** in the UI collects these, so NEXUS comes to you instead of
   waiting for you to type.
5. Privacy: everything here stays local by default.

---

## New and changed files

| File | Phase | Change |
|---|---|---|
| `config.py`, `nexus.toml.example`, `.env.example` | 0 | new |
| `store.py` | 0 | new |
| `engine.py` | 0, 1, 5 | `/api/chat`, history, cloud branches, council |
| `cloud_client.py` | 1 | new |
| `providers.py` | 1, 4 | richer `ModelSpec`, cloud catalogue, policy rules, learned scores |
| `router.py` | 4 | loads the trained classifier, keeps rules as fallback |
| `app.py`, `server.py`, `ui/index.html` | 0–6 | memory, cloud switch, badges, images, voice, truth colors, feedback, Arena, Inbox |
| `truth_check.py`, `eval_truth.py` | 2 | new |
| `train/`, `eval_router.py`, `requirements-train.txt` | 4 | new |
| `council.py` | 5 | new |
| `watcher.py`, `digest.py` | 6 | new |
| `tests/test_cloud.py`, `test_memory.py`, `test_truth.py`, `test_router_learned.py` | each phase | new |
| `.gitignore` | 0, 4 | `nexus.db`, `models/` |

---

## Results to show (for a portfolio or course report)

| Measure | Script | Compares |
|---|---|---|
| Retrieval recall@5, grounded@5 | `eval_rag.py` (exists) | retrieval setups |
| Task-routing accuracy | `eval_router.py` | hand rules vs trained classifier |
| Cost-quality curve | `train/train_router.py` | always local vs always cloud vs learned router |
| Truth-check accuracy | `eval_truth.py` | NLI labels vs hand labels |
| Personal leaderboard | Diagnostics tab | your Elo per model and task |

---

## Risks and how the plan handles them

| Risk | Handling |
|---|---|
| Free-tier limits change | daily limits in config; on 429 the chain falls back to local |
| Model names change | names live in `nexus.toml`, checked against `/models` at startup |
| Private files leaking to the cloud | RAG and PC actions are local by default; a test enforces it |
| Bad training run makes routing worse | a new model is used only if it beats the old one on the golden set |
| Slow on an 8 GB GPU | council and truth check run only when needed; cloud calls run in parallel |
| Dataset or provider licence limits | check licences before training; distillation is opt-in |
| CI has no keys or Ollama | every HTTP call is mocked in tests |
