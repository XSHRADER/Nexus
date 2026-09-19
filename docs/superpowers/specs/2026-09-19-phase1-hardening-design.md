# Phase 1 — Hardening: design

Date: 2026-09-19
Status: approved in brainstorming, awaiting spec review
Branch: `phase1-hardening`

## Why

NEXUS works end to end, but a read of the code turned up problems that make it
unsafe or unreliable to build more on:

1. **`server.py` lets any web page move your files.** It sends
   `Access-Control-Allow-Origin: *`, parses the request body as JSON whatever
   its `Content-Type`, and honours `confirm: true` from the request. While
   `run.py --server` is running, a page in any browser tab can POST
   `{"question": "sort my downloads", "confirm": true}` as `text/plain` (no
   CORS preflight) and `pc_agent.handle()` executes the move with no preview.
2. **No conversation memory.** `engine.answer()` sends only the newest message
   to the model, so follow-ups have no context.
3. **Prompts are silently truncated.** Nothing sets Ollama's `num_ctx`, so the
   window is Ollama's default (2k–4k tokens depending on version) and Ollama
   drops the start of anything longer. At `top_k=10` the retrieved context
   alone is ~2,200 tokens. Same class of bug as the embedding truncation fixed
   in `5210096`.
4. **New models are invisible to Auto.** `providers.plan()` only considers
   models in the hand-written `CATALOG`; anything else installed can only be
   pinned.
5. **Operational blind spots.** `router_logs.jsonl` grows forever; there is no
   record of latency, tokens/s or cold loads; a generation can hang for 300 s
   with no way to stop it; `engine.py` and `server.py` have no tests.
6. **The index holds stale facts about NEXUS itself.**
   `documents/checklist.txt` still gives the project's old
   `Desktop\3rd year\FML` path.

## Roadmap context

This is the first of two sub-projects. Phase 2 (its own spec) adds PC
operations as a **permission ladder** — tier 0 read-only (run immediately),
tier 1 reversible file changes (preview → confirm → journaled undo, deletes go
to the Recycle Bin), tier 2 open file/folder/app (confirm), tier 3 model-written
PowerShell (off by default, per-command confirm, audit log, not undoable) —
built as a **registry of declared operations** that a tool-calling model fills
in, with the current regex parser kept as the offline fallback. Phase 1 records
each model's `tools` capability so Phase 2 can use it.

Streamlit (`app.py`) is the primary UI from here on. `server.py` is kept as a
small, locked-down API; it does not get new features.

## Goals

- Close the cross-site file-operation hole in `server.py` and make it
  impossible to apply a change without a server-issued, single-use token.
- Multi-turn chat with history that is guaranteed to fit the context window,
  and chats that persist across restarts.
- Auto-selection that considers every installed model, not just catalogued ones.
- Per-answer metrics, bounded logs, sane timeouts, and a working Stop button.
- Tests for everything above.

## Non-goals

- New PC operations, tool calling, shell access (Phase 2).
- Image/audio attachments, auto-indexing (parked).
- Conversation history in `server.py` / `ui/index.html` (Streamlit only).
- Authentication or multi-user support — NEXUS stays single-user, localhost.

---

## 1. Storage — new `store.py`

One SQLite file at `data/nexus.db` (path overridable with the `NEXUS_DB`
environment variable; tests use a temp file). Stdlib `sqlite3`, no new
dependency. `data/` is added to `.gitignore`.

Connections open with `journal_mode=WAL` and `foreign_keys=ON`; the schema is
created on first connect and versioned with `PRAGMA user_version` (= 1).

```sql
CREATE TABLE chats (
    id       TEXT PRIMARY KEY,          -- uuid4 hex
    title    TEXT NOT NULL,             -- first user message, trimmed to 60 chars
    created  REAL NOT NULL,
    updated  REAL NOT NULL
);

CREATE TABLE messages (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id  TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role     TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content  TEXT NOT NULL,
    meta     TEXT,                      -- JSON: the dict app.py already renders
    created  REAL NOT NULL
);
CREATE INDEX messages_chat ON messages(chat_id, id);

CREATE TABLE turns (                    -- one row per engine.answer() call
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    chat_id       TEXT,
    task          TEXT,
    model         TEXT,                 -- model that answered, NULL if none did
    attempts      TEXT,                 -- JSON list of {model, error}
    rag_used      INTEGER,
    chunks_dropped INTEGER,             -- retrieved chunks cut to fit the window
    ttft_ms       REAL,                 -- request sent -> first token
    total_ms      REAL,
    load_ms       REAL,                 -- Ollama load_duration
    prompt_tokens INTEGER,              -- Ollama prompt_eval_count
    eval_tokens   INTEGER,              -- Ollama eval_count
    tokens_per_s  REAL,                 -- eval_count / eval_duration
    truncated     INTEGER,
    stopped       INTEGER,
    error         TEXT
);
CREATE INDEX turns_ts ON turns(ts);
```

Public functions (all take an optional `conn`; default is a module-level
connection to `NEXUS_DB`):

| function | purpose |
|---|---|
| `create_chat(title) -> str` | new chat, returns id |
| `list_chats(limit=20) -> list[dict]` | most recently updated first |
| `load_messages(chat_id) -> list[dict]` | `{role, content, meta}` in order |
| `append_message(chat_id, role, content, meta=None)` | also bumps `chats.updated` |
| `delete_chat(chat_id)` | cascades to messages |
| `record_turn(row: dict)` | insert into `turns` |
| `recent_turns(limit=20) -> list[dict]` | newest first |
| `model_stats(window=500) -> list[dict]` | per model over the last `window` turns: median `total_ms`, median `tokens_per_s`, failure rate (from `attempts`), cold loads (`load_ms > 1000`) |

**Failure policy:** storage is never allowed to fail an answer. Every store call
made from `engine.py` / `app.py` is wrapped; on error it prints one line to
stderr and the answer proceeds.

## 2. Engine — `/api/chat`, history, context budget

### Interface

```python
def answer(
    question: str,
    base_dir: Path | None = None,
    options: Options | None = None,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    history: list[dict] | None = None,   # [{"role": "user"|"assistant", "content": str}]
    chat_id: str | None = None,          # only used to tag the turns row
) -> dict
```

- **`confirm` is removed.** Mutating PC actions can only run through
  `apply_pending()`. This removes the path the server exploit uses.
- New result keys: `thinking` (str | None), `truncated` (bool),
  `chunks_dropped` (int), `stopped` (bool, always False from the engine itself),
  `metrics` (`{ttft_ms, total_ms, load_ms, prompt_tokens, eval_tokens,
  tokens_per_s}`).
- `engine.answer()` records exactly one `turns` row per call, including calls
  where every model failed (recorded before the `RuntimeError` is raised).

### Generation

`_ollama_generate` is replaced by `_ollama_chat`, which always streams
internally (`on_token` stays optional). Payload:

```json
{"model": "...", "messages": [...], "stream": true,
 "options": {"temperature": 0.7, "num_ctx": 8192},
 "think": true}
```

- `messages` = trimmed history + one final user message holding the question,
  wrapped in the RAG prompt from `rag_pipeline.build_prompt` when retrieval ran.
  Retrieved chunks are attached to the newest message only; past turns are
  plain text.
- `think` is sent only when the model's capabilities include `thinking`
  (Ollama errors if it is sent to a model without it). `message.thinking`
  chunks go to `on_thinking` and the `thinking` result key. For Ollama
  versions that inline reasoning instead, a `<think>…</think>` block in the
  final content is moved into `thinking`.
- Timeout is `(5, 120)`: 5 s to connect, and at most 120 s of silence between
  streamed chunks. A long answer is fine as long as tokens keep arriving; a
  stuck model fails and the chain moves to the next one. (Model load, ~30 s on
  this machine, happens before the first chunk and fits inside 120 s.)
- **Callback exceptions are not model failures.** An exception raised by
  `on_token` or `on_thinking` is re-raised immediately instead of being caught
  by the chain loop. Without this, Streamlit's interrupt (see §5) would be
  swallowed and the next model would silently start answering.
- Metrics come from the final (`done: true`) chunk: `load_duration`,
  `prompt_eval_count`, `eval_count`, `eval_duration` (nanoseconds → ms / s).
  `ttft_ms` is measured client-side, from sending the request to the first
  streamed token of either kind (thinking or answer) — it is the "is anything
  happening" number, and includes model load.

### Context budget

```
NUM_CTX       = int(env NEXUS_NUM_CTX or 8192), capped at the model's
                context_length when /api/show reports one
REPLY_RESERVE = 1024 tokens
estimate(t)   = ceil(len(t) / 3)      # deliberately pessimistic; English runs
                                      # ~4 chars/token on Llama/Qwen tokenizers
```

The budget is computed per model attempt (cheap, pure function), in priority
order:

1. The question and prompt template are always included. If they alone exceed
   the budget, the request is still sent (refusing would be worse), with no
   chunks or history, and the result is flagged `truncated`.
2. Retrieved chunks are added in rank order while they fit. Chunks that don't
   fit are dropped and counted in `chunks_dropped`; `info` tells the user
   ("3 of 20 chunks left out to fit the context window").
3. History fills what remains, newest turn first, whole messages only.

`num_ctx=8192` on an 8 GB card: llama3.1-8B Q4 weights ~4.9 GB + ~1 GB KV
cache at 8k context ≈ 6 GB, which fits.

**Truncation check.** The pre-send budget is the guarantee. After the reply, if
`prompt_eval_count >= 0.98 × num_ctx` the result is flagged `truncated` and
`info` says so. This post-check is best-effort only: Ollama excludes a cached
prompt prefix from `prompt_eval_count`, so it can under-report and miss a
truncation, but it never cries wolf.

### Follow-up retrieval

When `history` is non-empty, retrieval is on, and the new message is under 12
words, the retrieval query is `previous user message + " " + new message`. The
prompt still carries the new message verbatim. The eval set has no history, so
this cannot move eval numbers.

## 3. Model discovery — `providers.py`

- `capabilities(model) -> dict` calls Ollama `POST /api/show` once per model
  (cached for the process) and returns `{caps: set[str], context_length: int |
  None, parameter_size: float | None}`. It works for every installed model,
  catalogued or not, and degrades to `{caps: set(), ...None}` when Ollama is
  older or unreachable.
- `ModelSpec` gains `tools: bool = False` and `thinking: bool = False`, filled
  from `caps` at plan time (used by §2 now and by Phase 2).
- `discover(installed) -> list[ModelSpec]` builds a spec for each installed tag
  that `resolve_installed` does not match to the catalogue:

  | name contains / capability | strengths |
  |---|---|
  | `embed`, or caps has `embedding` but not `completion` | excluded |
  | `coder`, `code` | coding 0.80, general 0.45 |
  | `r1`, `qwq`, `think`, `reason`, or caps has `thinking` | reasoning 0.78, planning 0.68, coding 0.55, general 0.50 |
  | `vl`, `llava`, `vision`, `moondream`, or caps has `vision` | vision 0.72 |
  | anything else | general 0.75, coding 0.50, reasoning 0.50, planning 0.45 |

  Quality / speed from parameter size: ≤4B 0.45/0.90, ≤9B 0.58/0.80,
  ≤15B 0.70/0.50, larger 0.75/0.25 (won't fit in 8 GB, partly runs on CPU).
  Unknown size uses the ≤9B row.
- **All discovered strengths are multiplied by 0.9**, so a hand-tuned catalogue
  entry wins any tie. Discovered models are picked when nothing catalogued
  fits better; promoting one is a one-line `CATALOG` addition. Their `reason`
  says "profile inferred from name" so the Why panel is honest about it.
- `plan()` iterates `CATALOG + discover(installed)`.

## 4. `server.py` lockdown

- `Access-Control-Allow-Origin` is removed from every response.
- Every request: `Host` must be `127.0.0.1:<port>` or `localhost:<port>`
  (blocks DNS rebinding), else **403**. If an `Origin` header is present it must
  be `http://127.0.0.1:<port>` or `http://localhost:<port>`, else **403**.
  `<port>` is read from the running server's bound address, not hard-coded, so
  the tests' ephemeral port is checked by the same code as the default 8000.
- Every POST: `Content-Type` must be `application/json`, else **415**. A
  cross-site page can only send JSON after a CORS preflight, which this server
  never approves. Body over 1 MB → **413**. Malformed JSON → **400**.
- `POST /api/chat` ignores any `confirm` field. When the answer needs
  confirmation it returns `pending_id` (a `secrets.token_urlsafe(16)`) instead
  of the raw `pending` dict. The server keeps `{pending_id: (pending, expiry)}`
  in memory under a lock; entries expire after 10 minutes.
- New `POST /api/apply {"pending_id": "..."}`: pops the entry (single use) and
  runs `engine.apply_pending()`. Unknown or expired id → **404**. The client
  never sends a path to act on.
- `ui/index.html`'s Apply button calls `/api/apply` with the id.

`app.py` is not affected: Streamlit already keeps `pending` in server-side
session state and ships XSRF protection.

## 5. Streamlit (`app.py`)

**Chats.** `session_state.chat_id` holds the open chat; it is created lazily on
the first user message. Every message appended to `session_state.messages` is
also persisted with `store.append_message`. The sidebar gets a **Chats**
section above the existing "New chat" button: the 20 most recent chats as
buttons (title, relative time); clicking one loads it; a small delete control
removes it (after a confirm).

**History sent to the engine** = the persisted messages before the current
question, as `{role, content}` (metadata stripped). For a **regenerate** click,
the answer being regenerated and its duplicated question are excluded, so the
model never sees the answer it is being asked to redo.

**Reasoning panel.** When a result has `thinking`, a collapsed "Reasoning"
expander sits above the answer. While a thinking model is reasoning, the
expander streams live and the answer area shows "thinking…".

**Stop button.** A Stop button is rendered above the streaming answer. Any
widget click during a run makes Streamlit raise its rerun exception at the next
`st.*` call — which is the `placeholder.markdown` inside `on_token`. That
propagates out of the engine (callback exceptions are re-raised, §2), exits the
`requests` context manager, closes the socket, and Ollama stops generating when
its client disconnects. `run_turn` wraps the call in `try/finally`: if the turn
didn't complete and the buffer is non-empty, the partial answer is appended and
persisted with `meta.stopped = True`, and a `turns` row with `stopped = 1` is
recorded.

This is the one piece that depends on Streamlit internals, so it is built
**spike first** (build step 7): a throwaway script confirms (a) the interrupt
arrives inside `on_token`, (b) `finally` runs, and (c) Ollama's `/api/ps` or
logs show generation stopped. If any of these fails, the fallback is a
cooperative flag checked in `on_token`, and this section is revised before
building.

**Diagnostics tab** gains:
- *Recent answers* — last 20 `turns`: time, task, model, total, TTFT, tokens/s,
  cold load, fallbacks, truncated/stopped/error flags.
- *Per model* — `store.model_stats()`: median latency, median tokens/s, failure
  rate, cold loads.
- *Discovered models* — each non-catalogued installed model with its inferred
  profile and capabilities.

## 6. Logs

`DecisionLogger` rotates `router_logs.jsonl` when it passes 5 MB:
`.jsonl` → `.jsonl.1` → `.jsonl.2` → `.jsonl.3`, oldest deleted. The
Diagnostics "last 15 routing decisions" view keeps reading the live file.

## 7. Corpus fix

Replace the old `C:\Users\vashu\OneDrive\Desktop\3rd year\FML\Nexus` path in
`documents/checklist.txt` (lines 60, 63) and its twin in
`program_info/checklist.txt` with `C:\Users\vashu\OneDrive\Documents\GitHub\Nexus`. The files stay: 16 of the
18 eval questions are scored against `documents/`. Run `python eval_rag.py`
before and after; recall/MRR/grounded numbers must not drop.

## 8. Testing

All tests use the existing unittest style so CI needs no change.

| file | covers |
|---|---|
| `tests/test_engine.py` (new) | `requests.post` patched with a fake NDJSON stream. Chain fall-through; pinned model; `rag_mode` always/never/auto; history trimmed within budget, newest first, whole messages; chunks dropped + counted when over budget; truncation flag at ≥98% of `num_ctx`; `thinking` split (native field and inline `<think>`); `think` only sent to thinking-capable models; callback exception re-raised, not treated as a model failure; one `turns` row per call, including all-failed |
| `tests/test_server.py` (new) | Real `ThreadingHTTPServer` on an ephemeral port, `answer`/`apply_pending` stubbed. Foreign `Origin` → 403; bad `Host` → 403; `text/plain` POST → 415; no ACAO header; >1 MB → 413; bad JSON → 400; `pending_id` works once, then 404; expired id → 404; **regression: `{"question": "sort ...", "confirm": true}` to `/api/chat` executes nothing** |
| `tests/test_store.py` (new) | Temp-file DB: chat CRUD, cascade delete, message order, `record_turn`, `model_stats` medians and failure rate |
| `tests/test_providers.py` (extend) | Discovery rules per family; embedding models excluded; 0.9 scaling; catalogue entry beats an equal discovered one; `capabilities()` degrades when `/api/show` fails |
| `tests/test_pc_agent.py` (adjust) | Drop `confirm=True` usage; mutations only via `apply()` |

Plus: `python eval_rag.py` before/after (§7), and a manual run of `run.py
--check` with Ollama up.

## 9. Build order

One commit per step on `phase1-hardening`; the full test suite passes before
the next step starts.

1. `server.py` lockdown, `pending_id` + `/api/apply`, remove `confirm` from
   `engine.answer()` / `pc_agent.handle()`, `ui/index.html` update, `test_server.py`
2. Timeouts `(5, 120)` and log rotation
3. `store.py` + `record_turn` wiring + `test_store.py`
4. `/api/chat` switch, `num_ctx`, context budget, truncation flag, callback re-raise
5. History in `app.py`, persisted chats + sidebar, follow-up retrieval
6. `capabilities()` / `discover()`, `think` handling, Reasoning panel, Diagnostics additions
7. Stop-button spike (throwaway), then the Stop button
8. Checklist path fix + eval before/after
9. README: new controls, Diagnostics, `NEXUS_DB` / `NEXUS_NUM_CTX`, server API change

## 10. Risks

| risk | mitigation |
|---|---|
| Streamlit interrupt behaves differently than expected | Spike first (step 7); cooperative-flag fallback |
| Older Ollama lacks `capabilities`, `think`, or `context_length` | Each degrades independently: name-only discovery, no `think` param, 8192 default |
| Token estimate is off for code-heavy or non-English text | `len/3` over-estimates for English; post-check flags what slips through |
| 8k context doesn't fit a large model's KV cache in 8 GB | Ollama offloads to CPU (slower, not broken); `NEXUS_NUM_CTX` lowers it; Diagnostics shows the slowdown |
| Schema needs to change in Phase 2 | `PRAGMA user_version` gives a migration hook |

## Acceptance

- The cross-site request in "Why" §1 no longer changes anything on disk, and a
  test proves it.
- A follow-up question ("and the second one?") is answered using the previous
  turn; chats survive an app restart.
- No prompt is sent that the budget estimates to exceed `num_ctx`; dropped
  chunks are reported in the UI.
- An installed model that isn't in `CATALOG` can be chosen by Auto, and the Why
  panel says its profile was inferred.
- Diagnostics shows per-answer and per-model metrics; `router_logs.jsonl`
  stays under ~20 MB total.
- Stop ends generation within a second or two and keeps the partial answer.
- All tests pass in CI; eval numbers do not drop.
