# NEXUS AI — RAG Module: Setup Guide

[![tests](https://github.com/XSHRADER/Nexus/actions/workflows/tests.yml/badge.svg)](https://github.com/XSHRADER/Nexus/actions/workflows/tests.yml)

## Quick start

**Windows: double-click `start.bat`.**

That is the whole setup. On first run it creates the virtual environment and
installs dependencies; on every run it starts Ollama if it isn't already
listening, indexes anything new in `documents/`, puts a real question through
retrieval and generation to prove the pipeline works, and opens the UI.

From a terminal, or on macOS/Linux:

```bash
python run.py
```

Every step prints pass/fail and the exact command that fixes a failure, so a
broken setup tells you what is wrong instead of stack-tracing out of Streamlit.

| flag | does |
|---|---|
| *(none)* | preflight, index, launch the Streamlit UI |
| `--check` | preflight and smoke test only, then exit |
| `--eval` | score retrieval quality and exit |
| `--rebuild` | force a full re-index first |
| `--server` | use the dependency-free HTTP UI instead |
| `--pull` | download `llama3.1:8b` if no model is installed (multi-GB) |
| `--no-serve` | don't start Ollama automatically |

`start.bat` forwards any of these: `start.bat --check`.

Ollama is started detached, so it keeps running after you close the launcher
window. Downloading a model is the one prerequisite left to you — it is
several gigabytes, so `--pull` has to be asked for explicitly.

---

Everything here runs on CPU except the final generation step (which uses
Ollama and your RTX 4060, exactly as it already does today). Do these
phases in order — each one only takes a few minutes.

---

## Phase 1 — Environment setup

**Step 1.** Confirm Python is installed (3.9+):
```bash
python --version
```

**Step 2.** Create a virtual environment for this project (keeps it isolated
from other projects on your machine):
```bash
python -m venv nexus-env
```

**Step 3.** Activate it:
- Windows: `nexus-env\Scripts\activate`
- Mac/Linux: `source nexus-env/bin/activate`

**Step 4.** Copy the project files (`loaders.py`, `ingest.py`, `retrieve.py`,
`rag_pipeline.py`, `requirements.txt`) into a folder, e.g. `nexus_rag/`.

**Step 5.** Install dependencies:
```bash
pip install -r requirements.txt
```
This pulls in Chroma (vector store), sentence-transformers (embeddings),
rank-bm25 (keyword search), and file readers for PDF/DOCX. First run will
also download the small `all-MiniLM-L6-v2` embedding model (~80MB) —
one-time download, then it's cached locally.

---

## Phase 2 — Add your documents

**Step 1.** Inside `nexus_rag/`, put your files into the `documents/`
folder. Supported: `.txt`, `.md`, `.pdf`, `.docx`.

**Step 2.** Keep folder structure simple at first — subfolders are fine,
the ingestion script walks recursively.

**Step 3.** Don't worry about file size yet — chunking handles long
documents automatically (240 embedding tokens per chunk, 48-token overlap).

---

## Phase 3 — Build the vector index

**Step 1.** Run the ingestion script:
```bash
python ingest.py
```

**Step 2.** Watch the output — it will report how many files were found,
how many chunks were created, and confirm embedding completion.

**Step 3.** Check that a `vector_store/` folder was created — this is your
local, persistent index. It survives restarts; you don't need to rebuild
it unless documents change.

**Step 4.** Add or edit a document later? Just re-run `python ingest.py`.
It hashes files and **only re-embeds what changed** — so this stays fast
even with a growing document set.

---

## Phase 4 — Test retrieval on its own

**Step 1.** Before touching generation, confirm retrieval quality by
itself:
```bash
python retrieve.py
```

**Step 2.** Type a question related to your documents when prompted.

**Step 3.** Check the results: does it surface the right files? Read the
printed snippets — if results look off-topic, your chunk size or document
formatting may need adjusting (see Phase 6).

---

## Phase 5 — Connect to your Ollama model

**Step 1.** Make sure Ollama is running:
```bash
ollama serve
```
(On Windows/Mac it usually runs automatically after install — check the
system tray/menu bar.)

**Step 2.** Confirm you have a model pulled:
```bash
ollama list
```
If empty, pull one, e.g.:
```bash
ollama pull llama3.1:8b
```

**Step 3.** Open `rag_pipeline.py` and confirm `DEFAULT_MODEL` matches a
model you actually have pulled.

**Step 4.** Run the full pipeline:
```bash
python rag_pipeline.py
```

**Step 5.** Ask a question. It will retrieve relevant chunks, build a
grounded prompt, and send it to your local Ollama model — you'll get an
answer that cites the source filename when it uses your documents.

---

## Phase 6 — Tune and integrate

**Step 1.** If retrieved chunks feel too broad or too narrow, adjust
`chunk_size` / `overlap` in `ingest.py`'s call to `load_and_chunk_directory`
(defaults: 400 tokens, 60 overlap) — then delete `vector_store/` and
re-run `ingest.py` to rebuild from scratch with new settings.

**Step 2.** If certain files retrieve poorly, check they're extracting
text cleanly — scanned/image-only PDFs won't have selectable text and
will need OCR first (a separate step, not covered by this base pipeline).

**Step 3.** Wire this into your NEXUS AI router: instead of calling
`rag_query()` directly, have your router decide *when* to invoke RAG
(e.g., only when the query looks like it references personal documents)
versus routing straight to a model with no retrieval step.

**Step 4.** Once this is stable, this becomes one more "tool" your router
can call — same pattern as swapping between chat/coding/reasoning models,
just triggered by "does this need my local documents" instead of "what
kind of task is this."

---

## The interface

NEXUS makes three decisions for you on every message — which model answers,
whether to consult your documents, and which chunks to pull. The UI is built
so that all three are visible after the fact and overridable before it.

**Four tabs.**

| tab | what it's for |
|---|---|
| **Chat** | Answers stream in token by token. Under each one: which model answered and why, and the exact chunks that grounded it. |
| **Documents** | Per-file chunk counts, drag-and-drop upload, and re-indexing — no CLI needed. |
| **Retrieval lab** | Run one query through all four retrieval arms side by side and compare what each returns, with timings. No model runs; this is retrieval only. |
| **Diagnostics** | Installed models, what's resident in VRAM, index configuration, and the last 15 routing decisions. |

**Every automatic decision has an override**, and every control defaults to
Auto, so leaving them alone reproduces the untouched behaviour exactly:

| control | why you'd touch it |
|---|---|
| Model | Pin one instead of letting the scorer choose. Pinned models keep the rest of the chain as fallback. |
| Task | Override the intent classifier when it reads a question wrong. |
| Your documents | `Auto` uses the keyword gate, which can miss a question whose answer *is* in your files. `Always` forces retrieval; `Never` skips it. |
| Retrieval depth | How many chunks reach the model. Chunks are ~240 tokens, so 10 is roughly 2,200 tokens of context. |
| Cross-encoder reranking | Off is faster; on is sharper. The eval table above quantifies the difference. |
| Creativity | Sampling temperature, 0 for deterministic and factual. |

**Under every answer** sit two panels. *Why this model* shows the routed task,
the estimated difficulty, and every model that was scored with its reason —
plus anything that was skipped and the error that caused it. *Sources* lists
each retrieved chunk with its score and, more usefully, its rank in each arm:
`v=0 · b=10` means the vector search ranked that chunk first while BM25 put it
tenth. Seeing both numbers is what makes hybrid retrieval legible rather than
a claim.

If an answer looks wrong, the row of buttons beneath it re-runs the same
question on the next-best models NEXUS already scored, so comparing them costs
one click instead of a settings change.

---

## Conversation memory and saved chats

NEXUS remembers the conversation. Each question goes to the model through
Ollama's chat endpoint together with the last 8 messages (capped at about
2,000 tokens), so "what is its population?" after "what is the capital of
France?" knows what "its" means. A badge under the answer shows how many
earlier messages were used.

Retrieved document chunks are sent with the current question only and never
stored in the history, so old context can't crowd out the conversation. A
short follow-up is searched together with the previous question.

Every chat is saved to `nexus.db` (SQLite, on this PC) and listed under
**Past chats** in both UIs. Refreshing the page no longer loses anything.

Both limits and the database path can be changed in `nexus.toml`; copy
`nexus.toml.example` to start. Without that file the defaults apply.

To see it without a model installed, run `python demos/mock_ollama.py` in one
terminal and `python demos/phase0_demo.py` in another. With real Ollama
running, skip the mock.

---

## Cloud models (optional, off by default)

NEXUS runs fully on this PC until you switch cloud on. With it on, cloud
models compete with your local ones on the same scores, and NEXUS still picks
for you.

**1. Add keys** — copy `.env.example` to `.env` and fill in the ones you have:

| Provider | Key | Free tier | Used for |
|---|---|---|---|
| Gemini | `GEMINI_API_KEY` | yes | general use, long documents, **images** |
| Groq | `GROQ_API_KEY` | yes | very fast chat, **voice input** (Whisper) |
| Mistral | `MISTRAL_API_KEY` | yes | extra general model, images |
| DeepSeek | `DEEPSEEK_API_KEY` | no (cheap) | deep reasoning |
| OpenRouter | `OPENROUTER_API_KEY` | no (paid) | many models through one key |

**2. Pick a cloud mode** in the sidebar (or `[cloud] mode` in `nexus.toml`):

| Mode | What it does |
|---|---|
| **Off** (default) | Nothing leaves this PC. Exactly the old behaviour. |
| **Hard questions only** | Cloud answers hard prompts (difficulty ≥ 0.6), and things no local model can do, like an image when you have no local vision model. |
| **Allowed** | Cloud competes on every prompt, but easy ones still go local first; cloud is the fallback. |

**Rules that always hold**, whatever model wins or is pinned:

- Questions that use your documents stay on this PC unless **Send documents
  to cloud** is ticked. Earlier answers drawn from your documents are also
  removed from the history a cloud model sees.
- PC actions (sorting folders and so on) never go to the cloud.
- Paid models (OpenRouter auto) are used only with **Allow paid
  models** ticked.
- Each provider has a daily request limit (`[cloud.daily_limits]`); at the
  limit it is skipped.

**When a provider fails**, NEXUS moves to the next model and says why under
the answer ("DeepSeek was rate-limited…"). A rate-limited provider cools down
for 5 minutes, a rejected key stays off until you change it, and a retired
model name is skipped. With no internet at all, local models answer as before.
Every cloud answer carries a 🌐 badge naming the provider; local answers say
💻 This PC.

**Images and voice.** Attach an image with 📎 (browser UI) or the clip in the
chat box (Streamlit); only models that can read images are considered.
Attach or record audio and it is transcribed first: by Groq's Whisper when
cloud is on, or by `faster-whisper` on this PC when it is installed.

**Model names** live in `cloud_models.toml`, not in code, because providers
rename them often. Diagnostics → *Check model names with each provider*
flags any that a provider no longer serves. Add or replace models with
`[[cloud.models]]` in `nexus.toml`.

To try all of this without keys or internet:

```bash
python demos/mock_ollama.py      # terminal 1: stand-in for Ollama
python demos/mock_cloud.py       # terminal 2: stand-in for every provider
```

then point the providers at the mock in `nexus.toml` (see the top of
`demos/mock_cloud.py`) and set any non-empty keys.

---

## Truth check

Every answer can be checked, sentence by sentence, against your own
documents:

| colour | means |
|---|---|
| 🟢 supported | a passage in your files says the same thing |
| 🟡 not in your files | nothing in your files says it either way — the claim rests on the model's own knowledge |
| 🔴 contradicted | a passage in your files says something different |

The answer gets a **trust score** (the share of checked sentences that are
supported), and clicking a sentence shows the passage behind its colour.
Answers that used your documents are checked automatically; any other answer
has a **Check against my files** button. Results are saved with the chat.

Each sentence is compared with the passages retrieved for the answer plus the
best-matching passages in `documents/`, by a small NLI (natural-language
inference) cross-encoder, `cross-encoder/nli-deberta-v3-small`, running on the
CPU like the reranker. Code blocks, questions, headings and greetings are
skipped. When the model can't be loaded, a keyword checker stands in and the
result is marked *approximate*.

`python eval_truth.py` scores both checkers on 60 labelled claims in
`eval/claims_golden.json` (20 of each label, paraphrased, with contradictions
that change names and meanings rather than only numbers):

| checker | mode | accuracy | macro-F1 | contradictions caught |
|---|---|---|---|---|
| keyword (baseline) | given passage | 0.533 | 0.490 | 0.150 |
| keyword (baseline) | end to end | 0.533 | 0.506 | 0.200 |
| **NLI** `nli-deberta-v3-small` | given passage | **0.850** | **0.851** | **0.950** |
| **NLI** `nli-deberta-v3-small` | end to end | 0.750 | 0.754 | 0.700 |

The NLI model catches 19 of 20 contradictions where keyword matching catches
3, because most contradictions here change a name or a meaning (FAISS for
Chroma, "requires a GPU") rather than a number. Its errors lean cautious: 5 of
20 true statements came back *not found* (it hesitates when the supporting
line sits inside a long passage), and 3 of 20 unknowable claims were called
contradicted. A 🔴 is still worth reading as "go and look", not as a verdict.
*Given passage* hands the checker the exact passage each claim was written
against; *end to end* makes it find its own evidence in `documents/`, as it
does inside NEXUS. Thresholds and the model are set under `[truth]` in
`nexus.toml`.

---

## Feedback, Arena and your leaderboard

NEXUS keeps track of which models actually work **for you**. Everything is
stored in `nexus.db` on this PC, and it is the training data for the learning
router that comes next.

- **👍 / 👎** under every answer. After a 👎 you can say why: *wrong*, *too
  slow*, *too long*, *off-topic*. Changing your mind overwrites the rating.
- **⚔️ Arena mode** (toggle in the top bar or sidebar). Each question is
  answered by two models with their names hidden. Pick *A*, *B*, *Tie* or
  *Both bad*; the names are revealed and the chat continues from the answer
  you chose. One side is always NEXUS's own first choice, so you are testing
  its decision; the other is drawn from the rest of the routing chain, and
  comes from the cloud when the first choice is local (or the other way
  round) whenever cloud is on — Arena then answers "is cloud worth it for
  me?". Every privacy and cost rule still applies.
- **Corrections count too.** Asking another model to answer again is
  recorded as a quiet 👎 for the first one; overriding the task NEXUS guessed,
  or forcing documents on or off when it guessed the opposite, is recorded as
  a correction.
- **🏆 Leaderboard** (sidebar button / Leaderboard tab): an Elo rating per
  model, overall or per task, from your Arena votes — everyone starts at
  1000, beating a stronger model gains more, a tie is half a win and *both
  bad* moves neither — next to each model's 👍 approval. Ratings with fewer
  than 5 decided votes are marked *settling*: Elo needs a few games before a
  3–0 start means anything.

---

## Learned router

Before any model runs, NEXUS makes three routing decisions. Each one is now
a small trained classifier, with the old keyword rules kept as a fallback:

| decision | learned from |
|---|---|
| **Which task** is this (general, coding, reasoning, planning, vision, speech, PC action)? | 210 hand-labelled seed prompts, prompts labelled by a cloud "teacher" model, and **your corrections** (every time you override the task) |
| **Does it need your documents?** | the same, plus your document on/off overrides |
| **Does it need a strong (cloud) model?** | 55k human votes from the public [Chatbot Arena dataset](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) (the RouteLLM approach), plus **your Arena votes** between a local and a cloud model |

On top of that, your settled Arena results nudge each model's score for that
task, so models you keep preferring rise.

**Measured, not assumed.** `python eval_router.py` scores routing on 140
held-out prompts (`eval/router_golden.json`, 20 per task, never trained on),
including generic questions that mention "project", "file" or "source"
without being about your documents:

| router | features | task accuracy | documents: precision | recall | false alarms |
|---|---|---|---|---|---|
| keyword rules | keywords only | 0.729 | 0.256 | 0.769 | 29 of 127 |
| keyword rules | + MiniLM similarity | 0.821 | 0.256 | 0.769 | 29 of 127 |
| **learned** (seed data only) | hashed n-grams | **0.964** | **0.909** | 0.769 | **1** |
| **learned** (seed data only) | + MiniLM embeddings | **0.986** | **0.917** | **0.846** | **1** |

(The MiniLM rows come from the *train-router* workflow, where
sentence-transformers is installed, as it is on a normal NEXUS setup.)

The keyword gate turned retrieval on for "Explain what a project manager
does"; the learned one doesn't, and still catches "according to my notes…".
Cross-validation on the training prompts gives 0.795 task accuracy (0.871
with embeddings), lower than the golden set: the two sets were written by the same person, so expect
real-world accuracy somewhere in between until your own corrections and
teacher-labelled prompts are in the training data.

**Training is safe to repeat.** `python run.py --train` (or *Retrain now* in
the Leaderboard panel) takes seconds. Every version is saved in `models/`
with its measurements, and the new one is used only if it is at least as good
as both the rules and the router already in use on the held-out prompts —
a bad batch of feedback can't make routing worse. On first start NEXUS
trains itself. `[router] mode = "rules"` in `nexus.toml` switches it off.

The classifier is multinomial logistic regression written in numpy, on
hashed word and character n-grams, plus the MiniLM embedding when
sentence-transformers is installed. It needs no extra packages.

**More training data:**

```bash
# label any prompts with a cloud model (sends them to that provider)
python train/teacher_label.py my_prompts.txt --provider gemini --model gemini-3.7-flash
# or 2,000 prompts from the public dataset (pip install datasets)
python train/teacher_label.py --from-arena55k 2000 --provider groq --model openai/gpt-oss-120b
# learn "needs a strong model" from the 55k public votes
python train/train_router.py --arena55k
```

The *train-router* GitHub Actions workflow runs the last command in the cloud
and uploads the trained router as an artifact.

**The "needs a strong model" head did not work on public data — and the gate
caught it.** Trained on 16,686 battles between the dataset's strongest and
weakest models (tiers set by win rate; dataset licence confirmed Apache-2.0
on its card), it scored **AUC 0.526** on 3,338 held-out votes, barely above
the 0.5 of a coin flip; routing by it beat random routing by at most 1.3
percentage points at any budget. So it was left out, and cloud decisions
keep using the difficulty estimate. This matches the RouteLLM paper, where
routers trained on raw Arena votes alone were also near random: a prompt's
wording says little about whether a weaker model will do. The head is
retrained whenever you have enough Arena votes between a local and a cloud
model (weighted 5x) — votes about *your* prompts and *your* models, which is
the signal the public data lacks.

---

## Model council

For questions where being wrong is costly, switch on **🏛️ Council**. Up to
three different models answer the same question; a judge model then reads
all of them and returns:

- **what they agree on** — independent agreement is a good sign;
- **what they disagree on**, with what each model said — the places to look
  twice;
- **one merged answer**, keeping what is right in each.

Under the merged answer you also get an **agreement score**: how similar the
answers are to each other, measured without any model (embedding
similarity, or word overlap without sentence-transformers). If the judge
fails or returns something unreadable, the answer most like the others is
shown instead, and the note says so.

Members are the best different models from the request's own routing chain,
so the privacy and cost rules apply to every one of them and to the judge: a
question about your documents is answered and judged only on this PC unless
documents may go to the cloud. Cloud members run in parallel; local ones run
one after another, since an 8 GB card holds one model at a time — a local-only
council takes roughly three answers' time plus the judge's. The merged
answer can be truth-checked, rated and continued like any other.

Arena and Council are exclusive: Arena asks *you* to judge two answers,
Council asks a model to judge several. With `[council] auto = "hard"` in
`nexus.toml`, prompts the router marks as hard convene the council by
themselves (marked "convened automatically"); the default is `"off"`.

## Background brain

While the server (or the Streamlit app) runs, NEXUS keeps working when you
are not chatting. Everything below runs **on this PC with local models
only**, whatever the cloud switch says.

- **Watches your folders.** Every minute it looks at `documents/` (and any
  `[brain] watch_folders`). A new, changed or deleted file in `documents/`
  is indexed straight away, so questions use the new text without running
  `python ingest.py`. The first look after install only remembers what is
  there; it does not flood the inbox.
- **Makes flashcards.** A local model writes short question/answer cards
  from each new or changed file. Every answer is **truth-checked against the
  passage it came from**: a card its own source contradicts is thrown away
  and never shown; the rest are marked "verified" or "not found in source".
  Extra watched folders (lecture notes, for example) get flashcards and
  digest lines but are not added to the question index.
- **Study.** 🎓 Study shows due cards one at a time. "I knew it" moves a card
  up a box (it comes back after 1, 3, 7, then 14 days); "I didn't" sends it
  back to box 1 and it returns in 10 minutes (the Leitner system).
- **Weekly digest.** Once a week it writes a short report: files added and
  changed, a one-line local summary of each, what you asked (count, task
  mix, recurring topics), your 👍/👎 and Arena votes, and your study progress.
  If no local model is running, the digest is still written without the
  summaries.
- **📥 Inbox.** Everything the brain did lands here with an unread count:
  indexed files, new cards, the digest, and errors (for example "couldn't
  index — chromadb missing"). Buttons run each job now: check for changes,
  write the digest, make flashcards from all my notes.

Settings in `nexus.toml`:

```toml
[brain]
enabled = true          # false: no background thread (the buttons still work)
scan_seconds = 60       # how often to look for changes (minimum 5)
digest_days = 7         # 0 = no automatic digest
study = true            # make flashcards from new and changed files
cards_per_file = 6
watch_folders = []      # extra folders for flashcards and digests, e.g. ["~/Lectures"]
```

---

## Retrieval pipeline

Three stages, narrowing at each one:

```
vector top-40  +  BM25 top-40   ->   RRF fusion   ->   cross-encoder top-25   ->   top_k
```

**Chunks are sized in embedding tokens, not characters.** `all-MiniLM-L6-v2`
truncates at 256 tokens without raising an error, so the previous
1600-character chunks reached the embedder with their tails cut off — measured
on this repo's own documents, **34% of all tokens never reached the model**,
while BM25 went on indexing the full text. Chunks are now packed to 240 tokens
with 48 tokens of overlap, split on line and sentence boundaries. `ingest.py`
re-checks this on every run and warns if any chunk exceeds the limit.

**Fusion is Reciprocal Rank Fusion, not a weighted sum of scores.** Cosine
similarity and BM25 relevance live on different, query-dependent scales, so
`0.65 * cosine + 0.35 * bm25` adds two numbers that don't mean the same thing
from one query to the next. RRF throws the magnitudes away and fuses on rank,
which is the thing the two arms actually agree on.

**Vectors are normalised and compared by cosine.** Chroma defaults to squared
L2, which ranks by vector magnitude as well as direction.

The index records how it was built — embedding model, token budget, distance
metric. Change any of them and `ingest.py` rebuilds from scratch, because a
store half-written under one chunking scheme and half under another retrieves
worse than either alone and looks perfectly healthy from the outside.

### Measured, not assumed

`python eval_rag.py` scores retrieval against the 18 questions in
`eval/golden_set.json` — at file level (`recall@k`, `MRR`) and by whether the
assembled context actually contains the answer (`grounded@5`, which is
chunker-agnostic and so stays comparable when chunk sizes change).

| configuration | recall@1 | recall@5 | MRR | grounded@5 |
|---|---|---|---|---|
| dense only (vector) | 0.778 | 0.889 | 0.833 | 0.889 |
| BM25 only (keyword) | 0.278 | 0.778 | 0.472 | 0.722 |
| hybrid, RRF fusion | 0.667 | 0.944 | 0.773 | 0.889 |
| hybrid + cross-encoder | 0.556 | **1.000** | 0.724 | **0.944** |

Reranking buys recall@5 and groundedness and costs recall@1, where dense
retrieval on its own is still the sharpest. Both arms earn their keep: BM25
alone is much weaker, but it recovers questions the vector arm misses.

**What this corpus cannot tell you.** It is 2,835 words. At `top_k=5` a query
hands back roughly a sixth of everything indexed, so recall@5 saturates and the
pre-fix index scores just as well — run `python eval_rag.py --legacy` to see
that side by side. The honest reading is that the chunking change is a
*correctness* fix, not a measured quality win at this scale: a 490-token chunk
is simply not represented by a 256-token embedding. Making these numbers
discriminate needs a bigger corpus, not a better reranker.

The eval did produce one directly actionable result. Token-sized chunks carry
about a quarter of what the old ones did, so `top_k=5` was feeding the model
1096 context tokens where it used to get 1950, and `grounded@5` fell to 0.944.
At `top_k=10` it receives 2190 tokens and returns to 1.000 — so retrieval depth
now defaults to 10. Retrieval depth has to follow chunk size.

---

## Automatic AI selection

**You never pick a model.** Every message is classified, scored for difficulty,
and sent to whichever AI is strongest at that job among the ones actually
reachable right now (`providers.py` → `router.py` → `engine.py`).

By default every model runs on this machine through Ollama, with no API key
needed — NEXUS works fully offline. Cloud models are an opt-in extra (see
*Cloud models* above).

| Your message looks like | Goes to | Why |
|---|---|---|
| "make me a plan / roadmap / step-by-step" | `deepseek-r1` → `phi4` | chain-of-thought models are the strongest planners you have locally |
| "compare these trade-offs, justify it" | `deepseek-r1` → `phi4` | deep reasoning, same logic |
| "fix this Python error" | `qwen2.5-coder` → `deepseek-coder` | a specialist coder beats a generalist |
| "what is X", "hi" | `llama3.1` → `qwen2.5` | fast all-rounders for everyday chat |
| anything using your `documents/` | local models | your files never leave the PC |
| "describe this image" | `qwen2.5vl` → `llava` | vision-capable models only |
| "sort my Downloads" | built-in toolkit | no model involved at all |

Three things decide the pick: the routed **task**, an estimated **difficulty**
(prompt length, planning language, multi-part questions — long/complex prompts
pull toward higher-capability models), and **what's actually installed**.

A fourth factor is **what's already in VRAM**. Your 8GB card holds one 7B
model at a time, so switching costs ~30s of load. The selector reads Ollama's
`/api/ps` and gives a resident model a small bonus — enough to break a tie
between two similar chat models, never enough to take a coding task away from
the specialist coder.

Nothing here fails hard. The router returns a *chain*, not one model, and the
engine walks it top-down: if the best-fit model isn't pulled or fails to load,
the next one answers and the UI says so. If Ollama itself is down you get a
plain message explaining how to start it, and PC folder actions keep working
regardless — they never needed a model.

## PC automation (local file operations)

The router sends folder-management requests to the **system agent**
(`pc_agent.py` → `pc_tools.py`). No model or network needed.

| Say something like | It does |
|---|---|
| "analyze my Downloads folder" | file counts, size, category breakdown |
| "find duplicate files in `C:\path`" | SHA-256 duplicate groups |
| "show large files on my Desktop" | files over 50 MB, largest first |
| "sort my Downloads folder" | previews moves into `Documents/`, `Images/`, … then waits for confirm |
| "remove empty folders here" | previews, then deletes empty subdirs on confirm |
| "undo organize in `C:\path`" | reverses the last sort via its manifest |

Target folder is taken from the message: a known name (Downloads, Desktop,
Documents, Pictures, Music, Videos), a quoted path, or a `C:\...` path.
With nothing specified it uses the project folder.

**Safety:** moving/deleting always shows a preview first and only runs when
you click **Apply** (Streamlit) or **Apply changes** (browser UI). Sorting
is collision-safe, never overwrites, skips hidden files, writes an undo
manifest, and refuses drive roots, your home directory, and system paths.

## Quick troubleshooting

- **`ModuleNotFoundError`** → virtual environment isn't activated, or
  `pip install -r requirements.txt` didn't complete — re-run it.
- **Ingestion finds 0 files** → check `documents/` actually has supported
  file types, and check you're running commands from inside `nexus_rag/`.
- **Ollama connection refused** → `ollama serve` isn't running, or a
  different port is in use — check `http://localhost:11434` is reachable.
- **Answers ignore your documents** → run Phase 4 in isolation first to
  confirm retrieval itself works before blaming generation.
