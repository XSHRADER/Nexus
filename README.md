# NEXUS

[![tests](https://github.com/XSHRADER/Nexus/actions/workflows/tests.yml/badge.svg)](https://github.com/XSHRADER/Nexus/actions/workflows/tests.yml)

A private assistant that runs on your own PC. It answers from your documents,
picks the right local model for each question, shows how every answer was
made, and can tidy your folders. Cloud models are optional and off by default.

**Versions.** What changed in each one: [CHANGELOG.md](CHANGELOG.md). What is planned next: [ROADMAP.md](ROADMAP.md).

| It can | How |
|---|---|
| Answer from your files | hybrid search (vector + keyword, re-ranked) over PDF, DOCX, TXT and Markdown, with numbered sources |
| Choose the model for you | a router classifies each question and scores every installed model against it |
| Show its work | under each answer: which model, why, how fast, and which passages were used |
| Check itself | a truth check marks each sentence as supported, not found or contradicted by your files |
| Learn what you prefer | ratings, blind Arena comparisons, a personal leaderboard, a router that trains on them |
| Get a second opinion | a council of models answers and a judge merges them |
| Work while you are away | watches your documents, keeps an inbox and digest, makes checked flashcards |
| Tidy your PC | organize folders, find duplicates and large files — always previewed first, with undo |

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
| `--train` | retrain the learned router, then exit |
| `--test` | run the test suite (and lint, if ruff is installed) |
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

## Setting up by hand

`start.bat` and `run.py` do all of this for you. To do it yourself:

```bash
python -m venv nexus-env
nexus-env\Scripts\activate          # macOS/Linux: source nexus-env/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b             # any chat model; several GB
python -m nexus.ingest              # index what is in documents/
python run.py                       # or: python -m nexus   for a chat in the terminal
```

Python 3.11 to 3.13. Put your own files in `documents/` (PDF, DOCX, TXT,
Markdown) and index again; only new and changed files are read. The first run
downloads two small models for search (about 90 MB each), and the truth check
downloads one more the first time it runs.

---

## The interface

NEXUS makes three decisions for you on every message — which model answers,
whether to consult your documents, and which passages to pull. The UI is built
so that all three are visible after the fact and overridable before it.

| page | what it's for |
|---|---|
| **Chat** | Answers stream in word by word. Under each one: where it was written (this PC or a cloud provider), which model and why, the passages that grounded it, a truth check, and thumbs up/down. |
| **Inbox & study** | What NEXUS did on its own (indexing, the digest, new flashcards) and the flashcards that are due. |
| **Leaderboard** | Which models you prefer, from your Arena votes and ratings, and the state of the learned router. |
| **Documents** | Per-file status, drag-and-drop upload, and re-indexing — no command line needed. |
| **Retrieval lab** | Run one query through all four retrieval arms side by side. No model runs; this is search only. |
| **Diagnostics** | What NEXUS can reach, cloud provider status, per-model speed and failure rates, recent routing decisions, and the configuration in use. |

**Every automatic decision has an override** under **Answer settings** in the
chat, and every control defaults to Auto, so leaving them alone reproduces the
untouched behaviour exactly:

| control | why you'd touch it |
|---|---|
| Answer with | *One model* (the usual way), *Arena* (two models, names hidden, you pick) or *Council* (several models and a judge). |
| Model | Pin one instead of letting the scorer choose. A pinned model keeps the rest of the chain as fallback. |
| Task | Override the classifier when it reads a question wrong. |
| Use your documents | `Auto` searches your documents and uses them when they look relevant. `Always` forces it; `Never` skips it. |
| Retrieval depth | How many passages reach the model. Passages are about 240 tokens, so 10 is roughly 2,200 tokens of context. |
| Rerank with cross-encoder | Off is faster; on is sharper, and Auto needs it to judge relevance. |
| Temperature | 0 for deterministic and factual; higher wanders more. |
| Cloud models | *Off* (default), *Hard questions* or *Allowed*, plus whether documents and paid models may be used. |

Any setting that isn't automatic shows as a badge above the chat, with a
*Reset to automatic* button.

**Under every answer.** *How this was answered* shows the routed task, the
router that decided, the estimated difficulty, time to first word, speed,
prompt size, and every model that was scored with its reason — plus anything
that was skipped and why. *Sources* lists each retrieved passage with its score
and its rank in each search arm. If an answer looks wrong, the buttons beneath
it re-run the same question on the next-best models NEXUS already scored.

**Chats are saved and follow-ups work.** Every message is stored in a local
SQLite file (`data/nexus.db`), and the sidebar lists your chats to search,
reopen or delete. Earlier turns go to the model with each new question: the
newest within the memory budget (8 messages, about 2,000 tokens, both set under
`[memory]` in `nexus.toml`), then as many of those as fit the model's context
window. A short follow-up ("and the second one?") also borrows the previous
question for document search.

**Nothing is silently cut off.** Ollama drops the start of any prompt longer
than its context window, without an error. NEXUS asks for an 8,192-token window
explicitly and budgets every prompt into it: your question first, then
retrieved passages in rank order, then history. If passages had to be left
out, or the prompt still filled the window, the answer says so.

**Reasoning stays out of the answer.** Models that think before answering
(deepseek-r1, qwen3, …) stream their reasoning into a collapsed *Reasoning*
panel above the reply. **Stop** ends an answer mid-stream and keeps what was
written.

A second, dependency-free UI (`python run.py --server`, one HTML page) offers
the same features without Streamlit.

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

**2. Pick a cloud mode** under **Answer settings** in the chat (or `[cloud] mode` in `nexus.toml`):

| Mode | What it does |
|---|---|
| **Off** (default) | Nothing leaves this PC. Exactly the old behaviour. |
| **Hard questions** | Cloud answers hard prompts (difficulty ≥ 0.6), and things no local model can do, like an image when you have no local vision model. |
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
Every cloud answer carries a badge naming the provider; local answers say
*this PC*.

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

`python -m nexus.evaluate_truth` scores both checkers on 60 labelled claims in
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
stored in `data/nexus.db` on this PC, and it is the training data for the learned
router.

- **👍 / 👎** under every answer. After a 👎 you can say why: *wrong*, *too
  slow*, *too long*, *off-topic*. Changing your mind overwrites the rating.
- **Arena** (Answer settings → *Answer with* → Arena; a toggle in the browser UI). Each question is
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
- **Leaderboard** (its own page; a button in the browser UI): an Elo rating per
  model, overall or per task, from your Arena votes — everyone starts at
  1000, beating a stronger model gains more, a tie is half a win and *both
  bad* moves neither — next to each model's 👍 approval. Ratings with fewer
  than 5 decided votes are marked *settling*: Elo needs a few games before a
  3–0 start means anything.

---

## Learned router

Before any model runs, NEXUS makes three routing decisions. Each one is a
small trained classifier, with the keyword rules and labelled examples kept
as a fallback:

| decision | learned from |
|---|---|
| **Which task** is this (general, coding, reasoning, planning, PC action)? | 210 hand-labelled seed prompts, prompts labelled by a cloud "teacher" model, and **your corrections** (every time you override the task) |
| **Does it need your documents?** | the same, plus your document on/off overrides |
| **Does it need a strong (cloud) model?** | 55k human votes from the public [Chatbot Arena dataset](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) (the RouteLLM approach), plus **your Arena votes** between a local and a cloud model |

On top of that, your settled Arena results nudge each model's score for that
task, so models you keep preferring rise.

**Measured, not assumed.** `python -m nexus.evaluate_learned_router` scores
routing on 100 held-out prompts (`eval/router_golden.json`, 20 per task, never
trained on), including generic questions that mention "project", "file" or
"source" without being about your documents:

| router | task accuracy | documents: precision | recall | false alarms |
|---|---|---|---|---|
| rules and examples | 0.910 | 0.270 | 0.769 | 27 |
| **learned** (seed data, hashed n-grams + MiniLM) | **0.990** | **0.917** | **0.846** | **1** |

The file also holds 40 "vision" and "speech" prompts. They are left out of the
score: a text prompt is never routed to those (a vision model is only used when
an image is attached), so no router could be right on them.

The keyword gate turned retrieval on for "Explain what a project manager
does"; the learned one doesn't, and still catches "according to my notes…".
Cross-validation on the training prompts gives 0.871 task accuracy, lower than
the held-out set: the two sets were written by the same person, so expect
real-world accuracy somewhere in between until your own corrections and
teacher-labelled prompts are in the training data.

**The fallback is measured too.** When no router is trained, or it is unsure,
NEXUS classifies with keyword rules plus the nearest of 285 labelled example
prompts (`nexus/router_examples.json`). `python -m nexus.evaluate_router`
scores that on its own 150 held-out prompts: 89.3%. Add a line to the examples
file to teach it a phrasing.

**Training is safe to repeat.** `python run.py --train` (or *Retrain the router now* on
the Leaderboard page) takes seconds. Every version is saved in `models/`
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

For questions where being wrong is costly, choose **Council** under Answer
settings (a toggle in the browser UI). Up to
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
  `python -m nexus.ingest`. The first look after install only remembers what is
  there; it does not flood the inbox.
- **Makes flashcards.** A local model writes short question/answer cards
  from each new or changed file. Every answer is **truth-checked against the
  passage it came from**: a card its own source contradicts is thrown away
  and never shown; the rest are marked "verified" or "not found in source".
  Extra watched folders (lecture notes, for example) get flashcards and
  digest lines but are not added to the question index.
- **Study.** The *Inbox & study* page shows due cards one at a time. "I knew it" moves a card
  up a box (it comes back after 1, 3, 7, then 14 days); "I didn't" sends it
  back to box 1 and it returns in 10 minutes (the Leitner system).
- **Weekly digest.** Once a week it writes a short report: files added and
  changed, a one-line local summary of each, what you asked (count, task
  mix, recurring topics), your 👍/👎 and Arena votes, and your study progress.
  If no local model is running, the digest is still written without the
  summaries.
- **Inbox.** Everything the brain did lands on the *Inbox & study* page, with an unread count:
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

`python -m nexus.evaluate` (or `python run.py --eval`) scores retrieval against the 18 questions in
`eval/golden_set.json` — at file level (`recall@k`, `MRR`) and by whether the
assembled context actually contains the answer (`grounded@5`, which is
chunker-agnostic and so stays comparable when chunk sizes change).

| configuration | recall@1 | recall@5 | MRR | grounded@5 |
|---|---|---|---|---|
| dense only (vector) | 0.833 | 0.889 | 0.861 | 0.889 |
| BM25 only (keyword) | 0.333 | 0.778 | 0.500 | 0.722 |
| hybrid, RRF fusion | 0.667 | 0.944 | 0.782 | 0.889 |
| hybrid + cross-encoder | 0.556 | **1.000** | 0.733 | **0.944** |

Reranking buys recall@5 and groundedness and costs recall@1, where dense
retrieval on its own is still the sharpest. Both arms earn their keep: BM25
alone is much weaker, but it recovers questions the vector arm misses.

**What this corpus cannot tell you.** It is 2,835 words. At `top_k=5` a query
hands back roughly a sixth of everything indexed, so recall@5 saturates and the
pre-fix index scores just as well — run `python -m nexus.evaluate --legacy` to see
that side by side. The honest reading is that the chunking change is a
*correctness* fix, not a measured quality win at this scale: a 490-token chunk
is simply not represented by a 256-token embedding. Making these numbers
discriminate needs a bigger corpus, not a better reranker.

It is also why these numbers move when the corpus does. With only 31 chunks,
adding, removing or editing one small file shifts BM25's corpus-wide term
statistics enough to flip a near-tie between a question's first and second
hits, even for questions that share no words with the edit. The table above
was measured on the documents as they are in this version.

The eval did produce one directly actionable result. Token-sized chunks carry
about a quarter of what the old ones did, so `top_k=5` was feeding the model
1096 context tokens where it used to get 1950, and `grounded@5` fell to 0.944.
At `top_k=10` it receives 2190 tokens and returns to 1.000 — so retrieval depth
now defaults to 10. Retrieval depth has to follow chunk size.

---

## Automatic AI selection

**You never pick a model.** Every message is classified, scored for difficulty,
and sent to whichever AI is strongest at that job among the ones actually
reachable right now (`nexus/providers.py` → `router.py` → `engine.py`).

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

**Models NEXUS has never heard of still get used.** Any installed model that
isn't in the catalogue gets a profile inferred from its name (`coder` → coding,
`r1`/`qwq` → reasoning, `vl`/`llava`/`vision` → vision, anything else general),
its size, and the capabilities Ollama reports. Inferred profiles are scored at
90%, so a hand-tuned catalogue entry wins a tie. The *How this was answered* panel says
when a profile was inferred and Diagnostics lists them all; to promote one, add a
line to `CATALOG` in `nexus/providers.py`.

## PC automation (local file operations)

The router sends folder-management requests to the **system agent**
(`nexus/pc_agent.py` → `pc_tools.py`). No model or network needed.

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

## Configuration

Nothing needs configuring to start. Three optional sources:

**`nexus.toml`** — behaviour. Copy `nexus.toml.example`; every value in it is
the default.

| section | controls |
|---|---|
| `[memory]` | how many earlier messages, and how many characters, go back to the model |
| `[storage]` | where the database lives |
| `[cloud]` | cloud mode, documents and paid models, daily limits, extra models |
| `[speech]` | voice input: cloud provider and local Whisper size |
| `[truth]` | truth-check method, model and thresholds |
| `[router]` | learned router on/off, confidence thresholds, how much your Arena results count |
| `[council]` | members, automatic convening for hard questions, judge |
| `[brain]` | background watcher, digest interval, flashcards, extra folders |

**`.env`** — cloud API keys. Copy `.env.example`. Never committed.

**Environment variables** — where things live on this machine.

| variable | default | does |
|---|---|---|
| `NEXUS_OLLAMA_URL` | `OLLAMA_HOST`, else `http://127.0.0.1:11434` | where Ollama is |
| `NEXUS_NUM_CTX` | `8192` | context window requested from Ollama. Lower it if a large model spills out of VRAM |
| `NEXUS_DOCS_DIR` | `documents/` | the files to index |
| `NEXUS_INDEX_DIR` | `vector_store/` | the search index |
| `NEXUS_DATA_DIR` | `data/` | saved chats, metrics and logs |
| `NEXUS_DB` | `data/nexus.db` | the database file |
| `NEXUS_MODELS_DIR` | `models/` | trained routers |
| `NEXUS_CONFIG`, `NEXUS_ENV_FILE` | `nexus.toml`, `.env` | the two files above |
| `NEXUS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO` or `WARNING` |

## HTTP API (`run.py --server`)

The dependency-free server only answers its own page:

- Requests must be addressed to `127.0.0.1` or `localhost` on the server's
  port, and any `Origin` must match; otherwise `403`.
- `POST` bodies must be `application/json` (else `415`) and under 1 MB — or
  25 MB for `/api/chat`, `/api/arena` and `/api/council`, which can carry an
  image or a voice recording (else `413`).
- `POST /api/chat {"question": …}` never changes anything on disk. A file
  operation comes back as a preview with a `pending_id`.
- `POST /api/apply {"pending_id": …}` runs that one previewed action. Each id
  works once and expires after 10 minutes.

A page on another website can't trigger a file operation: it can't send JSON
without the server's permission, and it never sees a `pending_id`.

The server owns each conversation: `chat_id` in a request continues a saved
chat, and history comes from the database, not from the page. Other routes:
`/api/chats`, `/api/status`, `/api/truth`, `/api/feedback`, `/api/arena`,
`/api/arena/vote`, `/api/council`, `/api/leaderboard`, `/api/router`,
`/api/router/train`, `/api/inbox`, `/api/study`, `/api/brain/run`,
`/api/cloud/verify`.

## Project layout

| path | what's in it |
|---|---|
| `nexus/` | the app: `engine.py` (one entry point for every answer), `router.py`, `providers.py`, `retrieve.py`, `ingest.py`, `cloud.py`, `truth_check.py`, `feedback.py`, `learned_router.py`, `council.py`, `brain.py`, `store.py`, `server.py`, `ui.py` |
| `app.py`, `app_pages/` | the Streamlit UI, one file per page |
| `run.py`, `start.bat` | the launcher |
| `train/` | training the learned router |
| `eval/` | held-out test sets: retrieval, routing, truth check |
| `tests/`, `demos/` | the test suite, and stand-ins for Ollama and the cloud providers that it runs against |
| `documents/` | the files NEXUS answers from |
| `docs/` | the progress log and design notes |

`python run.py --test` runs the tests (no Ollama, network or API keys needed).

## Quick troubleshooting

- **`ModuleNotFoundError`** → the virtual environment isn't activated, or
  `pip install -r requirements.txt` didn't complete — re-run it.
- **Ollama isn't reachable** → start it with `ollama serve`, or set
  `NEXUS_OLLAMA_URL` if it runs elsewhere. `python run.py --check` says what
  is wrong.
- **Answers ignore your documents** → open the **Documents** page and check
  the file is indexed, then try the question in the **Retrieval lab** to see
  what search returns before blaming the model.
- **A cloud model is never used** → Diagnostics → *Cloud providers* shows each
  provider's status (no key, rejected key, cooling down, daily limit).
- **The first answer from a model is slow** → an 8 GB card holds one model at
  a time, so switching models means loading one (about 30 seconds).
