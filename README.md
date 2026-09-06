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

Every model runs on this machine through Ollama. There is no cloud provider
and no API key anywhere in the project — NEXUS works fully offline.

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
