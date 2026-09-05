# NEXUS AI — RAG Module: Setup Guide

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
documents automatically (400 tokens per chunk, 60-token overlap).

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

## Automatic AI selection

**You never pick a model.** Every message is classified, scored for difficulty,
and sent to whichever AI is strongest at that job among the ones actually
reachable right now (`providers.py` → `router.py` → `engine.py`).

| Your message looks like | Goes to | Why |
|---|---|---|
| "make me a plan / roadmap / step-by-step" | **Gemini 2.5 Pro** | planning is where the cloud model is decisively better |
| "compare these trade-offs, justify it" | **Gemini 2.5 Pro** → `deepseek-r1` | deep reasoning, same logic |
| "fix this Python error" | `qwen2.5-coder` (local) → Gemini | a specialist local coder is as good and free |
| "what is X", "hi" | `llama3.1` (local) → `gemini-flash` | fast, cheap, private |
| anything using your `documents/` | **local models only** | your files don't leave the PC |
| "describe this image" | `qwen2.5vl` / `llava` → Gemini | vision-capable models only |
| "sort my Downloads" | built-in toolkit | no model involved at all |

Three things decide the pick: the routed **task**, an estimated **difficulty**
(prompt length, planning language, multi-part questions — long/complex prompts
pull toward higher-capability models), and **what's reachable**. Local models
carry a preference bonus for being free and private — except on planning and
reasoning, where that bonus is dropped so Gemini wins on merit.

A fourth factor is **what's already in VRAM**. Your 8GB card holds one 7B
model at a time, so switching costs ~30s of load. The selector reads Ollama's
`/api/ps` and gives a resident model a small bonus — enough to break a tie
between two similar chat models, never enough to take a coding task away from
the specialist coder.

Nothing here fails hard. The router returns a *chain*, not one model, and the
engine walks it: Ollama down → the same request goes to Gemini; no Gemini key →
planning falls back to `deepseek-r1` locally; neither available → you get a
plain message saying how to fix it, and PC folder actions keep working.

The Gemini key is entered once in the sidebar and saved to `.gemini_key`
(gitignored) — no re-pasting on every run.

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
