# NEXUS user guide

Everything here runs on CPU except the final generation step (which uses Ollama and your RTX 4060, exactly as it already does today).

## Starting

On Windows, double-click start.bat. From a terminal, run python run.py. The launcher creates the environment on first run, starts Ollama if it is not running, indexes anything new in the documents folder, and opens the interface. Run python run.py --check to test the whole path and exit.

First run will also download the small all-MiniLM-L6-v2 embedding model (~80MB) — one-time download, then it's cached locally.

## Adding documents

Put TXT, Markdown, PDF or DOCX files into the documents folder, or drop them onto the Documents page.

Check that a vector_store/ folder was created — this is your local, persistent index. It survives restarts; you don't need to rebuild it unless documents change.

Add or edit a document later? Just re-run python -m nexus.ingest. It hashes files and only re-embeds what changed — so this stays fast even with a growing document set.

Scanned/image-only PDFs won't have selectable text and will need OCR first (a separate step, not covered by this base pipeline).

## How a model is chosen

You never pick a model. A router classifies each message as general chat, coding, reasoning, planning or a PC action, estimates how hard it is, and sends it to the strongest model for that job among the ones installed.

| Your message looks like | Goes to |
|---|---|
| a plan, roadmap or step-by-step request | deepseek-r1, then phi4 |
| comparing trade-offs | deepseek-r1, then phi4 |
| fixing a Python error | qwen2.5-coder, then deepseek-coder |
| a quick question or a greeting | llama3.1, then qwen2.5 |
| sorting the Downloads folder | the built-in toolkit, no model at all |

A model already loaded in GPU memory gets a small bonus. An 8 GB card holds one 7B model at a time, so switching costs about 30 seconds. The selector reads Ollama's /api/ps to see which model is resident; the bonus breaks a tie between two similar chat models but never takes a coding task away from the coder model.

The router returns a chain of models, not one. If the best model is missing or fails, the next one answers and the interface says so.

## The pages

- Chat: answers stream in, and under each one you can see the model, the reason, the sources, a truth check and thumbs up or down.
- Inbox and study: what NEXUS did on its own, and flashcards that are due.
- Leaderboard: which models you prefer, from Arena votes and ratings.
- Documents: per-file status, upload and re-indexing.
- Retrieval lab: compare the search methods on one query.
- Diagnostics: what NEXUS can reach and how answers have performed.

## Answer settings

Every automatic choice can be overridden under Answer settings in the chat: the model, the task, whether your documents are used, how many passages are retrieved, reranking, temperature, the cloud switch, and whether one model, Arena or Council answers.

## PC actions

Ask NEXUS to analyze the Downloads folder, find duplicate files, show large files, sort a folder into categories, remove empty folders, or undo the last sort. Moving and deleting always show a preview first and run only when you click Apply. Sorting never overwrites a file, writes an undo record, and refuses drive roots, the home directory and system paths.

## Troubleshooting

- Ollama is not reachable: start it with ollama serve.
- Answers ignore your documents: check the file is indexed on the Documents page, then try the question in the Retrieval lab.
- The first answer from a model is slow: the model is being loaded into GPU memory.
