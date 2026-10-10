# NEXUS AI Project Information

NEXUS AI is a local retrieval-augmented generation (RAG) assistant.

The project uses Chroma for persistent vector storage, the all-MiniLM-L6-v2 model for document embeddings, and BM25 keyword search for hybrid retrieval. Ollama provides local answer generation with the llama3.1:8b model.

Supported document formats are TXT, Markdown, PDF, and DOCX. Documents are placed in the documents folder and indexed by running python -m nexus.ingest. Questions are asked in the app started with python run.py.

The Python environment is stored in nexus-env. The vector index is stored in vector_store.

## What it does

NEXUS answers questions from your own files, chooses a suitable local model for each question, and shows how every answer was made: which model wrote it, why that model was chosen, and which passages it used. It can also tidy folders on the PC, always with a preview first.

Saved chats, ratings and settings data are kept in a single SQLite database, data/nexus.db.

## Optional features

Cloud models are optional and switched off by default. With cloud off, nothing leaves this PC. With it on, questions that use your documents and all PC actions still stay on this PC unless you allow otherwise.

A truth check marks each sentence of an answer as supported, not found, or contradicted by your documents.

In Arena mode two models answer the same question with their names hidden and you pick the better one. The leaderboard turns those votes into ratings.

In Council mode several models answer and a judge model merges their answers into one.

A background watcher notices new and changed files in the documents folder, indexes them, writes a weekly digest, and makes flashcards from your notes.

## Versions

The project uses version numbers of the form MAJOR.MINOR.PATCH. Released versions are listed in CHANGELOG.md and planned ones in ROADMAP.md. Running python -m nexus --version prints the version in use.
