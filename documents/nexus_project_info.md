# NEXUS AI Project Information

NEXUS AI is a local retrieval-augmented generation (RAG) assistant.

The project uses Chroma for persistent vector storage, the all-MiniLM-L6-v2 model for document embeddings, and BM25 keyword search for hybrid retrieval. Ollama provides local answer generation with the llama3.1:8b model.

Supported document formats are TXT, Markdown, PDF, and DOCX. Documents are placed in the documents folder and indexed by running ingest.py. Questions are answered by running rag_pipeline.py.

The Python environment is stored in nexus-env. The vector index is stored in vector_store.
