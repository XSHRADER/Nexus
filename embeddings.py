"""
embeddings.py
Central, cached loaders for the sentence-transformer and cross-encoder
models. The router, retriever, and ingest step all embed with the same
`all-MiniLM-L6-v2` model -- without this module each of them spins up its
own in-memory copy (~90 MB + load time) on a CPU-only machine.

`lru_cache` guarantees one instance per model name per process.
"""

from functools import lru_cache

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=4)
def get_sentence_transformer(name: str = EMBED_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@lru_cache(maxsize=4)
def get_cross_encoder(name: str = CROSS_ENCODER_NAME):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)
