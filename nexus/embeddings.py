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

# Fallback if the model can't be loaded to ask it directly. all-MiniLM-L6-v2
# is a 256-token model; anything longer is silently truncated at embed time.
FALLBACK_MAX_TOKENS = 256


@lru_cache(maxsize=4)
def get_sentence_transformer(name: str = EMBED_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@lru_cache(maxsize=4)
def get_cross_encoder(name: str = CROSS_ENCODER_NAME):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


@lru_cache(maxsize=4)
def get_tokenizer(name: str = EMBED_MODEL_NAME):
    """The embedding model's own tokenizer.

    Chunking has to measure length in the same units the embedder truncates
    on. Measuring in characters is what let 44% of the corpus get cut off
    before it was ever embedded.
    """
    return get_sentence_transformer(name).tokenizer


@lru_cache(maxsize=4)
def get_max_tokens(name: str = EMBED_MODEL_NAME) -> int:
    """Hard input limit of the embedding model, in tokens."""
    try:
        return int(get_sentence_transformer(name).max_seq_length)
    except Exception:
        return FALLBACK_MAX_TOKENS


def count_tokens(text: str, name: str = EMBED_MODEL_NAME) -> int:
    """Length of `text` in embedding-model tokens, excluding [CLS]/[SEP]."""
    return len(get_tokenizer(name).encode(text, add_special_tokens=False, verbose=False))
