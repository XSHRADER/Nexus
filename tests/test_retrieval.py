import shutil
import tempfile
import unittest
from pathlib import Path

from loaders import chunk_text
from retrieve import Retriever, tokenize


def fake_counter(text: str) -> int:
    """Deterministic stand-in for the real tokenizer.

    Chunking invariants shouldn't depend on downloading a model, so these
    tests count whitespace words instead.
    """
    return len(text.split())


class TokenizeTests(unittest.TestCase):
    def test_splits_identifiers_on_punctuation(self):
        # The whole reason for keeping a keyword arm: a query for "router"
        # has to match a chunk that says "router.py".
        self.assertIn("router", tokenize("see router.py for details"))
        self.assertIn("py", tokenize("see router.py for details"))

    def test_lowercases(self):
        self.assertEqual(tokenize("BM25 Hybrid"), ["bm25", "hybrid"])

    def test_drops_punctuation_only_input(self):
        self.assertEqual(tokenize("--- !!! ---"), [])


class ChunkBudgetTests(unittest.TestCase):
    BUDGET = 20

    def _chunks(self, text, overlap=4):
        return chunk_text(
            text, max_tokens=self.BUDGET, overlap_tokens=overlap, count_tokens=fake_counter
        )

    def test_no_chunk_exceeds_the_budget(self):
        # This is the regression that mattered: chunks were sized in
        # characters, silently overflowed the embedder, and the tail of every
        # chunk was dropped before it was ever embedded.
        text = "\n\n".join(" ".join(f"word{i}" for i in range(30)) for _ in range(6))
        for chunk in self._chunks(text):
            self.assertLessEqual(fake_counter(chunk), self.BUDGET)

    def test_long_unbroken_line_is_split(self):
        text = " ".join(f"w{i}" for i in range(200))
        chunks = self._chunks(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(fake_counter(chunk), self.BUDGET)

    def test_overlap_carries_context_forward(self):
        lines = [f"line{i} alpha beta gamma delta" for i in range(12)]
        chunks = self._chunks("\n".join(lines), overlap=8)
        self.assertGreater(len(chunks), 1)
        # Something from the tail of chunk 0 should reappear in chunk 1.
        tail = set(tokenize(chunks[0])[-8:])
        self.assertTrue(tail & set(tokenize(chunks[1])))

    def test_short_text_is_one_chunk(self):
        self.assertEqual(self._chunks("just a few words"), ["just a few words"])

    def test_empty_text(self):
        self.assertEqual(self._chunks("   \n\n  "), [])

    def test_every_chunk_is_non_empty(self):
        text = "\n\n".join(f"para {i} " + "x " * 40 for i in range(5))
        self.assertTrue(all(c.strip() for c in self._chunks(text)))


class RetrieverTests(unittest.TestCase):
    """End-to-end against a throwaway Chroma store."""

    DOCS = [
        ("alpha.md", "NEXUS stores embeddings in Chroma and searches them by cosine similarity."),
        ("beta.md", "The pc_toolkit sorts a Downloads folder into category subfolders."),
        ("gamma.md", "Ollama serves llama3.1:8b locally for final answer generation."),
    ]

    @classmethod
    def setUpClass(cls):
        import chromadb

        from embeddings import get_sentence_transformer

        cls.tmp = tempfile.mkdtemp(prefix="nexus_test_")
        model = get_sentence_transformer()
        client = chromadb.PersistentClient(path=cls.tmp)
        col = client.get_or_create_collection(
            "nexus_documents", metadata={"hnsw:space": "cosine"}
        )
        texts = [t for _, t in cls.DOCS]
        col.add(
            ids=[f"{src}::0" for src, _ in cls.DOCS],
            embeddings=model.encode(texts, normalize_embeddings=True).tolist(),
            documents=texts,
            metadatas=[{"source": src, "chunk_index": 0} for src, _ in cls.DOCS],
        )
        cls.retriever = Retriever(db_dir=cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_indexes_every_document(self):
        self.assertEqual(self.retriever._indexed_count, len(self.DOCS))

    def test_every_hit_carries_its_source(self):
        # Hits found only by BM25 used to come back with meta={}, so the
        # generator cited "unknown" instead of the filename.
        for hit in self.retriever.query("Downloads folder", top_k=3):
            self.assertTrue(hit["meta"].get("source"))

    def test_keyword_only_match_is_found(self):
        hits = self.retriever.query("pc_toolkit", top_k=3)
        self.assertIn("beta.md", [h["meta"]["source"] for h in hits])

    def test_semantic_match_without_shared_words(self):
        hits = self.retriever.query("where are the vectors kept?", top_k=3)
        self.assertEqual(hits[0]["meta"]["source"], "alpha.md")

    def test_results_are_ordered_best_first(self):
        hits = self.retriever.query("local answer generation", top_k=3)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_arms_can_be_ablated(self):
        self.assertTrue(self.retriever.query("Chroma", top_k=2, use_bm25=False))
        self.assertTrue(self.retriever.query("Chroma", top_k=2, use_vector=False))

    def test_both_arms_report_their_rank(self):
        hits = self.retriever.query("Chroma cosine similarity", top_k=3, rerank=False)
        self.assertTrue(
            any(h["vector_rank"] is not None and h["bm25_rank"] is not None for h in hits)
        )

    def test_top_k_is_respected(self):
        self.assertLessEqual(len(self.retriever.query("NEXUS", top_k=2)), 2)

    def test_empty_query_does_not_crash(self):
        self.assertIsInstance(self.retriever.query("   ", top_k=3), list)


if __name__ == "__main__":
    unittest.main()
