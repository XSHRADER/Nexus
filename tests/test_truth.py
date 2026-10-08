"""Truth check (Phase 2).

The splitting, evidence and labelling logic is tested with a scripted
scorer, so it runs anywhere. The NLI model itself is measured against
eval/claims_golden.json where sentence-transformers is installed (CI), with
the scores printed next to the keyword baseline.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import eval_truth
import truth_check as tc
from store import ChatStore


class ScriptedScorer:
    """Returns fixed probabilities chosen by a keyword in the claim."""

    name = "scripted"
    approximate = False

    def __init__(self, rules):
        self.rules = rules
        self.pairs = []

    def score(self, pairs):
        self.pairs.extend(pairs)
        out = []
        for premise, claim in pairs:
            probs = {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9}
            for word, (in_premise, result) in self.rules.items():
                if word in claim and in_premise in premise:
                    probs = result
            out.append(probs)
        return out


SUPPORT = {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}
CONTRA = {"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05}


class SplitClaimsTests(unittest.TestCase):
    def test_offsets_point_at_the_sentence(self):
        answer = "Sure! NEXUS stores vectors in Chroma. It embeds text with MiniLM models."
        claims = tc.split_claims(answer)
        self.assertEqual([c.text for c in claims],
                         ["NEXUS stores vectors in Chroma.", "It embeds text with MiniLM models."])
        for c in claims:
            self.assertEqual(answer[c.start:c.end], c.text)

    def test_skips_code_questions_headings_and_short_bits(self):
        answer = ("# Overview\n"
                  "Here is the summary:\n"
                  "- NEXUS uses BM25 for keyword search.\n"
                  "```python\nprint('this line is code, not a claim at all')\n```\n"
                  "Do you want more detail on that?\n"
                  "Yes.\n"
                  "| table | row with several words |\n")
        self.assertEqual([c.text for c in tc.split_claims(answer)],
                         ["NEXUS uses BM25 for keyword search."])

    def test_decimals_and_abbreviations_do_not_split(self):
        claims = tc.split_claims("Python 3.13 is required, e.g. for the new typing features.")
        self.assertEqual(len(claims), 1)

    def test_markdown_is_stripped_for_checking_but_offsets_kept(self):
        answer = "1. **Chroma** stores the `vector` index on disk."
        (claim,) = tc.split_claims(answer)
        self.assertEqual(claim.text, "Chroma stores the vector index on disk.")
        self.assertTrue(answer[claim.start:claim.end].startswith("**Chroma**"))


class KeywordScorerTests(unittest.TestCase):
    def test_number_mismatch_is_a_contradiction(self):
        (p,) = tc.KeywordScorer().score([("Ollama runs at http://localhost:11434 locally.",
                                          "Ollama runs at localhost:8080.")])
        self.assertGreater(p["contradiction"], 0.5)

    def test_overlap_is_support(self):
        (p,) = tc.KeywordScorer().score([("Supported formats are TXT, Markdown, PDF, and DOCX.",
                                          "Supported formats are PDF and DOCX.")])
        self.assertGreater(p["entailment"], 0.5)

    def test_unrelated_is_neutral(self):
        (p,) = tc.KeywordScorer().score([("Chroma stores vectors.",
                                          "The team met every Tuesday in Berlin.")])
        self.assertEqual(tc.label_for(p, 0.5, 0.5)[0], tc.NOT_FOUND)


class CheckTests(unittest.TestCase):
    PASSAGES = (
        tc.Passage("info.md", "NEXUS uses Chroma for vector storage and BM25 for keywords."),
        tc.Passage("models.txt", "Ollama runs at localhost:11434 and serves llama3.1:8b."),
    )

    def test_labels_trust_and_evidence(self):
        scorer = ScriptedScorer({"Chroma": ("Chroma", SUPPORT), "8080": ("11434", CONTRA)})
        answer = ("NEXUS keeps its vectors in Chroma. Ollama serves models on port 8080. "
                  "The project was started by five students.")
        report = tc.check(answer, scorer=scorer, passages=self.PASSAGES)
        labels = [c["label"] for c in report["claims"]]
        self.assertEqual(labels, [tc.SUPPORTED, tc.CONTRADICTED, tc.NOT_FOUND])
        self.assertAlmostEqual(report["trust"], 1 / 3, places=3)
        self.assertEqual(report["counts"], {"supported": 1, "not_found": 1, "contradicted": 1})
        self.assertEqual(report["claims"][0]["source"], "info.md")
        self.assertEqual(report["claims"][1]["source"], "models.txt")
        self.assertIsNone(report["claims"][2]["source"])
        self.assertEqual(report["evidence"], "your documents")

    def test_support_in_any_passage_beats_a_contradiction(self):
        passages = (tc.Passage("old.md", "Chunks are 400 tokens long."),
                    tc.Passage("new.md", "Chunks are 240 tokens long."))
        scorer = ScriptedScorer({"240": ("400", CONTRA)})
        scorer.rules["240 "] = ("240", SUPPORT)
        report = tc.check("Chunks are 240 tokens long now.", scorer=scorer, passages=passages)
        self.assertEqual(report["claims"][0]["label"], tc.SUPPORTED)
        self.assertEqual(report["claims"][0]["source"], "new.md")

    def test_answer_sources_are_used_first(self):
        scorer = ScriptedScorer({"Chroma": ("retrieved", SUPPORT)})
        sources = [{"source": "retrieved.md", "text": "retrieved: NEXUS uses Chroma for vectors."}]
        report = tc.check("NEXUS uses Chroma for its vectors.", sources=sources,
                          scorer=scorer, passages=())
        self.assertEqual(report["evidence"], "answer sources")
        self.assertEqual(report["claims"][0]["source"], "retrieved.md")

    def test_no_documents_and_nothing_to_check(self):
        self.assertEqual(tc.check("NEXUS uses Chroma for vectors.", scorer=ScriptedScorer({}),
                                  passages=())["status"], "no_documents")
        self.assertEqual(tc.check("Hi! Want more?", scorer=ScriptedScorer({}),
                                  passages=self.PASSAGES)["status"], "nothing_to_check")

    def test_pairs_are_capped_per_claim(self):
        many = tuple(tc.Passage(f"f{i}.md", f"NEXUS uses Chroma variant {i}.") for i in range(10))
        scorer = ScriptedScorer({})
        tc.check("NEXUS uses Chroma for storage.", scorer=scorer, passages=many)
        self.assertLessEqual(len(scorer.pairs), 3)

    def test_document_passages_from_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.md").write_text("First paragraph here.\n\nSecond one about Chroma.")
            (Path(tmp) / "skip.bin").write_bytes(b"\x00")
            passages = tc.document_passages(Path(tmp))
        self.assertEqual([p.text for p in passages],
                         ["First paragraph here.", "Second one about Chroma."])


class EvidenceExcerptTests(unittest.TestCase):
    def test_excerpt_centres_on_the_relevant_line(self):
        passage = "\n".join(
            [f"FILLER line number {i} about unrelated setup steps here." for i in range(8)]
            + ["RUNTIME: Ollama at http://localhost:11434 (GPU acceleration when available)"]
            + [f"MORE filler {i} about other things entirely." for i in range(8)])
        excerpt = tc.focus_excerpt(passage, "Ollama serves the models at localhost:8080.", 200)
        self.assertIn("localhost:11434", excerpt)
        self.assertLessEqual(len(excerpt), 210)
        self.assertTrue(excerpt.startswith("…"))

    def test_short_passage_is_returned_whole(self):
        self.assertEqual(tc.focus_excerpt("  Short passage.  ", "anything here"), "Short passage.")


class StoreMetaTests(unittest.TestCase):
    def test_truth_report_is_saved_on_the_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(Path(tmp) / "t.db")
            cid = store.create_chat("q")
            mid = store.add_message(cid, "assistant", "answer", {"model": "m"})
            store.update_meta(mid, {"truth": {"trust": 0.5}})
            msg = store.get_message(mid)
        self.assertEqual(msg["meta"], {"model": "m", "truth": {"trust": 0.5}})


class GoldenSetTests(unittest.TestCase):
    def test_golden_set_is_balanced_and_valid(self):
        items = eval_truth.load_items()
        self.assertEqual(len(items), 60)
        for label in tc.LABELS:
            self.assertEqual(sum(i["label"] == label for i in items), 20)
        self.assertEqual(len({i["id"] for i in items}), 60)

    def test_metrics(self):
        m = eval_truth.metrics(["supported", "contradicted", "not_found", "supported"],
                               ["supported", "not_found", "not_found", "contradicted"])
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["confusion"]["contradicted"]["not_found"], 1)
        self.assertEqual(m["per_label"]["not_found"]["precision"], 0.5)


def _nli_available():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_nli_available(), "sentence-transformers not installed")
class NLIModelTests(unittest.TestCase):
    """Measures the real model; prints the table into the test log."""

    def test_nli_beats_keyword_baseline(self):
        results = eval_truth.run("all")
        eval_truth.print_table(results)
        nli = next(v for k, v in results.items() if k != "keyword")["given_passage"]
        keyword = results["keyword"]["given_passage"]
        for e in nli["errors"]:
            print(f"  miss {e['id']}: {e['true']} -> {e['predicted']}: {e['claim']}")
        self.assertGreater(nli["accuracy"], keyword["accuracy"])
        self.assertGreaterEqual(nli["accuracy"], 0.7)
        self.assertGreaterEqual(nli["per_label"]["contradicted"]["recall"], 0.6)


if __name__ == "__main__":
    unittest.main()
