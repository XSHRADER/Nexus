"""The learned router (Phase 4).

Trains real models on the seed data (hashed features, so it is fast and the
same everywhere) in a temporary models/ folder, and checks the gate, the
router integration and the training-data plumbing.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demos"))

import fakes  # noqa: E402
import isolate  # noqa: F401  (must precede any nexus import)

from nexus import (
    cloud,  # noqa: E402
    config,  # noqa: E402
    feedback,  # noqa: E402
    providers,  # noqa: E402
    )
from nexus import evaluate_learned_router as eval_router  # noqa: E402
from nexus import learned_router as lr  # noqa: E402
from nexus.router import TaskRouter, preference_bonuses  # noqa: E402
from train import data_sources, teacher_label, train_router  # noqa: E402


class TempModels(unittest.TestCase):
    """Isolated models/ folder and nexus.db for each test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(lr, "MODELS_DIR", root / "models"),
            mock.patch.object(lr, "CURRENT", root / "models" / "current.json"),
            mock.patch.object(data_sources, "TEACHER", root / "teacher.jsonl"),
            mock.patch.object(data_sources, "ARENA55K", root / "arena.jsonl"),
        ]
        for p in self.patches:
            p.start()
        self.store = fakes.use_db(root / "t.db")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        fakes.restore_db()
        self.tmp.cleanup()

    def train(self, *extra):
        with mock.patch("builtins.print"):
            return train_router.main(["--embeddings", "off", *extra])


class FeatureTests(unittest.TestCase):
    def test_hashing_is_stable_and_normalised(self):
        a = lr.hashed_features("Write a Python function to sort a list")
        b = lr.hashed_features("Write a Python function to sort a list")
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(v * v for v in a.values()), 1.0, places=5)
        self.assertTrue(all(0 <= i < lr.HASH_DIM for i in a))

    def test_code_and_path_signals(self):
        idx_code, _ = lr._bucket("has:code")
        idx_path, _ = lr._bucket("has:path")
        self.assertIn(idx_code, lr.hashed_features("TypeError: x() missing 1 argument"))
        self.assertIn(idx_path, lr.hashed_features("sort C:\\Users\\me\\Downloads"))


class ModelTests(unittest.TestCase):
    def test_learns_a_simple_split_and_round_trips(self):
        texts = [f"apple banana {i}" for i in range(20)] + [f"engine wheel {i}" for i in range(20)]
        labels = ["fruit"] * 20 + ["car"] * 20
        X = lr.featurize(texts)
        model = lr.LinearModel(["fruit", "car"], X.n_features).fit(X, labels, epochs=20)
        self.assertEqual(model.predict(lr.featurize(["banana apple", "wheel engine"])),
                         ["fruit", "car"])
        probs = model.predict_proba(X)
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))
        with tempfile.TemporaryDirectory() as tmp:
            router = lr.LearnedRouter({"task": model}, {
                "version": "v9", "use_embeddings": False,
                "heads": {"task": {"classes": ["fruit", "car"]}}})
            router.save(Path(tmp) / "r.npz")
            again = lr.LearnedRouter.load(Path(tmp) / "r.npz", router.meta)
        self.assertEqual(router.predict("apple pie"), again.predict("apple pie"))

    def test_class_weights_handle_rare_labels(self):
        texts = [f"common thing {i}" for i in range(40)] + ["rare marker word"] * 3
        labels = ["no"] * 40 + ["yes"] * 3
        X = lr.featurize(texts)
        model = lr.LinearModel(["no", "yes"], X.n_features).fit(X, labels, epochs=30)
        self.assertEqual(model.predict(lr.featurize(["the rare marker word again"])), ["yes"])


class TrainingTests(TempModels):
    def test_trained_router_beats_rules_and_becomes_current(self):
        report = self.train()
        self.assertTrue(report["made_current"], report["gate"])
        g = report["metrics"]["golden"]
        self.assertGreater(g["new"]["task_accuracy"], g["rules"]["task_accuracy"])
        self.assertGreaterEqual(g["new"]["docs_f1"], g["rules"]["docs_f1"])
        self.assertLess(g["new"]["docs_false_alarms"], g["rules"]["docs_false_alarms"])
        self.assertEqual(lr.current_meta()["version"], report["version"])
        self.assertIsNotNone(lr.load_current())

    def test_worse_model_is_saved_but_not_used(self):
        first = self.train()
        tiny_task = [{"prompt": "hello", "label": "general", "weight": 1.0, "source": "seed"},
                     {"prompt": "fix bug", "label": "coding", "weight": 1.0, "source": "seed"}]
        tiny_docs = [{"prompt": "hello", "label": "no", "weight": 1.0, "source": "seed"},
                     {"prompt": "my notes", "label": "yes", "weight": 1.0, "source": "seed"}]
        with mock.patch.object(data_sources, "task_and_docs_rows",
                               return_value=(tiny_task, tiny_docs)):
            second = self.train()
        self.assertFalse(second["made_current"])
        self.assertEqual(lr.current_meta()["version"], first["version"])
        self.assertTrue((lr.MODELS_DIR / f"router_{second['version']}.npz").exists())

    def test_your_corrections_are_training_rows(self):
        feedback.record_signal("task_override", "summarise this paper for me", task="general",
                               value="reasoning")
        feedback.record_signal("rag_override", "what did I write about chroma", value="always")
        task, docs = data_sources.own_rows()
        self.assertEqual((task[0]["label"], task[0]["weight"]), ("reasoning", data_sources.OWN_WEIGHT))
        self.assertEqual(docs[0]["label"], "yes")
        report = self.train()
        self.assertEqual(report["heads"]["task"]["rows"]["yours"], 1)


class RouterIntegrationTests(TempModels):
    def test_learned_router_fixes_the_over_eager_document_gate(self):
        self.train()
        rules = TaskRouter(use_learned=False)
        learned = TaskRouter()
        q = "Explain what a project manager does."
        self.assertTrue(rules._needs_rag(q))         # the keyword gate fires on "project"
        self.assertFalse(learned._needs_rag(q))
        self.assertTrue(learned._needs_rag("What do my notes say about the vector store?"))
        decision = learned.route("Write a Python function that merges two dicts",
                                 available_models=["llama3.1:8b", "qwen2.5-coder:7b"])
        self.assertEqual(decision["task"], "coding")
        self.assertTrue(decision["router_method"].startswith("learned"))

    def test_unsure_learned_router_defers_to_rules(self):
        self.train()
        router = TaskRouter()
        flat = {"task": {t: 1 / 7 for t in data_sources.TASKS}, "docs": {"no": 0.5, "yes": 0.5}}
        with mock.patch.object(router.learned, "predict", return_value=flat):
            result = router.classify("Fix this Python TypeError")
        self.assertEqual(result["method"], "rules (learned router unsure)")
        self.assertEqual(result["task"], "coding")

    def test_rules_mode_ignores_the_trained_model(self):
        self.train()
        settings = mock.MagicMock(router_mode="rules")
        with mock.patch.object(config, "get_settings", return_value=settings):
            self.assertIsNone(TaskRouter().learned)


class StrongHeadTests(TempModels):
    def records(self):
        recs = []
        hard = ["prove the theorem about primes", "derive the gradient of attention",
                "design a distributed consensus protocol", "find the bug in this concurrent code"]
        easy = ["hello there", "what is the capital of france", "tell me a joke", "thanks a lot"]
        for i in range(160):
            recs.append({"prompt": f"{hard[i % 4]} case {i}", "model_a": "big", "model_b": "small",
                         "winner": "a"})
            recs.append({"prompt": f"{easy[i % 4]} number {i}", "model_a": "small", "model_b": "big",
                         "winner": "a" if i % 3 else "tie"})
            recs.append({"prompt": f"filler {i}", "model_a": "mid", "model_b": "small", "winner": "tie"})
            recs.append({"prompt": f"other {i}", "model_a": "mid", "model_b": "big", "winner": "b"})
        return recs

    def test_tiers_and_rows_from_votes(self):
        rows, info = data_sources.strong_rows_from_records(self.records(), min_battles=50)
        self.assertEqual(info["strong_models"], ["big"])
        self.assertIn("small", info["weak_models"])
        self.assertTrue({r["label"] for r in rows} == {"strong", "weak"})
        self.assertIn("tie", {r["outcome"] for r in rows})

    def test_strong_head_is_trained_measured_and_used(self):
        rows, _ = data_sources.strong_rows_from_records(self.records(), min_battles=50)
        with open(data_sources.ARENA55K, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        report = self.train()
        strong = report["metrics"]["strong"]
        self.assertGreater(strong["auc"], 0.9)
        self.assertTrue(strong["passed"])
        self.assertIn("strong", report["heads"])
        router = TaskRouter()
        self.assertTrue(router.needs_strong("prove the theorem about primes for n")[0])
        self.assertFalse(router.needs_strong("hello there friend")[0])

    def test_auc_and_quality_curve_math(self):
        self.assertAlmostEqual(train_router.auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertAlmostEqual(train_router.auc([1, 0], [0.1, 0.9]), 0.0)
        curve = train_router.quality_curve(["strong", "weak", "tie", "strong"], [0.9, 0.1, 0.5, 0.8])
        self.assertEqual(curve["points"][0]["quality"], curve["all_weak"])
        self.assertEqual(curve["points"][-1]["quality"], curve["all_strong"])
        self.assertGreater(curve["mean_gain_over_random"], 0)

    def test_learned_hard_flag_reaches_cloud_planning(self):
        policy = cloud.CloudPolicy(mode="hard")
        avail = {"ollama": ["llama3.1:8b"], "loaded": [],
                 "cloud": {"gemini": {"status": "ready"}}}
        with mock.patch.object(cloud, "is_missing", return_value=False):
            easy = providers.plan("general", "hi", 0.1, avail, policy=policy, hard=False)
            hard = providers.plan("general", "hi", 0.1, avail, policy=policy, hard=True)
        self.assertTrue(all(c.spec.is_local for c in easy))
        self.assertFalse(hard[0].spec.is_local)


class PreferenceTests(TempModels):
    def test_settled_arena_results_reorder_close_models(self):
        for _ in range(8):
            bid = feedback.create_battle("Explain x", "general", 0.1,
                                         {"model": "qwen2.5:7b", "provider": "ollama"},
                                         {"model": "llama3.1:8b", "provider": "ollama"})
            feedback.vote(bid, "a")
        bonuses = preference_bonuses("general")
        self.assertGreater(bonuses["qwen2.5:7b"], 0)
        self.assertLess(bonuses["llama3.1:8b"], 0)
        avail = {"ollama": ["llama3.1:8b", "qwen2.5:7b"], "loaded": []}
        before = providers.plan("general", "explain x", 0.1, avail)[0].model
        after = providers.plan("general", "explain x", 0.1, avail, bonuses=bonuses)[0].model
        self.assertEqual((before, after), ("llama3.1:8b", "qwen2.5:7b"))

    def test_unsettled_results_change_nothing(self):
        bid = feedback.create_battle("q", "general", 0.1, {"model": "a"}, {"model": "b"})
        feedback.vote(bid, "a")
        self.assertEqual(preference_bonuses("general"), {})

    def test_own_local_vs_cloud_battles_become_strong_rows(self):
        bid = feedback.create_battle("prove it", "reasoning", 0.9,
                                     {"model": "llama3.1:8b", "provider": "ollama"},
                                     {"model": "gemini-3.7-flash", "provider": "gemini"})
        feedback.vote(bid, "b")
        bid = feedback.create_battle("hi", "general", 0.0,
                                     {"model": "llama3.1:8b", "provider": "ollama"},
                                     {"model": "qwen2.5:7b", "provider": "ollama"})
        feedback.vote(bid, "a")  # local vs local: no strong/weak signal
        rows = data_sources.own_strong_rows()
        self.assertEqual([(r["prompt"], r["label"]) for r in rows], [("prove it", "strong")])


class TeacherTests(TempModels):
    def test_parse_label(self):
        self.assertEqual(teacher_label.parse_label('Sure: {"task": "coding", "needs_docs": "true"}'),
                         {"task": "coding", "needs_docs": True})
        self.assertIsNone(teacher_label.parse_label('{"task": "poetry"}'))
        self.assertIsNone(teacher_label.parse_label("no json here"))

    def test_labels_a_prompt_file_with_a_cloud_model(self):
        import mock_cloud

        server, url = mock_cloud.start_in_thread()
        prompts = Path(self.tmp.name) / "p.txt"
        prompts.write_text("Sort my downloads folder\nWhat do my notes say about Chroma?\n\n")
        settings = mock.MagicMock(provider_overrides={"groq": {"base_url": f"{url}/groq/v1"}},
                                  daily_limits={"groq": 100}, cooldown_minutes=5)
        try:
            with mock.patch.object(cloud, "get_settings", return_value=settings), \
                 mock.patch.dict("os.environ", {"GROQ_API_KEY": "k"}), \
                 mock.patch("builtins.print"):
                out = teacher_label.main([str(prompts), "--provider", "groq",
                                          "--model", "openai/gpt-oss-20b"])
                again = teacher_label.main([str(prompts), "--provider", "groq",
                                            "--model", "openai/gpt-oss-20b"])
        finally:
            server.shutdown()
        self.assertEqual((out["labelled"], again["labelled"]), (2, 0))  # resumable
        rows = list(lr.iter_jsonl(data_sources.TEACHER))
        self.assertEqual([(r["task"], r["needs_docs"]) for r in rows],
                         [("system_agent", False), ("general", True)])
        task, _ = data_sources.task_and_docs_rows(include_own=False)
        self.assertEqual(data_sources.counts(task)["teacher"], 2)


class TrainingDataTests(TempModels):
    """What the router learns from, and what it must never learn from."""

    def test_every_shipped_source_feeds_the_task_head(self):
        task, docs = data_sources.task_and_docs_rows(include_own=False)
        by_source = data_sources.counts(task)
        for source in ("seed", "curated", "examples"):
            self.assertGreater(by_source.get(source, 0), 100, source)
        # The examples carry a task label only, so they stay out of the documents head.
        self.assertNotIn("examples", data_sources.counts(docs))
        self.assertGreater(data_sources.counts(docs).get("curated", 0), 100)

    def test_a_held_out_prompt_is_never_trained_on_even_as_your_correction(self):
        held = eval_router.load_golden()[0]["prompt"]
        feedback.record_signal("task_override", held.upper() + " !!", task="general", value="coding")
        feedback.record_signal("task_override", "a brand new question of mine", task="general",
                               value="coding")
        task, _ = data_sources.task_and_docs_rows()
        yours = [r["prompt"] for r in task if r["source"] == "yours"]
        self.assertEqual(yours, ["a brand new question of mine"])
        self.assertFalse(data_sources.held_out_prompts()
                         & {data_sources.normalise(r["prompt"]) for r in task})

    def test_no_shipped_training_prompt_is_a_near_copy_of_a_test_prompt(self):
        # Word-for-word copies are filtered at training time; this catches the
        # reworded ones ("weigh the trade-offs of X" / "analyze the trade-offs
        # between X"), which inflate a score just as well.
        import json

        from nexus.embeddings import get_sentence_transformer

        model = get_sentence_transformer(config.EMBED_MODEL)
        held = []
        for path in data_sources.HELD_OUT:
            data = json.loads(path.read_text(encoding="utf-8"))
            held += [i.get("q") or i.get("prompt") for i in data.get("queries") or data.get("prompts")]
        task, _ = data_sources.task_and_docs_rows(include_own=False)
        prompts = sorted({r["prompt"] for r in task})
        sims = (model.encode(prompts, normalize_embeddings=True)
                @ model.encode(held, normalize_embeddings=True).T)
        close = [(prompts[i], held[int(sims[i].argmax())]) for i in range(len(prompts))
                 if sims[i].max() > 0.85]
        self.assertEqual(close, [])

    def test_the_gate_checks_the_larger_set_too(self):
        report = self.train("--no-own")
        wide = report["metrics"]["wide"]
        self.assertIn("wide_set_not_worse_than_rules", report["gate"])
        self.assertIn("wide_set_not_worse_than_current", report["gate"])
        self.assertGreater(wide["new"], 0.5)
        self.assertIsNone(wide["previous"])


class GoldenSetTests(unittest.TestCase):
    def test_golden_is_balanced_and_separate_from_training(self):
        # Only the tasks a text prompt can be routed to are scored: "vision"
        # and "speech" prompts would count against any router unfairly.
        golden = eval_router.load_golden()
        self.assertEqual(len(golden), 100)
        self.assertEqual({g["task"] for g in golden}, set(TaskRouter.TASKS))
        for task in TaskRouter.TASKS:
            self.assertEqual(sum(g["task"] == task for g in golden), 20)
        seed = {r["prompt"] for r in lr.iter_jsonl(data_sources.SEED)}
        self.assertFalse(seed & {g["prompt"] for g in golden})


if __name__ == "__main__":
    unittest.main()
