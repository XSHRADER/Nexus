"""
run.py
One entry point that takes NEXUS from a fresh clone to an answering app.

    python run.py            preflight -> index -> Streamlit UI
    python run.py --check    preflight + an end-to-end test answer, then exit
    python run.py --eval     preflight -> index -> retrieval eval, then exit
    python run.py --test     run the test suite (and lint, if ruff is installed)
    python run.py --server   launch the dependency-free HTTP UI instead
    python run.py --rebuild  force a full re-index first
    python run.py --train    retrain the learned router, then exit

Every step prints pass/fail and the exact command that fixes a failure, so a
broken setup says what is wrong instead of stack-tracing out of Streamlit.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Preflight is a progress report, so it has to appear as it happens. Python
# block-buffers stdout when it isn't a terminal, which hid every check
# behind the Streamlit banner when the output was piped or redirected.
# UTF-8 because Windows consoles default to cp1252, which can't print the
# model's answer in the --check smoke test if it contains an emoji.
try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - older/odd streams
    pass

PROJECT_DIR = Path(__file__).resolve().parent
STREAMLIT_PORT = 8501

OK = "  [ok]  "
BAD = "  [!!]  "
WAIT = "  [..]  "

REQUIRED = [
    ("chromadb", "chromadb"),
    ("rank_bm25", "rank-bm25"),
    ("sentence_transformers", "sentence-transformers"),
    ("pypdf", "pypdf"),
    ("docx", "python-docx"),
    ("requests", "requests"),
    ("streamlit", "streamlit"),
]


def _print(good: bool, message: str, fix: str = "") -> bool:
    print(f"{OK if good else BAD}{message}")
    if not good and fix:
        print(f"         fix: {fix}")
    return good


def check_python() -> bool:
    ok = sys.version_info >= (3, 11)
    return _print(ok, f"Python {sys.version.split()[0]}", "install Python 3.11 or newer")


def check_dependencies() -> bool:
    missing = [pkg for module, pkg in REQUIRED if importlib.util.find_spec(module) is None]
    return _print(
        not missing,
        "dependencies installed" if not missing else f"missing: {', '.join(missing)}",
        f"{Path(sys.executable).name} -m pip install -r requirements.txt",
    )


def start_ollama(wait_seconds: int = 45) -> tuple[bool, str]:
    """Launch `ollama serve` in the background and wait for it to answer.

    Detached on Windows so the daemon outlives this launcher window -- closing
    the terminal that started NEXUS shouldn't take Ollama down with it.
    """
    from nexus import ollama

    exe = shutil.which("ollama")
    if not exe:
        return False, "`ollama` is not on PATH - install it from https://ollama.com/download"

    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([exe, "serve"], **kwargs)
    except OSError as exc:
        return False, f"could not launch `ollama serve` ({exc})"

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if ollama.fetch_names("tags") is not None:
            return True, "started"
        time.sleep(1.0)
    return False, f"started but did not answer within {wait_seconds}s"


def pull_model(model: str) -> bool:
    """Download one model, streaming Ollama's own progress to the console."""
    exe = shutil.which("ollama")
    if not exe:
        return _print(False, "`ollama` is not on PATH", "install from https://ollama.com/download")
    print(f"\n-- pulling {model} (this is a multi-GB download) --")
    code = subprocess.call([exe, "pull", model])
    return _print(code == 0, f"pull {model}" + ("" if code == 0 else " failed"))


def check_ollama(auto_start: bool = True, auto_pull: bool = False) -> bool:
    from nexus import config, ollama

    models = ollama.fetch_names("tags")
    if models is None and auto_start:
        print(f"{WAIT}Ollama is not running - starting it")
        ok, detail = start_ollama()
        if not ok:
            return _print(False, f"Ollama unreachable - {detail}", "ollama serve")
        models = ollama.fetch_names("tags")

    if models is None:
        return _print(False, f"Ollama not reachable at {config.OLLAMA_URL}", "ollama serve")

    if not models:
        if not auto_pull:
            # A model is several GB; downloading one is not something a
            # launcher should decide on the user's behalf.
            return _print(
                False,
                "Ollama is up but no models are pulled",
                f"ollama pull {config.DEFAULT_PULL_MODEL}   (or re-run with --pull)",
            )
        if not pull_model(config.DEFAULT_PULL_MODEL):
            return False
        models = ollama.fetch_names("tags") or []

    shown = ", ".join(models[:4]) + (" ..." if len(models) > 4 else "")
    return _print(True, f"Ollama reachable, {len(models)} model(s): {shown}")


def check_documents() -> bool:
    from nexus import config, ingest

    files = ingest.scan(config.DOCS_DIR) if config.DOCS_DIR.is_dir() else {}
    return _print(
        bool(files),
        f"documents/: {len(files)} file(s)" if files else "documents/ is empty",
        f"add {', '.join(sorted(ingest.SUPPORTED))} files to {config.DOCS_DIR}",
    )


def build_index(rebuild: bool = False) -> bool:
    from nexus import ingest

    print("\n-- indexing --")
    try:
        report = ingest.run(force_rebuild=rebuild, echo=lambda line: print(f"         {line}"))
    except Exception as exc:  # reported, with the fix, below
        return _print(False, f"indexing failed: {exc}", "python -m nexus.ingest --rebuild")
    if report.failed:
        return _print(False, f"{len(report.failed)} file(s) could not be read (see above)")
    return True


def check_index() -> bool:
    from nexus.retrieve import get_retriever

    count = get_retriever().stats()["chunks"]
    return _print(count > 0, f"index holds {count} chunk(s)", "python -m nexus.ingest --rebuild")


def check_router() -> bool:
    """Train the learned router on first start (seconds); otherwise report it."""
    from nexus import learned_router

    meta = learned_router.current_meta()
    if meta is None:
        print("\n-- training the router (first start) --")
        try:
            from train.train_router import main as train_main

            report = train_main([])
        except Exception as exc:
            return _print(False, f"router training failed: {exc}", "python run.py --train")
        meta = learned_router.current_meta()
        if meta is None:
            return _print(False, f"router {report['version']} didn't pass its gate; "
                                 "keyword rules stay in charge", "python -m nexus.evaluate_learned_router")
    golden = meta["metrics"]["golden"]
    return _print(True, f"learned router {meta['version']}: task accuracy "
                        f"{golden['new']['task_accuracy']:.0%} vs rules "
                        f"{golden['rules']['task_accuracy']:.0%} on held-out prompts")


def smoke_test() -> bool:
    """Prove the whole path works: retrieve -> prompt -> local model -> text."""
    from nexus.engine import Options, answer

    print("\n-- end-to-end test answer --")
    question = "What does this project use to store embeddings?"
    try:
        result = answer(question, options=Options(rag_mode="always"))
    except Exception as exc:  # this is the report
        return _print(False, f"generation failed: {exc}")
    text = (result.get("answer") or "").strip()
    if not text:
        return _print(False, "model returned an empty answer")
    print(f"         q: {question}")
    print(f"         model: {result['model']} ({result['task']}, {len(result['sources'])} sources)")
    print(f"         a: {text[:160]}{'...' if len(text) > 160 else ''}")
    return _print(True, f"answered in {result['elapsed']:.1f}s")


def preflight(
    rebuild: bool = False,
    with_smoke: bool = False,
    auto_start: bool = True,
    auto_pull: bool = False,
) -> tuple[bool, bool]:
    """Returns (can_start, everything_passed).

    Ollama being down is not fatal -- the PC-automation toolkit still works
    and the UI explains how to start it -- but it must not be reported as a
    clean bill of health either.
    """
    print("-- preflight --")
    if not (check_python() and check_dependencies()):
        return False, False
    from nexus import log

    log.setup()
    ollama_ok = check_ollama(auto_start=auto_start, auto_pull=auto_pull)
    docs_ok = check_documents()
    index_built = build_index(rebuild)
    index_ok = check_index()
    check_router()
    smoke_ok = smoke_test() if (with_smoke and ollama_ok and index_ok) else True
    can_start = index_ok
    return can_start, can_start and ollama_ok and docs_ok and index_built and smoke_ok


def run_tests() -> int:
    """The same checks CI runs."""
    status = 0
    if importlib.util.find_spec("ruff"):
        print("-- lint --")
        status |= subprocess.call([sys.executable, "-m", "ruff", "check", "."], cwd=PROJECT_DIR)
    else:
        print("-- lint skipped (pip install ruff) --")
    print("-- tests --")
    status |= subprocess.call(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=PROJECT_DIR
    )
    return status


def launch(server: bool) -> int:
    if server:
        print("\nStarting the plain HTTP UI on http://127.0.0.1:8000 (Ctrl+C to stop)")
        return subprocess.call([sys.executable, "-m", "nexus.server"], cwd=PROJECT_DIR)
    print(f"\nStarting NEXUS on http://127.0.0.1:{STREAMLIT_PORT} (Ctrl+C to stop)")
    return subprocess.call(
        [
            sys.executable, "-m", "streamlit", "run", str(PROJECT_DIR / "app.py"),
            "--server.address", "127.0.0.1",
            "--server.port", str(STREAMLIT_PORT),
            "--browser.gatherUsageStats", "false",
        ],
        cwd=PROJECT_DIR,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NEXUS AI end to end.")
    parser.add_argument("--check", action="store_true",
                        help="Preflight plus one real test answer, then exit.")
    parser.add_argument("--eval", action="store_true", help="Run the retrieval eval, then exit.")
    parser.add_argument("--test", action="store_true", help="Run lint and the test suite, then exit.")
    parser.add_argument("--server", action="store_true", help="Use the plain HTTP UI.")
    parser.add_argument("--rebuild", action="store_true", help="Force a full re-index.")
    parser.add_argument("--train", action="store_true",
                        help="Retrain the learned router from seed data and your feedback, then exit.")
    parser.add_argument("--no-serve", action="store_true",
                        help="Do not start Ollama automatically if it is not running.")
    parser.add_argument("--pull", action="store_true",
                        help="Download the default model if none is installed (multi-GB).")
    args = parser.parse_args()

    if args.test:
        return run_tests()
    if args.train:
        from train.train_router import main as train_main

        report = train_main([])
        return 0 if report["made_current"] else 1

    can_start, all_clear = preflight(
        rebuild=args.rebuild,
        with_smoke=args.check,
        auto_start=not args.no_serve,
        auto_pull=args.pull,
    )
    if not can_start:
        print("\nPreflight failed. Fix the items marked [!!] above and re-run.")
        return 1

    if args.eval:
        print()
        return subprocess.call([sys.executable, "-m", "nexus.evaluate"], cwd=PROJECT_DIR)

    if args.check:
        print("\nAll checks passed. Run `python run.py` to start the app." if all_clear
              else "\nSome checks failed; see the [!!] items above.")
        return 0 if all_clear else 1

    if not all_clear:
        print("\nStarting anyway, but the [!!] items above are degraded.")
    return launch(args.server)


if __name__ == "__main__":
    raise SystemExit(main())
