"""
run.py
One entry point that takes NEXUS from a fresh clone to an answering app.

    python run.py            preflight -> index -> Streamlit UI
    python run.py --check    preflight only, report and exit
    python run.py --eval     preflight -> index -> retrieval eval, then exit
    python run.py --server   launch the dependency-free HTTP UI instead
    python run.py --rebuild  force a full re-index first

Every step prints pass/fail and the exact command that fixes a failure, so a
broken setup says what is wrong instead of stack-tracing out of Streamlit.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Preflight is a progress report, so it has to appear as it happens. Python
# block-buffers stdout when it isn't a terminal, which hid every check
# behind the Streamlit banner when the output was piped or redirected.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover - older/odd streams
    pass

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "documents"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
DEFAULT_PULL_MODEL = "llama3.1:8b"

OK = "  [ok]  "
BAD = "  [!!]  "


def _print(good: bool, message: str, fix: str = "") -> bool:
    print(f"{OK if good else BAD}{message}")
    if not good and fix:
        print(f"         fix: {fix}")
    return good


def check_dependencies() -> bool:
    missing = []
    for module, package in [
        ("chromadb", "chromadb"),
        ("rank_bm25", "rank-bm25"),
        ("sentence_transformers", "sentence-transformers"),
        ("requests", "requests"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return _print(
        not missing,
        f"dependencies installed" if not missing else f"missing: {', '.join(missing)}",
        "pip install -r requirements.txt",
    )


def _ollama_models(timeout: float = 3.0):
    """Installed model names, or None when the daemon isn't reachable."""
    import requests

    try:
        response = requests.get(OLLAMA_TAGS, timeout=timeout)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return None


def start_ollama(wait_seconds: int = 45) -> tuple[bool, str]:
    """Launch `ollama serve` in the background and wait for it to answer.

    Detached on Windows so the daemon outlives this launcher window -- closing
    the terminal that started NEXUS shouldn't take Ollama down with it.
    """
    exe = shutil.which("ollama")
    if not exe:
        return False, "`ollama` is not on PATH - install it from https://ollama.com/download"

    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True

    try:
        subprocess.Popen([exe, "serve"], **kwargs)
    except Exception as exc:
        return False, f"could not launch `ollama serve` ({exc})"

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _ollama_models() is not None:
            return True, "started"
        time.sleep(1.0)
    return False, f"started but did not answer within {wait_seconds}s"


def pull_model(model: str = DEFAULT_PULL_MODEL) -> bool:
    """Download one model, streaming Ollama's own progress to the console."""
    exe = shutil.which("ollama")
    if not exe:
        return _print(False, "`ollama` is not on PATH", "install from https://ollama.com/download")
    print(f"\n-- pulling {model} (this is a multi-GB download) --")
    code = subprocess.call([exe, "pull", model])
    return _print(code == 0, f"pull {model}" + ("" if code == 0 else " failed"))


def check_ollama(auto_start: bool = True, auto_pull: bool = False) -> bool:
    models = _ollama_models()

    if models is None and auto_start:
        print("  [..]  Ollama is not running - starting it")
        ok, detail = start_ollama()
        if not ok:
            return _print(False, f"Ollama unreachable - {detail}", "ollama serve")
        models = _ollama_models()

    if models is None:
        return _print(False, "Ollama not reachable on :11434", "ollama serve")

    if not models:
        if auto_pull:
            if not pull_model():
                return False
            models = _ollama_models() or []
        else:
            # A model is several GB; downloading one is not something a
            # launcher should decide on the user's behalf.
            return _print(
                False,
                "Ollama is up but no models are pulled",
                f"ollama pull {DEFAULT_PULL_MODEL}   (or re-run with --pull)",
            )

    shown = ", ".join(models[:4]) + (" ..." if len(models) > 4 else "")
    return _print(True, f"Ollama reachable, {len(models)} model(s): {shown}")

def check_documents() -> bool:
    supported = {".txt", ".md", ".pdf", ".docx"}
    files = [p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in supported]
    return _print(
        bool(files),
        f"documents/: {len(files)} file(s)" if files else "documents/ is empty",
        f"add .txt/.md/.pdf/.docx files to {DOCS_DIR}",
    )


def build_index(rebuild: bool = False) -> bool:
    import ingest

    print("\n-- indexing --")
    try:
        ingest.main(force_rebuild=rebuild)
    except Exception as exc:
        return _print(False, f"indexing failed: {exc}")
    return True


def check_index() -> bool:
    from retrieve import Retriever

    retriever = Retriever()
    count = retriever._indexed_count
    return _print(
        count > 0,
        f"index holds {count} chunk(s)",
        "python ingest.py --rebuild",
    )


def smoke_test() -> bool:
    """Prove the whole path works: retrieve -> prompt -> local model -> text."""
    from engine import answer

    print("\n-- end-to-end smoke test --")
    question = "What does this project use to store embeddings?"
    try:
        result = answer(question)
    except Exception as exc:
        return _print(False, f"generation failed: {exc}")
    text = (result.get("answer") or "").strip()
    if not text:
        return _print(False, "model returned an empty answer")
    print(f"         q: {question}")
    print(f"         model: {result['model']} ({result['task']}, rag={result['needs_rag']})")
    print(f"         a: {text[:160]}{'...' if len(text) > 160 else ''}")
    return _print(True, "end-to-end answer produced")


def preflight(
    rebuild: bool = False,
    with_smoke: bool = True,
    auto_start: bool = True,
    auto_pull: bool = False,
) -> tuple[bool, bool]:
    """Returns (can_start, everything_passed).

    Ollama being down is not fatal -- the PC-automation toolkit still works
    and the UI explains how to start it -- but it must not be reported as a
    clean bill of health either.
    """
    print("-- preflight --")
    deps_ok = check_dependencies()
    ollama_ok = check_ollama(auto_start=auto_start, auto_pull=auto_pull)
    docs_ok = check_documents()
    if not deps_ok:
        return False, False

    build_index(rebuild)
    index_ok = check_index()

    if with_smoke and ollama_ok and index_ok:
        smoke_test()

    can_start = deps_ok and index_ok
    return can_start, can_start and ollama_ok and docs_ok


def launch(server: bool) -> int:
    if server:
        print("\nStarting the plain HTTP UI on http://127.0.0.1:8000 (Ctrl+C to stop)")
        return subprocess.call([sys.executable, str(PROJECT_DIR / "server.py")])
    print("\nStarting the Streamlit UI on http://127.0.0.1:8501 (Ctrl+C to stop)")
    return subprocess.call(
        [
            sys.executable, "-m", "streamlit", "run", str(PROJECT_DIR / "app.py"),
            "--server.address", "127.0.0.1",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NEXUS AI end to end.")
    parser.add_argument("--check", action="store_true", help="Preflight only, then exit.")
    parser.add_argument("--eval", action="store_true", help="Run the retrieval eval, then exit.")
    parser.add_argument("--server", action="store_true", help="Use the plain HTTP UI.")
    parser.add_argument("--rebuild", action="store_true", help="Force a full re-index.")
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Do not start Ollama automatically if it is not running.",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help=f"Download {DEFAULT_PULL_MODEL} if no model is installed (multi-GB).",
    )
    args = parser.parse_args()

    can_start, all_clear = preflight(
        rebuild=args.rebuild,
        with_smoke=not args.eval,
        auto_start=not args.no_serve,
        auto_pull=args.pull,
    )
    if not can_start:
        print(chr(10) + "Preflight failed. Fix the items marked [!!] above and re-run.")
        return 1
    if not all_clear:
        print(chr(10) + "Starting anyway, but the [!!] items above are degraded.")

    if args.eval:
        print()
        return subprocess.call([sys.executable, str(PROJECT_DIR / "eval_rag.py")])

    if args.check:
        if all_clear:
            print(chr(10) + "All checks passed. Run `python run.py` to start the app.")
        return 0 if all_clear else 1

    return launch(args.server)


if __name__ == "__main__":
    raise SystemExit(main())
