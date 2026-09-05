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
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "documents"
OLLAMA_TAGS = "http://localhost:11434/api/tags"

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


def check_ollama() -> bool:
    import requests

    try:
        response = requests.get(OLLAMA_TAGS, timeout=4)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return _print(False, "Ollama not reachable on :11434", "ollama serve")
    if not models:
        return _print(False, "Ollama is up but no models are pulled", "ollama pull llama3.1:8b")
    return _print(True, f"Ollama reachable, {len(models)} model(s): {', '.join(models[:4])}"
                  + (" ..." if len(models) > 4 else ""))


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


def preflight(rebuild: bool = False, with_smoke: bool = True) -> tuple[bool, bool]:
    """Returns (can_start, everything_passed).

    Ollama being down is not fatal -- the PC-automation toolkit still works
    and the UI explains how to start it -- but it must not be reported as a
    clean bill of health either.
    """
    print("-- preflight --")
    deps_ok = check_dependencies()
    ollama_ok = check_ollama()
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
    args = parser.parse_args()

    can_start, all_clear = preflight(rebuild=args.rebuild, with_smoke=not args.eval)
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
