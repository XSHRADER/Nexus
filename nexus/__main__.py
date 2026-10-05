"""
Terminal chat with NEXUS: `python -m nexus`.

Same engine as the UI -- routing, retrieval, model fallback -- with the answer
streamed to the console. Commands: /new clears the conversation, /quit exits.
"""

from __future__ import annotations

import argparse
import sys

from nexus import log
from nexus.engine import Options, answer, apply_pending


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m nexus", description="Chat with NEXUS in the terminal.")
    parser.add_argument("question", nargs="*", help="Ask one question and exit.")
    parser.add_argument("--model", help="Pin a model instead of choosing automatically.")
    parser.add_argument("--docs", choices=["auto", "always", "never"], default="auto",
                        help="Whether to use your documents (default: auto).")
    args = parser.parse_args()
    # Windows consoles default to cp1252, which can't print the emoji and
    # arrows models (and the PC tools) use.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    log.setup()
    options = Options(force_model=args.model, rag_mode=args.docs)

    history: list[dict[str, str]] = []

    def ask(question: str) -> None:
        streamed = []

        def on_token(tok: str) -> None:
            streamed.append(tok)
            print(tok, end="", flush=True)

        print("\nNEXUS: ", end="", flush=True)
        try:
            result = answer(question, options=options, on_token=on_token, history=history)
        except RuntimeError as exc:
            print(f"\n[!!] {exc}")
            return
        if not streamed:  # PC actions and "no model" messages aren't streamed
            print(result["answer"], end="")
        sources = sorted({s["source"] for s in result["sources"]})
        print(f"\n\n  [{result['model'] or 'no model'} · {result['task']} · {result['elapsed']:.1f}s"
              + (f" · sources: {', '.join(sources)}" if sources else "") + "]")
        if result.get("info"):
            print(f"  note: {result['info']}")
        if result["requires_confirmation"]:
            if input("  Apply these changes? [y/N] ").strip().lower() == "y":
                print(apply_pending(result["pending"])["answer"])
            else:
                print("  Cancelled — nothing on disk changed.")
        history.extend([{"role": "user", "content": question},
                        {"role": "assistant", "content": result["answer"]}])

    if args.question:
        ask(" ".join(args.question))
        return 0

    print("NEXUS — ask anything. /new starts over, /quit exits.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question in ("/quit", "/exit"):
            return 0
        if question == "/new":
            history.clear()
            print("(new conversation)")
            continue
        if question:
            try:
                ask(question)
            except KeyboardInterrupt:
                print("\n(stopped)")


if __name__ == "__main__":
    sys.exit(main())
