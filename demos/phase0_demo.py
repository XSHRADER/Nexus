"""
demos/phase0_demo.py
Phase 0 in the terminal: conversation memory and saved chats.

    python demos/phase0_demo.py

Works against real Ollama (any chat model pulled) or demos/mock_ollama.py.
It writes to a throwaway database, so your real saved chats are untouched.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import Options, answer  # noqa: E402
from store import ChatStore  # noqa: E402

OPTS = Options(rag_mode="never", temperature=0.0)


def show(label: str, result: dict) -> None:
    print(f"  NEXUS ({result.get('model')}, remembers {result.get('history_used', 0)} "
          f"earlier messages):")
    for line in (result.get("answer") or "").splitlines():
        print(f"    {line}")
    print()


def ask(store: ChatStore, chat_id: int, question: str) -> dict:
    print(f"  You: {question}")
    result = answer(question, options=OPTS, history=store.get_messages(chat_id))
    store.add_message(chat_id, "user", question)
    store.add_message(chat_id, "assistant", result["answer"], {"model": result["model"]})
    show(question, result)
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ChatStore(Path(tmp) / "demo.db")

        print("\n=== 1. Before Phase 0: every question is sent alone ===\n")
        print("  You: What is its population?")
        show("", answer("What is its population?", options=OPTS, history=None))

        print("=== 2. After Phase 0: the same follow-up inside a conversation ===\n")
        chat = store.create_chat("What is the capital of France?")
        ask(store, chat, "What is the capital of France?")
        ask(store, chat, "What is its population?")

        print("=== 3. Chats are saved: reopen the database from disk ===\n")
        reopened = ChatStore(store.path)
        for c in reopened.list_chats():
            print(f"  chat #{c['id']}: {c['title']!r} · {c['message_count']} messages")
            for m in reopened.get_messages(c["id"]):
                first_line = m["content"].splitlines()[0]
                print(f"     [{m['role']}] {first_line}")
        print()


if __name__ == "__main__":
    main()
