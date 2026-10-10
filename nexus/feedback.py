"""
feedback.py
What you tell NEXUS about its answers, kept as training data for the
learning router (Phase 4).

Three kinds of signal, from most to least explicit:

  * Arena battles -- two models answer the same prompt with their names
    hidden and you pick the better one (or a tie, or "both bad"). The
    strongest signal: a direct comparison on the same question.
  * Ratings -- 👍 / 👎 on any answer, with an optional reason.
  * Implicit signals -- asking another model to answer again (a quiet 👎
    for the first one), correcting the task NEXUS guessed, or forcing your
    documents on or off. Each says what you wanted without a rating.

The leaderboard turns battles into Elo ratings, per task and overall, the
way chess ratings work: beating a strong model gains more than beating a
weak one, and the ratings settle as battles accumulate.
"""

from __future__ import annotations

import random
import time
from typing import Any

from nexus.store import ChatStore, get_store

REASONS = ("wrong", "too slow", "too long", "off-topic", "unsafe", "other")
WINNERS = ("a", "b", "tie", "both_bad")
SIGNAL_KINDS = ("regenerate", "task_override", "rag_override")

ELO_START = 1000.0
ELO_K = 32.0
# Fewer decided battles than this and a rating is still moving a lot from
# game to game; the UIs mark it "settling" rather than letting a 3-0 start
# read as a verdict.
SETTLED_AFTER = 5


def _store(store: ChatStore | None) -> ChatStore:
    return store or get_store()


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------


def _question_before(store: ChatStore, message: dict[str, Any]) -> str | None:
    """The user turn an answer was replying to."""
    messages = store.get_messages(message["chat_id"])
    ids = [m["id"] for m in messages]
    if message["id"] not in ids:
        return None
    for m in reversed(messages[: ids.index(message["id"])]):
        if m["role"] == "user":
            return m["content"]
    return None


def rate(message_id: int, rating: int, reason: str | None = None,
         store: ChatStore | None = None) -> dict[str, Any]:
    """👍 (+1) or 👎 (-1) on an answer; 0 removes the rating."""
    store = _store(store)
    message = store.get_message(message_id)
    if message is None or message["role"] != "assistant":
        raise ValueError("answer not found")
    if rating not in (-1, 0, 1):
        raise ValueError("rating must be -1, 0 or 1")
    if reason is not None and reason not in REASONS:
        reason = "other"
    meta = message.get("meta") or {}
    with store._connect() as db:
        if rating == 0:
            db.execute("DELETE FROM feedback WHERE message_id = ?", (message_id,))
        else:
            db.execute(
                """
                INSERT INTO feedback (message_id, rating, reason, question, task, model,
                                      provider, complexity, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    rating = excluded.rating, reason = excluded.reason,
                    created_at = excluded.created_at
                """,
                (message_id, rating, reason if rating < 0 else None,
                 _question_before(store, message), meta.get("task"), meta.get("model"),
                 meta.get("provider"), meta.get("complexity"), time.time()),
            )
    store.update_meta(message_id, {"rating": rating or None,
                                   "rating_reason": reason if rating < 0 else None})
    return {"message_id": message_id, "rating": rating, "reason": reason if rating < 0 else None}


def ratings(store: ChatStore | None = None) -> list[dict[str, Any]]:
    with _store(store)._connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM feedback ORDER BY created_at")]


# ---------------------------------------------------------------------------
# Implicit signals
# ---------------------------------------------------------------------------


def record_signal(kind: str, question: str | None = None, task: str | None = None,
                  model: str | None = None, value: str | None = None,
                  message_id: int | None = None, store: ChatStore | None = None) -> None:
    if kind not in SIGNAL_KINDS:
        raise ValueError(f"unknown signal {kind!r}")
    with _store(store)._connect() as db:
        db.execute(
            "INSERT INTO signals (kind, question, task, model, value, message_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, question, task, model, value, message_id, time.time()),
        )


def correct_task(question: str, wrong: str | None, right: str,
                 message_id: int | None = None, store: ChatStore | None = None) -> None:
    """You said a question was routed as the wrong task.

    Stored as a `task_override` signal, the same thing the Task setting
    records, so the next router training learns from it (weighted 3x).
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("no question to correct")
    if right == wrong:
        raise ValueError("that is the task it was already routed as")
    record_signal("task_override", question, task=wrong, value=right,
                  message_id=message_id, store=store)


def signals(kind: str | None = None, store: ChatStore | None = None) -> list[dict[str, Any]]:
    query, args = "SELECT * FROM signals", ()
    if kind:
        query, args = query + " WHERE kind = ?", (kind,)
    with _store(store)._connect() as db:
        return [dict(r) for r in db.execute(query + " ORDER BY id", args)]


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------


def pick_pair(chain: list[dict[str, Any]], rng: random.Random | None = None
              ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Two different models from a routing chain, in random A/B order.

    One side is NEXUS's own first choice -- that is the decision being
    tested. The other is drawn from the rest, weighted toward the next-best
    candidates but able to reach any of them, so models NEXUS rarely picks
    still get rated. If the chain mixes this PC and the cloud, the
    challenger comes from the other side when possible: "is cloud worth it
    for me?" is the most useful question Arena can answer.
    """
    rng = rng or random.Random()
    seen, unique = set(), []
    for c in chain:
        if c["model"] not in seen and c.get("provider") != "toolkit":
            seen.add(c["model"])
            unique.append(c)
    if len(unique) < 2:
        return None
    first, rest = unique[0], unique[1:]
    other_side = [c for c in rest if c.get("local", True) != first.get("local", True)]
    pool = other_side or rest
    weights = [1.0 / (i + 1) for i in range(len(pool))]
    challenger = rng.choices(pool, weights=weights, k=1)[0]
    pair = [first, challenger]
    rng.shuffle(pair)  # position must not give the answer away
    return pair[0], pair[1]


def create_battle(prompt: str, task: str | None, complexity: float | None,
                  a: dict[str, Any], b: dict[str, Any], chat_id: str | None = None,
                  store: ChatStore | None = None) -> int:
    """`a`/`b`: {"model", "provider", "answer"} as actually produced."""
    with _store(store)._connect() as db:
        cur = db.execute(
            """
            INSERT INTO battles (chat_id, prompt, task, complexity, model_a, provider_a,
                                 answer_a, model_b, provider_b, answer_b, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chat_id, prompt, task, complexity, a["model"], a.get("provider"), a.get("answer"),
             b["model"], b.get("provider"), b.get("answer"), time.time()),
        )
        return int(cur.lastrowid)


def get_battle(battle_id: int, store: ChatStore | None = None) -> dict[str, Any] | None:
    with _store(store)._connect() as db:
        row = db.execute("SELECT * FROM battles WHERE id = ?", (battle_id,)).fetchone()
    return dict(row) if row else None


def vote(battle_id: int, winner: str, store: ChatStore | None = None) -> dict[str, Any]:
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}")
    store = _store(store)
    battle = get_battle(battle_id, store)
    if battle is None:
        raise ValueError("battle not found")
    if battle["winner"] is not None:
        raise ValueError("this battle already has a vote")
    with store._connect() as db:
        db.execute("UPDATE battles SET winner = ?, decided_at = ? WHERE id = ?",
                   (winner, time.time(), battle_id))
    return get_battle(battle_id, store)


def battle_message(battle: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The answer a voted battle leaves in the chat, and its metadata.

    The chat continues with the answer you preferred (A on a tie), now with
    both model names revealed.
    """
    from nexus import providers

    meta: dict[str, Any] = {
        "task": battle["task"], "complexity": battle["complexity"],
        "arena": {"battle_id": battle["id"], "winner": battle["winner"],
                  "a": battle["model_a"], "b": battle["model_b"]},
    }
    if battle["winner"] == "both_bad":
        return "(Arena: both answers were marked bad.)", meta
    side = "b" if battle["winner"] == "b" else "a"
    model = battle[f"model_{side}"]
    spec = providers.spec_by_name(model)
    meta.update(model=model, provider=battle[f"provider_{side}"],
                local=spec.is_local if spec else True)
    return battle[f"answer_{side}"] or "", meta


def battles(decided_only: bool = True, store: ChatStore | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM battles" + (" WHERE winner IS NOT NULL" if decided_only else "")
    with _store(store)._connect() as db:
        return [dict(r) for r in db.execute(query + " ORDER BY COALESCE(decided_at, created_at), id")]


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo(battle_rows: list[dict[str, Any]], k: float = ELO_K) -> dict[str, float]:
    """Elo ratings from decided battles, in the order they were decided.

    A tie scores half a win each. "Both bad" says nothing about which is
    better, so it moves neither rating (it is still counted, below).
    """
    ratings_: dict[str, float] = {}
    for row in battle_rows:
        a, b, winner = row["model_a"], row["model_b"], row["winner"]
        ra, rb = ratings_.setdefault(a, ELO_START), ratings_.setdefault(b, ELO_START)
        if winner == "both_bad" or a == b:
            continue
        score_a = {"a": 1.0, "b": 0.0, "tie": 0.5}[winner]
        exp_a = expected_score(ra, rb)
        ratings_[a] = ra + k * (score_a - exp_a)
        ratings_[b] = rb + k * ((1.0 - score_a) - (1.0 - exp_a))
    return ratings_


def leaderboard(task: str | None = None, store: ChatStore | None = None) -> list[dict[str, Any]]:
    """One row per model: Elo, record, and 👍/👎 approval -- for one task or all."""
    store = _store(store)
    rows = [b for b in battles(store=store) if task in (None, b["task"])]
    ratings_ = elo(rows)
    table: dict[str, dict[str, Any]] = {}

    def entry(model: str, provider: str | None) -> dict[str, Any]:
        return table.setdefault(model, {
            "model": model, "provider": provider, "elo": ELO_START, "battles": 0,
            "wins": 0, "losses": 0, "ties": 0, "both_bad": 0, "thumbs_up": 0, "thumbs_down": 0,
        })

    for b in rows:
        for side, other in (("a", "b"), ("b", "a")):
            e = entry(b[f"model_{side}"], b[f"provider_{side}"])
            e["battles"] += 1
            if b["winner"] == side:
                e["wins"] += 1
            elif b["winner"] == other:
                e["losses"] += 1
            elif b["winner"] == "tie":
                e["ties"] += 1
            else:
                e["both_bad"] += 1
    for r in ratings(store):
        if task not in (None, r["task"]) or not r["model"]:
            continue
        e = entry(r["model"], r["provider"])
        e["thumbs_up" if r["rating"] > 0 else "thumbs_down"] += 1
    for model, e in table.items():
        e["elo"] = round(ratings_.get(model, ELO_START), 1)
        votes = e["thumbs_up"] + e["thumbs_down"]
        e["approval"] = round(e["thumbs_up"] / votes, 3) if votes else None
        decided = e["wins"] + e["losses"] + e["ties"]
        e["win_rate"] = round((e["wins"] + 0.5 * e["ties"]) / decided, 3) if decided else None
        e["settled"] = decided >= SETTLED_AFTER
    return sorted(table.values(), key=lambda e: (-e["elo"], -e["battles"], e["model"]))


def summary(store: ChatStore | None = None) -> dict[str, int]:
    """How much training data there is so far."""
    store = _store(store)
    with store._connect() as db:
        count = lambda q: int(db.execute(q).fetchone()[0])  # noqa: E731
        return {
            "battles": count("SELECT COUNT(*) FROM battles WHERE winner IS NOT NULL"),
            "ratings": count("SELECT COUNT(*) FROM feedback"),
            "signals": count("SELECT COUNT(*) FROM signals"),
        }
