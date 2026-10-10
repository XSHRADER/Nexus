# NEXUS versions

Version control for the `main` branch: every released version, what each one
contains, and what comes next. The detail of each release is in
[CHANGELOG.md](CHANGELOG.md).

## Released versions

Each row is a git tag on `main` and a GitHub release. The tests column is the
size of the test suite at that tag; every one passed on GitHub before tagging.

| Version | Status | Date | Tag | Tests | Theme |
|---|---|---|---|---|---|
| **0.1.0** | Released | 2026-09-06 | [v0.1.0](https://github.com/XSHRADER/Nexus/releases/tag/v0.1.0) | 73 | First version: local document search, task router, UI, PC tools |
| **0.2.0** | Released | 2026-09-20 | [v0.2.0](https://github.com/XSHRADER/Nexus/releases/tag/v0.2.0) | 133 | Hardening: server security fix, saved chats, history, streaming |
| **0.3.0** | Released | 2026-10-06 | [v0.3.0](https://github.com/XSHRADER/Nexus/releases/tag/v0.3.0) | 172 | Restructure: `nexus` package, terminal chat, multipage dark UI |
| **0.4.0** | Released | 2026-10-10 | [v0.4.0](https://github.com/XSHRADER/Nexus/releases/tag/v0.4.0) | 179 | Router trained on 285 examples; version labels and release workflow |
| **0.5.0** | Released | 2026-10-10 | [v0.5.0](https://github.com/XSHRADER/Nexus/releases/tag/v0.5.0) | 328 | Feature merge: cloud, truth check, Arena, learned router, council, brain |
| **1.0.0** | Released | 2026-10-10 | [v1.0.0](https://github.com/XSHRADER/Nexus/releases/tag/v1.0.0) | 329 | Complete local assistant: refreshed demo documents, routing log keeps questions |

## What each version contains

A filled cell is the version a capability arrived in; it is in every later
version too.

| Capability | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 1.0 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Answers from your documents (vector + keyword search, re-ranked) | ● | ● | ● | ● | ● | ● |
| Picks a local model per question | ● | ● | ● | ● | ● | ● |
| PC tools: organize, duplicates, large files, undo | ● | ● | ● | ● | ● | ● |
| Streamlit UI and a dependency-free web UI | ● | ● | ● | ● | ● | ● |
| Measured retrieval quality | ● | ● | ● | ● | ● | ● |
| Local server refuses requests from other websites |  | ● | ● | ● | ● | ● |
| Saved chats and conversation memory |  | ● | ● | ● | ● | ● |
| Streaming answers with Stop |  | ● | ● | ● | ● | ● |
| Uses installed models outside the catalogue |  | ● | ● | ● | ● | ● |
| Terminal chat (`python -m nexus`) |  |  | ● | ● | ● | ● |
| Multipage dark UI with live progress and per-answer details |  |  | ● | ● | ● | ● |
| One config module; tests isolated from your data; CI on Windows |  |  | ● | ● | ● | ● |
| Router trained on labelled examples, with a measured score |  |  |  | ● | ● | ● |
| Version labels, changelog, automatic GitHub releases |  |  |  | ● | ● | ● |
| Settings file (`nexus.toml`) and memory budget |  |  |  |  | ● | ● |
| Optional cloud models (off by default), with privacy rules |  |  |  |  | ● | ● |
| Image and voice input |  |  |  |  | ● | ● |
| Truth check against your documents |  |  |  |  | ● | ● |
| Ratings, Arena and a personal leaderboard |  |  |  |  | ● | ● |
| Learned router with a measured gate |  |  |  |  | ● | ● |
| Model council with a judge |  |  |  |  | ● | ● |
| Background brain: folder watcher, inbox, digest, flashcards |  |  |  |  | ● | ● |

## Measured at each version

| Measure | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| Tests | 73 | 133 | 172 | 179 | 328 | 329 |
| Routing accuracy, rules and examples (150 held-out prompts) | — | — | 62.0% | 89.3% | 89.3% | 89.3% |
| Routing accuracy, learned router (100 held-out prompts) | — | — | — | — | 99.0% | 99.0% |
| Retrieval: answer found in the top 5 passages (18 questions) | 94.4% | 94.4% | 94.4% | 94.4% | 94.4% | 88.9% |
| Retrieval: answer found in the top 10 passages, the app's default | — | — | — | — | — | 100% |
| Truth check accuracy, given the passage (60 labelled claims) | — | — | — | — | 85.0% | 83.3% |

The 1.0 retrieval and truth-check figures are measured on the refreshed demo
documents, with two test questions and seven claims updated to match, so they
are not a like-for-like drop from 0.5.

"—" means the measurement did not exist yet. The two routing rows use
different test sets, so they are not directly comparable; on the 100-prompt
set the rules and examples score 91.0%.

## Next versions

| Version | Status | Goal | Done when |
|---|---|---|---|
| **1.1.0** | Next | Cloud is no longer experimental | Each cloud provider has been tried with a real key, including an image and a voice question; the model names in `cloud_models.toml` are confirmed current |
| **1.2.0** | Idea | Learns from real use | The learned router's "needs a strong model" head trains on your own Arena votes once there are enough; misrouted questions can be turned into examples from the UI |
| **1.3.0** | Idea | Scales to a bigger library | Retrieval measured on a corpus large enough for the numbers to discriminate; indexing speed measured |

Ideas are not commitments; they are listed so they are not lost.

## How versions are numbered

`MAJOR.MINOR.PATCH`, for example `1.2.0`.

| Part | Goes up when | Example |
|---|---|---|
| MAJOR | Old settings or saved data stop working, or what NEXUS is changes | 2.0.0 if the database had to be rebuilt |
| MINOR | A new feature is added and everything old still works | 0.5.0 added cloud models, off by default |
| PATCH | Only fixes, no new features | 1.0.1 would fix a crash |

Versions before 1.0.0 were development versions. From 1.0.0 the local
assistant is called complete; cloud models stay marked experimental until
1.1.0.

## Branches

| Branch | What it is |
|---|---|
| `main` | The released line. Every tag above is on it. Tests must pass on GitHub before a tag. |
| anything else | Work in progress. It reaches `main` by a merge once its tests pass there. |

`nexus-features` was the branch the 0.5.0 features were built on. It is fully
merged into `main` and kept only as history.

## How to release a version

1. Set the new number in `pyproject.toml` and `nexus/__init__.py`.
2. Add a section for it at the top of `CHANGELOG.md`, and add its row and
   column to the tables above.
3. Commit, push `main`, and wait for the tests to pass on GitHub.
4. Tag and push the tag:

   ```bash
   git tag -a v1.0.0 -m "NEXUS 1.0.0"
   git push origin v1.0.0
   ```

The `release` workflow then checks that the tag, the code and the changelog
agree, and publishes the GitHub release with that changelog section as its
notes. `python -m nexus --version` prints the version you are running.
