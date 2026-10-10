# NEXUS versions

Every version of NEXUS in one place: what has shipped and what is planned.
Details of shipped versions are in [CHANGELOG.md](CHANGELOG.md).

## All versions

| Version | Status | Date | Theme | What it adds |
|---|---|---|---|---|
| **0.1.0** | Released | 2026-09-06 | First version | Local document search (vector + keyword), task router, Streamlit UI, PC tools, one-command start |
| **0.2.0** | Released | 2026-09-20 | Hardening | Security fix for the local server, saved chats, conversation history, streaming with Stop |
| **0.3.0** | Released | 2026-10-06 | Restructure and UI | `nexus` package, central config, terminal chat, multipage dark UI, CI on Windows |
| **0.4.0** | Released | 2026-10-10 | Router and versions | Router trained on 285 examples (62% to 89% accuracy), version labels, changelog, release workflow |
| **1.0.0** | Next | | Complete local assistant | Rewritten README, refreshed demo documents and test questions, router learns from real questions |
| **1.1.0** | Planned | | Settings and memory | One settings file (`nexus.toml`), longer conversation memory |
| **1.2.0** | Planned | | Truth check | Each sentence of an answer is checked against your documents and marked supported, not found or contradicted |
| **1.3.0** | Planned | | Ratings and Arena | Rate answers, compare two models blind, personal model leaderboard |
| **1.4.0** | Planned | | Learned router | The router trains on your ratings and only replaces the rules when it measures better |
| **2.0.0** | Planned | | Optional cloud | Cloud models for hard questions (off by default), image and voice input |
| **2.1.0** | Planned | | Model council | Several models answer, a judge merges them and shows where they disagree |
| **2.2.0** | Planned | | Background brain | Watches your folders, keeps an inbox and daily digest, makes checked flashcards |

## Planned versions in detail

| Version | Goal | Done when | Starting point |
|---|---|---|---|
| **1.0.0** | NEXUS can be called complete as a local, private assistant | README describes the current app; demo documents and the 18 retrieval questions match the current project and the retrieval score is re-measured; the router log saves the question text so misroutes can become new examples; CI green on Ubuntu and Windows | Open items in `docs/PROGRESS.md` |
| **1.1.0** | Settings in one file instead of environment variables | `nexus.toml` is read at start, with an example file; long chats keep their earliest context | Phase 0 on `nexus-features` |
| **1.2.0** | You can see which parts of an answer your documents back up | Truth check runs in both UIs with evidence excerpts and a measured accuracy | Phase 2 on `nexus-features` |
| **1.3.0** | You can tell which model is best for you | Thumbs up/down are stored; Arena hides model names until you pick; leaderboard page | Phase 3 on `nexus-features` |
| **1.4.0** | Routing improves from your own use | Trained router beats the rule-based one on the held-out set before it is switched on | Phase 4 on `nexus-features` |
| **2.0.0** | Hard questions can use stronger models, if you allow it | Cloud is off by default; document questions and PC actions stay local unless allowed; daily limits per provider; image and voice input work | Phase 1 on `nexus-features` |
| **2.1.0** | A second opinion on questions that matter | Council runs on demand with an agreement score and a disagreement map | Phase 5 on `nexus-features` |
| **2.2.0** | NEXUS works while you are away | Folder watcher, inbox, digest and flashcards, all switchable off | Phase 6 on `nexus-features` |

The `nexus-features` branch already has working code for 1.1.0 to 2.2.0, but
on the old flat file layout. Each version above means porting one phase onto
the `nexus` package on `main`, with its tests.

## How versions are numbered

`MAJOR.MINOR.PATCH`, for example `1.2.0`.

| Part | Goes up when | Example |
|---|---|---|
| MAJOR | What NEXUS *is* changes, or old settings or data stop working | 2.0.0 lets questions leave the PC for the first time |
| MINOR | A new feature is added and everything old still works | 1.2.0 adds the truth check |
| PATCH | Only fixes, no new features | 1.2.1 fixes a crash |

Versions before 1.0.0 are development versions: usable, but not yet called
complete.

## How to release a version

1. Set the new number in `pyproject.toml` and `nexus/__init__.py`.
2. Add a section for it at the top of `CHANGELOG.md`, and move its row in the
   table above to "Released" with the date.
3. Commit, push `main`, and wait for the tests to pass on GitHub.
4. Tag and push the tag:

   ```bash
   git tag -a v1.0.0 -m "NEXUS 1.0.0"
   git push origin v1.0.0
   ```

The `release` workflow then checks that the tag, the code and the changelog
agree, and publishes the GitHub release with that changelog section as its
notes. `python -m nexus --version` prints the version you are running.
