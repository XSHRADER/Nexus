"""Point NEXUS at a throwaway directory before any test imports it.

Every test module imports this first (`import isolate`). `nexus.config` reads
its paths once, at import, so this has to run before the first `nexus`
import -- otherwise the suite writes routing decisions, logs and chats into
the user's real data/ folder, and reads their real documents, index,
settings and API keys.
"""

import atexit
import os
import shutil
import tempfile

if "NEXUS_TEST_ROOT" not in os.environ:
    root = tempfile.mkdtemp(prefix="nexus_tests_")
    os.environ["NEXUS_TEST_ROOT"] = root
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    for name, sub in (("NEXUS_DATA_DIR", "data"), ("NEXUS_DOCS_DIR", "documents"),
                      ("NEXUS_INDEX_DIR", "vector_store")):
        path = os.path.join(root, sub)
        os.makedirs(path, exist_ok=True)
        os.environ[name] = path
    os.environ.setdefault("NEXUS_DB", os.path.join(root, "data", "nexus.db"))
    # No personal settings, API keys or trained router leak into the tests.
    os.environ["NEXUS_CONFIG"] = os.path.join(root, "nexus.toml")
    os.environ["NEXUS_ENV_FILE"] = os.path.join(root, ".env")
    os.environ["NEXUS_MODELS_DIR"] = os.path.join(root, "models")
    # Nothing runs in the background unless a test starts it. Left on, the
    # UI tests would start the real watcher thread, which then opens whatever
    # database is current once a minute -- by then another test's temp file,
    # which Windows refuses to delete while it is open.
    with open(os.environ["NEXUS_CONFIG"], "w", encoding="utf-8") as fh:
        fh.write("[brain]\nenabled = false\n")
