import http.client
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "server-test.db")

from nexus import server  # noqa: E402  (NEXUS_DB must be set before the engine loads)

PREVIEW = {
    "answer": "would move 2 files",
    "requires_confirmation": True,
    "pending": {"op": "organize", "path": "C:/somewhere"},
}


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        server.PENDING.clear()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        sent = {"Content-Type": "application/json"}
        sent.update(headers or {})
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
        conn.request(method, path, body=body, headers=sent)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        return resp.status, dict(resp.getheaders()), payload


class GuardTests(ServerTestCase):
    def test_no_cross_origin_header(self):
        status, headers, _ = self.request("GET", "/api/models")
        self.assertEqual(status, 200)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_foreign_origin_is_rejected(self):
        with mock.patch.object(server, "answer") as answer:
            status, _, _ = self.request(
                "POST", "/api/chat", {"question": "hi"},
                headers={"Origin": "https://evil.example"},
            )
        self.assertEqual(status, 403)
        answer.assert_not_called()

    def test_own_origin_is_allowed(self):
        with mock.patch.object(server, "answer", return_value={"answer": "ok"}):
            status, _, payload = self.request(
                "POST", "/api/chat", {"question": "hi"},
                headers={"Origin": f"http://127.0.0.1:{self.port}"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload["answer"], "ok")

    def test_foreign_host_is_rejected(self):
        status, _, _ = self.request("GET", "/", headers={"Host": "evil.example:8000"})
        self.assertEqual(status, 403)

    def test_text_plain_post_is_rejected(self):
        with mock.patch.object(server, "answer") as answer:
            status, _, _ = self.request(
                "POST", "/api/chat", b'{"question": "sort my downloads"}',
                headers={"Content-Type": "text/plain"},
            )
        self.assertEqual(status, 415)
        answer.assert_not_called()

    def test_oversized_body_is_rejected(self):
        status, _, _ = self.request(
            "POST", "/api/chat", b"{}",
            headers={"Content-Length": str(server.MAX_BODY + 1)},
        )
        self.assertEqual(status, 413)

    def test_malformed_json_is_rejected(self):
        status, _, _ = self.request("POST", "/api/chat", b"{not json")
        self.assertEqual(status, 400)


class PendingTests(ServerTestCase):
    def test_pending_id_replaces_raw_pending_and_works_once(self):
        with mock.patch.object(server, "answer", return_value=dict(PREVIEW)):
            status, _, payload = self.request("POST", "/api/chat", {"question": "sort it"})
        self.assertEqual(status, 200)
        self.assertNotIn("pending", payload)
        pending_id = payload["pending_id"]

        with mock.patch.object(server, "apply_pending", return_value={"answer": "done"}) as apply:
            first = self.request("POST", "/api/apply", {"pending_id": pending_id})
            second = self.request("POST", "/api/apply", {"pending_id": pending_id})
        self.assertEqual(first[0], 200)
        self.assertEqual(first[2]["answer"], "done")
        self.assertEqual(second[0], 404)
        apply.assert_called_once()
        self.assertEqual(apply.call_args.args[0], PREVIEW["pending"])

    def test_unknown_pending_id(self):
        status, _, _ = self.request("POST", "/api/apply", {"pending_id": "nope"})
        self.assertEqual(status, 404)

    def test_pending_ids_expire(self):
        now = [0.0]
        pending = server.PendingStore(ttl=10, clock=lambda: now[0])
        pending_id = pending.put({"op": "organize", "path": "x"})
        now[0] = 11.0
        self.assertIsNone(pending.pop(pending_id))


class ConfirmRegressionTests(ServerTestCase):
    def test_confirm_flag_cannot_apply_a_change(self):
        # The original exploit: a cross-site text/plain POST with confirm=true.
        # Even as a well-formed same-origin JSON request, confirm must do nothing.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "empty").mkdir()
            status, _, payload = self.request(
                "POST", "/api/chat",
                {"question": f'remove empty folders in "{root}"', "confirm": True},
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["requires_confirmation"])
            self.assertIn("pending_id", payload)
            self.assertTrue((root / "empty").is_dir())


if __name__ == "__main__":
    unittest.main()
