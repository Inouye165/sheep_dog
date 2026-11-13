"""Tests for non-blocking backend startup.

The API server binds its port immediately and restores training state on a background
thread. While restoring, /api/health answers 503 "starting", manager-backed routes answer a
fast 503 BACKEND_STARTING, and manager-free routes keep working.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sheepdog import server as server_module
from sheepdog.server import ThreadingHTTPServer, TrainingRequestHandler


@pytest.fixture(autouse=True)
def _reset_startup_state() -> Iterator[None]:
    server_module._MANAGER_STARTUP.finish()
    server_module._MANAGER_STARTUP.error = None
    yield
    server_module._MANAGER_STARTUP.finish()
    server_module._MANAGER_STARTUP.error = None


def _request_full(
    port: int, path: str, method: str = "GET"
) -> tuple[int, dict[str, Any], dict[str, str]]:
    data = b"{}" if method == "POST" else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), dict(err.headers)


def _request(port: int, path: str, method: str = "GET") -> tuple[int, dict[str, Any]]:
    status, body, _headers = _request_full(port, path, method)
    return status, body


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met before timeout")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_requests_are_gated_while_manager_is_restoring(tmp_path) -> None:
    manager = MagicMock()
    manager.snapshot.return_value = {"running": False, "phase": "idle"}
    history_path = tmp_path / "training_history.json"
    history_path.write_text("[1, 2]", encoding="utf-8")
    mock_cfg = MagicMock()
    mock_cfg.training.output_dir = str(tmp_path)

    with (
        patch.object(TrainingRequestHandler, "manager", manager),
        patch("sheepdog.server.LabConfig", return_value=mock_cfg),
        patch.object(BaseHTTPRequestHandler, "log_message") as access_log,
    ):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
        port = httpd.server_port
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            server_module._MANAGER_STARTUP.begin()
            server_module._MANAGER_STARTUP.set_step("restoring active run state")

            # Liveness: the process is up, so 200 with progress details.
            status, body = _request(port, "/api/health")
            assert status == 200
            assert body["status"] == "starting"
            assert body["ready"] is False
            assert body["step"] == "restoring active run state"

            # Readiness: not usable yet, so 503 with Retry-After.
            status, body, headers = _request_full(port, "/api/ready")
            assert status == 503
            assert body["code"] == "BACKEND_STARTING"
            assert headers.get("Retry-After") == "2"

            status, body, headers = _request_full(port, "/api/training/status")
            assert status == 503
            assert headers.get("Retry-After") == "2"
            assert body["code"] == "BACKEND_STARTING"
            assert body["phase"] == "restoring"
            assert "restoring active run state" in body["error"]
            manager.snapshot.assert_not_called()

            status, body = _request(port, "/api/training/start", method="POST")
            assert status == 503
            assert body["code"] == "BACKEND_STARTING"
            manager.start.assert_not_called()

            # Manager-free routes still work during restore.
            status, body = _request(port, "/api/training/history")
            assert status == 200
            assert body == [1, 2]

            # Probes and expected startup 503s must not flood the access log.
            access_log.assert_not_called()

            server_module._MANAGER_STARTUP.finish()

            for path in ("/api/health", "/api/ready"):
                status, body = _request(port, path)
                assert status == 200
                assert body == {"status": "ok", "service": "sheepdog-api", "ready": True}

            status, body = _request(port, "/api/training/status")
            assert status == 200
            assert body["phase"] == "idle"
            manager.snapshot.assert_called_once()
        finally:
            httpd.shutdown()
            httpd.server_close()


class _CapturingServer(server_module._ExclusiveThreadingHTTPServer):
    instances: list[_CapturingServer] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _CapturingServer.instances.append(self)


def test_main_binds_port_before_restore_finishes() -> None:
    release = threading.Event()
    constructed = threading.Event()

    class SlowManager:
        def __init__(self) -> None:
            constructed.set()
            assert release.wait(timeout=10), "test never released the fake restore"
            self.runtime_tracker = MagicMock()
            self.snapshot = MagicMock(return_value={"running": False, "phase": "idle"})

    _CapturingServer.instances.clear()
    port = _free_port()
    with (
        patch("sheepdog.server.TrainingManager", SlowManager),
        patch("sheepdog.server._ExclusiveThreadingHTTPServer", _CapturingServer),
        # Other tests may replace the class-level descriptor; pin a fresh one here.
        patch.object(TrainingRequestHandler, "manager", server_module._ManagerDescriptor()),
        patch.object(TrainingRequestHandler, "_manager_instance", None, create=True),
    ):
        main_thread = threading.Thread(target=server_module.main, kwargs={"port": port}, daemon=True)
        main_thread.start()
        try:
            _wait_for(lambda: constructed.is_set() and bool(_CapturingServer.instances))

            # Port answers while the restore is still blocked.
            status, body = _request(port, "/api/health")
            assert status == 200
            assert body["status"] == "starting"
            assert body["ready"] is False
            status, body = _request(port, "/api/ready")
            assert status == 503
            status, body = _request(port, "/api/training/status")
            assert status == 503
            assert body["code"] == "BACKEND_STARTING"

            release.set()
            _wait_for(lambda: _request(port, "/api/ready")[0] == 200)
            status, body = _request(port, "/api/training/status")
            assert status == 200
            assert body["phase"] == "idle"
        finally:
            release.set()
            for httpd in _CapturingServer.instances:
                httpd.shutdown()
            main_thread.join(timeout=10)

        assert not main_thread.is_alive()
        manager = TrainingRequestHandler._manager_instance
        assert isinstance(manager, SlowManager)
        manager.runtime_tracker.end_session.assert_called_once_with("server_shutdown")


def test_main_exits_nonzero_when_restore_fails() -> None:
    class BrokenManager:
        def __init__(self) -> None:
            raise RuntimeError("corrupt run-state")

    outcome: dict[str, Any] = {}

    def run_main(port: int) -> None:
        try:
            server_module.main(port=port)
            outcome["exit"] = None
        except SystemExit as exc:
            outcome["exit"] = exc.code

    port = _free_port()
    with (
        patch("sheepdog.server.TrainingManager", BrokenManager),
        patch.object(TrainingRequestHandler, "manager", server_module._ManagerDescriptor()),
        patch.object(TrainingRequestHandler, "_manager_instance", None, create=True),
    ):
        main_thread = threading.Thread(target=run_main, args=(port,), daemon=True)
        main_thread.start()
        main_thread.join(timeout=15)

    assert not main_thread.is_alive(), "server kept running after a failed restore"
    assert outcome["exit"] == 1
    assert "corrupt run-state" in (server_module._MANAGER_STARTUP.error or "")


@pytest.mark.skipif(os.name != "nt", reason="SO_EXCLUSIVEADDRUSE is Windows-only")
def test_exclusive_server_refuses_to_share_port() -> None:
    first = server_module._ExclusiveThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
    try:
        port = first.server_port
        with pytest.raises(OSError):
            server_module._ExclusiveThreadingHTTPServer(("127.0.0.1", port), TrainingRequestHandler)
    finally:
        first.server_close()


def test_server_queues_dashboard_burst_while_accept_loop_is_busy() -> None:
    httpd = server_module._ExclusiveThreadingHTTPServer(
        ("127.0.0.1", 0), TrainingRequestHandler
    )
    clients: list[socket.socket] = []
    try:
        # Do not run serve_forever: a busy accept loop must still allow TCP handshakes.
        for _ in range(24):
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            clients.append(client)
            client.settimeout(1.0)
            client.connect(("127.0.0.1", httpd.server_port))
    finally:
        for client in clients:
            client.close()
        httpd.server_close()
