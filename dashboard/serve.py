#!/usr/bin/env python3
"""Serve the BenchFlow v0.5 dashboard on http://localhost:8777.

Refreshes ``data.json`` (test results + rollout artifacts + the authored
sections) on startup, then refreshes again when the repo state changes or the
periodic live-data interval elapses.

Usage::

    LINEAR_API_KEY=... python dashboard/serve.py             # serve
    LINEAR_API_KEY=... python dashboard/serve.py --run-tests # re-run tests first
    LINEAR_API_KEY=... python dashboard/serve.py --port 9000 # pick a port
    python dashboard/serve.py --allow-missing-linear         # local UI dev only

Re-run ``python dashboard/generate.py`` any time for a manual refresh. The
served page also polls ``data.json`` and rerenders when the feed changes.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from functools import partial
from pathlib import Path
from typing import ClassVar

try:
    from dashboard.generate import resolve_dashboard_jobs_root
except ModuleNotFoundError:  # pragma: no cover - used when run as dashboard/serve.py
    from generate import resolve_dashboard_jobs_root  # type: ignore[no-redef]

DASH = Path(__file__).resolve().parent
ROOT = DASH.parent


def _git_bytes(args: list[str]) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _dashboard_jobs_root() -> Path:
    return resolve_dashboard_jobs_root(ROOT, DASH / "data.json")


def _data_json_exists() -> bool:
    return (DASH / "data.json").is_file()


def _mark_cached_data_refresh_error(message: str) -> None:
    data_path = DASH / "data.json"
    with contextlib.suppress(Exception):
        data = json.loads(data_path.read_text())
        if isinstance(data, dict):
            data["sync_error"] = message
            data["sync_failed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            data_path.write_text(json.dumps(data, indent=2))


def repo_fingerprint() -> str:
    digest = hashlib.sha256()
    for args in (
        ["rev-parse", "--short", "HEAD"],
        ["branch", "--show-current"],
        ["status", "--porcelain=v1", "-z"],
        ["diff", "--no-ext-diff", "--binary"],
        ["diff", "--cached", "--no-ext-diff", "--binary"],
    ):
        try:
            digest.update("\0".join(args).encode())
            digest.update(b"\0")
            digest.update(_git_bytes(args))
        except Exception as exc:
            digest.update(f"error:{exc}".encode())

    with contextlib.suppress(Exception):
        for raw in _git_bytes(
            ["ls-files", "--others", "--exclude-standard", "-z"]
        ).split(b"\0"):
            if not raw:
                continue
            path = ROOT / raw.decode()
            digest.update(raw)
            if path.is_file():
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
                if stat.st_size <= 1_000_000:
                    digest.update(path.read_bytes())
    return digest.hexdigest()


class SyncingDashboardHandler(http.server.SimpleHTTPRequestHandler):
    gen_cmd: ClassVar[list[str]] = []
    last_repo_fingerprint: ClassVar[str | None] = None
    last_refresh_at: ClassVar[float] = 0.0
    last_failed_refresh_at: ClassVar[float | None] = None
    failed_refresh_error: ClassVar[str | None] = None
    refresh_interval: ClassVar[float] = 60.0
    sync_lock: ClassVar[threading.Lock] = threading.Lock()

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/data.json" and not self._sync_data_json():
            return
        super().do_GET()

    def _sync_data_json(self) -> bool:
        if self._in_failure_backoff():
            if _data_json_exists():
                return True
            self.send_error(503, "data.json refresh failed; retrying after cooldown")
            return False

        current = repo_fingerprint()
        stale_by_time = (
            time.monotonic() - type(self).last_refresh_at >= type(self).refresh_interval
        )
        if current == type(self).last_repo_fingerprint and not stale_by_time:
            return True
        with type(self).sync_lock:
            if self._in_failure_backoff():
                if _data_json_exists():
                    return True
                self.send_error(
                    503, "data.json refresh failed; retrying after cooldown"
                )
                return False

            current = repo_fingerprint()
            stale_by_time = (
                time.monotonic() - type(self).last_refresh_at
                >= type(self).refresh_interval
            )
            if current == type(self).last_repo_fingerprint and not stale_by_time:
                return True
            reason = "scheduled refresh" if stale_by_time else "repo status changed"
            print(f"{reason}; refreshing data.json ...", flush=True)
            result = subprocess.run(type(self).gen_cmd, check=False)
            if result.returncode != 0:
                type(self).last_failed_refresh_at = time.monotonic()
                type(self).failed_refresh_error = (
                    "data.json refresh failed; serving last successful data.json"
                )
                if _data_json_exists():
                    _mark_cached_data_refresh_error(type(self).failed_refresh_error)
                    print(
                        "data.json refresh failed; serving last successful data.json",
                        flush=True,
                    )
                    return True
                self.send_error(
                    503,
                    "data.json refresh failed and no cached data.json is available",
                )
                return False
            type(self).last_repo_fingerprint = repo_fingerprint()
            type(self).last_refresh_at = time.monotonic()
            type(self).last_failed_refresh_at = None
            type(self).failed_refresh_error = None
            return True

    def _in_failure_backoff(self) -> bool:
        failed_at = type(self).last_failed_refresh_at
        if failed_at is None:
            return False
        return time.monotonic() - failed_at < type(self).refresh_interval


def main() -> int:
    argv = sys.argv[1:]
    port = 8777
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])

    gen = [sys.executable, str(DASH / "generate.py")]
    if "--run-tests" in argv:
        gen.append("--run-tests")
    if "--allow-missing-linear" in argv:
        gen.append("--allow-missing-linear")
    print("refreshing data.json ...", flush=True)
    result = subprocess.run(gen, check=False)
    if result.returncode != 0:
        if not _data_json_exists():
            return result.returncode
        _mark_cached_data_refresh_error(
            "initial data.json refresh failed; serving last successful data.json"
        )
        print(
            "initial data.json refresh failed; serving last successful data.json",
            flush=True,
        )

    SyncingDashboardHandler.gen_cmd = gen
    SyncingDashboardHandler.last_repo_fingerprint = repo_fingerprint()
    SyncingDashboardHandler.last_refresh_at = time.monotonic()
    SyncingDashboardHandler.refresh_interval = float(
        os.environ.get("DASHBOARD_REFRESH_SECONDS", "60")
    )
    handler = partial(SyncingDashboardHandler, directory=str(DASH))
    # quieter logs — one line per request is enough
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/"
        print(f"\n  BenchFlow dashboard → {url}")
        print("  Ctrl-C to stop.\n")
        with contextlib.suppress(Exception):
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
