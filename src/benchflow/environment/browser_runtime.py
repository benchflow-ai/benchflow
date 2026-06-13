"""Shared browser-environment runtime helpers.

Browser-oriented agent loops should not each reinvent fixture serving and task
prompt rewriting. This module owns the small browser-world setup needed by the
0.7 Browser Use/Stagehand slices: serve a task's local fixture over localhost,
return provider-honest metadata, and build the instruction an agent loop should
consume.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import re
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


@dataclass(frozen=True)
class BrowserEnvironmentReadiness:
    """Safe readiness snapshot for a browser environment handle."""

    status: str
    url: str | None
    entrypoint: str
    duration_sec: float
    http_status: int | None = None
    content_bytes: int | None = None
    content_sha256: str | None = None
    title_sha256: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ready", "not-applicable"}

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "ok": self.ok,
            "url": self.url,
            "entrypoint": self.entrypoint,
            "duration_sec": self.duration_sec,
        }
        if self.http_status is not None:
            payload["http_status"] = self.http_status
        if self.content_bytes is not None:
            payload["content_bytes"] = self.content_bytes
        if self.content_sha256 is not None:
            payload["content_sha256"] = self.content_sha256
        if self.title_sha256 is not None:
            payload["title_sha256"] = self.title_sha256
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class BrowserEnvironmentHandle:
    """Runtime handle for a prepared browser environment slice."""

    url: str | None
    fixture_dir: Path | None
    entrypoint: str

    @property
    def allowed_domains(self) -> list[str] | None:
        return ["127.0.0.1"] if self.url else None

    def check_readiness(self, *, timeout: float = 5.0) -> BrowserEnvironmentReadiness:
        """Fetch the served entrypoint and return scrubbed readiness metadata."""

        started = time.perf_counter()
        if self.url is None:
            return BrowserEnvironmentReadiness(
                status="not-applicable",
                url=None,
                entrypoint=self.entrypoint,
                duration_sec=round(time.perf_counter() - started, 6),
            )
        try:
            with urlopen(self.url, timeout=timeout) as response:
                body = response.read()
                http_status = int(response.getcode())
        except (OSError, URLError) as exc:
            return BrowserEnvironmentReadiness(
                status="error",
                url=self.url,
                entrypoint=self.entrypoint,
                duration_sec=round(time.perf_counter() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
            )

        title = _html_title(body)
        return BrowserEnvironmentReadiness(
            status="ready" if 200 <= http_status < 400 else "error",
            url=self.url,
            entrypoint=self.entrypoint,
            duration_sec=round(time.perf_counter() - started, 6),
            http_status=http_status,
            content_bytes=len(body),
            content_sha256=hashlib.sha256(body).hexdigest(),
            title_sha256=hashlib.sha256(title.encode()).hexdigest()
            if title is not None
            else None,
            error=None if 200 <= http_status < 400 else f"HTTP {http_status}",
        )

    def to_dict(
        self, *, readiness: BrowserEnvironmentReadiness | None = None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "adapter": "browser",
            "served_url": self.url,
            "entrypoint": self.entrypoint,
            "allowed_domains": self.allowed_domains,
        }
        if self.fixture_dir is not None:
            payload["fixture"] = "browser_fixture"
        if readiness is not None:
            payload["readiness"] = readiness.to_dict()
        return payload


@dataclass(frozen=True)
class BrowserRuntimeSession:
    """First-class runtime evidence writer for a browser environment slice."""

    handle: BrowserEnvironmentHandle
    readiness: BrowserEnvironmentReadiness

    @property
    def environment(self) -> dict[str, object]:
        return self.handle.to_dict(readiness=self.readiness)

    def task_instruction(self, *, prompt: str, expected: str | None) -> str:
        return browser_task_instruction(
            prompt=prompt,
            expected=expected,
            handle=self.handle,
        )

    def build_trace_artifact(
        self,
        *,
        framework: str,
        steps: list[object],
        final_result: str,
        screenshots_b64: list[str] | None = None,
        duration_sec: float | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        artifact: dict[str, object] = {
            "schema": "benchflow.browser-runtime-trace.v1",
            "framework": framework,
            "steps": steps,
            "environment": self.environment,
            "screenshots_b64": [
                screenshot for screenshot in (screenshots_b64 or []) if screenshot
            ],
            "final_result": final_result,
        }
        if duration_sec is not None:
            artifact["duration_sec"] = duration_sec
        if extra:
            artifact.update(extra)
        return artifact

    def write_trace_artifact(
        self,
        path: Path,
        *,
        framework: str,
        steps: list[object],
        final_result: str,
        screenshots_b64: list[str] | None = None,
        duration_sec: float | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        artifact = self.build_trace_artifact(
            framework=framework,
            steps=steps,
            final_result=final_result,
            screenshots_b64=screenshots_b64,
            duration_sec=duration_sec,
            extra=extra,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        return artifact


class _QuietServer(socketserver.TCPServer):
    allow_reuse_address = True


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@contextlib.contextmanager
def browser_environment_runtime(
    cwd: Path,
    *,
    fixture_subdir: str = "browser_fixture",
    entrypoint: str = "index.html",
):
    """Serve ``cwd/browser_fixture`` on localhost when it exists.

    Foreign browser benchmarks often ship local fixture files. Browser Use Agent
    and Stagehand block or struggle with direct ``file://`` navigation, so the
    environment adapter presents the fixture as a localhost browser world and
    hands the agent loop a URL plus metadata.
    """

    fixture_dir = cwd / fixture_subdir
    if not fixture_dir.is_dir():
        yield BrowserEnvironmentHandle(
            url=None,
            fixture_dir=None,
            entrypoint=entrypoint,
        )
        return

    def handler(*args: Any, **kwargs: Any) -> _QuietHandler:
        return _QuietHandler(*args, directory=str(fixture_dir), **kwargs)

    with _QuietServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = int(server.server_address[1])
            yield BrowserEnvironmentHandle(
                url=f"http://127.0.0.1:{port}/{entrypoint}",
                fixture_dir=fixture_dir,
                entrypoint=entrypoint,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)


@contextlib.contextmanager
def browser_runtime_session(
    cwd: Path,
    *,
    fixture_subdir: str = "browser_fixture",
    entrypoint: str = "index.html",
    require_ready: bool = False,
):
    """Open a browser environment session with readiness evidence attached."""

    with browser_environment_runtime(
        cwd,
        fixture_subdir=fixture_subdir,
        entrypoint=entrypoint,
    ) as handle:
        readiness = handle.check_readiness()
        if require_ready and (not readiness.ok or readiness.status != "ready"):
            raise RuntimeError(f"browser environment not ready: {readiness.to_dict()}")
        yield BrowserRuntimeSession(handle=handle, readiness=readiness)


def expected_from_prompt(text: str) -> str | None:
    """Extract the standard "exactly: ..." final-answer hint from a prompt."""

    match = re.search(r"exactly:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"')
    return None


def _html_title(body: bytes) -> str | None:
    match = re.search(
        rb"<title[^>]*>(.*?)</title>",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    raw = re.sub(rb"\s+", b" ", match.group(1)).strip()
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace")


def browser_task_instruction(
    *,
    prompt: str,
    expected: str | None,
    handle: BrowserEnvironmentHandle,
) -> str:
    """Build the task text an agent loop receives for this browser world."""

    if handle.url is None:
        return prompt
    if expected is None:
        return f"Open {handle.url} and complete the task:\n\n{prompt}"
    return (
        f"Open {handle.url} and report the page status. "
        f"Final answer must be exactly: {expected}"
    )
