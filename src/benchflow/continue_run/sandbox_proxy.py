"""Sandbox-local replay proxy for ``benchflow continue``.

Remote sandboxes such as Daytona cannot reach a host-local replay proxy at
``127.0.0.1``. For those environments the continuation stack runs entirely
inside the sandbox:

    OpenHands -> sandbox replay proxy -> sandbox LiteLLM proxy -> provider

The host uploads the recorded exchanges and a small stdlib-only proxy script,
then downloads the live suffix after the rollout finishes so the normal stitched
``llm_trajectory.jsonl`` artifact remains identical in shape to host replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import tempfile
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.sandbox.files import upload_private_text
from benchflow.trajectories.redaction import canonical_redaction_module_source
from benchflow.trajectories.types import LLMExchange

SANDBOX_REPLAY_ROOT = "/tmp/benchflow-replay"
DEFAULT_SANDBOX_REPLAY_PORT = 61357
_CAPTURE_STATE_WRITE_FAILED = "BENCHFLOW_CAPTURE_STATE_WRITE_FAILED"


def sandbox_replay_base_url(port: int = DEFAULT_SANDBOX_REPLAY_PORT) -> str:
    """Return the in-sandbox OpenAI-compatible replay endpoint."""
    return f"http://127.0.0.1:{port}/v1"


def sandbox_replay_runtime_source() -> str:
    """Read the real replay runtime module for sandbox deployment."""

    package = files("benchflow.continue_run")
    resource = package.joinpath(
        "resources",
        "sandbox_replay_runtime.py.txt",
    )
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    return package.joinpath("sandbox_replay_runtime.py").read_text(encoding="utf-8")


def _ordered_live_exchange_log(text: str) -> tuple[list[LLMExchange], int]:
    """Parse sandbox live rows and restore their assigned attempt order."""
    sequenced: list[tuple[int, LLMExchange]] = []
    malformed = 0
    seen: set[int] = set()
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            exchange = LLMExchange.model_validate_json(raw)
        except Exception:
            malformed += 1
            continue
        attempt = exchange.metadata.get("continuation_attempt")
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt <= 0
            or attempt in seen
        ):
            malformed += 1
            continue
        seen.add(attempt)
        sequenced.append((attempt, exchange))
    return [exchange for _, exchange in sorted(sequenced)], malformed


async def _read_remote_text(sandbox: Any, path: str, *, timeout_sec: int = 15) -> str:
    download_file = getattr(sandbox, "download_file", None)
    if download_file is not None:
        with tempfile.NamedTemporaryFile("r", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            await download_file(path, tmp_path)
            return tmp_path.read_text()
        except Exception:
            pass
        finally:
            tmp_path.unlink(missing_ok=True)
    result = await sandbox.exec(
        f"cat {shlex.quote(path)} 2>/dev/null || true",
        timeout_sec=timeout_sec,
    )
    return result.stdout or ""


@dataclass
class SandboxReplayProxy:
    """A replay proxy process running on sandbox loopback."""

    sandbox: Any
    runtime_dir: str
    port: int
    pid_path: str
    live_log_path: str
    state_path: str
    stdout_path: str
    stderr_path: str
    live_exchanges: list[LLMExchange] = field(default_factory=list)
    live_attempt_count: int = 0
    live_errors: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return sandbox_replay_base_url(self.port)

    @classmethod
    async def start(
        cls,
        *,
        sandbox: Any,
        recorded: list[LLMExchange],
        upstream_url: str,
        upstream_api_key: str,
        upstream_model: str,
        strict_divergence: bool = False,
        port: int = DEFAULT_SANDBOX_REPLAY_PORT,
    ) -> SandboxReplayProxy:
        token = uuid4().hex[:16]
        runtime_dir = f"{SANDBOX_REPLAY_ROOT}/{token}"
        paths = {
            "script": f"{runtime_dir}/replay_proxy.py",
            "redaction": f"{runtime_dir}/benchflow_trajectory_redaction.py",
            "config": f"{runtime_dir}/config.json",
            "state": f"{runtime_dir}/state.json",
            "pid": f"{runtime_dir}/replay.pid",
            "stdout": f"{runtime_dir}/stdout.log",
            "stderr": f"{runtime_dir}/stderr.log",
            "live_log": f"{runtime_dir}/live_exchanges.jsonl",
        }
        result = await sandbox.exec(
            f"mkdir -p {shlex.quote(runtime_dir)} && "
            f"chmod 700 {shlex.quote(runtime_dir)}",
            timeout_sec=20,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "prepare sandbox replay runtime failed: "
                f"{result.stderr or result.stdout}"
            )

        recorded_rows = [
            exchange.model_dump(mode="json", exclude_none=True) for exchange in recorded
        ]
        config = {
            "recorded": recorded_rows,
            "upstream_url": upstream_url,
            "upstream_api_key": upstream_api_key,
            "upstream_model": upstream_model,
            "strict_divergence": strict_divergence,
            "port": port,
            "state_path": paths["state"],
            "live_log_path": paths["live_log"],
        }
        await upload_private_text(
            sandbox,
            sandbox_replay_runtime_source(),
            paths["script"],
            suffix=".py",
        )
        await upload_private_text(
            sandbox,
            canonical_redaction_module_source(),
            paths["redaction"],
            suffix=".py",
        )
        await upload_private_text(
            sandbox,
            json.dumps(config),
            paths["config"],
            suffix=".json",
        )

        command = (
            f"rm -f {shlex.quote(paths['state'])} {shlex.quote(paths['pid'])} "
            f"{shlex.quote(paths['live_log'])}; "
            f"(python3 {shlex.quote(paths['script'])} {shlex.quote(paths['config'])} "
            f"> {shlex.quote(paths['stdout'])} 2> {shlex.quote(paths['stderr'])} & "
            f"echo $! > {shlex.quote(paths['pid'])})"
        )
        result = await sandbox.exec(command, timeout_sec=20)
        if result.return_code != 0:
            raise RuntimeError(
                f"start sandbox replay proxy failed: {result.stderr or result.stdout}"
            )
        proxy = cls(
            sandbox=sandbox,
            runtime_dir=runtime_dir,
            port=port,
            pid_path=paths["pid"],
            live_log_path=paths["live_log"],
            state_path=paths["state"],
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
        )
        try:
            await proxy._wait_until_ready()
        except BaseException:
            await proxy.stop()
            raise
        return proxy

    async def _wait_until_ready(self) -> None:
        probe = (
            "python3 - <<'PY'\n"
            "import urllib.request\n"
            f"urllib.request.urlopen('http://127.0.0.1:{self.port}/health', timeout=2).read()\n"
            "PY"
        )
        last = ""
        for _ in range(120):
            result = await self.sandbox.exec(probe, timeout_sec=5)
            if result.return_code == 0:
                return
            last = (result.stderr or result.stdout or "").strip()
            await asyncio.sleep(0.25)
        stderr = await _read_remote_text(self.sandbox, self.stderr_path, timeout_sec=5)
        raise RuntimeError(
            f"sandbox replay proxy did not become healthy: {last or stderr.strip()}"
        )

    async def stop(self) -> None:
        await self._quiesce()
        await self._terminate()
        self.live_exchanges = await self._load_live_exchanges()
        await self._load_live_state()
        await self._load_runtime_errors()
        with contextlib.suppress(Exception):
            await self.sandbox.exec(
                f"rm -rf {shlex.quote(self.runtime_dir)}",
                timeout_sec=10,
            )

    async def _quiesce(self) -> None:
        command = (
            "python3 - <<'PY'\n"
            "import urllib.request\n"
            "request = urllib.request.Request(\n"
            f"    'http://127.0.0.1:{self.port}/benchflow/quiesce',\n"
            "    data=b'{}', method='POST')\n"
            "urllib.request.urlopen(request, timeout=615).read()\n"
            "PY"
        )
        try:
            result = await self.sandbox.exec(command, timeout_sec=620)
        except Exception as exc:
            self.live_errors.append(f"sandbox replay quiesce failed: {exc}")
            return
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "unavailable").strip()
            self.live_errors.append(f"sandbox replay quiesce failed: {detail}")

    async def _terminate(self) -> None:
        command = (
            f"if [ -s {shlex.quote(self.pid_path)} ]; then "
            f"pid=$(cat {shlex.quote(self.pid_path)}); "
            'kill -TERM "$pid" 2>/dev/null || true; i=0; '
            'while kill -0 "$pid" 2>/dev/null; do '
            '[ "$i" -ge 100 ] && exit 1; i=$((i + 1)); sleep 0.1; done; fi'
        )
        try:
            result = await self.sandbox.exec(command, timeout_sec=15)
        except Exception as exc:
            self.live_errors.append(f"sandbox replay termination failed: {exc}")
            return
        if result.return_code != 0:
            self.live_errors.append("sandbox replay termination did not quiesce")

    async def _load_live_exchanges(self) -> list[LLMExchange]:
        text = await _read_remote_text(self.sandbox, self.live_log_path)
        exchanges, malformed = _ordered_live_exchange_log(text)
        if malformed:
            self.live_errors.append(
                f"{malformed} sandbox live exchange record(s) were malformed"
            )
        return exchanges

    async def _load_live_state(self) -> None:
        text = await _read_remote_text(self.sandbox, self.state_path)
        try:
            state = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            state = None
        if not isinstance(state, dict):
            self.live_errors.append("sandbox live capture state was unavailable")
            return
        attempt_count = state.get("live_attempt_count")
        error_count = state.get("live_error_count")
        if (
            not isinstance(attempt_count, int)
            or isinstance(attempt_count, bool)
            or attempt_count < 0
        ):
            self.live_errors.append("sandbox live attempt count was invalid")
            return
        self.live_attempt_count = attempt_count
        if (
            not isinstance(error_count, int)
            or isinstance(error_count, bool)
            or error_count < 0
        ):
            self.live_errors.append("sandbox live error count was invalid")
        elif error_count:
            self.live_errors.append(
                f"{error_count} sandbox live provider request(s) failed before capture"
            )
        if self.live_attempt_count != len(self.live_exchanges):
            self.live_errors.append(
                "sandbox live exchange recovery mismatch: "
                f"attempted {self.live_attempt_count}, "
                f"recovered {len(self.live_exchanges)}"
            )

    async def _load_runtime_errors(self) -> None:
        stderr = await _read_remote_text(self.sandbox, self.stderr_path)
        failures = stderr.count(_CAPTURE_STATE_WRITE_FAILED)
        if failures:
            self.live_errors.append(
                f"sandbox live attempt journal failed {failures} time(s)"
            )
