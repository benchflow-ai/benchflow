"""One episode's BenchFlow sandbox, driven through ``bridge.py`` in a BenchFlow venv.

``Bridge`` owns the child process and its JSON-lines pipe. ``BenchFlowSession``
adds what the episode needs on top: the ``run_bash`` and ``submit`` tools, the
wall-clock budget, whether the policy acted, and the one verification.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

BRIDGE_SCRIPT = Path(__file__).with_name("bridge.py")

# Extra seconds a reply may take beyond the work it waits for.
REPLY_SLACK_SEC = 60.0

# What the bridge inherits from this process: what BenchFlow needs to reach its
# sandbox backend, and nothing else. In particular no model keys, and none of this
# venv's Python settings, which would break imports in the BenchFlow venv.
BRIDGE_ENV_PREFIXES = ("DAYTONA_", "BENCHFLOW_", "DOCKER_", "LC_")
BRIDGE_ENV_NAMES = frozenset(
    {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TZ", "TMPDIR", "TERM",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
        "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    }
)


def bridge_environment(extra: tuple[str, ...] = (), source: dict[str, str] | None = None) -> dict[str, str]:
    """The allowlisted environment the bridge process starts with."""
    source = dict(os.environ if source is None else source)
    keep = set(BRIDGE_ENV_NAMES) | set(extra)
    return {
        name: value
        for name, value in source.items()
        if name in keep or name.startswith(BRIDGE_ENV_PREFIXES)
    }


class BridgeError(RuntimeError):
    """The bridge process failed: it died, broke the protocol, or stopped answering.

    This is infrastructure (the policy cannot reach the bridge process), unlike a
    failed command, which the policy may have caused."""


class Bridge:
    """One ``bridge.py session`` child process.

    The child gets an allowlisted copy of this process's environment
    (``bridge_environment``), which is how it gets ``DAYTONA_API_KEY`` and
    ``BENCHFLOW_DAYTONA_OWNER``: they are set on the trainer's launch, never written
    to a file here, and model keys are not passed on. Closing stdin makes the child
    close its sandbox; so does SIGTERM.
    """

    def __init__(
        self,
        python: str,
        *,
        idle_timeout_sec: float,
        log_path: Path | None,
        env_passthrough: tuple[str, ...] = (),
    ) -> None:
        self.python = python
        self.idle_timeout_sec = idle_timeout_sec
        self.log_path = log_path
        self.env_passthrough = env_passthrough
        self.process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._log_handle: Any = None

    async def open(self) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_path, "ab")
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.python,
                str(BRIDGE_SCRIPT),
                "session",
                "--idle-timeout",
                str(self.idle_timeout_sec),
                env=bridge_environment(self.env_passthrough),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self._log_handle if self._log_handle is not None else asyncio.subprocess.DEVNULL,
                # Its own process group: a signal meant for the env worker does not
                # reach it first, so it can close its sandbox in order.
                start_new_session=True,
                limit=64 * 1024 * 1024,
            )
        except OSError as exc:
            self._close_log()
            raise BridgeError(f"cannot start the BenchFlow bridge with {self.python!r}: {exc}") from exc

    async def call(self, request: dict[str, Any], *, timeout_sec: float) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise BridgeError("the bridge is not running")
        async with self._lock:
            if process.returncode is not None:
                raise BridgeError(f"the bridge exited with code {process.returncode}{self._log_tail()}")
            try:
                process.stdin.write((json.dumps(request) + "\n").encode())
                await process.stdin.drain()
                line = await asyncio.wait_for(process.stdout.readline(), timeout_sec)
            except asyncio.TimeoutError as exc:
                raise BridgeError(f"the bridge did not answer {request.get('op')!r} within {timeout_sec:g}s") from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise BridgeError(f"the bridge pipe broke during {request.get('op')!r}{self._log_tail()}") from exc
            if not line:
                await asyncio.sleep(0.2)
                raise BridgeError(
                    f"the bridge exited during {request.get('op')!r} (code {process.returncode}){self._log_tail()}"
                )
            try:
                reply = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BridgeError(f"the bridge sent a line that is not JSON: {line[:200]!r}") from exc
            if not isinstance(reply, dict):
                raise BridgeError(f"the bridge sent {type(reply).__name__}, not an object")
            return reply

    async def close(self, *, timeout_sec: float = 240.0) -> None:
        """Ask the bridge to close its sandbox; escalate to SIGTERM, then SIGKILL."""
        process = self.process
        if process is None:
            self._close_log()
            return
        try:
            if process.returncode is None:
                with contextlib.suppress(BridgeError):
                    await self.call({"op": "close"}, timeout_sec=timeout_sec)
            if process.stdin is not None:
                with contextlib.suppress(Exception):
                    process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 30)
            except asyncio.TimeoutError:
                # SIGTERM makes the bridge close its sandbox before it exits.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 200)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
        finally:
            self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.close()
            self._log_handle = None

    def _log_tail(self, limit: int = 800) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        with contextlib.suppress(OSError):
            data = self.log_path.read_bytes()[-limit:].decode(errors="replace").strip()
            return f"; bridge log: {data}" if data else ""
        return ""


class SandboxStartError(RuntimeError):
    """The sandbox never started, so the policy never acted: infrastructure."""

    def __init__(self, message: str, decision: dict[str, Any] | None) -> None:
        super().__init__(message)
        self.decision = decision


def truncate(text: str, limit: int) -> str:
    """Keep the head and the tail; the middle of a long output matters least."""
    if len(text) <= limit:
        return text
    marker = f"\n[... {len(text) - limit} characters truncated ...]\n"
    keep = max(0, limit - len(marker))
    head = keep // 2
    return text[:head] + marker + text[len(text) - (keep - head) :]


class BenchFlowSession:
    """One task's sandbox for one episode: the tools, the budget, the verdict."""

    def __init__(
        self,
        bridge: Bridge,
        *,
        task_dir: str,
        rollout_name: str,
        environment: str,
        sandbox_user: str | None,
        jobs_dir: str,
        bash_timeout_sec: int,
        max_output_chars: int,
        submit_path: str | None,
        agent_budget_sec: float | None,
        sandbox_setup_timeout_sec: int,
        verify_timeout_sec: float,
    ) -> None:
        self.bridge = bridge
        self.task_dir = task_dir
        self.rollout_name = rollout_name
        self.environment = environment
        self.sandbox_user = sandbox_user
        self.jobs_dir = jobs_dir
        self.bash_timeout_sec = bash_timeout_sec
        self.max_output_chars = max_output_chars
        self.submit_path = submit_path
        self.agent_budget_sec = agent_budget_sec
        self.sandbox_setup_timeout_sec = sandbox_setup_timeout_sec
        self.verify_timeout_sec = verify_timeout_sec
        self.started = False
        self.workspace: str | None = None
        self.rollout_dir: str | None = None
        self.policy_acted = False
        self.submitted = False
        self.agent_started_at: float | None = None
        self.decision: dict[str, Any] | None = None
        self.verify_result: dict[str, Any] | None = None
        # Set when the bridge itself fails mid-episode: the episode ends and is dropped.
        self.infra_error: str | None = None
        self.stats = {
            "bash_calls": 0,
            "bash_timeouts": 0,
            "bash_nonzero": 0,
            "exec_errors": 0,
            "transient_errors": 0,
        }
        self._verify_lock = asyncio.Lock()

    # --- lifecycle -------------------------------------------------------------------

    async def start(self) -> None:
        """Start the sandbox. Raises ``SandboxStartError`` or ``BridgeError``."""
        await self.bridge.open()
        reply = await self.bridge.call(
            {
                "op": "start",
                "task_dir": self.task_dir,
                "environment": self.environment,
                "sandbox_user": self.sandbox_user,
                "jobs_dir": self.jobs_dir,
                "rollout_name": self.rollout_name,
                "sandbox_setup_timeout": self.sandbox_setup_timeout_sec,
            },
            # Image builds and setup commands happen here.
            timeout_sec=self.sandbox_setup_timeout_sec + 900.0,
        )
        if not reply.get("ok"):
            raise SandboxStartError(str(reply.get("error") or "sandbox start failed"), reply.get("decision"))
        self.started = True
        self.workspace = reply.get("workspace")
        self.rollout_dir = reply.get("rollout_dir")

    def mark_agent_start(self) -> None:
        self.agent_started_at = time.monotonic()

    def over_budget(self) -> bool:
        if self.agent_budget_sec is None or self.agent_started_at is None:
            return False
        return time.monotonic() - self.agent_started_at >= self.agent_budget_sec

    async def close(self) -> None:
        await self.bridge.close()

    # --- the tools ---------------------------------------------------------------------

    async def _call(self, request: dict[str, Any], *, timeout_sec: float) -> dict[str, Any] | None:
        """A tool's bridge call; a failed bridge ends the episode as infrastructure."""
        try:
            return await self.bridge.call(request, timeout_sec=timeout_sec)
        except BridgeError as exc:
            self.infra_error = str(exc)
            return None

    def episode_over(self) -> str | None:
        if self.infra_error is not None:
            return "error: the sandbox bridge failed; the episode is over."
        if self.submitted:
            return "error: the answer was already submitted; the episode is over."
        if self.decision is not None:
            return "error: the episode is over."
        return None

    async def run_bash(self, command: str) -> str:
        if (over := self.episode_over()) is not None:
            return over
        command = str(command or "")
        if not command.strip():
            return "error: empty command"
        self.policy_acted = True
        self.stats["bash_calls"] += 1
        reply = await self._call(
            {"op": "bash", "command": command, "timeout_sec": self.bash_timeout_sec},
            timeout_sec=self.bash_timeout_sec + REPLY_SLACK_SEC,
        )
        if reply is None:
            return "error: the sandbox bridge failed; the episode is over."
        if not reply.get("ok"):
            self.stats["exec_errors"] += 1
            if reply.get("transient"):
                self.stats["transient_errors"] += 1
            return f"error: the sandbox could not run the command: {reply.get('error')}"
        output = str(reply.get("stdout") or "") + str(reply.get("stderr") or "")
        output = truncate(output, self.max_output_chars)
        return_code = reply.get("return_code")
        if reply.get("timed_out"):
            self.stats["bash_timeouts"] += 1
            return f"{output}\n[command timed out after {self.bash_timeout_sec} seconds]".lstrip("\n")
        if return_code not in (0, None):
            self.stats["bash_nonzero"] += 1
            return f"{output}\n[exit code {return_code}]".lstrip("\n")
        return output

    async def submit(self, answer: str = "") -> str:
        if (over := self.episode_over()) is not None:
            return over
        answer = "" if answer is None else str(answer)
        if answer and self.submit_path:
            self.policy_acted = True
            reply = await self._call(
                {"op": "write", "path": self.submit_path, "text": answer},
                timeout_sec=30 + REPLY_SLACK_SEC,
            )
            if reply is None:
                return "error: the sandbox bridge failed; the episode is over."
            if not reply.get("ok"):
                self.stats["exec_errors"] += 1
                return f"error: could not write the answer: {reply.get('error')}"
        self.submitted = True
        where = f" to {self.submit_path}" if answer and self.submit_path else ""
        return f"Submitted{where}. The episode is over."

    # --- the verdict --------------------------------------------------------------------

    async def verify(self) -> dict[str, Any]:
        """Run BenchFlow's verifier once and return the decision (reward or drop)."""
        async with self._verify_lock:
            if self.decision is not None:
                return self.decision
            if self.infra_error is not None:
                raise BridgeError(self.infra_error)
            reply = await self.bridge.call(
                {"op": "verify", "policy_acted": self.policy_acted},
                timeout_sec=self.verify_timeout_sec + REPLY_SLACK_SEC,
            )
            if not reply.get("ok"):
                raise BridgeError(f"verify failed: {reply.get('error')}")
            decision = reply.get("decision")
            if not isinstance(decision, dict) or "reason" not in decision:
                raise BridgeError(f"verify returned no decision: {str(reply)[:300]}")
            self.decision = decision
            self.verify_result = reply.get("result") or {}
            return decision
