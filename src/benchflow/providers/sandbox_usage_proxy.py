"""Sandbox-local provider usage proxy runtime.

The host-side :class:`benchflow.trajectories.proxy.TrajectoryProxy` works when
the agent can route back to the host. Remote sandboxes such as Daytona cannot,
so this module starts a tiny byte-forwarding proxy inside the same sandbox
network namespace as the agent and imports its raw captures back into the host
trajectory model during cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.agents.registry import _NODE_INSTALL
from benchflow.trajectories.proxy import exchange_from_raw_capture
from benchflow.trajectories.types import Trajectory

logger = logging.getLogger(__name__)

_RUNTIME_ROOT = "/tmp/benchflow-usage-proxy"


def _read_node_proxy_source() -> str:
    return (
        Path(__file__).with_name("assets") / "sandbox_usage_proxy.js"
    ).read_text()


_NODE_PROXY_SOURCE = _read_node_proxy_source()
_NODE_LAUNCHER_SOURCE = r"""
const fs = require("fs");
const { spawn } = require("child_process");

const config = JSON.parse(process.env.BENCHFLOW_USAGE_PROXY_CONFIG || "{}");
const stdout = fs.openSync(config.stdout, "a");
const stderr = fs.openSync(config.stderr, "a");
const child = spawn(config.node, [config.script], {
  detached: true,
  stdio: ["ignore", stdout, stderr],
  env: { ...process.env, ...config.env },
});
child.unref();
console.log(child.pid);
"""


class SandboxUsageProxy:
    """Long-lived proxy process running in the agent sandbox."""

    def __init__(
        self,
        *,
        sandbox: Any,
        target: str,
        session_id: str,
        agent_name: str,
        prompt_cache_retention: str | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.target = target.rstrip("/")
        self.session_id = session_id
        self.agent_name = agent_name
        self.prompt_cache_retention = prompt_cache_retention
        self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
        self._token = uuid4().hex[:16]
        self._runtime_dir = f"{_RUNTIME_ROOT}/{self._token}"
        self._script_path = f"{self._runtime_dir}/proxy.js"
        self._state_path = f"{self._runtime_dir}/state.json"
        self._log_path = f"{self._runtime_dir}/captures.jsonl"
        self._pid_path = f"{self._runtime_dir}/proxy.pid"
        self._base_url: str | None = None

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("sandbox usage proxy has not started")
        return self._base_url

    async def start(self) -> None:
        await self._upload_proxy_script()
        node = await self._ensure_node()
        stdout_path = f"{_RUNTIME_ROOT}/{self._token}/stdout.log"
        stderr_path = f"{_RUNTIME_ROOT}/{self._token}/stderr.log"
        launcher_config = {
            "node": node,
            "script": self._script_path,
            "stdout": stdout_path,
            "stderr": stderr_path,
            "env": {
                "BENCHFLOW_USAGE_PROXY_TARGET": self.target,
                "BENCHFLOW_USAGE_PROXY_STATE_PATH": self._state_path,
                "BENCHFLOW_USAGE_PROXY_LOG_PATH": self._log_path,
                "BENCHFLOW_USAGE_PROXY_PID_PATH": self._pid_path,
                "BENCHFLOW_USAGE_PROXY_SESSION_ID": self.session_id,
                "BENCHFLOW_USAGE_PROXY_AGENT_NAME": self.agent_name,
                "BENCHFLOW_USAGE_PROXY_PROMPT_CACHE_RETENTION": (
                    self.prompt_cache_retention or ""
                ),
            },
        }
        command = " ".join(
            [
                "mkdir",
                "-p",
                shlex.quote(str(Path(self._script_path).parent)),
                "&&",
                "rm",
                "-f",
                shlex.quote(self._state_path),
                shlex.quote(self._log_path),
                shlex.quote(self._pid_path),
                "&&",
                f"BENCHFLOW_USAGE_PROXY_CONFIG={shlex.quote(json.dumps(launcher_config))}",
                shlex.quote(node),
                "-e",
                shlex.quote(_NODE_LAUNCHER_SOURCE),
            ]
        )
        result = await self.sandbox.exec(command, timeout_sec=15)
        if result.return_code != 0:
            raise RuntimeError(_exec_details("start sandbox usage proxy", result))
        state = await self._wait_for_state()
        self._base_url = f"http://127.0.0.1:{state['port']}"
        logger.info("Sandbox usage telemetry proxy listening on %s", self._base_url)

    async def is_running(self) -> bool:
        result = await self.sandbox.exec(
            (
                f"if [ -s {shlex.quote(self._pid_path)} ] && "
                f"kill -0 $(cat {shlex.quote(self._pid_path)}) 2>/dev/null; "
                "then echo yes; else echo no; fi"
            ),
            timeout_sec=5,
        )
        return result.return_code == 0 and (result.stdout or "").strip() == "yes"

    async def stop(self) -> None:
        try:
            await self._load_captures()
        except Exception as exc:
            logger.warning("Could not import sandbox usage captures: %s", exc)
        finally:
            await self._terminate()
            await self._cleanup_runtime_dir()

    async def _terminate(self) -> None:
        kill_cmd = (
            f"if [ -s {shlex.quote(self._pid_path)} ]; then "
            f"kill -TERM $(cat {shlex.quote(self._pid_path)}) 2>/dev/null || true; "
            "fi"
        )
        with contextlib.suppress(Exception):
            await self.sandbox.exec(kill_cmd, timeout_sec=10)

    async def _cleanup_runtime_dir(self) -> None:
        with contextlib.suppress(Exception):
            await self.sandbox.exec(
                f"rm -rf {shlex.quote(self._runtime_dir)}",
                timeout_sec=10,
            )

    async def _upload_proxy_script(self) -> None:
        parent = shlex.quote(str(Path(self._script_path).parent))
        result = await self.sandbox.exec(f"mkdir -p {parent}", timeout_sec=15)
        if result.return_code != 0:
            raise RuntimeError(_exec_details("prepare sandbox usage proxy dir", result))

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(_NODE_PROXY_SOURCE)
            tmp_path = Path(tmp.name)
        try:
            await self.sandbox.upload_file(tmp_path, self._script_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _ensure_node(self) -> str:
        node_probe = (
            "if [ -x /opt/benchflow/node/bin/node ]; then "
            "echo /opt/benchflow/node/bin/node; "
            "elif command -v node >/dev/null 2>&1; then command -v node; "
            "else echo ''; fi"
        )
        result = await self.sandbox.exec(node_probe, timeout_sec=10)
        node = (result.stdout or "").strip().splitlines()[-1:] or [""]
        if node[0]:
            return node[0]

        install = await self.sandbox.exec(_NODE_INSTALL, timeout_sec=300)
        if install.return_code != 0:
            raise RuntimeError(_exec_details("install Node for usage proxy", install))
        result = await self.sandbox.exec(node_probe, timeout_sec=10)
        node = (result.stdout or "").strip().splitlines()[-1:] or [""]
        if not node[0]:
            raise RuntimeError("Node.js was not available after usage proxy bootstrap")
        return node[0]

    async def _wait_for_state(self) -> dict[str, Any]:
        last_output = ""
        for _ in range(50):
            result = await self.sandbox.exec(
                f"cat {shlex.quote(self._state_path)} 2>/dev/null || true",
                timeout_sec=5,
            )
            last_output = (result.stdout or "").strip()
            if last_output:
                try:
                    state = json.loads(last_output)
                except (json.JSONDecodeError, ValueError):
                    await asyncio.sleep(0.2)
                    continue
                if int(state.get("port") or 0) > 0:
                    return state
            await asyncio.sleep(0.2)
        stderr = await self.sandbox.exec(
            f"cat {shlex.quote(f'{_RUNTIME_ROOT}/{self._token}/stderr.log')} "
            "2>/dev/null || true",
            timeout_sec=5,
        )
        raise RuntimeError(
            "sandbox usage proxy did not publish its state"
            f": {last_output or (stderr.stdout or '').strip()}"
        )

    async def _load_captures(self) -> None:
        capture_text = await self._read_capture_log()
        trajectory = Trajectory(session_id=self.session_id, agent_name=self.agent_name)
        for line in capture_text.splitlines():
            if not line.strip():
                continue
            try:
                trajectory.exchanges.append(exchange_from_raw_capture(json.loads(line)))
            except Exception as exc:
                logger.warning("Skipping malformed sandbox usage capture: %s", exc)
        self.trajectory = trajectory

    async def _read_capture_log(self) -> str:
        download_file = getattr(self.sandbox, "download_file", None)
        if download_file is not None:
            with tempfile.NamedTemporaryFile("r", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                await download_file(self._log_path, tmp_path)
                return tmp_path.read_text()
            except Exception as exc:
                logger.debug("Sandbox usage capture download failed: %s", exc)
            finally:
                tmp_path.unlink(missing_ok=True)

        result = await self.sandbox.exec(
            f"cat {shlex.quote(self._log_path)} 2>/dev/null || true",
            timeout_sec=15,
        )
        if result.return_code != 0:
            logger.warning("Could not read sandbox usage captures: %s", result.stderr)
            return ""
        return result.stdout or ""


def _exec_details(label: str, result: Any) -> str:
    stdout = (getattr(result, "stdout", "") or "").strip()
    stderr = (getattr(result, "stderr", "") or "").strip()
    details = [f"{label} failed with exit code {getattr(result, 'return_code', '?')}"]
    if stdout:
        details.append(f"stdout: {stdout[:1000]}")
    if stderr:
        details.append(f"stderr: {stderr[:1000]}")
    return "; ".join(details)
