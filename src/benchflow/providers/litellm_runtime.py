"""LiteLLM proxy runtime orchestration for host and sandbox runs."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import uuid4

import httpx
import yaml

from benchflow.agents.codex_config import apply_codex_provider_config
from benchflow.agents.env import uses_native_subscription_auth
from benchflow.agents.registry import AGENTS
from benchflow.providers.litellm_bedrock_preflight import (
    BEDROCK_PATCH_PREFLIGHT_SOURCE,
    BedrockPatchPreflightError,
    preflight_host_bedrock_patch,
    preflight_sandbox_bedrock_patch,
    route_requires_bedrock_patch,
)
from benchflow.providers.litellm_config import (
    LITELLM_MASTER_KEY_ENV,
    LITELLM_MODEL_ALIAS_ENV,
    LITELLM_MODEL_VIA_ENV,
    LiteLLMRoute,
    litellm_proxy_config,
    resolve_litellm_route,
    strip_provider_prefix,
)
from benchflow.providers.litellm_logging import (
    callback_module_source,
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)
from benchflow.sandbox.providers import OFF_BOX_MODEL_PROVIDERS
from benchflow.trajectories.types import Trajectory
from benchflow.usage_tracking import UsageTrackingConfig, usage_unavailable

logger = logging.getLogger(__name__)

LITELLM_VERSION_SPEC = "litellm[proxy]==1.89.0"
LITELLM_SANDBOX_ROOT = "/tmp/benchflow-litellm"
_CALLBACK_MODULE = "benchflow_litellm_callback"
_PATCH_MODULE = "benchflow_litellm_bedrock_patch"

# The proxy is an internal single-route gateway — it must never register the
# FastAPI Swagger docs route. litellm's `_get_docs_url()` honours an inherited
# `DOCS_URL` env *before* `NO_DOCS`, so a stray non-"/" `DOCS_URL` baked into a
# sandbox base image makes litellm call `add_route(DOCS_URL, ...)` and crash with
# `Routed paths must start with '/'` at startup. Force `DOCS_URL=""` (falsy, so
# it's ignored) and `NO_DOCS=true` so the docs route is skipped regardless of the
# inherited environment.
_PROXY_DOCS_DISABLE_ENV = {"DOCS_URL": "", "NO_DOCS": "true"}

# Agents that speak a provider-native wire protocol the LiteLLM proxy does not
# expose on its OpenAI/Anthropic surfaces. Routing them through the proxy would
# silently mis-wire the agent (e.g. the Gemini CLI speaks Google's
# GenerateContent format), so they talk to their provider directly and report
# usage_source='unavailable'. ``oracle`` has no model at all.
_NATIVE_PROTOCOL_AGENTS = frozenset({"oracle", "gemini"})
# Providers whose model traffic exits the sandbox to the host proxy (≡ non-docker);
# derived from the canonical registry so it can't drift from the provider set.
_SANDBOX_LOCAL_ENVIRONMENTS = OFF_BOX_MODEL_PROVIDERS


@dataclass(frozen=True)
class LiteLLMEndpoint:
    """Endpoint visible to the agent plus host-local endpoint for health checks."""

    agent_base_url: str
    local_base_url: str


class LiteLLMProcess:
    """Common interface for a running LiteLLM proxy."""

    route: LiteLLMRoute
    trajectory: Trajectory | None

    @property
    def base_url(self) -> str:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def is_running(self) -> bool:
        raise NotImplementedError


class HostLiteLLMProcess(LiteLLMProcess):
    def __init__(
        self,
        *,
        route: LiteLLMRoute,
        process: subprocess.Popen[bytes],
        runtime_dir: Path,
        endpoint: LiteLLMEndpoint,
        log_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        session_id: str,
        agent_name: str,
    ) -> None:
        self.route = route
        self.process = process
        self.runtime_dir = runtime_dir
        self.endpoint = endpoint
        self.log_path = log_path
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.session_id = session_id
        self.agent_name = agent_name
        self.trajectory: Trajectory | None = None

    @property
    def base_url(self) -> str:
        return self.endpoint.agent_base_url

    async def is_running(self) -> bool:
        return self.process.poll() is None

    async def stop(self) -> None:
        await _await_log_stable(self._log_size)
        if self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, 10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait, 10)
        self._load_callback_log()
        with contextlib.suppress(Exception):
            shutil.rmtree(self.runtime_dir, ignore_errors=True)

    def _log_size(self) -> int:
        try:
            return self.log_path.stat().st_size
        except OSError:
            return -1

    def _load_callback_log(self) -> None:
        if not self.log_path.exists():
            self.trajectory = Trajectory(
                session_id=self.session_id,
                agent_name=self.agent_name,
            )
            return
        self.trajectory = trajectory_from_litellm_callback_log(
            self.log_path.read_text(),
            session_id=self.session_id,
            agent_name=self.agent_name,
        )

    def log_tail(self) -> str:
        chunks: list[str] = []
        for label, path in (("stdout", self.stdout_path), ("stderr", self.stderr_path)):
            with contextlib.suppress(Exception):
                text = path.read_text()[-4000:]
                if text.strip():
                    chunks.append(f"{label}: {text.strip()}")
        return "\n".join(chunks)


class SandboxLiteLLMProcess(LiteLLMProcess):
    def __init__(
        self,
        *,
        sandbox: Any,
        route: LiteLLMRoute,
        runtime_dir: str,
        endpoint: LiteLLMEndpoint,
        log_path: str,
        pid_path: str,
        stdout_path: str,
        stderr_path: str,
        session_id: str,
        agent_name: str,
    ) -> None:
        self.sandbox = sandbox
        self.route = route
        self.runtime_dir = runtime_dir
        self.endpoint = endpoint
        self.log_path = log_path
        self.pid_path = pid_path
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.session_id = session_id
        self.agent_name = agent_name
        self.trajectory: Trajectory | None = None

    @property
    def base_url(self) -> str:
        return self.endpoint.agent_base_url

    async def is_running(self) -> bool:
        result = await self.sandbox.exec(
            (
                f"if [ -s {shlex.quote(self.pid_path)} ] && "
                f"kill -0 $(cat {shlex.quote(self.pid_path)}) 2>/dev/null; "
                "then echo yes; else echo no; fi"
            ),
            timeout_sec=5,
        )
        return result.return_code == 0 and (result.stdout or "").strip() == "yes"

    async def stop(self) -> None:
        await _await_log_stable(self._remote_log_size)
        with contextlib.suppress(Exception):
            await self.sandbox.exec(
                (
                    f"if [ -s {shlex.quote(self.pid_path)} ]; then "
                    f"kill -TERM $(cat {shlex.quote(self.pid_path)}) 2>/dev/null || true; "
                    "fi"
                ),
                timeout_sec=10,
            )
        await self._load_callback_log()
        with contextlib.suppress(Exception):
            await self.sandbox.exec(
                f"rm -rf {shlex.quote(self.runtime_dir)}", timeout_sec=10
            )

    async def _remote_log_size(self) -> int:
        with contextlib.suppress(Exception):
            result = await self.sandbox.exec(
                f"stat -c %s {shlex.quote(self.log_path)} 2>/dev/null || echo -1",
                timeout_sec=5,
            )
            return int((result.stdout or "-1").strip() or -1)
        return -1

    async def _load_callback_log(self) -> None:
        text = ""
        download_file = getattr(self.sandbox, "download_file", None)
        if download_file is not None:
            with tempfile.NamedTemporaryFile("r", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                await download_file(self.log_path, tmp_path)
                text = tmp_path.read_text()
            except Exception as exc:
                logger.debug("LiteLLM callback download failed: %s", exc)
            finally:
                tmp_path.unlink(missing_ok=True)
        if not text:
            result = await self.sandbox.exec(
                f"cat {shlex.quote(self.log_path)} 2>/dev/null || true",
                timeout_sec=15,
            )
            if result.return_code == 0:
                text = result.stdout or ""
        self.trajectory = trajectory_from_litellm_callback_log(
            text,
            session_id=self.session_id,
            agent_name=self.agent_name,
        )

    async def log_tail(self) -> str:
        chunks: list[str] = []
        for label, path in (("stdout", self.stdout_path), ("stderr", self.stderr_path)):
            with contextlib.suppress(Exception):
                result = await self.sandbox.exec(
                    f"tail -c 4000 {shlex.quote(path)} 2>/dev/null || true",
                    timeout_sec=5,
                )
                text = (result.stdout or "").strip()
                if text:
                    chunks.append(f"{label}: {text}")
        return "\n".join(chunks)


def needs_litellm_runtime(agent: str, model: str | None) -> bool:
    """True when an agent/model pair should be routed through LiteLLM."""
    return bool(model) and agent not in _NATIVE_PROTOCOL_AGENTS


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _await_log_stable(
    get_size: Callable[[], int | Awaitable[int]],
    *,
    deadline_s: float = 12.0,
    quiet_s: float = 0.5,
) -> None:
    """Wait until the callback JSONL stops growing.

    LiteLLM invokes its CustomLogger hooks fire-and-forget (``asyncio.create_task``)
    *after* returning the provider response, so the final — often largest —
    exchange may still be mid-append when BenchFlow tears the proxy down. A fixed
    sleep either truncates that record (undercounting usage while silently passing
    the ``required`` gate) or wastes time. Poll the log size until it has been
    stable for ``quiet_s`` (the final record landed) or ``deadline_s`` elapses.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    last_size = -2
    stable_since: float | None = None
    while loop.time() - start < deadline_s:
        raw: Any = get_size()
        if inspect.isawaitable(raw):
            raw = await raw
        size = cast(int, raw)
        if size > 0 and size == last_size:
            if stable_since is None:
                stable_since = loop.time()
            elif loop.time() - stable_since >= quiet_s:
                return
        else:
            stable_since = None
        last_size = size
        await asyncio.sleep(0.2)


def _host_litellm_executable() -> str:
    sibling = Path(sys.executable).with_name("litellm")
    if sibling.exists():
        return str(sibling)
    found = shutil.which("litellm")
    if found:
        return found
    raise RuntimeError(
        f"LiteLLM CLI is not installed. Install {LITELLM_VERSION_SPEC} in this environment."
    )


def _docker_host_address() -> str:
    import platform

    if platform.system().lower() != "linux":
        return "host.docker.internal"
    try:
        out = subprocess.check_output(
            [
                "docker",
                "network",
                "inspect",
                "bridge",
                "--format",
                "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
            ],
            text=True,
            timeout=10,
        ).strip()
        if out:
            return out
    except Exception:
        logger.debug("Could not detect Docker bridge gateway", exc_info=True)
    return "host.docker.internal"


def _host_bind_address(environment: str) -> str:
    """Interface the host LiteLLM proxy binds to.

    Local runs need no off-box reachability -> loopback only. Docker runs must be
    reachable from the agent container, so bind the concrete bridge-gateway IP
    (Linux) — reachable from containers but not the public network — falling back
    to 0.0.0.0 only when the gateway resolves to a hostname (e.g.
    host.docker.internal on macOS) that cannot be bound directly.
    """
    if environment != "docker":
        return "127.0.0.1"
    address = _docker_host_address()
    try:
        socket.inet_aton(address)
    except OSError:
        return "0.0.0.0"
    return address


def _agent_endpoint_for_environment(
    port: int, environment: str, bind: str
) -> LiteLLMEndpoint:
    if environment == "docker":
        agent_host = bind if bind != "0.0.0.0" else _docker_host_address()
        health_host = "127.0.0.1" if bind == "0.0.0.0" else bind
    else:
        agent_host = health_host = "127.0.0.1"
    return LiteLLMEndpoint(
        agent_base_url=f"http://{agent_host}:{port}",
        local_base_url=f"http://{health_host}:{port}",
    )


def _write_runtime_files(
    runtime_dir: Path,
    *,
    config: dict[str, object],
) -> tuple[Path, Path, Path]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    callback_path = runtime_dir / f"{_CALLBACK_MODULE}.py"
    patch_path = runtime_dir / f"{_PATCH_MODULE}.py"
    sitecustomize_path = runtime_dir / "sitecustomize.py"
    config_path = runtime_dir / "config.yaml"
    callback_path.write_text(callback_module_source())
    patch_source = Path(__file__).with_name("litellm_bedrock_patch.py").read_text()
    patch_path.write_text(patch_source)
    sitecustomize_path.write_text(f"import {_PATCH_MODULE}\n")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, callback_path, patch_path


async def _poll_host_health(process: HostLiteLLMProcess) -> None:
    last_error = ""
    for _ in range(120):
        if process.process.poll() is not None:
            raise RuntimeError(
                "LiteLLM exited before becoming healthy.\n" + process.log_tail()
            )
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                for path in ("/health/liveliness", "/health"):
                    response = await client.get(process.endpoint.local_base_url + path)
                    if response.status_code < 500:
                        return
        except Exception as exc:
            last_error = str(exc)
        await asyncio.sleep(0.25)
    raise RuntimeError(
        f"LiteLLM did not become healthy: {last_error}\n{process.log_tail()}"
    )


async def _start_host_litellm(
    *,
    route: LiteLLMRoute,
    master_key: str,
    agent_env: dict[str, str],
    environment: str,
    session_id: str,
    agent_name: str,
) -> HostLiteLLMProcess:
    runtime_dir = Path(tempfile.mkdtemp(prefix="benchflow-litellm-"))
    log_path = runtime_dir / "callback.jsonl"
    stdout_path = runtime_dir / "stdout.log"
    stderr_path = runtime_dir / "stderr.log"
    port = _find_free_port()
    bind = _host_bind_address(environment)
    config = litellm_proxy_config(route, master_key=master_key)
    config_path, _, _ = _write_runtime_files(runtime_dir, config=config)
    env = dict(os.environ)
    env.update(agent_env)
    env.update(
        {
            "PYTHONPATH": f"{runtime_dir}{os.pathsep}{env.get('PYTHONPATH', '')}",
            "LITELLM_MASTER_KEY": master_key,
            "BENCHFLOW_LITELLM_LOG_PATH": str(log_path),
            **_PROXY_DOCS_DISABLE_ENV,
        }
    )
    litellm_executable = _host_litellm_executable()
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    try:
        process = subprocess.Popen(
            [
                litellm_executable,
                "--config",
                str(config_path),
                "--host",
                bind,
                "--port",
                str(port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
    finally:
        stdout.close()
        stderr.close()
    runner = HostLiteLLMProcess(
        route=route,
        process=process,
        runtime_dir=runtime_dir,
        endpoint=_agent_endpoint_for_environment(port, environment, bind),
        log_path=log_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        session_id=session_id,
        agent_name=agent_name,
    )
    try:
        await _poll_host_health(runner)
        if route_requires_bedrock_patch(route):
            # Fail closed before the agent launches when the Bedrock 4.8+
            # thinking patch did not activate (#602).
            await asyncio.to_thread(
                preflight_host_bedrock_patch,
                env=env,
                litellm_executable=litellm_executable,
            )
    except BaseException:
        # A proxy that never became healthy still holds provider credentials and
        # an on-disk config with the master key — tear it down, don't leak it.
        with contextlib.suppress(Exception):
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(process.wait, 5)
                if process.poll() is None:
                    process.kill()
        with contextlib.suppress(Exception):
            shutil.rmtree(runtime_dir, ignore_errors=True)
        raise
    logger.info("LiteLLM proxy listening on %s", runner.base_url)
    return runner


def _sandbox_launcher_source() -> str:
    return r"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

# Read launch config from a file (argv[1]), never the command line: provider
# keys live in cfg["env"], and a shared sandbox exposes exec argv via /proc.
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])

env = os.environ.copy()
env.update(cfg["env"])
stdout = open(cfg["stdout"], "ab")
stderr = open(cfg["stderr"], "ab")
cmd = [
    cfg["litellm"],
    "--config",
    cfg["config"],
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
]
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=stdout,
    stderr=stderr,
    env=env,
    start_new_session=True,
)
with open(cfg["pid"], "w", encoding="utf-8") as handle:
    handle.write(str(proc.pid))
with open(cfg["state"], "w", encoding="utf-8") as handle:
    json.dump({"pid": proc.pid, "port": port}, handle)
print(json.dumps({"pid": proc.pid, "port": port}))
"""


async def _upload_text(sandbox: Any, text: str, target_path: str, suffix: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        await sandbox.upload_file(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)


async def _upload_runtime_files_to_sandbox(
    sandbox: Any,
    *,
    runtime_dir: str,
    config: dict[str, object],
) -> dict[str, str]:
    paths = {
        "config": f"{runtime_dir}/config.yaml",
        "callback": f"{runtime_dir}/{_CALLBACK_MODULE}.py",
        "patch": f"{runtime_dir}/{_PATCH_MODULE}.py",
        "sitecustomize": f"{runtime_dir}/sitecustomize.py",
        "launcher": f"{runtime_dir}/launcher.py",
        "stdout": f"{runtime_dir}/stdout.log",
        "stderr": f"{runtime_dir}/stderr.log",
        "log": f"{runtime_dir}/callback.jsonl",
        "pid": f"{runtime_dir}/litellm.pid",
        "state": f"{runtime_dir}/state.json",
        "venv": f"{runtime_dir}/venv",
        "launch_config": f"{runtime_dir}/launch_config.json",
        "preflight": f"{runtime_dir}/bedrock_patch_preflight.py",
    }
    result = await sandbox.exec(f"mkdir -p {shlex.quote(runtime_dir)}", timeout_sec=20)
    if result.return_code != 0:
        raise RuntimeError(_exec_details("prepare LiteLLM runtime directory", result))
    await _upload_text(
        sandbox, yaml.safe_dump(config, sort_keys=False), paths["config"], ".yaml"
    )
    await _upload_text(sandbox, callback_module_source(), paths["callback"], ".py")
    await _upload_text(
        sandbox,
        Path(__file__).with_name("litellm_bedrock_patch.py").read_text(),
        paths["patch"],
        ".py",
    )
    await _upload_text(
        sandbox, f"import {_PATCH_MODULE}\n", paths["sitecustomize"], ".py"
    )
    await _upload_text(sandbox, _sandbox_launcher_source(), paths["launcher"], ".py")
    await _upload_text(
        sandbox, BEDROCK_PATCH_PREFLIGHT_SOURCE, paths["preflight"], ".py"
    )
    return paths


async def _ensure_sandbox_litellm(sandbox: Any, *, venv_dir: str) -> str:
    vq = shlex.quote(venv_dir)
    # Prefer uv to bootstrap the venv: many sandbox base images ship a python3
    # without ensurepip and marked externally-managed (PEP 668), where both
    # `python -m venv` and `pip install` fail. uv needs neither (it is the same
    # mechanism the openhands agent install already uses in-sandbox), with a
    # stdlib-venv fallback for images that have a working venv and lack uv.
    command = f"""
set -eu
export PATH="$HOME/.local/bin:$PATH"
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
  export PATH="$HOME/.local/bin:$PATH"
  UV="$(command -v uv || true)"
fi
if [ -n "$UV" ]; then
  [ -x {vq}/bin/python ] || "$UV" venv {vq} >/dev/null 2>&1
  "$UV" pip install --python {vq}/bin/python -q '{LITELLM_VERSION_SPEC}' 'boto3>=1.40'
else
  PY="$(command -v python3 || command -v python)"
  if [ ! -x {vq}/bin/python ]; then
    "$PY" -m venv {vq} 2>/dev/null || (
      "$PY" -m pip install --user -q virtualenv &&
      "$PY" -m virtualenv {vq}
    )
  fi
  {vq}/bin/python -m pip install -q --upgrade pip
  {vq}/bin/python -m pip install -q '{LITELLM_VERSION_SPEC}' 'boto3>=1.40'
fi
{vq}/bin/python - <<'PY'
import litellm
print(litellm.__version__ if hasattr(litellm, "__version__") else "ok")
PY
"""
    result = await sandbox.exec(command, timeout_sec=600)
    if result.return_code != 0:
        raise RuntimeError(_exec_details("install LiteLLM in sandbox", result))
    return f"{venv_dir}/bin/python"


async def _wait_for_sandbox_state(
    sandbox: Any,
    *,
    state_path: str,
    stderr_path: str,
) -> dict[str, Any]:
    last_output = ""
    for _ in range(120):
        result = await sandbox.exec(
            f"cat {shlex.quote(state_path)} 2>/dev/null || true",
            timeout_sec=5,
        )
        last_output = (result.stdout or "").strip()
        if last_output:
            try:
                state = json.loads(last_output)
                if int(state.get("port") or 0) > 0:
                    return state
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        await asyncio.sleep(0.25)
    stderr = await sandbox.exec(
        f"tail -c 4000 {shlex.quote(stderr_path)} 2>/dev/null || true",
        timeout_sec=5,
    )
    raise RuntimeError(
        "sandbox LiteLLM did not publish its state: "
        f"{last_output or (stderr.stdout or '').strip()}"
    )


async def _poll_sandbox_health(
    sandbox: Any, *, python: str, port: int, stderr_path: str
) -> None:
    probe = (
        f"{shlex.quote(python)} - <<'PY'\n"
        "import sys, urllib.request\n"
        f"url='http://127.0.0.1:{port}/health/liveliness'\n"
        "try:\n"
        "    urllib.request.urlopen(url, timeout=2).read()\n"
        "except Exception:\n"
        "    urllib.request.urlopen(url.replace('/health/liveliness','/health'), timeout=2).read()\n"
        "PY"
    )
    for _ in range(120):
        result = await sandbox.exec(probe, timeout_sec=5)
        if result.return_code == 0:
            return
        await asyncio.sleep(0.25)
    stderr = await sandbox.exec(
        f"tail -c 4000 {shlex.quote(stderr_path)} 2>/dev/null || true",
        timeout_sec=5,
    )
    raise RuntimeError(f"sandbox LiteLLM did not become healthy: {stderr.stdout or ''}")


async def _terminate_sandbox_litellm(
    sandbox: Any, *, pid_path: str, runtime_dir: str
) -> None:
    with contextlib.suppress(Exception):
        await sandbox.exec(
            f"if [ -s {shlex.quote(pid_path)} ]; then "
            f"kill -TERM $(cat {shlex.quote(pid_path)}) 2>/dev/null || true; fi",
            timeout_sec=10,
        )
    with contextlib.suppress(Exception):
        await sandbox.exec(f"rm -rf {shlex.quote(runtime_dir)}", timeout_sec=10)


async def _start_sandbox_litellm(
    *,
    sandbox: Any,
    route: LiteLLMRoute,
    master_key: str,
    agent_env: dict[str, str],
    session_id: str,
    agent_name: str,
) -> SandboxLiteLLMProcess:
    token = uuid4().hex[:16]
    runtime_dir = f"{LITELLM_SANDBOX_ROOT}/{token}"
    config = litellm_proxy_config(route, master_key=master_key)
    paths = await _upload_runtime_files_to_sandbox(
        sandbox,
        runtime_dir=runtime_dir,
        config=config,
    )
    python = await _ensure_sandbox_litellm(sandbox, venv_dir=paths["venv"])
    env = dict(agent_env)
    env.update(
        {
            "PYTHONPATH": f"{runtime_dir}:{env.get('PYTHONPATH', '')}",
            "LITELLM_MASTER_KEY": master_key,
            "BENCHFLOW_LITELLM_LOG_PATH": paths["log"],
            **_PROXY_DOCS_DISABLE_ENV,
        }
    )
    launch_config = {
        "python": python,
        "litellm": f"{paths['venv']}/bin/litellm",
        "config": paths["config"],
        "stdout": paths["stdout"],
        "stderr": paths["stderr"],
        "pid": paths["pid"],
        "state": paths["state"],
        "env": env,
    }
    await _upload_text(
        sandbox, json.dumps(launch_config), paths["launch_config"], ".json"
    )
    command = (
        f"rm -f {shlex.quote(paths['state'])} {shlex.quote(paths['pid'])} "
        f"{shlex.quote(paths['log'])} && "
        f"{shlex.quote(python)} {shlex.quote(paths['launcher'])} "
        f"{shlex.quote(paths['launch_config'])}"
    )
    try:
        result = await sandbox.exec(command, timeout_sec=20)
        if result.return_code != 0:
            raise RuntimeError(_exec_details("start sandbox LiteLLM", result))
        state = await _wait_for_sandbox_state(
            sandbox,
            state_path=paths["state"],
            stderr_path=paths["stderr"],
        )
        port = int(state["port"])
        await _poll_sandbox_health(
            sandbox,
            python=python,
            port=port,
            stderr_path=paths["stderr"],
        )
        if route_requires_bedrock_patch(route):
            # Fail closed before the agent launches when the Bedrock 4.8+
            # thinking patch did not activate (#602). On Daytona this proxy is
            # the only protection — there is no host proxy to fall back to.
            await preflight_sandbox_bedrock_patch(
                sandbox,
                python=python,
                runtime_dir=runtime_dir,
                preflight_path=paths["preflight"],
            )
    except BaseException:
        # Never leak a half-started proxy (provider keys + master_key on disk).
        await _terminate_sandbox_litellm(
            sandbox, pid_path=paths["pid"], runtime_dir=runtime_dir
        )
        raise
    endpoint = LiteLLMEndpoint(
        agent_base_url=f"http://127.0.0.1:{port}",
        local_base_url=f"http://127.0.0.1:{port}",
    )
    logger.info("Sandbox LiteLLM proxy listening on %s", endpoint.agent_base_url)
    return SandboxLiteLLMProcess(
        sandbox=sandbox,
        route=route,
        runtime_dir=runtime_dir,
        endpoint=endpoint,
        log_path=paths["log"],
        pid_path=paths["pid"],
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
        session_id=session_id,
        agent_name=agent_name,
    )


def _exec_details(label: str, result: Any) -> str:
    stdout = (getattr(result, "stdout", "") or "").strip()
    stderr = (getattr(result, "stderr", "") or "").strip()
    details = [f"{label} failed with exit code {getattr(result, 'return_code', '?')}"]
    if stdout:
        details.append(f"stdout: {stdout[:2000]}")
    if stderr:
        details.append(f"stderr: {stderr[:2000]}")
    return "; ".join(details)


def _missing_required_env(route: LiteLLMRoute, env: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for key in route.required_env:
        if key == "AWS_REGION" and (
            env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
        ):
            continue
        if not env.get(key):
            missing.append(key)
    return missing


def _provider_secret_env_names() -> set[str]:
    """Upstream provider credentials the proxy owns and the agent must not see."""
    from benchflow.agents.providers import PROVIDERS

    names = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AZURE_API_KEY",
    }
    for cfg in PROVIDERS.values():
        if cfg.auth_env:
            names.add(cfg.auth_env)
    return names


def _provider_model_id(entry: object) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    entry_id = cast("Mapping[str, object]", entry).get("id")
    return entry_id if isinstance(entry_id, str) else None


def _provider_models_for_proxy_alias(
    *,
    raw: str | None,
    route: LiteLLMRoute,
) -> str | None:
    """Mirror model metadata onto the LiteLLM alias Pi sees in proxy mode.

    Pi resolves ``maxTokens``/``contextWindow`` by looking up the model it is
    told to use (the LiteLLM alias) in ``BENCHFLOW_PROVIDER_MODELS``. Without an
    alias entry that metadata is lost once traffic is routed through the proxy,
    so clone the source entry under the alias id/name.
    """
    if not raw:
        return None
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(entries, list):
        return None
    wanted = {
        route.requested_model,
        strip_provider_prefix(route.requested_model),
        route.upstream_model,
        strip_provider_prefix(route.upstream_model),
    }
    for entry in entries:
        entry_id = _provider_model_id(entry)
        if entry_id not in wanted:
            continue
        alias_entry = dict(cast("Mapping[str, Any]", entry))
        alias_entry["id"] = route.model_alias
        alias_entry["name"] = route.model_alias
        merged = list(entries)
        if not any(_provider_model_id(item) == route.model_alias for item in merged):
            merged.append(alias_entry)
        return json.dumps(merged)
    return None


# Caller-supplied provider endpoints. If any of these survive in the agent env,
# the agent could reach a provider directly and bypass the proxy (the exact way
# an Azure ``LLM_BASE_URL`` leaked past the gateway before this hardening). They
# are stripped before the proxy endpoint is wired in; the proxy process keeps
# whatever upstream base_url it needs in its own env / baked config.
_PROVIDER_ENDPOINT_ENV_NAMES = frozenset(
    {
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "AZURE_API_ENDPOINT",
        "AZURE_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_BEDROCK_BASE_URL",
        "GEMINI_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
    }
)


def _assert_proxy_isolated(agent: str, env: dict[str, str], *, master_key: str) -> None:
    """Fail closed if a *raw* provider secret would still reach the agent.

    Invariant after wiring: a routable agent sees only the proxy endpoint and
    the proxy master key — never an upstream provider credential. Some agents
    legitimately carry the master key in a provider-named slot (codex/opencode
    put it in ``OPENAI_API_KEY``, claude in ``ANTHROPIC_AUTH_TOKEN``), so a
    provider-named var holding exactly ``master_key`` is the proxy credential and
    is allowed; anything else in those slots is a raw upstream key. If one
    survives (e.g. an agent registry ``env_mapping`` re-exposed it), refuse to
    run so the agent cannot authenticate directly against a provider.
    """
    leaked = sorted(
        name
        for name in _provider_secret_env_names()
        if env.get(name) and env.get(name) != master_key
    )
    if leaked:
        raise RuntimeError(
            f"LiteLLM proxy isolation breached for agent {agent!r}: raw provider "
            f"secrets would reach the agent ({', '.join(leaked)}). Refusing to run "
            "to prevent direct-to-provider traffic."
        )


def _apply_litellm_agent_env(
    *,
    agent: str,
    agent_env: dict[str, str],
    route: LiteLLMRoute,
    base_url: str,
    master_key: str,
) -> dict[str, str]:
    """Rewrite the agent env to talk only to the proxy, then verify isolation."""
    updated = _wire_litellm_agent_env(
        agent=agent,
        agent_env=agent_env,
        route=route,
        base_url=base_url,
        master_key=master_key,
    )
    _assert_proxy_isolated(agent, updated, master_key=master_key)
    return updated


def _wire_litellm_agent_env(
    *,
    agent: str,
    agent_env: dict[str, str],
    route: LiteLLMRoute,
    base_url: str,
    master_key: str,
) -> dict[str, str]:
    updated = dict(agent_env)
    # Isolation: the agent must reach providers only through the proxy. Drop raw
    # upstream provider secrets AND endpoints so a compromised or curious agent
    # cannot bypass the gateway (and its usage metering) or read live keys. The
    # proxy process holds them via its own env. (In sandbox-local mode the proxy
    # shares the agent's sandbox, so this reduces — but cannot fully remove — key
    # visibility.)
    for secret_key in _provider_secret_env_names():
        updated.pop(secret_key, None)
    for endpoint_key in _PROVIDER_ENDPOINT_ENV_NAMES:
        updated.pop(endpoint_key, None)
    openai_base_url = f"{base_url.rstrip('/')}/v1"
    updated.update(
        {
            "BENCHFLOW_PROVIDER_NAME": "litellm",
            "BENCHFLOW_PROVIDER_BASE_URL": openai_base_url,
            "BENCHFLOW_PROVIDER_API_KEY": master_key,
            "BENCHFLOW_PROVIDER_MODEL": route.model_alias,
            "BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions",
            LITELLM_MODEL_ALIAS_ENV: route.model_alias,
            LITELLM_MASTER_KEY_ENV: master_key,
        }
    )
    if agent == "codex-acp":
        updated["OPENAI_BASE_URL"] = openai_base_url
        updated["OPENAI_API_KEY"] = master_key
        updated[LITELLM_MODEL_VIA_ENV] = "1"
        apply_codex_provider_config(
            updated,
            base_url=openai_base_url,
            model=route.model_alias,
            provider_name="litellm",
            strict=True,
        )
        return updated
    if agent == "opencode":
        updated["OPENAI_BASE_URL"] = openai_base_url
        updated["OPENAI_API_KEY"] = master_key
        return updated
    if agent == "openhands":
        updated["LLM_BASE_URL"] = openai_base_url
        updated["LLM_API_KEY"] = master_key
        updated["LLM_MODEL"] = f"openai/{route.model_alias}"
        updated[LITELLM_MODEL_VIA_ENV] = "1"
        return updated
    if agent == "claude-agent-acp":
        updated["ANTHROPIC_BASE_URL"] = base_url.rstrip("/")
        updated["ANTHROPIC_AUTH_TOKEN"] = master_key
        updated["ANTHROPIC_MODEL"] = route.model_alias
        updated[LITELLM_MODEL_VIA_ENV] = "1"
        for key in (
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
            "ANTHROPIC_BEDROCK_BASE_URL",
        ):
            updated.pop(key, None)
        return updated
    if agent == "pi-acp":
        updated["BENCHFLOW_PROVIDER_PROTOCOL"] = "openai-completions"
        updated["BENCHFLOW_PROVIDER_BASE_URL"] = openai_base_url
        updated["BENCHFLOW_PROVIDER_API_KEY"] = master_key
        updated["BENCHFLOW_PROVIDER_MODEL"] = route.model_alias
        updated["BENCHFLOW_PROVIDER_NAME"] = "litellm"
        alias_models = _provider_models_for_proxy_alias(
            raw=agent_env.get("BENCHFLOW_PROVIDER_MODELS"),
            route=route,
        )
        if alias_models:
            updated["BENCHFLOW_PROVIDER_MODELS"] = alias_models
        return updated

    agent_cfg = AGENTS.get(agent)
    if agent_cfg and agent_cfg.env_mapping:
        for src, dst in agent_cfg.env_mapping.items():
            if src in updated:
                updated[dst] = updated[src]
    return updated


async def _skip_litellm_runtime(
    agent_env: dict[str, str],
    runtime: Any | None,
    *,
    reason: str | None = None,
) -> tuple[dict[str, str], Any | None]:
    if runtime is not None:
        await stop_litellm_runtime(runtime)
    if reason:
        logger.info("Skipping LiteLLM proxy: %s", reason)
    return agent_env, None


async def _raise_litellm_unavailable(
    *,
    runtime: Any | None,
    error: str,
) -> NoReturn:
    """Fail closed when a routable agent cannot be put behind the proxy.

    BenchFlow always routes routable agents through the LiteLLM proxy so
    provider traffic is metered and captured (``llm_trajectory.jsonl``).
    Silently falling back to direct provider access would leak the raw key,
    bypass usage/cost tracking, and lose the trainable trajectory — so any
    proxy unavailability is fatal, independent of ``usage_tracking`` mode.
    """
    if runtime is not None:
        await stop_litellm_runtime(runtime)
    raise RuntimeError(error)


async def ensure_litellm_runtime(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    runtime: Any | None,
    environment: str,
    session_id: str = "",
    usage_tracking: UsageTrackingConfig | dict[str, Any] | str | None = None,
    sandbox: Any | None = None,
) -> tuple[dict[str, str], Any | None]:
    """Start/reuse LiteLLM and rewrite the agent env to talk to it.

    Every LiteLLM-routable agent is *always* routed through the proxy so
    provider traffic is metered and captured (``llm_trajectory.jsonl``) and the
    raw provider key never reaches the agent. ``usage_tracking`` no longer gates
    whether the proxy runs — it only governs whether trusted telemetry is
    *required* (``required`` fails closed when usage cannot be captured at all).
    The only agents that skip the proxy are those that physically cannot be
    routed through it: ``oracle`` (no model), native-protocol agents (e.g.
    ``gemini``), and native-subscription auth (no API key to proxy).
    """
    usage_cfg = UsageTrackingConfig.coerce(usage_tracking).with_env_defaults()

    if uses_native_subscription_auth(agent, model, agent_env):
        return await _skip_litellm_runtime(
            agent_env,
            runtime,
            reason="native subscription auth will use agent ACP usage telemetry",
        )

    if not needs_litellm_runtime(agent, model):
        if usage_cfg.mode == "required" and agent != "oracle":
            raise RuntimeError(
                "Token usage tracking is required, but agent "
                f"{agent!r} cannot be routed through LiteLLM."
            )
        return await _skip_litellm_runtime(agent_env, runtime)
    assert model is not None

    if environment in _SANDBOX_LOCAL_ENVIRONMENTS and sandbox is None:
        raise RuntimeError("sandbox-local LiteLLM requires a sandbox handle")

    try:
        route = resolve_litellm_route(model, agent_env)
    except ValueError as exc:
        await _raise_litellm_unavailable(
            runtime=runtime,
            error=(
                "LiteLLM proxy is mandatory but cannot resolve "
                f"model {model!r}: {exc}. BenchFlow never sends provider traffic "
                "directly — register the provider/model or fix the route."
            ),
        )
    missing = _missing_required_env(route, agent_env)
    if missing:
        missing_text = ", ".join(missing)
        await _raise_litellm_unavailable(
            runtime=runtime,
            error=(
                f"LiteLLM route for model {model!r} requires {missing_text}. "
                "Pass provider credentials via --agent-env/agent_env or define "
                "them in .env (the proxy is mandatory; traffic is never sent "
                "directly to the provider)."
            ),
        )

    master_key = (
        agent_env.get(LITELLM_MASTER_KEY_ENV)
        or f"sk-benchflow-{secrets.token_urlsafe(24)}"
    )
    config_key = f"{environment}:{route.config_key}:{agent}:{session_id}"
    if runtime is not None and getattr(runtime, "kind", None) == "litellm":
        server = getattr(runtime, "server", None)
        if getattr(runtime, "config_key", None) == config_key and server is not None:
            is_running = await server.is_running()
            if is_running:
                return (
                    _apply_litellm_agent_env(
                        agent=agent,
                        agent_env=agent_env,
                        route=route,
                        base_url=runtime.base_url,
                        master_key=getattr(runtime, "master_key", master_key),
                    ),
                    runtime,
                )
        await stop_litellm_runtime(runtime)

    try:
        if environment in _SANDBOX_LOCAL_ENVIRONMENTS:
            server = await _start_sandbox_litellm(
                sandbox=sandbox,
                route=route,
                master_key=master_key,
                agent_env=agent_env,
                session_id=session_id,
                agent_name=agent,
            )
        else:
            server = await _start_host_litellm(
                route=route,
                master_key=master_key,
                agent_env=agent_env,
                environment=environment,
                session_id=session_id,
                agent_name=agent,
            )
    except BedrockPatchPreflightError:
        raise
    except Exception as exc:
        await _raise_litellm_unavailable(
            runtime=None,
            error=(
                f"LiteLLM proxy failed to start for model {model!r}: {exc}. "
                "BenchFlow never sends provider traffic directly, so this is fatal."
            ),
        )

    from benchflow.providers.runtime import ProviderRuntime

    new_runtime = ProviderRuntime(
        kind="litellm",
        agent_base_url=server.base_url,
        backend_model=route.upstream_model,
        server=server,
        config_key=config_key,
        master_key=master_key,
    )
    return (
        _apply_litellm_agent_env(
            agent=agent,
            agent_env=agent_env,
            route=route,
            base_url=new_runtime.base_url,
            master_key=master_key,
        ),
        new_runtime,
    )


async def stop_litellm_runtime(runtime: Any | None) -> None:
    if runtime is None:
        return
    server = getattr(runtime, "server", None)
    if getattr(runtime, "kind", None) == "litellm" and server is not None:
        await server.stop()


def extract_usage(runtime: Any | None) -> dict[str, Any]:
    if runtime is None or getattr(runtime, "kind", None) != "litellm":
        return usage_unavailable()
    server = getattr(runtime, "server", None)
    trajectory = getattr(server, "trajectory", None)
    return extract_usage_from_trajectory(
        trajectory,
        fallback_model=getattr(runtime, "backend_model", None),
    )
