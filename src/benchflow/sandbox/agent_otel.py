"""Ingest the OpenTelemetry an agent sends during its turn.

When a run asks the agent to export OpenTelemetry and names no collector of
its own, BenchFlow listens for it: a loopback OTLP/HTTP receiver
(:mod:`benchflow.sandbox._otlp_receiver`) runs as root in the agent's sandbox,
the agent's ``OTEL_EXPORTER_OTLP_*`` variables point at it (JSON), and at
teardown what arrived is copied to the trial's ``trajectory/agent-otel.jsonl``,
one export per line. :mod:`benchflow.trajectories.model_calls` reads Claude
Code's ``claude_code.api_request`` events from it.

A run asks for it by enabling the agent's own exporter, for Claude Code
``--agent-env CLAUDE_CODE_ENABLE_TELEMETRY=1``; any ``OTEL_LOGS_EXPORTER``,
``OTEL_METRICS_EXPORTER`` or ``OTEL_TRACES_EXPORTER`` naming ``otlp`` does too.
An ``OTEL_EXPORTER_OTLP_ENDPOINT`` the run sets is the run's own collector:
BenchFlow then leaves the agent's telemetry alone. What the agent sends is
the agent's account of itself, kept as sent.
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchflow._root_python import FLAGS, root_python_pick_sh
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox.egress_denylist import _collect_log
from benchflow.trust.state import TRUSTED_TRANSFER, admit

PORT = 4318
ENDPOINT = f"http://127.0.0.1:{PORT}"
RUNTIME_DIR = "/opt/runtime-otel"
RECEIVED = f"{RUNTIME_DIR}/received.jsonl"
TRAJECTORY_NAME = "agent-otel.jsonl"
_PID = f"{RUNTIME_DIR}/receiver.pid"
_STDERR = f"{RUNTIME_DIR}/stderr.log"
_SCRIPT = Path(__file__).with_name("_otlp_receiver.py")
_EXPORTERS = ("OTEL_LOGS_EXPORTER", "OTEL_METRICS_EXPORTER", "OTEL_TRACES_EXPORTER")
# The kit holds a Python only when the task image has one (_runtime_tree).
_NO_PYTHON = 'echo "the task image has no Python 3 for the receiver" >&2; exit 87'

logger = logging.getLogger(__name__)


def wants_agent_otel(agent_env: Mapping[str, str]) -> bool:
    """Whether the run asks the agent to export OpenTelemetry to BenchFlow."""
    if agent_env.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    if str(agent_env.get("CLAUDE_CODE_ENABLE_TELEMETRY", "")).strip() == "1":
        return True
    return any("otlp" in str(agent_env.get(name, "")).lower() for name in _EXPORTERS)


def agent_otel_env(agent_env: Mapping[str, str]) -> dict[str, str]:
    """*agent_env* with the agent's OTLP exporters pointed at the receiver."""
    env = dict(agent_env)
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = ENDPOINT
    env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/json"
    if env.get("CLAUDE_CODE_ENABLE_TELEMETRY") == "1":
        # Claude Code exports nothing until an exporter is named; logs carry
        # one api_request event per model call.
        env.setdefault("OTEL_LOGS_EXPORTER", "otlp")
        env.setdefault("OTEL_METRICS_EXPORTER", "otlp")
        env.setdefault("OTEL_LOGS_EXPORT_INTERVAL", "2000")
        env.setdefault("OTEL_METRIC_EXPORT_INTERVAL", "10000")
    return env


def _start_command() -> str:
    q = shlex.quote
    receiver = (
        f'"$PY" {" ".join(FLAGS)} {q(RUNTIME_DIR + "/receiver.py")} '
        f"--port {PORT} --out {q(RECEIVED)}"
    )
    return (
        "set -e; cd /; "
        + root_python_pick_sh("PY", fallback="python3", refuse=_NO_PYTHON)
        + f"; touch {q(RECEIVED)}; chmod 600 {q(RECEIVED)}; "
        f'if [ -s {q(_PID)} ] && kill -0 "$(cat {q(_PID)})" 2>/dev/null; then exit 0; fi; '
        f"nohup {receiver} </dev/null >/dev/null 2>{q(_STDERR)} & "
        f"echo $! > {q(_PID)}"
    )


def _health_command() -> str:
    probe = (
        "import urllib.request, sys; "
        "opener = urllib.request.build_opener(urllib.request.ProxyHandler({})); "
        f'sys.exit(0 if opener.open("{ENDPOINT}/healthz", timeout=2).status == 200 else 1)'
    )
    return (
        root_python_pick_sh("PY", fallback="python3", refuse=_NO_PYTHON)
        + "; for i in 1 2 3 4 5 6 7 8 9 10; do "
        + f'"$PY" {" ".join(FLAGS)} -c {shlex.quote(probe)} && exit 0; sleep 0.5; done; '
        + f"tail -c 1000 {shlex.quote(_STDERR)} >&2; exit 1"
    )


async def start_agent_otel(env: Any) -> bool:
    """Start the loopback receiver as root (once per sandbox); whether it runs.

    Never fails the trial: without it (no Python 3 in the task image, which
    the receiver runs on as root's protected copy) the agent's exporter is
    left as the run set it, and the reason is logged.
    """
    try:
        await _start(env)
    except Exception as exc:
        logger.warning("The agent's OpenTelemetry is not received: %s", exc)
        return False
    return True


async def _start(env: Any) -> None:
    q = shlex.quote
    made = await exec_raw(
        env.exec,
        f"mkdir -p {q(RUNTIME_DIR)} && chmod 700 {q(RUNTIME_DIR)}",
        user="root",
        timeout_sec=30,
    )
    if getattr(made, "return_code", 1) != 0:
        raise RuntimeError("could not make the receiver's folder")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "receiver.py").write_bytes(_SCRIPT.read_bytes())
        await env.upload_dir(
            tmp,
            RUNTIME_DIR,
            admission=admit(
                env,
                "main",
                TRUSTED_TRANSFER,
                reason="BenchFlow's OpenTelemetry receiver",
            ),
        )
    for command, what in (
        (f"chmod 600 {q(RUNTIME_DIR + '/receiver.py')} && {_start_command()}", "start"),
        (_health_command(), "health check"),
    ):
        result = await exec_raw(env.exec, command, user="root", timeout_sec=60)
        if getattr(result, "return_code", 1) != 0:
            detail = (getattr(result, "stderr", "") or getattr(result, "stdout", ""))[
                -400:
            ]
            raise RuntimeError(f"receiver {what} failed: {detail.strip()}")


async def stop_agent_otel(env: Any, rollout_dir: Any) -> None:
    """Copy what arrived to ``trajectory/agent-otel.jsonl``, then stop; never raises."""
    await _collect_log(env, RECEIVED, rollout_dir, TRAJECTORY_NAME)
    with contextlib.suppress(Exception):
        await exec_raw(
            env.exec,
            f"kill -TERM $(cat {shlex.quote(_PID)}) 2>/dev/null; rm -rf {shlex.quote(RUNTIME_DIR)}",
            user="root",
            timeout_sec=30,
        )
