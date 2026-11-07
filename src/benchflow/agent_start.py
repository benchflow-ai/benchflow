"""``bench doctor --agent-start``: can an installed agent start at all?

Creates a sandbox for the bundled hello-world task, installs the agent the way
a run does, opens its ACP connection (``initialize`` and ``session/new``) and
closes it again, without sending a prompt, so no model call is made. A broken
install, a missing runtime (``Node.js v22.16+ is required``) or a rejected
login shows up here in a minute instead of as a silent 0 in every trial of a
batch. The sandbox is always cleaned up.

Control agents (``oracle``, ``nop``) have no ACP process: for them the check
stops after the install step.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass

from benchflow.integration_health import (
    _AUTH_MARKERS,
    _INSTALL_MARKERS,
    _marker_line,
    read_agent_logs,
)

_CONTROL_AGENTS = frozenset({"oracle", "nop"})
_TAIL_CHARS = 600


@dataclass(frozen=True)
class AgentStartOutcome:
    agent: str
    ok: bool
    seconds: float
    error: str | None = None
    cause: str | None = None  # agent_auth | agent_install | agent_start
    log_tail: str | None = None
    connected: bool = False


def classify_start_failure(error: str, logs: dict[str, str]) -> tuple[str, str]:
    """(cause, evidence line) for a failed start, from the error and logs."""
    texts = [("error", error), *logs.items()]
    for cause, pattern in (
        ("agent_auth", _AUTH_MARKERS),
        ("agent_install", _INSTALL_MARKERS),
    ):
        for _, text in texts:
            line = _marker_line(text, pattern)
            if line:
                return cause, line
    return "agent_start", error.strip().splitlines()[0][:300] if error.strip() else ""


async def _probe(agent: str, *, sandbox: str, model: str | None) -> AgentStartOutcome:
    from benchflow._utils.text import describe_exception
    from benchflow.doctor_smoke import BUNDLED_TASK_DIR, SMOKE_DEFAULT_MODELS
    from benchflow.rollout import Rollout, RolloutConfig

    started = time.monotonic()
    control = agent in _CONTROL_AGENTS
    with tempfile.TemporaryDirectory(prefix="bf-agent-start-") as jobs_dir:
        rollout = Rollout(
            RolloutConfig(
                task_path=BUNDLED_TASK_DIR,
                agent=agent,
                model=None if control else (model or SMOKE_DEFAULT_MODELS.get(agent)),
                environment=sandbox,
                jobs_dir=jobs_dir,
                job_name="agent-start",
                skip_verify=True,
            )
        )
        error: str | None = None
        connected = False
        try:
            await rollout.setup()
            await rollout.start()
            await rollout.install_agent()
            if not control:
                await rollout.connect()
                connected = True
        except BaseException as exc:  # report every failure, including timeouts
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
            error = describe_exception(exc)
        finally:
            logs = (
                read_agent_logs(rollout._rollout_dir)
                if rollout._rollout_dir is not None
                else {}
            )
            if connected:
                try:
                    await rollout.disconnect()
                except Exception as exc:
                    error = error or f"disconnect failed: {describe_exception(exc)}"
            try:
                await rollout.cleanup()
            except Exception as exc:
                error = error or f"sandbox cleanup failed: {describe_exception(exc)}"
    seconds = round(time.monotonic() - started, 1)
    if error is None:
        return AgentStartOutcome(agent, True, seconds, connected=connected)
    cause, line = classify_start_failure(error, logs)
    tail = "\n".join(
        text.strip()[-_TAIL_CHARS:] for text in logs.values() if text.strip()
    )
    return AgentStartOutcome(
        agent,
        False,
        seconds,
        error=error if cause == "agent_start" else f"{line} ({error[:200]})",
        cause=cause,
        log_tail=tail or None,
        connected=connected,
    )


def probe_agent_start(
    agent: str, *, sandbox: str, model: str | None = None
) -> AgentStartOutcome:
    """Install and start ``agent`` in a fresh ``sandbox`` without a prompt."""
    return asyncio.run(_probe(agent, sandbox=sandbox, model=model))


def agent_start_check(outcome: AgentStartOutcome):
    """The doctor row for one probe."""
    from benchflow.doctor import _checker

    row = _checker("agents", f"{outcome.agent} starts", f"agent_start.{outcome.agent}")
    details = {
        "seconds": outcome.seconds,
        "connected": outcome.connected,
        "cause": outcome.cause,
    }
    if outcome.ok:
        what = (
            "installed (control agent, no ACP process)"
            if outcome.agent in _CONTROL_AGENTS
            else "installed and answered the ACP handshake (no prompt sent)"
        )
        return row("pass", f"{what} in {outcome.seconds:.0f} s", details=details)
    fixes = {
        "agent_auth": "Log in again or replace the key the agent uses "
        "(bench doctor lists the credential it found)",
        "agent_install": "Fix the agent's install: the sandbox image lacks what "
        "the agent needs (see the log tail)",
    }
    details["log_tail"] = outcome.log_tail
    return row(
        "fail",
        f"cannot start [{outcome.cause}]: {outcome.error}",
        fixes.get(outcome.cause or "", "Run bench eval smoke for the full log"),
        details,
    )
