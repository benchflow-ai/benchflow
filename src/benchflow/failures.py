"""Why a trial has no score, whose fault it was, and what to do next.

One table for the end-of-run summary and for docs/when-a-run-fails.md. A
trial ends in one of three ways:

* **scored**: the verifier gave it a reward (passed or failed);
* **unscored**: the agent finished but the verifier could not score it (a
  task problem, or the verifier's sandbox failed);
* **errored**: the run stopped before a score (the agent, its login, the
  provider or the sandbox failed).

:func:`cause_of` names the cause of an unscored or errored trial from its
stored categories and diagnostics: a short label, whose fault it is (the
``task``, the ``agent``, the ``infrastructure``, or the developer's
``setup``), and the next step. It never changes a score: the buckets are the
ones :func:`benchflow._utils.scoring.classify_score_outcome` decides.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.errors import Fault

FAULT_WORDS: dict[str, str] = {
    "task": "task problem",
    "agent": "agent problem",
    "infrastructure": "infrastructure problem",
    "setup": "setup problem",
}


@dataclass(frozen=True)
class Cause:
    """Why one trial has no score."""

    key: str  # stable: the category, or a finer name (verifier_plugin_trust)
    label: str  # "usage limit on login X, 7-day window, resets ..."
    fault: Fault
    next_step: str  # a command or a fix; {job} and {task} are filled in

    def headline(self) -> str:
        return f"{self.label} ({FAULT_WORDS[self.fault]})"


# category -> (label, fault, next step). {job} is the job folder, {task} the
# task folder; both are filled in when known.
_ERRORED: dict[str, tuple[str, Fault, str]] = {
    "provider_auth": (
        "the model provider refused the credential",
        "setup",
        "`bench doctor` checks each login and key",
    ),
    "provider_rate_limit": (
        "the model provider rate-limited the run",
        "infrastructure",
        "lower --concurrency, then `bench eval resume {job}`",
    ),
    "provider_rejected": (
        "the model provider rejected the request (often a context too long)",
        "agent",
        "read the error in the trial's result.json; a model with a larger "
        "context, or a shorter prompt",
    ),
    "api_error": (
        "the model provider's API failed",
        "infrastructure",
        "`bench eval resume {job}` reruns it; api_error_info in result.json "
        "has the status",
    ),
    "suspected_api_error": (
        "the agent did nothing (no tokens, no tool calls)",
        "setup",
        "check the model id and the login with `bench doctor`, then "
        "`bench eval resume {job}`",
    ),
    "install_failure": (
        "the agent's install failed",
        "infrastructure",
        "`bench doctor` checks the network to npm and Node.js; the trial's "
        "agent/install-stdout.txt has the log",
    ),
    "pipe_closed": (
        "the connection to the agent closed",
        "infrastructure",
        "`bench eval resume {job}` reruns it; transport_error_info in "
        "result.json says why",
    ),
    "idle_timeout": (
        "the agent went silent and the idle watchdog stopped it",
        "agent",
        "raise --agent-idle-timeout if the agent thinks for long; "
        "idle_timeout_info in result.json has the details",
    ),
    "timeout": (
        "the agent ran out of time",
        "agent",
        "raise the task's agent timeout (task.md agent.timeout_sec) or --timeout",
    ),
    "infra_failure": (
        "the sandbox or its transport failed",
        "infrastructure",
        "`bench eval resume {job}` reruns it",
    ),
    "sandbox_setup": (
        "the sandbox did not start",
        "infrastructure",
        "`bench doctor`; sandbox_startup_info in result.json has the error",
    ),
    "acp_error": (
        "the agent reported an error",
        "agent",
        "the agent's log (agent/<agent>.txt in the trial folder) has its side",
    ),
}
_INTEGRATION: dict[str, tuple[str, Fault, str]] = {
    "agent_auth": ("its login or billing failed", "setup", "`bench doctor`"),
    "agent_model": (
        "it does not offer the model",
        "setup",
        "pick a model the agent lists in the error",
    ),
    "agent_install": (
        "its install or runtime is broken",
        "infrastructure",
        "`bench doctor --agent-start <agent>`",
    ),
}
_UNSCORED: dict[str, tuple[str, Fault, str]] = {
    "verifier_failure": (
        "the verifier crashed",
        "task",
        "`bench tasks check {task}`; the trial's verifier/test-stdout.txt has "
        "the output",
    ),
    "verifier_timeout": (
        "the verifier timed out",
        "task",
        "raise the task's verifier timeout (task.md verifier.timeout_sec)",
    ),
    "verifier_dep_install": (
        "the verifier's dependency install failed",
        "task",
        "pin or bake the verifier's dependencies into the image; "
        "`bench tasks check {task}`",
    ),
    "verifier_infra": (
        "the verifier's sandbox failed",
        "infrastructure",
        "`bench eval resume {job}` scores it again",
    ),
    "verifier_other": (
        "the verifier failed",
        "task",
        "the trial's verifier/ folder has the output",
    ),
}


def _read_result(result: Any) -> dict[str, Any]:
    """The trial's result.json fields the causes need (diagnostics included)."""
    if isinstance(result, Mapping):
        return dict(result)
    data: dict[str, Any] = {
        "task_name": getattr(result, "task_name", ""),
        "error": getattr(result, "error", None),
        "error_category": getattr(result, "error_category", None),
        "verifier_error": getattr(result, "verifier_error", None),
        "verifier_error_category": getattr(result, "verifier_error_category", None),
        "rewards": getattr(result, "rewards", None),
    }
    rollout_dir = getattr(result, "rollout_dir", None)
    if rollout_dir is not None:
        try:
            saved = json.loads((Path(rollout_dir) / "result.json").read_text())
        except (OSError, ValueError):
            saved = {}
        if isinstance(saved, dict):
            for key, value in saved.items():
                data.setdefault(key, value)
    return data


def _fill(text: str, job: str | None, task: str | None) -> str:
    return text.replace("{job}", job or "<job dir>").replace("{task}", task or "<task>")


def cause_of(
    result: Any, *, job_dir: Path | None = None, task_dir: Path | None = None
) -> Cause | None:
    """The cause of an unscored or errored trial; None for a scored one."""
    from benchflow._utils.scoring import (
        classify_error,
        classify_score_outcome,
        classify_verifier_error,
    )

    data = _read_result(result)
    outcome = classify_score_outcome(data)
    if outcome in ("passed", "failed"):
        return None
    job = str(job_dir) if job_dir is not None else None
    task = str(task_dir) if task_dir is not None else None
    if outcome == "errored" and data.get("error"):
        category = data.get("error_category") or classify_error(data.get("error"))
        return _errored_cause(data, category or "other", job, task)
    verifier_error = data.get("verifier_error")
    if verifier_error:
        category = data.get("verifier_error_category") or classify_verifier_error(
            verifier_error
        )
        if "PluginGuardLoadError" in verifier_error and (
            "installed after the agent stopped" in verifier_error
        ):
            return Cause(
                "verifier_plugin_trust",
                "verifier plugin trust: test.sh installs a pytest plugin where "
                "the agent could write",
                "task",
                _fill("`bench tasks check {task}`", job, task),
            )
        label, fault, step = _UNSCORED.get(category or "", _UNSCORED["verifier_other"])
        return Cause(category or "verifier_other", label, fault, _fill(step, job, task))
    return Cause(
        "no_verdict",
        "no verdict was recorded (an assessment still pending, or an "
        "interrupted trial)",
        "infrastructure",
        _fill("`bench eval score {job}` finishes pending reviews", job, task),
    )


def _errored_cause(
    data: Mapping[str, Any], category: str, job: str | None, task: str | None
) -> Cause:
    if category == "usage_limit":
        from benchflow.agents.errors import UsageLimitError
        from benchflow.agents.usage_limits import format_reset

        err = UsageLimitError.from_result(data)
        parts = ["usage limit"]
        if err is not None and err.login:
            parts[0] += f" on login {err.login}"
        if err is not None and err.window:
            parts.append(f"{err.window} window")
        if err is not None and err.resets_at is not None:
            parts.append(f"resets {format_reset(err.resets_at)}")
        return Cause(
            "usage_limit",
            ", ".join(parts),
            "setup",
            _fill(
                "switch to another login or wait for the reset, then "
                "`bench eval resume {job}`; `bench doctor` shows each login's "
                "headroom",
                job,
                task,
            ),
        )
    if category == "agent_integration":
        info = data.get("integration_failure_info")
        why = info.get("cause") if isinstance(info, Mapping) else None
        label, fault, step = _INTEGRATION.get(
            str(why),
            (
                "it did nothing useful",
                "agent",
                "integration_failure_info in result.json",
            ),
        )
        return Cause(
            f"agent_integration:{why or 'unknown'}",
            f"the agent integration broke: {label}",
            fault,
            _fill(step, job, task),
        )
    if category == "pipe_closed":
        info = data.get("transport_error_info")
        code = info.get("process_exit_code") if isinstance(info, Mapping) else None
        if isinstance(code, int) and not isinstance(code, bool):
            return Cause(
                "pipe_closed:exit",
                f"the agent process exited with code {code}",
                "agent",
                _fill(
                    "the agent's log (agent/<agent>.txt) has its last words; "
                    "`bench eval resume {job}` reruns it",
                    job,
                    task,
                ),
            )
    if category in _ERRORED:
        label, fault, step = _ERRORED[category]
        return Cause(category, label, fault, _fill(step, job, task))
    error = " ".join(str(data.get("error") or "").split())
    short = error if len(error) <= 80 else error[:77] + "..."
    return Cause(
        "other",
        f"other error: {short}",
        "infrastructure",
        "the trial's result.json and the run's log have the details",
    )


def task_dir_finder(job_dir: Path | None) -> Callable[[str], Path | None]:
    """Where each of a job's tasks lives, from the tasks_dir its evaluation.json
    recorded (read once)."""
    root: Path | None = None
    if job_dir is not None:
        try:
            record = json.loads((Path(job_dir) / "evaluation.json").read_text())
        except (OSError, ValueError):
            record = None
        tasks_dir = record.get("tasks_dir") if isinstance(record, dict) else None
        if isinstance(tasks_dir, str) and tasks_dir:
            root = Path(tasks_dir)

    def find(task_name: str) -> Path | None:
        if root is None:
            return None
        if root.name == task_name and not (root / task_name).is_dir():
            return root
        return root / task_name

    return find
