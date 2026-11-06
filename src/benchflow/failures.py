"""Why a trial has no score, whose fault it was, and what to do next.

One table for the end-of-run summary and for docs/jobs/when-a-run-fails.md. A
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

Each cause has an id ``bench explain`` has a page for (:attr:`Cause.code`):
Rust's run failure code where the meaning is the same
(:data:`RUST_CODES`), else the cause's key. :func:`cause_keys` lists every
key :func:`cause_of` returns.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow._utils.command_hints import command_arg
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

    @property
    def code(self) -> str:
        """The id ``bench explain`` explains: Rust's code, or the key."""
        return RUST_CODES.get(self.key, self.key)


# "SomeError: " opening an error message.
_EXCEPTION_PREFIX = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Unavailable): ")

# The opening of verifier_core's message for a plugin the guard refused; the
# guard's module name is one shell word that hardening chose.
_GUARD_REFUSED_RE = re.compile(r"the pytest plugin guard \S+ refused a pytest plugin ")
# The same refusal with no record of the agent's files to say who put the
# plugin there (V2-R2-A-1); the words after the guard's name are its own.
_GUARD_UNRECORDED_RE = re.compile(
    r"the pytest plugin guard \S+ refused a pytest plugin it could not trust, "
    r"with no complete record of the agent's files "
)
# The same refusal of a plugin test.sh installed after the agent stopped, from
# inputs the agent changed (V2-R1-B-2), or of a vouched release the host's
# table of published releases lacks (V2-R1-B-1).
_GUARD_UNPROVEN_RE = re.compile(
    r"the pytest plugin guard \S+ refused a pytest plugin it could not trust, "
    r"which the verifier installed after the agent stopped "
)
_GUARD_UNLISTED_RE = re.compile(
    r"the pytest plugin guard \S+ refused a pytest plugin it could not trust, "
    r"a release of a plugin BenchFlow vouches for "
)

# category -> (label, fault, next step). {job} is the job folder, {task} the
# task folder; both are filled in when known.
_ERRORED: dict[str, tuple[str, Fault, str]] = {
    "kept_service_freezer": (
        "the sandbox cannot freeze the task's kept services",
        "setup",
        "run with --sandbox docker or --sandbox remote-docker; "
        "Daytona also supports a Compose environment",
    ),
    "runtime_sealing": (
        "the protected runtime could not be sealed",
        "task",
        "fix the image's runtime paths and ownership named in result.json; "
        "keep them root-owned and outside the agent's writable workspace",
    ),
    "guard_timeout": (
        "the sandbox guard timed out or did not report",
        "infrastructure",
        "the error in result.json and the sandbox setup log say where the "
        "guard stopped; `bench eval resume {job}` runs the trial again",
    ),
    "invalid_provider_response": (
        "the provider returned an invalid response",
        "infrastructure",
        "check trajectory/provider_requests.jsonl for upstream status and "
        "timing; retry or use another provider endpoint",
    ),
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
        "raise the task's agent timeout (task.md agent.timeout_sec) or "
        "--timeout-multiplier",
    ),
    "infra_failure": (
        "the sandbox or its transport failed",
        "infrastructure",
        "`bench eval resume {job}` reruns it",
    ),
    "sandbox_setup": (
        "the sandbox did not start",
        "infrastructure",
        "`bench doctor`; error in result.json has the whole message",
    ),
    "image_build": (
        "the task's image did not build",
        "task",
        "fix that line in the task's environment/Dockerfile; `docker build "
        "{task}/environment` prints the whole build log",
    ),
    "acp_error": (
        "the agent reported an error",
        "agent",
        "the agent's log (agent/<agent>.txt in the trial folder) has its side",
    ),
    "registry_auth": (
        "the registry refused this machine's login for the base image",
        "setup",
        "check `docker login <registry>` on this machine, and that the FROM "
        "line names an image that login can read",
    ),
    "task_config": (
        "a task file does not load",
        "task",
        "`bench check {task}` names the problem; fix it, then run the task again",
    ),
    "disk_full": (
        "the disk that holds the job folder is full",
        "setup",
        "free space on that disk, or pass another --jobs-dir; then "
        "`bench eval resume {job}`",
    ),
    "unsupported_task_feature": (
        "the task asks for something this sandbox backend cannot do",
        "setup",
        "the error names what; run the task with `--sandbox docker`, or change "
        "what it asks for",
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
# An agent_integration failure whose cause is none of the above.
_INTEGRATION_UNKNOWN: tuple[str, Fault, str] = (
    "it did nothing useful",
    "agent",
    "integration_failure_info in result.json",
)
_UNSCORED: dict[str, tuple[str, Fault, str]] = {
    "verifier_failure": (
        "the verifier crashed",
        "task",
        "fix the verifier from its output (the trial's verifier/test-stdout.txt, "
        "whose last lines the summary shows), then run the task again",
    ),
    "verifier_timeout": (
        "the verifier timed out",
        "task",
        "raise the task's verifier timeout (task.md verifier.timeout_sec)",
    ),
    "verifier_dep_install": (
        "the verifier's dependency install failed",
        "task",
        "pin or bake the verifier's dependencies into the image; `bench check {task}`",
    ),
    "reward_out_of_range": (
        "the verifier wrote a reward outside the task's range",
        "task",
        "fix the verifier, or declare [verifier] reward_range in task.toml if "
        "the task means a wider range; reward_out_of_range_info in result.json "
        "keeps the raw value",
    ),
    "verifier_infra": (
        "the verifier's sandbox failed",
        "infrastructure",
        "`bench eval resume {job}` scores it again",
    ),
    "disk_full": (
        "the disk that holds the job folder is full, so the reward was lost",
        "setup",
        "free space on that disk, or pass another --jobs-dir; then "
        "`bench eval resume {job}`",
    ),
    "verifier_other": (
        "the verifier failed",
        "task",
        "the trial's verifier/ folder has the output",
    ),
}
# The causes no stored category names. Their labels are the general form:
# cause_of fills in the login, window and reset of a usage limit, the exit
# code of an agent process, and the first words of an other error.
_SPECIAL: dict[str, tuple[str, Fault, str]] = {
    "usage_limit": (
        "usage limit",
        "setup",
        "switch to another login or wait for the reset, then "
        "`bench eval resume {job}`; `bench doctor` shows each login's "
        "headroom",
    ),
    "pipe_closed:exit": (
        "the agent process exited",
        "agent",
        "the agent's log (agent/<agent>.txt) has its last words; "
        "`bench eval resume {job}` reruns it",
    ),
    "kept_service_freezer:engine": (
        "the Docker Engine cannot freeze the task's kept services",
        "setup",
        "use Docker Engine 28 or later with cgroup v2, not rootless and "
        "without user-namespace remapping (`docker info`)",
    ),
    "kept_service_freezer:probe": (
        "the sandbox's cgroups cannot freeze the task's kept services",
        "setup",
        "run it where Docker uses cgroup v2 with writable cgroups "
        "(`docker info` shows Cgroup Version: 2); the check's reason is in "
        "result.json",
    ),
    "other": (
        "other error",
        "infrastructure",
        "the trial's result.json and the run's log have the details",
    ),
    "verifier_plugin_trust": (
        "verifier plugin trust: test.sh installs a pytest plugin where "
        "the agent could write",
        "task",
        "`bench check {task}`",
    ),
    "verifier_plugin_unrecorded": (
        "verifier plugin trust: the plugin guard refused a pytest plugin "
        "in a place the agent could write, and with no record of the "
        "agent's files it cannot tell who put it there",
        "task",
        "the trial's verifier/ folder names the files; when test.sh "
        "installs the plugin, install it into the image: "
        "`bench tasks check {task}`",
    ),
    "verifier_plugin_unproven": (
        "verifier plugin trust: test.sh installs a pytest plugin BenchFlow "
        "cannot check, and the agent changed files that could have chosen it",
        "task",
        "pin the plugin's release in [verifier] pytest_plugin_releases or "
        "install it into the image: `bench tasks check {task}`",
    ),
    "verifier_plugin_unlisted": (
        "verifier plugin trust: test.sh installed a pytest-json-ctrf release "
        "missing from the table of published releases",
        "infrastructure",
        "check that this machine reaches pypi.org, then run the task again; "
        "the trial's verifier/ folder names the release",
    ),
    "verifier_plugin_refused": (
        "verifier plugin trust: the plugin guard refused a pytest plugin "
        "in a place the agent could write",
        "agent",
        "the trial's verifier/ folder names the plugin and why; when "
        "test.sh installs it, `bench tasks check {task}`",
    ),
    "no_verdict": (
        "no verdict was recorded (an assessment still pending, or an "
        "interrupted trial)",
        "infrastructure",
        "`bench eval score {job}` finishes pending reviews",
    ),
}

# Explain ids. A cause whose meaning is the same as one of Rust
# bench 0.8.2's run failure codes takes that code, so an id means one thing in
# both engines; every other cause keeps its key as its id. Not mapped where
# Rust's code means more or less than the cause: R214 (any provider failure)
# against api_error and provider_rate_limit, R103 (a sandbox that did not
# start or was lost) against sandbox_setup and infra_failure, R304 against
# verifier_failure (Python's also covers an unreadable reward), R201 (an
# install layer on an image that builds) against install_failure (any
# install, retried as infrastructure).
RUST_CODES: dict[str, str] = {
    "provider_auth": "R212",
    "usage_limit": "R213",
    "idle_timeout": "R203",
    "pipe_closed:exit": "R204",
    "acp_error": "R205",
    "verifier_timeout": "R303",
    "unsupported_task_feature": "R401",
}


def _integration_key(why: object) -> str:
    known = isinstance(why, str) and why in _INTEGRATION
    return f"agent_integration:{why if known else 'unknown'}"


def unscored_keys() -> tuple[str, ...]:
    """The keys of the causes of unscored trials (the agent finished)."""
    return (
        *_UNSCORED,
        "verifier_plugin_trust",
        "verifier_plugin_unrecorded",
        "verifier_plugin_unproven",
        "verifier_plugin_unlisted",
        "verifier_plugin_refused",
        "no_verdict",
    )


def cause_keys() -> tuple[str, ...]:
    """Every key :func:`cause_of` can return: unscored causes, then errored ones.

    A key both kinds share (``disk_full``) is listed once, as unscored.
    """
    return tuple(
        dict.fromkeys(
            (
                *unscored_keys(),
                *_ERRORED,
                *(_integration_key(why) for why in (*_INTEGRATION, None)),
                "usage_limit",
                "pipe_closed:exit",
                "kept_service_freezer:engine",
                "kept_service_freezer:probe",
                "other",
            )
        )
    )


def describe(key: str) -> Cause | None:
    """The cause ``key`` names in its general form, or None for no such key.

    {job} and {task} read ``<job dir>`` and ``<task>``; the details only a
    trial has (a login, an exit code) are left out.
    """
    if key.startswith("agent_integration:"):
        why = key.split(":", 1)[1]
        if _integration_key(why) != key and why != "unknown":
            return None
        label, fault, step = _INTEGRATION.get(why, _INTEGRATION_UNKNOWN)
        return Cause(key, f"the agent integration broke: {label}", fault, _fill(step))
    for table in (_UNSCORED, _ERRORED, _SPECIAL):
        if key in table:
            label, fault, step = table[key]
            return Cause(key, label, fault, _fill(step))
    return None


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


def _fill(text: str, job: str | None = None, task: str | None = None) -> str:
    # Format the trusted template once: a pathname containing {task} or {job}
    # must never become another substitution (D2b-CONF7-2).
    return text.format(
        job=command_arg(job, "<job dir>"),
        task=command_arg(task, "<task>"),
    )


def cause_of(
    result: Any, *, job_dir: Path | None = None, task_dir: Path | None = None
) -> Cause | None:
    """The cause of an unscored or errored trial; None for a scored one."""
    from benchflow._utils.scoring import (
        classify_error,
        classify_score_outcome,
        classify_verifier_error,
        plugin_guard_error_message,
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
        # Only the guard's own opening words decide the kind: the rest of its
        # message can quote a path or an error the agent chose (V2-R1-A-1).
        guard_message = plugin_guard_error_message(verifier_error) or ""
        if guard_message.startswith("pytest plugin ") and (
            " was installed after the agent stopped, by the verifier" in guard_message
        ):
            return _special("verifier_plugin_trust", job, task)
        if _GUARD_UNRECORDED_RE.match(guard_message):
            # Still a failure in the scores; but nothing says the agent put
            # the plugin there, and the record is missing for the task's
            # reasons (its verifier user, its image), not the agent's.
            return _special("verifier_plugin_unrecorded", job, task)
        if _GUARD_UNPROVEN_RE.match(guard_message):
            # Still a failure: the agent's files may have chosen the install.
            # But test.sh installed it, unpinned: the task's problem.
            return _special("verifier_plugin_unproven", job, task)
        if _GUARD_UNLISTED_RE.match(guard_message):
            # Still a failure, since the agent can leave inputs that install a
            # release by any name; but the table, not the agent, is behind it.
            return _special("verifier_plugin_unlisted", job, task)
        if _GUARD_REFUSED_RE.match(guard_message):
            return _special("verifier_plugin_refused", job, task)
        # A category none of the rows names (a record from another version)
        # is the general row, under its key: the key is the explain id.
        key = category if category in _UNSCORED else "verifier_other"
        label, fault, step = _UNSCORED[key]
        return Cause(key, label, fault, _fill(step, job, task))
    return _special("no_verdict", job, task)


def _special(key: str, job: str | None, task: str | None, label: str = "") -> Cause:
    general, fault, step = _SPECIAL[key]
    return Cause(key, label or general, fault, _fill(step, job, task))


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
        return _special("usage_limit", job, task, ", ".join(parts))
    if category == "agent_integration":
        info = data.get("integration_failure_info")
        why = info.get("cause") if isinstance(info, Mapping) else None
        # An unknown cause shares the "unknown" row and key; the key is the
        # explain id, so it never carries a word from the record.
        label, fault, step = _INTEGRATION.get(str(why), _INTEGRATION_UNKNOWN)
        return Cause(
            _integration_key(why),
            f"the agent integration broke: {label}",
            fault,
            _fill(step, job, task),
        )
    if category == "pipe_closed":
        info = data.get("transport_error_info")
        code = info.get("process_exit_code") if isinstance(info, Mapping) else None
        if isinstance(code, int) and not isinstance(code, bool):
            return _special(
                "pipe_closed:exit",
                job,
                task,
                f"the agent process exited with code {code}",
            )
    if category == "kept_service_freezer":
        # One category, three causes, each with its own fix: the sandbox
        # provider (the general row), the Docker Engine, or the cgroup probe.
        error = str(data.get("error") or "")
        if "Use Docker Engine 28 or later" in error:
            return _special("kept_service_freezer:engine", job, task)
        if "this sandbox cannot (the check's reason is below)" in error:
            return _special("kept_service_freezer:probe", job, task)
    if category == "task_config":
        # The file and the fix are the error's own words, after its class;
        # the task's folder is left out of the path (DF-p4-upgrader-8).
        label, fault, step = _ERRORED[category]
        said = str(data.get("error") or "").split(": ", 1)[-1]
        file, sep, rest = said.partition(": ")
        if task and sep:
            # The file under the task's folder, wherever the run named it.
            with contextlib.suppress(ValueError, OSError):
                file = str(
                    Path(os.path.abspath(file)).relative_to(os.path.abspath(task))
                )
            said = f"{file}{sep}{rest}"
        said = said.replace(": - ", ": ").replace("\n- ", "; ")
        return Cause(
            category,
            f"{label}: {_short(said, 200)}",
            fault,
            _fill(step, job, task),
        )
    if category in ("image_build", "registry_auth"):
        # The label names the Dockerfile line and BuildKit's reason, which
        # the error carries after its header (sandbox/image_build.py).
        label, fault, step = _ERRORED[category]
        detail = str(data.get("error") or "").split(f"{label}: ", 1)
        if len(detail) == 2:
            label = f"{label}: {_short(detail[1], 160)}"
        return Cause(category, label, fault, _fill(step, job, task))
    if category == "sandbox_setup" and data.get("error"):
        # The cause on the summary's line (DFR-p5-6): the error after its
        # "sandbox startup failed:" header, or after its class name.
        label, fault, step = _ERRORED[category]
        said = str(data["error"])
        _, sep, rest = said.partition("sandbox startup failed: ")
        said = rest if sep else _EXCEPTION_PREFIX.sub("", said, count=1)
        return Cause(
            category, f"{label}: {_short(said, 360)}", fault, _fill(step, job, task)
        )
    if category in _ERRORED:
        label, fault, step = _ERRORED[category]
        return Cause(category, label, fault, _fill(step, job, task))
    from benchflow._diagnostics import render_diagnostic
    from benchflow.cli._errors import _secrets

    def shorten(fields: Sequence[str]) -> str:
        error = " ".join(fields[0].split())
        return error if len(error) <= 80 else error[:77] + "..."

    short = render_diagnostic((str(data.get("error") or ""),), _secrets(), shorten)
    return _special("other", job, task, f"other error: {short}")


def _short(text: str, limit: int) -> str:
    """``text`` on one line, redacted, cut to ``limit`` characters."""
    from benchflow._diagnostics import render_diagnostic
    from benchflow.cli._errors import _secrets

    def shorten(fields: Sequence[str]) -> str:
        error = " ".join(fields[0].split())
        return error if len(error) <= limit else error[: limit - 3] + "..."

    return render_diagnostic((text,), _secrets(), shorten)


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
