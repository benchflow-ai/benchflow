"""One-command smoke run behind ``bench eval smoke``.

Runs the bundled hello-world task (``benchflow/demo_task``) once per agent that
``bench doctor`` found a working credential for, strictly one run at a time,
and reports agent, reward, wall time and trajectory path for each.

Each run is a separate ``bench eval run`` subprocess. That keeps one agent's
crash or hang from taking the others down, lets a hung run be killed with its
whole process group, and means the command in each log is exactly what the
user would type to reproduce it. Results are read back from the run's
``result.json``; the subprocess exit code is only used when no result exists.

:func:`run_smoke` takes the runner as an argument, so the orchestration is
tested with fakes and never starts Docker or a model.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from benchflow.doctor import DoctorReport, redact, secret_values

SmokeStatus = Literal["pass", "fail", "error"]

SMOKE_TASK_NAME = "hello-world"
BUNDLED_TASK_DIR = Path(__file__).parent / "demo_task"

# Cheap, fast default model per agent. Codex's matches the ChatGPT-subscription
# quickstart in README.md; Claude's is the eval default (Haiku 4.5).
SMOKE_DEFAULT_MODELS: dict[str, str] = {
    "claude-agent-acp": "claude-haiku-4-5-20251001",
    "codex-acp": "gpt-5.5",
    "gemini": "gemini-2.5-flash",
}
DEFAULT_TIMEOUT_SEC = 900
_TERMINATE_GRACE_SEC = 30


@dataclass(frozen=True)
class SmokeTarget:
    agent: str
    model: str
    auth: str  # credential name/path only, never a value
    explicit: bool = False


@dataclass(frozen=True)
class SmokeSkip:
    agent: str
    reason: str


@dataclass(frozen=True)
class SmokeJob:
    target: SmokeTarget
    task_dir: Path
    jobs_dir: Path
    log_path: Path
    sandbox: str
    timeout_sec: float


@dataclass(frozen=True)
class SmokeOutcome:
    target: SmokeTarget
    status: SmokeStatus
    reward: float | None
    seconds: float
    log_path: Path
    rollout_dir: Path | None = None
    trajectory: Path | None = None
    reason: str = ""
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.target.agent,
            "model": self.target.model,
            "auth": self.target.auth,
            "status": self.status,
            "reward": self.reward,
            "seconds": round(self.seconds, 1),
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            "trajectory": str(self.trajectory) if self.trajectory else None,
            "log": str(self.log_path),
            "reason": self.reason,
            "exit_code": self.exit_code,
        }


SmokeRunner = Callable[[SmokeJob], SmokeOutcome]


# ── Planning ────────────────────────────────────────────────────────────


def parse_agent_request(spec: str) -> tuple[str, str | None]:
    """``codex=gpt-5.6-sol`` -> ``("codex-acp", "gpt-5.6-sol")``; aliases resolve."""
    from benchflow.agents.registry import AGENT_ALIASES

    name, sep, model = spec.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"--agent {spec!r}: missing agent name")
    if sep and not model.strip():
        raise ValueError(f"--agent {spec!r}: empty model after '='")
    return AGENT_ALIASES.get(name, name), (model.strip() or None)


def _default_model(agent: str) -> str:
    if agent in SMOKE_DEFAULT_MODELS:
        return SMOKE_DEFAULT_MODELS[agent]
    from benchflow.evaluation import effective_model

    model = effective_model(agent, None)
    if not model:
        raise ValueError(
            f"agent {agent!r} has no default model; pass --agent {agent}=<model>"
        )
    return model


def _short_name(agent: str) -> str:
    from benchflow.agents.registry import AGENT_ALIASES

    aliases = [a for a, target in AGENT_ALIASES.items() if target == agent]
    return min(aliases, key=lambda alias: len(alias)) if aliases else agent


def _auth_label(report: DoctorReport, agent: str) -> str:
    auth = report.agents.get(agent)
    if auth is None or auth.effective is None:
        return "none found"
    src = auth.effective
    return src.name if src.origin == "file" else f"{src.name} ({src.origin})"


def plan_smoke(
    report: DoctorReport, requested: Iterable[str] = ()
) -> tuple[list[SmokeTarget], list[SmokeSkip]]:
    """Pick the agents to smoke.

    With no ``requested`` agents, every smoke-capable agent whose credential
    doctor marked ready (and whose model endpoint answered) runs; the rest are
    reported as skipped with the reason. Explicitly requested agents always
    run, so an expired login can be tried on purpose.
    """
    from benchflow.agents.registry import AGENTS

    targets: list[SmokeTarget] = []
    skipped: list[SmokeSkip] = []
    requested = list(requested)
    if requested:
        seen: set[str] = set()
        for spec in requested:
            agent, model = parse_agent_request(spec)
            if agent not in AGENTS:
                raise ValueError(f"unknown agent {agent!r} (see `bench agent list`)")
            if agent in seen:
                continue
            seen.add(agent)
            targets.append(
                SmokeTarget(
                    agent,
                    model or _default_model(agent),
                    _auth_label(report, agent),
                    explicit=True,
                )
            )
        return targets, skipped

    for agent, model in SMOKE_DEFAULT_MODELS.items():
        auth = report.agents.get(agent)
        alias = _short_name(agent)
        if auth is None or auth.effective is None:
            skipped.append(SmokeSkip(agent, "no credential found"))
            continue
        if not auth.ready:
            note = auth.effective.note or "credential not usable"
            skipped.append(
                SmokeSkip(
                    agent,
                    f"{auth.effective.name}: {note}; pass --agent {alias} to try it anyway",
                )
            )
            continue
        blocked = report.agent_blocked_by_network(agent)
        if blocked:
            skipped.append(
                SmokeSkip(
                    agent,
                    "model endpoint unreachable: "
                    + ", ".join(redact(url, ()) for url in blocked),
                )
            )
            continue
        targets.append(SmokeTarget(agent, model, _auth_label(report, agent)))
    return targets, skipped


# ── Running ─────────────────────────────────────────────────────────────


def stage_task(dest_root: Path, source: Path = BUNDLED_TASK_DIR) -> Path:
    """Copy the bundled task out of site-packages so runs never write into it."""
    dest = dest_root / SMOKE_TASK_NAME
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def eval_command(job: SmokeJob) -> list[str]:
    """The ``bench eval run`` argv for one smoke run (no secrets on argv)."""
    return [
        "eval",
        "run",
        "--tasks-dir",
        str(job.task_dir),
        "--agent",
        job.target.agent,
        "--model",
        job.target.model,
        "--sandbox",
        job.sandbox,
        "--concurrency",
        "1",
        "--jobs-dir",
        str(job.jobs_dir),
        "--quiet",
    ]


def _bench_argv(args: list[str]) -> list[str]:
    # Run the same interpreter and package as this process, whatever `bench`
    # happens to resolve to on PATH.
    launcher = "from benchflow.cli.main import app; app(prog_name='bench')"
    return [sys.executable, "-c", launcher, *args]


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


def _stop(proc: subprocess.Popen) -> None:
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SEC)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL)
        proc.wait()


def subprocess_runner(job: SmokeJob) -> SmokeOutcome:
    """Run one smoke job as a ``bench eval run`` subprocess, output to its log."""
    args = eval_command(job)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.jobs_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with job.log_path.open("w") as log:
        log.write("$ bench " + shlex.join(args) + "\n")
        log.flush()
        proc = subprocess.Popen(
            _bench_argv(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            # Own process group: a timeout kills bench and everything it spawned.
            start_new_session=True,
        )
        try:
            exit_code: int | None = proc.wait(timeout=job.timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop(proc)
            exit_code = proc.returncode
        except KeyboardInterrupt:
            # The child is in its own session, so the terminal's Ctrl-C does
            # not reach it; forward the interrupt, then stop it.
            _signal_group(proc, signal.SIGINT)
            try:
                proc.wait(timeout=_TERMINATE_GRACE_SEC)
            except subprocess.TimeoutExpired:
                _stop(proc)
            raise
    return read_outcome(
        job,
        seconds=time.monotonic() - started,
        exit_code=exit_code,
        timed_out=timed_out,
    )


def _log_hint(log_path: Path, secrets: list[str]) -> str:
    """The last line of the log that looks like an error, for the summary."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    markers = ("error", "Error", "ERROR", "Traceback", "failed", "Failed", "FAIL")
    # Redact each line before truncating: cutting a line that still holds a
    # secret can keep a prefix redact() no longer matches (a partial-key leak).
    for line in reversed(lines):
        text = redact(line.strip(), secrets)
        if text and any(marker in text for marker in markers):
            return text[:240]
    for line in reversed(lines):
        text = redact(line.strip(), secrets)
        if text:
            return text[:240]
    return ""


def _one_line(text: Any, limit: int = 240, *, secrets: Iterable[str] = ()) -> str:
    # Redact before the truncation to ``limit`` for the same reason as
    # _log_hint: a secret split by the cut would leak its surviving prefix.
    for line in str(text).splitlines():
        line = redact(line.strip(), secrets)
        if line:
            return line[:limit]
    return ""


def read_outcome(
    job: SmokeJob,
    *,
    seconds: float,
    exit_code: int | None,
    timed_out: bool = False,
    environ: Mapping[str, str] | None = None,
) -> SmokeOutcome:
    """Turn a finished run's ``result.json`` into a :class:`SmokeOutcome`."""
    secrets = secret_values(os.environ if environ is None else environ)
    results = sorted(
        job.jobs_dir.glob("*/*/result.json"), key=lambda p: p.stat().st_mtime
    )
    if not results:
        why = (
            f"timed out after {int(job.timeout_sec)}s"
            if timed_out
            else f"bench exited {exit_code} without writing result.json"
        )
        hint = _log_hint(job.log_path, secrets)
        return SmokeOutcome(
            job.target,
            "error",
            None,
            seconds,
            job.log_path,
            reason=f"{why}: {hint}" if hint else why,
            exit_code=exit_code,
        )
    result_path = results[-1]
    rollout_dir = result_path.parent
    try:
        data = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return SmokeOutcome(
            job.target,
            "error",
            None,
            seconds,
            job.log_path,
            rollout_dir=rollout_dir,
            reason=f"unreadable result.json: {type(exc).__name__}",
            exit_code=exit_code,
        )
    rewards = data.get("rewards") if isinstance(data, dict) else None
    raw_reward = rewards.get("reward") if isinstance(rewards, dict) else None
    reward = (
        float(raw_reward)
        if isinstance(raw_reward, int | float) and not isinstance(raw_reward, bool)
        else None
    )
    traj = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    trajectory = traj if traj.is_file() and traj.stat().st_size > 0 else None
    error = data.get("error") if isinstance(data, dict) else None
    verifier_error = data.get("verifier_error") if isinstance(data, dict) else None
    status: SmokeStatus
    reason = ""
    if error:
        status = "error"
        category = data.get("error_category") or "agent error"
        reason = f"{category}: {_one_line(error, secrets=secrets)}"
    elif verifier_error:
        status = "error"
        category = data.get("verifier_error_category") or "verifier error"
        reason = f"{category}: {_one_line(verifier_error, secrets=secrets)}"
    elif reward is None:
        status = "error"
        reason = "no reward recorded"
    elif reward >= 1.0 and trajectory is not None:
        status = "pass"
    elif reward >= 1.0:
        status = "fail"
        reason = "reward 1.0 but no ACP trajectory was captured"
    else:
        status = "fail"
        reason = f"agent did not solve the task (reward {reward:g})"
    if timed_out and status == "pass":
        status = "error"
        reason = f"timed out after {int(job.timeout_sec)}s after writing a result"
    return SmokeOutcome(
        job.target,
        status,
        reward,
        seconds,
        job.log_path,
        rollout_dir=rollout_dir,
        trajectory=trajectory,
        reason=redact(reason, secrets),
        exit_code=exit_code,
    )


def run_smoke(
    targets: list[SmokeTarget],
    *,
    root: Path,
    sandbox: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    runner: SmokeRunner | None = None,
    on_start: Callable[[int, SmokeTarget], None] | None = None,
    on_done: Callable[[int, SmokeOutcome], None] | None = None,
) -> list[SmokeOutcome]:
    """Run each target once, one after another, and write ``smoke-summary.json``."""
    runner = runner or subprocess_runner
    root.mkdir(parents=True, exist_ok=True)
    task_dir = stage_task(root / "task")
    started_at = datetime.now(UTC)
    outcomes: list[SmokeOutcome] = []
    for index, target in enumerate(targets):
        job = SmokeJob(
            target=target,
            task_dir=task_dir,
            jobs_dir=root / target.agent,
            log_path=root / "logs" / f"{target.agent}.log",
            sandbox=sandbox,
            timeout_sec=timeout_sec,
        )
        if on_start:
            on_start(index, target)
        outcome = runner(job)
        outcomes.append(outcome)
        if on_done:
            on_done(index, outcome)
    write_summary(root, outcomes, sandbox=sandbox, started_at=started_at)
    return outcomes


def write_summary(
    root: Path,
    outcomes: list[SmokeOutcome],
    *,
    sandbox: str,
    started_at: datetime,
) -> Path:
    from benchflow import __version__

    path = root / "smoke-summary.json"
    payload = {
        "ok": bool(outcomes) and all(o.status == "pass" for o in outcomes),
        "benchflow_version": __version__,
        "sandbox": sandbox,
        "task": SMOKE_TASK_NAME,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "runs": [o.to_dict() for o in outcomes],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
