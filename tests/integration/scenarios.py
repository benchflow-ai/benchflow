#!/usr/bin/env python3
"""Reusable building blocks for BenchFlow's integration scenarios.

These helpers layer on BenchFlow's own primitives — ``bench eval run``, the
:mod:`tests.integration.agent_judge` realness+judge gate, and the
``result.json`` / ATIF / ADP artifact contracts — so the integration suite
(:mod:`tests.test_integration_suite`) reads as a set of end-to-end assertions
rather than bespoke glue.

The scenarios are grounded in the v0.6 dogfooding findings:

- **oracle determinism** — an oracle (ground-truth ``solve.sh``) must score
  reward 1.0 on its own verifier; a broken example oracle (``3d-scan-calc``,
  which imported a skill that the no-skill oracle policy never injects) shipped
  precisely because nothing asserted this end to end.
- **docker <-> daytona parity** — the same task on both sandboxes must agree;
  the rollout/daytona package split could regress one backend silently.
- **real agent rollout** — a real harness (openhands + deepseek-v4-flash) must
  produce a REAL measurement (tool calls, tokens, scored) and pass the agent
  judge, not a resumed/empty/idle-timed-out shell.
- **llm-judge verifier** — the agent-as-judge verifier must score, and must
  **fail closed** when the judge can't run (a judge infra error silently
  recorded as reward 0.0 was a real reward-integrity bug).
- **trajectory artifact integrity** — ATIF/ADP must be well-formed and must not
  leak provider/secret values.
- **reaper ownership scoping** — ``environment cleanup --dry-run`` must never
  delete, and the reaper must only touch benchflow-labeled sandboxes.

Most scenarios run live (real sandbox + model); the integrity gates run
deterministically over synthetic rollout fixtures and need no credentials.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Same src bootstrap as the sibling integration tools so the module imports
# without installing the package as a script entry point.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[2]

# Secret shapes that must never appear in a rollout artifact. Kept deliberately
# broad (provider key prefixes) so a regression that writes a key into ATIF/ADP
# or a result blob is caught, while the public provider base URLs are not.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),  # OpenAI / DeepSeek style keys
    re.compile(r"dtn_[A-Za-z0-9]{16,}"),  # Daytona keys
    re.compile(r"AQ\.[A-Za-z0-9_\-]{16,}"),  # Google/Gemini AQ. keys
    re.compile(r"AKIA[0-9A-Z]{12,}"),  # AWS access key ids
)


# ------------------------------------------------------------------
# Running real evals
# ------------------------------------------------------------------


@dataclass(frozen=True)
class EvalOutcome:
    """The result of a ``bench eval run`` subprocess."""

    returncode: int
    jobs_dir: Path
    stdout: str
    stderr: str


def run_eval(
    *,
    jobs_dir: Path,
    agent: str,
    sandbox: str,
    tasks_dir: Path | None = None,
    source_repo: str | None = None,
    source_path: str | None = None,
    source_ref: str | None = None,
    include: Sequence[str] = (),
    model: str | None = None,
    concurrency: int = 1,
    extra_args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    timeout: float = 1800.0,
) -> EvalOutcome:
    """Run ``bench eval run`` and return its outcome.

    A thin, explicit wrapper so every scenario invokes the real CLI the same
    way a user would, with a per-batch ``jobs_dir`` (the resume trap: a reused
    jobs dir silently replays stale results).
    """
    cmd = [
        "uv",
        "run",
        "bench",
        "eval",
        "run",
        "--agent",
        agent,
        "--sandbox",
        sandbox,
        "--concurrency",
        str(concurrency),
        "--jobs-dir",
        str(jobs_dir),
    ]
    if tasks_dir is not None:
        cmd += ["--tasks-dir", str(tasks_dir)]
    if source_repo is not None:
        cmd += ["--source-repo", source_repo]
    if source_path is not None:
        cmd += ["--source-path", source_path]
    if source_ref is not None:
        cmd += ["--source-ref", source_ref]
    for name in include:
        cmd += ["--include", name]
    if model is not None:
        cmd += ["--model", model]
    cmd += list(extra_args)

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    return EvalOutcome(proc.returncode, jobs_dir, proc.stdout, proc.stderr)


def rollout_dirs(jobs_dir: Path) -> list[Path]:
    """Every rollout directory (one ``result.json`` each) under a jobs root."""
    return sorted({p.parent for p in jobs_dir.rglob("result.json")})


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def reward_of(rollout_dir: Path) -> float | None:
    """Canonical scalar reward for a rollout, or ``None`` if unscored.

    Prefers ``rewards.jsonl`` (the verifier's terminal reward event) and falls
    back to ``result.json``'s ``rewards.reward`` — the two places BenchFlow
    records the score.
    """
    rewards_jsonl = rollout_dir / "rewards.jsonl"
    if rewards_jsonl.is_file():
        terminal: float | None = None
        for line in rewards_jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(event, dict)
                and event.get("tag") == "reward"
                and isinstance(event.get("value"), (int, float))
                and not isinstance(event.get("value"), bool)
            ):
                terminal = float(event["value"])
        if terminal is not None:
            return terminal
    result = _read_json(rollout_dir / "result.json")
    if not result:
        return None
    rewards = result.get("rewards")
    if isinstance(rewards, Mapping) and isinstance(rewards.get("reward"), (int, float)):
        return float(rewards["reward"])
    return None


def task_name_of(rollout_dir: Path) -> str:
    result = _read_json(rollout_dir / "result.json") or {}
    return str(result.get("task_name") or rollout_dir.name)


# ------------------------------------------------------------------
# Synthetic rollout fixtures (deterministic, credential-free)
# ------------------------------------------------------------------


def synth_rollout(
    dest: Path,
    *,
    task_name: str = "synthetic-task",
    agent: str = "openhands",
    model: str = "deepseek/deepseek-v4-flash",
    reward: float | None = 1.0,
    n_tool_calls: int = 8,
    n_prompts: int = 1,
    total_tokens: int | None = 120_000,
    error: str | None = None,
    verifier_error: str | None = None,
    prompt: str = "Implement the requested function.",
    trajectory: Sequence[Mapping[str, Any]] | None = None,
    atif: Mapping[str, Any] | None = None,
    adp_lines: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Write a minimal but contract-correct rollout dir for the gate to read.

    Matches what :func:`agent_judge.load_rollout_evidence` consumes, so the
    deterministic integrity scenarios can exercise the realness gate, the
    fail-closed judge paths, and the artifact validators without a live run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "task_name": task_name,
        "agent": agent,
        "model": model,
        "rewards": {"reward": reward},
        "n_tool_calls": n_tool_calls,
        "n_prompts": n_prompts,
        "agent_result": {"total_tokens": total_tokens},
        "error": error,
        "verifier_error": verifier_error,
        "prompt": prompt,
    }
    (dest / "result.json").write_text(json.dumps(result, indent=2))
    if reward is not None:
        (dest / "rewards.jsonl").write_text(
            json.dumps({"type": "terminal", "tag": "reward", "value": reward}) + "\n"
        )
    if trajectory is not None:
        traj_dir = dest / "trajectory"
        traj_dir.mkdir(exist_ok=True)
        (traj_dir / "acp_trajectory.jsonl").write_text(
            "\n".join(json.dumps(e) for e in trajectory) + "\n"
        )
    if atif is not None or adp_lines is not None:
        trainer = dest / "trainer"
        trainer.mkdir(exist_ok=True)
        if atif is not None:
            (trainer / "atif.json").write_text(json.dumps(atif, indent=2))
        if adp_lines is not None:
            (trainer / "adp.jsonl").write_text(
                "\n".join(json.dumps(line) for line in adp_lines) + "\n"
            )
    return dest


# ------------------------------------------------------------------
# Artifact integrity validators
# ------------------------------------------------------------------


def atif_issues(rollout_dir: Path) -> list[str]:
    """Schema problems with a rollout's ATIF trainer artifact.

    Empty list == conformant. Mirrors the ATIF-v1.x contract: a top-level
    ``schema_version`` and a ``steps`` array whose every step declares a
    recognized ``source``.
    """
    path = rollout_dir / "trainer" / "atif.json"
    if not path.is_file():
        return ["trainer/atif.json missing"]
    atif = _read_json(path)
    if atif is None:
        return ["trainer/atif.json is not a JSON object"]
    issues: list[str] = []
    version = atif.get("schema_version")
    if not (isinstance(version, str) and version.startswith("ATIF-v1")):
        issues.append(f"schema_version={version!r} (expected ATIF-v1.x)")
    steps = atif.get("steps")
    if not isinstance(steps, list):
        issues.append("steps is not an array")
        return issues
    valid_sources = {"user", "agent", "oracle"}
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(f"step[{i}] is not an object")
            continue
        if step.get("source") not in valid_sources:
            issues.append(f"step[{i}].source={step.get('source')!r} unrecognized")
    return issues


def adp_issues(rollout_dir: Path) -> list[str]:
    """Schema problems with a rollout's ADP trainer artifact."""
    path = rollout_dir / "trainer" / "adp.jsonl"
    if not path.is_file():
        return ["trainer/adp.jsonl missing"]
    issues: list[str] = []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return ["trainer/adp.jsonl is empty"]
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"adp line {i} is not valid JSON: {exc}")
            continue
        if not isinstance(obj, dict):
            issues.append(f"adp line {i} is not a JSON object")
    return issues


def secret_leak_issues(root: Path) -> list[str]:
    """Files under ``root`` that contain a provider/secret-shaped value.

    Scans text files only (skips obvious binaries). Empty list == clean.
    """
    issues: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                rel = path.relative_to(root)
                issues.append(f"{rel}: matches {pattern.pattern}")
                break
    return issues


# ------------------------------------------------------------------
# Reaper safety (daytona)
# ------------------------------------------------------------------


def reaper_dryrun_issues(env: Mapping[str, str] | None = None) -> list[str]:
    """Assert ``environment cleanup --dry-run`` never deletes.

    Returns problems (non-empty == unsafe). Requires ``DAYTONA_API_KEY``; a
    missing key surfaces as a single "skipped" marker the caller can treat as a
    skip rather than a failure.
    """
    merged = {**os.environ, **(env or {})}
    if not merged.get("DAYTONA_API_KEY"):
        return ["__skip__: DAYTONA_API_KEY not set"]
    proc = subprocess.run(
        ["uv", "run", "bench", "environment", "cleanup", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=merged,
    )
    out = proc.stdout + proc.stderr
    issues: list[str] = []
    if proc.returncode != 0:
        issues.append(f"cleanup --dry-run exited {proc.returncode}: {out[-300:]}")
    # Dry-run must report candidates without claiming deletions.
    if re.search(r"\bdeleted\b", out) and "0 sandboxes deleted" not in out:
        issues.append(f"dry-run reported a deletion: {out[-300:]}")
    return issues
