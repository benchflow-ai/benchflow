"""Scripted stand-ins for the sandboxes behind ``bench hillclimb``.

- :func:`make_tasks` writes tiny task folders (``task.toml``, instruction,
  Dockerfile, verifier, oracle).
- :class:`FakeAgent` replaces ``Evaluation._run_single_task``, the one call
  that starts a sandbox. It writes a real trial folder (``result.json``,
  trajectory, verifier output) whose reward is a scripted function of the
  task, the deployed surface and the trial number. Everything around it runs
  for real: ``Evaluation`` (job folders, ``summary.json``), ``bf.load_job``,
  the hillclimb bookkeeping.
- :class:`FakeProposer` replaces ``proposer.run_rollout`` (``bf.run`` for the
  optimizer). Like a sandbox, it sees nothing but ``RolloutConfig.uploads``:
  it copies them into a private folder, applies a scripted edit there, runs
  the wrapper task's real ``tests/validate.py``, and leaves the trial folder
  the wrapper's ``test.sh`` would (``verifier/surface``,
  ``verifier/proposal.json``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.evaluation import Evaluation
from benchflow.hillclimbing import proposer as proposer_mod
from benchflow.models import RolloutResult

RewardFn = Callable[[str, str, int], float | None]


def instruction_of(task: str) -> str:
    return (
        f"Task {task}: compute the quarterly flood report for station {task} "
        f"and write the result to /app/{task}-answer.txt using the gauge readings "
        f"in /data. Identifier phrase unique to {task} for leak checks."
    )


def make_tasks(root: Path, names: list[str], *, category=lambda i: "hydro") -> Path:
    for i, name in enumerate(names):
        d = root / name
        (d / "environment").mkdir(parents=True)
        (d / "tests").mkdir()
        (d / "solution").mkdir()
        (d / "task.toml").write_text(
            'version = "1.0"\n[metadata]\n'
            f'category = "{category(i)}"\n'
            "[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n"
        )
        (d / "instruction.md").write_text(instruction_of(name) + "\n")
        (d / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (d / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
        (d / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    return root


def surface_text(cfg: Any) -> str:
    """Everything the agent under test would receive from the surface."""
    parts: list[str] = []
    if cfg.skills_dir:
        for path in sorted(Path(cfg.skills_dir).rglob("*")):
            if path.is_file():
                parts.append(path.read_text())
    override = cfg.config_override or {}
    prefix = (override.get("agent") or {}).get("prompt_prefix")
    if prefix:
        parts.append(prefix)
    return "\n".join(parts)


@dataclass
class FakeAgent:
    """The agent under test, scripted: ``reward(task, surface, trial)``."""

    reward: RewardFn
    cost: Callable[[str, str, int], float | None] = lambda task, surface, trial: 0.01
    oracle: Callable[[str], float | None] = lambda task: 1.0
    nop: Callable[[str], float | None] = lambda task: 0.0
    verifier_text: Callable[[str], str] = lambda task: f"FAILED test_{task}_output\n"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def install(self, monkeypatch) -> FakeAgent:
        fake = self

        async def run_single_task(ev: Evaluation, task_dir: Path, cfg: Any) -> RolloutResult:
            return fake.run(ev, task_dir, cfg)

        monkeypatch.setattr(Evaluation, "_run_single_task", run_single_task)
        return self

    def run(self, ev: Evaluation, task_dir: Path, cfg: Any) -> RolloutResult:
        task = task_dir.name
        folder = ev._jobs_dir.name
        trial = int(folder.split("-")[1]) if folder.startswith("trial-") else 1
        surface = surface_text(cfg)
        if cfg.agent == "oracle":
            reward, cost = self.oracle(task), None
        elif cfg.agent == "nop":
            reward, cost = self.nop(task), None
        else:
            reward = self.reward(task, surface, trial)
            cost = self.cost(task, surface, trial)
        self.calls.append(
            {
                "task": task,
                "trial": trial,
                "agent": cfg.agent,
                "jobs_dir": str(ev._jobs_dir),
                "surface": surface,
            }
        )
        rollout_name = f"{task}__{uuid.uuid4().hex[:8]}"
        rollout_dir = ev._jobs_dir / ev._job_name / rollout_name
        (rollout_dir / "trajectory").mkdir(parents=True)
        (rollout_dir / "verifier").mkdir()
        error = error_category = None
        if reward is None:
            error = "sandbox setup failed: scripted infrastructure error"
            error_category = "sandbox_setup"
        payload = {
            "task_name": task,
            "rollout_name": rollout_name,
            "agent": cfg.agent,
            "model": cfg.model,
            "rewards": None if reward is None else {"reward": reward},
            "error": error,
            "error_category": error_category,
            "verifier_error": None,
            "n_tool_calls": 3,
            "agent_result": {"cost_usd": cost, "total_tokens": 100},
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        (rollout_dir / "trajectory" / "acp_trajectory.jsonl").write_text(
            json.dumps({"type": "tool_call", "title": f"cat /app/{task}.csv"}) + "\n"
        )
        if reward is not None:
            (rollout_dir / "verifier" / "reward.txt").write_text(f"{reward}\n")
            (rollout_dir / "verifier" / "test-stdout.txt").write_text(
                self.verifier_text(task)
            )
        return RolloutResult(
            task_name=task,
            rollout_name=rollout_name,
            rewards=payload["rewards"],
            agent=cfg.agent,
            model=cfg.model,
            n_tool_calls=3,
            cost_usd=cost,
            total_tokens=100,
            error=error,
            error_category=error_category,
            rollout_dir=rollout_dir,
        )


Edit = Callable[[Path], dict[str, Any] | None]


def append_to_skill(text: str, change: str = "added a step") -> Edit:
    """An edit that appends ``text`` to the first skill's SKILL.md."""

    def edit(surface: Path) -> dict[str, Any]:
        skill = sorted((surface / "skills").glob("*/SKILL.md"))[0]
        skill.write_text(skill.read_text() + f"\n{text}\n")
        return {
            "root_cause": "the agent skips a step",
            "change": change,
            "rationale": "the step is general",
            "evidence": ["t0/trial-01"],
        }

    return edit


@dataclass
class FakeProposer:
    """The optimizer, scripted: one edit per proposer call, in order."""

    edits: list[Edit] = field(default_factory=list)
    analysis: dict[str, Any] | None = None
    probe: Callable[[Path], None] | None = None
    cost_usd: float | None = 0.05
    # Where the private sandbox copies go (outside the run folder).
    root: Path | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)
    sandboxes: list[Path] = field(default_factory=list)

    def install(self, monkeypatch) -> FakeProposer:
        monkeypatch.setattr(proposer_mod, "run_rollout", self.run)
        return self

    async def run(self, config: Any) -> RolloutResult:
        jobs_dir = Path(config.jobs_dir)
        sandbox = (self.root or jobs_dir.parent) / f"sandbox-{len(self.runs)}"
        # Like a sandbox: only the uploads exist inside.
        for host, target in config.uploads.items():
            dest = sandbox / target.lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(host, dest)
            for path in [dest, *dest.rglob("*")]:
                path.chmod(path.stat().st_mode | 0o200)
        self.sandboxes.append(sandbox)
        mode = "analyze" if config.task_path.name.endswith("analyze") else "propose"
        self.runs.append(
            {"mode": mode, "uploads": dict(config.uploads), "sandbox": sandbox}
        )
        if self.probe is not None:
            self.probe(sandbox)
        workdir = sandbox / "app"
        output_name = "proposal.json" if mode == "propose" else "analysis.json"
        if mode == "propose":
            index = sum(1 for r in self.runs if r["mode"] == "propose") - 1
            edit = self.edits[index] if index < len(self.edits) else None
            output = edit(workdir / "surface") if edit else None
        else:
            output = self.analysis
        if output is not None:
            (workdir / output_name).write_text(json.dumps(output))
        # What the wrapper's tests/test.sh does, with its real validator.
        rollout_name = f"{config.task_path.name}__{uuid.uuid4().hex[:8]}"
        rollout_dir = jobs_dir / "job" / rollout_name
        verifier = rollout_dir / "verifier"
        verifier.mkdir(parents=True)
        if (workdir / output_name).is_file():
            shutil.copyfile(workdir / output_name, verifier / output_name)
        if (workdir / "surface").is_dir():
            shutil.copytree(workdir / "surface", verifier / "surface")
        check = subprocess.run(
            [sys.executable, str(config.task_path / "tests" / "validate.py"), mode, str(workdir)],
            capture_output=True,
            text=True,
        )
        reward = 1.0 if check.returncode == 0 else 0.0
        (verifier / "reward.txt").write_text(f"{reward}\n")
        (verifier / "test-stdout.txt").write_text(check.stdout)
        payload = {
            "task_name": config.task_path.name,
            "rollout_name": rollout_name,
            "rewards": {"reward": reward},
            "n_tool_calls": 5,
            "agent_result": {"cost_usd": self.cost_usd},
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        return RolloutResult(
            task_name=config.task_path.name,
            rollout_name=rollout_name,
            rewards={"reward": reward},
            n_tool_calls=5,
            cost_usd=self.cost_usd,
            rollout_dir=rollout_dir,
        )


def read_tree(root: Path) -> dict[str, str]:
    """Every file under ``root``: relative path -> text."""
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = path.read_text(errors="replace")
    return out
