"""Scripted sandboxes for the hill-climb demo's tests (docs/examples/hillclimb).

- ``FakeAgent`` replaces ``Evaluation._run_single_task``, the one call that
  starts a sandbox: it writes a real trial folder whose reward is a scripted
  function of the task, the deployed skills and the trial number. Everything
  else (``bf.Evaluation`` jobs, ``bf.load_job``, the demo) runs for real.
- ``FakeOptimizer`` replaces ``bf.run`` for the optimizer's rollout. Like a
  sandbox it sees only ``RolloutConfig.uploads``: it copies them into a
  private folder, runs a scripted edit or probe there, runs the demo task's
  own ``tests/validate.py``, and leaves the verifier outputs ``test.sh`` would.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import benchflow as bf


def instruction_of(task: str) -> str:
    return (
        f"Task {task}: compute the quarterly flood report for station {task} and write it "
        f"to /app/{task}-answer.txt from the gauge readings. A phrase unique to {task}."
    )


def make_tasks(root: Path, names: list[str]) -> Path:
    for name in names:
        d = root / name
        for sub in ("environment", "tests", "solution"):
            (d / sub).mkdir(parents=True)
        (d / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n'
        )
        (d / "instruction.md").write_text(instruction_of(name) + "\n")
        (d / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (d / "tests" / "test.sh").write_text("#!/bin/bash\n")
        (d / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    return root


def skills_text(skills_dir) -> str:
    return (
        "\n".join(
            p.read_text() for p in sorted(Path(skills_dir).rglob("*")) if p.is_file()
        )
        if skills_dir
        else ""
    )


def session_log(usd: float, model: str = "claude-haiku-4-5-20251001") -> str:
    """A Claude Code session log: one response and a cost-state line counting it."""
    usage = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    state = {
        model: {
            "inputTokens": 1000,
            "outputTokens": 200,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
            "costUSD": usd,
        }
    }
    return "\n".join(
        json.dumps(line)
        for line in (
            {
                "type": "assistant",
                "requestId": "req_1",
                "message": {"id": "msg_1", "model": model, "usage": usage},
            },
            {"type": "cost-state", "totalCostUSD": usd, "modelUsage": state},
        )
    )


@dataclass
class FakeAgent:
    reward: Callable[
        [str, str, int], float | None
    ]  # (task, skills text, trial) -> reward; None = infra error
    oracle: Callable[[str], float] = lambda task: 1.0
    nop: Callable[[str], float] = lambda task: 0.0
    usd: float | None = 0.01  # what BenchFlow reports for each agent trial
    # Each agent trial leaves a Claude Code session log that says it cost this
    # much (a subscription login, where BenchFlow reports no USD: usd=None).
    session_usd: float | None = None
    seconds: float = 30.0  # each trial's sandbox wall-clock (timing.json)
    # Tokens (CLAUDE_CODE_OAUTH_TOKEN) whose subscription is used up: a trial
    # run on one ends on Claude Code's usage-limit message.
    limited: set[str] = field(default_factory=set)
    calls: list[dict] = field(default_factory=list)

    def install(self, monkeypatch) -> FakeAgent:
        async def run_single_task(ev, task_dir, cfg):
            return self.run(ev, Path(task_dir), cfg)

        monkeypatch.setattr(bf.Evaluation, "_run_single_task", run_single_task)
        return self

    def run(self, ev, task_dir: Path, cfg) -> bf.RolloutResult:
        task, folder = task_dir.name, ev._jobs_dir.name
        trial = int(folder.split("-")[1]) if folder.startswith("trial-") else 1
        skills = skills_text(cfg.skills_dir)
        reward = {
            "oracle": lambda: self.oracle(task),
            "nop": lambda: self.nop(task),
        }.get(cfg.agent, lambda: self.reward(task, skills, trial))()
        token = cfg.agent_env.get("CLAUDE_CODE_OAUTH_TOKEN")
        limited = token in self.limited
        if limited:
            reward = None
        self.calls.append(
            {
                "task": task,
                "trial": trial,
                "agent": cfg.agent,
                "skills": skills,
                "dir": str(ev._jobs_dir),
                "config_override": cfg.config_override,
                "budget": cfg.budget,
                "token": token,
            }
        )
        name = f"{task}__{uuid.uuid4().hex[:8]}"
        out = ev._jobs_dir / ev._job_name / name
        (out / "verifier").mkdir(parents=True)
        (out / "trajectory").mkdir()
        (out / "timing.json").write_text(json.dumps({"total": self.seconds}))
        error = (
            None
            if reward is not None
            else "ACP error -32603: Internal error: You've hit your session limit · "
            "resets 5pm (UTC)"
            if limited
            else "sandbox setup failed (scripted)"
        )
        category = "acp_error" if limited else "sandbox_setup"
        rewards = None if reward is None else {"reward": reward}
        scripted = cfg.agent in ("oracle", "nop")
        cost = None if scripted else self.usd
        if self.session_usd is not None and not scripted:
            log = out / "artifacts" / "claude-sessions" / "-app" / "s1.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text(session_log(self.session_usd) + "\n")
        (out / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task,
                    "rollout_name": name,
                    "agent": cfg.agent,
                    "model": cfg.model,
                    "rewards": rewards,
                    "error": error,
                    "error_category": category if error else None,
                    "n_tool_calls": 3,
                    "agent_result": {"cost_usd": cost},
                }
            )
        )
        (out / "trajectory" / "acp_trajectory.jsonl").write_text(
            json.dumps({"type": "tool_call", "title": "ls"}) + "\n"
        )
        if reward is not None:
            (out / "verifier" / "test-stdout.txt").write_text(f"GRADER-OUTPUT-{task}\n")
        return bf.RolloutResult(
            task_name=task,
            rollout_name=name,
            rewards=rewards,
            agent=cfg.agent,
            model=cfg.model,
            n_tool_calls=3,
            cost_usd=cost,
            error=error,
            error_category=category if error else None,
            rollout_dir=out,
        )


class FakeLimits:
    """The Messages API as the pool's probe sees it: per token, the share of the
    5-hour and 7-day windows used; a token in ``limited`` answers 429, rejected."""

    def __init__(self, util: dict[str, tuple[float, float]], limited=()):
        self.util, self.limited, self.probes = util, set(limited), []

    def __call__(self, request, timeout):
        token = request.get_header("Authorization").removeprefix("Bearer ")
        self.probes.append(token)
        h5, d7 = self.util.get(token, (0.0, 0.0))
        rejected = token in self.limited
        headers = {
            "anthropic-ratelimit-unified-5h-utilization": "1.0"
            if rejected
            else str(h5),
            "anthropic-ratelimit-unified-5h-reset": str(int(time.time()) + 3600),
            "anthropic-ratelimit-unified-7d-utilization": str(d7),
            "anthropic-ratelimit-unified-7d-reset": str(int(time.time()) + 86400),
            "anthropic-ratelimit-unified-status": "rejected" if rejected else "allowed",
            "anthropic-ratelimit-unified-5h-status": "rejected"
            if rejected
            else "allowed",
        }
        if token == "revoked":
            raise urllib.error.HTTPError(API, 401, "unauthorized", headers, None)
        if rejected:
            raise urllib.error.HTTPError(API, 429, "rate limited", headers, None)
        return FakeReply(200, headers)


@dataclass
class FakeReply:
    status: int
    headers: dict

    def close(self) -> None:
        pass


API = "https://api.anthropic.com/v1/messages"


def append_rule(text: str) -> Callable[[Path], dict]:
    """An edit: append a line to the first skill; returns the proposal."""

    def edit(surface: Path) -> dict:
        skill = sorted((surface / "skills").glob("*/SKILL.md"))[0]
        skill.write_text(skill.read_text() + f"\n{text}\n")
        return {
            "root_cause": "a skipped step",
            "change": f"add: {text[:40]}",
            "rationale": "general",
        }

    return edit


@dataclass
class FakeOptimizer:
    edits: list[Callable[[Path], dict | None]] = field(default_factory=list)
    analysis: dict | None = None
    probe: Callable[[Path], None] | None = None
    root: Path | None = None
    runs: list[dict] = field(default_factory=list)

    def install(self, monkeypatch) -> FakeOptimizer:
        monkeypatch.setattr(bf, "run", self.run)
        return self

    async def run(self, config) -> bf.RolloutResult:
        sandbox = (
            self.root or Path(config.jobs_dir).parent
        ) / f"sandbox-{len(self.runs)}"
        for (
            host,
            target,
        ) in config.uploads.items():  # nothing but the uploads exists inside
            shutil.copytree(host, sandbox / target.lstrip("/"))
        mode = (
            "analyze"
            if "analysis.json" in (config.task_path / "tests" / "test.sh").read_text()
            else "propose"
        )
        self.runs.append(
            {"mode": mode, "uploads": dict(config.uploads), "sandbox": sandbox}
        )
        if self.probe:
            self.probe(sandbox)
        app = sandbox / "app"
        output = "proposal.json" if mode == "propose" else "analysis.json"
        index = sum(r["mode"] == "propose" for r in self.runs) - 1
        data = (
            (self.edits[index](app / "surface") if index < len(self.edits) else None)
            if mode == "propose"
            else self.analysis
        )
        if data is not None:
            (app / output).write_text(json.dumps(data))
        name = f"{config.task_path.name}__{uuid.uuid4().hex[:8]}"
        out = Path(config.jobs_dir) / "job" / name
        (out / "verifier").mkdir(parents=True)
        if (app / output).exists():  # what the demo task's tests/test.sh does
            shutil.copyfile(app / output, out / "verifier" / output)
        shutil.copytree(app / "surface", out / "verifier" / "surface")
        ok = (app / output).exists() and subprocess.run(
            [
                sys.executable,
                str(config.task_path / "tests" / "validate.py"),
                str(app / output),
            ]
        ).returncode == 0
        (out / "result.json").write_text(
            json.dumps(
                {
                    "task_name": config.task_path.name,
                    "rollout_name": name,
                    "rewards": {"reward": float(ok)},
                    "n_tool_calls": 5,
                }
            )
        )
        return bf.RolloutResult(
            task_name=config.task_path.name,
            rollout_name=name,
            rewards={"reward": float(ok)},
            n_tool_calls=5,
            cost_usd=0.05,
            rollout_dir=out,
        )


def read_tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(errors="replace")
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
