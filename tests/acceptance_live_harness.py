"""Shared builders and rollout-layer fakes for acceptance-live tests.

Used by test_acceptance_live.py and test_acceptance_live_execution.py. The
fakes replace the seams acceptance_live imports from the rollout layer
(default_rollout_planes, _start_env_and_upload, _resolve_agent_cwd,
_verify_rollout, Rollout) so tests never touch a sandbox, model, or network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import benchflow.task.acceptance_live as acceptance_live
from benchflow.task.acceptance_live import run_live_acceptance_checks

AL = "acceptance_live"
REPORT_REL = "evidence/acceptance-live-report.json"
CALIBRATION_REL = "evidence/calibration-report.json"

LIVE_CALIBRATION_REPORT: dict[str, Any] = {
    "kind": "calibration-report",
    "cases": [
        {"name": "calnoop", "type": "no-op", "command": "true", "reward": 0.0},
        {"name": "calknownbad", "type": "known-bad", "command": "true", "reward": 0.2},
        {"name": "calpartial", "type": "partial", "command": "true", "reward": 0.5},
        {"name": "calreference", "type": "reference", "reward": 1.0},
    ],
}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_live_task(task_dir: Path, *, workdir: str | None = "/app") -> Path:
    frontmatter: dict[str, Any] = {
        "schema_version": "1.3",
        "task": {"name": "benchflow/acceptance-live-demo", "description": "demo"},
        "agent": {"timeout_sec": 60},
        "verifier": {"timeout_sec": 30},
    }
    if workdir is not None:
        frontmatter["environment"] = {"workdir": workdir}
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.md").write_text(
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False)
        + "---\n\n## prompt\n\nDo the work.\n"
    )
    solve = task_dir / "oracle" / "solve.sh"
    solve.parent.mkdir(exist_ok=True)
    solve.write_text("#!/bin/bash\nexit 0\n")
    solve.chmod(0o755)
    write_json(task_dir / CALIBRATION_REL, LIVE_CALIBRATION_REPORT)
    return task_dir


def green_case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "name": "greencase",
        "type": "verifier",
        "command": "echo ok",
        "reruns": 1,
        "expect": {"reward_min": 0.5},
    }
    case.update(overrides)
    return case


def oracle_case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "name": "oracleproof",
        "type": "oracle",
        "expect": {"reward_equals": 1.0},
    }
    case.update(overrides)
    return case


def full_live_evidence(*, calibration_flake_rate_max: float | None = None) -> dict:
    calibration: dict[str, Any] = {"from": "calibration.report", "reruns": 1}
    if calibration_flake_rate_max is not None:
        calibration["flake_rate_max"] = calibration_flake_rate_max
        calibration["reruns"] = 3
    return {
        AL: {
            "workspace": {"source": "current-worktree", "target": "/workspace"},
            "cases": [green_case(reruns=2), oracle_case()],
            "calibration": calibration,
            "leaderboard": {"required": True, "max_flake_rate": 0.0},
            "report": REPORT_REL,
        },
        "calibration": {
            "report": CALIBRATION_REL,
            "no_op_reward_max": 0.1,
            "known_bad_reward_max": 0.5,
            "partial_solution_range": [0.2, 0.8],
        },
    }


def case_evidence(
    case: dict[str, Any],
    *,
    report: str | None = REPORT_REL,
    leaderboard: dict[str, Any] | None = None,
    workspace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    acceptance: dict[str, Any] = {"cases": [case]}
    if workspace is not None:
        acceptance["workspace"] = workspace
    if report is not None:
        acceptance["report"] = report
    if leaderboard is not None:
        acceptance["leaderboard"] = leaderboard
    return {AL: acceptance}


def _case_run(rollout_name: str) -> tuple[str, int]:
    rest = rollout_name.removeprefix("acceptance-live-live-task-")
    name, run_index, _suffix = rest.rsplit("-", 2)
    return name, int(run_index)


class StubEnv:
    def __init__(self, harness: LiveHarness, rollout_name: str) -> None:
        self._harness = harness
        self.rollout_name = rollout_name

    async def exec(
        self, command: str, user: str | None = None, timeout_sec: float | None = None
    ) -> SimpleNamespace:
        name, run_index = _case_run(self.rollout_name)
        self._harness.execs.append(
            {
                "case": name,
                "run_index": run_index,
                "command": command,
                "user": user,
                "timeout_sec": timeout_sec,
            }
        )
        rc = self._harness.pick(self._harness.exec_rcs, name, run_index, 0)
        return SimpleNamespace(return_code=rc)

    async def upload_dir(self, source: Path, target: str) -> None:
        root = Path(source)
        self._harness.uploads.append(
            {
                "rollout_name": self.rollout_name,
                "target": target,
                "source": str(root),
                "files": sorted(
                    p.relative_to(root).as_posix()
                    for p in root.rglob("*")
                    if p.is_file()
                ),
            }
        )

    async def stop(self, delete: bool = False) -> None:
        self._harness.stops.append(
            {"rollout_name": self.rollout_name, "delete": delete}
        )


class LiveHarness:
    """Scripted fake rollout layer keyed by acceptance-live case name.

    Table values may be per-case scalars or per-run lists indexed by
    run_index. Defaults make the standard six-case happy path pass.
    """

    def __init__(self) -> None:
        self.rewards: dict[str, Any] = {
            "greencase": 1.0,
            "calnoop": 0.0,
            "calknownbad": 0.2,
            "calpartial": 0.5,
            "calreference": 1.0,
        }
        self.exec_rcs: dict[str, Any] = {}
        self.verifier_errors: dict[str, Any] = {}
        self.verify_exceptions: dict[str, Exception] = {}
        self.oracle_results: dict[str, Any] = {}
        self.oracle_create_error: Exception | None = None
        self.oracle_configs: list[Any] = []
        self.execs: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.stops: list[dict[str, Any]] = []
        self.starts: list[Path] = []
        self.verifies: list[dict[str, Any]] = []

    @staticmethod
    def pick(table: dict[str, Any], name: str, run_index: int, default: Any) -> Any:
        value = table.get(name, default)
        if isinstance(value, list):
            return value[run_index - 1]
        return value

    def oracle_result(self, name: str, run_index: int) -> Any:
        default = SimpleNamespace(
            trajectory=[{"type": "oracle", "return_code": 0}],
            error=None,
            verifier_error=None,
            rewards={"reward": 1.0},
        )
        return self.pick(self.oracle_results, name, run_index, default)


def install_live_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> LiveHarness:
    h = LiveHarness()

    def create_environment(
        sandbox_type: str,
        task: Any,
        task_dir: Path,
        rollout_name: str,
        rollout_paths: Any,
        preserve_agent_network: bool,
        environment_manifest: Any,
    ) -> StubEnv:
        return StubEnv(h, rollout_name)

    async def start_env(env: Any, task_path: Path, timing: dict, **kwargs: Any) -> None:
        h.starts.append(Path(task_path))

    async def agent_cwd(env: Any, task: Any) -> str:
        return "/resolved-agent-cwd"

    async def verify(
        env: Any,
        task: Any,
        rollout_paths: Any,
        timing: dict,
        planes: Any,
        sandbox_user: str | None = None,
        workspace: str | None = None,
    ) -> tuple[dict | None, str | None, None]:
        name, run_index = _case_run(env.rollout_name)
        h.verifies.append(
            {
                "case": name,
                "run_index": run_index,
                "sandbox_user": sandbox_user,
                "workspace": workspace,
            }
        )
        exc = h.verify_exceptions.get(name)
        if exc is not None:
            raise exc
        error = h.pick(h.verifier_errors, name, run_index, None)
        if error is not None:
            return None, error, None
        return {"reward": h.pick(h.rewards, name, run_index, 1.0)}, None, None

    class FakeRollout:
        def __init__(self, config: Any) -> None:
            self.config = config

        @classmethod
        async def create(cls, config: Any) -> FakeRollout:
            h.oracle_configs.append(config)
            if h.oracle_create_error is not None:
                raise h.oracle_create_error
            return cls(config)

        async def run(self) -> Any:
            for hook in self.config.pre_agent_hooks or []:
                await hook(StubEnv(h, self.config.rollout_name))
            name, run_index = _case_run(self.config.rollout_name)
            return h.oracle_result(name, run_index)

    monkeypatch.setattr(
        acceptance_live,
        "default_rollout_planes",
        lambda: SimpleNamespace(create_environment=create_environment),
    )
    monkeypatch.setattr(acceptance_live, "_start_env_and_upload", start_env)
    monkeypatch.setattr(acceptance_live, "_resolve_agent_cwd", agent_cwd)
    monkeypatch.setattr(acceptance_live, "_verify_rollout", verify)
    monkeypatch.setattr(acceptance_live, "Rollout", FakeRollout)

    staged_cwd = tmp_path / "staged-cwd"
    (staged_cwd / "nested").mkdir(parents=True)
    (staged_cwd / "workspace.txt").write_text("staged\n")
    (staged_cwd / "nested" / "inner.txt").write_text("inner\n")
    (staged_cwd / ".git").mkdir()
    (staged_cwd / ".git" / "config").write_text("ignored\n")
    (staged_cwd / "node_modules").mkdir()
    (staged_cwd / "node_modules" / "lib.js").write_text("ignored\n")
    monkeypatch.chdir(staged_cwd)
    return h


def run_live(
    task_dir: Path, evidence: dict, **kwargs: Any
) -> tuple[list[str], dict | None]:
    issues = run_live_acceptance_checks(
        task_dir, sandbox_type="docker", evidence=evidence, **kwargs
    )
    report_path = task_dir / REPORT_REL
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return issues, report
