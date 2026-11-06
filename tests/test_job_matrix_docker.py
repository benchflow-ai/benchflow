"""Guards PARITY-3 on real Docker: a YAML job config of
two agents (oracle and nop, so no model is called) and one dataset of two
tasks runs both agents, and each agent's trials have their own records.

At b1e738d29 only ``agents[0]`` ran: the nop agent's job never started.

The tasks are two Terminal-Bench 2 tasks when ``BENCHFLOW_TB2_TASKS_DIR``
names a Terminal-Bench 2 checkout (``fix-git`` and
``log-summary-date-ranges``, or the names in ``BENCHFLOW_TB2_TASK_NAMES``,
comma-separated), else two tasks of the same layout built here.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.exec_diet_fakes import docker_available

pytestmark = [
    pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon"),
    # A real run's protected verifier runtime (tests/conftest.py stubs it).
    pytest.mark.verifier_runtime,
]

#: ubuntu:24.04, by digest (build-vms README).
UBUNTU = (
    "ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"
)
TB2_TASKS = ("fix-git", "log-summary-date-ranges")


def _tb2_shaped(root: Path, name: str) -> None:
    task = root / name
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 120.0\n\n'
        "[agent]\ntimeout_sec = 120.0\n\n[environment]\nbuild_timeout_sec = 600.0\n"
    )
    (task / "instruction.md").write_text(f"Write {name} into /app/answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {UBUNTU}\nWORKDIR /app\nRUN mkdir -p /logs/verifier /logs/agent\n"
    )
    (task / "solution" / "solve.sh").write_text(
        f"#!/bin/bash\necho {name} > /app/answer.txt\n"
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\n"
        f'if [ "$(cat /app/answer.txt 2>/dev/null)" = {name} ]; then\n'
        "  echo 1 > /logs/verifier/reward.txt\n"
        "else\n  echo 0 > /logs/verifier/reward.txt\nfi\n"
    )


def _tasks(root: Path) -> list[str]:
    source = os.environ.get("BENCHFLOW_TB2_TASKS_DIR")
    names = (
        tuple(
            n.strip()
            for n in os.environ.get("BENCHFLOW_TB2_TASK_NAMES", "").split(",")
            if n.strip()
        )
        or TB2_TASKS
    )
    root.mkdir(parents=True)
    for name in names:
        if source:
            shutil.copytree(Path(source) / name, root / name)
        else:
            _tb2_shaped(root, name)
    return sorted(names)


def test_a_two_agent_job_config_runs_both_agents(tmp_path, monkeypatch) -> None:
    from benchflow.cli.main import app

    monkeypatch.chdir(tmp_path)
    if os.environ.get("BENCHFLOW_TB2_TASKS_DIR"):
        # Terminal-Bench 2 verifiers install pytest-json-ctrf; a real run
        # checks its release on PyPI, which the unit tests' offline table
        # (tests/conftest.py) leaves unproven.
        monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
    names = _tasks(tmp_path / "tb2")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "job_name": "parity3",
                "n_concurrent_trials": 2,
                "environment": {"type": "docker"},
                "agents": [{"name": "oracle"}, {"name": "nop"}],
                "datasets": [{"path": "tb2"}],
            }
        )
    )
    result = CliRunner().invoke(
        app,
        ["run", "--config", "config.json", "--jobs-dir", "jobs", "--quiet"],
        prog_name="bench",
    )
    assert result.exit_code == 0, result.output

    summary = json.loads((tmp_path / "jobs" / "matrix-summary.json").read_text())
    runs = {r["alias"]: r for r in summary["runs"]}
    assert sorted(runs) == ["nop__tb2", "oracle__tb2"], summary
    assert (runs["oracle__tb2"]["passed"], runs["oracle__tb2"]["total"]) == (2, 2)
    assert (runs["nop__tb2"]["passed"], runs["nop__tb2"]["total"]) == (0, 2)
    assert all(r["errored"] == 0 and r["verifier_errored"] == 0 for r in runs.values())

    # Each agent's trials have their own records, one per task.
    for alias, reward in (("oracle__tb2", 1.0), ("nop__tb2", 0.0)):
        job = tmp_path / "jobs" / alias / "trial-01" / "parity3"
        records = [
            json.loads(p.read_text())
            for p in job.glob("*/result.json")
            if "task_name" in json.loads(p.read_text())
        ]
        assert sorted(r["task_name"] for r in records) == names, (alias, records)
        assert {(r.get("rewards") or {}).get("reward") for r in records} == {reward}
        assert (job / "summary.json").is_file()


def test_a_two_by_two_by_two_matrix_runs_and_scores_every_cell(
    tmp_path, monkeypatch
) -> None:
    """Round 2 (reviews B-M1, B-L1): oracle and nop x two datasets of one task
    each x n_attempts 2 is 8 trial jobs, and every cell, the controls too,
    gets pass@k up to k = n_attempts."""
    from benchflow.cli.main import app

    monkeypatch.chdir(tmp_path)
    if os.environ.get("BENCHFLOW_TB2_TASKS_DIR"):
        monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
    names = _tasks(tmp_path / "pool")
    for index, name in enumerate(names[:2]):
        (tmp_path / f"ds-{index}").mkdir()
        shutil.move(tmp_path / "pool" / name, tmp_path / f"ds-{index}" / name)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "job_name": "p3x",
                "n_attempts": 2,
                "n_concurrent_trials": 2,
                "environment": {"type": "docker"},
                "agents": [{"name": "oracle"}, {"name": "nop"}],
                "datasets": [{"path": "ds-0"}, {"path": "ds-1"}],
            }
        )
    )
    result = CliRunner().invoke(
        app,
        ["run", "--config", "config.json", "--jobs-dir", "jobs", "--quiet"],
        prog_name="bench",
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "jobs" / "matrix-summary.json").read_text())
    cells = ["oracle__ds-0", "oracle__ds-1", "nop__ds-0", "nop__ds-1"]
    assert [c["alias"] for c in summary["job_cells"]] == cells
    assert summary["complete"] is True and summary["trials"] == 2
    assert sorted((r["alias"], r["trial"]) for r in summary["runs"]) == sorted(
        (cell, trial) for cell in cells for trial in (1, 2)
    )
    for run in summary["runs"]:
        oracle = run["alias"].startswith("oracle")
        assert (run["passed"], run["total"], run["errored"]) == (
            (1, 1, 0) if oracle else (0, 1, 0)
        ), run
    for cell in cells:
        rates = summary["solve_rates"][cell]
        assert rates is not None, (cell, summary["solve_rates"])
        assert rates["ks"] == [1, 2], rates
        want = 1.0 if cell.startswith("oracle") else 0.0
        assert {str(k): v for k, v in rates["pass_at_k"].items()} == {
            "1": want,
            "2": want,
        }, rates


def test_a_stopped_matrix_resumes_with_the_same_command(tmp_path, monkeypatch) -> None:
    """Round 3 (D949 resume case; reviews A M-A1 / B-M2): a 2x2x2 matrix stopped
    after its second trial job prints the whole-matrix line; running that same
    command finishes all 8 trial jobs and reruns none that finished."""
    import shlex

    from benchflow.cli._termination import RunStopped
    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation

    monkeypatch.chdir(tmp_path)
    if os.environ.get("BENCHFLOW_TB2_TASKS_DIR"):
        monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
    names = _tasks(tmp_path / "pool")
    for index, name in enumerate(names[:2]):
        (tmp_path / f"ds-{index}").mkdir()
        shutil.move(tmp_path / "pool" / name, tmp_path / f"ds-{index}" / name)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "job_name": "p3r",
                "n_attempts": 2,
                "n_concurrent_trials": 2,
                "environment": {"type": "docker"},
                "agents": [{"name": "oracle"}, {"name": "nop"}],
                "datasets": [{"path": "ds-0"}, {"path": "ds-1"}],
            }
        )
    )
    real_run = Evaluation.run
    started: list[str] = []

    async def stop_third(self):
        # The third trial job is stopped before it starts, as an orderly
        # SIGTERM between jobs; the first two ran for real.
        started.append(str(self.job_dir))
        if len(started) == 3:
            raise RunStopped("Terminated (SIGTERM)", 143)
        return await real_run(self)

    monkeypatch.setattr(Evaluation, "run", stop_third)
    line = ["run", "--config", "config.json", "--jobs-dir", "jobs", "--quiet"]
    stopped = CliRunner().invoke(app, line, prog_name="bench")
    assert stopped.exit_code == 143, stopped.output
    text = " ".join(stopped.output.split())
    marker = "trial jobs not finished; finished ones are kept): "
    assert "6 of 8" + " " + marker in text, text
    command = text.split(marker, 1)[1].split(" Matrix summary")[0].strip()
    # Outside the bench console script the line is rebuilt from the request.
    assert command == "bench run --config config.json --jobs-dir jobs", command
    summary_path = tmp_path / "jobs" / "matrix-summary.json"
    assert json.loads(summary_path.read_text())["complete"] is False
    finished = {
        p: p.stat().st_mtime_ns for p in (tmp_path / "jobs").rglob("*__*/result.json")
    }
    assert len(finished) == 2, sorted(finished)

    monkeypatch.setattr(Evaluation, "run", real_run)
    resumed = CliRunner().invoke(app, shlex.split(command)[1:], prog_name="bench")
    assert resumed.exit_code == 0, resumed.output
    summary = json.loads(summary_path.read_text())
    assert summary["complete"] is True and len(summary["runs"]) == 8
    records = list((tmp_path / "jobs").rglob("*__*/result.json"))
    assert len(records) == 8, sorted(map(str, records))
    # The two finished trial jobs were kept, not run again.
    assert all(p.stat().st_mtime_ns == t for p, t in finished.items())
    for run in summary["runs"]:
        oracle = run["alias"].startswith("oracle")
        assert (run["passed"], run["total"]) == ((1, 1) if oracle else (0, 1)), run
