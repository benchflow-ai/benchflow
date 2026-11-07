"""Restoring base folders removes the need for ownership files."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import Evaluation
from tests.test_raw_abs_compat import LEGACY_ALIASES, _result, _tasks


@pytest.mark.parametrize("alias", LEGACY_ALIASES)
@pytest.mark.parametrize("failure", ["interrupt", "agent-error"])
@pytest.mark.parametrize(
    "backend",
    ["docker", "remote-docker", "daytona", "modal", "apple-container", "agentcore"],
)
def test_matrix_fresh_interrupt_resume_complete_repeat_and_read(
    tmp_path, monkeypatch, alias, failure, backend
):
    """Guard CONF2/CONF3: no ledger or final summary is needed to resume base names."""
    import benchflow.eval_plan as planning
    from benchflow.cli.eval_artifacts import matrix_solve_rates

    monkeypatch.setattr(planning, "sandbox_sdk_missing", lambda _: False)
    tasks = _tasks(tmp_path)
    jobs = tmp_path / "jobs"
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "models:\n"
        + "".join(
            f"  {json.dumps(a, ensure_ascii=False)}: dummy-model\n"
            for a in [alias, "other-arm"]
        )
    )
    first = []
    resumed = []
    stopped = False

    async def run(self):
        nonlocal stopped
        result = _result(self)
        if not stopped:
            first.append(self.job_dir.resolve())
            if len(first) == 2:
                stopped = True
                if failure == "interrupt":
                    raise KeyboardInterrupt
                raise RuntimeError("dummy agent failure")
        else:
            resumed.append(self.job_dir.resolve())
        return result

    monkeypatch.setattr(Evaluation, "run", run)
    args = [
        "eval",
        "run",
        "--tasks-dir",
        str(tasks),
        "--jobs-dir",
        str(jobs),
        "--agent",
        "oracle",
        "--matrix",
        str(matrix),
        "--trials",
        "2",
        "--sandbox",
        backend,
    ]
    initial = CliRunner().invoke(app, args)
    assert initial.exit_code != 0
    assert len(first) == 2
    assert all(path.parent.parent == (jobs / alias).resolve() for path in first)
    saved = {p: (p / "task-a__abc/result.json").read_bytes() for p in first}
    for _ in range(2):
        resumed.clear()
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, (result.output, result.exception)
        assert resumed[:2] == first
        assert matrix_solve_rates(jobs, [alias])[alias] is not None
        assert {p: (p / "task-a__abc/result.json").read_bytes() for p in first} == saved
        assert not (jobs / "matrix-ownership.json").is_file()
