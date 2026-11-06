"""CI installs the same uv BenchFlow pins for sandboxes, on every Python it lists.

``astral-sh/setup-uv`` with no ``version`` installs the newest uv on the day
(run 2406 floated to 0.13.1, whose default Python changed under the proxy
install), so every workflow names the action by commit and uv by version and
sha256, and those must be ``benchflow._uv_pin``'s. The unit job's Python matrix
must be the package's classifiers (3.13 was listed but never tested), and it
runs pytest under xdist. A matrix job reports no check under its own name,
so a separate job keeps reporting the `test` check main requires.
"""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchflow._uv_pin import UV_SHA256, UV_VERSION

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
ADMIN_SETUP = ROOT / ".github" / "integration" / "ADMIN_SETUP.md"

# setup-uv v10.3.0 (2026-10-09). A tag can move; a commit cannot.
SETUP_UV = "astral-sh/setup-uv"
SETUP_UV_COMMIT = "1c37ad07a6a961277cf70c0d37d6f313000f5884"

# Every CI job runs on GitHub's x86_64 Linux runners, the platform whose sum
# the workflows carry.
RUNNER_TARGET = "x86_64-unknown-linux-gnu"

# A shell word `uv` or `uvx` (not `uv.lock`, `--extra uv-x`, `setup-uv`).
_UV_COMMAND = re.compile(r"(?<![\w./-])uvx?(?![\w./-])")


def _workflows() -> dict[str, dict[str, Any]]:
    found = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert found, f"no workflows under {WORKFLOWS}"
    return {path.name: yaml.safe_load(path.read_text()) for path in found}


def _jobs() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (name, job_id, job)
        for name, workflow in _workflows().items()
        for job_id, job in (workflow.get("jobs") or {}).items()
    ]


def _is_setup_uv(step: dict[str, Any]) -> bool:
    return str(step.get("uses", "")).split("@", 1)[0] == SETUP_UV


def _setup_uv_steps() -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    return [
        (name, job_id, job, step)
        for name, job_id, job in _jobs()
        for step in job.get("steps") or []
        if _is_setup_uv(step)
    ]


def _classifier_pythons() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    prefix = "Programming Language :: Python :: "
    return [
        c.removeprefix(prefix)
        for c in project["classifiers"]
        if c.startswith(prefix) and re.fullmatch(r"3\.\d+", c.removeprefix(prefix))
    ]


def test_every_workflow_installs_the_uv_benchflow_pins() -> None:
    steps = _setup_uv_steps()
    # test.yml alone has four jobs that install uv; an empty list would mean
    # the parsing broke, not that the pins hold.
    assert len(steps) >= 4, steps
    for name, job_id, job, step in steps:
        where = f"{name} job {job_id}"
        ref = str(step["uses"]).split("@", 1)[1]
        assert ref == SETUP_UV_COMMIT, f"{where}: {SETUP_UV}@{ref}"
        with_ = step.get("with") or {}
        assert str(with_.get("version")) == UV_VERSION, (
            f"{where}: setup-uv version {with_.get('version')!r}, "
            f"benchflow._uv_pin.UV_VERSION is {UV_VERSION!r}"
        )
        assert with_.get("checksum") == UV_SHA256[RUNNER_TARGET], (
            f"{where}: setup-uv checksum is not the {RUNNER_TARGET} sum "
            "in benchflow._uv_pin.UV_SHA256"
        )
        runner = str(job.get("runs-on"))
        assert runner.startswith("ubuntu-") and "arm" not in runner, (
            f"{where}: runs on {runner!r}; the checksum is the {RUNNER_TARGET} one"
        )
        assert "version-file" not in with_, where


def test_no_job_runs_uv_before_installing_the_pinned_one() -> None:
    """A runner image's own uv, or one a step fetched, would float again."""
    checked = 0
    for name, job_id, job in _jobs():
        pinned = False
        for step in job.get("steps") or []:
            if _is_setup_uv(step):
                pinned = True
                continue
            script = str(step.get("run", ""))
            if not script:
                continue
            assert not re.search(r"pip3? install[^\n]*\buv\b", script), (
                f"{name} job {job_id} installs uv with pip"
            )
            assert "uv self update" not in script, f"{name} job {job_id}"
            if _UV_COMMAND.search(script):
                checked += 1
                assert pinned, (
                    f"{name} job {job_id} runs uv before {SETUP_UV}: {script[:80]!r}"
                )
    assert checked >= 10, checked


@pytest.mark.parametrize(
    ("workflow", "job_id"),
    [("test.yml", "unit"), ("fresh-resolve.yml", "fresh-resolve")],
)
def test_python_matrix_is_the_package_classifiers(workflow: str, job_id: str) -> None:
    job = _workflows()[workflow]["jobs"][job_id]
    pythons = [str(v) for v in job["strategy"]["matrix"]["python-version"]]
    assert pythons == _classifier_pythons() == ["3.12", "3.13"]
    assert job["strategy"].get("fail-fast") is False


def test_unit_job_runs_pytest_under_xdist() -> None:
    steps = _workflows()["test.yml"]["jobs"]["unit"]["steps"]
    (pytest_step,) = [s for s in steps if "pytest" in str(s.get("run", ""))]
    assert re.search(r"\s-n auto\b", pytest_step["run"]), pytest_step["run"]
    setup = next(s for s in steps if _is_setup_uv(s))
    assert setup["with"]["python-version"] == "${{ matrix.python-version }}"


def test_fresh_resolve_job_runs_the_script() -> None:
    job = _workflows()["fresh-resolve.yml"]["jobs"]["fresh-resolve"]
    runs = [s for s in job["steps"] if "fresh_resolve.sh" in str(s.get("run", ""))]
    assert len(runs) == 1
    assert runs[0]["env"]["FRESH_RESOLVE_PYTHON"] == "${{ matrix.python-version }}"
    assert (ROOT / "tools" / "ci" / "fresh_resolve.sh").is_file()


def _check_name(job_id: str, job: dict[str, Any]) -> str:
    """The check-run name GitHub reports for a job (one leg's, for a matrix)."""
    name = str(job.get("name", job_id))
    matrix = (job.get("strategy") or {}).get("matrix")
    if matrix and "${{" not in name:
        # GitHub appends the leg's values when the name does not use them.
        name += " (" + ", ".join(str(v[0]) for v in matrix.values()) + ")"
    return name


def test_the_required_test_check_is_still_reported() -> None:
    """ADMIN_SETUP.md makes `test` a required check on main.

    If no job reports a check named `test` (a matrix job reports
    `test (3.12)` and `test (3.13)` instead), every PR and merge-queue
    entry waits on it forever.
    """
    assert re.search(r"^- `test` \(L0", ADMIN_SETUP.read_text(), re.M)
    jobs = _workflows()["test.yml"]["jobs"]
    gates = [j for j, job in jobs.items() if _check_name(j, job) == "test"]
    assert len(gates) == 1, [_check_name(j, job) for j, job in jobs.items()]
    gate = jobs[gates[0]]
    assert "strategy" not in gate
    matrix_jobs = [j for j, job in jobs.items() if "strategy" in job]
    assert "unit" in matrix_jobs
    needs = gate["needs"]
    needs = [needs] if isinstance(needs, str) else list(needs)
    # Without always() a failed leg skips the gate, and a skipped required
    # check counts as passed.
    assert gate["if"] == "always()"
    assert set(matrix_jobs) <= set(needs), (matrix_jobs, needs)


@pytest.mark.parametrize(
    ("result", "passes"),
    [("success", True), ("failure", False), ("cancelled", False), ("skipped", False)],
)
def test_the_test_check_passes_only_when_every_unit_leg_passed(
    result: str, passes: bool
) -> None:
    (step,) = _workflows()["test.yml"]["jobs"]["test"]["steps"]
    env = [
        key
        for key, value in (step.get("env") or {}).items()
        if value == "${{ needs.unit.result }}"
    ]
    assert env, step
    assert "${{" not in step["run"], step["run"]
    done = subprocess.run(
        ["bash", "-eu", "-c", step["run"]],
        env={
            "PATH": os.environ["PATH"],
            **dict.fromkeys(env, result),
            **{
                key: "success"
                for key, value in step["env"].items()
                if value == "${{ needs.unit-complete.result }}"
            },
        },
        check=False,
    )
    assert (done.returncode == 0) is passes, (result, done.returncode)


@pytest.mark.parametrize("completion", ["failure", "cancelled", "skipped", ""])
def test_required_check_rejects_partial_matrix_rerun(completion: str) -> None:
    """Guards fix 298d4b44: matrix success alone is insufficient.

    This injects the reported GitHub partial-rerun state, not a live service
    reproduction: unit says success while the success-only barrier did not.
    """
    (step,) = _workflows()["test.yml"]["jobs"]["test"]["steps"]
    context = {
        "${{ needs.unit.result }}": "success",
        "${{ needs.unit-complete.result }}": completion,
    }
    env = {key: context[value] for key, value in step["env"].items()}
    done = subprocess.run(
        ["bash", "-eu", "-c", step["run"]],
        env={"PATH": os.environ["PATH"], **env},
        check=False,
    )
    assert done.returncode != 0, (completion, done.returncode)


def test_matrix_completion_barrier_requires_all_legs_and_is_not_the_required_check() -> (
    None
):
    """Guards fix 298d4b44 without passing a skipped required check."""
    jobs = _workflows()["test.yml"]["jobs"]
    assert "unit-complete" in jobs
    barrier = jobs["unit-complete"]
    assert barrier["needs"] == ["unit"]
    assert barrier.get("if", "success()") == "success()"
    assert "strategy" not in barrier
    assert not barrier.get("continue-on-error", False)
    assert "unit-complete" in jobs["test"]["needs"]
    assert jobs["test"]["if"] == "always()"
