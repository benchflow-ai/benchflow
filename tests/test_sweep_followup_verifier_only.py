"""Guards FF-sw-cli-other-C4 against oracle-only preflight at f29e8174."""

import json
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.task_authoring import task_digest
from benchflow.eval_regrade import run_verifier_on_frozen_trial
from benchflow.review.evidence import EvidenceManifest
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout._verifier_recovery import recover_verifier
from benchflow.rollout_planes import DefaultRolloutPlanes
from benchflow.task import Task
from tests.test_check_run_plan import _task
from tests.test_verifier_recovery import trial as trial


class SandboxBoundary(BaseException):
    pass


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    # S1 e4db9aa7 admission is independent of whether a named value is needed
    # by this verifier-only child. Preserve the active-env assertions below.
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "PY8_ORACLE_DATA,PY8_VERIFIER_DATA")
    monkeypatch.delenv("PY8_ORACLE_DATA", raising=False)
    monkeypatch.delenv("PY8_VERIFIER_DATA", raising=False)

    def boundary(*args, **kwargs):
        raise SandboxBoundary

    monkeypatch.setattr(DefaultRolloutPlanes, "create_environment", boundary)
    monkeypatch.setattr(Rollout, "cleanup", AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["solution", "verifier"])
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
async def test_regrade_uses_only_active_env(tmp_path, missing, backend):
    """Guards f29e8174 through the real frozen-trial child's setup and gate."""
    variable = "PY8_ORACLE_DATA" if missing == "solution" else "PY8_VERIFIER_DATA"
    task = _task(tmp_path / "task", f'[{missing}.env]\nDATA="${{{variable}}}"\n')
    root = tmp_path / "trial"
    bundle = root / "evidence"
    bundle.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"agent": "nop", "environment": backend})
    )
    (bundle / "manifest.json").write_text(
        EvidenceManifest(
            workspace="/app", archive_sha256="0" * 64, entries=()
        ).model_dump_json()
    )
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    if missing == "solution":
        with pytest.raises(SandboxBoundary):
            await run_verifier_on_frozen_trial(root, task, attempt)
    else:
        with pytest.raises(ValueError, match=r"PY8_VERIFIER_DATA.*not found"):
            await run_verifier_on_frozen_trial(root, task, attempt)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["solution", "verifier"])
async def test_recovery_uses_only_active_env(trial, monkeypatch, missing):
    """Guards f29e8174 through real recovery setup without replaying a solver."""
    task_path = trial._config.task_path
    variable = "PY8_ORACLE_DATA" if missing == "solution" else "PY8_VERIFIER_DATA"
    with (task_path / "task.toml").open("a") as config:
        config.write(f'[{missing}.env]\nDATA="${{{variable}}}"\n')
    trial._task = Task(task_path)
    trial._config.task_digest = task_digest(task_path)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.prepare_workspace_upload",
        lambda *args: None,
    )
    # Keep setup real; stop at the sandbox factory, before any backend runs.
    trial._planes = DefaultRolloutPlanes()
    if missing == "solution":
        with pytest.raises(SandboxBoundary):
            await recover_verifier(trial)
    else:
        rewards, error = await recover_verifier(trial)
        assert rewards is None and "PY8_VERIFIER_DATA" in error


@pytest.mark.asyncio
async def test_oracle_still_requires_solution_env(tmp_path):
    """Guards f29e8174: normal oracle execution must still validate its inputs."""
    task = _task(tmp_path / "task", '[solution.env]\nDATA="${PY8_ORACLE_DATA}"\n')
    rollout = Rollout(
        RolloutConfig(task_path=task, agent="oracle", jobs_dir=tmp_path / "jobs")
    )
    with pytest.raises(ValueError, match=r"PY8_ORACLE_DATA.*not found"):
        await rollout.setup()
