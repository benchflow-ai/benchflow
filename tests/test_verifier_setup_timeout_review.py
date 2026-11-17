"""The verifier-setup budget reaches every phase and is validated (review).

Guards the review findings on c21bf7f1e: the budget lived in a context
variable the rollout set at setup, so a prestarted trial (set up in a task of
its own) ran its soft verify and turn-end kill at 180 s; regrade and verifier
recovery dropped the run's value, and then (e43c11dce) passed on the budget
the trial used, the task's or the default, as if the run had named it;
run-config values of 0, -5 or "900" ran at 180 silently; a step's own value
beat the run's flag; and 60 s or less left the plugin guard's input proof no
time.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from benchflow._verifier_setup import (
    MIN_VERIFIER_SETUP_TIMEOUT_SEC,
    apply_verifier_setup_timeout,
    recorded_verifier_setup_timeout,
    verifier_setup_timeout_sec,
)
from benchflow.sandbox import lockdown
from benchflow.task.config import TaskConfig
from tests.test_job_matrix import _bench, _config, _native
from tests.test_job_matrix import project as project

SRC = Path(__file__).resolve().parents[1] / "src" / "benchflow"


def _rollout(tmp_path: Path, **config):
    from benchflow.rollout import Rollout, RolloutConfig

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\nsetup_timeout_sec = 300\n'
    )
    (task / "instruction.md").write_text("Do it.")
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = {}
    planes.agent_launch.return_value = "oracle"
    planes.create_environment.return_value = SimpleNamespace()
    return Rollout(
        RolloutConfig.from_legacy(
            task_path=task,
            agent="oracle",
            jobs_dir=tmp_path / "jobs",
            planes=planes,
            **config,
        )
    )


# --- Rollout.setup wires it, and every phase keeps it -------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("flag", "used"), [(400, 400), (None, 300)])
async def test_setup_wires_the_flag_over_the_task_and_records_it(tmp_path, flag, used):
    rollout = _rollout(tmp_path, verifier_setup_timeout=flag)
    await asyncio.create_task(rollout.setup())
    assert rollout._task.config.verifier.setup_timeout_sec == used
    recorded = json.loads((rollout._rollout_dir / "config.json").read_text())
    assert recorded["verifier_setup_timeout"] == used


@pytest.mark.asyncio
async def test_a_prestarted_trials_soft_verify_keeps_the_runs_budget(
    tmp_path, monkeypatch
):
    """Set up in a task of its own, as begin_prestart does; the soft verify's
    hook purge then runs in the run's task with the run's 400 s."""
    rollout = _rollout(tmp_path, verifier_setup_timeout=400)
    await asyncio.create_task(rollout.setup())
    # The soft verify's hook purge is strict's.
    rollout._config.integrity = "strict"
    assert verifier_setup_timeout_sec() == 180  # this task never set it
    seen: list[int] = []

    async def purge(*args, **kwargs):
        seen.append(verifier_setup_timeout_sec())
        raise RuntimeError("stop after the purge")

    rollout._planes.cleanup_verifier_python_hooks = purge
    rollout._planes.clear_verifier_output_dir = AsyncMock()
    rollout._planes.ensure_legacy_app_dir = AsyncMock()
    monkeypatch.setattr(
        "benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()
    )
    rollout._env = SimpleNamespace()
    _, _, error = await rollout.soft_verify()
    assert seen == [400]
    assert error and "stop after the purge" in error
    assert verifier_setup_timeout_sec() == 180


@pytest.mark.asyncio
async def test_a_prestarted_trials_turn_end_kill_keeps_the_runs_budget(tmp_path):
    """disconnect()'s K1 quiesce (the turn-end kill) runs in the run's task,
    not the one that set the trial up: it has the run's 400 s, not 180."""
    rollout = _rollout(tmp_path, verifier_setup_timeout=400)
    await asyncio.create_task(rollout.setup())
    # The turn-end kill is strict's; off and audit kill after grading.
    rollout._config.integrity = "strict"
    assert verifier_setup_timeout_sec() == 180
    seen: list[int] = []

    async def quiesce(env, user):
        seen.append(verifier_setup_timeout_sec())
        raise RuntimeError("stop after the kill")

    rollout._planes.quiesce_agent = quiesce
    rollout._env = SimpleNamespace()
    rollout._connection_needs_quiescence = True
    with pytest.raises(RuntimeError, match="stop after the kill"):
        await rollout.disconnect()
    assert seen == [400]
    assert verifier_setup_timeout_sec() == 180


# --- regrade and recovery carry the run's own value ---------------------------


async def _recorded_config(tmp_path: Path, flag: int | None) -> dict:
    """config.json as Rollout.setup writes it, for a task that says 300 s."""
    tmp_path.mkdir(exist_ok=True)
    rollout = _rollout(tmp_path, verifier_setup_timeout=flag)
    await asyncio.create_task(rollout.setup())
    return json.loads((rollout._rollout_dir / "config.json").read_text())


def _task_dir(path: Path, seconds: int) -> Path:
    path.mkdir(parents=True)
    (path / "task.toml").write_text(
        f'version = "1.0"\n[verifier]\nsetup_timeout_sec = {seconds}\n'
    )
    (path / "instruction.md").write_text("Do it.")
    return path


def _budget(task_path: Path, run_seconds: int | None) -> int:
    """What a rollout of *task_path* settles with *run_seconds* as the run's."""
    from benchflow.task import Task

    task = Task(task_path)
    return contextvars.copy_context().run(
        apply_verifier_setup_timeout, task, run_seconds
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("flag", "recorded"), [(400, 400), (None, None)])
async def test_config_json_records_the_runs_own_value(tmp_path, flag, recorded):
    config = await _recorded_config(tmp_path, flag)
    assert config["verifier_setup_timeout"] == (flag or 300)
    assert config["verifier_setup_timeout_run"] == recorded
    assert recorded_verifier_setup_timeout(config) == recorded
    assert recorded_verifier_setup_timeout({}) is None
    assert recorded_verifier_setup_timeout({"verifier_setup_timeout": 300}) is None
    assert recorded_verifier_setup_timeout({"verifier_setup_timeout_run": "x"}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("flag", "used"), [(400, 400), (None, 600)])
async def test_a_regrade_passes_the_runs_value_not_the_one_it_used(
    tmp_path, monkeypatch, flag, used
):
    """The run named no value and its task said 300; the task now says 600.
    The regrade's rollout takes the task's 600, as a fresh run would; the
    run's own 400, when it named one, still wins."""
    import benchflow.rollout as rollout_module
    from benchflow import eval_regrade
    from benchflow.review.evidence import EvidenceManifest

    trial = tmp_path / "trial"
    (trial / "evidence").mkdir(parents=True)
    (trial / "evidence" / "manifest.json").write_text("{}")
    (trial / "config.json").write_text(
        json.dumps(await _recorded_config(tmp_path / "run", flag))
    )
    task_dir = _task_dir(tmp_path / "task-now", 600)
    seen: list = []

    class Stop(Exception):
        pass

    def capture(config):
        seen.append(config)
        raise Stop

    monkeypatch.setattr(
        EvidenceManifest,
        "model_validate_json",
        classmethod(lambda cls, raw: SimpleNamespace(workspace="/app")),
    )
    monkeypatch.setattr(eval_regrade, "pin_destination", lambda path: path)
    monkeypatch.setattr(eval_regrade, "ran_separate_verifier", lambda trial: False)
    monkeypatch.setattr(rollout_module, "Rollout", capture)
    with pytest.raises(Stop):
        await eval_regrade.run_verifier_on_frozen_trial(
            trial, task_dir, tmp_path / "attempt"
        )
    (config,) = seen
    assert config.verifier_setup_timeout == flag
    assert _budget(config.task_path, config.verifier_setup_timeout) == used


@pytest.mark.asyncio
@pytest.mark.parametrize(("flag", "used"), [(400, 400), (None, 900)])
async def test_a_recovery_passes_the_runs_value_not_the_one_it_used(
    tmp_path, monkeypatch, flag, used
):
    """Recovered from a trial whose run named no value: a step's own 900 s
    still holds over the task's 300, as it did in the run."""
    from benchflow.rollout import _verifier_recovery as recovery

    root = tmp_path / "trial"
    (root / "trajectory").mkdir(parents=True)
    (root / "trajectory" / "acp_trajectory.jsonl").write_text("")
    (root / "solver.json").write_text(
        json.dumps(
            {"agent": "claude-agent-acp", "task_digest": "d", "verifier_error": "x"}
        )
    )
    (root / "config.json").write_text(
        json.dumps(await _recorded_config(tmp_path / "run", flag))
    )
    seen: list = []

    class Stop(Exception):
        pass

    async def capture(trial, *, solver_agent):
        seen.append(trial._config)
        raise Stop

    monkeypatch.setattr(recovery, "needs_verifier_recovery", lambda error: True)
    monkeypatch.setattr(recovery, "recorded_recovery_ineligible_reason", lambda r: None)
    monkeypatch.setattr(recovery, "task_digest", lambda path: "d")
    monkeypatch.setattr(recovery, "recover_verifier", capture)
    with pytest.raises(Stop):
        await recovery.resume_verification(root, _task_dir(tmp_path / "task", 300))
    (config,) = seen
    assert config.verifier_setup_timeout == flag

    from benchflow.task.steps import step_task

    task = _steps_task(tmp_path / "steps-task")

    def step_budget() -> float | None:
        apply_verifier_setup_timeout(task, config.verifier_setup_timeout)
        return step_task(task, 0).config.verifier.setup_timeout_sec

    assert contextvars.copy_context().run(step_budget) == used


# --- values are validated -----------------------------------------------------


@pytest.mark.parametrize("bad", [0, -5, "900", 60, 900.5, math.inf, True])
def test_a_rollout_config_refuses_a_bad_value(tmp_path, bad):
    from benchflow.rollout import RolloutConfig

    with pytest.raises(ValueError, match="verifier_setup_timeout must be a whole"):
        RolloutConfig(task_path=tmp_path, verifier_setup_timeout=bad)


def test_a_whole_value_is_kept(tmp_path):
    from benchflow.rollout import RolloutConfig

    assert (
        RolloutConfig(
            task_path=tmp_path, verifier_setup_timeout=900
        ).verifier_setup_timeout
        == 900
    )
    assert (
        RolloutConfig(
            task_path=tmp_path, verifier_setup_timeout=900.0
        ).verifier_setup_timeout
        == 900
    )


@pytest.mark.parametrize("layout", ["native", "toml"])
@pytest.mark.parametrize("bad", ["0", "-5", '"900"', "60"])
def test_a_run_config_refuses_a_bad_value(project, layout, bad):
    path = (
        _native(project, f"verifier_setup_timeout: {bad}\n")
        if layout == "native"
        else _config(project, verifier_setup_timeout=json.loads(bad))
    )
    result = _bench("run", "--config", str(path))
    assert result.exit_code != 0
    assert "verifier_setup_timeout must be a whole number" in " ".join(
        result.output.split()
    )


def test_the_flag_refuses_60_or_less(project):
    result = _bench("run", "ds-a", "-a", "oracle", "--verifier-setup-timeout", "60")
    assert result.exit_code == 2


@pytest.mark.parametrize("bad", [60, 30])
def test_a_task_value_of_60_or_less_is_refused(bad):
    with pytest.raises(ValidationError):
        TaskConfig.model_validate({"verifier": {"setup_timeout_sec": bad}})


def test_the_least_value_leaves_the_input_proof_time():
    """61 s, the least budget, leaves the input proof at least a second."""
    assert MIN_VERIFIER_SETUP_TIMEOUT_SEC - lockdown._INPUT_PROOF_RESERVE_SEC >= 1


# --- a step's own value does not beat the run's -------------------------------


def _steps_task(tmp_path: Path) -> SimpleNamespace:
    config = TaskConfig.model_validate(
        {"steps": [{"name": "one", "verifier": {"setup_timeout_sec": 900}}]}
    )
    (tmp_path / "steps" / "one").mkdir(parents=True)
    (tmp_path / "steps" / "one" / "instruction.md").write_text("Step one.")
    return SimpleNamespace(config=config, task_dir=tmp_path, paths=None)


@pytest.mark.parametrize(("flag", "used"), [(400, 400), (None, 900)])
def test_the_runs_flag_wins_over_a_steps_own(tmp_path, flag, used):
    from benchflow.task.steps import step_task

    task = _steps_task(tmp_path)

    def run() -> float | None:
        apply_verifier_setup_timeout(task, flag)
        return step_task(task, 0).config.verifier.setup_timeout_sec

    assert contextvars.copy_context().run(run) == used


# --- the budget is not part of a verifier image's key ------------------------


@pytest.mark.parametrize("declared", ["", "setup_timeout_sec = 900\n"])
def test_the_verifier_layer_key_leaves_the_budget_out(tmp_path, monkeypatch, declared):
    """verifier_layers hashed the whole verifier config, so the new field
    changed every task's key from rc3's, and a task's value or the run's flag
    a reused image's."""
    from benchflow.sandbox import verifier_layers
    from benchflow.task import Task
    from benchflow.task.config import VerifierConfig
    from tests.test_verifier_layers import _task

    monkeypatch.setenv("BENCHFLOW_VERIFIER_LAYER", "1")
    seen: list[dict] = []
    digest = verifier_layers.verifier_inputs_digest

    def capture(tests_dir, verifier):
        seen.append(verifier)
        return digest(tests_dir, verifier)

    monkeypatch.setattr(verifier_layers, "verifier_inputs_digest", capture)

    def inputs(task: Task) -> str:
        recipe, record = verifier_layers.plan_verifier_layer(task, separate=True)
        assert recipe is not None, record
        return recipe.inputs

    plain = Task(_task(tmp_path / "plain"))
    task = Task(_task(tmp_path / "declared", verifier=declared))
    keys = [inputs(plain), inputs(task)]
    contextvars.copy_context().run(apply_verifier_setup_timeout, task, 400)
    keys.append(inputs(task))
    assert keys[1:] == keys[:1] * 2
    assert all(set(v) == set(VerifierConfig.model_fields) - {"setup_timeout_sec"}
               for v in seen)  # fmt: skip
