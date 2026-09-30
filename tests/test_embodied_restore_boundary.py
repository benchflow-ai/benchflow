"""Branching, checkpoint restores and replay refuse embodied tasks that cannot restore their world.

One declaration, ``metadata.embodied`` (every EmbodiedTaskFormat package has it), says what software may do to an
embodied task's world. A container snapshot holds neither a simulator's process state nor a real robot's arm and
scene, so these operations are refused before anything is quiesced, checkpointed, restored or replayed: a
simulator only when it declares that its world is restorable, a real robot never.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.branch_run import run_branch_trial
from benchflow.checkpoint_retry import parse_retry_policy, run_checkpoint_retry
from benchflow.continue_run.run_folder import RunFolderError, load_run_folder
from benchflow.embodied.spec import (
    RestoreBoundary,
    RestoreRefused,
    SpecError,
    restore_boundary,
)
from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.task.task import Task
from tests.continue_run._helpers import completion, exchange, write_run_folder
from tests.test_branch_isolated import IMAGES, IsoRollout
from tests.test_branch_run import ScriptedRollout, _plan
from tests.test_checkpoint_retry import _finished_trial, _result
from tests.test_embodied import ToyFormat, _toy_task

EMBODIED = {"embodied": {"format": "toy", "base_task": "toy-reach", "seed": 0}}


def _task(metadata):
    return SimpleNamespace(config=SimpleNamespace(metadata=metadata))


def test_software_tasks_keep_restore_and_replay():
    for metadata in (None, {}, {"category": "coding", "tags": ["physical"]}):
        assert restore_boundary(metadata) == RestoreBoundary()


def test_embodied_tasks_restore_nothing_unless_they_declare_it():
    sim = restore_boundary(EMBODIED)
    assert (sim.embodied, sim.mode, sim.physical) == (True, "sim", False)
    assert not sim.world_restore and not sim.action_replay
    assert restore_boundary({"embodied": None}).embodied  # an empty `embodied:`
    declared = restore_boundary({"embodied": {"world_restore": True}})
    assert declared.world_restore and not declared.action_replay
    for mode in ("real", "hil-mock"):
        physical = restore_boundary({"embodied": {"mode": mode}})
        assert physical.physical and not physical.world_restore
        assert not physical.action_replay


@pytest.mark.parametrize(
    "metadata",
    [
        {"embodied": {"mode": "real", "world_restore": True}},
        {"embodied": {"mode": "hil-mock", "action_replay": True}},
        {"embodied": {"mode": "robot"}},
        {"embodied": {"world_restore": "yes"}},
        {"embodied": "sim"},
        # The SDK update's key is retired: it never unlocks a restore.
        {"embodiment": "physical"},
        {"embodiment": {"kind": "simulated"}, **EMBODIED},
    ],
)
def test_invalid_overclaiming_or_retired_declarations_raise(metadata):
    with pytest.raises(SpecError):
        restore_boundary(metadata)


def test_sidecar_packages_are_embodied_and_keep_the_task_declaration(tmp_path):
    fmt = ToyFormat()
    src = _toy_task(tmp_path / "src")
    boundary = restore_boundary(
        Task(fmt.materialize(src, tmp_path / "a")).config.metadata
    )
    assert boundary.embodied and not boundary.world_restore
    task_md = src / "task.md"
    task_md.write_text(
        task_md.read_text().replace(
            "agent:\n", "metadata:\n  embodied:\n    mode: real\nagent:\n", 1
        )
    )
    real = Task(fmt.materialize(src, tmp_path / "b")).config.metadata
    assert real["embodied"]["mode"] == "real" and real["embodied"]["seed"] == 0
    assert restore_boundary(real).physical
    task_md.write_text(
        task_md.read_text().replace(
            "mode: real\n", "mode: real\n    world_restore: true\n"
        )
    )
    with pytest.raises(SpecError, match="real embodiment cannot declare world_restore"):
        fmt.materialize(src, tmp_path / "c")


def _rollout(tmp_path, metadata):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._rollout_dir = tmp_path / "run"
    rollout._rollout_dir.mkdir()
    rollout._task = _task(metadata)
    rollout.disconnect = AsyncMock()
    rollout._environment = SimpleNamespace(
        snapshot=AsyncMock(return_value=StateSnapshot(id="s")),
        restore=AsyncMock(),
    )
    return rollout


@pytest.mark.parametrize(
    "metadata",
    [
        EMBODIED,
        {"embodied": {"mode": "real"}},
        {"embodied": {"action_replay": True}},
        {"embodiment": "physical"},
    ],
)
async def test_branch_refuses_before_quiesce_or_checkpoint(tmp_path, metadata):
    rollout = _rollout(tmp_path, metadata)
    runner = AsyncMock(return_value=1.0)
    cursor = rollout._cursor
    with pytest.raises((RestoreRefused, SpecError)):
        await rollout.branch(2, runner, snapshot_layers={"environment", "sandbox"})
    rollout.disconnect.assert_not_awaited()
    rollout._environment.snapshot.assert_not_awaited()
    rollout._environment.restore.assert_not_awaited()
    runner.assert_not_awaited()
    assert not (rollout._rollout_dir / "tree.json").exists()
    assert rollout._cursor is cursor and not cursor.children


async def test_real_robot_refusal_names_the_operator_reset(tmp_path):
    rollout = _rollout(tmp_path, {"embodied": {"mode": "real"}})
    with pytest.raises(RestoreRefused) as caught:
        await rollout.branch(2, AsyncMock(return_value=1.0))
    assert caught.value.boundary.physical
    assert "branch refused" in str(caught.value)
    assert "operator-qualified reset" in str(caught.value)


@pytest.mark.parametrize(
    "metadata", [{"category": "coding"}, {"embodied": {"world_restore": True}}]
)
async def test_software_task_and_restorable_simulator_still_branch(tmp_path, metadata):
    rollout = _rollout(tmp_path, metadata)
    assert await rollout.branch(2, AsyncMock(return_value=1.0)) == 1.0
    rollout._environment.snapshot.assert_awaited_once()


class _EmbodiedScripted(ScriptedRollout):
    async def setup(self) -> None:
        await super().setup()
        self._task = _task(EMBODIED)


async def test_branch_trial_refuses_before_the_sandbox_starts(tmp_path, caplog):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Do it.")
    plan = _plan(tmp_path, task_paths=[task], prompts=["Draft first.", "@instruction"])
    outcome = await run_branch_trial(plan, task, rollout_factory=_EmbodiedScripted)
    assert outcome.error is not None and "branch refused" in outcome.error
    calls = [call[0] for call in _EmbodiedScripted.last.calls]
    assert "start" not in calls and "execute" not in calls
    # An expected refusal is logged as a warning, without a traceback.
    [record] = [r for r in caplog.records if "branch refused" in r.getMessage()]
    assert record.levelname == "WARNING" and record.exc_info is None


async def test_branch_trial_takes_a_task_format_folder(tmp_path, monkeypatch):
    """bf.branch hands run_branch_trial the path it was given; a source folder
    in a task format runs as its package (its own task.md is not a native one)."""
    from benchflow.task import formats

    monkeypatch.setenv(formats.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(formats, "_registered", [ToyFormat()])
    monkeypatch.setattr(formats, "_entry_point_formats", [])
    src = _toy_task(tmp_path / "suite")
    plan = _plan(tmp_path, task_paths=[src], agent="oracle", checkpoint_after=0)
    outcome = await run_branch_trial(plan, src, rollout_factory=_EmbodiedScripted)
    assert outcome.task == "toy-reach"
    assert outcome.error is not None and "branch refused" in outcome.error
    assert _EmbodiedScripted.last._config.task_path != src


async def test_retry_from_checkpoint_is_refused_and_recorded(tmp_path):
    IMAGES.clear()
    IsoRollout.all = []
    root, _ = await _finished_trial(tmp_path)
    root._task = _task(EMBODIED)
    result = _result(0.0)
    await run_checkpoint_retry(
        root, result, parse_retry_policy("on-failure", prompt=None)
    )
    assert IsoRollout.all[1:] == []  # nothing was forked
    assert result.retry["status"] == "refused"
    assert "retry from checkpoint refused" in result.retry["error"]
    saved = json.loads((root._rollout_dir / "result.json").read_text())
    assert saved["retry"] == result.retry


def test_continue_refuses_to_replay_an_embodied_task(tmp_path):
    pkg = ToyFormat().materialize(_toy_task(tmp_path / "src"), tmp_path / "cache")
    folder = write_run_folder(
        tmp_path / "run", exchanges=[exchange(completion(content="a"))]
    )
    config = json.loads((folder / "config.json").read_text())
    config["task_path"] = str(pkg)
    (folder / "config.json").write_text(json.dumps(config))
    with pytest.raises(RunFolderError, match="benchflow continue refused"):
        load_run_folder(folder)


def test_continue_refuses_a_run_folder_with_an_episode_record(tmp_path):
    """config.json records only the task's name, so the task is rarely local;
    an embodied trial's episode record is evidence enough, and fails closed."""
    folder = write_run_folder(
        tmp_path / "run", exchanges=[exchange(completion(content="a"))]
    )
    episode = folder / "verifier" / "episode"
    episode.mkdir(parents=True)
    (episode / "episode.json").write_text(
        json.dumps({"embodiment": {"name": "arm", "kind": "arm", "mode": "real"}})
    )
    with pytest.raises(RunFolderError, match="operator-qualified reset"):
        load_run_folder(folder)
    (episode / "episode.json").write_text("{truncated")
    with pytest.raises(RunFolderError, match="does not declare action_replay"):
        load_run_folder(folder)


def test_continue_still_replays_a_software_task(tmp_path):
    folder = write_run_folder(
        tmp_path / "run", exchanges=[exchange(completion(content="a"))]
    )
    assert load_run_folder(folder).agent == "openhands"


def test_bench_eval_branch_selects_a_task_format_folder_and_refuses_it(
    tmp_path, monkeypatch
):
    """bench eval branch picks up a source folder a task format claims (as bench
    eval run does) and refuses the embodied package before any sandbox starts."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.task import formats

    monkeypatch.setenv(formats.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(formats, "_registered", [ToyFormat()])
    monkeypatch.setattr(formats, "_entry_point_formats", [])
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    src = _toy_task(tmp_path / "suite")
    args = ["eval", "branch", "--tasks-dir", str(src), "--agent", "oracle"]
    args += ["--child", "label=a", "--child", "label=b"]
    args += ["--jobs-dir", str(tmp_path / "jobs"), "--job-name", "j"]
    result = CliRunner().invoke(app, args, terminal_width=200)
    assert result.exit_code == 1, result.output
    assert "toy-reach" in result.output and "branch refused" in result.output
