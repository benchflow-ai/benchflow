"""Automatic checkpoints: retained sandbox snapshots after chosen prompts.

``bench eval branch --from-checkpoint`` used to only branch from a trial made with
``bench eval branch --retain-snapshots``; a normal ``bench eval run`` trial
never took a snapshot, so it could not be branched later. The opt-in
``--checkpoints every-prompt|prompt:N,M`` takes a sandbox snapshot after the
chosen prompts, keeps at most ``--checkpoint-keep`` of them per trial (the
oldest is deleted first), records them in ``checkpoints.json``, and never
fails the run: an unsupported sandbox or a failed snapshot is recorded.

Unit tests against fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.branch_run import BranchPlanError, load_checkpoint_source
from benchflow.checkpoints import (
    CheckpointPolicy,
    after_prompt,
    parse_checkpoint_policy,
)
from benchflow.sandbox.protocol import SandboxImage


class Sandbox:
    supports_snapshot = True

    def __init__(self, fail_on: int | None = None) -> None:
        self.taken: list[str] = []
        self.deleted: list[str] = []
        self.fail_on = fail_on

    async def snapshot(self, name=None) -> SandboxImage:
        if self.fail_on == len(self.taken) + 1:
            self.taken.append("failed")
            raise RuntimeError("provider refused")
        ref = f"bf-snap-{len(self.taken)}"
        self.taken.append(ref)
        return SandboxImage(provider="daytona", ref=ref)

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        self.deleted.append(image.ref)
        return True


def _rollout(tmp_path: Path, policy: CheckpointPolicy, sandbox=None):
    run = tmp_path / "task__abc"
    run.mkdir()
    return SimpleNamespace(
        _config=SimpleNamespace(checkpoints=policy),
        _env=sandbox if sandbox is not None else Sandbox(),
        _rollout_dir=run,
        _rollout_name="task__abc",
        _cursor=SimpleNamespace(id="n3"),
    )


def _record(rollout) -> dict:
    return json.loads((rollout._rollout_dir / "checkpoints.json").read_text())


@pytest.mark.parametrize(
    ("spec", "every", "after"),
    [
        ("every-prompt", True, frozenset()),
        ("prompt:2", False, frozenset({2})),
        ("prompt:1,3", False, frozenset({1, 3})),
    ],
)
def test_policies_parse(spec, every, after):
    policy = parse_checkpoint_policy(spec, keep=2)
    assert (policy.every, policy.after, policy.keep) == (every, after, 2)


@pytest.mark.parametrize("spec", ["", "always", "prompt:", "prompt:0", "prompt:a"])
def test_bad_policies_are_refused(spec):
    with pytest.raises(ValueError, match="--checkpoints"):
        parse_checkpoint_policy(spec, keep=3)


def test_keep_must_be_positive():
    with pytest.raises(ValueError, match="--checkpoint-keep"):
        parse_checkpoint_policy("every-prompt", keep=0)


async def test_every_prompt_keeps_the_newest(tmp_path):
    rollout = _rollout(tmp_path, parse_checkpoint_policy("every-prompt", keep=2))
    for prompt in (1, 2, 3):
        await after_prompt(rollout, prompt)
    record = _record(rollout)
    assert record["kind"] == "benchflow-checkpoints"
    assert record["policy"] == {"every": True, "after": [], "keep": 2}
    rows = record["checkpoints"]
    assert [(r["after_prompt"], r["ref"], r["status"]) for r in rows] == [
        (1, "bf-snap-0", "deleted"),
        (2, "bf-snap-1", "kept"),
        (3, "bf-snap-2", "kept"),
    ]
    assert rollout._env.deleted == ["bf-snap-0"]
    assert rows[1]["node_id"] == "n3"
    assert rows[1]["provider"] == "daytona"
    assert isinstance(rows[1]["seconds"], float)


async def test_only_the_chosen_prompts_are_checkpointed(tmp_path):
    rollout = _rollout(tmp_path, parse_checkpoint_policy("prompt:2", keep=3))
    for prompt in (1, 2, 3):
        await after_prompt(rollout, prompt)
    assert [r["after_prompt"] for r in _record(rollout)["checkpoints"]] == [2]


async def test_no_policy_is_a_no_op(tmp_path):
    rollout = _rollout(tmp_path, None)
    await after_prompt(rollout, 1)
    assert not (rollout._rollout_dir / "checkpoints.json").exists()


async def test_a_sandbox_without_snapshots_is_recorded_not_fatal(tmp_path):
    sandbox = SimpleNamespace(supports_snapshot=False)
    rollout = _rollout(
        tmp_path, parse_checkpoint_policy("every-prompt", keep=3), sandbox
    )
    await after_prompt(rollout, 1)
    await after_prompt(rollout, 2)
    rows = _record(rollout)["checkpoints"]
    assert [r["status"] for r in rows] == ["unsupported"]


async def test_a_failed_snapshot_is_recorded_not_fatal(tmp_path):
    rollout = _rollout(
        tmp_path, parse_checkpoint_policy("every-prompt", keep=3), Sandbox(fail_on=1)
    )
    await after_prompt(rollout, 1)
    await after_prompt(rollout, 2)
    rows = _record(rollout)["checkpoints"]
    assert [(r["status"], r["ref"]) for r in rows] == [
        ("failed", None),
        ("kept", "bf-snap-1"),
    ]
    assert rows[0]["error"] == {"type": "RuntimeError", "code": "snapshot_failed"}


# ── --from-checkpoint on a normal run ────────────────────────────────


async def _normal_trial(tmp_path: Path) -> Path:
    rollout = _rollout(tmp_path, parse_checkpoint_policy("every-prompt", keep=2))
    for prompt in (1, 2, 3):
        await after_prompt(rollout, prompt)
    (rollout._rollout_dir / "config.json").write_text(
        json.dumps({"task_path": "/somewhere/task"})
    )
    return rollout._rollout_dir


async def test_from_checkpoint_reads_a_normal_runs_checkpoints(tmp_path):
    trial = await _normal_trial(tmp_path)
    latest = load_checkpoint_source(trial, None)
    assert (latest.ref, latest.provider, latest.task_name) == (
        "bf-snap-2",
        "daytona",
        "task",
    )
    assert latest.fork_id == "prompt:3"
    chosen = load_checkpoint_source(trial, "prompt:2")
    assert chosen.ref == "bf-snap-1"


async def test_a_deleted_checkpoint_cannot_be_chosen(tmp_path):
    trial = await _normal_trial(tmp_path)
    with pytest.raises(BranchPlanError, match="prompt:1"):
        load_checkpoint_source(trial, "prompt:1")


async def test_run_steps_checkpoints_after_each_prompt(tmp_path, monkeypatch):
    from benchflow.rollout import _user_loop

    seen: list[int] = []

    async def fake_after_prompt(rollout, number):
        seen.append(number)

    monkeypatch.setattr(_user_loop, "after_prompt", fake_after_prompt)
    monkeypatch.setattr(
        _user_loop,
        "scene_step_role",
        lambda step: SimpleNamespace(
            name="agent",
            agent="a",
            model=None,
            reasoning_effort=None,
            timeout_sec=None,
            idle_timeout_sec=None,
            env={},
        ),
    )
    monkeypatch.setattr(_user_loop, "scene_step_prompt", lambda step: step)

    class R:
        async def _activate_step_skills(self, step): ...
        async def connect_as(self, role): ...
        async def disconnect(self): ...
        async def execute(self, prompts): ...

    steps = [SimpleNamespace(id=f"s{i}", data={}) for i in range(3)]
    await _user_loop._run_steps(R(), steps)
    assert seen == [1, 2, 3]


# ── bench eval run plumbing ──────────────────────────────────────────


def test_evaluation_config_carries_the_policy():
    from benchflow.evaluation import EvaluationConfig

    config = EvaluationConfig(checkpoints="prompt:1,2", checkpoint_keep=1)
    policy = config.checkpoint_policy()
    assert (policy.after, policy.keep) == (frozenset({1, 2}), 1)
    assert EvaluationConfig().checkpoint_policy() is None
    with pytest.raises(ValueError, match="--checkpoints"):
        EvaluationConfig(checkpoints="sometimes")


def test_worker_payload_round_trips_the_policy():
    from benchflow.eval_sharding import _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig

    config = EvaluationConfig(checkpoints="every-prompt", checkpoint_keep=2)
    shard = SimpleNamespace(concurrency=1, task_names=["t"])
    payload = _config_payload(config, shard=shard)
    restored = _evaluation_config(payload)
    assert restored.checkpoint_policy() == config.checkpoint_policy()


def test_eval_run_rejects_a_bad_policy_before_running(tmp_path):
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tmp_path),
            "--agent",
            "oracle",
            "--checkpoints",
            "sometimes",
        ],
    )
    assert result.exit_code != 0
    assert "--checkpoints" in result.output


def test_eval_plan_passes_the_policy_to_the_evaluation(tmp_path):
    from benchflow.eval_plan import EvalCreateRequest, build_eval_plan

    task = Path(__file__).parent / "examples" / "hello-world-task"
    plan = build_eval_plan(
        EvalCreateRequest(
            tasks_dir=task,
            agent="oracle",
            checkpoints="every-prompt",
            checkpoint_keep=4,
        )
    )
    config = plan.make_eval_config()
    assert config.checkpoint_policy() == parse_checkpoint_policy("every-prompt", keep=4)


async def test_checkpoint_time_is_itemized_in_timing(tmp_path):
    """Checkpoint snapshot time was not itemized in
    timing.json; ``checkpoint_snapshot`` now sums the time spent on them."""
    rollout = _rollout(tmp_path, parse_checkpoint_policy("every-prompt", keep=2))
    rollout._timing = {"agent_execution": 9.3}
    for prompt in (1, 2):
        await after_prompt(rollout, prompt)
    rows = _record(rollout)["checkpoints"]
    assert rollout._timing["checkpoint_snapshot"] == pytest.approx(
        sum(r["seconds"] for r in rows), abs=0.01
    )
    assert rollout._timing["agent_execution"] == 9.3
