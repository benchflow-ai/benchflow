"""``bench eval branch`` forks from a snapshot it already has.

With ``--checkpoints every-prompt --checkpoint-after-prompt 1``
the automatic checkpoint ``prompt:1`` and the fork snapshot were two
snapshots of the same state.
``--from-checkpoint`` started a fresh sandbox, replaced it with the kept
checkpoint, then snapshotted that state again. Now the fork reuses the
automatic checkpoint taken at the fork point, and a trial from a kept
checkpoint forks from that checkpoint (when it forks before any new prompt)
and starts its sandbox straight from it where the provider can
(``start_from_snapshot``, Daytona).
"""

from __future__ import annotations

import json

from benchflow.branch_run import load_checkpoint_source, run_branch_trial
from benchflow.checkpoints import parse_checkpoint_policy
from benchflow.sandbox.protocol import SandboxImage
from tests.test_branch_run import (
    ScriptedRollout,
    WorldSandbox,
    _kept_trial,
    _named,
    _plan,
    task,  # noqa: F401  (fixture)
)


def _fork(rollout):
    tree = json.loads((rollout._rollout_dir / "tree.json").read_text())
    [fork] = tree["forks"]
    return fork


async def test_the_automatic_checkpoint_is_the_fork_snapshot(tmp_path, task):  # noqa: F811
    plan = _plan(
        tmp_path,
        task_paths=[task],
        checkpoint_after=1,
        checkpoints=parse_checkpoint_policy("every-prompt"),
    )
    plan.validate()
    await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    rollout = ScriptedRollout.last
    rows = json.loads((rollout._rollout_dir / "checkpoints.json").read_text())
    kept = [r for r in rows["checkpoints"] if r["id"] == "prompt:1"]
    assert kept and kept[0]["status"] == "kept"
    fork = _fork(rollout)
    assert fork["snapshot"]["reused"] is True
    # prompt:1 only; later prompts may take their own checkpoints, but the
    # fork took none.
    assert "bf-snap-0" in WorldSandbox.images
    assert kept[0]["ref"] == "bf-snap-0"
    assert rollout._env.deleted == []
    # Children restored the checkpoint.
    assert rollout._env.restores[:2] == ["bf-snap-0", "bf-snap-0"]


async def test_from_checkpoint_forks_from_the_kept_snapshot(tmp_path, task):  # noqa: F811
    source = load_checkpoint_source(_kept_trial(tmp_path), None)
    plan = _plan(
        tmp_path,
        task_paths=[task],
        checkpoint_after=0,
        parent_mode="discard",
        source=source,
    )
    plan.validate()
    await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    rollout = ScriptedRollout.last
    assert WorldSandbox.images == {}  # no snapshot taken
    assert rollout._env.deleted == []  # the kept checkpoint is not the fork's
    assert _fork(rollout)["snapshot"]["reused"] is True
    assert [call[2] for call in _named(rollout.calls, "execute")] == [
        ["kept-checkpoint"],
        ["kept-checkpoint"],
    ]


class FastStartSandbox(WorldSandbox):
    def start_from_snapshot(self, image: SandboxImage) -> bool:
        self.rollout.calls.append(("start_from_snapshot", image.ref))
        self.rollout.world = ["kept-checkpoint"]
        return True


class FastStartRollout(ScriptedRollout):
    async def setup(self) -> None:
        await super().setup()
        self._env = FastStartSandbox(self)

    async def start(self) -> None:
        self.calls.append(("start", self._from_branch_snapshot))


async def test_from_checkpoint_starts_straight_from_the_snapshot(tmp_path, task):  # noqa: F811
    source = load_checkpoint_source(_kept_trial(tmp_path), None)
    plan = _plan(
        tmp_path,
        task_paths=[task],
        checkpoint_after=0,
        parent_mode="discard",
        source=source,
    )
    plan.validate()
    await run_branch_trial(plan, task, rollout_factory=FastStartRollout)
    rollout = ScriptedRollout.last
    names = [call[0] for call in rollout.calls]
    assert names[:4] == ["setup", "start_from_snapshot", "start", "install_agent"]
    # start() knew the sandbox comes from a snapshot (no uploads, no setup
    # commands); no restore into a fresh sandbox was needed.
    assert rollout.calls[2] == ("start", True)
    assert ("restore", "bf-snap-kept") not in rollout.calls[:4]


async def test_a_reused_snapshot_is_not_reported_as_the_forks_own(tmp_path, task):  # noqa: F811
    """The reused image belongs to the checkpoint: the outcome does not call
    it the fork's kept snapshot, and --from-checkpoint on this trial finds
    the checkpoint row, not the fork."""
    plan = _plan(
        tmp_path,
        task_paths=[task],
        checkpoint_after=1,
        checkpoints=parse_checkpoint_policy("every-prompt"),
    )
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    assert outcome.kept_snapshot is None
    rollout = ScriptedRollout.last
    (rollout._rollout_dir / "config.json").write_text('{"task_path": "task"}')
    source = load_checkpoint_source(rollout._rollout_dir, None)
    assert source.fork_id.startswith("prompt:")
