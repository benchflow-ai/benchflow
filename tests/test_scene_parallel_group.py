"""Tests for Scene.parallel_group concurrent scheduling (issue #417).

Verifies that Rollout._run_scenes() groups consecutive scenes sharing a
non-empty ``parallel_group`` and schedules them via ``asyncio.gather``
instead of running every scene sequentially.

The current single-ACP-transport runtime serializes ACP critical sections
across same-group scenes (see ``Rollout._scene_lock``), but the scenes are
still concurrently *scheduled* — that is the observable difference this
test pins.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchflow.rollout import Role, Rollout, RolloutConfig, Scene, Turn


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class FakeEnv:
    """Minimal sandbox stub — every exec is a no-op."""

    async def exec(self, cmd: str, **kwargs) -> FakeExecResult:
        return FakeExecResult()

    async def upload_dir(self, src: Path, dst: str) -> None:
        return None


def _make_trial(scenes: list[Scene]) -> Rollout:
    config = RolloutConfig(
        task_path=Path("tasks/fake"),
        scenes=scenes,
        environment="docker",
    )
    trial = Rollout(config)
    trial._env = FakeEnv()
    trial._resolved_prompts = ["Solve the task"]
    return trial


def _scene(name: str, *, group: str | None, role_name: str = "agent") -> Scene:
    return Scene(
        name=name,
        roles=[Role(role_name, "gemini", "flash")],
        turns=[Turn(role_name)],
        parallel_group=group,
    )


async def test_no_parallel_group_runs_sequentially() -> None:
    """Scenes with no parallel_group run one at a time, in declared order."""
    scenes = [
        _scene("a", group=None, role_name="r_a"),
        _scene("b", group=None, role_name="r_b"),
    ]
    trial = _make_trial(scenes)

    timeline: list[tuple[str, str]] = []

    async def fake_run_scene(scene: Scene) -> None:
        timeline.append(("start", scene.name))
        await asyncio.sleep(0.01)
        timeline.append(("end", scene.name))

    trial._run_scene = fake_run_scene  # type: ignore[method-assign]

    await trial._run_scenes(scenes)

    assert timeline == [
        ("start", "a"),
        ("end", "a"),
        ("start", "b"),
        ("end", "b"),
    ]


async def test_same_parallel_group_scheduled_concurrently() -> None:
    """Two scenes sharing parallel_group are dispatched via asyncio.gather.

    Observable: both scenes are 'started' before either has 'ended'.
    Sequential execution would force end-of-A before start-of-B.
    """
    scenes = [
        _scene("a", group="g1", role_name="r_a"),
        _scene("b", group="g1", role_name="r_b"),
    ]
    trial = _make_trial(scenes)

    timeline: list[tuple[str, str]] = []
    started_a = asyncio.Event()

    async def fake_run_scene(scene: Scene) -> None:
        timeline.append(("start", scene.name))
        if scene.name == "a":
            started_a.set()
            # Yield until B has had a chance to start too.
            await asyncio.sleep(0.01)
        else:
            # B waits for A to start, proving concurrent scheduling.
            await asyncio.wait_for(started_a.wait(), timeout=1.0)
        timeline.append(("end", scene.name))

    trial._run_scene = fake_run_scene  # type: ignore[method-assign]

    await trial._run_scenes(scenes)

    starts = [name for kind, name in timeline if kind == "start"]
    ends = [name for kind, name in timeline if kind == "end"]
    assert sorted(starts) == ["a", "b"]
    assert sorted(ends) == ["a", "b"]
    # Both scenes started before either ended → concurrent scheduling.
    assert timeline.index(("start", "b")) < timeline.index(("end", "a"))


async def test_distinct_parallel_groups_run_sequentially() -> None:
    """Different group keys flush — group 'g1' fully drains before 'g2' starts."""
    scenes = [
        _scene("a", group="g1", role_name="r_a"),
        _scene("b", group="g1", role_name="r_b"),
        _scene("c", group="g2", role_name="r_c"),
        _scene("d", group="g2", role_name="r_d"),
    ]
    trial = _make_trial(scenes)

    timeline: list[tuple[str, str]] = []

    async def fake_run_scene(scene: Scene) -> None:
        timeline.append(("start", scene.name))
        await asyncio.sleep(0.005)
        timeline.append(("end", scene.name))

    trial._run_scene = fake_run_scene  # type: ignore[method-assign]

    await trial._run_scenes(scenes)

    # g1 scenes (a, b) must both end before g2 scenes (c, d) start.
    last_g1_end = max(
        i for i, (kind, n) in enumerate(timeline) if kind == "end" and n in ("a", "b")
    )
    first_g2_start = min(
        i for i, (kind, n) in enumerate(timeline) if kind == "start" and n in ("c", "d")
    )
    assert last_g1_end < first_g2_start


async def test_parallel_group_rejects_shared_role_names() -> None:
    """Outbox files key on role name, so same-group scenes must use disjoint roles."""
    scenes = [
        _scene("a", group="g1", role_name="shared"),
        _scene("b", group="g1", role_name="shared"),
    ]
    trial = _make_trial(scenes)

    async def fake_run_scene(scene: Scene) -> None:
        # Should never be reached for the failing group.
        raise AssertionError("validation should have raised before scheduling")

    trial._run_scene = fake_run_scene  # type: ignore[method-assign]

    with pytest.raises(ValueError, match=r"parallel_group='g1'.*role 'shared'"):
        await trial._run_scenes(scenes)


async def test_scene_lock_serializes_critical_section() -> None:
    """Even concurrently scheduled, same-group scenes serialize on _scene_lock.

    This pins the current safety property: ``connect_as``/``execute``/
    ``disconnect`` mutate shared state, so the lock guarantees only one
    scene body is inside the critical section at a time.
    """
    scenes = [
        _scene("a", group="g1", role_name="r_a"),
        _scene("b", group="g1", role_name="r_b"),
    ]
    trial = _make_trial(scenes)
    active = 0
    max_active = 0

    real_run_scene = trial._run_scene

    async def instrumented(scene: Scene) -> None:
        # Call into the real _run_scene so the lock is exercised.
        nonlocal active, max_active
        # We hook the inside of the lock by patching execute().
        await real_run_scene(scene)

    async def fake_execute(prompts=None):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.005)
        finally:
            active -= 1
        return [], 0

    from unittest.mock import AsyncMock

    trial.connect_as = AsyncMock()  # type: ignore[method-assign]
    trial.disconnect = AsyncMock()  # type: ignore[method-assign]
    trial.execute = fake_execute  # type: ignore[method-assign,assignment]

    await trial._run_scenes(scenes)

    # Lock guarantees at most one scene inside the critical section at a time.
    assert max_active == 1
