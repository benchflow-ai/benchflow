"""Scene scheduler removal tests."""

from __future__ import annotations

import ast
from pathlib import Path

from benchflow.rollout import Rollout, RolloutConfig


def test_rollout_has_no_scene_scheduler_methods() -> None:
    """Guards the fix from PR #515 for issue #413: Rollout only executes Steps."""
    assert not hasattr(Rollout, "_run_scene")
    assert not hasattr(Rollout, "_run_scenes")
    assert not hasattr(Rollout, "_read_scene_outbox")

    cfg = RolloutConfig(task_path=Path("tasks/fake"))
    rollout = Rollout(cfg)
    assert not hasattr(rollout, "_scene_lock")


def test_rollout_source_contains_no_parallel_group_scheduler() -> None:
    """Guards the fix from PR #515 for issue #413: no parallel_group scheduler."""
    source = Path("src/benchflow/rollout.py").read_text()

    assert "parallel_group" not in source
    assert "asyncio.gather(*(self._run_scene" not in source


def test_rollout_run_calls_scene_desugaring() -> None:
    """Guards the fix from PR #515 for issue #413: scenes compile before execute."""
    tree = ast.parse(Path("src/benchflow/rollout.py").read_text())
    calls = {
        getattr(node.func, "id", getattr(node.func, "attr", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert "compile_scenes_to_steps" in calls
    assert "_run_steps" in calls
