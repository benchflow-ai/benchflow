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


# ``rollout.py`` was split into the ``benchflow.rollout`` package; these
# source-level invariants now span every module in it.
_ROLLOUT_PKG = Path("src/benchflow/rollout")


def _rollout_package_sources() -> list[Path]:
    return sorted(_ROLLOUT_PKG.glob("*.py"))


def test_rollout_source_contains_no_parallel_group_scheduler() -> None:
    """Guards the fix from PR #515 for issue #413: no parallel_group scheduler."""
    source = "\n".join(p.read_text() for p in _rollout_package_sources())

    assert "parallel_group" not in source
    assert "asyncio.gather(*(self._run_scene" not in source


def test_rollout_run_calls_scene_desugaring() -> None:
    """Guards the fix from PR #515 for issue #413: scenes compile before execute."""
    calls: set[str] = set()
    for source in _rollout_package_sources():
        tree = ast.parse(source.read_text())
        calls |= {
            getattr(node.func, "id", getattr(node.func, "attr", ""))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }

    assert "compile_scenes_to_steps" in calls
    assert "_run_steps" in calls
