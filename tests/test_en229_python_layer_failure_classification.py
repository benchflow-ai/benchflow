"""Python recipes survive unrelated startup failures."""

from __future__ import annotations

import pytest

from tests.test_en229_layer_failure_classification import SCENARIOS, exercise_fallback


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("guard", [False, True])
async def test_only_failed_python_builds_disable_later_layers(
    tmp_path, monkeypatch, scenario, reuse, guard
):
    """Reviews 2/3, including 75e24e27's missed late success."""
    await exercise_fallback(
        tmp_path,
        monkeypatch,
        dict(
            agent=None,
            model=None,
            stage_python=True,
            guard_daemon=guard,
            only_if_reused=True,
        ),
        scenario,
        reuse,
    )
