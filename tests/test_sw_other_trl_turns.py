"""Guards FF-sw-other against splitting persistent turns at 363d815b."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.integrations.trl.spec import _rows_for_package


@pytest.mark.parametrize("count", [2, 3, 10])
def test_trl_refuses_multiturn_package(count):
    package = SimpleNamespace(
        task_dir=Path("dummy-task"),
        view=SimpleNamespace(
            config=SimpleNamespace(task=SimpleNamespace(name="dummy")),
            entrypoint="task.md",
        ),
        prompt_plan=SimpleNamespace(
            turns=[SimpleNamespace(prompt=f"turn {i}") for i in range(count)]
        ),
    )
    with pytest.raises(ValueError, match=r"multi.turn|single.turn"):
        _rows_for_package(package)
