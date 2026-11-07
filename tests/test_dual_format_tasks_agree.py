"""Every in-repo task folder with both task.md and task.toml describes one task.

Stale-audit judge default (claude-sonnet-5-5): the bundled
smoke task's task.md was a dump of every field, defaults included, so it
stated the old default judge (claude-sonnet-4-6). Once the default moved,
BenchFlow refused the task ("native task.md config and legacy task.toml
config differ") and `bench eval smoke` failed for every agent. The task.md
files are now `bench tasks migrate --overwrite` output, which writes only the
keys task.toml declares.
"""

from pathlib import Path

import pytest

from benchflow.task.config import TaskConfig
from benchflow.task.document import TaskDocument

REPO = Path(__file__).resolve().parents[1]
DUAL = sorted(
    p.parent
    for p in [*(REPO / "src").rglob("task.md"), *(REPO / "tests").rglob("task.md")]
    if (p.parent / "task.toml").is_file()
)


def test_the_bundled_smoke_task_is_one_of_them():
    assert REPO / "src/benchflow/demo_task" in DUAL


@pytest.mark.parametrize("task", DUAL, ids=lambda p: str(p.relative_to(REPO)))
def test_task_md_and_task_toml_agree(task: Path) -> None:
    document = TaskDocument.from_path(task / "task.md")
    legacy = TaskConfig.model_validate_toml((task / "task.toml").read_text())
    assert document.config.model_dump() == legacy.model_dump()
