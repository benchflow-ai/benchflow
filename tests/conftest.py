"""Test fixtures."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
REF_TASKS = REPO_ROOT / ".ref" / "harbor" / "examples" / "tasks"


@pytest.fixture
def hello_world_task_dir() -> Path:
    path = REF_TASKS / "hello-world"
    if not path.exists():
        pytest.skip("Harbor reference tasks not available")
    return path
