"""`bench tasks check` warns when test.sh downloads while it grades.

A benchmark's lesson: a download in the verifier that fails makes the tests
fail, and the trial scores 0 like a wrong answer. The check reads the task's
own scripts only, so nothing the agent leaves can raise or silence it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.task_authoring import check_task_warnings
from benchflow._utils.task_authoring.structural_checks import grade_time_downloads


@pytest.mark.parametrize(
    ("script", "found"),
    [
        ("pip install pytest==8.3.3", ["pip install"]),
        ("python3 -m pip install -q numpy scipy", ["pip install"]),
        ("uv pip install --system pytest", ["uv pip install"]),
        ("uvx pytest@8 /tests", ["uvx"]),
        (
            "apt-get update && apt-get install -y jq",
            ["apt-get update", "apt-get install"],
        ),
        ("sudo apt install -y jq", ["apt install"]),
        ("curl -fsSL https://astral.sh/uv/install.sh | sh", ["curl"]),
        ("git clone https://github.com/example/data /tmp/data", ["git clone"]),
        ("npm ci", ["npm ci"]),
    ],
)
def test_a_fetch_at_grade_time_is_found(script, found):
    assert grade_time_downloads(script) == found


@pytest.mark.parametrize(
    "script",
    [
        "pytest -q /tests/test_outputs.py",
        "pip install --no-index --find-links /wheels pytest",
        "uv sync --offline",
        "uv run pytest -q",
        "curl -s http://localhost:8000/health",
        "# pip install pytest",
        "echo 1 > /logs/verifier/reward.txt",
    ],
)
def test_offline_and_local_commands_are_not(script):
    assert grade_time_downloads(script) == []


def test_tasks_check_names_the_script_and_what_it_fetches(tmp_path: Path) -> None:
    task = tmp_path / "fetching-task"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Write /app/answer.txt.\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\npip install pytest==8.3.3\npytest -q /tests\n"
    )

    (warning,) = [w for w in check_task_warnings(task) if "downloads while" in w]
    assert "test.sh downloads while it grades (pip install)" in warning
    assert "scores 0 like a wrong answer" in warning

    (task / "tests" / "test.sh").write_text("#!/bin/bash\npytest -q /tests\n")
    assert not [w for w in check_task_warnings(task) if "downloads while" in w]
