"""``python -m benchflow.robotics`` reports operator mistakes without tracebacks.

Only the commands that read or score saved trials are exercised; nothing here
starts a bridge, a camera or an agent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _robotics(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "benchflow.robotics", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _smoke_trial(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"kind": "read_only_smoke", "status": "smoke_passed"})
    )
    return path


_SCORE = [
    "--placement",
    "tape=untied",
    "--reviewer",
    "r",
    "--interventions",
    "0",
    "--cups-upright",
    "yes",
    "--evidence",
    "side.mp4",
]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["index", "{missing}"], "manifest.json is missing"),
        (["score", "{smoke}", *_SCORE], "Only finished physical trials can be scored"),
    ],
)
def test_operator_mistakes_are_one_line_errors(tmp_path, argv, message):
    """Regression test: these printed tracebacks."""
    smoke = _smoke_trial(tmp_path / "smoke")
    args = [a.format(missing=str(tmp_path / "missing"), smoke=str(smoke)) for a in argv]
    result = _robotics(*args)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert message in result.stderr
    assert result.stderr.startswith("python -m benchflow.robotics: error:")


def test_help_names_the_module_invocation():
    result = _robotics("--help")
    assert result.returncode == 0
    assert result.stdout.startswith("usage: python -m benchflow.robotics")


def _finished_trial(path: Path, expected: dict) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "trial_id": path.name,
                "kind": "physical_trial",
                "task": path.name,
                "status": "awaiting_assessment",
                "expected": expected,
                "footage_complete": True,
                "started_utc_epoch": 1_767_225_600.0,
                "finished_utc_epoch": 1_767_225_660.0,
            }
        )
    )
    return path


_SORT = {"blue": "left", "green": "right", "yellow": "right"}
_REVIEW = ["--reviewer", "r", "--interventions", "0", "--evidence", "side.mp4"]


def test_cups_upright_is_optional_for_a_task_without_cups(tmp_path):
    """Regression test: ``score`` demanded
    ``--cups-upright`` for untie-knot, whose expected block has no cups.
    """
    trial = _finished_trial(tmp_path / "untie-knot", {"tape": "untied"})
    result = _robotics("score", str(trial), "--placement", "tape=untied", *_REVIEW)
    assert result.returncode == 0, result.stderr
    assessment = json.loads(result.stdout)
    assert assessment["cups_upright"] is None
    assert assessment["task_success"] is True
    assert json.loads((trial / "assessment.json").read_text()) == assessment


def test_cups_upright_is_still_required_when_blocks_go_in_cups(tmp_path):
    """Regression test: making ``--cups-upright``
    optional must not drop it for the sort tasks, whose blocks go in the left
    and right cups and whose instruction says both cups must stay upright.
    A missing answer is a one-line error, not a usage dump.
    """
    trial = _finished_trial(tmp_path / "sort-blue-left", _SORT)
    placements = [f"--placement={block}={cup}" for block, cup in _SORT.items()]
    result = _robotics("score", str(trial), *placements, *_REVIEW)
    assert result.returncode == 1
    assert result.stderr.startswith("python -m benchflow.robotics: error:")
    assert "--cups-upright" in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert not (trial / "assessment.json").exists()

    result = _robotics(
        "score", str(trial), *placements, "--cups-upright", "no", *_REVIEW
    )
    assert result.returncode == 0, result.stderr
    assessment = json.loads(result.stdout)
    assert assessment["cups_upright"] is False
    assert assessment["object_accuracy"] == 1.0
    assert assessment["task_success"] is False
    assert assessment["autonomous_success"] is False


def test_an_explicit_cups_answer_counts_for_any_task(tmp_path):
    """Regression test: an optional ``--cups-upright
    no`` given by the reviewer is still recorded and still fails the task.
    """
    trial = _finished_trial(tmp_path / "untie-knot", {"tape": "untied"})
    result = _robotics(
        "score",
        str(trial),
        "--placement",
        "tape=untied",
        "--cups-upright",
        "no",
        *_REVIEW,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["task_success"] is False


def test_cups_upright_stays_required_for_pick_yellow(tmp_path):
    """Keeps the conservative reading of ``--cups-upright``.

    pick-yellow runs in the cup scene and says "Do not move the other
    objects"; its placements cover only the yellow block, so the cups answer
    is the only record of whether the cups were disturbed. Only a task outside
    the cup scene (untie-knot) may omit it.
    """
    trial = _finished_trial(tmp_path / "pick-yellow", {"yellow": "lifted_5cm_for_3s"})
    result = _robotics(
        "score", str(trial), "--placement", "yellow=lifted_5cm_for_3s", *_REVIEW
    )
    assert result.returncode == 1
    assert "--cups-upright" in result.stderr
    assert not (trial / "assessment.json").exists()
