"""A comparison's gap interval is the one the docs' own paired script gives.

``docs/examples/compare/paired.py`` gives the difference between two arms a
95% t interval over tasks. ``bf.compare``'s gap interval (the scoring rule's
gap over the paired tasks, its cluster-robust standard error, Student t for
T-1 degrees of freedom) must agree with it when every task has the same
number of trials, as here. The script is run as a user runs it, on the same
two folders.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import benchflow as bf
from tests import _run_records as rr

PAIRED = (
    Path(__file__).resolve().parents[1] / "docs" / "examples" / "compare" / "paired.py"
)
# Passes of 3 trials per task, arm A then arm B.
PASSES = {
    "t1": (0, 1),
    "t2": (3, 3),
    "t3": (3, 3),
    "t4": (1, 3),
    "t5": (2, 1),
    "t6": (0, 2),
}


def _arm(root: Path, side: int) -> Path:
    call = rr.anthropic_call(
        model="anthropic/claude-opus-5",
        served="claude-opus-5-20260801",
        route="anthropic/claude-opus-5",
        effort="high",
    )
    for k in range(3):
        job = root / f"trial-{k + 1:02d}"
        for i, (task, passes) in enumerate(PASSES.items()):
            rr.trial(job, task, k * 10 + i, reward=float(k < passes[side]),
                     model="anthropic/claude-opus-5", effort="high", calls=[call])  # fmt: skip
    return root


def test_the_gap_interval_is_the_paired_scripts_t_interval(tmp_path: Path) -> None:
    a, b = _arm(tmp_path / "a", 0), _arm(tmp_path / "b", 1)
    out = subprocess.run(
        [sys.executable, "-I", str(PAIRED), str(a), str(b)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    mean = float(re.search(r"over 6 tasks \(B - A\): ([+-][\d.]+)", out).group(1))
    low, high = (
        float(x)
        for x in re.search(
            r"95% t interval over tasks: ([+-][\d.]+) to ([+-][\d.]+)", out
        ).groups()
    )
    scoring = bf.compare(a, b).scoring or {}
    assert scoring["tasks"] == 6
    assert scoring["gap"] == pytest.approx(mean, abs=5e-4)
    assert scoring["gap_ci95"][0] == pytest.approx(low, abs=5e-4)
    assert scoring["gap_ci95"][1] == pytest.approx(high, abs=5e-4)
    assert scoring["gap_inside_ci"] is (low <= 0 <= high)
