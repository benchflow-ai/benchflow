"""SFT conversion of a branch run points at --format branch-tree.

``bench train convert <branch job> --format
prime-sft`` used to fail with a raw ``No such file or directory: …/trajectory/
llm_trajectory.jsonl``. Runs that branch have no LLM-proxy trajectory; the
error now says so and names the format that works.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from tests.test_export_branch_tree_v2 import PARENT, _say, _user, _write_trial


@pytest.mark.parametrize("fmt", ["prime-sft", "trl-sft"])
def test_sft_on_a_branch_run_suggests_branch_tree(tmp_path, fmt):
    _write_trial(
        tmp_path / "job" / "task__a",
        parent_events=PARENT,
        children=[("n3", "a", 1.0, [_user("Go."), _say("Yes.")])],
    )
    result = CliRunner().invoke(
        app,
        [
            "train",
            "convert",
            str(tmp_path / "job"),
            "--format",
            fmt,
            "--out",
            str(tmp_path / "o.jsonl"),
        ],
    )
    assert result.exit_code == 1
    assert "--format branch-tree" in result.output
    assert "Traceback" not in result.output
