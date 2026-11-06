"""A discarded parent is unscored by design, and the records say so.

Branch parents unscored on purpose (``--parent discard``) were counted as
attempted-and-unscored like a failure, which lowered the attempted pass
rate. ``result.json`` ``branches.parent`` is now
``discarded`` or ``kept`` (a nested fork's parent is a child, not the
trial), and the branch view's trial block carries ``parent`` and
``unscored_by_design`` so a loader can leave such trials out of the
denominator.
"""

from __future__ import annotations

import json

from benchflow.branch_lineage import branch_summary
from benchflow.branch_view import load_branch_view
from tests.test_branch_cost import _fork_record


def _discarding(fork):
    fork["parent_restore"] = "skipped"
    return fork


def test_summary_marks_a_discarded_parent():
    outer = _discarding(_fork_record("f1", "task__x", 10.0, [("n1", 5.0)]))
    assert branch_summary([outer])["parent"] == "discarded"
    kept = _fork_record("f1", "task__x", 10.0, [("n1", 5.0)])
    kept["parent_restore"] = "restored"
    assert branch_summary([kept])["parent"] == "kept"
    # A nested fork that skips its parent restore discards a child, not the
    # trial's parent.
    nested = _discarding(_fork_record("f2", "n1", 3.0, [("n5", 1.0)]))
    assert branch_summary([kept, nested])["parent"] == "kept"


def test_view_says_unscored_by_design(tmp_path):
    trial = tmp_path / "job" / "task__x"
    trial.mkdir(parents=True)
    fork = _discarding(_fork_record("f" * 32, "task__x", 10.0, [("n1", 5.0)]))
    fork.update(parent_node="n0", status="completed", value=1.0)
    (trial / "tree.json").write_text(
        json.dumps(
            {"kind": "benchflow-branch-tree", "schema_version": 1, "forks": [fork]}
        )
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "rewards": None,
                "branches": branch_summary([fork]),
            }
        )
    )
    view = load_branch_view(trial)
    assert view["trial"]["parent"] == "discarded"
    assert view["trial"]["unscored_by_design"] is True
    # An older result.json without the marker: derived from the forks.
    result = json.loads((trial / "result.json").read_text())
    del result["branches"]["parent"]
    (trial / "result.json").write_text(json.dumps(result))
    assert load_branch_view(trial)["trial"]["parent"] == "discarded"
