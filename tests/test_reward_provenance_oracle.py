"""The oracle's leftover processes are checked like an agent's (#1247).

In integrity off and audit a process the oracle's ``solve.sh`` leaves running
lives through grading, as an agent's does. One that runs as root can change
what test.sh writes while it runs, or plant files in ``/logs/verifier`` that
the clear keeps (root's). On 0.10.0 the oracle was never marked: such a trial
read ``verified``. Now a process born in the oracle's turn and alive as
grading starts makes the reward ``unverifiable`` (check ``oracle-processes``,
the list a root agent's turn gets), and an oracle that runs as the task's
``[agent].user`` also gets rule S's question for that user. The reward is
never changed: the oracle's reward is scored as written.

Mutant: rule S (and the process list) skipped for the oracle.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchflow.rollout import _reward_provenance as provenance
from benchflow.sandbox import _agent_alive

#: The detector's record after a turn whose solve.sh left ``sleep 900``.
ALIVE = {
    "flags": [],
    "changes": [
        {
            "where": "processes",
            "category": "turn-process-not-server",
            "after": {"pid": 41, "argv": "sleep 900", "uid": 0},
        }
    ],
    "after": {"taken": "post-turn"},
}
NONE_ALIVE = {"flags": [], "changes": [], "after": {"taken": "post-turn"}}
#: Off's record: no walk after the turn, the processes listed on their own.
OFF_ALIVE = {
    "flags": [],
    "processes_at_grading": ALIVE["changes"],
}
NOT_LISTED = {"flags": [], "changes": []}
RAN = {"ran": True, "why": None, "links": []}
STILL_ALIVE = {
    "user": "solver",
    "uid": 1001,
    "alive": True,
    "by": "ionice",
    "why": None,
}
NOT_ALIVE = {"user": "solver", "uid": 1001, "alive": False, "by": "ionice", "why": None}


def _row(found: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [c for c in found["checks"] if c["check"] == name]
    assert rows, f"no {name} check: {found['checks']}"
    return rows[0]


def _verdict(**given: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "agent": "oracle",
        "sandbox_user": None,
        "record": NONE_ALIVE,
        "outputs_closed": True,
        "outputs_open_why": None,
        "links_before": RAN,
        "links_after": RAN,
    }
    return provenance.verdict(**{**kwargs, **given})


@pytest.mark.parametrize("sandbox_user", [None, "agent"])
@pytest.mark.parametrize("record", [ALIVE, OFF_ALIVE], ids=["audit", "off"])
def test_a_root_solve_sh_that_leaves_a_process_reads_unverifiable(sandbox_user, record):
    """E2. ``--sandbox-user null`` and the default sandbox user alike: solve.sh
    runs as root either way, and the process it left is listed."""
    found = _verdict(sandbox_user=sandbox_user, record=record)
    assert found["verdict"] == "unverifiable", found
    assert [p["pid"] for p in found["processes"]] == [41], found
    assert any("solve.sh" in reason for reason in found["reasons"]), found
    row = _row(found, "oracle-processes")
    assert row["ran"] is True and row["passed"] is False, row


def test_an_oracle_that_leaves_nothing_is_verified():
    """Guard row: nothing alive, every check ran: ``verified``."""
    found = _verdict()
    assert found["verdict"] == "verified", found
    row = _row(found, "oracle-processes")
    assert row["ran"] is True and row["passed"] is True, row


def test_a_daytona_oracle_that_leaves_nothing_is_verified():
    """G1/D1, end to end: the walk's diff over a Daytona sandbox (BenchFlow's
    sessions and its walk born after the walk before the turn) leaves the
    oracle check nothing, so ``solve.sh: true`` reads ``verified``; a
    ``sleep 900`` solve.sh left still reads ``unverifiable``."""
    from benchflow.sandbox import _image_trees
    from tests.test_image_trees import (
        DAYTONA_AFTER,
        DAYTONA_BEFORE,
        DAYTONA_SHELLS,
        _manifest,
        _proc,
    )

    before = _manifest(processes=DAYTONA_BEFORE)
    for left, verdict in (
        ((), "verified"),
        ((_proc(2700, 1, 925, "sleep 900"),), "unverifiable"),
    ):
        after = _manifest(processes=(*DAYTONA_AFTER, *left), walk_pid=2761)
        flags, changes, _ = _image_trees.diff_processes(
            before, after, {"own_shells": DAYTONA_SHELLS}
        )
        record = {"flags": flags, "changes": changes, "after": {"taken": "post-turn"}}
        found = _verdict(record=record)
        assert found["verdict"] == verdict, found
        assert [p["pid"] for p in found["processes"]] == [2700] * len(left), found


def test_an_oracle_whose_processes_were_not_listed_is_unverifiable():
    """Fail closed, as for a root agent: no list of the processes alive at
    grading is a check that did not run."""
    found = _verdict(record=NOT_LISTED)
    assert found["verdict"] == "unverifiable", found
    row = _row(found, "oracle-processes")
    assert row["ran"] is False, row


def test_an_oracle_run_as_its_agent_user_gets_rule_s():
    """E2, an oracle whose solve.sh ran as the task's ``[agent].user``: rule
    S's answer for that user decides as it does for an agent's."""
    found = _verdict(sandbox_user="agent", alive_at_test=STILL_ALIVE)
    assert found["verdict"] == "unverifiable", found
    assert _agent_alive.ALIVE in found["reasons"], found
    assert found["alive_at_test"] == STILL_ALIVE, found
    assert (
        _verdict(sandbox_user="agent", alive_at_test=NOT_ALIVE)["verdict"] == "verified"
    )


def test_the_reward_of_an_unverifiable_oracle_is_kept():
    """The verdict never changes the oracle's reward: its attempt ends as
    test.sh left it, without ``provenance_incomplete``."""
    from benchflow.rollout._grading_outcome import Attempt, final_outcome

    found = _verdict(record=ALIVE)
    assert found["verdict"] == "unverifiable", found
    attempt = Attempt(
        rewards={"reward": 1.0},
        verifier_error=None,
        diagnostic=None,
        test_sh_reward=True,
        provenance=found,
        agent="oracle",
        sandbox_user="agent",
    )
    assert final_outcome(attempt) == ({"reward": 1.0}, None, None)
    assert "outcome" not in found, found


def test_the_oracle_parity_tally_does_not_count_an_unverifiable_oracle(tmp_path):
    """Gate 1 (the oracle-parity job and the full G1 run, tools/parity) reads
    each trial's provenance ``outcome``, never its verdict: an oracle whose
    leftover process makes it unverifiable has none, so its row is a MATCH
    where its reward matches Harbor's, not a blocking COUNT."""
    import json

    from tools.parity import run, tally

    found = _verdict(record=ALIVE)
    assert found["verdict"] == "unverifiable" and "outcome" not in found, found
    trial = tmp_path / "jobs" / "job" / "task__0"
    (trial / "integrity").mkdir(parents=True)
    (trial / "integrity" / "verifier-inputs.json").write_text(
        json.dumps({"provenance": found})
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "rewards": {"reward": 1.0},
                "warnings": ["Reward provenance unverifiable: " + found["reasons"][0]],
            }
        )
    )
    (tmp_path / "task").mkdir()
    row = run.read_trial("task", tmp_path / "task", tmp_path / "jobs", 0, 1.0)
    assert row.provenance_outcome == "" and row.end == "SCORED", row
    assert tally._counts(row) is None
