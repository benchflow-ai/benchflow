"""The reward's provenance verdict, the links walk's resolution, and the run-start line (the lead's 20:31Z and 22:25Z rulings)."""

from __future__ import annotations

import itertools
import logging
from typing import Any

import pytest

from benchflow.rollout import _reward_provenance as provenance
from benchflow.sandbox import _verifier_links as links

ALIVE = {
    "flags": [],
    "changes": [
        {
            "where": "processes",
            "category": "turn-process-not-server",
            "after": {"pid": 41, "argv": "sleep 99", "uid": 0},
        }
    ],
    "after": {"taken": "post-turn"},
}
NONE_ALIVE = {"flags": [], "changes": [], "after": {"taken": "post-turn"}}
RAN = {"ran": True, "why": None, "links": []}
#: Rule S's records (sandbox/_agent_alive.py).
NOT_ALIVE = {"user": "agent", "uid": 1000, "alive": False, "by": "ionice", "why": None}
STILL_ALIVE = {"user": "agent", "uid": 1000, "alive": True, "by": "ionice", "why": None}


def _verdict(**given):
    kwargs: dict[str, Any] = {
        "agent": "claude-agent-acp",
        "sandbox_user": None,
        "record": NONE_ALIVE,
        "outputs_closed": True,
        "outputs_open_why": None,
        "links_before": RAN,
        "links_after": RAN,
    }
    return provenance.verdict(**{**kwargs, **given})


def test_verified_only_when_every_check_ran_and_passed():
    found = _verdict()
    assert found["verdict"] == "verified", found
    assert [c["check"] for c in found["checks"]] == [
        "outputs-closed",
        "links-before-test",
        "links-after-grading",
        "root-agent-processes",
    ]
    assert all(c["ran"] and c["passed"] for c in found["checks"])
    found = _verdict(record=ALIVE)
    assert found["verdict"] == "unverifiable"
    assert found["reasons"] == [provenance.ROOT_PROCESSES]
    assert found["processes"][0]["pid"] == 41


def test_a_check_that_could_not_run_is_named_and_fails_closed():
    cannot = {
        "ran": False,
        "why": "no record of the links before the agent's turn",
        "links": [],
    }
    for side in ("links_before", "links_after"):
        found = _verdict(**{side: cannot})
        assert found["verdict"] == "unverifiable", found
        assert found["links_checked"] is False
        (reason,) = found["reasons"]
        name = "links-before-test" if side == "links_before" else "links-after-grading"
        assert reason.startswith(f"the {name} check could not run (no record"), reason
    # Not made at all (the verify ended before the walk) is the same.
    assert _verdict(links_before=None)["verdict"] == "unverifiable"
    found = _verdict(record={"flags": [], "changes": []})
    assert found["verdict"] == "unverifiable"
    assert "root-agent-processes check could not run" in found["reasons"][0]


def test_verified_with_links_unchecked_is_impossible():
    """Every combination of the checks' states: never ``verified`` with
    ``links_checked`` false, and never ``verified`` with a check not run."""
    states = [
        {"ran": True, "why": None, "links": []},
        {"ran": False, "why": "the walk failed", "links": []},
        {
            "ran": True,
            "why": None,
            "links": [
                {"path": "/app/x", "target": "/logs/verifier/r", "removed": False}
            ],
        },
        None,
    ]
    for before, after, closed, record, user in itertools.product(
        states, states, (True, False, None), (ALIVE, NONE_ALIVE, {}), (None, "agent")
    ):
        found = _verdict(
            links_before=before,
            links_after=after,
            outputs_closed=closed,
            outputs_open_why=None if closed else "the close failed (rc=1)",
            record=record,
            sandbox_user=user,
        )
        if found["verdict"] == "verified":
            assert found["links_checked"] is True, found
            assert all(c["ran"] and c["passed"] for c in found["checks"]), found


def test_an_agent_user_is_not_marked_for_processes_and_the_oracle_is():
    """An agent user's processes are rule S's (asked, none alive); the
    oracle's are listed as a root agent's are (#1247)."""
    assert (
        _verdict(record=ALIVE, sandbox_user="agent", alive_at_test=NOT_ALIVE)["verdict"]
        == "verified"
    )
    found = _verdict(record=ALIVE, agent="oracle")
    assert found["verdict"] == "unverifiable", found
    assert found["reasons"] == [provenance.ORACLE_PROCESSES], found


def test_a_folder_not_closed_or_a_link_left_is_unverifiable():
    found = _verdict(
        outputs_closed=False, outputs_open_why="the verifier runs as uid 1000, not root"
    )
    assert "uid 1000, not root" in found["reasons"][0]
    left = {
        "ran": True,
        "why": None,
        "links": [{"path": "/app/g", "target": "/logs/verifier/reward.txt"}],
    }
    found = _verdict(links_after=left)
    assert found["verdict"] == "unverifiable"
    assert (
        "was there after grading (/app/g -> /logs/verifier/reward.txt)"
        in found["reasons"][0]
    )
    text = provenance.warning(_verdict(record=ALIVE))
    assert text.startswith("Reward provenance unverifiable: ") and "(pids 41)" in text
    assert provenance.warning(_verdict()) is None


def test_links_resolve_through_chains_and_folders():
    walked = {
        "/app/grade.log": "/logs/verifier/reward.txt",
        "/app/chain": "grade.log",
        "/app/out": "/logs/verifier",
        "/app/up": "../logs",
        "/home/agent/deep": "/app/out/sub/x",
        "/app/loop": "/app/loop2",
        "/app/loop2": "/app/loop",
        "/usr/bin/python": "python3.12",
        "/bin": "usr/bin",
    }
    assert links.resolve("/app/chain", walked) == "/logs/verifier/reward.txt"
    assert links.resolve("/home/agent/deep", walked) == "/logs/verifier/sub/x"
    assert links.resolve("/app/up", walked) == "/logs"
    assert links.resolve("/app/loop", walked) is None
    now = links.Walked("bfx", links=walked)
    found = {r["path"] for r in links.turn_links(None, now)}
    assert found == {
        "/app/grade.log",
        "/app/chain",
        "/app/out",
        "/app/up",
        "/home/agent/deep",
        "/app/loop",
        "/app/loop2",
    }
    # Links that were there before the turn, unchanged, are the image's.
    before = links.Walked(
        "bfx", links={"/app/out": "/logs/verifier", "/app/up": "/elsewhere"}
    )
    found = {r["path"] for r in links.turn_links(before, now)}
    assert "/app/out" not in found and "/app/up" in found  # its target changed


def test_the_run_start_line_is_said_once_for_a_root_agent_in_off_and_audit(caplog):
    provenance._SAID.clear()
    caplog.set_level(logging.WARNING, logger=provenance.__name__)
    for agent, mode, user in (
        ("claude-agent-acp", "audit", None),
        ("claude-agent-acp", "audit", None),
        ("claude-agent-acp", None, None),  # unset: audit, already said
        ("claude-agent-acp", "off", None),
        ("oracle", "audit", None),
        ("codex-acp", "audit", "agent"),
        ("codex-acp", "strict", None),
    ):
        provenance.say_run_start(agent, mode, user)
    said = [r.getMessage() for r in caplog.records if r.name == provenance.__name__]
    assert said == [
        provenance.run_start_line("audit"),
        provenance.run_start_line("off"),
    ]


@pytest.mark.bfx_refusal
def test_a_backend_with_no_bfx_route_says_so_in_plain_words():
    """Modal, AgentCore and Apple container have no bfx route, so a model
    agent's trial there is refused before its turn: the reason names the
    sandbox and what to use instead, and still starts with the text the
    ending is classified by."""

    class ModalSandbox:
        pass

    why = links.walker_unavailable(ModalSandbox())
    assert why.startswith(links.BFX_UNAVAILABLE), why
    assert "Modal can't run BenchFlow's integrity check yet" in why, why
    assert "Docker or Daytona" in why, why


# -- rule S (C3) ------------------------------------------------------------------------------------------------------


def test_s_a_process_of_the_agent_s_alive_at_test_sh_keeps_the_reward_unverifiable():
    """Rule S: a non-root agent's process alive when test.sh started reads `unverifiable` with the reason and the
    record; none alive reads `verified`. Root agents keep `root-agent-processes`; the oracle is asked only when its
    solve.sh ran as the task's `[agent].user` (#1247).

    Mutant: rule S's question skipped."""
    from benchflow.sandbox import _agent_alive

    found = _verdict(sandbox_user="agent", alive_at_test=STILL_ALIVE)
    assert found["verdict"] == "unverifiable", found
    assert found["reasons"] == [_agent_alive.ALIVE], found
    row = next(c for c in found["checks"] if c["check"] == _agent_alive.CHECK)
    assert row["ran"] is True and row["passed"] is False and "uid 1000" in row["why"], (
        row
    )
    assert found["alive_at_test"] == STILL_ALIVE, found
    assert (
        _verdict(sandbox_user="agent", alive_at_test=NOT_ALIVE)["verdict"] == "verified"
    )
    found = _verdict(alive_at_test=STILL_ALIVE)
    assert _agent_alive.CHECK not in [c["check"] for c in found["checks"]], found
    found = _verdict(agent="oracle")
    assert _agent_alive.CHECK not in [c["check"] for c in found["checks"]], found
    found = _verdict(sandbox_user="agent", agent="oracle", alive_at_test=STILL_ALIVE)
    assert found["verdict"] == "unverifiable", found
    assert found["alive_at_test"] == STILL_ALIVE, found


@pytest.mark.parametrize(
    "asked",
    [
        None,
        {
            "user": "agent",
            "alive": None,
            "why": "whether ... could not be told: PID 1 runs as the agent's uid",
        },
    ],
    ids=["not-made", "unanswered"],
)
def test_s_a_question_not_made_or_unanswered_is_never_verified(asked):
    """Fail closed: rule S's question not made (a verify that ended before it) or unanswered is a check that did
    not run, `unverifiable` with its reason; never `verified`. It never zeroes the reward (only a link check
    does)."""
    from benchflow.rollout import _grading_outcome

    found = _verdict(sandbox_user="agent", alive_at_test=asked)
    assert found["verdict"] == "unverifiable", found
    (reason,) = found["reasons"]
    assert reason.startswith("the agent-processes-at-test check could not run ("), (
        reason
    )
    assert _grading_outcome.incomplete_reason(found) is None, found


def _final(provenance, reward=1.0):
    from benchflow.rollout import _grading_outcome

    return _grading_outcome.final_outcome(
        _grading_outcome.Attempt(
            rewards={"reward": reward},
            verifier_error=None,
            diagnostic=None,
            test_sh_reward=True,
            provenance=provenance,
            agent="claude-agent-acp",
            sandbox_user="agent",
        )
    )


def test_s_keeps_the_reward_test_sh_wrote():
    """Rule S stops a `verified`, never a score: the reward test.sh wrote is kept."""
    rewards, error, _ = _final(
        _verdict(sandbox_user="agent", alive_at_test=STILL_ALIVE)
    )
    assert rewards == {"reward": 1.0} and error is None


@pytest.mark.parametrize(
    "how", ["walk-incomplete", "link-after-grading", "link-not-removed"]
)
def test_s_never_overrides_a_link_check_s_own_outcome(how):
    """Precedence: a link check that could not run, or that found a link, keeps its own outcome (the Q6 0), even
    with the agent's process alive at test.sh; rule S applies only where the trial would otherwise read
    `verified`.

    Mutant: the precedence flipped (rule S's kept reward over the Q6 0)."""
    from benchflow._utils.scoring import PROVENANCE_INCOMPLETE
    from benchflow.rollout import _grading_outcome

    link = {"path": "/app/x", "target": "/logs/verifier/reward.txt", "removed": False}
    given: dict[str, Any] = {
        "walk-incomplete": {
            "links_before": {
                "ran": False,
                "why": "1 name vanished during the walk",
                "links": [],
            }
        },
        "link-after-grading": {
            "links_after": {"ran": True, "why": None, "links": [link]}
        },
        "link-not-removed": {
            "links_before": {"ran": True, "why": None, "links": [link]}
        },
    }[how]
    provenance = _verdict(sandbox_user="agent", alive_at_test=STILL_ALIVE, **given)
    rewards, error, _ = _final(provenance)
    assert rewards == {"reward": 0.0} and error is None, provenance
    assert provenance["outcome"] == PROVENANCE_INCOMPLETE, provenance
    assert provenance["outcome_reason"] in (
        _grading_outcome.NOT_COMPLETE + " (1 name vanished during the walk)",
        _grading_outcome.APPEARED,
        _grading_outcome.NOT_REMOVED,
    ), provenance
