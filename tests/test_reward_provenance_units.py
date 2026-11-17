"""The reward's provenance verdict, the links walk's resolution, and the run-start line (the lead's 20:31Z and 22:25Z rulings)."""

from __future__ import annotations

import itertools
import logging

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


def _verdict(**given):
    kwargs = {
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


def test_an_agent_user_and_the_oracle_are_not_marked_for_processes():
    assert _verdict(record=ALIVE, sandbox_user="agent")["verdict"] == "verified"
    assert _verdict(record=ALIVE, agent="oracle")["verdict"] == "verified"


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
