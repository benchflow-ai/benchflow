"""Tests for the continual-learning skill producer (``learner_skills``).

Capability 5 (continual learning) has two halves: the :class:`LearnerStore`
that versions an evolving (memory + skills) state, and the Memory-space scorer
that reads a ``memory_delta`` off a tree node. ``learner_skills`` is the data
path that connects them — it materializes the store's skills into a directory
the agent starts from, captures the skills the agent evolved back into the
store, and records the before/after delta the scorer reads.

These are pure filesystem translators — no sandbox, no live runs.
"""

from __future__ import annotations

from pathlib import Path

from benchflow.learner_skills import (
    capture_skills,
    materialize_skills,
    skill_memory_delta,
)
from benchflow.learner_store import LearnerState

# materialize_skills — store skills -> a skills_dir the agent reads


def test_materialize_writes_one_skill_pack_per_skill(tmp_path: Path) -> None:
    state = LearnerState(skills={"git-bisect": "do the bisect", "grep": "use rg"})
    materialize_skills(state, tmp_path)
    assert (tmp_path / "git-bisect" / "SKILL.md").read_text() == "do the bisect"
    assert (tmp_path / "grep" / "SKILL.md").read_text() == "use rg"


def test_materialize_empty_state_creates_empty_dir(tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    materialize_skills(LearnerState(), dest)
    assert dest.is_dir()
    assert list(dest.iterdir()) == []


def test_materialize_round_trips_through_capture(tmp_path: Path) -> None:
    """What materialize writes, capture reads back identically."""
    skills = {"a": "skill a body", "b": "skill b body"}
    materialize_skills(LearnerState(skills=skills), tmp_path)
    assert capture_skills(tmp_path) == skills


def test_materialize_is_idempotent_replacing_prior_content(
    tmp_path: Path,
) -> None:
    """A second materialize over the same dir reflects the new state."""
    materialize_skills(LearnerState(skills={"old": "v1"}), tmp_path)
    materialize_skills(LearnerState(skills={"new": "v1"}), tmp_path)
    assert capture_skills(tmp_path) == {"new": "v1"}


# capture_skills — an exported skills dir -> a skills dict for the store


def test_capture_reads_exported_skill_packs(tmp_path: Path) -> None:
    (tmp_path / "fix-flaky").mkdir()
    (tmp_path / "fix-flaky" / "SKILL.md").write_text("retry the test")
    assert capture_skills(tmp_path) == {"fix-flaky": "retry the test"}


def test_capture_missing_dir_is_empty(tmp_path: Path) -> None:
    assert capture_skills(tmp_path / "never-created") == {}


def test_capture_ignores_dirs_without_skill_md(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "SKILL.md").write_text("body")
    (tmp_path / "not-a-skill").mkdir()
    (tmp_path / "not-a-skill" / "README.md").write_text("nope")
    assert capture_skills(tmp_path) == {"real": "body"}


def test_capture_ignores_loose_files(tmp_path: Path) -> None:
    (tmp_path / "loose.txt").write_text("ignore me")
    (tmp_path / "skill-x").mkdir()
    (tmp_path / "skill-x" / "SKILL.md").write_text("x")
    assert capture_skills(tmp_path) == {"skill-x": "x"}


# skill_memory_delta — the record the MemoryScorer reads off the node


def test_skill_memory_delta_records_before_and_after() -> None:
    delta = skill_memory_delta(
        before={"a": "v1"},
        after={"a": "v2", "b": "v1"},
    )
    assert delta["before"] == {"a": "v1"}
    assert delta["after"] == {"a": "v2", "b": "v1"}


def test_skill_memory_delta_carries_expected_when_given() -> None:
    delta = skill_memory_delta(
        before={},
        after={"new": "v1"},
        expected=["new"],
    )
    assert delta["expected"] == ["new"]


def test_skill_memory_delta_preserves_empty_expected_fixture() -> None:
    """Guards the ENG-125 follow-up on v0.5-integration@ffef85d."""
    delta = skill_memory_delta(before={"a": "v1"}, after={"a": "v1"}, expected=[])
    assert delta["expected"] == []


def test_skill_memory_delta_omits_expected_when_absent() -> None:
    delta = skill_memory_delta(before={}, after={"new": "v1"})
    assert "expected" not in delta


def test_skill_memory_delta_is_consumable_by_memory_scorer() -> None:
    """The delta dict is exactly the MemoryScorer's node.state contract."""
    import asyncio

    from benchflow.rewards.memory_scorer import MEMORY_STATE_KEY, MemoryScorer
    from benchflow.trajectories.tree import RolloutNode

    delta = skill_memory_delta(
        before={"a": "v1"},
        after={"a": "v2"},
        expected=["a"],
    )
    node = RolloutNode(id="leaf", state={MEMORY_STATE_KEY: delta})
    event = asyncio.run(MemoryScorer().score(node))
    assert event.reward == 1.0
