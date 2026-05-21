"""The continual-learning skill producer — the data path of capability 5.

Capability 5 (continual learning, ``architecture.md`` § "The eight
capabilities", row 5) has two halves that were built separately:

* the :class:`~benchflow.learner_store.LearnerStore` — a persistent,
  generation-versioned store of the agent's evolving (memory + skills) state;
* the Memory-space scorer (:class:`~benchflow.rewards.memory_scorer.MemoryScorer`)
  — which reads a ``memory_delta`` record off a tree node and scores whether
  the agent updated its skills correctly.

This module is the **producer** that connects them. Without it the store's
``skills`` are never populated by a rollout and the scorer's ``memory_delta``
is never written — two halves with no data path.

The mechanism deliberately *reuses* BenchFlow's existing skill machinery
(``RolloutConfig.skills_dir`` for injection, ``export_generated_skills_to`` for
capture — see ``rollout.py``), it does not reinvent it:

* :func:`materialize_skills` writes a :class:`LearnerState`'s skills out as a
  directory of ``<name>/SKILL.md`` packs — exactly the ``skills_dir`` layout
  the agent reads, so rollout N starts from the store's *evolved* skill set.
* :func:`capture_skills` reads an exported skills directory (the
  ``export_generated_skills_to`` target — again ``<name>/SKILL.md`` packs)
  back into a plain ``dict``, ready to be committed as the next
  :class:`LearnerState`.
* :func:`skill_memory_delta` builds the ``{before, after, expected}`` record
  the Memory scorer expects on ``node.state["memory_delta"]``.

Pure filesystem translation — no sandbox, no async, no live runs. The Job
orchestrator (``evaluation.py``) drives it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from benchflow.learner_store import LearnerState

#: The single skill file inside each ``<name>/`` skill pack directory.
SKILL_FILE = "SKILL.md"


def materialize_skills(state: LearnerState, dest: Path | str) -> Path:
    """Write ``state.skills`` to ``dest`` as a directory of skill packs.

    Each skill becomes ``dest/<name>/SKILL.md`` whose body is the skill's
    stored content. This is the ``skills_dir`` layout the rollout injects
    into the sandbox, so a continual-learning rollout starts from the
    store's current (evolved) skill set.

    ``dest`` is recreated from scratch each call so a stale skill from an
    earlier generation cannot leak forward. Returns ``dest`` as a ``Path``.
    """
    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    for name, body in state.skills.items():
        pack = dest_path / str(name)
        pack.mkdir(parents=True, exist_ok=True)
        (pack / SKILL_FILE).write_text(_body_text(body))
    return dest_path


def capture_skills(export_dir: Path | str) -> dict[str, str]:
    """Read an exported skills directory back into a ``name -> body`` dict.

    The inverse of :func:`materialize_skills`: scans ``export_dir`` for
    ``<name>/SKILL.md`` packs and returns their bodies. This is what a
    continual-learning rollout produced — the skills the agent generated or
    evolved — ready to commit as the next :class:`LearnerState`.

    A missing directory, loose files, and directories without a ``SKILL.md``
    are all ignored: the result is empty rather than an error, so a rollout
    that evolved nothing simply commits an unchanged skill set.
    """
    root = Path(export_dir)
    if not root.is_dir():
        return {}
    skills: dict[str, str] = {}
    for pack in sorted(p for p in root.iterdir() if p.is_dir()):
        skill_md = pack / SKILL_FILE
        if skill_md.is_file():
            skills[pack.name] = skill_md.read_text()
    return skills


def skill_memory_delta(
    *,
    before: dict[str, str],
    after: dict[str, str],
    expected: list[str] | None = None,
) -> dict[str, Any]:
    """Build the ``memory_delta`` record the Memory scorer reads off a node.

    The returned dict is exactly the contract
    :class:`~benchflow.rewards.memory_scorer.MemoryScorer` expects on
    ``node.state[MEMORY_STATE_KEY]``: ``before`` / ``after`` skill snapshots
    and an optional ``expected`` fixture (the skill names the task expected
    to change). ``expected`` is omitted entirely when ``None`` so the scorer
    falls back to scoring activity rather than correctness.
    """
    delta: dict[str, Any] = {"before": dict(before), "after": dict(after)}
    if expected is not None:
        delta["expected"] = list(expected)
    return delta


def _body_text(body: Any) -> str:
    """Coerce a stored skill body to text — bodies are normally strings."""
    return body if isinstance(body, str) else str(body)
