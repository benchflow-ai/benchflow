"""The LearnerStore — the persistent, versioned store behind continual learning.

Continual learning (architecture.md § "The eight capabilities", row 5) is a
**Job run in ``sequential-shared`` mode**: Rollouts run in order over one
persistent **learner store** of memory + skills. It is a Job *mode*, not a new
top-level object.

The store has three defining properties:

* **Persistent** — it lives across the whole Job; every rollout reads the
  current state and may write an evolved one back.
* **Versioned** — every write stamps a monotonic *generation* counter, so a
  rollout's contribution is addressable after the fact.
* **Rollback-capable** — :meth:`LearnerStore.commit_or_revert` rejects a
  regression when a learning-curve metric drops, keeping the store at the
  better generation.

The learner store is deliberately the *one* snapshot layer that does **not**
roll back with a ``Branch`` (architecture.md § "Lifecycles"): a Branch rolls
back the world (container / environment-state / agent-session) to estimate
``V(s)``; the learner store is the agent's accumulating memory and must survive
those forks. Its rollback is a *separate*, generation-scoped operation driven by
the learning curve, not by the tree.

This module is pure data — no I/O, no sandbox, no live runs. The Job
orchestrator (``evaluation.py``) drives it; the store itself just holds and
versions state.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["LearnerState", "LearnerStore"]


@dataclass
class LearnerState:
    """The (memory + skills) payload a continual-learning Job evolves.

    * ``memory`` — accumulated lessons / facts the agent carries forward.
    * ``skills`` — the evolving skill library (name -> skill body); skills stay
      useful only if continuously evolved (Han, architecture.md).

    Both are plain dicts so the store stays serialisable and the tests can drive
    it against fakes with no live runs.
    """

    memory: dict[str, Any] = field(default_factory=dict)
    skills: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> LearnerState:
        """A deep, independent copy — mutating it never touches the original."""
        return LearnerState(
            memory=copy.deepcopy(self.memory),
            skills=copy.deepcopy(self.skills),
        )


@dataclass
class Generation:
    """One versioned entry in the learner store's history.

    ``number`` is the generation counter (generation 0 is the empty store);
    ``state`` is the committed (memory + skills) snapshot; ``metric`` is the
    learning-curve value the rollout that produced this generation scored, used
    to detect regressions.
    """

    number: int
    state: LearnerState
    metric: float | None = None


class LearnerStore:
    """A persistent, generation-versioned store of memory + skills.

    Generation 0 is the empty store. Each :meth:`commit` deep-copies the passed
    state, stamps a **monotonic** generation number, and appends it to
    :attr:`history`. :meth:`_revert` rolls back to an earlier generation,
    dropping every later one — but it does *not* free the numbers it dropped:
    a later commit stamps the next never-used number, so every generation
    number is a durable, unique, per-rollout stamp (a reverted run never
    aliases the run that replaced it). :meth:`current` always returns a fresh
    copy, never the live object, so a reader can mutate it freely and only a
    :meth:`commit` makes the change stick.
    """

    def __init__(self) -> None:
        self.history: dict[int, Generation] = {
            0: Generation(number=0, state=LearnerState())
        }
        self._generation = 0
        # The next generation number to stamp — monotonic, never decreased
        # by revert(), so a generation number is never reused.
        self._next_number = 1

    @property
    def generation(self) -> int:
        """The current generation number (the live pointer into history)."""
        return self._generation

    def current(self) -> LearnerState:
        """A fresh deep copy of the current generation's state.

        Callers may mutate the returned object freely; nothing sticks until it
        is handed to :meth:`commit`.
        """
        return self.history[self._generation].state.copy()

    def commit(self, state: LearnerState, *, metric: float | None = None) -> int:
        """Stamp ``state`` as the next generation and return its number.

        The number is monotonic — it is the next never-used number, even
        after a :meth:`_revert` dropped higher ones. ``state`` is deep-copied
        on the way in, so mutating the caller's object afterwards never leaks
        into the committed generation.
        """
        number = self._next_number
        self.history[number] = Generation(
            number=number, state=state.copy(), metric=metric
        )
        self._generation = number
        self._next_number = number + 1
        return number

    def _revert(self, generation: int) -> None:
        """Roll the store back to ``generation``, dropping every later one.

        Internal rollback primitive. The public continual-learning step,
        :meth:`commit_or_revert`, *rejects* a regression before it is
        committed, so it never needs to roll back — :meth:`_revert` exists
        for an explicit, out-of-band rollback to a known-good generation.

        Only goes backward: ``generation`` must be a known generation no later
        than the current one. The dropped numbers are *not* freed — a later
        commit stamps a fresh number, so generation numbers stay monotonic and
        unique (each is a durable per-rollout stamp).
        """
        if generation not in self.history:
            raise ValueError(
                f"unknown generation {generation} — known: {sorted(self.history)}"
            )
        if generation > self._generation:
            raise ValueError(
                f"cannot revert forward to generation {generation} "
                f"(current is {self._generation}) — revert only goes back"
            )
        for later in [n for n in self.history if n > generation]:
            del self.history[later]
        self._generation = generation

    def learning_curve(self) -> list[float | None]:
        """The committed generations' metrics, in generation order.

        Generation 0 (the empty store) is excluded — it had no rollout and so
        no metric. The result is the learning curve the Job plots.
        """
        return [self.history[n].metric for n in sorted(self.history) if n != 0]

    def _best_metric(self) -> float | None:
        """The highest metric across committed generations, or None if none."""
        metrics = [m for m in self.learning_curve() if m is not None]
        return max(metrics) if metrics else None

    def _regressed(self, metric: float, *, tolerance: float = 0.0) -> bool:
        """Whether ``metric`` is a regression against the best generation so far.

        Internal — the public continual-learning step is
        :meth:`commit_or_revert`.

        A regression is a drop below ``best_so_far - tolerance``. With no prior
        metric there is nothing to regress against, so the result is ``False``.
        ``tolerance`` absorbs noise: a dip within it is not counted.
        """
        best = self._best_metric()
        if best is None:
            return False
        return metric < best - tolerance

    def commit_or_revert(
        self, state: LearnerState, *, metric: float, tolerance: float = 0.0
    ) -> bool:
        """The continual-learning step: keep an improvement, reject a regression.

        If ``metric`` regresses against the best generation so far the store is
        left untouched (the new ``state`` is discarded) and ``False`` is
        returned. Otherwise ``state`` is committed and ``True`` is returned.
        """
        if self._regressed(metric, tolerance=tolerance):
            return False
        self.commit(state, metric=metric)
        return True

    # --- persistence ---
    #
    # The store is otherwise pure data, but a continual-learning Job survives
    # process death only if it can be reloaded from disk on resume. These
    # helpers serialize the full history (every generation, in number order)
    # plus the live pointer and the next-number counter, so a reloaded store
    # is byte-identical to the in-memory one. The Job orchestrator
    # (``evaluation.py``) writes the file under the job directory and reads
    # it back on resume; the store itself just knows the shape.

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot of the entire store — every generation."""
        return {
            "version": 1,
            "generation": self._generation,
            "next_number": self._next_number,
            "history": [
                {
                    "number": gen.number,
                    "metric": gen.metric,
                    "memory": copy.deepcopy(gen.state.memory),
                    "skills": copy.deepcopy(gen.state.skills),
                }
                for gen in (self.history[n] for n in sorted(self.history))
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnerStore:
        """Rebuild a store from a :meth:`to_dict` snapshot."""
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(
                f"unsupported learner_store snapshot version {version!r} — expected 1"
            )
        store = cls()
        store.history = {
            int(entry["number"]): Generation(
                number=int(entry["number"]),
                state=LearnerState(
                    memory=copy.deepcopy(entry.get("memory") or {}),
                    skills=copy.deepcopy(entry.get("skills") or {}),
                ),
                metric=entry.get("metric"),
            )
            for entry in data.get("history") or []
        }
        if 0 not in store.history:
            # Generation 0 is the empty starting state; a snapshot that drops
            # it would corrupt :meth:`_revert` and :meth:`learning_curve`.
            store.history[0] = Generation(number=0, state=LearnerState())
        store._generation = int(data.get("generation", 0))
        # next_number defaults to one past the highest known number, so a
        # store with no explicit counter still stamps unique generations.
        store._next_number = int(data.get("next_number", max(store.history) + 1))
        if store._generation not in store.history:
            raise ValueError(
                f"learner_store snapshot pointer generation={store._generation} "
                f"missing from history {sorted(store.history)}"
            )
        return store

    def save(self, path: Path | str) -> None:
        """Atomically write a JSON snapshot to ``path``.

        Atomic via write-then-rename so a crash mid-save leaves the previous
        snapshot intact — a half-written file would silently corrupt the
        learning curve on the next resume.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> LearnerStore:
        """Load a snapshot from ``path``."""
        return cls.from_dict(json.loads(Path(path).read_text()))
