"""Train/test splits for ``bench hillclimb``.

Every task goes to ``train`` or ``test``. The optimizer may read the train
split's failures; the test split never reaches it, and only aggregate scores
with confidence intervals leave it (see :mod:`benchflow.hillclimbing.proposer`).

A split is either made here, at random with a seed, or read from a split
file. A made split is stratified by one task metadata key (``category`` by
default): each stratum sends ``round(n * test_frac)`` of its tasks to test,
with the rounding remainders shared out largest first, so the whole split has
``round(N * test_frac)`` test tasks and each stratum is represented about in
proportion. Tasks without the key form one stratum of their own.

The same seed and task set always give the same split. The split is saved as
``split.json`` in the run folder, and a split file has the same shape::

    {"train": ["task-a", "task-b"], "test": ["task-c"]}

>>> s = make_split({"a": None, "b": None, "c": None, "d": None}, test_frac=0.5, seed=1)
>>> sorted(s.train + s.test), len(s.test)
(['a', 'b', 'c', 'd'], 2)
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SPLIT_FILE = "split.json"
NO_STRATUM = "(none)"

SplitName = Literal["train", "test"]


class SplitError(ValueError):
    """A split that cannot be made or a split file that does not fit the tasks."""


@dataclass(frozen=True)
class Split:
    """Which tasks are train and which are test, and how that was decided."""

    train: tuple[str, ...]
    test: tuple[str, ...]
    method: Literal["random", "stratified", "file"]
    seed: int | None = None
    test_frac: float | None = None
    stratify_by: str | None = None
    # task -> stratum label, for made splits that were stratified.
    strata: dict[str, str] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        overlap = sorted(set(self.train) & set(self.test))
        if overlap:
            raise SplitError(f"tasks in both train and test: {', '.join(overlap)}")
        if not self.train or not self.test:
            raise SplitError("a split needs at least one train and one test task")

    def split_of(self, task: str) -> SplitName:
        if task in self.train:
            return "train"
        if task in self.test:
            return "test"
        raise KeyError(task)

    def tasks(self, split: SplitName) -> tuple[str, ...]:
        return self.train if split == "train" else self.test

    def without(self, tasks: Iterable[str]) -> Split:
        """This split with ``tasks`` removed from both sides."""
        drop = set(tasks)
        return Split(
            train=tuple(t for t in self.train if t not in drop),
            test=tuple(t for t in self.test if t not in drop),
            method=self.method,
            seed=self.seed,
            test_frac=self.test_frac,
            stratify_by=self.stratify_by,
            strata={k: v for k, v in self.strata.items() if k not in drop},
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "test": list(self.test),
            "method": self.method,
            "seed": self.seed,
            "test_frac": self.test_frac,
            "stratify_by": self.stratify_by,
            "strata": dict(sorted(self.strata.items())),
            "source": self.source,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def _quotas(sizes: Mapping[str, int], test_frac: float, rng: random.Random) -> dict:
    """Test tasks per stratum: floors first, then the largest remainders."""
    total = sum(sizes.values())
    target = min(max(round(total * test_frac), 1), total - 1)
    exact = {s: n * test_frac for s, n in sizes.items()}
    quota = {s: min(math.floor(x), sizes[s]) for s, x in exact.items()}
    # Ties in the remainder are broken at random (seeded), never by name.
    order = sorted(exact, key=lambda s: (-(exact[s] - quota[s]), rng.random()))
    i = 0
    while sum(quota.values()) < target and order:
        s = order[i % len(order)]
        if quota[s] < sizes[s]:
            quota[s] += 1
        i += 1
    while sum(quota.values()) > target:
        s = max(quota, key=lambda k: (quota[k], k))
        quota[s] -= 1
    return quota


def make_split(
    strata: Mapping[str, str | None],
    *,
    test_frac: float,
    seed: int,
    stratify_by: str | None = None,
) -> Split:
    """A seeded split of ``strata``'s tasks (task name -> stratum or None)."""
    if not 0 < test_frac < 1:
        raise SplitError(f"--test-frac must be between 0 and 1, got {test_frac}")
    names = sorted(strata)
    if len(names) < 2:
        raise SplitError(f"a train/test split needs at least 2 tasks, got {len(names)}")
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(strata[name] or NO_STRATUM, []).append(name)
    rng = random.Random(seed)
    quotas = _quotas({s: len(v) for s, v in groups.items()}, test_frac, rng)
    train: list[str] = []
    test: list[str] = []
    for stratum in sorted(groups):
        members = list(groups[stratum])
        rng.shuffle(members)
        test.extend(members[: quotas[stratum]])
        train.extend(members[quotas[stratum] :])
    stratified = len(groups) > 1
    return Split(
        train=tuple(sorted(train)),
        test=tuple(sorted(test)),
        method="stratified" if stratified else "random",
        seed=seed,
        test_frac=test_frac,
        stratify_by=stratify_by if stratified else None,
        strata=({n: strata[n] or NO_STRATUM for n in names} if stratified else {}),
    )


def load_split_file(path: str | Path, tasks: Iterable[str]) -> Split:
    """Read a split file and check it covers exactly ``tasks``.

    A task in the file that is not among ``tasks``, or a task of ``tasks``
    the file does not place, is an error: a silently dropped task would
    change what the test score means.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SplitError(f"cannot read split file {path}: {exc}") from None
    if not isinstance(raw, dict) or not all(
        isinstance(raw.get(k), list) for k in ("train", "test")
    ):
        raise SplitError(
            f'{path}: a split file is a JSON object {{"train": [...], "test": [...]}}'
        )
    train = [str(t) for t in raw["train"]]
    test = [str(t) for t in raw["test"]]
    known = set(tasks)
    placed = set(train) | set(test)
    unknown = sorted(placed - known)
    if unknown:
        raise SplitError(
            f"{path} names tasks that are not in the task set: {', '.join(unknown)}"
        )
    missing = sorted(known - placed)
    if missing:
        raise SplitError(
            f"{path} does not place these tasks in train or test: {', '.join(missing)}"
        )
    for side, items in (("train", train), ("test", test)):
        dupes = sorted({t for t in items if items.count(t) > 1})
        if dupes:
            raise SplitError(f"{path} lists {side} tasks twice: {', '.join(dupes)}")
    return Split(
        train=tuple(sorted(train)),
        test=tuple(sorted(test)),
        method="file",
        seed=raw.get("seed") if isinstance(raw.get("seed"), int) else None,
        test_frac=None,
        stratify_by=None,
        source=str(path),
    )


def task_strata(
    task_dirs: Mapping[str, Path], stratify_by: str | None
) -> dict[str, str | None]:
    """Each task's value of the ``stratify_by`` metadata key (None when absent)."""
    if not stratify_by:
        return dict.fromkeys(task_dirs)
    out: dict[str, str | None] = {}
    for name, path in task_dirs.items():
        value = task_metadata(path).get(stratify_by)
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        out[name] = None if value in (None, "") else str(value)
    return out


def task_metadata(task_dir: Path) -> dict[str, Any]:
    """The task's ``metadata`` table, or ``{}`` when it cannot be read."""
    try:
        from benchflow.task import Task

        metadata = Task(task_dir).config.metadata
    except Exception:
        return {}
    return dict(metadata) if isinstance(metadata, dict) else {}
