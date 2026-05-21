"""Inbound environment adapters — the shared edge contract.

An *inbound* adapter translates a foreign benchmark's task directory into
BenchFlow's native task format (architecture.md, "The edges — adapters &
trainers") — ``task.toml`` plus the ``instruction.md`` / ``environment/`` /
``tests/`` / ``solution/`` layout. Inbound adapters translate every other
format to it so a foreign benchmark runs natively.

This module defines the pieces every inbound adapter shares:

* :class:`InboundTask` — the translation *result* and the real contract: a
  benchmark-agnostic, in-memory view of one task in BenchFlow-native shape
  (instruction text, a validated :class:`~benchflow.task.config.TaskConfig`,
  and a file map keyed by BenchFlow-native relative paths). Every inbound
  adapter is just a ``from_task_dir(Path) -> InboundTask`` classmethod.
* :func:`detect_adapter` — sniffs a task directory and returns the adapter
  whose foreign format it matches.

Adapters are *pure translators*: they read a directory and return data. They
do not build sandboxes, run verifiers, or touch the rollout runtime. The
``files`` map carries source paths under their BenchFlow-native destination
so a caller can materialize a runnable task directory without the adapter
performing any I/O of its own beyond reading.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchflow.task.config import TaskConfig

# A BenchFlow-native task directory has exactly these top-level subtrees
# (see benchflow.task.task.Task). Inbound file maps key into them.
NATIVE_SUBTREES = ("environment", "tests", "solution")


def carry_native_subtrees(
    root: Path,
    place: Callable[[str, Path], None],
    subtrees: tuple[str, ...] = NATIVE_SUBTREES,
) -> None:
    """Carry every native-subtree file through ``place(native_rel, src)``.

    A native task directory's ``environment`` / ``tests`` / ``solution``
    subtrees are *already* in BenchFlow-native shape, so every file under
    them carries straight through under its own relative path. Both inbound
    adapters share this rglob-and-carry walk — harbor.py over all
    :data:`NATIVE_SUBTREES`, terminal_bench.py over ``tests`` alone.

    Files are visited in a stable, sorted order so the caller's collision
    handling is deterministic. The ``place`` callback owns the collision
    policy: ``dict.setdefault`` for a tolerant carry, or a raising check
    for strict adapters.
    """
    for subtree in subtrees:
        sub = root / subtree
        if not sub.is_dir():
            continue
        for src in sorted(p for p in sub.rglob("*") if p.is_file()):
            place(src.relative_to(root).as_posix(), src)


@dataclass(frozen=True)
class InboundTask:
    """A foreign task translated into BenchFlow-native shape.

    This is the single result type every inbound adapter returns. It is a
    pure data record — no behavior, no runtime coupling.

    Attributes:
        name: The task identity (BenchFlow ``org/name`` form when the source
            provides one; otherwise derived from the source).
        source: The foreign format this came from (e.g. ``"harbor"``,
            ``"terminal-bench"``) — recorded so downstream tooling can trace
            provenance.
        instruction: The task instruction, as it belongs in ``instruction.md``.
        config: A validated native :class:`TaskConfig` — the translated
            ``task.toml`` equivalent.
        files: Map of *BenchFlow-native relative path* -> *source file path*.
            Keys are paths under :data:`NATIVE_SUBTREES`; values are real,
            existing paths in the foreign task directory. A consumer copies
            each value to its key to materialize a runnable task.
    """

    name: str
    source: str
    instruction: str
    config: TaskConfig
    files: dict[str, Path] = field(default_factory=dict)


if TYPE_CHECKING:
    from benchflow.adapters.harbor import HarborAdapter
    from benchflow.adapters.terminal_bench import TerminalBenchAdapter

    # An inbound adapter is just a class with a ``from_task_dir(Path) ->
    # InboundTask`` classmethod — the two concrete ones. No standalone
    # Protocol: InboundTask is the real contract, the adapter is its producer.
    InboundAdapterType = type[HarborAdapter] | type[TerminalBenchAdapter]


def detect_adapter(task_dir: Path | str) -> InboundAdapterType:
    """Return the inbound adapter whose format ``task_dir`` matches.

    Detection is by signature file: Harbor task dirs carry a ``task.toml``,
    Terminal-Bench task dirs carry a ``task.yaml``. ``task.toml`` is checked
    first so a directory carrying both is treated as Harbor (the native
    superset format).

    Raises:
        ValueError: if the directory matches no known foreign format.
    """
    # Imported here to avoid a module-load cycle: the concrete adapters
    # import InboundTask from this module.
    from benchflow.adapters.harbor import HarborAdapter
    from benchflow.adapters.terminal_bench import TerminalBenchAdapter

    root = Path(task_dir)
    if (root / "task.toml").is_file():
        return HarborAdapter
    if (root / "task.yaml").is_file() or (root / "task.yml").is_file():
        return TerminalBenchAdapter
    raise ValueError(
        f"Unrecognized task format in {root}: expected a Harbor 'task.toml' "
        f"or a Terminal-Bench 'task.yaml'."
    )
