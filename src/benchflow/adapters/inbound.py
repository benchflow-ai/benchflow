"""Inbound environment adapters — the shared edge contract.

An *inbound* adapter translates a foreign benchmark's task directory into
BenchFlow's native task format (architecture.md, "The edges — adapters &
trainers"). The manifest is BenchFlow's native format; inbound adapters
translate every other format to it so a foreign benchmark runs natively.

This module defines the pieces every inbound adapter shares:

* :class:`InboundTask` — the translation *result*: a benchmark-agnostic,
  in-memory view of one task in BenchFlow-native shape (instruction text, a
  validated :class:`~benchflow.task.config.TaskConfig`, and a file map keyed
  by BenchFlow-native relative paths).
* :data:`InboundAdapter` — the structural protocol every adapter satisfies:
  a ``from_task_dir`` classmethod that is a pure ``Path -> InboundTask``
  translation.
* :func:`detect_adapter` — sniffs a task directory and returns the adapter
  whose foreign format it matches.

Adapters are *pure translators*: they read a directory and return data. They
do not build sandboxes, run verifiers, or touch the rollout runtime. The
``files`` map carries source paths under their BenchFlow-native destination
so a caller can materialize a runnable task directory without the adapter
performing any I/O of its own beyond reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from benchflow.task.config import TaskConfig

# A BenchFlow-native task directory has exactly these top-level subtrees
# (see benchflow.task.task.Task). Inbound file maps key into them.
NATIVE_SUBTREES = ("environment", "tests", "solution")


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


@runtime_checkable
class InboundAdapter(Protocol):
    """Structural protocol for an inbound environment adapter.

    An inbound adapter is anything exposing a ``from_task_dir`` classmethod
    that purely translates a foreign task directory to an :class:`InboundTask`.
    """

    source: str

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        """Translate a foreign task directory into an :class:`InboundTask`."""
        ...


def detect_adapter(task_dir: Path | str) -> type[InboundAdapter]:
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
