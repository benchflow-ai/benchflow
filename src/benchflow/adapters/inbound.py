"""Inbound environment adapters — the shared edge contract.

An *inbound* adapter translates a foreign benchmark's task directory into
BenchFlow's native **Environment-plane** representation (architecture.md,
"The edges — adapters & trainers", and "The Environment plane & the
manifest"). The native seam is the :class:`EnvironmentManifest`; the
legacy ``task.toml`` + ``environment/`` / ``tests/`` / ``solution/``
subtree layout is preserved on :class:`InboundTask` as a downstream
compatibility layer so existing materialization paths keep working.

This module defines the pieces every inbound adapter shares:

* :class:`InboundTask` — the translation *result* and the real contract: a
  benchmark-agnostic, in-memory view of one task carrying a validated
  :class:`~benchflow.environment.manifest.EnvironmentManifest` (the
  Environment-plane seam), the instruction text, a validated
  :class:`~benchflow.task.config.TaskConfig` (legacy task-file
  compatibility), and a file map keyed by BenchFlow-native relative
  paths. Every inbound adapter is just a
  ``from_task_dir(Path) -> InboundTask`` classmethod.
* :func:`manifest_from_task_config` — derive a baseline
  :class:`EnvironmentManifest` from a foreign task's
  :class:`TaskConfig`. Adapters that find a sibling ``environment.toml``
  load it directly; adapters with no manifest file fall through to this
  helper so the Environment-plane seam is always populated.
* :func:`detect_adapter` — sniffs a task directory and returns the adapter
  whose foreign format it matches.

Adapters are *pure translators*: they read a directory and return data. They
do not build sandboxes, run verifiers, or touch the rollout runtime. The
``files`` map carries source paths under their BenchFlow-native destination
so a caller can materialize a runnable task directory without the adapter
performing any I/O of its own beyond reading.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from benchflow.environment.manifest import EnvironmentManifest

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


# Docker image tags accept ``[a-z0-9._-]`` plus a single leading
# alphanumeric. Foreign task names carry ``/`` (Harbor ``org/name``) and
# ``__`` (subtasks); both must be sanitized for the synthesized manifest
# image tag.
_IMAGE_TAG_INVALID = re.compile(r"[^a-z0-9._-]+")


def _sanitize_manifest_image(name: str) -> str:
    """Synthesize a deterministic local image tag for a foreign task.

    The framework builds the task's ``environment/Dockerfile`` into an
    image tagged ``bf__<name>:latest`` when no prebuilt image is declared.
    The manifest needs *some* concrete image identifier — this returns the
    same tag the build pipeline produces so the manifest is consistent
    with the legacy build path.
    """
    slug = _IMAGE_TAG_INVALID.sub("-", name.lower()).strip("-._") or "task"
    return f"bf__{slug}:latest"


def manifest_from_task_config(
    *,
    name: str,
    config: TaskConfig,
) -> EnvironmentManifest:
    """Derive a baseline :class:`EnvironmentManifest` from a foreign task.

    A Harbor- or Terminal-Bench-style foreign task does not ship an
    ``environment.toml``; its environment is described by the
    ``[environment]`` (now ``sandbox``) section of its ``task.toml`` /
    ``task.yaml`` plus a buildable ``environment/Dockerfile``. This helper
    folds that legacy shape into the Environment-plane seam:

    * ``manifest.image`` is the task's prebuilt ``docker_image`` when set,
      otherwise the ``bf__<name>:latest`` tag the framework's Dockerfile
      build path produces — so the manifest names a real, resolvable image
      either way.
    * ``owns_lifecycle`` is true (no framework-started services) so the
      manifest validates without a ``[[services]]`` array — legacy
      single-container Harbor/Terminal-Bench tasks have no separate
      service plane.
    * ``forward_env`` carries the foreign task's ``sandbox.env`` keys so
      the manifest's host-env forwarding surface stays honest to what the
      legacy path forwarded.

    Adapters that locate a sibling ``environment.toml`` should load it
    directly; this helper is the fallback for foreign tasks that have
    none, which is the common case.
    """
    sandbox = config.sandbox
    image = sandbox.docker_image or _sanitize_manifest_image(name)
    forward_keys = sorted(sandbox.env.keys())
    payload: dict[str, object] = {
        "name": name,
        "image": image,
        "owns_lifecycle": True,
    }
    if forward_keys:
        payload["forward_env"] = {"keys": forward_keys}
    return EnvironmentManifest.model_validate(payload)


@dataclass(frozen=True)
class InboundTask:
    """A foreign task translated into BenchFlow-native shape.

    This is the single result type every inbound adapter returns. It is a
    pure data record — no behavior, no runtime coupling.

    The Environment-plane seam is ``manifest`` (architecture.md, "The
    Environment plane & the manifest"). The ``config`` / ``files`` fields
    are the legacy task-file compatibility layer — they let a downstream
    consumer still materialize a runnable BenchFlow-native task directory
    from foreign sources while the manifest is what the Environment plane
    actually runs.

    Attributes:
        name: The task identity (BenchFlow ``org/name`` form when the source
            provides one; otherwise derived from the source).
        source: The foreign format this came from (e.g. ``"harbor"``,
            ``"terminal-bench"``) — recorded so downstream tooling can trace
            provenance.
        instruction: The task instruction, as it belongs in ``instruction.md``.
        manifest: The validated :class:`EnvironmentManifest` for this task
            — the Environment-plane integration seam. Either loaded from a
            sibling ``environment.toml`` (when the foreign task ships one)
            or derived from :func:`manifest_from_task_config` (the common
            single-container Harbor / Terminal-Bench case).
        config: A validated native :class:`TaskConfig` — the legacy
            ``task.toml`` equivalent, kept for backward compatibility with
            file-materialization paths.
        files: Map of *BenchFlow-native relative path* -> *source file path*.
            Keys are paths under :data:`NATIVE_SUBTREES`; values are real,
            existing paths in the foreign task directory. A consumer copies
            each value to its key to materialize a runnable task.
    """

    name: str
    source: str
    instruction: str
    manifest: EnvironmentManifest
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
