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
import shutil
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

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
    them carries straight through under its own relative path. harbor.py
    uses this rglob-and-carry walk over all :data:`NATIVE_SUBTREES`.

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

    A Harbor-style foreign task does not ship an ``environment.toml``; its
    environment is described by the ``[environment]`` (now ``sandbox``)
    section of its ``task.toml`` plus a buildable ``environment/Dockerfile``.
    This helper folds that legacy shape into the Environment-plane seam:

    * ``manifest.image`` is the task's prebuilt ``docker_image`` when set,
      otherwise the ``bf__<name>:latest`` tag the framework's Dockerfile
      build path produces — so the manifest names a real, resolvable image
      either way.
    * ``owns_lifecycle`` is true (no framework-started services) so the
      manifest validates without a ``[[services]]`` array — legacy
      single-container Harbor tasks have no separate service plane.
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


def materialize_inbound_task_md(
    task: InboundTask,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an :class:`InboundTask` as a native ``task.md`` package.

    Inbound adapters stay pure: they read a foreign task and return an
    in-memory :class:`InboundTask`. This helper is the shared consumer-facing
    bridge from that adapter result to a runnable BenchFlow-native directory.
    Legacy compatibility file-map destinations are promoted while copying:
    ``tests/`` becomes ``verifier/`` and ``solution/`` becomes ``oracle/``.
    """

    copy_plan = [
        (_task_md_file_map_destination(native_rel), source)
        for native_rel, source in sorted(task.files.items())
    ]
    generated_plan = [
        (_task_md_file_map_destination(native_rel), content)
        for native_rel, content in sorted(task.generated_files.items())
    ]
    copy_targets = {target for target, _source in copy_plan}
    generated_targets = {target for target, _content in generated_plan}
    collisions = copy_targets & generated_targets
    if collisions:
        rendered = ", ".join(path.as_posix() for path in sorted(collisions))
        raise ValueError(f"Inbound generated file collision: {rendered}")

    dest = Path(output_dir)
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"Inbound materialization destination exists: {dest}")
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.mkdir(parents=True)

    frontmatter = tomllib.loads(task.config.model_dump_toml())
    if task.compatibility is not None and (
        task.compatibility.config_extra or task.compatibility.config_extra_paths
    ):
        benchflow = frontmatter.setdefault("benchflow", {})
        benchflow["compat"] = task.compatibility.to_dict()

    from benchflow.task.document import render_task_md

    (dest / "task.md").write_text(render_task_md(frontmatter, task.instruction))
    for target_rel, source in copy_plan:
        target = dest / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for target_rel, content in generated_plan:
        target = dest / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)

    _ensure_script_verifier_contract(dest)
    return dest


def _task_md_file_map_destination(native_rel: str) -> Path:
    path = PurePosixPath(native_rel)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Inbound file-map path is not safe relative: {native_rel}")
    parts = list(path.parts)
    match parts[0]:
        case "tests":
            parts[0] = "verifier"
        case "solution":
            parts[0] = "oracle"
        case "environment" | "verifier" | "oracle":
            pass
        case other:
            raise ValueError(f"Inbound file-map path uses unsupported subtree: {other}")
    return Path(*parts)


def _ensure_script_verifier_contract(task_dir: Path) -> None:
    verifier_dir = task_dir / "verifier"
    test_sh = verifier_dir / "test.sh"
    if not test_sh.is_file():
        return

    verifier_md = verifier_dir / "verifier.md"
    if not verifier_md.exists():
        verifier_md.write_text(
            """---
verifier:
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      correctness: {weight: 1.0, source: deterministic}
  outputs:
    reward_json: /logs/verifier/reward.json
---
"""
        )

    rubrics_dir = verifier_dir / "rubrics"
    rubrics_dir.mkdir(exist_ok=True)
    if not any(child.is_file() for child in rubrics_dir.rglob("*")):
        (rubrics_dir / "verifier.md").write_text(
            "Verify the foreign benchmark's original deterministic test contract.\n"
        )


@dataclass(frozen=True)
class InboundCompatibility:
    """Foreign-format data preserved outside the native task schema."""

    source: str
    config_extra: dict[str, Any] = field(default_factory=dict)
    config_extra_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_extra_paths"] = list(self.config_extra_paths)
        return data


@dataclass(frozen=True)
class InboundSupportReport:
    """Whether an adapter can translate a foreign task, and why not."""

    source: str
    supported: bool
    task_id: str | None = None
    dataset: str | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnsupportedInboundTaskError(ValueError):
    """Raised when an adapter recognizes a task but cannot translate it yet."""

    def __init__(self, report: InboundSupportReport) -> None:
        self.report = report
        task = f" {report.task_id}" if report.task_id else ""
        dataset = f" ({report.dataset})" if report.dataset else ""
        reason = report.reason or "unsupported task shape"
        super().__init__(f"{report.source}{task}{dataset}: {reason}")


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
        source: The foreign format this came from (e.g. ``"harbor"``) —
            recorded so downstream tooling can trace provenance.
        instruction: The task instruction, as it belongs in ``instruction.md``.
        manifest: The validated :class:`EnvironmentManifest` for this task
            — the Environment-plane integration seam. Either loaded from a
            sibling ``environment.toml`` (when the foreign task ships one)
            or derived from :func:`manifest_from_task_config` (the common
            single-container Harbor case).
        config: A validated native :class:`TaskConfig` — the legacy
            ``task.toml`` equivalent, kept for backward compatibility with
            file-materialization paths.
        files: Map of *BenchFlow-native relative path* -> *source file path*.
            Keys are paths under :data:`NATIVE_SUBTREES`; values are real,
            existing paths in the foreign task directory. A consumer copies
            each value to its key to materialize a runnable task.
        generated_files: Map of *BenchFlow-native relative path* -> in-memory
            content for native files synthesized from structured foreign
            schemas, such as evaluator JSON. This keeps adapters pure while
            still letting them materialize runnable verifier/oracle assets.
        compatibility: Explicit foreign-format metadata that could not become
            native ``TaskConfig`` fields. Native task authoring remains strict;
            adapters preserve this data for migration/export tooling.
    """

    name: str
    source: str
    instruction: str
    manifest: EnvironmentManifest
    config: TaskConfig
    files: dict[str, Path] = field(default_factory=dict)
    generated_files: dict[str, str | bytes] = field(default_factory=dict)
    compatibility: InboundCompatibility | None = None


if TYPE_CHECKING:
    # An inbound adapter is just a class with a ``from_task_dir(Path) ->
    # InboundTask`` classmethod. No standalone Protocol: InboundTask is the
    # real contract, the adapter is its producer. ``detect_adapter`` returns
    # one of several concrete adapter classes, so the type is left open.
    InboundAdapterType = type[Any]


def detect_adapter(task_dir: Path | str) -> InboundAdapterType:
    """Return the inbound adapter whose format ``task_dir`` matches.

    Detection is by signature file: native-compatible task dirs carry a
    ``task.toml`` and Browser Use slices carry a ``browser-use-task.json``;
    Stagehand slices carry a ``stagehand-task.json``; computer-use slices
    carry a ``computer-use-task.json``. iOSWorld sources carry
    ``iosworld-task.json`` or the upstream repository signatures. Cookbook
    task dirs are a tagged ``task.toml`` variant and are detected before the
    generic native-compatible fallback.

    Raises:
        ValueError: if the directory matches no known foreign format.
    """
    # Imported here to avoid a module-load cycle: the concrete adapters
    # import InboundTask from this module.
    from benchflow.adapters.browser_use import BrowserUseAdapter
    from benchflow.adapters.computer_use import ComputerUseAdapter
    from benchflow.adapters.harbor import HarborAdapter
    from benchflow.adapters.iosworld import IOSWorldAdapter
    from benchflow.adapters.stagehand import StagehandEvalAdapter
    from benchflow.adapters.use_computer_cookbook import UseComputerCookbookAdapter

    root = Path(task_dir)
    if (root / "task.toml").is_file():
        if UseComputerCookbookAdapter.is_task_dir(root):
            return UseComputerCookbookAdapter
        return HarborAdapter
    if (root / "browser-use-task.json").is_file():
        return BrowserUseAdapter
    if (root / "stagehand-task.json").is_file():
        return StagehandEvalAdapter
    if (root / "computer-use-task.json").is_file():
        return ComputerUseAdapter
    if IOSWorldAdapter.is_task_dir(root):
        return IOSWorldAdapter
    raise ValueError(
        f"Unrecognized task format in {root}: expected 'task.toml', "
        "'browser-use-task.json', 'stagehand-task.json', "
        "'computer-use-task.json', or 'iosworld-task.json'."
    )
