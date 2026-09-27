"""What a task acts on, and the physical-reset boundary that follows from it.

BenchFlow can roll back only state it owns: declared environment state, a
container snapshot, a workspace tarball, a simulator checkpoint, or a recorded
LLM session replayed into a fresh sandbox. A physical embodiment's arm pose and
scene are not such state. Restoring a container cannot put a block back in a
cup, and replaying a recorded session re-sends its motion commands to real
hardware. Branching and recovery therefore refuse physical embodiments before
touching anything; trying again means a new episode after an operator-qualified
physical reset, never a restore.

A task declares its embodiment in task metadata::

    metadata:
      embodiment: physical          # or: simulated | virtual (default)

or as a mapping that also states the software capabilities explicitly::

    metadata:
      embodiment:
        kind: simulated
        world_restore: false        # simulator has no checkpoint support yet
        action_replay: false

A physical declaration can never claim either capability. Task packages built
before this key existed are recognised by a ``physical`` metadata tag, so the
boundary fails closed for them too.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

EmbodimentKind = Literal["virtual", "simulated", "physical"]
EMBODIMENT_KINDS: frozenset[str] = frozenset({"virtual", "simulated", "physical"})

#: Embodied trial record written next to a trial's normal BenchFlow artifacts.
TRIAL_RECORD_FILENAME = "trial-record.json"
# ``benchflow.robotics`` manifests written before the trial record existed.
_LEGACY_PHYSICAL_MANIFEST_KINDS = frozenset(
    {"physical_trial", "agent_probe", "read_only_smoke"}
)


@dataclass(frozen=True)
class Embodiment:
    """The kind of world a task acts on and what software can do to it.

    ``world_restore`` means a software restore genuinely returns the task world
    to the checkpointed state. ``action_replay`` means re-executing recorded
    actions only affects that restorable world. Both are always false for a
    physical embodiment.
    """

    kind: EmbodimentKind = "virtual"
    world_restore: bool = True
    action_replay: bool = True
    source: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in EMBODIMENT_KINDS:
            raise ValueError(
                f"embodiment kind must be one of {sorted(EMBODIMENT_KINDS)}, "
                f"got {self.kind!r}"
            )
        if self.kind == "physical" and (self.world_restore or self.action_replay):
            raise ValueError(
                "a physical embodiment cannot declare world_restore or "
                "action_replay: software restore does not reset a real arm or "
                "scene, and replay would repeat real motion"
            )

    @property
    def physical(self) -> bool:
        return self.kind == "physical"

    def restoration_record(self) -> dict[str, Any]:
        """Evidence-class fields for lineage and trial records."""
        return {
            "kind": self.kind,
            "world_restore": self.world_restore,
            "action_replay": self.action_replay,
            "physical_state": "not_restorable" if self.physical else "not_applicable",
            "retry_requires": "new_qualified_episode" if self.physical else None,
        }


class PhysicalRestoreRefused(RuntimeError):
    """A branch or recovery would have claimed to restore or replay a world it cannot.

    Nothing was checkpointed, restored, or replayed before this was raised.
    """

    def __init__(self, operation: str, embodiment: Embodiment) -> None:
        self.operation = operation
        self.embodiment = embodiment
        if embodiment.physical:
            reason = (
                "the task acts on a physical embodiment. Restoring software "
                "state does not reset the arm or scene, and recovery never "
                "replays physical actions. Start a new episode after an "
                "operator-qualified physical reset instead"
            )
        else:
            reason = (
                f"the task's {embodiment.kind} embodiment declares "
                f"world_restore={embodiment.world_restore} and "
                f"action_replay={embodiment.action_replay}"
            )
        super().__init__(f"{operation} refused: {reason}.")


def embodiment_from_metadata(metadata: Mapping[str, Any] | None) -> Embodiment:
    """Resolve a task's embodiment from its metadata mapping.

    An explicit ``embodiment`` key wins. Without one, a ``physical`` tag marks
    a physical task. Anything else is virtual. Invalid declarations raise
    ``ValueError`` rather than silently defaulting to a restorable world.
    """
    if not metadata:
        return Embodiment()
    declared = metadata.get("embodiment")
    if declared is None:
        tags = metadata.get("tags")
        if isinstance(tags, list | tuple) and "physical" in tags:
            return Embodiment(
                kind="physical",
                world_restore=False,
                action_replay=False,
                source="tags",
            )
        return Embodiment()
    if isinstance(declared, str):
        declared = {"kind": declared}
    if not isinstance(declared, Mapping):
        raise ValueError(
            "metadata.embodiment must be a kind string or a mapping with 'kind'"
        )
    kind = declared.get("kind")
    if not isinstance(kind, str) or kind not in EMBODIMENT_KINDS:
        raise ValueError(
            f"metadata.embodiment.kind must be one of {sorted(EMBODIMENT_KINDS)}, "
            f"got {kind!r}"
        )
    unknown = set(declared) - {"kind", "world_restore", "action_replay"}
    if unknown:
        raise ValueError(f"unknown metadata.embodiment keys: {sorted(unknown)}")
    physical = kind == "physical"
    flags = {}
    for name in ("world_restore", "action_replay"):
        value = declared.get(name, not physical)
        if not isinstance(value, bool):
            raise ValueError(f"metadata.embodiment.{name} must be true or false")
        flags[name] = value
    return Embodiment(
        kind=cast(EmbodimentKind, kind),
        world_restore=flags["world_restore"],
        action_replay=flags["action_replay"],
        source="task_metadata",
    )


def task_embodiment(task: Any) -> Embodiment:
    """Embodiment of a loaded task object (``task.config.metadata``), if any."""
    metadata = getattr(getattr(task, "config", None), "metadata", None)
    return embodiment_from_metadata(metadata if isinstance(metadata, Mapping) else None)


def require_world_restore(embodiment: Embodiment, operation: str) -> None:
    """Refuse an operation whose correctness depends on restoring the world."""
    if not embodiment.world_restore:
        raise PhysicalRestoreRefused(operation, embodiment)


def require_action_replay(embodiment: Embodiment, operation: str) -> None:
    """Refuse an operation that re-executes recorded agent actions."""
    if not embodiment.action_replay:
        raise PhysicalRestoreRefused(operation, embodiment)


def _read_json_object(path: Path, *, strict: bool) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        if not strict:
            return None
        raise ValueError(f"unreadable embodiment evidence {path}: {exc}") from exc
    return data if isinstance(data, dict) else None


def recorded_embodiment(directory: Path, *, max_depth: int = 4) -> Embodiment | None:
    """Embodiment recorded by a trial enclosing ``directory``, if any.

    Checks ``directory`` and up to ``max_depth`` ancestors for a trial record
    (``trial-record.json``) or a legacy ``benchflow.robotics`` manifest, so a
    BenchFlow rollout nested inside a physical trial is recognised as physical.
    An unreadable record raises ``ValueError``: missing evidence is not
    evidence of a restorable world.
    """
    directory = Path(directory)
    for candidate in [directory, *list(directory.parents)[:max_depth]]:
        record = _read_json_object(candidate / TRIAL_RECORD_FILENAME, strict=True)
        if record is not None:
            block = record.get("embodiment")
            if isinstance(block, str):
                block = {"kind": block}
            if not isinstance(block, Mapping):
                raise ValueError(
                    f"{candidate / TRIAL_RECORD_FILENAME} has no embodiment block"
                )
            declared = {
                key: block[key]
                for key in ("kind", "world_restore", "action_replay")
                if key in block
            }
            resolved = embodiment_from_metadata({"embodiment": declared})
            return Embodiment(
                kind=resolved.kind,
                world_restore=resolved.world_restore,
                action_replay=resolved.action_replay,
                source="trial_record",
            )
        manifest = _read_json_object(candidate / "manifest.json", strict=False)
        if (
            manifest is not None
            and manifest.get("kind") in _LEGACY_PHYSICAL_MANIFEST_KINDS
            and "reset_id" in manifest
        ):
            return Embodiment(
                kind="physical",
                world_restore=False,
                action_replay=False,
                source="trial_manifest",
            )
    return None
