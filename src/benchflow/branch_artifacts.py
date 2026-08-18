"""Keeping the parent's on-disk rollout artifacts intact across a branch.

A branch child shares the parent's sandbox — that is the whole point: it
restores the parent's container and continues *that* world. But the sandbox
bind-mounts three of the parent rollout's own directories into the container
(``DockerSandbox.__init__``: ``host_agent_logs_path`` / ``host_artifacts_path``
/ ``host_verifier_logs_path`` -> ``/logs/agent``, ``/logs/artifacts``,
``/logs/verifier``), and :meth:`DockerSandbox.restore` deliberately replays
those mounts so a restored container still writes where the host reads. So
every child writes its verifier output, its published trajectory and its
collected artifacts straight into **the parent's** directories.

The result is not a wrong number — rewards are parsed in memory and persisted
per child — but it is wrong *evidence*: after a two-arm ablation the parent's
``verifier/`` holds the second arm's ``reward.txt`` / ``test-stdout.txt`` /
``judge_result.json``, and the parent's own scoring run has left no trace. It
is destructive rather than merely confusing, because a child whose rollout
paths are its own (a fresh-rollout child) sees ``has_host_mount() == False``
and therefore runs ``clear_verifier_output_dir``, whose
``find /logs/verifier -mindepth 1 -delete`` empties the parent's directory
before the child writes anything of its own.

This module is the fix, and it is deliberately mechanical: hold the parent's
entries aside for the duration of the fork, hand each child the entries it
produced, then put the parent's back. Layout under the run directory::

    <run_dir>/
      agent/ artifacts/ verifier/          # the parent's own, restored at the end
      branches/<parent-node-id>/
        parent/{agent,artifacts,verifier}/ # transient: the parent's entries,
                                           #   held only while children run
        children/<child-node-id>/
          mounted/{agent,artifacts,verifier}/  # what THIS child wrote to the
                                               #   shared mounts

Two properties worth stating. The mount roots are never removed or recreated,
only emptied entry by entry — deleting a live bind-mount source detaches it
from the running container. And ``parent/`` is transient: it exists on disk
only between the first child starting and the last child finishing, so finding
it after a run means the branch did not complete and the parent's evidence is
in there rather than at its canonical path.

Failure-isolated by design, like the lineage writers: every operation reports
failure through the caller's logger and never propagates, because losing a copy
of the evidence must not cost the rewards the fork was run to measure.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from benchflow.task.paths import RolloutPaths

logger = logging.getLogger(__name__)

#: Directory name under ``branches/<parent-node-id>/`` holding the parent's own
#: mounted-artifact entries while its children run. Transient — see the module
#: docstring.
PARENT_HOLD_DIRNAME = "parent"

#: Directory name under a child's artifact directory holding what that child
#: wrote to the shared mounts. A dedicated subdirectory, so it can never
#: collide with the ``verifier/`` / ``agent/`` a fresh-rollout child downloads
#: into its own run directory.
CHILD_MOUNT_DIRNAME = "mounted"


def mounted_artifact_dirs(rollout_dir: Path) -> tuple[Path, ...]:
    """The rollout directories a sandbox bind-mounts into the container.

    Derived from :class:`~benchflow.task.paths.RolloutPaths` rather than
    hard-coded, because these are exactly the three paths ``DockerSandbox``
    passes as ``host_*_path`` when it builds the compose environment: a fourth
    mount added there shows up here by construction.
    """
    paths = RolloutPaths(rollout_dir=Path(rollout_dir))
    return (paths.agent_dir, paths.artifacts_dir, paths.verifier_dir)


def parent_hold_dir(run_dir: Path, parent_id: str) -> Path:
    """Where the parent's mounted entries wait out the fork."""
    return Path(run_dir) / "branches" / parent_id / PARENT_HOLD_DIRNAME


def child_mount_dir(run_dir: Path, parent_id: str, child_id: str) -> Path:
    """Where one child's mounted output is kept, under its own artifact dir."""
    from benchflow.branch_lineage import branch_child_dir

    return branch_child_dir(Path(run_dir), parent_id, child_id) / CHILD_MOUNT_DIRNAME


def _move_entries(source: Path, target: Path) -> int:
    """Move every entry of ``source`` into ``target``; return how many moved.

    ``source`` itself is left in place and empty — it is a live bind-mount
    source, and removing it would detach the mount from the running container.
    An entry already present at the destination is replaced, so a retry is
    idempotent rather than a name collision.
    """
    if not source.is_dir():
        return 0
    entries = sorted(source.iterdir())
    if not entries:
        return 0
    target.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        destination = target / entry.name
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.is_dir():
            shutil.rmtree(destination)
        shutil.move(str(entry), str(destination))
    return len(entries)


@dataclass
class MountedArtifacts:
    """Custody of the parent's mounted artifact entries during a branch.

    Created by :meth:`hold` before the first child runs. :meth:`hand_off` is
    called after each child, moving whatever is now in the mounts into that
    child's own directory — which both preserves the child's evidence and
    leaves the mounts empty, so the next child cannot inherit the previous
    one's files. :meth:`release` puts the parent's entries back at the end.
    """

    rollout_dir: Path
    hold_dir: Path
    held: list[str] = field(default_factory=list)

    @classmethod
    def hold(cls, *, run_dir: Path, parent_id: str) -> MountedArtifacts:
        """Move the parent's mounted entries aside, before any child runs."""
        guard = cls(rollout_dir=Path(run_dir), hold_dir=parent_hold_dir(run_dir, parent_id))
        for source in mounted_artifact_dirs(guard.rollout_dir):
            try:
                if _move_entries(source, guard.hold_dir / source.name):
                    guard.held.append(source.name)
            except Exception:
                logger.warning(
                    "branch could not hold the parent's %s directory aside; a "
                    "child's output may overwrite it",
                    source,
                    exc_info=True,
                )
        return guard

    def hand_off(self, target: Path) -> None:
        """Move what the child just wrote to the mounts into ``target``."""
        for source in mounted_artifact_dirs(self.rollout_dir):
            try:
                _move_entries(source, Path(target) / source.name)
            except Exception:
                logger.warning(
                    "branch could not capture a child's %s output under %s; the "
                    "child's reward is unaffected",
                    source,
                    target,
                    exc_info=True,
                )

    def release(self) -> None:
        """Put the parent's own entries back at their canonical paths."""
        for source in mounted_artifact_dirs(self.rollout_dir):
            if source.name not in self.held:
                continue
            try:
                _move_entries(self.hold_dir / source.name, source)
            except Exception:
                logger.warning(
                    "branch could not restore the parent's %s directory from "
                    "%s — the parent's evidence is still there, not lost",
                    source,
                    self.hold_dir,
                    exc_info=True,
                )
        try:
            shutil.rmtree(self.hold_dir, ignore_errors=True)
        except Exception:  # pragma: no cover - rmtree already swallows errors
            logger.warning("branch could not remove %s", self.hold_dir, exc_info=True)
