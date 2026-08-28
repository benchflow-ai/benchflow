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
in there — preserved, never deleted — rather than at its canonical path.

Custody fails **closed**: this is audit-critical evidence, so a failure at any
step of the chain preserves the hold directory (nothing is ever removed with
``rmtree`` unless every held entry was confirmed moved back) and surfaces as a typed
:class:`ArtifactCustodyError` whose message names the preserved path. The one
concession to the engine's "an artifact error must never replace the real
failure" invariant is *when* the error is raised: a custody failure observed
while a child's own exception is propagating is recorded on the guard and
logged — with the preserved path — instead of raised, so the child's failure
stays the caller's diagnosis (:meth:`MountedArtifacts.hand_off` records;
:meth:`MountedArtifacts.raise_pending` and :meth:`MountedArtifacts.release`
raise it as soon as no more important exception is in flight).
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from benchflow.task.paths import RolloutPaths

logger = logging.getLogger(__name__)


class ArtifactCustodyError(RuntimeError):
    """A step of the mounted-artifact custody chain failed.

    Fail-closed marker for :class:`MountedArtifacts`: the hold directory named
    by :attr:`hold_dir` has been **preserved on disk** — whatever evidence the
    failed move left behind is in there, recoverable by hand — and the branch
    must not pretend the custody chain completed.
    """

    def __init__(self, message: str, *, hold_dir: Path) -> None:
        super().__init__(message)
        self.hold_dir = Path(hold_dir)


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


def child_artifact_roots(child_dir: Path) -> tuple[Path, ...]:
    """Rollout-shaped roots holding one child's own output, best first.

    A branch child's verifier output lands in one of two places, and *which*
    one depends on how the engine ran the child — which a report-time reader
    should not have to know:

    * a **fresh-rollout** child (every child of ``env-ready``) has a run
      directory of its own, and that run directory *is* its artifact
      directory, so its verifier output is downloaded to ``<child_dir>/verifier``;
    * an **in-place** child (``pre-verify`` / ``post-verify``) never had one.
      It wrote through the parent's bind mounts, and :meth:`MountedArtifacts.hand_off`
      archived what it wrote under ``<child_dir>/mounted/`` — same
      ``agent``/``artifacts``/``verifier`` shape, one level down.

    Both roots are :class:`~benchflow.task.paths.RolloutPaths`-shaped, so a
    reader tries them in order and takes the first that carries what it wants.
    Order is the child's *own* directory first: a fresh-rollout child can have
    both (it writes through the restored parent mounts too, and the hand-off
    archives that), and the copy it downloaded into its run directory is the
    authoritative one.

    This is the one place that knows the layout — ``bench eval ablate`` reads
    per-test outcomes through it rather than re-deriving ``mounted/`` for
    itself, so a change to :data:`CHILD_MOUNT_DIRNAME` cannot leave a second
    hard-coded path behind.
    """
    child_dir = Path(child_dir)
    return (child_dir, child_dir / CHILD_MOUNT_DIRNAME)


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

    Every step fails closed (:class:`ArtifactCustodyError`, module docstring):
    the hold directory is preserved on any failure, and the only step allowed
    to defer its raise is :meth:`hand_off`, which runs inside the child loop's
    ``finally`` where a raise would replace the child's own exception — it
    records into :attr:`custody_failures` instead, and :meth:`raise_pending`
    surfaces the record the moment no child failure is in flight.
    """

    rollout_dir: Path
    hold_dir: Path
    held: list[str] = field(default_factory=list)
    #: Custody failures observed where raising would have masked a more
    #: important exception (``hand_off`` inside the child loop's ``finally``).
    custody_failures: list[str] = field(default_factory=list)

    @classmethod
    def hold(cls, *, run_dir: Path, parent_id: str) -> MountedArtifacts:
        """Move the parent's mounted entries aside, before any child runs.

        Fails closed: a failure here means a child would overwrite the
        parent's evidence, so the branch must not proceed. Whatever was moved
        before the failure stays preserved in the hold directory — named in
        the raised :class:`ArtifactCustodyError` — and is never deleted.
        """
        guard = cls(
            rollout_dir=Path(run_dir), hold_dir=parent_hold_dir(run_dir, parent_id)
        )
        for source in mounted_artifact_dirs(guard.rollout_dir):
            try:
                if _move_entries(source, guard.hold_dir / source.name):
                    guard.held.append(source.name)
            except Exception as exc:
                message = (
                    f"branch could not hold the parent's {source} directory "
                    f"aside ({exc}); refusing to run children over the "
                    "parent's evidence. Entries already held are preserved at "
                    f"{guard.hold_dir}"
                )
                logger.error(message, exc_info=True)
                raise ArtifactCustodyError(message, hold_dir=guard.hold_dir) from exc
        return guard

    def hand_off(self, target: Path) -> None:
        """Move what the child just wrote to the mounts into ``target``.

        Runs inside the child loop's ``finally`` — a raise here would replace
        the child's own exception, so a failure is recorded on
        :attr:`custody_failures` (and logged, naming the preserved hold
        directory) for :meth:`raise_pending` / the caller to surface once the
        child's exception, if any, has been dealt with.
        """
        for source in mounted_artifact_dirs(self.rollout_dir):
            try:
                _move_entries(source, Path(target) / source.name)
            except Exception as exc:
                message = (
                    f"branch could not capture a child's {source.name} output "
                    f"under {target} ({exc}); files left in the mounts would "
                    "leak into the next child. The parent's own evidence is "
                    f"preserved at {self.hold_dir}"
                )
                self.custody_failures.append(message)
                logger.error(message, exc_info=True)

    def raise_pending(self) -> None:
        """Raise the custody failure :meth:`hand_off` had to swallow, if any.

        Called by the child loop after a child completes *without* raising —
        the point where no more important exception exists to preserve.
        """
        if self.custody_failures:
            raise ArtifactCustodyError(
                "; ".join(self.custody_failures), hold_dir=self.hold_dir
            )

    def release(self, *, raising: bool = True) -> None:
        """Put the parent's own entries back at their canonical paths.

        The hold directory is removed only after **every** held entry was
        confirmed moved back and nothing remains inside it — a directory whose
        contents were not confirmed moved is never deleted. On failure the
        hold directory is preserved and an :class:`ArtifactCustodyError`
        naming it is raised; ``raising=False`` (the caller is unwinding a
        child's own exception, which must stay the diagnosis) records and
        logs instead of raising.
        """
        failures: list[str] = []
        for source in mounted_artifact_dirs(self.rollout_dir):
            if source.name not in self.held:
                continue
            try:
                _move_entries(self.hold_dir / source.name, source)
            except Exception as exc:
                failures.append(
                    f"branch could not restore the parent's {source} directory ({exc})"
                )
                logger.error(
                    "branch could not restore the parent's %s directory from "
                    "%s — the parent's evidence is preserved there, not lost",
                    source,
                    self.hold_dir,
                    exc_info=True,
                )
        remaining = (
            sorted(
                str(entry.relative_to(self.hold_dir))
                for entry in self.hold_dir.rglob("*")
                if entry.is_symlink() or not entry.is_dir()
            )
            if self.hold_dir.is_dir()
            else []
        )
        if failures or remaining:
            if not failures:
                # Every move-back "succeeded" yet files remain: unaccounted
                # evidence. Deleting it would be exactly the fail-open rmtree
                # this class exists to prevent.
                failures.append(
                    f"unexpected entries remain after release: {remaining[:10]}"
                )
            message = (
                "; ".join(failures)
                + f" — the hold directory is preserved at {self.hold_dir}"
            )
            self.custody_failures.extend(failures)
            logger.error(message)
            if raising:
                raise ArtifactCustodyError(message, hold_dir=self.hold_dir)
            return
        shutil.rmtree(self.hold_dir, ignore_errors=True)
