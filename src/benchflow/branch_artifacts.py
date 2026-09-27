"""Mounted-entry custody, adapted from JeremyJC67 PR #1046.

Keep live mount roots in place. A unique fork directory owns held parent entries
and each child's archive; collisions fail closed rather than overwrite evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchflow.task.paths import RolloutPaths


class ArtifactCustodyError(RuntimeError):
    """Evidence remains in the named fork directory after a failed transfer."""


def _move_entries(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if source.is_symlink() or not source.is_dir():
        raise ArtifactCustodyError(f"Unsafe artifact root: {source}")
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        destination = target / entry.name
        if destination.exists() or destination.is_symlink():
            raise ArtifactCustodyError(f"Refusing to overwrite evidence: {destination}")
        entry.rename(destination)


@dataclass
class MountedArtifacts:
    roots: tuple[Path, ...]
    fork_dir: Path

    @classmethod
    def hold(cls, paths: RolloutPaths, fork_dir: Path) -> MountedArtifacts:
        holder = cls(
            (paths.agent_dir, paths.artifacts_dir, paths.verifier_dir), fork_dir
        )
        try:
            for root in holder.roots:
                _move_entries(root, fork_dir / "parent" / root.name)
        except BaseException as original:
            try:
                for root in holder.roots:
                    _move_entries(fork_dir / "parent" / root.name, root)
            except BaseException as recovery:
                raise BaseExceptionGroup(
                    "Artifact hold and rollback failed", [original, recovery]
                ) from None
            raise
        return holder

    def hand_off(self, child_dir: Path) -> None:
        for root in self.roots:
            _move_entries(root, child_dir / "mounted" / root.name)

    def release(self) -> None:
        errors = []
        for root in self.roots:
            try:
                # Preserve any output left after a partial handoff or failed restore.
                _move_entries(root, self.fork_dir / "unclaimed" / root.name)
                _move_entries(self.fork_dir / "parent" / root.name, root)
            except BaseException as exc:
                exc.add_note(
                    f"Parent evidence remains under {self.fork_dir / 'parent'}"
                )
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup("Parent artifact restoration failed", errors)
