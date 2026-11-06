"""The restore boundary of task folders and finished trials, on the host.

:func:`benchflow.embodied.spec.restore_boundary` reads what a task's ``metadata.embodied`` declares. Operations on a
finished trial (``benchflow continue``, ``bench eval regrade``) also have the trial's own evidence: an embodied trial
keeps its episode record, whose embodiment names the mode it ran on, and that evidence overrules the declaration of a
task that claims less than the trial shows.
"""

from __future__ import annotations

import json
from pathlib import Path

from .spec import MODES, PHYSICAL_MODES, RestoreBoundary, restore_boundary

#: The episode records an embodied trial keeps: the verifier's copy and the training export's.
EPISODE_RECORDS = ("verifier/episode/episode.json", "trainer/embodied_episode.json")


def task_dir_restore_boundary(task_dir: Path) -> tuple[Path, RestoreBoundary]:
    """A task folder's native package and the restore boundary its metadata declares.

    A folder in a registered task format is materialized first. Raises when the task cannot be loaded, or
    ``SpecError`` when its declaration is invalid.
    """
    from benchflow.task.formats import materialize_task_dir
    from benchflow.task.task import Task

    native = materialize_task_dir(task_dir)
    return native, restore_boundary(Task(native).config.metadata)


def recorded_modes(trial_dir: Path) -> list[str]:
    """The embodiment modes a trial's episode records name; ``sim`` for a record that cannot be read."""
    modes = []
    for name in EPISODE_RECORDS:
        record = Path(trial_dir) / name
        if not record.is_file():
            continue
        try:
            mode = json.loads(record.read_text())["embodiment"].get("mode", "sim")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            mode = "sim"
        modes.append(mode if mode in MODES else "sim")
    return modes


def trial_restore_boundary(
    trial_dir: Path, declared: RestoreBoundary | None = None
) -> RestoreBoundary:
    """What software may do to a finished trial's world: the task's declaration, overruled by the trial's evidence.

    An episode record that names a ``real`` or ``hil-mock`` embodiment wins over any declaration. Any other record
    (an unreadable one included) marks the trial embodied, keeping only what an embodied declaration grants. Without
    a record, the declaration stands (a software task when there is none).
    """
    declared = declared or RestoreBoundary()
    modes = recorded_modes(trial_dir)
    physical = next((mode for mode in modes if mode in PHYSICAL_MODES), None)
    if physical is not None:
        return RestoreBoundary(
            embodied=True, mode=physical, world_restore=False, action_replay=False
        )
    if not modes or declared.embodied:
        return declared
    return RestoreBoundary(
        embodied=True, mode="sim", world_restore=False, action_replay=False
    )
