"""Filesystem access for test setup/attacks, separate from controller handles.

Production callers never unwrap a capability. Tests model the sandbox's raw
filesystem access to plant malicious entries and inspect their targets.
"""

from pathlib import Path

from benchflow._utils.reachable_write import ReachablePath


def on_disk(value):
    if isinstance(value, ReachablePath):
        return value._anchor.path.joinpath(*value._parts)
    return value


def reachable(value):
    if isinstance(value, ReachablePath):
        return value
    value = Path(value)
    return ReachablePath.directory(value.parent) / value.name


def directory(value):
    return value if isinstance(value, ReachablePath) else ReachablePath.directory(value)
