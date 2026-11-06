"""Native evidence uses the same held capabilities as ACP evidence."""

import os

from benchflow._utils.reachable_write import ReachableLog, ReachablePath


def read_log(path: ReachablePath) -> str:
    """Read reconnect history only from an unaliased regular log inode."""
    with path.open() as source:
        info = os.fstat(source.fileno())
        if info.st_nlink != 1:
            raise OSError("Native history must be an unaliased regular file")
        return source.read(info.st_size)


def fresh_log(
    path: ReachablePath, *, append: bool = False, initial: str = ""
) -> ReachableLog:
    """Keep the writer private, publishing a new inode on each flush."""
    return path.log(append=append, initial=initial)
