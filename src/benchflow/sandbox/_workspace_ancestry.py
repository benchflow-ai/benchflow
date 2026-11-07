"""Refuse kept services when they could replace the verifier's workspace path.

Runs as root with python3 -I -S, after workspace ownership and mode closure,
while sandbox-user processes are stopped. Resolve one component at a time:
realpath alone loses directories used by intermediate symlink targets. Every
directory entry used by the kernel must be fixed against non-root writers.
This script also runs on older image Pythons; keep it annotation-free.
"""

import collections
import os
import stat
import sys


def check(workspace):
    if not workspace.startswith("/"):
        raise ValueError("workspace must be absolute")
    pending = collections.deque(workspace.split("/"))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open("/", flags)
    links = 0
    try:
        while pending:
            name = pending.popleft()
            if not name or name == ".":
                continue
            parent = os.fstat(directory)
            child = os.stat(name, dir_fd=directory, follow_symlinks=False)
            # An owner can chmod its directory back to writable. A root-owned
            # sticky directory (/tmp) protects only entries also owned by root.
            if parent.st_uid != 0 or (
                parent.st_mode & 0o022
                and not (parent.st_mode & stat.S_ISVTX and child.st_uid == 0)
            ):
                raise ValueError("replaceable workspace ancestor")
            if stat.S_ISLNK(child.st_mode):
                links += 1
                if links > 40:
                    raise ValueError("too many workspace symlinks")
                target = os.readlink(name, dir_fd=directory)
                if target.startswith("/"):
                    root = os.open("/", flags)
                    os.close(directory)
                    directory = root
                pending.extendleft(reversed(target.split("/")))
            else:
                opened = os.open(name, flags, dir_fd=directory)
                os.close(directory)
                directory = opened
        leaf = os.fstat(directory)
        if leaf.st_uid != 0 or leaf.st_mode & 0o022:
            raise ValueError("writable workspace root")
    finally:
        os.close(directory)


if __name__ == "__main__":
    try:
        check(sys.argv[1])
    except (OSError, ValueError) as error:
        sys.exit("Cannot protect workspace resolution: " + str(error))
