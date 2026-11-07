"""Give the workspace to root (freeze), or close it to kept services (close).

Runs as root with python3 -I -S while no sandbox-user process runs (killed,
or kept services frozen). One walk: it changes into each folder through a
no-follow descriptor and back through "..", checked by device and inode, so
it follows no link, resolves no long path and holds one descriptor at any
depth. Mounts are entered, as chown -R does. Each change is its own checked
syscall: the first that fails stops the walk and fails the step.

freeze: every entry becomes root's, except a regular file with more than one
name, which keeps its owner and mode. A regular file whose chown is denied
is replaced by a root-owned copy; other failed changes still stop the walk.

close (after freeze): a regular file that is not root's is replaced at its
workspace name by a root-owned copy without setuid, setgid or group and other
write; the file's other names keep the old one. Group and other write go from
every other entry. Any other entry that is not root's fails the step, and
every entry the step changes is read back.

This script also runs on older image Pythons; keep it annotation-free.
"""

import os
import stat
import sys

UID = GID = 0  # the owner the workspace goes to
DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def same(a, b):
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def copy(src, dst, size):
    done = 0
    while done < size:
        try:
            n = os.sendfile(dst, src, done, size - done)
        except OSError:
            os.lseek(src, done, os.SEEK_SET)
            n = os.write(dst, os.read(src, min(size - done, 1 << 20)))
        if n == 0:
            raise OSError("file shrank while copied")
        done += n


def replace(name, st):
    """Put a root-owned copy of the regular file *name* in its place."""
    # A short temporary name, so a 255-byte name still has room beside it.
    for _ in range(100):
        tmp = ".bf" + os.urandom(4).hex()
        try:
            dst = os.open(tmp, NEW, 0o600)
            break
        except FileExistsError:
            pass
    else:
        raise OSError(f"no free temporary name beside {name!r}")
    try:
        try:
            src = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                if not same(os.fstat(src), st):
                    raise OSError(f"{name!r} changed during the step")
                copy(src, dst, st.st_size)
            finally:
                os.close(src)
            os.fchown(dst, UID, GID)
            os.fchmod(dst, stat.S_IMODE(st.st_mode) & ~0o6022)
            os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
        finally:
            os.close(dst)
        # Nothing reads the name meanwhile, and a rename over a file makes
        # ext4 write the copy out first (auto_da_alloc): ten times slower.
        os.unlink(name)
        os.rename(tmp, name)
    except BaseException:
        os.unlink(tmp)  # still there: renaming it is the last step
        raise


def freeze(name, entry):
    """Give *name* to root unless it is a regular file with another name."""
    if (
        entry is not None
        and entry.is_file(follow_symlinks=False)
        and entry.stat(follow_symlinks=False).st_nlink > 1
    ):
        return False
    try:
        os.chown(name, UID, GID, follow_symlinks=False)
    except PermissionError:
        st = os.lstat(name)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise
        # An unmapped image owner can prevent chown even for namespace root.
        # Copy instead of trusting the overflow uid as proof of ownership.
        replace(name, st)
        print(f"workspace freeze: copied unchownable file {name!r}", file=sys.stderr)
    return entry is None or entry.is_dir(follow_symlinks=False)


def close(name, entry):
    """Close *name* to the sandbox user; True when it is a folder."""
    st = os.lstat(name) if entry is None else entry.stat(follow_symlinks=False)
    if st.st_uid != UID:
        if not stat.S_ISREG(st.st_mode):
            raise OSError(f"{name!r} is not root's after the freeze")
        replace(name, st)
        bits = 0o6022
    elif st.st_mode & 0o022 and not stat.S_ISLNK(st.st_mode):
        os.chmod(name, stat.S_IMODE(st.st_mode) & ~0o022)
        bits = 0o022
    else:
        return stat.S_ISDIR(st.st_mode)
    after = os.lstat(name)
    if after.st_uid != UID or after.st_mode & bits:
        raise OSError(f"{name!r} is still open after the step")
    return stat.S_ISDIR(after.st_mode)


def scan(settle):
    """Settle the current folder's entries; return its subfolders."""
    with os.scandir(".") as it:
        entries = list(it)
    return [e.name for e in entries if settle(e.name, e)]


def walk(workspace, settle):
    os.chdir(workspace)
    settle(".", None)
    stack = [(os.lstat("."), scan(settle))]
    while stack:
        subs = stack[-1][1]
        if subs:
            fd = os.open(subs.pop(), DIR)
            try:
                os.fchdir(fd)
            finally:
                os.close(fd)
            stack.append((os.lstat("."), scan(settle)))
            continue
        stack.pop()
        if stack:
            os.chdir("..")
            if not same(os.lstat("."), stack[-1][0]):
                raise OSError("a folder moved during the step")


if __name__ == "__main__":
    try:
        walk(sys.argv[2], {"freeze": freeze, "close": close}[sys.argv[1]])
    except OSError as e:
        sys.exit(f"workspace {sys.argv[1]}: {e}")
