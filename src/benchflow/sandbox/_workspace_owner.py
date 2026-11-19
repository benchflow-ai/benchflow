"""Give the workspace to root (freeze), or close it to kept services (close).

Runs as root with python3 -I -S while no sandbox-user process runs (killed,
or kept services frozen). One walk: it changes into each folder through a
no-follow descriptor and back through "..", checked by device and inode, so
it follows no link, resolves no long path and holds one descriptor at any
depth. Mounts are entered, as chown -R does. Each change is its own checked
syscall: the first that fails stops the walk and fails the step.

An entry that is gone when it is settled or entered (another process removed
it after its folder was listed: the Daytona daemon's finished session folders
under /root/.daytona/sessions) has nothing left to change and is skipped, as
find skips one. A name gone is also what a rename away looks like, so the
tree is then walked again: an entry moved into a folder already walked is
settled by the next walk, and a step whose every walk (WALKS) loses an entry
fails. The walks are paced (PAUSE, doubled before each further one: 7.5 s in
all) so that a burst of removals can end before the next. A name that holds
another entry when it is looked at again, a folder that moved while it was
walked and a link in a folder's place still fail it.

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
import time

UID = GID = 0  # the owner the workspace goes to
DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
WALKS = 5  # walks of the tree that may each lose an entry before the step fails
PAUSE = 0.5  # seconds before the second walk, doubled before each later one


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


def gone(name, error):
    """Whether *error* is the listed entry *name* no longer being there: it
    names that entry, and nothing holds the name now."""
    if not isinstance(error, FileNotFoundError) or error.filename not in (
        name,
        os.path.join(".", name),  # a DirEntry's own path
    ):
        return False
    try:
        os.lstat(name)
    except FileNotFoundError:
        return True
    return False


def scan(settle, lost):
    """Settle the current folder's entries; return its subfolders. One gone
    is counted in *lost* (a list of one count)."""
    with os.scandir(".") as it:
        entries = list(it)
    subs = []
    for e in entries:
        try:
            if settle(e.name, e):
                subs.append(e.name)
        except FileNotFoundError as error:
            if not gone(e.name, error):
                raise
            lost[0] += 1
    return subs


def walk(workspace, settle):
    """Settle every entry of the tree; return how many were gone."""
    lost = [0]
    os.chdir(workspace)
    settle(".", None)
    stack = [(os.lstat("."), scan(settle, lost))]
    while stack:
        subs = stack[-1][1]
        if subs:
            name = subs.pop()
            try:
                fd = os.open(name, DIR)
            except FileNotFoundError as error:
                if not gone(name, error):
                    raise
                lost[0] += 1
                continue
            try:
                os.fchdir(fd)
            finally:
                os.close(fd)
            stack.append((os.lstat("."), scan(settle, lost)))
            continue
        stack.pop()
        if stack:
            os.chdir("..")
            if not same(os.lstat("."), stack[-1][0]):
                raise OSError("a folder moved during the step")
    return lost[0]


def settle_tree(workspace, settle):
    """Walk the tree until a walk loses no entry, at most WALKS times."""
    pause = PAUSE
    for number in range(WALKS):
        if number:
            time.sleep(pause)
            pause *= 2
        if not walk(workspace, settle):
            return
    raise OSError(f"entries vanished during each of {WALKS} walks")


if __name__ == "__main__":
    try:
        settle_tree(sys.argv[2], {"freeze": freeze, "close": close}[sys.argv[1]])
    except OSError as e:
        sys.exit(f"workspace {sys.argv[1]}: {e}")
