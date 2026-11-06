"""The system PATH's identity walk on the kit's Python (PERF-SBX).

After the agent's turn, :func:`benchflow.sandbox.lockdown._trusted_path_extras_cmd`
proves in sh that no tool on the verifier's always-kept system route leads
into a tree the agent could write: for each system folder, its own route and
the route of every executable in it, each component checked against the
captured exclusions (``bf_under``) and each link followed one hop at a time
(``bf_route`` with ``bf_identity_only=1``; ``bf_fast_system`` is the same
walk with fewer processes).

:data:`SYSTEM_ROUTE_WALK` applies the same rules in one process on the kit's
Python (``-I -S -B``, root-only image files copied before the agent's turn,
the interpreter the other post-agent root helpers run on). It only ever
*clears* the route: it prints ``clear`` and exits 0 when every route passes.
On anything else, a route that fails (exit 5) or a case whose sh meaning it
does not reproduce exactly (exit 4), the sh walks run as before and decide,
with their own message. So it may be stricter than the sh, never looser:

- every name ``os.listdir`` returns is judged (the sh globs can only skip
  names, never add one);
- "executable" is any execute bit on a regular file, followed through links
  (root's ``[ -x ]`` is that, or less on a ``noexec`` mount);
- an identity is checked against the captured index only after the
  inspector's own ``stat -c %d:%i`` and Python's ``lstat`` agree on one
  indexed file; otherwise it defers.

The sh text is the specification; each function names the sh it mirrors.
Only os, stat, sys and json; Python 3.6 syntax.
"""

from __future__ import annotations

import json

__all__ = ["CLEAR", "SYSTEM_ROUTE_WALK", "walk_spec"]

# The one line the walk prints when every route passes.
CLEAR = "clear"

SYSTEM_ROUTE_WALK = r"""
import json, os, stat, sys


def defer():
    os._exit(4)


def refuse():
    os._exit(5)


spec = json.loads(sys.argv[1])
groups = [(t.encode('utf-8'), [p.encode('utf-8') for p in ps]) for t, ps in spec['groups']]
folders = [d.encode('utf-8') for d in spec['folders']]
idents = None
if spec['index']:
    index = spec['index'].encode('utf-8')
    inspector = spec['inspector'].encode('utf-8')
    try:
        names = os.listdir(index)
    except OSError:
        defer()
    idents = set(names)
    if names:
        # The index names come from the inspector's stat; prove that its
        # %d:%i and lstat's numbers spell one file the same way first.
        probe = index + b'/' + names[0]
        try:
            st = os.lstat(probe)
            r, w = os.pipe()
            pid = os.posix_spawn(
                inspector + b'/stat',
                [b'stat', b'-c', b'%d:%i', b'--', probe],
                {b'PATH': inspector},
                file_actions=[(os.POSIX_SPAWN_DUP2, w, 1)],
            )
            os.close(w)
            out = b''
            while True:
                chunk = os.read(r, 4096)
                if not chunk:
                    break
                out += chunk
            os.close(r)
            _, status = os.waitpid(pid, 0)
        except (OSError, AttributeError, TypeError, ValueError):
            defer()
        if status != 0 or out != ('%d:%d\n' % (st.st_dev, st.st_ino)).encode():
            defer()


def lstat_or_none(p):
    try:
        return os.lstat(p)
    except (OSError, ValueError):
        return None


def stat_or_none(p):
    try:
        return os.stat(p)
    except (OSError, ValueError):
        return None


def matches(p, q):
    # PathExclusions.shell_predicate's pattern(): '/*' for /, else q|q/*.
    if q == b'/':
        return p.startswith(b'/')
    return p == q or p.startswith(q + b'/')


def under(p):
    # bf_under: the captured inode first, then the longest enclosing mount.
    if idents is not None:
        ls = lstat_or_none(p)
        st = stat_or_none(p)
        is_link = ls is not None and stat.S_ISLNK(ls.st_mode)
        if is_link or (st is not None and not stat.S_ISDIR(st.st_mode)):
            if ls is None:
                return True  # "unreadable identity"
            if ('%d:%d' % (ls.st_dev, ls.st_ino)).encode() in idents:
                return True
    for target, paths in groups:
        if matches(p, target):
            for q in paths:
                if matches(p, q):
                    return True
            return False
    return False


SPECIAL = (b'/proc', b'/dev/fd', b'/dev/stdin', b'/dev/stdout', b'/dev/stderr')


def route(rest):
    # bf_route with bf_identity_only=1: True when the whole way passes.
    if not rest.startswith(b'/'):
        return False
    if under(b'/'):
        return False
    cur = b''
    n = 0
    while rest:
        if b'/' in rest:
            c, rest = rest.split(b'/', 1)
        else:
            c, rest = rest, b''
        if c in (b'', b'.'):
            continue
        if c == b'..':
            i = cur.rfind(b'/')
            if i >= 0:
                cur = cur[:i]
            continue
        p = cur + b'/' + c
        if under(p):
            return False
        if p in SPECIAL or p.startswith(b'/proc/') or p.startswith(b'/dev/fd/'):
            return False
        ls = lstat_or_none(p)
        if ls is not None and stat.S_ISLNK(ls.st_mode):
            n += 1
            if n > 40:
                return False
            try:
                t = os.readlink(p)
            except (OSError, ValueError):
                return False
            if not t:
                return False
            if t.startswith(b'/'):
                cur = b''
            rest = t + (b'/' + rest if rest else b'')
            continue
        if stat_or_none(p) is None and rest:
            return False
        cur = p
    return True


for d in folders:
    # The system check: [ -d "$d" ] || continue; bf_judge "$d".
    st = stat_or_none(d)
    if st is None or not stat.S_ISDIR(st.st_mode):
        continue
    if not route(d):
        refuse()
    try:
        entries = os.listdir(d)
    except OSError:
        defer()
    for name in entries:
        e = d + b'/' + name
        st = stat_or_none(e)
        # [ -f "$e" ] && [ -x "$e" ] || continue, as root, never narrower.
        if st is None or not stat.S_ISREG(st.st_mode) or not st.st_mode & 0o111:
            continue
        if not route(e):
            refuse()
sys.stdout.write('clear\n')
sys.stdout.flush()
"""


def walk_spec(
    groups: list[tuple[str, list[str]]],
    folders: tuple[str, ...],
    *,
    index: str,
    inspector: str,
) -> str | None:
    """The walk's one argument, or None when its text cannot carry the
    exclusions byte for byte (then the sh walk runs).

    *groups* is :meth:`PathExclusions.predicate_groups`, in the order
    ``bf_under`` tries them; *index* the captured inode index ('' when no
    inode was captured); *inspector* the pre-agent tool folder.
    """
    texts = [t for t, _ in groups] + [p for _, ps in groups for p in ps]
    texts += [*folders, index, inspector]
    for text in texts:
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return None
        if "\0" in text:
            return None
    if index and not inspector:
        return None
    return json.dumps(
        {
            "groups": [[t, list(ps)] for t, ps in groups],
            "folders": list(folders),
            "index": index,
            "inspector": inspector,
        }
    )
