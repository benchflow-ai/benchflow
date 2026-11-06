"""The kit's seal proofs as one walk on the kit's own Python (PERF-H).

:func:`benchflow._runtime_tree.kit_prepare_command`'s seal proves, in sh,
that the system loader's files and folders are root's alone, drops every
kit program whose libraries (or the folders the loader tries for them) are
not, and keeps each system folder of root's PATH only when the folder,
every folder above it, every file in it and every hop of every link in it
are root's alone (a view of its good programs otherwise). In sh that is a
``find`` or a subshell per path component: about 1.2 s a trial.

:data:`SEAL_WALK` applies the same rules in one process on the kit's
Python (``-I -S -B``, built from root-only image files before the
workspace grant; no agent has had a turn yet). The sh text is the
specification; each function names the sh it mirrors. The walk:

- decides everything before it changes anything; the loader's and the
  dynamic linker's answers come from the same commands the sh runs
  (``ldd``, ``<loader> --list``, ``LD_DEBUG=libs <loader> --list``), 16
  at a time;
- exits 3 after a ``benchflow:`` line where the sh would ``bf_die``, with
  the same text;
- exits 4 before any change when it meets something whose sh meaning it
  does not reproduce exactly (a glob or a word split it cannot mirror, a
  name with a newline): the sh seal then runs as before. Any other failure
  before its first change is treated the same way;
- creates ``.walk-applying`` in the kit before its first change; a failure
  after it is a refusal, never a second seal over half the changes;
- prints the trusted folders (``bf_trusted``) on success.

:data:`MANIFEST_WALK` writes MANIFEST's lines, byte for byte what ``find .
-type f ! -name 'MANIFEST*' -exec sha256sum {} +`` writes (find's order,
GNU sha256sum's format). A name sha256sum would
escape (a backslash, a newline) makes it fail, and the sh writes the file.
"""

from __future__ import annotations

__all__ = ["MANIFEST_WALK", "SEAL_WALK", "TOOLS_WALK"]

# The rules both walks share: bfc_dir, bfc_file, bfc_link and bfc_chain
# (benchflow._root_only.ROOT_CHAIN_FUNCS) and the sh tests they use. Only
# os, stat and sys: frozen or built in, nothing compiled at start.
_COMMON = r'''
class Defer(Exception):
    """The sh decides this one."""


def die(msg):
    sys.stderr.write('benchflow: ' + msg + '\n')
    sys.stderr.flush()
    os._exit(3)


def lst(p):
    try:
        return os.lstat(p)
    except (OSError, ValueError):
        return None


# bfc_dir / bfc_file / bfc_link: find "$1" -maxdepth 0 ... (never follows).
def bfc_dir(p, sticky=False):
    st = lst(p)
    if st is None or not stat.S_ISDIR(st.st_mode) or st.st_uid != 0:
        return False
    if sticky and st.st_mode & 0o1000:
        return True
    return not st.st_mode & 0o022


def bfc_file(p):
    st = lst(p)
    return (st is not None and not stat.S_ISDIR(st.st_mode)
            and not stat.S_ISLNK(st.st_mode) and st.st_uid == 0
            and not st.st_mode & 0o022)


def bfc_link(p):
    st = lst(p)
    return st is not None and stat.S_ISLNK(st.st_mode) and st.st_uid == 0


def is_link(p):
    st = lst(p)
    return st is not None and stat.S_ISLNK(st.st_mode)


def exists(p):
    try:
        os.stat(p)
        return True
    except (OSError, ValueError):
        return False


def is_dir(p):
    try:
        return stat.S_ISDIR(os.stat(p).st_mode)
    except (OSError, ValueError):
        return False


def is_file(p):
    try:
        return stat.S_ISREG(os.stat(p).st_mode)
    except (OSError, ValueError):
        return False


def readlink(p):
    """The exact link target, or None when it cannot be read."""
    try:
        t = os.readlink(p)
    except (OSError, ValueError):
        return None
    return t


def cd_p(t):
    """$(cd -P -- "$t" && pwd -P), or None."""
    if not is_dir(t):
        return None
    try:
        return os.path.realpath(t)
    except (OSError, ValueError):
        return None


def chain(p, hops=1):
    """bfc_chain "$p" <hops>: 0, 1 or 2, as the sh."""
    if hops > 40:
        return 1
    if not p.startswith('/'):
        return 1
    if not bfc_dir('/', True):
        return 1
    cur = ''
    rest = p[1:]
    while rest:
        if '/' in rest:
            c, rest = rest.split('/', 1)
        else:
            c, rest = rest, ''
        if c in ('', '.'):
            continue
        if c == '..':
            if '/' in cur:
                cur = cur[:cur.rfind('/')]
            continue
        nx = cur + '/' + c
        if is_link(nx):
            if not bfc_link(nx):
                return 1
            t = readlink(nx)
            if t is None:
                return 1
            if not t.startswith('/'):
                t = cur + '/' + t
            if not rest:
                return chain(t, hops + 1)
            s = chain(t, hops + 1)
            if s != 0:
                return s
            cur = cd_p(t)
            if cur is None:
                return 1
            if cur == '/':
                cur = ''
        elif rest:
            if not exists(nx):
                return 2 if bfc_dir(cur or '/') else 1
            if not (is_dir(nx) and bfc_dir(nx, True)):
                return 1
            cur = nx
        elif is_dir(nx):
            return 0 if bfc_dir(nx) else 1
        elif exists(nx):
            return 0 if bfc_file(nx) else 1
        else:
            return 2 if bfc_dir(cur or '/') else 1
    return 0
'''

# argv: the kit, then the system folders (one argument each, PATH order).
SEAL_WALK = (
    r"""
import fnmatch, os, sys, stat, re, subprocess
KIT = sys.argv[1]
FOLDERS = sys.argv[2:]
SPACE = ' \t\n\v\f\r'
"""
    + _COMMON
    + r'''
_MAGIC = re.compile(r'[*?\[]')


def sh_glob(pattern, literal_prefix='', strict=False):
    """sh's pathname expansion of literal_prefix + pattern, the prefix
    quoted: the sorted matches, or the word itself when none match.
    *strict*: the matches are expanded again later (the loader's file
    list), so one that is itself a pattern is the sh's to decide."""
    word = literal_prefix + pattern
    if not _MAGIC.search(pattern):
        return [word]
    # Bracket classes ([[:alpha:]], [[=a=]], [[.a.]], anywhere in a set, as
    # in [y[:digit:]]) and [^: Python's glob reads them differently. A lone
    # [[ (busybox's /usr/bin/[[) is a literal in both (PERF-H r3, B L-1).
    if (
        '\\' in pattern
        or '[^' in pattern
        or any(c in pattern for c in ('[:', '[=', '[.'))
        or '\n' in word
    ):
        raise Defer()
    import glob
    hits = glob.glob(glob.escape(literal_prefix) + pattern)
    if strict and any(_MAGIC.search(h) for h in hits):
        raise Defer()
    return sorted(hits, key=os.fsencode) or [word]


def sh_words(text):
    """for w in $(...): split on IFS's default (space, tab, newline), each
    word then expanded as a pattern."""
    out = []
    for w in re.split('[ \t\n]+', text):
        if w:
            out.extend(sh_glob(w))
    return out


def who(p):
    """' (owner U:G, mode M)' of *p* (links followed, else the link), as the
    sh's bf_who prints it, for a refusal's message; '' when unreadable."""
    try:
        st = os.stat(p)
    except (OSError, ValueError):
        st = lst(p)
    if st is None:
        return ''
    return ' (owner %d:%d, mode %s)' % (st.st_uid, st.st_gid, stat.filemode(st.st_mode))


def start(argv, env=None, err=False):
    """A command started with its stdout (or, *err*, its stderr) on a pipe
    and the other stream discarded; None when it cannot start (127)."""
    try:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL if err else subprocess.PIPE,
                                stderr=subprocess.PIPE if err else subprocess.DEVNULL,
                                env=env)
    except OSError:
        return None


def finish(proc, err=False):
    """(status, text) of a command :func:`start` started."""
    if proc is None:
        return 127, ''
    out, errout = proc.communicate()
    data = errout if err else out
    return proc.returncode, data.decode('utf-8', 'surrogateescape')


def first_existing(patterns):
    for pat in patterns:
        for l in sh_glob(pat):
            if exists(l):
                return l
    return ''


def walk():
    nl_check = [KIT] + FOLDERS
    if any('\n' in x for x in nl_check):
        raise Defer()
    # --- the system loader's configuration ---
    ldirs = ['/lib', '/lib64', '/lib32', '/usr/lib', '/usr/lib64', '/usr/lib32']
    for d in sh_glob('/lib/*-linux-*') + sh_glob('/usr/lib/*-linux-*'):
        if is_dir(d):
            ldirs.append(d)
    queue = []
    # bf_lq is a space-joined string re-split on IFS: a name with a blank
    # splits there.
    def enqueue(words):
        queue.extend(w for q in words for w in re.split('[ \t\n]+', q) if w)

    for pat in ('/etc/ld.so.conf', '/etc/ld.so.conf.d/*', '/etc/ld-musl-*.path'):
        enqueue(sh_glob(pat, strict=True))
    seen, files = set(), []
    while queue:
        f = queue.pop(0)
        if f in seen:
            continue
        seen.add(f)
        files.append(f)
        if len(files) > 256:
            die('the loader configuration includes too many files')
        if not is_file(f):
            continue
        base = f[:f.rfind('/')] if '/' in f else f
        with open(f, 'rb') as h:
            data = h.read().decode('utf-8', 'surrogateescape')
        if '\x00' in data:
            raise Defer()
        if fnmatch.fnmatchcase(f, '*/ld-musl-*.path'):
            for d in data.replace(':', '\n').split('\n'):
                if d.startswith('/'):
                    ldirs.append(d)
            continue
        lines = data.split('\n')
        if lines and lines[-1] == '':
            lines.pop()
        for l in lines:
            l = l.split('#', 1)[0].strip(SPACE)
            if not l:
                continue
            if l.startswith('include') and len(l) > 7 and l[7] in SPACE:
                for p in re.split('[ \t\n]+', l[7:]):
                    if not p:
                        continue
                    if not p.startswith('/'):
                        p = base + '/' + p
                    for w in re.split('[ \t\n]+', p):
                        if w:
                            enqueue(sh_glob(w, strict=True))
            elif l.startswith('hwcap') and len(l) > 5 and l[5] in SPACE:
                pass
            elif l.startswith('/'):
                ldirs.append(l)
                if '=' in l:
                    ldirs.append(l.split('=', 1)[0].rstrip(SPACE))
    for f in ['/etc/ld.so.cache', '/etc/ld.so.preload', '/etc/ld.so.conf.d'] + [
            w for x in files for w in sh_glob(x, strict=True)]:
        if chain(f) not in (0, 2):
            die('the system loader reads %s%s, which is not root\'s alone; '
                'BenchFlow will not run root steps whose libraries could change'
                % (f, who(f)))
    for d in [x for x in '\n'.join(ldirs).split('\n') if x]:
        s = chain(d)
        if s not in (0, 2):
            die('the system loader searches %s%s, which is not root\'s alone (a '
                'workspace or an image setting lets another user write it); '
                'BenchFlow will not run root steps whose libraries could change: '
                'set the task\'s workdir to a folder of its own' % (d, who(d)))
        if s != 0:
            continue
        for h in ('.', 'glibc-hwcaps', 'tls', 'x86_64', 'haswell', 'xeon_phi',
                  'avx512_1', 'i686', 'sse2', 'aarch64'):
            for p in [d + '/' + h] + sh_glob('*', d + '/' + h + '/'):
                if not is_dir(p):
                    continue
                if p.endswith('/.'):
                    p = p[:-2]
                if chain(p) != 0:
                    die('the system loader searches %s%s, which is not root\'s alone'
                        % (p, who(p)))
                try:
                    names = os.listdir(p)
                except OSError:
                    names = []
                if any('\n' in n for n in names):
                    die('the system loader searches %s (contains a name with a '
                        'newline), which is not root\'s alone; refusing it' % p)
                bad = []
                for n in names:
                    e = p + '/' + n
                    # find -L sees a link to a missing file as the link
                    # itself (GNU and busybox alike); a link that loops, or
                    # runs through a file, find reports or judges by its own
                    # rules (they differ between GNU and busybox): the sh
                    # and the image's find decide those (PERF-H r2, M-1).
                    try:
                        st = os.stat(e)
                    except FileNotFoundError:
                        st = lst(e)
                        if st is None:
                            continue
                        if '\n' in e:
                            raise Defer()
                        # A link to a missing file is judged by its hops, as
                        # a missing file is (DFX-SEAL): the sh's second find.
                        if stat.S_ISLNK(st.st_mode):
                            if chain(e) in (0, 2):
                                continue
                            bad.append(e)
                            continue
                    except (OSError, ValueError):
                        raise Defer()
                    if stat.S_ISDIR(st.st_mode):
                        continue
                    if st.st_uid != 0 or st.st_mode & 0o022:
                        if '\n' in e:
                            raise Defer()
                        die('the system loader loads %s%s, which is not root\'s '
                            'alone; BenchFlow will not run root steps whose '
                            'libraries could change' % (e, who(e)))
                if bad:
                    die('the system loader loads %s%s, which is not root\'s '
                        'alone; BenchFlow will not run root steps whose '
                        'libraries could change' % (bad[0], who(bad[0])))
                if h == '.':
                    break
    # --- every library each kit program resolves ---
    ld = first_existing(['/lib64/ld-linux-*.so.*', '/lib/ld-linux-*.so.*',
                         '/lib/ld-musl-*.so.1'])
    glibc = '/ld-linux-' in ld
    progs = []
    for x in sh_glob('*', KIT + '/libexec/'):
        if not is_file(x) or x.rsplit('/', 1)[-1] == 'python3':
            continue
        progs.append(x)

    # The same commands as the sh. On glibc one ``ldd`` names every program
    # first: it prints each program's own list under a ``<program>:`` line,
    # what ``ldd <program>`` prints alone. Used only when it succeeds and
    # every program's line is there (musl's ldd reads one file); else each
    # program gets its own ``ldd``, as in the sh. The rest, 16 at a time.
    answers = {}
    if glibc and len(progs) > 1:
        s_, out = finish(start(['ldd'] + progs))
        heads = set(x + ':' for x in progs)
        if s_ == 0 and sum(1 for line in out.split('\n') if line in heads) == len(progs):
            cur = None
            parts = dict((x, []) for x in progs)
            for line in out.split('\n'):
                if line in heads:
                    cur = line[:-1]
                elif cur is not None:
                    parts[cur].append(line)
            for x in progs:
                answers[(x, 'deps')] = '\n'.join(parts[x])
    jobs = [(x, k) for x in progs for k in ('deps', 'tried') if (x, k) not in answers]
    for i in range(0, len(jobs), 16):
        batch = jobs[i:i + 16]
        started = []
        for x, k in batch:
            if k == 'tried' and glibc:
                env = dict(os.environ)
                env['LD_DEBUG'] = 'libs'
                started.append(('err', start([ld, '--list', x], env, err=True)))
            else:
                started.append(('deps', start(['ldd', x])))
        for (x, k), (kind, proc) in zip(batch, started):
            if kind == 'err':
                answers[(x, k)] = finish(proc, err=True)[1]
                continue
            s_, out = finish(proc)
            if s_ != 0 and ld:
                out += finish(start([ld, '--list', x]))[1]
            answers[(x, k)] = out
    rx_arrow = re.compile(r'^.*=> (/[^ ]*) \(.*$')
    rx_plain = re.compile(r'^[ \t\n\v\f\r]*(/[^ ]*) \(.*$')
    rx_try = re.compile(r'^.*trying file=(/[^ ]*).*$')

    def sed_libs(text, plain=True):
        out = []
        for line in text.split('\n'):
            m = rx_arrow.match(line) or (plain and rx_plain.match(line))
            if m:
                out.append(m.group(1))
        return '\n'.join(out)

    libok, dirok, dropped = set(), set(), []
    for x in progs:
        ok = True
        for lib in sh_words(sed_libs(answers[(x, 'deps')])):
            if lib in libok:
                continue
            if chain(lib) != 0:
                ok = False
                break
            libok.add(lib)
        if ok:
            if glibc:
                found = '\n'.join(m.group(1) for m in map(rx_try.match, answers[(x, 'tried')].split('\n')) if m)
            else:
                found = sed_libs(answers[(x, 'tried')], plain=False)
            for p in sh_words(found):
                p = p[:p.rfind('/')] if '/' in p else p
                p = p or '/'
                if p in dirok:
                    continue
                if chain(p) not in (0, 2):
                    ok = False
                    break
                dirok.add(p)
        if not ok:
            dropped.append(x)
    gone = set(x.rsplit('/', 1)[-1] for x in dropped)
    refusals = []
    for l in sh_glob('*', KIT + '/bin/'):
        t = readlink(l)
        if t is not None and t.rsplit('/', 1)[-1] in gone:
            refusals.append(l)
    sh_link = KIT + '/bin/sh'
    if not is_link(sh_link) or sh_link in refusals:
        die('the image\'s /bin/sh loads a library another user can replace; '
            'BenchFlow will not run root steps with it')
    # --- the system folders root's PATH may keep ---

    def view_entries(dr):
        """bf_view "$dr": the links it would make, in order."""
        out = []
        for e in sh_glob('*', dr + '/') + sh_glob('.[!.]*', dr + '/'):
            if not exists(e) and not is_link(e):
                continue
            n = e.rsplit('/', 1)[-1]
            if not re.fullmatch(r'[A-Za-z0-9._+-]*', n):
                continue
            if is_dir(e):
                continue
            if is_link(e):
                t = readlink(e)
                if t is None:
                    continue
                if not t.startswith('/'):
                    t = dr + '/' + t
                if not (bfc_link(e) and chain(t, 1) == 0):
                    continue
            elif not bfc_file(e):
                continue
            out.append((e, n))
        return out

    trusted, good, bad, views = [], set(), set(), []
    for d in FOLDERS:
        if not is_dir(d) or chain(d) != 0:
            continue
        dr = cd_p(d)
        if dr is None:
            continue
        if '\n' in dr:
            raise Defer()
        if dr in bad:
            continue
        if dr in good:
            trusted.append(d)
            continue
        try:
            names = os.listdir(dr)
        except OSError:
            names = []
        foreign = False
        for n in names:
            e = dr + '/' + n
            # As the loader folders' check: a link to a missing file is
            # find -L's own link (skipped); any other stat error (a loop, a
            # name too long, a file on the way) the image's find judges by
            # its own rules, so the sh decides (PERF-H r3, A L-1).
            try:
                st = os.stat(e)
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                raise Defer()
            if stat.S_ISDIR(st.st_mode):
                continue
            if st.st_uid != 0 or st.st_mode & 0o022:
                foreign = True
                break
        ok = not foreign
        if ok:
            links = [dr + '/' + n for n in names if is_link(dr + '/' + n)]
            seen_t = set()
            for l in sh_words('\n'.join(links)):
                t = readlink(l)
                if t is None:
                    ok = False
                    break
                if not t.startswith('/'):
                    t = dr + '/' + t
                if t in seen_t:
                    continue
                seen_t.add(t)
                if chain(t, 1) not in (0, 2):
                    ok = False
                    break
        if ok:
            good.add(dr)
            trusted.append(d)
        else:
            bad.add(dr)
            v = KIT + '/view/' + dr.replace('/', '_')
            views.append((v, view_entries(dr)))
            trusted.append(v)
    return dropped, refusals, views, trusted


try:
    plan = walk()
except Exception:
    # Defer, or anything else before the first change: the sh decides.
    sys.exit(4)
dropped, refusals, views, trusted = plan
# --- changes, after every decision ---
os.close(os.open(KIT + '/.walk-applying', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
# Each change fails as the sh's does: silently, leaving what it left.
for x in dropped:
    name = x.rsplit('/', 1)[-1]
    for l in refusals:
        t = readlink(l)
        if t is None or t.rsplit('/', 1)[-1] != name:
            continue
        try:
            os.unlink(l)
            fd = os.open(l, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, 'w') as h:
                h.write('#!%s/bin/sh\necho "benchflow: %s loads a library another user '
                        'can replace; root does not run it" >&2\nexit 126\n'
                        % (KIT, l.rsplit('/', 1)[-1]))
            os.chmod(l, 0o755)
        except OSError:
            pass
    try:
        os.unlink(x)
    except OSError:
        pass
kept = []
for v, entries in views:
    try:
        os.makedirs(v, 0o755, exist_ok=True)
    except OSError:
        continue
    made = True
    for e, n in entries:
        try:
            os.symlink(e, v + '/' + n)
        except OSError:
            made = False
            break
    if made:
        kept.append(v)
trusted = [t for t in trusted if not t.startswith(KIT + '/view/') or t in kept]
sys.stdout.write(':'.join(trusted) + '\n')
sys.stdout.flush()
os.unlink(KIT + '/.walk-applying')
'''
).strip()

# The build's tool copies (kit_prepare_command's ``bf_tool``) on the image's
# own root-only python3, the one the build checks and copies, started -I -S -B
# before the agent's turn. argv: the kit, then the tools. For each tool: the
# first ``<folder>/<tool>`` that is a file root may run (the system folders
# first, /usr/local last), resolved as ``readlink -f`` does, kept only when
# bfc_chain passes, copied once per file under ``libexec/<path with / as _>``
# (content and permission bits, as ``cp`` without -p; the set-id bits are
# not copied) and linked from ``bin/<tool>``. Exit 4 before any change when
# a name is one the walk does not reproduce exactly; ``.tools-applying``
# marks the first change, after which a failure is a refusal.
TOOLS_WALK = (
    r"""
import os, sys, stat
KIT = sys.argv[1]
TOOLS = sys.argv[2:]
FOLDERS = ('/usr/sbin', '/usr/bin', '/sbin', '/bin', '/usr/local/sbin', '/usr/local/bin')
"""
    + _COMMON
    + r'''


def runnable(p):
    """[ -f "$p" ] && [ -x "$p" ], as root."""
    return is_file(p) and os.access(p, os.X_OK)


def plan():
    out = []
    for t in TOOLS:
        if not t or '/' in t or '\n' in t:
            raise Defer()
        p = None
        for d in FOLDERS:
            if runnable(d + '/' + t):
                p = d + '/' + t
                break
        if p is None:
            continue
        r = os.path.realpath(p)
        if '\n' in r or not exists(r):
            raise Defer()
        if chain(r) != 0:
            continue
        out.append((t, r, r.replace('/', '_')))
    # The loop copies with the image's cp from PATH: none there, and the loop
    # copies nothing. Coreutils' cp (a program named cp in /usr/bin or /bin;
    # the check goes by name and folder) drops the set-id and sticky bits;
    # another (busybox, toybox, a wrapper that runs cp -p) may keep them.
    # Then a tool with any of those bits is the loop's to copy (PERF-H r2,
    # r3: A L-2).
    cp = None
    for d in os.environ.get('PATH', '').split(':'):
        if d.startswith('/') and runnable(d + '/cp'):
            cp = os.path.realpath(d + '/cp')
            break
    if cp is None:
        raise Defer()
    coreutils = cp in ('/usr/bin/cp', '/bin/cp')
    if not coreutils and any(os.stat(r).st_mode & 0o7000 for _, r, _ in out):
        raise Defer()
    return out


try:
    steps = plan()
except Exception:
    sys.exit(4)
os.close(os.open(KIT + '/.tools-applying', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
for t, r, n in steps:
    dst = KIT + '/libexec/' + n
    if not exists(dst):
        try:
            mode = os.stat(r).st_mode & 0o777
            src = os.open(r, os.O_RDONLY)
            try:
                out = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
                try:
                    while True:
                        block = os.read(src, 1 << 20)
                        if not block:
                            break
                        while block:
                            block = block[os.write(out, block):]
                finally:
                    os.close(out)
            finally:
                os.close(src)
        except OSError:
            continue
    try:
        os.symlink('../libexec/' + n, KIT + '/bin/' + t)
    except OSError:
        die('cannot link ' + t)
os.unlink(KIT + '/.tools-applying')
'''
).strip()

# cwd: the kit. stdout: MANIFEST's text.
MANIFEST_WALK = r"""
import hashlib, os, stat, sys, fnmatch


def entries(d):
    out = []
    with os.scandir(d) as it:
        for e in it:
            out.append(e)
    return out


files = []


def walk(d):
    for e in entries(d):
        path = d + '/' + e.name
        st = os.lstat(path)
        if stat.S_ISREG(st.st_mode) and not fnmatch.fnmatchcase(e.name, 'MANIFEST*'):
            if '\\' in path or '\n' in path or '\r' in path:
                sys.exit(5)
            files.append(path)
        if stat.S_ISDIR(st.st_mode):
            walk(path)


walk('.')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


sums = [digest(p) for p in files]
out = sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout
for s, p in zip(sums, files):
    out.write((s + '  ' + p + '\n').encode('utf-8', 'surrogateescape'))
""".strip()
