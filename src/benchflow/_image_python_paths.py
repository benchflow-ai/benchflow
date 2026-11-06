"""Image Python search paths, captured before the agent and later read as data.

These payloads run in the sandbox. Only the captures start with site
processing: the image and its installed packages are still trusted then.
READ runs on the isolated protected interpreter and never adds these paths
to sys.path.

Three records, all under the kit's MANIFEST:

- ``IMAGE_PYTHON_PATHS.json``: root's ``python3`` (CAPTURE).
- ``IMAGE_PYTHON_PATHS.jsonl``: one line per interpreter on the image's own
  PATH (CAPTURE_ONE), keyed by the exact path it is started by: a venv's
  ``bin/python`` is its own interpreter (its pyvenv.cfg, its site-packages,
  its .pth files), never the base it links to. The verifier's trusted PATH
  is drawn from these folders, so test.sh's interpreter is among them
  (PT-R13-A-2). The interpreters of each conda base's envs (``envs/*/bin/
  python3``) are recorded too, though off PATH: ``conda activate`` puts
  them first (PT-R14-A-1).
- ``IMAGE_PYTHON_GUARD``: the pytest guard's search path per interpreter its
  installer may meet (GUARD_SEARCH, under ``-I -S -B``), so the installer
  never starts the image's Python as root after the agent's turn
  (PT-C8-CONF-2b). Lines, not JSON: the installer reads it in sh.

Each of the first two also lists the start-up hooks on its search path, by the one
rule site and the import system follow (HOOK_RULE, shared with the clean-up
before the verifier, PT-R15-B1/B3):

- ``site``: the site folders, the only folders whose ``*.pth`` files site
  executes (``site.getsitepackages()``, a venv's included, and the user
  site). A ``.pth`` anywhere else on the search path is plain data, such as
  PyTorch weights saved as ``/app/model.pth``.
- ``hooks``: per folder on the search path, every entry named
  ``sitecustomize`` or ``usercustomize``, alone or followed by ``.`` (a
  ``.py``, a sourceless ``.pyc``, an extension module, a package folder),
  and, in a site folder, every ``*.pth``, each with its digest (a file's
  sha256; a folder's over its whole tree; a link's over its target). Key
  ``""`` is the search path entry itself when it is not a plain folder (a
  zip archive, a link).

The clean-up removes every hook that is new or changed against it and
rewrites each kept one as a fresh root-owned copy: an honest editable
install's ``.pth`` stays, wherever its venv lives, and one the agent wrote
or edited goes, wherever it is (PT-R14-B2, PT-R14-A-1, PT-R15-B2). The
folders root-owned ``.pth`` files name are on the search path already: the
capture runs after site.
"""

IMAGE_PYTHON_PATHS = "/opt/benchflow/rt/IMAGE_PYTHON_PATHS.json"
IMAGE_PYTHON_PATHS_EACH = "/opt/benchflow/rt/IMAGE_PYTHON_PATHS.jsonl"
# The pytest guard's search path per interpreter (GUARD_SEARCH), taken before
# the agent's turn so the guard installer never starts the image's Python as
# root after it (PT-C8-CONF-2b).
IMAGE_PYTHON_GUARD = "/opt/benchflow/rt/IMAGE_PYTHON_GUARD"
# The folders the guard capture reads besides PID 1's PATH: the verifier's
# PATH when the image's cannot be judged (_hardening_python.TRUSTED_SYSTEM_PATH).
GUARD_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Shell assignments put before a command: test.sh's Python search-path inputs
# as root's route keeps them as data (the
# ``_BENCHFLOW_IMAGE_*`` copies, else a preparation script's ``__bf_hook_*``
# ones, else the shell's own), handed to the capture in its environment as
# ``_BF_HOOK_<name>``, never in argv. The clean-up
# (lockdown._cleanup_site_cmd) reads the same names.
SEARCH_ENV = (
    '_BF_HOOK_PYTHONPATH="${_BENCHFLOW_IMAGE_PYTHONPATH-${__bf_hook_pythonpath-${PYTHONPATH-}}}" '
    '_BF_HOOK_PYTHONHOME="${_BENCHFLOW_IMAGE_PYTHONHOME-${__bf_hook_pythonhome-${PYTHONHOME-}}}" '
    '_BF_HOOK_PYTHONUSERBASE="${_BENCHFLOW_IMAGE_PYTHONUSERBASE-${PYTHONUSERBASE-}}" '
)
# The values back in the capture or the clean-up, from its environment.
SEARCH_VALUES = (
    "[os.environ.get('_BF_HOOK_' + n, '')"
    " for n in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE')]"
)

# What Python runs at start-up, as one rule for the record and the clean-up
# (lockdown._CLEANUP_SITE_SCRIPT). It runs on the image's interpreters at
# capture, so it keeps to Python 3.4 syntax. Nothing here follows a link,
# except to read the file a link at a hook name points to, once.
HOOK_RULE = r"""
import ast, binascii, glob, hashlib, os, stat
_BF_NB = getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0)
_BF_MAX = 64 << 20


def _bf_is_hook(name, site_folder):
    '''site imports sitecustomize and usercustomize from any folder on the
    search path, in any form the import system finds (name.py, a sourceless
    name.pyc, name.<abi>.so, a package folder), and executes *.pth only in
    a site folder.'''
    for _bf_n in ('sitecustomize', 'usercustomize'):
        if name == _bf_n or name.startswith(_bf_n + '.'):
            return True
    return bool(site_folder) and name.endswith('.pth')


def _bf_read_fd(fd):
    '''The bytes of the open regular file *fd* (at most _BF_MAX), else None.'''
    if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_size > _BF_MAX:
        return None
    _bf_chunks = []
    while True:
        _bf_chunk = os.read(fd, 1 << 20)
        if not _bf_chunk:
            return b''.join(_bf_chunks)
        _bf_chunks.append(_bf_chunk)
        if sum(len(c) for c in _bf_chunks) > _BF_MAX:
            return None


def _bf_node(name, dir_fd, depth=0):
    '''The entry *name* in *dir_fd* as read once, never following a link:
    ('f', bytes, mode), ('l', target, bytes of the regular file it points
    to or None), ('d', mode, [(name, node)]), or None (anything else, too
    deep or too big, unreadable).'''
    try:
        _bf_st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if stat.S_ISLNK(_bf_st.st_mode):
            _bf_data = None
            try:
                _bf_fd = os.open(name, os.O_RDONLY | _BF_NB, dir_fd=dir_fd)
                try:
                    _bf_data = _bf_read_fd(_bf_fd)
                finally:
                    os.close(_bf_fd)
            except OSError:
                pass
            return ('l', os.readlink(name, dir_fd=dir_fd), _bf_data)
        if stat.S_ISREG(_bf_st.st_mode):
            _bf_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | _BF_NB, dir_fd=dir_fd)
            try:
                _bf_data = _bf_read_fd(_bf_fd)
                _bf_mode = os.fstat(_bf_fd).st_mode
            finally:
                os.close(_bf_fd)
            return None if _bf_data is None else ('f', _bf_data, _bf_mode)
        if stat.S_ISDIR(_bf_st.st_mode) and depth < 16:
            _bf_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                             | getattr(os, 'O_CLOEXEC', 0), dir_fd=dir_fd)
            try:
                _bf_names = sorted(os.listdir(_bf_fd))
                if len(_bf_names) > 4096:
                    return None
                _bf_kids = [(n, _bf_node(n, _bf_fd, depth + 1)) for n in _bf_names]
                _bf_mode = os.fstat(_bf_fd).st_mode
            finally:
                os.close(_bf_fd)
            if any(k is None for _, k in _bf_kids):
                return None
            return ('d', _bf_mode, _bf_kids)
    except (OSError, ValueError):
        pass
    return None


def _bf_sha(data):
    return hashlib.sha256(data).hexdigest()


def _bf_digest(node):
    '''A file: its sha256. A link: over its target and what it points to. A
    folder: over its whole tree (sorted names, each entry's digest).'''
    if node is None:
        return None
    if node[0] == 'f':
        return _bf_sha(node[1])
    if node[0] == 'l':
        _bf_t = node[1].encode('utf-8', 'surrogateescape')
        _bf_c = b'-' if node[2] is None else _bf_sha(node[2]).encode()
        return 'link:' + _bf_sha(_bf_t + b'\0' + _bf_c)
    _bf_lines = [n.encode('utf-8', 'surrogateescape') + b'\0'
                 + _bf_digest(k).encode() + b'\n' for n, k in node[2]]
    return 'tree:' + _bf_sha(b''.join(_bf_lines))


def _bf_open_dir(path):
    '''A descriptor on the folder *path* as the import system sees it (a
    link to a folder followed), or None: a path no folder can have (an
    embedded NUL) included.'''
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_CLOEXEC', 0))
    except (OSError, ValueError):
        return None


_BF_INPUT_NAMES = ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE')


def _bf_env_inputs(values):
    '''PYTHONPATH, PYTHONHOME and PYTHONUSERBASE as test.sh's Python gets
    them, as text: *values* (the route's data copies) and PID
    1's (the image's, which the task route gives test.sh back).'''
    out = dict((n, [v] if v else []) for n, v in zip(_BF_INPUT_NAMES, values))
    for n in _BF_INPUT_NAMES:
        out.setdefault(n, [])
    try:
        with open('/proc/1/environ', 'rb') as h:
            for item in h.read().split(b'\0'):
                name, _, value = item.decode('utf-8', 'surrogateescape').partition('=')
                if name in out and value and value not in out[name]:
                    out[name].append(value)
    except OSError:
        pass
    return out


def _bf_pth_folders(d, dirs, sites):
    '''Add the folders the .pth files in the site folder *d* name, read as
    text and never run: a path line as site joins it, an import line's
    literal sys.path.append/insert argument; a literal site.addsitedir
    argument is a site folder itself.'''
    fd = _bf_open_dir(d)
    if fd is None:
        return
    try:
        names = sorted(n for n in os.listdir(fd) if n.endswith('.pth'))
        for n in names:
            try:
                pfd = os.open(n, os.O_RDONLY | _BF_NB, dir_fd=fd)
                try:
                    data = _bf_read_fd(pfd)
                finally:
                    os.close(pfd)
            except OSError:
                continue
            for line in (data or b'').decode('utf-8', 'replace').splitlines():
                # A line with a NUL names no folder: a binary file named
                # *.pth (model weights beside the packages), which site
                # also passes over (PT-R16-R2-1).
                if not line.strip() or line.startswith('#') or '\0' in line:
                    continue
                if not line.startswith(('import ', 'import\t')):
                    dirs.add(os.path.normpath(os.path.join(d, line.rstrip())))
                    continue
                try:
                    tree = ast.parse(line)
                except (SyntaxError, ValueError):
                    continue
                for call in ast.walk(tree):
                    attr = getattr(getattr(call, 'func', None), 'attr', '')
                    if attr not in ('append', 'insert', 'addsitedir'):
                        continue
                    for arg in getattr(call, 'args', []):
                        value = getattr(arg, 'value', getattr(arg, 's', None))
                        if isinstance(value, str) and os.path.isabs(value):
                            (sites if attr == 'addsitedir' else dirs).add(
                                os.path.normpath(value))
    finally:
        os.close(fd)


def _bf_search_inputs(values, cwd, sites, harvest=True):
    '''The folders test.sh's Python searches that its interpreter's own
    search path may not show: each
    PYTHONPATH entry (an empty or relative one against *cwd*, the original
    working folder), PYTHONHOME's prefix's site folders, the user site, and
    the folders the .pth files of *sites* and of those name (unless
    *harvest* is false: the clean-up reads those only from the .pth files
    it keeps). Returns (folders, site folders); only the latter run *.pth.'''
    inputs = _bf_env_inputs(values)
    dirs, found = set(), set()
    for value in inputs['PYTHONPATH']:
        for entry in value.split(':'):
            if '\0' not in entry and (cwd or os.path.isabs(entry)):
                dirs.add(os.path.normpath(os.path.join(cwd, entry)))
    # PYTHONHOME is prefix[:exec_prefix]; site takes site folders under both
    # (site.PREFIXES, PT-R16-R2-4).
    for value in inputs['PYTHONHOME']:
        for prefix in value.split(':')[:2]:
            if os.path.isabs(prefix) and '\0' not in prefix:
                for pat in ('lib/python3*/site-packages', 'lib/python3*/dist-packages'):
                    found.update(glob.glob(os.path.join(glob.escape(prefix), pat)))
    for base in inputs['PYTHONUSERBASE'] + ['/root/.local']:
        if os.path.isabs(base):
            found.update(glob.glob(os.path.join(glob.escape(base), 'lib/python3*/site-packages')))
    found = set(os.path.normpath(p) for p in found)
    if not harvest:
        return dirs - found, found
    done = set()
    while len(done) < 4096:
        todo = sorted((set(sites) | found) - done)
        if not todo:
            break
        for d in todo:
            done.add(d)
            _bf_pth_folders(d, dirs, found)
    return dirs - found, found
"""

_HOOKS = (
    HOOK_RULE
    + r"""
import json, site, sys
def _bf_entry_hooks(d, site_folders):
    '''{name: digest} of the hooks in the search path entry *d*.'''
    files = {}
    if os.path.islink(d) or (os.path.lexists(d) and not os.path.isdir(d)):
        digest = _bf_digest(_bf_node(d, None))
        if digest is not None:
            files[''] = digest
    fd = _bf_open_dir(d)
    if fd is None:
        return files
    try:
        for f in os.listdir(fd):
            if _bf_is_hook(f, d in site_folders):
                digest = _bf_digest(_bf_node(f, fd))
                if digest is not None:
                    files[f] = digest
    finally:
        os.close(fd)
    return files
def _bf_hooks(dirs, site_folders):
    return dict((d, _bf_entry_hooks(d, site_folders)) for d in dirs)
_bf_path = [os.path.abspath(p) for p in sys.path if p]
_bf_user = [site.getusersitepackages()]
# A legacy virtualenv's own site.py has no getsitepackages: its site
# folders are then the search path's site-packages and dist-packages.
if hasattr(site, 'getsitepackages'):
    _bf_site = list(site.getsitepackages())
else:
    _bf_site = [p for p in _bf_path
                if os.path.basename(p) in ('site-packages', 'dist-packages')]
_bf_site = sorted(set(os.path.abspath(p) for p in _bf_site + _bf_user))
"""
)

# argv: the record, then the working folder; test.sh's search-path inputs
# (PYTHONPATH, PYTHONHOME, PYTHONUSERBASE) come in the environment
# (SEARCH_ENV): the -I capture does not see the folders they add, which the
# verifier's Python still searches (PT-R16-B-1), so they are recorded here too.
CAPTURE = (
    _HOOKS
    + r"""
_bf_more, _bf_more_sites = _bf_search_inputs(
    """
    + SEARCH_VALUES
    + r""", (sys.argv[2:3] or [''])[0], _bf_site)
_bf_all_sites = sorted(set(_bf_site) | _bf_more_sites)
with open(sys.argv[1], 'w') as handle:
    json.dump({'path': _bf_path, 'user_site': _bf_user,
               'site': _bf_all_sites,
               'hooks': _bf_hooks(_bf_path + _bf_user
                                  + sorted(_bf_more | _bf_more_sites),
                                  _bf_all_sites)}, handle)
"""
).strip()

# argv: the record, the path the interpreter was started by.
CAPTURE_ONE = (
    _HOOKS
    + r"""
with open(sys.argv[1], 'a') as handle:
    handle.write(json.dumps({'exe': sys.argv[2], 'path': _bf_path,
                             'user_site': _bf_user,
                             'site': _bf_site,
                             'hooks': _bf_hooks(_bf_path + _bf_user, _bf_site)}) + '\n')
"""
).strip()

READ = r"""
import json, os, site, sys, sysconfig
_bf_record = '/opt/benchflow/rt/IMAGE_PYTHON_PATHS.json'
_bf_each = '/opt/benchflow/rt/IMAGE_PYTHON_PATHS.jsonl'
if os.path.isfile(_bf_record):
    with open(_bf_record) as _bf_handle:
        _bf_paths = json.load(_bf_handle)
elif os.path.isfile('/opt/benchflow/rt/RECORD'):
    raise RuntimeError('missing pre-agent Python search paths; start a fresh sandbox')
else:
    # No kit means no sandbox-user turn. Keep this fallback isolated too;
    # explicit verifier PYTHONPATH is supplied separately as data.
    _bf_paths = {'path': list(sys.path) + site.getsitepackages(),
                 'user_site': [site.getusersitepackages()],
                 'site': site.getsitepackages() + [site.getusersitepackages()]}
_bf_paths['interpreters'] = {}
if os.path.isfile(_bf_each):
    with open(_bf_each) as _bf_handle:
        for _bf_line in _bf_handle:
            _bf_entry = json.loads(_bf_line)
            if not isinstance(_bf_entry, dict) or not isinstance(_bf_entry.get('exe'), str):
                raise RuntimeError('invalid pre-agent Python search paths')
            _bf_paths['interpreters'][_bf_entry['exe']] = _bf_entry
for _bf_entry in [_bf_paths] + list(_bf_paths['interpreters'].values()):
    if not isinstance(_bf_entry, dict) or any(
        not isinstance(_bf_entry.get(k), list) or any(
            not isinstance(p, str) or not os.path.isabs(p) or '\x00' in p
            for p in _bf_entry[k]) for k in ('path', 'user_site')) or (
        'site' in _bf_entry and (not isinstance(_bf_entry['site'], list) or any(
            not isinstance(p, str) or not os.path.isabs(p) for p in _bf_entry['site']))):
        raise RuntimeError('invalid pre-agent Python search paths')
    _bf_h = _bf_entry.get('hooks', {})
    if not isinstance(_bf_h, dict) or any(
        not isinstance(d, str) or not os.path.isabs(d) or not isinstance(fs, dict)
        or any(not isinstance(n, str) or not isinstance(v, str) for n, v in fs.items())
        for d, fs in _bf_h.items()):
        raise RuntimeError('invalid pre-agent Python search paths')


def _bf_paths_for(path_value):
    '''The record of the first python3 or python on *path_value* (the
    verifier's PATH) that has one; root's python3 when none does.'''
    for _bf_dir in (path_value or '').split(':'):
        if not os.path.isabs(_bf_dir):
            continue
        for _bf_name in ('python3', 'python'):
            _bf_exe = os.path.join(_bf_dir, _bf_name)
            if _bf_exe in _bf_paths['interpreters']:
                return _bf_paths['interpreters'][_bf_exe]
            if os.path.exists(_bf_exe):
                return _bf_paths
    return _bf_paths


def _bf_recorded_hooks():
    '''Every recorded folder and the hooks it held before the agent:
    {folder: {name: digest}} (HOOK_RULE). A folder no record lists is absent.'''
    _bf_out = {}
    for _bf_entry in [_bf_paths] + list(_bf_paths['interpreters'].values()):
        for _bf_d, _bf_files in (_bf_entry.get('hooks') or {}).items():
            _bf_out.setdefault(_bf_d, {}).update(_bf_files)
    return _bf_out


def _bf_site_folders():
    '''Every folder whose *.pth files site executes for some recorded
    interpreter. A record from before PT round 16 has no 'site': each of
    its folders (search path, user site, hook folders) counts as one.'''
    _bf_out = set()
    for _bf_entry in [_bf_paths] + list(_bf_paths['interpreters'].values()):
        if 'site' in _bf_entry:
            _bf_out.update(_bf_entry['site'])
        else:
            _bf_out.update(_bf_entry.get('hooks') or {})
            _bf_out.update(_bf_entry['path'] + _bf_entry['user_site'])
    return _bf_out


def _bf_all_paths():
    '''Every folder any recorded interpreter searches, its user site included.'''
    _bf_all = list(_bf_paths['path']) + list(_bf_paths['user_site'])
    for _bf_entry in _bf_paths['interpreters'].values():
        _bf_all += _bf_entry['path'] + _bf_entry['user_site']
    return _bf_all
""".strip()


# --- The pytest guard's search path (PT-C8-CONF-2b) -------------------------

# The search path an interpreter has when it runs the tests, worked out as
# data by that interpreter under ``-I -S -B``: sets ``exe`` and ``search``
# (its standard library first, then its site or venv folders). The guard
# installer runs it (lockdown._INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT) where no
# kit exists; with a kit, only the capture before the agent's turn runs it
# (GUARD_SEARCH). It keeps to syntax
# Python 2 parses: a Python 2 on PATH reports itself and is skipped.
GUARD_SEARCH_RULE = r"""
import os, site, sys, sysconfig
# -S keeps site.main (and the .pth hooks we guard against) from running, and
# before Python 3.14 it also leaves sys.prefix at the base interpreter even
# in a venv, so neither sysconfig's default scheme nor site.getsitepackages()
# names a venv's folders (PT-R13-B1: Debian's posix_local scheme put the
# guard where the venv's pytest never looks).
exe = os.path.abspath(sys.executable)
venv = None
for folder in (os.path.dirname(exe), os.path.dirname(os.path.dirname(exe))):
    if os.path.isfile(os.path.join(folder, 'pyvenv.cfg')):  # PEP 405
        venv = folder
        break
version = 'python%d.%d' % sys.version_info[:2]
libdirs = [getattr(sys, 'platlibdir', 'lib')]
if libdirs[0] != 'lib':
    libdirs.append('lib')
stdlib = [sysconfig.get_path('stdlib'), sysconfig.get_path('platstdlib')]
if venv is None:
    sites = list(getattr(site, 'getsitepackages', list)())
else:
    sites = [os.path.join(venv, d, version, 'site-packages') for d in libdirs]
    include = False
    with open(os.path.join(venv, 'pyvenv.cfg')) as cfg:
        for line in cfg:
            key, _, value = line.partition('=')
            if key.strip().lower() == 'include-system-site-packages':
                include = value.strip().lower() == 'true'
    if include:
        sites += list(getattr(site, 'getsitepackages', list)())
search = [p for p in stdlib + sites if p]
""".strip()

# stdout: ``py3 0`` (not a Python 3: the installer skips it, as it skips one
# that does not start), else ``py3 1``, a ``dir <folder>`` line per search
# path entry in order, and ``end``. A run that stops before ``end`` (an
# entry with a newline included) is the installer's error, as before.
GUARD_SEARCH = (
    r"""
import sys
if sys.version_info[0] != 3:
    sys.stdout.write('py3 0\n')
    sys.exit(0)
sys.stdout.write('py3 1\n')
sys.stdout.flush()
"""
    + GUARD_SEARCH_RULE
    + r"""
for p in search:
    if '\n' in p:
        sys.exit('a search path entry holds a newline')
    sys.stdout.buffer.write(b'dir ' + os.fsencode(p) + b'\n')
sys.stdout.buffer.write(b'end\n')
"""
).strip()

# bf_guard_id <python>: what the record's entry for *python* holds it to, one
# ``ino:size:mtime`` each for the file it names (links followed) and the
# two pyvenv.cfg files GUARD_SEARCH_RULE reads: ``-`` for one that is not
# there, ``?`` for one that is but whose stat failed (no stat in the image or
# the kit). The same sh at the capture and at the install: an interpreter or
# venv changed since is not the one recorded, and the installer counts an
# id holding ``?`` as changed (CONF-2b round 2). No device number: an
# overlay's differs between two containers of one image, and a kit restored
# into another container must still match its record.
GUARD_ID_SH = (
    'bf_guard_id() { for bf_gf in "$1" "${1%/*}/pyvenv.cfg" "${1%/*/*}/pyvenv.cfg"; do '
    'if [ -e "$bf_gf" ] || [ -L "$bf_gf" ]; then '
    "bf_gs=$(stat -L -c '%i:%s:%Y' \"$bf_gf\" 2>/dev/null) || bf_gs='?'; "
    "else bf_gs=-; fi; "
    "printf '%s ' \"${bf_gs:-?}\"; done; }"
)

# bf_guard_dirs <PATH>: the folders the guard capture reads, into bf_gdirs
# (``:``-separated): *PATH*'s, then GUARD_SYSTEM_PATH's, in that order, each
# absolute one once, as the installer meets them. The two are joined into
# one value before the split: a shell splits only what an expansion gives,
# so in ``$path:/usr/bin`` the last folder of $path and the literal after it
# stay one word (CONF-2b round 3).
GUARD_DIRS_SH = (
    'bf_guard_dirs() { bf_gdirs=; bf_gall="$1:' + GUARD_SYSTEM_PATH + '"; '
    "IFS=:; set -f; for bf_gdir in $bf_gall; do unset IFS; set +f; "
    'case "$bf_gdir" in /opt/benchflow|/opt/benchflow/*) continue;; /*) ;; *) continue;; esac; '
    'case ":$bf_gdirs:" in *":$bf_gdir:"*) continue;; esac; '
    'bf_gdirs="$bf_gdirs${bf_gdirs:+:}$bf_gdir"; done; unset IFS; set +f; }'
)

# bf_shebang_py <script>: the interpreter a ``#!/...python*`` script names.
SHEBANG_PY_SH = (
    'bf_shebang_py() { unset IFS; bf_l=; IFS= read -r bf_l 2>/dev/null <"$1"; '
    "case \"$bf_l\" in '#!/'*) ;; *) return 0 ;; esac; set -f; set -- ${bf_l#??}; set +f; "
    'case "${1##*/}" in python*) echo "$1" ;; esac; }'
)
