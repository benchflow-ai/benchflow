"""The image manifest: the sha256 of the image's own conftest.py and pytest
configuration files, wherever they sit.

It is recorded as root in a sandbox where no agent process has run (before
the agent's turn; in a separate verifier sandbox, before the transfer), read
back to the host at once, and decided from there: the host's copy is the one
that counts, so nothing an agent does to the sandbox's copy changes it. Before
verification the host writes its copy back to :data:`MANIFEST_PATH`, root's
alone, for the readers in the sandbox (the conftest sweep's shell form and the
pytest plugin guard).

The conftest sweep keeps a conftest.py whose bytes match the manifest at the
same absolute path, inside or outside the workspace, and removes every other
one. Without a manifest (no Python to record it with, a walk that failed, a
copy that cannot be read) everything behaves as before it existed.

It is also the detector's baseline (the faithful default): after the
turn the same script walks the sandbox again (:func:`walk_after`), and the
host diffs the two (:func:`detector_record`). What changed is written to the
trial's ``integrity/verifier-inputs.json``: the files a test run reads as
hooks (conftest.py, pytest configuration, start-up modules, bytecode without
a source, pytest plugin modules) as flags, everything else as changes. A
flag never changes the reward.
"""

from __future__ import annotations

import bisect
import json
import logging
import posixpath
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from benchflow._root_python import root_python_sh
from benchflow._runtime_tree import KIT_PYTHON, RUNTIME_ROOT, VERIFIER_DIR
from benchflow.sandbox import _image_trees

logger = logging.getLogger(__name__)

# Its own folder: /var/lib/benchflow holds the verifier env on some lines.
MANIFEST_DIR = "/var/lib/benchflow-image"
MANIFEST_PATH = f"{MANIFEST_DIR}/image-manifest.json"
VERSION = 1
# What it records: the files pytest reads as configuration or conftests, and
# every Python source and byte-code file (the plugin guard's).
NAMES = (
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
)
SUFFIXES = (".py", ".pyc")
# Left out: the kernel's virtual filesystems, BenchFlow's own runtime and the
# manifest's folder, the run's logs, temporary and cache folders, and the
# Python installation trees (site-packages, dist-packages, a lib/pythonX.Y
# standard library, conda's package cache), which the plugin guard trusts by
# its own rule and its RECORD rule keeps a package's conftest.py in. A
# file left out is never the image's by this manifest: leaving a tree out only
# takes trust away. A mounted workspace is walked like any folder.
PRUNED = (
    "/proc",
    "/sys",
    "/dev",
    VERIFIER_DIR,
    MANIFEST_DIR,
    "/logs",
    "/tmp",
    "/root/.cache",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
# The entries both walks may walk through folder links, in all: a link past
# it is kept unwalked, which flags it.
LINK_CAP = 200_000
# A kept link's target is hashed only when it is a regular file this size or
# smaller; anything else is kept by its type.
LINK_TARGET_CAP = 64 << 20

# argv: the names, the suffixes and the pruned paths (JSON each), when it
# was taken, where it goes ("" for nowhere: the walk after the turn), and
# the Python trees' and PATH tools' options (JSON; _image_trees). Prints
# the manifest as one JSON document and writes the same bytes to the
# manifest's file, root's alone. Regular files only, opened without following
# a link; a folder it cannot read is skipped, so a file there is not the
# image's.
_RECORD_SCRIPT = (
    r"""
import hashlib, json, os, re, stat, sys
names = set(json.loads(sys.argv[1]))
manifest_names = frozenset(names)
suffixes = tuple(json.loads(sys.argv[2]))
pruned = set(json.loads(sys.argv[3]))
taken, target = sys.argv[4], sys.argv[5]
options = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
stdlib = re.compile(r"python[0-9][0-9.]*$")
# After the turn, the Python trees the walk before found: a folder named like
# one that wasn't one then is walked like any other.
known_roots = set(options["roots"]) if isinstance(options.get("roots"), list) else None
link_cap = int(options.get("link_cap", 200000))
# BenchFlow's runtime tree, root's alone: a Python tree installed there after
# the walk before (the model proxy's managed CPython, agent tools) stays one.
runtime = (options.get("runtime_root") or "").rstrip("/")
# The names pytest before 8.3 looks for in a prefix's bin/ to take it as a venv.
ACTIVATES = {"activate", "activate.csh", "activate.fish", "Activate", "Activate.bat", "Activate.ps1"}
target_cap = int(options.get("target_cap", 64 << 20))
sectioned = {"pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml", "tox.ini", "setup.cfg", "pyproject.toml"}
def pytest_section(name, data):
    # The text pytest reads its options from: a pytest file whole, else its
    # pytest section(s), without blank or comment lines; "" without one.
    if name in ("pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml"):
        return hashlib.sha256(data).hexdigest()
    wanted = {"tox.ini": ("pytest",), "setup.cfg": ("tool:pytest",),
              "pyproject.toml": ("tool.pytest.ini_options", "tool.pytest")}[name]
    kept, inside = [], False
    for line in data.decode("utf-8", "replace").splitlines():
        bare = line.strip()
        if bare.startswith("[") and bare.endswith("]"):
            head = bare.strip("[]")
            if name == "pyproject.toml":
                head = head.replace(" ", "")
                inside = head in wanted or head.startswith("tool.pytest.")
            else:
                inside = head in wanted
            if inside:
                kept.append("[" + head + "]")
            continue
        if inside and bare and bare[:1] not in "#;":
            kept.append(line.rstrip())
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest() if kept else ""
sites = []
# The folders the files walk leaves to the trees section (and links named
# like one), as python_named decides.
roots = []
def by_name(name, parent):
    # A Python tree by its name: a site folder, a lib/pythonX.Y standard
    # library, conda's package cache.
    if name in ("site-packages", "dist-packages"):
        return True
    if stdlib.match(name) and os.path.basename(parent) == "lib":
        return True
    return name == "pkgs" and os.path.isdir(os.path.join(parent, "conda-meta"))
def in_runtime(path):
    return bool(runtime) and path.startswith(runtime + "/")
def interpreter_tree(name, parent):
    # The prefix of a folder named like a Python tree that the walk before
    # didn't find, when it sits under a prefix every pytest skips: a venv
    # (pyvenv.cfg, which pytest 8.3 and later look for, and an activate
    # script in bin/, which earlier ones do) at its own prefix, or a user
    # site (<home>/.local, a dot folder); "" for any other. A conda env made
    # in the turn is walked like any folder (pytest before 8.3 collects it).
    if name in ("site-packages", "dist-packages"):
        lib = os.path.dirname(parent)
        if not (stdlib.match(os.path.basename(parent)) and os.path.basename(lib) == "lib"):
            return ""
        prefix = os.path.dirname(lib)
    elif stdlib.match(name) and os.path.basename(parent) == "lib":
        prefix = os.path.dirname(parent)
    else:
        return ""
    if os.path.isfile(os.path.join(prefix, "pyvenv.cfg")):
        try:
            if ACTIVATES & set(os.listdir(os.path.join(prefix, "bin"))):
                return prefix
        except OSError:
            pass
    home = os.path.dirname(prefix)
    if os.path.basename(prefix) == ".local" and (home == "/root" or os.path.dirname(home) == "/home"):
        return prefix
    return ""
def python_named(path, name, parent):
    # Which folders the files walk leaves to the trees section: by name
    # before the turn; after it, a tree found then, one in BenchFlow's
    # runtime tree, or one made in the turn under a venv or user site. Any
    # other is walked like any folder, so a conftest.py planted in
    # /app/tests/site-packages is seen.
    if known_roots is not None:
        if path in known_roots:
            return True
        if in_runtime(path):
            return by_name(name, parent)
        return bool(interpreter_tree(name, parent))
    return by_name(name, parent)
# The trees made in the turn under a venv prefix: their conftest.py
# and pytest configuration names are kept with the prefix, which the host
# drops unless the prefix held a file the walk before recorded (a venv made
# in the turn holds none; /app/tests made into one does).
new_trees = []
def left_out(path, name, parent):
    if path in pruned:
        return True
    if not python_named(path, name, parent):
        return False
    roots.append(path)
    if known_roots is not None and path not in known_roots and not in_runtime(path):
        prefix = interpreter_tree(name, parent)
        # A user site is a dot folder pytest never enters, whatever it held.
        if not prefix.endswith("/.local"):
            new_trees.append((path, prefix))
    if name in ("site-packages", "dist-packages"):
        sites.append(path)
    elif name != "pkgs":
        sites.extend(os.path.join(path, sub) for sub in ("site-packages", "dist-packages"))
    return True
KINDS = {stat.S_IFIFO: "fifo", stat.S_IFCHR: "char", stat.S_IFBLK: "block", stat.S_IFSOCK: "socket"}
def resolved(path):
    # What a kept link points at: a regular file's sha256 (within the cap),
    # else its type, never read: a FIFO or a device must not stall the walk.
    try:
        st = os.stat(path)
    except OSError:
        return "dangling"
    if stat.S_ISDIR(st.st_mode):
        return "folder"
    if not stat.S_ISREG(st.st_mode):
        return KINDS.get(stat.S_IFMT(st.st_mode), "other") + ":not-hashed"
    if st.st_size > target_cap:
        # Not read: what the kernel keeps of it, which no task can set back
        # but the ctime a snapshot moves.
        return "regular:%d:not-hashed:%d:%d:%d:%d:%d:%o" % (
            st.st_size, st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns,
            st.st_uid, st.st_mode & 0o7777)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return "unreadable"
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return "other:not-hashed"
        data = handle.read(target_cap)
    return "sha256:" + hashlib.sha256(data).hexdigest()
class OverCap(Exception):
    pass
files = {}
folder_links = []
def walk(top, via=None, into=files, links=folder_links, budget=None, tree=None):
    # *via*: the folder link this walk reached *top* through (its files are
    # recorded by their path through it); *budget*: entries it may still
    # walk (OverCap past it); *tree*: the interpreter prefix of a tree made in
    # the turn (only conftest.py and pytest configuration names, kept with it).
    try:
        listing = os.scandir(top)
    except OSError:
        return
    with listing:
        for entry in listing:
            if budget is not None:
                budget[0] -= 1
                if budget[0] < 0:
                    raise OverCap()
            path = entry.path
            try:
                if entry.is_dir(follow_symlinks=False):
                    if via is not None or tree is not None or not left_out(path, entry.name, top):
                        walk(path, via, into, links, budget, tree)
                    continue
                if tree is None and entry.is_symlink() and entry.is_dir():
                    if via is not None:
                        links.append(path)
                    elif path not in pruned:
                        if python_named(path, entry.name, top):
                            roots.append(path)
                        else:
                            links.append(path)
                wanted = entry.name in names or (tree is None and entry.name.endswith(suffixes))
                if not wanted:
                    continue
                if entry.is_symlink():
                    # pytest and Python read through a link: it is kept by its
                    # target and what that holds, so one planted, retargeted or
                    # whose target changed in the turn differs.
                    link = os.readlink(path)
                    into[path] = {"sha256": "link:" + link, "size": 0, "link": link,
                                  "target": resolved(path)}
                    if via is not None:
                        into[path]["via"] = via
                    if tree is not None:
                        into[path]["tree"] = tree
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError:
                continue
            with os.fdopen(fd, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    continue
                digest, size, text = hashlib.sha256(), 0, []
                while True:
                    chunk = handle.read(1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    if entry.name in sectioned and size <= (8 << 20):
                        text.append(chunk)
            into[path] = {"sha256": digest.hexdigest(), "size": size}
            if entry.name in sectioned:
                into[path]["pytest"] = pytest_section(entry.name, b"".join(text))
            if via is not None:
                into[path]["via"] = via
            if tree is not None:
                into[path]["tree"] = tree
def under(path, tops):
    return any(path == t or path.startswith(t.rstrip("/") + "/") for t in tops)
def recorded(real):
    # Whether the walk by real path records (and flags) what is at *real*:
    # not in a pruned folder nor in a Python tree (the trees section records
    # those, and flags no conftest.py there).
    if under(real, pruned):
        return False
    parts = real.split("/")
    for i in range(1, len(parts)):
        parent = "/".join(parts[:i]) or "/"
        if python_named("/".join(parts[: i + 1]), parts[i], parent):
            return False
    return True
def root_only(path):
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return st.st_uid == 0 and not st.st_mode & 0o022
def exposed(tree):
    # A Python tree the files walk leaves to the trees section (which flags no
    # conftest.py), that pytest would collect in through a link to a folder
    # above it: one made in the turn, or one the agent user could write (its
    # folder, or its own site-packages or dist-packages, not root's alone: an
    # image that lets the agent pip install by giving it the site). BenchFlow's
    # runtime tree is root's alone.
    if in_runtime(tree):
        return False
    if known_roots is not None and tree not in known_roots:
        return True
    if not root_only(tree):
        return True
    for site in ("site-packages", "dist-packages"):
        below = os.path.join(tree, site)
        if os.path.lexists(below) and not root_only(below):
            return True
    return False
def hides_tree(real):
    # Whether an exposed Python tree is at or below *real*.
    start = real.rstrip("/") + "/"
    return any((r == real or r.startswith(start)) and exposed(r) for r in set(roots))
def unwalked(link, why, via):
    entry = dict(files.get(link) or {})
    if not entry:
        try:
            text = os.readlink(link)
        except OSError:
            text = "?"
        entry = {"sha256": "link:" + text, "size": 0, "link": text, "target": "folder"}
    entry["unwalked"] = why
    if via is not None:
        entry["via"] = via
    files[link] = entry
def follow_links():
    # Each folder link once, by its link path (pytest collects through one),
    # after the walk by real path: skipped when the walk already records its
    # target by its real path (the kernel's target is the textual real path
    # and lies in a walked tree) and no exposed Python tree is below it
    # (merged /usr's /lib -> usr/lib), or when the target is in /proc or
    # /sys, where nothing can be planted. Each real folder is walked once (a cycle
    # stops) and nested links are followed in turn, depth first, in path
    # order, up to link_cap entries in all. A link left unwalked (past the
    # cap, or a folder root can't read) is kept with why, which flags it.
    seen, left = set(), [link_cap]
    stack = [(l, None) for l in sorted(folder_links, reverse=True)]
    while stack:
        link, parent = stack.pop()
        try:
            st = os.stat(link)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode):
            continue
        key = (st.st_dev, st.st_ino)
        real = os.path.realpath(link)
        try:
            rst = os.stat(real)
            same = (rst.st_dev, rst.st_ino) == key
        except OSError:
            same = False
        if same and (
            under(real, ("/proc", "/sys")) or (recorded(real) and not hides_tree(real))
        ):
            continue
        if key in seen:
            continue
        seen.add(key)
        if left[0] < 1:
            unwalked(link, "over-cap", parent)
            continue
        try:
            os.scandir(link).close()
        except OSError:
            unwalked(link, "unreadable", parent)
            continue
        found, nested, budget = {}, [], [left[0] - 1]
        try:
            walk(link, link, found, nested, budget)
        except (OverCap, RecursionError):
            unwalked(link, "over-cap", parent)
            left[0] = 0
            continue
        left[0] = budget[0]
        files.update(found)
        stack.extend((l, link) for l in sorted(nested, reverse=True))
"""
    + _image_trees.SECTIONS_SCRIPT
    + r"""
walk("/")
for top, prefix in new_trees:
    walk(top, tree=prefix)
follow_links()
# Each installed pytest's version, from its metadata's name (pip's
# pytest-X.Y.Z.dist-info, conda's or setuptools' pytest-X.Y.Z-py3.6.egg-info),
# read without running anything of the image's.
dist = re.compile(r"^pytest-([0-9][^-]*?)(?:-py[0-9.]+)?\.(?:dist|egg)-info$")
pytests = {}
for site in dict.fromkeys(sites):
    try:
        listed = os.listdir(site)
    except OSError:
        continue
    for name in listed:
        found = dist.match(name)
        if found:
            pytests[os.path.join(site, name)] = found.group(1)
record = {"version": 1, "taken": taken, "files": files, "pytest": pytests}
_bf_trees(record, pruned, options)
text = json.dumps(record, sort_keys=True)
if target:
    folder = os.path.dirname(target)
    os.makedirs(folder, mode=0o700, exist_ok=True)
    os.chown(folder, 0, 0)
    os.chmod(folder, 0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
sys.stdout.write(text)
"""
)
# When a walk was taken: before the agent's turn, before a separate
# verifier's transfer, or after the turn (the detector's).
TAKEN = ("pre-agent", "pre-transfer", "post-turn")


@dataclass(frozen=True)
class ImageManifest:
    """The host's copy: absolute path -> the image's sha256."""

    taken: str
    files: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # The Python trees, interpreters, PATH tools and shell start-up files
    # (_image_trees); None in a record without them. The write-back leaves
    # them out.
    trees: Mapping[str, Any] | None = None
    # Each installed pytest's metadata path -> its version (strict
    # leaves out an option the image's pytest lacks).
    pytests: Mapping[str, str] = field(default_factory=dict)

    def digest(self, path: str) -> str | None:
        """The image's sha256 for *path*, or None: not the image's file."""
        entry = self.files.get(path)
        digest = entry.get("sha256") if isinstance(entry, Mapping) else None
        return digest if isinstance(digest, str) and _SHA256.fullmatch(digest) else None

    def named(self, name: str) -> dict[str, str]:
        """path -> sha256 of every recorded file called *name*."""
        found = {}
        for path in self.files:
            digest = self.digest(path)
            if digest and path.rsplit("/", 1)[-1] == name:
                found[path] = digest
        return found

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": VERSION,
                "taken": self.taken,
                "files": dict(self.files),
                "pytest": dict(self.pytests),
            },
            sort_keys=True,
        )


def _kept_entry(entry: Mapping[str, Any]) -> bool:
    """A file's sha256, or a link's target (``link:<target>``, with its
    ``link``)."""
    digest, link = entry.get("sha256"), entry.get("link")
    if isinstance(link, str):
        return digest == "link:" + link
    return isinstance(digest, str) and _SHA256.fullmatch(digest) is not None


def load(text: str | None) -> ImageManifest | None:
    """The manifest in *text*, or None when it is missing or not one."""
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return None
    files = data.get("files")
    taken = data.get("taken")
    if not isinstance(files, dict) or taken not in TAKEN:
        return None
    clean = {
        path: {"sha256": entry["sha256"], "size": entry.get("size")}
        | {
            key: entry[key]
            for key in ("pytest", "link", "target", "via", "unwalked", "tree")
            if isinstance(entry.get(key), str)
        }
        for path, entry in files.items()
        if isinstance(path, str)
        and path.startswith("/")
        and isinstance(entry, dict)
        and _kept_entry(entry)
    }
    found = data.get("pytest")
    pytests = (
        {
            path: version
            for path, version in found.items()
            if isinstance(path, str) and isinstance(version, str)
        }
        if isinstance(found, dict)
        else {}
    )
    return ImageManifest(
        taken=taken, files=clean, pytests=pytests, **_image_trees.parse(data)
    )


def _with_caps(options: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "link_cap": LINK_CAP,
        "target_cap": LINK_TARGET_CAP,
        "runtime_root": RUNTIME_ROOT,
        **dict(options or {}),
    }


def record_command(
    taken: str,
    target: str = MANIFEST_PATH,
    options: Mapping[str, Any] | None = None,
) -> str:
    """The root command recording the manifest (stdout and its file, none
    for an empty *target*). Exits non-zero when the sandbox has neither the
    kit's nor a system python3. *options* go to the Python trees and PATH
    tools sections (:func:`_image_trees.after_options` after the turn)."""
    walk = root_python_sh(
        _RECORD_SCRIPT,
        json.dumps(list(NAMES)),
        json.dumps(list(SUFFIXES)),
        json.dumps(list(PRUNED)),
        taken,
        target,
        json.dumps(_with_caps(options)),
    )
    return (
        f"if [ -x {KIT_PYTHON} ] || command -v python3 >/dev/null 2>&1; "
        f"then {walk}; else exit 127; fi"
    )


async def record(env: Any, taken: str, *, timeout_sec: int) -> ImageManifest | None:
    """Record the manifest in *env*, where no agent process has run yet, and
    keep the host's copy on *env*. None (and nothing kept) when it cannot be
    recorded: the sweep and the guard then behave as before."""
    from benchflow.sandbox._exec_output import exec_raw

    try:
        result = await exec_raw(
            env.exec, record_command(taken), user="root", timeout_sec=timeout_sec
        )
    except Exception as exc:
        logger.warning("image manifest not recorded: %s", exc)
        return None
    manifest = None
    code = getattr(result, "return_code", 1)
    if code == 0:
        manifest = load(getattr(result, "stdout", "") or "")
    elif code == 127:
        # No Python in the image (nor the kit's yet): the shell walk (F-C).
        manifest, code = await _shell_walk(env, taken, None, timeout_sec=timeout_sec)
    if manifest is None:
        logger.warning(
            "image manifest not recorded (rc %s): the detector records this "
            "trial as unchecked; in strict, the conftest sweep removes every "
            "conftest.py outside the verifier's tests, as before (no keep rule)",
            code,
        )
        return None
    env._benchflow_image_manifest = manifest
    return manifest


#: The pytest configuration files whose pytest section the walk hashes.
SECTIONED = (
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "tox.ini",
    "setup.cfg",
    "pyproject.toml",
)


async def _shell_walk(
    env: Any, taken: str, options: Mapping[str, Any] | None, *, timeout_sec: int
) -> tuple[ImageManifest | None, Any]:
    """The walk from the shell, for an image without a Python (F-C): the
    same record (_image_trees.shell_walk_command). (None, rc) when it
    cannot be taken: without GNU find or sha256sum the trial stays
    unchecked."""
    from benchflow.sandbox._exec_output import exec_raw

    command = _image_trees.shell_walk_command(
        PRUNED, NAMES, SUFFIXES, SECTIONED, _with_caps(options)
    )
    try:
        result = await exec_raw(env.exec, command, user="root", timeout_sec=timeout_sec)
    except Exception as exc:
        logger.warning("the shell walk failed: %s", exc)
        return None, None
    code = getattr(result, "return_code", 1)
    found = _image_trees.load_shell(getattr(result, "stdout", "") or "", taken)
    if code != 0 or found is None:
        return None, code
    files, trees = found
    return ImageManifest(taken=taken, files=files, trees=trees), code


def kept(env: Any) -> ImageManifest | None:
    """The host's copy recorded for *env*, or None."""
    manifest = getattr(env, "_benchflow_image_manifest", None)
    return manifest if isinstance(manifest, ImageManifest) else None


# -- the detector (the faithful default) ------------------------------

DETECTOR_SCHEMA = "benchflow.verifier_inputs.v1"
#: The trial's record, in its integrity folder.
DETECTOR_RECORD = "verifier-inputs.json"
# Flags: the test machinery was changed (the lead's 10:30Z scope). Every
# other difference is a change: the inventory of what an honest solution does,
# never a flag (_image_trees has the trees' and tools' categories).
FLAG_CATEGORIES = frozenset(
    {"conftest", "pytest-config", "startup", "walk-failed", "verifier-output"}
    | _image_trees.CATEGORIES
)
_CONFIG_NAMES = frozenset(NAMES) - {"conftest.py"}
# Changes listed in full up to this many; the rest are counted.
CHANGES_MAX = 200


def category(path: str) -> str:
    """What *path* is to a test run: ``conftest``, ``pytest-config``,
    ``startup`` (sitecustomize, usercustomize, a .pth), ``bytecode`` (a .pyc
    outside ``__pycache__``, which imports without a source), ``pytest-plugin``
    (a ``pytest_*.py`` module), ``bytecode-cache`` or ``python``."""
    folder, _, name = path.rpartition("/")
    if name == "conftest.py":
        return "conftest"
    if name in _CONFIG_NAMES:
        return "pytest-config"
    if name in ("sitecustomize.py", "usercustomize.py") or name.endswith(".pth"):
        return "startup"
    if name.endswith(".pyc"):
        return "bytecode-cache" if folder.endswith("/__pycache__") else "bytecode"
    if name.startswith(("pytest_", "_pytest_")) and name.endswith(".py"):
        return "pytest-plugin"
    return "python"


def _fingerprint(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    found = {"sha256": entry.get("sha256"), "size": entry.get("size")}
    if "pytest" in entry:
        found["pytest_section"] = entry["pytest"] or None
    for key in ("target", "via", "unwalked"):
        if key in entry:
            found[key] = entry[key]
    return found


def _snapshot_moved_target(
    was: Any, now: Any, context: Mapping[str, Any] | None
) -> str:
    """For a kept link's target over the hash cap (kept by size, device,
    inode, mtime, ctime, owner and mode): a note when the ctime is the one
    field that moved, into a snapshot window the trial recorded
    (``context["snapshot_windows"]``, :func:`_image_trees.snapshot_during`),
    on the same file (device and inode) that only root can write in place
    (:func:`_image_trees.only_root_writes`). A snapshot sets the ctime of
    every file in the sandbox's writable layer, an edited one's too, so any
    other target is compared as without a snapshot. "" otherwise."""
    windows = (context or {}).get("snapshot_windows") or ()
    if not windows or not isinstance(was, str) or not isinstance(now, str):
        return ""
    old, new = was.split(":"), now.split(":")
    if len(old) != 9 or len(new) != 9 or old[0] != "regular" or old[2] != "not-hashed":
        return ""
    if [old[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)] != [
        new[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)
    ]:
        return ""
    try:
        root_only = _image_trees.only_root_writes(int(new[8], 8), int(new[7]))
        start = _image_trees.snapshot_during(int(new[6]) / 1e9, windows)
    except ValueError:
        return ""
    if not root_only or start is None:
        return ""
    return _image_trees.snapshot_note(start)


def _uv_owned(path: str, owned: Mapping[str, str]) -> bool:
    """A file of a wheel in BenchFlow's own uv cache, as unpacked: the walk
    found its archive entry's RECORD lists it (``uv:``), and it is there. Only
    a path in normal form (no ``..``, ``.`` or empty component), as the walk
    gives one."""
    return (
        posixpath.normpath(path) == path
        and path.startswith(_image_trees.UV_ARCHIVE + "/")
        and owned.get(path, "").startswith("uv:")
    )


def _installed(path: str, owned: Mapping[str, str]) -> bool:
    """A package's own conftest.py or pytest configuration file: a wheel's in
    BenchFlow's uv cache (:func:`_uv_owned`), or one dpkg lists with its md5
    (apt-get installing python3-numpy in a Python tree made in the turn).
    dpkg's database is root's alone."""
    return _uv_owned(path, owned) or owned.get(path, "").startswith("dpkg:")


def _new_tree_entries(before: ImageManifest, after: ImageManifest) -> set[str]:
    """The paths the walk after kept in a Python tree made in the turn
    (``tree``: its interpreter prefix) whose prefix held no file the walk
    before recorded: a venv or user site made in the turn, where pytest
    doesn't collect. A folder that held files before (the tests' own) made
    into a venv prefix keeps them, so a conftest.py there is compared."""
    keys = sorted(before.files)
    dropped = set()
    for path, entry in after.files.items():
        prefix = entry.get("tree")
        if not isinstance(prefix, str):
            continue
        start = prefix.rstrip("/") + "/"
        at = bisect.bisect_left(keys, start)
        if not (at < len(keys) and keys[at].startswith(start)):
            dropped.add(path)
    return dropped


def _diff_files(
    before: ImageManifest,
    after: ImageManifest,
    context: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """The walk's files before and after the turn: (flags, changes, how many
    paths were compared). Edited, planted (new) or removed, by sha256.

    Flags: a conftest; a pytest configuration file whose pytest section
    changed (its own hash; an edit elsewhere in tox.ini, setup.cfg or
    pyproject.toml is a change); a ``.pth`` or customize module added or
    changed, unless a package owns it as installed. A link (kept by its
    target and what that holds) is never a package's: one planted,
    retargeted, or whose target changed is a flag whatever it points at. A
    folder link the walk left unwalked (``unwalked``: past the cap, or a
    folder root can't read) is a ``walk-failed`` flag, unless it is the same
    link, unwalkable before the turn too. Anything else is a change."""
    flags: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    owned = _image_trees.owned(after)
    paths = sorted(
        (set(before.files) | set(after.files)) - _new_tree_entries(before, after)
    )
    for path in paths:
        was, now = before.files.get(path), after.files.get(path)
        if now is not None and now.get("unwalked"):
            if (
                was is not None
                and was.get("unwalked")
                and was.get("sha256") == now.get("sha256")
                and was.get("target") == now.get("target")
            ):
                # The same link, unwalkable before the turn too (an image's
                # own): unchanged.
                continue
            # A folder link the walk could not follow (past the cap, or a
            # folder root can't read), new or changed: what is behind it is
            # unknown.
            flags.append(
                {
                    "where": "files",
                    "path": path,
                    "kind": "failed",
                    "category": "walk-failed",
                    "before": _fingerprint(was),
                    "after": _fingerprint(now),
                }
            )
            continue
        if (
            was is not None
            and now is not None
            and was.get("sha256") == now.get("sha256")
        ):
            if was.get("target") == now.get("target"):
                continue
            note = _snapshot_moved_target(was.get("target"), now.get("target"), context)
            if note:
                changes.append(
                    {
                        "where": "files",
                        "path": path,
                        "kind": "edited",
                        "category": "snapshot-ctime",
                        "before": _fingerprint(was),
                        "after": _fingerprint(now),
                        "note": note,
                    }
                )
                continue
        kind = "planted" if was is None else "removed" if now is None else "edited"
        found = category(path)
        linked = bool((now or {}).get("link"))
        if found == "pytest-config":
            old = was.get("pytest") if was is not None else ""
            new = now.get("pytest") if now is not None else ""
            # A record without section hashes cannot tell: the whole file.
            flag = old is None or new is None or old != new
            if not flag:
                found = "config"
            elif kind != "removed" and not linked and _installed(path, owned):
                flag, found = False, "pytest-config-owned"
        elif found == "startup":
            flag = kind != "removed"
            if flag and not linked and path in owned:
                # A package's own, as installed: apt-get installing python3.12
                # writes /etc/python3.12/sitecustomize.py (a conffile).
                flag, found = False, "startup-owned"
        else:
            flag = found == "conftest"
            if flag and kind != "removed" and not linked and _installed(path, owned):
                # A wheel's own conftest.py, as BenchFlow's uv cache unpacked
                # it for the model proxy (its dist RECORD lists it): an install.
                flag, found = False, "conftest-owned"
        row = {
            "where": "files",
            "path": path,
            "kind": kind,
            "category": found,
            "before": _fingerprint(was),
            "after": _fingerprint(now),
        }
        (flags if flag else changes).append(row)
    return flags, changes, len(paths)


#: Each section of the walks and its diff: (before, after, context) ->
#: (flags, changes, paths compared). *context* has ``tool_names``, the tools
#: test.sh names (_image_trees.tool_names), ``own_processes``, BenchFlow's
#: own daemons (_image_trees.own_processes), and ``snapshot_windows``, the
#: snapshots the trial took (_image_trees.snapshot_during).
SECTION_DIFFS: dict[str, Any] = {
    "files": _diff_files,
    "python_trees": _image_trees.diff_python_trees,
    "interpreters": _image_trees.diff_interpreters,
    "path_tools": _image_trees.diff_path_tools,
    "startup": _image_trees.diff_startup,
    "processes": _image_trees.diff_processes,
}


async def walk_after(
    env: Any, *, timeout_sec: int, startup_extra: tuple[str, ...] = ()
) -> ImageManifest | None:
    """The same walk as the record, after the agent's turn, kept on the host
    only (nothing is written in the sandbox). None when it fails."""
    from benchflow.sandbox._exec_output import exec_raw

    before = kept(env)
    # The same PATH folders as the walk before, and its start, so what
    # changed since is hashed for the flags (_image_trees).
    options = _image_trees.after_options(before, startup_extra)
    if before is not None and (before.trees or {}).get("walker") == "shell":
        # The same walk as before the turn, so the two compare like for like.
        walked, _ = await _shell_walk(
            env, "post-turn", options, timeout_sec=timeout_sec
        )
        return walked
    from benchflow.diagnostics import TransientSandboxTransportError
    from benchflow.sandbox._verifier_links import READ_ATTEMPTS

    # Asked again when the sandbox's transport loses the exec (a Daytona
    # snapshot drops the command channel), as bfx's reads are: the walk
    # writes nothing in the sandbox.
    attempt = 1
    while True:
        try:
            result = await exec_raw(
                env.exec,
                record_command("post-turn", "", options),
                user="root",
                timeout_sec=timeout_sec,
            )
            break
        except TransientSandboxTransportError as exc:
            if attempt >= READ_ATTEMPTS:
                logger.warning("the post-turn walk failed: %s", exc)
                return None
            attempt += 1
            logger.warning(
                "the post-turn walk lost its exec to the sandbox's transport "
                "(%s); asking again",
                exc,
            )
        except Exception as exc:
            logger.warning("the post-turn walk failed: %s", exc)
            return None
    if getattr(result, "return_code", 1) != 0:
        logger.warning(
            "the post-turn walk failed (rc %s)", getattr(result, "return_code", None)
        )
        return None
    walked = load(getattr(result, "stdout", "") or "")
    return walked if walked is not None and walked.taken == "post-turn" else None


def detector_record(
    mode: str,
    before: ImageManifest | None,
    after: ImageManifest | None,
    *,
    context: Mapping[str, Any] | None = None,
    walk_failed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The trial's ``verifier-inputs.json``: every registered section's
    flags and changes between the record before the turn and the walk after
    it. ``off`` runs nothing and says so. Without a record before the turn
    it is ``unchecked``, with the reason; a walk after a good record that
    fails is *walk_failed*'s flags (:func:`detect`), never a quiet status.
    ``notes`` names what the detector does not cover. ``reward_changed`` is
    always false: a flag never changes the reward."""
    record: dict[str, Any] = {
        "schema": DETECTOR_SCHEMA,
        "mode": mode,
        "status": "unchecked",
        "unchecked_because": None,
        "before": {"taken": before.taken} if before is not None else None,
        "after": {"taken": after.taken} if after is not None else None,
        "sections": {},
        "flags": [],
        "changes": [],
        "changes_by_category": {},
        "notes": list(_image_trees.NOTES),
        "reward_changed": False,
    }
    if mode == "off":
        record["status"] = "off"
        record["unchecked_because"] = "integrity is off: the detector does not run"
        return record
    if before is None:
        record["unchecked_because"] = "no record of the image from before the turn"
        return record
    if after is None:
        record["unchecked_because"] = "the walk after the turn failed"
        if walk_failed:
            record["flags"] = list(walk_failed)
            record["status"] = "flagged"
        return record
    for where, differ in SECTION_DIFFS.items():
        flags, changes, checked = differ(before, after, context)
        record["sections"][where] = {
            "checked": checked,
            "flags": len(flags),
            "changes": len(changes),
        }
        record["flags"].extend(flags)
        record["changes"].extend(changes)
    counts: dict[str, int] = {}
    for change in record["changes"]:
        counts[change["category"]] = counts.get(change["category"], 0) + 1
    record["changes_by_category"] = dict(sorted(counts.items()))
    if len(record["changes"]) > CHANGES_MAX:
        record["changes_truncated"] = len(record["changes"]) - CHANGES_MAX
        record["changes"] = record["changes"][:CHANGES_MAX]
    record["status"] = "flagged" if record["flags"] else "checked"
    return record


async def _walk_failed(env: Any, before: ImageManifest, *, timeout_sec: int):
    """The walk after a good record failed (its Python replaced, say): a
    ``walk-failed`` flag, and a shell-only look at the interpreters."""
    from benchflow.sandbox._exec_output import exec_raw

    found: list[dict[str, Any]] = [
        {
            "where": "walk",
            "path": "",
            "kind": "failed",
            "category": "walk-failed",
            "before": None,
            "after": None,
        }
    ]
    command = _image_trees.interpreter_stat_command(before)
    if command:
        output = ""
        try:
            result = await exec_raw(
                env.exec, command, user="root", timeout_sec=timeout_sec
            )
            output = getattr(result, "stdout", "") or ""
        except Exception as exc:
            logger.warning("the interpreters could not be looked at: %s", exc)
        notes: list[dict[str, Any]] = []
        found.extend(
            _image_trees.interpreter_stat_flags(
                before, output, getattr(env, "_snapshot_windows", ()), notes
            )
        )
        if notes:
            found[0]["ctime_not_compared"] = notes
        unchecked = _image_trees.interpreter_stat_unchecked(before, output)
        if unchecked:
            # Not checked: the walk-failed flag stands, and says which.
            found[0]["interpreters_not_checked"] = unchecked
    return found


async def detect(
    env: Any,
    mode: str,
    *,
    timeout_sec: int,
    test_script: str | None = None,
    verifier_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Walk after the turn (not in ``off``) and diff against the host's
    record from before it; the result is kept on *env* for the trial record
    and returned. *test_script* (the host's text of the task's test script)
    names the tools a PATH entry may shadow; *verifier_env* (test.sh's env)
    names a ``BASH_ENV``/``ENV`` start-up file."""
    before = kept(env)
    after = None
    walk_failed = None
    if mode != "off" and before is not None:
        after = await walk_after(
            env,
            timeout_sec=timeout_sec,
            startup_extra=tuple(_image_trees.startup_extra(verifier_env)),
        )
        if after is None:
            walk_failed = await _walk_failed(env, before, timeout_sec=timeout_sec)
    record = detector_record(
        mode,
        before,
        after,
        context={
            "tool_names": _image_trees.tool_names(test_script),
            "own_processes": _image_trees.own_processes(env),
            "own_shells": _image_trees.session_shells(env),
            "snapshot_windows": tuple(getattr(env, "_snapshot_windows", None) or ()),
        },
        walk_failed=walk_failed,
    )
    env._benchflow_verifier_inputs = record
    return record


def write_detector_record(record_dir: Any, record: Mapping[str, Any]) -> None:
    """Write *record* to the trial's ``integrity/verifier-inputs.json``, the
    credentials BenchFlow staged for the trial masked: its process rows hold
    command lines (``_utils.staged_secrets``)."""
    from pathlib import Path

    from benchflow._utils import staged_secrets

    folder = Path(record_dir) / "integrity"
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, sort_keys=True)
    (folder / DETECTOR_RECORD).write_text(staged_secrets.mask(text) + "\n")


# Up to this size the host's copy goes in the exec's own text (a quoted
# here-document); a larger one is uploaded.
_INLINE_MAX = 256 << 10


async def write_back(env: Any, manifest: ImageManifest, *, timeout_sec: int) -> bool:
    """Replace the sandbox's copy with the host's, in a new folder only root
    may enter, for the readers in the sandbox. False when it could not be
    written: those readers then find no manifest and behave as before."""
    import os
    import tempfile

    from benchflow.sandbox._exec_output import exec_raw

    folder = shlex.quote(MANIFEST_DIR)
    target = shlex.quote(MANIFEST_PATH)
    fresh = f"umask 077 && rm -rf -- {folder} && mkdir -m 700 -- {folder} && chown 0:0 -- {folder}"
    text = manifest.to_json()
    marker = "BF_IMAGE_MANIFEST_" + os.urandom(8).hex()
    try:
        if len(text) <= _INLINE_MAX and "\n" not in text and marker not in text:
            done = await exec_raw(
                env.exec,
                f"{fresh} && cat > {target} <<'{marker}'\n{text}\n{marker}\n"
                f"chown 0:0 -- {target} && chmod 600 -- {target}",
                user="root",
                timeout_sec=timeout_sec,
            )
            return getattr(done, "return_code", 1) == 0
        made = await exec_raw(env.exec, fresh, user="root", timeout_sec=timeout_sec)
        if getattr(made, "return_code", 1) != 0:
            return False
        with tempfile.TemporaryDirectory(prefix="bf-image-manifest-") as tmp:
            local = os.path.join(tmp, "image-manifest.json")
            with open(local, "w") as handle:
                handle.write(text)
            # BenchFlow's own data, as the verifier runtime's tree is.
            from benchflow.trust.state import TRUSTED_TRANSFER, admit

            await env.upload_file(
                local,
                MANIFEST_PATH,
                admission=admit(
                    env, "main", TRUSTED_TRANSFER, reason="BenchFlow's image manifest"
                ),
            )
        done = await exec_raw(
            env.exec,
            f"chown 0:0 -- {target} && chmod 600 -- {target}",
            user="root",
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        logger.warning("image manifest not written back: %s", exc)
        return False
    return getattr(done, "return_code", 1) == 0
