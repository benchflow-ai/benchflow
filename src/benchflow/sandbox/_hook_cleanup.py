"""The verifier's hook clean-up on ``bfx`` (V8-MERGE-1).

Where ``bfx`` is delivered, the host runs the policy of today's clean-up
(:func:`benchflow.sandbox.lockdown._build_cleanup_cmd`) itself, unchanged,
and no interpreter in the sandbox is needed: an image without Python runs it
too. Every fact comes from ``bfx scan`` (lstat facts, link texts, folder
listings and file bytes, never following a link and never entering a mount
unless the host names it) and every change is a ``bfx`` request (``remove``
expecting the inode it saw, ``write``, ``mkdir``, ``chmod``, ``clear``), all
over one ``bfx serve`` session under one deadline for the whole step.

The three parts, in today's order:

1. ``conftest.py`` outside ``/tests`` and ``/verifier`` (unless the task
   sets ``cleanup_conftests = false``): today's ``find /`` with its prunes.
   Mounted folders are walked as roots of their own, as ``find`` crosses
   into them. An installed package's own file stays (:func:`package_owned`),
   and so does the repository's own, unchanged since the pre-agent copy
   (:func:`unchanged_in_workspace`), written again as a fresh root-owned file.
2. ``*.py`` in ``/tmp`` and ``/var/tmp`` outside the workspace and its
   aliases.
3. Python start-up hooks (:data:`benchflow._image_python_paths.HOOK_RULE`):
   the folders test.sh's Python would search, judged against the pre-agent
   records; a new or changed hook is removed, a recorded one is written
   again as a fresh root-owned copy.

Parts 1 and 2 are best effort, as ``find ... 2>/dev/null`` is, but never
silently partial: an entry that vanishes while it is walked is skipped and
recorded (:attr:`Report.vanished`), as ``find`` skips it, and any other
failure to walk a root completely fails the step. A root is skipped only when
it is itself absent. A tree of any depth is walked (bfx hands back a folder
past its depth for the host to walk on), and every path is carried as raw
bytes, so names that are not text are handled as today. Part 3 fails the
step when a hook that must go cannot be removed, or when a folder it must
judge cannot be listed (``os.listdir`` raises today). Any answer ``bfx``
cannot account for (an ambiguous or lost request) fails the step; the shell
form is never tried after it.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import csv
import fnmatch
import hashlib
import io
import json
import posixpath
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from benchflow._bfx.client import BfxAmbiguous, BfxError, BfxProtocolError

# HOOK_RULE's _BF_MAX: a hook larger than this has no digest (it is removed,
# as today), and a .pth larger than this names no folder.
MAX_HOOK_BYTES = 64 << 20
# HOOK_RULE's limits for a hook package's tree.
_MAX_DEPTH = 16
_MAX_NAMES = 4096
# The kernel's limit on links followed in one lookup.
_MAX_HOPS = 40
# The pre-agent records (benchflow._image_python_paths).
_RECORD = "/opt/runtime/rt/IMAGE_PYTHON_PATHS.json"
_RECORD_EACH = "/opt/runtime/rt/IMAGE_PYTHON_PATHS.jsonl"
_KIT_RECORD = "/opt/runtime/rt/RECORD"
_EROFS = 30
_EXDEV = 18
_EMFILE = 24

Call = Callable[[str, Mapping[str, Any], float], Awaitable[dict[str, Any]]]


class HookCleanupError(RuntimeError):
    """The clean-up did not complete; the message says what was left."""


class _Unaddressable(Exception):
    """A path bfx cannot be given: one holding NUL, which no file system
    name holds."""


@dataclass
class Report:
    """What the clean-up skipped without failing: entries that vanished while
    they were walked (absolute paths, surrogate-escaped bytes)."""

    vanished: list[str] = field(default_factory=list)
    # New or changed hooks kept because only root could have written them.
    root_made: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Inputs:
    """test.sh's Python search-path inputs, as root's preparation route holds
    them (the same expansions today's step reads): PYTHONPATH, PYTHONHOME and
    PYTHONUSERBASE, the original working folder, the route's PATH and the
    image's PATH."""

    values: tuple[str, str, str]
    cwd: str
    path: str
    image_path: str


# A NUL-separated record of :class:`Inputs`, printed by the preparation route.
INPUTS_CMD = (
    "printf '%s\\0%s\\0%s\\0%s\\0%s\\0%s\\0' "
    '"${_BENCHFLOW_IMAGE_PYTHONPATH-${__bf_hook_pythonpath-${PYTHONPATH-}}}" '
    '"${_BENCHFLOW_IMAGE_PYTHONHOME-${__bf_hook_pythonhome-${PYTHONHOME-}}}" '
    '"${_BENCHFLOW_IMAGE_PYTHONUSERBASE-${PYTHONUSERBASE-}}" '
    '"$PWD" "$PATH" "${_BENCHFLOW_IMAGE_PATH-}"'
)


def parse_inputs(stdout: str | bytes) -> Inputs:
    raw = (
        stdout.encode("utf-8", "surrogateescape") if isinstance(stdout, str) else stdout
    )
    parts = raw.split(b"\0")
    if len(parts) != 7 or parts[-1] != b"":
        raise HookCleanupError("the search-path inputs were not printed whole")
    text = [p.decode("utf-8", "surrogateescape") for p in parts[:6]]
    return Inputs((text[0], text[1], text[2]), text[3], text[4], text[5])


def _text(entry: Mapping[str, Any], key: str) -> str | None:
    if isinstance(entry.get(key), str):
        return entry[key]
    raw = entry.get(f"{key}_b64")
    if isinstance(raw, str):
        return base64.b64decode(raw).decode("utf-8", "surrogateescape")
    return None


def _bytes(path: str) -> bytes:
    """*path*'s bytes: paths are kept as text with each byte that is not
    UTF-8 surrogate-escaped, as ``os.fsdecode`` keeps them."""
    try:
        raw = path.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        raise _Unaddressable(path) from None
    if b"\0" in raw:
        raise _Unaddressable(path)
    return raw


def _put(key: str, path: str) -> dict[str, str]:
    """*path* as bfx takes it: text under *key*, else its raw bytes, base64,
    under ``<key>_b64``."""
    raw = _bytes(path)
    try:
        return {key: raw.decode("utf-8")}
    except UnicodeDecodeError:
        return {f"{key}_b64": base64.b64encode(raw).decode()}


def _where(**paths: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, path in paths.items():
        out.update(_put(key, path))
    return out


def _split(path: str) -> tuple[str, str]:
    parent, name = posixpath.split(path)
    return parent or "/", name


class _Facts:
    """Facts read through ``bfx scan``, and changes made through ``bfx``,
    within one deadline."""

    def __init__(self, call: Call, deadline: float) -> None:
        self._call = call
        self._deadline = deadline
        self._lstat: dict[str, dict[str, Any] | None] = {}
        self.report = Report()

    def _left(self) -> float:
        left = self._deadline - asyncio.get_running_loop().time()
        if left <= 0:
            raise HookCleanupError("the hook clean-up ran out of its time budget")
        return left

    async def request(self, cmd: str, args: Mapping[str, Any]) -> dict[str, Any]:
        return await self._call(cmd, args, self._left())

    async def scan(self, **args: Any) -> dict[str, Any]:
        return await self.request(
            "scan", {"mounts": False, "acl": False, "max_bytes": 256 << 20, **args}
        )

    def forget(self) -> None:
        self._lstat.clear()

    # -- reads --------------------------------------------------------------

    async def lstat(self, path: str) -> dict[str, Any] | None:
        """lstat facts of *path*, whose parent holds no link; None when it
        does not exist (or a parent is not a folder)."""
        if path in self._lstat:
            return self._lstat[path]
        try:
            given = _put("paths", path)
        except _Unaddressable:
            return None
        answer = await self.scan(**{k: [v] for k, v in given.items()})
        found = None
        for entry in answer.get("entries") or ():
            if _text(entry, "path") == path and entry.get("exists", True):
                found = entry
        self._lstat[path] = found
        return found

    async def listdir(self, real: str) -> list[tuple[str, dict[str, Any]]] | None:
        """The entries of the folder *real* (no link in it), each with its
        lstat facts; None when it cannot be listed whole. A name that
        vanished while it was listed is left out, as ``os.listdir`` would
        not have shown it a moment later; any other per-name failure means
        not whole."""
        try:
            root = _put("root", real)
        except _Unaddressable:
            return None
        answer = await self.scan(walks=[{**root, "depth": 1}])
        for error in answer.get("errors") or ():
            where = _text(error, "path") if isinstance(error, dict) else None
            if where is not None and (
                where == real or posixpath.dirname(where) == real
            ):
                return None
        rows: list[tuple[str, dict[str, Any]]] = []
        listed = False
        for entry in answer.get("entries") or ():
            path = _text(entry, "path")
            if path == real:
                listed = entry.get("listing") is True and entry.get("type") == "dir"
                continue
            if path is None or posixpath.dirname(path) != real:
                continue
            self._lstat[path] = entry
            rows.append((posixpath.basename(path), entry))
        return rows if listed else None

    async def read(self, real: str) -> bytes | None:
        """The bytes of the regular file *real* (no link anywhere in it), or
        None: not a regular file, larger than :data:`MAX_HOOK_BYTES`, or
        unreadable."""
        try:
            given = _put("contents", real)
        except _Unaddressable:
            return None
        [(key, value)] = given.items()
        answer = await self.scan(**{key: [value]}, content_max_bytes=MAX_HOOK_BYTES)
        found = (answer.get(key) or {}).get(value) or {}
        if not isinstance(found.get("b64"), str):
            return None
        data = base64.b64decode(found["b64"])
        return data if len(data) <= MAX_HOOK_BYTES else None

    async def resolve(self, path: str) -> str | None:
        """*path* with every link followed, as the kernel's open() follows
        them; None when a component is missing, not a folder, or the links
        loop."""
        if not path.startswith("/") or "\0" in path:
            return None
        todo = [c for c in path.split("/") if c]
        cur = "/"
        hops = 0
        while todo:
            name = todo.pop(0)
            if name == ".":
                continue
            if name == "..":
                cur = posixpath.dirname(cur) or "/"
                continue
            nxt = posixpath.join(cur, name)
            entry = await self.lstat(nxt)
            if entry is None:
                return None
            if entry.get("type") == "link":
                hops += 1
                target = _text(entry, "link")
                if hops > _MAX_HOPS or target is None:
                    return None
                if target.startswith("/"):
                    cur = "/"
                todo[:0] = [c for c in target.split("/") if c]
                continue
            if todo and entry.get("type") != "dir":
                return None
            cur = nxt
        return cur

    async def resolve_parent(self, path: str) -> str | None:
        """*path* with every link but the last component's followed."""
        parent, name = _split(path)
        real = await self.resolve(parent)
        if real is None or not name:
            return None
        return posixpath.join(real, name)

    async def realpath(self, path: str) -> str:
        """posixpath.realpath (not strict), from facts."""
        out, _ = await self._join_realpath("", path, {})
        return posixpath.abspath(out)

    async def _join_realpath(
        self, path: str, rest: str, seen: dict[str, str | None]
    ) -> tuple[str, bool]:
        if rest.startswith("/"):
            rest = rest[1:]
            path = "/"
        while rest:
            name, _, rest = rest.partition("/")
            if not name or name == ".":
                continue
            if name == "..":
                if path:
                    path, name = posixpath.split(path)
                    if name == "..":
                        path = posixpath.join(path, "..", "..")
                else:
                    path = ".."
                continue
            newpath = posixpath.join(path, name)
            entry = await self.lstat(newpath) if newpath.startswith("/") else None
            if entry is None or entry.get("type") != "link":
                path = newpath
                continue
            if newpath in seen:
                known = seen[newpath]
                if known is not None:
                    path = known
                    continue
                return posixpath.join(newpath, rest), False
            seen[newpath] = None
            path, ok = await self._join_realpath(path, _text(entry, "link") or "", seen)
            if not ok:
                return posixpath.join(path, rest), False
            seen[newpath] = path
        return path, True

    async def isfile(self, path: str) -> bool:
        real = await self.resolve(path)
        entry = await self.lstat(real) if real else None
        return entry is not None and entry.get("type") == "file"

    async def isdir(self, path: str) -> bool:
        return await self.open_dir(path) is not None

    async def islink(self, path: str) -> bool:
        real = await self.resolve_parent(path)
        entry = await self.lstat(real) if real else None
        return entry is not None and entry.get("type") == "link"

    async def lexists(self, path: str) -> bool:
        real = await self.resolve_parent(path)
        return real is not None and await self.lstat(real) is not None

    async def open_dir(self, path: str) -> str | None:
        """HOOK_RULE's _bf_open_dir: the folder *path* names, every link
        followed, as a path with no link in it; None otherwise."""
        real = await self.resolve(path)
        if real is None:
            return None
        entry = await self.lstat(real)
        return real if entry is not None and entry.get("type") == "dir" else None

    async def follow_read(self, path: str) -> bytes | None:
        """The bytes open() of *path* would read, every link followed; None
        for anything but a regular file within :data:`MAX_HOOK_BYTES`."""
        real = await self.resolve(path)
        if real is None:
            return None
        entry = await self.lstat(real)
        if entry is None or entry.get("type") != "file":
            return None
        if int(entry.get("size") or 0) > MAX_HOOK_BYTES:
            return None
        return await self.read(real)

    async def glob_site(self, base: str, last: str) -> list[str]:
        """glob.glob(os.path.join(glob.escape(base), 'lib/python3*/' + last)),
        from facts: ``base/lib`` listed with links followed, names matching
        ``python3*`` that are folders (links followed, hidden names left
        out), then ``last`` beneath each, not following its final link."""
        lib = posixpath.join(base, "lib")
        real = await self.open_dir(lib)
        if real is None:
            return []
        rows = await self.listdir(real)
        out = []
        for name, entry in rows or ():
            if name.startswith(".") or not fnmatch.fnmatchcase(name, "python3*"):
                continue
            if entry.get("type") != "dir" and not (
                entry.get("type") == "link"
                and await self.open_dir(posixpath.join(real, name)) is not None
            ):
                continue
            if await self.lexists(posixpath.join(real, name, last)):
                out.append(posixpath.join(lib, name, last))
        return out


# -- HOOK_RULE, from facts ---------------------------------------------------


def is_hook(name: str, site_folder: bool) -> bool:
    for module in ("sitecustomize", "usercustomize"):
        if name == module or name.startswith(module + "."):
            return True
    return bool(site_folder) and name.endswith(".pth")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(node: Any) -> str | None:
    """HOOK_RULE's _bf_digest."""
    if node is None:
        return None
    if node[0] == "f":
        return _sha(node[1])
    if node[0] == "l":
        target = node[1].encode("utf-8", "surrogateescape")
        content = b"-" if node[2] is None else _sha(node[2]).encode()
        return "link:" + _sha(target + b"\0" + content)
    lines = [
        name.encode("utf-8", "surrogateescape")
        + b"\0"
        + (digest(kid) or "").encode()
        + b"\n"
        for name, kid in node[2]
    ]
    return "tree:" + _sha(b"".join(lines))


async def node(facts: _Facts, real_dir: str, name: str, depth: int = 0) -> Any:
    """HOOK_RULE's _bf_node: ('f', bytes, mode), ('l', target, bytes or
    None), ('d', mode, [(name, node)]) or None."""
    path = posixpath.join(real_dir, name)
    entry = await facts.lstat(path)
    if entry is None:
        return None
    kind = entry.get("type")
    mode = int(entry.get("mode") or 0)
    if kind == "link":
        return ("l", _text(entry, "link") or "", await facts.follow_read(path))
    if kind == "file":
        if int(entry.get("size") or 0) > MAX_HOOK_BYTES:
            return None
        data = await facts.read(path)
        return None if data is None else ("f", data, mode)
    if kind == "dir" and depth < _MAX_DEPTH:
        rows = await facts.listdir(path)
        if rows is None or len(rows) > _MAX_NAMES:
            return None
        kids = []
        for kid_name in sorted(n for n, _ in rows):
            kid = await node(facts, path, kid_name, depth + 1)
            if kid is None:
                return None
            kids.append((kid_name, kid))
        return ("d", mode, kids)
    return None


async def pth_folders(facts: _Facts, d: str, dirs: set[str], sites: set[str]) -> None:
    """HOOK_RULE's _bf_pth_folders, from facts."""
    real = await facts.open_dir(d)
    if real is None:
        return
    rows = await facts.listdir(real)
    if rows is None:
        return
    for name in sorted(n for n, _ in rows if n.endswith(".pth")):
        data = await facts.follow_read(posixpath.join(real, name))
        for line in (data or b"").decode("utf-8", "replace").splitlines():
            if not line.strip() or line.startswith("#") or "\0" in line:
                continue
            if not line.startswith(("import ", "import\t")):
                dirs.add(posixpath.normpath(posixpath.join(d, line.rstrip())))
                continue
            try:
                tree = ast.parse(line)
            except (SyntaxError, ValueError):
                continue
            for call in ast.walk(tree):
                attr = getattr(getattr(call, "func", None), "attr", "")
                if attr not in ("append", "insert", "addsitedir"):
                    continue
                for arg in getattr(call, "args", []):
                    value = getattr(arg, "value", None)
                    if isinstance(value, str) and posixpath.isabs(value):
                        (sites if attr == "addsitedir" else dirs).add(
                            posixpath.normpath(value)
                        )


_INPUT_NAMES = ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE")


async def env_inputs(facts: _Facts, values: Iterable[str]) -> dict[str, list[str]]:
    """HOOK_RULE's _bf_env_inputs: the route's values and PID 1's."""
    out: dict[str, list[str]] = {
        n: ([v] if v else []) for n, v in zip(_INPUT_NAMES, values, strict=True)
    }
    data = await facts.read("/proc/1/environ")
    for item in (data or b"").split(b"\0"):
        name, _, value = item.decode("utf-8", "surrogateescape").partition("=")
        if name in out and value and value not in out[name]:
            out[name].append(value)
    return out


async def search_inputs(
    facts: _Facts, values: Iterable[str], cwd: str
) -> tuple[set[str], set[str]]:
    """HOOK_RULE's _bf_search_inputs with ``harvest`` false (the clean-up's
    form): the folders and site folders test.sh's Python adds."""
    inputs = await env_inputs(facts, values)
    dirs: set[str] = set()
    found: set[str] = set()
    for value in inputs["PYTHONPATH"]:
        for entry in value.split(":"):
            if "\0" not in entry and (cwd or posixpath.isabs(entry)):
                dirs.add(posixpath.normpath(posixpath.join(cwd, entry)))
    for value in inputs["PYTHONHOME"]:
        for prefix in value.split(":")[:2]:
            if posixpath.isabs(prefix) and "\0" not in prefix:
                for last in ("site-packages", "dist-packages"):
                    found.update(await facts.glob_site(prefix, last))
    for base in [*inputs["PYTHONUSERBASE"], "/root/.local"]:
        if posixpath.isabs(base):
            found.update(await facts.glob_site(base, "site-packages"))
    found = {posixpath.normpath(p) for p in found}
    return dirs - found, found


# -- the pre-agent records (benchflow._image_python_paths.READ) --------------


def _invalid() -> HookCleanupError:
    return HookCleanupError("invalid pre-agent Python search paths")


def _check_record(entry: Any) -> None:
    def abs_list(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(p, str) and posixpath.isabs(p) and "\0" not in p for p in value
        )

    if not isinstance(entry, dict) or not all(
        abs_list(entry.get(k)) for k in ("path", "user_site")
    ):
        raise _invalid()
    if "site" in entry and not (
        isinstance(entry["site"], list)
        and all(isinstance(p, str) and posixpath.isabs(p) for p in entry["site"])
    ):
        raise _invalid()
    hooks = entry.get("hooks", {})
    if not isinstance(hooks, dict) or any(
        not isinstance(d, str)
        or not posixpath.isabs(d)
        or not isinstance(fs, dict)
        or any(not isinstance(n, str) or not isinstance(v, str) for n, v in fs.items())
        for d, fs in hooks.items()
    ):
        raise _invalid()


async def read_records(facts: _Facts) -> dict[str, Any]:
    """The pre-agent records as READ loads them. Without a record and
    without a kit, the folders come from the other inputs alone (no
    interpreter's own search path is read)."""
    paths: Any
    if await facts.isfile(_RECORD):
        data = await facts.follow_read(_RECORD)
        try:
            paths = json.loads(data if data is not None else b"")
        except ValueError:
            raise _invalid() from None
    elif await facts.isfile(_KIT_RECORD):
        raise HookCleanupError(
            "missing pre-agent Python search paths; start a fresh sandbox"
        )
    else:
        empty: dict[str, Any] = {"path": [], "user_site": [], "site": []}
        paths = empty
    if not isinstance(paths, dict):
        raise _invalid()
    paths["interpreters"] = {}
    if await facts.isfile(_RECORD_EACH):
        data = await facts.follow_read(_RECORD_EACH)
        if data is None:
            raise _invalid()
        for line in io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"):
            try:
                entry = json.loads(line)
            except ValueError:
                raise _invalid() from None
            if not isinstance(entry, dict) or not isinstance(entry.get("exe"), str):
                raise _invalid()
            paths["interpreters"][entry["exe"]] = entry
    for entry in [paths, *paths["interpreters"].values()]:
        _check_record(entry)
    return paths


def recorded_hooks(paths: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for entry in [paths, *paths["interpreters"].values()]:
        for d, files in (entry.get("hooks") or {}).items():
            out.setdefault(d, {}).update(files)
    return out


def site_folders(paths: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for entry in [paths, *paths["interpreters"].values()]:
        if "site" in entry:
            out.update(entry["site"])
        else:
            out.update(entry.get("hooks") or {})
            out.update(entry["path"] + entry["user_site"])
    return out


# -- changes -----------------------------------------------------------------


@dataclass
class _Changes:
    facts: _Facts
    failed: list[str] = field(default_factory=list)
    unaddressable: list[str] = field(default_factory=list)

    async def remove(self, real_dir: str, name: str, entry: Mapping[str, Any]) -> None:
        """Remove *name* in *real_dir* and everything beneath it, never
        following a link, expecting the inode that was seen. A mount on the
        way has its contents removed and is left, failing (as rmdir of a
        mount point fails). A tree deeper than one ``remove`` goes is
        removed from the bottom, a walk's handed-back folders first."""
        path = posixpath.join(real_dir, name)
        try:
            where = _where(root=real_dir, path=name)
        except _Unaddressable:
            self.unaddressable.append(path)
            raise
        if entry.get("mount"):
            if entry.get("type") == "dir":
                await self._clear_mount(path)
            raise _Refused(path, "a mount point")
        for attempt in (1, 2):
            try:
                await self.facts.request(
                    "remove",
                    {
                        **where,
                        "expect": {"dev": entry["dev"], "ino": entry["ino"]},
                        "recursive": True,
                    },
                )
                return
            except BfxError as error:
                if error.kind == "missing" or error.errno == 2:
                    return
                if attempt == 1 and _too_deep(error):
                    await self._remove_deeper(path)
                    continue
                for mount in _left(error, _EXDEV):
                    await self._clear_mount(posixpath.join(real_dir, mount))
                raise _Refused(
                    path,
                    f"{error.kind}: {error.message}",
                    error.errno,
                    _left_errnos(error),
                ) from None
            finally:
                self.facts.forget()

    async def _remove_deeper(self, path: str) -> None:
        """Remove each folder a walk of *path* hands back (``deeper``), so
        what is left of *path* is within one ``remove``'s depth."""
        # The pattern "" matches no entry: only the handed-back folders.
        answer = await self.facts.request(
            "walk", {**_put("root", path), "path": "", "patterns": [""]}
        )
        for entry in answer.get("deeper") or ():
            rel = _text(entry, "path")
            if rel:
                parent, name = _split(posixpath.join(path, rel))
                await self.remove(parent, name, entry)

    async def _clear_mount(self, path: str) -> None:
        try:
            await self.facts.request("clear", {**_put("root", path), "path": ""})
        except BfxError as error:
            for mount in _left(error, _EXDEV):
                await self._clear_mount(posixpath.join(path, mount))
        except _Unaddressable:
            self.unaddressable.append(path)

    async def write_fresh(self, real_dir: str, name: str, kept: Any) -> None:
        """Write the kept hook *kept* again as a fresh root-owned copy at
        *name* (HOOK_RULE's ``fresh``): a file through a new private file
        renamed over the name; a folder as a new tree. Raises
        :class:`_ReadOnly` where the folder is on a read-only mount."""
        if kept[0] == "d":
            entry = await self.facts.lstat(posixpath.join(real_dir, name))
            if entry is None:
                raise _Refused(posixpath.join(real_dir, name), "gone")
            try:
                await self.remove(real_dir, name, entry)
            except _Refused as refused:
                # bfx gives a partial removal's errnos in its detail only.
                if refused.errno == _EROFS or _EROFS in refused.left_errnos:
                    raise _ReadOnly from None
                raise
            await self._write_tree(real_dir, name, kept)
            return
        data, mode = (kept[1], kept[2]) if kept[0] == "f" else (kept[2], 0o644)
        await self._write_file(real_dir, name, data, mode)

    async def _write_file(
        self, real_dir: str, name: str, data: bytes, mode: int
    ) -> None:
        try:
            await self.facts.request(
                "write",
                {
                    **_where(root=real_dir, path=name),
                    "data_b64": base64.b64encode(data).decode(),
                    "mode": f"{mode & 0o7777 & 0o755:o}",
                    "uid": 0,
                    "gid": 0,
                },
            )
        except BfxError as error:
            if error.errno == _EROFS:
                raise _ReadOnly from None
            raise _Refused(posixpath.join(real_dir, name), error.message) from None
        finally:
            self.facts.forget()

    async def _write_tree(self, real_dir: str, name: str, kept: Any) -> None:
        path = posixpath.join(real_dir, name)
        try:
            await self.facts.request(
                "mkdir",
                {
                    **_where(root=real_dir, path=name),
                    "mode": "700",
                    "uid": 0,
                    "gid": 0,
                },
            )
            for kid_name, kid in kept[2]:
                if kid[0] == "l" and kid[2] is None:
                    continue
                if kid[0] == "d":
                    await self._write_tree(path, kid_name, kid)
                else:
                    data, mode = (kid[1], kid[2]) if kid[0] == "f" else (kid[2], 0o644)
                    await self._write_file(path, kid_name, data, mode)
            await self.facts.request(
                "chmod",
                {
                    **_put("root", path),
                    "path": "",
                    "mode": f"{kept[1] & 0o7777 & 0o755:o}",
                },
            )
        except BfxError as error:
            raise _Refused(path, error.message) from None
        finally:
            self.facts.forget()


class _Refused(Exception):
    def __init__(
        self,
        path: str,
        why: str,
        errno: int | None = None,
        left_errnos: frozenset[int] = frozenset(),
    ) -> None:
        super().__init__(f"{path}: {why}")
        self.path = path
        self.why = why
        self.errno = errno
        self.left_errnos = left_errnos


class _ReadOnly(Exception):
    pass


def _left_rows(error: BfxError) -> list[dict[str, Any]]:
    """A partial ``remove``/``clear``'s entries that stayed: ``{path,
    error}``, relative to the request's root."""
    detail = error.detail if isinstance(error.detail, dict) else {}
    return [
        row
        for row in detail.get("left") or ()
        if isinstance(row, dict) and isinstance(row.get("error"), dict)
    ]


def _left_errnos(error: BfxError) -> frozenset[int]:
    return frozenset(
        row["error"]["errno"]
        for row in _left_rows(error)
        if isinstance(row["error"].get("errno"), int)
    )


def _left(error: BfxError, errno: int) -> list[str]:
    return [
        row["path"]
        for row in _left_rows(error)
        if row["error"].get("errno") == errno and isinstance(row.get("path"), str)
    ]


def _too_deep(error: BfxError) -> bool:
    """bfx's ``remove`` stopped at its depth: its limit, or the descriptors
    a deep tree takes."""
    return (
        error.kind == "too-deep"
        or error.errno == _EMFILE
        or _EMFILE in _left_errnos(error)
    )


# -- the three parts ---------------------------------------------------------


async def _mount_points(facts: _Facts) -> list[str]:
    """Every folder mount point in the sandbox's mount table."""
    answer = await facts.scan(mounts=True)
    points = []
    for mount in answer.get("mounts") or ():
        point = mount.get("mount_point") if isinstance(mount, dict) else None
        if isinstance(point, str) and point.startswith("/") and point != "/":
            points.append(posixpath.normpath(point))
    return sorted(set(points))


async def _walk_tree(
    facts: _Facts,
    root: str,
    pattern: str,
    pruned: Callable[[str], bool],
    mounts: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Every entry beneath *root* (the root itself excluded) whose name
    matches *pattern*, as ``find root -prune ... -name pattern`` lists them:
    no link followed; mounts crossed, each mount point beneath *root* walked
    as a root of its own; a pruned path and everything beneath it left out.

    Complete or failing: an entry that vanished while it was walked is
    skipped and recorded in ``facts.report``, a folder bfx hands back past
    its depth is walked on as a root of its own, and a root is skipped only
    when it is itself absent (or not a folder, which ``find`` does not
    enter). Any other failure raises :class:`HookCleanupError`."""

    def cut(path: str) -> bool:
        while path not in ("", "/"):
            if pruned(path):
                return True
            path = posixpath.dirname(path)
        return False

    tops = [root] + [
        m for m in mounts if m.startswith(root.rstrip("/") + "/") and not cut(m)
    ]
    out: list[tuple[str, dict[str, Any]]] = []
    while tops:
        top = tops.pop(0)
        try:
            answer = await facts.request(
                "walk",
                {
                    **_put("root", top),
                    "path": "",
                    "patterns": [f"**/{pattern}"],
                    "max_entries": 1 << 40,
                },
            )
        except BfxError as error:
            if error.kind in ("missing", "link", "not-dir") or error.errno in (
                2,
                20,
                40,
            ):
                facts.forget()
                here = await facts.lstat(top)
                if here is None or here.get("type") != "dir":
                    # A mount point that is a file (Docker's /etc/hosts) is
                    # not a folder find enters; only one that is gone vanished.
                    if here is None and top != root:
                        facts.report.vanished.append(top)
                    continue
            raise HookCleanupError(
                f"could not walk {top!r} completely: {error}"
            ) from None
        for entry in answer.get("entries") or ():
            rel = _text(entry, "path")
            if rel is None:
                continue
            path = posixpath.join(top, rel)
            if not cut(path):
                out.append((path, entry))
        for entry in answer.get("vanished") or ():
            rel = _text(entry, "path")
            if rel is not None:
                facts.report.vanished.append(posixpath.join(top, rel))
        for entry in answer.get("deeper") or ():
            rel = _text(entry, "path")
            if rel is not None and not cut(posixpath.join(top, rel)):
                tops.append(posixpath.join(top, rel))
    return out


async def _remove_matches(
    changes: _Changes,
    matches: Iterable[tuple[str, dict[str, Any]]],
    *,
    empty_folders: bool = False,
) -> None:
    """``-exec rm -f -- {} +``: every match that is not a folder, best
    effort; a name bfx cannot be given is reported. With *empty_folders*
    (``-delete``, which works depth first), a matching folder left empty
    goes too."""
    folders = []
    for path, entry in matches:
        if entry.get("type") == "dir":
            folders.append((path, entry))
            continue
        parent, name = _split(path)
        try:
            await changes.remove(parent, name, entry)
        except (_Unaddressable, _Refused):
            continue
    if not empty_folders:
        return
    for path, entry in sorted(folders, key=lambda pe: -pe[0].count("/")):
        if entry.get("mount"):
            continue
        rows = await changes.facts.listdir(path)
        if rows != []:
            continue
        parent, name = _split(path)
        try:
            await changes.remove(parent, name, entry)
        except (_Unaddressable, _Refused):
            continue


def _find_path(pattern: str, path: str) -> bool:
    # find -path: a shell glob over the whole path, where * also matches /.
    return fnmatch.fnmatchcase(path, pattern)


# The conftest sweep's one exception: an installed package's own
# conftest.py. pip and uv list every file they install, with its sha256, in
# the distribution's *.dist-info/RECORD (PEP 376, PEP 627). beartype ships a
# module named _conf/conftest.py; removing it from an agent's install tree
# stopped that agent from starting again in a later round. The shell form
# (lockdown._CONFTEST_SWEEP_SCRIPT) applies the same rule in the sandbox.
SITE_FOLDER_NAMES = ("site-packages", "dist-packages")


def site_folder_of(path: str) -> tuple[str, str] | None:
    """The nearest site folder above *path*, and *path* relative to it."""
    parts = path.split("/")
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] in SITE_FOLDER_NAMES:
            return "/".join(parts[: i + 1]), "/".join(parts[i + 1 :])
    return None


def record_digest(data: bytes) -> str:
    """*data*'s digest as a RECORD line writes it."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode()


def record_lines(data: bytes) -> dict[str, set[str]]:
    """The paths a RECORD file lists, each with its sha256 digests."""
    out: dict[str, set[str]] = {}
    for row in csv.reader(io.StringIO(data.decode("utf-8", "replace"))):
        if len(row) >= 2 and row[1].startswith("sha256="):
            out.setdefault(row[0], set()).add(row[1])
    return out


def _inside(path: str, roots: Iterable[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


async def _site_records(
    facts: _Facts, site: str, cache: dict[str, dict[str, set[str]]]
) -> dict[str, set[str]]:
    if site not in cache:
        found: dict[str, set[str]] = {}
        for name, entry in await facts.listdir(site) or ():
            if name.endswith(".dist-info") and entry.get("type") == "dir":
                data = await facts.read(posixpath.join(site, name, "RECORD"))
                for rel, digests in record_lines(data or b"").items():
                    found.setdefault(rel, set()).update(digests)
        cache[site] = found
    return cache[site]


async def package_owned(
    facts: _Facts,
    path: str,
    entry: Mapping[str, Any],
    workspace_roots: Iterable[str],
    cache: dict[str, dict[str, set[str]]],
) -> bool:
    """Whether the conftest.py at *path* is an installed package's own file:
    a regular file outside the workspace, inside a site-packages or
    dist-packages folder, that a RECORD in that folder lists with this exact
    digest. One the agent planted or edited matches no RECORD line, and one
    in the workspace never counts: the agent writes there, RECORD files
    included."""
    where = site_folder_of(path)
    if where is None or entry.get("type") != "file":
        return False
    if _inside(path, workspace_roots):
        return False
    site, rel = where
    data = await facts.read(path)
    if data is None:
        return False
    records = await _site_records(facts, site, cache)
    return record_digest(data) in records.get(rel, ())


async def unchanged_in_workspace(
    facts: _Facts,
    path: str,
    entry: Mapping[str, Any],
    workspace_roots: Iterable[str],
    baseline: str | None,
) -> bytes | None:
    """The bytes of the conftest.py at *path* when it is the repository's own
    file (#1171, #1177): a regular file in the workspace, byte for byte the
    file at the same place in *baseline*, the pre-agent copy, which must be
    root's and closed to everyone else. ``read`` follows no link at any
    level of either path. None otherwise, and always without a baseline
    (the caller passes none when the agent ran as root)."""
    if not baseline or entry.get("type") != "file":
        return None
    base = await facts.lstat(baseline)
    if (
        base is None
        or base.get("type") != "dir"
        or int(base.get("uid", -1)) != 0
        or int(base.get("mode") or 0) & 0o077
    ):
        return None
    for root in workspace_roots:
        prefix = root.rstrip("/") + "/"
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix) :]
        if any(part in ("", ".", "..") for part in rel.split("/")):
            return None
        here = await facts.read(path)
        if here is None:
            return None
        there = await facts.read(posixpath.join(baseline, rel))
        return here if there == here else None
    return None


async def _workspace_roots(
    facts: _Facts, workspace: str | None, aliases: tuple[str, ...]
) -> list[str]:
    if not workspace:
        return []
    roots = [posixpath.normpath(workspace), *(posixpath.normpath(a) for a in aliases)]
    physical = await facts.resolve(posixpath.normpath(workspace))
    if physical:
        roots.append(physical)
    return roots


async def remove_conftests(
    facts: _Facts,
    changes: _Changes,
    verifier_dir: str,
    keep: tuple[str, ...],
    workspace: str | None = None,
    workspace_aliases: tuple[str, ...] = (),
    baseline: str | None = None,
) -> None:
    prunes = ["/proc", "/sys", "/dev", verifier_dir, *keep]

    def pruned(path: str) -> bool:
        return any(_find_path(p, path) for p in prunes)

    mounts = await _mount_points(facts)
    matches = [
        (path, entry)
        for path, entry in await _walk_tree(facts, "/", "conftest.py", pruned, mounts)
        if not _find_path("/verifier/*", path) and not _find_path("/tests/*", path)
    ]
    roots = await _workspace_roots(facts, workspace, workspace_aliases)
    cache: dict[str, dict[str, set[str]]] = {}
    # Every decision before any change: the matches include the pre-agent
    # copy's own conftest.py files.
    doomed: list[tuple[str, dict[str, Any]]] = []
    kept: list[tuple[str, bytes]] = []
    for path, entry in matches:
        if await package_owned(facts, path, entry, roots, cache):
            continue
        data = await unchanged_in_workspace(facts, path, entry, roots, baseline)
        if data is None:
            doomed.append((path, entry))
        else:
            kept.append((path, data))
    for path, data in kept:
        # A new root-owned file renamed over the name, as for a kept hook: no
        # descriptor a kept service holds reaches what the verifier runs.
        real_dir, name = posixpath.split(path)
        try:
            await changes.write_fresh(real_dir, name, ("f", data, 0o644))
        except _ReadOnly:
            continue
        except _Refused:
            entry = await facts.lstat(path)
            if entry is not None:
                doomed.append((path, entry))
    await _remove_matches(changes, doomed)


async def remove_tmp_py(
    facts: _Facts,
    changes: _Changes,
    workspace: str | None,
    aliases: tuple[str, ...],
) -> None:
    prunes: list[str] = []
    if workspace:
        physical = await facts.resolve(posixpath.normpath(workspace))
        entry = await facts.lstat(physical) if physical else None
        if entry is None or entry.get("type") != "dir":
            # cd -P "$workspace" || exit 1: the step fails.
            raise HookCleanupError(
                f"the workspace {workspace} is not a folder; the clean-up stops"
            )
        assert physical is not None
        prunes = [
            posixpath.normpath(workspace),
            physical,
            *(
                posixpath.normpath(a)
                for a in aliases
                if posixpath.normpath(a) != posixpath.normpath(workspace)
            ),
        ]

    exact = set(prunes)

    def pruned(path: str) -> bool:
        # The shell form escapes the glob alphabet: each -path is one path.
        return path in exact

    mounts = await _mount_points(facts)
    for top in ("/tmp", "/var/tmp"):
        entry = await facts.lstat(top)
        if entry is None or entry.get("type") != "dir" or pruned(top):
            continue
        # Without a workspace today's purge is ``find ... -delete``.
        await _remove_matches(
            changes,
            await _walk_tree(facts, top, "*.py", pruned, mounts),
            empty_folders=not workspace,
        )


# Trees the agent user may write whatever their owner says (the freeze,
# before this step, gives the workspace to root except files with another
# name).
ROOT_MADE_EXCLUDED = ("/tmp", "/var/tmp", "/home", "/logs", "/testbed")


def _root_alone(entry: Mapping[str, Any] | None) -> bool:
    return (
        entry is not None
        and int(entry.get("uid", -1)) == 0
        and not int(entry.get("mode") or 0) & 0o022
    )


async def _root_tree(facts: _Facts, path: str, depth: int = 0) -> bool:
    entry = await facts.lstat(path)
    if not _root_alone(entry):
        return False
    assert entry is not None
    if entry.get("type") == "file":
        # Another name (nlink > 1) may be one the agent user could reach; the
        # freeze leaves such a file's owner as it is, so one the agent user
        # owned is refused above. A missing count counts as more.
        return int(entry.get("nlink") or 2) == 1
    if entry.get("type") != "dir" or depth >= _MAX_DEPTH:
        return False
    rows = await facts.listdir(path)
    if rows is None or len(rows) > _MAX_NAMES:
        return False
    for name, _ in rows:
        if not await _root_tree(facts, posixpath.join(path, name), depth + 1):
            return False
    return True


async def root_made(
    facts: _Facts, real_dir: str, name: str, excluded: Iterable[str]
) -> bool:
    """HOOK_RULE's ``_bf_root_made``: the hook *name* in *real_dir* (a path
    with no link in it) is a file with one name or a folder of such files,
    no link anywhere,
    owned by root and closed to group and others, and so is every folder
    from *real_dir* up to ``/``, none in *excluded* (ACC-RUN2). Asked only
    in a trial with a sandbox user: its agent cannot add or change such an
    entry, so root wrote it during the turn (the task's solution, a service
    of the task, a package it installed)."""
    here = posixpath.normpath(real_dir)
    if any((here + "/").startswith(e.rstrip("/") + "/") for e in excluded):
        return False
    folder = here
    while True:
        entry = await facts.lstat(folder)
        if not _root_alone(entry) or (entry or {}).get("type") != "dir":
            return False
        if folder == "/":
            break
        folder = posixpath.dirname(folder)
    return await _root_tree(facts, posixpath.join(here, name))


async def clean_site_hooks(
    facts: _Facts,
    changes: _Changes,
    inputs: Inputs,
    workspace: str | None,
    *,
    agent_user: bool = False,
    workspace_aliases: tuple[str, ...] = (),
) -> None:
    """``_CLEANUP_SITE_SCRIPT``'s policy, from facts. Hooks that must go and
    cannot are left in ``changes.failed``. *agent_user*: the trial has a
    sandbox user, so a new or changed hook only root could have written
    (:func:`root_made`) is kept, written again as a fresh root-owned copy."""
    paths = await read_records(facts)
    dirs = set(paths["path"] + paths["user_site"])
    # glob.glob('/root/.local/lib/python3*/site-packages')
    globbed = set(await facts.glob_site("/root/.local", "site-packages"))
    prefixes: list[str] = []
    for search in (inputs.path, inputs.image_path):
        for p in search.split(":"):
            exe = posixpath.join(p, "python3")
            if (
                not posixpath.isabs(p)
                or p.startswith("/opt/runtime/")
                or not await facts.isfile(exe)
            ):
                continue
            prefixes.append(
                posixpath.dirname(posixpath.dirname(await facts.realpath(exe)))
            )
            bindir = posixpath.dirname(posixpath.abspath(exe))
            for cfg in (bindir, posixpath.dirname(bindir)):
                if await facts.isfile(posixpath.join(cfg, "pyvenv.cfg")):
                    prefixes.append(cfg)
                    break
            break
    prefix_sites: set[str] = set()
    for prefix in prefixes:
        for last in ("site-packages", "dist-packages"):
            prefix_sites.update(await facts.glob_site(prefix, last))
    prefix_sites = {posixpath.normpath(p) for p in prefix_sites}
    globbed |= prefix_sites
    dirs |= globbed
    w = (await facts.realpath(workspace)) + "/" if workspace else ""
    recorded = recorded_hooks(paths)
    sites = site_folders(paths) | set(paths["user_site"]) | globbed
    more, more_sites = await search_inputs(facts, inputs.values, inputs.cwd)
    sites |= more_sites
    dirs |= more | more_sites
    excluded = [
        await facts.realpath(p)
        for p in (workspace, *workspace_aliases, *ROOT_MADE_EXCLUDED)
        if p
    ]

    async def keep_or_remove(
        real_dir: str, name: str, entry: Mapping[str, Any], want: str | None
    ) -> None:
        kept = None
        if want is not None:
            kept = await node(facts, real_dir, name)
            if not (
                kept is not None
                and digest(kept) == want
                and not (kept[0] == "l" and kept[2] is None)
            ):
                kept = None
        if (
            kept is None
            and agent_user
            and await root_made(facts, real_dir, name, excluded)
        ):
            kept = await node(facts, real_dir, name)
            if kept is not None and kept[0] != "l":
                facts.report.root_made.append(posixpath.join(real_dir, name))
            else:
                kept = None
        if kept is not None:
            try:
                await changes.write_fresh(real_dir, name, kept)
                return
            except _ReadOnly:
                return
            except _Refused:
                pass
            entry = await facts.lstat(posixpath.join(real_dir, name)) or {}
            if not entry:
                return
        await changes.remove(real_dir, name, entry)

    async def clean(real_dir: str, known: Mapping[str, str] | None, site: bool) -> None:
        rows = await facts.listdir(real_dir)
        if rows is None:
            # A folder test.sh's Python searches that cannot be listed:
            # today's os.listdir raises and the step fails.
            changes.failed.append(real_dir)
            return
        for f, entry in rows:
            try:
                if f == "__pycache__":
                    if entry.get("type") == "link":
                        await changes.remove(real_dir, f, entry)
                        continue
                    if entry.get("type") != "dir":
                        raise _Refused(posixpath.join(real_dir, f), "not a folder")
                    cache = posixpath.join(real_dir, f)
                    inner = await facts.listdir(cache)
                    if inner is None:
                        raise _Refused(cache, "cannot be listed")
                    for n, kid in inner:
                        if is_hook(n, False):
                            await changes.remove(cache, n, kid)
                elif is_hook(f, site):
                    await keep_or_remove(
                        real_dir, f, entry, None if known is None else known.get(f)
                    )
            except _Unaddressable:
                changes.failed.append(f)
            except _Refused:
                changes.failed.append(f)

    async def visit(d: str) -> None:
        known = recorded.get(d)
        if known is None and d.startswith(("/usr/lib", "/usr/local/lib")):
            return
        if known is not None and (
            await facts.islink(d)
            or (await facts.lexists(d) and not await facts.isdir(d))
        ):
            parent = await facts.open_dir(posixpath.dirname(d))
            if parent is None:
                return
            name = posixpath.basename(d)
            try:
                here = await node(facts, parent, name)
                if here is None or here[0] != "l" or digest(here) != known.get(""):
                    entry = await facts.lstat(posixpath.join(parent, name))
                    if entry is not None:
                        await keep_or_remove(parent, name, entry, known.get(""))
                    return
            except (_Unaddressable, _Refused):
                changes.failed.append(d)
                return
        real = await facts.open_dir(d)
        if real is None:
            return
        spare = (
            bool(w)
            and known is None
            and d in prefix_sites
            and ((await facts.realpath(d)) + "/").startswith(w)
        )
        await clean(real, known, d in sites and not spare)

    visited: dict[str, bool] = {}
    for d in sorted(set(recorded) | dirs):
        visited[d] = d in sites
        await visit(d)
    for _ in range(64):
        named: set[str] = set()
        named_sites: set[str] = set()
        for d in sorted(sites):
            await pth_folders(facts, d, named, named_sites)
        sites |= named_sites
        todo = sorted(
            d
            for d in named | named_sites
            if d not in visited or (d in sites and not visited[d])
        )
        if not todo:
            break
        for d in todo:
            visited[d] = d in sites
            await visit(d)


async def run(
    call: Call,
    inputs: Inputs,
    *,
    cleanup_conftests: bool,
    workspace: str | None,
    workspace_aliases: tuple[str, ...],
    verifier_dir: str,
    keep: tuple[str, ...] = (),
    deadline: float,
    agent_user: bool = False,
    baseline: str | None = None,
) -> Report:
    """The whole clean-up over *call* (one ``bfx serve`` session), by
    *deadline* (the event loop's clock). Raises :class:`HookCleanupError`
    when it did not complete; returns what it skipped without failing."""
    facts = _Facts(call, deadline)
    changes = _Changes(facts)
    try:
        if cleanup_conftests:
            await remove_conftests(
                facts,
                changes,
                verifier_dir,
                keep,
                workspace,
                workspace_aliases,
                baseline if agent_user else None,
            )
        await remove_tmp_py(facts, changes, workspace, workspace_aliases)
        await clean_site_hooks(
            facts,
            changes,
            inputs,
            workspace,
            agent_user=agent_user,
            workspace_aliases=workspace_aliases,
        )
    except (BfxAmbiguous, BfxProtocolError) as exc:
        raise HookCleanupError(
            f"bfx could not account for a request ({exc}); the clean-up stops"
        ) from exc
    except BfxError as exc:
        raise HookCleanupError(f"bfx refused a request: {exc}") from exc
    if changes.unaddressable:
        raise HookCleanupError(
            f"{len(changes.unaddressable)} path(s) the clean-up must change "
            f"cannot be given to bfx, e.g. {changes.unaddressable[0]!r}"
        )
    if changes.failed:
        raise HookCleanupError(
            f"could not remove {len(changes.failed)} Python start-up hook(s), "
            f"e.g. {changes.failed[0]!r}"
        )
    return facts.report
