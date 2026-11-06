"""The protected verifier runtime: the verifier's pytest plugins in C8's tree.

Four review rounds (V2-REDESIGN-R1 to R4) found ways around authenticating
plugin bytes that sit where the agent can write, so the verifier's plugins
come from BenchFlow's protected runtime tree (C8, ``benchflow._runtime_tree``)
instead. Before the agent's turn BenchFlow fills
``VERIFIER_DIR`` (``/opt/runtime/verifier``, which C8 reserves and makes
root's) from wheels it fetched on the host and checked against the sha256
PyPI lists for each file:

- ``site/``: the plugins the verifier will load (the task's pinned releases,
  ``[verifier] pytest_plugin_releases``, and pytest-json-ctrf when
  test.sh asks for ``--ctrf``), their whole dependency closure, and BenchFlow's
  own pytest when the sandbox has C8's kit Python;
- ``bin/pytest`` (``$BENCHFLOW_PYTEST``, the protected runner): C8's
  ``KIT_PYTHON`` (the image's interpreter and a copy of its standard library,
  ``-I -S``) with ``-B``, ``sys.path`` the site folder, the kit's standard library
  and the tests folder alone, run from ``/``. At every launch it first checks
  the kit and this tree root's alone and byte for byte against their
  manifests (no file added, none changed); in the process it serves every
  standard-library and closure name from those folders whatever
  ``sys.path`` says later, and lets pytest register only one allowlist of
  plugins: the task's pinned plugin releases (by ``-p`` or a conftest's
  ``pytest_plugins``, each once), pytest's own, and modules of the tests
  folder, plus unchanged image helpers authenticated against the recorded
  digests of their pre-agent workspace copy;
- ``manifest.json`` and ``SHA256SUMS``.

In whatever other pytest test.sh starts, the plugin guard loads the same
plugins from the site folder, compiling their sources, and serves the
closure's names from there. Nothing here reads test.sh beyond the existing
``--ctrf`` inference, which only decides what to offer.

When the tree cannot be built the trial stops before the agent runs: PyPI
or the network failing is the sandbox's startup failure, retried; a pin
PyPI does not have, a dependency the task did not pin, or a wheel no
Python here can load is the task's, with what to change. A verifier in
another service (#248) gets that service's own tree and runner. A separate
verifier first prepares its own C8 kit and then builds its own tree from
its own image; neither uses the solver sandbox's Python.
"""

from __future__ import annotations

import contextvars
import hashlib
import io
import json
import logging
import os
import re
import shlex
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from benchflow._root_only import root_only_tree_check
from benchflow._runtime_tree import KIT, KIT_BIN, KIT_PYTHON, KIT_SH, VERIFIER_DIR
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.trust.state import TRUSTED_TRANSFER, admit

logger = logging.getLogger(__name__)

RUNTIME = VERIFIER_DIR
SITE = f"{RUNTIME}/site"
RUNNER = f"{RUNTIME}/bin/pytest"
RUN_PY = f"{RUNTIME}/run.py"
MANIFEST = f"{RUNTIME}/manifest.json"
SUMS = f"{RUNTIME}/SHA256SUMS"
# Where the runner looks for byte code: never made, under a root-only folder,
# so Python compiles every module from the checked sources and reads no cache.
NO_CACHE = f"{RUNTIME}/no-cache"
# The running pytest's own packages: in test.sh's own pytest the guard never
# serves these from the tree (pytest is already imported from the image).
RUNNING_PYTEST = ("pytest", "_pytest", "pluggy", "py")

# BenchFlow's own pytest for the protected runner, and what it needs on the
# Pythons it supports (3.9 and later). A task's pin of the same name wins.
BENCHFLOW_PYTEST = (
    "pytest==8.4.2",
    "pluggy==1.6.0",
    "iniconfig==2.1.0",
    "packaging==25.0",
    "pygments==2.19.2",
    "exceptiongroup==1.3.0",
    "tomli==2.2.1",
    "typing-extensions==4.15.0",
)
# The ctrf release offered when test.sh asks for --ctrf and the task pins
# none: the one Terminal-Bench 2's verifiers install.
CTRF = "pytest-json-ctrf==0.3.5"

# ``offline`` keeps to the host's cache (the test suite sets it).
_ENV = "BENCHFLOW_PLUGIN_RELEASES"
# Local wheel folders an offline host may migrate a pre-0.8.4 cache from.
_WHEELHOUSE_ENV = "BENCHFLOW_PLUGIN_WHEELHOUSE"
_WHEELHOUSES: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "verifier_runtime_wheelhouses", default=()
)
_TIMEOUT_SEC = 10.0
_ATTEMPTS = 3
_MAX_WHEEL_BYTES = 20 << 20
_MAX_MEMBERS = 5000
_PIN_RE = re.compile(
    r"([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==([0-9][0-9A-Za-z.!+_-]*)"
)
_MEMBER_RE = re.compile(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*")


class RuntimeUnavailable(RuntimeError):
    """PyPI or the network could not serve a wheel: the sandbox's startup
    failure, retried like one (``_utils/scoring.py``)."""

    def __init__(self, why: str):
        super().__init__(
            "sandbox startup failed: the protected verifier runtime could not "
            f"be prepared before the agent's turn: {why}"
        )


class RuntimePinError(RuntimeError):
    """The task's pins cannot make the tree: the task's to change."""

    def __init__(self, why: str):
        super().__init__(
            "the task's pytest plugin releases cannot be prepared before the "
            f"agent's turn: {why}. Pin every release the verifier's plugins "
            "need in [verifier] pytest_plugin_releases (name==version), or "
            "install them into the image"
        )


@dataclass
class Interpreter:
    """What the probe read from the image's root-only Python."""

    exe: str
    version: tuple[int, int]
    machine: str
    glibc: tuple[int, int] | None
    musl: tuple[int, int] | None
    markers: dict[str, str]
    stdlib: list[str] = field(default_factory=list)
    # Whether this is C8's kit Python, which the protected runner runs.
    runner: bool = True
    image_files: dict[str, str] = field(default_factory=dict)
    workspace: str = ""
    stdlib_names: list[str] = field(default_factory=list)
    image_sites: list[str] = field(default_factory=list)
    image_top: list[str] = field(default_factory=list)


@dataclass
class Wheel:
    name: str
    version: str
    filename: str
    sha256: str
    data: bytes


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "benchflow" / "verifier-runtime"


def _offline() -> bool:
    return os.environ.get(_ENV) == "offline"


def parse_pin(pin: str) -> tuple[str, str]:
    match = _PIN_RE.fullmatch(pin)
    if match is None:
        raise RuntimePinError(f"{pin!r} is not a pinned release (name==version)")
    return canonicalize_name(match.group(1)), match.group(2)


def supported_tags(interpreter: Interpreter | None) -> list[Tag]:
    """The wheel tags *interpreter* loads; pure wheels only without one."""
    if interpreter is None:
        return [Tag("py3", "none", "any")]
    major, minor = interpreter.version
    arch = interpreter.machine
    platforms = []
    if interpreter.glibc is not None:
        glibc_major, glibc_minor = interpreter.glibc
        for low in range(glibc_minor, 4, -1):
            platforms.append(f"manylinux_{glibc_major}_{low}_{arch}")
            legacy = {17: "manylinux2014", 12: "manylinux2010", 5: "manylinux1"}
            if glibc_major == 2 and low in legacy:
                platforms.append(f"{legacy[low]}_{arch}")
    if interpreter.musl is not None:
        musl_major, musl_minor = interpreter.musl
        platforms += [
            f"musllinux_{musl_major}_{low}_{arch}" for low in range(musl_minor, -1, -1)
        ]
    platforms.append(f"linux_{arch}")
    tags = list(cpython_tags((major, minor), platforms=platforms))
    tags += list(
        compatible_tags(
            (major, minor), interpreter=f"cp{major}{minor}", platforms=platforms
        )
    )
    return tags


def _http_get(url: str):
    import httpx

    from benchflow._utils.log_context import quiet_own_requests

    # Through the Python API a run logs at its caller's level: one "HTTP
    # Request" line per PyPI fetch is no step of the run. Only this fetch's
    # line is dropped; the host's own httpx lines are left as they were.
    with quiet_own_requests():
        return httpx.get(url, timeout=_TIMEOUT_SEC, follow_redirects=True)


def _release_files(name: str, version: str) -> list[dict[str, Any]]:
    """PyPI's file list for one release, from the cache or PyPI.

    A release's file list is cached only whole, as PyPI served it; a 404 is
    the task's (no such release), anything else the network's, retried.
    """
    cached = _cache_dir() / "index" / f"{name}-{version}.json"
    try:
        files = json.loads(cached.read_text())
        if isinstance(files, list):
            return files
    except (OSError, ValueError):
        pass
    if _offline():
        migrated = _migrate_legacy(name, version)
        if migrated:
            return migrated
        raise RuntimeUnavailable(
            f"{name}=={version} is not in the release cache and {_ENV}=offline "
            "keeps PyPI from being asked; run once with PyPI reachable, or name "
            f"a folder of its wheels in {_WHEELHOUSE_ENV}"
        )
    error: Exception | None = None
    for _ in range(_ATTEMPTS):
        try:
            response = _http_get(f"https://pypi.org/pypi/{name}/{version}/json")
            if response.status_code == 404:
                raise RuntimePinError(f"PyPI has no release {name}=={version}")
            response.raise_for_status()
            files = response.json().get("urls")
            if not isinstance(files, list):
                raise ValueError("no file list")
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_name(f".{cached.name}.{uuid.uuid4().hex}")
            tmp.write_text(json.dumps(files))
            tmp.replace(cached)
            return files
        except RuntimePinError:
            raise
        except Exception as exc:
            error = exc
    raise RuntimeUnavailable(_fetch_failure(f"{name}=={version}'s file list", error))


def _fetch_failure(
    what: str, error: Exception | str | None, *, verb: str = "read"
) -> str:
    """Why ``what`` could not be had, and, when PyPI could not be reached at
    all, that the network is needed and how to do without it (DFR-p5-6)."""
    import httpx

    reason = error if isinstance(error, str) else type(error).__name__
    if not isinstance(error, httpx.TransportError):
        return f"{what} could not be {verb} ({reason})"
    cache = str(_cache_dir())
    home = str(Path.home())
    if cache.startswith(home + os.sep):
        cache = "~" + cache[len(home) :]
    return (
        f"no network to PyPI ({reason}), and this host's cache has no checked "
        f"copy of {what}: run a task once where PyPI is reachable (HTTPS_PROXY "
        f"is honored) to fill {cache}; later runs need no network"
    )


def _legacy_cache_dir() -> Path:
    """Where earlier BenchFlow kept what it checked of each
    release: every PyPI wheel's files and their sha256, no wheel bytes."""
    return Path.home() / ".cache" / "benchflow" / "plugin-releases"


def _legacy_table(name: str, version: str) -> dict[str, list[dict[str, str]]]:
    """The pre-0.8.4 cache's files of every PyPI wheel of *name*==*version*
    (format 2), from a pin's own entry or a vouched distribution's."""
    table: dict[str, list[dict[str, str]]] = {}
    for key in (f"pinned-{name}-{version}", name):
        try:
            data = json.loads((_legacy_cache_dir() / f"{key}.json").read_text())
            if data.get("format") != 2:
                continue
            for info, releases in data["releases"].items():
                if canonicalize_name(str(info).split("-", 1)[0]) != name:
                    continue
                if not str(info).startswith(f"{str(info).split('-', 1)[0]}-{version}."):
                    continue
                table.setdefault(str(info), []).extend(dict(r) for r in releases)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return table


def _wheel_files(data: bytes) -> tuple[str, dict[str, str]] | None:
    """The dist-info name and the files (path: sha256) of a wheel, as the
    pre-0.8.4 cache recorded them (its RECORD left out)."""
    files: dict[str, str] = {}
    infos = set()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as wheel:
            members = wheel.infolist()
            if len(members) > _MAX_MEMBERS:
                return None
            for member in members:
                if member.is_dir():
                    continue
                name = member.filename
                if not _MEMBER_RE.fullmatch(name) or ".." in name.split("/"):
                    return None
                top = name.split("/", 1)[0]
                if top.endswith(".data"):
                    return None
                if top.endswith(".dist-info"):
                    infos.add(top)
                    if name == top + "/RECORD":
                        continue
                files[name] = hashlib.sha256(wheel.read(member)).hexdigest()
    except (zipfile.BadZipFile, OSError, ValueError):
        return None
    if len(infos) != 1:
        return None
    return infos.pop(), files


def _migrate_legacy(name: str, version: str) -> list[dict[str, Any]]:
    """Offline, the release's wheels a pre-0.8.4 cache vouched for (V2-R4-5).

    That cache kept each PyPI wheel's files and their sha256, not the wheel:
    a local wheel (in the folders :data:`_WHEELHOUSES` names) whose every
    file matches one recorded wheel exactly is that wheel. It is kept in the
    new cache by its own sha256, with the file list, so the next offline
    run needs no migration. Nothing unmatched is ever used.
    """
    table = _legacy_table(name, version)
    if not table:
        return []
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for folder in _WHEELHOUSES.get():
        try:
            candidates = sorted(Path(folder).rglob("*.whl"))[:500]
        except OSError:
            continue
        for path in candidates:
            try:
                wheel_name, wheel_version, _, _ = parse_wheel_filename(path.name)
                if canonicalize_name(wheel_name) != name or str(wheel_version) != str(
                    Version(version)
                ):
                    continue
                if path.stat().st_size > _MAX_WHEEL_BYTES:
                    continue
                data = path.read_bytes()
            except Exception:
                continue
            found = _wheel_files(data)
            if found is None or found[1] not in table.get(found[0], []):
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            cached = _cache_dir() / "wheels" / f"{digest}.whl"
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_name(f".{cached.name}.{uuid.uuid4().hex}")
            tmp.write_bytes(data)
            tmp.replace(cached)
            entries.append(
                {
                    "filename": path.name,
                    "packagetype": "bdist_wheel",
                    "digests": {"sha256": digest},
                    "url": f"https://files.pythonhosted.org/migrated/{path.name}",
                }
            )
    if entries:
        index = _cache_dir() / "index" / f"{name}-{version}.json"
        index.parent.mkdir(parents=True, exist_ok=True)
        tmp = index.with_name(f".{index.name}.{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(entries))
        tmp.replace(index)
        logger.info(
            "verifier runtime: %s==%s migrated offline from %d local wheel(s) "
            "the earlier release cache vouched for",
            name,
            version,
            len(entries),
        )
    return entries


def _wheel_bytes(entry: dict[str, Any]) -> bytes:
    """One wheel's bytes, checked against the sha256 PyPI lists for it.

    Cached by that sha256 only once checked, so a failed or partial download
    is never kept; a missing one is fetched again (retried here).
    """
    digest = (entry.get("digests") or {}).get("sha256", "")
    url = entry.get("url", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or not url.startswith("https://"):
        raise RuntimeUnavailable(f"PyPI lists no sha256 for {entry.get('filename')}")
    cached = _cache_dir() / "wheels" / f"{digest}.whl"
    try:
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == digest:
            return data
    except OSError:
        pass
    if _offline():
        raise RuntimeUnavailable(
            f"{entry.get('filename')} is not in the release cache and "
            f"{_ENV}=offline keeps PyPI from being asked"
        )
    error: Exception | str = "no attempt"
    for _ in range(_ATTEMPTS):
        try:
            response = _http_get(url)
            response.raise_for_status()
            data = response.content
            if len(data) > _MAX_WHEEL_BYTES:
                raise RuntimePinError(f"{entry.get('filename')} is too large")
            if hashlib.sha256(data).hexdigest() != digest:
                error = "sha256 differs from PyPI's"
                continue
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_name(f".{cached.name}.{uuid.uuid4().hex}")
            tmp.write_bytes(data)
            tmp.replace(cached)
            return data
        except RuntimePinError:
            raise
        except Exception as exc:
            error = exc
    raise RuntimeUnavailable(
        _fetch_failure(f"{entry.get('filename')}", error, verb="fetched")
    )


def _pick_wheel(name: str, version: str, tags: list[Tag]) -> tuple[dict[str, Any], str]:
    """The release's wheel this Python prefers, by tag order."""
    rank = {tag: index for index, tag in enumerate(tags)}
    best = None
    for entry in _release_files(name, version):
        filename = entry.get("filename", "")
        if entry.get("packagetype") != "bdist_wheel" or entry.get("yanked"):
            continue
        try:
            _, _, _, wheel_tags = parse_wheel_filename(filename)
        except Exception:
            continue
        score = min((rank[t] for t in wheel_tags if t in rank), default=None)
        if score is not None and (best is None or score < best[0]):
            best = (score, entry)
    if best is None:
        raise RuntimePinError(
            f"{name}=={version} has no wheel the verifier's Python can load"
        )
    return best[1], best[1]["filename"]


def wheel_metadata(data: bytes) -> tuple[list[str], dict[str, str], str | None]:
    """A wheel's Requires-Dist lines, its pytest11 entry points and Requires-Python."""
    requires: list[str] = []
    entry_points: dict[str, str] = {}
    python = None
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        infos = {n.split("/", 1)[0] for n in wheel.namelist() if ".dist-info/" in n}
        if len(infos) != 1:
            raise RuntimePinError("a wheel without one dist-info folder")
        info = infos.pop()
        metadata = wheel.read(f"{info}/METADATA").decode("utf-8", "replace")
        for line in metadata.splitlines():
            if not line.strip():
                break
            if line.startswith("Requires-Dist:"):
                requires.append(line.split(":", 1)[1].strip())
            elif line.startswith("Requires-Python:"):
                python = line.split(":", 1)[1].strip()
        if f"{info}/entry_points.txt" in wheel.namelist():
            section = None
            text = wheel.read(f"{info}/entry_points.txt").decode("utf-8", "replace")
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("["):
                    section = line.strip("[]").strip()
                elif section == "pytest11" and "=" in line:
                    # The whole object reference, ``module:attribute.chain``:
                    # pytest registers that object, not the module (V2-R6-C2).
                    # Extras (``[name]``) select nothing to import.
                    key, value = line.split("=", 1)
                    module, _, attribute = value.split("[", 1)[0].partition(":")
                    reference = module.strip()
                    if attribute.strip():
                        reference += ":" + attribute.strip()
                    entry_points[key.strip()] = reference
    return requires, entry_points, python


def _marker_environment(interpreter: Interpreter | None) -> dict[str, str]:
    if interpreter is not None and interpreter.markers:
        return dict(interpreter.markers)
    # No Python of the image's to read them from: a Linux CPython 3.12.
    environment = dict(default_environment())
    environment.update(
        python_version="3.12",
        python_full_version="3.12.0",
        sys_platform="linux",
        platform_system="Linux",
        os_name="posix",
        implementation_name="cpython",
        platform_python_implementation="CPython",
        platform_machine="x86_64",
    )
    return environment


@dataclass
class Resolution:
    wheels: list[Wheel]
    plugins: dict[str, str]  # pytest11 name -> module, from plugin releases
    plugin_top: list[str]  # top-level names of the plugin releases: the allowlist
    runner: bool
    # Every top-level name the closure installs: served from the site folder
    # alone, in the runner and (but for the running pytest's own) in the guard
    # (V2-R4-2: a dependency of a plugin must not come from the workspace).
    protected_top: list[str] = field(default_factory=list)


def resolve(
    pins: list[str],
    *,
    ctrf: bool,
    interpreter: Interpreter | None,
    wheelhouses: tuple[str, ...] = (),
) -> Resolution:
    """Every wheel the tree needs: the plugin releases, what they need, and
    BenchFlow's pytest when the runner can run, each pinned and checked.

    Offline, a release the host's cache lacks may come from a local wheel
    that a pre-0.8.4 cache vouched for (*wheelhouses*, V2-R4-5); when
    BenchFlow's own pytest cannot be had that way, the tree holds the
    plugins alone and has no runner, as before the runner existed."""
    token = _WHEELHOUSES.set(tuple(wheelhouses))
    try:
        try:
            return _resolve(pins, ctrf=ctrf, interpreter=interpreter)
        except RuntimeUnavailable:
            if not (_offline() and interpreter is not None and interpreter.runner):
                raise
            logger.warning(
                "verifier runtime: BenchFlow's pytest is not in the offline "
                "cache; building the plugins without the protected runner"
            )
            return _resolve(
                pins, ctrf=ctrf, interpreter=replace(interpreter, runner=False)
            )
    finally:
        _WHEELHOUSES.reset(token)


def _resolve(
    pins: list[str], *, ctrf: bool, interpreter: Interpreter | None
) -> Resolution:
    tags = supported_tags(interpreter)
    environment = _marker_environment(interpreter)
    versions: dict[str, str] = {}
    for pin in BENCHFLOW_PYTEST:
        name, version = parse_pin(pin)
        versions[name] = version
    roots: list[str] = []
    if ctrf:
        name, version = parse_pin(CTRF)
        versions[name] = version
        roots.append(name)
    for pin in pins:
        name, version = parse_pin(pin)
        versions[name] = version
        roots.append(name)
    pytest_set = {parse_pin(pin)[0] for pin in BENCHFLOW_PYTEST}
    runner = interpreter is not None and interpreter.runner
    if runner:
        roots = ["pytest", *roots]
    wheels: dict[str, Wheel] = {}
    plugins: dict[str, str] = {}
    plugin_names: set[str] = set()
    queue = [(root, None) for root in dict.fromkeys(roots)]
    while queue:
        name, needed_by = queue.pop(0)
        if name in wheels:
            continue
        if name not in versions:
            raise RuntimePinError(
                f"{needed_by} needs {name}, which the task does not pin"
            )
        version = versions[name]
        entry, filename = _pick_wheel(name, version, tags)
        data = _wheel_bytes(entry)
        requires, entry_points, python = wheel_metadata(data)
        if python and interpreter is not None:
            from packaging.specifiers import SpecifierSet

            full = environment.get("python_full_version", "")
            try:
                if not SpecifierSet(python).contains(full, prereleases=True):
                    raise RuntimePinError(
                        f"{name}=={version} needs Python {python}, the "
                        f"verifier's is {full}"
                    )
            except InvalidVersion:
                pass
        wheels[name] = Wheel(name, version, filename, entry["digests"]["sha256"], data)
        if name not in pytest_set or name in {parse_pin(p)[0] for p in pins}:
            plugin_names.add(name)
            plugins.update(entry_points)
        for line in requires:
            try:
                requirement = Requirement(line)
            except InvalidRequirement as exc:
                raise RuntimePinError(f"{name}=={version}: {exc}") from exc
            if requirement.marker is not None and not requirement.marker.evaluate(
                {**environment, "extra": ""}
            ):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency in {"pytest", "pluggy"} and not runner:
                # The pytest that test.sh starts provides itself.
                continue
            pinned = versions.get(dependency)
            if pinned is None:
                raise RuntimePinError(
                    f"{name}=={version} needs {requirement}, which the task does not pin"
                )
            if requirement.specifier and not requirement.specifier.contains(
                Version(pinned), prereleases=True
            ):
                raise RuntimePinError(
                    f"{name}=={version} needs {requirement}; the pin is {pinned}"
                )
            queue.append((dependency, f"{name}=={version}"))
    top: set[str] = set()
    for name in plugin_names:
        top |= _top_level(wheels[name].data)
    protected: set[str] = set()
    for wheel in wheels.values():
        protected |= _top_level(wheel.data)
    return Resolution(
        wheels=list(wheels.values()),
        plugins=plugins,
        plugin_top=sorted(top),
        runner=runner,
        protected_top=sorted(protected),
    )


def _top_level(data: bytes) -> set[str]:
    names = set()
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        for member in wheel.namelist():
            first = member.split("/", 1)[0]
            if first.endswith(".dist-info"):
                continue
            if first.endswith(".data"):
                scheme, _, inner = member.split("/", 1)[1].partition("/")
                if scheme not in ("purelib", "platlib") or not inner:
                    continue
                member = inner
                first = member.split("/", 1)[0]
            if "/" in member:
                names.add(first)
            elif member.endswith(".py"):
                names.add(first[:-3])
            elif ".cpython-" in member or member.endswith((".so", ".pyd")):
                names.add(first.split(".", 1)[0])
    return names


_SCAN_MAX_FILES = 5000
_SCAN_MAX_BYTES = 4 * 1024 * 1024


# Calls that import the module their first, literal argument names, as the
# tests make them: pytest.importorskip, importlib.import_module, __import__.
_STRING_IMPORTS = frozenset({"importorskip", "import_module", "__import__"})


def scan_tests(folder: Path | str | None) -> tuple[list[str], list[str]]:
    """``(module names, unread files)`` of the task's trusted tests.

    The names are those the tests import themselves: every ``import a.b``
    (``a`` and ``a.b``), absolute ``from a.b import c`` (``a``, ``a.b`` and
    ``a.b.c``), and ``importorskip("a.b")``, ``import_module("a.b")`` or
    ``__import__("a.b")`` with a literal argument (L1), in the ``.py`` files
    of the task's tests folder, the copy BenchFlow uploads root-owned. Read
    as text, never run (V2PT-R2-1). A file that does not parse adds nothing;
    one too large or too deeply nested to read is listed as unread, so a
    refusal can say so instead of preparation stopping (L5).
    """
    import ast

    if not folder or not Path(folder).is_dir():
        return [], []
    names: set[str] = set()
    unread: list[str] = []

    def add(dotted: str) -> None:
        parts = dotted.split(".")
        names.update(".".join(parts[: i + 1]) for i in range(len(parts)))

    root = Path(folder)
    files = sorted(p for p in root.rglob("*.py") if p.is_file())
    unread += [p.relative_to(root).as_posix() for p in files[_SCAN_MAX_FILES:]]
    for path in files[:_SCAN_MAX_FILES]:
        try:
            if path.stat().st_size > _SCAN_MAX_BYTES:
                unread.append(path.relative_to(root).as_posix())
                continue
            tree = ast.parse(path.read_bytes())
        except (OSError, SyntaxError, ValueError):
            continue
        except (RecursionError, MemoryError):
            unread.append(path.relative_to(root).as_posix())
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                add(node.module)
                for alias in node.names:
                    if alias.name != "*":
                        add(node.module + "." + alias.name)
            elif isinstance(node, ast.Call) and node.args:
                func = node.func
                called = getattr(func, "attr", None) or getattr(func, "id", None)
                first = node.args[0]
                if (
                    called in _STRING_IMPORTS
                    and isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value
                    and not first.value.startswith(".")
                ):
                    add(first.value)
    return sorted(names), sorted(unread)


def scan_tests_imports(folder: Path | str | None) -> list[str]:
    """The module names of :func:`scan_tests`."""
    return scan_tests(folder)[0]


def build_tree(
    resolution: Resolution,
    *,
    interpreter: Interpreter | None,
    tests_dir: str,
    into: Path,
    tests_imports: list[str] | None = None,
    tests_unread: list[str] | None = None,
    runner_wanted: bool = True,
) -> dict[str, Any]:
    """Lay the tree out in *into* (uploaded as is) and return its manifest.

    *tests_imports*: the module names the task's trusted tests import
    (:func:`scan_tests`), written into the runner, with *tests_unread*, the
    tests files that scan could not read. *runner_wanted* false (the task's
    tests never name ``$BENCHFLOW_PYTEST``, :func:`runner_requested`): no
    runner and no copy of the image's site folders for it; the site folder
    is the same either way."""
    site = into / "site"
    site.mkdir(parents=True)
    for wheel in resolution.wheels:
        with zipfile.ZipFile(io.BytesIO(wheel.data)) as archive:
            members = archive.infolist()
            if len(members) > _MAX_MEMBERS:
                raise RuntimePinError(f"{wheel.filename} has too many files")
            for member in members:
                name = member.filename
                if member.is_dir():
                    continue
                if not _MEMBER_RE.fullmatch(name) or ".." in name.split("/"):
                    raise RuntimePinError(f"{wheel.filename} holds {name!r}")
                first, _, rest = name.partition("/")
                if first.endswith(".data"):
                    scheme, _, inner = rest.partition("/")
                    if scheme not in ("purelib", "platlib") or not inner:
                        continue  # scripts, headers and data: not imported
                    name = inner
                if name.endswith(".dist-info/RECORD"):
                    continue
                if name.endswith(".pth"):
                    continue  # never processed: the runner runs with -S
                target = site / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
                target.chmod(0o755 if name.endswith((".so", ".pyd")) else 0o644)
    runner = (
        runner_wanted
        and resolution.runner
        and interpreter is not None
        and interpreter.runner
    )
    image_sites = (
        [
            f"{RUNTIME}/image-sites/{index}"
            for index in range(len(interpreter.image_sites))
        ]
        if runner and interpreter is not None
        else []
    )
    manifest = {
        "runtime": RUNTIME,
        "site": SITE,
        "tests": tests_dir,
        "python": KIT_PYTHON if runner else None,
        "runner": RUNNER if runner else None,
        "image_sites": image_sites,
        "plugins": resolution.plugins,
        "plugin_top": resolution.plugin_top,
        "protected_top": resolution.protected_top,
        "releases": [
            {
                "name": wheel.name,
                "version": wheel.version,
                "wheel": wheel.filename,
                "sha256": wheel.sha256,
            }
            for wheel in resolution.wheels
        ],
    }
    if runner and interpreter is not None:
        (into / "bin").mkdir()
        (into / "run.py").write_text(
            runner_source(
                site=SITE,
                stdlib=[p for p in interpreter.stdlib if p],
                tests=tests_dir,
                plugins=resolution.plugins,
                allowed_top=resolution.plugin_top,
                protected_top=resolution.protected_top,
                workspace=interpreter.workspace,
                image_files=interpreter.image_files,
                stdlib_names=interpreter.stdlib_names,
                image_sites=image_sites,
                image_top=interpreter.image_top,
                tests_imports=tests_imports,
                tests_unread=tests_unread,
            )
        )
        script = into / "bin" / "pytest"
        script.write_text(runner_script())
        script.chmod(0o755)
    (into / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    for path in into.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    sums = []
    for path in sorted(p for p in into.rglob("*") if p.is_file()):
        relative = "./" + path.relative_to(into).as_posix()
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    (into / "SHA256SUMS").write_text("".join(sums))
    (into / "SHA256SUMS").chmod(0o644)
    return manifest


def _same_files_check(
    folder: str, sums: str, what: str, *, caches_unread: bool = False
) -> str:
    """sh: every file in *folder* (but the sums file) is listed in *sums*,
    with those bytes, and no other file is there: an added module is refused
    as surely as a changed one. With *caches_unread* (C8's kit, which root's
    own uses of its Python fill with byte code), files added under a
    ``__pycache__`` folder are let be: the runner reads no cache at all
    (``-X pycache_prefix``, :data:`NO_CACHE`); listed ones must still match."""
    q = shlex.quote(folder)
    name = shlex.quote(os.path.basename(sums))
    fail = shlex.quote(f"benchflow: {what} ({folder}) changed since BenchFlow built it")
    skip = " ! -path '*/__pycache__/*'" if caches_unread else ""
    keep = " | grep -v '/__pycache__/'" if caches_unread else ""
    return (
        f"( cd {q} && sha256sum -c --quiet {name} >/dev/null 2>&1 && "
        f'[ "$(find . -type f ! -name {name} ! -name {name}.new{skip} | LC_ALL=C sort)" = '
        f"\"$(sed 's/^[0-9a-f]*  //' {name}{keep} | LC_ALL=C sort)\" ] ) "
        f"|| {{ echo {fail} >&2; false; }}"
    )


def custody_command() -> str:
    """sh: C8's kit and this tree are root's alone and hold exactly the bytes
    their manifests list. Run with PATH the kit's own bin."""
    # Each step braced: "a || { echo; false; }" joined by && would group as
    # ((a || echo) && b), printing later steps' refusals after the first.
    return " && ".join(
        "{ " + step + "; }"
        for step in (
            root_only_tree_check(KIT, "BenchFlow's protected runtime kit"),
            _same_files_check(
                KIT,
                f"{KIT}/MANIFEST",
                "BenchFlow's protected runtime kit",
                caches_unread=True,
            ),
            f"[ ! -e {NO_CACHE} ]",
            root_only_tree_check(RUNTIME, "BenchFlow's protected verifier runtime"),
            _same_files_check(RUNTIME, SUMS, "BenchFlow's protected verifier runtime"),
        )
    )


def runner_script() -> str:
    """``bin/pytest``: check custody at every launch, then start the runner
    on C8's kit Python with ``-B`` (the kit adds ``-I -S``)."""
    return (
        f"#!{KIT_SH}\n"
        "# BenchFlow's protected verifier pytest:\n"
        "# C8's kit Python and its copy of the standard library, this tree and\n"
        "# the tests folder, checked byte for byte before every run.\n"
        f"( PATH={KIT_BIN}; export PATH; {custody_command()} ) || exit 4\n"
        # A task may redirect its whole if/then block into reward.txt. Keep
        # pytest's report out of that scalar file (FIX-49 r4 advice.sh).
        "if [ /proc/self/fd/1 -ef /logs/verifier/reward.txt ]; then exec 1>&2; fi\n"
        "unset PYTEST_ADDOPTS PYTEST_PLUGINS\n"
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; export PYTEST_DISABLE_PLUGIN_AUTOLOAD\n"
        f'exec {KIT_PYTHON} -B -X pycache_prefix={NO_CACHE} {RUN_PY} "$@"\n'
    )


# The runner, as text: everything it trusts is written into it at build time.
_RUNNER = r'''"""BenchFlow's protected verifier pytest.

Started as ``python -I -S -B`` (no ``.pth`` file, no user site, no
``PYTHON*`` variable, no byte code written). Every standard-library and
closure name is served from the kit's library and this tree alone, whatever
``sys.path`` becomes; pytest may register only pytest's own plugins, the
task's pinned plugin releases (each once, by any name) and modules of the
tests folder.
"""
import hashlib
import hmac
import importlib.machinery as _machinery
import importlib.util as _util
import json
import os
import sys
import types

SITE = %(site)r
STDLIB = %(stdlib)r
TESTS = %(tests)r
PLUGINS = %(plugins)r
ALLOWED_TOP = frozenset(%(allowed_top)r)
PROTECTED_TOP = frozenset(%(protected_top)r)
STDLIB_NAMES = frozenset(%(stdlib_names)r) - PROTECTED_TOP
WORKSPACE = %(workspace)r
SNAPSHOT = %(snapshot)r
IMAGE_FILES = %(image_files)r
IMAGE_SITES = %(image_sites)r
IMAGE_TOP = frozenset(%(image_top)r)
# Module names the task's trusted tests import themselves (a static scan of
# their sources when this runner was built, V2PT-R2-1).
TESTS_IMPORTS = frozenset(%(tests_imports)r)
TESTS_UNREAD = %(tests_unread)r
PATH_OPTIONS = (
    "--ctrf", "--junitxml", "--junit-xml", "--json-report-file", "--html",
    "--rootdir", "-c", "--basetemp", "--report-log", "--resultlog",
    "--ignore", "--confcutdir", "--log-file", "--debug",
)

sys.path[:] = [SITE, *STDLIB, *IMAGE_SITES, TESTS]


def _under(path, folder):
    return path == folder or path.startswith(folder.rstrip("/") + "/")


class _Source(_machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile(self.get_data(self.path), self.path, "exec", dont_inherit=True)


class _Protected:
    """Top-level standard-library and closure names come from their own
    folders alone: a later ``sys.path.insert(0, "/app")`` in a conftest or
    test cannot put a workspace module in their place (V2-R4-2, Sol R4-1)."""

    @classmethod
    def find_spec(cls, name, path=None, target=None):
        top = name.partition(".")[0]
        if name in sys.builtin_module_names:
            return None
        if _machinery.FrozenImporter.find_spec(name) is not None:
            return None
        if top in PROTECTED_TOP:
            folders = [SITE]
        elif top in STDLIB_NAMES:
            folders = STDLIB
        else:
            return None
        if path is not None:
            folders = [p for p in path if any(
                _under(os.path.realpath(p), root) for root in folders
            )]
        spec = _machinery.PathFinder.find_spec(name, folders)
        if spec is None:
            raise ModuleNotFoundError(
                "No module named %%r in BenchFlow's protected runtime" %% name,
                name=name,
            )
        locations = ([spec.origin] if spec.origin else []) + list(spec.submodule_search_locations or [])
        roots = [SITE] if top in PROTECTED_TOP else STDLIB
        if not locations or any(not any(_under(os.path.realpath(p), root) for root in roots) for p in locations):
            raise ImportError(name + " resolves outside BenchFlow's protected runtime")
        if spec.origin and spec.origin.endswith(".py"):
            spec.loader = _Source(name, spec.origin)
        return spec

    @classmethod
    def find_distributions(cls, context=None):
        """Metadata of a closure distribution comes from the site folder
        first, whatever a conftest put on sys.path."""
        name = getattr(context, "name", None)
        if not name:
            return iter(())
        import importlib.metadata as metadata

        return iter(
            list(
                metadata.MetadataPathFinder.find_distributions(
                    metadata.DistributionFinder.Context(name=name, path=[SITE])
                )
            )
        )

    @classmethod
    def invalidate_caches(cls):
        pass


sys.meta_path.insert(0, _Protected)

_markers = os.environ.get("BENCHFLOW_PYTEST_MARKERS", "")
_key_file = os.environ.get("BENCHFLOW_PYTEST_KEY", "")
_token = "%%d-%%s" %% (os.getpid(), os.urandom(4).hex())


def _mark(kind, detail=""):
    """A signed marker the verifier reads, as the guard writes them."""
    if not (_markers and _key_file and _markers.startswith("/_runtime_guard_")):
        return
    try:
        with open(_key_file, "rb") as handle:
            key = handle.read().strip()
        if not key:
            return
        path = ".".join((_markers, _token, kind))
        body = json.dumps(detail[-4000:])
        signed = (os.path.basename(path) + "\n" + body).encode("utf-8")
        mac = hmac.new(key, signed, hashlib.sha256).hexdigest()
        with open(path, "x") as handle:
            handle.write(mac + "\t" + body + "\n")
    except Exception:
        pass


_CALLER_DIR = os.getcwd()


def _path_type(convert, node=False):
    def value(argument):
        argument = os.fspath(argument)
        if node:
            path, sep, rest = argument.partition("::")
            if path and os.path.exists(os.path.join(_CALLER_DIR, path)):
                argument = os.path.join(_CALLER_DIR, path) + sep + rest
        else:
            argument = os.path.join(_CALLER_DIR, argument)
        return convert(argument) if convert is not None else argument
    return value


ARGS = sys.argv[1:]

%(entry_points)s

import pytest  # noqa: E402
from _pytest.config import PytestPluginManager  # noqa: E402
from _pytest.config import builtin_plugins as _BUILTIN  # noqa: E402
from _pytest.config.argparsing import Parser as _Parser  # noqa: E402
from _pytest.config.argparsing import FILE_OR_DIR as _FILE_OR_DIR  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
from _pytest.assertion.rewrite import AssertionRewritingHook as _Rewriter  # noqa: E402


_original_rewrite = _Rewriter.find_spec


def _rewrite(self, name, path=None, target=None):
    if name.partition(".")[0] in PROTECTED_TOP | STDLIB_NAMES:
        return _Protected.find_spec(name, path, target)
    spec = _ImageFinder.find_spec(name, path, target)
    if spec is not None:
        return spec
    return _original_rewrite(self, name, path, target)


_Rewriter.find_spec = _rewrite


_original_parse = _Parser.parse


def _parse(self, args, *positional, **kwargs):
    # Final parsing happens after builtin, pinned and conftest plugins have
    # declared their arguments. Only path actions get a converter: pytest's
    # own parser consumes every option, nargs, attached value and '--'.
    # A text value is structurally unable to enter a path converter.
    parser = getattr(self, "optparser", None)
    if parser is None:
        parser = self._getparser()
    changed = []
    for action in parser._actions:
        node = action.dest == _FILE_OR_DIR and "--pyargs" not in args
        path = any(option in PATH_OPTIONS for option in action.option_strings) or action.type is _Path
        if not (node or path):
            continue
        changed.append((action, action.type))
        convert = parser._registry_get("type", action.type, action.type) if action.type is not None else None
        action.type = _path_type(convert, node=node)
    # Pytest 8 constructs a parser inside parse(); pytest 9 keeps optparser.
    # Both must use precisely the actions whose path converters we installed.
    factory = getattr(self, "_getparser", None)
    had_factory = "_getparser" in vars(self)
    if factory is not None:
        self._getparser = lambda: parser
    try:
        result = _original_parse(self, args, *positional, **kwargs)
    finally:
        for action, previous in changed:
            action.type = previous
        if factory is not None:
            if had_factory:
                self._getparser = factory
            else:
                del self._getparser
    os.chdir("/")
    return result


_Parser.parse = _parse


class Refused(pytest.UsageError):
    pass


def _origin(plugin):
    origin = getattr(plugin, "__file__", None)
    return os.path.realpath(origin) if origin else None


def _image_copy(origin):
    """Return the authenticated pre-agent source for an image helper."""
    if not WORKSPACE or not origin:
        return None
    root = next((r for r in (WORKSPACE, SNAPSHOT) if _under(origin, r)), None)
    if root is None:
        return None
    relative = os.path.relpath(origin, root)
    digest = IMAGE_FILES.get(relative)
    copied = os.path.realpath(os.path.join(SNAPSHOT, relative))
    if not digest or not _under(copied, SNAPSHOT):
        return None
    try:
        for path in {origin, copied, os.path.join(WORKSPACE, relative)}:
            with open(path, "rb") as handle:
                if hashlib.sha256(handle.read()).hexdigest() != digest:
                    return None
    except OSError:
        return None
    return copied


class _ImageSource(_Source):
    def __init__(self, name, origin, copied):
        super().__init__(name, origin)
        self.copied = copied

    def exec_module(self, module):
        global _IMAGE_LOADING
        _IMAGE_NAMES.add(module.__name__)
        _IMAGE_LOADING += 1
        try:
            return super().exec_module(module)
        finally:
            _IMAGE_LOADING -= 1

    def get_code(self, fullname):
        # Compile exactly the bytes authenticated here, without a second
        # source open or any byte-code cache. The inventory lives in run.py.
        source = self.get_data(self.copied)
        sealed = any(_under(os.path.realpath(self.copied), root) for root in IMAGE_SITES)
        relative = os.path.relpath(os.path.realpath(self.copied), SNAPSHOT)
        if not sealed and hashlib.sha256(source).hexdigest() != IMAGE_FILES.get(relative):
            message = "Verifier plugin trust rejected: image helper changed: " + fullname
            _mark("refused", message)
            raise Refused(message)
        _IMAGE_CODE.add(self.path)
        return compile(source, self.path, "exec", dont_inherit=True)


_IMAGE_LOADING = 0
_IMAGE_NAMES = set()
# File names of the image helper code compiled above: an import its hooks or
# fixtures run later is checked like one in its module body (V2-R6-C1).
_IMAGE_CODE = set()


def _image_caller():
    """Whether image helper code asked for the import running now: the
    nearest frame outside the import machinery, this runner and the protected
    runtime's own libraries (pytest, pluggy, the stdlib), so a helper's
    ``import`` or ``importlib.import_module`` counts, from a hook or a fixture
    alike, and a test's does not."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if (
            filename.startswith("<frozen")
            or filename == __file__
            or any(_under(filename, root) for root in (SITE, *STDLIB))
        ):
            frame = frame.f_back
            continue
        return filename in _IMAGE_CODE
    return False


def _system_linked_extension(path):
    """The sealed image loader paths may resolve basename dependencies.

    A native helper may not select a library by a path or change that
    search with RPATH/RUNPATH. Such releases need the pinned wheel tree.
    Inspect ELF without invoking its loader or executing the candidate.
    """
    import struct
    import os

    with open(path, "rb") as handle:
        data = handle.read()
    if data[:4] != b"\x7fELF" or data[4:6] not in (b"\x02\x01", b"\x01\x01"):
        return False
    wide = data[4] == 2
    header = struct.unpack_from("<HHIQQQIHHHHHH" if wide else "<HHIIIIIHHHHHH", data, 16)
    offset, size, count = header[4], header[8], header[9]
    segments = []
    dynamic = None
    for index in range(count):
        values = struct.unpack_from("<IIQQQQQQ" if wide else "<IIIIIIII", data, offset + index * size)
        kind = values[0]
        file_offset, address, length = (values[2], values[3], values[5]) if wide else (values[1], values[2], values[4])
        if kind == 1:
            segments.append((address, length, file_offset))
        if kind == 2:
            dynamic = (file_offset, length)
    if dynamic is None:
        return False
    needed = []
    searches = []
    strings = None
    step = 16 if wide else 8
    for offset in range(dynamic[0], dynamic[0] + dynamic[1], step):
        tag, value = struct.unpack_from("<qQ" if wide else "<iI", data, offset)
        if tag == 0:
            break
        if tag in (0x6ffffefa, 0x6ffffefb, 0x6ffffefc, 0x7ffffffd, 0x7fffffff):
            return False  # CONFIG, DEPAUDIT, AUDIT, AUXILIARY, FILTER
        if tag in (15, 29):
            searches.append(value)
        if tag == 5:
            strings = value
        if tag == 1:
            needed.append(value)
    base = next((pos + strings - address for address, length, pos in segments if strings is not None and address <= strings < address + length), None)
    if base is None:
        return not needed and not searches
    def string(offset):
        return data[base + offset:data.index(b"\0", base + offset)].decode("utf-8")
    for name in needed:
        if "/" in string(name):
            return False
    for search in searches:
        for directory in string(search).split(":"):
            directory = directory.replace("${ORIGIN}", os.path.dirname(path)).replace("$ORIGIN", os.path.dirname(path))
            if not os.path.isabs(directory) or "$" in directory:
                return False
            if os.path.commonpath((os.path.normpath(directory), os.path.dirname(SITE))) != os.path.dirname(SITE):
                return False
    return True


def _image_spec(name, copied, package, origin=None):
    origin = origin or copied
    location = origin if package and copied.endswith(".py") else copied
    spec = _util.spec_from_file_location(name, location,
        submodule_search_locations=[os.path.dirname(origin)] if package else None)
    if copied.endswith(".py"):
        spec.loader = _ImageSource(name, location, copied)
    else:
        # Native loaders open the file themselves. Its authenticated copy
        # and every parent must therefore be immutable to the sandbox user
        # before module_from_spec can call the extension's create_module.
        current = copied
        while True:
            st = os.lstat(current)
            if st.st_uid != 0 or st.st_mode & 0o022 or os.path.islink(current):
                message = "Verifier plugin trust rejected: image helper extension is not root's alone: " + name
                _mark("refused", message)
                raise Refused(message)
            if current == "/":
                break
            current = os.path.dirname(current)
        try:
            system_linked = _system_linked_extension(copied)
        except Exception:
            system_linked = False
        if not system_linked:
            message = "Verifier plugin trust rejected: image helper extension selects custom library paths; pin its release: " + name
            _mark("refused", message)
            raise Refused(message)
    return spec


class _ImageFinder:
    """Dependencies loaded from a helper's pre-agent package use checked source."""
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name.partition(".")[0] in PROTECTED_TOP | STDLIB_NAMES:
            return None
        # Image dependencies have their own fixed, sealed roots. An existing
        # workspace package remains the subject of grading; importing an
        # unchanged helper must not replace that package with its baseline.
        top = name.partition(".")[0]
        parent = sys.modules.get(top)
        parent_origin = _origin(parent) if parent is not None else None
        candidate = _machinery.PathFinder.find_spec(name, path)
        candidate_origin = os.path.realpath(candidate.origin) if candidate and candidate.origin else None
        subject = any(p and (_under(p, TESTS) or (WORKSPACE and _under(p, WORKSPACE) and os.path.relpath(p, WORKSPACE) in IMAGE_FILES)) for p in (parent_origin, candidate_origin))
        if top in IMAGE_TOP and not subject:
            search = IMAGE_SITES if path is None else [p for p in path if any(_under(os.path.realpath(p), root) for root in IMAGE_SITES)]
            spec = _machinery.PathFinder.find_spec(name, search)
            locations = ([spec.origin] if spec and spec.origin else []) + list(spec.submodule_search_locations or []) if spec else []
            if not locations or any(not any(_under(os.path.realpath(p), root) for root in IMAGE_SITES) for p in locations):
                raise ImportError(name + " is not in the protected image dependency copy")
            if spec.origin and spec.origin.endswith(".py"):
                spec.loader = _ImageSource(name, spec.origin, spec.origin)
            elif spec.origin and isinstance(spec.loader, _machinery.ExtensionFileLoader):
                spec = _image_spec(name, spec.origin, spec.submodule_search_locations is not None)
            return spec
        if not _IMAGE_LOADING and name not in _IMAGE_NAMES and not _image_caller():
            return None
        spec = candidate
        if spec is None:
            return None
        origin = os.path.realpath(spec.origin) if spec.origin else None
        if origin is None:
            locations = list(spec.submodule_search_locations or ())
            if locations and all(not any(_under(os.path.realpath(p), r) for r in (WORKSPACE, SNAPSHOT) if r) for p in locations):
                return None
            copied_locations = []
            for location in locations:
                location = os.path.realpath(location)
                root = next((r for r in (WORKSPACE, SNAPSHOT) if r and _under(location, r)), None)
                relative = os.path.relpath(location, root) if root else ".."
                if relative == ".." or not any(p.startswith(relative + "/") for p in IMAGE_FILES):
                    raise Refused("Verifier plugin trust rejected: unknown image helper namespace: " + name)
                copied_locations.append(location)
            if not copied_locations:
                raise Refused("Verifier plugin trust rejected: empty image helper namespace: " + name)
            spec.submodule_search_locations = copied_locations
            return spec
        if origin and not any(_under(origin, r) for r in (WORKSPACE, SNAPSHOT) if r):
            if name in _IMAGE_NAMES:
                raise Refused("Verifier plugin trust rejected: image helper moved: " + name)
            return None
        copied = _image_copy(origin)
        if copied is None:
            # A workspace module the agent changed or wrote. The submitted
            # version runs for helper code only when the task's trusted tests
            # import that module themselves, so its code runs in this pytest
            # anyway (V2PT-R2-1); any other is refused before it runs.
            if name in TESTS_IMPORTS:
                return None
            message = (
                "Verifier plugin trust rejected: image helper dependency changed: "
                + name
                + " (an image helper imports it, the agent changed or wrote it, "
                "and the task's tests do not import it themselves)"
            )
            if TESTS_UNREAD:
                message += (
                    "; BenchFlow could not read these tests files to see what "
                    "they import (too large or too deeply nested): "
                    + ", ".join(TESTS_UNREAD[:10])
                )
            _mark("refused", message)
            raise Refused(message)
        return _image_spec(name, copied, spec.submodule_search_locations is not None, origin)

    @classmethod
    def invalidate_caches(cls):
        pass

    @classmethod
    def find_distributions(cls, context=None):
        """Image metadata, except for a distribution the protected runtime
        itself installs: that one's metadata is the runtime's, by name and in
        any enumeration, so it matches the code that runs (V2-R6-C5)."""
        import importlib.metadata as metadata

        name = getattr(context, "name", None)
        runtime = _runtime_distributions()
        if name and _normal(name) in runtime:
            return iter(())
        return (
            distribution
            for distribution in metadata.MetadataPathFinder.find_distributions(
                metadata.DistributionFinder.Context(name=name, path=IMAGE_SITES)
            )
            if _normal(distribution.metadata.get("Name") or "") not in runtime
        )


_RUNTIME_DISTRIBUTIONS = None


def _normal(name):
    # PEP 503: runs of "-", "_" and "." are one "-", case-insensitively.
    return "-".join(filter(None, name.replace("_", "-").replace(".", "-").lower().split("-")))


def _runtime_distributions():
    """Normalized names of the distributions in the protected site folder."""
    global _RUNTIME_DISTRIBUTIONS
    if _RUNTIME_DISTRIBUTIONS is None:
        import importlib.metadata as metadata

        _RUNTIME_DISTRIBUTIONS = frozenset(
            _normal(distribution.metadata.get("Name") or "")
            for distribution in metadata.MetadataPathFinder.find_distributions(
                metadata.DistributionFinder.Context(path=[SITE])
            )
        )
    return _RUNTIME_DISTRIBUTIONS


sys.meta_path.insert(0, _ImageFinder)


def _allowed(plugin):
    """Why pytest may not register *plugin*, or None."""
    if not isinstance(plugin, types.ModuleType):
        return None  # pytest's own objects (config, session, managers)
    name = plugin.__name__
    origin = _origin(plugin)
    if origin is None:
        return "%%s has no file" %% name
    if _under(origin, TESTS):
        return None  # a conftest or helper of the task's tests
    if _image_copy(origin):
        return None
    if any(_under(origin, root) for root in IMAGE_SITES):
        return None
    top = name.partition(".")[0]
    if _under(origin, SITE) and (top in ("_pytest", "pytest") or top in ALLOWED_TOP):
        return None
    return "%%s (%%s) is not one of the task's pinned plugin releases" %% (name, origin)


_original_register = PytestPluginManager.register


def _register(self, plugin, name=None):
    if self.is_registered(plugin):
        # The same module asked for again by another name (-p timeout and a
        # conftest's pytest_plugins = ["pytest_timeout"]): it is there once.
        return self.get_name(plugin)
    why = _allowed(plugin)
    if why is not None:
        message = (
            "Verifier plugin trust rejected: " + why + "; BenchFlow's protected "
            "pytest loads only the task's pinned plugin releases "
            "([verifier] pytest_plugin_releases) and modules of " + TESTS
        )
        _mark("refused", message)
        raise Refused(message)
    return _original_register(self, plugin, name)


PytestPluginManager.register = _register


def _checked_plugin(name):
    """Authenticate and execute the same spec, including every dotted parent.

    Neither ambient entry point metadata nor a package's extended __path__
    chooses executable plugin code. Namespace packages are allowed only when
    every location is in the authorized tree.
    """
    roots = [SITE] if name.partition(".")[0] in ALLOWED_TOP else [TESTS]
    if roots == [TESTS] and WORKSPACE and IMAGE_FILES:
        roots = [TESTS, WORKSPACE, SNAPSHOT]
    if roots != [SITE]:
        roots += IMAGE_SITES
    paths = [p for p in sys.path if any(_under(os.path.realpath(p), root) for root in roots)] + roots
    for index, part in enumerate(name.split(".")):
        current = ".".join(name.split(".")[:index + 1])
        module = sys.modules.get(current)
        spec = getattr(module, "__spec__", None) if module else None
        if spec is None:
            spec = _machinery.PathFinder.find_spec(current, paths)
        origin = os.path.realpath(spec.origin) if spec and spec.origin else None
        locations = list(spec.submodule_search_locations or []) if spec else []
        image = origin and any(_under(origin, r) for r in (WORKSPACE, SNAPSHOT) if r)
        copied = _image_copy(origin) if image else None
        if (spec is None or (origin is None and not locations)
            or (image and copied is None)
            or (origin is not None and not any(_under(origin, r) for r in roots))
            or any(not any(_under(os.path.realpath(p), r) for r in roots)
                   for p in locations)):
            message = (
                "Verifier plugin trust rejected: %%s (%%s) is not one of "
                "the task's pinned plugin releases or modules of %%s"
                %% (current, origin, TESTS)
            )
            _mark("refused", message)
            raise Refused(message)
        if module is None:
            if copied is not None:
                # Never read a source's unchecked legacy or __pycache__ file.
                spec = _image_spec(current, copied, bool(locations), origin)
            elif origin and any(_under(origin, root) for root in IMAGE_SITES):
                if origin.endswith(".py"):
                    spec.loader = _ImageSource(current, origin, origin)
                elif isinstance(spec.loader, _machinery.ExtensionFileLoader):
                    spec = _image_spec(current, origin, bool(locations))
            module = _util.module_from_spec(spec)
            sys.modules[current] = module
            try:
                if spec.loader is not None:
                    spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(current, None)
                raise
        if index:
            # Match Python's import contract as well as its sys.modules
            # entry: ``import package.child; package.child`` must work.
            parent = current.rpartition(".")[0]
            setattr(sys.modules[parent], part, module)
        # A package may extend its search path on import. The next component
        # is still resolved only inside the tree this plugin is authorized for.
        paths = [p for p in getattr(module, "__path__", ())
                 if any(_under(os.path.realpath(p), r) for r in roots)]
    return module


_original_import_plugin = PytestPluginManager.import_plugin


def _image_alias(name):
    """Image aliases are read only from the sealed pre-agent image copy."""
    import importlib.metadata as metadata

    for distribution in metadata.MetadataPathFinder.find_distributions(
        metadata.DistributionFinder.Context(path=IMAGE_SITES)
    ):
        try:
            for entry in distribution.entry_points:
                if entry.group == "pytest11" and entry.name == name:
                    return entry.value
        except Exception:
            continue  # unrelated malformed image metadata cannot stop pytest
    return name


def _import_plugin(self, modname, consider_entry_points=False):
    """Refuse a plugin module before pytest imports it: importing runs it
    (V2-R4-1, a conftest's ``sys.path.insert(0, "/app")`` and
    ``pytest_plugins = ["helpers"]``)."""
    name = str(modname)
    top = name.partition(".")[0]
    known = (
        self.is_blocked(name)
        or self.get_plugin(name) is not None
        or name in _BUILTIN  # pytest's own, imported as _pytest.<name>
    )
    if known:
        return _original_import_plugin(self, modname, False)
    pinned = name in PLUGINS
    # One loader with the guard (the _bf_* block above, V2PT-2, V2PT-3):
    # the entry point's object, its owning distribution recorded.
    target, attributes = _bf_entry_parts(PLUGINS[name] if pinned else _image_alias(name))
    module = _checked_plugin(target)
    plugin = _bf_entry_object(module, attributes)
    registered = self.register(plugin, name)
    _bf_record_distribution(self, plugin, name, [SITE] if pinned else IMAGE_SITES)
    return registered


PytestPluginManager.import_plugin = _import_plugin


class _Report:
    @staticmethod
    def pytest_configure(config):
        _mark("registered")


args = ["-p", "no:cacheprovider", "--confcutdir=" + TESTS]
for _name in sorted(PLUGINS):
    args += ["-p", _name]
sys.exit(pytest.main([*args, *ARGS], plugins=[_Report()]))
'''


_ENTRY_BEGIN = "# --- BenchFlow pinned-plugin entry points: begin ---"
_ENTRY_END = "# --- BenchFlow pinned-plugin entry points: end ---"


def entry_points_source() -> str:
    """The guard's pinned-plugin entry-point block, verbatim, which the runner
    carries too: one loader for both paths (V2PT-2, V2PT-3)."""
    text = Path(__file__).with_name("_pytest_plugin_guard.py").read_text()
    start = text.index(_ENTRY_BEGIN)
    end = text.index(_ENTRY_END, start) + len(_ENTRY_END)
    return text[start:end]


def runner_source(
    *,
    site: str,
    stdlib: list[str],
    tests: str,
    plugins: dict[str, str],
    allowed_top: list[str],
    protected_top: list[str],
    workspace: str = "",
    snapshot: str = "/testbed_verify",
    image_files: dict[str, str] | None = None,
    stdlib_names: list[str] | None = None,
    image_sites: list[str] | None = None,
    image_top: list[str] | None = None,
    tests_imports: list[str] | None = None,
    tests_unread: list[str] | None = None,
) -> str:
    """The runner's Python (``run.py``), everything it trusts written in."""
    return _RUNNER % {
        "site": site,
        "stdlib": list(stdlib),
        "tests": tests,
        "plugins": dict(sorted(plugins.items())),
        "allowed_top": sorted(allowed_top),
        "protected_top": sorted(protected_top),
        "workspace": workspace,
        "snapshot": snapshot,
        "image_files": image_files or {},
        "stdlib_names": sorted(
            stdlib_names if stdlib_names is not None else _local_stdlib_names(stdlib)
        ),
        "image_sites": image_sites or [],
        "image_top": image_top or [],
        "tests_imports": sorted(tests_imports or []),
        "tests_unread": sorted(tests_unread or []),
        "entry_points": entry_points_source(),
    }


def _local_stdlib_names(paths: list[str]) -> set[str]:
    """Inventory a library tree even on Python 3.9, without importing it."""
    import sys

    names = set(sys.builtin_module_names)
    for folder in paths:
        try:
            entries = Path(folder).iterdir()
            for entry in entries:
                name = entry.name
                if name in {"site-packages", "dist-packages", "__pycache__"}:
                    continue
                if entry.is_dir() or name.endswith((".py", ".so", ".pyd")):
                    top = name.split(".", 1)[0]
                    if top.isidentifier():
                        names.add(top)
        except OSError:
            continue
    return names


# Reads a Python's version, tags and standard library, stdlib only. On the
# kit (C8) this is KIT_PYTHON, the interpreter the runner will run.
_PROBE = r"""
import json, os, platform, sys, sysconfig, glob, hashlib, site
paths = sysconfig.get_paths()
stdlib = [paths["stdlib"], paths["platstdlib"], os.path.join(paths["platstdlib"], "lib-dynload")]
stdlib = [p for p in dict.fromkeys(stdlib) if os.path.isdir(p)]
stdlib_names = set(sys.builtin_module_names)
for folder in stdlib:
    for name in os.listdir(folder):
        if name in ("site-packages", "dist-packages", "__pycache__"):
            continue
        path = os.path.join(folder, name)
        if os.path.isdir(path) or name.endswith((".py", ".so", ".pyd")):
            top = name.split(".", 1)[0]
            if top.isidentifier():
                stdlib_names.add(top)
libc = None
try:
    libc = os.confstr("CS_GNU_LIBC_VERSION")
except (AttributeError, ValueError, OSError):
    pass
musl = "1.2" if glob.glob("/lib/ld-musl-*.so.1") else None
image_files = {}
image_sites = []
image_top = set()
# Recover the selected image interpreter's prefix without executing it or
# processing site hooks. C8 copied this exact executable before the grant.
with open(sys.executable, "rb") as handle:
    executable_digest = hashlib.sha256(handle.read()).digest()
prefixes = []
for candidate in ("/usr/local/bin/python3", "/usr/bin/python3", "/bin/python3"):
    try:
        with open(candidate, "rb") as handle:
            if hashlib.sha256(handle.read()).digest() != executable_digest:
                continue
        prefixes.append(os.path.dirname(os.path.dirname(os.path.realpath(candidate))))
        break
    except OSError:
        pass
for folder in site.getsitepackages(prefixes) if prefixes else []:
    if os.path.isdir(folder) and folder not in image_sites:
        image_sites.append(folder)
        for name in os.listdir(folder):
            if name.endswith((".dist-info", ".egg-info")) or name == "__pycache__":
                continue
            if os.path.isdir(os.path.join(folder, name)) or name.endswith((".py", ".so", ".pyd")):
                top = name.split(".", 1)[0]
                if top.isidentifier():
                    image_top.add(top)
for folder, dirs, files in os.walk("/testbed_verify", followlinks=False):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".venv")]
    for name in files:
        path = os.path.join(folder, name)
        if name.endswith((".py", ".so", ".pyd")) and not os.path.islink(path):
            with open(path, "rb") as handle:
                image_files[os.path.relpath(path, "/testbed_verify")] = hashlib.sha256(handle.read()).hexdigest()
markers = {
    "implementation_name": sys.implementation.name,
    "implementation_version": platform.python_version(),
    "os_name": os.name,
    "platform_machine": platform.machine(),
    "platform_python_implementation": platform.python_implementation(),
    "platform_release": platform.release(),
    "platform_system": platform.system(),
    "platform_version": platform.version(),
    "python_full_version": platform.python_version(),
    "python_version": "%d.%d" % sys.version_info[:2],
    "sys_platform": sys.platform,
}
print(json.dumps({"exe": os.path.realpath(sys.executable),
    "version": list(sys.version_info[:2]), "machine": platform.machine(),
    "glibc": libc, "musl": musl, "markers": markers, "stdlib": stdlib,
    "image_files": image_files, "stdlib_names": sorted(stdlib_names),
    "workspace": os.path.realpath(WORKSPACE) if globals().get("WORKSPACE") else "",
    "image_sites": image_sites, "image_top": sorted(image_top)}))
"""


def _version_pair(text: str | None) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def _parse_probe(stdout: str, *, runner: bool) -> Interpreter | None:
    for line in reversed((stdout or "").splitlines()):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if not isinstance(data, dict):
            return None
        version = tuple(data.get("version") or ())
        if len(version) != 2 or tuple(version) < (3, 9):
            return None
        return Interpreter(
            exe=str(data.get("exe") or ""),
            version=(int(version[0]), int(version[1])),
            machine=str(data.get("machine") or "x86_64"),
            glibc=_version_pair(data.get("glibc")),
            musl=_version_pair(data.get("musl")),
            markers={str(k): str(v) for k, v in (data.get("markers") or {}).items()},
            stdlib=[str(p) for p in data.get("stdlib") or []],
            runner=runner,
            image_files={
                str(k): str(v) for k, v in (data.get("image_files") or {}).items()
            },
            stdlib_names=[str(n) for n in data.get("stdlib_names") or []],
            workspace=str(data.get("workspace") or ""),
            image_sites=[str(p) for p in data.get("image_sites") or []],
            image_top=[str(p) for p in data.get("image_top") or []],
        )
    return None


async def probe_interpreter(env: Any, service: str = "main") -> Interpreter | None:
    """The verifier's Python, read in the service that runs it.

    With C8's kit (that service's own, round 9): the kit's Python, which the
    runner runs; otherwise the image's ``python3``, for the wheel tags only
    (no runner without the kit). None when there is no Python 3.9 or later.
    """
    kit = None
    loader = getattr(env, "runtime_kit_for_route", None)
    if loader is not None:
        try:
            kit = await loader(service)
        except TypeError:
            kit = await loader()
        except Exception as exc:
            logger.debug("verifier runtime: no kit in %s: %s", service, exc)
    on_kit = bool(getattr(kit, "python", False))
    source = (
        "WORKSPACE = "
        + repr(getattr(env, "_benchflow_verifier_workspace", ""))
        + "\n"
        + _PROBE
    )
    if on_kit:
        command = f"cd / && {KIT_PYTHON} -B -c {shlex.quote(source)}"
    else:
        command = (
            "cd / && for p in /usr/local/bin/python3 /usr/bin/python3 python3; do "
            f'command -v "$p" >/dev/null 2>&1 || continue; '
            f'"$p" -I -S -c {shlex.quote(source)} && exit 0; done; exit 3'
        )
    result = await exec_raw(
        env.exec, command, user="root", timeout_sec=60, service=service
    )
    return _parse_probe(getattr(result, "stdout", "") or "", runner=on_kit)


def runtime_requested(task: Any) -> tuple[list[str], bool]:
    """The task's pins, and whether test.sh asks for --ctrf."""
    from benchflow.sandbox.lockdown import _infer_pytest_plugins_from_test_script

    verifier = getattr(getattr(task, "config", None), "verifier", None)
    pins = list(getattr(verifier, "pytest_plugin_releases", None) or [])
    declared = list(getattr(verifier, "pytest_plugins", None) or [])
    ctrf = "ctrf" in _infer_pytest_plugins_from_test_script(task) or "ctrf" in declared
    if any(parse_pin(pin)[0] == "pytest-json-ctrf" for pin in pins):
        ctrf = False  # the task's own pin of it is in the tree
    return pins, ctrf


# The runner's variable as test.sh names it: not BENCHFLOW_PYTEST_MARKERS,
# _KEY or a task's own BENCHFLOW_PYTEST_BIN.
_RUNNER_NAME = re.compile(rb"BENCHFLOW_PYTEST(?![A-Za-z0-9_])")
# Past this much text in the tests folder, the runner is built as before.
_RUNNER_SCAN_MAX_BYTES = 256 * 1024 * 1024


def runner_requested(task: Any) -> bool:
    """Whether the task's verifier can start the protected runner: a file of
    its tests folder (test.sh and every helper beside it) or a ``[verifier]
    env`` value names ``$BENCHFLOW_PYTEST`` (ACC-RUN2).

    The runner is something a task asks for by that name; a verifier that
    never names it never starts it. Without it the tree holds no runner and
    no copy of the image's site folders, which is as large as the image's
    site-packages (a CUDA torch is several GB) and was written to the
    sandbox's disk on every trial. Read as bytes, never run; a folder that
    cannot be read through, or holds more than
    :data:`_RUNNER_SCAN_MAX_BYTES`, keeps the runner. A multi-step task's
    are each step's folder and ``[steps.verifier] env`` (task/steps.py):
    the runtime is prepared once, before the first step."""
    from benchflow.task.steps import has_steps, step_paths

    config = getattr(task, "config", None)
    verifiers = [getattr(config, "verifier", None)]
    task_dir = getattr(task, "task_dir", None)
    if not task_dir:
        return True
    if has_steps(task):
        steps = task.config.steps
        verifiers += [step.verifier for step in steps]
        folders = [step_paths(task, i).tests_dir for i in range(len(steps))]
    else:
        native = getattr(
            getattr(task, "paths", None), "uses_native_verifier_dir", False
        )
        folders = [Path(task_dir) / ("verifier" if native is True else "tests")]
    for verifier in verifiers:
        for value in (getattr(verifier, "env", None) or {}).values():
            if _RUNNER_NAME.search(str(value).encode("utf-8", "surrogateescape")):
                return True
    total = 0
    for folder in dict.fromkeys(folders):
        if not folder.is_dir():
            return True
        try:
            for path in sorted(folder.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                total += path.stat().st_size
                if total > _RUNNER_SCAN_MAX_BYTES:
                    return True
                if _RUNNER_NAME.search(path.read_bytes()):
                    return True
        except OSError:
            return True
    return False


def _task_wheelhouses(task: Any) -> tuple[str, ...]:
    """Local wheel folders an offline host may migrate from (V2-R4-5): the
    ones ``BENCHFLOW_PLUGIN_WHEELHOUSE`` names, then the task's folder."""
    folders = [p for p in os.environ.get(_WHEELHOUSE_ENV, "").split(os.pathsep) if p]
    task_dir = getattr(getattr(task, "paths", None), "task_dir", None) or getattr(
        task, "task_dir", None
    )
    if task_dir:
        folders.append(str(task_dir))
    return tuple(folders)


_COPY_IMAGE_SITES = r"""
import errno, hashlib, json, os, pathlib, shutil, stat, sys
root = pathlib.Path(sys.argv[1])
dest = root / "image-sites"
dest.mkdir()

def private_folder(parent):
    if not parent.stat().st_mode & 0o001:
        return True
    try:
        if os.getxattr(parent, "system.posix_acl_access"):
            return True
    except OSError:
        pass
    return False

def private_ancestors(path):
    for parent in pathlib.Path(os.path.realpath(path)).parents:
        if private_folder(parent):
            return True
    return False

# The same answer once per folder (PERF-SBX): a path that is not a link
# resolves to its folder's resolved path plus its name, so its ancestors are
# that folder and the folder's own ancestors. The tree does not change while
# it is copied (the image, before the agent's turn).
real_folders = {}
private_chains = {}

def private_chain(folder):
    if folder not in private_chains:
        parent = folder.parent
        private_chains[folder] = private_folder(folder) or (
            parent != folder and private_chain(parent)
        )
    return private_chains[folder]

def private_ancestors_once(path):
    if os.path.islink(path):
        return private_ancestors(path)
    folder = os.path.dirname(os.path.abspath(path))
    if folder not in real_folders:
        real_folders[folder] = pathlib.Path(os.path.realpath(folder))
    return private_chain(real_folders[folder])

for index, source in enumerate(json.loads(sys.argv[2])):
    source = pathlib.Path(source)
    target = dest / str(index)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    for copied in [target, *target.rglob("*")]:
        original = source / copied.relative_to(target)
        info = original.stat()
        # Keep group read authority, and never turn an image's private
        # ancestor or link target into public data under the runtime tree.
        os.chown(copied, 0, info.st_gid)
        mode = stat.S_IMODE(info.st_mode) & 0o755
        if private_ancestors_once(original):
            mode &= 0o700
        if info.st_uid != 0:
            if not mode & 0o400:
                mode &= ~0o044
            if copied.is_dir() and not mode & 0o100:
                mode &= ~0o011
        os.chmod(copied, mode)
        try:
            os.removexattr(copied, "security.capability")
        except OSError as exc:
            if exc.errno not in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
with (root / "SHA256SUMS").open("a") as handle:
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            handle.write(hashlib.sha256(path.read_bytes()).hexdigest() + "  ./" + path.relative_to(root).as_posix() + "\n")
"""


async def prepare_verifier_runtime(
    env: Any,
    task: Any,
    *,
    tests_dir: str,
    service: str = "main",
) -> dict[str, Any] | None:
    """Build the tree in *service* (C8's ``VERIFIER_DIR``), or None when the
    task needs neither its runner nor its plugins. Raises :class:`RuntimeUnavailable` or
    :class:`RuntimePinError` rather than leave the verifier without it."""
    import asyncio

    pins, ctrf = runtime_requested(task)
    runner_wanted = runner_requested(task)
    if not pins and not ctrf and not runner_wanted:
        return None
    interpreter = await probe_interpreter(env, service)
    if interpreter is not None:
        interpreter.workspace = interpreter.workspace or getattr(
            env, "_benchflow_verifier_workspace", ""
        )
    resolution = await asyncio.to_thread(
        resolve,
        pins,
        ctrf=ctrf,
        interpreter=interpreter,
        wheelhouses=_task_wheelhouses(task),
    )
    if runner_wanted and not resolution.runner:
        raise RuntimeUnavailable(
            "The task names $BENCHFLOW_PYTEST, but its protected runner could "
            "not be prepared: it needs the kit Python and BenchFlow's cached "
            "pytest wheels"
        )
    staging = f"{RUNTIME}/.staging-{uuid.uuid4().hex}"
    folder = shlex.quote(RUNTIME)
    # C8's kit makes VERIFIER_DIR before the agent's turn; a sandbox without
    # the kit (a separate verifier sandbox, which never ran the agent) gets
    # it here, root's, under a route only root may write.
    made = await exec_raw(
        env.exec,
        "umask 022 && mkdir -p "
        + folder
        + " && chown 0:0 "
        + folder
        + " && chmod 755 "
        + folder
        + " && "
        + root_only_tree_check(RUNTIME, "BenchFlow's protected verifier runtime"),
        user="root",
        timeout_sec=120,
        service=service,
    )
    if getattr(made, "return_code", 0) not in (0, None):
        logger.warning(
            "verifier runtime (%s): %s",
            service,
            diagnostic_result(made).stderr.render(),
        )
        raise RuntimeUnavailable(
            f"{RUNTIME} is not a folder only root may write (the log has the reason)"
        )
    # What the trusted tests import, read once from the task's own copy.
    scanned = await asyncio.to_thread(
        scan_tests, getattr(getattr(task, "paths", None), "tests_dir", None)
    )
    with tempfile.TemporaryDirectory(prefix="bf-verifier-runtime-") as tmp:
        tree = Path(tmp) / "tree"
        manifest = build_tree(
            resolution,
            interpreter=interpreter,
            tests_dir=tests_dir,
            into=tree,
            tests_imports=scanned[0],
            tests_unread=scanned[1],
            runner_wanted=runner_wanted,
        )
        await env.upload_dir(
            tree,
            staging,
            service=service,
            admission=admit(
                env, service, TRUSTED_TRANSFER, reason="BenchFlow's verifier runtime"
            ),
        )
    if manifest["image_sites"] and interpreter is not None:
        # The image is still the pre-agent image (also in separate mode,
        # before transfer). Materialize dependencies and resources here;
        # the runner never consults the original site directories or .pth.
        copied = await exec_raw(
            env.exec,
            f"{KIT_PYTHON} -B -c {shlex.quote(_COPY_IMAGE_SITES)} "
            + shlex.quote(staging)
            + " "
            + shlex.quote(json.dumps(interpreter.image_sites)),
            user="root",
            timeout_sec=300,
            service=service,
        )
        if getattr(copied, "return_code", 0) not in (0, None):
            raise RuntimeUnavailable(
                "the image dependencies could not be copied before the agent's turn"
            )
    quoted = shlex.quote(staging)
    place = (
        f"set -e; cd {folder}; "
        "rm -rf site image-sites bin run.py manifest.json SHA256SUMS; "
        f"for e in site image-sites bin run.py manifest.json SHA256SUMS; do "
        f'[ -e {quoted}/"$e" ] && mv {quoted}/"$e" "$e"; done; '
        f"rm -rf {quoted}; chown -R 0 .; chmod -R go-w .; "
        f"{_same_files_check(RUNTIME, SUMS, 'the uploaded verifier runtime')}"
    )
    result = await exec_raw(
        env.exec, place, user="root", timeout_sec=120, service=service
    )
    if getattr(result, "return_code", 0) not in (0, None):
        logger.warning(
            "verifier runtime (%s): %s",
            service,
            diagnostic_result(result).stderr.render(),
        )
        raise RuntimeUnavailable(
            "the tree could not be put in place (the log has the reason)"
        )
    logger.debug(
        "verifier runtime (%s): %d wheels, plugins %s, runner %s",
        service,
        len(resolution.wheels),
        ",".join(sorted(manifest["plugins"])) or "-",
        manifest["runner"] or "none",
    )
    return manifest


def runtime_check_command() -> str:
    """sh, as root after the agent's turn: the tree is still root's alone and
    holds exactly what BenchFlow put there; then its manifest. Exit 4: no
    tree was built."""
    return (
        f"[ -f {shlex.quote(MANIFEST)} ] || exit 4; "
        + "{ "
        + root_only_tree_check(RUNTIME, "BenchFlow's protected verifier runtime")
        + "; } && { "
        + _same_files_check(RUNTIME, SUMS, "BenchFlow's protected verifier runtime")
        + "; } || exit 5; "
        + f"cat {shlex.quote(MANIFEST)}"
    )


RUNTIME_CHECK_CMD = runtime_check_command()
