"""Test-suite inventory: no test removed or newly skipped without a decision.

A green test command says nothing about the tests that did not run: a test
that is deleted, renamed, deselected by a new marker, hidden by a new
``collect_ignore`` entry or a module-level ``importorskip``, or that now
skips or xfails, leaves ``pytest`` at exit 0. This tool records which tests
exist, which the round's command selects, how each one ended, and every
skip/xfail site in the source, then compares two arms (a base and a tip) and
exits 1 on each change that no exact-id decision authorizes.

Subcommands:

- ``run-leg``: run one pytest leg of the test command with an xunit1 JUnit
  file and record its argv, selection arguments, exit code and JUnit hash in
  a run manifest (primary legs and serial retry legs are kept apart). Retain
  hashed runtime marker snapshots and skip/xfail/XPASS reports separately
  from JUnit, including each local xdist worker.
- ``collect``: in an arm's tree and under that arm's interpreter, run
  pytest's CLI with ``--collect-only`` once with the effective selection (one
  run per primary leg) and once with ``-o addopts=``, read module-level
  collection skips and errors from pytest's own JUnit, inventory the marks
  pytest actually collects across lifecycle boundaries plus static AST sites,
  and map every JUnit testcase back to a collected id. Writes one JSON inventory.
- ``skips``: the skips and xfails an inventory saw, one ``id<TAB>kind<TAB>reason``
  line each.
- ``ids``: the sorted selected or collected ids, one per line.
- ``failed``: the ids that failed or errored in the primary legs (the serial
  retry leg's arguments).
- ``compare``: base against tip, ``--mode collection`` or ``--mode runtime``,
  against an allow file and the decisions file. Exit 0: nothing unexplained.
  Exit 1: a finding with no authorized allow line, an allow line that does
  not resolve to a decision naming it, or (``--strict-allow``) a stale allow
  line. Exit 2: the evidence is invalid or incomplete.

The analyzer is stdlib only and imports nothing from benchflow or pytest;
only its pytest subprocesses import the tested tree's conftest files and
plugins. Run one pinned copy, outside both trees, under each arm's own venv.
It catches ordinary suite regressions, not code written to falsify its own
collection output.

Supported exemption evidence is pytest's built-in Function/TestCaseFunction
and pytest-asyncio's four Function backends and skip/skipif/xfail marks, plus
unittest's skip/expectedFailure
metadata. Arguments use a closed encoding of None, bool, int, str, float,
list/tuple, string-keyed dict and classes; other values/backends
invalidate collection. Static literal forms and captured aliases supplement
that evidence. Unresolved exemption-capable constructions require an exact
``unknown-exemption`` allow entry, even when inherited from the base; rewrite
them into supported literal forms when authorization is inappropriate.
Marker application accepts literal string names or modeled pytest.mark
values, with positional/keyword bindings and literal expansions. Namespace
escapes, metadata access, opaque callable construction and reflection outside
the literal model require authorization; operator/builtins/importlib/inspect
imports and reflective primitives are deliberately conservative. Unknown IDs
bind normalized source constructions and individual duplicate occurrences;
old generic authorizations must be recollected and reviewed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as _dt
import hashlib
import inspect
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA = "suite-inventory/1"
MANIFEST_SCHEMA = "suite-inventory-run/1"

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_INVALID = 2

# Finding kinds. Each gating kind needs an allow line whose decision names it.
REMOVED = "removed"
DESELECTED = "deselected"
NEW_SKIP = "new-skip"
NEW_XFAIL = "new-xfail"
NOT_RUN = "not-run"
MODULE_SKIPPED = "module-skipped"
MODULE_ERROR = "module-error"
XFAIL_SITE = "xfail-site"
UNKNOWN_EXEMPTION = "unknown-exemption"
UNAUTHORIZED_ALLOW = "unauthorized-allow"
STALE_ALLOW = "stale-allow"
RENAMED = "renamed"  # an allow kind: covers a `removed` finding of its old id

FINDING_KINDS = (
    REMOVED,
    DESELECTED,
    NEW_SKIP,
    NEW_XFAIL,
    NOT_RUN,
    MODULE_SKIPPED,
    MODULE_ERROR,
    XFAIL_SITE,
    UNKNOWN_EXEMPTION,
)
ALLOW_KINDS = (*FINDING_KINDS, RENAMED)

# Exit codes a primary or retry leg may have: 0 (all passed) or 1 (some
# tests failed; the ordinary failure gate and its serial retries judge
# those). 2 interrupted, 3 internal error, 4 usage error and 5 no tests are
# never valid evidence.
VALID_LEG_EXITS = (0, 1)

# Environment variables whose values decide which tests skip or are
# selected. They are recorded and must match between runtime arms. Only the
# _ENV_PLAIN ones keep their value in clear; every other value is recorded
# as a short sha256 (equality is all a compare needs, and a value under a
# listed prefix can carry a credential its name does not reveal:
# ANTHROPIC_CUSTOM_HEADERS, a UV_INDEX_URL with user:token@, a *_BASE_URL
# with a key in its query). A secret-looking name is recorded as present only.
_ENV_EXACT = (
    "CI",
    "GITHUB_ACTIONS",
    "HOME",
    "LANG",
    "LC_ALL",
    "TZ",
    "PYTHONHASHSEED",
    "PYTHONWARNINGS",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "LITELLM_MODE",
)
_ENV_PREFIXES = (
    "PYTEST_",
    "BENCHFLOW_",
    "DOCKER_",
    "DAYTONA_",
    "MODAL_",
    "E2B_",
    "ANTHROPIC_",
    "OPENAI_",
    "GEMINI_",
    "GOOGLE_",
    "LITELLM_",
    "PLAYWRIGHT_",
    "UV_",
)
# Values short and harmless enough to show in clear when the arms differ.
_ENV_PLAIN = frozenset(
    {
        "CI",
        "GITHUB_ACTIONS",
        "LANG",
        "LC_ALL",
        "TZ",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONSAFEPATH",
        "LITELLM_MODE",
        "BENCHFLOW_DETERMINISTIC_SANDBOX",
        "BENCHFLOW_E2E_SANDBOX",
    }
)
# Set by pytest or xdist inside a running session, or differ per arm by
# design (each arm has its own venv); recorded, never compared.
_ENV_VOLATILE = (
    "PYTEST_CURRENT_TEST",
    "PYTEST_XDIST_WORKER",
    "PYTEST_XDIST_WORKER_COUNT",
    "PYTEST_XDIST_TESTRUNUID",
    "UV_PROJECT_ENVIRONMENT",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "UV_RUN_RECURSION_DEPTH",
)
_SECRET_NAME = re.compile(
    r"KEY|TOKEN|SECRET|PASSW|AUTH|COOKIE|CREDENTIAL|SESSION|PRIVATE|SIGN",
    re.IGNORECASE,
)
_PACKAGES = (
    "pytest",
    "pytest-xdist",
    "pytest-asyncio",
    "pytest-timeout",
    "pytest-playwright",
    "playwright",
)
_TREE_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".cache",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
    }
)
# Run output the suite writes into the tree root (bench writes job folders to
# ./jobs, which .gitignore lists): a leg's run must not change the digest the
# next leg and collect compare against.
_TREE_SKIP_ROOT_DIRS = frozenset({"jobs"})


class InventoryError(Exception):
    """Invalid or incomplete evidence: the caller exits 2."""


# --------------------------------------------------------------------------
# small helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise InventoryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InventoryError(f"{path} is not a JSON object")
    return data


def tree_digest(root: Path) -> dict[str, Any]:
    """Content digest of a tree (git archive shape), skipping caches and venvs.

    ``tests/test_integration_check_results.py`` writes a git snapshot under
    ``.cache/`` at import, so ``.cache`` is left out with the other caches;
    the deterministic tier writes job folders to ``./jobs`` (seen on a build
    VM, round PY8 review 1), so the root's ``jobs`` is left out too.
    """
    digest = hashlib.sha256()
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        skip = _TREE_SKIP_DIRS
        if Path(dirpath) == root:
            skip = _TREE_SKIP_DIRS | _TREE_SKIP_ROOT_DIRS
        # os.walk puts directory links in dirnames but never descends into
        # them. Hash the link itself, just as for a symlink to a file.
        links = [
            d for d in dirnames if d not in skip and (Path(dirpath) / d).is_symlink()
        ]
        dirnames[:] = sorted(d for d in dirnames if d not in skip and d not in links)
        for name in sorted([*filenames, *links]):
            if name.endswith(".pyc"):
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                payload = b"link:" + os.readlink(path).encode()
            elif path.is_file():
                payload = path.read_bytes()
            else:
                continue
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode() + b"\0")
            digest.update(hashlib.sha256(payload).digest())
            count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def _env_value(name: str, value: str) -> str:
    if _SECRET_NAME.search(name):
        return "<set>"
    if name in _ENV_PLAIN:
        return value
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:16]


def sanitized_env(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Relevant variables: plain values for a short allowlist, hashes otherwise.

    ``compared`` holds what must match between runtime arms; ``other_names``
    lists every other variable name (never a value). No value outside
    ``_ENV_PLAIN`` is written in clear, so a manifest, an inventory or a
    compare report never carries one.
    """
    env = dict(os.environ if environ is None else environ)
    compared: dict[str, str] = {}
    volatile: dict[str, str] = {}
    other: list[str] = []
    for name in sorted(env):
        relevant = name in _ENV_EXACT or name.startswith(_ENV_PREFIXES)
        if not relevant:
            other.append(name)
            continue
        value = _env_value(name, env[name])
        if name in _ENV_VOLATILE:
            volatile[name] = value
        else:
            compared[name] = value
    return {"compared": compared, "volatile": volatile, "other_names": other}


def sanitized_argv(
    argv: Sequence[str], redactions: dict[str, str] | None = None
) -> list[str]:
    """Keep useful pytest arguments; never retain assignment or URL values."""
    recorded = []
    secret_next = False
    ini_next = False
    for arg in argv:
        if secret_next:
            recorded.append("<redacted>")
            secret_next = False
        elif "://" in arg:
            # Mask before splitting: a URL can contain '=' in its query
            # or '.py::' in its path without becoming a safe test address.
            recorded.append("<redacted-url>")
        elif ini_next:
            # Pytest splits at the first '=' without restricting ini names.
            # Even a name matching one of our path options is still an ini
            # assignment here, never permission to retain its value.
            name, _, value = arg.partition("=")
            recorded.append(f"{name}={_argument_value(name, value)}")
        elif not arg.startswith("-") and ".py::" in arg:
            # Parameterized test IDs may themselves contain '='. These are
            # selection addresses, unless an ini option consumed them above.
            recorded.append(arg)
        elif "=" in arg:
            name, value = arg.split("=", 1)
            if name in ("--junitxml", "--ignore", "--deselect"):
                recorded.append(f"{name}={value}")
            else:
                recorded.append(f"{name}={_argument_value(name, value)}")
        else:
            recorded.append(arg)
            secret_next = arg.startswith("-") and bool(_SECRET_NAME.search(arg))
        # A trailing -o may follow plugin-defined short flags too. Treat
        # unknown flags conservatively, but stop at known value-taking
        # options: -kfoo is a selection expression, not an ini override.
        # Attached values (including -o=NAME=VALUE) are opaque above.
        ini_next = "=" not in arg and (
            (
                arg.startswith("-")
                and not arg.startswith("--")
                and arg.endswith("o")
                and not set(arg[1:-1]) & set("ckmnopr")
            )
            or arg == "--override-ini"
        )
    # Imported selections also carry the consumption context of their own
    # recorded command, which can differ from the collector's invocation.
    return [
        (redactions or {}).get(raw, safe)
        for raw, safe in zip(argv, recorded, strict=True)
    ]


def _argument_value(name: str, value: str) -> str:
    # The environment's plain-value allowlist does not apply to ini keys.
    if _SECRET_NAME.search(name):
        return "<set>"
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:16]


def _argv_sha256(argv: Sequence[str]) -> str:
    """Compare the original vector without retaining its credential values."""
    return hashlib.sha256(json.dumps(list(argv)).encode()).hexdigest()


def _selection_redactions(manifest: dict[str, Any]) -> dict[str, str]:
    """Retain imported tokens only when their command proves them safe."""
    redactions = {}
    for leg in manifest["legs"]:
        command = leg.get("argv") or []
        recorded = sanitized_argv(command)
        select = leg.get("select_args") or []
        for raw, safe in zip(select, sanitized_argv(select), strict=True):
            matches = [
                value
                for arg, value in zip(command, recorded, strict=True)
                if arg == raw
            ]
            # An old recorder may already have sanitized the command, losing
            # the association with its raw selection. Hash the whole token
            # when no association survives, or any occurrence consumed it as
            # a value. Do not guess from address-like syntax or another leg.
            if raw != safe or not matches or any(value != raw for value in matches):
                redactions[raw] = "sha256:" + _argv_sha256([raw])[:16]
    return redactions


# Parse in a child, where pytest already imports the tested tree's plugins.
# Config.parse expands both ini addopts and PYTEST_ADDOPTS into `effective`;
# fromdictargs deliberately disables that expansion and is not sufficient.
_CONFIG_PROBE = """
import json
import sys
from _pytest.config import get_config
effective = sys.argv[2:]
config = get_config(effective)
try:
    config.parse(effective)
    data = {
        "args": effective,
        "inipath": str(config.inipath) if config.inipath else None,
        "options": {name: action.nargs for name, action in config._parser.optparser._option_string_actions.items()},
        "explicit_workers": bool(config.getoption("tx", default=[]) or config.getoption("px", default=[])),
    }
    with open(sys.argv[1], "w") as handle:
        json.dump(data, handle)
finally:
    config._ensure_unconfigure()
"""


def _pytest_configuration(
    root: Path, argv: Sequence[str], env: dict[str, str]
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="suite-config-") as tmp:
        result = Path(tmp) / "config.json"
        proc = subprocess.run(
            [sys.executable, "-c", _CONFIG_PROBE, str(result), *argv],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode or not result.exists():
            # Plugin diagnostics can echo credential-bearing arguments.
            raise InventoryError("cannot validate effective pytest configuration")
        data = _read_json(result)
    if data["explicit_workers"]:
        raise InventoryError(
            "explicit xdist --tx/--px workers have unverified interpreters; use local -n workers"
        )
    return data


def _discovery_args(argv: Sequence[str], options: dict[str, Any]) -> list[str]:
    """Keep discovery/plugin configuration while removing selection filters.

    Unknown plugin options keep their separate value. Bare paths/node ids
    select only part of a tree and are removed, including those after `--`.
    """
    argv = list(argv)
    result: list[str] = []
    filters = ("-k", "-m", "--ignore", "--ignore-glob", "--deselect")
    flags = {
        "--lf",
        "--last-failed",
        "--ff",
        "--failed-first",
        "--nf",
        "--new-first",
        "--sw",
        "--stepwise",
        "--sw-skip",
        "--stepwise-skip",
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            break
        if (
            arg not in options
            and arg.startswith("-")
            and not arg.startswith("--")
            and arg[:2] in options
        ):
            # argparse accepts -xktest_keep as -x -k test_keep. Exact
            # option strings win; only zero-argument short options can
            # precede another option in a group. Values are never split.
            expanded = []
            tail = arg[1:]
            while tail:
                option = "-" + tail[0]
                if option not in options:
                    raise InventoryError("cannot expand grouped pytest option")
                expanded.append(option)
                tail = tail[1:]
                if options[option] != 0:
                    if tail:
                        expanded.append(tail.removeprefix("="))
                    break
            argv[i : i + 1] = expanded
            arg = argv[i]
        if arg in filters:
            i += 2
            continue
        if any(arg.startswith(f + "=") for f in filters) or (
            arg.startswith(("-k", "-m")) and len(arg) > 2 and not arg.startswith("--")
        ):
            i += 1
            continue
        if arg in flags:
            i += 1
            continue
        if arg.startswith("-"):
            result.append(arg)
            if arg in options and options[arg] != 0:
                nargs = options[arg]
                count = 1 if nargs is None else nargs
                if isinstance(count, int):
                    result.extend(argv[i + 1 : i + 1 + count])
                    i += count
                else:
                    while i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                        result.append(argv[i + 1])
                        i += 1
                        if nargs == "?":
                            break
        i += 1
    return result


def _insert_pytest_options(argv: Sequence[str], options: Sequence[str]) -> list[str]:
    """Force options after user overrides, before pytest's path terminator."""
    result = list(argv)
    index = result.index("--") if "--" in result else len(result)
    result[index:index] = options
    return result


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def provenance(root: Path, tree_label: str | None) -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "root": str(root),
        "tree_label": tree_label,
        "tree": tree_digest(root),
        "packages": package_versions(),
        "env": sanitized_env(),
        "docker": {
            "cli": shutil.which("docker") is not None,
            "socket": Path("/var/run/docker.sock").exists(),
        },
        "created": _now(),
    }


# --------------------------------------------------------------------------
# pytest's JUnit address contract (xunit1)
#
# These two functions copy pytest's own `mangle_test_address` and
# `bin_xml_escape` (_pytest/junitxml.py, pytest 9.0.3), including its
# character class verbatim, so a collected id maps to exactly the
# (classname, name) pytest writes.


def mangle_test_address(address: str) -> list[str]:
    path, bracket, params = address.partition("[")
    names = path.split("::")
    names[0] = names[0].replace("/", ".")
    names[0] = re.sub(r"\.py$", "", names[0])
    names[-1] += bracket + params
    return names


# A raw string: `re` reads the \u escapes as Python read them in pytest's
# literal (so \u10000 is \u1000 followed by "0", exactly as there).
_ILLEGAL_XML = (
    r"[^\u0009\u000a\u000d\u0020-\u007e\u0080-\ud7ff\ue000-\ufffd\u10000-\u10ffff]"
)


def bin_xml_escape(arg: object) -> str:
    def repl(match: re.Match[str]) -> str:
        code = ord(match.group())
        return f"#x{code:02X}" if code <= 0xFF else f"#x{code:04X}"

    return re.sub(_ILLEGAL_XML, repl, str(arg))


def junit_key(nodeid: str) -> tuple[str, str]:
    names = mangle_test_address(nodeid)
    return ".".join(names[:-1]), bin_xml_escape(names[-1])


def id_file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _testcase_children(case: ET.Element) -> list[dict[str, str]]:
    children = []
    for child in case:
        if child.tag in ("failure", "error", "skipped"):
            children.append(
                {
                    "tag": child.tag,
                    "type": child.get("type", ""),
                    "message": child.get("message", ""),
                }
            )
    return children


def _collector_id(case: ET.Element) -> str | None:
    """The collector id of a collection report's testcase, or None."""
    file = case.get("file")
    if not file:
        return None
    classname = case.get("classname", "")
    name = case.get("name", "")
    module_key = bin_xml_escape(mangle_test_address(file)[0])
    if classname == "" and name == module_key:
        return file
    full = f"{classname}.{name}" if classname else name
    if not full.startswith(module_key + "."):
        return None
    return file + "::" + "::".join(full[len(module_key) + 1 :].split("."))


def is_collection_report(children: list[dict[str, str]]) -> bool:
    return any(
        (c["tag"] == "error" and c["message"] == "collection failure")
        or (c["tag"] == "skipped" and c["message"] == "collection skipped")
        for c in children
    )


def _collection_reason(case: ET.Element) -> str:
    for child in case:
        if child.tag not in ("error", "skipped"):
            continue
        text = (child.text or "").strip()
        if child.tag == "skipped":
            # pytest writes the longrepr tuple ('/abs/path', line, 'Skipped: why');
            # keep only the reason, so it reads the same from any tree.
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, tuple) and len(value) == 3:
                reason = str(value[2])
                return reason.removeprefix("Skipped: ")[:500]
        lines = [line for line in text.splitlines() if line.startswith("E ")]
        return (lines[-1] if lines else text.splitlines()[-1] if text else "")[:500]
    return ""


def parse_junit(path: Path) -> list[ET.Element]:
    """Every testcase element of an xunit1 file; invalid XML is an error."""
    try:
        tree = ET.parse(path)
    except FileNotFoundError as exc:
        raise InventoryError(f"JUnit file missing: {path}") from exc
    except (ET.ParseError, OSError) as exc:
        raise InventoryError(f"JUnit file invalid or truncated: {path}: {exc}") from exc
    root = tree.getroot()
    if root.tag not in ("testsuites", "testsuite"):
        raise InventoryError(f"JUnit file has root <{root.tag}>: {path}")
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise InventoryError(f"JUnit file has no <testsuite>: {path}")
    return list(root.iter("testcase"))


def outcome_of(children: list[dict[str, str]]) -> str:
    """One outcome for every testcase element one id produced in one leg."""
    tags = {c["tag"] for c in children}
    if "failure" in tags:
        return "failed"
    if "error" in tags:
        return "error"
    types = {c["type"] for c in children if c["tag"] == "skipped"}
    if "pytest.xfail" in types:
        return "xfailed"
    if "skipped" in tags:
        return "skipped"
    return "passed"


def exemptions_of(children: list[dict[str, str]]) -> dict[str, str]:
    """Exemption kind -> reason (skip or xfail) seen in an id's reports."""
    found: dict[str, str] = {}
    for c in children:
        if c["tag"] != "skipped":
            continue
        kind = "xfail" if c["type"] == "pytest.xfail" else "skip"
        found.setdefault(kind, c["message"])
    return found


class JunitIndex:
    """Collected ids indexed by their xunit1 (classname, name)."""

    def __init__(self, ids: Iterable[str]) -> None:
        self._by_key: dict[tuple[str, str], list[str]] = {}
        for nodeid in sorted(set(ids)):
            self._by_key.setdefault(junit_key(nodeid), []).append(nodeid)

    def match(self, case: ET.Element) -> str:
        key = (case.get("classname", ""), case.get("name", ""))
        candidates = self._by_key.get(key, [])
        if len(candidates) > 1:
            # `file` is where the test function is defined, which differs
            # from the id's module for an inherited test: it can only narrow.
            file = case.get("file", "")
            narrowed = [c for c in candidates if id_file(c) == file]
            candidates = narrowed if len(narrowed) == 1 else candidates
        if len(candidates) == 1:
            return candidates[0]
        what = "ambiguous" if candidates else "unmatched"
        raise InventoryError(
            f"{what} JUnit testcase classname={key[0]!r} name={key[1]!r} "
            f"file={case.get('file')!r}"
        )


def read_leg_junit(
    path: Path, index: JunitIndex
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]], list[str]]:
    """(id -> report children, collection reports, evidence errors)."""
    reports: dict[str, list[dict[str, str]]] = {}
    modules: list[dict[str, str]] = []
    errors: list[str] = []
    for case in parse_junit(path):
        children = _testcase_children(case)
        if case.get("classname") == "pytest" and case.get("name") == "internal":
            errors.append(f"{path.name}: pytest internal error")
            continue
        if is_collection_report(children):
            collector = _collector_id(case)
            if collector is None:
                errors.append(
                    f"{path.name}: collection report without a collector address "
                    f"({case.get('classname')!r}, {case.get('name')!r})"
                )
                continue
            outcome = (
                "error" if any(c["tag"] == "error" for c in children) else "skipped"
            )
            modules.append(
                {
                    "id": collector,
                    "outcome": outcome,
                    "reason": _collection_reason(case),
                }
            )
            continue
        try:
            nodeid = index.match(case)
        except InventoryError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        for c in children:
            if c["tag"] == "error" and "crashed while running" in c["message"]:
                errors.append(
                    f"{path.name}: xdist worker crashed: {c['message'][:300]}"
                )
        # A testcase with no failure/error/skipped child is a passed call.
        reports.setdefault(nodeid, []).append(
            {"tag": "testcase", "type": "", "message": ""}
        )
        reports[nodeid].extend(children)
    return reports, modules, errors


# --------------------------------------------------------------------------
# collection

# Loaded only in the arm's pytest subprocess. The parent stays stdlib only.
# Observe whole lifecycle hook calls outside pluggy's wrapper stacks, then the
# CLI's final return. Keep immutable multiset unions: teardown cannot erase an
# exemption that was active at an earlier boundary. Construction spelling does
# not matter; unsupported metadata is an error rather than absent evidence.
_COLLECTOR_PLUGIN = """\
import json
import os
from collections import Counter
from pathlib import Path
from _pytest.python import Function
from _pytest.unittest import TestCaseFunction

_backends = [Function, TestCaseFunction]
try:
    from pytest_asyncio.plugin import (Coroutine, AsyncGenerator,
                                      AsyncStaticMethod, AsyncHypothesisTest)
except ImportError:
    pass
else:
    _backends.extend([Coroutine, AsyncGenerator, AsyncStaticMethod, AsyncHypothesisTest])

_session = None
_seen = {}
_errors = set()
_reports = {}
_executed = set()
_runtime = False
_boundaries = {"pytest_collection_modifyitems", "pytest_collection_finish",
               "pytest_collection", "pytest_sessionfinish", "pytest_unconfigure"}


def pytest_sessionstart(session):
    global _session
    _session = session
    # Monitoring surrounds the entire hook call, including both wrapper APIs
    # at every priority. A hook implementation cannot unwind after `after`.
    session.config.pluginmanager.add_hookcall_monitoring(before, after)


def before(name, implementations, kwargs):
    if _runtime and name == "pytest_runtest_protocol":
        _executed.add(kwargs["item"].nodeid)
    if name in _boundaries:
        snapshot()
    elif _runtime:
        observe(kwargs)


def after(outcome, name, implementations, kwargs):
    if name in _boundaries:
        snapshot()
    elif _runtime:
        observe(kwargs)
        if name == "pytest_runtest_makereport":
            report = outcome.get_result()
            if hasattr(report, "wasxfail") or report.skipped:
                kind = "xfail" if hasattr(report, "wasxfail") else "skip"
                _reports.setdefault(report.nodeid, {})[kind] = str(
                    getattr(report, "wasxfail", "runtime skip"))
    if _runtime and name == "pytest_unconfigure":
        capture()


def observe(kwargs):
    item = kwargs.get("item")
    if item is None and "request" in kwargs:
        item = kwargs["request"].node
    if isinstance(item, Function):
        snapshot([item])


def value(v):
    if v is None or type(v) in (bool, int, str):
        return v
    if type(v) is float:
        return {"float": v.hex()}
    if type(v) in (list, tuple):
        return {type(v).__name__: [value(x) for x in v]}
    if type(v) is dict and all(type(k) is str for k in v):
        return {"dict": {k: value(x) for k, x in v.items()}}
    if isinstance(v, type):
        return {"type": v.__module__ + "." + v.__qualname__}
    # Never use repr: it can contain addresses, secrets or arbitrary code.
    raise ValueError("unsupported exemption argument")


def snapshot(items=None):
    if _session is None:
        return
    for item in _session.items if items is None else items:
        marks = []
        try:
            # Check trusted class identities, never spoofable module/name text.
            # Asyncio types inherit Function's marker model; AsyncGenerator
            # also adds an ordinary xfail mark.
            if type(item) not in _backends:
                raise ValueError("unsupported collection backend")
            for mark in item.iter_markers():
                if mark.name in ("skip", "skipif", "xfail"):
                    marks.append({"kind": "pytest.mark." + mark.name,
                                  "args": value(mark.args),
                                  "kwargs": value(mark.kwargs)})
            # pytest's unittest backend consumes these attributes directly,
            # rather than converting them to pytest marks.
            if type(item).__module__ == "_pytest.unittest":
                for obj in (item.obj, item.cls):
                    if getattr(obj, "__unittest_skip__", False):
                        marks.append({"kind": "unittest.skip", "args": value(()),
                                      "kwargs": value({"reason": getattr(obj, "__unittest_skip_why__", "")})})
                    if getattr(obj, "__unittest_expecting_failure__", False):
                        marks.append({"kind": "unittest.expectedFailure",
                                      "args": value(()), "kwargs": value({})})
        except Exception:
            _errors.add("cannot encode collected exemptions: " + item.nodeid
                        + "; use literal arguments or supported pytest/unittest metadata")
        # Encode now, never retain references to mutable mark arguments. Max
        # multiplicity keeps duplicates without counting repeated observations.
        current = Counter(json.dumps(mark, sort_keys=True) for mark in marks)
        _seen[item.nodeid] = _seen.get(item.nodeid, Counter()) | current


def capture():
    if _session is None:
        return
    snapshot()
    ids = sorted(_seen) if _runtime else [item.nodeid for item in _session.items]
    records = [{"id": nodeid,
                "marks": [json.loads(mark) for mark, count in sorted(_seen[nodeid].items())
                          for _ in range(count)]}
               for nodeid in ids]
    path = Path(__file__).with_suffix(".json")
    if _runtime:
        path = path.with_name(path.stem + "-" + str(os.getpid()) + ".json")
    path.write_text(json.dumps({"items": records, "reports": _reports,
                                "executed": sorted(_executed),
                                "errors": sorted(_errors)}, sort_keys=True))
"""

# Run pytest's real CLI entry point, preserving its exit and argument behavior.
# The finally block runs after pytest.main and all plugin teardown have returned,
# including collection, sessionfinish and unconfigure wrappers of either API.
_COLLECTOR_RUNNER = """\
import runpy
import _suite_inventory_collector as collector
try:
    runpy.run_module("pytest", run_name="__main__", alter_sys=True)
finally:
    collector.capture()
"""


def _collector_argv(argv: Sequence[str]) -> list[str]:
    """Wrap `python -m pytest` while retaining that stable provenance spelling."""
    return [argv[0], "-c", _COLLECTOR_RUNNER, *argv[3:]]


def _collected_mark_sites(
    runs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union mark variants across discovery legs; keep duplicate multiplicity."""
    counts: Counter[str] = Counter()
    sites: dict[str, dict[str, Any]] = {}
    for run in runs:
        current: Counter[str] = Counter()
        for item in run["marked_items"]:
            for mark in item["marks"]:
                file, _, owner = item["id"].partition("::")
                site = {
                    "file": file,
                    "owner": owner,
                    "kind": mark["kind"],
                    "family": "xfail"
                    if mark["kind"] in ("pytest.mark.xfail", "unittest.expectedFailure")
                    else "skip",
                    "expr": "pytest-collected " + json.dumps(mark, sort_keys=True),
                    "line": 0,
                    "source": "pytest-collection",
                }
                key = site_key(site)
                sites[key] = site
                current[key] += 1
        counts |= current
    return [sites[key] for key in sorted(counts) for _ in range(counts[key])]


def _parse_collect_stdout(
    stdout: str, root: Path
) -> tuple[list[str], list[str], list[str]]:
    """(ids in order, -rs SKIPPED lines, unparsed lines in the id block)."""
    lines = stdout.splitlines()
    ids: list[str] = []
    bad: list[str] = []
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("="):
            break
        path = id_file(line)
        if path.endswith(".py") and (root / path).is_file():
            ids.append(line)
        elif not ids and re.match(
            r"^(no tests|\d+ tests? collected|\d+/\d+ tests? collected)", line
        ):
            break
        else:
            bad.append(line)
        i += 1
    skipped = [line for line in lines[i:] if line.startswith("SKIPPED [")]
    # A verbose pytest session begins with '=' too. An empty parse must
    # never be mistaken for an empty suite: reconcile with pytest's count.
    summaries = [
        int(match.group(1))
        for line in lines
        if (match := re.search(r"\b(\d+)(?:/\d+)? tests? collected\b", line))
    ]
    if not summaries and any("no tests collected" in line for line in lines):
        summaries = [0]
    if not summaries or summaries[-1] != len(ids):
        bad.append(
            f"collection summary {summaries[-1:]!r} does not match {len(ids)} parsed ids"
        )
    return ids, skipped, bad


def run_collect_only(
    root: Path,
    extra: Sequence[str],
    pytest_args: Sequence[str],
    *,
    complete: bool = False,
    redactions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One `pytest --collect-only` run: ids, collection reports, exit code."""
    with tempfile.TemporaryDirectory(prefix="suite-inventory-") as tmp:
        junit = Path(tmp) / "collect.xml"
        plugin = Path(tmp) / "_suite_inventory_collector.py"
        plugin.write_text(_COLLECTOR_PLUGIN)
        env = dict(os.environ)
        configured_args = [*pytest_args, *extra]
        # Retained fragments must come from the assembled vector: a trailing
        # option in pytest_args can consume the first selection argument.
        recorded_args = sanitized_argv(configured_args, redactions)
        recorded_selection = recorded_args[len(pytest_args) :]
        if complete:
            config = _pytest_configuration(root, configured_args, env)
            configured_args = _discovery_args(config["args"], config["options"])
            if config["inipath"]:
                configured_args += ["-c", config["inipath"]]
            configured_args += ["-o", "addopts="]
            env.pop("PYTEST_ADDOPTS", None)
        # The temporary import path is an instrument detail, not an arm's
        # test environment. Keep its random directory out of provenance.
        collection_env = dict(env)
        collection_env["PYTHONPATH"] = str(plugin.parent) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        argv = _insert_pytest_options(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-rs",
                "-p",
                "no:cacheprovider",
                "-p",
                "_suite_inventory_collector",
                "--continue-on-collection-errors",
                f"--junitxml={junit}",
                "-o",
                "junit_family=xunit1",
                *configured_args,
            ],
            ["--verbosity=-1", "-o", "verbosity_test_cases=-1"],
        )
        started = time.monotonic()
        proc = subprocess.run(
            _collector_argv(argv),
            cwd=root,
            env=collection_env,
            capture_output=True,
            text=True,
            check=False,
        )
        duration = round(time.monotonic() - started, 3)
        ids, _, bad = _parse_collect_stdout(proc.stdout, root)
        errors: list[str] = []
        marked_items: list[dict[str, Any]] = []
        try:
            evidence = json.loads(plugin.with_suffix(".json").read_text())
            marked_items = evidence["items"]
            if [item["id"] for item in marked_items] != ids:
                errors.append("collected mark evidence does not match collected ids")
            # Validate structure before allowing it to become inventory sites.
            for item in marked_items:
                for mark in item["marks"]:
                    if mark["kind"] not in (
                        *_MARK_KINDS,
                        "unittest.skip",
                        "unittest.expectedFailure",
                    ):
                        raise ValueError("unknown exemption kind")
                    if set(mark) != {"kind", "args", "kwargs"}:
                        raise ValueError("incomplete exemption")
            errors.extend(evidence["errors"])
        except (OSError, ValueError, KeyError, TypeError):
            marked_items = []
            errors.append("missing or invalid collected mark evidence")
        if len(bad) == 1 and bad[0].startswith("collection summary "):
            # pytest's summary counts before collection_modifyitems: a hook
            # removing one item can print six ids and "7 tests collected".
            # Its -qq per-file counts use the final items instead. Verify
            # those independently before accepting a discrepant summary;
            # a truncated/verbose listing still cannot pass this check.
            counted = subprocess.run(
                _collector_argv(
                    _insert_pytest_options(
                        argv, ["--verbosity=-2", "-o", "verbosity_test_cases=-2"]
                    )
                ),
                cwd=root,
                env=collection_env,
                capture_output=True,
                text=True,
                check=False,
            )
            counts = {
                match.group(1): int(match.group(2))
                for line in counted.stdout.splitlines()
                if (match := re.fullmatch(r"(.+\.py): (\d+)", line))
            }
            empty = counted.returncode == 5 and re.search(
                r"\bno tests collected\b", counted.stdout
            )
            if (
                (counts or empty)
                and counted.returncode == proc.returncode
                and counts == dict(Counter(id_file(i) for i in ids))
            ):
                bad = []
        duration = round(time.monotonic() - started, 3)
        modules: list[dict[str, str]] = []
        if proc.returncode not in (0, 1, 5):
            errors.append(
                f"collect-only {shlex.join(recorded_args) or '(default)'} "
                f"exited {proc.returncode}; subprocess diagnostics omitted"
            )
        if bad:
            errors.append(f"unparsed lines in collect-only id list: {len(bad)}")
        if len(ids) != len(set(ids)):
            dups = [k for k, v in Counter(ids).items() if v > 1]
            errors.append(f"duplicate collected ids: {dups[:5]}")
        if junit.exists():
            try:
                for case in parse_junit(junit):
                    children = _testcase_children(case)
                    if not is_collection_report(children):
                        continue
                    collector = _collector_id(case)
                    if collector is None:
                        errors.append("collection report without a collector address")
                        continue
                    outcome = (
                        "error"
                        if any(c["tag"] == "error" for c in children)
                        else "skipped"
                    )
                    modules.append(
                        {
                            "id": collector,
                            "outcome": outcome,
                            "reason": (
                                "collection failed (subprocess diagnostics omitted)"
                                if outcome == "error"
                                else _collection_reason(case)
                            ),
                        }
                    )
            except InventoryError as exc:
                errors.append(str(exc))
        elif proc.returncode in (0, 1, 5):
            errors.append("collect-only wrote no JUnit file")
    if proc.returncode == 1 and not any(m["outcome"] == "error" for m in modules):
        errors.append("collect-only exited 1 with no collection error recorded")
    # Recorded without the interpreter path and the temporary JUnit path, so
    # the same selection reads the same in both arms.
    recorded = sanitized_argv(
        [arg for arg in argv[1:] if not arg.startswith("--junitxml=")], redactions
    )
    return {
        "argv": recorded,
        "select_args": recorded_selection,
        "env": sanitized_env(env),
        "complete": complete,
        "exit_code": proc.returncode,
        "duration_s": duration,
        "ids": ids,
        "marked_items": marked_items,
        "order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "module_reports": sorted(modules, key=lambda m: (m["id"], m["outcome"])),
        # Do not retain free-form subprocess text: plugins and usage errors
        # can echo credentials. Structured JUnit records carry skip reasons.
        "rs_lines": [],
        "summary": f"{len(ids)} parsed ids; exit {proc.returncode}",
        "errors": errors,
    }


# --------------------------------------------------------------------------
# static skip/xfail site inventory

_MARK_KINDS = {
    "pytest.mark.xfail": "xfail",
    "pytest.mark.skip": "skip",
    "pytest.mark.skipif": "skip",
}
_CALL_KINDS = {
    "pytest.skip": "skip",
    "pytest.xfail": "xfail",
    "pytest.importorskip": "skip",
    "unittest.skip": "skip",
    "unittest.skipIf": "skip",
    "unittest.skipUnless": "skip",
    "unittest.expectedFailure": "xfail",
    "unittest.TestCase.skipTest": "skip",
}
_RAISE_KINDS = {
    "pytest.skip.Exception": "skip",
    "pytest.xfail.Exception": "xfail",
    "unittest.SkipTest": "skip",
}
_EXEMPTION_ATTRIBUTES = {
    name.rsplit(".", 1)[-1] for name in (*_MARK_KINDS, *_CALL_KINDS)
}
_PYTEST_NAMES = {"mark", "skip", "xfail", "importorskip", "param", "raises", "fail"}
_UNITTEST_NAMES = {
    "skip",
    "skipIf",
    "skipUnless",
    "expectedFailure",
    "SkipTest",
    "TestCase",
}
_OPAQUE_REFLECTION_MODULES = {"operator", "builtins", "importlib", "inspect"}
_REFLECTION_PRIMITIVES = {
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
}

type _AliasValue = str | tuple[_AliasValue | None, ...] | dict[str, _AliasValue | None]


class _Aliases:
    """Names bound to pytest/unittest objects or to a mark expression."""

    def __init__(self) -> None:
        self.dotted: dict[str, str] = {}
        self.expr: dict[str, ast.expr] = {}
        # Values captured in the caller's scope, including variadic containers.
        # A missing/unknown element stays unknown instead of losing its binding.
        self.bound: dict[str, _AliasValue | None] = {}
        self.dynamic: list[tuple[int, str]] = []


def _alias_value(node: ast.AST, aliases: _Aliases) -> _AliasValue | None:
    if isinstance(node, ast.Name):
        return aliases.dotted.get(node.id, aliases.bound.get(node.id))
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_alias_value(item, aliases) for item in node.elts)
    if isinstance(node, ast.Dict):
        values: dict[str, _AliasValue | None] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return None
            values[key.value] = _alias_value(value, aliases)
        return values
    if isinstance(node, ast.Subscript):
        container = _alias_value(node.value, aliases)
        try:
            key = ast.literal_eval(node.slice)
        except (ValueError, TypeError):
            return None
        if isinstance(container, tuple) and type(key) is int:
            return container[key] if -len(container) <= key < len(container) else None
        if isinstance(container, dict) and isinstance(key, str):
            return container.get(key)
        return None
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, aliases)
        return f"{base}.{node.attr}" if base else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
    ):
        try:
            args, keywords = _literal_call_arguments(node)
        except ValueError:
            return None
        if (
            len(args) in (2, 3)
            and not keywords
            and isinstance(args[1], ast.Constant)
            and isinstance(args[1].value, str)
        ):
            base = _dotted(args[0], aliases)
            return f"{base}.{args[1].value}" if base else None
    return None


def _dotted(node: ast.AST, aliases: _Aliases) -> str | None:
    value = _alias_value(node, aliases)
    return value if isinstance(value, str) else None


def _returned_family(node: ast.AST, aliases: _Aliases) -> str | None:
    """Only literal immediately invoked lambdas have modeled return values."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Lambda):
        bindings = _parameter_aliases(node.func.args, aliases, aliases, node)
        if bindings.dynamic:
            return None
        return _returned_family(node.func.body, bindings)
    return _dotted(node, aliases)


def _collect_aliases(
    tree: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    inherited: _Aliases | None = None,
) -> _Aliases:
    aliases = _Aliases()
    if inherited is not None:
        aliases.dotted.update(inherited.dotted)
        aliases.expr.update(inherited.expr)
        aliases.bound.update(inherited.bound)

    def scope_nodes(node: ast.AST) -> Iterable[ast.AST]:
        yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        for child in ast.iter_child_nodes(node):
            yield from scope_nodes(child)

    nodes = [node for stmt in tree.body for node in scope_nodes(stmt)]
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("pytest", "unittest"):
                    aliases.dotted[alias.asname or alias.name] = alias.name
                elif alias.name.startswith("_pytest"):
                    aliases.dynamic.append((node.lineno, f"import {alias.name}"))
                elif alias.name.split(".", 1)[0] in _OPAQUE_REFLECTION_MODULES:
                    aliases.dynamic.append(
                        (node.lineno, f"unmodeled reflection import {alias.name}")
                    )
        elif isinstance(node, ast.ImportFrom) and node.module in ("pytest", "unittest"):
            known = _PYTEST_NAMES if node.module == "pytest" else _UNITTEST_NAMES
            for alias in node.names:
                if alias.name == "*":
                    aliases.dynamic.append(
                        (node.lineno, f"from {node.module} import *")
                    )
                elif alias.name in known:
                    aliases.dotted[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "_pytest"
        ):
            aliases.dynamic.append((node.lineno, f"from {node.module} import ..."))
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".", 1)[0] in _OPAQUE_REFLECTION_MODULES
        ):
            aliases.dynamic.append(
                (node.lineno, f"unmodeled reflection import {node.module}")
            )
    # Capture modeled literal aliases before marker-instance expressions. This
    # uses the same closed resolver as direct calls, including literal getattr.
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            value = stmt.value
            resolved = _dotted(value, aliases)
            if resolved in (*_MARK_KINDS, *_CALL_KINDS, *_RAISE_KINDS):
                aliases.dotted[stmt.targets[0].id] = resolved
                continue
            func = value.func if isinstance(value, ast.Call) else value
            dotted = _dotted(func, aliases)
            if dotted in _MARK_KINDS:
                aliases.expr[stmt.targets[0].id] = value
    return aliases


class _Canonical(ast.NodeTransformer):
    """Rewrite alias names into their canonical pytest/unittest spelling."""

    def __init__(self, aliases: _Aliases, seen: frozenset[str] = frozenset()) -> None:
        self.aliases = aliases
        self.seen = seen

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.aliases.dotted:
            return ast.parse(self.aliases.dotted[node.id], mode="eval").body
        if node.id in self.aliases.expr and node.id not in self.seen:
            inner = _Canonical(self.aliases, self.seen | {node.id})
            return inner.visit(copy.deepcopy(self.aliases.expr[node.id]))
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        dotted = _dotted(node, self.aliases)
        if dotted is not None:
            return ast.parse(dotted, mode="eval").body
        return self.generic_visit(node)


def _normalize(node: ast.AST, aliases: _Aliases) -> str:
    return ast.unparse(_Canonical(aliases).visit(copy.deepcopy(node)))


def _bound_expression(value: object) -> ast.expr:
    """Keep every Signature binding in the same unevaluated AST value domain."""
    if isinstance(value, ast.expr):
        return value
    if isinstance(value, tuple):
        return ast.Tuple(elts=[_bound_expression(v) for v in value], ctx=ast.Load())
    if isinstance(value, dict):
        keys: list[ast.expr | None] = []
        for key in value:
            if not isinstance(key, str):
                raise ValueError("unknown keyword binding")
            keys.append(ast.Constant(key))
        return ast.Dict(
            keys=keys,
            values=[_bound_expression(v) for v in value.values()],
        )
    raise ValueError("unknown argument binding")


def _parameter_aliases(
    args: ast.arguments,
    inherited: _Aliases,
    evaluated: _Aliases,
    call: ast.Call | None = None,
) -> _Aliases:
    """Parameters shadow free names; their defaults use the enclosing scope."""
    aliases = _Aliases()
    aliases.dotted.update(inherited.dotted)
    aliases.expr.update(inherited.expr)
    aliases.bound.update(inherited.bound)
    positional = [*args.posonlyargs, *args.args]
    parameters = [*positional, *args.kwonlyargs]
    parameters += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    for arg in parameters:
        aliases.dotted.pop(arg.arg, None)
        aliases.expr.pop(arg.arg, None)
        aliases.bound.pop(arg.arg, None)
    defaults = [
        *zip(
            positional[len(positional) - len(args.defaults) :],
            args.defaults,
            strict=True,
        ),
        *zip(args.kwonlyargs, args.kw_defaults, strict=True),
    ]
    bindings = {arg.arg: value for arg, value in defaults if value is not None}
    if call is not None:
        # Use Python's binding rules on AST values, without evaluating code.
        # An explicit unknown value must erase a known default just as a
        # known argument does; a failed bind must never fall back to defaults.
        kinds = (
            (args.posonlyargs, inspect.Parameter.POSITIONAL_ONLY),
            (args.args, inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ([args.vararg] if args.vararg else [], inspect.Parameter.VAR_POSITIONAL),
            (args.kwonlyargs, inspect.Parameter.KEYWORD_ONLY),
            ([args.kwarg] if args.kwarg else [], inspect.Parameter.VAR_KEYWORD),
        )
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    arg.arg,
                    kind,
                    default=bindings.get(arg.arg, inspect.Parameter.empty),
                )
                for group, kind in kinds
                for arg in group
            ]
        )
        try:
            actual, keywords = _literal_call_arguments(call)
            bound = signature.bind(*actual, **keywords)
            bound.apply_defaults()
            bindings = {
                name: _bound_expression(value)
                for name, value in bound.arguments.items()
            }
        except (TypeError, ValueError):
            bindings = {}
            aliases.dynamic.append((call.lineno, "unresolved lambda argument binding"))
    for name, value in bindings.items():
        if value is None:
            continue
        if isinstance(value, ast.Name) and value.id in evaluated.expr:
            value = evaluated.expr[value.id]
        resolved = _alias_value(value, evaluated)
        if resolved is None and isinstance(value, ast.Call):
            base = _dotted(value.func, evaluated)
            if base in _MARK_KINDS:
                resolved = base
        aliases.bound[name] = resolved
        if isinstance(resolved, str):
            aliases.dotted[name] = resolved
        elif resolved is None and call is not None:
            aliases.dynamic.append((call.lineno, f"unresolved lambda argument {name}"))
    return aliases


def _literal_call_arguments(
    call: ast.Call,
) -> tuple[list[ast.expr], dict[str, ast.expr]]:
    """Expand only literal argument containers; unknown expansions stay unknown."""
    positional: list[ast.expr] = []
    keywords: dict[str, ast.expr] = {}

    def extend(values: list[ast.expr]) -> None:
        for value in values:
            if isinstance(value, ast.Starred):
                if not isinstance(value.value, (ast.List, ast.Tuple)):
                    raise ValueError("unknown positional expansion")
                extend(value.value.elts)
            else:
                positional.append(value)

    def mapping(value: ast.expr) -> dict[str, ast.expr]:
        if not isinstance(value, ast.Dict):
            raise ValueError("unknown keyword expansion")
        result: dict[str, ast.expr] = {}
        for key, item in zip(value.keys, value.values, strict=True):
            if key is None:
                result.update(mapping(item))
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                result[key.value] = item
            else:
                raise ValueError("unknown keyword name")
        return result

    extend(call.args)
    for keyword in call.keywords:
        expanded = (
            {keyword.arg: keyword.value}
            if keyword.arg is not None
            else mapping(keyword.value)
        )
        if keywords.keys() & expanded.keys():
            raise ValueError("duplicate keyword argument")
        keywords.update(expanded)
    return positional, keywords


def _comprehension_aliases(
    target: ast.expr, iterable: ast.expr, inherited: _Aliases, evaluated: _Aliases
) -> _Aliases:
    aliases = _Aliases()
    aliases.dotted.update(inherited.dotted)
    aliases.expr.update(inherited.expr)
    aliases.bound.update(inherited.bound)
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            aliases.dotted.pop(node.id, None)
            aliases.expr.pop(node.id, None)
            aliases.bound.pop(node.id, None)
    # A literal iterable can capture a class binding just like a lambda
    # default. Other targets still shadow free aliases and are reported
    # as unresolved if used to construct a mark.
    if isinstance(target, ast.Name) and isinstance(
        iterable, (ast.List, ast.Tuple, ast.Set)
    ):
        values = {_dotted(value, evaluated) for value in iterable.elts}
        if len(values) == 1:
            value = values.pop()
            if value is not None:
                aliases.dotted[target.id] = value
    return aliases


class _SiteVisitor(ast.NodeVisitor):
    def __init__(self, file: str, aliases: _Aliases) -> None:
        self.file = file
        self.aliases = aliases
        # Class namespaces apply to decorators, but methods close over the
        # enclosing function/module, never the class namespace.
        self._function_aliases = aliases
        self.owner: list[str] = []
        self.sites: list[dict[str, Any]] = []
        self.dynamic: list[dict[str, Any]] = [
            {"file": file, "line": line, "owner": "<module>", "what": what}
            for line, what in aliases.dynamic
        ]
        self.asserts: dict[str, int] = {}
        self._consumed: set[int] = set()
        self._current_test: str | None = None
        self._stack: list[ast.AST] = []

    def visit(self, node: ast.AST) -> Any:
        # Namespace values may only feed a literal attribute or literal
        # getattr operation. Every other consumer is outside our model.
        if self._stack and isinstance(node, (ast.Name, ast.Attribute)):
            parent = self._stack[-1]
            if _dotted(node, self.aliases) in {"pytest", "pytest.mark", "unittest"}:
                safe = (
                    isinstance(parent, ast.Attribute)
                    and parent.value is node
                    and not parent.attr.startswith("_")
                )
                # Literal containers, explicit lambda bindings and defaults
                # already have a closed model in _parameter_aliases.
                for ancestor in reversed(self._stack):
                    if isinstance(
                        ancestor,
                        (ast.Tuple, ast.List, ast.Dict, ast.Starred, ast.keyword),
                    ):
                        continue
                    if isinstance(ancestor, ast.arguments) or (
                        isinstance(ancestor, ast.Call)
                        and isinstance(ancestor.func, ast.Lambda)
                    ):
                        safe = True
                    break
                call = parent
                if (
                    isinstance(parent, (ast.Tuple, ast.List))
                    and len(self._stack) >= 3
                    and isinstance(self._stack[-2], ast.Starred)
                ):
                    call = self._stack[-3]
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "getattr"
                ):
                    try:
                        args, kwargs = _literal_call_arguments(call)
                        safe = (
                            len(args) in (2, 3)
                            and not kwargs
                            and isinstance(args[1], ast.Constant)
                            and isinstance(args[1].value, str)
                            and not args[1].value.startswith("_")
                        )
                    except ValueError:
                        pass
                if not safe:
                    self._dynamic(node, "escaped exemption namespace")
        self._stack.append(node)
        try:
            return super().visit(node)
        finally:
            self._stack.pop()

    def _owner(self) -> str:
        return ".".join(self.owner) or "<module>"

    def _site(self, node: ast.AST, kind: str, family: str) -> None:
        self.sites.append(
            {
                "file": self.file,
                "owner": self._owner(),
                "kind": kind,
                "family": family,
                "expr": _normalize(node, self.aliases),
                "line": getattr(node, "lineno", 0),
            }
        )

    def _dynamic(self, node: ast.AST, what: str) -> None:
        # Identity covers the construction, including its consumer, rather
        # than an isolated namespace token shared by many different uses.
        construction = next(
            (
                parent
                for parent in reversed(self._stack)
                if isinstance(parent, ast.stmt)
            ),
            node,
        )
        self.dynamic.append(
            {
                "file": self.file,
                "line": getattr(node, "lineno", 0),
                "owner": self._owner(),
                "what": what,
                "expr": ast.unparse(construction),
            }
        )

    def _visit_def(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        self.owner.append(node.name)
        for deco in node.decorator_list:
            self.visit(deco)
        if not isinstance(node, ast.ClassDef):
            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)
        else:
            for base in node.bases:
                self.visit(base)
        outer_aliases = self.aliases
        outer_function_aliases = self._function_aliases
        # Class decorators use the surrounding class above; free names in
        # the new class body skip that namespace, just as method bodies do.
        inherited = self._function_aliases
        if not isinstance(node, ast.ClassDef):
            inherited = _parameter_aliases(node.args, inherited, outer_aliases)
        self.aliases = _collect_aliases(node, inherited)
        if not isinstance(node, ast.ClassDef):
            self._function_aliases = self.aliases
        self.dynamic.extend(
            {"file": self.file, "line": line, "owner": self._owner(), "what": what}
            for line, what in self.aliases.dynamic
        )
        outer_test = self._current_test
        in_test_class = all(part.startswith("Test") for part in self.owner[:-1])
        if (
            outer_test is None
            and not isinstance(node, ast.ClassDef)
            and node.name.startswith("test")
            and in_test_class
        ):
            # Asserts in helpers nested in a test count toward that test.
            self._current_test = f"{self.file}::{'::'.join(self.owner)}"
            self.asserts.setdefault(self._current_test, 0)
        for stmt in node.body:
            self.visit(stmt)
        self._current_test = outer_test
        self.aliases = outer_aliases
        self._function_aliases = outer_function_aliases
        self.owner.pop()

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def
    visit_ClassDef = _visit_def

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_lambda(node)

    def _visit_lambda(self, node: ast.Lambda, call: ast.Call | None = None) -> None:
        # Defaults are evaluated here; the body closes over the nearest
        # function/module, excluding any surrounding class namespace.
        self.visit(node.args)
        outer = self.aliases
        outer_function = self._function_aliases
        if call is not None:
            for value in [*call.args, *call.keywords]:
                self.visit(value)
        self.aliases = _parameter_aliases(node.args, outer_function, outer, call)
        for line, what in self.aliases.dynamic:
            self.dynamic.append(
                {"file": self.file, "line": line, "owner": self._owner(), "what": what}
            )
        self._function_aliases = self.aliases
        self.visit(node.body)
        self.aliases = outer
        self._function_aliases = outer_function

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        # Only the first iterable is evaluated in the surrounding class.
        # The remaining comprehension runs in its own function-like scope.
        outer = self.aliases
        outer_function = self._function_aliases
        for i, generator in enumerate(node.generators):
            self.visit(generator.iter)
            self.aliases = _comprehension_aliases(
                generator.target,
                generator.iter,
                self.aliases if i else outer_function,
                self.aliases,
            )
            self._function_aliases = self.aliases
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.aliases = outer
        self._function_aliases = outer_function

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def _count(self) -> None:
        if self._current_test is not None:
            self.asserts[self._current_test] += 1

    def visit_Assert(self, node: ast.Assert) -> None:
        self._count()
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            dotted = _dotted(exc, self.aliases)
            if dotted in _RAISE_KINDS:
                self._site(node.exc, f"raise {dotted}", _RAISE_KINDS[dotted])
                self._consumed.add(id(exc))
                if isinstance(node.exc, ast.Call):
                    self._consumed.add(id(node.exc))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Lambda):
            self._visit_lambda(func, node)
            return
        dotted = _dotted(func, self.aliases)
        if (
            isinstance(func, ast.Call)
            and dotted is None
            and _returned_family(func, self.aliases) not in (*_MARK_KINDS, *_CALL_KINDS)
        ):
            self._dynamic(node, "unmodeled callable construction")
        if (
            dotted is None
            and isinstance(func, ast.Attribute)
            and func.attr in _EXEMPTION_ATTRIBUTES
        ):
            self._dynamic(node, f"unresolved {ast.unparse(func)}")
        # getattr(pytest.mark, "xfail") with a literal name is the mark itself.
        itself = _dotted(node, self.aliases)
        if id(node) not in self._consumed:
            if itself in _MARK_KINDS:
                self._site(node, itself, _MARK_KINDS[itself])
            elif itself == "unittest.expectedFailure":
                self._site(node, itself, "xfail")
            elif dotted in _MARK_KINDS:
                self._site(node, dotted, _MARK_KINDS[dotted])
                self._consumed.add(id(func))
            elif isinstance(func, ast.Name) and func.id in self.aliases.expr:
                value = self.aliases.expr[func.id]
                base = _dotted(
                    value.func if isinstance(value, ast.Call) else value, self.aliases
                )
                if base in _MARK_KINDS:
                    self._site(node, base, _MARK_KINDS[base])
                    self._consumed.add(id(func))
            elif dotted in _CALL_KINDS:
                self._site(node, dotted, _CALL_KINDS[dotted])
                self._consumed.add(id(func))
            elif isinstance(func, ast.Attribute) and func.attr == "skipTest":
                self._site(node, "unittest.TestCase.skipTest", "skip")
                self._consumed.add(id(func))
        if dotted == "pytest.raises" or (
            isinstance(func, ast.Attribute)
            and func.attr.startswith("assert")
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            self._count()
        if isinstance(func, ast.Name) and func.id == "getattr":
            self._consumed.add(id(func))
            try:
                args, keywords = _literal_call_arguments(node)
                if keywords or len(args) not in (2, 3):
                    raise ValueError("unsupported getattr signature")
            except ValueError:
                args = []
                self._dynamic(node, "getattr(<unknown argument expansion>)")
            target = _dotted(args[0], self.aliases) if args else None
            if itself is None:
                self._dynamic(node, "unmodeled reflective lookup")
            attribute = (
                args[1].value
                if len(args) >= 2 and isinstance(args[1], ast.Constant)
                else None
            )
            if itself is None and attribute in _EXEMPTION_ATTRIBUTES:
                self._dynamic(node, f"unresolved {ast.unparse(node)}")
            if target and (
                len(args) < 2
                or not (
                    isinstance(args[1], ast.Constant) and isinstance(args[1].value, str)
                )
            ):
                self._dynamic(node, f"getattr({target}, <dynamic>)")
        if isinstance(func, ast.Name) and func.id == "__import__":
            self._dynamic(node, "__import__(...)")
        if (
            dotted is None
            and isinstance(func, ast.Attribute)
            and func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in ("pytest", "unittest", "_pytest")
        ):
            self._dynamic(node, f"import_module({node.args[0].value!r})")
        if isinstance(func, ast.Attribute) and func.attr in (
            "add_marker",
            "applymarker",
        ):
            self._consumed.add(id(func))
            try:
                args, kwargs = _literal_call_arguments(node)
                if len(args) > 2 or (args and "marker" in kwargs):
                    raise ValueError("ambiguous marker binding")
                arg = args[0] if args else kwargs.get("marker")
                if arg is None or set(kwargs) - {"marker", "append"}:
                    raise ValueError("unsupported marker binding")
            except ValueError:
                self._dynamic(node, "unmodeled marker application")
                self.generic_visit(node)
                return
            inner = arg.func if isinstance(arg, ast.Call) else arg
            resolved = (
                _dotted(arg, self.aliases)
                or _dotted(inner, self.aliases)
                or (isinstance(inner, ast.Name) and inner.id in self.aliases.expr)
            )
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value in ("skip", "skipif", "xfail"):
                    self._site(
                        node,
                        "pytest.mark." + arg.value,
                        "xfail" if arg.value == "xfail" else "skip",
                    )
            elif not (
                resolved is True
                or (
                    isinstance(resolved, str)
                    and resolved.startswith("pytest.mark.")
                    and len(resolved.split(".")) == 3
                    and not resolved.rsplit(".", 1)[-1].startswith("_")
                )
            ):
                self._dynamic(node, f"{func.attr}(<not a literal mark>)")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Metadata is an escape hatch from the supported marker API. Refuse
        # every operation on it, including aliases, reads, reflection and
        # mutations, rather than guessing which list/dict methods are safe.
        if node.attr in {
            "own_markers",
            "keywords",
            "pytestmark",
        } or node.attr.startswith("__unittest_"):
            self._dynamic(node, "unmodeled exemption metadata")
        if (
            node.attr in {"add_marker", "applymarker"}
            and id(node) not in self._consumed
        ):
            self._dynamic(node, "escaped marker application")
        if node.attr.startswith("__") and node.attr not in {
            "__unittest_skip__",
            "__unittest_skip_why__",
            "__unittest_expecting_failure__",
        }:
            self._dynamic(node, "unmodeled reflection attribute")
        if id(node) not in self._consumed:
            dotted = _dotted(node, self.aliases)
            if dotted in _MARK_KINDS:
                self._site(node, dotted, _MARK_KINDS[dotted])
                self._consumed.add(id(node))
                return
            if dotted is None and node.attr in _EXEMPTION_ATTRIBUTES:
                self._dynamic(node, f"unresolved {ast.unparse(node)}")
            if dotted == "unittest.expectedFailure":
                self._site(node, dotted, "xfail")
                return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if id(node) in self._consumed:
            return
        if isinstance(node.ctx, ast.Load) and node.id in _REFLECTION_PRIMITIVES:
            self._dynamic(node, "unmodeled reflection primitive")
        if node.id in self.aliases.expr and isinstance(node.ctx, ast.Load):
            value = self.aliases.expr[node.id]
            base = _dotted(
                value.func if isinstance(value, ast.Call) else value, self.aliases
            )
            if base in _MARK_KINDS:
                self._site(node, base, _MARK_KINDS[base])
            return
        dotted = self.aliases.dotted.get(node.id)
        if isinstance(node.ctx, ast.Load) and dotted in _MARK_KINDS:
            self._site(node, dotted, _MARK_KINDS[dotted])
            return
        if isinstance(node.ctx, ast.Load) and dotted == "unittest.expectedFailure":
            self._site(node, dotted, "xfail")


def scan_file(path: Path, rel: str) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_bytes(), filename=rel)
    except (SyntaxError, ValueError) as exc:
        return {
            "sites": [],
            "dynamic": [],
            "asserts": {},
            "parse_error": f"{rel}: {exc}",
        }
    aliases = _collect_aliases(tree)
    visitor = _SiteVisitor(rel, aliases)
    visitor.visit(tree)
    # Alias diagnostics originate in scope binding, before visiting the node.
    # Include their complete source statement, never only a generic label.
    statements = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
    counts: Counter[tuple[str, str, str]] = Counter()
    for dynamic in visitor.dynamic:
        if "expr" not in dynamic:
            enclosing = [
                node
                for node in statements
                if node.lineno <= dynamic["line"] <= (node.end_lineno or node.lineno)
            ]
            statement = min(
                enclosing,
                key=lambda node: (node.end_lineno or node.lineno) - node.lineno,
                default=tree,
            )
            dynamic["expr"] = ast.unparse(statement)
        key = (dynamic["owner"], dynamic["what"], dynamic["expr"])
        counts[key] += 1
        dynamic["occurrence"] = counts[key]
    return {
        "sites": visitor.sites,
        "dynamic": visitor.dynamic,
        "asserts": visitor.asserts,
        "parse_error": None,
    }


def scan_roots(collected_files: Iterable[str]) -> list[str]:
    """Where the static inventory looks when no --tests-root is given.

    Every top-level folder that holds a collected module (`tests` here), and
    each collected module or `conftest.py` that sits at the root itself.
    """
    roots = {path.split("/", 1)[0] for path in collected_files}
    roots.add("conftest.py")
    return sorted(roots)


def _python_files(root: Path, scan: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for entry in scan:
        base = root / entry
        if base.is_file() and base.suffix == ".py":
            files.add(base)
            continue
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _TREE_SKIP_DIRS)
            files.update(Path(dirpath) / n for n in filenames if n.endswith(".py"))
    return sorted(files)


def scan_tests(root: Path, scan: Sequence[str]) -> dict[str, Any]:
    sites: list[dict[str, Any]] = []
    dynamic: list[dict[str, Any]] = []
    asserts: dict[str, int] = {}
    parse_errors: list[str] = []
    for path in _python_files(root, scan):
        rel = path.relative_to(root).as_posix()
        result = scan_file(path, rel)
        sites.extend(result["sites"])
        dynamic.extend(result["dynamic"])
        asserts.update(result["asserts"])
        if result["parse_error"]:
            parse_errors.append(result["parse_error"])
    return {
        "roots": list(scan),
        "sites": sites,
        "dynamic": dynamic,
        "asserts": asserts,
        "parse_errors": parse_errors,
    }


def site_key(site: dict[str, Any]) -> str:
    return f"{site['file']}::{site['owner']} {site['expr']}"


def unknown_key(site: dict[str, Any]) -> str:
    # Versioned, expression-specific and occurrence-specific. No generic old
    # authorization may silently cover a different construction or duplicate.
    expression = site.get("expr")
    occurrence = site.get("occurrence")
    if not isinstance(expression, str) or type(occurrence) is not int or occurrence < 1:
        raise InventoryError(
            "unknown site lacks construction identity; recollect with this tool"
        )
    digest = hashlib.sha256(expression.encode()).hexdigest()
    return f"{site['file']}::{site['owner']} {site['what']} [v2:{digest}:{occurrence}]"


# --------------------------------------------------------------------------
# run manifests


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise InventoryError(f"{path}: not a {MANIFEST_SCHEMA} manifest")
    legs = manifest.get("legs")
    if not isinstance(legs, list):
        raise InventoryError(f"{path}: manifest has no legs")
    names = [leg.get("name") for leg in legs]
    if len(names) != len(set(names)):
        raise InventoryError(f"{path}: duplicate leg names")
    return manifest


def _leg_junit_path(manifest_path: Path, leg: dict[str, Any]) -> Path:
    junit = Path(str(leg.get("junit", "")))
    return junit if junit.is_absolute() else manifest_path.parent / junit


def cmd_run_leg(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    legs: list[dict[str, Any]] = (
        _load_manifest(manifest_path)["legs"] if manifest_path.exists() else []
    )
    if any(leg["name"] == args.name for leg in legs):
        raise InventoryError(f"leg {args.name!r} is already in {manifest_path}")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise InventoryError("run-leg needs a command after --")
    # Arbitrary wrappers can run another Python or change the child's env
    # behind the recorder. Accept only this interpreter's pytest, with
    # explicit leading env assignments applied before execution/recording.
    env = dict(os.environ)
    if command[0] == "env":
        command.pop(0)
        while command and re.match(r"^[A-Za-z_]\w*=", command[0]):
            name, value = command.pop(0).split("=", 1)
            env[name] = value
    executable = shutil.which(command[0], path=env.get("PATH")) if command else None
    if (
        executable is None
        or os.path.abspath(executable) != os.path.abspath(sys.executable)
        or command[1:3] != ["-m", "pytest"]
    ):
        raise InventoryError(
            "run-leg requires this tool's Python executable followed by -m pytest; prepare uv/environment before run-leg"
        )
    command[0] = sys.executable
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSAFEPATH"):
        if env.get(name) != os.environ.get(name):
            raise InventoryError(f"prepare {name} before invoking run-leg")
    select = list(args.select)
    if args.ids_from:
        # One test id per line, appended to the command as its own argument
        # (ids can hold spaces, which a shell loop would split).
        ids = [line for line in Path(args.ids_from).read_text().splitlines() if line]
        if not ids:
            raise InventoryError(f"--ids-from {args.ids_from} lists no ids")
        select += ids
        command += ids
    missing = [arg for arg in select if arg not in command]
    if missing:
        raise InventoryError("--select arguments are not in the command")
    recorded_command = sanitized_argv(command)
    if sanitized_argv(select) != select or any(
        raw in select and raw != safe
        for raw, safe in zip(command, recorded_command, strict=True)
    ):
        raise InventoryError("--select contains a credential-bearing argument")
    _pytest_configuration(Path(args.root), command[3:], env)
    junit = Path(args.junit)
    if not junit.is_absolute():
        junit = manifest_path.parent / junit
    if junit.exists():
        raise InventoryError(
            f"JUnit file already exists (each leg needs a fresh one): {junit}"
        )
    for leg in legs:
        if _leg_junit_path(manifest_path, leg).resolve() == junit.resolve():
            raise InventoryError(f"JUnit file reused by leg {leg['name']!r}")
    argv = _insert_pytest_options(
        command, [f"--junitxml={junit}", "-o", "junit_family=xunit1"]
    )
    # The tree the leg ran against: `collect` refuses a leg whose tree, cwd
    # or host is not the one it collects, so one arm's runs can never stand
    # in for the other's.
    tree = tree_digest(Path(args.root).resolve())
    started = _now()
    t0 = time.monotonic()
    runtime_path = junit.with_suffix(".runtime.json")
    if runtime_path.exists():
        raise InventoryError(f"runtime evidence already exists: {runtime_path}")
    with tempfile.TemporaryDirectory(prefix="suite-runtime-") as tmp:
        plugin = Path(tmp) / "_suite_inventory_collector.py"
        plugin.write_text(_COLLECTOR_PLUGIN + "\n_runtime = True\n")
        instrument_env = dict(env)
        instrument_env["PYTHONPATH"] = str(plugin.parent) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        instrument_argv = _insert_pytest_options(argv, ["-p", plugin.stem])
        proc = subprocess.run(
            _collector_argv(instrument_argv),
            cwd=args.root,
            env=instrument_env,
            check=False,
        )
        captures = []
        for path in sorted(plugin.parent.glob(plugin.stem + "-*.json")):
            try:
                captures.append(_read_json(path))
            except InventoryError:
                captures.append(
                    {
                        "items": [],
                        "reports": {},
                        "errors": ["invalid runtime exemption capture"],
                    }
                )
        if not captures:
            captures = [
                {
                    "items": [],
                    "reports": {},
                    "errors": ["missing runtime exemption capture; rerun with run-leg"],
                }
            ]
        _write_json(runtime_path, {"captures": captures})
    duration = round(time.monotonic() - t0, 3)
    try:
        rel_junit = str(junit.relative_to(manifest_path.parent))
    except ValueError:
        rel_junit = str(junit)
    legs.append(
        {
            "name": args.name,
            "role": args.role,
            "select_args": select,
            "worker_policy": "local-only",
            "argv": sanitized_argv(argv),
            "cwd": str(Path(args.root).resolve()),
            "tree": tree,
            "exit_code": proc.returncode,
            "junit": rel_junit,
            "junit_sha256": sha256_file(junit) if junit.exists() else None,
            "runtime_evidence": str(runtime_path),
            "runtime_sha256": sha256_file(runtime_path),
            "started": started,
            "duration_s": duration,
            "host": socket.gethostname(),
            "env": sanitized_env(env),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "packages": package_versions(),
        }
    )
    _write_json(manifest_path, {"schema": MANIFEST_SCHEMA, "legs": legs})
    print(
        f"leg {args.name} ({args.role}) exited {proc.returncode} in {duration} s; "
        f"JUnit {junit}",
        file=sys.stderr,
    )
    return proc.returncode


# --------------------------------------------------------------------------
# collect


def _leg_selections(
    root: Path,
    legs: list[dict[str, Any]],
    pytest_args: Sequence[str],
    redactions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    runs = []
    for leg in legs:
        run = run_collect_only(
            root, leg["select_args"], pytest_args, redactions=redactions
        )
        run["leg"] = leg["name"]
        run["select_args_sha256"] = _argv_sha256(leg["select_args"])
        runs.append(run)
    return runs


def _complete_collection(
    root: Path,
    legs: list[dict[str, Any]],
    pytest_args: Sequence[str],
    redactions: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not legs:
        raise InventoryError("manifest has no primary leg")
    # Split primary legs can use different discovery/plugin settings. Each
    # contributes its complete discovery, with selection filters removed.
    selections = dict.fromkeys(tuple(leg["select_args"]) for leg in legs)
    runs = [
        run_collect_only(
            root, selection, pytest_args, complete=True, redactions=redactions
        )
        for selection in selections
    ]
    result = dict(runs[0])
    if len(runs) > 1:
        result["discovery_runs"] = runs
        result["ids"] = sorted({i for run in runs for i in run["ids"]})
        result["order_sha256"] = hashlib.sha256(
            "\n".join(result["ids"]).encode()
        ).hexdigest()
        for key in ("errors", "module_reports", "rs_lines"):
            result[key] = [item for run in runs for item in run[key]]
        result["duration_s"] = round(sum(run["duration_s"] for run in runs), 3)
    return result


def _runtime(
    manifest_path: Path,
    manifest: dict[str, Any],
    selections: dict[str, list[str]],
    index: JunitIndex,
    origin: dict[str, str] | None,
    redactions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Map each leg's JUnit back to collected ids.

    ``origin`` (the collected root, this host and the collected tree's
    digest) is given for a run manifest: a leg that ran in another folder,
    on another host or against another tree is not this arm's evidence (a
    base manifest passed to the tip's collect, or one copied from another
    VM, would otherwise stand in for this arm's runs).
    """
    legs_out: list[dict[str, Any]] = []
    mark_runs: list[dict[str, Any]] = []
    errors: list[str] = []
    primary_cover: dict[str, list[str]] = {}
    for original in manifest["legs"]:
        # Older manifests may already contain raw argv. Do not propagate
        # those values into a newly retained inventory, even if refused.
        leg = dict(original)
        if isinstance(leg.get("argv"), list):
            leg["argv"] = sanitized_argv(leg["argv"], redactions)
        if isinstance(leg.get("select_args"), list):
            leg["select_args"] = sanitized_argv(leg["select_args"], redactions)
        name = leg["name"]
        role = leg.get("role")
        if role not in ("primary", "retry"):
            errors.append(f"leg {name}: role {role!r} is neither primary nor retry")
            continue
        if origin is not None:
            if leg.get("worker_policy") != "local-only":
                errors.append(
                    f"leg {name}: missing validated worker interpreter policy"
                )
            ran = {
                "cwd": leg.get("cwd"),
                "host": leg.get("host"),
                "tree": (leg.get("tree") or {}).get("sha256"),
            }
            for key, want in origin.items():
                if ran[key] != want:
                    errors.append(
                        f"leg {name}: ran with {key} {ran[key]!r}, not this arm's "
                        f"{want!r} (another arm's or another machine's run)"
                    )
            for key, want in (
                ("python", platform.python_version()),
                ("python_implementation", platform.python_implementation()),
            ):
                if leg.get(key) != want:
                    errors.append(
                        f"leg {name}: incompatible {key} (run under this collector's interpreter)"
                    )
        path = _leg_junit_path(manifest_path, leg)
        exit_code = leg.get("exit_code")
        # None only for `collect --junit`, which runtime compare refuses.
        if exit_code is not None and exit_code not in VALID_LEG_EXITS:
            errors.append(
                f"leg {name}: exited {exit_code} (interrupted or invalid run)"
            )
        recorded = leg.get("junit_sha256")
        if not path.exists():
            errors.append(f"leg {name}: JUnit file missing: {path}")
            legs_out.append({**leg, "outcomes": {}, "module_reports": []})
            continue
        if recorded and sha256_file(path) != recorded:
            errors.append(f"leg {name}: JUnit file changed since the leg ran: {path}")
        try:
            reports, modules, leg_errors = read_leg_junit(path, index)
        except InventoryError as exc:
            errors.append(f"leg {name}: {exc}")
            legs_out.append({**leg, "outcomes": {}, "module_reports": []})
            continue
        errors.extend(f"leg {name}: {e}" for e in leg_errors)
        selected = set(selections.get(name, []))
        if role == "primary":
            extra = sorted(set(reports) - selected)
            if extra:
                errors.append(
                    f"leg {name}: JUnit has ids outside the leg's recorded selection: {extra[:5]}"
                    f" ({len(extra)} in all)"
                )
            for nodeid in reports:
                primary_cover.setdefault(nodeid, []).append(name)
        outcomes: dict[str, Any] = {
            nodeid: {
                "outcome": outcome_of(children),
                "exemptions": exemptions_of(children),
                "testcases": sum(1 for c in children if c["tag"] == "testcase"),
                "detail": [c for c in children if c["tag"] != "testcase"][:4],
            }
            for nodeid, children in sorted(reports.items())
        }
        if origin is not None:
            try:
                evidence_path = Path(leg.get("runtime_evidence") or "")
                if not evidence_path.is_file() or sha256_file(evidence_path) != leg.get(
                    "runtime_sha256"
                ):
                    raise InventoryError(
                        "missing or changed runtime exemption evidence; rerun with run-leg"
                    )
                captures = _read_json(evidence_path)["captures"]
                covered: set[str] = set()
                for capture in captures:
                    mark_runs.append({"marked_items": capture["items"]})
                    covered.update(capture["executed"])
                    errors.extend(f"leg {name}: {error}" for error in capture["errors"])
                    if not set(capture["executed"]) <= {
                        item["id"] for item in capture["items"]
                    }:
                        raise InventoryError(
                            "executed item lacks runtime exemption snapshot"
                        )
                    for item in capture["items"]:
                        for mark in item["marks"]:
                            if mark["kind"] not in {
                                *_MARK_KINDS,
                                "unittest.skip",
                                "unittest.expectedFailure",
                            } or set(mark) != {"kind", "args", "kwargs"}:
                                raise InventoryError("invalid runtime exemption mark")
                            # Marks retain construction/argument evidence below.
                            # Only reports prove an active runtime exemption:
                            # skipif(False) is a supported passing test.
                    for nodeid, exemptions in capture["reports"].items():
                        if set(exemptions) - {"skip", "xfail"}:
                            raise InventoryError("invalid runtime exemption report")
                        if nodeid in outcomes:
                            for kind, reason in exemptions.items():
                                outcomes[nodeid]["exemptions"].setdefault(kind, reason)
                if set(outcomes) - covered:
                    raise InventoryError(
                        "incomplete runtime exemption capture; rerun with run-leg"
                    )
            except (InventoryError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"leg {name}: invalid runtime exemption evidence: {exc}")
        legs_out.append({**leg, "outcomes": outcomes, "module_reports": modules})
    if not any(leg.get("role") == "primary" for leg in manifest["legs"]):
        errors.append("manifest has no primary leg")
    sites = _collected_mark_sites(mark_runs)
    for site in sites:
        site["source"] = "pytest-runtime"
    return {
        "legs": legs_out,
        "errors": errors,
        "primary_cover": primary_cover,
        "sites": sites,
    }


def build_inventory(
    root: Path,
    legs: list[dict[str, Any]],
    pytest_args: Sequence[str],
    tests_roots: Sequence[str],
    tree_label: str | None,
    manifest_path: Path | None,
    manifest: dict[str, Any] | None,
    reuse: dict[str, Any] | None,
) -> dict[str, Any]:
    tool_before = _tool_sha256()
    current = provenance(root, tree_label)
    request = {
        "pytest_args": sanitized_argv(pytest_args),
        "pytest_args_sha256": _argv_sha256(pytest_args),
        "tests_roots": list(tests_roots),
    }
    errors: list[str] = []
    redactions = _selection_redactions(manifest) if manifest is not None else None
    if reuse is not None:
        collections = reuse["collections"]
        prov = reuse["provenance"]
        static = reuse["static"]
        if reuse["tool"]["sha256"] != tool_before:
            raise InventoryError("--collection-from was written by another tool copy")
        reused_legs = [
            (run.get("leg"), run.get("select_args_sha256"))
            for run in collections["selected"]
        ]
        if reused_legs != [
            (leg["name"], _argv_sha256(leg["select_args"])) for leg in legs
        ]:
            raise InventoryError(
                "--collection-from was collected for other primary legs"
            )
        if prov.get("root") != str(root):
            raise InventoryError("--collection-from was collected in another tree")
        # The same path can hold another tree (each tip is re-extracted into
        # one gate folder), and the reused inventory's host stands for this run.
        if tree_digest(root)["sha256"] != prov["tree"]["sha256"]:
            raise InventoryError(
                "--collection-from was collected from other contents of this tree"
            )
        if prov.get("host") != socket.gethostname():
            raise InventoryError("--collection-from was collected on another host")
        for key in (
            "env",
            "python",
            "python_implementation",
            "executable",
            "packages",
            "docker",
            "platform",
            "machine",
        ):
            if prov.get(key) != current.get(key):
                raise InventoryError(
                    f"--collection-from has incompatible current {key}"
                )
        if reuse.get("collection_request") != request:
            raise InventoryError(
                "--collection-from has incompatible collection arguments or tests roots"
            )
    else:
        prov = current
        collections: dict[str, Any] = {
            "selected": _leg_selections(root, legs, pytest_args, redactions),
            "all": _complete_collection(root, legs, pytest_args, redactions),
        }
        collected_files = {id_file(i) for i in collections["all"]["ids"]} | {
            m["id"]
            for run in [*collections["selected"], collections["all"]]
            for m in run["module_reports"]
        }
        static = scan_tests(root, tests_roots or scan_roots(collected_files))
        complete = collections["all"]
        static["sites"].extend(
            _collected_mark_sites(
                [*collections["selected"], *complete.get("discovery_runs", [complete])]
            )
        )
    for run in [*collections["selected"], collections["all"]]:
        errors.extend(run["errors"])
    selected_by_leg = {run["leg"]: run["ids"] for run in collections["selected"]}
    selected = sorted({i for ids in selected_by_leg.values() for i in ids})
    all_ids = sorted(set(collections["all"]["ids"]))
    outside = sorted(set(selected) - set(all_ids))
    if outside:
        # A path named on the command line is collected even when
        # collect_ignore or a pytest_ignore_collect hook hides it from the
        # whole tree, so the leg would run tests the tree no longer has.
        errors.append(
            "ids selected but not collected with -o addopts= (a leg names a path "
            f"the whole-tree collection hides): {outside[:5]} ({len(outside)})"
        )
    module_reports: dict[str, dict[str, str]] = {}
    for run in [*collections["selected"], collections["all"]]:
        for report in run["module_reports"]:
            module_reports.setdefault(report["id"], report)
    runtime = None
    if manifest is not None:
        # `collect --junit` has no manifest file; its JUnit paths are absolute.
        where = manifest_path or root / "manifest.json"
        index = JunitIndex([*all_ids, *selected])
        origin = (
            {
                "cwd": str(root),
                "host": socket.gethostname(),
                "tree": prov["tree"]["sha256"],
            }
            if manifest_path is not None
            else None
        )
        runtime = _runtime(where, manifest, selected_by_leg, index, origin, redactions)
        errors.extend(runtime["errors"])
    tool_after = _tool_sha256()
    if tool_after != tool_before:
        errors.append("the inventory tool changed while it ran")
    return {
        "schema": SCHEMA,
        "tool": {"path": str(Path(__file__).resolve()), "sha256": tool_before},
        "provenance": prov,
        "collection_request": request,
        "collections": collections,
        "selected_ids": selected,
        "all_ids": all_ids,
        "module_reports": sorted(module_reports.values(), key=lambda m: m["id"]),
        "static": static,
        "runtime": runtime,
        "evidence_errors": errors,
    }


def cmd_collect(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if args.expect_tool_sha256 and _tool_sha256() != args.expect_tool_sha256:
        raise InventoryError("tool copy does not match --expect-tool-sha256")
    manifest_path = Path(args.run_manifest).resolve() if args.run_manifest else None
    if manifest_path is not None and args.junit:
        raise InventoryError("give --run-manifest or --junit, not both")
    if (manifest_path is not None or args.junit) and args.leg:
        raise InventoryError(
            "--leg is for collection only; with --run-manifest the primary legs' "
            "selections come from it, and --junit uses the default selection"
        )
    manifest: dict[str, Any] | None = None
    if args.junit:
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "legs": [
                {
                    "name": f"junit-{n}",
                    "role": "primary",
                    "select_args": [],
                    "argv": None,
                    "exit_code": None,
                    "junit": str(Path(path).resolve()),
                    "junit_sha256": None,
                }
                for n, path in enumerate(args.junit, 1)
            ],
        }
    elif manifest_path is not None:
        manifest = _load_manifest(manifest_path)
    if manifest is not None:
        legs = [
            {"name": leg["name"], "select_args": list(leg.get("select_args") or [])}
            for leg in manifest["legs"]
            if leg.get("role") == "primary"
        ]
    else:
        leg_args = args.leg or [""]
        legs = [
            {"name": f"leg-{n}", "select_args": shlex.split(spec)}
            for n, spec in enumerate(leg_args, 1)
        ]
    reuse = _read_json(Path(args.collection_from)) if args.collection_from else None
    inventory = build_inventory(
        root,
        legs,
        args.pytest_arg,
        args.tests_root,
        args.tree_label,
        manifest_path,
        manifest,
        reuse,
    )
    if args.junit:
        inventory["runtime_source"] = "--junit (no run manifest: exit codes unknown)"
    _write_json(Path(args.out), inventory)
    selected = len(inventory["selected_ids"])
    print(
        f"collected {len(inventory['all_ids'])} ids, {selected} selected, "
        f"{len(inventory['module_reports'])} module reports, "
        f"{len(inventory['static']['sites'])} skip/xfail sites; "
        f"{len(inventory['evidence_errors'])} evidence errors -> {args.out}",
        file=sys.stderr,
    )
    for error in inventory["evidence_errors"]:
        print(f"evidence error: {error}", file=sys.stderr)
    return EXIT_INVALID if inventory["evidence_errors"] else EXIT_OK


# --------------------------------------------------------------------------
# views of one inventory


def _load_inventory(path: Path) -> dict[str, Any]:
    inventory = _read_json(path)
    if inventory.get("schema") != SCHEMA:
        raise InventoryError(f"{path}: not a {SCHEMA} inventory")
    return inventory


def primary_outcomes(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """id -> merged primary-leg outcome (worst outcome, every exemption)."""
    rank = {"failed": 4, "error": 4, "xfailed": 3, "skipped": 2, "passed": 1}
    merged: dict[str, dict[str, Any]] = {}
    runtime = inventory.get("runtime") or {}
    for leg in runtime.get("legs", []):
        if leg.get("role") != "primary":
            continue
        for nodeid, result in leg["outcomes"].items():
            current = merged.get(nodeid)
            if current is None:
                merged[nodeid] = {
                    "outcome": result["outcome"],
                    "exemptions": dict(result["exemptions"]),
                    "legs": [leg["name"]],
                }
                continue
            if rank[result["outcome"]] > rank[current["outcome"]]:
                current["outcome"] = result["outcome"]
            for kind, reason in result["exemptions"].items():
                current["exemptions"].setdefault(kind, reason)
            current["legs"].append(leg["name"])
    return merged


def all_exemptions(inventory: dict[str, Any]) -> dict[str, dict[str, str]]:
    """id -> every skip/xfail seen in any leg, retries included."""
    found: dict[str, dict[str, str]] = {}
    runtime = inventory.get("runtime") or {}
    for leg in runtime.get("legs", []):
        for nodeid, result in leg["outcomes"].items():
            for kind, reason in result["exemptions"].items():
                found.setdefault(nodeid, {}).setdefault(kind, reason)
    return found


def retry_passes(inventory: dict[str, Any]) -> set[str]:
    passed: set[str] = set()
    runtime = inventory.get("runtime") or {}
    for leg in runtime.get("legs", []):
        if leg.get("role") != "retry":
            continue
        for nodeid, result in leg["outcomes"].items():
            if result["outcome"] == "passed":
                passed.add(nodeid)
    return passed


def runtime_modules(inventory: dict[str, Any]) -> list[dict[str, str]]:
    runtime = inventory.get("runtime") or {}
    found = []
    for leg in runtime.get("legs", []):
        found.extend(leg.get("module_reports", []))
    return found


def cmd_skips(args: argparse.Namespace) -> int:
    inventory = _load_inventory(Path(args.inventory))
    lines = set()
    for nodeid, kinds in all_exemptions(inventory).items():
        for kind, reason in kinds.items():
            lines.add(f"{nodeid}\t{kind}\t{_one_line(reason)}")
    for report in [*inventory["module_reports"], *runtime_modules(inventory)]:
        kind = "module-skip" if report["outcome"] == "skipped" else "module-error"
        lines.add(f"{report['id']}\t{kind}\t{_one_line(report['reason'])}")
    Path(args.out).write_text("".join(f"{line}\n" for line in sorted(lines)))
    print(f"{len(lines)} skip/xfail lines -> {args.out}", file=sys.stderr)
    return EXIT_OK


def cmd_ids(args: argparse.Namespace) -> int:
    inventory = _load_inventory(Path(args.inventory))
    ids = (
        inventory["selected_ids"] if args.which == "selected" else inventory["all_ids"]
    )
    Path(args.out).write_text("".join(f"{i}\n" for i in ids))
    return EXIT_OK


def cmd_failed(args: argparse.Namespace) -> int:
    inventory = _load_inventory(Path(args.inventory))
    failed = sorted(
        nodeid
        for nodeid, result in primary_outcomes(inventory).items()
        if result["outcome"] in ("failed", "error")
    )
    sys.stdout.write("".join(f"{i}\n" for i in failed))
    return EXIT_OK


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


# --------------------------------------------------------------------------
# decisions and the allow file


# The round's internal decisions (PY8-Dn) or, in public CI, a pull request.
_DECISION_REF = r"(?:PY8-D|PR-)\d+"


def parse_decisions(text: str) -> list[dict[str, str]]:
    """Every decision entry: its number, header (author/date/question) and text.

    Two shapes: a table row ``| PY8-Dn | question | decision | reason |``
    and a bullet ``- **PY8-Dn (author, date):** text`` with its indented
    continuation lines. Public CI cites ``PR-<number>`` entries instead, in the
    same bullet shape, from a file written from the pull request's description.
    """
    entries: list[dict[str, str]] = []
    section = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        row = re.match(r"^\|\s*(" + _DECISION_REF + r")\s*\|([^|]*)\|(.*)$", line)
        bullet = re.match(r"^- \*\*(" + _DECISION_REF + r")\b([^*]*)\*\*(.*)$", line)
        if row:
            entries.append(
                {
                    "number": row.group(1),
                    "header": f"{row.group(2).strip()} [{section}]",
                    "text": line,
                }
            )
        elif bullet:
            body = [line]
            j = i + 1
            while (
                j < len(lines) and lines[j].startswith((" ", "\t")) and lines[j].strip()
            ):
                body.append(lines[j])
                j += 1
            entries.append(
                {
                    "number": bullet.group(1),
                    "header": f"{bullet.group(2).strip()} [{section}]",
                    "text": "\n".join(body),
                }
            )
            i = j
            continue
        i += 1
    return entries


def resolve_decision(
    ref: str, entries: list[dict[str, str]]
) -> tuple[dict[str, str] | None, str]:
    """The one entry a ``PY8-Dn[@discriminator...]`` (or ``PR-n``) ref names, or why not."""
    parts = ref.split("@")
    number, discriminators = parts[0], [p for p in parts[1:] if p]
    if not re.fullmatch(_DECISION_REF, number):
        return None, (
            f"decision ref {ref!r} is not PY8-D<number> or PR-<number>[@discriminator]"
        )
    matches = [
        e
        for e in entries
        if e["number"] == number and all(d in e["header"] for d in discriminators)
    ]
    if not matches:
        return None, f"{ref} matches no decision entry"
    if len(matches) > 1:
        return None, (
            f"{ref} matches {len(matches)} decision entries; add an @author, @date or "
            "@section discriminator"
        )
    return matches[0], ""


def _names_exactly(text: str, token: str) -> bool:
    """True when ``token`` appears whole: not inside a longer word or test id.

    ``tests/a.py::test_x`` is not named by ``tests/a.py::test_xy``,
    ``tests/a.py::test_x[1]`` or ``tests/a.py::test_x::sub``.
    """
    pattern = r"(?<![\w\-./])(?<!::)" + re.escape(token) + r"(?![\w\-/\[]|::)"
    return re.search(pattern, text) is not None


def authorize(line: dict[str, str], entries: list[dict[str, str]]) -> str:
    """'' when the allow line's decision names its kind and exact ids."""
    entry, why = resolve_decision(line["decision"], entries)
    if entry is None:
        return why
    text = entry["text"]
    missing = []
    if not _names_exactly(text, line["kind"]):
        missing.append(f"the kind {line['kind']!r}")
    if not _names_exactly(text, line["id"]):
        missing.append(f"the id {line['id']!r}")
    if line["new_id"] != "-" and not _names_exactly(text, line["new_id"]):
        missing.append(f"the new id {line['new_id']!r}")
    if missing:
        return f"{line['decision']} does not name {', '.join(missing)}"
    return ""


def parse_allow(text: str) -> list[dict[str, str]]:
    lines = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 6:
            raise InventoryError(
                f"allow line {number}: {len(fields)} tab-separated fields, want 6 "
                "(kind, id, new_id or -, decision_ref, owner, reason)"
            )
        kind, nodeid, new_id, decision, owner, reason = (f.strip() for f in fields)
        if kind not in ALLOW_KINDS:
            raise InventoryError(f"allow line {number}: unknown kind {kind!r}")
        if (kind == RENAMED) == (new_id == "-"):
            raise InventoryError(
                f"allow line {number}: a renamed line names its new id; other kinds use -"
            )
        if not nodeid or not decision or not owner:
            raise InventoryError(f"allow line {number}: empty id, decision or owner")
        lines.append(
            {
                "line": str(number),
                "kind": kind,
                "id": nodeid,
                "new_id": new_id,
                "decision": decision,
                "owner": owner,
                "reason": reason,
            }
        )
    return lines


# --------------------------------------------------------------------------
# compare


def _in_scope(path: str, only: Sequence[str]) -> bool:
    if not only:
        return True
    return any(path == o or path.startswith(o.rstrip("/") + "/") for o in only)


def _leg_env_values(inv: dict[str, Any]) -> dict[str, set[str | None]]:
    """Each compared variable's values over an arm's run legs (None: unset)."""
    legs = [
        leg["env"].get("compared", {})
        for leg in (inv.get("runtime") or {}).get("legs", [])
        if isinstance(leg.get("env"), dict)
    ]
    names = {name for env in legs for name in env}
    return {name: {env.get(name) for env in legs} for name in names}


def _show(values: set[str | None]) -> str:
    ordered = sorted(values, key=lambda v: (v is not None, v or ""))
    return ", ".join(repr(v) for v in ordered)


def _compat(
    base: dict[str, Any], tip: dict[str, Any], runtime: bool
) -> tuple[list[str], list[str]]:
    """(blocking incompatibilities, differences to review)."""
    blocking: list[str] = []
    review: list[str] = []
    if base["tool"]["sha256"] != tip["tool"]["sha256"]:
        blocking.append("the arms were collected by different tool copies")
    pb, pt = base["provenance"], tip["provenance"]
    for key in ("python", "python_implementation", "machine"):
        if pb.get(key) != pt.get(key):
            (blocking if runtime else review).append(
                f"{key}: base {pb.get(key)!r}, tip {pt.get(key)!r}"
            )
    if pb.get("host") != pt.get("host"):
        (blocking if runtime else review).append(
            f"host: base {pb.get('host')!r}, tip {pt.get('host')!r}"
        )
    if pb.get("docker") != pt.get("docker"):
        (blocking if runtime else review).append(
            f"docker: base {pb.get('docker')}, tip {pt.get('docker')}"
        )
    eb, et = pb["env"]["compared"], pt["env"]["compared"]
    for name in sorted(set(eb) | set(et)):
        if eb.get(name) != et.get(name):
            (blocking if runtime else review).append(
                f"environment {name}: base {eb.get(name)!r}, tip {et.get(name)!r}"
            )
    if runtime:
        # The skips come from the legs, so their environments must match too
        # (collect can run in a shell the runs did not). Each variable's set
        # of values over an arm's legs must be the same in both arms.
        lb, lt = _leg_env_values(base), _leg_env_values(tip)
        for name in sorted(set(lb) | set(lt)):
            vb, vt = lb.get(name, {None}), lt.get(name, {None})
            if vb != vt:
                blocking.append(
                    f"leg environment {name}: base {_show(vb)}, tip {_show(vt)}"
                )
    for name, version in sorted(pb["packages"].items()):
        if pt["packages"].get(name) != version:
            review.append(
                f"package {name}: base {version}, tip {pt['packages'].get(name)} "
                "(different locks: review)"
            )
    sel_b = [run["argv"] for run in base["collections"]["selected"]]
    sel_t = [run["argv"] for run in tip["collections"]["selected"]]
    if sel_b != sel_t:
        review.append(f"selection argv differs: base {sel_b}, tip {sel_t}")
    return blocking, review


def _finding(kind: str, nodeid: str, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "id": nodeid, **extra}


def compare(
    base: dict[str, Any],
    tip: dict[str, Any],
    mode: str,
    allow: list[dict[str, str]],
    decisions: list[dict[str, str]],
    strict_allow: bool,
    only: Sequence[str],
) -> dict[str, Any]:
    runtime = mode == "runtime"
    invalid: list[str] = []
    for arm, inv in (("base", base), ("tip", tip)):
        invalid.extend(f"{arm}: {e}" for e in inv["evidence_errors"])
        if runtime and not inv.get("runtime"):
            invalid.append(
                f"{arm}: no run evidence (collect --run-manifest) for runtime mode"
            )
        if runtime and inv.get("runtime_source"):
            invalid.append(f"{arm}: {inv['runtime_source']}")
    blocking, review = _compat(base, tip, runtime)
    invalid.extend(f"incompatible arms: {b}" for b in blocking)

    findings: list[dict[str, Any]] = []
    report: dict[str, list[Any]] = {
        "pending": [],
        "inherited_blind_spots": [],
        "inherited_module_skips": [],
        "new_skip_sites": [],
        "removed_xfail_sites": [],
        "assertion_drops": [],
        "new_dynamic": [],
        "outcome_changes": [],
        "skip_reason_changes": [],
        "new_ids": [],
        "stale_allow": [],
        "out_of_scope_allow": [],
    }
    base_all, tip_all = set(base["all_ids"]), set(tip["all_ids"])
    base_sel, tip_sel = set(base["selected_ids"]), set(tip["selected_ids"])

    def mods(inv: dict[str, Any]) -> dict[str, dict[str, str]]:
        found: dict[str, dict[str, str]] = {}
        sources = inv["module_reports"] + (runtime_modules(inv) if runtime else [])
        for report_ in sources:
            current = found.get(report_["id"])
            if current is None or report_["outcome"] == "error":
                found[report_["id"]] = report_
        return found

    base_mods, tip_mods = mods(base), mods(tip)
    base_files = {id_file(i) for i in base_all}
    hidden_by: dict[str, list[str]] = {}
    for module, rep in sorted(tip_mods.items()):
        before = base_mods.get(module)
        hidden = sorted(
            i
            for i in base_all - tip_all
            if i == module or i.startswith((module + "::", module + "/"))
        )
        if before is not None and before["outcome"] == rep["outcome"]:
            key = (
                "inherited_blind_spots"
                if rep["outcome"] == "error"
                else "inherited_module_skips"
            )
            report[key].append(
                {"id": module, "outcome": rep["outcome"], "reason": rep["reason"]}
            )
            continue
        kind = MODULE_ERROR if rep["outcome"] == "error" else MODULE_SKIPPED
        hidden_by[module] = hidden
        findings.append(
            _finding(
                kind,
                module,
                reason=rep["reason"],
                hidden_ids=hidden,
                new_module=module not in base_files and before is None,
            )
        )
    hidden_ids = {i for ids in hidden_by.values() for i in ids}

    for nodeid in sorted(base_all - tip_all):
        if nodeid not in hidden_ids:
            findings.append(
                _finding(REMOVED, nodeid, selected_at_base=nodeid in base_sel)
            )
    for nodeid in sorted((base_sel - tip_sel) & tip_all):
        findings.append(_finding(DESELECTED, nodeid))
    report["new_ids"] = sorted(tip_all - base_all)

    # Static sites: a multiset compare keyed by file, owner and expression.
    def exemption_sites(inv: dict[str, Any]) -> list[dict[str, Any]]:
        sites = list(inv["static"]["sites"])
        if runtime:
            # Collection and runtime see many of the same marks. Union their
            # multisets rather than double-counting observations; runtime-only
            # variants and increased multiplicity still require authorization.
            counts = Counter(site_key(s) for s in sites)
            seen: Counter[str] = Counter()
            for site in (inv.get("runtime") or {}).get("sites", []):
                key = site_key(site)
                seen[key] += 1
                if seen[key] > counts[key]:
                    sites.append(site)
        return sites

    b_sites, t_sites = exemption_sites(base), exemption_sites(tip)
    base_sites = Counter(site_key(s) for s in b_sites)
    tip_sites = Counter(site_key(s) for s in t_sites)
    families = {site_key(s): s for s in [*b_sites, *t_sites]}
    for key in sorted(tip_sites):
        added = tip_sites[key] - base_sites.get(key, 0)
        if added <= 0:
            continue
        site = families[key]
        if site["family"] == "xfail":
            findings.append(
                _finding(
                    XFAIL_SITE,
                    key,
                    file=site["file"],
                    line=site["line"],
                    added=added,
                    **({"source": site["source"]} if "source" in site else {}),
                )
            )
        else:
            report["new_skip_sites"].append(
                {"site": key, "line": site["line"], "added": added}
            )
    for key in sorted(base_sites):
        if families[key]["family"] == "xfail" and base_sites[key] > tip_sites.get(
            key, 0
        ):
            report["removed_xfail_sites"].append(key)
    base_asserts, tip_asserts = base["static"]["asserts"], tip["static"]["asserts"]
    for test, count in sorted(base_asserts.items()):
        if test in tip_asserts and tip_asserts[test] < count:
            report["assertion_drops"].append(
                {"test": test, "base": count, "tip": tip_asserts[test]}
            )
    base_dyn = Counter(unknown_key(d) for d in base["static"]["dynamic"])
    for d in tip["static"]["dynamic"]:
        key = unknown_key(d)
        if base_dyn[key] > 0:
            base_dyn[key] -= 1
        else:
            report["new_dynamic"].append(d)
        # Uncertainty is not made safe by appearing in the base too. Require
        # an exact-site decision until it is rewritten into a supported form.
        findings.append(
            _finding(
                UNKNOWN_EXEMPTION,
                key,
                file=d["file"],
                line=d["line"],
                construction=d["expr"],
                reason="cannot attribute an exemption with certainty; use a supported literal form or list this exact site in the allow file",
            )
        )

    if runtime and not invalid:
        b_out, t_out = primary_outcomes(base), primary_outcomes(tip)
        b_ex, t_ex = all_exemptions(base), all_exemptions(tip)
        for nodeid in sorted(base_sel - set(b_out)):
            invalid.append(f"base: selected id has no primary outcome: {nodeid}")
        for nodeid in sorted(tip_sel - set(t_out)):
            findings.append(_finding(NOT_RUN, nodeid))
        for nodeid in sorted(t_ex):
            new_kinds = set(t_ex[nodeid]) - set(b_ex.get(nodeid, {}))
            for kind in sorted(new_kinds):
                findings.append(
                    _finding(
                        NEW_XFAIL if kind == "xfail" else NEW_SKIP,
                        nodeid,
                        reason=t_ex[nodeid][kind],
                        base_exemptions=b_ex.get(nodeid, {}),
                        new_test=nodeid not in base_all,
                    )
                )
            for kind in sorted(set(t_ex[nodeid]) & set(b_ex.get(nodeid, {}))):
                if t_ex[nodeid][kind] != b_ex[nodeid][kind]:
                    report["skip_reason_changes"].append(
                        {
                            "id": nodeid,
                            "kind": kind,
                            "base": b_ex[nodeid][kind],
                            "tip": t_ex[nodeid][kind],
                        }
                    )
        for nodeid in sorted(set(b_out) & set(t_out)):
            if b_out[nodeid]["outcome"] != t_out[nodeid]["outcome"]:
                report["outcome_changes"].append(
                    {
                        "id": nodeid,
                        "base": b_out[nodeid]["outcome"],
                        "tip": t_out[nodeid]["outcome"],
                    }
                )
    elif not runtime:
        report["pending"].append(
            "collection mode: run outcomes (new skips, new xfails, ids with no result) "
            "were not checked"
        )

    # Scope (--only) is a declared fixture or mutant scope; say so.
    def finding_path(f: dict[str, Any]) -> str:
        return f.get("file") or id_file(f["id"])

    def item_path(item: Any) -> str | None:
        if isinstance(item, dict):
            for field in ("file", "id", "test", "site"):
                if item.get(field):
                    return id_file(str(item[field])).split(" ", 1)[0]
            return None
        text = str(item)
        return id_file(text).split(" ", 1)[0] if text.startswith(tuple(only)) else None

    findings = [f for f in findings if _in_scope(finding_path(f), only)]
    if only:
        for key in report:
            if key == "pending":
                continue
            kept = []
            for item in report[key]:
                path = item_path(item)
                if path is None or _in_scope(path, only):
                    kept.append(item)
            report[key] = kept

    # Allow lines.
    t_out_all = primary_outcomes(tip) if runtime and not invalid else {}
    t_retry = retry_passes(tip) if runtime and not invalid else set()
    allow_errors: list[dict[str, Any]] = []
    used: set[str] = set()
    for f in findings:
        f["allowed_by"] = None
    for line in allow:
        target_kind = REMOVED if line["kind"] == RENAMED else line["kind"]
        matched = [
            f for f in findings if f["kind"] == target_kind and f["id"] == line["id"]
        ]
        if not matched:
            in_scope = _in_scope(id_file(line["id"]).split(" ", 1)[0], only)
            if not in_scope:
                report["out_of_scope_allow"].append(line)
                continue
            stale = dict(line)
            stale["authorization"] = authorize(line, decisions) or "ok"
            report["stale_allow"].append(stale)
            if strict_allow:
                allow_errors.append(
                    _finding(
                        STALE_ALLOW,
                        line["id"],
                        line=line["line"],
                        allow_kind=line["kind"],
                    )
                )
            continue
        why = authorize(line, decisions)
        if not why and line["kind"] == RENAMED:
            new = line["new_id"]
            if new not in tip_all:
                why = f"renamed: new id {new} is not collected at the tip"
            elif new in base_all:
                why = f"renamed: new id {new} already existed at the base"
            elif line["id"] in base_sel and new not in tip_sel:
                # Decidable at collection time: a test the base ran, renamed
                # into a deselected id (a new `live` mark), never runs again.
                why = f"renamed: new id {new} is not selected at the tip"
            elif runtime:
                result = t_out_all.get(new)
                passed = result is not None and (
                    result["outcome"] == "passed"
                    or (result["outcome"] in ("failed", "error") and new in t_retry)
                )
                if not passed:
                    got = result["outcome"] if result else "no primary result"
                    why = f"renamed: new id {new} did not pass in the tip's run ({got})"
            else:
                report["pending"].append(
                    f"renamed {line['id']} -> {new}: the new id exists; its pass is checked "
                    "in runtime mode"
                )
        if why:
            allow_errors.append(
                _finding(
                    UNAUTHORIZED_ALLOW,
                    line["id"],
                    line=line["line"],
                    allow_kind=line["kind"],
                    why=why,
                )
            )
            continue
        for f in matched:
            f["allowed_by"] = f"allow line {line['line']} ({line['decision']})"
        used.add(line["line"])

    open_findings = [f for f in findings if not f["allowed_by"]]
    if invalid:
        exit_code = EXIT_INVALID
    elif open_findings or allow_errors:
        exit_code = EXIT_FINDINGS
    else:
        exit_code = EXIT_OK
    return {
        "schema": "suite-inventory-compare/1",
        "mode": mode,
        "strict_allow": strict_allow,
        "scope": list(only),
        "exit_code": exit_code,
        "invalid": invalid,
        "findings": findings,
        "allow_errors": allow_errors,
        "report_only": report,
        "review": review,
        "arms": {
            arm: {
                "tree_label": inv["provenance"].get("tree_label"),
                "tree_sha256": inv["provenance"]["tree"]["sha256"],
                "host": inv["provenance"].get("host"),
                "python": inv["provenance"].get("python"),
                "all": len(inv["all_ids"]),
                "selected": len(inv["selected_ids"]),
                "tool_sha256": inv["tool"]["sha256"],
            }
            for arm, inv in (("base", base), ("tip", tip))
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    out: list[str] = []
    arms = result["arms"]
    out.append(
        f"# Test inventory compare ({result['mode']}): exit {result['exit_code']}"
    )
    out.append("")
    out.append(
        f"Base {arms['base']['tree_label'] or arms['base']['tree_sha256'][:12]} "
        f"({arms['base']['all']} collected, {arms['base']['selected']} selected); "
        f"tip {arms['tip']['tree_label'] or arms['tip']['tree_sha256'][:12]} "
        f"({arms['tip']['all']} collected, {arms['tip']['selected']} selected); "
        f"host {arms['tip']['host']}, Python {arms['tip']['python']}."
    )
    if result["scope"]:
        out.append("")
        out.append(f"Scope limited to: {', '.join(result['scope'])} (`--only`).")

    def section(title: str, rows: list[str]) -> None:
        if rows:
            out.append("")
            out.append(f"## {title}")
            out.append("")
            out.extend(f"- {row}" for row in rows)

    section("Invalid or incomplete evidence (exit 2)", result["invalid"])
    section(
        "Findings without an allow line",
        [_describe(f) for f in result["findings"] if not f["allowed_by"]],
    )
    section(
        "Allow-line errors",
        [
            f"line {e['line']} ({e['allow_kind']} {e['id']}): {e.get('why', 'stale: matches no finding')}"
            for e in result["allow_errors"]
        ],
    )
    section(
        "Findings covered by an allow line",
        [
            f"{_describe(f)}: {f['allowed_by']}"
            for f in result["findings"]
            if f["allowed_by"]
        ],
    )
    rep = result["report_only"]
    section("Pending (not checked in this mode)", rep["pending"])
    section(
        "Collection errors in both arms (inherited blind spots; their tests are not verified)",
        [
            f"{m['id']}: {_one_line(m['reason'])[:200]}"
            for m in rep["inherited_blind_spots"]
        ],
    )
    section(
        "Module skips in both arms",
        [
            f"{m['id']}: {_one_line(m['reason'])[:200]}"
            for m in rep["inherited_module_skips"]
        ],
    )
    section(
        "New skip-family sites (report only)",
        [f"{s['site']} (line {s['line']})" for s in rep["new_skip_sites"]],
    )
    section("Removed xfail sites (report only)", rep["removed_xfail_sites"])
    section(
        "Fewer assert/pytest.raises in a surviving test (report only)",
        [f"{a['test']}: {a['base']} -> {a['tip']}" for a in rep["assertion_drops"]],
    )
    section(
        "New unresolved constructions (require unknown-exemption authorization)",
        [
            f"{d['file']}:{d['line']} {d['owner']}: {d['what']}"
            for d in rep["new_dynamic"]
        ],
    )
    section(
        "Outcome changes (report only; flapping ids in an A/A)",
        [f"{c['id']}: {c['base']} -> {c['tip']}" for c in rep["outcome_changes"]],
    )
    section(
        "Skip reason changes (report only)",
        [
            f"{c['id']} ({c['kind']}): {c['base']!r} -> {c['tip']!r}"
            for c in rep["skip_reason_changes"]
        ],
    )
    section(
        "Stale allow lines"
        + (" (red under --strict-allow)" if result["strict_allow"] else ""),
        [
            f"line {s['line']}: {s['kind']} {s['id']} ({s['decision']}; authorization: {s['authorization']})"
            for s in rep["stale_allow"]
        ],
    )
    section("Differences to review", result["review"])
    if len(out) <= 3 + (2 if result["scope"] else 0):
        out.append("")
        out.append("No findings.")
    return "\n".join(out) + "\n"


def _describe(f: dict[str, Any]) -> str:
    text = f"{f['kind']}: {f['id']}"
    if f["kind"] in (MODULE_SKIPPED, MODULE_ERROR):
        text += f" ({_one_line(f.get('reason', ''))[:160]}; hides {len(f.get('hidden_ids', []))} base ids)"
    elif f["kind"] in (NEW_SKIP, NEW_XFAIL):
        text += f" ({_one_line(f.get('reason', ''))[:160]})"
    elif f["kind"] == UNKNOWN_EXEMPTION:
        text += f" ({f['reason']})"
    return text


def cmd_compare(args: argparse.Namespace) -> int:
    base = _load_inventory(Path(args.base))
    tip = _load_inventory(Path(args.tip))
    if args.expect_tool_sha256:
        for arm, inv in (("base", base), ("tip", tip)):
            if inv["tool"]["sha256"] != args.expect_tool_sha256:
                raise InventoryError(
                    f"{arm} inventory was written by another tool copy"
                )
    allow = parse_allow(Path(args.allow).read_text()) if args.allow else []
    try:
        decisions = (
            parse_decisions(Path(args.decisions).read_text()) if args.decisions else []
        )
    except OSError as exc:
        raise InventoryError(f"cannot read decisions: {exc}") from exc
    result = compare(
        base, tip, args.mode, allow, decisions, args.strict_allow, args.only
    )
    markdown = render_markdown(result)
    if args.out:
        prefix = Path(args.out)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{prefix}.md").write_text(markdown)
        _write_json(Path(f"{prefix}.json"), result)
    sys.stdout.write(markdown)
    return int(result["exit_code"])


# --------------------------------------------------------------------------
# command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="suite_inventory.py",
        description="Record and compare which tests exist, run, skip and xfail.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    leg = sub.add_parser("run-leg", help="run one pytest leg with a JUnit file")
    leg.add_argument("--manifest", required=True)
    leg.add_argument("--name", required=True)
    leg.add_argument("--role", choices=("primary", "retry"), required=True)
    leg.add_argument("--junit", required=True, help="a fresh, unique JUnit path")
    leg.add_argument(
        "--select",
        action="append",
        default=[],
        help="an argument of the command that selects tests (paths, --ignore=...); repeat",
    )
    leg.add_argument(
        "--ids-from",
        help="a file of test ids, one per line, appended to the command (a retry leg)",
    )
    leg.add_argument("--root", default=".")
    leg.add_argument("command", nargs=argparse.REMAINDER)
    leg.set_defaults(func=cmd_run_leg)

    col = sub.add_parser("collect", help="inventory one arm")
    col.add_argument("--out", required=True)
    col.add_argument("--root", default=".")
    col.add_argument(
        "--leg",
        action="append",
        default=[],
        help="selection arguments of one primary leg (a shell-quoted string); repeat",
    )
    col.add_argument("--run-manifest")
    col.add_argument("--junit", action="append", default=[])
    col.add_argument("--collection-from", help="reuse this inventory's collection")
    col.add_argument("--pytest-arg", action="append", default=[])
    col.add_argument("--tests-root", action="append", default=[])
    col.add_argument("--tree-label")
    col.add_argument("--expect-tool-sha256")
    col.set_defaults(func=cmd_collect)

    skips = sub.add_parser(
        "skips", help="id<TAB>kind<TAB>reason of every skip and xfail"
    )
    skips.add_argument("--inventory", required=True)
    skips.add_argument("--out", required=True)
    skips.set_defaults(func=cmd_skips)

    ids = sub.add_parser("ids", help="sorted selected or collected ids")
    ids.add_argument("--inventory", required=True)
    ids.add_argument("--which", choices=("selected", "all"), default="all")
    ids.add_argument("--out", required=True)
    ids.set_defaults(func=cmd_ids)

    failed = sub.add_parser("failed", help="ids that failed in the primary legs")
    failed.add_argument("--inventory", required=True)
    failed.set_defaults(func=cmd_failed)

    cmp_ = sub.add_parser("compare", help="compare a base and a tip inventory")
    cmp_.add_argument("--mode", choices=("collection", "runtime"), required=True)
    cmp_.add_argument("--base", required=True)
    cmp_.add_argument("--tip", required=True)
    cmp_.add_argument("--allow")
    cmp_.add_argument("--decisions")
    cmp_.add_argument("--strict-allow", action="store_true")
    cmp_.add_argument("--only", action="append", default=[])
    cmp_.add_argument("--out", help="write OUT.md and OUT.json")
    cmp_.add_argument("--expect-tool-sha256")
    cmp_.set_defaults(func=cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except InventoryError as exc:
        print(f"suite_inventory: invalid evidence: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
