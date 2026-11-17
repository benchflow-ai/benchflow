"""The pristine re-judgment of a plugin guard stop ((g2) R).

Nothing in the trial's sandbox can take a trial out of the score. test.sh's
user can write every marker, record and file there, and in a SWE-bench run
that user is root and has already run the agent's setup.py. So when the guard
stops a trial, the host re-runs the guard's judgment in a fresh, network-less
container from the image the trial started from:

* The guard rejects or fails on the pristine image too: the image's doing or
  BenchFlow's, so the trial is unscored, with the reason.
* It trusts everything there: the trial's own state stopped pytest, so the
  trial scores 0.
* The trial's pytest left no whole end-of-session record (``no_record``:
  none at all though test.sh starts a pytest the guard reaches, one that
  ended before its cleanup, or one that disagrees with pytest's exit status),
  and the guard gives no verdict on the pristine image either: its own
  failure there, so the trial is unscored (the lead, 2026-11-16
  13:57Z and 15:41Z).

Every input to that decision comes from the host, never from the trial: the
image, by ID; the guard's source as hardening generated it, with the
host's own copy of its judge; the host's copy
of the image manifest; the Pythons hardening gave a guard copy; and the
plugins the task names: ``-p`` in its ``[verifier] env`` PYTEST_ADDOPTS and,
where the verifier reads them, in the pytest commands of its test.sh, and
``pytest_plugins`` in its own tests folder (the host's copy). The guard judges what a pytest on that image
would load, knowable without the trial (its ``--judge`` mode with no paths).
Of what it reports, only what the trial's own pytest reads counts (R334 F1
and its delta): with autoload off, a pytest11 entry point loads only by a
name something asks for, the armed guard's required and optional plugins, a
``-p`` of the task, of test.sh or of the one ini strict gives pytest with
``-c``, or the ``pytest_plugins`` of a conftest.py pytest collects for
test.sh's paths, and not one ``-p no:`` disables. The image's other ini
files are never read under ``-c``, and a conftest pytest never collects
never loads: what they name counts for nothing, either way. An entry point the
verifier never loads cannot end a trial either way. The trial's
record names files, but it is the agent's to write: a record that named an
image file the guard rejects would turn the agent's own 0 into unscored. So
it never chooses what is judged.

Only a sandbox that can start such a container (``pristine_image`` and
``run_pristine``: Docker) re-judges. Elsewhere, and whenever the
re-judgment itself cannot run, the record's 0 stands.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import posixpath
import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow.sandbox._image_manifest import kept as kept_image_manifest
from benchflow.sandbox.lockdown import pytest_plugin_guard_judge_source

#: How long one pristine judgment may take.
JUDGE_TIMEOUT_SEC = 180
#: At most this many Pythons judge a stop; at most this many conftest.py
#: files of the task's tests folder are read for the plugins they name.
_MAX_PYTHONS = 4
_MAX_CONFTESTS = 64
_KINDS = ("trusted", "rejected", "guard_error", "absent")
_WHY_CHARS = 300

# ``python -c``, in the pristine image's own Python (3.6 at the oldest): the
# guard's source and the request arrive on stdin; the guard runs as a script
# with ``--judge``, the request as its stdin, and prints its verdicts.
BOOTSTRAP = (
    "import io, json, os, runpy, sys, tempfile\n"
    "d = json.load(sys.stdin)\n"
    "t = tempfile.mkdtemp()\n"
    "p = os.path.join(t, d['name'] + '.py')\n"
    "with open(p, 'w') as h:\n"
    "    h.write(d['guard'])\n"
    "sys.argv = [p, '--judge']\n"
    "sys.stdin = io.StringIO(json.dumps(d['request']))\n"
    "runpy.run_path(p, run_name='__main__')\n"
)


@dataclass(frozen=True)
class Nomination:
    """What the pristine guard judges, beside its discover() set: modules the
    host names, in the host's Pythons. ``paths`` stays empty in a decision."""

    paths: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    pythons: tuple[str, ...] = ()
    #: Plugins the task or test.sh turns off (``-p no:NAME``).
    disabled: tuple[str, ...] = ()
    #: The stop is a run with no whole end-of-session record of the guard.
    no_record: bool = False
    #: The paths the trial's pytest collects from (test.sh's, absolute), and
    #: its ``--confcutdir``: which image conftest.py files it loads.
    anchors: tuple[str, ...] = ()
    confcutdir: str = ""


@dataclass(frozen=True)
class Judged:
    what: str
    kind: str
    why: str = ""
    python: str = ""


@dataclass(frozen=True)
class Rejudgment:
    outcome: Literal["unscored", "zero"]
    reason: str
    judged: tuple[Judged, ...] = field(default=(), compare=False)


# (image id, guard source and manifest digest, Python, paths, modules) -> what
# it judged. The digest leaves out the guard's name, the one thing that
# differs from trial to trial, so a task's trials share one judgment per image
# and Python.
_CACHE: dict[
    tuple[str, str, str, tuple[str, ...], tuple[str, ...]], tuple[Judged, ...]
] = {}


def nominate(
    tests_dir: Any,
    pythons: Iterable[str],
    addopts: Any = "",
    guard: str = "",
    script_plugins: Iterable[str] = (),
    script_disabled: Iterable[str] = (),
    anchors: Iterable[str] = (),
    confcutdir: str = "",
) -> Nomination:
    """What the host puts to the pristine guard: the plugins the task names
    (``-p`` in its *addopts*, the guard's own aside, in its test.sh's pytest
    commands, *script_plugins*, and its own tests folder, the host's copy,
    *tests_dir*), in *pythons* (those hardening gave a guard copy). Nothing
    of the trial's."""
    disabled = (*addopts_disabled(addopts), *script_disabled)
    return Nomination(
        (),
        tuple(
            dict.fromkeys(
                name
                for name in (
                    *addopts_plugins(addopts, guard),
                    *(p for p in script_plugins if isinstance(p, str)),
                    *tests_plugins(tests_dir),
                )
                if name and name != guard and name not in disabled
            )
        ),
        tuple(
            p
            for p in dict.fromkeys(pythons)
            if isinstance(p, str) and p.startswith("/")
        )[:_MAX_PYTHONS],
        disabled=tuple(dict.fromkeys(d for d in disabled if isinstance(d, str) and d)),
        anchors=tuple(
            dict.fromkeys(
                a for a in anchors if isinstance(a, str) and a.startswith("/")
            )
        ),
        confcutdir=confcutdir if isinstance(confcutdir, str) else "",
    )


def addopts_disabled(addopts: Any) -> tuple[str, ...]:
    """The plugins ``-p no:NAME`` turns off in *addopts*."""
    if not isinstance(addopts, str):
        return ()
    try:
        words = shlex.split(addopts)
    except ValueError:
        return ()
    found = []
    for i, word in enumerate(words):
        name = words[i + 1] if word == "-p" and i + 1 < len(words) else None
        if word.startswith("-p") and len(word) > 2:
            name = word[2:]
        if name and name.startswith("no:"):
            found.append(name[3:].strip())
    return tuple(found)


_ARMED_LIST = re.compile(
    r"^(_BENCHFLOW_REQUESTED|_BENCHFLOW_OPTIONAL) = (\[.*\])$", re.MULTILINE
)


def armed_loads(source: str) -> frozenset[str]:
    """The plugins the armed guard *source* (the host's copy) loads by name:
    the ``-p`` plugins every pytest must load and the optional ones it loads
    where installed. Its last assignment of each is the one hardening wrote."""
    lists: dict[str, list[str]] = {}
    for found in _ARMED_LIST.finditer(source):
        try:
            value = ast.literal_eval(found.group(2))
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            lists[found.group(1)] = value
    return frozenset(name for names in lists.values() for name in names)


def addopts_plugins(addopts: Any, guard: str = "") -> tuple[str, ...]:
    """The plugins ``-p NAME`` (or ``-pNAME``) loads in *addopts*: not
    ``no:`` ones, not the guard *guard*."""
    if not isinstance(addopts, str):
        return ()
    try:
        words = shlex.split(addopts)
    except ValueError:
        return ()
    found = []
    for i, word in enumerate(words):
        name = words[i + 1] if word == "-p" and i + 1 < len(words) else None
        if word.startswith("-p") and len(word) > 2:
            name = word[2:]
        if name and not name.startswith("no:") and name != guard:
            found.append(name)
    return tuple(found)


def tests_plugins(tests_dir: Any) -> tuple[str, ...]:
    """The module names a literal ``pytest_plugins`` names in the
    conftest.py files of *tests_dir*, read with ``ast``, never imported."""
    try:
        conftests = sorted(Path(tests_dir).rglob("conftest.py"))[:_MAX_CONFTESTS]
    except (OSError, TypeError):
        return ()
    found: list[str] = []
    for conftest in conftests:
        try:
            tree = ast.parse(conftest.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError, ValueError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not any(
                isinstance(t, ast.Name) and t.id == "pytest_plugins"
                for t in node.targets
            ):
                continue
            value = node.value
            items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            found.extend(
                item.value
                for item in items
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return tuple(dict.fromkeys(found))


def payload(
    name: str, source: str, nomination: Nomination, manifest: str | None
) -> bytes:
    return json.dumps(
        {
            "name": name,
            "guard": source,
            "request": {
                "paths": list(nomination.paths),
                "modules": list(nomination.modules),
                "manifest": manifest,
            },
        }
    ).encode()


def parse_judgment(
    stdout: str, nomination: Nomination, loads: Iterable[str] | None = None
) -> tuple[Judged, ...] | None:
    """The guard's verdicts, or None unless they are a whole answer: a row
    for each module the host named, beside its discover() set. With *loads*
    (the names the trial's armed guard loads, :func:`armed_loads` and the
    nomination's), a pytest11 entry point's rows count only when that pytest
    would load it (:func:`_loaded_rows`)."""
    try:
        data: Any = (
            json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else None
        )
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    judged = []
    for item in data:
        if not isinstance(item, dict) or item.get("kind") not in _KINDS:
            return None
        what = item.get("path") or item.get("module")
        why, python = item.get("why", ""), item.get("python", "")
        if not all(isinstance(v, str) for v in (what, why, python)) or not what:
            return None
        judged.append(Judged(what, item["kind"], why[:_WHY_CHARS], python[:32]))
    asked = {*nomination.paths, *nomination.modules}
    if not asked <= {j.what for j in judged}:
        return None
    if loads is not None:
        kept = _loaded_rows(
            data, {*loads, *nomination.modules}, nomination.disabled, nomination
        )
        judged = [j for j, keep in zip(judged, kept, strict=True) if keep]
    return tuple(judged)


def collected(conftest: str, anchors: Iterable[str], confcutdir: str = "") -> bool:
    """Whether pytest, given the paths *anchors* (absolute), loads the image
    file *conftest*: its folder holds one of them, or is a folder below one
    pytest collects into (not one its default ``norecursedirs`` skips), and is
    not above *confcutdir*."""
    folder = posixpath.dirname(posixpath.normpath(conftest)) or "/"
    if confcutdir:
        cut = posixpath.normpath(confcutdir)
        if folder != cut and (folder == "/" or cut.startswith(folder + "/")):
            return False
    for anchor in anchors:
        anchor = posixpath.normpath(anchor)
        if anchor == folder or folder == "/" or anchor.startswith(folder + "/"):
            return True
        if folder.startswith(anchor + "/") and not any(
            fnmatch.fnmatch(part, skip)
            for part in folder[len(anchor) + 1 :].split("/")
            for skip in _NORECURSE
        ):
            return True
    return False


# pytest's default ``norecursedirs``.
_NORECURSE = (
    "*.egg",
    ".*",
    "_darcs",
    "build",
    "CVS",
    "dist",
    "node_modules",
    "venv",
    "{arch}",
)


def _loaded_rows(
    rows: list[dict[str, Any]],
    loads: set[str],
    disabled: Iterable[str],
    nomination: Nomination | None = None,
) -> list[bool]:
    """Which of the judge's *rows* judge what the trial's pytest reads. A
    conftest's rows count when pytest collects it for the nomination's
    paths (:func:`collected`); an image ini's never do, as strict gives pytest
    only its ``-c`` copy, whose ``-p`` names are the nomination's. A row of an
    entry point (``from``: ``entry_point:<name>``) counts when its name or its
    module is asked for by name, here or by a conftest that counts, and no
    ``-p no:`` turns it off; the rows of a table of entry points that cannot
    be read count when anything is asked for. Every other row counts. With
    no *nomination*, every conftest and ini row counts (the judge's own
    tests)."""
    off = set(disabled)
    names = set(loads)

    def read(origin: Any) -> bool:
        if not isinstance(origin, str):
            return True
        if origin.startswith("ini:"):
            return nomination is None
        if origin.startswith("conftest:"):
            return nomination is None or collected(
                origin[len("conftest:") :], nomination.anchors, nomination.confcutdir
            )
        return True

    for row in rows:
        origin = row.get("from")
        if (
            isinstance(origin, str)
            and origin.startswith(("conftest:", "ini:"))
            and read(origin)
        ):
            module = row.get("module")
            if isinstance(module, str):
                names.add(module)
    names -= off
    modules: dict[str, str] = {}
    for row in rows:
        origin, module = row.get("from"), row.get("module")
        if isinstance(origin, str) and isinstance(module, str):
            modules.setdefault(origin, module)
    kept = []
    for row in rows:
        origin = row.get("from")
        if not isinstance(origin, str) or not origin.startswith("entry_point"):
            kept.append(read(origin))
        elif origin == "entry_point":
            kept.append(bool(names))
        else:
            name = origin[len("entry_point:") :]
            kept.append(
                name not in off and (name in names or modules.get(origin) in names)
            )
    return kept


def decide(judged: Iterable[Judged], stop: str) -> Rejudgment:
    """*stop* is the trial's own reason (the verifier's)."""
    judged = tuple(judged)
    for j in judged:
        if j.kind in ("rejected", "guard_error"):
            done = "fails while judging" if j.kind == "guard_error" else "rejects"
            cause = ", ".join(filter(None, (j.why, j.python and f"Python {j.python}")))
            return Rejudgment(
                "unscored",
                f"on the pristine image the pytest plugin guard {done} {j.what} too"
                + (f" ({cause})" if cause else "")
                + ": the image's or BenchFlow's doing, not the agent's, so this "
                f"run is not scored. The trial's own stop: {stop}",
                judged,
            )
    return Rejudgment(
        "zero",
        f"{stop}; on the pristine image the pytest plugin guard trusts what it "
        f"judged ({len(judged)} file{'s' if len(judged) != 1 else ''} or module"
        f"{'s' if len(judged) != 1 else ''}), so the trial's own state stopped "
        "pytest and it scores 0",
        judged,
    )


def _no_verdict(
    nomination: Nomination, result: Any, python: str, stop: str
) -> Rejudgment | None:
    """A pristine judgment that gave no whole answer: for a run with no whole
    end-of-session record, the guard's own failure on the pristine image (its
    Python ran it and it ended with an error, or printed no verdicts), and the
    trial is unscored; otherwise None, and the record's 0 stands. Docker's own
    failures (125-127) and a timeout are never the guard's."""
    code = result.return_code
    if not nomination.no_record or code not in (0, 1):
        return None
    lines = (getattr(result, "stderr", "") or "").strip().splitlines()
    cause = lines[-1].strip()[:_WHY_CHARS] if lines else f"exit {code}"
    return Rejudgment(
        "unscored",
        f"on the pristine image the pytest plugin guard gives no verdict either "
        f"in {python} ({cause}): the image's or BenchFlow's doing, not the "
        f"agent's, so this run is not scored. The trial's own stop: {stop}",
    )


async def rejudge(
    env: Any, guard: str, nomination: Nomination, stop: str
) -> Rejudgment | None:
    """The pristine image's judgment of the guard's stop, *stop* (the
    verifier's reason), or None: this sandbox cannot start a pristine
    container, it kept no guard source or Python, or the judgment gave no
    whole answer."""
    pristine_image = getattr(env, "pristine_image", None)
    run_pristine = getattr(env, "run_pristine", None)
    armed = getattr(env, "_benchflow_guard_source", None)
    if not callable(pristine_image) or not callable(run_pristine):
        return None
    if not isinstance(armed, str) or not armed or not nomination.pythons:
        return None
    # The armed guard, then its judge: the judge never enters the sandbox
    # (hardening writes the armed source in one argument, capped at 128 KiB).
    source = pytest_plugin_guard_judge_source(armed)
    image = await pristine_image()
    if not image:
        return None
    manifest = kept_image_manifest(env)
    manifest_text = manifest.to_json() if manifest is not None else None
    data = payload(guard, source, nomination, manifest_text)
    # The digest leaves out the guard's name; the manifest the judgment read
    # is in it.
    digest = hashlib.sha256(
        source.replace(guard, "\0").encode() + b"\0" + (manifest_text or "").encode()
    ).hexdigest()
    loads = armed_loads(armed)
    found: list[Judged] = []
    for python in nomination.pythons:
        key = (image, digest, python, nomination.paths, nomination.modules)
        judged = _CACHE.get(key)
        if judged is None:
            result = await run_pristine(
                image, python, BOOTSTRAP, data, JUDGE_TIMEOUT_SEC
            )
            judged = (
                parse_judgment(result.stdout or "", nomination, loads)
                if result.return_code == 0
                else None
            )
            if judged is None:
                return _no_verdict(nomination, result, python, stop)
            _CACHE[key] = judged
        found.extend(judged)
    return decide(found, stop)
