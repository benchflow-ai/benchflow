"""The pytest plugins a task's own verifier scripts install.

The plugin guard never loads code test.sh installs after the agent's turn: a
plugin installed there runs only as a pinned release BenchFlow serves from its
protected verifier runtime (``[verifier] pytest_plugin_releases``). A test.sh
that installs one otherwise runs its pytest without it, and an option the
plugin owns stops pytest with a usage error that test.sh scores 0 (a wrong 0).
So it is decided before the run, from the task's own files, never from the
guard's markers (code test.sh runs could write those):

* a pinned install (``name==version``) is served from the runtime, as if the
  task had pinned it;
* an unpinned one refuses the task before any sandbox, unscored, with its
  reason, unless the plugin is in :data:`INERT_WHEN_UNUSED` and nothing in the
  task's files uses it: then it is never loaded and the trial says so
  (:data:`INERT_WARNING`).

pytest-json-ctrf is BenchFlow's to serve whenever test.sh asks for ``--ctrf``
and needs neither. Every pinned release, declared or test.sh's, then has
its dependency closure checked before any sandbox (:func:`closure_refusal`):
one that needs a release nobody pins is refused there, in the dry run as
in the run.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CTRF = "pytest-json-ctrf"
# Unpinned installs left unloaded instead of refused, each only while nothing
# in the task's files uses it. Adding a plugin here needs a proof that it is
# inert when unused (what in a task's files would reach it) and the
# lead's OK (2026-11-16).
INERT_WHEN_UNUSED = ("pytest-timeout",)
INERT_WARNING = "pytest-timeout installed unpinned, not loaded (inert)"
_EXACT_PIN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?==[0-9][0-9A-Za-z.!+_-]*"
)
_NAME = re.compile(r"[\[<>=!~;@ ]")
# What uses pytest-timeout, in the task's own files.
_TIMEOUT_FLAG = re.compile(r"--timeout\b")
_TIMEOUT_INI_KEY = re.compile(
    r"^\s*timeout(?:_method|_func_only|_disable_debugger_detection)?\s*=", re.M
)
_TIMEOUT_ENV = re.compile(r"PYTEST_TIMEOUT")
_TIMEOUT_MARK = re.compile(r"pytest\.mark\.timeout\b")
_STRICT_MARKERS = re.compile(
    r"--strict-markers\b|--strict\b|^\s*strict(?:_markers)?\s*=\s*(?:true|1)\b",
    re.M | re.I,
)
_TEXT_SUFFIXES = (".sh", ".py", ".ini", ".cfg", ".toml", ".txt")
_INI_SUFFIXES = (".ini", ".cfg", ".toml")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _name(spec: str) -> str:
    return _canonical(_NAME.split(spec.strip(), maxsplit=1)[0])


def verifier_folders(task: Any) -> list[Path]:
    """The task's verifier folders (each step's, on a multi-step task)."""
    # A line without multi-step tasks (0.9) has no benchflow.task.steps.
    steps: Any = None
    with contextlib.suppress(ImportError):
        steps = importlib.import_module("benchflow.task.steps")
    task_dir = getattr(task, "task_dir", None)
    if not task_dir:
        return []
    if steps is not None and steps.has_steps(task):
        return list(
            dict.fromkeys(
                steps.step_paths(task, i).tests_dir
                for i in range(len(task.config.steps))
            )
        )
    native = getattr(getattr(task, "paths", None), "uses_native_verifier_dir", False)
    return [Path(task_dir) / ("verifier" if native is True else "tests")]


def _declared(task: Any) -> set[str]:
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    return {
        _name(pin) for pin in (getattr(verifier, "pytest_plugin_releases", None) or [])
    }


def installs(task: Any) -> list[tuple[str, str]]:
    """``(script, spec)`` for every pytest plugin a verifier script installs."""
    from benchflow._utils.task_authoring.structural_checks import (
        workspace_plugin_installs,
    )

    found: list[tuple[str, str]] = []
    for folder in verifier_folders(task):
        if not folder.is_dir():
            continue
        for script in sorted(folder.glob("*.sh")):
            try:
                text = script.read_text(errors="replace")
            except OSError:
                continue
            _, plugins = workspace_plugin_installs(text)
            found += [(f"{folder.name}/{script.name}", spec) for spec in plugins]
    return list(dict.fromkeys(found))


def implied_pins(task: Any) -> list[str]:
    """The pinned releases test.sh installs that the task does not declare."""
    declared = _declared(task)
    pins: dict[str, str] = {}
    for _, spec in installs(task):
        name = _name(spec)
        if name != CTRF and name not in declared and _EXACT_PIN.fullmatch(spec):
            pins.setdefault(name, spec)
    return list(pins.values())


def _timeout_used(task: Any) -> str | None:
    """What in the task's own files uses pytest-timeout, or None."""
    mark = strict = None
    for folder in verifier_folders(task):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix not in _TEXT_SUFFIXES
            ):
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            where = f"{folder.name}/{path.relative_to(folder)}"
            if _TIMEOUT_FLAG.search(text):
                return f"{where} passes --timeout"
            if _TIMEOUT_ENV.search(text):
                return f"{where} sets PYTEST_TIMEOUT"
            if path.suffix in _INI_SUFFIXES and _TIMEOUT_INI_KEY.search(text):
                return f"{where} sets a timeout ini key"
            if _TIMEOUT_MARK.search(text):
                mark = mark or where
            if _STRICT_MARKERS.search(text):
                strict = strict or where
    if mark and strict:
        return (
            f"{mark} marks tests @pytest.mark.timeout under strict markers ({strict})"
        )
    return None


def assess(task: Any) -> tuple[list[str], list[str]]:
    """The unpinned plugin installs that refuse the task (each with why), and
    the inert ones left unloaded."""
    declared = _declared(task)
    refused: list[str] = []
    inert: list[str] = []
    for where, spec in installs(task):
        name = _name(spec)
        if name == CTRF or name in declared or _EXACT_PIN.fullmatch(spec):
            continue
        if name in INERT_WHEN_UNUSED:
            use = _timeout_used(task)
            if use is None:
                inert.append(f"{where}: {spec}")
                continue
            refused.append(f"{where} installs {spec} unpinned, and {use}")
        else:
            refused.append(f"{where} installs {spec} unpinned")
    return list(dict.fromkeys(refused)), list(dict.fromkeys(inert))


def refusal(task: Any) -> str | None:
    """Why the task is refused before any sandbox, or None."""
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    if getattr(verifier, "service", "main") != "main":
        return None  # no plugin guard runs in another service (#248)
    refused, _ = assess(task)
    if not refused:
        return None
    return (
        "; ".join(refused)
        + ". The pytest plugin guard never loads a plugin test.sh installs, so "
        "its pytest would run without it and score the solution 0. Pin each in "
        "[verifier] pytest_plugin_releases (name==version), which BenchFlow "
        "serves from its protected verifier runtime, or install it into the image"
    )


def closure_refusal(task: Any) -> str | None:
    """Why the task's plugin releases (declared, and the pinned ones test.sh
    installs) cannot make the protected runtime, decided before any sandbox
    from their pinned PyPI metadata, or None. When neither PyPI nor the
    host's cache can say, the sandbox's own preparation decides, as an
    infrastructure failure there."""
    from benchflow.sandbox import _verifier_runtime as runtime

    pins, ctrf = runtime.runtime_requested(task)
    if not pins:
        return None  # BenchFlow's own pytest-json-ctrf pin alone
    try:
        return runtime.plan_closure_problem(
            pins,
            ctrf=ctrf,
            runner=runtime.runner_requested(task),
            wheelhouses=runtime._task_wheelhouses(task),
        )
    except runtime.RuntimeUnavailable as exc:
        logger.warning("pytest plugin releases not checked before the run: %s", exc)
        return None


def inert_installs(task: Any) -> list[str]:
    """The unpinned installs left unloaded because nothing in the task uses them."""
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    if getattr(verifier, "service", "main") != "main":
        return []
    return assess(task)[1]
