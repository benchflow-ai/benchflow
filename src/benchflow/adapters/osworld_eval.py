"""Real OSWorld evaluator orchestration (faithful to xlang-ai/OSWorld).

OSWorld scores a task by, in order: running the evaluator's ``postconfig`` setup
steps in the desktop, running each ``result`` getter (``vm_command_line`` POSTs the
command to the desktop control server's ``/execute`` and returns its stdout),
resolving each ``expected`` getter (a ``rule`` getter just returns its ``rules``),
and applying the named metric ``func`` to ``(result, expected)`` — combining
multiple funcs with ``conj`` ("and"/"or"). See OSWorld ``desktop_env.py`` +
``evaluators/getters,metrics``.

This module ports that orchestration. The desktop's ``/execute`` is abstracted as an
injected ``run_command(command, shell) -> str`` so the logic is unit-testable without
a live desktop; the Cua-local desktop wiring is a later increment (see
~/benchflow-context/0.7-real-evals-goal.md). It replaces the degenerate
``use_computer_cookbook._expected_result`` "answer must be exactly 'observed'" stub.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:
    # In-repo (tested) import path.
    from benchflow.adapters.osworld_metrics import resolve_metric
except ImportError:  # pragma: no cover - standalone in a sandbox verifier
    # When carried into a task sandbox as a sibling file (benchflow not installed),
    # osworld_metrics.py sits next to this module.
    from osworld_metrics import resolve_metric  # type: ignore[no-redef]

# Runs a command in the desktop sandbox and returns its stdout (the benchflow
# analogue of OSWorld's POST to the desktop server's /execute endpoint).
RunCommand = Callable[[Any, bool], str]

# OSWorld defaults: desktop resolution 1920x1080 and the public-evaluation VM
# password (desktop_env/controllers/setup.py + desktop_env.py screen_size).
_DEFAULT_PASSWORD = "password"
_DEFAULT_SCREEN = (1920, 1080)


def substitute(value: Any, *, password: str, width: int, height: int) -> Any:
    """Substitute OSWorld command template variables, faithful to OSWorld
    ``controllers/setup.py``: ``{CLIENT_PASSWORD}``, ``{SCREEN_WIDTH[_HALF]}``,
    ``{SCREEN_HEIGHT[_HALF]}``. Applies to a string command or a list of parts.
    """
    repls = {
        "{CLIENT_PASSWORD}": password,
        "{SCREEN_WIDTH_HALF}": str(width // 2),
        "{SCREEN_HEIGHT_HALF}": str(height // 2),
        "{SCREEN_WIDTH}": str(width),
        "{SCREEN_HEIGHT}": str(height),
    }

    def _one(text: str) -> str:
        for token, repl in repls.items():
            text = text.replace(token, repl)
        return text

    if isinstance(value, str):
        return _one(value)
    if isinstance(value, list):
        return [_one(part) if isinstance(part, str) else part for part in value]
    return value


class UnsupportedGetterError(NotImplementedError):
    """An OSWorld getter ``type`` that has not been ported yet."""


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def _run_postconfig(
    steps: Sequence[Mapping[str, Any]],
    run_command: RunCommand,
    subst: Callable[[Any], Any],
) -> None:
    """Run evaluator ``postconfig`` setup in the desktop before scoring.

    Mirrors the OSWorld config vocabulary used by evaluator setup: ``execute`` /
    ``command`` run a shell command; ``download`` fetches each file to its path
    (via ``curl`` in the desktop, the benchflow analogue of OSWorld's host download
    + upload). Unknown step types raise so we never silently under-provision.
    """
    for step in steps:
        if not isinstance(step, dict):
            continue
        stype = step.get("type")
        params = step.get("parameters") or {}
        if stype in {"execute", "command"}:
            run_command(subst(params.get("command")), bool(params.get("shell", False)))
        elif stype == "download":
            for entry in params.get("files") or []:
                url = entry.get("url")
                path = entry.get("path")
                if url and path:
                    run_command(
                        f"curl -fsSL {shlex.quote(url)} -o {shlex.quote(path)}", True
                    )
        else:
            raise UnsupportedGetterError(
                f"OSWorld postconfig step type {stype!r} is not ported yet"
            )


def _get_result(
    config: Mapping[str, Any] | None,
    run_command: RunCommand,
    subst: Callable[[Any], Any],
) -> Any:
    """Resolve a ``result`` getter to its value (port of OSWorld getters)."""
    if not config:
        return None
    gtype = config.get("type")
    if gtype == "vm_command_line":
        return run_command(subst(config.get("command")), bool(config.get("shell", False)))
    raise UnsupportedGetterError(f"OSWorld result getter {gtype!r} is not ported yet")


def _get_expected(config: Mapping[str, Any] | None) -> Any:
    """Resolve an ``expected`` getter (the common ``rule`` getter returns its rules)."""
    if not config:
        return None
    gtype = config.get("type")
    if gtype == "rule":
        return config.get("rules")
    raise UnsupportedGetterError(f"OSWorld expected getter {gtype!r} is not ported yet")


def evaluate(
    osworld_task: Mapping[str, Any],
    run_command: RunCommand,
    *,
    password: str = _DEFAULT_PASSWORD,
    screen: tuple[int, int] = _DEFAULT_SCREEN,
) -> float:
    """Score a real OSWorld task: postconfig → result getter → metric vs expected.

    ``run_command(command, shell)`` runs a command in the desktop and returns stdout.
    ``password``/``screen`` resolve the template variables in commands. Returns the
    OSWorld reward (1.0 / 0.0), combining multiple metrics via ``conj``.
    """
    evaluator = osworld_task.get("evaluator") or {}
    if not isinstance(evaluator, dict):
        raise ValueError("OSWorld task 'evaluator' must be an object")

    width, height = screen

    def subst(value: Any) -> Any:
        return substitute(value, password=password, width=width, height=height)

    _run_postconfig(evaluator.get("postconfig") or [], run_command, subst)

    funcs = _as_list(evaluator["func"])
    results = _as_list(evaluator.get("result"))
    expecteds = _as_list(evaluator.get("expected"))
    # Pad result/expected to the number of metrics (OSWorld requires equal lengths
    # for the list form; the scalar form broadcasts to one).
    while len(results) < len(funcs):
        results.append(None)
    while len(expecteds) < len(funcs):
        expecteds.append(None)

    scores: list[float] = []
    for func, result_cfg, expected_cfg in zip(funcs, results, expecteds, strict=False):
        result = _get_result(result_cfg, run_command, subst)
        expected = _get_expected(expected_cfg)
        scores.append(float(resolve_metric(func)(result, expected)))

    conj = evaluator.get("conj", "and")
    if conj == "or":
        return 1.0 if any(s >= 1.0 for s in scores) else 0.0
    return 1.0 if scores and all(s >= 1.0 for s in scores) else 0.0
