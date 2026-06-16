"""Faithful ports of OSWorld evaluator metric functions.

OSWorld scores a task by running a *getter* (e.g. ``vm_command_line`` runs a shell
command in the desktop VM and returns its stdout) and applying a *metric* function
to that result against the task's ``expected`` rules. Upstream lives in
``desktop_env/evaluators/metrics/`` of xlang-ai/OSWorld; these are byte-faithful
ports of the metric functions so BenchFlow scores a real OSWorld task the same way
the official harness does — NOT the degenerate "did the agent print 'observed'"
stub the use_computer_cookbook adapter currently ships.

Only the metric *functions* live here (pure, unit-testable). The getter execution
(run the command in the sandbox), ``postconfig`` setup, and template-variable
substitution are the evaluator-orchestration layer (built next; see
~/benchflow-context/0.7-real-evals-goal.md).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def check_include_exclude(result: str | None, rules: Mapping[str, Sequence[str]]) -> float:
    """1.0 iff ``result`` contains every ``include`` string and no ``exclude`` string.

    Faithful to OSWorld ``metrics/general.py::check_include_exclude``.
    """
    if result is None:
        return 0.0
    include = rules.get("include", [])
    exclude = rules.get("exclude", [])
    if all(token in result for token in include) and all(
        token not in result for token in exclude
    ):
        return 1.0
    return 0.0


def exact_match(result: Any, rules: Mapping[str, Any]) -> float:
    """1.0 iff ``result == rules['expected']``.

    Faithful to OSWorld ``metrics/general.py::exact_match``.
    """
    return 1.0 if result == rules["expected"] else 0.0


# OSWorld metric funcs are referenced by name in a task's ``evaluator.func``. This
# registry resolves the name → callable, mirroring upstream's ``getattr(metrics, func)``.
# Extend as real adopted tasks require more funcs (faithfully ported from upstream).
METRICS: dict[str, Callable[..., float]] = {
    "check_include_exclude": check_include_exclude,
    "exact_match": exact_match,
}


def resolve_metric(func: str) -> Callable[..., float]:
    """Resolve an OSWorld metric name to its callable, or raise for an unported one."""
    try:
        return METRICS[func]
    except KeyError as exc:  # pragma: no cover - guard for unported funcs
        raise NotImplementedError(
            f"OSWorld metric {func!r} is not ported yet; add a faithful port to "
            "benchflow.adapters.osworld_metrics.METRICS"
        ) from exc
