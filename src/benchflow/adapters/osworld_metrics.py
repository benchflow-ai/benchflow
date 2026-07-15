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

import importlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast


def check_include_exclude(
    result: str | None, rules: Mapping[str, Sequence[str]], **_options: Any
) -> float:
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


def exact_match(result: Any, rules: Mapping[str, Any], **_options: Any) -> float:
    """1.0 iff ``result == rules['expected']``.

    Faithful to OSWorld ``metrics/general.py::exact_match``.
    """
    return 1.0 if result == rules["expected"] else 0.0


def is_utc_0(
    timedatectl_output: str | None, rules: Any = None, **_options: Any
) -> float:
    """1.0 iff ``timedatectl`` reports the Universal-time line ends in ``+0000)``.

    Faithful to OSWorld ``metrics/basic_os.py::is_utc_0`` (checks line index 3 of
    the ``timedatectl status`` output). ``rules`` is unused (this metric takes no
    expected value) but accepted for a uniform ``metric(result, expected)`` call.
    """
    if not timedatectl_output:
        return 0.0
    lines = timedatectl_output.split("\n")
    if len(lines) <= 3:
        return 0.0
    return 1.0 if lines[3].endswith("+0000)") else 0.0


def check_list(result: str | None, rules: Mapping[str, Any], **_options: Any) -> float:
    """1.0 iff every ``expect`` regex matches some line and no ``unexpect`` regex matches.

    Faithful to OSWorld ``metrics/general.py::check_list`` (``result`` is a path to a
    text file; ``rules`` carries ``expect``/``unexpect`` lists of regex patterns).
    """
    if result is None:
        return 0.0
    expect_patterns = [re.compile(ptt) for ptt in rules.get("expect", [])]
    unexpect_patterns = [re.compile(ptt) for ptt in rules.get("unexpect", [])]
    expect_metrics = [False] * len(expect_patterns)
    unexpect_metric = True
    with open(result) as f:
        for line in f:
            for i, r in enumerate(expect_patterns):
                expect_metrics[i] = expect_metrics[i] or (r.search(line) is not None)
            unexpect_metric = unexpect_metric and all(
                r.search(line) is None for r in unexpect_patterns
            )
    return float(all(expect_metrics) and unexpect_metric)


def check_direct_json_object(
    result: Any, rules: Mapping[str, Any], **_options: Any
) -> float:
    """Compare a JSON object against ``rules['expected']`` (key-by-key).

    Faithful to OSWorld ``metrics/general.py::check_direct_json_object``. Supports
    ``expect_in_result`` (substring/membership match) and ``ignore_list_order``.
    """
    if isinstance(result, str):
        result = result.strip().replace("'", '"')
        try:
            result = json.loads(result)
        except Exception:
            return 0.0
    if result is None:
        return 0.0
    expected_json = rules.get("expected", {})
    if expected_json:
        for value in expected_json.values():
            if value == "__EVALUATION_FAILED__":
                return 0.0
    try:
        if not rules.get("expect_in_result", False):
            expected_json = rules["expected"]
            for key in expected_json:
                expected_value = expected_json.get(key)
                actual_value = result.get(key)
                if rules.get("ignore_list_order", False):
                    if sorted(expected_value) != sorted(result.get(key)):
                        return 0.0
                elif expected_value != actual_value:
                    return 0.0
            return 1.0
        expected_json = rules["expected"]
        for key in expected_json:
            if isinstance(expected_json.get(key), list):
                flag = 0
                for each_expected_value in expected_json.get(key):
                    actual = result.get(key)
                    if isinstance(actual, list) and each_expected_value in actual:
                        flag = 1
                        break
                    if isinstance(actual, str) and each_expected_value == actual:
                        flag = 1
                        break
                if flag == 0:
                    return 0.0
            elif isinstance(expected_json.get(key), str):
                if expected_json.get(key) not in result.get(key):
                    return 0.0
            else:
                return 0.0
        return 1.0
    except Exception:
        return 0.0


def compare_text_file(actual: str | None, expected: str, **options: Any) -> float:
    """1.0 iff two text files are equal (optionally ignoring blanks/case).

    Faithful to OSWorld ``metrics/vscode.py::compare_text_file``. ``actual`` and
    ``expected`` are file paths; ``options`` carries ``ignore_blanks``/``ignore_case``.
    """
    if not actual:
        return 0.0
    with open(actual) as f1:
        actual_text = f1.read()
    with open(expected) as f2:
        expected_text = f2.read()
    if options.get("ignore_blanks", False):
        actual_text = re.sub(r"\s+", " ", re.sub(r"[\t\n]", " ", actual_text).strip())
        expected_text = re.sub(
            r"\s+", " ", re.sub(r"[\t\n]", " ", expected_text).strip()
        )
    if options.get("ignore_case", False):
        actual_text = actual_text.lower()
        expected_text = expected_text.lower()
    return 1.0 if actual_text == expected_text else 0.0


# OSWorld metric funcs are referenced by name in a task's ``evaluator.func``. This
# registry resolves the name → callable, mirroring upstream's ``getattr(metrics, func)``.
# Extend as real adopted tasks require more funcs (faithfully ported from upstream).
METRICS: dict[str, Callable[..., float]] = {
    "check_include_exclude": check_include_exclude,
    "exact_match": exact_match,
    "is_utc_0": is_utc_0,
    "check_list": check_list,
    "check_direct_json_object": check_direct_json_object,
    "compare_text_file": compare_text_file,
}


VendoredMetricResolver = Callable[[str], Callable[..., Any] | None]


def _load_vendored_metric_resolver() -> VendoredMetricResolver | None:
    """Load the vendored metric resolver in-repo or from sibling verifier files."""
    for module_name in ("benchflow.adapters.osworld_vendor", "osworld_vendor"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        return cast(VendoredMetricResolver, module.__dict__["resolve_vendored_metric"])
    return None


# The other ~116 OSWorld metrics are not re-implemented: they are resolved from
# the vendored upstream suite (``_osworld_vendor/``), so scoring is OSWorld's own
# code by construction.
_RESOLVE_VENDORED_METRIC = _load_vendored_metric_resolver()


def resolve_metric(func: str) -> Callable[..., float]:
    """Resolve an OSWorld metric name to its callable.

    The few hand-ports (proven reward-identical to upstream) win first — they are
    light and dependency-free. Everything else resolves from the vendored upstream
    suite, so BenchFlow scores with OSWorld's own code (exact parity).
    """
    if func in METRICS:
        return METRICS[func]
    if _RESOLVE_VENDORED_METRIC is not None:
        fn = _RESOLVE_VENDORED_METRIC(func)  # raises if deps missing; None if unknown
        if fn is not None:
            return fn
    raise NotImplementedError(
        f"OSWorld metric {func!r} is not available; it is neither a local port nor in "
        "the vendored OSWorld suite (benchflow.adapters._osworld_vendor)."
    )
