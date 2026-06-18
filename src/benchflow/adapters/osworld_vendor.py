"""Bridge to the vendored OSWorld evaluator suite (``_osworld_vendor/``).

To "finish the whole benchmark" with exact parity, BenchFlow runs OSWorld's *own*
metric functions rather than re-implementing 122 of them. They are vendored
verbatim from xlang-ai/OSWorld (Apache-2.0) under ``_osworld_vendor/``; this module
resolves a metric name to the upstream callable, importing only the one module that
metric lives in (lazy) so a task needing ``compare_table`` (openpyxl/pandas) never
drags in the torch-class deps of ``docs``/``gimp``/``vlc``.

Resolution order in :func:`benchflow.adapters.osworld_metrics.resolve_metric` is
local-port-first (the few hand-ports are proven reward-identical to upstream and
need no deps), then this vendored registry for everything else.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from collections.abc import Callable
from typing import Any

_VENDOR_ROOT = os.path.join(os.path.dirname(__file__), "_osworld_vendor")
_PKG = "desktop_env.evaluators.metrics"


def _ensure_path() -> None:
    if _VENDOR_ROOT not in sys.path:
        sys.path.insert(0, _VENDOR_ROOT)


def _build_func_module_map() -> dict[str, str]:
    """Map ``metric_func -> module`` by scanning each vendored metric module's
    top-level ``def``s (the package ``__init__`` is intentionally minimal, so we do
    not parse it). Private helpers (leading ``_``) are skipped; ``utils`` is excluded
    (it holds shared helpers, not metric funcs)."""
    metrics_dir = os.path.join(_VENDOR_ROOT, "desktop_env", "evaluators", "metrics")
    mapping: dict[str, str] = {}
    try:
        files = sorted(os.listdir(metrics_dir))
    except OSError:
        return {}
    for fname in files:
        if not fname.endswith(".py") or fname in {"__init__.py", "utils.py"}:
            continue
        module = fname[:-3]
        try:
            with open(os.path.join(metrics_dir, fname), encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        for name in re.findall(r"^def ([a-zA-Z]\w*)\s*\(", src, re.M):
            mapping.setdefault(name, module)
    return mapping


FUNC_MODULE: dict[str, str] = _build_func_module_map()


def vendored_metric_names() -> frozenset[str]:
    """All metric func names the vendored suite exposes (regardless of dep install)."""
    return frozenset(FUNC_MODULE)


class VendoredMetricUnavailable(RuntimeError):
    """The metric's module could not be imported (its deps are not installed)."""


def resolve_vendored_metric(func: str) -> Callable[..., Any] | None:
    """Return the upstream callable for ``func``, or ``None`` if it is not a known
    vendored metric. Raises :class:`VendoredMetricUnavailable` if the metric exists
    but its module's deps are missing (so the caller can surface an actionable error
    rather than silently scoring 0)."""
    module = FUNC_MODULE.get(func)
    if module is None:
        return None
    _ensure_path()
    try:
        mod = importlib.import_module(f"{_PKG}.{module}")
    except Exception as exc:  # missing heavy dep (torch/cv2/librosa/…)
        raise VendoredMetricUnavailable(
            f"OSWorld metric {func!r} lives in vendored module {module!r}, whose "
            f"dependencies are not installed ({type(exc).__name__}: {exc}). Install "
            "the 'osworld' extra (and 'osworld-cv' for cv2/easyocr/librosa metrics)."
        ) from exc
    fn = getattr(mod, func, None)
    return fn
