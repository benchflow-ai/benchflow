"""Real OSWorld evaluator orchestration (faithful to xlang-ai/OSWorld).

OSWorld scores a task by, in order: running the evaluator's ``postconfig`` setup
steps in the desktop, resolving each ``result`` getter, resolving each ``expected``
getter, and applying the named metric ``func`` to ``(result, expected, **options)``
— combining multiple funcs with ``conj`` ("and"/"or"). See OSWorld ``desktop_env.py``
+ ``evaluators/getters,metrics``.

This module ports that orchestration. The desktop's ``/execute`` is abstracted as an
injected ``run_command(command, shell) -> str``; file getters (``vm_file``,
``cloud_file``, ``cache_file`` …) resolve to local paths, which is correct when the
verifier runs *in the guest* (files are local) — the benchflow OSWorld verifier runs
in-sandbox. ``cache_dir`` is where ``cloud_file`` reference downloads land.
"""

from __future__ import annotations

import importlib
import os
import shlex
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

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
VendoredGetter = Callable[..., Any]
VendoredGetterResolver = Callable[[str], VendoredGetter | None]
ShimEnvFactory = Callable[[RunCommand, str], Any]


def _load_vendored_getters() -> tuple[ShimEnvFactory, VendoredGetterResolver] | None:
    """Load vendored getter shims in-repo or from sibling verifier files."""
    for module_name in ("benchflow.adapters.osworld_getters", "osworld_getters"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        return (
            cast(ShimEnvFactory, module.__dict__["ShimEnv"]),
            cast(VendoredGetterResolver, module.__dict__["resolve_vendored_getter"]),
        )
    return None


# Vendored OSWorld getters (chrome/vlc/gimp/accessibility/...) for getter types
# the native handling below does not cover. They run through an in-guest
# controller shim so scoring uses OSWorld's own getter code.
_VENDORED_GETTERS = _load_vendored_getters()

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
    ``command`` run a shell command; ``download`` fetches each file to its path;
    ``sleep`` waits. Unknown step types raise so we never silently under-provision.
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
        elif stype == "sleep":
            run_command(f"sleep {float(params.get('seconds', 1))}", True)
        else:
            raise UnsupportedGetterError(
                f"OSWorld postconfig step type {stype!r} is not ported yet"
            )


def _download(url: str, dest: str) -> str | None:
    """Download ``url`` to ``dest`` (atomic), returning the path or ``None`` on failure."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = f"{dest}.tmp"
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        os.replace(tmp, dest)
        return dest
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


def _get_state(
    config: Mapping[str, Any] | None,
    run_command: RunCommand,
    subst: Callable[[Any], Any],
    cache_dir: str,
) -> Any:
    """Resolve a getter config to its value (port of OSWorld getters).

    File getters resolve to *local paths* — correct for the in-guest verifier, where
    the result file is already on disk and the expected reference is downloaded.
    """
    if not config:
        return None
    gtype = config.get("type")
    if gtype == "vm_command_line":
        return run_command(
            subst(config.get("command")), bool(config.get("shell", False))
        )
    if gtype == "rule":
        return config.get("rules")
    if gtype in {"vm_file", "local"}:
        path = subst(config.get("path"))
        return path if path and os.path.exists(path) else None
    if gtype == "cloud_file":
        dest = os.path.join(
            cache_dir,
            str(config.get("dest") or os.path.basename(str(config.get("path")))),
        )
        return _download(str(config.get("path")), dest)
    if gtype == "cache_file":
        return os.path.join(cache_dir, str(config.get("path")))
    if gtype == "content_from_vm_file":
        return _content_from_vm_file(subst(config.get("path")), config)
    # Anything else: run OSWorld's own getter via the in-guest controller shim.
    if _VENDORED_GETTERS is not None:
        shim_env, resolve_vendored_getter = _VENDORED_GETTERS
        getter = resolve_vendored_getter(str(gtype))  # raises if deps missing
        if getter is not None:
            return getter(shim_env(run_command, cache_dir), dict(config))
    raise UnsupportedGetterError(f"OSWorld getter {gtype!r} is not ported yet")


def _content_from_vm_file(path: str | None, config: Mapping[str, Any]) -> Any:
    """Port of OSWorld ``get_content_from_vm_file`` (xlsx ``last_row`` only so far)."""
    if not path or not os.path.exists(path):
        return None
    if config.get("file_type") == "xlsx" and config.get("file_content") == "last_row":
        import pandas as pd  # lazy: heavy dep only needed for this getter

        return pd.read_excel(path).iloc[-1].astype(str).tolist()
    raise UnsupportedGetterError(
        f"content_from_vm_file {config.get('file_type')}/{config.get('file_content')} not ported"
    )


def evaluate(
    osworld_task: Mapping[str, Any],
    run_command: RunCommand,
    *,
    password: str = _DEFAULT_PASSWORD,
    screen: tuple[int, int] = _DEFAULT_SCREEN,
    cache_dir: str | None = None,
) -> float:
    """Score a real OSWorld task: postconfig → result/expected getters → metric.

    ``run_command(command, shell)`` runs a command in the desktop and returns stdout.
    ``password``/``screen`` resolve template variables. ``cache_dir`` holds
    ``cloud_file`` reference downloads. Returns the OSWorld reward (1.0 / 0.0),
    combining multiple metrics via ``conj``.
    """
    evaluator = osworld_task.get("evaluator") or {}
    if not isinstance(evaluator, dict):
        raise ValueError("OSWorld task 'evaluator' must be an object")

    width, height = screen
    cache = cache_dir or tempfile.mkdtemp(prefix="osworld-cache-")

    def subst(value: Any) -> Any:
        return substitute(value, password=password, width=width, height=height)

    _run_postconfig(evaluator.get("postconfig") or [], run_command, subst)

    funcs = _as_list(evaluator["func"])
    results = _as_list(evaluator.get("result"))
    has_expected = bool(evaluator.get("expected"))
    expecteds = _as_list(evaluator.get("expected"))
    options = _as_list(evaluator.get("options"))
    # Pad result/expected/options to the number of metrics.
    for seq in (results, expecteds, options):
        while len(seq) < len(funcs):
            seq.append(None)

    scores: list[float] = []
    for func, result_cfg, expected_cfg, opt in zip(
        funcs, results, expecteds, options, strict=False
    ):
        result = _get_state(result_cfg, run_command, subst, cache)
        metric = resolve_metric(func)
        kwargs = opt if isinstance(opt, dict) else {}
        if has_expected and expected_cfg is not None:
            expected = _get_state(expected_cfg, run_command, subst, cache)
            scores.append(float(metric(result, expected, **kwargs)))
        else:
            scores.append(float(metric(result, **kwargs)))

    conj = evaluator.get("conj", "and")
    if conj == "or":
        return 1.0 if any(s >= 1.0 for s in scores) else 0.0
    return 1.0 if scores and all(s >= 1.0 for s in scores) else 0.0
