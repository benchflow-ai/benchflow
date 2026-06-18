"""In-guest shim + registry for the vendored OSWorld getters.

OSWorld's getters (``get_bookmarks``, ``get_vlc_config``, ``get_accessibility_tree``,
…) are written against an ``env`` whose ``env.controller`` talks to the in-guest
controller server (``/execute``, ``/file``, …). To run them with exact parity
*without* that server, this module provides a controller shim that routes those two
dominant calls — ``execute_python_command`` (76 uses) and ``get_file`` (20 uses) —
through the same injected ``run_command`` the verifier already uses. So the vendored
getters run unchanged, in whatever sandbox the verifier runs in.

A few getters need live state the shim can't synthesize without the real controller
(``get_accessibility_tree`` — pyatspi; chrome CDP tab state). Those raise so the
caller surfaces it rather than silently scoring 0.
"""

from __future__ import annotations

import base64
import importlib
import os
import re
import shlex
import sys
from collections.abc import Callable
from typing import Any

_VENDOR_ROOT = os.path.join(os.path.dirname(__file__), "_osworld_vendor")
_PKG = "desktop_env.evaluators.getters"


def _ensure_path() -> None:
    if _VENDOR_ROOT not in sys.path:
        sys.path.insert(0, _VENDOR_ROOT)


def _build_getter_module_map() -> dict[str, str]:
    """Map getter ``type`` -> module by scanning each getters/*.py for ``def get_<type>``.
    A task's getter ``type`` X is served by upstream ``get_X``."""
    gdir = os.path.join(_VENDOR_ROOT, "desktop_env", "evaluators", "getters")
    mapping: dict[str, str] = {}
    try:
        files = sorted(os.listdir(gdir))
    except OSError:
        return {}
    for fname in files:
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        module = fname[:-3]
        try:
            with open(os.path.join(gdir, fname), encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        for name in re.findall(r"^def get_([a-zA-Z]\w*)\s*\(", src, re.M):
            mapping.setdefault(name, module)
    return mapping


GETTER_MODULE: dict[str, str] = _build_getter_module_map()


class _ShimController:
    """Minimal OSWorld controller backed by the verifier's ``run_command`` channel.

    Implements the calls the getters actually use; everything routes through the
    same in-sandbox exec the rest of the evaluator uses, so behaviour matches
    whether the verifier runs in-guest (subprocess) or over a remote exec.
    """

    def __init__(self, run_command: Callable[[Any, bool], str]) -> None:
        self._run = run_command

    def execute_python_command(self, command: str) -> dict[str, Any]:
        # Faithful to PythonController.execute_python_command: run `python3 -c`,
        # return {status, output, error, returncode}. Getters read ['output'].
        out = self._run(f"python3 -c {shlex.quote(command)}", True) or ""
        return {"status": "success", "output": out, "error": "", "returncode": 0}

    def get_file(self, file_path: str) -> bytes | None:
        # Byte-exact read over the exec channel (base64, like the controller's /file).
        b64 = (
            self._run(f"base64 -w0 {shlex.quote(file_path)} 2>/dev/null", True) or ""
        ).strip()
        if not b64:
            return None
        try:
            return base64.b64decode(b64)
        except Exception:
            return None

    def get_vm_directory_tree(self, path: str) -> Any:
        cmd = (
            "import os, json\n"
            f"r={path!r}\n"
            "def walk(p):\n"
            " n={'type':'directory','name':os.path.basename(p) or p,'children':[]}\n"
            " try:\n"
            "  for e in sorted(os.listdir(p)):\n"
            "   fp=os.path.join(p,e)\n"
            "   n['children'].append(walk(fp) if os.path.isdir(fp) else {'type':'file','name':e})\n"
            " except Exception: pass\n"
            " return n\n"
            "print(json.dumps(walk(r)) if os.path.exists(r) else 'null')"
        )
        import json

        out = self.execute_python_command(cmd)["output"].strip()
        try:
            return json.loads(out)
        except Exception:
            return None

    def get_terminal_output(self) -> str | None:
        return None

    def __getattr__(
        self, name: str
    ) -> Any:  # live-state controller calls we can't shim
        def _unsupported(*_a: Any, **_k: Any) -> Any:
            raise NotImplementedError(
                f"OSWorld controller.{name} needs the live controller server "
                "(run OSWorld's own env, route A); not available in the in-guest shim."
            )

        return _unsupported


class ShimEnv:
    """Stand-in for OSWorld ``DesktopEnv`` exposing what getters read off ``env``."""

    def __init__(self, run_command: Callable[[Any, bool], str], cache_dir: str) -> None:
        self.controller = _ShimController(run_command)
        self.cache_dir = cache_dir
        self.vm_platform = "Linux"
        self.vm_ip = "localhost"
        self.server_port = 5000
        self.chromium_port = 9222
        self.vlc_port = 8080
        self.current_use_proxy = False


def resolve_vendored_getter(gtype: str) -> Callable[..., Any] | None:
    """Return upstream ``get_<gtype>`` (lazy import), or ``None`` if unknown."""
    module = GETTER_MODULE.get(gtype)
    if module is None:
        return None
    _ensure_path()
    try:
        mod = importlib.import_module(f"{_PKG}.{module}")
    except Exception as exc:
        raise NotImplementedError(
            f"OSWorld getter {gtype!r} lives in vendored module {module!r}, whose "
            f"deps are not installed ({type(exc).__name__})."
        ) from exc
    return getattr(mod, f"get_{gtype}", None)
