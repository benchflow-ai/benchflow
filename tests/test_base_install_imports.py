"""Base-install import smoke tests — guards against #358.

Issue #358: a base install of ``benchflow`` (no ``sandbox-daytona`` /
``sandbox-modal`` extras) used to fail at ``import benchflow`` because the
import chain reached ``benchflow.sandbox.daytona`` which imported ``tenacity``
at module top. Optional sandbox SDKs (``tenacity``, the ``daytona`` python
package) must now be lazy-loaded inside the methods that actually need them.

The tests below run inside the project's own venv (where tenacity may well be
installed) but exercise the structural guarantees that keep #358 from
regressing: ``SandboxStartupError`` must live in the optional-dep-free
``protocol`` module, ``rollout.py`` must import it from there, and the
``daytona`` module must import cleanly even when ``tenacity`` is hidden.
"""

from __future__ import annotations

import builtins
import importlib
import sys


def test_sandbox_startup_error_lives_in_protocol():
    """Guards the fix from PR #486 for #358: the exception must be importable
    from ``benchflow.sandbox.protocol`` —
    the module that has no optional-dep imports — so a base install can
    reference it via ``benchflow.rollout`` without pulling Daytona deps."""
    from benchflow.sandbox.protocol import SandboxStartupError

    err = SandboxStartupError(
        "boom", sandbox_id="sb-1", sandbox_state="error", attempts=3
    )
    assert isinstance(err, RuntimeError)
    info = err.diagnostic.to_dict()
    assert info["sandbox_id"] == "sb-1"
    assert info["attempts"] == 3


def test_legacy_sandbox_startup_error_import_path_still_works():
    """Guards the fix from PR #486 for #358: existing code does
    ``from benchflow.sandbox.daytona import
    SandboxStartupError``; the re-export must point at the same class as the
    new protocol-module home so isinstance checks across both paths agree."""
    from benchflow.sandbox.daytona import (
        SandboxStartupError as DaytonaPathSandboxStartupError,
    )
    from benchflow.sandbox.protocol import (
        SandboxStartupError as ProtocolPathSandboxStartupError,
    )
    assert DaytonaPathSandboxStartupError is ProtocolPathSandboxStartupError


def test_rollout_imports_sandbox_startup_error_from_protocol_not_daytona():
    """Guards the fix from PR #486 for #358: ``rollout.py`` must not import
    from ``benchflow.sandbox.daytona`` —
    the daytona module path forces evaluation of optional Daytona deps and
    breaks the base install. This guards against #358 regressing as someone
    moves the import back."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "benchflow" / "rollout.py"
    text = src.read_text()
    assert "from benchflow.sandbox.protocol import SandboxStartupError" in text, (
        "rollout.py must source SandboxStartupError from protocol (no optional "
        "deps), not from benchflow.sandbox.daytona (forces tenacity / "
        "daytona-SDK at import time)"
    )
    assert "from benchflow.sandbox.daytona import SandboxStartupError" not in text


def test_daytona_module_imports_without_tenacity():
    """Guards the fix from PR #486 for #358: hide ``tenacity`` from
    ``sys.modules`` and pretend the daytona SDK is missing, then re-import
    ``benchflow.sandbox.daytona``. The module must
    load cleanly — the optional deps are only materialized when
    ``DaytonaSandbox`` is actually instantiated (#358)."""

    real_import = builtins.__import__
    blocked = {"tenacity", "daytona", "daytona._async.snapshot"}

    def guarded_import(name, *args, **kwargs):
        if name in blocked or any(name.startswith(b + ".") for b in blocked):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    # Drop any cached modules so the import below actually re-executes.
    for mod_name in list(sys.modules):
        if mod_name == "tenacity" or mod_name.startswith("tenacity."):
            sys.modules.pop(mod_name)
        if mod_name == "daytona" or mod_name.startswith("daytona."):
            sys.modules.pop(mod_name)
        if mod_name == "benchflow.sandbox.daytona":
            sys.modules.pop(mod_name)

    builtins.__import__ = guarded_import
    try:
        mod = importlib.import_module("benchflow.sandbox.daytona")
        # The module loaded — that's the guarantee. The re-exported error
        # type must still be reachable through it.
        assert mod.SandboxStartupError is not None
    finally:
        builtins.__import__ = real_import
