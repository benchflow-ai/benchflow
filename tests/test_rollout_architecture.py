"""Rollout kernel architecture tests."""

from __future__ import annotations

import ast
from pathlib import Path

ROLL_OUT = Path("src/benchflow/rollout.py")

CONCRETE_PLANE_MODULES = {
    "benchflow.acp.client",
    "benchflow.acp.runtime",
    "benchflow.environment.manifest_env",
    "benchflow.providers.runtime",
    "benchflow.sandbox.daytona",
    "benchflow.sandbox.lockdown",
    "benchflow.sandbox.setup",
    "benchflow.sandbox.user",
}

COMPOSITION_BOUNDARY_MODULES = {
    "benchflow.acp.runtime",
    "benchflow.environment.manifest_env",
    "benchflow.providers.runtime",
    "benchflow.sandbox.lockdown",
    "benchflow.sandbox.setup",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_rollout_kernel_does_not_import_concrete_planes() -> None:
    """Guards the fix from PR #515 for issue #415: rollout imports contracts."""
    imported = _imported_modules(ROLL_OUT)

    assert imported.isdisjoint(CONCRETE_PLANE_MODULES)
    assert "benchflow.contracts" in imported
    assert "benchflow.rollout_planes" not in imported


def test_concrete_plane_bindings_live_at_composition_boundary() -> None:
    """Guards the fix from PR #515 for issue #415: concrete imports stay outside."""
    imported = _imported_modules(Path("src/benchflow/rollout_planes.py"))

    assert imported >= COMPOSITION_BOUNDARY_MODULES
