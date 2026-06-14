"""Regression tests for bundled docs/example tasks."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest


def _load_nanofirm_evaluator():
    path = Path("docs/examples/nanofirm-task/tests/evaluate.py")
    spec = importlib.util.spec_from_file_location("nanofirm_evaluate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nanofirm_perfect_analysis_reaches_full_reward(tmp_path, monkeypatch):
    """Guards ENG-87: visible scoring increments should allow reward 1.0."""
    module = _load_nanofirm_evaluator()
    analysis = {
        "risks": [
            {
                "clause": "1",
                "severity": "high",
                "issue": "uncapped liability",
                "recommendation": "cap liability",
            },
            {
                "clause": "2",
                "severity": "medium",
                "issue": "broad assignment",
                "recommendation": "narrow assignment",
            },
            {
                "clause": "3",
                "severity": "medium",
                "issue": "weak termination rights",
                "recommendation": "add termination trigger",
            },
        ],
        "compound_risks": [
            {
                "clauses": ["1", "3"],
                "severity": "high",
                "issue": "uncapped post-termination exposure",
                "recommendation": "align cap and termination language",
            }
        ],
        "deal_breakers": ["uncapped liability"],
        "summary": "This analysis identifies the main commercial risks clearly.",
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))
    monkeypatch.setattr(module, "ANALYSIS_PATH", str(analysis_path))

    assert module.evaluate() == 1.0


# Issue #368 guards
# Example scripts under docs/examples/ must import from real, public
# modules. ModuleNotFoundError on import is a hard regression.

_DOCS_EXAMPLE_SCRIPTS = [
    Path("docs/examples/user_dogfood.py"),
    Path("docs/examples/swebench_pro_user_dogfood.py"),
]


def _imports_from(path: Path) -> set[str]:
    """Return the set of module names this script imports from."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


@pytest.mark.parametrize("script", _DOCS_EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_docs_example_imports_resolve(script: Path) -> None:
    """ENG-368: docs/examples scripts must not reference removed modules."""
    assert script.exists(), f"missing example script: {script}"
    imports = _imports_from(script)
    # benchflow.user was removed; FunctionUser/RoundResult now live at top-level
    # benchflow (re-exported from benchflow.sandbox.user).
    assert "benchflow.user" not in imports, (
        f"{script} imports the removed `benchflow.user` module; "
        "use `from benchflow import FunctionUser, RoundResult` instead"
    )


def test_use_cases_mcp_import_path_is_experimental() -> None:
    """ENG-368: docs/use-cases.md must use benchflow.experimental.mcp.*."""
    text = Path("docs/use-cases.md").read_text()
    # Reject the stale path; require the current one.
    assert "from benchflow.mcp.hooks" not in text, (
        "docs/use-cases.md references the stale benchflow.mcp.hooks path; "
        "use benchflow.experimental.mcp.hooks instead"
    )
    assert "benchflow.experimental.mcp.hooks" in text
