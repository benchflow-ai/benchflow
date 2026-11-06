"""Regression tests for bundled docs/example tasks."""

from __future__ import annotations
import __future__

import ast
import importlib.util
import json
import subprocess
import sys
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
    """Visible scoring increments should allow reward 1.0."""
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
    """docs/examples scripts must not reference removed modules."""
    assert script.exists(), f"missing example script: {script}"
    imports = _imports_from(script)
    # benchflow.user was removed; FunctionUser/RoundResult now live at top-level
    # benchflow (re-exported from benchflow.sandbox.user).
    assert "benchflow.user" not in imports, (
        f"{script} imports the removed `benchflow.user` module; "
        "use `from benchflow import FunctionUser, RoundResult` instead"
    )


@pytest.mark.parametrize("task", ["citation-check", "citation-check-network"])
def test_citation_check_oracle_keeps_type_hints_unevaluated(task: str) -> None:
    """Guards PR #1160: the oracle's Python runs on whatever python3 the
    environment has, which can be older than 3.10, so its ``str | None``
    hints must stay unevaluated or defining the function raises TypeError.
    The network variant keeps its own copy of the oracle."""
    script = Path(
        f"docs/examples/task-md/real-skillsbench/{task}/oracle/solve.sh"
    ).read_text()
    body = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    assert "str | None" in body
    code = compile(body, "solve.sh", "exec", dont_inherit=True)
    assert code.co_flags & __future__.annotations.compiler_flag


def test_use_cases_mcp_import_path_is_experimental() -> None:
    """docs/tasks/scenes.md must use benchflow.experimental.mcp.*.

    Docs v2 moved scenes.md to docs/tasks/multi-turn.md, which shows
    no MCP hooks; no page may name the stale path."""
    for page in [Path("README.md"), *sorted(Path("docs").rglob("*.md"))]:
        assert "from benchflow.mcp.hooks" not in page.read_text(), (
            f"{page} references the stale benchflow.mcp.hooks path; "
            "use benchflow.experimental.mcp.hooks instead"
        )


def test_task_authoring_docs_require_dockerfile() -> None:
    """The environment page must not say compose can replace Dockerfile.

    `check_task` always requires `environment/Dockerfile`, so the multi-container
    section must not promise compose-only tasks.
    """
    # Docs v2: docs/tasks/environment.md is a section of task-format.md.
    text = Path("docs/tasks/task-format.md").read_text()
    assert "alongside (or instead of)" not in text, (
        "docs/tasks/task-format.md claims docker-compose can replace the "
        "Dockerfile, but `bench tasks check` rejects compose-only tasks"
    )


def test_reproduce_page_shows_what_compare_five_prints() -> None:
    """Reproduce a published score quotes compare_five.py's output; a change
    to either must show in the other."""
    out = subprocess.run(
        [sys.executable, "-I", "docs/examples/reproduce/compare_five.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() in Path("docs/audit/reproduce.md").read_text()
