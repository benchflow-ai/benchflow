"""The CPU-time ceiling checks run with no xdist neighbours (the lead's ruling,
2026-11-18).

tests/_timing.py's within_ceiling holds one call to 10 s of CPU time (the
lead's ruling, 2026-11-14). Beside an xdist neighbour on the same core, CPU
time inflates: PR #378's run on 2-vCPU runners measured 12.7-12.8 s for the
10,000 unclosed documents, which take well under the ceiling alone. Every
test that calls within_ceiling carries the ``ceiling`` mark; conftest leaves
it out of an xdist run; within_ceiling fails if it is called under xdist;
test.yml runs them with -n 0 after the shard's run."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

TESTS = Path(__file__).resolve().parent
WORKFLOW = TESTS.parent / ".github" / "workflows" / "test.yml"


def _marked_ceiling(node: ast.AST) -> bool:
    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Attribute) and dec.attr == "ceiling":
            return True
    return False


def test_every_test_that_calls_within_ceiling_is_marked_ceiling():
    unmarked, found = [], 0
    for path in sorted(TESTS.rglob("*.py")):
        if path.name == "_timing.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) or not node.name.startswith("test"):
                continue
            calls = any(
                isinstance(n, ast.Call)
                and getattr(n.func, "id", getattr(n.func, "attr", None))
                == "within_ceiling"
                for n in ast.walk(node)
            )
            if calls:
                found += 1
                if not _marked_ceiling(node):
                    unmarked.append(f"{path.relative_to(TESTS)}::{node.name}")
    assert found >= 3, found
    assert unmarked == [], unmarked


@pytest.mark.ceiling
def test_within_ceiling_fails_under_xdist(monkeypatch):
    from tests import _timing

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    with pytest.raises(AssertionError, match="under xdist"):
        _timing.within_ceiling(lambda: None)


def test_ci_runs_the_ceiling_checks_alone_after_the_shard():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    for name in ("unit", "unit-latest-os"):
        steps = jobs[name]["steps"]
        runs = [step.get("run") or "" for step in steps]
        main = [
            i
            for i, run in enumerate(runs)
            if "pytest tests/" in run and "-n auto" in run
        ]
        alone = [
            i for i, run in enumerate(runs) if "-m ceiling" in run and "-n 0" in run
        ]
        assert main and alone and main[0] < alone[0], (name, main, alone)
