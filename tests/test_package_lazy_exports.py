"""``benchflow``'s public names resolve on first use (PEP 562).

``import benchflow`` loads no submodule; every name in ``__all__`` still
imports and is the object its defining module holds; the TYPE_CHECKING
imports type checkers read and the runtime table agree.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import pytest

import benchflow

_INIT = Path(benchflow.__file__)


def _run(code: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return proc.stdout


def _type_checking_imports() -> dict[str, tuple[str, str]]:
    """``alias -> (module, name)`` for every import under TYPE_CHECKING."""
    tree = ast.parse(_INIT.read_text(encoding="utf-8"))
    names: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for stmt in node.body:
                assert isinstance(stmt, ast.ImportFrom) and stmt.module
                for alias in stmt.names:
                    names[alias.asname or alias.name] = (stmt.module, alias.name)
    return names


def test_import_benchflow_loads_no_submodule():
    loaded = _run(
        "import sys, benchflow; "
        "print('\\n'.join(m for m in sys.modules if m.startswith('benchflow')))"
    ).split()
    assert loaded == ["benchflow"]


def test_type_checking_imports_match_the_runtime_table():
    assert _type_checking_imports() == benchflow._LAZY


def test_all_is_exactly_the_lazy_names_and_the_version():
    assert sorted(benchflow.__all__) == sorted([*benchflow._LAZY, "__version__"])
    assert len(benchflow.__all__) == len(set(benchflow.__all__))


@pytest.mark.parametrize("name", sorted(benchflow.__all__))
def test_every_public_name_imports(name):
    namespace: dict[str, object] = {}
    exec(f"from benchflow import {name}", namespace)
    value = namespace[name]
    if name == "__version__":
        assert value == version("benchflow")
        return
    module, attr = benchflow._LAZY[name]
    assert value is getattr(importlib.import_module(module), attr)
    assert getattr(benchflow, name) is value


def test_star_import_binds_every_public_name():
    out = _run(
        "from benchflow import *\n"
        "import benchflow\n"
        "missing = [n for n in benchflow.__all__ if n not in globals()]\n"
        "print(missing)\n"
    )
    assert out.strip() == "[]"


def test_public_branch_function_wins_over_the_branch_submodule():
    """``benchflow.branch`` is the function even when the submodule loads first."""
    out = _run(
        "import benchflow.branch as _m\n"
        "import benchflow\n"
        "from benchflow import branch\n"
        "from benchflow.branch_api import branch as real\n"
        "print(branch is real, benchflow.branch is real, callable(branch))\n"
    )
    assert out.split() == ["True", "True", "True"]


def test_deprecated_snapshot_aliases_still_warn_and_resolve():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = benchflow.snapshot
    assert alias is benchflow.workspace_snapshot
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_submodules_still_resolve_as_attributes():
    assert benchflow.doctor is importlib.import_module("benchflow.doctor")
    with pytest.raises(AttributeError):
        _ = benchflow.no_such_name


def test_dir_lists_the_public_names():
    assert set(benchflow.__all__) <= set(dir(benchflow))
