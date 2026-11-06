"""The Python API reference's snippets compile and name real APIs.

Skill references once imported ``Job``/``JobConfig`` long
after they were removed. This compiles every ``python`` block of
docs/reference/python-api.md (top-level ``await`` allowed, as in a notebook)
and checks that every ``bf.<name>`` and ``from benchflow import <name>`` they
use exists.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import benchflow as bf

DOC = Path(__file__).resolve().parents[1] / "docs/reference/python-api.md"
BLOCKS = re.findall(r"```python\n(.*?)```", DOC.read_text(), flags=re.S)


def test_the_page_has_snippets() -> None:
    assert len(BLOCKS) > 10


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_snippet_compiles_and_names_real_apis(index: int) -> None:
    source = BLOCKS[index]
    tree = compile(
        source,
        f"python-api.md[{index}]",
        "exec",
        flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "bf"
        ):
            assert hasattr(bf, node.attr), f"bf.{node.attr} does not exist"
        if isinstance(node, ast.ImportFrom) and node.module == "benchflow":
            for alias in node.names:
                assert hasattr(bf, alias.name), f"benchflow.{alias.name} does not exist"
