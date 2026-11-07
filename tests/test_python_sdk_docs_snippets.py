"""The docs' Python snippets compile and name real APIs.

Skill references once imported ``Job``/``JobConfig`` long
after they were removed. This compiles every ``python`` block of the README
and the docs' written pages (top-level ``await`` allowed, as in a notebook)
and checks that every ``bf.<name>`` and ``from benchflow import <name>`` they
use exists. Docs v2 spread the worked examples of
docs/reference/python-api.md over the pages that use them; the generated
docs/reference/python.md shows signatures, not code.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

import benchflow as bf

REPO = Path(__file__).resolve().parents[1]
NAV = json.loads((REPO / "docs" / "nav.json").read_text())
PAGES = [REPO / "README.md"] + [
    REPO / "docs" / f"{page['path']}.md"
    for section in NAV["sections"]
    for page in section["pages"]
    if page["kind"] != "generated"
]
BLOCKS = [
    (page.relative_to(REPO).as_posix(), block)
    for page in PAGES
    for block in re.findall(r"```python\n(.*?)```", page.read_text(), flags=re.S)
]


def test_the_page_has_snippets() -> None:
    assert BLOCKS


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_snippet_compiles_and_names_real_apis(index: int) -> None:
    page, source = BLOCKS[index]
    tree = compile(
        source,
        f"{page}[{index}]",
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
