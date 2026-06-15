#!/usr/bin/env python3
"""Spot-check: list public top-level exports of a file (matches manifest logic)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: exports.py <file>\n")
        return 2
    p = Path(sys.argv[1])
    src = p.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)

    all_list: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    all_list = [
                        e.value
                        for e in node.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]

    for node in tree.body:
        name: str | None = None
        kind = ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name, kind = node.name, "func"
        elif isinstance(node, ast.ClassDef):
            name, kind = node.name, "class"
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, kind = node.targets[0].id, "const"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, kind = node.target.id, "annot"
        if name is None or name == "__all__":
            continue
        is_public = (
            name in all_list if all_list is not None else not name.startswith("_")
        )
        if is_public:
            print(f"{kind}\t{name}\t{node.lineno}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
