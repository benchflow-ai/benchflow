"""Prove that splitting string literals kept every runtime value.

    python tools/foldproof.py <tree> <base-rev>

Every ``.py`` file changed since ``<base-rev>`` must give the same AST it had at ``<base-rev>`` once each
concatenation of string (or bytes) constants and f-strings is folded. ``tools/keyshape_fix.py`` splits a
provider-shaped test literal (``"sk" + "_live_..."``) so no source line carries the shape; this is its proof that
the values the code builds are the ones it had. Exit 0 when every changed file folds equal, 1 otherwise.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


class Fold(ast.NodeTransformer):
    """Folds every concatenation of string or bytes constants and f-strings, and merges each f-string's adjacent
    constant parts, so a literal split into pieces folds back to the literal's own AST."""

    @staticmethod
    def _norm(values: list[ast.expr]) -> list[ast.expr]:
        out: list[ast.expr] = []
        for v in values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                if not v.value:
                    continue
                last = out[-1] if out else None
                if isinstance(last, ast.Constant) and isinstance(last.value, str):
                    out[-1] = ast.Constant(last.value + v.value)
                    continue
            out.append(v)
        return out

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        self.generic_visit(node)
        node.values = self._norm(node.values)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.op, ast.Add):
            return node
        left, right = node.left, node.right
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            lv, rv = left.value, right.value
            if isinstance(lv, str) and isinstance(rv, str):
                return ast.Constant(lv + rv)
            if isinstance(lv, bytes) and isinstance(rv, bytes):
                return ast.Constant(lv + rv)

        def is_str(n: ast.expr) -> bool:
            return isinstance(n, ast.JoinedStr) or (
                isinstance(n, ast.Constant) and isinstance(n.value, str)
            )

        if (
            is_str(left)
            and is_str(right)
            and (isinstance(left, ast.JoinedStr) or isinstance(right, ast.JoinedStr))
        ):

            def parts(n: ast.expr) -> list[ast.expr]:
                return list(n.values) if isinstance(n, ast.JoinedStr) else [n]

            return ast.JoinedStr(values=self._norm(parts(left) + parts(right)))
        return node


def folded(src: str) -> str:
    """The module's AST with every constant string concatenation folded, as text."""
    return ast.dump(Fold().visit(ast.parse(src)), include_attributes=False)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print((__doc__ or "").split("\n\n")[1].strip(), file=sys.stderr)
        return 2
    tree, base = Path(args[0]), args[1]
    names = subprocess.run(
        ["git", "-C", str(tree), "diff", "--name-only", base, "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    bad = []
    for rel in names:
        if not (tree / rel).exists():
            continue
        old = subprocess.run(
            ["git", "-C", str(tree), "show", f"{base}:{rel}"],
            capture_output=True,
            text=True,
        )
        if old.returncode:
            continue  # a new file has nothing to keep
        if folded(old.stdout) != folded((tree / rel).read_text()):
            bad.append(rel)
    print(
        f"changed .py files {len(names)}; folded AST equal {len(names) - len(bad)}; different {len(bad)}"
    )
    for rel in bad:
        print(f"  DIFFERENT {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
