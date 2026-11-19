#!/usr/bin/env python3
"""The rc's cleanup after its last merge, before its gate.

    uv run python tools/release_gate/rc_cleanup.py

Run on a clean checkout of the rc branch's head, in the test environment (the
generated pages come out the same only there). It does two things and nothing
else, then checks its own diff:

1. strips our tracker's keys and the internal review tags from the comments
   and docstrings of the Python files under src, tests and tools
   (``tools/strip_internal_refs.py --write``);
2. regenerates the pages generated from the code: the reference pages and the
   changelog page, the error index and the CLI help golden (the commands in
   ``run_gate.GENERATED_CHECKS``).

The diff may hold only those changes: a generated page, or a Python file whose
code is unchanged (its syntax tree, docstrings aside, is the same: only
comments and docstrings changed). Any other change, or a file left untracked,
fails it, with each file named; so does a reference the strip cannot remove
(in code, a string, Markdown, a fixture): those go in a PR of their own. When
it passes, commit the diff as the cleanup PR; then ``tools/public_tree_guard.py``
must report nothing on it.

Exit 0: in scope and nothing left; 1: out of scope, a generator failed or a
reference is left for a person; 2: the checkout was not clean.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import public_tree_guard as guard  # noqa: E402
from tools import strip_internal_refs as strip  # noqa: E402
from tools.release_gate import run_gate  # noqa: E402

STRIP_PATHS = ("src", "tests", "tools")
# What the generators write (docs_reference.PAGES, docs_errors.PAGE,
# cli_help_golden.GOLDEN; tests/test_rc_cleanup.py checks they agree).
GENERATED_PAGES = frozenset(
    {
        "docs/reference/commands.md",
        "docs/reference/python.md",
        "docs/reference/cli-python.md",
        "docs/reference/files.md",
        "docs/versions/changelog.md",
        "docs/reference/errors.md",
        "tests/fixtures/cli_help_golden.txt",
    }
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def code_of(source: str) -> str | None:
    """A Python source's code: its syntax tree with docstrings left out
    (comments are not in it). None when it does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body[0] = ast.Pass()
    return ast.dump(tree)


def out_of_scope(root: Path) -> list[tuple[str, str]]:
    """Every change against HEAD that is not a generated page or a Python
    file with only its comments and docstrings changed, with why."""
    found: list[tuple[str, str]] = []
    for line in _git(
        root, "diff", "--name-status", "--no-renames", "HEAD"
    ).splitlines():
        status, _, path = line.partition("\t")
        if path in GENERATED_PAGES:
            continue
        if status != "M":
            found.append((path, f"{status}: only existing files may change"))
        elif not path.endswith(".py"):
            found.append((path, "not a generated page or a Python file"))
        else:
            before = _git(root, "show", f"HEAD:{path}")
            after = (root / path).read_text()
            code = code_of(after)
            if code is None or code != code_of(before):
                found.append(
                    (path, "its code changed, not only comments and docstrings")
                )
    for path in _git(root, "ls-files", "--others", "--exclude-standard").splitlines():
        found.append((path, "untracked: the cleanup adds no file"))
    return found


def _regenerate_all(root: Path) -> list[str]:
    failed = []
    for _what, args, regenerate in run_gate.GENERATED_CHECKS:
        tool = args[1]
        done = subprocess.run([sys.executable, tool, "--write"], cwd=root)
        if done.returncode:
            failed.append(regenerate)
    return failed


def main(
    argv: list[str] | None = None,
    root: Path = ROOT,
    regenerate: Callable[[Path], list[str]] = _regenerate_all,
) -> int:
    del argv
    if _git(root, "status", "--porcelain", "--untracked-files=normal").strip():
        print(
            "rc_cleanup: the checkout has changes: start from the rc head",
            file=sys.stderr,
        )
        return 2
    here = Path.cwd()
    os.chdir(root)
    try:
        left = strip.main(["--write", *STRIP_PATHS])
    finally:
        os.chdir(here)
    failed = regenerate(root)
    problems = [f"{p}: {why}" for p, why in out_of_scope(root)]
    problems += [f"regenerating failed: {cmd}" for cmd in failed]
    left_keys = []
    for path in _git(root, "ls-files", "-z").split("\0"):
        file = root / path
        if not path or not file.is_file() or file.is_symlink():
            continue
        try:
            text = file.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        left_keys += guard.scan_text(path, text)
    changed = _git(root, "diff", "--name-only", "HEAD").split()
    print(
        f"rc_cleanup: {len(changed)} file(s) changed; {len(problems)} out of scope; "
        f"{len(left_keys)} tracker key(s) left for a PR of their own",
        file=sys.stderr,
    )
    for problem in problems:
        print(f"out of scope: {problem}")
    for f in left_keys:
        print(
            f"left: {f['path']}:{f['line']}: {f['kind']} {f.get('token', '')}".rstrip()
        )
    return 1 if problems or left_keys or left else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
