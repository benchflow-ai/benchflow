"""tools/strip_internal_refs.py --write changes only comments and docstrings.

The tracker-key cleanup before an rc runs it with --write over the tree, and
then the public tree guard must find nothing. Nothing it rewrites may be code,
or text a test asserts on or runs (fixtures, scripts, Markdown): those it
reports for a person. The keys here are spelled in pieces, so the public tree
guard passes on this file.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from tools import public_tree_guard as guard
from tools import strip_internal_refs as strip

KEY = "E" + "N-12"
# One key of each of our tracker's teams.
TEAM_KEYS = [
    "E" + "N-12",
    "EN" + "G-7",
    "BEN" + "3-45",
    "B" + "D-6",
    "CE" + "A-3",
    "OS" + "S-9",
]


def _code(source: str) -> str:
    """The module's code, docstrings left out (comments are not in an AST)."""
    tree = ast.parse(source)
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


def test_write_changes_comments_and_docstrings_and_nothing_else(tmp_path, capsys):
    source = (
        f'"""The module ({KEY})."""\n'
        f"# Fixed ({KEY}).\n"
        f'X = "{KEY}"\n'
        f'Y = f"{{X}} ({KEY})"\n'
        "\n\n"
        "def f():\n"
        f'    """Guards {KEY}: the edge."""\n'
        f'    return "({KEY})"  # ({KEY})\n'
    )
    module = tmp_path / "pkg" / "mod.py"
    module.parent.mkdir()
    module.write_text(source)
    others = {
        tmp_path / "pkg" / "fixture.json": json.dumps({"note": f"seen ({KEY})"}),
        tmp_path / "pkg" / "run.sh": f"#!/bin/sh\n# ({KEY})\necho ok\n",
        tmp_path / "pkg" / "task.md": f"Solve it ({KEY}).\n",
        tmp_path / "pkg" / "conf.yaml": f"note: ({KEY})  # ({KEY})\n",
    }
    for path, text in others.items():
        path.write_text(text)

    assert strip.main(["--write", str(tmp_path / "pkg")]) == 1
    out = capsys.readouterr().out

    cleaned = module.read_text()
    assert cleaned.splitlines()[:2] == ['"""The module."""', "# Fixed."]
    assert '    """the edge."""' in cleaned
    assert _code(cleaned) == _code(source)
    # The code's strings keep their keys and are reported.
    assert f'X = "{KEY}"' in cleaned and f'return "({KEY})"' in cleaned
    for number in (3, 4, 9):
        assert f"mod.py:{number}:" in out
    # Every other file is reported, byte for byte as it was.
    for path, text in others.items():
        assert path.read_text() == text
        assert f"{path.name}:" in out


def test_without_write_nothing_changes(tmp_path):
    module = tmp_path / "mod.py"
    module.write_text(f"# Fixed ({KEY}).\n")
    assert strip.main([str(module)]) == 0
    assert module.read_text() == f"# Fixed ({KEY}).\n"


@pytest.mark.parametrize("key", TEAM_KEYS)
def test_the_tool_strips_every_key_the_guard_refuses(key):
    assert guard.TRACKER_KEY.fullmatch(key)
    assert strip.REF.fullmatch(key)
    assert strip.clean(f"# Fixed ({key}).") == "# Fixed."


@pytest.mark.parametrize("key", TEAM_KEYS)
def test_a_lower_case_key_is_a_reference_too(key, tmp_path):
    """Lead ruling on PR #146 (M3): the strip and the guard match our team
    keys in any case. A key in parentheses goes; inside a branch name it is
    left for a person, never cut out of the name."""
    lower = key.lower()
    assert strip.REF.fullmatch(lower)
    assert strip.clean(f"# Fixed ({lower}).") == "# Fixed."
    branch = f"``{lower}-audit-made-up``"
    assert strip.clean(f'"""Guards the fix on branch {branch}."""') == (
        f'"""Guards the fix on branch {branch}."""'
    )
    module = tmp_path / "m.py"
    module.write_text(f'"""Guards the fix on branch {branch}."""\n')
    assert strip.main(["--write", str(module)]) == 1  # left for a person
    assert branch in module.read_text()


def test_after_write_the_guard_finds_nothing_it_cleaned(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    module = repo / "src" / "m.py"
    module.write_text(
        '"""Stops early (' + ", ".join(TEAM_KEYS) + ')."""\n'
        f"x = 1  # see {KEY}\n"
        f"# ({KEY}): why x is 1\n"
    )
    assert strip.main(["--write", str(repo / "src")]) == 1  # "see <key>" is left
    module.write_text(module.read_text().replace(f"  # see {KEY}", ""))
    assert strip.main([str(repo / "src")]) == 0

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "tree")
    assert guard.scan_tree(repo) == []
    assert module.read_text() == '"""Stops early."""\nx = 1\n# why x is 1\n'


def test_the_tool_names_no_key_itself():
    """Its own source and docstring spell the teams, never a key."""
    root = Path(strip.__file__)
    assert guard.scan_text("tools/strip_internal_refs.py", root.read_text()) == []
