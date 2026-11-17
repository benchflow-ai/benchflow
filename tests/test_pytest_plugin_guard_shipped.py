"""The pytest plugin guard that goes into the sandbox fits one ``sh -c`` argument, and is the guard file's code.

Hardening writes the armed guard inside one ``sh -c`` argument, which Linux
caps at 128 KiB (MAX_ARG_STRLEN). With R334's fix the rel guard's install
reached 131,577 bytes in the exec-diet goldens, and every one failed with
``Argument list too long``. What ships now has no comments and no
docstrings (each one ``""``), and the judge section stays out as before: its
code, and every line number in a traceback, are the guard file's.

The cut's check list runs this file::

    PYTHONPATH=$PWD/src:$PWD python -m pytest -q -p no:cacheprovider \
        tests/test_pytest_plugin_guard_shipped.py
"""

from __future__ import annotations

import ast
import asyncio
import io
import tokenize
from pathlib import Path
from types import SimpleNamespace

from benchflow.sandbox import _image_manifest, lockdown
from benchflow.sandbox import _pytest_plugin_guard as guard

CAP = 128 * 1024
# Real strict trials on rel measured 123,435 bytes each (SWE-bench Verified
# astropy-7166 and -14508, sympy-20590, scikit-learn-10297 and -25747,
# matplotlib-24149, django-14915, SWT sphinx-9602), rc 109,852: about 550
# bytes above the shape below, which leaves out the trial's longer paths. A
# real task stays at least 4 KiB under the cap while this holds.
ROOM = 5 * 1024


def _file_without_the_judge() -> str:
    text = Path(guard.__file__).read_text()
    head, begin, rest = text.partition(lockdown._PYTEST_PLUGIN_GUARD_JUDGE_BEGIN)
    _, end, tail = rest.partition(lockdown._PYTEST_PLUGIN_GUARD_JUDGE_END)
    assert begin and end
    return head + tail


def _docstrings_emptied(tree: ast.AST) -> ast.AST:
    """*tree* with each docstring's value ``""``: the transform the shipped
    guard had, applied to a tree."""
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            first.value.value = ""
    return tree


def _starts(tree: ast.AST) -> list[tuple[str, int | None, int | None]]:
    return [
        (type(n).__name__, getattr(n, "lineno", None), getattr(n, "col_offset", None))
        for n in ast.walk(tree)
    ]


def _first_difference(left, right) -> str:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return f"at {index}: {str(a)[:200]!r} != {str(b)[:200]!r}"
    return f"lengths {len(left)} != {len(right)}"


def test_the_shipped_guard_is_the_files_code_without_comments_or_docstrings():
    """The same transform on both sides: the shipped script and the guard
    file (without its judge section) are one module once each docstring is
    ``""``, and every node starts on the same line and column."""
    shipped = lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT
    source = _file_without_the_judge()
    left = ast.dump(_docstrings_emptied(ast.parse(shipped)))
    right = ast.dump(_docstrings_emptied(ast.parse(source)))
    assert left == right, _first_difference(left, right)
    left_starts, right_starts = _starts(ast.parse(shipped)), _starts(ast.parse(source))
    assert left_starts == right_starts, _first_difference(left_starts, right_starts)
    assert shipped.count("\n") == source.count("\n")
    comments = [
        token
        for token in tokenize.generate_tokens(io.StringIO(shipped).readline)
        if token.type == tokenize.COMMENT
    ]
    assert comments == []
    assert len(shipped) < len(source)
    ast.parse(shipped, feature_version=(3, 6))


def test_a_docstring_is_emptied_and_its_lines_kept():
    source = 'def f():\n    """One\n\n    two."""\n    return "x"\n'
    assert lockdown._without_docstrings(source) == (
        'def f():\n    ""\n\n\n    return "x"\n'
    )


def test_a_comment_inside_a_string_stays():
    source = 'x = "# kept"  # cut\ny = 1\n# cut too\n'
    assert lockdown._without_comments(source) == 'x = "# kept"\ny = 1\n\n'


def test_the_guard_install_fits_one_sh_c_argument_with_room(monkeypatch):
    """A real strict trial's guard install: the agent's trees as a shared
    verifier blocks them, a root test.sh with its record, the image
    manifest, a conda base on the verifier PATH."""
    commands = []

    async def read(env, command, *args, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(lockdown, "_checked_read", read)
    asyncio.run(
        lockdown._install_pytest_plugin_guard(
            SimpleNamespace(),
            "agent",
            "/testbed",
            "",
            verifier_path="/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin",
            tests_dir="/tests",
            blocked=(
                "/logs",
                "/logs/agent",
                "/logs/artifacts",
                "/logs/verifier",
                "/testbed",
            ),
            verifier_uid="0",
            agent_gids=(1000,),
            snapshot="/testbed_verify",
            record=True,
            named_inputs=(),
            manifest=_image_manifest.MANIFEST_PATH,
        )
    )
    (command,) = commands
    assert len(command.encode()) <= CAP - ROOM, len(command.encode())
