"""No module of the package warns as it compiles: the first import printed
``SyntaxWarning: invalid escape sequence '\\;'`` (sandbox/_base.py)."""

from __future__ import annotations

import warnings
from pathlib import Path

import benchflow

SRC = Path(benchflow.__file__).parent


def test_every_module_compiles_without_a_warning():
    warned = []
    for path in sorted(SRC.rglob("*.py")):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                compile(path.read_bytes(), str(path), "exec")
            except SyntaxError as error:
                warned.append(f"{path.relative_to(SRC)}: {error}")
    assert warned == []


def test_the_value_check_keeps_its_backslash():
    """The shell text is unchanged: a ';' in a loader list is refused."""
    from benchflow.sandbox._base import env_value_check_func

    assert ") case $2 in *\\;*) exit 1;; esac; " in env_value_check_func()
