"""Files the benchflow package ships, found wherever the package is installed.

The suite runs against a source checkout (``uv run pytest``, the package in
``src/benchflow``) and against the installed wheel (``tools/ci/fresh_resolve.sh``
runs a copy of the checkout without ``src/``). A test that builds a
``src/benchflow/...`` path only passes in the first, so tests reach package
files through ``importlib.resources`` instead.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def package_path(*parts: str) -> Path:
    """The path of a file or folder inside the installed ``benchflow``
    package, e.g. ``package_path("demo_task")``.

    benchflow is a regular package installed as files (source tree, editable
    or wheel install), so its resources are real paths a test can hand to
    code that wants a folder.
    """
    path = Path(str(files("benchflow").joinpath(*parts)))
    if not path.exists():
        raise FileNotFoundError(f"benchflow ships no {'/'.join(parts)} ({path})")
    return path


PACKAGE_ROOT = package_path()
