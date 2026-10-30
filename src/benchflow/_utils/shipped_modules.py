"""Ship benchflow modules into a sandbox, where benchflow itself is not installed."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def package_files(*modules: str) -> dict[str, bytes]:
    """Each module's source at its package path, with an empty ``__init__.py`` per package.

    Placed in a directory on ``sys.path``, these let code shipped next to them
    import ``benchflow.x.y`` exactly as it does in-process. The modules must be
    self-contained: the stub packages provide nothing else.
    """
    files: dict[str, bytes] = {}
    for module in modules:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(f"cannot ship {module}: no source file")
        parts = module.split(".")
        for depth in range(1, len(parts)):
            files["/".join(parts[:depth]) + "/__init__.py"] = b""
        files["/".join(parts) + ".py"] = Path(spec.origin).read_bytes()
    return files
