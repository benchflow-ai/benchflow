"""Bake the source commit into BenchFlow's sdist and wheel.

An installed wheel has no ``.git``, so ``benchflow.build_info`` reads the
commit and dirty state a build saw from ``benchflow/_build.json``, which this
hook writes into both distributions. ``uv build`` makes the sdist from the git
checkout, then the wheel from the unpacked sdist, which has no ``.git`` either:
that wheel keeps the file the sdist carries. A build outside any checkout bakes
null for both. An editable install bakes nothing; the checkout answers at run
time.

The file's bytes depend only on the commit and the dirty state, so rebuilding
the same commit gives the same distributions (the release's exact-hash check,
tools/check_release_acceptance.py).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import types
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACKAGE = Path("src") / "benchflow"


def _build_info(root: Path) -> types.ModuleType:
    """benchflow/build_info.py, run from its source: BenchFlow is not installed
    here, and an import would write __pycache__ into the tree it then calls
    dirty."""
    path = root / PACKAGE / "build_info.py"
    module = types.ModuleType("_benchflow_build_info")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


class BakeBuildIdentity(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            return
        root = Path(self.root).resolve()
        info = _build_info(root)
        identity = None
        if info.checkout_root(root) == root:
            identity = info.git_identity(root)
        if identity is None:
            if (root / PACKAGE / info.BAKED_FILE).is_file():
                # The unpacked sdist: keep what it was baked with.
                return
            identity = {"commit": None, "dirty": None}
        self._scratch = tempfile.mkdtemp(prefix="benchflow-build-")
        baked = Path(self._scratch) / info.BAKED_FILE
        baked.write_text(json.dumps(identity, sort_keys=True) + "\n")
        os.chmod(baked, 0o644)
        package = PACKAGE if self.target_name == "sdist" else Path("benchflow")
        build_data["force_include"][str(baked)] = (package / info.BAKED_FILE).as_posix()

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        scratch = getattr(self, "_scratch", None)
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
