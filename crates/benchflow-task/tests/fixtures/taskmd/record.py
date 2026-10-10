"""Record what task-md's reference checker says about each fixture package.

    python3 record.py <task-md checkout> <fixtures dir>

For every package <fixtures>/<group>/<name>/ (groups: examples, imported, cases), writes
golden/<group>/<name>.json, what `python3 tools/taskmd.py json <name>` prints, and golden/<group>/<name>.check.txt,
what `check` prints, both run from the group's folder. Without PyYAML, a stand-in that reads JSON takes its place, as
task-md's tools/selftest.py does; the fixtures write their YAML (Compose files, frontmatter) as JSON, so either reads it
the same.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import types
from pathlib import Path

repo, fixtures = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo / "tools"))
try:
    import yaml  # noqa: F401
except ImportError:
    stub = types.ModuleType("yaml")
    stub.safe_load = lambda s: json.loads(s) if s and s.strip() else None
    stub.YAMLError = ValueError
    sys.modules["yaml"] = stub

import taskmd  # noqa: E402


def run(argv: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        taskmd.main(argv)
    return buf.getvalue()


count = 0
for group in ("examples", "imported", "cases"):
    root = fixtures / group
    if not root.is_dir():
        continue
    out = fixtures / "golden" / group
    out.mkdir(parents=True, exist_ok=True)
    for pkg in sorted(p for p in root.iterdir() if p.is_dir()):
        os.chdir(root)
        (out / f"{pkg.name}.json").write_text(run(["json", pkg.name]), encoding="utf-8")
        (out / f"{pkg.name}.check.txt").write_text(run(["check", pkg.name]), encoding="utf-8")
        count += 1
print(f"recorded {count} packages")
