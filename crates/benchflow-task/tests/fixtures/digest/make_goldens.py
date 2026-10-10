"""Golden digests for benchflow-task's digest tests (tests/digest_golden.rs).

usage: python make_goldens.py <fixtures/digest dir> <task-md tools dir> <BenchFlow 0.8 src dir>

Builds each tree in cases.json in a temporary folder and writes expected.json with three digests per tree:

- harbor: Harbor's package digest (Packager.compute_content_hash, prefixed with "sha256:"), written out below:
  task.toml, instruction.md, README.md, and trajectory.json, then every file under environment/, tests/, solution/,
  and steps/ (Path.rglob, which does not enter linked folders), minus what the task's .gitignore ignores, or Harbor's
  default ignores when it has none, read with pathspec's "gitignore" patterns. Files are sorted by path, and each
  adds "{path}\\0{sha256 hex}\\n" to an outer SHA-256. Harbor's own code is not run. For trees without .gitignore,
  task.md's tools/evidence.py harbor_digest, an independent implementation, must agree.
- tree: task.md's tree hash, tools/evidence.py tree_hash.
- benchflow: BenchFlow 0.8's task_digest (benchflow._utils.task_authoring in the 0.8 source given, such as this
  repository's src/); tools/evidence.py benchflow_digest must agree.

Needs pathspec 1.x and BenchFlow 0.8's dependencies (the repository's own virtual environment has both).
"""

import base64
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pathspec

fixtures, tools, src = (Path(a).resolve() for a in sys.argv[1:4])
sys.path[:0] = [str(tools), str(src)]
import evidence  # noqa: E402
from benchflow._utils.task_authoring import task_digest  # noqa: E402

DEFAULT_IGNORES = ["__pycache__/", "*.pyc", ".DS_Store", "*.swp", "*.swo", "*~"]


def harbor(task_dir: Path) -> str:
    task_dir = task_dir.resolve()
    files = [task_dir / n for n in ("task.toml", "instruction.md", "README.md", "trajectory.json")]
    files = [f for f in files if f.exists()]
    for name in ("environment", "tests", "solution", "steps"):
        folder = task_dir / name
        if folder.exists():
            files += [p for p in folder.rglob("*") if p.is_file()]
    gitignore = task_dir / ".gitignore"
    lines = gitignore.read_text().splitlines() if gitignore.exists() else DEFAULT_IGNORES
    spec = pathspec.PathSpec.from_lines("gitignore", lines)
    files = [f for f in files if not spec.match_file(f.relative_to(task_dir).as_posix())]
    files.sort(key=lambda p: p.relative_to(task_dir).as_posix())
    outer = hashlib.sha256()
    for f in files:
        rel = f.relative_to(task_dir).as_posix()
        outer.update(f"{rel}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return "sha256:" + outer.hexdigest()


def build(root: Path, files: dict) -> None:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(body["base64"]) if isinstance(body, dict) else body.encode("utf-8")
        path.write_bytes(data)


cases = json.loads((fixtures / "cases.json").read_text(encoding="utf-8"))
expected = {}
for name, case in cases.items():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / name
        build(root, case["files"])
        digests = {"harbor": harbor(root), "tree": evidence.tree_hash(root), "benchflow": task_digest(root)}
        assert digests["benchflow"] == evidence.benchflow_digest(root), name
        if not (root / ".gitignore").exists():
            assert digests["harbor"] == evidence.harbor_digest(root), name
        expected[name] = digests

(fixtures / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
print(f"wrote {len(expected)} cases")
