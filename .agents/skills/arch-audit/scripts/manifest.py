#!/usr/bin/env python3
"""arch-audit manifest — one AST + git pass over src/benchflow/**/*.py (non-test)."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO / "src" / "benchflow"
CACHE_DIR = Path(__file__).parent / ".cache"
PKG_PREFIX = "benchflow"


def is_test_path(p: Path) -> bool:
    parts = p.parts
    return p.name.startswith("test_") or p.name == "conftest.py" or "tests" in parts


def collect_src_files() -> list[Path]:
    files: list[Path] = []
    for p in SRC_ROOT.rglob("*.py"):
        if is_test_path(p):
            continue
        files.append(p)
    files.sort()
    return files


def module_path(p: Path) -> str:
    """benchflow.foo.bar from src/benchflow/foo/bar.py (or /__init__.py)."""
    rel = p.relative_to(SRC_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def count_loc(p: Path) -> int:
    """Non-blank line count — matches `wc -l`-style intuition without counting pure whitespace."""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def extract_exports(tree: ast.Module, src: str) -> list[str]:
    """Public top-level value exports. Honors __all__ if present; else anything not prefixed with _ and not type-only."""
    all_list = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(
            node.value, (ast.List, ast.Tuple)
        ):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                all_list = [
                    e.value
                    for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
    exports: list[str] = []
    for node in tree.body:
        name: str | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            tgt = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(tgt, ast.Name):
                name = tgt.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name is None or name == "__all__":
            continue
        if all_list is not None:
            if name in all_list:
                exports.append(name)
        else:
            if not name.startswith("_"):
                exports.append(name)
    return exports


def is_barrel(tree: ast.Module) -> bool:
    """__init__.py whose body is only imports / __all__ / docstrings."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Assign):
            tgt = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                continue
        return False
    return True


def scan_all_py_files() -> list[Path]:
    """All .py files in repo excluding venv/node_modules/.cache."""
    out: list[Path] = []
    skip_parts = {
        ".venv",
        "node_modules",
        ".cache",
        ".git",
        "__pycache__",
        "dist",
        "build",
        ".ref",
        "labs",
    }
    # Names that are legitimate submodules of src/benchflow/ but also appear as
    # top-level run-output dirs — skip only when they're directly under REPO.
    skip_top_level = {"trajectories", ".smoke-jobs"}
    for p in REPO.rglob("*.py"):
        if any(part in skip_parts for part in p.parts):
            continue
        try:
            rel = p.relative_to(REPO)
        except ValueError:
            rel = p
        if rel.parts and rel.parts[0] in skip_top_level:
            continue
        out.append(p)
    return out


def build_importer_index(target_modules: set[str]) -> dict[str, list[tuple[Path, str]]]:
    """
    For each benchflow.* module, find importers.
    Returns: module_dotted → [(importer_path, specific_symbol_or_empty), ...].
    Detects:
      from benchflow.X import Y       → module=benchflow.X, symbol=Y
      from benchflow.X.Y import Z     → module=benchflow.X.Y, symbol=Z
      import benchflow.X              → module=benchflow.X, symbol=""
      from . import X (within pkg)    → resolved to parent.X
      from .X import Y                → resolved to parent.X, symbol=Y
    """
    idx: dict[str, list[tuple[Path, str]]] = {m: [] for m in target_modules}
    files = scan_all_py_files()
    for fp in files:
        try:
            src = fp.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(fp))
        except (OSError, SyntaxError):
            continue

        # Determine this file's package for relative-import resolution.
        try:
            rel = fp.relative_to(SRC_ROOT.parent)
            if rel.parts[0] != PKG_PREFIX:
                this_pkg = None
            else:
                parts = list(rel.with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                this_pkg = ".".join(parts[:-1]) if parts else PKG_PREFIX
        except ValueError:
            this_pkg = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    if this_pkg is None:
                        continue
                    base_parts = this_pkg.split(".")
                    if node.level > len(base_parts):
                        continue
                    base = ".".join(base_parts[: len(base_parts) - node.level + 1])
                    target = f"{base}.{mod}" if mod else base
                else:
                    target = mod
                if not target.startswith(PKG_PREFIX):
                    continue
                for alias in node.names:
                    # from benchflow.X import Y  → module=target, symbol=Y
                    if target in idx:
                        idx[target].append((fp, alias.name))
                    # Could also be from benchflow import X (imports submodule X)
                    sub = f"{target}.{alias.name}"
                    if sub in idx:
                        idx[sub].append((fp, ""))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(PKG_PREFIX) and alias.name in idx:
                        idx[alias.name].append((fp, ""))
    return idx


def git_commits_6mo(file_path: Path) -> int:
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "--since=6.months",
                "--format=%H",
                "--",
                str(file_path.relative_to(REPO)),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except (subprocess.TimeoutExpired, ValueError):
        return 0


def cache_key() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=5
        ).strip()
    except subprocess.SubprocessError:
        sha = "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "diff", "HEAD", "--", str(SRC_ROOT.relative_to(REPO))],
            cwd=REPO,
            text=True,
            timeout=10,
        )
    except subprocess.SubprocessError:
        dirty = ""
    if dirty.strip():
        suffix = hashlib.sha1(dirty.encode()).hexdigest()[:8]
        return f"{sha}-dirty-{suffix}"
    return sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    key = cache_key()
    cache_file = CACHE_DIR / f"{key}.json"
    if not args.no_cache and cache_file.exists():
        sys.stdout.write(cache_file.read_text())
        return 0

    files = collect_src_files()

    modules: dict[str, Path] = {}
    file_info: dict[Path, dict] = {}
    for p in files:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(p))
        except (OSError, SyntaxError):
            continue
        mod = module_path(p)
        modules[mod] = p
        file_info[p] = {
            "tree": tree,
            "src": src,
            "is_barrel": p.name == "__init__.py" and is_barrel(tree),
            "mod": mod,
        }

    idx = build_importer_index(set(modules.keys()))

    out_rows: list[dict] = []
    for p in files:
        if p not in file_info:
            continue
        info = file_info[p]
        mod = info["mod"]
        rel = str(p.relative_to(REPO))
        importer_hits = idx.get(mod, [])

        importers: list[str] = []
        barrel_importers: list[str] = []
        selector_map: dict[str, list[str]] = {}
        for imp_path, sym in importer_hits:
            imp_rel = str(imp_path.relative_to(REPO))
            # Classify: if importer is inside SRC_ROOT and is a barrel, count as barrel.
            is_barrel_importer = (
                imp_path.name == "__init__.py"
                and imp_path in file_info
                and file_info[imp_path]["is_barrel"]
            )
            if is_barrel_importer:
                barrel_importers.append(imp_rel)
            else:
                importers.append(imp_rel)
            if sym:
                selector_map.setdefault(sym, []).append(imp_rel)

        importers = sorted(set(importers))
        barrel_importers = sorted(set(barrel_importers))

        loc = count_loc(p)
        commits = git_commits_6mo(p)
        churn = round(commits / loc, 4) if loc else 0.0

        out_rows.append(
            {
                "file": rel,
                "module": mod,
                "loc": loc,
                "commits_6mo": commits,
                "churn_ratio": churn,
                "is_barrel": info["is_barrel"],
                "exports": extract_exports(info["tree"], info["src"]),
                "importers": importers,
                "barrel_importers": barrel_importers,
                "importer_selectors": {
                    k: sorted(set(v)) for k, v in selector_map.items()
                },
            }
        )

    payload = json.dumps(out_rows, indent=2)
    if not args.no_cache:
        cache_file.write_text(payload)
    sys.stdout.write(payload)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
