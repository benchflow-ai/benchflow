#!/usr/bin/env python3
"""split_costs.py <file> --half-a=names --half-b=names
Emit {shared_types, shared_helpers, importer_payoff, importers} JSON for a proposed 2-way split."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.py"
REPO = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)


def collect_def_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def names_used_by(node: ast.AST) -> set[str]:
    used: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            used.add(n.value.id)
    return used


def symbol_nodes(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def is_type_only(node: ast.AST) -> bool:
    """Heuristic: ClassDef inheriting from Protocol/TypedDict/NamedTuple or decorated @dataclass frozen; or a bare TypeAlias."""
    if isinstance(node, ast.ClassDef):
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name in {
                "Protocol",
                "TypedDict",
                "NamedTuple",
                "Enum",
                "IntEnum",
                "StrEnum",
            }:
                return True
    return bool(isinstance(node, ast.AnnAssign) and node.value is None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--half-a", required=True, help="comma-separated symbol names")
    parser.add_argument("--half-b", required=True)
    args = parser.parse_args()

    target = Path(args.file)
    src = target.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)

    all_names = collect_def_names(tree)
    half_a = set(x.strip() for x in args.half_a.split(",") if x.strip())
    half_b = set(x.strip() for x in args.half_b.split(",") if x.strip())

    missing = (half_a | half_b) - all_names
    if missing:
        sys.stderr.write(
            f"warning: names not defined at module scope: {sorted(missing)}\n"
        )
    overlap = half_a & half_b
    if overlap:
        sys.stderr.write(f"warning: names in both halves: {sorted(overlap)}\n")

    rest = all_names - half_a - half_b
    nodes = symbol_nodes(tree)

    used_by_a: set[str] = set()
    used_by_b: set[str] = set()
    for name in half_a:
        if name in nodes:
            used_by_a |= names_used_by(nodes[name])
    for name in half_b:
        if name in nodes:
            used_by_b |= names_used_by(nodes[name])

    # Anything in `rest` that both halves reference.
    both = (used_by_a & used_by_b) & rest
    shared_types: list[str] = []
    shared_helpers: list[str] = []
    for name in sorted(both):
        node = nodes.get(name)
        if node is not None and is_type_only(node):
            shared_types.append(name)
        else:
            shared_helpers.append(name)

    # Importer payoff: distinct importer sets per half.
    try:
        man_out = subprocess.run(
            [sys.executable, str(MANIFEST)], capture_output=True, text=True, check=True
        )
        rows = json.loads(man_out.stdout)
        file_rel = str(target.resolve().relative_to(REPO))
        row = next((r for r in rows if r["file"] == file_rel), None)
    except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        row = None

    selectors = row.get("importer_selectors", {}) if row else {}
    importers_count = len(row.get("importers", [])) if row else 0
    a_importers: set[str] = set()
    b_importers: set[str] = set()
    for sym, imps in selectors.items():
        if sym in half_a:
            a_importers |= set(imps)
        if sym in half_b:
            b_importers |= set(imps)

    only_a = a_importers - b_importers
    only_b = b_importers - a_importers
    total = len(a_importers | b_importers)
    disjoint_ratio = (len(only_a) + len(only_b)) / total if total else 0.0

    if importers_count <= 1:
        payoff = "none"
    elif disjoint_ratio >= 0.6 and total >= 4:
        payoff = "high"
    elif importers_count >= 2:
        payoff = "medium"
    else:
        payoff = "none"

    result = {
        "file": str(target),
        "shared_types": shared_types,
        "shared_helpers": shared_helpers,
        "importer_payoff": payoff,
        "importers": importers_count,
        "half_a_importers": sorted(a_importers),
        "half_b_importers": sorted(b_importers),
        "disjoint_ratio": round(disjoint_ratio, 2),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
