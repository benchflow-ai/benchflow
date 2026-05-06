"""LAB → BenchFlow adapter CLI.

Translates Harvey AI's Legal Agent Bench (`harveyai/harvey-labs`) into
BenchFlow's task format and (optionally) drives a benchflow run over the
generated tasks.

Usage
-----

    # 1. Materialise tasks (clones harvey-labs into .ref/lab/ if needed)
    python benchmarks/lab/benchflow.py translate \\
        --output-dir /tmp/lab-tasks

    # Subset:
    python benchmarks/lab/benchflow.py translate \\
        --output-dir /tmp/lab-tasks \\
        --task-list benchmarks/lab/scripts/parity_subset.txt

    # 2. Run benchflow over the generated tasks
    GEMINI_API_KEY=... bench run /tmp/lab-tasks/<task-id>/ \\
        --agent gemini --model gemini-3.1-flash-lite-preview --backend docker

    # 3. Validate adapter scaffolding (no run)
    python benchmarks/lab/benchflow.py check /tmp/lab-tasks/<task-id>/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Make the sibling adapter package importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter.translate import (
    LabTask,
    discover_tasks,
    write_task,
)

LOG = logging.getLogger("lab-adapter")

LAB_REPO = "https://github.com/harveyai/harvey-labs.git"
LAB_REF = "main"


# ── Source repo materialisation ───────────────────────────────────────


def ensure_lab_repo(ref_dir: Path, *, ref: str = LAB_REF) -> Path:
    """Clone harveyai/harvey-labs under ``ref_dir`` if not present."""
    if (ref_dir / "tasks").is_dir() and any((ref_dir / "tasks").iterdir()):
        return ref_dir

    LOG.info("Cloning %s @ %s into %s", LAB_REPO, ref, ref_dir)
    ref_dir.parent.mkdir(parents=True, exist_ok=True)
    if ref_dir.exists():
        shutil.rmtree(ref_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, LAB_REPO, str(ref_dir)],
        check=True,
    )
    return ref_dir


# ── Translation ──────────────────────────────────────────────────────


def _filter_tasks(tasks: list[LabTask], task_list: Path | None) -> list[LabTask]:
    if task_list is None:
        return tasks
    wanted = {
        line.strip()
        for line in task_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    selected = [t for t in tasks if t.relative_id in wanted or t.task_id in wanted]
    missing = wanted - {t.relative_id for t in selected} - {t.task_id for t in selected}
    if missing:
        LOG.warning("task list referenced unknown tasks: %s", sorted(missing))
    return selected


def cmd_translate(args: argparse.Namespace) -> int:
    lab_root = ensure_lab_repo(Path(args.lab_dir), ref=args.lab_ref)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(lab_root)
    selected = _filter_tasks(tasks, Path(args.task_list) if args.task_list else None)
    if args.limit:
        selected = selected[: args.limit]

    LOG.info("Translating %d / %d LAB tasks → %s",
             len(selected), len(tasks), out_dir)

    written: list[str] = []
    for t in selected:
        path = write_task(t, out_dir, force=args.force)
        written.append(t.task_id)
        if args.verbose:
            print(f"  ✓ {t.relative_id} → {path}")
    print(f"Wrote {len(written)} BenchFlow task(s) to {out_dir}")
    return 0


# ── Lightweight scaffolding sanity check ─────────────────────────────


def cmd_check(args: argparse.Namespace) -> int:
    """Validate that a translated task directory has the expected shape.

    This duplicates a tiny subset of `bench tasks check`, kept here so
    parity reviewers can re-validate without installing benchflow."""
    target = Path(args.task_dir)
    required = [
        target / "task.toml",
        target / "instruction.md",
        target / "environment" / "Dockerfile",
        target / "tests" / "test.sh",
        target / "tests" / "rubric_judge.py",
        target / "tests" / "criteria.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("MISSING:", *(str(p) for p in missing), sep="\n  ")
        return 1
    print(f"OK  {target}")
    return 0


# ── Inventory ────────────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> int:
    lab_root = ensure_lab_repo(Path(args.lab_dir), ref=args.lab_ref)
    tasks = discover_tasks(lab_root)
    rows = []
    for t in tasks:
        cfg = t.config
        rows.append({
            "task_id": t.task_id,
            "relative_id": t.relative_id,
            "title": cfg.get("title", ""),
            "work_type": cfg.get("work_type", ""),
            "n_criteria": len(cfg.get("criteria", [])),
            "n_deliverables": len(cfg.get("deliverables", {})),
        })
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['task_id']:<80} {r['n_criteria']:>3}c {r['n_deliverables']:>2}d  {r['title']}")
    return 0


# ── Argparse plumbing ────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab-adapter", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lab-dir", default=".ref/lab",
                        help="Where to clone harvey-labs (default: .ref/lab)")
    common.add_argument("--lab-ref", default=LAB_REF,
                        help="Git ref of harvey-labs to translate from")

    t = sub.add_parser("translate", parents=[common],
                       help="Materialise LAB tasks as BenchFlow tasks")
    t.add_argument("--output-dir", required=True)
    t.add_argument("--task-list", default=None,
                   help="Optional file with one task id per line (LAB or sanitised)")
    t.add_argument("--limit", type=int, default=0,
                   help="Stop after this many tasks (0 = all)")
    t.add_argument("--force", action="store_true",
                   help="Overwrite existing target directories")
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=cmd_translate)

    c = sub.add_parser("check", help="Validate a translated task dir")
    c.add_argument("task_dir")
    c.set_defaults(func=cmd_check)

    li = sub.add_parser("list", parents=[common],
                        help="List all LAB tasks with task counts")
    li.add_argument("--json", action="store_true")
    li.set_defaults(func=cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LAB_LOG", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
