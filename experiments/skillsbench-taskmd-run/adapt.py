#!/usr/bin/env python3
"""Fixed adaptation: SkillsBench legacy tasks -> native task.md packages.

Wraps the in-repo `migrate_task_to_task_md` over a set of tasks. Each task's
legacy split layout (task.toml + instruction.md + solution/ + tests/) becomes
the native layout (task.md + oracle/ + verifier/ + environment/), then is
structurally validated. Deterministic — no network, no sandbox, no model.

    python adapt.py --skillsbench /path/to/skillsbench --out ./adapted \
        --tasks-file simple_tasks.txt
    python adapt.py --skillsbench /path/to/skillsbench --out ./adapted --all
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from benchflow._utils.task_authoring import check_task, migrate_task_to_task_md


def _find_src(skillsbench: Path, name: str) -> Path | None:
    for sub in ("tasks", "tasks-extra"):
        p = skillsbench / sub / name
        if (p / "task.toml").exists():
            return p
    return None


def _all_task_names(skillsbench: Path) -> list[str]:
    names: list[str] = []
    for sub in ("tasks", "tasks-extra"):
        d = skillsbench / sub
        if d.is_dir():
            names += [p.name for p in sorted(d.iterdir()) if (p / "task.toml").exists()]
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skillsbench", required=True, type=Path, help="path to a skillsbench checkout")
    ap.add_argument("--out", required=True, type=Path, help="output dir for adapted task.md packages")
    ap.add_argument("--tasks-file", type=Path, help="newline-delimited task names (# comments ok)")
    ap.add_argument("--tasks", nargs="*", default=[], help="task names")
    ap.add_argument("--all", action="store_true", help="adapt every task in the checkout")
    args = ap.parse_args()

    names = list(args.tasks)
    if args.tasks_file:
        names += [
            ln.strip()
            for ln in args.tasks_file.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    if args.all:
        names += _all_task_names(args.skillsbench)
    names = sorted(set(names))
    if not names:
        ap.error("no tasks given (use --tasks, --tasks-file, or --all)")

    args.out.mkdir(parents=True, exist_ok=True)
    ok = warn = miss = 0
    for name in names:
        src = _find_src(args.skillsbench, name)
        if src is None:
            print(f"SKIP {name}: not found in {args.skillsbench}/tasks[-extra]")
            miss += 1
            continue
        dst = args.out / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        try:
            migrate_task_to_task_md(dst, overwrite=True, remove_legacy=True)
        except Exception as exc:
            print(f"FAIL {name}: migrate error: {exc}")
            warn += 1
            continue
        issues = check_task(dst)  # structural validation
        if issues:
            print(f"WARN {name}: {issues}")
            warn += 1
        else:
            print(f"OK   {name}")
            ok += 1

    print(f"\nadapted {ok} ok, {warn} with issues, {miss} missing -> {args.out}")
    return 1 if warn else 0


if __name__ == "__main__":
    sys.exit(main())
