"""Stage tasks authored with a repo-root build context into benchflow's.

Some benchmarks — notably smolclaws/env0 — author per-task Dockerfiles whose
``COPY`` paths are relative to the *repository root* (``COPY tasks/<name>/data
…``), because their own build pipeline uses the repo root as the docker build
context. benchflow builds each task with that task's ``environment/`` directory
as the context, so those COPYs fail (``path does not exist:
environment/tasks/<name>/data``).

This adapter produces benchflow-native copies: for each task it copies the
referenced repo-root paths into the task's ``environment/`` and rewrites the
``COPY tasks/<name>/<sub>`` lines to ``COPY <sub>``. Task content is otherwise
byte-for-byte unchanged. Stage once, then point ``--tasks-dir`` at the output::

    python -m benchflow._utils.build_context_stage <src_tasks_dir> <out_dir>
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def stage_task(task_dir: Path, out_task_dir: Path) -> bool:
    """Copy one task to ``out_task_dir``, staging its repo-root ``COPY`` paths.

    Returns ``True`` if the Dockerfile had repo-root COPYs that were rewritten.
    """
    shutil.copytree(task_dir, out_task_dir, ignore=_IGNORE)
    dockerfile = out_task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return False

    # ``COPY tasks/<this-task>/<sub> <dest>`` -> ``COPY <sub> <dest>`` and copy
    # ``<task_dir>/<sub>`` into ``environment/<sub>`` so it is in the context.
    pattern = re.compile(
        r"^(\s*COPY\s+)tasks/" + re.escape(task_dir.name) + r"/(\S+)(\s.*)$"
    )
    rewritten = False
    lines: list[str] = []
    for line in dockerfile.read_text().splitlines():
        match = pattern.match(line)
        if not match:
            lines.append(line)
            continue
        sub = match.group(2)
        src = task_dir / sub
        if src.exists():
            dst = out_task_dir / "environment" / sub
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_IGNORE)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        lines.append(f"{match.group(1)}{sub}{match.group(3)}")
        rewritten = True

    dockerfile.write_text("\n".join(lines) + "\n")
    return rewritten


def stage_tasks(src_tasks_dir: str | Path, out_dir: str | Path) -> list[str]:
    """Stage every task subdir under ``src_tasks_dir`` into ``out_dir`` (cleared).

    A subdir is a task when it holds ``environment/Dockerfile``. Dotfile- and
    underscore-prefixed dirs (e.g. ``_manifests``) are skipped. Returns the
    staged task names.
    """
    src = Path(src_tasks_dir)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    staged: list[str] = []
    for task in sorted(src.iterdir()):
        if not task.is_dir() or task.name.startswith((".", "_")):
            continue
        if not (task / "environment" / "Dockerfile").exists():
            continue
        stage_task(task, out / task.name)
        staged.append(task.name)
    return staged


def main(argv: list[str] | None = None) -> int:
    import sys

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(
            "usage: python -m benchflow._utils.build_context_stage "
            "<src_tasks_dir> <out_dir>",
            file=sys.stderr,
        )
        return 2
    names = stage_tasks(args[0], args[1])
    print(f"staged {len(names)} task(s) -> {args[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
