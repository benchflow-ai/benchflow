"""``bench tasks check`` for a task.md draft 2 package.

It prints two reports, one after the other:

1. **The reference checker's**: exactly what ``python3 tools/taskmd.py check``
   prints, from the vendored reference tool, including what grading needs and
   its worst case. For a family it runs the generator once on a sample seed,
   offline and confined, where this machine can.
2. **BenchFlow's**: every field BenchFlow refuses (each fails the check), the
   fields it refuses only when an agent runs, and how many fields it honors
   and records, each with how. When nothing is refused, the task is
   materialized (a family at the reference checker's sample seed) and the
   native package goes through BenchFlow's own checks.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from benchflow.taskmd._vendor import taskmd as ref
from benchflow.taskmd.materialize import FORMAT_LABEL, TaskMdError, TaskMdFormat
from benchflow.taskmd.plan import plan_package
from benchflow.taskmd.reference import errors, parse_package, reference_check


@dataclass
class CheckReport:
    ok: bool
    reference: list[str]
    benchflow: list[str]
    native_dir: Path | None = None
    notes: list[str] = field(default_factory=list)


def check_package(task_dir: Path) -> CheckReport:
    """Both reports for one package; ``native_dir`` is the materialized package when there is one."""
    task_dir = Path(task_dir).resolve()
    reference = reference_check(task_dir)
    lines: list[str] = []
    document = parse_package(task_dir)
    if errors(document):
        lines.append(
            f"not planned: the reference parser reports errors, so BenchFlow will not run this {FORMAT_LABEL} package"
        )
        return CheckReport(False, list(reference.lines), lines)
    plan = plan_package(document, task_dir)
    for finding in plan.refused:
        lines.append(f"refused: {finding}")
    for finding in plan.agent_refused:
        lines.append(
            f"refused when an agent runs (the oracle, controls, and --agent nop run): {finding}"
        )
    lines += [f"honored: {f}" for f in plan.honored]
    lines += [f"recorded: {f}" for f in plan.recorded]
    ok = reference.ok and plan.ok
    native_dir = None
    if plan.ok:
        from benchflow.task.formats import task_format_cache_root

        fmt = TaskMdFormat()
        out_root = task_format_cache_root() / fmt.name
        try:
            if plan.family is not None:
                seed = ref.sample_seed(document.config)
                if shutil.which("docker") is None:
                    lines.append(
                        f"warning: family@1 builds an instance in a container of the task's image, and this machine has "
                        f"no docker command, so seed {seed}'s native package was not built or checked"
                    )
                else:
                    native_dir = fmt.materialize_variant(task_dir, out_root, seed=seed)
                    lines.append(
                        f"materialized seed {seed}, the reference checker's sample seed"
                    )
            else:
                native_dir = fmt.materialize(task_dir, out_root)
        except TaskMdError as exc:
            ok = False
            lines += [f"refused: {r}" for r in exc.reasons]
    return CheckReport(ok, list(reference.lines), lines, native_dir)
