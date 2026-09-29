"""Run a split with a surface version as ordinary BenchFlow jobs, and read it back.

An evaluation of one surface version is a folder ``evals/<id>/`` with one
folder per split and one BenchFlow job per trial::

    evals/r01-c1/train/trial-01/job/<task>__<id>/result.json
    evals/r01-c1/train/trial-02/job/...
    evals/r01-c1/test/trial-01/job/...

Each ``trial-NN`` is an :class:`benchflow.Evaluation` job over the split's
tasks, so ``bench eval view`` opens it and ``bench eval metrics
evals/r01-c1/train`` pools the trials (pass@k included). Results are read
back with :func:`benchflow.load_job`, one trial per task per folder.

A trial that has no reward is an *infrastructure error*: the agent's
integration broke, the provider failed, the verifier crashed, the sandbox did
not start, or the trial is missing (cancelled by a budget stop, or crashed
before writing ``result.json``). It is left out of the score and counted
(:attr:`TrialRecord.infra`), the rule ``bf.load_job`` applies to unscored
trials.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow.hillclimbing.surface import SurfaceSpec, deploy_settings

logger = logging.getLogger(__name__)

SplitName = Literal["train", "test"]
JOB_NAME = "job"
MISSING = "missing"


class TaskSetError(ValueError):
    """The tasks given to ``bench hillclimb`` cannot be used."""


@dataclass(frozen=True)
class TaskSet:
    """The tasks of a run, by name."""

    dirs: dict[str, Path]

    @classmethod
    def resolve(
        cls,
        tasks: str | Path | Sequence[str | Path],
        *,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
    ) -> TaskSet:
        """A tasks folder (its task subfolders) or a list of task folders."""
        from benchflow.evaluation import _is_task_dir
        from benchflow.task.discovery import resolve_task_collection_root

        include, exclude = set(include), set(exclude)
        found: list[Path] = []
        if isinstance(tasks, str | Path):
            root = resolve_task_collection_root(Path(tasks).expanduser())
            if not root.is_dir():
                raise TaskSetError(f"not a directory: {root}")
            if _is_task_dir(root):
                found = [root]
            else:
                found = [
                    d for d in sorted(root.iterdir()) if d.is_dir() and _is_task_dir(d)
                ]
        else:
            for item in tasks:
                path = Path(item).expanduser()
                if not path.is_dir() or not _is_task_dir(path):
                    raise TaskSetError(f"not a runnable task folder: {path}")
                found.append(path)
        dirs: dict[str, Path] = {}
        for path in found:
            name = path.name
            if include and name not in include:
                continue
            if name in exclude:
                continue
            if name in dirs:
                raise TaskSetError(
                    f"two tasks are named {name!r}: {dirs[name]} and {path}"
                )
            dirs[name] = path.resolve()
        missing = sorted(include - set(dirs))
        if missing:
            roots = {p.parent for p in found} or (
                {Path(tasks).expanduser()} if isinstance(tasks, str | Path) else set()
            )
            broken = [m for m in missing if any((r / m).is_dir() for r in roots)]
            hint = (
                f" ({', '.join(broken)} exist but are not runnable tasks: "
                "run `bench tasks check` on them)"
                if broken
                else ""
            )
            raise TaskSetError(
                f"--include names unknown tasks: {', '.join(missing)}{hint}"
            )
        if not dirs:
            raise TaskSetError(f"no runnable task found in {tasks}")
        return cls(dict(sorted(dirs.items())))

    def groups(self, names: Iterable[str]) -> dict[Path, list[str]]:
        """Task names by parent folder: one Evaluation per parent."""
        out: dict[Path, list[str]] = {}
        for name in sorted(names):
            out.setdefault(self.dirs[name].parent, []).append(name)
        return out


@dataclass(frozen=True)
class TrialRecord:
    """One trial of one task."""

    task: str
    trial: int
    reward: float | None
    passed: bool | None
    cost_usd: float | None
    category: str | None = None
    error: str | None = None
    path: str | None = None
    n_tool_calls: int = 0

    @property
    def infra(self) -> bool:
        return self.reward is None

    @property
    def failed(self) -> bool:
        """Scored and not passed: the failures the proposer reads."""
        return self.reward is not None and not self.passed


@dataclass
class SplitRun:
    """One split's trials for one surface version."""

    split: SplitName
    tasks: list[str]
    trials: int
    dir: Path
    records: list[TrialRecord] = field(default_factory=list)

    def values(self, objective: str = "score") -> dict[str, list[float]]:
        """``{task: [reward or USD per scored trial]}`` (see stats)."""
        out: dict[str, list[float]] = {t: [] for t in self.tasks}
        for r in self.records:
            if r.infra:
                continue
            if objective == "cost":
                if r.cost_usd is not None:
                    out[r.task].append(r.cost_usd)
            elif r.reward is not None:
                out[r.task].append(r.reward)
        return out

    @property
    def infra_records(self) -> list[TrialRecord]:
        return [r for r in self.records if r.infra]

    @property
    def infra_rate(self) -> float:
        return len(self.infra_records) / len(self.records) if self.records else 0.0

    def infra_categories(self) -> dict[str, int]:
        return dict(Counter(r.category or "unscored" for r in self.infra_records))

    @property
    def cost_total(self) -> float | None:
        known = [r.cost_usd for r in self.records if r.cost_usd is not None]
        return math.fsum(known) if known else None

    @property
    def usd_unknown(self) -> int:
        return sum(
            1 for r in self.records if r.cost_usd is None and r.category != MISSING
        )

    def per_task(self) -> list[dict[str, Any]]:
        out = []
        for task in self.tasks:
            recs = sorted(
                (r for r in self.records if r.task == task), key=lambda r: r.trial
            )
            rewards = [r.reward for r in recs]
            scored = [x for x in rewards if x is not None]
            costs = [r.cost_usd for r in recs if not r.infra and r.cost_usd is not None]
            out.append(
                {
                    "task": task,
                    "rewards": rewards,
                    "mean_reward": math.fsum(scored) / len(scored) if scored else None,
                    "cost_usd": math.fsum(costs) / len(costs) if costs else None,
                }
            )
        return out


@dataclass
class EvalSettings:
    """How the agent under test runs: the usual ``bench eval run`` options."""

    agent: str
    model: str | None = None
    reasoning_effort: str | None = None
    environment: str = "docker"
    concurrency: int = 4
    agent_env: dict[str, str] = field(default_factory=dict)
    sandbox_user: str | None = "agent"
    agent_idle_timeout: int | None = 600
    retry_attempts: int | None = None
    config_override: dict[str, Any] | None = None
    preflight: bool = True

    def evaluation_config(
        self,
        *,
        names: Iterable[str],
        concurrency: int,
        deploy: Mapping[str, Any] | None = None,
        agent: str | None = None,
        budget_usd: float | None = None,
    ) -> Any:
        from benchflow.budget import Budget
        from benchflow.evaluation import EvaluationConfig, RetryConfig

        deploy = dict(deploy or {})
        control = agent in ("oracle", "nop")
        retry = RetryConfig()
        if self.retry_attempts is not None:
            retry = RetryConfig(max_retries=self.retry_attempts)
        return EvaluationConfig(
            agent=agent or self.agent,
            model=None if control else self.model,
            reasoning_effort=None if control else self.reasoning_effort,
            environment=self.environment,
            concurrency=max(1, concurrency),
            agent_env={} if control else dict(self.agent_env),
            retry=retry,
            sandbox_user=self.sandbox_user,
            agent_idle_timeout=self.agent_idle_timeout,
            include_tasks=set(names),
            skills_dir=deploy.get("skills_dir"),
            skill_mode=deploy.get("skill_mode", "no-skill"),
            config_override=deploy.get("config_override", self.config_override),
            budget=Budget(max_cost_usd=budget_usd)
            if budget_usd is not None and budget_usd > 0
            else None,
        )


def _job_name(index: int, parent: Path, n_groups: int) -> str:
    return JOB_NAME if n_groups == 1 else f"{JOB_NAME}-{index}-{parent.name}"


async def _run_job(
    tasks_dir: Path, jobs_dir: Path, job_name: str, config: Any, preflight: bool
) -> None:
    from benchflow.evaluation import Evaluation

    evaluation = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=jobs_dir,
        config=config,
        job_name=job_name,
        preflight=preflight,
    )
    await evaluation.run()


def _unscored_reason(trial: Any) -> tuple[str, str]:
    result = trial.result
    if trial.integration_failure is not None:
        cause = (trial.integration_failure or {}).get("cause") or "integration"
        return "agent_integration", f"agent integration failed ({cause})"
    if result.verifier_error:
        return (
            result.verifier_error_category or "verifier_error",
            str(result.verifier_error),
        )
    if result.error:
        return result.error_category or "agent_error", str(result.error)
    return "unscored", "the trial has no reward"


def record_from_trial(trial: Any, index: int) -> TrialRecord:
    """A :class:`benchflow.Trial` as a :class:`TrialRecord`."""
    scored = trial.assessment == "scored"
    category = error = None
    if not scored:
        category, error = _unscored_reason(trial)
    return TrialRecord(
        task=trial.task_name,
        trial=index,
        reward=trial.reward if scored else None,
        passed=bool(trial.passed) if scored else None,
        cost_usd=trial.cost_usd,
        category=category,
        error=(error or "")[:500] or None,
        path=str(trial.path),
        n_tool_calls=int(trial.result.n_tool_calls or 0),
    )


def collect_trial_folder(
    folder: Path, names: Sequence[str], index: int, *, controls: bool = False
) -> list[TrialRecord]:
    """One ``trial-NN`` folder's records, one per task (missing ones included)."""
    import benchflow as bf

    by_task: dict[str, Any] = {}
    if folder.is_dir():
        try:
            job = bf.load_job(folder)
        except FileNotFoundError:
            job = None
        if job is not None:
            for trial in job.trials if controls else job.agents():
                if trial.task_name in names:
                    by_task[trial.task_name] = trial
    records = []
    for name in names:
        trial = by_task.get(name)
        if trial is None:
            records.append(
                TrialRecord(
                    task=name,
                    trial=index,
                    reward=None,
                    passed=None,
                    cost_usd=None,
                    category=MISSING,
                    error="no result.json: cancelled (budget), or crashed before it was written",
                )
            )
        else:
            records.append(record_from_trial(trial, index))
    return records


def trial_folder(split_dir: Path, index: int) -> Path:
    return split_dir / f"trial-{index:02d}"


async def evaluate_version(
    taskset: TaskSet,
    splits: Mapping[SplitName, Sequence[str]],
    *,
    version_dir: Path,
    specs: Sequence[SurfaceSpec],
    settings: EvalSettings,
    out_dir: Path,
    trials: int,
    budget_usd: float | None = None,
) -> dict[SplitName, SplitRun]:
    """Run every split's tasks ``trials`` times with one surface version.

    All the (split, trial, parent folder) jobs run at once, sharing
    ``settings.concurrency`` between them (at least one rollout each). Two
    splits never share a task, and every job runs the same surface, so the
    runs cannot disturb each other's images. Two *different* versions must
    not run at once on Docker: a skills surface is baked into the task's
    image (``bf__<task>``), and a second build could retag it between the
    first build and its container start.
    """
    deploy = deploy_settings(version_dir, specs, settings.config_override)
    jobs: list[tuple[Path, Path, str, list[str]]] = []
    for split, names in splits.items():
        groups = taskset.groups(names)
        for k in range(1, trials + 1):
            for i, (parent, group) in enumerate(sorted(groups.items())):
                jobs.append(
                    (
                        parent,
                        trial_folder(out_dir / split, k),
                        _job_name(i, parent, len(groups)),
                        group,
                    )
                )
    per_job = max(1, round(settings.concurrency / max(1, len(jobs))))
    # Each job gets the whole remaining budget as its hard cap: splitting it
    # evenly would cut jobs whose tasks cost more than average, and the climb
    # checks the budget between phases anyway.
    budget_each = budget_usd
    await asyncio.gather(
        *(
            _run_job(
                parent,
                jobs_dir,
                job_name,
                settings.evaluation_config(
                    names=group,
                    concurrency=per_job,
                    deploy=deploy,
                    budget_usd=budget_each,
                ),
                settings.preflight,
            )
            for parent, jobs_dir, job_name, group in jobs
        )
    )
    runs: dict[SplitName, SplitRun] = {}
    for split, names in splits.items():
        run = SplitRun(split, sorted(names), trials, out_dir / split)
        for k in range(1, trials + 1):
            run.records.extend(
                collect_trial_folder(trial_folder(run.dir, k), run.tasks, k)
            )
        runs[split] = run
    return runs


# ---------------------------------------------------------------------------
# Controls: the oracle and a do-nothing run on every task
# ---------------------------------------------------------------------------


@dataclass
class ControlResult:
    task: str
    oracle: TrialRecord | None
    nop: TrialRecord | None
    flags: list[str]


def has_oracle(task_dir: Path) -> bool:
    try:
        from benchflow.task import Task

        return Task(task_dir).paths.solve_path.exists()
    except Exception:
        return False


def control_flags(oracle: TrialRecord | None, nop: TrialRecord | None) -> list[str]:
    """Grader-bug flags for one task: the oracle must pass, doing nothing must not."""
    flags = []
    if oracle is None:
        flags.append("no_oracle")
    elif oracle.infra:
        flags.append("oracle_unscored")
    elif not oracle.passed:
        flags.append("oracle_fails")
    if nop is not None:
        if nop.infra:
            flags.append("nop_unscored")
        elif nop.passed:
            flags.append("nop_passes")
    return flags


GRADER_BUG_FLAGS = frozenset({"oracle_fails", "oracle_unscored", "nop_passes"})


async def run_controls(
    taskset: TaskSet,
    names: Sequence[str],
    *,
    settings: EvalSettings,
    out_dir: Path,
) -> list[ControlResult]:
    """Run the task's own solution (oracle) and an empty agent (nop) once per task."""
    oracle_names = [n for n in names if has_oracle(taskset.dirs[n])]
    runs: list[tuple[str, list[str]]] = [("nop", list(names))]
    if oracle_names:
        runs.append(("oracle", oracle_names))
    jobs = []
    for agent, group_names in runs:
        groups = taskset.groups(group_names)
        for i, (parent, group) in enumerate(sorted(groups.items())):
            jobs.append((agent, parent, group, _job_name(i, parent, len(groups))))
    per_job = max(1, round(settings.concurrency / max(1, len(jobs))))
    await asyncio.gather(
        *(
            _run_job(
                parent,
                out_dir / agent,
                job_name,
                settings.evaluation_config(
                    names=group, concurrency=per_job, agent=agent
                ),
                settings.preflight,
            )
            for agent, parent, group, job_name in jobs
        )
    )
    nop = {
        r.task: r
        for r in collect_trial_folder(out_dir / "nop", list(names), 1, controls=True)
    }
    oracle = (
        {
            r.task: r
            for r in collect_trial_folder(
                out_dir / "oracle", oracle_names, 1, controls=True
            )
        }
        if oracle_names
        else {}
    )
    return [
        ControlResult(
            task=name,
            oracle=oracle.get(name),
            nop=nop.get(name),
            flags=control_flags(oracle.get(name), nop.get(name)),
        )
        for name in names
    ]
