"""Job-level views: the Outcomes grid, the Pareto chart and the training view.

:func:`build_outcomes` reads one or more jobs through :func:`benchflow.load_job`
and precomputes everything those views draw, so the page only regroups trial
indices and draws marks. The document it returns (``benchflow.outcomes/1``)
is column-oriented: one array per trial field, and one ``values``/``codes``
pair per grouping dimension.

What each trial carries, and where it comes from:

- ``reward`` and ``outcome``: the reward of a scored trial (partial credit
  kept); an unscored trial has no reward, never 0 (``Trial.assessment``).
- ``cause``: why an unscored or errored trial has no score, with its fault
  (task, agent, infrastructure, setup) from :func:`benchflow.failures.cause_of`.
- ``attempts``: every rollout of a retried trial (``Trial.attempts``).
- ``integrity``: the BenchShield verdict (``Trial.integrity``) when the trial
  was audited.
- cost, tokens, sandbox seconds and wall time summed over the trial's
  attempts, so a retried trial costs what its retries cost.

Dimensions a trial is grouped by: task, dataset, all, agent, model, harness
(``harness_mode``: acp or native), seed (a recorded ``seed``, else the
``trial-NN`` repeat folder), step (a recorded ``policy_version``/``step``/
``checkpoint``, a ``step`` in the job's ``rollouts.jsonl``, or a hill-climb
round), split (a hill-climb ``split``, a recorded ``split``, else a
``train``/``test`` folder in the trial's or its task's path), job and
outcome. :data:`SOURCES` names where each came from in a given document.

Statistics (pass@k, solve rate intervals) come from
:mod:`benchflow.pass_at_k`; Pareto intervals are bootstrap intervals that
resample tasks (a deterministic seed per point).
"""

from __future__ import annotations

import json
import math
import operator
import random
import re
import time
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.jobs import Trial

SCHEMA = "benchflow.outcomes/1"

DIMENSIONS: dict[str, str] = {
    "task": "task",
    "dataset": "dataset",
    "all": "all trials",
    "agent": "agent",
    "model": "model",
    "harness": "harness",
    "seed": "seed or repeat",
    "step": "step or checkpoint",
    "split": "split",
    "job": "job",
    "outcome": "outcome",
}
OUTCOMES = ("passed", "partial credit", "failed", "unscored")
EXECUTIONS = ("completed", "errored", "timed_out", "integration_failed")
ROLES = ("agent", "control", "optimizer")
# "other" is a verdict string this version does not know (never shown as Rejected).
VERDICTS = ("Checked", "VectorExposed", "AgentViolation", "Rejected", "other")
X_METRICS = {
    "usd": "cost (USD)",
    "tokens": "tokens",
    "sandbox_sec": "sandbox seconds",
    "wall_sec": "wall time (s)",
}
PARETO_GROUPS = ("model", "agent", "harness", "step", "job", "seed", "split")
PARETO_PARTITIONS = ("dataset", "split", "none")
UNRECORDED = "(unrecorded)"
BOOTSTRAP_SAMPLES = 200
# Pareto resamples shrink toward this floor when the job is large (_pareto).
MIN_BOOTSTRAP_SAMPLES = 50
BOOTSTRAP_BUDGET = 12_000_000
_MAX_EXTRA_REWARDS = 6
_DETAIL_CHARS = 240
_SPLIT_WORDS = {
    "train",
    "test",
    "val",
    "valid",
    "validation",
    "dev",
    "heldout",
    "held-out",
    "holdout",
}
_GENERIC_DIRS = _SPLIT_WORDS | {"tasks", "task"}
_STEP_KEYS = ("policy_version", "policy_step", "step", "global_step", "checkpoint")
_REPEAT = re.compile(r"^(?:trial|seed|repeat|sample)-?\d+$")


@dataclass
class _Context:
    """Per-trial facts read from files around the trial (cached per folder)."""

    hillclimb: dict[Path, dict[str, Any] | None] = field(default_factory=dict)
    rollout_steps: dict[Path, dict[str, Any] | None] = field(default_factory=dict)
    tasks_dirs: dict[Path, str | None] = field(default_factory=dict)
    sources: dict[str, set[str]] = field(default_factory=dict)

    def note(self, dim: str, source: str) -> None:
        self.sources.setdefault(dim, set()).add(source)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _hillclimb_at(ctx: _Context, folder: Path) -> dict[str, Any] | None:
    if folder not in ctx.hillclimb:
        data = _read_json(folder / "hillclimb.json")
        if isinstance(data, dict) and data.get("kind") == "hillclimb-demo":
            split = data.get("split") if isinstance(data.get("split"), dict) else {}
            task_split = {
                str(task): name
                for name in ("train", "test")
                for task in split.get(name) or []
            }
            order = ["baseline"] + [
                str((r.get("candidate") or {}).get("id"))
                for r in data.get("rounds") or []
                if isinstance(r, dict) and (r.get("candidate") or {}).get("id")
            ]
            best = data.get("best") if isinstance(data.get("best"), dict) else {}
            stop = data.get("stop") if isinstance(data.get("stop"), dict) else {}
            ctx.hillclimb[folder] = {
                "task_split": task_split,
                "order": order,
                "best_version": best.get("version"),
                "stop": stop.get("reason"),
            }
        else:
            ctx.hillclimb[folder] = None
    return ctx.hillclimb[folder]


def _rollout_steps_at(ctx: _Context, folder: Path) -> dict[str, Any] | None:
    """``step`` per rollout folder name from a ``rollouts.jsonl`` audit log
    (the RL adapters write one per jobs folder); None without one."""
    if folder not in ctx.rollout_steps:
        steps: dict[str, Any] = {}
        try:
            lines = (folder / "rollouts.jsonl").read_text().splitlines()
        except OSError:
            lines = None
        for line in lines or []:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("step") is None:
                continue
            rollout_dir = row.get("rollout_dir")
            if isinstance(rollout_dir, str) and rollout_dir:
                steps[Path(rollout_dir).name] = row["step"]
        ctx.rollout_steps[folder] = steps if lines is not None else None
    return ctx.rollout_steps[folder]


def _ancestors(path: Path, root: Path) -> list[Path]:
    """``path``'s parents from nearest to ``root`` inclusive, then up to six
    folders above ``root`` (a job served from inside a hill-climb run, as
    deep as ``evals/<version>/<split>/trial-NN/job``)."""
    out = []
    for parent in path.parents:
        out.append(parent)
        if parent == root:
            break
    else:
        return out
    out.extend(list(root.parents)[:6])
    return out


def _named_folder(folder: str) -> str | None:
    """A dataset name from the folder holding tasks: the nearer of its last
    two folder names that is not tasks/train/test and the like.

    A home folder prefix (``$HOME``, ``/home/<user>``, ``/Users/<user>``,
    ``C:\\Users\\<user>``) is removed first, so a user name is never taken
    for a dataset: ``~/tasks/<task>`` has none.
    """
    pure = PureWindowsPath(folder) if "\\" in folder else PurePosixPath(folder)
    parts = [p for p in pure.parts if p != pure.anchor]
    home = [p for p in PurePosixPath(str(Path.home())).parts if p != "/"]
    if home and parts[: len(home)] == home:
        parts = parts[len(home) :]
    elif len(parts) >= 2 and parts[0] in ("home", "Users"):
        parts = parts[2:]
    for part in reversed(parts[-2:]):
        if part not in ("~", ".", "..") and part.lower() not in _GENERIC_DIRS:
            return part
    return None


def _dataset_of(trial: Trial, ctx: _Context) -> tuple[str | None, str | None]:
    name = trial.raw.get("dataset_name") or trial.config.get("dataset_name")
    if name:
        return str(name), "recorded dataset_name"
    task_path = trial.config.get("task_path")
    if isinstance(task_path, str) and len(PurePosixPath(task_path).parts) > 1:
        folder = _named_folder(
            str(PureWindowsPath(task_path).parent)
            if "\\" in task_path
            else str(PurePosixPath(task_path).parent)
        )
        if folder:
            return folder, "the folder holding the task"
    job_dir = trial.path.parent
    if job_dir not in ctx.tasks_dirs:
        evaluation = _read_json(job_dir / "evaluation.json")
        tasks_dir = (
            evaluation.get("tasks_dir") if isinstance(evaluation, dict) else None
        )
        ctx.tasks_dirs[job_dir] = (
            _named_folder(tasks_dir) if isinstance(tasks_dir, str) else None
        )
    if ctx.tasks_dirs[job_dir]:
        return ctx.tasks_dirs[job_dir], "the job's tasks_dir (evaluation.json)"
    return None, None


def _recorded(trial: Trial, keys: Sequence[str]) -> Any:
    for key in keys:
        for source in (trial.raw, trial.config):
            value = source.get(key)
            if value is not None and value != "":
                return value
    return None


def _label(value: Any) -> str:
    return UNRECORDED if value is None else str(value)


def _sum_known(values: Iterable[float | int | None]) -> float | None:
    known = [
        float(v) for v in values if isinstance(v, int | float) and math.isfinite(v)
    ]
    return math.fsum(known) if known else None


def _sandbox_seconds(trial: Trial) -> float | None:
    total = trial.timing.get("total") if isinstance(trial.timing, dict) else None
    seconds = float(total) if isinstance(total, int | float) else trial.duration_sec
    if seconds is None:
        return None
    for fork in trial.forks:
        cost = fork.cost or {}
        extra = cost.get("children_sandbox_seconds")
        if isinstance(extra, int | float):
            seconds += float(extra)
    return seconds


def _outcome_code(trial: Trial) -> tuple[int, float | None]:
    sample = trial.solve_sample()
    reward = sample.reward
    if reward is None:
        return 3, None
    # The pass rule of benchflow.pass_at_k: the review gate's verdict, else reward == 1.
    passed = sample.passed if sample.passed is not None else reward == 1
    if passed:
        return 0, reward
    return (1 if reward > 0 else 2), reward


def _first_line(text: Any) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    line = text.strip().splitlines()[0]
    return line if len(line) <= _DETAIL_CHARS else line[: _DETAIL_CHARS - 1] + "…"


def _cause(trial: Trial) -> dict[str, str] | None:
    """Why the trial is unscored or errored; None for a clean scored run."""
    from benchflow.failures import FAULT_WORDS, cause_of

    failure = trial.integration_failure
    if failure is not None:
        return {
            "key": "integration_failure",
            "label": f"the agent's integration broke ({failure.get('cause') or 'unknown'}); reward withheld",
            "fault": "agent",
            "fault_words": FAULT_WORDS.get("agent", "agent problem"),
            "next_step": "",
        }
    if trial.assessment == "scored" and trial.execution == "completed":
        return None
    cause = cause_of(trial.raw)
    if cause is None:
        if trial.execution == "timed_out":
            return {
                "key": "timeout_scored",
                "label": "the agent timed out and the verifier scored what it left",
                "fault": "agent",
                "fault_words": FAULT_WORDS.get("agent", "agent problem"),
                "next_step": "",
            }
        if trial.execution == "errored":
            return {
                "key": "error_scored",
                "label": "the agent errored and the verifier scored what it left",
                "fault": "agent",
                "fault_words": FAULT_WORDS.get("agent", "agent problem"),
                "next_step": "",
            }
        return None
    return {
        "key": cause.key,
        "label": cause.label,
        "fault": cause.fault,
        "fault_words": FAULT_WORDS.get(cause.fault, cause.fault),
        "next_step": cause.next_step,
    }


@dataclass
class _Columns:
    """The per-trial arrays of the document, filled trial by trial."""

    reward: list[float | None] = field(default_factory=list)
    outcome: list[int] = field(default_factory=list)
    execution: list[int] = field(default_factory=list)
    role: list[int] = field(default_factory=list)
    attempts: list[int] = field(default_factory=list)
    integrity: list[int] = field(default_factory=list)
    usd: list[float | None] = field(default_factory=list)
    usd_estimated: list[int] = field(default_factory=list)
    tokens: list[float | None] = field(default_factory=list)
    sandbox_sec: list[float | None] = field(default_factory=list)
    wall_sec: list[float | None] = field(default_factory=list)
    cause: list[int] = field(default_factory=list)
    link: list[str | None] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    detail: list[str | None] = field(default_factory=list)
    extra_rewards: list[dict[str, float]] = field(default_factory=list)


class _Interner:
    """values/codes for one dimension."""

    def __init__(self) -> None:
        self.values: list[str] = []
        self.index: dict[str, int] = {}
        self.codes: list[int] = []

    def add(self, value: str) -> None:
        code = self.index.get(value)
        if code is None:
            code = self.index[value] = len(self.values)
            self.values.append(value)
        self.codes.append(code)


def _trial_facts(
    trial: Trial, root: Path, root_label: str, ctx: _Context
) -> dict[str, str]:
    """Every dimension's value for one trial."""
    try:
        rel = trial.path.relative_to(root)
    except ValueError:
        rel = Path(trial.path.name)
    rel_parts = rel.parts
    facts: dict[str, str] = {
        "task": trial.task_name or UNRECORDED,
        "agent": trial.agent or UNRECORDED,
        "model": trial.model or UNRECORDED,
        "all": "all trials",
    }
    mode = trial.config.get("harness_mode", "acp") if trial.config else None
    facts["harness"] = _label(mode)

    hill = None
    hill_dir = None
    for folder in _ancestors(trial.path, root):
        hill = _hillclimb_at(ctx, folder)
        if hill is not None:
            hill_dir = folder
            break
    hill_parts: tuple[str, ...] = ()
    if hill_dir is not None:
        try:
            hill_parts = trial.path.relative_to(hill_dir).parts
        except ValueError:
            hill_parts = ()

    dataset, source = _dataset_of(trial, ctx)
    if source:
        ctx.note("dataset", source)
    facts["dataset"] = _label(dataset)

    # split
    split = None
    if hill is not None and trial.task_name in hill["task_split"]:
        split = hill["task_split"][trial.task_name]
        ctx.note("split", "hillclimb.json split")
    if split is None:
        recorded = _recorded(trial, ("split",))
        if recorded is not None:
            split = str(recorded)
            ctx.note("split", "recorded split")
    if split is None:
        for part in rel_parts[:-1]:
            if part.lower() in _SPLIT_WORDS:
                split = part.lower()
                ctx.note("split", "a train/test folder in the trial's path")
                break
    if split is None:
        task_path = trial.config.get("task_path")
        # Only the folder holding the task (tasks/<tier>/train/<task>): a
        # split-like name higher up, such as ~/dev/..., is not a split.
        if isinstance(task_path, str) and len(PurePosixPath(task_path).parts) > 1:
            parent = PurePosixPath(task_path).parent.name.lower()
            if parent in _SPLIT_WORDS:
                split = parent
                ctx.note("split", "a train/test folder holding the task")
    facts["split"] = _label(split)

    # step
    step = _recorded(trial, _STEP_KEYS)
    if step is not None:
        ctx.note("step", "recorded policy_version/step/checkpoint")
    if step is None:
        for folder in _ancestors(trial.path, root)[:3]:
            steps = _rollout_steps_at(ctx, folder)
            if steps and trial.path.name in steps:
                step = steps[trial.path.name]
                ctx.note("step", "step in rollouts.jsonl")
                break
    if step is None and len(hill_parts) > 1 and hill_parts[0] == "evals":
        step = hill_parts[1]
        ctx.note("step", "hill-climb round (evals/<version>)")
    facts["step"] = _label(step)

    # seed
    seed = _recorded(trial, ("seed",))
    if seed is not None:
        ctx.note("seed", "recorded seed")
    else:
        for part in rel_parts[:-1]:
            if _REPEAT.match(part.lower()):
                seed = part
                ctx.note("seed", "a trial-NN repeat folder")
                break
    facts["seed"] = _label(seed)

    job = Path(*rel_parts[:-1]).as_posix() if len(rel_parts) > 1 else "."
    facts["job"] = f"{root_label}/{job}" if root_label else job

    role = "agent"
    if trial.control is not None:
        role = "control"
    elif hill_parts and hill_parts[0] == "proposer":
        role = "optimizer"
    facts["role"] = role
    return facts


def _stat(trials: list[int], samples: list[Any]) -> dict[str, Any]:
    """pass@k and the solve rate's interval over the given agent trials."""
    from benchflow.pass_at_k import solve_rates

    rates = solve_rates(samples[i] for i in trials)
    rewards = [samples[i].reward for i in trials if samples[i].reward is not None]
    return {
        "trials": len(trials),
        "tasks": rates.tasks,
        "scored": rates.trials,
        "unscored": rates.unscored,
        "solve_rate": rates.solve_rate,
        "interval": list(rates.interval) if rates.interval else None,
        "method": rates.interval_method,
        "mean_reward": math.fsum(rewards) / len(rewards) if rewards else None,
        "at_k": [
            [p.k, p.pass_at_k, p.pass_hat_k, p.tasks, p.tasks_short] for p in rates.at_k
        ],
    }


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the points no other point dominates (lower or equal x and
    higher or equal y, better in one), ordered by x.

    >>> pareto_frontier([(1, 0.2), (2, 0.5), (3, 0.4), (2, 0.5), (0.5, 0.1)])
    [4, 0, 1, 3]
    """
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    frontier: list[int] = []
    best_y = -math.inf
    last: tuple[float, float] | None = None
    for i in order:
        x, y = points[i]
        if y > best_y:
            frontier.append(i)
            best_y = y
            last = (x, y)
        elif last is not None and (x, y) == last:
            frontier.append(i)  # an exact tie of a frontier point is on it too
    return frontier


def _percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ratios(
    clusters: list[dict[str, tuple[float, int]]],
    metrics: Sequence[str],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, dict[str, float | int | None]]:
    """Ratio estimates ``sum / count`` per metric with 95% percentile
    bootstrap intervals, resampling whole clusters (tasks) with replacement.

    ``clusters`` holds, per cluster, ``metric -> (sum, count)``. A metric no
    cluster has a count for is None. Deterministic for a given ``seed``.
    """
    n = len(clusters)
    sums: dict[str, list[float]] = {
        m: [float(c.get(m, (0.0, 0))[0]) for c in clusters] for m in metrics
    }
    counts: dict[str, list[int]] = {
        m: [int(c.get(m, (0.0, 0))[1]) for c in clusters] for m in metrics
    }
    out: dict[str, dict[str, float | int | None]] = {}
    live = []
    for m in metrics:
        total = sum(counts[m])
        if total:
            out[m] = {
                "v": math.fsum(sums[m]) / total,
                "lo": None,
                "hi": None,
                "n": total,
            }
            live.append(m)
        else:
            out[m] = {"v": None, "lo": None, "hi": None, "n": 0}
    if n < 2 or not live:
        return out
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {m: [] for m in live}
    population = range(n)
    for _ in range(samples):
        # n >= 2, so the getter always returns a tuple.
        pick = operator.itemgetter(*rng.choices(population, k=n))
        for m in live:
            c = sum(pick(counts[m]))
            if c:
                draws[m].append(math.fsum(pick(sums[m])) / c)
    for m in live:
        values = sorted(draws[m])
        if values:
            out[m]["lo"] = _percentile(values, 0.025)
            out[m]["hi"] = _percentile(values, 0.975)
    return out


def _clusters(
    members: list[int],
    tasks: list[int],
    per_trial: dict[str, list[float | None]],
) -> list[dict[str, tuple[float, int]]]:
    """Per-task (sum, count) of each metric; per-trial when only one task."""
    by_task: dict[int, list[int]] = {}
    for i in members:
        by_task.setdefault(tasks[i], []).append(i)
    groups = list(by_task.values()) if len(by_task) > 1 else [[i] for i in members]
    out = []
    for group in groups:
        cluster: dict[str, tuple[float, int]] = {}
        for m, values in per_trial.items():
            known = [float(v) for v in (values[i] for i in group) if v is not None]
            cluster[m] = (math.fsum(known), len(known))
        out.append(cluster)
    return out


def _order_steps(
    values: list[str], firsts: dict[str, float], hill_order: list[str]
) -> list[str]:
    def numeric(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None

    real = [v for v in values if v != UNRECORDED]
    if real and all(numeric(v) is not None for v in real):
        return sorted(real, key=lambda v: numeric(v) or 0.0)
    if real and all(v in hill_order for v in real):
        return sorted(real, key=hill_order.index)
    return sorted(real, key=lambda v: (firsts.get(v, math.inf), v))


def build_outcomes(
    roots: Sequence[tuple[str, Path]],
    *,
    link_for: Callable[[Path], str | None] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """The ``benchflow.outcomes/1`` document for the jobs under ``roots``.

    ``roots`` pairs a label (empty for a single job) with a job folder.
    ``link_for`` gives the viewer id that opens a trial folder (None: no
    link); without it no trial is linked.
    """
    import benchflow as bf

    started = time.perf_counter()
    ctx = _Context()
    cols = _Columns()
    dims = {name: _Interner() for name in DIMENSIONS}
    causes: list[dict[str, str]] = []
    cause_index: dict[str, int] = {}
    attempt_outcomes: dict[str, list[list[Any]]] = {}
    integrity_details: dict[str, dict[str, Any]] = {}
    samples: list[Any] = []
    started_at: list[float] = []
    extra_keys: dict[str, int] = {}
    notes: list[str] = []
    load_sec = 0.0
    trials: list[tuple[Trial, Path, str]] = []
    for label, root in roots:
        t0 = time.perf_counter()
        try:
            job = bf.load_job(root)
        except FileNotFoundError:
            notes.append(f"{label or root.name}: no trials found")
            continue
        load_sec += time.perf_counter() - t0
        trials.extend((t, root, label) for t in job.trials)
    hill_order: list[str] = []
    for trial, root, label in trials:
        i = len(cols.reward)
        facts = _trial_facts(trial, root, label, ctx)
        code, reward = _outcome_code(trial)
        for name in DIMENSIONS:
            dims[name].add(OUTCOMES[code] if name == "outcome" else facts[name])
        cols.reward.append(reward)
        cols.outcome.append(code)
        cols.execution.append(EXECUTIONS.index(trial.execution))
        cols.role.append(ROLES.index(facts["role"]))
        attempts = trial.attempts
        cols.attempts.append(len(attempts))
        if len(attempts) > 1:
            attempt_outcomes[str(i)] = [
                [_outcome_code(a)[0], _outcome_code(a)[1], a.path.name]
                for a in attempts
            ]
        verdict = trial.integrity
        if verdict is None:
            cols.integrity.append(-1)
        else:
            vcode = (
                VERDICTS.index(verdict.verdict)
                if verdict.verdict in VERDICTS[:4]
                else VERDICTS.index("other")
            )
            cols.integrity.append(vcode)
            integrity_details[str(i)] = {
                "verdict": verdict.verdict,
                "exploited": verdict.exploited,
                "reason": _first_line(verdict.reason),
                "severity": verdict.severity,
                "certification": verdict.certification,
                "mode": verdict.mode,
            }
        cols.usd.append(_sum_known(a.cost_usd for a in attempts))
        cols.usd_estimated.append(
            int(any(a.result.price_source == "agent_session_log" for a in attempts))
        )
        cols.tokens.append(_sum_known(a.total_tokens for a in attempts))
        cols.sandbox_sec.append(_sum_known(_sandbox_seconds(a) for a in attempts))
        cols.wall_sec.append(_sum_known(a.duration_sec for a in attempts))
        cause = _cause(trial)
        if cause is None:
            cols.cause.append(-1)
        else:
            key = json.dumps(cause, sort_keys=True)
            if key not in cause_index:
                cause_index[key] = len(causes)
                causes.append(cause)
            cols.cause.append(cause_index[key])
        cols.link.append(link_for(trial.path) if link_for else None)
        cols.name.append(trial.path.name)
        cols.detail.append(
            _first_line(trial.result.error) or _first_line(trial.result.verifier_error)
        )
        extras: dict[str, float] = {}
        if reward is not None and isinstance(trial.result.rewards, dict):
            for key, value in trial.result.rewards.items():
                if key == "reward" or isinstance(value, bool):
                    continue
                if isinstance(value, int | float) and math.isfinite(value):
                    extras[str(key)] = float(value)
                    extra_keys[str(key)] = extra_keys.get(str(key), 0) + 1
        cols.extra_rewards.append(extras)
        samples.append(trial.solve_sample())
        begun = trial.result.started_at
        started_at.append(begun.timestamp() if begun is not None else math.inf)
        if not hill_order:
            for folder in _ancestors(trial.path, root):
                hill = ctx.hillclimb.get(folder)
                if hill:
                    hill_order = hill["order"]
                    if hill.get("best_version") or hill.get("stop"):
                        notes.append(
                            "hill-climb run: best version "
                            f"{hill.get('best_version') or 'n/a'}, stopped: "
                            f"{hill.get('stop') or 'n/a'}"
                        )
                    break

    n = len(cols.reward)
    agents = [i for i in range(n) if cols.role[i] == 0]

    # Statistics per value of every dimension (agent trials only).
    stats: dict[str, list[dict[str, Any] | None]] = {}
    for name, interner in dims.items():
        members: list[list[int]] = [[] for _ in interner.values]
        for i in agents:
            members[interner.codes[i]].append(i)
        stats[name] = [_stat(m, samples) if m else None for m in members]

    extra = sorted(extra_keys, key=lambda k: -extra_keys[k])[:_MAX_EXTRA_REWARDS]
    y_metrics = {"mean_reward": "mean reward", "solve_rate": "solve rate"}
    y_metrics.update({f"reward:{k}": f"reward: {k}" for k in extra})
    per_trial: dict[str, list[float | None]] = {
        "usd": cols.usd,
        "tokens": cols.tokens,
        "sandbox_sec": cols.sandbox_sec,
        "wall_sec": cols.wall_sec,
        "mean_reward": cols.reward,
        "solve_rate": [
            None if cols.reward[i] is None else float(cols.outcome[i] == 0)
            for i in range(n)
        ],
    }
    for k in extra:
        per_trial[f"reward:{k}"] = [cols.extra_rewards[i].get(k) for i in range(n)]

    pareto, pareto_samples = _pareto(
        dims, agents, per_trial, list(y_metrics), bootstrap_samples=bootstrap_samples
    )
    firsts: dict[str, float] = {}
    for i in agents:
        value = dims["step"].values[dims["step"].codes[i]]
        firsts[value] = min(firsts.get(value, math.inf), started_at[i])
    # Steps of agent runs only: a control run's step is not a training step.
    steps = _order_steps(list(firsts), firsts, hill_order)
    training = _training(dims, agents, samples, per_trial, steps, bootstrap_samples)
    if training is None:
        notes.append(
            "no training steps: the training view needs trials that record a "
            "policy_version, step or checkpoint (in result.json or config.json, or "
            "as step in the job's rollouts.jsonl), or a hill-climb run"
        )
    if agents and all(cols.usd[i] is None for i in agents):
        notes.append(
            "no trial recorded USD: cost needs a provider price or a Claude Code "
            "session log priced at the end of the run (price_source agent_session_log)"
        )

    def column(values: list[Any]) -> list[Any]:
        return [round(v, 6) if isinstance(v, float) else v for v in values]

    return {
        "schema": SCHEMA,
        "roots": [label or root.name for label, root in roots],
        "n": n,
        "columns": {
            "reward": column(cols.reward),
            "outcome": cols.outcome,
            "execution": cols.execution,
            "role": cols.role,
            "attempts": cols.attempts,
            "integrity": cols.integrity,
            "usd": column(cols.usd),
            "usd_estimated": cols.usd_estimated,
            "tokens": column(cols.tokens),
            "sandbox_sec": column(cols.sandbox_sec),
            "wall_sec": column(cols.wall_sec),
            "cause": cols.cause,
            "link": cols.link,
            "name": cols.name,
            "detail": cols.detail,
        },
        "vocab": {
            "outcome": list(OUTCOMES),
            "execution": list(EXECUTIONS),
            "role": list(ROLES),
            "integrity": list(VERDICTS),
        },
        "dims": {
            name: {"label": DIMENSIONS[name], "values": d.values, "codes": d.codes}
            for name, d in dims.items()
        },
        "causes": causes,
        "attempt_outcomes": attempt_outcomes,
        "integrity_details": integrity_details,
        "stats": stats,
        "x_metrics": X_METRICS,
        "y_metrics": y_metrics,
        "pareto": pareto,
        "pareto_samples": pareto_samples,
        "training": training,
        "sources": {k: sorted(v) for k, v in ctx.sources.items()},
        "notes": notes,
        "timing": {
            "load_sec": round(load_sec, 3),
            "build_sec": round(time.perf_counter() - started, 3),
        },
    }


def _pareto(
    dims: dict[str, _Interner],
    agents: list[int],
    per_trial: dict[str, list[float | None]],
    y_names: list[str],
    *,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], int]:
    """Points per (group dimension, partition dimension), with frontiers,
    and the resample count used.

    Every point's clusters are gathered first; the resample count is then
    lowered from ``bootstrap_samples`` (to no fewer than
    :data:`MIN_BOOTSTRAP_SAMPLES`) so the whole build stays within
    :data:`BOOTSTRAP_BUDGET` cluster-metric draws, however large the job.
    """
    tasks = dims["task"].codes
    metrics = [*X_METRICS, *y_names]
    plan: list[tuple[str, str, list[tuple[str, str, list[int], list[Any]]]]] = []
    for group in PARETO_GROUPS:
        g = dims[group]
        if group != "model" and len({g.codes[i] for i in agents}) < 2:
            continue
        for part in PARETO_PARTITIONS:
            if part != "none":
                p = dims[part]
                if len({p.codes[i] for i in agents}) < 2 and part != "dataset":
                    continue
            members: dict[tuple[str, str], list[int]] = {}
            for i in agents:
                pv = "all" if part == "none" else dims[part].values[dims[part].codes[i]]
                members.setdefault((pv, g.values[g.codes[i]]), []).append(i)
            plan.append(
                (
                    group,
                    part,
                    [
                        (pv, gv, idx, _clusters(idx, tasks, per_trial))
                        for (pv, gv), idx in sorted(members.items())
                    ],
                )
            )
    units = sum(len(c) for _, _, pts in plan for _, _, _, c in pts) * len(metrics)
    samples = bootstrap_samples
    if units and units * samples > BOOTSTRAP_BUDGET:
        samples = max(
            min(MIN_BOOTSTRAP_SAMPLES, bootstrap_samples), BOOTSTRAP_BUDGET // units
        )
    out: dict[str, Any] = {}
    for group, part, planned in plan:
        points = []
        for pv, gv, idx, clusters in planned:
            seed = zlib.crc32(f"{group}|{part}|{pv}|{gv}".encode())
            values = bootstrap_ratios(
                clusters,
                metrics,
                seed=seed,
                samples=samples,
            )
            points.append(
                {
                    "group": gv,
                    "partition": pv,
                    "trials": len(idx),
                    "tasks": len({tasks[i] for i in idx}),
                    "m": {
                        k: [
                            _round(v["v"]),
                            _round(v["lo"]),
                            _round(v["hi"]),
                            v["n"],
                        ]
                        for k, v in values.items()
                    },
                }
            )
        frontiers: dict[str, dict[str, list[int]]] = {}
        for x in X_METRICS:
            for y in y_names:
                by_part: dict[str, list[int]] = {}
                for pv in sorted({pt["partition"] for pt in points}):
                    idx = [
                        j
                        for j, pt in enumerate(points)
                        if pt["partition"] == pv
                        and pt["m"][x][0] is not None
                        and pt["m"][y][0] is not None
                    ]
                    coords = [(points[j]["m"][x][0], points[j]["m"][y][0]) for j in idx]
                    by_part[pv] = [idx[k] for k in pareto_frontier(coords)]
                frontiers[f"{x}|{y}"] = by_part
        out.setdefault(group, {})[part] = {"points": points, "frontiers": frontiers}
    return out, samples


def _round(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def _training(
    dims: dict[str, _Interner],
    agents: list[int],
    samples: list[Any],
    per_trial: dict[str, list[float | None]],
    steps: list[str],
    bootstrap_samples: int,
) -> dict[str, Any] | None:
    """Reward over steps per task group, and held-out tasks per step."""
    if len(steps) < 2:
        return None
    step_dim = dims["step"]
    split_dim = dims["split"]
    splits = sorted({split_dim.values[split_dim.codes[i]] for i in agents})
    group_dim = "split" if len(splits) > 1 else "dataset"
    gd = dims[group_dim]
    groups = sorted({gd.values[gd.codes[i]] for i in agents})
    step_of = {v: k for k, v in enumerate(steps)}
    tasks = dims["task"].codes
    # One pass buckets the agent trials by (group, step).
    by_cell: dict[tuple[str, str], list[int]] = {}
    for i in agents:
        key = (gd.values[gd.codes[i]], step_dim.values[step_dim.codes[i]])
        by_cell.setdefault(key, []).append(i)
    series = []
    for group in groups:
        rows = []
        for step in steps:
            idx = by_cell.get((group, step), [])
            if not idx:
                rows.append(None)
                continue
            stat = _stat(idx, samples)
            boot = bootstrap_ratios(
                _clusters(idx, tasks, {"mean_reward": per_trial["mean_reward"]}),
                ["mean_reward"],
                seed=zlib.crc32(f"train|{group}|{step}".encode()),
                samples=bootstrap_samples,
            )["mean_reward"]
            rows.append(
                {
                    "solve_rate": _round(stat["solve_rate"]),
                    "interval": [_round(v) for v in stat["interval"]]
                    if stat["interval"]
                    else None,
                    "mean_reward": _round(boot["v"]),
                    "mean_interval": [_round(boot["lo"]), _round(boot["hi"])]
                    if boot["lo"] is not None
                    else None,
                    "scored": stat["scored"],
                    "trials": stat["trials"],
                }
            )
        series.append({"group": group, "points": rows})
    heldout_names = [
        s
        for s in splits
        if s
        in {
            "test",
            "heldout",
            "held-out",
            "holdout",
            "val",
            "validation",
            "valid",
            "dev",
        }
    ]
    in_heldout = set(heldout_names)
    task_names = dims["task"].values
    held: dict[int, list[list[float]]] = {}
    for i in agents:
        if in_heldout and split_dim.values[split_dim.codes[i]] not in in_heldout:
            continue
        step = step_dim.values[step_dim.codes[i]]
        reward = per_trial["mean_reward"][i]
        if step not in step_of or reward is None:
            continue
        cells = held.setdefault(tasks[i], [[0.0, 0.0] for _ in steps])
        cells[step_of[step]][0] += reward
        cells[step_of[step]][1] += 1
    heldout = [
        {
            "task": task_names[t],
            "mean": [_round(s / c) if c else None for s, c in cells],
            "n": [int(c) for _, c in cells],
        }
        for t, cells in sorted(held.items(), key=lambda kv: task_names[kv[0]])
    ]
    return {
        "steps": steps,
        "group_dim": group_dim,
        "series": series,
        "heldout_split": heldout_names or None,
        "heldout": heldout,
    }
