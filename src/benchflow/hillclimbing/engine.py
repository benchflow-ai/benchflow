"""The climb: baseline, noise gate, rounds of propose, evaluate, keep or revert.

The loop follows "Automating eval design and hillclimbing with Claude"
(claude.dev, 2026-09-28), with its safeguards enforced by the runtime rather
than left to the optimizer:

1. Split the tasks at random into train and test (:mod:`.split`).
2. Check the graders: the oracle must pass and doing nothing must not
   (:func:`.evaluate.run_controls`).
3. Run the baseline ``trials`` times on both splits and refuse to climb when
   ``min_gain`` is not above the noise (:func:`.stats.noise_gate`).
4. Each round, a proposer rollout reads the train split's failures and edits
   the surface once (:mod:`.proposer`); it never sees the test split.
5. Run the candidate on train and test. Keep it only if train gains at least
   ``min_gain`` and test improves; revert a flat or lower test score (train
   up and test flat means overfitting), or any regression (:func:`decide`).
6. After ``stall_rounds`` rounds with nothing kept, stop and have the
   proposer sort the remaining train failures by root cause.
7. Report the best version on the test split against the baseline, with
   confidence intervals and whether the gain exceeds noise.

Infrastructure errors are left out of every score and counted; the climb
stops when they pass ``max_infra_error_rate``, and a candidate whose errors
rise is reverted, so a patch cannot look better by making hard trials crash.
"""

from __future__ import annotations

import asyncio
import logging
import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from benchflow.hillclimbing import record as rec
from benchflow.hillclimbing import stats
from benchflow.hillclimbing.evaluate import (
    GRADER_BUG_FLAGS,
    MISSING,
    EvalSettings,
    SplitName,
    SplitRun,
    TaskSet,
    evaluate_version,
    run_controls,
)
from benchflow.hillclimbing.proposer import (
    ProposerSettings,
    Workspace,
    build_workspace,
    exposure,
    instruction_text,
    run_proposer,
)
from benchflow.hillclimbing.split import (
    SPLIT_FILE,
    load_split_file,
    make_split,
    task_strata,
)
from benchflow.hillclimbing.surface import (
    SurfaceHistory,
    SurfaceStore,
    added_text,
    diff_versions,
    parse_surface,
    pasted_spans,
    truncate_diff,
)

logger = logging.getLogger(__name__)

EPS = 1e-9
REPORT_FILE = "report.html"
Objective = Literal["score", "cost"]
StopReason = Literal[
    "rounds", "stalled", "budget", "infra", "noise_gate", "error", "interrupted"
]
SPLITS: tuple[SplitName, ...] = ("train", "test")


class HillclimbError(ValueError):
    """A configuration the climb cannot run with."""


@dataclass
class HillclimbConfig:
    """Everything ``bench hillclimb`` takes; see docs/hillclimb.md."""

    tasks: str | Path | Sequence[str | Path]
    surface: str | Path | Sequence[str | Path]
    out: str | Path
    # The agent under test (the usual bench eval run options).
    agent: str = "claude-agent-acp"
    model: str | None = None
    reasoning_effort: str | None = None
    environment: str = "docker"
    concurrency: int = 4
    agent_env: dict[str, str] = field(default_factory=dict)
    sandbox_user: str | None = "agent"
    agent_idle_timeout: int | None = 600
    retry_attempts: int | None = None
    config_override: dict[str, Any] | None = None
    include: Sequence[str] = ()
    exclude: Sequence[str] = ()
    # The split.
    test_frac: float = 0.3
    seed: int = 0
    split_file: str | Path | None = None
    stratify_by: str | None = "category"
    # The objective and the loop.
    objective: Objective = "score"
    rounds: int = 5
    trials: int = 3
    min_gain: float = 0.05
    max_cost_usd: float | None = None
    stall_rounds: int = 3
    candidates: int = 1
    max_infra_error_rate: float = 0.25
    controls: bool = True
    exclude_broken_tasks: bool = False
    force: bool = False
    leak_check: Literal["reject", "warn", "off"] = "reject"
    bootstrap_samples: int = stats.DEFAULT_SAMPLES
    analyze_at_end: bool = False
    # The optimizer.
    proposer: ProposerSettings | None = None
    preflight: bool = True

    def __post_init__(self) -> None:
        from benchflow._utils.config import (
            normalize_agent_idle_timeout,
            normalize_reasoning_effort,
            normalize_sandbox_user,
        )

        checks = [
            (self.objective in ("score", "cost"), "--objective is score or cost"),
            (self.rounds >= 0, "--rounds must be >= 0"),
            (self.trials >= 1, "--trials must be >= 1"),
            (self.candidates >= 1, "--candidates must be >= 1"),
            (self.stall_rounds >= 1, "--stall-rounds must be >= 1"),
            (self.min_gain > 0, "--min-gain must be above 0"),
            (0 < self.test_frac < 1, "--test-frac must be between 0 and 1"),
            (
                0 < self.max_infra_error_rate <= 1,
                "--max-infra-error-rate must be in (0, 1]",
            ),
            (
                self.leak_check in ("reject", "warn", "off"),
                "--leak-check is reject, warn or off",
            ),
            (
                self.max_cost_usd is None or self.max_cost_usd > 0,
                "--max-cost-usd must be above 0",
            ),
            (self.bootstrap_samples >= 100, "--bootstrap-samples must be >= 100"),
        ]
        for ok, message in checks:
            if not ok:
                raise HillclimbError(message)
        self.reasoning_effort = normalize_reasoning_effort(self.reasoning_effort)
        self.sandbox_user = normalize_sandbox_user(self.sandbox_user)
        self.agent_idle_timeout = normalize_agent_idle_timeout(self.agent_idle_timeout)
        if self.proposer is None:
            self.proposer = ProposerSettings(environment=self.environment)

    @property
    def surfaces(self) -> list[str | Path]:
        if isinstance(self.surface, str | Path):
            return [self.surface]
        return list(self.surface)


@dataclass
class HillclimbResult:
    """What ``bf.hillclimb`` returns: the run folder and its record."""

    run_dir: Path
    record: rec.HillclimbDoc

    @property
    def status(self) -> str:
        return self.record.status

    @property
    def report(self) -> Path:
        return self.run_dir / REPORT_FILE

    @property
    def best_surface(self) -> Path | None:
        best = self.record.best
        return self.run_dir / best.surface_dir if best else None

    @property
    def verdict(self) -> str | None:
        best = self.record.best
        return best.verdict.text if best else None


# ---------------------------------------------------------------------------
# The keep/revert rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    keep: bool
    reasons: list[str]


def decide(
    *,
    objective: Objective,
    min_gain: float,
    train_delta: float | None,
    test_delta: float | None,
    train_cost_change: float | None = None,
    test_cost_change: float | None = None,
    train_noise: float | None = None,
    test_noise: float | None = None,
    infra_rise: Sequence[str] = (),
) -> Decision:
    """The post's rule, exactly.

    Score: keep only if train gains at least ``min_gain`` and test improves;
    a flat or lower test score, or any regression, reverts. Cost (lower is
    better): keep only if train cost falls by at least ``min_gain`` (a
    fraction), test cost falls too, and neither split's score drops by more
    than its noise band. ``infra_rise`` reasons always revert.
    """
    reasons = list(infra_rise)
    dt, ds = train_delta, test_delta
    if dt is None or ds is None:
        return Decision(False, [*reasons, "no task was scored on both sides"])
    if objective == "score":
        if dt < -EPS:
            reasons.append(f"train regressed ({dt:+.3f})")
        elif dt < min_gain - EPS:
            reasons.append(f"train gain {dt:+.3f} is below --min-gain {min_gain:g}")
        if ds < -EPS:
            reasons.append(f"test regressed ({ds:+.3f})")
        elif ds <= EPS and dt >= min_gain - EPS:
            reasons.append("train improved but test is flat: overfitting suspected")
        elif ds <= EPS:
            reasons.append("test is flat")
        if not reasons:
            return Decision(
                True,
                [f"train {dt:+.3f} (>= --min-gain {min_gain:g}) and test {ds:+.3f}"],
            )
        return Decision(False, reasons)
    # Cost: changes are relative (negative = cheaper); scores must hold.
    ct, cs = train_cost_change, test_cost_change
    if ct is None or cs is None:
        return Decision(False, [*reasons, "cost was not recorded on both sides"])
    band_t = train_noise or 0.0
    band_s = test_noise or 0.0
    if dt < -band_t - EPS:
        reasons.append(
            f"train score fell {dt:+.3f}, beyond the noise band {band_t:.3f}"
        )
    if ds < -band_s - EPS:
        reasons.append(f"test score fell {ds:+.3f}, beyond the noise band {band_s:.3f}")
    if -ct < min_gain - EPS:
        reasons.append(f"train cost changed {ct:+.1%}, short of a {min_gain:.0%} cut")
    if -cs <= EPS:
        reasons.append(f"test cost did not fall ({cs:+.1%}): overfitting suspected")
    if not reasons:
        return Decision(
            True, [f"cost {ct:+.1%} on train and {cs:+.1%} on test, scores held"]
        )
    return Decision(False, reasons)


def infra_rise(current: SplitRun, candidate: SplitRun) -> str | None:
    """A reason when the candidate's infrastructure errors rose notably."""
    before, after = len(current.infra_records), len(candidate.infra_records)
    allowed = max(2, math.ceil(0.1 * len(candidate.records)))
    if after - before > allowed:
        return (
            f"{candidate.split}: infrastructure errors rose from {before} to {after}; "
            "a patch must not raise its score by making trials crash"
        )
    return None


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _seed(base: int, label: str) -> int:
    return base * 1_000_003 + zlib.crc32(label.encode())


def _interval(ci: stats.Interval | None) -> rec.IntervalDoc | None:
    return rec.IntervalDoc(low=ci.low, high=ci.high) if ci else None


def _estimate(e: stats.Estimate) -> rec.EstimateDoc:
    return rec.EstimateDoc(
        value=e.value, se=e.se, ci=_interval(e.ci), tasks=e.tasks, trials=e.trials
    )


def _delta(d: stats.Delta) -> rec.DeltaDoc:
    return rec.DeltaDoc(
        value=d.value,
        se=d.se,
        ci=_interval(d.ci),
        paired_tasks=d.paired_tasks,
        samples=d.samples,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class _Evaluated:
    """One surface version's evaluation, with its record."""

    id: str
    version: str
    runs: dict[SplitName, SplitRun]
    doc: rec.EvaluationDoc

    def values(
        self, split: SplitName, objective: str = "score"
    ) -> dict[str, list[float]]:
        return self.runs[split].values(objective)

    def cost(self) -> float:
        return math.fsum(r.cost_total or 0.0 for r in self.runs.values())


class _Climb:
    def __init__(self, config: HillclimbConfig) -> None:
        self.cfg = config
        self.run_dir = Path(config.out).expanduser().resolve()
        self.proposer: ProposerSettings = config.proposer or ProposerSettings()
        self.settings = EvalSettings(
            agent=config.agent,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            environment=config.environment,
            concurrency=config.concurrency,
            agent_env=dict(config.agent_env),
            sandbox_user=config.sandbox_user,
            agent_idle_timeout=config.agent_idle_timeout,
            retry_attempts=config.retry_attempts,
            config_override=config.config_override,
            preflight=config.preflight,
        )
        self.agent_usd = 0.0
        self.proposer_usd = 0.0
        self.usd_unknown = 0
        self.budget_stopped = False
        self.warnings: list[str] = []
        self.doc: rec.HillclimbDoc | None = None
        self.best_round = 0
        self.test_instructions: dict[str, str] = {}

    # -- bookkeeping ------------------------------------------------------

    def rel(self, path: Path | str | None) -> str | None:
        if path is None:
            return None
        path = Path(path)
        try:
            return path.resolve().relative_to(self.run_dir).as_posix()
        except ValueError:
            return str(path)

    @property
    def spent(self) -> float:
        return self.agent_usd + self.proposer_usd

    def remaining(self) -> float | None:
        if self.cfg.max_cost_usd is None:
            return None
        return self.cfg.max_cost_usd - self.spent

    def save(self) -> None:
        assert self.doc is not None
        self.doc.updated_at = _now()
        self.doc.cost = rec.CostDoc(
            total_usd=round(self.spent, 6),
            agent_usd=round(self.agent_usd, 6),
            proposer_usd=round(self.proposer_usd, 6),
            usd_unknown_rollouts=self.usd_unknown,
            max_cost_usd=self.cfg.max_cost_usd,
            budget_stopped=self.budget_stopped,
        )
        self.doc.warnings = list(dict.fromkeys(self.warnings))
        rec.write_record(self.doc, self.run_dir)
        try:
            from benchflow.hillclimbing.report import write_report

            write_report(self.doc, self.run_dir / REPORT_FILE)
        except Exception:  # the record matters more than its rendering
            logger.warning("Could not render the hillclimb report", exc_info=True)

    def warn(self, message: str) -> None:
        logger.warning(message)
        self.warnings.append(message)

    # -- evaluation -------------------------------------------------------

    def split_doc(self, run: SplitRun, label: str) -> rec.SplitResultDoc:
        samples = self.cfg.bootstrap_samples
        score = stats.bootstrap_score(
            run.values("score"), samples=samples, seed=_seed(self.cfg.seed, label)
        )
        cost_values = run.values("cost")
        cost = (
            stats.bootstrap_score(
                cost_values, samples=samples, seed=_seed(self.cfg.seed, label + ":cost")
            )
            if any(cost_values.values())
            else None
        )
        return rec.SplitResultDoc(
            split=run.split,
            job_dir=self.rel(run.dir) or "",
            tasks=len(run.tasks),
            trials_per_task=run.trials,
            attempted_trials=len(run.records),
            score=_estimate(score),
            cost=_estimate(cost) if cost else None,
            cost_usd_total=run.cost_total,
            usd_unknown_trials=run.usd_unknown,
            infra_errors=len(run.infra_records),
            infra_error_rate=round(run.infra_rate, 6),
            infra_error_categories=run.infra_categories(),
            per_task=[rec.TaskResultDoc(**row) for row in run.per_task()],
        )

    async def evaluate(self, eval_id: str, version: str) -> _Evaluated:
        logger.info("hillclimb: evaluating %s (%s) on train and test", eval_id, version)
        runs = await evaluate_version(
            self.taskset,
            {"train": list(self.split.train), "test": list(self.split.test)},
            version_dir=self.store.path(version),
            specs=self.store.specs,
            settings=self.settings,
            out_dir=self.run_dir / "evals" / eval_id,
            trials=self.cfg.trials,
            budget_usd=self.remaining(),
        )
        for run in runs.values():
            self.agent_usd += run.cost_total or 0.0
            self.usd_unknown += run.usd_unknown
        doc = rec.EvaluationDoc(
            id=eval_id,
            version=version,
            train=self.split_doc(runs["train"], f"{eval_id}:train"),
            test=self.split_doc(runs["test"], f"{eval_id}:test"),
        )
        if self.cfg.max_cost_usd is not None and self.spent >= self.cfg.max_cost_usd:
            self.budget_stopped = True
        return _Evaluated(eval_id, version, runs, doc)

    def infra_breach(self, ev: _Evaluated) -> str | None:
        records = [r for run in ev.runs.values() for r in run.records]
        errors = [r for r in records if r.infra]
        if not records:
            return None
        rate = len(errors) / len(records)
        if rate <= self.cfg.max_infra_error_rate:
            return None
        cats: dict[str, int] = {}
        for r in errors:
            cats[r.category or "unscored"] = cats.get(r.category or "unscored", 0) + 1
        top = ", ".join(
            f"{k} {v}" for k, v in sorted(cats.items(), key=lambda x: -x[1])
        )
        return (
            f"{len(errors)} of {len(records)} trials of {ev.id} ({rate:.0%}) ended "
            f"without a score ({top}), above --max-infra-error-rate "
            f"{self.cfg.max_infra_error_rate:.0%}. The plumbing is broken: fix it "
            "before climbing (see the trials under "
            f"{self.rel(self.run_dir / 'evals' / ev.id)})."
        )

    # -- setup ------------------------------------------------------------

    def prepare(self) -> None:
        cfg = self.cfg
        if (self.run_dir / rec.RECORD_FILE).exists():
            raise HillclimbError(
                f"{self.run_dir} already holds a hillclimb run ({rec.RECORD_FILE}); "
                "pass a new --out"
            )
        self.taskset = TaskSet.resolve(
            cfg.tasks, include=cfg.include, exclude=cfg.exclude
        )
        if cfg.split_file:
            self.split = load_split_file(cfg.split_file, self.taskset.dirs)
        else:
            self.split = make_split(
                task_strata(self.taskset.dirs, cfg.stratify_by),
                test_frac=cfg.test_frac,
                seed=cfg.seed,
                stratify_by=cfg.stratify_by,
            )
        specs = [parse_surface(s) for s in cfg.surfaces]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.split.save(self.run_dir / SPLIT_FILE)
        self.store = SurfaceStore(self.run_dir / "surfaces", specs)
        baseline_version, skipped = self.store.baseline()
        if skipped:
            self.warn(f"symlinks in the surface were not copied: {', '.join(skipped)}")
        self.history = SurfaceHistory(self.run_dir / "surface-history")
        self.baseline_commit = self.history.commit(
            self.store.path(baseline_version),
            [s.name for s in specs],
            "baseline surface",
        )
        self.current_commit = self.baseline_commit
        cfg_doc = rec.ConfigDoc(
            tasks=[str(p) for p in self.taskset.dirs.values()],
            surfaces=[
                rec.SurfaceDoc(kind=s.kind, source=str(s.source), name=s.name)
                for s in specs
            ],
            agent=cfg.agent,
            model=cfg.model,
            reasoning_effort=cfg.reasoning_effort,
            environment=cfg.environment,
            concurrency=cfg.concurrency,
            objective=cfg.objective,
            rounds=cfg.rounds,
            trials=cfg.trials,
            min_gain=cfg.min_gain,
            max_cost_usd=cfg.max_cost_usd,
            stall_rounds=cfg.stall_rounds,
            candidates=cfg.candidates,
            max_infra_error_rate=cfg.max_infra_error_rate,
            leak_check=cfg.leak_check,
            bootstrap_samples=cfg.bootstrap_samples,
            seed=cfg.seed,
            force=cfg.force,
            proposer=rec.ProposerConfigDoc(
                agent=self.proposer.agent,
                model=self.proposer.model,
                reasoning_effort=self.proposer.reasoning_effort,
                environment=self.proposer.environment,
                timeout_sec=self.proposer.timeout_sec,
                image=self.proposer.image,
                open_network=self.proposer.open_network,
                max_failures=self.proposer.max_failures,
            ),
            config_override=cfg.config_override,
            agent_env_keys=sorted(cfg.agent_env),
        )
        now = _now()
        self.doc = rec.HillclimbDoc(
            status="running",
            benchflow_version=_version(),
            created_at=now,
            updated_at=now,
            config=cfg_doc,
            split=self.split_doc_(),
            cost=rec.CostDoc(
                total_usd=0.0,
                agent_usd=0.0,
                proposer_usd=0.0,
                usd_unknown_rollouts=0,
                max_cost_usd=cfg.max_cost_usd,
            ),
            paths=rec.PathsDoc(
                report=REPORT_FILE,
                split=SPLIT_FILE,
                surfaces="surfaces",
                surface_history="surface-history" if self.history.enabled else None,
                evals="evals",
                proposer="proposer",
            ),
        )
        if cfg.proposer and cfg.proposer.open_network:
            self.warn(
                "--proposer-open-network: the proposer sandbox has network access, "
                "so it could fetch public copies of the test tasks"
            )
        self.save()

    def split_doc_(self) -> rec.SplitDoc:
        return rec.SplitDoc(
            method=self.split.method,
            seed=self.split.seed,
            test_frac=self.split.test_frac,
            stratify_by=self.split.stratify_by,
            source=self.split.source,
            train=list(self.split.train),
            test=list(self.split.test),
            strata=dict(self.split.strata),
        )

    async def controls(self) -> None:
        assert self.doc is not None
        names = sorted([*self.split.train, *self.split.test])
        judged = [n for n in names if _llm_judged(self.taskset.dirs[n])]
        judge_note = (
            f"{len(judged)} task(s) use an LLM judge ({', '.join(judged[:5])}); grader "
            "consistency is not measured yet. TODO: freeze the baseline's workspaces "
            "and regrade a sample twice with bf.regrade to report a flip rate."
            if judged
            else "no LLM-judged tasks"
        )
        if judged:
            self.warn(judge_note)
        if not self.cfg.controls:
            self.doc.controls = rec.ControlsDoc(
                ran=False,
                skipped_reason="--skip-controls",
                judge_consistency=judge_note,
            )
            self.save()
            return
        out = self.run_dir / "controls"
        results = await run_controls(
            self.taskset, names, settings=self.settings, out_dir=out
        )
        tasks = []
        bugs = []
        for r in results:
            tasks.append(
                rec.ControlTaskDoc(
                    task=r.task,
                    split=self.split.split_of(r.task),
                    oracle_reward=r.oracle.reward if r.oracle else None,
                    oracle_error=r.oracle.error if r.oracle else None,
                    oracle_ran=r.oracle is not None,
                    nop_reward=r.nop.reward if r.nop else None,
                    nop_error=r.nop.error if r.nop else None,
                    flags=r.flags,
                )
            )
            if GRADER_BUG_FLAGS & set(r.flags):
                bugs.append(r.task)
                self.warn(
                    f"grader check: {r.task} ({self.split.split_of(r.task)}): "
                    + ", ".join(f for f in r.flags if f in GRADER_BUG_FLAGS)
                )
        excluded: list[str] = []
        if bugs and self.cfg.exclude_broken_tasks:
            new_split = self.split.without(bugs)  # raises when a side empties
            excluded = bugs
            self.split = new_split
            self.doc.split = self.split_doc_()
        self.doc.controls = rec.ControlsDoc(
            ran=True,
            oracle_job_dir=self.rel(out / "oracle")
            if (out / "oracle").exists()
            else None,
            nop_job_dir=self.rel(out / "nop"),
            tasks=tasks,
            grader_bugs=bugs,
            excluded=excluded,
            judge_consistency=judge_note,
        )
        self.save()

    # -- proposer ---------------------------------------------------------

    def scores_for_proposer(
        self, baseline: _Evaluated, current: _Evaluated
    ) -> dict[str, Any]:
        """What the proposer may know: train in detail, test as aggregates only."""
        assert self.doc is not None

        def agg(e: rec.EstimateDoc) -> dict[str, Any]:
            return {
                "score": e.value,
                "ci95": [e.ci.low, e.ci.high] if e.ci else None,
                "tasks": e.tasks,
            }

        gate = self.doc.noise_gate
        rounds = []
        for rd in self.doc.rounds:
            for c in rd.candidates:
                rounds.append(
                    {
                        "round": c.round,
                        "candidate": c.id,
                        "decision": c.decision,
                        "reasons": c.reasons,
                        "train_delta": c.train_delta.value if c.train_delta else None,
                        "train_delta_ci95": [
                            c.train_delta.ci.low,
                            c.train_delta.ci.high,
                        ]
                        if c.train_delta and c.train_delta.ci
                        else None,
                        "test_delta": c.test_delta.value if c.test_delta else None,
                        "test_delta_ci95": [c.test_delta.ci.low, c.test_delta.ci.high]
                        if c.test_delta and c.test_delta.ci
                        else None,
                    }
                )
        return {
            "objective": self.cfg.objective,
            "min_gain": self.cfg.min_gain,
            "note": (
                "Test scores are aggregates over held-out tasks you cannot see. "
                "Scores are mean rewards (0 to 1) with 95% bootstrap intervals."
            ),
            "baseline": {
                "train": agg(baseline.doc.train.score),
                "test": agg(baseline.doc.test.score),
            },
            "current": {
                "version": current.version,
                "train": {
                    **agg(current.doc.train.score),
                    "per_task": [
                        {
                            "task": t.task,
                            "mean_reward": t.mean_reward,
                            "rewards": t.rewards,
                        }
                        for t in current.doc.train.per_task
                    ],
                },
                "test": agg(current.doc.test.score),
            },
            "noise_95": {
                "train": gate.train.noise_95 if gate else None,
                "test": gate.test.noise_95 if gate else None,
            },
            "rounds": rounds,
        }

    def history_for_proposer(self) -> list[dict[str, Any]]:
        assert self.doc is not None
        out = []
        for rd in self.doc.rounds:
            for c in rd.candidates:
                out.append(
                    {
                        "id": c.id,
                        "round": c.round,
                        "decision": c.decision,
                        "reasons": c.reasons,
                        "root_cause": c.root_cause,
                        "change": c.change,
                        "rationale": c.rationale,
                        "train_delta": c.train_delta.value if c.train_delta else None,
                        "test_delta": c.test_delta.value if c.test_delta else None,
                        "diff": c.diff,
                    }
                )
        return out

    def leak_sources(self, evidence: Path) -> dict[str, str]:
        sources: dict[str, str] = {}
        train_root = evidence / "train"
        for path in sorted(train_root.rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(train_root).parts
            if parts[0] == "tasks" and path.name == "instruction.md":
                sources[f"train task instruction: {parts[1]}"] = path.read_text(
                    errors="replace"
                )
            elif parts[0] == "failures" and "verifier" in parts:
                sources[f"verifier output: {'/'.join(parts[1:])}"] = path.read_text(
                    errors="replace"
                )
        return sources

    def exposure_doc(self, ws: Workspace, label: str) -> rec.ExposureDoc:
        """Record what one optimizer sandbox received; refuse a test-split leak."""
        for name in self.split.test:
            if name not in self.test_instructions:
                self.test_instructions[name] = instruction_text(self.taskset.dirs[name])
        manifest = self.run_dir / "proposer" / label / "mounted.json"
        raw = exposure(
            ws,
            test_instructions={t: self.test_instructions[t] for t in self.split.test},
            open_network=self.proposer.open_network,
            manifest_path=manifest,
        )
        if raw["test_tasks_in_paths"]:  # built from train data only: a bug
            raise RuntimeError(
                f"test tasks in the {label} workspace: {raw['test_tasks_in_paths']}"
            )
        for task in raw["test_instructions_in_files"]:
            self.warn(
                f"{label}: the instruction of test task {task} appears in train "
                "material; a train task may duplicate it"
            )
        return rec.ExposureDoc(
            mounts=[rec.MountDoc(**m) for m in raw["mounts"]],
            network=raw["network"],
            manifest=self.rel(manifest) or "",
            train_tasks=raw["train_tasks"],
            failures=raw["failures"],
            infra_errors=raw["infra_errors"],
            test_tasks=raw["test_tasks"],
            test_tasks_in_paths=raw["test_tasks_in_paths"],
            test_instructions_in_files=raw["test_instructions_in_files"],
        )

    async def propose(
        self,
        cid: str,
        round_no: int,
        baseline: _Evaluated,
        current: _Evaluated,
        version: str,
    ) -> rec.CandidateDoc:
        ws = build_workspace(
            self.run_dir / "proposer" / cid / "workspace",
            mode="propose",
            version_dir=self.store.path(current.version),
            specs=self.store.specs,
            train=current.runs["train"],
            train_dirs=self.taskset.dirs,
            scores=self.scores_for_proposer(baseline, current),
            history=self.history_for_proposer(),
            objective=self.cfg.objective,
            max_failures=self.proposer.max_failures,
            extra_instructions=self.proposer.extra_instructions,
        )
        seen = self.exposure_doc(ws, cid)
        outcome = await run_proposer(
            ws,
            mode="propose",
            settings=self.proposer,
            task_dir=self.run_dir / "proposer" / cid / "task" / "hillclimb-propose",
            jobs_dir=self.run_dir / "proposer" / cid,
        )
        if outcome.cost_usd is not None:
            self.proposer_usd += outcome.cost_usd
        else:
            self.usd_unknown += 1
        proposer_doc = rec.ProposerDoc(
            status=outcome.status,
            error=outcome.error,
            rollout_dir=self.rel(outcome.rollout_dir),
            workspace_dir=self.rel(ws.root),
            exposure=seen,
            reward=outcome.reward,
            cost_usd=outcome.cost_usd,
        )
        out = outcome.output or {}

        def text(key: str) -> str | None:
            value = out.get(key)
            return value.strip() if isinstance(value, str) and value.strip() else None

        evidence = out.get("evidence")
        cand = rec.CandidateDoc(
            id=cid,
            round=round_no,
            base_version=current.version,
            version=None,
            proposer=proposer_doc,
            root_cause=text("root_cause"),
            change=text("change"),
            rationale=text("rationale"),
            evidence=[str(e) for e in evidence][:50]
            if isinstance(evidence, list)
            else [],
            decision="invalid",
        )
        if outcome.status != "ok" or outcome.surface_dir is None:
            cand.reasons = [outcome.error or "the proposer failed"]
            return cand
        problems, skipped = self.store.add(version, outcome.surface_dir)
        if skipped:
            self.warn(
                f"{cid}: symlinks in the edited surface were dropped: {', '.join(skipped)}"
            )
        diff, diff_stats = diff_versions(
            self.store.path(current.version), self.store.path(version)
        )
        cand.diff, cand.diff_truncated = truncate_diff(diff)
        cand.diff_stats = rec.DiffStatsDoc(**diff_stats.to_dict())
        if problems:
            cand.reasons = ["the edited surface is not usable: " + "; ".join(problems)]
            return cand
        cand.version = version
        if not diff.strip():
            cand.reasons = ["the proposer changed nothing"]
            return cand
        if self.cfg.leak_check == "off":
            cand.leak_check = rec.LeakCheckDoc(mode="off", status="skipped")
        else:
            matches = pasted_spans(added_text(diff), self.leak_sources(ws.evidence))
            cand.leak_check = rec.LeakCheckDoc(
                mode=self.cfg.leak_check,
                status="flagged" if matches else "clean",
                matches=matches[:10],
            )
            if matches and self.cfg.leak_check == "reject":
                cand.reasons = [
                    "the patch pastes text from what the proposer read ("
                    + "; ".join(m["source"] for m in matches[:3])
                    + "); pasted failures do not transfer"
                ]
                return cand
            if matches:
                self.warn(f"{cid}: the patch copies text from {matches[0]['source']}")
        cand.decision = "pending"
        return cand

    # -- the climb --------------------------------------------------------

    def score_noise(self) -> tuple[float | None, float | None]:
        assert self.doc is not None and self.doc.noise_gate is not None
        g = self.doc.noise_gate
        return g.train.noise_95, g.test.noise_95

    def judge(
        self, current: _Evaluated, ev: _Evaluated, cand: rec.CandidateDoc
    ) -> tuple[Decision, float | None]:
        """Deltas and the decision for one evaluated candidate; returns the
        train gain used to rank candidates."""
        samples = self.cfg.bootstrap_samples
        deltas: dict[SplitName, stats.Delta] = {}
        for split in SPLITS:
            deltas[split] = stats.paired_delta(
                current.values(split),
                ev.values(split),
                samples=samples,
                seed=_seed(self.cfg.seed, f"{cand.id}:{split}:delta"),
            )
        cand.train_delta = _delta(deltas["train"])
        cand.test_delta = _delta(deltas["test"])
        change = _cost_change(current, ev)
        cand.cost_change = rec.CostChangeDoc(train=change["train"], test=change["test"])
        rises = [
            reason
            for split in SPLITS
            if (reason := infra_rise(current.runs[split], ev.runs[split]))
        ]
        noise_t, noise_s = self.score_noise()
        decision = decide(
            objective=self.cfg.objective,
            min_gain=self.cfg.min_gain,
            train_delta=deltas["train"].value,
            test_delta=deltas["test"].value,
            train_cost_change=change["train"],
            test_cost_change=change["test"],
            train_noise=noise_t,
            test_noise=noise_s,
            infra_rise=rises,
        )
        if self.cfg.objective == "cost":
            gain = -change["train"] if change["train"] is not None else None
        else:
            gain = deltas["train"].value
        return decision, gain

    async def run(self) -> HillclimbResult:
        cfg = self.cfg
        self.prepare()
        assert self.doc is not None
        doc = self.doc
        try:
            await self.controls()
            baseline = await self.evaluate("baseline", "v000")
            doc.baseline = baseline.doc
            self.save()
            missing = sum(
                r.category == MISSING
                for run in baseline.runs.values()
                for r in run.records
            )
            if self.budget_stopped and missing:
                return self.finish(
                    "stopped",
                    "budget",
                    f"--max-cost-usd was reached during the baseline ({missing} "
                    "trial(s) did not run); raise it or run fewer trials",
                    baseline,
                    baseline,
                )
            breach = self.infra_breach(baseline)
            if breach:
                return self.finish("stopped", "infra", breach, baseline, baseline)
            self.gate(baseline)
            assert doc.noise_gate is not None
            if not doc.noise_gate.passed and not cfg.force:
                return self.finish(
                    "refused", "noise_gate", doc.noise_gate.message, baseline, baseline
                )
            if not doc.noise_gate.passed:
                self.warn(
                    "--force: climbing although the noise gate refused (ungated run)"
                )
            return await self.climb(baseline)
        except (KeyboardInterrupt, asyncio.CancelledError):
            doc.status = "stopped"
            doc.stop = rec.StopDoc(
                reason="interrupted", detail="the run was interrupted"
            )
            self.save()
            raise
        except Exception as exc:
            doc.status = "failed"
            doc.stop = rec.StopDoc(
                reason="error", detail=f"{type(exc).__name__}: {exc}"
            )
            self.save()
            raise

    def gate(self, baseline: _Evaluated) -> None:
        assert self.doc is not None
        cfg = self.cfg
        g = stats.noise_gate(
            baseline.values("train", cfg.objective),
            baseline.values("test", cfg.objective),
            min_gain=cfg.min_gain,
            objective=cfg.objective,
            trials=cfg.trials,
            samples=cfg.bootstrap_samples,
            seed=_seed(cfg.seed, "gate"),
        )
        # Score noise bands are the tolerance for the cost objective.
        score_gate = (
            g
            if cfg.objective == "score"
            else stats.noise_gate(
                baseline.values("train"),
                baseline.values("test"),
                min_gain=1.0,
                trials=cfg.trials,
                samples=cfg.bootstrap_samples,
                seed=_seed(cfg.seed, "gate:score"),
            )
        )

        def split_doc(
            gs: stats.GateSplit, score_gs: stats.GateSplit
        ) -> rec.GateSplitDoc:
            return rec.GateSplitDoc(
                split="train" if gs.split == "train" else "test",
                value=gs.value,
                se=gs.se,
                noise_se=score_gs.noise.se,
                noise_95=score_gs.noise.band95,
                threshold=None if math.isinf(gs.threshold) else gs.threshold,
                tasks=gs.noise.tasks,
                min_trials=gs.noise.min_trials,
                ok=gs.ok,
                reason=gs.reason,
            )

        self.doc.noise_gate = rec.NoiseGateDoc(
            passed=g.passed,
            forced=cfg.force and not g.passed,
            objective=cfg.objective,
            min_gain=cfg.min_gain,
            train=split_doc(g.train, score_gate.train),
            test=split_doc(g.test, score_gate.test),
            message=g.message,
            suggestion=rec.SuggestionDoc(
                trials=g.suggestion.trials,
                train_tasks=g.suggestion.train_tasks,
                test_tasks=g.suggestion.test_tasks,
                min_gain=g.suggestion.min_gain,
            ),
        )
        logger.info(g.message)
        self.save()

    async def climb(self, baseline: _Evaluated) -> HillclimbResult:
        assert self.doc is not None
        cfg = self.cfg
        doc = self.doc
        current = baseline
        current_round = 0
        stall = 0
        version_no = 0
        last_proposer_cost = 0.0
        stop: tuple[StopReason, str] = ("rounds", f"ran all {cfg.rounds} round(s)")
        for round_no in range(1, cfg.rounds + 1):
            remaining = self.remaining()
            if remaining is not None:
                estimate = cfg.candidates * (baseline.cost() + last_proposer_cost)
                if remaining <= 0 or remaining < estimate:
                    self.budget_stopped = True
                    stop = (
                        "budget",
                        f"${remaining:.2f} of --max-cost-usd {cfg.max_cost_usd:g} is left; "
                        f"a round costs about ${estimate:.2f}",
                    )
                    break
            spent_before = self.spent
            candidates: list[rec.CandidateDoc] = []
            evaluated: dict[str, _Evaluated] = {}
            for i in range(1, cfg.candidates + 1):
                cid = f"r{round_no:02d}-c{i}"
                version_no += 1
                cand = await self.propose(
                    cid, round_no, baseline, current, f"v{version_no:03d}"
                )
                last_proposer_cost = cand.proposer.cost_usd or last_proposer_cost
                candidates.append(cand)
            round_doc = rec.RoundDoc(
                round=round_no,
                base_version=current.version,
                candidates=candidates,
                kept=None,
                version_after=current.version,
            )
            doc.rounds.append(round_doc)
            self.save()
            # Candidates run one after another: on Docker two surfaces of one
            # task must not build its image at the same time.
            infra_stop: str | None = None
            for cand in candidates:
                if cand.version is None or cand.reasons:
                    continue
                remaining = self.remaining()
                if remaining is not None and remaining <= 0:
                    self.budget_stopped = True
                    cand.decision = "skipped"
                    cand.reasons = ["not evaluated: --max-cost-usd was reached"]
                    continue
                ev = await self.evaluate(cand.id, cand.version)
                cand.evaluation = ev.doc
                missing = sum(
                    r.category == MISSING
                    for run in ev.runs.values()
                    for r in run.records
                )
                if self.budget_stopped and missing:
                    # Cut short by the budget: judging partial scores would
                    # compare different task sets.
                    cand.decision = "skipped"
                    cand.reasons = [
                        f"--max-cost-usd was reached during its evaluation "
                        f"({missing} trial(s) did not run)"
                    ]
                    break
                evaluated[cand.id] = ev
                breach = self.infra_breach(ev)
                if breach:
                    cand.decision = "revert"
                    cand.reasons = [breach]
                    infra_stop = breach
                    break
                self.save()
            for cand in candidates:
                if cand.decision == "pending" and cand.id not in evaluated:
                    cand.decision = "skipped"
                    cand.reasons = ["not evaluated: the climb stopped first"]
            gains: dict[str, float] = {}
            for cand in candidates:
                if cand.id not in evaluated or cand.decision == "revert":
                    continue
                decision, gain = self.judge(current, evaluated[cand.id], cand)
                cand.decision = "keep" if decision.keep else "revert"
                cand.reasons = decision.reasons
                if decision.keep and gain is not None:
                    gains[cand.id] = gain
            kept_id = max(gains, key=lambda k: gains[k]) if gains else None
            for cand in candidates:
                if cand.decision == "keep" and cand.id != kept_id:
                    cand.decision = "revert"
                    cand.reasons = [
                        *cand.reasons,
                        f"{kept_id} gained more on train this round",
                    ]
            if kept_id is not None:
                kept = next(c for c in candidates if c.id == kept_id)
                current = evaluated[kept_id]
                current_round = round_no
                stall = 0
                self.current_commit = self.history.commit(
                    self.store.path(current.version),
                    [s.name for s in self.store.specs],
                    _commit_message(kept),
                )
                round_doc.kept = kept_id
                round_doc.version_after = current.version
            else:
                stall += 1
            round_doc.train_after = current.doc.train.score
            round_doc.test_after = current.doc.test.score
            round_doc.cost_usd = round(self.spent - spent_before, 6)
            self.best_round = current_round
            self.save()
            if infra_stop:
                stop = ("infra", infra_stop)
                break
            if self.budget_stopped:
                stop = ("budget", f"--max-cost-usd {cfg.max_cost_usd:g} was reached")
                break
            if stall >= cfg.stall_rounds:
                stop = (
                    "stalled",
                    f"nothing was kept in {stall} round(s) in a row (--stall-rounds "
                    f"{cfg.stall_rounds})",
                )
                break
        self.best_round = current_round
        if stop[0] == "stalled" or (cfg.analyze_at_end and stop[0] == "rounds"):
            await self.analyze(
                baseline, current, "stall" if stop[0] == "stalled" else "end"
            )
        status = "stopped" if stop[0] in ("budget", "infra") else "finished"
        return self.finish(status, stop[0], stop[1], baseline, current)

    async def analyze(
        self,
        baseline: _Evaluated,
        current: _Evaluated,
        trigger: Literal["stall", "end"],
    ) -> None:
        assert self.doc is not None
        ws = build_workspace(
            self.run_dir / "proposer" / "analysis" / "workspace",
            mode="analyze",
            version_dir=self.store.path(current.version),
            specs=self.store.specs,
            train=current.runs["train"],
            train_dirs=self.taskset.dirs,
            scores=self.scores_for_proposer(baseline, current),
            history=self.history_for_proposer(),
            objective=self.cfg.objective,
            max_failures=max(self.proposer.max_failures, 48),
            extra_instructions=self.proposer.extra_instructions,
        )
        seen = self.exposure_doc(ws, "analysis")
        outcome = await run_proposer(
            ws,
            mode="analyze",
            settings=self.proposer,
            task_dir=self.run_dir
            / "proposer"
            / "analysis"
            / "task"
            / "hillclimb-analyze",
            jobs_dir=self.run_dir / "proposer" / "analysis",
        )
        if outcome.cost_usd is not None:
            self.proposer_usd += outcome.cost_usd
        analysis = rec.AnalysisDoc(
            trigger=trigger,
            status=outcome.status,
            error=outcome.error,
            rollout_dir=self.rel(outcome.rollout_dir),
            workspace_dir=self.rel(ws.root),
            exposure=seen,
            cost_usd=outcome.cost_usd,
        )
        out = outcome.output or {}
        if outcome.status == "ok":
            summary = out.get("summary")
            analysis.summary = summary if isinstance(summary, str) else None
            dropped = 0
            for item in out.get("failures") or []:
                if not isinstance(item, dict):
                    dropped += 1
                    continue
                category = item.get("category")
                ident = str(item.get("id") or "")
                explanation = item.get("explanation")
                if category not in rec.ANALYSIS_CATEGORIES or not isinstance(
                    explanation, str
                ):
                    dropped += 1
                    continue
                analysis.failures.append(
                    rec.AnalysisFailureDoc(
                        id=ident,
                        task=ident.split("/", 1)[0] or None,
                        category=category,
                        explanation=explanation,
                    )
                )
            if dropped:
                self.warn(f"analysis: {dropped} malformed failure entries were dropped")
            counts: dict[str, int] = dict.fromkeys(rec.ANALYSIS_CATEGORIES, 0)
            for f in analysis.failures:
                counts[f.category] += 1
            analysis.counts = counts
            recs = out.get("recommendations")
            analysis.recommendations = (
                [str(r) for r in recs][:20] if isinstance(recs, list) else []
            )
        self.doc.analysis = analysis
        self.save()

    def finish(
        self,
        status: Literal["finished", "refused", "stopped"],
        reason: StopReason,
        detail: str,
        baseline: _Evaluated,
        best: _Evaluated,
    ) -> HillclimbResult:
        assert self.doc is not None
        self.doc.status = status
        self.doc.stop = rec.StopDoc(reason=reason, detail=detail)
        self.doc.best = self.best_doc(baseline, best)
        self.save()
        logger.info("hillclimb %s: %s", status, detail)
        return HillclimbResult(self.run_dir, self.doc)

    def best_doc(self, baseline: _Evaluated, best: _Evaluated) -> rec.BestDoc:
        assert self.doc is not None
        cfg = self.cfg
        samples = cfg.bootstrap_samples
        is_baseline = best.version == baseline.version
        d_train = stats.paired_delta(
            baseline.values("train"),
            best.values("train"),
            samples=samples,
            seed=_seed(cfg.seed, "best:train"),
        )
        d_test = stats.paired_delta(
            baseline.values("test"),
            best.values("test"),
            samples=samples,
            seed=_seed(cfg.seed, "best:test"),
        )
        cost_change = _cost_change(baseline, best)
        verdict = _verdict(
            objective=cfg.objective,
            is_baseline=is_baseline,
            baseline=baseline,
            best=best,
            d_test=d_test,
            cost_delta=stats.paired_delta(
                baseline.values("test", "cost"),
                best.values("test", "cost"),
                samples=samples,
                seed=_seed(cfg.seed, "best:test:cost"),
            ),
            test_noise=self.doc.noise_gate.test.noise_95
            if self.doc.noise_gate
            else None,
            gated=bool(self.doc.noise_gate and self.doc.noise_gate.passed),
        )
        candidate = None if is_baseline else best.id
        return rec.BestDoc(
            version=best.version,
            round=self.best_round if not is_baseline else 0,
            candidate=candidate,
            surface_dir=self.rel(self.store.path(best.version)) or "",
            git_commit=self.current_commit if not is_baseline else self.baseline_commit,
            train=best.doc.train.score,
            test=best.doc.test.score,
            train_delta_vs_baseline=_delta(d_train),
            test_delta_vs_baseline=_delta(d_test),
            cost_change_vs_baseline=rec.CostChangeDoc(
                train=cost_change["train"], test=cost_change["test"]
            ),
            verdict=verdict,
        )


def _cost_change(
    before: _Evaluated, after: _Evaluated
) -> dict[SplitName, float | None]:
    """Relative change in USD per scored trial, per split (negative = cheaper)."""
    out: dict[SplitName, float | None] = {}
    for split in SPLITS:
        a = stats.score(before.values(split, "cost"))
        b = stats.score(after.values(split, "cost"))
        out[split] = (b - a) / a if a and b is not None else None
    return out


def _fmt_ci(ci: stats.Interval | None, signed: bool = False) -> str:
    if ci is None:
        return "n/a"
    spec = "+.3f" if signed else ".3f"
    return f"[{ci.low:{spec}}, {ci.high:{spec}}]"


def _verdict(
    *,
    objective: Objective,
    is_baseline: bool,
    baseline: _Evaluated,
    best: _Evaluated,
    d_test: stats.Delta,
    cost_delta: stats.Delta,
    test_noise: float | None,
    gated: bool,
) -> rec.VerdictDoc:
    b, t = baseline.doc.test.score, best.doc.test.score

    def score_text(e: rec.EstimateDoc) -> str:
        if e.value is None:
            return "n/a"
        ci = f" [{e.ci.low:.3f}, {e.ci.high:.3f}]" if e.ci else ""
        return f"{e.value:.3f}{ci}"

    ungated = "" if gated else " This run was not gated on noise (--force)."
    if is_baseline:
        return rec.VerdictDoc(
            exceeds_noise=False,
            recommend_merge=False,
            text=(
                f"No patch was kept; the baseline stands (test {score_text(b)})."
                + ungated
            ),
        )
    head = (
        f"Best version {best.version} ({best.id}): test {score_text(t)} against the "
        f"baseline's {score_text(b)}"
    )
    if objective == "score":
        exceeds = bool(d_test.ci and d_test.ci.low > 0)
        delta = (
            f", a change of {d_test.value:+.3f} {_fmt_ci(d_test.ci, signed=True)}"
            if d_test.value is not None
            else ""
        )
        if exceeds and gated:
            tail = (
                " The gain exceeds noise: its 95% interval is above zero. "
                "Recommend merging."
            )
        elif exceeds:
            tail = (
                " Its 95% interval is above zero, but the run was not gated on "
                "noise (--force), so this alone does not justify merging."
            )
        else:
            tail = (
                " The gain is within noise: its 95% interval includes zero. "
                "Recommend against merging."
            )
        return rec.VerdictDoc(
            exceeds_noise=exceeds,
            recommend_merge=exceeds and gated,
            text=head + delta + "." + tail,
        )
    before = stats.score(baseline.values("test", "cost"))
    cut = (
        -cost_delta.value / before if cost_delta.value is not None and before else None
    )
    cheaper = bool(cost_delta.ci and cost_delta.ci.high < 0)
    held = bool(d_test.ci and test_noise is not None and d_test.ci.low >= -test_noise)
    parts = [head + "."]
    if cut is not None:
        parts.append(
            f" Test cost per trial fell {cut:.1%} (USD change "
            f"{cost_delta.value:+.4f} {_fmt_ci(cost_delta.ci, signed=True)})."
        )
    if cheaper and held and gated:
        parts.append(
            " The cut exceeds noise and the score held within noise. Recommend merging."
        )
    elif cheaper and held:
        parts.append(
            " The cut exceeds noise and the score held, but the run was not gated on "
            "noise (--force), so this alone does not justify merging."
        )
    else:
        parts.append(
            " The cut is within noise, or the score did not hold. "
            "Recommend against merging."
        )
    return rec.VerdictDoc(
        exceeds_noise=cheaper,
        recommend_merge=cheaper and held and gated,
        text="".join(parts),
    )


def _commit_message(cand: rec.CandidateDoc) -> str:
    title = cand.change or cand.root_cause or "surface change"
    lines = [f"{cand.id}: {title}"[:200], ""]
    if cand.root_cause:
        lines += [f"Root cause: {cand.root_cause}", ""]
    if cand.rationale:
        lines += [cand.rationale, ""]
    for label, d in (("train", cand.train_delta), ("test", cand.test_delta)):
        if d is not None and d.value is not None:
            ci = f" [{d.ci.low:+.3f}, {d.ci.high:+.3f}]" if d.ci else ""
            lines.append(f"{label} {d.value:+.3f}{ci}")
    return "\n".join(lines).strip() + "\n"


def _llm_judged(task_dir: Path) -> bool:
    try:
        from benchflow.task import Task

        return Task(task_dir).config.verifier.type == "llm-judge"
    except Exception:
        return False


def _version() -> str:
    import benchflow

    return benchflow.__version__


async def ahillclimb(
    config: HillclimbConfig | None = None, **kwargs: Any
) -> HillclimbResult:
    """Run a climb (async). Takes a :class:`HillclimbConfig` or its fields."""
    if config is None:
        config = HillclimbConfig(**kwargs)
    elif kwargs:
        raise TypeError("pass either a HillclimbConfig or keyword arguments, not both")
    return await _Climb(config).run()


def hillclimb(config: HillclimbConfig | None = None, **kwargs: Any) -> HillclimbResult:
    """Run a climb (blocking; also works inside a running event loop).

    ``bf.hillclimb(tasks=..., surface=..., out=..., agent=..., model=...)``
    returns a :class:`HillclimbResult`; the run folder holds
    ``hillclimb.json``, ``report.html``, ``surface-history/`` and every
    evaluation as normal BenchFlow jobs. See docs/hillclimb.md.
    """
    from benchflow.batch import run_blocking

    return run_blocking(lambda: ahillclimb(config, **kwargs))


def summarize(result: HillclimbResult) -> str:
    """A few lines for a terminal."""
    doc = result.record
    lines = [f"hillclimb {doc.status}: {doc.stop.detail if doc.stop else ''}"]
    if doc.baseline:
        lines.append(
            f"baseline: train {_score_line(doc.baseline.train.score)}, "
            f"test {_score_line(doc.baseline.test.score)}"
        )
    for rd in doc.rounds:
        for c in rd.candidates:
            dt = c.train_delta.value if c.train_delta else None
            ds = c.test_delta.value if c.test_delta else None
            nums = (
                f" train {dt:+.3f}, test {ds:+.3f}"
                if dt is not None and ds is not None
                else ""
            )
            lines.append(f"  {c.id}: {c.decision}{nums} ({'; '.join(c.reasons)[:160]})")
    if doc.best:
        lines.append(doc.best.verdict.text)
    if doc.analysis and doc.analysis.counts:
        counts = ", ".join(f"{k} {v}" for k, v in doc.analysis.counts.items() if v)
        lines.append(f"remaining train failures by cause: {counts}")
    lines.append(f"cost: ${doc.cost.total_usd:.2f}")
    lines.append(f"record: {result.run_dir / rec.RECORD_FILE}")
    lines.append(f"report: {result.report}")
    return "\n".join(lines)


def _score_line(e: rec.EstimateDoc) -> str:
    if e.value is None:
        return "n/a"
    ci = f" [{e.ci.low:.3f}, {e.ci.high:.3f}]" if e.ci else ""
    return f"{e.value:.3f}{ci}"


def load(path: str | Path) -> HillclimbResult:
    """Read a finished run folder back."""
    path = Path(path)
    run_dir = path.parent if path.is_file() else path
    return HillclimbResult(run_dir.resolve(), rec.load_record(run_dir))
