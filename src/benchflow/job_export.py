"""A stable, versioned JSON form of ``bf.load_trial`` / ``bf.load_job`` / ``bf.compare``.

Viewers and other tools read BenchFlow results through this format instead of
the on-disk layout, which varies across versions. Each document has
``kind`` (``benchflow.trial``, ``benchflow.job`` or ``benchflow.comparison``)
and ``schema_version``. The pydantic models below both build the documents
and generate their JSON Schemas, so the two cannot drift; the schema files are
committed under ``docs/reference/schemas/`` and a test fails when they are
stale (regenerate with ``python -m benchflow.job_export docs/reference/schemas``).

Versioning: a new optional field keeps the version; removing or renaming a
field, or changing its meaning or type, bumps ``schema_version`` and the
schema file name (``benchflow-job.v2.schema.json``). The published schemas
are open (no ``additionalProperties: false``), so a document with a field
added later still validates against the schema a reader already has.

>>> from benchflow.job_export import SCHEMA_VERSION, json_schema
>>> SCHEMA_VERSION, json_schema("job")["properties"]["kind"]["const"]
(1, 'benchflow.job')
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from benchflow.jobs import Comparison, Denominators, Job, Trial

SCHEMA_VERSION = 1
#: Bumped when optional fields are added within a schema_version; the
#: changelog is in docs/reference/json-export.md.
SCHEMA_MINOR: dict[str, int] = {"trial": 3, "job": 3, "comparison": 2, "run-summary": 0}
SCHEMA_ID_BASE = "https://benchflow.ai/schemas"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UsageExport(_Model):
    n_input_tokens: int | None = None
    n_output_tokens: int | None = None
    n_cache_read_tokens: int | None = None
    n_cache_creation_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    usage_source: str | None = Field(
        None, description="provider_response, agent_native_acp or unavailable"
    )
    cost_status: Literal["priced", "subscription", "unpriced", "unavailable"] | None = (
        Field(
            None,
            description="why cost_usd is or is not known (1.1): priced; subscription "
            "(the agent counted its own tokens on a login, which has no price); "
            "unpriced (tokens known, model not in the price table); unavailable "
            "(no usage recorded)",
        )
    )


class VerifierExport(_Model):
    reward_text: str | None = Field(None, description="verifier/reward.txt, stripped")
    reward_json: Any = Field(None, description="verifier/reward.json")
    stdout: str | None = Field(None, description="verifier/test-stdout.txt")
    stderr: str | None = Field(None, description="verifier/test-stderr.txt")
    ctrf: Any = Field(None, description="verifier/ctrf.json (test report)")
    reward_details: Any = Field(
        None, description="verifier/reward-details.json, per-check results (1.1)"
    )


class BranchChildExport(_Model):
    label: str
    status: str
    reward: float | None
    reward_source: str | None = Field(None, description="verifier or runner_return")
    path: str | None = Field(None, description="relative to the trial folder")
    fork_id: str | None
    parent_label: str | None = Field(
        None, description="the child a nested fork was made from; null at the top"
    )
    cost: dict[str, Any] | None = Field(None, description="the child's cost record")
    advantage: float | None = Field(
        None, description="reward minus the value of the child's fork"
    )


class ForkExport(_Model):
    id: str
    status: str | None
    value: float | None = Field(None, description="V(checkpoint): mean child reward")
    parent_restore: str | None
    children: list[BranchChildExport]
    snapshot: dict[str, Any] = Field(default_factory=dict)
    timing_sec: dict[str, Any] | None = None
    kind: str = Field("fork", description="fork, or retry (a retry from a checkpoint)")
    parent_node: str | None = None
    cost: dict[str, Any] | None = Field(
        None,
        description="tokens, usd, usd_known, wall, parent, children, sandbox seconds",
    )


class ReviewCriterionExport(_Model):
    name: str
    blocker: bool | None
    weight: int | None
    outcome: str | None
    score: int | None
    explanation: str


class ReviewExport(_Model):
    reviewer_model: str | None
    review_valid: bool
    scoring: dict[str, Any]
    summary: str
    criteria: list[ReviewCriterionExport]
    source: str = Field(description="the review_report.json it came from")


class RubricCriterionDefinitionExport(_Model):
    name: str
    kind: Literal["blocker", "scored", "legacy"] = Field(
        description="blocker: a pass/fail gate; scored: 0-2 times its weight; "
        "legacy: a v0.1 pass/fail/not_applicable criterion"
    )
    weight: int | None
    scale: list[str | int] = Field(description="the values a verdict can take")
    description: str | None = Field(
        None, description="null when only the criterion names were recorded"
    )
    guidance: str | None = None


class RubricDefinitionExport(_Model):
    contract: str = Field(description="v0.2 (blockers and weights) or v0.1")
    criteria: list[RubricCriterionDefinitionExport]
    path: str | None = Field(
        None, description="the rubric file: relative to the task for revisions"
    )
    sha256: str | None = Field(None, description="of the rubric file, when recorded")
    snapshot: str | None = Field(
        None, description="the copy the reviewer used, relative to the trial folder"
    )


class EvidenceRefExport(_Model):
    area: Literal["trial", "task", "workspace"] = Field(
        description="the reviewer's /evidence mount: the trial folder, the task, "
        "or the solver's final workspace"
    )
    path: str = Field(description="relative to that area")
    line: int | None = None


class RubricVerdictExport(_Model):
    name: str
    kind: Literal["blocker", "scored", "legacy"]
    weight: int | None
    outcome: str | None = Field(description="pass / fail (/ not_applicable)")
    score: int | None = Field(description="0-2 for scored criteria")
    points: int | None = Field(description="score x weight; null for blockers")
    max_points: int | None = Field(description="2 x weight; null for blockers")
    explanation: str = Field(description="the reviewer's reasoning, in full")
    evidence: list[EvidenceRefExport] = Field(
        description="the evidence files the explanation cites"
    )


class RubricReviewerExport(_Model):
    agent: str | None
    model: str | None
    reasoning_effort: str | None = Field(description="null when not recorded")
    environment: str | None
    run: str | None = Field(
        description="the reviewer's own run folder (its trajectory): relative to "
        "the trial folder for revisions, as recorded for audits"
    )


class RubricRewardExport(_Model):
    policy: str | None
    tests_pass: bool | None
    all_blockers_pass: bool | None
    failed_blockers: list[str]
    weighted_points: int | None
    max_weighted_points: int | None
    rubric_reward: float | None = Field(
        description="weighted_points / max_weighted_points (raw quality)"
    )
    gated_quality: float | None = Field(
        description="rubric_reward when tests and every blocker pass, else 0"
    )
    decision: str | None = Field(
        description="publishable / presentable_with_revisions / not_publishable"
    )
    verifier_reward: float | None = Field(description="the test script's reward")
    passed: bool | None
    reward: float | None = Field(
        description="the trial reward this review sets; null for audits and "
        "incomplete revisions"
    )
    formula: str = Field(description="the arithmetic, with this review's numbers")


class RubricReviewRecordExport(_Model):
    kind: Literal["revision", "audit"] = Field(
        description="revision: automatic scoring (bench eval run with a task "
        "rubric, bench eval score), which sets the reward; audit: a detached "
        "bench review, which does not"
    )
    id: str = Field(description="the scoring attempt, or the audit's report folder")
    source: str = Field(
        description="scoring/<attempt>.json in the trial, or the review_report.json"
    )
    current: bool = Field(description="the revision result.json's reward comes from")
    recorded_at: str | None
    status: Literal["complete", "error"]
    error: str | None
    review_valid: bool
    summary: str | None
    reviewer: RubricReviewerExport
    rubric: RubricDefinitionExport | None
    verdicts: list[RubricVerdictExport]
    reward: RubricRewardExport
    notes: list[str]


class SandboxExport(_Model):
    sandbox_id: str | None
    provider: str | None = Field(description="e.g. DaytonaSandbox, DockerSandbox")
    created_at: str | None


class TrialExport(_Model):
    kind: Literal["benchflow.trial"] = "benchflow.trial"
    schema_version: Literal[1] = SCHEMA_VERSION
    schema_minor: int = Field(
        SCHEMA_MINOR["trial"],
        description="optional fields added within schema_version; see the changelog",
    )
    source: Literal["result.json", "results.jsonl"]
    path: str = Field(description="trial folder, or the results.jsonl file")
    row: int | None = Field(None, description="row index for a results.jsonl trial")
    task_name: str
    rollout_name: str
    agent: str
    agent_name: str
    model: str | None
    reward: float | None = Field(description="rewards.reward; null when unscored")
    rewards: dict[str, Any] | None
    passed: bool
    score_outcome: Literal["passed", "failed", "errored", "verifier_errored"]
    execution: Literal["completed", "errored", "timed_out", "integration_failed"] = (
        Field(
            description="integration_failed (1.2): the agent did nothing because "
            "its integration broke; see integration_failure"
        )
    )
    assessment: Literal["scored", "error", "unscored"]
    control: Literal["oracle", "empty"] | None = Field(
        description="control runs check the task, not an agent"
    )
    attempts: int = Field(
        1,
        description="rollouts this trial took: 1, plus each retry (or resume "
        "re-run) of its task in an Evaluation job (1.3)",
    )
    error: str | None
    error_category: str | None
    verifier_error: str | None
    verifier_error_category: str | None
    n_tool_calls: int
    n_prompts: int
    usage: UsageExport
    started_at: str | None = Field(description="ISO 8601, local time as recorded")
    finished_at: str | None
    duration_sec: float | None
    timing: dict[str, Any]
    settings: dict[str, Any] = Field(description="what bf.compare checks; see SETTINGS")
    forks: list[ForkExport]
    checkpoints: list[dict[str, Any]]
    verifier: VerifierExport | None
    sandbox: SandboxExport | None = Field(
        None, description="sandbox.json, written when the sandbox was created (1.1)"
    )
    review: ReviewExport | None
    rubric_reviews: list[RubricReviewRecordExport] = Field(
        default_factory=list,
        description="every rubric review of the trial: scoring revisions oldest "
        "first, then detached audits (added in 1.1)",
    )
    scoring: dict[str, Any] | None = Field(
        None, description="the automatic reviewer's gate verdict"
    )
    integration_failure: dict[str, Any] | None = Field(
        None,
        description="why the agent did nothing useful when its integration broke: "
        "cause (agent_auth, agent_install, truncated_trajectory, empty_trajectory, "
        "immediate_exit, no_activity), evidence, evidence_source, activity counts, "
        "reward_withheld; detected: 'on read' for results written before 1.2 (1.2)",
    )
    trajectory: list[dict[str, Any]] | None = Field(
        None, description="ACP events; null when not included in the export"
    )


class DenominatorsExport(_Model):
    attempted: int
    scored: int
    assessment_errors: int
    unscored: int
    execution_errors: int
    passed: int
    mean_reward: float | None
    pass_rate_scored: float | None
    pass_rate_attempted: float | None
    clean_scored: int
    clean_passed: int
    pass_rate_clean: float | None
    controls_excluded: int
    integration_failures: int = Field(
        0,
        description="attempted runs whose agent integration broke; also counted in "
        "unscored and execution_errors (1.2)",
    )


class SolveRatesExport(_Model):
    """pass@k, pass^k and the solve rate (``benchflow.pass_at_k``)."""

    success_rule: str
    solve_threshold: float | None = Field(
        description="null: a trial succeeds when it passed; else reward >= threshold"
    )
    tasks: int
    trials: int = Field(description="scored trials (the samples)")
    unscored: int = Field(description="left out of n, not counted as failures")
    controls_excluded: int
    min_trials_per_task: int
    max_trials_per_task: int
    solve_rate: float | None
    nonbinary_rewards: int
    ks: list[int]
    pass_at_k: dict[str, float | None]
    pass_hat_k: dict[str, float | None]
    tasks_at_k: dict[str, int]
    tasks_short_at_k: dict[str, int] = Field(
        description="tasks with fewer than k scored trials, left out of that k"
    )
    caveats: list[str]


class GroupExport(_Model):
    agent: str | None
    model: str | None
    denominators: DenominatorsExport


class InterruptedExport(_Model):
    path: str
    task_name: str | None = Field(description="from the attempt's task_path")
    agent: str | None
    model: str | None
    started_at: str | None
    sandbox_id: str | None = Field(
        description="set when a sandbox was created; it may not have been deleted"
    )


class JobExport(_Model):
    kind: Literal["benchflow.job"] = "benchflow.job"
    schema_version: Literal[1] = SCHEMA_VERSION
    schema_minor: int = Field(
        SCHEMA_MINOR["job"],
        description="optional fields added within schema_version; see the changelog",
    )
    path: str
    job_kind: Literal["evaluation", "branch", "directory"]
    denominators: DenominatorsExport = Field(
        description="agent runs; controls left out"
    )
    denominators_with_controls: DenominatorsExport
    cost_usd: float | None
    solve_rates: SolveRatesExport | None = Field(
        None, description="agent runs; default ks and success rule"
    )
    summary: dict[str, Any] | None = Field(None, description="the job's summary.json")
    groups: list[GroupExport] = Field(
        default_factory=list,
        description="agent-run denominators per agent and model (1.1)",
    )
    interrupted: list[InterruptedExport] = Field(
        default_factory=list,
        description="attempt folders that never wrote result.json (1.1)",
    )
    error_categories: dict[str, int] = Field(
        default_factory=dict,
        description="agent runs that errored, by error_category (1.1)",
    )
    timing_totals: dict[str, float] = Field(
        default_factory=dict,
        description="seconds per timing phase, summed over agent runs (1.1)",
    )
    trials: list[TrialExport]


class SettingCheckExport(_Model):
    setting: str
    match: bool | None = Field(description="null when neither side recorded it")
    a: list[Any] | None = None
    b: list[Any] | None = None


class ComparisonRowExport(_Model):
    task: str
    status: Literal["paired", "only_a", "only_b"]
    n_a: int
    n_b: int
    scored_a: int
    scored_b: int
    reward_a: float | None
    reward_b: float | None
    delta: float | None = Field(description="reward_b - reward_a")
    checks: list[SettingCheckExport]
    group: dict[str, Any] = Field(
        default_factory=dict, description="the row's values of the by keys (1.1)"
    )


class SettingMismatchExport(_Model):
    task: str
    setting: str
    a: list[Any]
    b: list[Any]


class ComparisonSideExport(_Model):
    label: str
    path: str
    denominators: DenominatorsExport
    solve_rates: SolveRatesExport | None = None


class ComparisonSummaryExport(_Model):
    tasks: int
    paired: int
    only_a: int
    only_b: int
    both_scored: int
    same_reward: int
    b_higher: int
    b_lower: int
    mean_delta: float | None
    one_run_per_side: bool
    setting_mismatches: int


class ComparisonExport(_Model):
    kind: Literal["benchflow.comparison"] = "benchflow.comparison"
    schema_version: Literal[1] = SCHEMA_VERSION
    schema_minor: int = Field(
        SCHEMA_MINOR["comparison"],
        description="optional fields added within schema_version; see the changelog",
    )
    by: list[str] = Field(
        default_factory=list, description="extra pairing keys besides task (1.1)"
    )
    a_paired: DenominatorsExport | None = Field(
        None, description="side A over the tasks both sides ran (1.1)"
    )
    b_paired: DenominatorsExport | None = None
    a: ComparisonSideExport
    b: ComparisonSideExport
    include_controls: bool
    vary: list[str]
    summary: ComparisonSummaryExport
    rows: list[ComparisonRowExport]
    mismatches: list[SettingMismatchExport]
    caveats: list[str]


class GateExport(_Model):
    fail_under: float | None = Field(None, description="--fail-under, a pass rate")
    fail_on: list[str] = Field(default_factory=list, description="--fail-on kinds")
    failed: list[str] = Field(
        default_factory=list, description="one line per failed gate check"
    )


class RunSummaryExport(_Model):
    """The result of one ``bench eval run`` (``--summary-out``), for CI."""

    kind: Literal["benchflow.run-summary"] = "benchflow.run-summary"
    schema_version: Literal[1] = SCHEMA_VERSION
    job_dir: str | None
    job_name: str
    total: int
    passed: int
    failed: int
    errored: int
    verifier_errored: int
    timed_out: int = Field(description="trials whose agent timed out (scored or not)")
    pass_rate: float = Field(description="passed / total")
    mean_reward: float | None
    reused: int = Field(
        description="finished tasks reused from an earlier run (resume)"
    )
    ran: int = Field(description="tasks run by this invocation")
    gate: GateExport
    exit_code: int = Field(description="the process exit code of the run")
    budget: dict[str, Any] | None = Field(
        None,
        description=(
            "summary.json's budget block when the job had a budget cap "
            "(caps, spent, stopped, reason, cancelled, not_started)"
        ),
    )


DocumentKind = Literal["trial", "job", "comparison", "run-summary"]
KINDS: tuple[DocumentKind, ...] = ("trial", "job", "comparison", "run-summary")

_MODELS: dict[str, type[_Model]] = {
    "trial": TrialExport,
    "job": JobExport,
    "comparison": ComparisonExport,
    "run-summary": RunSummaryExport,
}


def json_schema(kind: DocumentKind) -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of one document kind.

    The schema is open: it does not refuse properties it does not list, so a
    reader validating with it keeps working when a later release adds an
    optional field within the same ``schema_version`` (the versioning rule in
    the module docstring). The models that build the documents stay strict
    (``extra="forbid"``), so BenchFlow itself cannot write an undeclared field.
    """
    schema = _open(_MODELS[kind].model_json_schema(mode="serialization"))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{SCHEMA_ID_BASE}/benchflow-{kind}.v{SCHEMA_VERSION}.schema.json",
        **schema,
    }


def _open(schema: Any) -> Any:
    """``schema`` without ``additionalProperties: false`` at any level."""
    if isinstance(schema, dict):
        return {
            key: _open(value)
            for key, value in schema.items()
            if not (key == "additionalProperties" and value is False)
        }
    if isinstance(schema, list):
        return [_open(value) for value in schema]
    return schema


def schema_filename(kind: str) -> str:
    return f"benchflow-{kind}.v{SCHEMA_VERSION}.schema.json"


def _safe(value: Any) -> Any:
    """Strict, finite JSON: NaN/inf become null, other objects become strings."""

    def fix(v: Any) -> Any:
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return None
        if isinstance(v, dict):
            return {str(k): fix(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [fix(x) for x in v]
        return v

    return fix(json.loads(json.dumps(value, default=str)))


def _denominators(d: Denominators) -> DenominatorsExport:
    return DenominatorsExport(
        attempted=d.attempted,
        scored=d.scored,
        assessment_errors=d.assessment_errors,
        unscored=d.unscored,
        execution_errors=d.execution_errors,
        passed=d.passed,
        mean_reward=d.mean_reward,
        pass_rate_scored=d.pass_rate_scored,
        pass_rate_attempted=d.pass_rate_attempted,
        clean_scored=d.clean_scored,
        clean_passed=d.clean_passed,
        pass_rate_clean=d.pass_rate_clean,
        controls_excluded=d.controls_excluded,
        integration_failures=d.integration_failures,
    )


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


CostStatus = Literal["priced", "subscription", "unpriced", "unavailable"]


def _cost_status(cost_usd: float | None, usage_source: str | None) -> CostStatus:
    if cost_usd is not None:
        return "priced"
    if usage_source == "agent_native_acp":
        return "subscription"
    if usage_source == "provider_response":
        return "unpriced"
    return "unavailable"


def _sandbox(trial_dir: Path) -> SandboxExport | None:
    data = _read_json_file(trial_dir / "sandbox.json")
    if not isinstance(data, dict):
        return None
    return SandboxExport(
        sandbox_id=_text(data.get("sandbox_id")),
        provider=_text(data.get("provider")),
        created_at=_text(data.get("created_at")),
    )


def _interrupted(folder: Path) -> InterruptedExport:
    config = _read_json_file(folder / "config.json")
    config = config if isinstance(config, dict) else {}
    sandbox = _read_json_file(folder / "sandbox.json")
    task_path = _text(config.get("task_path"))
    return InterruptedExport(
        path=str(folder),
        task_name=Path(task_path).name if task_path else folder.name.split("__")[0],
        agent=_text(config.get("agent")),
        model=_text(config.get("model")),
        started_at=_text(config.get("started_at")),
        sandbox_id=_text(sandbox.get("sandbox_id"))
        if isinstance(sandbox, dict)
        else None,
    )


def trial_export(
    trial: Trial, *, include_trajectory: bool = True, include_verifier: bool = True
) -> TrialExport:
    from benchflow.rubric_reviews import rubric_reviews

    r = trial.result
    record = r.to_record()
    review = trial.review
    return TrialExport(
        source=trial.source,
        path=str(trial.path),
        row=trial.row,
        task_name=r.task_name,
        rollout_name=r.rollout_name,
        agent=r.agent,
        agent_name=r.agent_name,
        model=r.model,
        reward=trial.reward,
        rewards=_safe(r.rewards) if r.rewards is not None else None,
        passed=r.passed,
        score_outcome=r.score_outcome,
        execution=trial.execution,
        assessment=trial.assessment,
        control=trial.control,
        attempts=len(trial.attempts),
        error=r.error,
        error_category=r.error_category,
        verifier_error=r.verifier_error,
        verifier_error_category=r.verifier_error_category,
        n_tool_calls=r.n_tool_calls,
        n_prompts=r.n_prompts,
        usage=UsageExport(
            n_input_tokens=r.n_input_tokens,
            n_output_tokens=r.n_output_tokens,
            n_cache_read_tokens=r.n_cache_read_tokens,
            n_cache_creation_tokens=r.n_cache_creation_tokens,
            total_tokens=r.total_tokens,
            cost_usd=r.cost_usd,
            usage_source=r.usage_source,
            cost_status=_cost_status(r.cost_usd, r.usage_source),
        ),
        started_at=record["started_at"],
        finished_at=record["finished_at"],
        duration_sec=record["duration_sec"],
        timing=_safe(trial.timing),
        settings=_safe(trial.settings()),
        forks=[
            ForkExport(
                id=f.id,
                status=f.status,
                value=f.value,
                parent_restore=f.parent_restore,
                children=[
                    BranchChildExport(
                        **c.to_record(),
                        cost=_safe(c.cost) if c.cost is not None else None,
                    )
                    for c in f.children
                ],
                snapshot=_safe(f.snapshot),
                timing_sec=_safe(f.timing_sec) if f.timing_sec is not None else None,
                kind=f.kind,
                parent_node=f.parent_node,
                cost=_safe(f.cost) if f.cost is not None else None,
            )
            for f in trial.forks
        ],
        checkpoints=_safe(trial.checkpoints),
        verifier=VerifierExport(
            **_safe(trial.verifier.__dict__),
            reward_details=_safe(
                _read_json_file(trial.path / "verifier" / "reward-details.json")
            ),
        )
        if include_verifier and trial.source == "result.json"
        else None,
        sandbox=_sandbox(trial.path) if trial.source == "result.json" else None,
        review=ReviewExport(
            reviewer_model=review.reviewer_model,
            review_valid=review.review_valid,
            scoring=_safe(dict(review.scoring)),
            summary=review.summary,
            criteria=[
                ReviewCriterionExport(
                    name=c.name,
                    blocker=c.blocker,
                    weight=c.weight,
                    outcome=c.outcome,
                    score=c.score,
                    explanation=c.explanation,
                )
                for c in review.criteria
            ],
            source=review.source,
        )
        if review is not None
        else None,
        rubric_reviews=[
            RubricReviewRecordExport.model_validate(_safe(entry))
            for entry in rubric_reviews(trial.path, trial.raw)
        ]
        if trial.source == "result.json"
        else [],
        scoring=r.scoring.to_dict() if r.scoring is not None else None,
        integration_failure=_safe(trial.integration_failure)
        if trial.integration_failure is not None
        else None,
        trajectory=_safe(trial.trajectory) if include_trajectory else None,
    )


def _error_categories(job: Job) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in job.agents():
        if t.result.error:
            category = t.result.error_category or "unknown"
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _timing_totals(job: Job) -> dict[str, float]:
    totals: dict[str, float] = {}
    for t in job.agents():
        for phase, seconds in (t.timing or {}).items():
            if isinstance(seconds, int | float) and not isinstance(seconds, bool):
                totals[phase] = round(totals.get(phase, 0.0) + seconds, 3)
    return totals


def job_export(
    job: Job, *, include_trajectories: bool = False, include_verifier: bool = True
) -> JobExport:
    return JobExport(
        path=str(job.path),
        job_kind=job.kind,
        denominators=_denominators(job.denominators()),
        denominators_with_controls=_denominators(
            job.denominators(include_controls=True)
        ),
        cost_usd=job.cost_usd,
        solve_rates=SolveRatesExport(**job.solve_rates().to_dict()),
        summary=_safe(job.summary) if job.summary is not None else None,
        groups=[
            GroupExport(
                agent=g.key.get("agent"),
                model=g.key.get("model"),
                denominators=_denominators(g.denominators),
            )
            for g in job.denominators_by(("agent", "model"))
        ],
        interrupted=[_interrupted(folder) for folder in job.interrupted],
        error_categories=_error_categories(job),
        timing_totals=_timing_totals(job),
        trials=[
            trial_export(
                t,
                include_trajectory=include_trajectories,
                include_verifier=include_verifier,
            )
            for t in job.trials
        ],
    )


def _solve_rates(rates: Any) -> SolveRatesExport | None:
    return None if rates is None else SolveRatesExport(**rates.to_dict())


def comparison_export(cmp: Comparison) -> ComparisonExport:
    s = cmp.summary
    return ComparisonExport(
        a=ComparisonSideExport(
            label=cmp.labels[0],
            path=str(cmp.job_a.path),
            denominators=_denominators(cmp.a),
            solve_rates=_solve_rates(cmp.solve_rates_a),
        ),
        b=ComparisonSideExport(
            label=cmp.labels[1],
            path=str(cmp.job_b.path),
            denominators=_denominators(cmp.b),
            solve_rates=_solve_rates(cmp.solve_rates_b),
        ),
        include_controls=cmp.include_controls,
        vary=list(cmp.vary),
        by=list(cmp.by),
        a_paired=_denominators(cmp.a_paired) if cmp.a_paired is not None else None,
        b_paired=_denominators(cmp.b_paired) if cmp.b_paired is not None else None,
        summary=ComparisonSummaryExport(**s.__dict__),
        rows=[
            ComparisonRowExport(
                task=r.task,
                status=r.status,
                n_a=r.n_a,
                n_b=r.n_b,
                scored_a=r.scored_a,
                scored_b=r.scored_b,
                reward_a=r.reward_a,
                reward_b=r.reward_b,
                delta=r.delta,
                checks=[
                    SettingCheckExport(
                        setting=c.setting, match=c.match, a=_safe(c.a), b=_safe(c.b)
                    )
                    for c in r.checks
                ],
                group=_safe(dict(r.group)),
            )
            for r in cmp.rows
        ],
        mismatches=[
            SettingMismatchExport(
                task=m.task, setting=m.setting, a=_safe(m.a), b=_safe(m.b)
            )
            for m in cmp.mismatches
        ],
        caveats=list(cmp.caveats),
    )


def run_summary_export(
    result: Any,
    *,
    job_dir: Path | None,
    timeouts: int,
    fail_under: float | None,
    fail_on: list[str],
    gate_failed: list[str],
    exit_code: int,
) -> RunSummaryExport:
    """The ``benchflow.run-summary`` document of an ``EvaluationResult``."""
    return RunSummaryExport(
        job_dir=str(job_dir) if job_dir is not None else None,
        job_name=result.job_name,
        total=result.total,
        passed=result.passed,
        failed=result.failed,
        errored=result.errored,
        verifier_errored=result.verifier_errored,
        timed_out=timeouts,
        pass_rate=result.score,
        mean_reward=result.mean_reward,
        reused=getattr(result, "reused", 0),
        ran=getattr(result, "ran", 0),
        gate=GateExport(fail_under=fail_under, fail_on=fail_on, failed=gate_failed),
        exit_code=exit_code,
        budget=getattr(result, "budget", None),
    )


def write_schemas(directory: str | Path) -> list[Path]:
    """Write every schema file into ``directory`` and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in KINDS:
        path = directory / schema_filename(kind)
        path.write_text(json.dumps(json_schema(kind), indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for written in write_schemas(sys.argv[1] if len(sys.argv) > 1 else "."):
        print(written)
