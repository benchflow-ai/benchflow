"""``hillclimb.json``: the versioned record of one ``bench hillclimb`` run.

The document has ``kind: "benchflow.hillclimb"`` and ``schema_version``. The
pydantic models below both build it and generate its JSON Schema, committed at
``docs/reference/schemas/benchflow-hillclimb.v1.schema.json``; a test fails
when the file is stale (regenerate with
``python -m benchflow.hillclimbing.record docs/reference/schemas``).

Versioning follows ``benchflow.job_export``: a new optional field keeps the
version, and removing or renaming a field, or changing its meaning or type,
bumps ``schema_version`` and the file name. So that a newer writer's optional
fields do not fail an older reader's validation, the schema allows properties
it does not list (the models refuse them when *building* a document); readers
ignore fields they do not know.

Paths in the document are relative to the run folder unless they say
otherwise.

>>> json_schema()["properties"]["kind"]["const"]
'benchflow.hillclimb'
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
KIND = "benchflow.hillclimb"
SCHEMA_FILENAME = f"benchflow-hillclimb.v{SCHEMA_VERSION}.schema.json"
SCHEMA_ID = f"https://benchflow.ai/schemas/{SCHEMA_FILENAME}"
RECORD_FILE = "hillclimb.json"

ANALYSIS_CATEGORIES = (
    "ambiguous_task",
    "grader_bug",
    "infrastructure",
    "capability_gap",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntervalDoc(_Model):
    low: float
    high: float


class EstimateDoc(_Model):
    value: float | None = Field(description="the task-averaged mean")
    se: float | None = Field(
        description="standard error, two-stage bootstrap over tasks and trials"
    )
    ci: IntervalDoc | None = Field(description="95% percentile bootstrap interval")
    tasks: int = Field(description="tasks with at least one scored trial")
    trials: int = Field(description="scored trials")


class DeltaDoc(_Model):
    value: float | None = Field(description="B minus A over the paired tasks")
    se: float | None
    ci: IntervalDoc | None
    paired_tasks: int
    samples: int = Field(description="bootstrap replicates")
    method: str = "paired two-stage bootstrap over tasks and trials"


class TaskResultDoc(_Model):
    task: str
    rewards: list[float | None] = Field(
        description="one per trial, in trial order; null = infrastructure error"
    )
    mean_reward: float | None
    cost_usd: float | None = Field(description="mean USD per scored trial")


class SplitResultDoc(_Model):
    split: Literal["train", "test"]
    job_dir: str = Field(
        description="the evaluation folder: trial-NN/ BenchFlow jobs inside"
    )
    tasks: int
    trials_per_task: int
    attempted_trials: int
    score: EstimateDoc
    cost: EstimateDoc | None = Field(
        None, description="USD per scored trial, when usage was priced"
    )
    cost_usd_total: float | None = Field(
        None, description="USD over every trial of this split, errored ones included"
    )
    usd_unknown_trials: int = 0
    infra_errors: int = Field(
        description="trials left out of the score: unscored or missing"
    )
    infra_error_rate: float
    infra_error_categories: dict[str, int] = Field(default_factory=dict)
    per_task: list[TaskResultDoc] = Field(default_factory=list)


class EvaluationDoc(_Model):
    id: str = Field(description="baseline, or the candidate id (r01-c1)")
    version: str = Field(description="the surface version it ran (v000, ...)")
    train: SplitResultDoc
    test: SplitResultDoc


class MountDoc(_Model):
    sandbox_path: str = Field(description="where the folder appeared in the sandbox")
    read_only: bool
    files: int
    bytes: int


class ExposureDoc(_Model):
    """What one optimizer sandbox received, checked against the test split."""

    mounts: list[MountDoc] = Field(
        description="every folder uploaded into the sandbox; nothing else was"
    )
    network: Literal["none", "open"] = Field(
        description="none: no egress except the model provider"
    )
    manifest: str = Field(description="every uploaded file with its size and sha256")
    train_tasks: list[str] = Field(description="train tasks whose instructions it held")
    failures: list[str] = Field(
        description="failed train trials it held (task/trial-NN)"
    )
    infra_errors: list[str] = Field(default_factory=list)
    test_tasks: int = Field(description="tasks in the test split")
    test_tasks_in_paths: list[str] = Field(
        description="test task names found in any uploaded path (none expected)"
    )
    test_instructions_in_files: list[str] = Field(
        description="test tasks whose instruction text was found in an uploaded file "
        "(none expected; a hit means a train task duplicates a test task)"
    )


class ProposerDoc(_Model):
    status: Literal["ok", "failed"]
    error: str | None = None
    rollout_dir: str | None = Field(
        None, description="the proposer rollout (a normal BenchFlow trial folder)"
    )
    workspace_dir: str | None = Field(
        None,
        description="exactly what was uploaded into the proposer sandbox, for audit",
    )
    exposure: ExposureDoc | None = None
    reward: float | None = Field(
        None, description="1 when the proposer left a well-formed proposal"
    )
    cost_usd: float | None = None


class DiffStatsDoc(_Model):
    files_changed: int
    added: int
    removed: int


class LeakCheckDoc(_Model):
    mode: Literal["reject", "warn", "off"]
    status: Literal["clean", "flagged", "skipped"]
    matches: list[dict[str, str]] = Field(
        default_factory=list,
        description="runs of words the patch copies from train instructions or verifier output",
    )


class CostChangeDoc(_Model):
    train: float | None = Field(
        description="relative change in USD per trial on train (negative = cheaper)"
    )
    test: float | None


class CandidateDoc(_Model):
    id: str
    round: int
    base_version: str
    version: str | None
    proposer: ProposerDoc
    root_cause: str | None = None
    change: str | None = None
    rationale: str | None = None
    evidence: list[str] = Field(default_factory=list)
    diff: str = ""
    diff_truncated: bool = False
    diff_stats: DiffStatsDoc | None = None
    leak_check: LeakCheckDoc | None = None
    evaluation: EvaluationDoc | None = None
    train_delta: DeltaDoc | None = None
    test_delta: DeltaDoc | None = None
    cost_change: CostChangeDoc | None = None
    decision: Literal["keep", "revert", "invalid", "pending", "skipped"] = Field(
        description="keep or revert (evaluated), invalid (no usable patch), pending "
        "(being evaluated), skipped (not evaluated: the climb stopped first)"
    )
    reasons: list[str] = Field(default_factory=list)


class RoundDoc(_Model):
    round: int
    base_version: str
    candidates: list[CandidateDoc]
    kept: str | None = Field(description="the kept candidate's id, or null")
    version_after: str
    train_after: EstimateDoc | None = Field(
        None, description="the current version's train score after the round"
    )
    test_after: EstimateDoc | None = None
    cost_usd: float | None = Field(None, description="USD spent in this round")


class GateSplitDoc(_Model):
    split: Literal["train", "test"]
    value: float | None
    se: float | None
    noise_se: float | None = Field(
        description="SE of a rerun's score difference, two-stage bootstrap"
    )
    noise_95: float | None = Field(description="1.96 x noise_se")
    threshold: float | None = Field(
        description="noise_95 in the objective's unit (a fraction of cost for cost)"
    )
    tasks: int
    min_trials: int
    ok: bool
    reason: str | None = None


class SuggestionDoc(_Model):
    trials: int | None = None
    train_tasks: int | None = None
    test_tasks: int | None = None
    min_gain: float | None = None


class NoiseGateDoc(_Model):
    passed: bool
    forced: bool = Field(description="--force: climbed although the gate refused")
    objective: Literal["score", "cost"]
    min_gain: float
    train: GateSplitDoc
    test: GateSplitDoc
    message: str
    suggestion: SuggestionDoc


class ControlTaskDoc(_Model):
    task: str
    split: Literal["train", "test"]
    oracle_reward: float | None = None
    oracle_error: str | None = None
    oracle_ran: bool = False
    nop_reward: float | None = None
    nop_error: str | None = None
    flags: list[str] = Field(
        default_factory=list,
        description="oracle_fails, nop_passes, oracle_unscored, nop_unscored, no_oracle",
    )


class ControlsDoc(_Model):
    ran: bool
    skipped_reason: str | None = None
    oracle_job_dir: str | None = None
    nop_job_dir: str | None = None
    tasks: list[ControlTaskDoc] = Field(default_factory=list)
    grader_bugs: list[str] = Field(
        default_factory=list,
        description="tasks whose oracle does not pass or where doing nothing passes",
    )
    excluded: list[str] = Field(
        default_factory=list, description="tasks dropped by --exclude-broken-tasks"
    )
    judge_consistency: str | None = Field(
        None, description="what was done to check LLM-judged tasks' graders"
    )


class AnalysisFailureDoc(_Model):
    id: str
    task: str | None = None
    category: Literal[
        "ambiguous_task", "grader_bug", "infrastructure", "capability_gap"
    ]
    explanation: str


class AnalysisDoc(_Model):
    trigger: Literal["stall", "end"]
    status: Literal["ok", "failed"]
    error: str | None = None
    rollout_dir: str | None = None
    workspace_dir: str | None = None
    exposure: ExposureDoc | None = None
    summary: str | None = None
    failures: list[AnalysisFailureDoc] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    cost_usd: float | None = None


class VerdictDoc(_Model):
    exceeds_noise: bool = Field(
        description="the test gain's 95% interval is above zero"
    )
    recommend_merge: bool
    text: str


class BestDoc(_Model):
    version: str
    round: int = Field(description="0 for the baseline")
    candidate: str | None = None
    surface_dir: str
    git_commit: str | None = None
    train: EstimateDoc
    test: EstimateDoc
    train_delta_vs_baseline: DeltaDoc | None = None
    test_delta_vs_baseline: DeltaDoc | None = None
    cost_change_vs_baseline: CostChangeDoc | None = None
    verdict: VerdictDoc


class StopDoc(_Model):
    reason: Literal[
        "rounds", "stalled", "budget", "infra", "noise_gate", "error", "interrupted"
    ]
    detail: str


class CostDoc(_Model):
    total_usd: float = Field(description="USD over every rollout with a known cost")
    agent_usd: float
    proposer_usd: float
    usd_unknown_rollouts: int = Field(
        description="rollouts that reported no USD (a subscription login, or unpriced)"
    )
    max_cost_usd: float | None
    budget_stopped: bool = False


class SurfaceDoc(_Model):
    kind: Literal["skills", "prompt"]
    source: str = Field(description="the path given (absolute)")
    name: str = Field(description="its path inside a version folder")


class ProposerConfigDoc(_Model):
    agent: str
    model: str | None
    reasoning_effort: str | None
    environment: str
    timeout_sec: int
    image: str
    open_network: bool
    max_failures: int


class ConfigDoc(_Model):
    tasks: list[str] = Field(description="the task folders (absolute)")
    surfaces: list[SurfaceDoc]
    agent: str
    model: str | None
    reasoning_effort: str | None
    environment: str
    concurrency: int
    objective: Literal["score", "cost"]
    rounds: int
    trials: int
    min_gain: float
    max_cost_usd: float | None
    stall_rounds: int
    candidates: int
    max_infra_error_rate: float
    leak_check: Literal["reject", "warn", "off"]
    bootstrap_samples: int
    seed: int
    force: bool
    proposer: ProposerConfigDoc
    config_override: dict[str, Any] | None = None
    agent_env_keys: list[str] = Field(default_factory=list)


class SplitDoc(_Model):
    method: Literal["random", "stratified", "file"]
    seed: int | None
    test_frac: float | None
    stratify_by: str | None
    source: str | None = None
    train: list[str]
    test: list[str]
    strata: dict[str, str] = Field(default_factory=dict)


class PathsDoc(_Model):
    report: str
    split: str
    surfaces: str
    surface_history: str | None
    evals: str
    proposer: str


class HillclimbDoc(_Model):
    """The record of one ``bench hillclimb`` run."""

    kind: Literal["benchflow.hillclimb"] = KIND
    schema_version: Literal[1] = SCHEMA_VERSION
    status: Literal["running", "finished", "refused", "stopped", "failed"]
    benchflow_version: str
    created_at: str
    updated_at: str
    config: ConfigDoc
    split: SplitDoc
    controls: ControlsDoc | None = None
    baseline: EvaluationDoc | None = None
    noise_gate: NoiseGateDoc | None = None
    rounds: list[RoundDoc] = Field(default_factory=list)
    best: BestDoc | None = None
    analysis: AnalysisDoc | None = None
    stop: StopDoc | None = None
    cost: CostDoc
    paths: PathsDoc
    warnings: list[str] = Field(default_factory=list)


def _open(schema: Any) -> Any:
    """Drop ``additionalProperties: false`` everywhere (see the module docstring)."""
    if isinstance(schema, dict):
        return {
            k: _open(v)
            for k, v in schema.items()
            if not (k == "additionalProperties" and v is False)
        }
    if isinstance(schema, list):
        return [_open(v) for v in schema]
    return schema


def json_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of ``hillclimb.json``."""
    schema = HillclimbDoc.model_json_schema(mode="serialization")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        **_open(schema),
    }


def write_schema(directory: str | Path) -> Path:
    path = Path(directory) / SCHEMA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2, sort_keys=True) + "\n")
    return path


def dump(doc: HillclimbDoc) -> dict[str, Any]:
    return json.loads(doc.model_dump_json())


def write_record(doc: HillclimbDoc, run_dir: Path) -> Path:
    """Write ``hillclimb.json`` atomically (a reader never sees half a file)."""
    path = run_dir / RECORD_FILE
    text = json.dumps(dump(doc), indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=".hillclimb-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_record(path: str | Path) -> HillclimbDoc:
    """Read a ``hillclimb.json`` (or the run folder holding one)."""
    path = Path(path)
    if path.is_dir():
        path = path / RECORD_FILE
    return HillclimbDoc.model_validate_json(path.read_text())


if __name__ == "__main__":
    print(write_schema(sys.argv[1] if len(sys.argv) > 1 else "."))
