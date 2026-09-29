"""BenchFlow: run agents on benchmark tasks in sandboxes and score them.

Entry points (see docs/reference/python-api.md; runnable examples in
docs/examples/python-sdk/, starting with quickstart.py):

- One rollout: ``run_sync(RolloutConfig(task_path=..., agent=..., model=...,
  environment="docker" | "daytona"))`` blocks; ``await arun(...)`` is the
  async form (``run`` is the same function). The result is a
  ``RolloutResult`` (``reward``, ``passed``, ``trajectory``, ``rollout_dir``).
- Many configs: ``run_batch(configs, concurrency=4)`` / ``arun_batch`` /
  ``as_completed``, returning ``Results`` with CSV and JSONL export.
- Every task under a directory, with retries and resume:
  ``Evaluation(tasks_dir, jobs_dir, config=EvaluationConfig(...))``, then
  ``Evaluation.run_sync``, ``Evaluation.stream`` or ``Evaluation.resume(job_dir)``.
- Branch a run into scored children: ``branch(task, agent=..., children={...})``.
- Read and compare finished jobs: ``load_job(path)``, ``load_trial(path)``,
  ``compare(job_a, job_b)``.
- Train on a job while it runs: ``stream_rollouts(job_dir)`` /
  ``astream_rollouts`` yield each finished rollout with reward, group id and
  captured token ids (``bench train stream``).
- Re-score stored trials with a changed verifier: ``regrade(job, tasks_dir=...)``.
- Hill-climb a skills folder or prompt against a held-out test split:
  ``hillclimb(tasks=..., surface=..., out=...)`` (``bench hillclimb``,
  docs/hillclimb.md).
- Save configs for the CLI: ``Evaluation.to_yaml`` and ``RolloutConfig.to_yaml``
  (``bench eval run --config``); docs/reference/cli-python-parity.md maps every
  ``bench eval run`` / ``bench eval branch`` flag to Python.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("benchflow")
except PackageNotFoundError:
    __version__ = "0+unknown"

from benchflow._types import Role, Scene, Turn
from benchflow._utils.benchmark_repos import resolve_source
from benchflow._utils.yaml_loader import rollout_config_from_yaml
from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.adapters import (
    InspectAdapter,
    ORSAdapter,
    ors_tool_outputs_to_reward_events,
    to_inspect_task,
    to_ors_reward,
    write_ors_tool_outputs_jsonl,
)
from benchflow.agents.registry import (
    AGENTS,
    get_agent,
    infer_env_key_for_model,
    is_vertex_model,
    list_agents,
    register_agent,
)
from benchflow.batch import (
    Completed,
    Results,
    arun,
    arun_batch,
    as_completed,
    run_batch,
    run_sync,
)
from benchflow.branch_api import (
    BranchChildResult,
    BranchPlanError,
    BranchResult,
    ChildSpec,
    abranch,
    branch,
)
from benchflow.budget import Budget
from benchflow.contracts.user import (
    BaseUser,
    DocumentNudgeUser,
    FunctionUser,
    ModelDocumentNudgeUser,
    PassthroughUser,
    RoundResult,
)
from benchflow.environment.manifest import EnvironmentManifest, load_manifest
from benchflow.eval_regrade import RegradeSummary, aregrade, regrade
from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    RetryConfig,
)
from benchflow.hillclimbing import (
    HillclimbConfig,
    HillclimbResult,
    ahillclimb,
    hillclimb,
)
from benchflow.jobs import (
    Comparison,
    ComparisonRow,
    ComparisonSummary,
    Denominators,
    Fork,
    GroupDenominators,
    Job,
    SettingCheck,
    SettingMismatch,
    Trial,
    VerifierOutput,
    compare,
    load_job,
    load_results_jsonl,
    load_trial,
)
from benchflow.metrics import BenchmarkMetrics, collect_metrics
from benchflow.models import AgentInstallError, AgentTimeoutError, RolloutResult
from benchflow.monitor import (
    Monitor,
    MonitorConfig,
    MonitorNotImplementedError,
    MonitorResult,
)
from benchflow.review import (
    PublicationDecision,
    ReviewerConfig,
    ReviewReport,
    ReviewRubricError,
    ReviewScoring,
    run_reviews,
    score_weighted_review,
)
from benchflow.review import Rubric as ReviewRubric
from benchflow.review import RubricCriterion as ReviewRubricCriterion
from benchflow.review import load_rubric as load_review_rubric

# Rewards plane. Reward is the canonical node-based contract
# (``score(node) -> VerifyResult``); RewardFunc is the legacy path-based shape
# (``score(rollout_dir) -> float``) adapted into Reward via PathReward.
from benchflow.rewards import (
    CodeExecRewardFunc,
    Criterion,
    JudgeConfig,
    LLMJudgeRewardFunc,
    PathReward,
    Reward,
    RewardEvent,
    RewardFunc,
    Rubric,
    RubricConfig,
    ScoringConfig,
    StringMatchRewardFunc,
    TestRewardFunc,
    VerifyResult,
    load_rubric,
    load_rubric_json,
    load_rubric_toml,
)
from benchflow.rollout import (
    BashToolResult,
    Rollout,
    RolloutConfig,
    TaskRuntime,
    TaskRuntimeConfig,
    TaskRuntimeResult,
)
from benchflow.runtime import (
    Agent,
    Environment,
    Runtime,
    RuntimeConfig,
    RuntimeResult,
    run,
)  # bf.run() — supports Agent, RolloutConfig, and str calling conventions
from benchflow.sandbox import (
    SERVICES,
    ImageBuilder,
    ImageConfig,
    ImageRef,
    Sandbox,
    SandboxImage,
    SandboxSnapshotNotSupported,
    build_service_hooks,
    detect_services_from_dockerfile,
    register_service,
)

# Sandbox protocol (v0.4)
from benchflow.sandbox import ExecResult as SandboxExecResult
from benchflow.sandbox.protocol import ExecResult
from benchflow.sandbox.setup import stage_dockerfile_deps
from benchflow.sandbox.snapshot import (
    list_workspace_snapshots,
    workspace_restore,
    workspace_snapshot,
)
from benchflow.scenes import compile_scenes_to_steps
from benchflow.sdk import SDK
from benchflow.skills import SkillInfo, discover_skills, install_skill, parse_skill
from benchflow.task import (
    TASK_DOCUMENT_FILENAME,
    Task,
    TaskConfig,
    TaskDocument,
    TaskDocumentParseError,
    Verifier,
    VerifierResult,
    render_task_md_from_legacy,
)
from benchflow.trajectories.rollout_stream import (
    StreamedRollout,
    astream_rollouts,
    stream_rollouts,
)
from benchflow.trajectories.types import Trajectory

# Public API surface. Anything not in this list is implementation detail and
# may change without notice.
__all__ = [
    "__version__",
    "Reward",
    "Rubric",
    "RewardFunc",
    "RewardEvent",
    "PathReward",
    "VerifyResult",
    "TestRewardFunc",
    "LLMJudgeRewardFunc",
    "StringMatchRewardFunc",
    "CodeExecRewardFunc",
    "Criterion",
    "JudgeConfig",
    "RubricConfig",
    "ScoringConfig",
    "load_rubric",
    "load_rubric_json",
    "load_rubric_toml",
    "ReviewReport",
    "ReviewerConfig",
    "PublicationDecision",
    "ReviewRubric",
    "ReviewRubricCriterion",
    "ReviewRubricError",
    "ReviewScoring",
    "load_review_rubric",
    "run_reviews",
    "score_weighted_review",
    "Sandbox",
    "SandboxExecResult",
    "SandboxImage",
    "SandboxSnapshotNotSupported",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    "ExecResult",
    "Task",
    "TaskConfig",
    "TASK_DOCUMENT_FILENAME",
    "TaskDocument",
    "TaskDocumentParseError",
    "render_task_md_from_legacy",
    "Verifier",
    "VerifierResult",
    "ACPClient",
    "ACPSession",
    "AGENTS",
    "get_agent",
    "infer_env_key_for_model",
    "is_vertex_model",
    "list_agents",
    "register_agent",
    "Evaluation",
    "EvaluationConfig",
    "Budget",
    "EvaluationResult",
    "RetryConfig",
    "BenchmarkMetrics",
    "collect_metrics",
    "AgentInstallError",
    "AgentTimeoutError",
    "RolloutResult",
    # Monitor mode — scaffolded API surface (#386)
    "Monitor",
    "MonitorConfig",
    "MonitorResult",
    "MonitorNotImplementedError",
    "Agent",
    "Environment",
    "EnvironmentManifest",
    "load_manifest",
    "Runtime",
    "RuntimeConfig",
    "RuntimeResult",
    "run",
    "arun",
    "run_sync",
    "as_completed",
    "arun_batch",
    "run_batch",
    "Completed",
    "Results",
    "load_job",
    "stream_rollouts",
    "astream_rollouts",
    "StreamedRollout",
    "load_trial",
    "load_results_jsonl",
    "SettingCheck",
    "SettingMismatch",
    "GroupDenominators",
    "compare",
    "regrade",
    "aregrade",
    "RegradeSummary",
    "hillclimb",
    "ahillclimb",
    "HillclimbConfig",
    "HillclimbResult",
    "Job",
    "Trial",
    "Fork",
    "VerifierOutput",
    "Denominators",
    "Comparison",
    "ComparisonRow",
    "ComparisonSummary",
    "branch",
    "abranch",
    "BranchResult",
    "BranchChildResult",
    "BranchPlanError",
    "ChildSpec",
    "Role",
    "Scene",
    "Turn",
    "compile_scenes_to_steps",
    # Workspace snapshots (filesystem helper — NOT the Sandbox primitive, #384)
    "workspace_snapshot",
    "workspace_restore",
    "list_workspace_snapshots",
    "Rollout",
    "RolloutConfig",
    "BashToolResult",
    "TaskRuntime",
    "TaskRuntimeConfig",
    "TaskRuntimeResult",
    "rollout_config_from_yaml",
    "resolve_source",
    "BaseUser",
    "DocumentNudgeUser",
    "FunctionUser",
    "ModelDocumentNudgeUser",
    "PassthroughUser",
    "RoundResult",
    "SDK",
    "SERVICES",
    "build_service_hooks",
    "detect_services_from_dockerfile",
    "register_service",
    "stage_dockerfile_deps",
    "SkillInfo",
    "discover_skills",
    "install_skill",
    "parse_skill",
    "Trajectory",
    "InspectAdapter",
    "ORSAdapter",
    "ors_tool_outputs_to_reward_events",
    "to_inspect_task",
    "to_ors_reward",
    "write_ors_tool_outputs_jsonl",
]


# Pre-#384 names for the workspace snapshot helpers. They still resolve, with a
# DeprecationWarning, and are no longer in ``__all__``.
_DEPRECATED_ALIASES = {
    "snapshot": "workspace_snapshot",
    "restore": "workspace_restore",
    "list_snapshots": "list_workspace_snapshots",
}


def __getattr__(name: str):
    """Deprecated aliases, then lazy submodule resolution."""
    import importlib

    if name in _DEPRECATED_ALIASES:
        import warnings

        target = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"benchflow.{name} is deprecated; use benchflow.{target} "
            "(it snapshots a workspace directory, not a sandbox).",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[target]

    try:
        return importlib.import_module(f"benchflow.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"benchflow.{name}":
            raise
    raise AttributeError(f"module 'benchflow' has no attribute {name!r}")
