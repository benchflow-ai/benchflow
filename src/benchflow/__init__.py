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
- Save configs for the CLI: ``Evaluation.to_yaml`` and ``RolloutConfig.to_yaml``
  (``bench eval run --config``); docs/reference/cli-python-parity.md maps every
  ``bench eval run`` / ``bench eval branch`` flag to Python.
"""

# Every public name below resolves on first use (PEP 562, ``__getattr__`` at
# the bottom), so ``import benchflow`` and the ``bench`` CLI, which imports
# ``benchflow.cli`` and so this package first, load none of the SDK: ``bench
# --version`` used to import the whole package (rollouts, sandboxes, the ACP
# schema) to print one line. The imports under TYPE_CHECKING are the same
# names for type checkers and editors; ``_EXPORTS`` and ``_ALIASES`` are where
# they resolve at runtime, and tests/test_package_lazy_exports.py keeps the two
# lists equal and imports every name in ``__all__``.

import sys
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from benchflow.agents.errors import UsageLimitError
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
    "UsageLimitError",
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


# Where each public name lives: the module, then the names imported from it
# under the same name. ``_ALIASES`` are the names exported under another name.
_EXPORTS: dict[str, tuple[str, ...]] = {
    "benchflow._types": ("Role", "Scene", "Turn"),
    "benchflow._utils.benchmark_repos": ("resolve_source",),
    "benchflow._utils.yaml_loader": ("rollout_config_from_yaml",),
    "benchflow.acp.client": ("ACPClient",),
    "benchflow.acp.session": ("ACPSession",),
    "benchflow.adapters": (
        "InspectAdapter",
        "ORSAdapter",
        "ors_tool_outputs_to_reward_events",
        "to_inspect_task",
        "to_ors_reward",
        "write_ors_tool_outputs_jsonl",
    ),
    "benchflow.agents.errors": ("UsageLimitError",),
    "benchflow.agents.registry": (
        "AGENTS",
        "get_agent",
        "infer_env_key_for_model",
        "is_vertex_model",
        "list_agents",
        "register_agent",
    ),
    "benchflow.batch": (
        "Completed",
        "Results",
        "arun",
        "arun_batch",
        "as_completed",
        "run_batch",
        "run_sync",
    ),
    "benchflow.branch_api": (
        "BranchChildResult",
        "BranchPlanError",
        "BranchResult",
        "ChildSpec",
        "abranch",
        "branch",
    ),
    "benchflow.budget": ("Budget",),
    "benchflow.contracts.user": (
        "BaseUser",
        "DocumentNudgeUser",
        "FunctionUser",
        "ModelDocumentNudgeUser",
        "PassthroughUser",
        "RoundResult",
    ),
    "benchflow.environment.manifest": ("EnvironmentManifest", "load_manifest"),
    "benchflow.eval_regrade": ("RegradeSummary", "aregrade", "regrade"),
    "benchflow.evaluation": (
        "Evaluation",
        "EvaluationConfig",
        "EvaluationResult",
        "RetryConfig",
    ),
    "benchflow.jobs": (
        "Comparison",
        "ComparisonRow",
        "ComparisonSummary",
        "Denominators",
        "Fork",
        "GroupDenominators",
        "Job",
        "SettingCheck",
        "SettingMismatch",
        "Trial",
        "VerifierOutput",
        "compare",
        "load_job",
        "load_results_jsonl",
        "load_trial",
    ),
    "benchflow.metrics": ("BenchmarkMetrics", "collect_metrics"),
    "benchflow.models": ("AgentInstallError", "AgentTimeoutError", "RolloutResult"),
    "benchflow.monitor": (
        "Monitor",
        "MonitorConfig",
        "MonitorNotImplementedError",
        "MonitorResult",
    ),
    "benchflow.review": (
        "PublicationDecision",
        "ReviewerConfig",
        "ReviewReport",
        "ReviewRubricError",
        "ReviewScoring",
        "run_reviews",
        "score_weighted_review",
    ),
    "benchflow.rewards": (
        "CodeExecRewardFunc",
        "Criterion",
        "JudgeConfig",
        "LLMJudgeRewardFunc",
        "PathReward",
        "Reward",
        "RewardEvent",
        "RewardFunc",
        "Rubric",
        "RubricConfig",
        "ScoringConfig",
        "StringMatchRewardFunc",
        "TestRewardFunc",
        "VerifyResult",
        "load_rubric",
        "load_rubric_json",
        "load_rubric_toml",
    ),
    "benchflow.rollout": (
        "BashToolResult",
        "Rollout",
        "RolloutConfig",
        "TaskRuntime",
        "TaskRuntimeConfig",
        "TaskRuntimeResult",
    ),
    "benchflow.runtime": (
        "Agent",
        "Environment",
        "Runtime",
        "RuntimeConfig",
        "run",
    ),
    "benchflow.sandbox": (
        "SERVICES",
        "ImageBuilder",
        "ImageConfig",
        "ImageRef",
        "Sandbox",
        "SandboxImage",
        "SandboxSnapshotNotSupported",
        "build_service_hooks",
        "detect_services_from_dockerfile",
        "register_service",
    ),
    "benchflow.sandbox.protocol": ("ExecResult",),
    "benchflow.sandbox.setup": ("stage_dockerfile_deps",),
    "benchflow.sandbox.snapshot": (
        "list_workspace_snapshots",
        "workspace_restore",
        "workspace_snapshot",
    ),
    "benchflow.scenes": ("compile_scenes_to_steps",),
    "benchflow.sdk": ("SDK",),
    "benchflow.skills": (
        "SkillInfo",
        "discover_skills",
        "install_skill",
        "parse_skill",
    ),
    "benchflow.task": (
        "TASK_DOCUMENT_FILENAME",
        "Task",
        "TaskConfig",
        "TaskDocument",
        "TaskDocumentParseError",
        "Verifier",
        "VerifierResult",
        "render_task_md_from_legacy",
    ),
    "benchflow.trajectories.rollout_stream": (
        "StreamedRollout",
        "astream_rollouts",
        "stream_rollouts",
    ),
    "benchflow.trajectories.types": ("Trajectory",),
}
_ALIASES: dict[str, tuple[str, str]] = {
    "ReviewRubric": ("benchflow.review", "Rubric"),
    "ReviewRubricCriterion": ("benchflow.review", "RubricCriterion"),
    "load_review_rubric": ("benchflow.review", "load_rubric"),
    "SandboxExecResult": ("benchflow.sandbox", "ExecResult"),
}
_LAZY: dict[str, tuple[str, str]] = {
    name: (module, name) for module, names in _EXPORTS.items() for name in names
} | _ALIASES


# Pre-#384 names for the workspace snapshot helpers. They still resolve, with a
# DeprecationWarning, and are no longer in ``__all__``.
_DEPRECATED_ALIASES = {
    "snapshot": "workspace_snapshot",
    "restore": "workspace_restore",
    "list_snapshots": "list_workspace_snapshots",
}


class _Package(ModuleType):
    """This package's module type: a public name wins over a same-named submodule.

    ``benchflow.branch`` is both the public ``branch()`` function and the
    submodule ``benchflow/branch.py``. When the import system first loads that
    submodule it binds it on the package, which would shadow the function for
    ``from benchflow import branch``. The eager imports used to rebind the
    function afterwards; with lazy names, the binding is skipped instead.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if (
            name in _LAZY
            and isinstance(value, ModuleType)
            and value.__name__ == f"{self.__name__}.{name}"
        ):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Package


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("benchflow")
    except PackageNotFoundError:
        return "0+unknown"


def __getattr__(name: str):
    """Public names on first use, then deprecated aliases, then submodules."""
    import importlib

    target = _LAZY.get(name)
    if target is not None:
        value = getattr(importlib.import_module(target[0]), target[1])
        globals()[name] = value
        return value

    if name == "__version__":
        value = globals()["__version__"] = _version()
        return value

    if name == "RuntimeResult":
        # Not cached: every lookup warns, so an isinstance check that now
        # reads False is named wherever it is (DF-p4-upgrader-6).
        import warnings

        runtime = importlib.import_module("benchflow.runtime")
        warnings.warn(
            runtime.RUNTIME_RESULT_DEPRECATION, DeprecationWarning, stacklevel=2
        )
        return runtime._RuntimeResult

    if name in _DEPRECATED_ALIASES:
        import warnings

        target_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"benchflow.{name} is deprecated; use benchflow.{target_name} "
            "(it snapshots a workspace directory, not a sandbox).",
            DeprecationWarning,
            stacklevel=2,
        )
        return __getattr__(target_name)

    try:
        return importlib.import_module(f"benchflow.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"benchflow.{name}":
            raise
    raise AttributeError(f"module 'benchflow' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
