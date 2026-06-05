"""Task package — native BenchFlow types with RL-first terminology.

- Task, TaskPaths: problem specification ($T$)
- TaskConfig, SandboxConfig, VerifierConfig, AgentConfig: configuration ($C$)
- RolloutPaths: rollout output paths
- SandboxPaths: container mount points
- Verifier, VerifierResult: reward function ($V$)
- resolve_env_vars: env template resolution
"""

from benchflow.task.config import (
    ORG_NAME_PATTERN,
    ArtifactConfig,
    Author,
    HealthcheckConfig,
    JudgeVerifierConfig,
    MCPServerConfig,
    MultiStepRewardStrategy,
    NetworkMode,
    PackageInfo,
    SandboxConfig,
    SolutionConfig,
    StepConfig,
    TaskConfig,
    TaskOS,
    TpuSpec,
    VerifierConfig,
    VerifierEnvironmentMode,
    VerifierHardeningConfig,
)
from benchflow.task.config import (
    AgentConfig as TaskAgentConfig,
)
from benchflow.task.document import (
    TASK_DOCUMENT_FILENAME,
    TaskDocument,
    TaskDocumentParseError,
    render_task_md_from_legacy,
)
from benchflow.task.env import resolve_env_vars
from benchflow.task.export import (
    ExportLoss,
    ExportMode,
    ExportResult,
    ExportTarget,
    ImportResult,
    NativeComparison,
    export_task_package,
    import_split_task_package,
    materialize_export_result,
    validate_export_round_trip,
)
from benchflow.task.package import (
    AliasCollisionStatus,
    TaskPackage,
    TaskRuntimeView,
)
from benchflow.task.paths import (
    RolloutPaths,
    SandboxPaths,
    TaskPaths,
)
from benchflow.task.runtime_capabilities import (
    UnsupportedTaskFeature,
    UnsupportedTaskRuntimeError,
    ensure_task_runtime_support,
    validate_task_runtime_support,
)
from benchflow.task.task import Task
from benchflow.task.verifier import (
    AddTestsDirError,
    DownloadVerifierDirError,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    RubricNotFoundError,
    UnsupportedVerifierStrategyError,
    Verifier,
    VerifierOutputParseError,
    VerifierResult,
)
from benchflow.task.verifier_document import (
    VERIFIER_DOCUMENT_FILENAME,
    VerifierDocument,
    VerifierDocumentParseError,
    VerifierOutputs,
    VerifierRubricFiles,
    is_executable_script_strategy,
    resolve_default_strategy,
    resolve_verifier_spec_path,
    verifier_document_issues,
    verifier_strategy_type,
)

__all__ = [
    # Task ($T$)
    "Task",
    "TaskPaths",
    "TaskConfig",
    "TASK_DOCUMENT_FILENAME",
    "TaskDocument",
    "TaskDocumentParseError",
    "TaskPackage",
    "TaskRuntimeView",
    "AliasCollisionStatus",
    "UnsupportedTaskFeature",
    "UnsupportedTaskRuntimeError",
    "ensure_task_runtime_support",
    "validate_task_runtime_support",
    "VERIFIER_DOCUMENT_FILENAME",
    "VerifierDocument",
    "VerifierDocumentParseError",
    "VerifierOutputs",
    "VerifierRubricFiles",
    "resolve_verifier_spec_path",
    "resolve_default_strategy",
    "verifier_strategy_type",
    "is_executable_script_strategy",
    "verifier_document_issues",
    # Configuration ($C$)
    "SandboxConfig",
    "VerifierConfig",
    "JudgeVerifierConfig",
    "VerifierHardeningConfig",
    "HealthcheckConfig",
    "TaskAgentConfig",
    "SolutionConfig",
    "MCPServerConfig",
    "ArtifactConfig",
    "StepConfig",
    "TpuSpec",
    "NetworkMode",
    "TaskOS",
    "VerifierEnvironmentMode",
    "MultiStepRewardStrategy",
    "PackageInfo",
    "Author",
    "ORG_NAME_PATTERN",
    # Rollout paths
    "RolloutPaths",
    # Sandbox paths
    "SandboxPaths",
    # Verifier ($V$)
    "Verifier",
    "VerifierResult",
    "RewardFileEmptyError",
    "RewardFileNotFoundError",
    "RubricNotFoundError",
    "UnsupportedVerifierStrategyError",
    "VerifierOutputParseError",
    "AddTestsDirError",
    "DownloadVerifierDirError",
    # Utilities
    "resolve_env_vars",
    "render_task_md_from_legacy",
    # Export / import
    "ExportLoss",
    "ExportMode",
    "ExportResult",
    "ExportTarget",
    "ImportResult",
    "NativeComparison",
    "export_task_package",
    "import_split_task_package",
    "materialize_export_result",
    "validate_export_round_trip",
]
