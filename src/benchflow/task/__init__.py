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
    normalize_task_document_frontmatter,
    render_normalized_task_md,
    render_task_md,
    render_task_md_from_legacy,
)
from benchflow.task.env import resolve_env_vars
from benchflow.task.export import (
    CompatibilityExportLoss,
    CompatibilityExportReport,
    HarborRoundTripConformanceReport,
    HarborRoundTripMismatch,
    build_compatibility_export_report,
    build_harbor_roundtrip_conformance_report,
    export_task_to_split_layout,
)
from benchflow.task.imports import (
    ImportedTaskConfig,
    TaskConfigImportReport,
    import_task_config_toml,
    merge_compat_extra,
)
from benchflow.task.output_format import (
    TASK_OUTPUT_FORMATS,
    TaskOutputFormat,
    ensure_existing_task_output_format,
    oracle_dir_name,
    task_entrypoint_name,
    validate_task_output_format,
    verifier_dir_name,
)
from benchflow.task.package import TaskPackage
from benchflow.task.paths import (
    RolloutPaths,
    SandboxPaths,
    TaskPaths,
)
from benchflow.task.prompts import (
    CompiledPromptTurn,
    CompiledUserRuntime,
    PromptPart,
    TaskPromptPlan,
    UserRuntimeContract,
    compile_document_user_runtime,
    compile_task_prompt_plan,
    materialize_prompt_plan_scenes,
)
from benchflow.task.runtime_capabilities import (
    UnsupportedTaskFeature,
    UnsupportedTaskFeatureError,
    raise_for_task_runtime_support,
    validate_task_runtime_support,
)
from benchflow.task.runtime_view import (
    TaskEntrypoint,
    TaskRuntimeCompatibility,
    TaskRuntimeView,
)
from benchflow.task.task import Task
from benchflow.task.verifier import (
    AddTestsDirError,
    DownloadVerifierDirError,
    ORSEpisodeInputError,
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
    VerifierOutputContract,
    VerifierStrategy,
    load_verifier_document,
)

__all__ = [
    "Task",
    "TaskPaths",
    "TaskConfig",
    "TASK_DOCUMENT_FILENAME",
    "TaskDocument",
    "TaskDocumentParseError",
    "normalize_task_document_frontmatter",
    "render_normalized_task_md",
    "render_task_md",
    "CompatibilityExportLoss",
    "CompatibilityExportReport",
    "HarborRoundTripConformanceReport",
    "HarborRoundTripMismatch",
    "build_compatibility_export_report",
    "build_harbor_roundtrip_conformance_report",
    "export_task_to_split_layout",
    "ImportedTaskConfig",
    "TaskConfigImportReport",
    "import_task_config_toml",
    "merge_compat_extra",
    "TASK_OUTPUT_FORMATS",
    "TaskOutputFormat",
    "ensure_existing_task_output_format",
    "oracle_dir_name",
    "task_entrypoint_name",
    "validate_task_output_format",
    "verifier_dir_name",
    "TaskPackage",
    "TaskPromptPlan",
    "CompiledPromptTurn",
    "CompiledUserRuntime",
    "PromptPart",
    "UserRuntimeContract",
    "compile_document_user_runtime",
    "compile_task_prompt_plan",
    "materialize_prompt_plan_scenes",
    "UnsupportedTaskFeature",
    "UnsupportedTaskFeatureError",
    "raise_for_task_runtime_support",
    "validate_task_runtime_support",
    "TaskEntrypoint",
    "TaskRuntimeCompatibility",
    "TaskRuntimeView",
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
    "RolloutPaths",
    "SandboxPaths",
    "Verifier",
    "VerifierResult",
    "VERIFIER_DOCUMENT_FILENAME",
    "VerifierDocument",
    "VerifierDocumentParseError",
    "VerifierOutputContract",
    "VerifierStrategy",
    "load_verifier_document",
    "RewardFileEmptyError",
    "RewardFileNotFoundError",
    "RubricNotFoundError",
    "UnsupportedVerifierStrategyError",
    "VerifierOutputParseError",
    "AddTestsDirError",
    "DownloadVerifierDirError",
    "ORSEpisodeInputError",
    "resolve_env_vars",
    "render_task_md_from_legacy",
]
