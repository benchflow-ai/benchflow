"""Task package — native BenchFlow types with RL-first terminology.

- Task, TaskPaths: problem specification ($T$)
- TaskConfig, SandboxConfig, VerifierConfig, AgentConfig: configuration ($C$)
- RolloutPaths: rollout output paths
- SandboxPaths: container mount points
- Verifier, VerifierResult: reward function ($V$)
- resolve_env_vars: env template resolution
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
        SetupCommandConfig,
        SolutionConfig,
        StepConfig,
        TaskConfig,
        TaskOS,
        TpuSpec,
        VerifierConfig,
        VerifierHardeningConfig,
        VerifierSandboxMode,
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
    from benchflow.task.equivalence import (
        EquivalenceReport,
        check_equivalence,
        check_equivalence_async,
        run_equivalence_battery,
    )
    from benchflow.task.export import (
        CompatibilityExportLoss,
        CompatibilityExportReport,
        TaskTomlRoundTripConformanceReport,
        TaskTomlRoundTripMismatch,
        build_compatibility_export_report,
        build_task_toml_roundtrip_conformance_report,
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

# Each export's module and name there. They are imported on first use (PEP
# 562), so importing one submodule (``benchflow.task.config`` for a task file's
# settings) does not import every other one.
_EXPORTS: dict[str, tuple[str, str]] = {
    "ORG_NAME_PATTERN": ("benchflow.task.config", "ORG_NAME_PATTERN"),
    "ArtifactConfig": ("benchflow.task.config", "ArtifactConfig"),
    "Author": ("benchflow.task.config", "Author"),
    "HealthcheckConfig": ("benchflow.task.config", "HealthcheckConfig"),
    "JudgeVerifierConfig": ("benchflow.task.config", "JudgeVerifierConfig"),
    "MCPServerConfig": ("benchflow.task.config", "MCPServerConfig"),
    "MultiStepRewardStrategy": ("benchflow.task.config", "MultiStepRewardStrategy"),
    "NetworkMode": ("benchflow.task.config", "NetworkMode"),
    "PackageInfo": ("benchflow.task.config", "PackageInfo"),
    "SandboxConfig": ("benchflow.task.config", "SandboxConfig"),
    "SetupCommandConfig": ("benchflow.task.config", "SetupCommandConfig"),
    "SolutionConfig": ("benchflow.task.config", "SolutionConfig"),
    "StepConfig": ("benchflow.task.config", "StepConfig"),
    "TaskConfig": ("benchflow.task.config", "TaskConfig"),
    "TaskOS": ("benchflow.task.config", "TaskOS"),
    "TpuSpec": ("benchflow.task.config", "TpuSpec"),
    "VerifierConfig": ("benchflow.task.config", "VerifierConfig"),
    "VerifierHardeningConfig": ("benchflow.task.config", "VerifierHardeningConfig"),
    "VerifierSandboxMode": ("benchflow.task.config", "VerifierSandboxMode"),
    "TaskAgentConfig": ("benchflow.task.config", "AgentConfig"),
    "TASK_DOCUMENT_FILENAME": ("benchflow.task.document", "TASK_DOCUMENT_FILENAME"),
    "TaskDocument": ("benchflow.task.document", "TaskDocument"),
    "TaskDocumentParseError": ("benchflow.task.document", "TaskDocumentParseError"),
    "normalize_task_document_frontmatter": (
        "benchflow.task.document",
        "normalize_task_document_frontmatter",
    ),
    "render_normalized_task_md": (
        "benchflow.task.document",
        "render_normalized_task_md",
    ),
    "render_task_md": ("benchflow.task.document", "render_task_md"),
    "render_task_md_from_legacy": (
        "benchflow.task.document",
        "render_task_md_from_legacy",
    ),
    "resolve_env_vars": ("benchflow.task.env", "resolve_env_vars"),
    "EquivalenceReport": ("benchflow.task.equivalence", "EquivalenceReport"),
    "check_equivalence": ("benchflow.task.equivalence", "check_equivalence"),
    "check_equivalence_async": (
        "benchflow.task.equivalence",
        "check_equivalence_async",
    ),
    "run_equivalence_battery": (
        "benchflow.task.equivalence",
        "run_equivalence_battery",
    ),
    "CompatibilityExportLoss": ("benchflow.task.export", "CompatibilityExportLoss"),
    "CompatibilityExportReport": ("benchflow.task.export", "CompatibilityExportReport"),
    "TaskTomlRoundTripConformanceReport": (
        "benchflow.task.export",
        "TaskTomlRoundTripConformanceReport",
    ),
    "TaskTomlRoundTripMismatch": ("benchflow.task.export", "TaskTomlRoundTripMismatch"),
    "build_compatibility_export_report": (
        "benchflow.task.export",
        "build_compatibility_export_report",
    ),
    "build_task_toml_roundtrip_conformance_report": (
        "benchflow.task.export",
        "build_task_toml_roundtrip_conformance_report",
    ),
    "export_task_to_split_layout": (
        "benchflow.task.export",
        "export_task_to_split_layout",
    ),
    "ImportedTaskConfig": ("benchflow.task.imports", "ImportedTaskConfig"),
    "TaskConfigImportReport": ("benchflow.task.imports", "TaskConfigImportReport"),
    "import_task_config_toml": ("benchflow.task.imports", "import_task_config_toml"),
    "merge_compat_extra": ("benchflow.task.imports", "merge_compat_extra"),
    "TASK_OUTPUT_FORMATS": ("benchflow.task.output_format", "TASK_OUTPUT_FORMATS"),
    "TaskOutputFormat": ("benchflow.task.output_format", "TaskOutputFormat"),
    "ensure_existing_task_output_format": (
        "benchflow.task.output_format",
        "ensure_existing_task_output_format",
    ),
    "oracle_dir_name": ("benchflow.task.output_format", "oracle_dir_name"),
    "task_entrypoint_name": ("benchflow.task.output_format", "task_entrypoint_name"),
    "validate_task_output_format": (
        "benchflow.task.output_format",
        "validate_task_output_format",
    ),
    "verifier_dir_name": ("benchflow.task.output_format", "verifier_dir_name"),
    "TaskPackage": ("benchflow.task.package", "TaskPackage"),
    "RolloutPaths": ("benchflow.task.paths", "RolloutPaths"),
    "SandboxPaths": ("benchflow.task.paths", "SandboxPaths"),
    "TaskPaths": ("benchflow.task.paths", "TaskPaths"),
    "CompiledPromptTurn": ("benchflow.task.prompts", "CompiledPromptTurn"),
    "CompiledUserRuntime": ("benchflow.task.prompts", "CompiledUserRuntime"),
    "PromptPart": ("benchflow.task.prompts", "PromptPart"),
    "TaskPromptPlan": ("benchflow.task.prompts", "TaskPromptPlan"),
    "UserRuntimeContract": ("benchflow.task.prompts", "UserRuntimeContract"),
    "compile_document_user_runtime": (
        "benchflow.task.prompts",
        "compile_document_user_runtime",
    ),
    "compile_task_prompt_plan": ("benchflow.task.prompts", "compile_task_prompt_plan"),
    "materialize_prompt_plan_scenes": (
        "benchflow.task.prompts",
        "materialize_prompt_plan_scenes",
    ),
    "UnsupportedTaskFeature": (
        "benchflow.task.runtime_capabilities",
        "UnsupportedTaskFeature",
    ),
    "UnsupportedTaskFeatureError": (
        "benchflow.task.runtime_capabilities",
        "UnsupportedTaskFeatureError",
    ),
    "raise_for_task_runtime_support": (
        "benchflow.task.runtime_capabilities",
        "raise_for_task_runtime_support",
    ),
    "validate_task_runtime_support": (
        "benchflow.task.runtime_capabilities",
        "validate_task_runtime_support",
    ),
    "TaskEntrypoint": ("benchflow.task.runtime_view", "TaskEntrypoint"),
    "TaskRuntimeCompatibility": (
        "benchflow.task.runtime_view",
        "TaskRuntimeCompatibility",
    ),
    "TaskRuntimeView": ("benchflow.task.runtime_view", "TaskRuntimeView"),
    "Task": ("benchflow.task.task", "Task"),
    "AddTestsDirError": ("benchflow.task.verifier", "AddTestsDirError"),
    "DownloadVerifierDirError": ("benchflow.task.verifier", "DownloadVerifierDirError"),
    "ORSEpisodeInputError": ("benchflow.task.verifier", "ORSEpisodeInputError"),
    "RewardFileEmptyError": ("benchflow.task.verifier", "RewardFileEmptyError"),
    "RewardFileNotFoundError": ("benchflow.task.verifier", "RewardFileNotFoundError"),
    "RubricNotFoundError": ("benchflow.task.verifier", "RubricNotFoundError"),
    "UnsupportedVerifierStrategyError": (
        "benchflow.task.verifier",
        "UnsupportedVerifierStrategyError",
    ),
    "Verifier": ("benchflow.task.verifier", "Verifier"),
    "VerifierOutputParseError": ("benchflow.task.verifier", "VerifierOutputParseError"),
    "VerifierResult": ("benchflow.task.verifier", "VerifierResult"),
    "VERIFIER_DOCUMENT_FILENAME": (
        "benchflow.task.verifier_document",
        "VERIFIER_DOCUMENT_FILENAME",
    ),
    "VerifierDocument": ("benchflow.task.verifier_document", "VerifierDocument"),
    "VerifierDocumentParseError": (
        "benchflow.task.verifier_document",
        "VerifierDocumentParseError",
    ),
    "VerifierOutputContract": (
        "benchflow.task.verifier_document",
        "VerifierOutputContract",
    ),
    "VerifierStrategy": ("benchflow.task.verifier_document", "VerifierStrategy"),
    "load_verifier_document": (
        "benchflow.task.verifier_document",
        "load_verifier_document",
    ),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = [
    "Task",
    "EquivalenceReport",
    "check_equivalence",
    "check_equivalence_async",
    "run_equivalence_battery",
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
    "TaskTomlRoundTripConformanceReport",
    "TaskTomlRoundTripMismatch",
    "build_compatibility_export_report",
    "build_task_toml_roundtrip_conformance_report",
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
    "SetupCommandConfig",
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
    "VerifierSandboxMode",
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
