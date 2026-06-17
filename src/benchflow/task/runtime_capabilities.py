"""Runtime capability checks for parsed task packages.

The authoring parser intentionally accepts the Harbor-compatible task surface,
plus BenchFlow-native document fields. This module is the first runtime-facing
gate: it reports parsed semantics that the selected backend cannot currently
honor, so callers can fail closed before sandbox launch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from benchflow.rewards.rubric_config import criteria_aggregate_policy_from_rubric
from benchflow.sandbox.providers import SANDBOX_PROVIDER_SET, providers_phrase
from benchflow.task.config import (
    NetworkMode,
    TaskConfig,
    TaskOS,
    VerifierEnvironmentMode,
)
from benchflow.task.document import TaskDocument
from benchflow.task.paths import TaskPaths, local_script_strategy_files
from benchflow.task.prompts import (
    CompiledUserRuntime,
    compile_document_user_runtime,
    compile_task_prompt_plan,
)
from benchflow.task.verifier_document import load_verifier_document

# Back-compat name kept for this module's existing membership check; the
# canonical set lives in benchflow.sandbox.providers.
SUPPORTED_SANDBOX_BACKENDS = SANDBOX_PROVIDER_SET


@dataclass(frozen=True)
class UnsupportedTaskFeature:
    """A parsed task feature the selected runtime cannot execute."""

    path: str
    reason: str
    sandbox: str

    def format(self) -> str:
        return f"{self.path}: {self.reason} (sandbox={self.sandbox})"


class UnsupportedTaskFeatureError(RuntimeError):
    """Raised when a task cannot be launched on the selected sandbox."""

    def __init__(self, features: list[UnsupportedTaskFeature]) -> None:
        self.features = tuple(features)
        lines = "\n".join(f"- {feature.format()}" for feature in self.features)
        super().__init__(
            "Task uses parsed runtime features that BenchFlow cannot execute "
            f"on the selected sandbox:\n{lines}"
        )


def validate_task_runtime_support(
    task: TaskDocument | TaskConfig,
    *,
    sandbox: str,
    task_dir: str | Path | None = None,
) -> list[UnsupportedTaskFeature]:
    """Return unsupported parsed features for ``sandbox``.

    ``task`` may be a full :class:`TaskDocument` when document-only fields need
    checking, or a plain :class:`TaskConfig` for legacy split packages.
    """

    config = task.config if isinstance(task, TaskDocument) else task
    document = task if isinstance(task, TaskDocument) else None
    unsupported: list[UnsupportedTaskFeature] = []
    if sandbox not in SUPPORTED_SANDBOX_BACKENDS:
        _issue(
            unsupported,
            path="sandbox",
            reason=f"unknown sandbox backend; use {providers_phrase()}",
            sandbox=sandbox,
        )
        return unsupported

    _append_config_issues(unsupported, config=config, sandbox=sandbox)
    if document is not None:
        _append_document_issues(unsupported, document=document, sandbox=sandbox)
    if task_dir is not None:
        _append_layout_issues(unsupported, task_dir=Path(task_dir), sandbox=sandbox)
    return unsupported


def raise_for_task_runtime_support(
    task: TaskDocument | TaskConfig,
    *,
    sandbox: str,
    task_dir: str | Path | None = None,
) -> None:
    """Raise if ``task`` contains parsed semantics unsupported by ``sandbox``."""

    unsupported = validate_task_runtime_support(
        task,
        sandbox=sandbox,
        task_dir=task_dir,
    )
    if unsupported:
        raise UnsupportedTaskFeatureError(unsupported)


def _issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    path: str,
    reason: str,
    sandbox: str,
) -> None:
    unsupported.append(
        UnsupportedTaskFeature(path=path, reason=reason, sandbox=sandbox)
    )


def _append_config_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    config: TaskConfig,
    sandbox: str,
) -> None:
    if config.steps:
        _issue(
            unsupported,
            path="steps",
            reason="Harbor multi-step execution is parsed but not runtime-gated",
            sandbox=sandbox,
        )
        for i, step in enumerate(config.steps):
            if step.artifacts:
                _issue(
                    unsupported,
                    path=f"steps[{i}].artifacts",
                    reason="step artifact transfer is parsed but not implemented",
                    sandbox=sandbox,
                )
            if step.healthcheck is not None:
                _issue(
                    unsupported,
                    path=f"steps[{i}].healthcheck",
                    reason="step healthchecks are parsed but not implemented",
                    sandbox=sandbox,
                )

    if config.artifacts:
        _issue(
            unsupported,
            path="artifacts",
            reason="root artifact collection is parsed but not runtime-gated",
            sandbox=sandbox,
        )

    _append_network_issue(
        unsupported,
        path="agent.network_mode",
        mode=config.agent.network_mode,
        allowed_hosts=tuple(getattr(config.agent, "allowed_hosts", None) or ()),
        sandbox=sandbox,
    )
    _append_network_issue(
        unsupported,
        path="environment.network_mode",
        mode=config.environment.network_mode,
        allowed_hosts=tuple(getattr(config.environment, "allowed_hosts", None) or ()),
        sandbox=sandbox,
    )
    _append_network_issue(
        unsupported,
        path="verifier.network_mode",
        mode=config.verifier.network_mode,
        allowed_hosts=tuple(getattr(config.verifier, "allowed_hosts", None) or ()),
        sandbox=sandbox,
    )

    if config.verifier.environment_mode == VerifierEnvironmentMode.SEPARATE:
        _issue(
            unsupported,
            path="verifier.environment_mode",
            reason="separate verifier environments are parsed but not executed",
            sandbox=sandbox,
        )
    if config.verifier.environment is not None:
        _issue(
            unsupported,
            path="verifier.environment",
            reason="verifier-specific environment materialization is not implemented",
            sandbox=sandbox,
        )
    if config.verifier.service != "main" and sandbox != "docker":
        _issue(
            unsupported,
            path="verifier.service",
            reason="non-main verifier services require Docker compose support",
            sandbox=sandbox,
        )

    env = config.environment
    if env.os == TaskOS.WINDOWS:
        _issue(
            unsupported,
            path="environment.os",
            reason="Windows task environments are parsed but not executable",
            sandbox=sandbox,
        )
    if env.tpu is not None:
        _issue(
            unsupported,
            path="environment.tpu",
            reason="TPU scheduling is parsed but not implemented",
            sandbox=sandbox,
        )
    if env.gpus or env.gpu_types:
        _issue(
            unsupported,
            path="environment.gpus",
            reason="GPU scheduling is parsed but not capability-gated",
            sandbox=sandbox,
        )
    if env.healthcheck is not None:
        _issue(
            unsupported,
            path="environment.healthcheck",
            reason="environment healthchecks are parsed but not runtime-gated",
            sandbox=sandbox,
        )
    _append_workdir_issue(
        unsupported,
        workdir=env.workdir,
        sandbox=sandbox,
    )


def _append_workdir_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    workdir: str | None,
    sandbox: str,
) -> None:
    if workdir is None:
        return
    if not isinstance(workdir, str) or not workdir.strip():
        _issue(
            unsupported,
            path="environment.workdir",
            reason="configured workdir must be a non-empty absolute path",
            sandbox=sandbox,
        )
        return
    path = PurePosixPath(workdir)
    if not path.is_absolute() or path == PurePosixPath("/"):
        _issue(
            unsupported,
            path="environment.workdir",
            reason="configured workdir must be an absolute non-root path",
            sandbox=sandbox,
        )


def _append_network_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    path: str,
    mode: NetworkMode | None,
    sandbox: str,
    allowed_hosts: tuple[str, ...] = (),
) -> None:
    if mode == NetworkMode.ALLOWLIST:
        from benchflow.sandbox.network_policy import (
            allowlist_is_ip_based,
            sandbox_supports_allowlist,
        )

        if not sandbox_supports_allowlist(sandbox):
            _issue(
                unsupported,
                path=path,
                reason=(
                    "network_mode='allowlist' is enforced only on the 'docker' "
                    "sandbox (egress proxy); not available on this sandbox — use "
                    "'docker', 'no-network', or 'public' (ENG-219)"
                ),
                sandbox=sandbox,
            )
            return
        # IP-CIDR allowlist sandboxes (daytona) can't express wildcards. This
        # is the only model-independent reject we can make at preflight; the
        # >10-IP / unresolvable checks need DNS + the model host and fail
        # closed at lockdown (relock_network).
        if allowlist_is_ip_based(sandbox):
            wild = sorted(h for h in allowed_hosts if h.startswith("*."))
            if wild:
                _issue(
                    unsupported,
                    path=path,
                    reason=(
                        f"network_mode='allowlist' on the '{sandbox}' sandbox uses "
                        f"an IPv4 allow list and cannot express wildcard host(s) "
                        f"{wild}; use the 'docker' sandbox for wildcard allowlists "
                        "or list exact hostnames"
                    ),
                    sandbox=sandbox,
                )


def _append_document_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    document: TaskDocument,
    sandbox: str,
) -> None:
    user_runtime = compile_document_user_runtime(document)
    _append_document_user_issues(
        unsupported,
        document=document,
        sandbox=sandbox,
        user_runtime=user_runtime,
    )
    _append_prompt_policy_issues(unsupported, document=document, sandbox=sandbox)
    _append_benchflow_verifier_issues(
        unsupported,
        verifier=document.benchflow.get("verifier"),
        sandbox=sandbox,
    )
    _append_benchflow_namespace_issues(
        unsupported,
        benchflow=document.benchflow,
        sandbox=sandbox,
        user_runtime=user_runtime,
    )


def _append_benchflow_namespace_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    benchflow: Mapping[str, object],
    sandbox: str,
    user_runtime: CompiledUserRuntime | None = None,
) -> None:
    unsupported_keys = {
        "agent_policy": "agent policy is parsed as metadata but not enforced",
        "runtime_policy": "runtime policy is parsed as metadata but not enforced",
        "teams": "agent-team handoff policy is parsed but not executable",
    }
    for key, reason in unsupported_keys.items():
        if key in benchflow:
            if key == "teams" and user_runtime is not None:
                contract = getattr(user_runtime, "contract", None)
                if (
                    getattr(contract, "status", None) == "supported"
                    and getattr(contract, "handoff_kind", None) == "sequential-shared"
                ):
                    continue
                reason = getattr(contract, "reason", None) or reason
            _issue(
                unsupported,
                path=f"benchflow.{key}",
                reason=reason,
                sandbox=sandbox,
            )


def _append_benchflow_verifier_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    verifier: object,
    sandbox: str,
) -> None:
    if verifier is None:
        return
    if not isinstance(verifier, Mapping):
        _issue(
            unsupported,
            path="benchflow.verifier",
            reason="benchflow.verifier must be a mapping",
            sandbox=sandbox,
        )
        return

    verifier_mapping = cast(Mapping[str, object], verifier)
    implementation = verifier_mapping.get("implementation")
    if implementation is None:
        return
    if not isinstance(implementation, Mapping):
        _issue(
            unsupported,
            path="benchflow.verifier.implementation",
            reason="verifier implementation metadata must be a mapping",
            sandbox=sandbox,
        )
        return

    implementation_mapping = cast(Mapping[str, object], implementation)
    implementation_type = implementation_mapping.get("type")
    if implementation_type not in {None, "test-script", "script", "deterministic"}:
        if implementation_type == "hybrid":
            _append_benchflow_verifier_strategy_issues(
                unsupported,
                strategies=implementation_mapping.get("strategies"),
                sandbox=sandbox,
            )
            return
        _issue(
            unsupported,
            path="benchflow.verifier.implementation.type",
            reason=(
                "verifier implementation type is parsed but not executable; "
                "use test-script/script or declare concrete verifier strategies"
            ),
            sandbox=sandbox,
        )


def _append_benchflow_verifier_strategy_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    strategies: object,
    sandbox: str,
) -> None:
    if strategies is None:
        return
    if not isinstance(strategies, Mapping):
        _issue(
            unsupported,
            path="benchflow.verifier.implementation.strategies",
            reason="verifier implementation strategies must be a mapping",
            sandbox=sandbox,
        )
        return
    strategies_mapping = cast(Mapping[str, object], strategies)
    for name in strategies_mapping:
        strategy_name = str(name)
        normalized = strategy_name.replace("-", "_")
        if normalized in {
            "deterministic",
            "script",
            "test_script",
            "llm_judge",
            "agent_judge",
            "ors_episode",
            "rewardkit",
            "reward_kit",
        }:
            continue
        _issue(
            unsupported,
            path=f"benchflow.verifier.implementation.strategies.{strategy_name}",
            reason=(
                "verifier implementation strategy is parsed but not executable; "
                "select script, llm-judge, reward-kit, agent-judge, or "
                "ors-episode until the engine lands"
            ),
            sandbox=sandbox,
        )


def _append_document_user_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    document: TaskDocument,
    sandbox: str,
    user_runtime: CompiledUserRuntime | None = None,
) -> None:
    runtime = user_runtime or compile_document_user_runtime(document)
    if runtime.contract.status != "unsupported":
        return
    reason = runtime.contract.reason or "document user runtime is unsupported"
    if document.user:
        _issue(unsupported, path="user", reason=reason, sandbox=sandbox)
    if document.user_persona:
        _issue(unsupported, path="## user-persona", reason=reason, sandbox=sandbox)
    if "nudges" in document.benchflow:
        _issue(unsupported, path="benchflow.nudges", reason=reason, sandbox=sandbox)


def _append_prompt_policy_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    document: TaskDocument,
    sandbox: str,
) -> None:
    if "prompt" not in document.benchflow:
        return
    try:
        compile_task_prompt_plan(
            document,
            fallback_prompt=document.instruction,
            scenes=document.scenes,
        )
    except ValueError as e:
        _issue(
            unsupported,
            path="benchflow.prompt",
            reason=str(e),
            sandbox=sandbox,
        )


def _append_layout_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    task_dir: Path,
    sandbox: str,
) -> None:
    paths = TaskPaths(task_dir)

    task_md = paths.task_document_path
    has_legacy_definition = (
        paths.config_path.exists() or paths.instruction_path.exists()
    )
    if task_md.exists() and has_legacy_definition:
        if not (paths.config_path.exists() and paths.instruction_path.exists()):
            _issue(
                unsupported,
                path="task.md",
                reason="native task.md coexists with an incomplete legacy split definition",
                sandbox=sandbox,
            )
        else:
            _append_definition_drift_issue(unsupported, paths=paths, sandbox=sandbox)

    _append_alias_drift_issue(
        unsupported,
        native_dir=paths.oracle_dir,
        legacy_dir=paths.legacy_solution_dir,
        path="oracle|solution",
        reason="oracle/ and solution/ both exist but are not byte-equivalent",
        sandbox=sandbox,
    )
    _append_alias_drift_issue(
        unsupported,
        native_dir=paths.verifier_source_dir,
        legacy_dir=paths.legacy_tests_dir,
        path="verifier|tests",
        reason="verifier/ and tests/ both exist but are not byte-equivalent",
        sandbox=sandbox,
    )
    _append_verifier_strategy_issue(unsupported, paths=paths, sandbox=sandbox)


def _append_verifier_strategy_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    paths: TaskPaths,
    sandbox: str,
) -> None:
    try:
        document = load_verifier_document(paths.tests_dir)
    except Exception as e:
        _issue(
            unsupported,
            path="verifier/verifier.md",
            reason=f"cannot parse verifier package document: {e}",
            sandbox=sandbox,
        )
        return
    if document is None:
        return

    strategy = document.selected_strategy
    if strategy.type == "reward-kit":
        _append_reward_kit_strategy_issue(
            unsupported,
            paths=paths,
            strategy_name=strategy.name,
            root=strategy.root_path,
            criteria=strategy.criteria_path,
            entrypoint=strategy.entrypoint,
            sandbox=sandbox,
        )
        return
    if strategy.type == "script":
        _append_script_strategy_issue(
            unsupported,
            paths=paths,
            strategy_name=strategy.name,
            command=strategy.command,
            sandbox=sandbox,
        )
        return
    if strategy.type == "llm-judge":
        _append_llm_judge_strategy_issue(
            unsupported,
            paths=paths,
            strategy_name=strategy.name,
            rubric=strategy.rubric_path,
            context_file=strategy.context_file,
            sandbox=sandbox,
        )
        return
    if strategy.type not in {"script", "llm-judge", "agent-judge", "ors-episode"}:
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy.name}",
            reason=(
                f"{strategy.type} verifier strategies are parsed but not "
                "executable; select script, llm-judge, reward-kit, agent-judge, "
                "or ors-episode"
            ),
            sandbox=sandbox,
        )


def _append_script_strategy_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    paths: TaskPaths,
    strategy_name: str,
    command: str | None,
    sandbox: str,
) -> None:
    local_files = local_script_strategy_files(command, verifier_dir=paths.tests_dir)
    if not local_files:
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason="script command does not reference a packaged verifier artifact",
            sandbox=sandbox,
        )
        return
    for local_file in local_files:
        if local_file.is_file():
            continue
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=f"script artifact not found at {local_file}",
            sandbox=sandbox,
        )


def _append_llm_judge_strategy_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    paths: TaskPaths,
    strategy_name: str,
    rubric: str | None,
    context_file: str | None,
    sandbox: str,
) -> None:
    if not _strategy_file_exists(rubric, verifier_dir=paths.tests_dir):
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=f"llm-judge rubric not found at {rubric}",
            sandbox=sandbox,
        )
    if context_file is not None and not _strategy_file_exists(
        context_file,
        verifier_dir=paths.tests_dir,
    ):
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=f"llm-judge context file not found at {context_file}",
            sandbox=sandbox,
        )


def _append_reward_kit_strategy_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    paths: TaskPaths,
    strategy_name: str,
    root: str | None,
    criteria: str | None,
    entrypoint: str | None,
    sandbox: str,
) -> None:
    if root is None:
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason="reward-kit strategy must declare root",
            sandbox=sandbox,
        )
        return
    try:
        root_path = _safe_relative_runtime_path(root)
        entrypoint_path = _safe_relative_runtime_path(entrypoint or "reward.py")
    except ValueError as e:
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=str(e),
            sandbox=sandbox,
        )
        return
    runner = paths.tests_dir / Path(*root_path.parts) / Path(*entrypoint_path.parts)
    if not runner.is_file():
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=f"reward-kit runner not found at {runner}",
            sandbox=sandbox,
        )
    if criteria is None:
        return
    try:
        criteria_path = _safe_relative_runtime_path(criteria)
    except ValueError as e:
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=str(e),
            sandbox=sandbox,
        )
        return
    criteria_file = paths.tests_dir / Path(*criteria_path.parts)
    if not criteria_file.is_file():
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=f"reward-kit criteria not found at {criteria_file}",
            sandbox=sandbox,
        )
        return
    try:
        criteria_aggregate_policy_from_rubric(criteria_file)
    except ValueError as e:
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=f"reward-kit criteria are invalid: {e}",
            sandbox=sandbox,
        )


def _strategy_file_exists(value: str | None, *, verifier_dir: Path) -> bool:
    if value is None:
        return False
    try:
        relative = _safe_relative_runtime_path(value)
    except ValueError:
        return False
    return (verifier_dir / Path(*relative.parts)).is_file()


def _safe_relative_runtime_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError("reward-kit paths must be safe relative paths")
    return path


def _append_definition_drift_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    paths: TaskPaths,
    sandbox: str,
) -> None:
    try:
        document = TaskDocument.from_path(paths.task_document_path)
        legacy = TaskConfig.model_validate_toml(paths.config_path.read_text())
    except Exception as e:
        _issue(
            unsupported,
            path="task.md",
            reason=f"cannot prove native and legacy definitions equivalent: {e}",
            sandbox=sandbox,
        )
        return

    if document.config.model_dump() != legacy.model_dump():
        _issue(
            unsupported,
            path="task.md",
            reason="native task.md config and legacy task.toml config differ",
            sandbox=sandbox,
        )
    if _normalize_prompt(document.instruction) != _normalize_prompt(
        paths.instruction_path.read_text()
    ):
        _issue(
            unsupported,
            path="task.md",
            reason="native task.md prompt and legacy instruction.md prompt differ",
            sandbox=sandbox,
        )


def _append_alias_drift_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    native_dir: Path,
    legacy_dir: Path,
    path: str,
    reason: str,
    sandbox: str,
) -> None:
    if not (native_dir.exists() and legacy_dir.exists()):
        return
    if _file_digest_map(native_dir) != _file_digest_map(legacy_dir):
        _issue(unsupported, path=path, reason=reason, sandbox=sandbox)


def _file_digest_map(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _normalize_prompt(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())
