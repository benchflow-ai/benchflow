"""Runtime capability checks for parsed task packages.

The authoring parser intentionally accepts the task.toml task surface,
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

from benchflow._utils.modal_gpu import ModalGPURequestError, modal_gpu_request
from benchflow.rewards.rubric_config import criteria_aggregate_policy_from_rubric
from benchflow.sandbox._compose import compose_definition_path
from benchflow.sandbox.providers import (
    ALLOWLIST_SERVICE_CHECK_PROVIDERS,
    ALLOWLIST_UNSUPPORTED_PROVIDERS,
    DENYLIST_UNSUPPORTED_PROVIDERS,
    NO_NETWORK_UNSUPPORTED_PROVIDERS,
    SANDBOX_PROVIDER_SET,
    SANDBOX_PROVIDERS,
    SINGLE_CONTAINER_PROVIDERS,
    VERIFIER_SERVICE_PROVIDERS,
    providers_phrase,
)
from benchflow.task.artifacts import ARTIFACT_SANDBOXES, artifact_spec_issue
from benchflow.task.config import (
    NetworkMode,
    TaskConfig,
    TaskOS,
)
from benchflow.task.document import TaskDocument
from benchflow.task.imports import unhonoured_foreign_key
from benchflow.task.paths import TaskPaths, local_script_strategy_files
from benchflow.task.prompts import (
    CompiledUserRuntime,
    compile_document_user_runtime,
    compile_task_prompt_plan,
)
from benchflow.task.verifier_document import load_verifier_document
from benchflow.task.verifier_sandbox import (
    separate_verifier_issues,
    separate_verifier_requested,
    verifier_image_issue,
    verifier_owned_overlap,
    verifier_services_issue,
)

#: Where the separate-mode transfer refusals are filed. Strict integrity keeps
#: every issue under ``verifier.sandbox`` (integrity.trial.strict_launch_issues).
VERIFIER_TRANSFER_ISSUE_PATH = "verifier.sandbox.transfer"
#: Where the refusal of a separate verifier for a task that keeps services is
#: filed (DF-4).
KEPT_SERVICES_ISSUE_PATH = "metadata.x-keep-services"
#: Where the refusal of an identity-switch task on a backend without the
#: container's no_new_privs is filed (K1-R15-2).
IDENTITY_SWITCH_ISSUE_PATH = "metadata.x-agent-identity-switch"
#: The one backend that runs a task asking for GPUs (sandbox.gpus or
#: sandbox.gpu_types); every other backend refuses it below.
GPU_SANDBOX = "modal"


def sandbox_for_task(config: object) -> str:
    """The backend a check assumes when it is given none: docker, or
    :data:`GPU_SANDBOX` for a task that asks for GPUs (docker refuses one,
    so checking it there could only fail)."""
    sandbox = getattr(config, "sandbox", None)
    if getattr(sandbox, "gpus", 0) or getattr(sandbox, "gpu_types", None):
        return GPU_SANDBOX
    return "docker"


# Back-compat name kept for this module's existing membership check; the
# canonical set lives in benchflow.sandbox.providers.
SUPPORTED_SANDBOX_BACKENDS = SANDBOX_PROVIDER_SET


@dataclass(frozen=True)
class UnsupportedTaskFeature:
    """A parsed task feature the selected runtime cannot execute."""

    path: str
    reason: str
    sandbox: str
    # The refusal's code, shared with the Rust engine (R401: the task asks
    # for what this runner cannot do yet; `bench explain R401`).
    code: str = "R401"

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
    _append_kept_services_issue(
        unsupported,
        config=config,
        task_dir=Path(task_dir) if task_dir is not None else None,
        sandbox=sandbox,
    )
    _append_identity_switch_issue(
        unsupported,
        config=config,
        task_dir=Path(task_dir) if task_dir is not None else None,
        sandbox=sandbox,
    )
    _append_allowlist_services_issue(
        unsupported,
        config=config,
        task_dir=Path(task_dir) if task_dir is not None else None,
        sandbox=sandbox,
    )
    if document is not None:
        _append_document_issues(unsupported, document=document, sandbox=sandbox)
    if task_dir is not None:
        _append_layout_issues(unsupported, task_dir=Path(task_dir), sandbox=sandbox)
        _append_verifier_image_issue(
            unsupported, config=config, task_dir=Path(task_dir), sandbox=sandbox
        )
    return unsupported


def _append_kept_services_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    config: TaskConfig,
    task_dir: Path | None,
    sandbox: str,
) -> None:
    """A separate verifier sandbox has none of the services the task keeps.

    Its tests would reach no listener and score a plain 0, like an agent
    failure (DF-4), so the task is refused before anything runs: by the
    run's plan gate, at launch for a task that declares the separate verifier
    itself, and by ``bench check`` with the same words.
    """
    if not separate_verifier_requested(config):
        return
    from types import SimpleNamespace

    from benchflow.task import keep_services

    kept = keep_services.for_task(SimpleNamespace(config=config, task_dir=task_dir))
    if kept is None:
        return
    ports = ", ".join(service.label() for service in kept.services)
    _issue(
        unsupported,
        path=KEPT_SERVICES_ISSUE_PATH,
        reason=(
            f"the tests talk to the services the agent leaves running on port "
            f"{ports} (from {kept.describe_source()}), and a separate verifier "
            "sandbox has none of them, so every trial would score 0; run the "
            "verifier in the agent's sandbox (no --verifier-sandbox separate, "
            "--separate-verifier or --integrity strict, and no separate "
            f"verifier sandbox_mode in the task), or remove {keep_services.KEY}"
        ),
        sandbox=sandbox,
    )


def _append_identity_switch_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    config: TaskConfig,
    task_dir: Path | None,
    sandbox: str,
) -> None:
    """An identity-switch task runs only where the agent's container has
    no_new_privs from creation (K1-R15-2): refused before any sandbox starts
    by the run's plan gate, at launch, and by ``bench check``. Without the
    task's folder a compose file cannot be seen, so Daytona is refused."""
    from benchflow.task import identity_switch

    if not identity_switch.declared(config):
        return
    compose = (
        task_dir is not None
        and compose_definition_path(task_dir / "environment") is not None
    )
    why = identity_switch.backend_refusal(sandbox, compose=compose)
    if why is not None:
        _issue(
            unsupported,
            path=IDENTITY_SWITCH_ISSUE_PATH,
            reason=why,
            sandbox=sandbox,
        )


def _append_allowlist_services_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    config: TaskConfig,
    task_dir: Path | None,
    sandbox: str,
) -> None:
    """Refuse an allowlist task with compose services where they go unchecked.

    Only backends that read the running project keep an allowlist's services
    from fetching what it refuses (``egress_denylist.task_service_policy``).
    Elsewhere such a task is refused before any sandbox starts: the launch
    gate always passes the task's folder. A check of the config alone, with
    no folder, cannot see a compose file and leaves the call to that gate.
    Single-container backends refuse any compose file on their own.
    """
    if (
        sandbox in ALLOWLIST_SERVICE_CHECK_PROVIDERS
        or sandbox in SINGLE_CONTAINER_PROVIDERS
        or NetworkMode.ALLOWLIST
        not in (config.sandbox.network_mode, config.agent.network_mode)
    ):
        return
    if task_dir is None or compose_definition_path(task_dir / "environment") is None:
        return
    checked = " and ".join(sorted(ALLOWLIST_SERVICE_CHECK_PROVIDERS))
    _issue(
        unsupported,
        path="sandbox.network_mode",
        reason=(
            "network_mode='allowlist' with compose services needs a sandbox that "
            f"checks they have no route out ({checked}); {sandbox} does not"
        ),
        sandbox=sandbox,
    )


def _append_verifier_image_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    config: TaskConfig,
    task_dir: Path,
    sandbox: str,
) -> None:
    """A separate verifier needs an image to start from (see plan_verifier_image)."""
    # Not when the mode itself is refused here: the image would not help.
    if not separate_verifier_requested(config) or any(
        issue.path == "verifier.sandbox_mode" for issue in unsupported
    ):
        return
    problem = verifier_image_issue(config, task_dir)
    if problem is not None:
        _issue(
            unsupported,
            path=(
                "verifier.sandbox"
                if config.verifier.sandbox is not None
                else "verifier.sandbox_mode"
            ),
            reason=problem,
            sandbox=sandbox,
        )
        return
    services = verifier_services_issue(config, task_dir)
    if services is not None:
        _issue(
            unsupported,
            path="verifier.sandbox.services",
            reason=services,
            sandbox=sandbox,
        )


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
        # Multi-step tasks run their steps as rounds of one sandbox
        # (task/steps.py); each step's soft verify needs the tests in it.
        if separate_verifier_requested(config):
            _issue(
                unsupported,
                path="steps",
                reason="a multi-step task's steps are verified in the agent's "
                "sandbox, so it cannot use a separate verifier sandbox",
                sandbox=sandbox,
            )
        names = [step.name for step in config.steps]
        if len(set(names)) != len(names):
            _issue(
                unsupported,
                path="steps",
                reason="step names must be unique (each is a folder under steps/)",
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
            if step.verifier.collect:
                _issue(
                    unsupported,
                    path=f"steps[{i}].verifier.collect",
                    reason="a step's collect hooks are parsed but not executed "
                    "(only the task's [[verifier.collect]] hooks run)",
                    sandbox=sandbox,
                )

    refused: set[str] = set()
    for path in config.ignored_keys:
        found = unhonoured_foreign_key(path)
        if found is not None and found[0] not in refused:
            refused.add(found[0])
            _issue(unsupported, path=found[0], reason=found[1], sandbox=sandbox)

    if config.artifacts and sandbox not in ARTIFACT_SANDBOXES:
        _issue(
            unsupported,
            path="artifacts",
            reason="root artifact collection is implemented for docker and daytona",
            sandbox=sandbox,
        )
    for i, item in enumerate(config.artifacts):
        problem = artifact_spec_issue(item)
        if problem is not None:
            _issue(unsupported, path=f"artifacts[{i}]", reason=problem, sandbox=sandbox)

    _append_network_issue(
        unsupported,
        path="agent.network_mode",
        mode=config.agent.network_mode,
        sandbox=sandbox,
    )
    if config.agent.network_mode == NetworkMode.ALLOWLIST and (
        config.sandbox.network_mode in (NetworkMode.NO_NETWORK, NetworkMode.DENYLIST)
    ):
        _issue(
            unsupported,
            path="agent.network_mode",
            reason=(
                "agent.network_mode='allowlist' cannot combine with "
                f"sandbox.network_mode={config.sandbox.network_mode.value!r}"
            ),
            sandbox=sandbox,
        )
    _append_network_issue(
        unsupported,
        path="sandbox.network_mode",
        mode=config.sandbox.network_mode,
        sandbox=sandbox,
    )
    if config.verifier.network_mode == NetworkMode.ALLOWLIST:
        # The allowlist binds the sandbox user's uid; the verifier runs as
        # root outside that firewall, so a verifier allowlist is not enforced.
        _issue(
            unsupported,
            path="verifier.network_mode",
            reason="verifier network allowlists are not enforced (the verifier runs outside the agent uid firewall)",
            sandbox=sandbox,
        )
    else:
        _append_network_issue(
            unsupported,
            path="verifier.network_mode",
            mode=config.verifier.network_mode,
            sandbox=sandbox,
        )

    if separate_verifier_requested(config):
        for reason in separate_verifier_issues(config, sandbox=sandbox):
            _issue(
                unsupported,
                path=(
                    "verifier.sandbox.network_mode"
                    if reason.startswith("verifier.sandbox.network_mode")
                    else "verifier.sandbox_mode"
                ),
                reason=reason,
                sandbox=sandbox,
            )
        _append_verifier_transfer_issues(unsupported, config=config, sandbox=sandbox)
    if config.verifier.service != "main" and sandbox not in VERIFIER_SERVICE_PROVIDERS:
        _issue(
            unsupported,
            path="verifier.service",
            reason=(
                "a verifier in a compose service other than main needs a backend "
                "that runs it in that service: "
                + ", ".join(
                    name
                    for name in SANDBOX_PROVIDERS
                    if name in VERIFIER_SERVICE_PROVIDERS
                )
            ),
            sandbox=sandbox,
        )

    env = config.sandbox
    if env.os == TaskOS.WINDOWS:
        _issue(
            unsupported,
            path="sandbox.os",
            reason="Windows task environments are parsed but not executable",
            sandbox=sandbox,
        )
    if env.tpu is not None:
        _issue(
            unsupported,
            path="sandbox.tpu",
            reason="TPU scheduling is parsed but not implemented",
            sandbox=sandbox,
        )
    if sandbox == GPU_SANDBOX:
        try:
            modal_gpu_request(env.gpus, env.gpu_types)
        except ModalGPURequestError as exc:
            _issue(unsupported, path=exc.path, reason=str(exc), sandbox=sandbox)
    elif env.gpus or env.gpu_types:
        _issue(
            unsupported,
            path="sandbox.gpus",
            reason="GPU task environments require the Modal sandbox backend",
            sandbox=sandbox,
        )
    _append_workdir_issue(
        unsupported,
        workdir=env.workdir,
        sandbox=sandbox,
    )


def _append_verifier_transfer_issues(
    unsupported: list[UnsupportedTaskFeature],
    *,
    config: TaskConfig,
    sandbox: str,
) -> None:
    """Refuse declared transfer roots that overlap a verifier-owned path.

    A separate verifier gets the workspace and the declared artifacts at their
    sandbox paths. One at, under or above ``/tests``, ``/logs/verifier`` or
    another verifier-owned path would put the agent's files where the
    verifier's own files go (``tests/`` is uploaded over ``/tests`` without
    clearing it). The image's WORKDIR and overlay artifacts are checked again
    at transfer time, where they are known.
    """
    workdir = config.sandbox.workdir
    if not isinstance(workdir, str) or not workdir.strip():
        workdir = None
    if workdir is not None:
        owned = verifier_owned_overlap(workdir)
        if owned is not None:
            _issue(
                unsupported,
                path=VERIFIER_TRANSFER_ISSUE_PATH,
                reason=(
                    f"sandbox.workdir {workdir!r} overlaps verifier-owned {owned!r}; "
                    "the separate verifier would receive the workspace there"
                ),
                sandbox=sandbox,
            )
    for index, item in enumerate(config.artifacts):
        source = item if isinstance(item, str) else item.source
        if not source.startswith("/") and workdir is not None:
            source = str(PurePosixPath(workdir) / source)
        owned = verifier_owned_overlap(source)
        if owned is not None:
            _issue(
                unsupported,
                path=VERIFIER_TRANSFER_ISSUE_PATH,
                reason=(
                    f"artifacts[{index}] {source!r} overlaps verifier-owned "
                    f"{owned!r}; the separate verifier would receive it there"
                ),
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
            path="sandbox.workdir",
            reason="configured workdir must be a non-empty absolute path",
            sandbox=sandbox,
        )
        return
    path = PurePosixPath(workdir)
    if not path.is_absolute() or path == PurePosixPath("/"):
        _issue(
            unsupported,
            path="sandbox.workdir",
            reason="configured workdir must be an absolute non-root path",
            sandbox=sandbox,
        )


def _append_network_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    path: str,
    mode: NetworkMode | None,
    sandbox: str,
) -> None:
    if mode == NetworkMode.NO_NETWORK and sandbox in NO_NETWORK_UNSUPPORTED_PROVIDERS:
        _issue(
            unsupported,
            path=path,
            reason=f"network_mode='no-network' is not enforced by {sandbox}",
            sandbox=sandbox,
        )
    if mode == NetworkMode.ALLOWLIST and sandbox in ALLOWLIST_UNSUPPORTED_PROVIDERS:
        _issue(
            unsupported,
            path=path,
            reason=f"network_mode='allowlist' is not enforced by {sandbox}",
            sandbox=sandbox,
        )
    if mode == NetworkMode.DENYLIST and sandbox in DENYLIST_UNSUPPORTED_PROVIDERS:
        _issue(
            unsupported,
            path=path,
            reason=f"network_mode='denylist' is not enforced by {sandbox}",
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
            "criteria_runner",
        }:
            continue
        _issue(
            unsupported,
            path=f"benchflow.verifier.implementation.strategies.{strategy_name}",
            reason=(
                f"verifier strategy {strategy_name!r} is an unknown value; "
                "choose one of: script, deterministic, test-script, llm-judge, "
                "agent-judge, ors-episode, criteria-runner"
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
    _append_compose_issue(unsupported, task_dir=task_dir, sandbox=sandbox)


def _append_compose_issue(
    unsupported: list[UnsupportedTaskFeature],
    *,
    task_dir: Path,
    sandbox: str,
) -> None:
    """Refuse a multi-service task on a backend that runs one container.

    Such a task would otherwise launch with its side services missing, and the
    resulting failure would be attributed to the agent rather than reported as
    an environment the backend cannot host.
    """
    if sandbox not in SINGLE_CONTAINER_PROVIDERS:
        return
    compose = compose_definition_path(task_dir / "environment")
    if compose is None:
        return
    _issue(
        unsupported,
        path=f"environment/{compose.name}",
        reason=(
            f"multi-service compose topologies are not supported by {sandbox}; "
            "use the docker or daytona sandbox"
        ),
        sandbox=sandbox,
    )


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
    if strategy.type == "criteria-runner":
        _append_criteria_runner_strategy_issue(
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
    if strategy.type == "taskmd":
        # A materialized task.md draft 2 package (benchflow.taskmd): its
        # writer checked everything else when it refused unsupported fields.
        command = strategy.command
        if command is not None and not (paths.tests_dir / command).is_file():
            _issue(
                unsupported,
                path=f"verifier.strategies.{strategy.name}.command",
                reason=f"the verifier package has no {command}",
                sandbox=sandbox,
            )
        return
    if strategy.type not in {"script", "llm-judge", "agent-judge", "ors-episode"}:
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy.name}",
            reason=(
                f"{strategy.type} verifier strategies are parsed but not "
                "executable; select script, llm-judge, criteria-runner, agent-judge, "
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


def _append_criteria_runner_strategy_issue(
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
            reason="criteria-runner strategy must declare root",
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
            reason=f"criteria-runner runner not found at {runner}",
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
            reason=f"criteria-runner criteria not found at {criteria_file}",
            sandbox=sandbox,
        )
        return
    try:
        criteria_aggregate_policy_from_rubric(criteria_file)
    except ValueError as e:
        _issue(
            unsupported,
            path=f"verifier.strategies.{strategy_name}",
            reason=f"criteria-runner criteria are invalid: {e}",
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
        raise ValueError("criteria-runner paths must be safe relative paths")
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
