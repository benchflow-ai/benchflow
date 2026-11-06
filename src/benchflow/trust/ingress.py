"""The ingress registry (TRUSTED-RUNTIME v2 §5).

The rule that holds every transfer is structural: each transfer primitive
(``upload_file``, ``upload_dir``, ``live_process``,
``script_runner_upload``) of every sandbox backend
requires an :class:`~benchflow.trust.state.Admission`, which only
:func:`benchflow.trust.state.admit` makes. ``BaseSandbox`` installs the check
on every backend's primitives (:func:`benchflow.trust.state.guard_transfer`),
so a call on any path without an admission for that sandbox, in its current
generation, reaching that service, raises
:class:`~benchflow.trust.state.AdmissionRequired` before any byte moves.
An ingress is admitted with its :class:`~benchflow.trust.state.IngressKind`
(TAINTED is on disk first); bytes that are the task's or BenchFlow's own are
admitted as :data:`~benchflow.trust.state.TRUSTED_TRANSFER` with the reason.

``tests/test_trust_ingress_registry.py`` reads this module and the source
tree, and holds these rules besides:

1. Every reference to a transfer primitive in ``benchflow`` is a call that
   passes ``admission=``, or the callable of a ``functools.partial`` that
   does, or is inside a backend's own definition of a primitive (the guard
   admitted that call), or is listed in :data:`NOT_SANDBOX`. Any other form
   (an alias, a bare callback) is refused.
2. Every ``TRUSTED_TRANSFER`` admission states its reason as a literal.
3. Every definition of a **self-admitting primitive**
   (:data:`SELF_ADMITTING`) calls ``admit`` or ``restore_from``, or is listed
   in :data:`SELF_ADMITTING_EXEMPT` with the reason.
4. Every function in :data:`ADMITTING` calls ``admit`` with the kind listed.
5. Every :class:`~benchflow.trust.state.IngressKind` has at least one
   admitting site, and no entry names a function that no longer exists.

Agents reach the sandbox only through admitted paths: ACP and native launches
(``live_process``), the operator script agent (``_run_script``), the agent
exec barrier (``agent_exec_call``), session
factories (``connect_session_factory``, whose sandbox wrapper passes every
other method through after that admit) and the SDK's ``Environment``.
Shared volumes and sockets are reached through the default of
:func:`benchflow.trust.state.admit`: an ingress taints every known service
unless the backend names exactly which ones it reaches.
"""

from __future__ import annotations

from benchflow.trust.state import TRANSFER_PRIMITIVES as _PRIMITIVES
from benchflow.trust.state import IngressKind

#: Calls that move bytes or start processes in a sandbox.
TRANSFER_PRIMITIVES: frozenset[str] = frozenset(_PRIMITIVES)

#: Primitives whose own definitions admit, so their callers need not.
SELF_ADMITTING: frozenset[str] = frozenset(
    {
        "_run_oracle",
        "adopt_snapshot",
        "agent_exec_call",
        "connect_session_factory",
        "install_uploaded_workspace",
        "restore",
        "restore_submission",
        "workspace_restore",
    }
)

#: Definitions named like a self-admitting primitive that do not admit
#: themselves: ``module:qualname`` -> reason.
SELF_ADMITTING_EXEMPT: dict[str, str] = {
    "benchflow.sandbox._base:BaseSandbox.restore": (
        "raises SandboxSnapshotNotSupported; restores nothing"
    ),
    "benchflow.sandbox.remote_docker:RemoteDockerSandbox.restore": (
        "delegates to DockerSandbox.restore, which inherits the lineage"
    ),
    "benchflow.sandbox.remote_docker:RemoteDockerSandbox.adopt_snapshot": (
        "delegates to DockerSandbox.adopt_snapshot, which admits"
    ),
    "benchflow.sandbox.daytona_strategies:_DaytonaDirect.restore": (
        "called only by DaytonaSandbox.restore, which inherits the lineage first"
    ),
    "benchflow.sandbox.daytona_strategies:_DaytonaDirect.adopt_snapshot": (
        "called only by DaytonaSandbox.adopt_snapshot, which admits first"
    ),
    "benchflow.branch:restore": (
        "calls the sandbox's restore and the environment's restore, which admit"
    ),
    "benchflow.rollout_planes:DefaultRolloutPlanes.connect_session_factory": (
        "delegates to session_factory_runtime.connect_session_factory"
    ),
    "benchflow.sdk:SDK._run_oracle": (
        "delegates to rollout._setup._run_oracle, which admits"
    ),
    "benchflow.budget:BudgetGuard.restore": "a budget counter, not a sandbox",
    "benchflow.environment.manifest_env:ManifestEnvironment.restore": (
        "calls ManifestEnvironment._restore, which admits"
    ),
    "benchflow.sandbox._recovery_submission:restore_submission": (
        "the in-sandbox replacement step that rollout._recovery_submission."
        "restore_submission runs after it admits"
    ),
    "benchflow.sandbox.daytona_strategies:_DaytonaStrategy.restore": (
        "raises SandboxSnapshotNotSupported; restores nothing"
    ),
}

#: Functions that must admit, with the kind they admit.
ADMITTING: dict[str, IngressKind] = {
    "benchflow.runtime:Environment.exec": IngressKind.SDK_EXEC,
    "benchflow.runtime:Environment.exec_in_service": IngressKind.SDK_EXEC,
    "benchflow.runtime:Environment.upload_file": IngressKind.SDK_UPLOAD,
    "benchflow.runtime:Environment.upload_dir": IngressKind.SDK_UPLOAD,
    "benchflow.sandbox.agent_exec:agent_exec_call": IngressKind.SDK_EXEC,
    "benchflow.rollout.session_factory_runtime:connect_session_factory": (
        IngressKind.SESSION_FACTORY
    ),
    "benchflow.acp.runtime:connect_acp": IngressKind.AGENT_LAUNCH,
    "benchflow.native_harness.client:NativeCLIClient.prompt": (
        IngressKind.AGENT_LAUNCH
    ),
    "benchflow.rollout._setup:_run_oracle": IngressKind.ORACLE_LAUNCH,
    "benchflow.rollout._setup:_run_script": IngressKind.AGENT_LAUNCH,
    "benchflow.rollout._setup:_start_env_and_upload": (
        IngressKind.AGENT_DERIVED_UPLOAD
    ),
    "benchflow.rollout._setup:_publish_trajectory_for_verifier": (
        IngressKind.AGENT_DERIVED_UPLOAD
    ),
    # A learner store's skills, which earlier agents wrote (review M3).
    "benchflow.agents.install:_upload_skills": IngressKind.AGENT_DERIVED_UPLOAD,
    "benchflow.review.evidence:install_uploaded_workspace": (
        IngressKind.AGENT_DERIVED_UPLOAD
    ),
    "benchflow.taskmd.verify:SandboxRunner._stash": IngressKind.AGENT_DERIVED_UPLOAD,
    "benchflow.task.equivalence:SandboxAnswerGrader._write": (
        IngressKind.AGENT_DERIVED_UPLOAD
    ),
    "benchflow.eval_regrade:_restore_logs_artifacts": IngressKind.REGRADE_TRANSFER,
    "benchflow.eval_regrade:_install_bundle": IngressKind.REGRADE_TRANSFER,
    "benchflow.rollout._verifier_recovery:_recover_verifier": (
        IngressKind.RECOVERY_TRANSFER
    ),
    "benchflow.rollout._recovery_submission:restore_submission": (
        IngressKind.RECOVERY_TRANSFER
    ),
    "benchflow.rollout._separate_verifier:_check_destinations": (
        IngressKind.SEPARATE_VERIFIER_TRANSFER
    ),
    "benchflow.rollout._separate_verifier:_install_payload": (
        IngressKind.SEPARATE_VERIFIER_TRANSFER
    ),
    "benchflow.sandbox.docker:DockerSandbox.adopt_snapshot": (
        IngressKind.SNAPSHOT_RESTORE
    ),
    "benchflow.sandbox.daytona:DaytonaSandbox.adopt_snapshot": (
        IngressKind.SNAPSHOT_RESTORE
    ),
    "benchflow.sandbox.snapshot:workspace_restore": IngressKind.SNAPSHOT_RESTORE,
    "benchflow.environment.manifest_env:ManifestEnvironment._restore": (
        IngressKind.SNAPSHOT_RESTORE
    ),
    "benchflow.continue_run.sandbox_proxy:SandboxReplayProxy.start": (
        IngressKind.CONTINUATION_REPLAY
    ),
    "benchflow.rollout._user_loop:_activate_step_skills": IngressKind.USER_CONTEXT,
}

#: Functions that inherit a snapshot's trust state (restore_from) before any
#: restored process starts: restores, and starts of branch children.
INHERITING: frozenset[str] = frozenset(
    {
        "benchflow.sandbox.docker:DockerSandbox.restore",
        "benchflow.sandbox.daytona:DaytonaSandbox.restore",
        "benchflow.sandbox.daytona:DaytonaSandbox.start",
    }
)

#: Kinds whose ingress is admitted inside the state module or by a default
#: rather than at a call site of their own: kind -> where.
KINDS_ADMITTED_ELSEWHERE: dict[IngressKind, str] = {
    IngressKind.BRANCH_CHILD: (
        "branch children start from a snapshot (DaytonaSandbox.start) or are "
        "restored (DockerSandbox.restore); restore_from admits SNAPSHOT_RESTORE"
    ),
    IngressKind.REVIEW_RESUME_TRANSFER: (
        "review resume moves bytes only through rollout._verifier_recovery "
        "(RECOVERY_TRANSFER) and the reviewer's configured uploads "
        "(AGENT_DERIVED_UPLOAD)"
    ),
    IngressKind.SHARED_MOUNT: (
        "every admit reaches every known service unless the backend names "
        "exactly which services an ingress reaches"
    ),
}

#: Functions whose calls named like a transfer primitive are not a sandbox's:
#: ``module:qualname`` -> what they call.
NOT_SANDBOX: dict[str, str] = {
    "benchflow.publish.huggingface:publish_file_to_hf": (
        "HfApi.upload_file, a Hugging Face upload"
    ),
    "benchflow.sandbox.daytona:DaytonaSandbox._sdk_upload_file": (
        "the Daytona SDK's fs.upload_file, beneath the guarded primitives and "
        "BenchFlow's own DinD staging"
    ),
    "benchflow.sandbox.daytona:DaytonaSandbox._sdk_upload_dir": (
        "the Daytona SDK's fs.upload_file for one directory's archive, beneath "
        "the guarded upload_dir and BenchFlow's own DinD staging, into a fresh "
        "folder of root's staging folder that root's shell proves root's alone, "
        "unpacks and removes"
    ),
    "benchflow.sandbox.daytona:DaytonaSandbox._sdk_upload_bytes": (
        "the Daytona SDK's fs.upload_file for one exec's own command text, "
        "staged by the exec transport (sandbox/_transport.py) into root's "
        "staging folder, read by root's shell as that exec's stdin and removed "
        "after it; it lands no file any later step reads"
    ),
    "benchflow.sandbox.daytona_strategies:_DaytonaCredentialOps.run_as": (
        "the Daytona SDK's fs.upload_file for a credential file the host kept "
        "out of a snapshot, put back on restore"
    ),
}
