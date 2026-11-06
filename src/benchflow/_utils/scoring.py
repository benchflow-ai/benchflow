"""Pure scoring and classification helpers — no external dependencies."""

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from benchflow.diagnostics import (
    DIAGNOSTIC_REASON_IDLE_TIMEOUT,
    TRANSIENT_SANDBOX_TRANSPORT_MARKER,
)
from benchflow.rewards.validation import REWARD_OUT_OF_RANGE

# Error category constants
INSTALL_FAILED = "install_failure"
PIPE_CLOSED = "pipe_closed"
ACP_ERROR = "acp_error"
IDLE_TIMEOUT = DIAGNOSTIC_REASON_IDLE_TIMEOUT
INFRA_ERROR = "infra_failure"
SANDBOX_SETUP = "sandbox_setup"
KEPT_SERVICE_FREEZER = "kept_service_freezer"
RUNTIME_SEALING = "runtime_sealing"
GUARD_TIMEOUT = "guard_timeout"
INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
PROVIDER_AUTH = "provider_auth"
# A provider rate-limit/quota failure that surfaced as a *raised*
# ``AgentProtocolError -32603`` and was sanitized at the ACP boundary
# (``_classify_acp_error``). This is the manifestation of a Bedrock daily token
# cap ("Too many tokens per day"), which crashes the agent and is futile to
# retry in-batch — so it is non-retryable (no retry branch + listed in
# ``RetryConfig.exclude_categories``). A 429 that instead surfaces *silently*
# (zero tokens, no raise) is handled by the post-rollout ``_maybe_classify_api_error``
# path as ``api_error[rate_limit/transient]`` and stays retryable, because a
# self-surfacing throttle self-heals on backoff. The two paths track genuinely
# different failure shapes, so the differing retry verdicts are intentional.
PROVIDER_RATE_LIMIT = "provider_rate_limit"
# A provider *rejected* the request (HTTP 400) and it surfaced as a *raised*
# ``AgentProtocolError -32603`` sanitized at the ACP boundary
# (``_classify_acp_error``). The canonical case is a context-window/token-budget
# overflow (issue #830): the prompt + completion cap exceeds the model's context,
# which is deterministic and futile to retry in-batch. Like ``provider_auth`` it
# is permanent — no retry branch + listed in ``RetryConfig.exclude_categories``.
# A 400 that instead surfaces post-rollout (every captured request failed, zero
# tokens) is classified by ``_maybe_classify_api_error`` as
# ``api_error[rejected_request/permanent]`` and is already non-retryable.
PROVIDER_REJECTED = "provider_rejected"
TIMED_OUT = "timeout"
# Provider API failures detected post-rollout (rate limit, quota, rejected
# request, 5xx). "api_error" is proxy-proven (every captured provider request
# failed); "suspected_api_error" is the zero-signal heuristic (no proxy
# evidence, but the agent ended with zero tokens AND zero tool calls). Both
# null the reward so the slot is excluded from score denominators.
API_ERROR = "api_error"
SUSPECTED_API_ERROR = "suspected_api_error"
# The agent did nothing useful because its integration broke (install, login,
# launch); set by benchflow.integration_health with a named cause, unscored.
AGENT_INTEGRATION = "agent_integration"
# The login the agent runs on has used up its subscription usage (Claude's
# 5-hour or 7-day window, a Codex plan limit): a typed
# benchflow.agents.errors.UsageLimitError. Unscored, never retried (every
# retry on that login fails the same way until the reset), and it stops an
# Evaluation from starting more trials.
USAGE_LIMIT = "usage_limit"
# The task asks for something the selected sandbox cannot run, refused before
# any sandbox starts (benchflow.task.runtime_capabilities, code R401). The
# Rust engine records the same refusal under the same name
# (crates/benchflow-run/src/record.rs, ``category()``). Unscored, never
# retried: another attempt refuses the same way.
UNSUPPORTED_TASK_FEATURE = "unsupported_task_feature"
# UnsupportedTaskFeatureError's header. The recorded error is
# describe_exception(e), "UnsupportedTaskFeatureError: Task uses ...", or the
# bare message: the header opens the error either way. Anywhere else it is
# someone else's text: an ACP error is "ACP error <code>: <the agent's
# message>", and the agent must not make its own failure read as the task's
# refusal, which is never retried.
_UNSUPPORTED_TASK_FEATURE_MARKER = (
    "task uses parsed runtime features that benchflow cannot execute"
)
_UNSUPPORTED_TASK_FEATURE_CLASS = "unsupportedtaskfeatureerror: "
# The task's image did not build, for a reason another attempt repeats
# (benchflow.sandbox.image_build.ImageBuildError): the task's problem, never
# retried. Read head-anchored like the refusal above, so docker's or an
# agent's text quoted elsewhere in an error cannot claim it.
IMAGE_BUILD = "image_build"
IMAGE_BUILD_HEADER = "the task's image did not build"
# The registry refused this machine's login for the task's base image
# (benchflow.sandbox.image_build.RegistryAuthError): the user's setup, not
# the task's Dockerfile, never retried (review A-1).
REGISTRY_AUTH = "registry_auth"
REGISTRY_AUTH_HEADER = "the registry refused this machine's login for the base image"
_REGISTRY_AUTH_CLASS = "registryautherror: "
_IMAGE_BUILD_CLASS = "imagebuilderror: "
# The host's disk (the one holding the job folder) is full: the user's to
# fix, never retried (benchflow._utils.disk_space). An errored trial records
# the OSError BenchFlow's own write raised; an unscored one the empty reward
# file the verifier could not fill. Both open with the host's words, never
# the sandbox's: a full disk inside the sandbox is the task's or the agent's.
# A task file that does not load (a task.toml typo, a task.md that does not
# parse, a verifier document or rubric that is invalid): the task's problem,
# never retried; `bench check` names it (DF-p4-upgrader-8). Read from the
# recorded error's class, which opens it (describe_exception).
TASK_CONFIG = "task_config"
_TASK_CONFIG_CLASSES = (
    "taskconfigkeyerror: ",
    "taskdocumentparseerror: ",
    "verifierdocumentparseerror: ",
    "malformedtaskerror: ",
    "reviewrubricerror: ",
)
DISK_FULL = "disk_full"
DISK_FULL_MARKER = "the disk that holds the job folder is full"
_DISK_FULL_HEADS = (
    "unexpected: [errno 28]",
    "[errno 28]",
    "oserror: [errno 28]",
    "diskfullerror: ",
)

# Matched case-insensitively against the error string. Covers the
# human-authored markers plus the sanitized "provider auth failed (HTTP 401)"
# marker injected at the rollout/provider boundary (#546/#564), where the real
# 401/403 is visible only in the proxy trajectory, not the top-level ACP error.
_PROVIDER_AUTH_MARKERS = (
    "permission_denied",
    "leaked",
    "failed to authenticate",
    "invalid bearer token",
    "invalid api key",
    "was rejected as invalid",
    "unauthorized",
    "provider auth failed",
    "http 401",
    "http 403",
)
# Sanitized markers appended by ``_classify_acp_error`` when a raised ACP error
# hides a provider rate-limit (429) or outage (503). Matched case-insensitively
# against the lowercased error, same as the auth markers above.
_PROVIDER_RATE_LIMIT_MARKERS = (
    "provider rate limited",
    "rate limit",
    "too many requests",
    "http 429",
)
_PROVIDER_UNAVAILABLE_MARKERS = (
    "provider unavailable",
    "http 503",
)
# Sanitized marker appended by ``_classify_acp_error`` when a raised ACP error
# hides a provider request rejection (400) — most commonly a context-window
# overflow (#830). Matched case-insensitively, same as the markers above.
_PROVIDER_REJECTED_MARKERS = (
    "provider rejected request",
    "http 400",
)

# Verifier error category constants
VERIFIER_FAILED = "verifier_failure"
VERIFIER_INFRA = "verifier_infra"
VERIFIER_TIMEOUT = "verifier_timeout"
VERIFIER_DEP_INSTALL = "verifier_dep_install"
# The verifier wrote a reward outside the task's range (#1178): unscored, the
# raw number kept in result.json's reward_out_of_range_info.
VERIFIER_REWARD_OUT_OF_RANGE = "reward_out_of_range"

# Catalogs for fixed-schema exports; reuse the classifier constants rather
# than admitting arbitrary saved text as a category.
OTHER_ERROR = "other"
VERIFIER_OTHER = "verifier_other"
ERROR_CATEGORIES = frozenset(
    {
        INSTALL_FAILED,
        PIPE_CLOSED,
        ACP_ERROR,
        IDLE_TIMEOUT,
        INFRA_ERROR,
        SANDBOX_SETUP,
        KEPT_SERVICE_FREEZER,
        RUNTIME_SEALING,
        GUARD_TIMEOUT,
        INVALID_PROVIDER_RESPONSE,
        PROVIDER_AUTH,
        PROVIDER_RATE_LIMIT,
        PROVIDER_REJECTED,
        TIMED_OUT,
        API_ERROR,
        SUSPECTED_API_ERROR,
        AGENT_INTEGRATION,
        USAGE_LIMIT,
        UNSUPPORTED_TASK_FEATURE,
        IMAGE_BUILD,
        REGISTRY_AUTH,
        TASK_CONFIG,
        DISK_FULL,
        OTHER_ERROR,
    }
)
VERIFIER_ERROR_CATEGORIES = frozenset(
    {
        VERIFIER_FAILED,
        VERIFIER_INFRA,
        VERIFIER_TIMEOUT,
        VERIFIER_DEP_INSTALL,
        VERIFIER_REWARD_OUT_OF_RANGE,
        VERIFIER_OTHER,
        DISK_FULL,
        INFRA_ERROR,  # automatic scoring infrastructure errors
    }
)

# Canonical dependency-install markers shared by verifier stdout scanning and
# verifier-error classification. Keep these lower-case; the helper below
# performs case-insensitive matching.
VERIFIER_DEP_INSTALL_MARKERS: tuple[str, ...] = (
    "dependency install failed",
    "failed to download `",
    "failed to fetch:",
    "error sending request for url",
    "failed to lookup address information",
    "no solution found",
    "could not find a version",
    "resolution impossible",
)

ScoreOutcome = Literal["passed", "failed", "errored", "verifier_errored"]
ResultOutcome = Literal["passed", "failed", "errored", "verifier_errored", "unscored"]

# Outcome assessment for results whose execution and verdict are separate
# (embodied trials: a completed physical run is scored only after a reviewer
# has assessed its recorded evidence). Only an assessed result carries a score.
AssessmentStatus = Literal["pending", "verified", "failed", "unassessable"]
ASSESSMENT_STATUSES: frozenset[str] = frozenset(
    {"pending", "verified", "failed", "unassessable"}
)
ASSESSED_STATUSES: frozenset[str] = frozenset({"verified", "failed"})


def assessment_status(result: Mapping[str, Any]) -> str | None:
    """The result's outcome-assessment status, or None when it declares none.

    ``assessment`` may be a status string or a mapping with ``status``. A
    legacy row with ``status: "awaiting_assessment"`` (an older physical-trial
    manifest) is pending. An unrecognised value is reported as
    ``"unassessable"`` so it can never unlock a score.
    """
    assessment = result.get("assessment")
    if assessment is None:
        return "pending" if result.get("status") == "awaiting_assessment" else None
    status = assessment.get("status") if isinstance(assessment, Mapping) else assessment
    # A list or mapping here is malformed, not a reason to crash a summary.
    if isinstance(status, str) and status in ASSESSMENT_STATUSES:
        return status
    return "unassessable"


def assessment_withholds_score(result: Mapping[str, Any]) -> bool:
    """True when a declared assessment is not complete (pending/unassessable).

    Such a result is unscored: its reward is ignored, it can never count as
    passed, and it stays out of reward means, whatever else it carries.
    """
    status = assessment_status(result)
    return status is not None and status not in ASSESSED_STATUSES


def extract_reward(result: Mapping[str, Any]) -> float | None:
    """Extract the reward value from a result dict, or None if absent.

    A result whose declared assessment is pending or unassessable has no
    reward, even if a stale or agent-derived one is present.
    """
    if assessment_withholds_score(result):
        return None
    if result.get("scoring") is not None:
        from benchflow.review.outcome import scoring_from_result

        try:
            scoring = scoring_from_result(result)
        except ValueError:
            return None
        if scoring is None or scoring.status != "complete":
            return None
        rewards = scoring.numeric_rewards()
        assert rewards is not None
        return rewards["reward"]
    rewards = result.get("rewards")
    if not isinstance(rewards, dict):
        return None
    return rewards.get("reward")


#: Categories that name a cause and its remedy for reports only. Retry and
#: resume decide on the category the same error gets without them.
REPORTING_CATEGORIES = frozenset(
    {KEPT_SERVICE_FREEZER, RUNTIME_SEALING, GUARD_TIMEOUT, INVALID_PROVIDER_RESPONSE}
)
#: An invalid provider response keeps the error it replaced after this.
ORIGINAL_ERROR = "\noriginal error: "
#: A model request that got no response when the agent's deadline ended: a
#: transient provider error, so it is retried and resumed like one.
PROVIDER_STALL = "provider api error [provider_stall/transient]"


def classify_error(error: str | None) -> str | None:
    """Classify an error string into a category, or None if no error."""
    if not error:
        return None
    lower = error.lower()
    if lower.startswith("keptservicefreezerunavailable:"):
        return KEPT_SERVICE_FREEZER
    if lower.startswith("invalid provider response"):
        return INVALID_PROVIDER_RESPONSE
    if "benchflow could not prepare its protected runtime" in lower:
        return RUNTIME_SEALING
    if "sandboxdaemonguarderror:" in lower and (
        "exec status 124" in lower
        or "timed out" in lower
        or "guard did not report" in lower
    ):
        return GUARD_TIMEOUT
    return _classify_error(error)


def retry_category(error: str | None, category: str | None) -> str | None:
    """The category retry and resume decide on: a reporting category is
    replaced by the one its error (the original error, if it was replaced)
    gets without the reporting rules."""
    if category not in REPORTING_CATEGORIES or not error:
        return category
    return _classify_error(error.partition(ORIGINAL_ERROR)[2] or error)


def _classify_error(error: str) -> str | None:
    lower = error.lower()
    # First of all: a runtime refusal lists each feature with a free-text
    # reason ("... timed out", "install failed" can appear there), and none
    # of the markers below may claim it.
    head = lower.lstrip()
    head = head.removeprefix(_UNSUPPORTED_TASK_FEATURE_CLASS)
    if head.startswith(_UNSUPPORTED_TASK_FEATURE_MARKER):
        return UNSUPPORTED_TASK_FEATURE
    if head.startswith(_DISK_FULL_HEADS) or head.startswith(DISK_FULL_MARKER):
        return DISK_FULL
    if head.startswith(_TASK_CONFIG_CLASSES):
        return TASK_CONFIG
    built = lower.lstrip().removeprefix("sandbox startup failed: ")
    if built.removeprefix(_REGISTRY_AUTH_CLASS).startswith(REGISTRY_AUTH_HEADER):
        return REGISTRY_AUTH
    if built.removeprefix(_IMAGE_BUILD_CLASS).startswith(IMAGE_BUILD_HEADER):
        return IMAGE_BUILD
    # First: the evidence quoted after it may contain any other marker.
    if lower.startswith("agent integration failure"):
        return AGENT_INTEGRATION
    if lower.startswith("usage limit reached"):
        return USAGE_LIMIT
    from benchflow.agents.usage_limits import is_usage_limit_text

    if is_usage_limit_text(error):
        # An agent's own words ("You've hit your weekly limit · resets ..."),
        # possibly wrapped in "ACP error -32603: Internal error:".
        return USAGE_LIMIT
    if "agent idle for" in lower:
        return IDLE_TIMEOUT
    if "install failed" in lower:
        return INSTALL_FAILED
    if "closed stdout" in lower or lower.startswith(("pty ", "daytonaptyprocess:")):
        # The agent's pipe ended: a local or SSH process closed stdout, or the
        # Daytona PTY closed ("PTY closed by the peer: ...") or stayed silent
        # ("PTY readline timeout"). The transport's diagnostic says the same.
        return PIPE_CLOSED
    # Order matters: "suspected provider api error" contains "provider api
    # error", so the heuristic marker must be checked first.
    if "suspected provider api error" in lower:
        return SUSPECTED_API_ERROR
    if "provider api error" in lower:
        return API_ERROR
    # A native-harness CLI error (benchflow.native_harness) is the same kind
    # of failure as an ACP adapter's error response: the agent reported a
    # failed turn, or died mid-turn. One category keeps the two harnesses'
    # outcomes comparable.
    if (
        "acp error" in lower
        or "native harness error" in lower
        or "was rejected as invalid" in lower
    ):
        if any(m in lower for m in _PROVIDER_AUTH_MARKERS):
            return PROVIDER_AUTH
        if any(m in lower for m in _PROVIDER_RATE_LIMIT_MARKERS):
            return PROVIDER_RATE_LIMIT
        if any(m in lower for m in _PROVIDER_UNAVAILABLE_MARKERS):
            return INFRA_ERROR
        if any(m in lower for m in _PROVIDER_REJECTED_MARKERS):
            return PROVIDER_REJECTED
        return ACP_ERROR
    if (
        "sandbox startup" in lower
        or "sandbox creation" in lower
        or _looks_like_docker_registry_metadata_error(lower)
    ):
        return SANDBOX_SETUP
    if "prompt exceeded wall-clock budget" in lower:
        return TIMED_OUT
    if _looks_like_infra_error(lower):
        return INFRA_ERROR
    if "timed out" in lower:
        return TIMED_OUT
    return OTHER_ERROR


# Sandbox startup failures that another attempt cannot fix: the task's build
# context lacks a path the Dockerfile copies. Daytona's SDK checks the context
# locally ("Path does not exist: <path>"); BuildKit reports a missing COPY
# source as a cache-key checksum miss ending in '"<path>": not found'. And a
# Docker daemon that is not running (rollout/_setup.py names it) is still not
# running seconds later.
_UNRECOVERABLE_STARTUP_RES = (
    re.compile(r"path does not exist: "),
    # The image cannot keep the agent from Daytona's root daemon.
    re.compile(
        r"agent user could not be kept away from daytona.s root toolbox daemon: (?:the image has no iptables and no package manager|the image has ipv6 but no ip6tables)"
    ),
    # The image hands the agent user the firewall that refuses the cloud
    # metadata endpoints (lockdown.CloudMetadataPowerError).
    re.compile(r"the agent user '[^']*' can change the firewall through "),
    # The image lets another user change a library root's steps load
    # (lockdown.LoaderCustodyError).
    re.compile(
        r"sandbox startup refused: (?:in the task's image, |benchflow could not "
        r"seal its protected runtime: )the system loader "
    ),
    # The task's compose file puts main on the host's network, so the block
    # cannot go in, and the run did not opt out.
    re.compile(
        r"cloud metadata endpoints open to it \(the block is not_installed: "
        r"main (?:shares|may share) the host's network"
    ),
    re.compile(
        r"sandbox startup refused: the restricted agent launch could not be verified: "
    ),
    re.compile(r'failed to calculate checksum of ref \S+ "[^"]+": not found'),
    # The Daytona API rejected the key at create (sandbox/daytona_start).
    re.compile(r"daytona rejected daytona_api_key \(http 40[13]\)"),
    re.compile(r"docker daemon unreachable: "),
    re.compile(r"remote docker host unreachable: "),
    re.compile(r"remote docker host lacks resources: "),
    re.compile(r"remote docker host refuses this task: "),
)


def is_unrecoverable_startup_error(error: str | None) -> bool:
    """True for a sandbox startup failure that a retry would only repeat."""
    if not error:
        return False
    lower = error.lower()
    return any(pattern.search(lower) for pattern in _UNRECOVERABLE_STARTUP_RES)


def api_error_is_transient(error: str | None) -> bool:
    """True when an api_error string carries the transient marker.

    Provider-api-error strings are formatted by the rollout classifier as
    ``provider api error [<subcategory>/transient] ...`` or ``[.../permanent]``
    — transient (rate limit, 5xx) is retryable, permanent (auth, quota,
    model-not-found, rejected request) is not.
    """
    return bool(error) and "/transient]" in error


def _looks_like_infra_error(error: str) -> bool:
    return any(
        marker in error
        for marker in (
            "connection lost",
            "connection reset",
            "connection refused",
            "broken pipe",
            # Stamped by TransientSandboxTransportError at the sandbox-SDK
            # boundary, where the vendor exception type is still available to
            # judge. One marker covers every provider call — exec, upload,
            # download, stat — instead of one entry per vendor message
            # prefix, which would be endless and silently incomplete.
            TRANSIENT_SANDBOX_TRANSPORT_MARKER,
            # Predates the marker above and is kept for strings rebuilt
            # outside that boundary (e.g. a verifier error re-raised as text).
            "failed to get session command",
            "failed to execute session command",
            "command timed out after",
            "sandbox not found",
            # Daytona stopped the sandbox (sandbox/daytona_pty names the
            # inactivity stop); the SDK's error code, in lower case.
            "sandbox_not_running",
            "workspace not found",
            "api connection",
            "api timeout",
            "temporarily unavailable",
            "rollout exceeded host hard deadline",
        )
    )


def _looks_like_docker_registry_metadata_error(error: str) -> bool:
    if "docker compose command failed" not in error:
        return False
    return (
        "failed to resolve source metadata" in error
        or "failed to do request: head" in error
        or "deadlineexceeded" in error
    )


# How the rollout records a PluginGuardLoadError (``rollout/_setup.py``: the
# exception's class leads), optionally behind the recovery contract's
# ``[solver-preserved]`` tag (``rollout._verifier_recovery.PRESERVED_SOLVER``).
_PLUGIN_GUARD_ERROR_RE = re.compile(
    r"^(?:\[solver-preserved\] )?verifier crashed: PluginGuardLoadError: "
)


# The rollout's own infrastructure wrappers (``rollout/_setup.py``'s
# ``verifier_wedge:``, ``rollout/_verifier_recovery.py``'s recovery outcomes),
# matched at the start of the lower-cased error only.
_APP_INFRA_WRAPPER_RE = re.compile(
    r"^(?:\[solver-preserved\] )?(?:verifier crashed: (?:[a-z_][\w.]*: )?)?"
    r"(?:verifier_wedge:|verifier recovery )"
)
# The rollout's own wrapper of a verifier-only re-run that timed out again
# , at the start only, like the wrappers above.
_RECOVERY_TIMED_OUT_RE = re.compile(
    r"^(?:\[solver-preserved\] )?verifier recovery failed: verifier timed out"
)
# A plugin guard marker read that failed before any verdict was read
# (``task/verifier_core.py``): the sandbox's failure, not the guard's verdict
# (V2-R1-A-2). Matched at the start only, like the guard's own errors.
_PLUGIN_GUARD_MARKERS_UNREAD_RE = re.compile(
    r"^(?:\[solver-preserved\] )?verifier crashed: "
    r"PluginGuardMarkersUnreadableError: "
)
# BenchFlow's protected verifier runtime that a separate verifier
# sandbox could not build from PyPI before test.sh ran
# (``sandbox/_verifier_runtime.py``): the network's failure, retried, never a
# refusal that counts against the agent (V2-R1-B-1). Matched at the start only.
_PINNED_RELEASE_UNAVAILABLE_RE = re.compile(
    r"^(?:\[solver-preserved\] )?verifier crashed: RuntimeUnavailable: "
)


def plugin_guard_error_message(verifier_error: str | None) -> str | None:
    """The plugin guard's own message when ``verifier_error`` is its error.

    None for any other verifier error. Matched only at the start: the message
    carries text the agent chose (a planted dist-info name or path, or the
    error from inspecting a file it planted), so no marker inside it may decide
    what kind of error this is (V2-R1-A-1).
    """
    if not verifier_error:
        return None
    match = _PLUGIN_GUARD_ERROR_RE.match(verifier_error)
    return verifier_error[match.end() :] if match else None


# How a refusal opens the guard's message (``task/verifier_core.py``). The
# guard's name is hardening's, so no agent-chosen text can make an error one.
# A pytest stopped while the guard was still checking plugins counts too: the
# agent can bring that about (a FIFO it left as plugin metadata, under
# test.sh's own timeout), so it must not raise a score either (V2-R1-B-1).
_GUARD_REFUSAL_RE = re.compile(
    r"the pytest plugin guard _benchflow_guard_[0-9a-f]+ refused a pytest "
    r"plugin it could not trust"
    r"|pytest stopped while the plugin guard _benchflow_guard_[0-9a-f]+ was "
    r"still checking pytest plugins"
)


def is_guard_refusal(verifier_error: str | None) -> bool:
    """Whether *verifier_error* is a plugin guard refusal (or a stopped check)."""
    message = plugin_guard_error_message(verifier_error)
    return message is not None and _GUARD_REFUSAL_RE.match(message) is not None


def count_guard_refusals(results: Iterable[Mapping[str, Any]]) -> int:
    """Verifier-errored trials whose verifier error is a plugin guard refusal.

    Only those the score outcome puts in ``verifier_errored``, as
    ``BenchmarkMetrics.score_excl_errors`` counts them: a trial whose agent
    errored is excluded as errored, refusal or not (V2-R1-A-5).
    """
    return sum(
        1
        for r in results
        if is_guard_refusal(r.get("verifier_error"))
        and classify_score_outcome(r) == "verifier_errored"
    )


def classify_verifier_error(verifier_error: str | None) -> str | None:
    """Classify a verifier error string, or None if no error."""
    if not verifier_error:
        return None
    if plugin_guard_error_message(verifier_error) is not None:
        # The guard decided, deterministically, from what the sandbox holds:
        # never infrastructure, whatever words a refused path contains.
        return VERIFIER_FAILED
    if _PLUGIN_GUARD_MARKERS_UNREAD_RE.match(verifier_error):
        return VERIFIER_INFRA
    if _PINNED_RELEASE_UNAVAILABLE_RE.match(verifier_error):
        return VERIFIER_INFRA
    lower = verifier_error.lower()
    if lower.startswith("separate verifier "):
        # The verifier sandbox never received the agent's outputs (or never
        # started): nothing was assessed.
        return VERIFIER_INFRA
    # A reward out of range is the task's verdict even when a recovery
    # verifier wrote it ("verifier recovery failed: reward out of range: ..."),
    # so it is named before the recovery marker below: blaming infrastructure
    # would also send it back through recovery on resume.
    if REWARD_OUT_OF_RANGE in lower:
        return VERIFIER_REWARD_OUT_OF_RANGE
    if _RECOVERY_TIMED_OUT_RE.match(lower):
        # The one verifier-only re-run of a timed-out verifier timed out
        # again: the task's verifier_timeout, not the sandbox's.
        return VERIFIER_TIMEOUT
    if _APP_INFRA_WRAPPER_RE.match(lower):
        # Only the rollout's own wrappers, at the start: a phrase inside an
        # error's detail may be text the agent chose.
        return VERIFIER_INFRA
    if "verifier crashed" in verifier_error:
        if DISK_FULL_MARKER in lower:
            # The host's disk, not the task's verifier (benchflow._utils.disk_space).
            return DISK_FULL
        if contains_verifier_dep_install_marker(lower):
            return VERIFIER_DEP_INSTALL
        if _looks_like_verifier_infra_error(lower):
            return VERIFIER_INFRA
        return VERIFIER_FAILED
    if "verifier timed out" in verifier_error:
        return VERIFIER_TIMEOUT
    return VERIFIER_OTHER


def contains_verifier_dep_install_marker(text: str) -> bool:
    """Detect verifier dependency installation failures."""
    lower = text.lower()
    return any(marker in lower for marker in VERIFIER_DEP_INSTALL_MARKERS)


def _looks_like_verifier_infra_error(error: str) -> bool:
    # Verifier execution uses the same sandbox transport as agent execution.
    # Preserve retry parity for transport failures such as Daytona losing its
    # session command between execution phases.
    return _looks_like_infra_error(error) or any(
        marker in error
        for marker in (
            "failed to add tests directory",
            "failed to download verifier directory",
            "failed to download llm-judge input",
            "verifier setup failed",
        )
    )


def classify_result(
    *,
    reward: float | None,
    error: str | None,
    verifier_error: str | None,
) -> ScoreOutcome:
    """Classify a single result into exactly one score bucket.

    Returns one of ``"passed"``, ``"failed"``, ``"errored"``,
    ``"verifier_errored"``. The buckets are disjoint and exhaustive, so
    ``passed + failed + errored + verifier_errored == total`` holds
    structurally for any set of results.

    Precedence:

    1. A result with a reward is ``"passed"`` (reward == 1.0) or
       ``"failed"`` (any other reward) — an explicit reward always wins,
       even when an ``error`` string is also present (e.g. a warning).
    2. With no reward, an agent ``error`` makes it ``"errored"``.
    3. With no reward and no agent error, a ``verifier_error`` makes it
       ``"verifier_errored"``. ``errored`` therefore takes precedence over
       ``verifier_errored`` when both errors are present.
    4. With no reward and no error of either kind, the result is
       ``"errored"`` — a terminal result must land in some bucket, and an
       absent reward with no recorded cause is still a failure to produce
       a verdict.
    """
    if reward is not None:
        return "passed" if reward == 1.0 else "failed"
    if error:
        return "errored"
    if verifier_error:
        return "verifier_errored"
    return "errored"


def classify_score_outcome(result: Mapping[str, Any]) -> ScoreOutcome:
    """Classify a persisted result for score/invariant accounting.

    Integrated review uses its explicit gate verdict. Legacy results keep
    reward/agent-error precedence. Malformed or incomplete scoring is never a
    capability failure or an accidental pass from a stale reward. A pending or
    unassessable assessment has no reward, so it lands with the other
    no-verdict results (never ``passed``, outside ``score_excl_errors``).
    """
    if result.get("scoring") is not None and not assessment_withholds_score(result):
        from benchflow.review.outcome import scoring_from_result

        try:
            scoring = scoring_from_result(result)
        except ValueError:
            scoring = None
        if scoring is not None and scoring.status == "complete":
            return "passed" if scoring.passed else "failed"
        return "errored" if result.get("error") else "verifier_errored"
    return classify_result(
        reward=extract_reward(result),
        error=result.get("error"),
        verifier_error=result.get("verifier_error"),
    )


def classify_result_dict(result: Mapping[str, Any]) -> ScoreOutcome:
    """Backward-compatible alias for score/invariant accounting."""
    return classify_score_outcome(result)


def classify_audit_outcome(result: Mapping[str, Any]) -> ResultOutcome:
    """Classify a result for audit/reporting evidence.

    Audit summaries intentionally surface verifier evidence first. A task with
    ``verifier_error`` is counted as ``verifier_errored`` even if an agent
    error or stale reward is also present, because result auditors need the
    verifier failure to be visible instead of hidden behind the score bucket.
    """
    if result.get("verifier_error"):
        return "verifier_errored"
    if result.get("scoring") is not None and not assessment_withholds_score(result):
        return classify_score_outcome(result)
    reward = extract_reward(result)
    if reward == 1.0:
        return "passed"
    if reward is not None:
        return "failed"
    if result.get("error"):
        return "errored"
    return "unscored"


def classify_result_outcome(result: Mapping[str, Any]) -> ResultOutcome:
    """Backward-compatible alias for audit/reporting outcome accounting."""
    return classify_audit_outcome(result)


def _empty_counts(*, include_unscored: bool) -> dict[str, int]:
    counts = {
        "passed": 0,
        "failed": 0,
        "errored": 0,
        "verifier_errored": 0,
    }
    if include_unscored:
        counts["unscored"] = 0
    return counts


def count_score_outcomes(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count score buckets with reward/agent-error precedence."""
    counts = _empty_counts(include_unscored=False)
    for result in results:
        counts[classify_score_outcome(result)] += 1
    return counts


def count_audit_outcomes(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count audit buckets with verifier-evidence precedence."""
    counts = _empty_counts(include_unscored=True)
    for result in results:
        counts[classify_audit_outcome(result)] += 1
    return counts


def count_result_outcomes(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Backward-compatible alias for audit/reporting outcome accounting."""
    return count_audit_outcomes(results)


def finite_reward(value: object) -> float | None:
    """Return a numeric reward as a finite float, or None when it is not one.

    Bools are not reward magnitudes. Python's ``json`` parses an integer literal
    such as ``10**1000`` exactly, and ``float()`` rejects it with
    ``OverflowError``; it is treated like the infinity it would overflow to.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


#: What summary.json's ``mean_reward`` is, written beside it so a reader of
#: the file alone (a 0.7 CI gate on ``jq .mean_reward``) can tell.
MEAN_REWARD_RULE = (
    "every trial, an unscored one as 0, over mean_reward_n trials; "
    "mean_reward_scored_only is over scored trials only (BenchFlow 0.7's mean_reward)"
)


def mean_scored_reward(results: Iterable[Mapping[str, Any]]) -> float | None:
    """Mean reward over scored rollouts only, or None when nothing scored.

    BenchFlow 0.7's mean reward, kept as ``mean_reward_scored_only``; the
    default ``mean_reward`` is :func:`mean_reward_every_trial`. A rollout is
    scored when ``rewards.reward`` is a finite numeric value. Bools are
    excluded — ``classify_result`` treats a persisted ``true`` as a pass
    (``True == 1``), but a bool is not a reward magnitude — and so are
    non-finite or float-overflowing values, which the resume path can feed in
    unvalidated (Python's ``json`` round-trips NaN and huge integers).

    A trial the plugin guard refused (:func:`is_guard_refusal`) has no reward
    either, but counts here as 0: it is the agent's disqualification, and
    leaving it out would let a refusal planted only where the agent failed
    raise the mean.
    """
    rows = list(results)
    scored = _scored_rewards(rows)
    count = len(scored) + count_guard_refusals(rows)
    return sum(scored) / count if count else None


def scored_trial_count(results: Iterable[Mapping[str, Any]]) -> int:
    """The n of :func:`mean_scored_reward`: trials with a scored reward."""
    return len(_scored_rewards(results))


def mean_reward_every_trial(results: Iterable[Mapping[str, Any]]) -> float | None:
    """Mean reward over every rollout, an unscored one counting as 0.

    None only when there is no rollout. Leaving errored rollouts out of the
    mean let an agent that errors outscore one that fails cleanly; the Rust
    ``bench`` 0.8.2 counts every trial the same way for its pass rate and
    pass@k. Which rewards count as scored is :func:`mean_scored_reward`'s rule.
    """
    rows = list(results)
    return sum(_scored_rewards(rows)) / len(rows) if rows else None


def mean_reward_caveat(results: Iterable[Mapping[str, Any]]) -> str | None:
    """:func:`below_zero_caveat` over result rows (scored by
    :func:`mean_scored_reward`'s rule)."""
    rows = list(results)
    rewards = _scored_rewards(rows)
    return below_zero_caveat(rewards, unscored=len(rows) - len(rewards))


def below_zero_caveat(rewards: Iterable[float], *, unscored: int) -> str | None:
    """Why the every-trial mean reward can favour errors here, or None.

    A task may widen its ``reward_range`` below 0 (``[-1.0, 1.0]`` scores
    deliberate harm below doing nothing). The every-trial mean counts an
    unscored trial as 0, which is above such a reward, so there an agent
    that errors can score above one scored below 0. Only the mean is
    affected: an unscored trial never passes, so the solve rate and pass@k
    still never favour errors. None when no trial is unscored or no scored
    reward is below 0.
    """
    below = [r for r in rewards if r < 0]
    if not unscored or not below:
        return None
    return (
        f"{len(below)} scored reward(s) are below 0 (lowest {min(below):g}); "
        f"the mean reward counts each of the {unscored} unscored trial(s) as 0, "
        "above them, so read the scored-only mean reward too."
    )


def _scored_rewards(results: Iterable[Mapping[str, Any]]) -> list[float]:
    return [rw for r in results if (rw := finite_reward(extract_reward(r))) is not None]


def pass_rate(*, passed: int, total: int) -> float:
    """Pass rate over all tasks."""
    return passed / total if total > 0 else 0.0


def pass_rate_excl_errors(*, passed: int, failed: int, refused: int = 0) -> float:
    """Pass rate excluding errored tasks.

     *refused* trials, the plugin guard's refusals, stay in the denominator
     as failures: an error the agent caused must not raise the rate
    .
    """
    completed = passed + failed + refused
    return passed / completed if completed > 0 else 0.0


def score_summary_fields(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Scoring fields shared by initial evaluation and scoring-only resume.

    Callers retain run configuration, solver usage, timing, and provenance; only
    fields derived from the current terminal scoring evidence are rebuilt.
    """
    rows = list(results)
    counts = count_audit_outcomes(rows)
    error_categories: Counter[str] = Counter()
    verifier_categories: Counter[str] = Counter()
    integration_causes: Counter[str] = Counter()
    for row in rows:
        category = row.get("error_category") or classify_error(row.get("error"))
        if category:
            error_categories[category] += 1
        if category == AGENT_INTEGRATION:
            info = row.get("integration_failure_info")
            cause = info.get("cause") if isinstance(info, Mapping) else None
            integration_causes[cause if isinstance(cause, str) else "unknown"] += 1
        verifier_category = row.get(
            "verifier_error_category"
        ) or classify_verifier_error(row.get("verifier_error"))
        if verifier_category:
            verifier_categories[verifier_category] += 1
    refused = count_guard_refusals(rows)
    ratio = pass_rate(passed=counts["passed"], total=len(rows))
    scored_ratio = pass_rate_excl_errors(
        passed=counts["passed"], failed=counts["failed"], refused=refused
    )
    return {
        "schema_version": 2,
        "total": len(rows),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errored": counts["errored"],
        "verifier_errored": counts["verifier_errored"],
        "unscored": counts["unscored"],
        # Unscored trials the plugin guard refused: failures in the
        # error-excluding score and the mean reward.
        "guard_refusals": refused,
        "pass": counts["passed"],
        "fail": counts["failed"],
        "error": counts["errored"],
        "idle_timeout": error_categories.get(IDLE_TIMEOUT, 0),
        "error_categories": dict(error_categories) or None,
        "verifier_error_categories": dict(verifier_categories) or None,
        # Trials whose agent integration broke (unscored, never 0), by cause.
        "integration_failures": {
            "total": sum(integration_causes.values()),
            "by_cause": dict(sorted(integration_causes.items())),
        },
        "score": f"{ratio:.1%}",
        "score_ratio": ratio,
        "score_excl_errors": f"{scored_ratio:.1%}",
        "score_excl_errors_ratio": scored_ratio,
        # Every trial counted (an unscored one as 0), and the mean over scored
        # trials only (BenchFlow 0.7's mean_reward), each with its n and the
        # rule named in the file itself (DF-p4-upgrader-1).
        "mean_reward": mean_reward_every_trial(rows),
        "mean_reward_rule": MEAN_REWARD_RULE,
        "mean_reward_n": len(rows),
        "mean_reward_scored_only": mean_scored_reward(rows),
        "mean_reward_scored_only_n": len(_scored_rewards(rows)),
        # Set when a scored reward is below 0 and some trial is unscored: the
        # every-trial mean then counts an error above those rewards.
        "mean_reward_caveat": mean_reward_caveat(rows),
    }
