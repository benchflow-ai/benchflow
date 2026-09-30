"""``bf.rollout_group``: N rollouts of one task against a trainer's policy.

A trainer asks for a group of rollouts of a task (GRPO's group, typically 8)
and gets typed results back as each finishes::

    import benchflow as bf

    policy = bf.Policy("vllm/my-policy", base_url="http://127.0.0.1:8000/v1",
                       api_key_env="POLICY_KEY", version=0)
    config = bf.RolloutConfig(task_path="tasks/sql-000004", agent="opencode",
                              environment="docker", jobs_dir="jobs/rl")
    async with policy:
        group = bf.rollout_group(config, n=8, policy=policy)
        async for rollout in group:          # bf.TrainingRollout, as each finishes
            print(rollout.reward, rollout.attribution, len(rollout.segments))
        result = await group.wait()          # advantages, zero-variance flag
        policy.version += 1                  # after the trainer's weight update

Each member runs as an ordinary BenchFlow rollout (``bf.arun``) with the
policy reached through the relay (:mod:`benchflow.training.relay`), so its
folder under ``jobs_dir`` holds the usual result, trajectory and verifier
output, plus ``trajectory/policy_relay.jsonl``.

What a result carries, and the rules behind it:

- **Reward.** ``reward`` is the training reward: the verifier's reward
  (partial credit when the task gives it), or with ``reward="tests"`` the
  share of the verifier's tests that passed (its CTRF report), or 0.0 for a
  rollout an integrity audit caught exploiting the grader (``flagged``).
  ``tests`` lists every test the verifier reported.
- **Attribution.** ``attribution`` is ``"score"`` (the reward counts, even
  when it is 0) or ``"mask"`` (leave the rollout out: ``reward`` is None).
  A failure is masked only when the policy cannot have caused it: the
  sandbox, agent install or agent startup failed, the model endpoint failed,
  or the verifier failed on its own infrastructure or on a sandbox the
  policy never touched. Every other failure scores: an agent that ran out of
  time is a *scored timeout* (the verifier still ran), and a crash or lost
  sandbox after the policy acted scores 0, because the policy can kill its
  own sandbox. ``attribution_reason`` names the case.
- **Retries.** A masked rollout is run again, up to ``attempts`` times in
  all; earlier attempts are listed in ``attempts``. Scored rollouts are
  never retried (that would bias the group toward success). After the last
  attempt, ``on_failure`` decides: ``"mask"`` (default), ``"zero"`` (score
  0.0) or ``"raise"`` (the group raises :class:`RolloutGroupError`).
- **Tokens.** ``segments`` are exact token segments of the policy's own
  calls (:mod:`benchflow.trajectories.token_segments`): prompt ids,
  completion ids, an action mask (1 = sampled by the policy, 0 = added by
  the environment), logprobs, the policy version of each call and, when the
  server returned it, MoE routing. Helper calls are in
  ``excluded_segments``; ``tokens`` reports every call left out and why,
  failed attempts, and the attestation that the server's answers (seen at
  the relay), the gateway's store and these segments hold identical tokens.
- **Versions.** ``policy_version`` is ``policy.version`` when the rollout
  started; each segment lists the version that served each call, for
  asynchronous or off-policy training.
- **Group.** When every member is final, ``advantage`` is set on each
  scored member (GRPO ``(r - mean) / (std + 1e-4)`` or ``"loo"``), over the
  group's scored members only; masked members get None and do not count.
  With ``drop_zero_variance=True`` a group whose scored rewards are all
  equal gets no advantages and ``GroupResult.dropped == "zero_variance"``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import inspect
import json
import logging
import math
import re
import secrets
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from benchflow._utils.scoring import (
    ACP_ERROR,
    AGENT_INTEGRATION,
    API_ERROR,
    IDLE_TIMEOUT,
    INFRA_ERROR,
    INSTALL_FAILED,
    PROVIDER_AUTH,
    PROVIDER_RATE_LIMIT,
    SANDBOX_SETUP,
    SUSPECTED_API_ERROR,
    TIMED_OUT,
    VERIFIER_DEP_INSTALL,
    VERIFIER_INFRA,
    VERIFIER_TIMEOUT,
    classify_error,
    classify_verifier_error,
    extract_reward,
)
from benchflow.sandbox.leases import SandboxLifetime
from benchflow.trajectories.token_segments import (
    DEFAULT_TRAINABLE_KINDS,
    read_llm_trajectory,
    segment_digest,
    segment_rollout,
)

logger = logging.getLogger(__name__)

Attribution = Literal["score", "mask"]
FailurePolicy = Literal["mask", "zero", "raise"]
Outcome = Literal["passed", "failed", "masked", "cancelled"]

# Reasons a rollout is masked: the policy cannot have caused them.
SANDBOX_START = "sandbox_start"
AGENT_SETUP = "agent_setup"
MODEL_ENDPOINT = "model_endpoint"
VERIFIER_INFRASTRUCTURE = "verifier_infra"
VERIFIER_CRASH_CLEAN_RUN = "verifier_crash_clean_run"
INFRASTRUCTURE = "infrastructure"
CANCELLED = "cancelled"
MASK_REASONS = frozenset(
    {
        SANDBOX_START,
        AGENT_SETUP,
        MODEL_ENDPOINT,
        VERIFIER_INFRASTRUCTURE,
        VERIFIER_CRASH_CLEAN_RUN,
        INFRASTRUCTURE,
        CANCELLED,
    }
)
# Reasons a rollout scores: "scored" is a verifier reward, the rest score 0.
SCORED = "scored"
TIMEOUT = "timeout"
VERIFIER_ERROR = "verifier_error"
RUN_ERROR = "run_error"
NO_REWARD = "no_reward"
INTEGRITY_VIOLATION = "integrity_violation"
FAILURE_POLICY_ZERO = "failure_policy_zero"

_MODEL_ENDPOINT_CATEGORIES = frozenset(
    {API_ERROR, SUSPECTED_API_ERROR, PROVIDER_AUTH, PROVIDER_RATE_LIMIT, "usage_limit"}
)
_AGENT_SETUP_CATEGORIES = frozenset({INSTALL_FAILED, AGENT_INTEGRATION})
_VERIFIER_INFRA_CATEGORIES = frozenset({VERIFIER_INFRA, VERIFIER_DEP_INSTALL})
_ENDPOINT_MARKERS = ("provider unavailable", "http 503", "litellm proxy failed to start")
_STARTUP_DIAGNOSES = {
    "acp_initialize_timeout": "acp_initialize",
    "acp_session_new_timeout": "acp_session_new",
    "pty_startup_timeout": "pty_startup",
}


class RolloutGroupError(RuntimeError):
    """A group member failed on infrastructure with ``on_failure="raise"``."""

    def __init__(self, message: str, rollout: TrainingRollout | None = None) -> None:
        super().__init__(message)
        self.rollout = rollout


# --- small typed pieces ---------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """One test the verifier reported (its CTRF report, ``verifier/ctrf.json``)."""

    name: str
    status: str  # passed | failed | skipped | pending | other
    duration_ms: float | None = None
    message: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def read_tests(rollout_dir: str | Path | None) -> tuple[TestResult, ...]:
    """The verifier's per-test results, from ``verifier/ctrf.json``; () without one."""
    if rollout_dir is None:
        return ()
    path = Path(rollout_dir) / "verifier" / "ctrf.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return ()
    results = data.get("results") if isinstance(data, dict) else None
    tests = results.get("tests") if isinstance(results, dict) else None
    out = []
    for test in tests if isinstance(tests, list) else []:
        if not isinstance(test, dict):
            continue
        duration = test.get("duration")
        message = test.get("message")
        out.append(
            TestResult(
                name=str(test.get("name") or "test"),
                status=str(test.get("status") or "other"),
                duration_ms=float(duration)
                if isinstance(duration, int | float) and not isinstance(duration, bool)
                else None,
                message=" ".join(message.split())[:500]
                if isinstance(message, str) and message.strip()
                else None,
            )
        )
    return tuple(out)


def tests_pass_fraction(tests: Iterable[TestResult]) -> float | None:
    """Passed over passed + failed tests (skipped ones do not count); None without any."""
    counted = [t for t in tests if t.status in {"passed", "failed"}]
    if not counted:
        return None
    return sum(1 for t in counted if t.passed) / len(counted)


@dataclass(frozen=True)
class TokenSegment:
    """One exact token segment of a rollout (see :mod:`benchflow.trajectories.token_segments`).

    ``prompt_ids`` then ``completion_ids`` is the sequence the model saw;
    ``action_mask`` and ``logprobs`` are aligned with ``completion_ids``
    (1 and the sampled logprob for tokens the policy sampled; 0 and 0.0 for
    tokens the environment added). ``policy_versions`` is aligned with
    ``calls``. ``start_reason`` says why a segment after the first began
    (``rerender``, ``compaction``, ``after_dropped_call``).
    """

    kind: str
    thread: int
    calls: tuple[int, ...]
    prompt_ids: list[int]
    completion_ids: list[int]
    action_mask: list[int]
    logprobs: list[float]
    policy_versions: tuple[Any, ...]
    call_spans: tuple[dict[str, int], ...]
    digest: str
    trainable: bool
    excluded: str | None = None
    start_reason: str | None = None
    routing: tuple[dict[str, Any], ...] | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TokenSegment:
        start = raw.get("start") or {}
        return cls(
            kind=str(raw["kind"]),
            thread=int(raw["thread"]),
            calls=tuple(raw["calls"]),
            prompt_ids=list(raw["prompt_ids"]),
            completion_ids=list(raw["completion_ids"]),
            action_mask=list(raw["action_mask"]),
            logprobs=list(raw["logprobs"]),
            policy_versions=tuple(raw.get("policy_versions") or ()),
            call_spans=tuple(raw.get("call_spans") or ()),
            digest=str(raw["digest"]),
            trainable=bool(raw.get("trainable")),
            excluded=raw.get("excluded"),
            start_reason=start.get("reason") if isinstance(start, Mapping) else None,
            routing=tuple(raw["routing"]) if raw.get("routing") else None,
        )

    @property
    def n_action_tokens(self) -> int:
        return sum(self.action_mask)

    def verify(self) -> bool:
        """Recompute the digest over these lists: True when they are the ones BenchFlow built."""
        return (
            len(self.completion_ids) == len(self.action_mask) == len(self.logprobs)
            and segment_digest(
                {
                    "prompt_ids": self.prompt_ids,
                    "completion_ids": self.completion_ids,
                    "action_mask": self.action_mask,
                    "logprobs": self.logprobs,
                }
            )
            == self.digest
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "thread": self.thread,
            "calls": list(self.calls),
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "action_mask": self.action_mask,
            "logprobs": self.logprobs,
            "policy_versions": list(self.policy_versions),
            "call_spans": [dict(s) for s in self.call_spans],
            "digest": self.digest,
            "trainable": self.trainable,
            "excluded": self.excluded,
            "start_reason": self.start_reason,
            "routing": list(self.routing) if self.routing else None,
        }


@dataclass(frozen=True)
class FailedAttempt:
    """An earlier attempt of a group member that a retry replaced."""

    attempt: int
    rollout: str
    rollout_dir: Path | None
    error: str | None
    error_category: str | None
    attribution_reason: str


@dataclass(frozen=True)
class StartupTimeouts:
    """Startup timeouts for a group's rollouts (None keeps BenchFlow's default).

    ``sandbox_sec`` bounds the sandbox's start (``RolloutConfig``'s
    ``sandbox_setup_timeout``, which also bounds the agent's install);
    ``acp_handshake_sec`` the agent's ACP ``initialize`` and ``session/new``
    (default 60 s, ``BENCHFLOW_ACP_HANDSHAKE_TIMEOUT``); ``gateway_sec`` the
    model gateway's start (default 180 s,
    ``BENCHFLOW_LITELLM_HEALTH_TIMEOUT_SEC``). A result's ``startup`` block
    reports the values used and which phase, if any, ran out of time.
    """

    sandbox_sec: int | None = None
    acp_handshake_sec: float | None = None
    gateway_sec: float | None = None


# --- attribution ------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def startup_failure(result: Mapping[str, Any]) -> str | None:
    """The startup phase that failed or ran out of time, or None.

    ``sandbox``, ``agent_install``, ``gateway``, ``acp_initialize``,
    ``acp_session_new`` or ``pty_startup``, from the result's diagnostics
    and error.
    """
    if isinstance(result.get("sandbox_startup_info"), Mapping):
        return "sandbox"
    transport = result.get("transport_error_info")
    if isinstance(transport, Mapping):
        phase = _STARTUP_DIAGNOSES.get(str(transport.get("transport_diagnosis")))
        if phase is not None:
            return phase
    error = str(result.get("error") or "").lower()
    category = result.get("error_category") or classify_error(result.get("error"))
    if category == SANDBOX_SETUP:
        return "sandbox"
    if category == INSTALL_FAILED:
        return "agent_install"
    if "litellm proxy failed to start" in error or "litellm proxy is mandatory" in error:
        return "gateway"
    if "before the first prompt" in error:
        return "acp_initialize" if "initialize" in error else "acp_session_new"
    return None


def attribute(
    result: Mapping[str, Any],
    *,
    policy_acted: bool,
    endpoint_failed: bool = False,
) -> tuple[Attribution, str, str | None]:
    """``(attribution, reason, failure)`` for one finished rollout's ``result.json``.

    ``policy_acted``: the policy answered at least one model call (or the
    agent ran a tool). ``endpoint_failed``: the model endpoint's last answer
    to this rollout was a failure (the relay saw it). ``failure`` is
    ``"infrastructure"``, ``"policy"`` or None (no failure).
    """
    reward = _finite(extract_reward(result))
    error = result.get("error")
    verifier_error = result.get("verifier_error")
    category = result.get("error_category") or classify_error(error)
    verifier_category = result.get("verifier_error_category") or classify_verifier_error(
        verifier_error
    )
    if reward is not None:
        if endpoint_failed and not _passed(result):
            # The model endpoint failed on the rollout's last call: the agent
            # stopped for a reason outside the policy, and the verifier scored
            # what it left. A pass stands; anything less is not the policy's.
            return "mask", MODEL_ENDPOINT, "infrastructure"
        failure = "policy" if category in {TIMED_OUT, IDLE_TIMEOUT} else None
        return "score", SCORED, failure
    text = str(error or "").lower()
    if category == SANDBOX_SETUP:
        return "mask", SANDBOX_START, "infrastructure"
    if (
        category in _MODEL_ENDPOINT_CATEGORIES
        or (category == INFRA_ERROR and any(m in text for m in _ENDPOINT_MARKERS))
        or endpoint_failed
    ):
        return "mask", MODEL_ENDPOINT, "infrastructure"
    if category in _AGENT_SETUP_CATEGORIES or startup_failure(result) is not None:
        return "mask", AGENT_SETUP, "infrastructure"
    if not policy_acted:
        if verifier_error:
            return "mask", VERIFIER_CRASH_CLEAN_RUN, "infrastructure"
        if category == ACP_ERROR or error:
            return "mask", INFRASTRUCTURE, "infrastructure"
    if verifier_category in _VERIFIER_INFRA_CATEGORIES:
        return "mask", VERIFIER_INFRASTRUCTURE, "infrastructure"
    if category in {TIMED_OUT, IDLE_TIMEOUT} or verifier_category == VERIFIER_TIMEOUT:
        return "score", TIMEOUT, "policy"
    if verifier_error:
        return "score", VERIFIER_ERROR, "policy"
    if error:
        return "score", RUN_ERROR, "policy"
    return "score", NO_REWARD, "policy"


# --- one rollout, as a trainer reads it --------------------------------------------


@dataclass
class TrainingRollout:
    """One finished member of a :class:`RolloutGroup` (see the module docstring)."""

    group_id: str
    index: int
    attempt: int
    task: str
    rollout: str
    rollout_dir: Path | None
    reward: float | None
    verifier_reward: float | None
    rewards: dict[str, Any] | None
    reward_source: str
    passed: bool
    outcome: Outcome
    attribution: Attribution
    attribution_reason: str
    failure: str | None
    error: str | None
    error_category: str | None
    verifier_error: str | None
    verifier_error_category: str | None
    policy_version: Any
    policy_versions: tuple[Any, ...]
    tests: tuple[TestResult, ...]
    segments: tuple[TokenSegment, ...]
    excluded_segments: tuple[TokenSegment, ...]
    tokens: dict[str, Any]
    startup: dict[str, Any]
    started_at: datetime | None
    finished_at: datetime | None
    attempts: tuple[FailedAttempt, ...] = ()
    flagged: bool = False
    integrity: dict[str, Any] | None = None
    advantage: float | None = None

    @property
    def masked(self) -> bool:
        return self.attribution == "mask"

    @property
    def scored(self) -> bool:
        return self.reward is not None

    @property
    def duration_sec(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def n_action_tokens(self) -> int:
        return sum(s.n_action_tokens for s in self.segments)

    def to_dict(self, *, include_tokens: bool = False) -> dict[str, Any]:
        """A JSON-safe record; token lists only with ``include_tokens``."""
        record: dict[str, Any] = {
            "group_id": self.group_id,
            "index": self.index,
            "attempt": self.attempt,
            "task": self.task,
            "rollout": self.rollout,
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            "reward": self.reward,
            "verifier_reward": self.verifier_reward,
            "rewards": self.rewards,
            "reward_source": self.reward_source,
            "passed": self.passed,
            "outcome": self.outcome,
            "attribution": self.attribution,
            "attribution_reason": self.attribution_reason,
            "failure": self.failure,
            "error": self.error,
            "error_category": self.error_category,
            "verifier_error": self.verifier_error,
            "verifier_error_category": self.verifier_error_category,
            "policy_version": _jsonable(self.policy_version),
            "policy_versions": [_jsonable(v) for v in self.policy_versions],
            "tests": [
                {
                    "name": t.name,
                    "status": t.status,
                    "duration_ms": t.duration_ms,
                    "message": t.message,
                }
                for t in self.tests
            ],
            "segments": len(self.segments),
            "action_tokens": self.n_action_tokens,
            "tokens": self.tokens,
            "startup": self.startup,
            "attempts": [
                {
                    "attempt": a.attempt,
                    "rollout": a.rollout,
                    "error": a.error,
                    "error_category": a.error_category,
                    "attribution_reason": a.attribution_reason,
                }
                for a in self.attempts
            ],
            "flagged": self.flagged,
            "integrity": self.integrity,
            "advantage": self.advantage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_sec": self.duration_sec,
        }
        if include_tokens:
            record["segments"] = [s.to_dict() for s in self.segments]
            record["excluded_segments"] = [s.to_dict() for s in self.excluded_segments]
        return record


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


@dataclass
class GroupResult:
    """A finished group: members in order, with advantages."""

    group_id: str
    task: str
    rollouts: list[TrainingRollout]
    advantage: str | None
    mean_reward: float | None
    std_reward: float | None
    zero_variance: bool
    dropped: str | None

    @property
    def scored(self) -> int:
        return sum(1 for r in self.rollouts if r.scored)

    @property
    def masked(self) -> int:
        return sum(1 for r in self.rollouts if r.outcome == "masked")

    @property
    def cancelled(self) -> int:
        return sum(1 for r in self.rollouts if r.outcome == "cancelled")

    @property
    def flagged(self) -> int:
        return sum(1 for r in self.rollouts if r.flagged)

    def trainable(self) -> list[TrainingRollout]:
        """Members a trainer should use: scored, with an advantage, group not dropped."""
        if self.dropped is not None:
            return []
        return [r for r in self.rollouts if r.scored and r.advantage is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "task": self.task,
            "n": len(self.rollouts),
            "scored": self.scored,
            "masked": self.masked,
            "cancelled": self.cancelled,
            "flagged": self.flagged,
            "advantage": self.advantage,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "zero_variance": self.zero_variance,
            "dropped": self.dropped,
            "rollouts": [r.to_dict() for r in self.rollouts],
        }


# --- the group ------------------------------------------------------------------------

_DEFAULT_JOB: str | None = None
_GROUP_COUNTER = 0


def _default_job_name() -> str:
    global _DEFAULT_JOB
    if _DEFAULT_JOB is None:
        _DEFAULT_JOB = f"rl-{datetime.now():%Y-%m-%d__%H-%M-%S}"
    return _DEFAULT_JOB


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-") or "x"


def _new_group_id() -> str:
    global _GROUP_COUNTER
    _GROUP_COUNTER += 1
    return f"g{_GROUP_COUNTER:04d}-{secrets.token_hex(2)}"


def _read_result(rollout_dir: Path | None) -> dict[str, Any]:
    if rollout_dir is None:
        return {}
    try:
        data = json.loads((rollout_dir / "result.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_integrity(rollout_dir: Path | None) -> dict[str, Any] | None:
    """BenchShield's verdict for the rollout (``integrity/claim_verdict.json``), if any."""
    if rollout_dir is None:
        return None
    try:
        claim = json.loads((rollout_dir / "integrity" / "claim_verdict.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(claim, dict):
        return None
    exploited = claim.get("exploited")
    if not isinstance(exploited, bool):
        core = claim.get("core") if isinstance(claim.get("core"), dict) else {}
        verdict = claim.get("core_verdict") or core.get("core_verdict")
        exploited = verdict == "AgentViolation"
    return {"exploited": exploited, "reason": str(claim.get("reason") or "")}


def _verdict_dict(verdict: Any) -> dict[str, Any] | None:
    if verdict is None:
        return None
    if isinstance(verdict, Mapping):
        exploited, reason = verdict.get("exploited"), verdict.get("reason")
    else:
        exploited, reason = getattr(verdict, "exploited", None), getattr(verdict, "reason", None)
    if not isinstance(exploited, bool):
        raise ValueError("an integrity verdict needs a boolean 'exploited'")
    return {"exploited": exploited, "reason": str(reason or "")}


class RolloutGroup:
    """N rollouts of one task; iterate for results as they finish.

    Made by :func:`rollout_group`. ``async for rollout in group`` yields each
    member once, when final (after its retries). ``await group.wait()``
    returns the :class:`GroupResult` with advantages. ``await group.cancel()``
    stops the members still running (their sandboxes are deleted); it is also
    what leaving ``async with group:`` early does.
    """

    def __init__(
        self,
        config: Any,
        *,
        n: int,
        policy: Any | None,
        attempts: int,
        on_failure: FailurePolicy,
        concurrency: int | None,
        group_id: str | None,
        advantage: Literal["grpo", "loo"] | None,
        drop_zero_variance: bool,
        reward: Literal["verifier", "tests"],
        trainable_kinds: Iterable[str],
        integrity: Literal["auto", "off"] | Callable[[TrainingRollout], Any],
        startup_timeouts: StartupTimeouts | None,
        sandbox_lifetime: SandboxLifetime | None,
        run: Callable[[Any], Any] | None = None,
    ) -> None:
        from benchflow.rollout import RolloutConfig

        if not isinstance(config, RolloutConfig):
            raise TypeError("rollout_group needs a bf.RolloutConfig")
        if n < 1:
            raise ValueError("n must be at least 1")
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        if on_failure not in {"mask", "zero", "raise"}:
            raise ValueError("on_failure must be 'mask', 'zero' or 'raise'")
        if advantage not in {"grpo", "loo", None}:
            raise ValueError("advantage must be 'grpo', 'loo' or None")
        if reward not in {"verifier", "tests"}:
            raise ValueError("reward must be 'verifier' or 'tests'")
        if concurrency is not None and concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if not (integrity in {"auto", "off"} or callable(integrity)):
            raise ValueError("integrity must be 'auto', 'off' or a callable")
        if config.scenes and policy is not None and any(
            role.model != policy.model for scene in config.scenes for role in scene.roles
        ):
            raise ValueError(
                "rollout_group runs one agent against the policy: leave "
                "RolloutConfig.scenes empty and set agent=, or give every role "
                f"model={policy.model!r}"
            )
        if policy is not None and config.primary_model not in (None, policy.model):
            raise ValueError(
                f"the config's model {config.primary_model!r} is not the policy's "
                f"{policy.model!r}; leave RolloutConfig.model unset"
            )
        if policy is None and config.primary_agent not in {"oracle", "nop"}:
            raise ValueError(
                "rollout_group needs a policy= unless the agent is 'oracle' or 'nop'"
            )
        self.config = config
        self.n = n
        self.policy = policy
        self.attempts = attempts
        self.on_failure = on_failure
        self.concurrency = concurrency or n
        self.group_id = _safe(group_id) if group_id else _new_group_id()
        self.advantage = advantage
        self.drop_zero_variance = drop_zero_variance
        self.reward_mode = reward
        self.trainable_kinds = tuple(trainable_kinds)
        self.integrity = integrity
        self.startup_timeouts = startup_timeouts or StartupTimeouts()
        self.sandbox_lifetime = sandbox_lifetime or SandboxLifetime(
            auto_stop_min=60, auto_delete_min=0
        )
        self.task = Path(config.task_path).name
        self.job_name = config.job_name or _default_job_name()
        self._run = run
        self._members: list[TrainingRollout | None] = [None] * n
        self._finished: asyncio.Queue[int] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._gate = asyncio.Semaphore(self.concurrency)
        self._started = False
        self._error: BaseException | None = None
        self._uninstall: Callable[[], None] | None = None
        self._result: GroupResult | None = None

    # --- lifecycle ----------------------------------------------------------------

    async def start(self) -> RolloutGroup:
        """Start every member (iteration and :meth:`wait` call this)."""
        if self._started:
            return self
        self._started = True
        from benchflow.providers.litellm_runtime import _SANDBOX_LOCAL_ENVIRONMENTS
        from benchflow.runtime import _HOST_CHECKED, check_host, check_rollout_config
        from benchflow.sandbox.leases import install_signal_cleanup, reap_dead_leases

        check_rollout_config(self.config)
        if self.policy is not None:
            await self.policy.start()
            environment = self.config.environment
            if (
                environment in _SANDBOX_LOCAL_ENVIRONMENTS
                and await self.policy.relay.url_for(environment) is None
            ):
                raise ValueError(
                    f"environment={environment!r} runs the model gateway inside each "
                    "sandbox, which must reach the policy through the relay: give "
                    "bf.Policy(relay_bind='0.0.0.0:<port>', relay_public_url='https://...') "
                    "an address those sandboxes can reach (ideally behind TLS)"
                )
        await asyncio.to_thread(reap_dead_leases)
        if self._run is None:
            await asyncio.to_thread(check_host, [self.config])
        self._uninstall = install_signal_cleanup()
        context = contextvars.copy_context()
        context.run(_HOST_CHECKED.set, True)
        self._tasks = [
            asyncio.get_running_loop().create_task(self._member(i), context=context)
            for i in range(self.n)
        ]
        return self

    async def __aenter__(self) -> RolloutGroup:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        if any(not t.done() for t in self._tasks):
            await self.cancel()
        self._release()

    def _release(self) -> None:
        if self._uninstall is not None and all(t.done() for t in self._tasks):
            self._uninstall()
            self._uninstall = None

    async def cancel(self, *, grace_sec: float = 180.0) -> None:
        """Stop the members still running; their sandboxes are torn down."""
        running = [t for t in self._tasks if not t.done()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.wait(running, timeout=grace_sec)
        for index, member in enumerate(self._members):
            if member is None:
                self._members[index] = self._cancelled(index)
        self._release()

    # --- results ----------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[TrainingRollout]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[TrainingRollout]:
        await self.start()
        seen: set[int] = set()
        try:
            while len(seen) < self.n:
                if self._error is not None:
                    raise self._error
                if all(t.done() for t in self._tasks) and self._finished.empty():
                    break
                index = await self._finished.get()
                if index < 0:
                    continue
                if self._error is not None:
                    raise self._error
                member = self._members[index]
                if index in seen or member is None:
                    continue
                seen.add(index)
                if member.outcome == "cancelled":
                    continue
                yield member
            if self._error is not None:
                raise self._error
            self._complete()
        finally:
            if self._error is not None:
                await self.cancel()
            self._release()

    async def wait(self) -> GroupResult:
        """Wait for every member; the group with advantages."""
        await self.start()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._release()
        if self._error is not None:
            raise self._error
        return self._complete()

    @property
    def results(self) -> list[TrainingRollout]:
        """Members finished so far, in member order."""
        return [m for m in self._members if m is not None]

    def _complete(self) -> GroupResult:
        if self._result is not None:
            return self._result
        from benchflow.trajectories.training_signal import _advantages

        members = [m if m is not None else self._cancelled(i) for i, m in enumerate(self._members)]
        rewards = [m.reward for m in members]
        scored = [r for r in rewards if r is not None]
        mean = sum(scored) / len(scored) if scored else None
        std = (
            math.sqrt(sum((r - mean) ** 2 for r in scored) / (len(scored) - 1))
            if mean is not None and len(scored) >= 2
            else None
        )
        zero_variance = len(scored) >= 1 and max(scored) == min(scored)
        dropped = None
        if not scored:
            dropped = "no_scored_rollouts"
        elif self.drop_zero_variance and zero_variance:
            dropped = "zero_variance"
        if self.advantage is not None and dropped is None:
            for member, value in zip(members, _advantages(rewards, self.advantage), strict=True):
                member.advantage = value
        result = GroupResult(
            group_id=self.group_id,
            task=self.task,
            rollouts=members,
            advantage=self.advantage,
            mean_reward=mean,
            std_reward=std,
            zero_variance=zero_variance,
            dropped=dropped,
        )
        self._result = result
        self._write_group_record(result)
        return result

    def _write_group_record(self, result: GroupResult) -> None:
        path = Path(self.config.jobs_dir) / self.job_name / "groups" / f"{self.group_id}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        except OSError as exc:
            logger.warning("Could not write the group record %s: %s", path, exc)

    def _cancelled(self, index: int) -> TrainingRollout:
        now = datetime.now()
        return TrainingRollout(
            group_id=self.group_id,
            index=index,
            attempt=0,
            task=self.task,
            rollout="",
            rollout_dir=None,
            reward=None,
            verifier_reward=None,
            rewards=None,
            reward_source="none",
            passed=False,
            outcome="cancelled",
            attribution="mask",
            attribution_reason=CANCELLED,
            failure=None,
            error="cancelled",
            error_category=None,
            verifier_error=None,
            verifier_error_category=None,
            policy_version=getattr(self.policy, "version", None),
            policy_versions=(),
            tests=(),
            segments=(),
            excluded_segments=(),
            tokens={},
            startup={},
            started_at=now,
            finished_at=now,
        )

    # --- one member -------------------------------------------------------------------------

    async def _member(self, index: int) -> None:
        failures: list[FailedAttempt] = []
        try:
            for attempt in range(1, self.attempts + 1):
                async with contextlib.AsyncExitStack() as stack:
                    gate = getattr(self.policy, "_gate", None)
                    if gate is not None:
                        await stack.enter_async_context(gate)
                    await stack.enter_async_context(self._gate)
                    rollout = await self._attempt(index, attempt)
                if rollout.masked and attempt < self.attempts:
                    failures.append(
                        FailedAttempt(
                            attempt=attempt,
                            rollout=rollout.rollout,
                            rollout_dir=rollout.rollout_dir,
                            error=rollout.error,
                            error_category=rollout.error_category,
                            attribution_reason=rollout.attribution_reason,
                        )
                    )
                    await asyncio.sleep(min(2.0 * 2 ** (attempt - 1), 30.0))
                    continue
                rollout.attempts = tuple(failures)
                self._members[index] = self._apply_failure_policy(rollout)
                break
        except asyncio.CancelledError:
            if self._members[index] is None:
                self._members[index] = self._cancelled(index)
            raise
        except RolloutGroupError as exc:
            self._error = exc
        except Exception as exc:  # a bug here must surface, not hang the group
            logger.exception("rollout group member %d failed", index)
            self._error = exc
        finally:
            self._finished.put_nowait(index if self._members[index] is not None else -1)

    def _apply_failure_policy(self, rollout: TrainingRollout) -> TrainingRollout:
        if not rollout.masked:
            return rollout
        if self.on_failure == "raise":
            raise RolloutGroupError(
                f"{rollout.rollout}: {rollout.attribution_reason} after "
                f"{rollout.attempt} attempt(s): {rollout.error or rollout.verifier_error}",
                rollout,
            )
        if self.on_failure == "zero":
            rollout.reward = 0.0
            rollout.reward_source = "failure_policy"
            rollout.attribution = "score"
            rollout.attribution_reason = FAILURE_POLICY_ZERO
            rollout.outcome = "failed"
        return rollout

    def _attempt_config(self, index: int, attempt: int, agent_env: dict[str, str]) -> Any:
        config = copy.copy(self.config)
        config.agent_env = {**(self.config.agent_env or {}), **agent_env}
        config.job_name = self.job_name
        config.rollout_name = f"{_safe(self.task)}__{self.group_id}-r{index:02d}-a{attempt}"
        if self.policy is not None:
            config.model = self.policy.model
        if self.startup_timeouts.sandbox_sec is not None:
            config.sandbox_setup_timeout = int(self.startup_timeouts.sandbox_sec)
        return config

    async def _attempt(self, index: int, attempt: int) -> TrainingRollout:
        from benchflow._utils.startup_timeouts import startup_timeout_overrides
        from benchflow.sandbox.leases import sandbox_lifetime

        grant = None
        agent_env: dict[str, str] = {}
        version = getattr(self.policy, "version", None)
        if self.policy is not None:
            grant = self.policy.relay.grant(
                f"{self.task}/{self.group_id}/{index}/{attempt}"
            )
            agent_env = await self.policy.agent_env(grant.token, self.config.environment)
        config = self._attempt_config(index, attempt, agent_env)
        started = datetime.now()
        relay_calls: list[Any] = []
        try:
            with (
                startup_timeout_overrides(
                    acp_handshake_sec=self.startup_timeouts.acp_handshake_sec,
                    gateway_sec=self.startup_timeouts.gateway_sec,
                ),
                sandbox_lifetime(self.sandbox_lifetime),
            ):
                result = await self._run_rollout(config)
        finally:
            if grant is not None:
                relay_calls = self.policy.relay.revoke(grant)
        return await self._build(index, attempt, config, result, relay_calls, version, started)

    async def _run_rollout(self, config: Any) -> Any:
        from benchflow.models import RolloutResult

        run = self._run
        if run is None:
            from benchflow.runtime import run as arun

            run = arun
        try:
            return await run(config)
        except Exception as exc:
            logger.exception("rollout %s raised", config.rollout_name)
            return RolloutResult(
                task_name=self.task,
                rollout_name=config.rollout_name or "",
                agent=config.primary_agent or "",
                model=config.primary_model,
                error=f"unexpected exception: {type(exc).__name__}: {exc}",
                error_category="other",
            )

    async def _build(
        self,
        index: int,
        attempt: int,
        config: Any,
        result: Any,
        relay_calls: list[Any],
        version: Any,
        started: datetime,
    ) -> TrainingRollout:
        rollout_dir = getattr(result, "rollout_dir", None)
        rollout_dir = Path(rollout_dir) if rollout_dir else None
        raw = _read_result(rollout_dir)
        if not raw:
            raw = result.to_record() if hasattr(result, "to_record") else {}
            raw["rewards"] = getattr(result, "rewards", None)
        relay_dicts = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in relay_calls]
        if rollout_dir is not None and relay_dicts:
            with contextlib.suppress(OSError):
                path = rollout_dir / "trajectory" / "policy_relay.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "".join(json.dumps(c, default=str) + "\n" for c in relay_dicts)
                )
        ok_calls = [c for c in relay_dicts if c.get("status") == "ok"]
        endpoint_failed = bool(relay_dicts) and relay_dicts[-1].get("status") != "ok" and (
            relay_dicts[-1].get("http_status") is None
            or int(relay_dicts[-1].get("http_status") or 0) >= 500
            or relay_dicts[-1].get("http_status") == 429
        )
        tool_calls = int(getattr(result, "n_tool_calls", 0) or 0)
        policy_acted = bool(ok_calls) or tool_calls > 0
        attribution, reason, failure = attribute(
            raw, policy_acted=policy_acted, endpoint_failed=endpoint_failed
        )

        exchanges = await asyncio.to_thread(read_llm_trajectory, rollout_dir) if rollout_dir else None
        report = segment_rollout(
            exchanges or [],
            relay_calls=relay_dicts if self.policy is not None else None,
            trainable_kinds=self.trainable_kinds,
        )
        segments = [TokenSegment.from_dict(s) for s in report["segments"]]
        tokens = {k: v for k, v in report.items() if k != "segments"}
        tokens["captured"] = exchanges is not None
        tests = read_tests(rollout_dir)
        verifier_reward = _finite(extract_reward(raw)) if raw else None
        reward: float | None = None
        reward_source = "none"
        if attribution == "score":
            reward, reward_source = (
                (verifier_reward, "verifier") if verifier_reward is not None else (0.0, reason)
            )
            if self.reward_mode == "tests" and reason == SCORED:
                fraction = tests_pass_fraction(tests)
                if fraction is not None:
                    reward, reward_source = fraction, "tests"
        passed = bool(raw) and _passed(raw)
        outcome: Outcome
        if attribution == "mask":
            outcome = "masked"
        else:
            outcome = "passed" if passed else "failed"
        seen_versions: list[Any] = []
        for call in ok_calls:
            if call.get("version") not in seen_versions:
                seen_versions.append(call.get("version"))
        rollout = TrainingRollout(
            group_id=self.group_id,
            index=index,
            attempt=attempt,
            task=self.task,
            rollout=config.rollout_name or "",
            rollout_dir=rollout_dir,
            reward=reward,
            verifier_reward=verifier_reward,
            rewards=raw.get("rewards") if isinstance(raw.get("rewards"), dict) else None,
            reward_source=reward_source,
            passed=passed,
            outcome=outcome,
            attribution=attribution,
            attribution_reason=reason,
            failure=failure,
            error=raw.get("error") or getattr(result, "error", None),
            error_category=raw.get("error_category")
            or classify_error(raw.get("error") or getattr(result, "error", None)),
            verifier_error=raw.get("verifier_error"),
            verifier_error_category=raw.get("verifier_error_category")
            or classify_verifier_error(raw.get("verifier_error")),
            policy_version=version,
            policy_versions=tuple(seen_versions),
            tests=tests,
            segments=tuple(s for s in segments if s.trainable),
            excluded_segments=tuple(s for s in segments if not s.trainable),
            tokens=tokens,
            startup=self._startup_report(raw, config, rollout_dir),
            started_at=started,
            finished_at=datetime.now(),
        )
        await self._apply_integrity(rollout)
        return rollout

    def _startup_report(
        self, raw: Mapping[str, Any], config: Any, rollout_dir: Path | None
    ) -> dict[str, Any]:
        from benchflow.acp.runtime import _acp_handshake_timeout_sec
        from benchflow.agents.install import effective_install_timeout
        from benchflow.providers.litellm_runtime import _HEALTH_DEADLINE_SEC

        timeouts = self.startup_timeouts
        timing: dict[str, Any] = {}
        if rollout_dir is not None:
            with contextlib.suppress(OSError, ValueError):
                timing = json.loads((rollout_dir / "timing.json").read_text())
        return {
            "timeouts": {
                "sandbox_sec": config.sandbox_setup_timeout,
                "agent_install_sec": effective_install_timeout(
                    config.primary_agent, config.sandbox_setup_timeout
                ),
                "acp_handshake_sec": timeouts.acp_handshake_sec
                if timeouts.acp_handshake_sec is not None
                else _acp_handshake_timeout_sec(),
                "gateway_sec": timeouts.gateway_sec
                if timeouts.gateway_sec is not None
                else _HEALTH_DEADLINE_SEC,
            },
            "failed_phase": startup_failure(raw) if raw else None,
            "phases_sec": {
                k: v
                for k, v in timing.items()
                if k in {"environment_setup", "agent_setup"} and isinstance(v, int | float)
            },
        }

    async def _apply_integrity(self, rollout: TrainingRollout) -> None:
        if self.integrity == "off":
            return
        verdict: Any
        if self.integrity == "auto":
            verdict = _read_integrity(rollout.rollout_dir)
        else:
            assert callable(self.integrity)
            verdict = self.integrity(rollout)
            if inspect.isawaitable(verdict):
                verdict = await verdict
        verdict = _verdict_dict(verdict)
        rollout.integrity = verdict
        if verdict is None or not verdict["exploited"]:
            return
        # An exploit is the policy's doing: a flagged 0, never a mask.
        rollout.flagged = True
        rollout.reward = 0.0
        rollout.reward_source = "integrity"
        rollout.attribution = "score"
        rollout.attribution_reason = INTEGRITY_VIOLATION
        rollout.failure = "policy"
        rollout.passed = False
        rollout.outcome = "failed"


def _passed(raw: Mapping[str, Any]) -> bool:
    from benchflow._utils.scoring import classify_score_outcome

    try:
        return classify_score_outcome(raw) == "passed"
    except Exception:
        return False


def rollout_group(
    config: Any,
    *,
    n: int,
    policy: Any | None,
    attempts: int = 2,
    on_failure: FailurePolicy = "mask",
    concurrency: int | None = None,
    group_id: str | None = None,
    advantage: Literal["grpo", "loo"] | None = "grpo",
    drop_zero_variance: bool = False,
    reward: Literal["verifier", "tests"] = "verifier",
    trainable_kinds: Iterable[str] = DEFAULT_TRAINABLE_KINDS,
    integrity: Literal["auto", "off"] | Callable[[TrainingRollout], Any] = "auto",
    startup_timeouts: StartupTimeouts | None = None,
    sandbox_lifetime: SandboxLifetime | None = None,
) -> RolloutGroup:
    """N rollouts of ``config``'s task against ``policy``, as a :class:`RolloutGroup`.

    ``config`` is an ordinary :class:`~benchflow.RolloutConfig` (task, agent,
    sandbox, timeouts, jobs folder); leave its ``model`` unset, the policy
    supplies it. ``policy`` is a :class:`~benchflow.Policy` (None only for
    the ``oracle`` and ``nop`` controls).

    ``attempts`` tries per member, retrying only masked (infrastructure)
    failures; ``on_failure`` what a member masked on its last attempt
    becomes (``"mask"``, ``"zero"``, ``"raise"``). ``concurrency`` caps this
    group's members running at once (default ``n``; ``Policy.max_concurrency``
    caps all groups). ``advantage`` is ``"grpo"``, ``"loo"`` or None;
    ``drop_zero_variance`` drops a group whose scored rewards are all equal.
    ``reward="tests"`` trains on the share of verifier tests passed.
    ``trainable_kinds`` are the segment kinds returned in ``segments``
    (default agent, subagent, chat; helpers and compaction calls go to
    ``excluded_segments``). ``integrity`` is ``"auto"`` (apply a BenchShield
    verdict the rollout wrote), ``"off"``, or a callable returning a verdict
    with a boolean ``exploited`` and a ``reason``: an exploited rollout
    scores 0 and is ``flagged``. ``startup_timeouts`` and
    ``sandbox_lifetime`` (default: Daytona sandboxes stop after 60 idle
    minutes and are deleted when stopped) apply to every member.
    """
    return RolloutGroup(
        config,
        n=n,
        policy=policy,
        attempts=attempts,
        on_failure=on_failure,
        concurrency=concurrency,
        group_id=group_id,
        advantage=advantage,
        drop_zero_variance=drop_zero_variance,
        reward=reward,
        trainable_kinds=trainable_kinds,
        integrity=integrity,
        startup_timeouts=startup_timeouts,
        sandbox_lifetime=sandbox_lifetime,
    )


__all__ = [
    "FailedAttempt",
    "GroupResult",
    "MASK_REASONS",
    "RolloutGroup",
    "RolloutGroupError",
    "StartupTimeouts",
    "TestResult",
    "TokenSegment",
    "TrainingRollout",
    "attribute",
    "read_tests",
    "rollout_group",
    "startup_failure",
    "tests_pass_fraction",
]
