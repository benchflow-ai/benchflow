"""Harness-level loop strategies — a declarative evaluand axis for rollouts.

A loop strategy describes how the harness re-prompts the agent across rounds
(``verify-retry``: retry with policy-filtered soft-verifier feedback until the
soft reward clears ``pass_threshold`` or ``k`` retries are spent). Strategies
are materialized into :class:`~benchflow.contracts.BaseUser` instances and run
by the existing user-loop engine (:mod:`benchflow.rollout._user_loop`); only
the final hardened ``verify()`` scores the rollout — soft rewards are advisory.

The spec is parsed once (CLI ``--loop-strategy``), validated, and stamped into
``config.json``/``result.json``/``summary.json`` so results from different
strategies never mix silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from benchflow.contracts import BaseUser, RoundResult

__all__ = [
    "FeedbackLevel",
    "LoopStrategySpec",
    "LoopStrategyUser",
    "SINGLE_SHOT",
    "VERIFY_RETRY",
    "VerifyRetryUser",
    "build_loop_user",
    "collect_loop_metadata",
    "filter_verifier_feedback",
    "loop_block",
    "parse_loop_strategy_spec",
]

SINGLE_SHOT = "single-shot"
VERIFY_RETRY = "verify-retry"

DEFAULT_VERIFY_RETRY_K = 3
DEFAULT_PASS_THRESHOLD = 1.0
DEFAULT_MAX_FEEDBACK_CHARS = 4000


class FeedbackLevel(StrEnum):
    """How much verifier output a retry prompt may carry back to the agent.

    Mid-loop verifier output can leak ground truth (pytest assertion diffs
    print expected values), so the default is ``NAMES``; ``RAW`` is opt-in
    and truncated.
    """

    NONE = "none"
    COUNTS = "counts"
    NAMES = "names"
    RAW = "raw"


def _normalized_params(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize strategy params, filling strategy defaults."""
    if name == SINGLE_SHOT:
        if params:
            raise ValueError(
                f"loop strategy 'single-shot' takes no params, got {sorted(params)}"
            )
        return {}
    if name == VERIFY_RETRY:
        unknown = set(params) - {"k", "feedback"}
        if unknown:
            raise ValueError(f"unknown verify-retry param(s): {sorted(unknown)}")
        k = params.get("k", DEFAULT_VERIFY_RETRY_K)
        if isinstance(k, str):
            try:
                k = int(k)
            except ValueError:
                raise ValueError(
                    f"verify-retry k must be an integer, got {k!r}"
                ) from None
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"verify-retry k must be a positive integer, got {k!r}")
        feedback = params.get("feedback", FeedbackLevel.NAMES.value)
        if isinstance(feedback, FeedbackLevel):
            feedback = feedback.value
        try:
            FeedbackLevel(feedback)
        except ValueError:
            valid = ", ".join(level.value for level in FeedbackLevel)
            raise ValueError(
                f"verify-retry feedback must be one of {valid}, got {feedback!r}"
            ) from None
        return {"k": k, "feedback": feedback}
    raise ValueError(
        f"unknown loop strategy {name!r} (expected {SINGLE_SHOT!r} or {VERIFY_RETRY!r})"
    )


@dataclass(frozen=True)
class LoopStrategySpec:
    """Declarative loop-strategy identity, stamped into run artifacts."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _normalized_params(self.name, self.params))

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"strategy": self.name}
        if self.params:
            payload["params"] = dict(self.params)
        return payload

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> LoopStrategySpec:
        name = raw.get("strategy")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"loop strategy mapping requires a 'strategy' name, got {raw!r}"
            )
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(
                f"loop strategy 'params' must be a mapping, got {params!r}"
            )
        return cls(name=name, params=dict(params))


def loop_block(
    spec: LoopStrategySpec | str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``loop`` block stamped into config.json/result.json/summary.json.

    No strategy stamps ``{"strategy": "single-shot"}`` so downstream joins
    never miss keys. *metadata* (``collect_loop_metadata``) is merged into
    the block for result.json.
    """
    if not isinstance(spec, LoopStrategySpec):
        return {"strategy": SINGLE_SHOT}
    block = spec.to_mapping()
    if metadata:
        block.update(metadata)
    return block


def parse_loop_strategy_spec(value: str) -> LoopStrategySpec:
    """Parse a CLI spec like ``"verify-retry:k=3,feedback=names"``.

    Bare names (``"single-shot"``, ``"verify-retry"``) are accepted and take
    the strategy defaults.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty loop strategy spec")
    name, sep, params_text = text.partition(":")
    name = name.strip()
    params: dict[str, Any] = {}
    if sep:
        for item in params_text.split(","):
            key, eq, raw_val = item.partition("=")
            key, raw_val = key.strip(), raw_val.strip()
            if not key or not eq or not raw_val:
                raise ValueError(
                    f"malformed loop strategy param {item.strip()!r} in {value!r} "
                    "(expected key=value)"
                )
            if key in params:
                raise ValueError(f"duplicate loop strategy param {key!r} in {value!r}")
            params[key] = raw_val
    return LoopStrategySpec(name=name, params=params)


# ``FAILED tests/test_x.py::test_y - AssertionError: ...`` — the node id stops
# at the first whitespace, so assertion detail (which can embed expected
# ground-truth values) never survives the NAMES filter.
_FAILED_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)
_PARAM_ID_RE = re.compile(r"\[.*\]$")
_SUMMARY_LINE_RE = re.compile(
    r"^=+\s+(.+?(?:passed|failed|error|errors|skipped|xfailed|xpassed)[^=]*?)\s+=+\s*$",
    re.MULTILINE,
)


def _failing_node_ids(text: str) -> list[str]:
    """Failing pytest node ids, deduplicated, with two leak guards.

    Tokens must look like node ids — contain ``::`` or end with ``.py`` —
    so captured-log lines (``ERROR    root:app.py:7 the answer is 42``)
    that happen to match the FAILED/ERROR prefix never reach the agent.
    Parametrized id suffixes are stripped: ids like ``test_y[expected-42]``
    embed the expected values the NAMES level exists to withhold.
    """
    ids = [
        _PARAM_ID_RE.sub("", token)
        for token in _FAILED_LINE_RE.findall(text)
        if "::" in token or token.endswith(".py")
    ]
    return list(dict.fromkeys(ids))


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def filter_verifier_feedback(
    output: str | None,
    level: FeedbackLevel | str,
    *,
    max_chars: int = DEFAULT_MAX_FEEDBACK_CHARS,
) -> str:
    """Reduce raw verifier output to the policy-allowed feedback level.

    Pure function: NONE → empty; COUNTS → pytest summary counts only;
    NAMES → failing test node ids only (fails closed to empty when the output
    has no recognizable pytest result lines); RAW → full output truncated to
    ``max_chars``.
    """
    level = FeedbackLevel(level)
    text = (output or "").strip()
    if level is FeedbackLevel.NONE or not text:
        return ""
    if level is FeedbackLevel.RAW:
        return _truncate(text, max_chars)
    failures = _failing_node_ids(text)
    if level is FeedbackLevel.COUNTS:
        summaries = _SUMMARY_LINE_RE.findall(text)
        if summaries:
            return _truncate(summaries[-1].strip(), max_chars)
        return f"{len(failures)} failing test(s)" if failures else ""
    return _truncate("\n".join(failures), max_chars)


def _soft_reward(rewards: dict[str, Any] | None) -> float | None:
    """The canonical soft-verify reward reader for loop strategies."""
    if not isinstance(rewards, dict):
        return None
    value = rewards.get("reward")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class LoopStrategyUser(BaseUser):
    """Base class for users materialized from a :class:`LoopStrategySpec`.

    The user-loop engine and :func:`collect_loop_metadata` read
    ``pass_threshold`` and ``feedback_level`` as typed attributes — every
    strategy user carries them, no duck typing.
    """

    def __init__(
        self,
        *,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        feedback: FeedbackLevel | str = FeedbackLevel.NAMES,
    ) -> None:
        self.pass_threshold = pass_threshold
        self.feedback_level = FeedbackLevel(feedback)


class VerifyRetryUser(LoopStrategyUser):
    """Retry the instruction until the soft verifier passes the bar.

    Round 0 sends the task instruction unchanged. Each later round inspects
    the previous round's soft-verify reward: at or above ``pass_threshold``
    the loop stops; otherwise the agent gets a retry prompt carrying only the
    policy-filtered verifier feedback. The retry budget is owned by the
    engine: :func:`build_loop_user` caps the loop at ``max_user_rounds =
    k + 1``, and the engine derives ``stop_reason`` from the round log.
    """

    def __init__(
        self,
        *,
        k: int = DEFAULT_VERIFY_RETRY_K,
        feedback: FeedbackLevel | str = FeedbackLevel.NAMES,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        max_feedback_chars: int = DEFAULT_MAX_FEEDBACK_CHARS,
    ) -> None:
        if k < 1:
            raise ValueError(f"verify-retry k must be >= 1, got {k}")
        super().__init__(pass_threshold=pass_threshold, feedback=feedback)
        self.k = k
        self.max_feedback_chars = max_feedback_chars

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        if round == 0:
            return instruction
        reward = _soft_reward(round_result.rewards if round_result else None)
        if reward is not None and reward >= self.pass_threshold:
            return None
        return self._retry_prompt(round_result)

    def _retry_prompt(self, round_result: RoundResult | None) -> str:
        feedback = filter_verifier_feedback(
            round_result.verifier_output if round_result else None,
            self.feedback_level,
            max_chars=self.max_feedback_chars,
        )
        lines = ["Your previous attempt did not pass verification."]
        if feedback:
            lines.append(f"Verifier feedback:\n{feedback}")
        lines.append(
            "Review your work in the workspace, fix the remaining issues, "
            "and try again."
        )
        return "\n\n".join(lines)


def build_loop_user(spec: LoopStrategySpec) -> tuple[LoopStrategyUser, int] | None:
    """Materialize the runtime user for *spec*.

    Returns ``(user, max_user_rounds)``, or ``None`` for ``single-shot`` —
    today's plain one-prompt path with no user loop.
    """
    if spec.name == SINGLE_SHOT:
        return None
    if spec.name == VERIFY_RETRY:
        k = int(spec.params["k"])
        user = VerifyRetryUser(k=k, feedback=spec.params["feedback"])
        return user, k + 1
    raise ValueError(f"unknown loop strategy {spec.name!r}")


def collect_loop_metadata(
    user: LoopStrategyUser,
    rounds_log: list[dict],
    *,
    max_rounds: int,
    error: str | None,
) -> dict[str, Any]:
    """Summarize a loop-strategy run for the result.json ``loop`` block.

    Derived purely from the engine's per-round log and the rollout's final
    error state, so it is correct on every path — including rollouts whose
    loop died mid-round (timeout, ACP error): the engine logs each round
    before anything afterwards can raise, and the caller computes this at
    result-build time when the error state is final.
    """
    trajectory = [_soft_reward(record.get("rewards")) for record in rounds_log]
    first_pass = next(
        (
            record["round"]
            for record, reward in zip(rounds_log, trajectory, strict=True)
            if reward is not None and reward >= user.pass_threshold
        ),
        None,
    )
    last_reward = trajectory[-1] if trajectory else None
    if error is not None:
        stop_reason = "error"
    elif last_reward is not None and last_reward >= user.pass_threshold:
        # Checked before the max_iterations fallback: a pass on the final
        # allowed round is passed_bar, not max_iterations.
        stop_reason = "passed_bar"
    elif len(rounds_log) >= max_rounds:
        stop_reason = "max_iterations"
    else:
        stop_reason = "strategy_stop"
    return {
        "iterations_run": len(rounds_log),
        "stop_reason": stop_reason,
        "reward_trajectory": trajectory,
        "first_pass_iteration": first_pass,
    }
