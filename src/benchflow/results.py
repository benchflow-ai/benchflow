"""Trial result types — TrialResult, VerifierResult, TrajectorySource.

Split out of the former ``benchflow.models`` (which mixed result dataclasses
with error classes). See PLAN_V2_shaping §3.4 / Phase 9.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

TrajectorySource = Literal["acp", "scraped", "partial_acp"]
"""Provenance label for a captured trajectory. See TrialResult.trajectory_source."""


class VerifierResult(BaseModel):
    """Outcome of a verifier run — the reward dict or None."""

    rewards: dict[str, float | int] | None = None


class TrialResult:
    """Outcome of a single trial.

    Attributes:
        task_name:    Task directory name (e.g. "swe-bench/django__django-11848").
        trial_name:   Unique trial identifier within a job run.
        rewards:      Verifier-produced reward dict (e.g. {"exact_match": 1.0}).
                      None if verification was skipped or failed.
        trajectory:   Ordered list of ACP session-update dicts (tool calls,
                      messages, thoughts) captured during execution.
        agent:        Harness name from the registry (e.g. "openclaw").
        agent_name:   Name reported by the agent via ACP initialize handshake.
        model:        Model ID used (e.g. "google/gemini-3.1-flash-lite-preview").
        n_tool_calls: Total tool calls observed during the session.
        n_prompts:    Number of user prompts sent to the agent.
        error:        Error description string, or None on success.
        verifier_error: Verifier error description, or None if verifier succeeded
                      or was not reached. Separate from ``error`` (agent errors).
        partial_trajectory: True when the trajectory was salvaged from a timed-out
                      or crashed session and may be incomplete.
        trajectory_source: Provenance label for ``trajectory`` — one of
                      ``"acp"`` (trusted), ``"scraped"`` (UNTRUSTED, agent-writable,
                      forgeable), ``"partial_acp"`` (partial ACP capture). Verifier
                      and metrics consumers decide trust per source. None if no
                      trajectory was captured.
        input_tokens: Cumulative prompt/input token count across all LLM calls in
                      this trial, or ``None`` if the source did not report it.
                      Populated either by an agent self-reporting via the BYOA
                      sidecar (``$BENCHFLOW_USAGE_PATH``) or by an OTel collector
                      consuming gen_ai.usage.input_tokens spans.
        output_tokens: Cumulative completion/output tokens, or ``None``.
        cache_tokens: Cumulative cache-read + cache-creation tokens (provider
                      semantics vary), or ``None``.
        cost_usd:     Cumulative cost in USD as reported by the source, or
                      ``None``. Benchflow does NOT compute pro-rata: a missing
                      value is reported honestly as ``null``.
        started_at:   Wall-clock start time.
        finished_at:  Wall-clock end time.
    """

    def __init__(
        self,
        task_name: str,
        trial_name: str = "",
        rewards: dict[str, float | int] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
        agent: str = "",
        agent_name: str = "",
        model: str = "",
        n_tool_calls: int = 0,
        n_prompts: int = 0,
        error: str | None = None,
        verifier_error: str | None = None,
        partial_trajectory: bool = False,
        trajectory_source: TrajectorySource | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_tokens: int | None = None,
        cost_usd: float | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ):
        self.task_name = task_name
        self.trial_name = trial_name
        self.rewards = rewards
        self.trajectory = trajectory or []
        self.agent = agent
        self.agent_name = agent_name
        self.model = model
        self.n_tool_calls = n_tool_calls
        self.n_prompts = n_prompts
        self.error = error
        self.verifier_error = verifier_error
        self.partial_trajectory = partial_trajectory
        self.trajectory_source = trajectory_source
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_tokens = cache_tokens
        self.cost_usd = cost_usd
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def success(self) -> bool:
        """True when the trial completed without agent or verifier error."""
        return self.error is None and self.verifier_error is None

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error or self.verifier_error}"
        return (
            f"TrialResult(task={self.task_name}, {status}, "
            f"rewards={self.rewards}, "
            f"trajectory={len(self.trajectory)} events)"
        )
