"""Data classes and exceptions for benchflow results.

Related: rollout.py (produces RolloutResult), evaluation.py (aggregates results),
_scoring.py (extracts rewards and classifies errors from results).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from benchflow.rewards.events import RewardEvent

TrajectorySource = Literal["acp", "scraped", "partial_acp", "hosted_env"]
"""Provenance label for a captured trajectory. See RunResult.trajectory_source.

``"hosted_env"`` marks UNTRUSTED imported evidence produced by an external
hub (e.g. PrimeIntellect Verifiers). The events are reconstructed from
``vf-eval`` results, not captured over BenchFlow's ACP transport.
"""


class AgentInstallError(RuntimeError):
    """Agent installation failed in the sandbox.

    Raised by ``_agent_setup.install_agent()`` when the agent's install
    script exits non-zero. ``diagnostics`` contains the last N lines of
    output for triage; ``log_path`` points to the full log on disk.
    """

    def __init__(
        self,
        agent: str,
        return_code: int,
        stdout: str,
        diagnostics: str,
        log_path: str = "",
    ):
        self.agent = agent
        self.return_code = return_code
        self.stdout = stdout
        self.diagnostics = diagnostics
        self.log_path = log_path
        super().__init__(f"Agent {agent} install failed (rc={return_code})")


class AgentTimeoutError(RuntimeError):
    """Agent execution exceeded the allowed wall-clock time.

    Raised by ``_acp_run.execute_prompts()`` when the agent does not
    complete within ``timeout_sec`` seconds.
    """

    def __init__(self, agent: str, timeout_sec: float):
        self.agent = agent
        self.timeout_sec = timeout_sec
        super().__init__(f"Agent {agent} timed out after {timeout_sec}s")


class RolloutResult:
    """Outcome of a single rollout execution.

    Attributes:
        task_name:    Task directory name (e.g. "swe-bench/django__django-11848").
        rollout_name:   Unique trial identifier within a job run.
        rewards:      Verifier-produced reward dict (e.g. {"exact_match": 1.0}).
                      None if verification was skipped or failed.
        trajectory:   Ordered list of ACP session-update dicts (tool calls,
                      messages, thoughts) captured during execution.
        agent:        Harness name from the registry (e.g. "openclaw").
        agent_name:   Name reported by the agent via ACP initialize handshake.
        model:        Model ID used (e.g. "google/gemini-3.1-flash-lite-preview").
        n_tool_calls: Total tool calls observed during the session.
        n_prompts:    Number of user prompts sent to the agent.
        n_input_tokens: Cumulative provider prompt/input tokens, or None when
                      provider telemetry was unavailable.
        n_output_tokens: Cumulative provider completion/output tokens, or None
                      when provider telemetry was unavailable.
        n_cache_read_tokens: Provider prompt-cache read tokens, or None when
                      provider telemetry was unavailable.
        n_cache_creation_tokens: Provider prompt-cache creation tokens, or None
                      when provider telemetry was unavailable.
        total_tokens: Sum of input, output, cache-read, and cache-creation tokens,
                      or None when provider telemetry was unavailable.
        cost_usd:     Provider cost estimate in USD, or None when unavailable.
        usage_source: Provider telemetry source. One of "provider_response" or
                      "unavailable".
        price_source: Pricing table version used for cost_usd, or None.
        error:        Error description string, or None on success.
        verifier_error: Verifier error description, or None if verifier succeeded
                      or was not reached. Separate from ``error`` (agent errors).
        export_error: Skill-export error description, or None if export succeeded
                      or was not configured. Separate from ``error`` (which would
                      mis-classify export-time infra failures as agent failures)
                      and ``verifier_error``. See #389 follow-up.
        partial_trajectory: True when the trajectory was salvaged from a timed-out
                      or crashed session and may be incomplete.
        trajectory_source: Provenance label for ``trajectory`` — one of
                      ``"acp"`` (trusted), ``"scraped"`` (UNTRUSTED, agent-writable,
                      forgeable), ``"partial_acp"`` (partial ACP capture). Verifier
                      and metrics consumers decide trust per source. None if no
                      trajectory was captured.
        reward_events: Dense and terminal reward events from Rubric scoring.
                      None when the new reward pipeline was not used.
        evolved_skills: The skills the rollout's agent generated or evolved,
                      as a ``name -> body`` dict. Populated only by a
                      continual-learning (``sequential-shared``) rollout that
                      captured an exported skill set; None otherwise. This is
                      the data path that feeds the persistent LearnerStore
                      (capability 5).
        source_provenance: Source repository/ref/file-hash evidence for the task.
        timing:       Phase timing metrics persisted in result.json.
        started_at:   Wall-clock start time.
        finished_at:  Wall-clock end time.
    """

    def __init__(
        self,
        task_name: str,
        rollout_name: str = "",
        rewards: dict[str, Any] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
        agent: str = "",
        agent_name: str = "",
        model: str | None = None,
        n_tool_calls: int = 0,
        n_prompts: int = 0,
        n_input_tokens: int | None = None,
        n_output_tokens: int | None = None,
        n_cache_read_tokens: int | None = None,
        n_cache_creation_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        usage_source: str = "unavailable",
        price_source: str | None = None,
        error: str | None = None,
        verifier_error: str | None = None,
        export_error: str | None = None,
        partial_trajectory: bool = False,
        trajectory_source: TrajectorySource | None = None,
        reward_events: list[RewardEvent] | None = None,
        evolved_skills: dict[str, str] | None = None,
        source_provenance: dict[str, Any] | None = None,
        timing: dict[str, float] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ):
        self.task_name = task_name
        self.rollout_name = rollout_name
        self.rewards = rewards
        self.trajectory = trajectory or []
        self.agent = agent
        self.agent_name = agent_name
        self.model = model
        self.n_tool_calls = n_tool_calls
        self.n_prompts = n_prompts
        self.n_input_tokens = n_input_tokens
        self.n_output_tokens = n_output_tokens
        self.n_cache_read_tokens = n_cache_read_tokens
        self.n_cache_creation_tokens = n_cache_creation_tokens
        self.total_tokens = total_tokens
        self.cost_usd = cost_usd
        self.usage_source = usage_source
        self.price_source = price_source
        self.error = error
        self.verifier_error = verifier_error
        self.export_error = export_error
        self.partial_trajectory = partial_trajectory
        self.trajectory_source = trajectory_source
        self.reward_events = reward_events
        self.evolved_skills = evolved_skills
        self.source_provenance = source_provenance
        self.timing = timing or {}
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def success(self) -> bool:
        """True when the trial completed without agent, verifier, or export error.

        Agent errors (error), verifier errors (verifier_error), and skill-export
        errors (export_error) all indicate an incomplete trial. Rewards may
        still be zero on success.
        """
        return (
            self.error is None
            and self.verifier_error is None
            and self.export_error is None
        )

    def __repr__(self) -> str:
        status = (
            "OK"
            if self.success
            else f"ERROR: {self.error or self.verifier_error or self.export_error}"
        )
        return (
            f"RolloutResult(task={self.task_name}, {status}, "
            f"rewards={self.rewards}, "
            f"trajectory={len(self.trajectory)} events)"
        )


# Backward-compat alias
RunResult = RolloutResult
