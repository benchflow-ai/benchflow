"""Data classes and exceptions for benchflow results.

Related: rollout.py (produces RolloutResult), evaluation.py (aggregates results),
_scoring.py (extracts rewards and classifies errors from results).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from benchflow.usage_tracking import UsageSource

if TYPE_CHECKING:
    from benchflow._utils.scoring import ScoreOutcome
    from benchflow.review.outcome import ScoringResult
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
        scoring:      Explicit gate verdict for automatically reviewed tasks.
                      Its pass flag is independent of the quality reward.
        purpose:      Distinguishes task trials from nested reviewer runs.
        trajectory:   Ordered list of ACP session-update dicts (tool calls,
                      messages, thoughts) captured during execution.
        agent:        Harness name from the registry (e.g. "codex-acp").
        agent_name:   Name reported by the agent via ACP initialize handshake.
        model:        Model ID used (e.g. "google/gemini-3.1-flash-lite-preview").
        n_tool_calls: Total tool calls observed during the session.
        n_skill_invocations: Total skill tool calls observed in structured ACP
                      trajectory events. Counts only ``tool_call`` events whose
                      structured ``kind`` is ``"skill"``.
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
        usage_source: Token telemetry source. One of "provider_response",
                      "agent_native_acp", or "unavailable".
        price_source: Pricing table version used for cost_usd, or None.
        usage_details: Optional source-specific telemetry details.
        error:        Error description string, or None on success.
        error_category: Stable category for ``error``, or None on success.
        verifier_error: Verifier error description, or None if verifier succeeded
                      or was not reached. Separate from ``error`` (agent errors).
        verifier_error_category: Stable category for ``verifier_error``, or None.
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
                      the data path that feeds the persistent LearnerStore.
        source_provenance: Source repository/ref/file-hash evidence for the task.
        started_at:   Wall-clock start time.
        finished_at:  Wall-clock end time.
        rollout_dir:  Directory holding this rollout's artifacts (result.json,
                      trajectory/, verifier/, ...), or None when the rollout
                      failed before the directory was created.
        retry:        The ``--retry-from-checkpoint`` outcome (status, reason,
                      checkpoint, reward, original_reward), or None. The
                      retry's reward never replaces ``rewards``.

    Convenience properties: ``reward`` (the canonical ``rewards["reward"]``),
    ``passed`` (the scoring outcome is a pass), ``success`` (no agent, verifier
    or export error) and ``score_outcome``. ``RolloutResult.load(path)`` reads
    a finished rollout back from its ``result.json``.
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
        n_skill_invocations: int = 0,
        n_prompts: int = 0,
        n_input_tokens: int | None = None,
        n_output_tokens: int | None = None,
        n_cache_read_tokens: int | None = None,
        n_cache_creation_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        usage_source: UsageSource = "unavailable",
        price_source: str | None = None,
        usage_details: dict[str, Any] | None = None,
        error: str | None = None,
        error_category: str | None = None,
        verifier_error: str | None = None,
        verifier_error_category: str | None = None,
        export_error: str | None = None,
        partial_trajectory: bool = False,
        trajectory_source: TrajectorySource | None = None,
        reward_events: list[RewardEvent] | None = None,
        evolved_skills: dict[str, str] | None = None,
        source_provenance: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        scoring: ScoringResult | None = None,
        purpose: Literal["task", "reviewer"] = "task",
        parent_rollout: str | None = None,
        rollout_dir: Path | None = None,
        retry: dict[str, Any] | None = None,
    ):
        self.task_name = task_name
        self.rollout_name = rollout_name
        self.rewards = rewards
        self.trajectory = trajectory or []
        self.agent = agent
        self.agent_name = agent_name
        self.model = model
        self.n_tool_calls = n_tool_calls
        self.n_skill_invocations = n_skill_invocations
        self.n_prompts = n_prompts
        self.n_input_tokens = n_input_tokens
        self.n_output_tokens = n_output_tokens
        self.n_cache_read_tokens = n_cache_read_tokens
        self.n_cache_creation_tokens = n_cache_creation_tokens
        self.total_tokens = total_tokens
        self.cost_usd = cost_usd
        self.usage_source = usage_source
        self.price_source = price_source
        self.usage_details = usage_details
        self.error = error
        self.error_category = error_category
        self.verifier_error = verifier_error
        self.verifier_error_category = verifier_error_category
        self.export_error = export_error
        self.partial_trajectory = partial_trajectory
        self.trajectory_source = trajectory_source
        self.reward_events = reward_events
        self.evolved_skills = evolved_skills
        self.source_provenance = source_provenance
        self.started_at = started_at
        self.finished_at = finished_at
        self.scoring = scoring
        self.purpose = purpose
        self.parent_rollout = parent_rollout
        self.rollout_dir = rollout_dir
        # A retry from the trial's last checkpoint (benchflow.checkpoint_retry),
        # reported next to ``rewards``, never merged into them.
        self.retry = retry

    @property
    def reward(self) -> float | None:
        """The canonical scalar reward, ``rewards["reward"]``; None when unscored.

        >>> RolloutResult("t", rewards={"reward": 0.5, "tests": 1.0}).reward
        0.5
        >>> RolloutResult("t", error="agent crashed").reward is None
        True
        """
        if not self.rewards:
            return None
        value = self.rewards.get("reward")
        return None if value is None else float(value)

    @property
    def passed(self) -> bool:
        """True when the rollout was scored and passed (``score_outcome == "passed"``).

        >>> RolloutResult("t", rewards={"reward": 1.0}).passed
        True
        >>> RolloutResult("t", rewards={"reward": 0.0}).score_outcome
        'failed'
        """
        return self.score_outcome == "passed"

    def to_record(self) -> dict[str, Any]:
        """One flat, JSON-safe dict of the headline fields (a CSV/JSONL row).

        >>> RolloutResult("t", rewards={"reward": 1.0}).to_record()["passed"]
        True
        """
        duration = (
            (self.finished_at - self.started_at).total_seconds()
            if self.started_at is not None and self.finished_at is not None
            else None
        )
        return {
            "task_name": self.task_name,
            "rollout_name": self.rollout_name,
            "agent": self.agent,
            "model": self.model,
            "reward": self.reward,
            "passed": self.passed,
            "score_outcome": self.score_outcome,
            "error": self.error,
            "error_category": self.error_category,
            "verifier_error": self.verifier_error,
            "verifier_error_category": self.verifier_error_category,
            "n_tool_calls": self.n_tool_calls,
            "n_prompts": self.n_prompts,
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
            "n_cache_read_tokens": self.n_cache_read_tokens,
            "n_cache_creation_tokens": self.n_cache_creation_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "usage_source": self.usage_source,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_sec": duration,
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
        }

    @property
    def verified(self) -> bool:
        """Deprecated: use ``score_outcome in ("passed", "failed")``.

        True when the verifier produced a verdict (pass or fail). Kept from the
        retired ``RuntimeResult``.
        """
        import warnings

        warnings.warn(
            "RolloutResult.verified is deprecated; use "
            'result.score_outcome in ("passed", "failed").',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.score_outcome in ("passed", "failed")

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Deprecated and always empty (``RuntimeResult`` never populated it).

        The agent's messages are the ``agent_message`` events in ``trajectory``.
        """
        import warnings

        warnings.warn(
            "RolloutResult.messages is deprecated and always empty; read the "
            '"agent_message" events in result.trajectory.',
            DeprecationWarning,
            stacklevel=2,
        )
        return []

    @property
    def snapshots(self) -> list[str]:
        """Deprecated and always empty (``RuntimeResult`` never populated it)."""
        import warnings

        warnings.warn(
            "RolloutResult.snapshots is deprecated and always empty; branch "
            "checkpoints are recorded in the rollout's tree.json.",
            DeprecationWarning,
            stacklevel=2,
        )
        return []

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, rollout_dir: Path | None = None
    ) -> RolloutResult:
        """Build a result from a persisted ``result.json`` payload.

        Keys the constructor does not take (``timing``, ``sandbox_id``, ...)
        are ignored; read them from the dict or the file directly. Token and
        cost fields are taken from the ``agent_result`` block when they are not
        at the top level.
        """
        import inspect

        from benchflow.review.outcome import ScoringResult

        accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
        merged = {**(data.get("agent_result") or {}), **data}
        kwargs = {k: v for k, v in merged.items() if k in accepted}
        kwargs.pop("rollout_dir", None)
        for key in ("started_at", "finished_at"):
            value = kwargs.get(key)
            if isinstance(value, str):
                try:
                    kwargs[key] = datetime.fromisoformat(value)
                except ValueError:
                    kwargs[key] = None
        scoring = kwargs.get("scoring")
        if isinstance(scoring, dict):
            kwargs["scoring"] = ScoringResult.model_validate(scoring, strict=False)
        kwargs.setdefault("task_name", "")
        return cls(**kwargs, rollout_dir=rollout_dir)

    @classmethod
    def load(cls, path: str | Path) -> RolloutResult:
        """Read a finished rollout from its directory (or its ``result.json``).

        The trajectory is read from ``trajectory/acp_trajectory.jsonl`` when
        present. Raises ``FileNotFoundError`` naming the missing file.
        """
        path = Path(path)
        result_file = path if path.name.endswith(".json") else path / "result.json"
        if not result_file.is_file():
            raise FileNotFoundError(
                f"No result.json at {result_file}; pass a rollout directory "
                "(jobs/<job>/<task>__<id>) or its result.json"
            )
        rollout_dir = result_file.parent
        data = json.loads(result_file.read_text())
        result = cls.from_dict(data, rollout_dir=rollout_dir)
        traj_file = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
        if traj_file.is_file():
            result.trajectory = [
                json.loads(line)
                for line in traj_file.read_text().splitlines()
                if line.strip()
            ]
        return result

    @property
    def score_outcome(self) -> ScoreOutcome:
        """Canonical scoring classification shared by runtime and saved reports."""
        from benchflow._utils.scoring import classify_score_outcome

        return classify_score_outcome(
            {
                "rewards": self.rewards,
                "scoring": self.scoring.to_dict() if self.scoring else None,
                "error": self.error,
                "verifier_error": self.verifier_error,
            }
        )

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
            and (self.scoring is None or self.scoring.status == "complete")
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
