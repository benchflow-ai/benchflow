"""Data classes and exceptions for benchflow SDK results.

Related: sdk.py (produces RunResult), job.py (aggregates RunResults),
_scoring.py (extracts rewards and classifies errors from RunResults).
"""

from datetime import datetime
from typing import Any, Literal

TrajectorySource = Literal["acp", "scraped", "partial_acp"]
"""Provenance label for a captured trajectory. See RunResult.trajectory_source."""


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


class RunResult:
    """Outcome of a single SDK.run() trial.

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
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def success(self) -> bool:
        """True when the trial completed without agent or verifier error.

        Agent errors (error) and verifier errors (verifier_error) both
        indicate an incomplete trial. Rewards may still be zero on success.
        """
        return self.error is None and self.verifier_error is None

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error or self.verifier_error}"
        return (
            f"RunResult(task={self.task_name}, {status}, "
            f"rewards={self.rewards}, "
            f"trajectory={len(self.trajectory)} events)"
        )
