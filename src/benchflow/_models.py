"""Data classes and exceptions for benchflow SDK results."""

from datetime import datetime
from typing import Any


class AgentInstallError(RuntimeError):
    """Agent installation failed in the sandbox."""
    def __init__(self, agent: str, return_code: int, stdout: str, diagnostics: str, log_path: str = ""):
        self.agent = agent
        self.return_code = return_code
        self.stdout = stdout
        self.diagnostics = diagnostics
        self.log_path = log_path
        super().__init__(f"Agent {agent} install failed (rc={return_code})")


class AgentTimeoutError(RuntimeError):
    """Agent execution timed out."""
    def __init__(self, agent: str, timeout_sec: float):
        self.agent = agent
        self.timeout_sec = timeout_sec
        super().__init__(f"Agent {agent} timed out after {timeout_sec}s")


class RunResult:
    """Result of a benchflow run."""

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
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ):
        self.task_name = task_name
        self.trial_name = trial_name
        self.rewards = rewards
        self.trajectory = trajectory or []
        self.agent = agent  # harness name (e.g. "openclaw")
        self.agent_name = agent_name  # ACP-reported name
        self.model = model  # model ID (e.g. "google/gemini-3.1-flash-lite-preview")
        self.n_tool_calls = n_tool_calls
        self.n_prompts = n_prompts
        self.error = error
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error}"
        return (
            f"RunResult(task={self.task_name}, {status}, "
            f"rewards={self.rewards}, "
            f"trajectory={len(self.trajectory)} events)"
        )
