"""Job config dataclasses — RetryConfig, JobConfig, JobResult.

CONTRACT SURFACE — semver-stable. Changes here break downstream importers.
Prefer extending in periphery (``benchflow.job``) unless the shape itself
must change.

Declarative inputs to ``benchflow.job.Job``. Lives in ``contracts/`` so the
scheduler in ``job.py`` can evolve without dragging semver guarantees into
every orchestration refactor. ``job.py`` re-exports these names for
backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from benchflow.contracts.scoring import (
    ACP_ERROR,
    INSTALL_FAILED,
    PIPE_CLOSED,
    classify_error,
    pass_rate,
    pass_rate_excl_errors,
)


# Defaults: works out-of-the-box with `claude login`
# (subscription auth, no API key needed)
DEFAULT_AGENT = "claude-agent-acp"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def effective_model(agent: str, model: str | None) -> str | None:
    """Resolve the model an agent should run with.

    Oracle runs solve.sh and never calls an LLM, so it never receives a model
    (the chokepoint in resolve_agent_env defends, but callers should also stop
    materializing DEFAULT_MODEL into oracle configs to keep the data honest —
    e.g. result-summary JSON shows model=null instead of a bogus default).
    """
    if agent == "oracle":
        return None
    return model or DEFAULT_MODEL


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Matches Harbor's RetryConfig pattern: exponential backoff with
    configurable exception filtering. Legacy boolean fields are
    preserved for backwards compat but the category-based check
    covers all cases.
    """

    max_retries: int = 2
    retry_on_install: bool = True
    retry_on_pipe: bool = True
    retry_on_acp: bool = True
    wait_multiplier: float = 2.0
    min_wait_sec: float = 1.0
    max_wait_sec: float = 30.0
    exclude_categories: set[str] = field(default_factory=lambda: {"timeout"})

    def should_retry(self, error: str | None) -> bool:
        """Check if an error is retryable."""
        category = classify_error(error)
        if not category:
            return False
        if category in self.exclude_categories:
            return False
        if self.retry_on_install and category == INSTALL_FAILED:
            return True
        if self.retry_on_pipe and category == PIPE_CLOSED:
            return True
        if self.retry_on_acp and category == ACP_ERROR:
            return True
        return False

    def backoff_delay(self, attempt: int) -> float:
        """Exponential backoff delay for retry attempt."""
        delay = self.min_wait_sec * (self.wait_multiplier**attempt)
        return min(delay, self.max_wait_sec)


@dataclass
class JobConfig:
    """Configuration for a benchmark job."""

    agent: str = DEFAULT_AGENT
    model: str | None = None
    environment: str = "docker"
    concurrency: int = 4
    prompts: list[str | None] | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    retry: RetryConfig = field(default_factory=RetryConfig)
    skills_dir: str | None = None
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    context_root: str | None = None
    exclude_tasks: set[str] = field(default_factory=set)

    # Registry validation — previously in __post_init__ — moved to Job.__init__
    # (benchflow.job) so JobConfig has no runtime dependency on the agents
    # registry. Contracts/ must not import periphery; the registry check runs
    # at the scheduler boundary instead, which is where the warning actually
    # matters (direct JobConfig() construction without running a Job is a
    # library-use case where the warning is noise).


@dataclass
class JobResult:
    """Aggregated results for a job."""

    job_name: str
    config: JobConfig
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    verifier_errored: int = 0
    elapsed_sec: float = 0.0

    @property
    def score(self) -> float:
        """Pass rate over all tasks."""
        return pass_rate(passed=self.passed, total=self.total)

    @property
    def score_excl_errors(self) -> float:
        """Pass rate excluding errored tasks."""
        return pass_rate_excl_errors(passed=self.passed, failed=self.failed)
