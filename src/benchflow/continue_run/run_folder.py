"""Load and validate an original run's output folder for ``benchflow continue``.

A benchflow rollout (or an HF-downloaded copy of one) is a directory containing:

- ``config.json``   — task identity, agent, model, environment, original timeout,
  and ``source`` provenance (written by ``benchflow.rollout._write_config``).
- ``result.json``   — terminal status (``error`` / ``error_category``), rewards.
- ``prompts.json``  — the prompts handed to the agent (the first is the task
  instruction).
- ``trajectory/llm_trajectory.jsonl`` — one :class:`LLMExchange` per line
  (``Trajectory.to_jsonl``): the recorded LLM request/response pairs that
  drive record-replay.

This module is pure (no sandbox, no network) so the whole load + validate path
is unit-testable offline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.trajectories.types import LLMExchange

logger = logging.getLogger(__name__)

# error_category values (see benchflow._utils.scoring.classify_error) that mean
# "the agent ran out of time", i.e. the run is genuinely *unfinished* and worth
# continuing rather than a clean pass/fail.
_TIMEOUT_CATEGORIES = frozenset({"timeout", "idle_timeout"})

# ── which agents ``benchflow continue`` can resume ────────────────────────────
#
# Single source of truth: a future replay ingress adds its agent here and
# nowhere else. The membership is *protocol*-derived, not a policy about open
# vs closed model weights — see ``_WHY_PROTOCOL`` below for the mechanism.
CONTINUE_SUPPORTED_AGENTS: frozenset[str] = frozenset({"openhands"})

_TRAJECTORY_RELPATH = "trajectory/llm_trajectory.jsonl"

_WHY_PROTOCOL = (
    "That set is derived from the replay wire protocol, not from a policy about "
    "which models are open: the replay proxy serves POST /v1/chat/completions "
    "only, and the continuation hands the sandbox LLM_BASE_URL / LLM_API_KEY / "
    "LLM_MODEL, which only the OpenHands agent template consumes. An agent that "
    "speaks a different wire (Anthropic Messages, OpenAI Responses, Google "
    "native) would 404 at the proxy and fall back to host credentials, burning a "
    "full live run that the artifacts would then mislabel as a replay."
)

_NO_RECORDING_VERDICT = (
    f"This run has no {_TRAJECTORY_RELPATH}, which is what a subscription-auth "
    "run looks like: it bypasses the recording gateway, so nothing was captured "
    "and it can never be continued — no replay ingress, present or future, can "
    "reconstruct it."
)

# Offered as a possibility, not a promise: snapshot capture is opt-in and most
# runs will not have one.
_SNAPSHOT_HINT = (
    "If a sandbox snapshot of the original container was captured, that is the "
    "only remaining route to its state."
)


def _supported_agents_phrase() -> str:
    names = sorted(CONTINUE_SUPPORTED_AGENTS)
    return ", ".join(repr(name) for name in names)


class RunFolderError(ValueError):
    """Raised when a run folder is missing required artifacts or malformed."""


class ContinueUnsupportedError(RunFolderError):
    """Triage verdict: ``benchflow continue`` cannot resume *this* run.

    Distinct from a failure — nothing went wrong, the run is simply out of
    scope. Batch mode records these as skips and keeps going (see
    :mod:`benchflow.continue_run.batch`); the single-run CLI still exits 1.
    """

    #: stable, machine-readable reason for batch summaries
    reason_code: str = "continue_unsupported"
    #: whether a *future* replay ingress could ever rescue this run
    recoverable_in_principle: bool = False


class UnsupportedAgentError(ContinueUnsupportedError):
    """The run's agent has no replay ingress (see ``CONTINUE_SUPPORTED_AGENTS``)."""

    reason_code = "unsupported_agent"

    def __init__(self, *, agent: str, has_recording: bool) -> None:
        self.agent = agent
        self.has_recording = has_recording
        self.recoverable_in_principle = has_recording
        self.supported_agents: tuple[str, ...] = tuple(
            sorted(CONTINUE_SUPPORTED_AGENTS)
        )
        used = f"used agent {agent!r}" if agent else "recorded no agent"
        if has_recording:
            verdict = (
                f"This run does have {_TRAJECTORY_RELPATH}, so it becomes "
                f"continuable as soon as a replay ingress for {agent!r} ships — "
                "the recording is not the blocker."
            )
        else:
            verdict = _NO_RECORDING_VERDICT
        super().__init__(
            f"benchflow continue cannot resume this run: it {used}, and the only "
            f"supported agent(s) are {_supported_agents_phrase()}. "
            f"{_WHY_PROTOCOL} {verdict} {_SNAPSHOT_HINT}"
        )


class MissingRecordingError(ContinueUnsupportedError):
    """No LLM recording exists, so no replay can reconstruct the run."""

    reason_code = "no_llm_recording"
    recoverable_in_principle = False

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"missing required artifact: {path} — record-replay needs the LLM "
            f"trajectory. {_NO_RECORDING_VERDICT} {_SNAPSHOT_HINT}"
        )


@dataclass(frozen=True)
class RunFolder:
    """Parsed view of an original run's output folder.

    Only fields ``benchflow continue`` needs are surfaced; the raw ``config``
    and ``result`` dicts are kept for anything else the orchestrator wants.
    """

    path: Path
    config: dict[str, Any]
    result: dict[str, Any]
    prompts: list[str]
    exchanges: list[LLMExchange]

    # ── derived task identity (from config.json) ──────────────────────────
    @property
    def task_path(self) -> str:
        return str(self.config.get("task_path") or "")

    @property
    def task_name(self) -> str:
        """The task directory name (e.g. ``energy-unit-commitment``)."""
        tp = self.task_path
        return Path(tp).name if tp else str(self.result.get("task_name") or "")

    @property
    def agent(self) -> str:
        return str(self.config.get("agent") or self.result.get("agent") or "")

    @property
    def model(self) -> str | None:
        model = self.config.get("model") or self.result.get("model")
        return str(model) if model else None

    @property
    def environment(self) -> str:
        return str(self.config.get("environment") or "docker")

    @property
    def sandbox_user(self) -> str | None:
        user = self.config.get("sandbox_user")
        return str(user) if user else None

    @property
    def reasoning_effort(self) -> str | None:
        effort = self.config.get("reasoning_effort")
        return str(effort) if effort else None

    @property
    def timeout_sec(self) -> int | None:
        value = self.config.get("timeout_sec")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def agent_idle_timeout_sec(self) -> int | None:
        value = self.config.get("agent_idle_timeout_sec")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def error_category(self) -> str | None:
        cat = self.result.get("error_category")
        return str(cat) if cat else None

    @property
    def is_timeout(self) -> bool:
        """Whether the recorded terminal status is a timeout/idle-timeout."""
        return self.error_category in _TIMEOUT_CATEGORIES

    @property
    def n_recorded_exchanges(self) -> int:
        return len(self.exchanges)


def _read_json(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise RunFolderError(f"missing required artifact: {path}")
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunFolderError(f"could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunFolderError(
            f"expected a JSON object in {path}, got {type(data).__name__}"
        )
    return data


def _load_prompts(path: Path) -> list[str]:
    """Read ``prompts.json`` — a JSON list of strings (or ``{"prompts": [...]}``)."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunFolderError(f"could not parse {path}: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("prompts", [])
    if not isinstance(data, list):
        raise RunFolderError(f"expected a JSON list in {path}")
    return [str(p) for p in data if p is not None]


def load_llm_exchanges(path: Path) -> list[LLMExchange]:
    """Parse ``llm_trajectory.jsonl`` into ordered :class:`LLMExchange` records.

    One exchange per line (``Trajectory.to_jsonl``). Blank lines are skipped;
    a malformed line is skipped with a warning rather than aborting the whole
    resume (a single bad record should not strand a recoverable run).
    """
    if not path.is_file():
        raise MissingRecordingError(path)
    exchanges: list[LLMExchange] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            exchanges.append(LLMExchange.model_validate_json(raw))
        except Exception as exc:
            logger.warning("skipping malformed llm_trajectory line %d: %s", lineno, exc)
    if not exchanges:
        raise RunFolderError(
            f"{path} contained no usable LLM exchanges — nothing to replay."
        )
    return exchanges


def load_run_folder(folder: str | Path, *, require_timeout: bool = False) -> RunFolder:
    """Load + validate an original run folder.

    ``require_timeout`` rejects runs whose recorded status is not a
    timeout/idle-timeout. The default is permissive (warn only): a run with no
    recorded ``error_category`` may still be worth continuing, and the user can
    opt into strictness.
    """
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise RunFolderError(f"not a directory: {path}")

    config = _read_json(path / "config.json", required=True)
    result = _read_json(path / "result.json", required=False)
    prompts = _load_prompts(path / "prompts.json")

    # Triage before parsing. The agent gate runs first so an unsupported run is
    # told whether it is blocked on a missing ingress (recoverable later) or on
    # a missing recording (never recoverable) — the caller should learn that now,
    # not after a protocol ingress ships.
    trajectory_path = path / "trajectory" / "llm_trajectory.jsonl"
    agent = str(config.get("agent") or result.get("agent") or "")
    if agent not in CONTINUE_SUPPORTED_AGENTS:
        raise UnsupportedAgentError(
            agent=agent, has_recording=trajectory_path.is_file()
        )

    exchanges = load_llm_exchanges(trajectory_path)

    run = RunFolder(
        path=path,
        config=config,
        result=result,
        prompts=prompts,
        exchanges=exchanges,
    )

    if not run.is_timeout:
        msg = (
            f"run {path.name} has error_category={run.error_category!r}, not a "
            "timeout/idle_timeout — continuing it may not be meaningful."
        )
        if require_timeout:
            raise RunFolderError(msg)
        logger.warning(msg)

    return run
