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


class RunFolderError(ValueError):
    """Raised when a run folder is missing required artifacts or malformed."""


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
        raise RunFolderError(f"expected a JSON object in {path}, got {type(data).__name__}")
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
        raise RunFolderError(
            f"missing required artifact: {path} — record-replay needs the LLM "
            "trajectory. Was this run captured with usage tracking enabled?"
        )
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
    exchanges = load_llm_exchanges(path / "trajectory" / "llm_trajectory.jsonl")

    run = RunFolder(
        path=path,
        config=config,
        result=result,
        prompts=prompts,
        exchanges=exchanges,
    )

    if run.agent != "openhands":
        raise RunFolderError(
            f"benchflow continue currently supports the 'openhands' agent only; "
            f"this run used {run.agent!r}."
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
