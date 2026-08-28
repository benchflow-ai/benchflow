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
from dataclasses import dataclass, field
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
    ``exchange_lines[i]`` is the verbatim source line of ``exchanges[i]`` —
    malformed lines are skipped at load time and therefore never appear here,
    so a stitched trajectory built from these lines contains exactly what the
    replay serves.
    """

    path: Path
    config: dict[str, Any]
    result: dict[str, Any]
    prompts: list[str]
    exchanges: list[LLMExchange]
    exchange_lines: list[str]
    # ``stage_snapshots.json``'s ``stages`` mapping when the original run
    # recorded stage boundaries (rollout-branching RFC §3.2/§3.5) — empty for
    # runs captured before stage snapshots existed or without them. Each entry
    # may carry ``exchanges_completed``, the completed-exchange index recorded
    # at capture time, which is what a ``--cut-stage`` request resolves.
    stage_registry: dict[str, Any] = field(default_factory=dict)

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

    # ── recorded stage boundaries (stage_snapshots.json) ──────────────────
    @property
    def recorded_stages(self) -> list[str]:
        """The stage boundaries the original run recorded, in file order."""
        return list(self.stage_registry)

    @property
    def stage_exchange_tags(self) -> dict[str, int]:
        """``stage -> completed-exchange count`` for stages with a usable index.

        A stage recorded with ``exchanges_completed: null`` (the run's usage
        gateway could not count at capture time) is omitted — callers that
        need to distinguish "never recorded" from "recorded without an index"
        read :attr:`stage_registry` directly.
        """
        tags: dict[str, int] = {}
        for stage, entry in self.stage_registry.items():
            value = (
                entry.get("exchanges_completed") if isinstance(entry, dict) else None
            )
            if isinstance(value, int) and not isinstance(value, bool):
                tags[stage] = value
        return tags


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


def _load_stage_registry(path: Path) -> dict[str, Any]:
    """Read ``stage_snapshots.json``'s ``stages`` mapping, tolerantly.

    The file is optional (runs recorded before stage snapshots existed, or
    without any stage capture, do not have it) and advisory: a malformed
    registry degrades to empty with a warning rather than stranding a
    continuation that never asked for a stage-named cut. The strict, typed
    errors live where a ``--cut-stage`` request actually resolves
    (:func:`benchflow.continue_run.orchestrator.stage_tags_from_run`).
    """
    if not path.is_file():
        return {}
    try:
        data = _read_json(path, required=False)
    except RunFolderError as exc:
        logger.warning("ignoring unreadable stage registry: %s", exc)
        return {}
    stages = data.get("stages")
    if not isinstance(stages, dict):
        logger.warning(
            "ignoring %s: expected a 'stages' mapping, got %s",
            path,
            type(stages).__name__,
        )
        return {}
    return stages


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


def load_llm_exchanges_with_lines(
    path: Path,
) -> tuple[list[LLMExchange], list[str]]:
    """Parse ``llm_trajectory.jsonl`` into exchanges plus their raw lines.

    One exchange per line (``Trajectory.to_jsonl``). Blank lines are skipped;
    a malformed line is skipped with a warning rather than aborting the whole
    resume (a single bad record should not strand a recoverable run). The
    second list carries, per *parsed* exchange, its verbatim source line —
    the replay counts parsed exchanges, so consumers that reconstruct the
    replayed prefix (trajectory stitching) must select these lines rather
    than truncating by raw file-line index, or a skipped malformed line would
    shift the cut.
    """
    if not path.is_file():
        raise RunFolderError(
            f"missing required artifact: {path} — record-replay needs the LLM "
            "trajectory. Was this run captured with usage tracking enabled?"
        )
    exchanges: list[LLMExchange] = []
    lines: list[str] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            exchanges.append(LLMExchange.model_validate_json(raw))
        except Exception as exc:
            logger.warning("skipping malformed llm_trajectory line %d: %s", lineno, exc)
        else:
            lines.append(raw)
    if not exchanges:
        raise RunFolderError(
            f"{path} contained no usable LLM exchanges — nothing to replay."
        )
    return exchanges, lines


def load_llm_exchanges(path: Path) -> list[LLMExchange]:
    """Parse ``llm_trajectory.jsonl`` into ordered :class:`LLMExchange` records.

    The original public contract, kept backward compatible; see
    :func:`load_llm_exchanges_with_lines` for the raw-line-aware variant.
    """
    return load_llm_exchanges_with_lines(path)[0]


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
    exchanges, exchange_lines = load_llm_exchanges_with_lines(
        path / "trajectory" / "llm_trajectory.jsonl"
    )
    stage_registry = _load_stage_registry(path / "stage_snapshots.json")

    run = RunFolder(
        path=path,
        config=config,
        result=result,
        prompts=prompts,
        exchanges=exchanges,
        exchange_lines=exchange_lines,
        stage_registry=stage_registry,
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
