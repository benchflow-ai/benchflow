"""Public evidence contracts for stopping and observing an ACP session."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from benchflow.trajectories.types import redact_trajectory_obj

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTerminationReceipt:
    """Observed outcomes from one idempotent ACP stop operation."""

    cancel_requested: bool
    cancel_acknowledged: bool
    session_closed: bool
    graceful_termination: bool
    force_kill_required: bool
    process_tree_stopped: bool

    def __post_init__(self) -> None:
        if self.cancel_acknowledged and not self.cancel_requested:
            raise ValueError(
                "cancellation cannot be acknowledged when it was not requested"
            )
        if self.graceful_termination and self.force_kill_required:
            raise ValueError("graceful termination cannot require a force kill")
        if self.graceful_termination and not self.process_tree_stopped:
            raise ValueError("graceful termination requires a stopped process tree")

    @property
    def capture_safe(self) -> bool:
        """Whether artifact capture may safely observe the final workspace."""
        return self.session_closed and self.process_tree_stopped


@dataclass(frozen=True, slots=True, kw_only=True)
class AcpSessionObservation:
    """Facts reported by a live ACP session and its retained trajectory."""

    agent_name: str
    model_id: str | None
    mode_id: str | None
    stop_reason: str | None
    trajectory_path: str | None
    trajectory_digest: str | None

    def __post_init__(self) -> None:
        if not self.agent_name:
            raise ValueError("agent_name must come from a non-empty ACP observation")
        if (self.trajectory_path is None) != (self.trajectory_digest is None):
            raise ValueError(
                "trajectory_path and trajectory_digest must be present together"
            )


def _state_string(state: object | None, wire_name: str, attr_name: str) -> str | None:
    if isinstance(state, dict):
        value = state.get(wire_name)
    else:
        value = getattr(state, attr_name, None)
    return value if isinstance(value, str) and value else None


def _observed_config_value(session: object, category: str) -> str | None:
    options = getattr(session, "config_options", None)
    if not isinstance(options, (list, tuple)):
        return None
    for option in options:
        if isinstance(option, dict):
            option_id = option.get("id")
            option_category = option.get("category")
            current_value = option.get("currentValue")
        else:
            option_id = getattr(option, "id", None)
            option_category = getattr(option, "category", None)
            current_value = getattr(option, "current_value", None)
        if (option_id == category or option_category == category) and isinstance(
            current_value, str
        ):
            return current_value
    return None


def _trajectory_identity(path: Path | None) -> tuple[str | None, str | None]:
    if path is None or not path.is_file():
        return None, None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as trajectory:
            for chunk in iter(lambda: trajectory.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        logger.warning("Could not digest ACP trajectory %s", path, exc_info=True)
        return None, None
    return str(path), f"sha256:{digest.hexdigest()}"


def _redact_observation_fields(
    session: object, fields: dict[str, str | None]
) -> dict[str, str | None]:
    redactor = getattr(type(session), "redact_for_persistence", None)
    redacted = (
        redactor(session, fields)
        if callable(redactor)
        else redact_trajectory_obj(fields)
    )
    return redacted if isinstance(redacted, dict) else fields


def _observe_acp_session(
    session: object | None,
    *,
    trajectory_path: Path | None,
) -> AcpSessionObservation | None:
    """Project only facts returned by the live ACP session."""
    if session is None:
        return None
    agent_info = getattr(session, "agent_info", None)
    if isinstance(agent_info, dict):
        agent_name = agent_info.get("name")
    else:
        agent_name = getattr(agent_info, "name", None)
    if not isinstance(agent_name, str) or not agent_name:
        return None

    model_id = _observed_config_value(session, "model") or _state_string(
        getattr(session, "model_state", None), "currentModelId", "current_model_id"
    )
    mode_id = _observed_config_value(session, "mode") or _state_string(
        getattr(session, "mode_state", None), "currentModeId", "current_mode_id"
    )
    stop_reason = getattr(session, "stop_reason", None)
    if stop_reason is not None:
        stop_reason = getattr(stop_reason, "value", stop_reason)
    if not isinstance(stop_reason, str):
        stop_reason = None
    fields = _redact_observation_fields(
        session,
        {
            "agent_name": agent_name,
            "model_id": model_id,
            "mode_id": mode_id,
            "stop_reason": stop_reason,
        },
    )
    redacted_agent_name = fields.get("agent_name")
    if not isinstance(redacted_agent_name, str) or not redacted_agent_name:
        return None
    observed_path, observed_digest = _trajectory_identity(trajectory_path)
    return AcpSessionObservation(
        agent_name=redacted_agent_name,
        model_id=fields.get("model_id"),
        mode_id=fields.get("mode_id"),
        stop_reason=fields.get("stop_reason"),
        trajectory_path=observed_path,
        trajectory_digest=observed_digest,
    )
