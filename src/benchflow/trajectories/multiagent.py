"""Relationship-aware trajectory artifacts for real multi-agent rollouts.

This module is intentionally framework-agnostic: a multi-agent run is a graph of
real agent sessions, not a graph of LLM calls. The rollout/user-loop drivers call
this recorder around actual agent sessions and it writes durable artifacts next
to the existing ACP trajectory.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class RealAgentSessionRecord:
    """One launched or attached real agent session in a rollout."""

    session_id: str
    agent_id: str
    agent_type: str
    model: str | None
    driver: str
    workspace_mode: str
    scene: str | None
    scene_index: int | None
    turn_index: int | None
    started_at: str
    ended_at: str | None = None
    trajectory_path: str | None = None
    native_transcript_path: str | None = None
    workspace_diff_path: str | None = None
    n_trajectory_events: int = 0
    n_tool_calls: int = 0
    handoff_in: list[str] = field(default_factory=list)
    handoff_out: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class RealAgentHandoffRecord:
    """Relationship between two real agent sessions or their artifacts."""

    handoff_id: str
    from_session_id: str
    to_session_id: str
    from_agent_id: str
    to_agent_id: str
    relation: str
    scene: str | None
    created_at: str
    artifacts: list[str] = field(default_factory=list)


class RealAgentTraceRecorder:
    """Write per-agent trajectories and a relationship graph for a rollout.

    The recorder is additive. It does not replace ``trajectory/acp_trajectory``;
    it preserves isolated source views under ``trajectory/agents`` and writes a
    normalized graph for viewers and downstream analysis.
    """

    def __init__(self, rollout_dir: Path, *, rollout_id: str | None = None) -> None:
        self.rollout_dir = Path(rollout_dir)
        self.rollout_id = rollout_id or self.rollout_dir.name
        self.trajectory_dir = self.rollout_dir / "trajectory"
        self.agents_dir = self.trajectory_dir / "agents"
        self._sessions: list[RealAgentSessionRecord] = []
        self._handoffs: list[RealAgentHandoffRecord] = []
        self._events: list[dict[str, Any]] = []
        self._last_finished_session_id: str | None = None
        self._session_counts: dict[str, int] = {}

    @classmethod
    def for_rollout(cls, rollout: Any) -> "RealAgentTraceRecorder | None":
        rollout_dir = getattr(rollout, "_rollout_dir", None)
        if rollout_dir is None:
            return None
        rollout_id = getattr(rollout, "_rollout_name", None) or Path(rollout_dir).name
        return cls(Path(rollout_dir), rollout_id=rollout_id)

    def start_session(
        self,
        *,
        agent_id: str,
        agent_type: str,
        model: str | None,
        driver: str,
        workspace_mode: str = "shared",
        scene: str | None = None,
        scene_index: int | None = None,
        turn_index: int | None = None,
    ) -> str:
        """Create a session record and connect it to the previous finished session."""

        ordinal = self._session_counts.get(agent_id, 0) + 1
        self._session_counts[agent_id] = ordinal
        session_id = f"sess_{_slug(agent_id)}_{ordinal:03d}"
        record = RealAgentSessionRecord(
            session_id=session_id,
            agent_id=agent_id,
            agent_type=agent_type,
            model=model,
            driver=driver,
            workspace_mode=workspace_mode,
            scene=scene,
            scene_index=scene_index,
            turn_index=turn_index,
            started_at=_now_iso(),
        )
        self._sessions.append(record)
        self._events.append(
            {
                "schema_version": "benchflow.real_agents.event.v0",
                "event_id": f"{session_id}:start",
                "rollout_id": self.rollout_id,
                "timestamp": record.started_at,
                "source": "benchflow-runtime",
                "session_id": session_id,
                "agent_id": agent_id,
                "role": agent_id,
                "scene": scene,
                "turn_index": turn_index,
                "relation": "starts",
            }
        )
        if self._last_finished_session_id is not None:
            self._record_handoff(self._last_finished_session_id, session_id, scene=scene)
        self.write_indexes()
        return session_id

    def finish_session(
        self,
        session_id: str | None,
        trajectory: list[dict[str, Any]],
        *,
        error: str | None = None,
    ) -> None:
        """Persist one session's isolated trajectory and update graph artifacts."""

        if session_id is None:
            return
        record = self._find_session(session_id)
        if record is None:
            return
        trajectory = list(trajectory or [])
        session_dir = self.agents_dir / _slug(record.agent_id) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        acp_path = session_dir / "acp.jsonl"
        _write_jsonl(acp_path, trajectory)

        record.ended_at = _now_iso()
        record.trajectory_path = _relative(acp_path, self.rollout_dir)
        record.n_trajectory_events = len(trajectory)
        record.n_tool_calls = sum(
            1
            for event in trajectory
            if isinstance(event, dict) and event.get("type") == "tool_call"
        )
        record.error = error

        for index, event in enumerate(trajectory):
            event_type = event.get("type", "event") if isinstance(event, dict) else "event"
            self._events.append(
                {
                    "schema_version": "benchflow.real_agents.event.v0",
                    "event_id": f"{session_id}:acp:{index}",
                    "rollout_id": self.rollout_id,
                    "timestamp": record.ended_at,
                    "source": "acp",
                    "session_id": session_id,
                    "agent_id": record.agent_id,
                    "role": record.agent_id,
                    "scene": record.scene,
                    "turn_index": record.turn_index,
                    "relation": "agent_event",
                    "acp_event_index": index,
                    "event_type": event_type,
                    "trajectory_path": record.trajectory_path,
                }
            )
        self._events.append(
            {
                "schema_version": "benchflow.real_agents.event.v0",
                "event_id": f"{session_id}:end",
                "rollout_id": self.rollout_id,
                "timestamp": record.ended_at,
                "source": "benchflow-runtime",
                "session_id": session_id,
                "agent_id": record.agent_id,
                "role": record.agent_id,
                "scene": record.scene,
                "turn_index": record.turn_index,
                "relation": "terminates",
                "n_trajectory_events": record.n_trajectory_events,
                "n_tool_calls": record.n_tool_calls,
                "error": error,
            }
        )
        self._last_finished_session_id = session_id
        self.write_indexes()

    def write_indexes(self) -> None:
        """Rewrite index artifacts atomically enough for local readers."""

        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            self.trajectory_dir / "sessions.jsonl",
            [asdict(record) for record in self._sessions],
        )
        _write_jsonl(
            self.trajectory_dir / "handoffs.jsonl",
            [asdict(record) for record in self._handoffs],
        )
        _write_jsonl(self.trajectory_dir / "multiagent_events.jsonl", self._events)
        (self.trajectory_dir / "agent_graph.json").write_text(
            json.dumps(self._graph(), sort_keys=True, indent=2) + "\n"
        )

    def _record_handoff(
        self, from_session_id: str, to_session_id: str, *, scene: str | None
    ) -> None:
        from_record = self._find_session(from_session_id)
        to_record = self._find_session(to_session_id)
        if from_record is None or to_record is None:
            return
        relation = "continues_as" if from_record.agent_id == to_record.agent_id else "handoff_to"
        handoff_id = f"handoff_{len(self._handoffs) + 1:03d}"
        record = RealAgentHandoffRecord(
            handoff_id=handoff_id,
            from_session_id=from_session_id,
            to_session_id=to_session_id,
            from_agent_id=from_record.agent_id,
            to_agent_id=to_record.agent_id,
            relation=relation,
            scene=scene,
            created_at=_now_iso(),
        )
        self._handoffs.append(record)
        from_record.handoff_out.append(handoff_id)
        to_record.handoff_in.append(handoff_id)
        self._events.append(
            {
                "schema_version": "benchflow.real_agents.event.v0",
                "event_id": f"{handoff_id}:edge",
                "rollout_id": self.rollout_id,
                "timestamp": record.created_at,
                "source": "benchflow-runtime",
                "session_id": from_session_id,
                "agent_id": from_record.agent_id,
                "role": from_record.agent_id,
                "scene": scene,
                "relation": relation,
                "handoff_id": handoff_id,
                "handoff_from": from_record.agent_id,
                "handoff_to": to_record.agent_id,
                "related_session_id": to_session_id,
            }
        )

    def _find_session(self, session_id: str) -> RealAgentSessionRecord | None:
        for record in self._sessions:
            if record.session_id == session_id:
                return record
        return None

    def _graph(self) -> dict[str, Any]:
        return {
            "schema_version": "benchflow.real_agents.graph.v0",
            "rollout_id": self.rollout_id,
            "nodes": [
                {
                    "id": record.session_id,
                    "kind": "agent_session",
                    "agent_id": record.agent_id,
                    "agent_type": record.agent_type,
                    "model": record.model,
                    "driver": record.driver,
                    "scene": record.scene,
                    "trajectory_path": record.trajectory_path,
                    "n_trajectory_events": record.n_trajectory_events,
                    "n_tool_calls": record.n_tool_calls,
                }
                for record in self._sessions
            ],
            "edges": [
                {
                    "id": record.handoff_id,
                    "from": record.from_session_id,
                    "to": record.to_session_id,
                    "relation": record.relation,
                    "from_agent_id": record.from_agent_id,
                    "to_agent_id": record.to_agent_id,
                    "scene": record.scene,
                    "artifacts": record.artifacts,
                }
                for record in self._handoffs
            ],
            "metrics": {
                "n_sessions": len(self._sessions),
                "n_handoffs": len(self._handoffs),
                "n_events": len(self._events),
            },
        }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "agent"
