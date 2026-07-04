"""On-disk session store for `bench agent run` — the resume state.

One directory per session under the store root (default
``~/.benchflow/agent-sessions``), holding a single ``meta.json``. The record is
the minimum a later invocation needs to resume: which agent/model, the cwd the
conversation is scoped to (claude-style), the agent's real ACP ``sessionId``,
and the capabilities it advertised at ``initialize`` (``loadSession`` gates
resume).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".benchflow" / "agent-sessions"


@dataclass
class SessionRecord:
    session_id: str
    agent: str
    model: str
    cwd: str
    created: float
    last_used: float
    acp_session_id: str = ""
    capabilities: dict = field(default_factory=dict)


class SessionStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_ROOT

    def _meta(self, session_id: str) -> Path:
        return self.root / session_id / "meta.json"

    def create(self, *, agent: str, model: str, cwd: str) -> SessionRecord:
        now = time.time()
        rec = SessionRecord(
            session_id=uuid.uuid4().hex[:12],
            agent=agent,
            model=model,
            cwd=cwd,
            created=now,
            last_used=now,
        )
        self._write(rec)
        return rec

    def load(self, session_id: str) -> SessionRecord:
        meta = self._meta(session_id)
        if not meta.exists():
            raise KeyError(f"no agent session {session_id!r} under {self.root}")
        return SessionRecord(**json.loads(meta.read_text()))

    def update(self, session_id: str, **fields) -> SessionRecord:
        rec = self.load(session_id)
        for k, v in fields.items():
            if not hasattr(rec, k):
                raise AttributeError(f"SessionRecord has no field {k!r}")
            setattr(rec, k, v)
        rec.last_used = time.time()
        self._write(rec)
        return rec

    def latest_for_cwd(self, cwd: str) -> SessionRecord | None:
        if not self.root.exists():
            return None
        matches = [
            rec
            for meta in self.root.glob("*/meta.json")
            if (rec := SessionRecord(**json.loads(meta.read_text()))).cwd == cwd
        ]
        return max(matches, key=lambda r: r.last_used, default=None)

    def _write(self, rec: SessionRecord) -> None:
        meta = self._meta(rec.session_id)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(asdict(rec), indent=2))
