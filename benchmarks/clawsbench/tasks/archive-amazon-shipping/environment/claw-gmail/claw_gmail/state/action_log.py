"""Action logging middleware for audit trail."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field


@dataclass
class ActionLogEntry:
    timestamp: str
    method: str
    path: str
    user_id: str
    request_body: dict | None = None
    response_status: int = 200


class ActionLog:
    """In-memory action log for tracking all API calls."""

    def __init__(self):
        self._entries: list[ActionLogEntry] = []

    def record(
        self,
        method: str,
        path: str,
        user_id: str = "",
        request_body: dict | None = None,
        response_status: int = 200,
    ):
        self._entries.append(ActionLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            method=method,
            path=path,
            user_id=user_id,
            request_body=request_body,
            response_status=response_status,
        ))

    def get_entries(self) -> list[dict]:
        return [
            {
                "timestamp": e.timestamp,
                "method": e.method,
                "path": e.path,
                "user_id": e.user_id,
                "request_body": e.request_body,
                "response_status": e.response_status,
            }
            for e in self._entries
        ]

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)


# Global singleton
action_log = ActionLog()
