"""Trajectory types — raw LLM API request/response pairs captured by proxy."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """A single request to an LLM API, captured by the proxy."""

    timestamp: datetime = Field(default_factory=datetime.now)
    method: str = "POST"
    path: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """A single response from an LLM API, captured by the proxy."""

    timestamp: datetime = Field(default_factory=datetime.now)
    status_code: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)


class LLMExchange(BaseModel):
    """A request-response pair."""

    request: LLMRequest
    response: LLMResponse
    duration_ms: float = 0.0


class Trajectory(BaseModel):
    """Raw trajectory: ordered list of LLM API exchanges captured by proxy."""

    session_id: str
    agent_name: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = None
    exchanges: list[LLMExchange] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_input_tokens(self) -> int:
        total = 0
        for ex in self.exchanges:
            usage = ex.response.body.get("usage", {})
            total += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        return total

    @property
    def total_output_tokens(self) -> int:
        total = 0
        for ex in self.exchanges:
            usage = ex.response.body.get("usage", {})
            total += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        return total

    @property
    def total_cost_usd(self) -> float | None:
        """Extract cost if the API returns it (Anthropic does not, OpenAI does not)."""
        return self.metadata.get("cost_usd")

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Extract all messages from all exchanges (the conversation history)."""
        msgs: list[dict[str, Any]] = []
        for ex in self.exchanges:
            # Request messages
            req_msgs = ex.request.body.get("messages", [])
            if req_msgs and (not msgs or req_msgs != msgs):
                msgs = list(req_msgs)  # latest request has full history
            # Response message
            resp_content = ex.response.body.get("content", [])
            if resp_content:
                msgs.append({"role": "assistant", "content": resp_content})
            # OpenAI format
            choices = ex.response.body.get("choices", [])
            if choices and "message" in choices[0]:
                msgs.append(choices[0]["message"])
        return msgs

    def to_jsonl(self, *, redact_keys: bool = True) -> str:
        """Export as JSONL (one exchange per line)."""
        import json
        import re

        lines = []
        for ex in self.exchanges:
            data = ex.model_dump(mode="json")
            raw = json.dumps(data, default=str)
            if redact_keys:
                raw = re.sub(
                    r"(sk-ant-[a-zA-Z0-9_-]{10})[a-zA-Z0-9_-]+",
                    r"\1***REDACTED***",
                    raw,
                )
                raw = re.sub(
                    r"(sk-[a-zA-Z0-9]{10})[a-zA-Z0-9]+",
                    r"\1***REDACTED***",
                    raw,
                )
            lines.append(raw)
        return "\n".join(lines)
