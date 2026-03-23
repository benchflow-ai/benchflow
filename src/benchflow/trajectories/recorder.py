"""Trajectory recorder — captures ACP session updates into ATIF format."""

from datetime import datetime, timezone

from benchflow.acp.session import ACPSession, ToolCallRecord
from benchflow.acp.types import ToolCallStatus

from .atif import (
    Agent,
    ATIFTrajectory,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
)


class TrajectoryRecorder:
    """Converts an ACPSession's state into an ATIF Trajectory."""

    def __init__(
        self, session_id: str, agent_name: str = "unknown", agent_version: str = ""
    ):
        self._trajectory = ATIFTrajectory(
            session_id=session_id,
            agent=Agent(name=agent_name, version=agent_version),
        )
        self._step_count = 0

    @property
    def trajectory(self) -> ATIFTrajectory:
        return self._trajectory

    def record_user_prompt(self, text: str) -> None:
        self._step_count += 1
        self._trajectory.steps.append(
            Step(
                step_id=self._step_count,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=text,
            )
        )

    def record_agent_response(self, session: ACPSession) -> None:
        """Record the agent's response from the accumulated session state."""
        self._step_count += 1

        tool_calls = [
            ToolCall(
                tool_call_id=tc.tool_call_id,
                function_name=tc.kind,
                arguments={"title": tc.title},
            )
            for tc in session.tool_calls
        ]

        observation = None
        completed_calls = [
            tc for tc in session.tool_calls if tc.status == ToolCallStatus.COMPLETED
        ]
        if completed_calls:
            observation = Observation(
                results=[
                    ObservationResult(
                        source_call_id=tc.tool_call_id,
                        content=_extract_tool_output(tc),
                    )
                    for tc in completed_calls
                ]
            )

        self._trajectory.steps.append(
            Step(
                step_id=self._step_count,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                message=session.full_message,
                reasoning_content=session.full_thought or None,
                tool_calls=tool_calls or None,
                observation=observation,
            )
        )

    def finalize(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> ATIFTrajectory:
        if any(v is not None for v in (input_tokens, output_tokens, cost_usd)):
            self._trajectory.final_metrics = Metrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        return self._trajectory


def _extract_tool_output(tc: ToolCallRecord) -> str:
    """Extract text content from a tool call's content list."""
    parts = []
    for item in tc.content:
        if isinstance(item, dict):
            content = item.get("content", {})
            if isinstance(content, dict) and content.get("type") == "text":
                parts.append(content.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
    return "\n".join(parts) if parts else ""
