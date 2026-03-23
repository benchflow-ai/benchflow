"""Convert Claude Code stream-json output to ATIF trajectory format.

Groups events into logical steps:
  PROMPT → AGENT (thinking + tool call + tool result) → AGENT → ... → RESULT
"""

import json
from pathlib import Path
from typing import Any

from .atif import (
    Agent,
    ATIFTrajectory,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
)


def parse_stream_json(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def to_atif(
    events: list[dict[str, Any]],
    session_id: str = "",
    initial_prompt: str | None = None,
) -> ATIFTrajectory:
    """Convert Claude Code stream-json events to ATIF trajectory.

    Groups consecutive assistant events into single steps.
    Merges tool_result into the tool call step as an observation.
    """
    system_event = next((e for e in events if e.get("type") == "system"), None)

    agent_name = "claude-code"
    agent_version = ""
    model_name = None
    if system_event:
        agent_version = system_event.get("claude_code_version", "")
        model_name = system_event.get("model")
        session_id = session_id or system_event.get("session_id", "")

    trajectory = ATIFTrajectory(
        session_id=session_id,
        agent=Agent(name=agent_name, version=agent_version, model_name=model_name),
    )

    # Add initial prompt if provided
    step_id = 0
    if initial_prompt:
        step_id += 1
        trajectory.steps.append(
            Step(step_id=step_id, source="user", message=initial_prompt)
        )

    # Group events into logical steps
    # Pattern: assistant events accumulate until a tool_result resets, or text ends
    current_thinking = ""
    current_text = ""
    current_tools: list[ToolCall] = []
    current_model = None
    current_metrics: Metrics | None = None
    current_observation: Observation | None = None

    def flush_step():
        nonlocal step_id, current_thinking, current_text, current_tools
        nonlocal current_model, current_metrics, current_observation
        if not (current_thinking or current_text or current_tools):
            return
        step_id += 1
        trajectory.steps.append(
            Step(
                step_id=step_id,
                source="agent",
                model_name=current_model,
                message=current_text,
                reasoning_content=current_thinking or None,
                tool_calls=current_tools or None,
                observation=current_observation,
                metrics=current_metrics,
            )
        )
        current_thinking = ""
        current_text = ""
        current_tools = []
        current_model = None
        current_metrics = None
        current_observation = None

    for event in events:
        event_type = event.get("type", "")

        if event_type == "assistant":
            msg = event.get("message", {})
            content_blocks = msg.get("content", [])
            usage = msg.get("usage", {})
            model = msg.get("model")

            for block in content_blocks:
                block_type = block.get("type", "")

                if block_type == "thinking":
                    current_thinking += block.get("thinking", "")
                    current_model = model

                elif block_type == "text":
                    # Text after a tool+result = new step
                    if current_observation:
                        flush_step()
                    current_text += block.get("text", "")
                    current_model = model

                elif block_type == "tool_use":
                    # If we had a previous tool+result, flush first
                    if current_observation:
                        flush_step()
                    current_tools.append(
                        ToolCall(
                            tool_call_id=block.get("id", ""),
                            function_name=block.get("name", ""),
                            arguments=block.get("input", {}),
                        )
                    )
                    current_model = model

            if usage:
                current_metrics = Metrics(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )

        elif event_type == "user":
            msg = event.get("message", {})
            content = msg.get("content", "")

            if isinstance(content, list):
                is_tool_result = any(
                    b.get("type") == "tool_result"
                    for b in content
                    if isinstance(b, dict)
                )
                if is_tool_result:
                    # Merge tool result as observation on current step
                    obs_parts = []
                    for block in content:
                        if block.get("type") == "tool_result":
                            raw = str(block.get("content", ""))[:500]
                            # Filter binary
                            printable = sum(
                                1 for c in raw if c.isprintable() or c in "\n\t"
                            )
                            if len(raw) > 20 and printable / len(raw) < 0.7:
                                obs_parts.append("[binary content]")
                            else:
                                obs_parts.append(raw)
                    current_observation = Observation(
                        results=[ObservationResult(content="\n".join(obs_parts))]
                    )
                else:
                    # Real user message (follow-up prompt)
                    flush_step()
                    step_id += 1
                    text = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    trajectory.steps.append(
                        Step(
                            step_id=step_id, source="user", message=text or str(content)
                        )
                    )
            else:
                # Plain text user message
                flush_step()
                step_id += 1
                trajectory.steps.append(
                    Step(step_id=step_id, source="user", message=str(content))
                )

        elif event_type == "result":
            flush_step()
            usage = event.get("usage", {})
            trajectory.final_metrics = Metrics(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cost_usd=event.get("total_cost_usd"),
            )

    # Flush any remaining
    flush_step()
    return trajectory


def convert_trial_dir(
    trial_dir: Path, initial_prompt: str | None = None
) -> ATIFTrajectory | None:
    """Convert turn*.txt files in a trial directory to a single ATIF trajectory."""
    turn_files = sorted(trial_dir.glob("turn*.txt"))
    if not turn_files:
        single = trial_dir / "claude-code.txt"
        if single.exists():
            turn_files = [single]
        else:
            return None

    # Read instruction.md for initial prompt if not provided
    if initial_prompt is None:
        for candidate in [
            trial_dir.parent.parent / "instruction.md",
            trial_dir / "instruction.md",
        ]:
            if candidate.exists():
                initial_prompt = candidate.read_text().strip()
                break

    all_events: list[dict[str, Any]] = []
    session_id = ""
    for tf in turn_files:
        events = parse_stream_json(tf)
        all_events.extend(events)
        for e in events:
            if e.get("type") == "result" and e.get("session_id"):
                session_id = e["session_id"]

    if not all_events:
        return None

    return to_atif(all_events, session_id=session_id, initial_prompt=initial_prompt)


def save_atif(trajectory: ATIFTrajectory, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trajectory.to_json_dict(), indent=2, default=str))
