"""The solver's ``trajectory-1`` record, as the judges read it.

task.md's judges read the solver's trajectory as ``/judge/trajectory.jsonl``,
the view of its ``trajectory-1`` record (docs/runtime/episodes.md,
``trajectory-1``; docs/runtime/judging.md, "The trajectory view"). BenchFlow
builds the record from its own trial records:

- A **scripted seat** (the oracle, a control, or ``nop``) is staged blind
  (docs/runtime/judging.md, "Blind staging"): one step whose one ``bash`` call
  runs ``bash /opt/seat/run.sh``, with the result ``exit <code>`` and the
  output, capped as ``shell@1`` caps it. The judge sees what the seat did and
  printed, never its source or where it came from.
- An **agent**'s record is built from BenchFlow's ACP trajectory
  (``acp_trajectory.jsonl``): each run of agent text and thought, and the tool
  calls that follow it, is one step, and each later user message (a stage's
  prompt) is a runtime step. ACP does not mark where one model response ends,
  so these step boundaries approximate the spec's "one step per model
  response"; the record says so in ``x-benchflow-source``.
"""

from __future__ import annotations

import json
from typing import Any

SEAT_COMMAND = "bash /opt/seat/run.sh"
SHELL_CAP = 16_384
SHELL_HALF = 8_192
SCRIPTED = ("oracle", "nop")


def shell_output(data: bytes) -> tuple[str, bool]:
    """Output as ``shell@1`` keeps it: past 16,384 bytes, the first and last 8,192
    bytes with ``[... <m> bytes omitted ...]`` between; each part decoded with
    replacement. Returns (text, truncated)."""
    if len(data) <= SHELL_CAP:
        return data.decode("utf-8", "replace"), False
    omitted = len(data) - 2 * SHELL_HALF
    head = data[:SHELL_HALF].decode("utf-8", "replace")
    tail = data[-SHELL_HALF:].decode("utf-8", "replace")
    return f"{head}\n[... {omitted} bytes omitted ...]\n{tail}", True


def scripted_seat_record(exit_code: int, output: bytes) -> dict[str, Any]:
    """The one-step record of a scripted seat (blind staging)."""
    text, truncated = shell_output(output)
    return {
        "$schema": "https://task.md/schema/trajectory-1.json",
        "seat": "solver",
        "surface": "shell@1",
        "threads": [{"id": 1, "kind": "agent", "parent": None}],
        "steps": [
            {
                "id": 1,
                "source": "model",
                "thread": 1,
                "parent": None,
                "text": "",
                "reasoning": None,
                "tool_calls": [
                    {
                        "id": "seat-1",
                        "name": "bash",
                        "arguments": json.dumps({"command": SEAT_COMMAND}),
                        "result": {
                            "text": f"exit {exit_code}\n{text}",
                            "origin": "sandbox",
                            "exit": exit_code,
                            "timed_out": False,
                            "truncated": truncated,
                            "output_bytes": len(output),
                        },
                    }
                ],
                "world_time": None,
            }
        ],
        "x-benchflow-source": "scripted seat, staged blind",
    }


def _tool_text(event: dict[str, Any]) -> str:
    """A tool call's result text from ACP content blocks (text parts, joined)."""
    parts: list[str] = []
    for block in event.get("content") or []:
        if isinstance(block, dict):
            inner = (
                block.get("content")
                if isinstance(block.get("content"), dict)
                else block
            )
            text = inner.get("text") if isinstance(inner, dict) else None
            if isinstance(text, str):
                parts.append(text)
    if not parts and event.get("raw_output") is not None:
        raw = event["raw_output"]
        parts.append(raw if isinstance(raw, str) else json.dumps(raw, default=str))
    return "\n".join(parts)


def acp_record(events: list[dict[str, Any]]) -> dict[str, Any]:
    """An agent's ``trajectory-1`` record from BenchFlow's ACP trajectory events."""
    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    seen_user = False

    def new_step(source: str = "model") -> dict[str, Any]:
        step: dict[str, Any] = {
            "id": len(steps) + 1,
            "source": source,
            "thread": 1,
            "parent": None,
            "text": "",
            "reasoning": None,
            "tool_calls": [],
            "world_time": None,
        }
        steps.append(step)
        return step

    for event in events:
        kind = event.get("type")
        if kind == "user_message":
            # The instruction is not a step; a later user message is a stage prompt the runtime delivered.
            if seen_user:
                step = new_step("runtime")
                step["text"] = str(event.get("text") or "")
                step["delivered_as"] = "user"
            seen_user = True
            current = None
        elif kind in ("agent_message", "agent_thought"):
            if current is None or current["tool_calls"]:
                current = new_step()
            text = str(event.get("text") or "")
            if kind == "agent_message":
                current["text"] += text
            else:
                current["reasoning"] = (current["reasoning"] or "") + text
        elif kind == "tool_call":
            if current is None:
                current = new_step()
            raw_input = event.get("raw_input")
            arguments = (
                raw_input
                if isinstance(raw_input, dict)
                else {"title": event.get("title")}
            )
            current["tool_calls"].append(
                {
                    "id": str(
                        event.get("tool_call_id")
                        or f"call-{len(steps)}-{len(current['tool_calls'])}"
                    ),
                    "name": str(event.get("tool_name") or event.get("kind") or "tool"),
                    "arguments": json.dumps(arguments, default=str),
                    "result": {
                        "text": _tool_text(event),
                        "origin": "sandbox",
                        "status": event.get("status"),
                    },
                }
            )
    return {
        "$schema": "https://task.md/schema/trajectory-1.json",
        "seat": "solver",
        "surface": "acp",
        "threads": [{"id": 1, "kind": "agent", "parent": None}],
        "steps": steps,
        "x-benchflow-source": "BenchFlow ACP trajectory; step boundaries approximate one model response",
    }


def solver_record(
    trajectory: list[dict[str, Any]], oracle_output: bytes | None = None
) -> dict[str, Any]:
    """The solver's record from a trial's trajectory (``acp_trajectory.jsonl`` events).

    An oracle trial's trajectory is one ``{"type": "oracle", "return_code", "stdout"}``
    event and a ``nop`` trial's one ``{"type": "nop"}`` event: both are scripted
    seats. ``oracle_output`` is the oracle's full output when the trial kept it
    (``agent/oracle.txt``); otherwise the event's ``stdout`` tail is used.
    """
    if trajectory and trajectory[0].get("type") == "oracle":
        event = trajectory[0]
        code = event.get("return_code")
        output = (
            oracle_output
            if oracle_output is not None
            else str(event.get("stdout") or "").encode()
        )
        return scripted_seat_record(int(code) if isinstance(code, int) else 0, output)
    if trajectory and trajectory[0].get("type") == "nop":
        return scripted_seat_record(0, b"")
    return acp_record(trajectory)


def is_scripted(trajectory: list[dict[str, Any]]) -> bool:
    return bool(trajectory) and trajectory[0].get("type") in SCRIPTED


def load_events(data: bytes) -> list[dict[str, Any]]:
    """Events of an ``acp_trajectory.jsonl`` file; malformed lines are skipped."""
    events = []
    for line in data.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events
