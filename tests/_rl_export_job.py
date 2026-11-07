"""A recorded rollout for the verl, SkyRL and OpenEnv export tests.

Shaped like the 30-call Claude Code run on MiMo v2.6 flash these exports were checked on: one
conversation through the proxy, each request carrying the turns before it.

    call 0  Write, cut at the output cap (finish_reason "length", arguments cut short)
    call 1  Bash whose input the provider's adapter could not read ({"__unparsedToolInput": ...})
    call 2  Bash {command, description}
    call 3  Write {file_path, content}: a second argument shape in the same conversation
    call 4  the final answer

BenchFlow trains calls 2 to 4. Calls 0 and 1 stay in the conversation as context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

WRITE = {
    "type": "function",
    "function": {
        "name": "Write",
        "description": "Write a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
}
BASH = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}
TOOLS = [WRITE, BASH]
BASH_ARGS = {
    "command": "touch /app/essay.txt && echo ok",
    "description": "Create the essay file",
}
WRITE_ARGS = {
    "file_path": "/app/essay.txt",
    "content": "The printing press multiplied thought.",
}
ANSWER = "The essay is in /app/essay.txt."
UNPARSED = {
    "__unparsedToolInput": json.dumps(
        {"raw": '{"command": "touch /app/essay.txt"}', "len": 33}
    )
}

# Message index of each assistant turn in the whole conversation:
# user 0, then (assistant, tool) pairs from 1, and the final answer at 9.
TRAINED = [5, 7, 9]
CONTEXT = {1: "output_cap", 3: "unparsed_tool_input"}


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


CALLS: list[tuple[str, dict[str, Any], str | None]] = [
    (
        "length",
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _call(
                    "call_0",
                    "Write",
                    '{"file_path": "/app/essay.txt", "content": "The prin',
                )
            ],
        },
        "<tool_use_error>Write was called with input that could not be parsed as JSON</tool_use_error>",
    ),
    (
        "tool_calls",
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("call_1", "Bash", json.dumps(UNPARSED))],
        },
        "<tool_use_error>An unexpected parameter `__unparsedToolInput` was provided</tool_use_error>",
    ),
    (
        "tool_calls",
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("call_2", "Bash", json.dumps(BASH_ARGS))],
        },
        "ok",
    ),
    (
        "tool_calls",
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("call_3", "Write", json.dumps(WRITE_ARGS))],
        },
        "File written",
    ),
    ("stop", {"role": "assistant", "content": ANSWER}, None),
]


def write_job(
    root: Path,
    *,
    reward: float | None = 1.0,
    trial: str = "essay__abc123",
    order: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> Path:
    """Write the rollout under ``root/jobs/run/<trial>``; return the jobs directory.

    ``order`` picks and orders the calls of CALLS; the final answer belongs last.
    """
    rollout = root / "jobs" / "run" / trial
    (rollout / "trajectory").mkdir(parents=True)
    result: dict[str, Any] = {
        "task_name": "essay",
        "agent": "claude-agent-acp",
        "model": "mimo",
    }
    if reward is not None:
        result["rewards"] = {"reward": reward}
    (rollout / "result.json").write_text(json.dumps(result))
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Write a 1500-word essay to /app/essay.txt."}
    ]
    lines = []
    for finish_reason, reply, tool_result in [CALLS[i] for i in order]:
        lines.append(
            {
                "request": {
                    "body": {"model": "mimo", "messages": list(history), "tools": TOOLS}
                },
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"finish_reason": finish_reason, "message": reply}]
                    },
                },
                "duration_ms": 1,
            }
        )
        history.append(reply)
        if tool_result is not None:
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": reply["tool_calls"][0]["id"],
                    "content": tool_result,
                }
            )
    (rollout / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    return root / "jobs"
