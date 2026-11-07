"""JSON Schemas for the rows ``bench train convert --format branch-tree`` writes.

``branch_child`` rows (``children.jsonl``) and ``branch_pair`` rows
(``pairs.jsonl``), ``schema_version`` 2. The files are committed under
``docs/reference/schemas/`` and a test fails when they are stale
(regenerate with ``python -m benchflow.trajectories.branch_row_schema
docs/reference/schemas``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_ID_BASE = "https://benchflow.ai/schemas"

_NUM = {"type": "number"}
_OPT_NUM = {"type": ["number", "null"]}
_OPT_STR = {"type": ["string", "null"]}
_OPT_INT = {"type": ["integer", "null"]}

_DEFS: dict[str, Any] = {
    "tool_call": {
        "type": "object",
        "required": ["id", "type", "function"],
        "properties": {
            "id": {"type": "string"},
            "type": {"const": "function"},
            "function": {
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The agent's tool name (Write, Bash, …); "
                        "inferred from the call when the agent did not send one "
                        "(see tool_names).",
                    },
                    "arguments": {
                        "type": "string",
                        "description": "The call's input as a JSON string.",
                    },
                },
            },
        },
    },
    "message": {
        "type": "object",
        "required": ["role"],
        "properties": {
            "role": {"enum": ["system", "user", "assistant", "tool"]},
            "content": {"type": "string"},
            "reasoning_content": {
                "type": "string",
                "description": "The agent's visible thinking for this turn.",
            },
            "tool_calls": {"type": "array", "items": {"$ref": "#/$defs/tool_call"}},
            "tool_call_id": {
                "type": "string",
                "description": "On role tool: the id of the call it answers.",
            },
        },
    },
    "messages": {"type": "array", "items": {"$ref": "#/$defs/message"}},
    "tool": {
        "type": "object",
        "required": ["type", "function"],
        "properties": {
            "type": {"const": "function"},
            "function": {
                "type": "object",
                "required": ["name", "parameters"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {"type": "object"},
                },
            },
        },
        "description": "Inferred from the recorded calls (argument names only); "
        "the agent's own tool definitions are not recorded.",
    },
    "tools": {"type": "array", "items": {"$ref": "#/$defs/tool"}},
    "session": {
        "type": "string",
        "description": "fork snapshot agent_session: fresh, resumed, …",
    },
}

_COMMON: dict[str, Any] = {
    "schema_version": {"const": 2},
    "task": _OPT_STR,
    "trial": {"type": "string", "description": "Trial folder name."},
    "agent": _OPT_STR,
    "model": _OPT_STR,
    "fork_id": {"type": "string"},
    "parent_node": _OPT_STR,
    "depth": {
        "type": "integer",
        "minimum": 1,
        "description": "1 for a fork of the trial, 2 for a fork inside a child…",
    },
    "session": {"$ref": "#/$defs/session"},
    "prefix_complete": {
        "type": "boolean",
        "description": "False when the trial started from a kept checkpoint whose "
        "source conversation could not be found, so the prefix lacks it.",
    },
    "tools": {"$ref": "#/$defs/tools"},
}

CHILD_ROW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"{SCHEMA_ID_BASE}/benchflow-branch-child-row.v2.schema.json",
    "title": "BenchFlow branch-tree child row (children.jsonl)",
    "description": "One child of one fork: the shared prefix, this child's "
    "continuation and its reward, value and advantage.",
    "type": "object",
    "required": [
        "kind",
        "schema_version",
        "trial",
        "fork_id",
        "depth",
        "value",
        "child",
        "advantage",
        "session",
        "prefix_complete",
        "prefix",
        "continuation",
        "messages",
        "tools",
    ],
    "properties": {
        "kind": {"const": "branch_child"},
        **_COMMON,
        "fork_kind": {"enum": ["fork", "retry"]},
        "parent_child": {
            "type": ["string", "null"],
            "description": "Node id of the child whose fork this is (nested), "
            "else null.",
        },
        "fork_status": _OPT_STR,
        "value": {**_OPT_NUM, "description": "Mean reward of the fork's children."},
        "value_stderr": _OPT_NUM,
        "siblings": {"type": "integer", "minimum": 1},
        "child": {
            "type": "object",
            "required": ["node_id", "reward"],
            "properties": {
                "index": _OPT_INT,
                "node_id": {"type": "string"},
                "label": _OPT_STR,
                "requested": _OPT_STR,
                "status": _OPT_STR,
                "reward": {**_OPT_NUM, "description": "null when unscored."},
                "reward_source": _OPT_STR,
                "path": _OPT_STR,
            },
        },
        "advantage": {**_OPT_NUM, "description": "reward - value."},
        "prefix_source": {
            "type": ["object", "null"],
            "description": "For a trial started from a kept checkpoint: the "
            "source trial, checkpoint and number of events prepended.",
            "properties": {
                "trial": _OPT_STR,
                "checkpoint": _OPT_STR,
                "events": _OPT_INT,
            },
        },
        "prefix": {"$ref": "#/$defs/messages"},
        "continuation": {"$ref": "#/$defs/messages"},
        "messages": {
            "$ref": "#/$defs/messages",
            "description": "prefix + continuation, ready for apply_chat_template.",
        },
        "tool_names": {
            "enum": ["agent", "inferred", "mixed", None],
            "description": "Whether tool names came from the agent or were "
            "inferred from the calls.",
        },
    },
    "$defs": _DEFS,
}

PAIR_ROW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"{SCHEMA_ID_BASE}/benchflow-branch-pair-row.v2.schema.json",
    "title": "BenchFlow branch-tree pair row (pairs.jsonl)",
    "description": "Two siblings of one fork with different rewards, in "
    "prompt/chosen/rejected form for preference training.",
    "type": "object",
    "required": [
        "kind",
        "schema_version",
        "trial",
        "fork_id",
        "same_request",
        "prompt",
        "chosen",
        "rejected",
        "chosen_reward",
        "rejected_reward",
        "margin",
        "tools",
    ],
    "properties": {
        "kind": {"const": "branch_pair"},
        **_COMMON,
        "same_request": {
            "type": "boolean",
            "description": "Both siblings were asked the same thing; only these "
            "are exported unless --pairs-any-request is given.",
        },
        "prompt": {
            "$ref": "#/$defs/messages",
            "description": "The shared prefix plus, when same_request, the "
            "shared user turn.",
        },
        "chosen": {"$ref": "#/$defs/messages"},
        "rejected": {"$ref": "#/$defs/messages"},
        "chosen_reward": _NUM,
        "rejected_reward": _NUM,
        "chosen_label": _OPT_STR,
        "rejected_label": _OPT_STR,
        "margin": {"type": "number", "exclusiveMinimum": 0},
    },
    "$defs": _DEFS,
}

SCHEMAS = {
    "benchflow-branch-child-row.v2.schema.json": CHILD_ROW_SCHEMA,
    "benchflow-branch-pair-row.v2.schema.json": PAIR_ROW_SCHEMA,
}


def write_schemas(directory: str | Path) -> list[Path]:
    """Write both row schemas into ``directory`` and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in SCHEMAS.items():
        path = directory / name
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for written in write_schemas(sys.argv[1] if len(sys.argv) > 1 else "."):
        print(written)
