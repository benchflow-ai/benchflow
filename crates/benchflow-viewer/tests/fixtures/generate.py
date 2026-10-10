#!/usr/bin/env python3
"""Regenerate the viewer's synthetic fixtures and golden outputs from BenchFlow 0.8.

The Rust viewer must produce exactly what the 0.8 Python viewer produces, so
this script writes edge-case trial folders and records the Python viewer's
output for every trial under ``jobs/``. Run it from the repository root with
the 0.8 sources (``src/``) on the path and BenchFlow's dependencies installed:

    PYTHONPATH=src python crates/benchflow-viewer/tests/fixtures/generate.py

It rewrites ``jobs/synthetic/``, ``jobs/branched/`` and ``golden/``. The real
jobs copied from disk (``jobs/oracle-hello``, ``jobs/test-agent``,
``jobs/branch-tree``) are inputs and stay as they are. ``tests/golden.rs``
compares the Rust output with ``golden/``.

Absolute paths in outputs are written as ``<root>`` (the ``jobs`` folder).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

HERE = Path(__file__).resolve().parent
ROOT = HERE / "jobs"
SYN = ROOT / "synthetic"
BRANCHED = ROOT / "branched"
GOLDEN = HERE / "golden"


# ── synthetic trials ──────────────────────────────────────────────────────


def _write(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(json.dumps(content), encoding="utf-8")


def trial(
    name: str,
    events: list[dict] | None = None,
    *,
    raw: str | bytes | None = None,
    result: object = None,
    timing: object = None,
    prompts: object = None,
    files: dict[str, object] | None = None,
) -> Path:
    """One trial folder under jobs/synthetic: trajectory plus sidecars."""
    d = SYN / name
    traj = d / "trajectory" / "acp_trajectory.jsonl"
    if raw is not None:
        _write(traj, raw)
    else:
        _write(traj, "\n".join(json.dumps(e) for e in events or []))
    if result is not None:
        _write(d / "result.json", result)
    if timing is not None:
        _write(d / "timing.json", timing)
    if prompts is not None:
        _write(d / "prompts.json", prompts)
    for rel, content in (files or {}).items():
        _write(d / rel, content)
    return d


def say(text, parent=None, **extra):
    event = {"type": "agent_message", "text": text, **extra}
    if parent is not None:
        event["parent_tool_call_id"] = parent
    return event


def tool(call_id, parent=None, **extra):
    event = {
        "type": "tool_call",
        "tool_call_id": call_id,
        "kind": "think",
        "title": "Task",
        "status": "completed",
        "content": [],
        **extra,
    }
    if parent is not None:
        event["parent_tool_call_id"] = parent
    return event


def result_doc(task="demo", agent="claude-agent-acp", model="sonnet-x", reward=1.0, **extra):
    out = {
        "task_name": task,
        "agent_name": agent,
        "model": model,
        "rewards": {"reward": reward} if reward is not None else None,
        "skill_mode": "no-skill",
        "timing": {"agent_execution": 10.0, "total": 12.0},
        "agent_result": {"total_tokens": 1234, "cost_usd": 0.05},
        "n_tool_calls": 1,
    }
    out.update(extra)
    return out


def nested_array(depth: int) -> str:
    return "[" * depth + "0" + "]" * depth


def build_synthetic() -> None:
    # Every event variant the renderer knows, with a realistic result.json.
    trial(
        "events-all__00000001",
        [
            {"type": "user_message", "text": "do it", "ts": "2026-08-15T09:41:03+00:00"},
            {"type": "agent_thought", "text": "hmm", "ts": "2026-08-15T09:41:04.250000+00:00"},
            {
                "type": "tool_call",
                "tool_call_id": "c1",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "content": [{"type": "content", "content": {"type": "text", "text": "out"}}],
                "started_at": "2026-08-15T09:41:05+00:00",
                "finished_at": "2026-08-15T09:41:07.500000+00:00",
            },
            {"type": "agent_message", "text": "done", "ts": "2026-08-15T09:41:08Z"},
            {
                "type": "agent_timeout",
                "reason": "wall_clock_timeout",
                "timeout_sec": 5.0,
                "pending_tool_call_ids": ["c1", 7, None],
                "terminal_trajectory_complete": False,
            },
            {
                "type": "notice",
                "severity": "warning",
                "title": "Model fallback",
                "description": "gpt-x is unavailable; using gpt-y",
            },
            {"type": "notice", "title": "Plain notice"},
            {"type": "future_thing", "x": 1, "nested": {"a": [1, 2.5, None, True]}},
            {"type": "plan", "entries": [{"content": "step", "status": "pending"}]},
        ],
        result={
            "task_name": "events-all",
            "rollout_name": "events-all__00000001",
            "rewards": {"reward": 0.75},
            "agent": "claude-agent-acp",
            "agent_name": "@agentclientprotocol/claude-agent-acp",
            "model": "claude-sonnet-4-6",
            "skill_mode": "with-skill",
            "n_tool_calls": 1,
            "n_prompts": 1,
            "agent_result": {
                "n_tool_calls": 1,
                "n_skill_invocations": 0,
                "n_prompts": 1,
                "n_input_tokens": 120,
                "n_output_tokens": 45,
                "n_cache_read_tokens": 3000,
                "n_cache_creation_tokens": 200,
                "total_tokens": 3365,
                "cost_usd": 0.0123,
                "usage_source": "agent_native_acp",
                "price_source": "litellm",
                "usage_details": {"thought_tokens": 12},
            },
            "error": None,
            "verifier_error": None,
            "partial_trajectory": False,
            "trajectory_source": "acp",
            "started_at": "2026-08-15 09:41:00.000000",
            "finished_at": "2026-08-15 09:42:00.000000",
            "timing": {"environment_setup": 4.0, "agent_execution": 7.0, "verifier": 2.0, "total": 60.0},
        },
        timing={"environment_setup": 4.0, "agent_setup": 1, "agent_execution": 7.25, "verifier": 2.0, "total": 60.5},
        prompts=["do it"],
        files={
            "verifier/reward.txt": "0.75\n",
            "verifier/test-stdout.txt": "3 passed, 1 failed\n",
            "verifier/test-stderr.txt": "warning: deprecated\n",
            "verifier/ctrf.json": {
                "results": {
                    "tests": [
                        {"name": "test_a", "status": "passed", "duration": 12},
                        {"name": "test_b", "status": "FAILED ", "duration": 3.5},
                        {"name": "test_c", "status": "skipped", "duration": "7"},
                        {"name": "test_d", "status": "pending"},
                        {"name": "test_e", "status": "other", "duration": None},
                        {"status": "passed"},
                    ]
                }
            },
        },
    )

    trial("unknown-event__00000001", [{"type": "future_thing", "x": 1}])

    trial(
        "tool-content__00000001",
        [
            {
                "type": "tool_call",
                "tool_call_id": "flat",
                "kind": "read",
                "title": "Read a.py",
                "status": "completed",
                "content": [
                    {"text": "flat"},
                    {"type": "content", "content": {"type": "text", "text": "nested"}},
                    {"type": "content", "content": {"type": "image", "data": "abc"}},
                    {"type": "diff", "path": "a.py", "oldText": "x=1", "newText": "x=2"},
                    {"type": "diff", "path": "b.py", "oldText": None, "newText": "new"},
                    {"type": "terminal", "terminalId": "term-1"},
                    {"type": "resource_link", "uri": "file:///a.py", "name": "a.py"},
                    "bare string",
                    42,
                    None,
                    [1, "two"],
                    {"text": ""},
                    {"text": {"deep": [1, 2]}},
                ],
            },
            {
                "type": "tool_call",
                "tool_call_id": "exec-list",
                "kind": "execute",
                "title": "Run tests",
                "status": "failed",
                "content": [{"type": "terminal", "terminalId": "t2"}],
                "raw_input": {"command": ["pytest", "-q", 3, None, True, 1.5, {"k": "v"}, ["x"]], "cwd": "/app"},
                "raw_output": {"formatted_output": "1 failed\n\n", "exit_code": 1},
            },
            {
                "type": "tool_call",
                "tool_call_id": "exec-str",
                "kind": "execute",
                "title": "bash",
                "status": "completed",
                "raw_input": {"command": "echo hi"},
                "raw_output": {"formatted_output": "hi\n", "exit_code": 0},
            },
            {
                "type": "tool_call",
                "tool_call_id": "exec-false",
                "kind": "execute",
                "title": "bash",
                "status": "completed",
                "raw_input": {"command": "true"},
                "raw_output": {"formatted_output": "", "exit_code": False},
            },
            {
                "type": "tool_call",
                "tool_call_id": "exec-strcode",
                "kind": "execute",
                "title": "bash",
                "status": "completed",
                "raw_input": {"command": "false"},
                "raw_output": {"formatted_output": "nope  ", "exit_code": "2"},
            },
            {
                "type": "tool_call",
                "tool_call_id": "exec-float-code",
                "kind": "execute",
                "title": "bash",
                "status": "completed",
                "raw_input": {"command": "x"},
                "raw_output": {"formatted_output": "out", "exit_code": 0.0},
            },
            {
                "type": "tool_call",
                "tool_call_id": "raw-other",
                "kind": "fetch",
                "title": "Fetch https://example.com",
                "status": "completed",
                "raw_input": {"url": "https://example.com", "headers": {"accept": "*/*"}},
                "raw_output": "<html>…</html>",
            },
            {
                "type": "tool_call",
                "tool_call_id": "raw-dict-out",
                "kind": "search",
                "title": "grep foo",
                "status": "completed",
                "raw_input": None,
                "raw_output": {"matches": 3, "files": ["a", "b"]},
            },
            {
                "type": "tool_call",
                "tool_call_id": "exec-no-command",
                "kind": "execute",
                "title": "bash",
                "status": "completed",
                "raw_input": {"cmd": "ls"},
            },
            {"type": "tool_call", "kind": None, "title": None, "status": None},
            {"type": "tool_call", "tool_call_id": 17, "kind": 3, "title": 4.5, "status": "completed"},
            {"type": "tool_call", "tool_call_id": "", "kind": "", "title": "", "content": "not-a-list"},
        ],
    )

    def call(call_id, kind, title, **extra):
        return {"type": "tool_call", "tool_call_id": call_id, "kind": kind, "title": title, "status": "completed", **extra}

    diff = {"type": "diff", "path": "src/a.py", "oldText": "x", "newText": "y"}
    trial(
        "provisional-titles__00000001",
        [
            call(
                "write",
                "edit",
                "Preparing file…",
                raw_input={"file_path": "/app/hello.txt", "content": "Hello, world!"},
                content=[{**diff, "path": "/app/hello.txt", "oldText": None}],
            ),
            call("edit", "edit", "Edit", content=[diff, diff]),
            call("read", "read", "Read File", raw_input={"file_path": "/app/c.txt"}),
            call("read-bare", "read", "Read File"),
            call("two-files", "edit", "Edit", content=[diff, {**diff, "path": "b"}]),
            call("final", "edit", "Write notes.md", raw_input={"file_path": "x"}),
            call("padded", "edit", "  Write  ", raw_input={"file_path": "/p"}),
            call("empty-path", "edit", "Edit", raw_input={"file_path": ""}, content=[diff]),
            call("nonstr-path", "edit", "Edit", raw_input={"file_path": 5}, content=[{**diff, "path": 7}]),
            call("diff-none-path", "edit", "Edit", content=[{"type": "diff", "oldText": "a"}]),
            call("wrong-kind", "read", "Edit", raw_input={"file_path": "/p"}),
        ],
    )

    trial(
        "tool-hues__00000001",
        [
            call("c1", "other", "ToolSearch"),
            call("c2", "delete", "remove temp dir"),
            call("c3", "mystery", "??"),
            call("c4", " Execute ", "run"),
            call("c5", "agent", "spawn"),
            call("c6", "switch_mode", "Switch to plan mode"),
            call("c7", "other", "browser_navigate"),
            call("c8", "other", "NotebookEdit"),
            call("c9", "other", "oracle"),
            call("c10", "other", "list files"),
            call("c11", "think", "Plan the next step"),
            call("c12", "THINK", "Agent run"),
            call("c13", "other", "Skill: pdf"),
            call("c14", "move", "mv a b"),
            call("c15", "other", "terminal command"),
        ],
    )

    hostile = '"><script>globalThis.pwned=true</script>'
    trial(
        "tool-status__00000001",
        [
            call("s1", "other", "WebSearch", status=hostile),
            call("s2", "execute", "ls", status="IN-PROGRESS"),
            call("s3", "execute", "ls", status="in progress"),
            call("s4", "execute", "ls", status=" Completed "),
            call("s5", "execute", "ls", status="cancelled"),
            call("s6", "execute", "ls", status=5),
            call("s7", "execute", "ls", status=None),
            call("s8", "execute", "ls", status="pending"),
        ],
    )

    spawn = {"type": "tool_call", "kind": "think", "status": "completed"}
    trial(
        "spawn-labels__00000001",
        [
            {**spawn, "tool_call_id": "claude", "title": "Task"},
            {
                **spawn,
                "tool_call_id": "claude-described",
                "title": "Create child.txt",
                "raw_input": {"description": "Create child.txt", "prompt": "go"},
            },
            {**spawn, "tool_call_id": "opencode", "title": "task"},
            {**spawn, "tool_call_id": "opencode-args", "title": 'task {"description": "Read the 2024 filings"}'},
            {**spawn, "tool_call_id": "plan", "title": "Plan the next step"},
            {**spawn, "tool_call_id": "typed", "title": "x", "raw_input": {"prompt": "p", "subagent_type": "general"}},
            {**spawn, "tool_call_id": "prompt-only", "title": "x", "raw_input": {"prompt": "p"}},
        ],
    )

    claude_child = [
        {"type": "user_message", "text": "use one child agent"},
        tool(
            "toolu_task",
            raw_input={
                "description": "Create child.txt",
                "prompt": "create /app/child.txt",
                "subagent_type": "general-purpose",
            },
        ),
        {"type": "agent_thought", "text": "child thinking", "parent_tool_call_id": "toolu_task"},
        tool("toolu_bash", "toolu_task", kind="execute", title="printf x > f"),
        tool("toolu_xxd", "toolu_task", kind="execute", title="xxd f", status="failed"),
        say("child done", "toolu_task"),
        say("root continues"),
    ]
    trial("subagents-claude__00000001", claude_child, result=result_doc("sub-task", "claude-agent-acp", "sonnet-x", 1.0))

    max_depth = 8
    chain = [tool("a0")]
    for level in range(1, max_depth + 2):
        chain.append(tool(f"a{level}", f"a{level - 1}"))
    chain.append(say("deepest", f"a{max_depth + 1}"))
    trial("subagents-depth__00000001", chain)

    trial(
        "subagents-orphans__00000001",
        [
            say("root first"),
            say("orphan one", "toolu_missing"),
            tool("child_call", "toolu_missing"),
            say("grandchild", "child_call"),
            say("root last"),
        ],
    )
    trial("subagents-cyclic__00000001", [say("root"), tool("x", "y"), tool("y", "x"), tool("self", "self")])
    trial(
        "subagents-mixed__00000001",
        [
            *claude_child,
            say("orphan", "nope"),
            tool("z", "z"),
            tool("toolu_task", kind="execute", title="duplicate call id"),
            say("after duplicate", "toolu_task"),
            {"type": "agent_message", "text": "non-str parent", "parent_tool_call_id": 5},
            {"type": "agent_message", "text": "empty parent", "parent_tool_call_id": ""},
            {"type": "tool_call", "tool_call_id": 9, "kind": "think", "title": "Task", "status": "completed"},
            say("child of numeric id", "9"),
        ],
    )

    hostile_text = '<img src=x onerror="alert(1)"> and </script><script>alert(2)</script> <!--<script> & \u2028\u2029 done'
    trial(
        "hostile-text__00000001",
        [say(hostile_text), {"type": "user_message", "text": "</script>"}],
        result=result_doc("<b>task</b> & co", "gemini", "flash-y", 1.0),
    )
    trial("__BENCHFLOW_PAYLOAD__", [say("payload survived")])
    trial("title-&-'quotes'__00000001", [say("title escaping")])

    trial(
        "lone-surrogate__00000001",
        raw='{"type":"agent_message","text":"lead \\ud800 tail"}\n'
        '{"type":"agent_message","text":"low \\udc00 only"}\n'
        '{"type":"agent_message","text":"pair \\ud83d\\ude00 ok"}\n'
        '{"type":"agent_message","text":"high then A \\ud800\\u0041"}\n'
        '{"type":"agent_message","text":"bad hex \\ud800\\uZZZZ"}\n'
        '{"type":"agent_message","text":"\\ud83d"}\n',
    )

    trial(
        "pending-not-list__00000001",
        [
            {
                "type": "agent_timeout",
                "reason": "x",
                "timeout_sec": 1.0,
                "pending_tool_call_ids": "not-a-list",
                "terminal_trajectory_complete": False,
            },
            {"type": "agent_timeout", "timeout_sec": "12", "terminal_trajectory_complete": "yes"},
        ],
    )

    trial(
        "non-finite__00000001",
        [{"type": "tool_call", "kind": "read", "title": "file", "status": "completed", "ts": float("nan")}],
        result={"rewards": {"reward": float("nan")}, "agent_result": {"total_tokens": float("inf"), "cost_usd": float("-inf")}},
        timing={"total": float("inf"), "agent_execution": 2.0, "nan": float("nan")},
    )

    trial(
        "prompts-shapes__00000001",
        [say("done")],
        prompts=["text", 7, {"nested": "prompt"}, None, 1.5, True, [1, "x"], 1e16, -0.0],
    )
    trial("prompts-dedup__00000001", [{"type": "user_message", "text": "UNIQ-42"}], prompts=["UNIQ-42"])
    trial("prompts-wrong-top__00000001", [say("x")], prompts={"prompt": "wrong top level"})
    trial("prompts-string__00000001", [say("x")], prompts="wrong top level")
    trial("prompts-task__00000001", [say("done")], prompts=[{"task": "go"}])

    trial(
        "binary-sidecars__00000001",
        [say("hi")],
        files={
            "result.json": b"\xff\xfe\x00 not utf8",
            "verifier/test-stdout.txt": b"\xff\xfe\x00 binary \xe2\x82 end\n",
            "verifier/reward.txt": b"\xef\xbb\xbf1.0\n",
        },
    )
    trial(
        "binary-result-valid__00000001",
        [say("hi")],
        files={"result.json": b'{"task_name": "caf\xc3\xa9 \xff", "rewards": {"reward": 1}}'},
    )

    for index, (result, timing, ctrf) in enumerate(
        [
            ([], [], {"results": []}),
            ({"rewards": []}, {"total": {"not": "numeric"}}, {"results": 1}),
            ({"agent_result": "not-an-object"}, {"total": "not-a-number"}, {"results": {"tests": "not-a-list"}}),
        ]
    ):
        trial(
            f"wrong-shapes-{index}__00000001",
            [say("ok")],
            result=result,
            timing=timing,
            files={"verifier/ctrf.json": ctrf},
        )

    trial(
        "malformed-ctrf__00000001",
        [],
        files={
            "verifier/ctrf.json": {
                "results": {
                    "tests": [
                        {"name": {"nested": "name"}, "status": "<style>bad</style>", "duration": float("inf")},
                        "not-an-object",
                        {"name": None, "status": "Passed", "duration": True},
                    ]
                }
            }
        },
    )

    for depth in (80, 1200):
        deep = '{"type":"future","payload":' + nested_array(depth) + "}"
        trial(f"deep-jsonl-{depth}__00000001", raw=deep + "\n" + json.dumps({"type": "user_message", "text": "safe"}))
    edge = nested_array(63)
    over = nested_array(64)
    trial(
        "depth-edges__00000001",
        raw="\n".join(
            [
                '{"type":"future","payload":' + nested_array(62) + "}",
                '{"type":"future","payload":' + edge + "}",
                '{"type":"agent_message","text":"kept"}',
                over,
            ]
        ),
    )

    nested = nested_array(80)
    trial(
        "deep-sidecars__00000001",
        [say("still renders")],
        files={
            "result.json": '{"agent_timeout_info":' + nested + "}",
            "timing.json": '{"total":' + nested + "}",
            "prompts.json": "[" + nested + "]",
            "verifier/ctrf.json": '{"results":{"tests":' + nested + "}}",
        },
    )

    trial(
        "catalog-shared__00000001",
        [say("ok")],
        result={
            "task_name": 1034,
            "agent_name": ["wrong-shape"],
            "agent": "fallback-agent",
            "model": {"wrong": "shape"},
            "skill_mode": "no-skill",
            "rewards": {"reward": "1"},
            "n_tool_calls": "4",
            "agent_result": {"n_tool_calls": "3", "total_tokens": "42", "cost_usd": "0.25"},
            "timing": {"total": 999},
            "export_error": "could not export",
        },
        timing={"total": "12.5", "agent_execution": "2"},
    )
    trial("timing-fallback__00000001", [], result={"timing": {"total": 7}}, timing=["wrong-shape"])
    trial("timing-embedded-only__00000001", [say("x")], result={"timing": {"total": 7.5, "agent": None, "x": "nope"}})

    trial(
        "timestamps-aware__00000001",
        [
            {"type": "user_message", "text": "go", "ts": "2026-08-15T09:41:03+00:00"},
            {
                "type": "tool_call",
                "tool_call_id": "c1",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "content": [],
                "started_at": "2026-08-15T09:41:05+00:00",
                "finished_at": "2026-08-15T09:41:07.500000+00:00",
            },
            {
                "type": "tool_call",
                "tool_call_id": "c2",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "ts": "2026-08-15T11:41:05+02:00",
                "finished_at": "2026-08-15T09:41:05.25Z",
            },
            {
                "type": "tool_call",
                "tool_call_id": "c3",
                "kind": "execute",
                "title": "backwards",
                "status": "completed",
                "started_at": "2026-08-15T09:41:09+00:00",
                "finished_at": "2026-08-15T09:41:08+00:00",
            },
            {"type": "agent_message", "text": "epoch", "ts": 1786786863.5},
            {"type": "agent_message", "text": "epoch int", "ts": 1786786864},
            {"type": "agent_message", "text": "week date", "ts": "2026-W33-6T09:41:03-0530"},
            {"type": "agent_message", "text": "basic", "ts": "20260815T094103.123456789+0000"},
            {"type": "agent_message", "text": "epoch string", "ts": "1786786863"},
            {"type": "agent_thought", "text": "offset seconds", "ts": "2026-08-15T09:41:03+05:30:15.5"},
        ],
    )
    trial(
        "timestamps-bad__00000001",
        [
            {"type": "agent_message", "text": "hi", "ts": "not-a-time"},
            {
                "type": "tool_call",
                "tool_call_id": "c",
                "kind": "read",
                "title": "x",
                "status": "completed",
                "content": [],
                "started_at": True,
            },
            {"type": "agent_message", "text": "bad month", "ts": "2026-13-01T00:00:00Z"},
            {"type": "agent_message", "text": "list", "ts": [1, 2]},
            {"type": "agent_message", "text": "huge", "ts": 1e400},
        ],
    )

    trial("diag-info__00000001", [say("hi")], result={"idle_timeout_info": {"idle_timeout_sec": 30}, "usage_limit_info": "hit"})
    trial(
        "diag-error__00000001",
        [say("hi")],
        result={
            "error": "agent timed out",
            "agent_timeout_info": {"timeout_sec": 900},
            "integration_failure_info": {"cause": "login", "detail": "no token"},
            "api_error_info": [],
            "suspected_api_error_info": 0,
        },
    )
    big = "x" * 5000
    trial("untruncated__00000001", [{"type": "future_thing", "blob": big}], result={"error": "boom", "agent_timeout_info": {"detail": big}})

    complete_scoring = {
        "schema_version": 1,
        "policy": "tests-blockers-quality-v1",
        "status": "complete",
        "passed": True,
        "tests_pass": True,
        "all_blockers_pass": True,
        "failed_blockers": [],
        "verifier_reward": 1.0,
        "rubric_reward": 0.75,
        "reviewer_run": "reviews/abc",
    }
    status_cases = [
        {"rewards": {"reward": 1.0}},
        {"rewards": None, "verifier_error": "verifier timed out after 30.0s", "verifier_error_category": "verifier_timeout"},
        {"verifier_error": "verifier crashed: Failed to get session command: "},
        {"error": "agent timed out", "error_category": "agent_timeout"},
        {"error": "ACP error -32603: Internal error"},
        {"error": "Prompt exceeded wall-clock budget", "partial_trajectory": True, "rewards": {"reward": 0.0}},
        {"rewards": {"reward": 0.75}, "scoring": complete_scoring},
        {"rewards": {"reward": 1.0}, "scoring": {"status": "error", "error": "reviewer crashed"}},
        {"rewards": {"reward": 1.0}, "scoring": {"status": "complete"}},
        # More scoring shapes.
        {
            "rewards": {"reward": 0.0, "verifier_reward": 1.0, "rubric_reward": 0.4},
            "scoring": {**complete_scoring, "passed": False, "all_blockers_pass": False, "failed_blockers": ["no_fab"], "rubric_reward": 0.4},
        },
        {"rewards": {"reward": 0.5}, "scoring": complete_scoring},
        {"reward": 0.75, "scoring": complete_scoring},
        {"rewards": {"reward": 0.75, "verifier_reward": 0.5}, "scoring": complete_scoring},
        {"rewards": {"reward": 0.75}, "scoring": {**complete_scoring, "extra": 1}},
        {"rewards": {"reward": 0.75}, "scoring": {**complete_scoring, "verifier_reward": True}},
        {"rewards": {"reward": 0.75}, "scoring": {**complete_scoring, "schema_version": 1.0}},
        {"rewards": {"reward": 0.75}, "scoring": {**complete_scoring, "revision": "scoring/a1.json"}},
        {"rewards": {"reward": 0.75}, "scoring": {**complete_scoring, "revision": "scoring/../x.json"}},
        {"rewards": {"reward": 0.75}, "scoring": {**complete_scoring, "reviewer_run": "  "}},
        {"rewards": {"reward": 0.75}, "scoring": {**complete_scoring, "failed_blockers": ["a", "a"]}},
        {"rewards": {"reward": 1}, "scoring": {"status": "error", "error": "transport", "error_category": "reviewer_transport", "tests_pass": True, "verifier_reward": 1}},
        {"rewards": {"reward": 1}, "scoring": {"status": "error", "error": "  "}},
        {"rewards": {"reward": 1}, "scoring": {"status": "error", "error": "x", "passed": False}},
        {"rewards": {"reward": 1}, "scoring": []},
        {"rewards": {"reward": 1}, "scoring": {}},
        {"rewards": {"reward": 1}, "scoring": None},
        {"rewards": {"reward": 1}, "scoring": {"status": "errored"}},
        {"rewards": {"reward": 1}, "scoring": {**complete_scoring, "policy": "other"}},
        {"rewards": {"reward": 1}, "scoring": {**complete_scoring, "rubric_reward": 1.5}},
        {"rewards": {"reward": 1}, "scoring": {**complete_scoring, "tests_pass": 1}},
        # Assessment declarations.
        {"rewards": {"reward": 1.0}, "assessment": {"status": "pending", "reason": "awaiting_reviewer"}},
        {"rewards": {"reward": 1.0}, "assessment": "unassessable"},
        {"status": "awaiting_assessment", "rewards": {"reward": 1.0}},
        {"rewards": {"reward": 1.0}, "assessment": {"status": ["verified"]}},
        {"rewards": {"reward": 0.0}, "assessment": {"status": "failed"}},
        {"rewards": {"reward": 0.5}, "assessment": "verified", "scoring": {"status": "error", "error": "x"}},
        {"rewards": {"reward": 0.5}, "assessment": {"status": "pending", "reason": ""}},
        {"rewards": {"reward": 0.5}, "assessment": None, "status": "awaiting_assessment"},
        # Error text edge cases (truthiness, NaN, containers).
        {"error": 0, "verifier_error": "", "export_error": False},
        {"error": float("nan")},
        {"error": {"type": "RuntimeError", "message": "boom"}, "error_category": ""},
        {"error": ["a", 1], "error_category": 5},
        {"verifier_error": "x", "verifier_error_category": None},
        {"agent_name": "", "agent": "fallback"},
        {"agent_name": "", "agent": None},
        {"agent_name": float("inf"), "agent": "inf-fallback", "task_name": float("nan"), "model": True, "skill_mode": 3.0},
        {"started_at": 1786786863, "finished_at": None, "trajectory_source": 5, "partial_trajectory": "yes"},
    ]
    for index, result in enumerate(status_cases):
        trial(f"status-{index:02d}__00000001", [say("done")], result=result)

    trial("status-none__00000001", [say("done")])

    error_strings = [
        "Agent integration failure: login missing; ACP error http 401",
        "Usage limit reached for the 5-hour window",
        "ACP error -32603: Internal error: You've hit your weekly limit · resets 4pm (UTC)",
        "ACP error -32603: Internal error: You\u2019ve reached your usage limit",
        "agent idle for 600s",
        "install failed: npm ERR",
        "process closed stdout",
        "pty closed by the peer",
        "DaytonaPtyProcess: readline timeout",
        "suspected provider api error: zero tokens",
        "provider api error [rate_limit/transient] 429",
        "ACP error -32603: provider auth failed (HTTP 401)",
        "ACP error -32603: provider rate limited (HTTP 429)",
        "ACP error -32603: provider unavailable (HTTP 503)",
        "ACP error -32603: provider rejected request (HTTP 400)",
        "ACP error -32603: Internal error",
        "native harness error: crashed",
        "key was rejected as invalid",
        "Sandbox startup failed: image missing",
        "docker compose command failed: failed to resolve source metadata for x",
        "Prompt exceeded wall-clock budget",
        "connection reset by peer",
        "transient sandbox transport failure: 502",
        "agent timed out after 900s",
        "something unexpected",
    ]
    for index, error in enumerate(error_strings):
        trial(f"classify-error-{index:02d}__00000001", [say("x")], result={"error": error})

    verifier_errors = [
        "separate verifier sandbox never started",
        "verifier_wedge: gone",
        "[solver-preserved] verifier recovery failed: boom",
        "verifier crashed: dependency install failed",
        "verifier crashed: Failed to download `x`",
        "verifier crashed: connection reset",
        "verifier crashed: verifier setup failed",
        "verifier crashed: boom",
        "Verifier crashed: boom",
        "verifier timed out after 30s",
        "something else",
    ]
    for index, error in enumerate(verifier_errors):
        trial(f"classify-verifier-{index:02d}__00000001", [say("x")], result={"verifier_error": error})

    def recovery(root: Path, attempt: str, record: object, *, point: bool = True) -> None:
        folder = root / "verifier-recovery" / attempt
        folder.mkdir(parents=True)
        body = {"attempt": f"verifier-recovery/{attempt}", **record} if isinstance(record, dict) else record
        _write(folder / "recovery.json", body)
        if point:
            _write(root / "verification.json", {"attempt": f"verifier-recovery/{attempt}"})

    root = trial(
        "recovery-two__00000001",
        [say("done")],
        result={
            "verifier_error": "[solver-preserved] verifier recovery failed: boom",
            "verifier_error_category": "verifier_infra",
        },
    )
    recovery(root, "aaa111", {"status": "unavailable", "error": "lease missing"}, point=False)
    recovery(
        root,
        "bbb222",
        {
            "status": "failed",
            "original_error": "[solver-preserved] verifier_wedge: gone",
            "evidence": "workspace-only",
            "solver_replayed": False,
            "timing": {"verifier": 12.5},
            "rewards": None,
            "error": "verifier crashed: boom",
            "admission": "admitted",
            "cleanup_error": "",
        },
    )
    root = trial("recovery-publication__00000001", [say("done")], result={"rewards": {"reward": 1}})
    recovery(
        root,
        "ccc333",
        {"status": "complete", "rewards": {"reward": 1}, "publication_error": "OSError: [Errno 28] No space left on device"},
    )
    root = trial("recovery-bad-pointer__00000001", [say("done")], result={"rewards": {"reward": 0}})
    _write(root / "verification.json", {"attempt": "../../etc"})
    root = trial("recovery-missing-receipt__00000001", [say("done")], result={"rewards": {"reward": 0}})
    _write(root / "verification.json", {"attempt": "verifier-recovery/ddd444"})
    root = trial("recovery-entries__00000001", [say("done")], result={"rewards": {"reward": 1}})
    _write(root / "verifier-recovery" / "plainfile", "x")
    (root / "verifier-recovery" / "bad-name").mkdir(parents=True)
    _write(root / "verifier-recovery" / "bad-name" / "recovery.json", {"status": "ignored"})
    _write(root / "verifier-recovery" / "noreceipt" / "keep.txt", "an attempt folder without a receipt")
    _write(root / "verifier-recovery" / "listreceipt" / "recovery.json", [1, 2])
    _write(root / "verification.json", ["not", "a", "dict"])
    root = trial("recovery-pointer-file-only__00000001", [say("done")])
    _write(root / "verification.json", "{not json")

    # Rollout-branch lineage written by hand (the engine writes jobs/branched).
    trial("lineage-bad-json__00000001", [], files={"tree.json": "{not json"})
    trial("lineage-other-kind__00000001", [], files={"tree.json": {"kind": "other"}})
    trial("lineage-schema-9__00000001", [], files={"tree.json": {"kind": "benchflow-branch-tree", "schema_version": 9}})
    trial("lineage-schema-none__00000001", [], files={"tree.json": {"kind": "benchflow-branch-tree"}})
    trial(
        "lineage-schema-float__00000001",
        [],
        files={"tree.json": {"kind": "benchflow-branch-tree", "schema_version": 1.0, "nodes": "x", "forks": {"a": 1}}},
    )
    root = trial(
        "lineage-hand__00000001",
        [say("parent")],
        result={"task_name": "hand", "agent_name": "dummy", "model": "m", "skill_mode": "no-skill", "rewards": {"reward": 1.0}},
    )
    child = {"status": "available", "path": "branches/f1/children/n1"}
    tree = {
        "schema_version": 1,
        "kind": "benchflow-branch-tree",
        "nodes": [{"id": "root"}, {"id": "n1"}, {"id": "n2"}],
        "forks": [
            {
                "id": "f1",
                "parent_node": "root",
                "requested_children": True,
                "snapshot": {"requested_layers": ["sandbox", 5, None, ""], "captured_layers": "nope"},
                "status": "partial",
                "parent_restore": "restored",
                "value": "0.25",
                "error": {"type": "RuntimeError", "code": "fork_failed", "message": "secret"},
                "parent_restore_error": "plain text",
                "artifact_error": {"type": None, "code": None},
                "children": [
                    {
                        "index": 0,
                        "node_id": "n1",
                        "status": "scored",
                        "reward": 1,
                        "reward_source": "verifier",
                        "error": None,
                        "cleanup_error": {"type": "OSError"},
                        "intervention": {"label": "baseline", "requested": "", "execution": 5, "evidence": {"k": 1}},
                        "artifacts": child,
                    },
                    {
                        "index": 1.0,
                        "node_id": "n2",
                        "status": "unscored",
                        "reward": None,
                        "error": {"code": "missing_verifier_reward"},
                        "intervention": "not-a-dict",
                        "artifacts": {"status": "available", "path": "branches/f1/children/n2"},
                    },
                    {
                        "index": True,
                        "node_id": "n3",
                        "status": "scored",
                        "reward": 0.0,
                        "artifacts": {"status": "available", "path": "../lineage-outside-target__00000001"},
                    },
                    {"node_id": "../x", "artifacts": {"status": "available", "path": "branches/f1/children/n1"}},
                    {"node_id": "n1\n", "artifacts": {"status": "available", "path": "branches/f1/children/n1"}},
                    {"node_id": "n4", "artifacts": {"status": "unavailable", "path": None}},
                    {"node_id": "n5", "artifacts": {"status": "available", "path": "branches/f1/children/missing"}},
                    {"node_id": "n6", "artifacts": {"status": "available", "path": ""}},
                    {"node_id": "n7", "artifacts": {"status": "available", "path": "."}},
                    "not-a-dict",
                ],
            },
            {
                "id": "f1\n",
                "parent_node": "n1",
                "children": [{"node_id": "n1", "artifacts": child}],
            },
            {
                "id": "bad/id",
                "children": [{"node_id": "n1", "artifacts": child}],
            },
            "not-a-dict",
            {"children": "nope"},
        ],
    }
    _write(root / "tree.json", tree)
    n1 = root / "branches" / "f1" / "children" / "n1"
    _write(
        n1 / "observation.json",
        {
            "trajectory": [say("continuation of n1"), "not-an-event", {"type": "tool_call", "tool_call_id": "t", "kind": "execute", "title": "ls", "status": "completed"}],
            "native_usage": {"total_tokens": 10, "cost_usd": "0.5"},
            "timing": {"agent_execution": 4.0, "total": 9},
            "error": None,
            "export_error": "child export failed",
        },
    )
    _write(n1 / "mounted" / "verifier" / "reward.txt", "1.0")
    _write(n1 / "verifier" / "reward.txt", "should not be read")
    n2 = root / "branches" / "f1" / "children" / "n2"
    _write(
        n2 / "observation.json",
        {"trajectory": "nope", "error": "child crashed", "verifier_error": None, "timing": ["bad"]},
    )
    _write(n2 / "verifier" / "reward.txt", "0.0")
    _write(n2 / "verifier" / "test-stdout.txt", "isolated child stdout\n")
    outside = trial("lineage-outside-target__00000001", [say("outside")])
    _write(outside / "observation.json", {"trajectory": []})

    # Review reports beside the trials (the ``bench review`` layout).
    def report(dirname: str, name: str, *, valid: bool, gated: float = 0.5) -> None:
        body = {
            "trial_name": name,
            "review_valid": valid,
            "summary": "Derivation complete, one unsupported claim.",
            "criterion_metadata": [
                {"name": "no_fabrication", "blocker": 1, "weight": 1},
                {"name": "derivation", "blocker": 0, "weight": 5},
                {"name": "style", "blocker": "yes", "weight": 2.0},
                {"weight": True},
                "not-a-dict",
            ],
            "checks": {
                "no_fabrication": {"outcome": "pass", "explanation": "cites the data"},
                "derivation": {"score": 1, "explanation": "skips one step"},
                "style": {"score": 1.5, "outcome": 3, "explanation": None},
            },
            "scoring": {
                "deterministic_pass": True,
                "all_blockers_pass": True,
                "failed_blockers": [],
                "raw_quality": gated,
                "gated_quality": gated,
                "decision": "presentable_with_revisions",
            }
            if valid
            else None,
        }
        _write(SYN / dirname / "review_report.json", {"reviewer": {"model": "gateway/reviewer-x"}, "trials": [body]})

    rubric_tool = [{"type": "tool_call", "tool_call_id": "t1", "kind": "execute", "title": "ls", "status": "completed", "content": []}]
    trial("rubric-sibling__abcd1234", rubric_tool, result={"task_name": "t", "model": "m"})
    report("review-a", "rubric-sibling__abcd1234", valid=True, gated=0.2)
    report("review-b/nested", "rubric-sibling__abcd1234", valid=True, gated=0.9)
    report("review-c", "rubric-sibling__abcd1234", valid=False)
    trial("rubric-invalid-only__abcd1234", rubric_tool, result={"task_name": "t"})
    report("review-c/other", "rubric-invalid-only__abcd1234", valid=False)
    trial("rubric-legacy__abcd1234", rubric_tool, result={"task_name": "t"})
    _write(
        SYN / "review-legacy" / "review_report.json",
        {
            "reviewer": {"model": 5},
            "trials": [
                {
                    "trial_name": "rubric-legacy__abcd1234",
                    "review_valid": 1,
                    "summary": "ok",
                    "criterion_metadata": [{"name": "cites_sources", "blocker": None, "weight": None}],
                    "checks": {"cites_sources": {"outcome": "fail", "explanation": "none"}},
                    "scoring": None,
                },
                {"trial_name": "rubric-legacy__abcd1234", "review_valid": 0, "summary": ["second", "entry"], "scoring": {"later": True}},
            ],
        },
    )
    _write(SYN / "review-malformed" / "review_report.json", {"trials": {"oops": 1}})
    _write(SYN / "review-not-a-dir", "a file named like a review dir")

    root = trial("rubric-auto__00000001", rubric_tool)
    attempt = "a1b2"
    _write(
        root / "reviews" / attempt / "rubric.json",
        {"criteria": [{"name": "notes_present", "blocker": 1, "weight": 5}, {"name": "notes_accuracy", "blocker": 0, "weight": 5}]},
    )
    auto_report = {
        "attempt": attempt,
        "reviewer": {"agent": "claude-agent-acp", "model": "reviewer-model"},
        "rubric_snapshot": f"reviews/{attempt}/rubric.json",
        "checks": {
            "notes_present": {"outcome": "pass", "explanation": "NOTES.md exists"},
            "notes_accuracy": {"score": 2, "explanation": "4 rows, mean"},
        },
        "summary": "Correct answer and notes.",
        "review_valid": True,
        "scoring": {"passed": True, "rubric_reward": 1.0, "status": "complete"},
    }
    _write(root / "scoring" / f"{attempt}.json", auto_report)
    _write(root / "reviews" / attempt / "review_report.json", auto_report)
    _write(
        root / "result.json",
        {"task_name": "t", "model": "m", "scoring": {"passed": True, "revision": f"scoring/{attempt}.json"}},
    )
    root = trial("rubric-auto-escape__00000001", rubric_tool)
    _write(root / "result.json", {"task_name": "t", "scoring": {"revision": "../rubric-auto__00000001/scoring/a1b2.json"}})
    root = trial("rubric-auto-snapshot-escape__00000001", rubric_tool)
    _write(root / "scoring" / "x.json", {**auto_report, "rubric_snapshot": "../rubric-auto__00000001/reviews/a1b2/rubric.json"})
    _write(root / "result.json", {"task_name": "t", "scoring": {"revision": "scoring/x.json"}})

    # Line splitting exactly as Python's str.splitlines() does it.
    lines = [
        json.dumps({"type": "agent_message", "text": "first"}),
        json.dumps({"type": "agent_message", "text": "crlf"}),
        json.dumps({"type": "agent_message", "text": "cr"}),
        json.dumps({"type": "agent_message", "text": "vt"}),
        json.dumps({"type": "agent_message", "text": "ff"}),
        json.dumps({"type": "agent_message", "text": "fs"}),
        json.dumps({"type": "agent_message", "text": "nel"}),
        json.dumps({"type": "agent_message", "text": "ls"}),
        '{"type": "agent_message", "text": "raw \u2028 inside"}',
        json.dumps({"type": "agent_message", "text": "after"}),
    ]
    raw = (
        "\ufeff"
        + lines[0]
        + "\r\n"
        + lines[1]
        + "\r\n  \n\t\n"
        + lines[2]
        + "\r"
        + lines[3]
        + "\x0b"
        + lines[4]
        + "\x0c"
        + lines[5]
        + "\x1c"
        + lines[6]
        + "\x85"
        + lines[7]
        + "\u2029"
        + lines[8]
        + "\n"
        + lines[9]
        + "   "
    )
    trial("jsonl-separators__00000001", raw=raw.encode("utf-8"))
    trial(
        "jsonl-bom-only__00000001",
        raw=("\ufeff" + json.dumps({"type": "agent_message", "text": "dropped"}) + "\n" + json.dumps({"type": "agent_message", "text": "kept"})).encode(),
    )
    trial(
        "jsonl-invalid-utf8__00000001",
        raw=b'{"type": "agent_message", "text": "bad \xff\xfe byte"}\n{"type": "agent_message", "text": "trunc \xe2\x82"}\n',
    )

    trial(
        "json-edge__00000001",
        raw="\n".join(
            [
                '{"type": "future", "big": 123456789012345678901234567890, "neg": -9223372036854775809, "f": [1.0, 1e16, 1E-5, 0.1, -0.0, 1e400, 5e-324, 2.5e+300, 123456789.125]}',
                '{"type": "future", "dup": 1, "dup": 2, "dup2": {"a": 1}, "dup2": [3]}',
                '{"type": "future", "esc": "tab\\tnl\\nnul\\u0000ctl\\u001fdel\\u007f quote\\" back\\\\ slash\\/", "uni": "caf\\u00e9 \u4e16\u754c \U0001f389"}',
                '{"type": 5, "text": "numeric type"}',
                '{"type": null, "text": "null type"}',
                '{"type": ["user_message"], "text": "list type"}',
                '{"type": "agent_message", "text": 12345678901234567890}',
                '{"type": "agent_message", "text": 1.5e-7}',
                '{"type": "agent_message", "text": false}',
                '{"type": "agent_message", "text": {"structured": [1, {"b": null}]}}',
                '{"type": "agent_message"}',
                '[{"type": "agent_message", "text": "a list line"}]',
                '"just a string"',
                '{"type": "agent_message", "text": "trailing comma",}',
                '{"type": "agent_message", "text": "NaN value", "n": NaN, "i": -Infinity}',
                '  {"type": "agent_message", "text": "padded"}  ',
                '{"type": "agent_thought", "text": "\\ud800\\udc00 pair at start"}',
            ]
        ),
    )

    trial(
        "codex-shapes__00000001",
        [
            {"type": "user_message", "text": "fix the bug"},
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "kind": "execute",
                "title": "Run rg -n TODO",
                "status": "completed",
                "content": [{"type": "terminal", "terminalId": "call_1"}],
                "raw_input": {"command": ["/bin/bash", "-lc", "rg -n TODO"], "cwd": "/app", "parsed_cmd": [{"type": "search"}]},
                "raw_output": {"formatted_output": "src/a.py:3: # TODO\n", "exit_code": 0, "duration": {"secs": 0, "nanos": 1200}},
            },
            {
                "type": "tool_call",
                "tool_call_id": "call_2",
                "kind": "execute",
                "title": "Run pytest",
                "status": "failed",
                "content": [{"type": "terminal", "terminalId": "call_2"}],
                "raw_input": {"command": ["/bin/bash", "-lc", "pytest -q"]},
                "raw_output": {"formatted_output": "E   assert 1 == 2\n", "exit_code": 1},
            },
            {"type": "agent_message", "text": "Fixed."},
        ],
        result=result_doc("bugfix", "@agentclientprotocol/codex-acp", "gpt-5.5", 1.0, skill_mode="with-skill"),
    )

    trial("empty-trajectory__00000001", raw="", result={"rewards": {"reward": 1}})
    # A folder where the trajectory file should be (it holds a file so git
    # keeps it).
    d = SYN / "trajectory-is-dir__00000001" / "trajectory" / "acp_trajectory.jsonl"
    _write(d / "keep.txt", "this folder stands where a file is expected")
    _write(d.parent.parent / "result.json", {"task_name": "dir"})
    trial("result-not-dict__00000001", [say("x")], result="just a string")
    trial(
        "usage-shapes__00000001",
        [say("x")],
        result={
            "agent_result": {
                "zeta": 1,
                "n_tool_calls": "42",
                "n_prompts": 3.0,
                "n_input_tokens": -1,
                "n_output_tokens": 2.5,
                "n_cache_read_tokens": True,
                "n_cache_creation_tokens": 10**20,
                "total_tokens": "1e3",
                "cost_usd": "0.25",
                "usage_source": 5,
                "price_source": None,
                "extra": {"nested": [float("nan"), 1]},
            },
            "n_tool_calls": None,
        },
    )
    trial("usage-big__00000001", [say("x")], result={"agent_result": {"total_tokens": 12345678901234567890123, "n_tool_calls": 2**63}})


async def build_branched() -> None:
    """A parent rollout with two forks, written by 0.8's real branch engine."""
    from benchflow.branch_lineage import UnscoredChildError
    from benchflow.environment.protocol import StateSnapshot
    from benchflow.rollout import Rollout, RolloutConfig, Scene
    from benchflow.task.paths import RolloutPaths

    run = BRANCHED / "jobs" / "job" / "count-files__b1a2c3d4"
    rollout = Rollout(RolloutConfig(task_path=BRANCHED / "task", scenes=[Scene.single(agent="dummy --agent")]))
    rollout._rollout_dir = run
    rollout._rollout_paths = RolloutPaths(run)
    rollout._rollout_paths.mkdir()
    rollout.disconnect = AsyncMock()
    rollout._environment = type(
        "Environment",
        (),
        {
            "snapshot": AsyncMock(
                side_effect=[
                    StateSnapshot(id="first", path="/state/first"),
                    StateSnapshot(id="second", path="/state/second"),
                ]
            ),
            "restore": AsyncMock(),
        },
    )()
    outcomes = iter([1.0, 0.0, 1.0, None])

    async def child(node):
        reward = next(outcomes)
        rollout._trajectory.append({"type": "agent_message", "text": f"continuation of {node.id}"})
        verifier = rollout._rollout_paths.verifier_dir
        verifier.mkdir(parents=True, exist_ok=True)
        if reward is None:
            rollout._verifier_error = "verifier timed out after 600s"
            raise UnscoredChildError("no canonical reward")
        (verifier / "reward.txt").write_text(str(reward))
        return reward

    await rollout.branch(2, child, child_labels=["baseline", "hint: off-by-one"])
    try:
        await rollout.branch(2, child)
    except UnscoredChildError:
        pass
    (run / "trajectory").mkdir(exist_ok=True)
    (run / "trajectory" / "acp_trajectory.jsonl").write_text(json.dumps({"type": "agent_message", "text": "parent"}))
    (run / "result.json").write_text(
        json.dumps({"task_name": "count-files", "agent_name": "dummy", "model": "m", "rewards": {"reward": 1.0}})
    )


# ── goldens ───────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Write absolute fixture paths as <root> so goldens are machine-independent."""
    for prefix in {str(ROOT.resolve()), str(ROOT)}:
        text = text.replace(prefix, "<root>")
    return text


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_parts(page: str) -> str:
    """The boot data a page embeds (the text of its bf-payload script)."""
    return page.split('type="application/json">', 1)[1].split("</script>", 1)[0]


def build_goldens() -> None:
    from benchflow.trajectories.viewer.catalog import _discover_rollouts, _rollout_summary
    from benchflow.trajectories.viewer.legacy import render_rollout
    from benchflow.trajectories.viewer.payload import (
        _build_acp_payload,
        _build_branch_child_payload,
        _safe_json,
    )
    from benchflow.trajectories.viewer.render import _json_for_script, _load_template, _render_shell

    GOLDEN.mkdir(parents=True)
    (GOLDEN / "payloads").mkdir()
    (GOLDEN / "pages").mkdir()
    (GOLDEN / "template.sha256").write_text(sha256(_load_template()) + "\n")

    ids = _discover_rollouts(ROOT, cap=100_000)
    index = []
    for number, rid in enumerate(ids):
        payload = _build_acp_payload(ROOT / rid, None)
        name = f"payloads/{number:03d}.json"
        (GOLDEN / name).write_text(normalize(_safe_json(payload.to_payload())), encoding="utf-8")
        entry = {"id": rid, "payload": name, "branches": {}}
        lineage = payload.lineage
        refs = []
        for fork in lineage.forks if lineage is not None else []:
            for child in fork.children:
                if child.ref is not None and child.ref not in refs:
                    refs.append(child.ref)
        for child_number, ref in enumerate(refs):
            child_payload = _build_branch_child_payload(ROOT / rid, ref)
            child_name = f"payloads/{number:03d}-b{child_number}.json"
            text = "null" if child_payload is None else _safe_json(child_payload.to_payload())
            (GOLDEN / child_name).write_text(normalize(text), encoding="utf-8")
            entry["branches"][ref] = child_name
        index.append(entry)
    (GOLDEN / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    boot = {"mode": "browse", "capped": False, "rollouts": [_rollout_summary(ROOT, rid) for rid in ids]}
    (GOLDEN / "browse.json").write_text(normalize(_safe_json(boot)), encoding="utf-8")
    shell = normalize(_render_shell(ROOT.name, boot))
    (GOLDEN / "pages" / "browse.boot.txt").write_text(page_parts(shell), encoding="utf-8")
    (GOLDEN / "pages" / "browse.sha256").write_text(sha256(shell) + "\n")
    assert page_parts(shell) == normalize(_json_for_script(boot))

    static = [
        "branch-tree/claude-continue/hello-world-task__0000abcd",
        "branched/jobs/job/count-files__b1a2c3d4",
        "oracle-hello/2026-09-29__12-46-15/hello-2__f2da4092",
        "synthetic/hostile-text__00000001",
        "synthetic/__BENCHFLOW_PAYLOAD__",
        "synthetic/title-&-'quotes'__00000001",
        "synthetic/rubric-sibling__abcd1234",
        "synthetic/lineage-hand__00000001",
    ]
    pages = []
    for number, rid in enumerate(static):
        page = normalize(render_rollout(ROOT / rid))
        name = f"pages/{number:02d}"
        (GOLDEN / f"{name}.boot.txt").write_text(page_parts(page), encoding="utf-8")
        (GOLDEN / f"{name}.sha256").write_text(sha256(page) + "\n")
        pages.append({"id": rid, "page": name})
    (GOLDEN / "pages" / "index.json").write_text(json.dumps(pages, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def build_corpora() -> None:
    """Value-level corpora: float repr, float(), JSON, and timestamps."""
    from benchflow.trajectories.viewer.payload import _parse_json, _parse_ts
    from benchflow._utils.json_safe import dumps_finite

    rng = random.Random(20261010)
    floats = [
        0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1.5, 100.0, 1e15, 1e16, 1e17, 1e-4, 1e-5, 1e22, 1e23,
        9.999999999999999e22, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 2.0**53,
        2.0**53 + 2, 123456789012345680.0, 1234567890123456.0, 12345678901234567.0, 0.0001234,
        0.00001234, 3.141592653589793, 2.718281828459045, 1 / 3, 2 / 3, 1e300, 1e-300, 4.35, 0.07,
        59.99999999999999, 12.5, 31.499059915542603, 1786786863.5, 1786786863.25, 999999999999999.9,
        9999999999999998.0, 0.000123, 1e-7, 123e-20,
    ]
    for _ in range(800):
        bits = rng.getrandbits(64)
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if math.isfinite(value):
            floats.append(value)
    for _ in range(400):
        floats.append(round(rng.uniform(-1e6, 1e6), rng.randint(0, 8)))
        floats.append(rng.random() * 10 ** rng.randint(-30, 30))
    (GOLDEN / "floats.json").write_text(
        json.dumps([[struct.unpack("<Q", struct.pack("<d", v))[0].to_bytes(8, "big").hex(), repr(v)] for v in floats]) + "\n"
    )

    def py_float(text: str):
        try:
            value = float(text)
        except ValueError:
            return None
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)

    float_strings = [
        "1", "1.5", " 2.5 ", "\t7\n", "1_000", "1__0", "_1", "1_", "1_000.000_1", "1e1_0", "1_e5", "+3", "-0",
        "-0.0", "inf", "-Infinity", "INF", "+inf", "infinity", "nan", "NaN", "-nan", "+nan", "1e5", "1E-5", ".5",
        "5.", "0x10", "", " ", "1.5.2", "1e", "e5", "in f", "1e400", "-1e400", "1e-400", "00012",
        "0.000000000000000000000000000001", "123456789012345678901234567890", "1.7976931348623159e308",
        "4.9406564584124654e-324", "2.4703282292062328e-324", "  -  1", "1d5", "0b1", "\u00a01\u00a0",
        "\u20001", "\x1c1\x1d", " 1_0 ", "1_ 0", "+.5", ".", "-.e1", "1e+_5", "infinit", "nan(1)",
    ]
    (GOLDEN / "pyfloat.json").write_text(json.dumps([[s, py_float(s)] for s in float_strings], ensure_ascii=False) + "\n")

    json_texts = [
        '{"a": 1, "b": [1, 2.5, -0.0, 1e400, -1e400, NaN, Infinity, -Infinity], "c": "x\\u2028y\\ud800z\\udc00\\ud83d\\ude00"}',
        "[1.0, 100, 1e16, 12345678901234567890, -12345678901234567890123, 0.1, 1E5, 1e-7, 2.5e+300, 1.0e0]",
        '"\\u0000\\u001f\\b\\f\\n\\r\\t\\"\\\\\\/"',
        '{"a":1,"a":2,"b":3}',
        '{"a":1,}', "[1,]", "01", "1.", "-", "  []  ", "\ufeff{}", '"\x01"', '"\x7f"', "nul", "true", "false",
        "null", "NaN", "-NaN", "Infinity", "-Infinity", "1e5", "-0", "-0.0", "[-]", "[1 2]", '{"a" 1}', "{1: 2}",
        "[" * 64 + "]" * 64, "[" * 65 + "]" * 65, '{"\\ud800": 1}', '"\\ud800\\u0041"', '"\\ud800\\uZZZZ"',
        '"\\ud800\\u12"', '"\\uDBFF\\uDFFF"', '"\\ud800"', '"\\ud800\\ud800"', '"\\udc00\\ud800"', "1e309",
        "123.456e-500", "1E400", "-1e-400", "9007199254740993", "-9223372036854775808", "9223372036854775808",
        "18446744073709551616", " \t\n\r1\r\n", '"héllo 世界 🎉"', '"\\x41"', '"\\U0001F600"', '"abc', "[1, 2",
        '{"a": [1, {"b": null}], "c": {}}', '{"a": "\u2028\u2029\u0085"}', '"\\u00e9"', '"\\u00E9"', "0", "-01",
        "1.5e", "1.5e+", "1.5E-3", ".5", "+1", "[1,,2]", '{"a":1 "b":2}', '{"":""}', "[[], {}, [[]]]",
        '"tab\there"', "1 2", "[true, false, null]", "tru", "[1e1000]", '{"k": 1e-1000}', "-", "--1",
        "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "0.1e1", "1E+2", "[1.0, -1.0, 1.5, 0.0, 10.0]", '"\\/"',
    ]

    def json_case(text: str):
        value = _parse_json(text)
        if value is None:
            return None
        return dumps_finite(value, ensure_ascii=False).encode("utf-8", "replace").decode("utf-8")

    (GOLDEN / "json.json").write_text(json.dumps([[t, json_case(t)] for t in json_texts], ensure_ascii=False) + "\n")

    dates = [
        "2026-08-15", "20260815", "2026-W33", "2026W33", "2026-W33-6", "2026W336", "2026-W53-1", "2025-W53-1",
        "2026-W00-1", "2026-W33-8", "0001-01-01", "9999-12-31", "2026-02-29", "2024-02-29", "2026-00-10",
        "2026-08-32", "2026-0815", "202608-15", "2026-8-15", "0000-01-01", "9999-W52-6", "0001-W01-1", "2026W33-6",
    ]
    times = [
        "", "T09:41:03", "T09:41", "T09", "T094103", "T0941", "T09:41:03.5", "T09:41:03,123456789", "T09:41:03.",
        " 09:41:03", "x09:41:03", "\u00e909:41:03", "\U0001F60009:41", "T25:00", "T09:60", "T09:41:60",
        "T09:41:03:5", "T09:4103", "T0941:03", "T", "T9", "T09:41:03.1234567", "T09:41:03.12a", "T23:59:59.999999",
    ]
    zones = ["", "Z", "+00:00", "-05:00", "+0530", "+05", "+05:30:15", "+05:30:15.123", "+24:00", "+23:59", "+05:99",
             "-00:00", "z", "+5", "Z junk", "+05:30Z", "-", "+", "+0", "+123", "+05:3", "+00:00:00.5"]
    stamps: list[str] = []
    for date in dates:
        stamps.append(date)
        for time_part in times[1:6]:
            stamps.append(date + time_part + "Z")
    for time_part in times:
        for zone in zones:
            stamps.append("2026-08-15" + time_part + zone)
    stamps += ["2026-08-15T09:41:03+00:00\x00junk", "2026-08-15T09:41:03\x00", "abc", "2026-08", "2026-08-1", "1786786863",
               "2026-08-15T09:41:03+00:00 ", " 2026-08-15", "２０２６-08-15", "2026-08-15T０9:41"]
    aware = [[s, _parse_ts(s)] for s in stamps if not _is_naive(s)]
    (GOLDEN / "timestamps.json").write_text(json.dumps(aware, ensure_ascii=False) + "\n")

    # Naive stamps read as local time: computed in America/Los_Angeles.
    naive = [s for s in stamps if _is_naive(s)] + [
        "2026-03-08T02:30:00", "2026-03-08T01:59:59.999999", "2026-03-08T03:00:00", "2026-11-01T01:30:00",
        "2026-11-01T00:59:59", "2026-11-01T02:00:00", "2026-07-01T12:00:00.25", "2026-01-15T08:00:00",
        "2026-03-08", "2026-11-01",
    ]
    script = (
        "import json, os, sys, time\n"
        "os.environ['TZ'] = 'America/Los_Angeles'\n"
        "time.tzset()\n"
        "from benchflow.trajectories.viewer.payload import _parse_ts\n"
        "stamps = json.loads(sys.stdin.read())\n"
        "print(json.dumps([[s, _parse_ts(s)] for s in stamps], ensure_ascii=False))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], input=json.dumps(naive), capture_output=True, text=True, check=True
    ).stdout
    (GOLDEN / "timestamps-la.json").write_text(out)


def _is_naive(stamp: str) -> bool:
    """Whether a stamp that parses has no UTC offset (its value depends on TZ)."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(stamp).tzinfo is None
    except ValueError:
        return False


def main() -> None:
    os.environ.setdefault("TZ", "UTC")
    time.tzset()
    for folder in (SYN, BRANCHED, GOLDEN):
        if folder.exists():
            shutil.rmtree(folder)
    build_synthetic()
    asyncio.run(build_branched())
    build_goldens()
    build_corpora()
    print(f"wrote {sum(1 for _ in GOLDEN.rglob('*'))} golden files")


if __name__ == "__main__":
    main()
