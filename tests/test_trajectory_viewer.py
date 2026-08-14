"""Slice-one trace viewer: the payload Python produces and the page it embeds.

Rendering lives in ``viewer_assets/render.js`` and is covered by
``tests/test_trajectory_viewer_render.py``. Here the subject is the payload —
the contract between the two, and what slice 5 will publish as
``viewer_data/<run_id>.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from benchflow.rollout import _build_rollout_result
from benchflow.trajectories.viewer import (
    PAYLOAD_SCHEMA_VERSION,
    build_run_payload,
    render_rollout,
)


def _canonical_rollout(
    rollout_dir: Path,
    *,
    trajectory: list[dict[str, Any]],
    rewards: dict[str, Any] | None = None,
    error: str | None = None,
    verifier_error: str | None = None,
    partial_trajectory: bool = False,
) -> Path:
    _build_rollout_result(
        rollout_dir,
        task_name="viewer-task",
        rollout_name=rollout_dir.name,
        agent="codex-acp",
        agent_name="Codex",
        model="openai/gpt-5.6",
        n_tool_calls=sum(event.get("type") == "tool_call" for event in trajectory),
        prompts=[
            str(event.get("text") or "")
            for event in trajectory
            if event.get("type") == "user_message"
        ],
        error=error,
        verifier_error=verifier_error,
        trajectory=trajectory,
        partial_trajectory=partial_trajectory,
        trajectory_source="partial_acp" if partial_trajectory else "acp",
        rewards=rewards,
        started_at=datetime.now() - timedelta(seconds=21),
        timing={
            "environment_setup": 2.0,
            "agent_setup": 3.0,
            "agent_execution": 12.0,
            "verifier": 4.0,
        },
        n_input_tokens=1234,
        n_output_tokens=321,
        n_cache_creation_tokens=67,
        n_cache_read_tokens=890,
        total_tokens=2512,
        cost_usd=0.012345,
        usage_source="provider_response",
    )
    return rollout_dir


def _events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for turn in payload["turns"] for event in turn["events"]]


def _tool_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in _events(payload) if event["type"] == "tool_call"]


def test_payload_reads_canonical_acp_and_agent_result(tmp_path: Path) -> None:
    long_output = "canonical-tool-output-needle\n" + "x" * 1800 + "TAIL-KEPT"
    _canonical_rollout(
        tmp_path,
        rewards={"reward": 0.0},
        trajectory=[
            {"type": "user_message", "text": "Solve <this> safely"},
            {"type": "agent_thought", "text": "I should inspect the files."},
            {
                "type": "tool_call",
                "tool_call_id": "tc-1",
                "kind": "bash",
                "title": "pytest -q",
                "status": "completed",
                "content": [{"type": "text", "text": long_output}],
            },
            {"type": "agent_message", "text": "The implementation is ready."},
        ],
    )
    # Canonical ACP must win even when a legacy stream file is also present.
    (tmp_path / "turn1.txt").write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "legacy-only-marker"}]
                },
            }
        )
    )
    result_data = json.loads((tmp_path / "result.json").read_text())
    result_data["final_metrics"] = {
        key: 987654321
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "cache_creation_tokens",
        )
    }
    (tmp_path / "result.json").write_text(json.dumps(result_data))

    payload = build_run_payload(tmp_path)
    assert payload is not None

    assert payload["schema_version"] == PAYLOAD_SCHEMA_VERSION
    assert payload["source"] == "acp"
    assert [event["type"] for event in _events(payload)] == [
        "user_message",
        "agent_thought",
        "tool_call",
        "agent_message",
    ]
    # The whole observation survives normalization; only rendering folds it.
    assert _tool_events(payload)[0]["blocks"] == [{"kind": "text", "text": long_output}]
    assert payload["reward"] == 0.0
    assert payload["status"] == {"slug": "failed", "label": "Failed"}
    # agent_result, never final_metrics: the latter drops cache creation.
    assert payload["usage"] == {
        "input": 1234,
        "output": 321,
        "cache_creation": 67,
        "cache_read": 890,
        "total": 2512,
        "cost_usd": pytest.approx(0.012345),
        "source": "provider_response",
        "price_source": None,
    }
    assert payload["timing"]["environment_setup"] == 2.0
    assert payload["timing"]["verifier"] == 4.0
    assert payload["artifacts"]["trajectory"] == "trajectory/acp_trajectory.jsonl"
    assert "legacy-only-marker" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("rewards", "error", "verifier_error", "extra_events", "expected"),
    [
        ({"reward": 1.0}, None, None, [], "passed"),
        ({"reward": 0.5}, None, None, [], "failed"),
        (None, None, None, [], "not-scored"),
        (None, "agent crashed", None, [], "errored"),
        (None, None, "verifier crashed", [], "verifier-errored"),
        (
            {"reward": 0.0},
            "Agent prompt exceeded wall-clock budget 900s",
            None,
            [
                {
                    "type": "agent_timeout",
                    "reason": "wall_clock_timeout",
                    "timeout_sec": 900.0,
                    "pending_tool_call_ids": ["tc-pending"],
                    "terminal_trajectory_complete": False,
                }
            ],
            "timeout",
        ),
    ],
)
def test_status_follows_result_evidence_not_trace_guessing(
    tmp_path: Path,
    rewards: dict[str, Any] | None,
    error: str | None,
    verifier_error: str | None,
    extra_events: list[dict[str, Any]],
    expected: str,
) -> None:
    _canonical_rollout(
        tmp_path,
        rewards=rewards,
        error=error,
        verifier_error=verifier_error,
        partial_trajectory=bool(extra_events),
        trajectory=[{"type": "user_message", "text": "solve"}, *extra_events],
    )

    payload = build_run_payload(tmp_path)
    assert payload is not None

    assert payload["status"]["slug"] == expected
    assert payload["reward"] == (rewards or {}).get("reward")
    titles = [notice["title"] for notice in payload["notices"]]
    if error:
        assert "Agent error" in titles
        assert payload["notices"][0]["body"] == error
    if verifier_error:
        assert "Verifier error" in titles
    if extra_events:
        assert "Partial trajectory" in titles
        timeout_event = _events(payload)[-1]
        assert timeout_event["timeout_sec"] == 900.0
        assert timeout_event["pending_tool_call_ids"] == ["tc-pending"]


def test_status_reclassifies_a_legacy_result_without_error_category(
    tmp_path: Path,
) -> None:
    """A pre-#503 rollout has no error_category; the shared classifier fills in."""
    _canonical_rollout(
        tmp_path,
        rewards={"reward": 0.0},
        error="Agent prompt exceeded wall-clock budget 900s",
        trajectory=[{"type": "user_message", "text": "solve"}],
    )
    result_data = json.loads((tmp_path / "result.json").read_text())
    del result_data["error_category"]
    (tmp_path / "result.json").write_text(json.dumps(result_data))

    payload = build_run_payload(tmp_path)
    assert payload is not None
    assert payload["status"]["slug"] == "timeout"


def test_setup_events_group_apart_from_numbered_turns(tmp_path: Path) -> None:
    _canonical_rollout(
        tmp_path,
        rewards={"reward": 1.0},
        trajectory=[
            {
                "type": "oracle",
                "command": "bash oracle/solve.sh",
                "return_code": 0,
                "stdout": "oracle complete",
            },
            {"type": "user_message", "text": "first"},
            {"type": "agent_message", "text": "done"},
            {"type": "user_message", "text": "second"},
        ],
    )

    payload = build_run_payload(tmp_path)
    assert payload is not None

    assert [turn["number"] for turn in payload["turns"]] == [None, 1, 2]
    setup = payload["turns"][0]["events"][0]
    assert setup["type"] == "oracle"
    assert setup["status"] == "completed"
    assert setup["blocks"] == [{"kind": "text", "text": "oracle complete"}]
    assert [event["type"] for event in payload["turns"][1]["events"]] == [
        "user_message",
        "agent_message",
    ]


def test_every_content_block_of_one_tool_call_survives(tmp_path: Path) -> None:
    """Slice 1's first contract: no observed block is silently dropped."""
    _canonical_rollout(
        tmp_path,
        rewards={"reward": 1.0},
        trajectory=[
            {"type": "user_message", "text": "edit it"},
            {
                "type": "tool_call",
                "tool_call_id": "edit-1",
                "kind": "edit",
                "title": "main.py",
                "status": "completed",
                "content": [
                    {"type": "content", "content": {"type": "text", "text": "patched"}},
                    # An ACP diff renders to no text at all via content_blocks_to_text.
                    {
                        "type": "diff",
                        "path": "/app/main.py",
                        "oldText": "value = 1",
                        "newText": "value = 2",
                    },
                    # An image sibling must not suppress the text blocks.
                    {"type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo="},
                    # A text-bearing resource is text, not binary.
                    {
                        "type": "resource",
                        "resource": {
                            "uri": "file:///app/main.py",
                            "mimeType": "text/plain",
                            "text": "resource-text-needle",
                        },
                    },
                    # An unknown provider block keeps its structure as JSON.
                    {"type": "chart", "series": [1, 2, 3]},
                ],
            },
        ],
    )

    payload = build_run_payload(tmp_path)
    assert payload is not None
    blocks = _tool_events(payload)[0]["blocks"]

    assert blocks[0] == {"kind": "text", "text": "patched"}
    assert blocks[1] == {
        "kind": "diff",
        "path": "/app/main.py",
        "old": "value = 1",
        "new": "value = 2",
    }
    assert blocks[2] == {"kind": "binary"}
    assert blocks[3] == {"kind": "text", "text": "resource-text-needle"}
    assert blocks[4]["kind"] == "json"
    assert '"series"' in blocks[4]["text"]
    assert "iVBORw0KGgo=" not in json.dumps(payload)


def test_one_runaway_observation_is_clipped_and_points_at_the_artifact(
    tmp_path: Path,
) -> None:
    body = "\n".join(f"line-{index}" for index in range(60_000))
    _canonical_rollout(
        tmp_path,
        rewards={"reward": 0.0},
        trajectory=[
            {"type": "user_message", "text": "run the suite"},
            {
                "type": "tool_call",
                "tool_call_id": "tc-big",
                "kind": "bash",
                "title": "pytest -q",
                "status": "failed",
                "content": [
                    {
                        "type": "content",
                        "content": {
                            "type": "text",
                            "text": f"HEAD-NEEDLE\n{body}\nFAILED-TAIL-NEEDLE",
                        },
                    }
                ],
            },
        ],
    )

    page = render_rollout(tmp_path)
    payload = build_run_payload(tmp_path)
    assert payload is not None
    block = _tool_events(payload)[0]["blocks"][0]

    assert block["text"].startswith("HEAD-NEEDLE")
    assert block["text"].endswith("FAILED-TAIL-NEEDLE")
    assert len(block["text"]) == 100_000
    assert block["clip"]["dropped"] > 400_000
    assert block["clip"]["artifact"] == "trajectory/acp_trajectory.jsonl"
    assert len(page) < 500_000
    # Whatever the page dropped is still on disk.
    acp = (tmp_path / "trajectory" / "acp_trajectory.jsonl").read_text()
    assert acp.count("line-") > 59_000


def test_stream_json_fallback_is_used_only_without_acp(tmp_path: Path) -> None:
    (tmp_path / "prompts.json").write_text(json.dumps(["legacy prompt"]))
    stream = [
        {"type": "system", "session_id": "session-1", "model": "claude-test"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "legacy thought"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "printf hello"},
                    },
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "legacy-output-needle",
                    }
                ]
            },
        },
        {"type": "result", "total_cost_usd": 0.25, "result": "done"},
    ]
    (tmp_path / "turn1.txt").write_text(
        "\n".join(json.dumps(event) for event in stream)
    )

    payload = build_run_payload(tmp_path)
    assert payload is not None

    assert payload["source"] == "stream-json"
    assert payload["meta"]["model"] == "claude-test"
    assert payload["usage"]["cost_usd"] == pytest.approx(0.25)
    assert payload["artifacts"]["trajectory"] is None
    tool_event = _tool_events(payload)[0]
    assert tool_event["title"] == "printf hello"
    assert tool_event["status"] == "completed"
    assert tool_event["blocks"] == [{"kind": "text", "text": "legacy-output-needle"}]
    assert [event["type"] for event in _events(payload)] == [
        "user_message",
        "agent_thought",
        "tool_call",
        "agent_message",
    ]


def test_directory_without_a_trajectory_has_no_payload(tmp_path: Path) -> None:
    assert build_run_payload(tmp_path) is None


def test_document_inlines_assets_and_neutralizes_markup_in_the_payload(
    tmp_path: Path,
) -> None:
    """Tool output routinely contains ``</script>``; the island must survive it."""
    _canonical_rollout(
        tmp_path,
        rewards={"reward": 1.0},
        trajectory=[
            {"type": "user_message", "text": "grep the page"},
            {
                "type": "tool_call",
                "tool_call_id": "tc-1",
                "kind": "bash",
                "title": "cat index.html",
                "status": "completed",
                "content": [
                    {
                        "type": "text",
                        "text": "</script><img src=x onerror=alert(1)>",
                    }
                ],
            },
        ],
    )

    page = render_rollout(tmp_path)

    assert "</script><img" not in page
    assert "\\u003c/script\\u003e" in page
    island = page.split('<script id="bf-run-data" type="application/json">')[1]
    island = island.split("</script>")[0]
    assert json.loads(island)["schema_version"] == PAYLOAD_SCHEMA_VERSION
    assert "__BF_" not in page
    assert "data:font/woff2;base64," in page
    assert "benchflow-viewer-theme" in page
    assert 'id="bf-app"' in page
    assert "<noscript>" in page


def test_job_directory_lists_its_rollouts_without_a_run_payload(tmp_path: Path) -> None:
    rollout = tmp_path / "task-a__trial-1"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "user_message", "text": "hi"})
    )

    page = render_rollout(tmp_path)

    assert "job directory" in page
    assert "task-a__trial-1" in page
    # No mount point → boot.js must not try to render a run page here.
    assert 'id="bf-app"' not in page
    assert "Slice" not in page
