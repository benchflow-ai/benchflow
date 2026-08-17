"""Tests for the interactive ACP trajectory renderer (viewer v2).

Covers the payload contract, escaping of untrusted trajectory content, and
degrade-don't-crash behavior on hostile or partial rollout artifacts. The
hostile cases mirror verified adversarial-review findings from the viewer-v2
prototype branch.
"""

import json
from pathlib import Path

from benchflow.trajectories.viewer import (
    _discover_rollouts,
    _render_acp_trajectory,
    _rollout_summary,
    _tool_content_texts,
    render_rollout,
)


def _write_rollout(tmp_path: Path, events: list[dict]) -> Path:
    traj = tmp_path / "trajectory"
    traj.mkdir(parents=True, exist_ok=True)
    (traj / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events)
    )
    return tmp_path


def _extract_payload(page: str) -> dict:
    data = page.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    boot = json.loads(data.replace("<\\/", "</"))
    assert boot["mode"] == "single"
    return boot["payload"]


class TestPayloadContract:
    def test_all_five_event_types_normalize(self, tmp_path):
        events = [
            {"type": "user_message", "text": "do it"},
            {"type": "agent_thought", "text": "hmm"},
            {
                "type": "tool_call",
                "tool_call_id": "c1",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "content": [{"type": "content", "content": {"type": "text", "text": "out"}}],
            },
            {"type": "agent_message", "text": "done"},
            {
                "type": "agent_timeout",
                "reason": "wall_clock_timeout",
                "timeout_sec": 5.0,
                "pending_tool_call_ids": ["c1"],
                "terminal_trajectory_complete": False,
            },
        ]
        rollout = _write_rollout(tmp_path, events)
        payload = _extract_payload(render_rollout(rollout))
        kinds = [s["kind"] for s in payload["steps"]]
        assert kinds == ["prompt", "thought", "tool", "message", "timeout"]
        assert payload["steps"][0]["label"] == "PROMPT 1"
        assert payload["steps"][2]["tool"]["content"] == ["out"]
        assert payload["steps"][4]["timeout"]["pending"] == ["c1"]

    def test_unknown_event_type_renders_generic_step(self, tmp_path):
        rollout = _write_rollout(tmp_path, [{"type": "future_thing", "x": 1}])
        payload = _extract_payload(render_rollout(rollout))
        assert payload["steps"][0]["kind"] == "unknown"
        assert payload["steps"][0]["type"] == "future_thing"

    def test_tool_content_accepts_flat_and_diff_shapes(self):
        texts = _tool_content_texts(
            [
                {"text": "flat"},
                {"type": "diff", "path": "a.py", "oldText": "x=1", "newText": "x=2"},
            ]
        )
        assert texts[0] == "flat"
        assert "--- old" in texts[1] and "+++ new" in texts[1] and "a.py" in texts[1]


class TestUntrustedContent:
    def test_script_breakout_is_escaped(self, tmp_path):
        hostile = '</script><script>alert(1)</script>'
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": hostile}])
        page = render_rollout(rollout)
        payload_zone = page.split('type="application/json">', 1)[1]
        assert "</script><script>" not in payload_zone.split("</script>", 1)[0]

    def test_lone_surrogate_page_still_utf8_encodable(self, tmp_path):
        # Guards the serve() write path: json.dumps(ensure_ascii=False) keeps
        # lone surrogates, which crash .encode()/write_text() if unsanitized.
        traj = tmp_path / "trajectory"
        traj.mkdir()
        (traj / "acp_trajectory.jsonl").write_text(
            '{"type":"agent_message","text":"lead \\ud800 tail"}'
        )
        page = render_rollout(tmp_path)
        page.encode("utf-8")  # must not raise

    def test_non_list_pending_tool_calls_coerced(self, tmp_path):
        rollout = _write_rollout(
            tmp_path,
            [
                {
                    "type": "agent_timeout",
                    "reason": "x",
                    "timeout_sec": 1.0,
                    "pending_tool_call_ids": "not-a-list",
                    "terminal_trajectory_complete": False,
                }
            ],
        )
        payload = _extract_payload(render_rollout(rollout))
        assert payload["steps"][0]["timeout"]["pending"] == []


class TestDegradation:
    def test_binary_sidecar_files_degrade(self, tmp_path):
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": "hi"}])
        (rollout / "result.json").write_bytes(b"\xff\xfe\x00 not utf8")
        vdir = rollout / "verifier"
        vdir.mkdir()
        (vdir / "test-stdout.txt").write_bytes(b"\xff\xfe\x00 binary")
        page = render_rollout(rollout)
        assert isinstance(page, str) and page

    def test_prompts_deduplicated_when_inline_user_messages_exist(self, tmp_path):
        rollout = _write_rollout(tmp_path, [{"type": "user_message", "text": "UNIQ-42"}])
        page = _render_acp_trajectory(
            rollout, rollout / "trajectory" / "acp_trajectory.jsonl", prompts=["UNIQ-42"]
        )
        assert page.count("UNIQ-42") == 1


class TestBrowseMode:
    def test_discovery_finds_nested_rollouts_and_skips_hidden(self, tmp_path):
        _write_rollout(tmp_path / "job-a" / "task-1__aaaa0000", [{"type": "agent_message", "text": "x"}])
        _write_rollout(tmp_path / "job-b" / "nested" / "task-2__bbbb0000", [{"type": "agent_message", "text": "y"}])
        _write_rollout(tmp_path / ".hidden" / "task-3__cccc0000", [{"type": "agent_message", "text": "z"}])
        (tmp_path / "not-a-rollout").mkdir()

        ids = _discover_rollouts(tmp_path)
        assert ids == [
            "job-a/task-1__aaaa0000",
            "job-b/nested/task-2__bbbb0000",
        ]

    def test_discovered_ids_resolve_inside_base_only(self, tmp_path):
        # The API resolves ids solely by exact membership in this list, so the
        # anti-traversal property reduces to: every id is a clean relative
        # path that stays under base.
        _write_rollout(tmp_path / "job" / "t__dddd0000", [{"type": "agent_message", "text": "x"}])
        for rid in _discover_rollouts(tmp_path):
            assert ".." not in rid.split("/")
            assert not rid.startswith("/")
            assert (tmp_path / rid).resolve().is_relative_to(tmp_path.resolve())

    def test_rollout_summary_reads_result_json(self, tmp_path):
        rollout = _write_rollout(tmp_path / "j" / "t__eeee0000", [{"type": "agent_message", "text": "x"}])
        (rollout / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "demo-task",
                    "rewards": {"reward": 1.0},
                    "agent_name": "gemini",
                    "skill_mode": "no-skill",
                }
            )
        )
        summary = _rollout_summary(tmp_path, "j/t__eeee0000")
        assert summary["task_name"] == "demo-task"
        assert summary["reward"] == 1.0
        assert summary["agent_name"] == "gemini"
        assert summary["has_error"] is False
