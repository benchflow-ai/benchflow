"""Tests for the interactive ACP trajectory renderer (viewer v2).

Covers the payload contract, escaping of untrusted trajectory content, and
degrade-don't-crash behavior on hostile or partial rollout artifacts. The
hostile cases mirror verified adversarial-review findings from the viewer-v2
prototype branch.
"""

import json
from pathlib import Path

import pytest

from benchflow.trajectories.viewer import (
    _DIAGNOSTIC_KEYS_FALLBACK,
    _HF_VIEWER_FILES,
    HfDatasetSource,
    LocalPathSource,
    _diagnostic_keys,
    _discover_rollouts,
    _render_acp_trajectory,
    _resolve_browse_rollout,
    _rollout_summary,
    _tool_content_texts,
    parse_source,
    render_rollout,
    resolve_hf_dataset,
)
from benchflow.trajectories.viewer.payload import VERIFIER_SIDECARS


def _write_rollout(tmp_path: Path, events: list[dict]) -> Path:
    traj = tmp_path / "trajectory"
    traj.mkdir(parents=True, exist_ok=True)
    (traj / "acp_trajectory.jsonl").write_text("\n".join(json.dumps(e) for e in events))
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
                "content": [
                    {"type": "content", "content": {"type": "text", "text": "out"}}
                ],
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
        hostile = "</script><script>alert(1)</script>"
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": hostile}])
        page = render_rollout(rollout)
        # The raw sequence anywhere in the emitted page would terminate the
        # payload <script> tag early and execute the rest as live markup —
        # it must survive only in escaped form. (This assertion fails if the
        # "</" → "<\\/" escaping in _render_shell is removed.)
        assert hostile not in page
        assert "<\\/script>" in page

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
        rollout = _write_rollout(
            tmp_path, [{"type": "user_message", "text": "UNIQ-42"}]
        )
        page = _render_acp_trajectory(
            rollout,
            rollout / "trajectory" / "acp_trajectory.jsonl",
            prompts=["UNIQ-42"],
        )
        assert page.count("UNIQ-42") == 1


class TestBrowseMode:
    def test_discovery_finds_nested_rollouts_and_skips_hidden(self, tmp_path):
        _write_rollout(
            tmp_path / "job-a" / "task-1__aaaa0000",
            [{"type": "agent_message", "text": "x"}],
        )
        _write_rollout(
            tmp_path / "job-b" / "nested" / "task-2__bbbb0000",
            [{"type": "agent_message", "text": "y"}],
        )
        _write_rollout(
            tmp_path / ".hidden" / "task-3__cccc0000",
            [{"type": "agent_message", "text": "z"}],
        )
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
        _write_rollout(
            tmp_path / "job" / "t__dddd0000", [{"type": "agent_message", "text": "x"}]
        )
        for rid in _discover_rollouts(tmp_path):
            assert ".." not in rid.split("/")
            assert not rid.startswith("/")
            assert (tmp_path / rid).resolve().is_relative_to(tmp_path.resolve())

    def test_api_id_resolution_enforces_whitelist_membership(self, tmp_path):
        base = tmp_path / "base"
        _write_rollout(
            base / "inside" / "run__aaaa0000", [{"type": "agent_message", "text": "x"}]
        )
        # A REAL rollout outside the served base: reachable as a path, valid
        # in shape — it must still never resolve, because access is granted
        # by whitelist membership, not by the path existing. (This fails if
        # the membership check in _resolve_browse_rollout is removed.)
        _write_rollout(
            tmp_path / "outside" / "secret__bbbb0000",
            [{"type": "agent_message", "text": "s"}],
        )

        inside = _resolve_browse_rollout(base, "inside/run__aaaa0000")
        assert inside == base / "inside" / "run__aaaa0000"
        assert _resolve_browse_rollout(base, "../outside/secret__bbbb0000") is None
        assert _resolve_browse_rollout(base, None) is None

    def test_discovery_cap_env_override(self, tmp_path, monkeypatch):
        for i in range(4):
            _write_rollout(
                tmp_path / f"job-{i}" / f"t__{i:04d}0000",
                [{"type": "agent_message", "text": "x"}],
            )
        monkeypatch.setenv("BENCHFLOW_VIEWER_MAX_RUNS", "2")
        # Exact prefix, not just the count: the handler's ids[:cap] slice and
        # _resolve_browse_rollout's membership set both rely on the capped
        # scan being a deterministic prefix of the full scan.
        assert _discover_rollouts(tmp_path) == [
            "job-0/t__00000000",
            "job-1/t__00010000",
        ]
        assert _discover_rollouts(tmp_path, cap=10) == [
            "job-0/t__00000000",
            "job-1/t__00010000",
            "job-2/t__00020000",
            "job-3/t__00030000",
        ]

    def test_discovery_reaches_dataset_root_depth(self, tmp_path):
        # HF trajectory datasets nest one level deeper than local job dirs:
        # <root>/jobs/<run>/<timestamp>/<rollout>.
        _write_rollout(
            tmp_path / "jobs" / "run-a" / "2026-04-22__01-27-25" / "t__ffff0000",
            [{"type": "agent_message", "text": "x"}],
        )
        assert _discover_rollouts(tmp_path) == [
            "jobs/run-a/2026-04-22__01-27-25/t__ffff0000"
        ]

    def test_rollout_summary_reads_result_json(self, tmp_path):
        rollout = _write_rollout(
            tmp_path / "j" / "t__eeee0000", [{"type": "agent_message", "text": "x"}]
        )
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


class TestSources:
    def test_parse_hf_specs(self):
        assert parse_source("hf://benchflow/skillsbench-trajectories-apr2026") == (
            HfDatasetSource("benchflow/skillsbench-trajectories-apr2026", None, "")
        )
        assert parse_source("hf://org/name/jobs/run-1") == HfDatasetSource(
            "org/name", None, "jobs/run-1"
        )
        assert parse_source("hf://org/name@abc123/jobs") == HfDatasetSource(
            "org/name", "abc123", "jobs"
        )

    def test_parse_local_path(self):
        src = parse_source("jobs/run/task__abc123")
        assert isinstance(src, LocalPathSource)
        assert src.path == Path("jobs/run/task__abc123")

    def test_parse_rejects_path_mangled_spelling(self):
        # A pathlib.Path round trip collapses hf:// into hf:/ — that spelling
        # must be rejected with a pointer, never silently accepted.
        with pytest.raises(ValueError, match="hf://"):
            parse_source("hf:/org/name/sub")

    @pytest.mark.parametrize(
        "spec",
        [
            "hf://just-one-part",
            "hf://org/name/../escape",
            "hf://org/name/a/../b",
            "hf://org/na me",
            "hf://org/name@",
            "hf://org/name/sub\\path",
        ],
    )
    def test_parse_rejects_malformed_specs(self, spec):
        with pytest.raises(ValueError):
            parse_source(spec)

    def test_resolve_downloads_scoped_patterns(self, tmp_path, monkeypatch):
        calls = {}

        def fake_snapshot_download(*, repo_id, repo_type, revision, allow_patterns):
            calls.update(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                allow_patterns=allow_patterns,
            )
            (tmp_path / "jobs" / "run-1").mkdir(parents=True)
            return str(tmp_path)

        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", fake_snapshot_download
        )
        root = resolve_hf_dataset(HfDatasetSource("org/name", None, "jobs/run-1"))
        assert root == tmp_path / "jobs" / "run-1"
        assert calls["repo_id"] == "org/name"
        assert calls["repo_type"] == "dataset"
        assert all(p.startswith("jobs/run-1/") for p in calls["allow_patterns"])
        assert "jobs/run-1/trajectory/acp_trajectory.jsonl" in calls["allow_patterns"]
        assert not any("llm_trajectory" in p for p in calls["allow_patterns"])

    def test_allowlist_is_precise_and_cannot_widen(self):
        # Regression for the review finding that verifier/* pulled ~31 MB of
        # unconsumed artifacts: every allowlist entry is an exact path (the
        # only wildcard anywhere is the **/ recursion prefix added at
        # download time), and the verifier portion is exactly the set
        # payload._load_verifier reads — derived from VERIFIER_SIDECARS, so
        # widening one without the other fails here.
        for name in _HF_VIEWER_FILES:
            assert "*" not in name, name
        verifier_entries = {n for n in _HF_VIEWER_FILES if n.startswith("verifier/")}
        assert verifier_entries == {f"verifier/{n}" for n in VERIFIER_SIDECARS}
        assert set(VERIFIER_SIDECARS) == {
            "reward.txt",
            "test-stdout.txt",
            "test-stderr.txt",
            "ctrf.json",
        }


class TestTimestamps:
    """Forward-compat passthrough for the capture-side timestamp proposal
    (benchflow#1033): render when present, invisible when absent."""

    def test_timestamps_pass_through_when_present(self, tmp_path):
        events = [
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
        ]
        payload = _extract_payload(render_rollout(_write_rollout(tmp_path, events)))
        s0, s1 = payload["steps"]
        assert s1["t"] - s0["t"] == pytest.approx(2.0)
        assert s1["dur"] == pytest.approx(2.5)

    def test_no_timestamp_keys_when_absent(self, tmp_path):
        # Today's captures carry none — steps must not grow t/dur keys.
        events = [
            {"type": "agent_message", "text": "hi"},
            {
                "type": "tool_call",
                "tool_call_id": "c",
                "kind": "read",
                "title": "x",
                "status": "completed",
                "content": [],
            },
        ]
        payload = _extract_payload(render_rollout(_write_rollout(tmp_path, events)))
        for step in payload["steps"]:
            assert "t" not in step and "dur" not in step

    def test_unparseable_timestamps_ignored(self, tmp_path):
        events = [
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
        ]
        payload = _extract_payload(render_rollout(_write_rollout(tmp_path, events)))
        for step in payload["steps"]:
            assert "t" not in step and "dur" not in step


class TestDiagnostics:
    def test_keys_derive_from_registry(self):
        from benchflow.diagnostics import DIAGNOSTIC_REGISTRY

        assert _diagnostic_keys() == tuple(d.field for d in DIAGNOSTIC_REGISTRY)
        # the 0.7.4 set must stay covered even as the registry grows
        assert set(_DIAGNOSTIC_KEYS_FALLBACK) <= set(_diagnostic_keys())

    def test_flag_without_error_renders_info_level(self, tmp_path):
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": "hi"}])
        (rollout / "result.json").write_text(
            json.dumps({"idle_timeout_info": {"idle_timeout_sec": 30}})
        )
        payload = _extract_payload(render_rollout(rollout))
        (entry,) = payload["meta"]["errors"]
        assert entry["level"] == "info"

    def test_diagnostic_alongside_error_stays_error_level(self, tmp_path):
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": "hi"}])
        (rollout / "result.json").write_text(
            json.dumps(
                {
                    "error": "agent timed out",
                    "agent_timeout_info": {"timeout_sec": 900},
                }
            )
        )
        payload = _extract_payload(render_rollout(rollout))
        levels = {e["level"] for e in payload["meta"]["errors"] if "level" in e}
        assert levels == {"error"}


class TestTypedContract:
    """The models.py contract is the single Python↔JS boundary."""

    def test_tool_hue_is_classified_server_side(self, tmp_path):
        events = [
            {
                "type": "tool_call",
                "tool_call_id": "c1",
                "kind": "other",
                "title": "ToolSearch",
                "status": "completed",
                "content": [],
            },
            {
                "type": "tool_call",
                "tool_call_id": "c2",
                "kind": "delete",
                "title": "remove temp dir",
                "status": "completed",
                "content": [],
            },
            {
                "type": "tool_call",
                "tool_call_id": "c3",
                "kind": "mystery",
                "title": "??",
                "status": "completed",
                "content": [],
            },
        ]
        payload = _extract_payload(render_rollout(_write_rollout(tmp_path, events)))
        hues = [s["tool"]["hue"] for s in payload["steps"]]
        assert hues == ["search", "edit", "other"]

    def test_unknown_events_and_diagnostics_ship_untruncated(self, tmp_path):
        big = "x" * 5000
        rollout = _write_rollout(tmp_path, [{"type": "future_thing", "blob": big}])
        (rollout / "result.json").write_text(
            json.dumps({"error": "boom", "agent_timeout_info": {"detail": big}})
        )
        payload = _extract_payload(render_rollout(rollout))
        assert big in payload["steps"][0]["text"]
        (banner,) = [
            e for e in payload["meta"]["errors"] if e["label"] == "agent timeout"
        ]
        assert big in banner["text"]

    def test_payload_roundtrip_shape(self, tmp_path):
        from benchflow.trajectories.viewer.payload import _build_acp_payload

        rollout = _write_rollout(tmp_path, [{"type": "user_message", "text": "hello"}])
        typed = _build_acp_payload(rollout, None)
        wire = typed.to_payload()
        assert set(wire) == {
            "schema_version",
            "rollout_name",
            "meta",
            "steps",
            "verifier",
        }
        assert wire["steps"][0] == {
            "i": 1,
            "kind": "prompt",
            "label": "PROMPT 1",
            "text": "hello",
        }
