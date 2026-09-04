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
    ViewerSourceError,
    _diagnostic_keys,
    _discover_rollouts,
    _load_prompts,
    _render_acp_trajectory,
    _resolve_browse_rollout,
    _rollout_summary,
    _safe_json,
    _tool_content_texts,
    parse_source,
    render_jsonl_file,
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


def _nested_array_json(depth: int) -> str:
    return "[" * depth + "0" + "]" * depth


class TestPayloadContract:
    def test_all_five_event_types_normalize(self, tmp_path):
        """Guards PR #1034's canonical projection of every supported ACP event."""
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
        """Guards PR #1034 against dropping future or unknown ACP events."""
        rollout = _write_rollout(tmp_path, [{"type": "future_thing", "x": 1}])
        payload = _extract_payload(render_rollout(rollout))
        assert payload["steps"][0]["kind"] == "unknown"
        assert payload["steps"][0]["type"] == "future_thing"

    def test_tool_content_accepts_flat_and_diff_shapes(self):
        """Guards PR #1034's normalization of flat, nested, and diff tool output."""
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
        """Guards PR #1034 against script-tag breakout from trajectory content."""
        hostile = "</script><script>alert(1)</script>"
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": hostile}])
        page = render_rollout(rollout)
        # The raw sequence anywhere in the emitted page would terminate the
        # payload <script> tag early and execute the rest as live markup —
        # it must survive only in escaped form. Escaping every ``<`` also
        # prevents the HTML tokenizer's ``<!--<script>`` double-escaped state.
        assert hostile not in page
        assert "\\u003c/script\\u003e" in page

    def test_title_cannot_consume_the_payload_placeholder(self, tmp_path):
        """Guards PR #1034 against title-controlled template marker collision."""
        rollout = _write_rollout(
            tmp_path / "__BENCHFLOW_PAYLOAD__",
            [{"type": "agent_message", "text": "payload survived"}],
        )

        page = render_rollout(rollout)
        payload = _extract_payload(page)

        assert payload["rollout_name"] == "__BENCHFLOW_PAYLOAD__"
        assert payload["steps"][0]["text"] == "payload survived"

    def test_lone_surrogate_page_still_utf8_encodable(self, tmp_path):
        """Guards PR #1034 against lone-surrogate failures during UTF-8 serving."""
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
        """Guards PR #1034 against malformed timeout pending-call shapes."""
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

    def test_non_finite_values_emit_strict_json(self, tmp_path):
        """Guards PR #1034 against JSON.parse failures from NaN or Infinity."""
        rollout = _write_rollout(
            tmp_path,
            [
                {
                    "type": "tool_call",
                    "kind": "read",
                    "title": "file",
                    "status": "completed",
                    "ts": float("nan"),
                }
            ],
        )
        (rollout / "result.json").write_text(
            json.dumps(
                {
                    "rewards": {"reward": float("nan")},
                    "agent_result": {
                        "total_tokens": float("inf"),
                        "cost_usd": float("-inf"),
                    },
                }
            )
        )
        (rollout / "timing.json").write_text(
            json.dumps({"total": float("inf"), "agent_execution": 2.0})
        )

        payload = _extract_payload(render_rollout(rollout))

        assert payload["meta"]["reward"] is None
        assert payload["meta"]["usage"]["total_tokens"] is None
        assert payload["meta"]["usage"]["cost_usd"] is None
        assert payload["meta"]["timing"]["total"] is None
        assert "t" not in payload["steps"][0]
        assert json.loads(_safe_json({"nan": float("nan")})) == {"nan": None}

    def test_tool_status_is_whitelisted_in_both_renderers(self, tmp_path):
        """Guards PR #1034 against source-controlled status/CSS injection."""
        hostile = '"><script>globalThis.pwned=true</script>'
        events = [
            {
                "type": "tool_call",
                "kind": "other",
                "title": "WebSearch",
                "status": hostile,
            },
            {
                "type": "tool_call",
                "kind": "execute",
                "title": "ls",
                "status": "IN-PROGRESS",
            },
        ]
        rollout = _write_rollout(tmp_path / "rollout", events)
        payload = _extract_payload(render_rollout(rollout))
        assert [step["tool"]["status"] for step in payload["steps"]] == [
            "unknown",
            "in_progress",
        ]
        assert payload["steps"][0]["tool"]["hue"] == "search"

        session = tmp_path / "session.jsonl"
        session.write_text("\n".join(json.dumps(event) for event in events))
        legacy_page = render_jsonl_file(session)
        assert hostile not in legacy_page
        assert "acc-web" in legacy_page


class TestPromptLoading:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ({"prompt": "wrong top level"}, None),
            ("wrong top level", None),
            (42, None),
            (None, None),
            (
                ["text", 7, {"nested": "prompt"}, None],
                ["text", "7", '{"nested": "prompt"}', ""],
            ),
        ],
    )
    def test_prompt_loader_has_one_total_shape_boundary(self, tmp_path, raw, expected):
        """Guards PR #1034's canonical prompt loader against every JSON shape."""
        (tmp_path / "prompts.json").write_text(json.dumps(raw))
        assert _load_prompts(tmp_path) == expected

    def test_prompt_loader_is_shared_by_acp_and_legacy_paths(self, tmp_path):
        """Guards PR #1034 against ACP/legacy prompt-loading drift."""
        raw_prompts = [{"task": "go"}]

        acp = _write_rollout(
            tmp_path / "acp", [{"type": "agent_message", "text": "done"}]
        )
        (acp / "prompts.json").write_text(json.dumps(raw_prompts))
        acp_payload = _extract_payload(render_rollout(acp))
        assert acp_payload["steps"][0]["text"] == '{"task": "go"}'

        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "turn1.txt").write_text(
            json.dumps({"type": "system", "session_id": "s", "model": "m"})
        )
        (legacy / "prompts.json").write_text(json.dumps(raw_prompts))
        legacy_page = render_rollout(legacy)
        assert "{&quot;task&quot;: &quot;go&quot;}" in legacy_page


class TestDegradation:
    def test_binary_sidecar_files_degrade(self, tmp_path):
        """Guards PR #1034 against non-UTF-8 verifier sidecars blanking the page."""
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": "hi"}])
        (rollout / "result.json").write_bytes(b"\xff\xfe\x00 not utf8")
        vdir = rollout / "verifier"
        vdir.mkdir()
        (vdir / "test-stdout.txt").write_bytes(b"\xff\xfe\x00 binary")
        page = render_rollout(rollout)
        assert isinstance(page, str) and page

    def test_prompts_deduplicated_when_inline_user_messages_exist(self, tmp_path):
        """Guards PR #1034 against duplicating prompts already present in ACP."""
        rollout = _write_rollout(
            tmp_path, [{"type": "user_message", "text": "UNIQ-42"}]
        )
        page = _render_acp_trajectory(
            rollout,
            rollout / "trajectory" / "acp_trajectory.jsonl",
            prompts=["UNIQ-42"],
        )
        assert page.count("UNIQ-42") == 1

    @pytest.mark.parametrize(
        ("result", "timing", "ctrf"),
        [
            ([], [], {"results": []}),
            ({"rewards": []}, {"total": {"not": "numeric"}}, {"results": 1}),
            (
                {"agent_result": "not-an-object"},
                {"total": "not-a-number"},
                {"results": {"tests": "not-a-list"}},
            ),
        ],
    )
    def test_wrong_shaped_sidecars_degrade(self, tmp_path, result, timing, ctrf):
        """Guards PR #1034 against hostile result, timing, and CTRF shapes."""
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": "ok"}])
        (rollout / "result.json").write_text(json.dumps(result))
        (rollout / "timing.json").write_text(json.dumps(timing))
        verifier = rollout / "verifier"
        verifier.mkdir()
        (verifier / "ctrf.json").write_text(json.dumps(ctrf))

        payload = _extract_payload(render_rollout(rollout))

        assert payload["steps"][0]["text"] == "ok"
        assert payload["verifier"]["ctrf"] is None
        if isinstance(timing, dict):
            assert payload["meta"]["timing"]["total"] is None

    def test_malformed_ctrf_tests_are_normalized(self, tmp_path):
        """Guards PR #1034 against malformed fields inside otherwise-valid CTRF."""
        rollout = _write_rollout(tmp_path, [])
        verifier = rollout / "verifier"
        verifier.mkdir()
        (verifier / "ctrf.json").write_text(
            json.dumps(
                {
                    "results": {
                        "tests": [
                            {
                                "name": {"nested": "name"},
                                "status": "<style>bad</style>",
                                "duration": float("inf"),
                            },
                            "not-an-object",
                        ]
                    }
                }
            )
        )

        payload = _extract_payload(render_rollout(rollout))

        assert payload["verifier"]["ctrf"] == [
            {"name": '{"nested": "name"}', "status": "unknown", "duration": None}
        ]

    @pytest.mark.parametrize("depth", [80, 1200])
    def test_deep_jsonl_record_is_skipped_without_losing_later_records(
        self, tmp_path, depth
    ):
        """Guards PR #1034 against bounded and decoder-level JSON recursion."""
        deep_record = '{"type":"future","payload":' + _nested_array_json(depth) + "}"
        valid_record = json.dumps({"type": "user_message", "text": "safe"})

        rollout = tmp_path / f"rollout-{depth}"
        trajectory = rollout / "trajectory"
        trajectory.mkdir(parents=True)
        (trajectory / "acp_trajectory.jsonl").write_text(
            deep_record + "\n" + valid_record
        )
        payload = _extract_payload(render_rollout(rollout))
        assert [step["text"] for step in payload["steps"]] == ["safe"]

        session = tmp_path / f"session-{depth}.jsonl"
        session.write_text(deep_record + "\n" + valid_record)
        legacy_page = render_jsonl_file(session)
        assert "safe" in legacy_page
        assert "future" not in legacy_page

    def test_deep_sidecars_degrade_before_recursive_projection(self, tmp_path):
        """Guards PR #1034 against deeply nested JSON sidecar projection."""
        nested = _nested_array_json(80)
        rollout = _write_rollout(
            tmp_path, [{"type": "agent_message", "text": "still renders"}]
        )
        (rollout / "result.json").write_text('{"agent_timeout_info":' + nested + "}")
        (rollout / "timing.json").write_text('{"total":' + nested + "}")
        (rollout / "prompts.json").write_text("[" + nested + "]")
        verifier = rollout / "verifier"
        verifier.mkdir()
        (verifier / "ctrf.json").write_text('{"results":{"tests":' + nested + "}}")

        payload = _extract_payload(render_rollout(rollout))

        assert [step["text"] for step in payload["steps"]] == ["still renders"]
        assert payload["meta"]["errors"] == []
        assert payload["meta"]["timing"] is None
        assert payload["verifier"]["ctrf"] is None

    def test_safe_json_rejects_deep_or_cyclic_projection(self):
        """Guards PR #1034's final JSON projection from recursive structures."""
        deep: object = 0
        for _ in range(120):
            deep = [deep]
        assert _safe_json(deep) == "null"

        cyclic: list[object] = []
        cyclic.append(cyclic)
        assert _safe_json(cyclic) == "null"


class TestBrowseMode:
    def test_discovery_finds_nested_rollouts_and_skips_hidden(self, tmp_path):
        """Guards PR #1034's bounded nested discovery and hidden-dir policy."""
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
        """Guards PR #1034 against traversal through browse rollout ids."""
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
        """Guards PR #1034 by pinning exact discovery-whitelist enforcement."""
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
        """Guards PR #1034's explicit, configurable run-discovery cap."""
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
        """Guards PR #1034's supported dataset directory depth."""
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
        """Guards PR #1034's catalog summary projection from result metadata."""
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

    def test_catalog_and_detail_share_normalized_metadata(self, tmp_path):
        """Guards PR #1034 against catalog/detail metadata drift."""
        rollout = _write_rollout(
            tmp_path / "job" / "run", [{"type": "agent_message", "text": "ok"}]
        )
        (rollout / "result.json").write_text(
            json.dumps(
                {
                    "task_name": 1034,
                    "agent_name": ["wrong-shape"],
                    "agent": "fallback-agent",
                    "model": {"wrong": "shape"},
                    "skill_mode": "no-skill",
                    "rewards": {"reward": "1"},
                    "n_tool_calls": "4",
                    "agent_result": {
                        "n_tool_calls": "3",
                        "total_tokens": "42",
                        "cost_usd": "0.25",
                    },
                    "timing": {"total": 999},
                    "export_error": "could not export",
                }
            )
        )
        (rollout / "timing.json").write_text(
            json.dumps({"total": "12.5", "agent_execution": "2"})
        )

        detail = _extract_payload(render_rollout(rollout))["meta"]
        summary = _rollout_summary(tmp_path, "job/run")

        assert detail["task_name"] == summary["task_name"] == "1034"
        assert detail["agent_name"] == summary["agent_name"] == "fallback-agent"
        assert detail["model"] is summary["model"] is None
        assert detail["reward"] == summary["reward"] == 1.0
        assert detail["duration_sec"] == summary["duration_sec"] == 12.5
        assert detail["timing"] == {"total": 12.5, "agent_execution": 2.0}
        assert summary["total_tokens"] == 42
        assert summary["cost_usd"] == 0.25
        assert summary["n_tool_calls"] == 4
        assert summary["has_error"] is True
        assert {error["label"] for error in detail["errors"]} == {"export error"}

    def test_invalid_timing_sidecar_falls_back_to_embedded_timing(self, tmp_path):
        """Guards PR #1034's timing.json preference with a safe legacy fallback."""
        rollout = _write_rollout(tmp_path / "job" / "run", [])
        (rollout / "result.json").write_text(json.dumps({"timing": {"total": 7}}))
        (rollout / "timing.json").write_text(json.dumps(["wrong-shape"]))

        detail = _extract_payload(render_rollout(rollout))["meta"]
        summary = _rollout_summary(tmp_path, "job/run")

        assert detail["duration_sec"] == summary["duration_sec"] == 7.0


class TestSources:
    def test_parse_hf_specs(self):
        """Guards PR #1034 while preserving the accepted HF source spellings."""
        assert parse_source("hf://benchflow/skillsbench-trajectories-apr2026") == (
            HfDatasetSource("benchflow/skillsbench-trajectories-apr2026", None, "")
        )
        assert parse_source("hf://org/name/jobs/run-1") == HfDatasetSource(
            "org/name", None, "jobs/run-1"
        )
        assert parse_source("hf://org/name@abc123/jobs") == HfDatasetSource(
            "org/name", "abc123", "jobs"
        )
        assert parse_source("hf://org/name/") == HfDatasetSource("org/name", None, "")

    def test_parse_local_path(self):
        """Guards PR #1034's typed local-path source boundary."""
        src = parse_source("jobs/run/task__abc123")
        assert isinstance(src, LocalPathSource)
        assert src.path == Path("jobs/run/task__abc123")

    def test_parse_rejects_path_mangled_spelling(self):
        """Guards PR #1034 against silently accepting Path-mangled hf specs."""
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
            "hf://org/name//run",
            "hf://org//name/run",
            "hf://org/name/C:/Users/secret",
            "hf://org/name/jobs/*/run",
            "hf://org/name/jobs/[ab]/run",
        ],
    )
    def test_parse_rejects_malformed_specs(self, spec):
        """Guards PR #1034 against ambiguous or escaping HF source paths."""
        with pytest.raises(ValueError):
            parse_source(spec)

    def test_resolve_downloads_scoped_patterns(self, tmp_path, monkeypatch):
        """Guards PR #1034's subpath-scoped Hugging Face allowlist."""
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
        rollout_patterns = [p for p in calls["allow_patterns"] if "review" not in p]
        assert all(p.startswith("jobs/run-1/") for p in rollout_patterns)
        assert "jobs/run-1/trajectory/acp_trajectory.jsonl" in calls["allow_patterns"]
        # review reports sit beside the run, so the parent's review-* dirs are in scope
        assert "jobs/review*/**/review_report.json" in calls["allow_patterns"]
        assert not any("llm_trajectory" in p for p in calls["allow_patterns"])

    def test_resolve_raises_typed_error_for_missing_subpath(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #1034 against resolver-owned process exits."""
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            lambda **_kwargs: str(tmp_path),
        )

        with pytest.raises(ViewerSourceError, match="No such path"):
            resolve_hf_dataset(HfDatasetSource("org/name", None, "missing"))

    def test_resolve_wraps_download_failures(self, monkeypatch):
        """Guards PR #1034 with a typed source boundary for provider failures."""
        import huggingface_hub

        def fail_download(**_kwargs):
            raise OSError("offline")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_download)

        with pytest.raises(ViewerSourceError, match="Could not fetch") as exc_info:
            resolve_hf_dataset(HfDatasetSource("org/name", "rev", ""))
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_resolve_rechecks_subpath_containment(self, tmp_path, monkeypatch):
        """Guards PR #1034 against constructed HF sources escaping the snapshot."""
        import huggingface_hub

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = snapshot / "outside-link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are unavailable on this platform")
        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            lambda **_kwargs: str(snapshot),
        )

        with pytest.raises(ViewerSourceError, match="outside downloaded snapshot"):
            resolve_hf_dataset(HfDatasetSource("org/name", None, "outside-link"))

    def test_resolve_rejects_constructed_glob_scope_before_download(self, monkeypatch):
        """Guards PR #1034 against constructed HF sources widening downloads."""
        import huggingface_hub

        def unexpected_download(**_kwargs):
            pytest.fail("invalid subpaths must be rejected before snapshot_download")

        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            unexpected_download,
        )

        with pytest.raises(ViewerSourceError, match="Invalid dataset subpath"):
            resolve_hf_dataset(HfDatasetSource("org/name", None, "jobs/*"))

    def test_allowlist_is_precise_and_cannot_widen(self):
        """Guards PR #1034 against reintroducing broad HF artifact downloads."""
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
        """Guards PR #1034's finite event timestamp and duration projection."""
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
        """Guards PR #1034's wire compatibility when captures lack timestamps."""
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
        """Guards PR #1034 against malformed timestamps breaking rendering."""
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
        """Guards PR #1034 against diagnostic-registry drift."""
        from benchflow.diagnostics import DIAGNOSTIC_REGISTRY

        assert _diagnostic_keys() == tuple(d.field for d in DIAGNOSTIC_REGISTRY)
        # the 0.7.4 set must stay covered even as the registry grows
        assert set(_DIAGNOSTIC_KEYS_FALLBACK) <= set(_diagnostic_keys())

    def test_flag_without_error_renders_info_level(self, tmp_path):
        """Guards PR #1034's neutral presentation of non-error diagnostics."""
        rollout = _write_rollout(tmp_path, [{"type": "agent_message", "text": "hi"}])
        (rollout / "result.json").write_text(
            json.dumps({"idle_timeout_info": {"idle_timeout_sec": 30}})
        )
        payload = _extract_payload(render_rollout(rollout))
        (entry,) = payload["meta"]["errors"]
        assert entry["level"] == "info"

    def test_diagnostic_alongside_error_stays_error_level(self, tmp_path):
        """Guards PR #1034's error severity when diagnostics accompany failure."""
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
        """Guards PR #1034's canonical server-side tool classification."""
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
        """Guards PR #1034's full-fidelity unknown-event and diagnostic payloads."""
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
        """Guards PR #1034's explicit Python-to-JavaScript payload contract."""
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
            "rubric",
        }
        assert wire["steps"][0] == {
            "i": 1,
            "kind": "prompt",
            "label": "PROMPT 1",
            "text": "hello",
        }

    def test_step_model_is_a_discriminated_variant(self, tmp_path):
        """Guards PR #1034 against returning to an all-optional step model."""
        from benchflow.trajectories.viewer.models import PromptStep, ToolStep
        from benchflow.trajectories.viewer.payload import _build_acp_payload

        rollout = _write_rollout(
            tmp_path,
            [
                {"type": "user_message", "text": "go"},
                {
                    "type": "tool_call",
                    "kind": "read",
                    "title": "file",
                    "status": "completed",
                },
            ],
        )
        prompt, tool = _build_acp_payload(rollout, None).steps
        assert isinstance(prompt, PromptStep)
        assert isinstance(tool, ToolStep)
