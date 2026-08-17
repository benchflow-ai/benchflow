"""Chat-only completion capture (#988).

The complement of the zero-signal net in test_api_error_capture.py: an agent
that produced real model output (tokens, agent messages) but ended its turn
without a single tool call. Motivating fixture: a baseline rollout that
narrated "Proceeding to add PLAN.md with the required sections." (2,727
output tokens) and returned — previously recorded as a clean scored fail,
indistinguishable in aggregates from a genuine attempt.

Semantics under test: the diagnostic is VISIBILITY ONLY — reward and error
stay exactly as the verifier/agent left them (chat-only completion is agent
behavior, not infrastructure failure), unlike suspected_api_error which
nulls the reward.
"""

from types import SimpleNamespace

from benchflow.diagnostics import (
    DIAGNOSTIC_BY_FIELD,
    DIAGNOSTIC_REGISTRY,
    NoToolCallCompletionDiagnostic,
)
from benchflow.models import RunResult


def _chat_only_trajectory() -> list[dict]:
    return [
        {"type": "user_message", "text": "Add PLAN.md with the required sections."},
        {
            "type": "agent_message",
            "text": "Proceeding to add PLAN.md with the required sections.",
        },
    ]


class _DiagBag:
    def __init__(self):
        self.recorded = []

    def set(self, diag):
        self.recorded.append(diag)


def _rollout_double(
    *,
    error=None,
    executed_prompts=("p",),
    n_tool_calls=0,
    agent="claude-agent-acp",
    agent_env=None,
    trajectory=None,
    usage_metrics=None,
):
    from benchflow.rollout import Rollout

    r = Rollout.__new__(Rollout)
    r._error = error
    r._executed_prompts = list(executed_prompts)
    r._agent_env = agent_env or {"BENCHFLOW_PROVIDER_NAME": "litellm"}
    r._config = SimpleNamespace(
        agent=agent, model="claude-haiku-4-5-20251001", primary_agent=agent
    )
    r._usage_metrics = usage_metrics or {
        "total_tokens": 30000,
        "n_output_tokens": 2727,
    }
    r._n_tool_calls = n_tool_calls
    r._trajectory = _chat_only_trajectory() if trajectory is None else trajectory
    r._rewards = {"reward": 0.0}
    r._diagnostics = _DiagBag()
    return r


class TestDetection:
    def test_chat_only_completion_flagged_reward_and_error_untouched(self):
        r = _rollout_double()
        r._maybe_flag_no_tool_completion()
        assert len(r._diagnostics.recorded) == 1
        diag = r._diagnostics.recorded[0]
        assert isinstance(diag, NoToolCallCompletionDiagnostic)
        assert diag.n_output_tokens == 2727
        assert diag.n_agent_messages == 1
        assert diag.n_message_chars > 0
        # The whole point of the design: visibility only.
        assert r._rewards == {"reward": 0.0}
        assert r._error is None

    def test_run_with_tool_calls_never_flagged(self):
        r = _rollout_double(n_tool_calls=9)
        r._maybe_flag_no_tool_completion()
        assert r._diagnostics.recorded == []

    def test_errored_rollout_not_flagged(self):
        # api_error / suspected_api_error / timeout paths own their channels.
        r = _rollout_double(error="suspected provider api error: ...")
        r._maybe_flag_no_tool_completion()
        assert r._diagnostics.recorded == []

    def test_setup_failure_path_not_flagged(self):
        # No executed prompts -> the agent never ran (#389).
        r = _rollout_double(executed_prompts=())
        r._maybe_flag_no_tool_completion()
        assert r._diagnostics.recorded == []

    def test_oracle_never_flagged(self):
        r = _rollout_double(agent="oracle")
        r._maybe_flag_no_tool_completion()
        assert r._diagnostics.recorded == []

    def test_no_agent_messages_not_flagged(self):
        # Zero output AND zero messages is zero-signal/capture territory
        # (suspected_api_error, #982) — not a chat-only completion.
        r = _rollout_double(trajectory=[{"type": "user_message", "text": "p"}])
        r._maybe_flag_no_tool_completion()
        assert r._diagnostics.recorded == []

    def test_native_subscription_telemetry_gap_not_flagged(self):
        # Same exemption as the zero-signal heuristic (PR #886): flat-telemetry
        # agents look tool-free on every healthy run.
        r = _rollout_double(agent_env={"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"})
        r._maybe_flag_no_tool_completion()
        assert r._diagnostics.recorded == []

    def test_scraped_trajectory_with_tool_calls_not_flagged(self):
        # Salvage paths (e.g. the gemini scraped-trajectory fallback) rebuild
        # tool_call events the ACP session never counted: _n_tool_calls stays
        # 0 while the trajectory shows real tool activity. The trajectory is
        # authoritative — never flag those.
        trajectory = [
            *_chat_only_trajectory(),
            {
                "type": "tool_call",
                "tool_call_id": "call-1",
                "kind": "execute",
                "title": "python probe.py",
                "status": "completed",
                "content": [],
            },
        ]
        r = _rollout_double(trajectory=trajectory)
        assert r._n_tool_calls == 0
        r._maybe_flag_no_tool_completion()
        assert r._diagnostics.recorded == []


class TestPrecedence:
    def test_zero_token_case_still_routes_to_suspected_api_error(self):
        # The zero-signal net runs first and claims the error channel; the
        # chat-only flag must then decline (error is set).
        r = _rollout_double(
            usage_metrics={"total_tokens": 0, "n_output_tokens": 0},
            trajectory=[],
        )
        r._api_failure_summary_cached = None
        r._maybe_classify_api_error()
        assert "suspected provider api error" in (r._error or "")
        assert r._rewards is None
        r._maybe_flag_no_tool_completion()
        chat_only = [
            d
            for d in r._diagnostics.recorded
            if isinstance(d, NoToolCallCompletionDiagnostic)
        ]
        assert chat_only == []


class TestDiagnosticsRegistry:
    def test_registered(self):
        assert NoToolCallCompletionDiagnostic in DIAGNOSTIC_REGISTRY
        assert (
            DIAGNOSTIC_BY_FIELD["no_tool_call_completion_info"]
            is NoToolCallCompletionDiagnostic
        )

    def test_no_error_category(self):
        # Deliberate: chat-only completion is not an error, so it must never
        # surface an error_category or leave the scored-fail bucket.
        assert NoToolCallCompletionDiagnostic.category is None

    def test_format_issue(self):
        diag = NoToolCallCompletionDiagnostic(
            total_tokens=30000,
            n_output_tokens=2727,
            n_agent_messages=1,
            n_message_chars=55,
        )
        line = diag.format_issue("some-task")
        assert "chat-only" in line and "2727" in line and "0 tool calls" in line


class TestResultSurface:
    def test_run_result_defaults_false(self):
        assert RunResult(task_name="t").no_tool_completion is False

    def test_run_result_carries_flag(self):
        assert RunResult(task_name="t", no_tool_completion=True).no_tool_completion


class TestFreshRunSummaryEnrichment:
    """Fresh runs build summary rows from in-memory RolloutResult via
    rollout_result_payload(), which carries no diagnostic payloads — only the
    persisted result.json has them. The enrichment step must pull them back
    so fresh and resumed runs aggregate identically (PR #1025 review)."""

    def _evaluation_double(self, jobs_dir):
        from benchflow.evaluation import Evaluation

        ev = Evaluation.__new__(Evaluation)
        ev._jobs_dir = jobs_dir
        ev._job_name = "job"
        return ev

    def test_payload_gains_persisted_diagnostic_fields(self, tmp_path):
        import json

        rollout_dir = tmp_path / "job" / "r1"
        rollout_dir.mkdir(parents=True)
        info = {
            "total_tokens": 30000,
            "n_output_tokens": 2727,
            "n_agent_messages": 1,
            "n_message_chars": 55,
        }
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "timing": {"total": 12.5},
                    "no_tool_call_completion_info": info,
                    "suspected_api_error_info": None,
                }
            )
        )
        ev = self._evaluation_double(tmp_path)
        payload = {}
        result = RunResult(task_name="t", rollout_name="r1")
        ev._enrich_payload_with_persisted_fields(payload, result)
        assert payload["no_tool_call_completion_info"] == info
        assert payload["timing"] == {"total": 12.5}
        # Null diagnostics must not materialize as keys.
        assert "suspected_api_error_info" not in payload

    def test_missing_result_json_is_silent(self, tmp_path):
        ev = self._evaluation_double(tmp_path)
        payload = {}
        ev._enrich_payload_with_persisted_fields(
            payload, RunResult(task_name="t", rollout_name="absent")
        )
        assert payload == {}

    def test_existing_payload_fields_not_overwritten(self, tmp_path):
        import json

        rollout_dir = tmp_path / "job" / "r1"
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "result.json").write_text(
            json.dumps({"timing": {"total": 99.0}})
        )
        ev = self._evaluation_double(tmp_path)
        payload = {"timing": {"total": 1.0}}
        ev._enrich_payload_with_persisted_fields(
            payload, RunResult(task_name="t", rollout_name="r1")
        )
        assert payload["timing"] == {"total": 1.0}
