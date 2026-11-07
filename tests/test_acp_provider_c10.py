"""The native-only sibling omitted by 585a8aea."""

from types import SimpleNamespace

import pytest

from benchflow.acp.session import ACPSession
from tests.test_session_cost import _log, _response, _state, _unpriced_claude_rollout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other", ["codex-acp", "gemini", "muse-code", "vendor/codex-acp"]
)
@pytest.mark.parametrize("order", [(False, True), (True, False), (False, True, False)])
@pytest.mark.parametrize("proxy_covered", [False, True])
async def test_native_mixture_never_gets_claude_only_price(
    tmp_path, monkeypatch, other, order, proxy_covered
):
    home = tmp_path / "sandbox-home"
    _log(
        home / ".claude" / "projects",
        [
            _response("dummy-message", input_tokens=160, output_tokens=40),
            _state(0.01, input_tokens=160, output_tokens=40),
        ],
    )
    rollout, trial = _unpriced_claude_rollout(tmp_path, monkeypatch, home, {})
    rollout._native_usage_metrics = {}
    for foreign in order:
        rollout._native_usage_proxy_covered = proxy_covered and foreign
        rollout._active_role = SimpleNamespace(
            agent=other if foreign else "claude-agent-acp"
        )
        rollout._native_usage_checkpoint = None
        rollout._session = ACPSession("dummy")
        rollout._session.record_prompt_usage(
            SimpleNamespace(
                input_tokens=80 if foreign else 160,
                output_tokens=20 if foreign else 40,
                total_tokens=100 if foreign else 200,
            )
        )
        rollout._collect_native_acp_usage()
        rollout._collect_native_acp_usage()
    rollout._session = None
    for _ in range(2):
        await rollout._cleanup_before_teardown()
        assert rollout._usage_metrics["cost_usd"] is None
        assert rollout._usage_metrics["price_source"] is None
        assert rollout._usage_metrics["total_tokens"] == sum(
            100 if x else 200 for x in order
        )
        assert "cost_estimate" not in rollout._usage_metrics["usage_details"]
    assert not (trial / "agent" / "claude-sessions").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agents", [("claude-agent-acp",), ("claude", "vendor/claude-agent-acp", "claude")]
)
async def test_all_claude_native_roles_keep_estimation(tmp_path, monkeypatch, agents):
    home = tmp_path / "sandbox-home"
    _log(
        home / ".claude" / "projects",
        [
            _response("dummy-message", input_tokens=160, output_tokens=40),
            _state(0.01, input_tokens=160, output_tokens=40),
        ],
    )
    rollout, _ = _unpriced_claude_rollout(tmp_path, monkeypatch, home, {})
    rollout._native_usage_metrics = {}
    for agent in agents:
        rollout._active_role = SimpleNamespace(agent=agent)
        rollout._native_usage_checkpoint = None
        rollout._session = ACPSession("dummy")
        rollout._session.record_prompt_usage(
            SimpleNamespace(input_tokens=160, output_tokens=40, total_tokens=200)
        )
        rollout._collect_native_acp_usage()
    rollout._session = None
    await rollout._cleanup_before_teardown()
    assert rollout._usage_metrics["cost_usd"] == 0.01
    assert rollout._usage_metrics["price_source"] == "agent_session_log"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage", [None, SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)]
)
@pytest.mark.parametrize("agent", [SimpleNamespace(agent="codex-acp"), object()])
async def test_unreported_native_role_is_not_free(tmp_path, monkeypatch, usage, agent):
    home = tmp_path / "sandbox-home"
    _log(
        home / ".claude" / "projects",
        [_response("dummy"), _state(0.01, input_tokens=1000, output_tokens=50)],
    )
    rollout, _ = _unpriced_claude_rollout(tmp_path, monkeypatch, home, {})
    rollout._active_role = agent
    rollout._session = ACPSession("dummy")
    if usage is not None:
        rollout._session.record_prompt_usage(usage)
    rollout._collect_native_acp_usage()
    rollout._session = None
    await rollout._cleanup_before_teardown()
    assert rollout._usage_metrics["cost_usd"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary", "roles", "kept"),
    [
        ("claude-agent-acp", set(), "claude-sessions"),
        ("claude-agent-acp", {"claude", "vendor/claude-agent-acp"}, "claude-sessions"),
        ("claude-agent-acp", {"codex-acp"}, None),
        ("codex-acp", set(), "codex-sessions"),
        ("codex-acp", {"claude-agent-acp"}, None),
        ("gemini", set(), None),
    ],
)
async def test_a_session_log_is_kept_only_when_its_agent_made_every_call(
    tmp_path, monkeypatch, primary, roles, kept
):
    """The model-call record copies an agent's own log when the proxy
    saw no call. With another agent's turns in the trial, that log would give
    one agent's calls and price as the trial's: none is copied, and no
    model-calls file is written from it."""
    from benchflow.rollout import _session_log
    from benchflow.task import RolloutPaths
    from benchflow.trajectories.model_calls import write_model_calls
    from tests.test_artifacts_collection import LocalTransport

    home = tmp_path / "sandbox-home"
    claude = [
        _response("msg_1", input_tokens=160, output_tokens=40),
        _state(0.01, input_tokens=160, output_tokens=40),
    ]
    _log(home / ".claude" / "projects", claude)
    _log(home / ".codex" / "sessions", claude, name="rollout-1")
    monkeypatch.setattr(_session_log, "_home", lambda cfg: str(home))
    trial = tmp_path / "trial"
    paths = RolloutPaths(rollout_dir=trial)
    paths.mkdir()
    rollout = SimpleNamespace(
        _config=SimpleNamespace(primary_agent=primary, sandbox_user=None),
        _native_usage_agents={primary, *roles},
        _usage_metrics={"usage_source": "agent_native_acp", "cost_usd": None},
        _env=LocalTransport(),
        _rollout_paths=paths,
        _agent_env={},
    )

    await _session_log.keep_session_logs(rollout)

    copied = sorted(
        p.name for p in (trial / "agent").iterdir() if p.name.endswith("-sessions")
    )
    assert copied == ([kept] if kept else [])
    if kept == "claude-sessions":
        assert write_model_calls(trial)["calls"] == 1
    if kept is None:
        assert write_model_calls(trial) is None
