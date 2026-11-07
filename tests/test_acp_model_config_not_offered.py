"""A Claude model the adapter refuses fails once, naming what it offers.

Guards the dx/errors fix for dx/first-run's finding: `--agent claude --model
claude-sonnet-9` made three attempts over 203.6 s, each ending
``ACP session/set_config_option failed for agent=claude-agent-acp
config=model value=claude-sonnet-9: ACP error -32603: Internal error``
(acp_error, retried), although the adapter's session/new lists its model
values. It is now the same unscored, never retried ``agent_model``
integration failure codex-acp's set_model path raises.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._utils.scoring import AGENT_INTEGRATION, classify_error
from benchflow.acp.client import ACPClient, ACPError, acp_error
from benchflow.diagnostics import AgentModelNotOfferedError
from benchflow.evaluation import RetryConfig

MODEL_OPTION = {
    "id": "model",
    "name": "Model",
    "type": "select",
    "currentValue": "default",
    "options": [
        {"value": "default", "name": "Default (recommended)"},
        {"value": "sonnet", "name": "Sonnet"},
        {
            "group": "more",
            "name": "More",
            "options": [{"value": "opus"}, {"value": "haiku"}],
        },
    ],
}


def _acp(set_error: BaseException) -> AsyncMock:
    session = MagicMock()
    session.session_id = "s1"
    session.config_options = [MODEL_OPTION]
    session.model_state = None
    init = MagicMock()
    init.agent_info = None
    acp = AsyncMock(spec=ACPClient)
    acp.initialize = AsyncMock(return_value=init)
    acp.session_new = AsyncMock(return_value=session)
    acp.set_config_option = AsyncMock(side_effect=set_error)
    return acp


async def _connect(
    tmp_path, acp, model: str, agent_env: dict[str, str] | None = None
) -> None:
    from benchflow.acp.runtime import connect_acp

    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp),
    ):
        await connect_acp(
            env=AsyncMock(),
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp",
            agent_env=agent_env or {},
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )


# claude-agent-acp throws `Error("Invalid value for config option model: <v>")`;
# the ACP TypeScript SDK (1.5.0, jsonrpc.ts errorToResult) answers it as
# -32603 "Internal error" with data {"details": <the message>}.
REFUSED = {"details": "Invalid value for config option model: claude-sonnet-9"}


async def test_a_refused_model_names_the_offered_ones_and_is_not_retried(tmp_path):
    acp = _acp(acp_error(-32603, "Internal error", REFUSED))
    with pytest.raises(AgentModelNotOfferedError) as caught:
        await _connect(tmp_path, acp, "claude-sonnet-9")
    message = str(caught.value)
    assert message == (
        "agent integration failure [agent_model]: claude-agent-acp does not offer "
        "model 'claude-sonnet-9' for this login; it offers: default, sonnet, opus, haiku"
    )
    assert classify_error(message) == AGENT_INTEGRATION
    assert RetryConfig().should_retry(message) is False


async def test_a_timeout_is_not_called_a_refusal(tmp_path):
    acp = _acp(TimeoutError())
    with pytest.raises(RuntimeError) as caught:
        await _connect(tmp_path, acp, "claude-sonnet-9")
    assert not isinstance(caught.value, AgentModelNotOfferedError)
    assert "Failed to set ACP model config option" in str(caught.value)


@pytest.mark.parametrize(
    ("error", "model"),
    [
        # An opaque -32603 with nothing saying the value was refused.
        (ACPError(-32603, "Internal error"), "claude-sonnet-9"),
        # A listed value that fails is not a value the agent does not offer.
        (ACPError(-32603, "Internal error", REFUSED), "sonnet"),
        # Second review finding: "not available" read as a refusal, so a
        # provider that is briefly down made the model permanently wrong.
        (
            ACPError(-32603, "Internal error", {"details": "model not available"}),
            "claude-sonnet-9",
        ),
        # A refusal that names a different value is not this value's refusal.
        (
            ACPError(
                -32603,
                "Internal error",
                {"details": "Invalid value for config option model: other-model"},
            ),
            "claude-sonnet-9",
        ),
    ],
)
async def test_other_config_errors_stay_retryable(tmp_path, error, model):
    """Review finding: every -32603 on the model option became a permanent,
    never retried agent_model failure that also trips the circuit breaker."""
    acp = _acp(error)
    with pytest.raises(RuntimeError) as caught:
        await _connect(tmp_path, acp, model)
    assert not isinstance(caught.value, AgentModelNotOfferedError)
    assert RetryConfig().should_retry(str(caught.value)) is True


MIMO = {
    "ANTHROPIC_BASE_URL": "https://token-plan-sgp.xiaomimimo.com/anthropic",
    "ANTHROPIC_MODEL": "mimo-v2.6-pro",
}


async def test_a_direct_provider_s_own_model_is_left_to_anthropic_model(tmp_path):
    """A benchmark's MiMo route (Claude Code straight against MiMo's Anthropic
    API): claude-agent-acp 0.81 lists only Claude names and refused
    'mimo-v2.6-pro', so every such trial failed at connect on 0.10. The model
    is in ANTHROPIC_MODEL, the adapter's default: the option is not set."""
    acp = _acp(AssertionError("set_config_option must not be called"))
    await _connect(tmp_path, acp, "mimo-v2.6-pro", agent_env=dict(MIMO))
    acp.set_config_option.assert_not_called()


async def test_a_session_that_lists_no_models_leaves_it_to_the_env_too(tmp_path):
    """With ANTHROPIC_MODEL set to a provider's id, claude-agent-acp 0.81's
    session lists no model values; a benchmark's probe on f59efba3 still set the
    option and failed three times with an opaque -32603."""
    acp = _acp(AssertionError("set_config_option must not be called"))
    acp.session_new.return_value.config_options = [
        {"id": "model", "name": "Model", "type": "select", "currentValue": "default"}
    ]
    await _connect(tmp_path, acp, "mimo-v2.6-pro", agent_env=dict(MIMO))
    acp.set_config_option.assert_not_called()


@pytest.mark.parametrize(
    "agent_env",
    [
        # Anthropic's own endpoint: an unlisted name is a login problem.
        {**MIMO, "ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
        # The env names another model, or none.
        {**MIMO, "ANTHROPIC_MODEL": "other"},
        {"ANTHROPIC_BASE_URL": MIMO["ANTHROPIC_BASE_URL"]},
    ],
)
async def test_otherwise_an_unlisted_model_still_fails_as_not_offered(
    tmp_path, agent_env
):
    refused = {"details": "Invalid value for config option model: mimo-v2.6-pro"}
    acp = _acp(acp_error(-32603, "Internal error", refused))
    with pytest.raises(AgentModelNotOfferedError):
        await _connect(tmp_path, acp, "mimo-v2.6-pro", agent_env=agent_env)


@pytest.mark.parametrize(
    ("agent_env", "named"),
    [
        # A benchmark's direct route: its URL and key on Claude Code's variables.
        (
            {
                "ANTHROPIC_BASE_URL": MIMO["ANTHROPIC_BASE_URL"],
                "ANTHROPIC_AUTH_TOKEN": "t",
            },
            "mimo-v2.6-pro",
        ),
        # Anthropic itself, or a model the env already names: unchanged.
        (
            {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                "ANTHROPIC_AUTH_TOKEN": "t",
            },
            None,
        ),
        ({**MIMO, "ANTHROPIC_AUTH_TOKEN": "t"}, "mimo-v2.6-pro"),
        ({"ANTHROPIC_AUTH_TOKEN": "t"}, None),
    ],
)
async def test_off_the_proxy_a_provider_s_own_model_goes_in_anthropic_model(
    monkeypatch, agent_env, named
):
    """On the subscription route (no proxy), nothing set ANTHROPIC_MODEL for
    a benchmark's direct providers, and the 0.81 adapter refused the id at connect."""
    from benchflow.providers import litellm_runtime

    monkeypatch.setattr(
        litellm_runtime, "uses_native_subscription_auth", lambda *_a, **_k: True
    )
    env, runtime = await litellm_runtime.ensure_litellm_runtime(
        agent="claude-agent-acp",
        agent_env=dict(agent_env),
        model="mimo-v2.6-pro",
        runtime=None,
        environment="docker",
    )
    assert runtime is None
    assert env.get("ANTHROPIC_MODEL") == named


async def test_a_model_the_session_already_runs_is_not_set_again(tmp_path):
    """A benchmark's direct route reaches the adapter through a loopback proxy,
    with ANTHROPIC_MODEL set: the session starts on 'mimo-v2.6-pro' (its
    currentValue) and refuses to set that id again (-32603, three retries)."""
    acp = _acp(AssertionError("set_config_option must not be called"))
    acp.session_new.return_value.config_options = [
        {**MODEL_OPTION, "currentValue": "mimo-v2.6-pro"}
    ]
    env = {**MIMO, "ANTHROPIC_BASE_URL": "http://127.0.0.1:47319"}
    # The loopback host is not Anthropic's, so the env rule applies too; the
    # current value alone must also be enough.
    await _connect(tmp_path, acp, "mimo-v2.6-pro", agent_env=env)
    acp.set_config_option.assert_not_called()
    acp2 = _acp(AssertionError("set_config_option must not be called"))
    acp2.session_new.return_value.config_options = [
        {**MODEL_OPTION, "currentValue": "sonnet"}
    ]
    acp2.set_config_option = AsyncMock()
    await _connect(tmp_path, acp2, "sonnet")
    acp2.set_config_option.assert_not_called()
