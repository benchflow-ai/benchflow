"""Capability-first ACP model/effort dispatch (``connect_acp``).

These cover the runtime behavior that lets the ``@agentclientprotocol`` family
migrate ``session/set_model`` → a ``"model"`` config option with no registry
change: the agent's advertised ``session/new`` config options drive the
dispatch, and the registry's ``acp_model_config_id`` is only an override/hint.

The broader connect_acp model-id formatting cases live in
``tests/test_acp.py::TestConnectAcpModelSelection``; the fail-closed lifecycle
cases live in ``tests/test_acp_setup_failure_propagation.py``.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient
from benchflow.providers.litellm_config import (
    LITELLM_MODEL_ALIAS_ENV,
    LITELLM_MODEL_VIA_ENV,
)


def _make_mocks(config_options=None, model_state=None):
    mock_session = MagicMock()
    mock_session.session_id = "s1"
    mock_session.config_options = [] if config_options is None else config_options
    mock_session.model_state = model_state
    mock_init = MagicMock()
    mock_init.agent_info = None

    mock_acp = AsyncMock(spec=ACPClient)
    mock_acp.connect = AsyncMock()
    mock_acp.initialize = AsyncMock(return_value=mock_init)
    mock_acp.session_new = AsyncMock(return_value=mock_session)
    mock_acp.set_model = AsyncMock()
    mock_acp.set_config_option = AsyncMock()
    mock_acp.close = AsyncMock()
    return mock_acp


@contextlib.contextmanager
def _runtime_patches(mock_acp):
    with (
        patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
        patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
    ):
        yield


async def _connect(
    mock_acp, *, agent, model, tmp_path, agent_env=None, reasoning_effort=None
):
    from benchflow.acp.runtime import connect_acp

    with _runtime_patches(mock_acp):
        await connect_acp(
            env=AsyncMock(),
            agent=agent,
            agent_launch=agent,
            agent_env={} if agent_env is None else agent_env,
            sandbox_user=None,
            model=model,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
            reasoning_effort=reasoning_effort,
        )


@pytest.mark.asyncio
async def test_codex_with_only_fastmode_option_uses_set_model(tmp_path):
    """A codex-acp advertising only 'fast-mode' (no 'model') dispatches via
    session/set_model — capability-first must NOT regress it."""
    mock_acp = _make_mocks(config_options=[{"id": "fast-mode"}])
    await _connect(mock_acp, agent="codex-acp", model="gpt-5.5", tmp_path=tmp_path)

    mock_acp.set_model.assert_awaited_once_with("gpt-5.5")
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_litellm_alias_uses_bare_model_for_set_model(tmp_path):
    """Codex validates set_model against its own model catalog, not proxy aliases.

    This guards against a false-green CI path where BenchFlow recorded the
    requested model but codex-acp fell back to its own default at request time.
    """
    mock_acp = _make_mocks(
        config_options=[{"id": "fast-mode"}],
        model_state={
            "availableModels": [
                {"modelId": "gpt-5.4-mini[low]"},
                {"modelId": "gpt-5.4-mini[medium]"},
            ],
            "currentModelId": "gpt-5.5[medium]",
        },
    )
    await _connect(
        mock_acp,
        agent="codex-acp",
        model="openai/gpt-5.4-mini",
        tmp_path=tmp_path,
        agent_env={
            "BENCHFLOW_PROVIDER_MODEL": "benchflow-openai-gpt-5.4-mini",
            LITELLM_MODEL_ALIAS_ENV: "benchflow-openai-gpt-5.4-mini",
            LITELLM_MODEL_VIA_ENV: "1",
        },
    )

    mock_acp.set_model.assert_awaited_once_with("gpt-5.4-mini[medium]")
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_off_catalog_model_owned_by_launch_config_skips_set_model(
    tmp_path,
):
    """Guards the fix for codex-acp@1.6.0 catalog-validated set_model.

    1.6.0 rejects ``session/set_model`` for any model absent from its built-in
    catalog ("Unknown model gpt-5.4-mini[medium]", verified live 2026-08-21) —
    even when that model is already the session's current model via the
    ``CODEX_CONFIG`` injection ``apply_codex_provider_config`` writes for the
    LiteLLM gateway route. When the requested model maps to no advertised
    ``model[effort]`` variant and the session already runs BenchFlow's own
    injected route, the runtime must skip the doomed call instead of failing
    the whole rollout."""
    import json

    mock_acp = _make_mocks(
        config_options=[{"id": "model"}],
        model_state={
            "availableModels": [
                {"modelId": "gpt-5.6-sol[medium]"},
                {"modelId": "gpt-5.5[medium]"},
            ],
            "currentModelId": "benchflow-us-openai-gpt-5.4-mini[medium]",
        },
    )
    await _connect(
        mock_acp,
        agent="codex-acp",
        model="us-openai/gpt-5.4-mini",
        tmp_path=tmp_path,
        agent_env={
            LITELLM_MODEL_VIA_ENV: "1",
            LITELLM_MODEL_ALIAS_ENV: "benchflow-us-openai-gpt-5.4-mini",
            "CODEX_CONFIG": json.dumps(
                {
                    "model": "benchflow-us-openai-gpt-5.4-mini",
                    "model_provider": "benchflow-litellm",
                }
            ),
        },
    )

    mock_acp.set_model.assert_not_awaited()
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_off_catalog_skip_satisfies_effort_carried_by_launch_config(
    tmp_path,
):
    """A requested effort that the CODEX_CONFIG-injected current model already
    carries is satisfied by the skip — the effort step must not fail closed
    for an effort that is in place."""
    import json

    mock_acp = _make_mocks(
        config_options=[{"id": "model"}],
        model_state={
            "availableModels": [{"modelId": "gpt-5.6-sol[medium]"}],
            "currentModelId": "benchflow-us-openai-gpt-5.4-mini[xhigh]",
        },
    )
    await _connect(
        mock_acp,
        agent="codex-acp",
        model="us-openai/gpt-5.4-mini",
        tmp_path=tmp_path,
        reasoning_effort="xhigh",
        agent_env={
            LITELLM_MODEL_VIA_ENV: "1",
            "CODEX_CONFIG": json.dumps({"model": "benchflow-us-openai-gpt-5.4-mini"}),
        },
    )

    mock_acp.set_model.assert_not_awaited()
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_off_catalog_skip_with_unsatisfied_effort_fails_closed(
    tmp_path,
):
    """When the launch config owns an off-catalog model but does NOT carry the
    requested effort, there is no channel left to deliver it — the existing
    effort step must still fail closed rather than silently drop it."""
    import json

    mock_acp = _make_mocks(
        config_options=[{"id": "model"}],
        model_state={
            "availableModels": [{"modelId": "gpt-5.6-sol[medium]"}],
            "currentModelId": "benchflow-us-openai-gpt-5.4-mini[medium]",
        },
    )
    with pytest.raises(RuntimeError, match="does not declare an ACP effort"):
        await _connect(
            mock_acp,
            agent="codex-acp",
            model="us-openai/gpt-5.4-mini",
            tmp_path=tmp_path,
            reasoning_effort="xhigh",
            agent_env={
                LITELLM_MODEL_VIA_ENV: "1",
                "CODEX_CONFIG": json.dumps(
                    {"model": "benchflow-us-openai-gpt-5.4-mini"}
                ),
            },
        )

    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_in_catalog_model_still_uses_set_model_despite_launch_config(
    tmp_path,
):
    """An in-catalog model keeps the set_model path even when CODEX_CONFIG
    injected a gateway route: the skip is only for models set_model cannot
    express (no advertised variant), so the 62cc7e41 bare→``model[effort]``
    mapping must not regress."""
    import json

    mock_acp = _make_mocks(
        config_options=[{"id": "model"}],
        model_state={
            "availableModels": [
                {"modelId": "gpt-5.4-mini[low]"},
                {"modelId": "gpt-5.4-mini[medium]"},
            ],
            "currentModelId": "benchflow-us-openai-gpt-5.4-mini[medium]",
        },
    )
    await _connect(
        mock_acp,
        agent="codex-acp",
        model="us-openai/gpt-5.4-mini",
        tmp_path=tmp_path,
        agent_env={
            LITELLM_MODEL_VIA_ENV: "1",
            "CODEX_CONFIG": json.dumps({"model": "benchflow-us-openai-gpt-5.4-mini"}),
        },
    )

    mock_acp.set_model.assert_awaited_once_with("gpt-5.4-mini[medium]")
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_with_model_option_still_uses_set_model(tmp_path):
    """codex-acp@1.6.0 advertises a 'model' config option whose values reject
    the ``model[effort]`` ids its own session/set_model requires (-32602
    Invalid params, verified live 2026-08-19), so codex is the documented
    exception to capability-first and stays on session/set_model."""
    mock_acp = _make_mocks(config_options=[{"id": "model"}])
    await _connect(mock_acp, agent="codex-acp", model="gpt-5.5", tmp_path=tmp_path)

    mock_acp.set_model.assert_awaited_once_with("gpt-5.5")
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_reasoning_effort_rides_the_model_id(tmp_path):
    """codex-acp declares no ACP effort config option; a requested effort must
    select the matching advertised ``model[effort]`` variant instead of
    failing closed, and the effort step must treat it as satisfied."""
    mock_acp = _make_mocks(
        config_options=[{"id": "fast-mode"}],
        model_state={
            "availableModels": [
                {"modelId": "gpt-5.5[medium]"},
                {"modelId": "gpt-5.5[xhigh]"},
            ],
            "currentModelId": "gpt-5.4-mini[medium]",
        },
    )
    await _connect(
        mock_acp,
        agent="codex-acp",
        model="gpt-5.5",
        tmp_path=tmp_path,
        reasoning_effort="xhigh",
    )

    mock_acp.set_model.assert_awaited_once_with("gpt-5.5[xhigh]")
    mock_acp.set_config_option.assert_not_awaited()


@pytest.mark.asyncio
async def test_unregistered_agent_with_model_option_uses_config_option(tmp_path):
    """An agent not in the registry that advertises a 'model' option is still
    handled via the config option — discovery is from the session, not the
    registry."""
    mock_acp = _make_mocks(config_options=[{"id": "model"}])
    await _connect(
        mock_acp, agent="brand-new-acp", model="some-model", tmp_path=tmp_path
    )

    mock_acp.set_config_option.assert_awaited_once_with("model", "some-model")
    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_hint_overrides_when_session_advertises_nothing(tmp_path):
    """The registry's acp_model_config_id is honored as an override even when
    session/new echoes no config options (thin transports), preserving the
    claude-agent-acp config-option path."""
    mock_acp = _make_mocks(config_options=[])
    await _connect(
        mock_acp, agent="claude-agent-acp", model="claude-opus-4-8", tmp_path=tmp_path
    )

    mock_acp.set_config_option.assert_awaited_once_with("model", "claude-opus-4-8")
    mock_acp.set_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_effort_without_effort_config_id_fails_closed(tmp_path):
    """reasoning_effort requested for an agent that declares no effort config
    option must fail closed rather than silently drop the effort."""
    mock_acp = _make_mocks(config_options=[])
    with pytest.raises(RuntimeError, match="does not declare an ACP effort"):
        await _connect(
            mock_acp,
            agent="test-agent",
            model=None,
            tmp_path=tmp_path,
            reasoning_effort="max",
        )

    mock_acp.close.assert_awaited()


# Request-global classification (PR #1046 second review, P2-A): with Gemini
# and --reasoning-effort high the run built the environment, snapshotted,
# installed and connected the agent, and only then hit the effort rejection —
# which ablation then retried in a branch child. These pin the two halves of
# the fix's classification layer: a deterministic rejection of a global
# request setting (effort/model) surfaces as ACPRequestGlobalError and
# classifies as REQUEST_GLOBAL (non-retryable, never task-attributable),
# while a mere timeout on the same call proves nothing about compatibility
# and must NOT be branded request-global.


@pytest.mark.asyncio
async def test_unsupported_effort_is_a_request_global_rejection(tmp_path):
    """The reviewer's exact shape: gemini declares no ACP effort config option,
    so a requested effort is rejected — and the rejection must classify as
    request-global, not as a retryable/task-attributable agent error."""
    from benchflow._utils.scoring import REQUEST_GLOBAL, classify_error
    from benchflow._utils.text import describe_exception
    from benchflow.acp.runtime import ACPRequestGlobalError

    mock_acp = _make_mocks(config_options=[])
    with pytest.raises(
        ACPRequestGlobalError, match="does not declare an ACP effort"
    ) as excinfo:
        await _connect(
            mock_acp,
            agent="gemini",
            model=None,
            tmp_path=tmp_path,
            reasoning_effort="high",
        )

    assert classify_error(describe_exception(excinfo.value)) == REQUEST_GLOBAL


@pytest.mark.asyncio
async def test_agent_rejected_effort_option_is_request_global(tmp_path):
    """An agent *answering* set_config_option with a protocol error is a
    deterministic rejection of the requested value — request-global."""
    from unittest.mock import AsyncMock

    from benchflow._utils.scoring import REQUEST_GLOBAL, classify_error
    from benchflow._utils.text import describe_exception
    from benchflow.acp.client import ACPError
    from benchflow.acp.runtime import ACPRequestGlobalError

    mock_acp = _make_mocks(config_options=[{"id": "effort"}])
    mock_acp.set_config_option = AsyncMock(
        side_effect=ACPError(-32602, "unsupported effort: xhigh")
    )
    with pytest.raises(ACPRequestGlobalError) as excinfo:
        await _connect(
            mock_acp,
            agent="claude-agent-acp",
            model=None,
            tmp_path=tmp_path,
            reasoning_effort="xhigh",
        )

    assert classify_error(describe_exception(excinfo.value)) == REQUEST_GLOBAL


@pytest.mark.asyncio
async def test_timed_out_effort_option_is_not_request_global(tmp_path):
    """A timeout on set_config_option is transport trouble, not evidence the
    setting is unsupported — it must keep its ordinary (retryable) class."""
    from unittest.mock import AsyncMock

    from benchflow._utils.scoring import REQUEST_GLOBAL, classify_error
    from benchflow._utils.text import describe_exception
    from benchflow.acp.runtime import ACPRequestGlobalError

    mock_acp = _make_mocks(config_options=[{"id": "effort"}])
    mock_acp.set_config_option = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(RuntimeError) as excinfo:
        await _connect(
            mock_acp,
            agent="claude-agent-acp",
            model=None,
            tmp_path=tmp_path,
            reasoning_effort="high",
        )

    assert not isinstance(excinfo.value, ACPRequestGlobalError)
    assert classify_error(describe_exception(excinfo.value)) != REQUEST_GLOBAL


def test_reasoning_effort_preflight_matches_the_runtime_dispatch():
    """The static pre-flight must reject exactly what the runtime would reject
    from registry facts alone: no acp_effort_config_id and no codex-style
    effort-in-model-id. Agents the registry cannot vouch for are left to the
    runtime's own fail-closed check."""
    from benchflow.acp.runtime import reasoning_effort_preflight_error

    assert reasoning_effort_preflight_error("gemini", None) is None
    assert reasoning_effort_preflight_error("claude-agent-acp", "high") is None
    assert reasoning_effort_preflight_error("codex-acp", "high") is None
    assert reasoning_effort_preflight_error("no-such-agent", "high") is None
    error = reasoning_effort_preflight_error("gemini", "high")
    assert error is not None
    assert "not supported by agent 'gemini'" in error


@pytest.mark.asyncio
async def test_env_owned_model_skips_advertised_model_option(tmp_path):
    """A manifest-shaped agent (supports_acp_set_model=False + a
    BENCHFLOW_PROVIDER_MODEL env mapping) with the via-env flag set must get NO
    ACP model configuration — several registry agents (qwen-code, kilo,
    dimcode) advertise a ``model`` config option but validate values against
    their own catalog and reject the gateway alias with -32603."""
    from benchflow.agents.registry import AGENT_INSTALLERS, AGENT_LAUNCH, AGENTS
    from benchflow.agents.registry import AgentConfig as _AC

    AGENTS["env-owned-probe"] = _AC(
        name="env-owned-probe",
        install_cmd="true",
        launch_cmd="true",
        supports_acp_set_model=False,
        env_mapping={"BENCHFLOW_PROVIDER_MODEL": "OPENAI_MODEL"},
    )
    try:
        opt = MagicMock()
        opt.id = "model"
        mock_acp = _make_mocks(config_options=[opt])
        await _connect(
            mock_acp,
            agent="env-owned-probe",
            model="deepseek/deepseek-v4-flash",
            tmp_path=tmp_path,
            agent_env={
                LITELLM_MODEL_VIA_ENV: "1",
                LITELLM_MODEL_ALIAS_ENV: "benchflow-deepseek-deepseek-v4-flash",
                "OPENAI_MODEL": "benchflow-deepseek-deepseek-v4-flash",
            },
        )
        mock_acp.set_config_option.assert_not_awaited()
        mock_acp.set_model.assert_not_awaited()
    finally:
        AGENTS.pop("env-owned-probe", None)
        AGENT_INSTALLERS.pop("env-owned-probe", None)
        AGENT_LAUNCH.pop("env-owned-probe", None)
