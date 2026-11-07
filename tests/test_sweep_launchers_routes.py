"""Guards admission regressions at 04d3d5f9."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from benchflow.agents import registry
from benchflow.sandbox.providers import PROVIDERS_BY_NAME
from benchflow.task.plan_validation import validate_task_plan
from benchflow.task.runtime_capabilities import UnsupportedTaskFeatureError
from tests.test_check_run_plan import _task


@pytest.fixture
def modal_without_live_transport(monkeypatch):
    """Modal as it used to be: a backend with no live transport.

    Modal now hosts ACP agents (ModalProcess); the routes a backend
    without one must refuse stay guarded on its registry entry.
    """
    monkeypatch.setitem(
        PROVIDERS_BY_NAME,
        "modal",
        replace(PROVIDERS_BY_NAME["modal"], supports_live_process=False),
    )


@pytest.mark.parametrize("agent", ["codex-acp", "claude-agent-acp", "acp/codex-acp"])
@pytest.mark.parametrize("harness", ["acp", "native"])
def test_modal_live_routes_refused_before_setup(
    tmp_path, monkeypatch, modal_without_live_transport, agent, harness
):
    task = _task(tmp_path / "task")
    proxy = MagicMock(side_effect=AssertionError("costly setup reached"))
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", proxy
    )
    with pytest.raises(UnsupportedTaskFeatureError, match=r"live.*transport"):
        validate_task_plan(
            task, sandbox="modal", agent=agent, harness=harness, agent_env={}
        )
    proxy.assert_not_called()


@pytest.mark.parametrize(
    "spelling", ["fixture-factory", "acp/fixture-factory", "fixture-alias"]
)
def test_modal_session_factory_route_admitted(tmp_path, monkeypatch, spelling):
    task = _task(tmp_path / "task")
    monkeypatch.setitem(
        registry.AGENTS,
        "fixture-factory",
        registry.AgentConfig(
            name="fixture-factory",
            install_cmd="",
            launch_cmd="",
            protocol="session-factory",
            session_factory="fixture:connect",
        ),
    )
    monkeypatch.setitem(registry.AGENT_ALIASES, "fixture-alias", "fixture-factory")
    proxy = MagicMock(return_value=None)
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", proxy
    )
    validate_task_plan(
        task, sandbox="modal", agent=spelling, harness="acp", agent_env={}
    )
    proxy.assert_called_once()


@pytest.mark.parametrize(
    "protocol,factory", [("session-factory", ""), ("acp", "fixture:connect")]
)
def test_only_complete_session_factory_route_is_exempt(
    tmp_path, monkeypatch, modal_without_live_transport, protocol, factory
):
    task = _task(tmp_path / "task")
    monkeypatch.setitem(
        registry.AGENTS,
        "fixture-incomplete",
        registry.AgentConfig(
            name="fixture-incomplete",
            install_cmd="",
            launch_cmd="",
            protocol=protocol,
            session_factory=factory,
        ),
    )
    with pytest.raises(UnsupportedTaskFeatureError):
        validate_task_plan(
            task, sandbox="modal", agent="fixture-incomplete", agent_env={}
        )


@pytest.mark.parametrize("harness", ["acp", "native"])
@pytest.mark.parametrize("agent", ["codex-acp", "claude-agent-acp"])
@pytest.mark.parametrize(
    "sandbox",
    [
        "docker",
        "remote-docker",
        "daytona",
        "modal",
        "apple-container",
        "agentcore",
    ],
)
def test_declared_live_backends_remain_admitted(
    tmp_path, monkeypatch, harness, agent, sandbox
):
    task = _task(tmp_path / "task")
    proxy = MagicMock(return_value=None)
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", proxy
    )
    validate_task_plan(
        task, sandbox=sandbox, agent=agent, harness=harness, agent_env={}
    )
    proxy.assert_called_once()
