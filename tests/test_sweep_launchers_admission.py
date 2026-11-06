"""Guards FF-sw-launchers against paid setup before impossible ACP (363d815b)."""

from dataclasses import replace

import pytest

from benchflow.sandbox.providers import PROVIDERS_BY_NAME
from benchflow.task.plan_validation import validate_task_plan
from benchflow.task.runtime_capabilities import UnsupportedTaskFeatureError
from tests.test_check_run_plan import _task


@pytest.mark.parametrize("agent", ["codex-acp", "claude-agent-acp"])
@pytest.mark.parametrize("harness", ["acp", "native"])
def test_acp_without_a_live_transport_refused_before_proxy_or_sandbox(
    tmp_path, monkeypatch, agent, harness
):
    """Also guards an admission regression: native uses live_process at 04d3d5f9.

    Modal used to be the backend without a live transport; the registry
    entry is set back to that here, so the refusal stays guarded.
    """
    task = _task(tmp_path / "task")
    monkeypatch.setitem(
        PROVIDERS_BY_NAME,
        "modal",
        replace(PROVIDERS_BY_NAME["modal"], supports_live_process=False),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("reached costly setup")

    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", unexpected
    )
    with pytest.raises(UnsupportedTaskFeatureError, match=r"live.*transport"):
        validate_task_plan(
            task, sandbox="modal", agent=agent, harness=harness, agent_env={}
        )


@pytest.mark.parametrize(
    "agent,key",
    [("codex-acp", "OPENAI_API_KEY"), ("claude-agent-acp", "ANTHROPIC_API_KEY")],
)
@pytest.mark.parametrize("harness", ["acp", "native"])
def test_modal_acp_and_native_agents_are_admitted(
    tmp_path, monkeypatch, agent, key, harness
):
    """Modal hosts ACP and native agents over Sandbox.exec stdio
    (ModalProcess), so their plan reaches the proxy like any backend's."""
    from unittest.mock import MagicMock

    task = _task(tmp_path / "task")
    proxy = MagicMock(return_value=None)
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", proxy
    )
    validate_task_plan(
        task,
        sandbox="modal",
        agent=agent,
        harness=harness,
        agent_env={key: "dummy-key"},
    )
    proxy.assert_called_once()


@pytest.mark.parametrize(
    "agent,harness",
    [
        ("oracle", "acp"),
        ("nop", "acp"),
        ("task-runtime", "acp"),
    ],
)
def test_modal_noninteractive_paths_remain_admitted(
    tmp_path, monkeypatch, agent, harness
):
    from unittest.mock import MagicMock

    task = _task(tmp_path / "task")
    proxy = MagicMock(return_value=None)
    monkeypatch.setattr(
        "benchflow.providers.litellm_runtime.proxy_install_record", proxy
    )
    validate_task_plan(
        task, sandbox="modal", agent=agent, harness=harness, agent_env={}
    )
    proxy.assert_called_once()
