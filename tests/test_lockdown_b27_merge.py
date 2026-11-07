"""Guard mount provenance across the verifier batching merge at ada45055."""

from unittest.mock import AsyncMock

import pytest

from benchflow._utils.trusted_path import TrustedPathReport
from benchflow.sandbox import lockdown
from tests.test_sandbox_hardening import _make_env, _make_task


@pytest.mark.parametrize("agent_paths", [None, ("/app",)])
@pytest.mark.parametrize("probe", [None, {}], ids=["fallback", "batched"])
async def test_batched_verifier_keeps_captured_mount_provenance(
    monkeypatch, agent_paths, probe
):
    """Every trust consumer must receive the pre-freeze alias set unchanged."""
    blocked = ("/app", "/opt/answer-alias", "/logs")
    monkeypatch.setattr(
        lockdown, "_probe_verifier_reads", AsyncMock(return_value=probe)
    )
    consumers = {}
    for name, result in (
        ("_judge_verifier_path", TrustedPathReport(path="/usr/bin:/bin")),
        ("_isolate_verifier_tool_state", {}),
        # The guard's discovery (V2: required and optional plugins).
        ("_discover_pytest_plugins", ([], [])),
        ("_install_pytest_plugin_guard", ("/guard", [], True, ())),
    ):
        consumers[name] = AsyncMock(return_value=result)
        monkeypatch.setattr(lockdown, name, consumers[name])
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_pythonpath", AsyncMock(return_value="")
    )
    monkeypatch.setattr(lockdown, "_distro_pip_env", AsyncMock(return_value={}))
    monkeypatch.setattr(lockdown, "_verifier_uid", AsyncMock(return_value=0))
    result = await lockdown._build_verifier_env(
        _make_env(),
        _make_task(),
        "agent",
        "/app",
        agent_paths=agent_paths,
        bash_env="",
        blocked=blocked,
    )
    for consumer in consumers.values():
        assert consumer.await_args.kwargs["blocked"] == blocked
    assert result["PIP_CONFIG_FILE"] == "/dev/null"
