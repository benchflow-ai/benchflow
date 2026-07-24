from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from benchflow.task.config import SandboxConfig


def test_provider_host_for_model_honors_explicit_generic_endpoint():
    """Guards PR #785: restrictive lockdown must allowlist custom endpoints."""
    from benchflow.agents.providers import provider_host_for_model

    assert (
        provider_host_for_model(
            "deepseek/deepseek-v4-flash",
            {
                "BENCHFLOW_PROVIDER_BASE_URL": "https://proxy.example.test/v1",
                "BENCHFLOW_PROVIDER_API_KEY": "sk-proxy",
            },
        )
        == "proxy.example.test"
    )


def test_provider_host_for_model_resolves_user_supplied_vllm_endpoint():
    """Guards PR #785: vllm/ models need the user endpoint allowlisted."""
    from benchflow.agents.providers import provider_host_for_model

    assert (
        provider_host_for_model(
            "vllm/Qwen/Qwen3-Coder",
            {
                "BENCHFLOW_PROVIDER_BASE_URL": "http://vllm.internal:8000/v1",
                "OPENAI_API_KEY": "sk-vllm",
            },
        )
        == "vllm.internal"
    )


def _make_daytona_stub(task_env_config):
    from benchflow.sandbox.daytona import DaytonaSandbox

    sb = DaytonaSandbox.__new__(DaytonaSandbox)
    sb.task_env_config = task_env_config
    sb.logger = logging.getLogger("test-daytona-pr785")
    sb._compose_mode = False
    applied = {}

    class _Inner:
        async def update_network_settings(
            self, *, network_allow_list=None, network_block_all=None
        ):
            applied["allow_list"] = network_allow_list
            applied["block_all"] = network_block_all

    sb._sandbox = _Inner()
    return sb, applied


@pytest.mark.asyncio
async def test_daytona_relock_no_network_without_model_host_blocks_all():
    """Guards PR #785: Daytona oracle no-network must not require a model host."""
    sb, applied = _make_daytona_stub(SandboxConfig(network_mode="no-network"))

    async def _unreachable(canary=None):
        return False

    sb._egress_reachable = _unreachable
    out = await sb.relock_network(extra_allowed_hosts=())

    assert out == {}
    assert applied["block_all"] is True


@pytest.mark.asyncio
async def test_daytona_relock_threads_every_model_lane_host(monkeypatch):
    """Guards PR #785: Daytona role-provider relock must keep every host."""
    from benchflow.sandbox import network_policy

    sb, applied = _make_daytona_stub(
        SandboxConfig(network_mode="allowlist", allowed_hosts=["task.example.test"])
    )
    seen = {}

    def _plan(hosts, *, model_host, resolve=None):
        seen["hosts"] = hosts
        seen["model_host"] = model_host
        return network_policy.DaytonaAllowlistPlan(
            cidrs=("10.0.0.1/32", "10.0.0.2/32", "10.0.0.3/32")
        )

    monkeypatch.setattr(network_policy, "plan_daytona_allowlist", _plan)

    async def _unreachable(canary=None):
        return False

    sb._egress_reachable = _unreachable
    await sb.relock_network(extra_allowed_hosts=("api.openai.com", "api.deepseek.com"))

    assert seen == {
        "hosts": ("task.example.test", "api.openai.com", "api.deepseek.com"),
        "model_host": None,
    }
    assert sb._extra_allowed_hosts == ("api.openai.com", "api.deepseek.com")
    assert applied["allow_list"] == "10.0.0.1/32,10.0.0.2/32,10.0.0.3/32"


@pytest.mark.asyncio
async def test_daytona_restore_relocks_restrictive_allowlist(monkeypatch):
    """Guards PR #785: Daytona snapshot restore must reapply network allowlists."""
    from benchflow.sandbox import daytona
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect
    from benchflow.sandbox.protocol import SandboxImage

    class _Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _OldSandbox:
        async def delete(self):
            deleted.append(True)

    class _Logger:
        def info(self, *_args, **_kwargs):
            return None

        def warning(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(daytona, "CreateSandboxFromSnapshotParams", _Params)
    deleted = []
    created = []
    relocked = []
    env = SimpleNamespace(
        _sandbox=_OldSandbox(),
        _auto_delete_interval=17,
        _auto_stop_interval=19,
        _network_block_all=False,
        _extra_allowed_hosts=("proxy.example.test",),
        task_env_config=SandboxConfig(
            network_mode="allowlist", allowed_hosts=["a.com"]
        ),
        logger=_Logger(),
    )

    async def _create_sandbox(*, params):
        created.append(params)
        env._sandbox = object()

    async def _relock_network(*, extra_allowed_hosts=()):
        relocked.append(extra_allowed_hosts)
        return {}

    env._create_sandbox = _create_sandbox
    env.relock_network = _relock_network

    await _DaytonaDirect(env).restore(SandboxImage(provider="daytona", ref="snap-a"))

    assert deleted == [True]
    assert created[0].snapshot == "snap-a"
    assert created[0].network_block_all is False
    assert relocked == [("proxy.example.test",)]


@pytest.mark.asyncio
async def test_ensure_litellm_required_usage_fails_before_restrictive_skip():
    """Guards PR #785: required usage must not skip proxy and fail after launch."""
    from benchflow.providers import litellm_runtime

    class _FakeSandbox:
        task_env_config = SandboxConfig(
            network_mode="allowlist", allowed_hosts=["api.deepseek.com"]
        )

    with pytest.raises(RuntimeError, match="Token usage tracking is required"):
        await litellm_runtime.ensure_litellm_runtime(
            agent="openhands",
            agent_env={"DEEPSEEK_API_KEY": "sk-test"},
            model="deepseek/deepseek-v4-flash",
            runtime=None,
            environment="daytona",
            usage_tracking="required",
            sandbox=_FakeSandbox(),
        )
