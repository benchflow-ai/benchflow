import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._types import Role
from benchflow.agents import registry
from benchflow.agents.codex_config import apply_codex_launch_config, disable_codex_apps
from benchflow.agents.codex_connector_policy import (
    effective_apps_policy,
    enforce_codex_apps_policy,
)
from benchflow.agents.registry import AGENTS
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox._base import ExecResult


def test_existing_web_policy_does_not_disable_hosted_apps():
    # GH1107: no-web config leaves Apps to codex_apps_policy, so inherit works.
    result, _ = apply_codex_launch_config(
        "codex-acp",
        {"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"},
        model=None,
        reasoning_effort=None,
    )
    config = json.loads(result["CODEX_CONFIG"])
    assert config["web_search"] == "disabled"
    assert config.get("features", {}).get("apps") is not False


@pytest.mark.parametrize(
    "override,purpose,skip,expected",
    [
        (None, "task", False, "disabled"),
        (None, "task", True, "inherit"),
        (None, "reviewer", False, "inherit"),
        ("disabled", "task", True, "disabled"),
        ("inherit", "task", False, "inherit"),
    ],
)
def test_classifier(override, purpose, skip, expected):
    assert (
        effective_apps_policy(override, purpose=purpose, skip_verify=skip) == expected
    )


def test_config_preserves_mcp_and_other_features():
    original = {
        "CODEX_CONFIG": json.dumps(
            {
                "features": {"apps": True, "other": True},
                "mcp_servers": {"local": {"command": "mock"}},
            }
        )
    }
    result = disable_codex_apps(original)
    assert json.loads(result["CODEX_CONFIG"])["features"] == {
        "apps": False,
        "other": True,
    }
    assert json.loads(result["CODEX_CONFIG"])["mcp_servers"] == {
        "local": {"command": "mock"}
    }
    assert json.loads(original["CODEX_CONFIG"])["features"]["apps"] is True


def response(stdout="", code=0):
    return ExecResult(
        return_code=code, stdout=stdout, stderr="private error must not leak"
    )


def args(tmp_path):
    return dict(
        agent="codex-acp",
        agent_launch=AGENTS["codex-acp"].launch_cmd,
        agent_env={},
        sandbox_user="agent",
        policy="disabled",
        requested=None,
        rollout_dir=tmp_path,
    )


ADAPTER = "@agentclientprotocol/codex-acp"
INSTALLED = response(json.dumps({"requirements_sha256": "a" * 64}))
APPS_OFF = response("apps stable false\n")


# The pinned adapter's own npm metadata: codex-acp 1.13.1 declares
# ``"@openai/codex": "^0.156.1"`` (``npm view @agentclientprotocol/codex-acp@1.13.1
# dependencies``), and npm resolves that range to codex-cli 0.156.1.
PINNED_DECLARED = "^0.156.1"
PINNED_NATIVE = "0.156.1"
# The previous pin, used where a test needs an adapter the registry does not pin.
OLD_ADAPTER, OLD_DECLARED, OLD_NATIVE = "1.6.0", "^0.148.0", "0.148.0"


def admission(adapter=None, declared=PINNED_DECLARED, native=PINNED_NATIVE):
    """Probe replies in order: UID, adapter, installed adapter manifest, native.

    The adapter defaults to the registry pin, so the happy path follows a bump.
    """
    adapter = adapter or registry.pinned_npm_package("codex-acp")[1]
    manifest = {"name": ADAPTER, "version": adapter, "codex": declared}
    return [
        response("1000"),
        response(f"{ADAPTER} {adapter}"),
        response(json.dumps(manifest)),
        response(f"codex-cli {native}"),
    ]


async def test_disabled_requires_native_precedence_and_writes_safe_receipt(tmp_path):
    env = SimpleNamespace(
        exec=AsyncMock(side_effect=[*admission(), INSTALLED, APPS_OFF])
    )
    updated = await enforce_codex_apps_policy(env, **args(tmp_path))
    assert json.loads(updated["CODEX_CONFIG"])["features"]["apps"] is False
    receipt = json.loads((tmp_path / "codex_apps_policy.json").read_text())
    assert receipt["effective_apps"] is False
    assert receipt["authenticated_tool_absence_verified"] is False
    assert "private" not in json.dumps(receipt)
    assert env.exec.call_args_list[-1].kwargs["user"] == "root"
    assert "features.apps=true" in env.exec.call_args_list[-1].args[0]
    assert "--reuid=agent" in env.exec.call_args_list[-1].args[0]


@pytest.mark.parametrize("raw", ["notjson", "[]", '{"features":[]}'])
async def test_malformed_config_aborts_before_execution(tmp_path, raw):
    env = SimpleNamespace(exec=AsyncMock())
    kw = args(tmp_path)
    kw["agent_env"] = {"CODEX_CONFIG": raw}
    with pytest.raises(ValueError):
        await enforce_codex_apps_policy(env, **kw)
    env.exec.assert_not_awaited()


@pytest.mark.parametrize(
    "responses",
    [
        [response("0")],
        admission(native="0.149.0"),
        [*admission(), response(code=1)],
        [*admission(), INSTALLED, response("apps stable true")],
    ],
)
async def test_failed_enforcement_does_not_publish_success(tmp_path, responses):
    env = SimpleNamespace(exec=AsyncMock(side_effect=responses))
    with pytest.raises(RuntimeError):
        await enforce_codex_apps_policy(env, **args(tmp_path))
    assert not (tmp_path / "codex_apps_policy.json").exists()


@pytest.mark.parametrize(
    "overrides,responses",
    [
        ({"sandbox_user": "root"}, []),
        ({"sandbox_user": None}, []),
        ({"agent_launch": "custom-codex --acp"}, []),
        ({"agent_env": {"CODEX_PATH": "/tmp/codex"}}, []),
        ({}, [response("0")]),
        ({}, admission(adapter="1.5.0")),
        ({}, [*admission()[:2], response(code=1)]),
        ({}, admission(native="0.149.0")),
        ({}, [*admission(), response(code=1)]),
        ({}, [*admission(), response("notjson")]),
        ({}, [*admission(), INSTALLED, response("apps stable true")]),
    ],
)
async def test_refusal_names_the_inherit_escape_hatch(tmp_path, overrides, responses):
    """Guards this fix for the Apps-off default for scored Codex tasks (GH #1107): every Apps admission refusal
    names the documented opt-out, not just the failed check."""
    env = SimpleNamespace(exec=AsyncMock(side_effect=responses))
    with pytest.raises(RuntimeError) as refused:
        await enforce_codex_apps_policy(env, **{**args(tmp_path), **overrides})
    assert "--codex-apps-policy inherit" in str(refused.value)
    assert "codex_apps_policy: inherit" in str(refused.value)


async def test_inherit_does_not_loosen_or_probe_existing_policy(tmp_path):
    env = SimpleNamespace(exec=AsyncMock())
    kw = args(tmp_path)
    kw["policy"] = "inherit"
    assert await enforce_codex_apps_policy(env, **kw) == {}
    env.exec.assert_not_awaited()


async def test_failed_reconnect_removes_stale_receipt(tmp_path):
    (tmp_path / "codex_apps_policy.json").write_text('{"effective_apps":false}')
    env = SimpleNamespace(exec=AsyncMock(side_effect=[response("0")]))
    with pytest.raises(RuntimeError):
        await enforce_codex_apps_policy(env, **args(tmp_path))
    assert not (tmp_path / "codex_apps_policy.json").exists()


async def test_role_reconnect_without_credential_changes_enforces_before_connect(
    tmp_path, monkeypatch
):
    r = Rollout(RolloutConfig(task_path=tmp_path, agent="codex-acp", model="gpt-test"))
    r._rollout_dir = tmp_path
    r._env = object()
    r._agent_cwd = "/app"
    r._timeout = 10
    r._disallow_web_tools = False
    monkeypatch.setattr(r, "_require_rollout_dir", lambda: tmp_path)
    monkeypatch.setattr(r._planes, "resolve_agent_env", lambda *a: {})
    monkeypatch.setattr(
        r._planes, "ensure_litellm_runtime", AsyncMock(return_value=({}, None))
    )
    credentials = AsyncMock()
    connect = AsyncMock()
    monkeypatch.setattr(r._planes, "write_credential_files", credentials)
    monkeypatch.setattr(r._planes, "connect_acp", connect)
    # A failed admission proves the connection cannot start, even though this
    # same-agent/model role does not enter the credential setup conditional.
    policy = AsyncMock(side_effect=RuntimeError("policy admission blocked"))
    monkeypatch.setattr("benchflow.rollout.enforce_codex_apps_policy", policy)
    with pytest.raises(RuntimeError, match="policy admission blocked"):
        await r.connect_as(Role(name="same", agent="codex-acp", model="gpt-test"))
    credentials.assert_not_awaited()
    connect.assert_not_awaited()
    assert policy.await_args.kwargs["policy"] == "disabled"


async def test_apps_gate_follows_registry_codex_acp_pin(tmp_path, monkeypatch):
    """Guards this fix against the duplicated pins from the Apps-off default for scored Codex tasks (GH #1107).

    The gate hard-coded codex-acp 1.6.0 / codex-cli 0.148.0 beside the registry
    pin, so PR #1139's bump to codex-acp 1.13.1 (codex ^0.156.1) left every test
    green and refused every scored codex-acp run at admission.
    """
    monkeypatch.setattr(registry, "_CODEX_ACP_PACKAGE", f"{ADAPTER}@1.13.1")
    bumped = admission("1.13.1", "^0.156.1", "0.156.1")
    env = SimpleNamespace(exec=AsyncMock(side_effect=[*bumped, INSTALLED, APPS_OFF]))
    await enforce_codex_apps_policy(env, **args(tmp_path))
    receipt = json.loads((tmp_path / "codex_apps_policy.json").read_text())
    assert receipt["native_version"] == "0.156.1"
    assert receipt["adapter_version"] == "1.13.1"
    stale = SimpleNamespace(
        exec=AsyncMock(side_effect=admission(OLD_ADAPTER, OLD_DECLARED, OLD_NATIVE))
    )
    with pytest.raises(RuntimeError, match=r"Codex ACP 1\.13\.1"):
        await enforce_codex_apps_policy(stale, **args(tmp_path))


@pytest.mark.parametrize(
    "declared,native,admitted",
    [
        ("^0.148.0", "0.148.3", True),
        ("^0.156.1", "0.156.1", True),
        ("^0.156.1", "0.156.0", False),
        ("^0.148.0", "0.149.0", False),
        ("^1.2.0", "1.9.0", True),
        ("^1.2.0", "2.0.0", False),
        ("0.148.0", "0.148.0", True),
        ("0.148.0", "0.148.1", False),
        ("^0.148.0", "0.148.1-alpha.1", False),
        (">=0.148.0", "0.148.0", False),
        ("latest", "0.148.0", False),
        (None, "0.148.0", False),
    ],
)
async def test_native_version_must_satisfy_adapter_declared_range(
    tmp_path, declared, native, admitted
):
    """Guards this fix for the Apps-off default for scored Codex tasks (GH #1107): the registry pins only codex-acp,
    so native Codex is admitted by the range that adapter declares, and range
    forms we cannot evaluate fail closed."""
    replies = admission(declared=declared, native=native)
    env = SimpleNamespace(exec=AsyncMock(side_effect=[*replies, INSTALLED, APPS_OFF]))
    if admitted:
        await enforce_codex_apps_policy(env, **args(tmp_path))
    else:
        with pytest.raises(RuntimeError, match="native Codex"):
            await enforce_codex_apps_policy(env, **args(tmp_path))
        assert not (tmp_path / "codex_apps_policy.json").exists()


async def test_native_version_output_must_name_codex_cli(tmp_path):
    """Guards this fix for the Apps-off default for scored Codex tasks (GH #1107): a bare version string is not the
    native Codex identity."""
    replies = admission()
    replies[3] = response(PINNED_NATIVE)
    env = SimpleNamespace(exec=AsyncMock(side_effect=replies))
    with pytest.raises(RuntimeError, match="native Codex"):
        await enforce_codex_apps_policy(env, **args(tmp_path))


async def test_declared_range_must_come_from_the_pinned_adapter(tmp_path):
    """Guards this fix for the Apps-off default for scored Codex tasks (GH #1107): a manifest from another adapter
    version cannot supply the native range."""
    replies = admission()
    replies[2] = response(
        json.dumps({"name": ADAPTER, "version": OLD_ADAPTER, "codex": OLD_DECLARED})
    )
    replies[3] = response(f"codex-cli {OLD_NATIVE}")
    env = SimpleNamespace(exec=AsyncMock(side_effect=replies))
    with pytest.raises(RuntimeError, match="native Codex"):
        await enforce_codex_apps_policy(env, **args(tmp_path))


def test_manifest_override_cannot_redefine_probed_launcher(tmp_path):
    """Fresh import exercises the supported override before helper initialization."""
    manifest = tmp_path / "agents" / "codex"
    manifest.mkdir(parents=True)
    (manifest / "manifest.toml").write_text(
        'contract_version="1.0"\nname="codex-acp"\n'
        'install_cmd="true"\nlaunch_cmd="custom-codex --acp"\n'
    )
    program = """
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from benchflow.agents.registry import AGENTS
from benchflow.agents.codex_connector_policy import enforce_codex_apps_policy
async def main():
    assert AGENTS['codex-acp'].launch_cmd == 'custom-codex --acp'
    env = SimpleNamespace(exec=AsyncMock())
    try:
        await enforce_codex_apps_policy(env, agent='codex-acp',
            agent_launch=AGENTS['codex-acp'].launch_cmd, agent_env={},
            sandbox_user='agent', policy='disabled', requested=None,
            rollout_dir=Path('.'))
    except RuntimeError as exc:
        assert 'managed Codex ACP launcher' in str(exc)
    else:
        raise AssertionError('custom launcher admitted')
    env.exec.assert_not_awaited()
asyncio.run(main())
"""
    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={
            **os.environ,
            "BENCHFLOW_AGENTS_DIR": str(manifest.parent),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_policy_doc_names_the_registry_pin():
    """Guards the codex-acp 1.13.1 bump (#1145, #1146): the operator
    doc states the pinned adapter and its declared native range; a bump that
    leaves it naming the old pin misstates what admission accepts."""
    doc = (Path(__file__).parents[1] / "docs" / "codex-apps-policy.md").read_text()
    _, version = registry.pinned_npm_package("codex-acp")
    assert f"currently {version})" in doc
    assert f"(`{PINNED_DECLARED}` for {version}" in doc
