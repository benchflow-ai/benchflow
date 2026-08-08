import json
import os
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents.install import install_agent
from benchflow.agents.registry import _OPENHANDS_SETTINGS_WRITER, AGENTS
from benchflow.rollout_planes import DefaultRolloutPlanes


def test_openhands_settings_reserve_context_for_output(tmp_path):
    """Guards PR #927 against losing OpenHands context/output budget settings."""
    target = tmp_path / "agent_settings.json"
    env = {
        **os.environ,
        "LLM_MODEL": "openai/qwen35-9b-base",
        "LLM_API_KEY": "placeholder",
        "LLM_BASE_URL": "http://example.test/v1",
        "LLM_NATIVE_TOOL_CALLING": "true",
        "LLM_CACHING_PROMPT": "false",
        "LLM_DROP_PARAMS": "true",
        "LLM_MODIFY_PARAMS": "true",
        "BENCHFLOW_OPENHANDS_CONTEXT_LIMIT": "262144",
        "BENCHFLOW_OPENHANDS_OUTPUT_LIMIT": "32768",
        "BENCHFLOW_OPENHANDS_CONTEXT_RESERVE": "4096",
    }

    completed = subprocess.run(
        [sys.executable, "-c", _OPENHANDS_SETTINGS_WRITER, str(target)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    settings = json.loads(target.read_text())

    assert completed.stdout == ""
    assert settings["kind"] == "Agent"
    assert settings["llm"]["max_input_tokens"] == 262144
    assert settings["llm"]["max_output_tokens"] == 32768
    assert settings["llm"]["caching_prompt"] is False
    assert settings["condenser"]["kind"] == "LLMSummarizingCondenser"
    assert settings["condenser"]["max_tokens"] == 225280
    assert settings["condenser"]["llm"]["usage_id"] == "condenser"


def test_openhands_settings_preserves_override_launch_llm_options(tmp_path):
    """Guards PR #927 follow-up against commit 64edbc1f bypassing PR #921 settings."""
    target = tmp_path / "agent_settings.json"
    env = {
        **os.environ,
        "LLM_MODEL": "openai/gpt-5.6-sol",
        "LLM_API_KEY": "proxy-key",
        "LLM_BASE_URL": "http://127.0.0.1:4000/v1",
        "LLM_API_VERSION": "preview",
        "LLM_TIMEOUT": "115200",
        "LLM_REASONING_EFFORT": "max",
    }

    subprocess.run(
        [sys.executable, "-c", _OPENHANDS_SETTINGS_WRITER, str(target)],
        env=env,
        check=True,
    )

    settings = json.loads(target.read_text())
    assert settings["llm"]["timeout"] == 115200
    assert settings["llm"]["litellm_extra_body"] == {"reasoning": {"effort": "max"}}
    assert "reasoning_effort" not in settings["llm"]
    assert settings["condenser"]["llm"]["timeout"] == 115200
    assert settings["condenser"]["llm"]["litellm_extra_body"] == {
        "reasoning": {"effort": "max"}
    }


def test_openhands_settings_preserves_typed_reasoning_effort(tmp_path):
    """Guards PR #927 against launch_override_cmd dropping typed OpenHands effort."""
    target = tmp_path / "agent_settings.json"
    env = {
        **os.environ,
        "LLM_MODEL": "openai/gpt-5.6-sol",
        "LLM_API_KEY": "proxy-key",
        "LLM_REASONING_EFFORT": "xhigh",
    }

    subprocess.run(
        [sys.executable, "-c", _OPENHANDS_SETTINGS_WRITER, str(target)],
        env=env,
        check=True,
    )

    settings = json.loads(target.read_text())
    assert settings["llm"]["reasoning_effort"] == "xhigh"
    assert settings["llm"]["litellm_extra_body"] == {"reasoning_effort": "xhigh"}


def test_openhands_settings_can_disable_subagents(tmp_path):
    """Guards PR #927 against launch_override_cmd bypassing subagent disable."""
    target = tmp_path / "agent_settings.json"
    tool_root = tmp_path / "tools"
    package_root = tool_root / "openhands" / "site-packages"
    package_dir = package_root / "openhands_cli"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    utils_path = package_dir / "utils.py"
    utils_path.write_text(
        "def get_default_cli_tools():\n"
        "    return [\n"
        "        Tool(name=task_tool_name),\n"
        "    ]\n"
    )
    bin_dir = tool_root / "openhands" / "bin"
    bin_dir.mkdir(parents=True)
    python_wrapper = bin_dir / "python"
    python_wrapper.write_text(
        f'#!/bin/sh\nPYTHONPATH={package_root} exec {sys.executable} "$@"\n'
    )
    python_wrapper.chmod(0o755)
    openhands = bin_dir / "openhands"
    openhands.write_text("#!/bin/sh\nexit 0\n")
    openhands.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "openhands").symlink_to(openhands)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LLM_MODEL": "openai/gpt-5.6-sol",
        "LLM_API_KEY": "proxy-key",
        "BENCHFLOW_OPENHANDS_DISABLE_SUBAGENTS": "1",
    }

    subprocess.run(
        [sys.executable, "-c", _OPENHANDS_SETTINGS_WRITER, str(target)],
        env=env,
        check=True,
    )

    patched = utils_path.read_text()
    assert "Tool(name=task_tool_name)" not in patched
    assert "BenchFlow: delegation disabled for this run." in patched


def test_openhands_settings_rejects_non_numeric_timeout(tmp_path):
    """Guards PR #927 against malformed timeout JSON in override launch settings."""
    target = tmp_path / "agent_settings.json"
    env = {
        **os.environ,
        "LLM_MODEL": "openai/gpt-5.6-sol",
        "LLM_API_KEY": "proxy-key",
        "LLM_TIMEOUT": "none",
    }

    result = subprocess.run(
        [sys.executable, "-c", _OPENHANDS_SETTINGS_WRITER, str(target)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "LLM_TIMEOUT must be a non-negative integer" in result.stderr


def test_openhands_launch_installs_and_runs_settings_writer():
    """Guards PR #927 against bypassing the effective OpenHands launch override."""
    path = "/opt/benchflow/bin/openhands-settings-writer"
    launch = DefaultRolloutPlanes().agent_launch("openhands", disallow_web_tools=False)
    assert path in AGENTS["openhands"].install_setup_cmd
    assert path in AGENTS["openhands"].launch_override_cmd
    assert "mkdir -p ~/.openhands" in AGENTS["openhands"].launch_override_cmd
    assert path in launch
    assert launch == AGENTS["openhands"].launch_override_cmd


@pytest.mark.asyncio
async def test_openhands_install_runs_root_owned_settings_setup(tmp_path):
    """Guards PR #927 against installing the settings writer after user launch."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

    await install_agent(env, "openhands", tmp_path)

    assert env.exec.await_count == 2
    setup_cmd = env.exec.await_args_list[1].args[0]
    assert "/opt/benchflow/bin/openhands-settings-writer" in setup_cmd
