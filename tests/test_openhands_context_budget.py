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


def test_openhands_launch_installs_and_runs_settings_writer():
    path = "/opt/benchflow/bin/openhands-settings-writer"
    launch = DefaultRolloutPlanes().agent_launch("openhands", disallow_web_tools=False)
    assert path in AGENTS["openhands"].install_setup_cmd
    assert path in AGENTS["openhands"].launch_override_cmd
    assert "mkdir -p ~/.openhands" in AGENTS["openhands"].launch_override_cmd
    assert path in launch
    assert launch == AGENTS["openhands"].launch_override_cmd


@pytest.mark.asyncio
async def test_openhands_install_runs_root_owned_settings_setup(tmp_path):
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

    await install_agent(env, "openhands", tmp_path)

    assert env.exec.await_count == 2
    setup_cmd = env.exec.await_args_list[1].args[0]
    assert "/opt/benchflow/bin/openhands-settings-writer" in setup_cmd
