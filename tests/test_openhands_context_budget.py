import json
import os
import subprocess
import sys

from benchflow.agents.registry import AGENTS, _OPENHANDS_SETTINGS_WRITER


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
    config = AGENTS["openhands"]
    path = "/opt/benchflow/bin/openhands-settings-writer"
    assert path in config.install_cmd
    assert path in config.launch_cmd
