"""Offline tests for the Baseten cookbook: key handling and the Truss config."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "rl" / "baseten"
sys.path.insert(0, str(EXAMPLE))

import baseten_deploy as bd  # noqa: E402

KEY = "test-baseten-key-not-real"


def test_deployment_url_is_the_pinned_openai_route():
    assert (
        bd.deployment_url("abcd1234", "xyz9876")
        == "https://model-abcd1234.api.baseten.co/deployment/xyz9876/sync/v1"
    )


def test_missing_key_stops_before_any_request(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        bd._client()


def test_client_sends_the_key_only_as_a_header(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", KEY)
    with bd._client() as client:
        assert client.headers["Authorization"] == f"Api-Key {KEY}"


def test_push_hands_the_key_to_truss_by_environment_not_argv(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", KEY)
    seen = {}

    def fake_call(command, env):
        seen["command"] = command
        seen["env"] = env
        return 0

    monkeypatch.setattr(bd.subprocess, "call", fake_call)
    with pytest.raises(SystemExit) as exit_info:
        bd.main(["push", "truss/qwen35-9b-lora", "--deployment-name", "eval-1"])
    assert exit_info.value.code == 0
    assert not any(KEY in part for part in seen["command"])
    assert seen["command"][:3] == ["truss", "push", "truss/qwen35-9b-lora"]
    assert seen["env"]["BASETEN_TRUSS_AUTH_API_KEY"] == KEY
    assert seen["env"]["BASETEN_TRUSS_AUTH_REMOTE_URL"] == "https://app.baseten.co"


def _truss_config() -> dict:
    return yaml.safe_load((EXAMPLE / "truss" / "qwen35-9b-lora" / "config.yaml").read_text())


def test_truss_config_serves_tool_calls_for_evaluate_py():
    start = _truss_config()["docker_server"]["start_command"]
    # evaluate.py sends tools with tool_choice "auto"; vLLM refuses that without these.
    assert "--enable-auto-tool-choice" in start
    assert "--tool-call-parser qwen3_coder" in start
    assert "--lora-modules benchflow-sft=/app/lora/benchflow-sft" in start


def test_truss_config_pins_its_image_and_weights():
    config = _truss_config()
    assert not config["base_image"]["image"].endswith((":latest", ":nightly"))
    for weight in config["weights"]:
        assert re.search(r"@[0-9a-f]{40}$", weight["source"]), weight["source"]
    mounts = {w["mount_location"] for w in config["weights"]}
    start = config["docker_server"]["start_command"]
    assert all(mount in start for mount in mounts)
