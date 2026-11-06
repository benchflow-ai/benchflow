"""Opt-in network spike without changing the public default."""

import json
from types import SimpleNamespace

import pytest

from benchflow.experimental.agent_network import run_arguments
from benchflow.sandbox.egress_denylist import agent_network_sandbox_config
from benchflow.task.config import TaskConfig


def test_flag_disabled_keeps_arguments_and_public_default():
    args = ["task", "--model", "example/model"]
    assert run_arguments(args, no_network_by_default=False) == args
    assert TaskConfig().sandbox.network_mode == "public"
    assert TaskConfig().agent.network_mode is None


def test_flag_records_agent_policy_while_setup_and_verifier_stay_online():
    from benchflow.rollout._setup import _task_agent_offline, _task_disallows_internet

    args = run_arguments(["task"], no_network_by_default=True)
    assert args[1] == "--config-override"
    config = TaskConfig.model_validate(json.loads(args[2]))
    task = SimpleNamespace(config=config)
    assert _task_disallows_internet(task)
    assert _task_agent_offline(task)
    assert agent_network_sandbox_config(config).network_mode == "no-network"
    assert config.sandbox.network_mode == "public"
    assert config.sandbox.allow_internet is not False
    assert config.verifier.network_mode is None
    assert json.loads(config.model_dump_json())["agent"]["network_mode"] == "no-network"


@pytest.mark.parametrize(
    "option",
    ["--config=x.json", "--config", "--config-override={}", "--config-override"],
)
def test_spike_refuses_to_overwrite_another_overlay(option):
    with pytest.raises(ValueError, match="cannot combine"):
        run_arguments(["task", option], no_network_by_default=True)
