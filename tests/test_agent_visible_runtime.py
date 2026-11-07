"""Host aliases never become agent-visible identifiers."""

import base64
import os
import re
import subprocess

from benchflow._runtime_tree import RUNTIME_ROOT
from benchflow.agents import registry
from benchflow.agents.env import resolve_agent_env


def test_managed_paths_do_not_identify_runner():
    assert "benchflow" not in RUNTIME_ROOT.lower()
    for config in registry.AGENTS.values():
        assert "/opt/benchflow" not in config.install_cmd
        assert "/opt/benchflow" not in config.launch_cmd


def test_launched_process_has_neutral_env_and_paths():
    from benchflow.agents.visible_runtime import visible_env, visible_launch

    host = {
        "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000",
        "BENCHFLOW_PROVIDER_MODEL": "m",
        "PATH": "/usr/bin:/bin",
    }
    child = subprocess.check_output(
        ["/bin/sh", "-c", visible_launch("env")],
        env={
            **os.environ,
            "BENCHFLOW_OLD_IMAGE_FLAG": "yes",
            "_BENCHFLOW_IMAGE_PATH": "/usr/bin",
            **visible_env(host),
        },
        text=True,
    )
    # Scan process output, including inherited image env, not just a dict.
    assert not re.search(r"(?im)^[^=]*benchflow[^=]*=", child)
    assert "RUNTIME_PROVIDER_BASE_URL=http://127.0.0.1:4000" in child
    assert "RUNTIME_PROVIDER_MODEL=m" in child


def test_deployed_launchers_read_only_neutral_variable_names():
    for name in (
        "pi-acp",
        "deepagents",
        "harvey-lab-harness",
        "muse-code",
        "antigravity",
        "opencode",
        "mimo",
        "codex-acp",
    ):
        config = registry.AGENTS.get(name)
        if config is None:
            continue
        for encoded in re.findall(
            r"[A-Za-z0-9+/]{120,}={0,2}", config.install_cmd + config.launch_cmd
        ):
            try:
                source = base64.b64decode(encoded).decode()
            except (ValueError, UnicodeDecodeError):
                continue
            assert "BENCHFLOW_" not in source, name
            assert "/opt/benchflow" not in source, name


def test_provider_host_aliases_keep_identical_routing():
    legacy = {
        "BENCHFLOW_PROVIDER_BASE_URL": "http://127.0.0.1:4000",
        "BENCHFLOW_PROVIDER_API_KEY": "test-key",
    }
    neutral = {
        key.replace("BENCHFLOW_", "RUNTIME_"): value for key, value in legacy.items()
    }
    assert resolve_agent_env(
        "codex-acp", "openai/gpt-test", legacy
    ) == resolve_agent_env("codex-acp", "openai/gpt-test", neutral)
