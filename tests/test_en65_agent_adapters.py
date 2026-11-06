"""Guards adapter regressions retained at ac3bedaa1."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.agents import registry, remote_manifests
from benchflow.agents.codex_config import (
    apply_codex_launch_config,
    apply_codex_provider_config,
)


@pytest.mark.parametrize("suffix", ["#new", "?new"])
@pytest.mark.parametrize(
    "variable", ["BENCHFLOW_AGENTS_SOURCE", "BENCHFLOW_AGENTS_DIR"]
)
def test_catalog_uses_complete_local_path(tmp_path, monkeypatch, suffix, variable):
    """Guards ac3bedaa1: URL redaction must not choose a different local agent."""
    root = tmp_path / "catalog"
    for tail, command in (("", "old-agent"), (suffix, "selected-agent")):
        folder = Path(str(root) + tail) / "probe"
        folder.mkdir(parents=True)
        (folder / "manifest.toml").write_text(
            'contract_version = "1.0"\nname = "en65-catalog-probe"\n'
            f'install_cmd = "true"\nlaunch_cmd = "{command}"\n'
        )
    monkeypatch.delenv(remote_manifests.AGENTS_DIR_ENV, raising=False)
    monkeypatch.setenv(variable, str(root) + suffix)
    spec, _ = remote_manifests._effective_source()
    catalog, loaded = remote_manifests._read_catalog(spec)
    assert not catalog.issues
    assert [item.config.launch_cmd for item in loaded.values()] == ["selected-agent"]


@pytest.mark.parametrize(
    "source", ["owner/agents", "https://example.invalid/agents.git"]
)
def test_catalog_preserves_valid_hash_in_git_ref(monkeypatch, tmp_path, source):
    """Guards ac3bedaa1: a valid feature#2 ref reaches source resolution intact."""
    calls = []

    def resolve(repo, ref=None):
        calls.append((repo, ref))
        return tmp_path

    monkeypatch.setattr("benchflow._utils.benchmark_repos.resolve_source", resolve)
    request = remote_manifests._parse_source(source + "@feature#2")
    assert remote_manifests._source_root(request) == tmp_path
    assert calls == [(source, "feature#2")]


def test_url_diagnostics_still_hide_credentials():
    """Guards ac3bedaa1 and PR #1090: source identity is separate from diagnostics."""
    request = remote_manifests._parse_source(
        "https://user:secret@example.invalid/agents.git@feature#private"
    )
    assert request.raw_source == "https://user:secret@example.invalid/agents.git"
    assert request.raw_ref == "feature#private"
    assert request.source == "https://***@example.invalid/agents.git"
    assert request.ref == "feature"


def test_gateway_launcher_preserves_existing_codex_settings(tmp_path):
    """Guards ac3bedaa1/0a56a672: a routed launch cannot erase image config."""
    home = tmp_path / "home"
    config = home / ".codex/config.toml"
    config.parent.mkdir(parents=True)
    original = (
        'model_reasoning_effort = "high"\n'
        '[mcp_servers.task_local]\ncommand = "/app/task-tool"\n'
    )
    config.write_text(original)
    fake = tmp_path / "codex-acp"
    fake.write_text('#!/bin/sh\nprintf "%s" "$CODEX_PATH"\n')
    fake.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "BENCHFLOW_AGENT_HOME": str(home)}
    apply_codex_provider_config(
        env,
        base_url="http://127.0.0.1:8999/v1",
        model="gpt-6-astra",
        provider_name="litellm",
    )
    env, _ = apply_codex_launch_config(
        "codex-acp", env, model="gpt-6-astra", reasoning_effort=None
    )
    # A previous role's binary override must not replace the installed wrapper.
    env["CODEX_PATH"] = "/previous-role/codex"
    done = subprocess.run(
        [
            "sh",
            "-c",
            registry.CODEX_ACP_BUILTIN_LAUNCH.replace(
                "/opt/runtime/bin/codex-acp", str(fake)
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert config.read_text() == original
    assert done.stdout == "/opt/runtime/bin/codex-app-server"


def test_venv_shim_deployment_has_no_system_python_dependency(tmp_path):
    """Guards ac3bedaa1/df6e024f: a venv shim needs only its selected interpreter."""
    target = tmp_path / "probe.py"
    command = registry._install_python_script(
        str(target), "print(42)\n", python=sys.executable
    )
    subprocess.run(["sh", "-c", command], check=True, capture_output=True)
    assert target.read_text() == "print(42)\n"
    assert subprocess.check_output([sys.executable, "-I", "-S", str(target)]) == b"42\n"
    missing = registry._install_python_script(
        str(tmp_path / "missing.py"), "print(42)\n", python=str(tmp_path / "no-python")
    )
    assert subprocess.run(["sh", "-c", missing], capture_output=True).returncode != 0
    assert not (tmp_path / "missing.py").exists()


def test_codex_app_server_config_matches_native_overrides(tmp_path):
    """Guards ac3bedaa1: app-server defaults keep explicit effort and routing."""
    import shutil
    import tomllib

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is installed by Codex ACP; unavailable on this test host")
    fake_cli = tmp_path / "codex.js"
    fake_cli.write_text("process.stdout.write(JSON.stringify(process.argv.slice(2)))")
    launcher = tmp_path / "app-server"
    launcher.write_text(registry.codex_app_server_launcher(node, str(fake_cli)))
    launcher.chmod(0o755)
    config = {
        "model_provider": "benchflow-proxy",
        "model_reasoning_effort": "high",
        "model_providers": {
            "benchflow-proxy": {
                "base_url": "http://127.0.0.1:8999/v1",
                "http_headers": {"X-Probe": 'quoted"value'},
            }
        },
        "mcp_servers": {"task_local": {"command": "/app/tool", "args": ["--test"]}},
    }
    done = subprocess.run(
        [str(launcher), "app-server"],
        env={"PATH": os.environ["PATH"], "CODEX_CONFIG": json.dumps(config)},
        capture_output=True,
        text=True,
        check=True,
    )
    argv = json.loads(done.stdout)
    assert argv[-1] == "app-server"
    assert argv[:-1:2] == ["-c"] * ((len(argv) - 1) // 2)
    actual = tomllib.loads("\n".join(argv[1:-1:2]))
    assert actual == {**config, "openai_base_url": "http://127.0.0.1:8999/v1"}
