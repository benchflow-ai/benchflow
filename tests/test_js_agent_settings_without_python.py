"""Guards #1047: JS agent settings use their installed Node runtime."""

import json
import os
import shutil
import subprocess

import pytest

from benchflow.agents.registry import AGENTS


@pytest.mark.parametrize(
    "agent,relative,key",
    [
        ("gemini", ".gemini/settings.json", "exclude"),
        ("opencode", ".config/opencode/opencode.json", "webfetch"),
        ("mimo", ".config/mimocode/mimocode.json", "webfetch"),
    ],
)
@pytest.mark.parametrize(
    "policy", ["disallow_web_tools_setup_cmd", "disallow_hosted_search_setup_cmd"]
)
def test_settings_without_image_python(tmp_path, agent, relative, key, policy):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node")
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "node").symlink_to(node)
    settings = tmp_path / relative
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"keep": 7, "tools": {"other": True}}))
    command = getattr(AGENTS[agent], policy)
    # Model the fixed interpreter locations of an image with no Python.
    command = (
        command.replace("/usr/bin/python3", "/missing/python3")
        .replace("/usr/local/bin/python3", "/missing/local-python3")
        .replace("/opt/runtime/node/bin/node", str(bins / "node"))
    )
    for _ in range(2):
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            env={
                **os.environ,
                "PATH": str(bins),
                "BENCHFLOW_AGENT_HOME": str(tmp_path),
            },
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
    data = json.loads(settings.read_text())
    assert data["keep"] == 7 and data["tools"]["other"] is True
    if agent == "gemini":
        assert data["tools"][key] == ["google_web_search", "web_fetch"]
    else:
        key = "websearch" if "hosted" in policy else key
        assert data["tools"][key] is False
