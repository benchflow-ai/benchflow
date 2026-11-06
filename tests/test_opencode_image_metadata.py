"""The gateway model retains declared image input."""

import base64
import json
import os
import re
import shutil
import subprocess

import pytest

from benchflow.agents.registry import AGENTS, OPENCODE_PROXY_PROVIDER_ID


@pytest.mark.parametrize("inputs", [["text", "image"], ["text"], None])
def test_proxy_model_receives_only_declared_modalities(tmp_path, inputs):
    if shutil.which("node") is None:
        pytest.skip("requires Node")
    match = re.search(
        r"printf '%s' '([A-Za-z0-9+/=]+)' \| base64 -d > \S*/opencode-proxy",
        AGENTS["opencode"].install_cmd,
    )
    assert match
    wrapper = base64.b64decode(match[1]).decode()
    js = wrapper.split("<<'JSEOF'\n", 1)[1].split("\nJSEOF\n", 1)[0]
    entry = {"id": "alias", **({"input": inputs} if inputs is not None else {})}
    result = subprocess.run(
        ["node", "-e", js],
        env={
            **os.environ,
            "BENCHFLOW_AGENT_HOME": str(tmp_path),
            "BENCHFLOW_LITELLM_MODEL_ALIAS": "alias",
            "BENCHFLOW_PROVIDER_MODELS": json.dumps(
                [{"id": "other", "input": ["image"]}, entry]
            ),
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    model = json.loads((tmp_path / ".config/opencode/opencode.json").read_text())[
        "provider"
    ][OPENCODE_PROXY_PROVIDER_ID]["models"]["alias"]
    if inputs is None:
        assert "modalities" not in model and "attachment" not in model
    else:
        assert model["modalities"] == {"input": inputs, "output": ["text"]}
        assert model["attachment"] is ("image" in inputs)
