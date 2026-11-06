"""Real shell tools preserve task Python and umask.

Guards the system-Python / launch-umask regression present at 01727c09e
(agents PR #80, 3de8bc91). Uses the deterministic fake, no model credentials.
Fresh runs install inside the trial; cached runs prove a cold miss then a hit.
OpenHands' install is not cacheable in this release: its cached case is an
explicit skip, while its fresh case runs the actual terminal tool.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h
from tests.test_deterministic_mini_swe import _plugin_pythonpath

pytestmark = [pytest.mark.deterministic, pytest.mark.integration]
AGENTS = ("mini-swe", "claude-agent-acp", "codex-acp", "openhands")
# Print the failed Python command in the actual tool output, then keep going
# to record whether && incorrectly allowed creation of the task's sentinel.
PROBE = (
    "test ! -e cmain || exit 9; "
    "{ printf 'python3=%s\\n' \"$(command -v python3 || true)\"; "
    "printf 'umask=%s\\n' \"$(umask)\"; "
    "python3 -c pass && touch cmain; "
    "if [ -e cmain ]; then echo cmain=yes; else echo cmain=no; fi; "
    # Check after the real CLI has initialized/authenticated, not just after
    # BenchFlow's writer (092eb18ad, 3c9388059).
    "for p in .codex/auth.json .codex/config.toml .openhands/agent_settings.json; do "
    '[ ! -f "$HOME/$p" ] || printf "credential-mode=%s:%s\\n" '
    '"$(stat -c %a "$HOME/$p")" "$p"; done; '
    "} 2>&1 | tee /app/tool-environment.txt"
)


def docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, timeout=600
    ).stdout


@pytest.fixture(scope="module", params=["python-free", "task-python"])
def task_image(request, tmp_path_factory):
    sandbox, reason = h.select_sandbox()
    if sandbox != "docker":
        (
            pytest.fail
            if sandbox is None and os.environ.get(h.SANDBOX_ENV) == "docker"
            else pytest.skip
        )(reason if sandbox is None else "requires real Docker")
    root = tmp_path_factory.mktemp("tool-env-image")
    tag = f"ate-fix-{request.param}-{uuid.uuid4().hex[:12]}"
    # The unique label guarantees a new content digest and a cold agent layer.
    source = (
        "FROM alexgshaw/polyglot-c-py:20251031\n"
        if request.param == "python-free"
        else "FROM python:3.13.9-slim-bookworm\n"
    )
    (root / "Dockerfile").write_text(
        source + f'WORKDIR /app\nLABEL benchflow.test.tool-env="{tag}"\n'
    )
    docker("build", "-q", "-t", tag, str(root))
    baseline = docker(
        "run", "--rm", "--name", tag + "-baseline", tag, "sh", "-c", PROBE
    )
    (root / "baseline.txt").write_text(baseline)
    if request.param == "task-python":
        # A task-owned 3.13 interpreter, distinct from the agent's private 3.12.
        # Custom PATH/login-shell policy belongs to the 0.8.4 contract work.
        version = docker(
            "run", "--rm", "--name", tag + "-version", tag, "python3", "--version"
        )
        (root / "python-version.txt").write_text(version)
        assert version.strip() == "Python 3.13.9"
    keys = set()
    try:
        yield tag, baseline, request.param, keys
    finally:
        from benchflow.sandbox.env_layers import AGENT, layer_tag

        (root / "layer-keys.json").write_text(json.dumps(sorted(keys)))
        # Only layers recorded by this fixture, whose unique base label prevents
        # sharing with other test processes. Never prune unrelated images.
        for key in sorted(keys):
            docker("image", "rm", layer_tag(AGENT, key))
        docker("image", "rm", tag)


def _receipt(text: str) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in text.splitlines()
        if line.startswith(("python3=", "umask=", "cmain="))
    )


def _task(root: Path, image: str, script: str) -> Path:
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment" / "Dockerfile").write_text(f"FROM {image}\n")
    (task / "task.md").write_text(f"""---
schema_version: '1.0'
metadata:
  name: tool-environment
  category: sanity
verifier:
  type: test-script
  timeout_sec: 60
agent:
  timeout_sec: 180
sandbox:
  docker_image: {image}
  cpus: 1
  memory_mb: 4096
  build_timeout_sec: 900
---

## prompt

Run the requested shell probe in /app. [[fake-llm:{script}]]
""")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        "cp /app/tool-environment.txt /logs/verifier/tool-environment.txt\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    return task


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("mode", ["fresh", "cached"])
def test_agent_tool_environment(task_image, agent, mode, tmp_path, monkeypatch):
    if agent == "openhands" and mode == "cached":
        pytest.skip(
            "OpenHands' installer is not cacheable in 0.8.3; fresh terminal tested"
        )
    image, baseline, kind, keys = task_image
    script = "tool-env-" + agent
    tool = {"mini-swe": "bash", "openhands": "terminal"}.get(agent, "Bash")
    steps = [
        {
            "text": "Inspect the task environment.",
            "tool": tool,
            "input": {"command": PROBE},
        }
    ]
    steps.append(
        {
            "tool": "bash",
            "input": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
        }
        if agent == "mini-swe"
        else {"text": "Probe complete."}
    )
    load = h._load_fake_llm_module

    def fake_module():
        module = load()
        original = module._Handler.do_POST

        # host_fake_llm loads standard scripts after this factory returns.
        def post(handler):
            handler.scripts[script] = steps
            return original(handler)

        module._Handler.do_POST = post
        return module

    monkeypatch.setattr(h, "_load_fake_llm_module", fake_module)
    task = _task(tmp_path, image, script)
    route = {"codex-acp": "codex", "claude-agent-acp": "proxy"}.get(agent, "vllm")
    plugin_env = (
        {"PYTHONPATH": _plugin_pythonpath(tmp_path / "plugin")}
        if agent == "mini-swe"
        else {}
    )
    layers = []
    for phase in ["fresh"] if mode == "fresh" else ["cold", "warm"]:
        extra = ["--retry-attempts", "0"]
        if phase == "fresh":
            extra.append("--fresh-install")
        if agent == "openhands":
            extra += ["--agent-env", "BENCHFLOW_OPENHANDS_DISABLE_SUBAGENTS=1"]
        with h.host_fake_llm(tmp_path / f"{phase}-fake.jsonl") as url:
            run = h.run_bench(
                ["eval", "run"],
                tasks_dir=task,
                jobs_dir=tmp_path / phase,
                sandbox="docker",
                host_fake_url=url,
                route=route,
                agent=agent,
                extra=extra,
                env_extra=plugin_env,
            )
        trial = run.trial(task.name)
        config = h.read_json(trial / "config.json")
        layer = config.get("layers", {}).get("agent")
        if layer and layer.get("key") and layer.get("status") in {"hit", "miss"}:
            keys.add(layer["key"])
        result = h.read_json(trial / "result.json")
        assert result.get("error") is None, (result, run.output[-6000:])
        assert run.returncode == 0, run.output[-6000:]
        text = (trial / "verifier" / "tool-environment.txt").read_text()
        credential_modes = {
            line.removeprefix("credential-mode=")
            for line in text.splitlines()
            if line.startswith("credential-mode=")
        }
        if agent == "codex-acp":
            assert {"600:.codex/auth.json", "600:.codex/config.toml"}.issubset(
                credential_modes
            ), text
        elif agent == "openhands":
            assert "600:.openhands/agent_settings.json" in credential_modes, text
        assert _receipt(text) == _receipt(baseline), (baseline, text)
        assert _receipt(text)["python3"] == (
            "" if kind == "python-free" else "/usr/local/bin/python3"
        )
        assert _receipt(text)["cmain"] == ("no" if kind == "python-free" else "yes")
        if kind == "python-free":
            assert "not found" in text, text
        calls = h.llm_calls(trial)
        assert any(
            tc["arguments"].get("command") == PROBE
            or tc["arguments"].get("cmd") == PROBE
            for c in calls
            for tc in c["tool_calls"]
        ), calls
        if mode == "cached":
            layers.append(h.read_json(trial / "config.json")["layers"]["agent"])
    if mode == "cached":
        assert [layer["status"] for layer in layers] == ["miss", "hit"], layers
        assert all(layer["install"] == "from layer" for layer in layers), layers
        assert layers[0]["key"] == layers[1]["key"], layers
