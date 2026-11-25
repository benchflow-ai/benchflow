"""On a real Docker daemon: an ACP agent's trial whose verifier runs in its own
sandbox keeps the verifier's reward under ``--integrity audit`` and ``off``.

In those modes the agent's turn end stops none of its processes, and the
separate verifier's overlap order stops the agent's sandbox before the
verifier runs. The trial's last disconnect then found the stop still pending
and ran it in a sandbox that was gone: compose answered ``service "main" is
not running``, the cleanup failed, the trial ended as an infrastructure
failure with no reward, and the default retry ran the agent again.

The agent is a shell script speaking ACP that writes the answer at its
prompt; the verifier scores it 1.0. The tests skip where no Docker daemon
answers. Everything they start is the trial's own sandbox, removed by the
trial.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from tests.test_cloud_metadata_failclosed_docker import ACP_SH, MANIFEST, _docker_ready

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
]

# What the stub agent does at its prompt, as the sandbox user.
ANSWER_SH = "#!/bin/sh\necho done > /app/submission/answer.txt\n"

DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends procps iptables \\
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app/submission && chmod 0777 /app/submission
COPY acp.sh probe.sh /opt/bfprobe/
RUN chmod 755 /opt/bfprobe /opt/bfprobe/acp.sh /opt/bfprobe/probe.sh
WORKDIR /app
"""

VERIFIER_DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends procps iptables \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
"""

TEST_SH = """#!/bin/sh
if [ "$(cat /app/submission/answer.txt 2>/dev/null)" = "done" ]; then r=1.0; else r=0.0; fi
printf '{"reward": %s}\\n' "$r" > /logs/verifier/reward.json
"""

TASK_MD = """---
schema_version: "1.3"
task:
  name: benchflow/separate-verifier-stopped-agent-{tag}
  version: "0.1.0"
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 300
  sandbox_mode: separate
  sandbox:
    workdir: /app
    network_mode: no-network
sandbox:
  workdir: /app
  network_mode: no-network
---
# Write one file

Create `/app/submission/answer.txt` containing `done`.
"""


def _task(root: Path) -> Path:
    tag = uuid.uuid4().hex[:8]
    task = root / f"sepverif-{tag}"
    (task / "environment").mkdir(parents=True)
    (task / "verifier").mkdir()
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "environment" / "acp.sh").write_text(ACP_SH)
    (task / "environment" / "probe.sh").write_text(ANSWER_SH)
    (task / "verifier" / "Dockerfile").write_text(VERIFIER_DOCKERFILE)
    test_sh = task / "verifier" / "test.sh"
    test_sh.write_text(TEST_SH)
    test_sh.chmod(0o755)
    (task / "task.md").write_text(TASK_MD.format(tag=tag))
    return task


def _bench() -> Path:
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no bench script beside this Python")
    return bench


def _run(tmp_path: Path, task: Path, integrity: str) -> tuple[Path, dict, str]:
    """One trial of the stub agent; its folder, result.json and the output."""
    agents = tmp_path / "agents" / "bfprobe"
    agents.mkdir(parents=True)
    (agents / "manifest.toml").write_text(MANIFEST)
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        HARBOR_TELEMETRY="off",
        DO_NOT_TRACK="1",
        PYTHONUNBUFFERED="1",
        BENCHFLOW_AGENTS_DIR=str(agents.parent),
        BENCHFLOW_AGENTS_SOURCE=str(agents.parent),
        BENCHFLOW_RESTRICTED_WEB_MODELS="openai/gpt-4.1-mini",
    )
    if "PYTHONPATH" in os.environ:  # a tree under test, not this venv's
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    jobs = tmp_path / "jobs"
    done = subprocess.run(
        [
            str(_bench()),
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "bfprobe",
            "--sandbox",
            "docker",
            "--integrity",
            integrity,
            "--concurrency",
            "1",
            "--retry-attempts",
            "0",
            "--usage-tracking",
            "off",
            "--agent-env",
            "OPENAI_API_KEY=placeholder",
            "--agent-env",
            "BENCHFLOW_PROVIDER_BASE_URL=http://127.0.0.1:9/v1",
            "--agent-env",
            "HARBOR_TELEMETRY=off",
            "--agent-env",
            "DO_NOT_TRACK=1",
            "--jobs-dir",
            str(jobs),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = done.stdout + done.stderr
    results = [p for p in jobs.rglob("result.json") if "__" in p.parent.name]
    assert len(results) == 1, ("one trial", results, output[-6000:])
    return results[0].parent, json.loads(results[0].read_text()), output


@pytest.mark.parametrize("integrity", ["audit", "off"])
def test_the_trial_keeps_the_separate_verifiers_reward(tmp_path, integrity):
    """The agent's sandbox stopped before the verifier, so the trial's end
    has nothing left in it to stop: the verifier's 1.0 is the trial's reward
    and the cleanup does not fail (before: ``trial cleanup failed``,
    ``service "main" is not running``, rewards null, an infrastructure
    failure)."""
    trial, result, output = _run(tmp_path, _task(tmp_path), integrity)
    tail = output[-6000:]

    # The path this row guards: the agent's sandbox stopped before the verifier.
    record_file = trial / "verifier-sandbox" / "verifier-sandbox.json"
    record = json.loads(record_file.read_text())
    assert record["order"] == "overlap", record
    assert record["agent_sandbox_alive_during_verify"] is False, record
    assert record["custody"]["agent_stop_error"] is None, record
    # The agent ran and the verifier, in its own sandbox, scored its answer.
    verifier_reward = json.loads((trial / "verifier" / "reward.json").read_text())
    assert verifier_reward == {"reward": 1.0}, tail
    assert result.get("verifier_isolation") == "separate", tail
    assert result.get("rewards") == {"reward": 1.0}, (result.get("error"), tail)
    assert not result.get("error"), tail
    assert result.get("error_category") is None, (result.get("error"), tail)
    assert "trial cleanup failed" not in json.dumps(result), tail
    assert "trial cleanup failed" not in output
    assert "is not running" not in output
    assert "Credential cleanup at disconnect failed" not in output
