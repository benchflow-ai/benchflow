"""The agent's sandbox cannot reach a separate verifier sandbox (PY8 SV3).

On Docker SV3 starts the verifier sandbox while the agent's sandbox still
exists (``VerifierSandboxRun.begin_start`` at verify() entry). Both are
Compose projects of their own: the agent's, and ``<rollout>-verifier``, each
on its own default network. This sets the two up the way SV3 does (the
rollout's ``create_environment`` for the agent, ``VerifierSandboxRun`` for the
verifier), starts listeners in the verifier container, and probes from the
agent container: no TCP connection to the verifier's address on ports 1-1024
or on any port it listens on, and no name of the verifier container resolves
to it. A probe from the verifier container to its own address is the
positive control: it reaches every listener.

Deterministic tier (``tests/integration/README.md``): real Docker, no model;
skips with a reason where no Docker daemon answers.
"""

from __future__ import annotations

import asyncio
import shlex
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.contracts.planes import default_rollout_planes
from benchflow.rollout import _separate_verifier as sv
from benchflow.task import Task
from benchflow.task.paths import RolloutPaths
from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
LISTEN_PORTS = (80, 8080, 47123)

# ubuntu:24.04 is the deterministic tier's base image, so its hosts hold it.
DOCKERFILE = "FROM ubuntu:24.04\n"
TESTS_DOCKERFILE = "FROM ubuntu:24.04\nCOPY test.sh /tests/test.sh\n"

# Listeners in the verifier container: perl-base is in ubuntu:24.04.
LISTEN = (
    "setsid nohup perl -MIO::Socket::INET -e "
    + shlex.quote(
        "my @s = map { IO::Socket::INET->new(LocalAddr => '0.0.0.0', "
        'LocalPort => $_, Listen => 16, ReuseAddr => 1) or die "$_: $!" } '
        "@ARGV; sleep 600"
    )
    + " "
    + " ".join(str(p) for p in LISTEN_PORTS)
    + " >/dev/null 2>&1 &"
)

# The ports a container listens on, from /proc (no ss in the image).
LISTENING = r"""
for f in /proc/net/tcp /proc/net/tcp6; do
  [ -r "$f" ] || continue
  tail -n +2 "$f" | while read -r _ local _ st _; do
    [ "$st" = 0A ] && echo $((16#${local##*:}))
  done
done | sort -un
"""

# Connect to "$1" on every other argument, 128 at a time, 2 s each; print the
# ports that accept.
CONNECT = r"""
ip=$1; shift; n=0
for p in "$@"; do
  ( timeout 2 bash -c "exec 3<>/dev/tcp/$ip/$p" 2>/dev/null && echo "open $p" ) &
  n=$((n + 1)); [ $((n % 128)) -eq 0 ] && wait
done
wait
"""

# Resolve every argument; print what resolves.
RESOLVE = r"""
for name in "$@"; do
  getent hosts "$name" | while read -r addr _; do echo "resolved $name $addr"; done
done
"""


@pytest.fixture
def require_docker() -> None:
    if SANDBOX != "docker":
        reason = SKIP_REASON if SANDBOX is None else f"Docker-only scenario ({SANDBOX})"
        pytest.skip(reason)


async def _docker(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    assert proc.returncode == 0, out
    return out.decode(errors="replace").strip()


async def _main_container(project: str) -> str:
    names = await _docker(
        "ps",
        "--filter",
        f"label=com.docker.compose.project={project}",
        "--filter",
        "label=com.docker.compose.service=main",
        "--format",
        "{{.Names}}",
    )
    assert len(names.split()) == 1, names
    return names


async def _bash(env, script: str, *args: str) -> list[str]:
    result = await env.exec(
        shlex.join(["bash", "-c", script, "probe", *args]),
        user="root",
        timeout_sec=300,
    )
    assert result.return_code == 0, result
    return (result.stdout or "").split("\n")


def _task(tmp_path: Path) -> Path:
    # Its own name: the image tag (bf__<name>) is not shared with another test.
    task = tmp_path / "sv3-reach-task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "tests" / "Dockerfile").write_text(TESTS_DOCKERFILE)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "instruction.md").write_text("nothing\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\nenvironment_mode = "separate"\n'
    )
    return task


@pytest.mark.usefixtures("require_docker")
async def test_the_agent_cannot_reach_a_started_verifier_sandbox(tmp_path):
    task_dir = _task(tmp_path)
    task = Task(task_dir)
    planes = default_rollout_planes()
    name = f"sv3-reach-{uuid.uuid4().hex[:8]}"
    paths = RolloutPaths(rollout_dir=tmp_path / "trial")
    paths.mkdir()
    # The agent's sandbox, as Rollout.setup creates it.
    agent = planes.create_environment(
        "docker",
        task,
        task_dir,
        name,
        paths,
        preserve_agent_network=False,
        environment_manifest=None,
    )
    # The verifier's, as Rollout.verify starts it on Docker.
    rollout = SimpleNamespace(
        _rollout_dir=paths.rollout_dir,
        _agent_cwd="/app",
        _timing={},
        _task=task,
        _rollout_name=name,
        _planes=planes,
        _config=SimpleNamespace(
            environment="docker",
            context_root=None,
            base_image_override=None,
            sandbox_setup_timeout=600,
        ),
    )
    run = sv.VerifierSandboxRun(rollout, order=sv.OVERLAP)
    try:
        await agent.start(force_build=False)
        run.begin_start()
        assert run.start_task is not None
        await run.start_task
        verifier = run.env
        assert verifier.compose_project_name != agent.compose_project_name

        await verifier.exec(LISTEN, user="root", timeout_sec=30)
        for _ in range(50):
            listening = {int(p) for p in await _bash(verifier, LISTENING) if p.strip()}
            if set(LISTEN_PORTS) <= listening:
                break
            await asyncio.sleep(0.2)
        assert set(LISTEN_PORTS) <= listening, listening

        container = await _main_container(verifier.compose_project_name)
        addresses = (
            await _docker(
                "inspect",
                "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
                container,
            )
        ).split()
        hostname = await _docker("inspect", "-f", "{{.Config.Hostname}}", container)
        assert addresses, container
        ports = [str(p) for p in sorted(set(range(1, 1025)) | listening)]

        # Positive controls: the verifier reaches its own listeners this way...
        for address in addresses:
            own = await _bash(verifier, CONNECT, address, *ports)
            assert {f"open {p}" for p in LISTEN_PORTS} <= set(own), own

        # ...and resolves its own names.
        own_names = await _bash(verifier, RESOLVE, container, hostname)
        assert any(line.startswith(f"resolved {container} ") for line in own_names)
        assert any(line.startswith(f"resolved {hostname} ") for line in own_names)

        # From the agent: no connection, by address...
        for address in addresses:
            seen = [
                line for line in await _bash(agent, CONNECT, address, *ports) if line
            ]
            assert seen == [], (address, seen)
        # ...and no name of the verifier container resolves to it.
        names = [
            container,
            hostname,
            f"{verifier.compose_project_name}-main-1",
            f"{verifier.compose_project_name}_main_1",
            "main",
        ]
        resolved = [
            line for line in await _bash(agent, RESOLVE, *names) if line.strip()
        ]
        print(
            f"reach probe: verifier {container} at {addresses}, listening "
            f"{sorted(listening)}; {len(ports)} ports probed from the agent: "
            f"0 open; resolved from the agent: {resolved}"
        )
        # "main" is the agent's own service: it may resolve, never to the
        # verifier.
        for line in resolved:
            _, which, address = line.split()
            assert which == "main", resolved
            assert address not in addresses, resolved
    finally:
        await run.discard(cancelled=False)
        await agent.stop(delete=True)
