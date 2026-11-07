"""``--sandbox remote-docker``: tasks on a Docker host reached over SSH.

The scenario brings its own host: a Daytona sandbox from the public
``docker:dind`` image running ``dockerd``, reached through Daytona's SSH
access (``ssh <token>@<gateway>``). ``BENCHFLOW_E2E_REMOTE_DOCKER_HOST``
(an ``ssh://`` URL your ssh setup can reach) uses an existing host instead.
The SSH user of a Daytona access is a token, so every scenario also checks
that it appears in no job file and no CLI output.

Scenarios: an oracle batch (plain, ``no-network``, separate verifier
sandbox) and a ``nop`` control through ``bench eval run``; the host-side
refusals (a task larger than the host, an unreachable host before a job
exists); ``network_mode: allowlist`` armed on a remote container through the
provider API; and, after each run, nothing of the rollout left on the host.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e import harness as h

HOST_ENV = "BENCHFLOW_E2E_REMOTE_DOCKER_HOST"
DIND_IMAGE = "docker:28.3.3-dind"

NET_SOLVE = """#!/bin/bash
python3 - <<'PY'
import urllib.request
try:
    urllib.request.urlopen("https://example.com/", timeout=5)
    result = "reachable"
except Exception:
    result = "blocked"
open("/app/net.txt", "w").write(result)
PY
"""
NET_TEST = """#!/bin/bash
cat /app/net.txt
if [ "$(cat /app/net.txt)" = blocked ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""
SEPARATE_SOLVE = h.HELLO_SOLVE + "touch /tmp/planted-by-agent\n"
SEPARATE_TEST = """#!/bin/bash
ok=0
if [ "$(tr -d '\\n' < /app/hello.txt)" = "Hello, world!" ] && [ ! -e /tmp/planted-by-agent ]; then ok=1; fi
echo "planted=$([ -e /tmp/planted-by-agent ] && echo yes || echo no)"
echo $ok > /logs/verifier/reward.txt
"""
ALLOWLIST_DOCKERFILE = (
    "FROM ubuntu:24.04\n"
    "RUN apt-get update -qq && apt-get install -y -qq curl python3 openssl "
    "iptables ca-certificates && rm -rf /var/lib/apt/lists/*\n"
    "WORKDIR /app\n"
)


@dataclass(repr=False)
class RemoteHost:
    url: str
    env: dict[str, str]  # PATH (ssh options shim) and BENCHFLOW_REMOTE_DOCKER_HOST
    secret: str | None  # the ssh user when it is an access token
    dedicated: bool  # created by this module: nothing else runs on it


def _ssh_shim(root: Path) -> str:
    """An ``ssh`` first on PATH that adds batch mode and a private known_hosts."""
    real = shutil.which("ssh")
    if real is None:
        pytest.skip("remote-docker scenarios need an ssh client")
    shim = root / "ssh-shim"
    shim.mkdir(parents=True, exist_ok=True)
    script = shim / "ssh"
    script.write_text(
        "#!/bin/sh\n"
        f'exec {real} -o BatchMode=yes -o UserKnownHostsFile="{root}/known_hosts" '
        '-o StrictHostKeyChecking=accept-new "$@"\n'
    )
    script.chmod(0o755)
    return f"{shim}{os.pathsep}{os.environ.get('PATH', '')}"


def _daytona_dind_host(root: Path) -> Iterator[RemoteHost]:
    from benchflow.sandbox.daytona import build_sync_client
    from benchflow.sandbox.daytona_reaper import _benchflow_owned_labels

    client = build_sync_client()
    from daytona import CreateSandboxFromImageParams, Image, Resources

    sandbox = client.create(
        CreateSandboxFromImageParams(
            image=Image.base(DIND_IMAGE),
            resources=Resources(cpu=2, memory=4, disk=10),
            labels=_benchflow_owned_labels(),
            auto_stop_interval=30,
            auto_delete_interval=60,
            network_block_all=False,
        ),
        timeout=600,
    )
    try:
        sandbox.process.exec(
            "sh -c 'dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &'",
            timeout=10,
        )
        deadline = time.monotonic() + 120
        while sandbox.process.exec("docker info", timeout=10).exit_code != 0:
            assert time.monotonic() < deadline, "dockerd did not start"
            time.sleep(2)
        access = sandbox.create_ssh_access(expires_in_minutes=120)
        target = access.ssh_command.split()[-1]
        user, gateway = target.split("@", 1)
        url = f"ssh://{user}@{gateway}"
        yield RemoteHost(
            url=url,
            env={"PATH": _ssh_shim(root), "BENCHFLOW_REMOTE_DOCKER_HOST": url},
            secret=user,
            dedicated=True,
        )
    finally:
        client.delete(sandbox)


@pytest.fixture(scope="module")
def remote(sandbox: str, e2e_out: Path) -> Iterator[RemoteHost]:
    root = e2e_out / "remote-docker"
    root.mkdir(exist_ok=True)
    configured = os.environ.get(HOST_ENV, "").strip()
    if configured:
        yield RemoteHost(
            url=configured,
            env={"PATH": _ssh_shim(root), "BENCHFLOW_REMOTE_DOCKER_HOST": configured},
            secret=None,
            dedicated=False,
        )
        return
    yield from _daytona_dind_host(root)


def _docker(remote: RemoteHost, *args: str) -> str:
    env = h.clean_env(remote.env)
    env["DOCKER_HOST"] = remote.url
    env.pop("DOCKER_CONTEXT", None)
    proc = subprocess.run(
        ["docker", *args], env=env, capture_output=True, text=True, timeout=120
    )
    detail = proc.stderr[-500:]
    if remote.secret:
        detail = detail.replace(remote.secret, "***")
    assert proc.returncode == 0, (args, detail)
    return proc.stdout


def _assert_nothing_left(remote: RemoteHost, job: Path) -> None:
    """No container, network or volume of any trial of ``job`` on the host."""
    for trial in h.trial_dirs(job):
        # The trial's own project and its separate verifier sandbox's.
        for project in (trial.name.lower(), f"{trial.name.lower()}-verifier"):
            label = f"label=com.docker.compose.project={project}"
            for kind in (
                ["ps", "-aq"],
                ["network", "ls", "-q"],
                ["volume", "ls", "-q"],
            ):
                assert _docker(remote, *kind, "--filter", label).split() == [], (
                    project,
                    kind,
                )
    if remote.dedicated:
        assert _docker(remote, "ps", "-aq").split() == []
        assert _docker(remote, "volume", "ls", "-q").split() == []


def _assert_secret_absent(remote: RemoteHost, *places: Path | str) -> None:
    if not remote.secret:
        return
    for place in places:
        if isinstance(place, str):
            assert remote.secret not in place
            continue
        for path in [place] if place.is_file() else place.rglob("*"):
            if path.is_file():
                assert remote.secret not in path.read_text(errors="replace"), path


def _run(remote: RemoteHost, tasks: Path, job: Path, agent: str, log: Path):
    h.clear_job(job)
    return h.bench(
        "eval", "run", "--tasks-dir", tasks, "--agent", agent,
        "--sandbox", "remote-docker", "--jobs-dir", job.parent, "--job-name", job.name,
        "--concurrency", "3", "--max-sandbox-seconds", str(h.cap_seconds()),
        log=log, env=remote.env,
    )  # fmt: skip


def test_remote_docker_oracle_batch_nop_and_cleanup(
    remote: RemoteHost, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    root = tasks_root / "remote-docker"
    h.write_task(root, "e2e-rd-pass")
    h.write_task(
        root, "e2e-rd-no-network", solve=NET_SOLVE, test=NET_TEST,
        frontmatter={"sandbox": {"network_mode": "no-network"}},
    )  # fmt: skip
    h.write_task(
        root, "e2e-rd-separate", solve=SEPARATE_SOLVE, test=SEPARATE_TEST,
        frontmatter={"verifier": {"sandbox_mode": "separate"}},
    )  # fmt: skip
    job = jobs_root / "remote-docker-oracle"
    run = _run(remote, root, job, "oracle", jobs_root / "remote-docker-oracle.log")
    ledger.record("remote-docker oracle batch (plain, no-network, separate verifier)",
                  surface="CLI", seconds=run.seconds, job_dir=job)  # fmt: skip
    h.assert_exit(run, 0)
    for name in ("e2e-rd-pass", "e2e-rd-no-network", "e2e-rd-separate"):
        result = h.read_json(h.trial_of(job, name) / "result.json")
        assert result["rewards"] == {"reward": 1.0}, (name, result.get("error"))
    separate = h.trial_of(job, "e2e-rd-separate")
    assert "planted=no" in (separate / "verifier" / "test-stdout.txt").read_text()
    record = h.read_json(separate / "verifier-sandbox" / "verifier-sandbox.json")
    assert record["status"] == "complete"
    # Logs came back without a bind mount.
    assert (h.trial_of(job, "e2e-rd-pass") / "verifier" / "reward.txt").is_file()
    _assert_nothing_left(remote, job)
    _assert_secret_absent(remote, job, run.output)

    nop_job = jobs_root / "remote-docker-nop"
    only_pass = root / "e2e-rd-pass"
    run = _run(remote, only_pass, nop_job, "nop", jobs_root / "remote-docker-nop.log")
    ledger.record("remote-docker nop control", surface="CLI", seconds=run.seconds,
                  job_dir=nop_job)  # fmt: skip
    result = h.read_json(h.trial_of(nop_job, "e2e-rd-pass") / "result.json")
    assert result["rewards"] == {"reward": 0.0}, result.get("error")
    _assert_nothing_left(remote, nop_job)


def test_remote_docker_refuses_an_oversized_task_and_an_unreachable_host(
    remote: RemoteHost, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    root = tasks_root / "remote-docker-refusals"
    task = h.write_task(root, "e2e-rd-too-big", frontmatter={"sandbox": {"cpus": 4096}})
    job = jobs_root / "remote-docker-too-big"
    run = _run(remote, task, job, "oracle", jobs_root / "remote-docker-too-big.log")
    ledger.record("remote-docker task larger than the host", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    result = h.read_json(h.trial_of(job, "e2e-rd-too-big") / "result.json")
    assert "Remote Docker host lacks resources: task needs 4096 CPUs" in str(
        result.get("error")
    ), result.get("error")
    assert result.get("rewards") is None
    _assert_nothing_left(remote, job)
    _assert_secret_absent(remote, job, run.output)

    # tests/conftest.py opts every run out of the pre-job checks; this one needs it.
    unreachable = {
        **remote.env,
        "BENCHFLOW_REMOTE_DOCKER_HOST": "ssh://nobody@127.0.0.1:1",
        "BENCHFLOW_SKIP_PREFLIGHT": "0",
    }
    gone = jobs_root / "remote-docker-unreachable"
    h.clear_job(gone)
    run = h.bench(
        "eval", "run", "--tasks-dir", task, "--agent", "oracle",
        "--sandbox", "remote-docker", "--jobs-dir", gone.parent, "--job-name", gone.name,
        log=jobs_root / "remote-docker-unreachable.log", env=unreachable,
    )  # fmt: skip
    ledger.record("remote-docker unreachable host refused before a job",
                  surface="CLI", seconds=run.seconds)  # fmt: skip
    h.assert_exit(run, 1)
    assert "no job was created" in run.output
    assert "Remote Docker host unreachable: ssh://***@127.0.0.1:1" in run.output
    assert not gone.exists()


def test_remote_docker_allowlist_on_the_remote_container(
    remote: RemoteHost, tasks_root: Path, e2e_out: Path, ledger: h.Ledger, monkeypatch
):
    """The egress proxy and uid firewall a rollout arms before prompting, on a remote container."""
    from benchflow.sandbox.egress_denylist import (
        denylist_agent_env,
        egress_denylist_for,
        start_egress_denylist,
        stop_egress_denylist,
    )
    from benchflow.sandbox.lockdown import (
        enforce_agent_egress_firewall,
        setup_sandbox_user,
    )
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox
    from benchflow.task.config import SandboxConfig
    from benchflow.task.paths import RolloutPaths

    for key, value in remote.env.items():
        monkeypatch.setenv(key, value)
    environment = tasks_root / "remote-docker-allowlist" / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    (environment / "Dockerfile").write_text(ALLOWLIST_DOCKERFILE)
    paths = RolloutPaths(rollout_dir=e2e_out / "remote-docker-allowlist")
    paths.mkdir()
    config = SandboxConfig(
        network_mode="allowlist", allowed_hosts=["example.com"], cpus=1, memory_mb=1024
    )
    env = RemoteDockerSandbox(
        environment_dir=environment, environment_name="e2e-rd-allowlist",
        session_id="e2e-rd-allowlist__1", rollout_paths=paths, task_env_config=config,
    )  # fmt: skip
    probe = (
        "for u in https://example.com/ https://www.wikipedia.org/; do "
        "code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' \"$u\" || true); "
        'echo "proxy $u $code"; done; '
        "code=$(curl -s -o /dev/null -m 10 --noproxy '*' -w '%{http_code}' "
        'https://www.wikipedia.org/ || true); echo "direct $code"'
    )

    async def scenario() -> str:
        await env.start(force_build=True)
        try:
            await setup_sandbox_user(env, "agent", "/app")
            policy = egress_denylist_for(env.task_env_config)
            assert policy is not None
            await start_egress_denylist(env, "agent", policy)
            agent_env = denylist_agent_env({}, policy)
            await enforce_agent_egress_firewall(env, "agent", agent_env)
            result = await env.exec(probe, env=agent_env, user="agent", timeout_sec=90)
            await stop_egress_denylist(env, paths.rollout_dir)
            return result.stdout or ""
        finally:
            await env.stop(delete=True)

    started = time.monotonic()
    output = asyncio.run(scenario())
    ledger.record("remote-docker allowlist (provider API, no agent)", surface="Python",
                  seconds=time.monotonic() - started)  # fmt: skip
    assert "proxy https://example.com/ 200" in output, output
    assert "proxy https://www.wikipedia.org/ 200" not in output, output
    assert "direct 200" not in output, output
    blocked = h.read_jsonl(paths.rollout_dir / "trajectory" / "egress_denylist.jsonl")
    assert any(b["rule"] == "not-allowlisted" for b in blocked), blocked
    label = "label=com.docker.compose.project=e2e-rd-allowlist__1"
    assert _docker(remote, "ps", "-aq", "--filter", label).split() == []
    assert _docker(remote, "volume", "ls", "-q", "--filter", label).split() == []
