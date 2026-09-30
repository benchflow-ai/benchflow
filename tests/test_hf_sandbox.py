"""HF Sandbox backend: provider wiring, flavor/idle/label policy, exec/transfer, and the stdio bridge.

The bridge tests run the real in-sandbox bridge (``hf_bridge_server.py``) as a
local subprocess and drive it with the real ``HFBridgeProcess`` client over real
HTTP; only ``huggingface_hub.Sandbox`` is replaced by a local stand-in whose
``proxy_url_for`` points at 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.diagnostics import TransportClosedError
from benchflow.sandbox import hf_sandbox
from benchflow.sandbox.hf_bridge import HFBridgeProcess
from benchflow.sandbox.hf_sandbox import HFSandbox, idle_timeout_for, pick_flavor
from benchflow.sandbox.providers import OFF_BOX_MODEL_PROVIDERS, provider_extra
from benchflow.task.config import SandboxConfig

# --------------------------------------------------------------------------- local stand-in


class _LocalFiles:
    def __init__(self, root: Path | None) -> None:
        self.root = root

    def _p(self, path: str) -> Path:
        return Path(path) if self.root is None else self.root / path.lstrip("/")

    def write(self, path: str, data: str | bytes) -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode() if isinstance(data, str) else data)

    def read(self, path: str) -> bytes:
        return self._p(path).read_bytes()


class _LocalProc:
    def __init__(self, popen: subprocess.Popen) -> None:
        self.popen = popen

    def kill(self) -> bool:
        if self.popen.poll() is None:
            self.popen.kill()
            self.popen.wait(timeout=5)
            return True
        return False


class LocalSandbox:
    """Runs commands on this machine; enough of huggingface_hub.Sandbox for the backend."""

    id = "local-test"

    def __init__(self, root: Path | None = None) -> None:
        self.files = _LocalFiles(root)
        self.calls: list[dict] = []
        self.procs: list[_LocalProc] = []
        self.killed = False

    def run(self, cmd, *, shell=None, env=None, cwd=None, timeout=None, check=True, background=False, **_):
        self.calls.append({"cmd": cmd, "env": env, "cwd": cwd, "timeout": timeout, "background": background})
        import os

        full_env = {**os.environ, **(env or {})}
        if background:
            p = subprocess.Popen(cmd, env=full_env, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc = _LocalProc(p)
            self.procs.append(proc)
            return proc
        try:
            r = subprocess.run(cmd, env=full_env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return SimpleNamespace(exit_code=None, stdout="", stderr="", timed_out=True, signal=9)
        return SimpleNamespace(exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr, timed_out=False, signal=None)

    def proxy_url_for(self, port, path="/", scheme="http://"):
        return f"http://127.0.0.1:{port}{path}"

    @property
    def proxy_headers(self):
        return {"Authorization": "Bearer test"}

    def kill(self):
        self.killed = True
        for p in self.procs:
            p.kill()


def _config(**kw) -> SandboxConfig:
    return SandboxConfig(docker_image="python:3.12-slim", **kw)


def _sandbox(tmp_path: Path, **kw) -> HFSandbox:
    env_dir = tmp_path / "environment"
    env_dir.mkdir(exist_ok=True)
    return HFSandbox(
        environment_dir=env_dir,
        environment_name="demo",
        session_id="demo__abc",
        rollout_paths=None,
        task_env_config=_config(**kw),
        agent_timeout_sec=900,
        verifier_timeout_sec=900,
    )


# --------------------------------------------------------------------------- policy


def test_registered_as_off_box_provider_with_extra() -> None:
    assert "hf-sandbox" in OFF_BOX_MODEL_PROVIDERS
    assert provider_extra("hf-sandbox") == "sandbox-hf"


def test_flavor_is_smallest_cpu_flavor_that_fits(monkeypatch) -> None:
    monkeypatch.delenv("BENCHFLOW_HF_SANDBOX_FLAVOR", raising=False)
    assert pick_flavor(1, 2048) == "cpu-basic"
    assert pick_flavor(2, 16384) == "cpu-basic"
    assert pick_flavor(4, 8192) == "cpu-upgrade"
    assert pick_flavor(2, 20000) == "cpu-upgrade"
    assert pick_flavor(16, 65536) == "cpu-upgrade"  # warned, largest CPU flavor
    with pytest.raises(ValueError, match="GPU"):
        pick_flavor(1, 2048, gpus=1)
    monkeypatch.setenv("BENCHFLOW_HF_SANDBOX_FLAVOR", "a10g-small")
    assert pick_flavor(1, 2048, gpus=1) == "a10g-small"


def test_idle_timeout_covers_agent_verifier_and_build(monkeypatch) -> None:
    monkeypatch.delenv("BENCHFLOW_HF_SANDBOX_IDLE_TIMEOUT_SEC", raising=False)
    cfg = _config(build_timeout_sec=600)
    assert idle_timeout_for(cfg, 900, 900) == 900 + 900 + 600 + 600
    assert idle_timeout_for(_config(build_timeout_sec=10), 10, 10) == 1800  # floor
    assert idle_timeout_for(cfg, 12000, 900) == 12000 + 900 + 600 + 600  # long agent limits are covered
    monkeypatch.setenv("BENCHFLOW_HF_SANDBOX_IDLE_TIMEOUT_SEC", "5000")
    assert idle_timeout_for(cfg, 900, 900) == 5000


def test_labels_carry_managed_session_run_and_extras(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BENCHFLOW_HF_SANDBOX_LABELS", "posttrain=1,posttrain-run=base 0930")
    monkeypatch.setenv("BENCHFLOW_HF_SANDBOX_RUN", "baseline-hf-10010000")
    labels = _sandbox(tmp_path)._labels()
    assert labels == {
        "posttrain": "1",
        "posttrain-run": "base-0930",
        "benchflow-run": "baseline-hf-10010000",
        "benchflow-managed": "1",
        "benchflow-session": "demo__abc",
    }


def test_prebuilt_image_required(tmp_path) -> None:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    with pytest.raises(ValueError, match="prebuilt images only"):
        HFSandbox(
            environment_dir=env_dir,
            environment_name="demo",
            session_id="s",
            rollout_paths=None,
            task_env_config=SandboxConfig(),
        )


def test_setup_dispatches_hf_sandbox(monkeypatch, tmp_path) -> None:
    from benchflow.sandbox import setup

    monkeypatch.setattr(HFSandbox, "preflight", classmethod(lambda cls: None))
    monkeypatch.setattr(setup, "_validate_task_runtime_for_launch", lambda *a, **k: None)
    (tmp_path / "environment").mkdir()
    task = SimpleNamespace(
        config=SimpleNamespace(
            environment=_config(cpus=4, memory_mb=8192),
            agent=SimpleNamespace(timeout_sec=3600.0),
            verifier=SimpleNamespace(timeout_sec=600.0),
        ),
        paths=SimpleNamespace(environment_dir=tmp_path / "environment"),
    )
    monkeypatch.delenv("BENCHFLOW_HF_SANDBOX_FLAVOR", raising=False)
    monkeypatch.delenv("BENCHFLOW_HF_SANDBOX_IDLE_TIMEOUT_SEC", raising=False)
    sb = setup._create_sandbox_environment("hf-sandbox", task, tmp_path, "t__1", None)
    assert isinstance(sb, HFSandbox)
    assert sb._flavor == "cpu-upgrade"
    assert sb._idle_timeout == 3600 + 600 + 600 + 600


def test_acp_transport_selection_names_the_bridge() -> None:
    from benchflow.acp.selection import selected_acp_transport

    assert selected_acp_transport(agent="opencode", environment="hf-sandbox") == "hf-http-bridge"


# --------------------------------------------------------------------------- exec and transfer


def test_exec_argv_env_user_and_timeout(tmp_path) -> None:
    sb = _sandbox(tmp_path)
    local = LocalSandbox()
    sb._sandbox = local
    r = asyncio.run(sb.exec("echo $FOO; exit 3", env={"FOO": "bar"}))
    assert (r.return_code, r.stdout.strip()) == (3, "bar")
    call = local.calls[-1]
    assert call["cmd"][:2] == ["/bin/bash", "-c"] and "bar" not in call["cmd"][2]  # env not on the command line
    asyncio.run(sb.exec("id", user="agent"))
    assert local.calls[-1]["cmd"][2].startswith("su agent -s /bin/bash -c ")
    with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
        asyncio.run(sb.exec("sleep 5", timeout_sec=1))
    with pytest.raises(ValueError, match="single-container"):
        asyncio.run(sb.exec("true", service="target"))


def test_upload_and_download_dir_round_trip(tmp_path) -> None:
    sb = _sandbox(tmp_path)
    sb._sandbox = LocalSandbox()
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "sub" / "b.bin").write_bytes(b"\x00\x01")
    (src / "link").symlink_to("/etc/hosts")  # never shipped (#411)
    remote = tmp_path / "remote"
    asyncio.run(sb.upload_dir(src, str(remote)))
    assert (remote / "a.txt").read_text() == "alpha"
    assert not (remote / "link").exists()
    back = tmp_path / "back"
    asyncio.run(sb.download_dir(str(remote), back))
    assert (back / "sub" / "b.bin").read_bytes() == b"\x00\x01"
    asyncio.run(sb.download_dir(str(tmp_path / "missing"), tmp_path / "none"))  # absent dir: no error
    asyncio.run(sb.upload_file(src / "a.txt", str(tmp_path / "one" / "a.txt")))
    asyncio.run(sb.download_file(str(tmp_path / "one" / "a.txt"), tmp_path / "two" / "a.txt"))
    assert (tmp_path / "two" / "a.txt").read_text() == "alpha"


def test_stop_kills_and_forgets_live_sandbox(tmp_path) -> None:
    sb = _sandbox(tmp_path)
    local = LocalSandbox()
    sb._sandbox = local
    hf_sandbox._LIVE[local.id] = local
    asyncio.run(sb.stop(delete=True))
    assert local.killed and local.id not in hf_sandbox._LIVE and sb.sandbox_id is None


def test_kill_all_live_kills_everything_registered() -> None:
    a, b = LocalSandbox(), LocalSandbox()
    b.id = "other"
    hf_sandbox._LIVE.update({a.id: a, b.id: b})
    assert hf_sandbox.kill_all_live() == 2
    assert a.killed and b.killed and not hf_sandbox._LIVE


def test_sweep_only_touches_benchflow_sandboxes_of_the_run(monkeypatch) -> None:
    def job(i, stage="RUNNING", **labels):
        return SimpleNamespace(
            id=i, labels=labels, status=SimpleNamespace(stage=stage), created_at=None, owner=SimpleNamespace(name="ns")
        )

    jobs = [
        job("mine", **{"hf-sandbox": "1", "benchflow-managed": "1", "benchflow-run": "r1"}),
        job("other-run", **{"hf-sandbox": "1", "benchflow-managed": "1", "benchflow-run": "r2"}),
        job("not-benchflow", **{"hf-sandbox": "1", "benchflow-run": "r1"}),
        job("training-job", posttrain="1"),
        job("done", stage="CANCELED", **{"hf-sandbox": "1", "benchflow-managed": "1", "benchflow-run": "r1"}),
    ]
    canceled = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_jobs(self, namespace=None):
            return jobs

        def cancel_job(self, job_id, namespace=None):
            canceled.append(job_id)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    counts = hf_sandbox.sweep(namespace="ns", run="r1")
    assert canceled == ["mine"] and counts == {"found": 1, "canceled": 1, "failed": 0}


# --------------------------------------------------------------------------- stdio bridge (real HTTP)

_ECHO_AGENT = (
    "import sys, json\n"
    "print('agent banner (not protocol)', flush=True)\n"
    "for line in sys.stdin:\n"
    "    msg = json.loads(line)\n"
    "    if msg.get('method') == 'exit':\n"
    "        sys.stderr.write('bye\\n'); sys.exit(7)\n"
    "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': msg.get('params')}), flush=True)\n"
)


def _bridge(tmp_path: Path) -> tuple[HFBridgeProcess, LocalSandbox]:
    local = LocalSandbox()
    proc = HFBridgeProcess(local)
    proc.bridge_dir = str(tmp_path / "bridge")
    proc._stderr_path = f"{proc.bridge_dir}/{proc._id}.stderr"
    proc._config_path = f"{proc.bridge_dir}/{proc._id}.json"
    (tmp_path / "agent.py").write_text(_ECHO_AGENT)
    return proc, local


async def _echo_session(tmp_path: Path) -> dict:
    proc, local = _bridge(tmp_path)
    out: dict = {}
    await proc.start(f"{sys.executable} -u agent.py", env={"SECRET_KEY": "s3"}, cwd=str(tmp_path))
    try:
        bg = [c for c in local.calls if c["background"]][0]
        out["secret_on_argv"] = any("s3" in part for part in bg["cmd"])
        out["bg_env"] = bg["env"]
        out["banner"] = (await proc.readline()).decode()
        t = time.time()
        for i in range(20):
            await proc.writeline(json.dumps({"jsonrpc": "2.0", "id": i, "method": "ping", "params": i}))
            reply = json.loads(await proc.readline())
            assert reply["id"] == i and reply["result"] == i
        out["rtt_ms"] = (time.time() - t) / 20 * 1000
        big = "x" * 200_000
        await proc.writeline(json.dumps({"jsonrpc": "2.0", "id": "big", "params": big}))
        out["big_ok"] = json.loads(await proc.readline())["result"] == big
        await proc.writeline(json.dumps({"jsonrpc": "2.0", "id": "x", "method": "exit"}))
        with pytest.raises(TransportClosedError) as err:
            await proc.readline()
        out["closed"] = err.value
    finally:
        await proc.close()
    return out


def test_bridge_round_trips_lines_and_reports_exit(tmp_path) -> None:
    out = asyncio.run(_echo_session(tmp_path))
    assert out["banner"].strip() == "agent banner (not protocol)"
    assert out["big_ok"]
    assert not out["secret_on_argv"] and out["bg_env"] == {"SECRET_KEY": "s3"}
    diag = out["closed"].diagnostic
    assert diag.process_exit_code == 7 and diag.transport_diagnosis == "process_exited"
    assert "bye" in (diag.stderr_snippet or "")


def test_bridge_resumes_from_line_and_ignores_duplicate_writes(tmp_path) -> None:
    import httpx

    async def run() -> tuple[list[str], list[str]]:
        proc, _ = _bridge(tmp_path)
        await proc.start(f"{sys.executable} -u agent.py", cwd=str(tmp_path))
        try:
            await proc.readline()  # banner
            for i in range(3):
                await proc.writeline(json.dumps({"id": i, "params": i}))
                await proc.readline()
            async with httpx.AsyncClient() as c:
                # a retried POST with an old seq must not reach the agent twice
                r = await c.post(proc._base + "in", params={"seq": 3}, content=b'{"id": 99}\n')
                assert r.json() == {"duplicate": True}
                # a reconnect from line 2 replays lines 2.. (what the reader does after a drop)
                seen = []
                async with c.stream("GET", proc._base + "out", params={"from": 2}) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            seen.append(line)
                        if len(seen) == 2:
                            break
            health = (await httpx.AsyncClient().get(proc._base + "health")).json()
            return seen, [str(health["lines"])]
        finally:
            await proc.close()

    seen, lines = asyncio.run(run())
    assert [json.loads(s)["id"] for s in seen] == [1, 2]
    assert lines == ["4"]


def test_bridge_close_stops_child(tmp_path) -> None:
    async def run() -> int | None:
        proc, local = _bridge(tmp_path)
        await proc.start("sleep 300", cwd=str(tmp_path))
        await proc.close()
        bridge = local.procs[0].popen
        for _ in range(50):
            if bridge.poll() is not None:
                break
            await asyncio.sleep(0.1)
        return bridge.poll()

    assert asyncio.run(run()) is not None
