"""Guards PY8 PERF-F against 5653de15: private attachment and no replay."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest


def cli(args, env, timeout=60):
    return subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _parent_env(**extra: str) -> dict[str, str]:
    """This process's environment as a proxy parent gets it: python-dotenv off."""
    return dict(os.environ, PYTHON_DOTENV_DISABLED="1", **extra)


def test_daemon_verb_is_available():
    """Behavioral fails-before: 5653de15 rejects the daemon verb (exit 2)."""
    result = cli(["daemon", "--help"], os.environ.copy())
    assert result.returncode == 0, result.stderr
    assert "start" in result.stdout and "stop" in result.stdout


def test_socket_ownership_modes_and_peer(tmp_path):
    """Guards PERF-F's private transport, including a writable parent refusal."""
    from benchflow._daemon_wire import connect, listen, peer_uid

    root = tmp_path / "private"
    path = root / "runtime.sock"
    with listen(path) as server:
        assert root.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
        with connect(path) as client, server.accept()[0] as peer:
            assert peer_uid(peer) == os.getuid()
            assert peer_uid(client) == os.getuid()
        path.chmod(0o666)
        with pytest.raises(OSError, match="0600"):
            connect(path)
        path.chmod(0o600)
        root.chmod(0o755)
        with pytest.raises(OSError, match="0700"):
            connect(path)


@pytest.mark.parametrize("accepted", [False, True])
def test_disconnect_never_replays_an_accepted_call(tmp_path, monkeypatch, accepted):
    """Guards PERF-F: retry is safe before handoff, never after handoff."""
    from benchflow._daemon import try_attach
    from benchflow._daemon_wire import listen, receive, receive_fds, send

    monkeypatch.setenv("BENCHFLOW_DAEMON", "auto")
    monkeypatch.setenv("BENCHFLOW_DAEMON_DIR", str(tmp_path / "private"))
    monkeypatch.delenv("_BENCHFLOW_DAEMON_WORKER", raising=False)
    listener = listen(tmp_path / "private" / "runtime.sock")
    errors = []

    def peer():
        try:
            with listener.accept()[0] as conn:
                receive(conn)
                if accepted:
                    send(conn, {"ready": True})
                    for fd in receive_fds(conn, 3):
                        os.close(fd)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=peer)
    thread.start()
    try:
        assert try_attach(["eval", "run", "--help"]) == (1 if accepted else None)
    finally:
        thread.join(10)
        listener.close()
    assert not thread.is_alive() and not errors


def test_daemon_restart_and_cli_output_equivalence(tmp_path):
    """Guards PERF-F: cold/warm CLI bytes and restart without a lost request."""
    from benchflow._daemon_wire import connect, receive, send

    env = dict(os.environ, BENCHFLOW_DAEMON_DIR=str(tmp_path / "runtime"))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    path = tmp_path / "runtime" / "runtime.sock"
    cold = cli(["eval", "run", "--tasks-dir", str(tmp_path / "missing")], env)
    for _ in range(2):
        started = cli(["daemon", "start"], env)
        assert started.returncode == 0, started.stderr
        try:
            warm = cli(
                ["eval", "run", "--tasks-dir", str(tmp_path / "missing")],
                dict(env, BENCHFLOW_DAEMON="auto"),
            )
            assert warm.returncode == cold.returncode
            assert warm.stdout == cold.stdout
            assert warm.stderr == cold.stderr
            with connect(path) as conn:
                send(conn, {"action": "status"})
                state = receive(conn)
                assert state["pid"] != os.getpid()
                assert state["accepted_calls"] >= 1
        finally:
            # A completed worker can be waiting for its parent's next reap.
            for _ in range(100):
                stopped = cli(["daemon", "stop"], env)
                if stopped.returncode == 0:
                    break
                time.sleep(0.02)
            assert stopped.returncode == 0, stopped.stderr
            for _ in range(100):
                if not path.exists():
                    break
                time.sleep(0.02)
            assert not path.exists()
    fallback = cli(
        ["eval", "run", "--tasks-dir", str(tmp_path / "missing")],
        dict(env, BENCHFLOW_DAEMON="auto"),
    )
    assert (fallback.returncode, fallback.stdout, fallback.stderr) == (
        cold.returncode,
        cold.stdout,
        cold.stderr,
    )


def test_proxy_preload_forbids_trial_settings():
    """Guards PERF-F: transitive LiteLLM imports cannot capture trial values.

    Trial settings are present here, as a scan would see them in a cold proxy:
    a whole-environment scan counts only through its declared inputs.
    """
    script = (
        "from benchflow._proxy_zygote import preload; import json\n"
        "reads, scans = preload(); print(json.dumps([sorted(reads), scans]))"
    )
    env = _parent_env(
        LITELLM_MASTER_KEY="sk-trial", BENCHFLOW_LITELLM_LOG_PATH="/tmp/x"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    reads, scans = json.loads(result.stdout.splitlines()[-1])
    assert "DOCS_URL" in reads and "LITELLM_MODE" in reads
    assert "LITELLM_MASTER_KEY" not in reads
    assert "BENCHFLOW_LITELLM_LOG_PATH" not in reads
    assert "LITELLM_OTEL_V2" in scans


def test_proxy_preload_refuses_an_unaudited_environment_scan():
    """Guards PERF-F: a scan whose inputs nobody declared refuses the parent
    (trials then start their proxies cold); a per-name audit cannot see it."""
    script = "import benchflow._proxy_zygote as z\nz.KNOWN_SCANS.clear()\nz.preload()\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=_parent_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert (
        "Unaudited environment scan during proxy preload: "
        "litellm.integrations.otel.model.config.is_otel_v2_enabled"
    ) in result.stderr


def test_proxy_child_recomputes_scanned_settings(monkeypatch):
    """Guards PERF-F: a proxy child computes a scanned setting from its own
    environment, as a cold start does, not the parent's cached answer."""
    from litellm.integrations.otel.model.config import is_otel_v2_enabled

    from benchflow._proxy_zygote import _reset_scans

    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    is_otel_v2_enabled.cache_clear()
    try:
        assert is_otel_v2_enabled() is False
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        assert is_otel_v2_enabled() is False  # cached, as in the parent
        _reset_scans()
        assert is_otel_v2_enabled() is True
    finally:
        monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
        is_otel_v2_enabled.cache_clear()


async def test_forked_proxies_have_separate_keys_ports_logs_and_usage(monkeypatch):
    """Guards PERF-F/D931: real requests must stay in their owning proxy."""
    import httpx

    from benchflow._proxy_zygote import ProxyProcess
    from benchflow.providers import litellm_runtime as runtime
    from benchflow.providers.litellm_config import resolve_litellm_route
    from tests.fixtures.mock_token_logprobs_server import start_server

    with tempfile.TemporaryDirectory(prefix="bf-f-") as root:
        monkeypatch.setenv("BENCHFLOW_DAEMON_DIR", root + "/r")
        env = dict(os.environ)
        started = cli(["daemon", "start"], env)
        assert started.returncode == 0, started.stderr
        monkeypatch.setenv("_BENCHFLOW_DAEMON_WORKER", "1")
        mock = start_server(flavor="vllm")
        agent_env = {
            "OPENAI_API_KEY": "sk-mock",
            "BENCHFLOW_PROVIDER_BASE_URL": mock.base_url + "/v1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
        route = resolve_litellm_route("openai/test-model", agent_env)
        proxies = []
        try:
            for index in range(2):
                try:
                    proxy = await runtime._start_host_litellm(
                        route=route,
                        master_key=f"sk-master-{index}",
                        agent_env=agent_env,
                        environment="local",
                        session_id=f"trial-{index}",
                        agent_name="perf-f-test",
                    )
                except Exception as exc:
                    raise AssertionError(_runtime_logs(root + "/r")) from exc
                proxies.append(proxy)
            assert all(isinstance(proxy.process, ProxyProcess) for proxy in proxies), (
                _runtime_logs(root + "/r")
            )
            assert proxies[0].process.pid != proxies[1].process.pid
            assert proxies[0].base_url != proxies[1].base_url
            assert proxies[0].runtime_dir != proxies[1].runtime_dir
            async with httpx.AsyncClient(timeout=30) as client:
                for index, proxy in enumerate(proxies):
                    body = {
                        "model": route.model_alias,
                        "messages": [{"role": "user", "content": f"trial-{index}"}],
                    }
                    refused = await client.post(
                        proxy.base_url + "/v1/chat/completions",
                        headers={"Authorization": f"Bearer sk-master-{1 - index}"},
                        json=body,
                    )
                    assert 400 <= refused.status_code < 500
                    assert len(mock.requests) == index
                    response = await client.post(
                        proxy.base_url + "/v1/chat/completions",
                        headers={"Authorization": f"Bearer sk-master-{index}"},
                        json=body,
                    )
                    assert response.status_code == 200, response.text
            # Another process (another trial's CLI) can neither stop nor watch it.
            target = proxies[0].process
            other = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; from benchflow._proxy_zygote import ProxyProcess; "
                        f"p = ProxyProcess(Path({str(target.path)!r}), {target.pid}); p.kill(); print(p.poll())"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert other.stdout.strip() == "1", other.stderr
            assert target.poll() is None
            for index, proxy in enumerate(proxies):
                await proxy.stop()
                assert proxy.trajectory is not None
                text = proxy.trajectory.model_dump_json()
                assert f"trial-{index}" in text
                assert f"trial-{1 - index}" not in text
        finally:
            for proxy in proxies:
                if await proxy.is_running():
                    await proxy.stop()
            mock.shutdown()
            mock.server_close()
            monkeypatch.delenv("_BENCHFLOW_DAEMON_WORKER")
            stopped = cli(["daemon", "stop"], env)
            assert stopped.returncode == 0, stopped.stderr


def test_forked_workers_draw_their_own_guard_marker_secret():
    """Behavioral fails-before on 5653de15: every fork of one pre-imported
    parent shared the pytest guard's marker secret, which a fresh CLI draws."""
    script = (
        "import os\n"
        "from benchflow.sandbox import lockdown\n"
        "read, write = os.pipe()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.write(write, lockdown.pytest_plugin_guard_key('g').encode()); os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
        "child = os.read(read, 200).decode()\n"
        "print(child != lockdown.pytest_plugin_guard_key('g'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_terminal_calls_keep_the_cold_path(monkeypatch):
    """Guards PERF-F: job control and prompts stay with a terminal's process."""
    from benchflow import _daemon

    monkeypatch.setenv("BENCHFLOW_DAEMON", "auto")
    monkeypatch.delenv("_BENCHFLOW_DAEMON_WORKER", raising=False)
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    assert _daemon.attachable(["eval", "run"])
    assert not _daemon.attachable(["tasks", "check"])
    monkeypatch.setattr(os, "isatty", lambda fd: fd == 1)
    assert not _daemon.attachable(["eval", "run"])


def test_interpreter_start_settings_refuse_admission():
    """Guards PERF-F: PYTHONPATH, locale and time zone act at interpreter
    start, which a fork inherits from its parent instead of the caller."""
    from benchflow._daemon import _admission_mismatch, _start_settings

    env = {"PATH": "/usr/bin", "LANG": "C.UTF-8"}
    value = {"identity": "i", "interpreter": "x", "env": dict(env)}
    start = _start_settings(env)
    assert _admission_mismatch(value, "i", "x", start, {}) is None
    for key, new in (
        ("PYTHONPATH", "/tmp/elsewhere"),
        ("LANG", "en_US.UTF-8"),
        ("TZ", "UTC"),
    ):
        changed = dict(value, env=dict(env, **{key: new}))
        assert (
            _admission_mismatch(changed, "i", "x", start, {})
            == "interpreter start settings"
        )
    assert _admission_mismatch(value, "i", "y", start, {}) == "interpreter flags"
    assert _admission_mismatch(value, "other", "x", start, {}) == "installation"
    read = _admission_mismatch(value, "i", "x", start, {"NO_COLOR": "1"})
    assert read == "import-time setting NO_COLOR"


def test_attached_cli_forwards_signals_and_ends_as_its_worker(tmp_path):
    """Guards PERF-F: Ctrl-C or a harness's SIGTERM reaches the worker, which is
    outside the caller's process group, and the caller ends by that signal."""
    from benchflow._daemon_wire import listen, receive, receive_fds, send

    root = tmp_path / "private"
    listener = listen(root / "runtime.sock")
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    client_script = (
        "import sys\n"
        "from benchflow._daemon import try_attach\n"
        "sys.exit(try_attach(['eval', 'run', '--help']))\n"
    )
    env = dict(os.environ, BENCHFLOW_DAEMON="auto", BENCHFLOW_DAEMON_DIR=str(root))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    client = subprocess.Popen(
        [sys.executable, "-c", client_script],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with listener.accept()[0] as conn:
            receive(conn)
            send(conn, {"ready": True})
            for fd in receive_fds(conn, 3):
                os.close(fd)
            send(conn, {"pid": worker.pid})
            time.sleep(1)
            client.send_signal(15)
            assert worker.wait(20) == -15
        assert client.wait(20) == -15
    finally:
        for process in (worker, client):
            if process.poll() is None:
                process.kill()
        listener.close()


def test_worker_ends_when_its_caller_is_gone():
    """Guards PERF-F: a SIGKILLed caller cannot leave its job running unseen."""
    from benchflow._daemon import _watch_caller

    ours, theirs = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        ours.close()
        _watch_caller(theirs)
        time.sleep(30)
        os._exit(0)
    theirs.close()
    time.sleep(0.2)
    ours.close()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == -15


def test_proxy_parent_refuses_changed_import_settings_before_fork():
    """Guards PERF-F: a launch whose environment differs from the values the
    parent's imports read is refused, and the trial starts its proxy cold."""
    from benchflow._proxy_zygote import _import_time_mismatch, _trial_setting

    reads = {"DOCS_URL": None, "LITELLM_MODE": "PRODUCTION"}
    assert _import_time_mismatch(reads, {"LITELLM_MODE": "PRODUCTION"}) == []
    scanned: dict[str, list[list[str]]] = {"LITELLM_OTEL_V2": []}
    same = {"LITELLM_MODE": "PRODUCTION"}
    assert _import_time_mismatch(reads, same, scanned) == []
    changed = dict(same, litellm_otel_v2="1")
    assert _import_time_mismatch(reads, changed, scanned) == ["LITELLM_OTEL_V2"]
    assert _import_time_mismatch(reads, {"LITELLM_MODE": "DEV"}) == ["LITELLM_MODE"]
    assert _import_time_mismatch(
        reads, {"LITELLM_MODE": "PRODUCTION", "DOCS_URL": "/x"}
    ) == ["DOCS_URL"]
    assert _trial_setting("LITELLM_MASTER_KEY") and _trial_setting("BENCHFLOW_ANY")
    assert not _trial_setting("OPENAI_API_KEY")


def test_daemon_proxy_falls_back_only_before_launch(monkeypatch, tmp_path):
    """Guards PERF-F: a refusal starts the proxy cold; a lost answer after the
    launch request fails instead of starting a second proxy on the port."""
    from benchflow import _daemon
    from benchflow._daemon_wire import listen, receive, send
    from benchflow.providers import litellm_runtime as runtime

    monkeypatch.setenv("_BENCHFLOW_DAEMON_WORKER", "1")
    path = tmp_path / "private" / "z.sock"
    listener = listen(path)
    monkeypatch.setattr(
        _daemon, "request", lambda value, timeout=5: {"path": str(path)}
    )
    answers = [{"refused": "test"}, None]

    def peer():
        for answer in answers:
            with listener.accept()[0] as conn:
                receive(conn)
                if answer is not None:
                    send(conn, answer)

    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    try:
        kwargs = dict(env={}, runtime_dir=tmp_path, args=[], executable=sys.executable)
        assert runtime._daemon_proxy(**kwargs) is None
        with pytest.raises(RuntimeError, match="did not answer"):
            runtime._daemon_proxy(**kwargs)
    finally:
        thread.join(10)
        listener.close()


def test_console_script_from_another_folder_attaches(tmp_path):
    """Guards PERF-F: the `bench` script, run from any folder, attaches to a
    runtime started elsewhere (sys.path[0] differs between them)."""
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no installed bench console script")
    env = dict(os.environ, BENCHFLOW_DAEMON_DIR=str(tmp_path / "runtime"))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    started = cli(["daemon", "start"], env)
    assert started.returncode == 0, started.stderr
    try:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        call = subprocess.run(
            [str(bench), "eval", "run", "--tasks-dir", str(tmp_path / "missing")],
            env=dict(env, BENCHFLOW_DAEMON="auto"),
            cwd=elsewhere,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert call.returncode != 0
        status = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from benchflow._daemon import request; print(json.dumps(request({'action': 'status'})))",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        accepted = json.loads(status.stdout)["accepted_calls"]
        assert accepted == 1, (
            _refusal(dict(env, BENCHFLOW_DAEMON="auto"), elsewhere)
            + call.stderr[-1500:]
        )
    finally:
        for _ in range(100):
            if cli(["daemon", "stop"], env).returncode == 0:
                break
            time.sleep(0.05)


_WHY = (
    "import os, sys\n"
    "from benchflow import _daemon\n"
    "from benchflow._daemon_wire import connect, receive, send\n"
    "value = {'action': 'run', 'identity': _daemon.identity(), 'interpreter': _daemon._interpreter(),\n"
    "         'stdio': _daemon._stdio(), 'env': dict(os.environ), 'cwd': os.getcwd(), 'argv0': sys.argv[0],\n"
    "         'argv': ['eval', 'run', '--help'], 'umask': _daemon._umask()}\n"
    "with connect(_daemon.socket_path()) as conn:\n"
    "    send(conn, value); print(receive(conn))\n"
)


def _refusal(env, cwd):
    """The runtime's own answer to a call like this one (for failure messages)."""
    found = subprocess.run(
        [sys.executable, "-c", _WHY],
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return found.stdout + found.stderr[-800:]


def test_identity_ignores_bytecode_caches_and_the_script_folder(tmp_path, monkeypatch):
    """Guards PERF-F: a fresh runtime's own imports create __pycache__ folders,
    and every caller's sys.path[0] differs; neither is the installation."""
    from benchflow import _daemon

    site = tmp_path / "prefix" / "site-packages"
    site.mkdir(parents=True)
    (site / "pkg-1.0.dist-info").mkdir()
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "prefix"))
    monkeypatch.setattr(sys, "path", [str(tmp_path / "prefix" / "bin"), str(site)])
    before = _daemon.identity()
    (site / "__pycache__").mkdir()
    monkeypatch.setattr(sys, "path", [str(tmp_path / "elsewhere"), str(site)])
    assert _daemon.identity() == before
    (site / "pkg-2.0.dist-info").mkdir()
    assert _daemon.identity() != before


def test_proxy_child_replays_the_imports_environment_writes(monkeypatch):
    """Guards PERF-F: LiteLLM's import sets TIKTOKEN_CACHE_DIR over whatever the
    caller passed; a forked proxy child, whose import ran in the parent before
    its trial environment arrived, must end with the same value a cold one has."""
    script = (
        "import json, os\n"
        "from benchflow import _proxy_zygote as z\n"
        "z.preload()\n"
        "print(json.dumps(z.IMPORT_WRITES))\n"
    )
    env = _parent_env(TIKTOKEN_CACHE_DIR="/tmp/perf-f-callers-value")
    env.pop("CUSTOM_TIKTOKEN_CACHE_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    writes = json.loads(result.stdout.splitlines()[-1])
    assert writes["TIKTOKEN_CACHE_DIR"] != "/tmp/perf-f-callers-value"
    assert writes["TIKTOKEN_CACHE_DIR"].endswith("tokenizers")

    from benchflow import _proxy_zygote

    monkeypatch.setattr(
        _proxy_zygote, "IMPORT_WRITES", {"PERF_F_SET": "import", "PERF_F_GONE": None}
    )
    monkeypatch.setenv("PERF_F_SET", "caller")
    monkeypatch.setenv("PERF_F_GONE", "caller")
    _proxy_zygote._replay_import_writes()
    assert os.environ["PERF_F_SET"] == "import"
    assert "PERF_F_GONE" not in os.environ


def test_proxy_parent_key_ignores_what_litellms_import_sets():
    """Guards PERF-F: a CLI process gains TIKTOKEN_CACHE_DIR when its own
    LiteLLM import runs; that must not make its next trial need another
    proxy parent (two at most: the next would start cold)."""
    from benchflow._proxy_zygote import parent_setting

    assert not parent_setting("TIKTOKEN_CACHE_DIR")
    assert not parent_setting("LITELLM_MASTER_KEY")
    assert parent_setting("CUSTOM_TIKTOKEN_CACHE_DIR")
    assert parent_setting("OPENAI_API_KEY")


def _runtime_logs(folder):
    """The runtime's and its proxy parents' logs (for failure messages)."""
    parts = []
    for path in sorted(Path(folder).glob("*.log")):
        parts.append(f"== {path.name}\n" + path.read_text(errors="replace")[-3000:])
    return "\n".join(parts) or "(no runtime logs)"


def test_proxy_parent_watches_owners_without_pidfd():
    """Guards PERF-F: some interpreters (the gate image's) lack os.pidfd_open;
    an owner is known by pid and start time, so a reused pid is not it."""
    from benchflow._proxy_zygote import _start_time

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        started = _start_time(child.pid)
        assert isinstance(started, int)
        assert _start_time(child.pid) == started
        assert _start_time(os.getpid()) != started or os.getpid() == child.pid
    finally:
        child.kill()
        child.wait()
    assert _start_time(child.pid) is None


def _fake_runtime_client(root, worker_pid, *, preexec=None, after=None):
    """A fake runtime that accepts one attached call and hands it worker_pid;
    returns (client process, the accepted connection's closer)."""
    from benchflow._daemon_wire import listen, receive, receive_fds, send

    listener = listen(root / "runtime.sock")
    client_script = (
        "import sys\n"
        "from benchflow._daemon import try_attach\n"
        "sys.exit(try_attach(['eval', 'run', '--help']))\n"
    )
    env = dict(os.environ, BENCHFLOW_DAEMON="auto", BENCHFLOW_DAEMON_DIR=str(root))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    client = subprocess.Popen(
        [sys.executable, "-c", client_script],
        env=env,
        preexec_fn=preexec,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    conn = listener.accept()[0]
    request_value = receive(conn)
    send(conn, {"ready": True})
    for fd in receive_fds(conn, 3):
        os.close(fd)
    send(conn, {"pid": worker_pid})
    return client, conn, listener, request_value


def _ignore_hup_int():
    import signal as _signal

    _signal.signal(_signal.SIGHUP, _signal.SIG_IGN)
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)


def test_a_caller_ignoring_hup_and_int_survives_them_as_a_cold_cli_does(tmp_path):
    """Guards PERF-F-B-1: nohup and background jobs in scripts start the CLI
    with SIGHUP/SIGINT ignored; an attached call must not pass them on."""
    import signal as _signal

    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    # A worker as the runtime forks it before applying the caller's settings:
    # default dispositions, so only forwarding can stop it.
    worker = subprocess.Popen(sleeper)
    cold = subprocess.Popen(sleeper, preexec_fn=_ignore_hup_int, start_new_session=True)
    client, conn, listener, value = _fake_runtime_client(
        tmp_path / "rt", worker.pid, preexec=_ignore_hup_int
    )
    try:
        time.sleep(1)
        for signum in (_signal.SIGHUP, _signal.SIGINT):
            os.killpg(client.pid, signum)
            os.killpg(cold.pid, signum)
        time.sleep(1)
        assert cold.poll() is None  # the cold stand-in ignores both
        assert client.poll() is None and worker.poll() is None
        # What the worker is given, so it ignores them too.
        assert {int(_signal.SIGHUP), int(_signal.SIGINT)} <= set(
            value["signals"]["ignored"]
        )
        from benchflow._daemon_wire import send

        send(conn, {"exit": 0})
        assert client.wait(20) == 0
    finally:
        conn.close()
        listener.close()
        for process in (worker, cold, client):
            if process.poll() is None:
                process.kill()
            process.wait()


def test_a_worker_takes_its_callers_ignored_signals_and_mask():
    """Guards PERF-F-B-1: the worker, a fork of the runtime, gets the caller's
    dispositions (D961: SIGINT inherited as ignored stays ignored) and mask."""
    import signal as _signal

    from benchflow import _daemon

    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read)
            value = {
                "env": dict(os.environ),
                "cwd": os.getcwd(),
                "umask": 0o22,
                "argv0": "bench",
                "argv": [],
                "signals": {
                    "ignored": [int(_signal.SIGINT), int(_signal.SIGHUP)],
                    "mask": [int(_signal.SIGUSR1)],
                },
            }
            _daemon._enter_child(value)
            state = [
                _signal.getsignal(_signal.SIGINT) == _signal.SIG_IGN,
                _signal.getsignal(_signal.SIGHUP) == _signal.SIG_IGN,
                _signal.getsignal(_signal.SIGTERM) == _signal.SIG_DFL,
                _signal.SIGUSR1 in _signal.pthread_sigmask(_signal.SIG_BLOCK, []),
            ]
            os.write(write, json.dumps(state).encode())
        finally:
            os._exit(0)
    os.close(write)
    os.waitpid(pid, 0)
    assert json.loads(os.read(read, 200)) == [True, True, True, True]
    base = {
        "identity": "i",
        "interpreter": "x",
        "stdio": [],
        "env": {},
        "cwd": "/",
        "argv0": "b",
        "argv": [],
        "umask": 18,
    }
    one = dict(base, signals={"ignored": [], "mask": []})
    two = dict(base, signals={"ignored": [int(_signal.SIGHUP)], "mask": []})
    assert _daemon.call_key(one) != _daemon.call_key(two)


def test_a_different_rlimit_keeps_the_call_cold(tmp_path):
    """Guards PERF-F-B-4: a worker would keep the runtime's limits, groups,
    affinity, cgroup and priority; a call whose own differ runs cold."""
    import resource

    env = dict(os.environ, BENCHFLOW_DAEMON_DIR=str(tmp_path / "runtime"))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    assert cli(["daemon", "start"], env).returncode == 0
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    def lower():
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 256) - 1, hard))

    try:
        call = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchflow.cli.main",
                "eval",
                "run",
                "--tasks-dir",
                str(tmp_path / "missing"),
            ],
            env=dict(env, BENCHFLOW_DAEMON="auto"),
            preexec_fn=lower,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert call.returncode != 0
        status = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from benchflow._daemon import request; print(json.dumps(request({'action': 'status'})))",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert json.loads(status.stdout)["accepted_calls"] == 0
        assert (
            "process attributes: rlimits"
            in (tmp_path / "runtime" / "daemon.log").read_text()
        )
    finally:
        for _ in range(100):
            if cli(["daemon", "stop"], env).returncode == 0:
                break
            time.sleep(0.05)


async def test_forked_proxy_outlives_the_runtime_until_its_owner_goes(monkeypatch):
    """Guards PERF-F-B-2: SIGTERM, then SIGKILL, to the runtime while a call's
    forked proxy serves: the proxy keeps answering until its owner exits."""
    import httpx

    from benchflow._proxy_zygote import ProxyProcess
    from benchflow.providers import litellm_runtime as runtime
    from benchflow.providers.litellm_config import resolve_litellm_route
    from tests.fixtures.mock_token_logprobs_server import start_server

    with tempfile.TemporaryDirectory(prefix="bf-f-") as root:
        monkeypatch.setenv("BENCHFLOW_DAEMON_DIR", root + "/r")
        env = dict(os.environ)
        assert cli(["daemon", "start"], env).returncode == 0
        status = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from benchflow._daemon import request; print(json.dumps(request({'action': 'status'})))",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        runtime_pid = json.loads(status.stdout)["pid"]
        monkeypatch.setenv("_BENCHFLOW_DAEMON_WORKER", "1")
        mock = start_server(flavor="vllm")
        agent_env = {
            "OPENAI_API_KEY": "sk-mock",
            "BENCHFLOW_PROVIDER_BASE_URL": mock.base_url + "/v1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
        route = resolve_litellm_route("openai/test-model", agent_env)
        proxy = None
        try:
            proxy = await runtime._start_host_litellm(
                route=route,
                master_key="sk-master-0",
                agent_env=agent_env,
                environment="local",
                session_id="trial-0",
                agent_name="perf-f-test",
            )
            assert isinstance(proxy.process, ProxyProcess), _runtime_logs(root + "/r")
            body = {
                "model": route.model_alias,
                "messages": [{"role": "user", "content": "hi"}],
            }
            headers = {"Authorization": "Bearer sk-master-0"}
            async with httpx.AsyncClient(timeout=30) as client:
                assert (
                    await client.post(
                        proxy.base_url + "/v1/chat/completions",
                        headers=headers,
                        json=body,
                    )
                ).status_code == 200
                os.kill(runtime_pid, 15)
                for _ in range(200):
                    if not os.path.exists(root + "/r/runtime.sock"):
                        break
                    time.sleep(0.05)
                time.sleep(1)
                with contextlib.suppress(ProcessLookupError):
                    os.kill(runtime_pid, 9)
                time.sleep(2)
                response = await client.post(
                    proxy.base_url + "/v1/chat/completions", headers=headers, json=body
                )
                assert response.status_code == 200, _runtime_logs(root + "/r")
                assert proxy.process.poll() is None
            await proxy.stop()
            # With no live proxy left, the proxy parent ends too.
            for _ in range(200):
                if not list(Path(root, "r").glob("*.sock")):
                    break
                time.sleep(0.05)
            assert not list(Path(root, "r").glob("*.sock"))
        finally:
            if proxy is not None and await proxy.is_running():
                await proxy.stop()
            mock.shutdown()
            mock.server_close()


_PARK_PROBE = r"""
import asyncio, json, os, socket, sys
from pathlib import Path
from benchflow import _park
from benchflow._dotenv import load_dotenv_env
ours, theirs = socket.socketpair()
spool_dir = Path(sys.argv[1]) / "daemon"
spool_dir.mkdir()
spool = (spool_dir / "out", spool_dir / "err")
for path in spool:
    path.write_text("")
_park.begin_parked(theirs, spool)
load_dotenv_env()
_park._recording = False
baseline = _park.snapshot()
Path(os.environ.get("BENCHFLOW_DOTENV_PATH", ".env")).write_text("OPENAI_API_KEY=sk-new\n")
print(json.dumps({"same": _park.snapshot() == baseline, "cold": load_dotenv_env()}))
"""


@pytest.mark.parametrize("named", [False, True])
def test_a_new_dotenv_stops_the_hand_over(tmp_path, named):
    """Guards PERF-F-B-3: .env is only probed (exists) when absent; one made
    after preparation changes a cold call's credentials, so the prepared
    call's re-check must differ (it then runs cold)."""
    work = tmp_path / "call"
    work.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "BENCHFLOW_DOTENV_PATH"
    }
    if named:
        env["BENCHFLOW_DOTENV_PATH"] = str(tmp_path / "named.env")
    result = subprocess.run(
        [sys.executable, "-c", _PARK_PROBE, str(tmp_path)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    found = json.loads(result.stdout.splitlines()[-1])
    assert found["cold"] == {"OPENAI_API_KEY": "sk-new"}
    assert found["same"] is False


_PARK_PROBED = r"""
import json, os, socket, sys
from pathlib import Path
from benchflow import _park
ours, theirs = socket.socketpair()
spool_dir = Path(sys.argv[1]) / "daemon"
spool_dir.mkdir()
spool = (spool_dir / "out", spool_dir / "err")
_park.begin_parked(theirs, spool)
probe = Path(sys.argv[1]) / "credentials.json"
probe.exists()
_park._recording = False
baseline = _park.snapshot()
probe.write_text("{}")
print(json.dumps(_park.snapshot() == baseline))
"""


def test_a_file_only_probed_while_preparing_is_checked_again(tmp_path):
    """Guards PERF-F-B-3: any path probed by exists()/is_file() (credential
    and config lookups) counts, not only files opened."""
    result = subprocess.run(
        [sys.executable, "-c", _PARK_PROBED, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) is False


def test_preloads_refuse_import_time_randomness_they_do_not_redraw():
    """Guards PERF-F-B-L1: a value drawn at import is the same in every fork;
    each module drawing one must be redrawn (or shown harmless) by name."""
    proxy = "import benchflow._proxy_zygote as z\nz.IMPORT_RANDOMNESS.clear()\nz.preload()\n"
    result = subprocess.run(
        [sys.executable, "-c", proxy],
        env=_parent_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert (
        "Unaudited import-time randomness in proxy preload: litellm.integrations.custom_guardrail"
        in result.stderr
    )
    own = "import benchflow._daemon as d\nd._REDRAWN_AT_FORK = frozenset()\nd._preload()\n"
    result = subprocess.run(
        [sys.executable, "-c", own], capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert (
        "Runtime preload drew randomness at import (benchflow.sandbox.lockdown)"
        in result.stderr
    )


def test_a_proxy_child_draws_litellms_guardrail_token_again():
    """Guards PERF-F-B-L1: LiteLLM's per-process guardrail marker secret."""
    from litellm.integrations import custom_guardrail

    from benchflow._proxy_zygote import _redraw_import_randomness

    before = custom_guardrail._PRE_CALL_EXECUTED_TOKEN
    _redraw_import_randomness()
    assert before != custom_guardrail._PRE_CALL_EXECUTED_TOKEN
    assert len(custom_guardrail._PRE_CALL_EXECUTED_TOKEN) == 32


def test_identity_covers_package_data_and_start_hooks(tmp_path, monkeypatch):
    """Guards PERF-F-B-L2: package data (templates, allowlists) and the
    contents of .pth / sitecustomize files change the installation."""
    from benchflow import _daemon

    package = tmp_path / "pkg" / "benchflow"
    package.mkdir(parents=True)
    (package / "_daemon.py").write_text("")
    (package / "allowlist.json").write_text("[]")
    site = tmp_path / "prefix" / "site-packages"
    site.mkdir(parents=True)
    (site / "hook.pth").write_text("import a\n")
    monkeypatch.setattr(_daemon, "__file__", str(package / "_daemon.py"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "prefix"))
    monkeypatch.setattr(sys, "path", ["", str(site)])
    before = _daemon.identity()
    (package / "allowlist.json").write_text('["x"]')
    after_data = _daemon.identity()
    assert after_data != before
    (site / "hook.pth").write_text("import b\n")
    assert _daemon.identity() != after_data


_PARK_DISCARD = r"""
import asyncio, socket, sys
from pathlib import Path
from benchflow import _park
ours, theirs = socket.socketpair()
spool = (Path(sys.argv[1]) / "park.out", Path(sys.argv[1]) / "park.err")
for path in spool:
    path.write_text("held")
_park.begin_parked(theirs, spool)
ours.close()  # the runtime is gone


class Entry:
    pass


class Pool:
    def __init__(self, task):
        self._entries = {0: Entry()}
        self._entries[0].task = task

    async def drain(self):
        pass


async def main():
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)
    await _park.hold(Pool(done))  # prepared; telling the runtime fails


asyncio.run(main())
raise SystemExit("hold returned")
"""


def test_a_parked_call_whose_runtime_is_gone_removes_its_spool(tmp_path):
    """Guards PERF-F-B-L3: with nobody left to read it, a parked call ends
    quietly and removes its held output itself."""
    result = subprocess.run(
        [sys.executable, "-c", _PARK_DISCARD, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.glob("park.*"))


def test_resume_latest_skips_a_job_prepared_ahead(tmp_path):
    """Guards PERF-F-B-L4: a waiting prepared job is no job: an unnamed rerun
    resumes the finished one, as it would without the runtime."""
    from benchflow import _park
    from benchflow.evaluation import _job_folders

    done = tmp_path / "2026-10-19__10-00-00"
    (done / "task__aaaa").mkdir(parents=True)
    # The keys of the record every job writes as it starts.
    (done / "evaluation.json").write_text(
        '{"schema_version": 1, "tasks_dir": "/t", "job_name": "j", "config": {}}'
    )
    waiting = tmp_path / "2026-10-19__10-05-00"
    (waiting / "task__bbbb").mkdir(parents=True)
    (waiting / _park.PREPARED_MARKER).touch()
    assert _job_folders(tmp_path) == [done]


def test_dockerfile_bases_are_read_as_buildkit_does(tmp_path):
    """Guards PERF-F-A-H-1: the images a build comes from, named exactly
    (ARG defaults, the build's args, stages, scratch); None when one cannot be."""
    from benchflow._park import _dockerfile_bases

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        'ARG BASE=perf-f/base:t\nARG TOOL="perf-f/tool:1"\n'
        "FROM ${TOOL} AS tools\nFROM $BASE\nCOPY --from=tools /x /x\n"
        "FROM tools\nFROM scratch\n"
    )
    assert _dockerfile_bases(dockerfile, {}) == ["perf-f/tool:1", "perf-f/base:t"]
    assert _dockerfile_bases(dockerfile, {"BASE": "perf-f/other:2"}) == [
        "perf-f/tool:1",
        "perf-f/other:2",
    ]
    dockerfile.write_text("FROM ${UNSET}\n")
    assert _dockerfile_bases(dockerfile, {}) is None


def test_attached_calls_never_import_from_the_runtimes_start_folder(tmp_path):
    """Guards PERF-F-A-M-1: `python -m` put the folder `bench daemon start` ran
    in first on sys.path; a module there shadowed one attached calls import."""
    start = tmp_path / "start"
    start.mkdir()
    marker = tmp_path / "shadowed"
    for name in ("colorsys", "getpass", "configparser"):
        (start / f"{name}.py").write_text(
            f"open({str(marker)!r}, 'a').write({name!r} + '\\n')\n"
            "import importlib, sys\n"
            f"del sys.modules[{name!r}]\n"
            "sys.path = [p for p in sys.path if p not in ('', "
            + repr(str(start))
            + ")]\n"
            f"globals().update(vars(importlib.import_module({name!r})))\n"
        )
    env = dict(os.environ, BENCHFLOW_DAEMON_DIR=str(tmp_path / "runtime"))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    started = subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", "daemon", "start"],
        cwd=start,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert started.returncode == 0, started.stderr
    try:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        call = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchflow.cli.main",
                "eval",
                "run",
                "--tasks-dir",
                str(tmp_path / "missing"),
            ],
            env=dict(env, BENCHFLOW_DAEMON="auto"),
            cwd=elsewhere,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert call.returncode != 0
        status = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from benchflow._daemon import request; print(json.dumps(request({'action': 'status'})))",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert json.loads(status.stdout)["accepted_calls"] == 1
        assert not marker.exists(), marker.read_text()
    finally:
        for _ in range(100):
            if cli(["daemon", "stop"], env).returncode == 0:
                break
            time.sleep(0.05)


def test_reclaim_removes_only_what_a_killed_prepared_call_named(tmp_path, monkeypatch):
    """Guards PERF-F-A-L5: leftovers of a SIGKILLed prepared call (or runtime)
    go by recorded name; a folder without the marker, or with a result, stays."""
    from benchflow import _daemon, _park

    calls = []
    monkeypatch.setattr(
        _daemon.subprocess,
        "run",
        lambda *a, **k: (
            calls.append(a[0]) or subprocess.CompletedProcess(a[0], 0, "", "")
        ),
    )
    marked = tmp_path / "jobs" / "2026-10-20__01-00-00"
    (marked / "t__1").mkdir(parents=True)
    (marked / _park.PREPARED_MARKER).touch()
    finished = tmp_path / "jobs" / "2026-10-20__00-00-00"
    (finished / "t__0").mkdir(parents=True)
    _daemon.reclaim({"job": str(finished)}, {})
    _daemon.reclaim({"job": str(marked), "project": "perf-f-p-1"}, {})
    assert finished.exists() and not marked.exists()
    assert [
        "docker",
        "container",
        "ls",
        "-aq",
        "--filter",
        "label=com.docker.compose.project=perf-f-p-1",
    ] in calls

    runtime = _daemon._Runtime(tmp_path, 1, 10)
    spool = tmp_path / "park-x.out"
    spool.write_text("held")
    runtime._write(
        {"999999999": {"start": 1, "spool": [str(spool)], "made": {}, "docker_env": {}}}
    )
    runtime.reclaim_dead()
    assert not spool.exists() and runtime._read() == {}


def test_a_caller_ends_as_its_oom_killed_worker_did(tmp_path):
    """Guards PERF-F-A-L6: a worker killed by a signal nobody passed on (the
    OOM killer's SIGKILL) ends its caller the same way (137), not with 1."""
    from benchflow._daemon_wire import listen, receive, receive_fds, send

    root = tmp_path / "rt"
    listener = listen(root / "runtime.sock")
    client_script = "import sys\nfrom benchflow._daemon import try_attach\nsys.exit(try_attach(['eval', 'run', '--help']))\n"
    env = dict(os.environ, BENCHFLOW_DAEMON="auto", BENCHFLOW_DAEMON_DIR=str(root))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    client = subprocess.Popen(
        [sys.executable, "-c", client_script],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with listener.accept()[0] as conn:
            receive(conn)
            send(conn, {"ready": True})
            for fd in receive_fds(conn, 3):
                os.close(fd)
            send(conn, {"pid": 4242})
        # The connection broke; the caller asks how the worker ended.
        with listener.accept()[0] as conn:
            assert receive(conn) == {"action": "ended", "pid": 4242}
            send(conn, {"exit": -9})
        assert client.wait(30) == -9
    finally:
        if client.poll() is None:
            client.kill()
        listener.close()


def test_start_refuses_a_runtime_it_would_not_have_started(tmp_path):
    """Guards PERF-F-A-L7: other --prepare settings (or another installation)
    are an error with a next step, never "ready"."""
    env = dict(os.environ, BENCHFLOW_DAEMON_DIR=str(tmp_path / "runtime"))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    assert cli(["daemon", "start"], env).returncode == 0
    try:
        again = cli(["daemon", "start", "--prepare", "1"], env)
        assert again.returncode == 1
        assert "already running" in again.stderr and "--prepare 0" in again.stderr
        assert "bench daemon stop" in again.stderr
        assert "ready" not in again.stdout
        assert cli(["daemon", "start"], env).returncode == 0
    finally:
        for _ in range(100):
            if cli(["daemon", "stop"], env).returncode == 0:
                break
            time.sleep(0.05)


_PARK_SUPPORTS = r"""
import json, os, socket, sys
from pathlib import Path
from benchflow import _park
before = [os.stat in os.supports_dir_fd, os.stat in os.supports_follow_symlinks, os.lstat in os.supports_dir_fd, os.stat.__name__]
ours, theirs = socket.socketpair()
spool = (Path(sys.argv[1]) / "out", Path(sys.argv[1]) / "err")
_park.begin_parked(theirs, spool)
print(json.dumps([before, [os.stat in os.supports_dir_fd, os.stat in os.supports_follow_symlinks, os.lstat in os.supports_dir_fd, os.stat.__name__]]))
"""


def test_probe_recording_keeps_os_feature_detection():
    """Guards PERF-F: the safe writes ask os.supports_dir_fd; a
    recorded os.stat must answer as the original, or the trial fails."""
    with tempfile.TemporaryDirectory(prefix="bf-f-") as root:
        result = subprocess.run(
            [sys.executable, "-c", _PARK_SUPPORTS, root],
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, result.stderr
    before, after = json.loads(result.stdout.splitlines()[-1])
    assert before == after


def _attach_counted(tmp_path, env, wrap=(), preexec=None):
    """Run one attachable call (it fails fast: no tasks) and return how many
    calls the runtime accepted, and the runtime's log."""
    call = subprocess.run(
        [
            *wrap,
            sys.executable,
            "-m",
            "benchflow.cli.main",
            "eval",
            "run",
            "--tasks-dir",
            str(tmp_path / "missing"),
        ],
        env=dict(env, BENCHFLOW_DAEMON="auto"),
        preexec_fn=preexec,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert call.returncode != 0
    status = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from benchflow._daemon import request; print(json.dumps(request({'action': 'status'})))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(status.stdout)["accepted_calls"], (
        tmp_path / "runtime" / "daemon.log"
    ).read_text()


@contextlib.contextmanager
def _runtime(tmp_path, wrap=()):
    env = dict(os.environ, BENCHFLOW_DAEMON_DIR=str(tmp_path / "runtime"))
    env.pop("_BENCHFLOW_DAEMON_WORKER", None)
    started = subprocess.run(
        [*wrap, sys.executable, "-m", "benchflow.cli.main", "daemon", "start"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert started.returncode == 0, started.stderr
    try:
        yield env
    finally:
        for _ in range(100):
            if cli(["daemon", "stop"], env).returncode == 0:
                break
            time.sleep(0.05)


_SETPRIV = shutil.which("setpriv")


@pytest.mark.skipif(_SETPRIV is None, reason="needs util-linux setpriv")
def test_a_no_new_privs_caller_runs_cold(tmp_path):
    """Guards PERF-F-r3-B-M1: a confined caller (no_new_privs, as an
    agent sandbox sets it) never gets an unconfined worker."""
    with _runtime(tmp_path) as env:
        accepted, log = _attach_counted(
            tmp_path, env, wrap=(_SETPRIV, "--no-new-privs")
        )
        assert accepted == 0
        assert "process attributes: confinement (NoNewPrivs)" in log
        accepted, _ = _attach_counted(tmp_path, env)
        assert accepted == 1


def _allow_all_seccomp():
    """In a child: no_new_privs, then one allow-everything seccomp filter."""
    import ctypes
    import struct as _struct

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    assert libc.prctl(38, 1, 0, 0, 0) == 0  # PR_SET_NO_NEW_PRIVS
    instruction = _struct.pack(
        "HBBI", 0x06, 0, 0, 0x7FFF0000
    )  # BPF_RET|BPF_K, SECCOMP_RET_ALLOW
    filter_buffer = ctypes.create_string_buffer(instruction)
    program = _struct.pack("HxxxxxxP", 1, ctypes.addressof(filter_buffer))
    program_buffer = ctypes.create_string_buffer(program)
    assert (
        libc.prctl(22, 2, ctypes.addressof(program_buffer), 0, 0) == 0
    )  # PR_SET_SECCOMP, FILTER


@pytest.mark.skipif(
    _SETPRIV is None or sys.platform != "linux", reason="needs Linux and setpriv"
)
def test_a_caller_with_an_extra_seccomp_filter_runs_cold(tmp_path):
    """Under a runtime that already has no_new_privs, a caller
    that adds a seccomp filter differs only by it, and runs cold."""
    with _runtime(tmp_path, wrap=(_SETPRIV, "--no-new-privs")) as env:
        accepted, _ = _attach_counted(tmp_path, env, wrap=(_SETPRIV, "--no-new-privs"))
        assert accepted == 1  # the same confinement attaches
        accepted, log = _attach_counted(tmp_path, env, preexec=_allow_all_seccomp)
        assert accepted == 1
        # The filter count differs (its mode too, unless a container already
        # filters both sides).
        assert "process attributes: confinement (" in log and "Seccomp_filters" in log


def _can_unshare_net() -> list[str] | None:
    for wrap in (["unshare", "--net"], ["unshare", "-rn"]):
        if (
            shutil.which("unshare")
            and subprocess.run([*wrap, "true"], capture_output=True).returncode == 0
        ):
            return wrap
    return None


@pytest.mark.skipif(
    _can_unshare_net() is None,
    reason="needs a network namespace (root or user namespaces)",
)
def test_a_caller_in_another_network_namespace_runs_cold(tmp_path):
    """Guards PERF-F-r3-B-M1: a caller with no network (its own
    namespace) never gets a worker in the runtime's."""
    wrap = _can_unshare_net()
    with _runtime(tmp_path) as env:
        accepted, log = _attach_counted(tmp_path, env, wrap=wrap)
        assert accepted == 0
        assert "process attributes: namespaces (" in log and "net" in log


@pytest.mark.skipif(
    _SETPRIV is None or os.geteuid() != 0,
    reason="dropping capabilities needs root and setpriv",
)
def test_a_caller_with_dropped_capabilities_runs_cold(tmp_path):
    """Capabilities a caller dropped stay dropped: it runs cold."""
    with _runtime(tmp_path) as env:
        accepted, log = _attach_counted(
            tmp_path, env, wrap=(_SETPRIV, "--bounding-set", "-net_raw")
        )
        assert accepted == 0
        assert "process attributes: confinement (CapBnd" in log


def test_unreadable_confinement_is_never_assumed_equal():
    """A confinement field that cannot be read on either side
    keeps the call cold."""
    from benchflow import _daemon

    mine = _daemon._process()
    value = {
        "identity": "i",
        "interpreter": "x",
        "env": {},
        "process": json.loads(json.dumps(mine)),
    }
    assert _daemon._admission_mismatch(value, "i", "x", {}, {}, mine) is None
    theirs = json.loads(json.dumps(mine))
    theirs["lsm"]["current"] = None
    reason = _daemon._admission_mismatch(
        dict(value, process=theirs), "i", "x", {}, {}, mine
    )
    assert reason == "process attributes unreadable: lsm current"


def _eacces(*_args: Any, **_kwargs: Any) -> Any:
    raise PermissionError(errno.EACCES, "denied by the test")


def _deny_read(monkeypatch, path: str) -> None:
    real = Path.read_text

    def read_text(self, *args, **kwargs):
        if str(self) == path:
            _eacces()
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


def _deny_path(monkeypatch, function: str, match: Any) -> None:
    real = getattr(os, function)

    def denied(path, *args, **kwargs):
        if match(str(path)):
            _eacces()
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, function, denied)


_UNREADABLE = {
    "confinement": lambda m: _deny_read(m, "/proc/self/status"),
    "namespaces listing": lambda m: _deny_path(
        m, "listdir", lambda p: p == "/proc/self/ns"
    ),
    "namespaces link": lambda m: _deny_path(
        m, "readlink", lambda p: p.startswith("/proc/self/ns/")
    ),
    "lsm current": lambda m: _deny_read(m, "/proc/self/attr/current"),
    "lsm listing": lambda m: _deny_path(m, "listdir", lambda p: p == "/proc/self/attr"),
    "lsm entry": lambda m: _deny_path(
        m, "stat", lambda p: p.startswith("/proc/self/attr/")
    ),
    "cgroup": lambda m: _deny_read(m, "/proc/self/cgroup"),
    "groups": lambda m: m.setattr(os, "getgroups", _eacces),
    "rlimits": lambda m: m.setattr("resource.getrlimit", _eacces),
    "affinity": lambda m: m.setattr(os, "sched_getaffinity", _eacces),
    "priority": lambda m: m.setattr(os, "getpriority", _eacces),
}


@pytest.mark.parametrize("field", sorted(_UNREADABLE))
def test_every_compared_attribute_unreadable_on_both_sides_runs_cold(
    monkeypatch, field
):
    """Guards PERF-F-r4-M3 and r4-L1: EACCES on any field the check compares,
    the same on the caller's and the runtime's side, keeps the call cold:
    two unknowns are not equal (getgid and getegid cannot fail)."""
    from benchflow import _daemon

    if not Path("/proc/self/attr").is_dir():
        pytest.skip("needs Linux /proc")
    _UNREADABLE[field](monkeypatch)
    mine = _daemon._process()
    monkeypatch.undo()
    value = {
        "identity": "i",
        "interpreter": "x",
        "env": {},
        "process": json.loads(json.dumps(mine)),
    }
    reason = _daemon._admission_mismatch(value, "i", "x", {}, {}, mine)
    assert reason is not None
    assert reason.startswith("process attributes unreadable: " + field.split()[0]), (
        reason
    )


def test_every_benchflow_proxy_environment_turns_python_dotenv_off():
    """BenchFlow's LiteLLM proxy reads no .env file.
    The host proxy's environment (cold, and the parent's key and child, which
    are built from it) and the sandbox launcher's both turn python-dotenv
    off, whatever value the caller had."""
    from benchflow.providers.litellm_runtime import (
        _proxy_process_env,
        _sandbox_launcher_source,
    )

    env = _proxy_process_env({"PYTHON_DOTENV_DISABLED": "0", "KEY": "v"}, "/rt")
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert env["KEY"] == "v"
    assert 'env["PYTHON_DOTENV_DISABLED"] = "1"' in _sandbox_launcher_source()


def test_the_locked_litellm_reads_a_dotenv_only_through_the_switched_call():
    """The switch is enough: python-dotenv's load_dotenv() checks it before it
    searches for a file, and the locked LiteLLM reads a .env by no other
    call. A future one that does (dotenv_values, say) refuses the parent:
    test_a_dotenv_parsed_during_the_preload_refuses_the_parent."""
    import importlib.util
    import inspect

    import dotenv.main

    source = inspect.getsource(dotenv.main.load_dotenv)
    assert source.index("if _load_dotenv_disabled():") < source.index(
        "dotenv_path = find_dotenv()"
    )
    spec = importlib.util.find_spec("litellm")
    assert spec is not None and spec.origin is not None
    package = Path(spec.origin).parent
    calls = {}
    for path in package.rglob("*.py"):
        if {"tests", "test"} & set(path.relative_to(package).parts):
            continue
        text = path.read_text(errors="replace")
        for call in ("load_dotenv(", "dotenv_values(", "find_dotenv(", "DotEnv("):
            if call in text:
                calls.setdefault(call, []).append(str(path.relative_to(package)))
    assert {call: sorted(paths) for call, paths in calls.items()} == {
        "load_dotenv(": ["__init__.py", "proxy/proxy_cli.py"]
    }


def test_a_proxy_parent_or_launch_without_dotenv_off_refuses(tmp_path):
    """A parent only exists, and only forks, for an
    environment that turns python-dotenv off; otherwise proxies start cold."""
    from benchflow import _daemon
    from benchflow import _proxy_zygote as z

    env = {k: v for k, v in os.environ.items() if k != "PYTHON_DOTENV_DISABLED"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from benchflow._proxy_zygote import preload\npreload()",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "python-dotenv is not off in the proxy parent's environment" in (
        result.stderr
    )
    assert z._launch_refusal(({}, {}), {"PATH": "/bin"}) == (
        "python-dotenv is not off in this proxy's environment"
    )
    zygotes = _daemon._Zygotes(tmp_path)
    assert zygotes.get({"PATH": "/bin"}, wait=False) is None
    assert zygotes.entries == {}


_PARSES_A_DOTENV = """import io
import dotenv

dotenv.dotenv_values(stream=io.StringIO("PERF_F_FROM_DOTENV=one\\n"))
"""


def test_a_dotenv_parsed_during_the_preload_refuses_the_parent(tmp_path):
    """The switch covers load_dotenv() only. An
    import that parses a .env another way refuses the parent (proxies start
    cold); nothing is recorded or replayed."""
    package = tmp_path / "site" / "litellm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_PARSES_A_DOTENV)
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path / 'site')!r})\n"
        "from benchflow import _proxy_zygote as z\n"
        "z.PRELOAD = ('litellm',)\n"
        "z.PRELOAD_IF_INSTALLED = ()\n"
        "z.preload()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=_parent_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "litellm parsed a .env while python-dotenv was off" in result.stderr


# A real proxy parent (the locked LiteLLM), its real child (`_answer`) and a
# real cold proxy (the `litellm` script) on BenchFlow's real runtime files,
# each recording what its imports left in the environment. `.env` files sit
# where each one's python-dotenv search would start: the folder above the
# runtime folders (a cold proxy's search, from its working folder) and the
# parent's own working folder.
_REAL_COLD_AND_CHILD = r"""
import json, os, selectors, socket, subprocess, sys, tempfile
from pathlib import Path

from benchflow import _proxy_zygote as z
from benchflow._daemon_wire import receive, send
from benchflow.providers import litellm_runtime as lr

root = Path(sys.argv[1])
names = ("PERF_F_FROM_DOTENV", "PERF_F_FROM_PARENT_CWD")
(root / ".env").write_text("PERF_F_FROM_DOTENV=runtime-ancestry\n")
parent_cwd = root / "parent"
parent_cwd.mkdir()
(parent_cwd / ".env").write_text("PERF_F_FROM_PARENT_CWD=parent\n")
source = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
source["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def runtime():
    folder = Path(tempfile.mkdtemp(prefix="rt-", dir=root))
    lr._write_runtime_files(folder, config={"model_list": []}, lazy_start=True)
    record = folder / "record.json"
    with open(folder / "sitecustomize.py", "a") as handle:
        handle.write(
            "\nimport json as _j, os as _o\nimport litellm.proxy.proxy_cli\n"
            "import sys as _s\n"
            f"_path = [p.replace({str(folder)!r}, 'RT') for p in _s.path]\n"
            f"open({str(record)!r}, 'w').write(_j.dumps("
            f"[_o.environ.get(n) for n in {names!r}] + [_path]))\n"
            "raise SystemExit(0)\n"
        )
    return folder, record, lr._proxy_process_env(dict(source), str(folder))


def cold(env, folder, record):
    litellm = str(Path(sys.executable).with_name("litellm"))
    run = subprocess.run([litellm], env=env, cwd=folder, capture_output=True,
                         text=True, timeout=180)
    assert record.exists(), run.stderr[-2000:]
    return json.loads(record.read_text())


out = {}
folder, record, env = runtime()
os.chdir(parent_cwd)
os.environ.clear()
os.environ.update({k: v for k, v in env.items() if z.parent_setting(k)})
audit = z.preload()
ours, theirs = socket.socketpair()
listener, selector = socket.socket(socket.AF_UNIX), selectors.DefaultSelector()
send(ours, {"action": "launch", "env": env, "cwd": str(folder), "args": []})
z._answer(theirs, audit, {}, {}, listener, selector, -1)
reply = receive(ours)
out["launch"] = reply
if "pid" in reply:
    os.waitpid(reply["pid"], 0)
    out["child"] = json.loads(record.read_text())
folder, record, env = runtime()
out["cold"] = cold(env, folder, record)
# Control: the same cold proxy with python-dotenv on finds the planted file.
folder, record, env = runtime()
env.pop("PYTHON_DOTENV_DISABLED", None)
out["cold_dotenv_on"] = cold(env, folder, record)
print(json.dumps(out))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="proxy parents are Linux-only")
def test_a_forked_proxy_equals_a_real_cold_proxy_with_dotenv_files_around(tmp_path):
    """Guards PERF-F r5 review A H-1: a real cold proxy imports LiteLLM in
    BenchFlow's sitecustomize, so python-dotenv searched from its working
    folder (the runtime folder, then $TMPDIR, then /) while a forked child
    replayed a search from LiteLLM's files. Neither reads a .env now: the
    child's environment and the cold one's are the same, and the control
    shows the file was on the cold search path. Their import paths are the
    same too."""
    script = tmp_path / "real.py"
    script.write_text(_REAL_COLD_AND_CHILD)
    root = tmp_path / "tmp"
    root.mkdir()
    # -P: as the runtime starts a parent (-m, PYTHONSAFEPATH), with no
    # script folder on the import path.
    result = subprocess.run(
        [sys.executable, "-P", str(script), str(root)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    found = json.loads(result.stdout.splitlines()[-1])
    assert "pid" in found["launch"], found
    assert found["child"] == found["cold"]
    assert found["cold"][:2] == [None, None]
    assert found["cold_dotenv_on"][:2] == ["runtime-ancestry", None]
    # The same import path, from the runtime folder alone: no folder found
    # from either one's working folder (LiteLLM's proxy modules add two).
    path = found["cold"][2]
    assert path[0] == "RT" and path.count("RT") == 1
    assert str(root) not in " ".join(path), path


def test_status_says_a_draining_runtime_is_stopping(tmp_path):
    """Guards PERF-F-r3-B-L1: while it waits for running calls, the runtime is
    stopping, not "not running"."""
    from benchflow._daemon import _proc_start

    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    (root / "runtime.stopping").write_text(
        json.dumps(
            {"pid": os.getpid(), "start": _proc_start(os.getpid()), "running": 2}
        )
    )
    env = dict(os.environ, BENCHFLOW_DAEMON_DIR=str(root))
    status = cli(["daemon", "status"], env)
    assert status.returncode == 0, status.stderr
    assert "stopping; it waits for its 2 running calls" in status.stdout


def test_a_worker_takes_every_ignored_signal_of_its_caller():
    """Guards PERF-F-r3-B-L4: not only the four forwarded signals: a harness
    that ignores SIGUSR1 has it ignored in the worker too, as in a cold CLI."""
    import signal as _signal

    from benchflow import _daemon

    previous = _signal.signal(_signal.SIGUSR1, _signal.SIG_IGN)
    try:
        assert int(_signal.SIGUSR1) in _daemon._signals()["ignored"]
    finally:
        _signal.signal(_signal.SIGUSR1, previous)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read)
            _daemon._apply_signals(
                {
                    "signals": {
                        "ignored": [int(_signal.SIGUSR1), int(_signal.SIGUSR2)],
                        "mask": [],
                    }
                }
            )
            state = [
                _signal.getsignal(_signal.SIGUSR1) == _signal.SIG_IGN,
                _signal.getsignal(_signal.SIGUSR2) == _signal.SIG_IGN,
                _signal.getsignal(_signal.SIGALRM) == _signal.SIG_DFL,
            ]
            os.write(write, json.dumps(state).encode())
        finally:
            os._exit(0)
    os.close(write)
    os.waitpid(pid, 0)
    assert json.loads(os.read(read, 200)) == [True, True, True]


async def test_the_handover_rebuild_keeps_the_known_main_container():
    """Guards PERF-F round 4 (ab4: +3 s per prepared call): the hand-over's
    `compose build` touches no container; the trial's execs must keep going
    to the container directly, not through `docker compose exec`."""
    from benchflow import _park

    class Sandbox:
        _use_prebuilt = False
        _uses_compose = True
        compose_project_name = "perf-f-p"
        _main_container = "c0ffee"

        async def _build_task_image(self):
            self._main_container = None  # as _run_docker_compose_command("build") does

        async def _run_docker_compose_command(self, *a, **k):
            class R:
                stdout = json.dumps(
                    {"name": "perf-f-p", "services": {"main": {"image": "img:t"}}}
                )

            return R()

        def _docker_compose_global_args(self):
            return []

        async def _run_docker_process(self, *a, **k):
            class R:
                return_code = 0
                stdout = '["sha256:aa"] {} linux amd64'
                stderr = ""

            return R()

    sandbox = Sandbox()
    built = await _park._service_images(sandbox)
    assert await _park._rebuild_changes(sandbox, built) == ""
    assert sandbox._main_container == "c0ffee"
