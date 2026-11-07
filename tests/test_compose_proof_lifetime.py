"""Guards S1-CONF-6/7/8 at 6dfad9e5: incomplete proofs never authorize reuse."""

import json
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import SYSTEM_PATH
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult, SandboxImage
from benchflow.task import compose_env
from benchflow.task.compose_env import _ProjectCheck, check_compose_in_vm
from benchflow.task.config import SandboxConfig
from tests.test_compose_mutable_sources import _project


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in (
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "BENCHFLOW_ALLOW_TASK_HOST_ACCESS",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("alias", ["view", "sub/view", "sub/nested/view"])
@pytest.mark.parametrize(
    "writer", ["bind", "sidecar", "volume", "volume-rbind", "inherited"]
)
def test_repeated_identity_refuses_instead_of_skipping(
    tmp_path, monkeypatch, alias, writer
):
    """CONF-6: an alias at any depth can carry different descendant mounts."""
    root = tmp_path / "work"
    (root / alias / "slot").mkdir(parents=True)
    (tmp_path / "export/data").mkdir(parents=True)
    identity = compose_env._local_identity
    monkeypatch.setattr(
        compose_env,
        "_local_identity",
        lambda p: identity(str(root)) if p == str(root / alias) else identity(p),
    )
    refused = _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(
        _project(root, tmp_path / "export/data", writer, "bind")
    )
    assert refused, "a repeated identity is an incomplete proof, not a visited subtree"
    assert "repeated" in " ".join(r.describe() for r in refused)


@pytest.mark.parametrize(
    "flags", ["bind", "rw,bind", "bind,rw", "BIND", "bind,rbind", "rbind,bind"]
)
@pytest.mark.parametrize("subpath", [None, "sub/nested"])
@pytest.mark.parametrize("backend", ["local", "vm"])
async def test_nonrecursive_bind_with_descendant_mount_refuses(
    tmp_path, flags, subpath, backend
):
    """CONF-7: the hidden tree cannot be proved by walking the covering mount."""
    root = tmp_path / "work"
    (root / "sub/nested").mkdir(parents=True)
    (tmp_path / "export/data").mkdir(parents=True)
    project = _project(root, tmp_path / "export/data", "volume", "bind")
    project["volumes"]["work"]["driver_opts"]["o"] = flags
    if subpath:
        project["services"]["main"]["volumes"][0]["volume"] = {"subpath": subpath}
    mountinfo = (
        f"1 0 0:1 / / rw - ext4 dummy rw\n2 1 0:2 / {root}/sub rw - tmpfs dummy rw\n"
    )
    check = _ProjectCheck(str(tmp_path), (str(tmp_path),))
    check.mountpoints = lambda: {"/", str(root / "sub")}
    if backend == "local":
        refused = check.check(project)
    else:

        async def run(command):
            if command.startswith("awk ") and command.endswith("/proc/self/mountinfo"):
                framed = (
                    "".join(
                        f"{i} {line}\n"
                        for i, line in enumerate(mountinfo.splitlines(), 1)
                    )
                    + "complete 2\n"
                )
                return ExecResult(stderr="", return_code=0, stdout=framed)
            argv = shlex.split(command)
            if "config" in argv:
                return ExecResult(stderr="", return_code=0, stdout=json.dumps(project))
            if argv[:2] == ["sh", "-c"]:
                out = subprocess.run(argv, capture_output=True, text=True)
                return ExecResult(
                    stderr="", return_code=out.returncode, stdout=out.stdout
                )
            return ExecResult(stderr="", return_code=0, stdout="x$$y")

        refused = await check_compose_in_vm(
            run, ["docker", "compose"], str(tmp_path), [str(tmp_path)]
        )
    assert any("nonrecursive" in r.describe() for r in refused), (
        "a visible covering tree does not prove the nonrecursive bind"
    )


@pytest.mark.parametrize(
    "selection",
    [
        "host-tcp",
        "host-ssh",
        "host-unix",
        "env-context",
        "saved-context",
        "flag-host",
        "flag-equals",
        "flag-context",
        "unknown-flag",
    ],
)
def test_daemon_selection_cannot_use_client_proof(tmp_path, monkeypatch, selection):
    """CONF-8: every daemon selector makes client-side filesystem facts unknown."""
    # Exercise freeze_launch's policy selection without requiring any daemon.
    (tmp_path / "docker-compose.yaml").write_text("services: {main: {image: trusted}}")
    paths = [
        tmp_path / n
        for n in (
            "work",
            "export/data",
            "logs/verifier",
            "logs/agent",
            "logs/artifacts",
        )
    ]
    for p in paths:
        p.mkdir(parents=True)
    env = DockerSandbox(tmp_path, "proof", "proof", None, SandboxConfig())
    if selection.startswith("host-"):
        monkeypatch.setenv(
            "DOCKER_HOST",
            {
                "host-tcp": "tcp://worker:2376",
                "host-ssh": "ssh://worker",
                "host-unix": "unix:///tmp/alternate.sock",
            }[selection],
        )
    elif selection == "env-context":
        monkeypatch.setenv("DOCKER_CONTEXT", "worker")
    elif selection == "saved-context":
        config = tmp_path / "client"
        config.mkdir()
        (config / "config.json").write_text('{"currentContext":"worker"}')
        monkeypatch.setenv("DOCKER_CONFIG", str(config))
    else:
        monkeypatch.setattr(
            env,
            "_docker_compose_global_args",
            lambda: {
                "flag-host": ["-H", "tcp://worker:2376"],
                "flag-equals": ["--host=tcp://worker:2376"],
                "flag-context": ["--context", "worker"],
                "unknown-flag": ["--future-endpoint", "worker"],
            }[selection],
        )
    seen = []

    def freeze(**kwargs):
        seen.extend(kwargs["check"].check(_project(paths[0], paths[1], "bind", "bind")))
        return SimpleNamespace(refused=seen, launch=object())

    monkeypatch.setattr(compose_env, "freeze_compose", freeze)
    monkeypatch.setattr(
        env,
        "_stage",
        lambda: SimpleNamespace(environment_dir=tmp_path, task_dir=tmp_path),
    )
    try:
        env.freeze_launch()
    except Exception as exc:
        assert "local daemon" in str(exc)
    assert seen, "a selected daemon must never receive a client filesystem proof"


def make_env(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM trusted")
    return DockerSandbox(tmp_path, "proof", "proof", None, SandboxConfig())


@pytest.mark.parametrize("repeated", ["root", "nested"])
async def test_vm_receipt_cannot_deduplicate_aliases(tmp_path, repeated):
    """CONF-6: numbered, complete output still refuses a repeated directory."""
    root = tmp_path / "work"
    root.mkdir()
    (tmp_path / "export/data").mkdir(parents=True)
    project = _project(root, tmp_path / "export/data", "bind", "bind")

    async def run(command):
        argv = shlex.split(command)
        if "config" in argv:
            return ExecResult(stderr="", return_code=0, stdout=json.dumps(project))
        if command.startswith("awk ") and command.endswith("/proc/self/mountinfo"):
            return ExecResult(
                stderr="",
                return_code=0,
                stdout="1 1 0 0:1 / / rw - ext4 dummy rw\ncomplete 1\n",
            )
        if argv[:2] == ["sh", "-c"]:
            if "find" in argv[2]:
                ids = ["8:1", "8:2", "8:1" if repeated == "root" else "8:2"]
                text = (
                    "".join(f"{i} {v}\n" for i, v in enumerate(ids, 1)) + "complete 3\n"
                )
                return ExecResult(stderr="", return_code=0, stdout=text)
            out = subprocess.run(argv, capture_output=True, text=True)
            return ExecResult(stderr="", return_code=out.returncode, stdout=out.stdout)
        return ExecResult(stderr="", return_code=0, stdout="x$$y")

    refused = await check_compose_in_vm(
        run, ["docker", "compose"], str(tmp_path), [str(tmp_path)]
    )
    assert refused, "a repeated identity can hide a distinct nested mount"


@pytest.mark.parametrize("endpoint", ["local", "ssh", "tcp"])
@pytest.mark.parametrize("source_kind", ["directory", "file"])
async def test_real_symlink_source_swap_between_starts(tmp_path, endpoint, source_kind):
    """Guards 6dfad9e5: real stat follows a replaced source in the daemon namespace."""
    env = make_env(tmp_path)
    if endpoint != "local":
        from benchflow.sandbox.remote_docker import (
            RemoteDockerHost,
            RemoteDockerSandbox,
        )

        env = RemoteDockerSandbox(
            tmp_path,
            "proof",
            "proof",
            None,
            SandboxConfig(),
            docker_host=RemoteDockerHost(url=f"{endpoint}://worker:2376"),
        )
    for name in ("original", "dummy-operator-only"):
        path = tmp_path / name
        if source_kind == "directory":
            path.mkdir()
        else:
            path.write_text(name)
    source = tmp_path / "source"
    source.symlink_to(tmp_path / "original")
    cid = "a" * 64
    env._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stderr="", return_code=0, stdout=cid)
    )
    env._inspect_container = AsyncMock(
        return_value={
            "Config": {"Labels": {"com.docker.compose.service": "main"}},
            "Mounts": [{"Type": "bind", "Destination": str(source)}],
        }
    )

    async def cli(args, **kwargs):
        assert args[:3] == ["exec", "-u", "0"] and cid in args
        out = subprocess.run(
            args[args.index(cid) + 1 :], capture_output=True, text=True
        )
        return ExecResult(
            stderr=out.stderr, return_code=out.returncode, stdout=out.stdout
        )

    env._docker_cli = AsyncMock(side_effect=cli)
    await env._pin_or_verify_bind_targets()
    original = env._snapshot_bind_baseline()
    # Both first starts see the same actual source; the swap follows them.
    await env._pin_or_verify_bind_targets()
    source.unlink()
    source.symlink_to(tmp_path / "dummy-operator-only")
    with pytest.raises(RuntimeError, match="identity changed"):
        await env._pin_or_verify_bind_targets()
    assert env._bind_baseline == original


@pytest.mark.parametrize("endpoint", ["local", "ssh", "tcp"])
@pytest.mark.parametrize("changed", ["device", "inode", "missing", "new", "truncated"])
async def test_restore_rejects_changed_container_target(tmp_path, endpoint, changed):
    """CONF-6/7/8 lifetime: the daemon-side mount must retain its first identity."""
    env = make_env(tmp_path)
    # The seam below represents daemon responses, not client-side stat calls.
    if endpoint != "local":
        env._remote_daemon = True
    cid = "a" * 64
    replacement = "b" * 64
    mount = {
        "Type": "bind",
        "Source": "/srv/input",
        "Destination": "/data",
        "RW": False,
    }
    inspected = {
        "HostConfig": {"NetworkMode": "none"},
        "Config": {"Labels": {}},
        "Mounts": [mount],
    }
    receipt = {"main": {"/data": "1:10"}}

    def cli(args, **kwargs):
        if args[0] == "inspect":
            data = dict(inspected)
            if args[-1] == replacement and changed == "new":
                data["Mounts"] = [mount, {**mount, "Destination": "/extra"}]
            return ExecResult(stderr="", return_code=0, stdout=json.dumps([data]))
        if args[:2] == ["image", "inspect"]:
            return ExecResult(
                stderr="",
                return_code=0,
                stdout=json.dumps(
                    {"Labels": {"io.benchflow.bind-targets.v1": json.dumps(receipt)}}
                ),
            )
        if args[0] == "run":
            return ExecResult(stderr="", return_code=0, stdout=replacement)
        if args[0] == "exec" and "stat -L -c" not in args[-1]:
            return ExecResult(stderr="", return_code=0, stdout="")
        if args[0] == "exec":
            values = {
                "device": "2:10\n",
                "inode": "1:11\n",
                "missing": "",
                "new": "1:10\n1:12\n",
                "truncated": "1:",
            }
            return ExecResult(
                stderr="", return_code=int(changed == "missing"), stdout=values[changed]
            )
        return ExecResult(stderr="", return_code=0, stdout="")

    env._main_container_id = AsyncMock(return_value=cid)
    env._docker_cli = AsyncMock(side_effect=cli)
    with pytest.raises(RuntimeError, match=r"bind identit|bind identity"):
        await env.restore(SandboxImage(provider="docker", ref="checkpoint"))
    commands = [c.args[0] for c in env._docker_cli.call_args_list]
    assert ["rm", "-f", replacement] in commands


@pytest.mark.parametrize("kind", ["bind", "volume"])
async def test_restarts_verify_receipt_and_do_not_relearn(tmp_path, kind):
    """6dfad9e5 lifetime: normal restart, retry and reuse share one initial receipt."""
    from benchflow.sandbox import _bind_targets

    env = make_env(tmp_path)
    cid = "a" * 64
    inspected = {
        "Config": {"Labels": {"com.docker.compose.service": "main"}},
        "Mounts": [{"Type": kind, "Name": "work", "Destination": "/data"}],
    }
    env._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stderr="", return_code=0, stdout=cid)
    )
    env._inspect_container = AsyncMock(return_value=inspected)
    identity = "1:10"
    env._docker_cli = AsyncMock(
        side_effect=lambda args, **kw: ExecResult(
            stderr="",
            return_code=0,
            stdout=json.dumps([{"Options": {"o": "bind", "device": "/srv/work"}}])
            if args[0] == "volume"
            else identity,
        )
    )
    await env._pin_or_verify_bind_targets()
    assert env._bind_baseline == {"main": {"/data": "1:10"}}
    await env._pin_or_verify_bind_targets()
    identity = "1:11"
    with pytest.raises(RuntimeError, match="identity changed"):
        await env._pin_or_verify_bind_targets()
    assert env._bind_baseline == {"main": {"/data": "1:10"}}
    assert _bind_targets.decode(json.dumps(env._bind_baseline)) == env._bind_baseline


@pytest.mark.parametrize(
    "selector",
    [
        "-H",
        "--host",
        "--host=tcp://worker:2376",
        "-Hssh://worker",
        "--context",
        "--context=worker",
        "-c",
        "-cworker",
        "--tls",
    ],
)
def test_all_endpoint_flag_spellings_are_unproved(tmp_path, selector):
    """CONF-8: the native filesystem proof accepts only fixed config selection."""
    assert not compose_env.daemon_filesystem_local(
        {"HOME": str(tmp_path)}, [selector, "worker"]
    )
    assert compose_env.daemon_filesystem_local({"HOME": str(tmp_path)}, [])


@pytest.mark.parametrize("kind", ["missing", "malformed", "descendant", "escaped"])
def test_nonrecursive_mount_topology_must_be_complete(tmp_path, kind):
    """CONF-7: unavailable mountinfo and octal-escaped mountpaths fail closed."""
    root = tmp_path / ("work with space" if kind == "escaped" else "work")
    root.mkdir()
    (tmp_path / "export/data").mkdir(parents=True)
    check = _ProjectCheck(str(tmp_path), (str(tmp_path),))
    if kind in ("missing", "malformed"):
        check.mountpoints = lambda: None
    else:
        encoded = (
            str(root / "sub").replace(" ", r"\040")
            if kind == "escaped"
            else str(root / "sub")
        )
        check.mountpoints = lambda: compose_env._mountpoints(
            f"1 0 0:1 / {encoded} rw - tmpfs dummy rw\n"
        )
    assert check.check(_project(root, tmp_path / "export/data", "volume", "bind"))


@pytest.mark.parametrize("service", ["main", "sidecar"])
@pytest.mark.parametrize("fault", ["identity", "missing", "new"])
async def test_every_service_and_nested_target_is_pinned(tmp_path, service, fault):
    """6dfad9e5: all bind targets, including nested and sidecar mounts, must match."""
    env = make_env(tmp_path)
    first, second = "a" * 64, "b" * 64
    mounts = [
        {"Type": "bind", "Destination": p} for p in ("/workspace", "/workspace/nested")
    ]
    data = {
        c: {
            "Config": {"Labels": {"com.docker.compose.service": s}},
            "Mounts": list(mounts),
        }
        for c, s in ((first, "main"), (second, "sidecar"))
    }
    env._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stderr="", return_code=0, stdout=f"{first}\n{second}\n")
    )
    env._inspect_container = AsyncMock(side_effect=lambda c: data[c])
    changed = False

    def cli(args, **kwargs):
        values = ["1:10", "1:11"]
        if changed and (first if service == "main" else second) in args:
            if fault == "identity":
                values[-1] = "1:99"
            elif fault == "missing":
                values.pop()
            else:
                values.append("1:12")
        return ExecResult(stderr="", return_code=0, stdout="\n".join(values))

    env._docker_cli = AsyncMock(side_effect=cli)
    await env._pin_or_verify_bind_targets()
    changed = True
    if fault == "missing":
        data[first if service == "main" else second]["Mounts"].pop()
    elif fault == "new":
        data[first if service == "main" else second]["Mounts"].append(
            {"Type": "bind", "Destination": "/another"}
        )
    with pytest.raises(RuntimeError, match="identity changed"):
        await env._pin_or_verify_bind_targets()


async def test_bind_proof_failure_precedes_all_setup_exec(tmp_path, monkeypatch):
    """6dfad9e5: a failed receipt must prevent the first setup/agent exec."""
    env = make_env(tmp_path)
    env._pin_or_verify_bind_targets = AsyncMock(
        side_effect=RuntimeError("bind identity changed")
    )
    for name in (
        "_prepare_inert_bootstrap",
        "_release_inert_bootstrap",
        "_run_docker_compose_command",
        "_run_pre_compose_hook",
        "_run_docker_compose_build",
        "_start_from_layers",
        "_choose_subnets",
        "_name_trial_volumes",
        "_run_docker_compose_up",
        "_note_task_image_pull",
        "_learn_main_container",
        "_prepare_log_dirs_after_up",
    ):
        monkeypatch.setattr(env, name, AsyncMock())
    env._absent_task_image = AsyncMock(return_value=None)
    from benchflow.sandbox._docker_sweep import release_project

    try:
        with pytest.raises(RuntimeError, match="identity changed"):
            await env.start(force_build=False)
    finally:
        release_project(env.compose_project_name)
    env._prepare_log_dirs_after_up.assert_not_awaited()


@pytest.mark.parametrize("scheme", ["ssh", "tcp"])
async def test_remote_stat_uses_the_daemon_transport(tmp_path, monkeypatch, scheme):
    """CONF-8: the actual Docker exec transports the trusted stat, with no client stat."""
    from benchflow.sandbox import _bind_targets
    from benchflow.sandbox.remote_docker import RemoteDockerHost, RemoteDockerSandbox

    (tmp_path / "Dockerfile").write_text("FROM trusted")
    env = RemoteDockerSandbox(
        tmp_path,
        "proof",
        "proof",
        None,
        SandboxConfig(),
        docker_host=RemoteDockerHost(url=f"{scheme}://worker:2376"),
    )
    seen = []

    class Process:
        returncode = 0

        async def communicate(self):
            return b"7:19\n", b""

    async def execute(*args, **kwargs):
        seen.append((args, kwargs["env"]))
        return Process()

    monkeypatch.setattr("asyncio.create_subprocess_exec", execute)
    targets = await _bind_targets.inspect_targets(
        env, "a" * 64, {"Mounts": [{"Type": "bind", "Destination": "/data"}]}
    )
    assert targets == {"/data": "7:19"}
    args, transport = seen[0]
    assert args[:4] == ("docker", "exec", "-u", "0")
    # Clear loader controls before env itself starts, not only before stat.
    container_index = args.index("a" * 64)
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "BASH_ENV", "ENV"):
        assert f"{name}=" in args[4:container_index]
    assert args[-3:-1] == ("/bin/sh", "-c")
    assert args[-1].endswith("stat -L -c %d:%i -- /data")
    assert f"PATH={SYSTEM_PATH}" in args[-1]
    assert transport.get("DOCKER_HOST") is None
    assert "PATH" in transport
    assert env.docker_host.docker_args()[:2] == ["-H", f"{scheme}://worker:2376"]
    assert "DOCKER_CONTEXT" not in transport


@pytest.mark.parametrize(
    "receipt", ["", "null", "[]", "{}", '{"main": []}', '{"main": {"/data": "1:"}}']
)
def test_incomplete_checkpoint_receipt_has_a_remedy(receipt):
    """6dfad9e5: absent, partial and malformed receipts cannot become a new baseline."""
    from benchflow.sandbox import _bind_targets

    with pytest.raises(ValueError, match="fresh checkpoint"):
        _bind_targets.decode(receipt)


@pytest.mark.parametrize("volume", ["", "null", "[]", "[{}]", '[{"Options":"bind"}]'])
async def test_unknown_volume_backing_cannot_be_skipped(tmp_path, volume):
    """CONF-7 lifetime: a missing backing record cannot be treated as a plain volume."""
    from benchflow.sandbox import _bind_targets

    env = make_env(tmp_path)
    env._docker_cli = AsyncMock(
        return_value=ExecResult(stderr="", stdout=volume, return_code=0)
    )
    with pytest.raises(RuntimeError, match="recreate the sandbox"):
        await _bind_targets.inspect_targets(
            env,
            "a" * 64,
            {"Mounts": [{"Type": "volume", "Name": "work", "Destination": "/data"}]},
        )


@pytest.mark.parametrize(
    "fault", ["truncated", "dropped", "missing-count", "wrong-count"]
)
async def test_vm_mount_inventory_must_include_every_record(tmp_path, fault):
    """CONF-7: a partial mount inventory cannot prove a nonrecursive bind safe."""
    root = tmp_path / "work"
    (root / "sub").mkdir(parents=True)
    (tmp_path / "export/data").mkdir(parents=True)
    project = _project(root, tmp_path / "export/data", "volume", "bind")
    rows = [
        "1 1 0 0:1 / / rw - ext4 dummy rw",
        f"2 2 1 0:2 / {root}/sub rw - tmpfs dummy rw",
        "complete 2",
    ]
    if fault == "truncated":
        rows = rows[:1]
    elif fault == "dropped":
        rows.pop(1)
    elif fault == "missing-count":
        rows.pop()
    else:
        rows[-1] = "complete 1"

    async def run(command):
        if command == "cat /proc/self/mountinfo":
            # The old transport provides no way to distinguish this prefix
            # from a full inventory without any descendant mounts.
            return ExecResult(
                stderr="", return_code=0, stdout="1 0 0:1 / / rw - ext4 dummy rw\n"
            )
        if command.startswith("awk ") and command.endswith("/proc/self/mountinfo"):
            return ExecResult(stderr="", return_code=0, stdout="\n".join(rows))
        argv = shlex.split(command)
        if "config" in argv:
            return ExecResult(stderr="", return_code=0, stdout=json.dumps(project))
        if argv[:2] == ["sh", "-c"]:
            out = subprocess.run(argv, capture_output=True, text=True)
            return ExecResult(stderr="", return_code=out.returncode, stdout=out.stdout)
        return ExecResult(stderr="", return_code=0, stdout="x$$y")

    assert await check_compose_in_vm(
        run, ["docker", "compose"], str(tmp_path), [str(tmp_path)]
    ), "an incomplete inventory was accepted as complete"


async def test_stopped_host_mounted_service_is_not_omitted(tmp_path):
    """6dfad9e5 lifetime: exited init services cannot disappear from the bind proof."""
    env = make_env(tmp_path)
    main, init = "a" * 64, "b" * 64

    def compose(args, **kwargs):
        return ExecResult(
            stderr="",
            return_code=0,
            stdout=f"{main}\n{init}\n" if "--all" in args else main,
        )

    env._run_docker_compose_command = AsyncMock(side_effect=compose)
    env._inspect_container = AsyncMock(
        side_effect=lambda cid: {
            "Config": {
                "Labels": {
                    "com.docker.compose.service": "main" if cid == main else "init"
                }
            },
            "Mounts": [] if cid == main else [{"Type": "bind", "Destination": "/data"}],
        }
    )
    env._docker_cli = AsyncMock(
        return_value=ExecResult(
            stderr="container is not running", stdout="", return_code=1
        )
    )
    with pytest.raises(RuntimeError, match="bind identities"):
        await env._pin_or_verify_bind_targets()
