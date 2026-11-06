"""Guards SV4-B74-C1-1 (adc970f6) against integ 4fdadd52's build contract.

The ordinary tests compare the actual subprocess boundary with the original
launcher's argv/environment and batch 17's stream/process isolation. The native
dummy tools reproduce a trusted wrapper's effective setting.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import DockerLayers, LayerError
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task.config import SandboxConfig

BACKENDS = [DockerSandbox, RemoteDockerSandbox]
FLAGS = [
    [],
    ["--host", "unix:///tmp/dummy.sock"],
    ["-H", "unix:///tmp/dummy.sock"],
    ["--host=unix:///tmp/dummy.sock"],
    ["-Hunix:///tmp/dummy.sock"],
    ["--context", "default"],
    ["-c", "default"],
    ["--context=default"],
    ["-cdefault"],
    ["--config", "/tmp/dummy-config"],
    ["--config=/tmp/dummy-config"],
    ["--tls"],
    ["--tlsverify=true"],
    ["--future-option"],
]
LAUNCHERS = ["plain", "shell", "python", "elf", "symlink-shell", "symlink-plain"]


def sandbox(backend, root, monkeypatch, env, flags):
    (root / "Dockerfile").write_text("FROM scratch\n")
    box = backend.__new__(backend)
    DockerSandbox.__init__(box, root, "launcher", "launcher", None, SandboxConfig())
    box._compose_project = "sv4-launcher-contract"
    box.docker_host = SimpleNamespace(redact=lambda text, **kwargs: text)
    env.update(box._env_vars.to_env_dict(include_os_env=False))
    monkeypatch.setattr(box, "_docker_compose_env", lambda: dict(env))
    monkeypatch.setattr(box, "_compose_transport_env", lambda: {"TRANSPORT": "dummy"})
    monkeypatch.setattr(box, "_docker_compose_global_args", lambda: list(flags))

    async def identify():
        return SimpleNamespace(server_id="dummy-daemon")

    monkeypatch.setattr(
        box, "_docker_subnet_key", lambda: SimpleNamespace(identify=identify)
    )
    return box


def integration_argv(box, flags):
    # Launcher argv from 4fdadd52 remains unchanged after c51c88dc's stream
    # separation and cancellation-safe process-group handling (batch 17).
    result = [
        "docker",
        *flags,
        "compose",
        "--project-name",
        box.compose_project_name,
        "--project-directory",
        str(box.environment_dir.resolve().absolute()),
    ]
    for path in box._docker_compose_paths:
        result.extend(["-f", str(path.resolve().absolute())])
    return [*result, "build"]


@pytest.fixture
def dummy_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = {}
    for name in ("docker", "compose", "buildx"):
        path = tmp_path / name
        path.write_text(f"#!/bin/sh\n# dummy native {name}\n")
        path.chmod(0o755)
        paths[name] = path
    env = {
        "PATH": str(tmp_path),
        "HOME": str(tmp_path / "home"),
        "COMPOSE_BAKE": "false",
        "DOCKER_BUILDKIT": "1",
        "SOURCE_DATE_EPOCH": "1700000000",
    }
    info = {
        "ID": "dummy-daemon",
        "OSType": "linux",
        "Architecture": "amd64",
        "ServerVersion": "29.1.3",
        "ClientInfo": {
            "Version": "29.1.3",
            "Plugins": [
                {"Name": name, "Path": str(paths[name]), "Version": version}
                for name, version in [("compose", "v2.35.1"), ("buildx", "v0.30.1")]
            ],
        },
    }
    context = [
        {"Name": "default", "Endpoints": {"docker": {"Host": "unix:///tmp/dummy.sock"}}}
    ]
    calls = []

    async def execute(*argv, **kwargs):
        calls.append((argv, kwargs))
        if "info" in argv:
            output = json.dumps(info)
        elif "buildx" in argv:
            output = "Name: default\nDriver: docker\n\nNodes:\nEndpoint: default\nStatus: running\nBuildKit version: v0.26.2\n"
        elif "context" in argv:
            output = json.dumps(context)
        else:
            output = "build completed"

        async def communicate():
            return output.encode(), b""

        return SimpleNamespace(returncode=0, communicate=communicate)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", execute)
    return paths, env, info, context, calls


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("entry", ["command", "retry-helper", "task-image"])
async def test_ordinary_build_is_byte_identical_to_integration(
    dummy_tools, tmp_path, monkeypatch, backend, flags, launcher, entry
):
    paths, env, _, _, calls = dummy_tools
    if launcher in {"shell", "symlink-shell"}:
        paths["docker"].write_text(
            '#!/bin/sh\nexport COMPOSE_BAKE=true\nexec /tmp/native-docker "$@"\n'
        )
    elif launcher == "python":
        paths["docker"].write_text("#!/usr/bin/python3\n# trusted operator wrapper\n")
    elif launcher == "elf":
        paths["docker"].write_bytes(b"\x7fELF\x02dummy compiled operator wrapper")
    if launcher.startswith("symlink"):
        target = tmp_path / "launcher-target"
        paths["docker"].rename(target)
        paths["docker"].symlink_to(target)
    box = sandbox(backend, tmp_path, monkeypatch, env, flags)
    if entry == "command":
        await box._run_docker_compose_command(["build"])
    elif entry == "retry-helper":
        await box._run_docker_compose_build()
    else:
        # Integration inlines the ordinary build in start(). Exercise its
        # same helper on archives predating this branch's capture primitive.
        build = getattr(box, "_build_task_image", box._run_docker_compose_build)
        await build()
    assert len(calls) == 1, "ordinary grading must not resolve plugins"
    argv, options = calls[0]
    assert list(argv) == integration_argv(box, flags)
    assert options.pop("cwd", None) is None
    assert options == {
        "env": {**env, "TRANSPORT": "dummy"},
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "start_new_session": os.name == "posix",
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cache_attempt", [False, True])
@pytest.mark.parametrize(
    "flags", [[], ["--context", "default"], ["-Hunix:///tmp/dummy.sock"]]
)
async def test_trusted_wrapper_build_setting_survives(
    tmp_path, monkeypatch, backend, cache_attempt, flags
):
    monkeypatch.chdir(tmp_path)
    native = tmp_path / "native-docker"
    compose, buildx = tmp_path / "compose", tmp_path / "buildx"
    info = {
        "ID": "dummy",
        "OSType": "linux",
        "Architecture": "amd64",
        "ServerVersion": "29.1.3",
        "ClientInfo": {
            "Version": "29.1.3",
            "Plugins": [
                {"Name": "compose", "Version": "v2.35.1", "Path": str(compose)},
                {"Name": "buildx", "Version": "v0.30.1", "Path": str(buildx)},
            ],
        },
    }
    script = (
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"info = {info!r}\n"
        "if 'info' in sys.argv: print(json.dumps(info))\n"
        "elif 'context' in sys.argv: print(json.dumps([{'Name':'default', 'Endpoints':{'docker':{'Host':'unix:///tmp/dummy.sock'}}}]))\n"
        "else: print(json.dumps({'bake':os.environ['COMPOSE_BAKE'], 'epoch':os.environ['SOURCE_DATE_EPOCH']}))\n"
    )
    native.write_text(script)
    compose.write_text(script)
    buildx.write_text(
        f"#!{sys.executable}\nprint('Name: default\\nDriver: docker\\n\\nNodes:\\nEndpoint: default\\nStatus: running\\nBuildKit version: v0.26.2')\n"
    )
    wrapper = tmp_path / "docker"
    wrapper.write_text(f'#!/bin/sh\nexport COMPOSE_BAKE=true\nexec "{native}" "$@"\n')
    for path in (wrapper, native, compose, buildx):
        path.chmod(0o755)
    env = {
        "PATH": str(tmp_path) + os.pathsep + os.defpath,
        "HOME": str(tmp_path / "home"),
        "COMPOSE_BAKE": "false",
        "DOCKER_BUILDKIT": "1",
        "SOURCE_DATE_EPOCH": "1700000000",
    }
    box = sandbox(backend, tmp_path, monkeypatch, env, flags)
    if cache_attempt and hasattr(box, "_verifier_build_identity"):
        with contextlib.suppress(LayerError):
            await box._verifier_build_identity(DockerLayers(env={}))
    result = await box._run_docker_compose_command(["build"])
    assert json.loads(result.stdout) == {"bake": "true", "epoch": "1700000000"}


def certify_dummy_tools(monkeypatch, paths):
    from benchflow.sandbox import _compose_build

    monkeypatch.setattr(
        _compose_build,
        "_CERTIFIED_TOOLS",
        {
            name: {hashlib.sha256(path.read_bytes()).hexdigest(): version}
            for name, path, version in [
                (name, paths[name], version)
                for name, version in [
                    ("docker", "29.1.3"),
                    ("compose", "2.35.1"),
                    ("buildx", "0.30.1"),
                ]
            ]
        },
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("symlink", [False, True])
async def test_plain_resolved_cache_build_consumes_its_own_record(
    dummy_tools, tmp_path, monkeypatch, backend, symlink
):
    paths, env, _, _, calls = dummy_tools
    certify_dummy_tools(monkeypatch, paths)
    if symlink:
        for path in paths.values():
            target = path.with_name(path.name + "-native")
            path.rename(target)
            path.symlink_to(target)
    box = sandbox(backend, tmp_path, monkeypatch, env, ["--context=default"])
    await box._verifier_build_identity(DockerLayers(env={}))
    record = box._resolved_context_build
    assert record.uncertainty is None
    assert record.argv[0] == str(paths["compose"].resolve())
    assert dict(record.environment)["DOCKER_CLI_PLUGIN_ORIGINAL_CLI_COMMAND"] == str(
        paths["docker"].resolve()
    )
    calls.clear()
    env["SOURCE_DATE_EPOCH"] = "1800000000"
    await box._run_docker_compose_build(invocation=record)
    assert len(calls) == 1
    assert calls[0][0] == record.argv
    assert calls[0][1]["env"] == dict(record.environment)
    assert calls[0][1]["cwd"] == record.cwd
    # Even an admitted saved record cannot affect a later ordinary call.
    calls.clear()
    await box._run_docker_compose_build()
    assert list(calls[0][0]) == integration_argv(box, ["--context=default"])
    assert calls[0][1]["env"] == {**env, "TRANSPORT": "dummy"}
    assert calls[0][1]["cwd"] is None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("role", ["docker", "compose", "buildx"])
@pytest.mark.parametrize(
    "shape", ["shell", "python", "elf", "symlink", "unknown-native"]
)
async def test_any_uncertified_launcher_or_plugin_refuses_cache(
    dummy_tools, tmp_path, monkeypatch, backend, role, shape
):
    paths, env, _, _, calls = dummy_tools
    certify_dummy_tools(monkeypatch, paths)
    path = paths[role]
    path.write_bytes(
        {
            "shell": b"#!/bin/sh\nexport COMPOSE_BAKE=true\n",
            "python": b"#!/usr/bin/python3\n# wrapper\n",
            "elf": b"\x7fELFdummy wrapper",
            "symlink": b"# wrapper target",
            "unknown-native": b"\x7fELFunknown distributor build",
        }[shape]
    )
    if shape == "symlink":
        target = tmp_path / f"{role}-target"
        path.rename(target)
        path.symlink_to(target)
    box = sandbox(backend, tmp_path, monkeypatch, env, [])
    with pytest.raises(LayerError):
        await box._verifier_build_identity(DockerLayers(env={}))
    calls.clear()
    await box._run_docker_compose_build()
    assert calls[0][0][0] == "docker"
    assert calls[0][1]["env"] == {**env, "TRANSPORT": "dummy"}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "uncertain",
    [
        "plugin",
        "version",
        "context",
        "config",
        "dotenv",
        "flags",
        "inputs",
        "BUILDX_CONFIG",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_VERIFY",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DOCKER_HOST",
        "BUILDKIT_HOST",
        "context-tcp",
        "context-tls",
        "context-skip-tls",
        "host-tcp",
    ],
)
async def test_uncertain_cache_resolution_falls_back_exactly(
    dummy_tools, tmp_path, monkeypatch, backend, uncertain
):
    paths, env, info, context, calls = dummy_tools
    certify_dummy_tools(monkeypatch, paths)
    flags = []
    if uncertain == "plugin":
        info["ClientInfo"]["Plugins"] = []
    elif uncertain == "version":
        info["ClientInfo"]["Plugins"][0]["Version"] = "v5.0.0"
    elif uncertain == "context":
        context.clear()
    elif uncertain == "config":
        config = tmp_path / "config"
        config.mkdir()
        (config / "config.json").write_text("dummy marker, never parse")
        env["DOCKER_CONFIG"] = str(config)
    elif uncertain == "dotenv":
        (tmp_path / ".env").write_text("DUMMY_MARKER=true")
    elif uncertain == "flags":
        flags = ["--config=/tmp/dummy"]
    elif uncertain == "host-tcp":
        flags = ["--host=tcp://dummy:2375"]
    elif uncertain == "context-tcp":
        context[0]["Endpoints"]["docker"]["Host"] = "tcp://dummy:2375"
    elif uncertain == "context-tls":
        context[0]["TLSMaterial"] = {"docker": ["dummy-ca"]}
    elif uncertain == "context-skip-tls":
        context[0]["Endpoints"]["docker"]["SkipTLSVerify"] = True
    elif uncertain in {"DOCKER_HOST", "BUILDKIT_HOST"}:
        env[uncertain] = "tcp://dummy:2375"
    elif uncertain in {
        "BUILDX_CONFIG",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_VERIFY",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
    }:
        env[uncertain] = "dummy"
    else:
        env["COMPOSE_ENV_FILES"] = ""
    box = sandbox(backend, tmp_path, monkeypatch, env, flags)
    with pytest.raises(LayerError):
        await box._verifier_build_identity(DockerLayers(env={}))
    calls.clear()
    await box._run_docker_compose_build()
    assert len(calls) == 1
    assert list(calls[0][0]) == integration_argv(box, flags)
    assert calls[0][1]["env"] == {**env, "TRANSPORT": "dummy"}
    assert calls[0][1]["cwd"] is None
