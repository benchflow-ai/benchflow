"""Guards SV4-B69-C1-1 at 639043af: identity is the build's own invocation."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import DockerLayers, LayerError
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task.config import SandboxConfig
from tests.test_verifier_layers import isolated_layers  # noqa: F401


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    paths = {}
    for name in ("docker", "compose", "buildx", "other-compose"):
        path = binary / name
        path.write_text(f"#!/bin/sh\n# trusted dummy {name} executable\n")
        path.chmod(0o755)
        paths[name] = str(path)
    # Explicit in-memory certification, never executable-shape heuristics.
    monkeypatch.setattr(
        "benchflow.sandbox._compose_build._CERTIFIED_TOOLS",
        {
            name: {
                hashlib.sha256(Path(paths[part]).read_bytes()).hexdigest(): version
                for part in parts
            }
            for name, version, parts in (
                ("docker", "29.1.3", ["docker"]),
                ("compose", "2.35.1", ["compose", "other-compose"]),
                ("buildx", "0.30.1", ["buildx"]),
            )
        },
    )
    state = {
        "version": "v2.35.1",
        "plugin": paths["compose"],
        "builder": {
            "Name": "default",
            "Driver": "docker",
            "Nodes": [{"Endpoint": "default", "Buildkit": "v0.26.2"}],
        },
        "context": [
            {
                "Name": "default",
                "Endpoints": {"docker": {"Host": "unix:///tmp/dummy.sock"}},
            }
        ],
        "daemon": "dummy-daemon",
        "server": "29.1.3",
        "client": "29.1.3",
        "buildx_version": "v0.30.1",
        "flags": [],
        "calls": [],
        "raw": [],
    }
    env = {
        "PATH": str(binary),
        "HOME": str(tmp_path / "home"),
        "DOCKER_CONFIG": str(tmp_path / "task-config"),
        "COMPOSE_BAKE": "false",
        "DOCKER_BUILDKIT": "1",
    }

    async def process(*argv, **kwargs):
        state["calls"].append((argv, kwargs))
        if "info" in argv:
            value = {
                "ID": state["daemon"],
                "OSType": "linux",
                "Architecture": "x86_64",
                "ServerVersion": state["server"],
                "ClientInfo": {
                    "Version": state["client"],
                    "Plugins": [
                        {
                            "Name": "compose",
                            "Version": state["version"],
                            "Path": state["plugin"],
                        },
                        {
                            "Name": "buildx",
                            "Version": state["buildx_version"],
                            "Path": paths["buildx"],
                        },
                    ],
                },
            }
        elif "buildx" in argv:
            builder = state["builder"]
            node = builder["Nodes"][0]
            value = f"Name: {builder['Name']}\nDriver: {builder['Driver']}\n\nNodes:\nName: default\nEndpoint: {node['Endpoint']}\nStatus: running\nBuildKit version: {node['Buildkit']}\n"
        else:
            value = state["context"]

        async def communicate():
            return (
                value.encode() if isinstance(value, str) else json.dumps(value).encode()
            ), b""

        return SimpleNamespace(returncode=0, communicate=communicate)

    async def raw(*args, **kwargs):
        # Deliberately unchanged client-only metadata: the reviewed defect.
        state["raw"].append(args)
        return SimpleNamespace(code=0, stdout="raw-client-tool-version")

    monkeypatch.setattr("asyncio.create_subprocess_exec", process)
    monkeypatch.setattr(DockerLayers, "_docker", raw)

    def box(backend=DockerSandbox):
        instance = backend.__new__(backend)
        DockerSandbox.__init__(
            instance, context, "resolved", "resolved", None, SandboxConfig()
        )
        instance.docker_host = SimpleNamespace(redact=lambda text, **kwargs: text)
        monkeypatch.setattr(
            instance,
            "_docker_compose_env",
            lambda: {**instance._env_vars.to_env_dict(include_os_env=False), **env},
        )
        monkeypatch.setattr(instance, "_compose_transport_env", lambda: {})
        monkeypatch.setattr(
            instance, "_docker_compose_global_args", lambda: list(state["flags"])
        )
        monkeypatch.setattr(instance, "_docker_layers", lambda: DockerLayers(env={}))
        return instance

    return box, state, env, paths


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
async def test_task_selected_plugin_upgrade_cannot_keep_cache_identity(
    toolchain, backend
):
    """Same raw client and task env, upgraded task plugin: no stale hit (639043af)."""
    create, state, _, _ = toolchain
    box = create(backend)
    first = await box._verifier_build_identity(DockerLayers(env={}))
    state["version"] = "v5.0.0"
    try:
        second = await box._verifier_build_identity(DockerLayers(env={}))
    except LayerError:
        return  # Uncertified Compose must build without cache.
    assert second != first, (
        "task-selected plugin changed, but the cache identity did not"
    )


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize(
    "part",
    [
        "plugin",
        "plugin-bytes",
        "docker-bytes",
        "buildx-bytes",
        "client",
        "server",
        "buildx_version",
        "daemon",
        "builder",
        "context",
        "flags",
        "env",
        "compose-file",
        "cwd",
    ],
)
async def test_every_resolved_invocation_input_changes_identity(
    toolchain, backend, part, tmp_path, monkeypatch
):
    create, state, env, paths = toolchain
    box = create(backend)
    first = await box._verifier_build_identity(DockerLayers(env={}))
    if part == "plugin":
        state["plugin"] = paths["other-compose"]
    elif part.endswith("-bytes"):
        name = {
            "plugin-bytes": "compose",
            "docker-bytes": "docker",
            "buildx-bytes": "buildx",
        }[part]
        with open(paths[name], "a") as stream:
            stream.write("# changed trusted executable\n")
    elif part in {"client", "server", "buildx_version", "daemon"}:
        state[part] += "-changed"
    elif part == "builder":
        state["builder"]["Nodes"][0]["Buildkit"] = "v0.27.0"
    elif part == "context":
        state["context"][0]["Endpoints"]["docker"]["Host"] = "unix:///tmp/another.sock"
    elif part == "flags":
        state["flags"] = ["--host", "unix:///tmp/another.sock"]
    elif part == "env":
        env["FUTURE_COMPOSE_INPUT"] = "opaque-value"
    elif part == "compose-file":
        file = tmp_path / "extra.yaml"
        file.write_text("services: {}\n")
        monkeypatch.setattr(
            type(box), "_docker_compose_paths", property(lambda self: [file])
        )
    elif part == "cwd":
        folder = tmp_path / "cwd"
        folder.mkdir()
        monkeypatch.chdir(folder)
    if part.endswith("-bytes") or part in {"client", "buildx_version"}:
        # A new native build requires its own certification, even if its
        # version string is unchanged. Ordinary execution stays available.
        with pytest.raises(LayerError):
            await box._verifier_build_identity(DockerLayers(env={}))
        return
    second = await box._verifier_build_identity(DockerLayers(env={}))
    assert second != first
    assert not state["raw"], "there must be no client-only identity probe"
    for _, kwargs in state["calls"]:
        assert kwargs["env"]["DOCKER_CONFIG"] == env["DOCKER_CONFIG"]


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
async def test_build_consumes_exact_frozen_record_without_resolving_again(
    toolchain, backend, monkeypatch
):
    create, state, env, paths = toolchain
    box = create(backend)
    await box._verifier_build_identity(DockerLayers(env={}))
    invocation = box._resolved_context_build
    calls = len(state["calls"])
    env["DOCKER_CONFIG"] = "/tmp/changed-after-resolution"
    state["flags"] = ["--host", "unix:///tmp/changed.sock"]
    executions = []

    async def execute(argv, **kwargs):
        executions.append((argv, kwargs))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(box, "_run_docker_process", execute)
    await box._run_docker_compose_build(invocation=invocation)
    assert len(state["calls"]) == calls
    argv, kwargs = executions[0]
    assert tuple(argv) == invocation.argv
    assert argv[0] == paths["compose"]
    assert kwargs["environment"] == dict(invocation.environment)
    assert kwargs["cwd"] == invocation.cwd
    assert (
        kwargs["environment"]["DOCKER_CLI_PLUGIN_ORIGINAL_CLI_COMMAND"]
        == paths["docker"]
    )
    for field, value in [
        ("argv", (*invocation.argv, "--changed")),
        ("environment", (*invocation.environment, ("NEW", "VALUE"))),
        ("cwd", "/tmp/another"),
        ("inputs", invocation.inputs + " "),
    ]:
        assert replace(invocation, **{field: value}).identity() != invocation.identity()


@pytest.mark.parametrize(
    "uncertain",
    [
        "version",
        "builder",
        "context",
        "plugin",
        "config",
        "dotenv",
        "env-files",
        "source-policy",
        "unknown-flag",
    ],
)
async def test_uncertain_resolution_still_executes_ordinary_build(
    toolchain, uncertain, tmp_path, monkeypatch
):
    create, state, env, _ = toolchain
    box = create()
    if uncertain == "version":
        state["version"] = "v5.0.0"
    elif uncertain == "builder":
        state["builder"]["Driver"] = "docker-container"
    elif uncertain == "context":
        state["context"] = []
    elif uncertain == "plugin":
        state["plugin"] = "/tmp/missing-plugin"
    elif uncertain == "config":
        root = tmp_path / "task-config"
        root.mkdir()
        (root / "config.json").write_text("dummy marker; never read")
    elif uncertain == "dotenv":
        (tmp_path / ".env").write_text("DUMMY_MARKER=true")
    elif uncertain == "env-files":
        env["COMPOSE_ENV_FILES"] = ""
    elif uncertain == "source-policy":
        env["EXPERIMENTAL_BUILDKIT_SOURCE_POLICY"] = ""
    else:
        state["flags"] = ["--config=/tmp/config"]
    with pytest.raises(LayerError):
        await box._verifier_build_identity(DockerLayers(env={}))
    assert box._resolved_context_build.uncertainty
    executed = []

    async def execute(argv, **kwargs):
        executed.append((argv, kwargs))

    monkeypatch.setattr(box, "_run_docker_process", execute)
    await box._run_docker_compose_build()
    assert len(executed) == 1
    assert executed[0][0][-1] == "build"


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
async def test_unknown_plugin_refuses_cache_before_lookup(
    toolchain, backend, monkeypatch, tmp_path
):
    from tests.test_verifier_layers import _recipe

    create, state, _, _ = toolchain
    box = create(backend)
    (box.environment_dir / "Dockerfile").write_text(
        "FROM python@sha256:" + "1" * 64 + "\n"
    )
    box.use_verifier_layer(_recipe(tmp_path))
    state["version"] = "v5.0.0"
    lookups = []

    async def forbidden(*args, **kwargs):
        lookups.append(args)
        raise AssertionError("uncertain invocation reached image cache")

    monkeypatch.setattr(DockerLayers, "_ensure", forbidden)

    async def wrapper_platform(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "benchflow.sandbox.verifier_layers._require_cache_platform", wrapper_platform
    )
    assert not await box._start_verifier_from_context()
    assert not lookups
    assert "Compose version" in box.layer_records["verifier"]["reason"]


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize(
    "change",
    [
        "version",
        "plugin",
        "flags",
        "environment",
        "builder-context",
        "docker-wrapper",
        "compose-wrapper",
        "buildx-wrapper",
    ],
)
async def test_resolved_invocation_transition_cannot_consume_warm_entry(
    toolchain, backend, change, monkeypatch, tmp_path
):
    """Exercise real key/lookup/callback code across miss, hit and changed input."""
    from benchflow.sandbox._docker_subnets import daemon_key
    from tests.test_verifier_layers import BakeDocker, _recipe

    create, state, env, paths = toolchain
    box = create(backend)
    ref = "python@sha256:" + "1" * 64
    (box.environment_dir / "Dockerfile").write_text(
        f"FROM {ref}\nWORKDIR /opt/verifier-data\n"
    )
    fake = BakeDocker()
    fake.add_image(ref)
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    monkeypatch.setattr(box, "_docker_subnet_key", lambda: daemon_key({}))
    recipe = _recipe(tmp_path)
    builds = []

    async def build(*, invocation=None):
        if invocation is None:
            assert box._resolved_context_build.uncertainty
        else:
            assert invocation == box._resolved_context_build
        builds.append(box._resolved_context_build)
        fake.add_image(
            box._env_vars.main_image_name,
            labels={
                "com.docker.compose.project": box.compose_project_name,
                "com.docker.compose.service": "main",
            },
        )

    monkeypatch.setattr(box, "_run_docker_compose_build", build)
    for status in ("miss", "hit"):
        box.use_verifier_layer(recipe)
        assert await box._start_verifier_from_context()
        if status == "hit":
            assert builds[0].environment == box._resolved_context_build.environment
            assert builds[0].argv == box._resolved_context_build.argv
            assert builds[0].inputs == box._resolved_context_build.inputs
        assert box.verifier_layer.result.status == status
        box._env_vars.prebuilt_image_name = None
        box._use_prebuilt = False
    assert len(builds) == 1
    if change == "version":
        state["version"] = "v5.0.0"
    elif change == "plugin":
        state["plugin"] = paths["other-compose"]
    elif change == "flags":
        state["flags"] = ["--context=default"]
    elif change == "environment":
        env["FUTURE_BUILD_INPUT"] = "changed"
    elif change.endswith("-wrapper"):
        Path(paths[change.removesuffix("-wrapper")]).write_text(
            "#!/bin/sh\nexport COMPOSE_BAKE=true\n# trusted operator wrapper\n"
        )
    else:
        state["builder"]["Nodes"][0]["Endpoint"] = "another-context"
    box.use_verifier_layer(recipe)
    used = await box._start_verifier_from_context()
    if change == "version" or change.endswith("-wrapper"):
        assert not used
        assert len(builds) == 1
        await box._run_docker_compose_build()  # ordinary fallback still builds
    else:
        assert used
        assert box.verifier_layer.result.status == "miss"
    assert len(builds) == 2


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize(
    "flags",
    [
        ["--host", "unix:///tmp/daemon.sock"],
        ["-H", "unix:///tmp/daemon.sock"],
        ["--host=unix:///tmp/daemon.sock"],
        ["--context", "default"],
        ["-c", "default"],
        ["--context=default"],
    ],
)
async def test_all_supported_global_flag_forms_reach_resolution_and_execution(
    toolchain, backend, flags
):
    create, state, _, _ = toolchain
    state["flags"] = flags
    box = create(backend)
    await box._verifier_build_identity(DockerLayers(env={}))
    assert list(box._resolved_context_build.argv[1 : 1 + len(flags)]) == flags
    assert all(list(argv[1 : 1 + len(flags)]) == flags for argv, _ in state["calls"])


@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
@pytest.mark.parametrize(
    "flags",
    [
        ["--config", "/tmp/client"],
        ["--config=/tmp/client"],
        ["--tls"],
        ["--tlsverify=true"],
        ["-Hunix:///tmp/daemon.sock"],
        ["-cdefault"],
        ["--future-option"],
        ["--host"],
        ["--context"],
    ],
)
async def test_every_unresolved_global_flag_form_disables_cache(
    toolchain, backend, flags
):
    create, state, _, _ = toolchain
    state["flags"] = flags
    box = create(backend)
    with pytest.raises(LayerError):
        await box._verifier_build_identity(DockerLayers(env={}))
    assert box._resolved_context_build.uncertainty


@pytest.mark.parametrize(
    "context",
    [
        None,
        [],
        [{}],
        [{"Name": "default"}],
        [{"Name": "default", "Endpoints": {"docker": {}}}],
        [{"Name": "default", "Endpoints": {"docker": {"Host": ""}}}],
    ],
)
async def test_incomplete_context_metadata_disables_cache(toolchain, context):
    create, state, _, _ = toolchain
    state["context"] = context
    box = create()
    with pytest.raises(LayerError):
        await box._verifier_build_identity(DockerLayers(env={}))
