"""A task's Compose project is checked once and launched as checked.

The ``test_r3_*`` tests reproduce a review's findings, numbered 1
to 10: each is the review's input through the real gate, the sandbox a
run constructs (``Environment.from_task``, ``_create_sandbox_environment``)
or a rollout's staging, and each fails on 08d979ef by what it does. Every
value standing for an operator's secret is a dummy marker.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.test_compose_oracle import (
    _compose_available,
    _compose_task,
    _dind,
    _dind_refused,
    _gate,
    _refused,
)

pytestmark = pytest.mark.skipif(
    not _compose_available(),
    reason="the gate runs `docker compose config`, which needs docker compose",
)

ALLOW = "BENCHFLOW_ALLOW_TASK_ENV"
HOST_ACCESS = "BENCHFLOW_ALLOW_TASK_HOST_ACCESS"
MARKER = "s1-r3-dummy-marker"


@pytest.fixture(autouse=True)
def _operator(monkeypatch, tmp_path):
    """The operator names nothing, has no .env, has not allowed host access,
    and runs from a scratch folder (Environment.from_task writes jobs/)."""
    monkeypatch.setenv(ALLOW, "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.delenv(HOST_ACCESS, raising=False)
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)


def _from_task(task: Path, sandbox: str = "docker"):
    from benchflow.runtime import Environment

    return Environment.from_task(task, sandbox=sandbox)


def _from_task_refused(task: Path, sandbox: str = "docker") -> str:
    from benchflow.task.env import TaskEnvNotAllowedError

    with pytest.raises(TaskEnvNotAllowedError) as raised:
        _from_task(task, sandbox)
    return " ".join(str(raised.value).split())


def _launched(sandbox) -> dict:
    """What `docker compose config` gives for exactly the files, folder,
    flags and environment the sandbox's Compose calls use: what its build,
    up and down run on."""
    command = [
        "docker",
        *sandbox._docker_compose_global_args(),
        "compose",
        "--project-name",
        sandbox.compose_project_name,
        "--project-directory",
        str(getattr(sandbox, "compose_project_dir", sandbox.environment_dir)),
    ]
    for path in sandbox._docker_compose_paths:
        command += ["-f", str(path)]
    launch = getattr(sandbox, "_launch", None)
    env = dict(launch.env) if launch is not None else sandbox._docker_compose_env()
    out = subprocess.run(
        [*command, "config", "--format", "json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(out.stdout)


# A key Compose and PyYAML read differently


@pytest.mark.parametrize(
    "compose",
    [
        # The review's input: `include` as a !!binary key.
        "? !!binary aW5jbHVkZQ==\n: [{{path: child.yaml, env_file: {env}}}]\n"
        "services:\n  main: {{}}\n",
        # extends as a !!binary key, to a file outside the task.
        "services:\n  main:\n    image: alpine:3.20\n"
        "    ? !!binary ZXh0ZW5kcw==\n    : {{file: {base}, service: side}}\n",
    ],
    ids=["binary-include", "binary-extends"],
)
def test_r3_1_a_key_compose_reads_differently_is_refused(
    monkeypatch, tmp_path, compose
):
    operator = tmp_path / "operator"
    operator.mkdir()
    (operator / "s1-operator.env").write_text(f"S1_FILE_TOKEN={MARKER}\n")
    (operator / "base.yaml").write_text(
        f"services:\n  side:\n    environment: {{LEAK: {MARKER}}}\n"
    )
    task = _compose_task(
        tmp_path / "task",
        compose.format(env=operator / "s1-operator.env", base=operator / "base.yaml"),
        files={
            "child.yaml": "services:\n  side:\n    image: alpine:3.20\n"
            '    environment: {LEAK: "${S1_FILE_TOKEN}"}\n'
        },
    )
    message = _refused(task)
    assert "has a key that is not text" in message
    assert MARKER not in message
    # Nothing clears it: the file must be written so both parsers agree.
    monkeypatch.setenv(HOST_ACCESS, "1")
    assert "has a key that is not text" in _refused(task)


def test_r3_1_every_merged_value_of_a_dependency_key_is_followed(tmp_path):
    """A ``<<`` merge with two sources: the reading follows the env_file of
    both, whichever go-yaml lets win."""
    outside = tmp_path / "s1-operator.env"
    outside.write_text(f"S1_FILE_TOKEN={MARKER}\n")
    task = _compose_task(
        tmp_path / "task",
        "x-a: &a {env_file: own.env}\n"
        f"x-b: &b {{env_file: {outside}}}\n"
        "services:\n  main:\n    image: alpine:3.20\n    <<: [*a, *b]\n",
        files={"own.env": "A=1\n"},
    )
    assert f"reads the env file {outside}" in _refused(task)


# The gate resolves different launch inputs


def test_r3_2_an_alias_gets_the_value_the_run_gives(monkeypatch, tmp_path):
    """Input A: ``NM = "${S1_MODE}"`` with S1_MODE named and set to host is
    NM=host for Compose, at the gate as in the run."""
    monkeypatch.setenv(ALLOW, "S1_MODE")
    monkeypatch.setenv("S1_MODE", "host")
    monkeypatch.delenv("NM", raising=False)
    task = _compose_task(
        tmp_path / "task",
        'services:\n  main:\n    image: alpine:3.20\n    network_mode: "${NM:-bridge}"\n',
        '[environment.env]\nNM = "${S1_MODE}"\n',
    )
    assert "gives service main network_mode: host" in _refused(task)
    assert "network_mode: host" in _from_task_refused(task)


def test_r3_2_the_sandbox_checks_its_own_image_name_and_files(tmp_path):
    """Input B: a link named after the run's image (``bf__<task>``) to /,
    mounted through ${MAIN_IMAGE_NAME}: the run never builds from a folder
    holding a link out of it."""
    task = _compose_task(
        tmp_path / "s1-r3-task",
        'services:\n  main:\n    volumes: ["./${MAIN_IMAGE_NAME}:/host:ro"]\n',
    )
    (task / "environment" / "bf__compose-check").mkdir()
    (task / "environment" / "bf__s1-r3-task").symlink_to("/")
    message = _from_task_refused(task)
    assert "environment/bf__s1-r3-task is a link to /" in message


def test_r3_2_what_the_sandbox_checked_is_what_it_mounts(tmp_path):
    """Without the link, the sandbox checks and launches the folder named
    after its own image, in its own copy of the task. The image's name is
    the sandbox's private one (private_image_repository, f08fed68e), not
    ``bf__<task>``: the folder is the one its launch names."""
    task = _compose_task(
        tmp_path / "s1-r3-task",
        'services:\n  main:\n    volumes: ["./${MAIN_IMAGE_NAME}:/host:ro"]\n',
    )
    (task / "environment" / "bf__s1-r3-task").mkdir()
    sandbox = _from_task(task).inner
    launched = _launched(sandbox)
    sources = [v["source"] for v in launched["services"]["main"]["volumes"]]
    copy = sandbox.environment_dir
    image = dict(sandbox._launch.env)["MAIN_IMAGE_NAME"]
    assert image != "bf__s1-r3-task"
    assert str(copy / image) in sources
    assert not copy.is_relative_to(task)
    assert launched["name"] == sandbox.compose_project_name


def test_r3_2_force_build_freezes_the_build_files(monkeypatch, tmp_path):
    """A prebuilt task's sandbox froze the prebuilt files; start with
    force_build checks and freezes the build files instead."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    environment: {A: b}\n",
        '[environment]\ndocker_image = "alpine:3.20"\n',
    )
    sandbox = _from_task(task).inner
    prebuilt = sandbox._launch
    sandbox.freeze_launch(force_build=True)
    assert sandbox._launch.key != prebuilt.key
    assert "build" in _launched(sandbox)["services"]["main"]


# Compose prints a $ doubled


def test_r3_3_a_doubled_dollar_is_the_single_dollar_path_the_run_mounts(tmp_path):
    """The review's input through Environment.from_task: ``leak$dir`` (a
    link to /) is what Compose mounts for ``./leak$$dir``; the copy refuses
    the link before anything is checked."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes: ['./leak$$dir:/host:ro']\n",
    )
    (task / "environment" / "leak$$dir").mkdir()
    (task / "environment" / "leak$dir").symlink_to("/")
    assert "environment/leak$dir is a link to /" in _from_task_refused(task)


def test_r3_3_the_check_reads_the_value_not_its_printed_escape(tmp_path):
    from benchflow.task.compose_env import decode_dollars

    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes: ['./leak$$dir:/host:ro']\n"
        "    environment: {A: 'x$$y'}\n",
    )
    (task / "environment" / "leak$dir").mkdir()
    sandbox = _from_task(task).inner
    printed = json.loads(sandbox._launch.path.read_text())
    main = decode_dollars(printed)["services"]["main"]
    assert main["environment"]["A"] == "x$y"
    host = next(v for v in main["volumes"] if v["target"] == "/host")
    assert host["source"] == str(sandbox.environment_dir / "leak$dir")
    # Compose reads the launch file back to the very same values.
    assert _launched(sandbox)["services"]["main"]["environment"]["A"] == "x$$y"


async def test_r3_3_the_vm_check_reads_the_value_not_its_printed_escape(tmp_path):
    """Daytona DinD: a ``leak$dir`` link the task's hook makes in the VM is
    what Compose mounts for ``./leak$$dir``; the VM check follows it."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes: ['./leak$$dir:/host:ro']\n",
    )
    (task / "environment" / "leak$$dir").mkdir()
    strategy, vm, _ = _dind(tmp_path, task)
    (vm / "task" / "environment" / "leak$dir").symlink_to("/")
    message = await _dind_refused(strategy)
    assert "mounts /benchflow/task/environment/leak$dir for service main" in message


# A bind mount written as a tmpfs volume

_TMPFS_BIND = (
    "services:\n  main:\n    volumes: [host:/host:ro]\nvolumes:\n  host:\n"
    "    driver: local\n    driver_opts:\n      type: tmpfs\n      o: {o}\n"
    "      device: {device}\n"
)


@pytest.mark.parametrize("o", ["bind", "rbind,ro", "size=1m,bind"])
def test_r3_4_o_bind_is_a_host_path_whatever_the_type(tmp_path, o):
    task = _compose_task(tmp_path / "task", _TMPFS_BIND.format(o=o, device="/"))
    assert "binds / for volume host, outside the task's folder" in _refused(task)
    assert "binds / for volume host, a path on the remote Docker host" in _refused(
        task, "remote-docker"
    )


def test_r3_4_a_plain_tmpfs_volume_and_a_bind_inside_pass(tmp_path):
    plain = _compose_task(
        tmp_path / "plain",
        "services:\n  main:\n    volumes: [scratch:/s]\nvolumes:\n  scratch:\n"
        "    driver_opts: {type: tmpfs, o: 'size=1m', device: tmpfs}\n",
    )
    _gate(plain)
    inside = _compose_task(
        tmp_path / "inside", _TMPFS_BIND.format(o="bind", device="./data")
    )
    (inside / "environment" / "data").mkdir()
    _gate(inside)


@pytest.mark.parametrize(
    ("options", "shown"),
    [
        (
            "{type: ext4, device: /dev/sda1}",
            "mounts volume host as ext4 from /dev/sda1",
        ),
        ("{type: overlay, o: 'lowerdir=/etc', device: overlay}", "as overlay"),
    ],
    ids=["block-device", "overlay"],
)
def test_r3_4_a_local_filesystem_volume_is_refused(tmp_path, options, shown):
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes: [host:/h]\nvolumes:\n  host:\n"
        f"    driver_opts: {options}\n",
    )
    assert shown in _refused(task)


def test_r3_4_a_volume_plugin_is_refused(tmp_path):
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes: [host:/h]\nvolumes:\n  host:\n"
        "    driver: s1-local-persist\n    driver_opts: {mountpoint: /}\n",
    )
    assert "uses the volume driver s1-local-persist for volume host" in _refused(task)


# A volume the project does not own


@pytest.mark.parametrize(
    ("volumes", "shown"),
    [
        (
            "loot:\n    external: true\n    name: s1-operator-data\n",
            "uses the existing volume s1-operator-data (external) for volume loot",
        ),
        (
            "loot:\n    name: s1-operator-data\n",
            "names volume loot s1-operator-data, which reuses a volume",
        ),
    ],
    ids=["external", "explicit-name"],
)
def test_r3_5_a_volume_the_project_does_not_own_is_refused(
    monkeypatch, tmp_path, volumes, shown
):
    task = _compose_task(
        tmp_path / "task",
        f"services:\n  main:\n    volumes: [loot:/loot:ro]\nvolumes:\n  {volumes}",
    )
    assert shown in _refused(task)
    assert shown in _refused(task, "remote-docker")
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


def test_r3_5_a_network_the_project_does_not_own_is_refused(tmp_path):
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    networks: [ops]\nnetworks:\n"
        "  ops:\n    external: true\n    name: s1-operator-net\n",
    )
    assert "joins the existing network s1-operator-net (external)" in _refused(task)


def test_r3_5_the_projects_own_volumes_pass(tmp_path):
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    volumes: [data:/d, other:/o]\n"
        "volumes:\n  data: {}\n  other: {driver: local}\n",
    )
    _gate(task)


# The docker client configuration's proxies


def test_r3_6_compose_never_gets_the_operators_proxies(monkeypatch, tmp_path):
    config = tmp_path / "operator-docker"
    (config / "contexts").mkdir(parents=True)
    # The operator's own plugins, where docker finds them for this config
    # (a per-user Compose plugin lives in ~/.docker/cli-plugins): the
    # hermetic config must still find them, through cliPluginsExtraDirs.
    plugins = Path.home() / ".docker" / "cli-plugins"
    if plugins.is_dir():
        (config / "cli-plugins").symlink_to(plugins)
    (config / "config.json").write_text(
        json.dumps(
            {
                "auths": {"registry.invalid": {"auth": "czE6ZHVtbXk="}},
                "proxies": {
                    "default": {
                        "httpsProxy": f"http://review-user:{MARKER}@proxy.invalid:3128"
                    }
                },
            }
        )
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(config))
    task = _compose_task(tmp_path / "task", "services:\n  main: {}\n")
    sandbox = _from_task(task).inner

    def check(flags: list[str] | tuple[str, ...]) -> None:
        used = Path(flags[flags.index("--config") + 1])
        text = (used / "config.json").read_text()
        assert MARKER not in text and "proxies" not in text
        assert json.loads(text)["auths"] == {
            "registry.invalid": {"auth": "czE6ZHVtbXk="}
        }
        assert (used / "contexts").resolve() == (config / "contexts").resolve()

    check(sandbox._docker_compose_global_args())
    check(sandbox._launch.docker_args)


# The files are reopened at launch


def test_r3_7_a_task_rewritten_after_the_check_is_not_what_launches(tmp_path):
    """After a sandbox is checked, an external writer changes the original
    task. The sandbox still launches its frozen private copy. The copy is
    mounted read-only because its build inputs must survive agent execution.
    """
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    image: alpine:3.20\n    volumes: ['.:/task-files:ro']\n",
    )
    sandbox = _from_task(task).inner
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: alpine:3.20\n    privileged: true\n"
        "    volumes: ['/:/host:ro']\n"
    )
    main = _launched(sandbox)["services"]["main"]
    assert "privileged" not in main
    assert [v["target"] for v in main["volumes"] if v["type"] == "bind"].count(
        "/host"
    ) == 0
    # The task's own mount of its folder is the sandbox's private copy.
    task_files = next(v for v in main["volumes"] if v["target"] == "/task-files")
    assert Path(task_files["source"]) == sandbox.environment_dir


def test_r3_7_the_frozen_project_is_outside_what_a_task_may_mount(tmp_path):
    task = _compose_task(tmp_path / "task", "services:\n  main: {}\n")
    sandbox = _from_task(task).inner
    launch = sandbox._launch
    assert not launch.path.is_relative_to(sandbox.environment_dir)
    assert oct(launch.project_dir.stat().st_mode & 0o777) == "0o700"
    assert sorted(p.name for p in launch.project_dir.iterdir()) == ["compose.json"]


async def test_r3_7_start_builds_and_runs_only_the_frozen_project(
    monkeypatch, tmp_path
):
    """Every Compose call start() makes names the frozen file and folder."""
    task = _compose_task(tmp_path / "task", "services:\n  main: {}\n")
    sandbox = _from_task(task).inner
    calls: list[list[str]] = []

    async def fake_exec(*argv, **kwargs):
        calls.append(list(argv))
        raise RuntimeError("s1-stop-here")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    privileged: true\n"
    )
    with pytest.raises(RuntimeError, match="s1-stop-here"):
        await sandbox.start(force_build=False)
    compose = [c for c in calls if "compose" in c]
    assert compose
    for argv in compose:
        files = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
        assert not any(Path(f).is_relative_to(task) for f in files)
        assert files == [str(sandbox._launch.path)]
        assert argv[argv.index("--project-directory") + 1] == str(
            sandbox._launch.project_dir
        )


# Staging follows links


async def test_r3_8_a_rollouts_task_copy_never_reads_through_a_link(tmp_path):
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task.env import TaskEnvNotAllowedError

    secret = tmp_path / "s1-operator-secret"
    secret.write_text(f"{MARKER}\n")
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM alpine:3.20\nCOPY payload /p\n"
    )
    (task / "environment" / "payload").symlink_to(secret)
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "instruction.md").write_text("Do it.\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 60\n\n[verifier]\ntimeout_sec = 60\n'
    )
    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            agent="oracle",
            environment="docker",
            base_image_override="alpine:3.20",
            jobs_dir=tmp_path / "jobs",
        )
    )
    made: list[Path] = []
    rollout._planes.create_environment = lambda *a, **k: made.append(a[2])
    with pytest.raises(TaskEnvNotAllowedError) as raised:
        await rollout.setup()
    assert "environment/payload is a link to" in str(raised.value)
    assert made == []
    tmp = getattr(rollout, "_task_tmp", None)
    copies = list(Path(tmp).rglob("payload")) if tmp else []
    assert all(MARKER not in p.read_text() for p in copies if p.is_file())


def test_r3_8_the_copy_keeps_links_inside_and_refuses_special_files(tmp_path):
    from benchflow.task.staging import StagingRefused, copy_tree

    source = tmp_path / "src"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "real.txt").write_text("r\n")
    (source / "link").symlink_to("sub/real.txt")
    (source / "abs").symlink_to(source / "sub")
    (source / "run.sh").write_text("#!/bin/sh\n")
    (source / "run.sh").chmod(0o755)
    copy_tree(source, tmp_path / "dst")
    dst = tmp_path / "dst"
    assert os.readlink(dst / "link") == "sub/real.txt"
    assert os.readlink(dst / "abs") == "sub"
    assert (dst / "run.sh").stat().st_mode & 0o111
    os.mkfifo(source / "pipe")
    with pytest.raises(StagingRefused) as raised:
        copy_tree(source, tmp_path / "dst2", label="t")
    assert "t/pipe is a FIFO, not a file" in str(raised.value)


# Models


def test_r3_9_models_are_refused_unless_host_access(monkeypatch, tmp_path):
    task = _compose_task(
        tmp_path / "task",
        "models:\n  m: {model: ai/smollm2}\nservices:\n  main:\n"
        "    image: alpine:3.20\n    models: [m]\n",
    )
    message = _refused(task)
    assert "declares models, which Compose pulls and runs" in message
    assert "gives service main models" in message
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


def test_r3_9_a_key_benchflow_does_not_check_is_refused(tmp_path):
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    image: alpine:3.20\n    s1_future_key: on\n",
    )
    assert "gives service main the key s1_future_key" in _refused(task)


# Daytona refuses only after starting a sandbox


def test_r3_10_daytona_refuses_before_the_sandbox_is_created(monkeypatch, tmp_path):
    from benchflow.sandbox.daytona import DaytonaSandbox

    created: list[object] = []

    async def create(self, *args, **kwargs):
        created.append(self)

    monkeypatch.setattr(DaytonaSandbox, "_create_sandbox", create)
    task = _compose_task(
        tmp_path / "task", "services:\n  main:\n    privileged: true\n"
    )
    message = _from_task_refused(task, "daytona")
    assert "runs service main privileged, which reaches the sandbox's VM" in message
    assert created == []


def test_r3_10_without_compose_here_a_setting_through_a_variable_is_refused(
    monkeypatch, tmp_path
):
    from benchflow.task import compose_env

    monkeypatch.setattr(
        compose_env, "_compose_cli_available", lambda: False, raising=False
    )
    literal = _compose_task(
        tmp_path / "literal", "services:\n  main:\n    privileged: true\n"
    )
    assert "runs service main privileged" in _refused(literal, "daytona")
    variable = _compose_task(
        tmp_path / "variable",
        "services:\n  main:\n    privileged: ${S1_PRIV:-true}\n",
    )
    assert "sets privileged of service main through a variable" in _refused(
        variable, "daytona"
    )
    plain = _compose_task(tmp_path / "plain", "services:\n  main: {}\n")
    _gate(plain, "daytona")


# The frozen launch


def test_profiles_the_task_turns_on_are_launched(tmp_path):
    """A task's .env turning a profile on: the checked project has that
    service, and so does the launch, which reads no .env."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main: {}\n  extra:\n    image: alpine:3.20\n"
        "    profiles: [extra]\n",
        files={".env": "COMPOSE_PROFILES=extra\n"},
    )
    sandbox = _from_task(task).inner
    assert "extra" in _launched(sandbox)["services"]


def test_the_hook_runs_on_the_copy_and_is_checked_after(monkeypatch, tmp_path):
    monkeypatch.setenv(HOST_ACCESS, "1")
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main: {}\n",
        files={
            "benchflow-pre-compose.sh": "printf 'services:\\n  main:\\n"
            '    environment: {HOOKED: "1"}\\n\' > docker-compose.yaml\n'
        },
    )
    sandbox = _from_task(task).inner
    assert sandbox.environment_dir != task / "environment"
    import asyncio

    asyncio.run(sandbox._run_pre_compose_hook())
    sandbox._launch = None  # start() does the same once a hook ran
    sandbox.freeze_launch()
    assert _launched(sandbox)["services"]["main"]["environment"] == {"HOOKED": "1"}
    assert "HOOKED" not in (task / "environment" / "docker-compose.yaml").read_text()
