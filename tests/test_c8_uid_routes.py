"""Guard C8's routes by effective uid and the backends' single choke point
(PT-C8-R2-1 to PT-C8-R2-5 at 2b4eab29).

As in ``test_c8_loader_routes.py``: every test runs the text or argv a
backend really builds, here, with the environment that backend's process
would start with, whose ``LD_PRELOAD`` names a missing library. glibc names
it on stderr for every program that starts with it ("cannot be preloaded"),
so BenchFlow's plumbing must never print it. Programs run as the test's own
user; ``su`` is replayed as the shell it would start.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shlex
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox._base import ExecResult

MARK = "/nonexistent/app/lib/bf-agent-profile.so"
IMAGE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _image_env(**extra: str) -> dict[str, str]:
    return {"PATH": IMAGE_PATH, "HOME": "/nonexistent", "LD_PRELOAD": MARK, **extra}


def _run(argv, env) -> ExecResult:
    ran = subprocess.run(
        list(argv), env=dict(env), capture_output=True, text=True, timeout=60
    )
    return ExecResult(return_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr)


def _loaded(result: ExecResult) -> bool:
    return MARK in (result.stderr or "")


def _docker_exec_env(argv: list[str], image_env: dict[str, str]):
    """``[docker] exec [flags] <container> <cmd...>``: the env and the command."""
    env = dict(image_env)
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-e", "--env"):
            name, _, value = argv[i + 1].partition("=")
            env[name] = value
            i += 2
        elif arg in ("-u", "-w", "--user", "--workdir"):
            i += 2
        elif arg.startswith("-"):
            i += 1
        else:
            return env, argv[i + 1 :]
    raise AssertionError(f"no container in {argv}")


# The image's users: ``toor`` is a second name for uid 0, ``app`` is not.
UIDS = {"toor": "0", "app": "1000"}


def _lookup(command: str) -> ExecResult | None:
    """Answer exec_route's ``id -u <name>`` as the image would."""
    for name, uid in UIDS.items():
        if f"id -u {name}" in command:
            return ExecResult(return_code=0, stdout=f"{uid}\n")
    if re.search(r"id -u [a-z]", command):
        return ExecResult(return_code=1, stderr="id: no such user\n")
    return None


def _docker_sandbox(image_env, lookups: list[str] | None = None):
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = "ctr"
    sandbox.logger = logging.getLogger("test")

    async def run(argv, check=False, timeout_sec=None, **kwargs):
        env, command = _docker_exec_env(argv, image_env)
        answer = _lookup(command[-1])
        if answer is not None:
            if lookups is not None:
                lookups.append(command[-1])
            return answer
        return _run(command, env)

    sandbox._run_docker_container_command = run
    return sandbox


# PT-C8-R2-2: a root exec spelled as another uid-0 user is root's plumbing.


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["toor", "00", "0:0", 0])
async def test_docker_uid_zero_spellings_are_root_plumbing(user):
    sandbox = _docker_sandbox(_image_env())
    result = await sandbox.exec("cat /dev/null && printf ok", user=user)
    assert result.stdout == "ok", result.stderr
    assert not _loaded(result), result.stderr


@pytest.mark.asyncio
async def test_the_verifier_uid_probe_as_a_uid_zero_alias_loads_nothing():
    """The finding's own path: hardening's ``id -u`` as ``[verifier] user``."""
    from benchflow.sandbox import lockdown

    sandbox = _docker_sandbox(_image_env())
    seen: list[ExecResult] = []
    run = sandbox._run_docker_container_command

    async def record(argv, check=False, timeout_sec=None, **kwargs):
        result = await run(argv, check=check, timeout_sec=timeout_sec, **kwargs)
        seen.append(result)
        return result

    sandbox._run_docker_container_command = record
    uid = await lockdown._verifier_uid(sandbox, "toor")
    assert uid is not None and uid.isdigit()
    assert seen and not any(_loaded(r) for r in seen), [r.stderr for r in seen]


@pytest.mark.asyncio
async def test_a_non_root_named_user_keeps_its_own_environment():
    """By uid both ways: a uid-1000 user's command keeps the image's value."""
    sandbox = _docker_sandbox(_image_env())
    result = await sandbox.exec('printf %s "$LD_PRELOAD"', user="app")
    assert result.stdout == MARK


@pytest.mark.asyncio
async def test_a_name_is_looked_up_once_per_generation_and_unknown_is_root():
    lookups: list[str] = []
    sandbox = _docker_sandbox(_image_env(), lookups)
    for _ in range(3):
        await sandbox.exec("true", user="app")
    assert len(lookups) == 1
    sandbox._forget_kept_identity()  # a start, stop or restore
    await sandbox.exec("true", user="app")
    assert len(lookups) == 2
    # A name that does not resolve is routed as root and not kept.
    for _ in range(2):
        result = await sandbox.exec("cat /dev/null && printf ok", user="ghost")
        assert not _loaded(result), result.stderr
    assert len(lookups) == 4


# PT-C8-R2-3: Daytona DinD's file probes go through the route like any exec.


def _dind_sandbox(image_env):
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    sandbox = object.__new__(DaytonaSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox.logger = logging.getLogger("test")
    strategy = object.__new__(_DaytonaDinD)
    strategy._env = sandbox
    results: list[ExecResult] = []

    async def compose_exec(parts, timeout_sec=None, cleanup_session=False):
        env, command = _docker_exec_env(parts, image_env)
        result = _run(command, env)
        results.append(result)
        return result

    strategy._compose_exec = compose_exec
    sandbox._strategy = strategy
    return sandbox, results


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", ["is_file", "is_dir"])
async def test_dind_file_probes_start_no_program_with_the_image_preload(probe):
    sandbox, results = _dind_sandbox(_image_env())
    assert await getattr(sandbox, probe)("/" if probe == "is_dir" else "/etc/passwd")
    assert results and not any(_loaded(r) for r in results), [r.stderr for r in results]


def test_every_lowest_exec_takes_a_route_with_no_default():
    """By construction: what hands a command to the provider cannot be
    called without a route (a probe or clean-up that bypasses ``exec``)."""
    from benchflow.sandbox.agentcore import AgentCoreSandbox
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.daytona_dind import _DaytonaDinD
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect, _DaytonaStrategy

    lowest = [
        DaytonaSandbox._sandbox_exec,
        AgentCoreSandbox._dispatch_command,
    ]
    for cls in (_DaytonaStrategy, _DaytonaDirect, _DaytonaDinD):
        lowest += [cls.exec, cls.exec_transient, cls.is_file, cls.is_dir]
    lowest.append(_DaytonaDinD._exec)
    for fn in lowest:
        param = inspect.signature(fn).parameters.get("route")
        assert param is not None, fn.__qualname__
        assert param.default is inspect.Parameter.empty, fn.__qualname__
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, fn.__qualname__


# AgentCore: the runtime starts each command's first shell with the image's
# ENV as BenchFlow's image sets it (values aside, names empty).


def _dockerfile_env(text: str, image_env: dict[str, str]) -> dict[str, str]:
    env = dict(image_env)
    for line in text.splitlines():
        if not line.startswith("ENV "):
            continue
        before = dict(env)
        for pair in shlex.split(line[4:]):
            name, _, value = pair.partition("=")
            env[name] = re.sub(
                r"\$\{(\w+)\}", lambda m, b=before: b.get(m.group(1), ""), value
            )
    return env


def _generated_dockerfile() -> str:
    from benchflow.sandbox.agentcore_image import AgentCoreImagePublisher

    publisher = object.__new__(AgentCoreImagePublisher)
    publisher._task_env_config = SimpleNamespace(docker_image="python:3.12-slim")
    return publisher.generated_dockerfile_text()


_SU = re.compile(r"/bin/su \S+ -s (\S+) -c ")


def _agentcore_sandbox(monkeypatch, tmp_path, *, persistent=None):
    """An AgentCore sandbox whose runtime runs each payload here with
    ``/bin/sh`` (dash on Debian and Ubuntu, as in ``python:3.12-slim``)."""
    from benchflow.sandbox import agentcore_sealed
    from benchflow.sandbox.agentcore import AgentCoreSandbox
    from benchflow.sandbox.agentcore_sealed import SealedChannel

    monkeypatch.setattr(agentcore_sealed, "SEAL_DIR", str(tmp_path / "seal"))
    runtime_env = _dockerfile_env(_generated_dockerfile(), _image_env())
    sandbox = object.__new__(AgentCoreSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = dict(persistent or {})
    sandbox.runtime_arn = "arn"
    sandbox.runtime_session_id = "s"
    sandbox.logger = logging.getLogger("test")
    results: list[ExecResult] = []

    def invoke(payload, timeout):
        answer = _lookup(payload)
        if answer is not None:
            return answer
        result = _run(["/bin/sh", "-c", _SU.sub(r"\1 -c ", payload)], runtime_env)
        results.append(result)
        return result

    monkeypatch.setattr(sandbox, "_invoke_command", invoke)
    sandbox._sealed = SealedChannel(sandbox._exec_raw, sandbox.logger)
    return sandbox, results


def _need_dash_and_openssl():
    if not Path("/bin/bash").exists():
        pytest.skip("needs /bin/bash, as AgentCore images do")
    if subprocess.run(["sh", "-c", "command -v openssl"]).returncode != 0:
        pytest.skip("the sealed channel needs openssl")


# PT-C8-R2-4: sealed uploads and env staging use Bash syntax; /bin/sh can be
# dash. Honest tasks must still upload files and get their env.


@pytest.mark.asyncio
async def test_agentcore_root_plumbing_runs_bash_syntax(monkeypatch, tmp_path):
    _need_dash_and_openssl()
    sandbox, _ = _agentcore_sandbox(monkeypatch, tmp_path)
    result = await sandbox._exec_raw("set -o pipefail; printf ok", user="root")
    assert (result.return_code, result.stdout) == (0, "ok"), result.stderr


@pytest.mark.asyncio
async def test_agentcore_sealed_upload_and_env_staging_work(monkeypatch, tmp_path):
    _need_dash_and_openssl()
    sandbox, results = _agentcore_sandbox(monkeypatch, tmp_path)
    target = tmp_path / "up" / "file.txt"
    await sandbox._sealed.upload(b"payload\n", target=str(target), mode="600")
    assert target.read_bytes() == b"payload\n"
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    staged = await sandbox.exec('printf %s "$TASK_FLAG"', env={"TASK_FLAG": "v 1"})
    assert (staged.return_code, staged.stdout) == (0, "v 1"), staged.stderr
    assert not any(_loaded(r) for r in results), [r.stderr for r in results]


# PT-C8-R2-1: the task route's env file holds the task's loader values; the
# inline removal after sourcing it is BenchFlow's own root step.


@pytest.mark.asyncio
async def test_agentcore_task_route_removes_its_env_file_without_the_preload(
    monkeypatch, tmp_path
):
    _need_dash_and_openssl()
    from benchflow.sandbox._task_route import as_task_command

    sandbox, results = _agentcore_sandbox(
        monkeypatch, tmp_path, persistent={"LD_PRELOAD": MARK}
    )
    staged: list[str] = []
    stage_env = sandbox._sealed.stage_env

    async def record(env, owner=None):
        staged.append(await stage_env(env, owner=owner))
        return staged[-1]

    sandbox._sealed.stage_env = record
    # As the image's user (root): test.sh with the default verifier user.
    with as_task_command(sandbox):
        task = await sandbox.exec('printf %s "${LD_PRELOAD-unset}"')
    assert task.stdout == MARK, task.stderr
    assert not any(_loaded(r) for r in results), [r.stderr for r in results]
    assert staged and not Path(staged[0]).exists()


# PT-C8-R2-5: the ping shim starts with the image's loader values (a Python
# that needs LD_LIBRARY_PATH), though every runtime shell starts without.


def test_agentcore_ping_shim_gets_the_image_loader_values(tmp_path):
    text = _generated_dockerfile()
    cmd = next(line for line in text.splitlines() if line.startswith("CMD "))
    argv = json.loads(cmd[4:])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    out = tmp_path / "seen"
    fake = bin_dir / "python3"
    fake.write_text(
        f'#!/bin/sh\nprintf "%s|%s" "$LD_LIBRARY_PATH" "$*" > {shlex.quote(str(out))}\n'
    )
    fake.chmod(0o755)
    image = {
        "PATH": f"{bin_dir}:{IMAGE_PATH}",
        "HOME": "/nonexistent",
        "LD_LIBRARY_PATH": "/opt/python/lib",
    }
    runtime_env = _dockerfile_env(text, image)
    assert runtime_env["LD_LIBRARY_PATH"] == ""
    ran = _run(argv, runtime_env)
    assert ran.return_code == 0, ran.stderr
    assert out.read_text() == "/opt/python/lib|/opt/benchflow_agentcore_shim.py"


# FF-install-C16..C20 (at 8af3461b), as C8 on this
# branch: agent install, proxy preparation, JS setup and the Daytona cache
# run on root's plumbing route, and the runtimes they reuse must be root's
# alone. The tests run as an unprivileged user, so a tree here is never
# root's: they show what is refused; the container repro shows an honest,
# root-owned runtime passing.


def _planted_image(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """An image PATH and BASH_ENV leading into a folder an earlier role owns."""
    planted = tmp_path / "app-bin"
    planted.mkdir()
    for tool in ("sh", "bash", "id", "ls", "find", "rm", "mkdir", "cat"):
        exe = planted / tool
        exe.write_text(f"#!/bin/sh\necho PLANTED-{tool} >&2\n")
        exe.chmod(0o755)
    activate = tmp_path / "activate.sh"
    activate.write_text("echo PLANTED-bash-env >&2\n")
    return planted, {
        "PATH": f"{planted}:/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LD_PRELOAD": MARK,
        "BASH_ENV": str(activate),
        "PYTHONUSERBASE": str(tmp_path / "user-site"),
        "PYTHONWARNINGS": "ignore::agent_mod.Warning",
    }


@pytest.mark.asyncio
async def test_install_and_proxy_setup_execs_take_the_plumbing_route(tmp_path):
    """C16, C17: what install.py and the proxy preparation send (``exec``
    with no user: the image's, root) starts no planted shell, tool, BASH_ENV
    or library, on a PATH that is never the image's."""
    _, image = _planted_image(tmp_path)
    sandbox = _docker_sandbox(image)
    result = await sandbox.exec(
        "/bin/bash -c "
        + shlex.quote(
            "id -u >/dev/null; ls / >/dev/null; "
            'printf "%s|%s" "$PATH" "${PYTHONUSERBASE-unset}${PYTHONWARNINGS-unset}"'
        ),
        timeout_sec=30,
    )
    assert "PLANTED" not in result.stderr, result.stderr
    assert not _loaded(result), result.stderr
    path, python_vars = result.stdout.split("|")
    assert path and "app-bin" not in path
    assert python_vars == "unsetunset"


@pytest.mark.asyncio
async def test_root_plumbing_path_drops_unsafe_local_folders_and_is_never_empty(
    monkeypatch,
):
    """C20: the PATH root's plumbing pins never names a folder the PATH lock
    left open or one in the workspace, and is never empty."""
    from benchflow.sandbox import lockdown

    # The lock's report, as its receipt read gives it back.
    async def lock_report(env, command, **kwargs):
        return ExecResult(return_code=0, stdout="BF unlockable-path /usr/local/bin\n")

    monkeypatch.setattr(lockdown, "_read_exec", lock_report)
    for workspace, expect in (
        ("/app", "/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"),
        ("/usr/local", "/usr/sbin:/usr/bin:/sbin:/bin"),
    ):
        sandbox = _docker_sandbox({"PATH": IMAGE_PATH, "HOME": "/nonexistent"})
        await lockdown.lock_system_path(sandbox, workspace)
        result = await sandbox.exec('printf %s "$PATH"', user="root")
        assert result.stdout == expect, (workspace, result.stdout)


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "opt" / "benchflow"
    (root / "node" / "bin").mkdir(parents=True)
    node = root / "node" / "bin" / "node"
    node.write_text("#!/bin/sh\necho PLANTED-node >&2\n")
    node.chmod(0o755)
    return root


def test_the_root_only_check_refuses_a_tree_the_agent_user_owns(tmp_path):
    """C18, C19: the shared check refuses a runtime someone else owns or may
    write, and says which."""
    from benchflow._root_only import root_only_tree_check

    root = _tree(tmp_path)
    ran = _run(
        ["/bin/sh", "-c", root_only_tree_check(str(root), "the agent runtime")],
        {"PATH": "/usr/bin:/bin"},
    )
    assert ran.return_code != 0
    assert "is not root's alone" in ran.stderr and str(root) in ran.stderr


def test_the_js_runtime_is_checked_before_root_reuses_or_installs_into_it():
    """C18: every JS agent install (and the native Codex one) starts with the
    check of /opt/benchflow, before root runs its Node or npm."""
    from benchflow._root_only import root_only_tree_check
    from benchflow.agents import registry

    check = root_only_tree_check("/opt/benchflow", "BenchFlow's agent runtime folder")
    assert registry._NODE_INSTALL.startswith(f"mkdir -p /opt/benchflow && {check}")
    node_use = registry._NODE_INSTALL.index('"$BF_NODE_DIR/bin/node"')
    assert registry._NODE_INSTALL.index(check) < node_use


@pytest.mark.asyncio
async def test_a_cached_proxy_venv_the_agent_owns_is_not_reused(tmp_path, monkeypatch):
    """C19: the Daytona cache and the Docker proxy layer accept a venv only
    when it is root's alone, not merely when its key matches."""
    from benchflow.providers import litellm_runtime as lr
    from benchflow.sandbox import daytona_layers as dl
    from benchflow.sandbox.env_layers import LayerResult

    venv = tmp_path / "litellm"
    (venv / "bin").mkdir(parents=True)
    for exe in ("python", "litellm"):
        (venv / "bin" / exe).write_text("#!/bin/sh\necho PLANTED-python >&2\n")
        (venv / "bin" / exe).chmod(0o755)
    keys = tmp_path / "layers"
    keys.mkdir()

    class _Box:
        async def exec(self, command, **kwargs):
            return _run(["/bin/sh", "-c", command], {"PATH": "/usr/bin:/bin"})

    # Daytona's image layer.
    monkeypatch.setattr(dl, "LAYER_KEY_DIR", str(keys))
    monkeypatch.setattr(dl, "PROXY_LAYER_VENV", str(venv))
    layer = dl.InstallLayer(kind="proxy", name="litellm", script="s", venv=str(venv))
    (keys / "proxy-litellm.key").write_text("k1\n")

    class _Daytona(_Box):
        def cached_install_layer(self, kind, name):
            return layer, "k1"

    assert (
        await dl.cached_layer_in_sandbox(_Daytona(), "proxy", "litellm", script="s")
        is None
    )

    # Docker's proxy layer.
    install = lr.proxy_install(None)
    stamp = tmp_path / "stamp"
    stamp.write_text(lr.proxy_layer_recipe(install).stamp + "\n")
    monkeypatch.setattr(lr, "PROXY_LAYER_VENV", str(venv))
    monkeypatch.setattr(lr, "PROXY_LAYER_STAMP", str(stamp))
    box = _Box()
    box.proxy_layer = object.__new__(LayerResult)
    object.__setattr__(box.proxy_layer, "key", "k1")
    python, record = await lr._proxy_from_layer(box, install)
    assert python is None and record["install"] == "in trial", record
