"""C8 round 9: root processes start from nothing, per service.

The class tests enumerate every backend, every service that gets root's
plumbing and every phase of a trial (setup, the kept services, verification,
teardown) and check the route each root exec takes there; a source scan
holds every builder and every raw provider SDK call to that route. Then one
test per finding of the b92 review; the ones that need a real
container are in ``test_c8_protected_runtime_container.py``.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import os
import re
import shlex
import stat
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._root_only import root_only_path_check, root_only_tree_check
from benchflow._runtime_tree import (
    KIT_BIN,
    KIT_SH,
    RUNTIME_ROOT,
    RuntimeKit,
    kit_prepare_command,
    workspace_grant_problem,
)
from benchflow.sandbox._base import (
    CARRY_FUNC,
    SYSTEM_PATH,
    SYSTEM_SHELL,
    ExecResult,
    ExecRoute,
    RuntimeKitBrokenError,
    restore_image_env,
    system_shell_prefix,
    with_system_path,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "benchflow"
# Root's HOME on the kit: a mode-700 folder of the protected tree.
ROOT_HOME = "/opt/runtime/run/home"
KIT_RECORD = RuntimeKit(trusted=("/usr/bin", "/bin"), python=True, bash=True)
KIT_ROUTE = ExecRoute(
    pin=KIT_RECORD.path,
    task_root=False,
    system_shell=True,
    explicit_root=True,
    shell=KIT_SH,
    bash=f"{KIT_BIN}/bash",
    tools=KIT_RECORD.path,
    kit=True,
)
not_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="needs a user other than root to own the test files"
)


def _sh(command: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, **kwargs
    )


# --- The class: every backend, service and phase --------------------------


def _backend(name: str):
    """A sandbox of backend *name* as the trial holds it, built without a
    provider (exec_route needs none)."""
    if name == "docker":
        from benchflow.sandbox.docker import DockerSandbox as cls
    elif name == "remote-docker":
        from benchflow.sandbox.remote_docker import RemoteDockerSandbox as cls
    elif name == "daytona":
        from benchflow.sandbox.daytona import DaytonaSandbox as cls
    elif name == "modal":
        from benchflow.sandbox.modal_impl import ModalSandbox as cls
    elif name == "apple":
        from benchflow.sandbox.apple_container import AppleContainerSandbox as cls
    else:
        from benchflow.sandbox.agentcore import AgentCoreSandbox as cls
    sandbox = object.__new__(cls)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox.logger = logging.getLogger("test")
    return sandbox


BACKENDS = ("docker", "remote-docker", "daytona", "modal", "apple", "agentcore")
# Backends whose exec takes a compose service other than main.
SERVICE_BACKENDS = {"docker", "remote-docker", "daytona"}
PHASES = ("setup", "kept services", "verification", "teardown")


@contextlib.contextmanager
def _phase(sandbox, phase: str):
    """The trial's state in *phase*: before the agent the kit is being made;
    the kept services run beside root; verification runs the task's own
    command; teardown follows a restore, whose kit is read again."""
    if phase == "kept services":
        with sandbox.services_kept():
            yield
    elif phase == "verification":
        with sandbox.services_kept(), sandbox.task_command():
            yield
    else:
        yield


async def _arm(sandbox, phase: str, service_kit: bool):
    """Give *sandbox* the kits the trial has by *phase*."""
    sandbox._runtime_kit_fresh()
    if phase == "setup":
        return
    sandbox.set_runtime_kit(KIT_RECORD)
    if service_kit:
        sandbox.set_runtime_kit(KIT_RECORD, service="db")
    if phase == "teardown":
        # A restore: nothing is known, the first root step reads each kit.
        sandbox._forget_kept_identity()
        sandbox._runtime_kit_unknown()

        async def load(service="main"):
            return KIT_RECORD if service == "main" or service_kit else None

        sandbox._load_runtime_kit = load


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("backend", BACKENDS)
async def test_every_root_exec_in_main_starts_from_the_kit(backend, phase):
    """Every root exec in main, in every phase, on every backend: once the
    kit exists, its shell by absolute path, its PATH, and (plumbing) a
    process that starts from nothing (``env -i``). Before it (setup), the
    system shell by absolute path with the system PATH."""
    sandbox = _backend(backend)
    await _arm(sandbox, phase, service_kit=False)
    with _phase(sandbox, phase):
        for user, system_shell in (("root", True), ("root", False), (None, False)):
            route = await sandbox.exec_route(user, system_shell=system_shell)
            if phase == "setup":
                assert route.shell == SYSTEM_SHELL and not route.kit
                if system_shell:
                    assert route.pin == SYSTEM_PATH
                continue
            assert route.kit and route.shell == KIT_SH, (backend, phase, user)
            assert route.tools.split(":")[0] == KIT_BIN
            if route.pin is None:
                # The task's own command (verification): the image's PATH
                # and values come back only after the kit's shell checked
                # them (restore_image_env with the kit's tools).
                assert phase == "verification" and not system_shell
                continue
            assert route.pin.split(":")[0] == KIT_BIN
            text = with_system_path("true", route.pin)
            # Builtins only before it: bf_carry, which quotes the kept values
            # onto the pipe env -i's shell reads.
            assert text.startswith(f"{CARRY_FUNC}exec {KIT_BIN}/env -i "), text
            assert f"HOME={ROOT_HOME} {KIT_SH} -c " in text


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("backend", sorted(SERVICE_BACKENDS))
async def test_every_root_exec_in_a_service_takes_that_services_own_kit(backend, phase):
    """A service gets root's plumbing only on its
    own kit; without one (once main has its kit) root's steps there are
    refused, while a named non-root user's command there still runs."""
    for has_kit in (True, False):
        sandbox = _backend(backend)
        sandbox._named_uids = (0, {})
        await _arm(sandbox, phase, service_kit=has_kit)
        sandbox._named_uids = (sandbox._exec_generation, {("db", "app"): 1000})
        with _phase(sandbox, phase):
            if phase == "setup":
                route = await sandbox.exec_route("root", service="db")
                assert route.shell == SYSTEM_SHELL and not route.kit
                continue
            if not has_kit:
                for system_shell in (True, False):
                    with pytest.raises(RuntimeKitBrokenError, match="'db'"):
                        await sandbox.exec_route(
                            "root", system_shell=system_shell, service="db"
                        )
                route = await sandbox.exec_route("app", service="db")
                assert not route.root and not route.kit
                continue
            route = await sandbox.exec_route("root", system_shell=True, service="db")
            assert route.kit and route.shell == KIT_SH
            assert route.pin.split(":")[0] == KIT_BIN


# Raw provider SDK calls: each hands a command to the provider, so each
# sits in a function that builds it from a route (or is the route's
# lowest exec), listed with why.
SDK_CALLS = re.compile(
    r"\.process\.exec\(|execute_session_command\(|create_pty_session\(|"
    r"open_shell\(|invoke_agent_runtime_command\(|\.exec\.aio\("
)
SDK_CALLERS = {
    # The command channel adds two calls on its captured protected route.
    ("sandbox/daytona_channel.py", "DaytonaChannel.connect"): (
        "the route's protected Python; terminal_env before the terminal starts"
    ),
    ("sandbox/daytona_channel.py", "DaytonaChannels._control"): (
        "_session_command builds the control command on the captured root route"
    ),
    ("sandbox/daytona.py", "DaytonaSandbox._sandbox_session_exec"): (
        "lowest call: sends the session command _session_command built on the "
        "route _sandbox_exec passes (its payload, if any, as staged stdin)"
    ),
    ("sandbox/daytona.py", "DaytonaSandbox._sync_exec"): (
        "lowest call: sends the command _sandbox_exec built from its route"
    ),
    ("sandbox/modal_impl.py", "ModalSandbox._dispatch_exec"): (
        "the process start, built on the route exec passes (required)"
    ),
    ("sandbox/modal_impl.py", "ModalSandbox._check_bfx"): (
        "lowest call: argv exec of the pinned /.bfx/bfx version from its "
        "read-only Volume (bfx is the route, TRUSTED-RUNTIME I1; no shell, "
        "environment or secrets), before any root step"
    ),
    ("sandbox/daytona.py", "DaytonaSandbox._toolbox_startup_env"): (
        "lowest call: the start probe, builtins only (read, printf, compgen) "
        "in the toolbox's POSIX-mode shell, before any root step"
    ),
    ("sandbox/modal_impl.py", "ModalSandbox._refuse_startup_names"): (
        "lowest call: argv exec of /bin/cat /proc/self/environ by absolute "
        "path (no shell), before any root step"
    ),
    ("sandbox/agentcore.py", "AgentCoreSandbox._invoke_command"): (
        "lowest call; _dispatch_command takes the route"
    ),
    ("sandbox/process/daytona.py", "_cleanup_daytona_remote_env_file"): "_routed",
    ("sandbox/process/daytona.py", "_bootstrap_daytona_script_file"): "_routed",
    ("sandbox/process/daytona.py", "_bootstrap_daytona_env_file"): (
        "_bootstrap_env_command: _routed"
    ),
    ("sandbox/process/daytona.py", "DaytonaPtyProcess.start"): (
        "the route's bash; terminal_env before the terminal reads a file"
    ),
    ("sandbox/process/agentcore.py", "AgentCoreProcess.start"): (
        "require_protected_root_terminal before open_shell"
    ),
    ("sandbox/process/modal.py", "ModalProcess.start"): (
        "root's launch on the route's bash and PATH (launch_route), loader "
        "names empty, the env on stdin"
    ),
}


def _sdk_call_sites():
    found = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        text = path.read_text()
        if not SDK_CALLS.search(text):
            continue
        tree = ast.parse(text)

        def walk(node, prefix, rel=rel, text=text):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    walk(child, f"{prefix}{child.name}.")
                elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    source = ast.get_source_segment(text, child) or ""
                    inner = [
                        n
                        for n in ast.walk(child)
                        if n is not child
                        and isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                    ]
                    own = source
                    for n in inner:
                        own = own.replace(ast.get_source_segment(text, n) or "", "")
                    code = "\n".join(
                        line
                        for line in own.splitlines()
                        if not line.strip().startswith("#")
                    )
                    if SDK_CALLS.search(code):
                        found[(rel, f"{prefix}{child.name}")] = own
                    walk(child, f"{prefix}{child.name}.")

        walk(tree, "")
    return found


def test_every_raw_provider_sdk_call_is_on_a_route():
    """The source-scan invariant covers raw SDK calls too. A
    new call to the provider fails here until it takes a route."""
    found = _sdk_call_sites()
    for key, source in found.items():
        assert key in SDK_CALLERS, (
            f"{key[0]}:{key[1]} hands a command to the provider outside the "
            "route; build it from exec_route (and list it here)"
        )
        if not SDK_CALLERS[key].startswith("lowest call"):
            assert "route" in source or "_routed" in source, key
    stale = set(SDK_CALLERS) - set(found)
    assert not stale, f"listed SDK callers that no longer call the SDK: {stale}"


# --- One test per finding -------------------------------------------------


@not_root
def test_a_link_another_user_owns_in_a_sticky_folder_fails_the_walk():
    """A link in a sticky folder (/tmp) that
    another user owns can be replaced by that user, wherever it points
    today. Every walk refuses it, as a hop and as the path itself."""
    tmp = Path("/tmp")
    if not tmp.stat().st_mode & stat.S_ISVTX:
        pytest.skip("/tmp is not sticky here")
    link = tmp / f"bf-r9-{uuid.uuid4().hex}"
    link.symlink_to("/usr/bin")
    try:
        # The control: the folder it names passes on its own.
        assert _sh(root_only_path_check("/usr/bin/env", "env")).returncode == 0
        for check in (
            root_only_path_check(f"{link}/env", "env"),
            root_only_path_check(str(link), "a folder"),
            root_only_tree_check(str(link), "a tree"),
        ):
            result = _sh(check)
            assert result.returncode != 0, check
            assert "not root's alone" in result.stderr
    finally:
        link.unlink()


def test_the_seal_refuses_a_runtime_root_that_is_a_link():
    """The reserved root: /opt/runtime and its kit are
    folders, never aliases, before anything is made in them."""
    head = kit_prepare_command("build").split('bfc_chain "$RT" 0', 1)[0]
    assert '[ ! -L "$RT" ] && [ ! -L "$KIT" ] || bf_die' in head


def _daytona_sdk(seen: list[str]):
    from benchflow.sandbox.process.daytona import _BOOTSTRAP_DONE

    sandbox = MagicMock()

    async def process_exec(command, env=None, timeout=None):
        seen.append(command)
        return MagicMock(exit_code=0, result=f"{_BOOTSTRAP_DONE}\n")

    sandbox.process.exec = AsyncMock(side_effect=process_exec)
    return sandbox


@pytest.mark.asyncio
async def test_daytona_staging_pins_the_routes_path_before_any_program():
    """Each staging command Daytona's toolbox runs
    (the env file, the script file, the clean-up) starts the route's shell
    by absolute path, and its first text pins PATH: no program runs before
    it, the staging folder's mkdir and check included."""
    from benchflow._runtime_tree import RUN_DIR
    from benchflow.sandbox.process.daytona import (
        _bootstrap_daytona_env_file,
        _bootstrap_daytona_script_file,
        _cleanup_daytona_remote_env_file,
    )

    seen: list[str] = []
    sdk = _daytona_sdk(seen)
    path = f"{RUN_DIR}/stage-{uuid.uuid4().hex}"
    await _bootstrap_daytona_env_file(
        sdk,
        route=KIT_ROUTE,
        remote_env_path=path,
        env={"K": "v"},
        shell_exports=True,
        error_label="env",
    )
    await _bootstrap_daytona_script_file(
        sdk,
        route=KIT_ROUTE,
        remote_script_path=path + ".sh",
        script="true\n",
        error_label="script",
    )
    await _cleanup_daytona_remote_env_file(sdk, path, route=KIT_ROUTE)
    assert len(seen) == 3
    for command in seen:
        argv = shlex.split(command)
        assert argv[:2] == [KIT_SH, "-c"], command
        assert argv[2].startswith(system_shell_prefix(KIT_ROUTE.pin)), argv[2][:200]


def test_openhands_install_keeps_the_routes_path():
    """Root's OpenHands install puts no folder under a HOME on
    PATH (the uv guard and every later program stay the route's), and uv
    puts the tool and its entry point in BenchFlow's protected tree."""
    from benchflow.agents.registry import AGENTS

    install = AGENTS["openhands"].install_cmd
    for value in re.findall(r'export PATH="([^"]*)"', install):
        assert "$HOME" not in value and "~" not in value, value
    assert "export UV_TOOL_DIR=/opt/runtime/uv-tools " in install
    assert "UV_TOOL_BIN_DIR=/opt/runtime/bin" in install
    assert install.endswith("test -x /opt/runtime/bin/openhands")
    assert "command -v openhands" not in install
    launch = AGENTS["openhands"].launch_cmd
    # integ's umask (cli-other-blockers) comes first; then the PATH.
    assert launch.removeprefix("umask 077; ").startswith(
        'export PATH="/opt/runtime/bin:'
    )


@pytest.mark.asyncio
async def test_openhands_root_patch_runs_the_protected_entry_point():
    """The opt-in patch runs as root only from the protected
    tree, by absolute path, after its custody check."""
    from benchflow.acp.runtime import _prepare_openhands_direct_execution

    env = SimpleNamespace(exec=AsyncMock(return_value=ExecResult(return_code=0)))
    await _prepare_openhands_direct_execution(
        env,
        agent="openhands",
        agent_env={"BENCHFLOW_OPENHANDS_DISABLE_SUBAGENTS": "1"},
    )
    command = env.exec.await_args.args[0]
    assert "OH_BIN=/opt/runtime/bin/openhands" in command
    assert "command -v" not in command and "$HOME" not in command
    assert command.index("bfc_chain /opt/runtime/uv-tools 0") < command.index("OH_BIN=")
    assert '"$OH_PY" -I ' in command


@not_root
def test_the_task_route_restores_no_loader_value_another_user_can_replace(
    tmp_path, monkeypatch
):
    """The verifier's command (the task route as root on the
    kit) gets the image's LD_AUDIT back only when every file it names is
    root's alone; one the agent user could replace stays unset. Since round
    10 the value must also be on the seal's pre-agent record."""
    import benchflow.sandbox._base as base

    record = tmp_path / "IMAGE_ENV_OK"
    record.write_text("LD_AUDIT=/usr/lib\n")
    monkeypatch.setattr(base, "KIT_IMAGE_ENV_OK", str(record))
    audit = tmp_path / "audit.so"
    audit.write_text("")
    script = restore_image_env(tools=SYSTEM_PATH) + 'printf %s "${LD_AUDIT-unset}"'
    result = _sh(script, env={"PATH": SYSTEM_PATH, "_BF_IMAGE_LD_AUDIT": str(audit)})
    assert result.stdout == "unset"
    assert "not restoring LD_AUDIT" in result.stderr
    # A root-only value the seal recorded comes back as before.
    root_lib = "/usr/lib"
    result = _sh(script, env={"PATH": SYSTEM_PATH, "_BF_IMAGE_LD_AUDIT": root_lib})
    assert result.stdout == root_lib, result.stderr


@pytest.mark.asyncio
async def test_a_root_task_command_on_the_kit_checks_the_values_it_restores():
    """Docker's task route as root on the kit, at the exec."""
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = None
    sandbox.logger = logging.getLogger("test")
    sandbox._runtime_kit_fresh()
    sandbox.set_runtime_kit(KIT_RECORD)
    seen: list[list[str]] = []

    async def run(argv, check=False, timeout_sec=None, **kwargs):
        seen.append(list(argv))
        return ExecResult(return_code=0)

    sandbox._run_docker_compose_command = run
    with sandbox.task_command():
        await sandbox.exec("bash /tests/test.sh", user="root")
    argv = seen[-1]
    assert argv[argv.index("main") + 1] == KIT_SH
    assert "bf_safe_value" in argv[-1]


def test_a_root_terminal_starts_with_the_protected_home_and_no_startup_file():
    """The Daytona PTY (and any terminal on the kit) gets
    root's protected HOME and empty start-up settings before it reads a
    file; Docker's attach likewise."""
    from benchflow.sandbox._root_launch import terminal_env
    from benchflow.sandbox.docker import interactive_root_shell, route_exec_argv

    env = terminal_env(KIT_ROUTE)
    assert env["HOME"] == ROOT_HOME
    for name in ("BASH_ENV", "ENV", "PROMPT_COMMAND"):
        assert env[name] == ""
    assert env["INPUTRC"] == env["HISTFILE"] == "/dev/null"
    argv = route_exec_argv(
        KIT_ROUTE,
        "",
        container="main",
        user="0",
        stdin=True,
        tty=True,
        shell_argv=interactive_root_shell(KIT_ROUTE),
    )
    assert f"HOME={ROOT_HOME}" in argv and "INPUTRC=/dev/null" in argv
    assert argv[-4:] == [f"{KIT_BIN}/bash", "--noprofile", "--norc", "-i"]


def _terminal_sandbox(check_code: int, hooks: str = "|||"):
    calls: list[str] = []

    async def exec_route(user, **kwargs):
        return KIT_ROUTE

    async def exec(command, user=None, timeout_sec=None):
        calls.append(command)
        if command.startswith("printf"):
            return ExecResult(return_code=0, stdout=hooks)
        return ExecResult(
            return_code=check_code, stdout="" if check_code == 0 else "/root/.bashrc"
        )

    return SimpleNamespace(exec_route=exec_route, exec=exec), calls


@pytest.mark.asyncio
async def test_a_provider_terminal_that_reads_files_others_can_write_is_refused():
    """A terminal the provider creates itself (AgentCore's
    shell, Daytona's SSH session) starts only when what it reads at start is
    root's alone; otherwise the transport is refused."""
    from benchflow.sandbox._root_launch import require_protected_root_terminal

    ok, _ = _terminal_sandbox(0)
    await require_protected_root_terminal(ok, "the terminal")
    bad, _ = _terminal_sandbox(1)
    with pytest.raises(RuntimeError, match=r"/root/\.bashrc"):
        await require_protected_root_terminal(bad, "the terminal")
    hook, _ = _terminal_sandbox(0, hooks="x|||")
    with pytest.raises(RuntimeError, match="prompt hook"):
        await require_protected_root_terminal(hook, "the terminal")


def test_the_proxy_runtime_is_outside_any_workspace_a_task_may_have():
    """The sandbox-local proxy lives in BenchFlow's protected
    tree, so a /tmp workspace (accepted) no longer blocks it; a workspace
    inside that tree is refused."""
    from benchflow.providers.litellm_runtime import LITELLM_SANDBOX_ROOT

    assert LITELLM_SANDBOX_ROOT.startswith(RUNTIME_ROOT + "/")
    # /opt is granted without /opt/runtime (PT-C8-R3-11).
    assert workspace_grant_problem("/opt") is None
    for workspace in ("/tmp", "/tmp/work", "/app", "/root"):
        assert workspace_grant_problem(workspace) is None, workspace
        assert not LITELLM_SANDBOX_ROOT.startswith(workspace.rstrip("/") + "/")
    for workspace in ("/opt/runtime", "/opt/runtime/proxy"):
        assert workspace_grant_problem(workspace), workspace


def test_a_workspace_holding_a_folder_the_loader_searches_is_refused():
    """A workspace that is a folder the loader searches would
    give the agent a place the loader looks; since round 10 (PT-C8-R9-4) one
    that only holds such a folder (/usr/local) is granted without it."""
    assert workspace_grant_problem("/usr")
    for workspace in ("/usr/local/lib", "/usr/local/lib64"):
        assert workspace in (workspace_grant_problem(workspace) or ""), workspace
    assert workspace_grant_problem("/usr/local") is None
    assert workspace_grant_problem("/usr/local/app") is None
