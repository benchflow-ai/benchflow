"""C8 round 10: the regressions of the b114 review, one test each.

The cases that need a
real container are in ``test_c8_protected_runtime_container.py`` (round 10
section); these run anywhere, the root-owned folders they walk being the
test host's own.
"""

from __future__ import annotations

import logging
import os
import subprocess
from types import SimpleNamespace

import pytest

from benchflow._root_only import ROOT_CHAIN_FUNCS
from benchflow._runtime_tree import (
    RuntimeKit,
    kit_prepare_command,
    workspace_grant_command,
    workspace_grant_problem,
)
from benchflow.sandbox._base import ExecResult

KIT = RuntimeKit(trusted=("/usr/bin", "/bin"), python=True, bash=True)


def _chain(path: str) -> int:
    done = subprocess.run(
        ["/bin/sh", "-c", f'{ROOT_CHAIN_FUNCS}bfc_chain "$1" 0', "sh", path],
        capture_output=True,
        text=True,
    )
    return done.returncode


# --- PT-C8-R9-1 (both): a sidecar's kit is sealed after main's grant ---------


class _SetupEnv:
    """What setup_sandbox_user drives: a compose sandbox with one sidecar,
    recording the order of kit phases and the grant."""

    supports_runtime_kit = True
    supports_system_shell = True

    def __init__(self, sidecar_seal_ok: bool = True):
        self.steps: list[str] = []
        self.sidecar_seal_ok = sidecar_seal_ok
        self.logger = logging.getLogger("test")

    def set_runtime_kit(self, kit, service="main"):
        self.steps.append(f"record {service} {'kit' if kit else 'none'}")

    async def exec(self, command, **kwargs):
        if "useradd" in command:
            self.steps.append("grant")
        return ExecResult(return_code=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_seal_ok", [True, False])
async def test_a_sidecar_kit_is_sealed_after_mains_grant(monkeypatch, sidecar_seal_ok):
    """PT-C8-R9-1: every service's kit is built before the workspace grant
    and sealed after it (and before main's, PT-C8-R10-1), so a grant that reaches a sidecar's kit through a
    shared mount is undone by that seal; a sidecar whose seal fails after a
    good build (the grant took a folder above its kit) refuses the task
    before the agent runs."""
    from benchflow.sandbox import lockdown

    env = _SetupEnv()

    async def services(_env):
        return ["db"]

    async def prepare(_env, *, phase="all", timeout_sec=300, service="main"):
        env.steps.append(f"{phase} {service}")
        return not (service == "db" and phase == "seal" and not sidecar_seal_ok)

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(lockdown, "_kit_services", services)
    monkeypatch.setattr(lockdown, "prepare_runtime_tree", prepare)
    monkeypatch.setattr(lockdown, "lock_system_path", nothing)
    monkeypatch.setattr(lockdown, "guard_sandbox_daemon", nothing)
    if sidecar_seal_ok:
        await lockdown.setup_sandbox_user(env, "agent", "/app", guard_daemon=False)
    else:
        with pytest.raises(RuntimeError, match="'db'"):
            await lockdown.setup_sandbox_user(env, "agent", "/app", guard_daemon=False)
    steps = env.steps
    assert steps.index("build db") < steps.index("grant")
    assert steps.index("build main") < steps.index("grant")
    # The sidecar before main's kit is published (PT-C8-R10-1).
    assert steps.index("grant") < steps.index("seal db")
    if sidecar_seal_ok:
        assert steps.index("seal db") < steps.index("seal main")
    else:
        assert "seal main" not in steps
    assert "all db" not in steps


# --- PT-C8-R9-2: loader configuration comments ------------------------


def test_the_loader_configuration_is_read_as_ldconfig_reads_it():
    """PT-C8-R9-2: comments (leading whitespace, trailing), the
    whitespace around a line, ``include`` patterns and ``hwcap`` lines;
    each included file is proved root's alone as well."""
    seal = kit_prepare_command("seal")
    assert "l=${l%%#*}" in seal
    assert "include[[:space:]]*)" in seal and "hwcap[[:space:]]*) ;;" in seal
    assert "$bf_lfiles; do" in seal


# --- PT-C8-R9-2: a HOME with no .ssh ----------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="needs root-owned host folders")
def test_a_missing_folder_under_a_root_only_one_is_only_roots_to_create():
    """PT-C8-R9-2: a missing ``.ssh`` (or any missing folder on the
    way) under a folder only root may write is one only root could create
    (status 2), like a missing last entry; a missing one under a folder
    another user may write is not."""
    assert _chain("/usr/.bf-missing-ssh/rc") == 2
    assert _chain("/usr/.bf-missing-file") == 2
    assert _chain("/tmp/.bf-missing-ssh/rc") == 1


# --- PT-C8-R9-3: Gemini on native Daytona with /root ----------------


def _home_sandbox(answer: ExecResult):
    calls: list[str] = []

    async def exec_route(user, **kwargs):
        return SimpleNamespace(kit=True)

    async def exec(command, user=None, timeout_sec=None):
        calls.append(command)
        return answer

    return SimpleNamespace(exec_route=exec_route, exec=exec), calls


@pytest.mark.asyncio
async def test_daytonas_ssh_session_gets_a_protected_home_not_a_refusal():
    """PT-C8-R9-3: when root's HOME is the agent's (a workspace of
    /root), the SSH session's account entry is pointed at the protected
    HOME before it opens, so an honest Gemini task launches; it is refused
    only when that cannot be done."""
    from benchflow.sandbox._root_launch import protect_root_terminal_home

    ok, calls = _home_sandbox(ExecResult(return_code=0, stdout="moved\n"))
    await protect_root_terminal_home(ok, "Daytona's SSH session")
    assert "/opt/runtime/run/home" in calls[0] and "/etc/passwd" in calls[0]
    bad, _ = _home_sandbox(ExecResult(return_code=1, stdout="/etc/passwd\n"))
    with pytest.raises(RuntimeError, match="/etc/passwd"):
        await protect_root_terminal_home(bad, "Daytona's SSH session")


# --- PT-C8-R9-4: /usr/local workspaces --------------------------------


def test_a_usr_local_workspace_is_granted_without_the_loaders_folders():
    """PT-C8-R9-4: /usr/local is a workspace again; its grant keeps
    the loader's folders and the files directly in them root's (and makes
    them first when missing), and grants the rest."""
    assert workspace_grant_problem("/usr/local") is None
    for loader in ("/usr/local/lib", "/usr/local/lib64"):
        assert workspace_grant_problem(loader)
    assert workspace_grant_problem("/usr/local/lib/x86_64-linux-gnu")
    assert workspace_grant_problem("/usr/local/lib/python3.12") is None
    grant = workspace_grant_command("/usr/local", "agent")
    assert "chmod 1777 /usr/local" in grant
    assert "! -path /usr/local/lib ! -path /usr/local/lib64" in grant
    assert "mkdir -m 755 /usr/local/lib64" in grant
    assert "! -name '*-linux-*' -exec chown -R agent:agent" in grant
    assert workspace_grant_command("/app", "agent") == "chown -R agent:agent /app"


# --- The b122 review's mediums -----------------------------------------------


@pytest.mark.asyncio
async def test_a_fresh_sidecar_is_sealed_through_the_real_route(monkeypatch):
    """PT-C8-R10-1 (both): with the real prepare_runtime_tree and exec_route
    (only the container is a fake), a sidecar with no kit yet reaches its
    sealing command: it is sealed after the grant and before main's kit is
    published, when a service without one would be refused."""
    from benchflow.sandbox import lockdown
    from benchflow.sandbox.docker import DockerSandbox
    from tests._receipt_fakes import inner_command, receipt_answer

    record = (
        "BF kit-record\nkit=1\npython=0\nbash=0\ntrusted=/usr/bin:/bin\nBF kit-end\n"
    )
    seen: list[tuple[str, str]] = []
    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = None
    sandbox.logger = logging.getLogger("test")
    sandbox._runtime_kit_fresh()

    async def run(argv, check=False, timeout_sec=None, **kwargs):
        argv = list(argv)
        i = 1
        while argv[i].startswith("-"):
            i += 2 if argv[i] in ("-u", "-w", "-e", "--env") else 1
        service, payload = argv[i], argv[-1]
        inner = inner_command(payload)
        if 'echo "BF kit-built"' in inner:
            seen.append((service, "build"))
            answer = ExecResult(return_code=0, stdout="BF kit-built\n")
        elif "--- seal" in inner or 'echo "BF kit-record"' in inner:
            seen.append((service, "seal"))
            answer = ExecResult(return_code=0, stdout=record)
        else:
            answer = ExecResult(return_code=0)
        return receipt_answer(payload, answer)

    sandbox._run_docker_compose_command = run

    async def services(_env):
        return ["db"]

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(lockdown, "_kit_services", services)
    monkeypatch.setattr(lockdown, "lock_system_path", nothing)
    await lockdown.setup_sandbox_user(sandbox, "agent", "/app", guard_daemon=False)
    assert ("db", "seal") in seen and ("main", "seal") in seen
    assert seen.index(("db", "seal")) < seen.index(("main", "seal"))
    assert sandbox.runtime_kit is not None
    route = await sandbox.exec_route("root", service="db")
    assert route.kit


def test_the_loader_configuration_keeps_a_folders_name_as_written():
    """PT-C8-R10-2: glibc's ld.so.conf takes the rest of a line as
    one folder (commas and spaces included); only musl's .path files are
    split, on ``:`` and newlines."""
    seal = kit_prepare_command("seal")
    assert "tr ':,'" not in seal
    assert 'bf_ldirs="$bf_ldirs$bf_nl$l";;' in seal
    assert "l=${l%%=*}" in seal
    assert "*/ld-musl-*.path)" in seal


@pytest.mark.asyncio
async def test_agentcores_terminal_gets_the_protected_home_first(monkeypatch):
    """PT-C8-R10-3: AgentCore's terminal, like Daytona's SSH session, gets
    the protected HOME before the check, so a /root workspace launches."""
    import sys
    import types

    from benchflow.sandbox.process import agentcore as proc

    fake = types.ModuleType("bedrock_agentcore.runtime")
    fake.AgentCoreRuntimeClient = object
    monkeypatch.setitem(
        sys.modules, "bedrock_agentcore", types.ModuleType("bedrock_agentcore")
    )
    monkeypatch.setitem(sys.modules, "bedrock_agentcore.runtime", fake)
    calls: list[str] = []

    async def route(_sandbox):
        return SimpleNamespace(kit=True, pin="/usr/bin:/bin")

    async def protect(_sandbox, what):
        calls.append("protect")

    async def require(_sandbox, what):
        calls.append("require")
        raise RuntimeError("stop here")

    monkeypatch.setattr(proc, "launch_route", route)
    monkeypatch.setattr(proc, "protect_root_terminal_home", protect, raising=False)
    monkeypatch.setattr(proc, "require_protected_root_terminal", require)
    process = proc.AgentCoreProcess(object(), "arn", "s", "us-east-1")
    with pytest.raises(RuntimeError, match="stop here"):
        await process.start("agent")
    assert calls == ["protect", "require"]
