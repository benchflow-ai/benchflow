"""C8 round 10: the b114 review's pre-existing release
blockers, one test each (titles in the review;
the root Python helpers are ``test_c8_r10_root_python.py``; container cases
in ``test_c8_protected_runtime_container.py``)."""

from __future__ import annotations

import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from benchflow._runtime_tree import kit_prepare_command
from benchflow.sandbox._base import SYSTEM_PATH, ExecRoute, restore_image_env

not_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="needs a user other than root to own the test files"
)


def _sh(command: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, **kwargs
    )


# --- PT-C8-R9-5 / PT-C8-R9-3: provenance before the freeze ----


@not_root
def test_a_value_that_became_roots_after_the_agent_ran_is_not_restored(
    tmp_path, monkeypatch
):
    """Verification freezes the workspace (its files become root's), so a
    library the agent wrote passes an ownership check made then. The task
    route restores a value only when the seal recorded it before the
    agent's turn; a root-only value missing from that record stays unset."""
    import benchflow.sandbox._base as base

    record = tmp_path / "IMAGE_ENV_OK"
    record.write_text("")
    monkeypatch.setattr(base, "KIT_IMAGE_ENV_OK", str(record))
    script = restore_image_env(tools=SYSTEM_PATH) + 'printf %s "${LD_AUDIT-unset}"'
    # /usr/lib stands for the frozen file: root's alone now.
    env = {"PATH": SYSTEM_PATH, "_BF_IMAGE_LD_AUDIT": "/usr/lib"}
    result = _sh(script, env=env)
    assert result.stdout == "unset"
    assert "not restoring LD_AUDIT" in result.stderr
    record.write_text("LD_AUDIT=/usr/lib\n")
    assert _sh(script, env=env).stdout == "/usr/lib"


def test_the_seal_records_the_values_root_alone_owns_before_the_turn():
    seal = kit_prepare_command("seal")
    assert '"$KIT/IMAGE_ENV_OK"' in seal and "/proc/1/environ" in seal


# --- PT-C8-R9-4: hardware-capability folders ---------------------------


def test_the_seal_proves_the_loaders_capability_folders():
    seal = kit_prepare_command("seal")
    assert "glibc-hwcaps tls x86_64 haswell" in seal
    assert "the system loader loads $bf_lb" in seal
    # A program's own search folders (RUNPATH/RPATH and their capability
    # subfolders): every folder glibc tries before a hit is proven.
    assert 'LD_DEBUG=libs "$bf_ld" --list "$x"' in seal
    assert "trying file=" in seal


# --- PT-C8-R9-6: a dropped kit tool never falls through PATH ---------


def test_a_dropped_kit_tool_leaves_a_refusal_first_on_path():
    """The tool's name stays in the kit's bin as a refusal, so root's PATH
    never reaches the image's own copy; a refusal is not a shell or bash."""
    seal = kit_prepare_command("seal")
    assert "root does not run it" in seal and "exit 126" in seal
    assert '[ -L "$KIT/bin/sh" ] || bf_die' in seal
    assert 'bash=0; [ -L "$KIT/bin/bash" ] && bash=1' in seal


# --- PT-C8-R9-7 / PT-C8-R9-5: attach startup hooks -----------


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_an_interactive_root_shell_runs_no_prompt_hook(tmp_path):
    """Docker's (and DinD's, remote Docker's) attach: root's interactive
    bash starts from ``env -i`` with terminal data only, so an image's
    PROMPT_COMMAND or a PS1 that runs a command never runs."""
    from benchflow.sandbox.docker import interactive_root_shell

    bash = shutil.which("bash")
    route = ExecRoute(pin=SYSTEM_PATH, task_root=False, bash=bash, shell="/bin/sh")
    argv = interactive_root_shell(route)
    assert argv[:2] == ["/usr/bin/env", "-i"]
    mark = tmp_path / "hook-ran"
    done = subprocess.run(
        argv,
        input="exit\n",
        env={
            "PROMPT_COMMAND": f"touch {mark}",
            "PS1": f"$(touch {mark})",
            "BASH_ENV": f"{tmp_path}/rc",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr
    assert not mark.exists()


@pytest.mark.asyncio
async def test_daytona_attach_protects_or_refuses_before_the_ssh_shell(monkeypatch):
    """Native Daytona's attach opens the provider's root login shell: it
    gets the protected HOME and the terminal check first, and a refusal
    stops it before any SSH access is created."""
    from benchflow.sandbox import _root_launch
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect

    calls: list[str] = []

    async def protect(sandbox, what):
        calls.append("protect")

    async def refuse(sandbox, what):
        calls.append("require")
        raise RuntimeError("refused")

    monkeypatch.setattr(_root_launch, "protect_root_terminal_home", protect)
    monkeypatch.setattr(_root_launch, "require_protected_root_terminal", refuse)
    from benchflow.sandbox import daytona_strategies

    def no_exec(*args):
        # Before the fix attach reached the provider's shell: stop there.
        calls.append("execvp")
        raise RuntimeError("opened the SSH shell")

    monkeypatch.setattr(daytona_strategies.os, "execvp", no_exec)

    async def ssh_access():
        calls.append("ssh")
        return SimpleNamespace(token="t")

    async def exec_route(user, **kwargs):
        return ExecRoute(pin=SYSTEM_PATH, task_root=False)

    env = SimpleNamespace(
        _sandbox=SimpleNamespace(create_ssh_access=ssh_access), exec_route=exec_route
    )
    strategy = object.__new__(_DaytonaDirect)
    strategy._env = env
    with pytest.raises(RuntimeError):
        await strategy.attach()
    assert calls == ["protect", "require"]
