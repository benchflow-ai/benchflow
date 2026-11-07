"""One judged interpreter for V3's root Python steps.

The V3 integration review (V3-INTEG-1, -3, -4) found root's preparation
steps starting a ``python3`` by name under the system route (none on a
``python:*`` image), refusing honest conda-only images, and running the
chosen interpreter before V3's identity check. Every root Python step now
runs the one interpreter the host accepted first: safe folders, then the
image PATH, every hop of its route passing V3's identity rule.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.task import keep_services
from benchflow.task.keep_services import KeepService, KeptServices
from tests.test_sandbox_hardening import _make_env, _make_task

# A command token "python3" that the shell would look up by name.
_BARE_PYTHON = re.compile(r"(?:^|[\s;&|(){}`])python3(?=[\s;&|)]|$)")


def _unquoted(command: str) -> str:
    """*command* without its single-quoted text (shlex's '"'"' joins the
    pieces of one quoted word first)."""
    return re.sub(r"'[^']*'", "''", command.replace("'\"'\"'", ""))


def _rendered_steps(workspace: str) -> dict[str, str]:
    """Every command the verifier preparation runs as root, by step."""
    steps = [
        *lockdown._verifier_output_steps(),
        lockdown._reclaim_step(workspace),
        *lockdown._freeze_steps(workspace),
        lockdown._cleanup_step(lockdown.HARDENING_DEFAULTS, workspace),
    ]
    rendered = {step.name: step.command for step in steps}
    rendered["restore_workspace"] = lockdown._restore_workspace_cmd(workspace)
    return rendered


def test_no_preparation_step_starts_python3_by_name():
    """V3-INTEG-1: under the system route a bare ``python3`` misses
    /usr/local/bin (a python:* image's only one) and the step silently does
    nothing; any found by name had not been judged either."""
    bare = {
        name: command
        for name, command in _rendered_steps("/app").items()
        # Single-quoted text is data to the shell (a message, a -c script):
        # PT-R16's clean-up script names python3 in its own docstrings.
        if _BARE_PYTHON.search(_unquoted(command))
    }
    assert bare == {}, sorted(bare)


def test_python_steps_run_only_the_judged_interpreter(tmp_path, monkeypatch):
    """The route names the accepted file; each Python step starts that one."""
    stub = tmp_path / "judged-python3"
    marker = tmp_path / "ran"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$0" >> {shlex.quote(str(marker))}\n')
    stub.chmod(0o755)
    token = lockdown._root_python.set(str(stub))
    try:
        command = lockdown._prep_on_path(lockdown._purge_external_symlinks_cmd("/app"))
    finally:
        lockdown._root_python.reset(token)
    done = subprocess.run(
        ["sh", "-c", command], capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr
    assert marker.read_text().splitlines() == [str(stub)]


class ShellEnv:
    """Runs every command in ``sh`` as this user, with *image_path* as the
    sandbox's PATH (read as data by the judge)."""

    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.commands: list[str] = []

    async def exec(self, command: str, **kwargs) -> ExecResult:
        self.commands.append(command)
        done = subprocess.run(
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            env={"PATH": self.image_path, "HOME": "/"},
            timeout=60,
        )
        return ExecResult(
            stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
        )


@pytest.fixture
def closed_root(monkeypatch):
    """A folder tree no world can write (unlike pytest's tmp under /tmp), so
    its files can pass the route check for an agent user who is not us."""
    base = Path.home() / f".v3-root-python-{uuid.uuid4().hex}"
    base.mkdir(mode=0o755)
    base.chmod(0o755)
    yield base
    for path in sorted(base.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        else:
            path.unlink()
    base.rmdir()


def _python_in(folder: Path) -> Path:
    folder.mkdir(mode=0o755, parents=True, exist_ok=True)
    python = folder / "python3"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    return python


def _agent() -> str:
    """A user who is neither root nor us: the files we own are not theirs."""
    import pwd

    for name in ("nobody", "daemon"):
        try:
            if pwd.getpwnam(name).pw_uid not in (0, os.getuid()):
                return name
        except KeyError:
            continue
    pytest.skip("no second unprivileged user")


async def test_the_judge_prefers_a_safe_folder(closed_root, monkeypatch):
    safe = _python_in(closed_root / "safe")
    _python_in(closed_root / "conda" / "bin")
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(safe.parent),))
    env = ShellEnv(f"{closed_root / 'conda' / 'bin'}:/usr/bin:/bin")
    judged = await lockdown._judge_root_python(env, (), _agent())
    assert judged.path == str(safe)


async def test_a_conda_only_image_gets_its_image_path_python(closed_root, monkeypatch):
    """V3-INTEG-3: no python3 in the safe folders; the image PATH's passes."""
    empty = closed_root / "safe"
    empty.mkdir(mode=0o755)
    conda = _python_in(closed_root / "opt" / "conda" / "bin")
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(empty),))
    env = ShellEnv(f"{conda.parent}:/usr/bin:/bin")
    judged = await lockdown._judge_root_python(env, (), _agent())
    assert (judged.path, judged.rejected) == (str(conda), ())


async def test_an_excluded_route_is_rejected_and_the_next_judged(
    closed_root, monkeypatch
):
    """A candidate under the workspace is rejected with its reason, and the
    next candidate is judged and accepted."""
    workspace = _python_in(closed_root / "app" / "bin")
    good = _python_in(closed_root / "image" / "bin")
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(workspace.parent),))
    judged = await lockdown._judge_root_python(
        ShellEnv(str(good.parent)), (str(closed_root / "app"),), _agent()
    )
    assert judged.path == str(good)
    assert "under" in dict(judged.rejected)[str(workspace)], judged.rejected


async def test_a_route_the_agent_user_owns_is_rejected(closed_root, monkeypatch):
    """Our own files, judged with us as the agent user, never pass."""
    import pwd

    owned = _python_in(closed_root / "agent-owned")
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(owned.parent),))
    me = pwd.getpwuid(os.getuid()).pw_name
    judged = await lockdown._judge_root_python(ShellEnv(str(owned.parent)), (), me)
    assert judged.path is None
    assert "agent user" in dict(judged.rejected)[str(owned)], judged.rejected


async def test_no_python3_anywhere_is_none_without_rejections(closed_root, monkeypatch):
    empty = closed_root / "safe"
    empty.mkdir(mode=0o755)
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(empty),))
    judged = await lockdown._judge_root_python(ShellEnv(str(empty)), (), _agent())
    assert (judged.path, judged.rejected) == (None, ())


def _encoded(value: str) -> str:
    """A field as the judge prints it (percent-encoded bytes)."""
    return quote(value, safe="")


def _judge_answer(command: str, accepted: str | None, rejected=()) -> str | None:
    if "bf-root-python" not in command:
        return None
    lines = [f"R {_encoded(path)}\nW {_encoded(why)}" for path, why in rejected]
    if accepted:
        lines.append(f"A {_encoded(accepted)}")
    return "".join(line + "\n" for line in lines)


async def test_the_interpreter_is_judged_before_any_root_python_step(monkeypatch):
    """V3-INTEG-4: the judgement comes before the first step that could run
    Python, and every such step carries the accepted file."""
    seen: list[str] = []

    def execute(command, **kwargs):
        seen.append(command)
        answer = _judge_answer(command, "/opt/conda/bin/python3")
        return ExecResult(stdout=answer or "", stderr="", return_code=0)

    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    monkeypatch.setattr(lockdown, "_build_verifier_env", AsyncMock(return_value={}))
    env = _make_env(execute)
    await lockdown.harden_before_verify(
        env, _make_task(), sandbox_user="agent", workspace="/app"
    )
    judge = next(i for i, c in enumerate(seen) if "bf-root-python" in c)
    python_steps = [i for i, c in enumerate(seen) if "__bf_root_python" in c]
    assert python_steps and judge < min(python_steps), seen
    for index in python_steps:
        assert "__bf_root_python=/opt/conda/bin/python3; " in seen[index]


async def test_rejected_candidates_refuse_grading_naming_the_image(monkeypatch):
    def execute(command, **kwargs):
        answer = _judge_answer(
            command, None, [("/usr/local/bin/python3", "/app is under /app")]
        )
        return ExecResult(stdout=answer or "", stderr="", return_code=0)

    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    build = AsyncMock(return_value={})
    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    task = _make_task()
    task.config.environment.docker_image = "python:3.12-slim"
    with pytest.raises(RuntimeError) as caught:
        await lockdown.harden_before_verify(
            _make_env(execute), task, sandbox_user="agent", workspace="/app"
        )
    assert "python:3.12-slim" in str(caught.value)
    assert "/usr/local/bin/python3" in str(caught.value)
    build.assert_not_awaited()


async def test_kept_services_on_a_conda_image_check_ancestry_with_its_python(
    tmp_path, monkeypatch
):
    """V3-INTEG-3: the ancestry check runs the judged image-PATH interpreter
    instead of refusing an honest conda-only image."""
    workspace = tmp_path / "job" / "submission"
    workspace.mkdir(parents=True)
    ancestry: list[str] = []

    def execute(command, **kwargs):
        answer = _judge_answer(command, "/opt/conda/bin/python3")
        if "Cannot protect workspace resolution" in command:
            ancestry.append(command)
        return ExecResult(stdout=answer or "", stderr="", return_code=0)

    for name in ("_freeze_workspace", "_reclaim_disk"):
        monkeypatch.setattr(lockdown, name, AsyncMock())
    monkeypatch.setattr(lockdown, "_record_kept", lambda *a, **kw: None)
    monkeypatch.setattr(
        keep_services,
        "trial_keep",
        lambda *a: KeptServices((KeepService(5000),), "the task"),
    )
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    monkeypatch.setattr(
        lockdown,
        "_kill_sandbox_user_procs_keeping",
        AsyncMock(return_value=lockdown.KeepingKill(0, (), 0.0)),
    )
    build = AsyncMock(return_value={})
    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    await lockdown.harden_before_verify(
        _make_env(execute), _make_task(), sandbox_user="agent", workspace=str(workspace)
    )
    (command,) = ancestry
    assert "/opt/conda/bin/python3 -I -S -c " in command
    build.assert_awaited_once()


async def test_a_workspace_venv_link_yields_its_target_python(closed_root, monkeypatch):
    """V8-R2-A-1 kept: the image's only interpreter is reached through a
    workspace venv's link. The link is rejected; its target, judged on its
    own path, is accepted."""
    empty = closed_root / "safe"
    empty.mkdir(mode=0o755)
    real = _python_in(closed_root / "opt" / "python" / "bin")
    venv_bin = closed_root / "app" / ".venv" / "bin"
    venv_bin.mkdir(parents=True, mode=0o755)
    (venv_bin / "python3").symlink_to(real)
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(empty),))
    judged = await lockdown._judge_root_python(
        ShellEnv(str(venv_bin)), (str(closed_root / "app"),), _agent()
    )
    assert judged.path == str(real)
    assert str(venv_bin / "python3") in dict(judged.rejected)


async def test_a_rewritten_link_cannot_name_a_non_python_program(
    closed_root, monkeypatch
):
    """The agent controls a workspace link's text: a target not named
    python3 or python3.N is never a candidate, however trusted."""
    empty = closed_root / "safe"
    empty.mkdir(mode=0o755)
    shell = closed_root / "bin" / "sh"
    shell.parent.mkdir(mode=0o755)
    shell.write_text("#!/bin/sh\nexit 0\n")
    shell.chmod(0o755)
    venv_bin = closed_root / "app" / ".venv" / "bin"
    venv_bin.mkdir(parents=True, mode=0o755)
    (venv_bin / "python3").symlink_to(shell)
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(empty),))
    judged = await lockdown._judge_root_python(
        ShellEnv(str(venv_bin)), (str(closed_root / "app"),), _agent()
    )
    assert judged.path is None


async def test_a_kit_without_a_python_judges_and_runs_no_image_python(tmp_path):
    """With PT's kit and no kit Python, no image interpreter is judged or run
    as root; the steps that need one are skipped, and the reason names the
    kit."""
    from benchflow._runtime_tree import RuntimeKit

    env = ShellEnv(str(tmp_path))
    env.runtime_kit = RuntimeKit(trusted=("/usr/bin",), python=False, bash=False)
    judged = await lockdown._judge_root_python(env, (), _agent())
    assert judged.path is None and judged.kit_without_python
    assert env.commands == []  # no image PATH read, no candidate judged
    text = judged.missing_text("img")
    assert "holds no Python" in text
    assert "PY8-" not in text  # no internal ticket id in a log line (dogfood p1-5)
