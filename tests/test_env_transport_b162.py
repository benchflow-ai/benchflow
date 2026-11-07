"""The exec env's edges after b162 (main-sync round 11).

The b162 review of round 10 kept its stdin and root-file design and found:

- Daytona staged a large env in a fresh 0700 folder under /tmp,
  which an image may leave without its sticky bit: the folder could be
  renamed away and replaced before root sourced it.
- On Daytona the staged file stayed the command's stdin, so a
  command that reads stdin got the env's ``export`` lines back.
- Soft verification held the persistent env back from
  exec, so the BASH_ENV probe missed a trusted BASH_ENV the sandbox's own
  env names.
- A large DinD env was applied before the heartbeat set
  its own variables (``bf_out``, ...), which then replaced the task's.
- Pre-existing: a verifier running as another non-root user
  beside live agent code got its env through exec, which Daytona writes into
  a command line the agent can read.

The backends run through the local-transport harness of
``test_verifier_script_shell``; every value is a dummy marker.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from benchflow.task import RolloutPaths, Verifier
from tests.test_env_transport_b156 import SIZED, _exec_for
from tests.test_verifier_script_shell import (
    _DASH,
    LocalImage,
    VerifierSandbox,
    _backend,
    _stdout,
    _task,
    _test_inspector,
    _without_image_probes,
)

_PAD = "p" * 100_000  # past the inline limit: stdin on Docker, refused on Daytona


@pytest.fixture(autouse=True)
def _needs_dash():
    if _DASH is None:
        pytest.skip("needs dash, the sh of Debian and Ubuntu images")


# --- The command's stdin is /dev/null once the env is applied ---


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "size"), SIZED)
async def test_a_command_reading_stdin_sees_nothing_or_its_own_redirect(
    tmp_path, kind, size, monkeypatch
):
    """Failed on 09a5f45e for Daytona's staged env (gone since b173): ``cat``
    read it back."""
    exec_ = _exec_for(kind, tmp_path, monkeypatch)
    env = {"SECRET": "bf-dummy-r11"}
    if size == "large":
        env["PAD"] = _PAD
    command = (
        'left=$(cat); [ -z "$left" ] && echo nothing-left; '
        "printf own | cat; echo; cat <<'EOF'\nheredoc\nEOF"
    )
    try:
        result = await exec_(command, 600, env=env)
    except OSError as exc:
        pytest.fail(f"the command did not start: {exc}")
    assert result.stdout == "nothing-left\nown\nheredoc\n", (
        (result.stdout or "")[:300],
        (result.stderr or "")[-300:],
    )
    assert "bf-dummy-r11" not in (result.stdout or "")


# --- The DinD heartbeat shares no name with the task ------

_HEARTBEAT_NAMES = {
    "bf_out": "grade",
    "bf_err": "kept-err",
    "bf_stop": "kept-stop",
    "bf_pid": "kept-pid",
    "bf_rc": "kept-rc",
    "bf_heartbeat_pid": "kept-hb-pid",
    "bf_heartbeat_elapsed": "kept-elapsed",
}


@pytest.mark.asyncio
async def test_a_dind_env_keeps_task_variables_named_like_the_heartbeats(
    tmp_path, monkeypatch
):
    """The inline env applies inside the heartbeat's worker, after its own
    variables (09a5f45e's large env did not: b162; a large DinD
    env is refused since b173)."""
    exec_ = _exec_for("daytona-dind", tmp_path, monkeypatch)
    env = dict(_HEARTBEAT_NAMES)
    check = " && ".join(f'[ "${k}" = {v} ]' for k, v in _HEARTBEAT_NAMES.items())
    # 600 s: long enough that the heartbeat wraps the command.
    result = await exec_(f"{check} && echo kept", 600, env=env)
    assert (result.stdout or "").strip() == "kept", (result.stderr or "")[-400:]
    assert result.return_code == 0


# --- A trusted BASH_ENV the sandbox's own env names ------

_ACTIVATION = "/etc/bash.bashrc"
_NEEDS_ACTIVATION = (
    f'#!/bin/bash\n[ "$BASH_ENV" = {_ACTIVATION} ] && r=1 || r=0\n'
    'printf "%s\\n" "$r" > "$VERIFIER_DIR/reward.txt"\n'
)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("docker", "daytona"))
async def test_soft_verification_keeps_a_trusted_bash_env_from_the_persistent_env(
    tmp_path, kind, monkeypatch
):
    """Fails on 09a5f45e: the probe, run without the persistent env, saw none."""
    from benchflow.sandbox._base import BaseSandbox
    from benchflow.sandbox.lockdown import soft_verifier_bash_env

    if (
        not os.path.isfile(_ACTIVATION)
        or os.stat(_ACTIVATION).st_uid != 0
        or shutil.which("python3", path="/usr/local/bin:/usr/bin:/bin") is None
    ):
        pytest.skip(f"needs a system python3 and a root-owned {_ACTIVATION}")
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True)
    sandbox = VerifierSandbox(image, _backend(kind, image, monkeypatch))
    # The task's [environment] env, as BaseSandbox keeps it; the image has none.
    sandbox._persistent_env = {"BASH_ENV": _ACTIVATION}  # type: ignore[attr-defined]
    _test_inspector(sandbox, tmp_path / "inspector", image.bin)
    _without_image_probes(monkeypatch, image)
    task = _task(tmp_path, _NEEDS_ACTIVATION)

    with BaseSandbox.services_kept(sandbox):  # type: ignore[arg-type]
        bash_env = await soft_verifier_bash_env(
            sandbox, task, "agent", str(tmp_path / "app")
        )
    assert bash_env == _ACTIVATION
    verifier = Verifier(task=task, rollout_paths=paths, sandbox=sandbox)
    verifier.bash_env = bash_env
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}, _stdout(tmp_path)


@pytest.mark.asyncio
async def test_an_agent_writable_bash_env_in_the_persistent_env_is_still_refused(
    tmp_path, monkeypatch
):
    """The same trust check applies to the sandbox's value."""
    from benchflow.sandbox.lockdown import soft_verifier_bash_env

    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True)
    sandbox = VerifierSandbox(image, _backend("docker", image, monkeypatch))
    planted = tmp_path / "app" / "activate.sh"
    planted.parent.mkdir(parents=True)
    planted.write_text("export TASK_READY=planted\n")
    sandbox._persistent_env = {"BASH_ENV": str(planted)}  # type: ignore[attr-defined]
    _test_inspector(sandbox, tmp_path / "inspector", image.bin)
    _without_image_probes(monkeypatch, image)
    task = _task(tmp_path, _NEEDS_ACTIVATION)
    assert (
        await soft_verifier_bash_env(sandbox, task, "agent", str(tmp_path / "app"))
        == ""
    )


# --- A non-root verifier beside live agents needs a private env --

_JUDGE_KEY = "bf-dummy-judge-r11-41c0"
_NEEDS_KEY = (
    f'#!/bin/sh\n[ "$JUDGE_API_KEY" = {_JUDGE_KEY} ] && r=1 || r=0\n'
    'echo "$r" > "$VERIFIER_DIR/reward.txt"\n'
)


def _docker_class():
    from benchflow.sandbox.docker import DockerSandbox

    return DockerSandbox


def _daytona_class():
    from benchflow.sandbox.daytona import DaytonaSandbox

    return DaytonaSandbox


def _modal_class():
    from benchflow.sandbox.modal_impl import ModalSandbox

    return ModalSandbox


_BACKEND_CLASSES = {
    "docker": _docker_class,
    "daytona": _daytona_class,
    "modal": _modal_class,
}


def _non_root_verifier(tmp_path: Path, kind: str, monkeypatch):
    if os.getuid() == 0:
        pytest.skip("the verifier's uid probe answers the test's own uid")
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True)
    sandbox = VerifierSandbox(image, _backend(kind, image, monkeypatch))
    # What the Verifier asks of the backend it runs on.
    sandbox.exec_env_is_private = getattr(  # type: ignore[attr-defined]
        _BACKEND_CLASSES[kind](), "exec_env_is_private", False
    )
    task = _task(tmp_path, _NEEDS_KEY)
    task.config.verifier.user = "grader"
    task.config.verifier.env = {"JUDGE_API_KEY": _JUDGE_KEY}
    verifier = Verifier(task=task, rollout_paths=paths, sandbox=sandbox)
    verifier.agent_processes_alive = True  # soft verification
    return verifier


@pytest.mark.asyncio
async def test_a_non_root_verifier_beside_live_agents_is_refused_on_daytona(
    tmp_path, monkeypatch
):
    """Fails on 09a5f45e: the key went inline, as octal in the command."""
    pytest.importorskip("daytona")
    verifier = _non_root_verifier(tmp_path, "daytona", monkeypatch)
    with pytest.raises(Exception) as raised:
        await verifier.verify()
    message = str(raised.value)
    assert "only on a command line the agent can read" in message, message
    assert "Run the verifier as root" in message
    assert not (tmp_path / "rollout" / "verifier" / "reward.txt").exists()


@pytest.mark.asyncio
async def test_a_non_root_verifier_beside_live_agents_gets_its_env_on_docker(
    tmp_path, monkeypatch
):
    """Docker passes it on stdin, private to the verifier's own process."""
    verifier = _non_root_verifier(tmp_path, "docker", monkeypatch)
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}, _stdout(tmp_path)
