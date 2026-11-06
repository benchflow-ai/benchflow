"""The exec env's transport after b156 (main-sync round 10).

Round 9 wrote every env value into the command as octal escapes, set with the
shell's own printf and eval. The b156 review found three regressions and the
release blocker it left open:

- Size: four bytes per byte of the env put an env of 40 kB
  past Linux's 128 KiB limit for one argument, so the command never started.
- Scratch name: the wrapper's ``unset _bf_env`` removed a
  task variable of that name.
- A ``[verifier.env] BASH_ENV`` template was checked before it
  was resolved, so a templated, root-owned activation file was refused.
- Soft verification put the verifier's env on root's command
  line while the agent's processes were alive to read it in /proc.

Each backend's real ``exec`` runs here over a local transport (the harness of
``test_verifier_script_shell``); the soft verifier runs ``DockerSandbox``
itself with only ``docker exec`` swapped (``test_verifier_soft_transport``).
Every value is a dummy marker.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import threading
from pathlib import Path

import pytest

from benchflow.task import RolloutPaths, Verifier
from tests.test_verifier_script_shell import (
    _DASH,
    LocalImage,
    VerifierSandbox,
    _backend,
    _stdout,
    _task,
    _test_inspector,
    _verify_hardened,
    _without_image_probes,
)
from tests.test_verifier_soft_transport import _docker as _soft_docker
from tests.test_verifier_soft_transport import _soft, _task_for

KINDS = ("docker", "remote-docker", "apple-container", "daytona", "daytona-dind")
# The backends whose exec feeds the env on stdin, so it has no size limit of
# its own. Daytona refuses an env too large for one command (b173,
# test_env_transport_b173).
STDIN_KINDS = ("docker", "remote-docker", "apple-container")
# (kind, size) for the tests that run an env of both sizes.
SIZED = [
    (kind, size)
    for kind in KINDS
    for size in ("inline", "large")
    if size == "inline" or kind in STDIN_KINDS
]


@pytest.fixture(autouse=True)
def _needs_dash():
    if _DASH is None:
        pytest.skip("needs dash, the sh of Debian and Ubuntu images")


def _exec_for(kind: str, tmp_path: Path, monkeypatch):
    """``exec(command, timeout_sec, env=...)`` through *kind*'s own exec."""
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True)
    exec_ = _backend(kind, image, monkeypatch)
    return exec_


# --- An env of any size reaches the command ------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", STDIN_KINDS)
async def test_a_200_kb_value_reaches_the_command(tmp_path, kind, monkeypatch):
    """Fails on a7403b22: the 800 kB command is refused (E2BIG)."""
    exec_ = _exec_for(kind, tmp_path, monkeypatch)
    command = '[ "${#DATA}" -eq 200000 ] && [ "$SMALL" = ok ] && echo carried'
    try:
        result = await exec_(command, 60, env={"DATA": "d" * 200_000, "SMALL": "ok"})
    except OSError as exc:  # the transport could not start the command at all
        pytest.fail(f"the command did not start: {exc}")
    assert result.return_code == 0, (result.stdout or "")[-300:] + (
        result.stderr or ""
    )[-300:]
    assert (result.stdout or "").strip() == "carried"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", STDIN_KINDS)
async def test_200_kb_of_values_reach_a_program_the_command_starts(
    tmp_path, kind, monkeypatch
):
    """Exported, so a child gets them (each under the kernel's 128 KiB)."""
    exec_ = _exec_for(kind, tmp_path, monkeypatch)
    env = {f"PART_{i}": str(i) * 50_000 for i in range(4)}
    command = (
        "sh -c 'echo ${#PART_0}:${#PART_1}:${#PART_2}:${#PART_3}:${PART_3%${PART_3#?}}'"
    )
    try:
        result = await exec_(command, 60, env=env)
    except OSError as exc:
        pytest.fail(f"the command did not start: {exc}")
    assert (result.stdout or "").strip() == "50000:50000:50000:50000:3", result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_awkward_values_round_trip_at_any_size(tmp_path, kind, monkeypatch):
    """Quotes, newlines, escapes and UTF-8, inline and (on stdin) large alike."""
    exec_ = _exec_for(kind, tmp_path, monkeypatch)
    awkward = 'it\'s "q" `t` $(echo no) $HOME\nline\\n %s \\0101 é🙂'
    envs = [{"AWK": awkward}]
    if kind in STDIN_KINDS:
        envs.append({"AWK": awkward, "PAD": "p" * 100_000})
    for env in envs:
        result = await exec_('printf "%s" "$AWK"', 60, env=env)
        assert result.return_code == 0, result.stderr
        assert result.stdout == awkward


# --- No scratch name a task variable could share --------------

# Every name the env wrappers and the root-file handoff have used for their
# own state (round 9's ``_bf_env``, the handoff's ``rc``), and the names a
# reader of the new ones might guess.
_SCRATCH_LIKE = {
    "_bf_env": "grade",
    "rc": "kept-rc",
    "k": "kept-k",
    "v": "kept-v",
    "env_path": "kept-path",
    "_BF_ENV": "kept-upper",
}
_ALL_KEPT = " && ".join(
    f'[ "${name}" = {value} ]' for name, value in _SCRATCH_LIKE.items()
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "size"), SIZED)
async def test_a_task_variable_named_like_a_scratch_name_survives(
    tmp_path, kind, size, monkeypatch
):
    """Fails on a7403b22: its trailing ``unset _bf_env`` removed the task's."""
    exec_ = _exec_for(kind, tmp_path, monkeypatch)
    env = dict(_SCRATCH_LIKE)
    if size == "large":
        env["PAD"] = "p" * 100_000
    try:
        result = await exec_(f"{_ALL_KEPT} && echo kept", 60, env=env)
    except OSError as exc:
        pytest.fail(f"the command did not start: {exc}")
    assert (result.stdout or "").strip() == "kept", result.stderr


def test_the_wrappers_assign_no_variable_of_their_own():
    """By construction: outside the env's own lines nothing is assigned or unset."""
    from benchflow.sandbox._base import (
        ENV_FROM_STDIN,
        wrap_command_with_env_file,
    )
    from benchflow.task.verifier_core import _root_file_env_prefix

    inline = re.sub(
        r"printf '(?:\\[0-7]{3})*'",
        "printf ''",
        wrap_command_with_env_file({"A": "1"}, ""),
    )
    for text in (ENV_FROM_STDIN, inline, _root_file_env_prefix("/x")):
        assert not re.search(r"(^|[\s;])[A-Za-z_][A-Za-z0-9_]*=", text), text
        assert "unset" not in text and "read " not in text, text


@pytest.mark.asyncio
async def test_the_root_file_handoff_keeps_a_task_variable_named_rc(tmp_path):
    """Fails on a7403b22: the handoff's ``rc=$?`` replaced the task's ``rc``.

    Kept services (the scoring path) and soft verification share the handoff.
    """
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = _soft_docker(tmp_path, paths, {"PATH": "/usr/bin:/bin"})
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(
        "#!/bin/sh\n"
        f"if {_ALL_KEPT}; then echo 1; else echo 0; fi"
        ' > "$VERIFIER_DIR/reward.txt"\n'
    )
    task = _task_for(tests_dir)
    task.config.verifier.env = dict(_SCRATCH_LIKE)
    verifier = Verifier(task=task, rollout_paths=paths, sandbox=sandbox)

    async def release(code_dir: str) -> None:
        return None

    verifier.before_test = release  # services are kept: the handoff runs
    result = await verifier.verify()
    (command,) = sandbox.ran  # type: ignore[attr-defined]
    assert "/.benchflow-verifier-env-" in command
    assert result.rewards == {"reward": 1.0}


# --- Soft verification puts no value on any command line --------

_JUDGE_KEY = "bf-dummy-judge-7c41e0"


def _encodings(value: str) -> list[bytes]:
    """The value and the reversible forms a wrapper has written it in."""
    import base64

    raw = value.encode()
    return [
        raw,
        "".join(f"\\{b:03o}" for b in raw).encode(),
        base64.b64encode(f"export JUDGE_API_KEY={value}\n".encode()).rstrip(b"="),
    ]


class _CmdlineWatcher:
    """An agent's process: reads every /proc/<pid>/cmdline until stopped."""

    def __init__(self, secret: str):
        self.needles = _encodings(secret)
        self.hits: list[str] = []
        self.scans = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.scans += 1
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmdline = f.read()
                except OSError:
                    continue
                if any(needle in cmdline for needle in self.needles):
                    self.hits.append(cmdline[:120].decode(errors="replace"))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs /proc")
@pytest.mark.asyncio
async def test_soft_verification_puts_no_verifier_value_on_a_command_line(tmp_path):
    """Fails on a7403b22: root's shell carried the key as octal in its argv.

    The agent's processes stay alive through soft verification; one of them
    reads every process's command line while test.sh runs.
    """
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = _soft_docker(tmp_path, paths, {"PATH": "/usr/bin:/bin"})
    sandbox._persistent_env = {"TASK_SETTING": "bf-dummy-persistent-91d2"}
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(
        "#!/bin/sh\nsleep 1\n"
        f'if [ "$JUDGE_API_KEY" = {_JUDGE_KEY} ] && '
        '[ "$TASK_SETTING" = bf-dummy-persistent-91d2 ]; '
        'then echo 1; else echo 0; fi > "$VERIFIER_DIR/reward.txt"\n'
    )
    task = _task_for(tests_dir)
    task.config.verifier.env = {"JUDGE_API_KEY": _JUDGE_KEY}

    with _CmdlineWatcher(_JUDGE_KEY) as watcher:
        rewards, _, error = await _soft(tmp_path, sandbox, task, paths)
        await asyncio.sleep(0.2)

    assert error is None, error
    assert rewards == {"reward": 1.0}
    assert watcher.scans > 1
    assert watcher.hits == [], watcher.hits[:3]
    (command,) = sandbox.ran  # type: ignore[attr-defined]
    assert "/.benchflow-verifier-env-" in command  # the root-only file
    assert sandbox.stdin == []  # type: ignore[attr-defined]  # not even on stdin


# --- A task BASH_ENV template is resolved before the check -------

_ACTIVATION = "/etc/bash.bashrc"
_TEMPLATE = "${BF_R10_ACTIVATE:-/etc/bash.bashrc}"
_NEEDS_ACTIVATION = (
    f'#!/bin/bash\n[ "$BASH_ENV" = {_ACTIVATION} ] && r=1 || r=0\n'
    'printf "%s\\n" "$r" > "$VERIFIER_DIR/reward.txt"\n'
)


def _needs_root_owned_activation() -> None:
    if (
        not os.path.isfile(_ACTIVATION)
        or os.stat(_ACTIVATION).st_uid != 0
        or shutil.which("python3", path="/usr/local/bin:/usr/bin:/bin") is None
    ):
        pytest.skip(f"needs a system python3 and a root-owned {_ACTIVATION}")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("docker", "daytona"))
async def test_a_templated_root_owned_bash_env_is_kept_when_hardened(
    tmp_path, kind, monkeypatch
):
    """The scored path. Fails on a7403b22: ``$`` refused, BASH_ENV empty."""
    _needs_root_owned_activation()
    monkeypatch.delenv("BF_R10_ACTIVATE", raising=False)
    rewards, env = await _verify_hardened(
        tmp_path,
        _NEEDS_ACTIVATION,
        kind,
        monkeypatch,
        image_env={},
        task_env={"BASH_ENV": _TEMPLATE},
    )
    assert env["BASH_ENV"] == _ACTIVATION
    assert rewards == {"reward": 1.0}, _stdout(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("docker", "daytona"))
async def test_a_templated_root_owned_bash_env_is_kept_in_soft_verification(
    tmp_path, kind, monkeypatch
):
    """The soft path's check, then the Verifier with what it returned, as
    ``soft_verify`` does. Fails on a7403b22: the check refused the template."""
    from benchflow.sandbox.lockdown import soft_verifier_bash_env

    _needs_root_owned_activation()
    monkeypatch.delenv("BF_R10_ACTIVATE", raising=False)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True)
    sandbox = VerifierSandbox(image, _backend(kind, image, monkeypatch))
    _test_inspector(sandbox, tmp_path / "inspector", image.bin)
    _without_image_probes(monkeypatch, image)
    task = _task(tmp_path, _NEEDS_ACTIVATION)
    task.config.verifier.env = {"BASH_ENV": _TEMPLATE}

    bash_env = await soft_verifier_bash_env(
        sandbox, task, "agent", str(tmp_path / "app")
    )
    assert bash_env == _ACTIVATION
    verifier = Verifier(task=task, rollout_paths=paths, sandbox=sandbox)
    verifier.bash_env = bash_env
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}, _stdout(tmp_path)


@pytest.mark.asyncio
async def test_an_unresolvable_bash_env_template_activates_nothing(monkeypatch):
    """A required host variable that is missing: no activation, a warning."""
    from types import SimpleNamespace

    from benchflow.sandbox.lockdown import _trusted_verifier_bash_env

    monkeypatch.delenv("BF_R10_MISSING", raising=False)
    task = SimpleNamespace(
        config=SimpleNamespace(
            verifier=SimpleNamespace(
                service="main", env={"BASH_ENV": "${BF_R10_MISSING}"}
            )
        )
    )
    assert await _trusted_verifier_bash_env(object(), task, ()) == ""
