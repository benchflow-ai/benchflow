"""The root step before test.sh takes group and other write from a workspace the grant made sticky (#1249 (b),
(c)): unit rows on the verifier's close of ``/logs/verifier`` with a stand-in sandbox.

The grant records, per workspace it made root's and 1777, the mode the image had (``WORKSPACE_MODE_ATTR``). In
integrity audit and off, before a root test.sh, the same root exec that closes ``/logs/verifier`` runs
``chmod go-w`` on each such workspace and shows the mode it left (``BF workspace-mode <mode> <uid> <path>``); a mode
with group or other write left, or none shown, is a verifier error ("Verifier hardening failed"), never a quiet 0.
Strict is unchanged (its freeze gives the image's mode back first). A verifier with a user of its own keeps the
workspace as the grant made it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._runtime_tree import WORKSPACE_MODE_ATTR
from benchflow.task.verifier_core import _OUTPUTS_CLOSED, Verifier
from benchflow.task.verifier_errors import VerifierSetupError

ROOT_MODES = {"/root": "700"}
CLOSED = f"{_OUTPUTS_CLOSED}\n"
SHOWN = f"{CLOSED}BF workspace-mode 1755 0 /root\n"


def _verifier(
    stdout: str = SHOWN,
    *,
    uid: str = "0",
    faithful: bool = True,
    modes: dict[str, str] | None = ROOT_MODES,
    return_code: int = 0,
) -> Verifier:
    from benchflow.task import TaskConfig

    task = MagicMock()
    task.config = TaskConfig.model_validate({"verifier": {}})
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                stdout=stdout, stderr="", return_code=return_code, exit_code=return_code
            )
        )
    )
    if faithful:
        sandbox._benchflow_verifier_faithful = True
    if modes is not None:
        setattr(sandbox, WORKSPACE_MODE_ATTR, dict(modes))
    verifier = Verifier(task=task, rollout_paths=MagicMock(), sandbox=sandbox)
    verifier._verifier_uid_read = uid
    verifier._host_writes_outputs = lambda: False  # type: ignore[method-assign]
    return verifier


def _commands(verifier: Verifier) -> list[str]:
    return [str(c.args[0]) for c in verifier._sandbox.exec.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
async def test_a_root_test_sh_gets_the_workspace_without_group_or_other_write(mounted):
    """Unit. Audit and off (test.sh as written), root's test.sh, a recorded workspace: the close's
    one exec also takes group and other write from it, and the mode it shows is accepted.

    Mutant: skip the step."""
    verifier = _verifier()
    await verifier._close_output_dir("main", mounted=mounted)
    (command,) = _commands(verifier)
    assert "go-w" in command and "/root" in command, command
    assert verifier.outputs_closed is True


@pytest.mark.asyncio
async def test_strict_closes_no_workspace():
    """Unit, guard. Strict: the freeze gave the image's mode back before test.sh; the close adds
    nothing for the workspace."""
    verifier = _verifier(CLOSED, faithful=False)
    await verifier._close_output_dir("main", mounted=False)
    (command,) = _commands(verifier)
    assert "/root" not in command, command


@pytest.mark.asyncio
async def test_a_verifier_user_of_its_own_keeps_the_grant():
    """Guard. test.sh runs as a user of the task's own: the owner of a file is not stopped by
    ``fs.protected_regular``, and that user may need to create files in the workspace.

    Mutant: ``go-w`` for every verifier user."""
    verifier = _verifier(CLOSED, uid="4321")
    await verifier._close_output_dir("main", mounted=False)
    (command,) = _commands(verifier)
    assert "/root" not in command, command


@pytest.mark.asyncio
async def test_no_recorded_workspace_closes_nothing():
    verifier = _verifier(CLOSED, modes=None)
    await verifier._close_output_dir("main", mounted=False)
    (command,) = _commands(verifier)
    assert "/root" not in command, command


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout",
    [
        CLOSED,
        f"{CLOSED}BF workspace-mode 1777 0 /root\n",
        f"{CLOSED}BF workspace-mode 1775 0 /root\n",
    ],
    ids=["nothing-shown", "still-1777", "group-write-left"],
)
async def test_a_step_that_did_not_take_is_a_verifier_error(stdout):
    """The step shows the mode it left; none shown, or one with group or other write, ends the verify
    as "Verifier hardening failed" (a verifier error, no reward) before test.sh runs.

    Mutant: ignore what the step shows."""
    verifier = _verifier(stdout)
    with pytest.raises(VerifierSetupError, match=r"(?i)verifier hardening failed"):
        await verifier._close_output_dir("main", mounted=False)


@pytest.mark.asyncio
async def test_the_step_adds_no_exec():
    """Unit, guard. The workspace step rides in the close's exec: one exec with a recorded workspace, as
    without one.

    Mutant: an extra exec."""
    with_modes = _verifier()
    await with_modes._close_output_dir("main", mounted=False)
    without = _verifier(CLOSED, modes=None)
    await without._close_output_dir("main", mounted=False)
    assert len(_commands(with_modes)) == len(_commands(without)) == 1


def test_strict_refuses_an_opt_workspace_by_name():
    """Guard. Strict's separate verifier owns ``/opt/runtime``: the transfer of a workspace of ``/opt``
    is refused with that reason, unchanged."""
    from benchflow.rollout._separate_verifier import (
        SeparateVerifierError,
        check_declared_transfer,
    )

    with pytest.raises(
        SeparateVerifierError, match="/opt overlaps verifier-owned /opt/runtime"
    ):
        check_declared_transfer("/opt", [])


# -- the record before the grant, and the re-open before another turn ----------------------------------------------


@pytest.mark.parametrize(
    ("workspace", "sticky"),
    [("/root", False), ("/app", False), ("/opt", True), ("/usr/local", True)],
)
def test_the_probe_records_every_workspace_the_grant_makes_sticky(
    tmp_path, workspace, sticky
):
    """Row A3 (unit). The probe before the grant prints the mode for a workspace that holds a reserved folder
    (``/opt``, ``/usr/local``), daemon folder or not; ``/root`` and ``/app`` only with a daemon folder in them.

    Mutant: record the mode only for a daemon folder (0.10.0's rule)."""
    import subprocess

    from benchflow._runtime_tree import workspace_mode_probe

    probe = workspace_mode_probe(workspace)
    assert ("/opt" in probe) and ("/usr/local" in probe), probe
    # Run it where the physical path is the workspace itself: a shell that resolves `cd -P` to it.
    fake = tmp_path / workspace.strip("/").replace("/", "_")
    fake.mkdir()
    script = probe.replace(f"cd -P {workspace} ", f"cd -P {fake} ").replace(
        "pwd -P)", f"pwd -P | sed 's#^{fake}$#{workspace}#')"
    )
    script = script.replace(f"stat -L -c %a {workspace}", f"stat -L -c %a {fake}")
    shown = subprocess.run(["sh", "-c", script], capture_output=True, text=True)
    assert ("benchflow-workspace-mode" in shown.stdout) is sticky, (script, shown)


@pytest.mark.asyncio
async def test_the_next_turn_gets_the_grants_mode_back_first():
    """Unit. After a verify took group and other write, the next agent turn's connect gives each such
    workspace the grant's 1777 back (one root exec, checked), and only then; a re-open that doesn't take stops the
    trial.

    Mutant: no re-open after the verify."""
    from benchflow._runtime_tree import WORKSPACE_CLOSED_ATTR
    from benchflow.sandbox._workspace_grant import reopen_granted_workspaces

    def receipt_env(stdout: str):
        from benchflow.sandbox._exec_receipt import encode_receipt

        async def run(command, **kwargs):
            from benchflow.sandbox._exec_receipt import unwrap_receipt_command

            _, nonce = unwrap_receipt_command(command)
            return SimpleNamespace(
                stdout=encode_receipt(stdout.encode(), 0, nonce),
                stderr="",
                return_code=0,
                exit_code=0,
            )

        return SimpleNamespace(exec=AsyncMock(side_effect=run))

    untouched = receipt_env("")
    await reopen_granted_workspaces(untouched)
    untouched.exec.assert_not_awaited()

    env = receipt_env("BF workspace-mode 1777 0 /root\n")
    setattr(env, WORKSPACE_CLOSED_ATTR, {"/root": "700"})
    await reopen_granted_workspaces(env)
    assert env.exec.await_count == 1
    assert "chmod 1777 /root" in str(env.exec.await_args.args[0])
    assert getattr(env, WORKSPACE_CLOSED_ATTR) is None

    stuck = receipt_env("BF workspace-mode 1755 0 /root\n")
    setattr(stuck, WORKSPACE_CLOSED_ATTR, {"/root": "700"})
    with pytest.raises(RuntimeError, match="Re-opening the workspace"):
        await reopen_granted_workspaces(stuck)


def test_the_user_command_is_unchanged_where_no_uid_is_avoided():
    """The host-uid remap adds nothing to the sandbox user's command on a sandbox whose logs are not the host's
    mount."""
    from benchflow.sandbox import lockdown

    plain = lockdown._create_sandbox_user_cmd("agent")
    assert plain == lockdown._create_sandbox_user_cmd("agent", avoid_uid=None)
    assert "bf_moved" not in plain
    avoided = lockdown._create_sandbox_user_cmd("agent", avoid_uid=1001)
    assert 'bf_ids="${bf_ids}1001 "' in avoided, avoided
