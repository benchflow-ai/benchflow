"""A sandbox user that does not have its workspace stops the trial.

Guards the audit fix on branch ``audit-workspace-grant``
(finding SANDBOX-2, the lead's ruling of 2026-11-13) and the fixes made
after the independent review of PR #143 the same day. The user setup is one
chained command; when it failed and the user existed, the failure was only a
log warning. A failure before the grant skipped the grant, so the agent
started in a workspace it could not write and the trial was graded as its
failure.

After a failed setup step the workspace is now checked. The user cannot write
the root, or an entry under it was never granted: a sandbox startup failure,
retried. Exempt: a workspace root on a read-only mount (a write by root is
refused with EROFS). Tolerated as before: a chown that failed below the root
on a read-only mount.

From the review: any status of a check other than the ones its script sets
stops the trial (it used to count as "read-only"); a root that refuses root
without EROFS (root-squash, a user namespace) is not read-only; and a
world-writable or sticky root says nothing about the files already in it.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import SANDBOX_SETUP, classify_error
from benchflow.evaluation import RetryConfig
from benchflow.sandbox.lockdown import (
    _grant_reach_check,
    _write_check,
    setup_sandbox_user,
)
from benchflow.sandbox.protocol import SandboxStartupError
from tests._receipt_fakes import inner_command, receipt_answer

WRITABLE, READ_ONLY, REFUSED, NOT_GRANTED, MISSING, NO_OWNER = 0, 1, 2, 3, 4, 5
ROOT_CHECK = ("write check", "root")
USER_CHECK = ("write check", "agent")
REACH_CHECK = ("reach check", "root")


def _env(
    *,
    setup_status=1,
    root=WRITABLE,
    user=WRITABLE,
    reach=0,
    workspace="/app",
    reach_stdout=None,
    reach_receipt=True,
):
    """A sandbox whose user-setup command exits *setup_status* and whose
    three workspace checks exit as given. Reads answer with receipts, as a
    sandbox's shell prints them; with *reach_receipt* false the reach check
    is answered as a stand-in sandbox that prints none."""
    calls: list[tuple[str, str | None]] = []

    async def run(sent, **kwargs):
        command = inner_command(sent)
        who = kwargs.get("user")
        answer = MagicMock(return_code=0, stdout="", stderr="")
        if command == _write_check(workspace):
            calls.append(("write check", who))
            answer.return_code = root if who == "root" else user
        elif command == _grant_reach_check(workspace, "agent"):
            calls.append(("reach check", who))
            answer.return_code = reach
            if reach_stdout is not None:
                answer.stdout = reach_stdout
            elif reach == NOT_GRANTED:
                answer.stdout = f"{workspace}/src\n"
            if not reach_receipt:
                return answer
        else:
            calls.append((command, who))
            if "chown -R agent:agent /home/agent" in command:
                answer.return_code = setup_status
                answer.stderr = (
                    "chown: changing ownership of '/app/data': Read-only file system"
                )
        return receipt_answer(sent, answer)

    env = MagicMock()
    env.supports_system_shell = True
    env.exec = run
    return env, calls


def _retried_as_sandbox_setup(error: SandboxStartupError) -> bool:
    """As the lifecycle records it: a sandbox setup failure, retried."""
    recorded = f"Sandbox startup failed: {error}"
    return classify_error(recorded) == SANDBOX_SETUP and RetryConfig().should_retry(
        recorded
    )


@pytest.mark.asyncio
async def test_a_user_who_cannot_write_the_workspace_root_stops_the_trial():
    env, calls = _env(user=REFUSED)

    with pytest.raises(SandboxStartupError) as refused:
        await setup_sandbox_user(env, "agent", "/app")

    text = str(refused.value)
    assert "the sandbox user 'agent' cannot write its workspace /app" in text
    assert "Read-only file system" in text  # the failed step's own output
    assert _retried_as_sandbox_setup(refused.value)
    # Nothing ran after the refusal.
    assert calls[-1] == USER_CHECK


@pytest.mark.asyncio
async def test_a_chown_that_failed_below_a_writable_root_is_still_tolerated():
    env, calls = _env()

    assert await setup_sandbox_user(env, "agent", "/app") == "/app"
    assert [c for c in calls if c in (ROOT_CHECK, USER_CHECK, REACH_CHECK)] == [
        ROOT_CHECK,
        USER_CHECK,
        REACH_CHECK,
    ]


@pytest.mark.asyncio
async def test_a_workspace_on_a_read_only_mount_is_exempt():
    env, calls = _env(root=READ_ONLY, user=REFUSED)

    assert await setup_sandbox_user(env, "agent", "/app") == "/app"
    # The mount is read-only, so the user was not asked.
    assert USER_CHECK not in calls and REACH_CHECK not in calls


@pytest.mark.asyncio
async def test_a_workspace_the_image_lacks_is_left_as_before():
    """Nothing to grant: the setup went on without the folder before this
    check, and tests/test_b32b_kit_without_python_container.py relies on it."""
    env, calls = _env(root=MISSING, user=REFUSED)

    assert await setup_sandbox_user(env, "agent", "/app") == "/app"
    assert USER_CHECK not in calls and REACH_CHECK not in calls


@pytest.mark.asyncio
async def test_a_workspace_the_image_lacks_is_never_skipped_silently(caplog):
    """The lead's ruling of 2026-11-13 on PR #143: tolerated, and recorded in
    the log and in what result.json's ``warnings`` lists."""
    import logging

    from benchflow._utils.trial_warnings import recording_trial_warnings

    env, _ = _env(root=MISSING, user=REFUSED)
    note = "workspace /app absent in image; grant skipped"

    with (
        caplog.at_level(logging.WARNING, logger="benchflow.sandbox.lockdown"),
        recording_trial_warnings() as warnings,
    ):
        await setup_sandbox_user(env, "agent", "/app")

    assert warnings == [note]
    assert note in caplog.text


@pytest.mark.asyncio
async def test_the_rollout_keeps_that_warning_for_result_json():
    """What the two setup calls in ``Rollout.install_agent`` run under."""
    from types import SimpleNamespace

    from benchflow.rollout import _keeping_trial_warnings

    env, _ = _env(root=MISSING, user=REFUSED)
    rollout = SimpleNamespace(_warnings=[])

    with _keeping_trial_warnings(rollout):
        await setup_sandbox_user(env, "agent", "/app")

    assert rollout._warnings == ["workspace /app absent in image; grant skipped"]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["oracle", "claude-agent-acp"])
async def test_install_agent_keeps_the_setup_warning_for_result_json(
    tmp_path, monkeypatch, agent
):
    """Review of PR #143, 2026-11-13: the two sandbox-user setup calls in
    ``Rollout.install_agent``, the scripted agents' (the oracle) and a model
    agent's, each keep the warning the setup reports for result.json. The
    real method runs; only the planes around the setup are stubbed."""
    from benchflow._utils.trial_warnings import report_trial_warning
    from tests.test_trial_install_agent_timeout import _make_trial

    note = "workspace /app absent in image; grant skipped"
    trial = _make_trial(tmp_path, agent=agent, sandbox_setup_timeout=120)
    for name in (
        "install_agent",
        "write_credential_files",
        "upload_subscription_auth",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "deploy_skills",
        "lockdown_paths",
    ):
        monkeypatch.setattr(trial._planes, name, AsyncMock())

    async def setup(*args, **kwargs):
        report_trial_warning(note)
        return "/app"

    setup_sandbox_user = AsyncMock(side_effect=setup)
    monkeypatch.setattr(trial._planes, "setup_sandbox_user", setup_sandbox_user)

    await trial.install_agent()

    setup_sandbox_user.assert_awaited_once()
    # The scripted agents' call names guard_daemon; a model agent's does not.
    called = setup_sandbox_user.await_args.kwargs
    assert ("guard_daemon" in called) is (agent == "oracle")
    assert trial._warnings == [note]


@pytest.mark.asyncio
async def test_a_setup_that_succeeded_runs_no_extra_command():
    env, calls = _env(setup_status=0, user=REFUSED)

    assert await setup_sandbox_user(env, "agent", "/app") == "/app"
    assert not {ROOT_CHECK, USER_CHECK, REACH_CHECK} & set(calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [3, 5, 124, 125, 126, 127, 137, 255, -1])
async def test_a_root_check_that_could_not_run_is_not_read_only(status):
    """Review of PR #143 (S2, fails open): any non-zero status from root's
    check counted as a read-only workspace."""
    env, calls = _env(root=status, user=REFUSED)

    with pytest.raises(SandboxStartupError, match="could not check") as refused:
        await setup_sandbox_user(env, "agent", "/app")

    assert f"exited {status}" in str(refused.value)
    assert _retried_as_sandbox_setup(refused.value)
    assert calls[-1] == ROOT_CHECK


@pytest.mark.asyncio
async def test_a_root_that_refuses_root_without_erofs_is_not_read_only():
    """Review of PR #143: a root-squashed or user-namespace mount refuses
    root and is not read-only. It is judged by what the user can do."""
    env, _ = _env(root=REFUSED, user=REFUSED)
    with pytest.raises(SandboxStartupError, match="cannot write its workspace"):
        await setup_sandbox_user(env, "agent", "/app")

    env, calls = _env(root=REFUSED, user=WRITABLE)
    assert await setup_sandbox_user(env, "agent", "/app") == "/app"
    assert REACH_CHECK in calls


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", ["/app", "/opt"])
async def test_a_writable_root_whose_entries_were_never_granted_stops_the_trial(
    workspace,
):
    """Review of PR #143: a 777 root, or the sticky root a workspace holding
    a reserved folder gets (``/opt``), lets the user create files and says
    nothing about the files the grant never reached."""
    env, calls = _env(reach=NOT_GRANTED, workspace=workspace)

    with pytest.raises(SandboxStartupError) as refused:
        await setup_sandbox_user(env, "agent", workspace)

    assert f"was not given 'src' in its workspace {workspace}" in str(refused.value)
    assert _retried_as_sandbox_setup(refused.value)
    assert calls[-1] == REACH_CHECK


LONG_WORKSPACE = "/" + "w" * 325


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workspace", "printed", "named"),
    [
        ("/app", "/app/nl\nname\n", "'nl\\nname' in its workspace /app"),
        ("/app", "/app/cr\rname\n", "'cr\\rname' in its workspace /app"),
        ("/app", "/app/ff\x0cname\n", "'ff\\x0cname' in its workspace /app"),
        ("/app", "/app/ls\u2028name\n", "'ls\\u2028name' in its workspace /app"),
        ("/app", "/app/" + "x" * 300 + "\n", f"'{'x' * 200}'... in its workspace /app"),
        (
            LONG_WORKSPACE,
            f"{LONG_WORKSPACE}/src\n",
            f"'src' in its workspace {LONG_WORKSPACE}",
        ),
        ("/app", "", "an entry directly under its workspace /app"),
    ],
    ids=[
        "newline",
        "carriage return",
        "form feed",
        "U+2028",
        "longer than 200",
        "326-character workspace",
        "no name",
    ],
)
async def test_an_ungranted_entry_is_named_whole_under_its_workspace(
    workspace, printed, named
):
    """Re-review of PR #143, 2026-11-13: the name is all the check printed
    but its last newline, never cut at a newline, carriage return, form feed
    or U+2028; it is shown under the workspace, cut to 200 characters of its
    own, as a repr on the message's first line. Without one the entry is
    unnamed. The message still names the workspace."""
    env, _ = _env(reach=NOT_GRANTED, reach_stdout=printed, workspace=workspace)

    with pytest.raises(SandboxStartupError) as refused:
        await setup_sandbox_user(env, "agent", workspace)

    first_line = str(refused.value).splitlines()[0]
    assert f"was not given {named}:" in first_line
    assert "x" * 201 not in str(refused.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [WRITABLE, NOT_GRANTED, NO_OWNER])
async def test_the_reach_check_is_read_only_through_its_receipt(status):
    """Re-review of PR #143, 2026-11-13: the reach check's status and the
    entry it names are taken from its receipt. A sandbox that answers with
    no receipt has given no answer: whatever status and name it printed,
    the check could not run, and the trial stops as a sandbox failure."""
    env, calls = _env(reach=status, reach_stdout="/app/src\n", reach_receipt=False)

    with pytest.raises(SandboxStartupError) as refused:
        await setup_sandbox_user(env, "agent", "/app")

    text = str(refused.value)
    assert "could not check" in text and "could not be read" in text
    assert "was not given" not in text
    assert _retried_as_sandbox_setup(refused.value)
    assert calls[-1] == REACH_CHECK


@pytest.mark.asyncio
@pytest.mark.parametrize(("check", "status"), [("user", 127), ("reach", 124)])
async def test_a_later_check_that_could_not_run_stops_the_trial(check, status):
    env, _ = _env(**{check: status})

    with pytest.raises(SandboxStartupError, match=f"could not check.*exited {status}"):
        await setup_sandbox_user(env, "agent", "/app")


def _sh(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", "-c", command], capture_output=True, text=True)


def test_the_write_check_tells_writable_from_refused_and_leaves_nothing(tmp_path):
    """The script itself, under a real sh. EROFS needs a mount, so status 1
    is covered by the script's text below, not run here."""
    assert _sh(_write_check(str(tmp_path))).returncode == WRITABLE
    assert list(tmp_path.iterdir()) == []

    if os.geteuid() != 0:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o555)
        assert _sh(_write_check(str(locked))).returncode == REFUSED
    assert _sh(_write_check(str(tmp_path / "missing"))).returncode == MISSING
    assert '*"Read-only file system"*) exit 1' in _write_check("/app")
    assert _write_check("/app").startswith("LC_ALL=C sh -c ")


def test_the_reach_check_passes_entries_the_user_owns(tmp_path):
    import getpass

    (tmp_path / "src").mkdir()
    (tmp_path / "notes.txt").write_text("unchanged")
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "link").symlink_to("/")

    done = _sh(_grant_reach_check(str(tmp_path), getpass.getuser()))

    assert (done.returncode, done.stdout) == (0, "")
    assert (tmp_path / "notes.txt").read_text() == "unchanged"


@pytest.mark.skipif(os.geteuid() == 0, reason="needs entries root does not own")
def test_the_reach_check_names_an_entry_that_is_not_the_users(tmp_path):
    """Run as an ordinary user and asked about root: the entries are not
    root's and can be written, so the first is named and its file is not
    changed."""
    (tmp_path / "notes.txt").write_text("unchanged")

    done = _sh(_grant_reach_check(str(tmp_path), "root"))

    assert done.returncode == NOT_GRANTED
    assert done.stdout.strip() == str(tmp_path / "notes.txt")
    assert (tmp_path / "notes.txt").read_text() == "unchanged"


def test_the_reach_check_skips_the_folders_the_grant_leaves_roots():
    script = _grant_reach_check("/opt", "agent")

    assert "/opt/" in script and ") continue;; esac" in script
    assert (
        ") continue;; esac"
        not in _grant_reach_check("/app", "agent")
        .split("for e in")[1]
        .split('mine "$e"')[0]
    )


def _tools(tmp_path, *names: str, find_exits: int | None = None):
    """A PATH holding only *names* from this machine, and a ``find`` that
    exits *find_exits* when given: an image whose find is missing or cannot
    answer ``-user``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("sh", "rm", *names):
        (bin_dir / name).symlink_to(shutil.which(name))
    if find_exits is not None:
        fake = bin_dir / "find"
        fake.write_text(
            f"#!/bin/sh\necho 'find: unrecognized: -user' >&2\nexit {find_exits}\n"
        )
        fake.chmod(0o755)
    return {"PATH": str(bin_dir)}


def _reach(workspace, user: str, env) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shutil.which("sh"), "-c", _grant_reach_check(str(workspace), user)],
        capture_output=True,
        text=True,
        env=env,
    )


def _entries(tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "notes.txt").write_text("unchanged")
    (workspace / ".hidden").write_text("x")
    return workspace


@pytest.mark.parametrize("find_exits", [None, 1], ids=["no find", "find refuses"])
def test_the_reach_check_reads_owners_with_ls_when_find_cannot(tmp_path, find_exits):
    """Review of PR #143, 2026-11-13: without a find that answers ``-user``
    the check exited 0, so an image without one passed whatever it held.
    The owner is read from ``ls -n`` instead."""
    workspace = _entries(tmp_path)
    env = _tools(tmp_path, "ls", "id", find_exits=find_exits)

    done = _reach(workspace, getpass.getuser(), env)
    assert (done.returncode, done.stdout) == (0, "")

    if os.geteuid() != 0:
        done = _reach(workspace, "root", env)
        assert done.returncode == NOT_GRANTED
        assert (workspace / "notes.txt").read_text() == "unchanged"


@pytest.mark.parametrize("find_exits", [None, 1], ids=["no find", "find refuses"])
def test_the_reach_check_that_can_read_no_owner_says_so(tmp_path, find_exits):
    """Review of PR #143, 2026-11-13: neither find nor ls can read who owns
    an entry, so the check exits 5, a status it sets for nothing else."""
    workspace = _entries(tmp_path)
    env = _tools(tmp_path, "id", find_exits=find_exits)

    assert _reach(workspace, getpass.getuser(), env).returncode == NO_OWNER
    assert (workspace / "notes.txt").read_text() == "unchanged"


@pytest.mark.asyncio
async def test_a_reach_check_that_could_read_no_owner_stops_the_trial():
    """Review of PR #143, 2026-11-13: the check's 5 is one it could not run."""
    env, calls = _env(reach=NO_OWNER)

    with pytest.raises(SandboxStartupError, match=r"could not check.*exited 5") as e:
        await setup_sandbox_user(env, "agent", "/app")

    assert _retried_as_sandbox_setup(e.value)
    assert calls[-1] == REACH_CHECK
