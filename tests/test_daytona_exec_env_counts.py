"""Exec calls and uploads of one fake Daytona trial whose task has an env.

The trial of ``tests/test_daytona_fake_trial_exec_counts.py`` (the
deterministic task, the oracle, the real ``Rollout`` and ``DaytonaSandbox``
against its Docker-backed fake of the Daytona API), with a sandbox env and a
verifier env added to the task. Every exec then carries the sandbox env (the
verifier's own env goes in its root-only file while the task's service runs).
No value is in any session command or BenchFlow's log;
each distinct env is one upload into the sandbox's env folder, and the
staging folder's preparation before the first is one exec.

Set ``BENCHFLOW_EXEC_ENV_COUNT_REPORT`` to a file path to keep the counts as
JSON. Docker and the ``sandbox-daytona`` extra are needed; without them the
test skips.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h
from tests.test_daytona_exec_env_file import ENV_FOLDERS, leaks
from tests.test_daytona_fake_trial_exec_counts import (
    _Counts,
    _FakeClient,
    _FakeFs,
    _FakeProcess,
    _FakeSandbox,
    _manager_for,
    fake_daytona_image,  # noqa: F401 (the fixture)
)

pytestmark = [
    pytest.mark.deterministic,
    pytest.mark.xdist_group("fake_daytona_image"),
]

REPORT_ENV = "BENCHFLOW_EXEC_ENV_COUNT_REPORT"
SANDBOX_VALUE = "placeholder-sandbox-setting-6b2a"
VERIFIER_VALUE = "placeholder-verifier-setting-d41c"

# This trial's exec calls with the env written into each command (measured on
# the code before env files: 78, and 9 uploads, none of them an env).
EXECS_WITH_INLINE_ENV = 78


def _with_env(task: Path) -> None:
    doc = task / "task.md"
    text = doc.read_text()
    text = text.replace(
        "verifier:\n  type: test-script\n",
        f"verifier:\n  type: test-script\n  env:\n    JUDGE_SETTING: {VERIFIER_VALUE}\n",
        1,
    )
    text = text.replace(
        "sandbox:\n",
        f"sandbox:\n  env:\n    TASK_SETTING: {SANDBOX_VALUE}\n",
        1,
    )
    assert SANDBOX_VALUE in text and VERIFIER_VALUE in text
    doc.write_text(text)


async def test_a_fake_trial_with_an_env_uploads_each_env_once(
    fake_daytona_image,  # noqa: F811
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox import daytona as daytona_module

    caplog.set_level(logging.DEBUG)
    destinations: list[str] = []
    bodies: list[bytes] = []
    upload = _FakeFs.upload_file

    async def recorded(self, source, destination, timeout=None, **kwargs):
        destinations.append(destination)
        bodies.append(
            source if isinstance(source, bytes) else Path(source).read_bytes()
        )
        return await upload(self, source, destination, timeout, **kwargs)

    events: list[str] = []
    real_delete_file = _FakeFs.delete_file

    async def delete_file(self, path, recursive=False):
        events.append(f"delete {path}")
        return await real_delete_file(self, path, recursive)

    real_delete = _FakeSandbox.delete
    left_at_delete: list[str] = []

    async def sandbox_delete(self, timeout=60, wait=False):
        if "sandbox delete" not in events:
            # What the box holds when the stop deletes it.
            done = subprocess.run(
                ["docker", "exec", self.id, "ls", "-A", "/opt/runtime/run"],
                capture_output=True,
                text=True,
            )
            left_at_delete.extend(
                n for n in done.stdout.split() if n.startswith("env-")
            )
        events.append("sandbox delete")
        return await real_delete(self, timeout=timeout, wait=wait)

    monkeypatch.setattr(_FakeFs, "upload_file", recorded)
    monkeypatch.setattr(_FakeFs, "delete_file", delete_file)
    monkeypatch.setattr(_FakeSandbox, "delete", sandbox_delete)
    counts = _Counts()
    client = _FakeClient(counts)
    monkeypatch.setattr(daytona_module, "DaytonaClientManager", _manager_for(client))
    task = tmp_path / "deterministic"
    shutil.copytree(h.TEMPLATE_TASK, task)
    _with_env(task)
    try:
        result = await Rollout(
            RolloutConfig(
                task_path=task,
                agent="oracle",
                environment="daytona",
                jobs_dir=tmp_path / "jobs",
                job_name="exec-env-counts",
                rollout_name="deterministic__env",
            )
        ).run()
    finally:
        for sandbox in client.sandboxes:
            await sandbox.delete()
    env_files = [d for d in destinations if d.startswith(ENV_FOLDERS)]
    summary = {
        **counts.summary(),
        "env_file_uploads": len(env_files),
        "env_folders": len({d.rsplit("/", 1)[0] for d in env_files}),
    }
    report = os.environ.get(REPORT_ENV)
    if report:
        Path(report).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    assert result.error is None, result.error
    assert result.rewards == {"reward": 1.0}
    texts = [e.command for e in counts.execs]
    texts += [caplog.text, *(r.getMessage() for r in caplog.records)]
    assert leaks(texts, [SANDBOX_VALUE, VERIFIER_VALUE], bodies) == []
    # Each value went up as data: the sandbox env in the env files, the
    # verifier's in its own root-only file (agent code is alive at grading:
    # the task's service).
    for value in (SANDBOX_VALUE, VERIFIER_VALUE):
        assert any(value.encode() in b for b in bodies), value
    # The sandbox env alone, as plumbing (its loader names dropped) and as
    # the task's own route, and with the verifier's env: a handful of files,
    # each uploaded once, in one folder.
    assert 1 <= len(env_files) <= 6, env_files
    assert len(set(env_files)) == len(env_files)
    assert summary["env_folders"] == 1
    # The stop removes the env folder (one file-API delete) before the
    # sandbox's delete, and the box holds none then.
    (folder,) = {d.rsplit("/", 1)[0] for d in env_files}
    assert f"delete {folder}" in events, events
    assert events.index(f"delete {folder}") < events.index("sandbox delete"), events
    assert left_at_delete == [], left_at_delete
    assert events.count(f"delete {folder}") == 1, events
    # One exec more than with the env inline: the staging folder's
    # preparation before the first upload.
    assert summary["exec_calls"] == EXECS_WITH_INLINE_ENV + 1, summary


@pytest.mark.parametrize("timed_out", [False, True], ids=["passes", "times-out"])
async def test_the_verifier_envs_file_goes_when_its_command_ends(
    fake_daytona_image,  # noqa: F811
    monkeypatch,
    tmp_path,
    timed_out,
) -> None:
    """With no sandbox user the verifier's ``[verifier.env]`` is its exec's
    env, so a file in the sandbox: it is removed right after the verifier's
    command (also one that timed out), before the trial's next exec, not kept
    until stop. No later exec names the removed folder or exits 126 (a dead
    path read as a command's status); the next exec with an env uploads it
    into a new folder."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox import daytona as daytona_module

    uploads: list[tuple[str, bytes, int]] = []
    deletes: list[tuple[str, int]] = []
    upload = _FakeFs.upload_file
    real_delete_file = _FakeFs.delete_file
    counts = _Counts()

    async def recorded(self, source, destination, timeout=None, **kwargs):
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        uploads.append((destination, data, len(counts.execs)))
        return await upload(self, source, destination, timeout, **kwargs)

    async def delete_file(self, path, recursive=False):
        deletes.append((path, len(counts.execs)))
        return await real_delete_file(self, path, recursive)

    monkeypatch.setattr(_FakeFs, "upload_file", recorded)
    monkeypatch.setattr(_FakeFs, "delete_file", delete_file)
    client = _FakeClient(counts)
    monkeypatch.setattr(daytona_module, "DaytonaClientManager", _manager_for(client))
    task = tmp_path / "deterministic"
    shutil.copytree(h.TEMPLATE_TASK, task)
    _with_env(task)
    if timed_out:
        doc = task / "task.md"
        text = doc.read_text()
        assert "  timeout_sec: 60.0\n" in text  # the verifier's, first
        doc.write_text(text.replace("  timeout_sec: 60.0\n", "  timeout_sec: 3.0\n", 1))
        test_sh = task / "tests" / "test.sh"
        shebang, _, rest = test_sh.read_text().partition("\n")
        test_sh.write_text(f"{shebang}\nsleep 30\n{rest}")
    try:
        result = await Rollout(
            RolloutConfig(
                task_path=task,
                agent="oracle",
                environment="daytona",
                jobs_dir=tmp_path / "jobs",
                job_name="exec-env-verifier",
                rollout_name="deterministic__verifier_env",
                sandbox_user=None,
            )
        ).run()
    finally:
        for sandbox in client.sandboxes:
            await sandbox.delete()
    if timed_out:
        assert (result.rewards or {}).get("reward") != 1.0, result.rewards
    else:
        assert result.error is None, result.error
        assert result.rewards == {"reward": 1.0}
    (path,) = [
        p
        for p, data, _ in uploads
        if p.startswith(ENV_FOLDERS) and VERIFIER_VALUE.encode() in data
    ]
    folder = path.rsplit("/", 1)[0]
    (test_exec,) = [i for i, e in enumerate(counts.execs) if f"8<{path}" in e.command]
    removed = [after for p, after in deletes if p == folder]
    # Right after the command: a timed-out verify runs one exec of its own
    # first (the deadline's check), and no exec opens the verifier's file.
    assert removed, deletes
    between = counts.execs[test_exec + 1 : removed[0]]
    assert len(between) <= 1, [e.caller for e in between]
    assert not [
        e.caller for e in counts.execs[test_exec + 1 :] if f"8<{path}" in e.command
    ]
    # After the removal no exec names the removed folder or exits 126, and
    # the next exec with an env uploads it into a new folder.
    after = counts.execs[removed[0] :]
    assert after and not [e.caller for e in after if folder in e.command]
    assert not [e.caller for e in counts.execs[test_exec + 1 :] if e.exit_code == 126]
    assert [
        p
        for p, _, at in uploads
        if p.startswith(ENV_FOLDERS) and at >= removed[0] and not p.startswith(folder)
    ], "no exec after the removal uploaded its env afresh"
    assert leaks([e.command for e in counts.execs], [VERIFIER_VALUE]) == []


ROOT_ENV_PARENT = "/var/lib/runtime/verifier-env"


async def test_a_verifier_command_refused_before_it_starts_leaves_no_root_env_file(
    fake_daytona_image,  # noqa: F811
    monkeypatch,
    tmp_path,
) -> None:
    """With agent code alive at grading the verifier env goes in a root-only
    file the test command removes when it starts. A command refused before it
    starts (here its exec fails) leaves none: not at the trial's next exec,
    not when the sandbox goes (where a snapshot would have kept it)."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox import daytona as daytona_module

    marker = f"if . {ROOT_ENV_PARENT}/.runtime-verifier-env-"
    execute = _FakeProcess.execute_session_command
    refused: list[str] = []
    # (command, the root env folders there before it ran) after the refusal.
    after: list[tuple[str, list[str]]] = []

    def listing(sandbox_id: str) -> list[str]:
        done = subprocess.run(
            ["docker", "exec", sandbox_id, "ls", "-A", ROOT_ENV_PARENT],
            capture_output=True,
            text=True,
        )
        return [
            n for n in done.stdout.split() if n.startswith(".runtime-verifier-env-")
        ]

    from benchflow.sandbox.daytona import _SESSION_SHELL_PROBE

    async def refusing(self, session_id, request, timeout=None):
        if marker in request.command:
            refused.append(request.command)
            raise RuntimeError("the exec was refused before it started")
        # A kept session's shell probe comes before its exec; it is not one.
        if refused and request.command != _SESSION_SHELL_PROBE:
            after.append((request.command, listing(self._sandbox.id)))
        return await execute(self, session_id, request, timeout)

    real_delete = _FakeSandbox.delete
    at_delete: list[list[str]] = []

    async def sandbox_delete(self, timeout=60, wait=False):
        if not at_delete:
            at_delete.append(listing(self.id))
        return await real_delete(self, timeout=timeout, wait=wait)

    monkeypatch.setattr(_FakeProcess, "execute_session_command", refusing)
    monkeypatch.setattr(_FakeSandbox, "delete", sandbox_delete)
    counts = _Counts()
    client = _FakeClient(counts)
    monkeypatch.setattr(daytona_module, "DaytonaClientManager", _manager_for(client))
    task = tmp_path / "deterministic"
    shutil.copytree(h.TEMPLATE_TASK, task)
    _with_env(task)
    try:
        result = await Rollout(
            RolloutConfig(
                task_path=task,
                agent="oracle",
                environment="daytona",
                jobs_dir=tmp_path / "jobs",
                job_name="exec-env-root-file",
                rollout_name="deterministic__root_env_refused",
            )
        ).run()
    finally:
        for sandbox in client.sandboxes:
            await sandbox.delete()
    assert refused, "the test command never came"
    assert (result.rewards or {}).get("reward") != 1.0
    # The exec right after the refusal removes the folder; every later one,
    # and the sandbox's delete, finds none.
    assert (
        after
        and f"/bin/rm -rf -- {ROOT_ENV_PARENT}/.runtime-verifier-env-" in (after[0][0])
    ), after[:1]
    assert [names for _, names in after[1:] if names] == [], after
    assert at_delete == [[]], at_delete
