"""Separate verifier custody: only the declared transfer crosses, and the
record says what did (PY8 SV1).

A separate verifier receives the frozen workspace, the declared artifacts and
``/logs/artifacts``. These tests guard the holes that were still open in
that promise: files the agent's run left in the trial's ``verifier/`` (on
Docker the agent container's live ``/logs/verifier`` mount) surviving
publish or blocking it, a ``/logs/artifacts`` file hashed and then read
again, transfers into folders the verifier owns, an incomplete record, and
workspace recovery re-grading a separate trial outside the separate path.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import pathlib
import shlex
import shutil
import tarfile
import threading
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error
from benchflow.rollout._separate_verifier import (
    build_transfer_payload,
    run_separate_verifier,
)
from benchflow.task import TaskConfig, validate_task_runtime_support
from tests.test_separate_verifier import (
    FakeVerifierSandbox,
    LocalTransport,
    _agent_trial,
    _fake_rollout,
    _members,
    _task_dir,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sv"
RECORD = Path("verifier-sandbox") / "verifier-sandbox.json"
SEPARATE = '[verifier]\nenvironment_mode = "separate"\n'
SEPARATE_INELIGIBLE = (
    "separate verifier trials are re-graded only by the separate verifier"
)


def _record(trial: Path) -> dict[str, Any]:
    return json.loads((trial / RECORD).read_text())


def _writes_reward(value: str):
    """A verifier that writes only reward.txt, as the fixtures' test.sh does."""

    async def verify(env, vtask, vpaths, timing, *, workspace, agent_paths):
        vpaths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (vpaths.verifier_dir / "reward.txt").write_text(value + "\n")
        timing["verifier"] = 0.1
        return {"reward": float(value)}, None

    return verify


async def _separate_trial(tmp_path: Path):
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, SEPARATE)
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    box = FakeVerifierSandbox(tmp_path / "verifier-box")
    return trial, rollout, box


def _planted(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    listing = record["planted_in_verifier_dir"]
    return {entry["name"]: entry for entry in listing["entries"]}


# --- 1. publish by swap -------------------------------------------------------


@pytest.mark.asyncio
async def test_planted_verifier_files_are_moved_aside_not_published(
    tmp_path: Path, monkeypatch
) -> None:
    """The trial's verifier/ held what the agent's run planted there (on
    Docker it is the agent container's /logs/verifier), and publish merged the
    verifier's outputs into it: a planted reward.json survived, job export
    showed it, and the task.md timed_out recorder kept its reward. Publish now
    swaps a staged copy in and moves the agent's folder aside, listed."""
    from benchflow.jobs import VerifierOutput
    from benchflow.rollout import Rollout

    trial, rollout, box = await _separate_trial(tmp_path)
    planted = trial / "verifier"
    planted.mkdir(exist_ok=True)
    (planted / "reward.json").write_text('{"reward": 1}\n')
    (planted / "extra.txt").write_text("left by the agent\n")

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("0")
    )

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert sorted(p.name for p in (trial / "verifier").iterdir()) == ["reward.txt"]
    record = _record(trial)
    listed = _planted(record)
    assert set(listed) == {"reward.json", "extra.txt"}
    assert listed["reward.json"]["type"] == "file"
    assert (
        listed["reward.json"]["sha256"]
        == hashlib.sha256(b'{"reward": 1}\n').hexdigest()
    )
    assert listed["reward.json"]["bytes"] == len('{"reward": 1}\n')
    assert record["planted_in_verifier_dir"]["total"] == 2
    moved = trial / record["planted_in_verifier_dir"]["moved_to"]
    assert moved == trial / "verifier-sandbox" / "agent-verifier-dir"
    assert (moved / "reward.json").is_file()
    published = {e["name"]: e for e in record["published_verifier_files"]["entries"]}
    assert published["reward.txt"]["sha256"] == hashlib.sha256(b"0\n").hexdigest()

    # Job export reads only what the verifier wrote.
    output = VerifierOutput.read(trial / "verifier")
    assert output.reward_json is None and output.reward_text == "0"

    # The task.md timed_out recorder starts from the verifier's reward.
    monkeypatch.setattr(
        "benchflow.taskmd.materialize.taskmd_metadata", lambda path: {"draft": 2}
    )
    stand_in = SimpleNamespace(
        _rollout_paths=rollout._rollout_paths,
        _rewards=rewards,
        _config=SimpleNamespace(task_path=rollout._config.task_path),
        _agent_ran_out_of_time=False,
    )
    Rollout._record_taskmd_timed_out(stand_in)  # type: ignore[arg-type]
    recorded = json.loads((trial / "verifier" / "reward.json").read_text())
    assert recorded == {"reward": 0.0, "timed_out": False}


@pytest.mark.asyncio
async def test_a_writer_still_running_after_publish_never_reaches_verifier_dir(
    tmp_path: Path,
) -> None:
    """A process left running in the agent's container keeps writing through
    its live mount of /logs/verifier. A directory handle opened before publish
    stands in for that mount: its later writes land in the moved-aside folder,
    never in the published verifier/."""
    trial, rollout, box = await _separate_trial(tmp_path)
    (trial / "verifier").mkdir(exist_ok=True)
    mount = os.open(trial / "verifier", os.O_RDONLY | os.O_DIRECTORY)
    try:
        await run_separate_verifier(
            rollout, create_environment=lambda *a: box, verify=_writes_reward("0")
        )
        late = os.open("late.txt", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=mount)
        os.write(late, b"written after publish\n")
        os.close(late)
    finally:
        os.close(mount)

    assert not (trial / "verifier" / "late.txt").exists()
    moved = trial / "verifier-sandbox" / "agent-verifier-dir"
    assert (moved / "late.txt").read_text() == "written after publish\n"


@pytest.mark.asyncio
async def test_unreadable_plants_do_not_fail_publish_and_are_listed(
    tmp_path: Path,
) -> None:
    """A planted file the host cannot read and a planted folder it cannot
    list or remove: publish still succeeds, and both are listed."""
    trial, rollout, box = await _separate_trial(tmp_path)
    planted = trial / "verifier"
    planted.mkdir(exist_ok=True)
    (planted / "secret.txt").write_text("hidden\n")
    (planted / "private").mkdir()
    (planted / "private" / "inner.txt").write_text("hidden\n")
    (planted / "secret.txt").chmod(0o000)
    (planted / "private").chmod(0o000)
    try:
        rewards, error = await run_separate_verifier(
            rollout, create_environment=lambda *a: box, verify=_writes_reward("0")
        )
        record = _record(trial)
    finally:
        moved = trial / "verifier-sandbox" / "agent-verifier-dir"
        for path in (moved / "private", planted / "private"):
            if path.exists():
                path.chmod(0o755)
        for path in (moved / "secret.txt", planted / "secret.txt"):
            if path.exists():
                path.chmod(0o644)

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert sorted(p.name for p in (trial / "verifier").iterdir()) == ["reward.txt"]
    listed = _planted(record)
    assert listed["private"]["type"] == "directory"
    assert listed["secret.txt"]["type"] == "file"
    if os.geteuid() != 0:  # root reads mode-000 files and folders
        assert listed["private"]["unreadable"] is True
        assert listed["secret.txt"]["unreadable"] is True
        assert "sha256" not in listed["secret.txt"]


@pytest.mark.asyncio
async def test_a_plant_named_like_a_verifier_output_no_longer_unscores_a_zero(
    tmp_path: Path,
) -> None:
    """A root-owned reward.txt planted in verifier/ cannot be overwritten by
    the host, so the merge publish raised after the rewards were set: a trial
    scored 0 became unscored (verifier_infra, retryable). With the swap the
    plant blocks nothing; the trial scores 0 and the plant is listed. Run as
    root (which overwrites mode 444) the plant is a folder of that name."""
    trial, rollout, box = await _separate_trial(tmp_path)
    planted = trial / "verifier"
    planted.mkdir(exist_ok=True)
    if os.geteuid() == 0:
        (planted / "reward.txt").mkdir()
    else:
        (planted / "reward.txt").write_text("1\n")
        (planted / "reward.txt").chmod(0o444)

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("0")
    )

    assert error is None, error
    assert rewards == {"reward": 0.0}
    assert (trial / "verifier" / "reward.txt").read_text() == "0\n"
    record = _record(trial)
    assert record["status"] == "complete"
    assert "reward.txt" in _planted(record)


@pytest.mark.asyncio
async def test_a_folder_the_agent_made_unmovable_is_moved_beside_it(
    tmp_path: Path,
) -> None:
    """Root in the agent's container can take the host's write permission off
    the mounted folder; then it cannot move into verifier-sandbox/ (another
    parent), so it is renamed inside the trial folder instead."""
    trial, rollout, box = await _separate_trial(tmp_path)
    planted = trial / "verifier"
    planted.mkdir(exist_ok=True)
    (planted / "reward.json").write_text('{"reward": 1}\n')
    planted.chmod(0o555)
    try:
        rewards, error = await run_separate_verifier(
            rollout, create_environment=lambda *a: box, verify=_writes_reward("0")
        )
    finally:
        for path in (planted, trial / "agent-verifier-dir"):
            if path.exists():
                path.chmod(0o755)
        moved_in = trial / "verifier-sandbox" / "agent-verifier-dir"
        if moved_in.exists():
            moved_in.chmod(0o755)

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert sorted(p.name for p in (trial / "verifier").iterdir()) == ["reward.txt"]
    record = _record(trial)
    expected = (
        "verifier-sandbox/agent-verifier-dir"
        if os.geteuid() == 0  # root moves it anyway
        else "agent-verifier-dir"
    )
    assert record["planted_in_verifier_dir"]["moved_to"] == expected
    assert "reward.json" in _planted(record)


@pytest.mark.asyncio
async def test_a_failed_staging_still_moves_the_agents_folder_aside(
    tmp_path: Path, monkeypatch
) -> None:
    """Verifier outputs the host cannot copy keep today's unscored error, but
    verifier/ is left empty and the agent's files are moved aside."""
    from benchflow.rollout import _separate_verifier

    trial, rollout, box = await _separate_trial(tmp_path)
    (trial / "verifier").mkdir(exist_ok=True)
    (trial / "verifier" / "reward.json").write_text('{"reward": 1}\n')

    def refuse(*args, **kwargs):
        raise PermissionError("verifier output is not readable")

    monkeypatch.setattr(_separate_verifier.shutil, "copytree", refuse)
    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert rewards is None
    assert error is not None and error.startswith("separate verifier sandbox failed")
    assert classify_verifier_error(error) == VERIFIER_INFRA
    assert list((trial / "verifier").iterdir()) == []
    assert "reward.json" in _planted(_record(trial))


@pytest.mark.asyncio
async def test_a_trial_the_verifier_never_ran_for_has_no_planted_verifier_files(
    tmp_path: Path,
) -> None:
    """An unscored trial (here: no verifier image builds) does not leave the
    agent's plants where job export reads verifier outputs."""
    trial, rollout, box = await _separate_trial(tmp_path)
    (trial / "verifier").mkdir(exist_ok=True)
    (trial / "verifier" / "reward.json").write_text('{"reward": 1}\n')

    async def boom(force_build: bool = False) -> None:
        raise RuntimeError("image build failed")

    box.start = boom  # type: ignore[method-assign]
    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert rewards is None and error is not None
    assert list((trial / "verifier").iterdir()) == []
    assert "reward.json" in _planted(_record(trial))


@pytest.mark.asyncio
@pytest.mark.parametrize("verifier_runs", [True, False])
async def test_publish_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch, verifier_runs: bool
) -> None:
    """Publish lists and hashes what the agent left in /logs/verifier. On the
    event loop, an agent's large plant froze every other trial in the job
    until their timeouts fired at once (SV1-R1-A-1). Both call sites, the
    scored one and the one for a verifier that never ran, now run it in a
    worker thread."""
    from benchflow.rollout import _separate_verifier

    trial, rollout, box = await _separate_trial(tmp_path)
    (trial / "verifier").mkdir(exist_ok=True)
    (trial / "verifier" / "reward.json").write_text('{"reward": 1}\n')
    loop_thread = threading.get_ident()
    threads: list[int] = []
    publish = _separate_verifier._publish_verifier_outputs

    def recording(*args, **kwargs):
        threads.append(threading.get_ident())
        return publish(*args, **kwargs)

    monkeypatch.setattr(_separate_verifier, "_publish_verifier_outputs", recording)
    if not verifier_runs:

        async def boom(force_build: bool = False) -> None:
            raise RuntimeError("image build failed")

        box.start = boom  # type: ignore[method-assign]
    await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("0")
    )

    assert len(threads) == 1
    assert threads[0] != loop_thread
    assert "reward.json" in _planted(_record(trial))


@pytest.mark.asyncio
async def test_a_cancel_during_publish_waits_for_it_before_the_record(
    tmp_path: Path, monkeypatch
) -> None:
    """A trial cancelled while publish runs in its thread still writes a
    record that lists what was moved aside, and the published verifier/:
    the record is not written while the thread is still changing it."""
    from benchflow.rollout import _separate_verifier

    trial, rollout, box = await _separate_trial(tmp_path)
    (trial / "verifier").mkdir(exist_ok=True)
    (trial / "verifier" / "reward.json").write_text('{"reward": 1}\n')
    started, release = threading.Event(), threading.Event()
    publish = _separate_verifier._publish_verifier_outputs

    def slow(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return publish(*args, **kwargs)

    monkeypatch.setattr(_separate_verifier, "_publish_verifier_outputs", slow)
    run = asyncio.ensure_future(
        run_separate_verifier(
            rollout, create_environment=lambda *a: box, verify=_writes_reward("0")
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 10)
        run.cancel()
        await asyncio.sleep(0.05)
        assert not run.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await run

    record = _record(trial)
    assert record["status"] == "interrupted"
    assert "reward.json" in _planted(record)
    assert sorted(p.name for p in (trial / "verifier").iterdir()) == ["reward.txt"]


def test_a_listing_reads_only_the_files_it_keeps(tmp_path: Path, monkeypatch) -> None:
    """Only the first entries are kept in the record, and only those are
    read: an agent's 100,000 plants are counted, not hashed."""
    from benchflow.rollout import _separate_verifier

    for i in range(10):
        (tmp_path / f"f{i:02}").write_bytes(b"x" * 100)
    calls: list[str] = []
    digest = _separate_verifier._file_digest

    def counting(dir_fd, name, *args, **kwargs):
        calls.append(name)
        return digest(dir_fd, name, *args, **kwargs)

    monkeypatch.setattr(_separate_verifier, "_LIST_LIMIT", 3)
    monkeypatch.setattr(_separate_verifier, "_file_digest", counting)
    listed = _separate_verifier._list_tree(tmp_path)

    assert listed["total"] == 10
    assert [e["name"] for e in listed["entries"]] == ["f00", "f01", "f02"]
    assert calls == ["f00", "f01", "f02"]
    assert all("sha256" in e for e in listed["entries"])


def test_a_listing_hashes_at_most_its_budget(tmp_path: Path, monkeypatch) -> None:
    """Each file had its own 64 MiB cap but the listing had none, so sparse
    plants made the host hash terabytes. Past the listing's budget a file
    gets its size only, and the listing says the budget ran out."""
    from benchflow.rollout import _separate_verifier

    for name in ("a", "b", "c", "d"):
        (tmp_path / name).write_bytes(name.encode() * 1000)
    (tmp_path / "e").write_bytes(b"")
    monkeypatch.setattr(_separate_verifier, "_HASH_BUDGET", 2500, raising=False)
    listed = _separate_verifier._list_tree(tmp_path)

    entries = {e["name"]: e for e in listed["entries"]}
    assert entries["a"]["sha256"] == hashlib.sha256(b"a" * 1000).hexdigest()
    assert entries["b"]["sha256"] == hashlib.sha256(b"b" * 1000).hexdigest()
    for name in ("c", "d"):
        assert entries[name] == {"name": name, "type": "file", "bytes": 1000}
    assert entries["e"]["sha256"] == hashlib.sha256(b"").hexdigest()
    assert listed["hash_budget_exhausted"] is True
    assert "truncated" not in listed


def test_a_listing_under_its_budget_is_not_marked(tmp_path: Path) -> None:
    from benchflow.rollout import _separate_verifier

    (tmp_path / "a").write_bytes(b"a" * 1000)
    listed = _separate_verifier._list_tree(tmp_path)

    assert "sha256" in listed["entries"][0]
    assert "hash_budget_exhausted" not in listed


# --- 2. /logs/artifacts is read once ------------------------------------------


@pytest.mark.asyncio
async def test_a_logs_artifact_is_packed_from_the_bytes_that_were_hashed(
    tmp_path: Path, monkeypatch
) -> None:
    """The file was hashed with one read and packed with a second; a process
    still writing through the bind mount could change it in between, and the
    changed bytes crossed under the old hash. The second open here rewrites
    the file first, as such a writer would."""
    _, _, trial = await _agent_trial(tmp_path)
    target = trial / "artifacts" / "mode.txt"
    opened: list[int] = []
    real_open = pathlib.Path.open

    def racing_open(self, *args, **kwargs):
        if Path(self) == target:
            opened.append(1)
            if len(opened) == 2:
                target.write_bytes(b"swapped!\n")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", racing_open)
    try:
        build_transfer_payload(trial, tmp_path / "payload.tar")
    except Exception as exc:  # a mismatch the packer noticed is also safe
        assert "changed since collection" in str(exc)
        return
    packed = _members(tmp_path / "payload.tar")["logs/artifacts/mode.txt"]
    assert packed == b"separate\n"


@pytest.mark.asyncio
async def test_a_logs_artifact_swapped_for_a_symlink_is_not_followed(
    tmp_path: Path,
) -> None:
    """After collection the file is replaced by a symlink to a host file with
    the same bytes. It was followed and its target packed; it is now refused
    as changed, and nothing is packed."""
    from benchflow.rollout._separate_verifier import SeparateVerifierError

    _, _, trial = await _agent_trial(tmp_path)
    host_file = tmp_path / "host-file.txt"
    host_file.write_text("separate\n")
    target = trial / "artifacts" / "mode.txt"
    target.unlink()
    target.symlink_to(host_file)

    with pytest.raises(SeparateVerifierError, match="changed since collection"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.asyncio
async def test_a_logs_artifact_under_a_swapped_folder_is_not_followed(
    tmp_path: Path,
) -> None:
    """The same for a folder on the way swapped for a symlink."""
    from benchflow.rollout._separate_verifier import SeparateVerifierError

    _, _, trial = await _agent_trial(tmp_path)
    nested = trial / "artifacts" / "sub"
    nested.mkdir()
    (nested / "x.txt").write_text("x\n")
    manifest_path = trial / "artifacts-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append(
        {
            "collection": 0,
            "kind": "file",
            "path": "sub/x.txt",
            "size": 2,
            "sha256": hashlib.sha256(b"x\n").hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "x.txt").write_text("x\n")
    shutil.rmtree(nested)
    nested.symlink_to(elsewhere)

    with pytest.raises(SeparateVerifierError, match="changed since collection"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.asyncio
async def test_packing_is_timed_as_verifier_pack(tmp_path: Path) -> None:
    trial, rollout, box = await _separate_trial(tmp_path)
    await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )
    assert rollout._timing["verifier_pack"] >= 0
    assert "verifier_pack" in _record(trial)["timing"]


# --- 3. verifier-owned paths ---------------------------------------------------


def _relocate_artifact(trial: Path, sandbox_path: str) -> None:
    """Rewrite the captured manifests as if the declared artifact (the trial's
    only one) had been captured at ``sandbox_path`` in the agent sandbox."""
    main = trial / "evidence" / "manifest.json"
    data = json.loads(main.read_text())
    data["artifacts"][0]["source"] = sandbox_path
    main.write_text(json.dumps(data))
    sub = trial / "evidence" / data["artifacts"][0]["bundle_path"] / "manifest.json"
    sub_data = json.loads(sub.read_text())
    sub_data["workspace"] = str(Path(sandbox_path).parent)
    sub.write_text(json.dumps(sub_data))


async def _artifact_trial(tmp_path: Path, sandbox_path: str):
    """A separate trial whose one declared artifact sits at ``sandbox_path``
    in the agent sandbox (declared by a run overlay: the task.toml has none)."""
    box_file = tmp_path / "box" / "planted" / Path(sandbox_path).name
    box_file.parent.mkdir(parents=True)
    box_file.write_text("import sys\nsys.exit(0)\n")
    _, workspace, trial = await _agent_trial(tmp_path, artifacts=[str(box_file)])
    _relocate_artifact(trial, sandbox_path)
    task = _task_dir(tmp_path, SEPARATE)
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._task.config.artifacts = [sandbox_path]
    box = FakeVerifierSandbox(tmp_path / "verifier-box")
    return trial, rollout, box


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sandbox_path", "owned"),
    [("/tests/conftest.py", "/tests"), ("/verifier/x", "/verifier")],
)
async def test_an_overlay_artifact_in_a_verifier_owned_folder_never_crosses(
    tmp_path: Path, sandbox_path: str, owned: str
) -> None:
    """A declared artifact under /tests (or /verifier) became a tar member
    there and was unpacked where tests/ is then uploaded without clearing:
    a conftest.py the verifier's pytest loads. Declared by an overlay, which
    the launch gate does not see, it is refused at transfer time, unscored."""
    trial, rollout, box = await _artifact_trial(tmp_path, sandbox_path)

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / sandbox_path.lstrip("/")).exists()
    assert rewards is None
    assert error == (
        f"separate verifier transfer failed: {sandbox_path} overlaps "
        f"verifier-owned {owned}"
    )
    assert classify_verifier_error(error) == VERIFIER_INFRA
    assert _record(trial)["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_a_logs_workspace_carrying_logs_agent_never_crosses(
    tmp_path: Path,
) -> None:
    """A workspace of /logs (a declared workdir, or the image's WORKDIR, which
    only the started sandbox knows) carried /logs/agent/x into the verifier
    sandbox. It is refused at transfer time, unscored."""
    _, workspace, trial = await _agent_trial(tmp_path)
    (workspace / "agent").mkdir()
    (workspace / "agent" / "x").write_text("planted\n")
    shutil.rmtree(trial / "evidence")
    from benchflow.review.evidence import capture_task_evidence
    from tests.test_separate_verifier import LocalTransport

    await capture_task_evidence(LocalTransport(), str(workspace), trial / "evidence")
    manifest = json.loads((trial / "evidence" / "manifest.json").read_text())
    manifest["workspace"] = "/logs"
    (trial / "evidence" / "manifest.json").write_text(json.dumps(manifest))
    task = _task_dir(tmp_path, SEPARATE)
    rollout = _fake_rollout(tmp_path, trial, task, "/logs")
    verifier = FakeVerifierSandbox(tmp_path / "verifier-box")

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: verifier, verify=_writes_reward("1")
    )

    assert not (verifier.root / "logs" / "agent" / "x").exists()
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: /logs overlaps verifier-owned "
        "/logs/verifier"
    )
    assert _record(trial)["status"] == "transfer_failed"


@pytest.mark.parametrize(
    ("path", "owned"),
    [
        ("/tests", "/tests"),
        ("/tests/conftest.py", "/tests"),
        ("/verifier/x", "/verifier"),
        ("/logs/verifier", "/logs/verifier"),
        ("/logs/verifier/reward.txt", "/logs/verifier"),
        ("/logs/agent/x", "/logs/agent"),
        ("/logs", "/logs/verifier"),
        ("/", "/tests"),
        ("/_runtime_verifier_env", "/_runtime_verifier_*"),
        ("/_runtime_guard_123/x", "/_runtime_guard_*"),
        ("/app/../tests/x", "/tests"),
        ("//tests", "/tests"),
        ("/app", None),
        ("/logs/artifacts/x", None),
        ("/testsuite", None),
        ("/tmp/tests", None),
        ("relative/tests", None),
    ],
)
def test_verifier_owned_overlap(path: str, owned: str | None) -> None:
    from benchflow.task.verifier_sandbox import verifier_owned_overlap

    assert verifier_owned_overlap(path) == owned


def _gate(config: TaskConfig, task_dir: Path | None = None) -> list[Any]:
    return [
        issue
        for issue in validate_task_runtime_support(
            config, sandbox="docker", task_dir=task_dir
        )
        if issue.path == "verifier.sandbox.transfer"
    ]


@pytest.mark.parametrize(
    ("artifact", "owned"),
    [
        ("/tests/conftest.py", "/tests"),
        ("/verifier/x", "/verifier"),
        ("/logs/verifier/x", "/logs/verifier"),
        ("/logs/agent/x", "/logs/agent"),
        ({"source": "/tests", "destination": "t"}, "/tests"),
    ],
)
def test_a_declared_artifact_in_a_verifier_owned_folder_is_refused_at_launch(
    artifact: Any, owned: str
) -> None:
    config = TaskConfig.model_validate(
        {"verifier": {"sandbox_mode": "separate"}, "artifacts": [artifact]}
    )
    [issue] = _gate(config)
    assert f"overlaps verifier-owned {owned!r}" in issue.reason
    assert issue.reason.startswith("artifacts[0]")


@pytest.mark.parametrize(
    ("workdir", "owned"),
    [
        ("/logs", "/logs/verifier"),
        ("/tests", "/tests"),
        ("/logs/agent/w", "/logs/agent"),
    ],
)
def test_a_declared_workdir_in_a_verifier_owned_folder_is_refused_at_launch(
    workdir: str, owned: str
) -> None:
    config = TaskConfig.model_validate(
        {"verifier": {"sandbox_mode": "separate"}, "sandbox": {"workdir": workdir}}
    )
    [issue] = _gate(config)
    assert issue.reason.startswith(f"sandbox.workdir {workdir!r}")
    assert f"overlaps verifier-owned {owned!r}" in issue.reason


def test_a_relative_artifact_is_judged_under_the_declared_workdir() -> None:
    config = TaskConfig.model_validate(
        {
            "verifier": {"sandbox_mode": "separate"},
            "sandbox": {"workdir": "/logs/agent"},
            "artifacts": ["out.txt"],
        }
    )
    reasons = [issue.reason for issue in _gate(config)]
    assert any("'/logs/agent/out.txt'" in reason for reason in reasons)


def test_shared_mode_and_ordinary_paths_are_not_refused() -> None:
    assert not _gate(
        TaskConfig.model_validate(
            {"artifacts": ["/tests/x"], "sandbox": {"workdir": "/logs"}}
        )
    )
    assert not _gate(
        TaskConfig.model_validate(
            {
                "verifier": {"sandbox_mode": "separate"},
                "sandbox": {"workdir": "/app"},
                "artifacts": ["/tmp/out.txt", "out.txt", "/logs/artifacts/x"],
            }
        )
    )


def test_strict_integrity_prints_the_artifact_and_workdir_refusals() -> None:
    """--integrity strict forces the separate verifier and prints the launch
    gate's separate-mode refusals; the new ones are among them."""
    from benchflow.integrity.trial import strict_launch_issues

    config = TaskConfig.model_validate(
        {"artifacts": ["/tests/conftest.py"], "sandbox": {"workdir": "/logs"}}
    )
    lines = strict_launch_issues(config, sandbox="docker", task_dir=None)
    assert any(
        line.startswith("verifier.sandbox.transfer: artifacts[0] '/tests/conftest.py'")
        for line in lines
    ), lines
    assert any(
        line.startswith("verifier.sandbox.transfer: sandbox.workdir '/logs'")
        for line in lines
    ), lines


def test_the_fixture_declaring_a_tests_artifact_is_refused_at_launch() -> None:
    from benchflow.task import Task

    task = FIXTURES / "tamper-tests-artifact"
    issues = _gate(Task(task).config, task)
    assert [i.reason.split(";")[0] for i in issues] == [
        "artifacts[0] '/tests/conftest.py' overlaps verifier-owned '/tests'"
    ]


def test_the_image_workdir_fixture_passes_launch() -> None:
    """WORKDIR /logs in the image and no [sandbox].workdir: only the started
    sandbox knows the workspace, so the launch gate lets it through and the
    transfer refuses it (as a recorded Docker run shows)."""
    from benchflow.task import Task

    task = FIXTURES / "workdir-logs"
    config = Task(task).config
    assert config.sandbox.workdir is None
    assert "WORKDIR /logs" in (task / "environment" / "Dockerfile").read_text()
    assert validate_task_runtime_support(config, sandbox="docker", task_dir=task) == []


@pytest.mark.parametrize(
    "name",
    [
        "answer-42",
        "root-file-artifact",
        "root-file-artifact-python",
        "tamper-ancestor-link",
        "tamper-mount",
        "tamper-mount-collide",
        "tamper-system-folder",
        "tamper-verifier-ancestor",
        "workdir-logs",
    ],
)
def test_the_runnable_fixtures_pass_launch(name: str) -> None:
    from benchflow.task import Task

    task = FIXTURES / name
    assert (
        validate_task_runtime_support(
            Task(task).config, sandbox="docker", task_dir=task
        )
        == []
    )


@pytest.mark.asyncio
async def test_the_image_workdir_overlap_is_unscored_at_transfer_and_not_recovered(
    tmp_path: Path,
) -> None:
    """The workspace came from the image's WORKDIR (/logs): the transfer
    refuses it, and recovery does not re-grade the trial elsewhere."""
    from benchflow.rollout._verifier_recovery import (
        needs_verifier_recovery,
        recovery_ineligible_reason,
    )

    _, _, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, SEPARATE)
    rollout = _fake_rollout(tmp_path, trial, task, "/logs")
    created: list[Any] = []

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: created.append(a), verify=None
    )

    assert rewards is None and not created
    assert error is not None and error.startswith(
        "separate verifier transfer failed: /logs overlaps verifier-owned"
    )
    assert needs_verifier_recovery(error)
    assert recovery_ineligible_reason(rollout) == SEPARATE_INELIGIBLE


@pytest.mark.asyncio
async def test_an_artifact_the_agent_pointed_into_tests_scores_zero(
    tmp_path: Path,
) -> None:
    """Capture resolves a declared artifact in the agent's sandbox, so a
    symlink on its way (/app/out -> /tests) moved it into /tests, where the
    plant crossed. Resolved elsewhere before the agent ran, that is the
    solution's doing: it scores 0 as refused. Without the clean control it is
    unscored."""
    trial, rollout, box = await _artifact_trial(tmp_path, "/tests/conftest.py")
    rollout._task.config.artifacts = ["/app/out/conftest.py"]
    rollout._agent_cwd = "/app"
    rollout._pristine_outputs = {
        "workspace": "/app",
        "logs": "/logs/artifacts",
        "paths": {},
        "resolved": {
            "/app": "/app",
            "/app/out/conftest.py": "/app/out/conftest.py",
        },
    }

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "tests" / "conftest.py").exists()
    assert (rewards, error) == ({"reward": 0.0}, None)
    record = _record(trial)
    assert record["status"] == "refused"
    assert "/app/out/conftest.py resolves to /tests/conftest.py" in record["refusal"]

    # The same with no clean control: unscored.
    _, rollout2, box2 = await _artifact_trial(tmp_path / "b", "/tests/conftest.py")
    rollout2._task.config.artifacts = ["/app/out/conftest.py"]
    rollout2._agent_cwd = "/app"
    rewards, error = await run_separate_verifier(
        rollout2, create_environment=lambda *a: box2, verify=_writes_reward("1")
    )
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: /tests/conftest.py overlaps "
        "verifier-owned /tests"
    )


class _SwapAfterDiscovery(LocalTransport):
    """The agent sandbox, where a leftover process swaps ``swap`` for a link
    to ``link_to`` after artifact discovery and before capture reads it."""

    def __init__(self, swap: Path, link_to: Path) -> None:
        self.swap, self.link_to = swap, link_to

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
        from benchflow.review.evidence import _ARTIFACT_PROBE_SCRIPT

        result = await super().exec(cmd, user=user, timeout_sec=timeout_sec)
        # On PT's protected Python the probe is inside root_python_sh's
        # -c text, after its own prefix (C8), and since
        # integrity-on-PT inside a compile() of its repr, after the version
        # probe (review.evidence._isolated_python).
        probe = any(
            _ARTIFACT_PROBE_SCRIPT in token or repr(_ARTIFACT_PROBE_SCRIPT) in token
            for token in shlex.split(cmd)
        )
        if probe and self.swap.is_dir():
            self.swap.rename(self.swap.with_name(self.swap.name + "-before"))
            self.swap.symlink_to(self.link_to, target_is_directory=True)
        return result


def _as_sandbox_paths(trial: Path, box: Path) -> None:
    """Rewrite every path the captured manifests record from ``box/...`` to
    ``/...``, as if ``box`` were the agent sandbox's root. Discovery's path
    and capture's root are rewritten alike, so a split between them stays."""
    prefix = str(box.resolve())

    def rebased(path: str) -> str:
        assert path == prefix or path.startswith(prefix + "/"), path
        return "/" + path[len(prefix) :].lstrip("/")

    main = trial / "evidence" / "manifest.json"
    data = json.loads(main.read_text())
    data["workspace"] = rebased(data["workspace"])
    for artifact in data["artifacts"]:
        artifact["source"] = rebased(artifact["source"])
        if artifact["bundle_path"] is not None:
            sub = trial / "evidence" / artifact["bundle_path"] / "manifest.json"
            sub_data = json.loads(sub.read_text())
            sub_data["workspace"] = rebased(sub_data["workspace"])
            sub.write_text(json.dumps(sub_data))
    main.write_text(json.dumps(data))


async def _raced_trial(tmp_path: Path, declared: str, link_to: str | None):
    """A separate trial declaring ``declared`` (a file), captured for real
    in a sandbox rooted at ``box/``; with ``link_to``, the declared file's
    folder becomes a link to it between discovery and capture."""
    from benchflow.review.evidence import capture_task_evidence

    box, _, trial = await _agent_trial(tmp_path)
    shutil.rmtree(trial / "evidence")
    file = box / declared.lstrip("/")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("the agent's honest output\n")
    # In the agent sandbox's own /tests: the trusted helper's name, planted.
    (box / "tests" / file.name).write_text("echo 1 > /logs/verifier/reward.txt\n")
    (box / "elsewhere").mkdir()
    (box / "elsewhere" / file.name).write_text("another file\n")
    env = (
        LocalTransport()
        if link_to is None
        else _SwapAfterDiscovery(file.parent, box / link_to.lstrip("/"))
    )
    await capture_task_evidence(
        env, str(box / "app"), trial / "evidence", artifacts=[str(file)]
    )
    _as_sandbox_paths(trial, box)
    task = _task_dir(tmp_path, SEPARATE)
    rollout = _fake_rollout(tmp_path, trial, task, "/app")
    rollout._task.config.artifacts = [declared]
    rollout._pristine_outputs = {
        "workspace": "/app",
        "logs": "/logs/artifacts",
        "paths": {},
        "resolved": {"/app": "/app", declared: declared},
    }
    return trial, rollout, FakeVerifierSandbox(tmp_path / "verifier-box")


@pytest.mark.asyncio
async def test_a_folder_swapped_into_tests_after_discovery_scores_zero(
    tmp_path: Path,
) -> None:
    """Discovery resolved /out/helper.sh; a leftover root process then made
    /out a link to /tests, so capture read /tests and the bundle's root
    became /tests. The overlap check saw only discovery's path and packed
    tests/helper.sh, a planted helper the trusted test.sh sources: a wrong
    answer could score 1. Each packed member is now checked, and the
    declared path resolved elsewhere before the agent ran, so it scores 0."""
    trial, rollout, box = await _raced_trial(tmp_path, "/out/helper.sh", "/tests")
    manifest = json.loads((trial / "evidence" / "manifest.json").read_text())
    [artifact] = manifest["artifacts"]
    sub = trial / "evidence" / artifact["bundle_path"] / "manifest.json"
    # The split the race leaves: discovery's path, and capture's root.
    assert artifact["source"] == "/out/helper.sh"
    assert json.loads(sub.read_text())["workspace"] == "/tests"

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "tests" / "helper.sh").exists()
    assert (rewards, error) == ({"reward": 0.0}, None)
    record = _record(trial)
    assert record["status"] == "refused"
    assert record["refusal"] == (
        "the solution's outputs cannot reach the verifier: "
        "/out/helper.sh resolves to /tests/helper.sh, which overlaps "
        "verifier-owned /tests; before the agent ran it resolved to "
        "/out/helper.sh"
    )


@pytest.mark.asyncio
async def test_a_folder_swapped_after_discovery_without_a_clean_control_is_unscored(
    tmp_path: Path,
) -> None:
    trial, rollout, box = await _raced_trial(tmp_path, "/out/helper.sh", "/tests")
    rollout._pristine_outputs = None

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "tests" / "helper.sh").exists()
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: /tests/helper.sh overlaps "
        "verifier-owned /tests"
    )
    assert _record(trial)["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_a_folder_swapped_elsewhere_after_discovery_is_unscored(
    tmp_path: Path,
) -> None:
    """Swapped to a folder no verifier owns: what would cross is not what
    discovery found, so the transfer fails unscored, as any change during
    capture does."""
    trial, rollout, box = await _raced_trial(tmp_path, "/out/helper.sh", "/elsewhere")

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "elsewhere").exists()
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: declared artifact /out/helper.sh "
        "changed between discovery and capture (captured under /elsewhere)"
    )
    assert _record(trial)["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_a_single_file_artifact_captured_at_the_root_still_crosses(
    tmp_path: Path,
) -> None:
    """A declared file in / is captured with / as its root, which contains
    every verifier-owned path: the file itself is what is judged."""
    _, rollout, box = await _raced_trial(tmp_path, "/result.txt", None)

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert (box.root / "result.txt").read_text() == "the agent's honest output\n"


# The grader the verifier image trusts, and the agent's stand-in for it: a
# marker that only writes a file, so a packed replacement shows if it ran.
_PLANTED_GRADER = "#!/bin/sh\necho ran > /logs/verifier/planted-grader-ran\necho 1\n"


async def _linked_trial(
    tmp_path: Path,
    declared: str,
    *,
    files: dict[str, str],
    links: dict[str, str],
    resolved: dict[str, str] | None,
):
    """A separate trial declaring ``declared``, captured for real in a sandbox
    rooted at ``box/`` after the agent wrote ``files`` and made each of
    ``links`` (what was there moved aside) a symlink to its value, before
    discovery. ``resolved`` is the clean control's resolution of each
    declared path (None: no clean control)."""
    from benchflow.review.evidence import capture_task_evidence

    box, _, trial = await _agent_trial(tmp_path)
    shutil.rmtree(trial / "evidence")
    for path, text in files.items():
        file = box / path.lstrip("/")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
        file.chmod(0o755)
    for path, to in links.items():
        link = box / path.lstrip("/")
        if link.exists():
            link.rename(link.with_name(link.name + "-before"))
        link.symlink_to(box / to.lstrip("/"), target_is_directory=True)
    await capture_task_evidence(
        LocalTransport(),
        str(box / "app"),
        trial / "evidence",
        artifacts=[str(box / declared.lstrip("/"))],
    )
    _as_sandbox_paths(trial, box)
    task = _task_dir(tmp_path, SEPARATE)
    rollout = _fake_rollout(tmp_path, trial, task, "/app")
    rollout._task.config.artifacts = [declared]
    rollout._pristine_outputs = (
        None
        if resolved is None
        else {
            "workspace": "/app",
            "logs": "/logs/artifacts",
            "paths": {},
            "resolved": {"/app": "/app", **resolved},
        }
    )
    return trial, rollout, FakeVerifierSandbox(tmp_path / "verifier-box")


@pytest.mark.asyncio
async def test_an_ancestor_the_agent_linked_to_usr_scores_zero(
    tmp_path: Path,
) -> None:
    """The task declares the folder /out/bin. Before discovery the agent made
    /out a link to /usr, so discovery and capture both read /usr/bin, which
    no verifier owns: the agent's /usr/bin was packed over the verifier
    image's, and the planted grader test.sh trusts ran (a wrong answer could
    score 1). A packed root must land where the task declared it; it
    resolved there before the agent ran, so this scores 0."""
    trial, rollout, box = await _linked_trial(
        tmp_path,
        "/out/bin",
        files={"/usr/bin/sv-grade": _PLANTED_GRADER, "/out/bin/.keep": ""},
        links={"/out": "/usr"},
        resolved={"/out/bin": "/out/bin"},
    )
    manifest = json.loads((trial / "evidence" / "manifest.json").read_text())
    [artifact] = manifest["artifacts"]
    assert artifact["status"] == "captured"
    assert artifact["source"] == "/usr/bin"

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "usr" / "bin" / "sv-grade").exists()
    assert (rewards, error) == ({"reward": 0.0}, None)
    record = _record(trial)
    assert record["status"] == "refused"
    assert record["refusal"] == (
        "the solution's outputs cannot reach the verifier: declared artifact "
        "/out/bin resolves to /usr/bin; before the agent ran it resolved to "
        "/out/bin"
    )


@pytest.mark.asyncio
async def test_an_ancestor_linked_to_usr_without_a_clean_control_is_unscored(
    tmp_path: Path,
) -> None:
    trial, rollout, box = await _linked_trial(
        tmp_path,
        "/out/bin",
        files={"/usr/bin/sv-grade": _PLANTED_GRADER},
        links={"/out": "/usr"},
        resolved=None,
    )

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "usr").exists()
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: declared artifact /out/bin "
        "resolves to /usr/bin, not where it was declared"
    )
    assert _record(trial)["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_an_image_link_into_a_system_folder_is_unscored(
    tmp_path: Path,
) -> None:
    """/out already pointed at /usr before the agent ran: the task's image
    did it, not the solution, so the trial is unscored, not scored 0, and
    still nothing lands in /usr/bin."""
    trial, rollout, box = await _linked_trial(
        tmp_path,
        "/out/bin",
        files={"/usr/bin/sv-grade": _PLANTED_GRADER},
        links={"/out": "/usr"},
        resolved={"/out/bin": "/usr/bin"},
    )

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "usr").exists()
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: declared artifact /out/bin lands "
        "/usr/bin in system folder /usr/bin"
    )
    assert _record(trial)["status"] == "transfer_failed"
    assert classify_verifier_error(error) == VERIFIER_INFRA


@pytest.mark.asyncio
async def test_a_workspace_the_agent_linked_to_a_system_folder_scores_zero(
    tmp_path: Path,
) -> None:
    """The workspace is captured where it resolves, too: /app made a link to
    /usr/local/bin would lay the agent's files over (and the install script
    empty) the verifier image's /usr/local/bin."""
    trial, rollout, box = await _linked_trial(
        tmp_path,
        "/out/result.txt",
        files={
            "/usr/local/bin/sv-grade": _PLANTED_GRADER,
            "/out/result.txt": "42\n",
        },
        links={"/app": "/usr/local/bin"},
        resolved={"/out/result.txt": "/out/result.txt"},
    )
    trusted = box.root / "usr" / "local" / "bin" / "sv-grade"
    trusted.parent.mkdir(parents=True)
    trusted.write_text("#!/bin/sh\necho 0\n")

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert trusted.read_text() == "#!/bin/sh\necho 0\n"
    assert not any("usr/local/bin" in c for c in box.commands)
    assert (rewards, error) == ({"reward": 0.0}, None)
    assert _record(trial)["refusal"] == (
        "the solution's outputs cannot reach the verifier: workspace /app "
        "resolves to /usr/local/bin; before the agent ran it resolved to /app"
    )


@pytest.mark.asyncio
async def test_a_folder_the_task_declares_in_a_system_folder_is_unscored(
    tmp_path: Path,
) -> None:
    """Declared as is, a folder in a system folder would still replace every
    tool there with the agent's copy: the task's doing, unscored."""
    trial, rollout, box = await _linked_trial(
        tmp_path,
        "/usr/local/bin",
        files={"/usr/local/bin/sv-grade": _PLANTED_GRADER},
        links={},
        resolved={"/usr/local/bin": "/usr/local/bin"},
    )

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "usr").exists()
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: declared artifact /usr/local/bin "
        "lands /usr/local/bin in system folder /usr/local/bin"
    )
    assert _record(trial)["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_the_one_file_a_task_declares_in_a_system_folder_crosses(
    tmp_path: Path,
) -> None:
    """A task that has the agent install one tool and declares exactly that
    file names what may cross: it still does, and only it."""
    _, rollout, box = await _linked_trial(
        tmp_path,
        "/usr/local/bin/tool",
        files={"/usr/local/bin/tool": "#!/bin/sh\necho 42\n", "/usr/local/bin/x": ""},
        links={},
        resolved={"/usr/local/bin/tool": "/usr/local/bin/tool"},
    )

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert sorted(p.name for p in (box.root / "usr" / "local" / "bin").iterdir()) == [
        "tool"
    ]


@pytest.mark.asyncio
async def test_an_image_link_outside_system_folders_still_crosses(
    tmp_path: Path,
) -> None:
    """/out pointing at /data before the agent ran is the image's own layout:
    the folder crosses where it resolved then."""
    _, rollout, box = await _linked_trial(
        tmp_path,
        "/out/bin",
        files={"/data/bin/result.txt": "42\n"},
        links={"/out": "/data"},
        resolved={"/out/bin": "/data/bin"},
    )

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert (box.root / "data" / "bin" / "result.txt").read_text() == "42\n"


@pytest.mark.asyncio
async def test_the_trajectory_still_reaches_logs_agent(
    tmp_path: Path, monkeypatch
) -> None:
    """/logs/agent is verifier-owned for the transfer only: the published
    trajectory still goes there, and the record keeps its source and hash."""
    from benchflow.rollout import _setup

    trial, rollout, box = await _separate_trial(tmp_path)
    rollout._trajectory = [{"type": "agent_message", "content": "done"}]
    rollout._trajectory_source = "acp"

    # Integration 363d815b publishes through a private stage. Execute the
    # real publication commands against the fake sandbox's /logs tree so
    # this checks the final bytes, not the upload API's intermediate path.
    original_exec = box.exec

    async def publish_exec(command, **kwargs):
        if command == "mkdir -p /logs/agent" or "/logs/.runtime-publish-" in command:
            local = command.replace("/logs/", shlex.quote(str(box.root / "logs")) + "/")
            process = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-c",
                local,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return SimpleNamespace(
                return_code=process.returncode,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
            )
        return await original_exec(command, **kwargs)

    monkeypatch.setattr(box, "exec", publish_exec)

    async def fake_verify_rollout(env, task, paths, timing, planes, **kwargs):
        paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (paths.verifier_dir / "reward.txt").write_text("1\n")
        return {"reward": 1.0}, None, None

    monkeypatch.setattr(_setup, "_verify_rollout", fake_verify_rollout)
    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=None
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    published = box.root / "logs/agent/acp_trajectory.jsonl"
    sent = published.read_bytes()
    assert json.loads(sent) == rollout._trajectory[0]
    trajectory = _record(trial)["custody"]["trajectory"]
    assert trajectory == {
        "source": "acp",
        "path": "/logs/agent/acp_trajectory.jsonl",
        "bytes": len(sent),
        "sha256": hashlib.sha256(sent).hexdigest(),
    }


@pytest.mark.asyncio
async def test_an_include_path_crosses_as_capture_declared_it(tmp_path: Path) -> None:
    """Found regrading a benchmark's trials: a
    --separate-verifier-include path no declared artifact holds never crossed.

    Capture reads it after the task's artifacts, but the transfer was judged
    against the task's artifacts alone, so every such trial failed "captured
    artifact ... has no declaration": the root check (1f0b858e7) and the
    include capture (916636747, #1188) landed apart.
    """
    include = str(tmp_path / "box" / "tmp")
    _, workspace, trial = await _agent_trial(tmp_path, artifacts=[include])
    task = _task_dir(tmp_path, SEPARATE)
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._config.separate_verifier_include = (include,)
    box = FakeVerifierSandbox(tmp_path / "verifier-box")

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert (box.root / include.lstrip("/") / "agent-only.txt").is_file()
    assert _record(trial)["status"] == "complete"


# --- 4. the custody record ---------------------------------------------------


@pytest.mark.asyncio
async def test_the_record_says_exactly_what_crossed(tmp_path: Path) -> None:
    """The tar's sha256 was computed and dropped, and nothing recorded which
    members crossed or whether the sandbox checked the sha256."""
    declared = str(tmp_path / "box" / "tmp" / "configured.txt")
    box, workspace, trial = await _agent_trial(tmp_path, artifacts=[declared])
    task = _task_dir(tmp_path, SEPARATE)
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    # As capture read it: the task declares what crosses.
    rollout._task.config.artifacts = [declared]
    sent: dict[str, bytes] = {}
    verifier = FakeVerifierSandbox(tmp_path / "verifier-box")
    upload = verifier.upload_file

    async def keep(source, target, **kwargs):
        sent[target] = Path(source).read_bytes()
        await upload(source, target, **kwargs)

    verifier.upload_file = keep  # type: ignore[method-assign]
    real_exec = verifier.exec

    async def exec_reports_check(cmd: str, **kwargs):
        result = await real_exec(cmd, **kwargs)
        if "sha256sum -c" in cmd:
            return result.__class__(0, "benchflow-transfer-sha256=checked\n", "")
        return result

    verifier.exec = exec_reports_check  # type: ignore[method-assign]

    await run_separate_verifier(
        rollout, create_environment=lambda *a: verifier, verify=_writes_reward("1")
    )

    custody = _record(trial)["custody"]
    [tar_bytes] = [data for target, data in sent.items() if target.endswith(".tar")]
    assert custody["tar_sha256"] == hashlib.sha256(tar_bytes).hexdigest()
    assert custody["tar_bytes"] == len(tar_bytes)
    assert custody["sha256_checked_in_sandbox"] is True

    evidence = json.loads((trial / "evidence" / "manifest.json").read_text())
    sub = json.loads(
        (trial / "evidence" / "artifacts" / "0000" / "manifest.json").read_text()
    )
    collected = json.loads((trial / "artifacts-manifest.json").read_text())
    logs_files = sorted(
        f["path"]
        for f in collected["files"]
        if f["collection"] == 0 and f["kind"] == "file"
    )
    by_root = {group["root"]: group for group in custody["members"]}
    # A task.toml verifier built from tests/Dockerfile gets only the declared
    # artifacts: this one lies outside the workspace, so nothing of the
    # workspace crosses (test_separate_verifier_format_transfer.py).
    assert _record(trial)["transfer"]["artifacts_only"] is True
    assert evidence["entries"] and str(workspace) not in by_root
    assert set(by_root) == {str(box / "tmp"), "/logs/artifacts"}
    assert by_root[str(box / "tmp")]["members"] == sorted(
        e["path"] for e in sub["entries"]
    )
    assert by_root["/logs/artifacts"]["members"] == logs_files
    for group in by_root.values():
        assert group["count"] == len(group["members"])
        assert (
            group["members_sha256"]
            == hashlib.sha256("\n".join(group["members"]).encode()).hexdigest()
        )
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        names = sorted(m.name for m in tar.getmembers())
    listed = sorted(
        (group["root"].strip("/") + "/" + name)
        for group in by_root.values()
        for name in group["members"]
    )
    assert names == listed


def test_the_install_script_reports_whether_it_checked_the_sha256(
    tmp_path: Path,
) -> None:
    import subprocess

    from benchflow.rollout._separate_verifier import _install_script

    archive = tmp_path / "t.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("app/a.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"a\n"))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    root = tmp_path / "root"
    root.mkdir()
    script = _install_script(str(archive), "/app", digest, root=str(root))
    found = subprocess.run(
        ["sh", "-c", script], capture_output=True, text=True, check=True
    )
    assert found.stdout.splitlines()[-1] == "benchflow-transfer-sha256=checked"
    # An image without sha256sum (PATH without coreutils) says so.
    archive2 = tmp_path / "t2.tar"
    with tarfile.open(archive2, "w") as tar:
        info = tarfile.TarInfo("app/b.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"b\n"))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("sh", "tar", "rm", "mkdir", "gzip"):
        found_tool = shutil.which(tool)
        if found_tool:
            (bindir / tool).symlink_to(found_tool)
    script = _install_script(
        str(archive2),
        "/app",
        hashlib.sha256(archive2.read_bytes()).hexdigest(),
        root=str(root),
    )
    skipped = subprocess.run(
        [str(bindir / "sh"), "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": str(bindir)},
    )
    assert skipped.stdout.splitlines()[-1] == "benchflow-transfer-sha256=skipped"


# --- 5. recovery ---------------------------------------------------------------


def _recovery_task(tmp_path: Path, mode: str) -> Path:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n'
        f"[verifier]\nworkspace_recovery = true\n{mode}"
        '[sandbox]\ndocker_image = "python@sha256:' + "a" * 64 + '"\n'
    )
    (task / "instruction.md").write_text("Produce a file.")
    return task


@pytest.mark.asyncio
async def test_a_separate_trial_is_not_regraded_by_workspace_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    """A separate task with workspace_recovery = true and a failed transfer
    (verifier_infra) was re-graded by recover_verifier in a sandbox from the
    agent's baseline: another image, no custody record. Now it is not, and
    the reason is recorded."""
    from benchflow._utils.task_authoring import task_digest
    from benchflow.models import RolloutResult
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.rollout import _verifier_recovery as recovery
    from benchflow.task import Task

    task = _recovery_task(tmp_path, 'sandbox_mode = "separate"\n')
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "solver.json").write_text("{}")
    error = (
        "separate verifier transfer failed: /logs/artifacts/x changed since collection"
    )
    called: list[Any] = []

    async def fake_recover(rollout):
        called.append(rollout)
        return {"reward": 1.0}, None

    async def noop(rollout):
        return None

    monkeypatch.setattr(recovery, "recover_verifier", fake_recover)
    monkeypatch.setattr(recovery, "release_recovery_lease", noop)
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(task_path=task, task_digest=task_digest(task))
    rollout._task = Task(task)
    rollout._rollout_dir = trial
    rollout._review_plan = None
    rollout._phase = "cleaned"
    rollout._rewards, rollout._verifier_error = None, error

    def build_result(result_filename=None):
        return RolloutResult(
            task_name="task",
            rollout_name="task__a",
            rewards=rollout._rewards,
            verifier_error=rollout._verifier_error,
        )

    rollout._build_result = build_result  # type: ignore[method-assign]
    result = await Rollout._finish_scoring_locked(rollout, build_result())

    assert called == []
    assert result.rewards is None and result.verifier_error == error
    assert recovery.recovery_ineligible_reason(rollout) == (SEPARATE_INELIGIBLE)


def test_a_trial_with_a_separate_verifier_record_is_not_recoverable(
    tmp_path: Path,
) -> None:
    """Strict integrity forces separate mode on the rollout's config; the
    record on disk marks the trial separate whatever the task file says."""
    from benchflow._utils.task_authoring import task_digest
    from benchflow.rollout import RolloutConfig
    from benchflow.rollout import _verifier_recovery as recovery
    from benchflow.task import Task

    task = _recovery_task(tmp_path, "")
    trial = tmp_path / "trial"
    (trial / "verifier-sandbox").mkdir(parents=True)
    rollout = SimpleNamespace(
        _config=RolloutConfig(task_path=task, task_digest=task_digest(task)),
        _task=Task(task),
        _rollout_dir=trial,
    )
    assert recovery.recovery_ineligible_reason(rollout) is None
    (trial / "verifier-sandbox" / "verifier-sandbox.json").write_text("{}")
    assert recovery.recovery_ineligible_reason(rollout) == (SEPARATE_INELIGIBLE)


async def _regrade_job(tmp_path: Path, toml: str, *, config: dict[str, Any]):
    """A finished trial with a frozen workspace, scored 1, of a task whose
    agent image (environment/Dockerfile, python:3.12-slim) and verifier image
    (tests/Dockerfile, ubuntu:24.04) differ."""
    from benchflow.review.evidence import capture_workspace

    tasks = tmp_path / "tasks"
    _task_dir(tasks, toml)
    trial = tmp_path / "job" / "task__a"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps({"task_path": "task", "environment": "docker", **config})
    )
    (trial / "result.json").write_text(
        json.dumps(
            {"task_name": "task", "rewards": {"reward": 1.0}, "verifier_error": None}
        )
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("42\n")
    await capture_workspace(LocalTransport(), str(workspace), trial / "evidence")
    return trial, tasks


def _no_regrade_sandbox(monkeypatch) -> list[Any]:
    """Record (and stop) every regrade that gets as far as its sandbox."""
    from benchflow.rollout import Rollout

    started: list[Any] = []

    async def setup(self, *, verifier_only=False) -> None:
        started.append(self)
        raise RuntimeError("the regrade built a sandbox")

    monkeypatch.setattr(Rollout, "setup", setup)
    return started


async def _saved_separate_trial(tmp_path: Path, toml: str, config: dict[str, Any]):
    """A finished trial's folder as a separate verifier run saves it, scored 1:
    the frozen workspace, one declared artifact and ``/logs/artifacts``
    (``_agent_trial``). Its task's agent image (environment/Dockerfile,
    python:3.12-slim) and verifier image (tests/Dockerfile, ubuntu:24.04)
    differ."""
    declared = str(tmp_path / "box" / "tmp" / "configured.txt")
    # The task.toml format transfers only declared artifacts (abcf2781c).
    # This regrade fixture intentionally tests both the workspace and the
    # external file, so its contract must name both.
    artifacts = [str(tmp_path / "box" / "app"), declared]
    _, workspace, saved = await _agent_trial(tmp_path, artifacts=artifacts)
    tasks = tmp_path / "tasks"
    _task_dir(tasks, f"artifacts = {json.dumps(artifacts)}\n{toml}")
    trial = tmp_path / "job" / "task__a"
    trial.parent.mkdir()
    saved.rename(trial)
    (trial / "config.json").write_text(
        json.dumps({"task_path": "task", "environment": "docker", **config})
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "agent": "codex",
                "rewards": {"reward": 1.0},
                "verifier_error": None,
            }
        )
    )
    return trial, tasks, workspace


def _verifier_sandbox_only(monkeypatch, root: Path) -> tuple[list, list]:
    """Fail any start of the regrade's own (agent environment) sandbox, and
    give the separate verifier a local one; returns (agent starts, verifier
    build contexts)."""
    from benchflow.rollout import Rollout, _separate_verifier

    started: list[Any] = []
    contexts: list[Path] = []

    async def start(self) -> None:
        started.append(self)
        raise RuntimeError("the regrade started a sandbox from the agent's image")

    def create(rollout):
        def make(vtask, context_path, vpaths):
            contexts.append(context_path)
            return FakeVerifierSandbox(root)

        return make

    monkeypatch.setattr(Rollout, "start", start)
    monkeypatch.setattr(_separate_verifier, "_default_create", create)
    return started, contexts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("toml", "config", "record"),
    [
        pytest.param(SEPARATE, {}, False, id="task-separate"),
        pytest.param("", {}, True, id="separate-record"),
        pytest.param("", {"integrity": "strict"}, False, id="strict-run"),
        pytest.param("", {"verifier_sandbox": "separate"}, False, id="run-flag"),
    ],
)
async def test_a_separate_trial_is_regraded_by_the_separate_verifier(
    tmp_path: Path, monkeypatch, toml: str, config: dict[str, Any], record: bool
) -> None:
    """Found in a benchmark's regrade: a separate trial is regraded
    from its saved outputs through the separate verifier, never in the agent's
    environment.

    bench eval regrade once re-ran such a verifier in a sandbox built from the
    task's agent environment (python:3.12-slim here, not ubuntu:24.04), with
    no transfer and no custody record; then it refused them. Now the verifier
    sandbox is built from tests/Dockerfile, gets the transfer the run packed
    (workspace, declared artifact, /logs/artifacts) from the trial folder, and
    its record lands beside the new verifier/. No other sandbox starts, and
    nothing the run wrote changes.
    """
    from benchflow.eval_regrade import REGRADE_FILE, aregrade
    from benchflow.rollout import _setup

    trial, tasks, workspace = await _saved_separate_trial(tmp_path, toml, config)
    if record:
        (trial / RECORD).parent.mkdir()
        (trial / RECORD).write_text("{}")
    box = tmp_path / "verifier-box"
    started, contexts = _verifier_sandbox_only(monkeypatch, box)

    async def changed_verifier(env, task, paths, timing, planes, **kwargs):
        # The agent's work, where it left it; nothing planted elsewhere.
        assert (box / str(workspace).lstrip("/") / "src/answer.txt").is_file()
        assert (box / "logs/artifacts/mode.txt").is_file()
        assert (box / str(tmp_path / "box/tmp/configured.txt").lstrip("/")).is_file()
        assert not (box / str(tmp_path / "box/tmp/agent-only.txt").lstrip("/")).exists()
        assert not (box / str(tmp_path / "box/tests").lstrip("/")).exists()
        paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (paths.verifier_dir / "reward.txt").write_text("0\n")
        return {"reward": 0.0}, None, None

    monkeypatch.setattr(_setup, "_verify_rollout", changed_verifier)
    before = {path: path.read_bytes() for path in trial.rglob("*") if path.is_file()}

    summary = await aregrade(trial.parent, tasks_dir=tasks, reason="verifier fix")

    [row] = summary.trials
    assert (row.status, row.original_reward, row.new_reward, row.change) == (
        "regraded",
        1.0,
        0.0,
        "pass->fail",
    )
    assert not started
    [context] = contexts
    assert (
        (context / "environment/Dockerfile").read_text().startswith("FROM ubuntu:24.04")
    )
    attempt = trial / "regrade" / summary.regrade_id
    sandbox = json.loads((attempt / "verifier-sandbox.json").read_text())
    assert (sandbox["status"], sandbox["image_source"]) == (
        "complete",
        "tests/Dockerfile",
    )
    assert {group["root"] for group in sandbox["custody"]["members"]} == {
        str(workspace),
        str(tmp_path / "box/tmp"),
        "/logs/artifacts",
    }
    assert sandbox["agent_sandbox_alive_during_verify"] is False
    assert (attempt / "verifier/reward.txt").read_text() == "0\n"
    [block] = json.loads((trial / REGRADE_FILE).read_text())["regrades"]
    assert block["verifier_sandbox"] == {
        "image_source": "tests/Dockerfile",
        "status": "complete",
    }
    assert before == {
        path: path.read_bytes()
        for path in trial.rglob("*")
        if path.is_file()
        and "regrade" not in path.relative_to(trial).parts
        and path.name != REGRADE_FILE
    }


@pytest.mark.asyncio
async def test_a_separate_regrade_without_the_runs_collection_is_unscored(
    tmp_path: Path, monkeypatch
) -> None:
    """A separate trial whose folder lacks what the run packs
    (here the /logs/artifacts collection record) is unscored by the separate
    verifier's own transfer check, called through the runner directly. No
    verifier sandbox starts, and none from the agent's image."""
    from benchflow.eval_regrade import run_verifier_on_frozen_trial

    trial, tasks = await _regrade_job(tmp_path, SEPARATE, config={})
    started, contexts = _verifier_sandbox_only(monkeypatch, tmp_path / "box")
    attempt = trial / "regrade" / "direct"
    attempt.mkdir(parents=True)

    rewards, error = await run_verifier_on_frozen_trial(trial, tasks / "task", attempt)

    assert rewards is None
    assert error == (
        "separate verifier transfer failed: artifact collection manifest is "
        "missing or incomplete"
    )
    assert not started and not contexts
    sandbox = json.loads((attempt / "verifier-sandbox.json").read_text())
    assert sandbox["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_a_shared_trial_still_reaches_the_regrade_sandbox(
    tmp_path: Path, monkeypatch
) -> None:
    from benchflow.eval_regrade import aregrade

    trial, tasks = await _regrade_job(tmp_path, "", config={})
    started = _no_regrade_sandbox(monkeypatch)

    summary = await aregrade(trial.parent, tasks_dir=tasks)

    [row] = summary.trials
    assert len(started) == 1
    assert row.status == "failed" and "the regrade built a sandbox" in row.reason


# --- 7. compose sidecars and the environment/Dockerfile fallback ------------


def _sidecar_task(tmp_path: Path, *, tests_dockerfile: bool) -> Path:
    task = _task_dir(tmp_path, SEPARATE, tests_dockerfile=tests_dockerfile)
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    depends_on: [api]\n  api:\n    build:\n"
        "      context: ./api\n"
    )
    return task


def test_the_environment_fallback_with_sidecars_is_refused_at_launch(
    tmp_path: Path,
) -> None:
    """The pre-screen saw the engine
    bench's sidecar task, forced separate by --integrity strict, start its
    api service a second time beside the verifier: the environment/Dockerfile
    fallback copies the compose file into the verifier image's context."""
    from benchflow.integrity.trial import strict_launch_issues
    from benchflow.task import Task

    task = _sidecar_task(tmp_path, tests_dockerfile=False)
    config = Task(task).config
    issues = [
        i
        for i in validate_task_runtime_support(config, sandbox="docker", task_dir=task)
        if i.path == "verifier.sandbox.services"
    ]
    assert len(issues) == 1 and "starts api" in issues[0].reason
    shared = TaskConfig.model_validate({})
    lines = strict_launch_issues(shared, sandbox="docker", task_dir=task)
    assert any(line.startswith("verifier.sandbox.services:") for line in lines), lines
    # A verifier image from tests/Dockerfile has no compose file.
    other = _sidecar_task(tmp_path / "b", tests_dockerfile=True)
    assert not [
        i
        for i in validate_task_runtime_support(
            Task(other).config, sandbox="docker", task_dir=other
        )
        if i.path == "verifier.sandbox.services"
    ]


@pytest.mark.asyncio
async def test_the_environment_fallback_with_sidecars_never_starts(
    tmp_path: Path,
) -> None:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _sidecar_task(tmp_path, tests_dockerfile=False)
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    created: list[Any] = []

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: created.append(a), verify=None
    )

    assert rewards is None and not created
    assert error is not None and "would start compose services (api)" in error
    assert _record(trial)["status"] == "sandbox_failed"


# --- 6. what the verifier sandbox is given: the TaskConfig allowlist ---------

# Every TaskConfig field, as the verifier sandbox's task sees it after
# _verifier_task and _verifier_sandbox_config. "carried": the task's value
# reaches the verifier sandbox; "cleared": reset to its default there;
# "replaced": a new value (the planned verifier sandbox). The sandbox.*
# fields are carried from [verifier.sandbox] when the task declares one,
# else from [sandbox]. A new field fails this test until it is reviewed
# and added here.
VERIFIER_TASK_FIELDS: dict[str, str] = {
    "schema_version": "carried",
    "task": "carried",
    "task.name": "carried",
    "task.version": "carried",
    "task.description": "carried",
    "task.authors": "carried",
    "task.authors.name": "carried",
    "task.authors.email": "carried",
    "task.keywords": "carried",
    "metadata": "carried",
    "verifier": "carried",
    "verifier.type": "carried",
    "verifier.submission_files": "carried",
    "verifier.workspace_recovery": "carried",
    "verifier.timeout_sec": "carried",
    "verifier.env": "carried",
    "verifier.reward_range": "carried",
    "verifier.user": "carried",
    "verifier.network_mode": "carried",
    "verifier.allowed_hosts": "carried",
    "verifier.sandbox_mode": "cleared",
    "verifier.sandbox": "cleared",
    "verifier.service": "carried",
    "verifier.judge": "carried",
    "verifier.judge.model": "carried",
    "verifier.judge.rubric_path": "carried",
    "verifier.judge.input_dir": "carried",
    "verifier.judge.input_type": "carried",
    "verifier.judge.context": "carried",
    "verifier.memory": "carried",
    "verifier.memory.expected_skills": "carried",
    "verifier.hardening": "carried",
    "verifier.hardening.cleanup_conftests": "carried",
    "verifier.hardening.cleanup_git": "carried",
    "verifier.pytest_plugins": "carried",
    "verifier.pytest_plugin_releases": "carried",
    # The trusted task's tool contract must be checked in the verifier's
    # own image too; it is not an agent environment value.
    "verifier.required_tools": "carried",
    "verifier.optional_tools": "carried",
    "verifier.collect": "cleared",
    "verifier.collect.command": "cleared",
    "verifier.collect.service": "cleared",
    "verifier.collect.timeout_sec": "cleared",
    "verifier.collect.user": "cleared",
    "agent": "carried",
    "agent.prompt_prefix": "carried",
    "agent.timeout_sec": "carried",
    # A trial's budget is read where its agent's model proxy starts; the
    # verifier's sandbox runs neither.
    "agent.token_budget": "carried",
    "agent.turn_budget": "carried",
    "agent.user": "carried",
    "agent.network_mode": "cleared",
    "agent.allowed_hosts": "cleared",
    "sandbox": "replaced",
    "sandbox.network_mode": "carried",
    "sandbox.allowed_hosts": "carried",
    "sandbox.blocked_urls": "carried",
    "sandbox.blocked_hosts": "carried",
    "sandbox.build_timeout_sec": "carried",
    "sandbox.docker_image": "carried",
    "sandbox.os": "carried",
    "sandbox.cpus": "carried",
    "sandbox.memory_mb": "carried",
    "sandbox.storage_mb": "carried",
    "sandbox.gpus": "carried",
    "sandbox.gpu_types": "carried",
    "sandbox.mcp_servers": "cleared",
    "sandbox.mcp_servers.name": "cleared",
    "sandbox.mcp_servers.transport": "cleared",
    "sandbox.mcp_servers.url": "cleared",
    "sandbox.mcp_servers.command": "cleared",
    "sandbox.mcp_servers.args": "cleared",
    "sandbox.mcp_servers.cwd": "cleared",
    "sandbox.mcp_servers.env": "cleared",
    "sandbox.mcp_servers.headers": "cleared",
    "sandbox.mcp_servers.tools": "cleared",
    "sandbox.mcp_servers.include_tags": "cleared",
    "sandbox.mcp_servers.exclude_tags": "cleared",
    "sandbox.env": "carried",
    "sandbox.tpu": "carried",
    "sandbox.tpu.type": "carried",
    "sandbox.tpu.topology": "carried",
    "sandbox.skills_dir": "cleared",
    "sandbox.healthcheck": "cleared",
    "sandbox.healthcheck.command": "cleared",
    "sandbox.healthcheck.interval_sec": "cleared",
    "sandbox.healthcheck.timeout_sec": "cleared",
    "sandbox.healthcheck.start_period_sec": "cleared",
    "sandbox.healthcheck.start_interval_sec": "cleared",
    "sandbox.healthcheck.retries": "cleared",
    "sandbox.workdir": "carried",
    "sandbox.setup_commands": "cleared",
    "sandbox.setup_commands.command": "cleared",
    "sandbox.setup_commands.timeout_sec": "cleared",
    "sandbox.setup_commands.cwd": "cleared",
    "sandbox.setup_commands.env": "cleared",
    "sandbox.setup_commands.user": "cleared",
    "sandbox.setup_commands.service": "cleared",
    "sandbox.setup_commands.host_lock": "cleared",
    "sandbox.setup_commands.capture_dir": "cleared",
    "sandbox.setup_commands.capture_dir_b64_env": "cleared",
    "sandbox.setup_commands.capture_dir_b64_env_file_var": "cleared",
    "sandbox.bugswarm_image_tag": "carried",
    "sandbox.allow_internet": "carried",
    # Read only where an agent starts; the verifier's sandbox runs none.
    "sandbox.allow_cloud_metadata": "carried",
    "sandbox.require_cloud_metadata_block": "carried",
    "sandbox.memory": "carried",
    "sandbox.storage": "carried",
    "solution": "carried",
    "solution.env": "carried",
    "solution.timeout_sec": "carried",
    "source": "carried",
    "reward": "carried",
    "multi_step_reward_strategy": "carried",
    # Multi-step tasks are refused at launch; the list is carried untouched.
    "steps": "carried",
    "artifacts": "cleared",
    # The task's own preference table is for bench preferences; grading never
    # reads it.
    "preference": "carried",
    "artifacts.source": "cleared",
    "artifacts.destination": "cleared",
    "artifacts.exclude": "cleared",
    "artifacts.service": "cleared",
}


def _model_types(annotation: Any) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type[BaseModel]] = []
    for arg in typing.get_args(annotation):
        found.extend(_model_types(arg))
    return found


def _field_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for name, field in model.model_fields.items():
        path = prefix + name
        paths.append(path)
        # Not descended: steps are refused at launch and carried whole, and
        # [verifier.sandbox]'s fields are reviewed as sandbox.* (the view's
        # sandbox comes from it when the task declares one).
        if path in ("steps", "verifier.sandbox"):
            continue
        for sub in _model_types(field.annotation):
            paths.extend(_field_paths(sub, path + "."))
    return paths


def test_every_task_config_field_the_verifier_sandbox_gets_is_reviewed() -> None:
    found = _field_paths(TaskConfig)
    unreviewed = sorted(set(found) - set(VERIFIER_TASK_FIELDS))
    stale = sorted(set(VERIFIER_TASK_FIELDS) - set(found))
    assert not unreviewed, (
        "TaskConfig fields the separate verifier's task view has not been "
        f"reviewed for: {unreviewed}; mark each carried, cleared or replaced"
    )
    assert not stale, f"fields no longer in TaskConfig: {stale}"


def _dig(data: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(data, dict):
            return None
        data = data.get(part)
    return data


def test_the_verifier_task_view_carries_and_clears_as_reviewed(tmp_path: Path) -> None:
    """Each field set to a non-default value is carried, cleared to its
    default, or (sandbox) replaced by the planned verifier sandbox."""
    from benchflow.rollout._separate_verifier import _verifier_task
    from benchflow.task import Task

    task_dir = _task_dir(
        tmp_path,
        SEPARATE
        + '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.com"]\n'
        + "timeout_sec = 77.0\n"
        + '[environment]\nskills_dir = "/skills"\nworkdir = "/srv/app"\ncpus = 3\n'
        + 'env = { A = "1" }\n'
        + '[environment.healthcheck]\ncommand = "true"\n'
        + '[[environment.setup_commands]]\ncommand = "echo agent-only"\n'
        + '[[environment.mcp_servers]]\nname = "m"\ntransport = "stdio"\ncommand = "x"\n',
    )
    task = Task(task_dir)
    task.config.artifacts = ["/tmp/out.txt"]
    task.config.verifier.required_tools = ["python3"]
    task.config.verifier.optional_tools = ["dot"]
    from benchflow.task.verifier_sandbox import plan_verifier_image

    plan = plan_verifier_image(task.config, task_dir)
    view = _verifier_task(task, plan)
    before = task.config.model_dump()
    after = view.config.model_dump()
    default = TaskConfig().model_dump()
    for path, how in VERIFIER_TASK_FIELDS.items():
        # A table whose fields are marked differently is checked field by
        # field; the planned sandbox is checked whole below.
        mixed = any(
            other.startswith(path + ".") and other_how != how
            for other, other_how in VERIFIER_TASK_FIELDS.items()
        )
        if mixed or how == "replaced":
            continue
        got = _dig(after, path)
        if how == "carried":
            assert got == _dig(before, path), path
        elif how == "cleared":
            assert got == _dig(default, path), path
    assert after["sandbox"] == plan.sandbox.model_dump()
    # The non-defaults this test set really were set.
    assert before["agent"]["allowed_hosts"] == ["example.com"]
    assert before["sandbox"]["setup_commands"] and before["sandbox"]["mcp_servers"]
    assert before["sandbox"]["healthcheck"] and before["sandbox"]["skills_dir"]
    assert after["sandbox"]["workdir"] == "/srv/app" and after["sandbox"]["cpus"] == 3
    assert after["agent"]["timeout_sec"] == 77.0


# --- the layer hooks never reach the verifier sandbox ------------------------


class _LayeredVerifierSandbox(FakeVerifierSandbox):
    """A verifier env that offers the agent-layer hooks (U3a's
    use_agent_layer on the Docker env, U3b's configure_install_layers called
    on the env's type) and records any call to them."""

    calls: typing.ClassVar[list[str]] = []

    def use_agent_layer(self, *args, **kwargs):
        type(self).calls.append("use_agent_layer")

    def configure_install_layers(self, *args, **kwargs):
        type(self).calls.append("configure_install_layers")


@pytest.mark.asyncio
async def test_the_verifier_env_never_gets_a_layer_hook_call(
    tmp_path: Path, monkeypatch
) -> None:
    """Neither hook adds a SandboxConfig field, so the allowlist above cannot
    see them: the verifier sandbox must never be given an agent or proxy
    layer, which would put the agent's install into the verifier's image."""
    from unittest.mock import AsyncMock

    from benchflow.rollout import _separate_verifier, _setup

    _, rollout, _ = await _separate_trial(tmp_path)
    env = _LayeredVerifierSandbox(tmp_path / "verifier-box")
    _LayeredVerifierSandbox.calls = []

    def create_environment(*args, **kwargs):
        return env

    rollout._planes = SimpleNamespace(
        create_environment=create_environment,
        prepare_separate_verifier_runtime=rollout._planes.prepare_separate_verifier_runtime,
        snapshot_build_config=AsyncMock(),
    )
    rollout._config.environment = "docker"
    rollout._trajectory = [{"type": "agent_message", "content": "done"}]

    async def fake_verify_rollout(env, task, paths, timing, planes, **kwargs):
        paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (paths.verifier_dir / "reward.txt").write_text("1\n")
        return {"reward": 1.0}, None, None

    monkeypatch.setattr(_setup, "_verify_rollout", fake_verify_rollout)
    create = _separate_verifier._default_create(rollout)
    rewards, error = await run_separate_verifier(
        rollout, create_environment=create, verify=None
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert env.started
    assert _LayeredVerifierSandbox.calls == []


@pytest.mark.asyncio
async def test_explicit_package_subtree_crosses_but_is_still_agent_input(tmp_path):
    """A declared installed package can be edited."""
    package = "/usr/local/lib/python3.11/site-packages/nemo/lightning"
    trial, rollout, box = await _linked_trial(
        tmp_path,
        package,
        files={f"{package}/parallel.py": "answer = 42\n"},
        links={},
        resolved={package: package},
    )
    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )
    assert (rewards, error) == ({"reward": 1.0}, None)
    assert (
        box.root / package.lstrip("/") / "parallel.py"
    ).read_text() == "answer = 42\n"
    assert package in _record(trial)["transfer"]["agent_paths"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "/usr/local/lib/python3.11/site-packages",
        "/usr/local/lib/python3.11",
        "/usr/local/lib",
    ],
)
async def test_package_transfer_does_not_allow_library_roots(tmp_path, source):
    """Declaring a library root remains invalid."""
    _, rollout, box = await _linked_trial(
        tmp_path,
        source,
        files={f"{source}/planted.py": "answer = 42\n"},
        links={},
        resolved={source: source},
    )
    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )
    assert rewards is None
    assert "system folder /usr/local/lib" in error


@pytest.mark.asyncio
async def test_sidecar_files_are_marked_as_transferred_input(tmp_path):
    """Narrowing main artifacts must still distrust sidecar files."""
    trial, rollout, box = await _separate_trial(tmp_path)
    services = trial / "artifacts/services"
    services.mkdir(parents=True)
    body = b"print('agent code')\n"
    (services / "plugin.py").write_bytes(body)
    source = "/shared/plugin.py"
    (services / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "artifacts": [
                    {
                        "files": [
                            {
                                "path": source,
                                "local": "plugin.py",
                                "mode": 0o755,
                                "sha256": hashlib.sha256(body).hexdigest(),
                            }
                        ]
                    }
                ],
            }
        )
    )

    async def verify(env, task, paths, timing, *, workspace, agent_paths):
        assert source in agent_paths
        assert "/shared" not in agent_paths
        assert (box.root / source.lstrip("/")).read_bytes() == body
        return await _writes_reward("0")(
            env, task, paths, timing, workspace=workspace, agent_paths=agent_paths
        )

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=verify
    )
    assert (rewards, error) == ({"reward": 0.0}, None)
    assert source in _record(trial)["transfer"]["agent_paths"]
