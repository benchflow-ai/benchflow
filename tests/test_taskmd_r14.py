"""Publication findings of review B on 71bf6a8c2 (TASKMD-R14).

B-M1 Docker commits before the producer's exit status, B-M2 regrade and
skill exports lack their own root, B-M3 staging above a wholly mounted
rollout on another filesystem, B-L1 cleanup masks the primary error,
P-H1 egress collection opens a root after the agent ran, P-M1 a cancelled
non-streaming download returns while its copier still runs.
"""

import asyncio
import contextlib
import errno
import io
import json
import logging
import os
import tarfile
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.sandbox._download import private_download
from benchflow.sandbox.docker import DockerSandbox

SHM = Path("/dev/shm")


def _other_filesystem() -> Path:
    """A real second filesystem: /dev/shm is its own mount below /."""
    if not SHM.is_dir() or not os.access(SHM, os.W_OK):
        pytest.skip("needs a writable /dev/shm (Linux)")
    return Path(tempfile.mkdtemp(prefix="r14-", dir=SHM))


@pytest.fixture
def shm():
    root = _other_filesystem()
    try:
        yield root
    finally:
        import shutil

        for folder, dirs, _ in os.walk(root):
            for name in dirs:
                with contextlib.suppress(OSError):
                    os.chmod(os.path.join(folder, name), 0o755)
        shutil.rmtree(root, ignore_errors=True)


def _archive(*, directory: bool = True, readonly: bool = False, files=("first",)):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        if directory:
            info = tarfile.TarInfo("output")
            info.type = tarfile.DIRTYPE
            info.mode = 0o555 if readonly else 0o755
            tar.addfile(info)
        for name in files:
            info = tarfile.TarInfo(f"output/{name}" if directory else "output")
            info.size = 3
            tar.addfile(info, io.BytesIO(b"NEW"))
    return buffer.getvalue()


def _sandbox(monkeypatch, script: str, payload: Path):
    """A real subprocess transport standing in for ``docker cp``."""
    from benchflow.sandbox import docker

    real_spawn = asyncio.create_subprocess_exec
    spawned = []

    async def spawn(*args, **kwargs):
        process = await real_spawn(
            "/bin/sh",
            "-c",
            script,
            "producer",
            str(payload),
            stdout=kwargs["stdout"],
            stderr=asyncio.subprocess.PIPE,
        )
        spawned.append(process)
        return process

    monkeypatch.setattr(docker.asyncio, "create_subprocess_exec", spawn)
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._main_container = "c" * 64
    sandbox._docker_compose_global_args = Mock(return_value=[])
    sandbox._docker_compose_env = Mock(return_value={})
    sandbox._compose_transport_env = Mock(return_value={})
    return sandbox, spawned


def _old_output(root: Path) -> Path:
    out = root / "out"
    out.mkdir()
    (out / "old").write_text("OLD")
    return out


def _staging_left(root: Path) -> list[str]:
    return [p.name for p in root.iterdir() if p.name.startswith(".benchflow-")]


# --- B-M1 -------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", [1536, None], ids=["first-member", "whole-archive"])
async def test_failed_producer_keeps_old_destination(tmp_path, monkeypatch, cut):
    """Guards B-M1 introduced by 71bf6a8c2: commit only after exit 0."""
    data = _archive()
    payload = tmp_path / "stream.tar"
    payload.write_bytes(data[:cut] if cut else data)
    sandbox, _ = _sandbox(
        monkeypatch, 'cat "$1"; echo transport-failed >&2; exit 23', payload
    )
    out = _old_output(tmp_path)
    held = rw.ReachablePath.directory(tmp_path)
    with pytest.raises(RuntimeError, match=r"exit 23\): transport-failed"):
        await sandbox._stream_download("/output", held / "out", directory=True)
    assert sorted(p.name for p in out.iterdir()) == ["old"]
    assert (out / "old").read_text() == "OLD"
    assert _staging_left(tmp_path) == []


@pytest.mark.asyncio
async def test_clean_exit_commits(tmp_path, monkeypatch):
    """The same transport exiting 0 publishes the new tree."""
    payload = tmp_path / "stream.tar"
    payload.write_bytes(_archive())
    sandbox, _ = _sandbox(monkeypatch, 'cat "$1"', payload)
    out = _old_output(tmp_path)
    held = rw.ReachablePath.directory(tmp_path)
    await sandbox._stream_download("/output", held / "out", directory=True)
    assert sorted(p.name for p in out.iterdir()) == ["first"]
    assert _staging_left(tmp_path) == []


@pytest.mark.asyncio
async def test_disk_full_is_not_relabelled_as_docker_exit(tmp_path, monkeypatch):
    """Guards B-M1: ENOSPC during extraction stays ENOSPC, not the kill's exit."""
    payload = tmp_path / "stream.tar"
    payload.write_bytes(_archive(files=("first", "second")))
    # The producer keeps its pipe open, so only our kill ends it.
    sandbox, _ = _sandbox(monkeypatch, 'cat "$1"; exec sleep 30', payload)
    out = _old_output(tmp_path)

    def disk_full(source, sink, *args, **kwargs):
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(rw.shutil, "copyfileobj", disk_full)
    held = rw.ReachablePath.directory(tmp_path)
    with pytest.raises(OSError) as raised:
        await asyncio.wait_for(
            sandbox._stream_download("/output", held / "out", directory=True), 20
        )
    assert raised.value.errno == errno.ENOSPC, raised.value
    assert sorted(p.name for p in out.iterdir()) == ["old"]
    assert _staging_left(tmp_path) == []


@pytest.mark.asyncio
async def test_cancel_after_extraction_keeps_old_destination(tmp_path, monkeypatch):
    """Guards B-M1: a whole archive whose producer has not exited is not committed."""
    payload = tmp_path / "stream.tar"
    payload.write_bytes(_archive())
    sandbox, spawned = _sandbox(monkeypatch, 'cat "$1"; exec sleep 30', payload)
    out = _old_output(tmp_path)
    held = rw.ReachablePath.directory(tmp_path)
    transfer = asyncio.create_task(
        sandbox._stream_download("/output", held / "out", directory=True)
    )
    # Let the whole archive be read; the producer is still running.
    for _ in range(100):
        await asyncio.sleep(0.05)
        staged = [
            p
            for p in tmp_path.glob(".benchflow-download-*/payload/first")
            if p.is_file()
        ]
        if staged or sorted(q.name for q in out.iterdir()) != ["old"]:
            break
    await asyncio.sleep(0.2)
    transfer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await transfer
    assert sorted(p.name for p in out.iterdir()) == ["old"]
    assert _staging_left(tmp_path) == []
    assert spawned and spawned[0].returncode is not None


# --- B-L1 -------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode 0555")
def test_final_rename_disk_full_keeps_errno_and_cleans_readonly_tree(
    tmp_path, monkeypatch
):
    """Guards B-L1 introduced by 71bf6a8c2: ENOSPC survives a 0555 staging tree."""
    held = rw.ReachablePath.directory(tmp_path)
    real_rename = rw.os.rename

    def fail_final(source, target, *args, **kwargs):
        if source == "payload":
            raise OSError(errno.ENOSPC, "injected final-rename disk full")
        return real_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(rw.os, "rename", fail_final)
    with pytest.raises(OSError) as raised:
        rw.publish_tar_stream(
            io.BytesIO(_archive(readonly=True)), held / "out", directory=True
        )
    assert raised.value.errno == errno.ENOSPC, raised.value
    assert _staging_left(tmp_path) == []


def test_cleanup_failure_keeps_primary_error_and_reports_path(
    tmp_path, monkeypatch, caplog
):
    """Guards B-L1: a refused cleanup is reported, never replaces the error."""
    real_remove = rw._remove_beneath

    def refuse(parent, name, *, where):
        if name.startswith(".benchflow-download-"):
            raise PermissionError(errno.EACCES, "injected cleanup refusal")
        return real_remove(parent, name, where=where)

    def disk_full(source, sink, *args, **kwargs):
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(rw, "_remove_beneath", refuse)
    monkeypatch.setattr(rw.shutil, "copyfileobj", disk_full)
    held = rw.ReachablePath.directory(tmp_path)
    with caplog.at_level(logging.WARNING), pytest.raises(OSError) as raised:
        rw.publish_tar_stream(io.BytesIO(_archive()), held / "out", directory=True)
    assert raised.value.errno == errno.ENOSPC, raised.value
    kept = _staging_left(tmp_path)
    assert len(kept) == 1
    assert kept[0] in caplog.text


# --- B-M2 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regrade_receipt_lands_on_other_filesystem(tmp_path, shm, monkeypatch):
    """Guards B-M2 introduced by 71bf6a8c2: the attempt is pinned before use."""
    import benchflow.eval_regrade as regrade
    from benchflow.review import evidence_runtime
    from benchflow.review.evidence import EvidenceManifest
    from benchflow.rollout import Rollout
    from benchflow.rollout import _setup as setup_module
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from tests.test_check_run_plan import _task

    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    task = _task(tmp_path / "task")
    trial = shm / "jobs" / "trial"
    (trial / "evidence").mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps({"agent": "nop", "environment": "docker"})
    )
    (trial / "evidence" / "manifest.json").write_text(
        EvidenceManifest(
            workspace="/app", archive_sha256="0" * 64, entries=()
        ).model_dump_json()
    )
    attempt = trial / "regrade" / "attempt"
    attempt.mkdir(parents=True)

    async def start(self):
        self._agent_cwd = "/app"

    async def verify(env, task, paths, timing, planes, **kwargs):
        (paths.verifier_dir / "reward.txt").write_text("1")
        return {"reward": 1.0}, None, None

    monkeypatch.setattr(
        DefaultRolloutPlanes,
        "create_environment",
        lambda *a, **k: SimpleNamespace(exec=AsyncMock()),
    )
    monkeypatch.setattr(DefaultRolloutPlanes, "setup_sandbox_user", AsyncMock())
    monkeypatch.setattr(Rollout, "start", start)
    monkeypatch.setattr(Rollout, "install_agent", AsyncMock())
    monkeypatch.setattr(Rollout, "cleanup", AsyncMock())
    monkeypatch.setattr(evidence_runtime, "ensure_evidence_python", AsyncMock())
    monkeypatch.setattr(regrade, "_install_bundle", AsyncMock(return_value="/up"))
    monkeypatch.setattr(regrade, "root_python_command", AsyncMock(return_value="x"))
    monkeypatch.setattr(
        regrade, "exec_raw", AsyncMock(return_value=SimpleNamespace(return_code=0))
    )
    monkeypatch.setattr(setup_module, "_verify_rollout", verify)
    rewards, error = await regrade.run_verifier_on_frozen_trial(trial, task, attempt)
    assert error is None and rewards == {"reward": 1.0}
    assert (attempt / "verifier" / "reward.txt").read_text() == "1"


class _SkillTransport:
    @private_download(directory=True)
    async def download_dir(self, source_dir, target_dir):
        pack = Path(target_dir) / "example"
        pack.mkdir(parents=True)
        (pack / "SKILL.md").write_text("safe skill")


def _skill_rollout(target: Path, **extra):
    return SimpleNamespace(
        _env=_SkillTransport(),
        _config=SimpleNamespace(
            export_generated_skills_to=target,
            generated_skills_root="/skills",
            recorded_skill_mode=None,
        ),
        **extra,
    )


@pytest.mark.asyncio
async def test_skill_export_on_other_filesystem(shm, monkeypatch):
    """Guards B-M2 introduced by 71bf6a8c2: export and capture use its own root."""
    from benchflow.rollout import _user_loop

    monkeypatch.setattr(_user_loop.asyncio, "sleep", AsyncMock())
    rollout = _skill_rollout(shm / "skills")
    await _user_loop._export_generated_skills(rollout)
    assert rollout._evolved_skills == {"example": "safe skill"}
    assert (shm / "skills" / "example" / "SKILL.md").read_text() == "safe skill"


@pytest.mark.asyncio
async def test_rollout_setup_holds_skill_export_root(tmp_path, shm, monkeypatch):
    """B-M2: Rollout.setup pins the export root before an agent runs, and the
    export publishes and reads back through it on another filesystem."""
    from benchflow.rollout import Rollout, RolloutConfig, _user_loop
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from tests.test_check_run_plan import _task

    class Boundary(BaseException):
        pass

    def boundary(*args, **kwargs):
        raise Boundary

    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    monkeypatch.setattr(DefaultRolloutPlanes, "create_environment", boundary)
    monkeypatch.setattr(_user_loop.asyncio, "sleep", AsyncMock())
    export = shm / "out" / "skills"
    rollout = Rollout(
        RolloutConfig(
            task_path=_task(tmp_path / "task"),
            agent="oracle",
            jobs_dir=tmp_path / "jobs",
            export_generated_skills_to=export,
        )
    )
    with contextlib.suppress(Boundary):
        await rollout.setup()
    rollout._env = _SkillTransport()
    await _user_loop._export_generated_skills(rollout)
    assert rollout._evolved_skills == {"example": "safe skill"}
    assert (export / "example" / "SKILL.md").read_text() == "safe skill"


# --- B-M3 -------------------------------------------------------------------


def test_whole_mounted_rollout_stages_in_parent_on_other_filesystem(shm):
    """Guards B-M3 introduced by 71bf6a8c2 (Apple's /logs on an external volume)."""
    from benchflow.task.paths import RolloutPaths

    trial = shm / "jobs" / "trial"
    paths = RolloutPaths(rollout_dir=trial)
    paths.mkdir()
    rw.declare_harness_mount(trial)
    with rw.private_folder(paths.verifier_dir) as stage:
        assert stage.parent == trial.parent.resolve()
        assert stage.is_dir()
    assert _staging_left(trial.parent) == []


def test_pinned_root_alone_stages_in_parent(shm):
    """B-M3 with the review's exact topology: only the rollout is pinned."""
    trial = shm / "trial"
    (trial / "verifier").mkdir(parents=True)
    held = rw.pin_destination(trial)
    verifier = rw.ReachablePath.rollout(trial, "verifier")
    rw.declare_harness_mount(trial)
    with rw.private_folder(verifier) as stage:
        assert stage.parent == shm.resolve()
    assert held is not None


# --- P-H1 -------------------------------------------------------------------


class _EgressTransport:
    @private_download(directory=False)
    async def download_file(self, source_path, target_path):
        Path(target_path).write_text("PROXY LOG")

    async def exec(self, *args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="PROXY LOG", stderr="")


def _swap_jobs(root: Path, private: Path) -> Path:
    from benchflow.sandbox.egress_denylist import TRAJECTORY_LOG_NAME

    marker = private / "trial" / "trajectory" / TRAJECTORY_LOG_NAME
    marker.parent.mkdir(parents=True)
    marker.write_text("ORIGINAL")
    shared = root / "shared"
    (shared / "job").rename(shared / "old-job")
    (shared / "job").symlink_to(private, target_is_directory=True)
    return marker


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", [True, False], ids=["pinned", "initial-root"])
async def test_egress_log_never_follows_swapped_jobs_folder(tmp_path, pinned):
    """Guards P-H1 (pre-existing on integ): no root is opened after the agent."""
    from benchflow.sandbox.egress_denylist import (
        TRAJECTORY_LOG_NAME,
        stop_egress_denylist,
    )
    from benchflow.task.paths import RolloutPaths

    trial = tmp_path / "shared" / "job" / "trial"
    trial.mkdir(parents=True)
    paths = RolloutPaths(rollout_dir=trial)
    if pinned:
        paths.mkdir()
    marker = _swap_jobs(tmp_path, tmp_path / "private")
    await stop_egress_denylist(_EgressTransport(), trial)
    # Refused (the held root no longer matches its spelling), never followed.
    assert marker.read_text() == "ORIGINAL"
    assert sorted(p.name for p in marker.parent.iterdir()) == [TRAJECTORY_LOG_NAME]


@pytest.mark.asyncio
async def test_egress_log_lands_through_rollout_capability(tmp_path):
    """P-H1: a caller holding the rollout capability publishes through it."""
    from benchflow.sandbox.egress_denylist import (
        TRAJECTORY_LOG_NAME,
        stop_egress_denylist,
    )

    trial = tmp_path / "trial"
    trial.mkdir()
    held = rw.pin_destination(trial)
    await stop_egress_denylist(_EgressTransport(), held)
    log = trial / "trajectory" / TRAJECTORY_LOG_NAME
    assert log.read_text() == "PROXY LOG"


@pytest.mark.asyncio
async def test_init_rollout_pins_the_rollout_root(tmp_path):
    """P-H1: the rollout's own root is held from _init_rollout, before launch."""
    from benchflow.rollout._setup import _init_rollout
    from tests.test_check_run_plan import _task

    _, rollout_dir, _paths, *_ = _init_rollout(
        _task(tmp_path / "task"), "job", "trial", tmp_path / "jobs"
    )
    capability = rw.destination(rollout_dir / "trajectory" / "x.jsonl")
    assert capability._anchor.path == rollout_dir.resolve()


# --- P-M1 -------------------------------------------------------------------


class _CopyTransport:
    @private_download(directory=False)
    async def download_file(self, source_path, target_path):
        Path(target_path).write_text("NEW")


@pytest.mark.asyncio
async def test_cancelled_download_returns_after_its_copier(tmp_path, monkeypatch):
    """Guards P-M1 (pre-existing on integ): no write after cancellation returns."""
    out = tmp_path / "out"
    out.write_text("OLD")
    rw.pin_destination(tmp_path)
    entered, released, done = threading.Event(), threading.Event(), threading.Event()
    original = rw._copy_opened

    def hold(*args, **kwargs):
        entered.set()
        released.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            done.set()

    monkeypatch.setattr(rw, "_copy_opened", hold)
    task = asyncio.create_task(_CopyTransport().download_file("/answer", out))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    threading.Timer(0.3, released.set).start()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    returned = out.read_text()
    worker_done = done.is_set()
    released.set()
    await asyncio.to_thread(done.wait, 5)
    assert worker_done, "cancellation returned while the copier still ran"
    assert out.read_text() == returned
