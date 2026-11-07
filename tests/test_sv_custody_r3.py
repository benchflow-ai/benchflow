"""Separate verifier custody, Codex round 3 (PY8 SV1).

Where the transfer lands in the verifier sandbox (that image's own links),
a single file captured at / with Python present, a folder the agent made at a
declared system path, and host-side reads that must never open a device.
"""

from __future__ import annotations

import os
import shlex
import stat
import tempfile
import threading
import uuid
from pathlib import Path, PurePosixPath

import pytest

from benchflow.rollout import _separate_verifier as sv
from benchflow.rollout._separate_verifier import run_separate_verifier
from tests.test_separate_verifier import LocalTransport
from tests.test_sv_custody import (
    _PLANTED_GRADER,
    _linked_trial,
    _record,
    _writes_reward,
)

_TRUSTED = "#!/bin/sh\necho trusted\n"

# --- R3-1: the verifier image's own links on the way ----------------------------


def _verifier_link(root: Path, link: str, to: str) -> None:
    """In the verifier image at ``root``: ``link`` a relative link to ``to``."""
    path = root / link.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    target = root / to.lstrip("/")
    target.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)


@pytest.mark.asyncio
async def test_a_verifier_image_ancestor_link_into_usr_is_unscored(
    tmp_path: Path,
) -> None:
    """The task declares the folder /out/bin, an ordinary folder in the agent
    image; in the verifier image /out is a link to /usr, where test.sh's
    trusted /usr/bin/sv-grade lives. Discovery, capture and the clean control
    all said /out/bin, and the packed out/bin/sv-grade had no member for
    /out, so tar followed the verifier's link and replaced the trusted grader
    (a wrong answer could score 1). Where members land in the verifier
    sandbox is now checked before anything is emptied or unpacked."""
    trial, rollout, box = await _linked_trial(
        tmp_path,
        "/out/bin",
        files={"/out/bin/sv-grade": _PLANTED_GRADER},
        links={},
        resolved={"/out/bin": "/out/bin"},
    )
    grader = box.root / "usr" / "bin" / "sv-grade"
    grader.parent.mkdir(parents=True)
    grader.write_text(_TRUSTED)
    (box.root / "out").symlink_to("usr", target_is_directory=True)

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert grader.read_text() == _TRUSTED
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: /out in the verifier image is a "
        "link to /usr, so /out/bin/sv-grade would land at /usr/bin/sv-grade, "
        "in /usr/bin"
    )
    assert _record(trial)["status"] == "transfer_failed"


@pytest.mark.asyncio
async def test_a_verifier_image_link_moving_a_declared_file_into_a_system_folder_is_unscored(
    tmp_path: Path,
) -> None:
    """The same for a declared file: /opt/kit/tool, with /opt/kit a link to
    /usr/local/bin in the verifier image only."""
    _, rollout, box = await _linked_trial(
        tmp_path,
        "/opt/kit/tool",
        files={"/opt/kit/tool": _PLANTED_GRADER},
        links={},
        resolved={"/opt/kit/tool": "/opt/kit/tool"},
    )
    trusted = box.root / "usr" / "local" / "bin" / "tool"
    trusted.parent.mkdir(parents=True)
    trusted.write_text(_TRUSTED)
    _verifier_link(box.root, "/opt/kit", "/usr/local/bin")

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert trusted.read_text() == _TRUSTED
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: /opt/kit in the verifier image is "
        "a link to /usr/local/bin, so /opt/kit/tool would land at "
        "/usr/local/bin/tool, in /usr/local/bin"
    )


@pytest.mark.asyncio
async def test_a_verifier_image_link_inside_a_folder_artifact_is_judged_too(
    tmp_path: Path,
) -> None:
    """busybox tar follows a link at a folder that has a member of its own:
    /out/bin/sub in the verifier image is a link to /etc."""
    _, rollout, box = await _linked_trial(
        tmp_path,
        "/out/bin",
        files={"/out/bin/sub/passwd": "planted\n"},
        links={},
        resolved={"/out/bin": "/out/bin"},
    )
    (box.root / "out" / "bin").mkdir(parents=True)
    _verifier_link(box.root, "/out/bin/sub", "/etc")

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "etc" / "passwd").exists()
    assert rewards is None
    assert error == (
        "separate verifier transfer failed: /out/bin/sub in the verifier image "
        "is a link to /etc, so /out/bin/sub would land at /etc, in /etc"
    )


@pytest.mark.asyncio
async def test_a_verifier_image_link_elsewhere_crosses_and_is_recorded(
    tmp_path: Path,
) -> None:
    """A link to an ordinary folder is the image's layout: the transfer
    crosses, and custody says where it physically landed."""
    trial, rollout, box = await _linked_trial(
        tmp_path,
        "/out/bin",
        files={"/out/bin/tool": "built\n"},
        links={},
        resolved={"/out/bin": "/out/bin"},
    )
    _verifier_link(box.root, "/out", "/srv/out")

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert (box.root / "srv" / "out" / "bin" / "tool").read_text() == "built\n"
    record = _record(trial)
    assert record["custody"]["destinations"] == [
        {"path": "/out", "resolves_to": "/srv/out"}
    ]


@pytest.mark.asyncio
async def test_an_empty_artifact_root_under_a_verifier_link_is_not_judged(
    tmp_path: Path,
) -> None:
    """Tar writes nothing for an artifact root with no members (a nop run
    leaves /out/bin empty), so the verifier image's /out link moves nothing:
    no refusal. The workspace and /logs/artifacts are created (and the
    workspace emptied) even when empty, so those are still judged."""
    import io
    import tarfile

    from tests.test_separate_verifier import FakeVerifierSandbox

    archive = tmp_path / "t.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("app/answer.txt")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"41\n"))
    box = FakeVerifierSandbox(tmp_path / "v")
    _verifier_link(box.root, "/out", "/usr")
    summary = {
        "workspace": "/app",
        "agent_paths": ["/app", "/out/bin", "/logs/artifacts"],
    }
    assert await sv._check_destinations(box, archive, summary, 30) == []
    _verifier_link(box.root, "/logs", "/etc")
    with pytest.raises(sv.SeparateVerifierError, match="/logs in the verifier image"):
        await sv._check_destinations(box, archive, summary, 30)


def test_the_destination_probe_walks_ancestors_and_skips_a_dedicated_workspace() -> (
    None
):
    paths = sv._destination_paths(
        ["/app/src", "/out/bin", "/out/bin/sub", "/logs/artifacts"],
        ["/app", "/out/bin", "/logs/artifacts"],
        "/app",
    )
    # The workspace /app is emptied (a link there removed) before unpacking.
    assert paths == ["/out", "/out/bin", "/logs", "/logs/artifacts", "/out/bin/sub"]
    shared = sv._destination_paths(["/root/x"], ["/root"], "/root")
    assert shared == ["/root", "/root/x"]


def test_the_destination_probe_runs_in_plain_sh(tmp_path: Path) -> None:
    """Shell only, and odd names survive quoting."""
    import subprocess

    root = tmp_path / "v"
    _verifier_link(root, "/a b", "/usr/lib")
    (root / "dangling").symlink_to("nowhere")
    (root / "plain").mkdir()
    paths = ["/a b", "/dangling", "/plain", "/missing/x"]
    script = tmp_path / "probe.sh"
    script.write_text(sv._destination_script(paths))
    done = subprocess.run(
        ["sh", str(script), str(root.resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.splitlines() == ["0 /usr/lib", "1 ?", sv._DESTINATIONS_DONE]


# --- R3-2: a file at / with Python present ------------------------------------


class _OnlyTmp(LocalTransport):
    """Capture with /tmp as both temporary folders the script tries, and the
    captured file directly in /tmp: its folder (the capture root) contains
    every temporary folder, as / does for /result.txt. (A missing folder
    instead would fail with "No such file or directory", which reads as a
    missing python3 and takes the tar fallback.)"""

    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30):
        argv = shlex.split(cmd)
        argv = [a.replace('"/var/tmp"', '"/tmp"') for a in argv]
        return await super().exec(shlex.join(argv), user=user, timeout_sec=timeout_sec)


@pytest.mark.asyncio
async def test_python_capture_of_a_file_whose_folder_holds_every_temp_folder(
    tmp_path: Path,
) -> None:
    """With Python in the image, a declared /result.txt was captured with root
    /, and the script then required a temporary folder outside /: none
    exists, so capture failed (not as a missing Python, so the tar fallback
    never ran) and a right answer was unscored. A single file's listing is
    that file alone, so its archive may sit beside it."""
    from benchflow.review.evidence import capture_task_evidence

    name = f"benchflow-sv1-r3-{uuid.uuid4().hex}.txt"
    result = Path(tempfile.gettempdir()).resolve() / name
    assert result.parent == Path("/tmp")
    result.write_text("42\n")
    try:
        workspace = tmp_path / "app"
        workspace.mkdir()
        manifest = await capture_task_evidence(
            _OnlyTmp(), str(workspace), tmp_path / "evidence", artifacts=[str(result)]
        )
    finally:
        result.unlink()
    [artifact] = manifest.artifacts
    assert artifact.status == "captured"
    sub = tmp_path / "evidence" / artifact.bundle_path / "workspace" / name
    assert sub.read_text() == "42\n"


# --- R3-3: a folder the agent made at a declared system path -------------------


def _absent(dir_before: bool | None) -> dict:
    return {
        "exists": bool(dir_before),
        "link": False,
        "dir": dir_before,
        "entries": 0,
        "bytes": 0,
    }


@pytest.mark.asyncio
async def test_a_folder_the_agent_made_at_a_declared_system_file_scores_zero(
    tmp_path: Path,
) -> None:
    """The task declares the executable /usr/local/bin/tool, absent in the
    clean image. The agent made a folder there instead: unscored (and
    retried) as if the task were misconfigured, where the missing tool should
    score 0. The clean control says no folder was there."""
    trial, rollout, box = await _linked_trial(
        tmp_path,
        "/usr/local/bin/tool",
        files={"/usr/local/bin/tool/readme": "not a tool\n"},
        links={},
        resolved={"/usr/local/bin/tool": "/usr/local/bin/tool"},
    )
    rollout._pristine_outputs["paths"]["/usr/local/bin/tool"] = _absent(False)

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert not (box.root / "usr" / "local" / "bin" / "tool").exists()
    assert (rewards, error) == ({"reward": 0.0}, None)
    record = _record(trial)
    assert record["status"] == "refused"
    assert record["refusal"] == (
        "the solution's outputs cannot reach the verifier: declared artifact "
        "/usr/local/bin/tool is a folder that lands /usr/local/bin/tool in "
        "system folder /usr/local/bin; before the agent ran it was not a folder"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("dir_before", [True, None])
async def test_a_system_folder_already_there_or_unmeasured_stays_unscored(
    tmp_path: Path, dir_before: bool | None
) -> None:
    """A folder the clean image already had there is the task's doing; with
    no measurement (an older record, an image without the tools) nothing
    proves otherwise."""
    _, rollout, box = await _linked_trial(
        tmp_path,
        "/usr/local/bin/tool",
        files={"/usr/local/bin/tool/readme": "x\n"},
        links={},
        resolved={"/usr/local/bin/tool": "/usr/local/bin/tool"},
    )
    rollout._pristine_outputs["paths"]["/usr/local/bin/tool"] = _absent(dir_before)

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=_writes_reward("1")
    )

    assert rewards is None
    assert error == (
        "separate verifier transfer failed: declared artifact "
        "/usr/local/bin/tool lands /usr/local/bin/tool in system folder "
        "/usr/local/bin"
    )


def test_the_clean_control_records_whether_each_path_was_a_folder(
    tmp_path: Path,
) -> None:
    import asyncio

    (tmp_path / "d").mkdir()
    (tmp_path / "f").write_text("x")
    paths = [str(tmp_path / n) for n in ("d", "f", "absent")]

    class NoPython(LocalTransport):
        async def exec(self, cmd, *, user="root", timeout_sec=30):
            if shlex.split(cmd)[0] == "python3":
                from benchflow.sandbox.protocol import ExecResult

                return ExecResult(127, "", "sh: 1: python3: not found")
            return await super().exec(cmd, user=user, timeout_sec=timeout_sec)

    for transport in (LocalTransport(), NoPython()):
        measured = asyncio.run(
            sv.measure_outputs(
                transport, workspace=paths[0], artifacts=[paths[1], paths[2]]
            )
        )
        assert [measured["paths"][p]["dir"] for p in paths] == [True, False, False]


# --- R3-4: never open a device -------------------------------------------------


def _fifo_writer(path: Path) -> tuple[threading.Thread, threading.Event]:
    """A writer that blocks until something opens ``path`` for reading."""
    opened = threading.Event()

    def write() -> None:
        fd = os.open(path, os.O_WRONLY)
        opened.set()
        os.close(fd)

    thread = threading.Thread(target=write, daemon=True)
    thread.start()
    return thread, opened


def _release(path: Path, thread: threading.Thread) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    thread.join(5)
    os.close(fd)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_hashing_a_swapped_in_special_file_never_opens_it(tmp_path: Path) -> None:
    """A leftover agent process swaps a device node over a listed file in the
    live /logs/verifier mount: opening it runs its driver (a watchdog starts
    on open) before fstat could refuse it. A FIFO stands in for the device:
    its writer unblocks only if the host opened it."""
    fifo = tmp_path / "reward.txt"
    os.mkfifo(fifo)
    thread, opened = _fifo_writer(fifo)
    fd = sv.ReachablePath.directory(tmp_path)
    try:
        digest, read = sv._file_digest(fd, "reward.txt", 1 << 20)
        assert (digest, read) == ({"unreadable": True}, 0)
        assert not opened.wait(0.5)
    finally:
        _release(fifo, thread)
    (tmp_path / "ok.txt").write_text("1\n")
    fd = sv.ReachablePath.directory(tmp_path)
    digest, _ = sv._file_digest(fd, "ok.txt", 1 << 20)
    assert digest["bytes"] == 2 and "sha256" in digest


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_staging_a_swapped_in_special_file_never_opens_it(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "sub").mkdir(parents=True)
    fifo = artifacts / "sub" / "out.txt"
    os.mkfifo(fifo)
    thread, opened = _fifo_writer(fifo)
    try:
        with pytest.raises(sv.SeparateVerifierError, match="changed since collection"):
            sv._stage_logs_file(
                artifacts, PurePosixPath("sub/out.txt"), tmp_path / "staged", "0" * 64
            )
        assert not opened.wait(0.5)
    finally:
        _release(fifo, thread)
    assert not (tmp_path / "staged").exists()
    regular = artifacts / "sub" / "ok.txt"
    regular.write_text("ok\n")
    regular.chmod(0o755)
    import hashlib

    mode = sv._stage_logs_file(
        artifacts,
        PurePosixPath("sub/ok.txt"),
        tmp_path / "staged-ok",
        hashlib.sha256(b"ok\n").hexdigest(),
    )
    assert mode == 0o755
    assert stat.S_ISREG((tmp_path / "staged-ok").stat().st_mode)
