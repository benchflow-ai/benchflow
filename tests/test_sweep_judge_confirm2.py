"""Guard CONFIRM-taskmd-rewards-2 against reviewed commit 11fc9381."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.rewards.file_readers import find_deliverables
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.verifier_errors import (
    DownloadVerifierDirError,
    VerifierOutputParseError,
)
from benchflow.taskmd import verify


@pytest.fixture
def native_capture(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    monkeypatch.setattr(verify, "COPY_DIR", str(stage))
    monkeypatch.setattr(verify, "COPY_TAR", str(stage / "copy.tar"))
    monkeypatch.setitem(verify.KEPT_CAPS, "bytes", 2048)
    monkeypatch.setitem(verify.KEPT_CAPS, "files", 3)
    monkeypatch.setattr(verify, "COPY_TAR_LIMIT", 32768)
    source = tmp_path / "submission"
    source.mkdir()
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()

    class Box:
        supports_system_shell = True
        unavailable_tar = False
        unavailable_download = False

        def __init__(self):
            self.downloads = []

        async def exec(self, command, **kwargs):
            if self.unavailable_tar:
                return SimpleNamespace(stdout="", stderr="no tar", return_code=1)
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=10
            )
            return SimpleNamespace(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )

        async def download_file(self, source, target):
            self.downloads.append((source, Path(source).stat().st_size))
            if self.unavailable_download:
                raise OSError("download unavailable")
            shutil.copyfile(source, target)

        async def download_dir(self, source_dir, target_dir):
            self.downloads.append(("UNBOUNDED", 0))
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True, symlinks=True)

    box = Box()
    verifier = Verifier.__new__(Verifier)
    verifier._sandbox = box
    verifier._rollout_paths = paths
    verifier._task = SimpleNamespace(
        config=SimpleNamespace(
            verifier=SimpleNamespace(judge=SimpleNamespace(input_dir=str(source)))
        )
    )
    return verifier, box, source


@pytest.mark.parametrize("strategy", [False, True])
@pytest.mark.parametrize(
    "kind", ["bytes", "entries", "tar-unavailable", "download-unavailable"]
)
async def test_native_capture_refuses_before_unbounded_transfer(
    native_capture, kind, strategy
):
    """CONFIRM-2: every native configuration refuses over-cap or unavailable capture."""
    verifier, box, source = native_capture
    (source / "answer.txt").write_text("correct")
    if kind == "bytes":
        (source / "answer.txt").write_bytes(b"x" * 1000000)
    elif kind == "entries":
        for index in range(4):
            (source / f"{index}.txt").write_text("small")
    elif kind == "tar-unavailable":
        box.unavailable_tar = True
    else:
        box.unavailable_download = True
    selected = SimpleNamespace(input_dir=str(source)) if strategy else None
    own_failure = kind == "download-unavailable"
    expected = DownloadVerifierDirError if own_failure else VerifierOutputParseError
    with pytest.raises(expected) as caught:
        async with verifier._download_deliverables(selected):
            pytest.fail("unsafe capture was accepted")
    assert isinstance(caught.value, DownloadVerifierDirError) is own_failure
    if own_failure:
        cause = caught.value.__cause__
        assert cause is not None
        while cause.__cause__ is not None:
            cause = cause.__cause__
        assert isinstance(cause, OSError)
        assert str(cause) == "download unavailable"
    assert all(
        path != "UNBOUNDED" and size < verify.COPY_TAR_LIMIT
        for path, size in box.downloads
    )


async def test_native_capture_regular_and_link_controls(native_capture, tmp_path):
    """CONFIRM-2: bounded regular output survives; agent links never become host files."""
    verifier, box, source = native_capture
    (source / "answer.txt").write_text("correct")
    controller = tmp_path / "controller.txt"
    controller.write_text("DUMMY_CONTROLLER_MARKER")
    (source / "notes.txt").symlink_to(controller)
    async with verifier._download_deliverables() as dest:
        assert find_deliverables(dest, evidence_root=dest) == {"answer.txt": "correct"}
    assert box.downloads and all(path != "UNBOUNDED" for path, size in box.downloads)


@pytest.mark.parametrize("kind", ["bytes", "tar-unavailable", "download-unavailable"])
async def test_remote_agent_judge_inputs_also_require_bounded_capture(
    native_capture, kind, tmp_path
):
    """CONFIRM-2 sibling: a file declaration cannot bypass native capture caps."""
    from benchflow.task.verifier_errors import AgentJudgeInputError

    verifier, box, source = native_capture
    answer = source / "answer.txt"
    answer.write_text("correct")
    if kind == "bytes":
        answer.write_bytes(b"x" * 1000000)
    elif kind == "tar-unavailable":
        box.unavailable_tar = True
    else:
        box.unavailable_download = True
    dest = tmp_path / "private-inputs"
    dest.mkdir()
    own_failure = kind == "download-unavailable"
    expected = DownloadVerifierDirError if own_failure else AgentJudgeInputError
    with pytest.raises(expected) as caught:
        await verifier._resolve_agent_judge_input(str(answer), inputs_dir=dest, index=0)
    assert isinstance(caught.value, DownloadVerifierDirError) is own_failure
    if own_failure:
        cause = caught.value.__cause__
        assert cause is not None
        while cause.__cause__ is not None:
            cause = cause.__cause__
        assert isinstance(cause, OSError)
        assert str(cause) == "download unavailable"
    assert all(
        path != str(answer) and size < verify.COPY_TAR_LIMIT
        for path, size in box.downloads
    )


async def test_remote_agent_judge_regular_file_control(native_capture, tmp_path):
    verifier, box, source = native_capture
    answer = source / "answer.txt"
    answer.write_text("correct")
    dest = tmp_path / "private-inputs"
    dest.mkdir()
    captured = await verifier._resolve_agent_judge_input(
        str(answer), inputs_dir=dest, index=0
    )
    assert captured.read_text() == "correct"
    assert all(path != str(answer) for path, size in box.downloads)
