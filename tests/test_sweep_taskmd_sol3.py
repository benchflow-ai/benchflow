"""Guards the PY8 evidence fixes against commit 363d815b."""

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.taskmd import verify


class LocalBox:
    async def download_file(self, source, target):
        shutil.copyfile(source, target)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


@pytest.fixture
def capture_box(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    monkeypatch.setattr(verify, "COPY_DIR", str(stage))
    monkeypatch.setattr(verify, "COPY_TAR", str(stage / "copy.tar"))
    monkeypatch.setitem(verify.KEPT_CAPS, "bytes", 1024)
    monkeypatch.setattr(verify, "COPY_TAR_LIMIT", 32_768)

    async def execute(box, command, **kwargs):
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        return SimpleNamespace(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    monkeypatch.setattr(verify, "root_exec", execute)
    return LocalBox()


@pytest.mark.parametrize(
    "pattern,relative",
    [
        ("cache.bin", "cache.bin"),
        ("*.bin", "nested/cache.bin"),
        ("nested/*", "nested/cache.bin"),
        ("nested", "nested/cache.bin"),
        ("**/*.bin", "nested/cache.bin"),
    ],
)
async def test_output_exclusions_apply_before_capture_caps(
    tmp_path, capture_box, monkeypatch, pattern, relative
):
    """Excluded trees never consume transfer bytes or file budget."""
    source = tmp_path / "out"
    source.mkdir()
    (source / "report.txt").write_text("correct report")
    cache = source / relative
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"x" * 1_000_000)
    # Exercise the production verifier call, not an optional helper parameter.
    import json

    from benchflow.task.formats import materialize_task_dir
    from tests.test_taskmd_agent_judge import FIXTURE
    from tests.test_taskmd_verify import _strategy, _Verifier

    package = tmp_path / "task"
    shutil.copytree(FIXTURE, package)
    task = package / "task.md"
    text = task.read_text().replace(
        'workdir = "/app"',
        f"workdir = {json.dumps(str(source))}\noutputs = [{{path={json.dumps(str(source))}, exclude=[{json.dumps(pattern)}]}}]",
    )
    task.write_text(text)
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    native = materialize_task_dir(package)
    verifier = _Verifier(native, tmp_path / "trial", capture_box, report=None)
    captured = {}

    async def judged(**kwargs):
        root = kwargs["fs_root"] / str(source).lstrip("/")
        captured["refused"] = kwargs["refused"]
        captured["report"] = (
            (root / "report.txt").read_text() if root.is_dir() else None
        )
        captured["excluded"] = (root / relative).exists()
        return {"prints-42": {"_value": True, "verdict": "pass"}}, None, {}

    monkeypatch.setattr(verify, "_run_judges", judged)
    monkeypatch.setattr(verify, "restore_hidden", _noop)
    monkeypatch.setattr(verify, "take_offline", _noop)
    await verify.verify_taskmd(verifier, _strategy(native))
    assert not captured["refused"]
    assert captured["report"] == "correct report"
    assert not captured["excluded"]


async def _noop(*args, **kwargs):
    pass
