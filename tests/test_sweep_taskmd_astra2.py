"""Guards the PY8 evidence fixes against commit 363d815b."""

import shutil
from types import SimpleNamespace

import pytest

from benchflow.taskmd import verify


class LocalBox:
    async def download_file(self, source, target):
        shutil.copyfile(source, target)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


async def _noop(*args, **kwargs):
    pass


@pytest.mark.parametrize("declared", [False, True])
async def test_default_outputs_and_evidence_use_the_runtime_workdir(
    tmp_path, monkeypatch, declared
):
    """The image cwd is the default output and evidence base."""
    from benchflow.task.formats import materialize_task_dir
    from tests.test_taskmd_agent_judge import FIXTURE
    from tests.test_taskmd_verify import _strategy, _Verifier

    package = tmp_path / "task"
    shutil.copytree(FIXTURE, package)
    task = package / "task.md"
    task.write_text(
        task.read_text().replace(
            'workdir = "/app"\n', "" if not declared else 'workdir = "/app"\n'
        )
    )
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    native = materialize_task_dir(package)
    copied = []
    seen = {}

    async def execute(*args, **kwargs):
        return SimpleNamespace(stdout="/app\n", return_code=0)

    async def copy(box, paths, fs_root, modes):
        copied.extend(paths)
        return {}

    async def judged(**kwargs):
        seen.update(kwargs["config"]["sandbox"])
        return {"prints-42": {"_value": True, "verdict": "pass"}}, None, {}

    monkeypatch.setattr(verify, "root_exec", execute)
    monkeypatch.setattr(verify, "copy_kept", copy)
    monkeypatch.setattr(verify, "_run_judges", judged)
    monkeypatch.setattr(verify, "restore_hidden", _noop)
    monkeypatch.setattr(verify, "take_offline", _noop)
    verifier = _Verifier(native, tmp_path / "trial", LocalBox(), report=None)
    await verify.verify_taskmd(verifier, _strategy(native))
    assert copied == ["/app"]
    assert seen["workdir"] == "/app"
