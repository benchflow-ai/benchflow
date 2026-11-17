"""A local trial or job folder is never attached as a workspace.

``bench traj upload <trial> --workspace-dir <folder>`` zips the folder as it
is, with nothing masked. Pointed at a local trial folder, that shipped the
agent's files raw (review R307B on 8dcbc94ce: a planted ``password=`` line in
plain text). A trial folder, a job folder, or any folder inside one is refused
as the workspace, at ``--workspace-dir``, at the prompt and in
``attach_workspace_archive`` itself (the lead's ruling, 2026-11-17 00:28Z).

Every value here is made up.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.workspace_guard import WorkspaceRefusedError, workspace_refusal
from benchflow.cli.traj import _resolve_workspace_dir, _UploadOptions
from benchflow.publish.traj_capture import (
    attach_workspace_archive,
    stage_trajectory_artifacts,
)

PLANTED = "password=FX307Cmadeup0123456789"


def _trial(parent: Path, name: str = "task-a__trial-1") -> Path:
    """A trial folder as an eval writes it, with a planted secret in agent/."""
    trial = parent / name
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "notes.txt").write_text(PLANTED + "\n")
    (trial / "config.json").write_text(json.dumps({"agent": "fake"}))
    (trial / "result.json").write_text(json.dumps({"rewards": {"reward": 1.0}}))
    return trial


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "jobs" / "job-1"
    job.mkdir(parents=True)
    (job / "summary.json").write_text(json.dumps({"total": 1}))
    (job / "worker-shards").mkdir()
    _trial(job)
    return job


def _project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    (folder / "src").mkdir(parents=True)
    (folder / "src" / "main.py").write_text("print('hi')\n")
    (folder / "README.md").write_text("# demo\n")
    return folder


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps({"type": "message", "text": "demo"}) + "\n")
    return session


def _options(session: Path, workspace_dir: Path | None) -> _UploadOptions:
    return _UploadOptions(
        path=session,
        github_id="alice",
        email="alice@example.com",
        source_id=None,
        repo=False,
        direct=False,
        container_url=None,
        dry_run=True,
        preview_steps=0,
        workspace_dir=workspace_dir,
    )


def _refused_paths(tmp_path: Path) -> dict[str, Path]:
    trial = _trial(tmp_path / "local")
    job = _job(tmp_path)
    return {
        "trial": trial,
        "inside a trial": trial / "agent",
        "job": job,
        "inside a job": job / "worker-shards",
        "trial in a job": next(job.glob("*__trial-1")),
    }


@pytest.mark.parametrize(
    "which", ["trial", "inside a trial", "job", "inside a job", "trial in a job"]
)
def test_a_run_folder_is_refused_at_workspace_dir(tmp_path, which):
    folder = _refused_paths(tmp_path)[which]
    session = _session(tmp_path)
    with pytest.raises(WorkspaceRefusedError) as refused:
        _resolve_workspace_dir(_options(session, folder), session)
    message = str(refused.value)
    assert "uploaded as it is, with nothing masked" in message
    assert "project folder the agent worked in" in message


@pytest.mark.parametrize(
    "which", ["trial", "inside a trial", "job", "inside a job", "trial in a job"]
)
def test_a_run_folder_is_never_zipped(tmp_path, which):
    folder = _refused_paths(tmp_path)[which]
    with stage_trajectory_artifacts(_session(tmp_path), source_id="demo") as staged:
        with pytest.raises(WorkspaceRefusedError):
            attach_workspace_archive(staged, folder)
        assert [item.relname for item in staged.files] == ["trajectory/session.jsonl"]


def test_a_run_folder_is_refused_at_the_prompt(tmp_path, monkeypatch, capsys):
    trial = _trial(tmp_path / "local")
    project = _project(tmp_path)
    answers = iter([str(trial / "agent"), str(project)])
    monkeypatch.setattr(
        "benchflow.cli.traj.sys.stdin", SimpleNamespace(isatty=lambda: True)
    )
    monkeypatch.setattr(
        "benchflow.cli.traj.typer.prompt", lambda *a, **kw: next(answers)
    )
    session = _session(tmp_path)
    assert _resolve_workspace_dir(_options(session, None), session) == (project, True)
    printed = capsys.readouterr()
    assert "with nothing masked" in printed.out + printed.err


def test_a_project_folder_still_uploads(tmp_path):
    project = _project(tmp_path)
    session = _session(tmp_path)
    assert _resolve_workspace_dir(_options(session, project), session) == (
        project,
        False,
    )
    with stage_trajectory_artifacts(session, source_id="demo") as staged:
        result = attach_workspace_archive(staged, project)
        assert result.attached is not None
        with zipfile.ZipFile(result.attached.local_path) as archive:
            assert sorted(archive.namelist()) == ["README.md", "src/main.py"]


def test_a_folder_beside_a_stray_trial_is_not_refused(tmp_path):
    """A home folder that holds a trial folder is a job and refused itself,
    but a project folder beside the trial is not: a parent counts as a job
    only by its own record."""
    home = tmp_path / "home"
    _trial(home)
    project = _project(home)
    assert workspace_refusal(home) is not None
    assert workspace_refusal(project) is None


def _eval_layout(tmp_path: Path, record: bool = True) -> Path:
    """``project/jobs/myjob/<time>/<trial>`` as ``bench eval`` writes it in a
    project (summary.json at ``<time>``), or as a Rollout writes it directly
    (no record)."""
    project = _project(tmp_path)
    run = project / "jobs" / "myjob" / "2026-11-17__00-40-00"
    run.mkdir(parents=True)
    if record:
        (run / "summary.json").write_text(json.dumps({"total": 1}))
    _trial(run)
    (_trial(run, "task-b__trial-1") / "agent" / "oracle.txt").write_text(PLANTED)
    return project


def _zipped(tmp_path: Path, folder: Path):
    with stage_trajectory_artifacts(_session(tmp_path), source_id="demo") as staged:
        result = attach_workspace_archive(staged, folder)
        names: list[str] = []
        planted = False
        if result.attached is not None:
            with zipfile.ZipFile(result.attached.local_path) as archive:
                names = sorted(archive.namelist())
                planted = any(PLANTED.encode() in archive.read(n) for n in names)
        return result, names, planted


@pytest.mark.parametrize("record", [True, False], ids=["eval", "rollout-only"])
@pytest.mark.parametrize("where", ["project", "jobs", "jobs/myjob"])
def test_run_folders_inside_a_workspace_are_left_out(tmp_path, where, record):
    project = _eval_layout(tmp_path, record=record)
    folder = project if where == "project" else project / where
    assert workspace_refusal(folder) is None
    result, names, planted = _zipped(tmp_path, folder)
    assert not planted
    assert not any("__trial-" in name or "agent/" in name for name in names)
    if where == "project":
        assert result.run_folders_excluded == 1
        assert names == ["README.md", "src/main.py"]
    else:
        # Nothing is left once the run folder is out: nothing is uploaded.
        assert result.attached is None
        assert "1 trial or job folder(s) are left out" in result.skipped_reason


def test_a_rollout_only_job_folder_is_refused(tmp_path):
    run = (
        _eval_layout(tmp_path, record=False) / "jobs" / "myjob" / "2026-11-17__00-40-00"
    )
    with pytest.raises(WorkspaceRefusedError):
        attach_workspace_archive_from(tmp_path, run)


def attach_workspace_archive_from(tmp_path: Path, folder: Path):
    with stage_trajectory_artifacts(_session(tmp_path), source_id="demo") as staged:
        return attach_workspace_archive(staged, folder)
