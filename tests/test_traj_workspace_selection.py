"""Guards the fix at 363d815b: metadata is not consent."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchflow.cli.traj import _resolve_workspace_dir, _UploadOptions


@pytest.mark.parametrize("agent", ["claude", "codex"])
@pytest.mark.parametrize("tty", [False, True])
def test_recorded_directory_requires_explicit_selection(
    tmp_path, monkeypatch, agent, tty
):
    """Guards FF-sw-trajectories against both supported cwd spellings."""
    folder = tmp_path / "private"
    folder.mkdir()
    (folder / "dummy").write_text("DUMMY_PRIVATE_MARKER")
    session = tmp_path / "session.jsonl"
    event = (
        {"cwd": str(folder)}
        if agent == "claude"
        else {"type": "session_meta", "payload": {"cwd": str(folder)}}
    )
    session.write_text(json.dumps(event) + "\n")
    options = _UploadOptions(
        path=session,
        github_id="alice",
        email="alice@example.com",
        source_id=None,
        repo=False,
        direct=False,
        container_url=None,
        dry_run=True,
        preview_steps=0,
    )
    monkeypatch.setattr(
        "benchflow.cli.traj.sys.stdin", SimpleNamespace(isatty=lambda: tty)
    )
    monkeypatch.setattr("benchflow.cli.traj.typer.prompt", lambda *a, **kw: "")
    assert _resolve_workspace_dir(options, session) == (None, tty)
    options = replace(options, workspace_dir=folder)
    assert _resolve_workspace_dir(options, session) == (folder, False)
    options = replace(options, workspace=False)
    assert _resolve_workspace_dir(options, session) == (None, False)
