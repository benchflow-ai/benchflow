"""Workspace-attachment behavior for ``bench traj upload``.

The feature zips the session's workspace folder into the capture as
``workspace/<name>.zip``: auto-detected from the session's recorded cwd,
optional and skippable, capped at 1 GiB before compression, archived without
content redaction but with VCS/dependency/secret-shaped names excluded, and
always cleaned up with the staging directory.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

import benchflow.publish.traj_capture as capture_module
from benchflow.cli.main import app
from benchflow.publish.traj_capture import (
    attach_workspace_archive,
    stage_trajectory_artifacts,
)
from services.trajectory_upload.contract import UploadRequest
from services.trajectory_upload.validation import (
    CaptureRejected,
    validate_local_capture,
)

runner = CliRunner()
GITHUB_ID = "benchflow-user"
EMAIL = "user@example.com"


def _session(tmp_path: Path, *, cwd: Path | None = None, codex: bool = False) -> Path:
    """A minimal real-shaped session JSONL, optionally recording a cwd."""
    lines = []
    if cwd is not None and codex:
        lines.append(json.dumps({"type": "session_meta", "payload": {"cwd": str(cwd)}}))
    elif cwd is not None:
        lines.append(json.dumps({"type": "user", "cwd": str(cwd)}))
    lines.append(json.dumps({"type": "message", "text": "demo step"}))
    session = tmp_path / "session.jsonl"
    session.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return session


def _workspace(tmp_path: Path) -> Path:
    folder = tmp_path / "demo-workspace"
    (folder / "src").mkdir(parents=True)
    (folder / "src" / "main.py").write_text("print('hi')\n")
    (folder / "README.md").write_text("# demo\n")
    (folder / ".git").mkdir()
    (folder / ".git" / "config").write_text("[remote]\n")
    (folder / "node_modules").mkdir()
    (folder / "node_modules" / "big.js").write_text("x" * 100)
    (folder / ".env").write_text("SECRET=1\n")
    (folder / "server.key").write_text("private\n")
    return folder


def _worktree_workspace(tmp_path: Path) -> Path:
    """A linked git worktree: ``.git`` is a pointer FILE, not a directory."""
    folder = tmp_path / "linked-worktree"
    folder.mkdir()
    (folder / ".git").write_text("gitdir: /Users/someone/repo/.git/worktrees/x\n")
    (folder / "code.py").write_text("print('hi')\n")
    return folder


def _upload_command(path: Path, *args: str) -> list[str]:
    return [
        "traj",
        "upload",
        str(path),
        "--github-id",
        GITHUB_ID,
        "--email",
        EMAIL,
        *args,
    ]


def test_attach_archives_workspace_and_excludes_vcs_and_secrets(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    folder = _workspace(tmp_path)
    with stage_trajectory_artifacts(session, source_id="demo") as artifacts:
        before_digest = artifacts.traj_digest
        result = attach_workspace_archive(artifacts, folder)

        assert result.skipped_reason is None
        assert result.attached is not None
        assert result.attached.relname == "workspace/demo-workspace.zip"
        assert result.attached.content_type == "application/zip"
        assert result.file_count == 2
        assert result.excluded_count >= 4  # .git, node_modules, .env, server.key
        assert result.artifacts.traj_digest != before_digest
        assert [item.relname for item in result.artifacts.files] == [
            "trajectory/session.jsonl",
            "workspace/demo-workspace.zip",
        ]
        with zipfile.ZipFile(result.attached.local_path) as archive:
            assert sorted(archive.namelist()) == ["README.md", "src/main.py"]
        zip_path = result.attached.local_path
        assert zip_path.exists()
    assert not zip_path.exists()  # staging exit always removes the archive


def test_attach_excludes_worktree_git_pointer_file(tmp_path: Path) -> None:
    """Guards the 0.7.4 live-matrix find: a linked worktree's .git pointer
    file (carrying a local absolute path) was zipped because only .git
    directories were excluded."""
    session = _session(tmp_path)
    folder = _worktree_workspace(tmp_path)
    with stage_trajectory_artifacts(session, source_id="demo") as artifacts:
        result = attach_workspace_archive(artifacts, folder)
        assert result.attached is not None
        with zipfile.ZipFile(result.attached.local_path) as archive:
            assert archive.namelist() == ["code.py"]
        assert result.excluded_count >= 1


def test_attach_skips_oversized_workspace_without_zipping(tmp_path: Path) -> None:
    session = _session(tmp_path)
    folder = _workspace(tmp_path)
    with stage_trajectory_artifacts(session, source_id="demo") as artifacts:
        result = attach_workspace_archive(artifacts, folder, max_bytes=4)

        assert result.attached is None
        assert result.skipped_reason is not None
        assert "before" in result.skipped_reason
        assert result.artifacts is artifacts  # untouched
        staged_root = artifacts._staging_dir
        assert not list(staged_root.glob("workspace/*"))  # nothing was created


def test_attach_skips_missing_and_empty_folders(tmp_path: Path) -> None:
    session = _session(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with stage_trajectory_artifacts(session, source_id="demo") as artifacts:
        missing = attach_workspace_archive(artifacts, tmp_path / "nope")
        assert missing.attached is None
        assert "not found" in (missing.skipped_reason or "")
        hollow = attach_workspace_archive(artifacts, empty)
        assert hollow.attached is None
        assert "no files" in (hollow.skipped_reason or "")


def test_attach_skips_symlinked_files(tmp_path: Path) -> None:
    session = _session(tmp_path)
    folder = tmp_path / "ws"
    folder.mkdir()
    (folder / "real.txt").write_text("data\n")
    (folder / "link.txt").symlink_to(folder / "real.txt")
    with stage_trajectory_artifacts(session, source_id="demo") as artifacts:
        result = attach_workspace_archive(artifacts, folder)
        assert result.attached is not None
        with zipfile.ZipFile(result.attached.local_path) as archive:
            assert archive.namelist() == ["real.txt"]


def test_contract_accepts_one_workspace_archive_and_rejects_extras() -> None:
    def request_body(artifacts: list[dict]) -> dict:
        from services.trajectory_upload.contract import trajectory_digest

        parsed = [
            SimpleNamespace(name=a["name"], sha256=a["sha256"]) for a in artifacts
        ]
        return {
            "schema_version": "1.1.0",
            "kind": "bronze.trajectory",
            "source_id": "demo",
            "traj_digest": f"sha256:{trajectory_digest(parsed)}",
            "uploaded_by": None,
            "artifacts": artifacts,
            "contributor": {"github_id": GITHUB_ID, "email": EMAIL},
        }

    jsonl = {"name": "trajectory/a.jsonl", "sha256": "a" * 64, "bytes": 10}
    archive = {"name": "workspace/ws.zip", "sha256": "b" * 64, "bytes": 900 * 1024**2}
    UploadRequest.model_validate(request_body([jsonl, archive]))

    second = {"name": "workspace/other.zip", "sha256": "c" * 64, "bytes": 5}
    with pytest.raises(ValueError, match="at most one workspace archive"):
        UploadRequest.model_validate(request_body([jsonl, archive, second]))
    with pytest.raises(ValueError, match="at least one trajectory artifact"):
        UploadRequest.model_validate(request_body([archive]))
    oversized_zip = dict(archive, bytes=1024**3 + 1)
    with pytest.raises(ValueError):
        UploadRequest.model_validate(request_body([jsonl, oversized_zip]))
    oversized_jsonl = dict(jsonl, bytes=129 * 1024**2)
    with pytest.raises(ValueError, match="exceeds"):
        UploadRequest.model_validate(request_body([jsonl, oversized_jsonl]))


def test_validator_accepts_zip_artifact_and_rejects_fake_zip(tmp_path: Path) -> None:
    session = _session(tmp_path)
    folder = tmp_path / "ws"
    folder.mkdir()
    (folder / "file.txt").write_text("content\n")
    with stage_trajectory_artifacts(session, source_id="demo") as artifacts:
        result = attach_workspace_archive(artifacts, folder)
        assert result.attached is not None
        from benchflow.publish.traj_capture import finalize_trajectory_capture

        staged = finalize_trajectory_capture(
            result.artifacts,
            github_id=GITHUB_ID,
            email=EMAIL,
        )
        manifest_bytes = staged.files[-1].local_path.read_bytes()
        paths = {
            item.relname: item.local_path
            for item in staged.files
            if item.relname != "manifest.json"
        }
        validated = validate_local_capture(manifest_bytes, paths)
        assert "workspace/ws.zip" in validated.artifact_paths

        # Corrupt the archive: hash mismatch must reject before zip checks.
        result.attached.local_path.write_bytes(b"not a zip")
        with pytest.raises(CaptureRejected, match=r"size mismatch|sha256 mismatch"):
            validate_local_capture(manifest_bytes, paths)


def test_schema_12_manifest_with_workspace_validates_end_to_end(
    tmp_path: Path,
) -> None:
    """Guards the live rejection from the first workspace deploy: the manifest
    model cross-checked the trajectory report's file count and size against
    ALL artifacts, so any capture carrying a workspace zip failed with
    'trajectory report file count does not match artifacts'. The report
    describes the JSONL alone on both sides."""
    from benchflow.publish.traj_capture import finalize_trajectory_capture
    from benchflow.publish.traj_report import build_trajectory_report

    session = _session(tmp_path)
    folder = tmp_path / "ws"
    folder.mkdir()
    (folder / "file.txt").write_text("content\n")
    with stage_trajectory_artifacts(session, source_id="demo") as artifacts:
        report = build_trajectory_report(
            artifacts.files,
            masked_values=artifacts.redaction_replacements,
            preview_steps=0,
        )
        result = attach_workspace_archive(artifacts, folder)
        assert result.attached is not None
        staged = finalize_trajectory_capture(
            result.artifacts,
            github_id=GITHUB_ID,
            email=EMAIL,
            trajectory_report=report.as_manifest_metadata(),
        )
        assert staged.manifest["schema_version"] == "1.2.0"
        manifest_bytes = staged.files[-1].local_path.read_bytes()
        paths = {
            item.relname: item.local_path
            for item in staged.files
            if item.relname != "manifest.json"
        }
        validated = validate_local_capture(manifest_bytes, paths)
        assert "workspace/ws.zip" in validated.artifact_paths


def test_validator_zip_magic_rejects_non_zip_bytes(tmp_path: Path) -> None:
    from services.trajectory_upload.validation import _validate_zip_magic

    fake = tmp_path / "fake.zip"
    fake.write_bytes(b"ZZZZ not a zip")
    with pytest.raises(CaptureRejected, match="not a zip"):
        _validate_zip_magic(fake, "workspace/fake.zip")
    real = tmp_path / "real.zip"
    with zipfile.ZipFile(real, "w") as archive:
        archive.writestr("a.txt", "x")
    _validate_zip_magic(real, "workspace/real.zip")


def test_cli_attaches_detected_workspace_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude-style cwd detection feeds the zip through the whole client flow."""
    folder = _workspace(tmp_path)
    session = _session(tmp_path, cwd=folder)
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")
    declared: list[str] = []
    put_names: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            declared.extend(a["name"] for a in body["artifacts"])
            from tests.test_traj_upload_cli import _broker_payload

            return httpx.Response(200, json=_broker_payload(request))
        put_names.append(request.url.path.rsplit("/", 2)[-2:][0])
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("benchflow.publish.broker.httpx.Client", lambda: client)
    result = runner.invoke(app, _upload_command(session, "--no-repo", "--no-wait"))

    assert result.exit_code == 0, result.output
    assert "Workspace attached: workspace/demo-workspace.zip" in result.output.replace(
        "\n", ""
    )
    assert "trajectory/session.jsonl" in declared
    assert "workspace/demo-workspace.zip" in declared


def test_cli_codex_session_meta_detection(tmp_path: Path) -> None:
    folder = _workspace(tmp_path)
    session = _session(tmp_path, cwd=folder, codex=True)
    from benchflow.cli.traj import _session_cwd

    assert _session_cwd(session) == folder


def test_cli_no_workspace_flag_skips_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _workspace(tmp_path)
    session = _session(tmp_path, cwd=folder)
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")
    result = runner.invoke(
        app, _upload_command(session, "--no-repo", "--no-workspace", "--dry-run")
    )

    assert result.exit_code == 0, result.output
    assert "workspace/" not in result.output
    assert "Workspace" not in result.output


def test_cli_dry_run_lists_workspace_archive(tmp_path: Path) -> None:
    folder = _workspace(tmp_path)
    session = _session(tmp_path, cwd=folder)
    result = runner.invoke(app, _upload_command(session, "--no-repo", "--dry-run"))

    assert result.exit_code == 0, result.output
    flattened = result.output.replace("\n", "")
    assert "Workspace attached: workspace/demo-workspace.zip" in flattened
    assert "workspace/demo-workspace.zip" in flattened


def test_cli_oversized_workspace_prints_skip_and_uploads_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _workspace(tmp_path)
    session = _session(tmp_path, cwd=folder)
    monkeypatch.setattr(capture_module, "MAX_ATTACHMENT_BYTES", 4)
    result = runner.invoke(app, _upload_command(session, "--no-repo", "--dry-run"))

    assert result.exit_code == 0, result.output
    flattened = result.output.replace("\n", "")
    assert "Workspace skipped:" in flattened
    assert "workspace/demo-workspace.zip" not in flattened
    assert "trajectory/session.jsonl" in flattened


def test_workspace_prompt_is_optional_and_validating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Detection failure on a TTY asks once; Enter skips, bad paths re-ask."""
    from benchflow.cli.traj import _resolve_workspace_dir, _UploadOptions

    session = _session(tmp_path)  # records no cwd
    options = _UploadOptions(
        path=session,
        github_id=GITHUB_ID,
        email=EMAIL,
        source_id=None,
        repo=False,
        direct=False,
        container_url=None,
        dry_run=True,
        preview_steps=0,
    )
    monkeypatch.setattr(
        "benchflow.cli.traj.sys.stdin", SimpleNamespace(isatty=lambda: True)
    )
    answers = iter(["", str(tmp_path / "missing"), str(tmp_path)])
    monkeypatch.setattr(
        "benchflow.cli.traj.typer.prompt", lambda *a, **k: next(answers)
    )

    chosen, prompted = _resolve_workspace_dir(options, session)
    assert chosen is None and prompted  # Enter skipped

    chosen, prompted = _resolve_workspace_dir(options, session)
    assert chosen == tmp_path and prompted  # re-asked after the bad path


def test_workspace_prompt_skipped_off_tty(tmp_path: Path) -> None:
    from benchflow.cli.traj import _resolve_workspace_dir, _UploadOptions

    session = _session(tmp_path)
    options = _UploadOptions(
        path=session,
        github_id=GITHUB_ID,
        email=EMAIL,
        source_id=None,
        repo=False,
        direct=False,
        container_url=None,
        dry_run=True,
        preview_steps=0,
    )
    with io.StringIO() as fake_stdin:
        assert not fake_stdin.isatty()
    chosen, prompted = _resolve_workspace_dir(options, session)
    assert chosen is None and not prompted
