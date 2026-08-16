"""CLI and broker-protocol tests for ``bench traj upload``."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import click
import httpx
import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.publish.broker import upload_capture_via_broker
from benchflow.publish.traj_capture import stage_trajectory_capture

runner = CliRunner()
GITHUB_ID = "benchflow-user"
EMAIL = "user@example.com"


def _trial(tmp_path: Path) -> Path:
    trial = tmp_path / "trial-demo"
    trajectory = trial / "trajectory"
    trajectory.mkdir(parents=True)
    (trajectory / "acp_trajectory.jsonl").write_text(
        '{"type":"message","text":"demo"}\n', encoding="utf-8"
    )
    return trial


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


def _block_identity_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the prompt path: the dev machine's gh/git identity must not leak in."""
    monkeypatch.delenv("BENCHFLOW_GITHUB_ID", raising=False)
    monkeypatch.delenv("BENCHFLOW_EMAIL", raising=False)
    monkeypatch.setattr("benchflow.cli.traj._command_stdout", lambda *_args: None)


def test_stock_cli_has_the_verified_public_broker() -> None:
    """A wheel install can contribute without private endpoint configuration."""
    from benchflow.cli.traj import DEFAULT_TRAJ_BROKER_URL

    assert DEFAULT_TRAJ_BROKER_URL == (
        "https://tasksminer-traj-broker.nicewave-c3abaecf.westus2.azurecontainerapps.io"
    )


def _broker_payload(
    request: httpx.Request, *, objects: list[dict] | None = None
) -> dict:
    body = json.loads(request.content)
    digest = body["traj_digest"].removeprefix("sha256:")
    expected = [artifact["name"] for artifact in body["artifacts"]] + ["manifest.json"]
    return {
        "upload_id": "u_demo",
        "bucket": "bronze",
        "base_url": "https://tasksminerdata.blob.core.windows.net/bronze",
        "prefix": f"inbox/{digest}/",
        "objects": objects
        or [
            {
                "name": name,
                "put_url": f"https://upload.test/{name}",
                "headers": {"x-ms-blob-type": "BlockBlob", "If-None-Match": "*"},
            }
            for name in expected
        ],
        "expires_at": "2026-08-15T12:00:00Z",
    }


def test_dry_run_stages_without_constructing_a_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run lists canonical files and never constructs a network client."""
    trial = _trial(tmp_path)
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")

    def fail_client(*args, **kwargs):
        raise AssertionError("network client constructed during --dry-run")

    monkeypatch.setattr(httpx, "Client", fail_client)
    result = runner.invoke(app, _upload_command(trial, "--dry-run"))

    assert result.exit_code == 0, result.output
    assert "sha256:" in result.output
    assert "trajectory/acp_trajectory.jsonl" in result.output
    assert "manifest.json" in result.output
    assert EMAIL not in result.output
    assert "https://broker.test" not in result.output
    assert "no files uploaded" in result.output


def test_direct_mode_reports_azure_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI delegates direct mode and renders the returned Azure URL."""
    trial = _trial(tmp_path)

    def fake_upload(staged, *, container_url, on_file_complete, on_bytes):
        assert staged.manifest["contributor"] == {
            "github_id": GITHUB_ID,
            "email": EMAIL,
        }
        assert staged.manifest["schema_version"] == "1.2.0"
        assert staged.manifest["trajectory_report"]["primary_file"] == (
            "trajectory/acp_trajectory.jsonl"
        )
        for staged_file in staged.files:
            on_file_complete(staged_file)
        return SimpleNamespace(
            url=f"{container_url}/sources/demo/{staged.traj_digest}/",
            uploaded=("payload", "manifest"),
            skipped=(),
        )

    monkeypatch.setattr(
        "benchflow.publish.azure_blob.upload_capture_direct", fake_upload
    )
    result = runner.invoke(
        app,
        _upload_command(
            trial,
            "--direct",
            "--container-url",
            "https://tasksminerdata.blob.core.windows.net/bronze",
        ),
    )

    assert result.exit_code == 0, result.output
    assert "Uploaded trajectory" in result.output
    assert "tasksminerdata.blob.core.windows.net/bronze" in result.output
    assert "Upload this trajectory?" not in result.output
    assert "Upload complete" in result.output


def test_broker_mode_uses_exact_manifest_and_server_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Broker mode sends the manifest handshake and returned PUT headers verbatim."""
    trial = _trial(tmp_path)
    requests: list[httpx.Request] = []
    manifest_sha256 = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal manifest_sha256
        requests.append(request)
        if request.method == "POST":
            body = json.loads(request.content)
            assert set(body) == {
                "schema_version",
                "kind",
                "source_id",
                "traj_digest",
                "uploaded_by",
                "contributor",
                "artifacts",
                "manifest_sha256",
            }
            assert body["contributor"] == {
                "github_id": GITHUB_ID,
                "email": EMAIL,
            }
            assert body["schema_version"] == "1.2.0"
            manifest_sha256 = body["manifest_sha256"]
            return httpx.Response(200, json=_broker_payload(request))
        if request.url.path.endswith("manifest.json"):
            assert hashlib.sha256(request.content).hexdigest() == manifest_sha256
            manifest = json.loads(request.content)
            assert manifest["contributor"] == {
                "github_id": GITHUB_ID,
                "email": EMAIL,
            }
            assert manifest["trajectory_report"]["total_steps"] == 1
            assert manifest["trajectory_report"]["preview"] == [
                {"kind": "Assistant", "number": 1, "summary": "demo"}
            ]
        assert request.headers["x-ms-blob-type"] == "BlockBlob"
        assert request.headers["if-none-match"] == "*"
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("benchflow.publish.broker.httpx.Client", lambda: client)
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")
    result = runner.invoke(app, _upload_command(trial))

    assert result.exit_code == 0, result.output
    assert [request.method for request in requests] == ["POST", "PUT", "PUT"]
    assert requests[-1].url.path.endswith("manifest.json")


def test_broker_never_logs_signed_upload_urls(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Signed SAS query parameters never enter BenchFlow's global INFO log."""
    trial = _trial(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            payload = _broker_payload(request)
            for item in payload["objects"]:
                item["put_url"] += "?sig=must-not-be-logged"
            return httpx.Response(200, json=payload)
        return httpx.Response(201)

    caplog.set_level(logging.INFO)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        upload_capture_via_broker(
            staged,
            broker_url="https://broker.test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    assert "must-not-be-logged" not in caplog.text


def test_broker_conflict_is_success_and_rate_limit_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ingested digest no-ops while rate limits preserve Retry-After."""
    trial = _trial(tmp_path)
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")

    conflict = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                409,
                json={
                    "base_url": "https://tasksminerdata.blob.core.windows.net/bronze",
                    "prefix": "sources/community/demo/",
                },
            )
        )
    )
    limited = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                429, text="slow down", headers={"Retry-After": "60"}
            )
        )
    )
    monkeypatch.setattr("benchflow.publish.broker.httpx.Client", lambda: conflict)
    result = runner.invoke(app, _upload_command(trial))
    assert result.exit_code == 0, result.output
    assert "Already submitted" in result.output
    assert "blob.core.windows.net" not in result.output

    monkeypatch.setattr("benchflow.publish.broker.httpx.Client", lambda: limited)
    result = runner.invoke(app, _upload_command(trial))
    assert result.exit_code == 1
    assert "retry after 60" in result.output


def test_missing_broker_names_both_available_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A development build without a default endpoint explains both modes."""
    trial = _trial(tmp_path)
    monkeypatch.delenv("BENCHFLOW_TRAJ_BROKER_URL", raising=False)
    monkeypatch.setattr("benchflow.cli.traj.DEFAULT_TRAJ_BROKER_URL", None)
    result = runner.invoke(app, _upload_command(trial))

    assert result.exit_code == 1
    assert "BENCHFLOW_TRAJ_BROKER_URL" in result.output
    assert "--direct" in result.output
    assert "BENCHFLOW_AZURE_CONTAINER_URL" in result.output


def test_validation_failure_names_the_bad_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed contributor JSONL exits cleanly and identifies its source."""
    trial = _trial(tmp_path)
    path = trial / "trajectory" / "acp_trajectory.jsonl"
    path.write_text("{bad\n", encoding="utf-8")
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")
    result = runner.invoke(app, _upload_command(trial))

    assert result.exit_code == 1
    assert "acp_trajectory.jsonl" in result.output.replace("\n", "")
    assert "line 1" in result.output


@pytest.mark.parametrize("shape", ["unknown", "missing", "insecure_url"])
def test_broker_mapping_violation_sends_zero_puts(tmp_path: Path, shape: str) -> None:
    """A non-bijective broker response fails before any trajectory bytes leave."""
    trial = _trial(tmp_path)
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        payload = _broker_payload(request)
        if shape == "unknown":
            payload["objects"][0]["name"] = "trajectory/unknown.jsonl"
        elif shape == "missing":
            payload["objects"].pop()
        else:
            payload["objects"][0]["put_url"] = "http://upload.test/capture"
        return httpx.Response(200, json=payload)

    with (
        stage_trajectory_capture(trial, source_id="demo") as staged,
        pytest.raises(ValueError, match="protocol violation"),
    ):
        upload_capture_via_broker(
            staged,
            broker_url="https://broker.test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    assert methods == ["POST"]


@pytest.mark.parametrize("status", [409, 412])
def test_broker_put_conflicts_are_cloud_neutral_skips(
    tmp_path: Path, status: int
) -> None:
    """Azure 409 and GCS 412 both mean an idempotent create-only skip."""
    trial = _trial(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json=_broker_payload(request))
        return httpx.Response(status)

    with stage_trajectory_capture(trial, source_id="demo") as staged:
        completed: list[str] = []
        result = upload_capture_via_broker(
            staged,
            broker_url="https://broker.test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            on_file_complete=lambda staged_file: completed.append(staged_file.relname),
        )
    assert not result.uploaded
    assert len(result.skipped) == len(staged.files)
    assert completed == [item.relname for item in staged.files]


def test_help_exposes_setup_and_upload() -> None:
    """Guards PR #992 while ignoring Rich's environment-specific ANSI styling."""
    traj_group = next(group for group in app.registered_groups if group.name == "traj")
    assert {
        command.name for command in traj_group.typer_instance.registered_commands
    } == {"setup", "upload"}

    result = runner.invoke(app, ["traj", "--help"])
    assert result.exit_code == 0
    assert "upload" in result.output
    assert "setup" in result.output

    upload_help = runner.invoke(app, ["traj", "upload", "--help"])
    assert upload_help.exit_code == 0
    upload_help_output = click.unstyle(upload_help.output)
    assert "--github-id" in upload_help_output
    assert "--email" in upload_help_output
    assert "--preview-steps" in upload_help_output


def test_upload_prompts_for_path_github_id_and_email_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the interactive upload follow-up to PR #992."""
    _block_identity_inference(monkeypatch)
    trial = _trial(tmp_path)

    def fake_upload(staged, *, broker_url, on_file_complete, on_bytes):
        assert staged.manifest["contributor"] == {
            "github_id": GITHUB_ID,
            "email": EMAIL,
        }
        for staged_file in staged.files:
            on_file_complete(staged_file)
        return SimpleNamespace(
            url=f"{broker_url}/sources/community/{staged.traj_digest}/",
            uploaded=("payload", "manifest"),
            skipped=(),
        )

    monkeypatch.setattr(
        "benchflow.publish.broker.upload_capture_via_broker", fake_upload
    )
    result = runner.invoke(
        app,
        ["traj", "upload"],
        input=f"{trial}\n{GITHUB_ID}\n{EMAIL}\ny\n",
    )

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output)
    assert (
        output.index("Trajectory JSONL file or trial directory")
        < output.index("Trajectory report")
        < output.index("GitHub ID")
        < output.index("Email")
    )
    assert "Upload this trajectory?" in output
    assert "Submitted" in output
    assert "sources/community" not in output  # public success copy hides URLs


def test_interactive_preview_can_cancel_before_the_upload_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the trajectory-report follow-up to PR #992 confirmation gate."""
    _block_identity_inference(monkeypatch)
    trial = _trial(tmp_path)

    def fail_upload(*args, **kwargs):
        raise AssertionError("upload started after the contributor declined")

    monkeypatch.setattr(
        "benchflow.publish.broker.upload_capture_via_broker", fail_upload
    )
    result = runner.invoke(
        app,
        ["traj", "upload"],
        input=f"{trial}\n{GITHUB_ID}\n{EMAIL}\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Upload cancelled" in click.unstyle(result.output)


def test_cli_report_shows_redacted_preview_and_requested_step_counts(
    tmp_path: Path,
) -> None:
    """Guards the trajectory-report follow-up to PR #992 terminal report."""
    trial = _trial(tmp_path)
    secret = "sk-1234567890abcdefghijklmnop"
    trajectory = trial / "trajectory" / "acp_trajectory.jsonl"
    trajectory.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {"type": "user_message", "text": f"API_KEY={secret}"},
                {"type": "agent_thought", "text": "Inspect first"},
                {"type": "tool_call", "kind": "read", "title": "Open README"},
                {"type": "agent_message", "text": "Done"},
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        _upload_command(trial, "--dry-run", "--preview-steps", "2"),
    )

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output)
    assert "Trajectory report" in output
    assert "Total steps" in output and "4" in output
    assert "Thinking steps" in output
    assert "Tool-call steps" in output
    assert "Human steps" in output
    assert "API keys / secrets masked" in output
    assert "<XXX-benchflow-key-values-XXX>" in output
    assert "First 2 trajectory steps" in output
    assert "up to 100 words each" in output
    assert secret not in output


def test_upload_prompts_only_for_missing_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #992's explicit form while adding partial interactive input."""
    _block_identity_inference(monkeypatch)
    trial = _trial(tmp_path)

    result = runner.invoke(
        app,
        [
            "traj",
            "upload",
            str(trial),
            "--github-id",
            GITHUB_ID,
            "--dry-run",
        ],
        input=f"{EMAIL}\n",
    )

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output)
    assert "Email:" in output
    assert "GitHub ID:" not in output
    assert "Trajectory JSONL file or trial directory:" not in output


def test_interactive_prompts_reask_after_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1008: a typo at any interactive prompt re-asks in place
    instead of aborting the staged upload, and dragged-in quoted paths are
    accepted as typed."""
    _block_identity_inference(monkeypatch)
    trial = _trial(tmp_path)

    result = runner.invoke(
        app,
        ["traj", "upload", "--dry-run"],
        input=(
            f"{tmp_path / 'missing'}\n"
            f"'{trial}'\n"
            "-bad-\n"
            f"{GITHUB_ID}\n"
            "not-an-email\n"
            f"{EMAIL}\n"
        ),
    )

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output)
    assert "path not found" in output
    assert "invalid GitHub ID" in output
    assert "invalid contributor email" in output
    assert "Dry run" in output


def test_broker_upload_reports_streamed_byte_progress(tmp_path: Path) -> None:
    """Guards PR #1008: single-file uploads stream byte counts to the progress
    callback instead of jumping only at file boundaries."""
    trial = _trial(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json=_broker_payload(request))
        request.read()
        return httpx.Response(201)

    byte_counts: list[int] = []
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        upload_capture_via_broker(
            staged,
            broker_url="https://broker.test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            on_bytes=byte_counts.append,
        )
        assert sum(byte_counts) == sum(item.size_bytes for item in staged.files)
    assert all(count > 0 for count in byte_counts)


def test_upload_rejects_preview_counts_above_the_terminal_bound(tmp_path: Path) -> None:
    """Guards the trajectory-report follow-up to PR #992 preview bound."""
    result = runner.invoke(
        app,
        _upload_command(_trial(tmp_path), "--dry-run", "--preview-steps", "21"),
    )

    assert result.exit_code == 2
    assert "20" in click.unstyle(result.output)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--github-id", "@not-a-github-id", "--email", EMAIL), "GitHub ID"),
        (("--github-id", GITHUB_ID, "--email", "not-an-email"), "email"),
    ],
)
def test_upload_validates_contributor_parameters_locally(
    tmp_path: Path, args: tuple[str, ...], message: str
) -> None:
    """Malformed contributor provenance fails before the upload handshake."""
    result = runner.invoke(
        app,
        ["traj", "upload", str(_trial(tmp_path)), *args, "--dry-run"],
    )

    assert result.exit_code == 1
    assert message in result.output


def test_upload_infers_contributor_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-argument command uses local identity when flags are omitted."""
    _block_identity_inference(monkeypatch)
    monkeypatch.setenv("BENCHFLOW_GITHUB_ID", GITHUB_ID)
    monkeypatch.setenv("BENCHFLOW_EMAIL", EMAIL)
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")
    result = runner.invoke(app, ["traj", "upload", str(_trial(tmp_path)), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert EMAIL not in result.output


def test_upload_explains_missing_contributor_without_typer_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing identity with no prompt input gives the exact one-line fix."""
    _block_identity_inference(monkeypatch)
    result = runner.invoke(app, ["traj", "upload", str(_trial(tmp_path))])

    assert result.exit_code == 1
    output = click.unstyle(result.output)
    assert "need a GitHub username and email" in output
    assert "--github-id YOUR_ID --email YOU@example.com" in output


def test_handshake_timeout_tells_people_to_retry(tmp_path: Path) -> None:
    """A cold scale-to-zero broker should not look like a broken install."""
    trial = _trial(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with (
        stage_trajectory_capture(
            trial, source_id="demo", github_id=GITHUB_ID, email=EMAIL
        ) as staged,
        pytest.raises(ValueError, match="retries are safe"),
    ):
        upload_capture_via_broker(
            staged,
            broker_url="https://broker.test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_setup_prompt_prints_the_copy_paste_line() -> None:
    """The human path is one line to paste into an agent."""
    from benchflow.cli.traj import CONTRIBUTOR_PROMPT, SKILL_RAW_URL

    result = runner.invoke(app, ["traj", "setup", "--prompt"])
    readme = Path(__file__).resolve().parents[1] / "README.md"

    assert result.exit_code == 0, result.output
    assert CONTRIBUTOR_PROMPT in click.unstyle(result.output)
    assert SKILL_RAW_URL in result.output
    assert "bench traj upload" not in result.output
    assert CONTRIBUTOR_PROMPT in readme.read_text(encoding="utf-8")


def test_setup_yes_installs_the_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optional interactive setup can run non-interactively with --yes."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["traj", "setup", "--yes"])

    assert result.exit_code == 0, result.output
    skill = tmp_path / ".agents" / "skills" / "benchflow-traj-upload" / "SKILL.md"
    assert skill.is_file()
    text = skill.read_text(encoding="utf-8")
    assert "open the viewer" in text
    assert "Paste this to your agent" in click.unstyle(result.output)


def test_list_recent_sessions_scans_all_projects_most_recent_first(
    tmp_path: Path,
) -> None:
    """Sessions from other project dirs must be found: people submit from a
    different directory than the one they worked in."""
    import os

    from benchflow.trajectories.sessions import (
        encode_claude_project_dir,
        list_recent_sessions,
    )

    cwd = tmp_path / "proj"
    cwd.mkdir()
    home = tmp_path / "home"
    project = home / ".claude" / "projects" / encode_claude_project_dir(str(cwd))
    project.mkdir(parents=True)
    older = project / "abc.jsonl"
    older.write_text(
        '{"type":"user","message":{"content":"prize session please"}}\n',
        encoding="utf-8",
    )
    os.utime(older, (1_000_000, 1_000_000))
    newer = home / ".claude" / "projects" / "-tmp-bio-work" / "bio.jsonl"
    newer.parent.mkdir(parents=True)
    newer.write_text(
        '{"type":"user","message":{"content":"compute GC content of sample.fasta"}}\n',
        encoding="utf-8",
    )
    os.utime(newer, (2_000_000, 2_000_000))

    hits = list_recent_sessions(cwd=cwd, home=home, limit=8)

    assert [hit.path for hit in hits] == [newer, older]
    assert all(hit.source == "claude" for hit in hits)
    assert "GC content" in hits[0].snippet
    assert "prize session please" in hits[1].snippet
