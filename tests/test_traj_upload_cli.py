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


def _enable_update_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the suite-wide BENCHFLOW_SKIP_UPDATE_CHECK conftest fixture."""
    monkeypatch.delenv("BENCHFLOW_SKIP_UPDATE_CHECK", raising=False)


def test_outdated_install_prints_the_upgrade_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1014 (the version-check follow-up to PR #1013): an install
    older than the PyPI latest gets one hint line on setup and upload."""
    _enable_update_check(monkeypatch)
    monkeypatch.setattr("benchflow.cli.traj._installed_version", lambda: "0.7.0")
    monkeypatch.setattr("benchflow.cli.traj._fetch_latest_version", lambda: "0.7.2")

    hint = (
        "A newer BenchFlow (0.7.2) is available — run: "
        "uv tool install --python 3.12 --upgrade --force benchflow"
    )
    setup_result = runner.invoke(app, ["traj", "setup", "--prompt"])
    assert setup_result.exit_code == 0, setup_result.output
    assert hint in click.unstyle(setup_result.output)

    upload_result = runner.invoke(app, _upload_command(_trial(tmp_path), "--dry-run"))
    assert upload_result.exit_code == 0, upload_result.output
    assert hint in click.unstyle(upload_result.output)


def test_update_check_network_failure_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards PR #1014 (the version-check follow-up to PR #1013): a PyPI
    failure prints nothing and leaves the exit code untouched."""
    from benchflow.cli.traj import CONTRIBUTOR_PROMPT

    _enable_update_check(monkeypatch)

    def fail_fetch() -> str | None:
        raise httpx.ConnectError("no network")

    monkeypatch.setattr("benchflow.cli.traj._fetch_latest_version", fail_fetch)
    result = runner.invoke(app, ["traj", "setup", "--prompt"])

    assert result.exit_code == 0, result.output
    assert "newer BenchFlow" not in result.output
    assert CONTRIBUTOR_PROMPT in click.unstyle(result.output)


@pytest.mark.parametrize(
    "installed", ["0.7.2", "0.8.0", "0.7.2.dev0", "0.7.3.dev1", "0.7.2rc1"]
)
def test_current_or_newer_dev_install_prints_no_hint(
    installed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1014 (the version-check follow-up to PR #1013): up-to-date
    installs and dev/prereleases of a newer-or-equal base are not outdated."""
    _enable_update_check(monkeypatch)
    monkeypatch.setattr("benchflow.cli.traj._installed_version", lambda: installed)
    monkeypatch.setattr("benchflow.cli.traj._fetch_latest_version", lambda: "0.7.2")
    result = runner.invoke(app, ["traj", "setup", "--prompt"])

    assert result.exit_code == 0, result.output
    assert "newer BenchFlow" not in result.output


def test_skip_update_check_env_var_never_fetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards PR #1014 (the version-check follow-up to PR #1013): the conftest
    hermeticity guard BENCHFLOW_SKIP_UPDATE_CHECK short-circuits the fetch."""

    def fail_fetch() -> str | None:
        raise AssertionError("update check fetched despite skip env var")

    monkeypatch.setattr("benchflow.cli.traj._fetch_latest_version", fail_fetch)
    result = runner.invoke(app, ["traj", "setup", "--prompt"])

    assert result.exit_code == 0, result.output
    assert "newer BenchFlow" not in result.output


def test_setup_prompt_prints_the_copy_paste_line() -> None:
    """The human path is one prompt block to send to an agent.

    Guards the version step and hand-off framing added in PR #1017
    (follow-up to PRs #1013 and #1014): the prompt tells the agent to
    upgrade BenchFlow before reading the skill, the upgrade command appears
    before the skill URL, and the URL sits on one unbroken physical line.
    README.md renders the prompt as a blockquote (a fenced code block would
    not soft-wrap on GitHub), so each unwrapped prompt line is asserted to be
    present in the README instead of the whole string verbatim.
    """
    from benchflow.cli.traj import (
        CONTRIBUTOR_PROMPT,
        CONTRIBUTOR_PROMPT_FRAMING,
        SKILL_RAW_URL,
        UPGRADE_COMMAND,
    )

    result = runner.invoke(app, ["traj", "setup", "--prompt"])
    readme = Path(__file__).resolve().parents[1] / "README.md"
    readme_text = readme.read_text(encoding="utf-8")

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output)
    assert CONTRIBUTOR_PROMPT_FRAMING in output
    assert CONTRIBUTOR_PROMPT in output
    assert SKILL_RAW_URL in result.output
    assert "bench traj upload" not in result.output

    prompt_lines = [line for line in CONTRIBUTOR_PROMPT.splitlines() if line]
    assert len(prompt_lines) == 3
    for line in prompt_lines:
        assert line in readme_text
    # Wording updated on main in 8cae2a42 ("Send these to your coding agent.").
    assert "Send these to your coding agent" in readme_text

    assert UPGRADE_COMMAND in CONTRIBUTOR_PROMPT
    assert CONTRIBUTOR_PROMPT.index(UPGRADE_COMMAND) < CONTRIBUTOR_PROMPT.index(
        SKILL_RAW_URL
    )
    assert any(SKILL_RAW_URL in line for line in prompt_lines)


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
    # PR #1017 reframed the hand-off copy: the block is a prompt for the
    # agent, not steps for the human.
    assert "Send this to your coding agent" in click.unstyle(result.output)


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


def test_setup_list_prints_path_on_its_own_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the collector-audit friction fix from PR #1024: ``bench traj
    setup --list`` used Rich wrapping on ``index. [source] when  /very/long/path``,
    which split session paths mid-token and made them unselectable. The path
    must sit on its own physical line after the index/source/time line."""
    from benchflow.trajectories.sessions import SessionHit

    long_path = Path(
        "/Users/someone/.claude/projects/"
        "-Users-someone-very-long-project-name-that-used-to-wrap/"
        "01234567-89ab-cdef-0123-456789abcdef.jsonl"
    )
    monkeypatch.setattr(
        "benchflow.trajectories.sessions.list_recent_sessions",
        lambda: [
            SessionHit(
                path=long_path,
                source="claude",
                mtime=1_700_000_000,
                snippet="fit a Michaelis-Menten curve",
            )
        ],
    )
    result = runner.invoke(app, ["traj", "setup", "--list"])

    assert result.exit_code == 0, result.output
    lines = [line.rstrip() for line in click.unstyle(result.output).splitlines()]
    assert f"   {long_path}" in lines
    path_line = next(line for line in lines if str(long_path) in line)
    assert path_line.strip() == str(long_path)
    assert "fit a Michaelis-Menten curve" in result.output


def test_broker_upload_does_not_claim_the_service_is_waking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the collector-audit friction fix from PR #1024: a warm rerun
    still printed ``while the service wakes up``, which is stale once the
    broker is up. The progress line must stay honest on both cold and warm
    runs."""
    captured: dict[str, str] = {}
    _capture_broker_source_id(monkeypatch, captured)
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")
    result = runner.invoke(app, _upload_command(_trial(tmp_path)))

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output)
    assert "wakes up" not in output
    assert "Uploading… this can take up to a minute; retries are safe." in output


def _repo_line(session_cwd: Path) -> str:
    return (
        f"Repo: benchflow-ai/benchflow (from session cwd {session_cwd}; "
        "use --no-repo to omit)"
    )


def _session_with_cwd(tmp_path: Path, event: dict) -> Path:
    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps(event) + "\n", encoding="utf-8")
    return session


def _git_remote_stub(url: str | None, workdir: Path):
    """A ``_command_stdout`` stub answering only the origin-remote lookup."""

    def fake(*args: str) -> str | None:
        if args == ("git", "-C", str(workdir), "remote", "get-url", "origin"):
            return url
        return None

    return fake


def _capture_broker_source_id(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, str]
) -> None:
    def fake_upload(staged, *, broker_url, on_file_complete, on_bytes):
        captured["source_id"] = staged.manifest["source_id"]
        for staged_file in staged.files:
            on_file_complete(staged_file)
        return SimpleNamespace(url=broker_url, uploaded=("payload",), skipped=())

    monkeypatch.setattr(
        "benchflow.publish.broker.upload_capture_via_broker", fake_upload
    )


@pytest.mark.parametrize(
    ("event_for_cwd", "remote"),
    [
        (
            lambda cwd: {"type": "user", "cwd": cwd, "message": {"content": "hi"}},
            "https://github.com/benchflow-ai/benchflow.git",
        ),
        (
            lambda cwd: {"type": "session_meta", "payload": {"cwd": cwd}},
            "git@github.com:benchflow-ai/benchflow.git",
        ),
    ],
    ids=["claude-https", "codex-ssh"],
)
def test_upload_tags_source_id_with_the_session_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event_for_cwd, remote: str
) -> None:
    """Guards PR #1015 (repo tagging follow-up to PR #1013): by default the
    upload's source id becomes ``repo/<owner>/<name>``, resolved from the
    session's recorded cwd git remote, and the CLI prints the repo line
    naming the session cwd it came from (terminal output only — the local
    path never enters the uploaded source id)."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    session = _session_with_cwd(tmp_path, event_for_cwd(str(workdir)))
    monkeypatch.setattr(
        "benchflow.cli.traj._command_stdout", _git_remote_stub(remote, workdir)
    )
    captured: dict[str, str] = {}
    _capture_broker_source_id(monkeypatch, captured)

    result = runner.invoke(app, _upload_command(session))

    assert result.exit_code == 0, result.output
    assert captured["source_id"] == "repo/benchflow-ai/benchflow"
    assert _repo_line(workdir) in click.unstyle(result.output)
    assert str(workdir) not in captured["source_id"]


def test_no_repo_keeps_the_default_source_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1015 (repo tagging follow-up to PR #1013): ``--no-repo``
    opts out even when a repo is detectable, keeping the path-derived id."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    session = _session_with_cwd(
        tmp_path, {"type": "user", "cwd": str(workdir), "message": {"content": "hi"}}
    )
    monkeypatch.setattr(
        "benchflow.cli.traj._command_stdout",
        _git_remote_stub("https://github.com/benchflow-ai/benchflow.git", workdir),
    )
    captured: dict[str, str] = {}
    _capture_broker_source_id(monkeypatch, captured)

    result = runner.invoke(app, _upload_command(session, "--no-repo"))

    assert result.exit_code == 0, result.output
    assert captured["source_id"] == "session"
    assert "Repo:" not in click.unstyle(result.output)


def test_explicit_source_id_wins_over_repo_tagging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1015 (repo tagging follow-up to PR #1013): an explicit
    ``--source-id`` always wins and repo detection never runs."""

    def fail_detection(path: Path) -> str | None:
        raise AssertionError("repo detection ran despite explicit --source-id")

    monkeypatch.setattr("benchflow.cli.traj._detect_repo_slug", fail_detection)
    captured: dict[str, str] = {}
    _capture_broker_source_id(monkeypatch, captured)

    result = runner.invoke(
        app, _upload_command(_trial(tmp_path), "--source-id", "my-project/run-42")
    )

    assert result.exit_code == 0, result.output
    assert captured["source_id"] == "my-project/run-42"
    assert "Repo:" not in click.unstyle(result.output)


def test_undetectable_repo_falls_back_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1015 (repo tagging follow-up to PR #1013): a session with
    no recorded cwd and no reachable git remote uploads with the default
    source id, no repo line, and no error."""
    session = _session_with_cwd(tmp_path, {"type": "message", "text": "demo"})
    monkeypatch.setattr("benchflow.cli.traj._command_stdout", lambda *_args: None)
    captured: dict[str, str] = {}
    _capture_broker_source_id(monkeypatch, captured)

    result = runner.invoke(app, _upload_command(session))

    assert result.exit_code == 0, result.output
    assert captured["source_id"] == "session"
    assert "Repo:" not in click.unstyle(result.output)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "message", "text": "demo"},
        None,  # replaced with a cwd pointing at a non-repo directory below
    ],
    ids=["no-recorded-cwd", "cwd-is-not-a-repo"],
)
def test_session_without_usable_cwd_never_tags_from_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event: dict | None
) -> None:
    """Guards this PR's fix to PR #1015 against the invocation-directory
    fallback the collector-side audit flagged: a session recorded outside any
    git repo (e.g. /tmp/collector-walkthrough), uploaded from inside the
    benchflow checkout, was mis-tagged ``repo/benchflow-ai/benchflow`` — two
    community-dataset entries carry that mis-attribution. The repo tag must
    derive ONLY from the trajectory's own recorded cwd, so even when the
    upload invocation directory resolves to a GitHub remote, a session with
    no usable cwd uploads untagged with the default source id."""
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    if event is None:
        event = {"type": "user", "cwd": str(not_a_repo), "message": {"content": "hi"}}
    session = _session_with_cwd(tmp_path, event)

    def invocation_dir_has_a_remote(*args: str) -> str | None:
        # Answer every origin-remote lookup EXCEPT the session's recorded cwd:
        # the invocation directory (and anything else) looks like a checkout
        # of benchflow, exactly the audit's mis-attribution scenario.
        if args[:2] == ("git", "-C") and args[2] == str(not_a_repo):
            return None
        if args[3:] == ("remote", "get-url", "origin"):
            return "https://github.com/benchflow-ai/benchflow.git"
        return None

    monkeypatch.setattr(
        "benchflow.cli.traj._command_stdout", invocation_dir_has_a_remote
    )
    captured: dict[str, str] = {}
    _capture_broker_source_id(monkeypatch, captured)

    result = runner.invoke(app, _upload_command(session))

    assert result.exit_code == 0, result.output
    assert captured["source_id"] == "session"
    assert "Repo:" not in click.unstyle(result.output)
