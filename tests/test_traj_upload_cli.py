"""CLI and broker-protocol tests for ``bench traj upload``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.publish.broker import upload_capture_via_broker
from benchflow.publish.traj_capture import stage_trajectory_capture

runner = CliRunner()


def _trial(tmp_path: Path) -> Path:
    trial = tmp_path / "trial-demo"
    trajectory = trial / "trajectory"
    trajectory.mkdir(parents=True)
    (trajectory / "acp_trajectory.jsonl").write_text(
        '{"type":"message","text":"demo"}\n', encoding="utf-8"
    )
    return trial


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
    result = runner.invoke(app, ["traj", "upload", str(trial), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "sha256:" in result.output
    assert "trajectory/acp_trajectory.jsonl" in result.output
    assert "manifest.json" in result.output


def test_direct_mode_reports_azure_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI delegates direct mode and renders the returned Azure URL."""
    trial = _trial(tmp_path)

    def fake_upload(staged, *, container_url):
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
        [
            "traj",
            "upload",
            str(trial),
            "--direct",
            "--container-url",
            "https://tasksminerdata.blob.core.windows.net/bronze",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Uploaded trajectory" in result.output
    assert "tasksminerdata.blob.core.windows.net/bronze" in result.output


def test_broker_mode_uses_exact_manifest_and_server_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Broker mode sends the manifest handshake and returned PUT headers verbatim."""
    trial = _trial(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            body = json.loads(request.content)
            assert set(body) == {
                "schema_version",
                "kind",
                "source_id",
                "traj_digest",
                "uploaded_by",
                "artifacts",
            }
            return httpx.Response(200, json=_broker_payload(request))
        assert request.headers["x-ms-blob-type"] == "BlockBlob"
        assert request.headers["if-none-match"] == "*"
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("benchflow.publish.broker.httpx.Client", lambda: client)
    monkeypatch.setenv("BENCHFLOW_TRAJ_BROKER_URL", "https://broker.test")
    result = runner.invoke(app, ["traj", "upload", str(trial)])

    assert result.exit_code == 0, result.output
    assert [request.method for request in requests] == ["POST", "PUT", "PUT"]
    assert requests[-1].url.path.endswith("manifest.json")


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
    result = runner.invoke(app, ["traj", "upload", str(trial)])
    assert result.exit_code == 0, result.output
    assert "Already uploaded" in result.output

    monkeypatch.setattr("benchflow.publish.broker.httpx.Client", lambda: limited)
    result = runner.invoke(app, ["traj", "upload", str(trial)])
    assert result.exit_code == 1
    assert "retry after 60" in result.output


def test_missing_broker_names_both_available_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A development build without a default endpoint explains both modes."""
    trial = _trial(tmp_path)
    monkeypatch.delenv("BENCHFLOW_TRAJ_BROKER_URL", raising=False)
    monkeypatch.setattr("benchflow.cli.traj.DEFAULT_TRAJ_BROKER_URL", None)
    result = runner.invoke(app, ["traj", "upload", str(trial)])

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
    result = runner.invoke(app, ["traj", "upload", str(trial)])

    assert result.exit_code == 1
    assert "acp_trajectory.jsonl" in result.output.replace("\n", "")
    assert "line 1" in result.output


@pytest.mark.parametrize("shape", ["unknown", "missing"])
def test_broker_mapping_violation_sends_zero_puts(tmp_path: Path, shape: str) -> None:
    """A non-bijective broker response fails before any trajectory bytes leave."""
    trial = _trial(tmp_path)
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        payload = _broker_payload(request)
        if shape == "unknown":
            payload["objects"][0]["name"] = "trajectory/unknown.jsonl"
        else:
            payload["objects"].pop()
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
        result = upload_capture_via_broker(
            staged,
            broker_url="https://broker.test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    assert not result.uploaded
    assert len(result.skipped) == len(staged.files)


def test_help_exposes_only_the_planned_upload_command() -> None:
    """The trajectory CLI surface remains a single clean command."""
    result = runner.invoke(app, ["traj", "--help"])
    assert result.exit_code == 0
    assert "upload" in result.output
