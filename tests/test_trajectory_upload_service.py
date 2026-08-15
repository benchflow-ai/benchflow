"""Offline contract and promotion tests for the Azure upload services."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from benchflow.publish.traj_capture import stage_trajectory_capture
from services.trajectory_upload.azure_backend import AzureUploadBroker
from services.trajectory_upload.broker_app import (
    AlreadyUploaded,
    RateLimited,
    RejectedUpload,
    create_app,
)
from services.trajectory_upload.contract import (
    UploadGrant,
    UploadObject,
    UploadRequest,
)
from services.trajectory_upload.validation import (
    CaptureRejected,
    _validate_and_scan_jsonl,
    validate_local_capture,
)
from services.trajectory_upload.validator import (
    AzureCaptureValidator,
    _capture_from_event,
)


def _trial(tmp_path: Path, text: str = "safe") -> Path:
    trial = tmp_path / "trial"
    trajectory = trial / "trajectory"
    trajectory.mkdir(parents=True)
    (trajectory / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "message", "text": text}) + "\n",
        encoding="utf-8",
    )
    return trial


def _request_from_manifest(manifest: dict) -> dict:
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "kind",
            "source_id",
            "traj_digest",
            "uploaded_by",
            "artifacts",
        )
    }


class FakeBroker:
    def __init__(self, result: UploadGrant | Exception) -> None:
        self.result = result
        self.client_ip: str | None = None

    def create_upload(self, request: UploadRequest, *, client_ip: str) -> UploadGrant:
        self.client_ip = client_ip
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_first_delegation_key_request_does_not_underflow() -> None:
    """Guards the live Azure fix against the underflow in commit 158ef108."""
    blob_service = SimpleNamespace(
        get_user_delegation_key=lambda **_kwargs: "delegation-key"
    )
    backend = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=SimpleNamespace(),
        blob_service=blob_service,
        ip_hash_key=b"test",
    )

    assert backend._user_delegation_key(datetime.now(UTC)) == "delegation-key"


def test_broker_sas_permission_is_service_enforced_create_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller cannot omit If-None-Match to turn a grant into an overwrite."""
    captured: dict = {}

    def fake_generate_blob_sas(**kwargs) -> str:
        captured.update(kwargs)
        return "sp=c&sig=test"

    monkeypatch.setattr("azure.storage.blob.generate_blob_sas", fake_generate_blob_sas)
    backend = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=SimpleNamespace(),
        blob_service=SimpleNamespace(),
        ip_hash_key=b"test",
    )
    now = datetime.now(UTC)

    grant = backend._upload_object(
        prefix="inbox/" + "a" * 64 + "/",
        relname="trajectory/capture.jsonl",
        content_type="application/jsonl",
        delegation_key="delegation-key",
        starts_at=now,
        expires_at=now + timedelta(minutes=5),
    )

    assert captured["permission"].create is True
    assert captured["permission"].write is False
    assert grant.headers["If-None-Match"] == "*"


def test_upload_contract_recomputes_digest_and_rejects_object_injection(
    tmp_path: Path,
) -> None:
    """The broker accepts only content-addressed trajectory JSONL object names."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        body = _request_from_manifest(staged.manifest)
        request = UploadRequest.model_validate(body)
        assert request.traj_digest == staged.manifest["traj_digest"]

        body["artifacts"][0]["name"] = "../sources/private.jsonl"
        with pytest.raises(ValidationError, match="outside trajectory"):
            UploadRequest.model_validate(body)

    with stage_trajectory_capture(trial, source_id="demo") as staged:
        body = _request_from_manifest(staged.manifest)
        body["traj_digest"] = "sha256:" + "0" * 64
        with pytest.raises(ValidationError, match="does not match"):
            UploadRequest.model_validate(body)


def test_broker_http_surface_returns_scoped_grants_and_protocol_statuses(
    tmp_path: Path,
) -> None:
    """The public endpoint emits v1 grants, conflict, and Retry-After responses."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        body = _request_from_manifest(staged.manifest)
        grant = UploadGrant(
            upload_id="u_demo",
            bucket="bronze",
            base_url="https://account.blob.core.windows.net/bronze",
            prefix=f"inbox/{staged.traj_digest}/",
            objects=tuple(
                UploadObject(
                    name=item.relname,
                    put_url=f"https://upload.test/{item.relname}?sig=test",
                    headers={"If-None-Match": "*"},
                )
                for item in staged.files
            ),
            expires_at=datetime.fromisoformat(
                staged.manifest["created_at"].replace("Z", "+00:00")
            ),
        )

    backend = FakeBroker(grant)
    response = TestClient(create_app(backend)).post(
        "/v1/uploads",
        json=body,
        headers={"x-forwarded-for": "spoofed, 203.0.113.9"},
    )
    assert response.status_code == 200
    assert response.json()["objects"][-1]["name"] == "manifest.json"
    assert backend.client_ip == "203.0.113.9"

    conflict = TestClient(
        create_app(
            FakeBroker(
                AlreadyUploaded(
                    base_url="https://account.blob.core.windows.net/bronze",
                    prefix="sources/community/demo/",
                )
            )
        )
    ).post("/v1/uploads", json=body)
    assert conflict.status_code == 409
    assert conflict.json()["prefix"].startswith("sources/community/")

    limited = TestClient(create_app(FakeBroker(RateLimited(42)))).post(
        "/v1/uploads", json=body
    )
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "42"

    rejected = TestClient(create_app(FakeBroker(RejectedUpload()))).post(
        "/v1/uploads", json=body
    )
    assert rejected.status_code == 422
    assert "previously rejected" in rejected.json()["detail"]


def test_broker_validation_errors_are_fail_closed_and_json_safe(
    tmp_path: Path,
) -> None:
    """Guards the live Azure fix after malformed handshakes returned HTTP 500."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        body = _request_from_manifest(staged.manifest)

    injected = json.loads(json.dumps(body))
    injected["artifacts"][0]["name"] = "../private.jsonl"
    injection_response = TestClient(create_app(FakeBroker(AssertionError()))).post(
        "/v1/uploads", json=injected
    )
    assert injection_response.status_code == 400
    assert injection_response.json()["detail"][0]["type"] == "value_error"

    oversized = json.loads(json.dumps(body))
    oversized["artifacts"][0]["bytes"] = 1024**3 + 1
    oversized_response = TestClient(create_app(FakeBroker(AssertionError()))).post(
        "/v1/uploads", json=oversized
    )
    assert oversized_response.status_code == 413
    assert oversized_response.json()["detail"][0]["type"] == "less_than_equal"

    body_limit_response = TestClient(create_app(FakeBroker(AssertionError()))).post(
        "/v1/uploads",
        content=b"{}",
        headers={"Content-Length": str(1024**2 + 1)},
    )
    assert body_limit_response.status_code == 413


def test_validator_recomputes_bytes_jsonl_and_secret_scan(tmp_path: Path) -> None:
    """Promotion validation rejects digest corruption, malformed JSONL, and secrets."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        manifest_bytes = staged.files[-1].local_path.read_bytes()
        paths = {item.relname: item.local_path for item in staged.files[:-1]}
        validated = validate_local_capture(manifest_bytes, paths)
        assert validated.manifest.source_id == "demo"

        staged.files[0].local_path.write_text('{"type":"changed"}\n', encoding="utf-8")
        with pytest.raises(CaptureRejected, match=r"size mismatch|sha256 mismatch"):
            validate_local_capture(manifest_bytes, paths)

    secret_trial = _trial(tmp_path / "secret", "sk-1234567890abcdefghijklmnop")
    with stage_trajectory_capture(
        secret_trial, source_id="demo", redact=False
    ) as staged:
        manifest = dict(staged.manifest)
        manifest["redaction"] = {"applied": True, "replacements": 0}
        manifest_bytes = json.dumps(manifest).encode()
        paths = {item.relname: item.local_path for item in staged.files[:-1]}
        with pytest.raises(CaptureRejected, match="secret-like"):
            validate_local_capture(manifest_bytes, paths)

    invalid_utf8 = tmp_path / "llm_trajectory.jsonl"
    invalid_utf8.write_bytes(b"\xff\n")
    with pytest.raises(CaptureRejected, match="must be UTF-8"):
        _validate_and_scan_jsonl(invalid_utf8, "trajectory/llm_trajectory.jsonl")


class FakeDownloader:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def readall(self) -> bytes:
        return self.content

    def readinto(self, stream) -> int:
        return stream.write(self.content)


class FakeBlobClient:
    def __init__(self, container: FakeContainer, name: str) -> None:
        self.container = container
        self.name = name

    def get_blob_properties(self):
        if self.name not in self.container.blobs:
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError("missing test blob")
        content = self.container.blobs[self.name]
        return SimpleNamespace(size=len(content))

    def download_blob(self, **_kwargs) -> FakeDownloader:
        return FakeDownloader(self.container.blobs[self.name])


class FakeContainer:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = dict(blobs)
        self.uploaded: list[str] = []
        self.requested: list[str] = []

    def get_blob_client(self, name: str) -> FakeBlobClient:
        self.requested.append(name)
        return FakeBlobClient(self, name)

    def upload_blob(self, *, name: str, data, **_kwargs) -> None:
        content = data if isinstance(data, bytes) else data.read()
        self.blobs[name] = content
        self.uploaded.append(name)

    def list_blobs(self, *, name_starts_with: str):
        return [
            SimpleNamespace(name=name)
            for name in tuple(self.blobs)
            if name.startswith(name_starts_with)
        ]

    def delete_blob(self, name: str) -> None:
        self.blobs.pop(name, None)


class FakeQueue:
    def __init__(self, content: str) -> None:
        self.message = SimpleNamespace(id="m1", pop_receipt="p1", content=content)
        self.deleted: list[tuple[str, str]] = []

    def receive_messages(self, **_kwargs):
        return [self.message]

    def delete_message(self, message_id: str, pop_receipt: str) -> None:
        self.deleted.append((message_id, pop_receipt))


class FakeEntity(dict):
    def __init__(self, entity: dict, etag: str) -> None:
        super().__init__(entity)
        self.metadata = {"etag": etag}


class FakeTable:
    def __init__(self, entities: list[dict] | None = None) -> None:
        self.entities = entities or []
        self.version = len(self.entities)

    def upsert_entity(self, entity: dict) -> None:
        self.entities.append(entity)
        self.version += 1

    def update_entity(self, *, entity: dict, etag: str, **_kwargs) -> None:
        assert etag == str(self.version)
        self.entities.append(entity)
        self.version += 1

    def get_entity(self, *, partition_key: str, row_key: str) -> dict:
        from azure.core.exceptions import ResourceNotFoundError

        for entity in reversed(self.entities):
            if entity["PartitionKey"] == partition_key and entity["RowKey"] == row_key:
                return FakeEntity(entity, str(self.version))
        raise ResourceNotFoundError("not found")


def _pending_table(digest: str) -> FakeTable:
    return FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "pending",
            }
        ]
    )


@pytest.mark.parametrize("status", ["pending", "validating"])
def test_broker_regrant_does_not_downgrade_ledger_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """Guards PR #989 against a retry clearing an active validation lease."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        request = UploadRequest.model_validate(_request_from_manifest(staged.manifest))
        digest = staged.traj_digest
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": status,
                "validation_lease_until": "future" if status == "validating" else "",
            }
        ]
    )
    backend = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=table,
        blob_service=SimpleNamespace(
            get_user_delegation_key=lambda **_kwargs: "delegation-key"
        ),
        ip_hash_key=b"test",
    )
    monkeypatch.setattr(backend, "_consume_rate_limit", lambda _client_ip: None)
    monkeypatch.setattr(
        "azure.storage.blob.generate_blob_sas", lambda **_kwargs: "sp=c&sig=test"
    )

    grant = backend.create_upload(request, client_ip="127.0.0.1")

    assert grant.prefix == f"inbox/{digest}/"
    assert table.entities[-1]["status"] == status
    assert len(table.entities) == 1


def test_broker_does_not_reopen_rejected_digest(tmp_path: Path) -> None:
    """Guards PR #989 against downgrading a terminal rejected capture."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        request = UploadRequest.model_validate(_request_from_manifest(staged.manifest))
        digest = staged.traj_digest
    backend = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=FakeTable(
            [
                {
                    "PartitionKey": "capture",
                    "RowKey": digest,
                    "status": "rejected",
                }
            ]
        ),
        blob_service=SimpleNamespace(),
        ip_hash_key=b"test",
    )

    with pytest.raises(RejectedUpload):
        backend.create_upload(request, client_ip="127.0.0.1")


def _quarantine_capture(tmp_path: Path) -> tuple[str, dict[str, bytes]]:
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        digest = staged.traj_digest
        prefix = f"inbox/{digest}/"
        blobs = {
            prefix + item.relname: item.local_path.read_bytes() for item in staged.files
        }
    return digest, blobs


def test_queue_validator_promotes_manifest_last_and_cleans_quarantine(
    tmp_path: Path,
) -> None:
    """A valid Event Grid capture reaches community sources with manifest last."""
    digest, blobs = _quarantine_capture(tmp_path)
    container = FakeContainer(blobs)
    event = json.dumps(
        {
            "data": {
                "url": (
                    "https://account.blob.core.windows.net/bronze/"
                    f"inbox/{digest}/manifest.json"
                )
            }
        }
    )
    queue = FakeQueue(event)
    table = _pending_table(digest)
    validator = AzureCaptureValidator(container=container, queue=queue, table=table)

    assert validator.run_once() is True
    assert container.uploaded[-1] == f"sources/community/{digest}/manifest.json"
    assert not any(name.startswith(f"inbox/{digest}/") for name in container.blobs)
    assert table.entities[-1]["status"] == "ingested"
    assert queue.deleted == [("m1", "p1")]


def test_event_grid_queue_base64_envelope_is_decoded(tmp_path: Path) -> None:
    """Guards the live Azure queue fix after commit 0717c061 discarded events."""
    digest, _ = _quarantine_capture(tmp_path)
    event = json.dumps(
        {
            "data": {
                "url": (
                    "https://account.blob.core.windows.net/bronze/"
                    f"inbox/{digest}/manifest.json"
                )
            }
        }
    )
    encoded = base64.b64encode(event.encode()).decode()

    assert _capture_from_event(encoded) == (f"inbox/{digest}/", digest, True)


def test_pending_artifact_event_waits_for_manifest_commit(tmp_path: Path) -> None:
    """Guards PR #989 against treating artifact creation as a partial upload."""
    digest, blobs = _quarantine_capture(tmp_path)
    artifact = next(name for name in blobs if name.endswith(".jsonl"))
    event = json.dumps(
        {"data": {"url": ("https://account.blob.core.windows.net/bronze/" + artifact)}}
    )
    queue = FakeQueue(event)
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "pending",
            }
        ]
    )
    container = FakeContainer(blobs)

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()
    assert container.blobs == blobs
    assert container.uploaded == []
    assert table.entities[-1]["status"] == "pending"
    assert queue.deleted == [("m1", "p1")]


def test_terminal_artifact_replay_is_cleaned(tmp_path: Path) -> None:
    """Guards PR #989 against replaying a grant after terminal cleanup."""
    digest, blobs = _quarantine_capture(tmp_path)
    artifact = next(name for name in blobs if name.endswith(".jsonl"))
    event = json.dumps(
        {"data": {"url": ("https://account.blob.core.windows.net/bronze/" + artifact)}}
    )
    queue = FakeQueue(event)
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "ingested",
            }
        ]
    )
    container = FakeContainer(blobs)

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()
    assert not any(name.startswith(f"inbox/{digest}/") for name in container.blobs)
    assert queue.deleted == [("m1", "p1")]


@pytest.mark.parametrize("terminal_status", ["ingested", "rejected"])
def test_duplicate_terminal_event_is_an_idempotent_no_op(
    tmp_path: Path, terminal_status: str
) -> None:
    """At-least-once Event Grid delivery cannot reopen a terminal capture."""
    digest, blobs = _quarantine_capture(tmp_path)
    event = json.dumps(
        {
            "data": {
                "url": (
                    "https://account.blob.core.windows.net/bronze/"
                    f"inbox/{digest}/manifest.json"
                )
            }
        }
    )
    queue = FakeQueue(event)
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": terminal_status,
            }
        ]
    )

    container = FakeContainer(blobs)
    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()
    assert not any(name.startswith(f"inbox/{digest}/") for name in container.blobs)
    assert queue.deleted == [("m1", "p1")]


def test_queue_validator_rejects_corruption_without_promotion(tmp_path: Path) -> None:
    """Corrupt quarantine bytes are deleted and never enter the trusted namespace."""
    digest, blobs = _quarantine_capture(tmp_path)
    artifact = next(name for name in blobs if name.endswith(".jsonl"))
    blobs[artifact] = b'{"type":"corrupted"}\n'
    container = FakeContainer(blobs)
    queue = FakeQueue(
        json.dumps(
            {
                "data": {
                    "url": (
                        "https://account.blob.core.windows.net/bronze/"
                        f"inbox/{digest}/manifest.json"
                    )
                }
            }
        )
    )
    table = _pending_table(digest)

    AzureCaptureValidator(container=container, queue=queue, table=table).run_once()

    assert not any(name.startswith("sources/community/") for name in container.blobs)
    assert table.entities[-1]["status"] == "rejected"
    assert queue.deleted == [("m1", "p1")]


def test_queue_validator_rejects_missing_declared_artifact(tmp_path: Path) -> None:
    """An incomplete anonymous commit is terminal and cannot retry for seven days."""
    digest, blobs = _quarantine_capture(tmp_path)
    artifact = next(name for name in blobs if name.endswith(".jsonl"))
    blobs.pop(artifact)
    container = FakeContainer(blobs)
    queue = FakeQueue(
        json.dumps(
            {
                "data": {
                    "url": (
                        "https://account.blob.core.windows.net/bronze/"
                        f"inbox/{digest}/manifest.json"
                    )
                }
            }
        )
    )
    table = _pending_table(digest)

    AzureCaptureValidator(container=container, queue=queue, table=table).run_once()

    assert not any(name.startswith("sources/community/") for name in container.blobs)
    assert not any(name.startswith(f"inbox/{digest}/") for name in container.blobs)
    assert table.entities[-1]["status"] == "rejected"
    assert "missing" in table.entities[-1]["detail"]
    assert queue.deleted == [("m1", "p1")]


def test_queue_validator_rejects_invalid_utf8_manifest(tmp_path: Path) -> None:
    """Guards PR #989 against retrying an undecodable anonymous manifest."""
    digest, blobs = _quarantine_capture(tmp_path)
    manifest_name = f"inbox/{digest}/manifest.json"
    blobs[manifest_name] = b"\xff"
    container = FakeContainer(blobs)
    queue = FakeQueue(
        json.dumps(
            {
                "data": {
                    "url": (
                        "https://account.blob.core.windows.net/bronze/" + manifest_name
                    )
                }
            }
        )
    )
    table = _pending_table(digest)

    AzureCaptureValidator(container=container, queue=queue, table=table).run_once()

    assert table.entities[-1]["status"] == "rejected"
    assert "invalid manifest" in table.entities[-1]["detail"]
    assert queue.deleted == [("m1", "p1")]


def test_queue_validator_rejects_excessively_nested_manifest(tmp_path: Path) -> None:
    """Guards PR #989 against retrying JSON parser recursion failures."""
    digest, blobs = _quarantine_capture(tmp_path)
    manifest_name = f"inbox/{digest}/manifest.json"
    blobs[manifest_name] = b"[" * 2_000 + b"]" * 2_000
    container = FakeContainer(blobs)
    queue = FakeQueue(
        json.dumps(
            {
                "data": {
                    "url": (
                        "https://account.blob.core.windows.net/bronze/" + manifest_name
                    )
                }
            }
        )
    )
    table = _pending_table(digest)

    AzureCaptureValidator(container=container, queue=queue, table=table).run_once()

    assert table.entities[-1]["status"] == "rejected"
    assert "invalid manifest" in table.entities[-1]["detail"]
    assert queue.deleted == [("m1", "p1")]


def test_deploy_selects_event_topic_by_storage_source() -> None:
    """Guards PR #989 against attaching events to an unrelated system topic."""
    script = Path("infra/trajectory-upload/deploy-azure.sh").read_text()

    assert (
        "map(select((.source | ascii_downcase) == ($source | ascii_downcase)))"
        in script
    )
    assert "--query '[0].name'" not in script


def test_manifest_contract_is_validated_before_artifact_downloads(
    tmp_path: Path,
) -> None:
    """Guards PR #989 against manifest-driven download amplification."""
    digest, blobs = _quarantine_capture(tmp_path)
    manifest_name = f"inbox/{digest}/manifest.json"
    manifest = json.loads(blobs[manifest_name])
    manifest["artifacts"] = manifest["artifacts"] * 9
    blobs[manifest_name] = json.dumps(manifest).encode()
    container = FakeContainer(blobs)
    queue = FakeQueue(
        json.dumps(
            {
                "data": {
                    "url": (
                        "https://account.blob.core.windows.net/bronze/" + manifest_name
                    )
                }
            }
        )
    )
    table = _pending_table(digest)

    AzureCaptureValidator(container=container, queue=queue, table=table).run_once()

    assert container.requested == [manifest_name]
    assert table.entities[-1]["status"] == "rejected"
    assert queue.deleted == [("m1", "p1")]


def test_concurrent_manifest_event_waits_for_validation_lease(
    tmp_path: Path,
) -> None:
    """Guards PR #989 against duplicate workers downgrading an ingested capture."""
    digest, blobs = _quarantine_capture(tmp_path)
    event = json.dumps(
        {
            "data": {
                "url": (
                    "https://account.blob.core.windows.net/bronze/"
                    f"inbox/{digest}/manifest.json"
                )
            }
        }
    )
    queue = FakeQueue(event)
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "validating",
                "validation_lease_until": (
                    datetime.now(UTC) + timedelta(minutes=10)
                ).isoformat(),
            }
        ]
    )
    container = FakeContainer(blobs)

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()
    assert container.blobs == blobs
    assert container.uploaded == []
    assert queue.deleted == []


def test_expired_validation_lease_is_reclaimed(tmp_path: Path) -> None:
    """Guards PR #989 against stranding a capture after a validator crash."""
    digest, blobs = _quarantine_capture(tmp_path)
    event = json.dumps(
        {
            "data": {
                "url": (
                    "https://account.blob.core.windows.net/bronze/"
                    f"inbox/{digest}/manifest.json"
                )
            }
        }
    )
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "validating",
                "validation_lease_until": (
                    datetime.now(UTC) - timedelta(minutes=1)
                ).isoformat(),
            }
        ]
    )
    container = FakeContainer(blobs)
    queue = FakeQueue(event)

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()
    assert table.entities[-1]["status"] == "ingested"
    assert container.uploaded[-1] == f"sources/community/{digest}/manifest.json"
    assert queue.deleted == [("m1", "p1")]
