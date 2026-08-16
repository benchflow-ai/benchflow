"""Offline contract and promotion tests for the Azure upload services."""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from benchflow.publish.traj_capture import (
    finalize_trajectory_capture,
    stage_trajectory_artifacts,
    stage_trajectory_capture,
)
from benchflow.publish.traj_report import build_trajectory_report
from services.trajectory_upload.azure_backend import AzureUploadBroker
from services.trajectory_upload.broker_app import (
    AlreadyUploaded,
    RateLimited,
    RejectedUpload,
    UploadDeclarationConflict,
    _backend,
    create_app,
)
from services.trajectory_upload.contract import (
    CaptureStatusInfo,
    UploadGrant,
    UploadObject,
    UploadRequest,
)
from services.trajectory_upload.validation import (
    CaptureRejected,
    _validate_and_scan_jsonl,
    validate_local_capture,
    validate_manifest_bytes,
)
from services.trajectory_upload.validator import (
    AzureCaptureValidator,
    _capture_from_event,
    drain,
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
    request = {
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
    if "contributor" in manifest:
        request["contributor"] = manifest["contributor"]
    return request


class FakeBroker:
    def __init__(self, result: UploadGrant | Exception) -> None:
        self.result = result
        self.client_ip: str | None = None

    def create_upload(self, request: UploadRequest, *, client_ip: str) -> UploadGrant:
        self.client_ip = client_ip
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def get_capture_status(self, digest: str, *, client_ip: str) -> CaptureStatusInfo:
        raise AssertionError("handshake tests must not reach the status route")


def test_broker_cold_start_constructs_one_shared_backend() -> None:
    """Guards PR #989 against cold-start requests splitting quota locks."""
    service = FakeBroker(AssertionError())
    calls = 0

    def factory() -> FakeBroker:
        nonlocal calls
        calls += 1
        time.sleep(0.02)
        return service

    app = create_app(backend_factory=factory)
    barrier = Barrier(8)

    def load_backend() -> object:
        barrier.wait()
        return _backend(app)

    with ThreadPoolExecutor(max_workers=8) as executor:
        backends = list(executor.map(lambda _item: load_backend(), range(8)))

    assert calls == 1
    assert all(backend is service for backend in backends)


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
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        body = _request_from_manifest(staged.manifest)
        request = UploadRequest.model_validate(body)
        assert request.traj_digest == staged.manifest["traj_digest"]

        body["artifacts"][0]["name"] = "../sources/private.jsonl"
        with pytest.raises(ValidationError, match="outside trajectory"):
            UploadRequest.model_validate(body)

    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        body = _request_from_manifest(staged.manifest)
        body["traj_digest"] = "sha256:" + "0" * 64
        with pytest.raises(ValidationError, match="does not match"):
            UploadRequest.model_validate(body)


def test_contributor_contract_is_validated_at_broker_and_manifest_boundaries(
    tmp_path: Path,
) -> None:
    """Self-asserted GitHub and email provenance remains typed server-side."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        request = UploadRequest.model_validate(_request_from_manifest(staged.manifest))
        manifest = validate_manifest_bytes(staged.files[-1].local_path.read_bytes())

    assert request.contributor is not None
    assert request.schema_version == "1.1.0"
    assert request.contributor.github_id == "benchflow-ai"
    assert request.contributor.email == "contributor@benchflow.ai"
    assert manifest.contributor == request.contributor

    invalid = _request_from_manifest(staged.manifest)
    invalid["contributor"] = {
        "github_id": "@invalid",
        "email": "not-an-email",
    }
    with pytest.raises(ValidationError):
        UploadRequest.model_validate(invalid)

    missing = _request_from_manifest(staged.manifest)
    missing.pop("contributor")
    with pytest.raises(ValidationError, match="Field required"):
        UploadRequest.model_validate(missing)

    legacy_request = dict(missing)
    legacy_request["schema_version"] = "1.0.0"
    with pytest.raises(ValidationError, match="literal_error"):
        UploadRequest.model_validate(legacy_request)

    with stage_trajectory_capture(trial, source_id="legacy") as legacy:
        legacy_manifest = validate_manifest_bytes(
            legacy.files[-1].local_path.read_bytes()
        )
    assert legacy_manifest.schema_version == "1.0.0"
    assert legacy_manifest.contributor is None


def test_schema_12_handshake_requires_exact_manifest_digest(tmp_path: Path) -> None:
    """Guards PR #1008 against accepting an unbound schema 1.2 manifest."""
    with stage_trajectory_capture(
        _trial(tmp_path),
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        body = _request_from_manifest(staged.manifest)
    body["schema_version"] = "1.2.0"

    with pytest.raises(ValidationError, match="requires manifest sha256"):
        UploadRequest.model_validate(body)

    body["manifest_sha256"] = "a" * 64
    assert UploadRequest.model_validate(body).manifest_sha256 == "a" * 64

    body["manifest_sha256"] = "A" * 64
    with pytest.raises(ValidationError, match="64 lowercase hex"):
        UploadRequest.model_validate(body)


def test_broker_rejects_new_legacy_uploads_without_contributor(
    tmp_path: Path,
) -> None:
    """Guards PR #992 against granting contributor-free schema 1.0 uploads."""
    with stage_trajectory_capture(_trial(tmp_path), source_id="legacy") as staged:
        body = _request_from_manifest(staged.manifest)

    response = TestClient(create_app(FakeBroker(AssertionError()))).post(
        "/v1/uploads", json=body
    )

    assert response.status_code == 400
    assert response.json()["detail"][0]["type"] == "literal_error"


def test_broker_http_surface_returns_scoped_grants_and_protocol_statuses(
    tmp_path: Path,
) -> None:
    """The public endpoint emits v1 grants, conflict, and Retry-After responses."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
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

    conflicting = TestClient(create_app(FakeBroker(UploadDeclarationConflict()))).post(
        "/v1/uploads", json=body
    )
    assert conflicting.status_code == 422
    assert "conflicting active manifest" in conflicting.json()["detail"]


def test_broker_validation_errors_are_fail_closed_and_json_safe(
    tmp_path: Path,
) -> None:
    """Guards the live Azure fix after malformed handshakes returned HTTP 500."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        body = _request_from_manifest(staged.manifest)

    injected = json.loads(json.dumps(body))
    injected["artifacts"][0]["name"] = "../private.jsonl"
    injection_response = TestClient(create_app(FakeBroker(AssertionError()))).post(
        "/v1/uploads", json=injected
    )
    assert injection_response.status_code == 400
    assert injection_response.json()["detail"][0]["type"] == "value_error"

    secret_named = json.loads(json.dumps(body))
    secret_named["artifacts"][0]["name"] = "trajectory/sk-abcdefghijklmnop.jsonl"
    secret_name_response = TestClient(create_app(FakeBroker(AssertionError()))).post(
        "/v1/uploads", json=secret_named
    )
    assert secret_name_response.status_code == 400
    assert secret_name_response.json()["detail"][0]["type"] == "value_error"

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

    aws_session_key = tmp_path / "aws-session.jsonl"
    aws_session_key.write_text(
        json.dumps({"output": "ASIAQWERTYUIOPASDFGH"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CaptureRejected, match="secret-like"):
        _validate_and_scan_jsonl(aws_session_key, "trajectory/aws-session.jsonl")

    structured_header = tmp_path / "structured-header.jsonl"
    structured_header.write_text(
        json.dumps(
            {"headers": [{"name": "Authorization", "value": "opaque-prefixless-value"}]}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CaptureRejected, match="secret-like"):
        _validate_and_scan_jsonl(
            structured_header, "trajectory/structured-header.jsonl"
        )


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_validator_rejects_non_utf8_manifest_bytes(
    tmp_path: Path, encoding: str
) -> None:
    """Guards PR #989 against JSON byte-parser encoding auto-detection."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        encoded = json.dumps(staged.manifest).encode(encoding)

    with pytest.raises(CaptureRejected, match="must be UTF-8"):
        validate_manifest_bytes(encoded)


@pytest.mark.parametrize(
    "record, message",
    [
        ('{"password":"live","password":"[REDACTED]"}\n', "duplicate JSON"),
        ('{"reward":NaN}\n', "non-finite JSON"),
        ('{"reward":Infinity}\n', "non-finite JSON"),
        ('{"reward":-Infinity}\n', "non-finite JSON"),
    ],
)
def test_validator_rejects_ambiguous_or_nonstandard_json(
    tmp_path: Path, record: str, message: str
) -> None:
    """Guards PR #989 against duplicate keys and non-finite JSON numbers."""
    artifact = tmp_path / "strict.jsonl"
    artifact.write_text(record, encoding="utf-8")

    with pytest.raises(CaptureRejected, match=message):
        _validate_and_scan_jsonl(artifact, "trajectory/strict.jsonl")


def test_validator_scans_manifest_strings_before_artifacts(tmp_path: Path) -> None:
    """Guards PR #989 against secrets in contributor and run metadata."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(trial, source_id="demo") as staged:
        manifest = dict(staged.manifest)
        manifest["uploaded_by"] = "dtn_1234567890abcdefghijklmnop"
        manifest_bytes = json.dumps(manifest).encode()
        paths = {item.relname: item.local_path for item in staged.files[:-1]}

        with pytest.raises(CaptureRejected, match="manifest contains a secret-like"):
            validate_local_capture(manifest_bytes, paths)


def test_validator_rejects_credential_shaped_mapping_keys(tmp_path: Path) -> None:
    """Guards PR #989 against tokens used as JSON object keys."""
    artifact = tmp_path / "credential-key.jsonl"
    artifact.write_text(
        '{"dtn_1234567890abcdefghijklmnop":"result"}\n',
        encoding="utf-8",
    )

    with pytest.raises(CaptureRejected, match="secret-like"):
        _validate_and_scan_jsonl(artifact, "trajectory/credential-key.jsonl")


@pytest.mark.parametrize(
    "field_name",
    ["token", "passwd", "accessToken", "clientSecret", "api key", "access.token"],
)
def test_validator_rejects_prefixless_credentials_in_common_fields(
    tmp_path: Path, field_name: str
) -> None:
    """Guards PR #989 against promoting raw token and camelCase fields."""
    artifact = tmp_path / "credential-field.jsonl"
    artifact.write_text(
        json.dumps({field_name: "opaque-prefixless-value"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CaptureRejected, match="secret-like"):
        _validate_and_scan_jsonl(artifact, "trajectory/credential-field.jsonl")


def test_validator_rejects_prefixless_secret_in_argv_sequence(tmp_path: Path) -> None:
    """Guards PR #989 against raw credentials following sensitive CLI flags."""
    artifact = tmp_path / "command.jsonl"
    artifact.write_text(
        json.dumps({"command": ["tool", "--api-key", "opaque-prefixless-value"]})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CaptureRejected, match="secret-like"):
        _validate_and_scan_jsonl(artifact, "trajectory/command.jsonl")


def test_validator_bounds_record_bytes_and_complexity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #989 against JSONL record memory and recursion bombs."""
    import services.trajectory_upload.validation as validation_module

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_text(json.dumps({"text": "x" * 64}) + "\n", encoding="utf-8")
    monkeypatch.setattr(validation_module, "MAX_JSONL_RECORD_BYTES", 32)
    with pytest.raises(CaptureRejected, match="JSONL record exceeds"):
        _validate_and_scan_jsonl(oversized, "trajectory/oversized.jsonl")

    nested: object = "leaf"
    for _ in range(101):
        nested = {"child": nested}
    recursive = tmp_path / "recursive.jsonl"
    recursive.write_text(json.dumps(nested) + "\n", encoding="utf-8")
    monkeypatch.setattr(validation_module, "MAX_JSONL_RECORD_BYTES", 8 * 1024**2)
    with pytest.raises(CaptureRejected, match="JSON nesting exceeds"):
        _validate_and_scan_jsonl(recursive, "trajectory/recursive.jsonl")


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

    def create_entity(self, entity: dict) -> None:
        from azure.core.exceptions import ResourceExistsError

        for existing in self.entities:
            if (
                existing["PartitionKey"] == entity["PartitionKey"]
                and existing["RowKey"] == entity["RowKey"]
            ):
                raise ResourceExistsError("entity already exists")
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
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
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
    monkeypatch.setattr(backend, "_consume_rate_limit", lambda *_args: None)
    monkeypatch.setattr(
        "azure.storage.blob.generate_blob_sas", lambda **_kwargs: "sp=c&sig=test"
    )

    grant = backend.create_upload(request, client_ip="127.0.0.1")

    assert grant.prefix == f"inbox/{digest}/"
    assert table.entities[-1]["status"] == status
    assert len(table.entities) == 1


def test_broker_persists_declared_artifact_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #989 against artifact-only uploads exceeding their declaration."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        request = UploadRequest.model_validate(_request_from_manifest(staged.manifest))
        expected = {item.name: item.bytes for item in request.artifacts}
    table = FakeTable()
    backend = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=table,
        blob_service=SimpleNamespace(
            get_user_delegation_key=lambda **_kwargs: "delegation-key"
        ),
        ip_hash_key=b"test",
    )
    monkeypatch.setattr(backend, "_consume_rate_limit", lambda *_args: None)
    monkeypatch.setattr(
        "azure.storage.blob.generate_blob_sas", lambda **_kwargs: "sp=c&sig=test"
    )

    grant = backend.create_upload(request, client_ip="127.0.0.1")

    assert json.loads(table.entities[-1]["declared_artifacts"]) == expected
    assert table.entities[-1]["attempt_id"] == grant.upload_id
    assert grant.prefix.endswith(f"/{grant.upload_id}/")


def test_broker_rejects_manifest_change_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1008 against mutating a manifest during validation."""
    with stage_trajectory_capture(
        _trial(tmp_path),
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        body = _request_from_manifest(staged.manifest)
        body["schema_version"] = "1.2.0"
        body["manifest_sha256"] = "a" * 64
        request = UploadRequest.model_validate(body)
        digest = staged.traj_digest
    attempt_id = "u_" + "1" * 32
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "validating",
                "attempt_id": attempt_id,
                "declared_manifest_sha256": "b" * 64,
            }
        ]
    )
    backend = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=table,
        blob_service=SimpleNamespace(),
        ip_hash_key=b"test",
    )
    monkeypatch.setattr(backend, "_consume_rate_limit", lambda *_args: None)

    with pytest.raises(UploadDeclarationConflict):
        backend.create_upload(request, client_ip="127.0.0.1")

    assert len(table.entities) == 1


def test_broker_supersedes_conflicting_incomplete_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1008 against interrupted uploads permanently blocking retries."""
    with stage_trajectory_capture(
        _trial(tmp_path),
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        body = _request_from_manifest(staged.manifest)
        body["schema_version"] = "1.2.0"
        body["manifest_sha256"] = "a" * 64
        request = UploadRequest.model_validate(body)
        digest = staged.traj_digest
    old_attempt = "u_" + "1" * 32
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "pending",
                "attempt_id": old_attempt,
                "declared_manifest_sha256": "b" * 64,
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
    monkeypatch.setattr(backend, "_consume_rate_limit", lambda *_args: None)
    monkeypatch.setattr(
        "azure.storage.blob.generate_blob_sas", lambda **_kwargs: "sp=c&sig=test"
    )

    grant = backend.create_upload(request, client_ip="127.0.0.1")

    assert grant.upload_id != old_attempt
    assert table.entities[-1]["attempt_id"] == grant.upload_id
    assert table.entities[-1]["declared_manifest_sha256"] == "a" * 64


def test_broker_reopens_rejected_digest_in_isolated_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #989 against a rejected attempt poisoning a valid digest."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        request = UploadRequest.model_validate(_request_from_manifest(staged.manifest))
        digest = staged.traj_digest
    old_attempt = "u_" + "0" * 32
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "rejected",
                "attempt_id": old_attempt,
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
    monkeypatch.setattr(backend, "_consume_rate_limit", lambda *_args: None)
    monkeypatch.setattr(
        "azure.storage.blob.generate_blob_sas", lambda **_kwargs: "sp=c&sig=test"
    )

    grant = backend.create_upload(request, client_ip="127.0.0.1")

    assert grant.upload_id != old_attempt
    assert grant.prefix == f"inbox/{digest}/{grant.upload_id}/"
    assert table.entities[-1]["status"] == "pending"
    assert table.entities[-1]["attempt_id"] == grant.upload_id


def test_terminal_digest_handshakes_consume_rate_limit(
    tmp_path: Path,
) -> None:
    """Guards PR #989 against terminal ledger lookups bypassing broker quotas."""
    trial = _trial(tmp_path)
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        request = UploadRequest.model_validate(_request_from_manifest(staged.manifest))
        digest = staged.traj_digest
    table = FakeTable(
        [{"PartitionKey": "capture", "RowKey": digest, "status": "ingested"}]
    )
    backend = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=table,
        blob_service=SimpleNamespace(),
        ip_hash_key=b"test",
        rate_limit=1,
    )

    with pytest.raises(AlreadyUploaded):
        backend.create_upload(request, client_ip="127.0.0.1")
    with pytest.raises(RateLimited):
        backend.create_upload(request, client_ip="127.0.0.1")


def _rate_backend(
    table: FakeTable, *, rate_limit: int = 20, ip_rate_limit: int = 500
) -> AzureUploadBroker:
    return AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=table,
        blob_service=SimpleNamespace(),
        ip_hash_key=b"test",
        rate_limit=rate_limit,
        ip_rate_limit=ip_rate_limit,
    )


def test_shared_nat_contributors_have_independent_budgets() -> None:
    """A venue NAT full of contributors must not share one hourly budget."""
    backend = _rate_backend(FakeTable(), rate_limit=1, ip_rate_limit=400)
    for index in range(300):
        backend._consume_rate_limit("203.0.113.7", f"contributor-{index}")

    with pytest.raises(RateLimited):
        backend._consume_rate_limit("203.0.113.7", "contributor-0")


def test_contributor_budget_refills_continuously() -> None:
    """Retry-After reflects the next token, not the top of the clock hour."""
    table = FakeTable()
    backend = _rate_backend(table, rate_limit=2)
    backend._consume_rate_limit("198.51.100.9", "octocat")
    backend._consume_rate_limit("198.51.100.9", "octocat")

    with pytest.raises(RateLimited) as excinfo:
        backend._consume_rate_limit("198.51.100.9", "octocat")
    assert 1 <= excinfo.value.retry_after <= 1800

    bucket = next(
        entity
        for entity in reversed(table.entities)
        if entity["PartitionKey"] == "ratebucket"
        and entity["RowKey"].startswith("contributor-")
    )
    bucket["updated_at"] = (datetime.now(UTC) - timedelta(minutes=31)).isoformat()
    backend._consume_rate_limit("198.51.100.9", "octocat")


def test_token_bucket_survives_contended_conditional_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards PR #1027's burst fix: ETag races must not fail a crowd closed."""

    class ContendedTable(FakeTable):
        def __init__(self) -> None:
            super().__init__()
            self.conflicts = 0

        def update_entity(self, *, entity: dict, etag: str, **kwargs) -> None:
            from azure.core.exceptions import ResourceModifiedError

            if self.conflicts:
                self.conflicts -= 1
                raise ResourceModifiedError("etag changed")
            super().update_entity(entity=entity, etag=etag, **kwargs)

    table = ContendedTable()
    backend = _rate_backend(table, rate_limit=20, ip_rate_limit=500)
    waits: list[float] = []
    monkeypatch.setattr(
        "services.trajectory_upload.azure_backend.time",
        SimpleNamespace(sleep=waits.append),
    )
    backend._consume_rate_limit("203.0.113.9", "contended")

    table.conflicts = 8
    backend._consume_rate_limit("203.0.113.9", "contended")
    assert waits  # the survivor backed off between optimistic retries


def test_ip_backstop_caps_identity_rotation() -> None:
    """One host rotating github ids is still bounded by the per-IP bucket."""
    backend = _rate_backend(FakeTable(), rate_limit=20, ip_rate_limit=3)
    for index in range(3):
        backend._consume_rate_limit("192.0.2.4", f"rotating-{index}")

    with pytest.raises(RateLimited):
        backend._consume_rate_limit("192.0.2.4", "rotating-3")

    backend._consume_rate_limit("192.0.2.5", "rotating-3")


def test_validator_drain_stops_on_empty_queue_and_budget() -> None:
    class CountingValidator:
        def __init__(self, available: int) -> None:
            self.available = available
            self.calls = 0

        def run_once(self) -> bool:
            self.calls += 1
            return self.calls <= self.available

    counting = CountingValidator(available=3)
    assert drain(counting, budget_seconds=30.0) == 3
    assert counting.calls == 4

    exhausted = CountingValidator(available=10)
    assert drain(exhausted, budget_seconds=0.0) == 0
    assert exhausted.calls == 0


def _quarantine_capture(tmp_path: Path) -> tuple[str, dict[str, bytes]]:
    trial = _trial(tmp_path)
    with stage_trajectory_capture(
        trial,
        source_id="demo",
        github_id="benchflow-ai",
        email="contributor@benchflow.ai",
    ) as staged:
        digest = staged.traj_digest
        prefix = f"inbox/{digest}/"
        blobs = {
            prefix + item.relname: item.local_path.read_bytes() for item in staged.files
        }
    return digest, blobs


def _quarantine_report_capture(
    tmp_path: Path,
) -> tuple[str, str, str, dict[str, bytes]]:
    source = tmp_path / "capture.jsonl"
    source.write_text(
        '{"type":"user_message","text":"Please inspect"}\n', encoding="utf-8"
    )
    with stage_trajectory_artifacts(source, source_id="report-demo") as artifacts:
        report = build_trajectory_report(
            artifacts.files,
            masked_values=artifacts.redaction_replacements,
        )
        staged = finalize_trajectory_capture(
            artifacts,
            github_id="benchflow-ai",
            email="contributor@benchflow.ai",
            trajectory_report=report.as_manifest_metadata(),
        )
        digest = staged.traj_digest
        attempt_id = "u_" + "2" * 32
        prefix = f"inbox/{digest}/{attempt_id}/"
        blobs = {
            prefix + item.relname: item.local_path.read_bytes() for item in staged.files
        }
        manifest_sha256 = staged.files[-1].sha256
    return digest, attempt_id, manifest_sha256, blobs


def test_queue_validator_enforces_schema_12_manifest_binding(tmp_path: Path) -> None:
    """Guards PR #1008 against promoting a manifest changed after handshake."""
    digest, attempt_id, _manifest_sha256, blobs = _quarantine_report_capture(tmp_path)
    manifest_name = f"inbox/{digest}/{attempt_id}/manifest.json"
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
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "pending",
                "attempt_id": attempt_id,
                "declared_manifest_sha256": "0" * 64,
            }
        ]
    )

    AzureCaptureValidator(container=container, queue=queue, table=table).run_once()

    assert table.entities[-1]["status"] == "rejected"
    assert "manifest sha256 does not match declaration" in table.entities[-1]["detail"]
    assert container.uploaded == []


def test_queue_validator_promotes_bound_schema_12_report(tmp_path: Path) -> None:
    """Guards PR #1008: a bound, recomputed report reaches community storage."""
    digest, attempt_id, manifest_sha256, blobs = _quarantine_report_capture(tmp_path)
    manifest_name = f"inbox/{digest}/{attempt_id}/manifest.json"
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
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "pending",
                "attempt_id": attempt_id,
                "declared_manifest_sha256": manifest_sha256,
            }
        ]
    )

    AzureCaptureValidator(container=container, queue=queue, table=table).run_once()

    assert table.entities[-1]["status"] == "ingested"
    promoted = json.loads(container.blobs[f"sources/community/{digest}/manifest.json"])
    assert promoted["trajectory_report"]["human_steps"] == 1


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
    promoted_manifest = json.loads(
        container.blobs[f"sources/community/{digest}/manifest.json"]
    )
    assert promoted_manifest["contributor"] == {
        "github_id": "benchflow-ai",
        "email": "contributor@benchflow.ai",
    }
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

    assert _capture_from_event(encoded) == (
        f"inbox/{digest}/",
        digest,
        None,
        "manifest.json",
    )


def test_attempt_scoped_event_path_is_parsed(tmp_path: Path) -> None:
    """Guards PR #989 attempt isolation in Event Grid validation."""
    digest, _ = _quarantine_capture(tmp_path)
    attempt_id = "u_" + "a" * 32
    event = json.dumps(
        {
            "data": {
                "url": (
                    "https://account.blob.core.windows.net/bronze/"
                    f"inbox/{digest}/{attempt_id}/manifest.json"
                )
            }
        }
    )

    assert _capture_from_event(event) == (
        f"inbox/{digest}/{attempt_id}/",
        digest,
        attempt_id,
        "manifest.json",
    )


def test_stale_attempt_event_cleans_only_its_prefix(tmp_path: Path) -> None:
    """Guards PR #989 against an old SAS attempt changing the active ledger."""
    digest, legacy_blobs = _quarantine_capture(tmp_path)
    old_attempt = "u_" + "0" * 32
    current_attempt = "u_" + "1" * 32
    old_prefix = f"inbox/{digest}/{old_attempt}/"
    current_prefix = f"inbox/{digest}/{current_attempt}/"
    blobs = {
        name.replace(f"inbox/{digest}/", old_prefix): value
        for name, value in legacy_blobs.items()
    }
    blobs[current_prefix + "trajectory/current.jsonl"] = b'{"safe":true}\n'
    event = json.dumps(
        {
            "data": {
                "url": (
                    "https://account.blob.core.windows.net/bronze/"
                    f"{old_prefix}manifest.json"
                )
            }
        }
    )
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "pending",
                "attempt_id": current_attempt,
            }
        ]
    )
    container = FakeContainer(blobs)
    queue = FakeQueue(event)

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()

    assert not any(name.startswith(old_prefix) for name in container.blobs)
    assert current_prefix + "trajectory/current.jsonl" in container.blobs
    assert table.entities[-1]["status"] == "pending"
    assert table.entities[-1]["attempt_id"] == current_attempt
    assert queue.deleted == [("m1", "p1")]


def test_queue_validator_promotes_attempt_scoped_capture(tmp_path: Path) -> None:
    """Guards PR #989 end-to-end validation of an isolated upload attempt."""
    digest, legacy_blobs = _quarantine_capture(tmp_path)
    attempt_id = "u_" + "a" * 32
    legacy_prefix = f"inbox/{digest}/"
    prefix = f"{legacy_prefix}{attempt_id}/"
    blobs = {
        name.replace(legacy_prefix, prefix): value
        for name, value in legacy_blobs.items()
    }
    event = json.dumps(
        {
            "data": {
                "url": (
                    "https://account.blob.core.windows.net/bronze/"
                    f"{prefix}manifest.json"
                )
            }
        }
    )
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "pending",
                "attempt_id": attempt_id,
            }
        ]
    )
    container = FakeContainer(blobs)
    queue = FakeQueue(event)

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()

    assert container.uploaded[-1] == f"sources/community/{digest}/manifest.json"
    assert not any(name.startswith(prefix) for name in container.blobs)
    assert table.entities[-1]["status"] == "ingested"
    assert table.entities[-1]["attempt_id"] == attempt_id
    assert queue.deleted == [("m1", "p1")]


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


def test_pending_oversized_artifact_event_is_rejected_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #989 against retaining artifact-only storage amplification."""
    digest, blobs = _quarantine_capture(tmp_path)
    artifact = next(name for name in blobs if name.endswith(".jsonl"))
    event = json.dumps(
        {"data": {"url": ("https://account.blob.core.windows.net/bronze/" + artifact)}}
    )
    queue = FakeQueue(event)
    table = _pending_table(digest)
    container = FakeContainer(blobs)
    monkeypatch.setattr("services.trajectory_upload.validator.MAX_ARTIFACT_BYTES", 1)

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()

    assert not any(name.startswith(f"inbox/{digest}/") for name in container.blobs)
    assert table.entities[-1]["status"] == "rejected"
    assert "artifact exceeds 1 bytes" in table.entities[-1]["detail"]
    assert queue.deleted == [("m1", "p1")]


def test_pending_artifact_event_enforces_declared_size(tmp_path: Path) -> None:
    """Guards PR #989 against uploads larger than broker-declared artifact bytes."""
    digest, blobs = _quarantine_capture(tmp_path)
    prefix = f"inbox/{digest}/"
    artifact = next(name for name in blobs if name.endswith(".jsonl"))
    relname = artifact.removeprefix(prefix)
    event = json.dumps(
        {"data": {"url": ("https://account.blob.core.windows.net/bronze/" + artifact)}}
    )
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "pending",
                "declared_artifacts": json.dumps({relname: 1}),
            }
        ]
    )
    container = FakeContainer(blobs)
    queue = FakeQueue(event)

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()

    assert not any(name.startswith(prefix) for name in container.blobs)
    assert table.entities[-1]["status"] == "rejected"
    assert "does not match declaration" in table.entities[-1]["detail"]
    assert queue.deleted == [("m1", "p1")]


def test_pending_artifact_event_enforces_legacy_aggregate_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #989 against aggregate amplification without ledger metadata."""
    digest, blobs = _quarantine_capture(tmp_path)
    prefix = f"inbox/{digest}/"
    artifact = next(name for name in blobs if name.endswith(".jsonl"))
    content = blobs[artifact]
    blobs[prefix + "trajectory/second.jsonl"] = content
    capture_limit = len(content) * 2 - 1
    event = json.dumps(
        {"data": {"url": ("https://account.blob.core.windows.net/bronze/" + artifact)}}
    )
    table = _pending_table(digest)
    container = FakeContainer(blobs)
    queue = FakeQueue(event)
    monkeypatch.setattr(
        "services.trajectory_upload.validator.MAX_ARTIFACT_BYTES", len(content) + 1
    )
    monkeypatch.setattr(
        "services.trajectory_upload.validator.MAX_CAPTURE_BYTES", capture_limit
    )

    assert AzureCaptureValidator(
        container=container, queue=queue, table=table
    ).run_once()

    assert not any(name.startswith(prefix) for name in container.blobs)
    assert table.entities[-1]["status"] == "rejected"
    assert f"capture exceeds {capture_limit} bytes" in table.entities[-1]["detail"]
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
    script = Path("services/trajectory_upload/scripts/deploy.sh").read_text()

    assert (
        "map(select((.source | ascii_downcase) == ($source | ascii_downcase)))"
        in script
    )
    assert "--query '[0].name'" not in script


def test_deploy_scopes_event_delivery_identity_to_validation_queue() -> None:
    """Guards PR #989 against granting Event Grid account-wide queue access."""
    script = Path("services/trajectory_upload/scripts/deploy.sh").read_text()

    assert (
        '"$task_system_topic_principal_id" \\\n'
        '    "Storage Queue Data Message Sender" \\\n'
        '    "$task_queue_scope"'
    ) in script
    assert (
        "az role assignment delete \\\n"
        '        --assignee-object-id "$task_system_topic_principal_id" \\\n'
        '        --role "Storage Queue Data Message Sender" \\\n'
        '        --scope "$task_storage_id"'
    ) in script


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


def test_manifest_rejects_unsafe_metadata_version_before_promotion(
    tmp_path: Path,
) -> None:
    """Guards PR #989 against injecting invalid Azure metadata headers."""
    digest, blobs = _quarantine_capture(tmp_path)
    manifest_name = f"inbox/{digest}/manifest.json"
    manifest = json.loads(blobs[manifest_name])
    manifest["tool"]["version"] = "bad\nvalue"
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
    assert container.uploaded == []
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


class FakeStatusBroker:
    """Status-route double; the handshake route is never exercised here."""

    def __init__(self, result: CaptureStatusInfo | Exception) -> None:
        self.result = result
        self.client_ip: str | None = None
        self.digest: str | None = None

    def create_upload(self, request: UploadRequest, *, client_ip: str) -> UploadGrant:
        raise AssertionError("status tests must not reach the handshake route")

    def get_capture_status(self, digest: str, *, client_ip: str) -> CaptureStatusInfo:
        self.digest = digest
        self.client_ip = client_ip
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_capture_status_route_reports_states_and_rate_limit() -> None:
    """GET /v1/uploads/{digest} maps ledger states, 429s, and forwarded IPs."""
    digest = "ab" * 32
    backend = FakeStatusBroker(
        CaptureStatusInfo(
            digest=digest,
            status="ingested",
            prefix=f"sources/community/{digest}/",
            updated_at="2026-08-16T00:00:00+00:00",
        )
    )
    response = TestClient(create_app(backend)).get(
        f"/v1/uploads/{digest}",
        headers={"x-forwarded-for": "spoofed, 203.0.113.9"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["digest"] == f"sha256:{digest}"
    assert payload["status"] == "ingested"
    assert payload["prefix"] == f"sources/community/{digest}/"
    assert backend.digest == digest
    assert backend.client_ip == "203.0.113.9"

    rejected = TestClient(
        create_app(
            FakeStatusBroker(
                CaptureStatusInfo(
                    digest=digest, status="rejected", detail="size mismatch"
                )
            )
        )
    ).get(f"/v1/uploads/{digest}")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert rejected.json()["detail"] == "size mismatch"
    assert "prefix" not in rejected.json()

    unknown = TestClient(
        create_app(FakeStatusBroker(CaptureStatusInfo(digest=digest, status="unknown")))
    ).get(f"/v1/uploads/{digest}")
    assert unknown.status_code == 200
    assert unknown.json() == {"digest": f"sha256:{digest}", "status": "unknown"}

    limited = TestClient(create_app(FakeStatusBroker(RateLimited(42)))).get(
        f"/v1/uploads/{digest}"
    )
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "42"


def test_capture_status_route_rejects_malformed_digests() -> None:
    """Only a 64-lowercase-hex digest reaches the backend; 404 stays reserved
    for deployments that predate the endpoint."""
    backend = FakeStatusBroker(AssertionError("malformed digest reached the backend"))
    client = TestClient(create_app(backend))

    non_hex = client.get(f"/v1/uploads/{'zz' * 32}")
    assert non_hex.status_code == 400

    uppercase = client.get(f"/v1/uploads/{'AB' * 32}")
    assert uppercase.status_code == 400

    too_short = client.get(f"/v1/uploads/{'ab' * 31}")
    assert too_short.status_code == 400


def test_azure_backend_status_reads_ledger_without_upload_quota() -> None:
    """Status polls consume a separate, larger budget than upload grants."""
    digest = "ab" * 32
    table = FakeTable(
        [
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": "ingested",
                "updated_at": "2026-08-16T00:00:00+00:00",
            }
        ]
    )
    backend = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=table,
        blob_service=SimpleNamespace(),
        ip_hash_key=b"test",
        rate_limit=0,  # any upload-quota consumption would 429 immediately
        status_rate_limit=3,
    )

    info = backend.get_capture_status(digest, client_ip="127.0.0.1")
    assert info.status == "ingested"
    assert info.prefix == f"sources/community/{digest}/"
    assert info.updated_at == "2026-08-16T00:00:00+00:00"

    assert backend.get_capture_status("cd" * 32, client_ip="127.0.0.1").status == (
        "unknown"
    )
    backend.get_capture_status(digest, client_ip="127.0.0.1")
    with pytest.raises(RateLimited):
        backend.get_capture_status(digest, client_ip="127.0.0.1")

    partitions = {entity["PartitionKey"] for entity in table.entities}
    assert any(partition.startswith("rate-status-") for partition in partitions)
    assert not any(
        partition.startswith("rate-") and not partition.startswith("rate-status-")
        for partition in partitions
    )


def test_azure_backend_status_bounds_detail_and_hides_corrupt_states() -> None:
    """Rejection detail is truncated and non-public ledger states stay opaque."""
    digest = "ab" * 32
    rejected = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=FakeTable(
            [
                {
                    "PartitionKey": "capture",
                    "RowKey": digest,
                    "status": "rejected",
                    "detail": "x" * 600,
                }
            ]
        ),
        blob_service=SimpleNamespace(),
        ip_hash_key=b"test",
    )
    info = rejected.get_capture_status(digest, client_ip="127.0.0.1")
    assert info.status == "rejected"
    assert info.detail == "x" * 512

    corrupt = AzureUploadBroker(
        account_name="account",
        container="bronze",
        table=FakeTable(
            [{"PartitionKey": "capture", "RowKey": digest, "status": "exploded"}]
        ),
        blob_service=SimpleNamespace(),
        ip_hash_key=b"test",
    )
    assert corrupt.get_capture_status(digest, client_ip="127.0.0.1").status == "unknown"
