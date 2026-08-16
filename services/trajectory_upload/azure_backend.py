"""Managed-identity Azure backend for broker SAS grants and durable rate limits."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

from services.trajectory_upload.broker_app import (
    AlreadyUploaded,
    RateLimited,
    UploadDeclarationConflict,
)
from services.trajectory_upload.contract import (
    CaptureStatusInfo,
    UploadGrant,
    UploadObject,
    UploadRequest,
)

# Ledger states a contributor may see. Anything else in the table (corruption,
# a future migration) is reported as ``unknown`` rather than leaked verbatim.
_PUBLIC_STATUSES = frozenset({"pending", "validating", "ingested", "rejected"})


class AzureUploadBroker:
    """Mint blob-scoped user-delegation SAS URLs without blob read authority."""

    def __init__(
        self,
        *,
        account_name: str,
        container: str,
        table: Any,
        blob_service: Any,
        ip_hash_key: bytes,
        rate_limit: int = 20,
        ip_rate_limit: int = 500,
        status_rate_limit: int = 720,
        sas_minutes: int = 15,
    ) -> None:
        self.account_name = account_name
        self.container = container
        self.table = table
        self.blob_service = blob_service
        self.ip_hash_key = ip_hash_key
        self.rate_limit = rate_limit
        self.ip_rate_limit = ip_rate_limit
        # Status polls are cheap ledger reads, so their ceiling is far above
        # the upload grant limit: one CLI wait polls a handful of times per
        # minute for a few minutes.
        self.status_rate_limit = status_rate_limit
        self.sas_minutes = min(max(sas_minutes, 1), 15)
        self._state_lock = threading.Lock()
        self._delegation_key: Any = None
        self._delegation_key_expires = datetime.min.replace(tzinfo=UTC)

    @classmethod
    def from_env(cls) -> AzureUploadBroker:
        from azure.data.tables import TableClient
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        account_name = _required_env("AZURE_STORAGE_ACCOUNT_NAME")
        container = os.environ.get("AZURE_BLOB_CONTAINER", "bronze")
        table_name = os.environ.get("AZURE_LEDGER_TABLE", "trajectoryuploads")
        credential = DefaultAzureCredential()
        return cls(
            account_name=account_name,
            container=container,
            table=TableClient(
                endpoint=f"https://{account_name}.table.core.windows.net",
                table_name=table_name,
                credential=credential,
            ),
            blob_service=BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=credential,
            ),
            ip_hash_key=_required_env("TRAJ_UPLOAD_IP_HASH_KEY").encode(),
            rate_limit=int(os.environ.get("TRAJ_UPLOAD_RATE_LIMIT", "20")),
            ip_rate_limit=int(os.environ.get("TRAJ_UPLOAD_IP_RATE_LIMIT", "500")),
            status_rate_limit=int(os.environ.get("TRAJ_STATUS_RATE_LIMIT", "720")),
            sas_minutes=int(os.environ.get("TRAJ_UPLOAD_SAS_MINUTES", "15")),
        )

    def create_upload(self, request: UploadRequest, *, client_ip: str) -> UploadGrant:
        digest = request.traj_digest.removeprefix("sha256:")
        upload_id = f"u_{uuid.uuid4().hex}"
        self._consume_rate_limit(client_ip, request.contributor.github_id)
        entity = self._capture_entity(digest)
        status = entity.get("status") if entity is not None else None
        if status == "ingested":
            raise AlreadyUploaded(
                base_url=self.container_url,
                prefix=f"sources/community/{digest}/",
            )
        declaration_conflict = (
            status in {"pending", "validating"}
            and request.manifest_sha256 is not None
            and entity is not None
            and entity.get("declared_manifest_sha256") != request.manifest_sha256
        )
        if status == "validating" and declaration_conflict:
            raise UploadDeclarationConflict
        if status in {"pending", "validating"} and not declaration_conflict:
            if entity is None:  # pragma: no cover - implied by status above
                raise RuntimeError("active capture ledger entry is missing")
            persisted_attempt = entity.get("attempt_id")
            if persisted_attempt and not _valid_upload_id(persisted_attempt):
                raise RuntimeError("active capture ledger has an invalid attempt id")
            if isinstance(persisted_attempt, str) and persisted_attempt:
                upload_id = persisted_attempt
                prefix = f"inbox/{digest}/{upload_id}/"
            else:
                # Preserve the pre-attempt namespace for captures that were
                # already in flight when this version was deployed.
                prefix = f"inbox/{digest}/"
        else:
            self.table.upsert_entity(
                {
                    "PartitionKey": "capture",
                    "RowKey": digest,
                    "status": "pending",
                    "attempt_id": upload_id,
                    "source_id": request.source_id,
                    "schema_version": request.schema_version,
                    "declared_manifest_sha256": request.manifest_sha256 or "",
                    "declared_artifacts": json.dumps(
                        {
                            artifact.name: artifact.bytes
                            for artifact in request.artifacts
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "validation_lease_until": "",
                    "detail": "",
                }
            )
            prefix = f"inbox/{digest}/{upload_id}/"

        now = datetime.now(UTC)
        expires_at = now + timedelta(minutes=self.sas_minutes)
        delegation_key = self._user_delegation_key(now)
        objects = tuple(
            self._upload_object(
                prefix=prefix,
                relname=name,
                content_type=content_type,
                delegation_key=delegation_key,
                starts_at=now - timedelta(minutes=5),
                expires_at=expires_at,
            )
            for name, content_type in [
                *((item.name, "application/jsonl") for item in request.artifacts),
                ("manifest.json", "application/json"),
            ]
        )
        return UploadGrant(
            upload_id=upload_id,
            bucket=self.container,
            base_url=self.container_url,
            prefix=prefix,
            objects=objects,
            expires_at=expires_at,
        )

    def get_capture_status(self, digest: str, *, client_ip: str) -> CaptureStatusInfo:
        """Report the ledger state of one digest for the public status route.

        The ledger row is written by the broker at handshake time and advanced
        by the validator, and ``ingested`` is only recorded after promotion to
        ``sources/community/<digest>/`` completes — so that status is proof the
        capture is present in durable storage. Only the bounded, user-fixable
        rejection detail is exposed; source ids and attempt internals are not.
        """
        self._consume_token("status", client_ip, self.status_rate_limit)
        entity = self._capture_entity(digest)
        if entity is None:
            return CaptureStatusInfo(digest=digest, status="unknown")
        status = entity.get("status")
        if not isinstance(status, str) or status not in _PUBLIC_STATUSES:
            return CaptureStatusInfo(digest=digest, status="unknown")
        updated_at = entity.get("updated_at")
        info = CaptureStatusInfo(
            digest=digest,
            status=status,
            updated_at=updated_at if isinstance(updated_at, str) else None,
        )
        if status == "ingested":
            return CaptureStatusInfo(
                digest=info.digest,
                status=info.status,
                updated_at=info.updated_at,
                prefix=f"sources/community/{digest}/",
            )
        if status == "rejected":
            detail = entity.get("detail")
            return CaptureStatusInfo(
                digest=info.digest,
                status=info.status,
                updated_at=info.updated_at,
                detail=detail[:512] if isinstance(detail, str) and detail else None,
            )
        return info

    @property
    def container_url(self) -> str:
        return f"https://{self.account_name}.blob.core.windows.net/{self.container}"

    def _capture_entity(self, digest: str) -> Any | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return self.table.get_entity(partition_key="capture", row_key=digest)
        except ResourceNotFoundError:
            return None

    def _consume_rate_limit(self, client_ip: str, github_id: str) -> None:
        # Fairness first: every contributor gets an independent budget, so a
        # venue NAT full of people cannot starve each other. The per-IP bucket
        # is a wider abuse backstop against one host rotating identities.
        contributor_key = f"{client_ip}\n{github_id.strip().lower()}"
        self._consume_token("contributor", contributor_key, self.rate_limit)
        self._consume_token("ip", client_ip, self.ip_rate_limit)

    def _consume_token(self, scope: str, key: str, capacity: int) -> None:
        """Take one token from an hourly-refilling bucket, atomically.

        The bucket admits a burst of ``capacity`` and refills continuously at
        ``capacity`` per hour, so Retry-After is seconds until the next token
        rather than the remainder of a clock hour.
        """
        from azure.core import MatchConditions
        from azure.core.exceptions import (
            ResourceExistsError,
            ResourceModifiedError,
            ResourceNotFoundError,
        )

        digest = hmac.new(self.ip_hash_key, key.encode(), hashlib.sha256).hexdigest()
        row = f"{scope}-{digest}"
        for attempt in range(10):
            if attempt:
                # A whole venue shares one IP bucket row, so a synchronized
                # burst loses ETag races in lockstep. Jitter de-synchronizes
                # the optimistic retries instead of failing crowds closed.
                time.sleep(random.uniform(0.005, 0.05) * attempt)
            now = datetime.now(UTC)
            try:
                entity = self.table.get_entity(partition_key="ratebucket", row_key=row)
            except ResourceNotFoundError:
                try:
                    self.table.create_entity(
                        {
                            "PartitionKey": "ratebucket",
                            "RowKey": row,
                            "tokens": float(capacity - 1),
                            "updated_at": now.isoformat(),
                        }
                    )
                except ResourceExistsError:
                    continue
                return
            tokens = entity.get("tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, (int, float)):
                tokens = 0.0
            refilled_at = _parse_datetime(entity.get("updated_at")) or now
            elapsed = max((now - refilled_at).total_seconds(), 0.0)
            tokens = min(float(capacity), float(tokens) + elapsed * capacity / 3600.0)
            if tokens < 1.0:
                retry_after = math.ceil((1.0 - tokens) * 3600.0 / capacity)
                raise RateLimited(max(retry_after, 1))
            updated = dict(entity)
            updated.update({"tokens": tokens - 1.0, "updated_at": now.isoformat()})
            try:
                self.table.update_entity(
                    entity=updated,
                    mode="replace",
                    etag=entity.metadata["etag"],
                    match_condition=MatchConditions.IfNotModified,
                )
            except ResourceModifiedError:
                continue
            return
        # Contention exhausted the optimistic retries; ask for a short retry
        # instead of silently admitting the request past the limiter.
        raise RateLimited(1)

    def _user_delegation_key(self, now: datetime):
        with self._state_lock:
            if (
                self._delegation_key is not None
                and now < self._delegation_key_expires - timedelta(minutes=10)
            ):
                return self._delegation_key
            self._delegation_key_expires = now + timedelta(hours=1)
            self._delegation_key = self.blob_service.get_user_delegation_key(
                key_start_time=now - timedelta(minutes=5),
                key_expiry_time=self._delegation_key_expires,
            )
            return self._delegation_key

    def _upload_object(
        self,
        *,
        prefix: str,
        relname: str,
        content_type: str,
        delegation_key: Any,
        starts_at: datetime,
        expires_at: datetime,
    ) -> UploadObject:
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        blob_name = prefix + relname
        sas = generate_blob_sas(
            account_name=self.account_name,
            container_name=self.container,
            blob_name=blob_name,
            user_delegation_key=delegation_key,
            permission=BlobSasPermissions(create=True),
            start=starts_at,
            expiry=expires_at,
            protocol="https",
        )
        return UploadObject(
            name=relname,
            put_url=f"{self.container_url}/{quote(blob_name, safe='/')}?{sas}",
            headers={
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": content_type,
                "If-None-Match": "*",
            },
        )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is not set: {name}")
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _valid_upload_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 34
        and value.startswith("u_")
        and all(char in "0123456789abcdef" for char in value[2:])
    )
