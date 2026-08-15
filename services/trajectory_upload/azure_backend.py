"""Managed-identity Azure backend for broker SAS grants and durable rate limits."""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

from services.trajectory_upload.broker_app import (
    AlreadyUploaded,
    RateLimited,
    RejectedUpload,
)
from services.trajectory_upload.contract import (
    UploadGrant,
    UploadObject,
    UploadRequest,
)


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
        sas_minutes: int = 15,
    ) -> None:
        self.account_name = account_name
        self.container = container
        self.table = table
        self.blob_service = blob_service
        self.ip_hash_key = ip_hash_key
        self.rate_limit = rate_limit
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
            sas_minutes=int(os.environ.get("TRAJ_UPLOAD_SAS_MINUTES", "15")),
        )

    def create_upload(self, request: UploadRequest, *, client_ip: str) -> UploadGrant:
        digest = request.traj_digest.removeprefix("sha256:")
        with self._state_lock:
            status = self._capture_status(digest)
            if status == "ingested":
                raise AlreadyUploaded(
                    base_url=self.container_url,
                    prefix=f"sources/community/{digest}/",
                )
            if status == "rejected":
                raise RejectedUpload
            self._consume_rate_limit(client_ip)
            if status is None:
                self.table.upsert_entity(
                    {
                        "PartitionKey": "capture",
                        "RowKey": digest,
                        "status": "pending",
                        "source_id": request.source_id,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )

        now = datetime.now(UTC)
        expires_at = now + timedelta(minutes=self.sas_minutes)
        delegation_key = self._user_delegation_key(now)
        prefix = f"inbox/{digest}/"
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
            upload_id=f"u_{uuid.uuid4().hex}",
            bucket=self.container,
            base_url=self.container_url,
            prefix=prefix,
            objects=objects,
            expires_at=expires_at,
        )

    @property
    def container_url(self) -> str:
        return f"https://{self.account_name}.blob.core.windows.net/{self.container}"

    def _capture_status(self, digest: str) -> str | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            entity = self.table.get_entity(partition_key="capture", row_key=digest)
        except ResourceNotFoundError:
            return None
        status = entity.get("status")
        return status if isinstance(status, str) else None

    def _consume_rate_limit(self, client_ip: str) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        now = datetime.now(UTC)
        partition = f"rate-{now:%Y%m%d%H}"
        row = hmac.new(
            self.ip_hash_key,
            client_ip.encode(),
            hashlib.sha256,
        ).hexdigest()
        try:
            entity = self.table.get_entity(partition_key=partition, row_key=row)
        except ResourceNotFoundError:
            count = 0
        else:
            count = int(entity.get("count", 0))
        if count >= self.rate_limit:
            retry_after = 3600 - now.minute * 60 - now.second
            raise RateLimited(max(retry_after, 1))
        self.table.upsert_entity(
            {
                "PartitionKey": partition,
                "RowKey": row,
                "count": count + 1,
                "updated_at": now.isoformat(),
            }
        )

    def _user_delegation_key(self, now: datetime):
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
