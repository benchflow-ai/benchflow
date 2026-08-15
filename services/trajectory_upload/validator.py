"""Azure Queue-driven quarantine validator and manifest-last promoter."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from services.trajectory_upload.contract import MAX_MANIFEST_BYTES
from services.trajectory_upload.validation import (
    CaptureRejected,
    ValidatedCapture,
    validate_local_capture,
)

logger = logging.getLogger(__name__)


class AzureCaptureValidator:
    """Consume one Event Grid message and promote a valid capture atomically."""

    def __init__(self, *, container: Any, queue: Any, table: Any) -> None:
        self.container = container
        self.queue = queue
        self.table = table

    @classmethod
    def from_env(cls) -> AzureCaptureValidator:
        from azure.data.tables import TableClient
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient
        from azure.storage.queue import QueueClient

        account_name = _required_env("AZURE_STORAGE_ACCOUNT_NAME")
        container_name = os.environ.get("AZURE_BLOB_CONTAINER", "bronze")
        queue_name = os.environ.get("AZURE_VALIDATION_QUEUE", "trajectory-validation")
        table_name = os.environ.get("AZURE_LEDGER_TABLE", "trajectoryuploads")
        credential = DefaultAzureCredential()
        return cls(
            container=ContainerClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                container_name=container_name,
                credential=credential,
            ),
            queue=QueueClient(
                account_url=f"https://{account_name}.queue.core.windows.net",
                queue_name=queue_name,
                credential=credential,
            ),
            table=TableClient(
                endpoint=f"https://{account_name}.table.core.windows.net",
                table_name=table_name,
                credential=credential,
            ),
        )

    def run_once(self) -> bool:
        """Process at most one queue message; return whether one was received."""
        messages = self.queue.receive_messages(
            messages_per_page=1,
            visibility_timeout=900,
        )
        message = next(iter(messages), None)
        if message is None:
            return False
        try:
            prefix, digest = _capture_from_event(message.content)
        except CaptureRejected as exc:
            logger.warning("discarding invalid validation event: %s", exc)
            self._delete_message(message)
            return True

        try:
            with tempfile.TemporaryDirectory(prefix="benchflow-validator-") as name:
                validated = self._download_and_validate(prefix, Path(name))
                if validated.manifest.traj_digest != f"sha256:{digest}":
                    raise CaptureRejected(
                        "manifest digest does not match its quarantine prefix"
                    )
                self._promote(validated, digest)
        except CaptureRejected as exc:
            logger.warning("capture %s rejected: %s", digest, exc)
            self._cleanup_prefix(prefix)
            self._record_status(digest, "rejected", detail=str(exc)[:512])
        else:
            self._cleanup_prefix(prefix)
            self._record_status(digest, "ingested")
            logger.info("capture %s promoted", digest)
        self._delete_message(message)
        return True

    def _download_and_validate(
        self, prefix: str, staging_dir: Path
    ) -> ValidatedCapture:
        manifest_blob = self.container.get_blob_client(prefix + "manifest.json")
        properties = manifest_blob.get_blob_properties()
        if properties.size > MAX_MANIFEST_BYTES:
            raise CaptureRejected("manifest exceeds the 1 MiB limit")
        manifest_bytes = manifest_blob.download_blob().readall()
        try:
            manifest_data = json.loads(manifest_bytes)
            artifacts = manifest_data["artifacts"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise CaptureRejected(f"invalid manifest: {exc}") from exc
        if not isinstance(artifacts, list):
            raise CaptureRejected("manifest artifacts must be a list")

        artifact_paths: dict[str, Path] = {}
        for item in artifacts:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise CaptureRejected("manifest contains an invalid artifact entry")
            relname = item["name"]
            blob = self.container.get_blob_client(prefix + relname)
            expected_size = item.get("bytes")
            if blob.get_blob_properties().size != expected_size:
                raise CaptureRejected(f"size mismatch for {relname}")
            local_path = staging_dir / Path(relname).name
            with local_path.open("wb") as output:
                blob.download_blob(max_concurrency=1).readinto(output)
            artifact_paths[relname] = local_path
        return validate_local_capture(manifest_bytes, artifact_paths)

    def _promote(self, capture: ValidatedCapture, digest: str) -> None:
        from azure.core.exceptions import ResourceExistsError
        from azure.storage.blob import ContentSettings

        prefix = f"sources/community/{digest}/"
        metadata = {
            "source_id": capture.manifest.source_id,
            "traj_digest": capture.manifest.traj_digest,
            "schema_version": capture.manifest.schema_version,
            "manifest": "manifest.json",
            "benchflow_version": capture.manifest.tool.version,
        }
        for artifact in capture.manifest.artifacts:
            with (
                suppress(ResourceExistsError),
                capture.artifact_paths[artifact.name].open("rb") as stream,
            ):
                self.container.upload_blob(
                    name=prefix + artifact.name,
                    data=stream,
                    overwrite=False,
                    metadata=metadata,
                    content_settings=ContentSettings(content_type="application/jsonl"),
                )
        with suppress(ResourceExistsError):
            self.container.upload_blob(
                name=prefix + "manifest.json",
                data=capture.manifest_bytes,
                overwrite=False,
                metadata=metadata,
                content_settings=ContentSettings(content_type="application/json"),
            )

    def _cleanup_prefix(self, prefix: str) -> None:
        for blob in self.container.list_blobs(name_starts_with=prefix):
            self.container.delete_blob(blob.name)

    def _record_status(self, digest: str, status: str, **extra: str) -> None:
        self.table.upsert_entity(
            {
                "PartitionKey": "capture",
                "RowKey": digest,
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(),
                **extra,
            }
        )

    def _delete_message(self, message: Any) -> None:
        self.queue.delete_message(message.id, message.pop_receipt)


def _capture_from_event(content: str) -> tuple[str, str]:
    try:
        event = json.loads(content)
        if isinstance(event, list):
            if len(event) != 1:
                raise CaptureRejected("event batch must contain exactly one event")
            event = event[0]
        blob_url = event["data"]["url"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CaptureRejected(f"invalid Event Grid message: {exc}") from exc
    if not isinstance(blob_url, str):
        raise CaptureRejected("Event Grid blob URL must be a string")
    parts = unquote(urlparse(blob_url).path).strip("/").split("/")
    if len(parts) != 4 or parts[1] != "inbox" or parts[3] != "manifest.json":
        raise CaptureRejected("event is not for inbox/<digest>/manifest.json")
    digest = parts[2]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CaptureRejected("event contains an invalid trajectory digest")
    return f"inbox/{digest}/", digest


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is not set: {name}")
    return value


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    AzureCaptureValidator.from_env().run_once()


if __name__ == "__main__":
    main()
