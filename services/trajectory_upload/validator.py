"""Azure Queue-driven quarantine validator and manifest-last promoter."""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from services.trajectory_upload.contract import ARTIFACT_NAME, MAX_MANIFEST_BYTES
from services.trajectory_upload.validation import (
    CaptureRejected,
    ValidatedCapture,
    validate_local_capture,
    validate_manifest_bytes,
)

logger = logging.getLogger(__name__)
# The Container Apps Job times out after 30 minutes. A slightly longer lease
# prevents a replacement worker from overlapping a replica that Azure has not
# terminated yet; an abandoned queue message can reclaim the digest afterward.
VALIDATION_LEASE = timedelta(minutes=35)


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
            prefix, digest, is_manifest = _capture_from_event(message.content)
        except CaptureRejected as exc:
            logger.warning("discarding invalid validation event: %s", exc)
            self._delete_message(message)
            return True

        if self._capture_status(digest) in {"ingested", "rejected"}:
            self._cleanup_prefix(prefix)
            self._delete_message(message)
            return True

        # Artifacts arrive before the manifest. Their events keep the Job able
        # to clean terminal-state replays without treating an in-flight capture
        # as a prematurely committed upload.
        if not is_manifest:
            self._delete_message(message)
            return True

        if not self._claim_capture(digest):
            # A concurrent worker owns an unexpired validation lease. Leave the
            # queue message for a later retry so a crashed owner cannot strand
            # the capture permanently.
            status = self._capture_status(digest)
            if status in {"ingested", "rejected"} or status is None:
                self._cleanup_prefix(prefix)
                self._delete_message(message)
            return True

        from azure.core.exceptions import ResourceNotFoundError

        try:
            with tempfile.TemporaryDirectory(prefix="benchflow-validator-") as name:
                try:
                    validated = self._download_and_validate(prefix, Path(name))
                except ResourceNotFoundError as exc:
                    raise CaptureRejected("a declared capture blob is missing") from exc
                if validated.manifest.traj_digest != f"sha256:{digest}":
                    raise CaptureRejected(
                        "manifest digest does not match its quarantine prefix"
                    )
                self._promote(validated, digest)
        except CaptureRejected as exc:
            logger.warning("capture %s rejected: %s", digest, exc)
            self._record_status(digest, "rejected", detail=str(exc)[:512])
            self._cleanup_prefix(prefix)
        else:
            self._record_status(digest, "ingested")
            self._cleanup_prefix(prefix)
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
        manifest = validate_manifest_bytes(manifest_bytes)

        artifact_paths: dict[str, Path] = {}
        for artifact in manifest.artifacts:
            relname = artifact.name
            blob = self.container.get_blob_client(prefix + relname)
            if blob.get_blob_properties().size != artifact.bytes:
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
                "validation_lease_until": "",
                **extra,
            }
        )

    def _claim_capture(self, digest: str) -> bool:
        from azure.core import MatchConditions
        from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError
        from azure.data.tables import UpdateMode

        now = datetime.now(UTC)
        try:
            entity = self.table.get_entity(partition_key="capture", row_key=digest)
        except ResourceNotFoundError:
            return False
        status = entity.get("status")
        lease_until = _parse_datetime(entity.get("validation_lease_until"))
        if status in {"ingested", "rejected"}:
            return False
        if status == "validating" and lease_until is not None and lease_until > now:
            return False
        if status not in {"pending", "validating"}:
            return False

        claimed = dict(entity)
        claimed.update(
            {
                "status": "validating",
                "updated_at": now.isoformat(),
                "validation_lease_until": (now + VALIDATION_LEASE).isoformat(),
            }
        )
        try:
            self.table.update_entity(
                entity=claimed,
                mode=UpdateMode.MERGE,
                etag=entity.metadata["etag"],
                match_condition=MatchConditions.IfNotModified,
            )
        except ResourceModifiedError:
            return False
        return True

    def _capture_status(self, digest: str) -> str | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            entity = self.table.get_entity(partition_key="capture", row_key=digest)
        except ResourceNotFoundError:
            return None
        status = entity.get("status")
        return status if isinstance(status, str) else None

    def _delete_message(self, message: Any) -> None:
        self.queue.delete_message(message.id, message.pop_receipt)


def _capture_from_event(content: str) -> tuple[str, str, bool]:
    # Event Grid's Storage Queue destination base64-encodes its JSON envelope;
    # accepting plain JSON as well keeps the parser usable with test and replay
    # tools that already decode queue messages.
    if not content.lstrip().startswith(("{", "[")):
        try:
            content = base64.b64decode(content, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise CaptureRejected("invalid base64 Event Grid message") from exc
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
    if len(parts) < 4 or parts[1] != "inbox":
        raise CaptureRejected("event is not for an inbox capture object")
    digest = parts[2]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CaptureRejected("event contains an invalid trajectory digest")
    relname = "/".join(parts[3:])
    is_manifest = relname == "manifest.json"
    if not is_manifest and ARTIFACT_NAME.fullmatch(relname) is None:
        raise CaptureRejected("event is not for a declared capture object")
    return f"inbox/{digest}/", digest, is_manifest


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    AzureCaptureValidator.from_env().run_once()


if __name__ == "__main__":
    main()
