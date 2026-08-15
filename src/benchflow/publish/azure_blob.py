"""Direct Azure Blob publishing for staged trajectory captures."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from benchflow import __version__
from benchflow.publish.traj_capture import StagedCapture


@dataclass(frozen=True)
class BlobPublishResult:
    account: str
    container: str
    prefix: str
    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]

    @property
    def url(self) -> str:
        return (
            f"https://{self.account}.blob.core.windows.net/"
            f"{self.container}/{self.prefix}"
        )


def upload_capture_direct(
    staged: StagedCapture,
    *,
    container_url: str,
    credential: Any = None,
) -> BlobPublishResult:
    """Create staged files in an Azure container without overwriting blobs."""
    try:
        from azure.core.exceptions import (
            ClientAuthenticationError,
            HttpResponseError,
            ResourceExistsError,
            ResourceNotFoundError,
        )
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient, ContentSettings
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ValueError(
            "azure-storage-blob and azure-identity are required for --direct; "
            "install with: pip install 'benchflow[azure]'"
        ) from exc

    with _quiet_azure_request_logging():
        account, container = _parse_container_url(container_url)
        auth = credential if credential is not None else DefaultAzureCredential()
        client = ContainerClient.from_container_url(container_url, credential=auth)
        prefix = f"sources/{staged.source_id}/{staged.traj_digest}/"
        metadata = {
            "source_id": staged.source_id,
            "traj_digest": f"sha256:{staged.traj_digest}",
            "schema_version": str(staged.manifest["schema_version"]),
            "manifest": "manifest.json",
            "benchflow_version": __version__,
        }
        uploaded: list[str] = []
        skipped: list[str] = []
        for staged_file in staged.files:
            blob_name = prefix + staged_file.relname
            try:
                with staged_file.local_path.open("rb") as stream:
                    client.upload_blob(
                        name=blob_name,
                        data=stream,
                        overwrite=False,
                        content_settings=ContentSettings(
                            content_type=staged_file.content_type
                        ),
                        metadata=metadata,
                    )
            except ResourceExistsError:
                skipped.append(blob_name)
            except ResourceNotFoundError as exc:
                raise ValueError(
                    f"Azure container not found: {container_url}; create the storage "
                    "account and private container before using --direct"
                ) from exc
            except ClientAuthenticationError as exc:
                raise ValueError(
                    "Azure authentication failed; run 'az login' (or configure managed "
                    "identity) and grant the Blob Data Creator role"
                ) from exc
            except HttpResponseError as exc:
                if getattr(exc, "status_code", None) == 403:
                    raise ValueError(
                        "Azure upload was forbidden; run 'az login' (or configure "
                        "managed identity) and grant the Blob Data Creator role"
                    ) from exc
                raise ValueError(
                    f"Azure Blob upload failed for {blob_name}: {exc}"
                ) from exc
            else:
                uploaded.append(blob_name)
    return BlobPublishResult(
        account=account,
        container=container,
        prefix=prefix,
        uploaded=tuple(uploaded),
        skipped=tuple(skipped),
    )


@contextmanager
def _quiet_azure_request_logging():
    """Keep Azure SDK request diagnostics out of the one-line CLI result."""
    azure_logger = logging.getLogger("azure")
    previous_level = azure_logger.level
    azure_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        azure_logger.setLevel(previous_level)


def _parse_container_url(container_url: str) -> tuple[str, str]:
    parsed = urlparse(container_url.rstrip("/"))
    host_suffix = ".blob.core.windows.net"
    container = parsed.path.strip("/")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(host_suffix)
        or not container
        or "/" in container
    ):
        raise ValueError(
            "--container-url must look like "
            "https://<account>.blob.core.windows.net/<container>"
        )
    account = parsed.hostname[: -len(host_suffix)]
    if not account:
        raise ValueError("--container-url is missing the storage account name")
    return account, container
