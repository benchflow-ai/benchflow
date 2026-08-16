"""Cloud-neutral client for the public trajectory upload broker."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from benchflow.publish._progress import ProgressReader
from benchflow.publish.traj_capture import StagedCapture, StagedFile

# Scale-to-zero brokers can take longer than a default HTTP timeout to mint
# the first SAS grant. Retries are safe; keep this under two minutes.
HANDSHAKE_TIMEOUT_SEC = 90.0
PUT_TIMEOUT_SEC = 300.0
# Workspace archives can approach 1 GiB; give every PUT at least the base
# budget plus a floor of ~1 MiB/s of transfer time so slow uplinks finish.
PUT_TIMEOUT_BYTES_PER_SEC = 1024**2
# The broker's token buckets answer 429 with an honest, usually-short
# Retry-After. Waiting it out inline turns a crowd burst into a smooth queue;
# anything longer than this cap is surfaced to the contributor instead.
HANDSHAKE_RETRY_LIMIT = 3
HANDSHAKE_RETRY_MAX_WAIT_SEC = 120.0

# Capture states the broker's status endpoint may report. ``unsupported`` is
# client-synthesized for deployments that predate the endpoint (HTTP 404), and
# ``throttled`` for a rate-limited poll; neither ever comes from the ledger.
CAPTURE_STATES = frozenset({"pending", "validating", "ingested", "rejected", "unknown"})


@dataclass(frozen=True)
class BrokerPublishResult:
    base_url: str
    prefix: str
    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]
    # True only for a handshake 409: the ledger already recorded this digest as
    # ingested, so the capture is verified in storage without polling.
    already_ingested: bool = False

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.prefix.lstrip('/')}"


@dataclass(frozen=True)
class CaptureStatus:
    """One poll of the broker's capture-status endpoint."""

    status: str
    detail: str | None = None
    retry_after: float | None = None

    @property
    def terminal(self) -> bool:
        return self.status in {"ingested", "rejected"}


def fetch_capture_status(
    *,
    broker_url: str,
    traj_digest: str,
    http_client: httpx.Client | None = None,
) -> CaptureStatus:
    """Read the validation state of one uploaded capture from the broker.

    Returns a :class:`CaptureStatus` for every well-formed outcome, including
    ``unsupported`` when the deployed broker predates the endpoint (404) and
    ``throttled`` on HTTP 429, so a polling loop can decide without exception
    plumbing. Raises :class:`ValueError` only for transport failures or a
    malformed response, which callers should treat as transient.
    """
    digest = traj_digest.removeprefix("sha256:")
    endpoint = f"{broker_url.rstrip('/')}/v1/uploads/{digest}"
    manager = nullcontext(http_client) if http_client is not None else httpx.Client()
    try:
        with _quiet_httpx_request_logging(), manager as client:
            response = client.get(endpoint, timeout=HANDSHAKE_TIMEOUT_SEC)
    except httpx.HTTPError as exc:
        raise ValueError(f"capture status request failed: {exc}") from exc
    if response.status_code == 404:
        return CaptureStatus(status="unsupported")
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        try:
            seconds = float(retry_after) if retry_after else None
        except ValueError:
            seconds = None
        return CaptureStatus(status="throttled", retry_after=seconds)
    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:300]
        raise ValueError(
            f"capture status request failed with HTTP {response.status_code}: {detail}"
        )
    payload = _response_object(response)
    status = payload.get("status")
    if not isinstance(status, str) or status not in CAPTURE_STATES:
        raise ValueError("trajectory broker returned an unknown capture status")
    detail = payload.get("detail")
    return CaptureStatus(
        status=status,
        detail=detail if isinstance(detail, str) and detail else None,
    )


def upload_capture_via_broker(
    staged: StagedCapture,
    *,
    broker_url: str,
    http_client: httpx.Client | None = None,
    on_file_complete: Callable[[StagedFile], None] | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> BrokerPublishResult:
    """Request scoped upload URLs and PUT every staged file in server order."""
    endpoint = f"{broker_url.rstrip('/')}/v1/uploads"
    artifacts = staged.manifest["artifacts"]
    request_body = {
        "schema_version": staged.manifest["schema_version"],
        "kind": staged.manifest["kind"],
        "source_id": staged.manifest["source_id"],
        "traj_digest": staged.manifest["traj_digest"],
        "uploaded_by": staged.manifest["uploaded_by"],
        "artifacts": artifacts,
        "manifest_sha256": staged.files[-1].sha256,
    }
    if contributor := staged.manifest.get("contributor"):
        request_body["contributor"] = contributor
    manager = nullcontext(http_client) if http_client is not None else httpx.Client()
    try:
        with _quiet_httpx_request_logging(), manager as client:
            response = _post_handshake(client, endpoint, request_body)
            if response.status_code == 409:
                return _already_uploaded_result(response, staged, broker_url)
            _raise_for_broker_response(response, operation="upload handshake")
            payload = _response_object(response)
            objects = _validated_objects(payload, staged)
            base_url, prefix = _destination(payload)

            uploaded: list[str] = []
            skipped: list[str] = []
            for staged_file, upload in zip(staged.files, objects, strict=True):
                object_name = prefix + staged_file.relname
                with staged_file.local_path.open("rb") as stream:
                    content = (
                        stream if on_bytes is None else ProgressReader(stream, on_bytes)
                    )
                    put_response = client.put(
                        upload["put_url"],
                        headers=upload["headers"],
                        content=content,
                        timeout=PUT_TIMEOUT_SEC
                        + staged_file.size_bytes / PUT_TIMEOUT_BYTES_PER_SEC,
                    )
                if _is_create_only_conflict(put_response):
                    skipped.append(object_name)
                else:
                    _raise_for_broker_response(
                        put_response,
                        operation=f"upload of {staged_file.relname}",
                    )
                    uploaded.append(object_name)
                if on_file_complete is not None:
                    on_file_complete(staged_file)
    except httpx.TimeoutException as exc:
        raise ValueError(
            "upload timed out while reaching the service. "
            "Run the same command again — retries are safe."
        ) from exc
    except httpx.HTTPError as exc:
        raise ValueError(f"trajectory broker request failed: {exc}") from exc

    return BrokerPublishResult(
        base_url=base_url,
        prefix=prefix,
        uploaded=tuple(uploaded),
        skipped=tuple(skipped),
    )


def _post_handshake(
    client: httpx.Client, endpoint: str, request_body: dict[str, Any]
) -> httpx.Response:
    """POST the handshake, waiting out short broker 429s instead of failing."""
    response = client.post(endpoint, json=request_body, timeout=HANDSHAKE_TIMEOUT_SEC)
    for _retry in range(HANDSHAKE_RETRY_LIMIT):
        if response.status_code != 429:
            return response
        retry_after = _retry_after_seconds(response)
        if retry_after is None or retry_after > HANDSHAKE_RETRY_MAX_WAIT_SEC:
            return response
        time.sleep(retry_after + random.uniform(0.0, 1.0))
        response = client.post(
            endpoint, json=request_body, timeout=HANDSHAKE_TIMEOUT_SEC
        )
    return response


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _is_create_only_conflict(response: httpx.Response) -> bool:
    """Treat create-only blob conflicts as an idempotent skip.

    Azure user-delegation SAS with ``create`` and ``If-None-Match: *`` returns
    403 ``UnauthorizedBlobOverwrite`` when the blob already exists. GCS uses
    412. Neither should fail a contributor retry.
    """
    if response.status_code in {409, 412}:
        return True
    if response.status_code != 403:
        return False
    detail = response.text
    return (
        "UnauthorizedBlobOverwrite" in detail
        or "BlobAlreadyExists" in detail
        or "already exists" in detail.lower()
    )


@contextmanager
def _quiet_httpx_request_logging():
    """Keep short-lived signed upload URLs out of the global INFO stream."""
    httpx_logger = logging.getLogger("httpx")
    previous_level = httpx_logger.level
    httpx_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        httpx_logger.setLevel(previous_level)


def _already_uploaded_result(
    response: httpx.Response,
    staged: StagedCapture,
    broker_url: str,
) -> BrokerPublishResult:
    try:
        payload = _response_object(response)
        base_url, prefix = _destination(payload)
    except ValueError:
        base_url = broker_url.rstrip("/")
        prefix = f"sources/community/{staged.traj_digest}/"
    return BrokerPublishResult(
        base_url=base_url,
        prefix=prefix,
        uploaded=(),
        skipped=tuple(prefix + item.relname for item in staged.files),
        already_ingested=True,
    )


def _response_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("trajectory broker returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("trajectory broker returned a non-object response")
    return payload


def _validated_objects(
    payload: dict[str, Any], staged: StagedCapture
) -> list[dict[str, Any]]:
    objects = payload.get("objects")
    if not isinstance(objects, list) or not all(
        isinstance(item, dict) for item in objects
    ):
        raise ValueError("trajectory broker protocol violation: objects must be a list")
    expected_names = [item.relname for item in staged.files]
    names = [item.get("name") for item in objects]
    if names != expected_names:
        raise ValueError(
            "trajectory broker protocol violation: response objects must match "
            "the staged files in canonical manifest-last order"
        )
    for item in objects:
        put_url = item.get("put_url")
        if not isinstance(put_url, str):
            raise ValueError("trajectory broker protocol violation: missing put_url")
        parsed_url = urlparse(put_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise ValueError(
                "trajectory broker protocol violation: put_url must be an "
                "authenticated HTTPS URL"
            )
        headers = item.get("headers")
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ValueError(
                "trajectory broker protocol violation: headers must map strings to strings"
            )
    return objects


def _destination(payload: dict[str, Any]) -> tuple[str, str]:
    prefix = payload.get("prefix")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("trajectory broker protocol violation: missing prefix")
    base_url = payload.get("base_url")
    if isinstance(base_url, str) and base_url.startswith("https://"):
        return base_url, prefix
    bucket = payload.get("bucket")
    if isinstance(bucket, str) and bucket:
        return f"gs://{bucket}", prefix
    raise ValueError("trajectory broker protocol violation: missing destination")


def _raise_for_broker_response(response: httpx.Response, *, operation: str) -> None:
    if response.status_code < 400:
        return
    detail = response.text.strip().replace("\n", " ")[:300]
    if response.status_code == 413:
        raise ValueError(f"trajectory broker rejected an oversized capture: {detail}")
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        suffix = f"; retry after {retry_after}" if retry_after else ""
        raise ValueError(f"trajectory broker rate limit exceeded{suffix}: {detail}")
    raise ValueError(
        f"trajectory broker {operation} failed with HTTP {response.status_code}: {detail}"
    )
