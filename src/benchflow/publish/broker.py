"""Cloud-neutral client for the public trajectory upload broker."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from benchflow.publish.traj_capture import StagedCapture


@dataclass(frozen=True)
class BrokerPublishResult:
    base_url: str
    prefix: str
    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.prefix.lstrip('/')}"


def upload_capture_via_broker(
    staged: StagedCapture,
    *,
    broker_url: str,
    http_client: httpx.Client | None = None,
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
    }
    if contributor := staged.manifest.get("contributor"):
        request_body["contributor"] = contributor
    manager = nullcontext(http_client) if http_client is not None else httpx.Client()
    try:
        with _quiet_httpx_request_logging(), manager as client:
            response = client.post(endpoint, json=request_body, timeout=30)
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
                    put_response = client.put(
                        upload["put_url"],
                        headers=upload["headers"],
                        content=stream,
                        timeout=300,
                    )
                if put_response.status_code in {409, 412}:
                    skipped.append(object_name)
                    continue
                _raise_for_broker_response(
                    put_response,
                    operation=f"upload of {staged_file.relname}",
                )
                uploaded.append(object_name)
    except httpx.HTTPError as exc:
        raise ValueError(f"trajectory broker request failed: {exc}") from exc

    return BrokerPublishResult(
        base_url=base_url,
        prefix=prefix,
        uploaded=tuple(uploaded),
        skipped=tuple(skipped),
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
