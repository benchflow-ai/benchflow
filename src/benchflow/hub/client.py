"""A client for a BenchFlow Hub: datasets by name, their control results, and pushed runs.

The wire format is the Hub registry v1 (``/api/v1/registry/...``, ``/api/v1/blobs/...``, ``/api/v1/runs/upload``):
JSON over HTTPS with a ``Bearer`` API key, errors as ``{"detail": "..."}``. Standard library only, so ``bench`` needs
no extra package to read a Hub.

Where a Hub is named:
- ``--hub <url>`` (on ``bench run``, the same as ``--registry hub:<url>``) reads that Hub only, as ``--registry <file
  or URL>`` reads one registry only;
- ``BENCHFLOW_HUB_URL`` adds a Hub after BenchFlow's own registry, for a dataset that registry does not hold;
- ``BENCHFLOW_HUB_TOKEN`` is the API key (``read`` to pull, ``upload`` to publish or push).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from benchflow.errors import UserError

HUB_URL_ENV = "BENCHFLOW_HUB_URL"
HUB_TOKEN_ENV = "BENCHFLOW_HUB_TOKEN"
HUB_REGISTRY_PREFIX = "hub:"
DATASET_SCHEMA = "benchflow.hub.dataset/1"
REGISTRY_PATH = "/api/v1/registry"
# The Hub's own list of its routes, which it serves to anyone: how bench tells a Hub without the dataset
# registry (every registry route unknown) from a registry that lacks one dataset.
ROUTE_LIST_PATH = "/api/openapi.json"
NO_REGISTRY = "no_registry"  # HubError.kind for a Hub without the dataset registry
_TIMEOUT_S = 60
_CHUNK = 1 << 20


_CREDENTIAL_HEADERS = ("Authorization", "Cookie")


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    port = parts.port or {"http": 80, "https": 443}.get(scheme)
    return scheme, (parts.hostname or "").lower(), port


class _DowngradeRefused(HTTPError):
    """A Hub redirected an HTTPS request to another scheme."""


class _SameOriginCredentials(HTTPRedirectHandler):
    """Follow a Hub's redirects without sending its key anywhere else.

    The API key goes with a redirect only to the origin it was sent to
    (scheme, host and port); a redirect to any other origin (a signed blob
    URL, a proxy, an attacker's server) is followed without the
    ``Authorization`` and ``Cookie`` headers, and stays without them for
    the rest of the chain. A redirect from HTTPS to anything else is refused.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urlsplit(req.full_url)
        if old.scheme.lower() == "https" and urlsplit(newurl).scheme.lower() != "https":
            raise _DowngradeRefused(
                req.full_url,
                code,
                f"refused a redirect from HTTPS to {newurl}",
                headers,
                fp,
            )
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _origin(req.full_url) != _origin(newurl):
            for name in _CREDENTIAL_HEADERS:
                new.remove_header(name)
        return new


_OPENER = build_opener(_SameOriginCredentials)


class HubError(UserError):
    """A Hub could not be reached, refused a request, or answered something unreadable."""

    exit_code: int = 1

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        hint: str | None = None,
        kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        if hint:
            self.hint = hint


def hub_from_registry(registry: str | None) -> str | None:
    """The Hub URL a ``--registry`` value names (``hub:<url>``), else None."""
    if registry and registry.startswith(HUB_REGISTRY_PREFIX):
        return normalize_hub_url(registry[len(HUB_REGISTRY_PREFIX) :])
    return None


def normalize_hub_url(url: str) -> str:
    """``http(s)://host[:port][/prefix]`` without a trailing slash; anything else is refused."""
    url = url.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        error = HubError(f"not a Hub address: {url!r}")
        error.exit_code = 2
        error.hint = (
            "name the Hub as http(s)://host[:port], e.g. --hub https://hub.example.org"
        )
        raise error
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        error = HubError(
            f"refusing plain http for a Hub that is not on this machine: {url}"
        )
        error.exit_code = 2
        error.hint = "use https://, or an SSH tunnel to a local port"
        raise error
    return url


def split_dataset_name(name: str) -> tuple[str, str]:
    """``org/name`` -> (org, name); Hub datasets always have an owner."""
    org, sep, rest = name.partition("/")
    if not sep or not org or not rest or "/" in rest:
        error = HubError(f"a Hub dataset is named org/name, got {name!r}")
        error.exit_code = 2
        raise error
    return org, rest


@dataclass(frozen=True)
class HubClient:
    """One Hub, with or without an API key."""

    url: str
    token: str | None = None

    @classmethod
    def from_env(cls, url: str | None = None) -> HubClient:
        base = url or os.environ.get(HUB_URL_ENV)
        if not base:
            error = HubError("no Hub named")
            error.exit_code = 2
            error.hint = f"pass --hub <url> or set {HUB_URL_ENV}"
            raise error
        return cls(normalize_hub_url(base), os.environ.get(HUB_TOKEN_ENV) or None)

    # ---- requests ---------------------------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        accept: str = "application/json",
    ) -> Any:
        headers = {"Accept": accept, "User-Agent": "benchflow-hub-client"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(self.url + path, data=body, method=method, headers=headers)
        try:
            return _OPENER.open(request, timeout=_TIMEOUT_S)
        except _DowngradeRefused as exc:
            raise HubError(
                f"{self.url} {exc.msg} for {method} {path}",
                hint="use the Hub's HTTPS address; BenchFlow never sends a key over a downgrade",
            ) from None
        except HTTPError as exc:
            detail = _detail(exc)
            hint = None
            if exc.code in (401, 403):
                hint = f"set {HUB_TOKEN_ENV} to an API key of this Hub with the scope this needs"
            raise HubError(
                f"{self.url} answered {exc.code} for {method} {path}: {detail}",
                status=exc.code,
                hint=hint,
            ) from None
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise HubError(
                f"could not reach the Hub at {self.url}: {reason}",
                hint="check the address and this machine's network",
            ) from None

    def get_json(self, path: str) -> Any:
        with self._request("GET", path) as response:
            raw = response.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HubError(f"{self.url}{path} did not answer JSON: {exc}") from None

    def post_multipart(
        self, path: str, fields: dict[str, str], files: Iterable[tuple[str, str, Path]]
    ) -> tuple[int, Any]:
        """POST ``fields`` and ``files`` (form name, file name, path) as multipart/form-data; (status, JSON)."""
        boundary = "benchflow-" + uuid.uuid4().hex
        body = bytearray()
        for name, value in fields.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            body += value.encode() + b"\r\n"
        for name, filename, file_path in files:
            body += (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                "Content-Type: application/gzip\r\n\r\n"
            ).encode()
            body += file_path.read_bytes() + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        with self._request(
            "POST",
            path,
            body=bytes(body),
            content_type=f"multipart/form-data; boundary={boundary}",
        ) as response:
            raw = response.read()
            status = response.status
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HubError(f"{self.url}{path} did not answer JSON: {exc}") from None

    def download(self, path: str, dest: Path, *, max_bytes: int) -> None:
        """Stream ``path`` into ``dest`` (refusing more than ``max_bytes``)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with (
            self._request("GET", path, accept="application/octet-stream") as response,
            open(dest, "wb") as out,
        ):
            while chunk := response.read(_CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HubError(
                        f"{self.url}{path} sent more than the {max_bytes} bytes the manifest allows"
                    )
                out.write(chunk)

    # ---- what this Hub serves --------------------------------------------------------------------------
    def has_registry(self) -> bool | None:
        """Whether this Hub serves the dataset registry, as its own route list says: True or False, or None when
        it gives no list bench can read (another kind of server, a proxy's page, an outage)."""
        try:
            doc = self.get_json(ROUTE_LIST_PATH)
        except HubError:
            return None
        paths = doc.get("paths") if isinstance(doc, dict) else None
        if not isinstance(paths, dict):
            return None
        return any(str(path).startswith(REGISTRY_PATH) for path in paths)

    def _registry_failure(self, exc: HubError, method: str, path: str) -> HubError:
        """``exc``, from a registry route, said for what it is. A Hub without the registry answers 404 or 405
        to every registry request, so "check the name and version" would send the reader the wrong way."""
        if exc.status not in (404, 405):
            return exc
        has = self.has_registry()
        if has:
            return exc
        if has is False:
            error = HubError(
                f"{self.url} has no dataset registry: it answered {exc.status} to {method} {path}, "
                f"and its route list has no {REGISTRY_PATH}",
                status=exc.status,
                hint=(
                    "this Hub can't publish, show or serve datasets by name; run the tasks from a "
                    "folder (bench run <tasks-dir>), or name a Hub that has the dataset registry"
                ),
                kind=NO_REGISTRY,
            )
            error.exit_code = 2
            return error
        exc.hint = exc.hint or (
            "check the name and version, or that this Hub has the dataset registry: a Hub "
            f"without the dataset registry answers {exc.status} here too, and this one gave "
            "no route list to tell"
            if method == "GET"
            else f"this Hub may have no dataset registry: it answered {exc.status} to {method} "
            f"{path} and gave no route list to tell; name a Hub that has the dataset registry"
        )
        return exc

    # ---- the registry -------------------------------------------------------------------------------------
    def _registry_get(self, path: str) -> Any:
        try:
            return self.get_json(path)
        except HubError as exc:
            raise self._registry_failure(exc, "GET", path) from None

    def dataset(self, name: str, version: str) -> dict[str, Any]:
        org, short = split_dataset_name(name)
        doc = self._registry_get(
            f"{REGISTRY_PATH}/{quote(org)}/{quote(short)}/{quote(version)}"
        )
        if not isinstance(doc, dict) or doc.get("schema") != DATASET_SCHEMA:
            raise HubError(
                f"{self.url} answered a dataset document this bench does not read (schema {doc.get('schema') if isinstance(doc, dict) else None!r})"
            )
        return doc

    def versions(self, name: str) -> dict[str, Any]:
        org, short = split_dataset_name(name)
        return self._registry_get(f"{REGISTRY_PATH}/{quote(org)}/{quote(short)}")

    def controls(self, name: str, version: str) -> dict[str, Any]:
        org, short = split_dataset_name(name)
        return self._registry_get(
            f"{REGISTRY_PATH}/{quote(org)}/{quote(short)}/{quote(version)}/controls"
        )

    def publish(
        self, manifest: dict[str, Any], archives: Iterable[tuple[str, Path]]
    ) -> tuple[bool, dict[str, Any]]:
        """Publish a version; returns (created, the Hub's document of it). 201 is a new version; 200 means the
        same content was already published and nothing changed."""
        try:
            status, doc = self.post_multipart(
                REGISTRY_PATH,
                {"manifest": json.dumps(manifest)},
                [("files", f"{task}.tar.gz", path) for task, path in archives],
            )
        except HubError as exc:
            raise self._registry_failure(exc, "POST", REGISTRY_PATH) from None
        return status == 201, doc

    def run(self, run_id: str) -> dict[str, Any]:
        """A stored run as the Hub reports it (needs the ``read`` scope)."""
        doc = self.get_json(f"/api/v1/runs/{quote(run_id)}")
        if not isinstance(doc, dict):
            raise HubError(
                f"{self.url} answered run {run_id} with something other than a run",
                hint="check that --hub names a BenchFlow Hub, not a proxy or another server",
            )
        return doc

    def upload_run(
        self, archive: Path, *, name: str | None = None, dataset: str | None = None
    ) -> dict[str, Any]:
        fields = {k: v for k, v in {"name": name, "dataset": dataset}.items() if v}
        _status, answer = self.post_multipart(
            "/api/v1/runs/upload", fields, [("file", archive.name, archive)]
        )
        return answer


def _detail(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read() or b"{}")
    except (OSError, json.JSONDecodeError):
        return exc.reason or ""
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return str(detail if detail is not None else payload)[:500]
