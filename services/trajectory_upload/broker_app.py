"""FastAPI surface for public, anonymous trajectory-upload handshakes."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.trajectory_upload.contract import UploadGrant, UploadRequest

MAX_HANDSHAKE_BYTES = 1024**2


class AlreadyUploaded(Exception):
    def __init__(self, *, base_url: str, prefix: str) -> None:
        self.base_url = base_url
        self.prefix = prefix


class RateLimited(Exception):
    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after


class RejectedUpload(Exception):
    """The same immutable capture digest already failed validation."""


class UploadBroker(Protocol):
    def create_upload(
        self, request: UploadRequest, *, client_ip: str
    ) -> UploadGrant: ...


def create_app(
    backend: UploadBroker | None = None,
    *,
    backend_factory: Callable[[], UploadBroker] | None = None,
) -> FastAPI:
    """Create the broker app, with an injectable backend for offline tests."""
    app = FastAPI(
        title="BenchFlow trajectory upload broker", docs_url=None, redoc_url=None
    )
    if backend is not None:
        app.state.backend = backend
    app.state.backend_factory = backend_factory or _azure_backend
    app.state.backend_lock = Lock()

    @app.middleware("http")
    async def limit_handshake_body(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/v1/uploads":
            content_length = request.headers.get("content-length")
            if content_length is None:
                return JSONResponse(
                    status_code=411,
                    content={"detail": "Content-Length is required"},
                )
            try:
                body_size = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "invalid Content-Length"},
                )
            if body_size < 0 or body_size > MAX_HANDSHAKE_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "upload handshake exceeds 1 MiB"},
                )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        oversized = any(
            error["type"] in {"less_than_equal", "too_long"} for error in errors
        )
        details = [
            {
                "type": str(error["type"]),
                "loc": [str(part) for part in error["loc"]],
                "msg": str(error["msg"]),
            }
            for error in errors
        ]
        return JSONResponse(
            status_code=413 if oversized else 400,
            content={"detail": details},
        )

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/uploads")
    def create_upload(request: Request, body: UploadRequest) -> JSONResponse:
        service = _backend(request.app)
        try:
            grant = service.create_upload(body, client_ip=_client_ip(request))
        except AlreadyUploaded as exc:
            return JSONResponse(
                status_code=409,
                content={"base_url": exc.base_url, "prefix": exc.prefix},
            )
        except RateLimited as exc:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(exc.retry_after)},
                content={"detail": "upload rate limit exceeded"},
            )
        except RejectedUpload:
            return JSONResponse(
                status_code=422,
                content={"detail": "trajectory capture was previously rejected"},
            )
        return JSONResponse(status_code=200, content=grant.as_dict())

    return app


def _backend(app: FastAPI) -> UploadBroker:
    if not hasattr(app.state, "backend"):
        with app.state.backend_lock:
            if not hasattr(app.state, "backend"):
                app.state.backend = app.state.backend_factory()
    return app.state.backend


def _azure_backend() -> UploadBroker:
    from services.trajectory_upload.azure_backend import AzureUploadBroker

    return AzureUploadBroker.from_env()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.rsplit(",", maxsplit=1)[-1].strip()
    return request.client.host if request.client is not None else "unknown"


app = create_app()
