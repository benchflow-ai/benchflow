"""Cua cloud (cyclops-cs claim API) sandbox provider unit tests.

Two layers:

* Unit tests mock the ``cua_train`` SDK and the authed httpx client it hands
  back, so the claim lifecycle (create -> bind -> service-ready -> release with
  no leak), the MCP handshake parse, and the honestly-unsupported shell-shaped
  ops are exercised without touching a real pool.
* One live dogfood test, gated on CUA_CLIENT_ID/CUA_CLIENT_SECRET/
  BENCHFLOW_CUA_CLOUD_POOL all being set, claims a real warm sandbox, runs the
  MCP handshake, releases it, and asserts the claim is gone — the provider's
  real no-leak lifecycle contract. It is a no-op in CI / here (no healthy pool).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from benchflow.sandbox import cua_cloud
from benchflow.sandbox.cua_cloud import CuaCloudSandbox, _parse_mcp_result
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.config import SandboxConfig, TaskOS

_CLAIM_PREFIX = "benchflow-"
_POOL = "test-pool"


# --------------------------------------------------------------------------
# Fakes: a scriptable authed httpx client + a TrainClient that hands it back
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_body: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttp:
    """Records requests and replays scripted responses.

    ``get_responses`` / ``post_responses`` are per-URL-suffix queues consumed in
    order; the bind/service-ready polls need different answers across calls.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []  # (method, url)
        self.deletes: list[str] = []
        self.create_status = 200
        self.bind_responses: list[_FakeResponse] = []
        self.service_responses: list[_FakeResponse] = []
        self.post_bodies: list[dict[str, Any]] = []
        self.mcp_responses: list[_FakeResponse] = []

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> _FakeResponse:
        self.requests.append(("POST", url))
        if json is not None:
            self.post_bodies.append(json)
        if url.endswith("/mcp"):
            return self.mcp_responses.pop(0)
        # claim create
        return _FakeResponse(status_code=self.create_status)

    def get(self, url: str) -> _FakeResponse:
        self.requests.append(("GET", url))
        if url.rstrip("/").endswith(f"-{cua_cloud._DRIVER_SERVICE}"):
            return self.service_responses.pop(0)
        if "/osgymsandboxclaims/" in url:
            return self.bind_responses.pop(0)
        return _FakeResponse()

    def delete(self, url: str) -> _FakeResponse:
        self.requests.append(("DELETE", url))
        self.deletes.append(url)
        return _FakeResponse(status_code=200)


class _FakeTrainClient:
    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, http: _FakeHttp) -> None:
        self._http = http

    @classmethod
    def from_key(cls, **kwargs: Any) -> _FakeTrainClient:
        cls.last_kwargs = kwargs
        return cls(http=_CURRENT_HTTP[0])

    def get_httpx_client(self) -> _FakeHttp:
        return self._http


# Bridge so the classmethod ``from_key`` can reach the per-test http fake.
_CURRENT_HTTP: list[_FakeHttp] = [_FakeHttp()]


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch) -> _FakeHttp:
    http = _FakeHttp()
    _CURRENT_HTTP[0] = http
    monkeypatch.setattr(cua_cloud, "_load_train_client", lambda: _FakeTrainClient)
    # No real sleeping in the polling loops.
    monkeypatch.setattr(cua_cloud.time, "sleep", lambda _s: None)
    return http


def _sandbox(tmp_path: Path, pool: str | None = _POOL) -> CuaCloudSandbox:
    return CuaCloudSandbox(
        environment_dir=tmp_path / "environment",
        environment_name="desktop-001",
        session_id="rollout-xyz",
        rollout_paths=None,
        task_env_config=SandboxConfig(os=TaskOS.LINUX),
        pool=pool,
    )


def _bound_body(sandbox: str = "sbx-42") -> _FakeResponse:
    return _FakeResponse(
        json_body={"status": {"phase": "Bound", "sandbox": {"name": sandbox}}}
    )


def _pending_body() -> _FakeResponse:
    return _FakeResponse(json_body={"status": {"phase": "Pending"}})


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def test_preflight_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> Any:
        raise RuntimeError("Missing optional dependency for the 'cua-cloud' sandbox.")

    monkeypatch.setattr(cua_cloud, "_load_train_client", _boom)
    with pytest.raises(RuntimeError, match="cua-cloud"):
        CuaCloudSandbox.preflight()


def test_preflight_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cua_cloud, "_load_train_client", lambda: _FakeTrainClient)
    monkeypatch.delenv("CUA_CLIENT_ID", raising=False)
    monkeypatch.delenv("CUA_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("BENCHFLOW_CUA_CLOUD_POOL", _POOL)
    with pytest.raises(SystemExit, match="CUA_CLIENT_ID and CUA_CLIENT_SECRET"):
        CuaCloudSandbox.preflight()


def test_preflight_missing_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cua_cloud, "_load_train_client", lambda: _FakeTrainClient)
    monkeypatch.setenv("CUA_CLIENT_ID", "ukey-abc")
    monkeypatch.setenv("CUA_CLIENT_SECRET", "secret")
    monkeypatch.delenv("BENCHFLOW_CUA_CLOUD_POOL", raising=False)
    with pytest.raises(SystemExit, match="BENCHFLOW_CUA_CLOUD_POOL"):
        CuaCloudSandbox.preflight()


def test_preflight_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cua_cloud, "_load_train_client", lambda: _FakeTrainClient)
    monkeypatch.setenv("CUA_CLIENT_ID", "ukey-abc")
    monkeypatch.setenv("CUA_CLIENT_SECRET", "secret")
    monkeypatch.setenv("BENCHFLOW_CUA_CLOUD_POOL", _POOL)
    CuaCloudSandbox.preflight()  # no raise


def test_pool_resolved_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BENCHFLOW_CUA_CLOUD_POOL", "env-pool")
    sandbox = _sandbox(tmp_path, pool=None)
    assert sandbox.pool == "env-pool"


# --------------------------------------------------------------------------
# start(): claim create body/URL, bind polling, service-ready polling
# --------------------------------------------------------------------------


def _seed_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUA_CLIENT_ID", "ukey-abc")
    monkeypatch.setenv("CUA_CLIENT_SECRET", "secret")


async def test_start_creates_claim_with_correct_url_and_body(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_pending_body(), _bound_body("sbx-42")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]

    sandbox = _sandbox(tmp_path)
    await sandbox.start()

    # Claim name is benchflow-owned and tracked for cleanup.
    assert sandbox.claim_name is not None
    assert sandbox.claim_name.startswith(_CLAIM_PREFIX)
    assert sandbox.sandbox_id == "sbx-42"

    # First POST is the claim create against the right /api/k8s URL.
    create_url = next(u for m, u in fake_http.requests if m == "POST")
    expected = (
        f"/api/k8s/apis/{cua_cloud._CLAIM_GROUP}/{cua_cloud._CLAIM_VERSION}"
        f"/namespaces/{_POOL}/{cua_cloud._CLAIM_PLURAL}"
    )
    assert create_url == expected

    body = fake_http.post_bodies[0]
    assert body["apiVersion"] == f"{cua_cloud._CLAIM_GROUP}/{cua_cloud._CLAIM_VERSION}"
    assert body["kind"] == "OSGymSandboxClaim"
    assert body["metadata"]["name"] == sandbox.claim_name
    assert body["spec"]["sandboxTemplateRef"]["name"] == f"{_POOL}-template"

    # TrainClient.from_key got the auth/base-url args.
    assert _FakeTrainClient.last_kwargs["client_id"] == "ukey-abc"
    assert _FakeTrainClient.last_kwargs["base_url"] == cua_cloud._DEFAULT_BASE_URL


async def test_start_polls_until_bound(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_pending_body(), _pending_body(), _bound_body("sbx-7")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]

    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    assert sandbox.sandbox_id == "sbx-7"
    # Three GETs against the claim before Bound.
    claim_gets = [
        u for m, u in fake_http.requests if m == "GET" and "/osgymsandboxclaims/" in u
    ]
    assert len(claim_gets) == 3


async def test_start_raises_on_failed_phase(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [
        _FakeResponse(json_body={"status": {"phase": "Failed", "reason": "nope"}})
    ]
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="failed to bind"):
        await sandbox.start()
    # Claim name was tracked before the bind, so cleanup can still release it.
    assert sandbox.claim_name is not None


async def test_start_bind_timeout(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    monkeypatch.setenv("BENCHFLOW_CUA_CLOUD_BIND_TIMEOUT_SEC", "0")
    fake_http.bind_responses = [_pending_body()]
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="did not reach 'Bound'"):
        await sandbox.start()


async def test_start_403_is_per_pool_key_hint(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.create_status = 403
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="per-USER key"):
        await sandbox.start()


async def test_start_polls_service_until_ready(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_bound_body("sbx-9")]
    fake_http.service_responses = [
        _FakeResponse(status_code=502),
        _FakeResponse(status_code=503),
        _FakeResponse(status_code=200),
    ]
    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    svc_gets = [
        u
        for m, u in fake_http.requests
        if m == "GET" and u.rstrip("/").endswith(f"-{cua_cloud._DRIVER_SERVICE}")
    ]
    assert len(svc_gets) == 3


async def test_start_service_ready_timeout(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    monkeypatch.setenv("BENCHFLOW_CUA_CLOUD_READY_TIMEOUT_SEC", "0")
    fake_http.bind_responses = [_bound_body("sbx-9")]
    fake_http.service_responses = [_FakeResponse(status_code=503)]
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="did not become ready"):
        await sandbox.start()


# --------------------------------------------------------------------------
# MCP handshake + capabilities
# --------------------------------------------------------------------------


async def test_mcp_handshake_parses_sse_tools(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_bound_body("sbx-1")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]
    sandbox = _sandbox(tmp_path)
    await sandbox.start()

    # initialize (sets session header), notifications/initialized, tools/list.
    fake_http.mcp_responses = [
        _FakeResponse(status_code=200, headers={"mcp-session-id": "sess-123"}),
        _FakeResponse(status_code=200),
        _FakeResponse(
            status_code=200,
            text='data: {"result": {"tools": [{"name": "click"}, {"name": "type"}]}}\n',
        ),
    ]
    result = sandbox.mcp_handshake()
    assert result["session_id"] == "sess-123"
    assert sorted(result["tools"]) == ["click", "type"]

    mcp_url = sandbox.driver_mcp_url()
    assert mcp_url == f"/api/svc/{_POOL}/sbx-1-{cua_cloud._DRIVER_SERVICE}/mcp"


async def test_capabilities_reports_control_plane(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_bound_body("sbx-2")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]
    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    fake_http.mcp_responses = [
        _FakeResponse(status_code=200, headers={"mcp-session-id": "s"}),
        _FakeResponse(status_code=200),
        _FakeResponse(
            status_code=200, text='{"result": {"tools": [{"name": "screenshot"}]}}'
        ),
    ]
    caps = await sandbox.capabilities()
    assert caps["provider"] == "cua-cloud"
    assert caps["control_plane"] == "cua-driver-mcp"
    assert caps["pool"] == _POOL
    assert caps["sandbox"] == "sbx-2"
    assert caps["tools"] == ["screenshot"]


def test_parse_mcp_result_handles_bare_json_and_sse() -> None:
    assert _parse_mcp_result('{"result": {"tools": []}}') == {"tools": []}
    assert _parse_mcp_result('data: {"result": {"x": 1}}\n') == {"x": 1}
    assert _parse_mcp_result("not json") == {}


# --------------------------------------------------------------------------
# stop(): DELETE the claim, no-leak tracking, idempotency
# --------------------------------------------------------------------------


async def test_stop_deletes_claim_no_leak(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_bound_body("sbx-5")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]
    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    claim_name = sandbox.claim_name
    assert claim_name is not None

    await sandbox.stop(delete=True)
    expected_url = (
        f"/api/k8s/apis/{cua_cloud._CLAIM_GROUP}/{cua_cloud._CLAIM_VERSION}"
        f"/namespaces/{_POOL}/{cua_cloud._CLAIM_PLURAL}/{claim_name}"
    )
    assert fake_http.deletes == [expected_url]
    # State cleared: nothing left to leak.
    assert sandbox.claim_name is None
    assert sandbox.sandbox_id is None


async def test_stop_releases_claim_even_when_bind_failed(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [
        _FakeResponse(json_body={"status": {"phase": "Failed"}})
    ]
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError):
        await sandbox.start()
    # The claim was created before the bind failed — stop() must release it.
    await sandbox.stop(delete=True)
    assert len(fake_http.deletes) == 1


async def test_stop_idempotent_when_not_started(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    await sandbox.stop(delete=True)  # no http client, no raise
    assert sandbox.claim_name is None


async def test_stop_delete_false_keeps_claim(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_bound_body("sbx-8")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]
    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    await sandbox.stop(delete=False)
    assert fake_http.deletes == []
    # Claim is intentionally retained so it can still be released later.
    assert sandbox.claim_name is not None


async def test_stop_swallows_delete_errors(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_bound_body("sbx-6")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]
    sandbox = _sandbox(tmp_path)
    await sandbox.start()

    def _raise(_url: str) -> Any:
        raise RuntimeError("network down")

    monkeypatch.setattr(fake_http, "delete", _raise)
    await sandbox.stop(delete=True)  # best-effort: must not raise
    assert sandbox.claim_name is None


# --------------------------------------------------------------------------
# honestly-unsupported shell-shaped ops
# --------------------------------------------------------------------------


async def test_exec_is_structured_unsupported(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_bound_body("sbx-3")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]
    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    with pytest.raises(NotImplementedError) as exc:
        await sandbox.exec("echo hi")
    msg = str(exc.value)
    assert "cua-driver MCP" in msg
    assert cua_cloud._SUPERSEDED_CMD_404 in msg


async def test_file_transfer_ops_unsupported(
    fake_http: _FakeHttp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_creds(monkeypatch)
    fake_http.bind_responses = [_bound_body("sbx-4")]
    fake_http.service_responses = [_FakeResponse(status_code=200)]
    sandbox = _sandbox(tmp_path)
    await sandbox.start()
    with pytest.raises(NotImplementedError, match="upload_file"):
        await sandbox.upload_file("/tmp/x", "/tmp/y")
    with pytest.raises(NotImplementedError, match="download_file"):
        await sandbox.download_file("/tmp/x", "/tmp/y")
    with pytest.raises(NotImplementedError, match="upload_dir"):
        await sandbox.upload_dir("/tmp/x", "/tmp/y")
    with pytest.raises(NotImplementedError, match="download_dir"):
        await sandbox.download_dir("/tmp/x", "/tmp/y")


async def test_snapshot_not_supported(tmp_path: Path) -> None:
    from benchflow.sandbox.protocol import SandboxSnapshotNotSupported

    sandbox = _sandbox(tmp_path)
    assert sandbox.supports_snapshot is False
    with pytest.raises(SandboxSnapshotNotSupported):
        await sandbox.snapshot()
    with pytest.raises(SandboxSnapshotNotSupported):
        await sandbox.restore(SimpleNamespace())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Live dogfood: real claim lifecycle, gated on creds + a reachable pool
# --------------------------------------------------------------------------


def _live_creds_set() -> bool:
    if not all(
        os.environ.get(name)
        for name in ("CUA_CLIENT_ID", "CUA_CLIENT_SECRET", "BENCHFLOW_CUA_CLOUD_POOL")
    ):
        return False
    try:
        import cua_train  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


@pytest.mark.skipif(
    not _live_creds_set(),
    reason=(
        "live cua-cloud dogfood requires CUA_CLIENT_ID/CUA_CLIENT_SECRET/"
        "BENCHFLOW_CUA_CLOUD_POOL set and the cua-train SDK installed against a "
        "reachable pool"
    ),
)
async def test_live_claim_lifecycle_no_leak(tmp_path: Path) -> None:
    """Real claim -> bind -> service-ready -> MCP handshake -> release; no leak."""
    sandbox = _sandbox(tmp_path, pool=os.environ["BENCHFLOW_CUA_CLOUD_POOL"])
    claim_name: str | None = None
    try:
        await sandbox.start()
        claim_name = sandbox.claim_name
        assert sandbox.sandbox_id is not None
        caps = await sandbox.capabilities()
        assert caps["control_plane"] == "cua-driver-mcp"
        assert isinstance(caps["tools"], list)
    finally:
        await sandbox.stop(delete=True)
    # After release the claim CR must be gone (no leaked warm sandbox).
    assert sandbox.claim_name is None
    assert claim_name is not None
