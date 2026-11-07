"""Host proxy start-up with route analysis and model builds deferred to first use.

``litellm_lazy_start`` (run by the host proxy's sitecustomize before LiteLLM is
imported) leaves FastAPI's per-route analysis and pydantic's model builds to
their first use: the same code, later, so the proxy answers exactly as before
while starting about 2 s sooner (PERF-A).
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import benchflow.providers.litellm_lazy_start as lazy
import benchflow.providers.litellm_runtime as runtime_mod


def _checked_versions() -> bool:
    import fastapi
    import pydantic

    return (
        lazy._litellm_checked()
        and fastapi.__version__ == lazy.FASTAPI_VERSION
        and pydantic.VERSION == lazy.PYDANTIC_VERSION
    )


# The deferrals apply only to the versions they were checked against; with any
# other lock they stay off and the proxy starts as LiteLLM does, which the other
# tests here cover.
needs_checked_versions = pytest.mark.skipif(
    not _checked_versions(),
    reason=f"the lazy start applies only to LiteLLM {lazy.LITELLM_VERSION}, "
    f"FastAPI {lazy.FASTAPI_VERSION} and pydantic {lazy.PYDANTIC_VERSION}",
)


@pytest.fixture
def deferred():
    """The deferrals in place for one test, and gone after it."""
    lazy.install()
    try:
        yield
    finally:
        lazy.uninstall()


def _app() -> tuple[Any, list[str]]:
    """A FastAPI app with the shapes LiteLLM's routes have: path, query, header
    and body parameters, router and route dependencies, a response model, an
    included router with a prefix, and routes nobody requests."""
    from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str
        size: int = 1

    class Saved(BaseModel):
        id: int
        name: str
        size: int
        key: str

    calls: list[str] = []

    def key(authorization: str = Header()) -> str:
        calls.append("key")
        if authorization != "Bearer ok":
            raise HTTPException(status_code=401, detail="bad key")
        return authorization.split()[1]

    def audit() -> None:
        calls.append("audit")

    router = APIRouter(dependencies=[Depends(audit)])

    @router.post("/items/{item_id}", response_model=Saved, status_code=201)
    async def save(
        item_id: int, item: Item, verbose: bool = False, k: str = Depends(key)
    ) -> dict[str, Any]:
        return {
            "id": item_id,
            "name": item.name,
            "size": item.size,
            "key": k,
            "v": verbose,
        }

    @router.get("/never")
    async def never() -> dict[str, Any]:
        return {}

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(router, prefix="/v1")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    return app, calls


_REQUESTS = [
    ("POST", "/v1/items/7?verbose=true", {"authorization": "Bearer ok"}, {"name": "a"}),
    ("POST", "/v1/items/7", {"authorization": "Bearer no"}, {"name": "a"}),
    ("POST", "/v1/items/x", {"authorization": "Bearer ok"}, {"name": "a"}),
    ("POST", "/v1/items/7", {"authorization": "Bearer ok"}, {"size": "big"}),
    ("POST", "/v1/items/7", {}, {"name": "a"}),
    ("GET", "/v1/items/7", {}, None),
    ("GET", "/health", {}, None),
    ("GET", "/missing", {}, None),
]


def _answers(app: Any) -> list[tuple[int, Any]]:
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        return [
            (r.status_code, r.json())
            for method, path, headers, body in _REQUESTS
            for r in [client.request(method, path, headers=headers, json=body)]
        ]


def _api_routes(app: Any) -> list[Any]:
    from fastapi.routing import APIRoute

    return [r for r in app.routes if isinstance(r, APIRoute)]


@needs_checked_versions
def test_a_deferred_route_answers_as_before_and_is_analysed_only_when_used(deferred):
    lazy.uninstall()
    eager_app, eager_calls = _app()
    eager = _answers(eager_app)
    lazy.install()
    lazy_app, lazy_calls = _app()
    routes = {r.path: r for r in _api_routes(lazy_app)}
    assert all(lazy._DEFERRED in r.__dict__ for r in routes.values())

    assert _answers(lazy_app) == eager
    assert lazy_calls == eager_calls
    assert eager[0] == (201, {"id": 7, "name": "a", "size": 1, "key": "ok"})
    # Requested routes were analysed; the one nobody asked for never was.
    assert lazy._DEFERRED not in routes["/v1/items/{item_id}"].__dict__
    assert lazy._DEFERRED not in routes["/health"].__dict__
    assert lazy._DEFERRED in routes["/v1/never"].__dict__


@needs_checked_versions
def test_an_included_route_gets_what_the_eager_one_would_pass(deferred):
    """include_router copies a route from the fields it reads; a deferred route
    has them all, as FastAPI's own __init__ computed them."""
    lazy.uninstall()
    eager_app, _ = _app()
    lazy.install()
    lazy_app, _ = _app()
    fields = (
        "path",
        "methods",
        "name",
        "unique_id",
        "status_code",
        "response_model",
        "tags",
        "summary",
        "description",
        "response_description",
        "deprecated",
        "operation_id",
        "include_in_schema",
        "responses",
        "path_format",
    )

    def view(app: Any) -> list[tuple[Any, ...]]:
        # A class by its name: each _app() defines its models anew.
        return [
            (
                *(
                    getattr(getattr(r, f), "__qualname__", getattr(r, f))
                    for f in fields
                ),
                len(r.dependencies),
            )
            for r in _api_routes(app)
        ]

    deferred_view = view(lazy_app)
    assert all(lazy._DEFERRED in r.__dict__ for r in _api_routes(lazy_app))
    assert deferred_view == view(eager_app)


def test_only_the_checked_versions_are_deferred(monkeypatch):
    import fastapi
    import pydantic
    from fastapi import routing

    before = routing.APIRoute.__init__
    monkeypatch.setattr(fastapi, "__version__", "0.0.0")
    monkeypatch.setattr(pydantic, "VERSION", "1.10.0")
    monkeypatch.setattr(lazy, "LITELLM_VERSION", "0.0.1")
    try:
        lazy.install()
        assert lazy.installed() == ()
        assert routing.APIRoute.__init__ is before
    finally:
        lazy.uninstall()


@needs_checked_versions
def test_install_twice_and_uninstall(monkeypatch):
    """Guards 764325430: a dependency upgrade retains the version refusal."""
    import pydantic
    from fastapi import routing
    from pydantic._internal import _config

    before = routing.APIRoute.__init__
    before_defer_build = _config.config_defaults["defer_build"]
    lazy.install()
    lazy.install()
    try:
        # The tokenizer only where LiteLLM has not loaded its own yet.
        assert "routes" in lazy.installed()
        if pydantic.VERSION == lazy.PYDANTIC_VERSION:
            assert "models" in lazy.installed()
            assert _config.config_defaults["defer_build"] is True
        else:
            assert "models" not in lazy.installed()
            assert _config.config_defaults["defer_build"] is before_defer_build
    finally:
        lazy.uninstall()
    assert lazy.installed() == ()
    assert routing.APIRoute.__init__ is before
    assert _config.config_defaults["defer_build"] is before_defer_build


def test_a_model_defined_after_install_validates_the_same(deferred):
    from pydantic import BaseModel, ValidationError

    def model() -> type[BaseModel]:
        class Usage(BaseModel):
            prompt_tokens: int
            completion_tokens: int = 0

        return Usage

    lazy_model = model()
    lazy.uninstall()
    eager_model = model()
    lazy.install()

    good = {"prompt_tokens": "3"}
    assert lazy_model(**good).model_dump() == eager_model(**good).model_dump()
    errors = []
    for cls in (lazy_model, eager_model):
        with pytest.raises(ValidationError) as caught:
            cls(prompt_tokens="many")
        errors.append(caught.value.errors(include_url=False))
    assert errors[0] == errors[1]


@needs_checked_versions
def test_a_route_whose_analysis_fails_fails_at_its_first_use_and_every_use(deferred):
    """The one difference: an invalid route fails when it is first used instead
    of when it is registered (LiteLLM's routes are all valid)."""
    from fastapi import Depends, FastAPI

    class NoSignature:
        @property
        def __signature__(self):
            raise RuntimeError("no signature")

        def __call__(self) -> int:
            return 1

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/x", dependencies=[Depends(NoSignature())])
    def x() -> dict[str, Any]:
        return {}

    (route,) = _api_routes(app)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="no signature"):
            route.app  # noqa: B018
    lazy.uninstall()
    with pytest.raises(RuntimeError, match="no signature"):
        app.get("/y", dependencies=[Depends(NoSignature())])(x)


def test_runtime_files_defer_only_when_asked(tmp_path):
    config = {"model_list": []}
    runtime_mod._write_runtime_files(tmp_path / "off", config=config)
    runtime_mod._write_runtime_files(tmp_path / "on", config=config, lazy_start=True)

    off = (tmp_path / "off" / "sitecustomize.py").read_text()
    on = (tmp_path / "on" / "sitecustomize.py").read_text()
    module = f"{runtime_mod._LAZY_START_MODULE}.py"
    assert not (tmp_path / "off" / module).exists()
    assert (tmp_path / "on" / module).read_text() == Path(lazy.__file__).read_text()
    # Before anything that imports LiteLLM.
    assert on == (
        f"import {runtime_mod._LAZY_START_MODULE}\n"
        f"{runtime_mod._LAZY_START_MODULE}.install()\n" + off
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected"), [(None, True), ("0", False), ("1", True)]
)
async def test_host_proxies_start_deferred_unless_turned_off(
    tmp_path, monkeypatch, value, expected
):
    from benchflow.providers.litellm_config import resolve_litellm_route

    if value is None:
        monkeypatch.delenv(runtime_mod.PROXY_LAZY_START_ENV, raising=False)
    else:
        monkeypatch.setenv(runtime_mod.PROXY_LAZY_START_ENV, value)
    seen: list[bool] = []

    class Stop(Exception):
        pass

    def write(runtime_dir, *, config, lazy_start=False):
        seen.append(lazy_start)
        raise Stop

    folder = tmp_path / "rt"
    folder.mkdir()
    monkeypatch.setattr(runtime_mod, "_write_runtime_files", write)
    monkeypatch.setattr(runtime_mod.tempfile, "mkdtemp", lambda **_: str(folder))
    env = {"OPENAI_API_KEY": "sk-test"}
    with pytest.raises(Stop):
        await runtime_mod._start_host_litellm(
            route=resolve_litellm_route("openai/gpt-4o-mini", env),
            master_key="sk-master",
            agent_env=env,
            environment="local",
            session_id="s",
            agent_name="lazy-start-test",
        )
    assert seen == [expected]


# --- LiteLLM's own proxy ----------------------------------------------------

_ROUTES_PROBE = r"""
import json, sys
from fastapi.routing import APIRoute
app = sys.modules["litellm.proxy.proxy_server"].app
routes = [r for r in app.routes if isinstance(r, APIRoute)]
deferred = sum(1 for r in routes if "_benchflow_deferred_route" in r.__dict__)

def name(x):
    x = getattr(x, "value", x)
    return getattr(x, "__qualname__", None) or type(x).__qualname__

def view(r):
    flat = r._flat_dependant
    params = sorted(
        (kind, p.name)
        for kind, group in (
            ("path", flat.path_params), ("query", flat.query_params),
            ("header", flat.header_params), ("cookie", flat.cookie_params),
            ("body", flat.body_params),
        )
        for p in group
    )
    return [
        r.path, sorted(r.methods), r.name, r.unique_id, r.status_code,
        name(r.response_model), name(r.response_class), [str(t) for t in r.tags],
        [name(d.call) for d in r.dependant.dependencies], params,
        r.body_field.name if r.body_field is not None else None,
        r.response_field is not None, sorted(map(str, r.response_fields)),
        r.is_json_stream, r.is_sse_stream,
    ]

import os
encoding_module = sys.modules["litellm.litellm_core_utils.default_encoding"]
encoding = encoding_module.encoding
tokenizer = {
    "type": type(encoding).__name__,
    "loaded_at_start": getattr(encoding, "_real", "eager") is not None,
    "cache_dir": os.environ.get("TIKTOKEN_CACHE_DIR"),
}
from litellm.litellm_core_utils.token_counter import default_encoding
tokenizer["count"] = len(default_encoding.encode("hello world, twice", disallowed_special=()))
tokenizer["same_object"] = default_encoding is encoding
print(json.dumps({"routes": len(routes), "deferred": deferred, "views": [view(r) for r in routes], "tokenizer": tokenizer}))
"""


def _litellm_routes(tmp_path: Path, *, lazy_start: bool) -> dict[str, Any]:
    folder = tmp_path / ("lazy" if lazy_start else "eager")
    runtime_mod._write_runtime_files(
        folder, config={"model_list": []}, lazy_start=lazy_start
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(folder),
        "PYTHONSAFEPATH": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_MASTER_KEY": "sk-master",
        **runtime_mod._PROXY_DOCS_DISABLE_ENV,
    }
    result = subprocess.run(
        [sys.executable, "-c", _ROUTES_PROBE],
        env=env,
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@needs_checked_versions
def test_litellms_routes_are_analysed_the_same_on_first_use(tmp_path):
    """Every route of LiteLLM's proxy, as BenchFlow's host proxy starts it, is
    deferred at start-up and, once analysed, the same as at an eager start."""
    eager = _litellm_routes(tmp_path, lazy_start=False)
    deferred = _litellm_routes(tmp_path, lazy_start=True)

    assert eager["routes"] > 100 and eager["deferred"] == 0
    assert deferred["deferred"] == deferred["routes"] == eager["routes"]
    assert deferred["views"] == eager["views"]
    # LiteLLM's tokenizer: not loaded at start-up, then the same counts and
    # the same TIKTOKEN_CACHE_DIR as the module that loads it at import.
    assert eager["tokenizer"]["type"] == "Encoding"
    assert deferred["tokenizer"]["type"] == "_LazyEncoding"
    assert not deferred["tokenizer"]["loaded_at_start"]
    assert deferred["tokenizer"]["same_object"]
    for key in ("count", "cache_dir"):
        assert deferred["tokenizer"][key] == eager["tokenizer"][key]


def _steady(value: Any) -> Any:
    """A record without the fields that differ between any two runs (clock
    readings, and the response id LiteLLM makes up for a Responses call)."""
    if isinstance(value, dict):
        return {
            k: _steady(v)
            for k, v in value.items()
            if k not in {"timestamp", "duration_ms", "created", "id"}
        }
    if isinstance(value, list):
        return [_steady(v) for v in value]
    return value


def test_the_proxy_answers_and_records_as_an_eager_start_does(tmp_path, monkeypatch):
    """The real host proxy in front of a mock server: chat, Messages and
    Responses calls, streamed or not, send the same upstream requests and
    record the same trajectory either way."""
    from tests.test_gateway_token_capture import _CASES, _drive_gateway

    runs = {}
    for mode in ("eager", "lazy"):
        if mode == "eager":
            monkeypatch.setenv(runtime_mod.PROXY_LAZY_START_ENV, "0")
        else:
            monkeypatch.delenv(runtime_mod.PROXY_LAZY_START_ENV)
        folder = tmp_path / mode
        folder.mkdir()
        runs[mode] = asyncio.run(
            _drive_gateway(folder, "vllm/mock-policy", list(_CASES))
        )

    (eager, eager_upstream), (lazy_records, lazy_upstream) = runs["eager"], runs["lazy"]
    assert sorted(lazy_records) == sorted(_CASES)
    assert _steady(lazy_records) == _steady(eager)
    assert lazy_upstream == eager_upstream


def test_the_tokenizer_loads_once_at_first_use_and_retries_like_litellm(monkeypatch):
    import tiktoken

    calls: list[str] = []
    real = SimpleNamespace(encode=lambda text, **_: [len(text)], name="cl100k_base")

    def get_encoding(name: str):
        calls.append(name)
        if len(calls) == 1:
            raise OSError("a cache file another process is writing")
        return real

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
    monkeypatch.setattr("time.sleep", lambda _: None)
    encoding = lazy._LazyEncoding()

    assert calls == []
    assert encoding.encode("abc") == [3]
    assert encoding.name == "cl100k_base"
    assert calls == ["cl100k_base", "cl100k_base"]


def test_the_tokenizer_is_deferred_only_for_the_checked_litellm(monkeypatch):
    import importlib.metadata

    monkeypatch.delitem(sys.modules, lazy._ENCODING_MODULE, raising=False)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "/before")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.1")
    lazy._defer_tokenizer()
    assert lazy._ENCODING_MODULE not in sys.modules
    assert "tokenizer" not in lazy.installed()

    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: lazy.LITELLM_VERSION
    )
    lazy._defer_tokenizer()
    try:
        module = sys.modules[lazy._ENCODING_MODULE]
        assert isinstance(module.encoding, lazy._LazyEncoding)
        assert os.environ["TIKTOKEN_CACHE_DIR"].endswith(
            os.path.join("litellm_core_utils", "tokenizers")
        )
    finally:
        lazy.uninstall()
    assert lazy._ENCODING_MODULE not in sys.modules
    assert os.environ["TIKTOKEN_CACHE_DIR"] == "/before"


@needs_checked_versions
def test_a_route_changed_before_its_first_use_refuses_to_complete(deferred):
    """The real __init__ sets every field again from the route's arguments:
    a dependency added or an endpoint swapped since would be lost, so the
    first use refuses instead."""
    from fastapi import Depends, FastAPI

    def build(path: str) -> Any:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.get(path)
        def endpoint() -> dict[str, Any]:
            return {}

        (route,) = _api_routes(app)
        return route

    added = build("/added")
    added.dependencies.append(Depends(lambda: None))
    with pytest.raises(
        RuntimeError, match=r"changed before its first use \(dependencies\)"
    ):
        added.app  # noqa: B018

    swapped = build("/swapped")
    swapped.endpoint = lambda: {"other": True}
    with pytest.raises(RuntimeError, match=r"\(endpoint\)"):
        swapped.app  # noqa: B018

    untouched = build("/untouched")
    assert untouched.app is not None


def test_nothing_is_deferred_for_another_litellm(monkeypatch):
    import importlib.metadata

    from fastapi import routing

    before = routing.APIRoute.__init__
    version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.0.1" if name == "litellm" else version(name),
    )
    monkeypatch.delitem(sys.modules, lazy._ENCODING_MODULE, raising=False)
    try:
        lazy.install()
        assert lazy.installed() == ()
        assert routing.APIRoute.__init__ is before
    finally:
        lazy.uninstall()


def test_the_first_load_puts_the_real_encoding_in_litellms_module(monkeypatch):
    import types

    import tiktoken

    real = SimpleNamespace(encode=lambda text, **_: [len(text)])
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: real)
    module = types.ModuleType(lazy._ENCODING_MODULE)
    stand_in = lazy._LazyEncoding()
    module.__dict__["encoding"] = stand_in
    monkeypatch.setitem(sys.modules, lazy._ENCODING_MODULE, module)

    assert stand_in.encode("abc") == [3]
    assert module.__dict__["encoding"] is real
