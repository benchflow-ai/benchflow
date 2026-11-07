"""Host proxy start-up: route analysis and model builds deferred to first use.

LiteLLM's proxy registers about 490 FastAPI routes when it is imported, and
FastAPI analyses each route's signature and dependencies as it registers it
(twice for a route on an included router). Pydantic likewise builds every
model's validator and serializer when the class is defined, and LiteLLM and
the OpenAI SDK define thousands of models. A BenchFlow proxy serves a handful
of those routes (``litellm_request_guard`` refuses every other path before
routing) and uses a fraction of the models, yet paid for all of them on every
start: about 2.1 s of the proxy's 4.1 s from spawn to healthy on the engine
bench's VM (PERF-A, 2026-10-19).

``install()``, which the proxy's sitecustomize runs before LiteLLM is
imported, defers both to first use:

- A route at once gets every field FastAPI's own ``APIRoute.__init__`` sets
  for matching and for ``include_router`` (path, methods, name, response
  model, tags, its declared dependencies ...), computed by that ``__init__``
  with its analysis steps left out. The first read of any other field (the
  handler, the dependency tree, the body field) runs the real ``__init__``,
  which sets every field again, the same way. A route nobody requests is never
  analysed.
- Pydantic models and type adapters default to ``defer_build``: the same
  build, at their first validation or serialization.
- LiteLLM's default tokenizer (tiktoken's cl100k_base, which
  ``litellm_core_utils.token_counter`` loads at import, about 0.1 s) is
  loaded at its first use, the way that module loads it, with
  ``TIKTOKEN_CACHE_DIR`` set as that module sets it.

Each runs the same code at a later time, so a request is handled exactly as
before. They apply only with the LiteLLM, FastAPI and pydantic versions they
were checked against (the lock's); with any other the proxy starts as before.
A route changed between its creation and its first use (LiteLLM 1.91.5 has
no such code) refuses that use rather than lose the change.
This file is copied into the proxy's private runtime, independent of BenchFlow.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

FASTAPI_VERSION = "0.136.3"
PYDANTIC_VERSION = "2.12.5"
LITELLM_VERSION = "1.91.5"
_ENCODING_MODULE = "litellm.litellm_core_utils.default_encoding"

# The fields APIRoute.__init__'s analysis steps set; every other field it sets
# is cheap and is set at once.
_ANALYSED = (
    "dependant",
    "_flat_dependant",
    "_embed_body_fields",
    "body_field",
    "is_sse_stream",
    "is_json_stream",
    "app",
    "response_field",
    "stream_item_field",
    "response_fields",
)
_DEFERRED = "_benchflow_deferred_route"
# The fields the cut-down __init__ set, as it set them: a later change to
# one would be lost when the real __init__ runs (refused instead).
_AS_MADE = "_benchflow_route_as_made"

# Held while fastapi.routing's analysis functions are swapped for stand-ins,
# and by every completion, so no route is ever completed with them in place.
_LOCK = threading.RLock()
_INSTALLED: dict[str, Any] = {}


def _stand_ins() -> dict[str, Any]:
    """fastapi.routing's analysis steps, as APIRoute.__init__ calls them."""

    def dependant(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            dependencies=[], is_async_gen_callable=False, is_gen_callable=False
        )

    def flat_dependant(*_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(body_params=[])

    def nothing(*_: Any, **__: Any) -> None:
        return None

    return {
        "get_dependant": dependant,
        "get_parameterless_sub_dependant": nothing,
        "get_flat_dependant": flat_dependant,
        "_should_embed_body_fields": lambda *_: False,
        "get_body_field": nothing,
        "create_model_field": nothing,
        "get_request_handler": nothing,
        "request_response": nothing,
    }


def install() -> None:
    """Defer each where the versions match; a second call does nothing."""
    _defer_route_analysis()
    _defer_model_builds()
    _defer_tokenizer()


def installed() -> tuple[str, ...]:
    """Which deferrals are in place: ``"routes"``, ``"models"``, ``"tokenizer"``."""
    return tuple(_INSTALLED)


# APIRoute's own __init__, saved the first time routes are deferred.
_ORIGINAL: dict[str, Any] = {}


def _route_init(self: Any, path: str, endpoint: Any, **kwargs: Any) -> None:
    from fastapi import routing

    stand_ins = _ORIGINAL["stand_ins"]
    with _LOCK:
        saved = {name: getattr(routing, name) for name in stand_ins}
        for name, stand_in in stand_ins.items():
            setattr(routing, name, stand_in)
        try:
            _ORIGINAL["init"](self, path, endpoint, **kwargs)
        finally:
            for name, value in saved.items():
                setattr(routing, name, value)
    for name in _ANALYSED:
        self.__dict__.pop(name, None)
    self.__dict__[_AS_MADE] = {
        field: (value, _contents(value)) for field, value in self.__dict__.items()
    }
    self.__dict__[_DEFERRED] = (path, endpoint, kwargs)


def _contents(value: Any) -> Any:
    """A copy of a list's, dict's or set's items, to see them changed in place."""
    if isinstance(value, (list, dict, set)):
        return type(value)(value)
    return None


def _route_getattr(self: Any, name: str) -> Any:
    # Python calls this only for a field the instance does not have.
    if _DEFERRED not in self.__dict__:
        raise AttributeError(name)
    with _LOCK:
        deferred = self.__dict__.get(_DEFERRED)
        if deferred is not None:
            # The real __init__ sets every field again from the route's
            # arguments: a field changed since would be lost. Refused, so a
            # LiteLLM that edits its routes fails loudly instead.
            made = self.__dict__.get(_AS_MADE, {})
            changed = sorted(
                field
                for field, (value, contents) in made.items()
                if self.__dict__.get(field, made) is not value
                or (contents is not None and _contents(value) != contents)
            )
            if changed:
                raise RuntimeError(
                    f"FastAPI route {deferred[0]!r} was changed before its first "
                    f"use ({', '.join(changed)}), which its deferred analysis "
                    "would lose; start the proxy with BENCHFLOW_PROXY_LAZY_START=0"
                )
            del self.__dict__[_DEFERRED]
            self.__dict__.pop(_AS_MADE, None)
            path, endpoint, kwargs = deferred
            try:
                _ORIGINAL["init"](self, path, endpoint, **kwargs)
            except BaseException:
                # As before this try (the failed __init__ set some fields
                # anew), so the next use fails the same way.
                for field in _ANALYSED:
                    self.__dict__.pop(field, None)
                for field, (value, _) in made.items():
                    self.__dict__[field] = value
                self.__dict__[_DEFERRED] = deferred
                self.__dict__[_AS_MADE] = made
                raise
    return object.__getattribute__(self, name)


def _litellm_checked() -> bool:
    """Whether the installed LiteLLM is the one these deferrals were checked
    with (its routes, models and tokenizer module)."""
    import importlib.metadata

    try:
        return importlib.metadata.version("litellm") == LITELLM_VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


def _defer_route_analysis() -> None:
    import fastapi
    from fastapi import routing

    if (
        "routes" in _INSTALLED
        or fastapi.__version__ != FASTAPI_VERSION
        or not _litellm_checked()
    ):
        return
    route = routing.APIRoute
    if route.__dict__.get("__getattr__", _route_getattr) is not _route_getattr:
        return
    _ORIGINAL.setdefault("init", route.__init__)
    _ORIGINAL.setdefault("stand_ins", _stand_ins())
    _INSTALLED["routes"] = True
    route.__init__ = _route_init  # ty: ignore[invalid-assignment]
    route.__getattr__ = _route_getattr  # ty: ignore[unresolved-attribute]


def _defer_model_builds() -> None:
    import pydantic
    from pydantic._internal import _config

    if (
        "models" in _INSTALLED
        or pydantic.VERSION != PYDANTIC_VERSION
        or not _litellm_checked()
    ):
        return
    _INSTALLED["models"] = _config.config_defaults.get("defer_build", False)
    _config.config_defaults["defer_build"] = True


class _LazyEncoding:
    """tiktoken's cl100k_base, loaded at its first use as LiteLLM's
    ``default_encoding`` module loads it (retrying a cache file another
    process is writing).

    It forwards every attribute, so calls behave as on the real ``Encoding``,
    but it is not one: ``isinstance(x, tiktoken.Encoding)``, pickling or C
    code given it would differ. LiteLLM 1.91.5 (the version these deferrals
    are checked with) only calls methods on it. The first load also puts the
    real ``Encoding`` in the module, so code importing it later gets the real
    object; only what imported it at start-up (``token_counter``) keeps this
    stand-in.
    """

    __slots__ = ("_real",)

    def __init__(self) -> None:
        object.__setattr__(self, "_real", None)

    def _load(self) -> Any:
        real = object.__getattribute__(self, "_real")
        if real is not None:
            return real
        with _LOCK:
            real = object.__getattribute__(self, "_real")
            if real is None:
                import random
                import time

                import tiktoken

                for attempt in range(5):
                    try:
                        real = tiktoken.get_encoding("cl100k_base")
                        break
                    except (FileExistsError, OSError):
                        if attempt == 4:
                            raise
                        time.sleep(0.1 * (2**attempt) + random.uniform(0, 0.1))
                object.__setattr__(self, "_real", real)
                import sys

                module = sys.modules.get(_ENCODING_MODULE)
                if module is not None and module.__dict__.get("encoding") is self:
                    module.__dict__["encoding"] = real
        return real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


def _defer_tokenizer() -> None:
    import importlib.util
    import os
    import sys
    import types

    if (
        "tokenizer" in _INSTALLED
        or _ENCODING_MODULE in sys.modules
        or not _litellm_checked()
    ):
        return
    spec = importlib.util.find_spec("litellm")
    if spec is None or not spec.submodule_search_locations:
        return
    package = next(iter(spec.submodule_search_locations))
    # What the module does when it is imported, before its encoding.
    custom = os.getenv("CUSTOM_TIKTOKEN_CACHE_DIR")
    if custom:
        os.makedirs(custom, exist_ok=True)
    _INSTALLED["tokenizer"] = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = custom or os.path.join(
        package, "litellm_core_utils", "tokenizers"
    )
    module = types.ModuleType(_ENCODING_MODULE)
    module.__dict__["encoding"] = _LazyEncoding()
    sys.modules[_ENCODING_MODULE] = module


def uninstall() -> None:
    """Undo ``install()`` (tests). A route made while it was in place still
    completes on its first read."""
    from fastapi import routing
    from pydantic._internal import _config

    if _INSTALLED.pop("routes", None):
        routing.APIRoute.__init__ = _ORIGINAL["init"]
    if "models" in _INSTALLED:
        _config.config_defaults["defer_build"] = _INSTALLED.pop("models")
    if "tokenizer" in _INSTALLED:
        import os
        import sys

        before = _INSTALLED.pop("tokenizer")
        if isinstance(
            getattr(sys.modules.get(_ENCODING_MODULE), "encoding", None), _LazyEncoding
        ):
            del sys.modules[_ENCODING_MODULE]
        if before is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = before
