"""Binding selectors, parsed once at contract load into one typed form.

Rounds 10 and 11 traded regressions between rewriting a selector's spelling
(wrong bytes) and keeping it (containment and URL detection broke). A
selector is written by the task author, who is trusted, so a selector that is
not already canonical is an authoring mistake: it is refused with the
canonical spelling, never interpreted. Downstream (containment,
probe selection, closure matching) only the typed canonical form is used.

- **A path** must be a canonical absolute POSIX path: it starts with ``/``,
  has no empty (``//``, a trailing ``/``), ``.`` or ``..`` component and no
  NUL. Every other byte is a legal name byte and is kept exactly, a literal
  backslash included, and case is kept.
- **A URL** (http, https, ws, wss) is parsed by :mod:`benchflow.integrity.urls`,
  the URL Standard's parser that the emitter also applies to every observed
  URL, and typed as :meth:`~benchflow.integrity.urls.SpecialURL.key`: so
  ``http:\\\\host\\x`` is ``http://host/x``, a default port is dropped on
  both sides, and any host the Standard accepts (``grading_service``, a
  trailing dot) is kept. Userinfo, a query or a fragment cannot be part of a
  prefix and are refused; a non-ASCII host (IDNA) is refused as
  unsupported, and so is a URL the Standard rejects.
- Anything else (relative paths, other schemes, ``file:``) is refused.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Literal

from benchflow.integrity.urls import URLFailure, URLUnsupported, parse_special_url

SelectorKind = Literal["path", "service"]

_SCHEME_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*):")
_SPECIAL_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


class SelectorError(ValueError):
    """A selector that cannot be typed; the message says what to write."""


@dataclass(frozen=True)
class TypedSelector:
    kind: SelectorKind
    value: str


def parse_selector(text: str) -> TypedSelector:
    """``text`` as a canonical path or URL, or :class:`SelectorError`."""

    scheme = _SCHEME_RE.match(text)
    if scheme is not None and not text.startswith("/"):
        name = scheme.group(1).lower()
        if name in _SPECIAL_PORTS:
            return TypedSelector("service", _route(text))
        if name == "file":
            raise SelectorError(
                f"selector {text!r}: write a file: URL as its absolute path"
            )
        raise SelectorError(
            f"selector {text!r}: scheme {name!r} is not supported; use an "
            "absolute path or an http(s) or ws(s) URL"
        )
    return TypedSelector("path", canonical_path(text, what="selector"))


def canonical_path(text: str, *, what: str = "path") -> str:
    """``text`` when it is a canonical absolute POSIX path, else
    :class:`SelectorError` naming the canonical spelling."""

    if "\0" in text:
        raise SelectorError(f"{what} {text!r}: a path cannot contain NUL")
    if not text.startswith("/"):
        raise SelectorError(
            f"{what} {text!r}: not an absolute path; write it from the root, "
            f"as {suggest_path('/' + text)!r} if it is relative to /"
        )
    if text == "/":
        return text
    parts = text.split("/")[1:]
    if any(part in {"", ".", ".."} for part in parts):
        raise SelectorError(
            f"{what} {text!r}: not canonical (an empty, '.' or '..' component, "
            f"or a trailing '/'); write {suggest_path(text)!r}"
        )
    return text


def suggest_path(text: str) -> str:
    """The canonical spelling a non-canonical absolute path means lexically."""
    return "/" + posixpath.normpath(text).lstrip("/") if text.strip("/") else "/"


def _route(text: str) -> str:
    """A route selector's typed form: the URL Standard's parse (the same
    parser the emitter applies to observed URLs), serialized by
    :meth:`SpecialURL.key`."""
    try:
        url = parse_special_url(text)
    except URLUnsupported as exc:
        raise SelectorError(
            f"selector {text!r}: {exc}; write the host in ASCII"
        ) from None
    except URLFailure as exc:
        raise SelectorError(f"selector {text!r}: not a valid URL ({exc})") from None
    if url.has_userinfo or url.has_query or url.has_fragment:
        raise SelectorError(
            f"selector {text!r}: a route selector is a prefix; write it without "
            f"userinfo, query or fragment, as {url.key()!r}"
        )
    return url.key()


__all__ = [
    "SelectorError",
    "TypedSelector",
    "canonical_path",
    "parse_selector",
    "suggest_path",
]
