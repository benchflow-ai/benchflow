"""Binding selectors are typed once, strictly.

Every generated selector (canonical and non-canonical paths, relative paths,
literal-backslash names, URL spellings with backslashes, mixed case, ports,
userinfo, queries and other schemes) is either refused, with the canonical
spelling in the message, or typed exactly as an independent reference here
computes it.
"""

from __future__ import annotations

import posixpath
import random

import pytest

from benchflow.integrity.selectors import SelectorError, parse_selector

SEED = 20261021
CASES = 6000

_NAMES = ("a", "app", "rules", "g\\h", "\\", "-i", "data\n", "j\udcff", "A b", "\t")
_SPECIAL = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _reference_path(text: str) -> tuple[str | None, str | None]:
    """``(typed, suggestion)``: the canonical path, or why not and what to write."""
    if "\0" in text or not text.startswith("/"):
        return None, None
    if text == "/":
        return text, None
    parts = text.split("/")[1:]
    if all(part not in {"", ".", ".."} for part in parts):
        return text, None
    return None, "/" + posixpath.normpath(text).lstrip("/")


# URL parts and what the URL Standard makes of each (None: the selector is
# refused). Selectors allow http, https, ws and wss; ftp, file and others are
# refused by policy, and userinfo, a query or a fragment cannot be a prefix.
_URL_SCHEMES = {
    "http": "http",
    "HTTP": "http",
    "Https": "https",
    "ws": "ws",
    "WSS": "wss",
    "ftp": None,
    "file": None,
    "foo": None,
}
_URL_SEPS = ("://", ":\\\\", ":/\\", ":", ":///", ":\\")
_URL_HOSTS = {
    "localhost": "localhost",
    "LocalHost": "localhost",
    "127.0.0.1": "127.0.0.1",
    "0x7f.1": "127.0.0.1",
    "[::1]": "[::1]",
    "[0:0::1]": "[::1]",
    "u@h": None,
    "-x": "-x",
    "a_b": "a_b",
    "grader.example.": "grader.example.",
}
_URL_PORTS = {
    "": "",
    ":9005": ":9005",
    ":80": "80",
    ":443": "443",
    ":0": ":0",
    ":x": None,
    ":": "",
}
_URL_PATHS = {
    "": "",
    "/_admin": "/_admin",
    "\\_admin": "/_admin",
    "/_Admin/state": "/_Admin/state",
    "\\_admin\\state": "/_admin/state",
    "/a/../b": "/b",
    "/a/./b": "/a/b",
    "?q": None,
    "#f": None,
    "/": "",
}
_DEFAULTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _url_case(rng: random.Random) -> tuple[str, str | None]:
    """A generated route selector and its expected typed form (or None)."""
    scheme, sep = rng.choice(list(_URL_SCHEMES)), rng.choice(_URL_SEPS)
    host, port, path = (
        rng.choice(list(_URL_HOSTS)),
        rng.choice(list(_URL_PORTS)),
        rng.choice(list(_URL_PATHS)),
    )
    text = f"{scheme}{sep}{host}{port}{path}"
    parts = (_URL_SCHEMES[scheme], _URL_HOSTS[host], _URL_PORTS[port], _URL_PATHS[path])
    if any(part is None for part in parts):
        return text, None
    s, h, p, q = parts
    if p in ("80", "443"):
        p = "" if int(p) == _DEFAULTS[s] else f":{p}"
    return text, f"{s}://{h}{p}{q}"


def _path_selector(rng: random.Random) -> str:
    parts = [rng.choice(_NAMES) for _ in range(rng.randint(1, 3))]
    if rng.random() < 0.4:
        parts.insert(rng.randrange(len(parts) + 1), rng.choice(("", ".", "..")))
    text = "/".join(parts)
    if rng.random() < 0.85:
        text = "/" + text
    if rng.random() < 0.15:
        text += "/"
    return text


def test_generated_selectors_are_typed_exactly_or_refused():
    rng = random.Random(SEED)
    typed_paths = typed_urls = refused = 0
    for _ in range(CASES):
        if rng.random() < 0.4:
            text, url = _url_case(rng)
        else:
            text, url = _path_selector(rng), None
        path, suggestion = _reference_path(text)
        try:
            typed = parse_selector(text)
        except SelectorError as exc:
            refused += 1
            assert url is None and path is None, (text, str(exc))
            if suggestion is not None:
                assert repr(suggestion) in str(exc), (text, str(exc))
            continue
        if typed.kind == "service":
            typed_urls += 1
            assert typed.value == url, (text, typed)
        else:
            typed_paths += 1
            assert typed.value == path == text, (text, typed)
    assert typed_paths > 500 and typed_urls > 200 and refused > 1000


@pytest.mark.parametrize(
    ("text", "kind", "value"),
    [
        ("http:\\\\localhost:9005\\_admin", "service", "http://localhost:9005/_admin"),
        ("HTTP://LocalHost:9005/_Admin/", "service", "http://localhost:9005/_Admin"),
        ("https://localhost:443/x", "service", "https://localhost/x"),
        ("http://localhost:80/_admin", "service", "http://localhost/_admin"),
        (
            "http://grading_service:9005/_admin",
            "service",
            "http://grading_service:9005/_admin",
        ),
        (
            "http://grader.example.:9005/_admin",
            "service",
            "http://grader.example.:9005/_admin",
        ),
        ("/shared/g\\h", "path", "/shared/g\\h"),
        ("/App/Rules", "path", "/App/Rules"),
        ("/", "path", "/"),
    ],
)
def test_named_selectors_are_typed(text, kind, value):
    typed = parse_selector(text)
    assert (typed.kind, typed.value) == (kind, value)


@pytest.mark.parametrize(
    ("text", "canonical"),
    [
        ("/./app/rules", "/app/rules"),
        ("/app//rules", "/app/rules"),
        ("/app/rules/", "/app/rules"),
        ("/app/x/../rules", "/app/rules"),
        ("app/rules", None),
        ("C:\\rules", None),
        ("file:///app/rules", None),
        ("http://u@localhost/x", None),
        ("http://localhost/x?y", None),
        ("http://:9005/x", None),
        ("http://\u00e9x.test/x", None),
    ],
)
def test_non_canonical_selectors_are_refused_with_their_canonical_spelling(
    text, canonical
):
    with pytest.raises(SelectorError) as exc:
        parse_selector(text)
    if canonical is not None:
        assert repr(canonical) in str(exc.value)
