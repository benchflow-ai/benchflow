"""Egress policy shared by the denylist proxy and the model gateway guard. Stdlib only.

``Policy`` decides which URLs a denylist task keeps out of the agent's reach: the
listed pages, any request that names one for another service to fetch (reader,
archive and translation services take the address in a path, query or body),
and the mirrors that serve a listed GitHub or Hugging Face repository by name.
``Log`` appends each refusal to the block log the enforcement points share.
Model requests that have the provider fetch the web are judged by
``benchflow.sandbox._hosted_fetch``.

The proxy and the gateway guard run inside the sandbox, where benchflow is not
installed, so benchflow ships this file there at its package path
(``benchflow/sandbox/_egress_policy.py``) next to them; both import it exactly
as they do in-process.
"""

from __future__ import annotations

import ipaddress
import json
import re
import sys
import threading
import unicodedata
import urllib.parse
from datetime import datetime, timezone


def host_key(host: str) -> str:
    """Comparison form of a hostname: lowercase, no trailing dot, no leading www."""
    host = host.strip().rstrip(".").lower()
    return host[4:] if host.startswith("www.") else host


def _unquoted(text: str) -> str:
    """``text`` with percent-encoding decoded, repeatedly (up to three layers)."""
    for _ in range(3):
        decoded = urllib.parse.unquote(text)
        if decoded == text:
            break
        text = decoded
    return text


def _path_key(path: str) -> str:
    """Comparison form of a path: decoded, dot segments resolved, one slash between segments."""
    path = _unquoted(path.split("?", 1)[0].split("#", 1)[0])
    path = path.replace("\\", "/")
    segments: list[str] = []
    for segment in path.split("/"):
        segment = segment.split(";", 1)[0]
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    key = "/" + "/".join(segments)
    if path.endswith(("/", "/.", "/..")) and key != "/":
        key += "/"
    return key.lower()


_EMBEDDED_IPV4 = re.compile(r"(?:^|[.-])(?:\d{1,3}[.-]){3}\d{1,3}(?:[.-]|$)")
_HEX_IPV4_LABEL = re.compile(r"^[0-9a-f]{8}$")
_WILDCARD_DNS = (
    "nip.io",
    "sslip.io",
    "xip.io",
    "traefik.me",
    "localtest.me",
    "lvh.me",
    "vcap.me",
)
# A hostname (a trailing dot allowed) and the path after it, written inside
# another URL's path or query.
_EMBEDDED_URL = re.compile(
    r"(?<![a-z0-9.-])((?:[a-z0-9-]+\.)+[a-z0-9-]+)\.?(?::[0-9]*)?(/[^?#&\s]*)?"
)
# What a URL parser reads differently from the text: it drops tab and newlines
# anywhere (WHATWG); IDNA drops stringprep table B.1 (soft hyphen, zero-width
# joiners, variation selectors); a lenient decoder drops what it cannot decode
# (U+FFFD); and ideographic and fullwidth full stops separate labels (UTS #46).
URL_IGNORED = frozenset(
    "\t\n\r\u00ad\u034f\u1806\u180b\u180c\u180d\u200b\u200c\u200d\u2060\ufeff\ufffd"
    + "".join(map(chr, range(0xFE00, 0xFE10)))
)
_URL_READING = str.maketrans(
    {**dict.fromkeys(URL_IGNORED), "\u3002": ".", "\uff0e": ".", "\uff61": "."}
)


def url_text(text: str) -> str:
    """``text`` as a URL parser reads an address in it: NFKC, dropped characters gone."""
    return unicodedata.normalize("NFKC", text.translate(_URL_READING)).translate(
        _URL_READING
    )


# Google Translate's site proxy serves a page under the origin's own name,
# dots as hyphens and hyphens doubled: arxiv-org.translate.goog/abs/<id>.
_TRANSLATE_PROXY = ".translate.goog"
# Services that serve a GitHub repository by its name, fetching it from GitHub
# on request: (host, path before "/<owner>/<repo>").
_GITHUB_MIRRORS = (
    ("cdn.jsdelivr.net", "/gh"),
    ("fastly.jsdelivr.net", "/gh"),
    ("gcore.jsdelivr.net", "/gh"),
    ("testingcf.jsdelivr.net", "/gh"),
    ("quantil.jsdelivr.net", "/gh"),
    ("originfastly.jsdelivr.net", "/gh"),
    ("cdn.statically.io", "/gh"),
    ("raw.githack.com", ""),
    ("rawcdn.githack.com", ""),
    ("raw.githubusercontent.com", ""),
    ("codeload.github.com", ""),
    ("api.github.com", "/repos"),
)
# Hugging Face repositories named in a path other than the model's own.
_HUB_KINDS = ("datasets", "spaces", "buckets")
# The Hugging Face dataset viewer names its dataset in the query.
_DATASET_VIEWER = "datasets-server.huggingface.co"


def _translated_origin(host: str) -> str | None:
    """The origin host a Google Translate proxy host serves, if ``host`` is one."""
    host = host_key(host)
    if not host.endswith(_TRANSLATE_PROXY):
        return None
    label = host[: -len(_TRANSLATE_PROXY)]
    return label.replace("--", "\0").replace("-", ".").replace("\0", "-")


def looks_like_address(host: str) -> bool:
    """True for anything that names an address rather than a site.

    glibc accepts decimal, octal, hex and short dotted forms (``3232235777``,
    ``0xc0a80101``, ``0300.0250.1.1``, ``127.1``), and wildcard DNS services
    resolve an address embedded in the name (``1-2-3-4.sslip.io``).
    """
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    labels = host.split(".")
    if not any(c.isalpha() for c in host):
        return True
    if labels[-1].isdigit():
        return True
    if any(label.startswith("0x") or _HEX_IPV4_LABEL.match(label) for label in labels):
        return True
    if _EMBEDDED_IPV4.search(host):
        return True
    return any(host == d or host.endswith("." + d) for d in _WILDCARD_DNS)


class Policy:
    """Match hosts and URLs against the denylist; scheme, port and query are ignored."""

    def __init__(
        self,
        blocked_urls: list[str],
        blocked_hosts: list[str],
        model_gateway_port: int | None = None,
    ):
        # Controller-supplied runtime state, never a task-authored allowlist.
        if model_gateway_port is not None and (
            type(model_gateway_port) is not int
            or not 1024 <= model_gateway_port <= 65535
        ):
            raise ValueError("invalid model gateway port")
        self.model_gateway_port = model_gateway_port
        self.prefixes: list[tuple[str, str]] = []
        for raw in blocked_urls:
            url = raw if "://" in raw else "https://" + raw
            parts = urllib.parse.urlsplit(url)
            if not parts.hostname:
                raise ValueError(f"blocked_urls entry has no host: {raw!r}")
            prefix = _path_key(parts.path).rstrip("/") or "/"
            self.prefixes.append((host_key(parts.hostname), prefix))
        self.prefixes += _mirrored(self.prefixes)
        self.hosts = {h.strip().rstrip(".").lower() for h in blocked_hosts if h.strip()}
        self.inspect_hosts = {host for host, _ in self.prefixes}
        self._paths: dict[str, list[str]] = {}
        for host, path in self.prefixes:
            self._paths.setdefault(host, []).append(path)

    @classmethod
    def load(cls, path: str) -> Policy:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            list(data.get("blocked_urls") or []),
            list(data.get("blocked_hosts") or []),
            data.get("model_gateway_port"),
        )

    def host_rule(self, host: str, port: int = 0) -> str | None:
        if (host, port) == ("127.0.0.1", self.model_gateway_port):
            return None
        name = host.strip().rstrip(".").lower()
        if looks_like_address(name):
            return "ip-literal"
        return self._listed_host(name)

    def url_rule(self, host: str, path: str, port: int = 0) -> str | None:
        """The rule ``host`` + ``path`` breaks, including a listed page named inside the path or query."""
        return (
            self.host_rule(host, port)
            or self._listed_prefix(host, path)
            or self.named_page_rule(path)
            or self._translated_rule(host, path)
            or self._dataset_viewer_rule(host, path)
        )

    def named_page_rule(self, text: str) -> str | None:
        """The listed page ``text`` names, as a service handed it would read the address.

        Reader services, web archives and redirectors fetch the address they
        are given (``r.jina.ai/https://arxiv.org/abs/2401.12345``, or a form
        field ``url=...``), so the page would reach the agent from a host the
        denylist does not name. The address is read decoded, as a URL parser
        reads it (``url_text``).
        """
        text = url_text(_unquoted(text)).replace("\\", "/").lower()
        for match in _EMBEDDED_URL.finditer(text):
            host, path = match.group(1), match.group(2) or "/"
            rule = self._listed_host(host) or self._listed_prefix(host, path)
            if rule:
                return rule
        return None

    def inspect(self, host: str) -> bool:
        return host_key(host) in self.inspect_hosts

    def _listed_host(self, name: str) -> str | None:
        for blocked in self.hosts:
            if name == blocked or name.endswith("." + blocked):
                return f"host:{blocked}"
        return None

    def _listed_prefix(self, host: str, path: str) -> str | None:
        key = host_key(host)
        paths = self._paths.get(key)
        if not paths:
            return None
        pkey = _path_key(path)
        return next((f"url:{key}{p}" for p in paths if pkey.startswith(p)), None)

    def _translated_rule(self, host: str, path: str) -> str | None:
        """Refuse a listed page served through Google Translate's site proxy."""
        origin = _translated_origin(host)
        if not origin:
            return None
        return self._listed_host(origin) or self._listed_prefix(origin, path)

    def _dataset_viewer_rule(self, host: str, path: str) -> str | None:
        """Refuse the dataset viewer's rows of a listed Hugging Face dataset."""
        if host_key(host) != _DATASET_VIEWER:
            return None
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
        return next(
            filter(
                None,
                (
                    self._listed_prefix(hub, f"/datasets/{name}")
                    for name in query.get("dataset", ())
                    for hub in ("huggingface.co", "hf.co")
                ),
            ),
            None,
        )


def _mirrored(prefixes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The mirrors of each listed GitHub repository or owner and Hugging Face repository.

    A GitHub listing of one page inside a repository stays a listing of that
    page. A Hugging Face listing is mirrored at the same path on hf-mirror.com
    and under the Hub API.
    """
    mirrored: list[tuple[str, str]] = []
    for host, path in prefixes:
        segments = path.strip("/").split("/")
        if not segments[0]:
            continue
        if host == "github.com" and len(segments) <= 2:
            mirrored += [(mirror, base + path) for mirror, base in _GITHUB_MIRRORS]
        elif host in ("huggingface.co", "hf.co") and segments[0] != "api":
            api = "/api" if segments[0] in _HUB_KINDS else "/api/models"
            mirrored += [("hf-mirror.com", path), ("huggingface.co", api + path)]
    return mirrored


class Log:
    """The block log: one JSON object per refused request, from the proxy and the gateway guard."""

    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()

    def write(self, **fields: object) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017
        line = json.dumps({"ts": stamp, **fields}, sort_keys=True)
        with self._lock:
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            else:
                print(line, file=sys.stderr, flush=True)
