"""Release leak gate: refuse a wheel, sdist or docs build that carries private material.

Usage:
    python tools/release_leak_gate.py --patterns P --allowlist A
        [--scanners DIR | --no-scanners] [--report-dir DIR] [--propose FILE]
        ARTIFACT...

ARTIFACT is a wheel (``.whl``), an sdist (``.tar.gz``), or ``LABEL=DIR`` for a
directory such as a docs build. Archives are unpacked recursively (zip, wheel,
Office files, tar, gzip) within fixed limits; a symlink, a path that leaves the
archive, a limit reached or an archive that does not open stops the gate.
Binaries are read as their printable strings (as ``strings -a``); a binary of
a kind with no reviewed reader is a finding of its own.

P is a JSON object, category to Python regex, of *candidate* patterns. Every
text line, member path and binary string is matched, and RULES below decide
which candidates are findings (``process`` matches the word "audit", but only a
review tag or the narration words are findings). A category the rules do not
know is strict: every match is a finding. An optional ``partner-code-tokens``
entry is no category: a partner match it matches whole is a code token (a
variable, an env var, a key prefix), not a finding. gitleaks and trufflehog,
pinned by tools/install_leak_scanners.sh, scan the same unpacked tree, and
each of their detections is a finding.

A finding passes only with an exception in A that binds its rule, its member
path, the SHA-256 of the matched value and of the member's bytes, with a
reason, an approval reference and an unexpired date, so a changed file needs a
new approval. P and A name what must not ship: they are passed in, never
committed. Output names the rule, member, line and a hash of the matched
value, never the value; ``--report-dir`` keeps the values, for a private copy.

Exit 0: no finding outside the allowlist. 1: findings. 2: the gate could not
decide (bad patterns or allowlist, an archive it would not unpack, a scanner
missing, of another version, or failing).
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import datetime as dt
import gzip
import hashlib
import html
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

GITLEAKS_VERSION = "8.30.1"
TRUFFLEHOG_VERSION = "3.99.8"

MAX_DEPTH = 5
MAX_MEMBER_BYTES = 256 << 20
MAX_TOTAL_BYTES = 2 << 30
MAX_MEMBERS = 200_000

# Categories P must have: without them the gate would pass what it never checked.
REQUIRED_CATEGORIES = (
    "secrets",
    "oauth-token",
    "credential-file",
    "internal-keys",
    "internal-infra",
    "personal",
    "partner",
    "hard-rule",
    "process",
    "internal-review-tag",
)
SEVERITY = {
    "secrets": "blocker",
    # OAuth tokens (a Google refresh or access token's shape): every match is
    # a finding, with no credential-shape check after it.
    "oauth-token": "blocker",
    # A login or credentials file shipped as a file, whatever it holds.
    "credential-file": "blocker",
    "partner": "blocker",
    "hard-rule": "high",
    "internal-keys": "high",
    "internal-infra": "high",
    "personal": "high",
    "private-lab": "high",
    "gcp-project-context": "high",
    "private-ip": "high",
    "real-run-path": "high",
    "trace-file": "high",
    "unknown-binary": "high",
    "process": "medium",
    "internal-review-tag": "medium",
    "office-personal-metadata": "low",
    "seed-credentials": "low",
}
SCANNER_SEVERITY = "blocker"
# Categories matched against member paths only, not file text.
PATH_ONLY = {"credential-file", "real-run-path", "seed-credentials", "trace"}
SEVERITY_ORDER = ("blocker", "high", "medium", "low")

_TAG = re.compile(
    r"\bPY\d+-D\d+(?:-[A-Za-z0-9-]+)?|\b(?:[A-Z0-9]+-)*(?:CODEX|OPUS|SOL|MUSE)-[A-Z0-9-]+"
)
# Optional patterns entry: a regex a partner match must match whole to count as
# a code token (a variable, an env var, a key prefix) that shares a spelling
# with a listed name. It lives in the private patterns, beside the names; with
# no entry, every partner match is a finding.
PARTNER_CODE_TOKENS = "partner-code-tokens"
_RESERVED_EMAIL_DOMAINS = re.compile(
    r"(?:^|\.)(?:example\.(?:com|org|net)|example|test|invalid|localhost)$"
)
_FILE_SUFFIXES = {
    "json", "jsonl", "toml", "yaml", "yml", "py", "md", "txt", "lock", "cfg",
    "ini", "sh", "js", "ts", "html", "css", "csv", "tsv",
}  # fmt: skip
_PUBLIC_DISCORD_INVITE = "mz9rc8q8w3"
_ARCHIVE_ZIP_SUFFIXES = (
    ".zip", ".whl", ".jar", ".xlsx", ".xlsm", ".docx", ".pptx",
    ".odt", ".ods", ".odp", ".epub",
)  # fmt: skip
_STRINGS = re.compile(rb"[\x20-\x7e\t]{4,}")
_REQUIRED_EXCEPTION_FIELDS = (
    "rule",
    "artifact_member_path",
    "matched_value_sha256",
    "file_sha256",
    "reason",
    "approval_reference",
    "expires_at",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


class GateError(Exception):
    """The gate cannot decide: exit 2."""


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8", "surrogatepass")
    return hashlib.sha256(data).hexdigest()


@dataclasses.dataclass(frozen=True)
class Member:
    """One file inside an artifact, by its normalized path (``wheel!a/b.py``)."""

    path: str
    data: bytes

    @property
    def sha256(self) -> str:
        return sha256(self.data)


@dataclasses.dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    member: str
    line: int
    value: str
    file_sha256: str
    quote: str = ""

    @property
    def value_sha256(self) -> str:
        return sha256(self.value)

    def key(self) -> tuple[str, str, str, str]:
        return (self.rule, self.member, self.value_sha256, self.file_sha256)


# --------------------------------------------------------------------------
# Unpacking
# --------------------------------------------------------------------------


class _Budget:
    def __init__(self) -> None:
        self.members = 0
        self.total = 0

    def take(self, where: str, size: int) -> None:
        self.members += 1
        self.total += size
        if size > MAX_MEMBER_BYTES:
            raise GateError(f"{where}: {size} bytes, over the {MAX_MEMBER_BYTES} limit")
        if self.members > MAX_MEMBERS:
            raise GateError(f"{where}: more than {MAX_MEMBERS} members")
        if self.total > MAX_TOTAL_BYTES:
            raise GateError(f"{where}: more than {MAX_TOTAL_BYTES} bytes unpacked")


def _check_name(where: str, name: str) -> str:
    if not name or "\0" in name or "\\" in name:
        raise GateError(f"{where}: unsafe member name {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or re.match(r"[A-Za-z]:", name) or ".." in path.parts:
        raise GateError(f"{where}: member {name!r} leaves the archive")
    return str(path)


class _Readable(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


def _read_limited(fh: _Readable, where: str) -> bytes:
    body = fh.read(MAX_MEMBER_BYTES + 1)
    if len(body) > MAX_MEMBER_BYTES:
        raise GateError(f"{where}: over the {MAX_MEMBER_BYTES}-byte member limit")
    return body


def _zip_entries(where: str, data: bytes, budget: _Budget):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as e:
        raise GateError(f"{where}: not a readable zip ({e})") from None
    seen: set[str] = set()
    with archive:
        for info in archive.infolist():
            name = _check_name(where, info.filename.rstrip("/") or info.filename)
            if info.is_dir():
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise GateError(f"{where}!{name}: a symlink in an archive")
            if name in seen:
                raise GateError(f"{where}!{name}: a duplicate member")
            seen.add(name)
            budget.take(f"{where}!{name}", info.file_size)
            try:
                with archive.open(info) as fh:
                    yield name, _read_limited(fh, f"{where}!{name}")
            except (zipfile.BadZipFile, OSError, RuntimeError) as e:
                raise GateError(f"{where}!{name}: unreadable ({e})") from None


def _tar_entries(where: str, data: bytes, budget: _Budget):
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for entry in archive:
                name = _check_name(where, entry.name)
                if entry.isdir():
                    continue
                if not entry.isreg():
                    raise GateError(
                        f"{where}!{name}: a link or special file in an archive"
                    )
                if name in seen:
                    raise GateError(f"{where}!{name}: a duplicate member")
                seen.add(name)
                budget.take(f"{where}!{name}", entry.size)
                fh = archive.extractfile(entry)
                if fh is None:
                    raise GateError(f"{where}!{name}: unreadable")
                yield name, _read_limited(fh, f"{where}!{name}")
    except (tarfile.TarError, OSError, EOFError) as e:
        raise GateError(f"{where}: not a readable tar ({e})") from None


def _archive_kind(name: str, data: bytes) -> str | None:
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06") or name.endswith(
        _ARCHIVE_ZIP_SUFFIXES
    ):
        return "zip"
    if len(data) > 262 and data[257:262] == b"ustar":
        return "tar"
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if name.endswith((".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")):
        return "tar"
    return None


def _expand(path: str, data: bytes, depth: int, budget: _Budget):
    """``(path, bytes)`` of every leaf below an archive member, recursively."""
    kind = _archive_kind(path.rsplit("!", 1)[-1], data)
    if kind is None:
        yield path, data
        return
    if depth >= MAX_DEPTH:
        raise GateError(f"{path}: archives nested deeper than {MAX_DEPTH}")
    if kind == "gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as fh:
                body = _read_limited(fh, path)
        except (OSError, EOFError) as e:
            raise GateError(f"{path}: not a readable gzip ({e})") from None
        inner = path[:-3] if path.endswith(".gz") else path + ".gunzip"
        if len(body) > 262 and body[257:262] == b"ustar":
            yield from _expand_entries(
                path, _tar_entries(path, data, budget), depth, budget
            )
        else:
            budget.take(inner, len(body))
            yield from _expand(inner, body, depth + 1, budget)
        return
    entries = (_zip_entries if kind == "zip" else _tar_entries)(path, data, budget)
    yield from _expand_entries(path, entries, depth, budget)


def _expand_entries(path, entries, depth, budget):
    for name, body in entries:
        yield from _expand(f"{path}!{name}", body, depth + 1, budget)


def _normalize(label: str, names: list[str]) -> dict[str, str]:
    """Member name to its normalized path: the sdist's top folder and the wheel's
    versioned ``.dist-info``/``.data`` folders lose their version."""
    out = {}
    tops = {n.split("/", 1)[0] for n in names}
    strip_top = label == "sdist" and len(tops) == 1 and all("/" in n for n in names)
    for name in names:
        rel = name.split("/", 1)[1] if strip_top else name
        if label == "wheel":
            rel = re.sub(r"^[^/]+\.(dist-info|data)/", r"\1/", rel)
        out[name] = f"{label}!{rel}"
    return out


def artifact_members(spec: str) -> tuple[str, str, list[Member]]:
    """The label, SHA-256 and leaf members of one ARTIFACT argument."""
    budget = _Budget()
    if "=" in spec and not spec.endswith((".whl", ".tar.gz")):
        label, _, folder = spec.partition("=")
        root = Path(folder)
        if not label or not root.is_dir():
            raise GateError(f"{spec}: LABEL=DIR needs an existing directory")
        names, blobs = [], {}
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for d in dirnames:
                if (Path(dirpath) / d).is_symlink():
                    raise GateError(f"{spec}: {Path(dirpath, d)} is a symlink")
            for f in filenames:
                p = Path(dirpath) / f
                if p.is_symlink() or not p.is_file():
                    raise GateError(f"{spec}: {p} is a symlink or special file")
                rel = p.relative_to(root).as_posix()
                budget.take(rel, p.stat().st_size)
                names.append(rel)
                blobs[rel] = p.read_bytes()
        digest = sha256(json.dumps({n: sha256(b) for n, b in sorted(blobs.items())}))
    else:
        path = Path(spec)
        if not path.is_file():
            raise GateError(f"{spec}: no such artifact")
        data = path.read_bytes()
        digest = sha256(data)
        if path.name.endswith(".whl"):
            label, entries = "wheel", _zip_entries(path.name, data, budget)
        elif path.name.endswith(".tar.gz"):
            label, entries = "sdist", _tar_entries(path.name, data, budget)
        else:
            raise GateError(f"{spec}: not a .whl, a .tar.gz or LABEL=DIR")
        blobs = dict(entries)
        names = list(blobs)
    normalized = _normalize(label, names)
    members = []
    for name in names:
        for leaf, body in _expand(normalized[name], blobs[name], 0, budget):
            members.append(Member(leaf, body))
    return label, digest, members


# --------------------------------------------------------------------------
# Deciding candidates
# --------------------------------------------------------------------------


def _bounded(pattern: str) -> str:
    # The audit's bounds on unanchored URL and email runs, so a long binary
    # string cannot make a match quadratic.
    return (
        pattern.replace("[\\w.-]+", "[\\w.-]{1,253}")
        .replace("[A-Za-z0-9._%+-]+", "[A-Za-z0-9._%+-]{1,128}")
        .replace("[A-Za-z0-9.-]+", "[A-Za-z0-9.-]{1,253}")
    )


def load_patterns(path: Path) -> dict[str, re.Pattern[str]]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise GateError(f"patterns {path}: {e}") from None
    if not isinstance(raw, dict) or not raw:
        raise GateError(f"patterns {path}: not a non-empty JSON object")
    missing = [c for c in REQUIRED_CATEGORIES if c not in raw]
    if missing:
        raise GateError(f"patterns {path}: no {', '.join(missing)}")
    out = {}
    for category, pattern in raw.items():
        if not isinstance(pattern, str) or not pattern:
            raise GateError(f"patterns {path}: {category!r} is not a regex string")
        try:
            out[category] = re.compile(_bounded(pattern))
        except re.error as e:
            raise GateError(f"patterns {path}: {category!r}: {e}") from None
    return out


class _Context:
    """What a decision may need to know about the member a line is in."""

    def __init__(
        self,
        member: Member,
        lines: list[str] | None,
        partner_code_tokens: re.Pattern[str] | None = None,
    ) -> None:
        self.name = member.path.rsplit("!", 1)[-1].rsplit("/", 1)[-1]
        self.lines = lines or []
        self.partner_code_tokens = partner_code_tokens
        self._authorship: set[int] | None = None

    def authorship(self, line: int) -> bool:
        """The line is an author or maintainer field of package metadata."""
        if self._authorship is None:
            self._authorship = set()
            if self.name in ("METADATA", "PKG-INFO"):
                for n, text in enumerate(self.lines, 1):
                    if re.match(r"(?:Author|Maintainer)(?:-email)?: ", text):
                        self._authorship.add(n)
            elif self.name in ("task.md", "task.toml"):
                # A task's own author fields (task.md front matter, task.toml).
                for n, text in enumerate(self.lines, 1):
                    if re.match(r"\s*author_(?:name|email)\s*[:=]\s*\S", text):
                        self._authorship.add(n)
            elif self.name == "pyproject.toml":
                inside = False
                for n, text in enumerate(self.lines, 1):
                    if re.match(r"\s*(?:authors|maintainers)\s*=\s*\[", text):
                        inside = True
                    elif inside and re.match(r"\s*\]", text):
                        inside = False
                    elif inside and re.fullmatch(
                        r'\s*\{\s*name\s*=\s*"[^"]*"\s*,\s*email\s*=\s*"[^"@]+@[^"]+"\s*\},?\s*',
                        text,
                    ):
                        self._authorship.add(n)
        return line in self._authorship


def _looks_like_credential(value: str) -> bool:
    if value.startswith("-----BEGIN"):
        # A PEM private key's own header, not a pattern or a public key.
        return (
            re.fullmatch(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*", value)
            is not None
        )
    if value.startswith("eyJ"):
        return len(value) >= 40
    bearer = re.match(r"(?i)bearer\s+(\S*)", value)
    token = bearer.group(1) if bearer else value
    if bearer and token[:1] in ("$", "{", "<"):
        return False
    if not bearer:
        token = re.sub(
            r"(?i)^(?:github_pat_|sk-|hf_|ghp_|gho_|xox[baprs]-|AKIA|AIza|dtn_)",
            "",
            value,
        )
    runs: list[str] = re.findall(r"[A-Za-z0-9]+", token)
    longest = max(runs, key=lambda run: len(run)) if runs else ""
    return len(longest) >= 16 and any(c.isdigit() for c in longest)


def _public_infra(value: str, line: str) -> bool:
    low = value.lower()
    if low == "skillsbench":
        # SkillsBench, the public benchmark; its GCP-project spelling is
        # gcp-project-context's.
        return True
    if (
        _PUBLIC_DISCORD_INVITE in low
        and "slack.com" not in low
        and "linear.app" not in low
    ):
        return True  # the README's public Discord invite and its badge
    if re.match(r"(?:https://)?storage\.googleapis\.com/antigravity-public/", value):
        return True  # Google's public Antigravity download
    if re.fullmatch(r"gs://\{[a-z_]+\}|gs://dummy/", value):
        return True  # placeholders
    # A runtime temp-file prefix.
    return low == "benchflow-private" and ".benchflow-private-" in line


# Homes that name no one: the sandbox's own account and the usual stand-ins.
_PLACEHOLDER_HOME = re.compile(
    r"/(?:home|Users)/(?:agent|dev|me|name|runner|ubuntu|user|username|you)(?:/|$)"
)


def _personal(value: str, line: int, ctx: _Context) -> bool:
    if value.startswith("/root/"):
        return False  # container and tool homes
    if value.startswith(("/Users/", "/home/")):
        return _PLACEHOLDER_HOME.match(value) is None
    if "@" in value:
        local, _, domain = value.rpartition("@")
        domain = domain.lower()
        if domain.rsplit(".", 1)[-1] in _FILE_SUFFIXES:
            return False  # "name@1.json", a file name
        if _RESERVED_EMAIL_DOMAINS.search(domain) or local.lower() == "git":
            return False
        if local.lower().startswith(("noreply", "no-reply")) or domain.endswith(
            "users.noreply.github.com"
        ):
            return False
    return not ctx.authorship(line)


def decide(
    category: str, value: str, line: int, text: str, ctx: _Context
) -> str | None:
    """The finding's severity, or None for a candidate that is not one."""
    if category == "trace":
        return None
    severity = SEVERITY.get(category, "high")
    if category == "secrets":
        return severity if _looks_like_credential(value) else None
    if category == "internal-infra":
        return None if _public_infra(value, text) else severity
    if category == "personal":
        return severity if _personal(value, line, ctx) else None
    if category == "partner":
        tokens = ctx.partner_code_tokens
        return None if tokens is not None and tokens.fullmatch(value) else severity
    if category == "process":
        if value.lower() in ("council", "founder") or _TAG.fullmatch(value):
            return severity
        return None
    if category == "private-ip":
        return None if value.endswith(".0") else severity  # a network, as in a CIDR
    return severity


def _decoded_variants(line: str) -> list[str]:
    out = []
    if "&" in line:
        out.append(html.unescape(line))
    if re.search(r"%[0-9A-Fa-f]{2}", line):
        out.append(urllib.parse.unquote(line))
    if "\\" in line:
        out.append(
            re.sub(
                r"\\u([0-9a-fA-F]{4})|\\x([0-9a-fA-F]{2})|\\(/)",
                lambda m: (
                    chr(int(m.group(1) or m.group(2), 16)) if not m.group(3) else "/"
                ),
                line,
            )
        )
    return [v for v in out if v != line]


_KNOWN_BINARY = (
    b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
    b"MZ", b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"wOFF", b"wOF2", b"OTTO",
    b"\x00\x01\x00\x00", b"\x00\x00\x01\x00",
)  # fmt: skip


def _text(data: bytes) -> str | None:
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_member(
    member: Member, patterns: dict[str, re.Pattern[str]]
) -> tuple[list[Finding], Counter[str]]:
    findings: list[Finding] = []
    candidates: Counter[str] = Counter()
    text = _text(member.data)
    # (line, text) units: the whole text file as line -1 (each match is placed
    # on its own line), each decoded variant of a line, or a binary's strings.
    units: list[tuple[int, str]] = []
    breaks: list[int] = []  # where each line after the first starts
    if text is None:
        known = member.data.startswith(_KNOWN_BINARY) or (
            member.data[:4] == b"RIFF" and member.data[8:12] == b"WEBP"
        )
        if not known:
            findings.append(
                Finding(
                    "unknown-binary", SEVERITY["unknown-binary"], member.path, 0,
                    member.sha256, member.sha256, "a binary of a kind with no reviewed reader",
                )
            )  # fmt: skip
        # A binary's "line" is the byte offset of its string, as strings -t d.
        units = [
            (m.start(), m.group().decode("ascii"))
            for m in _STRINGS.finditer(member.data)
        ]
        ctx = _Context(member, None, patterns.get(PARTNER_CODE_TOKENS))
    else:
        lines = re.split(r"\r\n|\n|\r", text)
        ctx = _Context(member, lines, patterns.get(PARTNER_CODE_TOKENS))
        units.append((-1, text))
        breaks = [m.end() for m in re.finditer(r"\r\n|\n|\r", text)]
        for n, line in enumerate(lines, 1):
            units.extend((n, v) for v in _decoded_variants(line))
        if member.path.endswith(".jsonl"):
            findings.append(
                Finding(
                    "trace-file", SEVERITY["trace-file"], member.path, 0,
                    member.path, member.sha256, "a JSON-lines data file",
                )
            )  # fmt: skip
    seen: set[tuple[str, int, str]] = set()
    path_text = member.path.replace("!", "/")
    for category, rx in patterns.items():
        if category == PARTNER_CODE_TOKENS:
            continue
        for m in rx.finditer(path_text):
            candidates[category] += 1
            severity = decide(category, m.group(), 0, path_text, ctx)
            if severity and (category, 0, m.group()) not in seen:
                seen.add((category, 0, m.group()))
                findings.append(
                    Finding(
                        category,
                        severity,
                        member.path,
                        0,
                        m.group(),
                        member.sha256,
                        path_text[-80:],
                    )
                )
        if category in PATH_ONLY:
            continue
        for unit_line, unit in units:
            for m in rx.finditer(unit):
                candidates[category] += 1
                value = m.group()
                if unit_line == -1:  # the whole file: the match's own line
                    index = bisect.bisect_right(breaks, m.start())
                    lo = breaks[index - 1] if index else 0
                    n, line, start = index + 1, lines[index], m.start() - lo
                else:
                    n, line, start = unit_line, unit, m.start()
                severity = decide(category, value, n, line, ctx)
                if severity and (category, n, value) not in seen:
                    seen.add((category, n, value))
                    quote = line[max(0, start - 20) : start + len(value) + 50][:80]
                    findings.append(
                        Finding(
                            category,
                            severity,
                            member.path,
                            n,
                            value,
                            member.sha256,
                            quote,
                        )
                    )
    return findings, candidates


# --------------------------------------------------------------------------
# Secret scanners
# --------------------------------------------------------------------------


def _stage(members: list[Member], root: Path) -> dict[str, Member]:
    """Write every leaf under ``root/payload`` (a binary's strings under
    ``root/strings``) and map each written path back to its member."""
    mapping: dict[str, Member] = {}
    for i, member in enumerate(members):
        rel = "/".join(
            re.sub(r"[^A-Za-z0-9._@+=-]", "_", part) or "_"
            for part in member.path.replace("!", "/").split("/")
        )
        target = root / "payload" / f"{i:06d}" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(member.data)
        mapping[str(target.resolve())] = member
        if _text(member.data) is None:
            strings = root / "strings" / f"{i:06d}" / f"{rel}.strings"
            strings.parent.mkdir(parents=True, exist_ok=True)
            strings.write_bytes(
                b"\n".join(m.group() for m in _STRINGS.finditer(member.data))
            )
            mapping[str(strings.resolve())] = member
    return mapping


def _scanner(scanners: Path, name: str, version: str, version_args: list[str]) -> str:
    binary = (scanners / name).absolute()  # the scanners run from a temp folder
    if not os.access(binary, os.X_OK):
        raise GateError(
            f"{binary}: missing; install with tools/install_leak_scanners.sh"
        )
    try:
        out = subprocess.run(
            [str(binary), *version_args], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GateError(f"{binary}: {e}") from None
    if version not in (out.stdout + out.stderr).split():
        raise GateError(f"{binary}: not version {version}")
    return str(binary)


def _member_for(mapping: dict[str, Member], file: str, cwd: Path) -> Member:
    path = Path(file)
    resolved = str((path if path.is_absolute() else cwd / path).resolve())
    if resolved not in mapping:
        raise GateError(f"a scanner reported {file!r}, which the gate did not write")
    return mapping[resolved]


def run_scanners(members: list[Member], scanners: Path, work: Path) -> list[Finding]:
    gitleaks = _scanner(scanners, "gitleaks", GITLEAKS_VERSION, ["version"])
    trufflehog = _scanner(scanners, "trufflehog", TRUFFLEHOG_VERSION, ["--version"])
    mapping = _stage(members, work)
    findings: list[Finding] = []
    targets = [p for p in (work / "payload", work / "strings") if p.exists()]
    for target in targets:
        report = work / f"gitleaks-{target.name}.json"
        try:
            done = subprocess.run(
                [
                    gitleaks, "dir", str(target), "--no-banner", "--log-level", "error",
                    "--ignore-gitleaks-allow", "--max-archive-depth", "10",
                    "--max-decode-depth", "5", "--report-format", "json",
                    "--report-path", str(report), "--exit-code", "3",
                ],
                capture_output=True, text=True, timeout=3600, cwd=work,
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired) as e:
            raise GateError(f"gitleaks: {e}") from None
        if done.returncode not in (0, 3):
            raise GateError(f"gitleaks exited {done.returncode}: {done.stderr[-400:]}")
        try:
            rows = json.loads(report.read_text() or "[]")
        except (OSError, ValueError) as e:
            raise GateError(f"gitleaks report: {e}") from None
        if done.returncode == 3 and not rows:
            raise GateError("gitleaks reported leaks but wrote none")
        for row in rows:
            member = _member_for(mapping, row["File"], work)
            findings.append(
                Finding(
                    f"gitleaks:{row['RuleID']}", SCANNER_SEVERITY, member.path,
                    int(row.get("StartLine") or 0), row.get("Secret") or row.get("Match") or "",
                    member.sha256, "xxxx…",
                )
            )  # fmt: skip
        report.unlink()
    try:
        done = subprocess.run(
            [
                trufflehog, "filesystem", *map(str, targets), "--no-verification",
                "--no-update", "--no-ignore-tag", "--json", "--fail-on-scan-errors",
                "--results=verified,unknown,unverified,filtered_unverified",
            ],
            capture_output=True, text=True, timeout=3600, cwd=work,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GateError(f"trufflehog: {e}") from None
    if done.returncode != 0:
        raise GateError(f"trufflehog exited {done.returncode}: {done.stderr[-400:]}")
    for line in done.stdout.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            where = row["SourceMetadata"]["Data"]["Filesystem"]
        except (ValueError, KeyError, TypeError):
            raise GateError("trufflehog wrote a line the gate cannot read") from None
        member = _member_for(mapping, where["file"], work)
        findings.append(
            Finding(
                f"trufflehog:{row.get('DetectorName', 'unknown')}", SCANNER_SEVERITY,
                member.path, int(where.get("line") or 0),
                row.get("RawV2") or row.get("Raw") or "", member.sha256, "xxxx…",
            )
        )  # fmt: skip
    return findings


def _in_record_digest(member: Member, value: str) -> bool:
    """The value is inside one of the wheel RECORD's own ``sha256=`` digests,
    which a scanner can take for a key: decided from the file's fields, not
    its path."""
    if not member.path.endswith("dist-info/RECORD") or len(value) < 8:
        return False
    for line in (_text(member.data) or "").splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) == 3 and parts[1].startswith("sha256=") and value in parts[1]:
            return True
    return False


# --------------------------------------------------------------------------
# The allowlist
# --------------------------------------------------------------------------


def _expiry(value: str) -> dt.datetime:
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            day = dt.date.fromisoformat(value)
            return dt.datetime.combine(day, dt.time.max, tzinfo=dt.UTC)
        moment = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise GateError(f"allowlist: expires_at {value!r} is not an ISO date") from None
    if moment.tzinfo is None:
        raise GateError(f"allowlist: expires_at {value!r} has no time zone")
    return moment


def load_allowlist(path: Path, now: dt.datetime) -> tuple[dict, int]:
    """Active exceptions by (rule, member, value hash, file hash), and how many
    had expired."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise GateError(f"allowlist {path}: {e}") from None
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise GateError(f"allowlist {path}: not a version 1 allowlist")
    entries = doc.get("exceptions")
    if not isinstance(entries, list):
        raise GateError(f"allowlist {path}: no exceptions list")
    active, expired = {}, 0
    for n, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise GateError(f"allowlist {path}: exception {n} is not an object")
        entry = cast("dict[str, Any]", raw_entry)
        for field in _REQUIRED_EXCEPTION_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise GateError(f"allowlist {path}: exception {n} has no {field}")
        for field in ("matched_value_sha256", "file_sha256"):
            if not _HEX64.fullmatch(entry[field]):
                raise GateError(
                    f"allowlist {path}: exception {n}: {field} is not a SHA-256"
                )
        if _expiry(entry["expires_at"]) <= now:
            expired += 1
            continue
        key = (
            entry["rule"],
            entry["artifact_member_path"],
            entry["matched_value_sha256"],
            entry["file_sha256"],
        )
        active[key] = entry
    return active, expired


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def gate(
    artifacts: list[str],
    patterns: dict[str, re.Pattern[str]],
    allowlist: dict,
    scanners: Path | None,
) -> dict:
    members_by_artifact = {}
    for spec in artifacts:
        label, digest, members = artifact_members(spec)
        members_by_artifact[spec] = (label, digest, members)
    paths = [m.path for _, _, ms in members_by_artifact.values() for m in ms]
    duplicates = [p for p, k in Counter(paths).items() if k > 1]
    if duplicates:
        raise GateError(f"two artifacts share member paths, first {duplicates[0]!r}")
    findings: list[Finding] = []
    candidates: Counter[str] = Counter()
    every = [m for _, _, ms in members_by_artifact.values() for m in ms]
    for member in every:
        found, seen = scan_member(member, patterns)
        findings += found
        candidates.update(seen)
    if scanners is not None:
        with tempfile.TemporaryDirectory(prefix="leak-gate-") as work:
            findings += run_scanners(every, scanners, Path(work))
    by_path = {m.path: m for m in every}
    findings = [
        f
        for f in findings
        if not (f.rule == "secrets" or ":" in f.rule)
        or not _in_record_digest(by_path[f.member], f.value)
    ]
    allowed = [f for f in findings if f.key() in allowlist]
    used = {f.key() for f in allowed}
    open_findings = [f for f in findings if f.key() not in allowlist]
    open_findings.sort(
        key=lambda f: (SEVERITY_ORDER.index(f.severity), f.member, f.line, f.rule)
    )
    return {
        "artifacts": [
            {"artifact": spec, "label": label, "sha256": digest, "members": len(ms)}
            for spec, (label, digest, ms) in members_by_artifact.items()
        ],
        "scanners": None
        if scanners is None
        else {"gitleaks": GITLEAKS_VERSION, "trufflehog": TRUFFLEHOG_VERSION},
        "candidates": dict(sorted(candidates.items())),
        "findings": findings,
        "allowed": allowed,
        "open": open_findings,
        "unused_exceptions": [k for k in allowlist if k not in used],
    }


def _finding_row(f: Finding, *, values: bool) -> dict:
    row = {
        "rule": f.rule,
        "severity": f.severity,
        "artifact_member_path": f.member,
        "line": f.line,
        "matched_value_sha256": f.value_sha256,
        "file_sha256": f.file_sha256,
    }
    if values:
        secret = f.rule in ("secrets", "oauth-token") or ":" in f.rule
        row["quote"] = "xxxx…" if secret else f.quote
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("artifacts", nargs="+", metavar="ARTIFACT")
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scanners", type=Path, help="folder with gitleaks and trufflehog"
    )
    group.add_argument(
        "--no-scanners",
        action="store_true",
        help="patterns only (a local check; a release gate never uses it)",
    )
    parser.add_argument(
        "--report-dir", type=Path, help="private folder for values and quotes"
    )
    parser.add_argument(
        "--propose", type=Path, help="write exception drafts for open findings"
    )
    args = parser.parse_args(argv)
    try:
        patterns = load_patterns(args.patterns)
        allowlist, expired = load_allowlist(args.allowlist, dt.datetime.now(dt.UTC))
        result = gate(
            args.artifacts,
            patterns,
            allowlist,
            None if args.no_scanners else args.scanners,
        )
    except GateError as e:
        print(f"LEAK GATE ERROR: {e}")
        return 2
    for a in result["artifacts"]:
        print(
            f"artifact {a['label']} {a['sha256']} members={a['members']} ({a['artifact']})"
        )
    print("scanners:", result["scanners"] or "none (patterns only)")
    print("candidates:", json.dumps(result["candidates"], sort_keys=True))
    by_rule = Counter(f.rule for f in result["findings"])
    print("findings:", json.dumps(dict(sorted(by_rule.items()))))
    for f in result["open"]:
        print(
            f"FINDING {f.severity} {f.rule} {f.member}:{f.line} "
            f"value_sha256={f.value_sha256[:16]} file_sha256={f.file_sha256[:16]}"
        )
    print(
        f"allowlisted {len(result['allowed'])}; open {len(result['open'])}; "
        f"expired exceptions {expired}; unused exceptions {len(result['unused_exceptions'])}"
    )
    summary = {
        "result": "pass" if not result["open"] else "fail",
        "artifacts": result["artifacts"],
        "scanners": result["scanners"],
        "candidates": result["candidates"],
        "findings_by_rule": dict(sorted(by_rule.items())),
        "allowlisted": len(result["allowed"]),
        "open": len(result["open"]),
        "expired_exceptions": expired,
        "unused_exceptions": len(result["unused_exceptions"]),
    }
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (args.report_dir / "findings.json").write_text(
            json.dumps(
                {
                    "open": [_finding_row(f, values=True) for f in result["open"]],
                    "allowlisted": [
                        _finding_row(f, values=True) for f in result["allowed"]
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    if args.propose:
        drafts = {}
        for f in result["open"]:
            drafts.setdefault(
                f.key(),
                {
                    "rule": f.rule,
                    "artifact_member_path": f.member,
                    "matched_value_sha256": f.value_sha256,
                    "file_sha256": f.file_sha256,
                    "reason": "",
                    "approval_reference": "",
                    "expires_at": "",
                },
            )
        args.propose.write_text(json.dumps(list(drafts.values()), indent=2) + "\n")
    print("LEAK GATE PASS" if not result["open"] else "LEAK GATE FAIL")
    return 0 if not result["open"] else 1


if __name__ == "__main__":
    sys.exit(main())
