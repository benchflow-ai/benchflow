"""Check that the published docs carry nothing of ours that isn't meant to be public.

    python tools/docs_leaks.py            # list every finding; exit 1 if there is one

It reads every file under ``docs/`` (the pages, the generated reference and the
examples the pages download), the members of Office and zip files among them,
and the printable strings of other binaries, and reports, with file and line:

- secrets: provider key prefixes, private keys, JWTs and bearer tokens;
- private ticket keys and internal review tags;
- internal hosts, projects, repositories and the people and partners named in
  our own work; these names are private in themselves, so this file holds only
  the SHA-256 of each one's lowercase spelling, and a word is checked whole,
  without a trailing number, and in each of its hyphen or dot parts;
- home folders, private preview hosts, storage buckets, chat and tracker links,
  and links to repositories that aren't public;
- author and editor names in Office documents;
- captured runs (job, trial and trajectory files) among the examples;
- words that narrate how the docs were made.

With ``--site URL`` it reads a built docs site instead: ``/docs/llms.txt``, the
BenchFlow section bundles that index links (``/docs/llms-full/benchflow/*.txt``),
and each page in ``docs/nav.json`` as served, the article of its HTML and its
Markdown. Other collections on the site, and ``llms-full.txt``, which bundles them
all, are theirs to check.

    python tools/docs_leaks.py --site http://127.0.0.1:3000

A finding that is meant to be there goes in ``tools/docs_leaks_allow.tsv``:
the rule, the file, the SHA-256 of the matched text and the reason. The
release gate's private regular expressions (rule name to regex, JSON) can be
added with ``BENCHFLOW_LEAK_PATTERNS=<file>``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ALLOW = ROOT / "tools" / "docs_leaks_allow.tsv"


def _digest(text: str) -> str:
    return hashlib.sha256(text.lower().encode()).hexdigest()[:16]


# First 16 hex digits of the SHA-256 of each private name's lowercase spelling.
HIDDEN = {
    # internal hosts, projects and repositories
    "f9b4827d02cf5913": "internal-infra",
    "4ec3bf03f33a997e": "internal-infra",
    "05eb28be5eda9165": "internal-infra",
    "9d258cbb696f1aa3": "internal-infra",
    "42705aab0bc61565": "internal-infra",
    "79c79b3199656382": "internal-infra",
    "cfc384c2a2f34240": "internal-infra",
    "9f4711dc9e5475f9": "internal-infra",
    "319534a9f8becfea": "internal-infra",
    "245bfabdfe4ef7a2": "internal-infra",
    "2c037b2a3ecb2c45": "internal-infra",
    # partners and customers
    "47a6164d98ac0064": "partner",
    "2270e73a86e507f7": "partner",
    "d3a053245104ec44": "partner",
    "245f1e177c5169cb": "partner",
    # people
    "184b9308d77d874d": "personal",
    "a2050b1e737999c3": "personal",
    "9cb0017963c605bf": "personal",
    # names the docs never use
    "c1d64b2d4cb30f1b": "hard-rule",
    "8d48839ab701047b": "hard-rule",
    # words that narrate how the docs were made
    "59458508a0827cff": "process",
    "a0c7ec73f6d7f2b4": "process",
}
# Prefixes of our tracker's ticket keys (KEY-123).
TICKET_PREFIXES = {
    "dbd3a49d0d906b4e",
    "85ea2356aea7b503",
    "5e657ff6158d3e2a",
    "82fe032bd9337b5d",
}
# Repositories under the benchflow-ai organization that are public.
PUBLIC_REPOS = {
    "benchflow",
    "skillsbench",
    "agents",
    "robouse",
}  # robouse: public since 2026-10-21 (GitHub API)

PATTERNS: dict[str, re.Pattern[str]] = {
    "secret": re.compile(
        r"\b(?:sk-(?:ant-|proj-)?|hf_|ghp_|gho_|github_pat_|xox[baprs]-|AKIA|AIza|dtn_)[A-Za-z0-9_\-]{16,}"
        r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
        r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
        r"|\b[Bb]earer [A-Za-z0-9_\-.=]{24,}"
    ),
    "review-tag": re.compile(r"\bPY\d+-D\d+(?:-[A-Za-z0-9-]+)?|\bK\d+-R\d+-\d+\b"),
    # A home folder on someone's machine; a sandbox user's home (/home/agent) is the product's own path.
    "home-path": re.compile(
        r"/(?:Users|home)/(?!(?:agent|user|runner|ubuntu|node)\b)[A-Za-z0-9_.-]+(?=/|\b)"
    ),
    "private-host": re.compile(
        r"\b[\w-]+\.vercel\.app\b|\bgs://[\w.-]+|\bstorage\.googleapis\.com/[\w.-]+"
    ),
    "chat-or-tracker": re.compile(
        r"\b(?:[\w-]+\.)?slack\.com/|\blinear\.app/|\bdiscord(?:app)?\.com/channels/"
    ),
}
TICKET = re.compile(r"\b([A-Z][A-Z0-9]{1,4})-\d{1,5}\b")
REPO_LINK = re.compile(r"github\.com[:/]benchflow-ai/([A-Za-z0-9_.-]+)")
WORD = re.compile(r"[a-z0-9][a-z0-9.\-]*[a-z0-9]|[a-z0-9]")
OFFICE_PEOPLE = re.compile(r"<(dc:creator|cp:lastModifiedBy)>([^<]+)</\1>")
# Files a run writes; among the examples they would be a captured run.
RUN_FILES = re.compile(
    r"(?:^|/)(?:result\.json|results\.jsonl|user_rounds\.jsonl|rewards\.jsonl|acp_trajectory\.jsonl|llm_trajectory\.jsonl|timing\.json)$"
)
ZIPPED = {".xlsx", ".docx", ".pptx", ".zip"}


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int
    text: str

    def key(self) -> tuple[str, str, str]:
        return (self.rule, self.path, hashlib.sha256(self.text.encode()).hexdigest())


def _private_patterns() -> dict[str, re.Pattern[str]]:
    path = os.environ.get("BENCHFLOW_LEAK_PATTERNS")
    if not path:
        return {}
    return {
        f"private:{rule}": re.compile(rx)
        for rule, rx in json.loads(Path(path).read_text()).items()
    }


def _scan_text(
    rel: str, text: str, extra: dict[str, re.Pattern[str]]
) -> Iterator[Finding]:
    for number, line in enumerate(text.splitlines(), start=1):
        for rule, rx in {**PATTERNS, **extra}.items():
            for m in rx.finditer(line):
                yield Finding(rule, rel, number, m.group(0))
        for m in TICKET.finditer(line):
            if _digest(m.group(1)) in TICKET_PREFIXES:
                yield Finding("ticket-key", rel, number, m.group(0))
        for m in REPO_LINK.finditer(line):
            repo = m.group(1).removesuffix(".git")
            if repo.lower() not in PUBLIC_REPOS:
                yield Finding("private-repo", rel, number, m.group(0))
        lower = line.lower()
        for m in WORD.finditer(lower):
            word = m.group(0)
            parts = {word, re.sub(r"-\d+$", "", word), *re.split(r"[.-]", word)}
            for part in parts:
                rule = HIDDEN.get(_digest(part)) if part else None
                if rule:
                    yield Finding(rule, rel, number, line[m.start() : m.end()])
                    break


def _strings(data: bytes) -> str:
    return "\n".join(s.decode("ascii") for s in re.findall(rb"[\x20-\x7e]{6,}", data))


def _files(docs: Path) -> list[Path]:
    """The files that publish: those git tracks under ``docs`` (a test run's
    ``__pycache__`` never does); outside a checkout, every file but those."""
    try:
        listed = (
            subprocess.run(
                ["git", "ls-files", "-z", "--", str(docs)],
                cwd=ROOT,
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .split("\0")
        )
        files = [ROOT / name for name in listed if name]
        if files:
            return sorted(p for p in files if p.is_file())
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(
        p for p in docs.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )


def findings(docs: Path = DOCS) -> list[Finding]:
    extra = _private_patterns()
    out: list[Finding] = []
    for path in _files(docs):
        rel = path.relative_to(ROOT if ROOT in path.parents else docs.parent).as_posix()
        if "/examples/" in f"/{rel}" and RUN_FILES.search(rel):
            out.append(Finding("captured-run", rel, 1, path.name))
        data = path.read_bytes()
        if path.suffix.lower() in ZIPPED:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    member = z.read(name).decode("utf-8", errors="replace")
                    for m in OFFICE_PEOPLE.finditer(member):
                        out.append(
                            Finding("office-author", f"{rel}!/{name}", 1, m.group(0))
                        )
                    out.extend(_scan_text(f"{rel}!/{name}", member, extra))
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = _strings(data)
        out.extend(_scan_text(rel, text, extra))
    return out


def allowed() -> set[tuple[str, str, str]]:
    keys = set()
    if ALLOW.exists():
        for row in ALLOW.read_text().splitlines()[1:]:
            if row.strip():
                rule, path, digest, _reason = row.split("\t", 3)
                keys.add((rule, path, digest))
    return keys


def _fetch(url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def site_findings(base: str) -> list[Finding]:
    """The built site's agent index, BenchFlow's bundles, and every docs page as served."""
    import html
    from urllib.parse import urlsplit

    base = base.rstrip("/")
    extra = _private_patterns()
    index = _fetch(f"{base}/docs/llms.txt")
    # The index roots its links at the site's configured origin; read each path from this build.
    linked = {urlsplit(u).path for u in re.findall(r"https?://[^\s)\]]+\.txt", index)}
    paths = {"/docs/llms.txt"} | {
        p for p in linked if p.startswith("/docs/llms-full/benchflow/")
    }
    out: list[Finding] = []
    for path in sorted(paths):
        out.extend(_scan_text(path, _fetch(base + path), extra))
    nav = json.loads((DOCS / "nav.json").read_text(encoding="utf-8"))
    for section in nav["sections"]:
        for page in section["pages"]:
            path = "/docs/benchflow" + (
                "" if page["path"] == "README" else "/" + page["path"]
            )
            article = re.search(
                r"<article[^>]*>(.*?)</article>", _fetch(base + path), re.S
            )
            if not article:
                out.append(Finding("no-article", path, 1, "the page has no <article>"))
                continue
            text = html.unescape(re.sub(r"<[^>]+>", " ", article.group(1)))
            out.extend(_scan_text(path, text, extra))
            out.extend(_scan_text(path + ".md", _fetch(base + path + ".md"), extra))
    return out


def main() -> int:
    ok = allowed()
    args = sys.argv[1:]
    if args[:1] == ["--site"] and len(args) == 2:
        left = [f for f in site_findings(args[1]) if f.key() not in ok]
    else:
        left = [f for f in findings() if f.key() not in ok]
    for f in left:
        shown = f.text if f.rule != "secret" else f.text[:6] + "…"
        print(f"{f.path}:{f.line}: {f.rule}: {shown}")
    print(f"{len(left)} finding(s)" if left else "no findings")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
