"""The published docs carry nothing of ours that isn't meant to be public.

``tools/docs_leaks.py`` reads every file under ``docs/`` (pages, generated
reference, example downloads, Office members, binary strings) for secrets,
private ticket keys and review tags, internal hosts, repositories and names,
home folders, private preview hosts, chat and tracker links, Office author
fields and captured runs (release leak audit, 2026-11-05). A finding meant to
be there is listed, exactly, in ``tools/docs_leaks_allow.tsv``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from tools.docs_leaks import allowed, findings


def _shown(found) -> str:
    return "\n".join(
        f"{f.path}:{f.line}: {f.rule}: {f.text if f.rule != 'secret' else f.text[:6] + '…'}"
        for f in found[:20]
    )


def test_the_docs_carry_no_leak() -> None:
    left = [f for f in findings() if f.key() not in allowed()]
    assert not left, (
        f"{len(left)} finding(s); fix them or list them in tools/docs_leaks_allow.tsv:\n{_shown(left)}"
    )


def test_every_allowed_finding_is_still_in_the_docs() -> None:
    stale = allowed() - {f.key() for f in findings()}
    assert not stale, (
        f"tools/docs_leaks_allow.tsv lists findings the docs no longer have: {sorted(stale)}"
    )


def test_the_check_finds_each_kind(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    (docs / "examples" / "run").mkdir(parents=True)
    key = "sk-" + "ant-" + "a1B2" * 6
    (docs / "page.md").write_text(
        f"export ANTHROPIC_API_KEY={key}\n"
        "Fixed in EN-1234, see PY9-D123-X.\n"
        "Logs in /Users/someone/runs and /home/agent/workspace.\n"
        "Preview: https://docs-v9-benchflow.vercel.app/x and https://linear.app/x/issue/1\n"
        "Code: https://github.com/benchflow-ai/not-public and https://github.com/benchflow-ai/benchflow\n"
    )
    (docs / "examples" / "run" / "result.json").write_text("{}")
    sheet = io.BytesIO()
    with zipfile.ZipFile(sheet, "w") as z:
        z.writestr("docProps/core.xml", "<dc:creator>Someone</dc:creator>")
    (docs / "examples" / "book.xlsx").write_bytes(sheet.getvalue())
    rules = sorted({f.rule for f in findings(docs)})
    assert rules == [
        "captured-run",
        "chat-or-tracker",
        "home-path",
        "office-author",
        "private-host",
        "private-repo",
        "review-tag",
        "secret",
        "ticket-key",
    ], rules
    assert not [f for f in findings(docs) if "/home/agent" in f.text], (
        "a sandbox user's home is the product's path"
    )


def test_the_site_mode_reads_the_index_its_files_and_every_page(monkeypatch) -> None:
    """``--site`` scans what a built site serves: llms.txt, the BenchFlow bundles it
    links (at whatever origin the build was given), and every nav page's article and
    Markdown; other collections' files are left to their owners."""
    import tools.docs_leaks as docs_leaks

    base = "http://127.0.0.1:3999"
    key = "sk-" + "proj-" + "c3D4" * 6
    served = {
        f"{base}/docs/llms.txt": (
            "- [Start](https://www.example.com/docs/llms-full/benchflow/start.txt)\n"
            "- [Other](https://www.example.com/docs/llms-full/other/guide.txt)\n"
        ),
        f"{base}/docs/llms-full/benchflow/start.txt": f"the key {key}\n",
        f"{base}/docs/llms-full/other/guide.txt": f"not ours {key}\n",
        f"{base}/docs/benchflow/start/quick-start.md": "see github.com/benchflow-ai/not-public\n",
    }
    asked = []

    def fetch(url: str) -> str:
        asked.append(url)
        return served.get(url, "<html><article class='x'><p>clean</p></article></html>")

    monkeypatch.setattr(docs_leaks, "_fetch", fetch)
    found = docs_leaks.site_findings(base)
    assert {(f.path, f.rule) for f in found} == {
        ("/docs/llms-full/benchflow/start.txt", "secret"),
        ("/docs/benchflow/start/quick-start.md", "private-repo"),
    }
    assert f"{base}/docs/benchflow" in asked and f"{base}/docs/benchflow.md" in asked
    assert f"{base}/docs/llms-full/other/guide.txt" not in asked
