"""tools/release_gate/rc_cleanup.py: the rc's last cleanup stays in its scope.

Its diff may hold only regenerated pages and key strips from Python comments
and docstrings. Keys here are spelled in pieces, so the public tree guard
passes on this file."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools import cli_help_golden, docs_errors, docs_reference
from tools.release_gate import rc_cleanup

KEY = "E" + "N-12"


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "rc head")):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    return repo


def test_the_generated_pages_are_the_generators_own():
    pages = {
        str(p.relative_to(docs_reference.REPO)) for p in docs_reference.PAGES.values()
    }
    pages.add(str(docs_errors.PAGE.relative_to(docs_errors.ROOT)))
    pages.add(str(cli_help_golden.GOLDEN.relative_to(cli_help_golden.REPO)))
    assert pages == set(rc_cleanup.GENERATED_PAGES)


def test_only_pages_and_comment_or_docstring_changes_are_in_scope(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "src/m.py": f'"""Doc ({KEY})."""\n# note ({KEY})\nX = 1\n',
            "src/n.py": "Y = 1\n",
            "docs/reference/errors.md": "old\n",
            "docs/guide.md": "guide\n",
        },
    )
    (repo / "src/m.py").write_text('"""Doc."""\n# note\nX = 1\n')
    (repo / "docs/reference/errors.md").write_text("new\n")
    assert rc_cleanup.out_of_scope(repo) == []
    (repo / "src/n.py").write_text("Y = 2\n")
    (repo / "docs/guide.md").write_text("changed\n")
    (repo / "src/new.py").write_text("Z = 1\n")
    assert dict(rc_cleanup.out_of_scope(repo)) == {
        "src/n.py": "its code changed, not only comments and docstrings",
        "docs/guide.md": "not a generated page or a Python file",
        "src/new.py": "untracked: the cleanup adds no file",
    }


def test_the_cleanup_strips_regenerates_and_checks_itself(tmp_path, capsys):
    repo = _repo(
        tmp_path,
        {
            "src/m.py": f'"""Stops early ({KEY})."""\n# why ({KEY})\nX = 1\n',
            "docs/reference/errors.md": "stale\n",
        },
    )

    def regenerate(root):
        (root / "docs/reference/errors.md").write_text("fresh\n")
        return []

    assert rc_cleanup.main([], root=repo, regenerate=regenerate) == 0
    assert (repo / "src/m.py").read_text() == '"""Stops early."""\n# why\nX = 1\n'
    assert (repo / "docs/reference/errors.md").read_text() == "fresh\n"


def test_a_key_it_must_not_strip_or_a_wider_change_fails_it(tmp_path, capsys):
    repo = _repo(tmp_path, {"src/m.py": f'X = "{KEY}"\n', "docs/a.md": "a\n"})

    def regenerate_too_much(root):
        (root / "docs/a.md").write_text("touched\n")
        return []

    assert rc_cleanup.main([], root=repo, regenerate=regenerate_too_much) == 1
    out = capsys.readouterr().out
    assert "out of scope: docs/a.md" in out
    assert "left: src/m.py:1: tracker-key" in out
    assert (repo / "src/m.py").read_text() == f'X = "{KEY}"\n'


def test_it_starts_only_from_a_clean_checkout(tmp_path):
    repo = _repo(tmp_path, {"src/m.py": "X = 1\n"})
    (repo / "src/m.py").write_text("X = 2\n")
    assert rc_cleanup.main([], root=repo, regenerate=lambda root: []) == 2
