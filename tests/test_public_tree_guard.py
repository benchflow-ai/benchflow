"""The public tree guard (tools/public_tree_guard.py).

The private release branches merge into the public main, so whatever their
tree holds is published. public-release.yml refuses a tagged tree with a
``release/`` path, a linear.app URL or a tracker key. This file spells the
forbidden strings in pieces, so the guard passes on it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from tools import public_tree_guard as guard

ROOT = Path(__file__).resolve().parents[1]
# A key of one of our tracker's teams, refused anywhere; a key-shaped token
# of no team of ours, refused only in the public text (CHANGELOG, README).
KEY = "E" + "N-12"
SHAPED = "TEAM" + "-12"
URL = "https://linear" + ".app/team/issue/abc"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "tree")
    return repo


def test_a_clean_tree_passes(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "README.md": "Uses SHA-256, UTF-8, ISO-8601, PEP-440, GPT-6, CVE-2026-1, RFC-9110, AES-256.\n",
            "src/x.py": "x = 1\n",
            # Key-shaped, but no tracker's: review labels, regex text, a
            # citation key and dataset names, in code, tests and docs.
            "src/y.py": '"""The CONF-1 and R4-1 cases."""\nRE = r"[A-Z0-9-]+"\n',
            "tests/test_y.py": f'"""{SHAPED} is a review label here."""\n',
            "docs/a.md": "CIFAR-10 and COVID-19, cited as P17-1147.\n",
        },
    )
    assert guard.scan_tree(repo) == []
    assert guard.main([str(repo)]) == 0


@pytest.mark.parametrize(
    ("files", "kind"),
    [
        ({"release/receipts/1.0.md": "gate receipt\n"}, "release-path"),
        ({"docs/a.md": f"See {URL} for the plan.\n"}, "linear-url"),
        ({"tests/test_a.py": f'"""Guards the {KEY} fix."""\n'}, "tracker-key"),
        ({"src/a.py": f"# See {'B' + 'D-6'} and {'BEN' + '3-45'}.\n"}, "tracker-key"),
        ({"CHANGELOG.md": f"- Fixed ({SHAPED}).\n"}, "tracker-key"),
        ({"README.md": f"Tracked as {SHAPED}.\n"}, "tracker-key"),
    ],
)
def test_each_kind_of_internal_material_is_refused(tmp_path, files, kind):
    if "README.md" not in files:
        files = {"README.md": "ok\n", **files}
    repo = _repo(tmp_path, files)
    findings = guard.scan_tree(repo)
    assert {f["kind"] for f in findings} == {kind}
    assert guard.main([str(repo)]) == 1


TEAMS = ("EN", "ENG", "BEN3", "BD", "CEA", "OSS")


@pytest.mark.parametrize("team", TEAMS)
@pytest.mark.parametrize(
    "spell",
    [
        lambda team: f"{team}-" + "12",
        lambda team: f"{team.lower()}-" + "12",
        lambda team: f"{team.title()}-" + "12",
        lambda team: f"branch ``{team.lower()}-" + "12-audit-made-up``",
        lambda team: f"branches: [main, {team.lower()}-" + "278-hosted-sdk]",
        lambda team: f"refs/heads/{team}-" + "12-fix",
    ],
    ids=["upper", "lower", "title", "branch", "workflow-filter", "ref"],
)
def test_a_team_key_is_found_in_any_case_and_inside_a_branch_name(team, spell):
    """Lead ruling on PR #146 (M3): branch names are tracker keys. A
    lower-case key in a branch name (docstrings, workflow branch filters,
    README) passed the guard."""
    text = spell(team)
    for path in ("src/a.py", "tests/test_a.py", ".github/workflows/t.yml", "README.md"):
        tokens = guard.tracker_keys(path, text)
        assert [t.lower() for t in tokens if t.lower().startswith(team.lower())] == [
            f"{team.lower()}-" + ("278" if "278" in text else "12")
        ], (path, text, tokens)


@pytest.mark.parametrize(
    "text",
    [
        "CONF-1 and R4-1, review labels",
        "COVID-19, CIFAR-10, P17-1147",
        'RE = r"[a-z]-9"; ALSO = r"[A-Z0-9-]+"; R = "[0-9a-f]-1"',
        "0.10.0rc4.dev0, 1.45.2-0ubuntu1, 5:29.9.3-1~ubuntu.24.04~noble",
        "openai/gpt-oss-120b and gpt-oss-20b",
        "locale en-US, en_12, pen-12, OPEN-12, token-12, BEN-3, EN-12a",
        "SHA-256, UTF-8, ISO-8601",
    ],
    ids=[
        "labels",
        "datasets",
        "regex-ranges",
        "versions",
        "model-names",
        "near-misses",
        "standards",
    ],
)
def test_text_that_only_looks_like_a_key_is_not_one(tmp_path, text):
    assert guard.tracker_keys("src/a.py", text) == []
    repo = _repo(tmp_path, {"README.md": "ok\n", "src/a.py": f"# {text}\n"})
    assert guard.scan_tree(repo) == []


def test_a_finding_never_prints_the_url(tmp_path, capsys):
    repo = _repo(tmp_path, {"docs/a.md": f"{URL}\n"})
    guard.main([str(repo)])
    out = capsys.readouterr()
    assert "linear" + ".app" not in out.out + out.err
    assert "docs/a.md:1: linear-url" in out.out


def test_only_committed_files_are_read(tmp_path):
    repo = _repo(tmp_path, {"README.md": "ok\n"})
    (repo / "notes.md").write_text(f"{KEY}\n")  # untracked: not published
    assert guard.scan_tree(repo) == []


def test_this_tree_would_publish_no_release_record_or_linear_link():
    """The committed tree itself: no ``release/`` path and no linear.app URL
    (release records live in the internal lab repository). Its tracker keys
    are cleaned before the rc is tagged, and public-release.yml refuses the
    tag while any is left."""
    if subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", "-q", "HEAD"],
        capture_output=True,
    ).returncode:
        pytest.skip("not a git checkout: nothing committed to read")
    findings = guard.scan_tree(ROOT)
    assert [f for f in findings if f["kind"] != "tracker-key"] == []


def test_the_public_release_guards_the_tree_before_it_builds():
    workflow = yaml.load(
        (ROOT / ".github/workflows/public-release.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["publish"]["steps"]
    names = [step.get("name") for step in steps]
    guard_step = names.index("Public tree guard")
    assert "tools/public_tree_guard.py ." in steps[guard_step]["run"]
    assert guard_step < names.index("Build distributions")
    assert guard_step < names.index("Publish public release")
