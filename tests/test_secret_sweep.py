"""The push-protection sweep (tools/secret_sweep.py).

GitHub push protection refused 0.10.0rc4's public push on a synthetic Stripe
live key in a test file: the release leak gate reads only the wheel, the sdist
and the docs, and tests are in none of them. The sweep reads what push
protection reads: every commit in the pushed range, and the tree. Every key
shape in this file is built at runtime, so this file carries none.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import ghpp_patterns
from tools.secret_sweep import (
    SweepError,
    apply_allowlist,
    load_allowlist,
    main,
    scan_range,
    scan_tree,
)

# Built at runtime: no source line holds the shape.
STRIPE = "sk" + "_live_" + "Ab3dE5gH" * 3
GITHUB = "gh" + "p_" + "q7Rk2mW9" * 5
AWS_ID = "AK" + "IA" + "QWERTYUIOP" + "ASDFGH"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("clean\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "base")
    return r


def _commit(repo: Path, name: str, text: str, message: str = "c") -> str:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_the_shapes_here_are_what_push_protection_refuses():
    for value, rule in (
        (STRIPE, "stripe-secret-live"),
        (GITHUB, "github-token"),
        (AWS_ID, "aws-access-key-id"),
    ):
        assert rule in {r for r, *_ in ghpp_patterns.scan_line(value)}, rule


def test_a_shape_in_the_tree_is_found(repo):
    _commit(repo, "tests/test_x.py", f'VALUE = "{STRIPE}"\n')
    hits = scan_tree(repo, "HEAD")
    assert [(h.rule, h.path, h.line) for h in hits] == [
        ("stripe-secret-live", "tests/test_x.py", 1)
    ]


def test_a_shape_added_then_removed_inside_the_range_is_found(repo):
    """Push protection reads each pushed commit, not only the last tree."""
    base = _git(repo, "rev-parse", "HEAD")
    added = _commit(repo, "a.py", f'TOKEN = "{GITHUB}"\n', "add")
    _commit(repo, "a.py", "TOKEN = None\n", "remove")
    assert scan_tree(repo, "HEAD") == []
    hits = scan_range(repo, base, "HEAD")
    assert [(h.rule, h.path, h.line, h.commit) for h in hits] == [
        ("github-token", "a.py", 1, added)
    ]


def test_a_shape_split_in_the_source_is_not_found(repo):
    _commit(repo, "b.py", 'VALUE = "sk" + "_live_" + "Ab3dE5gH" * 3\n')
    assert scan_tree(repo, "HEAD") == []


def test_the_output_never_holds_the_value(repo, capsys):
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "c.py", f'ID = "{AWS_ID}"\nK = "{STRIPE}"\n')
    code = main(["--repo", str(repo), "--base", base, "--tip", "HEAD", "--no-scanners"])
    out = capsys.readouterr()
    assert code == 1
    text = out.out + out.err
    assert AWS_ID not in text and STRIPE not in text
    assert "aws-access-key-id" in text and "c.py:1" in text


def test_a_provider_pattern_cannot_be_allowlisted(tmp_path):
    allow = tmp_path / "allow.toml"
    allow.write_text(
        '[[allow]]\ntool = "ghpp"\nrule = "stripe-secret-live"\npath = "a.py"\n'
        'value_sha256 = "' + "0" * 64 + '"\nreason = "test"\nexpires = "2099-01-01"\n'
    )
    with pytest.raises(SweepError, match="push protection"):
        load_allowlist(allow)


def test_a_generic_scanner_hit_is_allowlisted_by_its_value_hash(tmp_path):
    from tools.secret_sweep import Hit, sha256_text

    hit = Hit("gitleaks", "generic-api-key", "x.py", 3, "", sha256_text("abc"))
    allow = tmp_path / "allow.toml"
    allow.write_text(
        '[[allow]]\ntool = "gitleaks"\nrule = "generic-api-key"\npath = "x.py"\n'
        f'value_sha256 = "{sha256_text("abc")}"\nreason = "a fixture"\n'
        'expires = "2099-01-01"\n'
    )
    open_hits, allowed = apply_allowlist([hit], load_allowlist(allow))
    assert open_hits == [] and allowed == [hit]
    other = Hit("gitleaks", "generic-api-key", "x.py", 3, "", sha256_text("abd"))
    assert apply_allowlist([other], load_allowlist(allow))[0] == [other]


def test_an_expired_exception_no_longer_holds(tmp_path):
    from tools.secret_sweep import Hit, sha256_text

    hit = Hit("trufflehog", "URI", "y.json", 1, "", sha256_text("v"))
    allow = tmp_path / "allow.toml"
    allow.write_text(
        '[[allow]]\ntool = "trufflehog"\nrule = "URI"\npath = "y.json"\n'
        f'value_sha256 = "{sha256_text("v")}"\nreason = "r"\nexpires = "2001-01-01"\n'
    )
    assert apply_allowlist([hit], load_allowlist(allow))[0] == [hit]


def test_the_rc4_refused_shape_is_found(repo):
    """Recall: the shape push protection refused in 0.10.0rc4's range, a
    synthetic Stripe live key in a redaction test, rebuilt here at runtime."""
    base = _git(repo, "rev-parse", "HEAD")
    line = 'CASES = [("stripe", "' + "sk" + "_live_" + "51Hx" + "Q" * 20 + '")]\n'
    _commit(repo, "tests/test_redaction_values.py", line)
    rules = {h.rule for h in scan_range(repo, base, "HEAD")}
    assert "stripe-secret-live" in rules


def test_this_tree_has_no_provider_shaped_literal():
    """The rule since rc4: no provider-shaped literal in any tracked file.
    Build test key shapes at runtime (tools/keyshape_fix.py splits them)."""
    root = Path(__file__).resolve().parents[1]
    hits = scan_tree(root, "HEAD")
    assert hits == [], [(h.rule, h.path, h.line) for h in hits]


def test_keyshape_fix_splits_a_literal_without_changing_its_value():
    from tools.foldproof import folded
    from tools.keyshape_fix import split_source

    src = f'VALUE = "{STRIPE}"\n'
    fixed = split_source(src)
    assert not any(ghpp_patterns.scan_line(line) for line in fixed.splitlines())
    assert folded(fixed) == folded(src)
    scope: dict = {}
    exec(compile(fixed, "fixed.py", "exec"), scope)
    assert scope["VALUE"] == STRIPE
