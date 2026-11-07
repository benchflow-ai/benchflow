"""tools/release_leak_gate.py: what it finds, what it lets through, and when it
refuses to decide. The patterns here are synthetic stand-ins for the private
pattern file the release passes in."""

from __future__ import annotations

import datetime as dt
import importlib.util
import io
import json
import re
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_leak_gate", ROOT / "tools/release_leak_gate.py"
)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["release_leak_gate"] = gate
_spec.loader.exec_module(gate)

PATTERNS = {
    "secrets": r"(?i)(?:\b(?:sk-|hf_|AIza)[A-Za-z0-9_./+\-=]*|-----BEGIN[^\r\n]*)",
    "oauth-token": r"1//0[0-9A-Za-z_-]{40,}|\bya29\.[0-9A-Za-z_-]{20,}",
    "credential-file": r"(?:^|/)(?:auth|credentials)\.json$|(?:^|/)\.config/gcloud/",
    "internal-keys": r"\b(?:TEAM)-\d+\b",
    "internal-infra": r"(?i)(?:\bbuildhost-[\w-]+|\bbenchflow-private\b|\bskillsbench\b|gs://[^\s\x22\x27<>`)]+)",
    "personal": r"(?:/home/someone\b[^\s\x22\x27<>`)]*|/(?:home|Users)/[A-Za-z0-9._-]{2,}/[^\s\x22\x27<>`)]*|/root/[^\s\x22\x27<>`)]+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|\bSomeone\b)",
    "partner": r"(?i)\b(?:acmepartner|Initech)\b",
    "partner-code-tokens": r"initech",
    "hard-rule": r"(?i)forbiddenname",
    "process": r"(?i)\b(?:council|founder|audit)\b|\b(?:[A-Z0-9]+-)*(?:CODEX)-[A-Z0-9-]+",
    "internal-review-tag": r"\b(?:[A-Z0-9]+-)*(?:CODEX)-[A-Z0-9-]+",
    "private-ip": r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2})\b",
    "real-run-path": r"tests/fixtures/captured/",
    "trace": r"(?i)(?:trajectory|\.jsonl$)",
}
FUTURE = (dt.date.today() + dt.timedelta(days=30)).isoformat()
PAST = (dt.date.today() - dt.timedelta(days=1)).isoformat()


def _write_wheel(path: Path, files: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return path


def _write_sdist(path: Path, files: dict[str, bytes | str], top="pkg-1.0") -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, body in files.items():
            data = body.encode() if isinstance(body, str) else body
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _run(tmp_path, artifacts, exceptions=(), patterns=PATTERNS, extra=()):
    p = tmp_path / "patterns.json"
    p.write_text(json.dumps(patterns))
    a = tmp_path / "allowlist.json"
    a.write_text(json.dumps({"version": 1, "exceptions": list(exceptions)}))
    report = tmp_path / "report"
    code = gate.main(
        [
            *map(str, artifacts),
            "--patterns",
            str(p),
            "--allowlist",
            str(a),
            "--report-dir",
            str(report),
            *extra,
        ]
        + ([] if "--scanners" in extra else ["--no-scanners"])
    )
    findings = (
        json.loads((report / "findings.json").read_text())
        if (report / "findings.json").exists()
        else {"open": [], "allowlisted": []}
    )
    return code, findings


def _rules(findings):
    return sorted(
        (f["rule"], f["artifact_member_path"], f["line"]) for f in findings["open"]
    )


def _exception(finding, **over):
    entry = {
        "rule": finding["rule"],
        "artifact_member_path": finding["artifact_member_path"],
        "matched_value_sha256": finding["matched_value_sha256"],
        "file_sha256": finding["file_sha256"],
        "reason": "reviewed: synthetic example",
        "approval_reference": "release review",
        "expires_at": FUTURE,
    }
    entry.update(over)
    return entry


def test_findings_are_decided_not_every_candidate(tmp_path):
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl",
        {
            "pkg/a.py": "\n".join(
                [
                    "# see TEAM-12 for why",  # internal key
                    "# the integrity audit records it",  # product word: candidate only
                    "# the founder asked for this",  # narration
                    "# fixed in X1-CODEX-R2-3",  # review tag
                    "initech = [1, 2]",  # a variable spelled like a partner name
                    "# Initech runs this benchmark",  # the name itself
                    "home = '/root/.cache/uv'",  # a container home
                    "where = '/home/someone/work'",  # an operator home
                    "me = 'dev@example.com'; you = 'real.person@corp.io'",
                    "tmp = f'.benchflow-private-{n}'",  # a runtime temp name
                    "repo = 'benchflow-private'",  # the private repo
                    "net = '10.0.0.0/8'; host = '10.1.2.3'",
                    "url = 'gs://{bucket}'; bucket = 'gs://team-data/x'",
                    "fn = hf_hub_download",
                    "key = 'sk-Ab3dEf6hIj9kLm2nOp5q'",
                    "-----BEGIN PUBLIC KEY-----",
                    "SkillsBench is a public benchmark",
                    r"pem = re.compile(r'-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----')",
                    "-----BEGIN OPENSSH PRIVATE KEY-----",
                ]
            ),
        },
    )
    code, findings = _run(tmp_path, [wheel])
    assert code == 1
    assert _rules(findings) == [
        ("internal-infra", "wheel!pkg/a.py", 11),
        ("internal-infra", "wheel!pkg/a.py", 13),
        ("internal-keys", "wheel!pkg/a.py", 1),
        ("internal-review-tag", "wheel!pkg/a.py", 4),
        ("partner", "wheel!pkg/a.py", 6),
        ("personal", "wheel!pkg/a.py", 8),
        ("personal", "wheel!pkg/a.py", 9),
        ("private-ip", "wheel!pkg/a.py", 12),
        ("process", "wheel!pkg/a.py", 3),
        ("process", "wheel!pkg/a.py", 4),
        ("secrets", "wheel!pkg/a.py", 15),
        ("secrets", "wheel!pkg/a.py", 19),
    ]
    personal = [
        f for f in findings["open"] if f["rule"] == "personal" and f["line"] == 9
    ]
    assert "real.person@corp.io" in personal[0]["quote"]
    secret = next(f for f in findings["open"] if f["rule"] == "secrets")
    assert secret["quote"] == "xxxx…"


def test_without_code_tokens_every_partner_match_is_a_finding(tmp_path):
    """The code-token spellings live in the private patterns; without that
    entry the gate fails closed."""
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl", {"pkg/a.py": "initech = [1, 2]\n"}
    )
    assert _run(tmp_path, [wheel])[0] == 0
    strict = {k: v for k, v in PATTERNS.items() if k != "partner-code-tokens"}
    code, findings = _run(tmp_path, [wheel], patterns=strict)
    assert code == 1
    assert _rules(findings) == [("partner", "wheel!pkg/a.py", 1)]


def test_output_names_hashes_never_values(tmp_path, capsys):
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl", {"pkg/a.py": "x = 'acmepartner'\n"}
    )
    assert _run(tmp_path, [wheel])[0] == 1
    out = capsys.readouterr().out
    assert "acmepartner" not in out
    assert re.search(
        r"FINDING blocker partner wheel!pkg/a.py:1 value_sha256=[0-9a-f]{16}", out
    )
    assert "LEAK GATE FAIL" in out


def test_planted_oauth_tokens_and_credential_files_are_blockers(tmp_path, capsys):
    """A Google refresh or access token anywhere, and a login or credentials
    file shipped as a file, fail the gate; a path that only names such a file
    does not, and no output or report carries a token."""
    refresh = "1//0" + "Ab3_x9-Q" * 6
    access = "ya29." + "c0Ab_9xY-" * 3
    # As a JSON writer that escapes slashes leaves it in an auth.json.
    escaped = refresh.replace("/", "\\/")
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl",
        {
            "pkg/a.py": f'TOKEN = "{refresh}"\nHEADER = "Bearer {access}"\n',
            "pkg/b.json": f'{{"refresh_token": "{escaped}"}}\n',
            "pkg/auth.json": "{}",
            "pkg/data/.config/gcloud/configurations/config_default": "[core]\n",
            "pkg/c.py": 'ADC = "~/.config/gcloud/application_default_credentials.json"\n',
        },
    )
    code, findings = _run(tmp_path, [wheel])
    assert code == 1
    assert _rules(findings) == [
        ("credential-file", "wheel!pkg/auth.json", 0),
        (
            "credential-file",
            "wheel!pkg/data/.config/gcloud/configurations/config_default",
            0,
        ),
        ("oauth-token", "wheel!pkg/a.py", 1),
        ("oauth-token", "wheel!pkg/a.py", 2),
        ("oauth-token", "wheel!pkg/b.json", 1),
    ]
    assert {f["severity"] for f in findings["open"]} == {"blocker"}
    out = capsys.readouterr().out
    report = (tmp_path / "report" / "findings.json").read_text()
    for value in (refresh, access, refresh[4:24], access[5:20]):
        assert value not in out and value not in report
    # A pattern file without the two categories cannot pass a release.
    for category in ("oauth-token", "credential-file"):
        partial = {k: v for k, v in PATTERNS.items() if k != category}
        assert _run(tmp_path, [wheel], patterns=partial)[0] == 2


def test_a_home_folder_names_someone_unless_it_is_a_stand_in(tmp_path):
    """A spreadsheet's saved path (/Users/<person>/...) is a finding; the
    sandbox's account and the usual stand-ins are not."""
    workbook = io.BytesIO()
    with zipfile.ZipFile(workbook, "w") as zf:
        zf.writestr(
            "xl/workbook.xml",
            '<x15ac:absPath url="/Users/alice.b/tasks/gdp/environment/"/>',
        )
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl",
        {
            "pkg/data/gdp.xlsx": workbook.getvalue(),
            "pkg/a.py": "\n".join(
                [
                    "# ``/Users/dev/my-project`` becomes ``-Users-dev-my-project``",
                    "HOME = '/home/agent/.cache'",
                    "cfg = '/home/me/.config/x'",
                    "seen = '/home/carol/work/run.log'",
                ]
            ),
        },
    )
    code, findings = _run(tmp_path, [wheel])
    assert code == 1
    assert _rules(findings) == [
        ("personal", "wheel!pkg/a.py", 4),
        ("personal", "wheel!pkg/data/gdp.xlsx!xl/workbook.xml", 1),
    ]


def test_an_exception_binds_the_value_and_the_file(tmp_path):
    wheel_path = tmp_path / "pkg-1.0-py3-none-any.whl"
    _write_wheel(wheel_path, {"pkg/a.py": "# TEAM-7\n"})
    code, findings = _run(tmp_path, [wheel_path])
    assert code == 1
    (finding,) = findings["open"]
    code, findings = _run(tmp_path, [wheel_path], [_exception(finding)])
    assert code == 0 and len(findings["allowlisted"]) == 1
    # The same value in a changed file needs a new approval.
    _write_wheel(wheel_path, {"pkg/a.py": "# TEAM-7\nx = 1\n"})
    assert _run(tmp_path, [wheel_path], [_exception(finding)])[0] == 1
    # An expired exception no longer applies.
    _write_wheel(wheel_path, {"pkg/a.py": "# TEAM-7\n"})
    assert _run(tmp_path, [wheel_path], [_exception(finding, expires_at=PAST)])[0] == 1


@pytest.mark.parametrize(
    "broken",
    [
        {"reason": ""},
        {"approval_reference": None},
        {"file_sha256": "abc"},
        {"expires_at": "soon"},
        {"expires_at": "2030-01-01T00:00:00"},  # no time zone
    ],
)
def test_a_malformed_exception_stops_the_gate(tmp_path, broken):
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl", {"pkg/a.py": "# TEAM-7\n"}
    )
    finding = _run(tmp_path, [wheel])[1]["open"][0]
    assert _run(tmp_path, [wheel], [_exception(finding, **broken)])[0] == 2


def test_missing_or_empty_patterns_stop_the_gate(tmp_path):
    wheel = _write_wheel(tmp_path / "pkg-1.0-py3-none-any.whl", {"pkg/a.py": "x = 1\n"})
    assert _run(tmp_path, [wheel], patterns={})[0] == 2
    partial = {k: v for k, v in PATTERNS.items() if k != "partner"}
    assert _run(tmp_path, [wheel], patterns=partial)[0] == 2
    p = tmp_path / "patterns.json"
    p.write_text("")
    a = tmp_path / "allowlist.json"
    a.write_text(json.dumps({"version": 1, "exceptions": []}))
    assert (
        gate.main(
            [str(wheel), "--patterns", str(p), "--allowlist", str(a), "--no-scanners"]
        )
        == 2
    )


def test_an_unknown_category_is_strict(tmp_path):
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl", {"pkg/a.py": "x = 'zeta-word'\n"}
    )
    code, findings = _run(
        tmp_path, [wheel], patterns={**PATTERNS, "extra": r"zeta-word"}
    )
    assert code == 1 and _rules(findings) == [("extra", "wheel!pkg/a.py", 1)]


def test_authorship_fields_are_public_and_other_names_are_not(tmp_path):
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl",
        {
            "pkg-1.0.dist-info/METADATA": "Name: pkg\nAuthor-email: Someone <someone@corp.io>\n",
            "pkg/a.py": "# thanks to Someone\n",
        },
    )
    sdist = _write_sdist(
        tmp_path / "pkg-1.0.tar.gz",
        {
            "pyproject.toml": '[project]\nauthors = [\n    { name = "Someone", email = "someone@corp.io" },\n]\n',
            "src/pkg/a.py": "x = 1\n",
        },
    )
    folder = tmp_path / "site"
    (folder / "task").mkdir(parents=True)
    (folder / "task" / "task.md").write_text(
        "---\nmetadata:\n  author_name: Someone\n  author_email: someone@corp.io\n---\n"
        "Write to someone@corp.io.\n"
    )
    (folder / "task" / "task.toml").write_text('author_email = "someone@corp.io"\n')
    code, findings = _run(tmp_path, [wheel, sdist, f"docs={folder}"])
    assert code == 1
    assert _rules(findings) == [
        ("personal", "docs!task/task.md", 6),
        ("personal", "wheel!pkg/a.py", 1),
    ]


def test_member_paths_are_normalized_and_scanned(tmp_path):
    sdist = _write_sdist(
        tmp_path / "pkg-1.0.tar.gz",
        {"tests/fixtures/captured/run.json": "{}", "src/pkg/a.py": "x = 1\n"},
    )
    code, findings = _run(tmp_path, [sdist])
    assert code == 1
    assert _rules(findings) == [
        ("real-run-path", "sdist!tests/fixtures/captured/run.json", 0)
    ]
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl",
        {"pkg-1.0.dist-info/METADATA": "x\n", "pkg/data/log.jsonl": "{}\n"},
    )
    code, findings = _run(tmp_path, [wheel])
    assert _rules(findings) == [("trace-file", "wheel!pkg/data/log.jsonl", 0)]


def test_nested_archives_and_encoded_text_are_read(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("docProps/core.xml", "<x>TEAM-3</x>")
    folder = tmp_path / "site"
    (folder / "assets").mkdir(parents=True)
    (folder / "assets" / "book.xlsx").write_bytes(inner.getvalue())
    (folder / "index.html").write_text(
        "<a>acme&#112;artner</a>\n<a href='x?q=TEAM%2D9'>\n"
    )
    code, findings = _run(tmp_path, [f"docs={folder}"])
    assert code == 1
    assert _rules(findings) == [
        ("internal-keys", "docs!assets/book.xlsx!docProps/core.xml", 1),
        ("internal-keys", "docs!index.html", 2),
        ("partner", "docs!index.html", 1),
    ]


def test_binaries_are_read_as_strings_and_unknown_kinds_are_findings(tmp_path):
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl",
        {
            "pkg/bin/tool": b"\x7fELF\x02\x01\x00\x00"
            + b"\x00" * 8
            + b"/home/someone/src\x00",
            "pkg/blob.bin": b"\x00\x01\x02 opaque \xff\xfe",
        },
    )
    code, findings = _run(tmp_path, [wheel])
    assert code == 1
    rules = {(f["rule"], f["artifact_member_path"]) for f in findings["open"]}
    assert rules == {
        ("personal", "wheel!pkg/bin/tool"),
        ("unknown-binary", "wheel!pkg/blob.bin"),
    }


def _zip_with(name: str, *, symlink: bool = False) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        info = zipfile.ZipInfo(name)
        if symlink:
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "target")
    return out.getvalue()


@pytest.mark.parametrize(
    "body",
    [
        _zip_with("../escape.py"),
        _zip_with("/abs.py"),
        _zip_with("pkg/link", symlink=True),
    ],
)
def test_an_unsafe_archive_stops_the_gate(tmp_path, body):
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    wheel.write_bytes(body)
    assert _run(tmp_path, [wheel])[0] == 2


def test_a_tar_link_and_a_docs_symlink_stop_the_gate(tmp_path):
    sdist = tmp_path / "pkg-1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as tf:
        info = tarfile.TarInfo("pkg-1.0/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    assert _run(tmp_path, [sdist])[0] == 2
    folder = tmp_path / "site"
    folder.mkdir()
    (folder / "page.md").symlink_to(tmp_path / "patterns.json")
    assert _run(tmp_path, [f"docs={folder}"])[0] == 2


def test_archives_nested_too_deep_stop_the_gate(tmp_path):
    body = b"x = 1\n"
    for _ in range(gate.MAX_DEPTH + 1):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as zf:
            zf.writestr("inner.zip", body)
        body = out.getvalue()
    wheel = _write_wheel(tmp_path / "pkg-1.0-py3-none-any.whl", {"pkg/deep.zip": body})
    assert _run(tmp_path, [wheel])[0] == 2


def test_a_record_digest_is_not_a_key(tmp_path):
    digest = "sha256=AIzaQ1w2E3r4T5y6U7i8O9p0AsDfGhJkL1"
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl",
        {
            "pkg-1.0.dist-info/RECORD": f"pkg/a.py,{digest},6\n",
            "pkg/a.py": "k = 'AIzaQ1w2E3r4T5y6U7i8O9p0AsDfGhJkL1'\n",
        },
    )
    code, findings = _run(tmp_path, [wheel])
    assert code == 1
    assert _rules(findings) == [("secrets", "wheel!pkg/a.py", 1)]


def _fake_scanners(
    tmp_path: Path, *, gitleaks_rc=0, trufflehog_rc=0, versions=None
) -> Path:
    versions = versions or (gate.GITLEAKS_VERSION, gate.TRUFFLEHOG_VERSION)
    folder = tmp_path / "scanners"
    folder.mkdir()
    gitleaks = folder / "gitleaks"
    gitleaks.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, pathlib\n"
        f"if sys.argv[1] == 'version': print({versions[0]!r}); sys.exit(0)\n"
        "target = pathlib.Path(sys.argv[2]); report = sys.argv[sys.argv.index('--report-path') + 1]\n"
        "rows = [{'RuleID': 'fake-key', 'File': str(p), 'StartLine': 1, 'Secret': 'S3CRETVALUE'}\n"
        "        for p in target.rglob('*') if p.is_file() and b'S3CRETVALUE' in p.read_bytes()]\n"
        "pathlib.Path(report).write_text(json.dumps(rows))\n"
        f"sys.exit({gitleaks_rc} if {gitleaks_rc} else (3 if rows else 0))\n"
    )
    trufflehog = folder / "trufflehog"
    trufflehog.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"if sys.argv[1] == '--version': print('trufflehog', {versions[1]!r}); sys.exit(0)\n"
        f"sys.exit({trufflehog_rc})\n"
    )
    for f in (gitleaks, trufflehog):
        f.chmod(0o755)
    return folder


def test_scanner_detections_are_findings_mapped_to_members(tmp_path, monkeypatch):
    wheel = _write_wheel(
        tmp_path / "pkg-1.0-py3-none-any.whl", {"pkg/a.py": "x = 'S3CRETVALUE'\n"}
    )
    _fake_scanners(tmp_path)
    monkeypatch.chdir(tmp_path)  # a relative --scanners, as a shell passes it
    code, findings = _run(tmp_path, [wheel], extra=("--scanners", "scanners"))
    assert code == 1
    assert _rules(findings) == [("gitleaks:fake-key", "wheel!pkg/a.py", 1)]
    assert findings["open"][0]["quote"] == "xxxx…"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gitleaks_rc": 1},
        {"trufflehog_rc": 1},
        {"versions": ("8.0.0", gate.TRUFFLEHOG_VERSION)},
    ],
)
def test_a_failing_or_unpinned_scanner_stops_the_gate(tmp_path, kwargs):
    wheel = _write_wheel(tmp_path / "pkg-1.0-py3-none-any.whl", {"pkg/a.py": "x = 1\n"})
    scanners = _fake_scanners(tmp_path, **kwargs)
    assert _run(tmp_path, [wheel], extra=("--scanners", str(scanners)))[0] == 2


def test_the_installer_pins_the_versions_the_gate_checks():
    script = (ROOT / "tools/install_leak_scanners.sh").read_text()
    assert f"GITLEAKS_VERSION={gate.GITLEAKS_VERSION}\n" in script
    assert f"TRUFFLEHOG_VERSION={gate.TRUFFLEHOG_VERSION}\n" in script
    assert re.search(r"GITLEAKS_SHA256=[0-9a-f]{64}\n", script)
    assert re.search(r"TRUFFLEHOG_SHA256=[0-9a-f]{64}\n", script)


def test_the_public_release_runs_the_gate_before_publishing():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/public-release.yml").read_text()
    )
    steps = workflow["jobs"]["publish"]["steps"]
    names = [s.get("name") for s in steps]
    gate_step = steps[names.index("Leak gate on the wheel, the sdist and the docs")]
    assert names.index("Build distributions") < names.index(
        "Install the leak scanners (pinned)"
    )
    assert names.index("Leak gate on the wheel, the sdist and the docs") < names.index(
        "Publish public release"
    )
    assert names.index("Leak gate on the wheel, the sdist and the docs") < names.index(
        "Create or update GitHub release"
    )
    run = gate_step["run"]
    assert "--scanners" in run and "--no-scanners" not in run
    assert "dist/*.whl" in run and "dist/*.tar.gz" in run and "docs=docs" in run
    assert set(gate_step["env"]) == {"RELEASE_LEAK_PATTERNS", "RELEASE_LEAK_ALLOWLIST"}


def test_the_sdist_ships_no_tests_or_docs():
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    include = config["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    assert sorted(include) == [
        "CHANGELOG.md",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "src",
    ]
