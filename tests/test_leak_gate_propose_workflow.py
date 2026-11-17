"""leak-gate-propose.yml regenerates the leak gate's allowlist without showing the patterns.

The job builds a tag as public-release.yml does and runs the gate with the
real patterns against an empty allowlist. These tests run the job's own
"Propose exceptions" script on planted artifacts, with stand-in scanners, and
check that neither a pattern nor a matched value (from a pattern or from a
scanner) reaches the log or the uploaded proposal; that a draft carrying any
text, or a gate that stops with a message, ends the step without echoing it;
and that the build steps are public-release.yml's.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "leak-gate-propose.yml"
RELEASE = ROOT / ".github" / "workflows" / "public-release.yml"
GATE = ROOT / "tools" / "release_leak_gate.py"

PARTNER_VALUE = "Plantedpartnerco"
SCANNER_SECRET = "PLANTED-SCANNER-SECRET-7f3a91c2"
MARKER = "qzpatternmarker"
REQUIRED = (
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
PATTERNS = {c: rf"\b{MARKER}{c.replace('-', '')}\b" for c in REQUIRED}
PATTERNS["partner"] = rf"(?i)\bplantedpartner\w*\b|\b{MARKER}partner\b"
PATTERNS_JSON = json.dumps(PATTERNS)
NEVER_SHOWN = (
    PARTNER_VALUE,
    PARTNER_VALUE.lower(),
    SCANNER_SECRET,
    MARKER,
    PATTERNS_JSON,
)


def _load(path: Path) -> dict:
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def _steps(path: Path, job: str) -> list[dict]:
    return _load(path)["jobs"][job]["steps"]


def _step(steps: list[dict], name: str) -> dict:
    return next(s for s in steps if s.get("name") == name)


def _propose_script() -> str:
    return _step(_steps(WORKFLOW, "propose"), "Propose exceptions")["run"]


def _executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _workspace(tmp_path: Path, gate_source: str | None = None) -> tuple[Path, dict]:
    """A checked-out tag with built distributions, docs, the gate and stand-in tools."""
    work = tmp_path / "work"
    (work / "tools").mkdir(parents=True)
    (work / "dist").mkdir()
    (work / "docs").mkdir()
    if gate_source is None:
        shutil.copy(GATE, work / "tools" / "release_leak_gate.py")
    else:
        (work / "tools" / "release_leak_gate.py").write_text(gate_source)
    files = {
        "pkg/__init__.py": f"# built with {PARTNER_VALUE} tooling\n",
        "pkg/keys.py": f'TOKEN = "{SCANNER_SECRET}"\n',
        "pkg-1.0.dist-info/METADATA": "Name: pkg\nVersion: 1.0\n",
    }
    with zipfile.ZipFile(work / "dist" / "pkg-1.0-py3-none-any.whl", "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    with tarfile.open(work / "dist" / "pkg-1.0.tar.gz", "w:gz") as tf:
        for name, body in files.items():
            data = body.encode()
            info = tarfile.TarInfo(f"pkg-1.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    (work / "docs" / "page.md").write_text(f"Thanks to {PARTNER_VALUE}.\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    subprocess.run([*git, "-C", str(work), "add", "-A"], check=True)
    subprocess.run([*git, "-C", str(work), "commit", "-qm", "tag"], check=True)

    runner = tmp_path / "runner"
    scanners = runner / "leak-scanners"
    scanners.mkdir(parents=True)
    _executable(
        scanners / "gitleaks",
        f"""#!{sys.executable}
import json, os, sys
args = sys.argv[1:]
if args[:1] == ["version"]:
    print("8.30.1")
    sys.exit(0)
target, report = args[1], args[args.index("--report-path") + 1]
rows = []
for folder, _, names in os.walk(target):
    for name in names:
        path = os.path.join(folder, name)
        if {SCANNER_SECRET!r} in open(path, errors="ignore").read():
            rows.append({{"RuleID": "generic-api-key", "File": path, "StartLine": 1,
                          "Secret": {SCANNER_SECRET!r}, "Match": {SCANNER_SECRET!r}}})
open(report, "w").write(json.dumps(rows))
sys.exit(3 if rows else 0)
""",
    )
    _executable(
        scanners / "trufflehog",
        f"""#!{sys.executable}
import sys
if sys.argv[1:] == ["--version"]:
    print("trufflehog 3.99.8")
""",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # `uv run --no-project python X` as the runner's uv runs it: this interpreter.
    _executable(
        bin_dir / "uv",
        f"""#!/bin/sh
[ "$1" = run ] && shift
[ "$1" = --no-project ] && shift
[ "$1" = python ] && shift
exec {sys.executable} "$@"
""",
    )
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(runner),
        "RELEASE_LEAK_PATTERNS": PATTERNS_JSON,
        "TAG_NAME": "v0.0.0",
    }
    return work, env


def _run_step(work: Path, env: dict) -> tuple[int, str]:
    # GitHub runs a `run:` step as `bash --noprofile --norc -eo pipefail {0}`.
    done = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _propose_script()],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return done.returncode, done.stdout + done.stderr


def _assert_nothing_shown(text: str) -> None:
    for secret in NEVER_SHOWN:
        assert secret not in text


def test_the_proposal_holds_rules_members_and_hashes_only(tmp_path):
    work, env = _workspace(tmp_path)
    code, log = _run_step(work, env)
    assert code == 0, log
    proposal_text = (work / "proposal.json").read_text()
    _assert_nothing_shown(log)
    _assert_nothing_shown(proposal_text)

    proposal = json.loads(proposal_text)
    assert set(proposal) == {
        "version",
        "tag",
        "commit",
        "artifacts_sha256",
        "exceptions",
    }
    assert proposal["tag"] == "v0.0.0"
    assert re.fullmatch(r"[0-9a-f]{40}", proposal["commit"])
    assert set(proposal["artifacts_sha256"]) == {"wheel", "sdist", "docs"}
    for entry in proposal["exceptions"]:
        assert set(entry) == {
            "rule",
            "artifact_member_path",
            "matched_value_sha256",
            "file_sha256",
        }
        assert re.fullmatch(r"[0-9a-f]{64}", entry["matched_value_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", entry["file_sha256"])
    rules = {(e["rule"], e["artifact_member_path"]) for e in proposal["exceptions"]}
    assert ("partner", "docs!page.md") in rules
    assert ("partner", "wheel!pkg/__init__.py") in rules
    # The sdist's member paths drop its top folder, as on the real gate.
    assert ("partner", "sdist!pkg/__init__.py") in rules
    assert any(rule == "gitleaks:generic-api-key" for rule, _ in rules)

    # The gate's count lines and the counts per rule are printed, and the
    # patterns' file is gone.
    assert re.search(r"^findings: ", log, re.M)
    assert re.search(r"^proposal: 5 exception drafts$", log, re.M)
    assert re.search(r"^  partner: 3$", log, re.M)
    assert re.search(r"^  gitleaks:generic-api-key: 2$", log, re.M)
    assert not (tmp_path / "runner" / "leak-gate" / "patterns.json").exists()


def test_a_draft_that_carries_text_is_refused_without_showing_it(tmp_path):
    tampered = f"""import json, sys
args = sys.argv[1:]
drafts = args[args.index("--propose") + 1]
open(drafts, "w").write(json.dumps([{{
    "rule": "partner", "artifact_member_path": "wheel!pkg/__init__.py",
    "matched_value_sha256": "{"a" * 64}", "file_sha256": "{"b" * 64}",
    "reason": "", "approval_reference": "", "expires_at": "", "quote": {PARTNER_VALUE!r}}}]))
print("artifact wheel {"c" * 64} members=1 (dist/x.whl)")
sys.exit(1)
"""
    work, env = _workspace(tmp_path, gate_source=tampered)
    code, log = _run_step(work, env)
    assert code != 0
    assert "proposal refused" in log
    _assert_nothing_shown(log)
    assert not (work / "proposal.json").exists()


def test_a_gate_that_stops_with_a_message_never_echoes_it(tmp_path):
    stopping = f"""import sys
print("LEAK GATE ERROR: patterns: {PARTNER_VALUE} " + open(sys.argv[sys.argv.index("--patterns") + 1]).read())
sys.exit(2)
"""
    work, env = _workspace(tmp_path, gate_source=stopping)
    code, log = _run_step(work, env)
    assert code != 0
    assert "message is withheld" in log
    _assert_nothing_shown(log)
    assert not (work / "proposal.json").exists()


def test_the_job_builds_the_tag_as_the_release_does():
    mine = _steps(WORKFLOW, "propose")
    release = _steps(RELEASE, "publish")
    for name in (
        "Install uv",
        "Set up Python",
        "Install the bfx toolchain",
        "Build bfx (pinned)",
        "Build distributions",
        "Check bfx in distributions",
        "Install the leak scanners (pinned)",
    ):
        a, b = _step(mine, name), _step(release, name)
        assert {k: a.get(k) for k in ("uses", "with", "run")} == {
            k: b.get(k) for k in ("uses", "with", "run")
        }, name
    checkout = _step(mine, "Check out tag")
    assert checkout["uses"] == _step(release, "Check out tag")["uses"]
    assert checkout["with"]["ref"] == "refs/tags/${{ inputs.tag }}"
    assert checkout["with"]["persist-credentials"] == "false"
    names = [s.get("name") for s in mine]
    assert names.index("Check the tag input") < names.index("Check out tag")
    assert names.index("Install the leak scanners (pinned)") < names.index(
        "Propose exceptions"
    )


def test_the_job_reads_the_patterns_once_and_uploads_only_the_proposal():
    text = WORKFLOW.read_text()
    loaded = _load(WORKFLOW)
    assert set(loaded["on"]) == {"workflow_dispatch"}
    assert loaded["on"]["workflow_dispatch"]["inputs"]["tag"]["required"] == "true"
    assert loaded["permissions"] == {"contents": "read"}
    job = loaded["jobs"]["propose"]
    assert job["environment"] == "pypi-public"
    assert re.findall(r"secrets\.[A-Z_]+", text) == ["secrets.RELEASE_LEAK_PATTERNS"]
    propose = _step(job["steps"], "Propose exceptions")
    assert (
        propose["env"]["RELEASE_LEAK_PATTERNS"]
        == "${{ secrets.RELEASE_LEAK_PATTERNS }}"
    )
    assert "--report-dir" not in text  # the gate's report keeps values and quotes
    for step in job["steps"]:
        assert "${{" not in step.get("run", ""), step.get(
            "name"
        )  # inputs reach scripts as env only
    upload = _step(job["steps"], "Upload the proposal")
    assert upload["with"]["path"] == "proposal.json"
    assert upload["with"]["if-no-files-found"] == "error"


@pytest.mark.parametrize("tag", ["v0.9.3", "v0.10.0rc5", "v1.2.3b1"])
def test_release_tags_pass_the_input_check(tag):
    script = _step(_steps(WORKFLOW, "propose"), "Check the tag input")["run"]
    done = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        env={"TAG_NAME": tag},
        capture_output=True,
    )
    assert done.returncode == 0


@pytest.mark.parametrize("tag", ["main", "v0.9.3; echo hi", "refs/tags/v0.9.3", "v0.9"])
def test_other_refs_fail_the_input_check(tag):
    script = _step(_steps(WORKFLOW, "propose"), "Check the tag input")["run"]
    done = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        env={"TAG_NAME": tag},
        capture_output=True,
    )
    assert done.returncode != 0
