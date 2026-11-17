#!/usr/bin/env python3
"""The release-candidate gate, on per-run VMs (the 0.10 line has no CI).

    # one VM per leg of tools/release_gate/legs.json, in parallel:
    python3 tools/release_gate/run_gate.py --leg py3.12 --records LAB/ops/releases/benchflow \
        [--base-commit SHA] --secrets DIR --gcs-prefix gs://BUCKET/PATH
    python3 tools/release_gate/run_gate.py --leg py3.13 --records ... --gcs-prefix ...
    python3 tools/release_gate/run_gate.py --leg docker-ce --records ... --gcs-prefix ...
    # then, with every leg's receipt committed in the records folder:
    python3 tools/release_gate/run_gate.py --combine --records LAB/ops/releases/benchflow

Each leg runs from a clean checkout of the candidate on a fresh Ubuntu 24.04
VM prepared by tools/release_gate/bootstrap_vm.sh (``--docker ce`` for the
docker-ce leg), detached (nohup). Per its stages it:

- syncs the locked environment on the leg's uv-managed Python, as test.yml
  does, then runs its static checks (ruff check, ruff format --check, ty);
- runs the unit suite (``pytest -n auto``, deterministic tier off), and on
  --base-commit (or a --base-failures list) so the receipt names which
  failures are new;
- runs pip-audit on the lock with every extra (test.yml audits three) and on
  each agent pin file (src/benchflow/agents/requirements/*.txt, which the
  lock does not hold), once with PyPI's advisories and once with OSV's, and
  lists each file and service with when its advisories were read;
- runs the Docker tier (tests marked ``deterministic``, on Docker), then
  tests/test_hillclimb_example_docker.py alone with ``--timeout 2400``;
- runs the cost matrix (``tools/cost_matrix/run_matrix.py``) with --secrets;
- scrubs the secrets' values from every log, uploads the logs to
  --gcs-prefix and checks each object's MD5, then writes its receipt,
  ``receipts/<version>-<commit12>-<leg>.md`` and ``.json``, into the records
  folder: the internal lab repository's ``ops/releases/benchflow``, in a
  checkout of lab (any other folder or repository is refused), never this
  repository, whose tree goes public.

The gate runs in a fixed environment, so every failure there fails it: a test
that cannot run on the VM is skipped by a reason in the test itself, or the
VM is fixed. The one way past a failing test is a waiver the lead writes by
name in the records folder's ``waivers.json`` (for this version, and for one
leg if it names one); the gate reads that file and never writes it, and reads
it only as committed. ``<version>-failures.json`` there, the record of why
tests failed on machines that were not prepared, is listed in the receipt and
decides nothing.

A release gate runs on the rc branch's head after every merge, never on a PR
alone: two PRs that each pass can fail together once merged (one removed an
import the other used). A leg's receipt is a release gate only with
--rc-branch BRANCH and the tested commit that branch's head (as fetched here,
and on GitHub when it answers); it names the branch, its head and every PR
merged since the last rc (the highest version among the accepted records
in .github/release-acceptance/ whose source commit is an ancestor). Any other run
(a PR branch, a scratch merge) may run, and its receipt says it is not a
release gate and is never a PASS. The static stage also checks the pages
generated from the code (the reference pages and the changelog page, the
error index, the CLI help golden) on the tested head, and names the command
that regenerates a stale one.

A leg passes when every one of its stages ran on the whole suite and passed:
no unit failure without a waiver; pip-audit clean; the Docker tier without a
failure (each skip listed with its reason) and the hill-climb test passed (a
skip of it fails); every matrix run passed on both sandboxes, with each
BLOCKED cell listed (never a pass, never a failure); every log uploaded and
verified. A stage left out (--skip), a subset (--unit-select,
--docker-select, --matrix-sandbox), no upload or --dry-run make it
INCOMPLETE. --combine passes only when every leg's receipt passes, and adds
the one-line public summary (version, commit, unit counts per leg, date) for
the public release notes. Exit 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 refused to
start. ``<work>/FINISHED`` holds the exit code.

Logs stay out of git: they are written under --work and uploaded. Every
command gets HARBOR_TELEMETRY=off and an environment built from a short list
of host variables, so no credential in the VM's environment variables reaches
the tests. HOME is still the VM user's, whose files (gcloud's credentials
among them) a test could read, and the VM's metadata server answers: give the
VM a service account that may only write to the logs bucket. Secrets reach
only the matrix, as the files in --secrets, never on a command line, and are
scrubbed from every uploaded file and from the receipt.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HILLCLIMB = "tests/test_hillclimb_example_docker.py"
# test.yml's unit job: the extras it syncs and its static checks.
EXTRAS = ("dev", "sandbox-daytona", "sandbox-modal")
# The agents' own pin files, installed into sandboxes, not in the lock.
AGENT_REQUIREMENTS = "src/benchflow/agents/requirements"
# pip-audit's advisory services, each asked about every audited file: PyPI's
# (pip-audit's default) missed CVE-2026-118667 on langgraph 1.2.4, which OSV
# has, so a file passes only when neither finds an advisory.
PIP_AUDIT_SERVICES = ("pypi", "osv")
STATIC_CHECKS = (
    ("ruff-check", ("ruff", "check", "src", "tests", "tools")),
    ("ruff-format", ("ruff", "format", "--check", "src", "tests", "tools")),
    ("ty-check", ("ty", "check")),
)
# Pages generated from the code, checked on the tested head (a merge of two
# PRs that each regenerated them can still leave one stale): what, the check,
# and the command that regenerates it (each tool's REGENERATE).
GENERATED_CHECKS = (
    (
        "the reference pages and the changelog page (docs/reference, docs/versions/changelog.md)",
        ("python", "tools/docs_reference.py", "--check"),
        "uv run python tools/docs_reference.py --write",
    ),
    (
        "the error index (docs/reference/errors.md)",
        ("python", "tools/docs_errors.py", "--check"),
        "uv run python tools/docs_errors.py --write",
    ),
    (
        "the CLI help golden",
        ("python", "tools/cli_help_golden.py", "--check"),
        "uv run python tools/cli_help_golden.py --write",
    ),
)
ACCEPTANCE_RECORDS = ".github/release-acceptance"
# A merge GitHub made: "Merge pull request #12 from ..." or a squash "... (#12)".
_PR_SUBJECT = re.compile(r"^Merge pull request #(\d+)\b|\(#(\d+)\)$")
DOCKER_MARKERS = "deterministic and not live and not integration and not e2e"
# Host variables a command may see; everything else (keys, tokens) stays out.
PASSTHROUGH = (
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
)
STAGES = ("static", "unit", "pip-audit", "docker", "hillclimb", "matrix")
LEGS_FILE = Path(__file__).resolve().parent / "legs.json"
STAGE_TIMEOUT_SEC = {
    "setup": 3600,
    "static": 1800,
    "unit": 4 * 3600,
    "pip-audit": 1800,
    "docker": 3 * 3600,
    "hillclimb": 2700,
    "matrix": 4 * 3600,
}
# Display names for a matrix auth, for the release note.
AUTH_NAMES = {
    "claude": "Claude",
    "codex": "Codex",
    "gemini": "Gemini",
    "mimo": "MiMo",
    "openrouter": "OpenRouter",
}
# The receipt's order: the base's run beside the candidate's.
ORDER = (
    "setup",
    "static",
    "unit",
    "unit-base",
    "pip-audit",
    "docker",
    "hillclimb",
    "matrix",
)
EXIT_PASS, EXIT_FAIL, EXIT_REFUSED, EXIT_INCOMPLETE = 0, 1, 2, 3
# GitHub's runners create files with umask 022. Ubuntu's login shells give a
# user 002, and then every tmp_path file is group-writable: the verifier PATH
# hardening drops such folders, and 165 unit tests failed for that alone.
UMASK = 0o022


class Refused(Exception):
    """The gate cannot start: nothing has run."""


# ---------------------------------------------------------------------------
# Test results
# ---------------------------------------------------------------------------


def _case_id(case: ET.Element) -> str:
    file = case.get("file", "")
    classname = case.get("classname", "")
    name = case.get("name", "")
    if not file:  # a collection error names only the module
        return ".".join(p for p in (classname, name) if p)
    module = file.removesuffix(".py").replace("/", ".")
    classes = (
        classname[len(module) + 1 :].split(".")
        if classname.startswith(module + ".")
        else []
    )
    return "::".join([file, *classes, name])


def junit_outcomes(path: Path) -> dict[str, str] | None:
    """Each test id's outcome in a pytest xunit1 report: passed, failed or
    skipped (an error is failed; an xfail is skipped). None when the report
    is missing or unreadable.

    The id is ``path::Class::name``, the form ``pytest -rf`` prints and
    tools/ci/fresh_resolve_known_failures.txt uses.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    outcomes: dict[str, str] = {}
    for case in root.iter("testcase"):
        test_id = _case_id(case)
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        # A test that passed its call and failed its teardown is reported
        # twice; any failed report fails it.
        if outcomes.get(test_id) != "failed":
            outcomes[test_id] = outcome
    return outcomes


def junit_skip_reasons(path: Path) -> dict[str, str]:
    """Each skipped test's reason, as pytest reports it."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return {}
    reasons = {}
    for case in root.iter("testcase"):
        skipped = case.find("skipped")
        if skipped is None:
            continue
        reasons[_case_id(case)] = (skipped.get("message") or "").strip()[:300]
    return reasons


def counts(outcomes: dict[str, str] | None) -> dict[str, int] | None:
    if outcomes is None:
        return None
    found = {"passed": 0, "failed": 0, "skipped": 0}
    for outcome in outcomes.values():
        found[outcome] += 1
    return {**found, "total": len(outcomes)}


def failed_ids(outcomes: dict[str, str] | None) -> list[str]:
    return sorted(t for t, o in (outcomes or {}).items() if o == "failed")


def read_failure_list(path: Path) -> list[str]:
    """Test ids, one per line; text after ``#`` and blank lines are ignored
    (tools/ci/fresh_resolve_known_failures.txt's format)."""
    ids = set()
    for line in path.read_text().splitlines():
        test_id = line.split("#", 1)[0].strip()
        if test_id:
            ids.add(test_id)
    return sorted(ids)


def compare_with_base(candidate: list[str], base: list[str]) -> dict[str, list[str]]:
    return {
        "new_failures": sorted(set(candidate) - set(base)),
        "fixed_since_base": sorted(set(base) - set(candidate)),
        "shared_failures": sorted(set(candidate) & set(base)),
    }


# ---------------------------------------------------------------------------
# Every failing unit test classified; real bugs only by the lead's waiver
# ---------------------------------------------------------------------------

CLASSES = ("env", "stale test", "real bug")
GITHUB_ORIGIN = re.compile(
    r"(https://github\.com/|git@github\.com:)benchflow-ai/benchflow(-private)?(\.git)?"
)
# The records folder: lab's ops/releases/benchflow, in a checkout of lab.
LAB_ORIGIN = re.compile(
    r"(https://github\.com/|git@github\.com:)benchflow-ai/lab(\.git)?"
)
RECORDS_PATH = "ops/releases/benchflow"
WAIVERS_FILE = "waivers.json"
ISSUE_KEY = re.compile(r"[A-Z][0-9A-Z]*-\d+")
# A development build of a release: 0.10.0rc4.dev0 builds toward 0.10.0rc4.
_DEV_BUILD = re.compile(r"\.dev\d+$")


def waiver_release(version: str) -> str:
    """The release a version is or builds toward, which a waiver names:
    ``0.10.0rc4`` and its development builds (``0.10.0rc4.dev0``) are both
    ``0.10.0rc4``; ``0.10.0rc5`` is the next."""
    return _DEV_BUILD.sub("", version.strip())


class BadRecord(ValueError):
    """A classification or waivers file the gate cannot use."""


def load_classification(path: Path) -> dict[str, Any]:
    """The records folder's ``<version>-failures.json``: why tests failed on
    machines that were not prepared (a record; it decides nothing), each
    test's cause, and each cause's class (env, stale test, real bug) and
    one-line reason.

    {"schema_version": 1, "version": ..., "made_from": {...},
     "causes": {ID: {"class", "cause", "detail", "resolution", "issue"}},
     "tests": {TEST ID: CAUSE ID}}

    A real bug names its issue key.
    """
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise BadRecord(f"{path}: not a schema_version 1 classification")
    causes, tests = data.get("causes"), data.get("tests")
    if not isinstance(causes, dict) or not isinstance(tests, dict):
        raise BadRecord(f"{path}: needs a causes object and a tests object")
    for cid, cause in causes.items():
        if not isinstance(cause, dict) or cause.get("class") not in CLASSES:
            raise BadRecord(f"{path}: cause {cid!r} has no class of {CLASSES}")
        if not str(cause.get("cause") or "").strip():
            raise BadRecord(f"{path}: cause {cid!r} has no one-line cause")
        if cause["class"] == "real bug" and not ISSUE_KEY.fullmatch(
            str(cause.get("issue") or "")
        ):
            raise BadRecord(f"{path}: real bug {cid!r} names no issue key")
    for test, cid in tests.items():
        if cid not in causes:
            raise BadRecord(f"{path}: {test} names an unknown cause {cid!r}")
    return data


def load_waivers(path: Path, version: str, leg: str) -> dict[str, dict[str, Any]]:
    """The lead's waivers for ``version`` on ``leg``, by test id. Only the
    lead edits ``waivers.json``; the gate reads it and never writes it.

    {"schema_version": 1, "waivers": [{"test", "version", "leg"?, "reason",
     "issue", "waived_by", "date"}]}

    A waiver counts for the release it names (:func:`waiver_release`: that
    release and its ``.devN`` builds), on every leg or on the one it names,
    and only with a reason, an issue key and who waived it. It lapses at the
    next release: a waiver for 0.10.0rc4 never applies to 0.10.0rc5 or to
    0.10.0rc3. A leg the gate does not have is refused, so a misspelt leg
    cannot widen or lose a waiver silently.
    """
    return waivers_from(path.read_text(), str(path), version, leg)


def waivers_from(
    text: str, path: str, version: str, leg: str
) -> dict[str, dict[str, Any]]:
    """:func:`load_waivers` of a waivers file's text (``path`` names it)."""
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise BadRecord(f"{path}: not a schema_version 1 waivers file")
    legs = load_legs()
    found: dict[str, dict[str, Any]] = {}
    for n, waiver in enumerate(data.get("waivers") or []):
        if not isinstance(waiver, dict):
            raise BadRecord(f"{path}: waiver {n} is not an object")
        missing = [
            k
            for k in ("test", "version", "reason", "issue", "waived_by")
            if not str(waiver.get(k) or "").strip()
        ]
        if missing:
            raise BadRecord(f"{path}: waiver {n} lacks {', '.join(missing)}")
        if not ISSUE_KEY.fullmatch(str(waiver["issue"])):
            raise BadRecord(f"{path}: waiver {n}'s issue is not an issue key")
        if waiver.get("leg") is not None and waiver["leg"] not in legs:
            raise BadRecord(f"{path}: waiver {n} names no leg of the gate")
        same_release = waiver_release(str(waiver["version"])) == waiver_release(version)
        if same_release and waiver.get("leg") in (None, leg):
            found[waiver["test"]] = waiver
    return found


def must_run_gaps(outcomes: dict[str, str], files: list[str]) -> list[str]:
    """The tests of each must-run file that skipped, and each such file with
    no test in the report at all."""
    gaps = []
    for file in files:
        ran = {t: o for t, o in outcomes.items() if t.split("::")[0] == file}
        if not ran:
            gaps.append(f"{file} (not in the report)")
        gaps += sorted(t for t, o in ran.items() if o == "skipped")
    return gaps


def judge_failures(
    failures: list[str], waivers: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    """Every failure fails the gate unless the lead waived it by name."""
    return {
        "waived": [t for t in failures if t in waivers],
        "not_waived": [t for t in failures if t not in waivers],
    }


def render_failures(data: dict[str, Any]) -> str:
    """The classification as the Markdown table committed beside it."""
    causes, tests = data["causes"], data["tests"]
    members: dict[str, list[str]] = {cid: [] for cid in causes}
    for test, cid in sorted(tests.items()):
        members[cid].append(test)
    made = data.get("made_from") or {}
    lines = [
        f"# Failing unit tests on BenchFlow {data.get('version')}, classified",
        "",
        "Generated from the JSON twin beside this file "
        "(`python3 tools/release_gate/run_gate.py --render-failures FILE`).",
        "",
    ]
    for key, value in made.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "| class | causes | tests |", "|---|---|---|"]
    for cls in CLASSES:
        ids = [cid for cid, c in causes.items() if c["class"] == cls]
        lines.append(f"| {cls} | {len(ids)} | {sum(len(members[c]) for c in ids)} |")
    lines.append(f"| all | {len(causes)} | {len(tests)} |")
    for cls in CLASSES:
        for cid, cause in causes.items():
            if cause["class"] != cls:
                continue
            lines += ["", f"## {cid}: {cls}, {len(members[cid])} tests", ""]
            lines.append(f"Cause: {cause['cause']}")
            if cause.get("detail"):
                lines += ["", cause["detail"]]
            if cause.get("resolution"):
                lines += ["", f"Resolution: {cause['resolution']}"]
            if cause.get("issue"):
                lines += ["", f"Issue: {cause['issue']}"]
            lines += ["", "| test id | class | cause |", "|---|---|---|"]
            short = cause["cause"].replace("|", "\\|")
            for test in members[cid]:
                safe = test.replace("|", "\\|")
                lines.append(f"| `{safe}` | {cls} | {short} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The cost matrix's own records
# ---------------------------------------------------------------------------


def blocked_cells(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """Every cell run_matrix.py would pick and does not run: matrix.json's
    ``blocked`` on a cell, an effort cell or a refusal cell."""
    found = []
    for group in ("cells", "effort_cells", "refusal_cells"):
        for cell in matrix.get(group) or []:
            if cell.get("blocked"):
                found.append(
                    {
                        "id": cell.get("id"),
                        "group": group,
                        "agent": cell.get("agent"),
                        "harness": cell.get("harness", "acp"),
                        "auth": cell.get("auth"),
                        "reason": cell["blocked"],
                    }
                )
    return found


def auth_name(auth: str | None) -> str:
    stem = re.sub(r"_(key|sub|login)$", "", auth or "unknown")
    return AUTH_NAMES.get(stem, stem)


def release_note(
    version: str, commit: str, matrix: dict[str, Any] | None
) -> str | None:
    """The cost lines the release notes carry: what no live cell verified."""
    if matrix is None:
        return None
    blocked = matrix["blocked"]
    names: list[str] = []
    for cell in blocked:
        name = auth_name(cell["auth"])
        if name not in names:
            names.append(name)
    lines = [f"Cost and token records in BenchFlow {version} ({commit[:12]}):"]
    if matrix.get("ran"):
        lines.append(
            f"- The cost matrix ran {matrix['runs']} live runs; "
            f"{matrix['passed']} passed."
        )
    else:
        lines.append("- The cost matrix did not run in this gate.")
    if names:
        joined = (
            names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        )
        verb, whose = ("is", "its") if len(names) == 1 else ("are", "their")
        cells = ", ".join(c["id"] for c in blocked)
        lines.append(
            f"- {joined} cost {verb} unverified live on this rc: "
            f"{whose} matrix cells are blocked ({cells})."
        )
    for item in matrix["offline_evidence"]:
        lines.append(
            f"- Checked offline only: {item.get('check')} on {item.get('data')}: "
            f"{item.get('result')}."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def secret_values(secrets: Path | None) -> list[str]:
    """Every value in the secrets folder's KEY=VALUE files, as run_matrix.py
    reads them; only values of 8 characters or more are scrubbed."""
    if secrets is None or not secrets.is_dir():
        return []
    values: set[str] = set()
    for path in sorted(secrets.iterdir()):
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            key, eq, value = line.strip().removeprefix("export ").partition("=")
            value = value.strip().strip('"').strip("'")
            if eq and key.strip() and not key.startswith("#") and len(value) >= 8:
                values.add(value)
    # The longest first, so a value inside another is not left half-replaced.
    return sorted(values, key=lambda value: len(value), reverse=True)


def scrub_text(text: str, values: list[str]) -> str:
    for value in values:
        text = text.replace(value, "[secret]")
    return text


def scrub(logs: Path, values: list[str]) -> list[str]:
    """Replace each secret value in every file under ``logs``, text or not
    (bytes, as UTF-8); the files changed."""
    changed = []
    if not values:
        return changed
    needles = [v.encode() for v in values]
    for path in sorted(p for p in logs.rglob("*") if p.is_file()):
        data = path.read_bytes()
        clean = data
        for needle in needles:
            clean = clean.replace(needle, b"[secret]")
        if clean != data:
            path.write_bytes(clean)
            changed.append(path.relative_to(logs).as_posix())
    return changed


def log_files(logs: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": p.relative_to(logs).as_posix(),
            "bytes": p.stat().st_size,
            "md5": md5_file(p),
        }
        for p in sorted(logs.rglob("*"))
        if p.is_file()
    ]


def gcs_md5s(listing: list[dict[str, Any]], dest: str) -> dict[str, str]:
    """``gcloud storage objects list --format=json`` rows under ``dest``, as
    relative path -> hex MD5 (GCS reports it base64-encoded)."""
    bucket, _, prefix = dest.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    found = {}
    for row in listing:
        name = row.get("name", "")
        if row.get("bucket") != bucket or not name.startswith(prefix):
            continue
        md5 = row.get("md5_hash")
        if md5:
            found[name[len(prefix) :]] = base64.b64decode(md5).hex()
    return found


def verify(
    files: list[dict[str, Any]], remote: dict[str, str], dest: str, folder: str
) -> bool:
    """Mark each file verified when its copy's MD5 matches, and record where
    the copy is (under ``folder``/, as a recursive copy of the folder puts
    it, or at the top); True when every file is verified."""
    for f in files:
        for rel in (f"{folder}/{f['path']}", f["path"]):
            if rel in remote:
                f["uri"] = f"{dest}/{rel}"
                f["verified"] = remote[rel] == f["md5"]
                break
        else:
            f["uri"], f["verified"] = None, False
    return bool(files) and all(f["verified"] for f in files)


def copy_records(src: Path, dest: Path, *, logs_folder: bool = False) -> None:
    """The small text records of a job folder tree (and its ``logs/`` when
    asked); trajectories and other bulk stay where they are."""
    names = ("report.md", "results.jsonl", "result.json", "summary.json")
    names += ("run-card.json", "config.json")
    for path in sorted(src.rglob("*")) if src.is_dir() else []:
        rel = path.relative_to(src)
        if path.is_file() and (
            rel.name in names or (logs_folder and rel.parts[0] == "logs")
        ):
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / rel)


def drop_tree(path: Path) -> None:
    """Remove a pytest basetemp. Some tests nest paths past PATH_MAX, which
    Python's path calls refuse (ENAMETOOLONG); rm -rf walks by file
    descriptor. Some leave folders read-only: what rm leaves is made writable
    and removed again (chmod -R over a whole suite's folders first took
    minutes)."""
    if not path.exists():
        return
    subprocess.run(["rm", "-rf", "--", str(path)], capture_output=True)
    if path.exists():
        subprocess.run(["chmod", "-R", "u+rwX", "--", str(path)], capture_output=True)
        subprocess.run(["rm", "-rf", "--", str(path)], capture_output=True)


def restore_stop_signals() -> list[str]:
    """Give the tests Ctrl-C and SIGQUIT back when the gate was started in
    the background.

    A shell starting a command with ``&`` and no job control (``nohup ...
    &`` over ssh) sets SIGINT and SIGQUIT to ignored, and every process the
    gate starts inherits that: a Ctrl-C test then finds its run deaf to
    Ctrl-C (tests/test_stop_summary_docker.py failed so). Python's own
    handler for SIGINT makes a child start with the default action again.
    Returns the names of the signals that were ignored.
    """
    restored = []
    for sig, handler in (
        (signal.SIGINT, signal.default_int_handler),
        (signal.SIGQUIT, signal.SIG_DFL),
    ):
        if signal.getsignal(sig) == signal.SIG_IGN:
            signal.signal(sig, handler)
            restored.append(sig.name)
    return restored


def group_problem(gid: int | None = None, account_gid: int | None = None) -> str | None:
    """Why this process's group is not its account's own, or None.

    ``sg docker -c ...`` (or ``newgrp``) runs a command with the docker
    group as its real group. The agent entry then cannot switch to the
    account's own group (``setpriv: setresgid failed: Operation not
    permitted``) and reports the wrong one: 24 tests failed so. A login
    started after ``usermod -aG docker`` has docker as an extra group.
    """
    gid = os.getgid() if gid is None else gid
    if account_gid is None:
        import pwd

        account_gid = pwd.getpwuid(os.getuid()).pw_gid
    if gid == account_gid and os.getegid() == gid:
        return None
    return (
        f"this process runs with group {gid}, not its account's group "
        f"{account_gid} (started through sg or newgrp?): start the gate "
        "from a new login, where the docker group is an extra group"
    )


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


_RELEASE = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?")
_PRE_RANK = {"a": 0, "b": 1, "rc": 2, None: 3}


def release_key(version: str) -> tuple[int, ...] | None:
    """A released version's order (0.9.0 < 0.10.0rc2 < 0.10.0rc3 < 0.10.0);
    None for anything else (a .dev build, a local version)."""
    match = _RELEASE.fullmatch(version or "")
    if match is None:
        return None
    major, minor, patch, pre, number = match.groups()
    return (int(major), int(minor), int(patch), _PRE_RANK[pre], int(number or 0))


def last_rc(checkout: Path, commit: str) -> dict[str, Any] | None:
    """The last release before ``commit``: of the accepted records in
    .github/release-acceptance/ whose source commit is an ancestor of
    ``commit``, the highest version. (Not the nearest by commits: a line
    that merged an older release's branch, as the 0.10 line merged public
    main, has that release's source closer.) None when there is none."""
    best: tuple[tuple[int, ...], dict[str, Any]] | None = None
    for path in sorted((checkout / ACCEPTANCE_RECORDS).glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        source = record.get("source_commit") if isinstance(record, dict) else None
        if record.get("status") != "accepted" or not isinstance(source, str):
            continue
        if subprocess.run(
            ["git", "-C", str(checkout), "merge-base", "--is-ancestor", source, commit],
            capture_output=True,
        ).returncode:
            continue
        key = release_key(str(record.get("version") or ""))
        if key is None:
            continue
        if best is None or key > best[0]:
            best = (
                key,
                {
                    "version": record.get("version"),
                    "source_commit": source,
                    "record": f"{ACCEPTANCE_RECORDS}/{path.name}",
                },
            )
    return None if best is None else best[1]


def merged_prs(checkout: Path, since: str, commit: str) -> list[dict[str, Any]]:
    """Every pull request merged between ``since`` and ``commit``, oldest
    first, from the merge commits GitHub writes (PRs merged into a branch that
    was then merged count too)."""
    out = git(checkout, "log", "--reverse", "--format=%H%x1f%s", f"{since}..{commit}")
    prs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in out.splitlines():
        sha, _, subject = line.partition("\x1f")
        match = _PR_SUBJECT.search(subject)
        if match is None:
            continue
        number = int(match.group(1) or match.group(2))
        if number not in seen:
            seen.add(number)
            prs.append({"pr": number, "commit": sha, "subject": subject[:200]})
    return prs


def rc_identity(checkout: Path, branch: str | None, commit: str) -> dict[str, Any]:
    """Whether ``commit`` is the rc branch's head, and what that head holds.

    A release gate tests the rc branch's head after every merge: the head as
    fetched into this checkout (``origin/BRANCH``) and, when GitHub answers
    (a VM has no credentials for a private repository), its head there.
    Each reason the run is not a release gate is in ``problems``."""
    info: dict[str, Any] = {
        "branch": branch,
        "head": None,
        "remote_head": None,
        "remote_checked": False,
        "last_rc": None,
        "merged_prs": [],
        "problems": [],
    }
    problems: list[str] = info["problems"]
    if not branch:
        problems.append("no --rc-branch: a run on a PR branch or a scratch merge")
    else:
        try:
            info["head"] = git(
                checkout,
                "rev-parse",
                "--verify",
                f"refs/remotes/origin/{branch}^{{commit}}",
            )
        except subprocess.CalledProcessError:
            problems.append(f"origin/{branch} is not in this checkout: fetch it")
        if info["head"] and info["head"] != commit:
            problems.append(
                f"the tested commit is not origin/{branch}'s head ({info['head'][:12]})"
            )
        try:
            remote = subprocess.run(
                ["git", "-C", str(checkout), "ls-remote", "--heads", "origin", branch],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            found = remote.stdout.split() if remote.returncode == 0 else []
        except (OSError, subprocess.SubprocessError):
            found = []
        if found:
            info["remote_checked"] = True
            info["remote_head"] = found[0]
            if found[0] != commit:
                problems.append(
                    f"{branch} on GitHub is at {found[0][:12]}, not the tested commit"
                )
    info["last_rc"] = last_rc(checkout, commit)
    if info["last_rc"] is None:
        problems.append(
            f"no accepted record in {ACCEPTANCE_RECORDS} is an ancestor: "
            "the PRs since the last rc cannot be listed"
        )
    else:
        info["merged_prs"] = merged_prs(
            checkout, info["last_rc"]["source_commit"], commit
        )
    return info


def verdict(receipt: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """PASS, FAIL or INCOMPLETE, with what failed and what was left out.

    A receipt that is not a release gate (``rc``: not the rc branch's head)
    is never a PASS."""
    fail: list[str] = []
    missing: list[str] = []
    rc = receipt.get("rc")
    if not isinstance(rc, dict):
        missing.append("not a release gate: the receipt names no rc branch head")
    elif rc.get("problems"):
        missing.append("not a release gate: " + "; ".join(rc["problems"]))
    stages = receipt["stages"]
    if receipt.get("gate_error"):
        fail.append(f"the gate broke: {receipt['gate_error']}")
    if not stages.get("setup", {}).get("ok"):
        fail.append("setup failed")
    for name in receipt.get("leg_stages") or STAGES:
        stage = stages.get(name)
        if stage is None or stage.get("skipped"):
            missing.append(
                f"{name}: not run ({(stage or {}).get('skipped', 'skipped')})"
            )
        elif not stage["ok"]:
            fail.extend(f"{name}: {why}" for why in stage["problems"])
        if stage and stage.get("subset"):
            missing.append(f"{name}: a subset, not the whole suite")
    logs: dict[str, Any] = receipt.get("logs") or {
        "destination": None,
        "verified": False,
    }
    destination = logs["destination"]
    if destination is None:
        missing.append("logs: not uploaded")
    elif not logs["verified"]:
        fail.append("logs: an uploaded copy does not match its MD5")
    elif str(destination).startswith("file://"):
        missing.append("logs: copied to a local folder, not uploaded")
    if receipt["dry_run"]:
        missing.append("a dry run")
    if fail:
        return "FAIL", fail, missing
    if missing:
        return "INCOMPLETE", fail, missing
    return "PASS", fail, missing


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------


def _rc_text(rc: dict[str, Any]) -> str:
    if not rc or not rc.get("branch"):
        return "none (not a release gate)"
    head = f"`{rc['head']}`" if rc.get("head") else "not fetched"
    remote = (
        f"GitHub `{rc['remote_head']}`"
        if rc.get("remote_checked")
        else "GitHub not checked (no access)"
    )
    last = rc.get("last_rc") or {}
    since = (
        f"; last rc {last.get('version')} `{str(last.get('source_commit'))[:12]}`"
        if last
        else ""
    )
    return f"`{rc['branch']}` head {head}, {remote}{since}"


def _advisories_text(stage: dict[str, Any] | None) -> str:
    adv = (stage or {}).get("advisories") or {}
    if not adv.get("queried_at"):
        return "not run"
    by_file: dict[str, list[str]] = {}
    for a in adv.get("audited") or []:
        by_file.setdefault(a["file"], []).append(
            (f"{a['service']} " if a.get("service") else "")
            + f"{'clean' if a.get('rc') == 0 else 'FAILED'} at {a['queried_at']}"
        )
    files = "; ".join(f"{name}: {', '.join(runs)}" for name, runs in by_file.items())
    return (
        f"queried {adv['queried_at']} (pip-audit {adv.get('pip_audit') or '?'}, "
        f"{adv.get('service')})" + (f"; audited: {files}" if files else "")
    )


def _prs_text(rc: dict[str, Any]) -> str:
    prs = (rc or {}).get("merged_prs") or []
    return ", ".join(f"#{p['pr']}" for p in prs) or "none listed"


def _record_text(record: dict[str, Any] | None) -> str:
    if not record:
        return "none"
    md5 = f" (md5 `{record['md5']}`)" if record.get("md5") else ""
    return (
        f"`{record['path']}` at {record.get('repo') or '?'} commit "
        f"`{record.get('commit') or '?'}`{md5}"
    )


def _ids(lines: list[str], ids: list[str]) -> None:
    lines.extend(f"- `{i}`" for i in ids) if ids else lines.append("- none")


def render_markdown(r: dict[str, Any]) -> str:
    src, st = r["source"], r["stages"]
    lines = [
        f"# Release gate receipt: BenchFlow {r['version']}, leg {r.get('leg', '-')}",
        "",
        f"**{r['verdict']}**" + (" (dry run)" if r["dry_run"] else ""),
        "",
    ]
    rc = r.get("rc") or {}
    if rc.get("problems") or not rc:
        lines += [
            "**NOT A RELEASE GATE**: this run did not test the rc branch's head.",
            "",
        ]
    for why in r["failures"]:
        lines.append(f"- FAIL: {why}")
    for why in r["incomplete"]:
        lines.append(f"- not covered: {why}")
    base = r["base"]
    if base["commit"]:
        base_text = f"`{base['commit']}` (run here)"
    elif base["failures_file"]:
        ff = base["failures_file"]
        base_text = f"recorded list `{ff['path']}` (md5 `{ff['md5']}`, {ff['ids']} ids)"
    else:
        base_text = "none: every failure counts as new"
    lines += [
        "",
        "| | |",
        "|---|---|",
        f"| source commit | `{src['commit']}` |",
        f"| rc branch | {_rc_text(rc)} |",
        f"| PRs merged since the last rc | {_prs_text(rc)} |",
        f"| pip-audit advisories | {_advisories_text(st.get('pip-audit'))} |",
        f"| source dirty | {str(src['dirty']).lower()} |",
        f"| base | {base_text} |",
        f"| leg | {r.get('leg', '-')}: stages {', '.join(r.get('leg_stages') or STAGES)} |",
        f"| records | {_record_text(r.get('records'))} |",
        f"| waivers | {_record_text(r.get('waivers'))} |",
        f"| failure table (decides nothing) | {_record_text(r.get('classification'))} |",
        f"| gate script | `{r['gate_script']['path']}` sha256 `{r['gate_script']['sha256']}` |",
        f"| started / finished (UTC) | {r['started_at']} / {r['finished_at']} |",
        f"| umask | {r.get('umask', '-')} |",
        f"| host | {r['host']['hostname']}, {r['host']['os']}, {r['host']['cpus']} CPUs, "
        f"Docker {r['host']['docker']}, Compose {r['host'].get('compose')}, uv {r['host']['uv']}, "
        f"Python {r.get('python_version') or r['python']} |",
        f"| logs | {r['logs']['destination'] or 'not uploaded'} |",
        "",
        "## Stages",
        "",
        "| stage | result | passed | failed | skipped | seconds |",
        "|---|---|---|---|---|---|",
    ]
    for name in ORDER:
        stage = st.get(name)
        if stage is None:
            continue
        if stage.get("skipped"):
            result = f"not run: {stage['skipped']}"
        else:
            result = "pass" if stage["ok"] else "FAIL"
        c = stage.get("counts") or {}
        lines.append(
            f"| {name} | {result} | {c.get('passed', '-')} | {c.get('failed', '-')} |"
            f" {c.get('skipped', '-')} | {stage.get('seconds', '-')} |"
        )
    lines += ["", "## Commands", ""]
    for name in ORDER:
        for cmd in (st.get(name) or {}).get("commands", []):
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in cmd["env"].items())
            lines.append(
                f"- {name}: exit {cmd['rc']}, {cmd['seconds']} s, in `{cmd['cwd']}`, log `{cmd['log']}`  "
            )
            lines.append(f"  `{env} {shlex.join(cmd['argv'])}`")
    unit = st.get("unit") or {}
    if not unit.get("skipped"):
        cmp = unit.get("comparison") or {}
        judged = unit.get("judged") or {}
        lines += ["", "## Unit failures", ""]
        lines += [
            "Every failure fails the gate unless the lead waived it by name.",
            "",
            f"Not waived ({len(judged.get('not_waived', []))}):",
            "",
        ]
        _ids(lines, judged.get("not_waived", []))
        lines += ["", f"Waived by the lead ({len(judged.get('waived', []))}):", ""]
        _ids(lines, judged.get("waived", []))
        lines += ["", "## Unit failures against the base", ""]
        lines.append(f"New failures ({len(cmp.get('new_failures', []))}):")
        lines.append("")
        _ids(lines, cmp.get("new_failures", []))
        lines += ["", f"Candidate failures ({len(unit.get('failures', []))}):", ""]
        _ids(lines, unit.get("failures", []))
        base_run = st.get("unit-base")
        if base_run is not None and not base_run["ok"]:
            lines += ["", "Base failures: unknown, the base suite did not finish."]
        else:
            lines += ["", f"Base failures ({len(r['base']['failures'])}):", ""]
            _ids(lines, r["base"]["failures"])
        lines += [
            "",
            f"Failing on the base, passing or absent here ({len(cmp.get('fixed_since_base', []))}):",
            "",
        ]
        _ids(lines, cmp.get("fixed_since_base", []))
    for name in ("docker", "hillclimb"):
        stage = st.get(name) or {}
        if stage.get("skipped"):
            continue
        lines += ["", f"## {name}: failed and skipped tests", "", "Failed:", ""]
        _ids(lines, stage.get("failures", []))
        lines += ["", "Skipped, with the reason the test gives:", ""]
        reasons = stage.get("skip_reasons") or {}
        skipped = stage.get("skipped_tests", [])
        lines.extend(f"- `{t}`: {reasons.get(t) or 'no reason'}" for t in skipped)
        if not skipped:
            lines.append("- none")
    m = r["matrix"]
    lines += ["", "## Cost matrix", ""]
    if m is None:
        lines.append("No matrix.json: the cost matrix is not in this checkout.")
    else:
        lines.append(
            f"matrix.json `{m['matrix_json']}` (md5 `{m['matrix_json_md5']}`"
            + (
                f", from commit `{m['matrix_commit']}`"
                if m.get("matrix_commit")
                else ""
            )
            + ")."
        )
        lines.append("")
        if m["ran"]:
            lines.append(f"{m['passed']} of {m['runs']} runs passed; exit {m['rc']}.")
            for row in m["failed_runs"]:
                lines.append(
                    f"- FAIL {row['cell']} {row['sandbox']} {row['task']}: "
                    + "; ".join(row.get("problems") or [])
                )
        else:
            lines.append("Not run in this gate.")
        lines += [
            "",
            f"BLOCKED cells ({len(m['blocked'])}): not run, neither passes nor failures.",
            "",
            "| cell | agent | harness | auth | reason |",
            "|---|---|---|---|---|",
        ]
        for c in m["blocked"]:
            lines.append(
                f"| {c['id']} | {c['agent']} | {c['harness']} | {c['auth']} | {c['reason']} |"
            )
        lines += ["", "Offline evidence (matrix.json, verbatim):", ""]
        if not m["offline_evidence"]:
            lines.append("- none")
        for item in m["offline_evidence"]:
            lines.append("- " + "; ".join(f"{k}: {v}" for k, v in item.items()))
    if r["release_note_draft"]:
        lines += ["", "## Release note draft", ""]
        lines += [
            f"> {line}" if line else ">"
            for line in r["release_note_draft"].splitlines()
        ]
    lines += ["", "## Logs", ""]
    logs = r["logs"]
    lines.append(
        f"{len(logs['files'])} files, uploaded to {logs['destination'] or 'nowhere'}"
        + (", every MD5 verified" if logs["verified"] else "")
        + f"; secret values scrubbed from {len(logs['scrubbed'])} files before hashing."
    )
    lines += ["", "| file | bytes | md5 | verified |", "|---|---|---|---|"]
    for f in logs["files"]:
        lines.append(
            f"| {f['path']} | {f['bytes']} | `{f['md5']}` | {str(f.get('verified', False)).lower()} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _read_log(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def load_legs() -> dict[str, dict[str, Any]]:
    """tools/release_gate/legs.json: each leg's Python, Docker and stages."""
    legs = json.loads(LEGS_FILE.read_text())["legs"]
    for name, leg in legs.items():
        if leg.get("docker") not in ("ubuntu", "ce") or not set(leg["stages"]) <= set(
            STAGES
        ):
            raise BadRecord(f"{LEGS_FILE}: leg {name} is not valid")
    return legs


def proven_compose_versions() -> frozenset[str]:
    """The Compose builds the direct Docker path is proven on."""
    source = (ROOT / "src" / "benchflow" / "sandbox" / "_direct_docker.py").read_text()
    found = re.search(r"PROVEN_COMPOSE_VERSIONS = frozenset\(\{([^}]*)\}\)", source)
    return frozenset(re.findall(r'"([^"]+)"', found.group(1))) if found else frozenset()


def check_records(records: Path, version: str) -> dict[str, Any]:
    """The records folder: lab's ops/releases/benchflow, in a checkout whose
    origin is github.com/benchflow-ai/lab, and its waivers and failure
    table, each only as committed there. Any other folder or repository is
    refused: a waiver counts only as the lead committed it in lab, and a
    ``git init`` anywhere else could commit one the lead never wrote. The
    receipt names lab's origin and the commit read, which --combine checks
    against lab (:func:`lab_problems`)."""
    try:
        top = Path(git(records, "rev-parse", "--show-toplevel")).resolve()
        commit = git(records, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, OSError):
        raise Refused(f"--records {records} is not in a git checkout") from None
    try:
        repo = git(records, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        repo = ""
    if not LAB_ORIGIN.fullmatch(repo):
        # Never the URL itself: one with a token in it stays out of the logs.
        raise Refused(
            f"--records {records} is not in a checkout of lab (its origin is "
            f"{'unset' if not repo else 'another repository'}): clone lab and "
            "git remote set-url origin https://github.com/benchflow-ai/lab.git"
        )
    try:
        where = records.resolve().relative_to(top).as_posix()
    except ValueError:
        where = None
    if where != RECORDS_PATH:
        raise Refused(f"--records {records} is not lab's {RECORDS_PATH}")
    found: dict[str, Any] = {
        "records": {"path": str(records), "repo": repo, "commit": commit},
        "waivers": None,
        "classification": None,
    }
    for key, name in (
        ("waivers", "waivers.json"),
        ("classification", f"{version}-failures.json"),
    ):
        path = records / name
        if not path.exists():
            continue
        rel = path.resolve().relative_to(top).as_posix()
        tracked = (
            subprocess.run(
                ["git", "-C", str(top), "ls-files", "--error-unmatch", "--", rel],
                capture_output=True,
            ).returncode
            == 0
        )
        changed = (
            subprocess.run(
                ["git", "-C", str(top), "diff", "--quiet", "HEAD", "--", rel],
                capture_output=True,
            ).returncode
            != 0
        )
        if not tracked or changed:
            raise Refused(f"{path} is not committed as it is: commit it first")
        found[key] = {
            "path": str(path),
            "repo": repo,
            "commit": commit,
            "md5": md5_file(path),
        }
    return found


class Gate:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.checkout = ROOT
        self.work: Path = args.work
        self.logs = self.work / "logs"
        self.tmp = self.work / "tmp"
        self.waived: dict[str, dict[str, Any]] = {}
        # Signals main() found ignored and gave back (restore_stop_signals).
        self.restored_signals: list[str] = []
        self.leg_name: str = args.leg
        self.leg: dict[str, Any] = load_legs()[args.leg]
        self.python: str = self.leg["python"]
        # bootstrap_vm.sh installs the pinned uv here; preflight checks it.
        local = Path.home() / ".local" / "bin" / "uv"
        self.uv = str(local) if local.is_file() else shutil.which("uv") or "uv"

    # -- commands ----------------------------------------------------------

    def env(self, extra: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        """The command's whole environment, and what the gate set itself."""
        set_here = {
            "HARBOR_TELEMETRY": "off",
            "UV_PYTHON": self.python,
            "LANG": "C.UTF-8",
            **extra,
        }
        path = os.pathsep.join(
            [str(Path(self.uv).parent), os.environ.get("PATH", "/usr/bin:/bin")]
        )
        env = {k: os.environ[k] for k in PASSTHROUGH if k in os.environ}
        env.update(set_here, PATH=path)
        return env, set_here

    def run(
        self,
        stage: dict[str, Any],
        log_name: str,
        argv: list[str],
        cwd: Path,
        timeout: int,
        extra_env: dict[str, str] | None = None,
    ) -> int | str:
        env, set_here = self.env(extra_env or {})
        log = self.logs / log_name
        log.parent.mkdir(parents=True, exist_ok=True)
        self.say(f"$ {shlex.join(argv)}  (in {cwd}, log {log_name})")
        start = time.monotonic()
        with log.open("ab") as fh:
            fh.write(f"# {utc_now()} $ {shlex.join(argv)}\n".encode())
            fh.flush()
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                rc: int | str = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                        proc.wait(timeout=60)
                        break
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        continue
                rc = "timeout"
            fh.write(f"# {utc_now()} exit {rc}\n".encode())
        seconds = round(time.monotonic() - start, 1)
        stage.setdefault("commands", []).append(
            {
                "argv": argv,
                "cwd": str(cwd),
                "env": set_here,
                "env_from_host": sorted(
                    ["PATH", *(k for k in PASSTHROUGH if k in os.environ)]
                ),
                "rc": rc,
                "seconds": seconds,
                "log": log_name,
            }
        )
        stage["seconds"] = round(stage.get("seconds", 0) + seconds, 1)
        self.say(f"  exit {rc} after {seconds} s")
        return rc

    def say(self, text: str) -> None:
        line = f"[{utc_now()}] {text}"
        print(line, flush=True)
        with (self.logs / "gate.log").open("a") as fh:
            fh.write(line + "\n")

    def uv_cmd(self, *args: str) -> list[str]:
        return [self.uv, *args]

    def pytest(self, *args: str, junit: str, basetemp: str) -> list[str]:
        return self.uv_cmd(
            "run",
            "--no-sync",
            "python",
            "-m",
            "pytest",
            *args,
            "-p",
            "no:cacheprovider",
            "-o",
            "junit_family=xunit1",
            f"--junitxml={self.logs / 'junit' / junit}",
            f"--basetemp={self.tmp / basetemp}",
            "-rfEs",
        )

    # -- stages ------------------------------------------------------------

    def setup(self, tree: Path, stage: dict[str, Any], prefix: str) -> bool:
        t = STAGE_TIMEOUT_SEC["setup"]
        steps = [
            self.uv_cmd("python", "install", self.python),
            self.uv_cmd("sync", *(f"--extra={e}" for e in EXTRAS), "--locked"),
            # test.yml installs the browser so the viewer's browser tests run.
            self.uv_cmd(
                "run", "--no-sync", "playwright", "install", "--with-deps", "chromium"
            ),
            # Images tests need present before they are collected: without
            # ubuntu:24.04, tests/test_direct_docker_container.py skips whole.
            *(["docker", "pull", "-q", image] for image in self.leg.get("images", [])),
        ]
        for i, argv in enumerate(steps):
            if self.run(stage, f"{prefix}setup-{i + 1}.log", argv, tree, t) != 0:
                stage["problems"].append(f"{shlex.join(argv[1:4])} failed")
                return False
        return True

    def static(self, stage: dict[str, Any]) -> None:
        for name, args in STATIC_CHECKS:
            rc = self.run(
                stage,
                f"static-{name}.log",
                self.uv_cmd("run", "--no-sync", *args),
                self.checkout,
                STAGE_TIMEOUT_SEC["static"],
            )
            if rc != 0:
                stage["problems"].append(f"{' '.join(args)} exited {rc}")
        stage["generated"] = []
        for n, (what, args, regenerate) in enumerate(GENERATED_CHECKS, 1):
            rc = self.run(
                stage,
                f"static-generated-{n}.log",
                self.uv_cmd("run", "--no-sync", *args),
                self.checkout,
                STAGE_TIMEOUT_SEC["static"],
            )
            stage["generated"].append(
                {"what": what, "check": " ".join(args), "exit": rc}
            )
            if rc != 0:
                stage["problems"].append(
                    f"{what} is stale on this head: regenerate with `{regenerate}`"
                    " (on a build VM, in the test environment) and commit"
                )
        stage["ok"] = not stage["problems"]

    def unit_run(
        self, tree: Path, stage: dict[str, Any], label: str, select: list[str]
    ) -> dict[str, str] | None:
        rc = self.run(
            stage,
            f"unit-{label}.log",
            self.pytest(
                *select,
                "-n",
                "auto",
                # A hung test fails as a timeout instead of holding the stage.
                # The thread method: the signal one arms SIGALRM in every
                # test, and a test that checks a forked child's signals
                # (test_local_daemon) then fails.
                "--timeout",
                str(self.args.unit_timeout),
                "--timeout-method=thread",
                junit=f"unit-{label}.xml",
                basetemp=f"unit-{label}",
            ),
            tree,
            STAGE_TIMEOUT_SEC["unit"],
            {"BENCHFLOW_DETERMINISTIC_SANDBOX": "off"},
        )
        drop_tree(self.tmp / f"unit-{label}")
        outcomes = junit_outcomes(self.logs / "junit" / f"unit-{label}.xml")
        stage["rc"] = rc
        stage["counts"] = counts(outcomes)
        stage["failures"] = failed_ids(outcomes)
        # pytest: 0 all passed, 1 some failed; anything else did not finish.
        if rc not in (0, 1) or outcomes is None:
            stage["problems"].append(f"the {label} suite did not finish (exit {rc})")
            return None
        if rc == 1 and not stage["failures"]:
            stage["problems"].append(
                f"pytest exited 1 on the {label} suite, but its report has no failure"
            )
            return None
        return outcomes

    def unit(self, stage: dict[str, Any], receipt: dict[str, Any]) -> None:
        select = self.args.unit_select
        stage["subset"] = select != ["tests/"]
        outcomes = self.unit_run(self.checkout, stage, "candidate", select)
        base = receipt["base"]
        base_failures: list[str] | None = None
        if self.args.base_failures is not None:
            base_failures = base["failures"]
        elif self.args.base_commit is not None:
            base_stage: dict[str, Any] = {
                "name": "unit-base",
                "problems": [],
                "ok": False,
            }
            receipt["stages"]["unit-base"] = base_stage
            tree = self.work / "base"
            git(self.checkout, "worktree", "add", "--detach", str(tree), base["commit"])
            try:
                if self.setup(tree, base_stage, "base-"):
                    # Paths the base does not have yet are new tests: left out.
                    kept = [s for s in select if (tree / s.split("::")[0]).exists()]
                    base_stage["select"] = kept
                    if (
                        kept
                        and self.unit_run(tree, base_stage, "base", kept) is not None
                    ):
                        base_failures = base_stage["failures"]
                        base_stage["ok"] = True
                    elif not kept:
                        base_failures = []
                        base_stage["ok"] = True
            finally:
                git(self.checkout, "worktree", "remove", "--force", str(tree))
            base["failures"] = base_failures or []
            if base_failures is None:
                stage["problems"].append("the base suite did not finish")
        else:
            base_failures = []
        if outcomes is not None and base_failures is not None:
            # Beside the base, for the reader: which failures are new.
            stage["comparison"] = compare_with_base(stage["failures"], base_failures)
        if outcomes is not None:
            judged = judge_failures(stage["failures"], self.waived)
            stage["judged"] = judged
            if judged["not_waived"]:
                stage["problems"].append(
                    f"{len(judged['not_waived'])} tests failed without a waiver"
                )
            if not stage.get("subset"):
                missing = must_run_gaps(outcomes, self.leg.get("must_run", []))
                stage["must_run_gaps"] = missing
                if missing:
                    stage["problems"].append(
                        f"{len(missing)} tests that must run here did not: "
                        + ", ".join(missing[:5])
                    )
        stage["ok"] = not stage["problems"]

    def pip_audit(self, stage: dict[str, Any]) -> None:
        """pip-audit at the rc cut: the lock with every extra (test.yml's job
        audits three of them) and every agent pin file in
        src/benchflow/agents/requirements/, which the lock does not hold, each
        once per advisory service (PIP_AUDIT_SERVICES). Every file and service
        is audited, also after one fails, and the receipt lists each with when
        its advisories were read."""
        requirements = self.work / "requirements-audit.txt"
        export = self.uv_cmd(
            "export",
            "--locked",
            "--all-extras",
            "--format",
            "requirements-txt",
            "--no-hashes",
            "--output-file",
            str(requirements),
        )
        audits = [("uv.lock (all extras)", requirements)] + [
            (path.relative_to(self.checkout).as_posix(), path)
            for path in sorted((self.checkout / AGENT_REQUIREMENTS).glob("*.txt"))
        ]
        # The advisories are read live from each service: what a pass proves
        # is dated by when this run asked, on the head it tested, not by when
        # the receipt is read.
        version_log = "pip-audit-version.log"
        self.run(
            stage, version_log,
            self.uv_cmd("run", "--with", "pip-audit", "pip-audit", "--version"),
            self.checkout, STAGE_TIMEOUT_SEC["pip-audit"],
        )  # fmt: skip
        found = re.search(
            r"pip-audit (\d+(?:\.\d+)+)", _read_log(self.logs / version_log)
        )
        stage["advisories"] = {
            "service": "PyPI's and OSV's vulnerability data, each read live",
            "services": list(PIP_AUDIT_SERVICES),
            "pip_audit": found.group(1) if found else None,
            "queried_at": None,
            "audited": [],
        }
        rc = self.run(
            stage, "pip-audit-1.log", export, self.checkout,
            STAGE_TIMEOUT_SEC["pip-audit"],
        )  # fmt: skip
        if rc != 0:
            stage["problems"].append(f"uv export exited {rc}")
            stage["ok"] = False
            return
        runs = [(name, path, s) for name, path in audits for s in PIP_AUDIT_SERVICES]
        for i, (name, path, service) in enumerate(runs):
            log = f"pip-audit-{i + 2}.log"
            queried = utc_now()
            stage["advisories"]["queried_at"] = (
                stage["advisories"]["queried_at"] or queried
            )
            argv = self.uv_cmd(
                "run", "--with", "pip-audit", "pip-audit", "-s", service,
                "-r", str(path), "--no-deps", "--disable-pip",
            )  # fmt: skip
            rc = self.run(
                stage, log, argv, self.checkout, STAGE_TIMEOUT_SEC["pip-audit"]
            )
            stage["advisories"]["audited"].append(
                {
                    "file": name,
                    "service": service,
                    "queried_at": queried,
                    "rc": rc,
                    "log": log,
                }
            )
            if rc != 0:
                stage["problems"].append(f"pip-audit ({service}) of {name} exited {rc}")
        stage["ok"] = not stage["problems"]

    def docker_preflight(self, stage: dict[str, Any]) -> bool:
        for i, argv in enumerate(
            (
                [
                    "docker",
                    "info",
                    "--format",
                    "{{.ServerVersion}} {{.OperatingSystem}}",
                ],
                ["docker", "compose", "version"],
                ["docker", "buildx", "version"],
            )
        ):
            if (
                self.run(
                    stage,
                    f"{stage['name']}-docker-{i + 1}.log",
                    argv,
                    self.checkout,
                    120,
                )
                != 0
            ):
                stage["problems"].append(
                    f"{shlex.join(argv[:3])} failed (tools/release_gate/bootstrap_vm.sh"
                    " installs Docker, Compose and buildx)"
                )
                return False
        return True

    def docker(self, stage: dict[str, Any]) -> None:
        select = self.args.docker_select
        stage["subset"] = select != ["tests/"]
        if not self.docker_preflight(stage):
            return
        paths = [s for s in select if s.split("::")[0] != HILLCLIMB]
        if not paths:
            stage["skipped"] = "no Docker test selected besides the hill-climb test"
            return
        # The hill-climb test runs alone (its own stage).
        extra = ["--ignore", HILLCLIMB] if "tests/" in paths else []
        jobs = self.work / "deterministic-jobs"
        rc = self.run(
            stage,
            "docker-tier.log",
            self.pytest(
                *paths,
                *extra,
                "-m",
                DOCKER_MARKERS,
                "-n",
                str(self.args.docker_workers),
                "-v",
                junit="docker-tier.xml",
                basetemp="docker-tier",
            ),
            self.checkout,
            STAGE_TIMEOUT_SEC["docker"],
            {
                "BENCHFLOW_DETERMINISTIC_SANDBOX": "docker",
                "BENCHFLOW_DETERMINISTIC_JOBS_DIR": str(jobs),
            },
        )
        self.judge_pytest(stage, rc, "docker-tier.xml", skips_fail=False)
        if not stage["ok"]:
            copy_records(jobs, self.logs / "docker-tier-jobs")
        drop_tree(self.tmp / "docker-tier")

    def hillclimb(self, stage: dict[str, Any]) -> None:
        if not self.docker_preflight(stage):
            return
        rc = self.run(
            stage,
            "hillclimb.log",
            self.pytest(
                HILLCLIMB,
                "-m",
                DOCKER_MARKERS,
                "--timeout",
                str(self.args.hillclimb_timeout),
                "-v",
                junit="hillclimb.xml",
                basetemp="hillclimb",
            ),
            self.checkout,
            STAGE_TIMEOUT_SEC["hillclimb"],
            {
                "BENCHFLOW_DETERMINISTIC_SANDBOX": "docker",
                "BENCHFLOW_DETERMINISTIC_JOBS_DIR": str(self.work / "hillclimb-jobs"),
            },
        )
        self.judge_pytest(stage, rc, "hillclimb.xml", skips_fail=True)
        if not stage["ok"]:
            copy_records(self.work / "hillclimb-jobs", self.logs / "hillclimb-jobs")
        drop_tree(self.tmp / "hillclimb")

    def judge_pytest(
        self, stage: dict[str, Any], rc: int | str, junit: str, *, skips_fail: bool
    ) -> None:
        outcomes = junit_outcomes(self.logs / "junit" / junit)
        stage["rc"] = rc
        stage["counts"] = c = counts(outcomes)
        stage["failures"] = failed_ids(outcomes)
        stage["skipped_tests"] = sorted(
            t for t, o in (outcomes or {}).items() if o == "skipped"
        )
        reasons = junit_skip_reasons(self.logs / "junit" / junit)
        stage["skip_reasons"] = {t: reasons.get(t, "") for t in stage["skipped_tests"]}
        if c is None:
            stage["problems"].append(f"no test report (exit {rc})")
        else:
            if rc != 0:
                stage["problems"].append(f"pytest exited {rc}")
                if rc == 1 and not c["failed"]:
                    stage["problems"].append("exit 1 but the report has no failure")
            if c["failed"]:
                stage["problems"].append(f"{c['failed']} tests failed")
            if not c["passed"]:
                stage["problems"].append("no test passed (a skip is not a pass)")
            if skips_fail and c["skipped"]:
                stage["problems"].append(f"{c['skipped']} tests skipped")
        stage["ok"] = not stage["problems"]

    def matrix_record(self) -> dict[str, Any] | None:
        path = self.args.matrix_dir / "matrix.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        record: dict[str, Any] = {
            "matrix_json": str(path),
            "matrix_json_md5": md5_file(path),
            "matrix_commit": None,
            "ran": False,
            "blocked": blocked_cells(data),
            "offline_evidence": list(data.get("offline_evidence") or []),
            "failed_runs": [],
        }
        if not self.args.matrix_dir.resolve().is_relative_to(self.checkout):
            try:
                record["matrix_commit"] = git(self.args.matrix_dir, "rev-parse", "HEAD")
                record["matrix_dirty"] = bool(
                    git(self.args.matrix_dir, "status", "--porcelain", "--", ".")
                )
            except (subprocess.CalledProcessError, OSError):
                pass
        return record

    def matrix(self, stage: dict[str, Any], record: dict[str, Any] | None) -> None:
        script = self.args.matrix_dir / "run_matrix.py"
        if record is None or not script.is_file():
            stage["problems"].append(f"no cost matrix at {self.args.matrix_dir}")
            stage["ok"] = False
            return
        # Docker or Daytona alone is a subset of the matrix: never a pass.
        stage["subset"] = self.args.matrix_sandbox != "both"
        out = self.work / "matrix"
        argv = self.uv_cmd(
            "run",
            "--no-sync",
            "python",
            str(script),
            "--bench",
            str(self.checkout / ".venv" / "bin" / "bench"),
            "--secrets",
            str(self.args.secrets),
            "--out",
            str(out),
            "--sandbox",
            self.args.matrix_sandbox,
        )
        rc = self.run(
            stage, "matrix.log", argv, self.checkout, STAGE_TIMEOUT_SEC["matrix"]
        )
        rows = []
        results = out / "results.jsonl"
        if results.is_file():
            rows = [
                json.loads(line)
                for line in results.read_text().splitlines()
                if line.strip()
            ]
        # The matrix's report, its rows, its per-run logs and each trial's
        # records; trajectories stay on the VM.
        copy_records(out, self.logs / "matrix", logs_folder=True)
        record.update(
            ran=True,
            rc=rc,
            runs=len(rows),
            passed=sum(1 for r in rows if r.get("ok")),
            failed_runs=[r for r in rows if not r.get("ok")],
        )
        stage["rc"] = rc
        stage["counts"] = {
            "passed": record["passed"],
            "failed": len(record["failed_runs"]),
            "skipped": len(record["blocked"]),
        }
        if rc != 0:
            stage["problems"].append(f"run_matrix.py exited {rc}")
        if not rows:
            stage["problems"].append("no matrix run recorded")
        if record["failed_runs"]:
            stage["problems"].append(f"{len(record['failed_runs'])} matrix runs failed")
        stage["ok"] = not stage["problems"]

    def upload(self, receipt: dict[str, Any], stamp: str) -> None:
        logs = receipt["logs"]
        logs["scrubbed"] = scrub(self.logs, secret_values(self.args.secrets))
        files = log_files(self.logs)
        logs["files"] = files
        if self.args.gcs_prefix:
            dest = self.destination(receipt, stamp)
            cp = subprocess.run(
                ["gcloud", "storage", "cp", "--recursive", str(self.logs), f"{dest}/"],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            listing = subprocess.run(
                ["gcloud", "storage", "objects", "list", f"{dest}/**", "--format=json"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            try:
                remote = gcs_md5s(json.loads(listing.stdout or "[]"), dest)
            except ValueError:
                remote = {}
            logs["upload_rc"] = cp.returncode
        elif self.args.local_logs_dir:
            leaf = self.destination(receipt, stamp, prefix="")
            target = (self.args.local_logs_dir / leaf).resolve()
            target.mkdir(parents=True)
            shutil.copytree(self.logs, target / self.logs.name)
            dest = target.as_uri()
            remote = {
                p.relative_to(target).as_posix(): md5_file(p)
                for p in target.rglob("*")
                if p.is_file()
            }
        else:
            return
        logs["destination"] = dest
        logs["verified"] = verify(files, remote, dest, self.logs.name)

    def destination(
        self, receipt: dict[str, Any], stamp: str, prefix: str | None = None
    ) -> str:
        """``<prefix>/<version>/<commit12>-<started>``: one folder per gate run."""
        prefix = self.args.gcs_prefix if prefix is None else prefix
        leaf = f"{receipt['version']}/{receipt['source']['commit'][:12]}-{stamp}"
        return f"{prefix.rstrip('/')}/{leaf}" if prefix else leaf

    # -- the whole gate ----------------------------------------------------

    def preflight(self, stamp: str) -> dict[str, Any]:
        a = self.args
        if git(self.checkout, "status", "--porcelain", "--untracked-files=normal"):
            raise Refused("the checkout has uncommitted changes: gate a commit")
        problem = group_problem()
        if problem:
            raise Refused(problem)
        if self.work.resolve().is_relative_to(self.checkout):
            raise Refused("--work must be outside the checkout (logs never go in git)")
        if self.work.exists() and any(self.work.iterdir()):
            raise Refused(f"--work {self.work} is not empty: one gate per folder")
        pin = re.search(
            r'^UV_VERSION = "([^"]+)"',
            (self.checkout / "src" / "benchflow" / "_uv_pin.py").read_text(),
            re.M,
        )
        try:
            found = subprocess.run(
                [self.uv, "--version"], capture_output=True, text=True, timeout=60
            ).stdout.split()
        except OSError:
            found = []
        if pin is None or found[1:2] != [pin.group(1)]:
            raise Refused(
                f"uv {' '.join(found[1:2]) or 'missing'} is not the pinned"
                f" {pin.group(1) if pin else '?'}: run tools/release_gate/bootstrap_vm.sh"
            )
        project = tomllib.loads((self.checkout / "pyproject.toml").read_text())
        version = project["project"]["version"]
        commit = git(self.checkout, "rev-parse", "HEAD")
        # Tests infer a task's provenance from the checkout's origin; a clone
        # made from a bundle or a local path names no repository.
        try:
            origin = git(self.checkout, "remote", "get-url", "origin")
        except subprocess.CalledProcessError:
            origin = ""
        if not GITHUB_ORIGIN.fullmatch(origin):
            raise Refused(
                f"origin is {origin or 'unset'!r}, not the GitHub repository: "
                "git remote set-url origin "
                "https://github.com/benchflow-ai/benchflow-private.git"
            )
        rc = rc_identity(self.checkout, a.rc_branch, commit)
        base: dict[str, Any] = {"commit": None, "failures_file": None, "failures": []}
        if a.base_commit:
            try:
                base["commit"] = git(
                    self.checkout,
                    "rev-parse",
                    "--verify",
                    f"{a.base_commit}^{{commit}}",
                )
            except subprocess.CalledProcessError:
                raise Refused(
                    f"base commit {a.base_commit} is not in this clone: fetch it"
                ) from None
        if a.base_failures:
            ids = read_failure_list(a.base_failures)
            base["failures_file"] = {
                "path": str(a.base_failures),
                "md5": md5_file(a.base_failures),
                "ids": len(ids),
            }
            base["failures"] = ids
        stages = self.leg["stages"]
        if "matrix" in stages and "matrix" not in a.skip:
            if a.secrets is None or not a.secrets.is_dir():
                raise Refused("the matrix needs --secrets DIR (or --skip matrix)")
            if not (a.matrix_dir / "run_matrix.py").is_file():
                raise Refused(f"no tools/cost_matrix/run_matrix.py at {a.matrix_dir}")
        if any(st in stages for st in ("unit", "docker", "hillclimb", "matrix")):
            self.check_docker()
        records = check_records(a.records, version)
        try:
            if records["waivers"]:
                self.waived = load_waivers(
                    Path(records["waivers"]["path"]), version, self.leg_name
                )
                records["waivers"]["for_this_version_and_leg"] = sorted(self.waived)
            if records["classification"]:
                load_classification(Path(records["classification"]["path"]))
        except (OSError, ValueError) as exc:
            raise Refused(str(exc)) from None
        name = f"{version}-{commit[:12]}-{self.leg_name}" + (
            "-dry-run" if a.dry_run else ""
        )
        if (a.records / "receipts" / f"{name}.json").exists():
            raise Refused(f"receipts/{name}.json exists already in the records folder")
        self.work.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        # The tests' folders go under /tmp, as on GitHub's runners: some
        # lockdown tests require every folder above their tmp_path to be
        # root-owned (a home folder is the user's, and 18 tests failed under
        # it). pytest makes each run's --basetemp, but not its parent.
        self.tmp = Path(tempfile.mkdtemp(prefix="benchflow-gate-", dir="/tmp"))
        # The PATH lock leaves every folder under /home alone, and the lock
        # tests then see none of theirs (tests/test_locked_system_path.py).
        if self.tmp.resolve().is_relative_to("/home"):
            raise Refused(f"the tests' folders ({self.tmp}) must not be under /home")
        if a.gcs_prefix:
            # Fail now, not after hours of tests, if the bucket refuses us.
            probe = self.logs / "STARTED.json"
            probe.write_text(json.dumps({"commit": commit, "at": utc_now()}) + "\n")
            leaf = self.destination(
                {"version": version, "source": {"commit": commit}}, stamp
            )
            dest = f"{leaf}/{self.logs.name}/{probe.name}"
            done = subprocess.run(
                ["gcloud", "storage", "cp", str(probe), dest],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if done.returncode:
                raise Refused(
                    f"cannot write to {a.gcs_prefix}: {done.stderr.strip()[-300:]}"
                )
        return {
            "version": version,
            "commit": commit,
            "base": base,
            "name": name,
            "rc": rc,
            **records,
        }

    def check_docker(self) -> None:
        """The leg's Docker: Ubuntu's packages with a proven Compose, or
        Docker CE (bootstrap_vm.sh)."""

        def package(name: str) -> str | None:
            done = subprocess.run(
                ["dpkg-query", "-W", "-f", "${Version}", name],
                capture_output=True, text=True,
            )  # fmt: skip
            return done.stdout.strip() if done.returncode == 0 else None

        compose = subprocess.run(
            ["docker", "compose", "version", "--short"], capture_output=True, text=True
        ).stdout.strip()
        if self.leg["docker"] == "ubuntu":
            if package("docker.io") is None:
                raise Refused("this leg runs on Ubuntu's docker.io: bootstrap_vm.sh")
            if compose not in proven_compose_versions():
                raise Refused(f"Compose {compose or 'missing'} is not a proven version")
        elif package("docker-ce") is None:
            raise Refused("this leg runs on Docker CE: bootstrap_vm.sh --docker ce")

    def python_version(self, tree: Path) -> str | None:
        """The interpreter the synced environment runs."""
        python = tree / ".venv" / "bin" / "python"
        try:
            done = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import platform; print(platform.python_version())",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except OSError:
            return None
        return done.stdout.strip() or None

    def host(self) -> dict[str, Any]:
        def out(*argv: str) -> str | None:
            try:
                done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                return None
            return done.stdout.strip() if done.returncode == 0 else None

        return {
            "hostname": platform.node(),
            "os": out("sh", "-c", ". /etc/os-release && echo $PRETTY_NAME")
            or platform.platform(),
            "kernel": platform.release(),
            "cpus": os.cpu_count(),
            "docker": out("docker", "version", "--format", "{{.Server.Version}}"),
            "compose": out("docker", "compose", "version", "--short"),
            "buildx": out("docker", "buildx", "version"),
            "uv": (out(self.uv, "--version") or "").removeprefix("uv ") or None,
        }

    def main(self) -> int:
        a = self.args
        started = utc_now()
        stamp = started.replace("-", "").replace(":", "")
        facts = self.preflight(stamp)
        self.say(f"gate {facts['name']}: work {self.work}")
        receipt: dict[str, Any] = {
            "schema_version": 2,
            "kind": "benchflow-release-gate-leg-receipt",
            "version": facts["version"],
            "leg": self.leg_name,
            "leg_stages": list(self.leg["stages"]),
            "dry_run": a.dry_run,
            "source": {"commit": facts["commit"], "dirty": False},
            "rc": facts["rc"],
            "release_gate": not facts["rc"]["problems"],
            "base": facts["base"],
            "records": facts["records"],
            "classification": facts["classification"],
            "waivers": facts["waivers"],
            "python": self.python,
            "umask": f"{UMASK:04o}",
            "signals_restored": self.restored_signals,
            "tests_tmp": str(self.tmp),
            "gate_script": {
                "path": "tools/release_gate/run_gate.py",
                "sha256": sha256_file(Path(__file__)),
            },
            "host": self.host(),
            "started_at": started,
            "stages": {},
            "matrix": None,
            "release_note_draft": None,
            "logs": {
                "destination": None,
                "verified": False,
                "files": [],
                "scrubbed": [],
            },
        }
        stages = receipt["stages"]

        def new(name: str) -> dict[str, Any]:
            stages[name] = {"name": name, "problems": [], "ok": False}
            return stages[name]

        setup = new("setup")
        try:
            setup["ok"] = self.setup(self.checkout, setup, "")
        except Exception as exc:
            setup["problems"].append(f"the gate broke: {exc!r}"[:500])
        receipt["python_version"] = self.python_version(self.checkout)
        for name in self.leg["stages"]:
            stage = new(name)
            if name in a.skip:
                stage["skipped"] = "--skip " + name
                continue
            if not setup["ok"]:
                stage["skipped"] = "setup failed"
                continue
            try:
                if name == "static":
                    self.static(stage)
                elif name == "unit":
                    self.unit(stage, receipt)
                elif name == "pip-audit":
                    self.pip_audit(stage)
                elif name == "docker":
                    self.docker(stage)
                elif name == "hillclimb":
                    self.hillclimb(stage)
                elif name == "matrix":
                    receipt["matrix"] = self.matrix_record()
                    self.matrix(stage, receipt["matrix"])
            except Exception as exc:  # the gate's own fault: a failed stage
                # and still a receipt, never a run that leaves none.
                stage["ok"] = False
                stage["problems"].append(f"the gate broke: {exc!r}"[:500])
                self.say(f"{name}: the gate broke: {exc!r}"[:500])
        # Anything that breaks from here on still leaves a receipt, as FAIL.
        try:
            if "matrix" in self.leg["stages"] and receipt["matrix"] is None:
                receipt["matrix"] = self.matrix_record()
            receipt["release_note_draft"] = release_note(
                facts["version"], facts["commit"], receipt["matrix"]
            )
            # A test that writes into the checkout shows here.
            receipt["source"]["dirty_after_run"] = bool(
                git(self.checkout, "status", "--porcelain", "--untracked-files=normal")
            )
            self.say("uploading logs")
            self.upload(receipt, stamp)
        except Exception as exc:
            receipt["gate_error"] = f"{exc!r}"[:500]
        receipt["finished_at"] = utc_now()
        receipt["verdict"], receipt["failures"], receipt["incomplete"] = verdict(
            receipt
        )
        json_path = self.write_receipt(receipt, facts["name"])
        print(
            f"{receipt['verdict']}: receipt {json_path} (sha256 {sha256_file(json_path)})",
            flush=True,
        )
        return {
            "PASS": EXIT_PASS,
            "FAIL": EXIT_FAIL,
            "INCOMPLETE": EXIT_INCOMPLETE,
        }[receipt["verdict"]]

    def write_receipt(self, receipt: dict[str, Any], name: str) -> Path:
        """The receipt into the records folder, scrubbed of every secret
        value; a Markdown that cannot be rendered still leaves the JSON."""
        values = secret_values(self.args.secrets)
        folder = self.args.records / "receipts"
        folder.mkdir(parents=True, exist_ok=True)
        json_path = folder / f"{name}.json"
        json_path.write_text(scrub_text(json.dumps(receipt, indent=2) + "\n", values))
        try:
            text = render_markdown(receipt)
        except Exception as exc:
            receipt["verdict"] = "FAIL"
            text = (
                f"# Release gate receipt: BenchFlow {receipt.get('version')}\n\n"
                f"**FAIL**: the receipt could not be rendered ({exc!r}); "
                f"the JSON beside this file is the record.\n"
            )
            json_path.write_text(
                scrub_text(json.dumps(receipt, indent=2) + "\n", values)
            )
        (folder / f"{name}.md").write_text(scrub_text(text, values))
        return json_path


def github_heads(checkout: Path, *branches: str) -> dict[str, str] | None:
    """``git ls-remote --heads origin [BRANCH...]`` in ``checkout``, as
    GitHub answers it where --combine runs: {branch: commit}, or None when
    it does not answer."""
    try:
        out = subprocess.run(
            ["git", "-C", str(checkout), "ls-remote", "--heads", "origin", *branches],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    heads: dict[str, str] = {}
    for line in out.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.startswith("refs/heads/"):
            heads[ref.removeprefix("refs/heads/")] = sha
    return heads


def lab_problems(
    records: Path, lab: dict[str, Any], version: str, found: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Why the legs' waivers are not lab's, checked in the lab checkout the
    combine runs in (``lab`` is its :func:`check_records`), and what was
    checked. Each leg's records must be a checkout of lab at a commit in
    this checkout's history and on a branch of lab on GitHub, whose
    waivers.json has the MD5 the leg recorded; and every waiver that applied
    to the leg must apply, the same, in waivers.json on lab's main (as GitHub
    names it, fetched here): the lead merges lab, so a waiver on a branch
    alone counts for nothing."""
    top = Path(git(records, "rev-parse", "--show-toplevel"))
    rel = f"{RECORDS_PATH}/{WAIVERS_FILE}"

    def have(sha: str | None) -> bool:
        return (
            bool(sha)
            and subprocess.run(
                ["git", "-C", str(top), "cat-file", "-e", f"{sha}^{{commit}}"],
                capture_output=True,
            ).returncode
            == 0
        )

    def ancestor(sha: str, of: str) -> bool:
        return (
            subprocess.run(
                ["git", "-C", str(top), "merge-base", "--is-ancestor", sha, of],
                capture_output=True,
            ).returncode
            == 0
        )

    def waivers_at(sha: str) -> bytes | None:
        done = subprocess.run(
            ["git", "-C", str(top), "show", f"{sha}:{rel}"], capture_output=True
        )
        return done.stdout if done.returncode == 0 else None

    def md5(data: bytes | None) -> str | None:
        return (
            None
            if data is None
            else hashlib.md5(data, usedforsecurity=False).hexdigest()
        )

    problems: list[str] = []
    head = lab["records"]["commit"]
    heads = github_heads(top)
    main = (heads or {}).get("main")
    main_waivers: bytes | None = None
    if heads is None:
        problems.append(
            "lab on GitHub could not be read: its main and the legs' commits are unchecked"
        )
    elif not have(main):
        problems.append(
            f"lab's main on GitHub ({str(main)[:12]}) is not in this checkout: git fetch origin"
        )
        main = None
    else:
        main_waivers = waivers_at(str(main))
    pushed = [sha for sha in (heads or {}).values() if have(sha)]
    checked: dict[str, Any] = {}
    for leg, r in found.items():
        if r is None:
            continue
        rec = r.get("records") or {}
        sha = rec.get("commit")
        if not LAB_ORIGIN.fullmatch(str(rec.get("repo") or "")):
            problems.append(f"{leg}: its records were not read from a checkout of lab")
            continue
        if not have(sha) or not ancestor(str(sha), head):
            problems.append(
                f"{leg}: its records commit {str(sha)[:12]} is not in this lab checkout's history"
            )
            continue
        if heads is not None and not any(ancestor(str(sha), p) for p in pushed):
            problems.append(
                f"{leg}: its records commit {str(sha)[:12]} is on no branch of lab on GitHub"
            )
        data = waivers_at(str(sha))
        leg_md5 = (r.get("waivers") or {}).get("md5")
        checked[leg] = {"records_commit": sha, "waivers_md5": leg_md5}
        if md5(data) != leg_md5:
            problems.append(
                f"{leg}: its waivers (md5 {leg_md5}) are not {rel} at {str(sha)[:12]} (md5 {md5(data)})"
            )
            continue
        if data is None or main is None:
            continue
        try:
            applied = waivers_from(
                data.decode(), f"{rel}@{str(sha)[:12]}", version, leg
            )
            on_main = (
                waivers_from(main_waivers.decode(), f"{rel}@main", version, leg)
                if main_waivers is not None
                else {}
            )
        except (ValueError, UnicodeDecodeError) as exc:
            problems.append(f"{leg}: {exc}")
            continue
        gone = sorted(t for t, w in applied.items() if on_main.get(t) != w)
        if gone:
            problems.append(
                f"{leg}: {len(gone)} waiver(s) it read are not on lab's main as they were: "
                + ", ".join(gone[:5])
            )
    return problems, {
        "repo": lab["records"]["repo"],
        "commit": head,
        "main_on_github": main,
        "waivers_md5_on_main": md5(main_waivers),
        "legs": checked,
    }


def combine(records: Path, commit: str, version: str, dry_run: bool = False) -> int:
    """Every leg's receipt for ``commit`` into the release receipt: PASS only
    when every leg in legs.json passed. Adds the public summary line only to
    a PASS, which is always a release gate: a FAIL, an INCOMPLETE or a run
    that is not a release gate has none (null in the receipt).

    ``records`` must be lab's records folder (:func:`check_records`), and
    every leg's waivers lab's (:func:`lab_problems`)."""
    lab = check_records(records, version)
    legs = load_legs()
    folder = records / "receipts"
    suffix = "-dry-run" if dry_run else ""
    found: dict[str, dict[str, Any] | None] = {}
    for leg in legs:
        path = folder / f"{version}-{commit[:12]}-{leg}{suffix}.json"
        found[leg] = json.loads(path.read_text()) if path.is_file() else None
    problems = [f"{leg}: no receipt" for leg, r in found.items() if r is None]
    rcs = {leg: (r.get("rc") or {}) for leg, r in found.items() if r is not None}
    problems += [
        f"{leg}: not a release gate"
        for leg, rc in rcs.items()
        if not rc or rc.get("problems")
    ]
    heads = {(rc.get("branch"), rc.get("head")) for rc in rcs.values() if rc}
    if len(heads) > 1:
        problems.append("the legs gated different rc branches or heads")
    rc_combined = next(iter(rcs.values()), {}) if len(heads) == 1 else {}
    remote = None
    if rc_combined.get("branch"):
        # Where the combine runs, GitHub answers: the head must still be it.
        remote = (github_heads(ROOT, rc_combined["branch"]) or {}).get(
            rc_combined["branch"]
        )
        if remote is None:
            problems.append(
                f"{rc_combined['branch']}'s head on GitHub could not be read"
            )
        elif remote != commit:
            problems.append(
                f"{rc_combined['branch']} on GitHub is at {remote[:12]}, not {commit[:12]}"
            )
    lab_found, lab_checked = lab_problems(records, lab, version, found)
    problems += lab_found
    problems += [
        f"{leg}: {r['verdict']}"
        for leg, r in found.items()
        if r is not None and r["verdict"] != "PASS"
    ]
    verdict_ = (
        "PASS"
        if not problems
        else (
            "FAIL"
            if any(r and r["verdict"] == "FAIL" for r in found.values())
            else "INCOMPLETE"
        )
    )
    units = []
    for leg, r in found.items():
        unit = ((r or {}).get("stages") or {}).get("unit") or {}
        c = unit.get("counts") or {}
        if c:
            units.append(
                f"{c.get('passed', 0):,} passed / {c.get('failed', 0):,} failed on {leg}"
            )
    date = max((r["finished_at"][:10] for r in found.values() if r), default="-")
    summary = (
        public_summary(version, commit, units, date) if verdict_ == "PASS" else None
    )
    rows: dict[str, dict[str, Any] | None] = {
        leg: None
        if r is None
        else {
            "verdict": r["verdict"],
            "receipt": f"receipts/{version}-{commit[:12]}-{leg}{suffix}.json",
            "receipt_sha256": sha256_file(
                folder / f"{version}-{commit[:12]}-{leg}{suffix}.json"
            ),
            "python": r.get("python_version") or r.get("python"),
            "docker": (r.get("host") or {}).get("docker"),
            "compose": (r.get("host") or {}).get("compose"),
            "unit": ((r.get("stages") or {}).get("unit") or {}).get("counts"),
        }
        for leg, r in found.items()
    }
    combined: dict[str, Any] = {
        "schema_version": 2,
        "kind": "benchflow-release-gate-receipt",
        "version": version,
        "source": {"commit": commit},
        "rc": {**rc_combined, "remote_head_at_combine": remote},
        "records": lab_checked,
        "pip_audit_advisories": {
            leg: ((r.get("stages") or {}).get("pip-audit") or {}).get("advisories")
            for leg, r in found.items()
            if r is not None
            and ((r.get("stages") or {}).get("pip-audit") or {}).get("advisories")
        },
        "verdict": verdict_,
        "problems": problems,
        "legs": rows,
        "public_summary": summary,
        "combined_at": utc_now(),
    }
    out = folder / f"{version}-{commit[:12]}{suffix}.json"
    out.write_text(json.dumps(combined, indent=2) + "\n")
    lines = [
        f"# Release gate: BenchFlow {version} at `{commit}`",
        "",
        f"**{verdict_}**",
        "",
        *[f"- {p}" for p in problems],
        "",
        "| leg | verdict | Python | Docker | Compose | unit |",
        "|---|---|---|---|---|---|",
    ]
    for leg in legs:
        info = rows[leg]
        if info is None:
            lines.append(f"| {leg} | no receipt | | | | |")
            continue
        u = info["unit"] or {}
        lines.append(
            f"| {leg} | {info['verdict']} | {info['python']} | {info['docker']} |"
            f" {info['compose']} | {u.get('passed', '-')} passed, {u.get('failed', '-')} failed |"
        )
    lines += (
        [
            "",
            "Public summary line (for the public release notes):",
            "",
            f"> {summary}",
        ]
        if summary
        else [
            "",
            f"No public summary line: the release is {verdict_}, and only a"
            " release gate that passed gets one.",
        ]
    )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"{verdict_}: {out} (sha256 {sha256_file(out)})")
    if summary:
        print(summary)
    return {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL}.get(verdict_, EXIT_INCOMPLETE)


def public_summary(version: str, commit: str, units: list[str], date: str) -> str:
    """One line for the public release notes: no paths, no links."""
    text = (
        f"BenchFlow {version} ({commit[:12]}): release gate {date}; unit tests "
        + ("; ".join(units) or "not run")
        + "."
    )
    if "/" in text.replace(" / ", "") or "://" in text:
        raise ValueError(f"the public summary must hold no path or link: {text}")
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="The release-candidate gate, on per-run VMs."
    )
    ap.add_argument("--leg", choices=sorted(load_legs()), default="py3.12")
    ap.add_argument(
        "--records",
        type=Path,
        help="the records folder: lab's ops/releases/benchflow (receipts, waivers)",
    )
    ap.add_argument(
        "--combine",
        action="store_true",
        help="combine every leg's receipt for this checkout's commit, and stop",
    )
    base = ap.add_mutually_exclusive_group()
    base.add_argument("--base-commit", help="run the unit suite on this commit too")
    base.add_argument(
        "--base-failures",
        type=Path,
        help="a recorded list of the base's failing test ids, one per line",
    )
    ap.add_argument(
        "--rc-branch",
        help="the rc branch this run gates: the tested commit must be its head "
        "(origin/BRANCH, fetched); without it the receipt is not a release gate",
    )
    ap.add_argument("--secrets", type=Path, help="the cost matrix's secrets folder")
    logs = ap.add_mutually_exclusive_group()
    logs.add_argument(
        "--gcs-prefix", help="gs://BUCKET/PATH the logs are uploaded under"
    )
    logs.add_argument(
        "--local-logs-dir",
        type=Path,
        help="copy the logs to this folder instead (a dry run: never a PASS)",
    )
    ap.add_argument("--work", type=Path, help="work folder, outside the checkout")
    ap.add_argument("--unit-select", nargs="+", default=["tests/"])
    ap.add_argument("--docker-select", nargs="+", default=["tests/"])
    ap.add_argument("--docker-workers", type=int, default=4)
    ap.add_argument(
        "--unit-timeout", type=int, default=900, help="seconds per unit test"
    )
    ap.add_argument("--hillclimb-timeout", type=int, default=2400)
    ap.add_argument("--matrix-dir", type=Path, default=ROOT / "tools" / "cost_matrix")
    ap.add_argument(
        "--matrix-sandbox", choices=("docker", "daytona", "both"), default="both"
    )
    ap.add_argument("--skip", action="append", choices=STAGES, default=[])
    ap.add_argument(
        "--dry-run", action="store_true", help="label the receipt a dry run"
    )
    ap.add_argument(
        "--render-failures",
        type=Path,
        metavar="JSON",
        help="write a failure table's Markdown beside it, and stop",
    )
    args = ap.parse_args(argv)
    if args.gcs_prefix and not args.gcs_prefix.startswith("gs://"):
        ap.error("--gcs-prefix must start with gs://")
    if args.records is None and not args.render_failures:
        ap.error("--records names the records folder (lab's ops/releases/benchflow)")
    if args.records is not None and args.records.resolve().is_relative_to(ROOT):
        ap.error("--records must be outside this repository: its tree goes public")
    if args.work is None:
        args.work = (
            Path.home()
            / "release-gate"
            / dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render_failures:
        data = load_classification(args.render_failures)
        args.render_failures.with_suffix(".md").write_text(render_failures(data))
        return 0
    if args.combine:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        try:
            return combine(
                args.records,
                git(ROOT, "rev-parse", "HEAD"),
                project["project"]["version"],
                args.dry_run,
            )
        except Refused as exc:
            print(f"run_gate: refused: {exc}", file=sys.stderr)
            return EXIT_REFUSED
    os.umask(UMASK)
    restored = restore_stop_signals()
    gate = Gate(args)
    gate.restored_signals = restored
    code: int | str = "error"
    try:
        code = gate.main()
    except Refused as exc:
        print(f"run_gate: refused: {exc}", file=sys.stderr)
        code = EXIT_REFUSED
    finally:
        # The finish marker, also when the gate itself broke ("error").
        if args.work.is_dir():
            (args.work / "FINISHED").write_text(f"{code}\n")
    return code if isinstance(code, int) else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
