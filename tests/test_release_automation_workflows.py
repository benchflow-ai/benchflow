"""The release gate runs in CI on the exact commit (lead ruling, 2026-11-18).

0.10.0rc4's VM gate passed and its tag's own GitHub tests then failed: the two
were different gates. These rows pin the hand-run gates as CI jobs, what the
publish waits on, and the rules that keep secrets away from pull requests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.load((WF / name).read_text(), Loader=yaml.BaseLoader)


def _runs(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


def _all() -> dict[str, dict]:
    return {
        p.name: yaml.load(p.read_text(), Loader=yaml.BaseLoader)
        for p in sorted(WF.glob("*.yml"))
    }


# -- test.yml: the hand gates as jobs -----------------------------------------------------------------


def test_every_test_job_has_a_time_limit():
    jobs = _load("test.yml")["jobs"]
    missing = [name for name, job in jobs.items() if "timeout-minutes" not in job]
    assert missing == [], missing


def test_the_grading_matrix_runs_in_two_shards_without_the_floods():
    job = _load("test.yml")["jobs"]["grading-matrix"]
    assert job["strategy"]["matrix"]["shard"] == ["1", "2"]
    runs = _runs(job)
    assert "tests/test_grading_integrity_matrix.py" in runs
    assert "not flood" in runs and "--splits 2" in runs
    assert "build_bfx.sh" in runs and "bootstrap_vm.sh" in runs


def test_the_grading_matrix_also_runs_as_a_run():
    job = _load("test.yml")["jobs"]["grading-matrix-run-defaults"]
    env = {k: v for step in job["steps"] for k, v in (step.get("env") or {}).items()}
    assert env.get("BENCHFLOW_TEST_RUN_DEFAULTS") == "1"
    assert "tests/test_grading_integrity_matrix.py" in _runs(job)


def test_the_window_check_runs_host_side_on_the_gates_docker():
    job = _load("test.yml")["jobs"]["window-check"]
    runs = _runs(job)
    assert "tests/test_egress_window_hostside.py" in runs
    assert "bootstrap_vm.sh" in runs and "build_bfx.sh" in runs


def test_oracle_parity_samples_tallies_and_uploads():
    job = _load("test.yml")["jobs"]["oracle-parity"]
    runs = _runs(job)
    for step in (
        "tools.parity.sample",
        "tools.parity.fetch",
        "tools.parity.run",
        "tools.parity.tally",
    ):
        assert step in runs, step
    assert "HARBOR_TELEMETRY" in yaml.safe_dump(job)


def test_no_upload_names_a_path_through_dot_dot():
    """upload-artifact refuses a path with "." or ".." in it ("Relative pathing '.' and '..' is not allowed"), so
    a folder set as ${{ github.workspace }}/../x lost every run's rows (oracle-parity on 0c6fc44e). The env
    values a workflow and its job set are read into each upload's paths."""
    for name, wf in _all().items():
        for job_name, job in (wf.get("jobs") or {}).items():
            env = {**(wf.get("env") or {}), **(job.get("env") or {})}
            for step in job.get("steps", []):
                if "actions/upload-artifact" not in step.get("uses", ""):
                    continue
                paths = (step.get("with") or {}).get("path", "")
                for key, value in env.items():
                    paths = re.sub(
                        r"\$\{\{\s*env\." + re.escape(key) + r"\s*\}\}", value, paths
                    )
                for line in paths.splitlines():
                    parts = line.strip().split("/")
                    assert ".." not in parts and "." not in parts, (
                        name,
                        job_name,
                        line.strip(),
                    )


def test_the_tree_guard_runs_on_every_pull_request():
    job = _load("test.yml")["jobs"]["public-tree-guard"]
    assert "tools/public_tree_guard.py" in _runs(job)


def test_two_builds_are_compared_byte_for_byte():
    jobs = _load("test.yml")["jobs"]
    runs = _runs(jobs["repro-build"])
    assert runs.count("uv build --no-sources") >= 2 or "repro_build" in runs
    assert "sha256sum" in runs or "cmp" in runs


# -- what the publish waits on ------------------------------------------------------------------------


def test_the_publish_waits_on_the_gate_receipts():
    release = _load("public-release.yml")
    publish = release["jobs"]["publish"]
    assert publish["needs"] == ["tests", "gate-receipts"]
    gate = release["jobs"]["gate-receipts"]
    assert "tools/check_gate_receipts.py" in _runs(gate)
    text = yaml.safe_dump(gate)
    assert "GATE_RECEIPTS_TOKEN" in text
    assert gate["permissions"] == {"contents": "read"}


# -- new workflows --------------------------------------------------------------------------------------


def test_the_secret_sweep_reads_the_range_with_no_secrets():
    wf = _load("secret-sweep.yml")
    assert {"pull_request", "push", "workflow_dispatch"} <= set(wf["on"])
    assert set(wf["on"]["workflow_dispatch"]["inputs"]) >= {"base", "tip"}
    text = (WF / "secret-sweep.yml").read_text()
    assert re.findall(r"secrets\.([A-Z_]+)", text) in ([], ["GITHUB_TOKEN"])
    job = next(iter(wf["jobs"].values()))
    runs = _runs(job)
    assert "tools/secret_sweep.py" in runs and "install_leak_scanners.sh" in runs
    checkout = next(s for s in job["steps"] if "actions/checkout" in s.get("uses", ""))
    assert checkout["with"]["fetch-depth"] == "0"


def test_the_dry_run_does_everything_but_publish():
    wf = _load("release-dryrun.yml")
    assert "pull_request" in wf["on"] and "workflow_dispatch" in wf["on"]
    assert ".github/release-acceptance/**" in wf["on"]["pull_request"]["paths"]
    runs = "\n".join(_runs(job) for job in wf["jobs"].values())
    for step in (
        "public_tree_guard.py",
        "release_version.py public-release",
        "check_dist_bfx.py",
        "check_release_acceptance.py",
        "build_bfx.sh",
    ):
        assert step in runs, step
    assert "uv publish" not in runs and "gh release" not in runs
    assert "secrets." not in (WF / "release-dryrun.yml").read_text()


def test_the_leak_gate_preflight_refuses_without_its_environment():
    wf = _load("leak-gate-preflight.yml")
    assert "pull_request" not in wf["on"] and "pull_request_target" not in wf["on"]
    job = wf["jobs"]["leak-gate"]
    assert job["environment"] == "release-leak-gate"
    first = job["steps"][0]
    assert "RELEASE_LEAK_GATE_ENV" in yaml.safe_dump(first)
    assert "exit 1" in first.get("run", "")


def test_the_nightly_never_runs_on_a_pull_request():
    wf = _load("nightly.yml")
    assert set(wf["on"]) == {"schedule", "workflow_dispatch"}
    names = set(wf["jobs"])
    assert {
        "g1-full",
        "matrix-floods",
        "window-full",
        "dogfood",
        "publish-nightly",
    } <= names
    publish = wf["jobs"]["publish-nightly"]
    assert "release_version.py nightly" in _runs(publish)


def test_the_published_bytes_are_checked_from_pypi():
    wf = _load("post-publish-verify.yml")
    assert wf["on"]["workflow_run"]["workflows"] == ["public-release"]
    runs = "\n".join(_runs(job) for job in wf["jobs"].values())
    assert "constraints.txt" in runs and "bench doctor" in runs
    assert "secrets." not in (WF / "post-publish-verify.yml").read_text()


@pytest.mark.parametrize("name", ["post-publish-verify.yml", "nightly.yml"])
def test_the_readme_oracle_check_reads_the_end_of_run_table(name):
    """The oracle step checks `1/1 passed` under the table's `Score` column, as README step 2
    says; v0.10.1's post-publish run went red on the old `Score: 1/1` line, which 0.10.0rc6 on
    never prints."""
    text = (WF / name).read_text()
    assert "Score: 1/1" not in text
    assert "^1/1 passed( |$)" in text


def test_the_post_publish_install_retry_asks_pypi_again():
    """Each retry of the install from PyPI refreshes benchflow's index page: uv otherwise
    reuses the cached page (PyPI sends a 10-minute max-age) and never sees the new version."""
    text = (WF / "post-publish-verify.yml").read_text()
    assert (
        'uv tool install --python 3.12 --refresh-package benchflow "benchflow==$VERSION"'
        in text
    )


# -- rules for every workflow ---------------------------------------------------------------------------


def test_no_workflow_runs_pull_request_target():
    """Since 2025-12-08 GitHub runs it from the default branch's file; and it
    hands a fork's code the repository's secrets."""
    for name, wf in _all().items():
        assert "pull_request_target" not in (wf.get("on") or {}), name


@pytest.mark.parametrize("name", sorted(p.name for p in WF.glob("*.yml")))
def test_a_pull_request_never_reaches_a_self_hosted_runner(name):
    wf = _all()[name]
    if "pull_request" not in (wf.get("on") or {}):
        return
    for job_name, job in wf["jobs"].items():
        runs_on = str(job.get("runs-on", ""))
        assert runs_on.startswith(("ubuntu-", "${{ matrix")) or not runs_on, (
            name,
            job_name,
            runs_on,
        )
        assert "self-hosted" not in runs_on, (name, job_name)


NODE24 = {
    "actions/checkout": {"fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"},
    "actions/upload-artifact": {"b7c566a772e6b6bfb58ed0dc250532a479d7789f"},
    "actions/download-artifact": {"37930b1c2abaa49bbe596cd826c3c89aef350131"},
    "azure/login": {"a641126d1b8aa4d1fa005f4f92df94a3a4c4c906"},
    "actions/github-script": {"ed597411d8f924073f98dfc5c65a23a2325f34cd"},
    "actions/setup-python": {"e797f83bcb11b83ae66e0230d6156d7c80228e7c"},
}


def test_actions_run_on_node_24():
    """The Node 20 runtime is deprecated: GitHub already forces these onto
    Node 24 with a warning and will remove it."""
    old = []
    for path in sorted(WF.glob("*.yml")):
        for ref in re.findall(r"uses:\s*([\w./-]+)@([0-9a-f]{40})", path.read_text()):
            action, sha = ref
            if action in NODE24 and sha not in NODE24[action]:
                old.append((path.name, action, sha[:9]))
    assert old == [], old


# -- the docs ------------------------------------------------------------------------------------------


def test_the_release_docs_name_ci_as_the_gate_and_the_checklist():
    release = (ROOT / "docs" / "contributing" / "release.md").read_text()
    assert "never instead" not in release
    assert "release-checklist.md" in release
    assert "## Cadence" in release
    checklist = (ROOT / "docs" / "contributing" / "release-checklist.md").read_text()
    for heading in (
        "## The owner and the cut",
        "## The version commit",
        "## Gate links",
        "## The exact artifacts",
        "## Release notes",
        "## Docs",
        "## The push",
        "## Publish",
        "## PyPI checks",
        "## Known failures",
        "## If it goes wrong",
    ):
        assert heading in checklist, heading
