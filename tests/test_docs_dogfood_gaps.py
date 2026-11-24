"""What 0.10.0rc4's dogfood found in the docs, read as a new Linux user would.

- No Linux Docker, Compose or buildx commands, and no docker-group step: only
  `bench doctor`'s fix lines named them.
- The docs named 0.10.0rc3 as the pre-release to install after rc4.
- README's "Run from Python" led to a page with no Python.
- README's "Branching" led to a page with no branching.
- README step 4 read `jobs/claude`, which the API-key path never writes.
- README said a checkout without bfx runs Docker trials; a model agent's is
  refused before its turn (`integrity_unavailable`).
- When a trial fails listed no row for `integrity_unavailable` or
  `provenance_incomplete`, the two codes a grading-integrity check ends a
  trial with.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
INSTALL = (ROOT / "docs" / "start" / "install.md").read_text()
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def _section(text: str, heading: str) -> str:
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert m, heading
    return m.group(1)


def test_install_names_the_linux_docker_steps_doctor_names():
    linux = _section(INSTALL, "Docker on Linux")
    doctor = (ROOT / "src" / "benchflow" / "doctor.py").read_text()
    for word in (
        "docker.io",
        "docker-compose-v2",
        "docker-buildx",
        "usermod -aG docker",
    ):
        assert word in linux, word
    for word in ("docker-compose-v2", "docker-buildx", "usermod -aG docker"):
        assert word in doctor, word
    assert "29" in linux  # the minimum Docker Engine for the network policies
    assert "install.md#docker-on-linux" in README


def test_the_docs_name_this_trees_pre_release():
    """One version for "the pre-release to install": pyproject's."""
    nav = json.loads((ROOT / "docs" / "nav.json").read_text())
    why = (ROOT / "docs" / "start" / "why-benchflow.md").read_text()
    if "rc" in VERSION:
        assert nav.get("prerelease") == VERSION
        named = set(re.findall(r"0\.10\.0rc\d+", why))
        assert named <= {VERSION}, named
        assert set(re.findall(r"benchflow==(0\.10\.0rc\d+)", INSTALL)) == {VERSION}
    else:
        assert not nav.get("prerelease")


def test_run_from_python_leads_to_python():
    m = re.search(
        r"\[Run from Python\]\(\./docs/start/quick-start\.md#([a-z-]+)\)", README
    )
    assert m, "README's Run from Python link names its section"
    page = (ROOT / "docs" / "start" / "quick-start.md").read_text()
    section = _section(page, "From Python")
    assert m.group(1) == "from-python"
    assert "```python" in section and "import benchflow" in section


def test_branching_leads_to_a_branching_section():
    assert "docs/train/environment.md#branch-a-rollout-at-a-checkpoint" in README
    page = (ROOT / "docs" / "train" / "environment.md").read_text()
    section = _section(page, "Branch a rollout at a checkpoint")
    assert "bench branch" in section and "--from-checkpoint" in section
    assert (
        "docs/examples/python-sdk/run-branch.py" in section
        or "run-branch.py" in section
    )


def test_step_four_reads_the_folder_every_model_run_writes():
    assert "jobs/claude" not in README
    model_runs = re.findall(r"--jobs-dir (jobs/\w+)", README)
    assert model_runs.count("jobs/model") == 2, model_runs
    assert "bench eval metrics jobs/model" in README
    assert "bench view jobs/model" in README


def test_the_readme_says_what_a_checkout_without_bfx_does():
    assert "runs without them, as with" not in README
    assert "Docker uses the fallback root steps" not in README
    assert "integrity_unavailable" in README


def test_failed_trials_lists_the_integrity_codes():
    page = (ROOT / "docs" / "evaluate" / "failed-trials.md").read_text()
    table = _section(page, "Codes and what to do")
    rows = {
        m.group(1): m.group(0) for m in re.finditer(r"^\| `(\w+)` \|.*$", table, re.M)
    }
    unavailable = rows["integrity_unavailable"]
    for words in (
        "| errored or unscored |",
        "BenchFlow",
        "never a 0",
        "make bfx",
        "bench eval resume <job>",
    ):
        assert words in unavailable, words
    incomplete = rows["provenance_incomplete"]
    for words in (
        "| scored |",
        "counts 0",
        "Verifier scored 0: provenance incomplete:",
    ):
        assert words in incomplete, words
