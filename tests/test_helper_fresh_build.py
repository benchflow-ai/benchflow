"""The daily fresh build of the metadata helper (helper-fresh-build.yml).

A new Docker host builds the helper from the module's recipe; Alpine can make
that build fail, or fall back from the exact pins, with no change here. The
job builds the recipe itself, read from the module, and goes red in one step
when the build fails and in another when the exact pins no longer resolve.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from benchflow.sandbox import _metadata_helper as helper_mod
from tools.ci import helper_fresh_build as hfb

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "helper-fresh-build.yml"
SCRIPT = ROOT / "tools" / "ci" / "helper_fresh_build.py"
IMAGE = "sha256:" + "c" * 64
BASE_WORLD = ["alpine-baselayout", "alpine-keys", "apk-tools", "busybox"]


def _workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def _steps() -> list[dict]:
    return _workflow()["jobs"]["fresh-build"]["steps"]


def test_it_runs_daily_and_by_hand_with_no_secrets():
    wf = _workflow()
    assert wf["on"]["schedule"][0]["cron"]
    assert "workflow_dispatch" in wf["on"]
    assert "pull_request_target" not in wf["on"]
    assert wf["permissions"] == {"contents": "read"}
    assert "secrets." not in WORKFLOW.read_text()
    (job,) = wf["jobs"].values()
    assert "permissions" not in job
    assert job["runs-on"].startswith("ubuntu-")
    assert 0 < int(job["timeout-minutes"]) <= 60
    assert job["env"]["HARBOR_TELEMETRY"] == "off"
    assert job["env"]["DO_NOT_TRACK"] == "1"


def test_it_also_runs_on_a_change_to_the_recipe_or_itself():
    paths = _workflow()["on"]["pull_request"]["paths"]
    assert "src/benchflow/sandbox/_metadata_helper.py" in paths
    assert "tools/ci/helper_fresh_build.py" in paths
    assert ".github/workflows/helper-fresh-build.yml" in paths


def test_the_pins_step_runs_only_after_the_build_step_passed():
    """Red at the build: fresh hosts are broken. Red at the pins: they are
    served by the fallback. The pins step never runs after a failed build,
    so the two read apart in the run's list of steps."""
    runs = [step.get("run", "") for step in _steps()]
    build = next(i for i, r in enumerate(runs) if "helper_fresh_build.py build" in r)
    pins = next(i for i, r in enumerate(runs) if "helper_fresh_build.py pins" in r)
    assert build < pins
    assert "if" not in _steps()[pins]
    assert runs[build].split()[-1] == runs[pins].split()[-1]  # the same folder


def test_the_job_builds_the_modules_recipe_never_a_copy():
    """The workflow and its script name no base, no package and no apk line:
    the Dockerfile is the module's own text."""
    assert hfb.HELPER_DOCKERFILE is helper_mod.HELPER_DOCKERFILE
    for text in (WORKFLOW.read_text(), SCRIPT.read_text()):
        assert "alpine:" not in text and "apk add --" not in text
        assert helper_mod.HELPER_BASE.split("@")[1] not in text
        for pin in helper_mod.HELPER_PACKAGES:
            assert pin not in text and pin.split("=")[1] not in text, pin


class FakeDocker:
    """docker for the script: a build that gives IMAGE (or fails), and the
    image's world, `apk info -v` and smoke as given."""

    def __init__(self, *, world=None, builds=None, build_rc=0, smoke_rc=0):
        self.world = BASE_WORLD + list(
            helper_mod.HELPER_PACKAGES if world is None else world
        )
        self.builds = (
            dict(pin.split("=", 1) for pin in helper_mod.HELPER_PACKAGES)
            | {"busybox": "1.37.0-r31"}
            | (builds or {})
        )
        self.build_rc = build_rc
        self.smoke_rc = smoke_rc
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, args, *, stdin=None, timeout=0):
        self.calls.append((list(args), stdin))
        out, rc = "", 0
        if args[0] == "build":
            rc = self.build_rc
            if rc == 0:
                Path(args[args.index("--iidfile") + 1]).write_text(IMAGE)
            out = "ERROR: unable to select packages\n" if rc else "#5 OK\n"
        elif args[0] == "run" and args[-1] == "cat /etc/apk/world":
            out = "\n".join(self.world) + "\n"
        elif args[0] == "run" and args[-1] == "apk info -v":
            out = "".join(f"{n}-{v}\n" for n, v in self.builds.items())
        elif args[0] == "run":
            out, rc = "iptables v1.8.13 (nf_tables)\n", self.smoke_rc
        return subprocess.CompletedProcess(args, rc, out, "")


@pytest.fixture
def summary(tmp_path, monkeypatch):
    path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    return path


def _run(monkeypatch, tmp_path, docker: FakeDocker) -> tuple[int, int]:
    monkeypatch.setattr(hfb, "docker", docker)
    out = tmp_path / "out"
    built = hfb.main(["x", "build", str(out)])
    return built, hfb.main(["x", "pins", str(out)])


def test_the_build_is_the_modules_recipe_with_no_cache(monkeypatch, tmp_path, summary):
    docker = FakeDocker()
    assert _run(monkeypatch, tmp_path, docker) == (0, 0)
    (args, stdin), *rest = [c for c in docker.calls if c[0][0] == "build"]
    assert rest == []
    assert stdin == helper_mod.HELPER_DOCKERFILE
    assert {"--no-cache", "--pull"} <= set(args)
    assert f"{helper_mod.HELPER_RECIPE_LABEL}={helper_mod.HELPER_RECIPE}" in args
    written = (tmp_path / "out" / "helper.Dockerfile").read_bytes()
    assert hashlib.sha256(written).hexdigest()[:16] == helper_mod.HELPER_RECIPE
    assert docker.calls[-1][0] == ["image", "rm", "--force", IMAGE]
    assert not any(c[0][0] in ("push", "tag", "login") for c in docker.calls)
    text = summary.read_text()
    assert "## The metadata helper builds fresh" in text
    assert "## The exact pins still resolve" in text
    assert json.loads((tmp_path / "out" / "result.json").read_text())["tier"] == 1


def test_a_dropped_pin_fails_the_pins_step_and_names_the_builds_served(
    monkeypatch, tmp_path, summary
):
    """Alpine replaced python3's pinned build: the image came from the
    fallback tier, so the build step passes and the pins step fails, with
    the build to pin now."""
    docker = FakeDocker(
        world=helper_mod.HELPER_TIERS[1], builds={"python3": "3.14.11-r0"}
    )
    assert _run(monkeypatch, tmp_path, docker) == (0, 1)
    text = summary.read_text()
    assert "## The metadata helper builds fresh" in text
    assert f"from tier 2 of {len(helper_mod.HELPER_TIERS)}" in text
    assert "## The exact pins no longer resolve; fresh hosts are served" in text
    assert '"python3=3.14.11-r0",' in text
    assert '"openssl=' not in text


def test_a_failed_build_fails_the_build_step_as_fresh_hosts_broken(
    monkeypatch, tmp_path, summary
):
    docker = FakeDocker(build_rc=1)
    assert _run(monkeypatch, tmp_path, docker) == (1, 2)
    text = summary.read_text()
    assert "## Fresh hosts can't build the metadata helper" in text
    assert "unable to select packages" in text
    assert "pins" not in text and "builds fresh" not in text
    assert not any(c[0][0] == "run" for c in docker.calls)


@pytest.mark.parametrize(
    "docker",
    [
        pytest.param(FakeDocker(smoke_rc=1), id="tools-fail"),
        pytest.param(FakeDocker(world=["python3"]), id="world-names-no-tier"),
    ],
)
def test_an_image_that_is_not_the_recipes_fails_the_build_step(
    monkeypatch, tmp_path, summary, docker
):
    assert _run(monkeypatch, tmp_path, docker) == (1, 2)
    assert "builds, but its image is not what the recipe says" in summary.read_text()
    assert "pins" not in summary.read_text()
    assert docker.calls[-1][0] == ["image", "rm", "--force", IMAGE]


def test_the_world_names_the_tier_that_installed():
    exact, series = helper_mod.HELPER_TIERS
    assert hfb.tier_of(BASE_WORLD + list(exact)) == 1
    assert hfb.tier_of(BASE_WORLD + list(series)) == 2
    assert hfb.tier_of(BASE_WORLD + list(exact[1:]) + [series[0]]) is None
    assert hfb.tier_of(BASE_WORLD) is None
    assert hfb.installed(["python3-3.14.10-r2", "iptables-legacy-1.8.13-r0"]) == {
        "python3": "3.14.10-r2",
        "iptables-legacy": "1.8.13-r0",
    }
