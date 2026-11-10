"""The reviewer's PDF tools, its image record, and the default left as it was.

A rubric reviewer judging a written deliverable needs the submitted PDF's
text and page images, but the default reviewer sandbox has pypdf and no
PyMuPDF or poppler, and every image but the default got no packages at all
(review/wrapper.py ``if image == REVIEWER_IMAGE`` at a61a26f4). These tests
guard the opt-in ``ReviewerConfig.pdf_tools``, the known-image package map
that replaced that gate, the image probe and digest in the review record, and
that a reviewer with no new option set gets the task a61a26f4 gave it, byte
for byte (tests/fixtures/review_default/, written by a61a26f4's code).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import benchflow
from benchflow.cli.main import app
from benchflow.review import probe
from benchflow.review.config import load_rubric
from benchflow.review.options import REVIEWER_IMAGE, ReviewerConfig
from benchflow.review.runner import run_review, run_reviews
from benchflow.review.wrapper import (
    KNOWN_REVIEWER_IMAGES,
    REVIEWER_ARTIFACT_PACKAGES,
    REVIEWER_PDF_IMAGE,
    REVIEWER_PDF_PACKAGES,
    reviewer_setup_packages,
)
from tests.test_review_automatic import auth as auth
from tests.test_review_automatic import prepared as prepared
from tests.test_review_automatic import saved as saved
from tests.test_review_runtime import (
    WEIGHTED_RUBRIC,
    FakeRun,
    good_review,
    good_weighted_review,
    make_rollout,
    make_task,
)
from tests.test_review_runtime_automatic import make_bundle

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "fixtures" / "review_default"
DOCKERFILE = REPO / "docker" / "reviewer-pdf" / "Dockerfile"
BASE = list(REVIEWER_ARTIFACT_PACKAGES)
WITH_PDF = BASE + list(REVIEWER_PDF_PACKAGES)


class ProbingRun(FakeRun):
    """FakeRun that also runs the reviewer's last pre-agent hook (the probe)
    against a sandbox whose ``exec`` answers with *probe_stdout*."""

    def __init__(self, *, probe_stdout: str, **kwargs):
        super().__init__(**kwargs)
        self.probe_stdout = probe_stdout

    async def __call__(self, config):
        stdout = self.probe_stdout

        class Sandbox:
            async def exec(self, command, **kwargs):
                return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

        await config.pre_agent_hooks[-1](Sandbox())
        return await super().__call__(config)


async def _detached(root: Path, monkeypatch, **options) -> tuple[FakeRun, Path]:
    task = make_task(root, with_rubric=True)
    rollout = make_rollout(root / "jobs", "rollout-a", task_path=task)
    fake = FakeRun(review_payload=good_review())
    monkeypatch.setattr(benchflow, "run", fake)
    await run_reviews(
        rollout,
        agent="gemini",
        model="gemini/test-model",
        out_dir=root / "out",
        tasks_root=root / "tasks",
        **options,
    )
    return fake, root / "out"


async def _automatic(root: Path, monkeypatch, config: ReviewerConfig, fake=None):
    task = make_task(root, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(root / "jobs", "rollout-a", reward=1.0)
    fake = fake or FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(benchflow, "run", fake)
    rubric_path = task / "verifier/rubric.json"
    trial = await run_review(
        source,
        task,
        load_rubric(rubric_path),
        rubric_path,
        config,
        root / "out",
        deterministic_pass=True,
        workspace_bundle=make_bundle(root),
    )
    return fake, trial, root / "out"


def _record(out: Path) -> dict:
    (path,) = out.rglob("reviewer-environment.json")
    return json.loads(path.read_text())


def _setup_command(task_md: str) -> str | None:
    found = re.search(r"^    - command: (.+)$", task_md, re.MULTILINE)
    return found.group(1) if found else None


async def test_the_default_reviewer_task_is_byte_identical(tmp_path, monkeypatch):
    """Guards the opt-in default against a61a26f4: with no new option the
    reviewer gets exactly the task.md a61a26f4 gave it, its instruction, image
    and setup command included, on the detached and the automatic path."""
    fake, _ = await _detached(tmp_path / "detached", monkeypatch)
    assert fake.task_docs[0] == (GOLDEN / "detached.md").read_text(encoding="utf-8")

    fake, _, _ = await _automatic(
        tmp_path / "automatic",
        monkeypatch,
        ReviewerConfig(agent="codex", model="azure/gpt5.6terra"),
    )
    assert fake.task_docs[0] == (GOLDEN / "automatic.md").read_text(encoding="utf-8")


async def test_the_default_record_keeps_its_fields_and_adds_the_image(
    tmp_path, monkeypatch
):
    """The record's a61a26f4 fields hold their values; the digest is the one
    the default image reference pins."""
    _, _, out = await _automatic(
        tmp_path, monkeypatch, ReviewerConfig(agent="codex", model="m")
    )
    record = _record(out)
    assert {key: record[key] for key in list(record)[:7]} == {
        "image": REVIEWER_IMAGE,
        "artifact_packages": BASE,
        "open_network": False,
        "memory_mb": 2048,
        "storage_mb": 4096,
        "setup_timeout_sec": 120,
        "evidence_timeout_sec": 600,
    }
    assert record["image_digest"] == REVIEWER_IMAGE.split("@")[1]
    assert record["image_id"] is None
    assert record["pdf_tools"] is False


async def test_pdf_tools_installs_pymupdf_on_the_default_image(tmp_path, monkeypatch):
    fake, _, out = await _automatic(
        tmp_path,
        monkeypatch,
        ReviewerConfig(agent="codex", model="m", pdf_tools=True),
    )
    assert _setup_command(fake.task_docs[0]) == (
        "python3 -m pip install --disable-pip-version-check --no-cache-dir "
        "numpy==2.2.6 pypdf==5.9.0 PyMuPDF==2.1"
    )
    assert _record(out)["artifact_packages"] == WITH_PDF
    assert _record(out)["pdf_tools"] is True


@pytest.mark.parametrize(
    ("image", "pdf_tools", "installed"),
    [
        (REVIEWER_IMAGE, False, BASE),
        (REVIEWER_IMAGE, True, WITH_PDF),
        (REVIEWER_PDF_IMAGE, False, []),
        (REVIEWER_PDF_IMAGE, True, []),
        ("ghcr.io/example/reviewer:1", False, []),
        ("ghcr.io/example/reviewer:1", True, WITH_PDF),
    ],
)
def test_the_known_image_package_map(image, pdf_tools, installed):
    """Replaces a61a26f4's gate (packages only on the default image): a known
    image gets what it lacks; another image nothing, as before, unless
    pdf_tools asks."""
    assert list(reviewer_setup_packages(image, pdf_tools=pdf_tools)) == installed


async def test_an_image_that_needs_nothing_gets_no_setup_command(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "local_image_id", lambda image: None)
    fake, _, out = await _automatic(
        tmp_path,
        monkeypatch,
        ReviewerConfig(agent="codex", model="m", image=REVIEWER_PDF_IMAGE),
    )
    assert _setup_command(fake.task_docs[0]) is None
    assert "setup_commands" not in fake.task_docs[0]
    assert _record(out)["artifact_packages"] == []


def test_the_pdf_image_tag_follows_its_dockerfile():
    """REVIEWER_PDF_IMAGE is benchflow-reviewer-pdf:<first 12 hex of the
    Dockerfile's sha256>, built from the default reviewer image with the
    packages the map says it holds."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    digest = hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()
    assert f"benchflow-reviewer-pdf:{digest[:12]}" == REVIEWER_PDF_IMAGE
    assert re.search(r"^FROM (\S+)$", text, re.MULTILINE).group(1) == REVIEWER_IMAGE
    pins = re.findall(r"\b[A-Za-z][\w.-]*==[\w.]+", text)
    assert pins == list(KNOWN_REVIEWER_IMAGES[REVIEWER_PDF_IMAGE]) == WITH_PDF
    assert "poppler-utils" in text


def test_image_digest_reads_only_a_pinned_reference():
    assert probe.image_digest(REVIEWER_IMAGE) == REVIEWER_IMAGE.split("@")[1]
    assert probe.image_digest("python:3.13-slim") is None
    assert probe.image_digest("python@sha256:abc") is None
    assert probe.image_digest("@sha256:" + "0" * 64) is None


def test_local_image_id(monkeypatch):
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    assert probe.local_image_id(REVIEWER_PDF_IMAGE) is None

    image_id = "sha256:" + "a" * 64
    calls = []

    def inspect(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=image_id + "\n")

    monkeypatch.setattr(probe.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(probe.subprocess, "run", inspect)
    assert probe.local_image_id(REVIEWER_PDF_IMAGE) == image_id
    assert calls == [
        [
            "/usr/bin/docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            REVIEWER_PDF_IMAGE,
        ]
    ]
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert probe.local_image_id(REVIEWER_PDF_IMAGE) is None


async def test_a_docker_tag_records_the_local_image_id(tmp_path, monkeypatch):
    image_id = "sha256:" + "b" * 64
    monkeypatch.setattr(probe, "local_image_id", lambda image: image_id)
    _, _, out = await _automatic(
        tmp_path,
        monkeypatch,
        ReviewerConfig(agent="codex", model="m", image=REVIEWER_PDF_IMAGE),
    )
    assert _record(out)["image_digest"] is None
    assert _record(out)["image_id"] == image_id


BARE_IMAGE_PROBE = "\n".join(
    [
        *(f"command\t{tool}\tmissing\t" for tool in probe.PROBE_COMMANDS),
        "python\t3.13.5",
        *(f"package\t{name}\t" for name in probe.PROBE_PACKAGES),
    ]
)
PDF_IMAGE_PROBE = "\n".join(
    [
        "command\tpdftotext\tfound\tpdftotext version 22.02.0",
        "command\tpdftoppm\tfound\t",
        "command\tmutool\tmissing\t",
        "python\t3.13.5",
        "package\tnumpy\t2.2.6",
        "package\tpypdf\t5.9.0",
        "package\tPyMuPDF\t2.1",
        "package\tunlisted\t9.9",
    ]
)


def test_parse_probe():
    bare = probe.parse_probe(BARE_IMAGE_PROBE)
    assert bare["python"] == "3.13.5"
    assert set(bare["commands"].values()) == {None}
    assert set(bare["packages"].values()) == {None}

    tools = probe.parse_probe(PDF_IMAGE_PROBE)
    assert tools["commands"]["pdftotext"] == "pdftotext version 22.02.0"
    assert tools["commands"]["pdftoppm"] == "unknown version"
    assert tools["commands"]["mutool"] is None
    assert tools["packages"]["PyMuPDF"] == "2.1"
    assert "unlisted" not in tools["packages"]
    assert probe.parse_probe("")["python"] is None


async def test_the_probe_script_runs_under_sh(tmp_path):
    """The script itself, run by this host's sh: every name is reported."""

    class Sandbox:
        async def exec(self, command, **kwargs):
            done = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60
            )
            return SimpleNamespace(
                return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
            )

    record: dict = {}
    path = tmp_path / "reviewer-environment.json"
    await probe.probe_reviewer_image(Sandbox(), record=record, record_path=path)
    tools = record["image_tools"]
    assert "error" not in tools
    assert set(tools["commands"]) == set(probe.PROBE_COMMANDS)
    assert set(tools["packages"]) == set(probe.PROBE_PACKAGES)
    assert re.fullmatch(r"3\.\d+\.\d+.*", tools["python"])
    assert json.loads(path.read_text()) == record


async def test_a_failed_probe_is_recorded_and_the_review_goes_on(tmp_path):
    class Sandbox:
        async def exec(self, command, **kwargs):
            raise RuntimeError("sandbox went away")

    record = {"image": REVIEWER_IMAGE}
    path = tmp_path / "reviewer-environment.json"
    await probe.probe_reviewer_image(Sandbox(), record=record, record_path=path)
    assert record["image_tools"] == {"error": "RuntimeError: sandbox went away"}
    assert json.loads(path.read_text()) == record


async def test_the_probe_is_the_last_hook_and_fills_the_record(tmp_path, monkeypatch):
    from benchflow.review.probe import probe_reviewer_image

    fake = ProbingRun(
        probe_stdout=PDF_IMAGE_PROBE, review_payload=good_weighted_review()
    )
    _, trial, out = await _automatic(
        tmp_path, monkeypatch, ReviewerConfig(agent="codex", model="m"), fake=fake
    )
    assert fake.configs[0].pre_agent_hooks[-1].func is probe_reviewer_image
    record = _record(out)
    assert record["image_tools"]["packages"]["PyMuPDF"] == "2.1"
    assert trial.reviewer_environment == record


async def test_the_scoring_record_carries_the_reviewer_environment(
    prepared,
    saved,
    monkeypatch,
):
    from benchflow.review import automatic

    monkeypatch.setattr(
        benchflow,
        "run",
        ProbingRun(
            probe_stdout=PDF_IMAGE_PROBE,
            review_payload=good_weighted_review(saved.name),
        ),
    )
    scoring = await automatic.finish_review(prepared, saved)
    assert scoring.status == "complete"
    (path,) = (saved / "scoring").glob("*.json")
    environment = json.loads(path.read_text())["reviewer_environment"]
    assert environment["image_digest"] == REVIEWER_IMAGE.split("@")[1]
    assert environment["image_tools"]["commands"]["pdftotext"] == (
        "pdftotext version 22.02.0"
    )


async def test_the_detached_report_carries_the_reviewer_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        benchflow,
        "run",
        ProbingRun(probe_stdout=BARE_IMAGE_PROBE, review_payload=good_review()),
    )
    task = make_task(tmp_path, with_rubric=True)
    rollout = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    _, report_path = await run_reviews(
        rollout,
        agent="gemini",
        model="gemini/test-model",
        out_dir=tmp_path / "out",
        tasks_root=tmp_path / "tasks",
        pdf_tools=True,
    )
    (entry,) = json.loads(report_path.read_text())["trials"]
    environment = entry["reviewer_environment"]
    assert environment["image_tools"]["python"] == "3.13.5"
    assert environment["artifact_packages"] == WITH_PDF
    assert environment["pdf_tools"] is True


def test_the_cli_flags_reach_the_reviewer(tmp_path, monkeypatch):
    from benchflow.evaluation import Evaluation

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    captured = []

    async def capture(self):
        captured.append(self._config.reviewer)
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    monkeypatch.setattr(Evaluation, "run", capture)
    ran = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--reviewer-pdf-tools",
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert captured == [ReviewerConfig(pdf_tools=True)]
