"""Shared reviewer execution for terminal scoring and detached audits."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import benchflow
from benchflow.review.config import load_rubric
from benchflow.review.evidence import (
    _UNPACK_SCRIPT,
    ArtifactEvidence,
    EvidenceEntry,
    EvidenceError,
    EvidenceManifest,
    install_review_evidence,
    refuse_linked_bundle,
    validate_workspace,
)
from benchflow.review.options import ReviewerConfig
from benchflow.review.outcome import ScoringResult
from benchflow.review.runner import (
    _lock_review_evidence,
    discover_rollouts,
    run_review,
    run_reviews,
)
from benchflow.review.wrapper import assemble_review_task, remove_review_task
from tests.test_review_runtime import (
    WEIGHTED_RUBRIC,
    FakeRun,
    good_weighted_review,
    make_rollout,
    make_task,
)


def make_bundle(root: Path) -> Path:
    bundle = root / "bundle"
    workspace = bundle / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "paper.txt").write_bytes(b"scientific output")
    (workspace / "link.txt").symlink_to("paper.txt")
    (workspace / "empty").mkdir()
    manifest = EvidenceManifest(
        workspace="/research/project",
        archive_sha256="0" * 64,
        entries=(
            EvidenceEntry(
                path="paper.txt",
                original_path="/research/project/paper.txt",
                kind="file",
                size=17,
                sha256=hashlib.sha256(b"scientific output").hexdigest(),
            ),
            EvidenceEntry(
                path="link.txt",
                original_path="/research/project/link.txt",
                kind="symlink",
                size=0,
                link_target="paper.txt",
            ),
            EvidenceEntry(
                path="empty",
                original_path="/research/project/empty",
                kind="directory",
                size=0,
            ),
        ),
    )
    (bundle / "manifest.json").write_text(manifest.model_dump_json())
    return bundle


@pytest.mark.asyncio
async def test_review_before_final_result_uses_explicit_test_gate(
    tmp_path, monkeypatch
):
    """Guards terminal integration after PR #1126 without a fake final result."""
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", reward=0.0)
    (source / "result.json").rename(source / "solver.json")
    fake = FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(benchflow, "run", fake)
    rubric_path = task / "verifier/rubric.json"
    trial = await run_review(
        source,
        task,
        load_rubric(rubric_path),
        rubric_path,
        ReviewerConfig(
            agent="codex", model="azure/gpt5.6terra", reasoning_effort="xhigh"
        ),
        tmp_path / "review-output",
        deterministic_pass=True,
        workspace_bundle=make_bundle(tmp_path),
    )
    assert trial.review_valid and trial.scoring is not None
    assert trial.scoring.gated_quality == pytest.approx(0.8)
    assert not (source / "result.json").exists()
    assert fake.configs[0].purpose == "reviewer"
    assert fake.configs[0].parent_rollout == "rollout-a"
    assert fake.configs[0].reasoning_effort == "xhigh"
    assert fake.configs[0].pre_agent_hooks[0].func is install_review_evidence
    assert fake.configs[0].pre_agent_hooks[1] is _lock_review_evidence
    assert "/evidence/workspace" in fake.task_docs[0]
    assert "solver.json" in fake.task_docs[0]
    assert not fake.configs[0].task_path.exists()
    assert Path(trial.reviewer_rollout, "result.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        {"error": "reviewer timed out"},
        {"verifier_error": "could not retrieve output"},
        {"partial_trajectory": True},
        {"export_error": "could not export reviewer workspace"},
    ],
)
async def test_interrupted_reviewer_cannot_score_valid_json(
    tmp_path, monkeypatch, failure
):
    """Guards terminal integration after PR #1126 against premature JSON."""
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    fake = FakeRun(review_payload=good_weighted_review())

    async def interrupted(config):
        result = await fake(config)
        leaf = Path(config.jobs_dir) / "job/wrapper__0000"
        artifact = leaf / "result.json"
        data = json.loads(artifact.read_text())
        data.update(failure)
        artifact.write_text(json.dumps(data))
        # A reviewer-generated config file inside the captured workspace is
        # evidence, not a second runtime leaf.
        (leaf / "evidence/workspace").mkdir(parents=True)
        (leaf / "evidence/workspace/config.json").write_text("{}")
        return result

    monkeypatch.setattr(benchflow, "run", interrupted)
    report, _ = await run_reviews(
        source, agent="gemini", tasks_root=tmp_path / "tasks", out_dir=tmp_path / "out"
    )
    trial = report.trials[0]
    assert not trial.review_valid
    assert trial.scoring is None
    assert trial.checks == good_weighted_review()["checks"]
    assert trial.error and trial.reviewer_rollout


def test_workspace_upload_preserves_links_and_redacts_provider_capture(tmp_path):
    """Guards terminal integration after PR #1126 on Daytona upload semantics."""
    source = make_rollout(tmp_path / "jobs", "rollout-a")
    (source / "trajectory/llm_trajectory.jsonl").write_text(
        json.dumps(
            {
                "headers": {"authorization": "Bearer sensitive-credential-value"},
                "response": {"text": "I computed the energy.", "usage": {"tokens": 42}},
            }
        )
        + "\n"
    )
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    bundle = make_bundle(tmp_path)
    external = make_bundle(tmp_path / "external")
    (bundle / "artifacts").mkdir()
    external.rename(bundle / "artifacts/0000")
    manifest = EvidenceManifest.model_validate_json(
        (bundle / "manifest.json").read_text()
    )
    manifest = manifest.model_copy(
        update={
            "artifacts": (
                ArtifactEvidence(
                    source="/outputs", status="captured", bundle_path="artifacts/0000"
                ),
                ArtifactEvidence(source="/absent/report.pdf", status="missing"),
            )
        }
    )
    (bundle / "manifest.json").write_text(manifest.model_dump_json())
    wrapper, uploads = assemble_review_task(
        source,
        task,
        load_rubric(task / "verifier/rubric.json"),
        tmp_path / "wrapper",
        workspace_bundle=bundle,
    )
    try:
        assert "/evidence/artifacts/0000.tar" in uploads.values()
        assert "/evidence/artifacts/0000-manifest.json" in uploads.values()
        assert len(uploads) == 6  # trial, task, and two archive/manifest pairs
        archive = next(
            Path(p) for p, dest in uploads.items() if dest == "/evidence/workspace.tar"
        )
        with tarfile.open(archive) as stream:
            members = {entry.name: entry for entry in stream.getmembers()}
            assert members["link.txt"].issym()
            assert members["link.txt"].linkname == "paper.txt"
            assert members["empty"].isdir()
            assert (
                stream.extractfile(members["paper.txt"]).read() == b"scientific output"
            )
        capture = (
            wrapper / "evidence/trial/trajectory/llm_trajectory.jsonl"
        ).read_text()
        assert "sensitive-credential-value" not in capture
        assert json.loads(capture)["response"]["usage"]["tokens"] == 42
        assert "I computed the energy." in capture
    finally:
        remove_review_task(wrapper)
    assert not wrapper.exists()


@pytest.mark.asyncio
async def test_detached_review_uses_preserved_test_gate_for_partial_quality(
    tmp_path, monkeypatch
):
    """Guards the partial-quality re-review trap after PR #1126."""
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    scoring = ScoringResult(
        status="complete",
        passed=True,
        tests_pass=True,
        all_blockers_pass=True,
        verifier_reward=1.0,
        rubric_reward=0.8,
        reviewer_run="reviews/attempt-001",
    )
    result = json.loads((source / "result.json").read_text())
    result.update(scoring=scoring.to_dict(), rewards=scoring.numeric_rewards())
    (source / "result.json").write_text(json.dumps(result))
    assert discover_rollouts(source, filter_passing=True) == [source]
    monkeypatch.setattr(
        benchflow, "run", FakeRun(review_payload=good_weighted_review())
    )
    report, _ = await run_reviews(
        source, agent="gemini", tasks_root=tmp_path / "tasks", out_dir=tmp_path / "out"
    )
    assert report.trials[0].scoring.gated_quality == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_default_artifact_tools_use_failing_setup_boundary(tmp_path):
    """Guards scientific reviewer bootstrap after PR #1126."""
    from types import SimpleNamespace

    from benchflow.rollout import _run_environment_setup_commands
    from benchflow.task import Task

    source = make_rollout(tmp_path / "jobs", "rollout-a")
    task_dir = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    wrapper, _ = assemble_review_task(
        source,
        task_dir,
        load_rubric(task_dir / "verifier/rubric.json"),
        tmp_path / "wrapper",
    )
    try:
        task = Task(wrapper)
        command = task.config.sandbox.setup_commands[0]
        assert "numpy==2.2.6 pypdf==5.9.0" in command.command
        assert command.user == "root"
        assert command.cwd == "/tmp"
        assert not task.config.sandbox.allow_internet

        class FailedBootstrap:
            async def exec(self, command, **kwargs):
                assert kwargs["user"] == "root"
                return SimpleNamespace(
                    return_code=1, stdout="", stderr="pip download failed"
                )

        with pytest.raises(RuntimeError, match="environment setup command 1 failed"):
            await _run_environment_setup_commands(FailedBootstrap(), task)
    finally:
        remove_review_task(wrapper)


def test_custom_reviewer_image_owns_its_artifact_tools(tmp_path):
    """Guards custom image independence after PR #1126."""
    from benchflow.task import Task

    source = make_rollout(tmp_path / "jobs", "rollout-a")
    task_dir = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    wrapper, _ = assemble_review_task(
        source,
        task_dir,
        load_rubric(task_dir / "verifier/rubric.json"),
        tmp_path / "wrapper",
        image="custom-reviewer:v1",
    )
    try:
        assert Task(wrapper).config.sandbox.setup_commands == []
    finally:
        remove_review_task(wrapper)


def _script_reviewer(sandbox: Path, listings: list[list[str]], fake: FakeRun):
    """A reviewer that lists /evidence as its sandbox holds it: each upload
    at its path, then the workspace archives unpacked by the pre-agent
    hook's own script (install_review_evidence)."""

    async def run(config):
        for host, dest in config.uploads.items():
            target = sandbox / dest.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(host).is_dir():
                shutil.copytree(host, target, symlinks=True)
            else:
                shutil.copyfile(host, target)
        evidence = sandbox / "evidence"
        if any(
            getattr(hook, "func", None) is install_review_evidence
            for hook in config.pre_agent_hooks
        ):
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _UNPACK_SCRIPT,
                    str(evidence / "workspace.tar"),
                    str(evidence / "workspace"),
                    str(evidence / "workspace-manifest.json"),
                ],
                check=True,
            )
        listings.append(
            sorted("/" + p.relative_to(sandbox).as_posix() for p in evidence.rglob("*"))
        )
        return await fake(config)

    return run


@pytest.mark.asyncio
async def test_bench_review_gives_the_reviewer_the_captured_workspace(
    tmp_path, monkeypatch
):
    """Guards #1231: ``bench review`` (run_reviews) hands the reviewer the
    rollout's captured workspace at /evidence/workspace, as the in-run review
    (finish_review) does. It uploaded only /evidence/trial and /evidence/task."""
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    make_bundle(tmp_path).rename(source / "evidence")
    listings: list[list[str]] = []
    fake = FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(
        benchflow, "run", _script_reviewer(tmp_path / "sandbox", listings, fake)
    )
    report, _ = await run_reviews(
        source, agent="gemini", tasks_root=tmp_path / "tasks", out_dir=tmp_path / "out"
    )
    (listing,) = listings
    for path in (
        "/evidence/workspace/paper.txt",
        "/evidence/workspace/link.txt",
        "/evidence/workspace/empty",
        "/evidence/workspace-manifest.json",
        "/evidence/trial/result.json",
        "/evidence/task/task.md",
    ):
        assert path in listing, path
    assert not any(path.startswith("/evidence/trial/evidence") for path in listing)
    assert "/evidence/workspace" in fake.task_docs[0]
    trial = report.trials[0]
    assert trial.review_valid and trial.error is None
    assert not any("workspace" in note for note in trial.notes)


@pytest.mark.asyncio
async def test_bench_review_records_a_rollout_without_a_captured_workspace(
    tmp_path, monkeypatch
):
    """Guards #1231: a rollout that captured no workspace is still reviewed,
    from its records, and its review says why the workspace is missing."""
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    listings: list[list[str]] = []
    fake = FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(
        benchflow, "run", _script_reviewer(tmp_path / "sandbox", listings, fake)
    )
    report, _ = await run_reviews(
        source, agent="gemini", tasks_root=tmp_path / "tasks", out_dir=tmp_path / "out"
    )
    assert not any(path.startswith("/evidence/workspace") for path in listings[0])
    trial = report.trials[0]
    assert trial.review_valid
    assert any("workspace evidence skipped" in note for note in trial.notes)


@pytest.mark.asyncio
async def test_bench_review_skips_a_linked_evidence_folder(tmp_path, monkeypatch):
    """An ``evidence`` link to another valid capture is not the rollout's own:
    the reviewer gets no workspace, and the review says why."""
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    (source / "evidence").symlink_to(make_bundle(tmp_path), target_is_directory=True)
    listings: list[list[str]] = []
    fake = FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(
        benchflow, "run", _script_reviewer(tmp_path / "sandbox", listings, fake)
    )
    report, _ = await run_reviews(
        source, agent="gemini", tasks_root=tmp_path / "tasks", out_dir=tmp_path / "out"
    )
    assert not any(path.startswith("/evidence/workspace") for path in listings[0])
    trial = report.trials[0]
    assert trial.review_valid
    assert any(
        "workspace evidence skipped" in note and "symlink" in note
        for note in trial.notes
    )


def test_validate_workspace_refuses_a_linked_workspace_root(tmp_path):
    """A real ``evidence/`` whose ``workspace`` is a link to a matching tree
    is refused before the link is resolved."""
    bundle = make_bundle(tmp_path)
    manifest = EvidenceManifest.model_validate_json(
        (bundle / "manifest.json").read_text()
    )
    (bundle / "workspace").rename(tmp_path / "elsewhere")
    (bundle / "workspace").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    validate_workspace(tmp_path / "elsewhere", manifest)
    with pytest.raises(EvidenceError, match="symlink"):
        validate_workspace(bundle / "workspace", manifest)


def _capture(root: Path, name: str, content: bytes, *, artifact: bool) -> Path:
    """A captured bundle at *root* whose workspace holds one file. With
    *artifact* its manifest also lists a captured artifact at
    ``artifacts/0000``, which the caller puts in place."""
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / name).write_bytes(content)
    manifest = EvidenceManifest(
        workspace="/app",
        archive_sha256="0" * 64,
        entries=(
            EvidenceEntry(
                path=name,
                original_path=f"/app/{name}",
                kind="file",
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            ),
        ),
        artifacts=(
            (
                ArtifactEvidence(
                    source="/outputs", status="captured", bundle_path="artifacts/0000"
                ),
            )
            if artifact
            else ()
        ),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json())
    return root


def _archive_reviewer(uploaded: set[str], fake: FakeRun):
    """A reviewer that records the file names in every archive it is given."""

    async def run(config):
        for host in config.uploads:
            if host.endswith(".tar"):
                with tarfile.open(host) as archive:
                    uploaded.update(Path(m.name).name for m in archive.getmembers())
        return await fake(config)

    return run


@pytest.mark.asyncio
@pytest.mark.parametrize("linked", ["artifacts/0000", "artifacts"])
async def test_bench_review_refuses_a_link_below_the_evidence_folder(
    tmp_path, monkeypatch, linked
):
    """Guards PR #1239 (xdotli's review, 11-19) after 5df9018b: a rollout's
    own ``evidence/`` whose ``artifacts/0000``, or ``artifacts`` itself, links
    to another valid capture would hand the reviewer that capture as
    ``/evidence/artifacts/0000.tar``. That trial's review fails instead,
    naming the link, and nothing is uploaded."""
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    own = _capture(tmp_path / "own", "paper.txt", b"own work", artifact=True)
    own.rename(source / "evidence")
    other = _capture(
        tmp_path / "other", "secret.txt", b"another rollout's workspace", artifact=False
    )
    if linked == "artifacts":
        (tmp_path / "elsewhere").mkdir()
        other.rename(tmp_path / "elsewhere" / "0000")
        (source / "evidence" / "artifacts").symlink_to(
            tmp_path / "elsewhere", target_is_directory=True
        )
    else:
        (source / "evidence" / "artifacts").mkdir()
        (source / "evidence" / "artifacts" / "0000").symlink_to(
            other, target_is_directory=True
        )
    uploaded: set[str] = set()
    fake = FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(benchflow, "run", _archive_reviewer(uploaded, fake))
    report, _ = await run_reviews(
        source, agent="gemini", tasks_root=tmp_path / "tasks", out_dir=tmp_path / "out"
    )
    assert "secret.txt" not in uploaded
    trial = report.trials[0]
    assert not trial.review_valid
    assert trial.error == f"Evidence bundle path is a symlink: {linked}"


@pytest.mark.parametrize(
    "name",
    [
        "manifest.json",
        "workspace",
        "artifacts",
        "artifacts/0000",
        "artifacts/0000/manifest.json",
        "artifacts/0000/workspace",
    ],
)
def test_refuse_linked_bundle_checks_every_name_a_bundle_is_read_through(
    tmp_path, name
):
    """Guards PR #1239 (xdotli's review, 11-19): each name below a bundle
    that its upload reads through is refused when it is a link."""
    bundle = _capture(tmp_path / "own", "paper.txt", b"own work", artifact=True)
    _capture(bundle / "artifacts" / "0000", "out.txt", b"output", artifact=False)
    manifest = EvidenceManifest.model_validate_json(
        (bundle / "manifest.json").read_text()
    )
    refuse_linked_bundle(bundle, manifest)
    target = bundle / name
    target.rename(tmp_path / "moved")
    target.symlink_to(
        tmp_path / "moved", target_is_directory=(tmp_path / "moved").is_dir()
    )
    with pytest.raises(EvidenceError, match=f"symlink: {name}$"):
        refuse_linked_bundle(bundle, manifest)
