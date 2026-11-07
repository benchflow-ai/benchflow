"""Guards CONF-rollout-3 against artifact transfer loss at 82e412f1."""

import json
import tarfile

import pytest

from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout._artifacts import MANIFEST_NAME, collect_artifacts
from benchflow.rollout._separate_verifier import (
    SeparateVerifierError,
    _Payload,
    build_transfer_payload,
)
from tests.reachable_fixture import directory as reachable_directory
from tests.test_separate_verifier import LocalTransport


async def _collected(tmp_path, mounted, shape):
    workspace = tmp_path / "app"
    workspace.mkdir()
    trial = tmp_path / "trial"
    artifacts = trial / "artifacts"
    logs = artifacts if mounted else tmp_path / "logs"
    (logs / "nested").mkdir(parents=True)
    (logs / "empty").mkdir()
    (logs / "content.txt").write_text("42\n")
    if shape == "relative":
        (logs / "answer.txt").symlink_to("content.txt")
    elif shape == "absolute":
        (logs / "answer.txt").symlink_to(logs / "content.txt")
    elif shape == "upward":
        (logs / "nested" / "answer.txt").symlink_to("../content.txt")
    elif shape == "directory":
        (logs / "nested" / "content.txt").write_text("42\n")
        (logs / "answer").symlink_to("nested", target_is_directory=True)
    elif shape == "empty-directory":
        (logs / "answer").symlink_to("empty", target_is_directory=True)
    elif shape == "chain":
        (logs / "answer.txt").symlink_to("middle.txt")
        (logs / "middle.txt").symlink_to("content.txt")
    else:
        (logs / "answer.txt").symlink_to(shape)
    env = LocalTransport()
    await capture_task_evidence(env, str(workspace), trial / "evidence")
    manifest = await collect_artifacts(
        env,
        artifacts=[],
        workspace=str(workspace),
        artifacts_dir=reachable_directory(artifacts),
        manifest_path=trial / MANIFEST_NAME,
        mounted=mounted,
        logs_source=str(logs),
    )
    return trial, manifest


@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        "relative",
        "absolute",
        "upward",
        "directory",
        "empty-directory",
        "chain",
    ],
)
async def test_transfer_preserves_internal_artifact_links(tmp_path, mounted, shape):
    """CONF3 at 2e65151b replaces rewriting with explicit link refusal."""
    trial, manifest = await _collected(tmp_path, mounted, shape)
    archive = tmp_path / "payload.tar"
    build_transfer_payload(trial, archive)
    with tarfile.open(archive) as tar:
        if shape in {"absolute", "upward", "chain"}:
            name = "nested/answer.txt" if shape == "upward" else "answer.txt"
            assert "logs/artifacts/" + name not in tar.getnames()
            assert any(
                item["path"] == name and item["reason"] == "symlink_refused"
                for item in manifest["collections"][0]["exclusions"]
            )
            return
        members = tar.getmembers()
        links = [m for m in members if m.issym()]
        assert links, "a completed inventory must not silently drop its links"
        assert all(not m.linkname.startswith("/") for m in links)
        assert all(m.isfile() or m.isdir() for m in members[: members.index(links[0])])
        restored = tmp_path / "restored"
        restored.mkdir()
        tar.extractall(restored, filter="data")
    root = restored / "logs" / "artifacts"
    if shape == "directory":
        assert (root / "answer" / "content.txt").read_text() == "42\n"
    elif shape == "empty-directory":
        assert (root / "answer").is_dir()
    else:
        assert (root / "answer.txt").read_text() == "42\n"


@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize(
    "target", ["/tmp/outside-marker", "../outside-marker", "answer.txt"]
)
async def test_transfer_keeps_escape_and_cycle_exclusions(tmp_path, mounted, target):
    trial, _ = await _collected(tmp_path, mounted, target)
    archive = tmp_path / "payload.tar"
    build_transfer_payload(trial, archive)
    with tarfile.open(archive) as tar:
        assert not any(m.issym() for m in tar.getmembers())
        assert tar.extractfile("logs/artifacts/content.txt").read() == b"42\n"


@pytest.mark.parametrize("mounted", [False, True])
async def test_transfer_refuses_a_link_changed_after_collection(tmp_path, mounted):
    trial, _ = await _collected(tmp_path, mounted, "relative")
    link = trial / "artifacts" / "answer.txt"
    link.unlink()
    link.symlink_to("/tmp/outside-marker")
    with pytest.raises(SeparateVerifierError, match=r"symlink|link"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.parametrize(
    "target", ["../outside-marker", "/tmp/outside-marker", "answer.txt"]
)
async def test_transfer_does_not_trust_a_false_escape_flag(tmp_path, target):
    trial, manifest = await _collected(tmp_path, True, "relative")
    link = trial / "artifacts" / "answer.txt"
    link.unlink()
    link.symlink_to(target)
    item = next(f for f in manifest["files"] if f["kind"] == "symlink")
    item.update(target=target, escapes=False)
    (trial / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(SeparateVerifierError, match=r"symlink|link"):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.parametrize("mounted", [False, True])
async def test_transfer_preserves_safe_dangling_links(tmp_path, mounted):
    trial, _ = await _collected(tmp_path, mounted, "missing.txt")
    archive = tmp_path / "payload.tar"
    build_transfer_payload(trial, archive)
    with tarfile.open(archive) as tar:
        link = tar.getmember("logs/artifacts/answer.txt")
        assert link.issym()
        assert link.linkname == "missing.txt"


@pytest.mark.parametrize(
    "change", ["duplicate", "unknown-kind", "ancestor-link", "path-escape"]
)
async def test_transfer_refuses_inconsistent_artifact_namespaces(tmp_path, change):
    trial, manifest = await _collected(tmp_path, True, "relative")
    item = next(f for f in manifest["files"] if f["kind"] == "symlink")
    if change == "duplicate":
        manifest["files"].append(dict(item))
    elif change == "unknown-kind":
        item["kind"] = "hardlink"
    elif change == "ancestor-link":
        manifest["files"].append(
            {
                "path": "answer.txt/child",
                "kind": "file",
                "collection": 0,
            }
        )
    else:
        item["path"] = "../answer.txt"
    (trial / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(SeparateVerifierError):
        build_transfer_payload(trial, tmp_path / "payload.tar")


@pytest.mark.parametrize("link_first", [False, True])
def test_payload_refuses_cross_bundle_symlink_ancestors(tmp_path, link_first):
    """Guards CONF-rollout-3's shared namespace across every transfer bundle."""
    from pathlib import PurePosixPath

    with tarfile.open(tmp_path / "payload.tar", "w") as archive:
        payload = _Payload(archive)
        root = PurePosixPath("/logs/artifacts/alias")

        def first():
            payload.symlink(root, "elsewhere")

        def second():
            payload.directory(root / "child")

        if not link_first:
            first, second = second, first
        first()
        with pytest.raises(SeparateVerifierError):
            second()
