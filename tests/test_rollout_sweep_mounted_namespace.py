"""Guards CONF2-rollout-1 against the incomplete link fix at 941a0cc5."""

import shlex
import tarfile

import pytest

from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout._artifacts import MANIFEST_NAME, collect_artifacts
from benchflow.rollout._separate_verifier import (
    SeparateVerifierError,
    _install_script,
    build_transfer_payload,
)
from tests.reachable_fixture import directory as reachable_directory
from tests.test_separate_verifier import LocalTransport


@pytest.mark.parametrize(
    "mounted,source",
    [(True, "/logs/artifacts"), (True, "/custom/artifacts"), (False, None)],
)
@pytest.mark.parametrize("prefix", ["", "nested/"])
@pytest.mark.parametrize("spelling", ["ordinary", "slashes", "dots"])
@pytest.mark.parametrize(
    "shape",
    ["file", "directory", "dangling", "chain", "absolute-chain", "parent", "root"],
)
async def test_mounted_links_use_the_sandbox_namespace(
    tmp_path, mounted, source, shape, prefix, spelling
):
    trial = tmp_path / "trial"
    root = trial / "artifacts" if mounted else tmp_path / "sandbox-artifacts"
    source = source or str(root)
    (root / "nested" / "deep").mkdir(parents=True)
    (root / "nested" / "content.txt").write_text("42\n")
    targets = {
        "file": f"{source}/nested/content.txt",
        "directory": f"{source}/nested",
        "dangling": f"{source}/missing",
        "chain": "middle",
        "absolute-chain": f"{source}/{prefix}middle",
        "parent": f"{source}/{prefix}middle/../content.txt",
        "root": source,
    }

    def spell(text):
        if spelling == "slashes":
            return text.replace("/", "//")
        if spelling == "dots":
            return text.replace("/", "/./")
        return text

    (root / prefix / "answer").symlink_to(spell(targets[shape]))
    if shape in {"chain", "absolute-chain", "parent"}:
        target = "nested/deep" if shape == "parent" else "nested/content.txt"
        (root / prefix / "middle").symlink_to(spell(f"{source}/{target}"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    await capture_task_evidence(LocalTransport(), str(workspace), trial / "evidence")
    manifest = await collect_artifacts(
        LocalTransport(),
        artifacts=[],
        workspace=str(workspace),
        artifacts_dir=reachable_directory(trial / "artifacts"),
        manifest_path=trial / MANIFEST_NAME,
        mounted=mounted,
        logs_source=source,
    )
    # CONF3 at 2e65151b replaces rewriting with a plain-relative allowlist.
    # This matrix consists entirely of absolute links or chains through one.
    assert manifest["collections"][0]["status"] == "collected"
    assert any(
        item["path"] == prefix + "answer"
        and item["reason"] == "symlink_refused"
        and item["link_target"] == spell(targets[shape])
        for item in manifest["collections"][0]["exclusions"]
    )
    archive = tmp_path / "payload.tar"
    transfer = build_transfer_payload(trial, archive)
    restored = tmp_path / "restored"
    restored.mkdir()
    with tarfile.open(archive) as tar:
        assert f"logs/artifacts/{prefix}answer" not in tar.getnames()
    # Use the production extractor: Python's data filter normalizes link text,
    # unlike the verifier's tar, changing a/../b through a directory symlink.
    installed = await LocalTransport().exec(
        shlex.join(
            [
                "sh",
                "-c",
                _install_script(
                    str(archive), str(workspace), transfer["sha256"], str(restored)
                ),
            ]
        )
    )
    assert installed.return_code == 0, installed.stderr
    answer = restored / "logs/artifacts" / prefix / "answer"
    assert not answer.exists() and not answer.is_symlink()


@pytest.mark.parametrize(
    "shape", ["outside", "host-alias", "prefix", "parent", "cycle", "chain"]
)
async def test_mounted_namespace_excludes_escapes_without_host_resolution(
    tmp_path, shape
):
    root = tmp_path / "trial/artifacts"
    root.mkdir(parents=True)
    (root / "content").write_text("42")
    targets = {
        "outside": "/tmp/outside-marker",
        "host-alias": str(root / "content"),
        "prefix": "/logs/artifacts-sibling/content",
        "parent": "/logs/artifacts/middle/../../outside-marker",
        "cycle": "/logs/artifacts/answer",
        "chain": "middle",
    }
    (root / "answer").symlink_to(targets[shape])
    if shape in {"parent", "chain"}:
        (root / "middle").symlink_to(
            "/logs/artifacts" if shape == "parent" else "/tmp/outside-marker"
        )
    manifest = await collect_artifacts(
        LocalTransport(),
        artifacts=[],
        workspace=str(tmp_path),
        artifacts_dir=reachable_directory(root),
        manifest_path=root.parent / MANIFEST_NAME,
        mounted=True,
    )
    answer = next(f for f in manifest["files"] if f["path"] == "answer")
    assert answer["escapes"] is True
    assert manifest["collections"][0]["status"] == "collected"


@pytest.mark.parametrize(
    "target", ["/logs/artifacts/middle/../outside", "middle/../outside"]
)
def test_transfer_rechecks_expansion_before_parent_components(tmp_path, target):
    """Guards 014cf443: lexical '..' removal cannot precede link expansion."""
    from benchflow.rollout._separate_verifier import _logs_entries

    (tmp_path / "middle").symlink_to("/logs/artifacts")
    (tmp_path / "answer").symlink_to(target)
    inventory = {
        "collections": [{"source": "/logs/artifacts"}],
        "files": [
            {
                "path": name,
                "kind": "symlink",
                "target": text,
                "collection": 0,
                "escapes": False,
            }
            for name, text in [("middle", "/logs/artifacts"), ("answer", target)]
        ],
    }
    with pytest.raises(SeparateVerifierError, match="symlink"):
        _logs_entries(inventory, tmp_path)


@pytest.mark.parametrize(
    "shape", ["empty", "file", "directory", "relative-link", "absolute-link"]
)
def test_legacy_inventory_requires_source_only_for_absolute_links(tmp_path, shape):
    """Guards 352cf14b: a source namespace is unnecessary without absolute links."""
    from benchflow.rollout._separate_verifier import _logs_entries

    files = []
    if shape == "file":
        (tmp_path / "answer").write_text("42")
        files.append({"path": "answer", "kind": "file", "collection": 0})
    elif shape == "directory":
        (tmp_path / "answer").mkdir()
        files.append({"path": "answer", "kind": "directory", "collection": 0})
    elif shape.endswith("link"):
        target = "/logs/artifacts/missing" if shape == "absolute-link" else "missing"
        (tmp_path / "answer").symlink_to(target)
        files.append(
            {"path": "answer", "kind": "symlink", "target": target, "collection": 0}
        )
    inventory = {"collections": [{"kind": "logs"}], "files": files}
    if shape == "absolute-link":
        with pytest.raises(SeparateVerifierError, match="plain relative path"):
            _logs_entries(inventory, tmp_path)
    else:
        assert _logs_entries(inventory, tmp_path) == files
