"""Guards the review evidence isolation boundary introduced by PR #942.

Exercise the real capture/admission scripts with a local subprocess transport;
these tests do not need credentials or pretend to test provider provisioning.
"""

from __future__ import annotations

import asyncio
import io
import os
import shlex
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

from benchflow.agents.credentials import credential_evidence_overrides
from benchflow.review.evidence import (
    EvidenceError,
    EvidenceManifest,
    _extract_archive,
    capture_task_evidence,
    capture_workspace,
    install_review_evidence,
    install_uploaded_workspace,
    prepare_workspace_upload,
    validate_uploaded_workspace,
    validate_workspace,
)
from benchflow.sandbox.protocol import ExecResult
from benchflow.task.config import ArtifactConfig


class LocalTransport:
    """Run the same stdlib scripts used by remote sandbox transports."""

    def __init__(self, *, corrupt_download: bool = False):
        self.corrupt_download = corrupt_download
        self.downloaded: list[str] = []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        assert process.returncode is not None
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def download_file(self, src: str, dst: Path) -> None:
        self.downloaded.append(src)
        shutil.copyfile(src, dst)
        if self.corrupt_download:
            with dst.open("ab") as output:
                output.write(b"corrupted transport")


@pytest.mark.asyncio
async def test_complete_workspace_roundtrip_and_admission(tmp_path):
    source = tmp_path / "actual-working-directory"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "empty").mkdir()
    payload = source / "nested" / "résult\n data.bin"
    payload.write_bytes(bytes(range(256)))
    (source / "executable").write_text("#!/bin/sh\nexit 0\n")
    (source / "executable").chmod(0o755)
    (source / "relative-link").symlink_to("nested/résult\n data.bin")
    (source / "absolute-link").symlink_to(payload)
    (source / "directory-link").symlink_to("nested", target_is_directory=True)
    os.link(payload, source / "hardlink")
    destination = tmp_path / "evidence"
    transport = LocalTransport()

    manifest = await capture_workspace(transport, str(source), destination)

    assert manifest.workspace == str(source.resolve())
    assert manifest.capture_status == "complete"
    assert (destination / "workspace" / "absolute-link").read_bytes() == bytes(
        range(256)
    )
    assert not os.readlink(destination / "workspace" / "absolute-link").startswith("/")
    assert (destination / "workspace" / "executable").stat().st_mode & 0o111
    assert (destination / "workspace" / "hardlink").read_bytes() == bytes(range(256))
    assert (destination / "workspace" / "nested" / "empty").is_dir()
    assert len(manifest.entries) == 8
    assert (
        EvidenceManifest.model_validate_json(
            (destination / "manifest.json").read_text()
        )
        == manifest
    )
    assert all(not Path(path).exists() for path in transport.downloaded)
    validate_workspace(destination / "workspace", manifest)
    await validate_uploaded_workspace(
        transport,
        str(destination / "workspace"),
        str(destination / "manifest.json"),
    )


@pytest.mark.asyncio
async def test_archive_hash_rejects_corrupt_download_atomically(tmp_path):
    """Guards PR #942's evidence boundary against corrupted transfers."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer.txt").write_text("answer")
    destination = tmp_path / "evidence"
    transport = LocalTransport(corrupt_download=True)

    with pytest.raises(EvidenceError, match="digest changed"):
        await capture_workspace(transport, str(source), destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".evidence-*"))
    assert all(not Path(path).exists() for path in transport.downloaded)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [{"max_bytes": 1}, {"max_entries": 1}])
async def test_limits_fail_instead_of_publishing_partial_evidence(tmp_path, limit):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_text("a full result")
    (source / "b").write_text("another result")

    with pytest.raises(EvidenceError, match="capture limits"):
        await capture_workspace(
            LocalTransport(), str(source), tmp_path / "evidence", **limit
        )

    assert not (tmp_path / "evidence").exists()


@pytest.mark.asyncio
async def test_external_symlink_never_imports_outside_files(tmp_path):
    """Guards PR #942's rule that evidence must not dereference host secrets."""
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "credential"
    secret.write_text("do not export")
    (source / "credential-link").symlink_to(secret)

    with pytest.raises(EvidenceError, match="symlink escapes"):
        await capture_workspace(LocalTransport(), str(source), tmp_path / "evidence")

    assert not (tmp_path / "evidence").exists()


@pytest.mark.asyncio
async def test_fifo_is_rejected_not_silently_omitted(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "results-pipe")

    with pytest.raises(EvidenceError, match="unsupported workspace entry"):
        await capture_workspace(LocalTransport(), str(source), tmp_path / "evidence")


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["bytes", "addition", "deletion", "symlink"])
async def test_host_and_reviewer_admission_reject_changed_evidence(tmp_path, mutation):
    """Guards PR #942's evidence boundary against post-capture substitution."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_text("correct")
    transport = LocalTransport()
    destination = tmp_path / "evidence"
    manifest = await capture_workspace(transport, str(source), destination)
    tree = destination / "workspace"
    if mutation == "bytes":
        (tree / "answer").write_text("changed")
    elif mutation == "addition":
        (tree / "unmanifested").write_text("added")
    elif mutation == "deletion":
        (tree / "answer").unlink()
    else:
        (tree / "answer").unlink()
        (tree / "answer").symlink_to(source / "answer")

    with pytest.raises(EvidenceError):
        validate_workspace(tree, manifest)
    with pytest.raises(EvidenceError, match="Uploaded workspace validation failed"):
        await validate_uploaded_workspace(
            transport, str(tree), str(destination / "manifest.json")
        )


@pytest.mark.parametrize(
    "members",
    [
        [("../escape", "file", "")],
        [("/absolute", "file", "")],
        [("duplicate", "file", ""), ("duplicate", "file", "")],
        [("link", "symlink", "../../outside")],
        [("link", "symlink", "/etc/passwd")],
        [("link", "symlink", "inside"), ("link/child", "file", "")],
        [("hardlink", "hardlink", "missing")],
    ],
)
def test_untrusted_archives_cannot_escape_or_alias_destination(tmp_path, members):
    """Guards PR #942's evidence isolation against tar path/link attacks."""
    archive_path = tmp_path / "attack.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, kind, target in members:
            member = tarfile.TarInfo(name)
            if kind == "file":
                member.size = 4
                archive.addfile(member, io.BytesIO(b"data"))
            else:
                member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                member.linkname = target
                archive.addfile(member)
    destination = tmp_path / "workspace"
    destination.mkdir()

    with pytest.raises(EvidenceError):
        _extract_archive(
            archive_path, destination, "/app", max_bytes=1024, max_entries=100
        )

    assert not (tmp_path / "escape").exists()


@pytest.mark.asyncio
async def test_existing_bundle_is_never_overwritten(tmp_path):
    destination = tmp_path / "evidence"
    destination.mkdir()
    (destination / "sentinel").write_text("previous evidence")

    with pytest.raises(EvidenceError, match="already exists"):
        await capture_workspace(LocalTransport(), "/unused", destination)

    assert (destination / "sentinel").read_text() == "previous evidence"


@pytest.mark.asyncio
async def test_packaged_upload_preserves_links_empty_directories_and_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "empty").mkdir()
    (source / "answer.txt").write_text("solver output")
    (source / "root-link").symlink_to(".", target_is_directory=True)
    (source / "answer-link").symlink_to(source / "answer.txt")
    transport = LocalTransport()
    bundle = tmp_path / "evidence"
    manifest = await capture_workspace(transport, str(source), bundle)
    archive = tmp_path / "upload.tar"
    prepare_workspace_upload(bundle, archive)
    receiving = tmp_path / "reviewer" / "workspace"

    await install_uploaded_workspace(
        transport, str(archive), str(receiving), str(bundle / "manifest.json")
    )

    validate_workspace(receiving, manifest)
    assert (receiving / "root-link").is_symlink()
    assert (receiving / "answer-link").read_text() == "solver output"
    assert (receiving / "empty").is_dir()
    with pytest.raises(EvidenceError, match="already exists"):
        await install_uploaded_workspace(
            transport, str(archive), str(receiving), str(bundle / "manifest.json")
        )


@pytest.mark.asyncio
async def test_receiver_rejects_unmanifested_archive_before_extracting(tmp_path):
    """Guards PR #942's transfer boundary against archive substitution."""
    source = tmp_path / "source"
    source.mkdir()
    transport = LocalTransport()
    bundle = tmp_path / "evidence"
    await capture_workspace(transport, str(source), bundle)
    archive_path = tmp_path / "upload.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 4
        archive.addfile(member, io.BytesIO(b"data"))
    receiving = tmp_path / "reviewer" / "workspace"

    with pytest.raises(EvidenceError, match="unsafe or unmanifested"):
        await install_uploaded_workspace(
            transport, str(archive_path), str(receiving), str(bundle / "manifest.json")
        )

    assert not receiving.exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.asyncio
async def test_cleanup_connection_failure_preserves_committed_evidence(
    tmp_path, caplog
):
    class CleanupFailureTransport(LocalTransport):
        async def exec(self, cmd: str, **kwargs) -> ExecResult:
            result = await super().exec(cmd, **kwargs)
            if shlex.split(cmd)[0] == "rm":
                raise OSError("cleanup connection disconnected")
            return result

    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_text("complete evidence")
    bundle = tmp_path / "evidence"

    manifest = await capture_workspace(CleanupFailureTransport(), str(source), bundle)

    validate_workspace(bundle / "workspace", manifest)
    assert "Could not remove temporary evidence archive" in caplog.text


@pytest.mark.asyncio
async def test_credential_exclusions_are_explicit_without_losing_agent_outputs(
    tmp_path,
):
    source = tmp_path / "agent-home"
    (source / ".codex").mkdir(parents=True)
    (source / ".codex" / "auth.json").write_text("credential sentinel")
    (source / ".codex" / "session.jsonl").write_text("agent trajectory")
    (source / ".ssh").mkdir()
    (source / ".ssh" / "id_ed25519").write_text("private key sentinel")
    (source / "injected-auth.json").write_text("custom provider auth")
    (source / "answer.txt").write_text("solver output")
    bundle = tmp_path / "evidence"

    manifest = await capture_workspace(
        LocalTransport(),
        str(source),
        bundle,
        excluded_paths=[str(source / "injected-auth.json")],
    )

    assert (bundle / "workspace" / "answer.txt").read_text() == "solver output"
    assert (
        bundle / "workspace" / ".codex" / "session.jsonl"
    ).read_text() == "agent trajectory"
    assert not (bundle / "workspace" / ".codex" / "auth.json").exists()
    assert not (bundle / "workspace" / ".ssh").exists()
    assert {entry.original_path for entry in manifest.exclusions} == {
        str(source / ".codex" / "auth.json"),
        str(source / ".ssh"),
        str(source / "injected-auth.json"),
    }
    assert all(entry.reason == "credential" for entry in manifest.exclusions)
    validate_workspace(bundle / "workspace", manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["/", "/etc", "/proc"])
async def test_system_root_capture_is_refused(tmp_path, source):
    with pytest.raises(EvidenceError, match="system directory"):
        await capture_workspace(LocalTransport(), source, tmp_path / "evidence")


@pytest.mark.asyncio
async def test_declared_artifacts_capture_and_review_installation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer.txt").write_text("workspace output")
    external_file = tmp_path / "separate-result.json"
    external_file.write_text('{"result": 42}')
    external_directory = tmp_path / "figures"
    external_directory.mkdir()
    (external_directory / "plot.svg").write_text("<svg/>")
    (external_directory / "scratch.tmp").write_text("not an artifact")
    (external_directory / "latest.svg").symlink_to("plot.svg")
    transport = LocalTransport()
    bundle = tmp_path / "evidence"

    manifest = await capture_task_evidence(
        transport,
        str(source),
        bundle,
        artifacts=[
            "answer.txt",
            str(tmp_path / "missing-output"),
            ArtifactConfig(
                source=str(external_file), destination="../../untrusted-destination"
            ),
            ArtifactConfig(source=str(external_directory), exclude=["*.tmp"]),
        ],
    )

    assert [artifact.status for artifact in manifest.artifacts] == [
        "in_workspace",
        "missing",
        "captured",
        "captured",
    ]
    assert [artifact.bundle_path for artifact in manifest.artifacts] == [
        None,
        None,
        "artifacts/0002",
        "artifacts/0003",
    ]
    assert (
        bundle / "artifacts/0002/workspace/separate-result.json"
    ).read_text() == '{"result": 42}'
    assert not (bundle / "artifacts/0003/workspace/scratch.tmp").exists()
    artifact_manifest = EvidenceManifest.model_validate_json(
        (bundle / "artifacts/0003/manifest.json").read_text()
    )
    assert artifact_manifest.exclusions[0].reason == "task_exclude"
    receiving = tmp_path / "reviewer"
    receiving.mkdir()
    prepare_workspace_upload(bundle, receiving / "workspace.tar")
    shutil.copyfile(bundle / "manifest.json", receiving / "workspace-manifest.json")
    for artifact in manifest.artifacts:
        if artifact.bundle_path is not None:
            upload = receiving / artifact.bundle_path
            prepare_workspace_upload(
                bundle / artifact.bundle_path, upload.with_suffix(".tar")
            )
            shutil.copyfile(
                bundle / artifact.bundle_path / "manifest.json",
                upload.with_name(upload.name + "-manifest.json"),
            )

    await install_review_evidence(transport, manifest, evidence_root=str(receiving))

    assert (receiving / "workspace/answer.txt").read_text() == "workspace output"
    assert (
        receiving / "artifacts/0002/separate-result.json"
    ).read_text() == '{"result": 42}'
    assert (receiving / "artifacts/0003/latest.svg").read_text() == "<svg/>"


@pytest.mark.asyncio
async def test_external_artifact_failure_does_not_publish_partial_task_evidence(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_text("fine")
    unsupported = tmp_path / "unsupported-artifact"
    os.mkfifo(unsupported)
    bundle = tmp_path / "evidence"

    with pytest.raises(EvidenceError, match="unsupported workspace entry"):
        await capture_task_evidence(
            LocalTransport(), str(source), bundle, artifacts=[str(unsupported)]
        )

    assert not bundle.exists()
    assert not list(tmp_path.glob(".task-evidence-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "carrier",
    [
        ".openhands/agent_settings.json",
        ".pi/agent/models.json",
        ".pi/agent/auth.json",
        ".openclaw/openclaw.json",
        ".openclaw/agents/main/agent/auth-profiles.json",
        ".config/opencode/opencode.json",
        ".config/mimocode/mimocode.json",
        "mimocode.json",
    ],
)
@pytest.mark.parametrize("nested_home", [False, True])
async def test_launcher_credential_carriers_never_enter_evidence(
    tmp_path, carrier, nested_home
):
    """Guards PR #942's evidence boundary for each launcher-written credential."""
    source = tmp_path / "workspace"
    home = source / "home/agent" if nested_home else source
    credential = home / carrier
    credential.parent.mkdir(parents=True)
    credential.write_text('{"api_key": "NEVER_EXPORT_CREDENTIAL"}')
    (home / "answer.txt").write_text("keep agent output")
    bundle = tmp_path / "evidence"

    manifest = await capture_workspace(LocalTransport(), str(source), bundle)

    relative = credential.relative_to(source)
    assert not (bundle / "workspace" / relative).exists()
    assert (
        bundle / "workspace" / (home / "answer.txt").relative_to(source)
    ).read_text() == "keep agent output"
    assert [(entry.original_path, entry.reason) for entry in manifest.exclusions] == [
        (str(credential), "credential")
    ]
    validate_workspace(bundle / "workspace", manifest)


def test_custom_credential_file_overrides_are_paths_not_inline_json():
    """Guards PR #942's evidence boundary for supported custom config paths."""
    exclusions = credential_evidence_overrides(
        {
            "MIMOCODE_CONFIG": "./private/../mimo-secrets.json",
            "OPENCODE_CONFIG": "~/opencode-secrets.json",
            "OPENCLAW_CONFIG_PATH": "/app/openclaw-secrets.json",
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/gcp.json",
            "AWS_SHARED_CREDENTIALS_FILE": "/app/aws-credentials",
            "CODEX_CONFIG": '{"model_providers":{"azure":{"env_key":"OPENAI_API_KEY"}}}',
        },
        workspace="/app",
        cred_home="/home/agent",
    )

    assert exclusions == (
        "/app/aws-credentials",
        "/app/gcp.json",
        "/app/mimo-secrets.json",
        "/app/openclaw-secrets.json",
        "/home/agent/opencode-secrets.json",
    )
