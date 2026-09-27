"""GH1136: transport of task-declared submission files and receipt admission."""

import json
import shlex
import shutil
from types import SimpleNamespace

import pytest

from benchflow.rollout import _recovery_submission as transport
from benchflow.sandbox import _recovery_submission as files
from benchflow.task.config import VerifierConfig

PATHS = ["/root/result.csv", "/root/summary.json", "/root/report.md"]


class LocalRuntime:
    def __init__(self, root, remote):
        self.root, self.remote = root, remote
        self.remote.mkdir()
        self.commands = []

    def location(self, remote):
        return self.remote / remote.rsplit("/", 1)[-1]

    async def exec(self, command, **kwargs):
        args = shlex.split(command)
        assert args[:3] == ["python3", "-I", "-c"]
        action, remote, paths = args[4], args[5], json.loads(args[6])
        assert paths == PATHS
        self.commands.append(action)
        if action == "capture":
            output = files.capture_submission(
                self.location(remote), paths=paths, filesystem_root=self.root
            )
        else:
            files.restore_submission(
                self.location(remote),
                paths=paths,
                expected_manifest_sha256=args[7],
                filesystem_root=self.root,
            )
            output = "ok"
        return SimpleNamespace(return_code=0, stdout=output)

    async def download_dir(self, remote, local):
        shutil.copytree(self.location(remote), local, dirs_exist_ok=True)

    async def upload_dir(self, local, remote):
        shutil.copytree(local, self.location(remote))


@pytest.fixture
def trial(tmp_path):
    root = tmp_path / "rollout"
    root.mkdir()
    source = tmp_path / "source"
    (source / "root").mkdir(parents=True)
    (source / "root/result.csv").write_bytes(b"a,b\n1,2\n")
    (source / "root/summary.json").write_bytes(b"{}")
    (source / "root/secret").write_text("do not copy")
    return SimpleNamespace(
        _config=SimpleNamespace(task_digest="task"),
        _task=SimpleNamespace(
            config=SimpleNamespace(
                verifier=VerifierConfig(submission_files=list(PATHS))
            )
        ),
        _docker_recovery_baseline=SimpleNamespace(
            task_digest="task",
            effective_config_digest="b" * 64,
            image_id="sha256:" + "a" * 64,
        ),
        _require_rollout_dir=lambda: root,
        _env=LocalRuntime(source, tmp_path / "remote-source"),
    )


@pytest.mark.asyncio
async def test_transport_preserves_missing_and_never_rewrites_checkpoint(
    trial, tmp_path
):
    """GH1136: immutable captured outputs are restored without recapturing later solver changes."""
    await transport.capture_submission(trial)
    root = trial._require_rollout_dir()
    receipt = (root / "submission.json").read_bytes()
    (trial._env.root / "root/result.csv").write_bytes(b"later mutation")
    await transport.capture_submission(trial)
    assert trial._env.commands == ["capture"]
    assert (root / "submission.json").read_bytes() == receipt
    target = tmp_path / "target"
    (target / "root").mkdir(parents=True)
    (target / "root/report.md").write_bytes(
        b"baseline default must not fill missing submission"
    )
    (target / "root/baseline").write_text("preserve")
    child = LocalRuntime(target, tmp_path / "remote-child")
    await transport.restore_submission(trial, child)
    assert (target / "root/result.csv").read_bytes() == b"a,b\n1,2\n"
    assert not (target / "root/report.md").exists()
    assert not (target / "root/secret").exists()
    assert (target / "root/baseline").read_text() == "preserve"


@pytest.mark.asyncio
async def test_tampered_bundle_is_rejected_before_upload(trial):
    """GH1136: checkpoint hash prevents altered submissions from entering fresh verification."""
    await transport.capture_submission(trial)
    (trial._require_rollout_dir() / "submission-evidence/files/0").write_bytes(
        b"tampered"
    )
    with pytest.raises(files.SubmissionError, match="mismatch"):
        await transport.restore_submission(trial, None)


@pytest.mark.parametrize(
    "config",
    [
        {"submission_files": ["/root"]},
        {"submission_files": list(PATHS), "workspace_recovery": True},
    ],
)
def test_explicit_contract_rejects_broad_or_ambiguous_restore(config):
    """GH1136: no accidental whole-home or mixed evidence recovery contract."""
    with pytest.raises(ValueError):
        VerifierConfig(**config)
