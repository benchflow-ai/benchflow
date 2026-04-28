"""Trial on-disk layout and in-sandbox path constants.

CONTRACT SURFACE — semver-stable. Changes here break downstream importers.
Prefer extending in periphery unless the shape itself must change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class EnvironmentPaths:
    """Static paths inside the Linux container."""

    logs_dir: PurePosixPath = PurePosixPath("/logs")
    agent_dir: PurePosixPath = logs_dir / "agent"
    verifier_dir: PurePosixPath = logs_dir / "verifier"
    artifacts_dir: PurePosixPath = logs_dir / "artifacts"
    tests_dir: PurePosixPath = PurePosixPath("/tests")
    solution_dir: PurePosixPath = PurePosixPath("/solution")
    reward_text_path: PurePosixPath = verifier_dir / "reward.txt"
    reward_json_path: PurePosixPath = verifier_dir / "reward.json"


@dataclass(frozen=True)
class TrialPaths:
    """Disk layout for one trial output directory."""

    trial_dir: Path

    def mkdir(self) -> None:
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.verifier_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_path(self) -> Path:
        return self.trial_dir / "config.json"

    @property
    def agent_dir(self) -> Path:
        return self.trial_dir / "agent"

    @property
    def artifacts_dir(self) -> Path:
        return self.trial_dir / "artifacts"

    @property
    def artifacts_manifest_path(self) -> Path:
        return self.artifacts_dir / "manifest.json"

    @property
    def verifier_dir(self) -> Path:
        return self.trial_dir / "verifier"

    @property
    def test_stdout_path(self) -> Path:
        return self.verifier_dir / "test-stdout.txt"

    @property
    def test_stderr_path(self) -> Path:
        return self.verifier_dir / "test-stderr.txt"

    @property
    def reward_text_path(self) -> Path:
        return self.verifier_dir / "reward.txt"

    @property
    def reward_json_path(self) -> Path:
        return self.verifier_dir / "reward.json"

    @property
    def result_path(self) -> Path:
        return self.trial_dir / "result.json"

    @property
    def exception_message_path(self) -> Path:
        return self.trial_dir / "exception.txt"

    @property
    def log_path(self) -> Path:
        return self.trial_dir / "trial.log"


__all__ = ["EnvironmentPaths", "TrialPaths"]
