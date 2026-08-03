"""Boundary tests for rubric review: evidence hygiene, stale artifacts,
rubric-dialect discrimination, and sealed AgentCore transfers.

Guards the review-isolation and correctness fixes on PR #942.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchflow
from benchflow.review.config import Rubric, find_task_rubric, load_rubric
from benchflow.review.runner import run_reviews
from benchflow.review.wrapper import assemble_review_task
from benchflow.rollout import RolloutConfig

RUBRIC = Rubric.model_validate(
    {
        "criteria": [
            {
                "name": "method_soundness",
                "description": "d",
                "guidance": "PASS when sound; FAIL otherwise.",
            }
        ]
    }
)

JUDGE_RUBRIC = {
    "criteria": [
        {"id": "matches-figure", "match_criteria": "The plot matches."},
        {"id": "cites-data", "match_criteria": "Cites the dataset."},
    ]
}


def make_rollout(root: Path, name: str = "rollout-a") -> Path:
    rollout = root / name
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "config.json").write_text("{}", encoding="utf-8")
    (rollout / "result.json").write_text(
        json.dumps({"rewards": {"reward": 0.0}, "error": None}), encoding="utf-8"
    )
    return rollout


class TestSymlinkRejection:
    def test_external_symlink_is_dropped_not_dereferenced(self, tmp_path):
        """A source-controlled symlink must not materialize a host file into
        the uploaded evidence."""
        secret = tmp_path / "host-secret.txt"
        secret.write_text("HOST-ONLY", encoding="utf-8")
        rollout = make_rollout(tmp_path)
        (rollout / "leak.txt").symlink_to(secret)
        (rollout / "leakdir").symlink_to(tmp_path)

        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")
        trial_copy = dest / "evidence" / "trial"
        assert not (trial_copy / "leak.txt").exists()
        assert not (trial_copy / "leakdir").exists()
        assert not list(trial_copy.rglob("host-secret.txt"))

    def test_nested_symlink_is_dropped(self, tmp_path):
        secret = tmp_path / "deep-secret.txt"
        secret.write_text("HOST-ONLY", encoding="utf-8")
        rollout = make_rollout(tmp_path)
        nested = rollout / "verifier"
        nested.mkdir()
        (nested / "inner-link.json").symlink_to(secret)

        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")
        assert not list((dest / "evidence").rglob("inner-link.json"))


class TestTaskEvidenceSanitization:
    def test_skills_and_rubric_are_excluded(self, tmp_path):
        rollout = make_rollout(tmp_path)
        task = tmp_path / "task"
        (task / "environment" / "skills" / "mesh").mkdir(parents=True)
        (task / "environment" / "skills" / "mesh" / "SKILL.md").write_text(
            "skill body", encoding="utf-8"
        )
        (task / "verifier").mkdir()
        (task / "verifier" / "rubric.json").write_text(
            json.dumps({"criteria": []}), encoding="utf-8"
        )
        (task / "verifier" / "test.sh").write_text("echo hi", encoding="utf-8")
        (task / "task.md").write_text("body", encoding="utf-8")

        dest, _ = assemble_review_task(rollout, task, RUBRIC, tmp_path / "w")
        task_copy = dest / "evidence" / "task"
        assert (task_copy / "task.md").is_file()
        assert (task_copy / "verifier" / "test.sh").is_file()
        assert not list(task_copy.rglob("SKILL.md"))
        assert not list(task_copy.rglob("skills"))
        assert not list(task_copy.rglob("rubric.json"))


class TestUntrustedTaskPath:
    """A rollout-recorded task_path must never be dereferenced.

    Guards the P0 from the PR #942 re-review: a downloaded rollout could
    name ~/.ssh (or any directory holding .env / cloud credentials) and have
    it recursively copied into reviewer evidence.
    """

    def _rollout(self, tmp_path: Path, task_path: str) -> Path:
        rollout = make_rollout(tmp_path)
        (rollout / "config.json").write_text(
            json.dumps({"task_path": task_path}), encoding="utf-8"
        )
        return rollout

    def test_absolute_outside_path_is_refused_without_root(self, tmp_path):
        from benchflow.review.runner import _source_task_dir

        secret = tmp_path / "victim"
        secret.mkdir()
        (secret / ".env").write_text("SECRET=1", encoding="utf-8")
        rollout = self._rollout(tmp_path, str(secret))
        resolved, reason = _source_task_dir(rollout, tasks_root=None)
        assert resolved is None
        assert "untrusted" in (reason or "")

    def test_path_escaping_the_root_is_refused(self, tmp_path):
        from benchflow.review.runner import _source_task_dir

        secret = tmp_path / "victim"
        secret.mkdir()
        root = tmp_path / "tasks"
        root.mkdir()
        rollout = self._rollout(tmp_path, str(secret))
        resolved, reason = _source_task_dir(rollout, tasks_root=root)
        assert resolved is None
        assert "tasks-root" in (reason or "")

    def test_traversal_in_recorded_path_is_refused(self, tmp_path):
        from benchflow.review.runner import _source_task_dir

        root = tmp_path / "tasks"
        (root / "real").mkdir(parents=True)
        rollout = self._rollout(tmp_path, "/x/../../../../etc/passwd")
        resolved, _ = _source_task_dir(rollout, tasks_root=root)
        assert resolved is None

    def test_name_inside_root_resolves(self, tmp_path):
        from benchflow.review.runner import _source_task_dir

        root = tmp_path / "tasks"
        (root / "real-task").mkdir(parents=True)
        rollout = self._rollout(tmp_path, "/anywhere/else/real-task")
        resolved, reason = _source_task_dir(rollout, tasks_root=root)
        assert resolved == (root / "real-task").resolve()
        assert reason is None


class TestNetworkPosture:
    def test_default_wrapper_declares_no_internet(self, tmp_path):
        rollout = make_rollout(tmp_path)
        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")
        task_md = (dest / "task.md").read_text(encoding="utf-8")
        assert "allow_internet: false" in task_md

    def test_docker_isolation_ships_net_admin_overlay(self, tmp_path):
        """The docker egress firewall programs iptables in-container and
        needs NET_ADMIN; the overlay is docker-only."""
        rollout = make_rollout(tmp_path)
        dest, _ = assemble_review_task(
            rollout, None, RUBRIC, tmp_path / "w", net_admin_overlay=True
        )
        overlay = (dest / "environment" / "docker-compose.yaml").read_text(
            encoding="utf-8"
        )
        assert "NET_ADMIN" in overlay

    def test_non_docker_wrapper_has_no_compose_file(self, tmp_path):
        """A task compose file flips other backends into compose strategies;
        the overlay must never ship unless requested."""
        rollout = make_rollout(tmp_path)
        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")
        assert not (dest / "environment").exists()

    def test_open_network_is_explicit_opt_in(self, tmp_path):
        rollout = make_rollout(tmp_path)
        dest, _ = assemble_review_task(
            rollout, None, RUBRIC, tmp_path / "w", open_network=True
        )
        task_md = (dest / "task.md").read_text(encoding="utf-8")
        assert "allow_internet" not in task_md


class TestRubricDialectDiscrimination:
    def test_judge_rubric_is_not_claimed(self, tmp_path):
        """llm-judge rubrics ({id, match_criteria}) share the filename and
        must be left alone."""
        task = tmp_path / "task"
        (task / "verifier").mkdir(parents=True)
        (task / "verifier" / "rubric.json").write_text(
            json.dumps(JUDGE_RUBRIC), encoding="utf-8"
        )
        assert find_task_rubric(task) is None

    def test_review_rubric_is_claimed(self, tmp_path):
        task = tmp_path / "task"
        (task / "verifier").mkdir(parents=True)
        target = task / "verifier" / "rubric.json"
        target.write_text(
            json.dumps(
                {"criteria": [{"name": "x", "description": "d", "guidance": "g"}]}
            ),
            encoding="utf-8",
        )
        assert find_task_rubric(task) == target

    def test_judge_rubric_passes_task_authoring_check(self, tmp_path):
        from benchflow._utils.task_authoring import _check_review_rubric

        verifier = tmp_path / "verifier"
        verifier.mkdir()
        (verifier / "rubric.json").write_text(
            json.dumps(JUDGE_RUBRIC), encoding="utf-8"
        )
        assert _check_review_rubric(verifier, verifier_label="verifier") == []

    def test_default_rubric_is_strict_valid(self):
        rubric = load_rubric(None)
        assert len(rubric.criteria) >= 1


class TestStaleOutDirReuse:
    @pytest.mark.asyncio
    async def test_second_run_never_sees_first_runs_artifacts(
        self, tmp_path, monkeypatch
    ):
        """Reusing --out-dir must not let an old reward-1 review make a new
        failed review look valid."""
        rollout = make_rollout(tmp_path / "jobs")
        out_dir = tmp_path / "out"

        class Run:
            def __init__(self, reward, payload):
                self.reward = reward
                self.payload = payload

            async def __call__(self, config: RolloutConfig):
                leaf = Path(config.jobs_dir) / "job" / "wrapper__0000"
                (leaf / "verifier").mkdir(parents=True)
                (leaf / "config.json").write_text("{}", encoding="utf-8")
                (leaf / "result.json").write_text(
                    json.dumps({"rewards": {"reward": self.reward}}),
                    encoding="utf-8",
                )
                if self.payload is not None:
                    (leaf / "verifier" / "review-result.json").write_text(
                        json.dumps(self.payload), encoding="utf-8"
                    )

                class _R:
                    error = None

                return _R()

        good = {
            "trial_name": "rollout-a",
            "summary": "fine",
            "checks": {
                "reward_hacking": {"explanation": "e", "outcome": "pass"},
                "task_specification": {"explanation": "e", "outcome": "pass"},
            },
        }
        monkeypatch.setattr(benchflow, "run", Run(1.0, good))
        first, _ = await run_reviews(rollout, agent="gemini", out_dir=out_dir)
        assert first.trials[0].review_valid is True

        # Second invocation, same out_dir: reviewer produces nothing at all.
        monkeypatch.setattr(benchflow, "run", Run(0.0, None))
        second, _ = await run_reviews(rollout, agent="gemini", out_dir=out_dir)
        trial = second.trials[0]
        assert trial.review_valid is False
        assert trial.checks is None
        assert "did not produce" in (trial.error or "")

    @pytest.mark.asyncio
    async def test_trial_records_rubric_and_exact_leaf(self, tmp_path, monkeypatch):
        rollout = make_rollout(tmp_path / "jobs")
        leaves: list[Path] = []

        async def run(config: RolloutConfig):
            leaf = Path(config.jobs_dir) / "job" / "wrapper__0000"
            (leaf / "verifier").mkdir(parents=True)
            (leaf / "config.json").write_text("{}", encoding="utf-8")
            (leaf / "result.json").write_text(
                json.dumps({"rewards": {"reward": 1.0}}), encoding="utf-8"
            )
            (leaf / "verifier" / "review-result.json").write_text(
                json.dumps(
                    {
                        "trial_name": "rollout-a",
                        "summary": "fine",
                        "checks": {
                            "reward_hacking": {
                                "explanation": "e",
                                "outcome": "pass",
                            },
                            "task_specification": {
                                "explanation": "e",
                                "outcome": "not_applicable",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            leaves.append(leaf)

            class _R:
                error = None

            return _R()

        monkeypatch.setattr(benchflow, "run", run)
        report, _ = await run_reviews(rollout, agent="gemini", out_dir=tmp_path / "out")
        trial = report.trials[0]
        assert trial.criteria == ["reward_hacking", "task_specification"]
        assert trial.rubric_path is not None
        assert Path(trial.reviewer_rollout) == leaves[0]
        assert report.criteria == ["reward_hacking", "task_specification"]


class TestSealedAgentCoreUploads:
    """The sealed channel is the only transport on a command-logging backend."""

    @staticmethod
    def _keypair():
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = (
            private.public_key()
            .public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode()
        )
        return private, pem

    def test_seal_hides_plaintext_and_roundtrips(self):
        """Ciphertext-only transport with an authenticating tag."""
        import base64

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives import hmac as _hmac
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.ciphers import (
            Cipher,
            algorithms,
            modes,
        )

        from benchflow.sandbox.agentcore_sealed import seal

        private, pem = self._keypair()
        payload = b'{"GEMINI_API_KEY": "sk-super-secret"}'
        sealed = seal(pem, payload)

        assert b"sk-super-secret" not in base64.b64decode(sealed.ciphertext_b64)
        assert len(bytes.fromhex(sealed.tag_hex)) == 32

        secret = private.decrypt(
            base64.b64decode(sealed.wrapped_key_b64),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        enc_key, mac_key = secret[:32], secret[32:]
        ciphertext = base64.b64decode(sealed.ciphertext_b64)
        verifier = _hmac.HMAC(mac_key, hashes.SHA256())
        verifier.update(ciphertext)
        verifier.verify(bytes.fromhex(sealed.tag_hex))
        decryptor = Cipher(
            algorithms.AES(enc_key), modes.CTR(bytes.fromhex(sealed.iv_hex))
        ).decryptor()
        assert decryptor.update(ciphertext) == payload

    @staticmethod
    def _channel_with_recorder():
        from benchflow.sandbox.agentcore_sealed import SealedChannel

        commands: list[str] = []

        class _R:
            return_code = 0
            stdout = ""
            stderr = ""

        async def exec_raw(command, **_kwargs):
            commands.append(command)
            return _R()

        import logging

        channel = SealedChannel(exec_raw, logging.getLogger("test"))
        return channel, commands

    @staticmethod
    def _assert_unrecoverable(commands: list[str], *needles: bytes) -> None:
        """Literal search alone is too weak: reversible base64 would pass.

        Decode every long token in every command and assert none of the
        needles is recoverable.
        """
        import base64 as _b64
        import re as _re

        assert commands, "no commands captured"
        for command in commands:
            for needle in needles:
                assert needle.decode(errors="ignore") not in command
            for blob in _re.findall(r"[A-Za-z0-9+/=]{16,}", command):
                try:
                    decoded = _b64.b64decode(blob + "==", validate=False)
                except Exception:
                    continue
                for needle in needles:
                    assert needle not in decoded

    @pytest.mark.asyncio
    async def test_upload_commands_never_contain_plaintext(self):
        channel, commands = self._channel_with_recorder()
        _, pem = self._keypair()
        channel._public_key = pem
        secret = b'launch {"env": {"GEMINI_API_KEY": "sk-super-secret-value"}}'
        await channel.upload(secret, target="/x/launch_config.json")
        self._assert_unrecoverable(
            commands, b"sk-super-secret-value", b"GEMINI_API_KEY"
        )

    @pytest.mark.asyncio
    async def test_staged_env_never_appears_in_commands(self):
        """exec(env=...) must not leak the environment into a command body."""
        channel, commands = self._channel_with_recorder()
        _, pem = self._keypair()
        channel._public_key = pem
        path = await channel.stage_env({"TOKEN": "tok-abc-123"})
        assert path.startswith("/tmp/.bf_sealed/env_")
        self._assert_unrecoverable(commands, b"tok-abc-123")

    def test_write_text_file_routes_through_sealed_channel(self):
        import inspect

        from benchflow.sandbox.agentcore import AgentCoreSandbox

        source = inspect.getsource(AgentCoreSandbox.write_text_file)
        assert "self._sealed.upload" in source
        assert "b64encode" not in source

    def test_tar_path_routes_through_sealed_channel(self):
        import inspect

        from benchflow.sandbox.agentcore import AgentCoreSandbox

        source = inspect.getsource(AgentCoreSandbox._upload_via_tar)
        assert "self._sealed.upload" in source
