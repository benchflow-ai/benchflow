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

WEIGHTED_RUBRIC = {
    "criteria": [
        {
            "name": "publication_gate",
            "blocker": 1,
            "weight": 10,
            "description": "d",
            "guidance": "PASS when publication requirements hold.",
        },
        {
            "name": "research_quality",
            "blocker": 0,
            "weight": 7,
            "description": "d",
            "guidance": "Score 2, 1, or 0.",
        },
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
        """The recorded path is looked up in the root by its name only. The
        note gives the tasks-root rule's refusal, where this guard
        gave its own "does not resolve to a directory inside --tasks-root"."""
        from benchflow.review.runner import _source_task_dir

        secret = tmp_path / "victim"
        secret.mkdir()
        root = tmp_path / "tasks"
        root.mkdir()
        rollout = self._rollout(tmp_path, str(secret))
        resolved, reason = _source_task_dir(rollout, tasks_root=root)
        assert resolved is None
        assert reason == (
            f"task evidence skipped: {root.resolve()} has no folder victim"
        )

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


class TestDigestEnforcement:
    @pytest.mark.asyncio
    async def test_mismatched_task_evidence_is_excluded(self, tmp_path, monkeypatch):
        """A digest mismatch drops the task from evidence, not just notes it.

        Guards the round-3 PR #942 finding: an old rollout must not be
        reviewed against silently changed task content.
        """
        import benchflow.review.runner as runner_mod

        root = tmp_path / "tasks"
        task = root / "t1"
        (task / "verifier").mkdir(parents=True)
        (task / "task.md").write_text("body", encoding="utf-8")
        rollout = make_rollout(tmp_path / "jobs")
        (rollout / "config.json").write_text(
            json.dumps({"task_path": "/x/t1", "task_digest": "sha256:" + "a" * 64}),
            encoding="utf-8",
        )
        (rollout / "result.json").write_text(
            json.dumps(
                {"rewards": {"reward": 0.0}, "task_digest": "sha256:" + "a" * 64}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "benchflow._utils.task_authoring.task_digest",
            lambda _p: "sha256:" + "b" * 64,
        )
        uploads_seen: list[dict] = []

        async def fake_run(config: RolloutConfig):
            uploads_seen.append(dict(config.uploads or {}))
            leaf = Path(config.jobs_dir) / "job" / "w__0000"
            (leaf / "verifier").mkdir(parents=True)
            (leaf / "config.json").write_text("{}", encoding="utf-8")
            (leaf / "result.json").write_text(
                json.dumps({"rewards": {"reward": 0.0}}), encoding="utf-8"
            )

            class _R:
                error = None

            return _R()

        monkeypatch.setattr(benchflow, "run", fake_run)
        report, _ = await runner_mod.run_reviews(
            rollout, agent="gemini", out_dir=tmp_path / "out", tasks_root=root
        )
        trial = report.trials[0]
        assert any("digest mismatch" in n and "excluded" in n for n in trial.notes)
        assert all("/evidence/task" not in u.values() for u in uploads_seen)


class TestNetworkPosture:
    def test_default_wrapper_declares_no_internet(self, tmp_path):
        rollout = make_rollout(tmp_path)
        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")
        task_md = (dest / "task.md").read_text(encoding="utf-8")
        assert "allow_internet: false" in task_md

    @staticmethod
    def _docker_compose_paths(wrapper: Path, rollout_dir: Path) -> list[Path]:
        """Compose files the Docker backend stacks for a reviewer rollout."""
        from benchflow.rollout._setup import _task_disallows_internet
        from benchflow.sandbox.setup import _create_sandbox_environment
        from benchflow.task import RolloutPaths, Task

        task = Task(wrapper)
        sandbox = _create_sandbox_environment(
            "docker",
            task,
            wrapper,
            "review",
            RolloutPaths(rollout_dir=rollout_dir),
            # What Rollout passes for a non-oracle agent on this task.
            preserve_agent_network=_task_disallows_internet(task),
        )
        assert not sandbox._uses_compose
        return sandbox._docker_compose_paths

    def test_isolated_docker_reviewer_gets_net_admin_from_shared_rule(self, tmp_path):
        """Guards the removal of the reviewer's own cap_add overlay (PR #942).

        The agent-UID egress firewall programs iptables in ``main`` and needs
        NET_ADMIN. The Docker backend's no-web rule from the no-web NET_ADMIN change now grants
        it to the isolated reviewer, as to any no-web agent run.
        """
        from benchflow.sandbox._compose import (
            COMPOSE_NET_ADMIN_PATH,
            COMPOSE_NO_NETWORK_PATH,
        )

        rollout = make_rollout(tmp_path)
        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")
        paths = self._docker_compose_paths(dest, tmp_path / "review-rollout")

        assert COMPOSE_NET_ADMIN_PATH in paths
        assert COMPOSE_NO_NETWORK_PATH not in paths

    def test_open_network_docker_reviewer_gets_no_net_admin(self, tmp_path):
        """Guards the removal of the reviewer overlay: open reviewers arm no
        firewall and never received NET_ADMIN."""
        from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH

        rollout = make_rollout(tmp_path)
        dest, _ = assemble_review_task(
            rollout, None, RUBRIC, tmp_path / "w", open_network=True
        )
        paths = self._docker_compose_paths(dest, tmp_path / "review-rollout")

        assert COMPOSE_NET_ADMIN_PATH not in paths

    def test_wrapper_has_no_compose_file(self, tmp_path):
        """A task compose file flips Daytona and other backends into compose
        strategies, so the wrapper never ships one."""
        rollout = make_rollout(tmp_path)
        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")
        assert not (dest / "environment").exists()

    @pytest.mark.asyncio
    async def test_docker_reviewer_wrapper_ships_no_compose_overlay(
        self, tmp_path, monkeypatch
    ):
        """Guards the removal of the reviewer's Docker-only cap_add overlay.

        PR #942 wrote ``environment/docker-compose.yaml`` for isolated Docker
        reviewers; the shared no-web rule makes it redundant.
        """
        rollout = make_rollout(tmp_path / "jobs")
        seen: list[tuple[str, bool]] = []

        async def fake_run(config: RolloutConfig):
            seen.append(
                (config.environment, (Path(config.task_path) / "environment").exists())
            )

            class _R:
                error = "stop"

            return _R()

        monkeypatch.setattr(benchflow, "run", fake_run)
        await run_reviews(rollout, agent="gemini", out_dir=tmp_path / "out")

        assert seen == [("docker", False)]

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

    def test_weighted_review_rubric_is_claimed(self, tmp_path):
        task = tmp_path / "weighted-task"
        (task / "verifier").mkdir(parents=True)
        target = task / "verifier" / "rubric.json"
        target.write_text(json.dumps(WEIGHTED_RUBRIC), encoding="utf-8")
        assert find_task_rubric(task) == target
        assert load_rubric(target).contract == "v0.2"

    def test_judge_rubric_passes_task_authoring_check(self, tmp_path):
        from benchflow._utils.task_authoring import _check_review_rubric

        verifier = tmp_path / "verifier"
        verifier.mkdir()
        (verifier / "rubric.json").write_text(
            json.dumps(JUDGE_RUBRIC), encoding="utf-8"
        )
        assert _check_review_rubric(verifier, verifier_label="verifier") == []

    def test_weighted_review_rubric_passes_task_authoring_check(self, tmp_path):
        from benchflow._utils.task_authoring import _check_review_rubric

        verifier = tmp_path / "verifier"
        verifier.mkdir()
        (verifier / "rubric.json").write_text(
            json.dumps(WEIGHTED_RUBRIC), encoding="utf-8"
        )
        assert _check_review_rubric(verifier, verifier_label="verifier") == []

    def test_invalid_weighted_review_rubric_fails_task_authoring_check(self, tmp_path):
        from benchflow._utils.task_authoring import _check_review_rubric

        verifier = tmp_path / "verifier"
        verifier.mkdir()
        invalid = json.loads(json.dumps(WEIGHTED_RUBRIC))
        invalid["criteria"][1].pop("weight")
        (verifier / "rubric.json").write_text(json.dumps(invalid), encoding="utf-8")
        problems = _check_review_rubric(verifier, verifier_label="verifier")
        assert len(problems) == 1
        assert "both blocker and weight" in problems[0]

    def test_all_misspelled_review_rubric_is_claimed(self, tmp_path):
        """Round-3 fix: only the judge dialect is disclaimed; a rubric with
        every review key misspelled must be claimed (and rejected loudly by
        load_rubric), never silently replaced by the default."""
        from benchflow.review.config import (
            ReviewRubricError,
            is_review_rubric_file,
            load_rubric,
        )

        task = tmp_path / "task"
        (task / "verifier").mkdir(parents=True)
        target = task / "verifier" / "rubric.json"
        target.write_text(
            json.dumps({"criteria": [{"nme": "x", "guidnce": "g"}]}),
            encoding="utf-8",
        )
        assert is_review_rubric_file(target)
        assert find_task_rubric(task) == target
        with pytest.raises(ReviewRubricError):
            load_rubric(target)

    @pytest.mark.parametrize(
        "document",
        [
            {"criteria": [{"id": "only-id"}]},
            {"criteria": [{"match_criteria": "only-match"}]},
            {"criteria": "not-a-list"},
        ],
    )
    def test_ambiguous_or_malformed_dialects_are_claimed(self, tmp_path, document):
        """Guards PR #942: only the full judge shape is disclaimed."""

        from benchflow.review.config import ReviewRubricError, is_review_rubric_file

        task = tmp_path / "task"
        (task / "verifier").mkdir(parents=True)
        target = task / "verifier" / "rubric.json"
        target.write_text(json.dumps(document), encoding="utf-8")
        assert is_review_rubric_file(target)
        assert find_task_rubric(task) == target
        with pytest.raises(ReviewRubricError):
            load_rubric(target)

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

        blob = base64.b64decode(sealed.blob_b64)
        assert b"sk-super-secret" not in blob
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
        # Round 4: the tag covers the WHOLE blob (IV ‖ ciphertext) and the
        # receiver derives its IV from those same authenticated bytes — no
        # second, unauthenticated IV copy exists.
        verifier = _hmac.HMAC(mac_key, hashes.SHA256())
        verifier.update(blob)
        verifier.verify(bytes.fromhex(sealed.tag_hex))
        iv, ciphertext = blob[:16], blob[16:]
        decryptor = Cipher(algorithms.AES(enc_key), modes.CTR(iv)).decryptor()
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
        # Behavior, not naming: the env file must live OUTSIDE the
        # root-only seal directory (or a non-root exec user could never
        # source it), and the upload must have been chowned to the owner.
        from benchflow.sandbox.agentcore_sealed import SEAL_DIR

        assert not path.startswith(SEAL_DIR + "/")
        self._assert_unrecoverable(commands, b"tok-abc-123")


_SOLVER = {
    "task_name": "t",
    "task_digest": "d1",
    "rewards": {"reward": 1.0},
    "verifier_error": None,
}


class TestReReviewHidesTheVerdict:
    """A re-review's evidence copy held the verdict of record: a scored
    result.json, the score exports and a regrade's outputs. In a reported
    rollout, 28 of 30 reviewers of one panel opened result.json."""

    def scored(self, tmp_path: Path) -> Path:
        rollout = make_rollout(tmp_path)
        (rollout / "solver.json").write_text(json.dumps(_SOLVER))
        scored = {
            **_SOLVER,
            "rewards": {"reward": 0.0},
            "scoring": {"status": "complete", "rubric_reward": 0.3},
        }
        (rollout / "result.json").write_text(json.dumps(scored))
        (rollout / "rewards.jsonl").write_text('{"reward": 0.0}\n')
        (rollout / "results.jsonl").write_text('{"reward": 0.0}\n')
        (rollout / "trainer").mkdir()
        (rollout / "trainer" / "sample.json").write_text('{"reward": 0.0}')
        (rollout / "regrade.json").write_text('{"latest": "r1"}')
        (rollout / "regrade" / "r1").mkdir(parents=True)
        (rollout / "regrade" / "r1" / "regrade.json").write_text('{"new": 0.0}')
        (rollout / "result.json.pre-score").write_text(json.dumps(scored))
        # The agent's own folder of that name, below the root, is evidence.
        (rollout / "artifacts" / "trainer").mkdir(parents=True)
        (rollout / "artifacts" / "trainer" / "notes.txt").write_text("mine")
        return rollout

    def test_a_scored_rollout_is_copied_as_before_its_review(self, tmp_path):
        rollout = self.scored(tmp_path)

        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")

        trial = dest / "evidence" / "trial"
        assert json.loads((trial / "result.json").read_text()) == _SOLVER
        for name in (
            "rewards.jsonl",
            "results.jsonl",
            "trainer",
            "regrade",
            "regrade.json",
            "result.json.pre-score",
        ):
            assert not (trial / name).exists(), name
        assert (trial / "artifacts" / "trainer" / "notes.txt").read_text() == "mine"
        assert (trial / "solver.json").is_file()
        assert not (trial / "result.json").stat().st_mode & 0o222

    def test_the_admitted_verifier_revision_is_the_record(self, tmp_path):
        rollout = self.scored(tmp_path)
        recovery = rollout / "verifier-recovery" / "a1"
        recovery.mkdir(parents=True)
        (recovery / "recovery.json").write_text(
            json.dumps(
                {"task_digest": "d1", "status": "complete", "rewards": {"reward": 0.5}}
            )
        )
        (rollout / "verification.json").write_text(
            json.dumps({"attempt": "verifier-recovery/a1"})
        )

        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")

        record = json.loads((dest / "evidence" / "trial" / "result.json").read_text())
        assert record["rewards"] == {"reward": 0.5} and "scoring" not in record

    def test_a_linked_solver_record_brings_no_host_file_in(self, tmp_path):
        rollout = self.scored(tmp_path)
        secret = tmp_path / "host-login.json"
        secret.write_text(json.dumps({"rewards": None, "token": "HOST-ONLY"}))
        (rollout / "solver.json").unlink()
        (rollout / "solver.json").symlink_to(secret)

        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")

        trial = dest / "evidence" / "trial"
        assert "HOST-ONLY" not in (trial / "result.json").read_text()
        assert "scoring" not in json.loads((trial / "result.json").read_text())

    def test_a_first_review_copies_the_rollout_as_it_is(self, tmp_path):
        rollout = make_rollout(tmp_path)
        (rollout / "rewards.jsonl").write_text('{"reward": 0.0}\n')
        (rollout / "trainer").mkdir()
        (rollout / "trainer" / "sample.json").write_text("{}")

        dest, _ = assemble_review_task(rollout, None, RUBRIC, tmp_path / "w")

        trial = dest / "evidence" / "trial"
        for name in ("result.json", "rewards.jsonl", "trainer/sample.json"):
            assert (trial / name).read_bytes() == (rollout / name).read_bytes()


def _tree(root: Path) -> dict[str, object]:
    """Every path below *root* with its mode, and each file's sha256."""
    import hashlib
    import stat

    out: dict[str, object] = {}
    for item in sorted(root.rglob("*")):
        info = item.lstat()
        key = str(item.relative_to(root))
        if item.is_file() and not item.is_symlink():
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            out[key] = (digest, stat.S_IMODE(info.st_mode))
        else:
            out[key] = stat.S_IMODE(info.st_mode)
    return out


# Records other tools keep beside a scored result.json, each carrying the
# verdict: a re-score's flip, publication history, an earlier review's state,
# a regrade summary, a verifier replay, and backups named ``<record>.pre-<x>``.
_VERDICT_RECORDS = {
    "rpass.json": '{"flip": "pass->fail"}',
    "publication_history/v1.json": '{"passed": false}',
    "reviews.pre-posthoc/old/judge.json": '{"passed": false}',
    "regrade-summary.json": '{"original_score": 0.0}',
    "verifier_replay/reward.txt": "0\n",
    "verifier.pre-replay/reward.txt": "1\n",
    "solver.json.pre-rescore": '{"rewards": {"reward": 0.0}}',
    "result.json.pre-tests-failed": '{"scoring": {"passed": false}}',
}


# A scored record: the verifier said 1.0, the review's verdict is 0.0.
_SCORED = {
    **_SOLVER,
    "rewards": {"reward": 0.0, "verifier_reward": 1.0, "rubric_reward": 0.4},
    "scoring": {"status": "complete", "passed": False},
}


class TestReReviewCopiesOnlyTheRolloutLayout:
    """When the rollout holds a verdict, the copy is an allow-list of the
    rollout layout; what it leaves out is named beside it, never shown."""

    def scored(self, tmp_path: Path, *, solver: bool = True) -> Path:
        rollout = make_rollout(tmp_path)
        if solver:
            (rollout / "solver.json").write_text(json.dumps(_SOLVER))
        (rollout / "result.json").write_text(json.dumps(_SCORED))
        (rollout / "rewards.jsonl").write_text('{"reward": 0.0}\n')
        (rollout / "results.jsonl").write_text('{"reward": 0.0, "passed": false}\n')
        (rollout / "trainer").mkdir()
        (rollout / "trainer" / "verifiers.jsonl").write_text('{"reward": 0.0}\n')
        return rollout

    @staticmethod
    def review(tmp_path: Path, rollout: Path, name: str = "w"):
        dest, uploads = assemble_review_task(rollout, None, RUBRIC, tmp_path / name)
        evidence = dest / "evidence"
        record_path = evidence / "trial-left-out.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        return evidence / "trial", record, uploads

    @pytest.mark.parametrize("planted", sorted(_VERDICT_RECORDS))
    def test_a_verdict_record_beside_the_result_is_left_out_and_named(
        self, tmp_path, planted
    ):
        rollout = self.scored(tmp_path)
        target = rollout / planted
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_VERDICT_RECORDS[planted])
        top = Path(planted).parts[0]

        trial, record, uploads = self.review(tmp_path, rollout)

        assert not (trial / top).exists()
        assert top in record["left_out"]
        assert record["count"] == len(record["left_out"])
        assert uploads[str(trial.parent / "trial-left-out.json")] == (
            "/evidence/trial-left-out.json"
        )

    def test_all_of_them_at_once_and_an_unknown_name(self, tmp_path):
        rollout = self.scored(tmp_path)
        for rel, body in _VERDICT_RECORDS.items():
            (rollout / rel).parent.mkdir(parents=True, exist_ok=True)
            (rollout / rel).write_text(body)
        # A name no writer is known to use: left out too, and named.
        (rollout / "zz-new-tool-output.json").write_text('{"verdict": "VERDICT-BODY"}')
        (rollout / "verifier").mkdir()
        (rollout / "verifier" / "reward.txt").write_text("1\n")

        trial, record, _ = self.review(tmp_path, rollout)

        assert sorted(p.name for p in trial.iterdir()) == [
            "config.json",
            "result.json",
            "solver.json",
            "trajectory",
            "verifier",
        ]
        expected = sorted(
            {Path(rel).parts[0] for rel in _VERDICT_RECORDS}
            | {"rewards.jsonl", "results.jsonl", "trainer", "zz-new-tool-output.json"}
        )
        assert record["left_out"] == expected
        assert record["count"] == len(expected)
        assert "VERDICT-BODY" not in json.dumps(record)
        assert json.loads((trial / "result.json").read_text()) == _SOLVER
        seen = [
            str(p.relative_to(trial))
            for p in trial.rglob("*")
            if p.is_file()
            and ('"reward": 0.0' in p.read_text() or "scoring" in p.read_text())
        ]
        assert seen == []

    def test_the_layout_names_are_copied(self, tmp_path):
        from benchflow.review import wrapper

        rollout = self.scored(tmp_path)
        kept = sorted(
            wrapper._REVIEW_LAYOUT
            - {"result.json"}
            - {name for name in wrapper._REVIEW_LAYOUT if (rollout / name).exists()}
        )
        for name in kept:
            target = rollout / name
            if "." in name:
                target.write_text("{}")
            else:
                target.mkdir()
                (target / "inside.txt").write_text("x")

        trial, record, _ = self.review(tmp_path, rollout)

        assert {p.name for p in trial.iterdir()} == set(wrapper._REVIEW_LAYOUT)
        assert record["left_out"] == ["results.jsonl", "rewards.jsonl", "trainer"]

    @pytest.mark.parametrize("solver", ["absent", "linked"])
    def test_without_a_solver_record_only_the_verifiers_reward_is_kept(
        self, tmp_path, solver
    ):
        """BenchFlow up to 0.7.7 wrote no solver.json, and the copy drops a
        linked one: the record falls back to result.json, whose scored reward
        and rubric score are the verdict's."""
        rollout = self.scored(tmp_path, solver=solver == "linked")
        if solver == "linked":
            elsewhere = tmp_path / "elsewhere-solver.json"
            elsewhere.write_text(json.dumps(_SOLVER))
            (rollout / "solver.json").unlink()
            (rollout / "solver.json").symlink_to(elsewhere)

        trial, record, _ = self.review(tmp_path, rollout)

        result = json.loads((trial / "result.json").read_text())
        assert result == {
            "task_name": "t",
            "task_digest": "d1",
            "rewards": {"reward": 1.0},
            "verifier_error": None,
        }
        assert "rubric_reward" not in (trial / "result.json").read_text()
        assert ("solver.json" in record["left_out"]) is (solver == "linked")

    def test_without_a_verifier_reward_no_reward_is_kept(self, tmp_path):
        rollout = self.scored(tmp_path, solver=False)
        scored = {**_SCORED, "rewards": {"reward": 0.0, "rubric_reward": 0.4}}
        (rollout / "result.json").write_text(json.dumps(scored))

        trial, _, _ = self.review(tmp_path, rollout)

        result = json.loads((trial / "result.json").read_text())
        assert "rewards" not in result and "scoring" not in result

    @pytest.mark.parametrize("where", ["inside", "outside"])
    @pytest.mark.parametrize("solver", [True, False])
    def test_a_linked_result_counts_as_a_held_verdict(self, tmp_path, where, solver):
        rollout = self.scored(tmp_path, solver=solver)
        target = (rollout if where == "inside" else tmp_path) / "scored-record.json"
        target.write_text(json.dumps(_SCORED))
        (rollout / "result.json").unlink()
        (rollout / "result.json").symlink_to(target)

        trial, record, _ = self.review(tmp_path, rollout)

        for name in ("rewards.jsonl", "results.jsonl", "trainer", "scored-record.json"):
            assert not (trial / name).exists(), name
        result = json.loads((trial / "result.json").read_text())
        assert result == (_SOLVER if solver else {})
        assert {"rewards.jsonl", "results.jsonl", "trainer"} <= set(record["left_out"])

    def test_a_rollout_without_a_verdict_is_copied_byte_for_byte(self, tmp_path):
        from benchflow.review.wrapper import copy_evidence

        rollout = make_rollout(tmp_path)
        (rollout / "solver.json").write_text(json.dumps(_SOLVER))
        (rollout / "rewards.jsonl").write_text('{"reward": 0.0}\n')
        (rollout / "trainer").mkdir()
        (rollout / "trainer" / "sample.json").write_text("{}")
        (rollout / "trajectory" / "acp_trajectory.jsonl").write_text("{}\n")
        for rel, body in _VERDICT_RECORDS.items():
            (rollout / rel).parent.mkdir(parents=True, exist_ok=True)
            (rollout / rel).write_text(body)
        (rollout / "zz-new-tool-output.json").write_text("{}")
        (rollout / "trainer" / "sample.json").chmod(0o600)

        trial, record, uploads = self.review(tmp_path, rollout)

        reference = tmp_path / "reference"
        copy_evidence(rollout, reference, extra_excludes=("evidence",))
        assert _tree(trial) == _tree(reference)
        assert len(_tree(trial)) == len(list(reference.rglob("*")))
        assert record is None
        assert set(uploads.values()) == {"/evidence/trial"}
