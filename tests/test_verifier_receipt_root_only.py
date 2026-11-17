"""A trial that keeps services reads its start receipt only from a folder only root can write.

The test command's start receipt and the ``.exit`` and ``.abandoned`` files
beside it live in ``/run/runtime``, which is 1777. While a trial's kept
services run beside the verifier, as the agent's uid, they can list that
folder and create names in it: an ``<receipt>.exit`` they wrote ("ended 0")
for tests still running at the deadline had them scored from whatever
``verifier/`` held, a passing reward written early included. Such a trial's
verifier is root, so each attempt's receipt now goes in a new folder of its
own that only root can write, and the probes read only that one. Since
its reward provenance every trial does (integrity off and audit keep
every process of the agent's alive through grading).
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from benchflow._utils import kill_record
from benchflow.rollout._setup import _verify_rollout
from benchflow.task import Verifier
from benchflow.task.verifier_core import _RECEIPT_DIR_COMMAND
from tests.test_keep_services_verify import _Planes
from tests.test_verifier_lost_exec import _score, _setup

ROOT_ONLY = re.compile(r"/run/runtime/verifier-([0-9a-f]{32})/verifier-started-\1")
# What a kept service can do in the 1777 folder: write the status beside the
# receipt's name there (its name is in the listing of the folders).
PLANT_OLD_EXIT = 'printf 0 > "/run/runtime/$BENCHFLOW_VERIFIER_ATTEMPT.exit"\n'


def _root_only(monkeypatch) -> None:
    """Verifiers made by ``_score`` are those of a trial that keeps services."""
    made = Verifier.__init__

    def init(self, *args, **kwargs):
        made(self, *args, **kwargs)
        self.receipt_root_only = True

    monkeypatch.setattr(Verifier, "__init__", init)


async def test_the_receipt_is_in_a_new_folder_only_root_can_write(tmp_path):
    task, paths, env = _setup(tmp_path, "echo 1 > /logs/verifier/reward.txt\nexit 3\n")
    verifier = Verifier(task, paths, env, execution_receipt=True)
    verifier.receipt_root_only = True
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    receipt, service = verifier.execution_receipt
    assert service == "main" and ROOT_ONLY.fullmatch(receipt), receipt
    folder = Path(env._map(str(Path(receipt).parent)))
    assert folder.stat().st_mode & 0o7777 == 0o755
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [Path(receipt).name, Path(receipt).name + ".exit"]
    )
    assert (folder / (Path(receipt).name + ".exit")).read_text() == "3"
    # The 1777 folder holds the attempt's folder and nothing else.
    assert [p.name for p in folder.parent.iterdir()] == [folder.name]
    (made,) = [c for c in env.commands if "mkdir -m 0755" in c]
    assert made == (
        "[ ! -L /run/runtime ] && mkdir -p /run/runtime && [ -O /run/runtime ] "
        f"&& chmod 1777 /run/runtime && mkdir -m 0755 {Path(receipt).parent}"
    )
    assert _RECEIPT_DIR_COMMAND not in env.commands


async def test_a_trial_without_kept_services_has_the_root_only_folder_too(
    tmp_path,
):
    task, paths, env = _setup(tmp_path, "echo 1 > /logs/verifier/reward.txt\n")
    verifier = Verifier(task, paths, env, execution_receipt=True)
    await verifier.verify()
    receipt, _ = verifier.execution_receipt
    assert ROOT_ONLY.fullmatch(receipt), receipt
    assert _RECEIPT_DIR_COMMAND not in env.commands
    assert any(f"mkdir -m 0755 {Path(receipt).parent}" in c for c in env.commands)


async def test_a_status_written_in_the_old_folder_is_not_read(tmp_path, monkeypatch):
    """Tests still running at the deadline score 0 although a status
    "ended 0" sits under the receipt's name in /run/runtime, and although
    they wrote a passing reward first."""
    _root_only(monkeypatch)
    task, paths, env = _setup(
        tmp_path, "echo 1 > /logs/verifier/reward.txt\n" + PLANT_OLD_EXIT + "sleep 60\n"
    )
    rewards, error, diagnostic = await _score(task, paths, env)
    planted = list((tmp_path / "sandbox-run-benchflow").glob("verifier-started-*.exit"))
    assert len(planted) == 1 and planted[0].read_text() == "0"
    assert (rewards, error) == ({"reward": 0.0}, None)
    assert diagnostic is not None and diagnostic.scored is True


async def test_nor_without_kept_services(tmp_path):
    """The same tests in a trial that keeps no services, whose agent's
    processes run beside the verifier in integrity off and audit: the status
    planted in /run/runtime was the receipt's own, and scored them 1.0
    ((c)2); it is not read now."""
    task, paths, env = _setup(
        tmp_path, "echo 1 > /logs/verifier/reward.txt\n" + PLANT_OLD_EXIT + "sleep 60\n"
    )
    rewards, error, diagnostic = await _score(task, paths, env)
    planted = list((tmp_path / "sandbox-run-benchflow").glob("verifier-started-*.exit"))
    assert len(planted) == 1 and planted[0].read_text() == "0"
    assert (rewards, error) == ({"reward": 0.0}, None)
    assert diagnostic is not None and diagnostic.scored is True


async def test_tests_that_ended_behind_a_lost_exec_are_scored_from_the_new_folder(
    tmp_path, monkeypatch, caplog
):
    _root_only(monkeypatch)
    task, paths, env = _setup(
        tmp_path, "echo 1 > /logs/verifier/reward.txt\n", lose_exec=True
    )
    with caplog.at_level("WARNING"):
        rewards, error, diagnostic = await _score(task, paths, env)
    assert (rewards, error, diagnostic) == ({"reward": 1.0}, None, None)
    assert "ended (exit status 0)" in caplog.text
    assert "scoring its outputs" in caplog.text
    (probe,) = [c for c in env.commands if "printf 'ended '" in c]
    assert ROOT_ONLY.search(probe), probe


async def test_a_receipt_folder_that_is_a_link_leaves_startup_unknown(tmp_path, caplog):
    task, paths, env = _setup(tmp_path, "echo 1 > /logs/verifier/reward.txt\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o755)
    (tmp_path / "sandbox-run-benchflow").symlink_to(elsewhere)
    verifier = Verifier(task, paths, env, execution_receipt=True)
    verifier.receipt_root_only = True
    with caplog.at_level("WARNING"):
        result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    assert verifier.execution_receipt is None
    assert "for the verifier start receipt" in caplog.text
    # Refused before anything was made or changed through the link.
    assert list(elsewhere.iterdir()) == []
    assert elsewhere.stat().st_mode & 0o7777 == 0o755


class _Seen(_Planes):
    def __init__(self, *, kept: bool):
        super().__init__()
        self.kept = kept
        self.order = []
        self.made = []

    async def harden_before_verify(self, env, task, sandbox_user, **kwargs):
        if self.kept:
            return await super().harden_before_verify(
                env, task, sandbox_user, integrity="strict"
            )
        kill_record.record_kill_before_grading(sandbox_user, 3)

    def verifier(self, **kwargs):
        made = super().verifier(**kwargs)
        self.made.append(made)
        return made


async def _rollout(tmp_path, planes):
    task = SimpleNamespace(
        name="svc",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(
                timeout_sec=30, type="test-script", reward_range=None
            )
        ),
    )
    paths = SimpleNamespace(verifier_dir=tmp_path / "verifier", rollout_dir=tmp_path)
    _rewards, error, _ = await _verify_rollout(
        object(), task, paths, {}, planes, sandbox_user="agent", workspace="/app"
    )
    assert error is None, error


async def test_the_scoring_path_asks_for_the_root_only_folder_when_services_are_kept(
    tmp_path,
):
    planes = _Seen(kept=True)
    await _rollout(tmp_path, planes)
    (verifier,) = planes.made
    assert verifier.receipt_root_only is True


async def test_the_scoring_path_keeps_run_runtime_without_kept_services(tmp_path):
    planes = _Seen(kept=False)
    await _rollout(tmp_path, planes)
    (verifier,) = planes.made
    assert getattr(verifier, "receipt_root_only", False) is False
