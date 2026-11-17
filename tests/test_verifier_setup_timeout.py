"""The verifier-setup timeout is the run's, then the task's, then 180 s.

Guards the fix against the fixed ``VERIFIER_SETUP_TIMEOUT_SEC = 180``
of sandbox/lockdown.py, which bounded every command that sets up the verifier
(hardening, a separate verifier's runtime). Terminal-Bench 4.1's
cumulative-layout-shift set up its separate verifier in 208 s on a busy
machine and failed with "ExecTimeoutError: Command timed out after 180
seconds". Now ``--verifier-setup-timeout`` wins over the task's
``verifier.setup_timeout_sec``, which wins over 180; config.json records the
value used.
"""

from __future__ import annotations

import ast
import inspect
import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from benchflow.sandbox import lockdown
from benchflow.task.config import TaskConfig
from benchflow.task.document import TaskDocument
from benchflow.task.imports import import_task_config_toml
from tests.test_job_matrix import _bench, _config, _native
from tests.test_job_matrix import project as project


def _task(seconds: float | None = None) -> SimpleNamespace:
    verifier = {} if seconds is None else {"setup_timeout_sec": seconds}
    return SimpleNamespace(config=TaskConfig.model_validate({"verifier": verifier}))


# --- the task setting ---------------------------------------------------------


def test_task_md_and_task_toml_declare_it(tmp_path):
    document = TaskDocument.from_text(
        '---\nschema_version: "1.3"\nverifier:\n  setup_timeout_sec: 600\n---\nDo it.\n',
        path=tmp_path / "task.md",
    )
    assert document.config.verifier.setup_timeout_sec == 600
    imported = import_task_config_toml(
        'version = "1.0"\n[verifier]\nsetup_timeout_sec = 900.5\n', source="legacy"
    )
    assert imported.config.verifier.setup_timeout_sec == 900.5
    assert TaskConfig.model_validate({}).verifier.setup_timeout_sec is None


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan, "soon"])
def test_bad_values_are_refused(bad):
    with pytest.raises(ValidationError):
        TaskConfig.model_validate({"verifier": {"setup_timeout_sec": bad}})


# --- lockdown's budget --------------------------------------------------------


def test_the_default_stays_180():
    assert lockdown.VERIFIER_SETUP_TIMEOUT_SEC == 180
    assert lockdown.verifier_setup_timeout_sec() == 180
    assert lockdown.task_verifier_setup_timeout(_task()) is None
    with lockdown.verifier_setup_timeout_for(_task()):
        assert lockdown.verifier_setup_timeout_sec() == 180


def test_a_task_value_holds_while_its_block_runs():
    with lockdown.verifier_setup_timeout_for(_task(600.2)):
        assert lockdown.verifier_setup_timeout_sec() == 601
    assert lockdown.verifier_setup_timeout_sec() == 180


def test_every_call_site_reads_the_trials_value():
    """No lockdown command takes the fixed constant any more: each reads
    verifier_setup_timeout_sec() (the constant is only its default)."""
    tree = ast.parse(inspect.getsource(lockdown))
    uses = [
        node.lineno
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Name) and node.id == "VERIFIER_SETUP_TIMEOUT_SEC"
    ]
    assert uses == []
    calls = inspect.getsource(lockdown).count("verifier_setup_timeout_sec()")
    assert calls >= 50


@pytest.mark.asyncio
async def test_the_entry_points_apply_the_tasks_value(monkeypatch):
    seen: dict[str, int] = {}

    def record(name):
        async def fake(*args, **kwargs):
            seen[name] = lockdown.verifier_setup_timeout_sec()
            return None

        return fake

    monkeypatch.setattr(lockdown, "_harden_steps", record("harden"))
    monkeypatch.setattr(lockdown, "prepare_runtime_tree", record("runtime_tree"))
    task = _task(900)
    task.name = "t"
    await lockdown.harden_before_verify(
        object(), task, None, "/app", integrity="strict"
    )
    env = SimpleNamespace(supports_runtime_kit=True)
    monkeypatch.setattr(
        "benchflow.sandbox._verifier_runtime.runtime_requested", lambda t: ((), False)
    )
    monkeypatch.setattr(
        "benchflow.sandbox._verifier_runtime.runner_requested", lambda t: False
    )
    await lockdown.prepare_separate_verifier_runtime(env, task, workspace="/app")
    assert seen == {"harden": 900, "runtime_tree": 900}
    assert lockdown.verifier_setup_timeout_sec() == 180


@pytest.mark.asyncio
async def test_the_verify_phase_applies_it(monkeypatch):
    from benchflow.rollout import _setup

    seen = []

    async def hardened(*args, **kwargs):
        seen.append(lockdown.verifier_setup_timeout_sec())
        return None, None, None

    monkeypatch.setattr(_setup, "_verify_hardened", hardened)
    await _setup._verify_rollout(object(), _task(450), None, {}, None)
    await _setup._verify_rollout(object(), _task(), None, {}, None)
    assert seen == [450, 180]


# --- the run flag -------------------------------------------------------------


@pytest.mark.parametrize("layout", ["native", "toml"])
def test_the_flag_wins_over_the_run_config(project, monkeypatch, layout):
    from benchflow.evaluation import Evaluation, EvaluationResult

    seen = []

    async def run(self):
        seen.append(self._config.verifier_setup_timeout)
        return EvaluationResult(
            job_name=self._job_name, config=self._config, total=1, passed=1
        )

    monkeypatch.setattr(Evaluation, "run", run)
    path = (
        _native(project, "verifier_setup_timeout: 77\n")
        if layout == "native"
        else _config(project, verifier_setup_timeout=77)
    )
    assert _bench("run", "--config", str(path)).exit_code == 0
    assert seen[-1] == 77
    result = _bench("run", "--config", str(path), "--verifier-setup-timeout", "900")
    assert result.exit_code == 0, result.output
    assert seen[-1] == 900
    result = _bench("run", "ds-a", "-a", "oracle", "--verifier-setup-timeout", "900")
    assert result.exit_code == 0, result.output
    assert seen[-1] == 900
    assert _bench("run", "ds-a", "-a", "oracle").exit_code == 0
    assert seen[-1] is None
    refused = _bench("run", "ds-a", "-a", "oracle", "--verifier-setup-timeout", "0")
    assert refused.exit_code == 2
