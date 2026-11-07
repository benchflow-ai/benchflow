"""PARITY-3 r4 regressions at 74aebf8a2 (review B3)."""

import importlib
import json

import pytest

from tests.test_job_matrix import (
    _bench,
    _config,
    _native,
)
from tests.test_job_matrix import (
    project as project,
)
from tests.test_job_matrix import (
    ran as ran,
)


def test_plain_job_config_accepts_verifier_only_cli_overlay(project, ran):
    """B3-M1: 74aebf8a2 refused B's base-passing verifier-only overlay."""
    from benchflow._utils.config_override import apply_config_override
    from benchflow.task.config import TaskConfig

    result = _bench(
        "run",
        "--config",
        str(_config(project)),
        "--config-override",
        '{"timeout_multipliers":{"verifier":2}}',
    )
    assert result.exit_code == 0, result.output
    overlay = ran[0]["config_override"]
    assert overlay == {"timeout_multipliers": {"build": 1.0, "verifier": 2}}
    task = TaskConfig()
    scaled = apply_config_override(task, overlay)
    assert scaled.verifier.timeout_sec == task.verifier.timeout_sec * 2
    assert scaled.sandbox.start_deadline_sec == task.sandbox.build_timeout_sec


@pytest.mark.parametrize(
    "file_phase,cli_phase", [("verifier", "build"), ("build", "verifier")]
)
def test_disjoint_explicit_phase_overlays_merge(project, ran, file_phase, cli_phase):
    """B3-M1 at 74aebf8a2 refused compatible explicit phase entries too."""
    key = {
        "verifier": "verifier_timeout_multiplier",
        "build": "environment_build_timeout_multiplier",
    }[file_phase]
    path = _config(project, **{key: 2})
    result = _bench(
        "run",
        "--config",
        str(path),
        "--config-override",
        json.dumps({"timeout_multipliers": {cli_phase: 3}}),
    )
    assert result.exit_code == 0, result.output
    assert ran[0]["config_override"] == {
        "timeout_multipliers": {file_phase: 2, cli_phase: 3}
    }


@pytest.mark.parametrize("phase", ["verifier", "build"])
def test_explicit_default_phase_still_conflicts(project, ran, phase):
    """B3-M1 guard: an explicit file default remains a user-supplied phase."""
    key = {
        "verifier": "verifier_timeout_multiplier",
        "build": "environment_build_timeout_multiplier",
    }[phase]
    result = _bench(
        "run",
        "--config",
        str(_config(project, **{key: 1})),
        "--config-override",
        json.dumps({"timeout_multipliers": {phase: 2}}),
    )
    assert result.exit_code == 1, result.output
    assert "both scale the timeouts" in " ".join(result.output.split())
    assert not ran


@pytest.mark.parametrize("default", [{"skip_install": False}, {"harness": "acp"}])
def test_implicit_pool_explicit_extension_default(project, ran, default):
    """B3-L1 at 74aebf8a2: explicit honored defaults split one agent pool."""
    path = _config(
        project,
        n_concurrent_trials=4,
        agents=[
            {"name": "oracle", "n_concurrent": 1},
            {"name": "oracle", "n_concurrent": 2, **default},
        ],
    )
    result = _bench("run", "--config", str(path), "--dry-run")
    assert result.exit_code == 2, result.output
    assert "same agent" in result.output
    # A real extension difference is still a different pool.
    path = _config(
        project,
        agents=[
            {"name": "oracle", "n_concurrent": 1},
            {"name": "oracle", "n_concurrent": 2, "skip_install": True},
        ],
    )
    assert _bench("run", "--config", str(path), "--dry-run").exit_code == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_seed", 0),
        ("freeze_workspace", False),
        ("retry_prompt", "wrong"),
        ("retry_resume_session", False),
    ],
)
def test_omitted_flag_guard_detects_corruption(project, ran, monkeypatch, field, value):
    """B3-L2: B's mutation passed the incomplete guard at 74aebf8a2."""
    from tests.test_job_matrix import (
        test_a_flag_left_out_keeps_the_files_value,
    )

    cli = importlib.import_module("benchflow.cli.main")
    real = cli._config_file_evaluation

    def corrupted(*args, **kwargs):
        job = real(*args, **kwargs)
        setattr(job._config, field, value)
        return job

    monkeypatch.setattr(cli, "_config_file_evaluation", corrupted)
    with pytest.raises(AssertionError):
        test_a_flag_left_out_keeps_the_files_value(project, ran)


@pytest.mark.parametrize("layout", ["native", "toml"])
@pytest.mark.parametrize(
    "field,file_value,cli_value",
    [
        ("sandbox_setup_timeout", 77, 120),
        ("base_image_override", "example.invalid/old:1", "example.invalid/new:2"),
    ],
)
def test_setup_and_image_cli_flags_win(
    project, monkeypatch, layout, field, file_value, cli_value
):
    """B3-P-M1 at 74aebf8a2 silently used the file's timeout/image."""
    from benchflow.evaluation import Evaluation, EvaluationResult

    seen = []

    async def run(self):
        seen.append(getattr(self._config, field))
        return EvaluationResult(
            job_name=self._job_name, config=self._config, total=1, passed=1
        )

    monkeypatch.setattr(Evaluation, "run", run)
    path = (
        _native(project, f"{field}: {file_value}\n")
        if layout == "native"
        else _config(project, **{field: file_value})
    )
    omitted = _bench("run", "--config", str(path))
    assert omitted.exit_code == 0, omitted.output
    assert seen[-1] == file_value
    flag = "--" + field.replace("_", "-")
    explicit = _bench("run", "--config", str(path), flag, str(cli_value))
    assert explicit.exit_code == 0, explicit.output
    assert seen[-1] == cli_value
    folder = _bench("run", "ds-a", "--agent", "oracle", flag, str(cli_value))
    assert folder.exit_code == 0, folder.output
    assert seen[-1] == cli_value
    default = _bench("run", "ds-a", "--agent", "oracle")
    assert default.exit_code == 0, default.output
    assert seen[-1] == (120 if field == "sandbox_setup_timeout" else None)


@pytest.mark.parametrize(
    "key",
    [
        "environment_build_timeout_multiplier",
        "verifier_timeout_multiplier",
        "agent_timeout_multiplier",
        "timeout_multiplier",
    ],
)
def test_json_scientific_multiplier_is_numeric(project, key):
    """B3-P-L1 at 74aebf8a2 parsed valid JSON 5e-05 as a YAML string."""
    from benchflow.evaluation import Evaluation
    from benchflow.job_config import job_plan

    path = _config(project, **{key: 5e-05, "agent_setup_timeout_multiplier": 1})
    assert "5e-05" in path.read_text()
    result = _bench("run", "--config", str(path), "--dry-run")
    assert result.exit_code == 0, result.output
    job = Evaluation.from_yaml(path, preflight=False)
    assert job_plan(path) is not None
    if key in ("agent_timeout_multiplier", "timeout_multiplier"):
        assert job._config.timeout_multiplier == 5e-05
    else:
        phase = "build" if key.startswith("environment") else "verifier"
        assert job._config.config_override["timeout_multipliers"][phase] == 5e-05


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "5e-05"])
def test_invalid_multiplier_remains_refused(project, value):
    """B3-P-L1 guard: JSON parsing keeps the positive, finite numeric rule."""
    result = _bench(
        "run",
        "--config",
        str(_config(project, environment_build_timeout_multiplier=value)),
        "--dry-run",
    )
    assert result.exit_code == 2, result.output
    assert "environment_build_timeout_multiplier" in result.output
