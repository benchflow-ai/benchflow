"""A verifier's model use is read from the task's own declared configuration.

The task.md checker warns when grading calls a model but ``[verifier]
models = true`` is missing (the script then gets no seat token on a task.md
runtime). What says grading calls a model is what the task declares: a
provider key its verifier env names, ``models = true``, or the judge files
an import recorded. It never guesses from the names of the commands test.sh
runs.
"""

from __future__ import annotations

from pathlib import Path

from benchflow.taskmd._vendor import taskmd as ref


def _task(tmp_path: Path, script: str) -> Path:
    (tmp_path / "verifier").mkdir(parents=True)
    (tmp_path / "verifier" / "test.sh").write_text(script)
    return tmp_path


def _warnings(config: dict, task_dir: Path) -> list[str]:
    diags: list = []
    ref.check_script_models(config, task_dir, diags)
    return [d.message for d in diags if d.level == "warning"]


def test_a_command_name_alone_declares_nothing(tmp_path):
    task = _task(tmp_path, "#!/bin/sh\nuvx grader-cli /tests --judge\ngrader-cli run\n")
    assert ref.script_models({}, task) == []
    assert _warnings({}, task) == []


def test_a_provider_key_in_the_verifier_env_declares_model_use(tmp_path):
    task = _task(tmp_path, "#!/bin/sh\npython judge.py\n")
    config = {"verifier": {"env": {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}}}
    assert ref.script_models(config, task) == ["[verifier] env names OPENAI_API_KEY"]
    (warning,) = _warnings(config, task)
    assert "[verifier] models is not true" in warning


def test_a_host_variable_naming_a_provider_key_counts(tmp_path):
    task = _task(tmp_path, "#!/bin/sh\npython judge.py\n")
    config = {"verifier": {"env": {"JUDGE_KEY": "${ANTHROPIC_API_KEY:-}"}}}
    assert ref.script_models(config, task) == ["[verifier] env names ANTHROPIC_API_KEY"]


def test_recorded_judge_files_declare_model_use(tmp_path):
    task = _task(tmp_path, "#!/bin/sh\n./run-checks\n")
    config = {"import": {"toml": {"model_judges": {"judges": ["judges/quality.toml"]}}}}
    (reason,) = ref.script_models(config, task)
    assert reason.startswith("[import.toml] model_judges records judge files")
    assert "judges/quality.toml" in reason
    assert len(_warnings(config, task)) == 1


def test_an_import_record_without_judges_declares_nothing(tmp_path):
    task = _task(tmp_path, "#!/bin/sh\n./run-checks\n")
    config = {"import": {"toml": {"model_judges": {"judges": []}}}}
    assert ref.script_models(config, task) == []


def test_models_true_is_declared_and_silences_the_warning(tmp_path):
    task = _task(tmp_path, "#!/bin/sh\npython judge.py\n")
    config = {
        "verifier": {"models": True, "env": {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}}
    }
    assert "[verifier] models = true" in ref.script_models(config, task)
    assert _warnings(config, task) == []


def test_a_stage_verifier_declares_its_own_model_use(tmp_path):
    task = _task(tmp_path, "#!/bin/sh\nexit 0\n")
    config = {"stages": {"grade": {"verifier": {"env": {"GEMINI_API_KEY": "x"}}}}}
    assert ref.script_models(config, task) == [
        "[stages.grade.verifier] env names GEMINI_API_KEY"
    ]
