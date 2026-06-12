"""Direct unit tests for the ``bench agent create`` scaffold templates.

The scaffold templates were previously exercised only indirectly through the
CLI e2e test. These pin the pure-data layer directly: every template renders
with no leftover ``{{NAME}}``/``{{TITLE}}`` token, the generated ``.py`` files
byte-compile, every YAML parses, and the generated job yaml resolves to a
runnable agent/model pair (the AR-1 regression). All templates use plain
``str.replace`` (no ``str.format``), so a stray ``{`` in shell/python bodies
must survive verbatim — these tests would catch a template that broke that.
"""

from __future__ import annotations

import json
import py_compile

import yaml

from benchflow import agent_router_scaffold as scaffold
from benchflow.agent_router import build_scaffold_files
from benchflow.evaluation import DEFAULT_AGENT, DEFAULT_MODEL

_ALL_TEMPLATES = {
    "CONVERTER_TEMPLATE": scaffold.CONVERTER_TEMPLATE,
    "MAIN_TEMPLATE": scaffold.MAIN_TEMPLATE,
    "PARITY_TEST_TEMPLATE": scaffold.PARITY_TEST_TEMPLATE,
    "RUNNER_TEMPLATE": scaffold.RUNNER_TEMPLATE,
    "BENCHMARK_YAML_TEMPLATE": scaffold.BENCHMARK_YAML_TEMPLATE,
    "JOB_YAML_TEMPLATE": scaffold.JOB_YAML_TEMPLATE,
    "README_TEMPLATE": scaffold.README_TEMPLATE,
}


def test_every_template_carries_at_least_one_token() -> None:
    # Guards against a template that silently lost its slug placeholder (which
    # would make the no-orphan-token assertions below vacuous for that file).
    for label, template in _ALL_TEMPLATES.items():
        assert "{{NAME}}" in template or "{{TITLE}}" in template, (
            f"{label} has no NAME/TITLE token to substitute"
        )


def test_rendered_scaffold_has_no_orphan_tokens() -> None:
    files = build_scaffold_files("my-bench")
    for rel, content in files.items():
        assert "{{NAME}}" not in content, f"{rel} kept a NAME token"
        assert "{{TITLE}}" not in content, f"{rel} kept a TITLE token"


def test_rendered_python_files_byte_compile(tmp_path) -> None:
    files = build_scaffold_files("my-bench")
    py_files = [rel for rel in files if rel.endswith(".py")]
    # The full set of generated python modules must be present and compile.
    assert {"benchflow.py", "main.py", "parity_test.py", "run_my_bench.py"} <= set(
        py_files
    )
    for rel in py_files:
        path = tmp_path / rel
        path.write_text(files[rel])
        py_compile.compile(str(path), doraise=True)


def test_rendered_yaml_and_json_parse_with_substituted_slug() -> None:
    files = build_scaffold_files("my-bench")
    descriptor = yaml.safe_load(files["benchmark.yaml"])
    assert descriptor["name"] == "my-bench"
    job = yaml.safe_load(files["my-bench.yaml"])
    assert job["tasks_dir"] == "benchmarks/my-bench/tasks"
    parity = json.loads(files["parity_experiment.json"])
    assert parity["benchmark"] == "my-bench"
    assert parity["status"] == "template"


def test_job_yaml_agent_model_pair_is_runnable() -> None:
    # AR-1 regression: the generated job yaml must name an agent/model pair that
    # effective_model resolves cleanly (an empty model under a non-default agent
    # raises at job-load time before any task runs).
    from benchflow.evaluation import effective_model

    job = yaml.safe_load(build_scaffold_files("my-bench")["my-bench.yaml"])
    assert job["agent"] == DEFAULT_AGENT
    resolved = effective_model(job["agent"], job["model"] or None)
    assert resolved == DEFAULT_MODEL


def test_module_suffix_underscores_the_runner_filename() -> None:
    files = build_scaffold_files("a-b-c")
    assert "run_a_b_c.py" in files
    assert "a-b-c.yaml" in files  # job yaml keeps the hyphenated slug
