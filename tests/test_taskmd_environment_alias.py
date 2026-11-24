"""task.md's old ``environment:`` key is read as ``sandbox:`` (0.10.1).

SkillsBench renamed task.md's ``environment:`` front-matter key to
``sandbox:`` in July 2026 (9a1f4dd5), after its ``v1.1`` tag (b63b7b28): all 87
task.md files there use the old key, and 0.10.0 refuses every one ("the
'environment' key was renamed to 'sandbox'"), so ``skillsbench@1.1`` does not
run as published. The task.md loader now reads ``environment:`` as
``sandbox:``; a task.md with both is refused, naming the two keys and the one to
keep; a task.md with ``sandbox:`` gives exactly what it gave before. task.toml's
``[environment]`` table is its own format's name and is untouched.

Rows: A1-alias (an old-key task.md parses as its ``sandbox:`` twin; every
task.md of a SkillsBench v1.1 checkout parses, where one is given), A2-alias
(both keys refused), A3-alias (``sandbox:`` task.md files derive what they did on
0.10.0, and task.toml and ``TaskConfig`` behave as they did; mutants: the alias
applied to every file, at every depth, in task.toml's loader, in the schema).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchflow.task import (
    TaskDocument,
    TaskDocumentParseError,
    render_normalized_task_md,
)
from benchflow.task.config import TaskConfig
from benchflow.task.imports import import_task_config_toml

FIXTURES = Path(__file__).parent / "fixtures" / "taskmd_environment_alias"

# The shape of a SkillsBench v1.1 task.md (adaptive-cruise-control's, trimmed).
V11_FRONT = """schema_version: '1.3'
metadata:
  author_name: Jiachen Li
  difficulty: medium
  category: industrial-physical-systems
verifier:
  type: test-script
  timeout_sec: 1800.0
  service: main
  hardening:
    cleanup_conftests: true
agent:
  timeout_sec: 1800.0
{key}:
  network_mode: public
  build_timeout_sec: 600.0
  os: linux
  cpus: 1
  memory_mb: 2048
  storage_mb: 5120
  gpus: 0
"""
PROMPT = "\n## prompt\n\nDo it.\n"


def _doc(front: str) -> TaskDocument:
    return TaskDocument.from_text(f"---\n{front}---\n{PROMPT}")


def test_an_old_key_task_md_parses_as_its_sandbox_twin():
    """A1-alias: ``environment:`` is read as ``sandbox:``, every key under it."""
    old = _doc(V11_FRONT.format(key="environment"))
    new = _doc(V11_FRONT.format(key="sandbox"))
    assert old.config.model_dump() == new.config.model_dump()
    assert old.frontmatter == new.frontmatter
    assert "environment" not in old.frontmatter
    assert old.config.sandbox.cpus == 1 and old.config.sandbox.memory_mb == 2048
    assert old.config.sandbox.build_timeout_sec == 600.0


def test_an_old_key_task_md_normalizes_to_its_sandbox_twin():
    """Normalized, an old-key task.md is its ``sandbox:`` twin's text, key
    order included: the key is renamed where it stands."""
    front = "schema_version: '1.3'\n{key}:\n  cpus: 2\nagent:\n  timeout_sec: 60.0\n"
    old = render_normalized_task_md(
        f"---\n{front.format(key='environment')}---\n{PROMPT}"
    )
    new = render_normalized_task_md(f"---\n{front.format(key='sandbox')}---\n{PROMPT}")
    assert old == new
    assert old.index("sandbox:") < old.index("agent:")


def test_a_task_md_with_both_keys_is_refused_naming_them():
    """A2-alias: both spellings in one task.md are never merged."""
    front = V11_FRONT.format(key="sandbox") + "environment:\n  cpus: 4\n"
    with pytest.raises(TaskDocumentParseError) as raised:
        _doc(front)
    message = str(raised.value)
    assert "'environment:'" in message and "'sandbox:'" in message, message
    assert "keep only 'sandbox:'" in message, message


def test_a_bad_value_under_the_old_key_names_its_line():
    """An error inside an old-key mapping names the file's own line, as it
    would under ``sandbox:``."""
    front = V11_FRONT.format(key="environment").replace("cpus: 1", "cpus: many")
    with pytest.raises(TaskDocumentParseError) as raised:
        _doc(front)
    line = 1 + front.splitlines().index("  cpus: many") + 1
    assert f"task.md:{line}: sandbox.cpus" in str(raised.value), str(raised.value)


@pytest.mark.parametrize(
    "name", sorted(p.name for p in (FIXTURES / "sandbox").glob("*.md"))
)
def test_a_sandbox_task_md_derives_what_it_did_before(name):
    """A3-alias: what BenchFlow derives from a ``sandbox:`` task.md (its
    front matter after normalization, its config, its instruction) equals what
    0.10.0 derived (``golden.json``, written with 4d4ca9c5's loader)."""
    golden = json.loads((FIXTURES / "golden.json").read_text())[name]
    doc = TaskDocument.from_path(FIXTURES / "sandbox" / name)
    got = {
        "frontmatter": json.loads(json.dumps(doc.frontmatter, default=str)),
        "config": json.loads(doc.config.model_dump_json()),
        "instruction": doc.instruction,
    }
    assert got == golden


TOML_GOLDEN = json.loads((FIXTURES / "golden-toml.json").read_text())


def test_a_task_toml_reads_as_it_did_before():
    """A3-alias: the alias is task.md's; task.toml's ``[environment]`` is
    converted where task.toml is read, into the config 0.10.0 gave
    (``golden-toml.json``, written with 4d4ca9c5's loaders)."""
    text = (FIXTURES / "toml" / "environment-table.toml").read_text()
    golden = TOML_GOLDEN["environment-table.toml"]
    native = TaskConfig.model_validate_toml(text)
    imported = import_task_config_toml(text, source="fixture").config
    assert json.loads(native.model_dump_json()) == golden["model_validate_toml"]
    assert json.loads(imported.model_dump_json()) == golden["import_task_config_toml"]


def test_a_task_toml_with_both_tables_is_refused_as_before():
    """task.toml keeps its own both-spellings message, not task.md's."""
    text = (FIXTURES / "toml" / "both-tables.toml").read_text()
    for load in (
        TaskConfig.model_validate_toml,
        lambda text: import_task_config_toml(text, source="fixture"),
    ):
        with pytest.raises(ValueError) as raised:
            load(text)
        assert str(raised.value) == TOML_GOLDEN["both-tables.toml"]["message"]


def test_a_config_mapping_still_refuses_the_old_key():
    """Python's ``TaskConfig`` still takes only ``sandbox``: the alias is read
    where task.md's front matter is, not in the schema."""
    with pytest.raises(ValidationError) as raised:
        TaskConfig.model_validate({"environment": {"cpus": 2}})
    messages = [error["msg"] for error in raised.value.errors()]
    assert messages == TOML_GOLDEN["config-mapping-environment"]["messages"]


def _v11_tasks() -> list[Path]:
    root = os.environ.get("BENCHFLOW_SKILLSBENCH_V11_TASKS")
    if not root:
        return []
    return sorted(Path(root).glob("*/task.md"))


@pytest.mark.skipif(
    not _v11_tasks(),
    reason="BENCHFLOW_SKILLSBENCH_V11_TASKS names no checkout of skillsbench@b63b7b28's tasks/",
)
def test_every_task_md_of_skillsbench_v11_parses():
    """A1-alias on the published tag: every task.md in skillsbench@b63b7b28's
    tasks/ (87, each with the old key) parses."""
    tasks = _v11_tasks()
    failed = {}
    for path in tasks:
        try:
            TaskDocument.from_path(path)
        except Exception as exc:  # each failure is named below
            failed[path.parent.name] = str(exc)[:300]
    assert not failed, failed
    assert len(tasks) == 87
