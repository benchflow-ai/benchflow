"""task.md draft 2 packages, run natively by BenchFlow.

A draft 2 package is a folder: ``task.md`` (the instruction, then typed
fenced blocks and a ``toml task`` config block), ``sandbox/``, ``verifier/``
(``test.sh``, ``rubric.json``, ``judge.md``), ``oracle/``, ``controls/``,
and ``evidence/`` (task.md's spec repository, branch
``draft2``). ``TaskMdFormat`` is the built-in ``taskmd`` task format
(``benchflow.task.formats``): ``bench eval run --tasks-dir``, ``bf.Evaluation``,
and ``bench tasks check`` take a draft 2 folder as they take a native one.

- ``benchflow.taskmd.reference``: the vendored reference parser, so a
  package parses exactly as ``tools/taskmd.py`` parses it.
- ``benchflow.taskmd.plan``: what BenchFlow honors, refuses, or records,
  field by field (``SUPPORT``).
- ``benchflow.taskmd.materialize``: the native package BenchFlow runs.
- ``benchflow.taskmd.family``: ``family@1`` instances, one per seed.
- ``benchflow.taskmd.grading`` and ``benchflow.taskmd.judging``: the
  ``taskmd`` verifier strategy's rubric scoring and model judges.

docs/task-authoring-taskmd-v2.md is the guide.
"""

from __future__ import annotations

from benchflow.taskmd.materialize import (
    FORMAT_NAME,
    TaskMdError,
    TaskMdFormat,
    controls,
    is_taskmd_package,
    load_plan,
    package_tree_hash,
    taskmd_metadata,
)
from benchflow.taskmd.reference import package_json, parse_package, reference_check

__all__ = [
    "FORMAT_NAME",
    "TaskMdError",
    "TaskMdFormat",
    "controls",
    "is_taskmd_package",
    "load_plan",
    "package_json",
    "package_tree_hash",
    "parse_package",
    "reference_check",
    "taskmd_metadata",
]
