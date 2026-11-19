"""bench run refuses --source-path and --source-ref beside a repository target.

Guards the audit fix on branch ``audit-cli-source-path``
(finding CLI-3): a repository target fills the source path and ref itself, so
the same flag given too reached ``eval_run`` twice and the command died with
``TypeError: eval_run() got multiple values for keyword argument
'source_path'``, reported as "a bug in BenchFlow".
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app

TARGET = "benchflow-ai/skillsbench/tasks/econ-detrending-correlation"


@pytest.mark.parametrize(
    "flag,value", [("--source-path", "tasks"), ("--source-ref", "main")]
)
def test_a_repository_target_refuses_a_second_path_or_ref(flag, value):
    result = CliRunner().invoke(app, ["run", TARGET, flag, value, "--dry-run"])

    assert result.exit_code == 2, result.output
    assert not isinstance(result.exception, TypeError)
    assert flag in result.output
    assert "org/repo/path@ref" in result.output
