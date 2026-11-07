"""Guards score-denominator guidance at PY8 base 363d815b."""

import json

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app


@pytest.mark.parametrize("cause", ["verifier_failure", "reward_out_of_range"])
@pytest.mark.parametrize("json_mode", [False, True])
def test_explain_distinguishes_primary_and_scored_only_metrics(cause, json_mode):
    """Guards 363d815b: unscored final trials remain in primary metrics."""
    result = CliRunner().invoke(
        app, ["explain", cause, *(["--json"] if json_mode else [])]
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)["body"] if json_mode else result.stdout
    assert "primary" in body.lower()
    assert "scored-only" in body
    assert "denominator" in body
