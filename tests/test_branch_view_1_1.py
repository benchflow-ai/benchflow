"""benchflow.branch-view 1.1 additions to the branch view and ``bench eval branches``.

- ``inspect``/``compare`` JSON carries ``kind`` +
  ``schema_version``; the branch view only ``schema``. 1.1 adds both
  (``schema`` stays ``benchflow.branch-view/1``, the compatible family).
"""

from __future__ import annotations

import jsonschema

from benchflow.branch_view import (
    BRANCH_VIEW_SCHEMA,
    SCHEMA_VERSION,
    load_branch_view,
)
from tests.test_branch_view import _trial


def test_envelope_has_kind_and_schema_version(tmp_path):
    view = load_branch_view(_trial(tmp_path))
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    assert view["kind"] == "benchflow.branch-view"
    assert view["schema_version"] == SCHEMA_VERSION == "1.1"
    assert view["schema"] == "benchflow.branch-view/1"
    assert list(view)[:3] == ["kind", "schema_version", "schema"]


def test_branches_on_a_missing_path_is_a_usage_error(tmp_path):
    """Inspect and compare exit 2 on a missing path;
    ``bench eval branches`` exited 1."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    result = CliRunner().invoke(app, ["eval", "branches", str(tmp_path / "nope")])
    assert result.exit_code == 2, result.output


# ── costs: token classes and an estimated USD ───────────────────────────
#
# The Tokens column summed
# cache reads with input and output tokens; USD was "-" for
# every subscription (OAuth) run with no way to estimate it; the summary said
# "native-subscription runs report no price" even for the oracle, which calls
# no model. 1.1 adds per-class usage per fork and in the totals, and an
# estimate at list price (``usd_estimate``, with ``pricing`` naming the price
# table), never mixed with the reported ``cost.usd``.

import json  # noqa: E402

import pytest  # noqa: E402

from benchflow.branch_view import build_branch_view  # noqa: E402

USAGE = {
    "n_input_tokens": 10,
    "n_output_tokens": 100,
    "n_cache_read_tokens": 40000,
    "n_cache_creation_tokens": 200,
    "total_tokens": 40310,
}
PRICES = {
    "model": "m",
    "source": "test table",
    "usd_per_token": {
        "input": 3e-6,
        "output": 15e-6,
        "cache_read": 3e-7,
        "cache_creation": 3.75e-6,
    },
}
ONE_CHILD = 10 * 3e-6 + 100 * 15e-6 + 40000 * 3e-7 + 200 * 3.75e-6


def _priced_trial(tmp_path, model="claude-sonnet-4-6"):
    trial = _trial(tmp_path)
    tree = json.loads((trial / "tree.json").read_text())
    for fork in tree["forks"]:
        for child in fork["children"]:
            child["usage"] = dict(USAGE)
    (trial / "tree.json").write_text(json.dumps(tree))
    result = json.loads((trial / "result.json").read_text())
    result["model"] = model
    (trial / "result.json").write_text(json.dumps(result))
    return trial, tree, result


def test_usage_classes_and_estimate_with_given_prices(tmp_path):
    trial, tree, result = _priced_trial(tmp_path)
    view = build_branch_view(
        tree,
        result,
        trial_name=trial.name,
        file_exists=lambda rel: (trial / rel).is_file(),
        prices=PRICES,
    )
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    assert view["pricing"] == PRICES
    child = view["forks"][0]["children"][0]
    assert child["usd_estimate"] == pytest.approx(ONE_CHILD)
    assert child["cost"]["usd"] is None  # the reported price stays unknown
    fork = view["forks"][0]
    assert fork["usage"] == {
        "input": 20,
        "output": 200,
        "cache_read": 80000,
        "cache_creation": 400,
        "total": 80620,
    }
    assert fork["usd_estimate"] == pytest.approx(2 * ONE_CHILD)
    totals = view["totals"]
    assert totals["usage"]["cache_read"] == 5 * 40000
    assert totals["usd_estimate"] == pytest.approx(5 * ONE_CHILD)
    # 4 of the 5 children were scored (n10 is unscored).
    per = totals["per_scored_child"]
    assert per["scored"] == 4
    assert per["usd"] is None
    assert per["usd_estimate"] == pytest.approx(5 * ONE_CHILD / 4)
    assert per["tokens"] == 5 * 40310 // 4


def test_load_branch_view_prices_from_the_bundled_litellm_table(tmp_path):
    pytest.importorskip("litellm")
    trial, _, _ = _priced_trial(tmp_path, model="anthropic/claude-sonnet-4-6")
    view = load_branch_view(trial)
    assert view["pricing"]["model"] == "claude-sonnet-4-6"
    assert "litellm" in view["pricing"]["source"]
    assert view["pricing"]["usd_per_token"]["cache_read"] > 0
    assert view["forks"][0]["children"][0]["usd_estimate"] > 0


def test_unknown_model_has_no_estimate_and_no_model_costs_zero(tmp_path):
    trial, tree, result = _priced_trial(tmp_path, model="no-such-model-xyz")
    view = load_branch_view(trial)
    assert view["pricing"] is None
    assert view["totals"]["usd_estimate"] is None
    # An oracle run calls no model: zero tokens estimate to 0, not unknown.
    for fork in tree["forks"]:
        for child in fork["children"]:
            child["usage"] = dict.fromkeys(USAGE, 0)
    result["model"] = None
    view = build_branch_view(
        tree, result, trial_name=trial.name, file_exists=lambda rel: False
    )
    assert view["totals"]["usd_estimate"] == 0.0
    assert view["forks"][0]["children"][0]["usd_estimate"] == 0.0


def test_cli_shows_token_classes_and_estimate(tmp_path):
    pytest.importorskip("litellm")
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    trial, _, _ = _priced_trial(tmp_path)
    result = CliRunner().invoke(
        app, ["eval", "branches", str(trial)], terminal_width=250
    )
    assert result.exit_code == 0, result.output
    for column in ("In", "Out", "Cache", "USD"):
        assert column in result.output
    assert "~0.0143" in result.output  # the estimate, marked
    assert "cache read 200,000" in result.output
    assert "estimate" in result.output and "list price" in result.output


def test_view_says_when_the_fork_reused_a_checkpoint(tmp_path):
    """A fork that reused an existing image says so."""
    trial = _trial(tmp_path)
    tree = json.loads((trial / "tree.json").read_text())
    tree["forks"][0]["snapshot"]["reused"] = True
    (trial / "tree.json").write_text(json.dumps(tree))
    view = load_branch_view(trial)
    assert view["forks"][0]["snapshot"]["reused"] is True
    assert view["forks"][1]["snapshot"]["reused"] is False


def test_view_names_the_checkpoint_a_trial_started_from(tmp_path):
    """Show which checkpoint a trial started from and
    how much of its conversation the exported prefix carries. 1.1 adds
    ``trial.checkpoint_source`` (from checkpoint_source.json, without the
    snapshot ref, a capability handle)."""
    trial = _trial(tmp_path)
    (trial / "checkpoint_source.json").write_text(
        json.dumps(
            {
                "trial": "task__old",
                "trial_path": "/somewhere/task__old",
                "fork_id": "prompt:1",
                "provider": "daytona",
                "ref": "bf-snap-secret",
                "prefix_events": 5,
                "snapshot_start": {"agent": "installed"},
            }
        )
    )
    view = load_branch_view(trial)
    jsonschema.validate(view, BRANCH_VIEW_SCHEMA)
    assert view["trial"]["checkpoint_source"] == {
        "trial": "task__old",
        "checkpoint": "prompt:1",
        "provider": "daytona",
        "prefix_events": 5,
    }
    assert "bf-snap-secret" not in json.dumps(view)
    assert (
        load_branch_view(_trial(tmp_path / "b"))["trial"]["checkpoint_source"] is None
    )
