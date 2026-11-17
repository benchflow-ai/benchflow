"""The lead's ruling D1: an in-place branch child is never shown as
complete with calls missing.

An in-place child's model proxy is stopped after the child and let go, and
its log is not read into the child's record (its import is): the
child's ``observation.json`` and its cost in the tree hold the agent's own
reports, which can miss calls. Such a child is marked ``usage_incomplete``
with the reason, as is one whose turn ended without its report, and so are
the fork's and the trial's totals. Unit tests against the branch fakes; no
Docker, Daytona or credentials.
"""

from __future__ import annotations

import json
import re

import pytest

from benchflow.branch_lineage import cost_totals
from benchflow.providers.runtime import ProviderRuntime
from benchflow.rollout._cost import INCOMPLETE_UNREPORTED
from tests.test_branch_isolated import IMAGES, IsoRollout
from tests.test_branch_restore_parent import _fork, _rollout


@pytest.fixture(autouse=True)
def _reset():
    IMAGES.clear()
    IsoRollout.all = []


# The note the branch CLI prints under a total that is a floor.
FLOOR_NOTE = (
    "+ = a floor: some calls were not counted (usage_incomplete; each "
    "child's usage_incomplete_reason says why)"
)


class _Proxy:
    """The child's in-sandbox model proxy: stopped after the child."""

    runs_in_sandbox = True
    usage_log_shortfall = None
    trajectory = None

    def __init__(self) -> None:
        self.stopped = False

    async def stop(self, *, retry_later: bool = False) -> None:
        self.stopped = True


def _observations(root) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(root.rglob("children/*/observation.json"))
    ]


async def test_an_in_place_child_whose_proxy_log_is_not_read_is_marked(tmp_path):
    """Fails before: both children's costs and observations read as whole."""
    rollout, _sandbox = _rollout(tmp_path)
    proxies: list[_Proxy] = []
    turn = iter(["proxied", "unreported"])

    async def child(_node):
        # The agent's own report: 100 tokens, $0.01.
        rollout._native_usage_metrics["total_tokens"] = 100
        rollout._native_usage_metrics["cost_usd"] = 0.01
        if next(turn) == "proxied":
            proxies.append(_Proxy())
            rollout._usage_runtime = ProviderRuntime(
                kind="litellm", agent_base_url="http://127.0.0.1:1", server=proxies[-1]
            )
        else:
            rollout._usage_report_missing = True
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    fork = _fork(rollout)
    first, second = (record["cost"] for record in fork["children"])
    assert proxies[0].stopped
    assert first.get("usage_incomplete") is True, first
    from benchflow.rollout._cost import INCOMPLETE_BRANCH_CHILD_PROXY

    assert first["usage_incomplete_reason"] == INCOMPLETE_BRANCH_CHILD_PROXY
    assert first["cost_incomplete"] is True
    assert second["usage_incomplete_reason"] == INCOMPLETE_UNREPORTED
    assert (fork["cost"]["usage_incomplete"], fork["cost"]["cost_incomplete"]) == (
        True,
        True,
    )
    assert cost_totals([fork])["usage_incomplete"] is True

    observed = _observations(tmp_path)
    assert len(observed) == 2
    for observation in observed:
        usage = observation["native_usage"]
        assert usage["usage_incomplete"] is True and usage["cost_incomplete"] is True
        assert usage["usage_incomplete_reason"] in (
            INCOMPLETE_BRANCH_CHILD_PROXY,
            INCOMPLETE_UNREPORTED,
        )


async def _d1_fork(tmp_path, *, proxied: bool = True):
    """D1's fork: two in-place children reporting 60 in and 40 out tokens and
    $0.01 each, the first through an in-sandbox proxy whose log is not read
    (``proxied``), the second with its turn's report missing."""
    rollout, _sandbox = _rollout(tmp_path)
    turn = iter(["proxied", "unreported"])

    async def child(_node):
        rollout._native_usage_metrics.update(
            n_input_tokens=60, n_output_tokens=40, total_tokens=100, cost_usd=0.01
        )
        if next(turn) == "proxied" and proxied:
            rollout._usage_runtime = ProviderRuntime(
                kind="litellm", agent_base_url="http://127.0.0.1:1", server=_Proxy()
            )
        elif proxied:
            rollout._usage_report_missing = True
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    return rollout, _fork(rollout)


def _branch_cli_output(monkeypatch, rollout) -> tuple[str, str]:
    """What ``bench eval branch`` prints after the run (the cost table and
    its total), and what ``bench eval branches`` prints for the trial."""
    import io
    from types import SimpleNamespace

    from rich.console import Console
    from typer.testing import CliRunner

    import benchflow
    from benchflow.branch_run import _fork_outcome
    from benchflow.branch_view import load_branch_view
    from benchflow.cli import branch as cli
    from benchflow.cli.main import app

    wide = Console(width=250, file=io.StringIO(), record=True)
    monkeypatch.setattr(cli, "console", wide)
    # The outcome the run builds from the rollout's forks.
    outcome = SimpleNamespace(task="task")
    _fork_outcome(outcome, rollout)
    cli._print_cost_table([outcome])
    run = wide.export_text()

    view = load_branch_view(rollout._rollout_dir, prices=None)
    monkeypatch.setattr(
        benchflow,
        "load_job",
        lambda _path: SimpleNamespace(branch_views=lambda: [view]),
    )
    monkeypatch.setattr(cli, "console", Console(width=250))
    shown = CliRunner().invoke(
        app, ["eval", "branches", str(rollout._rollout_dir)], terminal_width=250
    )
    assert shown.exit_code == 0, shown.output
    return run, shown.output


async def test_the_branch_cli_shows_the_floor_wherever_a_total_prints(
    tmp_path, monkeypatch
):
    """RUSAGE's MEDIUM on D1: tree.json said the totals were floors, the CLI
    printed them as whole. Every token and USD figure of an incomplete part
    carries ``+``, and the note says why. Fails before: no marker."""
    rollout, fork = await _d1_fork(tmp_path)
    tokens = fork["cost"]["tokens"]
    run, shown = _branch_cli_output(monkeypatch, rollout)

    # bench eval branch: the fork's row and the total.
    assert f"{tokens:,}+" in run, run
    assert f"{fork['cost']['usd']:.4f}+" in run, run
    assert f"Total: {tokens:,}+ tokens" in run, run
    assert FLOOR_NOTE in " ".join(run.split()), run
    # bench eval branches: each child's row, the totals and per scored child.
    assert shown.count("60+") == 2 and shown.count("40+") == 2, shown
    assert shown.count("0.0100+") >= 2, shown
    flat = " ".join(shown.split())
    assert "200+ tokens (in 120" in flat, shown
    assert "USD 0.0200+ reported" in flat, shown
    assert "per scored child (2): 100+ tokens, USD 0.0100+" in flat, shown
    assert FLOOR_NOTE in flat, shown


async def test_the_view_s_per_scored_child_is_marked_a_floor(tmp_path):
    """RUSAGE's nit: the view's per-scored-child figures are floors when the
    totals are; the JSON says so itself."""
    from benchflow.branch_view import load_branch_view

    rollout, _fork_record = await _d1_fork(tmp_path)
    per = load_branch_view(rollout._rollout_dir, prices=None)["totals"][
        "per_scored_child"
    ]
    assert (per["usage_incomplete"], per["cost_incomplete"]) == (True, True)


async def test_the_view_s_per_scored_child_of_a_whole_count_is_unmarked(tmp_path):
    from benchflow.branch_view import load_branch_view

    rollout, _fork_record = await _d1_fork(tmp_path, proxied=False)
    per = load_branch_view(rollout._rollout_dir, prices=None)["totals"][
        "per_scored_child"
    ]
    assert "usage_incomplete" not in per and "cost_incomplete" not in per


async def test_the_branch_cli_shows_no_floor_for_a_whole_count(tmp_path, monkeypatch):
    rollout, fork = await _d1_fork(tmp_path, proxied=False)
    assert "usage_incomplete" not in fork["cost"]
    run, shown = _branch_cli_output(monkeypatch, rollout)
    for text in (run, shown):
        assert not re.search(r"\d\+", text), text
        assert FLOOR_NOTE not in " ".join(text.split())


async def test_a_child_that_reported_every_turn_without_a_proxy_is_not_marked(
    tmp_path,
):
    rollout, _sandbox = _rollout(tmp_path)

    async def child(_node):
        rollout._native_usage_metrics["total_tokens"] = 100
        rollout._native_usage_metrics["cost_usd"] = 0.01
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    fork = _fork(rollout)
    for record in fork["children"]:
        assert "usage_incomplete" not in record["cost"]
    assert "usage_incomplete" not in fork["cost"]
    for observation in _observations(tmp_path):
        assert "usage_incomplete" not in observation["native_usage"]
