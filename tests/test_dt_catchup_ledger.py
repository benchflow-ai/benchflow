"""Guard DT3-CU1/CU3 at integ 7b4907c8 with narrow evidence-ledger rows."""

from pathlib import Path

import pytest

from tools.dt import compare

DT = Path(__file__).parents[1] / "tools" / "dt"
BINARY_EXPORTS = (
    "mean_reward",
    "mean_reward_scored_only",
    "score_excl_errors_ratio",
    "score_ratio",
    "solve_rates.pass_at_k.1",
    "solve_rates.pass_at_k_scored_only.1",
    "solve_rates.pass_hat_k.1",
    "solve_rates.pass_hat_k_scored_only.1",
    "solve_rates.solve_rate",
    "solve_rates.solve_rate_scored_only",
)
PERCENT_EXPORTS = ("score", "score_excl_errors")
INTERVAL_EXPORTS = (
    "solve_rates.solve_rate_interval",
    "solve_rates.solve_rate_scored_only_interval",
)


def summary_details(reverse=False):
    low, high = ("1.0", "0.0") if reverse else ("0.0", "1.0")
    details = [f"{field} {low} -> {high}" for field in BINARY_EXPORTS]
    low, high = ('"100.0%"', '"0.0%"') if reverse else ('"0.0%"', '"100.0%"')
    details += [f"{field} {low} -> {high}" for field in PERCENT_EXPORTS]
    low, high = "[0.0, 0.7934567085261071]", "[0.20654329147389294, 1.0]"
    if reverse:
        low, high = high, low
    return details + [f"{field} {low} -> {high}" for field in INTERVAL_EXPORTS]


@pytest.mark.parametrize("reverse", [False, True])
def test_flaky_row_explains_every_binary_summary_export(reverse):
    """Guards DT3-CU1: both observed directions, all 14 scoring fields."""
    findings = [
        compare.Finding("r2r-mpc-control/oracle", "summary_change", [1], [detail])
        for detail in summary_details(reverse)
    ]
    verdict = compare.judge(
        findings,
        mode="before-after",
        flaky=compare.load_rows(DT / "divergences/flaky.toml"),
    )
    assert verdict.ok, verdict.failures
    assert len(findings) == 14


@pytest.mark.parametrize(
    "cell,mode,detail",
    [
        ("r2r-mpc-control/nop", "before-after", "mean_reward 0.0 -> 1.0"),
        ("hello/oracle", "before-after", "mean_reward 0.0 -> 1.0"),
        ("r2r-mpc-control/oracle", "cross-engine", "mean_reward 0.0 -> 1.0"),
        ("r2r-mpc-control/oracle", "before-after", "mean_reward 0.0 -> 1.01"),
        ("r2r-mpc-control/oracle", "before-after", "mean_reward 0.0 -> 1.0 extra"),
        ("r2r-mpc-control/oracle", "before-after", "mean_reward_n 1 -> 2"),
        (
            "r2r-mpc-control/oracle",
            "before-after",
            "solve_rates.pass_at_k.2 0.0 -> 1.0",
        ),
    ],
)
def test_flaky_summary_row_does_not_cover_unrelated_changes(cell, mode, detail):
    """Guards DT3-CU1 without accepting arbitrary summary regressions."""
    assert not compare.judge(
        [compare.Finding(cell, "summary_change", [1], [detail])],
        mode=mode,
        flaky=compare.load_rows(DT / "divergences/flaky.toml"),
    ).ok


def d3_entries():
    return [
        entry
        for entry in compare.load_rows(
            DT / "expected-changes/d3-92ceb0a1.toml", "change"
        )
        if entry.get("merge") == "92ceb0a1"
    ]


def test_d3_expected_change_names_only_the_two_confirmed_cells_and_four_fields():
    """Guards DT3-CU3: intended early R401 refusal, not changed grading."""
    entries = d3_entries()
    assert len(entries) == 1
    assert entries[0]["target"] == "D3"
    assert entries[0]["item"].startswith("S1:")
    fields = [
        "counts.n_prompts",
        "prompt.n",
        "prompt.raw_sha256",
        "prompt.stripped_sha256",
    ]
    findings = [
        compare.Finding(f"snapshot-hook/{agent}", "field_change", [1, 2, 3], fields)
        for agent in ("oracle", "nop")
    ]
    verdict = compare.judge(findings, mode="before-after", expected=entries, rounds=3)
    assert verdict.ok, verdict.failures


@pytest.mark.parametrize(
    "cell,cls,detail",
    [
        ("snapshot-hook/oracle", "field_change", "task_digest"),
        ("snapshot-hook/nop", "field_change", "prompt.raw_sha256.extra"),
        ("hello/oracle", "field_change", "prompt.n"),
        ("snapshot-hook/oracle", "reward_change", "None -> 1.0"),
        ("snapshot-hook/nop", "newly_accepted", "refused -> passed"),
    ],
)
def test_d3_prompt_row_does_not_explain_other_fields_or_outcomes(cell, cls, detail):
    """Guards DT3-CU3: the entry cannot excuse a new outcome or reward."""
    assert not compare.judge(
        [compare.Finding(cell, cls, [1], [detail])],
        mode="before-after",
        expected=d3_entries(),
    ).ok
