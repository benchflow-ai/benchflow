"""Tests for the loopbench cost-curve sweep (loop_sweep).

Covers the pure assembly functions (grid expansion, summary→cell extraction,
matrix + cross-over assembly, markdown render) and the thin orchestrator with a
mocked Evaluation — no live runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.evaluation import EvaluationConfig
from benchflow.loop_sweep import (
    CellResult,
    SweepCell,
    build_cost_curve_matrix,
    cell_result_from_summary,
    expand_sweep_grid,
    render_matrix_markdown,
    run_sweep,
)

# --------------------------------------------------------------------------- #
# expand_sweep_grid
# --------------------------------------------------------------------------- #


def test_expand_grid_is_model_major_and_normalizes_single_shot():
    cells = expand_sweep_grid(["cheap", "pricey"], ["single-shot", "verify-retry:k=3"])
    assert [(c.model, c.loop) for c in cells] == [
        ("cheap", None),  # single-shot -> None
        ("cheap", "verify-retry:k=3"),
        ("pricey", None),
        ("pricey", "verify-retry:k=3"),
    ]


def test_expand_grid_dedups_single_shot_aliases():
    # "", "single-shot", and None all collapse to the one no-loop cell.
    cells = expand_sweep_grid(["m"], ["single-shot", "", None, "self-review:k=2"])
    assert [(c.model, c.loop) for c in cells] == [
        ("m", None),
        ("m", "self-review:k=2"),
    ]


def test_expand_grid_dedups_equivalent_loop_specs():
    # Same evaluand spelled three ways (implicit vs explicit default, param
    # order) collapses to ONE cell — keeping the first-seen spelling — so the
    # matrix never double-counts or burns a redundant run.
    cells = expand_sweep_grid(
        ["m"],
        [
            "verify-retry:k=3",
            "verify-retry:k=3,feedback=names",  # names is the default
            "verify-retry:feedback=names,k=3",  # reordered params
        ],
    )
    assert [(c.model, c.loop) for c in cells] == [("m", "verify-retry:k=3")]


def test_sweep_cell_labels_and_job_name_are_filesystem_safe():
    cell = SweepCell(model="deepseek/v4-pro", loop="verify-retry:k=3,feedback=names")
    assert cell.loop_label == "verify-retry:k=3,feedback=names"
    # Path-breaking chars are sanitized out of the spec; the only "=" left are
    # the structural "model=" / "__loop=" separators (safe in filenames).
    for bad in ("/", ":", ","):
        assert bad not in cell.job_name
    assert cell.job_name == "model=deepseek_v4-pro__loop=verify-retry_k3_feedbacknames"


def test_single_shot_cell_label():
    assert SweepCell(model="m", loop=None).loop_label == "single-shot"


# --------------------------------------------------------------------------- #
# cell_result_from_summary
# --------------------------------------------------------------------------- #


def _summary(
    *,
    score=0.0,
    passed=0,
    total=0,
    usage=None,
    loop_summary=None,
):
    """Build a summary.json-shaped dict matching the evaluation writer's layout:
    score is a formatted "%"-string, usage_summary fields are spread flat at the
    top level, and the loop block is nested under "loop_summary"."""
    out = {"score": f"{score:.1%}", "passed": passed, "total": total}
    if usage is not None:
        out.update(usage)  # usage_summary(...) is spread flat in summary.json
    if loop_summary is not None:
        out["loop_summary"] = loop_summary
    return out


def test_cell_from_summary_extracts_full_loop_and_usage():
    # The evaluation writer spreads loop_summary(results) into summary.json, so
    # the block sits at summary["loop_summary"] directly.
    summary = _summary(
        score=0.72,
        passed=36,
        total=50,
        usage={
            "total_tokens": 1_200_000,
            "total_cost_usd": 3.4,
            "telemetry_coverage": 1.0,
        },
        loop_summary={
            "mean_tokens_to_converge": 18_000.0,
            "pass_at_iteration": [0.4, 0.6, 0.72],
            "fraction_converged": 0.72,
        },
    )
    cell = cell_result_from_summary("cheap", "verify-retry:k=3", summary)
    assert cell.pass_rate == pytest.approx(0.72)
    assert cell.passed == 36
    assert cell.total == 50
    assert cell.total_tokens == 1_200_000
    assert cell.total_cost_usd == 3.4
    assert cell.mean_tokens_to_converge == 18_000.0
    assert cell.pass_at_iteration == [0.4, 0.6, 0.72]
    assert cell.fraction_converged == 0.72
    assert cell.telemetry_coverage == 1.0


def test_cell_from_summary_no_usage_yields_none_tokens_not_zero():
    # A job with zero telemetry coverage must not masquerade as zero-cost.
    summary = _summary(
        score=0.5,
        passed=1,
        total=2,
        usage={"total_tokens": 0, "total_cost_usd": 0.0, "telemetry_coverage": 0.0},
    )
    cell = cell_result_from_summary("m", "single-shot", summary)
    assert cell.total_tokens is None
    assert cell.total_cost_usd is None
    assert cell.telemetry_coverage == 0.0


def test_cell_from_summary_single_shot_has_empty_loop_fields():
    # single-shot jobs carry no loop_summary block.
    summary = _summary(
        score=0.6,
        passed=3,
        total=5,
        usage={"total_tokens": 500, "total_cost_usd": 0.1, "telemetry_coverage": 1.0},
    )
    cell = cell_result_from_summary("m", "single-shot", summary)
    assert cell.mean_tokens_to_converge is None
    assert cell.pass_at_iteration == []
    assert cell.fraction_converged is None


# --------------------------------------------------------------------------- #
# build_cost_curve_matrix + cross-over
# --------------------------------------------------------------------------- #


def _cell(model, loop, pass_rate, tokens):
    return CellResult(
        model=model,
        loop=loop,
        pass_rate=pass_rate,
        passed=0,
        total=0,
        total_tokens=tokens,
        total_cost_usd=None,
        mean_tokens_to_converge=None,
        pass_at_iteration=[],
        fraction_converged=None,
        telemetry_coverage=(1.0 if tokens is not None else 0.0),
    )


def test_matrix_axes_preserve_first_seen_order():
    cells = [
        _cell("cheap", "single-shot", 0.3, 100),
        _cell("cheap", "verify-retry", 0.7, 300),
        _cell("pricey", "single-shot", 0.7, 2000),
    ]
    matrix = build_cost_curve_matrix(cells)
    assert matrix["models"] == ["cheap", "pricey"]
    assert matrix["loops"] == ["single-shot", "verify-retry"]
    assert "cross_over" not in matrix  # no baseline given


def test_cross_over_true_cheap_loop_matches_baseline_at_lower_cost():
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),  # baseline
        _cell("cheap", "verify-retry", 0.72, 1200),  # matches @ < tokens
    ]
    matrix = build_cost_curve_matrix(cells, baseline=("pricey", "single-shot"))
    [verdict] = matrix["cross_over"]
    assert verdict["model"] == "cheap"
    assert verdict["matches_pass_rate"] is True
    assert verdict["at_lower_or_equal_cost"] is True
    assert verdict["crosses_over"] is True
    assert verdict["token_ratio_vs_baseline"] == pytest.approx(0.48)


def test_cross_over_false_when_pass_rate_below_baseline():
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),
        _cell("cheap", "verify-retry", 0.60, 1200),  # cheaper but worse
    ]
    matrix = build_cost_curve_matrix(cells, baseline=("pricey", "single-shot"))
    [verdict] = matrix["cross_over"]
    assert verdict["matches_pass_rate"] is False
    assert verdict["at_lower_or_equal_cost"] is True
    assert verdict["crosses_over"] is False


def test_cross_over_false_when_more_expensive_even_if_better():
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),
        _cell("cheap", "verify-retry", 0.80, 3000),  # better but pricier
    ]
    matrix = build_cost_curve_matrix(cells, baseline=("pricey", "single-shot"))
    [verdict] = matrix["cross_over"]
    assert verdict["matches_pass_rate"] is True
    assert verdict["at_lower_or_equal_cost"] is False
    assert verdict["crosses_over"] is False


def test_cross_over_undecidable_when_tokens_missing():
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),
        _cell("cheap", "verify-retry", 0.72, None),  # no telemetry
    ]
    matrix = build_cost_curve_matrix(cells, baseline=("pricey", "single-shot"))
    [verdict] = matrix["cross_over"]
    assert verdict["matches_pass_rate"] is True
    assert verdict["at_lower_or_equal_cost"] is None
    assert verdict["crosses_over"] is None
    assert verdict["token_ratio_vs_baseline"] is None


def test_cross_over_tolerance_allows_near_match():
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),
        _cell("cheap", "verify-retry", 0.68, 1000),  # 2pt below
    ]
    matrix = build_cost_curve_matrix(
        cells, baseline=("pricey", "single-shot"), pass_rate_tol=0.05
    )
    [verdict] = matrix["cross_over"]
    assert verdict["matches_pass_rate"] is True  # within 5pt tolerance
    assert verdict["crosses_over"] is True


def test_cross_over_zero_baseline_tokens_guards_ratio_division():
    # Baseline captured 0 tokens: the ratio is undefined (None) but the cost
    # comparison and the verdict stay real booleans (cand 0 <= base 0).
    cells = [
        _cell("pricey", "single-shot", 0.70, 0),
        _cell("cheap", "verify-retry", 0.72, 0),
    ]
    matrix = build_cost_curve_matrix(cells, baseline=("pricey", "single-shot"))
    [verdict] = matrix["cross_over"]
    assert verdict["token_ratio_vs_baseline"] is None  # no divide-by-zero
    assert verdict["at_lower_or_equal_cost"] is True
    assert verdict["crosses_over"] is True


def test_unknown_baseline_omits_cross_over():
    cells = [_cell("m", "single-shot", 0.5, 100)]
    matrix = build_cost_curve_matrix(cells, baseline=("nope", "single-shot"))
    assert "cross_over" not in matrix
    assert "baseline" not in matrix


def test_baseline_single_shot_alias_resolves_by_canonical_identity():
    # Baseline named with the "" alias still resolves to the single-shot cell.
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),
        _cell("cheap", "verify-retry:k=3", 0.72, 1200),
    ]
    matrix = build_cost_curve_matrix(cells, baseline=("pricey", ""))
    assert matrix["baseline"] == {"model": "pricey", "loop": "single-shot"}
    [verdict] = matrix["cross_over"]
    assert verdict["model"] == "cheap"
    assert verdict["crosses_over"] is True


def test_baseline_resolves_explicit_and_reordered_params():
    # Cell stores "verify-retry:k=3"; baseline names it with explicit-default +
    # reordered params — canonical identity still resolves it (Bugbot #720).
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),
        _cell("cheap", "verify-retry:k=3", 0.72, 1200),
    ]
    matrix = build_cost_curve_matrix(
        cells, baseline=("cheap", "verify-retry:feedback=names,k=3")
    )
    assert matrix["baseline"] == {"model": "cheap", "loop": "verify-retry:k=3"}


# --------------------------------------------------------------------------- #
# render_matrix_markdown
# --------------------------------------------------------------------------- #


def test_render_markdown_has_table_and_crossover_marks():
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),
        _cell("cheap", "single-shot", 0.30, 80),
        _cell("cheap", "verify-retry", 0.72, 1200),
    ]
    matrix = build_cost_curve_matrix(cells, baseline=("pricey", "single-shot"))
    md = render_matrix_markdown(matrix)
    assert "| model \\ loop | single-shot | verify-retry |" in md
    assert "`cheap`" in md and "`pricey`" in md
    assert "70% @ 2k" in md  # baseline cell formatting
    assert "72% @ 1k" in md
    assert "✅ crosses over" in md  # cheap/verify-retry crosses over
    assert "Cross-over vs baseline" in md


def test_render_markdown_marks_undecidable_cost():
    cells = [
        _cell("pricey", "single-shot", 0.70, 2500),
        _cell("cheap", "verify-retry", 0.72, None),
    ]
    matrix = build_cost_curve_matrix(cells, baseline=("pricey", "single-shot"))
    md = render_matrix_markdown(matrix)
    assert "n/a" in md  # missing token spend
    assert "undecidable" in md


# --------------------------------------------------------------------------- #
# run_sweep orchestrator (mocked Evaluation)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_sweep_drives_one_job_per_cell_and_writes_matrix(
    tmp_path: Path, monkeypatch
):
    """Each cell runs an Evaluation with the right (model, loop_strategy), and the
    assembled matrix.json/.md land under jobs_dir."""

    def _usage(tok, cost):
        return {
            "total_tokens": tok,
            "total_cost_usd": cost,
            "telemetry_coverage": 1.0,
        }

    # Full 2x2 grid (the orchestrator visits every cell), keyed by
    # (model, loop_label).
    fake_summaries = {
        ("pricey", "single-shot"): _summary(
            score=0.70, passed=35, total=50, usage=_usage(2_500_000, 12.0)
        ),
        ("pricey", "verify-retry:k=3"): _summary(
            score=0.74,
            passed=37,
            total=50,
            usage=_usage(3_000_000, 14.0),
            loop_summary={
                "mean_tokens_to_converge": 40_000.0,
                "pass_at_iteration": [0.6, 0.7, 0.74],
                "fraction_converged": 0.74,
            },
        ),
        ("cheap", "single-shot"): _summary(
            score=0.30, passed=15, total=50, usage=_usage(80_000, 0.2)
        ),
        ("cheap", "verify-retry:k=3"): _summary(
            score=0.72,
            passed=36,
            total=50,
            usage=_usage(1_200_000, 3.0),
            loop_summary={
                "mean_tokens_to_converge": 18_000.0,
                "pass_at_iteration": [0.5, 0.65, 0.72],
                "fraction_converged": 0.72,
            },
        ),
    }
    captured: list[tuple[str | None, str]] = []

    def _full_label(config) -> str:
        # Reconstruct the grid's full spec string from the parsed LoopStrategySpec
        # so it matches the matrix's cell.loop_label ("verify-retry:k=3"), not
        # just the bare strategy name.
        spec = config.loop_strategy
        if spec is None:
            return "single-shot"
        if spec.params.get("k") is not None:
            return f"{spec.name}:k={spec.params['k']}"
        return spec.name

    class FakeEvaluation:
        def __init__(self, *, tasks_dir, jobs_dir, config, job_name):
            self._jobs_dir = Path(jobs_dir)
            self._config = config
            self._job_name = job_name
            # loop_strategy is a LoopStrategySpec | None after __post_init__.
            captured.append((config.model, _full_label(config)))

        async def run(self):
            summary = fake_summaries[(self._config.model, _full_label(self._config))]
            job_dir = self._jobs_dir / self._job_name
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "summary.json").write_text(json.dumps(summary))

    monkeypatch.setattr("benchflow.loop_sweep.Evaluation", FakeEvaluation)

    jobs_dir = tmp_path / "jobs"
    matrix = await run_sweep(
        tasks_dir=tmp_path / "tasks",
        jobs_dir=jobs_dir,
        base_config=EvaluationConfig(concurrency=8),
        models=["pricey", "cheap"],
        loops=["single-shot", "verify-retry:k=3"],
        baseline=("pricey", "single-shot"),
    )

    # All 4 cells drove a job, each with the right (model, loop) config.
    assert captured == [
        ("pricey", "single-shot"),
        ("pricey", "verify-retry:k=3"),
        ("cheap", "single-shot"),
        ("cheap", "verify-retry:k=3"),
    ]

    # Matrix artifacts written and match the returned dict.
    assert (jobs_dir / "sweep-matrix.json").exists()
    assert (jobs_dir / "sweep-matrix.md").exists()
    on_disk = json.loads((jobs_dir / "sweep-matrix.json").read_text())
    assert on_disk == matrix

    # 3 non-baseline verdicts; only cheap/verify-retry crosses over (matches the
    # 70% baseline at < its token spend). cheap/single-shot fails on pass-rate;
    # pricey/verify-retry fails on cost.
    verdicts = {(v["model"], v["loop"]): v for v in matrix["cross_over"]}
    assert verdicts[("cheap", "verify-retry:k=3")]["crosses_over"] is True
    assert verdicts[("cheap", "single-shot")]["crosses_over"] is False
    assert verdicts[("pricey", "verify-retry:k=3")]["crosses_over"] is False
