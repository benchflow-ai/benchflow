"""The loopbench cost-curve: sweep ``{model x loop-strategy}`` and assemble a
pass-rate-vs-tokens matrix with cross-over detection.

This is the money-shot artifact for the loopbench thesis — *can a cheap model
plus loops match an expensive model single-shot at equal token spend?* The
x-axis (per-iteration token capture) already lands in ``result.json``'s loop
block and rolls up into ``summary.json``'s ``loop_summary`` / ``usage_summary``;
this module runs one evaluation job per grid cell, reads each cell's
``summary.json``, and assembles the cells into:

- a **matrix** (rows = models, cols = loop strategies, value = pass-rate @
  token spend), and
- a **cross-over verdict** per non-baseline cell against a chosen baseline cell
  (typically the strong model's single-shot run): does this cell match the
  baseline pass-rate at *less-or-equal* token spend? That boolean is the thesis,
  evaluated per cell.

The pure assembly functions (``expand_sweep_grid``, ``cell_result_from_summary``,
``build_cost_curve_matrix``, ``render_matrix_markdown``) carry no I/O and no live
runs, so the matrix shape and the cross-over logic are fully unit-testable. The
thin :func:`run_sweep` orchestrator is the only part that drives a real
:class:`~benchflow.evaluation.Evaluation`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.loop_strategies import SINGLE_SHOT, parse_loop_strategy_spec

# "single-shot", "", and None all denote the no-loop baseline path. The rollout
# treats loop_strategy=None as single-shot, so we normalize these aliases to
# None before handing them to the config and to "single-shot" for display.
_BASELINE_LOOP_ALIASES = frozenset({"", SINGLE_SHOT})


def _normalize_loop(loop: str | None) -> str | None:
    """Map the baseline aliases ("", "single-shot") to None (the no-loop path)."""
    if loop is None:
        return None
    text = loop.strip()
    return None if text in _BASELINE_LOOP_ALIASES else text


def _canonical_loop_key(loop: str | None) -> str | None:
    """A canonical dedup identity for a loop spec — None for the single-shot path.

    Two specs that mean the *same run* must collapse to one grid cell (one full
    evaluation), else the matrix double-counts an evaluand and burns a redundant
    run. Parsing fills the strategy's default params and ``to_mapping`` is
    order-independent, so "verify-retry:k=3", "verify-retry:k=3,feedback=names"
    (names being the default), and "verify-retry:feedback=names,k=3" all key
    identically — the grid spawns one cell for the three.
    """
    norm = _normalize_loop(loop)
    if norm is None:
        return None
    return json.dumps(parse_loop_strategy_spec(norm).to_mapping(), sort_keys=True)


@dataclass(frozen=True)
class SweepCell:
    """One ``{model x loop}`` point in the sweep grid.

    ``loop`` is normalized: ``None`` means the single-shot (no-loop) path, so the
    cell can be fed straight to ``EvaluationConfig.loop_strategy`` without further
    aliasing.
    """

    model: str
    loop: str | None

    @property
    def loop_label(self) -> str:
        return self.loop or SINGLE_SHOT

    @property
    def job_name(self) -> str:
        """Per-cell job dir name: a readable ``{model, loop}`` prefix plus a short
        deterministic hash of the *exact* ``(model, loop)`` identity.

        The hash is load-bearing for correctness, not decoration: sanitizing the
        prefix is lossy ("/" → "_", ":"/"," → "_", "=" dropped), so distinct
        cells like ``foo/bar`` vs ``foo_bar`` would otherwise share a job dir —
        the second cell would resume into the first's results and the matrix
        would misattribute one model's rollouts to another. The hash makes the
        dir injective; it's stable across reruns so resume still addresses the
        same cell.
        """
        safe_loop = self.loop_label.replace(":", "_").replace(",", "_").replace("=", "")
        safe_model = self.model.replace("/", "_")
        digest = hashlib.sha256(
            f"{self.model}\x00{self.loop_label}".encode()
        ).hexdigest()[:8]
        return f"model={safe_model}__loop={safe_loop}__{digest}"


def expand_sweep_grid(models: list[str], loops: list[str | None]) -> list[SweepCell]:
    """Cartesian product of ``models x loops``, de-duplicated and order-stable.

    Order is model-major (all loops for model A, then model B …) so the rendered
    matrix reads row-by-row. Cells are de-duplicated on their *canonical* loop
    identity (see :func:`_canonical_loop_key`), so equivalent specs — including
    the single-shot aliases and default-vs-explicit params — collapse to one
    cell that keeps the first-seen spelling as its display label.
    """
    seen: set[tuple[str, str | None]] = set()
    cells: list[SweepCell] = []
    for model in models:
        for loop in loops:
            key = (model, _canonical_loop_key(loop))
            if key in seen:
                continue
            seen.add(key)
            cells.append(SweepCell(model=model, loop=_normalize_loop(loop)))
    return cells


@dataclass(frozen=True)
class CellResult:
    """A sweep cell's headline metrics, extracted from its ``summary.json``.

    ``total_tokens`` / ``total_cost_usd`` are ``None`` when the job captured no
    trusted usage telemetry (so the cost axis can be honestly absent rather than
    a misleading zero). ``pass_at_iteration`` / ``mean_tokens_to_converge`` /
    ``fraction_converged`` are ``None``/empty for single-shot cells, which carry
    no ``loop_summary``.
    """

    model: str
    loop: str
    pass_rate: float
    passed: int
    total: int
    total_tokens: int | None
    total_cost_usd: float | None
    mean_tokens_to_converge: float | None
    pass_at_iteration: list[float]
    fraction_converged: float | None
    telemetry_coverage: float | None

    def to_dict(self) -> dict[str, Any]:
        # All fields are scalars/lists, so asdict is an exact, drift-proof mirror.
        return asdict(self)


def cell_result_from_summary(
    model: str, loop: str, summary: dict[str, Any]
) -> CellResult:
    """Extract the matrix-relevant metrics from a written ``summary.json`` dict.

    Mirrors the evaluation writer's layout (``evaluation.py`` summary assembly):

    - ``usage_summary(...)`` is spread **flat**, so ``total_tokens`` /
      ``total_cost_usd`` / ``telemetry_coverage`` are top-level keys (not nested).
    - ``loop_summary(...)`` is spread too but is itself ``{"loop_summary": {...}}``,
      so the convergence block lives at ``summary["loop_summary"]``.
    - ``score_ratio`` is the numeric pass rate. Older summaries only have
      formatted ``score`` strings, so those still fall back to integer counts.

    Token totals are read only when usage telemetry is **complete**
    (``telemetry_coverage == 1.0``); otherwise ``total_tokens`` /
    ``total_cost_usd`` stay ``None``. Partial coverage (``0 < cov < 1``) means
    ``usage_summary`` summed tokens for only *some* trials, so that partial total
    is NOT the cell's full spend — treating it as such could make a cell look
    falsely cheaper and even spuriously cross over. The honest answer for a
    partially- or un-instrumented cell is an undecidable cost axis (``None``),
    not a misleading number.
    """
    loop_block = summary.get("loop_summary") or {}
    coverage = summary.get("telemetry_coverage")
    has_usage = coverage is not None and coverage >= 1.0
    passed = int(summary.get("passed") or 0)
    total = int(summary.get("total") or 0)
    score_ratio = summary.get("score_ratio")
    if isinstance(score_ratio, int | float) and not isinstance(score_ratio, bool):
        pass_rate_value = float(score_ratio)
    else:
        pass_rate_value = passed / total if total else 0.0
    return CellResult(
        model=model,
        loop=loop,
        pass_rate=pass_rate_value,
        passed=passed,
        total=total,
        total_tokens=(summary.get("total_tokens") if has_usage else None),
        total_cost_usd=(summary.get("total_cost_usd") if has_usage else None),
        mean_tokens_to_converge=loop_block.get("mean_tokens_to_converge"),
        pass_at_iteration=list(loop_block.get("pass_at_iteration") or []),
        fraction_converged=loop_block.get("fraction_converged"),
        telemetry_coverage=coverage,
    )


def _cross_over_verdict(
    candidate: CellResult, baseline: CellResult, *, pass_rate_tol: float
) -> dict[str, Any]:
    """Does ``candidate`` match the baseline pass-rate at ≤ its token spend?

    The verdict is split into its two independent axes so a reader can see *why*
    a cell does or doesn't cross over:

    - ``matches_pass_rate`` — candidate pass-rate ≥ baseline − tolerance.
    - ``at_lower_or_equal_cost`` — candidate token spend ≤ baseline's. ``None``
      when either side lacks usage telemetry (the cost axis is undecidable, not
      "free").

    ``crosses_over`` is their conjunction, and is ``None`` (undecidable) whenever
    the cost axis is ``None``.
    """
    matches = candidate.pass_rate >= baseline.pass_rate - pass_rate_tol
    cand_tok = candidate.total_tokens
    base_tok = baseline.total_tokens
    if cand_tok is None or base_tok is None:
        at_lower_cost: bool | None = None
        token_ratio: float | None = None
        crosses: bool | None = None
    else:
        at_lower_cost = cand_tok <= base_tok
        token_ratio = round(cand_tok / base_tok, 4) if base_tok else None
        crosses = bool(matches and at_lower_cost)
    return {
        "model": candidate.model,
        "loop": candidate.loop,
        "pass_rate": candidate.pass_rate,
        "baseline_pass_rate": baseline.pass_rate,
        "matches_pass_rate": matches,
        "total_tokens": cand_tok,
        "baseline_total_tokens": base_tok,
        "token_ratio_vs_baseline": token_ratio,
        "at_lower_or_equal_cost": at_lower_cost,
        "crosses_over": crosses,
    }


def _resolve_baseline_cell(
    cells: list[CellResult], baseline: tuple[str, str] | None
) -> CellResult | None:
    """Find the cell matching ``baseline`` by model + *canonical* loop identity.

    Cells store the first-seen loop spelling, but the grid dedupes on a parsed
    canonical key — so an exact label match would miss a baseline spelled
    differently (``""`` vs ``"single-shot"``, default-vs-explicit params). Match
    on the canonical key instead, mirroring :func:`expand_sweep_grid`.
    """
    if not baseline:
        return None
    base_model, base_loop = baseline
    base_canon = _canonical_loop_key(base_loop)
    return next(
        (
            c
            for c in cells
            if c.model == base_model and _canonical_loop_key(c.loop) == base_canon
        ),
        None,
    )


def build_cost_curve_matrix(
    cells: list[CellResult],
    *,
    baseline: tuple[str, str] | None = None,
    pass_rate_tol: float = 0.0,
) -> dict[str, Any]:
    """Assemble the pass-rate-vs-tokens matrix + per-cell cross-over verdicts.

    ``baseline`` is a ``(model, loop)`` pair naming the reference cell —
    typically the expensive model's single-shot run. Its loop is matched by the
    same **canonical identity** the grid dedupes on, so the baseline resolves
    even when spelled differently from the cell's stored label (``""`` vs
    ``"single-shot"``, explicit-vs-default params, reordered params). When given
    (and resolved), each *other* cell gets a cross-over verdict against it; when
    omitted or unresolved, the matrix carries no ``cross_over`` section.
    """
    # Preserve first-seen order for both axes so the matrix reads grid-stable.
    models: list[str] = []
    loops: list[str] = []
    for c in cells:
        if c.model not in models:
            models.append(c.model)
        if c.loop not in loops:
            loops.append(c.loop)

    matrix: dict[str, Any] = {
        "models": models,
        "loops": loops,
        "cells": [c.to_dict() for c in cells],
    }

    baseline_cell = _resolve_baseline_cell(cells, baseline)
    if baseline_cell is not None:
        matrix["baseline"] = {
            "model": baseline_cell.model,
            "loop": baseline_cell.loop,
        }
        matrix["cross_over"] = [
            _cross_over_verdict(c, baseline_cell, pass_rate_tol=pass_rate_tol)
            for c in cells
            if (c.model, c.loop) != (baseline_cell.model, baseline_cell.loop)
        ]
    return matrix


def _fmt_tokens(tokens: int | None) -> str:
    if tokens is None:
        return "n/a"
    if tokens >= 1000:
        return f"{tokens / 1000:.0f}k"
    return str(tokens)


def render_matrix_markdown(matrix: dict[str, Any]) -> str:
    """Render the matrix as a human-readable markdown table + cross-over notes.

    Each cell shows ``pass-rate @ token-spend`` so the pass-rate/cost trade-off
    is legible at a glance; the cross-over section calls out which cheap-model
    loop cells match the baseline at lower cost (the thesis, per cell).
    """
    cells = {(c["model"], c["loop"]): c for c in matrix.get("cells", [])}
    models = matrix.get("models", [])
    loops = matrix.get("loops", [])

    lines: list[str] = ["# Loop cost-curve: pass-rate @ tokens", ""]
    header = "| model \\ loop | " + " | ".join(loops) + " |"
    sep = "| --- | " + " | ".join("---" for _ in loops) + " |"
    lines += [header, sep]
    for model in models:
        row = [f"`{model}`"]
        for loop in loops:
            cell = cells.get((model, loop))
            if cell is None:
                row.append("—")
            else:
                pr = f"{cell['pass_rate'] * 100:.0f}%"
                row.append(f"{pr} @ {_fmt_tokens(cell['total_tokens'])}")
        lines.append("| " + " | ".join(row) + " |")

    cross_over = matrix.get("cross_over")
    if cross_over:
        base = matrix.get("baseline", {})
        lines += [
            "",
            f"## Cross-over vs baseline `{base.get('model')}` / `{base.get('loop')}`",
            "",
        ]
        for v in cross_over:
            if v["crosses_over"] is True:
                mark = "✅ crosses over"
            elif v["crosses_over"] is False:
                mark = "❌ no"
            else:
                mark = "❓ undecidable (no token telemetry)"
            ratio = v["token_ratio_vs_baseline"]
            ratio_str = f" ({ratio:.2f}x tokens)" if ratio is not None else ""
            lines.append(
                f"- `{v['model']}` / `{v['loop']}`: "
                f"{v['pass_rate'] * 100:.0f}% vs "
                f"{v['baseline_pass_rate'] * 100:.0f}% baseline{ratio_str} — {mark}"
            )
    return "\n".join(lines) + "\n"


async def run_sweep(
    *,
    tasks_dir: str | Path,
    jobs_dir: str | Path,
    base_config: EvaluationConfig,
    models: list[str],
    loops: list[str | None],
    baseline: tuple[str, str] | None = None,
    pass_rate_tol: float = 0.0,
    matrix_prefix: str = "sweep",
) -> dict[str, Any]:
    """Run one evaluation job per ``{model x loop}`` cell, then assemble the matrix.

    Cells run sequentially — each :class:`Evaluation` already parallelizes its
    own tasks (``config.concurrency``), and sequential cells keep the shared
    sandbox-concurrency budget predictable. Each cell gets its **own** jobs_dir
    (``jobs_dir/<cell.job_name>/``), so cells never collide on shared-root files
    — notably ``Evaluation``'s backward-compat ``jobs_dir/summary.json`` copy,
    which N cells writing into one jobs_dir would otherwise clobber to the last
    cell's. The assembled matrix is written as ``{matrix_prefix}-matrix.json`` /
    ``-matrix.md`` under the top ``jobs_dir``, which now holds only the per-cell
    subdirs plus those two artifacts. The matrix dict is also returned.
    """
    tasks_dir = Path(tasks_dir)
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    cells = expand_sweep_grid(models, loops)
    results: list[CellResult] = []
    for cell in cells:
        cfg = replace(base_config, model=cell.model, loop_strategy=cell.loop)
        cell_dir = jobs_dir / cell.job_name
        evaluation = Evaluation(
            tasks_dir=tasks_dir,
            jobs_dir=cell_dir,
            config=cfg,
            job_name=cell.job_name,
        )
        await evaluation.run()
        # Read the per-cell top-level summary.json (Evaluation writes it under
        # this cell's isolated jobs_dir, so no cross-cell clobber).
        summary = json.loads((cell_dir / "summary.json").read_text())
        results.append(cell_result_from_summary(cell.model, cell.loop_label, summary))

    matrix = build_cost_curve_matrix(
        results, baseline=baseline, pass_rate_tol=pass_rate_tol
    )
    (jobs_dir / f"{matrix_prefix}-matrix.json").write_text(json.dumps(matrix, indent=2))
    (jobs_dir / f"{matrix_prefix}-matrix.md").write_text(render_matrix_markdown(matrix))
    return matrix
