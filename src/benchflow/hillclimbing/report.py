"""The static HTML report of a ``bench hillclimb`` run.

One self-contained page: inline CSS, inline SVG, and a small inline script for
the chart's hover readout. It makes no network request (no fonts, scripts or
images from elsewhere), so it opens from disk and screenshots the same
anywhere. Every string that came from a run (proposals, diffs, task names) is
escaped.

The page reads top to bottom as the post's story: the verdict on the test
split, the score curve over rounds (the accepted version's train and test
scores with 95% bands, and each candidate, filled when kept and hollow when
reverted), the decisions with their deltas, the setup (split, noise gate,
grader checks, infrastructure errors, cost), the stall analysis, and each
candidate's diff.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.hillclimbing import record as rec

# The reference palette (validated: categorical slots 1-2 pass every check in
# light and dark mode against these surfaces).
_CSS = """
.hc { color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --train: #2a78d6; --test: #eb6834;
  --good: #0ca30c; --warn: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --add-bg: rgba(12,163,12,0.10); --del-bg: rgba(208,59,59,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .hc { color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --train: #3987e5; --test: #d95926;
    --add-bg: rgba(12,163,12,0.18); --del-bg: rgba(208,59,59,0.20); }
}
:root[data-theme="dark"] .hc { color-scheme: dark;
  --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --train: #3987e5; --test: #d95926;
  --add-bg: rgba(12,163,12,0.18); --del-bg: rgba(208,59,59,0.20); }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); }
.hc { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink);
  background: var(--page); padding: 32px 24px 48px; line-height: 1.45; }
.hc main { max-width: 1040px; margin: 0 auto; }
.hc h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
.hc h2 { font-size: 16px; font-weight: 600; margin: 32px 0 12px; }
.hc .sub { color: var(--ink-2); font-size: 14px; margin: 0; }
.hc .card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px;
  padding: 20px 24px; }
.hc .verdict { display: grid; grid-template-columns: auto 1fr; gap: 8px 32px;
  align-items: center; margin-top: 20px; }
.hc .hero { font-size: 56px; font-weight: 600; line-height: 1; }
.hc .hero-label { color: var(--ink-2); font-size: 13px; margin-top: 6px; }
.hc .badge { display: inline-flex; align-items: center; gap: 6px; font-size: 13px;
  font-weight: 600; color: var(--ink); }
.hc .badge i { font-style: normal; display: inline-flex; width: 18px; height: 18px;
  border-radius: 9px; align-items: center; justify-content: center; color: #fff;
  font-size: 12px; }
.hc .verdict p { margin: 6px 0 0; font-size: 15px; }
.hc .isolation { margin: 12px 0 0; }
.hc .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; }
.hc .kpi .label { color: var(--ink-2); font-size: 13px; }
.hc .kpi .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.hc .kpi .note { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.hc .legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--ink-2);
  margin-bottom: 8px; }
.hc .legend span { display: inline-flex; align-items: center; gap: 6px; }
.hc .chart { position: relative; }
.hc svg { display: block; width: 100%; height: auto; overflow: visible; }
.hc .tip { position: absolute; pointer-events: none; background: var(--surface);
  border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: none; min-width: 150px; }
.hc .tip .row { display: flex; align-items: center; gap: 6px; }
.hc .tip b { font-variant-numeric: tabular-nums; }
.hc .tip .k { width: 12px; height: 2px; display: inline-block; }
.hc .tip .t { color: var(--ink-2); }
.hc table { width: 100%; border-collapse: collapse; font-size: 13px; }
.hc th { text-align: left; font-weight: 600; color: var(--ink-2); padding: 8px 10px;
  border-bottom: 1px solid var(--axis); white-space: nowrap; }
.hc td { padding: 8px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
.hc td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.hc td .ci { color: var(--muted); display: block; font-size: 12px; }
.hc .reasons { color: var(--ink-2); font-size: 12px; }
.hc details { margin-top: 8px; }
.hc summary { cursor: pointer; font-size: 14px; }
.hc pre.diff { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px;
  padding: 10px 0; font-size: 12px; overflow-x: auto; margin: 8px 0 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.hc pre.diff span { display: block; padding: 0 12px; white-space: pre; }
.hc pre.diff .add { background: var(--add-bg); }
.hc pre.diff .del { background: var(--del-bg); }
.hc pre.diff .hdr { color: var(--muted); }
.hc .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.hc dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0;
  font-size: 13px; }
.hc dt { color: var(--ink-2); }
.hc dd { margin: 0; font-variant-numeric: tabular-nums; }
.hc .warn-list { font-size: 13px; color: var(--ink-2); padding-left: 18px; margin: 0; }
.hc footer { color: var(--muted); font-size: 12px; margin-top: 40px; }
.hc code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
@media (max-width: 720px) { .hc .kpis, .hc .grid2 { grid-template-columns: 1fr 1fr; }
  .hc .verdict { grid-template-columns: 1fr; } }
"""

_SCRIPT = """
(function () {
  var data = JSON.parse(document.getElementById('hc-data').textContent);
  document.querySelectorAll('.chart[data-series]').forEach(function (box) {
    var key = box.getAttribute('data-series');
    var series = data[key];
    if (!series || !series.x.length) return;
    var svg = box.querySelector('svg');
    var tip = box.querySelector('.tip');
    var cross = svg.querySelector('.cross');
    function nearest(evt) {
      var r = svg.getBoundingClientRect();
      var vx = (evt.clientX - r.left) * (series.w / r.width);
      var best = 0, dist = Infinity;
      series.x.forEach(function (x, i) {
        var d = Math.abs(x - vx); if (d < dist) { dist = d; best = i; } });
      return best;
    }
    function row(color, value, label) {
      var d = document.createElement('div'); d.className = 'row';
      var k = document.createElement('span'); k.className = 'k';
      k.style.background = color; d.appendChild(k);
      var b = document.createElement('b'); b.textContent = value; d.appendChild(b);
      var t = document.createElement('span'); t.className = 't'; t.textContent = label;
      d.appendChild(t); return d;
    }
    function show(evt) {
      var i = nearest(evt), p = series.points[i];
      cross.setAttribute('x1', series.x[i]); cross.setAttribute('x2', series.x[i]);
      cross.style.display = 'block';
      tip.textContent = '';
      var h = document.createElement('div'); h.textContent = p.title;
      h.style.fontWeight = '600'; h.style.marginBottom = '4px'; tip.appendChild(h);
      var cs = getComputedStyle(box);
      p.rows.forEach(function (r) {
        tip.appendChild(row(cs.getPropertyValue(r.color), r.value, r.label)); });
      if (p.note) { var n = document.createElement('div'); n.className = 't';
        n.style.marginTop = '4px'; n.textContent = p.note; tip.appendChild(n); }
      var r = svg.getBoundingClientRect();
      var left = series.x[i] * r.width / series.w + 12;
      if (left > r.width - 180) left -= 200;
      tip.style.left = left + 'px'; tip.style.top = '8px'; tip.style.display = 'block';
    }
    function hide() { cross.style.display = 'none'; tip.style.display = 'none'; }
    svg.addEventListener('pointermove', show);
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('focus', function () {
      var r = svg.getBoundingClientRect();
      show({clientX: r.left + series.x[series.x.length - 1] * r.width / series.w}); });
    svg.addEventListener('blur', hide);
  });
})();
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def _ci(ci: rec.IntervalDoc | None, spec: str = "+.3f") -> str:
    return "" if ci is None else f"[{ci.low:{spec}}, {ci.high:{spec}}]"


def _usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}" if value >= 0.1 else f"${value:.4f}"


# ---------------------------------------------------------------------------
# The chart
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    value: float | None
    low: float | None
    high: float | None


@dataclass
class _Cand:
    x_index: int
    train: float | None
    test: float | None
    kept: bool
    label: str
    decision: str


def _nice_range(values: Sequence[float], floor0: bool) -> tuple[float, float, float]:
    lo, hi = min(values), max(values)
    if floor0:
        lo = 0.0
    span = max(hi - lo, 1e-9)
    pad = span * 0.12 if span > 1e-6 else max(abs(hi) * 0.2, 0.05)
    lo, hi = lo - (0 if floor0 else pad), hi + pad
    raw = (hi - lo) / 5
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 0.1
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    return lo, hi, step


def _chart(
    key: str,
    title: str,
    accepted: dict[str, list[_Point]],
    cands: list[_Cand],
    labels: list[str],
    fmt: Callable[[float], str],
    floor0: bool,
    clamp01: bool,
) -> tuple[str, dict[str, Any]]:
    """An SVG step chart of the accepted version per split, with candidates."""
    w, h = 960, 340
    ml, mr, mt, mb = 56, 120, 16, 40
    pw, ph = w - ml - mr, h - mt - mb
    n = len(labels)
    xs = [ml + (pw * i / (n - 1) if n > 1 else pw / 2) for i in range(n)]
    values: list[float] = []
    for pts in accepted.values():
        for p in pts:
            values += [v for v in (p.value, p.low, p.high) if v is not None]
    for c in cands:
        values += [v for v in (c.train, c.test) if v is not None]
    if not values:
        values = [0.0, 1.0]
    lo, hi, step = _nice_range(values, floor0)
    if clamp01:
        lo, hi = max(0.0, lo), min(1.0, hi) if max(values) <= 1 else hi

    def y(v: float) -> float:
        return mt + ph * (1 - (v - lo) / (hi - lo if hi > lo else 1))

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" tabindex="0" '
        f'aria-label="{_e(title)}">'
    ]
    # Grid and y ticks.
    t = lo
    while t <= hi + step / 1000:
        yy = y(t)
        parts.append(
            f'<line x1="{ml}" x2="{ml + pw}" y1="{yy:.1f}" y2="{yy:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{ml - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" '
            f'fill="var(--muted)" style="font-variant-numeric: tabular-nums">'
            f"{_e(fmt(t))}</text>"
        )
        t += step
    parts.append(
        f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
    )
    for i, label in enumerate(labels):
        parts.append(
            f'<text x="{xs[i]:.1f}" y="{mt + ph + 22}" text-anchor="middle" '
            f'font-size="12" fill="var(--muted)">{_e(label)}</text>'
        )
    colors = {"train": "var(--train)", "test": "var(--test)"}
    ends: list[tuple[float, str, str]] = []
    for split, pts in accepted.items():
        color = colors[split]
        # 95% band as a step area (a 10% wash).
        band = [
            (i, p) for i, p in enumerate(pts) if p.low is not None and p.high is not None
        ]
        if band:
            upper, lower = [], []
            for j, (i, p) in enumerate(band):
                x0 = xs[i]
                x1 = xs[band[j + 1][0]] if j + 1 < len(band) else xs[i]
                assert p.high is not None and p.low is not None
                upper += [(x0, y(p.high)), (x1, y(p.high))]
                lower += [(x0, y(p.low)), (x1, y(p.low))]
            poly = " ".join(f"{a:.1f},{b:.1f}" for a, b in upper + lower[::-1])
            parts.append(
                f'<polygon points="{poly}" fill="{color}" fill-opacity="0.10" '
                f'stroke="none"/>'
            )
        line = [(i, p) for i, p in enumerate(pts) if p.value is not None]
        if line:
            d = ""
            for j, (i, p) in enumerate(line):
                assert p.value is not None
                if j == 0:
                    d = f"M{xs[i]:.1f},{y(p.value):.1f}"
                else:
                    d += f" H{xs[i]:.1f} V{y(p.value):.1f}"
            parts.append(
                f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            last_i, last = line[-1]
            assert last.value is not None
            parts.append(
                f'<circle cx="{xs[last_i]:.1f}" cy="{y(last.value):.1f}" r="4" '
                f'fill="{color}" stroke="var(--surface)" stroke-width="2"/>'
            )
            ends.append(
                (y(last.value), "Train" if split == "train" else "Test", fmt(last.value))
            )
    # Candidates: filled when kept, hollow when reverted.
    for c in cands:
        for split, v in (("train", c.train), ("test", c.test)):
            if v is None:
                continue
            color = colors[split]
            fill = color if c.kept else "var(--surface)"
            parts.append(
                f'<circle cx="{xs[c.x_index]:.1f}" cy="{y(v):.1f}" r="5" fill="{fill}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
    # Direct end labels, only when they do not collide.
    if len(ends) == 2 and abs(ends[0][0] - ends[1][0]) < 16:
        ends = []
    for yy, name, value in ends:
        parts.append(
            f'<text x="{ml + pw + 12}" y="{yy + 4:.1f}" font-size="13" fill="var(--ink)">'
            f'{_e(name)} <tspan font-weight="600">{_e(value)}</tspan></text>'
        )
    parts.append(
        f'<line class="cross" x1="0" x2="0" y1="{mt}" y2="{mt + ph}" '
        f'stroke="var(--axis)" stroke-width="1" style="display:none"/>'
    )
    parts.append("</svg>")
    # Hover data: one readout per x position.
    points = []
    for i, label in enumerate(labels):
        rows = []
        for split, pts in accepted.items():
            p = pts[i]
            if p.value is not None:
                rows.append(
                    {
                        "color": "--train" if split == "train" else "--test",
                        "value": fmt(p.value),
                        "label": f"{split}, accepted version",
                    }
                )
        notes = []
        for c in cands:
            if c.x_index != i:
                continue
            for split, v in (("train", c.train), ("test", c.test)):
                if v is not None:
                    rows.append(
                        {
                            "color": "--train" if split == "train" else "--test",
                            "value": fmt(v),
                            "label": f"{split}, {c.label} ({c.decision})",
                        }
                    )
            notes.append(f"{c.label}: {c.decision}")
        points.append(
            {
                "title": "Baseline" if i == 0 else f"Round {label}",
                "rows": rows,
                "note": "; ".join(notes),
            }
        )
    series = {"x": [round(x, 1) for x in xs], "w": w, "points": points}
    body = (
        f'<div class="chart" data-series="{_e(key)}">{"".join(parts)}'
        f'<div class="tip" role="status"></div></div>'
    )
    return body, series


def _legend(show_candidates: bool) -> str:
    items = [
        '<span><svg width="18" height="10"><line x1="1" x2="17" y1="5" y2="5" '
        'stroke="var(--train)" stroke-width="2"/></svg>Train</span>',
        '<span><svg width="18" height="10"><line x1="1" x2="17" y1="5" y2="5" '
        'stroke="var(--test)" stroke-width="2"/></svg>Test</span>',
        '<span><svg width="18" height="10"><rect x="1" y="1" width="16" height="8" '
        'fill="var(--muted)" fill-opacity="0.2"/></svg>95% interval</span>',
    ]
    if show_candidates:
        items += [
            '<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" '
            'fill="var(--muted)" stroke="var(--muted)" stroke-width="2"/></svg>'
            "Candidate kept</span>",
            '<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" '
            'fill="var(--surface)" stroke="var(--muted)" stroke-width="2"/></svg>'
            "Candidate reverted</span>",
        ]
    return f'<div class="legend">{"".join(items)}</div>'


def _series_points(
    doc: rec.HillclimbDoc, pick: Callable[[rec.EvaluationDoc | None, str], Any]
) -> tuple[dict[str, list[_Point]], list[_Cand], list[str]]:
    labels = ["baseline"]
    accepted: dict[str, list[_Point]] = {"train": [], "test": []}
    for split in ("train", "test"):
        est = pick(doc.baseline, split)
        accepted[split].append(_point(est))
    evals: dict[str, rec.EvaluationDoc] = {}
    if doc.baseline:
        evals[doc.baseline.version] = doc.baseline
    cands: list[_Cand] = []
    current = doc.baseline
    for i, rd in enumerate(doc.rounds, start=1):
        labels.append(str(rd.round))
        for c in rd.candidates:
            if c.evaluation is not None:
                cands.append(
                    _Cand(
                        x_index=i,
                        train=_value(pick(c.evaluation, "train")),
                        test=_value(pick(c.evaluation, "test")),
                        kept=c.decision == "keep",
                        label=c.id,
                        decision=c.decision,
                    )
                )
                if c.decision == "keep":
                    current = c.evaluation
        for split in ("train", "test"):
            accepted[split].append(_point(pick(current, split)))
    return accepted, cands, labels


def _point(est: rec.EstimateDoc | None) -> _Point:
    if est is None:
        return _Point(None, None, None)
    return _Point(
        est.value, est.ci.low if est.ci else None, est.ci.high if est.ci else None
    )


def _value(est: rec.EstimateDoc | None) -> float | None:
    return est.value if est else None


def _score_of(ev: rec.EvaluationDoc | None, split: str) -> rec.EstimateDoc | None:
    if ev is None:
        return None
    return (ev.train if split == "train" else ev.test).score


def _cost_of(ev: rec.EvaluationDoc | None, split: str) -> rec.EstimateDoc | None:
    if ev is None:
        return None
    return (ev.train if split == "train" else ev.test).cost


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

_BADGES = {
    "good": ("var(--good)", "✓"),
    "warn": ("var(--warn)", "!"),
    "bad": ("var(--critical)", "✕"),
}


def _badge(kind: str, label: str) -> str:
    color, icon = _BADGES[kind]
    return (
        f'<span class="badge"><i style="background:{color}" aria-hidden="true">'
        f"{icon}</i>{_e(label)}</span>"
    )


def _verdict_section(doc: rec.HillclimbDoc) -> str:
    best = doc.best
    if doc.status == "refused" and doc.noise_gate is not None:
        return (
            '<section class="card verdict"><div><div class="hero">—</div>'
            '<div class="hero-label">No climb</div></div><div>'
            + _badge("bad", "Refused by the noise gate")
            + f"<p>{_e(doc.noise_gate.message)}</p></div></section>"
        )
    if best is None:
        detail = doc.stop.detail if doc.stop else "The run has not finished."
        return (
            '<section class="card verdict"><div><div class="hero">…</div>'
            f'<div class="hero-label">{_e(doc.status)}</div></div><div>'
            + _badge("warn", doc.status.capitalize())
            + f"<p>{_e(detail)}</p></div></section>"
        )
    objective = doc.config.objective
    if objective == "cost" and best.cost_change_vs_baseline and (
        best.cost_change_vs_baseline.test is not None
    ):
        hero = f"{best.cost_change_vs_baseline.test:+.1%}"
        hero_label = "Test cost per trial vs baseline"
    else:
        d = best.test_delta_vs_baseline
        hero = f"{d.value:+.3f}" if d and d.value is not None else "—"
        ci = _ci(d.ci) if d else ""
        hero_label = f"Test score vs baseline {ci}".strip()
    if best.verdict.recommend_merge:
        badge = _badge("good", "Gain exceeds noise")
    elif best.candidate is None:
        badge = _badge("warn", "No patch kept")
    else:
        badge = _badge("warn", "Within noise")
    return (
        f'<section class="card verdict"><div><div class="hero">{_e(hero)}</div>'
        f'<div class="hero-label">{_e(hero_label)}</div></div>'
        f"<div>{badge}<p>{_e(best.verdict.text)}</p></div></section>"
    )


def _kpis(doc: rec.HillclimbDoc) -> str:
    def tile(label: str, value: str, note: str = "") -> str:
        return (
            f'<div class="card kpi"><div class="label">{_e(label)}</div>'
            f'<div class="value">{_e(value)}</div><div class="note">{_e(note)}</div></div>'
        )

    base = doc.baseline
    best = doc.best
    kept = sum(1 for rd in doc.rounds if rd.kept)
    tiles = [
        tile(
            "Baseline test score",
            _num(base.test.score.value) if base else "n/a",
            _ci(base.test.score.ci, ".3f") if base else "",
        ),
        tile(
            "Best test score",
            _num(best.test.value) if best else "n/a",
            (_ci(best.test.ci, ".3f") + f" · {best.version}") if best else "",
        ),
        tile(
            "Rounds",
            f"{kept} kept of {len(doc.rounds)}",
            f"stopped: {doc.stop.reason}" if doc.stop else "running",
        ),
        tile(
            "Cost",
            _usd(doc.cost.total_usd),
            f"agent {_usd(doc.cost.agent_usd)} · proposer {_usd(doc.cost.proposer_usd)}",
        ),
    ]
    return f'<div class="kpis">{"".join(tiles)}</div>'


def _decisions(doc: rec.HillclimbDoc) -> str:
    rows = []
    for rd in doc.rounds:
        for c in rd.candidates:
            kind = {"keep": "good", "revert": "bad", "invalid": "warn"}[c.decision]
            label = {"keep": "Kept", "revert": "Reverted", "invalid": "Invalid"}[
                c.decision
            ]

            def delta(d: rec.DeltaDoc | None) -> str:
                if d is None or d.value is None:
                    return "—"
                return f"{d.value:+.3f}<span class=ci>{_e(_ci(d.ci))}</span>"

            change = c.change or c.root_cause or (c.proposer.error or "")
            rows.append(
                "<tr>"
                f'<td class="num">{rd.round}</td>'
                f"<td><code>{_e(c.id)}</code></td>"
                f"<td>{_e(change)}"
                + (
                    f'<div class="reasons">Root cause: {_e(c.root_cause)}</div>'
                    if c.root_cause and c.change
                    else ""
                )
                + "</td>"
                f'<td class="num">{delta(c.train_delta)}</td>'
                f'<td class="num">{delta(c.test_delta)}</td>'
                f"<td>{_badge(kind, label)}"
                f'<div class="reasons">{_e("; ".join(c.reasons))}</div></td>'
                "</tr>"
            )
    if not rows:
        return '<p class="sub">No round ran.</p>'
    return (
        "<table><thead><tr><th>Round</th><th>Candidate</th><th>Change</th>"
        "<th>Train Δ (95% CI)</th><th>Test Δ (95% CI)</th><th>Decision</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _setup(doc: rec.HillclimbDoc) -> str:
    split = doc.split
    gate = doc.noise_gate
    items = [
        ("Tasks", f"{len(split.train)} train, {len(split.test)} test"),
        (
            "Split",
            f"{split.method}"
            + (f", seed {split.seed}" if split.seed is not None else "")
            + (f", by {split.stratify_by}" if split.stratify_by else ""),
        ),
        ("Agent under test", f"{doc.config.agent} / {doc.config.model or 'default'}"),
        ("Proposer", f"{doc.config.proposer.agent} / {doc.config.proposer.model or 'default'}"),
        ("Surface", ", ".join(f"{s.kind} ({s.name})" for s in doc.config.surfaces)),
        ("Objective", doc.config.objective),
        ("Trials per task", str(doc.config.trials)),
        ("Min gain", f"{doc.config.min_gain:g}"),
    ]
    left = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in items)
    right_items: list[tuple[str, str]] = []
    if gate:
        right_items += [
            (
                "Noise gate",
                ("passed" if gate.passed else "refused")
                + (" (forced)" if gate.forced else ""),
            ),
            ("Train noise (95%)", _num(gate.train.noise_95)),
            ("Test noise (95%)", _num(gate.test.noise_95)),
        ]
    if doc.baseline:
        infra = doc.baseline.train.infra_errors + doc.baseline.test.infra_errors
        right_items.append(("Baseline infra errors", str(infra)))
    controls = doc.controls
    if controls:
        right_items.append(
            (
                "Grader checks",
                "skipped"
                if not controls.ran
                else (
                    f"{len(controls.grader_bugs)} flagged"
                    + (f", {len(controls.excluded)} excluded" if controls.excluded else "")
                ),
            )
        )
    right_items.append(
        (
            "Budget",
            f"{_usd(doc.cost.total_usd)} of "
            + (_usd(doc.cost.max_cost_usd) if doc.cost.max_cost_usd else "no cap"),
        )
    )
    right = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in right_items)
    warnings = ""
    if doc.warnings:
        warnings = (
            '<h2>Warnings</h2><ul class="warn-list">'
            + "".join(f"<li>{_e(w)}</li>" for w in doc.warnings)
            + "</ul>"
        )
    return (
        f'<div class="grid2"><div class="card"><dl>{left}</dl></div>'
        f'<div class="card"><dl>{right}</dl></div></div>{warnings}'
    )


def _analysis(doc: rec.HillclimbDoc) -> str:
    a = doc.analysis
    if a is None:
        return ""
    if a.status != "ok":
        return (
            f"<h2>Stall analysis</h2><p class=sub>The analysis did not finish: "
            f"{_e(a.error)}</p>"
        )
    names = {
        "ambiguous_task": "Ambiguous task",
        "grader_bug": "Grader bug",
        "infrastructure": "Infrastructure",
        "capability_gap": "Capability gap",
    }
    counts = "".join(
        f"<tr><td>{_e(names[k])}</td><td class=num>{v}</td></tr>"
        for k, v in a.counts.items()
    )
    rows = "".join(
        f"<tr><td><code>{_e(f.id)}</code></td><td>{_e(names[f.category])}</td>"
        f"<td>{_e(f.explanation)}</td></tr>"
        for f in a.failures
    )
    recs = "".join(f"<li>{_e(r)}</li>" for r in a.recommendations)
    return (
        "<h2>Stall analysis: remaining train failures by root cause</h2>"
        f'<div class="card"><p>{_e(a.summary)}</p>'
        f'<div class="grid2"><table><thead><tr><th>Cause</th><th>Failures</th></tr>'
        f"</thead><tbody>{counts}</tbody></table>"
        + (f'<ul class="warn-list">{recs}</ul>' if recs else "<div></div>")
        + "</div>"
        + (
            "<details><summary>Every failure</summary><table><thead><tr><th>Trial</th>"
            f"<th>Cause</th><th>Why</th></tr></thead><tbody>{rows}</tbody></table>"
            "</details>"
            if rows
            else ""
        )
        + "</div>"
    )


def _exposures(doc: rec.HillclimbDoc) -> list[tuple[str, rec.ExposureDoc]]:
    out = [
        (c.id, c.proposer.exposure)
        for rd in doc.rounds
        for c in rd.candidates
        if c.proposer.exposure is not None
    ]
    if doc.analysis is not None and doc.analysis.exposure is not None:
        out.append(("analysis", doc.analysis.exposure))
    return out


def _isolation_line(doc: rec.HillclimbDoc) -> str:
    """One sentence on what the optimizer saw of the test split (nothing)."""
    seen = _exposures(doc)
    if not seen:
        return ""
    leaked = sorted(
        {t for _, e in seen for t in e.test_tasks_in_paths + e.test_instructions_in_files}
    )
    n_test = len(doc.split.test)
    sandboxes = f"{len(seen)} optimizer sandbox" + ("es" if len(seen) != 1 else "")
    if leaked:
        return _badge(
            "bad", f"Test material found in {sandboxes}: {', '.join(leaked)}"
        )
    network = (
        "none had network access"
        if all(e.network == "none" for _, e in seen)
        else "some had network access"
    )
    return _badge(
        "good",
        f"Test split never mounted: 0 of {n_test} test tasks in {sandboxes}; "
        f"{network}",
    )


def _exposure_section(doc: rec.HillclimbDoc) -> str:
    seen = _exposures(doc)
    if not seen:
        return ""
    rows = []
    for label, e in seen:
        mounts = ", ".join(
            f"{m.sandbox_path} ({'read-only' if m.read_only else 'editable'}, "
            f"{m.files} files)"
            for m in e.mounts
        )
        test_hits = sorted(set(e.test_tasks_in_paths + e.test_instructions_in_files))
        rows.append(
            "<tr>"
            f"<td><code>{_e(label)}</code></td>"
            f"<td>{_e(mounts)}<span class=ci>{_e(e.manifest)}</span></td>"
            f'<td class="num">{len(e.train_tasks)}</td>'
            f'<td class="num">{len(e.failures)}</td>'
            f"<td>{_badge('bad', ', '.join(test_hits)) if test_hits else _badge('good', f'0 of {e.test_tasks}')}</td>"
            f"<td>{_e(e.network)}</td>"
            "</tr>"
        )
    return (
        "<h2>What the optimizer saw</h2>"
        '<div class="card"><p class="sub">Each optimizer run is a sandboxed rollout. '
        "These folders were uploaded into it and nothing else; the manifest lists "
        "every file with its sha256. Test results reached it only as aggregate "
        "scores with 95% intervals. Each upload was checked for every test task's "
        "name in its paths and instruction text in its files.</p>"
        "<table><thead><tr><th>Run</th><th>Mounted</th><th>Train tasks</th>"
        "<th>Failures shown</th><th>Test tasks mounted</th><th>Network</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _diff_html(diff: str) -> str:
    out = []
    for line in diff.splitlines():
        cls = ""
        if line.startswith(("+++", "---", "@@")):
            cls = "hdr"
        elif line.startswith("+"):
            cls = "add"
        elif line.startswith("-"):
            cls = "del"
        out.append(f'<span class="{cls}">{_e(line) or " "}</span>')
    return f'<pre class="diff">{"".join(out)}</pre>'


def _diffs(doc: rec.HillclimbDoc) -> str:
    blocks = []
    for rd in doc.rounds:
        for c in rd.candidates:
            if not c.diff:
                continue
            stats = (
                f" · {c.diff_stats.files_changed} file(s), +{c.diff_stats.added} "
                f"−{c.diff_stats.removed}"
                if c.diff_stats
                else ""
            )
            opened = " open" if c.decision == "keep" else ""
            rationale = (
                f'<p class="reasons">{_e(c.rationale)}</p>' if c.rationale else ""
            )
            blocks.append(
                f"<details{opened}><summary><code>{_e(c.id)}</code> "
                f"{_e(c.decision)}: {_e(c.change or '')}{_e(stats)}</summary>"
                f"{rationale}{_diff_html(c.diff)}"
                + ("<p class=reasons>(diff truncated)</p>" if c.diff_truncated else "")
                + "</details>"
            )
    if not blocks:
        return ""
    return "<h2>Diffs</h2>" + "".join(blocks)


def render(doc: rec.HillclimbDoc, run_name: str = "") -> str:
    """The report page for one run."""
    data: dict[str, Any] = {}
    charts = []
    fmt_score: Callable[[float], str] = lambda v: f"{v:.2f}"  # noqa: E731
    accepted, cands, labels = _series_points(doc, _score_of)
    body, series = _chart(
        "score",
        "Score by round: the accepted version's train and test score",
        accepted,
        cands,
        labels,
        fmt_score,
        floor0=False,
        clamp01=True,
    )
    data["score"] = series
    charts.append(
        "<h2>Score by round</h2>"
        '<div class="card">'
        + _legend(bool(cands))
        + body
        + '<p class="sub">Lines: the accepted version (a step when a patch is kept). '
        "Dots: each round's candidates. Scores are mean rewards over tasks.</p></div>"
    )
    if doc.config.objective == "cost":
        accepted_c, cands_c, labels_c = _series_points(doc, _cost_of)
        if any(p.value is not None for p in accepted_c["train"]):
            body_c, series_c = _chart(
                "cost",
                "Cost per trial by round",
                accepted_c,
                cands_c,
                labels_c,
                lambda v: f"${v:.3f}",
                floor0=True,
                clamp01=False,
            )
            data["cost"] = series_c
            charts.insert(
                0,
                "<h2>Cost per trial by round</h2>"
                '<div class="card">'
                + _legend(bool(cands_c))
                + body_c
                + '<p class="sub">USD per scored trial of the agent under test.</p></div>',
            )
    subtitle = (
        f"{doc.config.agent} / {doc.config.model or 'default model'} · "
        f"{len(doc.split.train)} train and {len(doc.split.test)} test tasks · "
        f"{doc.config.trials} trials per task · objective: {doc.config.objective}"
    )
    title = f"Hill-climb{': ' + run_name if run_name else ''}"
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head>"
        '<body><div class="hc"><main>'
        f"<h1>{_e(title)}</h1><p class=sub>{_e(subtitle)}</p>"
        + _verdict_section(doc)
        + (
            f'<p class="isolation">{_isolation_line(doc)}</p>'
            if _exposures(doc)
            else ""
        )
        + _kpis(doc)
        + "".join(charts)
        + "<h2>Decisions</h2>"
        + f'<div class="card">{_decisions(doc)}</div>'
        + "<h2>Setup</h2>"
        + _setup(doc)
        + _exposure_section(doc)
        + _analysis(doc)
        + _diffs(doc)
        + "<footer>Generated by <code>bench hillclimb</code> (BenchFlow "
        + _e(doc.benchflow_version)
        + ") from <code>hillclimb.json</code>; updated "
        + _e(doc.updated_at)
        + ". The loop follows “Automating eval design and hillclimbing with "
        "Claude” (claude.dev, 2026-09-28); here the test split is kept from the "
        "optimizer by the runtime.</footer>"
        + "</main></div>"
        + '<script type="application/json" id="hc-data">'
        + json.dumps(data).replace("</", "<\\/")
        + "</script>"
        + f"<script>{_SCRIPT}</script></body></html>\n"
    )


def write_report(doc: rec.HillclimbDoc, path: Path) -> Path:
    path.write_text(render(doc, path.parent.name))
    return path
