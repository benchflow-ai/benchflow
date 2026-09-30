"""The hill-climb demo's report: one static HTML page from hillclimb.json.

Inline CSS and SVG and a few lines of script for the chart's hover readout;
no network requests, so it opens from disk and screenshots the same anywhere.
Every string from the run is escaped. Colors are a validated two-series
palette (train blue, test orange) with a dark-mode variant.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
.hc{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.1);--train:#2a78d6;--test:#eb6834;--good:#0ca30c;
--warn:#fab219;--bad:#d03b3b;--add:rgba(12,163,12,.1);--del:rgba(208,59,59,.1)}
@media (prefers-color-scheme:dark){.hc{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.1);--train:#3987e5;--test:#d95926;
--add:rgba(12,163,12,.18);--del:rgba(208,59,59,.2)}}
body{margin:0}.hc{font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
background:var(--page);padding:32px 24px 48px}.hc main{max-width:1040px;margin:0 auto}
.hc h1{font-size:22px;margin:0 0 4px}.hc h2{font-size:16px;margin:32px 0 12px}.hc .sub{color:var(--ink2);margin:0}
.hc .card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:20px 24px}
.hc .verdict{display:grid;grid-template-columns:auto 1fr;gap:8px 32px;align-items:center;margin-top:20px}
.hc .hero{font-size:56px;font-weight:600;line-height:1}.hc .hero-label{color:var(--ink2);font-size:13px}
.hc .badge{display:inline-flex;gap:6px;align-items:center;font-weight:600;font-size:13px}
.hc .badge i{font-style:normal;width:18px;height:18px;border-radius:9px;color:#fff;font-size:12px;
display:inline-flex;align-items:center;justify-content:center}
.hc .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px}
.hc .kpi .v{font-size:24px;font-weight:600}.hc .kpi .n{color:var(--muted);font-size:12px}
.hc .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin-bottom:8px}
.hc .legend span{display:inline-flex;gap:6px;align-items:center}.hc .legend svg{flex:none}
.hc .chart{position:relative}.hc .chart>svg{display:block;width:100%;height:auto}
.hc .tip{position:absolute;display:none;pointer-events:none;background:var(--surface);border:1px solid var(--ring);
border-radius:8px;padding:8px 10px;font-size:12px;min-width:150px}
.hc table{width:100%;border-collapse:collapse;font-size:13px}.hc th{text-align:left;color:var(--ink2);
padding:8px 10px;border-bottom:1px solid var(--axis)}.hc td{padding:8px 10px;border-bottom:1px solid var(--grid);
vertical-align:top}.hc .num{font-variant-numeric:tabular-nums;white-space:nowrap}
.hc .ci,.hc .why{display:block;color:var(--muted);font-size:12px}
.hc pre{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:10px 0;font-size:12px;
overflow-x:auto}.hc pre span{display:block;padding:0 12px;white-space:pre}.hc .add{background:var(--add)}
.hc .del{background:var(--del)}.hc footer{color:var(--muted);font-size:12px;margin-top:40px}
"""

SCRIPT = """
const d=JSON.parse(document.getElementById('hc-data').textContent),box=document.querySelector('.chart');
if(box){const svg=box.querySelector('svg'),tip=box.querySelector('.tip'),x=svg.querySelector('.cross');
svg.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect(),vx=(e.clientX-r.left)*d.w/r.width;
let i=0;d.x.forEach((v,j)=>{if(Math.abs(v-vx)<Math.abs(d.x[i]-vx))i=j});x.setAttribute('x1',d.x[i]);
x.setAttribute('x2',d.x[i]);x.style.display='block';tip.textContent='';d.rows[i].forEach(t=>{
const p=document.createElement('div');p.textContent=t;tip.appendChild(p)});tip.style.display='block';
tip.style.left=Math.min(d.x[i]*r.width/d.w+12,r.width-200)+'px';tip.style.top='8px'});
svg.addEventListener('pointerleave',()=>{x.style.display='none';tip.style.display='none'})}
"""


def e(value) -> str:
    return html.escape("" if value is None else str(value))


def num(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def ci(interval, spec: str = "+.3f") -> str:
    return "" if not interval else f"[{interval[0]:{spec}}, {interval[1]:{spec}}]"


def badge(kind: str, label: str) -> str:
    color, icon = {
        "good": ("var(--good)", "✓"),
        "warn": ("var(--warn)", "!"),
        "bad": ("var(--bad)", "✕"),
    }[kind]
    return (
        f'<span class="badge"><i style="background:{color}">{icon}</i>{e(label)}</span>'
    )


def chart(doc: dict) -> tuple[str, dict]:
    """The accepted version's train and test score by round, as step lines with
    95% bands; each round's candidate as dots, filled if kept, hollow if not."""
    points = [("baseline", doc["baseline"], None)]
    for entry in doc["rounds"]:
        points.append((str(entry["round"]), None, entry))
    w, h, ml, mr, mt, mb = 960, 320, 56, 40, 16, 40
    n = len(points)
    xs = [ml + (w - ml - mr) * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]
    accepted = {s: [] for s in ("train", "test")}
    for _, base, entry in points:
        for s in accepted:
            accepted[s].append(base[s]["score"] if base else entry[f"{s}_after"])
    y = lambda v: mt + (h - mt - mb) * (1 - v)  # noqa: E731 - scores are in [0, 1]
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="score by round">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line x1="{ml}" x2="{w - mr}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="var(--grid)"/>'
            f'<text x="{ml - 8}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="12" '
            f'fill="var(--muted)">{t:.2f}</text>'
        )
    for i, (label, _, _) in enumerate(points):
        parts.append(
            f'<text x="{xs[i]:.1f}" y="{h - 16}" text-anchor="middle" font-size="12" '
            f'fill="var(--muted)">{e(label)}</text>'
        )
    for s, color in (("train", "var(--train)"), ("test", "var(--test)")):
        for i, entry in enumerate(p[2] for p in points):
            if entry and entry.get("evaluation"):
                v = entry["evaluation"][s]["score"]["value"]
                fill = color if entry["decision"] == "keep" else "var(--surface)"
                if v is not None:
                    parts.append(
                        f'<circle cx="{xs[i]:.1f}" cy="{y(v):.1f}" r="5" fill="{fill}" '
                        f'stroke="{color}" stroke-width="2"/>'
                    )
    for s, color in (("train", "var(--train)"), ("test", "var(--test)")):
        est = accepted[s]
        band = [(i, p["ci"]) for i, p in enumerate(est) if p and p.get("ci")]
        if (
            band
        ):  # the 95% interval as a step area: each value holds until the next round
            steps = [
                f"{xs[i]:.1f},{y(c[1]):.1f} {xs[min(i + 1, n - 1)]:.1f},{y(c[1]):.1f}"
                for i, c in band
            ]
            lows = [
                f"{xs[min(i + 1, n - 1)]:.1f},{y(c[0]):.1f} {xs[i]:.1f},{y(c[0]):.1f}"
                for i, c in reversed(band)
            ]
            parts.append(
                f'<polygon points="{" ".join(steps + lows)}" fill="{color}" fill-opacity=".1"/>'
            )
        line = [
            (i, p["value"])
            for i, p in enumerate(est)
            if p and p.get("value") is not None
        ]
        if line:
            d = f"M{xs[line[0][0]]:.1f},{y(line[0][1]):.1f}" + "".join(
                f" H{xs[i]:.1f} V{y(v):.1f}" for i, v in line[1:]
            )
            parts.append(
                f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>'
                f'<circle cx="{xs[line[-1][0]]:.1f}" cy="{y(line[-1][1]):.1f}" r="4" '
                f'fill="{color}" stroke="var(--surface)" stroke-width="2"/>'
            )
    parts.append(
        f'<line class="cross" x1="0" x2="0" y1="{mt}" y2="{h - mb}" stroke="var(--axis)" '
        'style="display:none"/></svg>'
    )
    rows = []
    for i, (label, _, entry) in enumerate(points):
        lines = [("Baseline" if i == 0 else f"Round {label}") + ":"]
        lines += [
            f"{s} {num((accepted[s][i] or {}).get('value'), '.2f')} (accepted)"
            for s in accepted
        ]
        if entry:
            lines.append(f"{entry['candidate']['id']}: {entry['decision']}")
        rows.append(lines)
    return "".join(parts), {"x": [round(x, 1) for x in xs], "w": w, "rows": rows}


def verdict(doc: dict) -> str:
    best, gate = doc.get("best"), doc.get("noise_gate") or {}
    if doc["status"] == "refused":
        return (
            f'<section class="card verdict"><div class="hero">—</div><div>'
            f"{badge('bad', 'Refused by the noise gate')}<p>{e(gate.get('message'))}</p></div></section>"
        )
    if not best:
        return f'<section class="card verdict"><div class="hero">…</div><div>{e(doc["status"])}</div></section>'
    d, v = best["test_delta"], best["verdict"]
    kind, label = (
        ("good", "Gain exceeds noise")
        if v["recommend_merge"]
        else ("warn", "Ungated run")
        if v["exceeds_noise"]
        else ("warn", "Within noise")
    )
    return (
        f'<section class="card verdict"><div><div class="hero">{num(d["value"], "+.3f")}</div>'
        f'<div class="hero-label">Test score vs baseline {e(ci(d["ci"]))}</div></div>'
        f"<div>{badge(kind, label)}<p>{e(v['text'])}</p></div></section>"
    )


def isolation(doc: dict) -> str:
    """One line: how much of the test split the optimizer's sandboxes held (none)."""
    seen = [
        r["candidate"]["mounted"]
        for r in doc["rounds"]
        if r["candidate"].get("mounted")
    ]
    seen += (
        [doc["analysis"]["mounted"]]
        if doc.get("analysis") and doc["analysis"].get("mounted")
        else []
    )
    if not seen:
        return ""
    leaked = sorted(
        {
            t
            for m in seen
            for t in m["test_tasks_in_paths"] + m["test_instructions_in_files"]
        }
    )
    if leaked:
        return f"<p>{badge('bad', 'Test material found in an optimizer sandbox: ' + ', '.join(leaked))}</p>"
    net = (
        "none had network access"
        if all(m["network"] == "none" for m in seen)
        else "some had network access"
    )
    label = f"Test split never mounted: 0 of {seen[0]['test_tasks']} test tasks in {len(seen)} optimizer sandbox(es); {net}"
    return f"<p>{badge('good', label)}</p>"


def kpis(doc: dict) -> str:
    base, best, cost = doc.get("baseline"), doc.get("best"), doc["cost"]
    kept = sum(r["decision"] == "keep" for r in doc["rounds"])
    tiles = [
        (
            "Baseline test score",
            num(base["test"]["score"]["value"]) if base else "n/a",
            ci(base["test"]["score"]["ci"], ".3f") if base else "",
        ),
        (
            "Best test score",
            num(best["test"]["value"]) if best else "n/a",
            f"{ci(best['test']['ci'], '.3f')} · {best['version']}" if best else "",
        ),
        (
            "Rounds",
            f"{kept} kept of {len(doc['rounds'])}",
            f"stopped: {(doc.get('stop') or {}).get('reason', '')}",
        ),
        (
            "Cost",
            f"${cost.get('total_usd', 0):,.2f}"
            if cost.get("source", "unknown") != "unknown"
            else "unknown",
            f"agent ${cost.get('agent_usd', 0):,.2f} · optimizer ${cost.get('proposer_usd', 0):,.2f}",
        ),
    ]
    return (
        '<div class="kpis">'
        + "".join(
            f'<div class="card kpi"><div class="sub">{e(a)}</div><div class="v">{e(b)}</div><div class="n">{e(c)}</div></div>'
            for a, b, c in tiles
        )
        + "</div>"
    )


SOURCES = {
    "claude-code-cost-state": "Claude Code session logs",
    "claude-code-usage-at-list-price": "session-log tokens at list prices",
    "benchflow": "the BenchFlow model proxy",
    "unknown": "unknown",
}


def spending(doc: dict) -> str:
    """Where the USD came from, and the caps that bind without it."""
    cost = doc["cost"]
    if "sources" not in cost:
        return ""
    sources = ", ".join(
        f"{SOURCES.get(k, k)} ({n} rollouts)" for k, n in cost["sources"].items()
    )
    rollouts, cap = cost["rollouts"], cost.get("max_rollouts")
    hours, hours_cap = cost["sandbox_seconds"] / 3600, cost.get("max_sandbox_seconds")
    parts = [
        f"Cost from {sources or 'no rollout yet'}"
        + (". The 1M-context model variant ran." if cost.get("context_1m") else "."),
        f"Spent {rollouts:,} model rollouts"
        + (f" of a {cap:,} cap" if cap else "")
        + f" and {hours:,.1f} sandbox-hours"
        + (f" of a {hours_cap / 3600:,.1f} cap." if hours_cap else "."),
    ]
    return (
        '<div style="margin-top:12px">'
        + "".join(f'<p class="sub">{e(p)}</p>' for p in parts)
        + "</div>"
    )


def decisions(doc: dict) -> str:
    rows = []
    for r in doc["rounds"]:
        c = r["candidate"]
        kind = {"keep": "good", "revert": "bad"}.get(r["decision"], "warn")

        def delta(d):
            return (
                "—"
                if not d or d["value"] is None
                else f'{d["value"]:+.3f}<span class="ci">{e(ci(d["ci"]))}</span>'
            )

        rows.append(
            f'<tr><td class="num">{r["round"]}</td><td>{e(c.get("change") or c.get("error"))}'
            f'<span class="why">{e(c.get("root_cause"))}</span></td><td class="num">{delta(r.get("train_delta"))}</td>'
            f'<td class="num">{delta(r.get("test_delta"))}</td><td>{badge(kind, r["decision"].capitalize())}'
            f'<span class="why">{e("; ".join(r["reasons"]))}</span></td></tr>'
        )
    return (
        (
            "<table><tr><th>Round</th><th>Change</th><th>Train Δ (95% CI)</th><th>Test Δ (95% CI)</th>"
            f"<th>Decision</th></tr>{''.join(rows)}</table>"
        )
        if rows
        else '<p class="sub">No round ran.</p>'
    )


def mounts(doc: dict) -> str:
    runs = [
        (r["candidate"]["id"], r["candidate"].get("mounted")) for r in doc["rounds"]
    ]
    runs += (
        [("analysis", doc["analysis"].get("mounted"))] if doc.get("analysis") else []
    )
    rows = "".join(
        f"<tr><td>{e(run)}</td><td>{e(', '.join(f'{x["sandbox_path"]} ({"read-only" if x["read_only"] else "editable"}, {x["files"]} files)' for x in m['mounts']))}"
        f'<span class="ci">{e(m["manifest"])}</span></td><td class="num">{len(m["train_tasks"])}</td>'
        f'<td class="num">{len(m["failures"])}</td><td>{badge("bad" if m["test_tasks_in_paths"] or m["test_instructions_in_files"] else "good", str(len(set(m["test_tasks_in_paths"] + m["test_instructions_in_files"]))) + " of " + str(m["test_tasks"]))}</td>'
        f"<td>{e(m['network'])}</td></tr>"
        for run, m in runs
        if m
    )
    return (
        (
            "<h2>What the optimizer saw</h2><div class=card><p class=sub>Each optimizer run is a sandboxed rollout. "
            "These folders were uploaded into it and nothing else; the manifest lists every file with its sha256. "
            "Test results reached it only as aggregate scores. Every upload was checked for each test task's name "
            "and instruction text.</p><table><tr><th>Run</th><th>Mounted</th><th>Train tasks</th><th>Failures shown</th>"
            f"<th>Test tasks mounted</th><th>Network</th></tr>{rows}</table></div>"
        )
        if rows
        else ""
    )


def analysis(doc: dict) -> str:
    a = doc.get("analysis")
    if not a:
        return ""
    counts = "".join(
        f"<tr><td>{e(k.replace('_', ' '))}</td><td class=num>{v}</td></tr>"
        for k, v in a["counts"].items()
    )
    items = "".join(
        f"<tr><td>{e(f['id'])}</td><td>{e(f['category'])}</td><td>{e(f.get('explanation'))}</td></tr>"
        for f in a["failures"]
    )
    return (
        f"<h2>Stall analysis: remaining train failures by root cause</h2><div class=card><p>{e(a['summary'] or a['error'])}</p>"
        f"<table>{counts}</table><details><summary>Every failure</summary><table>{items}</table></details></div>"
    )


def diffs(doc: dict) -> str:
    out = []
    for r in doc["rounds"]:
        c = r["candidate"]
        if c.get("diff"):
            lines = "".join(
                f'<span class="{"add" if ln.startswith("+") and not ln.startswith("+++") else "del" if ln.startswith("-") and not ln.startswith("---") else ""}">{e(ln) or " "}</span>'
                for ln in c["diff"].splitlines()
            )
            out.append(
                f"<details{' open' if r['decision'] == 'keep' else ''}><summary>{e(c['id'])} {e(r['decision'])}: "
                f"{e(c.get('change'))}</summary><pre>{lines}</pre></details>"
            )
    return "<h2>Diffs</h2>" + "".join(out) if out else ""


def write_report(doc: dict, path: Path) -> Path:
    s = doc["settings"]
    body, data = chart(doc) if doc.get("baseline") else ("", {})
    legend = (
        '<div class="legend"><span><svg width="18" height="10"><line x1="1" x2="17" y1="5" y2="5" '
        'stroke="var(--train)" stroke-width="2"/></svg>Train</span><span><svg width="18" height="10">'
        '<line x1="1" x2="17" y1="5" y2="5" stroke="var(--test)" stroke-width="2"/></svg>Test</span>'
        "<span>shaded: 95% interval</span><span>● kept candidate</span><span>○ reverted candidate</span></div>"
    )
    page = (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><title>Hill-climb {e(path.parent.name)}</title>"
        f"<style>{CSS}</style></head><body><div class=hc><main><h1>Hill-climb: {e(path.parent.name)}</h1>"
        f"<p class=sub>{e(s['agent'])} / {e(s['model'])} · {len(doc['split']['train'])} train and "
        f"{len(doc['split']['test'])} test tasks · {s['trials']} trials per task</p>"
        + verdict(doc)
        + isolation(doc)
        + kpis(doc)
        + spending(doc)
        + (
            f"<h2>Score by round</h2><div class=card>{legend}<div class=chart>{body}<div class=tip></div></div></div>"
            if body
            else ""
        )
        + f"<h2>Decisions</h2><div class=card>{decisions(doc)}</div>"
        + mounts(doc)
        + analysis(doc)
        + diffs(doc)
        + (
            f"<h2>Warnings</h2><ul>{''.join(f'<li>{e(w)}</li>' for w in doc['warnings'])}</ul>"
            if doc["warnings"]
            else ""
        )
        + "<footer>Written by docs/examples/hillclimb from hillclimb.json. The loop follows “Automating eval "
        "design and hillclimbing with Claude” (claude.dev, 2026-09-28); here the test split is kept from the "
        "optimizer by what its sandbox is given.</footer></main></div>"
        f'<script type="application/json" id="hc-data">{json.dumps(data).replace("</", "<\\/")}</script>'
        f"<script>{SCRIPT}</script></body></html>\n"
    )
    path.write_text(page)
    return path
