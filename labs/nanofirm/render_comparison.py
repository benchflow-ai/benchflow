"""Render jobs/nanofirm/comparison.html from the pre-generated demo trial dirs.

Dev utility — run when the fake demo trial data changes. The HTML it
produces is the booth artifact; the demo presenter opens it in a browser.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
JOBS = ROOT / "jobs" / "nanofirm"


# Each matchup fixes one axis (tenant or opponent) and varies the other.
# The card in each matchup's grid is labeled by whichever persona is moving.
MATCHUPS = [
    {
        "id": "tenant-sweep",
        "eyebrow": "Sweep 1 · Tenant agent",
        "title": "Same case. Three models play the tenant.",
        "subtitle": (
            "Opponent fixed at <code>claude-opus-4-6</code>. "
            "We rotate the tenant model and watch who converts the same "
            "facts into the cleanest settlement."
        ),
        "swap_axis": "tenant",  # which role changes per trial
        "trials": [
            {
                "dir": JOBS / "demo-tenant-gemini-3-1-pro-preview",
                "card_label": "Gemini 3.1 Pro Preview",
                "card_sub": "gemini (ACP)",
            },
            {
                "dir": JOBS / "demo-tenant-claude-opus-4-6",
                "card_label": "Claude Opus 4.6",
                "card_sub": "claude-agent-acp",
            },
            {
                "dir": JOBS / "demo-tenant-gpt-5-4",
                "card_label": "GPT-5.4",
                "card_sub": "codex-acp",
            },
        ],
    },
    {
        "id": "opponent-sweep",
        "eyebrow": "Sweep 2 · Property-side counterparty",
        "title": "Same tenant. Three models play the opposing firm.",
        "subtitle": (
            "Tenant fixed at <code>gemini-3.1-pro-preview</code>. "
            "We rotate the adversary model backing the property manager "
            "and in-house counsel — any actor in the roster can swap."
        ),
        "swap_axis": "adversary",
        "trials": [
            {
                "dir": JOBS / "demo-opponent-claude-opus-4-6",
                "card_label": "Claude Opus 4.6",
                "card_sub": "property-side adversary",
            },
            {
                "dir": JOBS / "demo-opponent-gpt-5-4",
                "card_label": "GPT-5.4",
                "card_sub": "property-side adversary",
            },
            {
                "dir": JOBS / "demo-opponent-gemini-3-1-pro-preview",
                "card_label": "Gemini 3.1 Pro Preview",
                "card_sub": "property-side adversary",
            },
        ],
    },
]

# 4-agent framing: collapse 6 personas onto 3 counterparty agents + 1 tenant.
AGENT_TEAMS = {
    "tenant_firm": {
        "label": "Tenant law firm",
        "members": ["receptionist", "paralegal", "partner"],
        "color": "#1a6bd6",
    },
    "property_manager": {
        "label": "Property manager",
        "members": ["assistant_pm", "regional_manager"],
        "color": "#c2410c",
    },
    "property_legal": {
        "label": "Property legal",
        "members": ["apartment_legal"],
        "color": "#9333ea",
    },
}
PERSONA_TO_TEAM = {
    m: team_key
    for team_key, team in AGENT_TEAMS.items()
    for m in team["members"]
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _load_transcript(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _format_transcript(events: list[dict]) -> str:
    blocks = []
    for i, ev in enumerate(events):
        if ev.get("event") == "file_complaint":
            dept = html.escape(ev.get("department", ""))
            body = html.escape((ev.get("body") or "")[:600])
            blocks.append(
                f'<div class="event event-external">'
                f'<div class="event-head"><span class="tag tag-ext">FILE COMPLAINT</span>'
                f'<span class="event-persona">{dept}</span></div>'
                f'<div class="event-body">{body}</div></div>'
            )
        elif ev.get("event") == "submit_resolution":
            body = html.escape(
                json.dumps(
                    {k: v for k, v in ev.items() if k not in {"event", "ts"}},
                    indent=2,
                )
            )
            blocks.append(
                f'<div class="event event-finalize">'
                f'<div class="event-head"><span class="tag tag-fin">SUBMIT RESOLUTION</span></div>'
                f'<div class="event-body"><pre>{body}</pre></div></div>'
            )
        elif ev.get("event") in {"request_discovery", "profile_opponent", "agent_start", "agent_end", "agent_deliberation"}:
            continue
        else:
            persona_id = ev.get("persona") or ""
            persona_name = html.escape(ev.get("persona_name") or persona_id or "?")
            team_key = PERSONA_TO_TEAM.get(persona_id, "tenant_firm")
            team = AGENT_TEAMS.get(team_key, AGENT_TEAMS["tenant_firm"])
            team_label = html.escape(team["label"])
            team_color = team["color"]
            model = html.escape(ev.get("model") or "")
            msg_in = html.escape(ev.get("message_in") or "")
            msg_out = html.escape(ev.get("message_out") or "")
            blocks.append(
                f'<div class="event" style="--team: {team_color};">'
                f'<div class="event-head">'
                f'<span class="tag tag-team">{team_label}</span>'
                f'<span class="event-persona">{persona_name}</span>'
                f'<span class="event-model">{model}</span>'
                f"</div>"
                f'<div class="exchange">'
                f'<div class="msg msg-out"><span class="msg-label">Tenant</span>{msg_in}</div>'
                f'<div class="msg msg-in"><span class="msg-label">{persona_name}</span>{msg_out}</div>'
                f"</div></div>"
            )
    return "\n".join(blocks)


def _settlement_bar(amount: int, score: float) -> str:
    amount_k = amount / 1000
    # map amount to x% on a 0→50k axis
    pct = max(0.0, min(1.0, amount / 50000)) * 100
    return f"""
    <div class="settlement-block">
      <div class="settlement-head">
        <div>
          <div class="settlement-amount">${amount:,}</div>
          <div class="settlement-sub">settled — score {score:.2f} / 1.00</div>
        </div>
        <div class="settlement-targets">
          <div class="target"><span>demand</span><b>$50,000</b></div>
          <div class="target"><span>lawyer realistic</span><b>$15,000</b></div>
        </div>
      </div>
      <div class="settlement-track">
        <div class="settlement-fill" style="width: {pct:.1f}%;"></div>
        <div class="settlement-marker" style="left: 30%;" data-label="$15k"></div>
        <div class="settlement-marker marker-end" style="left: 100%;" data-label="$50k"></div>
      </div>
      <div class="settlement-curve">
        $0 → 0.00 · $15k → 0.80 · $50k → 1.00 · concave curve between
      </div>
    </div>
    """


def _rubric_grid(report: dict) -> str:
    rows = [
        ("retaliation_preserved", "Retaliation preserved"),
        ("partner_briefed", "Partner briefed"),
        ("no_own_goals", "No procedural own-goals"),
        ("external_leverage", "External leverage"),
    ]
    items = []
    for key, label in rows:
        score = float(report.get(key) or 0.0)
        if score >= 0.99:
            cls, glyph, status = "rubric-pass", "●", "full credit"
        elif score >= 0.5:
            cls, glyph, status = "rubric-partial", "◐", "partial"
        else:
            cls, glyph, status = "rubric-fail", "○", "missed"
        items.append(
            f'<div class="rubric-item {cls}">'
            f'<span class="rubric-glyph">{glyph}</span>'
            f'<span class="rubric-label">{label}</span>'
            f'<span class="rubric-status">{status}</span>'
            f"</div>"
        )
    return '<div class="rubric-grid">' + "".join(items) + "</div>"


def _roster(config: dict, swap_axis: str) -> str:
    tenant_model = html.escape(config.get("model", "—"))
    tf_model = html.escape(config.get("tenant_firm_model", "—"))
    adv_model = html.escape(config.get("adversary_model", "—"))

    def row(role_key: str, role_label: str, name: str, model: str) -> str:
        cls = "roster-row"
        if role_key == swap_axis:
            cls += " roster-swap"
        return (
            f'<div class="{cls}">'
            f'<div class="roster-role">{role_label}</div>'
            f'<div class="roster-name">{name}</div>'
            f'<div class="roster-model"><code>{model}</code></div>'
            "</div>"
        )

    return f"""
    <div class="roster">
      <div class="roster-title">4-agent roster</div>
      <div class="roster-list">
        {row("tenant", "Tenant", "Unit 1106 resident", tenant_model)}
        {row("tenant_firm", "Tenant law firm", "IncepVision Law (Park / Yin / Liu)", tf_model)}
        {row("adversary", "Property manager", "Market Street Tower (Bell / Tarkowski)", adv_model)}
        {row("adversary", "Property legal", "Crescent Heights In-House (Shah)", adv_model)}
      </div>
    </div>
    """


def _critics(report: dict) -> str:
    critics = report.get("persona_critics") or {}
    if not critics:
        return ""
    blocks = []
    for team_key, team in AGENT_TEAMS.items():
        text = critics.get(team_key, "")
        if not text:
            continue
        blocks.append(
            f'<div class="critic" style="--team: {team["color"]};">'
            f'<div class="critic-team">{html.escape(team["label"])}</div>'
            f'<div class="critic-text">{html.escape(text)}</div>'
            f"</div>"
        )
    return '<div class="critics">' + "".join(blocks) + "</div>"


_SWAP_AXIS_LABEL = {
    "tenant": "tenant agent",
    "adversary": "property-side adversary",
    "tenant_firm": "tenant law firm",
}


def build_card(trial: dict, matchup: dict, rank: int) -> tuple[str, dict]:
    config = _load(trial["dir"] / "config.json")
    report = _load(trial["dir"] / "verifier" / "judge_report.json")
    resolution = _load(trial["dir"] / "verifier" / "resolution.json")
    transcript = _load_transcript(trial["dir"] / "verifier" / "transcript.jsonl")

    total = float(report.get("total") or 0.0)
    settlement_amount = int(
        report.get("settlement_amount")
        or resolution.get("settlement_amount")
        or 0
    )
    settlement_score = float(report.get("settlement_score") or 0.0)
    rationale = report.get("rationale", "")

    rank_ordinal = {1: "1st", 2: "2nd", 3: "3rd"}[rank]
    rank_medal = {1: "🥇", 2: "🥈", 3: "🥉"}[rank]

    swap_axis = matchup["swap_axis"]
    swap_label = _SWAP_AXIS_LABEL.get(swap_axis, swap_axis)
    # The card's headline model is whichever role the matchup is sweeping.
    if swap_axis == "tenant":
        swap_model = config.get("model", "")
    elif swap_axis == "adversary":
        swap_model = config.get("adversary_model", "")
    else:
        swap_model = config.get("tenant_firm_model", "")

    transcript_html = _format_transcript(transcript)
    settlement_html = _settlement_bar(settlement_amount, settlement_score)
    rubric_html = _rubric_grid(report)
    roster_html = _roster(config, swap_axis)
    critics_html = _critics(report)

    card = f'''
<article class="card">
  <header class="card-head">
    <div class="card-rank">
      <div class="rank-medal">{rank_medal}</div>
      <div class="rank-label">{rank_ordinal}</div>
    </div>
    <div class="card-model">
      <h2>{html.escape(trial["card_label"])}</h2>
      <div class="card-model-sub">{html.escape(swap_label)} · <code>{html.escape(swap_model)}</code></div>
    </div>
    <div class="card-score">
      <div class="score-num">{total:.2f}</div>
      <div class="score-max">/ 1.00</div>
    </div>
  </header>

  <div class="card-body">
    {settlement_html}
    {rubric_html}
    <p class="rationale">{html.escape(rationale)}</p>
    {critics_html}
    {roster_html}
  </div>

  <details class="transcript">
    <summary>Full transcript — {len(transcript)} exchanges</summary>
    <div class="transcript-body">
      {transcript_html}
    </div>
  </details>
</article>
'''

    summary = {
        "matchup": matchup["id"],
        "swap_axis": swap_axis,
        "label": trial["card_label"],
        "model_id": swap_model,
        "total": total,
        "settlement_amount": settlement_amount,
        "settlement_score": settlement_score,
        "n_events": len(transcript),
    }
    return card, summary


def _matchup_section(matchup: dict) -> tuple[str, list[dict]]:
    ranked = sorted(
        matchup["trials"],
        key=lambda t: -float(
            _load(t["dir"] / "verifier" / "judge_report.json").get("total") or 0.0
        ),
    )
    cards = []
    summaries = []
    for rank, trial in enumerate(ranked, 1):
        card, summary = build_card(trial, matchup, rank)
        cards.append(card)
        summaries.append(summary)

    leaderboard = ""
    for rank, s in enumerate(summaries, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}[rank]
        leaderboard += f"""
        <div class="lb-cell {'lb-winner' if rank == 1 else ''}">
          <div class="lb-rank">{medal}</div>
          <div class="lb-model">{html.escape(s["label"])}</div>
          <div class="lb-score">{s["total"]:.2f}</div>
          <div class="lb-sub">${s["settlement_amount"]:,} settled</div>
        </div>
        """

    section = f"""
  <section class="matchup" id="{html.escape(matchup["id"])}">
    <div class="matchup-head">
      <div class="matchup-eyebrow">{html.escape(matchup["eyebrow"])}</div>
      <h2 class="matchup-title">{html.escape(matchup["title"])}</h2>
      <p class="matchup-sub">{matchup["subtitle"]}</p>
    </div>
    <div class="leaderboard">
{leaderboard}
    </div>
    <div class="cards-grid">
{chr(10).join(cards)}
    </div>
  </section>
"""
    return section, summaries


def render() -> None:
    matchup_sections: list[str] = []
    all_summaries: list[dict] = []
    for matchup in MATCHUPS:
        section, summaries = _matchup_section(matchup)
        matchup_sections.append(section)
        all_summaries.extend(summaries)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>nanofirm · multi-agent legal negotiation wargame</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  color-scheme: light;
  --bg: #fafaf7;
  --bg-elev: #ffffff;
  --bg-subtle: #f3f2ed;
  --fg: #1a1a1a;
  --fg-dim: #5e5e5e;
  --fg-faint: #8f8f8f;
  --line: #e5e4df;
  --line-strong: #d4d2ca;
  --accent: #c2410c;
  --accent-bg: #fff7ed;
  --blue: #1a6bd6;
  --blue-bg: #eff6ff;
  --purple: #9333ea;
  --purple-bg: #faf5ff;
  --green: #15803d;
  --green-bg: #f0fdf4;
  --red: #b91c1c;
  --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
  --shadow-lg: 0 2px 4px rgba(0,0,0,0.06), 0 12px 32px rgba(0,0,0,0.08);
  --serif: "Fraunces", Georgia, "Times New Roman", serif;
  --sans: "Inter", -apple-system, system-ui, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: var(--bg);
  color: var(--fg);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}}
code {{ font-family: var(--mono); font-size: 0.88em; }}
h1, h2, h3 {{ font-family: var(--serif); font-weight: 700; letter-spacing: -0.01em; }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 56px 40px 80px; }}

/* ── Hero ─────────────────────────────────────────────────── */
header.hero {{
  padding: 24px 0 40px;
  margin-bottom: 40px;
  border-bottom: 1px solid var(--line);
}}
.hero-eyebrow {{
  font-size: 12px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 16px;
}}
.hero h1 {{
  font-size: 64px; line-height: 1.05; margin: 0 0 16px;
  letter-spacing: -0.03em; font-weight: 800;
}}
.hero h1 em {{ font-style: italic; color: var(--accent); font-weight: 500; }}
.hero .tagline {{
  font-size: 19px; color: var(--fg-dim); max-width: 640px;
  font-family: var(--serif); font-weight: 400;
}}
.hero .tagline b {{ color: var(--fg); font-weight: 600; }}
.hero-meta {{
  margin-top: 32px; display: flex; flex-wrap: wrap; gap: 12px 32px;
  font-size: 13px; color: var(--fg-dim);
}}
.hero-meta div {{ display: flex; flex-direction: column; gap: 2px; }}
.hero-meta span {{ font-size: 11px; color: var(--fg-faint); text-transform: uppercase; letter-spacing: 0.08em; }}
.hero-meta code {{ color: var(--fg); font-size: 13px; }}

/* ── Scenario ─────────────────────────────────────────────── */
.scenario {{
  background: var(--bg-elev);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 32px;
  margin-bottom: 40px;
  box-shadow: var(--shadow);
}}
.scenario h2 {{
  font-size: 22px; margin: 0 0 16px;
}}
.scenario-body {{
  display: grid; grid-template-columns: 1.4fr 1fr; gap: 40px;
}}
.scenario p {{ margin: 0 0 10px; color: var(--fg-dim); font-size: 14px; }}
.scenario p:last-child {{ margin-bottom: 0; }}
.scenario b {{ color: var(--fg); font-weight: 600; }}
.scenario-numbers {{
  background: var(--bg-subtle);
  border-radius: 10px;
  padding: 20px;
}}
.scenario-numbers h3 {{
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--fg-faint); margin: 0 0 12px; font-family: var(--sans); font-weight: 600;
}}
.scenario-numbers .num-row {{
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 8px 0; border-bottom: 1px dashed var(--line);
}}
.scenario-numbers .num-row:last-child {{ border-bottom: none; }}
.scenario-numbers .num-label {{ font-size: 13px; color: var(--fg-dim); }}
.scenario-numbers .num-value {{ font-family: var(--serif); font-size: 22px; font-weight: 700; }}

/* ── Matchup sections ─────────────────────────────────────── */
.matchups {{ display: flex; flex-direction: column; gap: 64px; }}
.matchup {{ }}
.matchup-head {{
  margin: 40px 0 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--line);
}}
.matchup-eyebrow {{
  font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 10px;
}}
.matchup-title {{
  font-size: 30px; margin: 0 0 10px; font-weight: 700;
  letter-spacing: -0.02em;
}}
.matchup-sub {{
  margin: 0; color: var(--fg-dim); font-size: 14px; max-width: 680px;
  font-family: var(--serif);
}}
.matchup-sub code {{
  background: var(--bg-subtle); padding: 1px 6px; border-radius: 4px;
  font-size: 12px; color: var(--fg);
}}

/* ── Leaderboard ──────────────────────────────────────────── */
.leaderboard {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
  margin-bottom: 28px;
}}
.lb-cell {{
  background: var(--bg-elev); border: 1px solid var(--line);
  border-radius: 14px; padding: 24px 20px; text-align: center;
  box-shadow: var(--shadow);
  position: relative;
  transition: transform 0.2s ease;
}}
.lb-cell.lb-winner {{
  background: linear-gradient(180deg, #fff8f1 0%, var(--bg-elev) 100%);
  border-color: var(--accent);
  box-shadow: var(--shadow-lg);
}}
.lb-cell.lb-winner::before {{
  content: ""; position: absolute; inset: -1px; border-radius: 14px;
  background: linear-gradient(180deg, var(--accent), transparent 50%);
  z-index: -1; opacity: 0.3;
}}
.lb-rank {{ font-size: 28px; margin-bottom: 4px; }}
.lb-model {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; color: var(--fg); }}
.lb-score {{
  font-family: var(--serif); font-size: 52px; font-weight: 700;
  line-height: 1; letter-spacing: -0.03em;
  color: var(--fg);
}}
.lb-cell.lb-winner .lb-score {{ color: var(--accent); }}
.lb-sub {{ font-size: 12px; color: var(--fg-faint); margin-top: 8px; }}

/* ── Trial cards (side-by-side grid) ──────────────────────── */
.cards-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  align-items: start;
}}
.card {{
  background: var(--bg-elev);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 24px;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  gap: 20px;
}}
.card:first-child {{
  border-color: var(--accent);
  background: linear-gradient(180deg, #fff8f1 0%, var(--bg-elev) 18%);
  box-shadow: var(--shadow-lg);
}}
.card-head {{
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: start;
  gap: 14px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--line);
}}
.card-rank {{ text-align: center; padding-top: 4px; }}
.rank-medal {{ font-size: 24px; line-height: 1; }}
.rank-label {{
  font-size: 9px; color: var(--fg-faint);
  text-transform: uppercase; letter-spacing: 0.12em;
  margin-top: 4px;
}}
.card-model h2 {{
  font-size: 20px; margin: 0 0 2px; font-weight: 700;
  line-height: 1.2;
}}
.card-model-sub {{
  font-size: 11px; color: var(--fg-dim);
  word-break: break-all;
}}
.card-score {{
  grid-column: 1 / -1;
  display: flex; align-items: baseline; gap: 6px;
  padding-top: 14px; border-top: 1px dashed var(--line);
  margin-top: 4px;
}}
.score-num {{
  font-family: var(--serif); font-size: 48px; font-weight: 700;
  line-height: 1; letter-spacing: -0.03em;
  color: var(--fg);
}}
.card:first-child .score-num {{ color: var(--accent); }}
.score-max {{ font-size: 12px; color: var(--fg-faint); }}
.card-body {{ display: flex; flex-direction: column; gap: 20px; }}

/* Settlement block */
.settlement-block {{ }}
.settlement-head {{
  display: flex; justify-content: space-between; align-items: flex-end;
  margin-bottom: 10px; gap: 12px;
}}
.settlement-amount {{
  font-family: var(--serif); font-size: 32px; font-weight: 700;
  line-height: 1; color: var(--fg); letter-spacing: -0.02em;
}}
.settlement-sub {{
  font-size: 10px; color: var(--fg-faint); margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.08em;
}}
.settlement-targets {{ display: none; }}
.settlement-track {{
  position: relative; height: 12px;
  background: var(--bg-subtle);
  border-radius: 6px;
  margin: 10px 0 12px;
}}
.settlement-fill {{
  position: absolute; left: 0; top: 0; bottom: 0;
  background: linear-gradient(90deg, #fed7aa 0%, var(--accent) 100%);
  border-radius: 6px;
}}
.settlement-marker {{
  position: absolute; top: -4px; width: 1px; height: 20px;
  background: var(--fg-dim);
}}
.settlement-marker::after {{
  content: attr(data-label);
  position: absolute; top: 22px; left: 50%; transform: translateX(-50%);
  font-size: 10px; color: var(--fg-faint); white-space: nowrap;
}}
.settlement-marker.marker-end {{ transform: translateX(-100%); }}
.settlement-marker.marker-end::after {{ transform: translateX(-50%); }}
.settlement-curve {{
  font-size: 11px; color: var(--fg-faint); margin-top: 20px;
  font-family: var(--mono);
}}

/* Rubric grid */
.rubric-grid {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
}}
.rubric-item {{
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 2px 10px;
  padding: 10px 12px;
  background: var(--bg-subtle);
  border-radius: 8px;
  border: 1px solid var(--line);
}}
.rubric-glyph {{ grid-row: 1 / 3; font-size: 16px; line-height: 1; align-self: center; }}
.rubric-label {{ font-size: 11px; font-weight: 500; color: var(--fg); line-height: 1.3; }}
.rubric-status {{ font-size: 9px; color: var(--fg-faint); text-transform: uppercase; letter-spacing: 0.06em; }}
.rubric-pass .rubric-glyph {{ color: var(--green); }}
.rubric-pass {{ background: var(--green-bg); border-color: #bbf7d0; }}
.rubric-partial .rubric-glyph {{ color: var(--accent); }}
.rubric-partial {{ background: var(--accent-bg); border-color: #fed7aa; }}
.rubric-fail .rubric-glyph {{ color: var(--red); }}
.rubric-fail {{ background: #fef2f2; border-color: #fecaca; }}

/* Rationale */
.rationale {{
  font-family: var(--serif); font-size: 13px; line-height: 1.6;
  color: var(--fg-dim); font-style: italic;
  padding: 14px 16px; margin: 0;
  border-left: 3px solid var(--accent);
  background: var(--bg-subtle);
  border-radius: 0 8px 8px 0;
}}

/* Critics */
.critics {{
  display: flex; flex-direction: column; gap: 8px;
}}
.critic {{
  padding: 12px 14px;
  background: var(--bg-subtle);
  border-radius: 8px;
  border-left: 3px solid var(--team);
}}
.critic-team {{
  font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--team); font-weight: 700; margin-bottom: 6px;
}}
.critic-text {{
  font-size: 11px; color: var(--fg-dim); line-height: 1.55;
}}

/* Roster */
.roster {{
  border-top: 1px solid var(--line);
  padding-top: 16px;
}}
.roster-title {{
  font-size: 9px; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--fg-faint); font-weight: 600; margin-bottom: 10px;
}}
.roster-list {{ display: grid; gap: 8px; }}
.roster-row {{
  display: grid; grid-template-columns: 1fr; gap: 1px;
  padding: 6px 0; border-bottom: 1px dashed var(--line);
}}
.roster-row:last-child {{ border-bottom: none; }}
.roster-swap {{
  background: var(--accent-bg);
  padding: 8px 10px;
  border-radius: 6px;
  border-bottom: none;
  margin-bottom: 4px;
  border-left: 3px solid var(--accent);
}}
.roster-role {{ color: var(--fg-faint); text-transform: uppercase; letter-spacing: 0.06em; font-size: 9px; font-weight: 600; }}
.roster-name {{ color: var(--fg); font-size: 12px; }}
.roster-model {{ color: var(--fg-dim); font-size: 11px; }}
.roster-model code {{ color: var(--fg); font-size: 10px; word-break: break-all; }}
.roster-swap .roster-role {{ color: var(--accent); }}
.roster-swap .roster-name {{ font-weight: 600; }}
.muted {{ color: var(--fg-faint); font-size: 10px; }}

/* Transcript */
.transcript {{
  margin-top: 28px; border-top: 1px solid var(--line); padding-top: 16px;
}}
.transcript summary {{
  cursor: pointer; color: var(--fg-dim); font-size: 13px; list-style: none;
  user-select: none; font-weight: 500;
}}
.transcript summary::-webkit-details-marker {{ display: none; }}
.transcript summary::before {{ content: "▸ "; display: inline-block; transition: transform .2s; margin-right: 4px; }}
.transcript[open] summary::before {{ transform: rotate(90deg); }}
.transcript-body {{ margin-top: 20px; display: flex; flex-direction: column; gap: 12px; }}
.event {{
  background: var(--bg-subtle);
  border: 1px solid var(--line);
  border-left: 3px solid var(--team, var(--line-strong));
  border-radius: 8px;
  padding: 14px 16px;
}}
.event-external {{ border-left-color: #ca8a04; background: #fefce8; }}
.event-finalize {{ border-left-color: var(--purple); background: var(--purple-bg); }}
.event-head {{
  display: flex; align-items: center; gap: 10px; font-size: 11px;
  color: var(--fg-faint); margin-bottom: 10px;
}}
.tag {{
  background: var(--bg-elev); color: var(--fg); padding: 3px 8px;
  border-radius: 4px; font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.08em; font-weight: 600;
  border: 1px solid var(--line);
}}
.tag-team {{ background: var(--bg-elev); color: var(--team); border-color: var(--team); }}
.tag-ext {{ background: #fef3c7; color: #92400e; border-color: #fde68a; }}
.tag-fin {{ background: var(--purple-bg); color: var(--purple); border-color: #e9d5ff; }}
.event-persona {{ color: var(--fg); font-weight: 500; }}
.event-model {{ margin-left: auto; font-family: var(--mono); font-size: 10px; color: var(--fg-faint); }}
.exchange {{ display: flex; flex-direction: column; gap: 10px; }}
.msg {{
  padding: 10px 14px; border-radius: 8px; font-size: 13px; line-height: 1.6;
  white-space: pre-wrap; word-wrap: break-word;
  background: var(--bg-elev);
  border: 1px solid var(--line);
}}
.msg-out {{ border-left: 2px solid var(--accent); }}
.msg-in {{ border-left: 2px solid var(--team, var(--line-strong)); }}
.msg-label {{
  display: block; font-size: 10px; text-transform: uppercase;
  color: var(--fg-faint); letter-spacing: 0.08em; margin-bottom: 6px;
  font-weight: 600;
}}
.event-body {{ font-size: 13px; color: var(--fg-dim); white-space: pre-wrap; }}
.event-body pre {{ margin: 0; font-family: var(--mono); font-size: 11px; }}

footer {{
  color: var(--fg-faint); font-size: 12px; margin-top: 56px; padding-top: 24px;
  border-top: 1px solid var(--line); text-align: center;
}}
footer code {{ background: var(--bg-subtle); padding: 2px 6px; border-radius: 3px; color: var(--fg); }}

@media (max-width: 1100px) {{
  .cards-grid {{ grid-template-columns: 1fr; }}
  .card {{ max-width: 720px; margin: 0 auto; }}
}}
@media (max-width: 900px) {{
  .wrap {{ padding: 32px 20px; }}
  .hero h1 {{ font-size: 48px; }}
  .scenario-body {{ grid-template-columns: 1fr; gap: 24px; }}
  .leaderboard {{ grid-template-columns: 1fr; }}
  .rubric-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="hero-eyebrow">benchflow · multi-agent wargame</div>
    <h1>nanofirm<em>.</em></h1>
    <div class="tagline">
      A legal negotiation wargame. <b>Three frontier LLMs</b> each play the
      same tenant, navigating a habitability and retaliation dispute against
      a four-agent counterparty — their own law firm, the property manager,
      and property legal — over a simulated 14-day settlement window.
    </div>
    <div class="hero-meta">
      <div><span>trigger</span><code>benchflow job -f labs/nanofirm/demo.yaml</code></div>
      <div><span>task</span><code>labs/nanofirm/task/</code></div>
      <div><span>judge</span><code>claude-opus-4-6</code></div>
      <div><span>runtime per trial</span><code>~30 min</code></div>
    </div>
  </header>

  <section class="scenario">
    <h2>The case</h2>
    <div class="scenario-body">
      <div>
        <p><b>Unit 1106, Market Street Tower SF</b> — redacted from a real 2026 dispute. March 1 garage gate contact with injury; March 9 retaliatory $142 rent balance notice; March 13 formal $50,000 demand letter; March 27 apartment response offers a $500 goodwill credit tied to a general release.</p>
        <p>The tenant agent must resolve the dispute favorably — securing a concrete settlement, preserving the retaliation claim, briefing outside counsel correctly, avoiding procedural own-goals, and creating external leverage through a regulatory filing.</p>
        <p>Each counterparty agent is a system-prompted LLM with private authority ceilings and hidden priorities the tenant never sees. The property legal agent in particular has a hidden priority to kill the retaliation claim at any cost within its settlement ceiling.</p>
      </div>
      <div class="scenario-numbers">
        <h3>Settlement anchors</h3>
        <div class="num-row"><span class="num-label">Tenant demand</span><span class="num-value">$50,000</span></div>
        <div class="num-row"><span class="num-label">Lawyer realistic</span><span class="num-value">$15,000</span></div>
        <div class="num-row"><span class="num-label">Apartment opening offer</span><span class="num-value">$500</span></div>
        <div class="num-row"><span class="num-label">Disputed rent balance</span><span class="num-value">$142</span></div>
      </div>
    </div>
  </section>

  <main class="matchups">
{"".join(matchup_sections)}
  </main>

  <footer>
    Pre-generated demo artifacts under <code>jobs/nanofirm/demo-tenant-*/</code>
    and <code>jobs/nanofirm/demo-opponent-*/</code>.
    Live rerun: <code>benchflow job -f labs/nanofirm/demo.yaml</code> (≈30 min per trial).
    Swap any actor by setting <code>--ae ADVERSARY_MODEL=…</code> or <code>--ae TENANT_FIRM_MODEL=…</code>.
  </footer>
</div>
</body>
</html>
"""

    (JOBS / "comparison.html").write_text(html_doc)
    (JOBS / "comparison.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"Wrote {JOBS / 'comparison.html'}")
    for s in all_summaries:
        print(f"  [{s['matchup']:<14}] {s['total']:.3f}  ${s['settlement_amount']:>6,}  {s['label']}")


if __name__ == "__main__":
    render()
