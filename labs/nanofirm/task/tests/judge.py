"""In-sandbox LLM-as-judge for the nanofirm task.

Reads /app/transcript.jsonl (every persona turn) and /app/resolution.json
(tenant's final submission), calls Claude as a judge against the 5-objective
rubric, and writes:
- /logs/verifier/reward.txt          (scalar 0.0–1.0)
- /logs/verifier/judge_report.json   (full rubric + rationale)
- /logs/verifier/transcript.jsonl    (copy for trial-dir sync)
- /logs/verifier/resolution.json     (copy for trial-dir sync)

ANTHROPIC_API_KEY is NOT in the verifier env whitelist, so we rely on
/app/.keys — the personas server drops it there on first tool call.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

TRANSCRIPT = Path("/app/transcript.jsonl")
RESOLUTION = Path("/app/resolution.json")
KEYS_FILE = Path("/app/.keys")
REWARD_FILE = Path("/logs/verifier/reward.txt")
REPORT_FILE = Path("/logs/verifier/judge_report.json")
STAGED_TRANSCRIPT = Path("/logs/verifier/transcript.jsonl")
STAGED_RESOLUTION = Path("/logs/verifier/resolution.json")

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-6")

RUBRIC = """\
You are an impartial judge evaluating whether a tenant agent achieved five
objectives in a simulated landlord-tenant dispute. Score each objective
independently as 0.0, 0.1, or 0.2 based ONLY on evidence in the transcript
and the tenant's submitted resolution. Be strict.

OBJECTIVES:

1. SETTLEMENT_SECURED (max 0.2)
   The apartment committed to a concrete dollar figure that the tenant
   accepted. A vague "we will consider" does not count. The submitted
   settlement_amount should be corroborated by apartment-side messages.

2. RETALIATION_PRESERVED (max 0.2)
   The tenant did NOT waive the retaliation claim (the March rent balance
   was plausibly retaliatory). Giving it up for settlement = 0. Explicit
   preservation OR an SF Rent Board complaint about retaliation = 0.2.

3. PARTNER_BRIEFED (max 0.2)
   The tenant reached the partner substantively — NOT just a paralegal-
   intake brush-off. A substantive partner reply must appear in the
   transcript. Also check resolution.counsel_briefed.

4. NO_OWN_GOALS (max 0.2)
   The tenant did NOT admit liability and did NOT sign anything releasing
   claims they should preserve. resolution.liability_admitted must be
   False. Written admissions in the transcript count against.

5. EXTERNAL_LEVERAGE (max 0.2)
   At least one complaint filed with a relevant external body
   (sf_rent_board, dbi, dfeh, small_claims, bar_association). Check the
   transcript for file_complaint events and resolution.filings.

Return STRICT JSON only, no prose or markdown fences:

{
  "settlement_secured": 0.0,
  "retaliation_preserved": 0.0,
  "partner_briefed": 0.0,
  "no_own_goals": 0.0,
  "external_leverage": 0.0,
  "total": 0.0,
  "rationale": "one paragraph explaining the scores"
}
"""


def _load_keys() -> None:
    if not KEYS_FILE.exists():
        return
    for line in KEYS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _load_transcript() -> list[dict]:
    if not TRANSCRIPT.exists():
        return []
    out = []
    for line in TRANSCRIPT.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _load_resolution() -> dict:
    if not RESOLUTION.exists():
        return {}
    try:
        return json.loads(RESOLUTION.read_text())
    except json.JSONDecodeError:
        return {}


def _summarize_transcript(events: list[dict]) -> str:
    lines = []
    for i, ev in enumerate(events):
        if ev.get("event") == "file_complaint":
            lines.append(
                f"[{i}] FILE_COMPLAINT dept={ev.get('department')} "
                f"body={(ev.get('body') or '')[:400]}"
            )
        elif ev.get("event") == "submit_resolution":
            body = {k: v for k, v in ev.items() if k not in {"event", "ts"}}
            lines.append(f"[{i}] SUBMIT_RESOLUTION {json.dumps(body)}")
        else:
            persona = ev.get("persona_name", ev.get("persona", "?"))
            side = ev.get("side", "?")
            msg_in = (ev.get("message_in") or "")[:450]
            msg_out = (ev.get("message_out") or "")[:450]
            lines.append(f"[{i}] -> {persona} ({side}): {msg_in}")
            lines.append(f"[{i}] <- {persona}: {msg_out}")
    return "\n".join(lines)


def _judge(summary: str, resolution: dict) -> dict:
    from anthropic import Anthropic

    client = Anthropic()
    user = (
        "RESOLUTION:\n"
        + json.dumps(resolution, indent=2)
        + "\n\nTRANSCRIPT:\n"
        + summary
        + "\n\nReturn strict JSON per the rubric."
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1500,
        system=RUBRIC,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "error": f"judge returned non-JSON: {e}",
            "raw": text,
            "total": 0.0,
        }


def main() -> int:
    REWARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    _load_keys()

    # Stage artifacts for the trial dir even on failure.
    if TRANSCRIPT.exists():
        shutil.copy(TRANSCRIPT, STAGED_TRANSCRIPT)
    if RESOLUTION.exists():
        shutil.copy(RESOLUTION, STAGED_RESOLUTION)

    events = _load_transcript()
    resolution = _load_resolution()

    if not events or not resolution or not os.environ.get("ANTHROPIC_API_KEY"):
        REPORT_FILE.write_text(
            json.dumps(
                {
                    "error": "missing transcript, resolution, or ANTHROPIC_API_KEY",
                    "events": len(events),
                    "resolution_present": bool(resolution),
                    "api_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
                    "total": 0.0,
                },
                indent=2,
            )
        )
        REWARD_FILE.write_text("0.0")
        print("JUDGE: missing inputs — reward 0.0")
        return 0

    summary = _summarize_transcript(events)
    report = _judge(summary, resolution)
    REPORT_FILE.write_text(json.dumps(report, indent=2))

    total = float(report.get("total") or 0.0)
    total = max(0.0, min(1.0, total))
    REWARD_FILE.write_text(f"{total:.3f}")
    print(f"JUDGE total={total:.3f}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
