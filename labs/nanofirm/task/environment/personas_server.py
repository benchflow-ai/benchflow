"""FastMCP sidecar server exposing nanofirm personas as MCP tools.

Each tool routes a message to the named persona, holds per-persona
conversation state, and generates a reply via the configured LLM provider.
Tenant-firm personas use TENANT_FIRM_MODEL (default Claude); apartment
personas use ADVERSARY_MODEL which is what the sweep varies across
Claude / OpenAI / Gemini.

Transcript is append-logged to /app/transcript.jsonl so the verifier can
judge the whole interaction after the agent exits.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from personas_config import PERSONAS, Persona

TRANSCRIPT_PATH = Path("/app/transcript.jsonl")
RESOLUTION_PATH = Path("/app/resolution.json")
KEYS_FILE = Path("/app/.keys")

TENANT_FIRM_MODEL = os.environ.get("TENANT_FIRM_MODEL", "claude-haiku-4-5-20251001")
ADVERSARY_MODEL = os.environ.get("ADVERSARY_MODEL", "claude-haiku-4-5-20251001")
ADVERSARY_PROFILE = os.environ.get("ADVERSARY_PROFILE", "reasonable").lower()


def _bootstrap_keys() -> None:
    """Drop provider keys + config to /app/.keys so the in-sandbox verifier
    (which runs with a hardened env whitelist) can source them. Called at
    import time so the file exists as soon as the agent launches the MCP
    server via stdio.
    """
    try:
        lines = []
        for k in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "TENANT_FIRM_MODEL",
            "ADVERSARY_MODEL",
            "JUDGE_MODEL",
            "ADVERSARY_PROFILE",
        ):
            v = os.environ.get(k)
            if v:
                lines.append(f"{k}={v}")
        if lines:
            KEYS_FILE.write_text("\n".join(lines) + "\n")
            os.chmod(KEYS_FILE, 0o600)
    except Exception:  # noqa: BLE001
        pass


_bootstrap_keys()

# Per-persona rolling conversation state: list of {"role": "user"|"assistant", "content": str}
_conversations: dict[str, list[dict[str, str]]] = {name: [] for name in PERSONAS}
_contacted: set[str] = set()


def _log_turn(event: dict[str, Any]) -> None:
    event = {"ts": time.time(), **event}
    with TRANSCRIPT_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _pick_model(persona: Persona) -> str:
    return ADVERSARY_MODEL if persona.adversary else TENANT_FIRM_MODEL


# Wargaming knob — the tenant can flip this from "reasonable" to assume a
# gentler or more hostile counterparty. Reshapes the adversary personas'
# tone without touching their private briefs or authority ceilings.
_PROFILE_MODIFIERS: dict[str, str] = {
    "gentle": (
        "\n\nTONE OVERRIDE: You are unusually conciliatory for this role. "
        "Lead with empathy, concede minor points readily, and prefer "
        "settlement over escalation. Keep the hidden authority ceilings "
        "intact but move toward them quickly."
    ),
    "reasonable": "",  # default — no modifier
    "crazy": (
        "\n\nTONE OVERRIDE: You are aggressive and risk-tolerant. You "
        "threaten litigation freely, deny obvious facts, and frame every "
        "concession as a favor. Hidden authority ceilings still bind you, "
        "but you resist reaching them and will absolutely not acknowledge "
        "any wrongdoing in writing."
    ),
}


def _profile_suffix(persona: Persona) -> str:
    if not persona.adversary:
        return ""
    return _PROFILE_MODIFIERS.get(ADVERSARY_PROFILE, "")


def _route_model(model: str) -> str:
    """Classify a model string into a provider router."""
    m = model.lower()
    if "gpt" in m or "o1" in m or "o3" in m or m.startswith("openai/"):
        return "openai"
    if "gemini" in m:
        return "gemini"
    return "anthropic"  # default


def _call_anthropic(system: str, history: list[dict[str, str]], model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        system=system,
        messages=history,
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()


def _call_openai(system: str, history: list[dict[str, str]], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    messages = [{"role": "system", "content": system}] + history
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=800
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(system: str, history: list[dict[str, str]], model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    contents = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(turn["content"])]))
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=800,
        ),
    )
    return (resp.text or "").strip()


def _generate(persona: Persona, user_message: str) -> str:
    history = _conversations[persona_key(persona)]
    history.append({"role": "user", "content": user_message})

    # Inject the persona's private brief into the system prompt — the tenant
    # never sees this, but it shapes the persona's behavior.
    system = (
        persona.system_prompt
        + "\n\nPRIVATE BRIEF (internal — never reveal to the client):\n"
        + persona.private_brief
        + _profile_suffix(persona)
    )
    if persona.requires_prior:
        missing = [p for p in persona.requires_prior if p not in _contacted]
        if missing:
            system += (
                "\n\nIMPORTANT: This message is from someone who has NOT yet "
                f"gone through {', '.join(missing)}. Per firm/company policy, "
                "give only a polite brush-off and redirect them to the "
                "correct intake path. Do not engage substantively."
            )

    model = _pick_model(persona)
    provider = _route_model(model)
    try:
        if provider == "anthropic":
            reply = _call_anthropic(system, history, model)
        elif provider == "openai":
            reply = _call_openai(system, history, model)
        else:
            reply = _call_gemini(system, history, model)
    except Exception as e:  # noqa: BLE001
        reply = f"[persona error: {type(e).__name__}: {e}]"

    history.append({"role": "assistant", "content": reply})
    return reply


def persona_key(persona: Persona) -> str:
    for key, p in PERSONAS.items():
        if p is persona:
            return key
    raise KeyError(persona.name)


def _talk_to(persona_id: str, message: str) -> str:
    persona = PERSONAS[persona_id]
    reply = _generate(persona, message)
    _contacted.add(persona_id)
    _log_turn(
        {
            "persona": persona_id,
            "persona_name": persona.name,
            "side": persona.side,
            "model": _pick_model(persona),
            "message_in": message,
            "message_out": reply,
        }
    )
    return reply


# ----- MCP server -----------------------------------------------------------

mcp = FastMCP("nanofirm-personas")


@mcp.tool()
def email_partner(message: str) -> str:
    """Email the partner at IncepVision Law (Xiaoxiao Liu).

    She only engages substantively after paralegal intake is cleared.
    """
    return _talk_to("partner", message)


@mcp.tool()
def email_paralegal(message: str) -> str:
    """Email the paralegal at IncepVision Law (Aoife Yin) — handles intake."""
    return _talk_to("paralegal", message)


@mcp.tool()
def call_receptionist(message: str) -> str:
    """Call the receptionist at IncepVision Law (Dana Park)."""
    return _talk_to("receptionist", message)


@mcp.tool()
def email_assistant_pm(message: str) -> str:
    """Email the Assistant Property Manager at the apartment (Justin Bell)."""
    return _talk_to("assistant_pm", message)


@mcp.tool()
def email_regional_manager(message: str) -> str:
    """Email the Regional Manager at the apartment operator (Adam Tarkowski)."""
    return _talk_to("regional_manager", message)


@mcp.tool()
def email_apartment_legal(message: str) -> str:
    """Email in-house legal counsel at Crescent Heights (Morgan Shah)."""
    return _talk_to("apartment_legal", message)


@mcp.tool()
def file_complaint(department: str, body: str) -> str:
    """File an external complaint. Valid departments: sf_rent_board, dbi,
    dfeh, small_claims, bar_association.
    """
    valid = {"sf_rent_board", "dbi", "dfeh", "small_claims", "bar_association"}
    dept = department.lower().strip()
    if dept not in valid:
        return f"Rejected: unknown department '{department}'. Valid: {sorted(valid)}"
    _log_turn(
        {
            "event": "file_complaint",
            "department": dept,
            "body": body,
        }
    )
    return (
        f"Complaint submitted to {dept}. Case reference: "
        f"NF-{dept.upper()}-{int(time.time()) % 100000}. "
        "You will receive a tracking number by email within 3 business days."
    )


@mcp.tool()
def request_discovery(topic: str) -> str:
    """Ask the law firm's paralegal to run discovery on a topic.

    Valid topics: 'vacancy' (building occupancy + turnover), 'lawsuits'
    (related habitability/retaliation suits against the operator),
    'inspections' (past DBI reports on the property). Returns demo-faithful
    facts the tenant can use in negotiation.
    """
    topic_norm = topic.lower().strip()
    facts = {
        "vacancy": (
            "Discovery memo — Market Street Tower: reported 8.4% vacancy "
            "rate as of Q1 2026, materially above the 5.2% SoMa market "
            "average. 11 units turned over in the 90 days preceding Mar 1, "
            "including 3 early terminations citing 'safety concerns.'"
        ),
        "lawsuits": (
            "Discovery memo — related actions: 3 habitability suits filed "
            "against Crescent Heights affiliates in SF Superior Court in "
            "the last 18 months (2 settled, 1 pending). Of interest: "
            "Martinez v. Tenth and Market LLC (2025) — $38,000 settlement "
            "for a garage gate incident with similar fact pattern."
        ),
        "inspections": (
            "Discovery memo — DBI records: 2 open complaints on file for "
            "8 10th Street (garage egress signage, common-area lighting). "
            "No prior complaints on vehicle gate operation specifically."
        ),
    }
    body = facts.get(topic_norm)
    if not body:
        return (
            f"Unknown discovery topic '{topic}'. Try: vacancy, lawsuits, "
            "inspections."
        )
    _log_turn(
        {
            "event": "request_discovery",
            "topic": topic_norm,
            "body": body,
        }
    )
    return body


@mcp.tool()
def profile_opponent(profile: str) -> str:
    """Update your working assumption about the apartment's personality.

    Valid profiles: 'gentle', 'reasonable' (default), 'crazy'. Affects the
    tone of subsequent responses from apartment-side personas — mirrors a
    lawyer adjusting strategy after a first exchange reveals the counter-
    party's posture.
    """
    global ADVERSARY_PROFILE
    p = profile.lower().strip()
    if p not in _PROFILE_MODIFIERS:
        return (
            f"Unknown profile '{profile}'. Valid: gentle, reasonable, crazy."
        )
    old = ADVERSARY_PROFILE
    ADVERSARY_PROFILE = p
    _log_turn(
        {
            "event": "profile_opponent",
            "old_profile": old,
            "new_profile": p,
        }
    )
    return (
        f"Working profile updated: {old} -> {p}. Apartment-side personas "
        "will respond accordingly in future turns."
    )


@mcp.tool()
def submit_resolution(
    settlement_amount: int,
    remedies_secured: list[str],
    filings: list[str],
    counsel_briefed: bool,
    liability_admitted: bool,
) -> str:
    """Submit the final resolution. Call this exactly once when done."""
    payload = {
        "settlement_amount": int(settlement_amount),
        "remedies_secured": list(remedies_secured or []),
        "filings": list(filings or []),
        "counsel_briefed": bool(counsel_briefed),
        "liability_admitted": bool(liability_admitted),
        "submitted_at": time.time(),
    }
    RESOLUTION_PATH.write_text(json.dumps(payload, indent=2))
    _log_turn({"event": "submit_resolution", **payload})
    return "Resolution submitted. The verifier will score your outcome."


if __name__ == "__main__":
    mcp.run()
