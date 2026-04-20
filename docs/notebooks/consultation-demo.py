"""Realistic multi-round demo — client↔advisor contract consultation.

Shows how benchflow's Scene lifecycle handles a real-world multi-agent
interaction: a client shares details gradually, an advisor analyzes and
recommends. This is the pattern harbor #1316 was designed for.

No Docker required — runs entirely via LLM API calls to demonstrate
the Scene/Role/Turn data model without needing Daytona.

Usage:
    export GEMINI_API_KEY="AIza..."
    python docs/notebooks/consultation-demo.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Show the Scene config — this is what matters for the demo
def show_scene_config():
    """Print the YAML-equivalent config that drives this interaction."""
    print("""
┌─────────────────────────────────────────────────────────┐
│  Scene Config (what you'd put in a trial YAML)          │
├─────────────────────────────────────────────────────────┤
│  scenes:                                                │
│    - name: contract-review                              │
│      roles:                                             │
│        - name: client                                   │
│          agent: gemini                                  │
│          model: gemini-3.1-flash-lite-preview            │
│        - name: advisor                                  │
│          agent: gemini                                  │
│          model: gemini-3.1-flash-lite-preview            │
│      turns:                                             │
│        - role: client   # shares initial situation      │
│        - role: advisor  # analyzes contract             │
│        - role: client   # clarifies priorities          │
│        - role: advisor  # gives final recommendation    │
└─────────────────────────────────────────────────────────┘

This is 15 lines of config. Harbor PR #1462 needs 600+ lines of
runtime code to support the same pattern.
""")


def run_consultation():
    """Simulate the multi-round consultation using direct LLM calls.

    This demonstrates the INTERACTION PATTERN, not the full benchflow
    sandbox. For the full pipeline (Docker, verifier, trajectory),
    use the Trial API with a real task.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("pip install google-generativeai")
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    contract = Path(__file__).parent / "consultation-task" / "contract.md"
    contract_text = contract.read_text()

    # ── Round 1: Client shares situation ──────────────────────────────
    print("=" * 60)
    print("Round 1: Client → Advisor")
    print("=" * 60)

    client_msg_1 = model.generate_content(
        f"You are a startup CTO. You received this vendor contract and "
        f"need help reviewing it. Share your main concerns in 3-4 sentences. "
        f"Focus on: you're a small company, cash-conscious, and worried about "
        f"lock-in.\n\nContract:\n{contract_text}"
    ).text
    print(f"\n[Client]: {client_msg_1[:300]}...")

    # ── Round 1: Advisor analyzes ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Round 1: Advisor → Client")
    print("=" * 60)

    advisor_msg_1 = model.generate_content(
        f"You are a contract review advisor. A client shared these concerns:\n\n"
        f"{client_msg_1}\n\n"
        f"Review this contract and provide a structured risk analysis:\n\n"
        f"{contract_text}\n\n"
        f"Format: list each clause with HIGH/MEDIUM/LOW risk and a 1-line explanation."
    ).text
    print(f"\n[Advisor]: {advisor_msg_1[:500]}...")

    # ── Round 2: Client clarifies priorities ──────────────────────────
    print("\n" + "=" * 60)
    print("Round 2: Client → Advisor")
    print("=" * 60)

    client_msg_2 = model.generate_content(
        f"You are the startup CTO. The advisor gave this analysis:\n\n"
        f"{advisor_msg_1[:500]}\n\n"
        f"Based on this, tell the advisor your top 2 priorities to negotiate. "
        f"Be specific about what terms you'd accept. 2-3 sentences."
    ).text
    print(f"\n[Client]: {client_msg_2[:300]}...")

    # ── Round 2: Advisor gives final recommendation ───────────────────
    print("\n" + "=" * 60)
    print("Round 2: Advisor → Client (final recommendation)")
    print("=" * 60)

    advisor_msg_2 = model.generate_content(
        f"You are the contract advisor. The client's priorities:\n\n"
        f"{client_msg_2}\n\n"
        f"Original contract:\n{contract_text}\n\n"
        f"Write a final recommendation with:\n"
        f"1. Specific redline changes to propose\n"
        f"2. Fallback positions if vendor pushes back\n"
        f"3. Deal-breakers (walk away if these aren't fixed)\n"
        f"Be concrete — reference specific clause numbers."
    ).text
    print(f"\n[Advisor]: {advisor_msg_2[:600]}...")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Rounds:    2 (client→advisor→client→advisor)")
    print(f"  Pattern:   Multi-ROUND (two roles communicating)")
    print(f"  Messages:  4 total")
    print(f"  Harbor equivalent: PR #1462 (BaseUser + UserFactory + per-round archiving)")
    print(f"  BenchFlow:         15 lines of YAML config")


if __name__ == "__main__":
    print("BenchFlow Scene Demo — Realistic Contract Consultation")
    print("=" * 60)
    show_scene_config()
    run_consultation()
