"""BenchFlow Scene Patterns — Contract Review Tutorial

One scenario, four patterns. Each pattern fixes a specific failure of
the previous one. All constructed here — no external tasks needed.

Scenario: A startup is reviewing a vendor SaaS contract. The contract
has buried risks (auto-renewal, uncapped price escalation, weak SLA,
short data export window). Can AI agents find them?

Pattern 1: Single agent reviews → misses subtle issues
Pattern 2: Same agent self-reviews → catches more with a second pass
Pattern 3: Two specialists (legal + commercial) → disagree and synthesize
Pattern 4: Client clarifies priorities → advisor gives targeted redlines

Run:
    export GEMINI_API_KEY="AIza..."
    python docs/examples/scene-patterns.py
"""

import os

# ── The contract (constructed here, no external files) ──────────────

CONTRACT = """
SOFTWARE LICENSE AGREEMENT

Between: CloudStack Inc. ("Vendor") and [Client] ("Licensee")

1. LICENSE GRANT
   Non-exclusive, non-transferable license for internal use only.

2. TERM AND RENEWAL
   Initial term: 36 months. Auto-renews for 12-month periods.
   Cancellation requires 90 days written notice before expiration.

3. FEES
   Year 1: $120,000 | Year 2: $138,000 (+15%) | Year 3: $158,700 (+15%)
   Renewal pricing: Vendor's then-current list price (no cap).

4. DATA OWNERSHIP
   Client data remains Client's property. Vendor may use anonymized
   aggregated data for product improvement. Upon termination, Vendor
   provides data export for 30 days, then may delete.

5. SERVICE LEVEL
   Vendor targets 99.5% monthly uptime. No credits or remedies.

6. LIABILITY
   Total liability capped at fees paid in prior 12 months.
   No liability for indirect, consequential, or special damages.

7. TERMINATION
   Material breach: 30 days notice + cure period.
   Payment default: Vendor may terminate after 15 days.

8. GOVERNING LAW
   State of Delaware.
"""

# ── Helpers ─────────────────────────────────────────────────────────

def llm(prompt, model="gemini-3.1-flash-lite"):
    """Call Gemini and return text. Handles rate limits gracefully."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("  pip install google-generativeai")
        return "[LLM unavailable]"

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "[No API key set]"

    genai.configure(api_key=api_key)
    try:
        response = genai.GenerativeModel(model).generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Error: {e}]"


def show_config(yaml_str):
    """Print a YAML config box."""
    print("  ┌─ TrialConfig (YAML) ─────────────────────────────────┐")
    for line in yaml_str.strip().split("\n"):
        print(f"  │ {line:<53} │")
    print("  └─────────────────────────────────────────────────────┘")


# ── Pattern 1: Single Agent Baseline ────────────────────────────────

def pattern_1():
    print("=" * 60)
    print("Pattern 1: Single Agent Baseline")
    print("=" * 60)
    print()
    print("Problem: Can one agent find all the risks in this contract?")
    print()

    show_config("""scenes:
  - name: review
    roles: [{name: reviewer, agent: gemini}]
    turns: [{role: reviewer}]""")
    print()

    result = llm(
        f"Review this contract. List the top 3 risks for a small startup. "
        f"Be specific — cite clause numbers.\n\n{CONTRACT}"
    )
    print(f"[Reviewer]: {result[:400]}")
    print()
    print("→ Single agent finds obvious issues but often misses the")
    print("  compounding risk: uncapped renewal + auto-renewal + short")
    print("  cancellation window = vendor lock-in trap.")
    return result


# ── Pattern 2: Single-Agent Multi-Turn (Self-Review) ────────────────

def pattern_2(first_review):
    print("\n" + "=" * 60)
    print("Pattern 2: Single-Agent Multi-Turn (Self-Review)")
    print("=" * 60)
    print()
    print("Problem: The first pass missed subtle compound risks.")
    print("Fix: Same agent gets a second turn to self-check.")
    print()

    show_config("""scenes:
  - name: self-review
    roles: [{name: reviewer, agent: gemini}]
    turns:
      - {role: reviewer}
      - {role: reviewer, prompt: "Re-examine for compound risks"}""")
    print()

    result = llm(
        f"You previously reviewed a contract and found these issues:\n\n"
        f"{first_review[:500]}\n\n"
        f"Now re-examine the contract for COMPOUND risks — combinations "
        f"of clauses that together create a worse situation than any single "
        f"clause alone. Focus on lock-in, cost escalation, and data leverage.\n\n"
        f"{CONTRACT}"
    )
    print(f"[Reviewer, turn 2]: {result[:400]}")
    print()
    print("→ Self-review catches compound risks (auto-renewal + uncapped")
    print("  pricing + 90-day notice = you're locked into a 15% annual")
    print("  increase with no negotiation leverage).")
    print()
    print("When to use: cheap, fast, works for well-defined tasks.")
    print("When NOT to use: when the agent's blind spots are systematic")
    print("  (same model won't catch what it missed the first time).")
    return result


# ── Pattern 3: Multi-Round (Legal + Commercial Specialists) ─────────

def pattern_3():
    print("\n" + "=" * 60)
    print("Pattern 3: Multi-Round (Two Specialists)")
    print("=" * 60)
    print()
    print("Problem: One agent has blind spots. A legal reviewer focuses")
    print("on liability; a commercial reviewer focuses on cost. Neither")
    print("alone catches everything.")
    print("Fix: Two roles with different system prompts, each reviewing")
    print("independently, then synthesizing.")
    print()

    show_config("""scenes:
  - name: specialist-review
    roles:
      - {name: legal, agent: gemini}
      - {name: commercial, agent: gemini}
    turns:
      - {role: legal}
      - {role: commercial}
      - {role: legal, prompt: "Synthesize with commercial review"}""")
    print()

    legal = llm(
        f"You are a legal reviewer. Focus ONLY on liability, "
        f"indemnification, termination rights, and governing law.\n\n{CONTRACT}"
    )
    print(f"[Legal]: {legal[:300]}")
    print()

    commercial = llm(
        f"You are a commercial reviewer. Focus ONLY on pricing, "
        f"renewal terms, SLA value, and total cost of ownership.\n\n{CONTRACT}"
    )
    print(f"[Commercial]: {commercial[:300]}")
    print()

    synthesis = llm(
        f"You are the lead reviewer. Synthesize these two specialist reviews:\n\n"
        f"LEGAL REVIEW:\n{legal[:400]}\n\n"
        f"COMMERCIAL REVIEW:\n{commercial[:400]}\n\n"
        f"Produce a unified risk assessment with specific redline recommendations."
    )
    print(f"[Synthesis]: {synthesis[:400]}")
    print()
    print("→ Two specialists catch different issues. Legal flags the")
    print("  liability cap; commercial flags the uncapped renewal pricing.")
    print("  Synthesis combines them into actionable redlines.")
    print()
    print("When to use: complex tasks requiring multiple perspectives.")
    print("When NOT to use: simple tasks where a second opinion adds cost")
    print("  but not insight.")


# ── Pattern 4: Interactive Client Consultation ──────────────────────

def pattern_4():
    print("\n" + "=" * 60)
    print("Pattern 4: Interactive Multi-Round (Client ↔ Advisor)")
    print("=" * 60)
    print()
    print("Problem: Generic advice isn't actionable. The client has")
    print("specific priorities (cash-conscious startup, worried about")
    print("lock-in) that change which terms matter most.")
    print("Fix: Client shares context, advisor tailors recommendations.")
    print()

    show_config("""scenes:
  - name: consultation
    roles:
      - {name: client, agent: gemini}
      - {name: advisor, agent: gemini}
    turns:
      - {role: client}
      - {role: advisor}
      - {role: client, prompt: "Clarify priorities"}
      - {role: advisor, prompt: "Final redlines"}""")
    print()

    client_1 = llm(
        f"You are a startup CTO (Series A, 20 employees, $3M ARR). "
        f"You received this vendor contract. In 2-3 sentences, explain "
        f"your situation and main concerns.\n\n{CONTRACT}"
    )
    print(f"[Client]: {client_1[:250]}")
    print()

    advisor_1 = llm(
        f"You are a contract advisor. A startup CTO said:\n\n"
        f"{client_1[:300]}\n\n"
        f"Given their situation, what are the top 3 clauses to negotiate? "
        f"Be specific about what to ask for.\n\n{CONTRACT}"
    )
    print(f"[Advisor]: {advisor_1[:300]}")
    print()

    client_2 = llm(
        f"You are the startup CTO. The advisor recommended:\n\n"
        f"{advisor_1[:300]}\n\n"
        f"In 1-2 sentences, tell them your #1 priority and what you'd "
        f"accept as a compromise."
    )
    print(f"[Client]: {client_2[:200]}")
    print()

    advisor_2 = llm(
        f"You are the advisor. Client's priority:\n\n{client_2[:200]}\n\n"
        f"Give 3 specific redline changes with fallback positions.\n\n{CONTRACT}"
    )
    print(f"[Advisor]: {advisor_2[:400]}")
    print()
    print("→ Iterative clarification produces targeted advice. The advisor")
    print("  focuses on the client's actual priority instead of generic")
    print("  contract boilerplate.")
    print()
    print("When to use: tasks where user context changes the answer.")
    print("When NOT to use: well-specified tasks with clear success criteria.")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("BenchFlow Scene Patterns — Contract Review Tutorial")
    print("One scenario, four patterns. Each fixes a failure of the previous.\n")

    r1 = pattern_1()
    r2 = pattern_2(r1)
    pattern_3()
    pattern_4()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  Pattern  │ Problem solved              │ Cost  │ Best for
  ─────────┼─────────────────────────────┼───────┼──────────────────
  1. Base   │ (baseline)                  │ 1x    │ simple tasks
  2. Multi  │ missed compound risks       │ 2x    │ self-checkable
     -turn  │                             │       │ tasks
  3. Multi  │ systematic blind spots      │ 3x    │ complex tasks
     -round │                             │       │ needing expertise
  4. Inter- │ generic ≠ actionable        │ 4x    │ user-specific
     active │                             │       │ decisions

Each pattern is a TrialConfig change — same API, same verifier,
same trajectory capture. No new runtime code needed.
""")
