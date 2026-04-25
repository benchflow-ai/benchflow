#!/bin/bash
cat > /app/analysis.json << 'JSON'
{
  "risks": [
    {"clause": "Section 2", "severity": "HIGH", "issue": "Auto-renewal with 90-day notice locks client into unwanted extensions", "recommendation": "Reduce notice period to 30 days or add opt-out after initial term"},
    {"clause": "Section 3", "severity": "HIGH", "issue": "Renewal pricing uncapped — vendor can raise prices arbitrarily", "recommendation": "Cap annual increases at CPI + 3% or prior-year rate + 5%"},
    {"clause": "Section 4", "severity": "HIGH", "issue": "30-day data export window is too short for migration", "recommendation": "Extend to 90 days minimum, require machine-readable export format"},
    {"clause": "Section 5", "severity": "MEDIUM", "issue": "No SLA credits or remedies for downtime", "recommendation": "Add service credits: 5% per 0.1% below 99.5%"},
    {"clause": "Section 7", "severity": "HIGH", "issue": "One-sided indemnification — no IP infringement protection for licensee", "recommendation": "Add mutual indemnification for IP claims"},
    {"clause": "Section 8", "severity": "MEDIUM", "issue": "Acceleration clause — all remaining fees due on termination for non-payment", "recommendation": "Remove acceleration or cap at current quarter fees"}
  ],
  "compound_risks": [
    {"clauses": ["Section 2", "Section 3"], "severity": "HIGH", "issue": "Auto-renewal + uncapped pricing + 90-day notice = vendor lock-in trap. Client misses the narrow cancellation window and faces arbitrary price increase with no negotiation leverage.", "recommendation": "Negotiate all three together: shorter notice (30 days), price cap (CPI+3%), and explicit opt-out right after Year 3"}
  ],
  "deal_breakers": [
    "Uncapped renewal pricing (Section 3) — must have a price escalation cap",
    "No IP indemnification (Section 7) — unacceptable risk for a startup"
  ],
  "summary": "This contract has a vendor lock-in trap: auto-renewal with uncapped pricing and a 90-day notice window means the client can be locked into arbitrary price increases. The one-sided indemnification and short data export window compound the risk. Recommend negotiating price caps, mutual indemnification, and extended data portability before signing."
}
JSON
