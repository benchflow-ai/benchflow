# Contract Risk Analysis

You are reviewing a vendor SaaS contract for a Series A startup (20 employees, $3M ARR). The contract is at /app/contract.md.

Produce a risk analysis at /app/analysis.json with this structure:
```json
{
  "risks": [
    {
      "clause": "Section number",
      "severity": "HIGH|MEDIUM|LOW",
      "issue": "One-line description",
      "recommendation": "Specific change to request"
    }
  ],
  "compound_risks": [
    {
      "clauses": ["Section numbers involved"],
      "severity": "HIGH|MEDIUM|LOW",
      "issue": "How these clauses interact to create a worse situation",
      "recommendation": "What to negotiate"
    }
  ],
  "deal_breakers": ["List of terms that should block signing if unchanged"],
  "summary": "2-3 sentence executive summary"
}
```

Requirements:
- Identify at least 3 individual clause risks
- Identify at least 1 compound risk (clauses that interact)
- Include at least 1 deal-breaker
- All recommendations must be specific (not "negotiate better terms")
