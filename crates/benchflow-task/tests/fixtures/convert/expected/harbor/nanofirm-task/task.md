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

```toml task
[about]
author_name = "benchflow"
author_email = "xiangyi@benchflow.ai"
tags = ["multi-agent", "contract-review", "nanofirm"]

[agent]
timeout = "5m"

[verifier]
timeout = "1m"
mount = "/tests"

[oracle]
mount = "/solution"

[import.harbor]
version = "1.0"

[import.harbor.originals."task.toml"]
original = "sha256:0afbff42218e6f76e043040302f52639cd513ec866bf0ebdd2bae7949fce0232"
converted = "sha256:84af9f140fb4b77fc9b18d2bf070e79780be183d47719d5c6f1778a7aad1c4fa"

[import.harbor.originals."instruction.md"]
original = "sha256:5dbf9647bf99595d682562f9f876b77011a84045de5a7355119a9f29a45fbd42"
converted = "sha256:5dbf9647bf99595d682562f9f876b77011a84045de5a7355119a9f29a45fbd42"
```
