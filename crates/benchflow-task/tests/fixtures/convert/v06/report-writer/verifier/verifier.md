---
document_version: "0.3"
verifier:
  name: report-writer-verifier
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
    judge:
      type: llm-judge
      rubric: rubrics/report.md
---

## verifier intent

Checks the counts in the report, then grades its prose.
