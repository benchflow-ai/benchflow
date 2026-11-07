---
verifier:
  strategies:
    judge:
      type: llm-judge
      rubric: judge.toml
      model: openai/mimo-v2.6-flash
---

MiMo V2.6 Flash reads `/app/poem.txt` and answers the two pass/fail questions
in `judge.toml`. The reward is 1 only when both pass.
