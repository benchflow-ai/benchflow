# Skill Eval Rubric

Skill: `code-specialist`

Case: `optimize-quadratic-to-nlogn`

The verifier scores the agent from 0.0 to 1.0 using the rubric in `case.json` and the observed trajectory under `/logs/agent`.

Expected behavior:

- The agent produced an O(n) or O(n log n) solution
- The optimized function produces identical output to the original
- The agent explained the complexity improvement
- The agent handled duplicate pairs correctly
- The agent wrote tests verifying equivalence with the original
