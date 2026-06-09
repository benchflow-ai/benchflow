# Skill Eval Rubric

Skill: `code-specialist`

Case: `topo-sort-with-cycle-detection`

The verifier scores the agent from 0.0 to 1.0 using the rubric in `case.json` and the observed trajectory under `/logs/agent`.

Expected behavior:

- The agent produced a working topo_sort function
- The function correctly detects cycles and raises ValueError
- The function handles edge cases (empty graph, single node)
- The agent wrote tests verifying the implementation
- The implementation is deterministic (alphabetical tiebreaking)
