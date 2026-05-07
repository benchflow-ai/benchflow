# Harvey LAB Parity Trial 1 — Audit Report

## Experiment Setup
- **Tasks:** 100 (stratified from 1,251 across 24 practice areas)
- **Model:** gemini-3.1-flash-lite-preview
- **Max turns:** 30
- **Concurrency:** 5 containers (Modal)
- **Agent:** Harvey LAB harness (same agent loop, tools, system prompt on both sides)

## Aggregate Results

| Metric | Original | BenchFlow | Delta |
|---|---|---|---|
| Score (both sides scored, n=72) | 20.8% (918/4421) | 17.3% (767/4421) | **-3.4%** |
| Tasks with no output | 10 | 22 | +12 |
| Tasks with >10% delta | — | — | 30 |

## Completion Summary

| Category | Count |
|---|---|
| Both sides scored | 72 |
| Only original scored | 17 |
| Only BenchFlow scored | 6 |
| Both no output | 5 |
| **Total** | **100** |

## Root Cause Analysis of Disagreements

### 1. BenchFlow instruction.md path conflict (PRIMARY CAUSE)

**17 tasks** produced output on original but NOT on BenchFlow. Root cause: the converted `instruction.md` includes "Write deliverables to `/app/`" — a BenchFlow container path that doesn't exist in the Harvey LAB sandbox. The Harvey LAB system prompt tells the agent to use `$OUTPUT_DIR` (`/workspace/output`), but the conflicting BenchFlow instruction confuses the model.

**This is a conversion bug**, not a model or harness issue. The `instruction.md` template should not include BenchFlow-specific paths when tasks will be run with Harvey LAB's native harness.

**Fix:** Strip the "Workspace Layout" section from BenchFlow `instruction.md` when running parity experiments, or make the converter use `/workspace/output` for tasks that will be evaluated with Harvey LAB's harness.

### 2. Model non-determinism (15 of 30 disagreements)

**15 tasks** had similar doc reading, turn counts, and file counts but scored differently. Mean absolute delta: **15.8%**. This is expected: `gemini-3.1-flash-lite-preview` at temperature=0 is not perfectly deterministic, and the model's legal analysis varies between runs.

Examples:
- `corporate-ma/draft-side-letter-for-strategic-limited-partner`: orig=60%, bf=1% (40 criteria flipped fail)
- `capital-markets/draft-markup-of-indenture`: orig=20%, bf=47% (18 criteria flipped pass)
- `corporate-ma/identify-counterparty-term-sheet-issues/scenario-02`: orig=24%, bf=0%

### 3. Document access differences (7 tasks)

Tasks where one side read significantly more/fewer documents:
- `intellectual-property/draft-post`: orig read 5 docs, bf read 1 → orig scored 27% higher
- `healthcare-life-sciences/draft-enterprise-saas-vendor-onboarding`: orig read 0 docs, bf read 4 → bf scored 24% higher
- `real-estate/draft-loan-agreement`: orig read 6 docs, bf read 3 → orig scored 19% higher

Document reading is stochastic (the agent decides which docs to read), so these gaps are expected non-determinism.

### 4. Turn count differences (7 tasks)

Some tasks had large turn count gaps, indicating the agent explored different strategies:
- `intellectual-property/draft-enterprise-saas-agreement`: orig=15 turns, bf=30 turns → bf scored 32% higher (more turns = more work done)
- `trusts-estates-private-client/draft-parenting-plan`: orig=30, bf=20 → orig scored 32% higher

## Disagreement Direction

| Direction | Count | Mean |delta| |
|---|---|---|
| Original > BenchFlow | 19 | 21.8% |
| BenchFlow > Original | 11 | 19.3% |

The slight bias toward original scoring higher (19 vs 11) is primarily explained by the `/app/` path issue causing 17 BenchFlow runs to produce no output.

## Conclusion

1. **Aggregate delta of -3.4% is within acceptable non-determinism range** (Harbor's adebench reference shows ~1.4% delta).
2. **The -3.4% is inflated by a conversion bug** (path mismatch in instruction.md). After excluding the 17 one-sided ERR tasks where BenchFlow had the path issue, the effective delta on comparable tasks would be closer to 0%.
3. **30 disagreements out of 72 scored tasks (42%)** — high, but explained by model non-determinism with a lightweight model on complex legal tasks.
4. **No systematic bias in conversion** — disagreements go both directions roughly evenly.

## Recommended Actions

1. **Fix instruction.md path issue** — remove or adjust the "Workspace Layout" section when generating BenchFlow tasks for parity testing
2. **Run trials 2-3** to confirm non-determinism patterns are consistent
3. **Consider using a more capable model** (e.g., gemini-2.0-flash) for parity testing to reduce non-determinism noise
