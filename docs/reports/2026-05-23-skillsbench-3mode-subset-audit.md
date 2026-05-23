# SkillsBench 9-Task Three-Mode Subset Audit

Generated: 2026-05-23 13:41:16

This is the complete 27-row trajectory/artifact audit for the 9-task SkillsBench subset across baseline, with-task-skills, and self-gen modes. It was produced from saved artifacts, not from a new run.

## Scope

- Run root: `/Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset`
- Modes: `baseline`, `with-task-skills`, `self-gen`
- Benchmark source: `benchflow-ai/skillsbench@main`
- Resolved source SHA in every row: `20149520474cfc8d7eb3c8000ec403d10145a9fd`
- Current `main` SHA verified during audit: `20149520474cfc8d7eb3c8000ec403d10145a9fd`
- Model/environment: `gemini-3.1-flash-lite-preview` / `daytona`
- Evidence caveat: the self-gen mode started at `2026-05-23 00:44:33 -0400` and summary finished before commit `4946481`. These self-gen rows do not validate the current ACP-native generated-skill handoff fix.

## Verdict

All 27 primary rows were audited. There were no retries in this subset (`max_retries: 0`). The subset is useful as pre-fix evidence, but it is not a release-quality validation of SkillsBench self-generation because generated solver skill packs were not persisted in artifacts and several solver trajectories did not activate generated skills. Only one row passed: `lake-warming-attribution` in `with-task-skills` mode.

## Mode Summaries

| Mode | Total | Passed | Failed | Errored | Verifier errored | Score | Model | Environment | Concurrency | Idle timeout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 9 | 0 | 8 | 0 | 1 | 0.0% | gemini-3.1-flash-lite-preview | daytona | 9 | 600 |
| with-task-skills | 9 | 1 | 6 | 1 | 1 | 11.1% | gemini-3.1-flash-lite-preview | daytona | 9 | 600 |
| self-gen | 9 | 0 | 8 | 0 | 1 | 0.0% | gemini-3.1-flash-lite-preview | daytona | 9 | 600 |

## Task Comparison Matrix

| Task | Baseline outcome/reward | With-task-skills outcome/reward | Self-gen outcome/reward | Self-gen audit note |
| --- | --- | --- | --- | --- |
| data-to-d3 | fail / 0.0 | fail / 0.0 | fail / 0.0 | CTRF 11/15; creator hit /app/generated-skills path write failure; no generated solver skill persisted |
| grid-dispatch-operator | fail / 0.0 | fail / 0.0 | fail / 0.0 | idle timeout 600s; only creator user message; CTRF 0/6; solver handoff did not complete |
| jax-computing-basics | fail / 0.0 | fail / 0.0 | fail / 0.0 | CTRF 4/5; generated skill activated, but stale evidence; numerical mismatch in jit_mlp |
| jpg-ocr-stat | fail / 0.0 | fail / 0.0 | fail / 0.0 | CTRF 0/1; skill created under /app/workspace/generated-skills, not solver mount |
| lake-warming-attribution | fail / 0.0 | pass / 1.0 | fail / 0.0 | CTRF 0/2; generated skill activated but had TODO-quality content; stale evidence |
| python-scala-translation | fail / 0.0 | fail / 0.0 | fail / 0.0 | solver did not show generated skill activation; /root/Tokenizer.scala missing |
| shock-analysis-supply | fail / 0.0 | fail / 0.0 | fail / 0.0 | CTRF 1/9; generated skill not activated; workbook failures remain |
| threejs-to-obj | verifier_error / n/a | verifier_error / n/a | verifier_error / n/a | CTRF 2/3; generated skill not persisted; same verifier exit-status issue |
| weighted-gdp-calc | fail / 0.0 | error / n/a | fail / 0.0 | CTRF 14/27; generated skill not activated; spreadsheet/stat failures remain |

## ACP-Native Self-Gen Findings

- Self-gen scenes point the solver at `/app/generated-skills`, but persisted host artifacts under `_self_gen` contain only copied `creator-skills/skill-creator`; no generated solver skill pack was found outside creator-skill scaffolding.
- Several self-gen solver trajectories never show generated skill activation. This is exactly the kind of issue the ACP-native handoff fix in `4946481` is meant to address, so this subset must be rerun before it can validate current architecture.
- Generated-skill artifact persistence is itself an audit gap: even if a solver activates a skill, the generated `SKILL.md` should be retained as first-class rollout evidence.

## Standards Gaps Against Release-Quality Adapter Evidence

- `threejs-to-obj` has reward and CTRF evidence for ordinary assertion failure, but BenchFlow records `verifier_error` because the verifier exits nonzero after failed tests.
- `python-scala-translation` uses custom verifier output without CTRF by task design, so dashboard/audit parity is weaker than the adapter tutorial standard.
- `weighted-gdp-calc` with task skills hit the ACP 400 token limit before verifier, creating a partial trajectory and no score.
- Token/cost telemetry remains unavailable across rows.

## Generated-Skill Artifact Check

| Check | Value |
| --- | --- |
| Creator skill SKILL.md files under _self_gen | 9 |
| Generated solver skill SKILL.md files persisted outside creator-skills | 0 |
| Retry rows found | 0 |

## Complete 27-Row Ledger

| Task | Mode | Outcome | Reward | Class | Tools | Partial trajectory | Trajectory | Scenes | Skills dirs | Verifier artifacts | Audit note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data-to-d3 | baseline | fail | 0.0 | model/verifier fail | 13 | False | present | 1 |  | ctrf.json: 10/15 pass, 5 fail, 0 skipped | CTRF 10/15; missing copied indiv-stock plus bubble render/tooltip/linking/legend failures |
| data-to-d3 | with-task-skills | fail | 0.0 | model/verifier fail | 12 | False | present | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/data-to-d3/environment/skills | ctrf.json: 8/15 pass, 7 fail, 0 skipped | CTRF 8/15; task skills present but chart/table/tooltip/linking failures remain |
| data-to-d3 | self-gen | fail | 0.0 | stale self-gen/dashboard mismatch | 33 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/data-to-d3-27b87141/creator-skills, /app/generated-skills | ctrf.json: 11/15 pass, 4 fail, 0 skipped | CTRF 11/15; creator hit /app/generated-skills path write failure; no generated solver skill persisted |
| grid-dispatch-operator | baseline | fail | 0.0 | model/verifier fail | 15 | False | present | 1 |  | ctrf.json: 4/6 pass, 2 fail, 0 skipped | CTRF 4/6; reserve constraint and optimality failures |
| grid-dispatch-operator | with-task-skills | fail | 0.0 | model/verifier fail | 12 | False | present | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/grid-dispatch-operator/environment/skills | ctrf.json: 5/6 pass, 1 fail, 0 skipped | CTRF 5/6; infeasible/invalid optimality result |
| grid-dispatch-operator | self-gen | fail | 0.0 | infra failure | 9 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/grid-dispatch-operator-6fc136e4/creator-skills, /app/generated-skills | ctrf.json: 0/6 pass, 6 fail, 0 skipped | idle timeout 600s; only creator user message; CTRF 0/6; solver handoff did not complete |
| jax-computing-basics | baseline | fail | 0.0 | model/verifier fail | 23 | False | present | 1 |  | ctrf.json: 4/5 pass, 1 fail, 0 skipped | CTRF 4/5; numerical mismatch in scan_rnn |
| jax-computing-basics | with-task-skills | fail | 0.0 | model/verifier fail | 12 | False | present | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/jax-computing-basics/environment/skills | ctrf.json: 4/5 pass, 1 fail, 0 skipped | CTRF 4/5; skill activated; numerical mismatch in jit_mlp |
| jax-computing-basics | self-gen | fail | 0.0 | model/verifier fail | 30 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/jax-computing-basics-92fd5389/creator-skills, /app/generated-skills | ctrf.json: 4/5 pass, 1 fail, 0 skipped | CTRF 4/5; generated skill activated, but stale evidence; numerical mismatch in jit_mlp |
| jpg-ocr-stat | baseline | fail | 0.0 | model/verifier fail | 14 | False | present | 1 |  | ctrf.json: 0/1 pass, 1 fail, 0 skipped | CTRF 0/1; OCR workbook content/null-handling mismatch |
| jpg-ocr-stat | with-task-skills | fail | 0.0 | model/verifier fail | 16 | False | present | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/jpg-ocr-stat/environment/skills | ctrf.json: 0/1 pass, 1 fail, 0 skipped | CTRF 0/1; OCR/xlsx skills present; workbook mismatch |
| jpg-ocr-stat | self-gen | fail | 0.0 | stale self-gen/dashboard mismatch | 27 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/jpg-ocr-stat-c1268f6a/creator-skills, /app/generated-skills | ctrf.json: 0/1 pass, 1 fail, 0 skipped | CTRF 0/1; skill created under /app/workspace/generated-skills, not solver mount |
| lake-warming-attribution | baseline | fail | 0.0 | model/verifier fail | 9 | False | present | 1 |  | ctrf.json: 0/2 pass, 2 fail, 0 skipped | CTRF 0/2; trend p-value missed threshold and dominant factor failed |
| lake-warming-attribution | with-task-skills | pass | 1.0 | pass | 15 | False | present | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/lake-warming-attribution/environment/skills | ctrf.json: 2/2 pass, 0 fail, 0 skipped | CTRF 2/2; only passing row in 27-row subset |
| lake-warming-attribution | self-gen | fail | 0.0 | model/verifier fail | 28 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/lake-warming-attribution-867b8434/creator-skills, /app/generated-skills | ctrf.json: 0/2 pass, 2 fail, 0 skipped | CTRF 0/2; generated skill activated but had TODO-quality content; stale evidence |
| python-scala-translation | baseline | fail | 0.0 | model/verifier fail | 7 | False | present | 1 |  | reward.txt: reward only; no CTRF | custom verifier log, no CTRF by task design; Scala compile failed |
| python-scala-translation | with-task-skills | fail | 0.0 | model/verifier fail | 17 | False | present | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/python-scala-translation/environment/skills | reward.txt: reward only; no CTRF | verifier says /root/Tokenizer.scala missing |
| python-scala-translation | self-gen | fail | 0.0 | stale self-gen/dashboard mismatch | 48 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/python-scala-translation-7b5470de/creator-skills, /app/generated-skills | reward.txt: reward only; no CTRF | solver did not show generated skill activation; /root/Tokenizer.scala missing |
| shock-analysis-supply | baseline | fail | 0.0 | model/verifier fail | 4 | False | present | 1 |  | ctrf.json: 1/9 pass, 8 fail, 0 skipped | CTRF 1/9; workbook mostly incomplete |
| shock-analysis-supply | with-task-skills | fail | 0.0 | model/verifier fail | 21 | False | present | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/shock-analysis-supply/environment/skills | ctrf.json: 1/9 pass, 8 fail, 0 skipped | CTRF 1/9; xlsx skill present but workbook failures remain |
| shock-analysis-supply | self-gen | fail | 0.0 | stale self-gen/dashboard mismatch | 27 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/shock-analysis-supply-569aa7cf/creator-skills, /app/generated-skills | ctrf.json: 1/9 pass, 8 fail, 0 skipped | CTRF 1/9; generated skill not activated; workbook failures remain |
| threejs-to-obj | baseline | verifier_error |  | verifier/task issue | 13 | False | present | 1 |  | ctrf.json: 2/3 pass, 1 fail, 0 skipped | CTRF 2/3 and reward.txt=0, but verifier rc=1 made BenchFlow record verifier_error |
| threejs-to-obj | with-task-skills | verifier_error |  | verifier/task issue | 17 | False | present | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/threejs-to-obj/environment/skills | ctrf.json: 2/3 pass, 1 fail, 0 skipped | CTRF 2/3; Chamfer distance mismatch; same verifier exit-status issue |
| threejs-to-obj | self-gen | verifier_error |  | verifier/task issue | 20 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/threejs-to-obj-6d9f3cb3/creator-skills, /app/generated-skills | ctrf.json: 2/3 pass, 1 fail, 0 skipped | CTRF 2/3; generated skill not persisted; same verifier exit-status issue |
| weighted-gdp-calc | baseline | fail | 0.0 | model/verifier fail | 19 | False | present | 1 |  | ctrf.json: 23/27 pass, 4 fail, 0 skipped | CTRF 23/27; percentile and weighted-mean calculations wrong/missing |
| weighted-gdp-calc | with-task-skills | error |  | infra failure | 10 | True | present-partial | 1 | /Users/lixiangyi/.codex/worktrees/v05-integration-merge-20260522114632/.cache/datasets/benchflow-ai/skillsbench__snapshots/20149520474cfc8d7eb3c8000ec403d10145a9fd/tasks/weighted-gdp-calc/environment/skills | none: no verifier artifacts found | ACP 400 token limit exceeded before verifier; partial ACP trajectory |
| weighted-gdp-calc | self-gen | fail | 0.0 | stale self-gen/dashboard mismatch | 99 | False | present | 2 | jobs/e2e/20260523-003051-skillsbench-3modes-gemini31-flash-lite-daytona-release-subset/self-gen/_self_gen/weighted-gdp-calc-adea9b4a/creator-skills, /app/generated-skills | ctrf.json: 14/27 pass, 13 fail, 0 skipped | CTRF 14/27; generated skill not activated; spreadsheet/stat failures remain |
