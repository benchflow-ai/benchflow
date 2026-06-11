# SkillsBench Full 94-Task Baseline Audit

Generated: 2026-05-23 13:41:16

This is a durable per-task audit of the full SkillsBench baseline run. It was produced from the saved BenchFlow artifacts, not from a new run.

## Scope

- Run root: `jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64`
- Summary: `jobs/e2e/20260522-165420-skillsbench-all-gemini31-flash-lite-daytona-c64/summary.json`
- Benchmark source: `benchflow-ai/skillsbench@main`
- Resolved source SHA in artifacts: `20149520474cfc8d7eb3c8000ec403d10145a9fd`
- Current `main` SHA verified during audit: `20149520474cfc8d7eb3c8000ec403d10145a9fd`
- Agent/model/environment: `gemini` / `gemini-3.1-flash-lite-preview` / `daytona`
- Concurrency and idle timeout: `64` / `600s`
- BenchFlow fix commit after this evidence: `4946481`

## Verdict

The full baseline did execute the latest SkillsBench `main` SHA now visible from GitHub, and the summary is internally consistent: 94 unique tasks, 109 rollout rows including retries, final summary 8 passed / 76 failed / 7 errored / 3 verifier_errored. It is not a clean release-quality validation set because retry rows include no-reward infra failures and three final verifier errors appear to be verifier-contract bugs rather than true missing-evidence cases.

## Aggregate Checks

| Metric | Value |
| --- | --- |
| Unique tasks | 94 |
| Rollout rows / retries | 109 |
| Task-final outcomes | error=7, fail=76, pass=8, verifier_error=3 |
| Retry-row outcomes | error=16, fail=80, pass=8, verifier_error=5 |
| Summary score | 8.5% |
| Score excluding errors | 9.5% |
| Token/cost telemetry | tokens=0, cost=0, coverage=0.0 |

## Pass List

| Task | Final rollout | Tool calls | Trajectory |
| --- | --- | --- | --- |
| 3d-scan-calc | 3d-scan-calc__41391d68 | 11 | present |
| citation-check | citation-check__a62a53c2 | 20 | present |
| econ-detrending-correlation | econ-detrending-correlation__21386bf8 | 16 | present |
| mars-clouds-clustering | mars-clouds-clustering__0d36d57e | 12 | present |
| parallel-tfidf-search | parallel-tfidf-search__67cba0d7 | 25 | present |
| pddl-tpp-planning | pddl-tpp-planning__785326e4 | 13 | present |
| radar-vital-signs | radar-vital-signs__45b21344 | 11 | present |
| spring-boot-jakarta-migration | spring-boot-jakarta-migration__2382b67d | 101 | present |

## P0 Invalid Measurement Blockers

| Blocker | Retry rows | Tasks | Affected tasks |
| --- | --- | --- | --- |
| Agent install failed | 9 | 3 | fix-visual-stability, gh-repo-analytics, pedestrian-traffic-counting |
| Artifact copy/setup failed | 2 | 2 | drone-planning-control, pg-essay-to-audiobook |
| ACP transport/stdout closed | 5 | 3 | mars-clouds-clustering, quantum-numerical-simulation, video-filler-word-remover |
| Verifier timeout | 2 | 1 | quantum-numerical-simulation |
| Verifier crash/contract refusal | 3 | 3 | dynamic-object-aware-egomotion, threejs-structure-parser, threejs-to-obj |
| Idle timeout with reward 0 diagnostics gap | 5 | 3 | court-form-filling, shock-analysis-supply, video-tutorial-indexer |

## Final Classification Counts

| Class | Task count |
| --- | --- |
| invalid: ACP transport/stdout closed | 2 |
| invalid: agent install failure | 3 |
| invalid: artifact copy/setup failure | 2 |
| pass | 8 |
| valid model failure: missing/incorrect artifact | 9 |
| valid model failure: reward only/no structured detail | 5 |
| valid model failure: verifier assertion/quality threshold | 62 |
| verifier contract/artifact error | 3 |

## Retry Reconciliation

Final task status is taken from the latest chronological rollout for that task. Summary semantics are `verifier_error` first, then reward, then agent error. This preserves BenchFlow's summary counts while exposing invalid retry rows for release gating.

| Task | Attempts | Attempt outcomes by rollout hash |
| --- | --- | --- |
| court-form-filling | 2 | 4d161ed7:fail, f5fed20d:fail |
| fix-visual-stability | 3 | f44ddba9:error, 371cbee3:error, da2e2eaa:error |
| gh-repo-analytics | 3 | 7e9bee41:error, 6e43ee2e:error, a40ea886:error |
| mars-clouds-clustering | 2 | 387d89ca:error, 0d36d57e:pass |
| pedestrian-traffic-counting | 3 | 87121f5c:error, 04c45e9c:error, 4daa7b37:error |
| quantum-numerical-simulation | 3 | 3031c9cf:verifier_error, a43c7238:verifier_error, 32f8a92b:error |
| shock-analysis-supply | 2 | d95da6be:fail, 78b391ae:fail |
| video-filler-word-remover | 3 | 4b9e74b6:error, bc926eda:error, 4f64d2f0:error |
| video-tutorial-indexer | 3 | 7fe67e9d:fail, fb0bedbb:fail, f1b368f8:fail |

## Standards Gaps Against Release-Quality Adapter Evidence

- Three final tasks wrote reward/verifier artifacts but were counted as `verifier_error` because the verifier process exited nonzero after normal assertion failures: `dynamic-object-aware-egomotion`, `threejs-structure-parser`, and `threejs-to-obj`.
- Several tasks rely on reward-only or task-specific logs instead of consistent CTRF-style structured failure details, which weakens dashboard and audit parity with the adapter tutorial standard.
- No token/cost telemetry was captured: all aggregate usage fields are zero or unavailable.
- At least 21 retry rows are invalid no-reward infra/verifier measurements and must not be used as clean agent-quality evidence.

## Final Per-Task Ledger

| Task | Attempts | Final outcome | Reward | Class | Tools | Trajectory | Verifier artifacts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3d-scan-calc | 1 | pass | 1.0 | pass | 11 | present | ctrf.json: 2/2 pass, 0 fail, 0 skipped | final accepted pass |
| ada-bathroom-plan-repair | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 43 | present | ctrf.json: 0/6 pass, 6 fail, 0 skipped | agent completed; verifier rejected output |
| adaptive-cruise-control | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 15 | present | ctrf.json: 6/12 pass, 6 fail, 0 skipped | agent completed; verifier rejected output |
| azure-bgp-oscillation-route-leak | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 5 | present | ctrf.json: 3/4 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| bike-rebalance | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 8 | present | ctrf.json: 19/20 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| citation-check | 1 | pass | 1.0 | pass | 20 | present | ctrf.json: 9/9 pass, 0 fail, 0 skipped | final accepted pass |
| civ6-adjacency-optimizer | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 2 | present | ctrf.json: 0/10 pass, 3 fail, 7 skipped | agent completed; verifier rejected output |
| court-form-filling | 2 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 11 | present | ctrf.json: 2/5 pass, 1 fail, 2 skipped | 2 attempts; idle timeout row still produced reward 0; diagnostics need improvement |
| crystallographic-wyckoff-position-analysis | 1 | fail | 0.91 | valid model failure: verifier assertion/quality threshold | 9 | present | reward.txt: reward only; no CTRF | agent completed; verifier rejected output |
| dapt-intrusion-detection | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 8 | present | ctrf.json: 11/14 pass, 3 fail, 0 skipped | agent completed; verifier rejected output |
| data-to-d3 | 1 | fail | 0.0 | valid model failure: missing/incorrect artifact | 15 | present | ctrf.json: 12/15 pass, 3 fail, 0 skipped | required output artifact wrong or missing |
| debug-trl-grpo | 1 | fail | 0.25 | valid model failure: verifier assertion/quality threshold | 32 | present | reward.txt: reward only; no CTRF | agent completed; verifier rejected output |
| dialogue-parser | 1 | fail | 0.667 | valid model failure: verifier assertion/quality threshold | 19 | present | ctrf.json: 4/6 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| drone-planning-control | 1 | error |  | invalid: artifact copy/setup failure | 0 | missing | none: no verifier artifacts found | historical /app/skills upload/setup failure; rerun needed after upload fix |
| dynamic-object-aware-egomotion | 1 | verifier_error |  | verifier contract/artifact error | 11 | present | ctrf.json: 8/10 pass, 2 fail, 0 skipped | reward artifact says 0 but verifier rc=1 made BenchFlow classify verifier_error |
| earthquake-phase-association | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 13 | present | ctrf.json: 0/1 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| earthquake-plate-calculation | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 14 | present | ctrf.json: 7/8 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| econ-detrending-correlation | 1 | pass | 1.0 | pass | 16 | present | ctrf.json: 4/4 pass, 0 fail, 0 skipped | final accepted pass |
| edit-pdf | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 8 | present | ctrf.json: 0/9 pass, 9 fail, 0 skipped | agent completed; verifier rejected output |
| energy-ac-optimal-power-flow | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 10 | present | ctrf.json: 12/24 pass, 11 fail, 1 skipped | agent completed; verifier rejected output |
| energy-market-pricing | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 20 | present | ctrf.json: 2/4 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| energy-unit-commitment | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 26 | present | ctrf.json: 9/14 pass, 5 fail, 0 skipped | agent completed; verifier rejected output |
| enterprise-information-search | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 26 | present | ctrf.json: 1/3 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| exam-block-sequencing | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 18 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| exceltable-in-ppt | 1 | fail | 0.0 | valid model failure: missing/incorrect artifact | 15 | present | ctrf.json: 6/8 pass, 2 fail, 0 skipped | required output artifact wrong or missing |
| exoplanet-detection-period | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 17 | present | ctrf.json: 3/4 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| financial-modeling-qa | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 21 | present | ctrf.json: 3/4 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| find-topk-similiar-chemicals | 1 | fail | 0.0 | valid model failure: reward only/no structured detail | 13 | present | reward.txt: reward only; no CTRF | reward present, structured failure details incomplete or task-specific |
| fix-build-agentops | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 42 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| fix-build-google-auto | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 50 | present | ctrf.json: 0/3 pass, 3 fail, 0 skipped | agent completed; verifier rejected output |
| fix-druid-loophole-cve | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 67 | present | ctrf.json: 3/4 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| fix-erlang-ssh-cve | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 28 | present | ctrf.json: 0/3 pass, 3 fail, 0 skipped | agent completed; verifier rejected output |
| fix-visual-stability | 3 | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found | 3 attempts; historical JS ACP install pipefail row; rerun needed after fix |
| flink-query | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 22 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| flood-risk-analysis | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 14 | present | ctrf.json: 1/2 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| gh-repo-analytics | 3 | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found | 3 attempts; historical JS ACP install pipefail row; rerun needed after fix |
| glm-lake-mendota | 1 | fail | 0.0 | valid model failure: reward only/no structured detail | 70 | present | reward.txt: reward only; no CTRF | reward present, structured failure details incomplete or task-specific |
| gravitational-wave-detection | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 13 | present | ctrf.json: 8/9 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| grid-dispatch-operator | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 20 | present | ctrf.json: 2/6 pass, 4 fail, 0 skipped | agent completed; verifier rejected output |
| hvac-control | 1 | fail | 0.0 | valid model failure: reward only/no structured detail | 18 | present | ctrf-report.json: 6/7 pass, 1 fail, 0 skipped | reward present, structured failure details incomplete or task-specific |
| invoice-fraud-detection | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 19 | present | ctrf.json: 1/2 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| jax-computing-basics | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 13 | present | ctrf.json: 4/5 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| jpg-ocr-stat | 1 | fail | 0.0 | valid model failure: missing/incorrect artifact | 22 | present | ctrf.json: 0/1 pass, 1 fail, 0 skipped | required output artifact wrong or missing |
| lab-unit-harmonization | 1 | fail | 0.312 | valid model failure: verifier assertion/quality threshold | 10 | present | ctrf.json: 15/48 pass, 33 fail, 0 skipped | agent completed; verifier rejected output |
| lake-warming-attribution | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 13 | present | ctrf.json: 0/2 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| latex-formula-extraction | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 6 | present | ctrf.json: 4/7 pass, 3 fail, 0 skipped | agent completed; verifier rejected output |
| lean4-proof | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 60 | present | ctrf.json: 2/4 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| llm-prefix-cache-replay | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 7 | present | ctrf.json: 2/6 pass, 4 fail, 0 skipped | agent completed; verifier rejected output |
| manufacturing-codebook-normalization | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 9 | present | ctrf.json: 13/16 pass, 3 fail, 0 skipped | agent completed; verifier rejected output |
| manufacturing-equipment-maintenance | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 31 | present | ctrf.json: 3/7 pass, 4 fail, 0 skipped | agent completed; verifier rejected output |
| manufacturing-fjsp-optimization | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 31 | present | ctrf.json: 13/15 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| mario-coin-counting | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 6 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| mars-clouds-clustering | 2 | pass | 1.0 | pass | 12 | present | ctrf.json: 8/8 pass, 0 fail, 0 skipped | 2 attempts; rc=255/stdout close seen in retry set; final accepted pass |
| multilingual-video-dubbing | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 15 | present | ctrf.json: 0/8 pass, 8 fail, 0 skipped | agent completed; verifier rejected output |
| offer-letter-generator | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 6 | present | ctrf.json: 2/4 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| organize-messy-files | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 35 | present | ctrf.json: 4/6 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| paper-anonymizer | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 16 | present | ctrf.json: 5/6 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| parallel-tfidf-search | 1 | pass | 1.0 | pass | 25 | present | ctrf.json: 5/5 pass, 0 fail, 0 skipped | final accepted pass |
| paratransit-routing | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 15 | present | ctrf.json: 4/7 pass, 3 fail, 0 skipped | agent completed; verifier rejected output |
| pddl-airport-planning | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 16 | present | ctrf.json: 1/2 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| pddl-tpp-planning | 1 | pass | 1.0 | pass | 13 | present | ctrf.json: 2/2 pass, 0 fail, 0 skipped | final accepted pass |
| pdf-excel-diff | 1 | fail | 0.0 | valid model failure: missing/incorrect artifact | 15 | present | ctrf.json: 6/11 pass, 5 fail, 0 skipped | required output artifact wrong or missing |
| pedestrian-traffic-counting | 3 | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found | 3 attempts; historical JS ACP install pipefail row; rerun needed after fix |
| pg-essay-to-audiobook | 1 | error |  | invalid: artifact copy/setup failure | 0 | missing | none: no verifier artifacts found | historical /app/skills upload/setup failure; rerun needed after upload fix |
| powerlifting-coef-calc | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 8 | present | ctrf.json: 9/11 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| pptx-reference-formatting | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 14 | present | ctrf.json: 8/12 pass, 4 fail, 0 skipped | agent completed; verifier rejected output |
| protein-expression-analysis | 1 | fail | 0.0 | valid model failure: reward only/no structured detail | 12 | present | reward.txt: reward only; no CTRF | reward present, structured failure details incomplete or task-specific |
| python-scala-translation | 1 | fail | 0.0 | valid model failure: missing/incorrect artifact | 9 | present | reward.txt: reward only; no CTRF | required output artifact wrong or missing |
| quantum-numerical-simulation | 3 | error |  | invalid: ACP transport/stdout closed | 6 | present-partial | none: no verifier artifacts found | 3 attempts; rc=255/stdout close seen in retry set; verifier hit 240s timeout in retries |
| r2r-mpc-control | 1 | fail | 0.0 | valid model failure: reward only/no structured detail | 43 | present | ctrf-report.json: 5/6 pass, 1 fail, 0 skipped | reward present, structured failure details incomplete or task-specific |
| radar-vital-signs | 1 | pass | 1.0 | pass | 11 | present | ctrf.json: 5/5 pass, 0 fail, 0 skipped | final accepted pass |
| react-performance-debugging | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 24 | present | reward.txt: reward only; no CTRF | agent completed; verifier rejected output |
| reserves-at-risk-calc | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 31 | present | ctrf.json: 1/5 pass, 4 fail, 0 skipped | agent completed; verifier rejected output |
| sales-pivot-analysis | 1 | fail | 0.0 | valid model failure: missing/incorrect artifact | 13 | present | ctrf.json: 6/10 pass, 3 fail, 1 skipped | required output artifact wrong or missing |
| sec-financial-report | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 26 | present | ctrf.json: 1/2 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| seismic-phase-picking | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 35 | present | ctrf.json: 0/2 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| setup-fuzzing-py | 1 | fail | 0.16 | valid model failure: verifier assertion/quality threshold | 15 | present | ctrf.json: 2/2 pass, 0 fail, 0 skipped | agent completed; verifier rejected output |
| shock-analysis-demand | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 4 | present | ctrf.json: 2/5 pass, 3 fail, 0 skipped | agent completed; verifier rejected output |
| shock-analysis-supply | 2 | fail | 0.0 | valid model failure: missing/incorrect artifact | 11 | present | ctrf.json: 2/9 pass, 7 fail, 0 skipped | 2 attempts; idle timeout row still produced reward 0; diagnostics need improvement |
| simpo-code-reproduction | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 69 | present | reward.txt: reward only; no CTRF | agent completed; verifier rejected output |
| software-dependency-audit | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 10 | present | ctrf.json: 2/4 pass, 2 fail, 0 skipped | agent completed; verifier rejected output |
| spring-boot-jakarta-migration | 1 | pass | 1.0 | pass | 101 | present | ctrf.json: 10/10 pass, 0 fail, 0 skipped | final accepted pass |
| suricata-custom-exfil | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 38 | present | reward.txt: reward only; no CTRF | agent completed; verifier rejected output |
| syzkaller-ppdev-syzlang | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 16 | present | ctrf.json: 1/7 pass, 6 fail, 0 skipped | agent completed; verifier rejected output |
| taxonomy-tree-merge | 1 | fail | 0.6962 | valid model failure: verifier assertion/quality threshold | 11 | present | ctrf.json: 15/22 pass, 7 fail, 0 skipped | agent completed; verifier rejected output |
| threejs-structure-parser | 1 | verifier_error |  | verifier contract/artifact error | 6 | present | ctrf.json: 1/3 pass, 2 fail, 0 skipped | reward artifact says 0 but verifier rc=1 made BenchFlow classify verifier_error |
| threejs-to-obj | 1 | verifier_error |  | verifier contract/artifact error | 8 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped | reward artifact says 0 but verifier rc=1 made BenchFlow classify verifier_error |
| tictoc-unnecessary-abort-detection | 1 | fail | 0.1 | valid model failure: verifier assertion/quality threshold | 17 | present | ctrf.json: 3/3 pass, 0 fail, 0 skipped | agent completed; verifier rejected output |
| travel-planning | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 11 | present | ctrf.json: 9/10 pass, 1 fail, 0 skipped | agent completed; verifier rejected output |
| video-filler-word-remover | 3 | error |  | invalid: ACP transport/stdout closed | 1 | present-partial | none: no verifier artifacts found | 3 attempts; rc=255/stdout close seen in retry set |
| video-silence-remover | 1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 30 | present | ctrf.json: 5/9 pass, 4 fail, 0 skipped | agent completed; verifier rejected output |
| video-tutorial-indexer | 3 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 12 | present-partial | ctrf.json: 0/2 pass, 2 fail, 0 skipped | 3 attempts; idle timeout row still produced reward 0; diagnostics need improvement |
| weighted-gdp-calc | 1 | fail | 0.0 | valid model failure: missing/incorrect artifact | 7 | present | ctrf.json: 14/27 pass, 13 fail, 0 skipped | required output artifact wrong or missing |
| xlsx-recover-data | 1 | fail | 0.0 | valid model failure: missing/incorrect artifact | 22 | present | ctrf.json: 1/8 pass, 7 fail, 0 skipped | required output artifact wrong or missing |

## Rollout Row Ledger

| Task | Rollout | Outcome | Reward | Class | Tools | Trajectory | Verifier artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3d-scan-calc | 3d-scan-calc__41391d68 | pass | 1.0 | pass | 11 | present | ctrf.json: 2/2 pass, 0 fail, 0 skipped |
| ada-bathroom-plan-repair | ada-bathroom-plan-repair__c22b215b | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 43 | present | ctrf.json: 0/6 pass, 6 fail, 0 skipped |
| adaptive-cruise-control | adaptive-cruise-control__246ffab4 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 15 | present | ctrf.json: 6/12 pass, 6 fail, 0 skipped |
| azure-bgp-oscillation-route-leak | azure-bgp-oscillation-route-leak__249d5823 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 5 | present | ctrf.json: 3/4 pass, 1 fail, 0 skipped |
| bike-rebalance | bike-rebalance__5d63d4b1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 8 | present | ctrf.json: 19/20 pass, 1 fail, 0 skipped |
| citation-check | citation-check__a62a53c2 | pass | 1.0 | pass | 20 | present | ctrf.json: 9/9 pass, 0 fail, 0 skipped |
| civ6-adjacency-optimizer | civ6-adjacency-optimizer__8346496c | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 2 | present | ctrf.json: 0/10 pass, 3 fail, 7 skipped |
| court-form-filling | court-form-filling__4d161ed7 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 1 | present-partial | ctrf.json: 0/5 pass, 5 fail, 0 skipped |
| court-form-filling | court-form-filling__f5fed20d | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 11 | present | ctrf.json: 2/5 pass, 1 fail, 2 skipped |
| crystallographic-wyckoff-position-analysis | crystallographic-wyckoff-position-analysis__42671fe2 | fail | 0.91 | valid model failure: verifier assertion/quality threshold | 9 | present | reward.txt: reward only; no CTRF |
| dapt-intrusion-detection | dapt-intrusion-detection__ed48011e | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 8 | present | ctrf.json: 11/14 pass, 3 fail, 0 skipped |
| data-to-d3 | data-to-d3__1b008b61 | fail | 0.0 | valid model failure: missing/incorrect artifact | 15 | present | ctrf.json: 12/15 pass, 3 fail, 0 skipped |
| debug-trl-grpo | debug-trl-grpo__54fced2d | fail | 0.25 | valid model failure: verifier assertion/quality threshold | 32 | present | reward.txt: reward only; no CTRF |
| dialogue-parser | dialogue-parser__bd2a954a | fail | 0.667 | valid model failure: verifier assertion/quality threshold | 19 | present | ctrf.json: 4/6 pass, 2 fail, 0 skipped |
| drone-planning-control | drone-planning-control__67030ee5 | error |  | invalid: artifact copy/setup failure | 0 | missing | none: no verifier artifacts found |
| dynamic-object-aware-egomotion | dynamic-object-aware-egomotion__c905a992 | verifier_error |  | verifier contract/artifact error | 11 | present | ctrf.json: 8/10 pass, 2 fail, 0 skipped |
| earthquake-phase-association | earthquake-phase-association__4accf70e | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 13 | present | ctrf.json: 0/1 pass, 1 fail, 0 skipped |
| earthquake-plate-calculation | earthquake-plate-calculation__7f006690 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 14 | present | ctrf.json: 7/8 pass, 1 fail, 0 skipped |
| econ-detrending-correlation | econ-detrending-correlation__21386bf8 | pass | 1.0 | pass | 16 | present | ctrf.json: 4/4 pass, 0 fail, 0 skipped |
| edit-pdf | edit-pdf__7f298015 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 8 | present | ctrf.json: 0/9 pass, 9 fail, 0 skipped |
| energy-ac-optimal-power-flow | energy-ac-optimal-power-flow__7c6e0167 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 10 | present | ctrf.json: 12/24 pass, 11 fail, 1 skipped |
| energy-market-pricing | energy-market-pricing__975ad1b3 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 20 | present | ctrf.json: 2/4 pass, 2 fail, 0 skipped |
| energy-unit-commitment | energy-unit-commitment__f5da9b1e | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 26 | present | ctrf.json: 9/14 pass, 5 fail, 0 skipped |
| enterprise-information-search | enterprise-information-search__ec530a1c | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 26 | present | ctrf.json: 1/3 pass, 2 fail, 0 skipped |
| exam-block-sequencing | exam-block-sequencing__c155d345 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 18 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped |
| exceltable-in-ppt | exceltable-in-ppt__9bfd48fd | fail | 0.0 | valid model failure: missing/incorrect artifact | 15 | present | ctrf.json: 6/8 pass, 2 fail, 0 skipped |
| exoplanet-detection-period | exoplanet-detection-period__fea4ca91 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 17 | present | ctrf.json: 3/4 pass, 1 fail, 0 skipped |
| financial-modeling-qa | financial-modeling-qa__0fe80c20 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 21 | present | ctrf.json: 3/4 pass, 1 fail, 0 skipped |
| find-topk-similiar-chemicals | find-topk-similiar-chemicals__efe101d9 | fail | 0.0 | valid model failure: reward only/no structured detail | 13 | present | reward.txt: reward only; no CTRF |
| fix-build-agentops | fix-build-agentops__668a8ca7 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 42 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped |
| fix-build-google-auto | fix-build-google-auto__11e51550 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 50 | present | ctrf.json: 0/3 pass, 3 fail, 0 skipped |
| fix-druid-loophole-cve | fix-druid-loophole-cve__459daafd | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 67 | present | ctrf.json: 3/4 pass, 1 fail, 0 skipped |
| fix-erlang-ssh-cve | fix-erlang-ssh-cve__edaf079b | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 28 | present | ctrf.json: 0/3 pass, 3 fail, 0 skipped |
| fix-visual-stability | fix-visual-stability__f44ddba9 | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| fix-visual-stability | fix-visual-stability__371cbee3 | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| fix-visual-stability | fix-visual-stability__da2e2eaa | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| flink-query | flink-query__10b16917 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 22 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped |
| flood-risk-analysis | flood-risk-analysis__832bc2bf | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 14 | present | ctrf.json: 1/2 pass, 1 fail, 0 skipped |
| gh-repo-analytics | gh-repo-analytics__7e9bee41 | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| gh-repo-analytics | gh-repo-analytics__6e43ee2e | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| gh-repo-analytics | gh-repo-analytics__a40ea886 | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| glm-lake-mendota | glm-lake-mendota__78fd070b | fail | 0.0 | valid model failure: reward only/no structured detail | 70 | present | reward.txt: reward only; no CTRF |
| gravitational-wave-detection | gravitational-wave-detection__0c3773ce | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 13 | present | ctrf.json: 8/9 pass, 1 fail, 0 skipped |
| grid-dispatch-operator | grid-dispatch-operator__7b781233 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 20 | present | ctrf.json: 2/6 pass, 4 fail, 0 skipped |
| hvac-control | hvac-control__9b5f745a | fail | 0.0 | valid model failure: reward only/no structured detail | 18 | present | ctrf-report.json: 6/7 pass, 1 fail, 0 skipped |
| invoice-fraud-detection | invoice-fraud-detection__834dca0f | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 19 | present | ctrf.json: 1/2 pass, 1 fail, 0 skipped |
| jax-computing-basics | jax-computing-basics__4eab7e57 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 13 | present | ctrf.json: 4/5 pass, 1 fail, 0 skipped |
| jpg-ocr-stat | jpg-ocr-stat__7e726840 | fail | 0.0 | valid model failure: missing/incorrect artifact | 22 | present | ctrf.json: 0/1 pass, 1 fail, 0 skipped |
| lab-unit-harmonization | lab-unit-harmonization__2669d398 | fail | 0.312 | valid model failure: verifier assertion/quality threshold | 10 | present | ctrf.json: 15/48 pass, 33 fail, 0 skipped |
| lake-warming-attribution | lake-warming-attribution__04583e77 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 13 | present | ctrf.json: 0/2 pass, 2 fail, 0 skipped |
| latex-formula-extraction | latex-formula-extraction__9ee9a39e | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 6 | present | ctrf.json: 4/7 pass, 3 fail, 0 skipped |
| lean4-proof | lean4-proof__187bc290 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 60 | present | ctrf.json: 2/4 pass, 2 fail, 0 skipped |
| llm-prefix-cache-replay | llm-prefix-cache-replay__9fc3c18a | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 7 | present | ctrf.json: 2/6 pass, 4 fail, 0 skipped |
| manufacturing-codebook-normalization | manufacturing-codebook-normalization__de94e7b5 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 9 | present | ctrf.json: 13/16 pass, 3 fail, 0 skipped |
| manufacturing-equipment-maintenance | manufacturing-equipment-maintenance__e38cbcd2 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 31 | present | ctrf.json: 3/7 pass, 4 fail, 0 skipped |
| manufacturing-fjsp-optimization | manufacturing-fjsp-optimization__2823d4a1 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 31 | present | ctrf.json: 13/15 pass, 2 fail, 0 skipped |
| mario-coin-counting | mario-coin-counting__0c5dea26 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 6 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped |
| mars-clouds-clustering | mars-clouds-clustering__387d89ca | error |  | invalid: ACP transport/stdout closed | 13 | present-partial | none: no verifier artifacts found |
| mars-clouds-clustering | mars-clouds-clustering__0d36d57e | pass | 1.0 | pass | 12 | present | ctrf.json: 8/8 pass, 0 fail, 0 skipped |
| multilingual-video-dubbing | multilingual-video-dubbing__41a28b1a | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 15 | present | ctrf.json: 0/8 pass, 8 fail, 0 skipped |
| offer-letter-generator | offer-letter-generator__1a728e32 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 6 | present | ctrf.json: 2/4 pass, 2 fail, 0 skipped |
| organize-messy-files | organize-messy-files__f21683ae | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 35 | present | ctrf.json: 4/6 pass, 2 fail, 0 skipped |
| paper-anonymizer | paper-anonymizer__55dae684 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 16 | present | ctrf.json: 5/6 pass, 1 fail, 0 skipped |
| parallel-tfidf-search | parallel-tfidf-search__67cba0d7 | pass | 1.0 | pass | 25 | present | ctrf.json: 5/5 pass, 0 fail, 0 skipped |
| paratransit-routing | paratransit-routing__7018fe1d | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 15 | present | ctrf.json: 4/7 pass, 3 fail, 0 skipped |
| pddl-airport-planning | pddl-airport-planning__3e023f0f | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 16 | present | ctrf.json: 1/2 pass, 1 fail, 0 skipped |
| pddl-tpp-planning | pddl-tpp-planning__785326e4 | pass | 1.0 | pass | 13 | present | ctrf.json: 2/2 pass, 0 fail, 0 skipped |
| pdf-excel-diff | pdf-excel-diff__c9cac2fd | fail | 0.0 | valid model failure: missing/incorrect artifact | 15 | present | ctrf.json: 6/11 pass, 5 fail, 0 skipped |
| pedestrian-traffic-counting | pedestrian-traffic-counting__87121f5c | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| pedestrian-traffic-counting | pedestrian-traffic-counting__04c45e9c | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| pedestrian-traffic-counting | pedestrian-traffic-counting__4daa7b37 | error |  | invalid: agent install failure | 0 | missing | none: no verifier artifacts found |
| pg-essay-to-audiobook | pg-essay-to-audiobook__fe4537cc | error |  | invalid: artifact copy/setup failure | 0 | missing | none: no verifier artifacts found |
| powerlifting-coef-calc | powerlifting-coef-calc__127d6e7e | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 8 | present | ctrf.json: 9/11 pass, 2 fail, 0 skipped |
| pptx-reference-formatting | pptx-reference-formatting__037ea65a | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 14 | present | ctrf.json: 8/12 pass, 4 fail, 0 skipped |
| protein-expression-analysis | protein-expression-analysis__8e93c6da | fail | 0.0 | valid model failure: reward only/no structured detail | 12 | present | reward.txt: reward only; no CTRF |
| python-scala-translation | python-scala-translation__5c9bcd0f | fail | 0.0 | valid model failure: missing/incorrect artifact | 9 | present | reward.txt: reward only; no CTRF |
| quantum-numerical-simulation | quantum-numerical-simulation__3031c9cf | verifier_error |  | invalid: verifier timeout | 11 | present | none: no verifier artifacts found |
| quantum-numerical-simulation | quantum-numerical-simulation__a43c7238 | verifier_error |  | invalid: verifier timeout | 10 | present | none: no verifier artifacts found |
| quantum-numerical-simulation | quantum-numerical-simulation__32f8a92b | error |  | invalid: ACP transport/stdout closed | 6 | present-partial | none: no verifier artifacts found |
| r2r-mpc-control | r2r-mpc-control__788d8a67 | fail | 0.0 | valid model failure: reward only/no structured detail | 43 | present | ctrf-report.json: 5/6 pass, 1 fail, 0 skipped |
| radar-vital-signs | radar-vital-signs__45b21344 | pass | 1.0 | pass | 11 | present | ctrf.json: 5/5 pass, 0 fail, 0 skipped |
| react-performance-debugging | react-performance-debugging__50a6157a | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 24 | present | reward.txt: reward only; no CTRF |
| reserves-at-risk-calc | reserves-at-risk-calc__afacb5a0 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 31 | present | ctrf.json: 1/5 pass, 4 fail, 0 skipped |
| sales-pivot-analysis | sales-pivot-analysis__3695b886 | fail | 0.0 | valid model failure: missing/incorrect artifact | 13 | present | ctrf.json: 6/10 pass, 3 fail, 1 skipped |
| sec-financial-report | sec-financial-report__c6f115db | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 26 | present | ctrf.json: 1/2 pass, 1 fail, 0 skipped |
| seismic-phase-picking | seismic-phase-picking__a1615427 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 35 | present | ctrf.json: 0/2 pass, 2 fail, 0 skipped |
| setup-fuzzing-py | setup-fuzzing-py__092b7ff9 | fail | 0.16 | valid model failure: verifier assertion/quality threshold | 15 | present | ctrf.json: 2/2 pass, 0 fail, 0 skipped |
| shock-analysis-demand | shock-analysis-demand__40298df2 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 4 | present | ctrf.json: 2/5 pass, 3 fail, 0 skipped |
| shock-analysis-supply | shock-analysis-supply__d95da6be | fail | 0.0 | valid model failure: missing/incorrect artifact | 21 | present-partial | ctrf.json: 1/9 pass, 8 fail, 0 skipped |
| shock-analysis-supply | shock-analysis-supply__78b391ae | fail | 0.0 | valid model failure: missing/incorrect artifact | 11 | present | ctrf.json: 2/9 pass, 7 fail, 0 skipped |
| simpo-code-reproduction | simpo-code-reproduction__82e67221 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 69 | present | reward.txt: reward only; no CTRF |
| software-dependency-audit | software-dependency-audit__4100e80d | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 10 | present | ctrf.json: 2/4 pass, 2 fail, 0 skipped |
| spring-boot-jakarta-migration | spring-boot-jakarta-migration__2382b67d | pass | 1.0 | pass | 101 | present | ctrf.json: 10/10 pass, 0 fail, 0 skipped |
| suricata-custom-exfil | suricata-custom-exfil__dbff8ae8 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 38 | present | reward.txt: reward only; no CTRF |
| syzkaller-ppdev-syzlang | syzkaller-ppdev-syzlang__241dfb9c | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 16 | present | ctrf.json: 1/7 pass, 6 fail, 0 skipped |
| taxonomy-tree-merge | taxonomy-tree-merge__de2f6499 | fail | 0.6962 | valid model failure: verifier assertion/quality threshold | 11 | present | ctrf.json: 15/22 pass, 7 fail, 0 skipped |
| threejs-structure-parser | threejs-structure-parser__94aa9b66 | verifier_error |  | verifier contract/artifact error | 6 | present | ctrf.json: 1/3 pass, 2 fail, 0 skipped |
| threejs-to-obj | threejs-to-obj__fd7de03b | verifier_error |  | verifier contract/artifact error | 8 | present | ctrf.json: 2/3 pass, 1 fail, 0 skipped |
| tictoc-unnecessary-abort-detection | tictoc-unnecessary-abort-detection__e53a4d34 | fail | 0.1 | valid model failure: verifier assertion/quality threshold | 17 | present | ctrf.json: 3/3 pass, 0 fail, 0 skipped |
| travel-planning | travel-planning__ba35e17e | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 11 | present | ctrf.json: 9/10 pass, 1 fail, 0 skipped |
| video-filler-word-remover | video-filler-word-remover__4b9e74b6 | error |  | invalid: ACP transport/stdout closed | 8 | present-partial | none: no verifier artifacts found |
| video-filler-word-remover | video-filler-word-remover__bc926eda | error |  | invalid: ACP transport/stdout closed | 13 | present-partial | none: no verifier artifacts found |
| video-filler-word-remover | video-filler-word-remover__4f64d2f0 | error |  | invalid: ACP transport/stdout closed | 1 | present-partial | none: no verifier artifacts found |
| video-silence-remover | video-silence-remover__008c6bef | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 30 | present | ctrf.json: 5/9 pass, 4 fail, 0 skipped |
| video-tutorial-indexer | video-tutorial-indexer__7fe67e9d | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 10 | present-partial | ctrf.json: 0/2 pass, 2 fail, 0 skipped |
| video-tutorial-indexer | video-tutorial-indexer__fb0bedbb | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 4 | present-partial | ctrf.json: 0/2 pass, 2 fail, 0 skipped |
| video-tutorial-indexer | video-tutorial-indexer__f1b368f8 | fail | 0.0 | valid model failure: verifier assertion/quality threshold | 12 | present-partial | ctrf.json: 0/2 pass, 2 fail, 0 skipped |
| weighted-gdp-calc | weighted-gdp-calc__be41ce68 | fail | 0.0 | valid model failure: missing/incorrect artifact | 7 | present | ctrf.json: 14/27 pass, 13 fail, 0 skipped |
| xlsx-recover-data | xlsx-recover-data__597fd77d | fail | 0.0 | valid model failure: missing/incorrect artifact | 22 | present | ctrf.json: 1/8 pass, 7 fail, 0 skipped |
