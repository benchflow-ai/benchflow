# BenchFlow supervisor update view

Updated: 2026-04-18 early AM
Owner view: Hermes supervisor summary for Xiangyi before handoff

## 1. Where we are overall

BenchFlow 0.3 is no longer in pure planning mode.
We are in **execution + validation phase**.

What is already materially shipped on the main 0.3 line:
- **A6 ACP conformance**: done and closed on PR #150
  - green: `claude-agent-acp`, `gemini`
  - experimental: `codex-acp`, `pi-acp`, `openclaw`
- **A2 dense rewards**: shipped
  - `rewards.jsonl`
  - terminal reward proof
  - rubric/process rewards
- **A2 snapshot/restore**: shipped behind unified API
  - Daytona proof passed
  - Docker implementation exists but not proved in this devcontainer
- **A1 multi-agent runtime (minimal)**: shipped as first slice
  - scene/role/message core types
  - scheduler/outbox bridge
  - 2-agent proof passed

What is **not** started as full implementation yet:
- full runtime unification around `Runtime.execute()` / `bf.run(scene, env)`
- A3 sidecars productization
- A5 CLI migration completion
- A7 selective Harbor internalization audit closure

So the current stage is:
- **core 0.3 foundation slices landed**
- **benchmark and parity diagnosis in progress**
- **runtime unification still next major build step**

## 2. Feature stage by area

### A6 ACP conformance
Stage: **complete for 0.3 gate**

Status:
- registry was forced into a clean bucket model
- conformance issue closed
- remaining non-green agents are explicitly experimental, not silently treated as release-ready

### A2 dense rewards + snapshots
Stage: **shipped first production slice**

Status:
- end-to-end reward persistence works
- process rewards work
- snapshot/restore API exists
- Daytona proof exists

Meaning:
- BenchFlow now has a real differentiator for RL-facing runtime work
- this is no longer just roadmap text

### A1 multi-agent runtime
Stage: **minimal viable slice landed, not final architecture**

Status:
- basic scene runtime exists
- 2-agent proof exists
- enough is built to validate multi-agent product questions

Meaning:
- we have the seed of the runtime-centered architecture
- but not yet the final unified execution center

### Runtime refactor (`Runtime.execute()` / `bf.run`)
Stage: **agreed design, implementation not started**

This is the next real architecture phase after the current benchmark clarification work.

## 3. What we learned from benchmarks

## TB2 diagnosis changed significantly today

Earlier conclusion was wrong because the harness had a verifier/plugin confound.
The important fix was:
- root-mode verifier preflight fix
- pytest plugin inference for legacy verifier scripts (`--ctrf` -> `-p ctrf`)

Without that, many tasks were false negatives.

### BenchFlow TB2 slices

#### Old Haiku baseline (legacy/confounded)
Job: `tb2-ci-baseline/2026-04-17__18-35-21`
- total: 89
- pass: 0
- fail: 80
- err: 9

This should now be treated as **legacy/confounded**, not as the real baseline.

#### Opus sandbox slice v1 (confounded + restricted)
Job: `tb2-opus-probe/2026-04-17__19-07-52`
- total: 10
- pass: 0
- fail: 8
- err: 2

#### Opus root slice v3 (first clean 10-task slice)
Job: `tb2-opus-probe-root/2026-04-17__20-10-14`
- total: 10
- pass: 4
- fail: 5
- err: 1
- score: **40%**

This is the first trustworthy Claude-era 10-task TB2 slice.

### Current Gemini TB2 baseline (current default line)
Job: `tb2-gemini-baseline/2026-04-17__20-37-20`
Current on-disk results at update time:
- completed: 86 / 89
- pass: 12
- fail: 69
- err: 5

Current passes seen on disk:
- `build-pmars`
- `cobol-modernization`
- `code-from-image`
- `crack-7z-hash`
- `git-leak-recovery`
- `headless-terminal`
- `multi-source-data-merger`
- `openssl-selfsigned-cert`
- `pypi-server`
- `pytorch-model-recovery`
- `sparql-university`
- `vulnerable-secret`

Most important signal so far:
- `pytorch-model-recovery` passed under Gemini after timing out in both Opus slices

So current TB2 lesson is:
- verifier correctness mattered a lot
- root-vs-non-root mattered somewhat
- model choice now matters a lot again after fixing the harness

## 4. Harbor status

### What was actually run
I only triggered a **single-task Harbor probe**, not full parity:
- Harbor 0.4.0
- `claude-code`
- `claude-opus-4-6`
- Daytona
- task: `adaptive-rejection-sampler`

Result:
- reward = 0.0
- verifier worked
- not the same fake `--ctrf` failure mode we saw in broken BenchFlow runs
- but the run had a Claude auth interruption confound later on

So current Harbor status is:
- **full parity not done yet**
- single-task Harbor probe is useful only as a directional datapoint
- Xiangyi clarified the real reusable Harbor integration line should be:
  - **gemini-cli + gemini-3.1-flash-lite-preview**
  - SkillsBench 10
  - Terminal Bench 2.0 89

That Harbor Gemini line should become one of the main long-term integration tests.

## 5. Integration-test view

The main reusable integration test shape is now:
- model: `gemini-3.1-flash-lite-preview`
- BenchFlow agent: `gemini`
- Harbor agent: `gemini-cli`
- backend: Daytona
- SkillsBench: 10 tasks
- Terminal Bench 2.0: 89 tasks

This is important because it gives us:
- one default model family
- one default backend
- one stable cross-harness comparison shape

## 6. What is blocked vs unblocked

### Unblocked
- A6 gate
- A2 rewards
- A2 Daytona snapshot proof
- A1 minimal multi-agent slice
- TB2 clean reruns under fixed verifier path

### Still blocked / incomplete
- full runtime unification work has not begun
- Docker snapshot proof unavailable in this environment
- Harbor full parity run on the aligned Gemini setup still needs to be triggered
- SkillsBench 10-task current Gemini integration line still needs to be run and logged into the comparison notebook
- TB2 Gemini full 89-task result was still finishing at the time of this update

## 7. Best current product reading

BenchFlow 0.3 is now credible as:
- a runtime-centered benchmark harness
- with shipped reward plumbing
- with first multi-agent execution primitives
- and with active benchmark diagnosis that is now much more trustworthy than earlier in the day

But 0.3 is **not done** because the most important conceptual center is still not implemented:
- `Runtime.execute()`
- `bf.run(scene, env)`

That is the next major engineering phase once the current integration-test baseline is stabilized.

## 8. Immediate next steps after sleep

Recommended order:
1. finish TB2 Gemini 89-task run and record final summary
2. run Harbor parity on the new reusable line:
   - `gemini-cli`
   - `gemini-3.1-flash-lite-preview`
   - Daytona
   - SkillsBench 10
   - TB2 89
3. add those results into the notebook and mark parity vs non-parity clearly
4. after integration line is stable, start runtime unification implementation

## 9. Artifact list

Previous planning artifact found:
- `docs/0.3-plan.md`
- `docs/0.3-plan.pdf`

New supervisor update artifacts:
- `context/2026-04-18-supervisor-view.md`
- `context/2026-04-18-supervisor-view.html`
- `context/2026-04-18-supervisor-view.pdf`
- `context/tb2-parity-results.ipynb`

## 10. Bottom line in one sentence

**We are past planning, through the first shipped 0.3 slices, in benchmark/integration-test clarification now, and the next true milestone is stabilizing the Gemini-based cross-harness integration line before moving into full runtime unification.**
