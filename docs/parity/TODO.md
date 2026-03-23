# Parity Testing — Status & TODOs

## Completed

### Code Fixes (this session)
- [x] Added Daytona environment support (`DaytonaProcess` via SSH, `DaytonaEnvironment` integration)
- [x] Abstracted `LiveProcess` base class — both `DockerProcess` and `DaytonaProcess` implement same interface
- [x] SDK `environment` parameter — supports `"docker"` or `"daytona"`
- [x] Fixed `npm install | tail -3` swallowing exit codes → `set -o pipefail` + post-install verification
- [x] Fixed Node version check → upgrades Node if version < 22
- [x] Fixed hardcoded `cwd="/app"` → reads actual WORKDIR from sandbox via `pwd`
- [x] Fixed token limit crashes → `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`
- [x] Fixed buffer overflow → increased readline buffer from 64KB → 10MB
- [x] Added `config/update` ACP method to set model via protocol (env var `ANTHROPIC_MODEL` is ignored by claude-agent-acp)
- [x] Reduced Docker concurrency to 4 + network pruning to avoid Docker network exhaustion
- [x] Better error diagnostics: stderr capture on agent crash, install failure diagnostics
- [x] Verifier dir creation for non-mounted environments (Daytona)
- [x] Dynamic agent binary path resolution via `which` for SSH sessions

### TB2 Single-Turn Run (Step 1)
- [x] Ran all 89 tasks — Docker first, then Daytona retry for errored tasks
- [x] **Result: 52/89 (58.4%)** with Sonnet 4.6 (claude-agent-acp defaults to Sonnet, not Haiku)
- [x] Parity validated: official Anthropic reports 59.1%, tbench.ai shows 59.55% for Sonnet 4.6
- [x] Our 58.4% is within ~1% — **pipeline produces valid results**

### Key Discovery
- claude-agent-acp v0.22.2 bundles Claude Code v2.1.76 (Claude Agent SDK v0.2.76)
- It ignores `ANTHROPIC_MODEL` env var — uses its own default (Sonnet 4.6)
- Model must be set via ACP `session/config/update` method (now implemented)
- tbench.ai's Claude Code + Haiku 4.5 entry (27.53%) uses Claude Code v2.0.31 from Nov 2025

### Error Analysis
52 passed, 23 failed, 14 errored:
- 9 errors = timeouts (agent hangs on first API call, 0 tool calls — model limitation)
- 5 errors = Daytona npm install timeout (>15 min, network bottleneck)

---

## TODO

### Immediate — Fix Remaining Errors
- [ ] **Retry 5 install-timeout tasks on Docker** (path-tracing-reverse, gpt2-codegolf, write-compressor, caffe-cifar-10, prove-plus-comm) — npm installs fast on Docker, Daytona is slow
- [ ] **Daytona snapshots** — pre-bake agent into Daytona snapshot images to eliminate install time entirely (future)

### Step 2: TB2 Multi-Turn Run
- [ ] Run all 89 tasks with recheck prompt: `[instruction, "Review your solution. Check for errors, test it, and fix any issues."]`
- [ ] Compare single-turn vs multi-turn scores
- [ ] Use Daytona, concurrency 4
- [ ] Note: must set model via ACP `config/update` if we want Haiku (currently defaults to Sonnet 4.6)

### Step 3: SkillsBench Sanity Check
- [ ] Run 20 random SkillsBench tasks (87 available, exclude external-dep tasks)
- [ ] Compare with reference trajectories at https://github.com/benchflow-ai/skillsbench-trajectories/ (xiangyi-completed branch)

### Step 4: SkillsBench Full Run
- [ ] Run all self-contained SkillsBench tasks
- [ ] Get overall score

### Step 5: Parity Analysis
- [ ] Write `parity/PARITY.md` with:
  - TB2 single-turn score vs tbench.ai leaderboard
  - TB2 multi-turn score (if run)
  - SkillsBench score
  - Error analysis and root causes
  - Agent/model version details

### Multi-Agent Testing
- [ ] Test pi-acp on hello-world + a few TB2 tasks
- [ ] Test openclaw on hello-world + a few TB2 tasks
- [ ] Optional: codex-acp (needs OPENAI_API_KEY), gemini (needs Google API key)

### Integration Test
- [ ] `tests/test_integration.py` — real Docker + API test on hello-world

### Documentation
- [ ] API docs for SDK, CLI
- [ ] Cookbooks (writing a task, adding an agent)

---

## Key Files

| File | Purpose |
|------|---------|
| `src/benchflow/sdk.py` | SDK.run() — orchestrates env + ACP + verifier |
| `src/benchflow/process.py` | LiveProcess abstraction (Docker + Daytona) |
| `src/benchflow/acp/client.py` | ACP JSON-RPC client (initialize, session, prompt, config/update) |
| `src/benchflow/acp/container_transport.py` | ACP over LiveProcess pipe |
| `parity/run_tb2_single.py` | TB2 single-turn runner (resume-capable) |
| `parity/run_tb2_multiturn.py` | TB2 multi-turn runner |
| `parity/run_skillsbench.py` | SkillsBench runner |
| `parity/terminal-bench-2.0/single-turn/` | Results from TB2 single-turn run |

## Versions

| Component | Version |
|-----------|---------|
| benchflow | 2.0.0 |
| claude-agent-acp | 0.22.2 |
| Claude Agent SDK | 0.2.76 |
| Embedded Claude Code | 2.1.76 |
| Model used | Sonnet 4.6 (default) |
| ACP SDK | 0.16.1 |
