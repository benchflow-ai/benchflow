# Benchflow v2 — Plan

## Completed

### Code (pushed to `main`)
- Harbor superset: import harbor as dependency, re-export everything
- ACP client: initialize, session/new, session/prompt, session/config/update, permission auto-approve
- Container transport: live stdio pipe via Docker compose exec or Daytona SSH
- SDK.run(): Harbor env (Docker or Daytona) + ACP agent + Harbor verifier
- Multi-turn: multiple prompts to same ACP session
- Multi-agent registry: claude-agent-acp, pi-acp, openclaw, codex-acp, gemini
- Model config: set via ACP `session/config/update` (env var ignored by claude-agent-acp)
- Result persistence: result.json, prompts.json, acp_trajectory.jsonl per trial
- Viewer: benchflow view renders HTML
- CLI: benchflow run, benchflow view
- Trajectory capture: ACP native
- Daytona environment support (DaytonaProcess via SSH, LiveProcess abstraction)
- Bug fixes: pipefail, node version check, dynamic WORKDIR, 10MB buffer, token limit, stderr capture

### TB2 Single-Turn (Step 1) — Done
- **52/89 (58.4%)** with Sonnet 4.6 via claude-agent-acp (Claude Code v2.1.76)
- Parity: official Anthropic reports 59.1%, tbench.ai shows 59.55% — **within ~1%**
- 14 errors: 9 timeouts (model limitation), 5 Daytona npm install bottleneck
- Discovery: claude-agent-acp ignores ANTHROPIC_MODEL env var, defaults to Sonnet 4.6
- Fixed: model now set via ACP config/update protocol

---

## Next Steps

### Retry Docker Errors (quick win)
- [ ] Retry 5 tasks that fail on Daytona npm install using Docker: path-tracing-reverse, gpt2-codegolf, write-compressor, caffe-cifar-10, prove-plus-comm
- Expected: some may pass, reducing error count

### Step 2: TB2 Multi-Turn Run
- [ ] Run all 89 tasks with recheck prompt
- [ ] Prompts: [instruction, "Review your solution. Check for errors, test it, and fix any issues."]
- [ ] Compare single-turn vs multi-turn delta
- [ ] Environment: Daytona, concurrency 4
- [ ] Model: Sonnet 4.6 (default) — or set to Haiku via config/update if we want Haiku numbers

### Step 3-4: SkillsBench
- [ ] Sanity check: 20 random tasks
- [ ] Full run: all self-contained tasks (~85 tasks)
- [ ] Compare with reference trajectories

### Step 5: Parity Report
- [ ] Write `parity/PARITY.md`:
  - TB2 scores vs tbench.ai leaderboard
  - Agent/model version matrix
  - Error analysis
  - Multi-turn improvement (if run)
  - SkillsBench results

### Multi-Agent Testing
- [ ] pi-acp on hello-world + TB2 subset
- [ ] openclaw on hello-world + TB2 subset

### Future Infrastructure
- [ ] Daytona snapshots — pre-bake agent to eliminate install time
- [ ] Haiku run — verify model selection works via config/update
- [ ] Higher concurrency on Daytona (no Docker network exhaustion)

---

## Key Facts

| Fact | Value |
|------|-------|
| claude-agent-acp version | 0.22.2 |
| Embedded Claude Code | v2.1.76 |
| Claude Agent SDK | v0.2.76 |
| Default model | Sonnet 4.6 (NOT Haiku) |
| ACP protocol SDK | 0.16.1 |
| TB2 task count | 89 |
| SkillsBench task count | 87 |
| Our TB2 score (Sonnet 4.6) | 52/89 = 58.4% |
| Official TB2 score (Sonnet 4.6) | 59.1% |
| Parity gap | ~0.7% |
