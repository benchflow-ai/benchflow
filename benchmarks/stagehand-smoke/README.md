# Stagehand Smoke

Import selected official Stagehand eval tasks into temporary BenchFlow task
directories without committing upstream task text:

```bash
git clone --depth 1 https://github.com/browserbase/stagehand.git /tmp/stagehand-main
uv run python benchmarks/stagehand-smoke/import_upstream.py \
  --stagehand-repo /tmp/stagehand-main \
  --out-dir /tmp/benchflow-stagehand-import \
  --tasks agent/sign_in \
  --overwrite
uv run bench tasks check /tmp/benchflow-stagehand-import/agent-sign_in \
  --level runtime-capability \
  --sandbox docker
source /Users/lixiangyi/context/benchflow-0.7/keys.env
uv run python benchmarks/stagehand-smoke/parity_test.py \
  --stagehand-repo /tmp/stagehand-main \
  --task agent/sign_in \
  --model google/gemini-3.5-flash \
  --parity-out /tmp/benchflow-stagehand-parity/stagehand-smoke/parity_experiment.json \
  --keep-work-dir
uv run bench agent verify stagehand-smoke \
  --benchmarks-dir /tmp/benchflow-stagehand-parity \
  --require-adoption-report \
  --loop-report-out /tmp/benchflow-stagehand-parity/stagehand-smoke/loop-report.json \
  --json

uv run python benchmarks/stagehand-smoke/import_upstream.py \
  --stagehand-repo /tmp/stagehand-main \
  --out-dir /tmp/benchflow-stagehand-inventory/tasks \
  --tasks all \
  --overwrite \
  --support-report-out /tmp/benchflow-stagehand-inventory/stagehand-support.json
```

The parity driver prefers Stagehand's built `dist/esm/tasks` tree when present,
then falls back to source tasks with `tsx`. It runs the original Stagehand
runner first, imports the selected task, runs BenchFlow with `stagehand-agent`,
checks Docker cleanup, and writes both `parity_experiment.json` and
`adoption_report.json`. It also writes `loop_state.json`, a resumable
adapter-adoption flight recorder with the Stagehand checkout revision, selected
task, replay commands, artifacts, role status, cleanup, unsupported summary,
and next queue items for verifier/expected-answer coverage.

Current dogfood result for `agent/sign_in`: the Stagehand eval runner passed
the same selected task with score `1.0` and final URL
`https://v0-modern-login-flow.vercel.app/authorized`. BenchFlow imported that
task as `stagehand-task.json`, passed with reward `1.0`, matched the final URL,
recorded Stagehand actions plus a screenshot, preserved timing and trajectory
summaries, and Docker cleanup returned zero BenchFlow containers/networks.

2026-06-12 strict-gate dogfood:
`/tmp/benchflow-stagehand-parity.aJhbsC/stagehand-smoke` returned
`parity-confirmed`, `8/8` criteria agreed, reward delta `0.0`, and
`adoption_loop.status=scale-ready`. The original runner recorded score `1.0`,
`7` log steps, and final URL
`https://v0-modern-login-flow.vercel.app/authorized`; BenchFlow recorded reward
`1.0`, `3` trajectory steps, `1` tool call, `11` Stagehand artifact steps,
`1` screenshot, the same final URL, and zero BenchFlow-owned Docker
containers/networks after cleanup.

If a reused Stagehand checkout reports missing `tsx`, `zod`, or `playwright`
internals, create a fresh checkout before rerunning parity; stale pnpm symlinks
can fail before the BenchFlow adapter path is exercised.

2026-06-12 coverage inventory against a fresh Stagehand checkout:

- official bench tasks requested: `146`
- currently supported/importable: `3` deterministic URL-check agent tasks
  (`agent/sign_in`, `agent/steam_games`, `agent/trivago`)
- unsupported reports: `143`
- unsupported issue counts: `missing-static-instruction=104`,
  `stagehand-verifier-not-mapped=38`,
  `stagehand-expected-answer-verifier-not-mapped=1`

`agent/steam_games` also passed the strict parity/adoption gate. Its original
and BenchFlow final URLs differed by path, but both satisfied the original
`url_contains` success contract for `https://store.steampowered.com/`; reward
delta was `0.0`, `8/8` criteria agreed, and cleanup stayed at zero
BenchFlow-owned Docker containers/networks.

When reproducing the original Stagehand CLI path with Gemini, export
`GOOGLE_GENERATIVE_AI_API_KEY` or `GOOGLE_API_KEY`; BenchFlow's Stagehand shim
can use `GEMINI_API_KEY` directly.
