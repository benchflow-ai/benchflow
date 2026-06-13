# benchflow

Multi-turn agent benchmarking with ACP. Docs live in [`docs/`](./docs/).

## Setup + test

```bash
uv sync --extra dev --locked
uv run python -m pytest tests/
uv run ty check src/
uv run ruff check .
```

## Conventions

- **Don't rewrite passing tests** to match new behavior. Update for shape changes, not for semantic changes you don't understand. No tautological tests.
- **Regression tests must name the PR/commit they guard** in the docstring (e.g. `Guards the fix from PR #198 against the regression introduced by PR #193`).
- **Human review before `main`.** PRs only. No force-pushes to `main`. Self-approval doesn't count.
- **Trunk-based:** branch off `main`, PR back to `main`. No long-lived release branches.
- **Releases:** current mechanics live in [`docs/release.md`](./docs/release.md). Merges to `main` publish internal preview `.devN` builds after CI passes; public releases require a reviewed stable-version PR, a matching `v<version>` tag on `main`, then a bump back to the next `.dev0`.
- **CUA / computer-use agent loop uses Gemini 3.5 Flash** (`gemini-3.5-flash`) as the default vision model, driving a prompted screenshot→action schema (click/type/key/scroll/done). The `computer-use-agent` ACP shim is model-pluggable via `session/set_model`, but `gemini-3.5-flash` is the default and the dogfood model for desktop/computer-use slices. Control runs in-sandbox (e.g. `xdotool`) so the Cua sandbox provider stays untouched.

## Experiment guidance (when using benchflow to run batch tasks experiments)

- When running tasks experiments on Daytona, keep in mind that Daytona supports up to 100 parallel VMs.
- When running tasks experiments by default use the highest inference level for all the model calls (For example, xhigh for GPT5.5, or MAX for Opus4.8).
- When running tasks experiments, for each API key that available to you, verify 1. whether the key is still working by sending basic LLM calls; 2. ask the user how many parallel tasks can each key support / figure it out by yourself. Thus you can use highest available concurrency when running experiments.
- **A revoked/wrong model key shows up as an opaque agent error, not an auth error.** With `openhands`, a dead key surfaces only as `ACP error -32603: Internal error` on the *first* model call (and `--usage-tracking off` / `--agent-env LLM_CACHING_PROMPT=false` do **not** help). So before any run, verify the *exact* key string with a raw provider call: Gemini `curl ".../v1beta/models/<model>:generateContent?key=$GEMINI_API_KEY"` must return 200; Bedrock a `converse` call with `Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK` must return 200. Keys of the same provider can rotate format (e.g. Google AI Studio `AQ.…` vs legacy `AIza…`) — only the live one works, so never assume a `.env`/`keys.env` entry is current.
- **Daytona hard-caps each sandbox at 10 GB** (a larger storage request is clamped — see the `Clamping storage_mb … -> 10240` log line). Tasks with heavy images — large HuggingFace model snapshots, Playwright, LaTeX/marker, e.g. `latex-formula-extraction` — overflow during bootstrap and fail with `No space left on device`, or **hang silently at "Sandbox user agent ready" with no trajectory** (this is an infra/disk failure, *not* a model or auth bug). On Daytona pick light tasks (e.g. `citation-check`, `3d-scan-calc`); for heavy tasks use `--sandbox docker` (host disk, no 10 GB cap).
- **Opus-4.8 (and other Claude 4.8+) on Bedrock needs the adaptive-thinking shim.** Without it the first call 400s (`thinking.type.enabled is not supported … use thinking.type.adaptive`). It ships in the `openhands` install on current main — verify the install log prints `opus-4.8 bedrock thinking shim ACTIVE`, i.e. the checkout contains `src/benchflow/agents/oh_bedrock_opus_patch.py`. On Docker the host Bedrock proxy injects the adaptive contract instead. Model string is `aws-bedrock/us.anthropic.claude-opus-4-8` with `AWS_REGION=us-west-2`. **MAX effort is opt-in:** set `BENCHFLOW_BEDROCK_THINKING_EFFORT=max` — without it the run uses the agent's default effort (adaptive `high`, **not** max). It is honored on both backends (Daytona forwards it into the sandbox; Docker reads it host-side) and lands as `"output_config": {"effort": "max"}` in `trajectory/llm_trajectory.jsonl`; verify there rather than trusting the "MAX" label.
- Use Huggingface as the ground truth of tasks traj database. The target is making sure number_of_tasks * 2 (with/without skills) * number_of_trials (usually 3 / 5) for each model + each harness is complete. Each traj on Huggingface must be "healthy" - "healthy" means 1. the trial has complete traj information (agent pass / fail / timeout the task); 2. the trial has complete meta information (for example: token usage; timing; etc.). For each traj, we should use subagents to audit 1. whether the fail / timeout is due to agent capability instead of task config or env setting; 2. there is no reward hacking behavior; 3. in "no-skill" trials the agents do not have access to any skill files or accessed any skill files in traj.
- When reviewing task-run result trajectories, agents must use the installed `benchflow-experiment-review` skill from the active harness's skill root; in this repo that copy lives at `.claude/skills/benchflow-experiment-review`.
