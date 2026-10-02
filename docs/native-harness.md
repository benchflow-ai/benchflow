# Native harness: Claude Code and Codex through their own CLIs

BenchFlow runs Claude Code and Codex two ways, and one run option picks between them:

- `harness="acp"` (the default) runs the agent through its ACP adapter: `claude-agent-acp`, which drives the Claude Code CLI through the Claude Agent SDK, and `codex-acp`, which drives Codex's app server.
- `harness="native"` runs the agent's own CLI in its headless JSON mode, as the agent's vendor documents it for scripts and CI: `claude -p --output-format stream-json` and `codex exec --json`.

Both harnesses are the same agent entry (`claude-agent-acp`, `codex-acp`): the same registry entry, install, credentials, skills, model routing through BenchFlow's model proxy, web and network policies, sandbox user, timeouts, budgets, trajectory, results, viewers and training exports. Only the program between BenchFlow and the model's tool loop changes.

```sh
bench eval run --tasks-dir ./tasks --agent claude-agent-acp --model claude-haiku-4-5-20251001 --harness native
bench eval run --tasks-dir ./tasks --agent codex-acp --model gpt-5.4 --harness native
bench eval branch --tasks-dir ./tasks --harness native --prompt "Write a draft first." --prompt @instruction \
  --checkpoint-after-prompt 1 --child "label=a,prompt=Finish the draft." --child "label=b,prompt=Start over." --resume-session
```

In Python, `EvaluationConfig(harness="native")`, `RolloutConfig(harness="native")` and `bf.branch(..., harness="native")`; in a run-config YAML, `harness: native`. An agent without a native harness is refused before any sandbox starts, naming the agents that have one. `config.json` records `harness_mode` (`acp` or `native`) and, for native runs, `native_harness` (CLI and pinned package); `bf.compare` treats `harness_mode` as a setting, so compare an ACP job with a native one with `vary=("harness_mode",)` (`bench eval compare --vary harness_mode`). It is not called `harness` there because `bf.compare` and published metadata already use that name for the agent.

## When to use which

Use the ACP harness (the default) unless you have a reason not to: it supports everything below.

Use the native harness to:

- run the agent the way its vendor documents for scripts and CI;
- take the ACP adapter out of the measurement, for example to check whether an adapter release changed the agent's behaviour, or while an adapter lags a CLI release (the native harness depends on the CLI's JSON output only);
- run a pinned CLI release that no adapter release supports yet (bump the native pin, re-record the samples, run the parity suite).

## What each harness supports

| | ACP | Native |
|---|---|---|
| Multi-turn runs, nudges (`--prompt` repeated) | yes | yes: each turn resumes the CLI's session (`claude -p --resume`, `codex exec resume`) |
| Simulated users (`user`, `--loop-strategy`, task.md user) | yes: each round opens a fresh ACP session | yes: each round starts a fresh CLI session |
| Branching, checkpoints, retry from a checkpoint | yes | yes, including `resume_session`: each child's first turn resumes the parent's CLI session, whose log the checkpoint holds |
| Agent-initiated ask-user (`on_ask_user`), `confirmation_policy: human` | yes | no: the CLIs run with their approvals off and have no permission channel, so a run that registers an ask-user handler is refused rather than run with every tool approved |
| Token ids and logprobs | from BenchFlow's gateway (`BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1` on a `vllm/` or `sglang/` route); ACP does not carry them | the same gateway capture |
| Task MCP servers | over ACP `session/new` | `--mcp-config` (Claude Code), `-c mcp_servers.*` (Codex) |
| Skills | yes | yes (same skill paths) |
| Subscription login | Claude Code (`CLAUDE_CODE_OAUTH_TOKEN`, `claude login`), Codex (ChatGPT) | Claude Code only; native Codex runs only through BenchFlow's model proxy (an API key), and a ChatGPT login is refused |
| No-web, denylist and allowlist network modes | yes | yes |
| Idle watchdog activity | per streamed token, and a running tool's heartbeat | Claude Code: the same (`--include-partial-messages`, and its 30 s `tool_progress` beats); Codex: per item, since `codex exec --json` streams no text deltas, so a single model call longer than `--agent-idle-timeout` (600 s by default) is cut: raise the idle timeout for long reasoning |
| Usage and cost | from the proxy for API-key runs; the adapter's report for subscription runs (no price) | the same: the proxy for API-key runs, the CLI's `result` usage for Claude Code subscription runs (no price) |

## How a native turn runs

Each prompt runs the CLI once, as the sandbox user, through the same live-process transport the ACP path uses (Docker `exec`, the Daytona PTY). The prompt goes to the CLI's stdin as one base64 line (not the command line or an environment variable, so its size is not limited). The CLI runs as a job in its own process group. Its JSON lines are parsed as they arrive into the same ACP session updates an adapter sends, and BenchFlow's ACP session records them, so the rollout kernel's prompt loop, the idle watchdog, the wall-clock budget and the bounded cancel are the ACP path's own code.

- Claude Code runs as `claude -p --output-format stream-json --verbose --include-partial-messages --forward-subagent-text --permission-mode bypassPermissions`, with `--session-id <uuid>` on the first turn and `--resume <uuid>` after, plus `--model` and `--effort` when the run sets them (with BenchFlow's proxy the model rides `ANTHROPIC_MODEL`, as over ACP). The parser is a port of `claude-agent-acp` 0.81.2's own mapping (tool titles and kinds, tool results, the Edit and Write diffs), so both harnesses record the same trajectory.
- Codex runs as `codex exec --json --ignore-user-config --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -c ... -` (and `codex exec resume <thread> ...` after the first turn). Every setting comes from the run's `CODEX_CONFIG` as `-c` overrides (model provider, model, effort, web search), so no `config.toml` in the sandbox takes part. BenchFlow also sets `features.plugins=false`, `analytics.enabled=false`, `feedback.enabled=false` and `check_for_update_on_startup=false`: without them Codex 0.156.1 opens github.com, api.github.com, chatgpt.com and ab.chatgpt.com at startup (measured through a logging proxy); and `model_reasoning_summary="auto"`, which codex-acp sets on every turn. Codex's approvals and its own sandbox are off, as `codex-acp` runs them (`INITIAL_AGENT_MODE=agent-full-access`); BenchFlow's non-root sandbox user and uid firewall are the isolation.
- Cancel (a wall-clock or idle timeout, or `Session.cancel()`) sends SIGINT to the CLI's process group so the CLI can write its result and session log; once the CLI has exited (at most 2 s later) the group and every process of the turn are killed, found by a marker (`BENCHFLOW_NATIVE_RUN`) every one of them inherits, including tool processes that left the group. The prompt returns once they are gone, or 3.5 s after the cancel (inside the kernel's 5 s bound for a timed-out prompt), in which case the kill goes on and the disconnect and the next turn wait for it. The trajectory stops at the cancel: a tool call that was running stays pending, as over ACP.
- A CLI that exits before it reports the end of its turn is an agent error (`Native harness error (claude-code): the CLI exited (exit code 137) ...`), classified `acp_error` like an adapter's error for the same failure; a transport or sandbox that went away is reported as the transport failure.
- Evidence per trial: `agent/<cli>.jsonl` (every JSON event, credentials redacted), `agent/native-turns.json` (per turn: arguments with MCP settings left out, session id, exit code, stop reason, usage, the CLI's own list-price cost estimate, and after a cancel whether any process was left), and the CLI's stderr in the agent log the ACP path writes (`agent/claude_agent_acp.txt`, `agent/codex_acp.txt`). A rollout that connects more than once (user-loop rounds, scenes) keeps adding to the same files, numbering turns on.

## Model routing

A native CLI reaches a model only through BenchFlow's route, and `connect_native` refuses anything else before a process starts:

- an API-key run goes through BenchFlow's model proxy, as over ACP: the CLI holds the proxy's endpoint and per-run key, never a provider key (`ANTHROPIC_API_KEY` in the CLI's environment is refused);
- a Claude Code subscription run uses the CLI's own login (`CLAUDE_CODE_OAUTH_TOKEN` or uploaded `claude login` files), as over ACP; on a no-web task the model-only TLS proxy admits it after checking the CLI package, its `--version` and that the executable resolves into the pinned package;
- native Codex runs only through the proxy; `CODEX_ACCESS_TOKEN`, `CODEX_AUTH_JSON` and `CODEX_API_KEY` are removed from its environment.

The deterministic tier checks that every model call a native CLI makes reaches the proxy (the model responses in the CLI's own stream equal the calls the proxy captured), and runs native Codex on a task whose agent network is an allowlist (`example.com` only): the run passes, and the egress proxy records no attempt to reach api.openai.com, chatgpt.com or any other host.

## A login with no usage left

A Claude Code subscription run can end because the login's usage is spent rather than because the task was hard, and the native harness tells the two apart. The CLI reports a spent login twice over: in its own words (for example `You've hit your weekly limit · resets Oct 3, 6:08pm (UTC)`) and as a machine-readable record, `rate_limit_event` (`{"status": "rejected", "rateLimitType": "seven_day", "resetsAt": <unix time>, "isUsingOverage": false}`). The failed turn then raises `bf.UsageLimitError`, as on the ACP harness: the window and the reset come from the record, or from the words when there is no record. The trial is filed as `usage_limit` (unscored, never retried), and an `Evaluation` starts no more trials; see [Usage limits](./when-a-run-fails.md#usage-limits). Any other failed turn raises `NativeHarnessError`, which keeps the CLI's words (`.agent_text`), the same message with the HTTP status appended (`.message`, what the rollout files), and any rejected record (`.rate_limit`). `tests/fixtures/native_harness/claude-code-2.1.280/usage-limit.jsonl` is that exchange, recorded from the pinned CLI against a server that answers the API's HTTP 429 with the `anthropic-ratelimit-unified-*` headers of a rejected claim.

This is a Claude Code path only. Native Codex runs solely through BenchFlow's proxy on an API key (`CODEX_ACCESS_TOKEN`, `CODEX_AUTH_JSON` and `CODEX_API_KEY` are removed from its environment), so it has no subscription to run out of; a 429 there is an ordinary provider rate limit.

How the CLI reports it depends on the login, which matters for anything matching on the words: on a subscription login (`CLAUDE_CODE_OAUTH_TOKEN`) it fails at once with the limit text and the record above, while with an API key the same 429 is an ordinary rate limit — retried ten times over about three minutes inside the CLI, then reported as `API Error: Request rejected (429) ...`, with no limit, window or reset in it and no `rate_limit_event`.

## Pins

| CLI | Pin | Where |
|---|---|---|
| Claude Code | `@anthropic-ai/claude-code@2.1.280` | `_CLAUDE_CODE_PACKAGE`: the same binary `claude-agent-acp` runs (`CLAUDE_CODE_EXECUTABLE`), so the two harnesses cannot drift apart on the CLI |
| Codex | `@openai/codex@0.159.3` | `_CODEX_CLI_PACKAGE`: the release `codex-acp` 2.0.1 resolves its `^0.159.1` dependency to, so both harnesses run one Codex core. Installed next to `codex-acp` for native runs only; `codex-acp` still runs its own nested copy |

Each native connect checks the CLI's `--version` against the pin and refuses another build. The Codex Apps policy checks and probes the native CLI when the harness is native.

When a pin moves:

1. Change the constant in `agents/registry.py`.
2. Re-record the samples on a machine with the pinned CLIs: `npm install -g --prefix DIR <pins>`, then `python tests/fixtures/native_harness/record_samples.py --npm-prefix DIR`. The samples are each CLI's output for a turn, a resumed turn, a failing tool call and a cancelled turn against the deterministic fake model; `tests/test_native_harness_parsers.py` replays them, and a pin without samples fails that test.
3. Run the deterministic tier (below) on both harnesses. For Claude Code, also re-check the parser port against the adapter's `tools.js` and `acp-agent.js` when the adapter pin moves.

## Parity

`tests/test_deterministic_integration.py` runs the deterministic tier's scenarios (a passing task, a wrong answer, a timeout during a tool call, a crash of the CLI, a broken verifier; an in-place branch) on both harnesses with the scripted fake model, so parity costs no model calls. A native branch whose children resume the parent's session (`--resume-session`) runs too:

- **Outcome and trajectory parity.** Claude Code native must reproduce the ACP runs' golden files field for field: outcome labels, rewards, counts, error categories, usage and cost, the provider calls, the ACP trajectory and the ATIF export. The one allowed difference is the ATIF agent name (the CLI instead of the adapter). Codex runs four scenarios (pass, wrong answer, a crash of the Codex CLI, a model call slower than the agent timeout) on both harnesses, which must agree on everything but tool call ids (`codex-acp` exposes the model's call id, `codex exec` only its own item id), with one written exception: when the Codex process dies mid-turn, `codex-acp` 1.13.1 does not report it and the prompt runs to the agent timeout (then the verifier scores the workspace), while the native harness reports the crash at once as an agent error, as both harnesses do for Claude Code. Up to the crash the two runs match (the model call, its usage, the trajectory).
- **Wire parity.** With `BENCHFLOW_LITELLM_WIRE_LOG_DIR` set, the proxy writes every request as the agent sent it (body without the proxy's own keys, identifying headers only, never credentials). The tier compares the two harnesses' requests for the same task after normalizing ids and removing, with one exact rewrite each, the written differences below; the rest must be identical.
- **The harness contract**, on every native trial: the pinned CLI ran with its headless flags, its JSON events are kinds the pin's recorded samples contain, and the proxy captured as many model calls as the CLI made.

Expected request differences (`WIRE_EXPECTED_DIFFERENCES` in the tier):

| CLI | Where | Why |
|---|---|---|
| Claude Code | `User-Agent`, and the billing line that opens the system prompt (`cc_entrypoint`) | Claude Code names its entrypoint: `sdk-ts` and the Agent SDK version (the SDK inside the adapter), or `sdk-cli` (print mode) |
| Claude Code | `tools` | the ACP session starts in the default permission mode and offers `EnterPlanMode` and `ExitPlanMode`, which bypassPermissions does not; the other 24 tool definitions are identical and in the same order |
| Claude Code | the first tool result | the adapter hands the SDK the session directory as an additional working directory, which adds a system reminder: "Environment update: Additional working directories added: /app" |
| Codex | `originator`, `User-Agent` | `codex-acp` names the ACP client (`benchflow` and its version), `codex exec` names itself (`codex_exec`) |

Everything else is identical, including the system prompt, the messages, the model, thinking and limits for Claude Code, and the instructions, tools, input items, reasoning settings and `store` for Codex. Session, thread, turn and installation ids, and Claude Code's `metadata.user_id`, are normalized away.

## Adding a CLI

A native harness is two functions and one entry (`benchflow/native_harness/spec.py`):

- a command builder: a `NativeTurn` (resume id, model, effort, MCP servers) and the launch environment in, the CLI's arguments out;
- a parser: the CLI's JSON events in, ACP `session/update` payloads out, and at the end a `NativeTurnOutcome` (stop reason, usage, error, session id);
- a `NativeHarness` in `native_harness/harnesses.py`: agent entry, executable, npm pin, `--version` output, extra install command, the two functions.

Gemini CLI (`gemini -p --output-format stream-json`, in 0.62.0) and OpenCode (`opencode run --format json`, raw JSON events, in 1.18.33) fit this shape. Each also needs recorded samples, a route through the proxy, and a place in the parity suite.

## Not covered yet

- The ACP adapters' AIR tool-call contract (`codex-acp` 2.0, `claude-agent-acp` 0.82 and later). `main` moved to `codex-acp` 2.0.1 (which runs `@openai/codex` 0.159.x) after this branch was cut; when that reaches this line, the native Codex pin should move to the same Codex release so both harnesses keep one core, and the Codex parity rows re-run. The native harness itself does not depend on the adapters' contract.
- Native harnesses for Gemini CLI and OpenCode (see above), and a bot that proposes pin bumps with re-recorded samples and a parity run.
- Daytona: the native harness uses the same live-process transport there (a PTY), but only Docker has run it so far.
