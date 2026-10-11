# Agents and harnesses

An agent is what attempts a task in a trial. A harness is how BenchFlow drives it. This page is the design the agent work follows: one manifest format, five harness kinds, one trajectory format.

## Harness kinds

| Harness | Who runs the loop | How BenchFlow sees the work | Used for |
|---|---|---|---|
| `acp` | the agent, in the sandbox | Agent Client Protocol over the agent's stdin and stdout: every message, thought, tool call, and usage update as it happens; model selection, permission prompts, and cancel are part of the protocol | every agent that speaks ACP (Claude Code, Codex, Gemini CLI, OpenCode, Goose, Kimi, Qwen Code, and the ACP registry) |
| `headless` | the agent, in the sandbox | the agent's own non-interactive mode (`claude -p --output-format stream-json`, `codex exec --json`, `fx ask --json`, ...): BenchFlow parses its output stream into the same events | CLI agents without ACP, and a second route for agents that have both |
| `native` | `bench`, on the host | the model is called from the host and its tools run in the sandbox through the sandbox API; every request and response is recorded | Terminus-2 (keystrokes to tmux), the builtin tool loop (bash, read, write, edit), computer-1 (screenshots and xdotool on a virtual display) |
| `backend` | an outside service, over HTTP | `bench` speaks an HTTP backend protocol from the host: the transcript streams to the service, its messages' tool calls run in the sandbox | `external`, for an RL trainer or any service that decides each step |
| `script` | nobody | the script's output | `oracle`, `nop`, negative controls, exploit probes |

`bench run -a codex` uses the agent's default harness; `--harness headless` picks another one the agent declares. Every agent that has both routes upstream has both here, on one install: the ACP agents that also run headless (Claude Code, Codex, Gemini CLI, OpenCode, goose, Cline, Cortex Code, Cursor, Antigravity, Grok Build, Junie, OpenHands, Kimi CLI, MiMo, Pi, Qwen Code, Copilot, Mistral Vibe) default to ACP, and Hermes, MiniMax Code and Devin default to headless and also run over their own ACP servers (`hermes-acp`, `mcode acp`, `devin acp`). `-a acp:<id>@<version>` runs an entry of the public ACP registry. `bench agents` lists every agent with the harnesses it supports and whether it was verified on this build.

## The public ACP registry

`-a acp:<id>[@<version>]` runs any entry of the [public ACP registry](https://agentclientprotocol.com/get-started/registry) without a manifest file: BenchFlow turns the entry into a manifest on the fly. It installs the entry's distribution for the sandbox's platform (a `binary` archive when there is one for every Linux platform, else the `npx` package, else the `uvx` package), launches the entry's command with its arguments and variables, and selects the model the `auto` way. The registry does not say which login an agent needs, so an entry runs on no login, and `--env KEY=VALUE` passes the key it reads; when a built-in manifest names the same agent (`acp_registry = "codex-acp"`), the entry uses that manifest's logins, skill paths, and launch variables with the registry's build, so `acp:codex-acp` runs on a Codex ChatGPT login.

The index is fetched from `https://cdn.agentclientprotocol.com/registry/v1/latest/registry.json` and cached for a day in BenchFlow's cache folder. A version other than the one listed is read from the registry's git history and cached for good. With no network, the copy built into `bench` is used, and `BENCHFLOW_ACP_REGISTRY=<file>` reads a local copy instead. An id or version the registry does not have is an error that names the closest ids or the versions it does list. `bench agents --registry` lists the entries and what each would install.

`bench agents check <name> -s <backend>` builds an agent's install layer on a small image (`ubuntu:24.04` unless `--image` says otherwise), starts the agent as the unprivileged agent user, and sends ACP's `initialize`: no prompt, no login. It reports the name, version, protocol version, and login methods the agent gives, and records the result (agent, version fingerprint, backend, date, outcome) in `agent-checks.jsonl` in BenchFlow's cache folder; `bench agents` shows the latest check of exactly this build in its `checked` column. A check says the agent installs and speaks ACP, nothing about whether it can log in or solve a task.

## Manifests

Every agent is a TOML manifest: the built-in ones in `crates/benchflow-agent/agents/`, and a project's own in `bench.toml` under `[agents.<name>]` (a table named like a built-in agent replaces it). The keys:

- `name`, `aliases`, `description`. Other common names for an agent are aliases, so `-a claude-code` and `-a claude` are the same agent.
- `[install]`: `npm` (with `bin`), `script`, `uv-tool`, `uv-venv` (a Python framework in `/opt/benchflow/agents/<name>/venv`), or `installer`; and the three ways the ACP registry distributes agents: `binary` (a per-platform archive with its SHA-256 when known, unpacked under `/opt/benchflow/agents/<name>/` and launched by its real path inside the archive), `npx` (an npm package, installed like `npm`, its executable picked the way `npx` picks it), and `uvx` (a pinned PyPI package installed with a pinned uv). Installs run once, as root, as a cached image layer on top of the task's image, so the second trial of an agent on a task starts in seconds. Installers that write to `$HOME` run with `HOME=/opt/benchflow/agents/<name>`, and their binaries are linked into `/opt/benchflow/bin`, which is on the agent user's `PATH`. An install never sees a credential.
- `[launch]`: the ACP command (harness `acp`). It gets the run's model as `$BENCHFLOW_MODEL` and the effort as `$BENCHFLOW_EFFORT`, for agents that read their model only when they start (Hermes and MiniMax Code write it into their config).
- `[headless]`: the command (harness `headless`), the output format its parser reads, extra environment, and files to write in the agent's home before it starts. The instruction reaches the command as a file (`$BENCHFLOW_INSTRUCTION_FILE`) and a variable (`$BENCHFLOW_INSTRUCTION`), never pasted into the command line.
- SDK agents (`openhands-sdk`, `nemo-agent`, `strands`, `deerflow`, `antigravity-sdk`, `langgraph`, `eve`, `dspy-rlm`) are a framework rather than a CLI: their `[headless]` table writes one of BenchFlow's runner scripts (`crates/benchflow-agent/runners/`, embedded in the binary) into the agent's home with `[[headless.files]] runner = "..."`, and the runner reports in the `benchflow-runner` JSON-lines format. Agents that run the user's code (a LangGraph graph, a Strands factory, an Eve app) take it from the task, named with `bench run --env`, and fail with R215 saying what to set when it is missing.
- `[credentials]`: which logins the agent accepts, in order (`anthropic`, `openai`, `gemini`, `model`, `env`, `none`), and `account`: the vendor account an agent such as Cursor or Copilot needs, shown by `bench agents <name>` and in the fix for a missing login (R211).
- `acp_registry`: the same agent's id in the public ACP registry.
- `[model]`: how the model and the reasoning effort are selected, and the default model.
- `[options.<name>]`: settings a run passes with `--ak` (below).
- `skill_paths`: where the agent's CLI discovers skills, each starting with `$HOME/` or `$WORKSPACE/` (`["$HOME/.claude/skills"]`). The run copies the trial's skills into each (below); an agent with none takes no skills. For a CLI that needs to be told where its skills are, the first folder, expanded (`/home/agent/.kimi-code/skills`), reaches the agent's commands as `$BENCHFLOW_SKILLS_DIR`, set only when the trial placed skills there: a launch or headless command passes it as a flag (`${BENCHFLOW_SKILLS_DIR:+--skills-dir "$BENCHFLOW_SKILLS_DIR"}`), writes it into a config file, or hands it to a runner.
- `skills_via`: one line on how the CLI finds its skills beyond the folder (`named to kimi with --skills-dir`), which `bench agents <name>` shows after the folders. It needs `skill_paths`.
- `[resources]`: the least the agent needs from its sandbox, `min_memory` (`"4 GB"`, `"512 MB"`) and `min_cpus` (whole CPUs). A task that asks for less, or leaves the size to the sandbox backend's default, gets the floor instead; a task that asks for more keeps its own. The run log says when the floor raised the size, and result.json records it as `resources_raised` (`{"agent": "openclaw", "memory_mb": {"task": null, "sandbox": 4096}}`, `task` being null when the task asked for nothing). OpenClaw has `min_memory = "4 GB"`: at Daytona's default size it was killed for lack of memory (exit 137) about 25 seconds into a run.
- `[resume]`: whether a later graded stage can resume the agent's own session, and which session file format `--load-trajectory` can place for it (below).
- `[providers]`: how to point the agent at an OpenAI-compatible or an Anthropic-compatible endpoint, or at a gateway's own API (`[providers.gateway]`, which fx speaks): which variables carry the base URL and the key, which must be cleared, and templates for the rest, so any provider below works with any agent that can take a base URL. A template line that uses `{model}` or `{effort}` when it stands for nothing is left out, and a variable left with nothing is not set.

## Agent options

`bench run -a <agent> --ak <name>=<value>` (also `--agent-option` and `--agent-kwarg`; repeatable) passes a setting to one agent, by the option's name in the agent's manifest. A job file or `bench.toml` takes the same settings as an `[agent-options]` table (`max_turns = 30`), and flags win. Each option is declared in the agent's manifest, so the run is refused before any sandbox starts when a name is unknown (with the closest name and the list), when a value has the wrong type or is out of range (saying what the option takes), or when the option does not reach the agent on the run's harness (saying which harness to pick). An agent without options says so and points at `--env`. `bench agents <name>` lists each option, what it takes, how it reaches the agent, its default, and what it does; each trial's `config.json` records the ones set as `agent_options`. An option the run does not set is not passed: `default` says what the agent does without it.

```toml
[options.max_turns]
type = "int"                 # string, int, float, bool, or enum (with choices)
description = "Most agent turns."
min = 1                      # int and float: min, max
flag = "--max-turns"         # or env, file + key, install
harnesses = ["headless"]     # default: every harness the agent has
```

| Key | Meaning |
|---|---|
| `type` | `string`, `int`, `float`, `bool`, or `enum` (then `choices`). Required, with `description`. |
| `default`, `min`, `max` | What the agent does when unset (shown, not passed); the range for numbers. |
| `env = "VAR"` | A variable for the agent process, on both harnesses. |
| `flag = "--max-turns"` | Words added to the command where it says `"$@"` (the default ACP command takes them at its end). `{value}` in the flag is replaced by the value (`-c web_search={value}`); without it the value follows as its own word; a `bool` adds the flag when true. Each word is quoted for the shell, so a value reaches the agent as written. |
| `file`, `key` | A key (dotted) set in a `.json` or `.toml` file in the agent's home, on top of the manifest's `[[headless.files]]` copy when it has one (Gemini CLI's `max_turns` is `model.maxSessionTurns` in `~/.gemini/settings.json`). |
| `install = "<package>"` | The version of a package the install pins (`@anthropic-ai/claude-code`). `default` must be the pinned version, and `verify` lines that name it follow. The install changes, so the image layer and the agent's fingerprint do too; asking for the pinned version is the built-in install. |
| `harnesses` | The harnesses the option reaches the agent on. Claude Code's print-mode flags reach only `headless`: its ACP adapter starts the CLI itself. |

Declared options, by agent:

| Agent | Options |
|---|---|
| `claude` | `version` (the Claude Code CLI), `max_thinking_tokens` (both harnesses); headless only: `max_turns`, `max_budget_usd`, `fallback_model`, `append_system_prompt`, `allowed_tools`, `disallowed_tools`, `disable_web_search`, `thinking_display` |
| `codex` | `version` (codex-acp, which carries the Codex CLI), `reasoning_summary`, `web_search` (as `-c` overrides, on both harnesses) |
| `gemini` | `version`, `max_turns` (a settings key) |
| `opencode` | `version` |
| `goose` | `max_turns` (headless), `temperature`, `max_tokens`, `context_limit` (variables); no `version`: the release archive is pinned by checksum |
| `aider` | `version`, `thinking_tokens`, `cache_prompts`, `auto_lint`, `auto_test`, `test_cmd`, `map_tokens` |
| `mini-swe-agent` | `version`, `cost_limit`, `max_tokens` |
| `openhands` | `version`, `max_iterations`, `temperature`, `num_retries` |
| `terminus-2` (native) | `max_turns`, `parser_name` (json or xml), `temperature`, `max_output_tokens`, `enable_summarize`, `proactive_summarization_threshold`, `tmux_pane_width`, `tmux_pane_height` |
| `builtin`, `computer-1` (native) | `temperature`, `max_output_tokens`, and `bash_timeout_sec` or `max_turns` |

The reasoning effort stays `-e`/`--effort`, which fills an agent's `reasoning_effort` option. On Codex, whose model ids carry the effort (`gpt-6.1-sol[high]`), `-e` without `-m` runs the session's default model in the variant it offers for that effort, and records that id; an effort the session does not offer for its model stops the trial, naming the ones it does. Options that need a host file (`config_file`, `memory_dir`, `prompt_template_path`) or that hand BenchFlow's isolation to the agent (`permission_mode`, Gemini CLI's `sandbox`, Codex's `dangerously_bypass_hook_trust`) are not declared.

## Resuming sessions across stages

A task with graded stages (task.md `stages/`, or `steps/` in the split layout) runs each stage in the same sandbox, and the handoff before each stage is graded kills every agent process. By default each stage starts a fresh session with its prompt. `bench run --resume-sessions` (`resume-sessions = true` in a job file) makes each stage after the first resume the agent's own session instead, for agents whose manifest has a `[resume]` table:

```toml
[resume]
acp = true                            # session/load with the previous stage's session id
headless = true                       # the command resumes when $BENCHFLOW_RESUME is set
session_dirs = [".claude/projects"]   # where the sessions live, under the agent's home
```

The agent's home outlives the handoff (it removes only files planted to steer the verifier), so the session's files are still there. Before a stage resumes, BenchFlow checks that one of `session_dirs` is not empty. Over ACP the harness then calls `session/load` with the previous stage's session id when the agent's `initialize` answer offers `loadSession`; the agent replays the conversation before it answers, and the replay is dropped (the stage before recorded it). Headless, the command runs with `BENCHFLOW_RESUME=1`, which it turns into its CLI's own resume flag: Claude Code `--continue`, Codex `exec resume --last`, Gemini CLI `--resume latest`, OpenCode `--continue`, goose `--resume`.

| Agent | ACP | Headless |
|---|---|---|
| `claude`, `codex`, `gemini`, `opencode`, `goose` | `session/load`, when the agent offers it | the CLI's resume flag |
| every other agent, the native loops, scripts | fresh each stage | fresh each stage |

An agent that cannot resume, an ACP agent that does not offer `session/load`, and a stage whose session folders are empty all start a fresh session, and the stage's row in result.json's `step_results` says so: `session` has the session `id` (ACP), `resumed`, and `why` when a stage that could have resumed did not. `bench run` also notes at the start when the agent cannot resume on its harness. config.json records `resume_sessions`. aider keeps its chat history in the working folder rather than its home, so it runs fresh. Live on Daytona (Claude Code on `vercel/deepseek/deepseek-v4-flash-0731`, one trial per arm, a two-stage task whose second stage needs a code word given only in the first): with `--resume-sessions` the second stage passed over ACP (claude-agent-acp 0.81.2 offers `session/load`) and headless, and without it the second stage failed on both. Not yet run live: whether codex-acp 2.0.1, Gemini CLI 0.42, OpenCode 1.18 and goose 1.54 offer `session/load` (the record says when one does not), and Codex's `exec resume --last` reading the prompt from stdin.

### Loading a session from a file

`bench run --load-trajectory <file>` (`load-trajectory` in a job file) starts the agent on a session from an earlier run: the file (at most 100 MB) goes where the agent's CLI keeps its sessions, owned by the agent user, and the first prompt resumes it the way a later stage resumes the one before (ACP `session/load` with the file's session id; headless with `BENCHFLOW_RESUME=1`). Later stages start fresh, unless the run also asks for `--resume-sessions`. The manifest names the file format in `[resume]`:

```toml
[resume]
acp = true
headless = true
session_dirs = [".claude/projects"]
load = "claude-jsonl"                 # or "codex-rollout"
```

| Agent | File | Placed at |
|---|---|---|
| `claude` | Claude Code's session, `<session-id>.jsonl` | `~/.claude/projects/<working folder, every character but letters, digits and - as ->/` (`-app` for `/app`) |
| `codex` | Codex's rollout, `rollout-<YYYY-MM-DDThh-mm-ss>-<session-id>.jsonl` | `~/.codex/sessions/<YYYY>/<MM>/<DD>/` |
| every other agent | refused before anything starts, with the next step | |

Gemini CLI, OpenCode, and goose resume across stages but keep a session as more than one file (or in a database), so they cannot load one yet; ATIF trajectories (`.json`) are refused too, since converting one to an agent's own format is not built yet. `bench run` checks the file before anything starts: its name must be the format's, and every line a JSON object. An ACP agent that does not offer `session/load` fails the trial with R215 rather than start fresh without the context. Each trial's config.json records `load_trajectory` (`path`, `format`, `session_id`, `sha256`, `placed_at`, `resumed`), and the trial folder keeps the file in `loaded-session/`. Live on Daytona (Claude Code over ACP, on `vercel/deepseek/deepseek-v4-flash-0731`): a code word that was only in the loaded session reached the agent, which wrote it out. Not yet run live: headless Claude Code, Codex (whose `exec resume --last` may look only at sessions from the same working folder), and a control run without the file.

## Skills

A skill is a folder with a `SKILL.md` and whatever it bundles. A trial's skills come from the task and from the run:

- the task's: a folder in the package (task.md `[sandbox] skills = "sandbox/skills"`, or a split task's `environment/skills/`, as SkillsBench ships them), copied whole; or a folder the image fills (a split task's `skills_dir`, an absolute path such as `/skills`, which task.md writes as `[sandbox] skills = "/skills"`), copied from inside the sandbox. `bench run --no-task-skills` (`task-skills = false`) leaves them out, BenchFlow 0.8's `no-skill` mode.
- the run's: `bench run --skill <folder>` (repeatable; `skills = [...]` in a job file), a skill or a folder whose folders are each a skill. A run skill replaces a task skill of the same name. A folder that is neither is refused before anything starts.

Before a model agent starts, BenchFlow uploads them once to a root-only folder in the sandbox and copies them into each of the manifest's `skill_paths`, creating the folders on the way and handing them and the copy to the agent user. The oracle and scripted attempts get none. Each trial's config.json records `skills`: `task` (the task's folder), `task_skills`, `run` (the `--skill` folders), `names` (the skills in place), `placed_in`, and `why` when the agent got none or fewer than declared.

| Agent | Skills go to |
|---|---|
| `claude` | `~/.claude/skills` |
| `codex` | `~/.agents/skills` |
| `gemini` | `~/.gemini/skills` |
| `opencode` | `~/.config/opencode/skills` |
| `goose`, `mimo` | `~/.config/goose/skills`, `~/.config/mimocode/skills` |
| `copilot`, `cursor`, `kimi`, `grok`, `cline`, `junie`, `openclaw` | `~/.copilot/skills`, `~/.cursor/skills`, and so on, one folder named for the agent |
| `qwen-code` | `~/.qwen/skills` |
| `fx`, `fx-dev` | `~/.fx/skills` |
| `hermes`, `mcode` | `~/.hermes/skills`, `~/.mcode/skills` (those CLIs' data folders as BenchFlow sets them) |
| `pi` | `~/.pi/agent/skills` and `~/.agents/skills` |
| `openhands` | `~/.agents/skills` and `<workdir>/.agents/skills` |
| `antigravity` | `~/.gemini/config/skills` and `<workdir>/.agents/skills` |
| `openhands-sdk`, `antigravity-sdk` | `~/.openhands-sdk/skills`, `~/.gemini/skills` |
| `muse-code` | `~/.config/muse/skills` (Muse's config home) |
| `kimi-code` | `~/.kimi-code/skills`, named to `kimi` with `--skills-dir`; no skills, no flag |
| `mistral-vibe` | `~/.vibe/skills` (Vibe's global skills folder), also listed as `skill_paths = [...]` at the top of the config.toml BenchFlow writes (on a provider, over ACP or headless; and headless on Mistral's API); no skills, no key |
| `eve` | `~/.benchflow/eve-skills`, then BenchFlow's runner copies each folder there that holds a `SKILL.md` into the app's skills folder (`agent/skills/`, or `skills/` for a flat app) as `benchflow_<name>` before it builds the app |
| every other agent, the native loops, the external agent | none |

A skills folder, the task's or one given with `--skill`, may hold at most 1 GiB of files; a larger one stops the trial (R107) before anything is uploaded. An agent with no skills folder runs without them: `bench run` notes it at the start, and config.json's `skills.why` says so; with `--skill`, such an agent is refused before anything starts. A Dockerfile that copies the skills folder into the image itself (two of SkillsBench's 87 tasks) still gives the agent the skills with `--no-task-skills`; config.json's `skills.why` warns when it does. Live on Daytona (Claude Code over ACP, on `vercel/deepseek/deepseek-v4-flash-0731`): both of a task's skills landed in `/home/agent/.claude/skills`, owned by the agent user, and the agent listed them among its skills. Also live, on a task whose code word only a skill holds: Mistral Vibe over ACP (config.toml began with `skill_paths = ["/home/agent/.vibe/skills"]`) and Kimi Code headless (with `--skills-dir`) each called the skill through their own skill tool, wrote the code word, and listed the task's three skills. Kimi Code also finds `~/.kimi-code/skills` by itself, and `--skills-dir` replaces that search (so project folders such as `<workdir>/.agents/skills` are not searched when skills were delivered). Not yet run live: the other CLIs finding their copies; eve (it needs your own Eve app) and muse-code (a Meta key) have unit tests only.

## Saved sessions and `bench handoff`

At the end of each agent turn (each stage's, on a staged task), BenchFlow saves the agent's `session_dirs` into the trial folder at `agent/sessions/`, laid out as in the agent's home (`agent/sessions/.claude/projects/-app/<id>.jsonl`). The files are packed in the sandbox under the same caps as saved outputs (100 MB a file, 10,000 files, 1 GiB in all; links and special files are never copied). Login files are never saved: Claude Code's `.credentials.json`, Codex's and OpenCode's `auth.json`, Gemini CLI's `oauth_creds.json`, goose's `secrets.yaml`, and every file the credential scan looks for. Of the five agents' session folders, only OpenCode's is known to hold a login (`~/.local/share/opencode/auth.json`); it is left out by name. result.json records what was saved as `agent_session`: `path`, `dirs`, `files`, `bytes`, the `session_id` the harness reported, and `left_out` (files not saved: login files, links, and files over the cap). A trial whose session folders were empty has no `agent_session`. Saving is never a failure of the trial; a session that could not be saved is logged.

`bench handoff <trial>` resumes that session in the agent's own CLI on this machine, in the current folder:

| Agent | Where the session goes | Command |
|---|---|---|
| `claude` (ACP or headless) | `$CLAUDE_CONFIG_DIR` (else `~/.claude`)`/projects/<this folder, with every character but letters, digits and - as ->/<id>.jsonl`, with its `<id>/` folder (subagents) | `claude --resume <id>` |
| `codex` (ACP or headless) | `$CODEX_HOME` (else `~/.codex`)`/sessions/YYYY/MM/DD/rollout-...-<id>.jsonl`, the rollout's own path | `codex resume <id>` |

The session is copied under its own id. A local file at that path with the same content is left alone; one with other content is never overwritten unless you pass `--replace`, and the handoff says what it copied or replaced. When the trial saved more than one session (a staged task that started fresh sessions), the one the harness recorded is resumed, else the newest; `--session <id>` picks another. `--dry-run` prints what would be copied and the command and copies nothing; arguments after `--` go to the CLI (`bench handoff <trial> -- --version`). From there the CLI uses your own login and model, not the trial's.

It is a top-level command next to `bench shell` rather than `bench trial handoff`: BenchFlow's commands are verbs on what you pass (`bench view <job>`, `bench shell <task>`), and there is no `trial` group. `bench shell` takes you into a task's sandbox; `bench handoff` brings a finished trial's agent to you.

An agent other than Claude Code and Codex is refused with a next step: Gemini CLI, OpenCode, and goose trials save their sessions too, and the refusal points at the folder, but where their CLIs look for a session depends on a hash of the project folder (Gemini CLI, OpenCode) or a database (goose), so copying them is not built yet. A trial that saved no session (an older bench, or an agent that stopped before writing one) is refused with the command that runs the task again. The `claude` or `codex` CLI must be on `PATH`; the refusal says how to install it.

## Providers

Providers are data too (`crates/benchflow-agent/providers/`). A model is named `<provider>/<model>`, as in `vercel/deepseek/deepseek-v4-flash-0731` or `anthropic/claude-haiku-4-5`. The provider says where its endpoints are (OpenAI-compatible, Anthropic-compatible, or both), where its key comes from, and how to check it. The Vercel AI Gateway (`vercel/`) serves OpenAI- and Anthropic-compatible endpoints for hundreds of models, so a Claude Code or Codex run can use an open-weight model with one key. The built-in providers are `vercel`, `openai`, `anthropic`, `openrouter`, `zai`, `deepseek`, and `mimo` (Xiaomi MiMo's token plan, `-m mimo/mimo-v2.6-flash` with `MIMO_API_KEY`; its keys work only on the plan's Singapore host). The native loops (`builtin`, `terminus-2`, `computer-1`) call every one of them from the host: the provider's OpenAI-compatible endpoint when it takes a bearer key, else its Anthropic-compatible one.

A provider key is never given to the agent. BenchFlow starts a small forwarder as root inside the sandbox, on loopback, that holds the key in a root-only file and adds it to each request; the agent gets the forwarder's URL and a placeholder key. This is the same on both manifest harnesses (ACP and headless). On a `<provider>/<model>` run the forwarder passes only that model, so an agent's helper cannot spend the key on another.

### Keys and the forwarder

Every agent that can take an OpenAI-compatible endpoint has a `[providers.openai]` table: Codex, OpenCode, aider, mini-swe-agent, SWE-agent, OpenHands and its SDK, Qwen Code, goose, Hermes, MiniMax Code, Pi, Mistral Vibe, MiMo, Kimi CLI and Kimi Code, Cline, Grok Build, deepagents, trae-agent, OpenClaw, Code Puppy, NeMo Agent, DeerFlow, and Junie; Claude Code has `[providers.anthropic]`, and fx `[providers.gateway]`. Two agents take a provider on their headless route only: Kimi CLI's ACP server starts no session without a Kimi Code sign-in, and Cline's ACP server takes a model only from its own catalog.

A plain API key the agent finds on this machine goes through the forwarder too, when one of the agent's `[providers]` tables can take an endpoint of a provider that reads that key: the agent gets a placeholder and keeps picking its own model (and its helpers' models) as it would on the key itself. Subscription logins are files or variables the agent's own CLI needs, so they go to the agent, and their values are redacted from every log and trajectory. `bench agents <name>` says, for each credential the agent reads, which way it goes.

| Credential | Agents | What happens |
|---|---|---|
| The provider's key on a `<provider>/<model>` run (`AI_GATEWAY_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, ...) | any agent with a `[providers]` table for one of the provider's endpoints | held by the forwarder; an agent without such a table is refused (R215) |
| `ANTHROPIC_API_KEY` | Claude Code | held by the forwarder (Anthropic's API) |
| `OPENAI_API_KEY`, `CODEX_API_KEY`, an API key stored in `~/.codex/auth.json` or `CODEX_AUTH_JSON` | Codex | held by the forwarder (OpenAI's Responses API) |
| `AI_GATEWAY_API_KEY`, `VERCEL_AI_GATEWAY_API_KEY`, `VERCEL_OIDC_TOKEN` | fx, fx-dev | held by the forwarder (the gateway's own API), and pinned |
| `CLAUDE_CODE_OAUTH_TOKEN`, `CLAUDE_OAUTH_TOKEN`, `~/.claude/.credentials.json` | Claude Code | given to the agent: a subscription login |
| `CODEX_ACCESS_TOKEN`, a ChatGPT login in `~/.codex/auth.json` or `CODEX_AUTH_JSON` | Codex | given to the agent: a subscription login |
| `ANTHROPIC_AUTH_TOKEN` | Claude Code | given to the agent: a bearer token for an endpoint of the user's choosing |
| `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `~/.gemini/oauth_creds.json` | Gemini CLI | given to the agent: no built-in provider serves Google's API |
| keys of agents without a `[providers]` table (Gemini CLI, Cursor, Copilot, Cortex Code, Antigravity, Devin, ...) | those agents | given to the agent: it cannot take a base URL from BenchFlow |
| a router's key such an agent would take to the router itself (Junie's `JUNIE_OPENROUTER_API_KEY`), or a `--env` base URL on a router's host | any | refused (R215, the agent's): BenchFlow could not keep its requests off denied upstreams |

A key stays with the agent when the run's `--env` points the agent at an endpoint of its own (the table's base-URL variable, or a `*_BASE_URL` it clears): the key is then meant for that endpoint.

### Routers and denied providers

A router that picks among upstream providers never gets to pick a denied one: those on BenchFlow's built-in list (which no file or setting can lift), a provider file's `[routing] deny`, and any named in `BENCHFLOW_DENY_UPSTREAMS`. The forwarder adds the rule to each request it forwards, overriding the agent's own, and the native clients add it to theirs:

- the Vercel AI Gateway takes an allowlist. BenchFlow reads the upstreams it lists for the model (once per process, no key needed), drops the denied ones, and names the rest in every request (`providerOptions.gateway.only`, on chat completions, the Responses API, Anthropic messages and its own AI SDK protocol). A model no allowed upstream serves, or a run that names no model, stops before any sandbox starts, and the forwarder passes only the run's model, whose list it was;
- OpenRouter takes a denylist. Every request names the denied upstreams in `provider.ignore` (OpenRouter's documented provider-routing field), added to any the agent sent; no listing is needed.

On a route where the forwarder checks the model or pins upstreams, a request body it cannot read as a JSON object (a compressed one, or one that is not JSON) is refused with HTTP 415 and a message the agent's log shows (and a line in `agent/forwarder.log`), so nothing it forwards there went unchecked. Where it checks the model, every model the body names counts: `model`, a router's fallback lists (`models`, `providerOptions.gateway.models`), and each request of a message batch; a model that is not a string, or a batch whose models sit in an uploaded file (`/batches`), is refused with HTTP 403. The body it forwards is the one it checked, written out again as JSON.

fx runs on the gateway's own API: BenchFlow points it at the forwarder with `FX_GATEWAY_BASE_URL` (its model list and usage lookups) and `FX_GATEWAY_CHAT_URL` (its model calls), which fx honors only as loopback `http` addresses. The forwarder serves that API from its root so fx's own paths pass through, and its model guard reads the model from that protocol's headers (fx's helper calls for another model are refused, and fx carries on). `agent/forwarder.log` starts with the pin (`{"upstreams": {"model", "only" or "ignore", "excluded"}}`) and has one line per request, marked `"pinned"` when the rule was added, with a short hash of the body it sent (`body_hash`): a request sent again after a refusal has the same one, which tells a client waiting out a rate limit from an agent going on without that answer when BenchFlow decides whether a timeout was the provider's (R214).

### Token counts and cost

An agent that reports a cost prices its own tokens at the prices it knows: Claude Code on DeepSeek V4 Flash through the gateway reported $0.1135 at Claude's prices for a run the gateway billed under $0.004. So on a run through the forwarder, the agent's figure is never kept. The forwarder reads each model call's answer (a JSON body or a stream of events, in OpenAI's Chat Completions and Responses formats, Anthropic's Messages, or the AI SDK's own protocol) and adds to that request's line in `agent/forwarder.log` the token counts it reported (`usage`: `input` not read from a cache, `cache_read`, `cache_write`, `output`) and the cost the provider charged (`cost`, in US dollars), when the answer said: the Vercel AI Gateway does in every answer (`usage.cost` or `provider_metadata.gateway.cost`), and so does OpenRouter (`usage.cost`); a provider's own API (Anthropic's, OpenAI's, DeepSeek's) does not. The trial's `agent_result` then holds:

- the provider's token counts (`usage_source` is `forwarder`), or the agent's when no answer had any;
- `cost_usd`, the sum of what the provider charged, with `cost_source = "provider"`, only when every model call it answered said what it cost;
- otherwise no `cost_usd`, and a `cost_note` saying why (and what the agent claimed, which is not kept).

BenchFlow keeps no price table. The gateway's model list has prices, but they are a floor across the upstreams that serve a model (`"varies_by_provider": true`); a test call on 2026-10-10 was billed 5.8 times the listed price, so the list is not used. A run on an agent's own login, without the forwarder, keeps the agent's figure with `cost_source = "agent"` when the agent reports one: on the headless harness Claude Code does (`claude -p`'s `total_cost_usd`), as do OpenCode, Goose, and the other headless agents whose output carries a cost. Over ACP no agent reports a cost (checked on 2026-10-10 with Claude Code on a subscription login: neither the ACP messages nor its session file carry one), and Codex reports none on either harness, so those trials show `cost not recorded`. On a subscription login, a figure Claude Code reports is a price at its list rates, not what the subscription is charged; the native loops read the provider's figure from its answers (`cost_source = "provider"`). Across the stages of a multi-stage task, costs add up only when every stage's is known.

## One trajectory

Every harness records the same events (`benchflow-agent`'s `Trajectory`): the user's prompts, the agent's messages and thoughts, tool calls with their updates and results, and usage. The trial folder gets `trajectory/acp_trajectory.jsonl` in BenchFlow 0.8's layout (0.8's viewer and `bf.load_job` read it), and `trainer/atif.json` holds the same trajectory as ATIF; `bench export <job>` turns a job's trajectories into a Hugging Face dataset or OpenTelemetry traces. Native harnesses also record every model request and response.

## Budgets

Every harness enforces the wall-clock budget and the idle timeout. Where the harness can count them (ACP, native, and headless formats that report them), `--max-tool-calls` and `--max-tokens` stop the agent too, and the trajectory records why it stopped.

## Failures

Agent failures have codes and owners like every other failure (`bench explain R2xx`): R201 install, R202 did not start, R203 idle, R204 crashed, R205 agent error, R206 did nothing before its time ran out (not graded), R211 no login, R212 login rejected, R213 usage limit, R214 provider failed, R215 a model or setting the agent lacks. Headless and native harnesses map their errors to the same codes.

## The external agent, DSPy-RLM, and computer-1

Three more agents, and what each does:

**`external`**: `bench run -a external --backend <url>`. An outside service decides every step; BenchFlow runs the steps. BenchFlow speaks the backend protocol from the host, with the trial's name as the run id: it asks `GET /api/internal/transcript_length` (and `POST create_transcript` on a 404), streams every event to `POST append_transcript` as `{"event", "seq"}` in the protocol's event format (`task_started`, `task_pre_hook_completed`, `message_added`, `step_started`, `tool_call_started`, `tool_call_completed`, `step_completed`, `scoring`, `task_completed`, `error`), posts `update_run_state` at the start and the end, and long-polls `GET get_message` (404 while there is none) and `POST delete_message` for the model's next message. Each message's tool calls run in the sandbox as the agent user with the builtin loop's tools (`bash`, `view`, `replace`, `write`, `submit`), listed for the service in the `task_pre_hook_completed` metadata under `tools`. Grading happens after the last step, so `scoring` (the reward, `continue_task: false`) and `task_completed` (`passed`, `failed`, or `error`) follow the last `step_completed`. A service silent for the idle timeout stops the trial with R203; one that cannot be reached or sends an unreadable message with R205; both name the service as the agent, the owner. A bearer token is sent when the file named by `BENCHFLOW_BACKEND_TOKEN_PATH` exists; with neither set, the file `/var/run/secrets/service-account-token` is read and sent when it exists, so on a machine that has one, point `--backend` only at a service that may see it. The trial folder keeps the transcript as sent (`agent/backend_transcript.json`). `-m` is only a label for the service.

**`dspy-rlm`** (a DSPy RLM agent): `bench run -a dspy-rlm -m <provider>/<model>`. BenchFlow runs DSPy inside the sandbox as an SDK runner (dspy 3.4.0 and deno 2.9.7 pinned in a venv, Pyodide cached at install), so its tools act on the task's files directly and the host needs no Python or Deno. The signature, limits, and seven tools are fixed by the runner; the file tree is listed from the task's working folder. The model is reached through the provider forwarder, which BenchFlow starts before the runner as it does for ACP agents, so the key stays out of the agent and a gateway's requests carry the allowed-upstream list. Each REPL step is a `python` tool call in the trajectory, and the tools that step called are calls of their own.

**`computer-1`** (computer-1's generic JSON flavor): `bench run -a computer-1 -m <provider>/<vision model>`. A native loop: it starts Xvfb, a window manager, and a browser in the sandbox (the task's image must have them; an image without Xvfb, xdotool, and ImageMagick's import or scrot is refused as the task's problem, R215), sends the model its prompt and a screenshot each turn, and runs its one JSON action through `xdotool` as the agent user. A `done` must come twice in a row; the second writes the answer to `/logs/agent/final_answer.txt` for the verifier. The run also ends after 50 turns. Only the newest three screenshots stay in the conversation; every screenshot is kept as `agent/screenshots/NNN.png`, and the call log records image sizes, not bytes. `crates/benchflow-run/tests/fixtures/tasks/hello-desktop` is a small task with such an image.

## Where the code is

- `crates/benchflow-agent/`: manifests and the registry, credentials and providers, the ACP client, headless output parsers, the native loops' model clients, and the trajectory.
- `crates/benchflow-run/src/acp_agent.rs`, `headless_agent.rs`, `native_agent.rs`, `external_agent.rs`, `dspy_agent.rs`: each joins a harness to a trial (login into the sandbox, start, record, map failures); `forwarding.rs` is the provider hook the ACP, headless, and DSPy-RLM harnesses share.
- `crates/benchflow-run/src/skills.rs` and `agent_files.rs`: skills and loaded sessions put in place before the agent starts; `crates/benchflow-agent/src/manifest/resume.rs`: the session file formats.
