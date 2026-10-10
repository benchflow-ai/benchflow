//! Agent manifests: an agent as data.
//!
//! A manifest is a TOML file that says how to install an agent into a
//! sandbox image, how to start it, which logins it accepts, and how BenchFlow
//! tells it which model to use. The built-in agents (`claude`, `codex`,
//! `gemini`, `opencode`, every other ACP agent in `agents/`, the
//! headless-first `hermes`, `mcode` and `devin` (which also speak ACP), the
//! headless-only `aider`, `fx`, `fx-dev`, `mini-swe-agent`, `swe-agent`,
//! `trae-agent`, `openclaw`, `kimi-code`, `code-puppy`, `rovodev-cli` and
//! `muse-code`, and the SDK agents run by a runner
//! script: `openhands-sdk`, `nemo-agent`, `strands`, `deerflow`,
//! `antigravity-sdk`, `langgraph` and `eve`) are manifests embedded in the
//! binary; a project adds or replaces agents with its own files
//! ([`Registry::with_manifests`]).
//!
//! ```toml
//! name = "claude"                          # what `--agent` takes
//! aliases = ["claude-agent-acp", "claude-code"]
//! description = "Claude Code via ACP"
//! skill_paths = ["$HOME/.claude/skills"]
//! home_dirs = [".claude"]
//!
//! [install]
//! npm = "@agentclientprotocol/claude-agent-acp@0.81.2"
//! bin = "claude-agent-acp"
//! companions = ["@anthropic-ai/claude-code@2.1.280"]
//! launcher_env = { CLAUDE_CODE_EXECUTABLE = "/opt/benchflow/js-agents/bin/claude" }
//! verify = "/opt/benchflow/js-agents/bin/claude --version"
//!
//! [launch]
//! env = { CLAUDE_CODE_MAX_OUTPUT_TOKENS = "128000" }
//!
//! [credentials]
//! family = "anthropic"
//!
//! [model]
//! select = "config-option"
//! config_id = "model"
//! effort_config_id = "effort"
//!
//! [acp]
//! typed_failures = true
//! ```
//!
//! # Top-level keys
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `name` | string, required | The agent's name: lowercase letters, digits and `-`. `oracle`, `nop` and `probe:*` are reserved for the scripted agents. |
//! | `aliases` | strings | Other names that resolve to this agent, such as 0.8's spellings (`claude-agent-acp`). |
//! | `description` | string | One line for `bench agents`. |
//! | `skill_paths` | strings | Where the agent discovers skills, each starting with `$HOME/` or `$WORKSPACE/`. A task's skills and `bench run --skill` folders are copied into each, owned by the agent user. Empty: the agent takes no skills, and a run that has some says so. The agent's command finds the first folder, once skills are in it, in `$BENCHFLOW_SKILLS_DIR` (unset when the trial placed none), for a CLI that takes the folder as a flag or a config key. |
//! | `skills_via` | string | How the CLI finds the skills, when more than the folder is needed (`named with --skills-dir`), shown by `bench agents <name>`. Needs `skill_paths`. |
//! | `home_dirs` | strings | Folders under the home directory where the agent keeps its state (`.claude`), which the run hands to the sandbox user (0.8 copied root's into the user's home). |
//! | `harness` | `"acp"` or `"headless"` | The harness `bench run` uses unless `--harness` says otherwise. It must be one the manifest declares: `acp` when it has a `[launch]` table or no `[headless]` table, `headless` when it has a `[headless]` table. Default: `acp` when declared, else `headless`. |
//! | `acp_registry` | string | The id of the same agent in the public ACP registry (`codex-acp`). `-a acp:<id>` then uses this manifest's logins, account note, skill paths, home folders, and launch variables with the registry's build. |
//!
//! # `[install]`
//!
//! How the agent gets into the image. The install runs once, as root, as a
//! cached image layer (a Docker `RUN` step), so it must be idempotent, must
//! not prompt, and gets no runtime secrets. Exactly one of `npm`, `script`,
//! `uv-tool`, `uv-venv`, `installer`, `binary`, `npx`, and `uvx` is
//! required; the last three, the ways the public ACP registry distributes
//! agents, are described in [`distribution`].
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `npm` | string | The agent's npm package spec. Pin it exactly (`name@1.2.3`); an unpinned spec installs `@latest` and is skipped when the agent is already installed. BenchFlow installs it with a private Node.js (22.20.0, in `/opt/benchflow/node`) into `/opt/benchflow/js-agents`, so the task image needs no Node of its own. |
//! | `bin` | string | The executable the package installs, required with `npm`. BenchFlow writes a launcher for it at `/opt/benchflow/bin/<bin>`. |
//! | `companions` | strings | More package specs installed into the same prefix, after `npm` (Claude Code's CLI next to its ACP adapter). |
//! | `launcher` | `"node"` or `"auto"` | How the launcher starts `bin`. `"node"` (default) runs it on BenchFlow's Node. `"auto"` checks the installed file: a script with a `#!` line runs on BenchFlow's Node, anything else (a native executable) runs directly. |
//! | `launcher_env` | table of strings | Variables the launcher exports before it starts the agent (`CLAUDE_CODE_EXECUTABLE`). Not secrets: they are baked into the image. |
//! | `verify` | string | A shell command run last; the install fails unless it succeeds (Claude's pinned-version check). Also allowed with `binary`, `npx`, and `uvx`. |
//! | `script` | string | The escape hatch for agents that are not npm packages: POSIX `sh` source that installs the agent. It runs as given, so make it stop on errors (`set -e`) and skip work already done. Requires `[launch] command`. |
//! | `uv-tool` | string | A Python CLI from PyPI, pinned exactly (`aider-chat==0.86.2`). BenchFlow installs a pinned uv into `/opt/benchflow/uv` and runs `uv tool install` with its own Python, linking the tool's executables into `/opt/benchflow/bin`. Requires `bin`. |
//! | `uv-venv` | strings | A Python framework from PyPI (an SDK agent, run by a runner script): each requirement pinned exactly (`strands-agents[openai]==1.59.0`), or a direct URL pinned to a commit (`pkg @ https://github.com/o/r/archive/<40-hex sha>.tar.gz`). BenchFlow installs a pinned uv and creates `/opt/benchflow/agents/<name>/venv` on its own Python with them; nothing is linked into `/opt/benchflow/bin`. |
//! | `python` | string | The Python version for `uv-tool` and `uv-venv` (`"3.12"`). Default: uv's choice. |
//! | `installer` | string | POSIX `sh` source for a vendor installer that writes to `$HOME` (`curl -fsSL https://fx.sh/setup.sh \| bash`). It runs as root with `HOME=/opt/benchflow/agents/<name>`; requires `links`. |
//! | `links` | strings | With `installer`: the executables it leaves, relative to that home (`.local/bin/fx`). Each is linked into `/opt/benchflow/bin`, which is on the agent user's `PATH`. |
//! | `timeout` | integer | Seconds the install may take. Default 900. |
//!
//! `verify` applies to `npm`, `uv-tool`, `uv-venv` and `installer` installs;
//! `bin` to `npm` and `uv-tool`.
//!
//! # `[launch]`
//!
//! How the agent process starts. BenchFlow speaks ACP over its stdin and
//! stdout.
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `command` | string | The shell command that starts the agent. Default (for `npm` and `uv-tool` installs): `/opt/benchflow/bin/<bin>`, followed by `args`. |
//! | `args` | strings | Arguments appended to the default command (each quoted for the shell). Not allowed with `command`. |
//! | `env` | table of strings | Variables set for the agent process. Not secrets: logins add their own (see [`crate::credentials`]). |
//!
//! The command also gets the run's model as `$BENCHFLOW_MODEL` (written as
//! `[model]` says, unless the run is on a provider, whose variables carry it)
//! and the reasoning effort as `$BENCHFLOW_EFFORT`, each unset when the run
//! names none, for agents that take the model only when they start
//! (`select = "none"`).
//!
//! # `[credentials]`
//!
//! Which logins the agent accepts; [`crate::credentials::find_login`] finds
//! one on this machine.
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `family` | string | `"anthropic"` (Claude Code's logins), `"openai"` (Codex's), `"gemini"` (Gemini CLI's), `"model"` (the provider in the model's `provider/model` id decides, as for OpenCode), `"env"` (the variables in `env`), or `"none"`. Default: `"env"` when `env` is set, else `"none"`. |
//! | `env` | strings | With `family = "env"`: the variables the agent reads its key from, in the order it prefers them. The first one set is the login. |
//! | `account` | string | The vendor account the agent's login comes from, when it needs one, and how to get its key, as one sentence (`a Cursor account: create an API key at https://cursor.com/dashboard`). `bench agents <name>` shows it, and it is the fix for a missing login (R211). |
//!
//! # `[model]`
//!
//! How BenchFlow selects the model and reasoning effort over ACP.
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `select` | string | `"auto"` (default): the session's `model` config option when it offers one, else `session/set_model`. `"config-option"`: always `session/set_config_option` with `config_id`. `"set-model"`: always `session/set_model`. `"none"`: BenchFlow leaves the model to the agent's own configuration. |
//! | `config_id` | string | The config option that selects the model with `select = "config-option"`. Default `"model"`. |
//! | `effort_config_id` | string | The config option that sets reasoning effort. Default: `"effort"` when the session offers it. |
//! | `format` | string | How a model id is written for the agent. `"bare"` (default): as given. `"provider/model"`: models.dev ids; a bare id gets its provider inferred (`gpt-*` becomes `openai/gpt-*`). `"model[effort]"`: Codex's ids, chosen from the variants the session advertises, with the requested effort when one is offered. |
//! | `default` | string | The model used when the run names none. |
//! | `strip_prefixes` | strings | Prefixes removed from a model id before it is sent (`google/` for Gemini CLI). |
//!
//! # `[headless]`
//!
//! How the agent runs without ACP, in its own non-interactive mode (harness
//! `headless`). BenchFlow writes the instruction to a file the agent user can
//! read and starts the command as the agent user in the task's working
//! folder; the command runs the agent once and exits. Its stdout is read by
//! the parser `output` names, into the same trajectory events as ACP.
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `command` | string, required | POSIX `sh` source. It gets the instruction as `$BENCHFLOW_INSTRUCTION_FILE` (a path) and `$BENCHFLOW_INSTRUCTION` (the text, when it is under 100 KB), the model as `$BENCHFLOW_MODEL` and the reasoning effort as `$BENCHFLOW_EFFORT` (each empty when the run names none). Never paste the instruction into the command: read the file (`< "$BENCHFLOW_INSTRUCTION_FILE"`) or quote the variable. |
//! | `output` | string, required | The parser for its stdout: `claude-stream-json`, `codex-json`, `opencode-json`, `gemini-text`, `goose-stream-json`, `fx-json`, `aider-text`, `kimi-stream-json`, `mcode-stream-json`, `muse-json`, `openclaw-json`, `mini-swe-agent-trajectory`, `swe-agent-trajectory`, `trae-trajectory`, `hermes-session`, `code-puppy-text`, `cursor-stream-json`, `antigravity-stream-json`, `grok-streaming-json`, `copilot-json`, `pi-json`, `cline-json`, `openhands-json`, `qwen-session`, `vibe-streaming`, `benchflow-runner` (BenchFlow's runner scripts), or `text` (every line is the agent's message). The trajectory formats read a file the agent writes as it works, through a follower script BenchFlow writes to `~/.benchflow/` (see `headless/parsers/follow.rs`). |
//! | `env` | table of strings | Variables for the command. Not secrets: logins add their own. |
//! | `files` | array of tables | Files written in the agent's home before it starts, each `{ path = ".gemini/settings.json", contents = "...", mode = 0o600 }`; `path` is relative to the home, `mode` defaults to `0o644`. Instead of `contents`, `runner = "run_strands.py"` writes one of BenchFlow's runner scripts ([`crate::runners`]). |
//!
//! # `[acp]`
//!
//! Protocol extensions to ask the agent for. Only set these for an adapter
//! version they were checked against.
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `typed_failures` | bool | Ask for typed session failures (the `sessionFailure` extension of claude-agent-acp 0.73+), so a spent subscription arrives as its own failure instead of an opaque `-32603`. |
//! | `subagent_transcript` | bool | Ask for claude-agent-acp's same-session subagent transcript. |
//!
//! # `[providers]`
//!
//! How to point the agent at a provider's endpoint (see
//! [`crate::providers`]): one table per protocol the agent can speak,
//! `[providers.anthropic]` (the Anthropic Messages API) and
//! `[providers.openai]` (an OpenAI-compatible API), in the order the agent
//! prefers them. A run on `<provider>/<model>` uses the first one the
//! provider serves; an agent with neither cannot use providers. The agent
//! never sees the provider's key: the variables below get the address of
//! BenchFlow's forwarder and a placeholder key.
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `api` | string | With `openai`: the API the agent calls, `"chat"` (Chat Completions, the default) or `"responses"` (the Responses API). |
//! | `base_url` | string | The variable that takes the endpoint's base URL (`ANTHROPIC_BASE_URL`). |
//! | `key` | string | The variable that takes the key; it gets a placeholder (`ANTHROPIC_AUTH_TOKEN`). |
//! | `clear` | strings | Variables set empty, so a login the agent would prefer cannot win (`ANTHROPIC_API_KEY`). |
//! | `env` | table of strings | More variables, as templates: `{base_url}`, `{key}`, `{model}` (the provider's model id) and `{effort}` are filled in (a model variable, a config file's text). |
//! | `model` | string | The model id BenchFlow selects over ACP, as a template. Default `"{model}"`; `""` selects none (the `env` templates carry it). |
//! | `effort` | string | The effort `{effort}` stands for when the run names none. Required when a template uses `{effort}`. |
//!
//! # `[options.<name>]`
//!
//! Settings a run passes with `bench run --ak <name>=<value>` (Harbor's
//! `--ak`), checked before any sandbox starts: an unknown name gets the
//! closest one, a wrong value says what the option takes. `bench agents
//! <name>` lists them, and each trial's config.json records the ones set
//! (`agent_options`). An option the run does not set is not passed. See
//! [`options`].
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `type` | string, required | `string`, `int`, `float`, `bool`, or `enum`. |
//! | `description` | string, required | One short line for `bench agents`. |
//! | `default` | value | What the agent does when the run does not set it (shown, not passed). |
//! | `choices` | strings | With `enum`: the values it takes. |
//! | `min`, `max` | numbers | With `int` and `float`: the range allowed. |
//! | `env` | string | Reaches the agent as this variable. |
//! | `flag` | string | Reaches the agent as these words in the launch or headless command, which says where with `"$@"` (the default ACP command takes them at its end). `{value}` is replaced by the value; without it the value follows as its own word; a `bool` adds the flag when true. |
//! | `file`, `key` | strings | Reaches the agent as `key` (dotted) in this `.json` or `.toml` file under its home, on top of the `[[headless.files]]` copy when there is one. |
//! | `install` | string | The package whose pinned version this option replaces (`@anthropic-ai/claude-code`); `type = "string"`, and `default` is the pinned version. `verify` lines naming it follow. The install, and so the agent's fingerprint, changes. |
//! | `harnesses` | strings | The harnesses it reaches the agent on (`["headless"]`). Default: every one; with another, the run is refused before it starts. |
//!
//! # `[resume]`
//!
//! Whether the agent's own session carries on into the next graded stage
//! with `bench run --resume-sessions` (Harbor's `--resume-trajectory`). See
//! [`resume`].
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `acp` | bool | Over ACP, a later stage calls `session/load` with the previous stage's session id when the agent offers `loadSession`. |
//! | `headless` | bool | Headless, a later stage runs with `$BENCHFLOW_RESUME=1`, which the command turns into its CLI's resume flag (`${BENCHFLOW_RESUME:+--continue}`). |
//! | `session_dirs` | strings, required with either | Folders under the home where the agent keeps sessions (`.claude/projects`); a stage resumes only when one is not empty. |
//! | `load` | string | The agent's session file format, `claude-jsonl` or `codex-rollout`, for `bench run --load-trajectory`: the file is placed where the CLI looks and the first prompt resumes it as a stage would. Needs `acp` or `headless`. |
//!
//! # `[resources]`
//!
//! The least the agent needs from its sandbox, whatever the task asks for.
//! A task that asks for less (or leaves the size to the sandbox backend's
//! default) gets the floor instead, and the trial's result.json says so
//! (`resources_raised`); a task that asks for more keeps its own.
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `min_memory` | string | Memory, as `"4 GB"` or `"512 MB"` (OpenClaw is killed for lack of memory at Daytona's default size). |
//! | `min_cpus` | integer | Whole CPUs. |
//!
//! Unknown keys anywhere are errors, with a suggestion when one is close.
//!
//! # Not ported from 0.8
//!
//! 0.8's LiteLLM gateway (`opencode-proxy`, `BENCHFLOW_PROVIDER_*` env
//! mapping, provider registries) and its `acpx/` protocol and session
//! factories are not part of this schema.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::providers::{OpenAiApi, Protocol};

pub mod distribution;
mod harness;
pub(crate) mod install;
pub(crate) mod keys;
pub mod options;
mod registry;
pub mod resume;

pub use distribution::{ArchiveKind, BinaryInstall, BinaryTarget, NpxInstall, UvxInstall};
pub use harness::{Harness, HeadlessSpec, HomeFile, RESERVED_ENV};
pub use options::{AgentOption, OptionError, OptionKind, OptionRoute, OptionValue, OptionValues};
pub use registry::{Registry, UnknownAgent};
pub use resume::{ResumeSpec, SessionFile};

/// The least an agent needs from its sandbox (`[resources]`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ResourceFloor {
    pub memory_mb: Option<u64>,
    pub cpus: Option<u32>,
}

impl ResourceFloor {
    pub fn is_empty(&self) -> bool {
        self.memory_mb.is_none() && self.cpus.is_none()
    }
}

/// `"4 GB"`, `"512 MB"` (also `G`, `GiB`, `M`, `MiB`) in MB.
fn memory_mb(text: &str) -> Option<u64> {
    let text = text.trim();
    let split = text.find(|c: char| !(c.is_ascii_digit() || c == '.'))?;
    let (number, unit) = text.split_at(split);
    let n: f64 = number.parse().ok().filter(|n: &f64| *n > 0.0)?;
    let mult = match unit.trim() {
        "MB" | "M" | "MiB" => 1.0,
        "GB" | "G" | "GiB" => 1024.0,
        _ => return None,
    };
    Some((n * mult).ceil() as u64)
}

/// Seconds an install may take when the manifest does not say (0.8's default).
pub const DEFAULT_INSTALL_TIMEOUT: Duration = Duration::from_secs(900);

/// One agent's manifest, validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentManifest {
    pub name: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub install: Install,
    pub launch: Launch,
    pub credentials: CredentialFamily,
    /// The vendor account the login comes from, and how to get its key
    /// (`[credentials] account`).
    pub account: Option<String>,
    pub model: ModelSpec,
    pub acp: AcpSpec,
    pub skill_paths: Vec<String>,
    /// How the CLI finds the skills beyond the folder (`skills_via`), for
    /// `bench agents`.
    pub skills_via: Option<String>,
    pub home_dirs: Vec<String>,
    /// The harness used unless the run picks another.
    pub harness: Harness,
    /// Every harness the manifest declares, in [`Harness::ALL`] order.
    pub harnesses: Vec<Harness>,
    /// How to run the agent headless, when it declares that harness.
    pub headless: Option<HeadlessSpec>,
    /// How to point the agent at a provider's endpoint, in the order the
    /// agent prefers them (`[providers]`).
    pub providers: Vec<AgentProviderSpec>,
    /// The same agent's id in the public ACP registry.
    pub acp_registry: Option<String>,
    /// Settings a run may pass with `--ak` (`[options.<name>]`).
    pub options: Vec<AgentOption>,
    /// The options applied to this manifest ([`Self::with_options`]); not a
    /// manifest key.
    pub chosen_options: OptionValues,
    /// Whether the agent's own session can be resumed on a later stage.
    pub resume: ResumeSpec,
    /// What the sandbox must have at least (`[resources]`).
    pub resources: ResourceFloor,
    /// Facts about this manifest worth showing with it (how an `acp:`
    /// manifest was built, which login it borrowed). Not a manifest key.
    pub notes: Vec<String>,
    /// Where the manifest came from, for `bench agents` and error messages.
    pub source: ManifestSource,
}

/// How an agent is pointed at a provider endpoint (`[providers.<protocol>]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentProviderSpec {
    pub protocol: Protocol,
    /// With `openai`: the API the agent calls.
    pub api: Option<OpenAiApi>,
    /// The variable that takes the base URL.
    pub base_url: Option<String>,
    /// The variable that takes the (placeholder) key.
    pub key: Option<String>,
    /// Variables set empty.
    pub clear: Vec<String>,
    /// More variables, as templates.
    pub env: IndexMap<String, String>,
    /// The model id selected over ACP, as a template; empty selects none.
    pub model: String,
    /// What `{effort}` stands for when the run names no effort.
    pub effort: Option<String>,
}

impl AgentProviderSpec {
    /// For people: `Anthropic-compatible`, `OpenAI-compatible (Responses
    /// API)`.
    pub fn describe(&self) -> String {
        match self.api {
            Some(api) => format!("{} ({})", self.protocol.describe(), api.describe()),
            None => self.protocol.describe().to_string(),
        }
    }
}

/// How the agent is installed into the image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Install {
    pub method: InstallMethod,
    /// How long the install may take.
    pub timeout: Duration,
}

/// The ways to install an agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InstallMethod {
    /// An npm package installed with BenchFlow's private Node.js.
    Npm(NpmInstall),
    /// A shell script, run as given.
    Script(String),
    /// A Python CLI from PyPI, installed with a pinned uv.
    UvTool(UvToolInstall),
    /// A vendor installer that writes to `$HOME`.
    Installer(HomeInstall),
    /// A per-platform archive (an ACP registry `binary` distribution).
    Binary(BinaryInstall),
    /// An npm package run as `npx` would (an ACP registry `npx` distribution).
    Npx(NpxInstall),
    /// A pinned PyPI package installed with uv (an ACP registry `uvx`
    /// distribution).
    Uvx(UvxInstall),
    /// A Python framework in its own virtual environment.
    UvVenv(UvVenvInstall),
}

/// A Python framework installed into `/opt/benchflow/agents/<name>/venv`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UvVenvInstall {
    /// Pinned requirements, each `name==version` or `name @ <url>`.
    pub requirements: Vec<String>,
    pub python: Option<String>,
    pub verify: Option<String>,
}

/// A Python CLI installed with `uv tool install`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UvToolInstall {
    /// The exact requirement, `name==version`.
    pub requirement: String,
    pub bin: String,
    pub python: Option<String>,
    pub verify: Option<String>,
}

/// A vendor installer run with `HOME=/opt/benchflow/agents/<name>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HomeInstall {
    pub script: String,
    /// Executables it leaves, relative to its home, linked into `/opt/benchflow/bin`.
    pub links: Vec<String>,
    pub verify: Option<String>,
}

/// An npm-distributed agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NpmInstall {
    pub package: String,
    pub bin: String,
    pub companions: Vec<String>,
    pub launcher: Launcher,
    pub launcher_env: IndexMap<String, String>,
    pub verify: Option<String>,
}

/// How the launcher BenchFlow writes starts the agent's executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Launcher {
    /// Run it on BenchFlow's private Node.js.
    #[default]
    Node,
    /// Run a `#!` script on BenchFlow's Node.js and anything else directly.
    Auto,
}

/// How the agent process starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Launch {
    /// The shell command; `None` uses the npm launcher with `args`.
    pub command: Option<String>,
    pub args: Vec<String>,
    /// Non-secret variables for the agent process.
    pub env: IndexMap<String, String>,
    /// Files written in the agent's home before it starts: those agent
    /// options set ([`OptionRoute::File`]); not a manifest key.
    pub files: Vec<HomeFile>,
}

/// Which logins an agent accepts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialFamily {
    /// Claude Code: Anthropic API keys, Claude OAuth tokens, `claude login`.
    Anthropic,
    /// Codex: OpenAI API keys, Codex access tokens, `codex login`.
    OpenAi,
    /// Gemini CLI: Gemini API keys, Google sign-in.
    Gemini,
    /// The provider in the model's `provider/model` id decides (OpenCode).
    Model,
    /// The first of these variables that is set.
    Env(Vec<String>),
    /// The agent needs no login from BenchFlow.
    None,
}

/// How BenchFlow selects the model and effort.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelSpec {
    pub select: ModelSelect,
    pub config_id: String,
    pub effort_config_id: Option<String>,
    pub format: ModelFormat,
    pub default: Option<String>,
    pub strip_prefixes: Vec<String>,
}

/// The ACP request that carries the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelSelect {
    /// The session's `model` config option when it has one, else `session/set_model`.
    #[default]
    Auto,
    /// Always `session/set_config_option` with the manifest's `config_id`.
    ConfigOption,
    /// Always `session/set_model`.
    SetModel,
    /// BenchFlow does not select a model.
    None,
}

/// How a model id is written for the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ModelFormat {
    /// As given.
    #[default]
    #[serde(rename = "bare")]
    Bare,
    /// models.dev `provider/model`.
    #[serde(rename = "provider/model")]
    ProviderModel,
    /// Codex's `model[effort]`, chosen from the session's advertised models.
    #[serde(rename = "model[effort]")]
    ModelEffort,
}

/// ACP extensions to request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AcpSpec {
    pub typed_failures: bool,
    pub subagent_transcript: bool,
}

/// Where a manifest came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestSource {
    /// Embedded in the binary; the file name under `crates/benchflow-agent/agents/`.
    Builtin(&'static str),
    /// A built-in provider; the file name under `crates/benchflow-agent/providers/`.
    BuiltinProvider(&'static str),
    /// A project's manifest file.
    File(PathBuf),
    /// Built from an entry of the public ACP registry (`acp:goose@1.54.0`).
    AcpRegistry(String),
}

impl fmt::Display for ManifestSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Builtin(file) => write!(f, "built-in agents/{file}"),
            Self::BuiltinProvider(file) => write!(f, "built-in providers/{file}"),
            Self::File(path) => write!(f, "{}", path.display()),
            Self::AcpRegistry(spec) => write!(f, "the public ACP registry ({spec})"),
        }
    }
}

/// Why a manifest could not be used. Every variant names the manifest and,
/// when it is in the file, the line and column; the fix is the manifest
/// author's.
#[derive(Debug)]
pub enum ManifestError {
    /// The file could not be read.
    Read {
        path: PathBuf,
        error: std::io::Error,
    },
    /// The file is not valid TOML, or a value has the wrong type.
    Syntax {
        origin: ManifestSource,
        line: usize,
        column: usize,
        message: String,
    },
    /// A key the schema does not have.
    UnknownKey(Box<UnknownKeyError>),
    /// A value the schema does not accept.
    Invalid {
        origin: ManifestSource,
        line: usize,
        column: usize,
        key: String,
        message: String,
    },
    /// Two manifests claim the same name or alias.
    Duplicate {
        name: String,
        first: ManifestSource,
        second: ManifestSource,
    },
}

/// Details of [`ManifestError::UnknownKey`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownKeyError {
    pub origin: ManifestSource,
    pub line: usize,
    pub column: usize,
    /// The full dotted path, such as `install.timout`.
    pub key: String,
    pub suggestion: Option<String>,
    /// The table the key is in, as written in TOML (`[install]`).
    pub table: String,
    pub allowed: Vec<String>,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, error } => {
                write!(f, "cannot read agent manifest {}: {error}", path.display())
            }
            Self::Syntax {
                origin,
                line,
                column,
                message,
            } => write!(f, "{origin}{}: {message}", at(*line, *column)),
            Self::UnknownKey(e) => {
                write!(
                    f,
                    "{}{}: unknown key `{}`",
                    e.origin,
                    at(e.line, e.column),
                    e.key
                )?;
                if let Some(suggestion) = &e.suggestion {
                    write!(f, "; did you mean `{suggestion}`?")?;
                }
                write!(f, " ({} takes {})", e.table, e.allowed.join(", "))
            }
            Self::Invalid {
                origin,
                line,
                column,
                key,
                message,
            } => write!(f, "{origin}{}: `{key}`: {message}", at(*line, *column)),
            Self::Duplicate {
                name,
                first,
                second,
            } => write!(
                f,
                "agent name `{name}` is claimed twice: by {first} and by {second}; rename one of them"
            ),
        }
    }
}

impl std::error::Error for ManifestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// `:line:column`, or nothing when the position is unknown (line 0).
fn at(line: usize, column: usize) -> String {
    if line == 0 {
        String::new()
    } else {
        format!(":{line}:{column}")
    }
}

/// The built-in manifests, in the order `bench agents` lists them.
const BUILTINS: &[(&str, &str)] = &[
    ("claude.toml", include_str!("../agents/claude.toml")),
    ("codex.toml", include_str!("../agents/codex.toml")),
    ("gemini.toml", include_str!("../agents/gemini.toml")),
    ("opencode.toml", include_str!("../agents/opencode.toml")),
    ("aider.toml", include_str!("../agents/aider.toml")),
    ("fx.toml", include_str!("../agents/fx.toml")),
    ("fx-dev.toml", include_str!("../agents/fx-dev.toml")),
    // Every other agent that speaks ACP, from the agent catalog: the ACP
    // registry's builds, and BenchFlow 0.8's ACP agents.
    ("goose.toml", include_str!("../agents/goose.toml")),
    ("kimi.toml", include_str!("../agents/kimi.toml")),
    ("qwen-code.toml", include_str!("../agents/qwen-code.toml")),
    ("cline.toml", include_str!("../agents/cline.toml")),
    ("pi.toml", include_str!("../agents/pi.toml")),
    (
        "mistral-vibe.toml",
        include_str!("../agents/mistral-vibe.toml"),
    ),
    ("grok.toml", include_str!("../agents/grok.toml")),
    ("deepagents.toml", include_str!("../agents/deepagents.toml")),
    ("mimo.toml", include_str!("../agents/mimo.toml")),
    ("openhands.toml", include_str!("../agents/openhands.toml")),
    ("cursor.toml", include_str!("../agents/cursor.toml")),
    ("copilot.toml", include_str!("../agents/copilot.toml")),
    ("junie.toml", include_str!("../agents/junie.toml")),
    (
        "cortex-code.toml",
        include_str!("../agents/cortex-code.toml"),
    ),
    (
        "antigravity.toml",
        include_str!("../agents/antigravity.toml"),
    ),
    (
        "mini-swe-agent.toml",
        include_str!("../agents/mini-swe-agent.toml"),
    ),
    ("swe-agent.toml", include_str!("../agents/swe-agent.toml")),
    ("trae-agent.toml", include_str!("../agents/trae-agent.toml")),
    ("hermes.toml", include_str!("../agents/hermes.toml")),
    ("openclaw.toml", include_str!("../agents/openclaw.toml")),
    ("kimi-code.toml", include_str!("../agents/kimi-code.toml")),
    ("mcode.toml", include_str!("../agents/mcode.toml")),
    ("code-puppy.toml", include_str!("../agents/code-puppy.toml")),
    (
        "rovodev-cli.toml",
        include_str!("../agents/rovodev-cli.toml"),
    ),
    ("devin.toml", include_str!("../agents/devin.toml")),
    ("muse-code.toml", include_str!("../agents/muse-code.toml")),
    (
        "openhands-sdk.toml",
        include_str!("../agents/openhands-sdk.toml"),
    ),
    ("nemo-agent.toml", include_str!("../agents/nemo-agent.toml")),
    ("strands.toml", include_str!("../agents/strands.toml")),
    ("deerflow.toml", include_str!("../agents/deerflow.toml")),
    (
        "antigravity-sdk.toml",
        include_str!("../agents/antigravity-sdk.toml"),
    ),
    ("langgraph.toml", include_str!("../agents/langgraph.toml")),
    ("eve.toml", include_str!("../agents/eve.toml")),
];

impl AgentManifest {
    /// Parse and validate a manifest's text.
    pub fn parse(text: &str, source: ManifestSource) -> Result<Self, ManifestError> {
        let doc = keys::Document::parse(text, &source)?;
        doc.check_keys()?;
        let raw: RawManifest = toml::from_str(text).map_err(|e| {
            let (line, column) = e
                .span()
                .map(|span| doc.position(span.start))
                .unwrap_or((0, 0));
            ManifestError::Syntax {
                origin: source.clone(),
                line,
                column,
                message: e.message().trim().to_string(),
            }
        })?;
        raw.validate(&doc, source)
    }

    /// Read and validate a manifest file.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(|error| ManifestError::Read {
            path: path.to_path_buf(),
            error,
        })?;
        Self::parse(&text, ManifestSource::File(path.to_path_buf()))
    }

    /// The built-in manifests. They are checked by this crate's tests, so a
    /// failure here is a bug in BenchFlow.
    pub fn builtins() -> Vec<Self> {
        BUILTINS
            .iter()
            .map(|(file, text)| {
                Self::parse(text, ManifestSource::Builtin(file))
                    .unwrap_or_else(|e| panic!("BenchFlow bug: invalid built-in manifest: {e}"))
            })
            .collect()
    }

    /// True when `name` is this agent's name or one of its aliases.
    pub fn answers_to(&self, name: &str) -> bool {
        self.name == name || self.aliases.iter().any(|a| a == name)
    }

    /// The shell command that starts the agent. It reads its environment from
    /// [`Self::launch_env`] plus the login's variables; run it with that
    /// environment as the agent user and speak ACP over its stdin and stdout.
    pub fn launch_command(&self) -> String {
        if let Some(command) = &self.launch.command {
            return command.clone();
        }
        // Validation guarantees a launcher here (other installs need a command).
        let bin = match &self.install.method {
            InstallMethod::Npm(npm) => npm.bin.as_str(),
            InstallMethod::UvTool(uv) => uv.bin.as_str(),
            InstallMethod::Script(_) | InstallMethod::Installer(_) | InstallMethod::UvVenv(_) => "",
            InstallMethod::Binary(b) => b.dir.as_str(),
            InstallMethod::Npx(n) => n.dir.as_str(),
            InstallMethod::Uvx(u) => u.dir.as_str(),
        };
        let mut command = format!("{}/{bin}", install::BIN_PREFIX);
        for arg in &self.launch.args {
            command.push(' ');
            command.push_str(&crate::shell::quote(arg));
        }
        command
    }

    /// Non-secret variables for the agent process (`[launch] env`).
    pub fn launch_env(&self) -> &IndexMap<String, String> {
        &self.launch.env
    }

    /// The shell script that installs the agent: POSIX `sh`, run once as root
    /// as a cached image layer. Idempotent and non-interactive; it needs no
    /// secrets.
    pub fn install_script(&self) -> String {
        match &self.install.method {
            InstallMethod::Npm(npm) => install::npm_install_script(npm),
            InstallMethod::Script(script) => script.clone(),
            InstallMethod::UvTool(uv) => install::uv_tool_install_script(uv),
            InstallMethod::Installer(home) => install::home_install_script(&self.name, home),
            InstallMethod::Binary(b) => distribution::binary_script(b),
            InstallMethod::Npx(n) => distribution::npx_script(n),
            InstallMethod::Uvx(u) => distribution::uvx_script(u),
            InstallMethod::UvVenv(venv) => install::uv_venv_install_script(&self.name, venv),
        }
    }

    /// True when the manifest declares `harness`.
    pub fn declares(&self, harness: Harness) -> bool {
        self.harnesses.contains(&harness)
    }

    /// The harness a run uses: `requested` when the manifest declares it,
    /// else the manifest's default. The error says what to pass instead.
    pub fn pick_harness(&self, requested: Option<Harness>) -> Result<Harness, String> {
        match requested {
            None => Ok(self.harness),
            Some(h) if self.declares(h) => Ok(h),
            Some(h) => Err(format!(
                "agent `{}` has no {h} harness; it supports {}",
                self.name,
                self.harnesses
                    .iter()
                    .map(|h| h.as_str())
                    .collect::<Vec<_>>()
                    .join(" and ")
            )),
        }
    }

    /// `model` as the agent's own CLI takes it (`$BENCHFLOW_MODEL` for a
    /// headless command): the manifest's prefixes stripped, then written in
    /// its `format` (`model[effort]` ids stay bare).
    pub fn cli_model(&self, model: &str) -> String {
        crate::model_id::format_model(&self.model, model)
    }

    /// A hash of everything that decides the installed bits (the install
    /// script, which embeds every pin and the launcher), as 64 hex digits.
    /// Equal fingerprints mean an image layer built for one serves the other.
    pub fn version_fingerprint(&self) -> String {
        install::fingerprint(&self.install_script())
    }

    /// The model to use when the run names none.
    pub fn default_model(&self) -> Option<&str> {
        self.model.default.as_deref()
    }

    /// Where the agent discovers skills, with `$HOME` and `$WORKSPACE`
    /// replaced by the agent user's home and the task's working folder:
    /// the run copies skills into each.
    pub fn skill_dirs(&self, home: &str, workspace: &str) -> Vec<String> {
        self.skill_paths
            .iter()
            .map(|path| {
                let expand = |prefix: &str, with: &str| {
                    path.strip_prefix(prefix)
                        .map(|rest| format!("{}{rest}", with.trim_end_matches('/')))
                };
                expand("$HOME", home)
                    .or_else(|| expand("$WORKSPACE", workspace))
                    .unwrap_or_else(|| path.clone())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Raw TOML shape and validation
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    description: String,
    #[serde(default)]
    skill_paths: Vec<String>,
    skills_via: Option<String>,
    #[serde(default)]
    home_dirs: Vec<String>,
    harness: Option<String>,
    acp_registry: Option<String>,
    install: RawInstall,
    launch: Option<RawLaunch>,
    headless: Option<harness::RawHeadless>,
    #[serde(default)]
    credentials: RawCredentials,
    #[serde(default)]
    model: RawModel,
    #[serde(default)]
    acp: AcpSpec,
    #[serde(default)]
    providers: IndexMap<String, RawAgentProvider>,
    #[serde(default)]
    options: IndexMap<String, options::RawOption>,
    resume: Option<resume::RawResume>,
    resources: Option<RawResources>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawResources {
    min_memory: Option<String>,
    min_cpus: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentProvider {
    api: Option<String>,
    base_url: Option<String>,
    key: Option<String>,
    #[serde(default)]
    clear: Vec<String>,
    #[serde(default)]
    env: IndexMap<String, String>,
    model: Option<String>,
    effort: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInstall {
    npm: Option<String>,
    bin: Option<String>,
    #[serde(default)]
    companions: Vec<String>,
    launcher: Option<Launcher>,
    #[serde(default)]
    launcher_env: IndexMap<String, String>,
    verify: Option<String>,
    script: Option<String>,
    #[serde(rename = "uv-tool")]
    uv_tool: Option<String>,
    #[serde(rename = "uv-venv")]
    uv_venv: Option<Vec<String>>,
    python: Option<String>,
    installer: Option<String>,
    #[serde(default)]
    links: Vec<String>,
    timeout: Option<u64>,
    binary: Option<IndexMap<String, distribution::RawBinaryTarget>>,
    npx: Option<String>,
    uvx: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLaunch {
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: IndexMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCredentials {
    family: Option<String>,
    #[serde(default)]
    env: Vec<String>,
    account: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    #[serde(default)]
    select: ModelSelect,
    config_id: Option<String>,
    effort_config_id: Option<String>,
    #[serde(default)]
    format: ModelFormat,
    default: Option<String>,
    #[serde(default)]
    strip_prefixes: Vec<String>,
}

/// Builds `Invalid` errors that point at a key's position.
pub(crate) struct Checker<'a> {
    pub(crate) doc: &'a keys::Document<'a>,
    pub(crate) source: &'a ManifestSource,
}

impl Checker<'_> {
    pub(crate) fn fail(&self, key: &str, message: impl Into<String>) -> ManifestError {
        let (line, column) = self.doc.key_position(key);
        ManifestError::Invalid {
            origin: self.source.clone(),
            line,
            column,
            key: key.to_string(),
            message: message.into(),
        }
    }

    pub(crate) fn require(
        &self,
        ok: bool,
        key: &str,
        message: impl FnOnce() -> String,
    ) -> Result<(), ManifestError> {
        if ok {
            Ok(())
        } else {
            Err(self.fail(key, message()))
        }
    }
}

/// An agent name or alias: lowercase letters, digits and `-`, starting with a
/// letter or digit.
fn is_agent_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// An environment variable name.
pub(crate) fn is_env_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A file name usable in `/opt/benchflow/bin/<bin>`.
fn is_bin_name(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// An npm spec BenchFlow can pass to `npm install` (quoted when needed).
fn is_npm_spec(s: &str) -> bool {
    !s.is_empty() && !s.starts_with('-') && !s.chars().any(|c| c.is_whitespace() || c.is_control())
}

impl RawManifest {
    fn validate(
        self,
        doc: &keys::Document<'_>,
        source: ManifestSource,
    ) -> Result<AgentManifest, ManifestError> {
        let check = Checker {
            doc,
            source: &source,
        };

        check.require(is_agent_name(&self.name), "name", || {
            format!(
                "`{}` is not a valid agent name; use lowercase letters, digits and `-`",
                self.name
            )
        })?;
        check.require(!crate::builtin::is_scripted(&self.name), "name", || {
            format!(
                "`{}` is reserved for a scripted agent; pick another name",
                self.name
            )
        })?;
        for (i, alias) in self.aliases.iter().enumerate() {
            let key = format!("aliases[{i}]");
            check.require(is_agent_name(alias), &key, || {
                format!(
                    "`{alias}` is not a valid agent name; use lowercase letters, digits and `-`"
                )
            })?;
            check.require(!crate::builtin::is_scripted(alias), &key, || {
                format!("`{alias}` is reserved for a scripted agent")
            })?;
            check.require(
                alias != &self.name && !self.aliases[..i].contains(alias),
                &key,
                || format!("`{alias}` is listed twice"),
            )?;
        }
        for (i, path) in self.skill_paths.iter().enumerate() {
            check.require(
                path.starts_with("$HOME/") || path.starts_with("$WORKSPACE/"),
                &format!("skill_paths[{i}]"),
                || format!("`{path}` must start with `$HOME/` or `$WORKSPACE/`"),
            )?;
        }
        check.require(
            self.skills_via.is_none() || !self.skill_paths.is_empty(),
            "skills_via",
            || "needs `skill_paths`, the folders the skills are copied into".to_string(),
        )?;
        for (i, dir) in self.home_dirs.iter().enumerate() {
            check.require(
                !dir.is_empty()
                    && !dir.starts_with('/')
                    && !dir.split('/').any(|part| part == ".."),
                &format!("home_dirs[{i}]"),
                || format!("`{dir}` must be a folder under the home directory, such as `.claude`"),
            )?;
        }

        if let Some(id) = &self.acp_registry {
            check.require(is_agent_name(id), "acp_registry", || {
                format!("`{id}` is not an ACP registry id (lowercase letters, digits and `-`)")
            })?;
        }
        let install = self.install.validate(&check, &self.name)?;
        let acp = self.launch.is_some() || self.headless.is_none();
        let launch = match self.launch {
            Some(launch) => launch.validate(&check, &install)?,
            None if acp => RawLaunch::default().validate(&check, &install)?,
            None => Launch {
                command: None,
                args: Vec::new(),
                env: IndexMap::new(),
                files: Vec::new(),
            },
        };
        let headless = self.headless.map(|h| h.validate(&check)).transpose()?;
        let (harness, harnesses) =
            harness::resolve(&check, self.harness.as_deref(), acp, headless.is_some())?;
        let mut raw_credentials = self.credentials;
        let account = raw_credentials.account.take();
        if let Some(account) = &account {
            check.require(!account.trim().is_empty(), "credentials.account", || {
                "is empty".into()
            })?;
        }
        let credentials = raw_credentials.validate(&check)?;
        let model = self.model.validate(&check)?;
        let mut providers = Vec::new();
        for (name, raw) in self.providers {
            providers.push(raw.validate(&check, &name)?);
        }
        let parts = options::Parts {
            install: &install,
            launch: &launch,
            headless: headless.as_ref(),
            harnesses: &harnesses,
        };
        let options = options::validate(self.options, &check, &parts)?;
        let resume = resume::validate(self.resume, &check, &parts)?;
        let mut resources = ResourceFloor::default();
        if let Some(raw) = self.resources {
            if let Some(text) = &raw.min_memory {
                resources.memory_mb = Some(memory_mb(text).ok_or_else(|| {
                    check.fail(
                        "resources.min_memory",
                        format!("`{text}` is not a size; write it as \"4 GB\" or \"512 MB\""),
                    )
                })?);
            }
            if let Some(cpus) = raw.min_cpus {
                check.require(cpus > 0, "resources.min_cpus", || {
                    "must be at least 1".to_string()
                })?;
                resources.cpus = Some(cpus);
            }
        }

        Ok(AgentManifest {
            name: self.name,
            aliases: self.aliases,
            description: self.description,
            install,
            launch,
            credentials,
            account,
            model,
            acp: self.acp,
            skill_paths: self.skill_paths,
            skills_via: self.skills_via,
            home_dirs: self.home_dirs,
            harness,
            harnesses,
            headless,
            providers,
            acp_registry: self.acp_registry,
            options,
            chosen_options: OptionValues::new(),
            resume,
            resources,
            notes: Vec::new(),
            source,
        })
    }
}

impl RawInstall {
    fn validate(mut self, check: &Checker<'_>, name: &str) -> Result<Install, ManifestError> {
        let timeout = match self.timeout {
            Some(0) => return Err(check.fail("install.timeout", "must be at least 1 second")),
            Some(secs) => Duration::from_secs(secs),
            None => DEFAULT_INSTALL_TIMEOUT,
        };
        if let Some(method) = self.distribution(check, name)? {
            return Ok(Install { method, timeout });
        }
        let methods: Vec<&str> = [
            ("npm", self.npm.is_some()),
            ("script", self.script.is_some()),
            ("uv-tool", self.uv_tool.is_some()),
            ("uv-venv", self.uv_venv.is_some()),
            ("installer", self.installer.is_some()),
        ]
        .into_iter()
        .filter_map(|(key, set)| set.then_some(key))
        .collect();
        match methods.as_slice() {
            [] => {
                return Err(check.fail(
                    "install",
                    "say how to install the agent: `npm = \"<package>@<version>\"` with `bin`, \
                     `uv-tool = \"<package>==<version>\"` with `bin`, `uv-venv = [\"<package>==<version>\"]`, \
                     `installer` with `links`, `script`, or a registry distribution (`binary`, `npx`, `uvx`)",
                ));
            }
            [_] => {}
            [first, second, ..] => {
                return Err(check.fail(
                    &format!("install.{second}"),
                    format!("use either `{first}` or `{second}`, not both"),
                ));
            }
        }
        let method = methods[0];
        // Keys that only some methods read.
        for (key, set, methods_taking) in [
            ("bin", self.bin.is_some(), &["npm", "uv-tool"][..]),
            ("companions", !self.companions.is_empty(), &["npm"][..]),
            ("launcher", self.launcher.is_some(), &["npm"][..]),
            ("launcher_env", !self.launcher_env.is_empty(), &["npm"][..]),
            (
                "verify",
                self.verify.is_some(),
                &["npm", "uv-tool", "uv-venv", "installer"][..],
            ),
            (
                "python",
                self.python.is_some(),
                &["uv-tool", "uv-venv", "uvx"][..],
            ),
            ("links", !self.links.is_empty(), &["installer"][..]),
        ] {
            if set && !methods_taking.contains(&method) {
                let takers: Vec<String> = methods_taking.iter().map(|m| format!("`{m}`")).collect();
                let takers = match takers.as_slice() {
                    [one] => format!("{one} installs"),
                    [a, b] => format!("{a} or {b} installs"),
                    [rest @ .., last] => format!("{}, or {last} installs", rest.join(", ")),
                    [] => String::new(),
                };
                let fix = if method == "script" {
                    "put the step in `script` instead".to_string()
                } else {
                    format!("remove it from this `{method}` install")
                };
                return Err(check.fail(
                    &format!("install.{key}"),
                    format!("only applies to {takers}; {fix}"),
                ));
            }
        }
        if let Some(verify) = &self.verify {
            check.require(!verify.trim().is_empty(), "install.verify", || {
                "is empty".into()
            })?;
        }
        let require_bin = |check: &Checker<'_>, bin: Option<String>, what: &str| {
            let Some(bin) = bin else {
                return Err(check.fail(
                    "install",
                    format!(
                        "`bin` is required with `{what}`: the name of the executable it installs"
                    ),
                ));
            };
            check.require(is_bin_name(&bin), "install.bin", || {
                format!("`{bin}` is not a file name")
            })?;
            Ok(bin)
        };
        if let Some(python) = &self.python {
            check.require(
                !python.is_empty()
                    && python
                        .split('.')
                        .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())),
                "install.python",
                || format!("`{python}` is not a Python version; write it like `3.12`"),
            )?;
        }
        if let Some(requirements) = self.uv_venv {
            check.require(!requirements.is_empty(), "install.uv-venv", || {
                "lists no packages; write `uv-venv = [\"<package>==<version>\"]`".into()
            })?;
            for (i, requirement) in requirements.iter().enumerate() {
                check.require(
                    is_pinned_requirement(requirement) || is_pinned_url(requirement),
                    &format!("install.uv-venv[{i}]"),
                    || {
                        format!(
                            "`{requirement}` is not pinned; write `<package>==<version>`, such as `strands-agents==1.59.0`, \
                             or `<package> @ <url>` with a 40-character commit in the URL"
                        )
                    },
                )?;
            }
            let method = InstallMethod::UvVenv(UvVenvInstall {
                requirements,
                python: self.python,
                verify: self.verify,
            });
            return Ok(Install { method, timeout });
        }
        let method = match (self.npm, self.script, self.uv_tool, self.installer) {
            (None, Some(script), None, None) => {
                check.require(!script.trim().is_empty(), "install.script", || {
                    "is empty".into()
                })?;
                InstallMethod::Script(script)
            }
            (None, None, Some(requirement), None) => {
                check.require(is_pinned_requirement(&requirement), "install.uv-tool", || {
                    format!(
                        "`{requirement}` is not an exactly pinned PyPI requirement; write `<package>==<version>`, such as `aider-chat==0.86.2`"
                    )
                })?;
                let bin = require_bin(check, self.bin, "uv-tool")?;
                InstallMethod::UvTool(UvToolInstall {
                    requirement,
                    bin,
                    python: self.python,
                    verify: self.verify,
                })
            }
            (None, None, None, Some(script)) => {
                check.require(!script.trim().is_empty(), "install.installer", || {
                    "is empty".into()
                })?;
                check.require(!self.links.is_empty(), "install", || {
                    "`links` is required with `installer`: the executables it leaves under its home, such as `[\".local/bin/fx\"]`".into()
                })?;
                let mut names: Vec<&str> = Vec::new();
                for (i, link) in self.links.iter().enumerate() {
                    let key = format!("install.links[{i}]");
                    let name = link.rsplit('/').next().unwrap_or_default();
                    check.require(
                        !link.starts_with('/')
                            && !link.split('/').any(|part| part.is_empty() || part == "..")
                            && is_bin_name(name),
                        &key,
                        || format!("`{link}` must be a file under the installer's home, such as `.local/bin/fx`"),
                    )?;
                    check.require(!names.contains(&name), &key, || {
                        format!("another link is also named `{name}`")
                    })?;
                    names.push(name);
                }
                InstallMethod::Installer(HomeInstall {
                    script,
                    links: self.links,
                    verify: self.verify,
                })
            }
            (Some(package), None, None, None) => {
                check.require(is_npm_spec(&package), "install.npm", || {
                    format!("`{package}` is not an npm package spec")
                })?;
                let bin = require_bin(check, self.bin, "npm")?;
                for (i, companion) in self.companions.iter().enumerate() {
                    check.require(
                        is_npm_spec(companion),
                        &format!("install.companions[{i}]"),
                        || format!("`{companion}` is not an npm package spec"),
                    )?;
                }
                for name in self.launcher_env.keys() {
                    check.require(
                        is_env_name(name),
                        &format!("install.launcher_env.{name}"),
                        || format!("`{name}` is not an environment variable name"),
                    )?;
                }
                InstallMethod::Npm(NpmInstall {
                    package,
                    bin,
                    companions: self.companions,
                    launcher: self.launcher.unwrap_or_default(),
                    launcher_env: self.launcher_env,
                    verify: self.verify,
                })
            }
            _ => unreachable!("exactly one install method was checked above"),
        };
        Ok(Install { method, timeout })
    }
}

/// `name @ https://...`: a direct reference whose URL names a 40-character
/// commit (a GitHub archive or a `git+https://...@<sha>` URL).
fn is_pinned_url(s: &str) -> bool {
    let Some((name, url)) = s.split_once(" @ ") else {
        return false;
    };
    let name = name.trim().split('[').next().unwrap_or_default();
    let has_commit = url
        .split(|c: char| !c.is_ascii_hexdigit())
        .any(|part| part.len() == 40);
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && (url.starts_with("https://") || url.starts_with("git+https://"))
        && has_commit
        && !url.chars().any(|c| c.is_whitespace() || c.is_control())
}

/// `name==1.2.3` or `name[extra]==1.2.3`: a requirement pinned to one version.
fn is_pinned_requirement(s: &str) -> bool {
    let Some((name, version)) = s.split_once("==") else {
        return false;
    };
    let name = name.split('[').next().unwrap_or_default();
    !name.is_empty()
        && !name.starts_with('-')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && !version.is_empty()
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '!'))
        && !s.chars().any(|c| c.is_whitespace() || c.is_control())
}

impl RawInstall {
    /// The `binary`, `npx`, or `uvx` install, when the manifest uses one.
    fn distribution(
        &mut self,
        check: &Checker<'_>,
        name: &str,
    ) -> Result<Option<InstallMethod>, ManifestError> {
        let chosen: Vec<&str> = [
            ("npm", self.npm.is_some()),
            ("script", self.script.is_some()),
            ("uv-tool", self.uv_tool.is_some()),
            ("uv-venv", self.uv_venv.is_some()),
            ("installer", self.installer.is_some()),
            ("binary", self.binary.is_some()),
            ("npx", self.npx.is_some()),
            ("uvx", self.uvx.is_some()),
        ]
        .into_iter()
        .filter_map(|(key, set)| set.then_some(key))
        .collect();
        let Some(&method) = chosen
            .iter()
            .find(|m| matches!(**m, "binary" | "npx" | "uvx"))
        else {
            return Ok(None);
        };
        if chosen.len() > 1 {
            return Err(check.fail(
                &format!("install.{}", chosen[1]),
                format!("use one install method, not {}", chosen.join(" and ")),
            ));
        }
        for (key, set) in [
            ("companions", !self.companions.is_empty()),
            ("launcher", self.launcher.is_some()),
            ("launcher_env", !self.launcher_env.is_empty()),
            ("bin", method == "binary" && self.bin.is_some()),
            ("python", method != "uvx" && self.python.is_some()),
        ] {
            if set {
                return Err(check.fail(
                    &format!("install.{key}"),
                    format!("does not apply to `{method}` installs"),
                ));
            }
        }
        if let Some(bin) = &self.bin {
            check.require(is_bin_name(bin), "install.bin", || {
                format!("`{bin}` is not a file name")
            })?;
        }
        if let Some(verify) = &self.verify {
            check.require(!verify.trim().is_empty(), "install.verify", || {
                "is empty".into()
            })?;
        }
        let verify = self.verify.take();
        let refused = |(key, message): distribution::Refusal| check.fail(&key, message);
        let method = match method {
            "binary" => InstallMethod::Binary(
                distribution::binary(name, self.binary.take().unwrap_or_default(), verify)
                    .map_err(refused)?,
            ),
            "npx" => InstallMethod::Npx(
                distribution::npx(
                    name,
                    self.npx.take().unwrap_or_default(),
                    self.bin.take(),
                    verify,
                )
                .map_err(refused)?,
            ),
            _ => InstallMethod::Uvx(
                distribution::uvx(
                    name,
                    self.uvx.take().unwrap_or_default(),
                    self.bin.take(),
                    self.python.take(),
                    verify,
                )
                .map_err(refused)?,
            ),
        };
        Ok(Some(method))
    }
}

impl RawLaunch {
    fn validate(self, check: &Checker<'_>, install: &Install) -> Result<Launch, ManifestError> {
        if let Some(command) = &self.command {
            check.require(!command.trim().is_empty(), "launch.command", || {
                "is empty".into()
            })?;
            check.require(self.args.is_empty(), "launch.args", || {
                "only applies to the default command; put the arguments in `command`".into()
            })?;
        } else if matches!(
            install.method,
            InstallMethod::Script(_) | InstallMethod::Installer(_) | InstallMethod::UvVenv(_)
        ) {
            return Err(check.fail(
                "launch",
                "`command` is required when the agent is installed with `script`, `installer` or `uv-venv`; \
                 a headless-only agent has a `[headless]` table and no `[launch]` table",
            ));
        }
        for name in self.env.keys() {
            check.require(is_env_name(name), &format!("launch.env.{name}"), || {
                format!("`{name}` is not an environment variable name")
            })?;
        }
        Ok(Launch {
            command: self.command,
            args: self.args,
            env: self.env,
            files: Vec::new(),
        })
    }
}

impl RawCredentials {
    fn validate(self, check: &Checker<'_>) -> Result<CredentialFamily, ManifestError> {
        for (i, name) in self.env.iter().enumerate() {
            check.require(is_env_name(name), &format!("credentials.env[{i}]"), || {
                format!("`{name}` is not an environment variable name")
            })?;
        }
        let family = match self.family.as_deref() {
            None if self.env.is_empty() => CredentialFamily::None,
            None | Some("env") => {
                check.require(!self.env.is_empty(), "credentials.env", || {
                    "list the variables the agent reads its key from".into()
                })?;
                CredentialFamily::Env(self.env)
            }
            Some(other) => {
                check.require(self.env.is_empty(), "credentials.env", || {
                    format!("only read with `family = \"env\"`, not `\"{other}\"`")
                })?;
                match other {
                    "anthropic" => CredentialFamily::Anthropic,
                    "openai" => CredentialFamily::OpenAi,
                    "gemini" => CredentialFamily::Gemini,
                    "model" => CredentialFamily::Model,
                    "none" => CredentialFamily::None,
                    _ => {
                        const FAMILIES: [&str; 6] =
                            ["anthropic", "openai", "gemini", "model", "env", "none"];
                        let hint = crate::suggest::did_you_mean(other, FAMILIES)
                            .map(|s| format!("did you mean `{s}`? "))
                            .unwrap_or_default();
                        return Err(check.fail(
                            "credentials.family",
                            format!(
                                "unknown family `{other}`; {hint}use one of {}",
                                FAMILIES.join(", ")
                            ),
                        ));
                    }
                }
            }
        };
        Ok(family)
    }
}

impl RawModel {
    fn validate(self, check: &Checker<'_>) -> Result<ModelSpec, ManifestError> {
        for (key, value) in [
            ("model.config_id", &self.config_id),
            ("model.effort_config_id", &self.effort_config_id),
            ("model.default", &self.default),
        ] {
            if let Some(value) = value {
                check.require(!value.trim().is_empty(), key, || "is empty".into())?;
            }
        }
        if self.config_id.is_some() {
            check.require(
                self.select == ModelSelect::ConfigOption,
                "model.config_id",
                || "only applies with `select = \"config-option\"`".into(),
            )?;
        }
        if self.format == ModelFormat::ModelEffort {
            check.require(self.select == ModelSelect::SetModel, "model.format", || {
                "`model[effort]` ids go through session/set_model; set `select = \"set-model\"`"
                    .into()
            })?;
        }
        for (i, prefix) in self.strip_prefixes.iter().enumerate() {
            check.require(
                !prefix.is_empty(),
                &format!("model.strip_prefixes[{i}]"),
                || "is empty".into(),
            )?;
        }
        Ok(ModelSpec {
            select: self.select,
            config_id: self.config_id.unwrap_or_else(|| "model".to_string()),
            effort_config_id: self.effort_config_id,
            format: self.format,
            default: self.default,
            strip_prefixes: self.strip_prefixes,
        })
    }
}

impl RawAgentProvider {
    fn validate(self, check: &Checker<'_>, name: &str) -> Result<AgentProviderSpec, ManifestError> {
        let at = |key: &str| format!("providers.{name}.{key}");
        // The key checker only lets `anthropic`, `openai` and `gateway`
        // through.
        let protocol = Protocol::parse(name).ok_or_else(|| {
            check.fail(
                &format!("providers.{name}"),
                "use `[providers.anthropic]`, `[providers.openai]` or `[providers.gateway]`",
            )
        })?;
        let api = match (protocol, self.api.as_deref()) {
            (Protocol::Anthropic | Protocol::Gateway, Some(_)) => {
                return Err(check.fail(&at("api"), "only `[providers.openai]` takes `api`"));
            }
            (Protocol::Anthropic | Protocol::Gateway, None) => None,
            (Protocol::OpenAi, None) => Some(OpenAiApi::Chat),
            (Protocol::OpenAi, Some(api)) => Some(OpenAiApi::parse(api).ok_or_else(|| {
                check.fail(&at("api"), format!("`{api}` is not `chat` or `responses`"))
            })?),
        };
        for (key, var) in [("base_url", &self.base_url), ("key", &self.key)] {
            if let Some(var) = var {
                check.require(is_env_name(var), &at(key), || {
                    format!("`{var}` is not an environment variable name")
                })?;
            }
        }
        for (i, var) in self.clear.iter().enumerate() {
            let key = format!("{}[{i}]", at("clear"));
            check.require(is_env_name(var), &key, || {
                format!("`{var}` is not an environment variable name")
            })?;
            check.require(
                Some(var) != self.base_url.as_ref()
                    && Some(var) != self.key.as_ref()
                    && !self.env.contains_key(var),
                &key,
                || format!("`{var}` cannot be both set and cleared"),
            )?;
        }
        let mut uses_effort = false;
        let mut uses_base_url = self.base_url.is_some();
        for (var, template) in &self.env {
            let key = format!("{}.{var}", at("env"));
            check.require(is_env_name(var), &key, || {
                format!("`{var}` is not an environment variable name")
            })?;
            let names = crate::providers::template_placeholders(template)
                .map_err(|bad| check.fail(&key, bad))?;
            uses_effort |= names.contains(&"effort");
            uses_base_url |= names.contains(&"base_url");
        }
        let model = self.model.unwrap_or_else(|| "{model}".to_string());
        let names = crate::providers::template_placeholders(&model)
            .map_err(|bad| check.fail(&at("model"), bad))?;
        uses_effort |= names.contains(&"effort");
        check.require(uses_base_url, &format!("providers.{name}"), || {
            "say where the endpoint goes: `base_url = \"<VARIABLE>\"`, or `{base_url}` in an `env` template"
                .into()
        })?;
        if let Some(effort) = &self.effort {
            check.require(crate::providers::is_effort(effort), &at("effort"), || {
                format!("`{effort}` is not an effort level (lowercase letters, digits, `-`, `_`)")
            })?;
        }
        check.require(!uses_effort || self.effort.is_some(), &at("effort"), || {
            "a template uses `{effort}`: say what it stands for when the run names no effort".into()
        })?;
        Ok(AgentProviderSpec {
            protocol,
            api,
            base_url: self.base_url,
            key: self.key,
            clear: self.clear,
            env: self.env,
            model,
            effort: self.effort,
        })
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_options;
