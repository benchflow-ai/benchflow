//! Who attempts the task in a trial: a model agent, the oracle, nothing, a
//! negative control, an exploit probe, or a user's own script.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use benchflow_sandbox::Sandbox;
use benchflow_task::package::Control;
use serde::{Deserialize, Serialize};

use crate::events::Events;
use crate::outcome::Problem;
use crate::probes::Probe;

/// Token usage, when the agent reports it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cost_usd: Option<f64>,
    /// Where the numbers came from (`acp`, `forwarder`, `unavailable`).
    pub source: String,
    /// Who priced `cost_usd`: [`COST_BY_PROVIDER`] or [`COST_BY_AGENT`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_source: Option<String>,
    /// Why no cost is recorded, when tokens were used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_note: Option<String>,
}

/// The cost is what the provider billed, as its answers said.
pub const COST_BY_PROVIDER: &str = "provider";
/// The cost is the agent's own figure (its run did not go through
/// BenchFlow's forwarder, which would have read the provider's).
pub const COST_BY_AGENT: &str = "agent";

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    /// Add a later stage's usage. The costs add up only when both are
    /// known (or the stage with none used no tokens): a total that left a
    /// stage out would claim less than was spent.
    pub fn add(&mut self, other: &Usage) {
        let (mine, theirs) = (self.total(), other.total());
        self.cost_usd = match (self.cost_usd, other.cost_usd) {
            (Some(x), Some(y)) => Some(x + y),
            (Some(x), None) if theirs == 0 => Some(x),
            (None, Some(y)) if mine == 0 => Some(y),
            _ => None,
        };
        self.cost_source = match (&self.cost_source, &other.cost_source) {
            _ if self.cost_usd.is_none() => None,
            (Some(a), Some(b)) if a != b => Some(COST_BY_AGENT.to_string()),
            (a, b) => a.clone().or_else(|| b.clone()),
        };
        self.cost_note = if self.cost_usd.is_some() {
            None
        } else {
            other
                .cost_note
                .clone()
                .or_else(|| self.cost_note.take())
                .or_else(|| {
                    (mine + theirs > 0)
                        .then(|| "the cost of part of the run is unknown".to_string())
                })
        };
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
        self.source = other.source.clone();
    }
}

/// What a model agent's run hands back.
#[derive(Debug, Clone, Default)]
pub struct AgentOutput {
    pub tool_calls: u64,
    pub prompts: Vec<String>,
    pub usage: Usage,
    /// The trajectory's summary block for result.json.
    pub trajectory_summary: serde_json::Value,
    pub timed_out: bool,
    /// The agent failed in a way that leaves nothing to grade fairly.
    pub problem: Option<Problem>,
    /// Environment variable names given to the agent (values are never
    /// recorded).
    pub env_keys: Vec<String>,
    /// The login used, as shown to people (`CLAUDE_CODE_OAUTH_TOKEN (environment)`).
    pub login: Option<String>,
    /// The model the agent's session said it was on, for a run that named
    /// none (so the record says what ran, not just what was asked).
    pub model: Option<String>,
    /// End the trial without grading (an interactive session the user
    /// chose not to grade).
    pub skip_grading: bool,
    /// The agent's own session id, when its harness reports one (ACP), so
    /// a later stage can resume it.
    pub session_id: Option<String>,
    /// The run resumed the session [`AgentContext::resume`] named.
    pub resumed: bool,
    /// Asked to resume and started a fresh session instead: why.
    pub resume_note: Option<String>,
}

/// Everything a model agent needs to run in a prepared sandbox.
pub struct AgentContext<'a> {
    pub sandbox: &'a dyn Sandbox,
    /// The user the agent runs as.
    pub user: &'a str,
    pub home: &'a str,
    pub workdir: &'a str,
    pub instruction: &'a str,
    /// Later prompts in the same session.
    pub stages: Vec<String>,
    pub timeout: Option<Duration>,
    pub idle_timeout: Duration,
    /// `--max-tool-calls`, for harnesses that can count tool calls.
    pub max_tool_calls: Option<u64>,
    /// `--max-tokens` (input plus output), for harnesses that can count them.
    pub max_tokens: Option<u64>,
    /// Task and run environment for the agent (no credentials).
    pub env: BTreeMap<String, String>,
    /// The trial folder; the agent writes `trajectory/` and `agent/` here.
    pub trial_dir: PathBuf,
    pub trial: &'a str,
    pub events: &'a Events,
    /// MCP servers the task declares.
    pub mcp: &'a [serde_json::Value],
    /// Resume the previous stage's session instead of starting a fresh one
    /// (`--resume-sessions`), for agents whose [`ModelAgent::session_dirs`]
    /// say they can.
    pub resume: Option<Resume>,
    /// A later run will resume this session (a branch parent at
    /// `prompt:N`): an agent that does not offer `session/load` is refused
    /// right after `initialize`, before any prompt.
    pub needs_load: bool,
    /// An agent already worked in this sandbox (an earlier stage), so what
    /// it did to its home is its own.
    pub earlier_turns: bool,
}

/// Values shorter than this are not redacted: masking a short value such
/// as `1` would garble every line containing it (the login's own rule).
const MIN_ENV_SECRET_LEN: usize = benchflow_agent::credentials::MIN_SECRET_LEN;

/// The values of the variables in `env` (the task's and `--env`'s) whose
/// names look secret, such as `DB_PASSWORD`, so they are redacted like the
/// login's own secrets even when no pattern recognizes them.
pub(crate) fn env_secrets(env: &BTreeMap<String, String>) -> Vec<String> {
    env.iter()
        .filter(|(k, v)| {
            benchflow_sandbox::redact::looks_secret(k) && v.len() >= MIN_ENV_SECRET_LEN
        })
        .map(|(_, v)| v.clone())
        .collect()
}

/// `login` with the secret-looking values of `env` added to what it
/// redacts.
pub(crate) fn with_env_secrets(
    mut login: benchflow_agent::credentials::Login,
    env: &BTreeMap<String, String>,
) -> benchflow_agent::credentials::Login {
    for value in env_secrets(env) {
        if !login.secrets.contains(&value) {
            login.secrets.push(value);
        }
    }
    login
}

/// The session a stage resumes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resume {
    /// The previous stage's session id, when its harness reported one.
    pub session_id: Option<String>,
    /// The session came from `--load-trajectory`: starting a fresh one
    /// instead would drop it, so the harness fails rather than fall back.
    pub loaded: bool,
}

/// A model agent: something installed in the sandbox and driven over a
/// protocol (ACP). Scripted agents do not use this.
#[async_trait]
pub trait ModelAgent: Send + Sync {
    /// The name shown in reports (`claude`).
    fn name(&self) -> &str;
    fn model(&self) -> Option<&str>;
    /// The reasoning effort asked for (`--effort`), recorded in config.json
    /// as BenchFlow 0.8 records it.
    fn effort(&self) -> Option<&str> {
        None
    }
    /// Root shell steps that install the agent, baked into an image layer.
    fn install_layers(&self) -> Vec<String>;
    /// Changes whenever the installed bits would change.
    fn fingerprint(&self) -> String;
    /// Check that a login exists before any sandbox starts.
    fn preflight(&self) -> Result<(), Problem>;
    /// The agent options the run set (`--ak`), by name, recorded in
    /// config.json.
    fn options(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::new()
    }
    /// Where under its home the agent keeps sessions, when it can resume
    /// one on a later stage on this harness; `None` when it cannot.
    fn session_dirs(&self) -> Option<Vec<String>> {
        None
    }
    /// The least the agent needs from its sandbox (its manifest's
    /// `[resources]`); the trial raises the task's size to it.
    fn resource_floor(&self) -> benchflow_agent::manifest::ResourceFloor {
        benchflow_agent::manifest::ResourceFloor::default()
    }
    /// Where the agent discovers skills (its manifest's `skill_paths`, with
    /// `$HOME/` and `$WORKSPACE/`); empty when it takes none.
    fn skill_paths(&self) -> Vec<String> {
        Vec::new()
    }
    /// The session file format `--load-trajectory` can place for the agent
    /// on this harness; `None` when it cannot load one.
    fn session_file(&self) -> Option<benchflow_agent::manifest::SessionFile> {
        None
    }
    /// Whether a later run can resume this agent's session over ACP
    /// (`session/load`), as a branch child at `prompt:N` does.
    fn resumes_over_acp(&self) -> bool {
        false
    }
    /// What the agent's login leaves in a sandbox whose agent user's home is
    /// `home`: the files it and the launch write, and the values that must not outlive
    /// the run (the login's secrets and the forwarder's key). A checkpoint
    /// deletes the files and refuses to keep any of the values.
    fn login_material(&self, home: &str) -> (Vec<String>, Vec<String>) {
        let _ = home;
        (Vec::new(), Vec::new())
    }
    /// Whether [`ModelAgent::login_material`] names everything the login
    /// left: only then can a checkpoint be taken after the agent ran.
    fn knows_login_material(&self) -> bool {
        false
    }
    /// Whether the agent needs the network to reach its model. A task with
    /// no network cannot run such an agent yet.
    fn needs_network(&self) -> bool {
        true
    }
    /// Run the agent on the instruction (and stages) in the sandbox.
    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput;
    /// The trial `trial` is graded (or failed) and recorded as `rec`; its
    /// sandbox is not torn down yet. For agents that report outcomes
    /// elsewhere, such as the external agent's backend.
    async fn graded(&self, _trial: &str, _rec: &crate::record::TrialRecord) {}
}

/// Who attempts the task.
#[derive(Clone)]
pub enum Attempt {
    /// The task's reference solution.
    Oracle,
    /// Nothing; the verifier scores the untouched workspace.
    Nop,
    /// A negative control script from the task's `controls/`.
    Control(Control),
    /// An exploit probe.
    Probe(&'static Probe),
    /// A model agent.
    Model(Arc<dyn ModelAgent>),
    /// A user's own script (`--agent script:<file>`), run as the oracle is:
    /// as the agent user, in the workdir. A task author runs a wrong answer
    /// this way to check that it fails.
    Script(UserScript),
}

/// A script from the user's machine that attempts the task.
#[derive(Debug, Clone)]
pub struct UserScript {
    /// The file's name, for labels (`wrong.sh`).
    pub name: String,
    /// Where it was read from.
    pub path: PathBuf,
    /// Its content, read once so every trial runs the same script.
    pub bytes: Arc<Vec<u8>>,
}

/// Read the regular file at `path` (a link to one is followed), refusing
/// it when it is larger than `max` bytes.
pub(crate) fn read_capped(path: &std::path::Path, max: u64) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    // Checked before opening: opening a FIFO would wait for a writer.
    if !std::fs::metadata(path)?.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a file",
        ));
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(max + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("larger than {} MB", max / (1024 * 1024)),
        ));
    }
    Ok(bytes)
}

/// The largest script `--agent script:` reads.
pub const MAX_SCRIPT_BYTES: u64 = 16 * 1024 * 1024;

impl UserScript {
    /// Read the script at `path`: a file of at most [`MAX_SCRIPT_BYTES`].
    pub fn read(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = read_capped(path, MAX_SCRIPT_BYTES)?;
        Ok(Self {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "script".into()),
            path: path.to_path_buf(),
            bytes: Arc::new(bytes),
        })
    }

    /// The SHA-256 of its content, hex.
    pub fn digest(&self) -> String {
        use sha2::Digest;
        hex::encode(sha2::Sha256::digest(self.bytes.as_slice()))
    }
}

impl Attempt {
    /// The label used in trial records and reports.
    pub fn label(&self) -> String {
        match self {
            Attempt::Oracle => "oracle".into(),
            Attempt::Nop => "nop".into(),
            Attempt::Control(c) => format!("control:{}", c.id),
            Attempt::Probe(p) => format!("probe:{}", p.id),
            Attempt::Model(m) => m.name().to_string(),
            Attempt::Script(s) => format!("script:{}", s.name),
        }
    }
    pub fn model(&self) -> Option<String> {
        match self {
            Attempt::Model(m) => m.model().map(str::to_string),
            _ => None,
        }
    }
    pub fn effort(&self) -> Option<String> {
        match self {
            Attempt::Model(m) => m.effort().map(str::to_string),
            _ => None,
        }
    }
    /// The agent options the run set; empty for scripted attempts.
    pub fn options(&self) -> serde_json::Map<String, serde_json::Value> {
        match self {
            Attempt::Model(m) => m.options(),
            _ => serde_json::Map::new(),
        }
    }
    /// Scripted attempts need no model and no login, and count as controls
    /// rather than agent runs in summaries.
    pub fn is_scripted(&self) -> bool {
        !matches!(self, Attempt::Model(_))
    }
}

impl std::fmt::Debug for Attempt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_script_must_be_a_file_of_at_most_16_mb() {
        let dir = tempfile::tempdir().unwrap();
        let ok = dir.path().join("wrong.sh");
        std::fs::write(&ok, "echo wrong\n").unwrap();
        assert_eq!(
            UserScript::read(&ok).unwrap().bytes.as_slice(),
            b"echo wrong\n"
        );
        let big = dir.path().join("big.sh");
        let f = std::fs::File::create(&big).unwrap();
        f.set_len(MAX_SCRIPT_BYTES + 1).unwrap();
        let e = UserScript::read(&big).unwrap_err();
        assert!(e.to_string().contains("larger than 16 MB"), "{e}");
        assert!(UserScript::read(dir.path()).is_err());
    }
    use benchflow_agent::credentials::Login;
    use benchflow_agent::trajectory::Redactor;

    #[test]
    fn env_values_with_secret_names_are_redacted() {
        let env = BTreeMap::from([
            ("DB_PASSWORD".to_string(), "plain-words-here".to_string()),
            ("MY_SERVICE_TOKEN".to_string(), "abc".to_string()),
            ("MODE".to_string(), "verbose-mode".to_string()),
        ]);
        let login = with_env_secrets(Login::none(), &env);
        let redactor = Redactor::new(login.secrets.iter().cloned());
        let line = redactor.redact("connect plain-words-here in verbose-mode abc");
        assert!(!line.contains("plain-words-here"), "{line}");
        assert!(
            line.contains("verbose-mode") && line.contains(" abc"),
            "{line}"
        );
    }
}
