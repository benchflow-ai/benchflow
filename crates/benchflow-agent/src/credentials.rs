//! Credentials: the login an agent runs on.
//!
//! [`find_login`] looks for the credentials an agent's family accepts, in the
//! order the agent's own CLI prefers them (as BenchFlow 0.8 did), and turns
//! the first one it finds into a [`Login`]: the variables and files the
//! sandbox needs, a label for the UI, and the secret values to redact from
//! every log and trajectory.
//!
//! | Family | Looked for, in order |
//! |---|---|
//! | `anthropic` (Claude Code) | `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` (alias `CLAUDE_OAUTH_TOKEN`), `ANTHROPIC_AUTH_TOKEN`, `~/.claude/.credentials.json` |
//! | `openai` (Codex) | `OPENAI_API_KEY`, `CODEX_API_KEY`, `CODEX_ACCESS_TOKEN`, `CODEX_AUTH_JSON` (an inline `auth.json`), `~/.codex/auth.json` |
//! | `gemini` (Gemini CLI) | `GEMINI_API_KEY`, `GOOGLE_API_KEY` (each mirrored to the other), `~/.gemini/oauth_creds.json` with `settings.json` and `google_accounts.json` |
//! | `model` (OpenCode) | the key of the provider in the model's `provider/model` id: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`/`GEMINI_API_KEY`/`GOOGLE_API_KEY`, else `<PROVIDER>_API_KEY` |
//! | `env` | the manifest's variables, in its order |
//!
//! Only the chosen login reaches the sandbox, so the label always says what
//! the agent actually runs on. A variable set to an empty or blank value
//! counts as unset. The environment is passed in as a [`HostEnv`], so tests
//! (and callers that load a `.env` file themselves) never touch the process
//! environment. Base URLs (`ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`) are not
//! credentials and are not carried by a login.
//!
//! [`login_status`] reports every credential found, for `bench agents` and
//! `bench doctor`; it never holds a value.
//!
//! A plain API key (the `ApiKey` kinds, and an API key stored in Codex's
//! `auth.json`) can be held by BenchFlow's forwarder instead of reaching the
//! agent (`crate::providers::resolve`); subscription logins always reach the
//! agent.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::manifest::{AgentManifest, CredentialFamily};

mod families;
mod files;

/// The variable that names the login in the UI instead of the credential's
/// variable (a pooled account's name, say). Never passed to the sandbox.
pub const LOGIN_LABEL_ENV: &str = "BENCHFLOW_LOGIN_LABEL";

/// Shortest value treated as a secret. Shorter values (test placeholders,
/// typos) would redact ordinary text everywhere they appear.
pub const MIN_SECRET_LEN: usize = 8;

/// The machine's credentials as BenchFlow sees them: environment variables,
/// variables the run sets explicitly, and the home folder that holds login
/// files.
#[derive(Clone)]
pub struct HostEnv {
    vars: BTreeMap<String, String>,
    overrides: BTreeMap<String, String>,
    home: Option<PathBuf>,
    now: DateTime<Utc>,
}

impl fmt::Debug for HostEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostEnv")
            .field("vars", &self.vars.keys().collect::<Vec<_>>())
            .field("overrides", &self.overrides.keys().collect::<Vec<_>>())
            .field("home", &self.home)
            .field("now", &self.now)
            .finish()
    }
}

/// Where a credential was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// The process environment (or the map a caller passed as such).
    Environment,
    /// A variable the run set explicitly, which wins over the environment.
    AgentEnv,
    /// A login file under the home folder.
    File,
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Environment => "environment",
            Self::AgentEnv => "agent_env",
            Self::File => "file",
        })
    }
}

impl HostEnv {
    /// This process's environment and the user's home folder.
    pub fn from_process() -> Self {
        let vars = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
            .collect();
        Self::new(vars, dirs::home_dir())
    }

    /// An environment from a map, with login files under `home`.
    pub fn new(vars: BTreeMap<String, String>, home: Option<PathBuf>) -> Self {
        Self {
            vars,
            overrides: BTreeMap::new(),
            home,
            now: Utc::now(),
        }
    }

    /// Variables the run sets explicitly (labelled `agent_env`); they win
    /// over the environment.
    /// Variables that count as this process's environment (a caller's own,
    /// such as a test pointing the judge at a fake server), unlike
    /// [`HostEnv::with_overrides`], which are the run's.
    pub fn with_environment(mut self, vars: BTreeMap<String, String>) -> Self {
        self.vars.extend(vars);
        self
    }

    pub fn with_overrides(mut self, overrides: BTreeMap<String, String>) -> Self {
        self.overrides = overrides;
        self
    }

    /// The time expiry checks compare against (default: now).
    pub fn at(mut self, now: DateTime<Utc>) -> Self {
        self.now = now;
        self
    }

    /// A variable's value and where it came from. Blank values count as
    /// unset, and an override set blank hides the environment's value (so a
    /// run can drop an API key to use the subscription login, as in 0.8).
    /// `name` from the process environment only, never from the run's own
    /// variables (`--env`, a job file, bench.toml's `[run] env`).
    pub fn environment(&self, name: &str) -> Option<&str> {
        self.vars
            .get(name)
            .map(String::as_str)
            .filter(|v| !v.trim().is_empty())
    }

    pub fn get(&self, name: &str) -> Option<(&str, Origin)> {
        let set = |v: &&String| !v.trim().is_empty();
        if let Some(value) = self.overrides.get(name) {
            return Some(value)
                .filter(set)
                .map(|v| (v.as_str(), Origin::AgentEnv));
        }
        self.vars
            .get(name)
            .filter(set)
            .map(|v| (v.as_str(), Origin::Environment))
    }

    /// The variables the run sets explicitly, blank ones left out.
    pub(crate) fn overrides(&self) -> impl Iterator<Item = (&str, &str)> {
        self.overrides
            .iter()
            .filter(|(_, v)| !v.trim().is_empty())
            .map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub(crate) fn home(&self) -> Option<&std::path::Path> {
        self.home.as_deref()
    }

    pub(crate) fn now(&self) -> DateTime<Utc> {
        self.now
    }
}

/// A file the sandbox needs before the agent starts.
#[derive(Clone, PartialEq, Eq)]
pub struct SandboxFile {
    /// Absolute path in the sandbox; `$HOME` stands for the agent user's home.
    pub path: String,
    pub contents: Vec<u8>,
    /// Unix permissions (`0o600` for credentials).
    pub mode: u32,
}

impl SandboxFile {
    /// The path with `$HOME` replaced by the agent user's home (`/home/agent`).
    pub fn path_for(&self, home: &str) -> String {
        match self.path.strip_prefix("$HOME") {
            Some(rest) => format!("{}{rest}", home.trim_end_matches('/')),
            None => self.path.clone(),
        }
    }
}

impl fmt::Debug for SandboxFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxFile")
            .field("path", &self.path)
            .field("contents", &format_args!("<{} bytes>", self.contents.len()))
            .field("mode", &format_args!("{:o}", self.mode))
            .finish()
    }
}

/// The login an agent runs on.
#[derive(Clone, PartialEq, Eq)]
pub struct Login {
    /// What the UI shows: `CLAUDE_CODE_OAUTH_TOKEN (environment)`, `host
    /// login (~/.claude/.credentials.json)`. Never a value.
    pub label: String,
    /// Variables for the agent process.
    pub env: BTreeMap<String, String>,
    /// Files to write before the agent starts.
    pub files: Vec<SandboxFile>,
    /// Values to redact from every log and trajectory.
    pub secrets: Vec<String>,
    /// Warnings about this login (an expired access token).
    pub notes: Vec<String>,
}

impl fmt::Debug for Login {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Login")
            .field("label", &self.label)
            .field("env", &self.env.keys().collect::<Vec<_>>())
            .field("files", &self.files)
            .field("secrets", &format_args!("<{} values>", self.secrets.len()))
            .field("notes", &self.notes)
            .finish()
    }
}

impl Login {
    /// The login of an agent that needs none from BenchFlow.
    pub fn none() -> Self {
        Self {
            label: "no login (the agent needs none from BenchFlow)".to_string(),
            env: BTreeMap::new(),
            files: Vec::new(),
            secrets: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub(crate) fn add_secret(&mut self, value: &str) {
        if value.len() >= MIN_SECRET_LEN && !self.secrets.iter().any(|s| s == value) {
            self.secrets.push(value.to_string());
        }
    }
}

/// A plain API key in the chosen login, for the forwarder: the variable it
/// is read under and where it was found. `Debug` never shows the value.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct PlainKey {
    /// The variable a provider reads it under (`OPENAI_API_KEY`).
    pub var: String,
    /// Where it was found, for people: `OPENAI_API_KEY (environment)`,
    /// `OPENAI_API_KEY in ~/.codex/auth.json`.
    pub source: String,
    pub value: String,
}

impl fmt::Debug for PlainKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlainKey")
            .field("var", &self.var)
            .field("source", &self.source)
            .field("value", &format_args!("<{} chars>", self.value.len()))
            .finish()
    }
}

/// The login [`find_login`] picks, and the plain API key it holds, if it
/// is one.
#[derive(Debug, Clone)]
pub(crate) struct Chosen {
    pub login: Login,
    pub api_key: Option<PlainKey>,
}

/// What kind of credential a source holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    ApiKey,
    AuthToken,
    OauthToken,
    AccessToken,
    LoginFile,
    AuthJson,
}

/// One credential found on this machine. Never holds the value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LoginSource {
    /// A variable name, or a `~/`-relative file path.
    pub name: String,
    pub kind: SourceKind,
    pub origin: Origin,
    /// False when the agent cannot run on it as it is (an unreadable file,
    /// an expired token it cannot refresh).
    pub usable: bool,
    pub note: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
}

impl fmt::Display for LoginSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.origin {
            Origin::File => write!(f, "{}", self.name)?,
            origin => write!(f, "{} ({origin})", self.name)?,
        }
        if let Some(note) = &self.note {
            write!(f, ": {note}")?;
        }
        Ok(())
    }
}

/// Every credential an agent could use here, for `bench agents` and
/// `bench doctor`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LoginStatus {
    pub agent: String,
    /// Every credential found, in the order the agent prefers them.
    pub sources: Vec<LoginSource>,
    /// The one [`find_login`] picks, as an index into `sources`.
    pub chosen: Option<usize>,
    /// Facts worth knowing (an API key overriding a subscription login).
    pub notes: Vec<String>,
    /// What to do when there is no usable login.
    pub fix: Option<String>,
}

impl LoginStatus {
    /// True when the chosen credential is usable as it is.
    pub fn ready(&self) -> bool {
        self.chosen.is_some_and(|i| self.sources[i].usable)
    }

    /// One line without values: the chosen login, the others, and notes.
    pub fn summary(&self) -> String {
        let Some(chosen) = self.chosen else {
            return "no login found".to_string();
        };
        let mut parts = vec![self.sources[chosen].to_string()];
        for (i, source) in self.sources.iter().enumerate() {
            if i != chosen && !source.usable {
                parts.push(format!("ignored {source}"));
            }
        }
        let also: Vec<&str> = self
            .sources
            .iter()
            .enumerate()
            .filter(|(i, s)| *i != chosen && s.usable)
            .map(|(_, s)| s.name.as_str())
            .collect();
        if !also.is_empty() {
            parts.push(format!("also found {}", also.join(", ")));
        }
        parts.extend(self.notes.iter().cloned());
        parts.join("; ")
    }
}

/// Why an agent has no login to run on. The fix is the user's: every
/// variant carries the next step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialError {
    /// Nothing the agent accepts was found.
    Missing {
        agent: String,
        /// What was looked for, in order.
        checked: Vec<String>,
        next_step: String,
    },
    /// A login file is there but cannot be used.
    BadFile {
        agent: String,
        /// `~/`-relative path, or the variable that held the file.
        path: String,
        problem: String,
        next_step: String,
    },
    /// The model does not say which provider's key the agent needs.
    UnknownProvider {
        agent: String,
        model: Option<String>,
        next_step: String,
    },
    /// The model names a provider and its key is nowhere to be found.
    NoProviderKey {
        provider: String,
        /// The variables and files looked at, in order.
        checked: Vec<String>,
        next_step: String,
    },
}

impl CredentialError {
    /// The next step for the user.
    pub fn next_step(&self) -> &str {
        match self {
            Self::Missing { next_step, .. }
            | Self::BadFile { next_step, .. }
            | Self::UnknownProvider { next_step, .. }
            | Self::NoProviderKey { next_step, .. } => next_step,
        }
    }
}

impl fmt::Display for CredentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { agent, checked, .. } => {
                write!(
                    f,
                    "no login for agent `{agent}`: looked for {}",
                    join_or(checked)
                )?;
            }
            Self::BadFile {
                agent,
                path,
                problem,
                ..
            } => write!(
                f,
                "cannot use {path} as a login for agent `{agent}`: {problem}"
            )?,
            Self::UnknownProvider { agent, model, .. } => match model {
                Some(model) => write!(
                    f,
                    "agent `{agent}` reads the key of the model's provider, and `{model}` names none"
                )?,
                None => write!(
                    f,
                    "agent `{agent}` reads the key of the model's provider, and no model was given"
                )?,
            },
            Self::NoProviderKey {
                provider, checked, ..
            } => write!(
                f,
                "no key for provider `{provider}`: looked for {}",
                join_or(checked)
            )?,
        }
        write!(f, "; next: {}", self.next_step())
    }
}

impl std::error::Error for CredentialError {}

/// `a, b, c or d`.
fn join_or(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

/// The login `manifest`'s agent would run on here, in the order its CLI
/// prefers credentials. `model` matters only for the `model` family.
pub fn find_login(
    manifest: &AgentManifest,
    model: Option<&str>,
    host: &HostEnv,
) -> Result<Login, CredentialError> {
    choose_login(manifest, model, host).map(|chosen| chosen.login)
}

/// [`find_login`], with what the login is.
pub(crate) fn choose_login(
    manifest: &AgentManifest,
    model: Option<&str>,
    host: &HostEnv,
) -> Result<Chosen, CredentialError> {
    if manifest.credentials == CredentialFamily::None {
        return Ok(Chosen {
            login: Login::none(),
            api_key: None,
        });
    }
    let scan = families::scan(manifest, model, host)?;
    let mut first_error = None;
    for candidate in scan.candidates {
        match candidate.login {
            Ok(mut login) => {
                if let Some((label, _)) = host.get(LOGIN_LABEL_ENV) {
                    login.label = custom_label(label);
                }
                return Ok(Chosen {
                    login,
                    api_key: candidate.api_key,
                });
            }
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    Err(first_error.unwrap_or(CredentialError::Missing {
        agent: manifest.name.clone(),
        checked: scan.checked,
        next_step: scan.next_step,
    }))
}

/// Every credential `manifest`'s agent reads, by name (a variable, or a
/// `~/`-relative file) and kind, in the order it prefers them, whether or not
/// it is here. Empty for the `model` family, whose key depends on the model.
pub fn credential_kinds(manifest: &AgentManifest) -> Vec<(String, SourceKind)> {
    families::kinds(manifest)
}

/// Every credential `manifest`'s agent could use here, without values.
pub fn login_status(manifest: &AgentManifest, model: Option<&str>, host: &HostEnv) -> LoginStatus {
    let agent = manifest.name.clone();
    if manifest.credentials == CredentialFamily::None {
        return LoginStatus {
            agent,
            sources: Vec::new(),
            chosen: None,
            notes: vec!["needs no login from BenchFlow".to_string()],
            fix: None,
        };
    }
    match families::scan(manifest, model, host) {
        Ok(scan) => {
            let chosen = scan.candidates.iter().position(|c| c.login.is_ok());
            let usable = chosen.is_some_and(|i| scan.candidates[i].source.usable);
            LoginStatus {
                agent,
                chosen,
                notes: scan.notes,
                fix: (!usable).then_some(scan.next_step),
                sources: scan.candidates.into_iter().map(|c| c.source).collect(),
            }
        }
        Err(e) => LoginStatus {
            agent,
            sources: Vec::new(),
            chosen: None,
            notes: Vec::new(),
            fix: Some(e.next_step().to_string()),
        },
    }
}

/// `BENCHFLOW_LOGIN_LABEL`'s value as a label: one line, at most 80 chars.
fn custom_label(raw: &str) -> String {
    raw.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(80)
        .collect()
}

#[cfg(test)]
mod tests;
