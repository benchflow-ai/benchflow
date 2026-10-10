//! [`AcpClient`]: BenchFlow's side of an ACP conversation.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use benchflow_sandbox::{Process, ProcessHandle};
use chrono::Utc;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use super::conn::{self, Answer, Closed, EventSink, Exit, Outgoing, Reply, RpcError, Shared};
use super::error::AcpError;
use super::mcp::McpServer;
use super::model;
use super::watchdog::Watchdog;
use super::{limits, wire};
use crate::credentials::Login;
use crate::manifest::{AgentManifest, ModelFormat, ModelSelect, ModelSpec};
use crate::model_id::format_model;
use crate::trajectory::{Event, Receipt, Redactor, Trajectory, Usage};

/// The ACP protocol version this client speaks.
pub const PROTOCOL_VERSION: u64 = 1;
/// How long `initialize`, `session/new` and model selection may take (0.8's default).
pub const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(60);
/// How long a timed-out prompt gets to finish after `session/cancel` (0.8: 5 s
/// in all, minus the final drain).
const CANCEL_GRACE: Duration = Duration::from_millis(4750);
/// How long after the process exits stdout may stay open (a background child
/// holding it) before the prompt counts as ended.
const EXIT_GRACE: Duration = Duration::from_secs(2);
/// How long a failed write waits for the agent's exit to show.
const WRITE_EXIT_GRACE: Duration = Duration::from_secs(3);

/// The agent process's pipes. [`Process`] converts into it.
pub struct AgentPipes {
    pub stdin: Pin<Box<dyn AsyncWrite + Send>>,
    pub stdout: Pin<Box<dyn AsyncRead + Send>>,
    /// Kept for error messages and passed on as agent output.
    pub stderr: Option<Pin<Box<dyn AsyncRead + Send>>>,
    /// Gives exit codes, and kills the process at shutdown.
    pub handle: Option<Box<dyn ProcessHandle>>,
}

impl AgentPipes {
    /// Just stdin and stdout.
    pub fn new(
        stdin: Pin<Box<dyn AsyncWrite + Send>>,
        stdout: Pin<Box<dyn AsyncRead + Send>>,
    ) -> Self {
        Self {
            stdin,
            stdout,
            stderr: None,
            handle: None,
        }
    }
}

impl From<Process> for AgentPipes {
    fn from(process: Process) -> Self {
        Self {
            stdin: process.stdin,
            stdout: process.stdout,
            stderr: Some(process.stderr),
            handle: Some(process.handle),
        }
    }
}

impl fmt::Debug for AgentPipes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentPipes")
            .field("stderr", &self.stderr.is_some())
            .field("handle", &self.handle.is_some())
            .finish_non_exhaustive()
    }
}

/// How the client behaves.
#[derive(Clone)]
pub struct ClientOptions {
    /// The agent's name, for messages.
    pub agent: String,
    /// Ask for typed session failures (`[acp] typed_failures`).
    pub typed_failures: bool,
    /// Ask for the same-session subagent transcript (`[acp] subagent_transcript`).
    pub subagent_transcript: bool,
    /// How long `initialize`, `session/new` and model selection may take.
    pub handshake_timeout: Duration,
    /// Lines over this many bytes are rewritten to fit (0.8's 1 MiB); `None`
    /// keeps them whole.
    pub line_limit: Option<usize>,
    /// Lines over this many bytes are skipped (256 MiB).
    pub max_line_bytes: usize,
    /// The login's label, named in usage-limit errors.
    pub login: Option<String>,
    /// Redacts agent output events and stderr in errors.
    pub redactor: Redactor,
    /// Receives [`AcpEvent`](super::AcpEvent)s as they happen.
    pub on_event: Option<EventSink>,
}

impl fmt::Debug for ClientOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientOptions")
            .field("agent", &self.agent)
            .field("typed_failures", &self.typed_failures)
            .field("subagent_transcript", &self.subagent_transcript)
            .field("handshake_timeout", &self.handshake_timeout)
            .field("line_limit", &self.line_limit)
            .field("max_line_bytes", &self.max_line_bytes)
            .field("login", &self.login)
            .field("on_event", &self.on_event.is_some())
            .finish_non_exhaustive()
    }
}

impl ClientOptions {
    /// Defaults for an agent named `agent`.
    pub fn new(agent: impl Into<String>) -> Self {
        Self {
            agent: agent.into(),
            typed_failures: false,
            subagent_transcript: false,
            handshake_timeout: DEFAULT_HANDSHAKE_TIMEOUT,
            line_limit: Some(wire::DEFAULT_LINE_LIMIT),
            max_line_bytes: wire::DEFAULT_MAX_LINE,
            login: None,
            redactor: Redactor::default(),
            on_event: None,
        }
    }

    /// Options for `manifest`'s agent running on `login`: its ACP
    /// extensions, the login's label, and its secrets redacted.
    pub fn for_agent(manifest: &AgentManifest, login: Option<&Login>) -> Self {
        let mut options = Self::new(&manifest.name);
        options.typed_failures = manifest.acp.typed_failures;
        options.subagent_transcript = manifest.acp.subagent_transcript;
        if let Some(login) = login {
            options.login = Some(login.label.clone());
            options.redactor = Redactor::new(login.secrets.iter().cloned());
        }
        options
    }

    /// Send events to `sink`.
    pub fn on_event(mut self, sink: impl Fn(super::AcpEvent) + Send + Sync + 'static) -> Self {
        self.on_event = Some(Arc::new(sink));
        self
    }
}

/// Who the agent says it is (`initialize`).
#[derive(Debug, Clone, PartialEq)]
pub struct AgentInfo {
    pub protocol_version: u64,
    pub name: Option<String>,
    pub version: Option<String>,
    pub title: Option<String>,
    /// `agentCapabilities`, as sent.
    pub capabilities: Value,
    /// `authMethods`, as sent.
    pub auth_methods: Vec<Value>,
}

/// The session `session/new` opened.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionInfo {
    /// The agent's id, or `default` when it sent none.
    pub session_id: String,
    /// `models`: the current and available models, when the agent says.
    pub models: Option<Value>,
    /// `configOptions`: the options `session/set_config_option` takes.
    pub config_options: Vec<Value>,
    /// `modes`, as sent.
    pub modes: Option<Value>,
}

/// What [`AcpClient::select_model`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelChoice {
    /// The model id sent, if one was.
    pub model_id: Option<String>,
    /// How: `session/set_model` or `session/set_config_option:<id>`.
    pub model_via: Option<String>,
    /// How the effort was set: the model id carried it (Codex), or a config option.
    pub effort_via: Option<String>,
}

/// Why the agent stopped (`stopReason`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
    /// A reason this build does not know, as sent.
    Other(String),
}

impl StopReason {
    fn parse(s: &str) -> Self {
        match s {
            "end_turn" => Self::EndTurn,
            "max_tokens" => Self::MaxTokens,
            "max_turn_requests" => Self::MaxTurnRequests,
            "refusal" => Self::Refusal,
            "cancelled" => Self::Cancelled,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::EndTurn => "end_turn",
            Self::MaxTokens => "max_tokens",
            Self::MaxTurnRequests => "max_turn_requests",
            Self::Refusal => "refusal",
            Self::Cancelled => "cancelled",
            Self::Other(s) => s,
        }
    }
}

/// A finished prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptOutcome {
    pub stop_reason: StopReason,
    /// The session's cumulative usage, when the agent reports it.
    pub usage: Option<Usage>,
    /// The answer's `_meta`, as sent.
    pub meta: Option<Value>,
    pub duration: Duration,
}

/// Budgets for one prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptLimits {
    /// Wall-clock budget; `None` waits as long as the agent works.
    pub wall: Option<Duration>,
    /// Stop when no tool call, message or thought arrives for this long.
    pub idle: Option<Duration>,
    /// How long a silent pending tool call may defer the idle stop (default
    /// three idle budgets).
    pub pending_tool_grace: Option<Duration>,
}

/// A cloneable view of the session's trajectory, for writing it while a
/// prompt runs.
#[derive(Clone)]
pub struct TrajectoryHandle(Arc<Shared>);

impl TrajectoryHandle {
    /// A copy of every event so far (the text still streaming included).
    pub fn snapshot(&self) -> Vec<Event> {
        self.0.state().trajectory.snapshot()
    }

    /// Run `f` on the trajectory. Do not block in it: the connection waits.
    pub fn with<R>(&self, f: impl FnOnce(&Trajectory) -> R) -> R {
        f(&self.0.state().trajectory)
    }

    /// Counts for a progress line.
    pub fn progress(&self) -> super::Progress {
        self.0.progress(&self.0.state())
    }
}

impl fmt::Debug for TrajectoryHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrajectoryHandle").finish_non_exhaustive()
    }
}

/// An ACP client speaking to one agent process: JSON-RPC 2.0, one message
/// per line, over the process's stdin and stdout.
///
/// ```no_run
/// # async fn demo(process: benchflow_sandbox::Process, manifest: &benchflow_agent::AgentManifest,
/// #               login: &benchflow_agent::credentials::Login) -> Result<(), benchflow_agent::acp::AcpError> {
/// use benchflow_agent::acp::{AcpClient, ClientOptions, PromptLimits};
/// use std::time::Duration;
///
/// let mut client = AcpClient::start(process, ClientOptions::for_agent(manifest, Some(login)));
/// client.initialize().await?;
/// client.new_session("/app", &[]).await?;
/// client.select_model(&manifest.model, Some("claude-opus-5-5"), Some("high")).await?;
/// let limits = PromptLimits { wall: Some(Duration::from_secs(1800)), idle: Some(Duration::from_secs(600)), ..Default::default() };
/// let outcome = client.prompt("Fix the failing test.", &limits).await?;
/// let events = client.trajectory().snapshot();
/// client.shutdown().await;
/// # Ok(()) }
/// ```
pub struct AcpClient {
    shared: Arc<Shared>,
    outgoing: mpsc::UnboundedSender<Outgoing>,
    kill: Option<mpsc::Sender<oneshot::Sender<()>>>,
    /// An exit watcher runs (the pipes came with a process handle).
    watches_exit: bool,
    tasks: Vec<JoinHandle<()>>,
    next_id: i64,
    options: ClientOptions,
    agent_info: Option<AgentInfo>,
    session: Option<SessionInfo>,
}

impl fmt::Debug for AcpClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AcpClient")
            .field("agent", &self.options.agent)
            .field("agent_info", &self.agent_info)
            .field("session", &self.session.as_ref().map(|s| &s.session_id))
            .finish_non_exhaustive()
    }
}

impl Drop for AcpClient {
    fn drop(&mut self) {
        // Without shutdown(): close stdin, ask for a kill, stop reading.
        let _ = self.outgoing.send(Outgoing::Close);
        if let Some(kill) = &self.kill {
            let (done, _) = oneshot::channel();
            let _ = kill.try_send(done);
        }
        for task in &self.tasks {
            task.abort();
        }
    }
}

impl AcpClient {
    /// Start talking to an agent process. Spawns the connection's tasks, so
    /// call it inside a Tokio runtime.
    pub fn start(pipes: impl Into<AgentPipes>, options: ClientOptions) -> Self {
        let pipes = pipes.into();
        let shared = Arc::new(Shared::new(
            options.redactor.clone(),
            options.on_event.clone(),
        ));
        let (outgoing, queue) = mpsc::unbounded_channel();
        let mut tasks = vec![
            tokio::spawn(conn::writer(pipes.stdin, queue)),
            tokio::spawn(conn::reader(
                pipes.stdout,
                shared.clone(),
                outgoing.clone(),
                options.line_limit,
                options.max_line_bytes,
            )),
        ];
        match pipes.stderr {
            Some(stderr) => tasks.push(tokio::spawn(conn::stderr_reader(stderr, shared.clone()))),
            None => shared.stderr().done = true,
        }
        let kill = pipes.handle.map(|handle| {
            let (kill, kills) = mpsc::channel(1);
            // Not kept in `tasks`: it must outlive the client to see the kill through.
            tokio::spawn(conn::exit_watcher(handle, shared.clone(), kills));
            kill
        });
        Self {
            shared,
            outgoing,
            watches_exit: kill.is_some(),
            kill,
            tasks,
            next_id: 100_000,
            options,
            agent_info: None,
            session: None,
        }
    }

    /// The `initialize` handshake: protocol version 1, no file system or
    /// terminal offered (the agent has its own in the sandbox), session
    /// notices taken, and the typed-failure and subagent-transcript
    /// extensions when the options ask for them.
    pub async fn initialize(&mut self) -> Result<AgentInfo, AcpError> {
        let mut capabilities = json!({
            "fs": {"readTextFile": false, "writeTextFile": false},
            "terminal": false,
            "auth": {"terminal": false},
            // ACP session notices: what an adapter composes itself arrives
            // as a `notice`, not as text in the agent's reply.
            "session": {"notices": {}},
        });
        let mut meta = Map::new();
        if self.options.typed_failures {
            meta.insert(
                "jetbrains".into(),
                json!({"air": {"version": 1, "capabilities": ["sessionFailure"]}}),
            );
        }
        if self.options.subagent_transcript {
            meta.insert("subagent-transcript".into(), Value::Bool(true));
        }
        if !meta.is_empty() {
            capabilities["_meta"] = Value::Object(meta);
        }
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": capabilities,
            "clientInfo": {"name": "benchflow", "version": env!("CARGO_PKG_VERSION")},
        });
        let timeout = self.options.handshake_timeout;
        let result = self.request("initialize", params, Some(timeout)).await?;
        let Some(protocol_version) = result.get("protocolVersion").and_then(Value::as_u64) else {
            return Err(AcpError::Protocol {
                message: format!(
                    "the agent's initialize answer has no protocolVersion: {}",
                    self.snippet(&result)
                ),
            });
        };
        if protocol_version != PROTOCOL_VERSION {
            tracing::warn!(
                "ACP protocol version negotiated to {protocol_version} (this client speaks {PROTOCOL_VERSION})"
            );
        }
        let info = result.get("agentInfo");
        let text = |key: &str| {
            info.and_then(|i| i.get(key))
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        let agent_info = AgentInfo {
            protocol_version,
            name: text("name"),
            version: text("version"),
            title: text("title"),
            capabilities: result
                .get("agentCapabilities")
                .cloned()
                .unwrap_or(Value::Null),
            auth_methods: result
                .get("authMethods")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        };
        self.agent_info = Some(agent_info.clone());
        Ok(agent_info)
    }

    /// Open a session in `cwd` with the task's MCP servers. Updates the agent
    /// sends while the session opens are kept for it.
    pub async fn new_session(
        &mut self,
        cwd: &str,
        mcp_servers: &[McpServer],
    ) -> Result<SessionInfo, AcpError> {
        let params = json!({
            "cwd": cwd,
            "mcpServers": mcp_servers.iter().map(McpServer::to_acp).collect::<Vec<_>>(),
        });
        self.shared.state().opening = Some(Vec::new());
        // Switches the buffer off however this ends, a dropped future
        // included: a failed handshake's updates belong to no session.
        let _opening = OpeningGuard(self.shared.clone());
        let timeout = self.options.handshake_timeout;
        let result = self.request("session/new", params, Some(timeout)).await?;
        let session_id = result
            .get("sessionId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string);
        let session = SessionInfo {
            session_id: session_id.clone().unwrap_or_else(|| "default".to_string()),
            models: result.get("models").filter(|m| !m.is_null()).cloned(),
            config_options: result
                .get("configOptions")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            modes: result.get("modes").filter(|m| !m.is_null()).cloned(),
        };
        let events = {
            // One critical section: an update arriving now is either still
            // buffered (and replayed here) or sees the session id.
            let mut state = self.shared.state();
            let buffered = state.opening.take().unwrap_or_default();
            state.session_id = Some(session.session_id.clone());
            state.session_placeholder = session_id.is_none();
            buffered
                .into_iter()
                .flat_map(|(params, receipt)| {
                    conn::apply_update(&self.shared, &mut state, &params, receipt)
                })
                .collect::<Vec<_>>()
        };
        for event in events {
            self.shared.emit(event);
        }
        self.session = Some(session.clone());
        Ok(session)
    }

    /// True when the agent's `initialize` answer offers `session/load`.
    pub fn can_load_session(&self) -> bool {
        self.agent_info
            .as_ref()
            .and_then(|i| i.capabilities.get("loadSession"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Resume the session `session_id` (`session/load`) in `cwd` with the
    /// task's MCP servers. The agent replays the session's conversation
    /// before it answers; the replay is dropped, since the run that held
    /// the session recorded it.
    pub async fn load_session(
        &mut self,
        session_id: &str,
        cwd: &str,
        mcp_servers: &[McpServer],
    ) -> Result<SessionInfo, AcpError> {
        let params = json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": mcp_servers.iter().map(McpServer::to_acp).collect::<Vec<_>>(),
        });
        self.shared.state().opening = Some(Vec::new());
        let _opening = OpeningGuard(self.shared.clone());
        let timeout = self.options.handshake_timeout;
        let result = self.request("session/load", params, Some(timeout)).await?;
        let session = SessionInfo {
            session_id: session_id.to_string(),
            models: result.get("models").filter(|m| !m.is_null()).cloned(),
            config_options: result
                .get("configOptions")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            modes: result.get("modes").filter(|m| !m.is_null()).cloned(),
        };
        {
            let mut state = self.shared.state();
            let replayed = state.opening.take().map_or(0, |b| b.len());
            tracing::debug!("session/load replayed {replayed} updates; dropped");
            state.session_id = Some(session.session_id.clone());
            state.session_placeholder = false;
        }
        self.session = Some(session.clone());
        Ok(session)
    }

    /// Select the model and reasoning effort as the manifest says (0.8's
    /// capability-first dispatch): a config option or `session/set_model`,
    /// with Codex's `model[effort]` ids chosen from what the session offers.
    /// The effort goes in the agent's effort config option unless the model
    /// id already carries it. A value the session already has is not sent
    /// again: an agent started on a provider's model (named in its
    /// variables) lists it as current, and Codex refuses to select a model
    /// its catalog lacks even when it is the current one.
    pub async fn select_model(
        &mut self,
        spec: &ModelSpec,
        model: Option<&str>,
        effort: Option<&str>,
    ) -> Result<ModelChoice, AcpError> {
        let session = self.session()?.clone();
        let mut choice = ModelChoice::default();
        let mut effort_done = false;
        if let Some(model) = model.filter(|_| spec.select != ModelSelect::None) {
            let formatted = format_model(spec, model);
            let id = if spec.format == ModelFormat::ModelEffort {
                model::codex_model_id(
                    &self.options.agent,
                    &formatted,
                    session.models.as_ref(),
                    effort,
                )?
            } else {
                formatted
            };
            if let Some(effort) = effort
                && spec.format == ModelFormat::ModelEffort
                && model::codex_effort(&id) == effort
            {
                effort_done = true;
                choice.effort_via = Some(format!("model id {id}"));
            }
            let offered = model::config_option_ids(&session.config_options);
            let option = match spec.select {
                ModelSelect::ConfigOption => Some(spec.config_id.as_str()),
                ModelSelect::Auto if offered.contains(&"model") => Some("model"),
                _ => None,
            };
            match option {
                Some(option) if self.current_value(option) == Some(id.as_str()) => {
                    tracing::info!("model {id} is already the session's {option}");
                    choice.model_via = Some(format!("current value of {option}"));
                }
                Some(option) => {
                    self.set_config_option(option, &id, "model").await?;
                    choice.model_via = Some(format!("session/set_config_option:{option}"));
                }
                None if self.current_model_id() == Some(id.as_str()) => {
                    tracing::info!("model {id} is already the session's model");
                    choice.model_via = Some("current session model".to_string());
                }
                None => {
                    self.set_model(&id).await?;
                    choice.model_via = Some("session/set_model".to_string());
                }
            }
            choice.model_id = Some(id);
        } else if model.is_some() {
            tracing::info!(
                "not selecting a model over ACP for {}: its manifest leaves it to the agent",
                self.options.agent
            );
        }

        if let Some(effort) = effort.filter(|_| !effort_done) {
            let offered = self.config_option_ids();
            let option = spec.effort_config_id.clone().or_else(|| {
                offered
                    .iter()
                    .any(|id| id == "effort")
                    .then(|| "effort".to_string())
            });
            let Some(option) = option else {
                return Err(AcpError::Unsupported {
                    message: format!(
                        "reasoning effort `{effort}` was requested for agent `{}`, but it declares no effort \
                         option and its session offers none",
                        self.options.agent
                    ),
                });
            };
            if self.current_value(&option) == Some(effort) {
                tracing::info!("reasoning effort {effort} is already the session's {option}");
                choice.effort_via = Some(format!("current value of {option}"));
            } else {
                self.set_config_option(&option, effort, "reasoning effort")
                    .await?;
                choice.effort_via = Some(format!("session/set_config_option:{option}"));
            }
        }
        Ok(choice)
    }

    /// A config option's current value, as the session last reported it.
    fn current_value(&self, config_id: &str) -> Option<&str> {
        self.session
            .as_ref()?
            .config_options
            .iter()
            .find(|o| o.get("id").and_then(Value::as_str) == Some(config_id))?
            .get("currentValue")?
            .as_str()
    }

    /// The session's current model id (`models.currentModelId`).
    fn current_model_id(&self) -> Option<&str> {
        self.session
            .as_ref()?
            .models
            .as_ref()?
            .get("currentModelId")?
            .as_str()
    }

    fn config_option_ids(&self) -> Vec<String> {
        self.session
            .as_ref()
            .map(|s| {
                model::config_option_ids(&s.config_options)
                    .into_iter()
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn set_model(&mut self, model_id: &str) -> Result<(), AcpError> {
        let session_id = self.session()?.session_id.clone();
        let timeout = self.options.handshake_timeout;
        self.request(
            "session/set_model",
            json!({"sessionId": session_id, "modelId": model_id}),
            Some(timeout),
        )
        .await?;
        tracing::info!("model set to {model_id}");
        Ok(())
    }

    async fn set_config_option(
        &mut self,
        config_id: &str,
        value: &str,
        label: &str,
    ) -> Result<(), AcpError> {
        let session = self.session()?.clone();
        let offered = model::config_option_ids(&session.config_options);
        if !offered.is_empty() && !offered.contains(&config_id) {
            return Err(AcpError::Unsupported {
                message: format!(
                    "agent `{}` has no {label} config option `{config_id}`; its options are {}",
                    self.options.agent,
                    offered.join(", ")
                ),
            });
        }
        let timeout = self.options.handshake_timeout;
        let params =
            json!({"sessionId": session.session_id, "configId": config_id, "value": value});
        match self
            .request("session/set_config_option", params, Some(timeout))
            .await
        {
            Ok(result) => {
                if let Some(options) = result.get("configOptions").and_then(Value::as_array)
                    && let Some(session) = self.session.as_mut()
                {
                    session.config_options = options.clone();
                }
                tracing::info!("ACP {label} option {config_id} set to {value}");
                Ok(())
            }
            Err(error) => {
                // A model the agent refuses and does not list is refused on
                // every retry: name what it offers. Anything else stays the
                // retried error it was.
                let values = model::config_option_values(&session.config_options, config_id);
                if label == "model"
                    && !values.is_empty()
                    && !values.iter().any(|v| v == value)
                    && model::refuses_value(&error, value)
                {
                    return Err(AcpError::ModelNotOffered {
                        agent: self.options.agent.clone(),
                        model: value.to_string(),
                        offered: values,
                    });
                }
                Err(error)
            }
        }
    }

    /// Send one prompt and wait for the agent's turn to end, recording every
    /// update into the trajectory. The idle watchdog and the wall-clock
    /// budget stop it: the agent gets `session/cancel` and a few seconds to
    /// finish its turn. No write to the agent can hold it past its budgets,
    /// and a caller that drops the future leaves the client usable (the
    /// request is forgotten and the turn's text closed).
    pub async fn prompt(
        &mut self,
        text: &str,
        limits: &PromptLimits,
    ) -> Result<PromptOutcome, AcpError> {
        let session_id = self.session()?.session_id.clone();
        let started = Instant::now();
        let progress = {
            let mut state = self.shared.state();
            state.trajectory.record_user_prompt(text, Receipt::now());
            self.shared.progress(&state)
        };
        self.shared.emit(super::AcpEvent::Progress(progress));
        let id = self.take_id();
        let _turn = TurnGuard(self.shared.clone());
        let Some((_waiting, mut reply)) = self.register(id) else {
            return Err(self.exited_error().await);
        };
        let params = json!({"sessionId": session_id, "prompt": [{"type": "text", "text": text}]});
        let Ok(mut written) = self.queue(request_message(id, "session/prompt", params)) else {
            return Err(self.exited_error().await);
        };
        let mut writing = true;

        let mut watchdog = limits.idle.map(|idle| {
            Watchdog::start(
                &self.activity(),
                idle,
                limits.pending_tool_grace,
                started,
                Utc::now(),
            )
        });
        let poll = limits.idle.map_or(Duration::from_secs(1), |idle| {
            Watchdog::poll_interval(idle, limits.wall)
        });
        let deadline = limits.wall.map(|wall| started + wall);
        let mut exit = self.shared.exit.subscribe();
        // A process that had already exited counts as well as one that exits now.
        let mut exited_at = matches!(*exit.borrow_and_update(), Exit::Exited(_)).then(Instant::now);
        let mut watch_exit = exited_at.is_none();

        let reply = loop {
            let mut wake = Instant::now() + poll;
            if let Some(deadline) = deadline {
                wake = wake.min(deadline);
            }
            if let Some(at) = exited_at {
                wake = wake.min(at + EXIT_GRACE);
            }
            tokio::select! {
                // An answer that is there wins over a timer that fired with it.
                biased;
                reply = &mut reply => break reply.unwrap_or(Err(Closed)),
                result = &mut written, if writing => {
                    writing = false;
                    let result = result.unwrap_or_else(|_| Err(std::io::ErrorKind::BrokenPipe.into()));
                    if let Err(source) = result {
                        return Err(self.write_error(source).await);
                    }
                }
                _ = tokio::time::sleep_until(wake) => {}
                changed = exit.changed(), if watch_exit => {
                    if changed.is_err() {
                        watch_exit = false;
                    } else if matches!(*exit.borrow(), Exit::Exited(_)) {
                        exited_at = Some(Instant::now());
                        watch_exit = false;
                    }
                }
            }
            let now = Instant::now();
            if exited_at.is_some_and(|at| now >= at + EXIT_GRACE) {
                // The process is gone but something still holds its stdout.
                return Err(self.exited_error().await);
            }
            if let Some(dog) = watchdog.as_mut() {
                let activity = self.activity();
                dog.observe(&activity, now, Utc::now());
                if dog.idle_expired(now) {
                    let error = dog.timeout_error(&activity, now, Utc::now());
                    self.cancel_after_timeout(&session_id, &mut reply).await;
                    return Err(error);
                }
            }
            if let (Some(deadline), Some(wall)) = (deadline, limits.wall)
                && now >= deadline
            {
                let finished = self.cancel_after_timeout(&session_id, &mut reply).await;
                return Err(self.timed_out(wall, finished));
            }
        };

        match reply {
            Ok(Answer {
                result: Ok(result),
                at,
            }) => {
                // A turn that finished after its budget is a timeout (0.8's
                // `_completed_prompt_result`), judged by when the answer
                // arrived, not when this loop saw it. Only a finished turn
                // is converted; an error answer stays what it is.
                if let (Some(deadline), Some(wall)) = (deadline, limits.wall)
                    && at > deadline
                {
                    return Err(self.timed_out(wall, true));
                }
                self.prompt_outcome(&result, started)
            }
            Ok(Answer {
                result: Err(error), ..
            }) => Err(self.agent_error(error, "session/prompt")),
            Err(Closed) => Err(self.exited_error().await),
        }
    }

    fn prompt_outcome(&self, result: &Value, started: Instant) -> Result<PromptOutcome, AcpError> {
        let meta = result.get("_meta").filter(|m| !m.is_null());
        // A turn the agent ended on a failure never passes as finished:
        // verifying it would score the untouched workspace.
        if let Some(failure) = limits::session_failure(meta)
            && !limits::is_advisory(failure)
        {
            return Err(self.session_failure_error(failure));
        }
        let Some(stop_reason) = result.get("stopReason").and_then(Value::as_str) else {
            return Err(AcpError::Protocol {
                message: format!(
                    "the agent's session/prompt answer has no stopReason: {}",
                    self.snippet(result)
                ),
            });
        };
        let usage = result.get("usage").and_then(Usage::from_acp);
        let progress = {
            let mut state = self.shared.state();
            if let Some(usage) = usage {
                state.trajectory.record_usage(usage);
            }
            self.shared.progress(&state)
        };
        self.shared.emit(super::AcpEvent::Progress(progress));
        Ok(PromptOutcome {
            stop_reason: StopReason::parse(stop_reason),
            usage,
            meta: meta.cloned(),
            duration: started.elapsed(),
        })
    }

    /// Agent text for an error message: redacted, and at most 500 characters.
    fn snippet(&self, value: &Value) -> String {
        let text = self.shared.redactor.redact(&value.to_string()).into_owned();
        let count = text.chars().count();
        if count <= 500 {
            text
        } else {
            format!(
                "{}... ({count} characters)",
                text.chars().take(500).collect::<String>()
            )
        }
    }

    /// The error a failed turn's typed record describes (0.8's
    /// `session_failure_error`): a spent subscription is a usage limit;
    /// anything else reads as the `-32603` the agent sent before typed
    /// failures, so it is classified the same way.
    fn session_failure_error(&self, failure: &Map<String, Value>) -> AcpError {
        let redact = |key| {
            let text = limits::field(failure, key);
            self.shared.redactor.redact(text.trim()).into_owned()
        };
        let title = redact("title");
        let details = redact("details");
        if limits::is_spent_subscription(failure) {
            let detail = if title.is_empty() {
                "the account has no available quota".to_string()
            } else {
                title
            };
            return self.usage_limit(&detail);
        }
        let mut text = if title.is_empty() {
            "the agent ended its turn on a failure".to_string()
        } else {
            title
        };
        if !details.is_empty() && !text.contains(&details) {
            text = format!("{text}: {details}");
        }
        if failure.get("category").and_then(Value::as_str) == Some("connection") {
            text = format!(
                "{}: the agent's own process ended or lost its stream (not BenchFlow's connection to the \
                 sandbox); the agent log in the trial's agent/ folder says which",
                text.trim_end_matches('.')
            );
        }
        AcpError::Agent {
            code: -32603,
            message: format!("Internal error: {text}"),
            data: None,
            method: "session/prompt".to_string(),
        }
    }

    /// A usage-limit error from the agent's words (redacted).
    fn usage_limit(&self, text: &str) -> AcpError {
        let text = self.shared.redactor.redact(text);
        let info = limits::parse_limit_text(&text, Utc::now());
        AcpError::UsageLimit {
            reset_at: info.as_ref().and_then(|i| i.resets_at),
            window: info.as_ref().and_then(|i| i.window.clone()),
            message: info.map_or_else(|| text.trim().to_string(), |i| i.detail),
            login: self.options.login.clone(),
        }
    }

    /// A JSON-RPC error as an [`AcpError`] (0.8's `acp_error`): codex-acp's
    /// `usageLimitExceeded` and the CLIs' usage-limit words are usage limits.
    /// The agent's message and data are redacted.
    fn agent_error(&self, error: RpcError, method: &str) -> AcpError {
        let message = self.shared.redactor.redact(&error.message).into_owned();
        let data = error.data.map(|mut data| {
            self.shared.redactor.redact_json(&mut data);
            data
        });
        if let Some(Value::Object(fields)) = &data
            && fields.get("codexErrorInfo").and_then(Value::as_str) == Some("usageLimitExceeded")
        {
            let detail = fields
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or(&message);
            return self.usage_limit(detail);
        }
        if limits::is_usage_limit_text(&message) {
            return self.usage_limit(&message);
        }
        AcpError::Agent {
            code: error.code,
            message,
            data: data.map(Box::new),
            method: method.to_string(),
        }
    }

    /// Ask the agent to cancel the current prompt (`session/cancel`). The
    /// message is queued, not waited for: a stalled stdin cannot hold the
    /// caller.
    pub async fn cancel(&self) -> Result<(), AcpError> {
        let session_id = self.session()?.session_id.clone();
        match self.queue(cancel_message(&session_id)) {
            Ok(_) => Ok(()),
            Err(Closed) => Err(self.exited_error().await),
        }
    }

    /// After a timeout: `session/cancel`, then a few seconds for the agent
    /// to answer the prompt. True when it did. Bounded however the agent
    /// behaves (0.8 bounds the whole cleanup to five seconds).
    async fn cancel_after_timeout(
        &self,
        session_id: &str,
        reply: &mut oneshot::Receiver<Reply>,
    ) -> bool {
        if self.queue(cancel_message(session_id)).is_err() {
            tracing::warn!("could not ask the agent to cancel its prompt: its stdin is closed");
        }
        matches!(
            tokio::time::timeout(CANCEL_GRACE, reply).await,
            Ok(Ok(Ok(_)))
        )
    }

    /// The wall-clock timeout, with 0.8's `agent_timeout` marker. 0.8 left
    /// the marker out only when its own prompt task would not stop, which a
    /// dropped Rust future always does; whether the agent ended its turn is
    /// `cancelled`, and the marker says whether a tool call was still open.
    fn timed_out(&self, wall: Duration, cancelled: bool) -> AcpError {
        self.shared
            .state()
            .trajectory
            .record_timeout(wall, Receipt::now());
        AcpError::Timeout {
            after: wall,
            cancelled,
        }
    }

    /// The agent's process ended: its exit code (waiting briefly for it) and
    /// stderr's tail.
    async fn exited_error(&self) -> AcpError {
        let mut exit = self.shared.exit.subscribe();
        let code = if self.watches_exit {
            match tokio::time::timeout(
                Duration::from_secs(5),
                exit.wait_for(|e| *e != Exit::Running),
            )
            .await
            {
                Ok(Ok(state)) => match *state {
                    Exit::Exited(code) => code,
                    Exit::Running => None,
                },
                _ => None,
            }
        } else {
            None
        };
        // Give stderr a moment to deliver its last lines.
        for _ in 0..20 {
            if self.shared.stderr().done {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let tail = self.shared.stderr().last(2000);
        AcpError::Exited {
            code,
            stderr_tail: self.shared.redactor.redact(&tail).into_owned(),
        }
    }

    fn activity(&self) -> crate::trajectory::Activity {
        self.shared.state().trajectory.activity()
    }

    fn session(&self) -> Result<&SessionInfo, AcpError> {
        self.session.as_ref().ok_or_else(|| AcpError::Protocol {
            message: "no ACP session yet: call new_session() first".to_string(),
        })
    }

    fn take_id(&mut self) -> i64 {
        self.next_id += 1;
        self.next_id
    }

    /// Register a request so the reader can route its answer; the guard
    /// forgets it however the caller stops waiting. `None` when stdout
    /// already ended: nothing would ever answer it.
    fn register(&self, id: i64) -> Option<(Waiting, oneshot::Receiver<Reply>)> {
        let mut state = self.shared.state();
        if state.closed {
            return None;
        }
        let (reply_tx, reply) = oneshot::channel();
        state.pending.insert(id, reply_tx);
        let waiting = Waiting {
            shared: self.shared.clone(),
            id,
        };
        Some((waiting, reply))
    }

    /// Queue a message for the writer; the receiver says when it is written.
    /// Never waits, so a stalled stdin cannot hold the caller.
    fn queue(&self, message: Value) -> Result<oneshot::Receiver<std::io::Result<()>>, Closed> {
        let (done, written) = oneshot::channel();
        self.outgoing
            .send(Outgoing::Line {
                line: message.to_string(),
                done: Some(done),
            })
            .map_err(|_| Closed)?;
        Ok(written)
    }

    /// A failed write: the agent's exit when it is gone, else the I/O error.
    /// A write to an agent that has just exited often fails a moment before
    /// its stdout ends or its exit is seen (Daytona answers with a missing
    /// input pipe), so the exit gets [`WRITE_EXIT_GRACE`] to show: an agent
    /// that quit is reported as having quit, with its last words, not as a
    /// broken connection.
    async fn write_error(&self, source: std::io::Error) -> AcpError {
        let gone = |client: &Self| {
            client.shared.state().closed || *client.shared.exit.borrow() != Exit::Running
        };
        if !gone(self) && self.watches_exit {
            let mut exit = self.shared.exit.subscribe();
            let _ = tokio::time::timeout(WRITE_EXIT_GRACE, exit.wait_for(|e| *e != Exit::Running))
                .await;
        }
        if gone(self) {
            self.exited_error().await
        } else {
            AcpError::Io {
                context: "writing to the agent's stdin".to_string(),
                source,
            }
        }
    }

    /// Send a request and wait for its answer. The timeout covers the write
    /// too: an agent whose stdin stays full is as silent as one that never
    /// answers.
    async fn request(
        &mut self,
        method: &'static str,
        params: Value,
        timeout: Option<Duration>,
    ) -> Result<Value, AcpError> {
        let id = self.take_id();
        let Some((_waiting, reply)) = self.register(id) else {
            return Err(self.exited_error().await);
        };
        let Ok(written) = self.queue(request_message(id, method, params)) else {
            return Err(self.exited_error().await);
        };
        let exchange = async move {
            match written.await {
                Ok(Ok(())) => {}
                Ok(Err(source)) => return Err(source),
                Err(_) => return Err(std::io::ErrorKind::BrokenPipe.into()),
            }
            Ok(reply.await.unwrap_or(Err(Closed)))
        };
        let outcome = match timeout {
            Some(after) => match tokio::time::timeout(after, exchange).await {
                Ok(outcome) => outcome,
                Err(_) => {
                    return Err(AcpError::HandshakeTimeout {
                        phase: method,
                        after,
                    });
                }
            },
            None => exchange.await,
        };
        match outcome {
            Err(source) => Err(self.write_error(source).await),
            Ok(Ok(answer)) => answer
                .result
                .map_err(|error| self.agent_error(error, method)),
            Ok(Err(Closed)) => Err(self.exited_error().await),
        }
    }

    /// The trajectory, shareable with a task that writes it while a prompt runs.
    pub fn trajectory(&self) -> TrajectoryHandle {
        TrajectoryHandle(self.shared.clone())
    }

    /// What `initialize` returned.
    pub fn agent_info(&self) -> Option<&AgentInfo> {
        self.agent_info.as_ref()
    }

    /// What `session/new` returned (config options updated since).
    pub fn session_info(&self) -> Option<&SessionInfo> {
        self.session.as_ref()
    }

    /// Lines the agent wrote on stdout that were not ACP.
    pub fn noise_lines(&self) -> u64 {
        self.shared.state().noise_lines
    }

    /// The last of what the agent wrote on stderr, redacted.
    pub fn stderr_tail(&self, chars: usize) -> String {
        let tail = self.shared.stderr().last(chars);
        self.shared.redactor.redact(&tail).into_owned()
    }

    /// End the conversation: close stdin, give the process five seconds to
    /// exit, then kill it. Returns its exit code when known.
    pub async fn shutdown(mut self) -> Option<i32> {
        let _ = self.outgoing.send(Outgoing::Close);
        let mut exit = self.shared.exit.subscribe();
        let mut code = None;
        if let Some(kill) = self.kill.take() {
            let exited = tokio::time::timeout(
                Duration::from_secs(5),
                exit.wait_for(|e| *e != Exit::Running),
            )
            .await
            .is_ok_and(|r| r.is_ok());
            if !exited {
                // A backend whose kill hangs must not hang shutdown.
                let _ = tokio::time::timeout(Duration::from_secs(5), async {
                    let (done, killed) = oneshot::channel();
                    if kill.send(done).await.is_ok() {
                        let _ = killed.await;
                    }
                })
                .await;
                let _ = tokio::time::timeout(
                    Duration::from_secs(5),
                    exit.wait_for(|e| *e != Exit::Running),
                )
                .await;
            }
            if let Exit::Exited(exit_code) = *exit.borrow() {
                code = exit_code;
            }
        }
        for task in self.tasks.drain(..) {
            task.abort();
        }
        code
    }
}

#[cfg(test)]
impl AcpClient {
    /// Requests waiting for an answer.
    pub(crate) fn pending_requests(&self) -> usize {
        self.shared.state().pending.len()
    }

    /// `session/new`'s update buffer is on.
    pub(crate) fn opening_buffer_on(&self) -> bool {
        self.shared.state().opening.is_some()
    }
}

fn request_message(id: i64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

fn cancel_message(session_id: &str) -> Value {
    json!({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": session_id}})
}

/// A registered request. Dropping it forgets the request, also when the
/// caller drops its future: a late answer then goes nowhere.
struct Waiting {
    shared: Arc<Shared>,
    id: i64,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        self.shared.state().pending.remove(&self.id);
    }
}

/// Closes a turn's streaming text however the prompt ends, also when the
/// caller drops its future.
struct TurnGuard(Arc<Shared>);

impl Drop for TurnGuard {
    fn drop(&mut self) {
        self.0.state().trajectory.finish_prompt();
    }
}

/// Switches off `session/new`'s update buffer however the request ends,
/// also when the caller drops its future (0.8 cleared it in `finally`).
struct OpeningGuard(Arc<Shared>);

impl Drop for OpeningGuard {
    fn drop(&mut self) {
        self.0.state().opening = None;
    }
}
