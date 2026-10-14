//! Model agents driven over the Agent Client Protocol.
//!
//! [`AcpAgent`] joins an agent manifest from `benchflow-agent` to a trial.
//! The manifest's install script becomes a cached image layer. The login
//! found on this machine is written into the sandbox for the agent user, and
//! its secrets are redacted from every log and trajectory. The agent process
//! starts as the agent user in the task's working folder, and
//! [`AcpClient`] drives it: `initialize`, a session with the task's MCP
//! servers, the model and effort, then the instruction and each later stage
//! as prompts under one wall-clock budget and the idle watchdog.
//!
//! The trajectory is rewritten to `trajectory/acp_trajectory.jsonl` while the
//! agent works, so a crash or a Ctrl-C keeps what happened. Everything the
//! agent writes that is not ACP goes to `agent/<agent>.txt` (stderr also to
//! `agent/stderr.txt`), as BenchFlow 0.8 does.
//!
//! A model named `<provider>/<model>` runs on that provider, and a plain API
//! key the agent found goes the same way when the agent can take a base URL
//! for it ([`crate::forwarding`]): before the agent starts, BenchFlow's
//! forwarder starts as root in the sandbox holding the key, and the agent
//! gets the forwarder's address and a placeholder key. Its log (one line per
//! request, no key) is kept as `agent/forwarder.log`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use benchflow_agent::AgentManifest;
use benchflow_agent::acp::{
    AcpClient, AcpError, AcpEvent, ClientOptions, McpServer, OutputStream, Owner as AcpOwner,
    PromptLimits, StopReason, broke_mid_stream, ends_mid_stream, provider_failure_in_reply,
    refused_request,
};
use benchflow_agent::credentials::{HostEnv, Login};
use benchflow_agent::manifest::InstallMethod;
use benchflow_agent::providers::{self, ProviderRegistry};
use benchflow_agent::trajectory::{
    Redactor, TrajectoryWriter, Usage as AcpUsage, count_tool_calls, summarize,
};
use benchflow_sandbox::Exec;

use crate::agent::{AgentContext, AgentOutput, ModelAgent, Usage};
use crate::events::Event;
use crate::forwarding::{self, ProviderAccess};
use crate::outcome::{Owner, Phase, Problem};
use crate::trial::sandbox_problem;

/// How often the trajectory file is rewritten while the agent works.
pub(crate) const TRAJECTORY_EVERY: Duration = Duration::from_secs(2);
/// Characters of stderr kept as a failure's evidence.
pub(crate) const STDERR_TAIL: usize = 4000;
/// How much of the agent's stderr is read for the cause of an error that
/// names none (OpenHands' panels push the cause far up).
const STDERR_READ: usize = 64 * 1024;

/// An agent from a manifest, on the login found for it.
pub struct AcpAgent {
    manifest: AgentManifest,
    model: Option<String>,
    effort: Option<String>,
    /// Found when the agent is built; a missing login is reported by
    /// [`ModelAgent::preflight`] before any sandbox starts.
    login: Result<Login, Problem>,
    /// Set when the run goes through the forwarder (a model that names a
    /// provider, or the agent's own API key): the endpoint, and the key the
    /// forwarder holds.
    provider: Option<ProviderAccess>,
}

impl AcpAgent {
    /// The agent `manifest` with `model` (else the manifest's default) and
    /// `effort`, on the first usable login in `host` (its plain API key held
    /// by the forwarder when it can be), or on the built-in provider the
    /// model names.
    pub fn new(
        manifest: AgentManifest,
        model: Option<&str>,
        effort: Option<&str>,
        host: &HostEnv,
    ) -> Self {
        Self::with_providers(manifest, model, effort, host, &ProviderRegistry::builtin())
    }

    /// [`Self::new`] with these providers.
    pub fn with_providers(
        manifest: AgentManifest,
        model: Option<&str>,
        effort: Option<&str>,
        host: &HostEnv,
        providers: &ProviderRegistry,
    ) -> Self {
        let model = model
            .map(str::to_string)
            .or_else(|| manifest.default_model().map(str::to_string));
        let effort = effort.map(str::to_string);
        let (login, provider) = forwarding::resolve(
            &manifest,
            model.as_deref(),
            effort.as_deref(),
            host,
            providers,
        );
        Self {
            manifest,
            model,
            effort,
            login,
            provider,
        }
    }

    pub fn manifest(&self) -> &AgentManifest {
        &self.manifest
    }

    /// The login label, as people see it (never a value).
    pub fn login_label(&self) -> Option<&str> {
        self.login.as_ref().ok().map(|l| l.label.as_str())
    }
}

#[async_trait]
impl ModelAgent for AcpAgent {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn effort(&self) -> Option<&str> {
        self.effort.as_deref()
    }

    fn install_layers(&self) -> Vec<String> {
        let mut layers = vec![self.manifest.install_layer()];
        // The forwarder runs on BenchFlow's Node.js, which npm agents bring.
        if self.provider.is_some() && !matches!(self.manifest.install.method, InstallMethod::Npm(_))
        {
            layers.push(providers::node_layer());
        }
        layers
    }

    fn fingerprint(&self) -> String {
        self.manifest.version_fingerprint()
    }

    fn preflight(&self) -> Result<(), Problem> {
        self.login.as_ref().map(|_| ()).map_err(Clone::clone)
    }

    fn options(&self) -> serde_json::Map<String, serde_json::Value> {
        chosen_options(&self.manifest)
    }

    fn resource_floor(&self) -> benchflow_agent::manifest::ResourceFloor {
        self.manifest.resources
    }

    fn session_dirs(&self) -> Option<Vec<String>> {
        self.manifest
            .resume
            .acp
            .then(|| self.manifest.resume.session_dirs.clone())
    }

    fn skill_paths(&self) -> Vec<String> {
        self.manifest.skill_paths.clone()
    }

    fn session_file(&self) -> Option<benchflow_agent::manifest::SessionFile> {
        self.manifest
            .resume
            .load
            .filter(|_| self.manifest.resume.acp)
    }

    fn resumes_over_acp(&self) -> bool {
        self.manifest.resume.acp
    }

    fn knows_login_material(&self) -> bool {
        true
    }

    fn login_material(&self, home: &str) -> (Vec<String>, Vec<String>) {
        let (mut files, mut secrets) = (Vec::new(), Vec::new());
        if let Ok(login) = &self.login {
            files.extend(login.files.iter().map(|f| f.path_for(home)));
            secrets.extend(login.secrets.iter().cloned());
        }
        // The family's own credential files, whatever login was chosen: a
        // login from a variable has no file, but the agent may write or
        // refresh one there during its turn (a rotated token, which the
        // value scan would not know).
        let home = home.trim_end_matches('/');
        for (name, kind) in benchflow_agent::credentials::credential_kinds(&self.manifest) {
            use benchflow_agent::credentials::SourceKind;
            if matches!(kind, SourceKind::LoginFile | SourceKind::AuthJson)
                && let Some(rest) = name.strip_prefix("~/")
            {
                files.push(format!("{home}/{rest}"));
            }
        }
        // The settings files the launch writes: a child writes them again.
        files.extend(
            self.manifest
                .launch
                .files
                .iter()
                .map(|f| format!("{home}/{}", f.path.trim_start_matches('/'))),
        );
        if let Some(p) = &self.provider {
            secrets.push(p.key.value().to_string());
        }
        secrets.retain(|s| s.len() >= benchflow_agent::credentials::MIN_SECRET_LEN);
        let mut seen = std::collections::BTreeSet::new();
        files.retain(|f| seen.insert(f.clone()));
        (files, secrets)
    }

    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput {
        let login = match &self.login {
            Ok(login) => crate::agent::with_env_secrets(login.clone(), &ctx.env),
            Err(p) => {
                return AgentOutput {
                    problem: Some(p.clone()),
                    ..AgentOutput::default()
                };
            }
        };
        let mut out = AgentOutput {
            login: Some(login.label.clone()),
            usage: Usage {
                source: "unavailable".into(),
                ..Usage::default()
            },
            ..AgentOutput::default()
        };
        if let Err(p) = place_login(&ctx, &login).await {
            out.problem = Some(p);
            return out;
        }
        // Settings files the run's agent options wrote.
        let files: Vec<benchflow_agent::credentials::SandboxFile> = self
            .manifest
            .launch
            .files
            .iter()
            .map(|f| benchflow_agent::credentials::SandboxFile {
                path: format!("$HOME/{}", f.path),
                contents: f.contents.clone().into_bytes(),
                mode: f.mode,
            })
            .collect();
        if let Err(p) = place_files(&ctx, &files).await {
            out.problem = Some(p);
            return out;
        }

        // Through the forwarder, it holds the key; the agent gets its
        // address and a placeholder.
        let forwarder = match &self.provider {
            None => None,
            Some(p) => match forwarding::start(&ctx, p).await {
                Ok(forwarder) => Some(forwarder),
                Err(problem) => {
                    out.problem = Some(problem);
                    return out;
                }
            },
        };
        let mut env = agent_env(&self.manifest, &ctx, &login);
        if let Some(p) = &self.provider {
            forwarding::withhold_key(&mut env, p);
        }
        // For agents that take the model only at startup (select = "none"),
        // as the headless harness gives it; on a provider its variables do.
        if self.provider.is_none()
            && let Some(model) = &self.model
        {
            env.insert("BENCHFLOW_MODEL".into(), self.manifest.cli_model(model));
        }
        if let Some(effort) = &self.effort {
            env.insert("BENCHFLOW_EFFORT".into(), effort.clone());
        }
        if let (Some(p), Some(f)) = (&self.provider, &forwarder) {
            // The effort was checked when the agent was built.
            if let Ok(vars) = p.route.agent_env(&f.base_url(), self.effort.as_deref()) {
                env.extend(vars);
            }
        }
        out.env_keys = env.keys().cloned().collect();
        let exec = Exec::new(self.manifest.launch_command())
            .user(ctx.user.to_string())
            .cwd(ctx.workdir.to_string())
            .envs(env);
        let process = match ctx.sandbox.spawn(exec).await {
            Ok(p) => p,
            Err(e) => {
                out.problem = Some(sandbox_problem(Phase::Agent, e));
                return out;
            }
        };

        let log = AgentLog::open(&ctx.trial_dir.join("agent"), &self.manifest.name);
        let options = ClientOptions::for_agent(&self.manifest, Some(&login)).on_event(event_sink(
            ctx.events.clone(),
            ctx.trial.to_string(),
            log,
        ));
        let mut client = AcpClient::start(process, options);

        // The trajectory, rewritten while the agent works.
        let redactor = Redactor::new(login.secrets.iter().cloned());
        let path = ctx
            .trial_dir
            .join("trajectory")
            .join("acp_trajectory.jsonl");
        let writer = TrajectoryWriter::new(&path, redactor).await.ok();
        let handle = client.trajectory();
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
        let background = {
            let handle = handle.clone();
            tokio::spawn(async move {
                let mut writer = writer;
                loop {
                    let stop = tokio::select! {
                        _ = tokio::time::sleep(TRAJECTORY_EVERY) => false,
                        _ = &mut stop_rx => true,
                    };
                    if let Some(w) = writer.as_mut() {
                        let _ = w.write(handle.snapshot()).await;
                    }
                    if stop {
                        break;
                    }
                }
            })
        };

        let started = Instant::now();
        let result = self.converse(&mut client, &ctx, started, &mut out).await;
        let _ = stop_tx.send(());
        let _ = background.await;

        let events = handle.snapshot();
        let activity = handle.with(|t| t.activity());
        out.tool_calls = count_tool_calls(events.iter()) as u64;
        if let Some(usage) = handle.with(|t| t.usage()) {
            out.usage = engine_usage(usage);
        }
        match result {
            Ok(()) => {
                // A provider's error returned as the agent's whole reply
                // (Hermes ends its turn with a 401 as an ordinary message):
                // the provider failed, and the trial is not graded. A model
                // that wrote output tokens was reached, so words that read
                // like an error are its answer, graded like any other;
                // unless they say the stream broke mid-response, when the
                // tokens are the partial answer's: then the reply may open
                // with that partial answer, and its last line is the error.
                let reply = handle.with(|t| t.final_text());
                // After a partial answer, the provider's error ends the
                // reply ([`partial_then_error`]).
                let tail = partial_then_error(&reply);
                let failure = provider_failure_in_reply(&reply)
                    .map(|f| (f, false))
                    .or_else(|| {
                        let tail = tail.filter(|_| out.usage.output_tokens > 0)?;
                        provider_failure_in_reply(tail).map(|f| (f, true))
                    });
                let failure = failure.and_then(|((owner, line), after_partial)| {
                    // A model that wrote output tokens was reached: only a
                    // stream that broke, or (after a partial answer) a
                    // provider that then failed, is not its answer.
                    let broke = broke_mid_stream(&line)
                        || (after_partial
                            && matches!(owner, AcpOwner::ModelProvider)
                            && ends_with_api_error(&reply));
                    (out.usage.output_tokens == 0 || broke).then_some((owner, line))
                });
                if out.tool_calls == 0
                    && let Some((owner, line)) = failure
                {
                    // The reply is the agent's words, which the trajectory
                    // keeps as they are; the record must not hold a key.
                    let line = Redactor::new(login.secrets.iter().cloned())
                        .redact(&line)
                        .into_owned();
                    let message = format!(
                        "{} ended its turn with no tool call, replying only with a provider error: {line}",
                        self.manifest.name
                    );
                    let log_hint = format!(
                        "agent/{}.txt and agent/stderr.txt in the trial folder",
                        self.manifest.name
                    );
                    out.problem = Some(with_detail(
                        provider_problem(owner, message, &log_hint, &line),
                        client.stderr_tail(STDERR_TAIL),
                    ));
                }
            }
            Err(AcpError::Timeout { .. }) => {
                out.timed_out = true;
                // BenchFlow 0.8 (3842ed22): an agent that did nothing before
                // its time ran out is not a 0, it is an integration failure.
                let output_tokens = handle
                    .with(|t| t.usage())
                    .and_then(|u| u.output_tokens)
                    .unwrap_or(0);
                if activity.tool_calls == 0
                    && activity.message_chunks == 0
                    && activity.thought_chunks == 0
                    && output_tokens == 0
                {
                    out.problem = Some(
                        Problem::new(
                            "R206",
                            Owner::Agent,
                            Phase::Agent,
                            format!(
                                "{} did nothing before its time ran out (no tool call, message, thought, or output token)",
                                self.manifest.name
                            ),
                            format!(
                                "read agent/{}.txt in the trial folder: the agent may never have reached its model",
                                self.manifest.name
                            ),
                        )
                        .detail(client.stderr_tail(STDERR_TAIL)),
                    );
                }
            }
            Err(e) => {
                let started_prompt = !out.prompts.is_empty();
                out.problem = Some(acp_problem_reading(
                    &e,
                    &self.manifest.name,
                    started_prompt,
                    client.stderr_tail(STDERR_TAIL),
                    &client.stderr_tail(STDERR_READ),
                ));
            }
        }
        let partial = out.timed_out || out.problem.is_some();
        out.trajectory_summary =
            serde_json::to_value(summarize(events.iter(), partial, Some("acp")))
                .unwrap_or_default();
        let _ = client.shutdown().await;
        if let (Some(forwarder), Some(access)) = (forwarder, &self.provider) {
            let report = forwarding::keep_log(forwarder, access.pin.as_ref(), &ctx).await;
            forwarding::provider_usage(&report, &self.manifest.name, &mut out.usage);
            if let Some(problem) = forwarding::rejected_key(&report, access) {
                out.problem = Some(problem);
            }
        }
        out
    }
}

impl AcpAgent {
    /// Initialize, open the session, select the model, and send every
    /// prompt. Records the prompts sent in `out`.
    async fn converse(
        &self,
        client: &mut AcpClient,
        ctx: &AgentContext<'_>,
        started: Instant,
        out: &mut AgentOutput,
    ) -> Result<(), AcpError> {
        client.initialize().await?;
        out.agent_version = client
            .agent_info()
            .and_then(|info| info.version.clone())
            .filter(|v| !v.trim().is_empty());
        let mcp: Vec<McpServer> = ctx
            .mcp
            .iter()
            .map(McpServer::from_task_table)
            .collect::<Result<_, _>>()
            .map_err(|message| AcpError::Unsupported {
                message: format!("the task's MCP server table is invalid: {message}"),
            })?;
        let previous = ctx.resume.as_ref().map(|r| r.session_id.as_deref());
        let loaded = ctx.resume.as_ref().is_some_and(|r| r.loaded);
        if loaded && !client.can_load_session() {
            return Err(AcpError::Unsupported {
                message: format!(
                    "{} does not offer session/load, so it cannot resume the session from --load-trajectory (or a branch's parent)",
                    self.manifest.name
                ),
            });
        }
        if ctx.needs_load && !client.can_load_session() {
            return Err(AcpError::Unsupported {
                message: format!(
                    "{} does not offer session/load, so a branch's children could not resume this session",
                    self.manifest.name
                ),
            });
        }
        let session = match previous {
            Some(Some(id)) if client.can_load_session() => {
                out.resumed = true;
                client.load_session(id, ctx.workdir, &mcp).await?
            }
            Some(previous) => {
                out.resume_note = Some(match previous {
                    Some(_) => format!(
                        "{} does not offer session/load, so this stage started a fresh session",
                        self.manifest.name
                    ),
                    None => "the previous stage reported no session id, so this stage started a fresh session".to_string(),
                });
                client.new_session(ctx.workdir, &mcp).await?
            }
            None => client.new_session(ctx.workdir, &mcp).await?,
        };
        out.session_id = Some(session.session_id.clone());
        if self.model.is_none() {
            out.model = session
                .models
                .as_ref()
                .and_then(|m| m.get("currentModelId"))
                .and_then(|id| id.as_str())
                .map(str::to_string);
        }
        // On a provider the model id is the provider's, as the agent's
        // `[providers]` table spells it (or none: its variables carry it). On
        // the agent's own key, the agent's own, as without the forwarder.
        let model = match &self.provider {
            Some(p) if !p.route.own_key => {
                p.route
                    .acp_model(self.effort.as_deref())
                    .map_err(|e| AcpError::Unsupported {
                        message: e.to_string(),
                    })?
            }
            _ => self.model.clone(),
        };
        let choice = client
            .select_model(
                &self.manifest.model,
                model.as_deref(),
                self.effort.as_deref(),
            )
            .await?;
        // With no model asked for, an effort may still have moved the session
        // to another variant of its model (`gpt-6.1-sol[high]`): record that.
        if self.model.is_none()
            && let Some(id) = choice.model_id
        {
            out.model = Some(id);
        }
        let idle = (!ctx.idle_timeout.is_zero()).then_some(ctx.idle_timeout);
        let prompts =
            std::iter::once(ctx.instruction.to_string()).chain(ctx.stages.iter().cloned());
        for text in prompts {
            let wall = ctx.timeout.map(|t| t.saturating_sub(started.elapsed()));
            if wall.is_some_and(|w| w.is_zero()) {
                return Err(AcpError::Timeout {
                    after: ctx.timeout.unwrap_or_default(),
                    cancelled: false,
                });
            }
            out.prompts.push(text.clone());
            let limits = PromptLimits {
                wall,
                idle,
                pending_tool_grace: None,
            };
            let outcome = client.prompt(&text, &limits).await?;
            if outcome.stop_reason == StopReason::Refusal {
                // The model refused; the verifier still grades what is there.
                break;
            }
        }
        Ok(())
    }
}

/// The agent options a manifest was built with, for config.json.
pub(crate) fn chosen_options(
    manifest: &AgentManifest,
) -> serde_json::Map<String, serde_json::Value> {
    manifest
        .chosen_options
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap_or_default()))
        .collect()
}

/// Write the login's files for the agent user: theirs to read, nobody
/// else's.
pub(crate) async fn place_login(ctx: &AgentContext<'_>, login: &Login) -> Result<(), Problem> {
    place_files(ctx, &login.files).await
}

/// Write files for the agent user (`$HOME` is its home), owned by it. A
/// non-root agent user writes them itself ([`crate::place`]), so the
/// folders it makes on the way are its own.
pub(crate) async fn place_files(
    ctx: &AgentContext<'_>,
    files: &[benchflow_agent::credentials::SandboxFile],
) -> Result<(), Problem> {
    let home = ctx.home.trim_end_matches('/');
    if ctx.user == "root" {
        for file in files {
            ctx.sandbox
                .write_file(&file.path_for(home), &file.contents, file.mode)
                .await
                .map_err(|e| crate::trial::put_problem(Phase::Setup, e, ctx.earlier_turns))?;
        }
        return Ok(());
    }
    let files: Vec<crate::place::ForAgent> = files
        .iter()
        .map(|f| crate::place::ForAgent {
            path: f.path_for(home),
            contents: &f.contents,
            mode: f.mode,
        })
        .collect();
    let user = crate::place::AgentUser {
        name: ctx.user,
        home,
        earlier_turns: ctx.earlier_turns,
    };
    crate::place::for_agent(ctx.sandbox, &user, &files).await
}

/// The agent process's environment: the manifest's settings, then the task
/// and run environment, then the login (which always wins, so a task cannot
/// point a login at another server), and `HOME`.
fn agent_env(
    manifest: &AgentManifest,
    ctx: &AgentContext<'_>,
    login: &Login,
) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = manifest
        .launch_env()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    env.extend(ctx.env.clone());
    env.extend(login.env.clone());
    env.insert("HOME".into(), ctx.home.to_string());
    env
}

/// The agent's own output, for `agent/<agent>.txt` and `agent/stderr.txt`.
pub(crate) struct AgentLog {
    all: Option<File>,
    stderr: Option<File>,
}

impl AgentLog {
    pub(crate) fn open(dir: &Path, agent: &str) -> Arc<Mutex<Self>> {
        let _ = std::fs::create_dir_all(dir);
        let name = agent.replace(['/', '\\'], "_");
        Arc::new(Mutex::new(Self {
            all: File::create(dir.join(format!("{name}.txt"))).ok(),
            stderr: File::create(dir.join("stderr.txt")).ok(),
        }))
    }

    fn line(&mut self, stream: Option<OutputStream>, line: &str) {
        if let Some(f) = self.all.as_mut() {
            let _ = writeln!(f, "{line}");
        }
        if stream == Some(OutputStream::Stderr)
            && let Some(f) = self.stderr.as_mut()
        {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Live progress for the run display, and the agent log.
pub(crate) fn event_sink(
    events: crate::events::Events,
    trial: String,
    log: Arc<Mutex<AgentLog>>,
) -> impl Fn(AcpEvent) + Send + Sync + 'static {
    move |event| match event {
        AcpEvent::Progress(p) => events.send(Event::Activity {
            trial: trial.clone(),
            tool_calls: p.tool_calls as u64,
            last_tool: p.last_tool_title,
            tokens: p.tokens.unwrap_or(0),
        }),
        AcpEvent::AgentOutput { stream, line } => {
            if let Ok(mut log) = log.lock() {
                log.line(Some(stream), &line);
            }
        }
        AcpEvent::Notice { severity, title } => {
            if let Ok(mut log) = log.lock() {
                log.line(
                    None,
                    &format!("[benchflow] agent notice ({severity}): {title}"),
                );
            }
        }
        AcpEvent::Permission {
            tool_call,
            option_id,
        } => {
            if let Ok(mut log) = log.lock() {
                log.line(
                    None,
                    &format!(
                        "[benchflow] permission answered with {option_id} for {}",
                        tool_call.as_deref().unwrap_or("a request")
                    ),
                );
            }
        }
    }
}

fn engine_usage(u: AcpUsage) -> Usage {
    let mut usage = Usage {
        input_tokens: u.input_tokens.unwrap_or(0),
        output_tokens: u.output_tokens.unwrap_or(0),
        cache_read_tokens: u.cached_read_tokens.unwrap_or(0),
        cache_write_tokens: u.cached_write_tokens.unwrap_or(0),
        cost_usd: None,
        source: "acp".into(),
        cost_source: None,
        cost_note: None,
    };
    // ACP's usage has no price. A run through the forwarder gets the
    // provider's (`forwarding::provider_usage` replaces this note); one on
    // a subscription login or the agent's own key says why it has none.
    if usage.total() > 0 {
        usage.cost_note = Some(UNPRICED_ACP.into());
    }
    usage
}

/// Why an ACP run that used tokens outside the forwarder has no cost.
const UNPRICED_ACP: &str = "the agent reports tokens but not their price, and the run did not go through BenchFlow's forwarder (a subscription login, or the agent's own key), so the cost is unknown";

/// A failed exchange as a trial problem: its code, whose it is, and what to
/// do. `prompting` says whether a prompt had been sent (a crash before that
/// means the agent never started).
pub fn acp_problem(e: &AcpError, agent: &str, prompting: bool, stderr_tail: String) -> Problem {
    let log = stderr_tail.clone();
    acp_problem_reading(e, agent, prompting, stderr_tail, &log)
}

/// [`acp_problem`], reading `log` (more of the agent's stderr than the
/// detail keeps) for the cause of an error that names none.
pub fn acp_problem_reading(
    e: &AcpError,
    agent: &str,
    prompting: bool,
    stderr_tail: String,
    log: &str,
) -> Problem {
    let message = e.to_string();
    let next = e.next_step();
    let log_hint = format!("agent/{agent}.txt and agent/stderr.txt in the trial folder");
    if let Some((owner, line)) = e.provider_failure_in_log(log) {
        let message = format!("{message}; the agent's log says: {line}");
        let p = provider_problem(owner, message, &log_hint, &line);
        return with_detail(p, stderr_tail);
    }
    let p = match e {
        AcpError::HandshakeTimeout { .. } => Problem::new(
            "R202",
            Owner::Agent,
            Phase::Agent,
            message,
            format!(
                "the agent started but did not answer; see {log_hint}, and `bench agents check {agent} -s <backend>` checks that it starts"
            ),
        ),
        AcpError::Exited { .. } | AcpError::Io { .. } if !prompting => Problem::new(
            "R202",
            Owner::Agent,
            Phase::Agent,
            message,
            format!("{next}; see {log_hint}"),
        ),
        AcpError::Exited { .. } => Problem::new(
            "R204",
            Owner::Agent,
            Phase::Agent,
            message,
            format!("{next}; see {log_hint}"),
        ),
        AcpError::Io { .. } => {
            Problem::new("R204", Owner::Sandbox, Phase::Agent, message, next).transient()
        }
        AcpError::Idle { .. } => {
            Problem::new("R203", Owner::Agent, Phase::Agent, message, next).transient()
        }
        AcpError::Timeout { .. } => Problem::new("R206", Owner::Agent, Phase::Agent, message, next),
        AcpError::UsageLimit { .. } => {
            Problem::new("R213", Owner::Model, Phase::Agent, message, next)
        }
        AcpError::ModelNotOffered { .. } | AcpError::Unsupported { .. } => {
            Problem::new("R215", Owner::You, Phase::Agent, message, next)
        }
        AcpError::Agent { .. } if e.needs_sign_in() => {
            Problem::new("R211", Owner::You, Phase::Agent, message, next)
        }
        AcpError::Agent { .. } => match e.owner() {
            AcpOwner::Login => Problem::new("R212", Owner::Model, Phase::Agent, message, next),
            AcpOwner::User => Problem::new("R215", Owner::You, Phase::Agent, message, next),
            // A refused request (0.8's `provider_rejected`) is permanent.
            AcpOwner::ModelProvider if e.category() == "provider_rejected" => {
                Problem::new("R214", Owner::Model, Phase::Agent, message, next)
            }
            AcpOwner::ModelProvider => {
                Problem::new("R214", Owner::Model, Phase::Agent, message, next).transient()
            }
            _ => Problem::new("R205", Owner::Agent, Phase::Agent, message, next),
        },
        AcpError::Protocol { .. } => {
            Problem::new("R205", Owner::Agent, Phase::Agent, message, next)
        }
    };
    with_detail(p, stderr_tail)
}

/// Whether the reply ends with Claude Code's own `API Error:` (on its last
/// line, or run into the end of the last sentence), the only provider error,
/// besides a broken stream, that ends a partial answer as the provider's: an
/// answer that ends quoting a bare `Error: Connection refused` is the
/// model's.
fn ends_with_api_error(reply: &str) -> bool {
    partial_then_error(reply).is_some_and(|tail| tail.starts_with("API Error:"))
}

/// The error line that ends a reply after a partial answer, when there is
/// one: its last `API Error:` (else `Error:`, not inside a word such as
/// `TypeError:`) that starts the reply, a line, or follows the end of a
/// sentence, and runs to the end of the reply on one short line. An answer
/// that only quotes such an error mid-sentence is no error. Only an
/// `API Error:` line, or a broken stream, makes it the provider's
/// ([`ends_with_api_error`]); a bare `Error:` line is the model's answer.
///
/// A stream that broke usually cut the answer mid-word, and an adapter that
/// adds no line break joins the error straight on (`...loop bouAPI Error:
/// Server error mid-response`): there, the last `API Error:` at any place
/// counts when its line says the stream broke and ends the reply
/// ([`ends_mid_stream`]), which a quote (in its closing mark) or an answer
/// that goes on does not.
fn partial_then_error(reply: &str) -> Option<&str> {
    let reply = reply.trim_end();
    let opening = |needle: &str| {
        reply
            .match_indices(needle)
            .map(|(i, _)| i)
            .filter(|&i| {
                let before = reply[..i].trim_end_matches([' ', '\t']);
                before.is_empty()
                    || before.ends_with(['\n', '.', '!', '?', ':'])
                        && !reply[..i].ends_with(|c: char| c.is_alphanumeric())
            })
            .last()
    };
    let one_line = |at: usize| {
        let tail = &reply[at..];
        (at > 0 && !tail.contains('\n') && tail.len() <= 200).then_some(tail)
    };
    opening("API Error:")
        .or_else(|| opening("Error:"))
        .and_then(one_line)
        .or_else(|| {
            let at = reply.rfind("API Error:")?;
            one_line(at).filter(|tail| ends_mid_stream(tail))
        })
}

#[cfg(test)]
mod partial_tests {
    use super::partial_then_error;

    /// Guards the D7d fix of REVIEW-OUTCOMES (d7c) L4.
    #[test]
    fn only_an_error_that_ends_a_partial_answer_counts() {
        for (reply, want) in [
            (
                "Here is the fix.API Error: Server error mid-response",
                Some("API Error: Server error mid-response"),
            ),
            (
                "Here is the fix.\nAPI Error: Request timed out.",
                Some("API Error: Request timed out."),
            ),
            // Quoted mid-sentence, or inside a word: no error.
            (
                "To handle `API Error: Server error mid-response`, retry with backoff.",
                None,
            ),
            ("It raised a TypeError: x is undefined", None),
            // The whole reply is the error: that is the other check's.
            ("API Error: Server error mid-response", None),
            // A broken stream joined onto an answer cut mid-word or inside
            // code (REVIEW-B (d7d) F1).
            (
                "change the loop bouAPI Error: Server error mid-response",
                Some("API Error: Server error mid-response"),
            ),
            (
                "for i in range(nAPI Error: Server error mid-response.",
                Some("API Error: Server error mid-response."),
            ),
            // ...but not a quote, nor any other error after a mid-word cut.
            (
                "Claude Code prints `API Error: Server error mid-response`.",
                None,
            ),
            ("then retry the loop bouAPI Error: 529 Overloaded", None),
        ] {
            assert_eq!(partial_then_error(reply), want, "{reply}");
        }
    }

    /// After a partial answer, a final bare `Error:` line the model wrote
    /// (quoting what a client prints) is its answer; only Claude Code's own
    /// `API Error:` opening is the provider's. Guards the orchestrator's fix
    /// of REVIEW-A (d7d) finding 2.
    #[test]
    fn a_quoted_error_line_is_the_answer() {
        assert!(super::ends_with_api_error(
            "Here is the fix.\nAPI Error: Request timed out."
        ));
        assert!(!super::ends_with_api_error(
            "So the client prints:\nError: Connection refused"
        ));
        assert!(!super::ends_with_api_error("Here is the fix."));
    }
}

/// A provider failure found in the agent's log or reply, as a trial problem.
/// `line` is the provider's own words (the reply's or the log's line):
/// whether they are a refused request decides the retry, not the message
/// built around them.
fn provider_problem(owner: AcpOwner, message: String, log_hint: &str, line: &str) -> Problem {
    // A refused request gets the same answer again: not retried.
    let retry = !refused_request(line);
    match owner {
        AcpOwner::Login => Problem::new(
            "R212",
            Owner::Model,
            Phase::Agent,
            message,
            format!("the model provider rejected the login; check the key, and see {log_hint}"),
        ),
        AcpOwner::User => Problem::new(
            "R215",
            Owner::You,
            Phase::Agent,
            message,
            "the provider does not serve this model; pick another with -m",
        ),
        _ if !retry => Problem::new(
            "R214",
            Owner::Model,
            Phase::Agent,
            message,
            format!(
                "the model provider refused the request, which a retry would not change; check the model and the agent's settings for it, and see {log_hint}"
            ),
        ),
        _ => Problem::new(
            "R214",
            Owner::Model,
            Phase::Agent,
            message,
            format!("the model provider failed; retry later, and see {log_hint}"),
        )
        .transient(),
    }
}

fn with_detail(p: Problem, stderr_tail: String) -> Problem {
    if stderr_tail.trim().is_empty() {
        p
    } else {
        p.detail(stderr_tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unpriced_acp_run_says_why_it_has_no_cost() {
        // A Claude Code run on a subscription login: tokens, no price.
        let u = engine_usage(AcpUsage {
            input_tokens: Some(565_300),
            output_tokens: Some(14_200),
            ..AcpUsage::default()
        });
        assert_eq!(u.cost_usd, None);
        let note = u.cost_note.expect("a cost note");
        assert!(note.contains("subscription login"), "{note}");
        // No tokens, nothing to explain.
        assert_eq!(engine_usage(AcpUsage::default()).cost_note, None);
    }

    #[test]
    fn a_missing_login_names_the_vendor_account() {
        let cursor = benchflow_agent::Registry::builtin()
            .find("cursor")
            .unwrap()
            .clone();
        let agent = AcpAgent::new(cursor, None, None, &HostEnv::new(BTreeMap::new(), None));
        let p = agent.preflight().unwrap_err();
        assert_eq!(p.code, "R211");
        assert!(p.next.contains("needs a Cursor account"), "{}", p.next);
        assert_eq!(
            p.next.matches("Cursor account").count(),
            1,
            "the account is named once: {}",
            p.next
        );
    }

    /// A login from an environment variable leaves no file of its own, but
    /// the agent may write or refresh its family's credential file during
    /// the run (a rotated token the value scan does not know): a checkpoint
    /// removes those known paths whatever login was chosen.
    #[test]
    fn login_material_lists_the_familys_credential_files_for_an_env_login() {
        let reg = benchflow_agent::Registry::builtin();
        for (agent, var, file) in [
            (
                "claude",
                "CLAUDE_CODE_OAUTH_TOKEN",
                "/home/agent/.claude/.credentials.json",
            ),
            (
                "codex",
                "CODEX_ACCESS_TOKEN",
                "/home/agent/.codex/auth.json",
            ),
            (
                "gemini",
                "GEMINI_API_KEY",
                "/home/agent/.gemini/oauth_creds.json",
            ),
        ] {
            let manifest = reg.find(agent).unwrap().clone();
            let host = HostEnv::new(
                BTreeMap::from([(var.to_string(), "tok-0123456789abcdef-fake".to_string())]),
                None,
            );
            let a = AcpAgent::new(manifest, None, None, &host);
            assert!(a.login.is_ok(), "{agent}: the env login is found");
            let (files, secrets) = a.login_material("/home/agent");
            assert!(files.iter().any(|f| f == file), "{agent}: {files:?}");
            assert!(
                secrets.iter().any(|s| s == "tok-0123456789abcdef-fake"),
                "{agent}: the env value is scanned for"
            );
            let mut dedup = files.clone();
            dedup.sort();
            dedup.dedup();
            assert_eq!(
                dedup.len(),
                files.len(),
                "{agent}: no path twice: {files:?}"
            );
        }
    }

    #[test]
    fn failures_get_codes_and_owners() {
        let exited = AcpError::Exited {
            code: Some(1),
            stderr_tail: String::new(),
        };
        assert_eq!(
            acp_problem(&exited, "codex", false, String::new()).code,
            "R202"
        );
        assert_eq!(
            acp_problem(&exited, "codex", true, String::new()).code,
            "R204"
        );
        let usage = AcpError::UsageLimit {
            reset_at: None,
            window: Some("5-hour".into()),
            message: "usage limit reached".into(),
            login: None,
        };
        let p = acp_problem(&usage, "claude-code", true, String::new());
        assert_eq!((p.code.as_str(), p.owner), ("R213", Owner::Model));
        let model = AcpError::ModelNotOffered {
            agent: "codex".into(),
            model: "gpt-0".into(),
            offered: vec!["gpt-5.5".into()],
        };
        let p = acp_problem(&model, "codex", false, String::new());
        assert_eq!((p.code.as_str(), p.owner), ("R215", Owner::You));
        let p = acp_problem(
            &AcpError::Protocol {
                message: "bad answer".into(),
            },
            "x",
            true,
            "tail".into(),
        );
        assert_eq!(p.code, "R205");
        assert_eq!(p.detail.as_deref(), Some("tail"));
    }

    #[test]
    fn todays_sign_in_and_log_only_failures() {
        // cline-acp in today's gateway sweep.
        let sign_in = AcpError::Agent {
            code: -32000,
            message: "Authentication required: Call authenticate before starting a session".into(),
            data: None,
            method: "session/new".into(),
        };
        let p = acp_problem(&sign_in, "cline", false, String::new());
        assert_eq!((p.code.as_str(), p.owner), ("R211", Owner::You));
        // openhands-acp: a bare -32603, the cause far up its stderr.
        let opaque = AcpError::Agent {
            code: -32603,
            message: "Internal error".into(),
            data: None,
            method: "session/prompt".into(),
        };
        let log = format!(
            "litellm.AuthenticationError:\n{}",
            "  │ /opt/benchflow/agents/openhands │\n".repeat(50)
        );
        let p = acp_problem_reading(&opaque, "openhands", true, "tail".into(), &log);
        assert_eq!((p.code.as_str(), p.owner), ("R212", Owner::Model));
        assert!(
            p.message.ends_with("litellm.AuthenticationError:"),
            "{}",
            p.message
        );
        let p = acp_problem(&opaque, "openhands", true, "nothing useful".into());
        assert_eq!((p.code.as_str(), p.owner), ("R205", Owner::Agent));
    }
}
