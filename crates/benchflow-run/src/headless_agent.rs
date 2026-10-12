//! Model agents run headless: the agent's own non-interactive mode.
//!
//! [`HeadlessAgent`] joins a manifest's `[headless]` table to a trial. The
//! install is the manifest's (the same cached image layer as its ACP
//! harness). Before the agent starts, the login's files and the table's
//! `files` are written into the agent user's home, and the instruction to
//! `~/.benchflow/instruction.md`. The command then runs as the agent user in
//! the task's working folder with `$BENCHFLOW_INSTRUCTION_FILE`,
//! `$BENCHFLOW_INSTRUCTION`, `$BENCHFLOW_MODEL` and `$BENCHFLOW_EFFORT` set,
//! and a [`Runner`] reads its stdout through the manifest's parser.
//!
//! As for ACP, the trajectory is rewritten to
//! `trajectory/acp_trajectory.jsonl` while the agent works, what the agent
//! writes that is not part of its format goes to `agent/<agent>.txt` (stderr
//! also to `agent/stderr.txt`), and failures get the same R2xx codes. A model
//! that names a provider, or a plain API key the agent found, runs through
//! BenchFlow's forwarder as for ACP ([`crate::forwarding`]): the command gets
//! the forwarder's address and a placeholder key, never the key, and on a
//! router every request is kept off denied upstreams.

use std::collections::BTreeMap;

use async_trait::async_trait;
use benchflow_agent::AgentManifest;
use benchflow_agent::acp::EventSink;
use benchflow_agent::credentials::{HostEnv, Login, SandboxFile};
use benchflow_agent::headless::{HeadlessError, Runner, RunnerOptions};
use benchflow_agent::manifest::{Harness, HeadlessSpec, InstallMethod};
use benchflow_agent::providers::{self, Forwarder, ProviderRegistry};
use benchflow_agent::trajectory::{
    Redactor, TrajectoryWriter, Usage as AgentUsage, count_tool_calls, summarize,
};
use benchflow_sandbox::Exec;

use crate::acp_agent::{
    AgentLog, STDERR_TAIL, TRAJECTORY_EVERY, event_sink, place_files, place_login,
};
use crate::agent::{AgentContext, AgentOutput, COST_BY_AGENT, ModelAgent, Usage};
use crate::forwarding::{self, ProviderAccess};
use crate::outcome::{Owner, Phase, Problem};
use crate::trial::sandbox_problem;

/// Where the instruction is written, under the agent user's home.
const INSTRUCTION_FILE: &str = ".benchflow/instruction.md";
/// Instructions up to this size also go in `$BENCHFLOW_INSTRUCTION`; a
/// larger one would overflow Linux's limit on one environment string.
const INSTRUCTION_ENV_MAX: usize = 100 * 1024;

/// An agent from a manifest, run headless on the login found for it.
pub struct HeadlessAgent {
    manifest: AgentManifest,
    /// `codex` on the agent's default harness, `codex:headless` otherwise,
    /// so a job of one harness never resumes a job of the other.
    label: String,
    model: Option<String>,
    effort: Option<String>,
    login: Result<Login, Problem>,
    /// Set when the run goes through the forwarder (a model that names a
    /// provider, or the agent's own API key): the endpoint, and the key the
    /// forwarder holds.
    provider: Option<ProviderAccess>,
}

impl HeadlessAgent {
    /// The agent `manifest` with `model` (else the manifest's default) and
    /// `effort`, on the first usable login in `host` (its plain API key held
    /// by the forwarder when it can be), or on the built-in provider the
    /// model names. The manifest must declare the headless harness.
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
        let (login, provider) =
            forwarding::resolve(&manifest, model.as_deref(), effort, host, providers);
        let label = if manifest.harness == Harness::Headless {
            manifest.name.clone()
        } else {
            format!("{}:headless", manifest.name)
        };
        Self {
            manifest,
            label,
            model,
            effort: effort.map(str::to_string),
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

    /// `$BENCHFLOW_MODEL`: on a provider, its model id as the agent's
    /// `[providers]` table spells it (empty when the table's variables carry
    /// it); otherwise the run's model as the agent's CLI takes it.
    fn cli_model(&self) -> String {
        match &self.provider {
            Some(p) if !p.route.own_key => p
                .route
                .acp_model(self.effort.as_deref())
                .ok()
                .flatten()
                .unwrap_or_default(),
            _ => self
                .model
                .as_deref()
                .map(|m| self.manifest.cli_model(m))
                .unwrap_or_default(),
        }
    }

    fn spec(&self) -> Result<&HeadlessSpec, Problem> {
        self.manifest.headless.as_ref().ok_or_else(|| {
            Problem::new(
                "R215",
                Owner::You,
                Phase::Setup,
                format!("agent `{}` has no headless harness", self.manifest.name),
                "run it with --harness acp, or add a [headless] table to its manifest",
            )
        })
    }

    /// The command's environment: the table's settings, then the task and
    /// run environment, then the login (which always wins), then, through
    /// the forwarder, its address and the placeholder key, then what
    /// BenchFlow sets.
    fn env(
        &self,
        spec: &HeadlessSpec,
        ctx: &AgentContext<'_>,
        login: &Login,
        instruction_file: &str,
        forwarder: Option<&Forwarder>,
    ) -> BTreeMap<String, String> {
        let mut env: BTreeMap<String, String> = spec
            .env
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        env.extend(ctx.env.clone());
        env.extend(login.env.clone());
        if let Some(p) = &self.provider {
            forwarding::withhold_key(&mut env, p);
        }
        if let (Some(p), Some(f)) = (&self.provider, forwarder) {
            // The effort was checked when the agent was built.
            if let Ok(vars) = p.route.agent_env(&f.base_url(), self.effort.as_deref()) {
                env.extend(vars);
            }
        }
        env.insert("HOME".into(), ctx.home.to_string());
        // The command turns this into the CLI's own resume flag.
        match &ctx.resume {
            Some(_) if self.manifest.resume.headless => {
                env.insert("BENCHFLOW_RESUME".into(), "1".into());
            }
            _ => {
                env.remove("BENCHFLOW_RESUME");
            }
        }
        env.insert(
            "BENCHFLOW_INSTRUCTION_FILE".into(),
            instruction_file.to_string(),
        );
        if ctx.instruction.len() <= INSTRUCTION_ENV_MAX {
            env.insert("BENCHFLOW_INSTRUCTION".into(), ctx.instruction.to_string());
        } else {
            env.remove("BENCHFLOW_INSTRUCTION");
        }
        env.insert("BENCHFLOW_MODEL".into(), self.cli_model());
        env.insert(
            "BENCHFLOW_EFFORT".into(),
            self.effort.clone().unwrap_or_default(),
        );
        env
    }
}

#[async_trait]
impl ModelAgent for HeadlessAgent {
    fn name(&self) -> &str {
        &self.label
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
        self.spec()?;
        self.login.as_ref().map(|_| ()).map_err(Clone::clone)
    }

    fn options(&self) -> serde_json::Map<String, serde_json::Value> {
        crate::acp_agent::chosen_options(&self.manifest)
    }

    fn resource_floor(&self) -> benchflow_agent::manifest::ResourceFloor {
        self.manifest.resources
    }

    fn session_dirs(&self) -> Option<Vec<String>> {
        self.manifest
            .resume
            .headless
            .then(|| self.manifest.resume.session_dirs.clone())
    }

    fn skill_paths(&self) -> Vec<String> {
        self.manifest.skill_paths.clone()
    }

    fn session_file(&self) -> Option<benchflow_agent::manifest::SessionFile> {
        self.manifest
            .resume
            .load
            .filter(|_| self.manifest.resume.headless)
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
        let agent = self.manifest.name.as_str();
        let spec = match self.spec() {
            Ok(spec) => spec,
            Err(p) => {
                out.problem = Some(p);
                return out;
            }
        };
        if let Some(p) = unsupported(agent, &ctx) {
            out.problem = Some(p);
            return out;
        }
        if let Err(p) = place_login(&ctx, &login).await {
            out.problem = Some(p);
            return out;
        }
        let mut files: Vec<SandboxFile> = spec
            .files
            .iter()
            .map(|f| SandboxFile {
                path: format!("$HOME/{}", f.path),
                contents: f.contents.clone().into_bytes(),
                mode: f.mode,
            })
            .collect();
        // Scripts the output format's command runs (a trajectory follower).
        files.extend(
            spec.output
                .helper_files()
                .iter()
                .map(|(path, contents)| SandboxFile {
                    path: format!("$HOME/{path}"),
                    contents: contents.as_bytes().to_vec(),
                    mode: 0o755,
                }),
        );
        files.push(SandboxFile {
            path: format!("$HOME/{INSTRUCTION_FILE}"),
            contents: ctx.instruction.as_bytes().to_vec(),
            mode: 0o600,
        });
        if let Err(p) = place_files(&ctx, &files).await {
            out.problem = Some(p);
            return out;
        }

        // Through the forwarder, it holds the key; the command gets its
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
        let pin = self.provider.as_ref().and_then(|p| p.pin.as_ref());

        let instruction_file = format!("{}/{INSTRUCTION_FILE}", ctx.home.trim_end_matches('/'));
        let env = self.env(spec, &ctx, &login, &instruction_file, forwarder.as_ref());
        out.env_keys = env.keys().cloned().collect();
        out.resumed = env.contains_key("BENCHFLOW_RESUME");
        out.prompts.push(ctx.instruction.to_string());
        let exec = Exec::new(spec.command.clone())
            .user(ctx.user.to_string())
            .cwd(ctx.workdir.to_string())
            .envs(env);
        let process = match ctx.sandbox.spawn(exec).await {
            Ok(p) => p,
            Err(e) => {
                out.problem = Some(sandbox_problem(Phase::Agent, e));
                if let Some(forwarder) = forwarder {
                    forwarding::keep_log(forwarder, pin, &ctx).await;
                }
                return out;
            }
        };

        let log = AgentLog::open(&ctx.trial_dir.join("agent"), agent);
        let sink: EventSink =
            std::sync::Arc::new(event_sink(ctx.events.clone(), ctx.trial.to_string(), log));
        let redactor = Redactor::new(login.secrets.iter().cloned());
        let runner = Runner::new(RunnerOptions {
            wall: ctx.timeout,
            idle: (!ctx.idle_timeout.is_zero()).then_some(ctx.idle_timeout),
            redactor: redactor.clone(),
            sink: Some(sink),
            login: Some(login.label.clone()),
            ..RunnerOptions::default()
        });
        let recording = runner.recording();

        // The trajectory, rewritten while the agent works.
        let path = ctx
            .trial_dir
            .join("trajectory")
            .join("acp_trajectory.jsonl");
        let writer = TrajectoryWriter::new(&path, redactor).await.ok();
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
        let background = {
            let recording = recording.clone();
            tokio::spawn(async move {
                let mut writer = writer;
                loop {
                    let stop = tokio::select! {
                        _ = tokio::time::sleep(TRAJECTORY_EVERY) => false,
                        _ = &mut stop_rx => true,
                    };
                    if let Some(w) = writer.as_mut() {
                        let _ = w.write(recording.snapshot()).await;
                    }
                    if stop {
                        break;
                    }
                }
            })
        };

        let result = runner
            .run(process, spec.output.parser(), ctx.instruction)
            .await;
        let _ = stop_tx.send(());
        let _ = background.await;

        let events = recording.snapshot();
        out.tool_calls = count_tool_calls(events.iter()) as u64;
        if let Some(usage) = recording.usage() {
            out.usage = engine_usage(usage, recording.cost_usd());
        } else if let Some(cost) = recording.cost_usd() {
            out.usage.cost_usd = Some(cost);
            out.usage.cost_source = Some(COST_BY_AGENT.into());
        }
        match result {
            Ok(_) => {}
            Err(HeadlessError::Timeout { .. }) => {
                out.timed_out = true;
                // As for ACP (0.8's 3842ed22): an agent that did nothing
                // before its time ran out is an integration failure, not a 0.
                let activity = recording.with(|t| t.activity());
                let output_tokens = recording.usage().and_then(|u| u.output_tokens).unwrap_or(0);
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
                                "{agent} did nothing before its time ran out (no tool call, message, thought, or output token)"
                            ),
                            format!(
                                "read agent/{agent}.txt and agent/stderr.txt in the trial folder: the agent may never have reached its model"
                            ),
                        )
                        .detail(recording.stderr_tail(STDERR_TAIL)),
                    );
                }
            }
            Err(e) => {
                out.problem = Some(headless_problem(
                    &e,
                    agent,
                    recording.stderr_tail(STDERR_TAIL),
                ));
            }
        }
        let partial = out.timed_out || out.problem.is_some();
        out.trajectory_summary =
            serde_json::to_value(summarize(events.iter(), partial, Some("headless")))
                .unwrap_or_default();
        if let (Some(forwarder), Some(access)) = (forwarder, &self.provider) {
            let report = forwarding::keep_log(forwarder, pin, &ctx).await;
            forwarding::provider_usage(&report, agent, &mut out.usage);
            if let Some(problem) = forwarding::rejected_key(&report, access) {
                out.problem = Some(problem);
            }
        }
        out
    }
}

/// What the headless harness cannot do for this task, as a problem.
fn unsupported(agent: &str, ctx: &AgentContext<'_>) -> Option<Problem> {
    if !ctx.stages.is_empty() {
        return Some(Problem::new(
            "R215",
            Owner::You,
            Phase::Setup,
            format!(
                "the headless harness runs one prompt, and this task has {} more stage(s)",
                ctx.stages.len()
            ),
            format!("run {agent} on this task with --harness acp"),
        ));
    }
    if !ctx.mcp.is_empty() {
        return Some(Problem::new(
            "R215",
            Owner::You,
            Phase::Setup,
            "the headless harness does not pass the task's MCP servers to the agent yet",
            format!("run {agent} on this task with --harness acp"),
        ));
    }
    None
}

fn engine_usage(u: AgentUsage, cost: Option<f64>) -> Usage {
    Usage {
        input_tokens: u.input_tokens.unwrap_or(0),
        output_tokens: u.output_tokens.unwrap_or(0),
        cache_read_tokens: u.cached_read_tokens.unwrap_or(0),
        cache_write_tokens: u.cached_write_tokens.unwrap_or(0),
        cost_usd: cost,
        source: "headless".into(),
        // The agent's own figure; a run through the forwarder replaces it
        // with the provider's (`forwarding::provider_usage`).
        cost_source: cost.map(|_| COST_BY_AGENT.into()),
        // As an ACP run: say why tokens have no cost.
        cost_note: (cost.is_none()
            && u.input_tokens.unwrap_or(0) + u.output_tokens.unwrap_or(0) > 0)
            .then(|| {
                "the agent reported tokens but no cost, and the run did not go through BenchFlow's forwarder, so the cost is unknown".into()
            }),
    }
}

/// A failed headless run as a trial problem: its code, whose it is, and
/// what to do. Codes match the ACP harness's.
pub fn headless_problem(e: &HeadlessError, agent: &str, stderr_tail: String) -> Problem {
    let message = e.to_string();
    let next = e.next_step(agent);
    let p = match e {
        HeadlessError::Exited { started: false, .. } => {
            Problem::new("R202", Owner::Agent, Phase::Agent, message, next)
        }
        HeadlessError::Exited { .. } => {
            Problem::new("R204", Owner::Agent, Phase::Agent, message, next)
        }
        HeadlessError::Io { .. } => {
            Problem::new("R204", Owner::Sandbox, Phase::Agent, message, next).transient()
        }
        HeadlessError::Idle { .. } => {
            Problem::new("R203", Owner::Agent, Phase::Agent, message, next).transient()
        }
        HeadlessError::Timeout { .. } => {
            Problem::new("R206", Owner::Agent, Phase::Agent, message, next)
        }
        HeadlessError::UsageLimit { .. } => {
            Problem::new("R213", Owner::Model, Phase::Agent, message, next)
        }
        HeadlessError::LoginRejected { .. } => {
            Problem::new("R212", Owner::Model, Phase::Agent, message, next)
        }
        HeadlessError::ModelUnavailable { .. } | HeadlessError::Setup { .. } => {
            Problem::new("R215", Owner::You, Phase::Agent, message, next)
        }
        HeadlessError::Provider { retryable, .. } => {
            let p = Problem::new("R214", Owner::Model, Phase::Agent, message, next);
            if *retryable { p.transient() } else { p }
        }
        HeadlessError::AgentFailed { .. } => {
            Problem::new("R205", Owner::Agent, Phase::Agent, message, next)
        }
    };
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
    fn an_unpriced_headless_run_says_why_it_has_no_cost() {
        let used = AgentUsage {
            input_tokens: Some(1_000),
            output_tokens: Some(50),
            ..AgentUsage::default()
        };
        let note = engine_usage(used, None).cost_note.expect("a cost note");
        assert!(note.contains("forwarder"), "{note}");
        // A cost of the agent's own, or no tokens: nothing to explain.
        assert_eq!(engine_usage(used, Some(0.01)).cost_note, None);
        assert_eq!(engine_usage(AgentUsage::default(), None).cost_note, None);
    }

    #[test]
    fn a_missing_login_names_the_vendor_account() {
        let manifest = benchflow_agent::Registry::builtin()
            .find("antigravity-sdk")
            .unwrap()
            .clone();
        let host = HostEnv::new(Default::default(), None);
        let agent = HeadlessAgent::new(manifest, None, None, &host);
        let p = agent.preflight().unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R211", Owner::You));
        assert!(p.next.contains("needs a Google account"), "{}", p.next);
        assert_eq!(
            p.next.matches("Google account").count(),
            1,
            "the account is named once: {}",
            p.next
        );
    }

    #[test]
    fn failures_get_the_acp_harness_codes() {
        let cases = [
            (
                HeadlessError::Exited {
                    code: Some(127),
                    stderr_tail: String::new(),
                    started: false,
                },
                "R202",
            ),
            (
                HeadlessError::Exited {
                    code: Some(1),
                    stderr_tail: String::new(),
                    started: true,
                },
                "R204",
            ),
            (
                HeadlessError::UsageLimit {
                    reset_at: None,
                    window: None,
                    message: "You've hit your usage limit".into(),
                    login: None,
                },
                "R213",
            ),
            (
                HeadlessError::LoginRejected {
                    message: "Invalid API key".into(),
                },
                "R212",
            ),
            (
                HeadlessError::Provider {
                    message: "overloaded".into(),
                    retryable: true,
                    category: "infra_failure",
                },
                "R214",
            ),
            (
                HeadlessError::ModelUnavailable {
                    message: "no such model".into(),
                },
                "R215",
            ),
            (
                HeadlessError::AgentFailed {
                    message: "x".into(),
                    code: None,
                },
                "R205",
            ),
            (
                HeadlessError::Setup {
                    message: "langgraph runs your graph, and /app/langgraph.json does not exist"
                        .into(),
                    fix: Some("pass --env LANGGRAPH_PROJECT=<folder>".into()),
                },
                "R215",
            ),
        ];
        for (e, code) in cases {
            let p = headless_problem(&e, "codex", String::new());
            assert_eq!(p.code, code, "{e}");
            assert!(!p.next.is_empty());
        }
        let p = headless_problem(
            &HeadlessError::LoginRejected {
                message: "Invalid API key".into(),
            },
            "codex",
            "tail".into(),
        );
        assert_eq!((p.owner, p.detail.as_deref()), (Owner::Model, Some("tail")));
    }
}
