//! Native agents: loops that run in `bench` on the host and act in the
//! sandbox through the sandbox API.
//!
//! [`NativeAgent`] joins a loop from `benchflow-agent`'s [`native`] module
//! to a trial. The model is called from the host, so no login, key, or
//! forwarder goes into the sandbox, and the agent needs no network there.
//! The only image layer is what the loop itself needs in the sandbox (tmux
//! for Terminus-2, nothing for `builtin`).
//!
//! The trajectory is rewritten to `trajectory/acp_trajectory.jsonl` while
//! the loop works; every model request and response goes to
//! `agent/model_calls.jsonl` with the key redacted. Token usage and the
//! provider's reported cost go into the trial record, and a loop's failure
//! becomes a trial problem with the R2xx code that names its owner.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::native::{
    self, BuiltinProviders, ClientOptions, Endpoint, EndpointResolver, Limits, ModelClient,
    ModelErrorKind, NativeLoop, Recorder, RunContext, Stop, TokenUsage, client::calls_path,
    terminus,
};
use benchflow_agent::providers::routing::{self, PinError};
use benchflow_agent::trajectory::{Redactor, TrajectoryWriter, count_tool_calls, summarize};
use sha2::{Digest, Sha256};

use crate::agent::{AgentContext, AgentOutput, COST_BY_PROVIDER, ModelAgent, Usage};
use crate::events::Event;
use crate::outcome::{Owner, Phase, Problem};
use crate::trial::sandbox_problem;

/// How often the trajectory file is rewritten while the loop works.
const TRAJECTORY_EVERY: Duration = Duration::from_secs(2);

/// A native agent (`builtin`, `terminus-2`) on a model endpoint.
pub struct NativeAgent {
    name: &'static str,
    model: Option<String>,
    native: NativeLoop,
    /// Resolved when the agent is built; a missing key or an unknown
    /// provider is reported by [`ModelAgent::preflight`].
    endpoint: Result<Endpoint, Problem>,
    client_options: ClientOptions,
    /// The agent options the run set (`--ak`).
    options: benchflow_agent::manifest::OptionValues,
}

impl NativeAgent {
    /// The native agent `name` (or alias) on `model`, with keys from
    /// `host`. `None` when `name` is not a native agent.
    pub fn new(
        name: &str,
        model: Option<&str>,
        effort: Option<&str>,
        host: &HostEnv,
    ) -> Option<Self> {
        Self::with_resolver(name, model, effort, host, &BuiltinProviders)
    }

    /// As [`NativeAgent::new`], resolving models with `resolver` (the
    /// provider registry, or a test's).
    pub fn with_resolver(
        name: &str,
        model: Option<&str>,
        effort: Option<&str>,
        host: &HostEnv,
        resolver: &dyn EndpointResolver,
    ) -> Option<Self> {
        let name = native::find(name)?;
        let mut native = NativeLoop::for_agent(name)?;
        match &mut native {
            NativeLoop::Builtin(c) => c.reasoning_effort = effort.map(str::to_string),
            NativeLoop::Terminus(c) => c.reasoning_effort = effort.map(str::to_string),
            NativeLoop::Computer(c) => c.reasoning_effort = effort.map(str::to_string),
        }
        let endpoint = match model {
            None => Err(Problem::new(
                "R215",
                Owner::You,
                Phase::Setup,
                format!("{name} needs a model and none was given"),
                "pass --model <provider>/<model>, such as --model vercel/deepseek/deepseek-v4-flash-0731",
            )),
            Some(m) => resolver
                .resolve(m, host)
                .map_err(|e| {
                    let code = match e {
                        native::EndpointError::NoKey { .. } => "R211",
                        _ => "R215",
                    };
                    Problem::new(code, Owner::You, Phase::Setup, e.to_string(), e.next_step())
                })
                .and_then(|endpoint| pin_upstreams(endpoint, host)),
        };
        Some(Self {
            name,
            model: model.map(str::to_string),
            native,
            endpoint,
            client_options: ClientOptions::default(),
            options: Default::default(),
        })
    }

    /// Change how the model client retries and waits.
    pub fn client_options(mut self, options: ClientOptions) -> Self {
        self.client_options = options;
        self
    }

    /// The options this agent takes (`--ak`).
    pub fn declared_options(&self) -> Vec<benchflow_agent::manifest::AgentOption> {
        native::options::declared(self.name)
    }

    /// Check the options `given` (name and text) and apply them to the loop.
    pub fn with_options(
        mut self,
        given: &[(String, String)],
    ) -> Result<Self, benchflow_agent::manifest::OptionError> {
        let values = benchflow_agent::manifest::options::resolve(
            self.name,
            &self.declared_options(),
            given,
            None,
        )?;
        self.native.apply_options(&values);
        self.options = values;
        Ok(self)
    }

    /// The loop and its settings.
    pub fn native_loop_mut(&mut self) -> &mut NativeLoop {
        &mut self.native
    }

    /// Where the key comes from, as people see it (never a value).
    pub fn login_label(&self) -> Option<&str> {
        self.endpoint.as_ref().ok().map(|e| e.key_label.as_str())
    }
}

/// On a router, keep every request off denied upstreams (never Baseten): on
/// the Vercel gateway the model's upstream list is read once per process, the
/// denied ones dropped, and the rest go in each request; on OpenRouter each
/// request names the denied ones. A model no allowed upstream serves stops
/// the run here, before any sandbox starts.
fn pin_upstreams(mut endpoint: Endpoint, host: &HostEnv) -> Result<Endpoint, Problem> {
    let Some(upstreams) = routing::for_native_endpoint(&endpoint.provider, &endpoint.base_url)
    else {
        return Ok(endpoint);
    };
    match routing::pin_blocking(&endpoint.provider, &upstreams, Some(&endpoint.model), host) {
        Ok(pin) => {
            endpoint.upstreams = Some(pin.upstreams);
            Ok(endpoint)
        }
        Err(e) => {
            let (code, owner) = match e {
                PinError::Unlisted { .. } => ("R214", Owner::Model),
                PinError::AllDenied { .. }
                | PinError::NotServed { .. }
                | PinError::NeedsModel { .. } => ("R215", Owner::You),
            };
            Err(Problem::new(
                code,
                owner,
                Phase::Setup,
                e.to_string(),
                e.next_step(),
            ))
        }
    }
}

#[async_trait]
impl ModelAgent for NativeAgent {
    fn name(&self) -> &str {
        self.name
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn effort(&self) -> Option<&str> {
        match &self.native {
            NativeLoop::Builtin(c) => c.reasoning_effort.as_deref(),
            NativeLoop::Terminus(c) => c.reasoning_effort.as_deref(),
            NativeLoop::Computer(c) => c.reasoning_effort.as_deref(),
        }
    }

    fn install_layers(&self) -> Vec<String> {
        match self.native {
            // computer-1 installs nothing: the task's image has the desktop.
            NativeLoop::Builtin(_) | NativeLoop::Computer(_) => Vec::new(),
            NativeLoop::Terminus(_) => vec![terminus::tmux::INSTALL_SCRIPT.to_string()],
        }
    }

    fn fingerprint(&self) -> String {
        let mut h = Sha256::new();
        for layer in self.install_layers() {
            h.update(layer.as_bytes());
        }
        format!("native:{}:{}", self.name, &hex::encode(h.finalize())[..12])
    }

    fn preflight(&self) -> Result<(), Problem> {
        self.endpoint.as_ref().map(|_| ()).map_err(Clone::clone)
    }

    fn options(&self) -> serde_json::Map<String, serde_json::Value> {
        self.options
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap_or_default()))
            .collect()
    }

    /// The model is called from the host; the sandbox needs no network.
    fn needs_network(&self) -> bool {
        false
    }

    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput {
        let endpoint = match &self.endpoint {
            Ok(e) => e.clone(),
            Err(p) => {
                return AgentOutput {
                    problem: Some(p.clone()),
                    ..AgentOutput::default()
                };
            }
        };
        let mut out = AgentOutput {
            login: Some(endpoint.key_label.clone()),
            usage: Usage {
                source: "unavailable".into(),
                ..Usage::default()
            },
            ..AgentOutput::default()
        };
        let provider = endpoint.provider.clone();
        let key_label = endpoint.key_label.clone();
        let redactor = Redactor::new([endpoint.key.expose().to_string()])
            .with_secrets(crate::agent::env_secrets(&ctx.env));
        let client = match ModelClient::new(endpoint, self.client_options.clone()) {
            Ok(c) => c.with_log(&calls_path(&ctx.trial_dir), redactor.clone()),
            Err(e) => {
                out.problem = Some(Problem::new(
                    "R901",
                    Owner::Benchflow,
                    Phase::Agent,
                    format!("the model client could not be built: {}", e.message),
                    "this is a BenchFlow bug; please report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
                ));
                return out;
            }
        };
        let client = Arc::new(client);

        // The key stays on this machine: a run variable holding it (`--env
        // MIMO_API_KEY=...`) never reaches the sandbox's commands.
        let mut env = without_key(&ctx.env, client.endpoint().key.expose());
        env.insert("HOME".into(), ctx.home.to_string());
        out.env_keys = env.keys().cloned().collect();
        let prompts: Vec<String> = std::iter::once(ctx.instruction.to_string())
            .chain(ctx.stages.iter().cloned())
            .collect();
        out.prompts = prompts.clone();
        let rctx = RunContext {
            sandbox: ctx.sandbox,
            user: ctx.user,
            workdir: ctx.workdir,
            env,
            prompts,
            limits: Limits {
                wall: ctx.timeout,
                idle: (!ctx.idle_timeout.is_zero()).then_some(ctx.idle_timeout),
                max_tool_calls: ctx.max_tool_calls,
                max_tokens: ctx.max_tokens,
            },
            mcp: ctx.mcp,
            agent_dir: Some(ctx.trial_dir.join("agent")),
        };

        // The trajectory, rewritten while the loop works, and live progress.
        let recorder = Recorder::new();
        let path = ctx
            .trial_dir
            .join("trajectory")
            .join("acp_trajectory.jsonl");
        let writer = TrajectoryWriter::new(&path, redactor.clone()).await.ok();
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
        let background = {
            let recorder = recorder.clone();
            let client = client.clone();
            let events = ctx.events.clone();
            let trial = ctx.trial.to_string();
            tokio::spawn(async move {
                let mut writer = writer;
                loop {
                    let stop = tokio::select! {
                        _ = tokio::time::sleep(TRAJECTORY_EVERY) => false,
                        _ = &mut stop_rx => true,
                    };
                    let (tool_calls, last_tool) = recorder.with(|t| {
                        (
                            t.tool_call_count() as u64,
                            t.tool_calls().last().map(|c| c.display_title().to_string()),
                        )
                    });
                    events.send(Event::Activity {
                        trial: trial.clone(),
                        tool_calls,
                        last_tool,
                        tokens: client.usage().total(),
                    });
                    if let Some(w) = writer.as_mut() {
                        let _ = w.write(recorder.snapshot()).await;
                    }
                    if stop {
                        break;
                    }
                }
            })
        };

        let stop = self.native.run(&client, &recorder, &rctx).await;
        let _ = stop_tx.send(());
        let _ = background.await;

        if matches!(self.native, NativeLoop::Terminus(_)) {
            // The pane log `pipe-pane` kept (best effort).
            let local = ctx.trial_dir.join("agent").join("terminus_2.pane");
            let _ = ctx
                .sandbox
                .download("/logs/agent/terminus_2.pane", &local)
                .await;
        }

        let events = recorder.snapshot();
        out.tool_calls = count_tool_calls(events.iter()) as u64;
        let usage = client.usage();
        if client.calls() > 0 {
            out.usage = engine_usage(&usage);
        }
        let activity = recorder.with(|t| t.activity());
        match stop {
            Stop::Finished | Stop::Refused | Stop::Budget(_) => {}
            Stop::Timeout(_) => {
                out.timed_out = true;
                // As for ACP agents (0.8's 3842ed22): nothing done before the
                // time ran out is not a 0 but a failed integration.
                if activity.tool_calls == 0
                    && activity.message_chunks == 0
                    && activity.thought_chunks == 0
                    && usage.output_tokens == 0
                {
                    out.problem = Some(Problem::new(
                        "R206",
                        Owner::Agent,
                        Phase::Agent,
                        format!(
                            "{} did nothing before its time ran out (no tool call, message, thought, or output token)",
                            self.name
                        ),
                        "read agent/model_calls.jsonl in the trial folder: the model may never have answered",
                    ));
                }
            }
            Stop::Idle { what, after } => {
                let owner = if what.contains("model") {
                    Owner::Model
                } else {
                    Owner::Agent
                };
                out.problem = Some(
                    Problem::new(
                        "R203",
                        owner,
                        Phase::Agent,
                        format!("{}: {what} was silent for {}s", self.name, after.as_secs()),
                        "raise --idle-timeout if the model is slow, or run again",
                    )
                    .transient(),
                );
            }
            Stop::Model(e) => {
                let owner = match e.kind {
                    ModelErrorKind::ModelNotOffered => Owner::You,
                    _ => Owner::Model,
                };
                let p = Problem::new(
                    e.code(),
                    owner,
                    Phase::Agent,
                    e.message.clone(),
                    e.next_step(&provider, &key_label),
                );
                out.problem = Some(if e.transient() { p.transient() } else { p });
            }
            Stop::Sandbox(e) => out.problem = Some(sandbox_problem(Phase::Agent, e)),
            Stop::Setup(why) => {
                let next = match self.native {
                    NativeLoop::Computer(_) => {
                        "read the desktop's logs in /tmp/computer-1 with `bench shell`, or give the sandbox more memory, then run again"
                    }
                    _ => {
                        "install tmux in the task image (or let the agent layer's package manager reach its mirrors), then run again"
                    }
                };
                out.problem = Some(Problem::new("R201", Owner::Agent, Phase::Agent, why, next));
            }
            Stop::Unsupported { why, next } => {
                out.problem = Some(Problem::new("R215", Owner::Task, Phase::Agent, why, next));
            }
            Stop::Harness(why) => {
                out.problem = Some(Problem::new(
                    "R205",
                    Owner::Agent,
                    Phase::Agent,
                    format!("{}: {why}", self.name),
                    "read trajectory/acp_trajectory.jsonl and agent/model_calls.jsonl in the trial folder",
                ));
            }
        }
        let partial = out.timed_out || out.problem.is_some();
        out.trajectory_summary =
            serde_json::to_value(summarize(events.iter(), partial, Some("native")))
                .unwrap_or_default();
        out
    }
}

/// Engine usage from the client's totals. The engine counts cached prompt
/// tokens apart from `input_tokens`, as ACP agents report them.
fn engine_usage(u: &TokenUsage) -> Usage {
    let cached = u.cache_read_tokens + u.cache_write_tokens;
    Usage {
        input_tokens: u.input_tokens.saturating_sub(cached),
        output_tokens: u.output_tokens,
        cache_read_tokens: u.cache_read_tokens,
        cache_write_tokens: u.cache_write_tokens,
        cost_usd: u.cost_usd,
        source: "native".into(),
        // The loop reads the cost from the provider's own answers.
        cost_source: u.cost_usd.map(|_| COST_BY_PROVIDER.into()),
        cost_note: None,
    }
}

/// `env` without the variables whose value is `key`.
fn without_key(
    env: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> std::collections::BTreeMap<String, String> {
    env.iter()
        // As the forwarder does: a value that holds the key anywhere
        // (`Bearer <key>`) stays out too.
        .filter(|(_, v)| key.trim().is_empty() || !v.contains(key.trim()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn host(vars: &[(&str, &str)]) -> HostEnv {
        HostEnv::new(
            vars.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            None,
        )
    }

    #[test]
    fn names_and_aliases_resolve_and_others_do_not() {
        // Not the gateway: its models are pinned to upstreams it lists over
        // the network (tests/native_trial.rs covers that with a fake one).
        let h = host(&[("OPENROUTER_API_KEY", "sk-or-1234567890")]);
        let a = NativeAgent::new("terminus2", Some("openrouter/x/y"), None, &h).unwrap();
        assert_eq!(a.name(), "terminus-2");
        assert_eq!(a.install_layers().len(), 1);
        assert!(a.install_layers()[0].contains("tmux"));
        assert!(!a.needs_network());
        assert!(a.preflight().is_ok());
        let b = NativeAgent::new("builtin", Some("openrouter/x/y"), None, &h).unwrap();
        assert!(b.install_layers().is_empty());
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert!(NativeAgent::new("claude", Some("openrouter/x/y"), None, &h).is_none());
    }

    #[test]
    fn a_missing_key_or_model_fails_preflight_with_a_code() {
        let a = NativeAgent::new("builtin", Some("vercel/x/y"), None, &host(&[])).unwrap();
        let p = a.preflight().unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R211", Owner::You));
        assert!(p.next.contains("AI_GATEWAY_API_KEY"));
        let a = NativeAgent::new("builtin", None, None, &host(&[])).unwrap();
        assert_eq!(a.preflight().unwrap_err().code, "R215");
    }

    #[test]
    fn cached_tokens_are_counted_once() {
        let u = engine_usage(&TokenUsage {
            input_tokens: 100,
            output_tokens: 10,
            cache_read_tokens: 60,
            cache_write_tokens: 0,
            reasoning_tokens: 0,
            cost_usd: Some(0.01),
        });
        assert_eq!(
            (u.input_tokens, u.cache_read_tokens, u.total()),
            (40, 60, 110)
        );
        assert_eq!(u.cost_usd, Some(0.01));
    }
}
