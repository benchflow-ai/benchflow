//! The external agent (`-a external --backend <url>`): an outside service,
//! such as an RL trainer, decides every step.
//!
//! [`ExternalAgent`] joins `benchflow-agent`'s [`external`] loop to a
//! trial. BenchFlow speaks Karotte's backend protocol to the URL, from the
//! host: the run id is the trial's name, the transcript streams to
//! `append_transcript` in Karotte's event format, the model's messages come
//! from `get_message`, and their tool calls run in the sandbox as the agent
//! user. After grading, the reward goes back as a `scoring` event and the
//! run ends with `task_completed` ([`ModelAgent::graded`]).
//!
//! No model key is involved, so nothing goes into the sandbox and the
//! sandbox needs no network. The trial folder gets the shared trajectory
//! (`trajectory/acp_trajectory.jsonl`) and Karotte's transcript as sent
//! (`agent/backend_transcript.json`). A backend that sends nothing for the
//! idle timeout stops the trial with R203, one that cannot be reached or
//! sends something unreadable with R205; both name the backend as the owner.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::external::{
    self, BackendClient, BackendError, BackendOptions, ExternalConfig, ExternalStop, Outcome,
    Session, backend,
};
use benchflow_agent::native::builtin::tools;
use benchflow_agent::native::{Limits, Recorder, RunContext, Stop};
use benchflow_agent::trajectory::{Redactor, TrajectoryWriter, count_tool_calls, summarize};
use serde_json::{Map, json};

use crate::agent::{AgentContext, AgentOutput, ModelAgent, Usage};
use crate::events::Event;
use crate::outcome::{Owner, Phase, Problem};
use crate::record::TrialRecord;
use crate::trial::sandbox_problem;

/// The agent's name.
pub const NAME: &str = "external";
/// An older name still accepted for it, never shown: it names another
/// framework, which BenchFlow's text does not.
const OLD_NAMES: &[&str] = &["karotte-external"];
/// What `bench.lock` records as the agent's install: nothing is installed,
/// so only the protocol counts.
pub(crate) const FINGERPRINT: &str = "external:backend-protocol";
/// The same, as earlier builds recorded it; `bench run --diff` treats the
/// two as one install.
const OLD_FINGERPRINT: &str = "external:karotte-backend-protocol";

/// `fingerprint` as this build records it: an earlier build's name for the
/// same install becomes today's.
pub(crate) fn current_fingerprint(fingerprint: &str) -> &str {
    if fingerprint == OLD_FINGERPRINT {
        FINGERPRINT
    } else {
        fingerprint
    }
}
/// How often the trajectory file is rewritten while the loop works.
const TRAJECTORY_EVERY: Duration = Duration::from_secs(2);
/// Where Karotte's transcript is kept in the trial folder.
const TRANSCRIPT: &str = "agent/backend_transcript.json";

/// Whether `name` means the external agent.
pub fn is_external(name: &str) -> bool {
    name == NAME || OLD_NAMES.contains(&name)
}

/// A run whose steps a backend decides.
pub struct ExternalAgent {
    backend: Result<String, Problem>,
    model: Option<String>,
    effort: Option<String>,
    token_path: Option<PathBuf>,
    options: BackendOptions,
    bash_timeout: Duration,
    /// Runs waiting for their grade, by trial name.
    open: Mutex<HashMap<String, (Session, PathBuf)>>,
}

impl ExternalAgent {
    /// The external agent on the backend at `backend`. `model` is only a
    /// label sent to the backend (`task_started.model`).
    pub fn new(
        backend: Option<&str>,
        model: Option<&str>,
        effort: Option<&str>,
        host: &HostEnv,
    ) -> Self {
        let backend = match backend.map(str::trim).filter(|b| !b.is_empty()) {
            None => Err(Problem::new(
                "R215",
                Owner::You,
                Phase::Setup,
                "the external agent takes its steps from a backend, and no --backend was given",
                "pass --backend <url> with the address of a service that speaks the backend protocol (GET /api/internal/get_message, ...)",
            )),
            Some(url) => BackendClient::new(url, "check", None, BackendOptions::default())
                .map(|c| c.base_url().to_string())
                .map_err(|why| {
                    Problem::new(
                        "R215",
                        Owner::You,
                        Phase::Setup,
                        format!("--backend: {why}"),
                        "pass --backend http://<host>:<port> (the backend's base URL, without /api)",
                    )
                }),
        };
        Self {
            backend,
            model: model.map(str::to_string),
            effort: effort.map(str::to_string),
            token_path: host
                .get(backend::TOKEN_PATH_ENV)
                .or_else(|| host.get(backend::TOKEN_PATH_ENV_OLD))
                .map(|(v, _)| PathBuf::from(v)),
            options: BackendOptions::default(),
            bash_timeout: Duration::from_secs(600),
            open: Mutex::new(HashMap::new()),
        }
    }

    /// Change how the backend client waits and retries.
    pub fn backend_options(mut self, options: BackendOptions) -> Self {
        self.options = options;
        self
    }

    /// The backend's base URL, when one was given and is valid.
    pub fn backend(&self) -> Option<&str> {
        self.backend.as_deref().ok()
    }
}

/// Keep Karotte's transcript in the trial folder (best effort).
fn keep_transcript(trial_dir: &Path, session: &Session) {
    let path = trial_dir.join(TRANSCRIPT);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let text = serde_json::to_string_pretty(&session.transcript()).unwrap_or_default();
    let _ = std::fs::write(path, text);
}

/// A backend failure as a trial problem; the backend is the agent here.
fn backend_problem(e: &BackendError) -> Problem {
    let p = Problem::new(
        "R205",
        Owner::Agent,
        Phase::Agent,
        format!("external: {e}"),
        e.next_step(),
    );
    if e.transient() { p.transient() } else { p }
}

#[async_trait]
impl ModelAgent for ExternalAgent {
    fn name(&self) -> &str {
        NAME
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The tools run through the sandbox API; nothing is installed.
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }

    fn fingerprint(&self) -> String {
        FINGERPRINT.into()
    }

    fn preflight(&self) -> Result<(), Problem> {
        self.backend.as_ref().map(|_| ()).map_err(Clone::clone)
    }

    /// The backend is called from the host.
    fn needs_network(&self) -> bool {
        false
    }

    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput {
        let mut out = AgentOutput {
            usage: Usage {
                source: "unavailable".into(),
                ..Usage::default()
            },
            ..AgentOutput::default()
        };
        let url = match &self.backend {
            Ok(url) => url.clone(),
            Err(p) => {
                out.problem = Some(p.clone());
                return out;
            }
        };
        out.login = Some(format!("backend {url}"));
        // A backend down for longer than the idle timeout is as silent as
        // one that never answers.
        let mut options = self.options.clone();
        if !ctx.idle_timeout.is_zero() {
            options.retry_budget = options.retry_budget.min(ctx.idle_timeout);
        }
        let client = match BackendClient::new(&url, ctx.trial, self.token_path.clone(), options) {
            Ok(c) => c,
            Err(why) => {
                out.problem = Some(Problem::new(
                    "R215",
                    Owner::You,
                    Phase::Agent,
                    format!("--backend: {why}"),
                    "pass --backend http://<host>:<port>",
                ));
                return out;
            }
        };
        ctx.events.log(
            ctx.trial,
            format!("external: run id {} on the backend at {url}", ctx.trial),
        );
        let mut session = Session::new(client);

        let mut env = ctx.env.clone();
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
        let mut metadata = Map::new();
        metadata.insert(
            "benchflow".into(),
            json!({"trial": ctx.trial, "workdir": ctx.workdir, "user": ctx.user,
                   "version": env!("CARGO_PKG_VERSION")}),
        );
        let config = ExternalConfig {
            task_id: ctx
                .trial
                .rsplit_once("__")
                .map_or(ctx.trial, |(task, _)| task)
                .to_string(),
            model: self.model.clone(),
            reasoning_effort: self.effort.clone(),
            bash_timeout: self.bash_timeout,
            metadata,
        };

        // The trajectory, rewritten while the loop works, and live progress.
        let recorder = Recorder::new();
        let path = ctx
            .trial_dir
            .join("trajectory")
            .join("acp_trajectory.jsonl");
        let writer =
            TrajectoryWriter::new(&path, Redactor::new(crate::agent::env_secrets(&ctx.env)))
                .await
                .ok();
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
        let background = {
            let recorder = recorder.clone();
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
                        tokens: 0,
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

        let specs = tools::specs(self.bash_timeout);
        let system = external::system(ctx.workdir);
        let start = session.start(&config, rctx.prompts.len(), &specs, &system);
        let result = match rctx.limits.idle {
            Some(idle) => match tokio::time::timeout(idle, start).await {
                Ok(r) => r.map_err(ExternalStop::Backend),
                Err(_) => Err(ExternalStop::Loop(Stop::Idle {
                    what: "the backend".into(),
                    after: idle,
                })),
            },
            None => start.await.map_err(ExternalStop::Backend),
        };
        let result = match result {
            Ok(()) => external::run(&mut session, &recorder, &rctx, &config).await,
            Err(e) => Err(e),
        };
        match &result {
            Ok(()) => {}
            Err(ExternalStop::Loop(stop)) => stop.record(&recorder),
            Err(ExternalStop::Backend(e)) => {
                recorder.notice("error", &format!("backend failed: {e}"), None)
            }
        }
        recorder.finish();
        let _ = stop_tx.send(());
        let _ = background.await;

        let events = recorder.snapshot();
        out.tool_calls = count_tool_calls(events.iter()) as u64;
        let activity = recorder.with(|t| t.activity());
        let problem = match result {
            Ok(()) => None,
            Err(ExternalStop::Backend(e)) => Some(backend_problem(&e)),
            Err(ExternalStop::Loop(stop)) => match stop {
                Stop::Finished | Stop::Refused | Stop::Budget(_) => None,
                Stop::Timeout(_) => {
                    out.timed_out = true;
                    (activity.tool_calls == 0 && activity.message_chunks == 0).then(|| {
                        Problem::new(
                            "R206",
                            Owner::Agent,
                            Phase::Agent,
                            format!("external: the backend at {url} sent no message before the time ran out"),
                            "check that the backend serves GET /api/internal/get_message for this run id (the trial's name)",
                        )
                    })
                }
                Stop::Idle { after, .. } => Some(
                    Problem::new(
                        "R203",
                        Owner::Agent,
                        Phase::Agent,
                        format!(
                            "external: the backend at {url} sent nothing for {}s (the idle timeout)",
                            after.as_secs()
                        ),
                        "check the backend (it must answer get_message for this run id), or raise --idle-timeout if it is slow",
                    )
                    .transient(),
                ),
                Stop::Sandbox(e) => Some(sandbox_problem(Phase::Agent, e)),
                Stop::Model(e) => Some(Problem::new(
                    "R205",
                    Owner::Agent,
                    Phase::Agent,
                    format!("external: {e}"),
                    "report this as a BenchFlow bug: the external agent calls no model",
                )),
                Stop::Unsupported { why, next } => Some(Problem::new(
                    "R215",
                    Owner::Task,
                    Phase::Agent,
                    format!("external: {why}"),
                    next,
                )),
                Stop::Setup(why) | Stop::Harness(why) => Some(Problem::new(
                    "R205",
                    Owner::Agent,
                    Phase::Agent,
                    format!("external: {why}"),
                    "read agent/backend_transcript.json in the trial folder",
                )),
            },
        };
        match &problem {
            // Graded later (a run that timed out too): the backend hears the
            // score from `graded`.
            None => {
                keep_transcript(&ctx.trial_dir, &session);
                if let Ok(mut open) = self.open.lock() {
                    open.insert(ctx.trial.to_string(), (session, ctx.trial_dir.clone()));
                }
            }
            Some(p) => {
                // Tell the backend the run is over, unless it is unreachable.
                if !(p.code == "R205" && p.transient) {
                    let tell = session.fail(&p.code, &p.message);
                    let _ = tokio::time::timeout(Duration::from_secs(30), tell).await;
                }
                keep_transcript(&ctx.trial_dir, &session);
            }
        }
        out.problem = problem;
        let partial = out.timed_out || out.problem.is_some();
        out.trajectory_summary =
            serde_json::to_value(summarize(events.iter(), partial, Some("external")))
                .unwrap_or_default();
        out
    }

    async fn graded(&self, trial: &str, rec: &TrialRecord) {
        let Some((mut session, dir)) = self.open.lock().ok().and_then(|mut o| o.remove(trial))
        else {
            return;
        };
        let status = match rec.passed {
            Some(true) => "passed",
            Some(false) => "failed",
            None => "error",
        };
        let mut metadata = Map::new();
        if let Some(rewards) = &rec.rewards {
            metadata.insert("rewards".into(), json!(rewards));
        }
        metadata.insert("benchflow_status".into(), json!(rec.status));
        if let Some(p) = &rec.problem {
            metadata.insert(
                "problem".into(),
                json!({"code": p.code, "owner": p.owner.label(), "message": p.message}),
            );
        }
        let outcome = Outcome {
            status,
            score: rec.reward,
            metadata,
        };
        let finish = session.finish(&outcome);
        match tokio::time::timeout(Duration::from_secs(60), finish).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!("external: could not report the outcome of {trial}: {e}"),
            Err(_) => tracing::warn!(
                "external: the backend did not take the outcome of {trial} within 60s"
            ),
        }
        keep_transcript(&dir, &session);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn a_missing_or_bad_backend_fails_preflight_with_a_fix() {
        let host = HostEnv::new(BTreeMap::new(), None);
        let p = ExternalAgent::new(None, None, None, &host)
            .preflight()
            .unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R215", Owner::You));
        assert!(p.next.contains("--backend"));
        let p = ExternalAgent::new(Some("ftp://x"), None, None, &host)
            .preflight()
            .unwrap_err();
        assert!(p.message.contains("--backend"));
        let a = ExternalAgent::new(Some("http://127.0.0.1:8000/"), None, None, &host);
        assert!(a.preflight().is_ok());
        assert_eq!(a.backend(), Some("http://127.0.0.1:8000"));
        assert!(!a.needs_network());
        assert!(a.install_layers().is_empty());
        assert!(is_external("karotte-external") && is_external("external"));
    }

    /// The token file's variable keeps working under its older name, and
    /// the new name wins when both are set.
    #[test]
    fn the_token_path_variable_is_read_under_either_name() {
        let path = |vars: &[(&str, &str)]| {
            let host = HostEnv::new(
                vars.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
                None,
            );
            ExternalAgent::new(Some("http://127.0.0.1:8000"), None, None, &host).token_path
        };
        assert_eq!(path(&[]), None);
        assert_eq!(
            path(&[("KAROTTE_BACKEND_TOKEN_PATH", "/old")]),
            Some(PathBuf::from("/old"))
        );
        assert_eq!(
            path(&[
                ("KAROTTE_BACKEND_TOKEN_PATH", "/old"),
                ("BENCHFLOW_BACKEND_TOKEN_PATH", "/new"),
            ]),
            Some(PathBuf::from("/new"))
        );
    }
}
