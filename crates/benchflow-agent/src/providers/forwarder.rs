//! Starting and stopping the forwarder in a sandbox.
//!
//! [`Forwarder::start`] runs as root, before the agent:
//!
//! 1. it makes [`FORWARDER_DIR`] (root's, mode 700) and checks for Node.js
//!    ([`NODE`], installed with every npm agent);
//! 2. it writes the forwarder's code, its config (no secret) and the key,
//!    each root's with mode 600, through the sandbox's file transfer, so the
//!    key is on no command line;
//! 3. it starts `node forwarder.js config.json` as root and waits for the
//!    line that gives its port. The forwarder deletes the key file as soon
//!    as it has read it.
//!
//! The agent then gets `http://127.0.0.1:<port>/<protocol>` and a
//! placeholder key ([`super::ProviderRoute::agent_env`]).
//! [`Forwarder::stop`] kills it and hands back its log: one line per request,
//! without the key.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use benchflow_sandbox::{Exec, Sandbox, SandboxError};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use super::key::ProviderKey;
use super::route::ProviderRoute;
use crate::trajectory::Redactor;

/// The forwarder's code: plain Node.js, no packages.
pub const FORWARDER_JS: &str = include_str!("../../forwarder/forwarder.js");
/// Where the forwarder's files go in the sandbox: root's, mode 700.
pub const FORWARDER_DIR: &str = "/opt/benchflow/forwarder";
/// The Node.js the forwarder runs on: BenchFlow's own, installed with every
/// npm agent.
pub const NODE: &str = "/opt/benchflow/node/bin/node";

/// The image layer that installs BenchFlow's Node.js for the forwarder, for
/// an agent not installed with npm (npm agents bring it). Safe to repeat.
pub fn node_layer() -> String {
    format!(
        "{} && chmod -R a+rX /opt/benchflow/node",
        crate::manifest::install::NODE_BOOTSTRAP
    )
}

/// How long the forwarder may take to say it is listening.
const READY_TIMEOUT: Duration = Duration::from_secs(30);
/// How much of the forwarder's log is kept.
const LOG_KEEP: usize = 512 * 1024;

/// Why the forwarder did not start.
#[derive(Debug)]
pub enum ForwarderError {
    /// A sandbox operation failed.
    Sandbox {
        step: &'static str,
        error: SandboxError,
    },
    /// The image has no Node.js for the forwarder.
    NoNode,
    /// The forwarder exited, or did not say it was listening in time.
    NotReady { detail: String },
}

impl ForwarderError {
    /// What to do about it.
    pub fn next_step(&self) -> &'static str {
        match self {
            Self::Sandbox { .. } => {
                "retry; if it persists, `bench doctor` checks the sandbox backend"
            }
            Self::NoNode => {
                "the agent's image layer installs it for the forwarder; image.log in the trial folder shows why it did not"
            }
            Self::NotReady { .. } => {
                "unless the sandbox ran out of memory, this is a BenchFlow bug: report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder"
            }
        }
    }
}

impl std::fmt::Display for ForwarderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sandbox { step, error } => {
                write!(
                    f,
                    "the provider forwarder could not be set up ({step}): {error}"
                )
            }
            Self::NoNode => write!(
                f,
                "the provider forwarder needs Node.js at {NODE}, and the image has none"
            ),
            Self::NotReady { detail } => {
                write!(f, "the provider forwarder did not start")?;
                if !detail.trim().is_empty() {
                    write!(f, ": {}", detail.trim())?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ForwarderError {}

/// What the forwarder did, once stopped.
#[derive(Debug, Clone, Default)]
pub struct ForwarderReport {
    /// Its log: one JSON line per request, without the key.
    pub log: String,
    /// Requests it answered.
    pub requests: usize,
    /// It exited before it was stopped (with its exit code when known): the
    /// agent lost its model then.
    pub exited_early: Option<Option<i32>>,
}

impl ForwarderReport {
    /// When the provider refused every model request the forwarder passed
    /// on, with an authentication error: the status (401 or 403) and how
    /// many. Probes (`HEAD`, `GET`) and the forwarder's own refusals (a
    /// model the run may not use) do not count.
    pub fn rejected_key(&self) -> Option<(u16, usize)> {
        let mut rejected: Option<(u16, usize)> = None;
        for line in self.log.lines() {
            let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let model_call =
                entry.get("method").and_then(serde_json::Value::as_str) == Some("POST");
            let own_refusal = entry.get("note").is_some();
            let Some(status) = entry.get("status").and_then(serde_json::Value::as_u64) else {
                continue;
            };
            if !model_call || own_refusal {
                continue;
            }
            if status != 401 && status != 403 {
                return None;
            }
            let (_, n) = rejected.get_or_insert((status as u16, 0));
            *n += 1;
        }
        rejected
    }

    /// The token counts and cost the provider reported for the model calls
    /// it answered (`POST`s with a 2xx status; the forwarder's own refusals
    /// and failed calls are not billed).
    pub fn spend(&self) -> ProviderSpend {
        let mut spend = ProviderSpend::default();
        for line in self.log.lines() {
            let Ok(entry) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let model_call = entry.get("method").and_then(Value::as_str) == Some("POST")
                && entry.get("note").is_none();
            let ok = entry
                .get("status")
                .and_then(Value::as_u64)
                .is_some_and(|s| (200..300).contains(&s));
            if !model_call || !ok {
                continue;
            }
            // A token count request (`count_tokens`) bills nothing.
            if entry
                .get("path")
                .and_then(Value::as_str)
                .is_some_and(|p| p.contains("count_tokens"))
            {
                continue;
            }
            spend.calls += 1;
            if let Some(usage) = entry.get("usage") {
                let n = |k: &str| usage.get(k).and_then(Value::as_f64).unwrap_or(0.0) as u64;
                spend.with_usage += 1;
                spend.input_tokens += n("input");
                spend.cache_read_tokens += n("cache_read");
                spend.cache_write_tokens += n("cache_write");
                spend.output_tokens += n("output");
            }
            if let Some(cost) = entry
                .get("cost")
                .and_then(Value::as_f64)
                .filter(|c| c.is_finite() && *c >= 0.0)
            {
                spend.priced += 1;
                spend.priced_usd += cost;
            }
        }
        spend
    }
}

/// What the provider said a run's model calls used and cost, summed from
/// the forwarder's log: the token counts each answer reported, and the cost
/// the provider charged for each (the Vercel AI Gateway and OpenRouter put
/// it in every answer; a provider's own API does not).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProviderSpend {
    /// Prompt tokens not read from or written to a cache.
    pub input_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub output_tokens: u64,
    /// Model calls the provider answered (a `POST` with a 2xx status).
    pub calls: usize,
    /// Of those, the ones whose answer reported token counts.
    pub with_usage: usize,
    /// Of those, the ones whose answer said what it cost.
    pub priced: usize,
    /// The cost of the priced calls, in US dollars.
    pub priced_usd: f64,
}

impl ProviderSpend {
    /// The run's cost as the provider billed it: known only when every
    /// model call it answered said what it cost.
    pub fn cost_usd(&self) -> Option<f64> {
        (self.calls > 0 && self.priced == self.calls).then_some(self.priced_usd)
    }

    /// Why [`Self::cost_usd`] is unknown, in a sentence for the record.
    pub fn unknown_cost(&self) -> Option<String> {
        if self.cost_usd().is_some() {
            return None;
        }
        Some(if self.calls == 0 {
            "no model call through BenchFlow's forwarder was answered".to_string()
        } else if self.priced == 0 {
            "the provider's answers did not say what they cost, so the cost is unknown".to_string()
        } else {
            format!(
                "the provider said what {} of {} model calls cost (${:.6}); the rest did not, so the run's cost is unknown",
                self.priced, self.calls, self.priced_usd
            )
        })
    }
}

/// A running forwarder.
pub struct Forwarder {
    port: u16,
    prefix: String,
    kill: mpsc::Sender<oneshot::Sender<()>>,
    exit: watch::Receiver<Option<Option<i32>>>,
    log: Arc<Mutex<String>>,
    tasks: Vec<JoinHandle<()>>,
    redactor: Redactor,
}

impl std::fmt::Debug for Forwarder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Forwarder")
            .field("port", &self.port)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl Drop for Forwarder {
    fn drop(&mut self) {
        // Without stop(): ask for a kill and stop reading.
        let (done, _) = oneshot::channel();
        let _ = self.kill.try_send(done);
        for task in &self.tasks {
            task.abort();
        }
    }
}

fn sandbox_error(step: &'static str) -> impl FnOnce(SandboxError) -> ForwarderError {
    move |error| ForwarderError::Sandbox { step, error }
}

impl Forwarder {
    /// Start the forwarder for `route` in `sandbox`, holding `key`. Call it
    /// before the agent starts.
    pub async fn start(
        sandbox: &dyn Sandbox,
        route: &ProviderRoute,
        key: &ProviderKey,
    ) -> Result<Self, ForwarderError> {
        let dir = FORWARDER_DIR;
        let prepare = format!(
            "test -x {NODE} || exit 3; umask 077; rm -rf {dir} && mkdir -p {dir} && chown root:root {dir} && chmod 700 {dir}"
        );
        let out = sandbox
            .exec(Exec::new(prepare).system().timeout(Duration::from_secs(60)))
            .await
            .map_err(sandbox_error("making its folder"))?;
        match out.exit_code {
            0 if !out.timed_out => {}
            3 => return Err(ForwarderError::NoNode),
            _ => {
                return Err(ForwarderError::NotReady {
                    detail: format!("could not make {dir}: {}", out.tail(500)),
                });
            }
        }
        let key_file = format!("{dir}/key");
        let config = route.forwarder_config(&key_file);
        for (name, contents, step) in [
            (
                "forwarder.js",
                FORWARDER_JS.as_bytes().to_vec(),
                "writing its code",
            ),
            (
                "config.json",
                config.to_string().into_bytes(),
                "writing its config",
            ),
            (
                "key",
                key.value().as_bytes().to_vec(),
                "writing the key file",
            ),
        ] {
            sandbox
                .write_file(&format!("{dir}/{name}"), &contents, 0o600)
                .await
                .map_err(sandbox_error(step))?;
        }
        let command = format!(
            "chown root:root {dir}/forwarder.js {dir}/config.json {dir}/key && chmod 600 {dir}/forwarder.js {dir}/config.json {dir}/key && exec {NODE} {dir}/forwarder.js {dir}/config.json"
        );
        let process = sandbox
            .spawn(Exec::new(command).system().cwd(dir.to_string()))
            .await
            .map_err(sandbox_error("starting it"))?;

        let redactor = Redactor::new([key.value().to_string()]);
        let log = Arc::new(Mutex::new(String::new()));
        let (stderr_done_tx, stderr_done) = oneshot::channel();
        let mut tasks = vec![tokio::spawn(keep_log(
            process.stderr,
            log.clone(),
            stderr_done_tx,
        ))];
        let (kill, kills) = mpsc::channel(1);
        let (exit_tx, exit) = watch::channel(None);
        tokio::spawn(watch_exit(process.handle, kills, exit_tx));

        let mut stdout = BufReader::new(process.stdout).lines();
        let ready = tokio::time::timeout(READY_TIMEOUT, async {
            while let Ok(Some(line)) = stdout.next_line().await {
                if let Some(port) = ready_port(&line) {
                    return Some(port);
                }
            }
            None
        })
        .await;
        let mut forwarder = Self {
            port: 0,
            prefix: route.prefix(),
            kill,
            exit,
            log,
            tasks: Vec::new(),
            redactor,
        };
        match ready {
            Ok(Some(port)) => {
                tasks.push(tokio::spawn(async move {
                    while let Ok(Some(_)) = stdout.next_line().await {}
                }));
                forwarder.port = port;
                forwarder.tasks = tasks;
                Ok(forwarder)
            }
            outcome => {
                // Its last words are on stderr; give them a moment to arrive.
                let _ = tokio::time::timeout(Duration::from_secs(3), stderr_done).await;
                let mut detail = forwarder.log_text();
                if outcome.is_err() {
                    detail = format!(
                        "it did not say it was listening within {}s. {detail}",
                        READY_TIMEOUT.as_secs()
                    );
                }
                forwarder.tasks = tasks;
                let _ = forwarder.stop_process().await;
                Err(ForwarderError::NotReady { detail })
            }
        }
    }

    /// The forwarder's port on the sandbox's loopback.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// What the agent uses as the endpoint's base URL:
    /// `http://127.0.0.1:<port>/<protocol>`.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}{}", self.port, self.prefix)
    }

    /// Whether it has exited on its own (its exit code, when known).
    pub fn exited(&self) -> Option<Option<i32>> {
        *self.exit.borrow()
    }

    fn log_text(&self) -> String {
        let log = self.log.lock().unwrap_or_else(|p| p.into_inner()).clone();
        self.redactor.redact(&log).into_owned()
    }

    async fn stop_process(&mut self) -> Option<Option<i32>> {
        let early = self.exited();
        if early.is_none() {
            let (done, killed) = oneshot::channel();
            if self.kill.send(done).await.is_ok() {
                let _ = tokio::time::timeout(Duration::from_secs(10), killed).await;
            }
        }
        early
    }

    /// Kill the forwarder and hand back what it did.
    pub async fn stop(mut self) -> ForwarderReport {
        // Let the log reader take the last lines, then keep the log as it is:
        // what the sandbox prints about the kill is not the forwarder's.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let log = self.log_text();
        let exited_early = self.stop_process().await;
        let requests = log
            .lines()
            .filter(|line| {
                serde_json::from_str::<Value>(line)
                    .is_ok_and(|v| v.get("status").is_some() && v.get("method").is_some())
            })
            .count();
        ForwarderReport {
            log,
            requests,
            exited_early,
        }
    }
}

/// The port in the forwarder's ready line.
fn ready_port(line: &str) -> Option<u16> {
    let value: Value = serde_json::from_str(line.trim()).ok()?;
    if value.get("benchflow_forwarder")?.as_str()? != "ready" {
        return None;
    }
    u16::try_from(value.get("port")?.as_u64()?)
        .ok()
        .filter(|p| *p != 0)
}

/// Keep the forwarder's stderr (its request log), bounded.
async fn keep_log(
    stderr: std::pin::Pin<Box<dyn AsyncRead + Send>>,
    log: Arc<Mutex<String>>,
    done: oneshot::Sender<()>,
) {
    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let mut log = log.lock().unwrap_or_else(|p| p.into_inner());
        log.push_str(&line);
        log.push('\n');
        if log.len() > LOG_KEEP {
            let mut cut = log.len() - LOG_KEEP;
            while !log.is_char_boundary(cut) {
                cut += 1;
            }
            log.drain(..cut);
        }
    }
    let _ = done.send(());
}

/// Own the process handle: publish an exit, kill on request.
async fn watch_exit(
    mut handle: Box<dyn benchflow_sandbox::ProcessHandle>,
    mut kills: mpsc::Receiver<oneshot::Sender<()>>,
    exit: watch::Sender<Option<Option<i32>>>,
) {
    tokio::select! {
        code = handle.wait() => {
            exit.send_replace(Some(code.ok()));
        }
        Some(done) = kills.recv() => {
            if let Err(e) = handle.kill().await {
                tracing::warn!("could not stop the provider forwarder: {e}");
            }
            let _ = done.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(lines: &[&str]) -> ForwarderReport {
        ForwarderReport {
            log: lines.join("\n"),
            ..ForwarderReport::default()
        }
    }

    #[test]
    fn a_key_rejected_on_every_model_request_is_spotted() {
        let probe = r#"{"method":"HEAD","path":"/anthropic/api/hello","status":200}"#;
        let rejected = r#"{"method":"POST","path":"/anthropic/v1/messages","status":401}"#;
        let refused_model = r#"{"method":"POST","path":"/v1/messages","status":403,"note":"this run may use only m"}"#;
        let ok = r#"{"method":"POST","path":"/anthropic/v1/messages","status":200}"#;
        assert_eq!(
            report(&[probe, rejected, refused_model, rejected]).rejected_key(),
            Some((401, 2))
        );
        // One answer is enough to say the key works; no model request says nothing.
        assert_eq!(report(&[rejected, ok]).rejected_key(), None);
        assert_eq!(report(&[probe, refused_model]).rejected_key(), None);
        assert_eq!(report(&["not json", ""]).rejected_key(), None);
    }

    #[test]
    fn the_ready_line_gives_the_port() {
        assert_eq!(
            ready_port(r#"{"benchflow_forwarder": "ready", "port": 41234}"#),
            Some(41234)
        );
        assert_eq!(
            ready_port(r#"{"benchflow_forwarder": "ready", "port": 0}"#),
            None
        );
        assert_eq!(ready_port(r#"{"port": 41234}"#), None);
        assert_eq!(ready_port("listening on 41234"), None);
    }
}
