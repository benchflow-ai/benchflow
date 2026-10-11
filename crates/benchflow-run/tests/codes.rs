//! Every run failure code `bench explain` lists, triggered through the real
//! engine: a scripted in-memory sandbox stands in for what a sandbox does
//! (an image that fails to build, a container that never starts, a verifier
//! that writes no reward), and a scripted ACP agent, driven by the real ACP
//! client, for what agents do (exit, go silent, crash, answer with an
//! error). Each case checks the code and whose problem it is.
//!
//! `tests/codes/run.txt` keeps what a user reads for each case: the code,
//! whose problem it is, the message and the next step, as `bench run`'s
//! report shows them. A change in wording shows up in review; regenerate it
//! with `UPDATE_GOLDEN=1 cargo test -p benchflow-run --test codes`. The test
//! also fails when `bench explain` lists a code no case here triggers.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_agent::AgentManifest;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::manifest::ManifestSource;
use benchflow_run::acp_agent::AcpAgent;
use benchflow_run::agent::{AgentContext, AgentOutput, Attempt, ModelAgent};
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::outcome::{EXPLANATIONS, Owner, Problem};
use benchflow_run::record::TrialRecord;
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::services::{ServiceHost, ServiceSpec};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, ProcessHandle, Provider,
    ProviderStatus, Sandbox, SandboxError, SandboxSpec,
};
use benchflow_task::TaskPackage;
use benchflow_task::package::*;
use benchflow_task::values::Network;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// What the scripted sandbox does wrong.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Fault {
    #[default]
    None,
    /// The task's Dockerfile fails to build.
    TaskImage,
    /// The backend fails to prepare the image for a reason outside the task.
    ImageBackend,
    /// The image builds, but not with the agent's install layers on top.
    AgentLayers,
    /// The container fails to start.
    Start,
    /// The backend cannot give the task what it asks for.
    NoGpus,
    /// The image has no useradd or adduser.
    Useradd,
    /// The readiness check never passes.
    NotReady,
    /// The verifier's files cannot be read on this machine.
    LocalRead,
    /// The connection to the sandbox drops before grading.
    Transport,
    /// Agent processes survive the kill before grading.
    Survivors,
    /// A program on the image's PATH answers in place of the shell.
    Shadowed,
    /// The agent's turn changed /etc/ld.so.preload, and it cannot be put back.
    PreloadStuck,
    /// The agent took ownership of BenchFlow's state folder before grading.
    StateTampered,
    /// The image's PATH holds a folder others can write.
    UncleanVerifierEnv,
    /// A hook before grading exits with an error.
    HookFails,
    /// A service beside the sandbox exits as it starts.
    ServiceFails,
    /// The image has no iptables for a blocklist's firewall.
    NoIptables,
    /// The firewall self-test reaches a blocked address.
    Leaky,
    /// The verifier writes no reward.
    NoReward,
    /// The verifier writes a reward that is not a number.
    BadReward,
    /// The verifier runs past its timeout.
    VerifierTimeout,
    /// The verifier exits with an error and writes no reward.
    VerifierFails,
    /// The connection to the sandbox drops while it sets up.
    SetupGone,
    /// The sandbox hangs while it sets up (for a Ctrl-C there).
    SetupHangs,
    /// The connection to the sandbox drops while the verifier runs.
    VerifierGone,
    /// The sandbox is killed under the verifier (exit 137) and is gone.
    VerifierKilled,
    /// The verifier runs, but its folder cannot be fetched.
    DownloadFails,
    /// The verifier hangs (for a Ctrl-C while grading).
    VerifierHangs,
    /// The verifier fails on a full disk and writes no reward.
    DiskFull,
    /// The out-of-memory killer ends the agent mid-prompt.
    AgentOom,
    /// The sandbox cannot start the agent's process: it is gone.
    AgentSpawnGone,
    /// The sandbox goes away while the agent works (the agent's process
    /// ends with it).
    AgentGone,
}

/// What the scripted ACP agent does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Act {
    #[default]
    Solve,
    /// Exits before answering `initialize`.
    ExitAtStart,
    /// Never answers the prompt.
    Silent,
    /// Exits in the middle of the prompt.
    Crash,
    /// Answers the prompt with a JSON-RPC error.
    Error(&'static str),
    /// Answers the prompt with codex-acp's usage-limit error.
    Spent,
    /// Ends its turn with this as its only message, and no tool call.
    Reply(&'static str),
    /// Refuses `session/new` until signed in with the vendor's account.
    NeedsSignIn,
}

#[derive(Default)]
struct World {
    solved: bool,
    /// The sandbox is gone: every later command fails.
    gone: bool,
}

fn gone() -> SandboxError {
    SandboxError::Transport("the sandbox's API stopped answering (connection reset by peer)".into())
}

struct FakeSandbox {
    world: Arc<Mutex<World>>,
    fault: Fault,
    act: Act,
}

fn output(code: i32, stdout: &str, stderr: &str) -> ExecOutput {
    ExecOutput {
        exit_code: code,
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
        timed_out: false,
        duration: Duration::from_millis(1),
    }
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-codes"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        let c = exec.command;
        let silent = self.fault == Fault::Shadowed;
        if self.world.lock().unwrap().gone {
            return Err(gone());
        }
        let verifier = c.contains("/verifier/test.sh") || c.contains("/tests/test.sh");
        match self.fault {
            Fault::SetupGone if c.contains("useradd") => return Err(gone()),
            Fault::SetupHangs if c.contains("useradd") => {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Fault::VerifierGone if verifier => return Err(gone()),
            Fault::VerifierHangs if verifier => {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Fault::DiskFull if verifier => {
                return Ok(output(
                    1,
                    "",
                    "OSError: [Errno 28] No space left on device: '/logs/verifier/reward.txt'\n",
                ));
            }
            Fault::VerifierKilled if verifier => {
                self.world.lock().unwrap().gone = true;
                return Ok(output(137, "", "Killed\n"));
            }
            _ => {}
        }
        Ok(if c.contains("BF oom-kill") {
            output(
                0,
                if self.fault == Fault::AgentOom {
                    "BF oom-kill 1\n"
                } else {
                    "BF oom-kill 0\n"
                },
                "",
            )
        } else if c.contains("jump_new") {
            // The network policy's firewall install.
            if self.fault == Fault::NoIptables {
                output(0, "BF fw-error no-iptables\n", "")
            } else {
                output(
                    0,
                    "BF fw /usr/sbin/iptables v6=absent fw6=none reject=REJECT\nBF fw-sum 42\n",
                    "",
                )
            }
        } else if c.contains("BF canary") {
            // The firewall self-test, as the agent user.
            let verdict = if self.fault == Fault::Leaky {
                "reached"
            } else {
                "refused"
            };
            output(
                0,
                &format!("BF canary block {verdict} 192.0.2.50:443\nBF tamper tried\n"),
                "",
            )
        } else if c.contains("fingerprint") {
            output(0, "BF fw-sum 42\n", "")
        } else if c.contains("useradd") {
            if self.fault == Fault::Useradd {
                output(
                    1,
                    "",
                    "sh: 1: useradd: not found\nsh: 1: adduser: not found\n",
                )
            } else {
                output(0, "BF user agent 1000 /home/agent\n", "")
            }
        } else if c.contains("check-ready") {
            if self.fault == Fault::NotReady {
                output(
                    1,
                    "",
                    "curl: (7) Failed to connect to localhost port 8080\n",
                )
            } else {
                output(0, "", "")
            }
        } else if c.contains("kill -9 -1") {
            match self.fault {
                Fault::Transport => {
                    return Err(SandboxError::Transport(
                        "the sandbox's API stopped answering (connection reset by peer)".into(),
                    ));
                }
                Fault::Survivors => output(1, "BF survivors 4242 sleep\n", ""),
                _ if silent => output(0, "", ""),
                _ => output(0, "BF killed 1\n", ""),
            }
        } else if c.contains("ld.so.preload") {
            match self.fault {
                Fault::PreloadStuck => output(1, "BF preload unrestorable\n", ""),
                _ => output(0, "BF preload absent\n", ""),
            }
        } else if c.contains("BF venv-done") {
            match self.fault {
                Fault::UncleanVerifierEnv => output(
                    0,
                    "BF venv-safepath 1\nBF venv-path-bad /usr/local/bin drwxrwxrwx\nBF venv-done\n",
                    "",
                ),
                _ => output(
                    0,
                    "BF venv-safepath 1\nBF venv-path /usr/local/bin:/usr/bin:/bin\nBF venv-done\n",
                    "",
                ),
            }
        } else if c.contains("/root/snapshot/status.txt") && self.fault == Fault::HookFails {
            output(
                2,
                "",
                "cp: cannot stat '/app/status.txt': No such file or directory\n",
            )
        } else if c.contains("BF handoff done") {
            match self.fault {
                Fault::StateTampered => output(
                    0,
                    "BF state-owner /var/lib/benchflow/baseline/list\nBF handoff stopped\n",
                    "",
                ),
                _ => output(0, if silent { "" } else { "BF handoff done\n" }, ""),
            }
        } else if c.contains("BF owned") {
            output(0, if silent { "" } else { "BF owned root\n" }, "")
        } else if c.contains("BF stage-ready") {
            output(0, if silent { "" } else { "BF stage-ready\n" }, "")
        } else if c.contains("BF staged") {
            output(0, if silent { "" } else { "BF staged\n" }, "")
        } else if c.contains("BF placement done") {
            output(0, if silent { "" } else { "BF placement done\n" }, "")
        } else if c.contains("BF path") {
            output(
                0,
                if silent {
                    ""
                } else {
                    "BF path /usr/local/bin:/usr/bin:/bin\n"
                },
                "",
            )
        } else if c.contains("baseline") && c.contains("files.tar") {
            output(0, "BF baseline 2\n", "")
        } else if c.contains("/verifier/test.sh") || c.contains("/tests/test.sh") {
            match self.fault {
                Fault::VerifierTimeout => ExecOutput {
                    timed_out: true,
                    ..output(124, "collecting ...\n", "")
                },
                Fault::VerifierFails => output(
                    2,
                    "",
                    "/verifier/test.sh: line 3: pytest: command not found\n",
                ),
                _ => output(0, "1 passed\n", ""),
            }
        } else if c.contains("/oracle/solve.sh") {
            self.world.lock().unwrap().solved = true;
            output(0, "", "")
        } else {
            output(0, "", "")
        })
    }
    async fn spawn(&self, _exec: Exec) -> Result<Process, SandboxError> {
        match self.fault {
            Fault::AgentSpawnGone => return Err(gone()),
            Fault::AgentOom => return Ok(fake_acp_agent(Act::Crash, self.world.clone())),
            Fault::AgentGone => {
                self.world.lock().unwrap().gone = true;
                return Ok(fake_acp_agent(Act::Crash, self.world.clone()));
            }
            _ => {}
        }
        Ok(fake_acp_agent(self.act, self.world.clone()))
    }
    async fn upload(&self, local: &Path, _remote: &str) -> Result<(), SandboxError> {
        if self.fault == Fault::LocalRead && local.ends_with("verifier") {
            return Err(SandboxError::io(
                format!("reading {} to upload it", local.display()),
                std::io::Error::from(std::io::ErrorKind::PermissionDenied),
            ));
        }
        Ok(())
    }
    async fn write_file(&self, _remote: &str, _c: &[u8], _mode: u32) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote == "/logs/verifier" {
            if self.world.lock().unwrap().gone || self.fault == Fault::DownloadFails {
                return Err(gone());
            }
            std::fs::create_dir_all(local).unwrap();
            let solved = self.world.lock().unwrap().solved;
            match self.fault {
                Fault::NoReward
                | Fault::VerifierFails
                | Fault::VerifierTimeout
                | Fault::DiskFull => {}
                Fault::BadReward => std::fs::write(local.join("reward.txt"), "passed\n").unwrap(),
                _ => std::fs::write(local.join("reward.txt"), if solved { "1\n" } else { "0\n" })
                    .unwrap(),
            }
        }
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// A scripted ACP agent as a process: stdin, stdout, stderr and a handle.
fn fake_acp_agent(act: Act, world: Arc<Mutex<World>>) -> Process {
    let (client_stdin, agent_stdin) = tokio::io::duplex(64 * 1024);
    let (mut agent_stdout, client_stdout) = tokio::io::duplex(64 * 1024);
    let (mut agent_stderr, client_stderr) = tokio::io::duplex(64 * 1024);
    let task = tokio::spawn(async move {
        if act == Act::ExitAtStart {
            let _ = agent_stderr
                .write_all(b"fatal: no such file: /opt/fake/config.json\n")
                .await;
            return 1;
        }
        let mut lines = BufReader::new(agent_stdin).lines();
        let mut pending: Option<Value> = None;
        while let Ok(Some(line)) = lines.next_line().await {
            let msg: Value = serde_json::from_str(&line).unwrap();
            let id = msg.get("id").cloned();
            let reply = |result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
            let update = |u: Value| {
                json!({"jsonrpc": "2.0", "method": "session/update",
                       "params": {"sessionId": "s1", "update": u}})
            };
            let mut out: Vec<Value> = Vec::new();
            match msg["method"].as_str().unwrap_or_default() {
                "initialize" => out.push(reply(json!({"protocolVersion": 1,
                    "agentCapabilities": {}, "authMethods": [],
                    "agentInfo": {"name": "fake", "version": "1.0"}}))),
                "session/new" if matches!(act, Act::NeedsSignIn) => {
                    out.push(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32000,
                        "message": "Authentication required: Call authenticate before starting a session"}}));
                }
                "session/new" => out.push(reply(json!({"sessionId": "s1",
                    "models": {"currentModelId": "fake-model",
                        "availableModels": [{"modelId": "fake-model", "name": "Fake"}]},
                    "configOptions": [{"id": "model", "name": "Model", "type": "select",
                        "currentValue": "fake-model",
                        "options": [{"value": "fake-model", "name": "Fake"}]}]}))),
                "session/set_config_option" => {
                    let value = msg["params"]["value"].as_str().unwrap_or_default();
                    if value == "fake-model" {
                        out.push(reply(json!({"configOptions": []})));
                    } else {
                        out.push(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602,
                            "message": format!("Invalid value for config option model: {value}")}}));
                    }
                }
                "session/set_model" => out.push(reply(json!({}))),
                "session/prompt" => match act {
                    Act::Solve => {
                        out.push(update(json!({"sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "Done."}})));
                        world.lock().unwrap().solved = true;
                        out.push(reply(json!({"stopReason": "end_turn"})));
                    }
                    Act::Crash => {
                        let _ = agent_stderr
                            .write_all(b"thread 'main' panicked at src/agent.rs:88: index out of bounds\n")
                            .await;
                        return 101;
                    }
                    Act::Error(message) => out.push(json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32603, "message": message}})),
                    Act::Spent => out.push(json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32603, "message": "Internal error",
                            "data": {"codexErrorInfo": "usageLimitExceeded",
                                "message": "You've hit your usage limit. Upgrade to Pro or try again at Oct 3rd, 2026 5:05 PM."}}})),
                    Act::Reply(text) => {
                        out.push(update(json!({"sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": text}})));
                        out.push(reply(json!({"stopReason": "end_turn"})));
                    }
                    Act::Silent | Act::ExitAtStart | Act::NeedsSignIn => pending = id.clone(),
                },
                "session/cancel" => {
                    if let Some(pid) = pending.take() {
                        out.push(json!({"jsonrpc": "2.0", "id": pid,
                            "result": {"stopReason": "cancelled"}}));
                    }
                }
                _ => {
                    if id.is_some() {
                        out.push(json!({"jsonrpc": "2.0", "id": id,
                            "error": {"code": -32601, "message": "method not found"}}));
                    }
                }
            }
            for v in out {
                if agent_stdout
                    .write_all(format!("{v}\n").as_bytes())
                    .await
                    .is_err()
                {
                    return 0;
                }
            }
        }
        0
    });
    Process {
        stdin: Box::pin(client_stdin),
        stdout: Box::pin(client_stdout),
        stderr: Box::pin(client_stderr),
        handle: Box::new(FakeHandle { task: Some(task) }),
    }
}

struct FakeHandle {
    task: Option<tokio::task::JoinHandle<i32>>,
}

#[async_trait]
impl ProcessHandle for FakeHandle {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        match self.task.take() {
            Some(t) => Ok(t.await.unwrap_or(137)),
            None => Ok(137),
        }
    }
    async fn kill(&mut self) -> Result<(), SandboxError> {
        if let Some(t) = self.task.take() {
            t.abort();
        }
        Ok(())
    }
}

struct FakeProvider {
    fault: Fault,
    act: Act,
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &str {
        "fake"
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            network_none: true,
            pids_limit: true,
            build_from_dockerfile: true,
            ..Capabilities::default()
        }
    }
    async fn status(&self) -> ProviderStatus {
        ProviderStatus {
            ok: true,
            summary: "fake".into(),
            fix: None,
        }
    }
    async fn prepare(
        &self,
        spec: &ImageSpec,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        log("building");
        let fail = |task_fault: bool, message: &str, tail: &str| SandboxError::Image {
            name: spec.name.clone(),
            message: message.to_string(),
            log_tail: tail.to_string(),
            task_fault,
        };
        match self.fault {
            Fault::TaskImage => Err(fail(
                true,
                "the task's Dockerfile failed to build (RUN pip install numpy==9.9). This is a problem in the task: fix the Dockerfile",
                "ERROR: No matching distribution found for numpy==9.9",
            )),
            Fault::ImageBackend => Err(fail(
                false,
                "`docker build` failed for a reason outside the task (failed to fetch oauth token: dial tcp: i/o timeout). Check the network and the docker host, then run again.",
                "dial tcp: i/o timeout",
            )),
            Fault::AgentLayers if !spec.layers.is_empty() => Err(fail(
                false,
                "installing the agent on top of the task image failed (npm ERR! 404 Not Found - GET https://registry.npmjs.org/@fake%2fagent). The agent's install step or the network is at fault, not the task; the log tail shows the failing command.",
                "npm ERR! 404 Not Found - GET https://registry.npmjs.org/@fake%2fagent",
            )),
            _ => Ok(Image {
                reference: format!("fake/{}:{}", spec.name, spec.layers.len()),
                fresh: true,
                duration: Duration::from_millis(1),
            }),
        }
    }
    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        match self.fault {
            Fault::Start => {
                return Err(SandboxError::Start(
                    "the container exited at once: exec /entrypoint.sh: no such file or directory"
                        .into(),
                ));
            }
            Fault::NoGpus => {
                return Err(SandboxError::Unsupported(format!(
                    "{} asks for 1 GPU(s), and BenchFlow's Daytona backend has none; run it with `--sandbox docker` on a host with GPUs",
                    spec.name
                )));
            }
            _ => {}
        }
        Ok(Box::new(FakeSandbox {
            world: Arc::new(Mutex::new(World::default())),
            fault: self.fault,
            act: self.act,
        }))
    }
    fn services(&self) -> Option<&dyn ServiceHost> {
        Some(self as &dyn ServiceHost)
    }
}

#[async_trait]
impl ServiceHost for FakeProvider {
    async fn create_network(
        &self,
        name: &str,
        _labels: &BTreeMap<String, String>,
    ) -> Result<String, SandboxError> {
        Ok(format!("net-{name}"))
    }
    async fn start_service(
        &self,
        _network: &str,
        spec: &ServiceSpec,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        if self.fault == Fault::ServiceFails {
            return Err(SandboxError::Start(format!(
                "{} exited at once (exit code 1): Error: Cannot find module '/srv/server.js'",
                spec.name
            )));
        }
        Ok(Box::new(FakeSandbox {
            world: Arc::new(Mutex::new(World::default())),
            fault: Fault::None,
            act: self.act,
        }))
    }
    async fn create_on_network(
        &self,
        spec: &SandboxSpec,
        _network: &str,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        self.create(spec).await
    }
    async fn logs(&self, _id: &str) -> String {
        String::new()
    }
    async fn remove_network(&self, _network: &str) -> Result<(), SandboxError> {
        Ok(())
    }
}

const MANIFEST: &str = r#"
name = "fake"
description = "a scripted ACP agent"
[install]
script = "npm install -g @fake/agent"
[launch]
command = "/opt/fake/bin/fake-acp"
[credentials]
env = ["FAKE_API_KEY"]
[model]
select = "set-model"
default = "fake-model"
"#;

fn manifest() -> AgentManifest {
    AgentManifest::parse(MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap()
}

/// The scripted agent selecting its model as a session config option.
fn config_option_manifest() -> AgentManifest {
    let text = MANIFEST.replace(
        "select = \"set-model\"",
        "select = \"config-option\"\nconfig_id = \"model\"",
    );
    AgentManifest::parse(&text, ManifestSource::Builtin("fake.toml")).unwrap()
}

fn logged_in() -> HostEnv {
    HostEnv::new(
        BTreeMap::from([(
            "FAKE_API_KEY".to_string(),
            "fake-key-0123456789".to_string(),
        )]),
        None,
    )
}

/// An agent that ends the trial ungraded, as `bench shell --no-grade` does.
struct NoGrade;

#[async_trait]
impl ModelAgent for NoGrade {
    fn name(&self) -> &str {
        "shell"
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }
    fn fingerprint(&self) -> String {
        String::new()
    }
    fn preflight(&self) -> Result<(), Problem> {
        Ok(())
    }
    async fn run(&self, _ctx: AgentContext<'_>) -> AgentOutput {
        AgentOutput {
            skip_grading: true,
            ..AgentOutput::default()
        }
    }
}

/// A native-style agent whose provider rate-limited it until its time ran
/// out: its model calls log holds four 429s among six calls.
struct Throttled;

#[async_trait]
impl ModelAgent for Throttled {
    fn name(&self) -> &str {
        "throttled"
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }
    fn fingerprint(&self) -> String {
        String::new()
    }
    fn preflight(&self) -> Result<(), Problem> {
        Ok(())
    }
    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput {
        let dir = ctx.trial_dir.join("agent");
        std::fs::create_dir_all(&dir).unwrap();
        let calls: String = [200, 429, 429, 200, 429, 429]
            .iter()
            .map(|s| format!("{{\"seq\":1,\"status\":{s}}}\n"))
            .collect();
        std::fs::write(dir.join("model_calls.jsonl"), calls).unwrap();
        AgentOutput {
            timed_out: true,
            tool_calls: 2,
            ..AgentOutput::default()
        }
    }
}

/// An agent harness with a bug: it panics.
struct Panics;

#[async_trait]
impl ModelAgent for Panics {
    fn name(&self) -> &str {
        "buggy"
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }
    fn fingerprint(&self) -> String {
        String::new()
    }
    fn preflight(&self) -> Result<(), Problem> {
        Ok(())
    }
    async fn run(&self, _ctx: AgentContext<'_>) -> AgentOutput {
        panic!("index out of bounds: the len is 0 but the index is 0")
    }
}

fn task(root: &Path) -> TaskPackage {
    std::fs::create_dir_all(root.join("oracle")).unwrap();
    std::fs::create_dir_all(root.join("verifier")).unwrap();
    std::fs::write(root.join("oracle/solve.sh"), "#!/bin/bash\n").unwrap();
    std::fs::write(root.join("verifier/test.sh"), "#!/bin/bash\n").unwrap();
    TaskPackage {
        root: root.to_path_buf(),
        format: Format::TaskMd,
        id: "fixtures/codes".into(),
        name: "codes".into(),
        title: None,
        version: None,
        instruction: "Make the test pass.".into(),
        stages: Vec::new(),
        agent: AgentPolicy {
            timeout: Some(Duration::from_secs(60)),
            ..AgentPolicy::default()
        },
        sandbox: SandboxPolicy {
            image: ImageRef::Pull {
                reference: "ubuntu:24.04".into(),
            },
            cpus: None,
            memory_mb: None,
            disk_mb: None,
            gpus: 0,
            build_timeout: None,
            network: Network::Open,
            workdir: Some("/app".into()),
            env: BTreeMap::new(),
            ready: None,
            outputs: Vec::new(),
            skills: None,
            skills_dir: None,
            mcp: Vec::new(),
        },
        verifier: VerifierPolicy {
            dir: root.join("verifier"),
            mount: "/verifier".into(),
            entry: "test.sh".into(),
            starts_in_mount: false,
            timeout: Some(Duration::from_secs(60)),
            user: None,
            env: BTreeMap::new(),
            network: None,
            isolation: Isolation::Shared,
            rubric: None,
        },
        oracle: Some(OraclePolicy {
            dir: root.join("oracle"),
            mount: "/oracle".into(),
            entry: "solve.sh".into(),
            env: BTreeMap::new(),
        }),
        controls: Vec::new(),
        unsupported: Vec::new(),
        extras: Default::default(),
        config: json!({}),
        digest: "sha256:codes".into(),
    }
}

/// Give the task a rubric with one criterion a model judges.
fn judged_rubric(t: &mut TaskPackage, reference: Option<&str>) {
    let mut c = json!({"id": "memo", "text": "The memo names the total.", "judge": "llm",
        "points": 1, "evidence": ["memo.md"]});
    if let Some(r) = reference {
        c["reference"] = json!(r);
    }
    t.verifier.rubric = Some(
        benchflow_task::rubric::Rubric::from_value(&json!({
            "$schema": "https://task.md/schema/rubric-1.json", "criteria": [c]
        }))
        .unwrap(),
    );
}

/// Who attempts the task in a case.
enum Who {
    Oracle,
    /// The scripted ACP agent, on a login, with this model.
    Acp(Option<&'static str>),
    /// The scripted ACP agent asked for a reasoning effort.
    AcpEffort(&'static str),
    /// The scripted ACP agent selecting models as a config option.
    AcpOption(&'static str),
    /// The scripted ACP agent with no login on this machine.
    AcpNoLogin,
    NoGrade,
    Panics,
    Throttled,
}

struct Case {
    code: &'static str,
    owner: Owner,
    what: &'static str,
    who: Who,
    fault: Fault,
    act: Act,
    /// Change the task.
    tweak: fn(&mut TaskPackage),
    /// Run on this fixture task (`tests/fixtures/tasks/<name>`) instead.
    fixture: Option<&'static str>,
    /// Change the run's options.
    options: fn(&mut RunOptions),
    /// Stop the run (Ctrl-C) this long after it starts.
    stop_after: Option<Duration>,
}

impl Case {
    fn new(code: &'static str, owner: Owner, what: &'static str) -> Self {
        Self {
            code,
            owner,
            what,
            who: Who::Oracle,
            fault: Fault::None,
            act: Act::Solve,
            tweak: |_| {},
            fixture: None,
            options: |_| {},
            stop_after: None,
        }
    }
    fn who(mut self, who: Who) -> Self {
        self.who = who;
        self
    }
    fn fault(mut self, fault: Fault) -> Self {
        self.fault = fault;
        self
    }
    fn act(mut self, act: Act) -> Self {
        self.who = Who::Acp(None);
        self.act = act;
        self
    }
    fn tweak(mut self, tweak: fn(&mut TaskPackage)) -> Self {
        self.tweak = tweak;
        self
    }
    fn fixture(mut self, name: &'static str) -> Self {
        self.fixture = Some(name);
        self
    }
    fn options(mut self, options: fn(&mut RunOptions)) -> Self {
        self.options = options;
        self
    }
}

fn cases() -> Vec<Case> {
    use Owner::*;
    vec![
        Case::new(
            "R000",
            You,
            "an interactive session the user chose not to grade",
        )
        .who(Who::NoGrade),
        Case::new("R001", You, "Ctrl-C while the agent works")
            .act(Act::Silent)
            .options(|o| o.idle_timeout = Duration::ZERO),
        Case::new("R101", Task, "the task's Dockerfile fails to build").fault(Fault::TaskImage),
        Case::new(
            "R101",
            Task,
            "the task's Dockerfile fails under a model agent's install layers",
        )
        .act(Act::Solve)
        .fault(Fault::TaskImage),
        Case::new("R102", Sandbox, "the backend cannot prepare the image")
            .fault(Fault::ImageBackend),
        Case::new("R103", Sandbox, "the container fails to start").fault(Fault::Start),
        Case::new(
            "R103",
            Sandbox,
            "the connection to the sandbox drops before grading",
        )
        .fault(Fault::Transport),
        Case::new("R104", Task, "the image has no useradd or adduser").fault(Fault::Useradd),
        Case::new(
            "R108",
            Task,
            "a blocklist on an image with no iptables to enforce it",
        )
        .fault(Fault::NoIptables)
        .tweak(|t| t.sandbox.network = Network::Block(vec!["192.0.2.50".into()])),
        Case::new(
            "R109",
            Benchflow,
            "the firewall self-test reaches a blocked address",
        )
        .fault(Fault::Leaky)
        .tweak(|t| t.sandbox.network = Network::Block(vec!["192.0.2.50".into()])),
        Case::new("R105", Task, "the readiness check never passes")
            .fault(Fault::NotReady)
            .tweak(|t| {
                t.sandbox.ready = Some(Ready {
                    run: "check-ready".into(),
                    interval: Duration::from_millis(10),
                    timeout: Duration::from_secs(5),
                    start_period: Duration::ZERO,
                    start_interval: Duration::from_millis(10),
                    retries: 2,
                })
            }),
        Case::new(
            "R107",
            You,
            "the verifier's files cannot be read on this machine",
        )
        .fault(Fault::LocalRead),
        Case::new(
            "R201",
            Agent,
            "the agent's install layer fails on a task image that builds",
        )
        .act(Act::Solve)
        .fault(Fault::AgentLayers),
        Case::new("R202", Agent, "the agent exits before the ACP handshake").act(Act::ExitAtStart),
        Case::new("R203", Agent, "the agent goes silent during the prompt")
            .act(Act::Silent)
            .options(|o| o.idle_timeout = Duration::from_secs(1)),
        Case::new("R204", Agent, "the agent crashes during the prompt").act(Act::Crash),
        Case::new(
            "R205",
            Agent,
            "the agent answers the prompt with an error of its own",
        )
        .act(Act::Error("Internal error: tool registry is empty")),
        Case::new(
            "R206",
            Agent,
            "the agent does nothing before its time runs out",
        )
        .act(Act::Silent)
        .options(|o| o.idle_timeout = Duration::ZERO)
        .tweak(|t| t.agent.timeout = Some(Duration::from_secs(1))),
        Case::new("R211", You, "no login for the agent on this machine").who(Who::AcpNoLogin),
        Case::new("R212", Model, "the provider rejects the key")
            .act(Act::Error("Invalid API key · Fix external API key")),
        Case::new("R213", Model, "the login's usage limit is reached").act(Act::Spent),
        Case::new("R214", Model, "the provider rate-limits the agent")
            .act(Act::Error("API Error: 429 rate limit exceeded")),
        Case::new(
            "R215",
            You,
            "the agent does not offer the model the run names",
        )
        .act(Act::Solve)
        .who(Who::AcpOption("other-model")),
        Case::new(
            "R215",
            You,
            "the run asks for a reasoning effort the agent lacks",
        )
        .act(Act::Solve)
        .who(Who::AcpEffort("high")),
        Case::new("R301", Task, "the verifier writes no reward").fault(Fault::NoReward),
        Case::new(
            "R302",
            Task,
            "the verifier writes a reward that is not a number",
        )
        .fault(Fault::BadReward),
        Case::new("R303", Task, "the verifier runs past its timeout").fault(Fault::VerifierTimeout),
        Case::new(
            "R304",
            Task,
            "the verifier exits with an error and writes no reward",
        )
        .fault(Fault::VerifierFails),
        Case::new(
            "R305",
            Sandbox,
            "agent processes survive the kill before grading",
        )
        .fault(Fault::Survivors),
        Case::new(
            "R306",
            Task,
            "--agent oracle on a task without a reference solution",
        )
        .tweak(|t| t.oracle = None),
        Case::new(
            "R307",
            Task,
            "a program on the image's PATH answers in place of the shell",
        )
        .fault(Fault::Shadowed),
        Case::new(
            "R309",
            Sandbox,
            "/etc/ld.so.preload cannot be put back before grading",
        )
        .fault(Fault::PreloadStuck),
        Case::new(
            "R310",
            Task,
            "the image's PATH holds a folder others can write",
        )
        .fault(Fault::UncleanVerifierEnv),
        Case::new("R311", Task, "a hook before grading fails")
            .fixture("snapshot-hook")
            .fault(Fault::HookFails),
        Case::new("R312", Model, "the rubric's model judge cannot be reached")
            .tweak(|t| judged_rubric(t, None))
            .options(|o| {
                o.judge_model = Some("vercel/judge/model".into());
                o.judge_env = benchflow_run::config::JudgeEnv(BTreeMap::from([
                    (
                        "AI_GATEWAY_API_KEY".to_string(),
                        "vck_codes_test_key".to_string(),
                    ),
                    // Nothing listens on the discard port.
                    (
                        "BENCHFLOW_NATIVE_BASE_URL".to_string(),
                        "http://127.0.0.1:9/v1".to_string(),
                    ),
                ]));
            }),
        Case::new(
            "R313",
            Task,
            "a rubric's reference names a file the package lacks",
        )
        .tweak(|t| judged_rubric(t, Some("verifier/answers.md"))),
        Case::new(
            "R314",
            Agent,
            "BenchFlow's state folder is no longer root's at the handoff",
        )
        .fault(Fault::StateTampered),
        Case::new(
            "R106",
            Task,
            "a service beside the sandbox exits as it starts",
        )
        .fixture("sidecar")
        .fault(Fault::ServiceFails),
        Case::new(
            "R401",
            You,
            "the backend has no GPUs for a task that asks for one",
        )
        .fault(Fault::NoGpus),
        Case::new("R401", You, "the task asks for a feature this runner lacks").tweak(|t| {
            t.unsupported.push(Unsupported {
                field: "[sandbox] clock".into(),
                reason: "a fixed clock is not in the Rust runner yet".into(),
                agents_only: false,
            })
        }),
        Case::new("R901", Benchflow, "a bug in an agent harness: it panics").who(Who::Panics),
    ]
}

/// The case's trial record, with the temporary folder it ran in written as
/// `<tmp>` so the golden file does not change from run to run.
async fn run(case: &Case) -> Option<TrialRecord> {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().display().to_string();
    let mut t = match case.fixture {
        Some(name) => benchflow_task::load(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tasks")
                .join(name),
        )
        .unwrap_or_else(|d| panic!("{name}: {d:#?}")),
        None => task(&tmp.path().join("codes")),
    };
    (case.tweak)(&mut t);
    let attempt = match &case.who {
        Who::Oracle => Attempt::Oracle,
        Who::Acp(model) => Attempt::Model(Arc::new(AcpAgent::new(
            manifest(),
            *model,
            None,
            &logged_in(),
        ))),
        Who::AcpEffort(effort) => Attempt::Model(Arc::new(AcpAgent::new(
            manifest(),
            None,
            Some(effort),
            &logged_in(),
        ))),
        Who::AcpOption(model) => Attempt::Model(Arc::new(AcpAgent::new(
            config_option_manifest(),
            Some(model),
            None,
            &logged_in(),
        ))),
        Who::AcpNoLogin => Attempt::Model(Arc::new(AcpAgent::new(
            manifest(),
            None,
            None,
            &HostEnv::new(BTreeMap::new(), None),
        ))),
        Who::NoGrade => Attempt::Model(Arc::new(NoGrade)),
        Who::Panics => Attempt::Model(Arc::new(Panics)),
        Who::Throttled => Attempt::Model(Arc::new(Throttled)),
    };
    let mut options = RunOptions {
        retries: 0,
        idle_timeout: Duration::from_secs(30),
        cache_dir: tmp.path().join("cache"),
        ..RunOptions::default()
    };
    (case.options)(&mut options);
    let (stop_tx, stop_rx) = stop_switch();
    let rt = Arc::new(Runtime {
        provider: Arc::new(FakeProvider {
            fault: case.fault,
            act: case.act,
        }),
        images: Images::new(),
        options,
        events: Events::none(),
        stop: stop_rx,
    });
    if let Some(after) = case.stop_after {
        tokio::spawn(async move {
            tokio::time::sleep(after).await;
            let _ = stop_tx.send(true);
        });
    } else {
        // A switch nobody holds any more must not read as a stop.
        drop(stop_tx);
    }
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![attempt])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("codes".into()),
        fresh: true,
        concurrency: 1,
        source: json!({"path": tmp.path()}),
    };
    let out = job::run(spec, rt).await.unwrap();
    out.records.first().cloned().map(|mut rec| {
        if let Some(p) = rec.problem.as_mut() {
            for text in [&mut p.message, &mut p.next] {
                *text = text.replace(&root, "<tmp>");
            }
        }
        rec
    })
}

/// A case's problem as a user reads it in `bench run`'s report.
fn shown(case: &Case, p: &Problem) -> String {
    let mut s = format!(
        "{} · {} · {} · {}\n  {}\n  next: {}\n",
        p.code,
        p.owner.problem(),
        p.phase.label(),
        case.what,
        p.message,
        p.next
    );
    if let Some(d) = &p.detail {
        let first = d.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
        s.push_str(&format!("  detail: {first}\n"));
    }
    s
}

#[tokio::test]
async fn every_run_code_is_triggered_and_reads_as_written() {
    let mut cases = cases();
    for c in &mut cases {
        if c.code == "R001" {
            c.stop_after = Some(Duration::from_millis(400));
        }
    }
    let mut failures = Vec::new();
    let mut golden = String::new();
    for case in &cases {
        let Some(rec) = run(case).await else {
            failures.push(format!("{} ({}): no trial record", case.code, case.what));
            continue;
        };
        let Some(p) = &rec.problem else {
            failures.push(format!(
                "{} ({}): no problem recorded",
                case.code, case.what
            ));
            continue;
        };
        if p.code != case.code || p.owner != case.owner {
            failures.push(format!(
                "{} ({}): got {} owned by {}: {}",
                case.code,
                case.what,
                p.code,
                p.owner.label(),
                p.message
            ));
        }
        for (field, text) in [("message", &p.message), ("next step", &p.next)] {
            if text.trim().is_empty() {
                failures.push(format!("{}: an empty {field}", case.code));
            }
        }
        golden.push_str(&shown(case, p));
        golden.push('\n');
    }
    let triggered: BTreeSet<&str> = cases.iter().map(|c| c.code).collect();
    let listed: BTreeSet<&str> = EXPLANATIONS.iter().map(|e| e.code).collect();
    for code in listed.difference(&triggered) {
        failures.push(format!(
            "{code} is in `bench explain` but no case triggers it"
        ));
    }
    for code in triggered.difference(&listed) {
        failures.push(format!(
            "{code} is triggered here but `bench explain` lacks it"
        ));
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/codes/run.txt");
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &golden).unwrap();
    } else {
        let want = std::fs::read_to_string(&path).unwrap_or_default();
        if want != golden {
            failures.push(format!(
                "tests/codes/run.txt is stale; run with UPDATE_GOLDEN=1 and review the diff. Got:\n{golden}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Failures beyond one per code: the status each records, and that a
/// sandbox or BenchFlow problem is never a score or the task's fault.
#[tokio::test]
async fn failures_record_the_right_status_owner_and_phase() {
    use Owner::*;
    use benchflow_run::outcome::{Phase, Status};
    let mut cases: Vec<(Case, Status, Phase)> = vec![
        (
            Case::new("R103", Sandbox, "the sandbox goes away during setup")
                .fault(Fault::SetupGone),
            Status::Errored,
            Phase::Setup,
        ),
        (
            Case::new("R103", Sandbox, "the sandbox cannot start the agent")
                .act(Act::Solve)
                .fault(Fault::AgentSpawnGone),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R103",
                Sandbox,
                "the sandbox goes away while the agent works",
            )
            .act(Act::Solve)
            .fault(Fault::AgentGone),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R001", You, "Ctrl-C while the verifier runs").fault(Fault::VerifierHangs),
            Status::Skipped,
            Phase::Verify,
        ),
        (
            Case::new("R304", Task, "the verifier fails on a full disk").fault(Fault::DiskFull),
            Status::Unscored,
            Phase::Verify,
        ),
        (
            Case::new("R214", Model, "429s use up the agent's time").who(Who::Throttled),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R204", Agent, "the out-of-memory killer ends the agent")
                .act(Act::Solve)
                .fault(Fault::AgentOom),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R103",
                Sandbox,
                "the connection drops while the verifier runs",
            )
            .fault(Fault::VerifierGone),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new("R103", Sandbox, "the sandbox is killed under the verifier")
                .fault(Fault::VerifierKilled),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new("R103", Sandbox, "the verifier's folder cannot be fetched")
                .fault(Fault::DownloadFails),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "agent processes survive the kill before grading",
            )
            .fault(Fault::Survivors),
            Status::Errored,
            Phase::Handoff,
        ),
        (
            Case::new("R309", Sandbox, "/etc/ld.so.preload cannot be put back")
                .fault(Fault::PreloadStuck),
            Status::Errored,
            Phase::Handoff,
        ),
        (
            Case::new("R301", Task, "the verifier writes no reward").fault(Fault::NoReward),
            Status::Unscored,
            Phase::Verify,
        ),
        (
            Case::new("R001", You, "Ctrl-C while the sandbox is set up").fault(Fault::SetupHangs),
            Status::Skipped,
            Phase::Setup,
        ),
        (
            Case::new("R212", Model, "the provider answers 401").act(Act::Error(
                "API Error: 401 {\"error\":{\"message\":\"Invalid API key\"}}",
            )),
            Status::Errored,
            Phase::Agent,
        ),
        (
            // hermes-acp today: a dead key's 401 returned as its reply.
            Case::new("R212", Model, "the provider's 401 is the agent's only reply").act(
                Act::Reply(
                    "HTTP 401: login fail: Please carry the API secret key in the 'X-Api-Key' field of the request header",
                ),
            ),
            Status::Errored,
            Phase::Agent,
        ),
        (
            // cline-acp and kimi-acp today.
            Case::new("R211", You, "the ACP server needs the vendor's sign-in")
                .act(Act::NeedsSignIn),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R215", You, "the provider retired the model")
                .act(Act::Error("Internal error: Exception: [410] Gone")),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R214", Model, "the provider answers 500")
                .act(Act::Error("API Error: 500 Internal Server Error")),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R214", Model, "the provider times out")
                .act(Act::Error("API Error: Request timed out.")),
            Status::Errored,
            Phase::Agent,
        ),
    ];
    for (c, _, _) in &mut cases {
        if matches!(c.fault, Fault::SetupHangs | Fault::VerifierHangs) {
            c.stop_after = Some(Duration::from_millis(400));
        }
    }
    let mut failures = Vec::new();
    for (case, status, phase) in &cases {
        let rec = run(case).await.expect("a trial record");
        let p = rec.problem.as_ref().expect("a problem");
        if p.code != case.code
            || p.owner != case.owner
            || rec.status != *status
            || p.phase != *phase
        {
            failures.push(format!(
                "{}: want {} {:?} {:?} {:?}, got {} {:?} {:?} {:?}: {}",
                case.what,
                case.code,
                case.owner,
                status,
                phase,
                p.code,
                p.owner,
                rec.status,
                p.phase,
                p.message
            ));
        }
        if case.fault == Fault::DiskFull && !p.message.contains("disk was full") {
            failures.push(format!("{}: the full disk is not named: {p:?}", case.what));
        }
        if case.fault == Fault::AgentOom
            && !(p.message.contains("out-of-memory killer") && p.next.contains("memory"))
        {
            failures.push(format!(
                "{}: the memory kill is not named: {p:?}",
                case.what
            ));
        }
        if p.owner != Task && rec.reward.is_some() {
            failures.push(format!(
                "{}: scored though {}",
                case.what,
                p.owner.problem()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
