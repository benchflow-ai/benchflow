//! Whole trials of a model agent through the real engine and the ACP
//! adapter, on an in-memory sandbox whose `spawn` starts a fake ACP agent.
//!
//! The fake agent speaks ACP over in-process pipes: it answers the
//! handshake, opens a session, takes `session/set_model`, and on
//! `session/prompt` either solves the task (a tool call, a message, usage),
//! stays silent until cancelled, or exits before answering.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_agent::AgentManifest;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::manifest::ManifestSource;
use benchflow_run::acp_agent::AcpAgent;
use benchflow_run::agent::Attempt;
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::outcome::Status;
use benchflow_run::record::TrialRecord;
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, ProcessHandle, Provider,
    ProviderStatus, Sandbox, SandboxError, SandboxSpec,
};
use benchflow_task::TaskPackage;
use benchflow_task::package::*;
use benchflow_task::values::Network;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Clone, Copy, PartialEq)]
enum Behavior {
    Solve,
    Silent,
    CrashAtStart,
}

#[derive(Default)]
struct World {
    solved: bool,
    spawned: Vec<Exec>,
    files: Vec<String>,
    /// Every file written, with its contents and mode.
    written: Vec<(String, Vec<u8>, u32)>,
}

struct FakeSandbox {
    world: Arc<Mutex<World>>,
    behavior: Behavior,
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-acp"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        let c = exec.command;
        let stdout = if c.contains("useradd") {
            "BF user agent 1000 /home/agent\n"
        } else if c.contains("kill -9 -1") {
            "BF killed 1\n"
        } else if c.contains("ld.so.preload") {
            "BF preload absent\n"
        } else if c.contains("BF venv-done") {
            "BF venv-safepath 1\nBF venv-path /usr/local/bin:/usr/bin:/bin\nBF venv-done\n"
        } else if c.contains("BF path") {
            "BF path /usr/local/bin:/usr/bin:/bin\n"
        } else if c.contains("BF handoff done") {
            "BF handoff done\n"
        } else if c.contains("BF owned") {
            "BF owned root\n"
        } else if c.contains("BF stage-ready") {
            "BF stage-ready\n"
        } else if c.contains("BF staged") {
            "BF staged\n"
        } else if c.contains("BF placement done") {
            "BF placement done\n"
        } else {
            ""
        };
        Ok(ExecOutput {
            exit_code: 0,
            stdout: stdout.to_string(),
            stderr: String::new(),
            timed_out: false,
            duration: Duration::from_millis(1),
        })
    }
    async fn spawn(&self, exec: Exec) -> Result<Process, SandboxError> {
        let forwarder = exec.command.contains("forwarder.js");
        self.world.lock().unwrap().spawned.push(exec);
        if forwarder {
            return Ok(fake_forwarder());
        }
        let (client_stdin, agent_stdin) = tokio::io::duplex(64 * 1024);
        let (mut agent_stdout, client_stdout) = tokio::io::duplex(64 * 1024);
        let (mut agent_stderr, client_stderr) = tokio::io::duplex(64 * 1024);
        let world = self.world.clone();
        let behavior = self.behavior;
        let task = tokio::spawn(async move {
            let _ = agent_stderr.write_all(b"fake agent starting\n").await;
            if behavior == Behavior::CrashAtStart {
                let _ = agent_stderr
                    .write_all(b"fatal: no model configured\n")
                    .await;
                return;
            }
            let mut lines = BufReader::new(agent_stdin).lines();
            let mut pending_prompt: Option<Value> = None;
            while let Ok(Some(line)) = lines.next_line().await {
                let msg: Value = serde_json::from_str(&line).unwrap();
                let id = msg.get("id").cloned();
                let method = msg["method"].as_str().unwrap_or_default().to_string();
                let send = |v: Value| {
                    let mut text = v.to_string();
                    text.push('\n');
                    text
                };
                let reply = |id: &Option<Value>, result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
                let update = |u: Value| {
                    json!({"jsonrpc": "2.0", "method": "session/update",
                           "params": {"sessionId": "s1", "update": u}})
                };
                let mut out = String::new();
                match method.as_str() {
                    "initialize" => out.push_str(&send(reply(
                        &id,
                        json!({"protocolVersion": 1, "agentCapabilities": {},
                               "authMethods": [], "agentInfo": {"name": "fake", "version": "1.0"}}),
                    ))),
                    "session/new" => out.push_str(&send(reply(
                        &id,
                        json!({"sessionId": "s1", "models": {"currentModelId": "fake-model",
                               "availableModels": [{"modelId": "fake-model", "name": "Fake"}]}}),
                    ))),
                    "session/set_model" => out.push_str(&send(reply(&id, json!({})))),
                    "session/prompt" => match behavior {
                        Behavior::Solve => {
                            out.push_str(&send(update(json!({"sessionUpdate": "tool_call",
                                "toolCallId": "t1", "title": "bash solve.sh", "kind": "execute",
                                "status": "pending"}))));
                            out.push_str(&send(update(
                                json!({"sessionUpdate": "tool_call_update",
                                "toolCallId": "t1", "status": "completed"}),
                            )));
                            out.push_str(&send(update(
                                json!({"sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": "Solved it."}}),
                            )));
                            world.lock().unwrap().solved = true;
                            out.push_str(&send(reply(
                                &id,
                                json!({"stopReason": "end_turn", "usage":
                                       {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}),
                            )));
                        }
                        _ => pending_prompt = id.clone(),
                    },
                    "session/cancel" => {
                        if let Some(pid) = pending_prompt.take() {
                            out.push_str(&send(reply(
                                &Some(pid),
                                json!({"stopReason": "cancelled"}),
                            )));
                        }
                    }
                    _ => {
                        if id.is_some() {
                            out.push_str(&send(json!({"jsonrpc": "2.0", "id": id,
                                "error": {"code": -32601, "message": "method not found"}})));
                        }
                    }
                }
                if !out.is_empty() && agent_stdout.write_all(out.as_bytes()).await.is_err() {
                    break;
                }
            }
        });
        Ok(Process {
            stdin: Box::pin(client_stdin),
            stdout: Box::pin(client_stdout),
            stderr: Box::pin(client_stderr),
            handle: Box::new(FakeHandle { task: Some(task) }),
        })
    }
    async fn upload(&self, _local: &Path, _remote: &str) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn write_file(&self, remote: &str, c: &[u8], mode: u32) -> Result<(), SandboxError> {
        let mut world = self.world.lock().unwrap();
        world.files.push(remote.to_string());
        world.written.push((remote.to_string(), c.to_vec(), mode));
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote == "/logs/verifier" {
            std::fs::create_dir_all(local).unwrap();
            let solved = self.world.lock().unwrap().solved;
            std::fs::write(local.join("reward.txt"), if solved { "1\n" } else { "0\n" }).unwrap();
        }
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// The provider forwarder: says where it listens, logs one request, and
/// runs until killed.
fn fake_forwarder() -> Process {
    let (client_stdin, _forwarder_stdin) = tokio::io::duplex(1024);
    let (mut out, client_stdout) = tokio::io::duplex(4096);
    let (mut err, client_stderr) = tokio::io::duplex(4096);
    let task = tokio::spawn(async move {
        let _ = out
            .write_all(b"{\"benchflow_forwarder\": \"ready\", \"port\": 41234}\n")
            .await;
        let _ = err
            .write_all(
                b"{\"method\":\"POST\",\"path\":\"/openai/chat/completions\",\"status\":200}\n",
            )
            .await;
        std::future::pending::<()>().await;
    });
    Process {
        stdin: Box::pin(client_stdin),
        stdout: Box::pin(client_stdout),
        stderr: Box::pin(client_stderr),
        handle: Box::new(FakeHandle { task: Some(task) }),
    }
}

struct FakeHandle {
    task: Option<tokio::task::JoinHandle<()>>,
}

#[async_trait]
impl ProcessHandle for FakeHandle {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        if let Some(t) = self.task.take() {
            let _ = t.await;
        }
        Ok(0)
    }
    async fn kill(&mut self) -> Result<(), SandboxError> {
        if let Some(t) = self.task.take() {
            t.abort();
        }
        Ok(())
    }
}

struct FakeProvider {
    behavior: Behavior,
    worlds: Mutex<Vec<Arc<Mutex<World>>>>,
    /// Every image spec prepared, agent layers included.
    prepared: Mutex<Vec<ImageSpec>>,
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
        _log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        self.prepared.lock().unwrap().push(spec.clone());
        Ok(Image {
            reference: format!("fake/{}", spec.name),
            fresh: true,
            duration: Duration::from_millis(1),
        })
    }
    async fn create(&self, _spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        let world = Arc::new(Mutex::new(World::default()));
        self.worlds.lock().unwrap().push(world.clone());
        Ok(Box::new(FakeSandbox {
            world,
            behavior: self.behavior,
        }))
    }
}

const MANIFEST: &str = r#"
name = "fake"
description = "a fake ACP agent for tests"
[install]
script = "true"
[launch]
command = "/opt/fake/bin/fake-acp"
env = { FAKE_MODE = "test" }
[model]
select = "set-model"
default = "fake-model"
"#;

fn task(root: &Path, agent_timeout: Duration) -> TaskPackage {
    std::fs::create_dir_all(root.join("verifier")).unwrap();
    std::fs::write(root.join("verifier/test.sh"), "#!/bin/bash\n").unwrap();
    TaskPackage {
        root: root.to_path_buf(),
        format: Format::TaskMd,
        id: "fixtures/acp".into(),
        name: "acp".into(),
        title: None,
        version: None,
        instruction: "Make the test pass.".into(),
        stages: Vec::new(),
        agent: AgentPolicy {
            timeout: Some(agent_timeout),
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
            env: BTreeMap::from([("TASK_VAR".to_string(), "1".to_string())]),
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
            timeout: Some(Duration::from_secs(60)),
            user: None,
            env: BTreeMap::new(),
            network: None,
            isolation: Isolation::Shared,
            rubric: None,
        },
        oracle: None,
        controls: Vec::new(),
        unsupported: Vec::new(),
        extras: Default::default(),
        config: json!({}),
        digest: "sha256:acp".into(),
    }
}

type RunOne = (
    TrialRecord,
    std::path::PathBuf,
    Arc<FakeProvider>,
    tempfile::TempDir,
);

async fn run_one(behavior: Behavior, agent_timeout: Duration) -> RunOne {
    run_with(MANIFEST, behavior, agent_timeout).await
}

async fn run_with(manifest: &str, behavior: Behavior, agent_timeout: Duration) -> RunOne {
    let manifest = AgentManifest::parse(manifest, ManifestSource::Builtin("fake.toml")).unwrap();
    let host = HostEnv::new(BTreeMap::new(), None);
    run_agent(
        behavior,
        agent_timeout,
        AcpAgent::new(manifest, None, None, &host),
    )
    .await
}

async fn run_manifest(
    manifest: AgentManifest,
    behavior: Behavior,
    agent_timeout: Duration,
) -> RunOne {
    let host = HostEnv::new(BTreeMap::new(), None);
    run_agent(
        behavior,
        agent_timeout,
        AcpAgent::new(manifest, None, None, &host),
    )
    .await
}

async fn run_agent(behavior: Behavior, agent_timeout: Duration, agent: AcpAgent) -> RunOne {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("acp"), agent_timeout));
    let provider = Arc::new(FakeProvider {
        behavior,
        worlds: Mutex::new(Vec::new()),
        prepared: Mutex::new(Vec::new()),
    });
    let (_tx, rx) = stop_switch();
    std::mem::forget(_tx);
    let rt = Arc::new(Runtime {
        provider: provider.clone(),
        images: Images::new(),
        options: RunOptions {
            idle_timeout: Duration::from_secs(30),
            ..RunOptions::default()
        },
        events: Events::none(),
        stop: rx,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Model(Arc::new(agent))])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("acp-job".into()),
        fresh: true,
        concurrency: 1,
        source: json!({"path": tmp.path()}),
    };
    let out = job::run(spec, rt).await.unwrap();
    assert_eq!(out.records.len(), 1);
    let rec = out.records[0].clone();
    let dir = out.dir.join(&rec.rollout_name);
    (rec, dir, provider, tmp)
}

#[tokio::test]
async fn a_fake_acp_agent_solves_a_task_through_the_engine() {
    let (rec, dir, provider, _tmp) = run_one(Behavior::Solve, Duration::from_secs(60)).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert_eq!(rec.reward, Some(1.0));
    assert_eq!(rec.agent, "fake");
    assert_eq!(rec.model.as_deref(), Some("fake-model"));
    assert_eq!(rec.tool_calls, 1);
    assert_eq!(rec.prompts, vec!["Make the test pass.".to_string()]);
    assert_eq!((rec.usage.input_tokens, rec.usage.output_tokens), (10, 5));
    assert_eq!(rec.usage.source, "acp");
    assert!(rec.env_keys.contains(&"FAKE_MODE".to_string()));
    assert!(rec.env_keys.contains(&"TASK_VAR".to_string()));
    assert_eq!(rec.trajectory_summary["tool_call_steps"], 1);
    let trajectory = std::fs::read_to_string(dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(trajectory.contains("bash solve.sh"), "{trajectory}");
    assert!(trajectory.contains("Solved it."), "{trajectory}");
    let log = std::fs::read_to_string(dir.join("agent/fake.txt")).unwrap();
    assert!(log.contains("fake agent starting"), "{log}");
    // The agent ran as the agent user, in the workdir, with HOME set.
    let world = provider.worlds.lock().unwrap()[0].clone();
    let spawned = world.lock().unwrap().spawned.clone();
    assert_eq!(spawned.len(), 1);
    assert_eq!(spawned[0].user.as_deref(), Some("agent"));
    assert_eq!(spawned[0].cwd.as_deref(), Some("/app"));
    assert_eq!(
        spawned[0].env.get("HOME").map(String::as_str),
        Some("/home/agent")
    );
}

/// A run that names no model (Codex's manifest has no default) records the
/// model the agent's session said it was on; it used to record none.
#[tokio::test]
async fn a_run_without_a_model_records_the_one_the_session_reports() {
    let manifest = MANIFEST.replace("default = \"fake-model\"\n", "");
    assert_ne!(manifest, MANIFEST);
    let (rec, _dir, _p, _tmp) = run_with(&manifest, Behavior::Solve, Duration::from_secs(60)).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert_eq!(rec.model.as_deref(), Some("fake-model"));
}

/// config.json records the effort asked for, as BenchFlow 0.8's does, so a
/// paired run can show both engines ran the same `--effort`. (The fake agent
/// offers no effort, so this trial itself fails at model selection.)
#[tokio::test]
async fn the_effort_asked_for_is_recorded_in_config_json() {
    let host = HostEnv::new(BTreeMap::new(), None);
    let effort_of = |dir: &Path| -> serde_json::Value {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        config["reasoning_effort"].clone()
    };
    let manifest = AgentManifest::parse(MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap();
    let agent = AcpAgent::new(manifest.clone(), Some("fake-model"), Some("low"), &host);
    let (_rec, dir, _p, _tmp) = run_agent(Behavior::Solve, Duration::from_secs(60), agent).await;
    assert_eq!(effort_of(&dir), json!("low"));
    let (_rec, dir, _p, _tmp) =
        run_manifest(manifest, Behavior::Solve, Duration::from_secs(60)).await;
    assert_eq!(effort_of(&dir), serde_json::Value::Null);
}

#[tokio::test]
async fn an_agent_that_does_nothing_before_its_time_runs_out_is_not_graded() {
    let (rec, _dir, _p, _tmp) = run_one(Behavior::Silent, Duration::from_secs(2)).await;
    let problem = rec.problem.expect("a problem");
    assert_eq!(problem.code, "R206", "{problem:?}");
    assert_ne!(rec.status, Status::Scored);
    assert_eq!(rec.reward, None);
}

#[tokio::test]
async fn an_agent_that_exits_before_the_handshake_did_not_start() {
    let (rec, _dir, _p, _tmp) = run_one(Behavior::CrashAtStart, Duration::from_secs(60)).await;
    let problem = rec.problem.expect("a problem");
    assert_eq!(problem.code, "R202", "{problem:?}");
    assert!(
        problem
            .detail
            .unwrap_or_default()
            .contains("no model configured"),
        "stderr is the evidence"
    );
}

const PROVIDER_KEY: &str = "sk-testprov-fake-key-0123456789abcdef";

/// The fake agent, able to take a provider's OpenAI-compatible endpoint.
const PROVIDER_MANIFEST: &str = r#"
name = "fake"
description = "a fake ACP agent for tests"
[install]
script = "true"
[launch]
command = "/opt/fake/bin/fake-acp"
[model]
select = "set-model"
default = "fake-model"
[providers.openai]
base_url = "FAKE_BASE_URL"
key = "FAKE_API_KEY"
clear = ["FAKE_OTHER_KEY"]
env = { FAKE_MODEL = "{model}" }
model = ""
"#;

fn provider_agent(manifest: &str, model: &str, vars: &[(&str, &str)]) -> AcpAgent {
    use benchflow_agent::providers::{Provider, ProviderRegistry};
    let manifest = AgentManifest::parse(manifest, ManifestSource::Builtin("fake.toml")).unwrap();
    let mut registry = ProviderRegistry::builtin();
    registry.add(
        Provider::parse(
            "name = \"testprov\"\n[openai]\nbase_url = \"https://llm.example/v1\"\n[key]\nenv = [\"TESTPROV_KEY\"]\n",
            ManifestSource::File("providers/testprov.toml".into()),
        )
        .unwrap(),
    );
    let vars = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    AcpAgent::with_providers(
        manifest,
        Some(model),
        None,
        &HostEnv::new(vars, None),
        &registry,
    )
}

/// Every file under `dir`, with its text.
fn files_under(dir: &Path) -> Vec<(std::path::PathBuf, String)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let text = String::from_utf8_lossy(&std::fs::read(&path).unwrap()).into_owned();
                out.push((path, text));
            }
        }
    }
    out
}

#[tokio::test]
async fn a_provider_model_runs_through_the_forwarder_and_the_agent_never_sees_the_key() {
    let agent = provider_agent(
        PROVIDER_MANIFEST,
        "testprov/some-org/some-model",
        &[("TESTPROV_KEY", PROVIDER_KEY)],
    );
    let (rec, dir, provider, _tmp) =
        run_agent(Behavior::Solve, Duration::from_secs(60), agent).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert_eq!(rec.reward, Some(1.0));
    assert_eq!(rec.model.as_deref(), Some("testprov/some-org/some-model"));
    assert!(
        rec.login
            .as_deref()
            .is_some_and(|l| l.contains("held by BenchFlow's forwarder")),
        "{:?}",
        rec.login
    );

    let world = provider.worlds.lock().unwrap()[0].clone();
    let world = world.lock().unwrap();
    // The forwarder started first, as root; then the agent, as the agent user.
    assert_eq!(world.spawned.len(), 2);
    let (forwarder, agent) = (&world.spawned[0], &world.spawned[1]);
    assert!(forwarder.command.contains("forwarder.js"));
    assert_eq!(forwarder.user.as_deref(), Some("root"));
    assert_eq!(agent.user.as_deref(), Some("agent"));
    let env = |k: &str| agent.env.get(k).map(String::as_str);
    assert_eq!(env("FAKE_BASE_URL"), Some("http://127.0.0.1:41234/openai"));
    assert_eq!(
        env("FAKE_API_KEY"),
        Some("benchflow-forwarder-holds-the-key")
    );
    assert_eq!(env("FAKE_OTHER_KEY"), Some(""));
    assert_eq!(env("FAKE_MODEL"), Some("some-org/some-model"));
    // The key is in the forwarder's root-only key file and nowhere else.
    for exec in &world.spawned {
        assert!(!exec.command.contains(PROVIDER_KEY));
        assert!(exec.env.values().all(|v| !v.contains(PROVIDER_KEY)));
    }
    let holding: Vec<_> = world
        .written
        .iter()
        .filter(|(_, c, _)| String::from_utf8_lossy(c).contains(PROVIDER_KEY))
        .map(|(path, _, mode)| (path.clone(), *mode))
        .collect();
    assert_eq!(
        holding,
        [("/opt/benchflow/forwarder/key".to_string(), 0o600)]
    );
    // Nothing in the job folder holds it either; the forwarder's log is kept.
    let job = dir.parent().unwrap();
    for (path, text) in files_under(job) {
        assert!(!text.contains(PROVIDER_KEY), "{}", path.display());
    }
    let log = std::fs::read_to_string(dir.join("agent/forwarder.log")).unwrap();
    assert!(log.contains("/openai/chat/completions"), "{log}");
}

/// `bench run` calls preflight before it starts anything.
#[test]
fn a_provider_the_agent_cannot_use_is_refused_at_preflight() {
    use benchflow_run::agent::ModelAgent;
    // The plain fake agent has no [providers] table.
    let agent = provider_agent(
        MANIFEST,
        "testprov/some-org/some-model",
        &[("TESTPROV_KEY", PROVIDER_KEY)],
    );
    let problem = agent.preflight().unwrap_err();
    assert_eq!(problem.code, "R215", "{problem:?}");
    assert!(
        problem.message.contains("cannot use provider `testprov`"),
        "{problem:?}"
    );

    // A provider without its key: R211, with where to put it.
    let agent = provider_agent(PROVIDER_MANIFEST, "testprov/some-org/some-model", &[]);
    let problem = agent.preflight().unwrap_err();
    assert_eq!(problem.code, "R211", "{problem:?}");
    assert_eq!(problem.next, "`export TESTPROV_KEY=<key>`");
    assert!(!format!("{problem:?}").contains(PROVIDER_KEY));

    // With the key, preflight passes. The fake agent is not an npm agent,
    // so on a provider its image gets BenchFlow's Node.js as a second layer,
    // for the forwarder.
    let agent = provider_agent(
        PROVIDER_MANIFEST,
        "testprov/some-org/some-model",
        &[("TESTPROV_KEY", PROVIDER_KEY)],
    );
    agent.preflight().unwrap();
    let layers = agent.install_layers();
    assert_eq!(layers.len(), 2);
    assert!(layers[1].contains("/opt/benchflow/node"));
    assert!(layers.iter().all(|l| !l.contains(PROVIDER_KEY)));
}

#[tokio::test]
async fn a_scored_trial_and_its_job_get_trainer_files() {
    let (rec, dir, _p, _tmp) = run_one(Behavior::Solve, Duration::from_secs(60)).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    let read = |path: &Path| std::fs::read_to_string(path).unwrap();
    // 0.8 hands the trainer writer the prompts and the trajectory, which
    // records the prompt too, so the prompt appears twice, as in 0.8.
    let verifiers = read(&dir.join("trainer/verifiers.jsonl"));
    assert_eq!(
        verifiers,
        concat!(
            r#"{"example_id": 0, "prompt": [{"role": "user", "content": "Make the test pass."}, {"role": "user", "content": "Make the test pass."}], "#,
            r#""completion": [{"role": "assistant", "content": "[tool_call: bash solve.sh]"}, {"role": "assistant", "content": "Solved it."}], "#,
            r#""reward": 1.0, "metrics": {}, "is_completed": true, "is_truncated": false, "#,
            r#""info": {"task_id": "acp", "environment": "acp", "model": "fake-model", "reward_valid": true, "#,
            r#""reward_metadata": {"items": {}, "events": [], "error": null, "space": "output", "granularity": "terminal"}}}"#,
            "\n"
        )
    );
    let adp = read(&dir.join("trainer/adp.jsonl"));
    assert_eq!(
        adp,
        format!(
            "{}{}{}{}\n",
            r#"{"schema_version": "1.3.1", "id": ""#,
            rec.rollout_name,
            r#"", "content": [{"class_": "text_observation", "content": "Make the test pass.", "source": "user"}, {"class_": "text_observation", "content": "Make the test pass.", "source": "user"}, "#,
            r#"{"class_": "api_action", "tool_call_id": "t1", "function": "execute", "kwargs": {}, "description": "bash solve.sh"}, {"class_": "text_observation", "tool_call_id": "t1", "content": "", "source": "environment"}, {"class_": "message_action", "content": "Solved it.", "reward": 1.0}], "details": {"task_id": "acp", "environment": "acp", "model": "fake-model"}}"#,
        )
    );
    let atif: Value = serde_json::from_str(&read(&dir.join("trainer/atif.json"))).unwrap();
    assert_eq!(atif["schema_version"], "ATIF-v1.8");
    assert_eq!(atif["session_id"], json!(rec.rollout_name));
    assert_eq!(
        atif["agent"],
        json!({"name": "fake", "version": "unknown", "model_name": "fake-model"})
    );
    let steps = atif["steps"].as_array().unwrap();
    let sources: Vec<&str> = steps
        .iter()
        .map(|s| s["source"].as_str().unwrap())
        .collect();
    assert_eq!(sources, ["user", "user", "agent", "agent"]);
    assert_eq!(steps[2]["tool_calls"][0]["tool_call_id"], "t1");
    assert_eq!(steps[3]["message"], "Solved it.");
    assert_eq!(
        atif["final_metrics"],
        json!({"total_steps": 4, "total_prompt_tokens": 10, "total_completion_tokens": 5, "total_cached_tokens": 0})
    );
    assert!(!dir.join("trainer/atif.coverage.json").exists());
    // The job's files hold its one trial's records.
    let job = dir.parent().unwrap();
    assert_eq!(read(&job.join("verifiers.jsonl")), verifiers);
    assert_eq!(read(&job.join("adp.jsonl")), adp);
}

#[tokio::test]
async fn an_unscored_trial_still_gets_its_trainer_files() {
    let (rec, dir, _p, _tmp) = run_one(Behavior::Silent, Duration::from_secs(2)).await;
    assert_eq!(rec.reward, None);
    let record: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("trainer/verifiers.jsonl")).unwrap(),
    )
    .unwrap();
    assert_eq!(record["reward"], json!(0.0));
    assert_eq!(record["info"]["reward_valid"], json!(false));
    assert!(dir.join("trainer/atif.json").is_file());
}

/// A fake gateway listing (`/v1/models/{model}/endpoints`) on its own thread's
/// runtime: Baseten serves both models, and is the only one serving the
/// second.
fn gateway_listing() -> String {
    use axum::extract::Path as UrlPath;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let app = axum::Router::new().route(
                "/v1/models/{*path}",
                axum::routing::get(|UrlPath(path): UrlPath<String>| async move {
                    let names: &[&str] = match path.as_str() {
                        "some-org/some-model/endpoints" => &["baseten", "novita"],
                        "some-org/only-baseten/endpoints" => &["baseten"],
                        _ => &[],
                    };
                    let endpoints: Vec<Value> =
                        names.iter().map(|n| json!({"provider_name": n})).collect();
                    axum::Json(json!({"data": {"endpoints": endpoints}}))
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(format!("http://{}/v1", listener.local_addr().unwrap()))
                .unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });
    rx.recv().unwrap()
}

/// The fake agent on `testprov` as a gateway whose upstreams are listed at
/// `listing`.
fn gateway_agent(model: &str, listing: &str) -> AcpAgent {
    use benchflow_agent::providers::{Provider, ProviderRegistry, Routing, Style};
    let manifest =
        AgentManifest::parse(PROVIDER_MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap();
    let mut gateway = Provider::parse(
        "name = \"testprov\"\n[openai]\nbase_url = \"https://llm.example/v1\"\n[key]\nenv = [\"TESTPROV_KEY\"]\n",
        ManifestSource::File("providers/testprov.toml".into()),
    )
    .unwrap();
    gateway.routing = Some(Routing {
        style: Style::Only {
            endpoints: format!("{listing}/models/{{model}}/endpoints"),
        },
        deny: Vec::new(),
    });
    let mut registry = ProviderRegistry::builtin();
    registry.add(gateway);
    AcpAgent::with_providers(
        manifest,
        Some(model),
        None,
        &HostEnv::new(
            BTreeMap::from([("TESTPROV_KEY".to_string(), PROVIDER_KEY.to_string())]),
            None,
        ),
        &registry,
    )
}

#[tokio::test]
async fn a_gateway_route_pins_the_forwarder_to_allowed_upstreams() {
    let listing = gateway_listing();
    let agent = gateway_agent("testprov/some-org/some-model", &listing);
    let (rec, dir, provider, _tmp) =
        run_agent(Behavior::Solve, Duration::from_secs(60), agent).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    // The forwarder's config names the upstreams it pins every request to.
    let world = provider.worlds.lock().unwrap()[0].clone();
    let world = world.lock().unwrap();
    let config = world
        .written
        .iter()
        .find(|(path, _, _)| path == "/opt/benchflow/forwarder/config.json")
        .map(|(_, c, _)| serde_json::from_slice::<Value>(c).unwrap())
        .expect("the forwarder's config");
    assert_eq!(config["routes"][0]["only"], json!(["novita"]));
    // And its log says so first.
    let log = std::fs::read_to_string(dir.join("agent/forwarder.log")).unwrap();
    let head: Value = serde_json::from_str(log.lines().next().unwrap()).unwrap();
    assert_eq!(
        head,
        json!({"upstreams": {"model": "some-org/some-model", "only": ["novita"], "excluded": ["baseten"]}})
    );
}

#[test]
fn a_model_only_baseten_serves_never_starts() {
    use benchflow_run::agent::ModelAgent;
    let listing = gateway_listing();
    let agent = gateway_agent("testprov/some-org/only-baseten", &listing);
    let problem = agent.preflight().unwrap_err();
    assert_eq!(problem.code, "R215", "{problem:?}");
    assert_eq!(
        problem.message,
        "no upstream testprov could send `some-org/only-baseten` to is allowed: it is served only by baseten, which BenchFlow does not use"
    );
    assert!(
        problem.next.contains("never uses baseten"),
        "{}",
        problem.next
    );
}

/// An ACP registry entry (`acp:<id>`) runs through the same engine as a
/// built-in agent: its distribution becomes the image's agent layer, its
/// command and variables start the agent, and the record names the entry
/// and version.
#[tokio::test]
async fn an_acp_registry_entry_runs_a_trial_like_any_agent() {
    use benchflow_agent::Registry;
    use benchflow_agent::acp_registry::{Index, resolve_in};

    let index = Index::parse(include_str!(
        "../../benchflow-agent/testdata/acp-registry-2026-10-10.json"
    ))
    .unwrap();
    // factory-droid: an npx package with arguments and variables.
    let manifest = resolve_in(&index, "acp:factory-droid", &Registry::builtin()).unwrap();
    let (rec, dir, provider, _tmp) =
        run_manifest(manifest, Behavior::Solve, Duration::from_secs(60)).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert_eq!(rec.reward, Some(1.0));
    assert_eq!(rec.agent, "acp:factory-droid@0.238.0");
    assert_eq!(rec.tool_calls, 1);
    // No built-in runs this agent and the registry names no login: it ran
    // on none, and said so.
    assert_eq!(
        rec.login.as_deref(),
        Some("no login (the agent needs none from BenchFlow)")
    );
    let trajectory = std::fs::read_to_string(dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(trajectory.contains("Solved it."), "{trajectory}");
    // The install layer is the npx distribution, pinned.
    let prepared = provider.prepared.lock().unwrap().clone();
    let layers: Vec<&String> = prepared.iter().flat_map(|s| s.layers.iter()).collect();
    assert_eq!(layers.len(), 1);
    assert!(layers[0].contains("npm install -g --prefix /opt/benchflow/js-agents droid@0.238.0"));
    assert!(layers[0].contains("> /opt/benchflow/bin/factory-droid"));
    // The agent started with the entry's command and variables.
    let world = provider.worlds.lock().unwrap()[0].clone();
    let spawned = world.lock().unwrap().spawned.clone();
    // Every agent command starts by raising its out-of-memory score.
    assert_eq!(
        spawned[0]
            .command
            .strip_prefix(benchflow_run::hardening::oom_prefix().as_str()),
        Some("/opt/benchflow/bin/factory-droid exec --output-format acp-daemon")
    );
    assert_eq!(
        spawned[0]
            .env
            .get("DROID_DISABLE_AUTO_UPDATE")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(spawned[0].user.as_deref(), Some("agent"));
}

/// `acp:codex-acp` borrows the built-in Codex agent's logins: a ChatGPT
/// login file goes into the sandbox for it.
#[tokio::test]
async fn an_acp_registry_entry_uses_the_logins_of_the_built_in_for_the_same_agent() {
    use benchflow_agent::Registry;
    use benchflow_agent::acp_registry::{Index, resolve_in};

    let index = Index::parse(include_str!(
        "../../benchflow-agent/testdata/acp-registry-2026-10-10.json"
    ))
    .unwrap();
    let manifest = resolve_in(&index, "acp:codex-acp", &Registry::builtin()).unwrap();
    let home = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(home.path().join(".codex")).unwrap();
    std::fs::write(
        home.path().join(".codex/auth.json"),
        r#"{"OPENAI_API_KEY": null, "tokens": {"id_token": "x.y.z", "access_token": "fake-access", "refresh_token": "fake-refresh", "account_id": "acct"}, "last_refresh": "2026-10-10T00:00:00Z"}"#,
    )
    .unwrap();
    let host = HostEnv::new(BTreeMap::new(), Some(home.path().to_path_buf()));
    let agent = AcpAgent::new(manifest, None, None, &host);
    assert!(
        agent.login_label().is_some_and(|l| l.contains("auth.json")),
        "{:?}",
        agent.login_label()
    );
    assert_eq!(
        agent.manifest().launch_env()["INITIAL_AGENT_MODE"],
        "agent-full-access"
    );
}
