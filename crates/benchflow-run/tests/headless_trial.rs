//! Whole trials of a headless agent through the real engine, on an
//! in-memory sandbox whose `spawn` plays a recorded `codex exec --json`
//! stream (or stays silent, or fails the way a spent login or a broken
//! install does), and plays BenchFlow's provider forwarder when it is
//! started.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_agent::AgentManifest;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::manifest::ManifestSource;
use benchflow_run::agent::Attempt;
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::headless_agent::HeadlessAgent;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::outcome::{Owner, Status};
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
use tokio::io::AsyncWriteExt;

#[derive(Clone, Copy, PartialEq)]
enum Behavior {
    Solve,
    /// The agent runs, and the provider answers every model request the
    /// forwarder passes on with 401.
    RejectedKey,
    Silent,
    SpentLogin,
    BrokenInstall,
}

#[derive(Default)]
struct World {
    solved: bool,
    spawned: Vec<Exec>,
    files: BTreeMap<String, Vec<u8>>,
}

struct FakeSandbox {
    world: Arc<Mutex<World>>,
    behavior: Behavior,
}

/// A recorded `codex exec --json` run that solves the task.
fn solving_stream() -> String {
    [
        json!({"type": "thread.started", "thread_id": "0199a"}),
        json!({"type": "turn.started"}),
        json!({"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "**Running the solver**"}}),
        json!({"type": "item.started", "item": {"id": "item_1", "type": "command_execution", "command": "bash -lc ./solve.sh", "aggregated_output": "", "exit_code": null, "status": "in_progress"}}),
        json!({"type": "item.completed", "item": {"id": "item_1", "type": "command_execution", "command": "bash -lc ./solve.sh", "aggregated_output": "solved\n", "exit_code": 0, "status": "completed"}}),
        json!({"type": "item.completed", "item": {"id": "item_2", "type": "agent_message", "text": "Solved it."}}),
        json!({"type": "turn.completed", "usage": {"input_tokens": 110, "cached_input_tokens": 100, "output_tokens": 5}}),
    ]
    .iter()
    .map(|v| format!("{v}\n"))
    .collect()
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-headless"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        let c = exec.command.clone();
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
            // The agent user's copy from BenchFlow's stage into its home.
            assert_eq!(exec.user.as_deref(), Some("agent"), "{c}");
            let mut w = self.world.lock().unwrap();
            let stage = c
                .split("s=")
                .nth(1)
                .and_then(|r| r.lines().next())
                .unwrap_or_default()
                .to_string();
            for line in c.lines() {
                let words: Vec<&str> = line.split_whitespace().collect();
                if let ["place", i, path, _mode] = words[..] {
                    let staged = w.files.get(&format!("{stage}/{i}")).cloned();
                    w.files
                        .insert(path.to_string(), staged.expect("a staged file"));
                }
            }
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
            return Ok(fake_forwarder(self.behavior));
        }
        let (client_stdin, _agent_stdin) = tokio::io::duplex(1024);
        let (mut agent_stdout, client_stdout) = tokio::io::duplex(64 * 1024);
        let (mut agent_stderr, client_stderr) = tokio::io::duplex(64 * 1024);
        let world = self.world.clone();
        let behavior = self.behavior;
        let task = tokio::spawn(async move {
            match behavior {
                Behavior::Solve | Behavior::RejectedKey => {
                    let _ = agent_stderr
                        .write_all(b"codex: reading prompt from stdin\n")
                        .await;
                    world.lock().unwrap().solved = true;
                    let _ = agent_stdout.write_all(solving_stream().as_bytes()).await;
                    0
                }
                Behavior::Silent => std::future::pending::<i32>().await,
                Behavior::SpentLogin => {
                    let line = json!({"type": "turn.failed", "error": {"message":
                        "You\u{2019}ve hit your usage limit. Upgrade to Plus to continue using Codex, or try again at Oct 3rd, 2026 5:05 PM."}});
                    let _ = agent_stdout.write_all(format!("{line}\n").as_bytes()).await;
                    1
                }
                Behavior::BrokenInstall => {
                    let _ = agent_stderr
                        .write_all(
                            b"BenchFlow: the Codex CLI is not at /opt/x; reinstall the agent\n",
                        )
                        .await;
                    127
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
    async fn write_file(&self, remote: &str, c: &[u8], _mode: u32) -> Result<(), SandboxError> {
        self.world
            .lock()
            .unwrap()
            .files
            .insert(remote.to_string(), c.to_vec());
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
fn fake_forwarder(behavior: Behavior) -> Process {
    let (client_stdin, _forwarder_stdin) = tokio::io::duplex(1024);
    let (mut out, client_stdout) = tokio::io::duplex(4096);
    let (mut err, client_stderr) = tokio::io::duplex(4096);
    let task = tokio::spawn(async move {
        let _ = out
            .write_all(b"{\"benchflow_forwarder\": \"ready\", \"port\": 41234}\n")
            .await;
        let status = if behavior == Behavior::RejectedKey {
            401
        } else {
            200
        };
        for _ in 0..if status == 401 { 3 } else { 1 } {
            let line = format!(
                "{{\"method\":\"POST\",\"path\":\"/openai/responses\",\"status\":{status},\"pinned\":\"ignore\"}}\n"
            );
            let _ = err.write_all(line.as_bytes()).await;
        }
        std::future::pending::<i32>().await
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
    behavior: Behavior,
    worlds: Mutex<Vec<Arc<Mutex<World>>>>,
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

/// An agent with both harnesses, ACP its default, like the built-in codex.
const MANIFEST: &str = r#"
name = "fake"
description = "a fake agent for tests"
[install]
script = "true"
[launch]
command = "/opt/fake/bin/fake-acp"
[headless]
command = 'exec /opt/fake/bin/fake exec --json ${BENCHFLOW_MODEL:+--model "$BENCHFLOW_MODEL"} - < "$BENCHFLOW_INSTRUCTION_FILE"'
output = "codex-json"
env = { FAKE_MODE = "test" }
[[headless.files]]
path = ".fake/config.toml"
contents = "approval = 'never'\n"
mode = 0o600
[credentials]
env = ["FAKE_API_KEY"]
[model]
select = "set-model"
default = "openai/fake-model"
strip_prefixes = ["openai/"]
"#;

const KEY: &str = "fake-key-0123456789";

fn task(root: &Path, agent_timeout: Duration) -> TaskPackage {
    std::fs::create_dir_all(root.join("verifier")).unwrap();
    std::fs::write(root.join("verifier/test.sh"), "#!/bin/bash\n").unwrap();
    TaskPackage {
        root: root.to_path_buf(),
        format: Format::TaskMd,
        id: "fixtures/headless".into(),
        name: "headless".into(),
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
            starts_in_mount: false,
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
        digest: "sha256:headless".into(),
    }
}

async fn run_one(
    behavior: Behavior,
    agent_timeout: Duration,
) -> (
    TrialRecord,
    std::path::PathBuf,
    Arc<FakeProvider>,
    tempfile::TempDir,
) {
    let manifest = AgentManifest::parse(MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap();
    let host = HostEnv::new(
        BTreeMap::from([("FAKE_API_KEY".to_string(), KEY.to_string())]),
        None,
    );
    let agent = HeadlessAgent::new(manifest, None, Some("high"), &host);
    run_agent(behavior, agent_timeout, agent).await
}

async fn run_agent(
    behavior: Behavior,
    agent_timeout: Duration,
    agent: HeadlessAgent,
) -> (
    TrialRecord,
    std::path::PathBuf,
    Arc<FakeProvider>,
    tempfile::TempDir,
) {
    run_agent_env(behavior, agent_timeout, agent, BTreeMap::new()).await
}

/// [`run_agent`], with `--env` variables for the agent.
async fn run_agent_env(
    behavior: Behavior,
    agent_timeout: Duration,
    agent: HeadlessAgent,
    agent_env: BTreeMap<String, String>,
) -> (
    TrialRecord,
    std::path::PathBuf,
    Arc<FakeProvider>,
    tempfile::TempDir,
) {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("headless"), agent_timeout));
    let provider = Arc::new(FakeProvider {
        behavior,
        worlds: Mutex::new(Vec::new()),
    });
    let (_tx, rx) = stop_switch();
    std::mem::forget(_tx);
    let rt = Arc::new(Runtime {
        provider: provider.clone(),
        images: Images::new(),
        options: RunOptions {
            idle_timeout: Duration::from_secs(30),
            agent_env,
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
        job_name: Some("headless-job".into()),
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
async fn a_headless_agent_solves_a_task_through_the_engine() {
    let (rec, dir, provider, _tmp) = run_one(Behavior::Solve, Duration::from_secs(60)).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert_eq!(rec.reward, Some(1.0));
    // Not the agent's default harness: its own label, so jobs never mix.
    assert_eq!(rec.agent, "fake:headless");
    assert_eq!(rec.model.as_deref(), Some("openai/fake-model"));
    assert_eq!(rec.tool_calls, 1);
    assert_eq!(rec.prompts, vec!["Make the test pass.".to_string()]);
    assert_eq!(
        (
            rec.usage.input_tokens,
            rec.usage.cache_read_tokens,
            rec.usage.output_tokens
        ),
        (10, 100, 5)
    );
    assert_eq!(rec.usage.source, "headless");
    assert_eq!(rec.trajectory_summary["trajectory_source"], "headless");
    assert_eq!(rec.trajectory_summary["tool_call_steps"], 1);
    let trajectory = std::fs::read_to_string(dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    for expected in [
        "bash -lc ./solve.sh",
        "Solved it.",
        "Running the solver",
        "Make the test pass.",
    ] {
        assert!(trajectory.contains(expected), "{expected}: {trajectory}");
    }
    let log = std::fs::read_to_string(dir.join("agent/fake.txt")).unwrap();
    assert!(log.contains("reading prompt from stdin"), "{log}");

    let world = provider.worlds.lock().unwrap()[0].clone();
    let world = world.lock().unwrap();
    // The instruction and the manifest's files are in the agent's home.
    assert_eq!(
        world
            .files
            .get("/home/agent/.benchflow/instruction.md")
            .map(Vec::as_slice),
        Some(b"Make the test pass.".as_slice())
    );
    assert!(world.files.contains_key("/home/agent/.fake/config.toml"));
    // The command ran as the agent user, in the workdir, with BenchFlow's
    // variables, the manifest's, the task's, and the login's.
    let spawned = &world.spawned;
    assert_eq!(spawned.len(), 1);
    let e = &spawned[0];
    // Every agent command starts by raising its out-of-memory score.
    let command = e
        .command
        .strip_prefix(benchflow_run::hardening::oom_prefix().as_str())
        .expect("the agent's command starts with the out-of-memory prefix");
    assert!(command.starts_with("exec /opt/fake/bin/fake exec --json"));
    assert_eq!(e.user.as_deref(), Some("agent"));
    assert_eq!(e.cwd.as_deref(), Some("/app"));
    let env = |k: &str| e.env.get(k).map(String::as_str);
    assert_eq!(env("HOME"), Some("/home/agent"));
    assert_eq!(
        env("BENCHFLOW_INSTRUCTION_FILE"),
        Some("/home/agent/.benchflow/instruction.md")
    );
    assert_eq!(env("BENCHFLOW_INSTRUCTION"), Some("Make the test pass."));
    assert_eq!(env("BENCHFLOW_MODEL"), Some("fake-model"));
    assert_eq!(env("BENCHFLOW_EFFORT"), Some("high"));
    assert_eq!(env("FAKE_MODE"), Some("test"));
    assert_eq!(env("TASK_VAR"), Some("1"));
    assert_eq!(env("FAKE_API_KEY"), Some(KEY));
    assert!(!rec.env_keys.is_empty());
    // The key reached the agent but no file BenchFlow wrote.
    for entry in walk(&dir) {
        let text = std::fs::read(&entry).unwrap();
        assert!(
            !String::from_utf8_lossy(&text).contains(KEY),
            "{} holds the key",
            entry.display()
        );
    }
}

fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

#[tokio::test]
async fn an_agent_that_does_nothing_before_its_time_runs_out_is_not_graded() {
    let (rec, dir, _p, _tmp) = run_one(Behavior::Silent, Duration::from_secs(2)).await;
    let problem = rec.problem.expect("a problem");
    assert_eq!(problem.code, "R206", "{problem:?}");
    assert_ne!(rec.status, Status::Scored);
    assert_eq!(rec.reward, None);
    let trajectory = std::fs::read_to_string(dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(trajectory.contains("agent_timeout"), "{trajectory}");
}

#[tokio::test]
async fn a_spent_login_is_a_usage_limit() {
    let (rec, _dir, _p, _tmp) = run_one(Behavior::SpentLogin, Duration::from_secs(60)).await;
    let problem = rec.problem.expect("a problem");
    assert_eq!(problem.code, "R213", "{problem:?}");
    assert!(
        problem.message.contains("usage limit reached"),
        "{problem:?}"
    );
    assert!(
        problem.message.contains("resets 2026-10-03 17:05 UTC"),
        "{problem:?}"
    );
}

#[tokio::test]
async fn a_broken_install_did_not_start_and_says_why() {
    let (rec, _dir, _p, _tmp) = run_one(Behavior::BrokenInstall, Duration::from_secs(60)).await;
    let problem = rec.problem.expect("a problem");
    assert_eq!(problem.code, "R202", "{problem:?}");
    assert!(
        problem.message.contains("exited with code 127"),
        "{problem:?}"
    );
    assert!(
        problem
            .detail
            .unwrap_or_default()
            .contains("reinstall the agent"),
        "stderr is the evidence"
    );
}

/// The fake agent with a `[providers.openai]` table, as Codex has.
const PROVIDER_MANIFEST: &str = r#"
name = "fake"
description = "a fake agent for tests"
[install]
script = "true"
[headless]
command = 'exec /opt/fake/bin/fake exec --json ${BENCHFLOW_MODEL:+--model "$BENCHFLOW_MODEL"} - < "$BENCHFLOW_INSTRUCTION_FILE"'
output = "codex-json"
[credentials]
env = ["FAKE_API_KEY"]
[model]
select = "none"
default = "fake-model"
[providers.openai]
api = "responses"
base_url = "FAKE_BASE_URL"
key = "FAKE_API_KEY"
clear = ["FAKE_OTHER_KEY"]
env = { FAKE_CONFIG = "model = {model}" }
"#;

const PROVIDER_KEY: &str = "sk-testprov-fake-key-0123456789";

/// `testprov`, a router that takes a denylist (as OpenRouter does) and
/// reads its key from `TESTPROV_KEY` or the fake agent's own `FAKE_API_KEY`.
fn headless_on(model: &str, vars: &[(&str, &str)]) -> HeadlessAgent {
    use benchflow_agent::providers::{Provider, ProviderRegistry};
    let manifest =
        AgentManifest::parse(PROVIDER_MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap();
    let mut registry = ProviderRegistry::builtin();
    registry.add(
        Provider::parse(
            "name = \"testprov\"\n[openai]\nbase_url = \"https://llm.example/v1\"\napis = [\"responses\"]\n[key]\nenv = [\"TESTPROV_KEY\", \"FAKE_API_KEY\"]\n[routing]\nignore = true\n",
            ManifestSource::File("providers/testprov.toml".into()),
        )
        .unwrap(),
    );
    let vars = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    HeadlessAgent::with_providers(
        manifest,
        Some(model),
        None,
        &HostEnv::new(vars, None),
        &registry,
    )
}

/// What a trial through the forwarder must show: the forwarder first, as
/// root, holding the key in its root-only file and nowhere else; the agent
/// on its address and a placeholder; the pin at the head of its log.
fn check_forwarded(world: &World, dir: &Path, key: &str, model_var: &str, benchflow_model: &str) {
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
    assert_eq!(
        env("FAKE_CONFIG"),
        Some(model_var).filter(|v| !v.is_empty())
    );
    assert_eq!(env("BENCHFLOW_MODEL"), Some(benchflow_model));
    for exec in &world.spawned {
        assert!(!exec.command.contains(key));
        assert!(exec.env.values().all(|v| !v.contains(key)));
    }
    let holding: Vec<&String> = world
        .files
        .iter()
        .filter(|(_, c)| String::from_utf8_lossy(c).contains(key))
        .map(|(path, _)| path)
        .collect();
    assert_eq!(holding, ["/opt/benchflow/forwarder/key"]);
    let config: Value =
        serde_json::from_slice(&world.files["/opt/benchflow/forwarder/config.json"]).unwrap();
    assert_eq!(config["routes"][0]["ignore"], json!(["baseten"]));
    assert!(config["routes"][0].get("only").is_none());
    // Nothing in the job folder holds the key; the forwarder's log is kept,
    // headed by the pin.
    for entry in walk(dir.parent().unwrap()) {
        let text = std::fs::read(&entry).unwrap();
        assert!(
            !String::from_utf8_lossy(&text).contains(key),
            "{} holds the key",
            entry.display()
        );
    }
    let log = std::fs::read_to_string(dir.join("agent/forwarder.log")).unwrap();
    let head: Value = serde_json::from_str(log.lines().next().unwrap()).unwrap();
    assert_eq!(head["upstreams"]["ignore"], json!(["baseten"]));
    assert!(log.contains(r#""pinned":"ignore""#), "{log}");
}

/// A headless agent on a provider key never sees the key, and every request
/// is kept off denied upstreams.
#[tokio::test]
async fn a_headless_agent_on_a_provider_runs_through_the_forwarder() {
    let agent = headless_on(
        "testprov/some-org/some-model",
        &[("TESTPROV_KEY", PROVIDER_KEY)],
    );
    let (rec, dir, provider, _tmp) =
        run_agent(Behavior::Solve, Duration::from_secs(60), agent).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert_eq!(rec.reward, Some(1.0));
    assert!(
        rec.login.as_deref().is_some_and(|l| l.starts_with(
            "TESTPROV_KEY (environment), held by BenchFlow's forwarder for testprov's OpenAI-compatible endpoint"
        )),
        "{:?}",
        rec.login
    );
    let world = provider.worlds.lock().unwrap()[0].clone();
    let world = world.lock().unwrap();
    // The provider's model id, in the table's template and in
    // $BENCHFLOW_MODEL.
    check_forwarded(
        &world,
        &dir,
        PROVIDER_KEY,
        "model = some-org/some-model",
        "some-org/some-model",
    );
}

/// A provider key given with `--env` is the forwarder's too: the agent gets
/// the placeholder, not the key under the name it was given.
#[tokio::test]
async fn a_provider_key_given_with_env_is_withheld_from_the_agent() {
    let env: BTreeMap<String, String> = [("TESTPROV_KEY".to_string(), PROVIDER_KEY.to_string())]
        .into_iter()
        .collect();
    let manifest =
        AgentManifest::parse(PROVIDER_MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap();
    let mut registry = benchflow_agent::providers::ProviderRegistry::builtin();
    registry.add(
        benchflow_agent::providers::Provider::parse(
            "name = \"testprov\"\n[openai]\nbase_url = \"https://llm.example/v1\"\napis = [\"responses\"]\n[key]\nenv = [\"TESTPROV_KEY\", \"FAKE_API_KEY\"]\n[routing]\nignore = true\n",
            ManifestSource::File("providers/testprov.toml".into()),
        )
        .unwrap(),
    );
    let agent = HeadlessAgent::with_providers(
        manifest,
        Some("testprov/some-org/some-model"),
        None,
        &HostEnv::new(BTreeMap::new(), None).with_overrides(env.clone()),
        &registry,
    );
    let (rec, dir, provider, _tmp) =
        run_agent_env(Behavior::Solve, Duration::from_secs(60), agent, env).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    let world = provider.worlds.lock().unwrap()[0].clone();
    let world = world.lock().unwrap();
    check_forwarded(
        &world,
        &dir,
        PROVIDER_KEY,
        "model = some-org/some-model",
        "some-org/some-model",
    );
}

/// The agent's own plain API key goes through the forwarder too; the agent
/// keeps choosing its model as it does on its own.
#[tokio::test]
async fn a_headless_agent_on_its_own_api_key_runs_through_the_forwarder() {
    let agent = headless_on("fake-model", &[("FAKE_API_KEY", KEY)]);
    let (rec, dir, provider, _tmp) =
        run_agent(Behavior::Solve, Duration::from_secs(60), agent).await;
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert!(
        rec.login.as_deref().is_some_and(
            |l| l.starts_with("FAKE_API_KEY (environment), held by BenchFlow's forwarder")
        ),
        "{:?}",
        rec.login
    );
    let world = provider.worlds.lock().unwrap()[0].clone();
    let world = world.lock().unwrap();
    // `{model}` stands for nothing, so the template sets no variable.
    check_forwarded(&world, &dir, KEY, "", "fake-model");
}

/// The provider rejected the key on every model request: whatever the
/// agent did meanwhile, it never had a model, so the trial is R212 (the
/// key's problem), not a score. Live, Claude Code on a rejected key retried
/// in silence until its time ran out, and the run said only R206.
#[tokio::test]
async fn a_key_the_provider_rejects_on_every_request_is_r212() {
    let agent = headless_on(
        "testprov/some-org/some-model",
        &[("TESTPROV_KEY", PROVIDER_KEY)],
    );
    let (rec, _dir, _provider, _tmp) =
        run_agent(Behavior::RejectedKey, Duration::from_secs(60), agent).await;
    assert_eq!(rec.status, Status::Errored);
    let p = rec.problem.expect("a problem");
    assert_eq!((p.code.as_str(), p.owner), ("R212", Owner::Model), "{p:?}");
    assert_eq!(
        p.message,
        "testprov rejected the key on every model request (3 answered HTTP 401); the key is TESTPROV_KEY (environment)"
    );
    assert!(p.next.contains("bench doctor"), "{}", p.next);
}
