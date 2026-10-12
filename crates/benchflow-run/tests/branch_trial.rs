//! `bench branch` through the real engine and the ACP adapter, on an
//! in-memory sandbox that takes checkpoints and whose `spawn` starts a fake
//! ACP agent that can load a session.
//!
//! The fake agent reads its prompt: `solve` solves the task, `break` undoes
//! it, `crash` exits in the middle of its turn, anything else ends the turn
//! without touching the task. A checkpoint keeps whether the task is
//! solved, and a sandbox started from it starts so.

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
use benchflow_run::trial::branch::{self, ChildPlan, ForkPlan, ForkPoint};
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::checkpoint::Checkpoint;
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, ProcessHandle, Provider,
    ProviderStatus, Sandbox, SandboxError, SandboxSpec,
};
use benchflow_task::TaskPackage;
use benchflow_task::package::*;
use benchflow_task::values::Network;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Default)]
struct World {
    solved: bool,
}

struct FakeSandbox {
    world: Arc<Mutex<World>>,
    checkpoints: Arc<Mutex<BTreeMap<String, bool>>>,
    /// `session/load` answers an error.
    fail_load: bool,
    /// Root's removal of the login files finds `~/.codex` a link.
    linked: bool,
    /// The parent's grading waits here (see [`Hold`]).
    hold: Option<Arc<Hold>>,
}

/// Holds a continuing parent's grading until the first child's end is in
/// the fork's records (or 10 s pass), and keeps what the records said
/// then: the parent is still running while a child has ended.
struct Hold {
    /// The job folder the records are written under.
    jobs: std::path::PathBuf,
    /// tree.json, and whether labels.json and the child's observation.json
    /// were there, as first seen with a child scored.
    seen: Mutex<Option<(Value, bool, bool)>>,
}

impl Hold {
    async fn wait(&self) {
        for _ in 0..200 {
            let trees = walk(&self.jobs, "tree.json");
            for t in trees {
                let Ok(tree) =
                    serde_json::from_str::<Value>(&std::fs::read_to_string(&t).unwrap_or_default())
                else {
                    continue;
                };
                if tree["forks"][0]["children"][0]["status"] == "scored" {
                    let trial = t.parent().unwrap();
                    let fork = trial
                        .join("branches")
                        .join(tree["forks"][0]["id"].as_str().unwrap());
                    let labels = fork.join("labels.json").is_file();
                    let observation = fork.join("children/n1/observation.json").is_file();
                    *self.seen.lock().unwrap() = Some((tree, labels, observation));
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

/// Every file named `name` under `dir`, at any depth.
fn walk(dir: &Path, name: &str) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            found.extend(walk(&p, name));
        } else if p.file_name().is_some_and(|n| n == name) {
            found.push(p);
        }
    }
    found
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-branch"
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
        } else if self.linked && c.contains("BF left") && c.contains("BF state-checked") {
            "BF left /home/agent/.codex/auth.json\nBF state-baseline\nBF state-checked\n"
        } else if c.contains("baseline") && c.contains("files.tar") && !c.contains("after") {
            // A stage's test-runner baseline.
            "BF baseline 1\n"
        } else if c.contains("BF state-checked") {
            "BF state-baseline\nBF state-checked\n"
        } else if c.contains("BF scanned") {
            "BF scanned 1\nBF done 1\n"
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
    async fn spawn(&self, _exec: Exec) -> Result<Process, SandboxError> {
        let (client_stdin, agent_stdin) = tokio::io::duplex(64 * 1024);
        let (mut agent_stdout, client_stdout) = tokio::io::duplex(64 * 1024);
        let (_agent_stderr, client_stderr) = tokio::io::duplex(64 * 1024);
        let world = self.world.clone();
        let fail_load = self.fail_load;
        let task = tokio::spawn(async move {
            let mut lines = BufReader::new(agent_stdin).lines();
            let mut stalled: Option<Value> = None;
            while let Ok(Some(line)) = lines.next_line().await {
                let msg: Value = serde_json::from_str(&line).unwrap();
                let id = msg.get("id").cloned();
                let method = msg["method"].as_str().unwrap_or_default().to_string();
                let send = |v: Value| format!("{v}\n");
                let reply = |id: &Option<Value>, result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
                let update = |u: Value| {
                    json!({"jsonrpc": "2.0", "method": "session/update",
                           "params": {"sessionId": "s1", "update": u}})
                };
                let mut out = String::new();
                match method.as_str() {
                    "initialize" => out.push_str(&send(reply(
                        &id,
                        json!({"protocolVersion": 1, "agentCapabilities": {"loadSession": true},
                               "authMethods": [], "agentInfo": {"name": "fake", "version": "1.0"}}),
                    ))),
                    "session/load" if fail_load => out.push_str(&send(json!({"jsonrpc": "2.0",
                        "id": id, "error": {"code": -32603, "message": "no such session"}}))),
                    "session/new" | "session/load" => out.push_str(&send(reply(
                        &id,
                        json!({"sessionId": "s1", "models": {"currentModelId": "fake-model",
                               "availableModels": [{"modelId": "fake-model", "name": "Fake"}]}}),
                    ))),
                    "session/set_model" => out.push_str(&send(reply(&id, json!({})))),
                    "session/prompt" if msg["params"]["prompt"].to_string().contains("stall") => {
                        // Works, then never ends its turn: the time limit
                        // stops it.
                        out.push_str(&send(update(json!({"sessionUpdate": "tool_call",
                            "toolCallId": "t0", "title": "bash long.sh", "kind": "execute",
                            "status": "pending"}))));
                        stalled = id.clone();
                    }
                    "session/cancel" => {
                        if let Some(pid) = stalled.take() {
                            out.push_str(&send(reply(
                                &Some(pid),
                                json!({"stopReason": "cancelled"}),
                            )));
                        }
                    }
                    "session/prompt" => {
                        let text = msg["params"]["prompt"].to_string();
                        if text.contains("crash") {
                            // Exits in the middle of its turn.
                            let note = send(update(json!({"sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": "Crashing."}})));
                            let _ = agent_stdout.write_all(note.as_bytes()).await;
                            return;
                        }
                        if text.contains("solve") {
                            world.lock().unwrap().solved = true;
                        }
                        if text.contains("break") {
                            world.lock().unwrap().solved = false;
                        }
                        out.push_str(&send(update(json!({"sessionUpdate": "tool_call",
                            "toolCallId": "t1", "title": "bash work.sh", "kind": "execute",
                            "status": "pending"}))));
                        out.push_str(&send(update(json!({"sessionUpdate": "tool_call_update",
                            "toolCallId": "t1", "status": "completed"}))));
                        out.push_str(&send(update(
                            json!({"sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "Done."}}),
                        )));
                        out.push_str(&send(reply(
                            &id,
                            json!({"stopReason": "end_turn", "usage":
                                   {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}),
                        )));
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
    async fn write_file(&self, _remote: &str, _c: &[u8], _mode: u32) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote == "/logs/verifier" {
            if let Some(hold) = &self.hold {
                hold.wait().await;
            }
            std::fs::create_dir_all(local).unwrap();
            let solved = self.world.lock().unwrap().solved;
            std::fs::write(local.join("reward.txt"), if solved { "1\n" } else { "0\n" }).unwrap();
        }
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn checkpoint(&self, name: &str, _scrub: &[String]) -> Result<Checkpoint, SandboxError> {
        let solved = self.world.lock().unwrap().solved;
        self.checkpoints
            .lock()
            .unwrap()
            .insert(name.to_string(), solved);
        Ok(Checkpoint {
            provider: "fake".into(),
            reference: name.to_string(),
            seconds: 0.1,
            stored_bytes: Some(1),
            default_user: None,
        })
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

#[derive(Default)]
struct FakeProvider {
    checkpoints: Arc<Mutex<BTreeMap<String, bool>>>,
    from_checkpoint: Mutex<usize>,
    /// The child sandbox (1-based, in start order) whose agent cannot load
    /// the parent's session.
    fail_load_in: Option<usize>,
    /// Sandboxes whose login files are behind a link.
    linked: bool,
    /// The parent's grading waits for a child's end to be in the records.
    hold: Option<Arc<Hold>>,
}

impl FakeProvider {
    fn sandbox(&self, solved: bool, fail_load: bool) -> Box<dyn Sandbox> {
        Box::new(FakeSandbox {
            world: Arc::new(Mutex::new(World { solved })),
            checkpoints: self.checkpoints.clone(),
            fail_load,
            linked: self.linked,
            hold: None,
        })
    }
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
            checkpoints: true,
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
        Ok(Box::new(FakeSandbox {
            world: Arc::new(Mutex::new(World { solved: false })),
            checkpoints: self.checkpoints.clone(),
            fail_load: false,
            linked: self.linked,
            hold: self.hold.clone(),
        }))
    }
    async fn create_from_checkpoint(
        &self,
        _spec: &SandboxSpec,
        checkpoint: &Checkpoint,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        let n = {
            let mut n = self.from_checkpoint.lock().unwrap();
            *n += 1;
            *n
        };
        // With a hold, the second child starts only once the first child's
        // end has been seen in the records (10 s at most), so the test tells
        // a write per child from one write after every child.
        if let Some(hold) = &self.hold
            && n == 2
        {
            for _ in 0..200 {
                if hold.seen.lock().unwrap().is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
        let solved = *self
            .checkpoints
            .lock()
            .unwrap()
            .get(&checkpoint.reference)
            .expect("a checkpoint the fake took");
        Ok(self.sandbox(solved, self.fail_load_in == Some(n)))
    }
    async fn delete_checkpoint(&self, checkpoint: &Checkpoint) -> Result<bool, SandboxError> {
        self.checkpoints
            .lock()
            .unwrap()
            .remove(&checkpoint.reference);
        Ok(true)
    }
}

const MANIFEST: &str = r#"
name = "fake"
description = "a fake ACP agent that loads sessions, for tests"
[install]
script = "true"
[launch]
command = "/opt/fake/bin/fake-acp"
[model]
select = "set-model"
default = "fake-model"
[resume]
acp = true
session_dirs = [".fake/sessions"]
"#;

fn task(root: &Path) -> TaskPackage {
    std::fs::create_dir_all(root.join("verifier")).unwrap();
    std::fs::write(root.join("verifier/test.sh"), "#!/bin/bash\n").unwrap();
    TaskPackage {
        root: root.to_path_buf(),
        format: Format::TaskMd,
        id: "fixtures/branch".into(),
        name: "branch".into(),
        title: None,
        version: None,
        instruction: "Please solve the task.".into(),
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
            reward_range: None,
        },
        oracle: None,
        controls: Vec::new(),
        unsupported: Vec::new(),
        extras: Default::default(),
        config: json!({}),
        digest: "sha256:branch".into(),
    }
}

fn read_json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// A `prompt:1` fork that resumes the parent's session: the parent solves
/// the task; one child keeps it (1), one undoes it (0), one's agent crashes
/// mid-turn (an agent failure, a scored 0). The value is 1/3 and each
/// child's advantage its reward minus that; the fork hangs off a node whose
/// step is the parent's last event; every child loaded the parent's
/// session and started from the checkpoint.
async fn run_fork(tmp: &Path, provider: Arc<FakeProvider>, children: &[&str]) -> branch::BranchJob {
    run_fork_timed(tmp, provider, children, Duration::from_secs(60)).await
}

async fn run_fork_timed(
    tmp: &Path,
    provider: Arc<FakeProvider>,
    children: &[&str],
    agent_timeout: Duration,
) -> branch::BranchJob {
    run_fork_with(
        tmp,
        provider,
        children,
        agent_timeout,
        false,
        Events::none(),
    )
    .await
}

async fn run_fork_with(
    tmp: &Path,
    provider: Arc<FakeProvider>,
    children: &[&str],
    agent_timeout: Duration,
    continue_parent: bool,
    events: Events,
) -> branch::BranchJob {
    let mut t = task(&tmp.join("branch"));
    t.agent.timeout = Some(agent_timeout);
    let t = Arc::new(t);
    let (_tx, rx) = stop_switch();
    std::mem::forget(_tx);
    let rt = Arc::new(Runtime {
        provider,
        images: Images::new(),
        options: RunOptions {
            idle_timeout: Duration::from_secs(30),
            ..RunOptions::default()
        },
        events,
        stop: rx,
    });
    let manifest = AgentManifest::parse(MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap();
    let agent = AcpAgent::new(manifest, None, None, &HostEnv::new(BTreeMap::new(), None));
    let plan = ForkPlan {
        at: ForkPoint::Prompt(1),
        children: children
            .iter()
            .map(|c| ChildPlan::parse(c).unwrap())
            .collect(),
        retain: false,
        resume: true,
        concurrency: 1,
        continue_parent,
    };
    branch::run_job(
        vec![t],
        Attempt::Model(Arc::new(agent)),
        plan,
        tmp.join("jobs"),
        Some("prompt-fork".into()),
        rt,
    )
    .await
    .unwrap()
}

/// A branch job tells the live view how many trial lines to expect (the
/// parent and each child), as `bench run` does; without it the progress
/// line read "1/0 done, 2 running". Guards the E7 examples fix on
/// `rust/e7-examples`.
#[tokio::test]
async fn a_branch_job_announces_its_parent_and_children_as_its_total() {
    let tmp = tempfile::tempdir().unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    run_fork_with(
        tmp.path(),
        Arc::new(FakeProvider::default()),
        &[
            "label=a,prompt=Check your work.",
            "label=b,prompt=Check your work.",
        ],
        Duration::from_secs(60),
        false,
        Events::new(tx),
    )
    .await;
    let mut started = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        if let benchflow_run::events::Event::JobStarted { total, reused, .. } = ev {
            started.push((total, reused));
        }
    }
    assert_eq!(started, [(3, 0)]);
}

/// A child whose agent cannot load the parent's session never ran from the
/// fork: `failed`, unscored, and the fork is `partial` with no value; the
/// others are scored as usual.
#[tokio::test]
async fn a_child_that_cannot_resume_leaves_a_partial_fork() {
    let tmp = tempfile::tempdir().unwrap();
    let provider = Arc::new(FakeProvider {
        fail_load_in: Some(2),
        ..FakeProvider::default()
    });
    let out = run_fork(
        tmp.path(),
        provider,
        &[
            "label=a,prompt=Check your work.",
            "label=b,prompt=Check your work.",
        ],
    )
    .await;
    let f = &out.forks[0];
    assert_eq!(f.status, "partial");
    assert_eq!(f.value, None);
    // The parent stopped after its agent's turn, not in setup.
    let p = f.parent.problem.as_ref().unwrap();
    assert_eq!(p.code, "R000");
    assert_eq!(p.phase, benchflow_run::outcome::Phase::Agent);
    let tree = read_json(&out.dir.join(&f.parent.rollout_name).join("tree.json"));
    let kids = tree["forks"][0]["children"].as_array().unwrap().clone();
    assert_eq!(kids[0]["status"], "scored");
    assert_eq!(kids[0]["reward"], 1.0);
    assert_eq!(kids[1]["status"], "failed", "{}", kids[1]);
    assert!(kids[1]["reward"].is_null());
    assert!(tree["forks"][0]["value"].is_null());
}

#[tokio::test]
async fn a_prompt_fork_resumes_the_session_and_scores_each_child() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("branch")));
    let provider = Arc::new(FakeProvider::default());
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
    let manifest = AgentManifest::parse(MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap();
    let agent = AcpAgent::new(manifest, None, None, &HostEnv::new(BTreeMap::new(), None));
    let child = |s: &str| ChildPlan::parse(s).unwrap();
    let plan = ForkPlan {
        at: ForkPoint::Prompt(1),
        children: vec![
            child("label=keep,prompt=Check your work."),
            child("label=undo,prompt=Now break it."),
            child("label=boom,prompt=Now crash."),
        ],
        retain: false,
        resume: true,
        concurrency: 1,
        continue_parent: false,
    };
    let out = branch::run_job(
        vec![t],
        Attempt::Model(Arc::new(agent)),
        plan,
        tmp.path().join("jobs"),
        Some("prompt-fork".into()),
        rt,
    )
    .await
    .unwrap();
    let f = &out.forks[0];
    let why: Vec<String> = f
        .children
        .iter()
        .map(|c| {
            let r = c.record.as_ref().unwrap();
            format!(
                "{} {:?} {:?} {:?}",
                c.label, r.status, r.problem, r.agent_session
            )
        })
        .collect();
    assert_eq!(f.status, "completed", "{why:#?}");
    assert_eq!(*provider.from_checkpoint.lock().unwrap(), 3);
    assert!(provider.checkpoints.lock().unwrap().is_empty(), "deleted");
    assert!(
        (f.value.unwrap() - 1.0 / 3.0).abs() < 1e-12,
        "{:?}",
        f.value
    );
    let trial = out.dir.join(&f.parent.rollout_name);
    let tree = read_json(&trial.join("tree.json"));
    let fork = &tree["forks"][0];
    assert_eq!(fork["snapshot"]["agent_session"], "resumed");
    let rows: Vec<(String, String, Value, Value, Value)> = fork["children"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["intervention"]["label"].as_str().unwrap().to_string(),
                c["status"].as_str().unwrap().to_string(),
                c["reward"].clone(),
                c["reward_source"].clone(),
                c["error"]["code"].clone(),
            )
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            (
                "keep".into(),
                "scored".into(),
                json!(1.0),
                json!("verifier"),
                Value::Null
            ),
            (
                "undo".into(),
                "scored".into(),
                json!(0.0),
                json!("verifier"),
                Value::Null
            ),
            (
                "boom".into(),
                "scored".into(),
                json!(0.0),
                json!("agent_failure"),
                json!("R204")
            ),
        ],
        "{fork}"
    );
    // The fork node: after prompt 1, at the parent's last event.
    let events = std::fs::read_to_string(trial.join("trajectory/acp_trajectory.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count();
    assert!(events > 0);
    assert_eq!(fork["parent_node"], "p1");
    let p1 = tree["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["id"] == "p1")
        .unwrap()
        .clone();
    let step = p1["step_id"].as_str().unwrap();
    assert!(step.starts_with(&format!("step-{}-", events - 1)), "{step}");
    let cps = read_json(&trial.join("checkpoints.json"));
    assert_eq!(cps["checkpoints"][0]["node_id"], "p1");
    assert_eq!(cps["checkpoints"][0]["agent_session_id"], "s1");
    // Every child loaded the parent's session; its own prompt is recorded.
    for c in &f.children {
        let rec = c.record.as_ref().unwrap();
        assert_eq!(rec.resumed_session.as_deref(), Some("s1"), "{}", c.label);
        let obs = read_json(&c.dir.join("observation.json"));
        assert_eq!(obs["lineage"]["parent_node"], "p1");
    }
    // results.jsonl agrees with tree.json, the crashed child included.
    let lines: Vec<Value> = std::fs::read_to_string(out.dir.join("results.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    let boom = &lines[2];
    assert_eq!(boom["info"]["branch"]["label"], "boom");
    assert_eq!(boom["reward"], 0.0, "{boom}");
    assert_eq!(boom["info"]["branch"]["child_reward"], 0.0);
    let adv = |i: usize| lines[i]["info"]["branch"]["advantage"].as_f64().unwrap();
    assert!((adv(0) - 0.666667).abs() < 1e-9, "{}", adv(0));
    assert!((adv(2) + 0.333333).abs() < 1e-9, "{}", adv(2));
}

/// A fork after a stage whose agent ran out of time (graded anyway, the
/// default `on_timeout`): the children's own agents did not time out, so
/// they are scored by the stage rule, not failed for the parent's time-out.
#[tokio::test]
async fn a_stage_fork_after_a_timed_out_stage_scores_its_children() {
    use benchflow_task::extras::{Combine, StepSpec, Steps, Unreached};
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("staged"));
    let stage = |name: &str, prompt: &str, timeout: Option<Duration>| StepSpec {
        name: name.into(),
        prompt: prompt.into(),
        graded: true,
        timeout,
        ..StepSpec::default()
    };
    t.extras.steps = Some(Steps {
        stages: vec![
            stage("first", "Please stall.", Some(Duration::from_secs(1))),
            stage("second", "Now solve it.", None),
        ],
        combine: Combine::Mean,
        unreached: Unreached::Zero,
    });
    let t = Arc::new(t);
    let provider = Arc::new(FakeProvider::default());
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
    let manifest = AgentManifest::parse(MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap();
    let agent = AcpAgent::new(manifest, None, None, &HostEnv::new(BTreeMap::new(), None));
    let plan = ForkPlan {
        at: ForkPoint::Stage("first".into()),
        children: ForkPlan::same_prompt(2),
        retain: false,
        resume: false,
        concurrency: 1,
        continue_parent: false,
    };
    let out = branch::run_job(
        vec![t],
        Attempt::Model(Arc::new(agent)),
        plan,
        tmp.path().join("jobs"),
        Some("staged-timeout".into()),
        rt,
    )
    .await
    .unwrap();
    let f = &out.forks[0];
    let first = read_json(&out.dir.join(&f.parent.rollout_name).join("tree.json"));
    assert_eq!(f.status, "completed", "{first} {:?}", f.parent.problem);
    for c in &f.children {
        let rec = c.record.as_ref().unwrap();
        // The carried time-out stays on the record (on_timeout = "fail"
        // would zero it); the child's own agent did not run out of time.
        assert!(rec.timed_out, "{}", c.label);
        assert!(!rec.own_timed_out, "{}", c.label);
    }
    // Each child: stage first 0 (carried), second 1: mean 0.5.
    assert_eq!(f.value, Some(0.5));
}

/// A child whose own agent runs out of time is failed, as 0.8 counts it,
/// and its results.jsonl row agrees with tree.json: no reward, though its
/// verifier graded it.
#[tokio::test]
async fn a_timed_out_child_is_failed_in_tree_and_results() {
    let tmp = tempfile::tempdir().unwrap();
    let provider = Arc::new(FakeProvider::default());
    let out = run_fork_timed(
        tmp.path(),
        provider,
        &[
            "label=keep,prompt=Check your work.",
            "label=slow,prompt=Now stall.",
        ],
        Duration::from_secs(1),
    )
    .await;
    let f = &out.forks[0];
    assert_eq!(f.status, "partial", "{:?}", f.parent.problem);
    let tree = read_json(&out.dir.join(&f.parent.rollout_name).join("tree.json"));
    let slow = &tree["forks"][0]["children"][1];
    assert_eq!(slow["status"], "failed", "{slow}");
    assert_eq!(slow["error"]["code"], "timeout");
    let rows: Vec<Value> = std::fs::read_to_string(out.dir.join("results.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let row = &rows[1];
    assert_eq!(row["info"]["branch"]["child_status"], "failed");
    assert!(row["reward"].is_null(), "{row}");
    assert!(row["passed"].is_null(), "{row}");
    assert!(row["metrics"]["reward"].is_null(), "{row}");
}

/// With a continuing parent, a child's end is written while the parent is
/// still running (as docs/rust/branching.md promises: "whenever the parent
/// or a child ends"): tree.json with the fork `running`, the parent with no
/// grade yet, the ended child scored and the next `not_started`;
/// labels.json; and the child's observation.json. The parent's grading
/// waits until the records show a child's end (10 s at most), so a run
/// killed then would leave them.
#[tokio::test]
async fn a_childs_end_is_recorded_while_a_continuing_parent_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let hold = Arc::new(Hold {
        jobs: tmp.path().join("jobs"),
        seen: Mutex::new(None),
    });
    let provider = Arc::new(FakeProvider {
        hold: Some(hold.clone()),
        ..FakeProvider::default()
    });
    let out = run_fork_with(
        tmp.path(),
        provider,
        &[
            "label=keep,prompt=Check your work.",
            "label=undo,prompt=Now break it.",
        ],
        Duration::from_secs(60),
        true,
        Events::none(),
    )
    .await;
    let f = &out.forks[0];
    assert_eq!(f.status, "completed", "{:?}", f.parent.problem);
    assert_eq!(f.parent.status, benchflow_run::outcome::Status::Scored);
    let (tree, labels, observation) = hold
        .seen
        .lock()
        .unwrap()
        .clone()
        .expect("no record of a child's end while the parent ran");
    let fork = &tree["forks"][0];
    assert_eq!(fork["status"], "running", "{fork}");
    assert!(fork["value"].is_null(), "{fork}");
    assert!(fork["parent_reward"].is_null(), "{fork}");
    assert!(fork["cost"]["parent_sandbox_seconds"].is_null(), "{fork}");
    assert_eq!(fork["children"][0]["status"], "scored", "{fork}");
    assert_eq!(fork["children"][1]["status"], "not_started", "{fork}");
    assert!(labels, "labels.json was not written while the parent ran");
    assert!(
        observation,
        "observation.json was not written while the parent ran"
    );
    // The final records replace them, with the parent's grade.
    let tree = read_json(&out.dir.join(&f.parent.rollout_name).join("tree.json"));
    assert_eq!(tree["forks"][0]["status"], "completed");
    assert_eq!(tree["forks"][0]["parent_reward"], 1.0);
    let result = read_json(&out.dir.join(&f.parent.rollout_name).join("result.json"));
    assert_eq!(result["branches"]["forks"][0]["parent_reward"], 1.0);
}

/// A credential file the agent's family writes, left in place because a
/// folder on its way is a link: its value may be a rotated token the scan
/// cannot know, so the fork is refused (the agent's doing), not taken.
#[tokio::test]
async fn a_credential_file_behind_a_link_refuses_the_fork() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("branch")));
    let provider = Arc::new(FakeProvider {
        linked: true,
        ..FakeProvider::default()
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
    let manifest = format!("{MANIFEST}[credentials]\nfamily = \"openai\"\n");
    let manifest = AgentManifest::parse(&manifest, ManifestSource::Builtin("fake.toml")).unwrap();
    let host = HostEnv::new(
        BTreeMap::from([(
            "OPENAI_API_KEY".to_string(),
            "sk-fake-0123456789abcdef0123456789".to_string(),
        )]),
        None,
    );
    let agent = AcpAgent::new(manifest, None, None, &host);
    let plan = ForkPlan {
        at: ForkPoint::Prompt(1),
        children: ForkPlan::same_prompt(2),
        retain: false,
        resume: false,
        concurrency: 1,
        continue_parent: false,
    };
    let out = branch::run_job(
        vec![t],
        Attempt::Model(Arc::new(agent)),
        plan,
        tmp.path().join("jobs"),
        Some("linked".into()),
        rt,
    )
    .await
    .unwrap();
    let f = &out.forks[0];
    assert_eq!(f.status, "failed");
    assert!(f.checkpoint.is_none(), "no checkpoint was taken");
    assert!(provider.checkpoints.lock().unwrap().is_empty());
    let p = f.parent.problem.as_ref().unwrap();
    assert_eq!(p.code, "R401", "{p:?}");
    assert_eq!(p.owner, benchflow_run::outcome::Owner::Agent, "{p:?}");
    assert!(p.message.contains("/home/agent/.codex/auth.json"), "{p:?}");
}
