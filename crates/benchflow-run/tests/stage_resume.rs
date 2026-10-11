//! `--resume-sessions` and agent options through the real engine, on the
//! `two-stage` fixture and a scripted sandbox whose `spawn` starts a fake
//! agent: over ACP (with or without `loadSession`) or as a headless CLI.
//!
//! Stage two resumes stage one's session when the agent can, its session
//! folders survived the handoff, and the run asked; otherwise it starts
//! fresh and its row in `step_results` says why.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_agent::AgentManifest;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::manifest::{Harness, ManifestSource};
use benchflow_run::acp_agent::AcpAgent;
use benchflow_run::agent::{AgentContext, AgentOutput, Attempt, ModelAgent};
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::headless_agent::HeadlessAgent;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::outcome::{Problem, Status};
use benchflow_run::record::TrialRecord;
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, ProcessHandle, Provider,
    ProviderStatus, Sandbox, SandboxError, SandboxSpec,
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// How the fake sandbox and its agent behave.
#[derive(Clone, Copy)]
struct Setup {
    /// The ACP agent offers `session/load`.
    load_session: bool,
    /// The agent's session folders are not empty after the handoff.
    sessions_kept: bool,
}

#[derive(Default)]
struct World {
    /// Every process spawned.
    spawned: Vec<Exec>,
    /// Every ACP request the fake agent got: (method, params).
    requests: Vec<(String, Value)>,
    spawns: usize,
    /// Every command run, in order, with `spawn` for each process spawned.
    log: Vec<String>,
    /// Each run of a readiness check: its command and `PATH`.
    ready: Vec<(String, Option<String>)>,
}

struct FakeSandbox {
    world: Arc<Mutex<World>>,
    setup: Setup,
}

fn ok(stdout: &str) -> ExecOutput {
    ExecOutput {
        exit_code: 0,
        stdout: stdout.to_string(),
        stderr: String::new(),
        timed_out: false,
        duration: Duration::from_millis(1),
    }
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-resume"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        let c = exec.command;
        self.world.lock().unwrap().log.push(c.clone());
        if c.contains("fake-ready-check") {
            let path = exec.env.get("PATH").cloned();
            self.world.lock().unwrap().ready.push((c.clone(), path));
        }
        Ok(if c.contains("/var/lib/benchflow/sessions.tar") {
            ok("BF skipped-large /home/agent/.fake/sessions/huge.bin\nBF packed 2 100\n")
        } else if c.contains("BF kept") {
            if self.setup.sessions_kept {
                ok("BF kept /home/agent/.fake/sessions\n")
            } else {
                ok("")
            }
        } else if c.contains("useradd") {
            ok("BF user agent 1000 /home/agent\n")
        } else if c.contains("baseline") && c.contains("files.tar") && !c.contains("after") {
            // Each stage retakes the test-runner baseline.
            ok("BF baseline 2\n")
        } else if c.contains("kill -9 -1") {
            ok("BF killed 1\n")
        } else if c.contains("ld.so.preload") {
            ok("BF preload absent\n")
        } else if c.contains("BF venv-done") {
            ok("BF venv-safepath 1\nBF venv-path /usr/bin:/bin\nBF venv-done\n")
        } else if c.contains("BF path") {
            ok("BF path /usr/bin:/bin\n")
        } else if c.contains("BF handoff done") {
            ok("BF handoff done\n")
        } else if c.contains("BF owned") {
            ok("BF owned root\n")
        } else if c.contains("BF stage-ready") {
            ok("BF stage-ready\n")
        } else if c.contains("BF staged") {
            ok("BF staged\n")
        } else if c.contains("BF placement done") {
            ok("BF placement done\n")
        } else {
            ok("")
        })
    }
    async fn spawn(&self, exec: Exec) -> Result<Process, SandboxError> {
        let headless = exec.command.contains("fake-cli");
        let n = {
            let mut w = self.world.lock().unwrap();
            w.spawned.push(exec);
            w.log.push("spawn".into());
            w.spawns += 1;
            w.spawns
        };
        if headless {
            return Ok(cli_process());
        }
        Ok(acp_process(self.world.clone(), self.setup, n))
    }
    async fn upload(&self, _local: &Path, _remote: &str) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn write_file(&self, _r: &str, _c: &[u8], _m: u32) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote == "/logs/verifier" {
            // Every stage passes, so the gate lets the second one run.
            std::fs::create_dir_all(local).unwrap();
            std::fs::write(local.join("reward.txt"), "1\n").unwrap();
        } else if remote.ends_with("sessions.tar") {
            // The agent's session folder, with a login file beside the
            // session that must never be saved.
            let mut b = tar::Builder::new(std::fs::File::create(local).unwrap());
            if self.setup.sessions_kept {
                for (path, text) in [
                    ("home/agent/.fake/sessions/sess-1.jsonl", "{\"turn\": 1}\n"),
                    ("home/agent/.fake/sessions/auth.json", "{\"key\": \"x\"}"),
                ] {
                    let mut h = tar::Header::new_gnu();
                    h.set_size(text.len() as u64);
                    h.set_mode(0o600);
                    h.set_mtime(1_760_000_000);
                    h.set_cksum();
                    b.append_data(&mut h, path, text.as_bytes()).unwrap();
                }
            }
            b.finish().unwrap();
        } else if remote.ends_with("outputs.tar") {
            let mut b = tar::Builder::new(std::fs::File::create(local).unwrap());
            b.finish().unwrap();
        }
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        Ok(())
    }
}

/// A fake ACP agent, the `n`th started in this sandbox: its new session is
/// `sess-<n>`; `session/load` replays a line of the old conversation first.
fn acp_process(world: Arc<Mutex<World>>, setup: Setup, n: usize) -> Process {
    let (client_stdin, agent_stdin) = tokio::io::duplex(64 * 1024);
    let (mut agent_stdout, client_stdout) = tokio::io::duplex(64 * 1024);
    let (_agent_stderr, client_stderr) = tokio::io::duplex(1024);
    let task = tokio::spawn(async move {
        let mut lines = BufReader::new(agent_stdin).lines();
        let mut session = format!("sess-{n}");
        while let Ok(Some(line)) = lines.next_line().await {
            let msg: Value = serde_json::from_str(&line).unwrap();
            let id = msg.get("id").cloned();
            let method = msg["method"].as_str().unwrap_or_default().to_string();
            world
                .lock()
                .unwrap()
                .requests
                .push((method.clone(), msg["params"].clone()));
            let reply = |result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
            let update = |session: &str, text: &str| {
                json!({"jsonrpc": "2.0", "method": "session/update", "params": {
                    "sessionId": session,
                    "update": {"sessionUpdate": "agent_message_chunk",
                               "content": {"type": "text", "text": text}}}})
            };
            let mut out: Vec<Value> = Vec::new();
            match method.as_str() {
                "initialize" => out.push(reply(json!({"protocolVersion": 1,
                    "agentCapabilities": {"loadSession": setup.load_session},
                    "authMethods": [], "agentInfo": {"name": "fake", "version": "1"}}))),
                "session/new" => out.push(reply(json!({"sessionId": session}))),
                "session/load" => {
                    session = msg["params"]["sessionId"].as_str().unwrap().to_string();
                    out.push(update(&session, "words replayed from the stage before"));
                    out.push(reply(json!({})));
                }
                "session/prompt" => {
                    out.push(update(&session, "working on this stage"));
                    out.push(reply(json!({"stopReason": "end_turn"})));
                }
                _ => {
                    if id.is_some() {
                        out.push(reply(json!({})));
                    }
                }
            }
            for v in out {
                let mut text = v.to_string();
                text.push('\n');
                if agent_stdout.write_all(text.as_bytes()).await.is_err() {
                    return;
                }
            }
        }
    });
    Process {
        stdin: Box::pin(client_stdin),
        stdout: Box::pin(client_stdout),
        stderr: Box::pin(client_stderr),
        handle: Box::new(Handle { task: Some(task) }),
    }
}

/// A headless CLI that prints a line and exits 0.
fn cli_process() -> Process {
    let (client_stdin, _cli_stdin) = tokio::io::duplex(1024);
    let (mut out, client_stdout) = tokio::io::duplex(1024);
    let (_err, client_stderr) = tokio::io::duplex(1024);
    let task = tokio::spawn(async move {
        let _ = out.write_all(b"I did this stage's work.\n").await;
    });
    Process {
        stdin: Box::pin(client_stdin),
        stdout: Box::pin(client_stdout),
        stderr: Box::pin(client_stderr),
        handle: Box::new(Handle { task: Some(task) }),
    }
}

struct Handle {
    task: Option<tokio::task::JoinHandle<()>>,
}

#[async_trait]
impl ProcessHandle for Handle {
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
    world: Arc<Mutex<World>>,
    setup: Setup,
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
        Ok(Box::new(FakeSandbox {
            world: self.world.clone(),
            setup: self.setup,
        }))
    }
}

const ACP_MANIFEST: &str = r#"
name = "fake"
[install]
script = "true"
[launch]
command = "/opt/fake/bin/fake-acp"
[model]
select = "none"
[resume]
acp = true
session_dirs = [".fake/sessions"]
"#;

const HEADLESS_MANIFEST: &str = r#"
name = "fakeh"
[install]
script = "true"
[headless]
command = 'fake-cli ${BENCHFLOW_RESUME:+--continue} "$@" < "$BENCHFLOW_INSTRUCTION_FILE"'
output = "text"
[resume]
headless = true
session_dirs = [".fake/sessions"]
[options.turns]
type = "int"
description = "Most turns."
min = 1
flag = "--turns"
"#;

fn manifest(text: &str) -> AgentManifest {
    AgentManifest::parse(text, ManifestSource::File(PathBuf::from("fake.toml"))).unwrap()
}

struct Run {
    rec: TrialRecord,
    dir: PathBuf,
    world: Arc<Mutex<World>>,
    _tmp: tempfile::TempDir,
}

impl Run {
    fn rows(&self) -> Vec<Value> {
        let result: Value =
            serde_json::from_str(&std::fs::read_to_string(self.dir.join("result.json")).unwrap())
                .unwrap();
        result["step_results"].as_array().unwrap().clone()
    }
    fn config(&self) -> Value {
        serde_json::from_str(&std::fs::read_to_string(self.dir.join("config.json")).unwrap())
            .unwrap()
    }
}

async fn run(agent: Arc<dyn ModelAgent>, setup: Setup, resume_sessions: bool) -> Run {
    run_with(agent, setup, resume_sessions, |_| {}).await
}

/// [`run`], on the fixture as `tweak` changes it.
async fn run_with(
    agent: Arc<dyn ModelAgent>,
    setup: Setup,
    resume_sessions: bool,
    tweak: impl FnOnce(&mut benchflow_task::TaskPackage),
) -> Run {
    let tmp = tempfile::tempdir().unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/two-stage");
    let mut task = benchflow_task::load(&dir).unwrap();
    tweak(&mut task);
    let task = Arc::new(task);
    let world = Arc::new(Mutex::new(World::default()));
    let (tx, rx) = stop_switch();
    std::mem::forget(tx);
    let rt = Arc::new(Runtime {
        provider: Arc::new(FakeProvider {
            world: world.clone(),
            setup,
        }),
        images: Images::new(),
        options: RunOptions {
            resume_sessions,
            idle_timeout: Duration::from_secs(20),
            ..RunOptions::default()
        },
        events: Events::none(),
        stop: rx,
    });
    let spec = JobSpec {
        units: vec![(task, vec![Attempt::Model(agent)])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("resume".into()),
        fresh: true,
        concurrency: 1,
        source: json!({}),
        carry: Vec::new(),
    };
    let out = job::run(spec, rt).await.unwrap();
    let rec = out.records[0].clone();
    let dir = out.dir.join(&rec.rollout_name);
    Run {
        rec,
        dir,
        world,
        _tmp: tmp,
    }
}

fn acp_agent() -> Arc<dyn ModelAgent> {
    let host = HostEnv::new(BTreeMap::new(), None);
    Arc::new(AcpAgent::new(manifest(ACP_MANIFEST), None, None, &host))
}

fn loads(world: &World) -> Vec<Value> {
    world
        .requests
        .iter()
        .filter(|(m, _)| m == "session/load")
        .map(|(_, p)| p.clone())
        .collect()
}

#[tokio::test]
async fn the_second_stage_loads_the_first_stages_acp_session() {
    let r = run(
        acp_agent(),
        Setup {
            load_session: true,
            sessions_kept: true,
        },
        true,
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let rows = r.rows();
    assert_eq!(rows[0]["session"]["id"], "sess-1");
    assert_eq!(
        rows[0]["session"]["resumed"], false,
        "the first stage is fresh"
    );
    assert_eq!(rows[1]["session"]["id"], "sess-1", "the same session");
    assert_eq!(rows[1]["session"]["resumed"], true);
    assert_eq!(rows[1]["session"]["why"], Value::Null);
    let w = r.world.lock().unwrap();
    let loaded = loads(&w);
    assert_eq!(loaded.len(), 1, "{:?}", w.requests);
    assert_eq!(loaded[0]["sessionId"], "sess-1");
    assert!(loaded[0]["cwd"].is_string());
    assert_eq!(
        w.requests
            .iter()
            .filter(|(m, _)| m == "session/new")
            .count(),
        1,
        "only the first stage opened a session"
    );
    // The replayed conversation is the first stage's record, not the second's.
    let second =
        std::fs::read_to_string(r.dir.join("steps/second/trajectory/acp_trajectory.jsonl"))
            .unwrap();
    assert!(second.contains("working on this stage"), "{second}");
    assert!(!second.contains("replayed"), "{second}");
    assert_eq!(r.config()["resume_sessions"], true);
}

#[tokio::test]
async fn the_agents_session_is_saved_into_the_trial_without_its_login_files() {
    let r = run(
        acp_agent(),
        Setup {
            load_session: true,
            sessions_kept: true,
        },
        true,
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let result: Value =
        serde_json::from_str(&std::fs::read_to_string(r.dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(
        result["agent_session"],
        json!({
            "path": "agent/sessions",
            "dirs": [".fake/sessions"],
            "files": 1,
            "bytes": 12,
            "session_id": "sess-1",
            "left_out": [".fake/sessions/huge.bin", ".fake/sessions/auth.json"],
        })
    );
    let saved = r.dir.join("agent/sessions/.fake/sessions");
    assert_eq!(
        std::fs::read_to_string(saved.join("sess-1.jsonl")).unwrap(),
        "{\"turn\": 1}\n"
    );
    assert!(
        !saved.join("auth.json").exists(),
        "a login file is never saved"
    );
    assert!(!r.dir.join("agent/sessions.tar").exists());
    let modified = std::fs::metadata(saved.join("sess-1.jsonl"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(
        modified,
        std::time::UNIX_EPOCH + Duration::from_secs(1_760_000_000),
        "files keep their time, so the newest session can be told"
    );
}

#[tokio::test]
async fn the_agents_processes_are_killed_before_root_packs_its_session() {
    let r = run(
        acp_agent(),
        Setup {
            load_session: true,
            sessions_kept: true,
        },
        true,
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    let mut killed = true;
    let mut packs = 0;
    for c in &w.log {
        if c == "spawn" {
            killed = false;
        } else if c.contains("kill -9 -1") {
            killed = true;
        } else if c.contains("/var/lib/benchflow/sessions.tar") {
            packs += 1;
            assert!(
                killed,
                "a session was packed while the agent's processes could still run"
            );
        }
    }
    assert!(packs >= 2, "each graded stage saves the session: {packs}");
}

#[tokio::test]
async fn without_the_option_every_stage_starts_fresh() {
    let r = run(
        acp_agent(),
        Setup {
            load_session: true,
            sessions_kept: true,
        },
        false,
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let rows = r.rows();
    assert_eq!(rows[1]["session"]["id"], "sess-2", "a new session");
    assert_eq!(rows[1]["session"]["resumed"], false);
    assert!(loads(&r.world.lock().unwrap()).is_empty());
    assert_eq!(r.config()["resume_sessions"], false);
}

#[tokio::test]
async fn an_agent_without_session_load_starts_fresh_and_says_so() {
    let r = run(
        acp_agent(),
        Setup {
            load_session: false,
            sessions_kept: true,
        },
        true,
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let s = &r.rows()[1]["session"];
    assert_eq!(
        (s["id"].as_str(), s["resumed"].as_bool()),
        (Some("sess-2"), Some(false))
    );
    assert_eq!(
        s["why"],
        "fake does not offer session/load, so this stage started a fresh session"
    );
    assert!(loads(&r.world.lock().unwrap()).is_empty());
}

#[tokio::test]
async fn empty_session_folders_start_a_fresh_session() {
    let r = run(
        acp_agent(),
        Setup {
            load_session: true,
            sessions_kept: false,
        },
        true,
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let s = &r.rows()[1]["session"];
    assert_eq!(s["resumed"], false);
    assert!(
        s["why"]
            .as_str()
            .unwrap()
            .contains("session folders (~/.fake/sessions) were empty"),
        "{s}"
    );
    assert!(loads(&r.world.lock().unwrap()).is_empty());
    assert!(r.rec.agent_session.is_none(), "nothing to save");
    assert!(!r.dir.join("agent/sessions").exists());
}

#[tokio::test]
async fn headless_stages_resume_with_the_clis_flag_and_options_reach_it() {
    let m = manifest(HEADLESS_MANIFEST);
    let values = m
        .resolve_options(&[("turns".into(), "4".into())], Harness::Headless)
        .unwrap();
    let host = HostEnv::new(BTreeMap::new(), None);
    let agent = Arc::new(HeadlessAgent::new(
        m.with_options(&values),
        None,
        None,
        &host,
    ));
    let r = run(
        agent,
        Setup {
            load_session: false,
            sessions_kept: true,
        },
        true,
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    let cli: Vec<&Exec> = w
        .spawned
        .iter()
        .filter(|e| e.command.contains("fake-cli"))
        .collect();
    assert_eq!(cli.len(), 2, "one run per stage");
    assert!(
        cli[0].command.contains("set -- --turns 4\nfake-cli "),
        "{}",
        cli[0].command
    );
    assert!(
        !cli[0].env.contains_key("BENCHFLOW_RESUME"),
        "stage one is fresh"
    );
    assert_eq!(
        cli[1].env.get("BENCHFLOW_RESUME").map(String::as_str),
        Some("1")
    );
    let rows = r.rows();
    assert_eq!(rows[1]["session"]["resumed"], true);
    assert_eq!(r.config()["agent_options"], json!({"turns": 4}));
}

/// An agent that cannot resume, and reports options.
struct Plain;

#[async_trait]
impl ModelAgent for Plain {
    fn name(&self) -> &str {
        "plain"
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }
    fn fingerprint(&self) -> String {
        "plain".into()
    }
    fn preflight(&self) -> Result<(), Problem> {
        Ok(())
    }
    fn options(&self) -> serde_json::Map<String, Value> {
        let mut m = serde_json::Map::new();
        m.insert("max_turns".into(), json!(3));
        m
    }
    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput {
        assert!(
            ctx.resume.is_none(),
            "an agent that cannot resume is never asked to"
        );
        AgentOutput {
            session_id: Some("p".into()),
            ..AgentOutput::default()
        }
    }
}

#[tokio::test]
async fn an_agent_that_cannot_resume_runs_each_stage_fresh_and_the_record_says_so() {
    let r = run(
        Arc::new(Plain),
        Setup {
            load_session: true,
            sessions_kept: true,
        },
        true,
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let s = &r.rows()[1]["session"];
    assert_eq!(s["resumed"], false);
    assert_eq!(
        s["why"],
        "plain cannot resume its own session on this harness, so this stage started a fresh one"
    );
    assert_eq!(r.config()["agent_options"], json!({"max_turns": 3}));
}

/// A later stage's readiness check runs after the agent's turn, as root:
/// it gets the trusted `PATH` and loader variables unset, as hooks do,
/// while the first stage's runs as the image has it.
#[tokio::test]
async fn a_later_stages_readiness_check_gets_the_trusted_path() {
    let check = |n: &str| benchflow_task::package::Ready {
        run: format!("fake-ready-check {n}"),
        interval: Duration::from_millis(10),
        timeout: Duration::from_secs(5),
        start_period: Duration::ZERO,
        start_interval: Duration::from_millis(10),
        retries: 0,
    };
    let r = run_with(
        acp_agent(),
        Setup {
            load_session: true,
            sessions_kept: true,
        },
        false,
        |task| {
            let stages = &mut task.extras.steps.as_mut().unwrap().stages;
            stages[0].ready = Some(check("first"));
            stages[1].ready = Some(check("second"));
        },
    )
    .await;
    let ready = r.world.lock().unwrap().ready.clone();
    assert_eq!(ready.len(), 2, "{ready:?}");
    assert_eq!(ready[0], ("fake-ready-check first".to_string(), None));
    let (command, path) = &ready[1];
    assert!(command.starts_with("unset LD_PRELOAD"), "{command}");
    assert!(command.ends_with("fake-ready-check second"), "{command}");
    assert_eq!(path.as_deref(), Some("/usr/bin:/bin"));
}
