//! Skills and loaded sessions through the real engine, on a scripted
//! sandbox that records every upload, file, and command, and whose `spawn`
//! starts a fake agent (over ACP, with or without `loadSession`, or as a
//! headless CLI).
//!
//! Skills: the task's folder (`[sandbox] skills`, a split task's
//! `environment/skills/`, or Harbor's in-sandbox `skills_dir`) and the run's
//! `--skill` folders are copied into each of the agent's skill folders,
//! owned by the agent user, before the agent starts; config.json says what
//! landed where, or why nothing did.
//!
//! Loaded sessions (`--load-trajectory`): the file goes where the agent's
//! CLI keeps sessions, owned by the agent user, and the first prompt
//! resumes it.

use std::collections::BTreeMap;
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
use benchflow_run::headless_agent::HeadlessAgent;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::outcome::{Problem, Status};
use benchflow_run::record::TrialRecord;
use benchflow_run::skills::{self, STAGING};
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, ProcessHandle, Provider,
    ProviderStatus, Sandbox, SandboxError, SandboxSpec,
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const SESSION_ID: &str = "0b9c8a52-6f3e-4f0e-9d7a-2a5c1e7b4d10";

/// How the fake sandbox and its agent behave.
#[derive(Clone, Default)]
struct Setup {
    /// The ACP agent offers `session/load`.
    load_session: bool,
    /// The skills in the image's own skills folder (Harbor's `skills_dir`);
    /// `None`: the folder is not there.
    image_skills: Option<Vec<&'static str>>,
    /// The agent user cannot write the files BenchFlow places for it.
    unplaced: bool,
}

/// One thing the trial did to the sandbox, in order.
#[derive(Debug, Clone)]
enum Op {
    Exec(Exec),
    Upload(PathBuf, String),
    Write(String, Vec<u8>, u32),
    Spawn(Exec),
}

#[derive(Default)]
struct World {
    ops: Vec<Op>,
    /// Every ACP request the fake agent got: (method, params).
    requests: Vec<(String, Value)>,
    spawns: usize,
}

impl World {
    fn uploads(&self) -> Vec<(PathBuf, String)> {
        self.ops
            .iter()
            .filter_map(|o| match o {
                Op::Upload(l, r) => Some((l.clone(), r.clone())),
                _ => None,
            })
            .collect()
    }
    fn execs(&self) -> Vec<&str> {
        self.ops
            .iter()
            .filter_map(|o| match o {
                Op::Exec(e) => Some(e.command.as_str()),
                _ => None,
            })
            .collect()
    }
    /// The script that placed the skills, and where it ran among the ops.
    fn placement(&self) -> Option<(usize, &str)> {
        self.ops.iter().enumerate().find_map(|(i, o)| match o {
            Op::Exec(e) if e.command.contains("BF skill ") => Some((i, e.command.as_str())),
            _ => None,
        })
    }
    fn first_spawn(&self) -> Option<(usize, &Exec)> {
        self.ops.iter().enumerate().find_map(|(i, o)| match o {
            Op::Spawn(e) => Some((i, e)),
            _ => None,
        })
    }
    fn methods(&self) -> Vec<&str> {
        self.requests.iter().map(|(m, _)| m.as_str()).collect()
    }
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

impl FakeSandbox {
    /// What the skills script prints, from what was uploaded and the image's
    /// own folder: one line per skill per target folder.
    fn skills_report(&self, script: &str) -> String {
        let w = self.world.lock().unwrap();
        let mut names = std::collections::BTreeSet::new();
        let mut out = String::new();
        if script.contains("if [ -d ") {
            match &self.setup.image_skills {
                Some(list) => names.extend(list.iter().map(|s| s.to_string())),
                None => out.push_str("BF skills-missing /skills\n"),
            }
        }
        for (local, remote) in w.uploads() {
            if remote == STAGING {
                names.extend(skills::names_in(&local));
            } else if let Some(n) = remote.strip_prefix(&format!("{STAGING}.run/")) {
                names.insert(n.to_string());
            }
        }
        let copy = format!("cp -R {STAGING}/. ");
        for line in script.lines() {
            if let Some(f) = line.strip_prefix(&copy) {
                let f = f.trim_end_matches('/');
                for n in &names {
                    out.push_str(&format!("BF skill {f} {n}\n"));
                }
            }
        }
        out
    }
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-files"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        let c = exec.command.clone();
        let reply = if c.contains("BF skill ") {
            ok(&self.skills_report(&c))
        } else if c.contains("useradd") {
            ok("BF user agent 1000 /home/agent\n")
        } else if c.contains("baseline")
            && c.contains("files.tar")
            && !c.contains("BF handoff done")
        {
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
            if self.setup.unplaced {
                let first = c.lines().find_map(|l| l.strip_prefix("place 0 "));
                let path = first.and_then(|l| l.rsplit_once(' ')).map(|(p, _)| p);
                ok(&format!(
                    "BF unplaced {}\nBF placement done\n",
                    path.unwrap_or_default()
                ))
            } else {
                ok("BF placement done\n")
            }
        } else if c.lines().last().map(str::trim) == Some("pwd") {
            ok("/app\n")
        } else {
            ok("")
        };
        self.world.lock().unwrap().ops.push(Op::Exec(exec));
        Ok(reply)
    }
    async fn spawn(&self, exec: Exec) -> Result<Process, SandboxError> {
        let headless = !exec.command.contains("fake-acp");
        let n = {
            let mut w = self.world.lock().unwrap();
            w.ops.push(Op::Spawn(exec));
            w.spawns += 1;
            w.spawns
        };
        if headless {
            return Ok(cli_process());
        }
        Ok(acp_process(self.world.clone(), self.setup.load_session, n))
    }
    async fn upload(&self, local: &Path, remote: &str) -> Result<(), SandboxError> {
        self.world
            .lock()
            .unwrap()
            .ops
            .push(Op::Upload(local.to_path_buf(), remote.to_string()));
        Ok(())
    }
    async fn write_file(&self, r: &str, c: &[u8], m: u32) -> Result<(), SandboxError> {
        self.world
            .lock()
            .unwrap()
            .ops
            .push(Op::Write(r.to_string(), c.to_vec(), m));
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote == "/logs/verifier" {
            std::fs::create_dir_all(local).unwrap();
            std::fs::write(local.join("reward.txt"), "1\n").unwrap();
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
fn acp_process(world: Arc<Mutex<World>>, load_session: bool, n: usize) -> Process {
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
                    "agentCapabilities": {"loadSession": load_session},
                    "authMethods": [], "agentInfo": {"name": "fake", "version": "1"}}))),
                "session/new" => out.push(reply(json!({"sessionId": session}))),
                "session/load" => {
                    session = msg["params"]["sessionId"].as_str().unwrap().to_string();
                    out.push(update(&session, "words replayed from the loaded session"));
                    out.push(reply(json!({})));
                }
                "session/prompt" => {
                    out.push(update(&session, "working on it"));
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
        let _ = out.write_all(b"I did the work.\n").await;
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
            setup: self.setup.clone(),
        }))
    }
}

/// An ACP agent that reads skills from its home and its working folder,
/// and loads Claude Code session files.
const ACP_MANIFEST: &str = r#"
name = "fake"
skill_paths = ["$HOME/.fake/skills", "$WORKSPACE/.agents/skills"]
[install]
script = "true"
[launch]
command = "/opt/fake/bin/fake-acp"
[model]
select = "none"
[resume]
acp = true
session_dirs = [".claude/projects"]
load = "claude-jsonl"
"#;

const HEADLESS_MANIFEST: &str = r#"
name = "fakeh"
skill_paths = ["$HOME/.fakeh/skills"]
[install]
script = "true"
[headless]
command = 'fake-cli ${BENCHFLOW_RESUME:+--continue} < "$BENCHFLOW_INSTRUCTION_FILE"'
output = "text"
[resume]
headless = true
session_dirs = [".claude/projects"]
load = "claude-jsonl"
"#;

fn manifest(text: &str) -> AgentManifest {
    AgentManifest::parse(text, ManifestSource::File(PathBuf::from("fake.toml"))).unwrap()
}

fn acp_agent() -> Arc<dyn ModelAgent> {
    let host = HostEnv::new(BTreeMap::new(), None);
    Arc::new(AcpAgent::new(manifest(ACP_MANIFEST), None, None, &host))
}

fn headless_agent() -> Arc<dyn ModelAgent> {
    let host = HostEnv::new(BTreeMap::new(), None);
    Arc::new(HeadlessAgent::new(
        manifest(HEADLESS_MANIFEST),
        None,
        None,
        &host,
    ))
}

/// An agent with no skill folders that cannot load sessions.
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
    async fn run(&self, _ctx: AgentContext<'_>) -> AgentOutput {
        AgentOutput::default()
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tasks")
        .join(name)
}

struct Run {
    rec: TrialRecord,
    dir: PathBuf,
    world: Arc<Mutex<World>>,
    _tmp: tempfile::TempDir,
}

impl Run {
    fn config(&self) -> Value {
        serde_json::from_str(&std::fs::read_to_string(self.dir.join("config.json")).unwrap())
            .unwrap()
    }
}

async fn run(task: &Path, attempt: Attempt, setup: Setup, options: RunOptions) -> Run {
    let tmp = tempfile::tempdir().unwrap();
    let task = Arc::new(benchflow_task::load(task).unwrap());
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
            idle_timeout: Duration::from_secs(20),
            ..options
        },
        events: Events::none(),
        stop: rx,
    });
    let spec = JobSpec {
        units: vec![(task, vec![attempt])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("files".into()),
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

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_tasks_skills_reach_every_skill_folder_owned_by_the_agent_user() {
    let r = run(
        &fixture("with-skills"),
        Attempt::Model(acp_agent()),
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    // The folder goes up whole, once, to the root-only staging folder.
    let uploads = w.uploads();
    let staged: Vec<_> = uploads
        .iter()
        .filter(|(_, r)| r.starts_with(STAGING))
        .collect();
    assert_eq!(staged.len(), 1, "{uploads:?}");
    assert!(staged[0].0.ends_with("with-skills/sandbox/skills"));
    assert_eq!(staged[0].1, STAGING);
    // Then into both folders, before the agent starts.
    let (at, script) = w.placement().expect("the skills were placed");
    let (spawned, _) = w.first_spawn().unwrap();
    assert!(at < spawned, "skills are in place before the agent starts");
    for line in [
        "mkdir -p /home/agent/.fake\nchown -h agent: /home/agent/.fake\n",
        "chown -h agent: /home/agent/.fake/skills\n",
        "cp -R /var/lib/benchflow/skills/. /home/agent/.fake/skills/\n",
        "chown -R -h agent: /home/agent/.fake/skills\n",
        "mkdir -p /app/.agents\nchown -h agent: /app/.agents\n",
        "cp -R /var/lib/benchflow/skills/. /app/.agents/skills/\n",
        "chown -R -h agent: /app/.agents/skills\n",
        "rm -rf /var/lib/benchflow/skills /var/lib/benchflow/skills.run\n",
    ] {
        assert!(script.contains(line), "{line}\n---\n{script}");
    }
    let skills = &r.config()["skills"];
    assert_eq!(skills["task"], "sandbox/skills");
    assert_eq!(skills["task_skills"], true);
    assert_eq!(skills["run"], json!([]));
    assert_eq!(
        skills["names"],
        json!(["csv-tools", "pdf-reading"]),
        "licenses/ holds no SKILL.md, so it is copied but is no skill"
    );
    assert_eq!(
        skills["placed_in"],
        json!(["/home/agent/.fake/skills", "/app/.agents/skills"])
    );
    assert_eq!(skills.get("why"), None);
}

/// A copy of the with-skills task in a temporary folder, with a skill
/// folder beside it (outside the task).
fn task_beside_a_folder() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("with-skills");
    let task = tmp.path().join("task");
    for entry in walk(&src) {
        let to = task.join(entry.strip_prefix(&src).unwrap());
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(&entry, &to).unwrap();
    }
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(outside.join("private")).unwrap();
    std::fs::write(
        outside.join("private/SKILL.md"),
        "---\nname: private\n---\n",
    )
    .unwrap();
    (tmp, task, outside)
}

async fn refused(task: &Path) {
    let r = run(
        task,
        Attempt::Model(acp_agent()),
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    let p = r.rec.problem.as_ref().expect("the trial stops");
    assert_eq!(p.code, "R107", "{p:?}");
    assert!(
        p.message.contains("outside the task's folder"),
        "{}",
        p.message
    );
    let w = r.world.lock().unwrap();
    assert!(
        !w.uploads().iter().any(|(_, r)| r.starts_with(STAGING)),
        "nothing outside the task is uploaded"
    );
}

/// Guards 5f4e1a45 (skills reach the agent): a task's skills path with `..`
/// was uploaded from outside the task.
#[tokio::test]
async fn a_skills_folder_outside_the_task_is_refused() {
    let (_tmp, task, _) = task_beside_a_folder();
    let md = task.join("task.md");
    let text = std::fs::read_to_string(&md).unwrap();
    std::fs::write(
        &md,
        text.replace("skills = \"sandbox/skills\"", "skills = \"../outside\""),
    )
    .unwrap();
    refused(&task).await;
}

#[cfg(unix)]
#[tokio::test]
async fn a_skills_folder_that_is_a_link_out_of_the_task_is_refused() {
    let (_tmp, task, outside) = task_beside_a_folder();
    std::fs::remove_dir_all(task.join("sandbox/skills")).unwrap();
    std::os::unix::fs::symlink(&outside, task.join("sandbox/skills")).unwrap();
    refused(&task).await;
}

#[tokio::test]
async fn a_run_skill_replaces_the_tasks_skill_of_the_same_name() {
    let extra = tempfile::tempdir().unwrap();
    for name in ["csv-tools", "sql"] {
        let d = extra.path().join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("SKILL.md"),
            format!("---\nname: {name}\n---\nnewer\n"),
        )
        .unwrap();
    }
    let skills = skills::resolve(&[extra.path().to_path_buf()]).unwrap();
    let r = run(
        &fixture("with-skills"),
        Attempt::Model(acp_agent()),
        Setup::default(),
        RunOptions {
            skills: skills.clone(),
            ..RunOptions::default()
        },
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    let uploads = w.uploads();
    for (dir, name) in skills.iter().zip(["csv-tools", "sql"]) {
        assert!(
            uploads
                .iter()
                .any(|(l, r)| l == dir && *r == format!("{STAGING}.run/{name}")),
            "{uploads:?}"
        );
    }
    let (_, script) = w.placement().unwrap();
    assert!(
        script.contains("rm -rf /var/lib/benchflow/skills/csv-tools; cp -R /var/lib/benchflow/skills.run/csv-tools /var/lib/benchflow/skills/csv-tools\n"),
        "{script}"
    );
    let rec = &r.config()["skills"];
    assert_eq!(rec["names"], json!(["csv-tools", "pdf-reading", "sql"]));
    assert_eq!(rec["run"], json!(skills));
}

#[tokio::test]
async fn without_the_tasks_skills_only_the_run_skills_are_installed() {
    let r = run(
        &fixture("with-skills"),
        Attempt::Model(acp_agent()),
        Setup::default(),
        RunOptions {
            task_skills: false,
            ..RunOptions::default()
        },
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    assert!(w.placement().is_none());
    assert!(!w.uploads().iter().any(|(_, r)| r.starts_with(STAGING)));
    let rec = &r.config()["skills"];
    assert_eq!(rec["task_skills"], false);
    assert_eq!(rec["why"], "--no-task-skills left out the task's skills");
}

#[tokio::test]
async fn an_agent_with_no_skill_folder_runs_without_them_and_the_record_says_so() {
    let r = run(
        &fixture("with-skills"),
        Attempt::Model(Arc::new(Plain)),
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert!(r.world.lock().unwrap().placement().is_none());
    assert_eq!(
        r.config()["skills"]["why"],
        "plain has no skills folder (its manifest names no skill_paths), so it got none of the skills"
    );
}

#[tokio::test]
async fn the_oracle_runs_without_skills() {
    let r = run(
        &fixture("with-skills"),
        Attempt::Oracle,
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert!(r.world.lock().unwrap().placement().is_none());
    assert_eq!(
        r.config()["skills"]["why"],
        "oracle runs without skills; only model agents get them"
    );
}

#[tokio::test]
async fn a_root_oracle_may_run_git_in_the_agent_owned_workdir() {
    let r = run(
        &fixture("hello"),
        Attempt::Oracle,
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    let execs = w.execs();
    let at = |needle: &str| execs.iter().position(|c| c.contains(needle));
    let chown = at("useradd").expect("the agent user owns the workdir");
    let trust = at("safe.directory").expect("root's git trusts the workdir");
    let solve = at("solve.sh").expect("the oracle ran");
    assert!(chown < trust && trust < solve, "{execs:#?}");
    assert!(
        execs[trust].contains("safe.directory /app"),
        "{}",
        execs[trust]
    );
    // The solution runs with HOME=/root; an image's ENV HOME must not send
    // the entry anywhere else.
    assert!(
        execs[trust].contains("HOME=/root git config --global"),
        "{}",
        execs[trust]
    );
}

#[tokio::test]
async fn a_task_without_skills_records_none() {
    let r = run(
        &fixture("hello"),
        Attempt::Model(acp_agent()),
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert!(r.world.lock().unwrap().placement().is_none());
    assert_eq!(r.config()["skills"], Value::Null);
}

/// A Harbor task whose image fills `skills_dir`: the skills are copied from
/// inside the sandbox.
fn harbor_task_with_skills_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let src = fixture("hello-split");
    for entry in walk(&src) {
        let rel = entry.strip_prefix(&src).unwrap();
        let to = tmp.path().join(rel);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(&entry, &to).unwrap();
    }
    let toml = tmp.path().join("task.toml");
    let text = std::fs::read_to_string(&toml).unwrap();
    std::fs::write(
        &toml,
        text.replace(
            "[environment]\n",
            "[environment]\nskills_dir = \"/skills\"\nworkdir = \"/app\"\n",
        ),
    )
    .unwrap();
    tmp
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

#[tokio::test]
async fn a_harbor_skills_dir_is_copied_from_inside_the_sandbox() {
    let task = harbor_task_with_skills_dir();
    let r = run(
        task.path(),
        Attempt::Model(acp_agent()),
        Setup {
            image_skills: Some(vec!["docx"]),
            ..Setup::default()
        },
        RunOptions::default(),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    assert!(!w.uploads().iter().any(|(_, r)| r.starts_with(STAGING)));
    let (_, script) = w.placement().unwrap();
    assert!(
        script.contains("if [ -d /skills ]; then cp -R /skills/. /var/lib/benchflow/skills/;"),
        "{script}"
    );
    let rec = &r.config()["skills"];
    assert_eq!(rec["task"], "/skills");
    assert_eq!(rec["names"], json!(["docx"]));
    assert_eq!(rec.get("why"), None);
}

#[tokio::test]
async fn a_missing_harbor_skills_dir_is_named_in_the_record() {
    let task = harbor_task_with_skills_dir();
    let r = run(
        task.path(),
        Attempt::Model(acp_agent()),
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert_eq!(
        r.config()["skills"]["why"],
        "the task's skills folder /skills is not in the image, so only the run's skills were installed"
    );
}

/// A real SkillsBench task (from a local checkout, when there is one):
/// `environment/skills/` reaches the agent's skill folders.
#[tokio::test]
async fn a_skillsbench_tasks_skills_reach_the_agent() {
    let task = Path::new("/tmp/bf-rs-corpus/skillsbench/tasks/3d-scan-calc");
    if !task.join("environment/skills").is_dir() {
        eprintln!("skipped: no SkillsBench checkout at {}", task.display());
        return;
    }
    let r = run(
        task,
        Attempt::Model(acp_agent()),
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    assert!(
        w.uploads()
            .iter()
            .any(|(l, r)| l.ends_with("3d-scan-calc/environment/skills") && r == STAGING)
    );
    let rec = &r.config()["skills"];
    assert_eq!(rec["task"], "environment/skills");
    assert_eq!(rec["names"], json!(["mesh-analysis"]));
    assert_eq!(rec["placed_in"][0], "/home/agent/.fake/skills");
}

// ---------------------------------------------------------------------------
// Loaded sessions
// ---------------------------------------------------------------------------

fn session_file(dir: &Path) -> PathBuf {
    let p = dir.join(format!("{SESSION_ID}.jsonl"));
    std::fs::write(
        &p,
        format!(
            "{{\"type\":\"user\",\"sessionId\":\"{SESSION_ID}\",\"message\":{{\"role\":\"user\",\"content\":\"earlier work\"}}}}\n"
        ),
    )
    .unwrap();
    p
}

fn loading(file: &Path) -> RunOptions {
    RunOptions {
        load_trajectory: Some(file.to_path_buf()),
        ..RunOptions::default()
    }
}

#[tokio::test]
async fn a_loaded_session_is_placed_for_the_cli_and_the_first_prompt_loads_it() {
    let tmp = tempfile::tempdir().unwrap();
    let file = session_file(tmp.path());
    let r = run(
        &fixture("with-skills"),
        Attempt::Model(acp_agent()),
        Setup {
            load_session: true,
            ..Setup::default()
        },
        loading(&file),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    let target = format!("/home/agent/.claude/projects/-app/{SESSION_ID}.jsonl");
    // Root writes the session only into a stage of its own, never into the
    // agent's home, where a link the agent planted could send it anywhere.
    let (staged, contents, mode) = w
        .ops
        .iter()
        .find_map(|o| match o {
            Op::Write(p, c, m) if c == &std::fs::read(&file).unwrap() => {
                Some((p.clone(), c.clone(), *m))
            }
            _ => None,
        })
        .expect("the session file was staged");
    assert!(staged.starts_with("/var/lib/benchflow-place/"), "{staged}");
    assert_eq!(contents, std::fs::read(&file).unwrap());
    assert_eq!(mode, 0o600);
    assert!(
        !w.ops
            .iter()
            .any(|o| matches!(o, Op::Write(p, _, _) if p.starts_with("/home/agent"))),
        "root wrote into the agent's home"
    );
    let stage = staged.rsplit_once('/').unwrap().0.to_string();
    let path = Some(benchflow_sandbox::shell::SYSTEM_PATH.to_string());
    let ran = |needle: &str| {
        w.ops.iter().find_map(|o| match o {
            Op::Exec(e) if e.command.contains(needle) => Some(e.clone()),
            _ => None,
        })
    };
    let given = ran("BF staged").expect("the staged file was given to the agent user");
    assert!(
        given
            .command
            .contains(&format!("chown -h agent:\"$g\" {staged}")),
        "{}",
        given.command
    );
    assert_eq!(given.user.as_deref(), Some("root"));
    assert_eq!(given.env.get("PATH"), path.as_ref());
    // The agent user copies it where Claude Code looks, finding its programs
    // in the system folders only.
    let copy = ran("BF placement done").expect("the agent user copied the file");
    assert!(
        copy.command.contains(&format!("place 0 {target} 600\n")),
        "{}",
        copy.command
    );
    assert_eq!(copy.user.as_deref(), Some("agent"));
    assert_eq!(copy.env.get("PATH"), path.as_ref());
    let cleanup = ran(&format!("rm -rf {stage}")).expect("the stage was removed");
    assert_eq!(cleanup.user.as_deref(), Some("root"));
    let loads: Vec<&Value> = w
        .requests
        .iter()
        .filter(|(m, _)| m == "session/load")
        .map(|(_, p)| p)
        .collect();
    assert_eq!(loads.len(), 1, "{:?}", w.methods());
    assert_eq!(loads[0]["sessionId"], SESSION_ID);
    assert_eq!(loads[0]["cwd"], "/app");
    assert!(!w.methods().contains(&"session/new"), "{:?}", w.methods());
    let rec = &r.config()["load_trajectory"];
    assert_eq!(rec["format"], "claude-jsonl");
    assert_eq!(rec["session_id"], SESSION_ID);
    assert_eq!(rec["placed_at"], target);
    assert_eq!(rec["resumed"], true);
    assert_eq!(rec["sha256"].as_str().unwrap().len(), 64);
    assert!(
        r.dir
            .join("loaded-session")
            .join(format!("{SESSION_ID}.jsonl"))
            .is_file()
    );
}

/// A file the agent user cannot write in its home (a folder on the way is
/// root's, or a link to one) stops the run before the agent starts, as the
/// task's problem; root writes nothing there in its place.
#[tokio::test]
async fn a_file_the_agent_user_cannot_write_stops_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    let file = session_file(tmp.path());
    let r = run(
        &fixture("with-skills"),
        Attempt::Model(acp_agent()),
        Setup {
            load_session: true,
            unplaced: true,
            ..Setup::default()
        },
        loading(&file),
    )
    .await;
    let p = r.rec.problem.as_ref().expect("a problem");
    assert_eq!(p.code, "R104", "{p:?}");
    assert_eq!(p.owner, benchflow_run::Owner::Task);
    let target = format!("/home/agent/.claude/projects/-app/{SESSION_ID}.jsonl");
    assert!(p.message.contains(&target), "{}", p.message);
    let w = r.world.lock().unwrap();
    assert!(w.first_spawn().is_none(), "the agent started");
    assert!(
        !w.ops
            .iter()
            .any(|o| matches!(o, Op::Write(p, _, _) if p.starts_with("/home/agent"))),
        "root wrote into the agent's home"
    );
    assert!(
        w.execs()
            .iter()
            .any(|c| c.contains("rm -rf /var/lib/benchflow-place/")),
        "the stage was left behind: {:#?}",
        w.execs()
    );
}

#[tokio::test]
async fn a_skills_folder_over_1_gib_is_refused_before_upload() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("big-skill");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("SKILL.md"), "---\nname: big\n---\n").unwrap();
    // Sparse: takes no room on disk.
    std::fs::File::create(dir.join("data.bin"))
        .unwrap()
        .set_len(1024 * 1024 * 1024 + 1)
        .unwrap();
    let r = run(
        &fixture("hello"),
        Attempt::Model(acp_agent()),
        Setup::default(),
        RunOptions {
            skills: vec![dir.clone()],
            ..RunOptions::default()
        },
    )
    .await;
    let p = r.rec.problem.unwrap();
    assert_eq!(p.code, "R107", "{p:?}");
    assert!(
        p.message.contains("holds more than 1024 MB"),
        "{}",
        p.message
    );
    assert!(
        !r.world
            .lock()
            .unwrap()
            .ops
            .iter()
            .any(|o| matches!(o, Op::Upload(p, _) if *p == dir)),
        "the folder was not uploaded"
    );
}

#[tokio::test]
async fn a_loaded_session_over_100_mb_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join(format!("{SESSION_ID}.jsonl"));
    std::fs::File::create(&file)
        .unwrap()
        .set_len(100 * 1024 * 1024 + 1)
        .unwrap();
    let r = run(
        &fixture("hello"),
        Attempt::Model(acp_agent()),
        Setup {
            load_session: true,
            ..Setup::default()
        },
        loading(&file),
    )
    .await;
    let p = r.rec.problem.unwrap();
    assert_eq!(p.code, "R107", "{p:?}");
    assert!(p.message.contains("larger than 100 MB"), "{}", p.message);
}

#[tokio::test]
async fn an_acp_agent_without_session_load_fails_rather_than_start_fresh() {
    let tmp = tempfile::tempdir().unwrap();
    let file = session_file(tmp.path());
    let r = run(
        &fixture("hello"),
        Attempt::Model(acp_agent()),
        Setup::default(),
        loading(&file),
    )
    .await;
    assert_eq!(r.rec.status, Status::Errored);
    let p = r.rec.problem.unwrap();
    assert_eq!(p.code, "R215", "{p:?}");
    assert!(
        p.message
            .contains("fake does not offer session/load, so it cannot resume the session from --load-trajectory"),
        "{}",
        p.message
    );
    assert!(!r.world.lock().unwrap().methods().contains(&"session/new"));
}

#[tokio::test]
async fn a_headless_agent_resumes_the_loaded_session_with_its_clis_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let file = session_file(tmp.path());
    let r = run(
        &fixture("hello"),
        Attempt::Model(headless_agent()),
        Setup::default(),
        loading(&file),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let w = r.world.lock().unwrap();
    let (_, cli) = w.first_spawn().unwrap();
    assert_eq!(
        cli.env.get("BENCHFLOW_RESUME").map(String::as_str),
        Some("1")
    );
    assert_eq!(r.config()["load_trajectory"]["resumed"], true);
}

#[tokio::test]
async fn only_the_first_stage_resumes_a_loaded_session_unless_asked() {
    let tmp = tempfile::tempdir().unwrap();
    let file = session_file(tmp.path());
    let r = run(
        &fixture("two-stage"),
        Attempt::Model(acp_agent()),
        Setup {
            load_session: true,
            ..Setup::default()
        },
        loading(&file),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let result: Value =
        serde_json::from_str(&std::fs::read_to_string(r.dir.join("result.json")).unwrap()).unwrap();
    let rows = result["step_results"].as_array().unwrap();
    assert_eq!(rows[0]["session"]["id"], SESSION_ID);
    assert_eq!(rows[0]["session"]["resumed"], true);
    assert_eq!(rows[1]["session"]["id"], "sess-2", "a fresh session");
    assert_eq!(rows[1]["session"]["resumed"], false);
}

#[tokio::test]
async fn an_agent_that_cannot_load_a_session_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let file = session_file(tmp.path());
    let r = run(
        &fixture("hello"),
        Attempt::Model(Arc::new(Plain)),
        Setup::default(),
        loading(&file),
    )
    .await;
    assert_eq!(r.rec.status, Status::Errored);
    let p = r.rec.problem.unwrap();
    assert_eq!(p.code, "R215");
    assert_eq!(
        p.message,
        "plain cannot load a session from a file on this harness"
    );
    assert!(r.world.lock().unwrap().first_spawn().is_none());
}

// ---------------------------------------------------------------------------
// Skills named to the CLI: $BENCHFLOW_SKILLS_DIR, a flag, a config key
// ---------------------------------------------------------------------------

/// A built-in agent on the headless harness, with a placeholder key.
fn builtin_headless(name: &str, model: &str, key: &str) -> Arc<dyn ModelAgent> {
    let host = HostEnv::new(
        BTreeMap::from([(key.to_string(), "placeholder-key-0123456789".to_string())]),
        None,
    );
    let m = benchflow_agent::Registry::builtin()
        .find(name)
        .unwrap()
        .clone();
    Arc::new(HeadlessAgent::new(m, Some(model), None, &host))
}

/// The agent's environment when it was spawned.
fn spawn_env(r: &Run) -> BTreeMap<String, String> {
    let w = r.world.lock().unwrap();
    w.first_spawn().expect("the agent started").1.env.clone()
}

#[tokio::test]
async fn the_skills_folder_is_named_to_the_agent_only_when_skills_were_placed() {
    let r = run(
        &fixture("with-skills"),
        Attempt::Model(builtin_headless(
            "kimi-code",
            "kimi/kimi-k2",
            "KIMI_API_KEY",
        )),
        Setup::default(),
        RunOptions::default(),
    )
    .await;
    // The fake CLI's output is not the agent's format, so the trial does not
    // score; what matters is what the agent was started with.
    assert_eq!(
        r.config()["skills"]["placed_in"],
        json!(["/home/agent/.kimi-code/skills"])
    );
    assert_eq!(
        spawn_env(&r).get(skills::DIR_ENV).map(String::as_str),
        Some("/home/agent/.kimi-code/skills")
    );
    // No skills: no variable, even when the run's --env sets one.
    let stray = BTreeMap::from([(skills::DIR_ENV.to_string(), "/elsewhere".to_string())]);
    for (task, task_skills) in [("hello", true), ("with-skills", false)] {
        let r = run(
            &fixture(task),
            Attempt::Model(builtin_headless(
                "mistral-vibe",
                "mistral-small",
                "MISTRAL_API_KEY",
            )),
            Setup::default(),
            RunOptions {
                task_skills,
                agent_env: stray.clone(),
                ..RunOptions::default()
            },
        )
        .await;
        assert_eq!(spawn_env(&r).get(skills::DIR_ENV), None, "{task}");
    }
}

/// Run a built-in agent's headless command (or its ACP launch command, for
/// `acp:<name>`) with a local `sh`, its
/// `/opt/benchflow` moved under `root` (where stub CLIs print their
/// arguments), and `env` added; the stubs' output.
fn run_headless_command(agent: &str, root: &Path, env: &[(&str, &str)]) -> String {
    let (acp, agent) = match agent.strip_prefix("acp:") {
        Some(a) => (true, a),
        None => (false, agent),
    };
    let m = benchflow_agent::Registry::builtin()
        .find(agent)
        .unwrap()
        .clone();
    let command = if acp {
        m.launch.command.clone().unwrap()
    } else {
        m.headless.as_ref().unwrap().command.clone()
    };
    let command = command.replace("/opt/benchflow/", &format!("{}/opt/", root.display()));
    let home = root.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let instruction = root.join("instruction.md");
    std::fs::write(&instruction, "Do the task.").unwrap();
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", &home)
        .env("BENCHFLOW_INSTRUCTION_FILE", &instruction)
        .envs(env.iter().copied())
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stub(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        "#!/bin/sh\nprintf 'ARGS:'; printf ' [%s]' \"$@\"; echo\n",
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn kimi_code_gets_skills_dir_only_when_skills_were_delivered() {
    let tmp = tempfile::tempdir().unwrap();
    stub(&tmp.path().join("opt/bin/kimi"));
    let model = ("BENCHFLOW_MODEL", "openai/some-model");
    let with = run_headless_command(
        "kimi-code",
        tmp.path(),
        &[model, (skills::DIR_ENV, "/home/agent/.kimi-code/skills")],
    );
    assert!(
        with.contains("ARGS: [--skills-dir] [/home/agent/.kimi-code/skills] [--prompt]"),
        "{with}"
    );
    let without = run_headless_command("kimi-code", tmp.path(), &[model]);
    assert!(without.contains("ARGS: [--prompt]"), "{without}");
    assert!(!without.contains("--skills-dir"), "{without}");
}

#[test]
fn mistral_vibe_config_lists_the_skills_folder_only_when_skills_were_delivered() {
    let provider_config =
        "enable_telemetry = false\nactive_model = \"m\"\n\n[[providers]]\nname = \"benchflow\"\n";
    for config in [None, Some(provider_config)] {
        let tmp = tempfile::tempdir().unwrap();
        stub(&tmp.path().join("opt/agents/mistral-vibe/bin/vibe"));
        let toml_at = tmp.path().join("home/.vibe/config.toml");
        let mut env = vec![("BENCHFLOW_MODEL", "mistral-small")];
        if let Some(c) = config {
            env.push(("BENCHFLOW_VIBE_CONFIG", c));
        }
        let out = run_headless_command("mistral-vibe", tmp.path(), &env);
        assert!(out.contains("ARGS: [--auto-approve]"), "{out}");
        let text = std::fs::read_to_string(&toml_at).unwrap();
        assert!(!text.contains("skill_paths"), "{text}");

        env.push((skills::DIR_ENV, "/home/agent/.vibe/skills"));
        run_headless_command("mistral-vibe", tmp.path(), &env);
        let text = std::fs::read_to_string(&toml_at).unwrap();
        // First, so it is a top-level key and not part of the last table.
        assert!(
            text.starts_with("skill_paths = [\"/home/agent/.vibe/skills\"]\n"),
            "{text}"
        );
        let doc: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(
            doc["skill_paths"],
            toml::Value::Array(vec!["/home/agent/.vibe/skills".into()])
        );
        assert!(doc.get("active_model").is_some(), "{text}");
    }
    // Over ACP, the config.toml written on a provider gets the key too.
    let tmp = tempfile::tempdir().unwrap();
    stub(&tmp.path().join("opt/bin/mistral-vibe"));
    let toml_at = tmp.path().join("home/.vibe/config.toml");
    let config = ("BENCHFLOW_VIBE_CONFIG", provider_config);
    let out = run_headless_command("acp:mistral-vibe", tmp.path(), &[config]);
    assert!(out.contains("ARGS:"), "{out}");
    assert!(
        !std::fs::read_to_string(&toml_at)
            .unwrap()
            .contains("skill_paths")
    );
    let dir = (skills::DIR_ENV, "/home/agent/.vibe/skills");
    run_headless_command("acp:mistral-vibe", tmp.path(), &[config, dir]);
    let doc: toml::Value = toml::from_str(&std::fs::read_to_string(&toml_at).unwrap()).unwrap();
    assert_eq!(
        doc["skill_paths"],
        toml::Value::Array(vec!["/home/agent/.vibe/skills".into()])
    );
}
