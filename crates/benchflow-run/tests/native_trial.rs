//! Whole trials of the native agents through the real engine, on an
//! in-memory sandbox, against a fake model server.
//!
//! The model server is a local HTTP server speaking OpenAI-compatible chat
//! completions, answering each request with the next scripted reply (tool
//! calls for `builtin`, JSON keystrokes for Terminus-2). The sandbox keeps a
//! file table: it runs the builtin tools' write and read commands, emulates
//! the few tmux commands Terminus-2 sends, and grades `fixtures/hello` by
//! the content of `/app/hello.txt`.
//!
//! The tests run on several threads: building a native agent on the gateway
//! blocks one while it lists the model's upstreams from the fake server, which
//! serves them like the gateway does, Baseten among them.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::Json;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use base64::Engine;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::native::ClientOptions;
use benchflow_run::agent::{Attempt, ModelAgent};
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::native_agent::NativeAgent;
use benchflow_run::outcome::Status;
use benchflow_run::record::TrialRecord;
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, Provider, ProviderStatus, Sandbox,
    SandboxError, SandboxSpec,
};
use serde_json::{Value, json};

const KEY: &str = "vck_test_key_0123456789abcdef";

// ---------------------------------------------------------------- model --

#[derive(Default)]
struct Script {
    replies: VecDeque<(u16, Value)>,
    /// Answer every request with this once the script runs out.
    repeat: Option<Value>,
    requests: Vec<Value>,
    bearer_ok: bool,
    /// Take every request and never answer it.
    stall: bool,
}

type Shared = Arc<Mutex<Script>>;

async fn completions(
    State(s): State<Shared>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    if s.lock().unwrap().stall {
        s.lock().unwrap().requests.push(body.clone());
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }
    let mut s = s.lock().unwrap();
    s.bearer_ok = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == format!("Bearer {KEY}"));
    s.requests.push(body);
    let (status, reply) = match s.replies.pop_front() {
        Some(r) => r,
        None => (200, s.repeat.clone().unwrap_or_else(|| text_reply("done"))),
    };
    (StatusCode::from_u16(status).unwrap(), Json(reply))
}

/// The gateway's list of upstreams serving a model: Baseten among them, as
/// for the real deepseek-v4-flash, and a model only Baseten serves.
async fn endpoints(
    axum::extract::Path(path): axum::extract::Path<String>,
) -> (StatusCode, Json<Value>) {
    let names: &[&str] = match path.as_str() {
        "fake/model/endpoints" => &["baseten", "novita", "alibaba"],
        "fake/only-baseten/endpoints" => &["baseten"],
        _ => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "no such model"})),
            );
        }
    };
    let endpoints: Vec<Value> = names
        .iter()
        .map(|n| json!({"provider_name": n, "name": n}))
        .collect();
    (
        StatusCode::OK,
        Json(json!({"data": {"endpoints": endpoints}})),
    )
}

async fn serve(script: Shared) -> String {
    let app = axum::Router::new()
        .route("/v1/chat/completions", post(completions))
        .route("/v1/models/{*path}", get(endpoints))
        // Read a request that carries a huge earlier reply whole, as a
        // provider does; axum's default 2 MB limit would refuse it.
        .layer(DefaultBodyLimit::disable())
        .with_state(script);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}/v1")
}

fn usage() -> Value {
    json!({"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001})
}

fn tool_reply(id: &str, name: &str, args: Value) -> Value {
    json!({"model": "fake/model", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": id, "type": "function",
                        "function": {"name": name, "arguments": args.to_string()}}]}}],
        "usage": usage()})
}

fn text_reply(text: &str) -> Value {
    json!({"model": "fake/model", "choices": [{"finish_reason": "stop",
        "message": {"role": "assistant", "content": text}}], "usage": usage()})
}

// -------------------------------------------------------------- sandbox --

#[derive(Default)]
struct World {
    files: BTreeMap<String, Vec<u8>>,
    commands: Vec<(String, Option<String>)>,
    written: Vec<String>,
    /// What the tmux pane shows.
    screen: String,
}

struct FakeSandbox {
    world: Arc<Mutex<World>>,
}

fn ok(stdout: impl Into<String>) -> ExecOutput {
    ExecOutput {
        exit_code: 0,
        stdout: stdout.into(),
        stderr: String::new(),
        timed_out: false,
        duration: Duration::from_millis(1),
    }
}

/// The shell words of `command`.
fn words(command: &str) -> Vec<String> {
    shlex::split(command).unwrap_or_default()
}

impl FakeSandbox {
    fn run(&self, c: &str) -> ExecOutput {
        let mut w = self.world.lock().unwrap();
        // The engine's own setup and teardown. The fixture's image builds
        // `WORKDIR /app`, which a task without a declared workdir works in.
        if c.lines().last() == Some("pwd") {
            return ok("/app\n");
        }
        if c.contains("useradd") {
            return ok("BF user agent 1000 /home/agent\n");
        }
        if c.contains("kill -9 -1") {
            return ok("BF killed 1\n");
        }
        if c.contains("ld.so.preload") {
            return ok("BF preload absent\n");
        }
        if c.contains("BF venv-done") {
            return ok(
                "BF venv-safepath 1\nBF venv-path /usr/local/bin:/usr/bin:/bin\nBF venv-done\n",
            );
        }
        if c.contains("BF path") {
            return ok("BF path /usr/local/bin:/usr/bin:/bin\n");
        }
        // The handoff program confirms it ran to the end.
        if c.contains("BF baseline $(") {
            return ok("BF baseline 0\n");
        }
        if c.contains("BF handoff done") {
            return ok("BF handoff done\n");
        }
        if c.contains("BF owned") {
            return ok("BF owned root\n");
        }
        if c.contains("BF stage-ready") {
            return ok("BF stage-ready\n");
        }
        if c.contains("BF staged") {
            return ok("BF staged\n");
        }
        if c.contains("BF placement done") {
            return ok("BF placement done\n");
        }
        // builtin: `printf %s <b64> | base64 -d > <path>` writes a file.
        if let Some(rest) = c.split("printf %s ").nth(1)
            && c.contains("| base64 -d > ")
        {
            let parts = words(rest);
            let data = base64::engine::general_purpose::STANDARD
                .decode(&parts[0])
                .unwrap();
            let path = parts.last().unwrap().clone();
            w.files.insert(path, data);
            return ok("");
        }
        // builtin: `view` reads lines with sed.
        if c.contains("sed -n") {
            let path = words(c.split("sed -n").nth(1).unwrap())[1].clone();
            return match w.files.get(&path) {
                Some(bytes) => ok(format!("{}\n", String::from_utf8_lossy(bytes))),
                None => ExecOutput {
                    exit_code: 3,
                    stderr: "BF_NOT_FOUND\n".into(),
                    ..ok("")
                },
            };
        }
        // Terminus-2's tmux.
        if c.starts_with("tmux send-keys") {
            let keys = words(c)[5..].join("");
            if keys.contains("printf 'Hello, world!' > /app/hello.txt") {
                w.files
                    .insert("/app/hello.txt".into(), b"Hello, world!".to_vec());
            }
            w.screen.push_str(&keys);
            w.screen.push_str("root@sandbox:/app# ");
            return ok("");
        }
        if c.starts_with("tmux capture-pane") {
            return ok(format!("root@sandbox:/app# {}", w.screen));
        }
        ok("")
    }
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-native"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        // Agent commands start by raising their out-of-memory score; record
        // and answer the command itself.
        let prefix = benchflow_run::hardening::oom_prefix();
        let command = exec
            .command
            .strip_prefix(prefix.as_str())
            .unwrap_or(&exec.command)
            .to_string();
        self.world
            .lock()
            .unwrap()
            .commands
            .push((command.clone(), exec.user.clone()));
        Ok(self.run(&command))
    }
    async fn spawn(&self, _exec: Exec) -> Result<Process, SandboxError> {
        Err(SandboxError::Unsupported(
            "native agents spawn nothing".into(),
        ))
    }
    async fn upload(&self, _local: &Path, _remote: &str) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn write_file(&self, remote: &str, _c: &[u8], _mode: u32) -> Result<(), SandboxError> {
        self.world.lock().unwrap().written.push(remote.to_string());
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote == "/logs/verifier" {
            std::fs::create_dir_all(local).unwrap();
            let w = self.world.lock().unwrap();
            let solved = w.files.get("/app/hello.txt").map(Vec::as_slice)
                == Some(b"Hello, world!".as_slice());
            std::fs::write(local.join("reward.txt"), if solved { "1\n" } else { "0\n" }).unwrap();
            // The fixture's rubric grades from the test report.
            let status = if solved { "passed" } else { "failed" };
            let ctrf = json!({"results": {"tool": {"name": "test.sh"},
                "tests": [{"name": "test_hello", "status": status}]}});
            std::fs::write(local.join("ctrf.json"), ctrf.to_string()).unwrap();
        } else if remote.ends_with(".tar") {
            // An archive arrives as a file, as from a real backend: here
            // an empty one (its end-of-archive blocks).
            std::fs::write(local, [0u8; 1024]).unwrap();
        } else if remote.ends_with(".pane") {
            // Terminus's pane log is a regular file too (review F2-B-r11
            // L3): an empty one here.
            std::fs::write(local, b"").unwrap();
        }
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        Ok(())
    }
}

struct FakeProvider {
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
        Ok(Box::new(FakeSandbox { world }))
    }
}

// ---------------------------------------------------------------- trial --

fn hello() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello")
}

struct Run {
    rec: TrialRecord,
    dir: PathBuf,
    world: Arc<Mutex<World>>,
    script: Shared,
    _tmp: tempfile::TempDir,
}

async fn run_trial(agent: &str, script: Script, options: RunOptions) -> Run {
    run_trial_on(&hello(), agent, script, options).await
}

/// [`run_trial`] on another task folder.
async fn run_trial_on(task_dir: &Path, agent: &str, script: Script, options: RunOptions) -> Run {
    let script: Shared = Arc::new(Mutex::new(script));
    let base = serve(script.clone()).await;
    let host = HostEnv::new(
        BTreeMap::from([
            ("AI_GATEWAY_API_KEY".to_string(), KEY.to_string()),
            ("BENCHFLOW_NATIVE_BASE_URL".to_string(), base),
        ]),
        None,
    );
    let agent = NativeAgent::new(agent, Some("vercel/fake/model"), None, &host)
        .expect("a native agent")
        .client_options(ClientOptions {
            timeout: Duration::from_secs(10),
            max_attempts: 3,
            backoff_base: Duration::from_millis(5),
            backoff_cap: Duration::from_millis(20),
            ..ClientOptions::default()
        });
    let tmp = tempfile::tempdir().unwrap();
    let task = Arc::new(benchflow_task::load(task_dir).expect("the task fixture loads"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
    });
    let (tx, rx) = stop_switch();
    std::mem::forget(tx);
    let rt = Arc::new(Runtime {
        provider: provider.clone(),
        images: Images::new(),
        options,
        events: Events::none(),
        stop: rx,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(task, vec![Attempt::Model(Arc::new(agent))])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("native-job".into()),
        fresh: true,
        concurrency: 1,
        source: json!({"path": task_dir}),
    };
    let out = job::run(spec, rt).await.unwrap();
    let rec = out.records[0].clone();
    let dir = out.dir.join(&rec.rollout_name);
    let world = provider.worlds.lock().unwrap()[0].clone();
    Run {
        rec,
        dir,
        world,
        script,
        _tmp: tmp,
    }
}

fn options() -> RunOptions {
    RunOptions {
        idle_timeout: Duration::from_secs(30),
        ..RunOptions::default()
    }
}

/// No file in the trial folder holds the key.
fn assert_no_key(dir: &Path) {
    for entry in walk(dir) {
        let text = std::fs::read_to_string(&entry).unwrap_or_default();
        assert!(!text.contains(KEY), "{} holds the key", entry.display());
    }
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

/// A run variable holding the endpoint's key (`--env AI_GATEWAY_API_KEY=...`,
/// the hint the native loops gave until 820ee045) never reaches the
/// sandbox's commands; the other run variables do.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_variable_holding_the_key_stays_out_of_the_sandbox() {
    let script = Script {
        replies: VecDeque::from([(
            200,
            tool_reply("c1", "submit", json!({"summary": "nothing to do"})),
        )]),
        ..Script::default()
    };
    let mut options = options();
    options.agent_env = BTreeMap::from([
        ("AI_GATEWAY_API_KEY".to_string(), KEY.to_string()),
        ("TASK_HINT".to_string(), "kept".to_string()),
        ("AUTH_HEADER".to_string(), format!("Bearer {KEY}")),
    ]);
    let r = run_trial("builtin", script, options).await;
    assert!(
        r.rec.env_keys.contains(&"TASK_HINT".to_string()),
        "{:?}",
        r.rec.env_keys
    );
    for held in ["AI_GATEWAY_API_KEY", "AUTH_HEADER"] {
        assert!(
            !r.rec.env_keys.contains(&held.to_string()),
            "{:?}",
            r.rec.env_keys
        );
    }
    assert_no_key(&r.dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn builtin_solves_hello_with_scripted_tool_calls() {
    let script = Script {
        replies: VecDeque::from([
            (
                200,
                tool_reply(
                    "c1",
                    "write",
                    json!({"file_path": "/app/hello.txt", "content": "Hello, world!"}),
                ),
            ),
            (
                200,
                tool_reply(
                    "c2",
                    "view",
                    json!({"file_path": "hello.txt", "from_line": 1, "to_line": 5}),
                ),
            ),
            (
                200,
                tool_reply("c3", "submit", json!({"summary": "wrote the file"})),
            ),
        ]),
        ..Script::default()
    };
    let r = run_trial("builtin", script, options()).await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert_eq!(r.rec.reward, Some(1.0));
    assert_eq!(r.rec.agent, "builtin");
    assert_eq!(r.rec.model.as_deref(), Some("vercel/fake/model"));
    assert_eq!(r.rec.tool_calls, 3);
    assert_eq!(
        (r.rec.usage.input_tokens, r.rec.usage.output_tokens),
        (30, 15)
    );
    assert!((r.rec.usage.cost_usd.unwrap() - 0.003).abs() < 1e-9);
    assert_eq!(r.rec.usage.source, "native");
    assert_eq!(r.rec.trajectory_summary["trajectory_source"], "native");
    assert_eq!(r.rec.trajectory_summary["tool_call_steps"], 3);

    // The model saw the tools, the instruction, and each result.
    let script = r.script.lock().unwrap();
    assert!(script.bearer_ok, "the key went in the Authorization header");
    assert_eq!(script.requests.len(), 3);
    let first = &script.requests[0];
    assert_eq!(first["model"], "fake/model");
    let tools: Vec<&str> = first["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(tools, ["bash", "view", "replace", "write", "submit"]);
    assert!(
        first["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("/app/hello.txt")
    );
    let third = &script.requests[2]["messages"];
    let view_result = third
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|m| m["role"] == "tool")
        .unwrap();
    assert!(
        view_result["content"]
            .as_str()
            .unwrap()
            .contains("Hello, world!"),
        "{view_result}"
    );
    drop(script);

    // Tools ran as the agent user; nothing was written as a login.
    let world = r.world.lock().unwrap();
    assert!(
        world.written.is_empty(),
        "no login files: {:?}",
        world.written
    );
    let write = world
        .commands
        .iter()
        .find(|(c, _)| c.contains("base64 -d >"))
        .unwrap();
    assert_eq!(write.1.as_deref(), Some("agent"));
    drop(world);

    let trajectory =
        std::fs::read_to_string(r.dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(trajectory.contains("write /app/hello.txt"), "{trajectory}");
    assert!(
        trajectory.contains("\"tool_name\": \"write\""),
        "{trajectory}"
    );
    let calls = std::fs::read_to_string(r.dir.join("agent/model_calls.jsonl")).unwrap();
    assert_eq!(calls.lines().count(), 3);
    let line: Value = serde_json::from_str(calls.lines().next().unwrap()).unwrap();
    assert_eq!(line["status"], 200);
    assert_eq!(line["request"]["model"], "fake/model");
    // Every gateway request named the allowed upstreams: never Baseten.
    assert_eq!(
        line["request"]["providerOptions"],
        json!({"gateway": {"only": ["novita", "alibaba"]}})
    );
    assert_no_key(&r.dir);
}

/// E1 round 2 (and E2's comparison with another harness's transcript): the
/// builtin loop's ATIF is what the model saw. The system prompt is a step,
/// each model call is one agent step with its usage as `metrics`, and a
/// tool result is recorded with the "Time remaining" note the model got.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn builtin_atif_is_one_step_per_model_call_with_what_the_model_saw() {
    let script = Script {
        replies: VecDeque::from([
            (
                200,
                tool_reply(
                    "c1",
                    "write",
                    json!({"file_path": "/app/hello.txt", "content": "Hello, world!"}),
                ),
            ),
            (
                200,
                tool_reply("c2", "submit", json!({"summary": "wrote the file"})),
            ),
        ]),
        ..Script::default()
    };
    let r = run_trial("builtin", script, options()).await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let script = r.script.lock().unwrap();
    let system = script.requests[0]["messages"][0]["content"]
        .as_str()
        .unwrap()
        .to_string();
    let second = script.requests[1]["messages"].as_array().unwrap();
    let seen = second.last().unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    drop(script);
    assert!(seen.contains("Time remaining:"), "{seen}");

    let atif: Value =
        serde_json::from_str(&std::fs::read_to_string(r.dir.join("trainer/atif.json")).unwrap())
            .unwrap();
    let steps = atif["steps"].as_array().unwrap();
    let sources: Vec<&str> = steps
        .iter()
        .map(|s| s["source"].as_str().unwrap())
        .collect();
    assert_eq!(sources, ["system", "user", "agent", "agent"], "{atif}");
    assert_eq!(steps[0]["message"], system.as_str());
    for step in &steps[2..] {
        assert_eq!(step["llm_call_count"], 1, "{step}");
        assert_eq!(
            step["metrics"],
            json!({"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0, "cost_usd": 0.001})
        );
        assert_eq!(step["extra"], json!({"finish_reason": "tool_calls"}));
    }
    // The write's result as the model read it, note included.
    assert_eq!(
        steps[2]["observation"]["results"][0]["content"],
        seen.as_str()
    );
    assert_eq!(atif["final_metrics"]["total_prompt_tokens"], 20);
    // The agent's version is BenchFlow's, which runs the loop.
    assert_eq!(atif["agent"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(
        r.rec.agent_version.as_deref(),
        Some(env!("CARGO_PKG_VERSION"))
    );
    // The model calls and the system prompt are not steps in result.json.
    assert_eq!(r.rec.trajectory_summary["other_steps"], 0);
}

/// E1 round 2: Terminus-2's ATIF has one agent step per model call, with
/// the turn's commands as its tool calls, as other ATIF writers do.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminus_2_atif_is_one_step_per_model_call() {
    let script = Script {
        replies: VecDeque::from([
            (
                200,
                keystrokes_reply(
                    "Need to write the file.",
                    &[
                        "printf 'Hello, world!' > /app/hello.txt\n",
                        "cat /app/hello.txt\n",
                    ],
                    true,
                ),
            ),
            (200, keystrokes_reply("The file is written.", &[], true)),
        ]),
        ..Script::default()
    };
    let r = run_trial("terminus-2", script, options()).await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let atif: Value =
        serde_json::from_str(&std::fs::read_to_string(r.dir.join("trainer/atif.json")).unwrap())
            .unwrap();
    let steps = atif["steps"].as_array().unwrap();
    let sources: Vec<&str> = steps
        .iter()
        .map(|s| s["source"].as_str().unwrap())
        .collect();
    assert_eq!(sources, ["user", "agent", "agent"], "{atif}");
    let first = &steps[1];
    assert!(
        first["message"]
            .as_str()
            .unwrap()
            .starts_with("Analysis: Need to write the file.")
    );
    // Both commands and the completion mark, one step.
    assert_eq!(first["tool_calls"].as_array().unwrap().len(), 3, "{first}");
    assert_eq!(first["llm_call_count"], 1);
    assert_eq!(first["metrics"]["prompt_tokens"], 10);
    assert_eq!(first["metrics"]["completion_tokens"], 5);
    assert_eq!(atif["final_metrics"]["total_steps"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_gateway_request_is_pinned_and_a_baseten_only_model_never_starts() {
    let script: Shared = Arc::new(Mutex::new(Script {
        repeat: Some(text_reply("done")),
        ..Script::default()
    }));
    let base = serve(script.clone()).await;
    let host = HostEnv::new(
        BTreeMap::from([
            ("AI_GATEWAY_API_KEY".to_string(), KEY.to_string()),
            ("BENCHFLOW_NATIVE_BASE_URL".to_string(), base),
            // The user's own denylist adds to Baseten.
            (
                "BENCHFLOW_DENY_UPSTREAMS".to_string(),
                "alibaba".to_string(),
            ),
        ]),
        None,
    );
    let pinned = NativeAgent::new("builtin", Some("vercel/fake/model"), None, &host).unwrap();
    pinned.preflight().unwrap();
    // A model only Baseten serves stops before any sandbox, naming both.
    let refused =
        NativeAgent::new("builtin", Some("vercel/fake/only-baseten"), None, &host).unwrap();
    let problem = refused.preflight().unwrap_err();
    assert_eq!(problem.code, "R215", "{problem:?}");
    assert_eq!(
        problem.message,
        "no upstream vercel could send `fake/only-baseten` to is allowed: it is served only by baseten, which BenchFlow does not use"
    );
    // So does one the gateway does not list.
    let unknown = NativeAgent::new("builtin", Some("vercel/fake/unknown"), None, &host).unwrap();
    assert_eq!(unknown.preflight().unwrap_err().code, "R215");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_tool_calls_stops_the_loop_and_the_trajectory_says_so() {
    let script = Script {
        repeat: Some(tool_reply("again", "bash", json!({"command": "echo step"}))),
        ..Script::default()
    };
    let r = run_trial(
        "builtin",
        script,
        RunOptions {
            max_tool_calls: Some(2),
            ..options()
        },
    )
    .await;
    // A budget stop is the agent's limit, not a failure: the work is graded.
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert!(r.rec.problem.is_none());
    assert_eq!(r.rec.reward, Some(0.0));
    assert_eq!(r.rec.tool_calls, 2);
    assert_eq!(r.script.lock().unwrap().requests.len(), 3);
    let trajectory =
        std::fs::read_to_string(r.dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(
        trajectory.contains("stopped by --max-tool-calls 2"),
        "{trajectory}"
    );
    assert!(trajectory.contains("asked for tool call 3"), "{trajectory}");
    let echoes = r
        .world
        .lock()
        .unwrap()
        .commands
        .iter()
        .filter(|(c, _)| c == "echo step")
        .count();
    assert_eq!(echoes, 2);
}

/// A model that answers every turn with tool calls whose arguments are not
/// JSON gets an error result each time; after three such turns in a row the
/// loop stops. The trial is still graded (DECISIONS row 27), and its summary
/// records the stop as the model's failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn endless_invalid_tool_calls_stop_the_loop_and_the_trial_is_graded() {
    let bad = json!({"model": "fake/model", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "bad", "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\": "}}]}}],
        "usage": usage()});
    let script = Script {
        repeat: Some(bad),
        ..Script::default()
    };
    let r = run_trial(
        "builtin",
        script,
        RunOptions {
            // hello's 2-minute agent limit becomes 6 s, so a loop without
            // the cap fails this test quickly by timing out instead.
            timeout_multiplier: 0.05,
            ..options()
        },
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert!(r.rec.problem.is_none(), "{:?}", r.rec.problem);
    assert_eq!(r.rec.reward, Some(0.0));
    assert!(!r.rec.timed_out);
    assert_eq!(r.script.lock().unwrap().requests.len(), 4);
    let stop = &r.rec.trajectory_summary["stop"];
    assert_eq!(stop["owner"], "model", "{stop}");
    assert_eq!(stop["turns"], 4, "{stop}");
    assert!(
        stop["reason"]
            .as_str()
            .unwrap()
            .contains("not JSON 4 turns in a row"),
        "{stop}"
    );
    let result: Value =
        serde_json::from_str(&std::fs::read_to_string(r.dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["trajectory_summary"]["stop"]["owner"], "model");
    let trajectory =
        std::fs::read_to_string(r.dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(
        trajectory.contains("not JSON 4 turns in a row"),
        "{trajectory}"
    );
    assert!(trajectory.contains("the model's failure"), "{trajectory}");
}

/// Review E2-C-A: in a task with stages, a stage's invalid-tool-call stop
/// stays in the record after a later stage ends normally: on that stage's
/// row of `step_results`, and in the trial's `trajectory_summary` (`stop`,
/// the first, and `stops`, every one with its stage).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_earlier_stages_invalid_call_stop_stays_in_the_record() {
    let bad = json!({"model": "fake/model", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "bad", "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\": "}}]}}],
        "usage": usage()});
    // The two-stage fixture without its gate, so stage two runs whatever
    // stage one scored.
    let tmp = tempfile::tempdir().unwrap();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/two-stage");
    let task = tmp.path().join("two-stage");
    copy_tree(&src, &task);
    let md = std::fs::read_to_string(task.join("task.md")).unwrap();
    std::fs::write(task.join("task.md"), md.replace("gate = 1.0\n", "")).unwrap();
    let script = Script {
        replies: VecDeque::from(vec![(200, bad); 4]),
        repeat: Some(text_reply("done")),
        ..Script::default()
    };
    let r = run_trial_on(&task, "builtin", script, options()).await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    let rows = r.rec.step_results.clone().expect("stage rows");
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert_eq!(rows[0]["stop"]["owner"], "model", "{rows:?}");
    assert!(rows[1].get("stop").is_none(), "{rows:?}");
    let summary = &r.rec.trajectory_summary;
    assert_eq!(summary["stop"]["stage"], "first", "{summary}");
    assert_eq!(
        summary["stops"].as_array().map(Vec::len),
        Some(1),
        "{summary}"
    );
    let result: Value =
        serde_json::from_str(&std::fs::read_to_string(r.dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["trajectory_summary"]["stop"]["owner"], "model");
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_tree(&e.path(), &dest);
        } else {
            std::fs::copy(e.path(), dest).unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rejected_key_is_r212_and_a_missing_model_is_r215() {
    let script = Script {
        replies: VecDeque::from([(401, json!({"error": {"message": "Invalid API key"}}))]),
        ..Script::default()
    };
    let r = run_trial("builtin", script, options()).await;
    let p = r.rec.problem.expect("a problem");
    assert_eq!(p.code, "R212", "{p:?}");
    assert!(p.message.contains("Invalid API key"), "{}", p.message);
    assert_eq!(
        r.script.lock().unwrap().requests.len(),
        1,
        "a 401 is not retried"
    );

    let script = Script {
        replies: VecDeque::from([(
            404,
            json!({"error": {"message": "model_not_found: fake/model"}}),
        )]),
        ..Script::default()
    };
    let r = run_trial("builtin", script, options()).await;
    assert_eq!(r.rec.problem.expect("a problem").code, "R215");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_errors_are_retried() {
    let script = Script {
        replies: VecDeque::from([
            (500, json!({"error": {"message": "overloaded"}})),
            (429, json!({"error": {"message": "rate limit"}})),
            (
                200,
                tool_reply(
                    "c1",
                    "write",
                    json!({"file_path": "/app/hello.txt", "content": "Hello, world!"}),
                ),
            ),
        ]),
        ..Script::default()
    };
    let r = run_trial("builtin", script, options()).await;
    assert_eq!(r.rec.reward, Some(1.0), "{:?}", r.rec.problem);
    let calls = std::fs::read_to_string(r.dir.join("agent/model_calls.jsonl")).unwrap();
    let statuses: Vec<i64> = calls
        .lines()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["status"]
                .as_i64()
                .unwrap()
        })
        .collect();
    assert_eq!(statuses, [500, 429, 200, 200]);
}

/// E5 (a fault proxy before the model, live on Docker): a provider that
/// took the request and never answered ran out the agent's time, and the
/// trial was R206, "the agent did nothing", blamed on the agent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_model_that_never_answers_is_the_providers() {
    let script = Script {
        stall: true,
        ..Script::default()
    };
    let r = run_trial(
        "builtin",
        script,
        RunOptions {
            timeout_multiplier: 0.02,
            ..options()
        },
    )
    .await;
    let p = r.rec.problem.expect("a problem");
    assert_eq!(
        (p.code.as_str(), p.owner),
        ("R214", benchflow_run::outcome::Owner::Model),
        "{p:?}"
    );
    assert!(p.message.contains("never answered"), "{}", p.message);
    // One request per try: a transient problem is tried once more.
    assert!(!r.script.lock().unwrap().requests.is_empty());
}

fn keystrokes_reply(analysis: &str, keystrokes: &[&str], complete: bool) -> Value {
    let commands: Vec<Value> = keystrokes
        .iter()
        .map(|k| json!({"keystrokes": k, "duration": 0.1}))
        .collect();
    let content =
        json!({"analysis": analysis, "plan": "p", "commands": commands, "task_complete": complete})
            .to_string();
    text_reply(&content)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminus_2_solves_hello_and_confirms_completion() {
    let script = Script {
        replies: VecDeque::from([
            (
                200,
                keystrokes_reply(
                    "Need to write the file.",
                    &["printf 'Hello, world!' > /app/hello.txt\n"],
                    true,
                ),
            ),
            (200, keystrokes_reply("The file is written.", &[], true)),
        ]),
        ..Script::default()
    };
    let r = run_trial("terminus-2", script, options()).await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert_eq!(r.rec.reward, Some(1.0));
    assert_eq!(r.rec.agent, "terminus-2");
    // The pane log is copied out as the file it is (review F2-B-r11 L3).
    assert!(r.dir.join("agent/terminus_2.pane").is_file());
    // The command, then task_complete twice (asked, then confirmed).
    assert_eq!(r.rec.tool_calls, 3);
    let script = r.script.lock().unwrap();
    assert_eq!(script.requests.len(), 2);
    let first = script.requests[0]["messages"][0]["content"]
        .as_str()
        .unwrap();
    assert!(first.starts_with("You are an AI assistant tasked with solving command-line tasks"));
    assert!(first.contains("Task Description:\nCreate a file `/app/hello.txt`"));
    assert!(first.contains("Current terminal state:\nCurrent Terminal Screen:"));
    assert!(
        script.requests[0].get("tools").is_none(),
        "Terminus-2 sends no tools"
    );
    let second = script.requests[1]["messages"].as_array().unwrap();
    let confirm = second.last().unwrap()["content"].as_str().unwrap();
    assert!(
        confirm.contains("Are you sure you want to mark the task as complete?"),
        "{confirm}"
    );
    drop(script);
    let world = r.world.lock().unwrap();
    assert!(
        world
            .commands
            .iter()
            .any(|(c, u)| c.contains("tmux new-session") && u.as_deref() == Some("agent"))
    );
    drop(world);
    let trajectory =
        std::fs::read_to_string(r.dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(
        trajectory.contains("\"tool_name\": \"bash_command\""),
        "{trajectory}"
    );
    assert!(trajectory.contains("call_1_task_complete"), "{trajectory}");
    assert!(
        trajectory.contains("Analysis: Need to write the file."),
        "{trajectory}"
    );
    assert_no_key(&r.dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminus_2_feeds_parse_errors_back() {
    let script = Script {
        replies: VecDeque::from([
            (200, text_reply("I will write the file now.")),
            (
                200,
                keystrokes_reply("ok", &["printf 'Hello, world!' > /app/hello.txt\n"], true),
            ),
            (200, keystrokes_reply("done", &[], true)),
        ]),
        ..Script::default()
    };
    let r = run_trial("terminus-2", script, options()).await;
    assert_eq!(r.rec.reward, Some(1.0), "{:?}", r.rec.problem);
    let script = r.script.lock().unwrap();
    let second = script.requests[1]["messages"].as_array().unwrap();
    let feedback = second.last().unwrap()["content"].as_str().unwrap();
    assert!(
        feedback.starts_with(
            "Previous response had parsing errors:\nERROR: No valid JSON found in response"
        ),
        "{feedback}"
    );
    assert!(feedback.ends_with("Please fix these issues and provide a proper JSON response."));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_huge_reply_is_kept_whole_once_and_capped_in_derived_copies() {
    // A 9 MiB reply: the raw log keeps it; the model-call log (whose later
    // requests carry it again), the trainer files and the job's
    // concatenations keep a marked cut, so the folder holds it about once.
    let huge = "y".repeat(9 << 20);
    let first = json!({"model": "fake/model", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": huge,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "bash", "arguments": json!({"command": "echo hi"}).to_string()}}]}}],
        "usage": usage()});
    let r = run_trial(
        "builtin",
        Script {
            replies: VecDeque::from([
                (200, first),
                (200, tool_reply("c2", "submit", json!({"summary": "done"}))),
            ]),
            ..Script::default()
        },
        options(),
    )
    .await;
    // One run of two calls: the second carries the reply back.
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert_eq!(r.script.lock().unwrap().requests.len(), 2);
    let read = |rel: &str| std::fs::read_to_string(r.dir.join(rel)).unwrap();
    let marker = "[BenchFlow cut this text: ";
    let raw = read("trajectory/acp_trajectory.jsonl");
    assert!(raw.contains(&huge), "the raw log keeps the reply whole");
    assert!(!raw.contains(marker));
    let calls = read("agent/model_calls.jsonl");
    assert!(calls.contains(marker), "the call log cuts it");
    for line in calls.lines() {
        assert!(
            line.len() < (1 << 20) + 65_536,
            "a call-log line of {} bytes",
            line.len()
        );
    }
    for rel in [
        "trainer/atif.json",
        "trainer/adp.jsonl",
        "trainer/verifiers.jsonl",
    ] {
        let text = read(rel);
        assert!(text.contains(marker), "{rel} marks the cut");
        assert!(
            text.contains("of 9437184 bytes kept"),
            "{rel} gives the size"
        );
        assert!(
            text.len() < (1 << 20) + 65_536,
            "{rel} is {} bytes",
            text.len()
        );
    }
    let job = r.dir.parent().unwrap();
    assert!(
        std::fs::read_to_string(job.join("verifiers.jsonl"))
            .unwrap()
            .contains(marker)
    );
    // The whole folder holds the reply about once, not seven times.
    let total: u64 = walk(&r.dir)
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    assert!(total < 2 * (9 << 20), "the trial folder is {total} bytes");
}
