//! Model-assisted review against a fake model server: `bench check
//! --review` (a task's quality) and `bench results --analyze` (why trials
//! failed).
//!
//! The server speaks OpenAI-compatible chat completions and lists the
//! model's upstreams like the Vercel AI Gateway does, Baseten among them. It
//! records every request, so the tests check what reaches the model: what
//! the prompt holds, that the gateway pin is there without Baseten, and that
//! no secret from a task file, the machine, or the key gets in.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::native::ClientOptions;
use benchflow_run::review::analyze::{self, Part};
use benchflow_run::review::task::{self, CRITERIA};
use benchflow_run::review::{ReviewErrorKind, Reviewer};
use benchflow_task::CheckOptions;
use benchflow_task::diag::Severity;
use serde_json::{Value, json};

const KEY: &str = "vck_review_test_key_0123456789";
/// A credential in the user's environment that a task file happens to hold.
const HOST_SECRET: &str = "hostsecret-9f8e7d6c5b4a";
/// A credential in a task file that only the redaction patterns know.
const FILE_SECRET: &str = "sk-proj-AAAABBBBCCCCDDDDEEEEFFFF0000";
const MODEL: &str = "vercel/review/model";

#[derive(Default)]
struct Script {
    replies: VecDeque<String>,
    /// Replies for requests that hold a marker (a trial's name), so trials
    /// analyzed at the same time get their own.
    keyed: Vec<(String, VecDeque<String>)>,
    requests: Vec<Value>,
    bearer_ok: bool,
}

type Shared = Arc<Mutex<Script>>;

async fn completions(
    State(s): State<Shared>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let mut s = s.lock().unwrap();
    s.bearer_ok = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == format!("Bearer {KEY}"));
    let seen = body.to_string();
    s.requests.push(body);
    let keyed = s
        .keyed
        .iter_mut()
        .find(|(marker, queue)| seen.contains(marker.as_str()) && !queue.is_empty())
        .and_then(|(_, queue)| queue.pop_front());
    let text = keyed
        .or_else(|| s.replies.pop_front())
        .unwrap_or_else(|| "{}".into());
    // `LENGTH:` makes the answer one cut off at the output limit.
    let (finish, text) = match text.strip_prefix("LENGTH:") {
        Some(rest) => ("length", rest.to_string()),
        None => ("stop", text),
    };
    (
        StatusCode::OK,
        Json(
            json!({"model": "review/model", "choices": [{"finish_reason": finish,
            "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.0005}}),
        ),
    )
}

async fn endpoints(axum::extract::Path(path): axum::extract::Path<String>) -> Json<Value> {
    assert_eq!(path, "review/model/endpoints");
    Json(json!({"data": {"endpoints": [
        {"provider_name": "baseten"}, {"provider_name": "novita"}, {"provider_name": "deepinfra"}
    ]}}))
}

async fn serve(replies: Vec<String>) -> (Shared, String) {
    serve_keyed(replies, Vec::new()).await
}

async fn serve_keyed(replies: Vec<String>, keyed: Vec<(&str, Vec<String>)>) -> (Shared, String) {
    let script: Shared = Arc::new(Mutex::new(Script {
        replies: replies.into(),
        keyed: keyed
            .into_iter()
            .map(|(m, r)| (m.to_string(), r.into()))
            .collect(),
        ..Script::default()
    }));
    let app = axum::Router::new()
        .route("/v1/chat/completions", post(completions))
        .route("/v1/models/{*path}", get(endpoints))
        .with_state(script.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (script, format!("http://{addr}/v1"))
}

async fn reviewer(base: &str) -> Reviewer {
    let host = HostEnv::new(
        BTreeMap::from([
            ("AI_GATEWAY_API_KEY".to_string(), KEY.to_string()),
            ("BENCHFLOW_NATIVE_BASE_URL".to_string(), base.to_string()),
        ]),
        None,
    );
    Reviewer::connect(
        MODEL,
        &host,
        vec![HOST_SECRET.to_string()],
        ClientOptions {
            timeout: Duration::from_secs(10),
            max_attempts: 2,
            backoff_base: Duration::from_millis(5),
            backoff_cap: Duration::from_millis(5),
        },
    )
    .await
    .unwrap()
}

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tasks")
        .join(name)
}

/// A copy of `always-passes` with secrets planted in its files.
fn task_with_secrets() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let src = fixture("always-passes");
    for e in walkdir::WalkDir::new(&src)
        .into_iter()
        .filter_map(Result::ok)
    {
        let to = dir.path().join(e.path().strip_prefix(&src).unwrap());
        if e.file_type().is_dir() {
            std::fs::create_dir_all(&to).unwrap();
        } else {
            std::fs::copy(e.path(), &to).unwrap();
        }
    }
    std::fs::write(
        dir.path().join("oracle/notes.txt"),
        format!("OPENAI_API_KEY={FILE_SECRET}\nthe host one: {HOST_SECRET}\ngateway: {KEY}\n"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join(".env"),
        "ANTHROPIC_API_KEY=only-in-dotenv-0123456789\n",
    )
    .unwrap();
    dir
}

fn review_answer() -> String {
    let verdicts: Vec<Value> = CRITERIA
        .iter()
        .map(|c| {
            let verdict = if c.id == "verifier-checks-success" {
                "fail"
            } else {
                "pass"
            };
            json!({"criterion": c.id, "verdict": verdict, "reason": "because"})
        })
        .collect();
    format!(
        "Here is my review.\n```json\n{}\n```",
        json!({"verdicts": verdicts, "findings": [
            {"criterion": "verifier-checks-success", "severity": "error", "file": "verifier/test.sh",
             "line": 3, "quote": "if [ -d /app ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi",
             "problem": "the verifier passes whenever /app exists, which it does before the agent starts",
             "fix": "check that /app/ready.txt says ready"},
            {"criterion": "environment-pinned", "severity": "note", "file": "sandbox/Dockerfile",
             "line": 40, "quote": "", "problem": "a line that is not there", "fix": "none"}
        ]})
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_task_review_sends_the_task_and_no_secret() {
    let (script, base) = serve(vec![review_answer()]).await;
    let r = reviewer(&base).await;
    let dir = task_with_secrets();
    let report = benchflow_task::check(dir.path(), &CheckOptions { deep: true });
    let review = task::review(&r, &report).await.unwrap();

    let s = script.lock().unwrap();
    assert!(s.bearer_ok, "the key goes in the Authorization header");
    assert_eq!(s.requests.len(), 1);
    let body = &s.requests[0];
    assert_eq!(
        body["providerOptions"]["gateway"]["only"],
        json!(["novita", "deepinfra"]),
        "pinned to the allowed upstreams, never Baseten"
    );
    assert_eq!(body["model"], "review/model");
    let text = body.to_string();
    // What the model needs.
    assert!(text.contains("Make the file `/app/ready.txt` say `ready`."));
    assert!(text.contains("=== verifier/test.sh (verifier, 3 lines) ==="));
    assert!(text.contains("3| if [ -d /app ]"));
    assert!(text.contains("=== oracle/solve.sh (oracle, 2 lines) ==="));
    assert!(text.contains("=== sandbox/Dockerfile (sandbox image, 2 lines) ==="));
    assert!(text.contains("mounted at /verifier only after the agent finishes"));
    for c in CRITERIA {
        assert!(text.contains(c.id), "{}", c.id);
    }
    // What it must never see.
    for secret in [KEY, HOST_SECRET, FILE_SECRET, "only-in-dotenv-0123456789"] {
        assert!(!text.contains(secret), "{secret} reached the request");
    }
    assert!(text.contains(".env (not sent: files like this usually hold credentials)"));

    // The answer, read and checked.
    assert_eq!(review.findings.len(), 2);
    let f = &review.findings[0];
    assert_eq!(
        (f.code, f.severity, f.line),
        ("M102", Severity::Error, Some(3))
    );
    assert_eq!(f.location_note, None);
    let g = &review.findings[1];
    assert_eq!(
        g.line, None,
        "line 40 of a 2-line file is not shown as a location"
    );
    assert!(g.location_note.as_ref().unwrap().contains("does not exist"));
    assert_eq!(review.verdicts.len(), CRITERIA.len());
    assert_eq!(review.spend.calls, 1);
    assert_eq!(review.spend.cost_usd, Some(0.0005));
    assert!(review.opinion.contains("opinion"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_malformed_answer_gets_one_more_try_then_a_clear_error() {
    let (script, base) = serve(vec![
        "I think the task is fine.".into(),
        "{\"verdicts\": [], \"findings\": []}".into(),
    ])
    .await;
    let r = reviewer(&base).await;
    let report = benchflow_task::check(&fixture("hello"), &CheckOptions { deep: true });
    let review = task::review(&r, &report).await.unwrap();
    assert!(review.findings.is_empty());
    assert_eq!(review.spend.calls, 2);
    {
        let s = script.lock().unwrap();
        let retry = s.requests[1]["messages"].to_string();
        assert!(retry.contains("holds no JSON object"), "{retry}");
    }

    let (_, base) = serve(vec![
        "no json".into(),
        "```json\n{\"verdicts\": [\n```".into(),
    ])
    .await;
    let r = reviewer(&base).await;
    let e = task::review(&r, &report).await.unwrap_err();
    assert_eq!(e.kind, ReviewErrorKind::Provider);
    assert!(
        e.message.contains("did not answer with the JSON asked for"),
        "{}",
        e.message
    );
    assert!(e.next.contains("nothing it said is shown as a finding"));
}

/// A job folder with a passing trial, a trial written by this engine that
/// errored, and a trial in 0.8's layout that scored 0.
fn job() -> tempfile::TempDir {
    let job = tempfile::tempdir().unwrap();
    let j = job.path();
    std::fs::write(j.join("summary.json"), "{\"agent\": \"codex\"}").unwrap();

    let pass = j.join("hello__aaaa1111");
    std::fs::create_dir_all(&pass).unwrap();
    std::fs::write(
        pass.join("result.json"),
        json!({"task_name": "hello", "rollout_name": "hello__aaaa1111", "status": "scored",
               "rewards": {"reward": 1.0}, "passed": true})
        .to_string(),
    )
    .unwrap();

    let rust = j.join("sdk-hello__bbbb2222");
    std::fs::create_dir_all(rust.join("agent")).unwrap();
    std::fs::write(
        rust.join("result.json"),
        json!({"task_name": "sdk-hello", "rollout_name": "sdk-hello__bbbb2222", "status": "errored",
               "agent": "nemo-agent", "model": "nvidia/meta/llama-3.3-70b-instruct",
               "problem": {"code": "R205", "owner": "agent", "phase": "agent",
                           "message": "the agent reported a failure and exited with code 1",
                           "next": "read agent/stderr.txt", "transient": false}})
        .to_string(),
    )
    .unwrap();
    std::fs::write(rust.join("prompts.json"), "[\"Create /app/hello.txt.\"]").unwrap();
    std::fs::write(
        rust.join("agent/stderr.txt"),
        format!("Exception: [410] Gone\nThe model has reached its end of life\nAuthorization: Bearer {HOST_SECRET}\n"),
    )
    .unwrap();

    let old = j.join("arc-gravity__cccc3333");
    std::fs::create_dir_all(old.join("trajectory")).unwrap();
    std::fs::create_dir_all(old.join("verifier")).unwrap();
    std::fs::write(
        old.join("result.json"),
        json!({"task_name": "arc-gravity", "rollout_name": "arc-gravity__cccc3333",
               "rewards": {"reward": 0.0}, "agent_name": "codex-acp", "error": null,
               "verifier_error": null})
        .to_string(),
    )
    .unwrap();
    std::fs::write(old.join("prompts.json"), "[\"Drop the ball on the pad.\"]").unwrap();
    std::fs::write(
        old.join("trajectory/acp_trajectory.jsonl"),
        "{\"type\": \"user_message\", \"text\": \"Drop the ball on the pad.\"}\n{\"type\": \"agent_message\", \"text\": \"unexpected status 401 Unauthorized: Incorrect API key provided\"}\n",
    )
    .unwrap();
    std::fs::write(
        old.join("verifier/test-stdout.txt"),
        "fail agent_exited steps=0/1400\n",
    )
    .unwrap();
    job
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_analysis_reads_failed_trials_of_both_layouts() {
    let answer = |part: &str, quote: &str| {
        json!({"part": part, "explanation": "it failed", "confidence": "high",
               "evidence": [{"file": "agent/stderr.txt", "quote": quote}]})
        .to_string()
    };
    let (script, base) = serve_keyed(
        Vec::new(),
        vec![
            (
                "arc-gravity__cccc3333",
                vec![answer("setup", "Incorrect API key provided")],
            ),
            (
                "sdk-hello__bbbb2222",
                vec![answer("provider", "Exception: [410] Gone")],
            ),
        ],
    )
    .await;
    let r = reviewer(&base).await;
    let dir = job();
    let a = analyze::analyze_job(&r, dir.path(), 10).await.unwrap();

    assert_eq!(a.failed, 2, "the passing trial is not analyzed");
    assert_eq!(a.trials.len(), 2);
    let bodies: Vec<String> = script
        .lock()
        .unwrap()
        .requests
        .iter()
        .map(Value::to_string)
        .collect();
    let old = bodies
        .iter()
        .find(|b| b.contains("arc-gravity__cccc3333"))
        .unwrap();
    assert!(old.contains("Drop the ball on the pad."));
    assert!(old.contains("#2 agent_message: unexpected status 401"));
    assert!(old.contains("fail agent_exited steps=0/1400"));
    let new = bodies
        .iter()
        .find(|b| b.contains("sdk-hello__bbbb2222"))
        .unwrap();
    assert!(new.contains("[R205] the agent reported a failure"));
    assert!(new.contains("Exception: [410] Gone"));
    for b in &bodies {
        assert!(!b.contains(HOST_SECRET) && !b.contains(KEY));
        assert!(b.contains("\"only\":[\"novita\",\"deepinfra\"]"));
    }

    let by_trial = |name: &str| a.trials.iter().find(|t| t.trial == name).unwrap();
    let new = by_trial("sdk-hello__bbbb2222");
    assert_eq!(new.recorded.code.as_deref(), Some("R205"));
    assert_eq!(new.recorded.owner.as_deref(), Some("the agent"));
    assert!(new.evidence[0].found);
    assert_eq!(a.by_cause.len(), 2);
    assert_eq!(a.spend.calls, 2);
    assert!(a.not_analyzed.is_empty());
    let parts: Vec<Part> = a.trials.iter().map(|t| t.part).collect();
    assert!(parts.contains(&Part::Provider) && parts.contains(&Part::Setup));

    // The cap leaves the rest listed, not analyzed.
    let (_, base) = serve(vec![answer("agent", "x")]).await;
    let r = reviewer(&base).await;
    let a = analyze::analyze_job(&r, dir.path(), 1).await.unwrap();
    assert_eq!(a.trials.len(), 1);
    assert_eq!(a.not_analyzed.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bad_answer_for_one_trial_does_not_stop_the_others() {
    let (_, base) = serve_keyed(
        Vec::new(),
        vec![
            (
                "arc-gravity__cccc3333",
                vec!["not json".into(), "still not json".into()],
            ),
            (
                "sdk-hello__bbbb2222",
                vec![
                    json!({"part": "agent", "explanation": "gave up", "evidence": []}).to_string(),
                ],
            ),
        ],
    )
    .await;
    let r = reviewer(&base).await;
    let dir = job();
    let a = analyze::analyze_job(&r, dir.path(), 10).await.unwrap();
    let errors: Vec<_> = a.trials.iter().filter(|t| t.error.is_some()).collect();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].error.as_ref().unwrap().contains("JSON"));
    assert_eq!(
        a.by_cause.len(),
        1,
        "a trial the model could not analyze is not grouped"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_answer_cut_off_at_the_limit_is_asked_again_with_more_room() {
    // Cut off mid-answer: the inner evidence object is complete, the
    // answer is not, and must not be taken for it.
    let cut = r#"LENGTH:{"part": "task", "evidence": [{"file": "x", "quote": "y"}], "expla"#;
    let whole = json!({"part": "agent", "explanation": "it gave up", "evidence": []}).to_string();
    let (script, base) = serve_keyed(
        Vec::new(),
        vec![("sdk-hello__bbbb2222", vec![cut.to_string(), whole])],
    )
    .await;
    let r = reviewer(&base).await;
    let dir = job();
    std::fs::remove_dir_all(dir.path().join("arc-gravity__cccc3333")).unwrap();
    let a = analyze::analyze_job(&r, dir.path(), 10).await.unwrap();
    let t = &a.trials[0];
    assert_eq!(t.error, None);
    assert_eq!(t.part, Part::Agent);
    assert_eq!(t.spend.calls, 2);
    let s = script.lock().unwrap();
    assert_eq!(s.requests[0]["max_tokens"], 12_000);
    assert_eq!(s.requests[1]["max_tokens"], 24_000);
    assert_eq!(
        s.requests[1]["messages"].as_array().unwrap().len(),
        s.requests[0]["messages"].as_array().unwrap().len(),
        "the same question again, not a conversation about the cut-off answer"
    );
}
