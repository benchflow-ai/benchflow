//! Whole trials of the external agent through the real engine, on an
//! in-memory sandbox, driven by a scripted local backend that speaks
//! Karotte's backend protocol (`/api/internal/*`).
//!
//! The backend answers `get_message` from a script (404 while it has
//! nothing, as a long poll would), stores every appended event by `seq`,
//! and keeps every run state, so the tests check what a Karotte-compatible
//! trainer would see.

mod harness_support;

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::external::BackendOptions;
use benchflow_run::external_agent::ExternalAgent;
use benchflow_run::outcome::{Owner, Status};
use harness_support::{Handler, ok, options, run_trial};
use serde_json::{Value, json};

/// One scripted answer to `get_message`.
#[derive(Clone)]
enum Answer {
    Message(Value),
    NotYet,
    Raw(&'static str),
    Hang,
}

#[derive(Default)]
struct Backend {
    answers: VecDeque<Answer>,
    /// What `get_message` answers once the script runs out.
    after: Option<Answer>,
    created: Vec<String>,
    events: BTreeMap<u64, Value>,
    states: Vec<Value>,
    deletes: u32,
    run_ids: Vec<String>,
}

type Shared = Arc<Mutex<Backend>>;

async fn transcript_length(
    State(b): State<Shared>,
    Query(q): Query<BTreeMap<String, String>>,
) -> (StatusCode, Json<Value>) {
    let b = b.lock().unwrap();
    if b.created.contains(&q["run_id"]) {
        (StatusCode::OK, Json(json!(b.events.len())))
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "no transcript"})),
        )
    }
}

async fn create(State(b): State<Shared>, Query(q): Query<BTreeMap<String, String>>) -> Json<Value> {
    b.lock().unwrap().created.push(q["run_id"].clone());
    Json(json!(null))
}

async fn append(
    State(b): State<Shared>,
    Query(q): Query<BTreeMap<String, String>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let mut b = b.lock().unwrap();
    b.run_ids.push(q["run_id"].clone());
    let seq = body["seq"].as_u64().unwrap();
    b.events.insert(seq, body["event"].clone());
    Json(json!(null))
}

async fn state(State(b): State<Shared>, Json(body): Json<Value>) -> Json<Value> {
    b.lock().unwrap().states.push(body);
    Json(json!(null))
}

async fn get_message(State(b): State<Shared>) -> (StatusCode, String) {
    let answer = {
        let b = b.lock().unwrap();
        match b.answers.front().cloned() {
            Some(a) => a,
            None => b.after.clone().unwrap_or(Answer::NotYet),
        }
    };
    match answer {
        Answer::Message(m) => (StatusCode::OK, m.to_string()),
        Answer::Raw(text) => (StatusCode::OK, text.to_string()),
        Answer::NotYet => {
            let mut b = b.lock().unwrap();
            if matches!(b.answers.front(), Some(Answer::NotYet)) {
                b.answers.pop_front();
            }
            (StatusCode::NOT_FOUND, "{\"detail\": \"no message\"}".into())
        }
        Answer::Hang => {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            (StatusCode::OK, String::new())
        }
    }
}

async fn delete_message(State(b): State<Shared>) -> Json<Value> {
    let mut b = b.lock().unwrap();
    b.answers.pop_front();
    b.deletes += 1;
    Json(json!(true))
}

async fn serve(backend: Shared) -> String {
    let app = axum::Router::new()
        .route("/api/internal/transcript_length", get(transcript_length))
        .route("/api/internal/create_transcript", post(create))
        .route("/api/internal/append_transcript", post(append))
        .route("/api/internal/update_run_state", post(state))
        .route("/api/internal/get_message", get(get_message))
        .route("/api/internal/delete_message", post(delete_message))
        .with_state(backend);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// A `message_added` event with one tool call, as a trainer sends it.
fn call(id: &str, name: &str, args: Value) -> Answer {
    Answer::Message(
        json!({"type": "message_added", "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": id, "type": "function",
                        "function": {"name": name, "arguments": args.to_string()}}]}}),
    )
}

fn agent(url: &str) -> Arc<ExternalAgent> {
    let host = HostEnv::new(BTreeMap::new(), None);
    Arc::new(
        ExternalAgent::new(Some(url), Some("pt/checkpoint-7"), None, &host).backend_options(
            BackendOptions {
                attempt_timeout: Duration::from_millis(500),
                retry_budget: Duration::from_secs(2),
                backoff_base: Duration::from_millis(10),
                backoff_cap: Duration::from_millis(50),
                poll_gap: Duration::from_millis(20),
            },
        ),
    )
}

fn no_handler() -> Handler {
    Arc::new(|_, _| None)
}

const HELLO: (&str, &[u8]) = ("/app/hello.txt", b"Hello, world!");

fn types(b: &Backend) -> Vec<String> {
    b.events
        .values()
        .map(|e| e["type"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_scripted_backend_solves_hello_and_hears_the_reward() {
    let backend: Shared = Arc::new(Mutex::new(Backend {
        answers: VecDeque::from([
            Answer::NotYet,
            call(
                "c1",
                "write",
                json!({"file_path": "/app/hello.txt", "content": "Hello, world!"}),
            ),
            Answer::NotYet,
            call("c2", "bash", json!({"command": "cat /app/hello.txt"})),
            call("c3", "submit", json!({"summary": "wrote it"})),
        ]),
        ..Backend::default()
    }));
    let url = serve(backend.clone()).await;
    let handler: Handler = Arc::new(|c, w| {
        c.contains("cat /app/hello.txt")
            .then(|| ok(String::from_utf8_lossy(&w.files["/app/hello.txt"]).to_string()))
    });
    let r = run_trial(
        "hello",
        agent(&url),
        handler,
        HELLO,
        options(Duration::from_secs(10)),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert_eq!(r.rec.reward, Some(1.0));
    assert_eq!(r.rec.agent, "external");
    assert_eq!(r.rec.model.as_deref(), Some("pt/checkpoint-7"));
    assert_eq!(r.rec.tool_calls, 3);
    assert_eq!(r.rec.trajectory_summary["trajectory_source"], "external");

    // The tools ran in the sandbox as the agent user.
    let world = r.world.lock().unwrap();
    let cat = world
        .commands
        .iter()
        .find(|(c, _)| c.contains("cat /app/hello.txt"))
        .unwrap();
    assert_eq!(cat.1.as_deref(), Some("agent"));
    drop(world);

    // The backend saw Karotte's event sequence, numbered from 0 without gaps.
    let b = backend.lock().unwrap();
    let run_id = &r.rec.rollout_name;
    assert_eq!(b.created, std::slice::from_ref(run_id));
    assert!(b.run_ids.iter().all(|id| id == run_id));
    assert_eq!(
        b.events.keys().copied().collect::<Vec<_>>(),
        (0..b.events.len() as u64).collect::<Vec<_>>()
    );
    let t = types(&b);
    assert_eq!(
        t[..5],
        [
            "task_started",
            "task_pre_hook_completed",
            "message_added",
            "step_started",
            "message_added"
        ]
    );
    assert_eq!(
        t[t.len() - 3..],
        ["step_completed", "scoring", "task_completed"]
    );
    assert_eq!(t.iter().filter(|e| *e == "tool_call_started").count(), 3);
    assert_eq!(t.iter().filter(|e| *e == "tool_call_completed").count(), 3);
    let started = &b.events[&0];
    assert_eq!(started["run_id"], run_id.as_str());
    assert_eq!(started["task_id"], "hello");
    assert_eq!(started["model"], "pt/checkpoint-7");
    assert_eq!(started["n_steps"], 1);
    let tools: Vec<&str> = b.events[&1]["metadata"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(tools, ["bash", "view", "replace", "write", "submit"]);
    assert_eq!(b.events[&2]["message"]["role"], "system");
    assert!(
        b.events[&4]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("hello.txt")
    );
    // The bash result the backend read, with the time note when a clock runs.
    let bash_result = b
        .events
        .values()
        .find(|e| e["type"] == "message_added" && e["message"]["tool_call_id"] == "c2")
        .unwrap();
    assert_eq!(bash_result["message"]["role"], "tool");
    assert!(
        bash_result["message"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Hello, world!")
    );
    let scoring = b.events.values().find(|e| e["type"] == "scoring").unwrap();
    assert_eq!(scoring["scoring"]["score"], 1.0);
    assert_eq!(scoring["scoring"]["continue_task"], false);
    assert_eq!(b.events.values().last().unwrap()["status"], "passed");
    assert_eq!(b.states.first().unwrap()["status"], "running");
    assert_eq!(b.states.last().unwrap()["status"], "passed");
    assert_eq!(b.states.last().unwrap()["score"], 1.0);
    assert_eq!(b.deletes, 3);
    drop(b);

    // The trial folder has the shared trajectory and Karotte's transcript.
    let transcript: Value = serde_json::from_str(
        &std::fs::read_to_string(r.dir.join("agent/backend_transcript.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(transcript["run_id"], run_id.as_str());
    assert_eq!(
        transcript["events"].as_array().unwrap().last().unwrap()["type"],
        "task_completed"
    );
    let trajectory =
        std::fs::read_to_string(r.dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(
        trajectory.contains("\"write /app/hello.txt\""),
        "{trajectory}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backend_with_nothing_to_say_stops_the_trial_at_the_idle_timeout() {
    for silence in [Answer::NotYet, Answer::Hang] {
        let backend: Shared = Arc::new(Mutex::new(Backend {
            after: Some(silence),
            ..Backend::default()
        }));
        let url = serve(backend.clone()).await;
        let r = run_trial(
            "hello",
            agent(&url),
            no_handler(),
            HELLO,
            options(Duration::from_secs(1)),
        )
        .await;
        assert_eq!(r.rec.status, Status::Errored);
        let p = r.rec.problem.clone().unwrap();
        assert_eq!((p.code.as_str(), p.owner), ("R203", Owner::Agent), "{p:?}");
        assert!(p.message.contains(&url), "{}", p.message);
        assert!(p.message.contains("sent nothing for 1s"), "{}", p.message);
        assert!(p.next.contains("--idle-timeout"));
        // The backend heard that the run ended in an error.
        let b = backend.lock().unwrap();
        let t = types(&b);
        assert_eq!(t[t.len() - 2..], ["error", "task_completed"]);
        assert_eq!(b.events.values().last().unwrap()["status"], "error");
        assert_eq!(b.states.last().unwrap()["status"], "error");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_malformed_message_stops_the_trial_naming_the_backend() {
    let backend: Shared = Arc::new(Mutex::new(Backend {
        answers: VecDeque::from([Answer::Raw(
            r#"{"type": "message_added", "message": {"role": "assistant", "tool_calls": "bash ls"}}"#,
        )]),
        ..Backend::default()
    }));
    let url = serve(backend.clone()).await;
    let r = run_trial(
        "hello",
        agent(&url),
        no_handler(),
        HELLO,
        options(Duration::from_secs(10)),
    )
    .await;
    assert_eq!(r.rec.status, Status::Errored);
    let p = r.rec.problem.clone().unwrap();
    assert_eq!((p.code.as_str(), p.owner), ("R205", Owner::Agent), "{p:?}");
    assert!(!p.transient);
    assert!(
        p.message.contains("GET /api/internal/get_message"),
        "{}",
        p.message
    );
    assert!(p.message.contains("`message.tool_calls`"), "{}", p.message);
    assert!(p.next.contains("message_added"), "{}", p.next);
    let b = backend.lock().unwrap();
    let error = b.events.values().find(|e| e["type"] == "error").unwrap();
    assert_eq!(error["exception_type"], "R205");
    assert_eq!(b.deletes, 0, "an unreadable message is not consumed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unreachable_backend_is_a_transient_failure_of_the_agent() {
    // Nothing listens on this port.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let r = run_trial(
        "hello",
        agent(&url),
        no_handler(),
        HELLO,
        options(Duration::from_secs(10)),
    )
    .await;
    let p = r.rec.problem.clone().unwrap();
    assert_eq!((p.code.as_str(), p.owner), ("R205", Owner::Agent), "{p:?}");
    assert!(p.transient);
    assert!(p.message.contains("did not answer"), "{}", p.message);
    assert!(p.next.contains("running and reachable"));
}
