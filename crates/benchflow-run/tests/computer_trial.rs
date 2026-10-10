//! Whole trials of computer-1 through the real engine, on an in-memory
//! sandbox that plays a desktop, against a fake vision model.
//!
//! The sandbox answers computer-1's desktop probe and start-up, returns a
//! tiny PNG for every screenshot, records each xdotool command, and keeps
//! the final answer file. The model server answers each request with the
//! next scripted JSON action, as a vision model in Harbor's format would.

mod harness_support;

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::routing::{get, post};
use base64::Engine;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::native::ClientOptions;
use benchflow_run::native_agent::NativeAgent;
use benchflow_run::outcome::{Owner, Status};
use harness_support::{Handler, World, ok, options, run_trial};
use serde_json::{Value, json};

const KEY: &str = "vck_test_key_0123456789abcdef";
/// A 1x1 PNG.
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
const ANSWER: (&str, &[u8]) = ("/logs/agent/final_answer.txt", b"HELLO-3534");

#[derive(Default)]
struct Model {
    replies: VecDeque<String>,
    requests: Vec<Value>,
}

type Shared = Arc<Mutex<Model>>;

async fn completions(State(m): State<Shared>, Json(body): Json<Value>) -> Json<Value> {
    let mut m = m.lock().unwrap();
    m.requests.push(body);
    let text = m
        .replies
        .pop_front()
        .unwrap_or_else(|| r#"{"analysis": "", "plan": "", "action": {"type": "wait"}}"#.into());
    Json(
        json!({"model": "fake/model", "choices": [{"finish_reason": "stop",
        "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 1500, "completion_tokens": 40, "cost": 0.0002}}),
    )
}

async fn endpoints() -> Json<Value> {
    Json(
        json!({"data": {"endpoints": [{"provider_name": "google", "name": "google"},
                                        {"provider_name": "baseten", "name": "baseten"}]}}),
    )
}

async fn serve(model: Shared) -> String {
    let app = axum::Router::new()
        .route("/v1/chat/completions", post(completions))
        .route("/v1/models/{*path}", get(endpoints))
        .with_state(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}/v1")
}

fn action(analysis: &str, action: Value) -> String {
    json!({"analysis": analysis, "plan": "next", "action": action}).to_string()
}

/// A desktop that has everything, or lacks the display.
fn desktop(has_display: bool) -> Handler {
    Arc::new(move |c: &str, w: &mut World| {
        if c.contains("BF missing") {
            return Some(if has_display {
                ok("BF missing\nBF browser /usr/bin/chromium\n")
            } else {
                ok("BF missing Xvfb xdotool\n")
            });
        }
        if c.contains("BF display ready") {
            return Some(ok("BF display ready\n"));
        }
        if c.contains("BF browser ready") {
            return Some(ok("BF browser ready\n"));
        }
        if c.contains("import -window root") {
            return Some(ok(PNG));
        }
        // The final answer: printf '%s' <base64> | base64 -d > <file>.
        if let Some(rest) = c.split("printf '%s' ").nth(1)
            && c.contains("final_answer.txt")
        {
            let b64 = shlex::split(rest).unwrap()[0].clone();
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .unwrap();
            w.files.insert(ANSWER.0.to_string(), bytes);
            return Some(ok(""));
        }
        None
    })
}

fn agent(base: &str) -> Arc<NativeAgent> {
    let host = HostEnv::new(
        BTreeMap::from([
            ("AI_GATEWAY_API_KEY".to_string(), KEY.to_string()),
            ("BENCHFLOW_NATIVE_BASE_URL".to_string(), base.to_string()),
        ]),
        None,
    );
    Arc::new(
        NativeAgent::new("computer-1", Some("vercel/fake/model"), None, &host)
            .expect("computer-1 is a native agent")
            .client_options(ClientOptions {
                timeout: Duration::from_secs(10),
                max_attempts: 2,
                backoff_base: Duration::from_millis(5),
                backoff_cap: Duration::from_millis(20),
            }),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn computer_1_types_clicks_confirms_done_and_writes_the_answer() {
    let model: Shared = Arc::new(Mutex::new(Model {
        replies: VecDeque::from([
            action(
                "A blank browser.",
                json!({"type": "navigate", "url": "file:///app/hello.html"}),
            ),
            action(
                "The page is open.",
                json!({"type": "click", "x": 200, "y": 240}),
            ),
            "I will type now.".into(),
            action(
                "The box has focus.",
                json!({"type": "type", "text": "BenchFlow"}),
            ),
            action("Typed.", json!({"type": "click", "x": 470, "y": 240})),
            action(
                "The code shows.",
                json!({"type": "done", "result": "HELLO-3534"}),
            ),
            action(
                "Confirming.",
                json!({"type": "done", "result": "HELLO-3534"}),
            ),
        ]),
        ..Model::default()
    }));
    let base = serve(model.clone()).await;
    let r = run_trial(
        "hello-desktop",
        agent(&base),
        desktop(true),
        ANSWER,
        options(Duration::from_secs(30)),
    )
    .await;
    assert_eq!(r.rec.status, Status::Scored, "{:?}", r.rec.problem);
    assert_eq!(r.rec.reward, Some(1.0));
    assert_eq!(r.rec.agent, "computer-1");
    // Six actions ran (the parse error cost a turn but no action).
    assert_eq!(r.rec.tool_calls, 6);
    assert_eq!(r.rec.usage.source, "native");
    assert!((r.rec.usage.cost_usd.unwrap() - 0.0014).abs() < 1e-9);

    // xdotool ran as the agent user on display :1.
    let world = r.world.lock().unwrap();
    let typed = world
        .commands
        .iter()
        .find(|(c, _)| c.contains("xdotool type --clearmodifiers -- BenchFlow"))
        .expect("the type action ran");
    assert_eq!(typed.1.as_deref(), Some("agent"));
    // After the confinement prefix every agent command carries.
    assert!(typed.0.contains("DISPLAY=:1 xdotool"), "{}", typed.0);
    assert!(
        world
            .commands
            .iter()
            .any(|(c, _)| c.contains("mousemove 470 240 click 1"))
    );
    assert!(
        world
            .commands
            .iter()
            .any(|(c, _)| c.contains("ctrl+l") && c.contains("file:///app/hello.html"))
    );
    drop(world);

    // The model saw Harbor's prompt and a screenshot each turn, at most
    // three at a time, and the "Are you sure" question after the first done.
    let m = model.lock().unwrap();
    assert_eq!(m.requests.len(), 7);
    let first = &m.requests[0]["messages"][0]["content"];
    assert!(
        first[0]["text"]
            .as_str()
            .unwrap()
            .contains("You are computer-1")
    );
    assert!(
        first[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    let last = m.requests[6]["messages"].as_array().unwrap();
    let images = last
        .iter()
        .filter(|msg| {
            msg["content"]
                .as_array()
                .is_some_and(|parts| parts.iter().any(|p| p["type"] == "image_url"))
        })
        .count();
    assert_eq!(images, 3);
    let repair = m.requests[3]["messages"]
        .as_array()
        .unwrap()
        .last()
        .unwrap();
    assert!(
        repair["content"]
            .as_str()
            .unwrap()
            .starts_with("Previous response had parsing errors:")
    );
    let asked = last.last().unwrap()["content"][0]["text"].as_str().unwrap();
    assert!(asked.contains("Are you sure you want to mark the task as complete?"));
    drop(m);

    // Every screenshot is in the trial folder; the call log has no image bytes or key.
    let shots = std::fs::read_dir(r.dir.join("agent/screenshots"))
        .unwrap()
        .count();
    assert_eq!(
        shots, 6,
        "the first, and one after each action but the confirming done"
    );
    let calls = std::fs::read_to_string(r.dir.join("agent/model_calls.jsonl")).unwrap();
    assert!(!calls.contains(PNG));
    assert!(calls.contains("characters not logged"));
    assert!(!calls.contains(KEY));
    let trajectory =
        std::fs::read_to_string(r.dir.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(
        trajectory.contains("type \\\"BenchFlow\\\""),
        "{trajectory}"
    );
    assert!(trajectory.contains("screenshots/001.png"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_image_without_a_display_is_refused_as_the_tasks_problem() {
    let model: Shared = Arc::new(Mutex::new(Model::default()));
    let base = serve(model.clone()).await;
    let r = run_trial(
        "hello",
        agent(&base),
        desktop(false),
        ("/app/hello.txt", b"Hello, world!"),
        options(Duration::from_secs(30)),
    )
    .await;
    assert_eq!(r.rec.status, Status::Errored);
    let p = r.rec.problem.clone().unwrap();
    assert_eq!((p.code.as_str(), p.owner), ("R215", Owner::Task), "{p:?}");
    assert!(p.message.contains("lacks Xvfb xdotool"), "{}", p.message);
    assert!(p.next.contains("hello-desktop"), "{}", p.next);
    assert!(
        model.lock().unwrap().requests.is_empty(),
        "no model call was made"
    );
}
