//! The rubric's model judge against a fake model server and a fake
//! sandbox: each criterion in a call of its own that holds only that
//! criterion, the reference labeled as the reference, evidence read from the
//! sandbox under its caps, and a reply that never becomes a verdict leaving
//! the criterion not judged.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::Json;
use axum::extract::State;
use axum::routing::{get, post};
use benchflow_run::judge::{self, Plan};
use benchflow_sandbox::{Exec, ExecOutput, Process, Sandbox, SandboxError};
use benchflow_task::rubric::{Rubric, Verdict};
use serde_json::{Value, json};

const KEY: &str = "vck_judge_test_key_0123456789";
const MODEL: &str = "vercel/review/model";

#[derive(Default)]
struct Script {
    /// Replies by criterion id: the request whose text names `CRITERION `<id>``
    /// takes the next one.
    replies: BTreeMap<String, VecDeque<String>>,
    requests: Vec<Value>,
}

type Shared = Arc<Mutex<Script>>;

async fn completions(State(s): State<Shared>, Json(body): Json<Value>) -> Json<Value> {
    let mut s = s.lock().unwrap();
    let seen = body.to_string();
    s.requests.push(body);
    let text = s
        .replies
        .iter_mut()
        .find(|(id, _)| seen.contains(&format!("CRITERION `{id}`")))
        .and_then(|(_, q)| q.pop_front())
        .unwrap_or_else(|| "no verdict here".into());
    Json(
        json!({"model": "review/model", "choices": [{"finish_reason": "stop",
        "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "cost": 0.0001}}),
    )
}

async fn endpoints() -> Json<Value> {
    Json(
        json!({"data": {"endpoints": [{"provider_name": "baseten"}, {"provider_name": "novita"}]}}),
    )
}

async fn serve(replies: &[(&str, &[&str])]) -> (Shared, BTreeMap<String, String>) {
    let script: Shared = Arc::new(Mutex::new(Script {
        replies: replies
            .iter()
            .map(|(id, r)| (id.to_string(), r.iter().map(|s| s.to_string()).collect()))
            .collect(),
        ..Script::default()
    }));
    let app = axum::Router::new()
        .route("/v1/chat/completions", post(completions))
        .route("/v1/models/{*path}", get(endpoints))
        .with_state(script.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let vars = BTreeMap::from([
        ("AI_GATEWAY_API_KEY".to_string(), KEY.to_string()),
        (
            "BENCHFLOW_NATIVE_BASE_URL".to_string(),
            format!("http://{addr}/v1"),
        ),
    ]);
    (script, vars)
}

/// A sandbox holding `files`: it answers the judge's probe and downloads.
struct Files(BTreeMap<String, Vec<u8>>, Mutex<Vec<String>>);

#[async_trait]
impl Sandbox for Files {
    fn id(&self) -> &str {
        "files"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        // The probe names the path as `f=<path>;`, quoted when it must be.
        let path = exec
            .command
            .strip_prefix("f=")
            .and_then(|r| r.split(';').next())
            .unwrap_or("")
            .trim_matches('\'');
        let stdout = match self.0.get(path) {
            Some(b) => format!("BF file {} {path}\n", b.len()),
            None => {
                let inside: Vec<_> = self
                    .0
                    .iter()
                    .filter(|(p, _)| p.starts_with(&format!("{path}/")))
                    .map(|(p, b)| format!("BF file {} {p}\n", b.len()))
                    .collect();
                if inside.is_empty() {
                    "BF missing\n".into()
                } else {
                    inside.concat()
                }
            }
        };
        Ok(ExecOutput {
            exit_code: 0,
            stdout,
            stderr: String::new(),
            timed_out: false,
            duration: Duration::from_millis(1),
        })
    }
    async fn spawn(&self, _exec: Exec) -> Result<Process, SandboxError> {
        Err(SandboxError::Unsupported("no processes".into()))
    }
    async fn upload(&self, _local: &Path, _remote: &str) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn write_file(&self, _r: &str, _c: &[u8], _m: u32) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        self.1.lock().unwrap().push(remote.to_string());
        std::fs::write(local, &self.0[remote]).unwrap();
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        Ok(())
    }
}

fn task(
    mut rubric: Value,
    files: &[(&str, &str)],
) -> (tempfile::TempDir, benchflow_task::TaskPackage) {
    let d = tempfile::tempdir().unwrap();
    let hello = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
    for f in [
        "task.md",
        "oracle/solve.sh",
        "sandbox/Dockerfile",
        "verifier/test.sh",
    ] {
        std::fs::create_dir_all(d.path().join(f).parent().unwrap()).unwrap();
        std::fs::copy(hello.join(f), d.path().join(f)).unwrap();
    }
    for (p, body) in files {
        let path = d.path().join(p);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }
    rubric["$schema"] = json!("https://task.md/schema/rubric-1.json");
    let rubric = Rubric::from_value(&rubric).unwrap_or_else(|e| panic!("{e:#?}"));
    let mut t = benchflow_task::load(d.path()).unwrap_or_else(|e| panic!("{e:#?}"));
    t.instruction = "Write /app/memo.md: a memo on Q3 revenue.".into();
    t.verifier.rubric = Some(rubric);
    (d, t)
}

fn rubric() -> Value {
    json!({
        "$schema": "https://task.md/schema/rubric-1.json",
        "x-judge": {"model": MODEL},
        "x-reference": "verifier/reference.md",
        "criteria": [
            {"id": "total", "text": "The memo states Q3 revenue of $4.7 billion.", "judge": "llm",
             "points": 2, "evidence": ["memo.md"], "reference": "verifier/answers.toml#q3.revenue"},
            {"id": "tone", "text": "The memo is written for a board audience.", "judge": "llm",
             "points": 1, "evidence": ["memo.md"]},
            {"id": "invented", "text": "The memo invents figures that are not in the sources.", "judge": "llm",
             "outcome": "bad", "points": 1, "evidence": ["memo.md"]},
            {"id": "depth", "text": "How thorough the memo is.", "judge": "llm",
             "points": 2,
             "levels": {"0": "a line", "1": "a paragraph", "2": "a full memo"}, "evidence": ["memo.md"]}
        ]
    })
}

const MEMO: &str = "# Q3 memo\n\nTo the board: Q3 revenue was $4.7 billion, up 8%.\n";

fn holds(h: bool) -> String {
    json!({"holds": h, "reason": "As the memo says.", "quotes": ["Q3 revenue was $4.7 billion"]})
        .to_string()
}

#[tokio::test]
async fn each_criterion_is_judged_alone_with_the_reference_labeled() {
    let (_d, t) = task(
        rubric(),
        &[
            (
                "verifier/reference.md",
                "REFERENCE-MEMO: revenue $4.7 billion.",
            ),
            (
                "verifier/answers.toml",
                "[q3]\nrevenue = \"$4.7 billion (ANSWER-KEY)\"\n",
            ),
        ],
    );
    let plan: Plan = judge::plan(&t, None).unwrap().unwrap();
    assert_eq!(plan.model, MODEL);
    assert_eq!(plan.items.len(), 4);
    let (script, vars) = serve(&[
        ("total", &[&holds(true)]),
        ("tone", &[&holds(true)]),
        ("invented", &[&holds(false)]),
        (
            "depth",
            &[r#"```json
{"level": "2", "reason": "A full memo.", "quotes": []}
```"#],
        ),
    ])
    .await;
    let sb = Files(
        BTreeMap::from([("/app/memo.md".to_string(), MEMO.as_bytes().to_vec())]),
        Mutex::new(Vec::new()),
    );
    let evidence = judge::fetch(&sb, "/app", &plan).await;
    // One download for an item four criteria name.
    assert_eq!(*sb.1.lock().unwrap(), vec!["/app/memo.md".to_string()]);
    let reviewer = judge::connect(&plan, &vars).await.unwrap();
    let out = tempfile::tempdir().unwrap();
    let judged = judge::judge(&plan, &evidence, &reviewer, out.path()).await;
    assert!(judged.failures.is_empty(), "{:?}", judged.failures);
    assert_eq!(judged.verdicts["total"], Verdict::Pass);
    // A bad-outcome criterion whose statement does not hold passes.
    assert_eq!(judged.verdicts["invented"], Verdict::Pass);
    assert_eq!(judged.verdicts["depth"], Verdict::Level("2".into()));
    assert_eq!(judged.spend.calls, 4);

    let requests = script.lock().unwrap().requests.clone();
    assert_eq!(requests.len(), 4);
    let ids = ["total", "tone", "invented", "depth"];
    for r in &requests {
        let text = r.to_string();
        // Exactly one criterion per request: its own id and text, no other.
        let named: Vec<&&str> = ids
            .iter()
            .filter(|id| text.contains(&format!("CRITERION `{id}`")))
            .collect();
        assert_eq!(named.len(), 1, "{text}");
        for other in [
            "states Q3 revenue",
            "board audience",
            "invents figures",
            "How thorough",
        ] {
            let mine = match *named[0] {
                "total" => "states Q3 revenue",
                "tone" => "board audience",
                "invented" => "invents figures",
                _ => "How thorough",
            };
            assert_eq!(text.contains(other), other == mine, "{other} in {text}");
        }
        assert_eq!(r["temperature"], json!(0.0));
        // The rubric's reference, labeled, before the agent's work.
        let user = r["messages"][1]["content"].as_str().unwrap();
        let reference = user
            .find("=== REFERENCE: verifier/reference.md ===")
            .unwrap();
        let work = user
            .find("=== AGENT'S WORK: /app/memo.md (markdown")
            .unwrap();
        assert!(reference < work);
        assert!(user.contains("NOT the agent's work"));
        assert!(user.contains("Q3 revenue was $4.7 billion"));
        // The criterion's own reference reaches only its call.
        assert_eq!(user.contains("ANSWER-KEY"), named[0] == &"total");
        assert!(!text.contains(KEY));
    }

    let lines: Vec<Value> = std::fs::read_to_string(out.path().join("judgments.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["criterion"], "total");
    assert_eq!(lines[0]["model"], MODEL);
    assert_eq!(lines[0]["verdict"], "pass");
    assert_eq!(lines[0]["quotes"][0]["found"], true);
    assert_eq!(
        lines[0]["references"],
        json!(["verifier/reference.md", "verifier/answers.toml#q3.revenue"])
    );
    assert!(
        lines[0]["attempts"][0]["reply"]
            .as_str()
            .unwrap()
            .contains("holds")
    );
    assert!(
        lines[0]["system"]
            .as_str()
            .unwrap()
            .contains("You are a verifier")
    );
    assert_eq!(lines[0]["input_tokens"], 1000);
}

#[tokio::test]
async fn a_reply_that_never_parses_leaves_its_criterion_not_judged() {
    let (_d, t) = task(
        rubric(),
        &[
            ("verifier/reference.md", "ref"),
            ("verifier/answers.toml", "[q3]\nrevenue = \"x\"\n"),
        ],
    );
    let plan = judge::plan(&t, Some("vercel/review/model"))
        .unwrap()
        .unwrap();
    // `tone` gets a malformed reply, then a good one; `total` never parses.
    let (script, vars) = serve(&[
        (
            "total",
            &["I think it passes.", "{\"holds\": \"yes\"}", "PASS"],
        ),
        ("tone", &["{\"holds\": true}", &holds(true)]),
        ("invented", &[&holds(false)]),
        ("depth", &[r#"{"level": "2", "reason": "ok"}"#]),
    ])
    .await;
    let sb = Files(
        BTreeMap::from([("/app/memo.md".to_string(), MEMO.as_bytes().to_vec())]),
        Mutex::new(Vec::new()),
    );
    let evidence = judge::fetch(&sb, "/app", &plan).await;
    let reviewer = judge::connect(&plan, &vars).await.unwrap();
    let out = tempfile::tempdir().unwrap();
    let judged = judge::judge(&plan, &evidence, &reviewer, out.path()).await;
    assert_eq!(judged.failures.len(), 1);
    assert_eq!(judged.failures[0].0, "total");
    assert!(!judged.verdicts.contains_key("total"));
    assert_eq!(judged.verdicts["tone"], Verdict::Pass);
    let reqs = script.lock().unwrap().requests.len();
    assert_eq!(reqs, 3 + 2 + 1 + 1);
    let line: Value = serde_json::from_str(
        std::fs::read_to_string(out.path().join("judgments.jsonl"))
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(line["status"], "not-judged");
    assert_eq!(line["attempts"].as_array().unwrap().len(), 3);
    // The retry says why, in the same criterion's context only.
    let retry = &script.lock().unwrap().requests[..];
    let second_total = retry
        .iter()
        .filter(|r| r.to_string().contains("CRITERION `total`"))
        .nth(1)
        .unwrap()
        .to_string();
    assert!(second_total.contains("Your answer cannot be used"));
    assert!(!second_total.contains("CRITERION `tone`"));
}

#[tokio::test]
async fn evidence_is_capped_and_unreadable_files_are_noted() {
    let (_d, t) = task(
        json!({
            "x-judge": {"model": MODEL},
            "criteria": [{"id": "all", "text": "The report covers every region.", "judge": "llm",
                "points": 1, "evidence": ["/out", "missing.md", "trajectory"]}]
        }),
        &[],
    );
    let plan = judge::plan(&t, None).unwrap().unwrap();
    assert!(plan.reference.is_none());
    let long = "region line\n".repeat(20_000); // 240,000 characters
    let sb = Files(
        BTreeMap::from([
            ("/out/report.md".to_string(), long.into_bytes()),
            ("/out/chart.png".to_string(), b"\x89PNG\0\0\0".to_vec()),
            ("/out/empty.csv".to_string(), Vec::new()),
        ]),
        Mutex::new(Vec::new()),
    );
    let evidence = judge::fetch(&sb, "/app", &plan).await;
    assert_eq!(evidence["/out"].len(), 3);
    let user = judge::user_prompt(&plan, &plan.items[0], &evidence);
    assert!(user.contains("=== AGENT'S WORK: /out/report.md (markdown, 240000 bytes) ==="));
    assert!(user.contains(&format!(
        "[... cut for size: {} more characters of this file were not shown]",
        240_000 - judge::FILE_TEXT_CAP
    )));
    assert!(user.contains("/out/chart.png — not shown: a binary .png file"));
    assert!(
        user.contains("/app/missing.md — not shown: the agent left no file here"),
        "{}",
        &user[user.len() - 600..]
    );
    assert!(user.contains("trajectory — not shown"));
    assert!(!user.contains("## REFERENCE"));
    assert!(user.len() < judge::FILE_TEXT_CAP + 4_000);
}

#[test]
fn bad_judge_settings_are_the_tasks_to_fix() {
    let (_d, t) = task(
        json!({"x-judge": {"model": "no-provider"}, "criteria": [
            {"id": "a", "text": "t", "judge": "llm", "points": 1, "evidence": ["a.md"]}]}),
        &[],
    );
    assert!(judge::plan(&t, None).unwrap_err().contains("x-judge.model"));
    let (_d, t) = task(
        json!({"criteria": [{"id": "a", "text": "t", "judge": "llm", "points": 1,
            "evidence": ["a.md"], "reference": "verifier/nope.md"}]}),
        &[],
    );
    assert!(
        judge::plan(&t, None)
            .unwrap_err()
            .starts_with("a: reference: verifier/nope.md cannot be read")
    );
    // The run's model wins; a rubric without x-judge gets the default.
    let (_d, t) = task(
        json!({"criteria": [{"id": "a", "text": "t", "judge": "llm", "points": 1, "evidence": ["a.md"]}]}),
        &[],
    );
    assert_eq!(
        judge::plan(&t, None).unwrap().unwrap().model,
        judge::DEFAULT_MODEL
    );
    assert_eq!(
        judge::plan(&t, Some("openai/gpt-x"))
            .unwrap()
            .unwrap()
            .model,
        "openai/gpt-x"
    );
}
