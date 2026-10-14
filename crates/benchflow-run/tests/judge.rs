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

/// Whether the `!slow-once:` probe already ran past its time.
static SLOW_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the `!exec-fail-once:` probe's exec already failed.
static FAIL_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What a runtime prints when it fails an exec in a running container.
const EXEC_FAILED: &str =
    "OCI runtime exec failed: exec failed: unable to start container process: unknown";

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
        // A sandbox that has gone away, for the test that says so.
        if self.0.contains_key("!gone") {
            return Err(SandboxError::Transport("connection reset by peer".into()));
        }
        let answer = |stdout: String, timed_out: bool| ExecOutput {
            exit_code: if timed_out { -1 } else { 0 },
            stdout,
            stderr: String::new(),
            timed_out,
            duration: Duration::from_millis(1),
        };
        // The backend failing an exec while the container runs: exit 126
        // and no output, as Docker's runtime does under load.
        let failed = || ExecOutput {
            exit_code: 126,
            stdout: String::new(),
            stderr: EXEC_FAILED.into(),
            timed_out: false,
            duration: Duration::from_millis(1),
        };
        // The liveness check after a slow probe; `!stuck` never answers,
        // and under `!exec-dead` the backend fails every exec.
        if exec.command == "true" {
            if self.0.contains_key("!exec-dead") {
                return Ok(failed());
            }
            return Ok(answer(String::new(), self.0.contains_key("!stuck")));
        }
        // The probe names the path as `f=<path>;`, quoted when it must be.
        let path = exec
            .command
            .strip_prefix("f=")
            .and_then(|r| r.split(';').next())
            .unwrap_or("")
            .trim_matches('\'');
        // `!slow:<path>`: the probe runs past its time; `!listing:<path>`:
        // what it prints.
        if self.0.contains_key(&format!("!slow:{path}")) {
            return Ok(answer(String::new(), true));
        }
        // `!slow-once:<path>`: only the first probe runs past its time.
        if self.0.contains_key(&format!("!slow-once:{path}"))
            && !SLOW_ONCE.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(answer(String::new(), true));
        }
        // `!exec-fail:<path>`: the backend fails the probe's exec every
        // time; `!exec-fail-once:<path>`: only the first time.
        if self.0.contains_key("!exec-dead") || self.0.contains_key(&format!("!exec-fail:{path}")) {
            return Ok(failed());
        }
        if self.0.contains_key(&format!("!exec-fail-once:{path}"))
            && !FAIL_ONCE.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(failed());
        }
        if let Some(listing) = self.0.get(&format!("!listing:{path}")) {
            return Ok(answer(String::from_utf8_lossy(listing).into(), false));
        }
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
        Ok(answer(stdout, false))
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
        // `!down-lost:<path>`: the sandbox drops the download;
        // `!down-gone:<path>`: the file is no longer there.
        if self.0.contains_key(&format!("!down-lost:{remote}")) {
            return Err(SandboxError::Transport("502 Bad Gateway".into()));
        }
        if self.0.contains_key(&format!("!down-full:{remote}")) {
            return Err(SandboxError::Transport(
                "copying it out of Daytona sandbox x failed: its disk is full".into(),
            ));
        }
        if self.0.contains_key(&format!("!down-gone:{remote}")) {
            return Err(SandboxError::io(
                format!("downloading {remote}"),
                std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
            ));
        }
        // `!grows:<path>`: after the probe, the agent's leftover process
        // grows the file far past the cap; `!endless:<path>`: the copy
        // never ends. Copied without a cap, it writes on and on.
        if self.0.contains_key(&format!("!grows:{remote}"))
            || self.0.contains_key(&format!("!endless:{remote}"))
        {
            self.1.lock().unwrap().push(format!("uncapped {remote}"));
            if self.0.contains_key(&format!("!endless:{remote}")) {
                std::future::pending::<()>().await;
            }
            std::fs::write(local, vec![0u8; 32 << 20]).unwrap();
            return Ok(());
        }
        self.1.lock().unwrap().push(remote.to_string());
        std::fs::write(local, &self.0[remote]).unwrap();
        // `!mode000:<path>`: the copy keeps the mode the agent gave the
        // file, none at all.
        if self.0.contains_key(&format!("!mode000:{remote}")) {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(local, std::fs::Permissions::from_mode(0o000)).unwrap();
        }
        Ok(())
    }
    async fn download_capped(
        &self,
        remote: &str,
        local: &Path,
        max_bytes: u64,
    ) -> Result<(), SandboxError> {
        if self.0.contains_key(&format!("!grows:{remote}")) {
            // Stopped at the cap, part of it written.
            std::fs::write(local, vec![0u8; 1 << 20]).unwrap();
            return Err(SandboxError::io(
                format!("copying {remote} out of the sandbox"),
                std::io::Error::new(
                    std::io::ErrorKind::FileTooLarge,
                    format!("past its {max_bytes}-byte cap"),
                ),
            ));
        }
        if self.0.contains_key(&format!("!endless:{remote}")) {
            std::fs::write(local, vec![0u8; 1 << 20]).unwrap();
            std::future::pending::<()>().await;
        }
        self.download(remote, local).await
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
    let evidence = judge::fetch(&sb, "/app", &plan).await.unwrap();
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
    let evidence = judge::fetch(&sb, "/app", &plan).await.unwrap();
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
    let evidence = judge::fetch(&sb, "/app", &plan).await.unwrap();
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

/// A sandbox that cannot say what the evidence is fails the fetch, rather
/// than have the agent's work judged as missing.
#[tokio::test]
async fn evidence_from_a_sandbox_that_is_gone_is_not_judged_missing() {
    let (_d, t) = task(
        json!({"criteria": [{"id": "a", "text": "t", "judge": "llm", "points": 1, "evidence": ["memo.md"]}]}),
        &[],
    );
    let plan = judge::plan(&t, Some("openai/gpt-x")).unwrap().unwrap();
    let sb = Files(
        BTreeMap::from([("!gone".to_string(), Vec::new())]),
        Mutex::new(Vec::new()),
    );
    assert!(judge::fetch(&sb, "/app", &plan).await.is_err());
}

fn one_item(evidence: &[&str]) -> (tempfile::TempDir, Plan) {
    let (d, t) = task(
        json!({"criteria": [{"id": "a", "text": "t", "judge": "llm", "points": 1, "evidence": evidence}]}),
        &[],
    );
    let plan = judge::plan(&t, Some("openai/gpt-x")).unwrap().unwrap();
    (d, plan)
}

/// A probe that runs past its time while the sandbox still answers (a
/// folder of very many files) leaves that item unreadable and the rest
/// judged; only a sandbox that then does not answer fails the fetch.
/// Guards the D7b fix of 431d4f35, which made every slow probe a lost
/// sandbox, so an agent could have its trial retried instead of judged.
#[tokio::test]
async fn a_slow_evidence_check_is_not_a_lost_sandbox() {
    let (_d, plan) = one_item(&["big", "memo.md"]);
    let mut files = BTreeMap::from([
        ("!slow:/app/big".to_string(), Vec::new()),
        ("/app/memo.md".to_string(), MEMO.as_bytes().to_vec()),
    ]);
    let sb = Files(files.clone(), Mutex::new(Vec::new()));
    let evidence = judge::fetch(&sb, "/app", &plan).await.unwrap();
    match &evidence["big"][..] {
        [f] => match &f.text {
            judge::Text::Unreadable(why) => assert!(why.contains("ran past 60 s"), "{why}"),
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        evidence["memo.md"][0].text,
        judge::Text::Read { .. }
    ));
    files.insert("!stuck".into(), Vec::new());
    let sb = Files(files, Mutex::new(Vec::new()));
    assert!(judge::fetch(&sb, "/app", &plan).await.is_err());
    // A probe that is slow once (a busy backend) is asked again, and the
    // item is read (REVIEW-CORRECT L8).
    let (_d, plan) = one_item(&["memo.md"]);
    let sb = Files(
        BTreeMap::from([
            ("!slow-once:/app/memo.md".to_string(), Vec::new()),
            ("/app/memo.md".to_string(), MEMO.as_bytes().to_vec()),
        ]),
        Mutex::new(Vec::new()),
    );
    let evidence = judge::fetch(&sb, "/app", &plan).await.unwrap();
    assert!(matches!(
        evidence["memo.md"][0].text,
        judge::Text::Read { .. }
    ));
}

/// Files the probe reached through links that stay in the agent's folder
/// are named as the item's and downloaded from where they are; a listed
/// path outside the item, a hard-linked file, and one without a size are
/// never downloaded, and the last two are named rather than dropped (the
/// item would read as an empty folder). Guards the D7b fixes of 859fc713.
#[tokio::test]
async fn evidence_found_through_a_link_is_named_as_the_item() {
    let (_d, plan) = one_item(&["out"]);
    let listing = "BF real /srv/app/build/out\n\
        BF file 6 /srv/app/build/out/r.md\n\
        BF file 6 /tests/answers.toml\n\
        BF hardlink /srv/app/build/out/h.md\n\
        BF unsized /srv/app/build/out/u.md\n\
        BF file ? /srv/app/build/out/v.md\n";
    let sb = Files(
        BTreeMap::from([
            ("!listing:/app/out".to_string(), listing.as_bytes().to_vec()),
            ("/srv/app/build/out/r.md".to_string(), b"report".to_vec()),
            ("/tests/answers.toml".to_string(), b"secret".to_vec()),
        ]),
        Mutex::new(Vec::new()),
    );
    let evidence = judge::fetch(&sb, "/app", &plan).await.unwrap();
    let got: Vec<(&str, bool)> = evidence["out"]
        .iter()
        .map(|f| (f.path.as_str(), matches!(f.text, judge::Text::Read { .. })))
        .collect();
    assert_eq!(
        got,
        vec![
            ("/app/out/r.md", true),
            ("/app/out/h.md", false),
            ("/app/out/u.md", false),
            ("/app/out/v.md", false),
        ]
    );
    assert_eq!(
        *sb.1.lock().unwrap(),
        vec!["/srv/app/build/out/r.md".to_string()]
    );
}

/// A file read only in part (too large to unpack to its end) says so to
/// the judge, and the count of characters not shown says "more than".
#[test]
fn a_file_read_only_in_part_says_so_to_the_judge() {
    let (_d, plan) = one_item(&["big.xlsx"]);
    let shown = |text: &str, cut: bool| {
        let evidence = BTreeMap::from([(
            "big.xlsx".to_string(),
            vec![judge::Fetched {
                path: "/app/big.xlsx".into(),
                bytes: Some(9),
                text: judge::Text::Read {
                    kind: "xlsx",
                    text: text.into(),
                    cut,
                },
            }],
        )]);
        judge::user_prompt(&plan, &plan.items[0], &evidence)
    };
    let short = shown("A1=1", true);
    assert!(
        short.contains("the rest of this file was not read; it is too large to read to its end"),
        "{short}"
    );
    let long = "x".repeat(judge::FILE_TEXT_CAP + 10);
    assert!(shown(&long, true).contains("more than 10 more characters"));
    assert!(shown(&long, false).contains("cut for size: 10 more characters"));
    assert!(!shown("A1=1", false).contains("cut for size"));
}

/// A download the sandbox fails is not judged as missing work: the fetch
/// fails (the trial ends as a lost sandbox and is retried), as when the
/// probe cannot run. A file gone from the sandbox is the work's own state
/// and is shown as such. Guards the D7c fix of REVIEW-OUTCOMES M1.
#[tokio::test]
async fn a_download_the_sandbox_fails_is_not_judged_missing() {
    let (_d, plan) = one_item(&["memo.md"]);
    let files = |fault: &str| {
        Files(
            BTreeMap::from([
                (format!("{fault}/app/memo.md"), Vec::new()),
                ("/app/memo.md".to_string(), MEMO.as_bytes().to_vec()),
            ]),
            Mutex::new(Vec::new()),
        )
    };
    let err = judge::fetch(&files("!down-lost:"), "/app", &plan)
        .await
        .unwrap_err();
    assert!(err.is_transient(), "{err}");
    let evidence = judge::fetch(&files("!down-gone:"), "/app", &plan)
        .await
        .unwrap();
    match &evidence["memo.md"][0].text {
        judge::Text::Unreadable(why) => assert!(why.contains("gone"), "{why}"),
        other => panic!("{other:?}"),
    }
}

/// What the agent did to its own evidence never ends the trial as an
/// error rerun on every resume: a file it made unreadable (`chmod 000`) is
/// read all the same (the host copy is BenchFlow's), a name as long as the
/// system allows is downloaded, and a disk it filled leaves the file
/// unreadable. A "gone" file from a sandbox that no longer answers fails
/// the fetch instead (a stopped container looks the same). Guards the D7d
/// fixes of REVIEW-OUTCOMES (d7c) H1, H2 and M1.
#[tokio::test]
async fn what_the_agent_did_to_its_evidence_is_judged_not_errored() {
    let one = |fault: &str, path: &str| {
        Files(
            BTreeMap::from([
                (format!("{fault}{path}"), Vec::new()),
                (path.to_string(), MEMO.as_bytes().to_vec()),
            ]),
            Mutex::new(Vec::new()),
        )
    };
    let (_d, plan) = one_item(&["memo.md"]);
    let evidence = judge::fetch(&one("!mode000:", "/app/memo.md"), "/app", &plan)
        .await
        .expect("a file the agent made unreadable is no error of yours");
    assert!(matches!(
        evidence["memo.md"][0].text,
        judge::Text::Read { .. }
    ));
    let long = format!("{}.md", "n".repeat(250));
    let (_d, plan) = one_item(&[long.as_str()]);
    let evidence = judge::fetch(&one("", &format!("/app/{long}")), "/app", &plan)
        .await
        .expect("a 253-byte name is downloaded");
    assert!(matches!(evidence[&long][0].text, judge::Text::Read { .. }));
    let (_d, plan) = one_item(&["memo.md"]);
    let evidence = judge::fetch(&one("!down-full:", "/app/memo.md"), "/app", &plan)
        .await
        .unwrap();
    match &evidence["memo.md"][0].text {
        judge::Text::Unreadable(why) => assert!(why.contains("disk was full"), "{why}"),
        other => panic!("{other:?}"),
    }
    let mut dead = one("!down-gone:", "/app/memo.md");
    dead.0.insert("!stuck".into(), Vec::new());
    assert!(judge::fetch(&dead, "/app", &plan).await.is_err());
}

/// A probe that runs always exits 0, so one the backend failed (exit 126,
/// no `BF` line, the container still running) is not an empty folder: the
/// sandbox is asked whether it answers, the probe runs once more, and a
/// second failure fails the fetch (R103, retried), as a dead sandbox's
/// does. A probe that ran and listed nothing is still an empty folder.
/// Guards the D7e fix of REVIEW-B (d7d) F2: the failed exec was judged as
/// "an empty folder", so the criterion was judged without the agent's work.
#[tokio::test]
async fn an_evidence_probe_the_backend_failed_is_not_an_empty_folder() {
    let (_d, plan) = one_item(&["out"]);
    let with = |fault: &str| {
        Files(
            BTreeMap::from([
                (fault.to_string(), Vec::new()),
                ("/app/out/r.md".to_string(), MEMO.as_bytes().to_vec()),
            ]),
            Mutex::new(Vec::new()),
        )
    };
    let err = judge::fetch(&with("!exec-fail:/app/out"), "/app", &plan)
        .await
        .expect_err("a probe the backend failed twice fails the fetch");
    assert!(matches!(err, SandboxError::Transport(_)), "{err:?}");
    assert!(err.to_string().contains("exit 126"), "{err}");
    let err = judge::fetch(&with("!exec-dead"), "/app", &plan)
        .await
        .expect_err("a sandbox that fails every exec does not answer");
    assert!(matches!(err, SandboxError::Transport(_)), "{err:?}");
    // A backend's hiccup passes: the second probe lists the agent's work.
    let evidence = judge::fetch(&with("!exec-fail-once:/app/out"), "/app", &plan)
        .await
        .unwrap();
    assert_eq!(evidence["out"][0].path, "/app/out/r.md");
    assert!(matches!(evidence["out"][0].text, judge::Text::Read { .. }));
    // A probe that ran (exit 0) and listed nothing is still an empty folder.
    let evidence = judge::fetch(&with("!listing:/app/out"), "/app", &plan)
        .await
        .unwrap();
    match &evidence["out"][..] {
        [f] => match &f.text {
            judge::Text::Unreadable(why) => assert_eq!(why, "an empty folder"),
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

/// Review F2-A-r8 M2: the probe reads an evidence file's size before the
/// copy, and a root agent's leftover process can grow the file in between,
/// or make the copy never end. The copy is capped and has a deadline. A
/// file grown past the cap is unreadable, never a host disk filled; a copy
/// that never ends fails the fetch (review F2-A-r9 M3: a slow backend
/// looks the same, and judging the work as missing would score it), and
/// the judge never waits for ever.
#[tokio::test(start_paused = true)]
async fn evidence_grown_after_its_probe_is_unreadable_or_fails_the_fetch() {
    let (_d, plan) = one_item(&["memo.md"]);
    for fault in ["!grows:", "!endless:"] {
        let sb = Files(
            BTreeMap::from([
                (format!("{fault}/app/memo.md"), Vec::new()),
                ("/app/memo.md".to_string(), MEMO.as_bytes().to_vec()),
            ]),
            Mutex::new(Vec::new()),
        );
        let start = tokio::time::Instant::now();
        let evidence =
            tokio::time::timeout(Duration::from_secs(3600), judge::fetch(&sb, "/app", &plan))
                .await
                .unwrap_or_else(|_| panic!("{fault} the copy never ended"));
        assert!(
            !sb.1
                .lock()
                .unwrap()
                .iter()
                .any(|c| c.starts_with("uncapped")),
            "{fault} the evidence was copied without a cap"
        );
        if fault == "!endless:" {
            // Cut at the copy's deadline, 3 minutes, as the sandbox's.
            let waited = start.elapsed().as_secs();
            assert!((180..240).contains(&waited), "waited {waited}s");
            match evidence {
                Err(SandboxError::Transport(why)) => {
                    assert!(why.contains("did not finish within 180s"), "{why}")
                }
                other => panic!("{fault} {other:?}"),
            }
            continue;
        }
        let evidence = evidence.expect("the file's own doing is no error");
        match &evidence["memo.md"][0].text {
            judge::Text::Unreadable(why) => assert!(why.contains("limit"), "{fault} {why}"),
            other => panic!("{fault} {other:?}"),
        }
    }
}
