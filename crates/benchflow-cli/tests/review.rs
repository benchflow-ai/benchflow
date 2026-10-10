//! `bench check --review` and `bench results --analyze` through the binary,
//! against a fake model server on loopback (no network): what is printed,
//! what is saved, and what a bad answer does.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use assert_cmd::Command;
use serde_json::{Value, json};

const KEY: &str = "vck_cli_review_test_key_0123456789";

/// A server that answers every chat completion with the next reply (the
/// last one repeats) and lists two upstreams, Baseten among them. Returns
/// its base URL and the request bodies it saw.
fn serve(replies: Vec<String>) -> (String, Arc<Mutex<Vec<Value>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let log = seen.clone();
    std::thread::spawn(move || {
        let mut replies = replies
            .into_iter()
            .collect::<std::collections::VecDeque<_>>();
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut length = 0;
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                if header.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = header.split_once(':')
                    && k.eq_ignore_ascii_case("content-length")
                {
                    length = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let answer = if request_line.starts_with("GET ") {
                json!({"data": {"endpoints": [{"provider_name": "baseten"}, {"provider_name": "novita"}]}})
            } else {
                log.lock()
                    .unwrap()
                    .push(serde_json::from_slice(&body).unwrap());
                let text = if replies.len() > 1 {
                    replies.pop_front().unwrap()
                } else {
                    replies.front().cloned().unwrap_or_default()
                };
                json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                       "usage": {"prompt_tokens": 2000, "completion_tokens": 300, "cost": 0.00042}})
            }
            .to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}",
                answer.len()
            );
        }
    });
    (base, seen)
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../benchflow-run/tests/fixtures/tasks")
        .join(name)
}

fn copy_dir(from: &Path, to: &Path) {
    for e in walkdir::WalkDir::new(from) {
        let e = e.unwrap();
        let dest = to.join(e.path().strip_prefix(from).unwrap());
        if e.file_type().is_dir() {
            std::fs::create_dir_all(&dest).unwrap();
        } else {
            std::fs::copy(e.path(), &dest).unwrap();
        }
    }
}

fn bench(dir: &Path, base: &str) -> Command {
    let mut c = Command::cargo_bin("bench").unwrap();
    c.current_dir(dir)
        .env("HOME", dir)
        .env("BENCHFLOW_CACHE_DIR", dir.join(".cache"))
        .env("NO_COLOR", "1")
        .env("AI_GATEWAY_API_KEY", KEY)
        .env("BENCHFLOW_NATIVE_BASE_URL", base);
    c
}

fn review_answer() -> String {
    json!({"verdicts": [{"criterion": "verifier-checks-success", "verdict": "fail", "reason": "it checks /app exists"}],
           "findings": [{"criterion": "verifier-checks-success", "severity": "error", "file": "verifier/test.sh",
                         "line": 3, "quote": "if [ -d /app ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi",
                         "problem": "the verifier passes whenever /app exists",
                         "fix": "compare /app/ready.txt with `ready`"}]})
    .to_string()
}

#[test]
fn check_review_shows_findings_as_the_models_opinion_and_saves_them() {
    let (base, seen) = serve(vec![review_answer()]);
    let d = tempfile::tempdir().unwrap();
    let task = d.path().join("always-passes");
    copy_dir(&fixture("always-passes"), &task);
    let out = bench(d.path(), &base)
        .args(["check", "always-passes", "--review", "--save"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stdout}\n{stderr}");
    assert!(stderr.contains("[M102]"), "{stderr}");
    assert!(stderr.contains("the verifier passes whenever /app exists"));
    assert!(stderr.contains("verifier/test.sh:3"), "{stderr}");
    assert!(stderr.contains("the model points here"));
    assert!(stderr.contains("opinion"));
    assert!(stderr.contains("the model suggests: compare /app/ready.txt"));
    assert!(stdout.contains("Model review by vercel/deepseek/deepseek-v4-flash-0731"));
    assert!(stdout.contains("fail: verifier-checks-success"));
    assert!(stdout.contains("2,300 tokens · $0.0004"), "{stdout}");
    let saved: Value =
        serde_json::from_str(&std::fs::read_to_string(task.join("review.json")).unwrap()).unwrap();
    assert_eq!(saved["findings"][0]["line"], 3);
    assert_eq!(saved["findings"][0]["code"], "M102");
    assert!(saved["opinion"].as_str().unwrap().contains("opinion"));
    let body = seen.lock().unwrap()[0].to_string();
    assert!(body.contains("\"only\":[\"novita\"]"), "never Baseten");
    assert!(!body.contains(KEY));
}

/// Every review code (one per criterion, and M100 for anything else) is
/// shown with its criterion and marked as the model's opinion, as
/// `bench explain` documents it.
#[test]
fn every_review_code_reaches_the_user() {
    let criteria = [
        (
            "M101",
            "success-stated",
            "the instruction states what success is",
        ),
        (
            "M102",
            "verifier-checks-success",
            "the verifier checks what the instruction asks",
        ),
        (
            "M103",
            "oracle-passes",
            "the oracle would pass and the agent cannot see it",
        ),
        (
            "M104",
            "reward-not-gameable",
            "the reward cannot be raised without doing the task",
        ),
        ("M105", "environment-pinned", "the environment is pinned"),
        ("M106", "time-limits-fit", "time limits fit the work"),
        (
            "M107",
            "controls-meaningful",
            "negative controls would fail",
        ),
        ("M100", "something-else", "another problem the model found"),
    ];
    let findings: Vec<Value> = criteria
        .iter()
        .map(|(code, id, _)| {
            json!({"criterion": id, "severity": "warning", "file": "verifier/test.sh", "line": 1,
                   "quote": "", "problem": format!("problem number {code}"), "fix": "do better"})
        })
        .collect();
    let verdicts: Vec<Value> = criteria[..7]
        .iter()
        .map(|(_, id, _)| json!({"criterion": id, "verdict": "fail", "reason": "see the finding"}))
        .collect();
    let answer = json!({"verdicts": verdicts, "findings": findings}).to_string();
    let (base, _) = serve(vec![answer]);
    let d = tempfile::tempdir().unwrap();
    copy_dir(&fixture("always-passes"), &d.path().join("t"));
    let out = bench(d.path(), &base)
        .args(["check", "t", "--review"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    for (code, id, title) in criteria {
        let at = stderr
            .find(&format!("[{code}]: problem number {code}"))
            .unwrap_or_else(|| panic!("{code} is not shown: {stderr}"));
        let block = &stderr[at..];
        let shown_id = if code == "M100" { "other" } else { id };
        assert!(
            block.contains(&format!("criterion: {title} ({shown_id})")),
            "{code}: {block}"
        );
        assert!(block.contains("opinion"), "{code}: {block}");
    }
}

#[test]
fn check_review_with_json_keeps_the_checkers_report() {
    let (base, _) = serve(vec![review_answer()]);
    let d = tempfile::tempdir().unwrap();
    copy_dir(&fixture("always-passes"), &d.path().join("t"));
    let out = bench(d.path(), &base)
        .args(["check", "t", "--review", "--json"])
        .output()
        .unwrap();
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v[0]["diagnostics"].is_array());
    assert_eq!(
        v[0]["review"]["findings"][0]["criterion"],
        "verifier-checks-success"
    );
    assert!(
        !d.path().join("t/review.json").exists(),
        "saved only with --save"
    );
}

#[test]
fn a_model_that_will_not_answer_in_json_is_a_clear_failure() {
    let (base, _) = serve(vec!["The task looks fine to me.".into()]);
    let d = tempfile::tempdir().unwrap();
    copy_dir(&fixture("hello"), &d.path().join("t"));
    let out = bench(d.path(), &base)
        .args(["check", "t", "--review"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{stdout}");
    assert!(
        stdout.contains("did not answer with the JSON asked for"),
        "{stdout}"
    );
    assert!(stdout.contains("The task looks fine to me."));
    assert!(!stdout.contains("panicked"));
}

#[test]
fn a_missing_key_says_where_to_put_one() {
    let d = tempfile::tempdir().unwrap();
    copy_dir(&fixture("hello"), &d.path().join("t"));
    let out = bench(d.path(), "http://127.0.0.1:9/v1")
        .env_remove("AI_GATEWAY_API_KEY")
        .args(["check", "t", "--review"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("AI_GATEWAY_API_KEY"), "{stderr}");
    assert!(stderr.contains("vercel-ai-gateway.env"), "{stderr}");
}

#[test]
fn results_analyze_explains_failed_trials_and_saves_the_analysis() {
    let answer = json!({"part": "provider", "explanation": "every model call was refused with 401",
                        "evidence": [{"file": "trajectory/acp_trajectory.jsonl", "quote": "401 Unauthorized"}],
                        "confidence": "high"})
    .to_string();
    let (base, seen) = serve(vec![answer]);
    let d = tempfile::tempdir().unwrap();
    let job = d.path().join("jobs/j1");
    let trial = job.join("t__1");
    std::fs::create_dir_all(trial.join("trajectory")).unwrap();
    std::fs::write(job.join("summary.json"), "{\"agent\": \"codex-acp\"}").unwrap();
    std::fs::write(
        trial.join("result.json"),
        json!({"task_name": "t", "rollout_name": "t__1", "rewards": {"reward": 0.0}}).to_string(),
    )
    .unwrap();
    std::fs::write(
        trial.join("trajectory/acp_trajectory.jsonl"),
        "{\"type\": \"agent_message\", \"text\": \"unexpected status 401 Unauthorized\"}\n",
    )
    .unwrap();
    let out = bench(d.path(), &base)
        .args(["results", "--analyze"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains("Why trials failed in jobs/j1"), "{stdout}");
    assert!(stdout.contains("the model provider"));
    assert!(stdout.contains("in the evidence sent"));
    assert!(stdout.contains("By cause, in the model's opinion:"));
    assert!(stdout.contains("analyzed 1 of 1 failed trial"));
    let saved: Value =
        serde_json::from_str(&std::fs::read_to_string(job.join("analysis.json")).unwrap()).unwrap();
    assert_eq!(saved["trials"][0]["part"], "provider");
    assert_eq!(saved["by_cause"][0]["trials"][0], "t__1");
    assert!(
        seen.lock().unwrap()[0]
            .to_string()
            .contains("401 Unauthorized")
    );

    // --no-save leaves the folder as it was.
    std::fs::remove_file(job.join("analysis.json")).unwrap();
    bench(d.path(), &base)
        .args(["results", "jobs/j1", "--analyze", "--no-save", "--json"])
        .assert()
        .success();
    assert!(!job.join("analysis.json").exists());
}
