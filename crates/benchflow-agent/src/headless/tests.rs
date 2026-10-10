//! The runner against fake processes, and helpers for the parser tests.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_sandbox::{Process, ProcessHandle, SandboxError};
use serde_json::json;
use tokio::io::AsyncWriteExt;

use super::*;
use crate::acp::{AcpEvent, OutputStream};
use crate::trajectory::{Event, Redactor, Usage};

/// What a fake process does.
#[derive(Clone, Default)]
pub(crate) struct Script {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit: i32,
    /// Keep running (silent) after writing, until killed.
    pub hang: bool,
}

struct FakeHandle {
    task: Option<tokio::task::JoinHandle<i32>>,
    killed: Arc<Mutex<bool>>,
}

#[async_trait]
impl ProcessHandle for FakeHandle {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        match self.task.take() {
            Some(t) => Ok(t.await.unwrap_or(137)),
            None => Ok(137),
        }
    }
    async fn kill(&mut self) -> Result<(), SandboxError> {
        *self.killed.lock().unwrap() = true;
        if let Some(t) = self.task.take() {
            t.abort();
        }
        Ok(())
    }
}

pub(crate) fn fake_process(script: Script) -> (Process, Arc<Mutex<bool>>) {
    let (client_stdin, _agent_stdin) = tokio::io::duplex(1024);
    let (mut agent_stdout, client_stdout) = tokio::io::duplex(64 * 1024);
    let (mut agent_stderr, client_stderr) = tokio::io::duplex(64 * 1024);
    let task = tokio::spawn(async move {
        let _ = agent_stderr.write_all(&script.stderr).await;
        let _ = agent_stdout.write_all(&script.stdout).await;
        if script.hang {
            std::future::pending::<()>().await;
        }
        drop(agent_stdout);
        drop(agent_stderr);
        script.exit
    });
    let killed = Arc::new(Mutex::new(false));
    (
        Process {
            stdin: Box::pin(client_stdin),
            stdout: Box::pin(client_stdout),
            stderr: Box::pin(client_stderr),
            handle: Box::new(FakeHandle {
                task: Some(task),
                killed: killed.clone(),
            }),
        },
        killed,
    )
}

/// A finished run, for assertions.
pub(crate) struct TestRun {
    pub result: Result<Finished, HeadlessError>,
    pub events: Vec<Event>,
    pub usage: Option<Usage>,
    pub cost_usd: Option<f64>,
    /// Agent-log lines (stdout noise and stderr), as the sink got them.
    pub log: Vec<String>,
    pub progress: usize,
}

pub(crate) fn events_of(run: &TestRun) -> &[Event] {
    &run.events
}

pub(crate) async fn run_script(format: &str, script: Script, options: RunnerOptions) -> TestRun {
    let format = OutputFormat::parse(format).expect("a known format");
    let log = Arc::new(Mutex::new(Vec::new()));
    let progress = Arc::new(Mutex::new(0usize));
    let sink = {
        let log = log.clone();
        let progress = progress.clone();
        Arc::new(move |event: AcpEvent| match event {
            AcpEvent::AgentOutput { line, .. } => log.lock().unwrap().push(line),
            AcpEvent::Progress(_) => *progress.lock().unwrap() += 1,
            _ => {}
        })
    };
    let runner = Runner::new(RunnerOptions {
        sink: Some(sink),
        ..options
    });
    let (process, _) = fake_process(script);
    let result = runner.run(process, format.parser(), "Do the task.").await;
    let rec = runner.recording();
    let log = log.lock().unwrap().clone();
    let progress = *progress.lock().unwrap();
    TestRun {
        result,
        events: rec.snapshot(),
        usage: rec.usage(),
        cost_usd: rec.cost_usd(),
        log,
        progress,
    }
}

pub(crate) async fn run_lines_exit(format: &str, lines: &[String], exit: i32) -> TestRun {
    let mut stdout = lines.join("\n").into_bytes();
    stdout.push(b'\n');
    run_script(
        format,
        Script {
            stdout,
            exit,
            ..Script::default()
        },
        RunnerOptions::default(),
    )
    .await
}

pub(crate) async fn run_lines(format: &str, lines: &[String]) -> TestRun {
    run_lines_exit(format, lines, 0).await
}

fn codex_message(text: &str) -> String {
    json!({"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": text}})
        .to_string()
}

#[test]
fn every_format_has_a_parser_and_a_name() {
    for format in OutputFormat::ALL {
        assert_eq!(OutputFormat::parse(format.name()), Some(format));
        let _ = format.parser();
    }
    assert_eq!(OutputFormat::parse("json"), None);
}

#[tokio::test]
async fn a_truncated_last_line_and_unknown_events_go_to_the_log() {
    let stdout = format!(
        "{}\n{}\n{{\"type\":\"item.completed\",\"item\":{{\"id\":\"x\",\"type\":\"agent_mes",
        json!({"type": "brand.new.event", "data": 1}),
        codex_message("hello"),
    );
    let run = run_script(
        "codex-json",
        Script {
            stdout: stdout.into_bytes(),
            ..Script::default()
        },
        RunnerOptions::default(),
    )
    .await;
    assert!(run.result.is_ok(), "{:?}", run.result);
    assert!(matches!(&run.events[1], Event::AgentMessage(t) if t.text == "hello"));
    assert_eq!(run.log.len(), 2, "{:?}", run.log);
    assert!(run.log[0].contains("brand.new.event"));
    assert!(run.log[1].ends_with("agent_mes"));
    assert!(run.progress >= 1);
}

#[tokio::test]
async fn a_multi_megabyte_line_is_shrunk_and_still_parsed() {
    let big = "x".repeat(5 * 1024 * 1024);
    let line = json!({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": big}
    ]}});
    let lines = vec![
        json!({"type": "assistant", "message": {"id": "m", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat big"}}]}})
        .to_string(),
        line.to_string(),
        json!({"type": "assistant", "message": {"id": "m2", "content": [{"type": "text", "text": "ok"}]}})
            .to_string(),
    ];
    let run = run_lines("claude-stream-json", &lines).await;
    assert!(run.result.is_ok(), "{:?}", run.result);
    let Event::ToolCall(call) = &run.events[1] else {
        panic!("{:?}", run.events)
    };
    assert_eq!(call.status, crate::trajectory::ToolCallStatus::Completed);
    let size = serde_json::to_string(&call.content).unwrap().len();
    assert!(
        size < 2 * 1024 * 1024,
        "content was not shrunk: {size} bytes"
    );
    // A long plain-text line is cut, not dropped.
    let run = run_lines("text", &["y".repeat(3 * 1024 * 1024)]).await;
    let Event::AgentMessage(t) = &run.events[1] else {
        panic!()
    };
    assert!(t.text.contains("BenchFlow cut this line"));
}

#[tokio::test(start_paused = true)]
async fn the_wall_clock_budget_stops_the_agent() {
    let run = run_script(
        "codex-json",
        Script {
            stdout: format!("{}\n", codex_message("working")).into_bytes(),
            hang: true,
            ..Script::default()
        },
        RunnerOptions {
            wall: Some(Duration::from_secs(30)),
            idle: Some(Duration::from_secs(600)),
            ..RunnerOptions::default()
        },
    )
    .await;
    let err = run.result.unwrap_err();
    assert!(matches!(err, HeadlessError::Timeout { .. }), "{err}");
    assert_eq!(err.category(), "timeout");
    assert!(
        matches!(run.events.last(), Some(Event::AgentTimeout(_))),
        "{:?}",
        run.events
    );
}

#[tokio::test(start_paused = true)]
async fn a_silent_agent_is_stopped_with_more_grace_while_a_tool_runs() {
    let idle = Duration::from_secs(60);
    let started = tokio::time::Instant::now();
    let run = run_script(
        "codex-json",
        Script {
            stdout: format!("{}\n", codex_message("thinking")).into_bytes(),
            hang: true,
            ..Script::default()
        },
        RunnerOptions {
            idle: Some(idle),
            ..RunnerOptions::default()
        },
    )
    .await;
    assert!(matches!(run.result, Err(HeadlessError::Idle { .. })));
    let quiet = started.elapsed();
    assert!(quiet >= idle && quiet < idle * 2, "{quiet:?}");

    let started = tokio::time::Instant::now();
    let open_call = json!({"type": "item.started", "item": {"id": "c", "type": "command_execution", "command": "make", "status": "in_progress"}});
    let run = run_script(
        "codex-json",
        Script {
            stdout: format!("{open_call}\n").into_bytes(),
            hang: true,
            ..Script::default()
        },
        RunnerOptions {
            idle: Some(idle),
            ..RunnerOptions::default()
        },
    )
    .await;
    let Err(HeadlessError::Idle {
        pending_tool_calls, ..
    }) = &run.result
    else {
        panic!("{:?}", run.result)
    };
    assert_eq!(pending_tool_calls, &["c".to_string()]);
    assert!(started.elapsed() >= idle * 3, "{:?}", started.elapsed());
}

#[tokio::test]
async fn a_crash_before_any_output_did_not_start_and_keeps_stderr() {
    let run = run_script(
        "claude-stream-json",
        Script {
            stderr: b"node: /lib/libc.so.6: version `GLIBC_2.34' not found\n".to_vec(),
            exit: 127,
            ..Script::default()
        },
        RunnerOptions::default(),
    )
    .await;
    let err = run.result.unwrap_err();
    let HeadlessError::Exited {
        code,
        started,
        stderr_tail,
    } = &err
    else {
        panic!("{err:?}")
    };
    assert_eq!((*code, *started), (Some(127), false));
    assert!(stderr_tail.contains("GLIBC_2.34"));
    assert!(err.to_string().contains("exited with code 127"), "{err}");
    assert_eq!(run.log.len(), 1);
}

#[tokio::test]
async fn a_login_error_on_stderr_is_recognized() {
    let run = run_script(
        "gemini-text",
        Script {
            stderr: b"Error: When using Gemini API, you must specify the GEMINI_API_KEY environment variable.\nAPI key not valid. Please pass a valid API key.\n".to_vec(),
            exit: 1,
            ..Script::default()
        },
        RunnerOptions::default(),
    )
    .await;
    let err = run.result.unwrap_err();
    assert_eq!(err.category(), "provider_auth", "{err}");
    assert_eq!(err.owner(), crate::acp::Owner::Login);
    assert!(!err.is_retryable());
}

#[tokio::test]
async fn an_agent_that_exits_cleanly_having_done_nothing_is_an_error() {
    let run = run_script("codex-json", Script::default(), RunnerOptions::default()).await;
    let err = run.result.unwrap_err();
    assert_eq!(err.category(), "agent_error", "{err}");
}

#[tokio::test]
async fn secrets_never_reach_the_log_or_the_error() {
    let secret = "sk-test-0123456789abcdef";
    let run = run_script(
        "codex-json",
        Script {
            stdout: format!("debug: using key {secret}\n").into_bytes(),
            stderr: format!("fatal: key {secret} refused\n").into_bytes(),
            exit: 2,
            ..Script::default()
        },
        RunnerOptions {
            redactor: Redactor::new([secret.to_string()]),
            ..RunnerOptions::default()
        },
    )
    .await;
    let err = run.result.unwrap_err();
    assert!(!err.to_string().contains(secret), "{err}");
    assert_eq!(run.log.len(), 2);
    assert!(run.log.iter().all(|l| !l.contains(secret)), "{:?}", run.log);
    let _ = OutputStream::Stdout;
}

#[tokio::test]
async fn a_long_error_line_is_cut_for_the_message() {
    // Gemini CLI prints the API's whole JSON error, newlines escaped.
    let error = format!(
        "Error when talking to Gemini API _ApiError: {{\"error\":{{\"message\":\"{{\\n  \\\"error\\\": {{\\n    \\\"message\\\": \\\"API key not valid. Please pass a valid API key.\\\",{}\"}}}}\n",
        "x".repeat(2000)
    );
    let run = run_script(
        "gemini-text",
        Script {
            stderr: error.into_bytes(),
            exit: 1,
            ..Script::default()
        },
        RunnerOptions::default(),
    )
    .await;
    let err = run.result.unwrap_err();
    assert_eq!(err.category(), "provider_auth");
    let text = err.to_string();
    assert!(text.chars().count() < 400, "{text}");
    assert!(!text.contains("\\n"), "{text}");
}
