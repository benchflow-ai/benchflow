//! ACP client scenarios against an in-process fake agent.
//!
//! The fake agent speaks over `tokio::io::duplex` pipes and plays message
//! shapes taken from 0.8's tests and recordings (claude-agent-acp 0.81,
//! codex-acp 2.0). Time-dependent scenarios run on Tokio's paused clock.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_sandbox::{ProcessHandle, SandboxError};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::sync::watch;

use super::*;
use crate::Registry;
use crate::trajectory::{Event, Redactor, ToolCallStatus, to_jsonl};

/// The agent's end of the pipes.
struct FakeAgent {
    from_client: Lines<BufReader<DuplexStream>>,
    stdout: Option<DuplexStream>,
    stderr: Option<DuplexStream>,
    exit: watch::Sender<Option<i32>>,
    killed: Arc<AtomicBool>,
}

/// The fake process handle: exits when the agent says, or when killed.
struct FakeHandle {
    exit: watch::Sender<Option<i32>>,
    killed: Arc<AtomicBool>,
}

#[async_trait]
impl ProcessHandle for FakeHandle {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        let mut exit = self.exit.subscribe();
        let code = exit
            .wait_for(Option::is_some)
            .await
            .map_err(|_| SandboxError::Transport("gone".into()))?;
        Ok(code.unwrap_or(-1))
    }

    async fn kill(&mut self) -> Result<(), SandboxError> {
        self.killed.store(true, Ordering::SeqCst);
        self.exit.send_replace(Some(137));
        Ok(())
    }
}

/// A process that never exits and whose kill never returns.
struct HangingHandle;

#[async_trait]
impl ProcessHandle for HangingHandle {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        std::future::pending().await
    }

    async fn kill(&mut self) -> Result<(), SandboxError> {
        std::future::pending().await
    }
}

/// Pipes for the client and the agent that drives them.
fn connect() -> (AgentPipes, FakeAgent) {
    connect_with(64 * 1024)
}

/// Pipes whose stdin holds `stdin_buffer` bytes: when the agent stops
/// reading, the client's writes stall.
fn connect_with(stdin_buffer: usize) -> (AgentPipes, FakeAgent) {
    let (client_stdin, agent_stdin) = tokio::io::duplex(stdin_buffer);
    let (agent_stdout, client_stdout) = tokio::io::duplex(64 * 1024);
    let (agent_stderr, client_stderr) = tokio::io::duplex(64 * 1024);
    let exit = watch::Sender::new(None);
    let killed = Arc::new(AtomicBool::new(false));
    let pipes = AgentPipes {
        stdin: Box::pin(client_stdin),
        stdout: Box::pin(client_stdout),
        stderr: Some(Box::pin(client_stderr)),
        handle: Some(Box::new(FakeHandle {
            exit: exit.clone(),
            killed: killed.clone(),
        })),
    };
    let agent = FakeAgent {
        from_client: BufReader::new(agent_stdin).lines(),
        stdout: Some(agent_stdout),
        stderr: Some(agent_stderr),
        exit,
        killed,
    };
    (pipes, agent)
}

impl FakeAgent {
    /// The next message from the client.
    async fn recv(&mut self) -> Value {
        let line = self
            .from_client
            .next_line()
            .await
            .unwrap()
            .expect("the client closed stdin");
        serde_json::from_str(&line).unwrap()
    }

    /// The next message, which must be a request for `method`: (id, params).
    async fn expect(&mut self, method: &str) -> (Value, Value) {
        let message = self.recv().await;
        assert_eq!(message["method"], method, "{message}");
        (message["id"].clone(), message["params"].clone())
    }

    async fn send_raw(&mut self, line: &str) {
        let stdout = self.stdout.as_mut().expect("stdout is open");
        stdout.write_all(line.as_bytes()).await.unwrap();
        stdout.write_all(b"\n").await.unwrap();
    }

    async fn send(&mut self, message: Value) {
        self.send_raw(&message.to_string()).await;
    }

    async fn respond(&mut self, id: &Value, result: Value) {
        self.send(json!({"jsonrpc": "2.0", "id": id, "result": result}))
            .await;
    }

    async fn fail(&mut self, id: &Value, error: Value) {
        self.send(json!({"jsonrpc": "2.0", "id": id, "error": error}))
            .await;
    }

    async fn update(&mut self, update: Value) {
        self.send(json!({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "sess-1", "update": update}}))
            .await;
    }

    async fn stderr(&mut self, line: &str) {
        let stderr = self.stderr.as_mut().unwrap();
        stderr
            .write_all(format!("{line}\n").as_bytes())
            .await
            .unwrap();
    }

    /// Answer `initialize` and `session/new` like claude-agent-acp 0.81.
    async fn handshake(&mut self, session: Value) -> Value {
        let (id, params) = self.expect("initialize").await;
        self.respond(
            &id,
            json!({
                "protocolVersion": 1,
                "agentInfo": {"name": "@agentclientprotocol/claude-agent-acp", "title": "Claude Agent", "version": "0.81.2"},
                "agentCapabilities": {"loadSession": true, "promptCapabilities": {"image": true, "embeddedContext": true}},
                "authMethods": [],
            }),
        )
        .await;
        let (id, _) = self.expect("session/new").await;
        self.respond(&id, session).await;
        params
    }

    /// Crash: stderr says why, stdout closes, the process exits.
    async fn crash(&mut self, code: i32, why: &str) {
        self.stderr(why).await;
        self.stdout = None;
        self.stderr = None;
        self.exit.send_replace(Some(code));
    }
}

const NO_LIMITS: PromptLimits = PromptLimits {
    wall: None,
    idle: None,
    pending_tool_grace: None,
};

fn session() -> Value {
    json!({"sessionId": "sess-1"})
}

/// Events the client delivered, in order.
#[derive(Clone, Default)]
struct Seen(Arc<Mutex<Vec<AcpEvent>>>);

impl Seen {
    fn options(&self, agent: &str) -> ClientOptions {
        let seen = self.0.clone();
        ClientOptions::new(agent).on_event(move |event| seen.lock().unwrap().push(event))
    }

    fn all(&self) -> Vec<AcpEvent> {
        self.0.lock().unwrap().clone()
    }
}

async fn ready_client(
    options: ClientOptions,
    session_result: Value,
) -> (AcpClient, FakeAgent, Value) {
    let (pipes, mut agent) = connect();
    let mut client = AcpClient::start(pipes, options);
    let (init_params, ()) = tokio::join!(agent.handshake(session_result), async {
        client.initialize().await.unwrap();
        client.new_session("/app", &[]).await.unwrap();
    });
    (client, agent, init_params)
}

fn text_chunk(kind: &str, text: &str) -> Value {
    json!({"sessionUpdate": kind, "content": {"type": "text", "text": text}})
}

#[tokio::test]
async fn initialize_says_what_benchflow_offers() {
    let options = ClientOptions::for_agent(Registry::builtin().find("claude").unwrap(), None);
    let (client, _agent, params) = ready_client(options, session()).await;
    assert_eq!(params["protocolVersion"], 1);
    assert_eq!(
        params["clientCapabilities"],
        json!({
            "fs": {"readTextFile": false, "writeTextFile": false},
            "terminal": false,
            "auth": {"terminal": false},
            "session": {"notices": {}},
            "_meta": {
                "jetbrains": {"air": {"version": 1, "capabilities": ["sessionFailure"]}},
                "subagent-transcript": true,
            },
        })
    );
    assert_eq!(
        params["clientInfo"],
        json!({"name": "benchflow", "version": env!("CARGO_PKG_VERSION")})
    );
    let info = client.agent_info().unwrap();
    assert_eq!(
        info.name.as_deref(),
        Some("@agentclientprotocol/claude-agent-acp")
    );
    assert_eq!(info.capabilities["loadSession"], true);
    assert_eq!(client.session_info().unwrap().session_id, "sess-1");

    // Codex asks for neither extension.
    let options = ClientOptions::for_agent(Registry::builtin().find("codex").unwrap(), None);
    let (_client, _agent, params) = ready_client(options, session()).await;
    assert!(params["clientCapabilities"].get("_meta").is_none());
}

#[tokio::test]
async fn a_turn_records_tool_calls_permissions_and_chunked_text() {
    let seen = Seen::default();
    let (mut client, mut agent, _) = ready_client(seen.options("claude"), session()).await;
    let agent_side = async {
        let (id, params) = agent.expect("session/prompt").await;
        assert_eq!(
            params,
            json!({"sessionId": "sess-1", "prompt": [{"type": "text", "text": "Fix the bug."}]})
        );
        agent
            .update(json!({"sessionUpdate": "available_commands_update", "availableCommands": []}))
            .await;
        agent
            .update(text_chunk("agent_thought_chunk", "Look first."))
            .await;
        agent
            .update(json!({
                "sessionUpdate": "tool_call", "toolCallId": "toolu_1", "title": "Terminal", "kind": "execute",
                "status": "pending", "rawInput": {}, "_meta": {"claudeCode": {"toolName": "Bash"}},
            }))
            .await;
        // claude-agent-acp asks before running a command.
        agent
            .send(json!({
                "jsonrpc": "2.0", "id": 0, "method": "session/request_permission",
                "params": {
                    "sessionId": "sess-1",
                    "toolCall": {"toolCallId": "toolu_1", "title": "pytest -q", "kind": "execute"},
                    "options": [
                        {"optionId": "allow_always", "name": "Always Allow", "kind": "allow_always"},
                        {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "reject", "name": "Reject", "kind": "reject_once"},
                    ],
                },
            }))
            .await;
        let answer = agent.recv().await;
        assert_eq!(
            answer,
            json!({"jsonrpc": "2.0", "id": 0, "result": {"outcome": {"outcome": "selected", "optionId": "allow_always"}}})
        );
        agent
            .update(json!({
                "sessionUpdate": "tool_call_update", "toolCallId": "toolu_1", "title": "pytest -q",
                "rawInput": {"command": "pytest -q"}, "_meta": {"claudeCode": {"toolName": "Bash"}},
            }))
            .await;
        agent
            .update(json!({"sessionUpdate": "usage_update", "used": 5300, "size": 200000}))
            .await;
        agent
            .update(json!({
                "sessionUpdate": "tool_call_update", "toolCallId": "toolu_1", "status": "completed",
                "content": [{"type": "content", "content": {"type": "text", "text": "1 passed"}}],
                "rawOutput": "1 passed\n",
            }))
            .await;
        agent.update(json!({"sessionUpdate": "plan", "entries": [{"content": "Run tests", "priority": "high", "status": "completed"}]})).await;
        for part in ["The tests ", "pass ", "now."] {
            agent.update(text_chunk("agent_message_chunk", part)).await;
        }
        agent
            .respond(
                &id,
                json!({"stopReason": "end_turn", "usage": {"inputTokens": 1200, "outputTokens": 80, "totalTokens": 1280, "cachedReadTokens": 900}}),
            )
            .await;
    };
    let limits = PromptLimits::default();
    let (outcome, ()) = tokio::join!(client.prompt("Fix the bug.", &limits), agent_side);
    let outcome = outcome.unwrap();
    assert_eq!(outcome.stop_reason, StopReason::EndTurn);
    assert_eq!(outcome.usage.unwrap().cached_read_tokens, Some(900));

    let events = client.trajectory().snapshot();
    let kinds: Vec<&str> = events.iter().map(Event::kind).collect();
    assert_eq!(
        kinds,
        [
            "user_message",
            "agent_thought",
            "tool_call",
            "agent_message"
        ]
    );
    let Event::ToolCall(call) = &events[2] else {
        panic!()
    };
    assert_eq!(call.title, "pytest -q");
    assert_eq!(call.status, ToolCallStatus::Completed);
    assert_eq!(call.tool_name.as_deref(), Some("Bash"));
    assert_eq!(call.raw_input, Some(json!({"command": "pytest -q"})));
    let Event::AgentMessage(text) = &events[3] else {
        panic!()
    };
    assert_eq!(text.text, "The tests pass now.");
    let handle = client.trajectory();
    assert_eq!(
        handle.with(|t| t.plan().cloned()).unwrap()[0]["content"],
        "Run tests"
    );
    assert_eq!(handle.with(|t| t.other_updates().len()), 3);

    let seen = seen.all();
    assert!(seen.contains(&AcpEvent::Permission {
        tool_call: Some("toolu_1".into()),
        option_id: "allow_always".into(),
    }));
    let last_progress = seen
        .iter()
        .rev()
        .find_map(|e| match e {
            AcpEvent::Progress(p) => Some(p.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(last_progress.tool_calls, 1);
    assert_eq!(last_progress.last_tool_title.as_deref(), Some("pytest -q"));
    assert_eq!(last_progress.tokens, Some(5300));
}

#[tokio::test]
async fn requests_benchflow_does_not_offer_get_method_not_found() {
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        agent
            .send(json!({"jsonrpc": "2.0", "id": "fs-1", "method": "fs/read_text_file", "params": {"sessionId": "sess-1", "path": "/etc/passwd"}}))
            .await;
        let answer = agent.recv().await;
        assert_eq!(
            answer,
            json!({"jsonrpc": "2.0", "id": "fs-1", "error": {"code": -32601, "message": "Method not found: fs/read_text_file"}})
        );
        agent
            .send(json!({"jsonrpc": "2.0", "id": 7, "method": "terminal/create", "params": {}}))
            .await;
        assert_eq!(agent.recv().await["error"]["code"], -32601);
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let (outcome, ()) = tokio::join!(client.prompt("go", &NO_LIMITS), agent_side);
    assert_eq!(outcome.unwrap().stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn garbage_on_stdout_is_noise_not_a_failure() {
    let seen = Seen::default();
    let mut options = seen.options("x");
    options.redactor = Redactor::new(["login-secret-value"]);
    let (pipes, mut agent) = connect();
    let mut client = AcpClient::start(pipes, options);
    let agent_side = async {
        agent.send_raw("Starting agent... (debug build)").await;
        agent
            .send_raw("{\"level\":\"info\",\"msg\":\"boot\"}")
            .await;
        agent.handshake(session()).await;
        let (id, _) = agent.expect("session/prompt").await;
        agent.send_raw("warning: token=login-secret-value").await;
        agent.send_raw("{not json at all").await;
        agent.send_raw("").await;
        agent.update(text_chunk("agent_message_chunk", "ok")).await;
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let client_side = async {
        client.initialize().await.unwrap();
        client.new_session("/app", &[]).await.unwrap();
        client.prompt("go", &NO_LIMITS).await.unwrap()
    };
    let (outcome, ()) = tokio::join!(client_side, agent_side);
    assert_eq!(outcome.stop_reason, StopReason::EndTurn);
    assert_eq!(client.noise_lines(), 4);
    let lines: Vec<String> = seen
        .all()
        .into_iter()
        .filter_map(|e| match e {
            AcpEvent::AgentOutput {
                stream: OutputStream::Stdout,
                line,
            } => Some(line),
            _ => None,
        })
        .collect();
    assert_eq!(
        lines,
        [
            "Starting agent... (debug build)",
            "{\"level\":\"info\",\"msg\":\"boot\"}",
            "warning: token=***REDACTED***",
            "{not json at all",
        ]
    );
}

#[tokio::test]
async fn a_crash_mid_prompt_reports_the_exit_code_and_stderr() {
    let mut options = ClientOptions::new("x");
    options.redactor = Redactor::new(["login-secret-value"]);
    let (mut client, mut agent, _) = ready_client(options, session()).await;
    let agent_side = async {
        agent.expect("session/prompt").await;
        agent
            .update(text_chunk("agent_message_chunk", "Working"))
            .await;
        agent.stderr("auth with login-secret-value failed").await;
        agent
            .crash(1, "Error: the CLI process exited unexpectedly")
            .await;
    };
    let (outcome, ()) = tokio::join!(client.prompt("go", &NO_LIMITS), agent_side);
    let error = outcome.unwrap_err();
    let AcpError::Exited { code, stderr_tail } = &error else {
        panic!("{error:?}")
    };
    assert_eq!(*code, Some(1));
    assert!(stderr_tail.contains("exited unexpectedly"), "{stderr_tail}");
    assert!(stderr_tail.contains("***REDACTED***") && !stderr_tail.contains("login-secret-value"));
    assert_eq!(error.category(), "pipe_closed");
    // What the agent did before it died is kept.
    let events = client.trajectory().snapshot();
    assert!(matches!(&events[1], Event::AgentMessage(t) if t.text == "Working"));
}

#[tokio::test(start_paused = true)]
async fn an_idle_agent_is_stopped_and_asked_to_cancel() {
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let limits = PromptLimits {
        idle: Some(Duration::from_secs(60)),
        wall: Some(Duration::from_secs(3600)),
        ..PromptLimits::default()
    };
    let agent_side = async {
        agent.expect("session/prompt").await;
        agent
            .update(text_chunk("agent_thought_chunk", "Thinking"))
            .await;
        // Then silence, until BenchFlow cancels.
        let cancel = agent.recv().await;
        assert_eq!(
            cancel,
            json!({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "sess-1"}})
        );
    };
    let started = tokio::time::Instant::now();
    let (outcome, ()) = tokio::join!(client.prompt("go", &limits), agent_side);
    let error = outcome.unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("Agent idle for 60s with no new tool call, message, or thought"),
        "{error}"
    );
    assert_eq!(error.category(), "idle_timeout");
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_secs(60) && elapsed < Duration::from_secs(80),
        "{elapsed:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_pending_tool_call_earns_extra_grace() {
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let limits = PromptLimits {
        idle: Some(Duration::from_secs(60)),
        ..PromptLimits::default()
    };
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        agent
            .update(json!({"sessionUpdate": "tool_call", "toolCallId": "slow", "title": "make build", "kind": "execute", "status": "in_progress"}))
            .await;
        // Silent for 150 s: past the idle budget, inside the 180 s grace.
        tokio::time::sleep(Duration::from_secs(150)).await;
        agent
            .update(json!({"sessionUpdate": "tool_call_update", "toolCallId": "slow", "status": "completed"}))
            .await;
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let (outcome, ()) = tokio::join!(client.prompt("go", &limits), agent_side);
    assert_eq!(outcome.unwrap().stop_reason, StopReason::EndTurn);
}

#[tokio::test(start_paused = true)]
async fn the_wall_clock_budget_cancels_and_marks_the_trajectory() {
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let limits = PromptLimits {
        wall: Some(Duration::from_secs(30)),
        idle: Some(Duration::from_secs(600)),
        ..PromptLimits::default()
    };
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        agent
            .update(json!({"sessionUpdate": "tool_call", "toolCallId": "t", "title": "sleep 999", "kind": "execute", "status": "in_progress"}))
            .await;
        let cancel = agent.recv().await;
        assert_eq!(cancel["method"], "session/cancel");
        agent
            .update(json!({"sessionUpdate": "tool_call_update", "toolCallId": "t", "status": "cancelled"}))
            .await;
        agent.respond(&id, json!({"stopReason": "cancelled"})).await;
    };
    let (outcome, ()) = tokio::join!(client.prompt("go", &limits), agent_side);
    let error = outcome.unwrap_err();
    assert!(
        matches!(
            error,
            AcpError::Timeout {
                cancelled: true,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "Agent prompt exceeded wall-clock budget 30s"
    );
    let events = client.trajectory().snapshot();
    let Event::AgentTimeout(marker) = events.last().unwrap() else {
        panic!("{events:?}")
    };
    assert_eq!(marker.timeout_sec, 30.0);
    assert!(marker.pending_tool_call_ids.is_empty());
    assert!(marker.terminal_trajectory_complete);
}

/// 0.8 recorded the marker once its own prompt task stopped, so an agent
/// that ignores `session/cancel` gets it too; the open tool call makes the
/// trajectory partial.
#[tokio::test(start_paused = true)]
async fn an_agent_that_ignores_cancel_gets_a_partial_marker() {
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let limits = PromptLimits {
        wall: Some(Duration::from_secs(30)),
        ..PromptLimits::default()
    };
    let agent_side = async {
        agent.expect("session/prompt").await;
        agent
            .update(json!({"sessionUpdate": "tool_call", "toolCallId": "t", "title": "sleep 999", "kind": "execute", "status": "in_progress"}))
            .await;
        assert_eq!(agent.recv().await["method"], "session/cancel");
    };
    let started = tokio::time::Instant::now();
    let (outcome, ()) = tokio::join!(client.prompt("go", &limits), agent_side);
    assert!(matches!(
        outcome.unwrap_err(),
        AcpError::Timeout {
            cancelled: false,
            ..
        }
    ));
    // The budget plus the cancel grace, no more.
    assert!(
        started.elapsed() <= Duration::from_secs(35),
        "{:?}",
        started.elapsed()
    );
    let events = client.trajectory().snapshot();
    let Event::AgentTimeout(marker) = events.last().unwrap() else {
        panic!("{events:?}")
    };
    assert_eq!(marker.pending_tool_call_ids, ["t"]);
    assert!(!marker.terminal_trajectory_complete);
}

#[tokio::test(start_paused = true)]
async fn a_stalled_stdin_cannot_hold_the_handshake() {
    // The agent never reads, and the initialize line does not fit the pipe.
    let (pipes, _agent) = connect_with(16);
    let mut client = AcpClient::start(pipes, ClientOptions::new("x"));
    let started = tokio::time::Instant::now();
    let error = tokio::time::timeout(Duration::from_secs(3600), client.initialize())
        .await
        .expect("initialize returned")
        .unwrap_err();
    assert!(
        matches!(
            error,
            AcpError::HandshakeTimeout {
                phase: "initialize",
                ..
            }
        ),
        "{error:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(61),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(client.pending_requests(), 0);
}

#[tokio::test(start_paused = true)]
async fn a_stalled_stdin_cannot_hold_a_prompt_past_its_budget() {
    let (mut client, _agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    // Far more than the pipe holds, and the agent stops reading.
    let prompt = "x".repeat(1024 * 1024);
    let limits = PromptLimits {
        wall: Some(Duration::from_secs(30)),
        ..PromptLimits::default()
    };
    let started = tokio::time::Instant::now();
    let error = tokio::time::timeout(Duration::from_secs(3600), client.prompt(&prompt, &limits))
        .await
        .expect("the prompt returned")
        .unwrap_err();
    assert!(
        matches!(
            error,
            AcpError::Timeout {
                cancelled: false,
                ..
            }
        ),
        "{error:?}"
    );
    assert!(
        started.elapsed() <= Duration::from_secs(35),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(client.pending_requests(), 0);
}

#[tokio::test(start_paused = true)]
async fn a_process_that_exited_before_the_prompt_ends_it() {
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    agent.stderr("fatal: out of memory").await;
    // The process is gone; a background child keeps stdout open.
    agent.exit.send_replace(Some(137));
    tokio::time::sleep(Duration::from_secs(1)).await;
    let started = tokio::time::Instant::now();
    let error = tokio::time::timeout(Duration::from_secs(3600), client.prompt("go", &NO_LIMITS))
        .await
        .expect("the prompt returned")
        .unwrap_err();
    let AcpError::Exited { code, stderr_tail } = &error else {
        panic!("{error:?}")
    };
    assert_eq!(*code, Some(137));
    assert!(stderr_tail.contains("out of memory"));
    assert!(
        started.elapsed() <= Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
}

/// Stdin that refuses every write, as Daytona does once the agent's
/// process is gone (its input pipe no longer exists).
struct Refusing;

impl tokio::io::AsyncWrite for Refusing {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        _buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::task::Poll::Ready(Err(std::io::Error::other(
            "Daytona failed while writing to a command's stdin: failed to open input pipe",
        )))
    }
    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }
    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }
}

/// An agent that quits at once: the first write fails a moment before its
/// exit shows. It is reported as having quit (R202), with its last words,
/// not as a broken connection to the sandbox.
#[tokio::test]
async fn a_write_to_an_agent_that_just_quit_reports_its_exit() {
    let (agent_stdout, client_stdout) = tokio::io::duplex(1024);
    let (mut agent_stderr, client_stderr) = tokio::io::duplex(1024);
    let exit = watch::Sender::new(None);
    let pipes = AgentPipes {
        stdin: Box::pin(Refusing),
        stdout: Box::pin(client_stdout),
        stderr: Some(Box::pin(client_stderr)),
        handle: Some(Box::new(FakeHandle {
            exit: exit.clone(),
            killed: Arc::new(AtomicBool::new(false)),
        })),
    };
    let mut client = AcpClient::start(pipes, ClientOptions::new("quits"));
    let quit = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        agent_stderr
            .write_all(b"fatal: cannot read /etc/quits.conf\n")
            .await
            .unwrap();
        drop(agent_stderr);
        exit.send_replace(Some(3));
        drop(agent_stdout);
    });
    let error = client.initialize().await.unwrap_err();
    quit.await.unwrap();
    assert!(
        matches!(error, AcpError::Exited { code: Some(3), .. }),
        "{error:?}"
    );
    assert!(
        error
            .to_string()
            .contains("fatal: cannot read /etc/quits.conf")
    );
}

#[tokio::test(start_paused = true)]
async fn a_crash_after_the_handshake_fails_the_next_request_as_exited() {
    let (mut client, mut agent, _) =
        ready_client(ClientOptions::new("claude"), claude_session()).await;
    agent.crash(1, "Error: invalid API key").await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    let started = tokio::time::Instant::now();
    let error = client
        .select_model(&spec("claude"), Some("claude-opus-5-5"), None)
        .await
        .unwrap_err();
    let AcpError::Exited { code, stderr_tail } = &error else {
        panic!("{error:?}")
    };
    assert_eq!(*code, Some(1));
    assert!(stderr_tail.contains("invalid API key"));
    // At once, not after the handshake timeout.
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn the_agents_words_are_redacted_in_errors_and_events() {
    const SECRET: &str = "login-secret-value-0042";
    let seen = Seen::default();
    let mut options = seen.options("claude");
    options.redactor = Redactor::new([SECRET]);
    let (mut client, mut agent, _) = ready_client(options, session()).await;
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        agent
            .update(json!({"sessionUpdate": "tool_call", "toolCallId": "t", "title": format!("curl -H 'token: {SECRET}' https://example.test"), "kind": "execute", "status": "completed"}))
            .await;
        agent
            .update(json!({"sessionUpdate": "notice", "severity": "warning", "title": format!("token {SECRET} expires soon")}))
            .await;
        agent
            .fail(
                &id,
                json!({"code": -32603, "message": format!("Internal error: key {SECRET} rejected"), "data": {"details": SECRET}}),
            )
            .await;
        let (id, _) = agent.expect("session/prompt").await;
        agent.respond(&id, json!({"note": SECRET})).await;
    };
    let client_side = async {
        let first = client.prompt("go", &NO_LIMITS).await.unwrap_err();
        let second = client.prompt("again", &NO_LIMITS).await.unwrap_err();
        (first, second)
    };
    let ((first, second), ()) = tokio::join!(client_side, agent_side);
    assert!(matches!(first, AcpError::Agent { .. }), "{first:?}");
    assert!(matches!(second, AcpError::Protocol { .. }), "{second:?}");
    for error in [&first, &second] {
        let shown = format!("{error} {error:?}");
        assert!(!shown.contains(SECRET), "{shown}");
        assert!(shown.contains("***REDACTED***"), "{shown}");
    }
    let events = seen.all();
    assert!(!format!("{events:?}").contains(SECRET));
    assert!(events.iter().any(|e| matches!(
        e,
        AcpEvent::Notice { title, .. } if title == "token ***REDACTED*** expires soon"
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AcpEvent::Progress(p) if p.last_tool_title.as_deref().is_some_and(|t| t.contains("***REDACTED***"))
    )));
}

#[tokio::test(start_paused = true)]
async fn shutdown_is_bounded_when_the_kill_hangs() {
    let (mut pipes, _agent) = connect();
    pipes.handle = Some(Box::new(HangingHandle));
    let client = AcpClient::start(pipes, ClientOptions::new("x"));
    let started = tokio::time::Instant::now();
    let code = tokio::time::timeout(Duration::from_secs(3600), client.shutdown())
        .await
        .expect("shutdown returned");
    assert_eq!(code, None);
    assert!(
        started.elapsed() <= Duration::from_secs(16),
        "{:?}",
        started.elapsed()
    );
}

/// A caller may drop `new_session` or `prompt` (its own timeout, a
/// cancelled trial): the client forgets the request and stays usable.
#[tokio::test(start_paused = true)]
async fn a_dropped_handshake_or_prompt_leaves_the_client_usable() {
    let (pipes, mut agent) = connect();
    let mut client = AcpClient::start(pipes, ClientOptions::new("x"));
    let agent_side = async {
        let (id, _) = agent.expect("initialize").await;
        agent.respond(&id, json!({"protocolVersion": 1})).await;
    };
    let (info, ()) = tokio::join!(client.initialize(), agent_side);
    info.unwrap();

    let (gave_up, _) = tokio::join!(
        tokio::time::timeout(Duration::from_secs(1), client.new_session("/app", &[])),
        agent.expect("session/new")
    );
    assert!(gave_up.is_err());
    assert!(!client.opening_buffer_on());
    assert_eq!(client.pending_requests(), 0);
    let agent_side = async {
        let (id, _) = agent.expect("session/new").await;
        agent.respond(&id, session()).await;
    };
    let (opened, ()) = tokio::join!(client.new_session("/app", &[]), agent_side);
    opened.unwrap();

    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        agent
            .update(text_chunk("agent_message_chunk", "Half"))
            .await;
        id
    };
    let (gave_up, first) = tokio::join!(
        tokio::time::timeout(Duration::from_secs(5), client.prompt("go", &NO_LIMITS)),
        agent_side
    );
    assert!(gave_up.is_err());
    assert_eq!(client.pending_requests(), 0);

    let agent_side = async {
        // The dropped prompt's late answer goes nowhere.
        agent
            .respond(&first, json!({"stopReason": "cancelled"}))
            .await;
        let (id, _) = agent.expect("session/prompt").await;
        agent
            .update(text_chunk("agent_message_chunk", "Done"))
            .await;
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let (outcome, ()) = tokio::join!(client.prompt("again", &NO_LIMITS), agent_side);
    assert_eq!(outcome.unwrap().stop_reason, StopReason::EndTurn);
    let texts: Vec<String> = client
        .trajectory()
        .snapshot()
        .into_iter()
        .filter_map(|e| match e {
            Event::AgentMessage(t) | Event::UserMessage(t) => Some(t.text),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["go", "Half", "again", "Done"]);
}

/// claude-agent-acp 0.81.2 settling a failed turn when typed failures are on
/// (the shape 0.8's test_acp_typed_failures.py builds).
fn failed_turn(category: &str, actions: Value, title: &str, severity: &str) -> Value {
    json!({
        "stopReason": "end_turn",
        "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
        "_meta": {"jetbrains": {"air": {"version": 1, "sessionFailure": {
            "id": "prompt-1:error", "revision": 1, "category": category, "severity": severity,
            "title": title, "actions": actions,
        }}}},
    })
}

async fn prompt_answered(
    answer: Result<Value, Value>,
    options: ClientOptions,
) -> Result<PromptOutcome, AcpError> {
    let (mut client, mut agent, _) = ready_client(options, session()).await;
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        match answer {
            Ok(result) => agent.respond(&id, result).await,
            Err(error) => agent.fail(&id, error).await,
        }
    };
    let (outcome, ()) = tokio::join!(client.prompt("go", &NO_LIMITS), agent_side);
    outcome
}

#[tokio::test]
async fn a_spent_subscription_is_a_usage_limit() {
    let mut options = ClientOptions::new("claude");
    options.login = Some("CLAUDE_CODE_OAUTH_TOKEN (environment)".into());
    let weekly = "You've hit your weekly limit \u{b7} resets Oct 2, 4pm (UTC)";
    let error = prompt_answered(
        Ok(failed_turn("limit", json!([]), weekly, "error")),
        options.clone(),
    )
    .await
    .unwrap_err();
    let AcpError::UsageLimit {
        window,
        reset_at,
        login,
        ..
    } = &error
    else {
        panic!("{error:?}")
    };
    assert_eq!(window.as_deref(), Some("7-day"));
    assert!(reset_at.is_some());
    assert_eq!(
        login.as_deref(),
        Some("CLAUDE_CODE_OAUTH_TOKEN (environment)")
    );
    assert!(error.to_string().starts_with(
        "usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (environment): 7-day window, resets"
    ));

    // A spent login marked as a warning still ends the turn.
    let error = prompt_answered(
        Ok(failed_turn(
            "limit",
            json!([]),
            "The Claude account has no available quota.",
            "warning",
        )),
        options.clone(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, AcpError::UsageLimit { .. }), "{error:?}");

    // A rate limit that offers a retry, marked as a warning, is only advice.
    let outcome = prompt_answered(
        Ok(failed_turn(
            "limit",
            json!(["retry"]),
            "Slow down.",
            "warning",
        )),
        options.clone(),
    )
    .await
    .unwrap();
    assert_eq!(outcome.stop_reason, StopReason::EndTurn);

    // Other failures read as the -32603 the agent sent before typed failures.
    let error = prompt_answered(
        Ok(failed_turn(
            "connection",
            json!(["new_session"]),
            "The connection to Claude was lost.",
            "error",
        )),
        options.clone(),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "ACP error -32603: Internal error: The connection to Claude was lost: the agent's own process ended or \
         lost its stream (not BenchFlow's connection to the sandbox); the agent log in the trial's agent/ folder says which"
    );

    // codex-acp's JSON-RPC answer for a spent ChatGPT plan.
    let error = prompt_answered(
        Err(json!({"code": -32603, "message": "Internal error", "data": {
            "message": "You\u{2019}ve hit your usage limit. Upgrade to Plus to continue using Codex \
                        (https://chatgpt.com/explore/plus), or try again at Oct 3rd, 2026 5:05 PM.",
            "codexErrorInfo": "usageLimitExceeded",
        }})),
        ClientOptions::new("codex"),
    )
    .await
    .unwrap_err();
    let AcpError::UsageLimit {
        reset_at, message, ..
    } = &error
    else {
        panic!("{error:?}")
    };
    assert_eq!(reset_at.unwrap().to_rfc3339(), "2026-10-03T17:05:00+00:00");
    assert!(message.starts_with("You've hit your usage limit."));
}

#[tokio::test]
async fn agent_errors_carry_code_message_data_and_category() {
    let error = prompt_answered(
        Err(json!({"code": -32603, "message": "Internal error", "data": {"details": "API Error: 401 {\"type\":\"authentication_error\"} Invalid bearer token"}})),
        ClientOptions::new("claude"),
    )
    .await
    .unwrap_err();
    let AcpError::Agent {
        code,
        message,
        data,
        method,
    } = &error
    else {
        panic!("{error:?}")
    };
    assert_eq!(
        (*code, message.as_str(), method.as_str()),
        (-32603, "Internal error", "session/prompt")
    );
    assert!(data.is_some());
    assert_eq!(error.category(), "provider_auth");
    assert_eq!(error.owner(), Owner::Login);

    let error = prompt_answered(
        Err(json!({"code": -32603, "message": "Internal error: API Error: Request rejected (429) \u{b7} rate limit exceeded"})),
        ClientOptions::new("claude"),
    )
    .await
    .unwrap_err();
    assert_eq!(error.category(), "provider_rate_limit");
    assert_eq!(error.owner(), Owner::ModelProvider);
}

#[tokio::test(start_paused = true)]
async fn a_silent_agent_times_out_the_handshake() {
    let (pipes, mut agent) = connect();
    let mut client = AcpClient::start(pipes, ClientOptions::new("x"));
    let agent_side = async {
        agent.expect("initialize").await;
    };
    let (result, ()) = tokio::join!(client.initialize(), agent_side);
    let error = result.unwrap_err();
    assert_eq!(
        error.to_string(),
        "ACP initialize timed out after 60s before the first prompt"
    );
    assert_eq!(error.category(), "timeout");
}

fn claude_session() -> Value {
    json!({
        "sessionId": "sess-1",
        "configOptions": [
            {"id": "mode", "name": "Mode", "type": "select", "currentValue": "default", "options": [{"value": "default", "name": "Default"}]},
            {"id": "model", "name": "Model", "type": "select", "currentValue": "default", "options": [
                {"value": "default", "name": "Default"},
                {"value": "claude-opus-5-5", "name": "Opus"},
                {"value": "claude-sonnet-4-6", "name": "Sonnet"},
            ]},
            {"id": "effort", "name": "Effort", "type": "select", "currentValue": "high", "options": [
                {"value": "low"}, {"value": "medium"}, {"value": "high"}, {"value": "max"},
            ]},
        ],
    })
}

fn spec(agent: &str) -> crate::manifest::ModelSpec {
    Registry::builtin().find(agent).unwrap().model.clone()
}

#[tokio::test]
async fn claude_takes_model_and_effort_as_config_options() {
    let (mut client, mut agent, _) =
        ready_client(ClientOptions::new("claude"), claude_session()).await;
    let agent_side = async {
        let (id, params) = agent.expect("session/set_config_option").await;
        assert_eq!(
            params,
            json!({"sessionId": "sess-1", "configId": "model", "value": "claude-opus-5-5"})
        );
        agent
            .respond(
                &id,
                json!({"configOptions": claude_session()["configOptions"]}),
            )
            .await;
        let (id, params) = agent.expect("session/set_config_option").await;
        assert_eq!(
            params,
            json!({"sessionId": "sess-1", "configId": "effort", "value": "max"})
        );
        agent.respond(&id, json!({})).await;
    };
    let claude_spec = spec("claude");
    let (choice, ()) = tokio::join!(
        client.select_model(&claude_spec, Some("claude-opus-5-5"), Some("max")),
        agent_side
    );
    let choice = choice.unwrap();
    assert_eq!(
        choice.model_via.as_deref(),
        Some("session/set_config_option:model")
    );
    assert_eq!(
        choice.effort_via.as_deref(),
        Some("session/set_config_option:effort")
    );
}

#[tokio::test]
async fn a_model_the_agent_refuses_and_does_not_list_is_not_offered() {
    let (mut client, mut agent, _) =
        ready_client(ClientOptions::new("claude"), claude_session()).await;
    let agent_side = async {
        let (id, _) = agent.expect("session/set_config_option").await;
        agent
            .fail(&id, json!({"code": -32603, "message": "Internal error", "data": {"details": "Invalid value for config option model: claude-nope"}}))
            .await;
    };
    let claude_spec = spec("claude");
    let (result, ()) = tokio::join!(
        client.select_model(&claude_spec, Some("claude-nope"), None),
        agent_side
    );
    let error = result.unwrap_err();
    assert_eq!(
        error.to_string(),
        "agent integration failure [agent_model]: claude does not offer model 'claude-nope' for this login; \
         it offers: default, claude-opus-5-5, claude-sonnet-4-6"
    );
    assert_eq!(error.category(), "agent_integration");
}

#[tokio::test]
async fn codex_gets_model_effort_ids_through_set_model() {
    let codex_session = json!({
        "sessionId": "sess-1",
        "models": {"currentModelId": "gpt-5.6-sol[medium]", "availableModels": [
            {"modelId": "gpt-6.1-sol[low]", "name": "x"},
            {"modelId": "gpt-6.1-sol[medium]", "name": "x"},
            {"modelId": "gpt-6.1-sol[high]", "name": "x"},
            {"modelId": "gpt-5.6-sol[medium]", "name": "x"},
        ]},
        "configOptions": [{"id": "model", "type": "select", "currentValue": "gpt-5.6-sol", "options": []}],
    });
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("codex"), codex_session).await;
    let agent_side = async {
        let (id, params) = agent.expect("session/set_model").await;
        assert_eq!(
            params,
            json!({"sessionId": "sess-1", "modelId": "gpt-6.1-sol[high]"})
        );
        agent.respond(&id, json!({})).await;
    };
    let codex_spec = spec("codex");
    let (choice, ()) = tokio::join!(
        client.select_model(&codex_spec, Some("openai/gpt-6.1-sol"), Some("high")),
        agent_side
    );
    let choice = choice.unwrap();
    assert_eq!(choice.model_id.as_deref(), Some("gpt-6.1-sol[high]"));
    assert_eq!(
        choice.effort_via.as_deref(),
        Some("model id gpt-6.1-sol[high]")
    );

    let error = client
        .select_model(&spec("codex"), Some("gpt-4o"), None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, AcpError::ModelNotOffered { .. }),
        "{error:?}"
    );
}

/// `-a codex -e high` with no `-m`: the session's own model at that
/// effort, from the variants it offers (on 2026-10-10 every trial failed
/// after it started, saying Codex offers no effort option). An effort the
/// session does not offer for its model is refused, naming the ones it does.
#[tokio::test]
async fn codex_effort_without_a_model_picks_the_sessions_variant() {
    let codex_session = |available: &[&str]| {
        json!({
            "sessionId": "sess-1",
            "models": {
                "currentModelId": "gpt-6.1-sol[low]",
                "availableModels": available.iter().map(|id| json!({"modelId": id, "name": "x"})).collect::<Vec<_>>(),
            },
            "configOptions": [{"id": "model", "type": "select", "currentValue": "gpt-6.1-sol", "options": []}],
        })
    };
    let session = codex_session(&[
        "gpt-6.1-sol[low]",
        "gpt-6.1-sol[medium]",
        "gpt-6.1-sol[high]",
    ]);
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("codex"), session).await;
    let agent_side = async {
        let (id, params) = agent.expect("session/set_model").await;
        assert_eq!(
            params,
            json!({"sessionId": "sess-1", "modelId": "gpt-6.1-sol[high]"})
        );
        agent.respond(&id, json!({})).await;
    };
    let codex_spec = spec("codex");
    let (choice, ()) = tokio::join!(
        client.select_model(&codex_spec, None, Some("high")),
        agent_side
    );
    let choice = choice.unwrap();
    assert_eq!(choice.model_id.as_deref(), Some("gpt-6.1-sol[high]"));
    assert_eq!(
        choice.effort_via.as_deref(),
        Some("model id gpt-6.1-sol[high]")
    );

    let session = codex_session(&["gpt-6.1-sol[low]", "gpt-6.1-sol[medium]"]);
    let (mut client, _agent, _) = ready_client(ClientOptions::new("codex"), session).await;
    let error = client
        .select_model(&spec("codex"), None, Some("xhigh"))
        .await
        .unwrap_err();
    let text = error.to_string();
    assert!(
        text.contains("offers gpt-6.1-sol only at low, medium"),
        "{text}"
    );
}

#[tokio::test]
async fn auto_selection_follows_what_the_session_offers() {
    // No model option: session/set_model.
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("gemini"), session()).await;
    let agent_side = async {
        let (id, params) = agent.expect("session/set_model").await;
        assert_eq!(params["modelId"], "gemini-2.5-pro");
        agent.respond(&id, json!({})).await;
    };
    let gemini_spec = spec("gemini");
    let (choice, ()) = tokio::join!(
        client.select_model(&gemini_spec, Some("google/gemini-2.5-pro"), None),
        agent_side
    );
    assert_eq!(
        choice.unwrap().model_via.as_deref(),
        Some("session/set_model")
    );
    // An effort with no option to carry it is refused.
    let error = client
        .select_model(&spec("gemini"), None, Some("high"))
        .await
        .unwrap_err();
    assert!(matches!(error, AcpError::Unsupported { .. }), "{error:?}");
    assert_eq!(error.owner(), Owner::User);

    // A model option: set through it.
    let with_option = json!({"sessionId": "sess-1", "configOptions": [{"id": "model", "type": "select", "currentValue": "a", "options": []}]});
    let (mut client, mut agent, _) =
        ready_client(ClientOptions::new("opencode"), with_option).await;
    let agent_side = async {
        let (id, params) = agent.expect("session/set_config_option").await;
        assert_eq!(params["value"], "openai/gpt-5");
        agent.respond(&id, json!({})).await;
    };
    let opencode_spec = spec("opencode");
    let (choice, ()) = tokio::join!(
        client.select_model(&opencode_spec, Some("gpt-5"), None),
        agent_side
    );
    assert_eq!(
        choice.unwrap().model_via.as_deref(),
        Some("session/set_config_option:model")
    );
}

#[tokio::test]
async fn a_line_of_many_megabytes_still_closes_its_tool_call() {
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("claude"), session()).await;
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        agent
            .update(json!({"sessionUpdate": "tool_call", "toolCallId": "read_pdf", "title": "Read report.pdf", "kind": "read", "status": "in_progress"}))
            .await;
        let page = "QUJD".repeat(1_500_000); // 6 MB of base64
        agent
            .update(json!({
                "sessionUpdate": "tool_call_update", "toolCallId": "read_pdf", "status": "completed",
                "content": [{"type": "content", "content": {"type": "image", "data": page, "mimeType": "image/png"}}],
            }))
            .await;
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let (outcome, ()) = tokio::join!(client.prompt("go", &NO_LIMITS), agent_side);
    outcome.unwrap();
    let events = client.trajectory().snapshot();
    let Event::ToolCall(call) = &events[1] else {
        panic!()
    };
    assert_eq!(call.status, ToolCallStatus::Completed);
    assert_eq!(
        call.content[0]["content"]["text"],
        "[BenchFlow removed image data (image/png) (6000000 bytes): the ACP line was over the 1048576-byte limit]"
    );
}

#[tokio::test]
async fn updates_before_session_new_answers_belong_to_the_session() {
    let (pipes, mut agent) = connect();
    let mut client = AcpClient::start(pipes, ClientOptions::new("x"));
    let agent_side = async {
        let (id, _) = agent.expect("initialize").await;
        agent.respond(&id, json!({"protocolVersion": 1})).await;
        let (id, _) = agent.expect("session/new").await;
        // A replayed history arrives before the answer.
        agent
            .update(text_chunk("agent_message_chunk", "Earlier answer"))
            .await;
        agent.respond(&id, session()).await;
        let (id, _) = agent.expect("session/prompt").await;
        // An update for another session is not this one's.
        agent
            .send(json!({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "other", "update": text_chunk("agent_message_chunk", "Not mine")}}))
            .await;
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let client_side = async {
        client.initialize().await.unwrap();
        client.new_session("/app", &[]).await.unwrap();
        client.prompt("go", &NO_LIMITS).await.unwrap();
    };
    tokio::join!(client_side, agent_side);
    let texts: Vec<String> = client
        .trajectory()
        .snapshot()
        .into_iter()
        .filter_map(|e| match e {
            Event::AgentMessage(t) | Event::UserMessage(t) => Some(t.text),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["Earlier answer", "go"]);
}

#[tokio::test]
async fn a_typed_advisory_becomes_a_notice_not_the_agents_words() {
    let seen = Seen::default();
    let (mut client, mut agent, _) = ready_client(seen.options("claude"), session()).await;
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        agent
            .update(json!({"sessionUpdate": "session_info_update", "_meta": {"jetbrains": {"air": {"version": 1, "sessionFailure": {
                "id": "n", "revision": 1, "category": "unknown", "severity": "warning",
                "title": "claude-opus-5-5 declined this request; retried with claude-sonnet-4-6.", "actions": [],
            }}}}}))
            .await;
        agent
            .update(json!({"sessionUpdate": "session_info_update", "_meta": {"jetbrains": {"air": {"version": 1, "sessionFailure": {
                "id": "r", "revision": 1, "category": "service", "severity": "warning",
                "title": "Retrying Claude, attempt 2 of 10.", "actions": [],
            }}}}}))
            .await;
        agent.update(json!({"sessionUpdate": "notice", "severity": "warning", "title": "Model metadata for `gpt-5.4` not found."})).await;
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let (outcome, ()) = tokio::join!(client.prompt("go", &NO_LIMITS), agent_side);
    outcome.unwrap();
    let notices: Vec<(String, String)> = client
        .trajectory()
        .snapshot()
        .into_iter()
        .filter_map(|e| match e {
            Event::Notice(n) => Some((n.severity, n.title)),
            _ => None,
        })
        .collect();
    assert_eq!(
        notices,
        [
            (
                "warning".into(),
                "claude-opus-5-5 declined this request; retried with claude-sonnet-4-6.".into()
            ),
            (
                "warning".into(),
                "Model metadata for `gpt-5.4` not found.".into()
            ),
        ]
    );
    assert_eq!(client.trajectory().with(|t| t.final_text()), "");
    assert!(seen.all().iter().any(
        |e| matches!(e, AcpEvent::Notice { title, .. } if title.starts_with("Model metadata"))
    ));
}

#[tokio::test(start_paused = true)]
async fn a_process_that_exits_while_stdout_stays_open_ends_the_prompt() {
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let agent_side = async {
        agent.expect("session/prompt").await;
        agent.stderr("fatal: out of memory").await;
        // A background child keeps stdout open; the process itself is gone.
        agent.exit.send_replace(Some(137));
        tokio::time::sleep(Duration::from_secs(30)).await;
        agent
    };
    let (outcome, _agent) = tokio::join!(client.prompt("go", &NO_LIMITS), agent_side);
    let error = outcome.unwrap_err();
    let AcpError::Exited { code, stderr_tail } = &error else {
        panic!("{error:?}")
    };
    assert_eq!(*code, Some(137));
    assert!(stderr_tail.contains("out of memory"));
}

#[tokio::test(start_paused = true)]
async fn shutdown_closes_stdin_then_kills_a_process_that_stays() {
    let (client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let killed = agent.killed.clone();
    let agent_side = async {
        // stdin closes: the next read ends.
        assert!(agent.from_client.next_line().await.unwrap().is_none());
    };
    let (code, ()) = tokio::join!(client.shutdown(), agent_side);
    assert_eq!(code, Some(137));
    assert!(killed.load(Ordering::SeqCst));

    // An agent that exits on its own is not killed.
    let (client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let killed = agent.killed.clone();
    let agent_side = async {
        assert!(agent.from_client.next_line().await.unwrap().is_none());
        agent.exit.send_replace(Some(0));
    };
    let (code, ()) = tokio::join!(client.shutdown(), agent_side);
    assert_eq!(code, Some(0));
    assert!(!killed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn mcp_servers_go_in_session_new() {
    let (pipes, mut agent) = connect();
    let mut client = AcpClient::start(pipes, ClientOptions::new("x"));
    let server = McpServer::from_task_table(
        &json!({"name": "web", "transport": "http", "url": "http://localhost:3000/mcp"}),
    )
    .unwrap();
    let agent_side = async {
        let (id, _) = agent.expect("initialize").await;
        agent.respond(&id, json!({"protocolVersion": 1})).await;
        let (id, params) = agent.expect("session/new").await;
        assert_eq!(
            params,
            json!({"cwd": "/work", "mcpServers": [{"type": "http", "name": "web", "url": "http://localhost:3000/mcp", "headers": []}]})
        );
        // No session id: the session runs under a placeholder.
        agent.respond(&id, json!({})).await;
    };
    let client_side = async {
        client.initialize().await.unwrap();
        client.new_session("/work", &[server]).await.unwrap()
    };
    let (info, ()) = tokio::join!(client_side, agent_side);
    assert_eq!(info.session_id, "default");
}

/// The 0.8 session fixture's updates, sent over the wire by a fake agent,
/// end up as the same events 0.8 recorded (receipt times aside).
#[tokio::test]
async fn the_wire_path_records_what_0_8_recorded() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../testdata/session-0.8.json")).unwrap();
    let steps = fixture["steps"].as_array().unwrap().clone();
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("claude"), session()).await;
    // Only the first prompt's updates; the second prompt ends in a timeout marker.
    let first: Vec<Value> = steps
        .iter()
        .skip(1)
        .take_while(|s| s.get("update").is_some())
        .map(|s| s["update"].clone())
        .collect();
    let prompt = steps[0]["prompt"].as_str().unwrap().to_string();
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        for update in first {
            agent.update(update).await;
        }
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let (outcome, ()) = tokio::join!(client.prompt(&prompt, &NO_LIMITS), agent_side);
    outcome.unwrap();

    let strip = |text: &str| -> Vec<Value> {
        text.lines()
            .map(|line| {
                let mut v: Value = serde_json::from_str(line).unwrap();
                for key in ["receipt", "ts", "started_at", "finished_at"] {
                    v.as_object_mut().unwrap().remove(key);
                }
                v
            })
            .collect()
    };
    let ours = to_jsonl(&client.trajectory().snapshot(), &Redactor::default());
    let theirs: Vec<Value> = strip(fixture["jsonl"].as_str().unwrap());
    // 0.8's events up to its second prompt.
    let cut = theirs
        .iter()
        .skip(1)
        .position(|e| e["type"] == "user_message")
        .unwrap()
        + 1;
    assert_eq!(strip(&ours), theirs[..cut]);
}

/// How the runner writes acp_trajectory.jsonl while a prompt runs: a task
/// snapshots the trajectory through its handle and rewrites the file.
#[tokio::test]
async fn the_trajectory_can_be_written_while_a_prompt_runs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trajectory/acp_trajectory.jsonl");
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("x"), session()).await;
    let handle = client.trajectory();
    let (stop, mut stopped) = watch::channel(false);
    let writer_path = path.clone();
    let writer = tokio::spawn(async move {
        let mut writer = crate::trajectory::TrajectoryWriter::new(writer_path, Redactor::default())
            .await
            .unwrap();
        let mut writes = 0;
        loop {
            if writer.write(&handle.snapshot()).await.unwrap() {
                writes += 1;
            }
            tokio::select! {
                _ = stopped.changed() => break,
                _ = tokio::time::sleep(Duration::from_millis(2)) => {}
            }
        }
        writer.write(&handle.snapshot()).await.unwrap();
        writes
    });
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        for i in 0..200 {
            agent
                .update(json!({"sessionUpdate": "tool_call", "toolCallId": format!("t{i}"), "title": format!("step {i}"), "kind": "execute", "status": "completed"}))
                .await;
            agent.update(text_chunk("agent_message_chunk", "ok ")).await;
            if i % 20 == 0 {
                tokio::task::yield_now().await;
            }
        }
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
    };
    let (outcome, ()) = tokio::join!(client.prompt("go", &NO_LIMITS), agent_side);
    outcome.unwrap();
    stop.send_replace(true);
    let writes = writer.await.unwrap();
    assert!(writes >= 1);
    let written = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        written,
        to_jsonl(&client.trajectory().snapshot(), &Redactor::default())
    );
    assert_eq!(client.trajectory().with(|t| t.tool_call_count()), 200);
}

/// An agent started on a provider's model lists it as current; BenchFlow does
/// not select it again (Claude Code would spend a request validating it, and
/// codex-acp refuses even the current id when its catalog lacks it).
#[tokio::test]
async fn a_model_and_effort_the_session_already_has_are_not_sent_again() {
    let codex_session = json!({
        "sessionId": "sess-1",
        "models": {"currentModelId": "deepseek/deepseek-v4-flash-0731[high]", "availableModels": [
            {"modelId": "gpt-6.1-sol[medium]", "name": "x"},
        ]},
    });
    let (mut client, mut agent, _) = ready_client(ClientOptions::new("codex"), codex_session).await;
    let choice = client
        .select_model(
            &spec("codex"),
            Some("deepseek/deepseek-v4-flash-0731"),
            Some("high"),
        )
        .await
        .unwrap();
    assert_eq!(
        choice.model_id.as_deref(),
        Some("deepseek/deepseek-v4-flash-0731[high]")
    );
    assert_eq!(choice.model_via.as_deref(), Some("current session model"));

    let mut session = claude_session();
    session["configOptions"][1]["currentValue"] = json!("claude-opus-5-5");
    session["configOptions"][2]["currentValue"] = json!("max");
    let (mut claude, mut claude_agent, _) =
        ready_client(ClientOptions::new("claude"), session).await;
    let choice = claude
        .select_model(&spec("claude"), Some("claude-opus-5-5"), Some("max"))
        .await
        .unwrap();
    assert_eq!(choice.model_via.as_deref(), Some("current value of model"));
    assert_eq!(
        choice.effort_via.as_deref(),
        Some("current value of effort")
    );

    // Neither agent was asked anything: the next message each gets is the prompt.
    let agent_side = async {
        let (id, _) = agent.expect("session/prompt").await;
        agent.respond(&id, json!({"stopReason": "end_turn"})).await;
        let (id, _) = claude_agent.expect("session/prompt").await;
        claude_agent
            .respond(&id, json!({"stopReason": "end_turn"}))
            .await;
    };
    let client_side = async {
        client.prompt("go", &NO_LIMITS).await.unwrap();
        claude.prompt("go", &NO_LIMITS).await.unwrap();
    };
    tokio::join!(agent_side, client_side);
}
