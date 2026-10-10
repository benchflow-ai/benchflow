//! The connection to one agent process.
//!
//! Four tasks serve it, so the client never blocks on the agent and the
//! agent never blocks on the client:
//!
//! - the **reader** reads stdout, answers the agent's requests (permission
//!   prompts) at once, records `session/update`s into the trajectory, routes
//!   responses to the requests waiting for them, and passes anything that is
//!   not JSON-RPC to the event sink as agent output;
//! - the **writer** owns stdin and writes queued lines in order (an unbounded
//!   queue: the reader must never wait on stdin while the agent waits on
//!   stdout);
//! - the **stderr reader** keeps the tail of stderr for error messages;
//! - the **exit watcher** owns the process handle, publishes the exit code,
//!   and kills the process when asked.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};

use benchflow_sandbox::ProcessHandle;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, watch};

use super::limits;
use super::wire::{self, Line, LineReader};
use super::{AcpEvent, OutputStream, Progress};
use crate::trajectory::{Change, Receipt, Redactor, Trajectory};

/// Called with every event, from the connection's tasks. Keep it quick.
pub type EventSink = Arc<dyn Fn(AcpEvent) + Send + Sync>;

/// A JSON-RPC error answer.
#[derive(Debug, Clone)]
pub(crate) struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

/// A JSON-RPC answer, and when the reader routed it.
#[derive(Debug)]
pub(crate) struct Answer {
    /// The result, or the agent's error.
    pub result: Result<Value, RpcError>,
    /// When the answer arrived: a turn that finished inside its budget is
    /// not late because the client looked late (0.8's `_timed_prompt`
    /// records completion the same way).
    pub at: tokio::time::Instant,
}

/// What a waiting request gets: the answer, or `Err` when the connection
/// closed first.
pub(crate) type Reply = Result<Answer, Closed>;

/// The connection closed before the answer came.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Closed;

/// The process's state as the exit watcher last saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    Running,
    /// Exited with this code (`None` when the backend could not say).
    Exited(Option<i32>),
}

/// A line for the writer.
pub(crate) enum Outgoing {
    Line {
        line: String,
        done: Option<oneshot::Sender<std::io::Result<()>>>,
    },
    /// Close stdin.
    Close,
}

/// Stderr's last bytes.
#[derive(Default)]
pub(crate) struct Tail {
    text: String,
    pub done: bool,
}

/// How much stderr is kept (0.8's `_STDERR_TAIL_LIMIT`).
const STDERR_KEEP: usize = 64 * 1024;

impl Tail {
    fn push(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
        if self.text.len() > STDERR_KEEP {
            let mut cut = self.text.len() - STDERR_KEEP;
            while !self.text.is_char_boundary(cut) {
                cut += 1;
            }
            self.text.drain(..cut);
        }
    }

    /// The last `chars` characters.
    pub(crate) fn last(&self, chars: usize) -> String {
        let count = self.text.chars().count();
        self.text
            .chars()
            .skip(count.saturating_sub(chars))
            .collect()
    }
}

/// The session as the reader sees it.
#[derive(Default)]
pub(crate) struct State {
    pub pending: HashMap<i64, oneshot::Sender<Reply>>,
    pub trajectory: Trajectory,
    /// Set once `session/new` answers.
    pub session_id: Option<String>,
    /// The agent sent no session id; every update counts as the session's.
    pub session_placeholder: bool,
    /// Updates that arrive while `session/new` is in flight, replayed once
    /// it answers (they may belong to the session it opens).
    pub opening: Option<Vec<(Value, Receipt)>>,
    pub noise_lines: u64,
    /// Stdout ended.
    pub closed: bool,
}

/// What the tasks and the client share.
pub(crate) struct Shared {
    state: Mutex<State>,
    pub stderr: Mutex<Tail>,
    pub exit: watch::Sender<Exit>,
    pub redactor: Redactor,
    pub sink: Option<EventSink>,
}

impl Shared {
    pub(crate) fn new(redactor: Redactor, sink: Option<EventSink>) -> Self {
        Self {
            state: Mutex::new(State::default()),
            stderr: Mutex::new(Tail::default()),
            exit: watch::Sender::new(Exit::Running),
            redactor,
            sink,
        }
    }

    /// The session state. Never held across an `.await`.
    pub(crate) fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn stderr(&self) -> MutexGuard<'_, Tail> {
        self.stderr
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn emit(&self, event: AcpEvent) {
        if let Some(sink) = &self.sink {
            sink(event);
        }
    }

    /// Counts for a progress line. The tool title is the agent's words, so
    /// it is redacted (a command line can carry a token).
    pub(crate) fn progress(&self, state: &State) -> Progress {
        let trajectory = &state.trajectory;
        let activity = trajectory.activity();
        Progress {
            tool_calls: activity.tool_calls,
            last_tool_title: trajectory
                .tool_calls()
                .last()
                .map(|call| self.redactor.redact(call.display_title()).into_owned())
                .filter(|t| !t.is_empty()),
            message_chunks: activity.message_chunks,
            thought_chunks: activity.thought_chunks,
            tokens: trajectory
                .context_usage()
                .and_then(|c| c.used)
                .or_else(|| trajectory.usage().and_then(|u| u.tokens())),
        }
    }
}

/// Write queued lines to stdin until told to close.
pub(crate) async fn writer(
    mut stdin: Pin<Box<dyn AsyncWrite + Send>>,
    mut queue: mpsc::UnboundedReceiver<Outgoing>,
) {
    while let Some(item) = queue.recv().await {
        match item {
            Outgoing::Line { line, done } => {
                let result = async {
                    stdin.write_all(line.as_bytes()).await?;
                    stdin.write_all(b"\n").await?;
                    stdin.flush().await
                }
                .await;
                if let Some(done) = done {
                    let _ = done.send(result);
                }
            }
            Outgoing::Close => break,
        }
    }
    let _ = stdin.shutdown().await;
}

/// Keep stderr's tail and pass its lines on as agent output.
pub(crate) async fn stderr_reader(stderr: Pin<Box<dyn AsyncRead + Send>>, shared: Arc<Shared>) {
    // One stderr line is never worth more than a megabyte of memory.
    let mut lines = LineReader::new(stderr, 1024 * 1024);
    while let Ok(Some(line)) = lines.next_line().await {
        let text = match line {
            Line::Text(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Line::TooLong(size) => format!("[BenchFlow skipped a {size}-byte line on stderr]"),
        };
        let text = keep_start(&text);
        shared.stderr().push(&text);
        shared.emit(AcpEvent::AgentOutput {
            stream: OutputStream::Stderr,
            line: shared.redactor.redact(&text).into_owned(),
        });
    }
    shared.stderr().done = true;
}

/// Own the process handle: publish the exit, kill on request.
pub(crate) async fn exit_watcher(
    mut handle: Box<dyn ProcessHandle>,
    shared: Arc<Shared>,
    mut kills: mpsc::Receiver<oneshot::Sender<()>>,
) {
    let mut killing = false;
    loop {
        tokio::select! {
            code = handle.wait() => {
                shared.exit.send_replace(Exit::Exited(code.ok()));
                return;
            }
            request = kills.recv(), if !killing => match request {
                Some(done) => {
                    if let Err(e) = handle.kill().await {
                        tracing::warn!("could not kill the agent process: {e}");
                    }
                    killing = true;
                    let _ = done.send(());
                }
                None => killing = true,
            },
        }
    }
}

/// Read stdout until it ends, dispatching every message.
pub(crate) async fn reader(
    stdout: Pin<Box<dyn AsyncRead + Send>>,
    shared: Arc<Shared>,
    outgoing: mpsc::UnboundedSender<Outgoing>,
    line_limit: Option<usize>,
    max_line: usize,
) {
    let mut lines = LineReader::new(stdout, max_line);
    let mut saw_protocol = false;
    loop {
        let line = match lines.next_line().await {
            Ok(Some(Line::Text(bytes))) => bytes,
            Ok(Some(Line::TooLong(size))) => {
                noise(
                    &shared,
                    &format!("[BenchFlow skipped a {size}-byte line from the agent]"),
                );
                continue;
            }
            Ok(None) => break,
            Err(e) => {
                tracing::warn!("reading the agent's stdout failed: {e}");
                break;
            }
        };
        let message = match line_limit {
            Some(limit) if line.len() > limit => {
                tracing::info!("shrank a {}-byte agent output line", line.len());
                match wire::shrink(&line, limit) {
                    Some(Value::Object(map)) if wire::is_envelope(&map) => Some(map),
                    _ => None,
                }
            }
            _ => {
                let text = String::from_utf8_lossy(&line);
                let text = text.trim();
                if text.is_empty() {
                    continue;
                }
                if saw_protocol {
                    wire::decode(text)
                } else {
                    wire::decode_first(text)
                }
            }
        };
        match message {
            Some(message) => {
                saw_protocol = true;
                dispatch(&shared, &outgoing, message);
            }
            None => {
                let text = String::from_utf8_lossy(&line);
                let text = text.trim();
                if !text.is_empty() {
                    noise(&shared, text);
                }
            }
        }
    }
    let pending = {
        let mut state = shared.state();
        state.closed = true;
        std::mem::take(&mut state.pending)
    };
    for (_, waiter) in pending {
        let _ = waiter.send(Err(Closed));
    }
}

/// A line that is not ACP: log it and pass it on, redacted.
fn noise(shared: &Shared, text: &str) {
    shared.state().noise_lines += 1;
    let line = shared.redactor.redact(&keep_start(text)).into_owned();
    tracing::debug!(
        "agent output: {}",
        line.chars().take(200).collect::<String>()
    );
    shared.emit(AcpEvent::AgentOutput {
        stream: OutputStream::Stdout,
        line,
    });
}

/// How much of one non-ACP line is kept (0.8 kept the first 64 KiB of an
/// oversized one).
const NOISE_KEEP: usize = 64 * 1024;

/// The first [`NOISE_KEEP`] bytes of a line, with a note when it was longer.
fn keep_start(text: &str) -> std::borrow::Cow<'_, str> {
    if text.len() <= NOISE_KEEP {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut end = NOISE_KEEP;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    std::borrow::Cow::Owned(format!(
        "{} [BenchFlow cut the rest of this {}-byte line]",
        &text[..end],
        text.len()
    ))
}

fn send(outgoing: &mpsc::UnboundedSender<Outgoing>, message: Value) {
    let _ = outgoing.send(Outgoing::Line {
        line: message.to_string(),
        done: None,
    });
}

fn dispatch(
    shared: &Shared,
    outgoing: &mpsc::UnboundedSender<Outgoing>,
    mut message: Map<String, Value>,
) {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .map(str::to_string);
    let has_id = message.contains_key("id");
    match (method, has_id) {
        (Some(method), true) => answer_request(shared, outgoing, &method, &message),
        (Some(method), false) => {
            if method == "session/update" {
                // Taken, not cloned: an update can be megabytes.
                let params = message.remove("params").unwrap_or(Value::Null);
                session_update(shared, params, Receipt::now());
            } else {
                tracing::debug!("ignoring agent notification {method}");
            }
        }
        (None, _) => deliver(shared, &message),
    }
}

/// Route a response to the request waiting for it.
fn deliver(shared: &Shared, message: &Map<String, Value>) {
    let id = message.get("id").and_then(|id| {
        id.as_i64()
            .or_else(|| id.as_str().and_then(|s| s.parse().ok()))
    });
    let Some(waiter) = id.and_then(|id| shared.state().pending.remove(&id)) else {
        tracing::debug!(
            "ignoring a response to no waiting request: {:?}",
            message.get("id")
        );
        return;
    };
    let at = tokio::time::Instant::now();
    let result = match message.get("error") {
        Some(error) if !error.is_null() => Err(RpcError {
            code: error.get("code").and_then(Value::as_i64).unwrap_or(-1),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Unknown error")
                .to_string(),
            data: error.get("data").filter(|d| !d.is_null()).cloned(),
        }),
        _ => Ok(message
            .get("result")
            .cloned()
            .unwrap_or(Value::Object(Map::new()))),
    };
    let _ = waiter.send(Ok(Answer { result, at }));
}

/// Answer an agent's request. A permission prompt gets the most permissive
/// option (BenchFlow runs agents unattended in a sandbox); anything else is
/// method-not-found, since BenchFlow offers no file system or terminal over
/// ACP and an empty success would let the agent think something ran.
fn answer_request(
    shared: &Shared,
    outgoing: &mpsc::UnboundedSender<Outgoing>,
    method: &str,
    message: &Map<String, Value>,
) {
    let id = message.get("id").cloned().unwrap_or(Value::Null);
    if method == "session/request_permission" {
        let params = message.get("params");
        let options: &[Value] = params
            .and_then(|p| p.get("options"))
            .and_then(Value::as_array)
            .map_or(&[], Vec::as_slice);
        let option_id = auto_approve_option_id(options);
        send(
            outgoing,
            json!({"jsonrpc": "2.0", "id": id, "result": {"outcome": {"outcome": "selected", "optionId": option_id}}}),
        );
        let tool_call = params
            .and_then(|p| p.get("toolCall"))
            .and_then(|t| t.get("toolCallId"))
            .and_then(Value::as_str)
            .map(str::to_string);
        shared.emit(AcpEvent::Permission {
            tool_call,
            option_id,
        });
    } else {
        tracing::warn!(
            "agent asked for {method}, which BenchFlow does not offer; answering method-not-found"
        );
        send(
            outgoing,
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("Method not found: {method}")}}),
        );
    }
}

/// The most permissive option (0.8's `_auto_approve_option_id`):
/// `bypassPermissions`, else the first `allow_always`, else the last
/// `allow_once`, else the first option.
pub(crate) fn auto_approve_option_id(options: &[Value]) -> String {
    let Some(first) = options.first() else {
        return "default".to_string();
    };
    let id_of = |o: &Value| {
        o.get("optionId")
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let mut chosen = id_of(first).unwrap_or_else(|| "default".to_string());
    for option in options {
        if option.get("optionId").and_then(Value::as_str) == Some("bypassPermissions") {
            return "bypassPermissions".to_string();
        }
        match option.get("kind").and_then(Value::as_str) {
            Some("allow_always") => {
                if let Some(id) = id_of(option) {
                    chosen = id;
                }
                break;
            }
            Some("allow_once") => {
                if let Some(id) = id_of(option) {
                    chosen = id;
                }
            }
            _ => {}
        }
    }
    chosen
}

/// Record a `session/update` notification's `params`.
pub(crate) fn session_update(shared: &Shared, params: Value, receipt: Receipt) {
    let events = {
        let mut state = shared.state();
        if let Some(buffer) = state.opening.as_mut() {
            buffer.push((params, receipt));
            return;
        }
        apply_update(shared, &mut state, &params, receipt)
    };
    for event in events {
        shared.emit(event);
    }
}

/// Apply one update to the trajectory; the events to emit (redacted:
/// the trajectory keeps the agent's words, events and logs do not).
pub(crate) fn apply_update(
    shared: &Shared,
    state: &mut State,
    params: &Value,
    receipt: Receipt,
) -> Vec<AcpEvent> {
    let Some(session_id) = state.session_id.clone() else {
        return Vec::new();
    };
    let from = params
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !state.session_placeholder && !from.is_empty() && from != session_id {
        tracing::warn!(
            "ignoring an ACP update for session {from:?} (this session is {session_id:?})"
        );
        return Vec::new();
    }
    let Some(update) = params.get("update") else {
        return Vec::new();
    };
    let update = session_notice(&shared.redactor, update);
    let change = state.trajectory.apply_update(&update, receipt);
    let mut events = Vec::new();
    if change == Change::Notice {
        let text = |key: &str| {
            let text = update.get(key).and_then(Value::as_str).unwrap_or("").trim();
            shared.redactor.redact(text).into_owned()
        };
        let severity = text("severity");
        events.push(AcpEvent::Notice {
            severity: if severity.is_empty() {
                "info".to_string()
            } else {
                severity
            },
            title: text("title"),
        });
    }
    if change != Change::Other {
        events.push(AcpEvent::Progress(shared.progress(state)));
    }
    events
}

/// A typed-failure `session_info_update` (claude-agent-acp's retries and
/// advisories): a failure or retry is logged; an advisory the adapter could
/// not classify becomes a `notice`, kept as evidence but never as the
/// agent's words (0.8's `_session_notice`). Logged titles are redacted; the
/// returned update is not (the trajectory writer redacts what it writes).
fn session_notice(redactor: &Redactor, update: &Value) -> Value {
    if update.get("sessionUpdate").and_then(Value::as_str) != Some("session_info_update") {
        return update.clone();
    }
    let Some(failure) = limits::session_failure(update.get("_meta")) else {
        return update.clone();
    };
    let title = limits::field(failure, "title").trim().to_string();
    if title.is_empty() {
        return update.clone();
    }
    let shown = redactor.redact(&title);
    if !limits::is_advisory(failure) {
        tracing::warn!("Agent reported: {shown}");
        return update.clone();
    }
    if failure.get("category").and_then(Value::as_str) == Some("unknown") {
        tracing::warn!("Agent notice: {shown}");
        let details = limits::field(failure, "details").trim().to_string();
        let mut notice = json!({"sessionUpdate": "notice", "severity": "warning", "title": title});
        if !details.is_empty() {
            notice["description"] = Value::String(details);
        }
        return notice;
    }
    tracing::info!("Agent notice: {shown}");
    update.clone()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn permission_picks_the_most_permissive_option() {
        let options = |v: Value| v.as_array().unwrap().clone();
        assert_eq!(auto_approve_option_id(&[]), "default");
        assert_eq!(
            auto_approve_option_id(&options(json!([
                {"optionId": "reject", "kind": "reject_once"},
                {"optionId": "once", "kind": "allow_once"},
                {"optionId": "always", "kind": "allow_always"},
            ]))),
            "always"
        );
        assert_eq!(
            auto_approve_option_id(&options(json!([
                {"optionId": "reject", "kind": "reject_once"},
                {"optionId": "once", "kind": "allow_once"},
            ]))),
            "once"
        );
        assert_eq!(
            auto_approve_option_id(&options(json!([
                {"optionId": "default", "kind": "allow_once"},
                {"optionId": "bypassPermissions", "kind": "allow_always"},
            ]))),
            "bypassPermissions"
        );
        assert_eq!(
            auto_approve_option_id(&options(
                json!([{"optionId": "only", "kind": "reject_once"}])
            )),
            "only"
        );
    }

    #[test]
    fn stderr_keeps_its_tail() {
        let mut tail = Tail::default();
        for i in 0..20_000 {
            tail.push(&format!("line {i}"));
        }
        assert!(tail.text.len() <= STDERR_KEEP);
        assert!(tail.last(20).ends_with("line 19999\n"));
    }
}
