//! Running a headless agent: its output read, parsed, recorded, and bounded.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use benchflow_sandbox::Process;
use chrono::Utc;
use tokio::io::AsyncRead;
use tokio::time::Instant;

use super::classify::{Signal, classify};
use super::error::HeadlessError;
use super::parser::{Failure, Hint, Out, Parser};
use crate::acp::wire::{self, Line, LineReader};
use crate::acp::{AcpEvent, EventSink, OutputStream, PENDING_GRACE_MULTIPLIER, Progress};
use crate::trajectory::{Change, Event, Receipt, Redactor, Trajectory, Usage};

/// Bytes of stderr kept for error messages (0.8's tail limit).
const STDERR_KEEP: usize = 64 * 1024;
/// Non-event stdout lines kept for classifying a failure.
const NOISE_KEEP: usize = 200;
/// How long the process may take to report its exit once stdout closes.
const EXIT_GRACE: Duration = Duration::from_secs(10);

/// How a headless run is bounded and reported.
#[derive(Clone)]
pub struct RunnerOptions {
    /// The wall-clock budget; the agent is stopped when it runs out.
    pub wall: Option<Duration>,
    /// Stop the agent when it writes nothing on stdout this long (three
    /// times as long while a tool call is open, as for ACP).
    pub idle: Option<Duration>,
    /// Removes the login's secrets from log lines and error messages.
    pub redactor: Redactor,
    /// Live progress and the agent's log lines.
    pub sink: Option<EventSink>,
    /// The login's label, for a usage-limit message.
    pub login: Option<String>,
    /// Lines longer than this are shrunk before parsing (JSON) or cut (text).
    pub line_limit: usize,
}

impl Default for RunnerOptions {
    fn default() -> Self {
        Self {
            wall: None,
            idle: None,
            redactor: Redactor::default(),
            sink: None,
            login: None,
            line_limit: wire::DEFAULT_LINE_LIMIT,
        }
    }
}

impl std::fmt::Debug for RunnerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnerOptions")
            .field("wall", &self.wall)
            .field("idle", &self.idle)
            .field("login", &self.login)
            .field("line_limit", &self.line_limit)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct State {
    trajectory: Trajectory,
    usage: Option<Usage>,
    cost: Option<f64>,
    stderr: String,
}

/// What a run recorded, readable while it runs (for the trajectory writer)
/// and after.
#[derive(Clone, Default)]
pub struct Recording(Arc<Mutex<State>>);

impl Recording {
    fn state(&self) -> MutexGuard<'_, State> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A copy of every event so far.
    pub fn snapshot(&self) -> Vec<Event> {
        self.state().trajectory.snapshot()
    }

    /// Run `f` on the trajectory.
    pub fn with<R>(&self, f: impl FnOnce(&Trajectory) -> R) -> R {
        f(&self.state().trajectory)
    }

    /// The agent's reported token counts, cumulative.
    pub fn usage(&self) -> Option<Usage> {
        self.state().usage
    }

    /// The agent's reported spend, in dollars.
    pub fn cost_usd(&self) -> Option<f64> {
        self.state().cost
    }

    /// The last `chars` characters of stderr, secrets redacted.
    pub fn stderr_tail(&self, chars: usize) -> String {
        let state = self.state();
        let count = state.stderr.chars().count();
        state
            .stderr
            .chars()
            .skip(count.saturating_sub(chars))
            .collect()
    }
}

impl std::fmt::Debug for Recording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recording").finish_non_exhaustive()
    }
}

/// How the process ended, when it ended on its own.
#[derive(Debug, Clone, PartialEq)]
pub struct Finished {
    pub exit_code: Option<i32>,
}

/// Runs one headless agent process.
pub struct Runner {
    options: RunnerOptions,
    recording: Recording,
}

enum Stop {
    Wall,
    Idle,
}

impl Runner {
    pub fn new(options: RunnerOptions) -> Self {
        Self {
            options,
            recording: Recording::default(),
        }
    }

    /// What the run records; clone it before [`Runner::run`] to watch.
    pub fn recording(&self) -> Recording {
        self.recording.clone()
    }

    /// Record `prompt` as the user's message, then read the process's stdout
    /// through `parser` until it ends, its budget runs out, or it goes idle,
    /// and say how it ended.
    pub async fn run(
        &self,
        process: Process,
        mut parser: Box<dyn Parser>,
        prompt: &str,
    ) -> Result<Finished, HeadlessError> {
        let Process {
            stdin,
            stdout,
            stderr,
            mut handle,
        } = process;
        // Headless commands read their instruction from a file; an open
        // stdin would only let a CLI wait for input that never comes.
        drop(stdin);
        self.recording
            .state()
            .trajectory
            .record_user_prompt(prompt, Receipt::now());

        let stderr_task = tokio::spawn(read_stderr(
            stderr,
            self.recording.clone(),
            self.options.redactor.clone(),
            self.options.sink.clone(),
        ));

        let started = Instant::now();
        let deadline = self.options.wall.map(|w| started + w);
        let mut last_output = Instant::now();
        let mut reader = LineReader::new(stdout, wire::DEFAULT_MAX_LINE);
        let mut failures: Vec<Failure> = Vec::new();
        let mut noise: VecDeque<String> = VecDeque::new();
        let mut understood = false;
        let mut outs = Vec::new();

        let stop = loop {
            let pending = !self
                .recording
                .with(|t| t.pending_tool_call_ids())
                .is_empty();
            let idle_deadline = self.options.idle.map(|idle| {
                last_output
                    + if pending {
                        idle * PENDING_GRACE_MULTIPLIER
                    } else {
                        idle
                    }
            });
            let wake = match (deadline, idle_deadline) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            tokio::select! {
                line = reader.next_line() => match line {
                    Ok(Some(Line::Text(bytes))) => {
                        last_output = Instant::now();
                        let text = self.fit(&bytes);
                        outs.clear();
                        if parser.line(&text, &mut outs) {
                            understood = true;
                        } else {
                            self.noise(&text, &mut noise);
                        }
                        self.apply(&mut outs, &mut failures);
                    }
                    Ok(Some(Line::TooLong(size))) => {
                        last_output = Instant::now();
                        self.noise(
                            &format!("[BenchFlow skipped a line of {size} bytes from the agent's stdout]"),
                            &mut noise,
                        );
                    }
                    Ok(None) => break None,
                    Err(source) => {
                        let _ = handle.kill().await;
                        stderr_task.abort();
                        return Err(HeadlessError::Io {
                            context: "reading the agent's output".into(),
                            source,
                        });
                    }
                },
                _ = sleep_until(wake), if wake.is_some() => {
                    let now = Instant::now();
                    if deadline.is_some_and(|d| now >= d) {
                        break Some(Stop::Wall);
                    }
                    if idle_deadline.is_some_and(|d| now >= d) {
                        break Some(Stop::Idle);
                    }
                }
            }
        };

        outs.clear();
        parser.finish(&mut outs);
        self.apply(&mut outs, &mut failures);

        let exit_code = match stop {
            Some(_) => {
                let _ = handle.kill().await;
                None
            }
            None => {
                let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
                let grace = remaining.map_or(EXIT_GRACE, |r| r.min(EXIT_GRACE));
                match tokio::time::timeout(grace, handle.wait()).await {
                    Ok(Ok(code)) => Some(code),
                    Ok(Err(_)) => None,
                    Err(_) => {
                        // Stdout closed but the process lingers (a child
                        // holding stderr): it is done talking.
                        let _ = handle.kill().await;
                        None
                    }
                }
            }
        };
        // The stderr reader ends when the process does; never wait long.
        let _ = tokio::time::timeout(Duration::from_secs(2), stderr_task).await;

        let mut state = self.recording.state();
        state.trajectory.finish_prompt();
        match stop {
            Some(Stop::Wall) => {
                let budget = self.options.wall.unwrap_or_default();
                state.trajectory.record_timeout(budget, Receipt::now());
                return Err(HeadlessError::Timeout { after: budget });
            }
            Some(Stop::Idle) => {
                return Err(HeadlessError::Idle {
                    after: self.options.idle.unwrap_or_default(),
                    pending_tool_calls: state.trajectory.pending_tool_call_ids(),
                });
            }
            None => {}
        }
        let activity = state.trajectory.activity();
        let did_something = activity.tool_calls > 0
            || activity.message_chunks > 0
            || activity.thought_chunks > 0
            || state.usage.and_then(|u| u.output_tokens).unwrap_or(0) > 0;
        let stderr_text = state.stderr.clone();
        drop(state);

        let fatal = failures.iter().rev().find(|f| f.fatal);
        let failed_exit = exit_code.is_some_and(|c| c != 0);
        if fatal.is_none() && !failed_exit && (did_something || exit_code.is_none()) {
            return Ok(Finished { exit_code });
        }
        Err(self.diagnose(
            fatal,
            &failures,
            &stderr_text,
            &noise,
            exit_code,
            understood,
            did_something,
        ))
    }

    /// Why a run that failed (or ended having done nothing) failed: the
    /// agent's own error first, then its hints, then stderr and stdout noise.
    #[allow(clippy::too_many_arguments)]
    fn diagnose(
        &self,
        fatal: Option<&Failure>,
        failures: &[Failure],
        stderr: &str,
        noise: &VecDeque<String>,
        exit_code: Option<i32>,
        understood: bool,
        did_something: bool,
    ) -> HeadlessError {
        let now = Utc::now();
        let redact = |s: &str| self.options.redactor.redact(s).into_owned();
        // A setting the agent says it lacks is the answer, whatever else
        // its words resemble ("missing API key" in a factory's message).
        if let Some(f) = fatal.filter(|f| f.hint == Some(Hint::Setup)) {
            return HeadlessError::Setup {
                message: cap(&redact(&f.message)),
                fix: f.fix.as_deref().map(redact),
            };
        }
        // BenchFlow's own launch wrapper refused the run's settings (a
        // missing variable, a model the agent cannot take): the user's to fix.
        // It refuses before the agent starts; once the agent has worked, a
        // line like it on stderr may be the agent's own output.
        if !did_something && let Some(line) = wrapper_refusal(stderr) {
            return HeadlessError::Setup {
                message: cap(&redact(line)),
                fix: None,
            };
        }
        let noise_text: Vec<&str> = noise.iter().map(String::as_str).collect();
        let candidates = fatal
            .map(|f| f.message.clone())
            .into_iter()
            .chain(failures.iter().rev().map(|f| f.message.clone()))
            .chain([stderr.to_string(), noise_text.join("\n")]);
        for text in candidates {
            if let Some((signal, line)) = classify(&text, now) {
                return self.error_for(signal, cap(&redact(&line)));
            }
        }
        // The agent's structured hint, when its words matched nothing.
        if let Some(f) = failures.iter().rev().find(|f| f.hint.is_some()) {
            let message = redact(&f.message);
            return match f.hint {
                Some(Hint::Login) => HeadlessError::LoginRejected { message },
                Some(Hint::Model) => HeadlessError::ModelUnavailable { message },
                Some(Hint::Setup) => HeadlessError::Setup {
                    message,
                    fix: f.fix.as_deref().map(redact),
                },
                Some(Hint::UsageLimit) => HeadlessError::UsageLimit {
                    reset_at: None,
                    window: None,
                    message,
                    login: self.options.login.clone(),
                },
                _ => HeadlessError::Provider {
                    message,
                    retryable: true,
                    category: "provider_rate_limit",
                },
            };
        }
        if let Some(f) = fatal {
            return HeadlessError::AgentFailed {
                message: redact(&f.message),
                code: exit_code.filter(|c| *c != 0),
            };
        }
        let tail = self.recording.stderr_tail(4000);
        if exit_code == Some(0) && !did_something {
            return HeadlessError::AgentFailed {
                message: "it exited without a message, a thought, or a tool call".into(),
                code: None,
            };
        }
        HeadlessError::Exited {
            code: exit_code,
            stderr_tail: if tail.trim().is_empty() {
                redact(&noise_text[noise_text.len().saturating_sub(5)..].join("\n"))
            } else {
                tail
            },
            started: understood,
        }
    }

    fn error_for(&self, signal: Signal, line: String) -> HeadlessError {
        match signal {
            Signal::UsageLimit(info) => HeadlessError::UsageLimit {
                reset_at: info.resets_at,
                window: info.window,
                message: if info.detail.is_empty() {
                    line
                } else {
                    self.options.redactor.redact(&info.detail).into_owned()
                },
                login: self.options.login.clone(),
            },
            Signal::Login => HeadlessError::LoginRejected { message: line },
            Signal::Model => HeadlessError::ModelUnavailable { message: line },
            Signal::RateLimit => HeadlessError::Provider {
                message: line,
                retryable: true,
                category: "provider_rate_limit",
            },
            Signal::Unavailable => HeadlessError::Provider {
                message: line,
                retryable: true,
                category: "infra_failure",
            },
            // A refused request is not retried (0.8: permanent).
            Signal::Rejected => HeadlessError::Provider {
                message: line,
                retryable: false,
                category: "provider_rejected",
            },
            // A prompt over the context window is the agent's to manage, as
            // on every ACP path (0.8's fault "agent").
            Signal::Overflow => HeadlessError::AgentFailed {
                message: line,
                code: None,
            },
        }
    }

    /// A line as text, shrunk to the line limit: JSON keeps its shape
    /// (images and long strings cut, as for ACP), text is cut.
    fn fit(&self, bytes: &[u8]) -> String {
        let limit = self.options.line_limit;
        if bytes.len() <= limit {
            return String::from_utf8_lossy(bytes).into_owned();
        }
        if let Some(shrunk) = wire::shrink(bytes, limit) {
            if !shrunk.fits {
                tracing::warn!(
                    "could not rewrite a {}-byte agent output line to fit the {limit}-byte limit (what may not be cut is too long); passed it on over the limit",
                    bytes.len()
                );
            }
            return shrunk.value.to_string();
        }
        let text = String::from_utf8_lossy(&bytes[..limit]);
        format!(
            "{}[BenchFlow cut this line at {limit} of its {} bytes]",
            text.trim_end_matches('\u{FFFD}'),
            bytes.len()
        )
    }

    /// A stdout line that is not part of the format: the agent log.
    fn noise(&self, text: &str, noise: &mut VecDeque<String>) {
        let line = self.options.redactor.redact(text).into_owned();
        if noise.len() == NOISE_KEEP {
            noise.pop_front();
        }
        noise.push_back(line.clone());
        if let Some(sink) = &self.options.sink {
            sink(AcpEvent::AgentOutput {
                stream: OutputStream::Stdout,
                line,
            });
        }
    }

    /// Record what a line said.
    fn apply(&self, outs: &mut Vec<Out>, failures: &mut Vec<Failure>) {
        if outs.is_empty() {
            return;
        }
        let mut changed = false;
        let progress = {
            let mut state = self.recording.state();
            for out in outs.drain(..) {
                match out {
                    Out::Update(update) => {
                        let change = state.trajectory.apply_update(&update, Receipt::now());
                        changed |= change != Change::Other;
                    }
                    Out::Usage(usage) => {
                        state.usage = Some(usage);
                        state.trajectory.record_usage(usage);
                        changed = true;
                    }
                    Out::Cost(cost) => state.cost = Some(cost),
                    Out::Failure(failure) => {
                        // The agent's error in its own words, for the log.
                        if let Some(sink) = &self.options.sink {
                            sink(AcpEvent::AgentOutput {
                                stream: OutputStream::Stdout,
                                line: format!(
                                    "[benchflow] the agent reported an error{}: {}",
                                    if failure.fatal { "" } else { " (not fatal)" },
                                    self.options.redactor.redact(&failure.message)
                                ),
                            });
                        }
                        failures.push(failure);
                    }
                }
            }
            changed.then(|| self.progress(&state))
        };
        if let (Some(progress), Some(sink)) = (progress, &self.options.sink) {
            sink(AcpEvent::Progress(progress));
        }
    }

    fn progress(&self, state: &State) -> Progress {
        let trajectory = &state.trajectory;
        let activity = trajectory.activity();
        Progress {
            tool_calls: activity.tool_calls,
            last_tool_title: trajectory
                .tool_calls()
                .last()
                .map(|call| {
                    self.options
                        .redactor
                        .redact(call.display_title())
                        .into_owned()
                })
                .filter(|t| !t.is_empty()),
            message_chunks: activity.message_chunks,
            thought_chunks: activity.thought_chunks,
            tokens: state.usage.and_then(|u| u.tokens()),
        }
    }
}

/// Characters of an agent's error line kept in a message; the rest is in
/// the agent log and the problem's detail.
const MESSAGE_CHARS: usize = 300;

/// The line, cut to [`MESSAGE_CHARS`], with runs of whitespace (escaped
/// JSON newlines included) folded to one space.
fn cap(line: &str) -> String {
    let folded = line
        .replace("\\n", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if folded.chars().count() <= MESSAGE_CHARS {
        folded
    } else {
        let cut: String = folded.chars().take(MESSAGE_CHARS).collect();
        format!("{cut}…")
    }
}

async fn sleep_until(at: Option<Instant>) {
    match at {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

/// Keep stderr's tail (redacted) and pass each line to the agent log.
async fn read_stderr(
    stderr: Pin<Box<dyn AsyncRead + Send>>,
    recording: Recording,
    redactor: Redactor,
    sink: Option<EventSink>,
) {
    let mut lines = LineReader::new(stderr, 1024 * 1024);
    while let Ok(Some(line)) = lines.next_line().await {
        let text = match line {
            Line::Text(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Line::TooLong(size) => format!("[BenchFlow skipped a stderr line of {size} bytes]"),
        };
        let text = redactor.redact(&text).into_owned();
        {
            let mut state = recording.state();
            state.stderr.push_str(&text);
            state.stderr.push('\n');
            if state.stderr.len() > STDERR_KEEP {
                let mut cut = state.stderr.len() - STDERR_KEEP;
                while !state.stderr.is_char_boundary(cut) {
                    cut += 1;
                }
                state.stderr.drain(..cut);
            }
        }
        if let Some(sink) = &sink {
            sink(AcpEvent::AgentOutput {
                stream: OutputStream::Stderr,
                line: text,
            });
        }
    }
}

/// A line in which a manifest's launch command refuses the run's settings:
/// `BenchFlow: Cortex Code needs SNOWFLAKE_ACCOUNT ...; pass them with
/// --env`, `BenchFlow: mcode takes anthropic/ ... models, not 'x'`. Other
/// `BenchFlow:` lines (a log the wrapper prints) are not refusals.
fn wrapper_refusal(stderr: &str) -> Option<&str> {
    stderr.lines().map(str::trim).find(|l| {
        l.strip_prefix("BenchFlow: ")
            .is_some_and(|rest| rest.contains(" needs ") || rest.contains(" takes "))
    })
}

#[cfg(test)]
mod wrapper_tests {
    use super::wrapper_refusal;

    #[test]
    fn a_wrappers_refusal_is_found_and_its_log_lines_are_not() {
        let cortex = "BenchFlow: Cortex Code needs SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER; pass them with --env\n";
        assert!(
            wrapper_refusal(cortex)
                .unwrap()
                .contains("SNOWFLAKE_ACCOUNT")
        );
        let mcode = "x\nBenchFlow: mcode takes anthropic/, minimax/, openai/ or deepseek/ models, not 'qwen/q'\n";
        assert!(wrapper_refusal(mcode).is_some());
        assert!(wrapper_refusal("BenchFlow: Junie's log /tmp/j.log says:\nERROR boom\n").is_none());
        assert!(wrapper_refusal("cortex needs a newer glibc\n").is_none());
    }
}
