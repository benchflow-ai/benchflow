//! Terminus-2 (agent `terminus-2`): the model reads the terminal and
//! answers with keystrokes, which are typed into a tmux session in the
//! sandbox.
//!
//! Ported from Harbor's Terminus-2 (`src/harbor/agents/terminus_2/`,
//! https://github.com/laude-institute/harbor, harbor 0.24.0, Apache License
//! 2.0). The prompt templates are Harbor's files, unchanged
//! (`templates/`), as are the parsers' checks and wording, the
//! `task_complete` double confirmation, the output limits, and the
//! three-step context summarization (summary, questions, answers) with its
//! short-summary and last-resort fallbacks. Differences:
//!
//! - Tokens are estimated at four characters each (Harbor asks LiteLLM),
//!   and the context window defaults to LiteLLM's fallback of 1,000,000
//!   tokens, which is what Harbor uses for a model LiteLLM does not know.
//! - A reply cut off at the output limit that the XML parser salvages is
//!   used as the reply (Harbor's code returns the bare string there and
//!   fails on it).
//! - Transient model errors are retried by the model client; Harbor adds
//!   three more attempts around each call.
//! - No asciinema recording, skills section, or per-summary subagent
//!   trajectory files; the summary calls are in `agent/model_calls.jsonl`
//!   and the handoff is a notice in the trajectory.

pub mod json_parser;
pub mod tmux;
pub mod xml_parser;

use serde_json::{Value, json};

use super::RunContext;
use super::budget::{Meter, Stop};
use super::chat::{ChatRequest, FinishReason, Message};
use super::client::ModelClient;
use super::error::ModelErrorKind;
use super::recorder::Recorder;
use crate::trajectory::ToolCallStatus;
use tmux::{StartError, TmuxSession};

/// Harbor's prompt for JSON replies.
pub const JSON_TEMPLATE: &str = include_str!("templates/terminus-json-plain.txt");
/// Harbor's prompt for XML replies.
pub const XML_TEMPLATE: &str = include_str!("templates/terminus-xml-plain.txt");
/// Harbor's prompt after a blocking command times out. Terminus-2 sends
/// every command without blocking, so Harbor never uses it either.
pub const TIMEOUT_TEMPLATE: &str = include_str!("templates/timeout.txt");

const SUMMARY_PROMPT: &str = "You are about to hand off your work to another AI agent.\n            Please provide a comprehensive summary of what you have\n            accomplished so far on this task:\n\nOriginal Task: {original_instruction}\n\nBased on the conversation history, please provide a detailed summary covering:\n1. **Major Actions Completed** - List each significant command you executed\n            and what you learned from it.\n2. **Important Information Learned** - A summary of crucial findings, file\n            locations, configurations, error messages, or system state discovered.\n3. **Challenging Problems Addressed** - Any significant issues you\n            encountered and how you resolved them.\n4. **Current Status** - Exactly where you are in the task completion process.\n\n\nBe comprehensive and detailed. The next agent needs to understand everything\n            that has happened so far in order to continue.";

const QUESTION_PROMPT: &str = "You are picking up work from a previous AI agent on this task:\n\n**Original Task:** {original_instruction}\n\n**Summary from Previous Agent:**\n{summary}\n\n**Current Terminal Screen:**\n{current_screen}\n\nPlease begin by asking several questions (at least five, more if necessary)\nabout the current state of the solution that are not answered in the summary\nfrom the prior agent. After you ask these questions you will be on your own,\nso ask everything you need to know.";

/// The tmux session's name (Harbor names it after the agent).
pub const SESSION_NAME: &str = "terminus-2";
/// Longest wait one command may ask for, in seconds.
const MAX_DURATION_SEC: f64 = 60.0;
/// Bytes of terminal output kept in a prompt (`_limit_output_length`).
const MAX_OUTPUT_BYTES: usize = 10_000;

/// A parsed command: keys to type and seconds to wait after.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCommand {
    pub keystrokes: String,
    pub duration: f64,
}

/// What a parser made of a reply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParseResult {
    pub commands: Vec<ParsedCommand>,
    pub is_task_complete: bool,
    /// Empty when the reply parsed.
    pub error: String,
    /// `- `-prefixed lines, or empty.
    pub warning: String,
    pub analysis: String,
    pub plan: String,
}

pub(crate) fn join_warnings(warnings: &[String]) -> String {
    if warnings.is_empty() {
        String::new()
    } else {
        format!("- {}", warnings.join("\n- "))
    }
}

pub(crate) fn combine_warnings(auto: &str, existing: &str) -> String {
    if existing.is_empty() {
        format!("- {auto}")
    } else {
        format!("- {auto}\n{existing}")
    }
}

/// A string as Python's `repr` writes it (for parser messages).
pub(crate) fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Fill a template the way Python's `str.format` does with named fields:
/// `{name}` takes a value, `{{` and `}}` are literal braces, and values are
/// not scanned again.
pub fn format_template(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(i) = rest.find(['{', '}']) {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        if let Some(after) = tail.strip_prefix("{{") {
            out.push('{');
            rest = after;
        } else if let Some(after) = tail.strip_prefix("}}") {
            out.push('}');
            rest = after;
        } else if tail.starts_with('{')
            && let Some(end) = tail.find('}')
            && let Some((_, v)) = values.iter().find(|(k, _)| *k == &tail[1..end])
        {
            out.push_str(v);
            rest = &tail[end + 1..];
        } else {
            out.push_str(&tail[..1]);
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Keep at most `max_bytes` of terminal output: the first and last halves
/// with a note between (Harbor's `_limit_output_length`).
pub fn limit_output_length(output: &str, max_bytes: usize) -> String {
    let bytes = output.as_bytes();
    if bytes.len() <= max_bytes {
        return output.to_string();
    }
    let portion = max_bytes / 2;
    let first = String::from_utf8_lossy(&bytes[..portion]);
    let first = first.trim_end_matches('\u{fffd}');
    let last = String::from_utf8_lossy(&bytes[bytes.len() - portion..]);
    let last = last.trim_start_matches('\u{fffd}');
    let omitted = bytes.len() - first.len() - last.len();
    format!(
        "{first}\n[... output limited to {max_bytes} bytes; {omitted} interior bytes omitted ...]\n{last}"
    )
}

/// Which reply format the model is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParserKind {
    #[default]
    Json,
    Xml,
}

impl ParserKind {
    fn parse(self, response: &str) -> ParseResult {
        match self {
            ParserKind::Json => json_parser::parse_response(response),
            ParserKind::Xml => xml_parser::parse_response(response),
        }
    }
    fn template(self) -> &'static str {
        match self {
            ParserKind::Json => JSON_TEMPLATE,
            ParserKind::Xml => XML_TEMPLATE,
        }
    }
    fn error_response_type(self) -> &'static str {
        match self {
            ParserKind::Json => "JSON response",
            ParserKind::Xml => "response",
        }
    }
    fn completion_confirmation(self, terminal_output: &str) -> String {
        match self {
            ParserKind::Json => format!(
                "Current terminal state:\n{terminal_output}\n\nAre you sure you want to mark the task as complete? This will trigger your solution to be graded and you won't be able to make any further corrections. If so, include \"task_complete\": true in your JSON response again."
            ),
            ParserKind::Xml => format!(
                "Current terminal state:\n{terminal_output}\n\nAre you sure you want to mark the task as complete? This will trigger your solution to be graded and you won't be able to make any further corrections. If so, include <task_complete>true</task_complete> again."
            ),
        }
    }
}

/// How Terminus-2 runs (Harbor's options that apply here).
#[derive(Debug, Clone)]
pub struct TerminusConfig {
    pub parser: ParserKind,
    /// Most turns; Harbor's default is effectively unlimited.
    pub max_episodes: u64,
    pub enable_summarize: bool,
    /// Summarize before a turn when fewer tokens than this are free.
    pub proactive_summarization_threshold: u64,
    /// The model's context window, in tokens.
    pub context_limit: u64,
    pub pane_width: u32,
    pub pane_height: u32,
    pub temperature: Option<f64>,
    pub reasoning_effort: Option<String>,
    pub max_output_tokens: Option<u32>,
}

impl Default for TerminusConfig {
    fn default() -> Self {
        Self {
            parser: ParserKind::Json,
            max_episodes: 1_000_000,
            enable_summarize: true,
            proactive_summarization_threshold: 8000,
            context_limit: 1_000_000,
            pane_width: 160,
            pane_height: 40,
            temperature: None,
            reasoning_effort: None,
            max_output_tokens: None,
        }
    }
}

/// Estimated tokens in a conversation: four characters a token, plus a few
/// per message for the chat format.
fn count_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(|m| (m.text_len() as u64).div_ceil(4) + 4)
        .sum()
}

/// The text of one reply.
struct Reply {
    content: String,
    reasoning: Option<String>,
}

struct Terminus<'a, 'c> {
    client: &'a ModelClient,
    recorder: &'a Recorder,
    config: &'a TerminusConfig,
    meter: Meter,
    session: TmuxSession<'c>,
    pending_handoff: Option<String>,
}

/// The tmux session start failing, as a stop.
fn start_stop(e: StartError) -> Stop {
    match e {
        StartError::NoTmux(out) => Stop::Setup(format!(
            "Terminus-2 needs tmux, and it is missing from the sandbox and could not be installed: {out}"
        )),
        StartError::Session(out) => Stop::Setup(format!("could not start the tmux session: {out}")),
        StartError::Sandbox(e) => Stop::Sandbox(e),
    }
}

/// Run Terminus-2 on every prompt of `ctx`. The session is shared between
/// prompts; each prompt starts a fresh conversation, as Harbor's multi-step
/// trials do.
pub async fn run(
    client: &ModelClient,
    recorder: &Recorder,
    ctx: &RunContext<'_>,
    config: &TerminusConfig,
) -> Stop {
    let meter = Meter::new(ctx.limits);
    let mut session = TmuxSession::new(ctx.sandbox, SESSION_NAME, ctx.user, ctx.workdir)
        .env(ctx.env.clone())
        .pane(config.pane_width, config.pane_height);
    match meter.bound_wall(session.start()).await {
        Err(stop) => return stop,
        Ok(Err(e)) => return start_stop(e),
        Ok(Ok(())) => {}
    }
    let mut t = Terminus {
        client,
        recorder,
        config,
        meter,
        session,
        pending_handoff: None,
    };
    for prompt in &ctx.prompts {
        if let Err(stop) = t.run_prompt(prompt, ctx.mcp).await {
            recorder.finish();
            return stop;
        }
    }
    recorder.finish();
    Stop::Finished
}

/// MCP servers as Harbor lists them after the instruction.
fn mcp_info(mcp: &[Value]) -> String {
    if mcp.is_empty() {
        return String::new();
    }
    let mut info =
        "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n".to_string();
    for s in mcp {
        let field = |k: &str| {
            s.get(k)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let transport = s
            .get("transport")
            .or_else(|| s.get("type"))
            .and_then(Value::as_str)
            .unwrap_or(if s.get("command").is_some() {
                "stdio"
            } else {
                "http"
            });
        if transport == "stdio" {
            let args: Vec<String> = s
                .get("args")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            info.push_str(&format!(
                "- {}: stdio transport, command: {} {}\n",
                field("name"),
                field("command"),
                args.join(" ")
            ));
        } else {
            info.push_str(&format!(
                "- {}: {transport} transport, url: {}\n",
                field("name"),
                field("url")
            ));
        }
    }
    info
}

impl Terminus<'_, '_> {
    fn sandbox_stop(e: benchflow_sandbox::SandboxError) -> Stop {
        Stop::Sandbox(e)
    }

    async fn incremental(&mut self) -> Result<String, Stop> {
        let out = self
            .meter
            .bound_wall(self.session.incremental_output())
            .await?
            .map_err(Self::sandbox_stop)?;
        Ok(limit_output_length(&out, MAX_OUTPUT_BYTES))
    }

    async fn run_prompt(&mut self, instruction: &str, mcp: &[Value]) -> Result<(), Stop> {
        let mut chat: Vec<Message> = Vec::new();
        let mut pending_completion = false;
        let terminal_state = self.incremental().await?;
        let augmented = format!("{instruction}{}", mcp_info(mcp));
        let initial_prompt = format_template(
            self.config.parser.template(),
            &[
                ("instruction", &augmented),
                ("terminal_state", &terminal_state),
            ],
        );
        self.recorder.user(&initial_prompt);
        let mut prompt = initial_prompt;
        let parser = self.config.parser;
        for episode in 0..self.config.max_episodes {
            let alive = self
                .meter
                .bound_wall(self.session.is_alive())
                .await?
                .map_err(Self::sandbox_stop)?;
            if !alive {
                self.recorder.notice(
                    "warning",
                    "the tmux session ended; Terminus-2 stopped",
                    None,
                );
                return Ok(());
            }
            if self.config.enable_summarize {
                let free = self
                    .config
                    .context_limit
                    .saturating_sub(count_tokens(&chat));
                if free < self.config.proactive_summarization_threshold {
                    match self.summarize(&mut chat, instruction).await {
                        Ok(handoff) => {
                            prompt = handoff.clone();
                            self.pending_handoff = Some(handoff);
                        }
                        Err(
                            stop @ (Stop::Timeout(_)
                            | Stop::Budget(_)
                            | Stop::Idle { .. }
                            | Stop::Sandbox(_)),
                        ) => {
                            return Err(stop);
                        }
                        Err(other) => tracing::warn!("proactive summarization failed: {other:?}"),
                    }
                }
            }

            let reply = self.query_llm(&mut chat, &prompt, instruction).await?;
            self.recorder.usage(&self.client.usage());
            if let Some(handoff) = self.pending_handoff.take() {
                self.recorder.notice(
                    "info",
                    "Performed context summarization and handoff to continue task.",
                    None,
                );
                self.recorder.user(&handoff);
            }
            let result = parser.parse(&reply.content);
            let mut feedback = String::new();
            if !result.error.is_empty() {
                feedback.push_str(&format!("ERROR: {}", result.error));
                if !result.warning.is_empty() {
                    feedback.push_str(&format!("\nWARNINGS: {}", result.warning));
                }
            } else if !result.warning.is_empty() {
                feedback.push_str(&format!("WARNINGS: {}", result.warning));
            }
            if let Some(r) = &reply.reasoning {
                self.recorder.thought(r);
            }
            if feedback.contains("ERROR:") {
                prompt = format!(
                    "Previous response had parsing errors:\n{feedback}\n\nPlease fix these issues and provide a proper {}.",
                    parser.error_response_type()
                );
                self.recorder.message(&reply.content);
                self.recorder.notice(
                    "warning",
                    "the reply could not be parsed; the model was asked to fix it",
                    Some(&feedback),
                );
                continue;
            }
            let mut parts = Vec::new();
            if !result.analysis.is_empty() {
                parts.push(format!("Analysis: {}", result.analysis));
            }
            if !result.plan.is_empty() {
                parts.push(format!("Plan: {}", result.plan));
            }
            self.recorder.message(&parts.join("\n"));

            let commands: Vec<ParsedCommand> = result
                .commands
                .iter()
                .map(|c| ParsedCommand {
                    keystrokes: c.keystrokes.clone(),
                    duration: c.duration.min(MAX_DURATION_SEC),
                })
                .collect();
            let ids = self.execute_commands(episode, &commands).await?;
            let terminal_output = self.incremental().await?;

            let was_pending = pending_completion;
            let observation = if result.is_task_complete {
                if pending_completion {
                    terminal_output.clone()
                } else {
                    pending_completion = true;
                    parser.completion_confirmation(&terminal_output)
                }
            } else {
                pending_completion = false;
                if feedback.contains("WARNINGS:") {
                    format!(
                        "Previous response had warnings:\n{feedback}\n\n{}",
                        limit_output_length(&terminal_output, MAX_OUTPUT_BYTES)
                    )
                } else {
                    limit_output_length(&terminal_output, MAX_OUTPUT_BYTES)
                }
            };

            // The observation belongs to the step's last call.
            for (i, id) in ids.iter().enumerate() {
                if i + 1 == ids.len() {
                    self.recorder.tool_finish(
                        id,
                        ToolCallStatus::Completed,
                        json!({"observation": observation}),
                        Some(&observation),
                    );
                } else {
                    self.recorder
                        .tool_finish(id, ToolCallStatus::Completed, json!({}), None);
                }
            }
            if result.is_task_complete {
                self.meter.before_tool_call()?;
                let id = format!("call_{episode}_task_complete");
                self.recorder.tool_start(
                    &id,
                    "other",
                    "mark_task_complete",
                    "mark_task_complete",
                    json!({}),
                );
                if commands.is_empty() {
                    self.recorder.tool_finish(
                        &id,
                        ToolCallStatus::Completed,
                        json!({"observation": observation}),
                        Some(&observation),
                    );
                } else {
                    self.recorder
                        .tool_finish(&id, ToolCallStatus::Completed, json!({}), None);
                }
                if was_pending {
                    return Ok(());
                }
            }
            prompt = observation;
        }
        Ok(())
    }

    /// Type each command and wait its duration. Returns the tool call ids,
    /// started in the trajectory and left for the caller to finish.
    async fn execute_commands(
        &mut self,
        episode: u64,
        commands: &[ParsedCommand],
    ) -> Result<Vec<String>, Stop> {
        let mut ids: Vec<String> = Vec::new();
        for (i, command) in commands.iter().enumerate() {
            if let Err(stop) = self.meter.before_tool_call() {
                for id in &ids {
                    self.recorder
                        .tool_finish(id, ToolCallStatus::Completed, json!({}), None);
                }
                return Err(stop);
            }
            let id = format!("call_{episode}_{}", i + 1);
            let title = command.keystrokes.trim_end_matches('\n').to_string();
            self.recorder.tool_start(
                &id,
                "execute",
                if title.is_empty() { "wait" } else { &title },
                "bash_command",
                json!({"keystrokes": command.keystrokes, "duration": command.duration}),
            );
            ids.push(id.clone());
            let started = std::time::Instant::now();
            let sent = self
                .meter
                .bound_wall(
                    self.session
                        .send_keys(std::slice::from_ref(&command.keystrokes)),
                )
                .await;
            match sent {
                Err(stop) => {
                    self.recorder
                        .tool_finish(&id, ToolCallStatus::Cancelled, json!({}), None);
                    return Err(stop);
                }
                Ok(Err(e)) => {
                    self.recorder.tool_finish(
                        &id,
                        ToolCallStatus::Failed,
                        json!({"error": e.to_string()}),
                        None,
                    );
                    return Err(Stop::Sandbox(e));
                }
                Ok(Ok(())) => {}
            }
            let wait = std::time::Duration::from_secs_f64(command.duration.max(0.0));
            let elapsed = started.elapsed();
            if elapsed < wait
                && let Err(stop) = self.meter.sleep(wait - elapsed).await
            {
                self.recorder
                    .tool_finish(&id, ToolCallStatus::Cancelled, json!({}), None);
                return Err(stop);
            }
        }
        Ok(ids)
    }

    fn request(&self, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            system: None,
            messages,
            tools: Vec::new(),
            max_tokens: self.config.max_output_tokens,
            temperature: self.config.temperature,
            stop: Vec::new(),
            reasoning_effort: self.config.reasoning_effort.clone(),
        }
    }

    /// One model call on `messages`, within the budgets.
    async fn call(
        &self,
        messages: Vec<Message>,
        purpose: &str,
    ) -> Result<Result<super::chat::ChatResponse, super::ModelError>, Stop> {
        self.meter.before_model_call(self.client.usage().total())?;
        let req = self.request(messages);
        self.meter
            .bound("the model call", self.client.chat(&req, purpose))
            .await
    }

    /// Harbor's `_query_llm`: the turn's call, with summarization when the
    /// context overflows and a retry when the reply hits the output limit.
    async fn query_llm(
        &mut self,
        chat: &mut Vec<Message>,
        prompt: &str,
        instruction: &str,
    ) -> Result<Reply, Stop> {
        let mut prompt = prompt.to_string();
        loop {
            let mut messages = chat.clone();
            messages.push(Message::user(prompt.clone()));
            match self.call(messages, "turn").await? {
                Ok(response) if response.finish == FinishReason::Length => {
                    let truncated = response.message.text.clone();
                    if self.config.parser == ParserKind::Xml
                        && let (Some(salvaged), _) =
                            xml_parser::salvage_truncated_response(&truncated)
                    {
                        chat.push(Message::user(prompt));
                        chat.push(Message::assistant(salvaged.clone()));
                        return Ok(Reply {
                            content: salvaged,
                            reasoning: response.message.reasoning,
                        });
                    }
                    let parsed = self.config.parser.parse(&truncated);
                    let limit = match self.config.max_output_tokens {
                        Some(n) => format!("{n} tokens"),
                        None => "the maximum output length".to_string(),
                    };
                    let mut error_msg = format!(
                        "ERROR!! NONE of the actions you just requested were performed because you exceeded {limit}. Your outputs must be less than {limit}. Re-issue this request, breaking it into chunks each of which is less than {limit}."
                    );
                    if !parsed.warning.is_empty() {
                        error_msg.push_str(&format!(
                            "\n\nParser warnings from your truncated response:\n{}",
                            parsed.warning
                        ));
                    }
                    self.recorder.notice(
                        "warning",
                        "the reply hit the output limit and was discarded",
                        None,
                    );
                    chat.push(Message::user(prompt));
                    chat.push(Message::assistant(truncated));
                    prompt = error_msg;
                }
                Ok(response) => {
                    chat.push(Message::user(prompt));
                    chat.push(Message::assistant(response.message.text.clone()));
                    return Ok(Reply {
                        content: response.message.text,
                        reasoning: response.message.reasoning,
                    });
                }
                Err(e) if e.kind == ModelErrorKind::ContextLength => {
                    if !self.config.enable_summarize {
                        return Err(Stop::Model(e));
                    }
                    return self.recover_from_overflow(chat, instruction).await;
                }
                Err(e) => return Err(Stop::Model(e)),
            }
        }
    }

    /// The context overflowed: unwind, summarize (or fall back), and ask
    /// again with the summary.
    async fn recover_from_overflow(
        &mut self,
        chat: &mut Vec<Message>,
        instruction: &str,
    ) -> Result<Reply, Stop> {
        self.unwind(chat, 4000);
        let mut summary_prompt = match self.summarize(chat, instruction).await {
            Ok(handoff) => {
                self.pending_handoff = Some(handoff.clone());
                Some(handoff)
            }
            Err(
                stop @ (Stop::Timeout(_) | Stop::Budget(_) | Stop::Idle { .. } | Stop::Sandbox(_)),
            ) => return Err(stop),
            Err(other) => {
                tracing::debug!("full summary failed: {other:?}");
                None
            }
        };
        let screen = || async {
            self.session
                .capture_pane(false)
                .await
                .map(|s| {
                    let chars: Vec<char> = s.chars().collect();
                    chars[chars.len().saturating_sub(1000)..]
                        .iter()
                        .collect::<String>()
                })
                .map_err(Stop::Sandbox)
        };
        if summary_prompt.is_none() {
            let limited = screen().await?;
            let short = format!(
                "Briefly continue this task: {instruction}\n\nCurrent state: {limited}\n\nNext steps (2-3 sentences):"
            );
            if let Ok(r) = self
                .call(vec![Message::user(short)], "short-summary")
                .await?
            {
                summary_prompt = Some(format!("{instruction}\n\nSummary: {}", r.message.text));
            }
        }
        let summary_prompt = match summary_prompt {
            Some(s) => s,
            None => format!("{instruction}\n\nCurrent state: {}", screen().await?),
        };
        let mut messages = chat.clone();
        messages.push(Message::user(summary_prompt.clone()));
        match self.call(messages, "turn").await? {
            Ok(r) => {
                chat.push(Message::user(summary_prompt));
                chat.push(Message::assistant(r.message.text.clone()));
                Ok(Reply {
                    content: r.message.text,
                    reasoning: r.message.reasoning,
                })
            }
            Err(e) => {
                tracing::error!("even the fallback call failed: {e}");
                Ok(Reply {
                    content: "Technical difficulties. Please continue with the task.".into(),
                    reasoning: None,
                })
            }
        }
    }

    /// Drop the latest user and assistant pairs until `target` tokens are free.
    fn unwind(&self, chat: &mut Vec<Message>, target: u64) {
        while chat.len() > 1 {
            let free = self.config.context_limit.saturating_sub(count_tokens(chat));
            if free >= target {
                break;
            }
            if chat.len() >= 2 {
                chat.truncate(chat.len() - 2);
            } else {
                break;
            }
        }
    }

    /// Harbor's three-step summarization. Replaces the conversation with
    /// the first prompt, the questions prompt and the questions, and
    /// returns the handoff prompt.
    async fn summarize(
        &mut self,
        chat: &mut Vec<Message>,
        instruction: &str,
    ) -> Result<String, Stop> {
        if chat.is_empty() {
            return Ok(instruction.to_string());
        }
        let summary_prompt =
            format_template(SUMMARY_PROMPT, &[("original_instruction", instruction)]);
        let mut messages = chat.clone();
        messages.push(Message::user(summary_prompt.clone()));
        let summary = self
            .call(messages, "summary")
            .await?
            .map_err(Stop::Model)?
            .message
            .text;

        let current_screen = self
            .meter
            .bound_wall(self.session.capture_pane(false))
            .await?
            .map_err(Stop::Sandbox)?;
        let question_prompt = format_template(
            QUESTION_PROMPT,
            &[
                ("original_instruction", instruction),
                ("summary", &summary),
                ("current_screen", &current_screen),
            ],
        );
        let questions = self
            .call(
                vec![Message::user(question_prompt.clone())],
                "summary-questions",
            )
            .await?
            .map_err(Stop::Model)?
            .message
            .text;

        let answer_request = format!(
            "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n{questions}"
        );
        let mut messages = chat.clone();
        messages.push(Message::user(summary_prompt));
        messages.push(Message::assistant(summary));
        messages.push(Message::user(answer_request));
        let answers = self
            .call(messages, "summary-answers")
            .await?
            .map_err(Stop::Model)?
            .message
            .text;

        let first = chat[0].clone();
        *chat = vec![
            first,
            Message::user(question_prompt),
            Message::assistant(questions),
        ];
        Ok(format!(
            "Here are the answers the other agent provided.\n\n{answers}\n\nContinue working on this task from where the previous agent left off. You can no longer ask questions. Please follow the spec to interact with the terminal."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_fill_like_python_format() {
        let p = format_template(
            JSON_TEMPLATE,
            &[("instruction", "Do {x}"), ("terminal_state", "$ ")],
        );
        assert!(p.contains("Task Description:\nDo {x}\n\nCurrent terminal state:\n$ "));
        assert!(
            p.contains("{\n  \"analysis\""),
            "double braces become single"
        );
        assert!(p.contains(r#"{"keystrokes": "", "duration": 10.0}"#));
        assert!(!p.contains("{{"));
        let t = format_template(
            TIMEOUT_TEMPLATE,
            &[
                ("command", "sleep 99"),
                ("timeout_sec", "5"),
                ("terminal_state", "x"),
            ],
        );
        assert!(t.contains("timed out after 5 seconds"));
        let x = format_template(
            XML_TEMPLATE,
            &[("instruction", "i"), ("terminal_state", "s")],
        );
        assert!(
            x.ends_with("Task Description:\ni\n\nCurrent terminal state:\ns\n"),
            "{x}"
        );
    }

    #[test]
    fn output_keeps_both_ends_within_the_limit() {
        let s = format!("{}{}", "a".repeat(8000), "b".repeat(8000));
        let out = limit_output_length(&s, 10_000);
        assert!(out.starts_with(&"a".repeat(5000)));
        assert!(out.ends_with(&"b".repeat(5000)));
        assert!(
            out.contains("[... output limited to 10000 bytes; 6000 interior bytes omitted ...]")
        );
        assert_eq!(limit_output_length("short", 10), "short");
    }

    #[test]
    fn python_repr_quotes_like_python() {
        assert_eq!(py_repr("a\nb"), "'a\\nb'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("'\""), "'\\'\"'");
    }

    #[test]
    fn mcp_servers_are_listed_after_the_instruction() {
        let info = mcp_info(&[
            json!({"name": "fs", "command": "npx", "args": ["-y", "server-fs"]}),
            json!({"name": "web", "transport": "sse", "url": "http://x/sse"}),
        ]);
        assert_eq!(
            info,
            "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n- fs: stdio transport, command: npx -y server-fs\n- web: sse transport, url: http://x/sse\n"
        );
    }
}
