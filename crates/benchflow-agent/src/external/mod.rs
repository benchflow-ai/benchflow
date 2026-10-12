//! The external agent (`external`): an outside service, such as an RL
//! trainer, decides every step over HTTP.
//!
//! This is Karotte's external agent (Preference Model, MIT License;
//! `agents/external.py`, `backend_source.py`, `message_loop.py`) with
//! BenchFlow's sandbox in place of Karotte's MCP tool server. BenchFlow
//! streams the run's transcript to the backend in Karotte's event format
//! (`task_started`, `message_added`, `tool_call_started`, ...), long-polls
//! the backend for the model's next message, runs the tool calls it holds
//! in the sandbox as the agent user (the builtin loop's tools: `bash`,
//! `view`, `replace`, `write`, `submit`), and streams the results back, so
//! a backend written for Karotte can drive BenchFlow's environments.
//!
//! The turn rules are Karotte's message loop: a message without tool calls
//! ends the step, unless it is empty and looks cut off, which earns the
//! nudge (three in a row at most); a `content_filter` finish ends the step;
//! the last tool result of a turn ends with `Time remaining: N seconds`
//! when the run has a wall clock. Differences: the tool definitions go to
//! the backend in the `task_pre_hook_completed` metadata (`tools`), since
//! there is no MCP server to list them; and grading happens once, after the
//! last step, so the `scoring` and `task_completed` events follow the last
//! `step_completed` ([`Session::finish`]).

pub mod backend;

use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::native::RunContext;
use crate::native::budget::{Meter, Stop};
use crate::native::builtin::tools::{self, ToolContext, ToolOutcome};
use crate::native::builtin::{EMPTY_TURN_NUDGE, MAX_CONSECUTIVE_EMPTY_TURNS, system_prompt};
use crate::native::chat::ToolSpec;
use crate::native::openai::sanitized_arguments;
use crate::native::recorder::Recorder;
use crate::trajectory::ToolCallStatus;
pub use backend::{BackendClient, BackendError, BackendMessage, BackendOptions};

/// Why an external run stopped early.
#[derive(Debug)]
pub enum ExternalStop {
    /// A budget, the clock, the idle timeout, or the sandbox.
    Loop(Stop),
    /// The backend failed or sent something BenchFlow cannot read.
    Backend(BackendError),
}

impl From<Stop> for ExternalStop {
    fn from(s: Stop) -> Self {
        ExternalStop::Loop(s)
    }
}

impl From<BackendError> for ExternalStop {
    fn from(e: BackendError) -> Self {
        ExternalStop::Backend(e)
    }
}

/// How the run is described to the backend.
#[derive(Debug, Clone)]
pub struct ExternalConfig {
    /// The task's name (`task_id` in `task_started`).
    pub task_id: String,
    /// The model the backend serves, when the run named one.
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    /// Default and longest per-call `bash` timeout.
    pub bash_timeout: Duration,
    /// More metadata for `task_pre_hook_completed` (trial, workdir, user).
    pub metadata: Map<String, Value>,
}

/// The final state reported to the backend after grading.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// Karotte's run status: `passed`, `failed`, or `error`.
    pub status: &'static str,
    pub score: Option<f64>,
    /// Goes into the `scoring` event's metadata (rewards, the problem).
    pub metadata: Map<String, Value>,
}

/// One run's conversation with the backend: every event goes to the
/// backend at the next `seq` and is kept for `agent/backend_transcript.json`.
#[derive(Debug)]
pub struct Session {
    client: BackendClient,
    events: Vec<Value>,
}

impl Session {
    pub fn new(client: BackendClient) -> Self {
        Self {
            client,
            events: Vec::new(),
        }
    }

    pub fn client(&self) -> &BackendClient {
        &self.client
    }

    /// Karotte's transcript: `{"run_id", "events"}`.
    pub fn transcript(&self) -> Value {
        json!({"run_id": self.client.run_id(), "events": self.events})
    }

    /// Send `event` (with its `timestamp`, added when missing) at the next
    /// position.
    pub async fn emit(&mut self, mut event: Value) -> Result<(), BackendError> {
        if let Some(obj) = event.as_object_mut()
            && !obj.contains_key("timestamp")
        {
            obj.insert("timestamp".into(), backend::now_iso().into());
        }
        let seq = self.events.len() as u64;
        self.events.push(event);
        self.client
            .append_transcript(&self.events[seq as usize], seq)
            .await
    }

    /// Open the transcript (create it unless the backend has one, as
    /// Karotte does on reconnect) and send the run's opening events.
    pub async fn start(
        &mut self,
        config: &ExternalConfig,
        n_steps: usize,
        tools: &[ToolSpec],
        system: &str,
    ) -> Result<(), BackendError> {
        if self.client.transcript_length().await?.is_none() {
            self.client.create_transcript().await?;
        }
        self.emit(json!({
            "type": "task_started",
            "run_id": self.client.run_id(),
            "task_id": config.task_id,
            "model": config.model,
            "n_steps": n_steps,
            "reasoning_effort": config.reasoning_effort,
            "use_hints": null,
        }))
        .await?;
        self.client
            .update_run_state(&run_state("running", None))
            .await?;
        let mut metadata = config.metadata.clone();
        metadata.insert("tools".into(), tool_params(tools));
        self.emit(json!({"type": "task_pre_hook_completed", "metadata": metadata}))
            .await?;
        self.emit(json!({"type": "message_added",
                         "message": {"role": "system", "content": system}}))
            .await
    }

    /// Report the graded outcome: `scoring` (when there is a score), then
    /// `task_completed` and the final run state.
    pub async fn finish(&mut self, outcome: &Outcome) -> Result<(), BackendError> {
        if let Some(score) = outcome.score {
            self.emit(json!({"type": "scoring", "scoring": {
                "score": score, "metadata": outcome.metadata, "continue_task": false}}))
                .await?;
        }
        self.emit(json!({"type": "task_completed", "status": outcome.status}))
            .await?;
        self.client
            .update_run_state(&run_state(outcome.status, outcome.score))
            .await
    }

    /// Report a run that stopped on an error: an `error` event, then
    /// `task_completed` with status `error`.
    pub async fn fail(&mut self, kind: &str, message: &str) -> Result<(), BackendError> {
        self.emit(json!({"type": "error", "exception_type": kind,
                         "message": message, "traceback": null}))
            .await?;
        self.finish(&Outcome {
            status: "error",
            score: None,
            metadata: Map::new(),
        })
        .await
    }
}

/// Karotte's `update_run_state` body. Token counts stay null: the backend
/// produced the messages and knows them.
fn run_state(status: &str, score: Option<f64>) -> Value {
    json!({"status": status, "score": score, "input_tokens": null, "output_tokens": null,
           "cache_read_tokens": null, "cache_write_tokens": null})
}

/// The tools in OpenAI's (and LiteLLM's) function-tool form.
pub fn tool_params(tools: &[ToolSpec]) -> Value {
    tools
        .iter()
        .map(|t| {
            json!({"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}})
        })
        .collect()
}

/// Run every prompt of `ctx` as one step each, recording into `recorder`.
/// The session must be started; the caller reports the outcome.
pub async fn run(
    session: &mut Session,
    recorder: &Recorder,
    ctx: &RunContext<'_>,
    config: &ExternalConfig,
) -> Result<(), ExternalStop> {
    let mut meter = Meter::new(ctx.limits);
    for (step, prompt) in ctx.prompts.iter().enumerate() {
        session
            .emit(json!({"type": "step_started", "step": step}))
            .await?;
        recorder.user(prompt);
        session
            .emit(json!({"type": "message_added",
                         "message": {"role": "user", "content": prompt}}))
            .await?;
        let result = step_loop(session, recorder, ctx, config, &mut meter).await;
        recorder.finish();
        result?;
        session
            .emit(json!({"type": "step_completed", "step": step}))
            .await?;
    }
    Ok(())
}

async fn step_loop(
    session: &mut Session,
    recorder: &Recorder,
    ctx: &RunContext<'_>,
    config: &ExternalConfig,
    meter: &mut Meter,
) -> Result<(), ExternalStop> {
    let mut empty_turns = 0;
    loop {
        meter.before_model_call(0)?;
        let client = session.client().clone();
        let message = meter.bound("the backend", client.next_message()).await??;
        if let Some(r) = &message.reasoning {
            recorder.thought(r);
        }
        recorder.message(&message.text);
        session.emit(message.event.clone()).await?;
        let finish = message.finish_reason.as_deref();
        if finish == Some("content_filter") {
            return Ok(());
        }
        if message.tool_calls.is_empty() {
            let truncated = finish == Some("length")
                || message
                    .reasoning
                    .as_ref()
                    .is_some_and(|r| !r.trim().is_empty());
            if !message.text.trim().is_empty() || !truncated {
                return Ok(());
            }
            empty_turns += 1;
            if empty_turns > MAX_CONSECUTIVE_EMPTY_TURNS {
                return Err(Stop::Harness(format!(
                    "the backend sent neither text nor tool calls {empty_turns} turns in a row"
                ))
                .into());
            }
            recorder.user(EMPTY_TURN_NUDGE);
            session
                .emit(json!({"type": "message_added",
                             "message": {"role": "user", "content": EMPTY_TURN_NUDGE}}))
                .await?;
            continue;
        }
        empty_turns = 0;
        // The turn's events are held back so the time note can join the
        // last tool result before any of them is sent, as Karotte does.
        let mut events: Vec<Value> = Vec::new();
        let mut submitted = false;
        // The last recorded result, so the time note can be recorded with it.
        let mut last_result: Option<(String, ToolOutcome)> = None;
        for call in &message.tool_calls {
            if submitted {
                let text = "Not run: the turn ended with submit.";
                events.push(started(call));
                events.push(completed(&call.id, text, true));
                events.push(tool_message(&call.id, text));
                continue;
            }
            if let Err(stop) = meter.before_tool_call() {
                flush(session, events).await?;
                return Err(stop.into());
            }
            events.push(started(call));
            let outcome = match run_call(recorder, ctx, config, meter, call).await {
                Ok(o) => o,
                Err(stop) => {
                    flush(session, events).await?;
                    return Err(stop.into());
                }
            };
            events.push(completed(&call.id, &outcome.text, outcome.is_error));
            events.push(tool_message(&call.id, &outcome.text));
            submitted = outcome.ends_turn;
            last_result = Some((call.id.clone(), outcome));
        }
        if !submitted
            && let Some(remaining) = meter.remaining()
            && let Some(last) = events.last_mut()
        {
            let note = format!("Time remaining: {} seconds", remaining.as_secs());
            append_text(last, &note);
            // Record the result as the backend's model reads it, note included.
            if let Some((id, outcome)) = &last_result {
                let status = if outcome.is_error {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                recorder.tool_finish(
                    id,
                    status,
                    outcome.result.clone(),
                    Some(&format!("{}\n{note}", outcome.text)),
                );
            }
        }
        flush(session, events).await?;
        if submitted {
            return Ok(());
        }
    }
}

async fn flush(session: &mut Session, events: Vec<Value>) -> Result<(), BackendError> {
    for event in events {
        session.emit(event).await?;
    }
    Ok(())
}

/// Run one tool call in the sandbox and record it.
async fn run_call(
    recorder: &Recorder,
    ctx: &RunContext<'_>,
    config: &ExternalConfig,
    meter: &Meter,
    call: &crate::native::chat::ToolCallRequest,
) -> Result<ToolOutcome, Stop> {
    let parsed: Option<Map<String, Value>> = if call.arguments.trim().is_empty() {
        Some(Map::new())
    } else {
        serde_json::from_str::<Value>(&call.arguments)
            .ok()
            .and_then(|v| v.as_object().cloned())
    };
    let args = parsed.clone().unwrap_or_default();
    let (kind, title) = tools::describe(&call.name, &args);
    let raw_input = match &parsed {
        Some(a) => Value::Object(a.clone()),
        None => json!({"_raw": call.arguments}),
    };
    recorder.tool_start(&call.id, kind, &title, &call.name, raw_input);
    let outcome = if parsed.is_none() {
        ToolOutcome {
            result: json!({"error": "Invalid JSON in tool arguments"}),
            text: "Error: Invalid JSON in tool arguments".into(),
            is_error: true,
            ends_turn: false,
        }
    } else {
        let bash_timeout = match meter.remaining() {
            Some(r) => config.bash_timeout.min(r.max(Duration::from_secs(1))),
            None => config.bash_timeout,
        };
        let tool_ctx = ToolContext {
            sandbox: ctx.sandbox,
            user: ctx.user,
            workdir: ctx.workdir,
            env: &ctx.env,
            bash_timeout,
        };
        match meter
            .bound_wall(tools::run(&tool_ctx, &call.name, &args))
            .await
        {
            Err(stop) => {
                recorder.tool_finish(
                    &call.id,
                    ToolCallStatus::Cancelled,
                    json!({"error": "stopped"}),
                    None,
                );
                return Err(stop);
            }
            Ok(Err(e)) => {
                recorder.tool_finish(
                    &call.id,
                    ToolCallStatus::Failed,
                    json!({"error": e.to_string()}),
                    None,
                );
                return Err(Stop::Sandbox(e));
            }
            Ok(Ok(o)) => o,
        }
    };
    let status = if outcome.is_error {
        ToolCallStatus::Failed
    } else {
        ToolCallStatus::Completed
    };
    recorder.tool_finish(
        &call.id,
        status,
        outcome.result.clone(),
        Some(&outcome.text),
    );
    Ok(outcome)
}

/// `tool_call_started`, with the arguments sanitized as Karotte does when
/// they are not valid JSON.
fn started(call: &crate::native::chat::ToolCallRequest) -> Value {
    json!({"type": "tool_call_started", "tool_call": {
        "id": call.id, "type": "function",
        "function": {"name": call.name, "arguments": sanitized_arguments(&call.arguments)}}})
}

/// `tool_call_completed` with an MCP `CallToolResult`.
fn completed(id: &str, text: &str, is_error: bool) -> Value {
    json!({"type": "tool_call_completed", "tool_call_id": id, "result": {
        "content": [{"type": "text", "text": text}],
        "structuredContent": {"result": text},
        "isError": is_error}})
}

/// The tool message the model reads.
fn tool_message(id: &str, text: &str) -> Value {
    json!({"type": "message_added", "message": {
        "role": "tool", "content": [{"type": "text", "text": text}], "tool_call_id": id}})
}

/// Add a text part to a `message_added` event's content.
fn append_text(event: &mut Value, text: &str) {
    if let Some(parts) = event
        .pointer_mut("/message/content")
        .and_then(Value::as_array_mut)
    {
        parts.push(json!({"type": "text", "text": text}));
    }
}

/// The system prompt sent as the first message.
pub fn system(workdir: &str) -> String {
    system_prompt(workdir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_events_have_karottes_shapes() {
        let call = crate::native::chat::ToolCallRequest {
            id: "c1".into(),
            name: "bash".into(),
            arguments: "{oops".into(),
        };
        let s = started(&call);
        assert_eq!(
            s["tool_call"]["function"]["arguments"],
            crate::native::openai::MALFORMED_ARGUMENTS
        );
        let c = completed("c1", "ok", false);
        assert_eq!(c["result"]["content"][0]["text"], "ok");
        assert_eq!(c["result"]["isError"], false);
        let mut m = tool_message("c1", "ok");
        append_text(&mut m, "Time remaining: 5 seconds");
        assert_eq!(
            m["message"]["content"][1]["text"],
            "Time remaining: 5 seconds"
        );
        assert_eq!(m["message"]["tool_call_id"], "c1");
    }
}
