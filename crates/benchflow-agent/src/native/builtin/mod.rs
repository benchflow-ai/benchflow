//! The builtin tool loop (agent `builtin`), after Karotte's builtin agent
//! (Preference Model, MIT License; `agents/message_loop.py`).
//!
//! Each turn the model gets the conversation and the tools; every tool call
//! it makes runs in the sandbox, one at a time, and its result goes back as
//! a tool message. A turn without tool calls ends the prompt, as does
//! `submit`. A turn with neither text nor tool calls that looks cut off
//! (it hit the output limit, or it only reasoned) gets Karotte's nudge, at
//! most three times in a row. When the run has a wall-clock budget, the
//! last tool result of each turn ends with `Time remaining: N seconds`, as
//! Karotte's `inject_time_remaining_counter` does.

pub mod tools;

use std::time::Duration;

use serde_json::{Map, Value, json};

use super::RunContext;
use super::budget::{Meter, Stop};
use super::chat::{ChatRequest, FinishReason, Message};
use super::client::ModelClient;
use super::recorder::Recorder;
use crate::trajectory::ToolCallStatus;
use tools::ToolContext;

/// Karotte's limit on consecutive empty turns.
pub const MAX_CONSECUTIVE_EMPTY_TURNS: u32 = 3;

/// Turns in a row whose every tool call had arguments that are not a JSON
/// object, before the loop stops. Each such call is answered with an error
/// so the model can correct itself; without a cap a model that keeps doing
/// it is asked again until the agent's time runs out (one faulty endpoint
/// made 1,528 calls in 134 s). The stop is the model's failure and the
/// trial is still graded.
pub const MAX_CONSECUTIVE_INVALID_TURNS: u32 = 3;

/// Karotte's nudge after an empty turn.
pub const EMPTY_TURN_NUDGE: &str = "Your last turn produced no message or tool call. Pick up where you left off. Do not apologize or recap. Break the remaining work into smaller pieces if necessary.";

/// How the builtin loop runs.
#[derive(Debug, Clone)]
pub struct BuiltinConfig {
    /// Default and longest per-call `bash` timeout.
    pub bash_timeout: Duration,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub reasoning_effort: Option<String>,
}

impl Default for BuiltinConfig {
    fn default() -> Self {
        Self {
            bash_timeout: Duration::from_secs(600),
            max_output_tokens: None,
            temperature: None,
            reasoning_effort: None,
        }
    }
}

/// The system prompt. Karotte takes it from the task; BenchFlow tasks have
/// none, so this one only says where the agent is and how to finish.
pub fn system_prompt(workdir: &str) -> String {
    format!(
        "You are an autonomous agent solving a task in a Linux sandbox. Your working folder is {workdir}. \
         Use the tools to inspect files, run commands, and make changes. Nobody will answer questions, \
         so act on your own judgment until the task is done, then call `submit`."
    )
}

/// Run the loop on every prompt of `ctx`, recording into `recorder`.
pub async fn run(
    client: &ModelClient,
    recorder: &Recorder,
    ctx: &RunContext<'_>,
    config: &BuiltinConfig,
) -> Stop {
    let mut meter = Meter::new(ctx.limits);
    let specs = tools::specs(config.bash_timeout);
    let mut messages: Vec<Message> = Vec::new();
    let system = system_prompt(ctx.workdir);
    recorder.system_prompt(&system);
    for prompt in &ctx.prompts {
        recorder.user(prompt);
        messages.push(Message::user(prompt.clone()));
        if let Err(stop) = prompt_loop(
            client,
            recorder,
            ctx,
            config,
            &specs,
            &system,
            &mut messages,
            &mut meter,
        )
        .await
        {
            recorder.finish();
            return stop;
        }
    }
    recorder.finish();
    Stop::Finished
}

#[allow(clippy::too_many_arguments)]
async fn prompt_loop(
    client: &ModelClient,
    recorder: &Recorder,
    ctx: &RunContext<'_>,
    config: &BuiltinConfig,
    specs: &[super::chat::ToolSpec],
    system: &str,
    messages: &mut Vec<Message>,
    meter: &mut Meter,
) -> Result<(), Stop> {
    let mut empty_turns = 0;
    let mut invalid_turns = 0;
    let mut invalid_cut = 0;
    loop {
        meter.before_model_call(client.usage().total())?;
        let req = ChatRequest {
            system: Some(system.to_string()),
            messages: messages.clone(),
            tools: specs.to_vec(),
            max_tokens: config.max_output_tokens,
            temperature: config.temperature,
            stop: Vec::new(),
            reasoning_effort: config.reasoning_effort.clone(),
        };
        let response = meter
            .bound("the model call", client.chat(&req, "turn"))
            .await?
            .map_err(Stop::Model)?;
        recorder.usage(&client.usage());
        recorder.model_call(
            &client.usage(),
            client.calls(),
            Some(response.finish.as_str()),
            response.model.as_deref(),
        );
        if let Some(r) = &response.message.reasoning {
            recorder.thought(r);
        }
        recorder.message(&response.message.text);
        messages.push(Message::Assistant(response.message.clone()));
        if response.finish == FinishReason::ContentFilter {
            return Err(Stop::Refused);
        }
        let calls = response.message.tool_calls.clone();
        if calls.is_empty() {
            let truncated = response.finish == FinishReason::Length
                || response
                    .message
                    .reasoning
                    .as_ref()
                    .is_some_and(|r| !r.trim().is_empty());
            if !response.message.text.trim().is_empty() || !truncated {
                return Ok(());
            }
            empty_turns += 1;
            if empty_turns > MAX_CONSECUTIVE_EMPTY_TURNS {
                return Err(Stop::Harness(format!(
                    "the model produced neither text nor tool calls {empty_turns} turns in a row"
                )));
            }
            recorder.user(EMPTY_TURN_NUDGE);
            messages.push(Message::user(EMPTY_TURN_NUDGE));
            continue;
        }
        empty_turns = 0;
        let mut submitted = false;
        let mut all_invalid = true;
        // The last recorded result, so the time note can be recorded with it.
        let mut last_result: Option<(String, ToolCallStatus, Value)> = None;
        for call in &calls {
            if submitted {
                // The turn ended at `submit`; later calls are answered so the
                // history stays valid, and never run.
                messages.push(Message::Tool {
                    call_id: call.id.clone(),
                    content: "Not run: the turn ended with submit.".into(),
                });
                continue;
            }
            meter.before_tool_call()?;
            let parsed: Option<Map<String, Value>> = if call.arguments.trim().is_empty() {
                Some(Map::new())
            } else {
                serde_json::from_str::<Value>(&call.arguments)
                    .ok()
                    .and_then(|v| v.as_object().cloned())
            };
            if parsed.is_some() {
                all_invalid = false;
            }
            let args = parsed.clone().unwrap_or_default();
            let (kind, title) = tools::describe(&call.name, &args);
            let raw_input = match &parsed {
                Some(a) => Value::Object(a.clone()),
                None => json!({"_raw": call.arguments}),
            };
            recorder.tool_start(&call.id, kind, &title, &call.name, raw_input);
            let outcome = if parsed.is_none() {
                tools::ToolOutcome {
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
                // Tools carry their own timeouts; only the wall clock bounds them here.
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
            last_result = Some((call.id.clone(), status, outcome.result.clone()));
            messages.push(Message::Tool {
                call_id: call.id.clone(),
                content: outcome.text,
            });
            submitted = outcome.ends_turn;
        }
        if submitted {
            return Ok(());
        }
        if all_invalid {
            invalid_turns += 1;
            if response.finish == FinishReason::Length {
                invalid_cut += 1;
            }
            if invalid_turns > MAX_CONSECUTIVE_INVALID_TURNS {
                let cut = if invalid_cut > 0 {
                    format!(" ({invalid_cut} of them cut off at the output-token limit)")
                } else {
                    String::new()
                };
                return Err(Stop::ModelLoop {
                    reason: format!(
                        "the model's tool calls had arguments that are not JSON {invalid_turns} turns in a row{cut}"
                    ),
                    turns: invalid_turns,
                });
            }
        } else {
            invalid_turns = 0;
            invalid_cut = 0;
        }
        if let Some(remaining) = meter.remaining()
            && let Some(Message::Tool { call_id, content }) = messages.last_mut()
        {
            content.push_str(&format!(
                "\n\nTime remaining: {} seconds",
                remaining.as_secs()
            ));
            // Record the result as the model will read it, note included.
            if let Some((id, status, result)) = &last_result
                && id == call_id
            {
                recorder.tool_finish(id, *status, result.clone(), Some(content));
            }
        }
    }
}
