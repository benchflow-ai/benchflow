//! computer-1: the screenshot, model, one action loop, ported from Harbor's
//! computer-1 agent (Apache-2.0, `harbor/agents/computer_1/`; its generic
//! JSON flavor, which works with any vision model).
//!
//! The loop starts a desktop in the sandbox ([`desktop`]): a virtual
//! display, a window manager, and a browser, which the task's image must
//! provide. Each turn the model gets a screenshot (as an image part) and
//! answers with one strict JSON object, `{"analysis", "plan", "action"}`;
//! the action runs through `xdotool` as the agent user, and the next turn
//! carries what happened and a fresh screenshot. A terminal action (`done`,
//! `answer`, `terminate`) must come twice in a row, as in Harbor: the first
//! gets Harbor's "Are you sure" question, the second writes the answer to
//! `/logs/agent/final_answer.txt` for the verifier. The run also ends at the
//! turn budget ([`ComputerConfig::max_turns`]) or a BenchFlow budget.
//!
//! The prompt (`templates/computer-1-json.txt`), the parser's rules, the
//! observation texts, and the wait-streak note are Harbor's. Differences:
//! only the last [`ComputerConfig::keep_screenshots`] screenshots stay in the
//! conversation (Harbor strips older ones only when the context runs low);
//! there is no `bash` action and no summarization; and every screenshot is
//! kept in the trial folder (`agent/screenshots/NNN.png`).

pub mod desktop;

use std::path::PathBuf;
use std::time::Duration;

use base64::Engine;
use serde_json::{Map, Value, json};

use super::RunContext;
use super::budget::{Meter, Stop};
use super::chat::{ChatRequest, FinishReason, Image, Message};
use super::client::ModelClient;
use super::recorder::Recorder;
use crate::trajectory::ToolCallStatus;
use desktop::{Action, Desktop, StartError};

/// Harbor's prompt template, unchanged.
pub const PROMPT_TEMPLATE: &str = include_str!("templates/computer-1-json.txt");
/// Harbor's timeout template, unchanged.
pub const TIMEOUT_TEMPLATE: &str = include_str!("templates/timeout.txt");
/// Harbor's cap on one observation's text.
pub const MAX_OBSERVATION_BYTES: usize = 10_000;
/// Where the verifier reads the answer (Harbor's path).
pub const FINAL_ANSWER: &str = "/logs/agent/final_answer.txt";

/// How computer-1 runs.
#[derive(Debug, Clone)]
pub struct ComputerConfig {
    /// The display's size in pixels (Harbor's default, 1024x900).
    pub width: u32,
    pub height: u32,
    /// Most model turns per prompt.
    pub max_turns: u32,
    /// Screenshots kept in the conversation, newest first.
    pub keep_screenshots: usize,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub reasoning_effort: Option<String>,
}

impl Default for ComputerConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 900,
            max_turns: 50,
            keep_screenshots: 3,
            max_output_tokens: None,
            temperature: None,
            reasoning_effort: None,
        }
    }
}

/// Fill a Python `str.format` template: `{name}` placeholders, `{{` and
/// `}}` for literal braces.
pub fn format_template(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(i) = rest.find(['{', '}']) {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        if tail.starts_with("{{") || tail.starts_with("}}") {
            out.push_str(&tail[..1]);
            rest = &tail[2..];
            continue;
        }
        if tail.starts_with('{')
            && let Some(end) = tail.find('}')
            && let Some((_, value)) = values.iter().find(|(k, _)| *k == &tail[1..end])
        {
            out.push_str(value);
            rest = &tail[end + 1..];
            continue;
        }
        out.push_str(&tail[..1]);
        rest = &tail[1..];
    }
    out.push_str(rest);
    out
}

/// The first prompt for `instruction` (bash disabled, as Harbor's default).
pub fn prompt(instruction: &str, width: u32, height: u32) -> String {
    let (w, h) = (width.to_string(), height.to_string());
    format_template(
        PROMPT_TEMPLATE,
        &[
            ("instruction", instruction),
            ("desktop_width", &w),
            ("desktop_height", &h),
            ("bash_type_option", ""),
            ("bash_fields", ""),
            ("bash_rule", ""),
        ],
    )
}

/// One parsed reply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Parsed {
    pub action: Option<Action>,
    pub error: String,
    pub warnings: Vec<String>,
    pub analysis: String,
    pub plan: String,
}

impl Parsed {
    /// Harbor's feedback: `ERROR: ...` and `WARNINGS: - ...`.
    pub fn feedback(&self) -> String {
        let warnings = if self.warnings.is_empty() {
            String::new()
        } else {
            format!("- {}", self.warnings.join("\n- "))
        };
        match (self.error.is_empty(), warnings.is_empty()) {
            (false, false) => format!("ERROR: {}\nWARNINGS: {warnings}", self.error),
            (false, true) => format!("ERROR: {}", self.error),
            (true, false) => format!("WARNINGS: {warnings}"),
            (true, true) => String::new(),
        }
    }
}

/// The first balanced top-level JSON object in `response`, with Harbor's
/// warnings about text around it.
fn extract_json_object(response: &str) -> (Option<&str>, Vec<String>) {
    let (mut start, mut depth, mut in_string, mut escape) = (None, 0i32, false, false);
    for (i, c) in response.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match c {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0
                    && let Some(s) = start
                {
                    let mut warnings = Vec::new();
                    if !response[..s].trim().is_empty() {
                        warnings.push("Extra text detected before JSON object".into());
                    }
                    if !response[i + 1..].trim().is_empty() {
                        warnings.push("Extra text detected after JSON object".into());
                    }
                    return (Some(&response[s..=i]), warnings);
                }
            }
            _ => {}
        }
    }
    (None, vec!["No valid JSON object found".into()])
}

const ALLOWED: &[&str] = &[
    "click",
    "double_click",
    "triple_click",
    "right_click",
    "mouse_down",
    "mouse_up",
    "mouse_move",
    "type",
    "key",
    "keypress",
    "hold_key",
    "scroll",
    "drag",
    "zoom",
    "navigate",
    "wait",
    "done",
    "answer",
    "terminate",
];

fn int(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Parse a reply the way Harbor's generic computer-1 parser does.
pub fn parse(response: &str) -> Parsed {
    let (json_text, mut warnings) = extract_json_object(response);
    let fail = |error: String, warnings: Vec<String>, analysis: String, plan: String| Parsed {
        action: None,
        error,
        warnings,
        analysis,
        plan,
    };
    let Some(json_text) = json_text else {
        return fail(
            "No valid JSON found in response".into(),
            warnings,
            String::new(),
            String::new(),
        );
    };
    let data: Value = match serde_json::from_str(json_text) {
        Ok(v) => v,
        Err(e) => {
            let preview: String = json_text.chars().take(100).collect();
            return fail(
                format!("Invalid JSON: {e} | Content preview: {preview:?}..."),
                warnings,
                String::new(),
                String::new(),
            );
        }
    };
    let Some(data) = data.as_object() else {
        return fail(
            "Response must be a JSON object".into(),
            warnings,
            String::new(),
            String::new(),
        );
    };
    let text_field = |key: &str, warnings: &mut Vec<String>| match data.get(key) {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => {
            warnings.push(format!("Field '{key}' should be a string"));
            String::new()
        }
    };
    let analysis = text_field("analysis", &mut warnings);
    let plan = text_field("plan", &mut warnings);
    let Some(action) = data.get("action") else {
        return fail(
            "Missing required field: action".into(),
            warnings,
            analysis,
            plan,
        );
    };
    let Some(a) = action.as_object() else {
        return fail(
            "Field 'action' must be an object".into(),
            warnings,
            analysis,
            plan,
        );
    };
    let Some(kind) = a
        .get("type")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    else {
        return fail(
            "Action 'type' is missing or not a string".into(),
            warnings,
            analysis,
            plan,
        );
    };
    if !ALLOWED.contains(&kind) {
        warnings.push(format!("Unknown action type: '{kind}'"));
    }
    let keys = match a.get("keys") {
        None | Some(Value::Null) => None,
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => Some(
            items
                .iter()
                .filter_map(|k| k.as_str().map(str::to_string))
                .collect(),
        ),
        Some(_) => {
            warnings.push("Action 'keys' must be a list of strings; ignoring".into());
            None
        }
    };
    let modifier = match a.get("modifier") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => {
            warnings.push("Action 'modifier' must be a string; ignoring".into());
            None
        }
    };
    let zoom_region = match a.get("zoom_region") {
        None | Some(Value::Null) => None,
        Some(Value::Array(v)) if v.len() == 4 => {
            let ints: Vec<i64> = v.iter().filter_map(|x| int(Some(x))).collect();
            match ints.as_slice() {
                [a, b, c, d] => Some([*a, *b, *c, *d]),
                _ => None,
            }
        }
        Some(_) => None,
    };
    if a.get("zoom_region").is_some_and(|z| !z.is_null()) && zoom_region.is_none() {
        warnings.push("Action 'zoom_region' must be a 4-element list of integers; ignoring".into());
    }
    let string = |key: &str| a.get(key).and_then(Value::as_str).map(str::to_string);
    Parsed {
        action: Some(Action {
            kind: kind.to_string(),
            x: int(a.get("x")),
            y: int(a.get("y")),
            end_x: int(a.get("end_x")),
            end_y: int(a.get("end_y")),
            text: string("text"),
            keys,
            url: string("url"),
            scroll_x: int(a.get("scroll_x")),
            scroll_y: int(a.get("scroll_y")),
            button: string("button"),
            modifier,
            duration: a.get("duration").and_then(|d| match d {
                Value::Number(n) => n.as_f64(),
                Value::String(s) => s.trim().parse().ok(),
                _ => None,
            }),
            zoom_region,
            result: string("result"),
        }),
        error: String::new(),
        warnings,
        analysis,
        plan,
    }
}

/// Cut `output` to [`MAX_OBSERVATION_BYTES`], keeping both ends (Harbor's
/// `_limit_output_length`).
pub fn limit_output(output: &str) -> String {
    let max = MAX_OBSERVATION_BYTES;
    if output.len() <= max {
        return output.to_string();
    }
    let half = max / 2;
    let mut head_end = half;
    while !output.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = output.len() - half;
    while !output.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let omitted = tail_start - head_end;
    format!(
        "{}\n[... output limited to {max} bytes; {omitted} interior bytes omitted ...]\n{}",
        &output[..head_end],
        &output[tail_start..]
    )
}

/// A short title for an action, for the trajectory.
pub fn title(a: &Action) -> String {
    match (a.kind.as_str(), a.x, a.y) {
        ("type", _, _) => format!("type {:?}", a.text.as_deref().unwrap_or_default()),
        ("navigate", _, _) => format!("navigate {}", a.url.as_deref().unwrap_or_default()),
        ("key" | "keypress" | "hold_key", _, _) => {
            format!(
                "{} {}",
                a.kind,
                a.keys.clone().unwrap_or_default().join("+")
            )
        }
        (k, Some(x), Some(y)) => format!("{k} ({x}, {y})"),
        (k, _, _) => k.to_string(),
    }
}

/// The conversation as sent: only the newest `keep` screenshots stay.
fn with_recent_screenshots(history: &[Message], keep: usize) -> Vec<Message> {
    let mut seen = 0;
    let mut out: Vec<Message> = history
        .iter()
        .rev()
        .map(|m| match m {
            Message::UserImages { text, images } if !images.is_empty() => {
                seen += 1;
                if seen <= keep {
                    m.clone()
                } else {
                    Message::User(format!("{text}\n\n(screenshot omitted)"))
                }
            }
            other => other.clone(),
        })
        .collect();
    out.reverse();
    out
}

/// Screenshots on the host, numbered in order.
struct Shots {
    dir: Option<PathBuf>,
    n: u32,
}

impl Shots {
    /// Keep `png` as the next screenshot; its path relative to the trial's
    /// `agent/` folder, when kept.
    fn keep(&mut self, png: &[u8]) -> Option<String> {
        let dir = self.dir.as_ref()?.join("screenshots");
        let name = format!("{:03}.png", self.n);
        self.n += 1;
        std::fs::create_dir_all(&dir).ok()?;
        std::fs::write(dir.join(&name), png).ok()?;
        Some(format!("screenshots/{name}"))
    }
}

fn png_image(png: &[u8]) -> Image {
    Image {
        media_type: "image/png".into(),
        base64: base64::engine::general_purpose::STANDARD.encode(png),
    }
}

/// Run computer-1 on every prompt of `ctx`, recording into `recorder`.
pub async fn run(
    client: &ModelClient,
    recorder: &Recorder,
    ctx: &RunContext<'_>,
    config: &ComputerConfig,
) -> Stop {
    let mut meter = Meter::new(ctx.limits);
    let mut desktop = Desktop::new(ctx.sandbox, ctx.user, &ctx.env, config.width, config.height);
    let started = meter.bound_wall(desktop.start()).await;
    match started {
        Err(stop) => return stop,
        Ok(Err(StartError::NoDisplay { missing })) => {
            return Stop::Unsupported {
                why: format!(
                    "computer-1 needs a desktop in the task's image, and this image lacks {missing}"
                ),
                next: format!(
                    "use a task whose image installs {}, a window manager, and a browser \
                     (crates/benchflow-run/tests/fixtures/tasks/hello-desktop is one), or pick an agent that works in a terminal",
                    desktop::REQUIRED
                ),
            };
        }
        Ok(Err(StartError::NotReady(log))) => {
            return Stop::Setup(format!(
                "the desktop did not come up in the sandbox: {}",
                limit_output(log.trim())
            ));
        }
        Ok(Err(StartError::Sandbox(e))) => return Stop::Sandbox(e),
        Ok(Ok(None)) => recorder.notice(
            "warning",
            "the image has no browser (chromium, google-chrome, or firefox); navigate actions will not work",
            None,
        ),
        Ok(Ok(Some(_))) => {}
    }
    let mut shots = Shots {
        dir: ctx.agent_dir.clone(),
        n: 0,
    };
    let mut history: Vec<Message> = Vec::new();
    for instruction in &ctx.prompts {
        let text = prompt(instruction, config.width, config.height);
        recorder.user(&text);
        let png = match meter.bound_wall(desktop.screenshot()).await {
            Err(stop) => return stop,
            Ok(Err(why)) => return Stop::Harness(why),
            Ok(Ok(png)) => png,
        };
        shots.keep(&png);
        history.push(Message::UserImages {
            text,
            images: vec![png_image(&png)],
        });
        if let Err(stop) = prompt_loop(
            client,
            recorder,
            ctx,
            config,
            &mut desktop,
            &mut shots,
            &mut history,
            &mut meter,
        )
        .await
        {
            recorder.finish();
            return stop;
        }
        recorder.finish();
    }
    Stop::Finished
}

#[allow(clippy::too_many_arguments)]
async fn prompt_loop(
    client: &ModelClient,
    recorder: &Recorder,
    ctx: &RunContext<'_>,
    config: &ComputerConfig,
    desktop: &mut Desktop<'_>,
    shots: &mut Shots,
    history: &mut Vec<Message>,
    meter: &mut Meter,
) -> Result<(), Stop> {
    let mut pending = false;
    let mut waits = 0u32;
    for turn in 1..=config.max_turns {
        meter.before_model_call(client.usage().total())?;
        if !desktop.alive().await {
            return Err(Stop::Harness(
                "the virtual display in the sandbox is gone".into(),
            ));
        }
        let req = ChatRequest {
            system: None,
            messages: with_recent_screenshots(history, config.keep_screenshots),
            tools: Vec::new(),
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
        if let Some(r) = &response.message.reasoning {
            recorder.thought(r);
        }
        history.push(Message::assistant(response.message.text.clone()));
        if response.finish == FinishReason::ContentFilter {
            return Err(Stop::Refused);
        }
        let parsed = parse(&response.message.text);
        let feedback = parsed.feedback();
        if !parsed.analysis.is_empty() || !parsed.plan.is_empty() {
            recorder.message(&format!("{}\n\nPlan: {}", parsed.analysis, parsed.plan));
        } else {
            recorder.message(&response.message.text);
        }
        let Some(action) = parsed.action else {
            let next = format!(
                "Previous response had parsing errors:\n{feedback}\n\nPlease fix these issues and provide a proper JSON response."
            );
            recorder.user(&next);
            history.push(Message::user(next));
            continue;
        };
        meter.before_tool_call()?;
        let id = format!("computer-{turn}");
        let raw_input = action_json(&action);
        let kind = if action.is_terminal() {
            "other"
        } else {
            "execute"
        };
        recorder.tool_start(&id, kind, &title(&action), "computer", raw_input);
        let outcome = match meter.bound_wall(desktop.act(&action)).await {
            Err(stop) => {
                recorder.tool_finish(
                    &id,
                    ToolCallStatus::Cancelled,
                    json!({"error": "stopped"}),
                    None,
                );
                return Err(stop);
            }
            Ok(Err(e)) => {
                recorder.tool_finish(
                    &id,
                    ToolCallStatus::Failed,
                    json!({"error": e.to_string()}),
                    None,
                );
                return Err(Stop::Sandbox(e));
            }
            Ok(Ok(o)) => o,
        };
        let complete = action.is_terminal();
        let action_text = if complete {
            format!("Terminal action committed: {}", action.kind)
        } else if outcome.text.contains("timed out") {
            format_template(
                TIMEOUT_TEMPLATE,
                &[("timeout_sec", "60"), ("action", action.kind.as_str())],
            )
        } else {
            outcome.text.clone()
        };
        // Harbor's `_build_observation`.
        let mut observation = if complete {
            if pending {
                action_text.clone()
            } else {
                format!(
                    "Current state:\n{action_text}\n\nAre you sure you want to mark the task as complete? \
                     This will trigger your solution to be graded and you won't be able to make any further corrections. \
                     If so, confirm again with the same final answer."
                )
            }
        } else if feedback.contains("WARNINGS:") {
            format!("Previous response had warnings:\n{feedback}\n\n{action_text}")
        } else {
            limit_output(&action_text)
        };
        // Harbor's wait streak.
        if complete {
            waits = 0;
        } else if action.kind == "wait" {
            waits += 1;
            if waits > 1 {
                observation.push_str(&format!(
                    "\n\nYou have now waited {waits} turns in a row without taking action."
                ));
            }
        } else {
            waits = 0;
        }
        let status = if outcome.status == "ok" || outcome.status == "done" {
            ToolCallStatus::Completed
        } else {
            ToolCallStatus::Failed
        };
        if complete && pending {
            let answer = action
                .result
                .clone()
                .or(action.text.clone())
                .unwrap_or_else(|| response.message.text.clone());
            let written = write_answer(ctx, &answer).await;
            recorder.tool_finish(
                &id,
                status,
                json!({"status": outcome.status, "final_answer": answer, "written": written}),
                Some(&observation),
            );
            return Ok(());
        }
        pending = complete;
        let png = match meter.bound_wall(desktop.screenshot()).await {
            Err(stop) => return Err(stop),
            Ok(Err(why)) => return Err(Stop::Harness(why)),
            Ok(Ok(png)) => png,
        };
        let path = shots.keep(&png);
        recorder.tool_finish(
            &id,
            status,
            json!({"status": outcome.status, "text": outcome.text, "screenshot": path}),
            Some(if observation.is_empty() {
                "(screenshot)"
            } else {
                &observation
            }),
        );
        history.push(Message::UserImages {
            text: observation,
            images: vec![png_image(&png)],
        });
    }
    recorder.notice(
        "warning",
        &format!(
            "stopped after {} turns (computer-1's turn budget)",
            config.max_turns
        ),
        Some("the model never confirmed a done action; no final answer was written"),
    );
    Ok(())
}

/// The action as the model wrote it, for the trajectory.
fn action_json(a: &Action) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), a.kind.clone().into());
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            m.insert(k.into(), v);
        }
    };
    put("x", a.x.map(Into::into));
    put("y", a.y.map(Into::into));
    put("end_x", a.end_x.map(Into::into));
    put("end_y", a.end_y.map(Into::into));
    put("text", a.text.clone().map(Into::into));
    put("keys", a.keys.clone().map(Into::into));
    put("url", a.url.clone().map(Into::into));
    put("scroll_x", a.scroll_x.map(Into::into));
    put("scroll_y", a.scroll_y.map(Into::into));
    put("button", a.button.clone().map(Into::into));
    put("modifier", a.modifier.clone().map(Into::into));
    put("duration", a.duration.map(Into::into));
    put("zoom_region", a.zoom_region.map(|z| json!(z)));
    put("result", a.result.clone().map(Into::into));
    Value::Object(m)
}

/// Write the final answer where the verifier reads it, as the agent user
/// (Harbor's `_write_final_answer`). Whether it was written.
async fn write_answer(ctx: &RunContext<'_>, answer: &str) -> bool {
    let encoded = base64::engine::general_purpose::STANDARD.encode(answer.as_bytes());
    let command = format!(
        "mkdir -p /logs/agent && printf '%s' {} | base64 -d > {FINAL_ANSWER}",
        crate::shell::quote(&encoded)
    );
    ctx.sandbox
        .exec(
            benchflow_sandbox::Exec::new(command)
                .user(ctx.user.to_string())
                .timeout(Duration::from_secs(30)),
        )
        .await
        .is_ok_and(|o| o.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_is_harbors_template_filled_in() {
        let p = prompt("Open the page.", 1024, 900);
        assert!(p.contains("Task instructions:\nOpen the page."));
        assert!(p.contains("The display is\n1024x900 pixels."));
        assert!(p.contains("\"action\": {\n"));
        assert!(p.contains("wait, done, answer>"));
        assert!(!p.contains("{bash"));
        assert!(!p.contains("{{"));
    }

    #[test]
    fn replies_parse_with_harbors_errors_and_warnings() {
        let p = parse(
            "Sure.\n{\"analysis\": \"a\", \"plan\": \"p\", \"action\": {\"type\": \"click\", \"x\": 5, \"y\": \"7\"}}",
        );
        let a = p.action.clone().unwrap();
        assert_eq!((a.kind.as_str(), a.x, a.y), ("click", Some(5), Some(7)));
        assert_eq!(
            p.feedback(),
            "WARNINGS: - Extra text detected before JSON object"
        );
        let p = parse("no json here");
        assert!(p.action.is_none());
        assert_eq!(
            p.feedback(),
            "ERROR: No valid JSON found in response\nWARNINGS: - No valid JSON object found"
        );
        let p = parse("{\"analysis\": \"a\"}");
        assert_eq!(p.error, "Missing required field: action");
        let p = parse("{\"action\": {\"type\": \"fly\", \"keys\": \"ctrl\"}}");
        assert_eq!(p.action.unwrap().kind, "fly");
        assert_eq!(
            p.warnings,
            [
                "Unknown action type: 'fly'",
                "Action 'keys' must be a list of strings; ignoring"
            ]
        );
        let p = parse("{\"action\": {\"type\": \"done\", \"result\": \"CODE 42 }\"}}");
        let a = p.action.unwrap();
        assert!(a.is_terminal());
        assert_eq!(a.result.as_deref(), Some("CODE 42 }"));
    }

    #[test]
    fn templates_keep_literal_braces_and_unknown_names() {
        assert_eq!(
            format_template("{{\"a\": {x}}} {y}", &[("x", "1")]),
            "{\"a\": 1} {y}"
        );
    }

    #[test]
    fn only_the_newest_screenshots_are_sent() {
        let img = || Image {
            media_type: "image/png".into(),
            base64: "AA==".into(),
        };
        let history: Vec<Message> = (0..5)
            .map(|i| Message::UserImages {
                text: format!("turn {i}"),
                images: vec![img()],
            })
            .collect();
        let sent = with_recent_screenshots(&history, 3);
        let with_images = sent
            .iter()
            .filter(|m| matches!(m, Message::UserImages { .. }))
            .count();
        assert_eq!(with_images, 3);
        assert_eq!(
            sent[0],
            Message::User("turn 0\n\n(screenshot omitted)".into())
        );
    }

    #[test]
    fn long_observations_keep_both_ends() {
        let long = "x".repeat(MAX_OBSERVATION_BYTES + 100);
        let cut = limit_output(&long);
        assert!(
            cut.contains("100 interior bytes omitted") || cut.contains("interior bytes omitted")
        );
        assert!(cut.len() < long.len());
    }
}
