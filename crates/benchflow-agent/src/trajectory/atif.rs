//! ATIF export: a BenchFlow trajectory as one Agent Trajectory Interchange
//! Format document (Harbor's format, schema `ATIF-v1.8`), so other viewers
//! and trainers can read it.
//!
//! The mapping is 0.8's (`trajectories/export_atif.py`, which wrote v1.7):
//!
//! - `user_message` becomes a `user` step;
//! - `agent_message` an `agent` step; `agent_thought`s become the next agent
//!   step's `reasoning_content` (joined by blank lines), or an agent step of
//!   their own when no agent step follows;
//! - `tool_call` an `agent` step with one tool call, whose output becomes the
//!   step's observation (a result naming the call even when it printed
//!   nothing, so every call is paired);
//! - `notice` a `system` step with its severity in `extra.notice`;
//! - `oracle` an agent step `[oracle: <command>]`.
//!
//! Events a subagent produced (Claude's attributed children) never enter
//! their parent's steps: each subagent becomes an embedded trajectory in
//! `subagent_trajectories` (`trajectory_id` `subagent:<tool call id>`),
//! referenced from the spawning tool call's observation, nested up to
//! [`MAX_SUBAGENT_DEPTH`] levels. `extra.acp_projection` accounts for every
//! attributed event.
//!
//! Beyond 0.8: steps carry `timestamp`s from the host receipts; a tool call's
//! `function_name` is the agent's own tool name when it reports one (Claude's
//! `Bash`, else the ACP kind), its `arguments` are the agent's `raw_input`
//! object, and an observation is the text the model was given when the agent
//! reports it as its raw output's `output` (OpenCode's content drops a read's
//! line numbers and its "(Showing lines …)" cut marker), else the content's
//! text, else the raw output (Codex reports command output only there). An
//! observation result's `extra` holds the command's `exit_code` and
//! `truncated: true` when the agent reports them.
//!
//! Redact the document before writing it (`Redactor::redact_json`).

use std::collections::{HashMap, VecDeque};

use serde_json::{Map, Value, json};

use super::event::{Event, MODEL_CALL, Receipt, SYSTEM_PROMPT, Text, ToolCall, iso_from_ns};
use super::export_common::str_or;

/// The ATIF schema version written.
pub const ATIF_SCHEMA_VERSION: &str = "ATIF-v1.8";
/// How deep embedded subagent trajectories nest.
pub const MAX_SUBAGENT_DEPTH: usize = 8;

/// What ATIF records about the run besides its events.
#[derive(Debug, Clone, Default)]
pub struct AtifMeta {
    /// The run's id (BenchFlow's trial name, say).
    pub session_id: Option<String>,
    /// The agent's name (`claude`); `unknown` when empty.
    pub agent_name: String,
    /// The agent's version; `unknown` when not known.
    pub agent_version: Option<String>,
    pub model: Option<String>,
    /// The prompts the agent was given. They become `user` steps only when
    /// the events record no prompt of the root agent's: BenchFlow records
    /// each prompt it sends as a `user_message` event, and 0.8, which wrote
    /// both, gave every prompt two steps.
    pub prompts: Vec<String>,
    pub total_prompt_tokens: Option<u64>,
    pub total_completion_tokens: Option<u64>,
    pub total_cached_tokens: Option<u64>,
    pub total_cost_usd: Option<f64>,
}

/// The ATIF document for `events`, or `None` when they hold no step (ATIF
/// requires at least one).
pub fn to_atif(events: &[Event], meta: &AtifMeta) -> Option<Value> {
    let agent_name = if meta.agent_name.is_empty() {
        "unknown"
    } else {
        &meta.agent_name
    };
    let agent_version = meta.agent_version.as_deref().unwrap_or("unknown");
    let plan = SubagentPlan::new(events, agent_name, agent_version);
    let recorded_prompt = events.iter().any(|event| {
        matches!(event, Event::UserMessage(text) if !text.text.is_empty())
            && parent_of(event).is_none()
    });
    let prompts: &[String] = if recorded_prompt { &[] } else { &meta.prompts };
    let steps = plan.steps(None, prompts);
    if steps.is_empty() {
        return None;
    }
    let mut agent = Map::new();
    agent.insert("name".into(), agent_name.into());
    agent.insert("version".into(), agent_version.into());
    if let Some(model) = &meta.model {
        agent.insert("model_name".into(), model.clone().into());
    }
    let mut final_metrics = Map::new();
    final_metrics.insert("total_steps".into(), steps.len().into());
    for (key, value) in [
        ("total_prompt_tokens", meta.total_prompt_tokens),
        ("total_completion_tokens", meta.total_completion_tokens),
        ("total_cached_tokens", meta.total_cached_tokens),
    ] {
        if let Some(value) = value {
            final_metrics.insert(key.into(), value.into());
        }
    }
    if let Some(cost) = meta.total_cost_usd {
        final_metrics.insert("total_cost_usd".into(), json!(cost));
    }
    let mut record = Map::new();
    record.insert("schema_version".into(), ATIF_SCHEMA_VERSION.into());
    if let Some(session_id) = meta.session_id.as_ref().filter(|s| !s.is_empty()) {
        record.insert("session_id".into(), session_id.clone().into());
    }
    record.insert("agent".into(), Value::Object(agent));
    record.insert("steps".into(), Value::Array(steps));
    record.insert("final_metrics".into(), Value::Object(final_metrics));
    if let Some(coverage) = plan.coverage(events) {
        record.insert("extra".into(), json!({ "acp_projection": coverage }));
    }
    let embedded = plan.embedded(None);
    if !embedded.is_empty() {
        record.insert("subagent_trajectories".into(), Value::Array(embedded));
    }
    Some(Value::Object(record))
}

/// The subagent an event belongs to, when the capture attributed it.
fn parent_of(event: &Event) -> Option<&str> {
    let parent = match event {
        Event::UserMessage(t) | Event::AgentMessage(t) | Event::AgentThought(t) => {
            t.parent_tool_call_id.as_deref()
        }
        Event::ToolCall(call) => call.parent_tool_call_id.as_deref(),
        Event::Other(map) => map.get("parent_tool_call_id").and_then(Value::as_str),
        _ => None,
    };
    parent.filter(|p| !p.is_empty())
}

fn subagent_trajectory_id(parent: &str) -> String {
    format!("subagent:{parent}")
}

/// A spawning call's `raw_input` field (Claude's Task tool: `prompt`,
/// `subagent_type`).
fn spawn_input<'a>(spawn: Option<&'a ToolCall>, key: &str) -> Option<&'a str> {
    spawn?
        .raw_input
        .as_ref()?
        .get(key)?
        .as_str()
        .filter(|s| !s.is_empty())
}

/// The root agent and its subagents, split by attribution (0.8's
/// `_SubagentPlan`). Scope `None` is the root.
struct SubagentPlan<'a> {
    scopes: HashMap<Option<&'a str>, Vec<&'a Event>>,
    spawns: HashMap<&'a str, &'a ToolCall>,
    children: HashMap<Option<&'a str>, Vec<&'a str>>,
    built: HashMap<&'a str, Value>,
}

impl<'a> SubagentPlan<'a> {
    fn new(events: &'a [Event], agent_name: &str, agent_version: &str) -> Self {
        let mut scopes: HashMap<Option<&str>, Vec<&Event>> = HashMap::from([(None, Vec::new())]);
        let mut scope_order: Vec<&str> = Vec::new();
        let mut spawns: HashMap<&str, &ToolCall> = HashMap::new();
        for event in events {
            let scope = parent_of(event);
            if let Some(scope) = scope
                && !scopes.contains_key(&Some(scope))
            {
                scope_order.push(scope);
            }
            scopes.entry(scope).or_default().push(event);
            if let Event::ToolCall(call) = event
                && !call.tool_call_id.is_empty()
            {
                spawns.entry(call.tool_call_id.as_str()).or_insert(call);
            }
        }

        // Breadth-first from the root, so cyclic attribution cannot recurse.
        let mut children: HashMap<Option<&str>, Vec<&str>> = HashMap::new();
        let mut depth: HashMap<Option<&str>, usize> = HashMap::from([(None, 0)]);
        let mut order: Vec<&str> = Vec::new();
        let mut queue: VecDeque<Option<&str>> = VecDeque::from([None]);
        while let Some(scope) = queue.pop_front() {
            let mut found: Vec<&str> = scopes[&scope]
                .iter()
                .filter_map(|event| match event {
                    Event::ToolCall(call)
                        if scopes.contains_key(&Some(call.tool_call_id.as_str()))
                            && spawns
                                .get(call.tool_call_id.as_str())
                                .is_some_and(|s| std::ptr::eq(*s, &**call)) =>
                    {
                        Some(call.tool_call_id.as_str())
                    }
                    _ => None,
                })
                .collect();
            if scope.is_none() {
                // Subagents whose spawning call was never captured hang,
                // unreferenced, under the root.
                found.extend(
                    scope_order
                        .iter()
                        .filter(|child| !spawns.contains_key(*child)),
                );
            }
            for child in found {
                if depth.contains_key(&Some(child)) || depth[&scope] >= MAX_SUBAGENT_DEPTH {
                    continue;
                }
                depth.insert(Some(child), depth[&scope] + 1);
                children.entry(scope).or_default().push(child);
                order.push(child);
                queue.push_back(Some(child));
            }
        }

        let mut plan = Self {
            scopes,
            spawns,
            children,
            built: HashMap::new(),
        };
        // Deepest first, so a parent only references children with steps.
        for scope in order.into_iter().rev() {
            plan.build(scope, agent_name, agent_version);
        }
        plan
    }

    fn build(&mut self, scope: &'a str, agent_name: &str, agent_version: &str) {
        let spawn = self.spawns.get(scope).copied();
        let prompt = spawn_input(spawn, "prompt");
        let prompts: Vec<String> = prompt.map(str::to_string).into_iter().collect();
        let steps = self.steps(Some(scope), &prompts);
        if steps.is_empty() {
            return;
        }
        let mut agent = json!({"name": agent_name, "version": agent_version});
        if let Some(kind) = spawn_input(spawn, "subagent_type") {
            agent["extra"] = json!({"subagent_type": kind});
        }
        let mut extra = json!({"parent_tool_call_id": scope});
        if spawn.is_none() {
            extra["parent_tool_call_captured"] = json!(false);
        } else if prompt.is_some() {
            extra["prompt_source"] = json!("parent_tool_call.raw_input.prompt");
        }
        let mut trajectory = Map::new();
        trajectory.insert("schema_version".into(), ATIF_SCHEMA_VERSION.into());
        trajectory.insert("trajectory_id".into(), subagent_trajectory_id(scope).into());
        trajectory.insert("agent".into(), agent);
        let total_steps = steps.len();
        trajectory.insert("steps".into(), Value::Array(steps));
        trajectory.insert(
            "final_metrics".into(),
            json!({ "total_steps": total_steps }),
        );
        trajectory.insert("extra".into(), extra);
        let embedded = self.embedded(Some(scope));
        if !embedded.is_empty() {
            trajectory.insert("subagent_trajectories".into(), Value::Array(embedded));
        }
        let trajectory = Value::Object(trajectory);
        self.built.insert(scope, trajectory);
    }

    fn built_children(&self, scope: Option<&str>) -> Vec<&'a str> {
        self.children
            .get(&scope)
            .map(|c| {
                c.iter()
                    .copied()
                    .filter(|c| self.built.contains_key(c))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Built subagent trajectories hanging directly under `scope`.
    fn embedded(&self, scope: Option<&str>) -> Vec<Value> {
        self.built_children(scope)
            .into_iter()
            .map(|c| self.built[c].clone())
            .collect()
    }

    fn steps(&self, scope: Option<&str>, prompts: &[String]) -> Vec<Value> {
        let refs: HashMap<&str, String> = self
            .built_children(scope)
            .into_iter()
            .filter(|child| self.spawns.contains_key(child))
            .map(|child| (child, subagent_trajectory_id(child)))
            .collect();
        scope_steps(&self.scopes[&scope], prompts, &refs)
    }

    /// Accounting of every attributed event against the embedded tree.
    fn coverage(&self, events: &[Event]) -> Option<Value> {
        let attributed = events.iter().filter(|e| parent_of(e).is_some()).count();
        if attributed == 0 {
            return None;
        }
        let embedded_events: usize = self
            .built
            .keys()
            .map(|s| self.scopes[&Some(*s)].len())
            .sum();
        let unlinked = self
            .built
            .keys()
            .filter(|s| !self.spawns.contains_key(*s))
            .count();
        Some(json!({
            "scope": "root_with_embedded_subagents",
            "attributed_child_events": attributed,
            "embedded_subagent_trajectories": self.built.len(),
            "unlinked_subagent_trajectories": unlinked,
            "excluded_attributed_child_events": attributed - embedded_events,
            "max_subagent_depth": MAX_SUBAGENT_DEPTH,
            "source_artifact": "trajectory/acp_trajectory.jsonl",
            "child_transcript_completeness": "unknown",
            "outcome_and_usage_scope": "rollout_not_partitioned_by_agent",
        }))
    }
}

/// Thoughts waiting for the next agent step.
#[derive(Default)]
struct Thoughts {
    texts: Vec<String>,
    first: Option<Receipt>,
}

impl Thoughts {
    fn push(&mut self, text: &Text) {
        self.first = self.first.or(text.receipt);
        self.texts.push(text.text.clone());
    }

    fn take(&mut self) -> Option<(String, Option<Receipt>)> {
        if self.texts.is_empty() {
            return None;
        }
        let joined = self.texts.join("\n\n");
        self.texts.clear();
        Some((joined, self.first.take()))
    }
}

fn is_kind(map: &Map<String, Value>, kind: &str) -> bool {
    map.get("type").and_then(Value::as_str) == Some(kind)
}

/// A recorded event's receipt, for an event kept as a map.
fn map_receipt(map: &Map<String, Value>) -> Option<Receipt> {
    map.get("receipt").and_then(Receipt::from_json)
}

/// A tool call's observation result. Every call has one, empty output
/// included, so each tool call is paired.
fn observation_result(
    call: &ToolCall,
    call_id: &str,
    refs: &HashMap<&str, String>,
) -> Map<String, Value> {
    let mut result = Map::new();
    result.insert("source_call_id".into(), call_id.into());
    let output = observation_text(call);
    if !output.is_empty() {
        result.insert("content".into(), output.into());
    }
    if let Some(extra) = observation_extra(call) {
        result.insert("extra".into(), Value::Object(extra));
    }
    if let Some(id) = refs.get(call.tool_call_id.as_str()) {
        result.insert(
            "subagent_trajectory_ref".into(),
            json!([{ "trajectory_id": id }]),
        );
    }
    result
}

/// One agent's events as ATIF steps.
fn scope_steps(events: &[&Event], prompts: &[String], refs: &HashMap<&str, String>) -> Vec<Value> {
    if events
        .iter()
        .any(|e| matches!(e, Event::Other(map) if is_kind(map, MODEL_CALL)))
    {
        return turn_steps(events, prompts, refs);
    }
    let mut steps: Vec<Value> = Vec::new();
    let mut thoughts = Thoughts::default();
    let push = |steps: &mut Vec<Value>,
                source: &str,
                receipt: Option<Receipt>,
                body: Map<String, Value>| {
        let mut step = Map::new();
        step.insert("step_id".into(), (steps.len() + 1).into());
        if let Some(receipt) = receipt {
            step.insert(
                "timestamp".into(),
                iso_from_ns(receipt.first_observed_ns).into(),
            );
        }
        step.insert("source".into(), source.into());
        step.extend(body);
        steps.push(Value::Object(step));
    };
    let message =
        |text: &str| Map::from_iter([("message".to_string(), Value::String(text.to_string()))]);
    let flush = |steps: &mut Vec<Value>, thoughts: &mut Thoughts| {
        if let Some((reasoning, receipt)) = thoughts.take() {
            let mut body = message("");
            body.insert("reasoning_content".into(), reasoning.into());
            push(steps, "agent", receipt, body);
        }
    };

    for prompt in prompts.iter().filter(|p| !p.is_empty()) {
        push(&mut steps, "user", None, message(prompt));
    }
    for event in events {
        match event {
            Event::UserMessage(text) if !text.text.is_empty() => {
                flush(&mut steps, &mut thoughts);
                push(&mut steps, "user", text.receipt, message(&text.text));
            }
            Event::AgentThought(text) if !text.text.is_empty() => thoughts.push(text),
            Event::AgentMessage(text) if !text.text.is_empty() => {
                let mut body = message(&text.text);
                if let Some((reasoning, _)) = thoughts.take() {
                    body.insert("reasoning_content".into(), reasoning.into());
                }
                push(&mut steps, "agent", text.receipt, body);
            }
            Event::ToolCall(call) => {
                let mut body = message("");
                let call_id = if call.tool_call_id.is_empty() {
                    format!("call_{}", steps.len() + 1)
                } else {
                    call.tool_call_id.clone()
                };
                body.insert("tool_calls".into(), json!([atif_tool_call(call, &call_id)]));
                if let Some((reasoning, _)) = thoughts.take() {
                    body.insert("reasoning_content".into(), reasoning.into());
                }
                // Every call gets its result, empty output included, so
                // each tool call is paired (chat formats need a reply).
                let result = observation_result(call, &call_id, refs);
                body.insert("observation".into(), json!({ "results": [result] }));
                push(&mut steps, "agent", call.receipt, body);
            }
            Event::Oracle(oracle) => {
                let command = if oracle.command.is_empty() {
                    "oracle"
                } else {
                    &oracle.command
                };
                push(
                    &mut steps,
                    "agent",
                    None,
                    message(&format!("[oracle: {command}]")),
                );
            }
            Event::Other(map) if is_kind(map, SYSTEM_PROMPT) => {
                flush(&mut steps, &mut thoughts);
                push(
                    &mut steps,
                    "system",
                    map_receipt(map),
                    message(&str_or(map, "text", "")),
                );
            }
            // An oracle event with keys 0.8 did not write (the Rust oracle
            // adds `timed_out`) is kept as `Other`; 0.8's walker read any
            // `oracle` dict.
            Event::Other(map) if map.get("type").and_then(Value::as_str) == Some("oracle") => {
                let command = str_or(map, "command", "oracle");
                push(
                    &mut steps,
                    "agent",
                    None,
                    message(&format!("[oracle: {command}]")),
                );
            }
            Event::Notice(notice) if !notice.title.trim().is_empty() => {
                flush(&mut steps, &mut thoughts);
                let title = notice.title.trim();
                let text = match notice
                    .description
                    .as_deref()
                    .map(str::trim)
                    .filter(|d| !d.is_empty())
                {
                    Some(description) => format!("{title}\n\n{description}"),
                    None => title.to_string(),
                };
                let mut body = message(&text);
                let severity = if notice.severity.is_empty() {
                    "info"
                } else {
                    &notice.severity
                };
                body.insert("extra".into(), json!({"notice": {"severity": severity}}));
                push(&mut steps, "system", notice.receipt, body);
            }
            _ => {}
        }
    }
    flush(&mut steps, &mut thoughts);
    steps
}

/// One model turn of a native loop, as it builds up.
#[derive(Default)]
struct Turn {
    receipt: Option<Receipt>,
    /// The `model_call` record that opened it.
    call: Option<Map<String, Value>>,
    text: Vec<String>,
    reasoning: Vec<String>,
    tool_calls: Vec<Value>,
    results: Vec<Value>,
    /// Notices recorded while the turn was open, written after its step so
    /// they do not split it.
    notices: Vec<(Option<Receipt>, Map<String, Value>)>,
}

impl Turn {
    fn is_empty(&self) -> bool {
        self.call.is_none()
            && self.text.is_empty()
            && self.reasoning.is_empty()
            && self.tool_calls.is_empty()
    }

    fn note(&mut self, receipt: Option<Receipt>) {
        self.receipt = self.receipt.or(receipt);
    }

    /// The turn as one agent step's fields.
    fn body(self) -> (Option<Receipt>, Map<String, Value>) {
        let mut body = Map::new();
        body.insert("message".into(), self.text.join("\n").into());
        if !self.reasoning.is_empty() {
            body.insert(
                "reasoning_content".into(),
                self.reasoning.join("\n\n").into(),
            );
        }
        if !self.tool_calls.is_empty() {
            body.insert("tool_calls".into(), Value::Array(self.tool_calls));
        }
        if !self.results.is_empty() {
            body.insert("observation".into(), json!({ "results": self.results }));
        }
        if let Some(call) = &self.call {
            let calls = call.get("calls").and_then(Value::as_u64).unwrap_or(1);
            body.insert("llm_call_count".into(), calls.into());
            let usage = call.get("usage").and_then(Value::as_object);
            if let (Some(usage), true) = (usage, calls > 0) {
                let n = |k: &str| usage.get(k).and_then(Value::as_u64);
                let mut metrics = Map::new();
                for (key, from) in [
                    ("prompt_tokens", "prompt_tokens"),
                    ("completion_tokens", "completion_tokens"),
                    ("cached_tokens", "cached_tokens"),
                ] {
                    if let Some(v) = n(from) {
                        metrics.insert(key.into(), v.into());
                    }
                }
                if let Some(cost) = usage.get("cost_usd").filter(|c| c.is_number()) {
                    metrics.insert("cost_usd".into(), cost.clone());
                }
                let mut extra = Map::new();
                for key in ["reasoning_tokens", "cache_write_tokens"] {
                    if let Some(v) = n(key).filter(|v| *v > 0) {
                        extra.insert(key.into(), v.into());
                    }
                }
                if !extra.is_empty() {
                    metrics.insert("extra".into(), Value::Object(extra));
                }
                body.insert("metrics".into(), Value::Object(metrics));
            }
            if let Some(finish) = call.get("finish_reason").and_then(Value::as_str) {
                body.insert("extra".into(), json!({ "finish_reason": finish }));
            }
        }
        (self.receipt, body)
    }
}

/// A native loop's events (it records a `model_call` before each turn's
/// events) as one agent step per model call: the step carries the call's
/// usage as `metrics`, `llm_call_count`, the turn's reasoning and text, its
/// tool calls, and their results. Calls that served no step of their own
/// (a retry, a summary) count with the turn they served.
fn turn_steps(events: &[&Event], prompts: &[String], refs: &HashMap<&str, String>) -> Vec<Value> {
    let mut steps: Vec<Value> = Vec::new();
    let push = |steps: &mut Vec<Value>,
                source: &str,
                receipt: Option<Receipt>,
                body: Map<String, Value>| {
        let mut step = Map::new();
        step.insert("step_id".into(), (steps.len() + 1).into());
        if let Some(receipt) = receipt {
            step.insert(
                "timestamp".into(),
                iso_from_ns(receipt.first_observed_ns).into(),
            );
        }
        step.insert("source".into(), source.into());
        step.extend(body);
        steps.push(Value::Object(step));
    };
    let message =
        |text: &str| Map::from_iter([("message".to_string(), Value::String(text.to_string()))]);
    let close = |steps: &mut Vec<Value>, turn: &mut Turn| {
        let mut turn = std::mem::take(turn);
        let notices = std::mem::take(&mut turn.notices);
        if !turn.is_empty() {
            let (receipt, body) = turn.body();
            push(steps, "agent", receipt, body);
        }
        for (receipt, body) in notices {
            push(steps, "system", receipt, body);
        }
    };
    for prompt in prompts.iter().filter(|p| !p.is_empty()) {
        push(&mut steps, "user", None, message(prompt));
    }
    let mut turn = Turn::default();
    for event in events {
        match event {
            Event::Other(map) if is_kind(map, MODEL_CALL) => {
                close(&mut steps, &mut turn);
                turn.call = Some(map.clone());
                turn.note(map_receipt(map));
            }
            Event::Other(map) if is_kind(map, SYSTEM_PROMPT) => {
                close(&mut steps, &mut turn);
                push(
                    &mut steps,
                    "system",
                    map_receipt(map),
                    message(&str_or(map, "text", "")),
                );
            }
            Event::UserMessage(text) if !text.text.is_empty() => {
                close(&mut steps, &mut turn);
                push(&mut steps, "user", text.receipt, message(&text.text));
            }
            Event::AgentThought(text) if !text.text.is_empty() => {
                turn.note(text.receipt);
                turn.reasoning.push(text.text.clone());
            }
            Event::AgentMessage(text) if !text.text.is_empty() => {
                turn.note(text.receipt);
                turn.text.push(text.text.clone());
            }
            Event::ToolCall(call) => {
                turn.note(call.receipt);
                let call_id = if call.tool_call_id.is_empty() {
                    format!("call_{}_{}", steps.len() + 1, turn.tool_calls.len() + 1)
                } else {
                    call.tool_call_id.clone()
                };
                turn.tool_calls.push(atif_tool_call(call, &call_id));
                turn.results
                    .push(Value::Object(observation_result(call, &call_id, refs)));
            }
            Event::Notice(notice) if !notice.title.trim().is_empty() => {
                let title = notice.title.trim();
                let text = match notice
                    .description
                    .as_deref()
                    .map(str::trim)
                    .filter(|d| !d.is_empty())
                {
                    Some(description) => format!("{title}\n\n{description}"),
                    None => title.to_string(),
                };
                let mut body = message(&text);
                let severity = if notice.severity.is_empty() {
                    "info"
                } else {
                    &notice.severity
                };
                body.insert("extra".into(), json!({"notice": {"severity": severity}}));
                if turn.is_empty() {
                    push(&mut steps, "system", notice.receipt, body);
                } else {
                    turn.notices.push((notice.receipt, body));
                }
            }
            _ => {}
        }
    }
    close(&mut steps, &mut turn);
    steps
}

/// The ATIF tool call for an ACP tool call.
fn atif_tool_call(call: &ToolCall, call_id: &str) -> Value {
    let kind = if call.kind.is_empty() {
        "tool"
    } else {
        &call.kind
    };
    let named = call.tool_name.as_deref().filter(|n| !n.is_empty());
    let mut extra = Map::new();
    for (key, value) in [
        ("title", call.title.as_str()),
        ("status", call.status.as_str()),
    ] {
        if !value.is_empty() {
            extra.insert(key.into(), value.into());
        }
    }
    if named.is_some() {
        extra.insert("kind".into(), kind.into());
    }
    let arguments = match &call.raw_input {
        Some(Value::Object(input)) => Value::Object(input.clone()),
        Some(other) => {
            extra.insert("raw_input".into(), other.clone());
            json!({})
        }
        None => json!({}),
    };
    let mut tool_call = json!({
        "tool_call_id": call_id,
        "function_name": named.unwrap_or(kind),
        "arguments": arguments,
    });
    if !extra.is_empty() {
        tool_call["extra"] = Value::Object(extra);
    }
    tool_call
}

/// A tool call's output as text: the raw output's `output` text (what the
/// model was given, when the agent reports it), else its content's text
/// blocks, else its raw output.
fn observation_text(call: &ToolCall) -> String {
    if let Some(Value::Object(raw)) = &call.raw_output
        && let Some(Value::String(output)) = raw.get("output")
        && !output.is_empty()
    {
        return output.clone();
    }
    let text = content_blocks_to_text(&call.content);
    if !text.is_empty() {
        return text;
    }
    match call.raw_output() {
        Some(Value::String(s)) => s,
        Some(Value::Object(map)) => map
            .get("formatted_output")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

/// What the agent's raw output says about the result besides its text: the
/// command's exit code (OpenCode's `metadata.exit`, Codex's `exit_code`) and
/// whether the agent cut the output it gave the model (`metadata.truncated`).
fn observation_extra(call: &ToolCall) -> Option<Map<String, Value>> {
    let Some(Value::Object(raw)) = call.raw_output() else {
        return None;
    };
    let metadata = raw.get("metadata").and_then(Value::as_object);
    let mut extra = Map::new();
    let exit = metadata
        .and_then(|m| m.get("exit"))
        .or_else(|| raw.get("exit_code"))
        .filter(|v| v.is_i64() || v.is_u64());
    if let Some(exit) = exit {
        extra.insert("exit_code".into(), exit.clone());
    }
    if metadata.and_then(|m| m.get("truncated")) == Some(&Value::Bool(true)) {
        extra.insert("truncated".into(), Value::Bool(true));
    }
    (!extra.is_empty()).then_some(extra)
}

/// 0.8's `content_blocks_to_text`: the text of flat (`{"text"}`) and nested
/// (`{"content": {"text"}}`) blocks, one per line.
fn content_blocks_to_text(content: &[Value]) -> String {
    let mut parts = Vec::new();
    for item in content {
        let Some(item) = item.as_object() else {
            continue;
        };
        let inner = match item.get("content") {
            Some(Value::Object(inner)) => inner.get("text"),
            other => other,
        };
        let text = item
            .get("text")
            .filter(|t| truthy(t))
            .or(inner.filter(|t| truthy(t)));
        match text {
            Some(Value::String(s)) => parts.push(s.clone()),
            Some(other) => parts.push(other.to_string()),
            None => {}
        }
    }
    parts.join("\n")
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
    }
}

#[cfg(test)]
mod tests;
