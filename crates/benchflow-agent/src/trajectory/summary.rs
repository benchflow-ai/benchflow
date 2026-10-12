//! The trajectory counts `result.json` carries, as 0.8 computes them.

use indexmap::IndexMap;
use serde::Serialize;

use super::event::Event;
use super::skill;

/// 0.8's `trajectory_summary` (Harbor's step counts): key names and order
/// match `result.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrajectorySummary {
    pub steps: usize,
    pub tool_call_steps: usize,
    pub user_message_steps: usize,
    pub agent_message_steps: usize,
    pub agent_thought_steps: usize,
    pub other_steps: usize,
    /// Events per type, in the order each type first appears.
    pub event_type_counts: IndexMap<String, usize>,
    /// Tool calls per final status, in first-seen order.
    pub tool_call_status_counts: IndexMap<String, usize>,
    /// The run ended before the agent finished (a timeout, a crash).
    pub partial_trajectory: bool,
    /// Where the events came from (`acp`, `oracle`, `nop`).
    pub trajectory_source: Option<String>,
}

/// Summarize events the way 0.8's `trajectory_summary_from_events` does.
pub fn summarize<'a, I>(
    events: I,
    partial_trajectory: bool,
    trajectory_source: Option<&str>,
) -> TrajectorySummary
where
    I: IntoIterator<Item = &'a Event>,
{
    let mut event_type_counts: IndexMap<String, usize> = IndexMap::new();
    let mut tool_call_status_counts: IndexMap<String, usize> = IndexMap::new();
    for event in events {
        // A native loop's model calls and system prompt are not steps.
        if matches!(
            event.kind(),
            super::event::MODEL_CALL | super::event::SYSTEM_PROMPT
        ) {
            continue;
        }
        *event_type_counts
            .entry(event.kind().to_string())
            .or_default() += 1;
        let status = match event {
            Event::ToolCall(call) => Some(call.status.as_str().to_string()),
            Event::Other(map) if event.kind() == "tool_call" => Some(
                map.get("status")
                    .and_then(serde_json::Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("unknown")
                    .to_string(),
            ),
            _ => None,
        };
        if let Some(status) = status {
            *tool_call_status_counts.entry(status).or_default() += 1;
        }
    }
    let count = |kind: &str| event_type_counts.get(kind).copied().unwrap_or(0);
    let steps = event_type_counts.values().sum();
    let tool_call_steps = count("tool_call");
    let known = count("user_message") + count("agent_message") + count("agent_thought");
    TrajectorySummary {
        steps,
        tool_call_steps,
        user_message_steps: count("user_message"),
        agent_message_steps: count("agent_message"),
        agent_thought_steps: count("agent_thought"),
        other_steps: steps - tool_call_steps - known,
        event_type_counts,
        tool_call_status_counts,
        partial_trajectory,
        trajectory_source: trajectory_source.map(str::to_string),
    }
}

/// `result.json`'s `n_tool_calls`: every tool call the agent made.
pub fn count_tool_calls<'a, I>(events: I) -> usize
where
    I: IntoIterator<Item = &'a Event>,
{
    events
        .into_iter()
        .filter(|e| e.kind() == "tool_call")
        .count()
}

/// `result.json`'s `n_skill_invocations`.
pub fn count_skill_invocations<'a, I>(events: I) -> usize
where
    I: IntoIterator<Item = &'a Event>,
{
    events
        .into_iter()
        .filter(|e| skill::is_skill_event(e))
        .count()
}
