use serde_json::{Value, json};

use super::*;
use crate::trajectory::event::{Notice, Oracle, Text, ToolCallStatus};
use crate::trajectory::tests::{fixture, replay};

fn meta() -> AtifMeta {
    AtifMeta {
        session_id: Some("trial-1".into()),
        agent_name: "claude".into(),
        ..AtifMeta::default()
    }
}

/// Drop what this export adds on purpose (timestamps, tool-call details, the
/// raw-output observation fallback) so the rest compares with 0.8's v1.7.
fn as_0_8(value: &Value) -> Value {
    let mut value = value.clone();
    normalize(&mut value);
    value
}

fn normalize(value: &mut Value) {
    let Some(map) = value.as_object_mut() else {
        return;
    };
    map.remove("schema_version");
    if let Some(Value::Array(steps)) = map.get_mut("steps") {
        for step in steps {
            let step = step.as_object_mut().unwrap();
            step.remove("timestamp");
            if let Some(Value::Array(calls)) = step.get_mut("tool_calls") {
                for call in calls {
                    *call = json!({"tool_call_id": call["tool_call_id"]});
                }
            }
            let mut keep_observation = false;
            if let Some(results) = step
                .get_mut("observation")
                .and_then(|o| o.get_mut("results"))
            {
                for result in results.as_array_mut().unwrap() {
                    let result = result.as_object_mut().unwrap();
                    result.remove("content");
                    keep_observation |= result.contains_key("subagent_trajectory_ref");
                }
            }
            if !keep_observation {
                step.remove("observation");
            }
        }
    }
    if let Some(Value::Array(subagents)) = map.get_mut("subagent_trajectories") {
        subagents.iter_mut().for_each(normalize);
    }
}

#[test]
fn the_document_has_0_8s_steps_and_subagent_tree() {
    let fixture = fixture();
    let events = replay(&fixture).snapshot();
    let mut ours = to_atif(&events, &meta()).unwrap();
    // 0.8 exported the redacted events; redact as a writer must.
    crate::trajectory::Redactor::default().redact_json(&mut ours);
    assert_eq!(ours["schema_version"], "ATIF-v1.8");
    assert_eq!(as_0_8(&ours), as_0_8(&fixture["atif_0_8"]));
    // The Task call's subagent is embedded and referenced.
    let subagent = &ours["subagent_trajectories"][0];
    assert_eq!(subagent["trajectory_id"], "subagent:toolu_task");
    assert_eq!(subagent["agent"]["extra"]["subagent_type"], "Plan");
    assert_eq!(
        subagent["steps"][0],
        json!({"step_id": 1, "source": "user", "message": "Review the plan."})
    );
    assert_eq!(
        ours["extra"]["acp_projection"]["attributed_child_events"],
        3
    );
}

#[test]
fn tool_calls_carry_the_agents_own_name_arguments_and_output() {
    let events = replay(&fixture()).snapshot();
    let ours = to_atif(&events, &meta()).unwrap();
    let steps = ours["steps"].as_array().unwrap();
    let step_for = |id: &str| {
        steps
            .iter()
            .find(|s| s["tool_calls"][0]["tool_call_id"] == id)
            .unwrap_or_else(|| panic!("no step for {id}"))
    };
    let bash = step_for("toolu_01");
    assert_eq!(
        bash["tool_calls"][0],
        json!({
            "tool_call_id": "toolu_01",
            "function_name": "Bash",
            "arguments": {"command": "ls -la /app", "description": "List files"},
            "extra": {"title": "ls -la /app", "status": "completed", "kind": "execute"},
        })
    );
    assert_eq!(
        bash["observation"]["results"][0]["content"],
        "```console\nREADME.md\nsrc\n```"
    );
    // Codex reports command output only in raw_output.
    let shell = step_for("call_shell");
    assert_eq!(shell["tool_calls"][0]["function_name"], "execute");
    assert_eq!(shell["observation"]["results"][0]["content"], "hello\n");
    // Steps are timed by their host receipts.
    assert_eq!(steps[0]["timestamp"], "2026-10-09T11:35:37.137001+00:00");
    assert!(steps.iter().all(|s| s["timestamp"].is_string()));
}

#[test]
fn an_empty_trajectory_has_no_document() {
    assert_eq!(to_atif(&[], &meta()), None);
    assert_eq!(to_atif(&[Event::nop()], &meta()), None);
}

#[test]
fn scripted_and_notice_events_become_steps() {
    let events = vec![
        Event::UserMessage(Text {
            text: "Solve it.".into(),
            parent_tool_call_id: None,
            receipt: None,
        }),
        Event::AgentThought(Text {
            text: "hmm".into(),
            parent_tool_call_id: None,
            receipt: None,
        }),
        Event::Notice(Notice {
            severity: "warning".into(),
            title: "Retrying".into(),
            description: Some("attempt 2".into()),
            receipt: None,
        }),
        Event::Oracle(Oracle {
            command: "oracle/solve.sh".into(),
            return_code: 0,
            stdout: String::new(),
        }),
    ];
    let doc = to_atif(
        &events,
        &AtifMeta {
            agent_name: String::new(),
            model: Some("claude-opus-5-5".into()),
            total_prompt_tokens: Some(100),
            total_cost_usd: Some(0.25),
            ..AtifMeta::default()
        },
    )
    .unwrap();
    assert_eq!(
        doc,
        json!({
            "schema_version": "ATIF-v1.8",
            "agent": {"name": "unknown", "version": "unknown", "model_name": "claude-opus-5-5"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "Solve it."},
                // A thought with no agent step after it stands alone.
                {"step_id": 2, "source": "agent", "message": "", "reasoning_content": "hmm"},
                {"step_id": 3, "source": "system", "message": "Retrying\n\nattempt 2",
                 "extra": {"notice": {"severity": "warning"}}},
                {"step_id": 4, "source": "agent", "message": "[oracle: oracle/solve.sh]"},
            ],
            "final_metrics": {"total_steps": 4, "total_prompt_tokens": 100, "total_cost_usd": 0.25},
        })
    );
}

#[test]
fn a_tool_call_without_an_id_gets_a_step_scoped_one() {
    let mut call = ToolCall::new(String::new(), String::new(), "t".into(), Receipt::at(5_000));
    call.status = ToolCallStatus::Completed;
    call.raw_input = Some(json!(["not", "an", "object"]));
    let doc = to_atif(&[Event::ToolCall(Box::new(call))], &meta()).unwrap();
    assert_eq!(
        doc["steps"][0]["tool_calls"][0],
        json!({
            "tool_call_id": "call_1",
            "function_name": "tool",
            "arguments": {},
            "extra": {"title": "t", "status": "completed", "raw_input": ["not", "an", "object"]},
        })
    );
}

#[test]
fn an_oracle_event_with_extra_keys_is_still_a_step() {
    // The Rust oracle writes `timed_out`, which 0.8's oracle event lacks, so
    // the event reads back as `Other`; 0.8's walker still made it a step.
    let line = r#"{"type": "oracle", "command": "oracle/solve.sh", "return_code": null, "stdout": "", "timed_out": false}"#;
    let events = crate::trajectory::parse_jsonl(line).unwrap();
    assert!(matches!(events[0], Event::Other(_)));
    let doc = to_atif(&events, &meta()).unwrap();
    assert_eq!(
        doc["steps"],
        json!([{"step_id": 1, "source": "agent", "message": "[oracle: oracle/solve.sh]"}])
    );
}
