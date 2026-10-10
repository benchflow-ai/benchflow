use std::time::Duration;

use serde_json::{Value, json};

use super::*;

#[test]
fn streamed_command_output_is_recorded_in_linear_time() {
    // codex-acp 2.x streams a command's output as many small chunks; folding
    // all of them on every chunk made recording quadratic.
    let mut trajectory = Trajectory::new();
    trajectory.apply_update(
        &json!({"sessionUpdate": "tool_call", "toolCallId": "c", "title": "make", "kind": "execute", "status": "in_progress"}),
        Receipt::at(1),
    );
    let line = "x".repeat(49) + "\n";
    for i in 0..20_000u64 {
        trajectory.apply_update(
            &json!({"sessionUpdate": "tool_call_update", "toolCallId": "c",
                    "_meta": {"terminal_output_delta": {"data": line, "terminal_id": "c"}}}),
            Receipt::at(2 + i),
        );
    }
    let call = trajectory.tool_call(0).unwrap();
    let output = call.raw_output().unwrap();
    assert_eq!(
        output["formatted_output"].as_str().unwrap().len(),
        1_000_000
    );
    assert_eq!(call.receipt.unwrap().last_observed_ns, 20_001);
}

/// testdata/session-0.8.json: a scripted session recorded by 0.8's
/// `ACPSession`, with what 0.8 wrote for it.
pub(super) fn fixture() -> Value {
    serde_json::from_str(include_str!("../../testdata/session-0.8.json")).unwrap()
}

/// Replay the fixture's steps into a trajectory.
pub(super) fn replay(fixture: &Value) -> Trajectory {
    let mut trajectory = Trajectory::new();
    for step in fixture["steps"].as_array().unwrap() {
        let at = || Receipt::at(step["ns"].as_u64().unwrap());
        if let Some(prompt) = step.get("prompt") {
            trajectory.record_user_prompt(prompt.as_str().unwrap(), at());
        } else if let Some(update) = step.get("update") {
            trajectory.apply_update(update, at());
        } else if step.get("finish_prompt").is_some() {
            trajectory.finish_prompt();
        } else if let Some(secs) = step.get("timeout_sec") {
            trajectory.finish_prompt();
            trajectory.record_timeout(Duration::from_secs_f64(secs.as_f64().unwrap()), at());
        } else {
            panic!("unknown step {step}");
        }
    }
    trajectory
}

#[test]
fn a_session_writes_the_same_bytes_0_8_wrote() {
    let fixture = fixture();
    let trajectory = replay(&fixture);
    let ours = to_jsonl(
        trajectory
            .events()
            .map(|e| e.into_owned())
            .collect::<Vec<_>>()
            .iter(),
        &Redactor::default(),
    );
    let theirs = fixture["jsonl"].as_str().unwrap();
    for (i, (a, b)) in ours.lines().zip(theirs.lines()).enumerate() {
        assert_eq!(a, b, "line {}", i + 1);
    }
    assert_eq!(ours, theirs);
}

#[test]
fn counts_match_what_0_8_put_in_result_json() {
    let fixture = fixture();
    let events = replay(&fixture).snapshot();
    let summary = summarize(&events, true, Some("acp"));
    assert_eq!(
        serde_json::to_value(&summary).unwrap(),
        fixture["trajectory_summary"]
    );
    // Key order matches result.json's.
    let keys: Vec<String> = serde_json::to_value(&summary)
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    let expected: Vec<String> = fixture["trajectory_summary"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(keys, expected);
    assert_eq!(
        count_tool_calls(&events),
        fixture["n_tool_calls"].as_u64().unwrap() as usize
    );
    assert_eq!(
        count_skill_invocations(&events),
        fixture["n_skill_invocations"].as_u64().unwrap() as usize
    );
    assert_eq!(replay(&fixture).tool_call_count(), 12);
}

#[test]
fn reading_and_writing_a_0_8_file_changes_nothing() {
    let theirs = fixture()["jsonl"].as_str().unwrap().to_string();
    let events = parse_jsonl(&theirs).unwrap();
    assert!(
        events.iter().all(|e| !matches!(e, Event::Other(_))),
        "every 0.8 event is modelled"
    );
    assert_eq!(to_jsonl(&events, &Redactor::default()), theirs);
}

/// Every `acp_trajectory.jsonl` listed in `$BENCHFLOW_REAL_TRAJECTORIES` (a
/// file of paths, one per line) round-trips byte for byte. Real 0.8 runs are
/// not checked into the repository; without the variable this test only
/// says so.
#[test]
fn real_0_8_trajectories_round_trip() {
    let Ok(list) = std::env::var("BENCHFLOW_REAL_TRAJECTORIES") else {
        eprintln!(
            "set BENCHFLOW_REAL_TRAJECTORIES to a file listing acp_trajectory.jsonl paths to check real runs"
        );
        return;
    };
    let paths = std::fs::read_to_string(&list).unwrap();
    let mut checked = 0;
    let mut typed = 0;
    let mut other = 0;
    let mut other_kinds = std::collections::BTreeMap::new();
    let mut other_shapes = std::collections::BTreeSet::new();
    for path in paths.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        if text.starts_with("version https://git-lfs.github.com/spec/") {
            continue; // a Git LFS pointer, not the file
        }
        // 0.8's live writer leaves no newline at the end; its copy for the
        // verifier adds one. Both read back the same.
        let body = text.strip_suffix('\n').unwrap_or(&text);
        let events = parse_jsonl(body).unwrap_or_else(|e| panic!("{path}: {e}"));
        for event in &events {
            match event {
                Event::Other(map) => {
                    other += 1;
                    *other_kinds
                        .entry(event.kind().to_string())
                        .or_insert(0usize) += 1;
                    let keys: Vec<&str> = map.keys().map(String::as_str).collect();
                    other_shapes.insert(format!("{} {keys:?}", event.kind()));
                }
                _ => typed += 1,
            }
        }
        // Serialization only: files from older 0.8 releases predate some of
        // today's patterns (testdata/redaction-0.8.json checks those).
        let ours = to_jsonl(&events, &Redactor::default().exact_only());
        if let Some(diff) = first_difference(&ours, body) {
            panic!("{path}: {diff}");
        }
        checked += 1;
    }
    eprintln!(
        "{checked} real trajectories round-tripped ({typed} typed events, {other} kept as written: {other_kinds:?})"
    );
    for shape in other_shapes.iter().take(20) {
        eprintln!("  kept as written: {shape}");
    }
    assert!(checked > 0, "no readable trajectory in {list}");
}

/// Where two texts first differ: line, column and both sides around it.
fn first_difference(ours: &str, theirs: &str) -> Option<String> {
    if ours == theirs {
        return None;
    }
    for (i, (a, b)) in ours.split('\n').zip(theirs.split('\n')).enumerate() {
        if a != b {
            let at = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
            let window = |s: &str| {
                let mut start = at.saturating_sub(80);
                while !s.is_char_boundary(start) {
                    start -= 1;
                }
                let mut end = (at + 80).min(s.len());
                while !s.is_char_boundary(end) {
                    end += 1;
                }
                s[start..end].to_string()
            };
            return Some(format!(
                "line {} byte {at}:\n ours: {}\n 0.8:  {}",
                i + 1,
                window(a),
                window(b)
            ));
        }
    }
    Some(format!(
        "{} lines vs {}",
        ours.split('\n').count(),
        theirs.split('\n').count()
    ))
}

#[test]
fn a_live_snapshot_includes_the_text_still_streaming() {
    let mut trajectory = Trajectory::new();
    trajectory.record_user_prompt("go", Receipt::at(1_000));
    let chunk = |text: &str| json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}});
    trajectory.apply_update(&chunk("Hel"), Receipt::at(2_000));
    trajectory.apply_update(&chunk("lo"), Receipt::at(3_000));
    let live = trajectory.snapshot();
    assert_eq!(live.len(), 2);
    let Event::AgentMessage(text) = &live[1] else {
        panic!("{live:?}")
    };
    assert_eq!(text.text, "Hello");
    assert_eq!(
        text.receipt,
        Some(Receipt {
            first_observed_ns: 2_000,
            last_observed_ns: 3_000
        })
    );
    // Snapshots do not close the run of text.
    trajectory.apply_update(&chunk("!"), Receipt::at(4_000));
    trajectory.finish_prompt();
    let Event::AgentMessage(text) = &trajectory.snapshot()[1] else {
        panic!()
    };
    assert_eq!(text.text, "Hello!");
    assert_eq!(trajectory.final_text(), "Hello!");
}

#[test]
fn an_unchanged_status_poll_does_not_move_the_receipt() {
    // 0.8's receipt rule: last_observed_ns follows the latest update that
    // changed what is written, so repeated polls leave the file identical.
    let mut trajectory = Trajectory::new();
    trajectory.apply_update(
        &json!({"sessionUpdate": "tool_call", "toolCallId": "t", "title": "x", "kind": "read"}),
        Receipt::at(10),
    );
    let poll =
        json!({"sessionUpdate": "tool_call_update", "toolCallId": "t", "status": "in_progress"});
    trajectory.apply_update(&poll, Receipt::at(20));
    trajectory.apply_update(&poll, Receipt::at(30));
    let call = trajectory.tool_call(0).unwrap();
    assert_eq!(call.receipt.unwrap().last_observed_ns, 20);
    assert_eq!(trajectory.activity().pending, [("t".to_string(), 2)]);
    trajectory.apply_update(
        &json!({"sessionUpdate": "tool_call_update", "toolCallId": "t", "status": "completed"}),
        Receipt::at(40),
    );
    let call = trajectory.tool_call(0).unwrap();
    assert_eq!(call.finished_observed_ns, Some(40));
    assert!(trajectory.pending_tool_call_ids().is_empty());
    assert_eq!(trajectory.activity().tool_call_updates, 3);
}

#[test]
fn other_updates_are_kept_but_are_not_events() {
    let mut trajectory = Trajectory::new();
    let plan = json!({"sessionUpdate": "plan", "entries": [{"content": "a", "priority": "high", "status": "pending"}]});
    assert_eq!(trajectory.apply_update(&plan, Receipt::at(1)), Change::Plan);
    let usage = json!({"sessionUpdate": "usage_update", "used": 900, "size": 200000});
    assert_eq!(
        trajectory.apply_update(&usage, Receipt::at(2)),
        Change::ContextUsage
    );
    let mode = json!({"sessionUpdate": "current_mode_update", "currentModeId": "plan"});
    assert_eq!(
        trajectory.apply_update(&mode, Receipt::at(3)),
        Change::Other
    );
    let future = json!({"sessionUpdate": "something_new", "x": 1});
    trajectory.apply_update(&future, Receipt::at(4));
    assert!(trajectory.snapshot().is_empty());
    let kept: Vec<&Value> = trajectory
        .other_updates()
        .iter()
        .map(|u| &u.update)
        .collect();
    assert_eq!(kept, [&plan, &usage, &mode, &future]);
    assert_eq!(trajectory.plan(), Some(&plan["entries"]));
    assert_eq!(trajectory.context_usage().unwrap().used, Some(900));
}

#[test]
fn usage_reads_both_spellings() {
    assert_eq!(
        Usage::from_acp(&json!({"inputTokens": 10, "outputTokens": "5", "cachedReadTokens": 3})),
        Some(Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            cached_read_tokens: Some(3),
            ..Usage::default()
        })
    );
    assert_eq!(
        Usage::from_acp(&json!({"total_tokens": 7}))
            .unwrap()
            .tokens(),
        Some(7)
    );
    assert_eq!(
        Usage::from_acp(&json!({"inputTokens": 2, "outputTokens": 3}))
            .unwrap()
            .tokens(),
        Some(5)
    );
    assert_eq!(Usage::from_acp(&json!({})), None);
    assert_eq!(Usage::from_acp(&json!({"inputTokens": null})), None);
}

#[test]
fn scripted_agents_have_their_0_8_events() {
    let events = vec![
        Event::Oracle(Oracle {
            command: "oracle/solve.sh".into(),
            return_code: 0,
            stdout: "done\n".into(),
        }),
        Event::nop(),
    ];
    assert_eq!(
        to_jsonl(&events, &Redactor::default()),
        "{\"type\": \"oracle\", \"command\": \"oracle/solve.sh\", \"return_code\": 0, \"stdout\": \"done\\n\"}\n\
         {\"type\": \"nop\", \"note\": \"nop agent: nothing ran; the verifier scores the untouched workspace\"}"
    );
    let summary = summarize(&events, false, Some("oracle"));
    assert_eq!(summary.other_steps, 2);
    assert_eq!(count_tool_calls(&events), 0);
}

#[test]
fn login_secrets_are_redacted_in_the_file() {
    let mut trajectory = Trajectory::new();
    trajectory.record_user_prompt("go", Receipt::at(1));
    trajectory.apply_update(
        &json!({"sessionUpdate": "tool_call", "toolCallId": "t", "title": "env", "kind": "execute",
                "status": "completed", "rawOutput": "MYAGENT=login-value-123456789"}),
        Receipt::at(2),
    );
    let events = trajectory.snapshot();
    let text = to_jsonl(&events, &Redactor::new(["login-value-123456789"]));
    assert!(!text.contains("login-value-123456789"), "{text}");
    assert!(text.contains("MYAGENT=***REDACTED***"), "{text}");
}
