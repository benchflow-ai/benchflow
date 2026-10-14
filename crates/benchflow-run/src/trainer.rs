//! Trainer files: what a trial and a job leave for trainers, as BenchFlow
//! 0.8 wrote them (its "train-mode seam", `rollout/_results.py`
//! `_write_trainer_artifact`).
//!
//! ```text
//! <job>/
//!   verifiers.jsonl  adp.jsonl         every trial's records, concatenated
//!   <task>__<id>/trainer/
//!     verifiers.jsonl                  one Verifiers / ORS record
//!     atif.json                        one ATIF document
//!     atif.coverage.json               only when atif.json is left out
//!     adp.jsonl                        one ADP 1.3.1 record
//! ```
//!
//! Every trial that writes a result.json gets them, scored or not, from its
//! `trajectory/acp_trajectory.jsonl`, its prompts, and its rewards. Each
//! file is written on its own: one that fails is reported and the others
//! are still written, and the trial's result stands (0.8's default; its
//! strict mode, used when a reviewed trial commits its score, is for the
//! caller to choose by treating any reported failure as fatal).
//!
//! - `atif.json` is left out when the trajectory has no step (ATIF requires
//!   one). When that is because every event was a subagent's,
//!   `atif.coverage.json` says so instead.
//! - `adp.jsonl` is left out when there is no content and no reward.
//! - The trajectory id (ATIF `session_id`, ADP `id`) is the trial name,
//!   prefixed with `<task>__` when it does not already start with it.
//!
//! Verifiers and ADP match 0.8 byte for byte. ATIF is the rewrite's v1.8
//! ([`benchflow_agent::trajectory::to_atif`]): 0.8's v1.7 document plus
//! step timestamps and the agent's own tool names, arguments, and output.

use std::path::{Path, PathBuf};

use benchflow_agent::trajectory::{
    AdpMeta, AtifMeta, Event, Redactor, VerifiersMeta, acp_projection_coverage, adp_line,
    parse_jsonl, to_adp_record, to_atif, to_python_json, to_python_json_pretty,
    to_verifiers_record, verifiers_line,
};
use serde_json::{Map, Value, json};

use crate::record::TrialRecord;

pub const VERIFIERS: &str = "trainer/verifiers.jsonl";
pub const ATIF: &str = "trainer/atif.json";
pub const ATIF_COVERAGE: &str = "trainer/atif.coverage.json";
pub const ADP: &str = "trainer/adp.jsonl";
/// The job-level files, next to the trial folders.
pub const JOB_VERIFIERS: &str = "verifiers.jsonl";
pub const JOB_ADP: &str = "adp.jsonl";

/// What the trainer files say about a trial besides its events.
#[derive(Debug, Clone, Default)]
pub struct RolloutInfo {
    pub task_name: String,
    /// The trial's name (its folder name).
    pub rollout_name: String,
    /// The agent's name, for ATIF.
    pub agent_name: String,
    /// The agent's version, for ATIF.
    pub agent_version: Option<String>,
    /// The prompts the agent was given.
    pub prompts: Vec<String>,
    /// The verifier's reward map, or `None` when the trial was not scored.
    pub rewards: Option<Map<String, Value>>,
    pub model: Option<String>,
    /// Why the verifier could not score the trial.
    pub verifier_error: Option<String>,
    pub total_prompt_tokens: Option<u64>,
    pub total_completion_tokens: Option<u64>,
    pub total_cached_tokens: Option<u64>,
    pub total_cost_usd: Option<f64>,
}

impl RolloutInfo {
    /// What a trial record holds, as 0.8 passed it.
    pub fn from_record(rec: &TrialRecord) -> Self {
        Self {
            task_name: rec.task_name.clone(),
            rollout_name: rec.rollout_name.clone(),
            agent_name: rec.agent.clone(),
            agent_version: rec.agent_version.clone(),
            prompts: rec.prompts.clone(),
            rewards: rec
                .rewards
                .as_ref()
                .map(|r| r.iter().map(|(k, v)| (k.clone(), json!(v))).collect()),
            model: rec.model.clone(),
            verifier_error: rec.result_file().verifier_error,
            // ATIF's prompt tokens include the cached ones; BenchFlow's
            // `input_tokens` leaves out cache reads and writes.
            total_prompt_tokens: Some(
                rec.usage.input_tokens + rec.usage.cache_read_tokens + rec.usage.cache_write_tokens,
            ),
            total_completion_tokens: Some(rec.usage.output_tokens),
            total_cached_tokens: Some(rec.usage.cache_read_tokens),
            total_cost_usd: rec.usage.cost_usd,
        }
    }

    /// Without the prompts when the trajectory records them. BenchFlow
    /// records each prompt it sends as a `user_message` event (a native loop
    /// records the prompt as the model got it, wrapped in its own template),
    /// and the trainer files turn both prompts and those events into user
    /// turns, so passing both wrote every prompt twice. The prompts stay only
    /// when the root agent has no recorded prompt (an oracle or scripted run).
    pub fn without_recorded_prompts(mut self, events: &[Event]) -> Self {
        let recorded = events.iter().any(|e| {
            matches!(e, Event::UserMessage(t) if t.parent_tool_call_id.is_none() && !t.text.is_empty())
        });
        if recorded {
            self.prompts.clear();
        }
        self
    }

    /// The id the ATIF and ADP files give the trajectory.
    pub fn trajectory_id(&self) -> String {
        trajectory_id(&self.task_name, &self.rollout_name)
    }
}

/// Trial names already start with the task (`<task>__<id>`); others are
/// prefixed so ids stay unique within a job.
pub fn trajectory_id(task_name: &str, rollout_name: &str) -> String {
    if rollout_name.starts_with(&format!("{task_name}__")) {
        rollout_name.to_string()
    } else {
        format!("{task_name}__{rollout_name}")
    }
}

/// A trainer file that could not be written.
#[derive(Debug, thiserror::Error)]
#[error("{file} was not written: {source}")]
pub struct TrainerError {
    pub file: &'static str,
    #[source]
    pub source: std::io::Error,
}

/// Write a finished trial's trainer files from its folder's trajectory.
/// Returns what failed; an empty list means everything was written.
pub fn write_trial(dir: &Path, rec: &TrialRecord) -> Vec<TrainerError> {
    let tagged = match read_tagged_events(dir) {
        Ok(events) => events,
        Err(source) => {
            return vec![TrainerError {
                file: "trajectory/acp_trajectory.jsonl",
                source,
            }];
        }
    };
    let info = RolloutInfo::from_record(rec);
    let (events, info) = with_stage_prompts(tagged, info);
    let info = info.without_recorded_prompts(&events);
    write_rollout(dir, &events, &info, &Redactor::default())
}

/// A trial folder's events, each with the stage it ran in when the task is
/// staged (`None` otherwise), the tag removed from the event.
fn read_tagged_events(dir: &Path) -> std::io::Result<Vec<(Option<String>, Event)>> {
    match std::fs::read_to_string(dir.join("trajectory/acp_trajectory.jsonl")) {
        Ok(text) => Ok(parse_jsonl(&text)
            .map_err(std::io::Error::other)?
            .into_iter()
            .map(|e| {
                let stage = match &e {
                    Event::Other(map) => {
                        map.get("stage").and_then(Value::as_str).map(str::to_string)
                    }
                    _ => None,
                };
                (stage, without_stage(e))
            })
            .collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// The events with each stage's prompt placed before that stage's first
/// event, when the trajectory records no prompt of its own (an oracle or
/// scripted run) and there is one prompt per stage. Without this the prompts
/// all led the document, then every stage's work followed.
fn with_stage_prompts(
    tagged: Vec<(Option<String>, Event)>,
    mut info: RolloutInfo,
) -> (Vec<Event>, RolloutInfo) {
    let mut stages: Vec<&str> = Vec::new();
    for (stage, _) in &tagged {
        if let Some(stage) = stage.as_deref()
            && !stages.contains(&stage)
        {
            stages.push(stage);
        }
    }
    let recorded = tagged
        .iter()
        .any(|(_, e)| matches!(e, Event::UserMessage(_)));
    if stages.len() < 2 || recorded || stages.len() != info.prompts.len() {
        return (tagged.into_iter().map(|(_, e)| e).collect(), info);
    }
    let prompts = std::mem::take(&mut info.prompts);
    let mut out = Vec::with_capacity(tagged.len() + prompts.len());
    let mut next = 0;
    let mut current: Option<String> = None;
    for (stage, event) in tagged {
        if stage.is_some() && stage != current {
            if let Some(prompt) = prompts.get(next) {
                out.push(Event::UserMessage(benchflow_agent::trajectory::Text {
                    text: prompt.clone(),
                    parent_tool_call_id: None,
                    receipt: None,
                }));
            }
            next += 1;
            current = stage;
        }
        out.push(event);
    }
    (out, info)
}

/// A trial folder's events: none when it has no trajectory.
///
/// A staged task's events carry the stage they ran in (`"stage"`, added
/// when each stage is archived). The trainer files describe the trial as one
/// rollout, so the tag is dropped here; kept, it would make every event read
/// as an unknown one, which no trainer file projects.
pub fn read_events(dir: &Path) -> std::io::Result<Vec<Event>> {
    match std::fs::read_to_string(dir.join("trajectory/acp_trajectory.jsonl")) {
        Ok(text) => Ok(parse_jsonl(&text)
            .map_err(std::io::Error::other)?
            .into_iter()
            .map(without_stage)
            .collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// `event` without the stage tag a staged task's events carry.
pub fn without_stage(event: Event) -> Event {
    match event {
        Event::Other(map) if map.get("stage").is_some_and(Value::is_string) => {
            Event::from_json(map.into_iter().filter(|(k, _)| k != "stage").collect())
        }
        other => other,
    }
}

/// Write the three trainer files for one rollout into `dir/trainer/`.
/// Returns what failed.
pub fn write_rollout(
    dir: &Path,
    events: &[Event],
    info: &RolloutInfo,
    redactor: &Redactor,
) -> Vec<TrainerError> {
    let json: Vec<Value> = events.iter().map(Event::to_json).collect();
    let mut failures = Vec::new();
    let mut check = |file, result: std::io::Result<()>| {
        if let Err(source) = result {
            failures.push(TrainerError { file, source });
        }
    };
    check(VERIFIERS, write_verifiers(dir, &json, info, redactor));
    check(ATIF, write_atif(dir, events, &json, info, redactor));
    check(ADP, write_adp(dir, &json, info, redactor));
    failures
}

fn write_file(dir: &Path, rel: &str, text: &str) -> std::io::Result<()> {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::record::write_atomic(&path, text.as_bytes())
}

fn remove_file(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

fn write_verifiers(
    dir: &Path,
    events: &[Value],
    info: &RolloutInfo,
    redactor: &Redactor,
) -> std::io::Result<()> {
    let record = to_verifiers_record(
        events,
        &VerifiersMeta {
            task_id: &info.task_name,
            prompts: &info.prompts,
            rewards: info.rewards.as_ref(),
            error: info.verifier_error.as_deref(),
            model: info.model.as_deref(),
        },
    );
    write_file(
        dir,
        VERIFIERS,
        &verifiers_line(&clean(record, redactor), redactor),
    )
}

/// A trainer record redacted value by value, and without the key body
/// lines between a key's markers that no single value shows as a key (a
/// key read in parts) ([`Redactor::redact_record`]). The trial's file and
/// the job's line are the same.
fn clean(mut record: Value, redactor: &Redactor) -> Value {
    redactor.redact_record(&mut record);
    record
}

fn write_adp(
    dir: &Path,
    events: &[Value],
    info: &RolloutInfo,
    redactor: &Redactor,
) -> std::io::Result<()> {
    let trajectory_id = info.trajectory_id();
    let record = to_adp_record(
        events,
        &AdpMeta {
            trajectory_id: &trajectory_id,
            task_id: &info.task_name,
            prompts: &info.prompts,
            model: info.model.as_deref(),
            reward: info.rewards.as_ref().and_then(|r| r.get("reward")),
        },
    );
    // Like 0.8, a record with nothing to say is not written, and an older
    // file is left alone.
    match record {
        Some(record) => write_file(dir, ADP, &adp_line(&clean(record, redactor), redactor)),
        None => Ok(()),
    }
}

/// What a trial's ATIF document says about it besides its events.
pub fn atif_meta(info: &RolloutInfo) -> AtifMeta {
    AtifMeta {
        session_id: Some(info.trajectory_id()),
        agent_name: info.agent_name.clone(),
        agent_version: info.agent_version.clone(),
        model: info.model.clone().filter(|m| !m.is_empty()),
        prompts: info.prompts.clone(),
        total_prompt_tokens: info.total_prompt_tokens,
        total_completion_tokens: info.total_completion_tokens,
        total_cached_tokens: info.total_cached_tokens,
        total_cost_usd: info.total_cost_usd,
    }
}

fn write_atif(
    dir: &Path,
    events: &[Event],
    json: &[Value],
    info: &RolloutInfo,
    redactor: &Redactor,
) -> std::io::Result<()> {
    let pretty = |mut value: Value| {
        redactor.redact_json(&mut value);
        benchflow_agent::trajectory::cap_long_strings(
            &mut value,
            benchflow_agent::trajectory::DERIVED_TEXT_CAP,
            "trajectory/acp_trajectory.jsonl",
        );
        to_python_json_pretty(&value) + "\n"
    };
    let coverage = acp_projection_coverage(json);
    match to_atif(events, &atif_meta(info)) {
        Some(doc) => {
            remove_file(&dir.join(ATIF_COVERAGE))?;
            write_file(dir, ATIF, &pretty(doc))
        }
        None => {
            match coverage {
                // Every event was a subagent's: say so rather than invent a
                // step.
                Some(coverage) => write_file(
                    dir,
                    ATIF_COVERAGE,
                    &pretty(json!({
                        "artifact_status": "omitted_empty_root_projection",
                        "acp_projection": coverage,
                    })),
                )?,
                None => remove_file(&dir.join(ATIF_COVERAGE))?,
            }
            remove_file(&dir.join(ATIF))
        }
    }
}

/// Write the job-level `verifiers.jsonl` and `adp.jsonl`: every trial
/// folder's file, in folder-name order, redacted again on the way through.
/// A file is not written when no trial has one. Returns what failed.
pub fn write_job(job: &Path) -> Vec<TrainerError> {
    let redactor = Redactor::default();
    let mut failures = Vec::new();
    for (rel, out, file) in [
        (VERIFIERS, JOB_VERIFIERS, "verifiers.jsonl"),
        (ADP, JOB_ADP, "adp.jsonl"),
    ] {
        if let Err(source) = aggregate(job, rel, out, &redactor) {
            failures.push(TrainerError { file, source });
        }
    }
    failures
}

/// 0.8's `aggregate_rollout_jsonl`: concatenate `job/*/<rel>` into
/// `job/<out>`. A trial file that cannot be read (or is not UTF-8) is
/// skipped with a warning.
fn aggregate(
    job: &Path,
    rel: &str,
    out: &str,
    redactor: &Redactor,
) -> std::io::Result<Option<PathBuf>> {
    if !job.is_dir() {
        return Ok(None);
    }
    let mut sources: Vec<(String, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(job)? {
        let entry = entry?;
        let path = entry.path().join(rel);
        if path.is_file() {
            sources.push((entry.file_name().to_string_lossy().into_owned(), path));
        }
    }
    if sources.is_empty() {
        return Ok(None);
    }
    sources.sort();
    let mut text = String::new();
    for (_, path) in &sources {
        match std::fs::read_to_string(path) {
            Ok(lines) => {
                // Each line is redacted as a value, never as text: 0.8's text
                // pass over the whole line could match inside a JSON escape,
                // pair a key's BEGIN and END markers across structures, or
                // cut the messages, the `completion` or the ADP `reward`
                // between two markers. A key read in parts (`head`, `sed -n`,
                // `tail`) loses its middle part to the walk over the record's
                // strings, which removes body lines inside strings and
                // nothing else ([`Redactor::redact_key_chunks`]). A line
                // that is not JSON gets the text pass, as 0.8 gave every
                // line.
                for line in lines.lines() {
                    match serde_json::from_str::<Value>(line) {
                        Ok(value) => text.push_str(&to_python_json(&clean(value, redactor))),
                        Err(_) => text.push_str(&redactor.redact(line)),
                    }
                    text.push('\n');
                }
            }
            Err(e) => {
                tracing::warn!("skipping unreadable trainer file {}: {e}", path.display());
            }
        }
    }
    let path = job.join(out);
    crate::record::write_atomic(&path, text.as_bytes())?;
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_ids_carry_the_task_once() {
        assert_eq!(trajectory_id("t", "t__1a2b"), "t__1a2b");
        assert_eq!(trajectory_id("t", "run-1"), "t__run-1");
        assert_eq!(trajectory_id("t", "tt__1"), "t__tt__1");
    }

    fn info() -> RolloutInfo {
        RolloutInfo {
            task_name: "t".into(),
            rollout_name: "t__1".into(),
            agent_name: "claude".into(),
            ..RolloutInfo::default()
        }
    }

    /// E1: ATIF's `total_prompt_tokens` includes cached tokens
    /// (`total_cached_tokens` is a subset of it); a MiMo trial had 11,278
    /// prompt and 58,688 cached tokens.
    #[test]
    fn atif_prompt_tokens_include_the_cached_ones() {
        let rec = crate::read::record_from_json(
            &json!({"status": "scored", "rewards": {"reward": 1.0}, "agent_result": {
                "n_input_tokens": 11278, "n_output_tokens": 2068,
                "n_cache_read_tokens": 58688, "n_cache_creation_tokens": 34}}),
            Path::new("t__1"),
        )
        .unwrap();
        let info = RolloutInfo::from_record(&rec);
        assert_eq!(info.total_prompt_tokens, Some(11278 + 58688 + 34));
        assert_eq!(info.total_cached_tokens, Some(58688));
        assert_eq!(info.total_completion_tokens, Some(2068));
    }

    #[test]
    fn an_empty_trajectory_has_no_atif_and_no_adp() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("trainer")).unwrap();
        std::fs::write(dir.path().join(ATIF), "stale").unwrap();
        let failures = write_rollout(dir.path(), &[], &info(), &Redactor::default());
        assert!(failures.is_empty(), "{failures:?}");
        assert!(dir.path().join(VERIFIERS).is_file());
        assert!(!dir.path().join(ATIF).exists());
        assert!(!dir.path().join(ATIF_COVERAGE).exists());
        assert!(!dir.path().join(ADP).exists());
    }

    #[test]
    fn a_trajectory_of_only_subagent_events_leaves_a_coverage_receipt() {
        let dir = tempfile::tempdir().unwrap();
        let line = r#"{"type": "agent_message", "text": "child", "parent_tool_call_id": "gone"}"#;
        let events = parse_jsonl(line).unwrap();
        let mut info = info();
        info.rewards = Some(json!({"reward": 0.0}).as_object().unwrap().clone());
        let failures = write_rollout(dir.path(), &events, &info, &Redactor::default());
        assert!(failures.is_empty(), "{failures:?}");
        // The child's events have a step in an embedded trajectory, but the
        // root has none, so there is no document.
        assert!(!dir.path().join(ATIF).exists());
        let receipt = std::fs::read_to_string(dir.path().join(ATIF_COVERAGE)).unwrap();
        // Python: _results._write_trainer_artifact on the same event.
        assert_eq!(
            receipt,
            "{\n  \"artifact_status\": \"omitted_empty_root_projection\",\n  \"acp_projection\": {\n    \"scope\": \"root_agent_projection\",\n    \"excluded_attributed_child_events\": 1,\n    \"source_artifact\": \"trajectory/acp_trajectory.jsonl\",\n    \"child_transcript_completeness\": \"unknown\",\n    \"outcome_and_usage_scope\": \"rollout_not_partitioned_by_agent\"\n  }\n}\n"
        );
        // ADP has no content but carries the reward.
        assert_eq!(
            std::fs::read_to_string(dir.path().join(ADP)).unwrap(),
            concat!(
                r#"{"schema_version": "1.3.1", "id": "t__1", "content": [], "details": {"task_id": "t", "environment": "t", "model": "", "#,
                r#""acp_projection": {"scope": "root_agent_projection", "excluded_attributed_child_events": 1, "source_artifact": "trajectory/acp_trajectory.jsonl", "child_transcript_completeness": "unknown", "outcome_and_usage_scope": "rollout_not_partitioned_by_agent"}, "#,
                r#""terminal_reward": 0.0}}"#,
                "\n"
            )
        );
        // A later write with a root step removes the receipt.
        let events = parse_jsonl(r#"{"type": "agent_message", "text": "root"}"#).unwrap();
        write_rollout(dir.path(), &events, &info, &Redactor::default());
        assert!(dir.path().join(ATIF).is_file());
        assert!(!dir.path().join(ATIF_COVERAGE).exists());
    }

    #[test]
    fn a_staged_trials_events_reach_the_trainer_files() {
        // Each stage's events are archived with the stage's name; they must
        // still project as messages and tool calls, not unknown events.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("trajectory")).unwrap();
        std::fs::write(
            dir.path().join("trajectory/acp_trajectory.jsonl"),
            concat!(
                r#"{"type": "user_message", "text": "write alpha", "stage": "first"}"#,
                "\n",
                r#"{"type": "tool_call", "tool_call_id": "c1", "kind": "execute", "title": "printf alpha", "status": "completed", "content": [], "stage": "first"}"#,
                "\n",
                r#"{"type": "agent_message", "text": "done", "stage": "first"}"#,
                "\n",
                r#"{"type": "user_message", "text": "write beta", "stage": "second"}"#,
                "\n",
                r#"{"type": "agent_message", "text": "done too", "stage": "second"}"#,
            ),
        )
        .unwrap();
        let events = read_events(dir.path()).unwrap();
        let kinds: Vec<&str> = events.iter().map(Event::kind).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "tool_call",
                "agent_message",
                "user_message",
                "agent_message"
            ]
        );
        assert!(events.iter().all(|e| !matches!(e, Event::Other(_))));
        let failures = write_rollout(dir.path(), &events, &info(), &Redactor::default());
        assert!(failures.is_empty(), "{failures:?}");
        let atif: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(ATIF)).unwrap()).unwrap();
        let steps = atif["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 5, "{atif}");
        assert_eq!(steps[1]["tool_calls"][0]["tool_call_id"], "c1");
        assert_eq!(steps[4]["message"], "done too");
        // An event that is unknown for another reason stays unknown.
        let other = parse_jsonl(r#"{"type": "plan", "entries": [], "stage": "first"}"#).unwrap();
        assert!(matches!(without_stage(other[0].clone()), Event::Other(_)));
    }

    #[test]
    fn a_recorded_prompt_is_one_user_turn_in_every_trainer_file() {
        let dir = tempfile::tempdir().unwrap();
        let events = parse_jsonl(concat!(
            r#"{"type": "user_message", "text": "fix the bug"}"#,
            "\n",
            r#"{"type": "agent_message", "text": "fixed"}"#,
        ))
        .unwrap();
        let mut info = info();
        info.prompts = vec!["fix the bug".into()];
        info.rewards = Some(json!({"reward": 1.0}).as_object().unwrap().clone());
        let info = info.without_recorded_prompts(&events);
        assert!(info.prompts.is_empty());
        assert!(write_rollout(dir.path(), &events, &info, &Redactor::default()).is_empty());
        let read = |rel: &str| std::fs::read_to_string(dir.path().join(rel)).unwrap();
        let atif: Value = serde_json::from_str(&read(ATIF)).unwrap();
        let users = atif["steps"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|s| s["source"] == "user")
            .count();
        assert_eq!(users, 1, "{atif}");
        assert_eq!(read(VERIFIERS).matches("fix the bug").count(), 1);
        assert_eq!(read(ADP).matches("fix the bug").count(), 1);

        // A native loop records the prompt wrapped in its template: the
        // template is the user turn, not the bare prompt as well.
        let wrapped = parse_jsonl(
            r#"{"type": "user_message", "text": "You are an assistant. Task: fix the bug. Reply in JSON."}"#,
        )
        .unwrap();
        let mut info = self::info();
        info.prompts = vec!["fix the bug".into()];
        assert!(info.without_recorded_prompts(&wrapped).prompts.is_empty());

        // An oracle run records no user message: its prompt stays.
        let oracle = parse_jsonl(r#"{"type": "oracle", "command": "solve.sh"}"#).unwrap();
        let mut info = self::info();
        info.prompts = vec!["fix the bug".into()];
        assert_eq!(
            info.without_recorded_prompts(&oracle).prompts,
            ["fix the bug"]
        );
    }

    #[test]
    fn a_staged_oracles_prompts_sit_before_their_stages() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("trajectory")).unwrap();
        std::fs::write(
            dir.path().join("trajectory/acp_trajectory.jsonl"),
            concat!(
                r#"{"type": "oracle", "command": "stages/first/oracle/solve.sh", "return_code": 0, "stdout": "", "timed_out": false, "stage": "first"}"#,
                "\n",
                r#"{"type": "oracle", "command": "stages/second/oracle/solve.sh", "return_code": 0, "stdout": "", "timed_out": false, "stage": "second"}"#,
            ),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("prompts.json"),
            r#"["write alpha", "write beta"]"#,
        )
        .unwrap();
        let rec = crate::read::record_from_json(
            &json!({"task_name": "t", "rollout_name": "t__1", "agent": "oracle", "status": "scored", "rewards": {"reward": 1.0}}),
            dir.path(),
        )
        .unwrap();
        assert!(write_trial(dir.path(), &rec).is_empty());
        let atif: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join(ATIF)).unwrap()).unwrap();
        let steps: Vec<(&str, &str)> = atif["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s["source"].as_str().unwrap(),
                    s["message"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            steps,
            [
                ("user", "write alpha"),
                ("agent", "[oracle: stages/first/oracle/solve.sh]"),
                ("user", "write beta"),
                ("agent", "[oracle: stages/second/oracle/solve.sh]"),
            ]
        );
    }

    /// Review FIX7-B 1: the job files are redacted as serialized text. A
    /// value with a line ending in a non-base64 character just before an END
    /// marker had the half-key rule match from the `n` of the `\n` escape and
    /// write `\***REDACTED***`, an invalid escape, so the line was lost.
    #[test]
    fn job_files_stay_json_around_a_key_marker() {
        let job = tempfile::tempdir().unwrap();
        let prompt = "The key's last line is:\n-----END OPENSSH PRIVATE KEY-----\nFind the key.";
        let read =
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ\n";
        let line = serde_json::to_string(&json!({"prompt": prompt, "tool": read})).unwrap();
        for rel in [VERIFIERS, ADP] {
            std::fs::create_dir_all(job.path().join("t__1/trainer")).unwrap();
            std::fs::write(job.path().join("t__1").join(rel), format!("{line}\n")).unwrap();
        }
        assert!(write_job(job.path()).is_empty());
        for out in [JOB_VERIFIERS, JOB_ADP] {
            let text = std::fs::read_to_string(job.path().join(out)).unwrap();
            for l in text.lines() {
                let v: Value =
                    serde_json::from_str(l).unwrap_or_else(|e| panic!("{out}: {e}: {l}"));
                assert!(
                    !v["tool"]
                        .as_str()
                        .unwrap()
                        .contains("b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ")
                );
            }
        }
    }

    /// Review FIX8-A 1, FIX8-B 1: non-ASCII text is written as `\\uXXXX`, so
    /// a half-key match could start on the hex digits before an END marker
    /// (`：` is `\\uff1a`) and break the escape. FIX8-A 2: the text pass
    /// paired a BEGIN in one message with an END in a tool call's arguments,
    /// a level deeper, and cut the JSON between them. Every job line parses,
    /// and no key body survives.
    #[test]
    fn job_files_stay_json_with_non_ascii_and_nested_markers() {
        let job = tempfile::tempdir().unwrap();
        let body = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ";
        let records = [
            json!({"prompt": "私钥最后一行是：\n-----END RSA PRIVATE KEY-----\n",
                   "tool": format!("-----BEGIN RSA PRIVATE KEY-----\n{body}\n")}),
            json!({"prompt": "The key ends…\n-----END RSA PRIVATE KEY-----"}),
            json!({"prompt": [
                {"role": "tool", "content": format!("-----BEGIN OPENSSH PRIVATE KEY-----\n{body}\n")},
                {"role": "assistant", "tool_calls": [{"function": {"name": "bash",
                    "arguments": "{\"cmd\": \"grep -n -- '-----END OPENSSH PRIVATE KEY-----' id\"}"}}]}]}),
        ];
        let lines: String = records
            .iter()
            .map(|r| benchflow_agent::trajectory::to_python_json(r) + "\n")
            .collect();
        for rel in [VERIFIERS, ADP] {
            std::fs::create_dir_all(job.path().join("t__1/trainer")).unwrap();
            std::fs::write(job.path().join("t__1").join(rel), &lines).unwrap();
        }
        assert!(write_job(job.path()).is_empty());
        for out in [JOB_VERIFIERS, JOB_ADP] {
            let text = std::fs::read_to_string(job.path().join(out)).unwrap();
            assert_eq!(text.lines().count(), 3, "{text}");
            for l in text.lines() {
                serde_json::from_str::<Value>(l).unwrap_or_else(|e| panic!("{out}: {e}: {l}"));
            }
            assert!(!text.contains(body), "{text}");
            // The message between the markers is still there.
            assert!(text.contains("grep -n --"), "{text}");
        }
    }

    /// Review FIX9-A 1: a key read in three chunks (`head`, `sed -n`,
    /// `tail`) has a middle chunk with no marker. The job pass's walk opens a
    /// span at the BEGIN in one message and removes the chunk's lines before
    /// the END in a later one. The key here is made up; `tests/trainer_keys.rs`
    /// covers the shapes the exporters write.
    #[test]
    fn job_files_drop_a_key_chunk_between_messages() {
        let job = tempfile::tempdir().unwrap();
        let body: Vec<String> = (0..7).map(|i| format!("{}{i}", "Zq".repeat(32))).collect();
        let record = json!({"prompt": [
            {"role": "tool", "content": format!("-----BEGIN RSA PRIVATE KEY-----\n{}\n{}\n", body[0], body[1])},
            {"role": "assistant", "content": "Reading the rest of the key."},
            {"role": "tool", "content": format!("{}\n{}\n{}\n", body[2], body[3], body[4])},
            {"role": "tool", "content": format!("{}\n{}\n-----END RSA PRIVATE KEY-----\n", body[5], body[6])},
        ]});
        let line = benchflow_agent::trajectory::to_python_json(&record) + "\n";
        for rel in [VERIFIERS, ADP] {
            std::fs::create_dir_all(job.path().join("t__1/trainer")).unwrap();
            std::fs::write(job.path().join("t__1").join(rel), &line).unwrap();
        }
        assert!(write_job(job.path()).is_empty());
        for out in [JOB_VERIFIERS, JOB_ADP] {
            let text = std::fs::read_to_string(job.path().join(out)).unwrap();
            assert_eq!(text.lines().count(), 1, "{text}");
            serde_json::from_str::<Value>(text.trim_end())
                .unwrap_or_else(|e| panic!("{out}: {e}: {text}"));
            for chunk in &body {
                assert!(!text.contains(chunk.as_str()), "{text}");
            }
        }
    }

    #[test]
    fn job_files_concatenate_trials_in_name_order() {
        let job = tempfile::tempdir().unwrap();
        for (name, text) in [
            ("b__2", "{\"b\": 1}\n"),
            (
                "a__1",
                "{\"a\": \"sk-ant-api03-abcdefghijklmnopqrstuvwxyz\"}",
            ),
        ] {
            std::fs::create_dir_all(job.path().join(name).join("trainer")).unwrap();
            std::fs::write(job.path().join(name).join(VERIFIERS), text).unwrap();
        }
        assert!(write_job(job.path()).is_empty());
        assert_eq!(
            std::fs::read_to_string(job.path().join(JOB_VERIFIERS)).unwrap(),
            "{\"a\": \"***REDACTED***\"}\n{\"b\": 1}\n"
        );
        // No trial has an ADP file, so there is no job-level one.
        assert!(!job.path().join(JOB_ADP).exists());
    }
}
