//! Trainer files rebuilt from real BenchFlow 0.8 rollouts, compared with
//! the files 0.8 wrote.
//!
//! Each rollout folder is rebuilt from what it keeps: its
//! `trajectory/acp_trajectory.jsonl`, `prompts.json`, and `result.json`
//! (rewards, verifier error, model, agent name, token totals).
//!
//! - `trainer/verifiers.jsonl` and `trainer/adp.jsonl` must match byte for
//!   byte.
//! - `trainer/atif.json` is the rewrite's ATIF v1.8, which adds step
//!   timestamps and the agent's tool names, arguments and raw output to
//!   0.8's v1.7. It is compared twice: the file as written must equal
//!   0.8's once the schema version is set back to v1.7 whenever the
//!   trajectory has nothing v1.8 adds (no tool call, no receipt); and the
//!   document built from the events without those inputs must equal 0.8's
//!   byte for byte, version aside, always.
//!
//! `tests/fixtures/trainer-0.8/` holds a few rollouts (copies; credentials
//! checked for). The full sweep over every 0.8 rollout on this machine is
//! ignored by default:
//!
//! ```text
//! BENCHFLOW_GOLDEN_ROOT=~/benchflow cargo test -p benchflow-run --test trainer_golden -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use benchflow_agent::trajectory::Event;
use benchflow_run::trainer::{
    self, ADP, ATIF, ATIF_COVERAGE, JOB_ADP, JOB_VERIFIERS, RolloutInfo, VERIFIERS,
};
use serde_json::Value;

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// What 0.8 passed its trainer writer, read back from the rollout folder.
fn info_of(dir: &Path) -> RolloutInfo {
    let result = read_json(&dir.join("result.json"));
    let s = |v: &Value| v.as_str().filter(|s| !s.is_empty()).map(str::to_string);
    let usage = &result["agent_result"];
    RolloutInfo {
        task_name: s(&result["task_name"]).unwrap(),
        rollout_name: s(&result["rollout_name"]).unwrap(),
        agent_name: s(&result["agent_name"])
            .or_else(|| s(&result["agent"]))
            .unwrap_or_default(),
        prompts: serde_json::from_value(read_json(&dir.join("prompts.json"))).unwrap(),
        rewards: result["rewards"].as_object().cloned(),
        model: result["model"].as_str().map(str::to_string),
        verifier_error: s(&result["verifier_error"]),
        total_prompt_tokens: usage["n_input_tokens"].as_u64(),
        total_completion_tokens: usage["n_output_tokens"].as_u64(),
        total_cached_tokens: usage["n_cache_read_tokens"].as_u64(),
        total_cost_usd: usage["cost_usd"].as_f64(),
    }
}

/// The events without what only v1.8 ATIF reads: receipts, the agent's
/// tool names, and (unless a subagent needs its spawn prompt) the tool
/// call's raw input and output.
fn without_v1_8_inputs(events: &[Event]) -> Vec<Event> {
    let mut json: Vec<Value> = events.iter().map(Event::to_json).collect();
    let attributed = json.iter().any(|e| {
        e["parent_tool_call_id"]
            .as_str()
            .is_some_and(|p| !p.is_empty())
    });
    for event in &mut json {
        let map = event.as_object_mut().unwrap();
        for key in [
            "receipt",
            "ts",
            "started_at",
            "finished_at",
            "tool_name",
            "raw_output",
        ] {
            map.remove(key);
        }
        if !attributed {
            map.remove("raw_input");
        }
    }
    json.into_iter()
        .map(|v| match v {
            Value::Object(map) => Event::from_json(map),
            _ => unreachable!(),
        })
        .collect()
}

fn has_v1_8_inputs(events: &[Event]) -> bool {
    events.iter().any(|e| {
        let json = e.to_json();
        matches!(e, Event::ToolCall(_)) || json.get("receipt").is_some()
    })
}

/// An event of a kind ATIF exports that this build keeps as written.
fn odd(event: &Event) -> bool {
    matches!(event, Event::Other(map) if matches!(
        map.get("type").and_then(Value::as_str),
        Some("user_message" | "agent_message" | "agent_thought" | "tool_call" | "notice")
    ))
}

fn as_v1_7(text: &str) -> String {
    text.replace(
        "\"schema_version\": \"ATIF-v1.8\"",
        "\"schema_version\": \"ATIF-v1.7\"",
    )
}

/// Where two texts first differ, with some context.
fn first_difference(ours: &str, theirs: &str) -> String {
    let at = ours
        .bytes()
        .zip(theirs.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(ours.len().min(theirs.len()));
    let window = |s: &str| {
        let mut start = at.saturating_sub(60).min(s.len());
        while !s.is_char_boundary(start) {
            start -= 1;
        }
        let mut end = (at + 60).min(s.len());
        while !s.is_char_boundary(end) {
            end += 1;
        }
        s[start..end].to_string()
    };
    format!(
        "at byte {at} (lengths {} vs {}):\n  ours:   {:?}\n  theirs: {:?}",
        ours.len(),
        theirs.len(),
        window(ours),
        window(theirs)
    )
}

#[derive(Default)]
struct Tally {
    rollouts: usize,
    /// Per file kind: agreements, and the differences found.
    agree: BTreeMap<&'static str, usize>,
    differ: BTreeMap<&'static str, Vec<String>>,
}

impl Tally {
    fn check(
        &mut self,
        kind: &'static str,
        dir: &Path,
        ours: Option<String>,
        theirs: Option<String>,
    ) {
        if ours == theirs {
            *self.agree.entry(kind).or_default() += 1;
            return;
        }
        let detail = match (&ours, &theirs) {
            (Some(o), Some(t)) => first_difference(o, t),
            (None, Some(_)) => "0.8 wrote it; we did not".into(),
            (Some(_), None) => "we wrote it; 0.8 did not".into(),
            (None, None) => unreachable!(),
        };
        self.differ
            .entry(kind)
            .or_default()
            .push(format!("{}: {detail}", dir.display()));
    }

    fn report(&self) -> String {
        let mut out = format!("{} rollouts\n", self.rollouts);
        let kinds: std::collections::BTreeSet<_> =
            self.agree.keys().chain(self.differ.keys()).collect();
        for kind in kinds {
            let differ = self.differ.get(kind).map_or(0, Vec::len);
            out.push_str(&format!(
                "  {kind}: {} agree, {differ} differ\n",
                self.agree.get(kind).copied().unwrap_or(0)
            ));
        }
        for (kind, diffs) in &self.differ {
            for d in diffs {
                out.push_str(&format!("DIFF {kind} {d}\n"));
            }
        }
        out
    }
}

fn read_opt(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Rebuild one rollout's trainer files into `out` and compare them with
/// the ones 0.8 wrote in `dir`.
fn check_rollout(dir: &Path, out: &Path, tally: &mut Tally) {
    tally.rollouts += 1;
    let info = info_of(dir);
    let events = trainer::read_events(dir).unwrap();
    let failures = trainer::write_rollout(
        out,
        &events,
        &info,
        &benchflow_agent::trajectory::Redactor::default(),
    );
    assert!(failures.is_empty(), "{}: {failures:?}", dir.display());
    for rel in [VERIFIERS, ADP] {
        tally.check(rel, dir, read_opt(&out.join(rel)), read_opt(&dir.join(rel)));
    }
    let theirs = read_opt(&dir.join(ATIF));
    let ours = read_opt(&out.join(ATIF));
    // `to_atif` reads the events this build models; one it keeps as
    // written (an odd shape) is not a step, where 0.8 read any dict.
    if events.iter().any(odd) {
        tally.check(
            "atif.json, trajectories with odd events",
            dir,
            ours.as_deref().map(as_v1_7),
            theirs,
        );
        tally.check(
            "atif.coverage.json, trajectories with odd events",
            dir,
            read_opt(&out.join(ATIF_COVERAGE)),
            read_opt(&dir.join(ATIF_COVERAGE)),
        );
        return;
    }
    tally.check(
        ATIF_COVERAGE,
        dir,
        read_opt(&out.join(ATIF_COVERAGE)),
        read_opt(&dir.join(ATIF_COVERAGE)),
    );
    if !has_v1_8_inputs(&events) {
        tally.check(
            "atif.json as written, version aside",
            dir,
            ours.as_deref().map(as_v1_7),
            theirs.clone(),
        );
    }
    // The 0.8 view: the same writer, without the inputs only v1.8 reads.
    let view = out.join("v1_7-view");
    let failures = trainer::write_rollout(
        &view,
        &without_v1_8_inputs(&events),
        &info,
        &benchflow_agent::trajectory::Redactor::default(),
    );
    assert!(failures.is_empty(), "{failures:?}");
    tally.check(
        "atif.json without v1.8 inputs, version aside",
        dir,
        read_opt(&view.join(ATIF)).as_deref().map(as_v1_7),
        theirs,
    );
}

/// Rebuild the job-level files from rebuilt rollouts and compare them with
/// 0.8's.
fn check_job(job: &Path, rebuilt: &Path, tally: &mut Tally) {
    let failures = trainer::write_job(rebuilt);
    assert!(failures.is_empty(), "{failures:?}");
    for rel in [JOB_VERIFIERS, JOB_ADP] {
        let theirs = read_opt(&job.join(rel));
        if theirs.is_none() {
            continue;
        }
        tally.check(
            if rel == JOB_VERIFIERS {
                "job verifiers.jsonl"
            } else {
                "job adp.jsonl"
            },
            job,
            read_opt(&rebuilt.join(rel)),
            theirs,
        );
    }
}

/// Check rollouts grouped by job folder.
fn check_all(rollouts: &[PathBuf]) -> Tally {
    let mut by_job: BTreeMap<PathBuf, Vec<&PathBuf>> = BTreeMap::new();
    for r in rollouts {
        by_job
            .entry(r.parent().unwrap().to_path_buf())
            .or_default()
            .push(r);
    }
    // BENCHFLOW_GOLDEN_KEEP=<dir> keeps the rebuilt files there.
    let scratch = tempfile::tempdir().unwrap();
    let base = std::env::var_os("BENCHFLOW_GOLDEN_KEEP")
        .map(PathBuf::from)
        .unwrap_or_else(|| scratch.path().to_path_buf());
    let mut tally = Tally::default();
    for (i, (job, dirs)) in by_job.iter().enumerate() {
        let rebuilt = base.join(i.to_string());
        for dir in dirs {
            check_rollout(dir, &rebuilt.join(dir.file_name().unwrap()), &mut tally);
        }
        // Only meaningful when every rollout of the job is here.
        let all_here = std::fs::read_dir(job)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().join(VERIFIERS).is_file())
            .count()
            == dirs.len();
        if all_here {
            // The v1.7 views are not trial folders' files; keep them out.
            for dir in dirs {
                let _ = std::fs::remove_dir_all(
                    rebuilt.join(dir.file_name().unwrap()).join("v1_7-view"),
                );
            }
            check_job(job, &rebuilt, &mut tally);
        }
    }
    tally
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/trainer-0.8")
}

/// The committed fixtures: every rollout and job agrees with 0.8.
#[test]
fn rebuilt_trainer_files_match_0_8_on_the_fixtures() {
    let mut rollouts = Vec::new();
    for job in std::fs::read_dir(fixtures()).unwrap() {
        let job = job.unwrap().path();
        if !job.is_dir() {
            continue;
        }
        for r in std::fs::read_dir(&job).unwrap() {
            let r = r.unwrap().path();
            if r.join(VERIFIERS).is_file() {
                rollouts.push(r);
            }
        }
    }
    rollouts.sort();
    assert!(rollouts.len() >= 8, "fixtures missing: {}", rollouts.len());
    let tally = check_all(&rollouts);
    println!("{}", tally.report());
    assert!(tally.differ.is_empty(), "{}", tally.report());
    for kind in [
        VERIFIERS,
        ADP,
        "atif.json without v1.8 inputs, version aside",
    ] {
        assert_eq!(tally.agree[kind], rollouts.len(), "{kind}");
    }
    assert!(tally.agree["job verifiers.jsonl"] >= 1);
}

/// Every 0.8 rollout with trainer files under `BENCHFLOW_GOLDEN_ROOT`
/// (default `~/benchflow`), at most 7 levels down; read in place.
#[test]
#[ignore = "reads 0.8 job folders on this machine"]
fn rebuilt_trainer_files_match_0_8_on_every_local_rollout() {
    let root = std::env::var_os("BENCHFLOW_GOLDEN_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::home_dir().unwrap().join("benchflow"));
    let mut rollouts: Vec<PathBuf> = walkdir::WalkDir::new(&root)
        .max_depth(7)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(name == "target" || name == ".git" || name == "node_modules" || name == "fixtures")
        })
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name() == "verifiers.jsonl"
                && e.path().parent().is_some_and(|p| p.ends_with("trainer"))
        })
        .map(|e| e.path().parent().unwrap().parent().unwrap().to_path_buf())
        .filter(|r| r.join("result.json").is_file())
        .collect();
    rollouts.sort();
    let tally = check_all(&rollouts);
    println!("{}", tally.report());
    if let Ok(path) = std::env::var("BENCHFLOW_GOLDEN_REPORT") {
        std::fs::write(path, tally.report()).unwrap();
    }
    assert!(!tally.differ.contains_key(VERIFIERS) && !tally.differ.contains_key(ADP));
}
