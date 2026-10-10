//! Reading jobs back from disk: the Rust engine's and BenchFlow 0.8's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local, NaiveDateTime, TimeZone};
use serde_json::Value;

use crate::agent::Usage;
use crate::outcome::{Defenses, Owner, Phase, Problem, Status};
use crate::record::{Custody, Timing, TrialRecord};

/// A job folder read back.
#[derive(Debug, Clone)]
pub struct JobView {
    pub dir: PathBuf,
    pub name: String,
    pub summary: Option<Value>,
    pub evaluation: Option<Value>,
    pub records: Vec<TrialRecord>,
}

impl JobView {
    pub fn agent(&self) -> String {
        self.summary
            .as_ref()
            .and_then(|s| s.get("agent"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.records.first().map(|r| r.agent.clone()))
            .unwrap_or_else(|| "?".into())
    }
    pub fn model(&self) -> Option<String> {
        self.summary
            .as_ref()
            .and_then(|s| s.get("model"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.records.iter().find_map(|r| r.model.clone()))
    }
    pub fn modified(&self) -> Option<std::time::SystemTime> {
        std::fs::metadata(&self.dir).and_then(|m| m.modified()).ok()
    }
}

/// True when `dir` looks like a job folder.
pub fn is_job(dir: &Path) -> bool {
    dir.join("evaluation.json").is_file()
        || dir.join("summary.json").is_file()
        || trial_dirs(dir).next().is_some()
}

/// The trial folders directly under a job folder.
pub fn trial_dirs(job: &Path) -> impl Iterator<Item = PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(job)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("result.json").is_file())
        .collect();
    v.sort();
    v.into_iter()
}

/// Every job folder under `root` (or `root` itself), newest last.
pub fn find_jobs(root: &Path) -> Vec<PathBuf> {
    if root.join("result.json").is_file() {
        return vec![root.to_path_buf()];
    }
    if is_job(root) && trial_dirs(root).next().is_some() {
        return vec![root.to_path_buf()];
    }
    let mut out = Vec::new();
    for e in walkdir::WalkDir::new(root)
        .min_depth(1)
        .max_depth(3)
        .into_iter()
        .filter_entry(|e| {
            let n = e.file_name().to_string_lossy();
            !n.starts_with('.') || e.depth() == 0
        })
        .filter_map(Result::ok)
    {
        if e.file_type().is_dir()
            && (e.path().join("evaluation.json").is_file()
                || e.path().join("summary.json").is_file())
        {
            out.push(e.path().to_path_buf());
        }
    }
    out.sort();
    out
}

/// Read one job folder (or one trial folder, as a job of one).
pub fn read_job(dir: &Path) -> JobView {
    let read = |name: &str| {
        std::fs::read_to_string(dir.join(name))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
    };
    let records: Vec<TrialRecord> = if dir.join("result.json").is_file() {
        read_trial(dir).into_iter().collect()
    } else {
        trial_dirs(dir).filter_map(|d| read_trial(&d)).collect()
    };
    JobView {
        dir: dir.to_path_buf(),
        name: dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        summary: read("summary.json"),
        evaluation: read("evaluation.json"),
        records,
    }
}

/// Read a job's `lock.json`.
pub fn read_lock(job: &Path) -> std::io::Result<crate::lock::Lock> {
    let text = std::fs::read_to_string(job.join(crate::lock::FILE_NAME))?;
    serde_json::from_str(&text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Read one trial folder.
pub fn read_trial(dir: &Path) -> Option<TrialRecord> {
    let text = std::fs::read_to_string(dir.join("result.json")).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    record_from_json(&v, dir)
}

fn parse_time(v: Option<&Value>) -> DateTime<Local> {
    let s = v.and_then(Value::as_str).unwrap_or("");
    for fmt in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(t) = NaiveDateTime::parse_from_str(s, fmt)
            && let Some(l) = Local.from_local_datetime(&t).single()
        {
            return l;
        }
    }
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return t.with_timezone(&Local);
    }
    Local
        .timestamp_opt(0, 0)
        .single()
        .unwrap_or_else(Local::now)
}

/// Build a record from a result.json, written by either engine.
pub fn record_from_json(v: &Value, dir: &Path) -> Option<TrialRecord> {
    let str_of = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let rewards: Option<BTreeMap<String, f64>> =
        v.get("rewards").and_then(Value::as_object).map(|m| {
            m.iter()
                .filter_map(|(k, x)| x.as_f64().map(|n| (k.clone(), n)))
                .collect()
        });
    let reward = rewards.as_ref().and_then(|r| r.get("reward").copied());
    let error = str_of("error");
    let verifier_error = str_of("verifier_error");
    let status: Status = v
        .get("status")
        .and_then(|s| serde_json::from_value(s.clone()).ok())
        .unwrap_or(if reward.is_some() {
            Status::Scored
        } else if error.is_some() {
            Status::Errored
        } else if verifier_error.is_some() {
            Status::Unscored
        } else {
            Status::Errored
        });
    let problem: Option<Problem> = v
        .get("problem")
        .and_then(|p| serde_json::from_value(p.clone()).ok())
        .or_else(|| {
            let msg = error.clone().or(verifier_error.clone())?;
            let unscored = error.is_none();
            Some(Problem::new(
                if unscored { "R304" } else { "R901" },
                if unscored { Owner::Task } else { Owner::Agent },
                if unscored {
                    Phase::Verify
                } else {
                    Phase::Agent
                },
                msg,
                "see the trial folder",
            ))
        });
    let passed = v
        .get("passed")
        .and_then(Value::as_bool)
        .or_else(|| reward.map(|r| r == 1.0));
    let ar = v.get("agent_result").cloned().unwrap_or(Value::Null);
    let u = |k: &str| ar.get(k).and_then(Value::as_u64).unwrap_or(0);
    let usage = Usage {
        input_tokens: u("n_input_tokens"),
        output_tokens: u("n_output_tokens"),
        cache_read_tokens: u("n_cache_read_tokens"),
        cache_write_tokens: u("n_cache_creation_tokens"),
        cost_usd: ar.get("cost_usd").and_then(Value::as_f64),
        cost_source: ar
            .get("cost_source")
            .and_then(Value::as_str)
            .map(str::to_string),
        cost_note: ar
            .get("cost_note")
            .and_then(Value::as_str)
            .map(str::to_string),
        source: ar
            .get("usage_source")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    };
    let timing: Timing = v
        .get("timing")
        .and_then(|t| serde_json::from_value(t.clone()).ok())
        .or_else(|| {
            std::fs::read_to_string(dir.join("timing.json"))
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
        })
        .unwrap_or_default();
    let defenses: Defenses = v
        .get("defenses")
        .and_then(|d| serde_json::from_value(d.clone()).ok())
        .unwrap_or_default();
    let custody: Custody = std::fs::read_to_string(dir.join("custody.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    let prompts: Vec<String> = std::fs::read_to_string(dir.join("prompts.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    Some(TrialRecord {
        task_name: str_of("task_name").unwrap_or_default(),
        task_id: str_of("task_id")
            .or_else(|| str_of("task_name"))
            .unwrap_or_default(),
        rollout_name: str_of("rollout_name").unwrap_or_else(|| {
            dir.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default()
        }),
        agent: str_of("agent")
            .or_else(|| str_of("agent_name"))
            .unwrap_or_default(),
        model: str_of("model"),
        status,
        rewards,
        reward,
        passed,
        timed_out: v.get("timed_out").and_then(Value::as_bool).unwrap_or(false)
            || v.get("agent_timeout_info").is_some_and(|x| !x.is_null()),
        problem,
        usage,
        tool_calls: v.get("n_tool_calls").and_then(Value::as_u64).unwrap_or(0),
        prompts,
        trajectory_summary: v.get("trajectory_summary").cloned().unwrap_or(Value::Null),
        timing,
        started_at: parse_time(v.get("started_at")),
        finished_at: parse_time(v.get("finished_at")),
        task_digest: str_of("task_digest").unwrap_or_default(),
        sandbox_id: str_of("sandbox_id"),
        sandbox_backend: String::new(),
        defenses,
        custody,
        login: str_of("login"),
        env_keys: Vec::new(),
        verifier_exit: v
            .get("verifier_exit_code")
            .and_then(Value::as_i64)
            .map(|n| n as i32),
        rubric: v.get("rubric").cloned().filter(|r| !r.is_null()),
        dataset: str_of("dataset_name").map(|name| crate::config::TrialDataset {
            name,
            version: str_of("dataset_version").unwrap_or_default(),
            registry_digest: str_of("registry_digest"),
        }),
        step_results: v.get("step_results").cloned().filter(|s| !s.is_null()),
        agent_session: v
            .get("agent_session")
            .and_then(|s| serde_json::from_value(s.clone()).ok()),
        resources_raised: v
            .get("resources_raised")
            .and_then(|s| serde_json::from_value(s.clone()).ok()),
    })
}
