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
            // A job whose first trial has not finished: what lock.json
            // planned.
            .or_else(|| self.planned("agent"))
            .unwrap_or_else(|| "?".into())
    }
    pub fn model(&self) -> Option<String> {
        self.summary
            .as_ref()
            .and_then(|s| s.get("model"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.records.iter().find_map(|r| r.model.clone()))
            .or_else(|| self.planned("model"))
    }
    /// A field of the first planned trial's attempt in lock.json, read
    /// loosely: what a job whose first trial has not finished will run.
    fn planned(&self, field: &str) -> Option<String> {
        let text = std::fs::read_to_string(self.dir.join(crate::lock::FILE_NAME)).ok()?;
        let lock: Value = serde_json::from_str(&text).ok()?;
        lock.pointer(&format!("/trials/0/attempt/{field}"))?
            .as_str()
            .map(str::to_string)
    }
    /// How many trials lock.json planned that have no record: never started
    /// (a drained or stopped run, or a login that ran out). 0 while the job
    /// is running, when its trials may still start.
    pub fn never_ran(&self) -> usize {
        if self.running() {
            return 0;
        }
        let Ok(text) = std::fs::read_to_string(self.dir.join(crate::lock::FILE_NAME)) else {
            return 0;
        };
        let Ok(lock) = serde_json::from_str::<Value>(&text) else {
            return 0;
        };
        let recorded: std::collections::BTreeSet<&str> = self
            .records
            .iter()
            .map(|r| r.rollout_name.as_str())
            .collect();
        lock.get("trials")
            .and_then(Value::as_array)
            .map(|trials| {
                trials
                    .iter()
                    .filter_map(|t| t.get("trial").and_then(Value::as_str))
                    .filter(|name| !recorded.contains(name))
                    .count()
            })
            .unwrap_or(0)
    }

    /// Whether a live `bench run` (or `bench regrade`) holds the job.
    pub fn running(&self) -> bool {
        crate::job::is_running(&self.dir)
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
            && let Some(l) = local(t)
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

/// A naive local time (0.8 writes `str(datetime.now())`) as CPython's
/// `datetime.fromisoformat(..).timestamp()` reads it in the machine's zone:
/// a port of `local_to_seconds` (Modules/_datetimemodule.c) with `fold=0`.
/// In a stretch repeated when clocks go back it is the first reading; in
/// one skipped when they go forward it is read with the UTC offset from
/// before the jump, so Lord Howe's `02:15` on its 30-minute jump is 02:45,
/// Troll's `01:30` on its 2-hour jump is 03:30, and Samoa's skipped
/// 2011-12-30 lands on the 31st.
fn local(t: NaiveDateTime) -> Option<DateTime<Local>> {
    // CPython's `local(u)`: the wall clock at instant u, both counted in
    // seconds as if the wall clock were UTC.
    let wall_at = |u: i64| -> Option<i64> {
        let utc = DateTime::from_timestamp(u, 0)?.naive_utc();
        Some(u + i64::from(Local.offset_from_utc_datetime(&utc).local_minus_utc()))
    };
    const MAX_FOLD_SECONDS: i64 = 24 * 3600;
    let t_secs = t.and_utc().timestamp();
    let a = wall_at(t_secs)? - t_secs;
    let u1 = t_secs - a;
    let t1 = wall_at(u1)?;
    let b = if t1 == t_secs {
        // A reading; look a day earlier for an earlier one (fold=0).
        let probe = u1 - MAX_FOLD_SECONDS;
        let b = wall_at(probe)? - probe;
        if a == b {
            return Local
                .timestamp_opt(u1, t.and_utc().timestamp_subsec_nanos())
                .single();
        }
        b
    } else {
        t1 - u1
    };
    let u2 = t_secs - b;
    let seconds = if wall_at(u2)? == t_secs {
        u2
    } else if t1 == t_secs {
        u1
    } else {
        // Neither offset gives this wall time: it is in a gap.
        u1.max(u2)
    };
    Local
        .timestamp_opt(seconds, t.and_utc().timestamp_subsec_nanos())
        .single()
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
        // result.json does not tell a carried time-out from the trial's own.
        own_timed_out: v.get("timed_out").and_then(Value::as_bool).unwrap_or(false)
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
        resumed_session: None,
        resources_raised: v
            .get("resources_raised")
            .and_then(|s| serde_json::from_value(s.clone()).ok()),
        environment_warnings: v
            .get("environment_warnings")
            .and_then(|s| serde_json::from_value(s.clone()).ok())
            .unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHILD: &str = "BENCHFLOW_TEST_NAIVE_TIME_ZONE";

    /// Runs only in the child process `in_zone` starts with `TZ` set (the
    /// zone is process-wide, and tests share one process): every case of
    /// that zone in the fixture, which CPython 3.12 wrote
    /// (`tests/fixtures/make_naive_local_times.py`), must read as the
    /// instant CPython's `.timestamp()` gave.
    #[test]
    fn naive_times_in_the_zone_of_this_process() {
        let Ok(zone) = std::env::var(CHILD) else {
            return;
        };
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/python-naive-local-times.json");
        let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut failures = Vec::new();
        let mut n = 0;
        for case in v["cases"].as_array().unwrap() {
            if case["zone"] != zone.as_str() {
                continue;
            }
            n += 1;
            let got = parse_time(case.get("naive"));
            let want = (
                case["seconds"].as_i64().unwrap(),
                case["micros"].as_u64().unwrap(),
            );
            let got_pair = (got.timestamp(), u64::from(got.timestamp_subsec_micros()));
            if got_pair != want {
                failures.push(format!(
                    "{}: read as {got} ({got_pair:?}), CPython {want:?}",
                    case["naive"]
                ));
            }
        }
        assert!(n > 100, "{n} cases for {zone}");
        assert!(
            failures.is_empty(),
            "{zone} (is its time zone data installed?): {} of {n} differ:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// The folders chrono 0.4 reads a zone named by `TZ` from (`find_tz_file`
    /// in its `offset/local/tz_info/timezone.rs`, on Linux and macOS alike);
    /// it does not read `TZDIR`.
    const ZONE_DIRS: [&str; 4] = [
        "/usr/share/zoneinfo",
        "/share/zoneinfo",
        "/etc/zoneinfo",
        "/usr/share/lib/zoneinfo",
    ];

    /// The file chrono reads for `TZ=zone` (a leading `:` and an absolute
    /// path as chrono takes them), if this machine has one in `dirs`.
    fn zone_file(zone: &str, dirs: &[&Path]) -> Option<PathBuf> {
        let path = Path::new(zone.strip_prefix(':').unwrap_or(zone));
        if path.is_absolute() {
            return path.is_file().then(|| path.to_path_buf());
        }
        dirs.iter().map(|d| d.join(path)).find(|p| p.is_file())
    }

    /// Runs `naive_times_in_the_zone_of_this_process` in a child process
    /// with `TZ=zone`, as the CLI test of the D5 fix (51f2a252) sets the
    /// zone of the `bench` it runs. Without the zone's file (a slim image
    /// with no tzdata), chrono reads every time in the machine's own zone or
    /// UTC, so the zone is skipped with a note instead. Returns whether it
    /// ran.
    fn in_zone(zone: &str) -> bool {
        if zone_file(zone, &ZONE_DIRS.map(Path::new)).is_none() {
            eprintln!(
                "skipped the naive times of {zone}: this machine has no time zone file for it in {} (install tzdata to run them)",
                ZONE_DIRS.join(", ")
            );
            return false;
        }
        let name = concat!(module_path!(), "::naive_times_in_the_zone_of_this_process");
        let name = name.split_once("::").unwrap().1;
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture", "--test-threads", "1"])
            .env("TZ", zone)
            .env(CHILD, zone)
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "{text}");
        assert!(text.contains("1 passed"), "the child ran no test: {text}");
        true
    }

    /// Guards the skip added after the D5b runtime review of cbf42071
    /// (finding 10): a zone this machine has no file for is skipped with a
    /// note, not failed, as on a slim CI image without tzdata.
    #[test]
    fn a_zone_without_its_file_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(zone_file("America/Los_Angeles", &[dir.path()]), None);
        std::fs::create_dir_all(dir.path().join("America")).unwrap();
        std::fs::write(dir.path().join("America/Los_Angeles"), "TZif").unwrap();
        let file = Some(dir.path().join("America/Los_Angeles"));
        assert_eq!(zone_file("America/Los_Angeles", &[dir.path()]), file);
        assert_eq!(zone_file(":America/Los_Angeles", &[dir.path()]), file);
        let absolute = dir.path().join("America/Los_Angeles");
        assert_eq!(zone_file(absolute.to_str().unwrap(), &[]), file);
        // No machine has this one: its child would read no case at all.
        assert!(!in_zone("Nowhere/No_Such_Zone"));
    }

    /// The D5 fix (51f2a252): a one-hour change, both ways.
    #[test]
    fn naive_times_around_a_one_hour_change_read_as_in_cpython() {
        in_zone("America/Los_Angeles");
    }

    /// A 30-minute change: `2026-10-04 02:15` read as 03:15 before D5b
    /// (CPython: 02:45).
    #[test]
    fn naive_times_around_a_30_minute_change_read_as_in_cpython() {
        in_zone("Australia/Lord_Howe");
    }

    /// A 2-hour change: `2026-03-29 01:30` was no time at all before D5b, so
    /// the trial showed as started in 1970 (CPython: 03:30).
    #[test]
    fn naive_times_around_a_2_hour_change_read_as_in_cpython() {
        in_zone("Antarctica/Troll");
    }

    /// Samoa skipped 2011-12-30: before D5b every time that day was 1970.
    #[test]
    fn naive_times_on_a_skipped_day_read_as_in_cpython() {
        in_zone("Pacific/Apia");
    }
}
