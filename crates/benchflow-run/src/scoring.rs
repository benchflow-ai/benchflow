//! Turning a downloaded `/logs/verifier` folder into a reward.
//!
//! Two contracts: a reward file (`reward.json` before `reward.txt`, read by
//! [`benchflow_task::reward::read_rewards`] with the rules BenchFlow 0.8 and
//! Harbor share), or, for task.md packages with a rubric, the verifier's
//! CTRF report scored against `verifier/rubric.json` by
//! [`benchflow_task::rubric::Rubric::grade`]. A rubric's reward is computed,
//! so a reward file the script also wrote is kept aside, not used.

use std::collections::BTreeMap;
use std::path::Path;

use benchflow_task::TaskPackage;
use benchflow_task::ctrf::Ctrf;
use benchflow_task::reward::{RewardError, read_rewards};
use serde_json::Value;

use crate::judge::Judged;
use crate::outcome::{Owner, Phase, Problem};

/// A scored verifier run.
#[derive(Debug, Clone)]
pub struct Score {
    pub reward: f64,
    pub values: BTreeMap<String, f64>,
    pub passed: bool,
    /// The rubric's review, when a rubric scored the trial.
    pub rubric: Option<Value>,
}

/// Score a trial from its downloaded verifier folder. `judged` holds the
/// model judge's verdicts on the rubric's `llm` criteria, and its model.
pub fn score(
    task: &TaskPackage,
    verifier_dir: &Path,
    judged: Option<(&Judged, &str)>,
) -> Result<Score, Problem> {
    if let Some(rubric) = &task.verifier.rubric {
        let ctrf = Ctrf::from_path(&verifier_dir.join("ctrf.json"));
        let empty = Judged::default();
        let (j, model) = judged.unwrap_or((&empty, ""));
        let mut s = rubric.grade(&ctrf, &j.verdicts);
        for c in &mut s.criteria {
            if let Some(reason) = j.reasons.get(&c.id) {
                c.rationale = Some(reason.chars().take(600).collect());
            }
        }
        let mut review = s.review_json();
        if let Some(verdicts) = review.get_mut("verdicts").and_then(Value::as_array_mut) {
            for v in verdicts {
                let id = v.get("id").and_then(Value::as_str).unwrap_or("");
                if j.verdicts.contains_key(id)
                    && let Some(judge) = v.get_mut("judge").and_then(Value::as_object_mut)
                {
                    judge.insert("model".into(), Value::String(model.to_string()));
                }
            }
        }
        if judged.is_some() {
            review["x-benchflow"]["judge"] = serde_json::json!({
                "model": model,
                "calls": j.spend.calls,
                "input_tokens": j.spend.input_tokens,
                "output_tokens": j.spend.output_tokens,
                "cost_usd": j.spend.cost_usd,
                "record": "verifier/judgments.jsonl",
            });
        }
        let _ = std::fs::write(
            verifier_dir.join("review.json"),
            serde_json::to_string_pretty(&review).unwrap_or_default(),
        );
        for name in ["reward.txt", "reward.json"] {
            let p = verifier_dir.join(name);
            if p.is_file() {
                let _ = std::fs::rename(&p, verifier_dir.join(format!("script-{name}")));
            }
        }
        let _ = std::fs::write(verifier_dir.join("reward.txt"), format!("{}\n", s.reward));
        let _ = std::fs::write(
            verifier_dir.join("reward.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                // Numbers, as 0.8 writes them: its reward validator refuses a
                // bool, so 0.8 could not re-read a bench-scored trial.
                "reward": s.reward, "partial": s.partial, "strict": if s.strict { 1.0 } else { 0.0 },
            }))
            .unwrap_or_default(),
        );
        let mut values = BTreeMap::new();
        values.insert("reward".to_string(), s.reward);
        values.insert("partial".to_string(), s.partial);
        values.insert("strict".to_string(), if s.strict { 1.0 } else { 0.0 });
        return Ok(Score {
            reward: s.reward,
            values,
            passed: s.strict,
            rubric: Some(review),
        });
    }
    match read_rewards(verifier_dir) {
        Ok(r) => Ok(Score {
            reward: r.reward,
            passed: r.reward == 1.0,
            values: r.values,
            rubric: None,
        }),
        Err(e) => Err(reward_problem(&e, verifier_dir)),
    }
}

/// A reward that is missing or unreadable. Paths are given from the trial
/// folder (`verifier/reward.txt`), as people find them there.
fn reward_problem(e: &RewardError, verifier_dir: &Path) -> Problem {
    let (code, message) = match e {
        RewardError::NotFound { .. } => (
            "R301",
            "the verifier wrote no reward (no reward.json or reward.txt in /logs/verifier)"
                .to_string(),
        ),
        other => (
            "R302",
            other
                .to_string()
                .replace(&format!("{}/", verifier_dir.display()), "verifier/"),
        ),
    };
    Problem::new(code, Owner::Task, Phase::Verify, message, e.hint())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for (n, c) in files {
            std::fs::write(d.path().join(n), c).unwrap();
        }
        d
    }

    /// Guards the D5 fix: a rubric-scored trial's reward.json said
    /// `"strict": true`; 0.8 writes 1.0 and its validator refuses a bool.
    #[test]
    fn a_rubric_writes_a_numeric_reward_json() {
        let task = benchflow_task::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello"),
        )
        .unwrap();
        let d = dir_with(&[(
            "ctrf.json",
            r#"{"results":{"tests":[{"name":"test_hello","status":"passed"}]}}"#,
        )]);
        let s = score(&task, d.path(), None).unwrap();
        assert!(s.passed);
        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(d.path().join("reward.json")).unwrap())
                .unwrap();
        assert_eq!(written["strict"], 1.0);
        assert!(
            written.as_object().unwrap().values().all(Value::is_number),
            "{written}"
        );
    }

    #[test]
    fn missing_and_bad_rewards_are_owned_by_the_task() {
        let empty = dir_with(&[]);
        let p = reward_problem(&read_rewards(empty.path()).unwrap_err(), empty.path());
        assert_eq!(p.code, "R301");
        assert_eq!(p.owner, Owner::Task);
        let bad = dir_with(&[("reward.txt", "yes")]);
        let p = reward_problem(&read_rewards(bad.path()).unwrap_err(), bad.path());
        assert_eq!(p.code, "R302");
        // The path is the one people find in the trial folder.
        assert!(
            p.message.starts_with("verifier/reward.txt is not a reward"),
            "{}",
            p.message
        );
    }
}
