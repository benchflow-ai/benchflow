//! Differential tests of rubric scoring and CTRF matching against task-md's
//! reference (`tools/rubrics.py`: `score`, `matching_tests`, `test_verdict`),
//! on the random battery `make_battery.py` records in
//! `tests/fixtures/taskmd/golden/scoring-battery.json`.

use std::collections::BTreeMap;
use std::path::Path;

use benchflow_task::ctrf::{CtrfTest, matching_tests, test_verdict};
use benchflow_task::rubric::{Consequence, Rubric, ScoreError, Verdict};
use serde_json::Value;

fn battery() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/taskmd/golden/scoring-battery.json");
    serde_json::from_str(
        &std::fs::read_to_string(&path).expect("the battery: run tests/fixtures/taskmd/regen.sh"),
    )
    .expect("battery JSON")
}

fn verdict(v: &Value) -> Verdict {
    match v {
        Value::Bool(true) => Verdict::Pass,
        Value::Bool(false) => Verdict::Fail,
        Value::String(s) => Verdict::Level(s.clone()),
        Value::Number(n) => Verdict::Value(n.clone()),
        other => panic!("not a verdict: {other}"),
    }
}

fn consequence(v: &Value) -> Consequence {
    match v {
        Value::String(s) if s == "fail" => Consequence::Fail,
        Value::String(_) => Consequence::Report,
        Value::Object(m) => {
            Consequence::Penalty(m["penalty"].as_number().cloned().expect("a penalty"))
        }
        other => panic!("not a consequence: {other}"),
    }
}

#[test]
fn score_equals_the_reference_scorer() {
    let battery = battery();
    let cases = battery["score"].as_array().expect("score cases");
    assert!(cases.len() >= 1000);
    let mut failures = Vec::new();
    let mut errors = 0;
    for (n, case) in cases.iter().enumerate() {
        let rubric = Rubric::from_value(&case["rubric"])
            .unwrap_or_else(|d| panic!("case {n}: the rubric does not check: {d:#?}"));
        let verdicts: BTreeMap<String, Verdict> = case["verdicts"]
            .as_object()
            .expect("verdicts")
            .iter()
            .map(|(k, v)| (k.clone(), verdict(v)))
            .collect();
        let behaviors: Vec<Consequence> = case["behaviors"]
            .as_array()
            .expect("behaviors")
            .iter()
            .map(consequence)
            .collect();
        let want = &case["expected"];
        let got = rubric.try_score_with(&verdicts, &behaviors);
        match (want.get("error").and_then(Value::as_str), got) {
            (Some("ValueError"), Err(ScoreError::NotALevel { .. })) => errors += 1,
            (Some(e), got) => failures.push(format!(
                "case {n}: the reference raised {e}, rust gave {got:?}"
            )),
            (None, Err(e)) => failures.push(format!("case {n}: rust failed: {e}")),
            (None, Ok(s)) => {
                let exact = s.exact().expect("small numbers score exactly");
                let got = serde_json::json!({
                    "raw": exact.raw, "reward": exact.partial, "partial": exact.partial, "earned": exact.earned,
                    "maximum": exact.maximum, "penalty": exact.penalty, "strict": s.strict, "failed": s.failed_gates,
                });
                if &got != want {
                    failures.push(format!(
                        "case {n}:\n  reference {want}\n  rust      {got}\n  rubric {}",
                        case["rubric"]
                    ));
                }
                // The headline is the partial score, or the strict one.
                let headline = case["rubric"]["scoring"]["headline"].as_str();
                let expected_reward = if headline == Some("strict") {
                    if s.strict { 1.0 } else { 0.0 }
                } else {
                    s.partial
                };
                assert_eq!(s.reward, expected_reward, "case {n}");
            }
        }
    }
    assert!(errors > 0, "the battery should hold refused levels");
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn matching_equals_the_reference() {
    let battery = battery();
    let cases = battery["matching"].as_array().expect("matching cases");
    let mut failures = Vec::new();
    for (n, case) in cases.iter().enumerate() {
        let tests: Vec<CtrfTest> = case["tests"]
            .as_array()
            .expect("tests")
            .iter()
            .map(|t| CtrfTest {
                name: t["name"].as_str().unwrap().into(),
                status: t["status"].as_str().unwrap().into(),
                file_path: t
                    .get("filePath")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                duration: None,
                message: None,
            })
            .collect();
        for c in case["checks"].as_array().expect("checks") {
            let check = c["check"].as_str().unwrap();
            let got: Vec<&str> = matching_tests(check, &tests)
                .iter()
                .map(|t| t.name.as_str())
                .collect();
            let want: Vec<&str> = c["matched"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            if got != want || test_verdict(check, &tests) != c["verdict"].as_bool().unwrap() {
                failures.push(format!(
                    "case {n}, check {check:?}: reference {want:?} {}, rust {got:?}",
                    c["verdict"]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The score a skip, a missing verdict, and an unknown level get: nothing
/// earned or charged, never a pass (BenchFlow 0.8's `skip_value`).
#[test]
fn skips_score_nothing() {
    let rubric = Rubric::from_value(&serde_json::json!({
        "$schema": "https://task.md/schema/rubric-1.json",
        "criteria": [
            {"id": "g", "gate": true, "judge": "test", "check": "t", "text": "G.", "implicit": "x"},
            {"id": "p", "points": 2, "judge": "test", "check": "t", "text": "P.", "implicit": "x"},
            {"id": "n", "points": -1, "judge": "test", "check": "t", "text": "N.", "implicit": "x"},
            {"id": "l", "points": 4, "judge": "agent", "evidence": ["r.md"], "levels": {"0": "a", "4": "b"}, "text": "L.", "implicit": "x"}
        ]
    }))
    .unwrap();
    let verdicts = BTreeMap::from([
        ("g".to_string(), Verdict::Pass),
        ("p".to_string(), Verdict::Skip),
        ("l".to_string(), Verdict::Level("2".into())),
    ]);
    let s = rubric.score(&verdicts);
    assert_eq!(s.exact().unwrap().earned, "0");
    assert_eq!(s.exact().unwrap().partial, "0");
    assert!(s.criteria.iter().all(|c| c.id == "g" || c.score.is_none()));
    assert!(
        s.criteria[3]
            .problem
            .as_deref()
            .unwrap()
            .contains("not one of its levels")
    );
    let review = s.review_json();
    assert_eq!(review["summary"]["skip"], 3);
}
