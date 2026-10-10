use base64::Engine;
use serde_json::{Value, json};

use super::*;

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../../testdata/redaction-0.8.json")).unwrap()
}

/// A fixture value: base64 of the text's bytes, each XOR 0x5A, so no token
/// shape appears in the file even to a scanner that decodes base64.
fn decode(value: &Value) -> String {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(value.as_str().unwrap())
        .unwrap();
    String::from_utf8(bytes.into_iter().map(|b| b ^ 0x5A).collect()).unwrap()
}

/// Every case of 0.8's redaction tests, plus look-around edge cases, run
/// through 0.8's `redact_trajectory_text` (testdata/redaction-0.8.json).
#[test]
fn text_redaction_matches_0_8() {
    let redactor = Redactor::default();
    let fixture = fixture();
    let rows = fixture["text"].as_array().unwrap();
    assert!(rows.len() > 200);
    for row in rows {
        let input = decode(&row["input_x"]);
        let expected = decode(&row["expected_x"]);
        assert_eq!(redactor.redact(&input), expected, "input: {input:?}");
    }
}

/// 0.8's `redact_trajectory_obj`: string fields are redacted with their key
/// in view.
#[test]
fn object_redaction_matches_0_8() {
    let redactor = Redactor::default();
    for row in fixture()["objects"].as_array().unwrap() {
        let mut value: Value = serde_json::from_str(&decode(&row["input_x"])).unwrap();
        let expected: Value = serde_json::from_str(&decode(&row["expected_x"])).unwrap();
        let input = value.clone();
        redactor.redact_json(&mut value);
        assert_eq!(value, expected, "input: {input}");
    }
}

#[test]
fn debug_output_counts_secrets_without_showing_them() {
    let redactor = Redactor::new(["login-secret-value-0042"]);
    let shown = format!("{redactor:?}");
    assert!(!shown.contains("login-secret-value-0042"), "{shown}");
    assert!(shown.contains("<1 values>"), "{shown}");
}

#[tokio::test]
async fn a_writer_does_not_show_its_secrets_either() {
    let dir = tempfile::tempdir().unwrap();
    let writer = crate::trajectory::TrajectoryWriter::new(
        dir.path().join("t.jsonl"),
        Redactor::new(["login-secret-value-0042"]),
    )
    .await
    .unwrap();
    assert!(!format!("{writer:?}").contains("login-secret-value-0042"));
}

#[test]
fn exact_secrets_go_whole_wherever_they_appear() {
    let redactor = Redactor::new(["oauth-FAKE-1234", "oauth-FAKE-1234-and-more"]).exact_only();
    assert_eq!(
        redactor.redact("a oauth-FAKE-1234-and-more b oauth-FAKE-1234 c"),
        "a ***REDACTED*** b ***REDACTED*** c"
    );
    let mut value = json!({"text": "token: oauth-FAKE-1234", "list": ["oauth-FAKE-1234"], "n": 3});
    redactor.redact_json(&mut value);
    assert_eq!(
        value,
        json!({"text": "token: ***REDACTED***", "list": ["***REDACTED***"], "n": 3})
    );
    // Untouched text is not copied.
    assert!(matches!(redactor.redact("nothing here"), Cow::Borrowed(_)));
}

#[test]
fn secrets_and_patterns_combine() {
    let redactor = Redactor::new(["plain-login-value-777"]);
    let mut value = json!({
        "type": "tool_call",
        "raw_output": "env: CLAUDE_CODE_OAUTH_TOKEN=plain-login-value-777\nGITHUB_TOKEN=abc",
        "x-api-key": "prefixless",
    });
    redactor.redact_json(&mut value);
    assert_eq!(
        value,
        json!({
            "type": "tool_call",
            "raw_output": "env: CLAUDE_CODE_OAUTH_TOKEN=***REDACTED***\nGITHUB_TOKEN=***REDACTED***",
            "x-api-key": "***REDACTED***",
        })
    );
}

#[test]
fn json_escapes_stay_valid() {
    // A secret next to a backslash escape must not leave a lone backslash.
    let redactor = Redactor::new(["s3cr3t-value-0001"]);
    let mut value = json!({"text": "path\\s3cr3t-value-0001\\x \"quoted\""});
    redactor.redact_json(&mut value);
    let line = crate::trajectory::jsonl::to_python_json(&value);
    let back: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(back["text"], "path\\***REDACTED***\\x \"quoted\"");
}

/// A larger corpus in the fixture's shape, from
/// `$BENCHFLOW_REDACTION_CORPUS` (random strings built from the patterns'
/// tricky fragments, with 0.8's output). Not checked in; without the
/// variable this test only says so.
#[test]
fn redaction_matches_0_8_on_an_external_corpus() {
    let Ok(path) = std::env::var("BENCHFLOW_REDACTION_CORPUS") else {
        eprintln!("set BENCHFLOW_REDACTION_CORPUS to a corpus file to compare more cases with 0.8");
        return;
    };
    let corpus: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let redactor = Redactor::default();
    let rows = corpus["text"].as_array().unwrap();
    let mismatches: Vec<String> = rows
        .iter()
        .filter_map(|row| {
            let input = decode(&row["input_x"]);
            let expected = decode(&row["expected_x"]);
            let ours = redactor.redact(&input);
            (ours != expected)
                .then(|| format!("input {input:?}\n  ours {ours:?}\n  0.8  {expected:?}"))
        })
        .collect();
    eprintln!("{} of {} cases differ", mismatches.len(), rows.len());
    assert!(
        mismatches.is_empty(),
        "{}",
        mismatches
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
