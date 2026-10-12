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
        let mut expected = decode(&row["expected_x"]);
        // One deliberate departure: the body next to an unpaired marker (or
        // one further than 8192 characters from its partner) goes too.
        if let Cow::Owned(halves) = redact_pem_halves(&expected) {
            expected = halves;
        }
        assert_eq!(redactor.redact(&input), expected, "input: {input:?}");
    }
}

/// Review E1-F-A 1, E1-F-B: an END half whose last body line is short (a
/// remainder; `AQIDBA==` ends every OpenSSH RSA-3072 key) was left whole.
/// E1-F-A 2: a value holding an END half and then a BEGIN half (two keys
/// read across) redacted neither. The bodies here are made up.
#[test]
fn half_keys_lose_their_bodies() {
    let r = Redactor::default();
    let line = |c: char, n: usize| std::iter::repeat_n(c, n).collect::<String>();
    let end_half = format!(
        "{}\n{}\nAQIDBA==\n-----END OPENSSH PRIVATE KEY-----\n",
        line('q', 70),
        line('Q', 70)
    );
    let out = r.redact(&end_half);
    assert!(
        !out.contains(&line('q', 70)) && !out.contains("AQIDBA=="),
        "{out}"
    );
    assert!(out.contains("-----END OPENSSH PRIVATE KEY-----"), "{out}");
    let two = format!(
        "{}\n{}\n-----END PRIVATE KEY-----\n-----BEGIN PRIVATE KEY-----\n{}\n{}",
        line('a', 64),
        line('b', 12),
        line('c', 64),
        line('d', 64)
    );
    let out = r.redact(&two);
    for body in [line('a', 64), line('b', 12), line('c', 64), line('d', 64)] {
        assert!(!out.contains(&body), "{out}");
    }
    assert!(
        out.contains("-----END PRIVATE KEY-----") && out.contains("-----BEGIN PRIVATE KEY-----")
    );
    // In a JSON field (the probe form), too.
    let mut v = json!({"content": two});
    r.redact_json(&mut v);
    let text = v["content"].as_str().unwrap();
    assert!(
        !text.contains(&line('a', 64)) && !text.contains(&line('d', 64)),
        "{text}"
    );
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

#[test]
fn gcs_signed_url_parameters_are_redacted() {
    let r = Redactor::default();
    let url = "https://storage.googleapis.com/b/o?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=svc%40p.iam.gserviceaccount.com%2F20261012&X-Goog-Signature=0a1b2c3d4e5f60718293a4b5c6d7e8f9";
    let out = r.redact(url);
    assert!(!out.contains("0a1b2c3d4e5f60718293a4b5c6d7e8f9"), "{out}");
    assert!(!out.contains("svc%40p.iam"), "{out}");
    assert!(out.contains("X-Goog-Algorithm=GOOG4-RSA-SHA256"), "{out}");
}

/// Review FIX7-B 2: a split key read through a line-numbering viewer
/// (`cat -n`, `nl`, `grep -n`, `    21→`) kept its whole body: the gutter
/// broke the chain of body lines. The bodies here are made up.
#[test]
fn half_keys_in_numbered_views_lose_their_bodies() {
    let r = Redactor::default();
    let line = |c: char| std::iter::repeat_n(c, 64).collect::<String>();
    let begin_half = format!(
        "    20\t-----BEGIN PRIVATE KEY-----\n    21\t{}\n    22\t{}\n",
        line('a'),
        line('b')
    );
    let end_half = format!(
        "    23\u{2192}{}\n    24\u{2192}AQIDBA==\n    25\u{2192}-----END PRIVATE KEY-----\n",
        line('c')
    );
    let grep_n = format!(
        "31:{}\n32:{}\n33:-----END PRIVATE KEY-----",
        line('e'),
        line('f')
    );
    // Review FIX8-B 2: grep's context lines (`-A`, `-B`) use `N-`.
    let grep_after = format!(
        "20:-----BEGIN PRIVATE KEY-----\n21-{}\n22-{}\n",
        line('a'),
        line('b')
    );
    let grep_before = format!(
        "31-{}\n32-{}\n33:-----END PRIVATE KEY-----",
        line('e'),
        line('f')
    );
    // Review FIX9-B 1: recursive or multi-file grep puts the file name first.
    let grep_recursive = format!(
        "./k/id_rsa:13:-----BEGIN PRIVATE KEY-----\n./k/id_rsa-14-{}\n./k/id_rsa-15-{}\n",
        line('a'),
        line('b')
    );
    let grep_files = format!(
        "key.pem-{}\nkey.pem-{}\nkey.pem:-----END PRIVATE KEY-----",
        line('e'),
        line('f')
    );
    for text in [
        begin_half,
        end_half,
        grep_n,
        grep_after,
        grep_before,
        grep_recursive,
        grep_files,
    ] {
        for value in [Value::String(text.clone()), json!({"content": text})] {
            let mut v = value;
            r.redact_json(&mut v);
            let out = v.to_string();
            for c in ['a', 'b', 'c', 'e', 'f'] {
                assert!(!out.contains(&line(c)), "{out}");
            }
            assert!(!out.contains("AQIDBA=="), "{out}");
        }
    }
}

/// Review FIX9-B 1: the file-name prefix must not reach prose, and a redacted
/// record must come out of a second pass unchanged.
#[test]
fn file_name_gutters_leave_prose_and_redacted_text_alone() {
    let r = Redactor::default();
    let body = "Q".repeat(64);
    let prose = "The key's last line is:\n-----END PRIVATE KEY-----";
    let mut v = json!({"content": prose});
    r.redact_json(&mut v);
    assert_eq!(v, json!({"content": prose}));
    for text in [
        format!("./k/id_rsa:13:-----BEGIN PRIVATE KEY-----\n./k/id_rsa-14-{body}\n"),
        format!("key.pem-{body}\nkey.pem:-----END PRIVATE KEY-----"),
        format!(
            "a.pem-{body}\na.pem-{body}\na.pem:-----END RSA PRIVATE KEY-----\nb.pem:-----BEGIN RSA PRIVATE KEY-----\nb.pem-{body}\n"
        ),
    ] {
        let mut once = json!({"content": text, "args": {"cmd": "grep -rn -A3 BEGIN ."}});
        r.redact_json(&mut once);
        assert!(!once.to_string().contains(&body), "{once}");
        let mut twice = once.clone();
        r.redact_json(&mut twice);
        assert_eq!(once, twice);
    }
}

/// Review FIX7-B 3: a key held on one line with `\n` escapes (a GCP
/// service-account file or Terraform state, read through `cat`) and cut by
/// characters kept its body: in a field's JSON probe the escape is `\\n`,
/// which the separator did not take. The key here is made up.
#[test]
fn a_cut_escaped_one_line_key_loses_its_body() {
    let r = Redactor::default();
    let body: Vec<String> = (0..6).map(|i| format!("{}{}", "K".repeat(60), i)).collect();
    let file = format!(
        "{{\n  \"type\": \"service_account\",\n  \"private_key\": \"-----BEGIN PRIVATE KEY-----\\n{}\\n-----END PRIVATE KEY-----\\n\",\n}}",
        body.join("\\n")
    );
    // Cut in the middle of the key: one value holds the head, the next the tail.
    let cut = file.find(&body[3]).unwrap();
    for part in [&file[..cut], &file[cut..]] {
        let mut v = json!({"content": part});
        r.redact_json(&mut v);
        let out = v.to_string();
        for line in &body {
            assert!(!out.contains(line.as_str()), "{out}");
        }
        serde_json::from_str::<Value>(&out).expect("JSON");
    }
}
