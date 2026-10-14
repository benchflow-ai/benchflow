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

/// Review G1-R1-A-3: a key's BEGIN half with armour header lines (OpenPGP's
/// `Version:` and `Comment:`, PEM's `Proc-Type:` and `DEK-Info:`) and the
/// blank line after them kept its whole body in every value redacted on its
/// own (the trajectory, `atif.json`, logs): the half rule allowed only a
/// gutter between the marker and the body. The tail half is checked too.
/// The bodies here are made up.
#[test]
fn half_keys_behind_armour_headers_lose_their_bodies() {
    let r = Redactor::default();
    let line = |c: char| std::iter::repeat_n(c, 64).collect::<String>();
    let body: Vec<String> = ['a', 'b', 'c', 'd'].into_iter().map(line).collect();
    let pgp = "PGP PRIVATE KEY BLOCK";
    for (label, headers) in [
        (pgp, "Version: GnuPG v1\n"),
        (pgp, "Comment: https://keybase.io/alice\n"),
        (
            pgp,
            "Version: Keybase OpenPGP v2.1.15\nComment: https://keybase.io/crypto\n",
        ),
        (
            "RSA PRIVATE KEY",
            "Proc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,0A1B2C3D4E5F60718293A4B5C6D7E8F9\n",
        ),
    ] {
        let head = format!(
            "-----BEGIN {label}-----\n{headers}\n{}\n{}\n",
            body[0], body[1]
        );
        let tail = format!("{}\n{}\n=AbCd\n-----END {label}-----\n", body[2], body[3]);
        for text in [head.clone(), tail, head.replace('\n', "\r\n")] {
            let plain = r.redact(&text).into_owned();
            let mut field = json!({"content": text});
            r.redact_json(&mut field);
            let mut bare = Value::String(text.clone());
            r.redact_json(&mut bare);
            for out in [plain, field.to_string(), bare.to_string()] {
                for l in &body {
                    assert!(!out.contains(l.as_str()), "{label} {headers:?}: {out}");
                }
                assert!(out.contains(label), "{out}");
            }
        }
    }
    // Prose with a `name: value` line after a marker and no body stays.
    let prose = "-----BEGIN PGP PRIVATE KEY BLOCK-----\nNote: none here\n\nDone.";
    assert_eq!(r.redact(prose), prose);
}

/// Review G1-R1-A-1 (round 1 at 6d38ceed): an OpenPGP END half ends in a
/// short remainder, the CRC-24 checksum line (`=XXXX`) and the marker; a
/// GnuPG default key's remainder is 12 characters. The rule took the
/// checksum alone and kept every body line. Plain, numbered, as a JSON
/// field and with the remainder and checksum alone (`tail -3`), each loses
/// its body and checksum, and a second pass changes nothing. The bodies
/// here are made up.
#[test]
fn a_pgp_end_half_loses_its_body_past_the_checksum() {
    let r = Redactor::default();
    let line = |c: char, n: usize| std::iter::repeat_n(c, n).collect::<String>();
    let end = "-----END PGP PRIVATE KEY BLOCK-----";
    for short in [1, 12, 15, 16, 28] {
        let lines = [
            line('a', 64),
            line('b', 64),
            line('c', short),
            "=Ab1C".into(),
        ];
        for (name, from) in [("tail", 0), ("tail -3", 2)] {
            let shown = &lines[from..];
            let plain: String = shown.iter().map(|l| format!("{l}\n")).collect();
            let numbered: String = shown
                .iter()
                .enumerate()
                .map(|(i, l)| format!("{:>6}\t{l}\n", 20 + i))
                .collect();
            for text in [
                format!("{plain}{end}\n"),
                format!("{numbered}{:>6}\t{end}\n", 20 + shown.len()),
            ] {
                let once = r.redact(&text).into_owned();
                let mut field = json!({"content": text});
                r.redact_json(&mut field);
                for out in [once.clone(), field.to_string()] {
                    for l in shown.iter().filter(|l| l.len() >= 12 || l.starts_with('=')) {
                        assert!(!out.contains(l.as_str()), "{short} {name}: {out}");
                    }
                    assert!(out.contains(end), "{out}");
                }
                assert_eq!(r.redact(&once), once);
            }
        }
    }
}

/// Review G1-R2-A-2: the head half of a key behind armour headers, in a
/// numbered view (`cat -n`, the Read tool, `grep -n ''`, `grep -rn -A`),
/// kept every body line in each value redacted on its own: a gutter before
/// a header line, or a gutter-only blank line, broke the match. With no
/// header, a PGP block's blank line alone did too. The bodies here are
/// made up.
#[test]
fn half_keys_behind_armour_headers_in_numbered_views_lose_their_bodies() {
    let r = Redactor::default();
    let line = |c: char| std::iter::repeat_n(c, 64).collect::<String>();
    let body: Vec<String> = ['a', 'b', 'c'].into_iter().map(line).collect();
    let pgp = "PGP PRIVATE KEY BLOCK";
    type Gutter = fn(usize) -> String;
    let gutters: [(&str, Gutter); 5] = [
        ("cat -n", |i| format!("{i:>6}\t")),
        ("Read", |i| format!("{i:>6}\u{2192}")),
        ("grep -n ''", |i| format!("{i}:")),
        ("grep -rn -A", |i| {
            if i == 1 {
                "./k.asc:1:".to_string()
            } else {
                format!("./k.asc-{i}-")
            }
        }),
        ("nl", |i| format!("{i:>6}  ")),
    ];
    for (label, headers) in [
        (pgp, vec![]),
        (pgp, vec!["Version: GnuPG v1"]),
        (pgp, vec!["Comment: https://keybase.io/alice"]),
        (
            "RSA PRIVATE KEY",
            vec![
                "Proc-Type: 4,ENCRYPTED",
                "DEK-Info: AES-128-CBC,0A1B2C3D4E5F60718293A4B5C6D7E8F9",
            ],
        ),
    ] {
        let mut lines = vec![format!("-----BEGIN {label}-----")];
        lines.extend(headers.iter().map(|h| h.to_string()));
        if label == pgp || !headers.is_empty() {
            lines.push(String::new());
        }
        lines.extend(body.iter().cloned());
        for (name, gutter) in gutters {
            let text: String = lines
                .iter()
                .enumerate()
                .map(|(i, l)| format!("{}{l}\n", gutter(i + 1)))
                .collect();
            let plain = r.redact(&text).into_owned();
            let mut field = json!({"content": text});
            r.redact_json(&mut field);
            let mut bare = Value::String(text.clone());
            r.redact_json(&mut bare);
            for out in [plain, field.to_string(), bare.to_string()] {
                for l in &body {
                    assert!(
                        !out.contains(l.as_str()),
                        "{label} {headers:?} {name}: {out}"
                    );
                }
                assert!(out.contains(label), "{out}");
            }
        }
    }
}

/// Review G1-R2-A-1: each event's redaction replaced a `grep -n` listing
/// of both marker lines whole, and the END half rule took a gutter's first
/// digit (`27:-----END …`, `2` as the remainder and `7:` as a gutter) and a
/// prose word (`ends at -----END …`) as a key's last line, so the trainer
/// files' walk, which reads the redacted trajectory, saw no BEGIN or took
/// the END for a key's end. All stay as they are now; a whole key and a
/// short last line at a line's start, behind a gutter or not, still go.
#[test]
fn markers_with_no_body_stay() {
    let r = Redactor::default();
    let (b, e) = (
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
    );
    for text in [
        format!("1:{b}\n27:{e}\n"),
        format!("./k/id_rsa:1:{b}\n./k/id_rsa:27:{e}"),
        format!("{b}\n{e}"),
        format!("27:{e}"),
        format!("[tool_call: grep -n -- '{e}' id]\n27:{e}\n"),
        format!("./k/id_rsa:27:{e}"),
        format!("The key ends at {e}, so I read the middle next."),
        format!("ends with AQIDBA== {e}"),
    ] {
        assert_eq!(r.redact(&text), text);
        let mut v = json!({"content": text});
        r.redact_json(&mut v);
        assert_eq!(v, json!({"content": text}));
    }
    for text in [
        format!("AQIDBA==\n{e}"),
        format!("$ tail -2 id\nAQIDBA==\n{e}"),
        format!("    26\tAQIDBA==\n    27\t{e}"),
        format!("26-AQIDBA==\n27:{e}"),
        format!("['AQIDBA==\\n{e}']"),
    ] {
        let mut v = json!({"content": text});
        r.redact_json(&mut v);
        let out = v.to_string();
        assert!(!out.contains("AQIDBA=="), "{text:?}: {out}");
        assert!(r.redact(&text).contains(e), "{text:?}");
        assert!(!r.redact(&text).contains("AQIDBA=="), "{text:?}");
    }
    assert_eq!(
        r.redact(&format!("1:{b}\n2:MIIEow\n3:{e}")),
        format!("1:{REDACTED}")
    );
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

/// The smallest `size_limit` (to 1 KiB) under which `regex` compiles.
fn compiled_size(regex: &Regex) -> usize {
    let fits = |limit: usize| {
        regex::RegexBuilder::new(regex.as_str())
            .size_limit(limit)
            .build()
            .is_ok()
    };
    let (mut low, mut high) = (0, 64 << 20);
    assert!(fits(high));
    while high - low > 1024 {
        let mid = (low + high) / 2;
        if fits(mid) { high = mid } else { low = mid }
    }
    high
}

/// Decision G-2: the file-name prefix takes `+ @ , % =` and non-ASCII
/// characters (and their `\uXXXX` escapes). Unicode `\w` did not fit the
/// regex size limit (10 MiB by default); this class does, at about 2.2 MiB.
#[test]
fn the_half_rules_fit_well_under_the_regex_size_limit() {
    for (name, regex) in [
        ("PEM_HALF_BEGIN", &*PEM_HALF_BEGIN),
        ("PEM_HALF_END", &*PEM_HALF_END),
        ("KEY_LINE", &*KEY_LINE),
    ] {
        let size = compiled_size(regex);
        eprintln!("{name}: {size} bytes");
        assert!(size < 4 << 20, "{name}: {size}");
    }
}

/// `value` after the walk, and the walk's count of the serialized line,
/// which must be `to_python_json`'s length (the span is measured on it).
/// An array stands for a list of message texts: it is walked as a record's
/// `completion` with one message per item, and given back as the texts.
fn walked(value: &Value) -> Value {
    if let Value::Array(texts) = value {
        let messages: Vec<Value> = texts.iter().map(|t| json!({"content": t})).collect();
        let out = walked(&json!({"completion": messages}));
        let texts = out["completion"].as_array().unwrap();
        return Value::Array(texts.iter().map(|m| m["content"].clone()).collect());
    }
    let mut v = value.clone();
    let mut walk = KeyWalk::default();
    walk.value(&mut v);
    assert_eq!(walk.at, to_python_json(&v).len(), "{v}");
    v
}

fn body(i: usize) -> String {
    format!("{}{i:02}", "Pq7+".repeat(15))
}

/// Decision G-1: inside a key's span the walk removes body lines inside
/// strings, one marker per run, and keeps the rest of each string: the
/// `[tool_call: …]` title, a `repr` wrapper, a fence, gutters' neighbours,
/// prose. It counts the serialized line exactly, escapes and non-ASCII
/// included, and a second walk changes nothing.
#[test]
fn the_key_walk_removes_body_lines_and_nothing_else() {
    let record = json!({"completion": [
        {"role": "assistant", "content": "[tool_call: head -n 2 id]\n-----BEGIN RSA PRIVATE KEY----- ***REDACTED***\n"},
        {"role": "assistant", "content": "Next, \"quoted\" and é → 😀 \u{1}"},
        {"role": "assistant", "content": format!(
            "[tool_call: sed -n 3,5p id]\n{{'type': 'text', 'text': '```console\\n{}\\n{}\\n```'}}", body(1), body(2))},
        {"role": "assistant", "content": format!("Read\n```\n     6\t{}\n     7\t{}\n```\n$ ", body(3), body(4))},
        {"role": "assistant", "content": format!("./k/key+old.pem-8-{}\n./密钥/id_rsa-9-{}", body(5), body(6))},
        {"role": "assistant", "content": "***REDACTED*** -----END RSA PRIVATE KEY-----"},
        {"role": "assistant", "content": body(7)},
    ], "reward": 1.0, "metrics": {"found_password": 1.0}});
    let out = walked(&record);
    let completion = out["completion"].as_array().unwrap();
    assert_eq!(completion[1], record["completion"][1]);
    assert_eq!(
        completion[2]["content"],
        "[tool_call: sed -n 3,5p id]\n{'type': 'text', 'text': '```console\\n***REDACTED***\\n```'}"
    );
    assert_eq!(
        completion[3]["content"],
        "Read\n```\n***REDACTED***\n```\n$ "
    );
    assert_eq!(completion[4]["content"], REDACTED);
    // An END closes nothing (review G1-R2-B-1): the span runs on.
    assert_eq!(completion[6]["content"], REDACTED);
    assert_eq!(out["metrics"], record["metrics"]);
    assert_eq!(walked(&out), out);
    // Without patterns, nothing is walked.
    let mut exact = record.clone();
    Redactor::new(["x"])
        .exact_only()
        .redact_key_chunks(&mut exact);
    assert_eq!(exact, record);
}

/// Review E1-BACKSTOP-A 2: a BEGIN marker quoted in prose opens no span,
/// and an open span ends 8,192 serialized characters after the marker or
/// the last line it removed, so a later hash or base64 answer stays.
#[test]
fn a_quoted_marker_opens_no_span_and_a_span_ends() {
    let sha = "9fceb02d0ae598e95dc970b74767f19372d61af8";
    let quoted = json!({"prompt": [
        {"role": "user", "content": "Find the file starting with -----BEGIN RSA PRIVATE KEY----- and report the commit hash."},
        {"role": "assistant", "content": sha}]});
    assert_eq!(walked(&quoted), quoted);
    // A marker with nothing after it (`head -n 1`) opens one.
    let bare = |gap: usize| {
        json!({"prompt": [
            {"role": "tool", "content": "-----BEGIN OPENSSH PRIVATE KEY-----\n"},
            {"role": "assistant", "content": "w ".repeat(gap / 2)},
            {"role": "assistant", "content": sha}]})
    };
    assert_eq!(walked(&bare(7_000))["prompt"][2]["content"], REDACTED);
    assert_eq!(walked(&bare(8_400))["prompt"][2]["content"], sha);
    // So does a marker the half rule took the body after, and one with a
    // body line it could not reach (behind a PGP armour header).
    for head in [
        "-----BEGIN PRIVATE KEY----- ***REDACTED***\n```".to_string(),
        format!(
            "-----BEGIN PGP PRIVATE KEY BLOCK-----\nComment: test\n\n{}\n",
            body(0)
        ),
    ] {
        let v = json!([head, body(1)]);
        let out = walked(&v);
        assert_eq!(out[1], REDACTED, "{out}");
        assert!(!out.to_string().contains(&body(0)), "{out}");
    }
    // An END alone on its line opens a span too (review G1-R1-B-1), and an
    // END after a body line in the same string does not close it (review
    // G1-R2-B-1).
    let v = json!([
        format!("{}\n-----END PRIVATE KEY-----\n{}", body(1), body(2)),
        "-----END PRIVATE KEY-----\n-----BEGIN PRIVATE KEY-----",
        format!("{}\n-----END PRIVATE KEY-----\n{}", body(3), body(4)),
    ]);
    assert_eq!(
        walked(&v),
        json!([
            format!("{}\n-----END PRIVATE KEY-----\n{REDACTED}", body(1)),
            v[1],
            format!("{REDACTED}\n-----END PRIVATE KEY-----\n{REDACTED}")
        ])
    );
    // An END quoted in prose opens none.
    let v = json!(["The key ends at -----END PRIVATE KEY----- here.", body(1)]);
    assert_eq!(walked(&v), v);
}

/// Review G1-R1-B-1: `tail -1 id` (or `grep -n END id`) shows only the END
/// line. An END opened a span only with redacted body lines before it, so
/// the middle part read next kept every body line. An END alone on its
/// line opens one now, behind a gutter, in a `repr` block or a fence.
#[test]
fn an_end_marker_alone_on_its_line_opens_a_span() {
    let e = "-----END RSA PRIVATE KEY-----";
    for tail in [
        format!("{e}\n"),
        e.to_string(),
        format!("[tool_call: tail -1 id]\n{e}\n"),
        format!("27:{e}\n"),
        format!("[tool_call: grep -n END id]\n{{'type': 'text', 'text': '27:{e}\\n'}}"),
        // Review G1-R2-A-1 (at cd992801): the same blocks with no newline
        // after the marker end `…-----'}`.
        format!("[tool_call: grep -n END id]\n{{'type': 'text', 'text': '27:{e}'}}"),
        format!("[tool_call: tail -1 id]\n{{'type': 'text', 'text': '{e}'}}"),
        format!(
            "[tool_call: tail -1 id]\n{{'type': 'text', 'text': '{e}'}}\n{{'type': 'text', 'text': 'done'}}"
        ),
        format!("```console\n{e}\n```"),
    ] {
        let v = json!([
            tail,
            "Reading the rest.",
            format!("{}\n{}", body(1), body(2))
        ]);
        let out = walked(&v);
        assert_eq!(out[2], REDACTED, "{tail:?}");
        assert_eq!(walked(&out), out);
    }
}

/// Review G1-R1-A-2: the walk rewrote every string in the record, so a
/// task id, environment or model name made only of base64 characters
/// (`sshkeyrotationaudit`) after a key's head part became `***REDACTED***`.
/// Only the messages' text is walked now; a marker outside it opens no span.
#[test]
fn the_key_walk_leaves_record_metadata_alone() {
    let head = "[tool_call: head -2 id]\n-----BEGIN OPENSSH PRIVATE KEY----- ***REDACTED***\n";
    let verifiers = json!({
        "prompt": [{"role": "user", "content": "Audit the key rotation."}],
        "completion": [{"role": "assistant", "content": head},
                       {"role": "assistant", "content": body(1)}],
        "reward": 1.0,
        "metrics": {"sshkeyrotationaudit": 1.0},
        "info": {"task_id": "sshkeyrotationaudit", "environment": "sshkeyrotationaudit",
                 "model": "org/modelname1234567", "reward_metadata": {"error": body(2)}},
    });
    let out = walked(&verifiers);
    assert_eq!(out["completion"][1]["content"], REDACTED);
    assert_eq!(out["info"], verifiers["info"]);
    assert_eq!(out["metrics"], verifiers["metrics"]);
    let adp = json!({
        "schema_version": "1.3.1",
        "id": "sshkeyrotationaudit__1",
        "content": [
            {"class_": "api_action", "tool_call_id": "toolu01ABCDEFGHIJKLMNOP", "function": "execute",
             "kwargs": {}, "description": "head -2 id", "reasoning_content": "Look at the key."},
            {"class_": "text_observation", "tool_call_id": "toolu01ABCDEFGHIJKLMNOP",
             "content": head, "source": "environment"},
            {"class_": "api_action", "tool_call_id": "toolu01QRSTUVWXYZABCDEF", "function": "execute",
             "kwargs": {}, "description": "sed -n 3,4p id", "reward": 1.0},
            {"class_": "text_observation", "tool_call_id": "toolu01QRSTUVWXYZABCDEF",
             "content": body(1), "source": "environment"},
        ],
        "details": {"task_id": "sshkeyrotationaudit", "environment": "sshkeyrotationaudit",
                    "model": "org/modelname1234567"},
    });
    let out = walked(&adp);
    assert_eq!(out["content"][3]["content"], REDACTED);
    for (i, item) in out["content"].as_array().unwrap().iter().enumerate() {
        assert_eq!(item["tool_call_id"], adp["content"][i]["tool_call_id"]);
    }
    assert_eq!(out["details"], adp["details"]);
    assert_eq!(out["id"], adp["id"]);
    // A marker in metadata opens no span.
    let v = json!({"info": {"note": "-----BEGIN RSA PRIVATE KEY-----"},
                   "completion": [{"content": body(1)}]});
    assert_eq!(walked(&v), v);
}

/// Review G1-R1-A-1: a BEGIN marker alone on its line opens a span whatever
/// the rest of its output says (`head -1 id; wc -l id` prints the marker
/// and then `27 id`), with a gutter before it or in a `repr` block. A
/// marker quoted inside a line of prose still opens none.
#[test]
fn a_marker_alone_on_its_line_opens_a_span() {
    let m = "-----BEGIN RSA PRIVATE KEY-----";
    for head in [
        format!("{m}\n27 id\n"),
        format!("[tool_call: head -1 id; wc -l id]\n{m}\n27 id\n"),
        format!("1:{m}\n27:-----BEGIN-not-a-marker\nok"),
        format!("    1\t{m}  \n27 id"),
        format!("{{'type': 'text', 'text': '{m}\\n27 id\\n'}}"),
        format!("ok\r\n{m}\r\nexit 0"),
    ] {
        let v = json!([head, "Reading lines 2 to 13.", body(1)]);
        let out = walked(&v);
        assert_eq!(out[2], REDACTED, "{head:?}");
        assert_eq!(out[0], v[0]);
        assert_eq!(walked(&out), out);
    }
    for prose in [
        format!("It starts with '{m}' and has 27 lines."),
        format!("It starts with {m}\nand has 27 lines."),
        format!("{m} is the first line, then 26 more."),
    ] {
        let v = json!([prose, body(1)]);
        assert_eq!(walked(&v), v, "{prose:?}");
    }
}

/// Review G1-R1-A-2 (and G1-R1-B-1): a `grep -n` listing of both markers, a
/// `grep` for the END line, or prose quoting it leaves the span open, so the
/// parts read after it lose their body lines. Since review G1-R2-B-1 no END
/// closes a span.
#[test]
fn an_end_with_no_body_before_it_leaves_the_span_open() {
    let (b, e) = (
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
    );
    for steps in [
        vec![format!(
            "[tool_call: grep -n 'PRIVATE KEY' id]\n1:{b}\n27:{e}\n"
        )],
        vec![
            format!("{b} ***REDACTED***\n"),
            format!("[tool_call: grep -n -- '{e}' id]\n27:{e}\n"),
        ],
        vec![
            format!("{b} ***REDACTED***\n"),
            format!("The key ends at {e}, so I read lines 10-18 next."),
        ],
    ] {
        let mut items: Vec<Value> = steps.into_iter().map(Value::String).collect();
        items.push(json!(body(1)));
        items.push(json!(format!("{}\n{e}", body(2))));
        items.push(json!(body(3)));
        let v = Value::Array(items);
        let out = walked(&v);
        let n = v.as_array().unwrap().len();
        assert_eq!(out[n - 3], REDACTED, "{v}");
        assert_eq!(out[n - 2], format!("{REDACTED}\n{e}"), "{v}");
        // The END after a body line does not close it either.
        assert_eq!(out[n - 1], REDACTED, "{v}");
        assert_eq!(walked(&out), out);
    }
    // Through the trainer path's order, a listing the value pass replaces
    // whole (marker and all) still opens the span, and a second call
    // changes nothing.
    let r = Redactor::default();
    let mut record = json!({"completion": [
        {"role": "tool", "content": format!("1:{b}\n27:{e}\n")},
        {"role": "tool", "content": body(1)},
    ]});
    r.redact_record(&mut record);
    assert_eq!(record["completion"][1]["content"], REDACTED, "{record}");
    let mut again = record.clone();
    r.redact_record(&mut again);
    assert_eq!(again, record);
}

/// The walk's count of the serialized line matches `to_python_json` on
/// random values (every escape, non-ASCII, numbers), and walking twice
/// changes nothing.
#[test]
fn the_key_walk_counts_the_serialized_line_exactly() {
    const ATOMS: &[&str] = &[
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
        "\n",
        "\\n",
        "\r",
        "\t",
        "\\t",
        "\"",
        "'",
        "\\",
        "\u{7f}",
        "\u{1}",
        "\u{8}",
        "\u{c}",
        "é",
        "😀",
        "→",
        "    12\t",
        "./密钥/id_rsa-2-",
        "word ",
        "Pq7+Pq7+Pq7+Pq7+Pq7+",
        "AQIDBA==",
    ];
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut below = |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    for _ in 0..3_000 {
        let mut string =
            || -> String { (0..below(10)).map(|_| ATOMS[below(ATOMS.len())]).collect() };
        let messages: Vec<Value> = (0..6)
            .map(|_| json!({"role": string(), "content": string(), "description": [string()]}))
            .collect();
        let value = json!({"completion": messages, "é\"k": [1, -0.0, 1e16, 2.5e-7, null, true],
                           "info": {"task_id": string()}});
        let once = walked(&value);
        assert_eq!(walked(&once), once);
    }
}

/// A string of many BEGIN markers (an agent can print one) is walked in
/// linear time: each marker reads only up to the next one.
#[test]
fn many_markers_in_one_string_walk_quickly() {
    let one = "-----BEGIN RSA PRIVATE KEY----- x\n";
    let text = one.repeat(40_000);
    let started = std::time::Instant::now();
    let out = walked(&json!([text, body(1)]));
    assert_eq!(out[1], body(1));
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
}

/// Review G1-R3-B-1: in `grep -rn 'PRIVATE KEY' .` a source file quoting a
/// BEGIN marker comes before the key file's marker lines. The quote's BEGIN
/// was paired with the key's END and the whole listing replaced, the key's
/// BEGIN with it, so the walk opened no span. An END pairs with the nearest
/// BEGIN before it now: the listing stays, and a key shown whole after a
/// quoted marker still goes, the quote kept.
#[test]
fn an_end_pairs_with_the_nearest_begin_before_it() {
    let r = Redactor::default();
    let (b, e) = (
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
    );
    let listing = format!(
        "./src/pem.py:12:    if line.startswith(\"{b}\"):\n./keys/id:1:{b}\n./keys/id:27:{e}\n"
    );
    assert_eq!(r.redact(&listing), listing);
    let mut v = json!({"content": listing});
    r.redact_json(&mut v);
    assert_eq!(v["content"], listing);
    let whole = format!("echo \"{b}\"\n{b}\n{}\n{}\n{e}\nok", body(1), body(2));
    assert_eq!(r.redact(&whole), format!("echo \"{b}\"\n{REDACTED}\nok"));
    // A BEGIN left with no END of its own is a half: its body goes.
    let cut = format!("{b}\n{}\n{b}\n{}\n{e}", body(1), body(2));
    let out = r.redact(&cut);
    assert!(!out.contains(&body(1)) && !out.contains(&body(2)), "{out}");
    assert_eq!(r.redact(&out), out);
    // Through the trainer path: the listing redacted as an event, then the
    // record, then the middle read later.
    let mut record = json!({"completion": [
        {"role": "tool", "content": v["content"]},
        {"role": "tool", "content": format!("{}\n{}", body(1), body(2))},
    ]});
    r.redact_record(&mut record);
    assert_eq!(record["completion"][1]["content"], REDACTED, "{record}");
}

/// Review G1-R3-A-1: `cat -n id | grep KEY` prints both marker lines behind
/// `N\t` gutters. The END's gutter digits were taken for a key's short last
/// line, so the first pass wrote a redaction between the markers and a
/// second pass replaced the whole listing.
#[test]
fn a_numbered_marker_listing_survives_two_passes() {
    let r = Redactor::default();
    let (b, e) = (
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
    );
    for text in [
        format!("     1\t{b}\n    27\t{e}"),
        format!("     1 {b}\n    27 {e}\n"),
        format!("['     1\\t{b}\\n    27\\t{e}']"),
    ] {
        assert_eq!(r.redact(&text), text, "{text:?}");
        let mut v = json!({"content": text});
        r.redact_json(&mut v);
        assert_eq!(v["content"], text);
    }
    // A real short last line behind a gutter still goes.
    let text = format!("    26\tAQIDBA==\n    27\t{e}");
    assert!(!r.redact(&text).contains("AQIDBA=="));
}

/// Review G1-R3-B-2: a `tail` part's body is redacted by the half rule
/// before the walk, so its END did not carry the span on, and the middle
/// read more than 8,192 characters after the BEGIN (but soon after the
/// `tail`) kept its body lines; with the `tail` read first, no span opened
/// at all. An END with a redacted body just before it now carries the span
/// on or opens one. Since review G1-R1-B-1 (round 1 at 6d38ceed) an END
/// alone on its line opens one too; one quoted in prose opens nothing.
#[test]
fn an_end_after_a_redacted_body_carries_the_span_on() {
    let (b, e) = (
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
    );
    let gap = |n: usize| json!("w ".repeat(n / 2));
    let v = json!([
        format!("{b} {REDACTED}\n"),
        gap(6_000),
        format!("{REDACTED} {e}\n"),
        gap(3_000),
        body(1)
    ]);
    let out = walked(&v);
    assert_eq!(out[4], REDACTED, "{out}");
    assert_eq!(walked(&out), out);
    for tail in [
        format!("{REDACTED} {e}\n"),
        format!("```\n{REDACTED}\n    27\t{e}\n```"),
    ] {
        let v = json!([tail, "Now the middle.", body(1)]);
        assert_eq!(walked(&v)[2], REDACTED, "{tail:?}");
    }
    let v = json!([format!("27:{e}"), body(1)]);
    assert_eq!(walked(&v)[1], REDACTED);
    let v = json!([format!("It ends with {e}."), body(1)]);
    assert_eq!(walked(&v), v);
}

/// Review G1-R3-B-3: every removed line re-armed the span, so after a
/// marker listing each later path or SHA-1 hash re-armed it and none was
/// left in the rest of the record. Only a key-width line re-arms it now:
/// the hashes and paths after the first 8,192 characters stay.
#[test]
fn hashes_and_paths_do_not_carry_the_span_on() {
    let (b, e) = (
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
    );
    let sha = "9fceb02d0ae598e95dc970b74767f19372d61af8";
    let mut items = vec![json!(format!("1:{b}\n27:{e}\n"))];
    for _ in 0..12 {
        items.push(json!("build output ".repeat(230)));
        items.push(json!(format!("/home/agent/project3\n{sha}\n")));
    }
    let out = walked(&Value::Array(items));
    let out = out.as_array().unwrap();
    assert_eq!(out[2], format!("{REDACTED}\n"));
    assert_eq!(out[out.len() - 1], format!("/home/agent/project3\n{sha}\n"));
    let kept = out
        .iter()
        .filter(|s| s.as_str().is_some_and(|s| s.contains(sha)))
        .count();
    assert!(kept >= 9, "{kept} of 12 hashes kept");
    // A key-width line carries it on as before.
    let mut items = vec![json!(format!("1:{b}\n27:{e}\n"))];
    for i in 0..4 {
        items.push(json!("build output ".repeat(460)));
        items.push(json!(body(i)));
    }
    let out = walked(&Value::Array(items));
    assert_eq!(out[8], REDACTED, "{out}");
}

/// Review G1-R7-A-1 (at 8290c756): a marker shown alone after an agent's
/// output prefix opens a span, raw, in a `repr` and in JSON, and the middle
/// part's first body line goes with its prefix.
#[test]
fn a_marker_after_an_output_prefix_opens_a_span() {
    let e = "-----END RSA PRIVATE KEY-----";
    for prefix in [
        "\x1b[?2004l",
        "\x1b[?2004l\x1b[0m",
        "\x1b[200~",
        "\x1b]0;root@box: /app\x07",
        "\x1b]2;title\x1b\\",
        "\x1b(B\x1b[m",
        "\x1b[?1l\x1b>",
        "\\x1b[?2004l\\x1b[0m",
        "\\x1b]0;title\\x07",
        "\\u001b[200~",
        "Output: ",
        "- **output:** ",
        "**output:** ",
    ] {
        let v = json!([
            format!("Result:\n{prefix}{e}\n"),
            "Reading the rest.",
            format!("{prefix}{}\n{}", body(1), body(2))
        ]);
        let out = walked(&v);
        assert_eq!(out[2], REDACTED, "{prefix:?}");
        assert_eq!(walked(&out), out);
    }
}

/// Review G1-R7-A-2 (at 8290c756): the half rule's line start steps over an
/// output prefix too, in every redacted text, so the short remainder of a
/// key's END half after `Output: ` goes, and the marker stays.
#[test]
fn the_half_rule_steps_over_an_output_prefix() {
    let r = Redactor::default();
    let e = "-----END RSA PRIVATE KEY-----";
    for prefix in ["Output: ", "\x1b[?2004l", "- **output:** "] {
        let text = format!("x\n{prefix}AQIDBA==\n{e}\n");
        let out = r.redact(&text);
        assert!(!out.contains("AQIDBA=="), "{prefix:?}");
        assert!(out.contains(e), "{prefix:?}");
        assert_eq!(r.redact(&out), out);
    }
}
