//! Package digests against goldens from other implementations.
//!
//! `tests/fixtures/digest/cases.json` holds small package trees, and
//! `expected.json` their digests as `make_goldens.py` computed them:
//! Harbor's packager digest written out in Python with pathspec (and
//! checked against task.md's `tools/evidence.py` where a tree has no
//! `.gitignore`), task.md's tree hash from `tools/evidence.py`, and
//! BenchFlow 0.8's `task_digest`. Each tree is built in a temporary folder
//! here exactly as there.

use std::fs;
use std::path::{Path, PathBuf};

use benchflow_task::digest;
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/digest")
}

fn json(name: &str) -> serde_json::Map<String, Value> {
    let text = fs::read_to_string(fixtures().join(name)).unwrap();
    match serde_json::from_str(&text).unwrap() {
        Value::Object(m) => m,
        _ => panic!("{name} is an object"),
    }
}

/// Write a case's files under `root`: text as UTF-8, or `{"base64": ...}`.
fn build(root: &Path, files: &serde_json::Map<String, Value>) {
    for (rel, body) in files {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let data = match body {
            Value::String(text) => text.as_bytes().to_vec(),
            Value::Object(o) => base64(o["base64"].as_str().unwrap()),
            _ => panic!("{rel}: text or {{\"base64\": ...}}"),
        };
        fs::write(path, data).unwrap();
    }
}

fn base64(text: &str) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0;
    for c in text.bytes().filter(|&c| c != b'=') {
        let v = ALPHABET.iter().position(|&a| a == c).expect("base64") as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    out
}

#[test]
fn digests_match_the_goldens() {
    let cases = json("cases.json");
    let expected = json("expected.json");
    assert_eq!(cases.len(), expected.len());
    for (name, case) in &cases {
        let work = tempfile::tempdir().unwrap();
        let root = work.path().join(name);
        build(&root, case["files"].as_object().unwrap());
        let want = &expected[name];
        assert_eq!(
            digest::harbor_digest(&root).unwrap(),
            want["harbor"],
            "{name}: harbor"
        );
        assert_eq!(
            digest::tree_hash(&root).unwrap(),
            want["tree"],
            "{name}: tree"
        );
        assert_eq!(
            digest::benchflow_digest(&root).unwrap(),
            want["benchflow"],
            "{name}: benchflow"
        );
        // A package's digest is the one its own format's tools record.
        let by_format = match name.as_str() {
            "taskmd" => "tree",
            "v06" => "benchflow",
            _ => "harbor",
        };
        assert_eq!(
            digest::digest(&root).unwrap(),
            want[by_format],
            "{name}: digest()"
        );
    }
}

/// BenchFlow 0.8's own golden digests for its fixtures
/// (`tests/test_task_digest.py` at the repository root).
#[test]
fn benchflow_digest_matches_0_8s_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/digest");
    assert_eq!(
        digest::benchflow_digest(&root.join("alpha-task")).unwrap(),
        "sha256:6a0c023a27c1dd1b68b16ed23a41cdd9b3ea3401599bc6abbb95ed77d41e08f9"
    );
    assert_eq!(
        digest::benchflow_digest(&root.join("beta-task")).unwrap(),
        "sha256:8f5a4f868cef8c5c50b700183d0b9bd7405d92245eed79a0d50494b7daa0de16"
    );
}

#[test]
fn a_folder_that_is_not_a_package_has_no_digest() {
    let work = tempfile::tempdir().unwrap();
    let e = digest::digest(work.path()).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    assert!(digest::tree_hash(&work.path().join("missing")).is_err());
}
