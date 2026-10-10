//! CTRF test reports (Common Test Report Format): what a task.md verifier
//! script writes to `/logs/verifier/ctrf.json` (task.md's docs/package.md,
//! "Verifier scripts").
//!
//! Only `results.tests` (each with a `name` and a `status`) and
//! `results.tool` are read; every other field is ignored. A missing report,
//! or one that is not a CTRF report, leaves no tests, so every test-judged
//! criterion matches nothing and fails.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One test in a report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CtrfTest {
    /// pytest's node id, such as `tests/test_retry.py::test_backoff[0.5]`.
    pub name: String,
    /// `passed`, `failed`, `skipped`, `pending`, or `other`.
    pub status: String,
    /// The test's file, when the report gives one.
    #[serde(rename = "filePath", skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    /// Milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// What the runner said about a failure. It often holds expected
    /// values, so it stays verifier-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// A parsed `ctrf.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Ctrf {
    /// The tests, in the report's order; empty when the report is missing
    /// or malformed.
    pub tests: Vec<CtrfTest>,
    /// The runner, as `name version` (`pytest 8.4.1`), when the report
    /// names it.
    pub tool: Option<String>,
    /// Why the report holds no tests, when it is missing or malformed.
    pub problem: Option<String>,
}

impl Ctrf {
    /// The report of a verifier that wrote none.
    pub fn missing() -> Ctrf {
        Ctrf {
            problem: Some("the verifier wrote no ctrf.json".into()),
            ..Ctrf::default()
        }
    }

    /// Read `ctrf.json`; a missing file is [`Ctrf::missing`].
    pub fn from_path(path: &Path) -> Ctrf {
        match std::fs::read(path) {
            Ok(bytes) => Ctrf::from_slice(&bytes),
            Err(_) => Ctrf::missing(),
        }
    }

    /// Parse a report's bytes.
    pub fn from_slice(bytes: &[u8]) -> Ctrf {
        let failed = |problem: String| Ctrf {
            problem: Some(problem),
            ..Ctrf::default()
        };
        let Ok(text) = std::str::from_utf8(bytes) else {
            return failed("ctrf.json is not UTF-8 text".into());
        };
        let report: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => return failed(format!("ctrf.json is not JSON: {e}")),
        };
        let results = report.get("results").filter(|r| r.is_object());
        let Some(Value::Array(raw)) = results.and_then(|r| r.get("tests")) else {
            return failed("ctrf.json has no results.tests list".into());
        };
        let mut tests = Vec::with_capacity(raw.len());
        for (n, t) in raw.iter().enumerate() {
            let (Some(name), Some(status)) = (
                t.get("name").and_then(Value::as_str),
                t.get("status").and_then(Value::as_str),
            ) else {
                return failed(format!(
                    "ctrf.json results.tests[{n}] needs a name and a status"
                ));
            };
            tests.push(CtrfTest {
                name: name.to_string(),
                status: status.to_string(),
                file_path: t
                    .get("filePath")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                duration: t
                    .get("duration")
                    .and_then(Value::as_f64)
                    .filter(|d| d.is_finite()),
                message: t.get("message").and_then(Value::as_str).map(str::to_string),
            });
        }
        let tool = results.and_then(|r| r.get("tool")).map(|tool| {
            [tool.get("name"), tool.get("version")]
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        });
        Ctrf {
            tests,
            tool: tool.filter(|t| !t.is_empty()),
            problem: None,
        }
    }

    /// The tests `check` names (see [`matching_tests`]).
    pub fn matching(&self, check: &str) -> Vec<&CtrfTest> {
        matching_tests(check, &self.tests)
    }

    /// Whether a test-judged criterion with `check` passes (see
    /// [`test_verdict`]).
    pub fn verdict(&self, check: &str) -> bool {
        test_verdict(check, &self.tests)
    }
}

/// `path` without a leading `./` or `/` (repeated).
fn bare(path: &str) -> &str {
    let mut p = path;
    loop {
        if let Some(rest) = p.strip_prefix("./") {
            p = rest;
        } else if let Some(rest) = p.strip_prefix('/') {
            p = rest;
        } else {
            return p;
        }
    }
}

/// The tests a test-judged criterion's `check` names (task.md's
/// docs/package.md, "How a check matches tests"). `*` names every test.
/// Otherwise a test matches by name: its name, or its name without a
/// bracketed parameter suffix such as `[0.5]`, equals the check or ends with
/// `::` or `/` and the check. Only when no test matches by name, every test
/// in the file the check names matches: its `filePath`, or else the part of
/// its name before the first `::`, equals the check or ends with `/` and the
/// check, ignoring a leading `./` or `/` on either.
pub fn matching_tests<'a>(check: &str, tests: &'a [CtrfTest]) -> Vec<&'a CtrfTest> {
    if check.trim() == "*" {
        return tests.iter().collect();
    }
    let named = |name: &str| {
        let base = name.split('[').next().unwrap_or(name);
        [name, base].iter().any(|n| {
            *n == check || n.ends_with(&format!("::{check}")) || n.ends_with(&format!("/{check}"))
        })
    };
    let by_name: Vec<&CtrfTest> = tests.iter().filter(|t| named(&t.name)).collect();
    if !by_name.is_empty() {
        return by_name;
    }
    let wanted = bare(check.trim());
    tests
        .iter()
        .filter(|t| {
            let file = match t.file_path.as_deref().filter(|f| !f.is_empty()) {
                Some(f) => Some(f),
                None => t.name.split_once("::").map(|(f, _)| f),
            };
            file.is_some_and(|f| {
                let f = bare(f);
                f == wanted || f.ends_with(&format!("/{wanted}"))
            })
        })
        .collect()
}

/// A test-judged criterion passes when at least one test matches its check
/// and every test that matches passed.
pub fn test_verdict(check: &str, tests: &[CtrfTest]) -> bool {
    let matched = matching_tests(check, tests);
    !matched.is_empty() && matched.iter().all(|t| t.status == "passed")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> Ctrf {
        Ctrf::from_slice(
            br#"{"results": {"tool": {"name": "pytest", "version": "8.4.1"}, "tests": [
                {"name": "tests/test_retry.py::test_timeout_retries", "status": "passed", "duration": 41},
                {"name": "tests/test_retry.py::test_backoff[0.5]", "status": "passed"},
                {"name": "tests/test_retry.py::test_backoff[1.0]", "status": "failed", "message": "assert 1.31 <= 1.2"}
            ]}, "summary": {"x": 1}}"#,
        )
    }

    /// docs/package.md, "How a check matches tests": the worked table.
    #[test]
    fn the_spec_table() {
        let r = report();
        assert_eq!(r.tool.as_deref(), Some("pytest 8.4.1"));
        for (check, matched, verdict) in [
            ("test_timeout_retries", 1, true),
            ("test_retry.py::test_timeout_retries", 1, true),
            ("test_backoff", 2, false),
            ("test_backoff[0.5]", 1, true),
            ("tests/test_retry.py", 3, false),
            ("test_retry.py", 3, false),
            ("retry.py::test_timeout_retries", 0, false),
            ("*", 3, false),
            ("./tests/test_retry.py", 3, false),
        ] {
            assert_eq!(r.matching(check).len(), matched, "{check}");
            assert_eq!(r.verdict(check), verdict, "{check}");
        }
    }

    #[test]
    fn star_needs_a_nonempty_report_that_all_passed() {
        assert!(!test_verdict("*", &[]));
        assert!(test_verdict("*", &report().tests[..2]));
    }

    #[test]
    fn file_paths_match_only_when_no_name_does() {
        let tests = vec![
            CtrfTest {
                name: "a".into(),
                status: "passed".into(),
                file_path: Some("/src/tests/test_x.py".into()),
                duration: None,
                message: None,
            },
            CtrfTest {
                name: "b".into(),
                status: "passed".into(),
                file_path: Some("./tests/test_x.py".into()),
                duration: None,
                message: None,
            },
        ];
        assert_eq!(matching_tests("tests/test_x.py", &tests).len(), 2);
        assert_eq!(matching_tests("a", &tests).len(), 1);
    }

    #[test]
    fn a_report_that_is_not_ctrf_has_no_tests() {
        assert_eq!(
            Ctrf::missing().problem.as_deref(),
            Some("the verifier wrote no ctrf.json")
        );
        assert!(Ctrf::from_slice(b"{").problem.is_some());
        assert!(Ctrf::from_slice(br#"{"results": {}}"#).problem.is_some());
        let partial = Ctrf::from_slice(
            br#"{"results": {"tests": [{"name": "x", "status": "passed"}, {"name": "y"}]}}"#,
        );
        assert!(partial.tests.is_empty());
        assert_eq!(
            partial.problem.as_deref(),
            Some("ctrf.json results.tests[1] needs a name and a status")
        );
        assert!(
            Ctrf::from_path(Path::new("/nonexistent/ctrf.json"))
                .tests
                .is_empty()
        );
    }
}
