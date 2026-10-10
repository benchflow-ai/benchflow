//! Judges: which model role grades what, each role's settings with the
//! defaults filled in, and the judges' briefs (task.md's docs/judges.md).
//!
//! `[verifier.judges]`'s own keys are checked with the config
//! (`config::judges`); this module holds what needs the rubric, the
//! behaviors file, or the package's files.

use std::path::Path;

use serde_json::{Map, Value};

use crate::diag::Diagnostic;

use super::code;
use super::pyfmt::str_of;
use super::pyval::{table, truthy};
use super::report::{At, ConfigLoc, FileLoc, Sink};
use super::scan::is_config_info;
use super::text::universal;
use super::vocab::{AGENT_JUDGE_KEYS, JUDGE_KEYS, MODEL_JUDGED, MONITOR_JUDGES};

/// A model role's settings, with the keys that took their default.
#[derive(Debug, Clone, Default)]
pub(crate) struct Settings {
    pub values: Map<String, Value>,
    pub filled: Vec<String>,
}

impl Settings {
    pub(crate) fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    pub(crate) fn is_filled(&self, key: &str) -> bool {
        self.filled.iter().any(|k| k == key)
    }
}

/// The default of a judge key for a role (task.md's docs/document.md,
/// "Defaults"); `brief` is found on disk instead.
fn default(key: &str, role: &str) -> Option<Value> {
    Some(match key {
        "samples" => Value::from(1),
        "aggregate" => Value::from("median"),
        "per" => Value::from(if role == "agent" {
            "rubric"
        } else {
            "criterion"
        }),
        "timeout" => Value::from(if role == "agent" { "20m" } else { "2m" }),
        "harness" => Value::from("judge-loop@1"),
        "services" => Value::Bool(false),
        _ => return None,
    })
}

/// A model role's settings: a stage's settings win over the task's, and at
/// each level the role's own table wins over `[verifier.judges]`'s shared
/// keys (the reference's `resolve_judge`).
pub(crate) fn resolve(
    config: &Value,
    role: &str,
    stage: Option<&str>,
    dir: Option<&Path>,
) -> Settings {
    let mut scopes: Vec<&Map<String, Value>> = Vec::new();
    if let Some(s) = stage {
        scopes.push(table(config, &["stages", s, "verifier", "judges"]));
    }
    scopes.push(table(config, &["verifier", "judges"]));
    let mut keys: Vec<&str> = JUDGE_KEYS.to_vec();
    if role == "agent" {
        keys.extend(AGENT_JUDGE_KEYS);
    }
    keys.sort_unstable();
    let mut out = Settings::default();
    for k in keys {
        let found = scopes.iter().find_map(|scope| {
            let rt = scope.get(role);
            if let Some(Value::Object(t)) = rt
                && let Some(v) = t.get(k)
            {
                return Some(v.clone());
            }
            if k == "model"
                && let Some(v @ (Value::String(_) | Value::Array(_))) = rt
            {
                return Some(v.clone());
            }
            if matches!(k, "samples" | "aggregate" | "min_samples")
                && let Some(v) = scope.get(k)
            {
                return Some(v.clone());
            }
            None
        });
        match found {
            Some(v) => {
                out.values.insert(k.to_string(), v);
            }
            None => {
                let value = if k == "brief" {
                    let folder = stage.map_or_else(
                        || "verifier".to_string(),
                        |s| format!("stages/{s}/verifier"),
                    );
                    [folder.as_str(), "verifier"]
                        .iter()
                        .find(|f| dir.is_some_and(|d| d.join(f).join("judge.md").is_file()))
                        .map(|f| Value::from(format!("{f}/judge.md")))
                } else {
                    default(k, role)
                };
                if let Some(v) = value {
                    out.values.insert(k.to_string(), v);
                    out.filled.push(k.to_string());
                }
            }
        }
    }
    if !out.values.contains_key("min_samples") {
        let samples = out.values.get("samples").cloned().unwrap_or(Value::from(1));
        out.values.insert("min_samples".into(), samples);
        out.filled.push("min_samples".into());
    }
    out
}

/// How many criteria and behaviors a role grades.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    pub criteria: usize,
    pub behaviors: usize,
}

/// The model roles a rubric and a behaviors file send work to, in the order
/// they first appear (the reference's `_roles_in`).
pub(crate) fn roles_in(rubric: Option<&Value>, behaviors: Option<&Value>) -> Vec<(String, Counts)> {
    let mut out: Vec<(String, Counts)> = Vec::new();
    let mut add = |role: &str, criterion: bool| {
        let i = match out.iter().position(|(r, _)| r == role) {
            Some(i) => i,
            None => {
                out.push((role.to_string(), Counts::default()));
                out.len() - 1
            }
        };
        if criterion {
            out[i].1.criteria += 1;
        } else {
            out[i].1.behaviors += 1;
        }
    };
    if let Some(Value::Array(crits)) = rubric.and_then(|r| r.get("criteria")) {
        for c in crits {
            if let Value::Object(m) = c
                && !m.contains_key("remove")
                && let Some(j) = m.get("judge").and_then(Value::as_str)
                && MODEL_JUDGED.contains(&j)
            {
                add(j, true);
            }
        }
    }
    if let Some(Value::Array(watch)) = behaviors.and_then(|b| b.get("watch")) {
        for b in watch {
            if let Value::Object(m) = b
                && m.get("detect").and_then(Value::as_str) == Some("monitor")
            {
                let role = m
                    .get("judge")
                    .and_then(Value::as_str)
                    .filter(|j| MONITOR_JUDGES.contains(j))
                    .unwrap_or("llm");
                add(role, false);
            }
        }
    }
    out
}

/// A brief is plain markdown: no frontmatter and no config block (the
/// reference's `_check_brief_file`).
pub(crate) fn brief_file(dir: &Path, rel: &str) -> Vec<Diagnostic> {
    let Ok(bytes) = std::fs::read(dir.join(rel)) else {
        return vec![
            Diagnostic::error(code::JUDGE_BRIEF, "a judge brief is UTF-8 markdown").in_file(rel),
        ];
    };
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&bytes);
    let Ok(raw) = std::str::from_utf8(bytes) else {
        return vec![
            Diagnostic::error(code::JUDGE_BRIEF, "a judge brief is UTF-8 markdown").in_file(rel),
        ];
    };
    let text = universal(raw);
    let loc = FileLoc {
        file: rel.into(),
        text: raw.to_string(),
    };
    let mut sink = Sink::new(Some(&loc));
    if text.starts_with("---\n") {
        sink.error(
            code::JUDGE_BRIEF,
            At::Line(1),
            "a judge brief is plain markdown with no frontmatter: judge settings go in [verifier.judges.<role>], and a brief has no \
name or version of its own",
        );
    }
    for (n, line) in text.split('\n').enumerate() {
        if let Some((_, _, info)) = super::scan::fence(line)
            && is_config_info(info)
        {
            sink.error(
                code::JUDGE_BRIEF,
                At::Line(n + 1),
                "a judge brief has no config block: move these settings to [verifier.judges.<role>]",
            );
        }
    }
    sink.into_diagnostics()
}

/// What `check_judges` does beyond the config: the default brief,
/// `verifier/judge.md`, and the agent judge's token budget.
pub(crate) fn check(
    config: &Value,
    rubric: Option<&Value>,
    behaviors: Option<&Value>,
    dir: &Path,
    loc: &ConfigLoc<'_>,
    deep: bool,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    if deep && dir.join("verifier/judge.md").is_file() {
        out.extend(brief_file(dir, "verifier/judge.md"));
    }
    if roles_in(rubric, behaviors)
        .iter()
        .any(|(r, _)| r == "agent")
    {
        let settings = resolve(config, "agent", None, Some(dir));
        let tokens = settings.get("budget").and_then(|b| b.get("tokens"));
        if !truthy(tokens) {
            let timeout = settings
                .get("timeout")
                .map_or_else(|| "None".to_string(), str_of);
            let mut sink = Sink::new(Some(loc));
            sink.warning(
                code::JUDGE_BUDGET,
                At::Key("verifier.judges.agent"),
                format!(
                    "the agent judge has no token budget ([verifier.judges.agent] budget = {{ tokens = ... }}): here it is bounded by its \
timeout, {timeout}, but purpose = \"rl\" and delivery grade require one"
                ),
            );
            out.extend(sink.into_diagnostics());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_are_marked() {
        let s = resolve(
            &json!({"verifier": {"judges": {"agent": "m"}}}),
            "agent",
            None,
            None,
        );
        assert_eq!(s.get("model"), Some(&json!("m")));
        assert_eq!(s.get("harness"), Some(&json!("judge-loop@1")));
        assert_eq!(s.get("timeout"), Some(&json!("20m")));
        assert_eq!(s.get("per"), Some(&json!("rubric")));
        assert_eq!(s.get("samples"), Some(&json!(1)));
        assert_eq!(s.get("min_samples"), Some(&json!(1)));
        assert!(s.is_filled("harness") && s.is_filled("min_samples") && !s.is_filled("model"));
        let llm = resolve(
            &json!({"verifier": {"judges": {"samples": 3}}}),
            "llm",
            None,
            None,
        );
        assert_eq!(llm.get("timeout"), Some(&json!("2m")));
        assert_eq!(llm.get("per"), Some(&json!("criterion")));
        assert_eq!(llm.get("min_samples"), Some(&json!(3)));
        assert!(!llm.is_filled("samples"));
    }

    #[test]
    fn stage_settings_win() {
        let cfg = json!({"verifier": {"judges": {"llm": {"timeout": "5m"}}},
                         "stages": {"s": {"verifier": {"judges": {"llm": {"timeout": "9m"}}}}}});
        assert_eq!(
            resolve(&cfg, "llm", Some("s"), None).get("timeout"),
            Some(&json!("9m"))
        );
        assert_eq!(
            resolve(&cfg, "llm", None, None).get("timeout"),
            Some(&json!("5m"))
        );
    }

    #[test]
    fn roles_count_criteria_and_monitors() {
        let rubric = json!({"criteria": [{"judge": "agent"}, {"judge": "test"}, {"judge": "llm"}, {"judge": "agent", "remove": true}]});
        let beh = json!({"watch": [{"detect": "monitor"}, {"detect": "monitor", "judge": "agent"}, {"detect": "rule"}]});
        let roles = roles_in(Some(&rubric), Some(&beh));
        assert_eq!(
            roles[0],
            (
                "agent".to_string(),
                Counts {
                    criteria: 1,
                    behaviors: 1
                }
            )
        );
        assert_eq!(
            roles[1],
            (
                "llm".to_string(),
                Counts {
                    criteria: 1,
                    behaviors: 1
                }
            )
        );
    }
}
