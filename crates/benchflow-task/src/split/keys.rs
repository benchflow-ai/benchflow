//! Keys a config holds that no reader would honor where they are.
//!
//! Harbor ignores every key its model does not define, without a word, so a
//! typo such as `network_mod` or a key in the wrong table such as
//! `multi_step_reward_strategy` under `[task]` silently falls back to a
//! default. This walk finds them.

use serde_json::{Map, Value};

use super::schema::{self, Dialect, Field, Shape};
use crate::source::join;
use crate::suggest::{did_you_mean, ratio};

/// What is wrong with a key.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Problem {
    /// No table holds a key of this name.
    Unknown,
    /// Other tables hold it (dotted table paths, `""` for the top level).
    Misplaced { homes: Vec<String> },
}

/// A key that would be ignored.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KeyIssue {
    /// The key's path, as [`crate::source::Source`] spells it.
    pub path: String,
    pub key: String,
    /// The table it is in, as a dotted path without list indices (`""` for
    /// the top level).
    pub table: String,
    /// The closest key the table holds, by difflib's ratio.
    pub suggestion: Option<String>,
    /// The suggestion is close enough (ratio at least 0.8, BenchFlow 0.8's
    /// typo threshold) to offer as a fix.
    pub close: bool,
    /// The keys the table holds.
    pub holds: Vec<&'static str>,
    pub problem: Problem,
    /// The value is a table (a rename would have to touch every header).
    pub is_table: bool,
}

/// The first spelling of bench's cloud metadata opt-out, in `[environment]`
/// (a BenchFlow 0.8 task.md's `sandbox`). No release read it; it moved to
/// `[metadata] x-allow-cloud-metadata`, which 0.8 loads. A task written
/// against the old spelling is told the new place, not only that the key is
/// unknown (F1-B-r2 L2); it still opts nothing out.
const STALE_OPT_OUT: &str = "allow_cloud_metadata";

/// Every key in `config` that the dialect's readers would not honor.
pub(crate) fn unknown_keys(config: &Value, dialect: Dialect) -> Vec<KeyIssue> {
    let mut out = Vec::new();
    if let Value::Object(map) = config {
        walk(map, schema::TOP, "", "", dialect, &mut out);
    }
    out
}

/// A key BenchFlow 0.8 reads under `[solution]` as an environment variable:
/// `[solution] API_KEY = "..."` is `[solution.env] API_KEY`.
pub(crate) fn is_env_shorthand(key: &str, value: &Value) -> bool {
    let mut chars = key.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_uppercase() || c == '_');
    first_ok
        && chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && value.is_string()
}

fn walk(
    map: &Map<String, Value>,
    fields: &'static [Field],
    prefix: &str,
    table: &str,
    dialect: Dialect,
    out: &mut Vec<KeyIssue>,
) {
    for (key, value) in map {
        let path = join(prefix, key);
        let below = if table.is_empty() {
            key.clone()
        } else {
            format!("{table}.{key}")
        };
        let Some(field) = schema::field(fields, key, dialect) else {
            if std::ptr::eq(fields, schema::SOLUTION) && is_env_shorthand(key, value) {
                continue;
            }
            out.push(issue(key, &path, table, fields, dialect, value));
            continue;
        };
        match (field.shape, value) {
            (Shape::Table(sub), Value::Object(m)) => walk(m, sub, &path, &below, dialect, out),
            (Shape::Tables(sub) | Shape::StringsOrTables(sub), Value::Array(items)) => {
                for (i, item) in items.iter().enumerate() {
                    if let Value::Object(m) = item {
                        walk(m, sub, &format!("{path}[{i}]"), &below, dialect, out);
                    }
                }
            }
            _ => {}
        }
    }
}

fn issue(
    key: &str,
    path: &str,
    table: &str,
    fields: &'static [Field],
    dialect: Dialect,
    value: &Value,
) -> KeyIssue {
    let holds = schema::names(fields, dialect);
    let stale_opt_out = key == STALE_OPT_OUT && matches!(table, "environment" | "sandbox");
    let suggestion = if stale_opt_out {
        Some(format!("[metadata] {}", crate::convert::METADATA_OPT_OUT))
    } else {
        did_you_mean(key, holds.iter().copied())
    };
    // The opt-out moves to another table, so it is never offered as a
    // rename in place.
    let close = !stale_opt_out && suggestion.as_deref().is_some_and(|s| ratio(key, s) >= 0.8);
    let homes = rank_homes(
        schema::homes(key, dialect)
            .into_iter()
            .filter(|h| h != table)
            .collect(),
        dialect,
    );
    let problem = if homes.is_empty() {
        Problem::Unknown
    } else {
        Problem::Misplaced { homes }
    };
    KeyIssue {
        path: path.to_string(),
        key: key.to_string(),
        table: table.to_string(),
        suggestion,
        close,
        holds,
        problem,
        is_table: value.is_object(),
    }
}

/// Order a key's homes for a message: shallow tables first, and each table
/// once, by the dialect's own spelling. In `task.toml`, where Harbor's
/// `[environment]` and `[solution]` and BenchFlow 0.8's `[sandbox]` and
/// `[oracle]` are the same tables, only Harbor's spelling is named.
pub(crate) fn rank_homes(homes: Vec<String>, dialect: Dialect) -> Vec<String> {
    let canonical = |home: &str| -> String {
        if dialect != Dialect::TaskToml {
            return home.to_string();
        }
        home.split('.')
            .map(|seg| match seg {
                "sandbox" => "environment",
                "oracle" => "solution",
                other => other,
            })
            .collect::<Vec<_>>()
            .join(".")
    };
    let mut out: Vec<String> = Vec::new();
    for home in homes {
        let name = canonical(&home);
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out.sort_by_key(|h| depth(h));
    out
}

/// How deep a table is: 0 for the top level, 1 for `[agent]`, 2 for
/// `[environment.healthcheck]`.
pub(crate) fn depth(table: &str) -> usize {
    if table.is_empty() {
        0
    } else {
        table.split('.').count()
    }
}

/// How a table path reads in a message: `[environment.healthcheck]`, or "the
/// top level".
pub(crate) fn table_name(table: &str) -> String {
    if table.is_empty() {
        "the top level".into()
    } else {
        format!("[{table}]")
    }
}

/// A list of table paths for a message: "[agent], [verifier], and 3 more".
pub(crate) fn list_tables(tables: &[String]) -> String {
    let shown: Vec<String> = tables.iter().take(3).map(|t| table_name(t)).collect();
    let more = tables.len().saturating_sub(shown.len());
    let mut text = match shown.len() {
        0 => String::new(),
        1 => shown[0].clone(),
        2 => format!("{} and {}", shown[0], shown[1]),
        _ => format!("{}, {}, and {}", shown[0], shown[1], shown[2]),
    };
    if more > 0 {
        text.push_str(&format!(" (and {more} more)"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn finds_typos_and_misplaced_keys() {
        let cfg = json!({
            "task": {"name": "a/b", "multi_step_reward_strategy": "mean"},
            "environment": {"network_mod": "no-network", "timeout_sec": 5},
            "verifier": {"timeout_sec": 60},
            "steps": [{"name": "one", "agent": {"timout_sec": 3}}],
        });
        let issues = unknown_keys(&cfg, Dialect::TaskToml);
        let by_key = |k: &str| issues.iter().find(|i| i.key == k).unwrap().clone();
        let msrs = by_key("multi_step_reward_strategy");
        assert_eq!(msrs.path, "task.multi_step_reward_strategy");
        assert_eq!(
            msrs.problem,
            Problem::Misplaced {
                homes: vec!["".into()]
            }
        );
        let typo = by_key("network_mod");
        assert_eq!(typo.problem, Problem::Unknown);
        assert_eq!(typo.suggestion.as_deref(), Some("network_mode"));
        assert!(typo.close);
        let step = by_key("timout_sec");
        assert_eq!(step.path, "steps[0].agent.timout_sec");
        assert_eq!(step.table, "steps.agent");
        assert!(matches!(
            by_key("timeout_sec").problem,
            Problem::Misplaced { .. }
        ));
        assert_eq!(issues.len(), 4);
    }

    /// F1-B-r2 L2: the opt-out's first spelling names its new place, in
    /// both dialects, and is never offered as a rename in place.
    #[test]
    fn the_stale_opt_out_names_its_new_place() {
        let place = Some("[metadata] x-allow-cloud-metadata");
        for (cfg, dialect) in [
            (
                json!({"environment": {"allow_cloud_metadata": true}}),
                Dialect::TaskToml,
            ),
            (
                json!({"sandbox": {"allow_cloud_metadata": true}}),
                Dialect::Frontmatter,
            ),
        ] {
            let issues = unknown_keys(&cfg, dialect);
            assert_eq!(issues.len(), 1, "{issues:?}");
            assert_eq!(issues[0].problem, Problem::Unknown);
            assert_eq!(issues[0].suggestion.as_deref(), place);
            assert!(!issues[0].close);
        }
        let issues = unknown_keys(
            &json!({"agent": {"allow_cloud_metadata": true}}),
            Dialect::TaskToml,
        );
        assert_ne!(issues[0].suggestion.as_deref(), place);
    }

    #[test]
    fn open_tables_and_env_shorthands_are_not_checked() {
        let cfg = json!({
            "metadata": {"anything": 1},
            "environment": {"env": {"ANY": "x"}},
            "solution": {"API_KEY": "x", "lowercase": "y"},
        });
        let issues = unknown_keys(&cfg, Dialect::TaskToml);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].key, "lowercase");
    }

    #[test]
    fn frontmatter_reads_only_benchflow_spellings() {
        let cfg = json!({"sandbox": {"cpus": 1}, "verifier": {"collect": []}});
        let issues = unknown_keys(&cfg, Dialect::Frontmatter);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].key, "collect");
        assert!(unknown_keys(&cfg, Dialect::TaskToml).is_empty());
    }

    #[test]
    fn homes_name_each_table_once_shallowest_first() {
        let cfg = json!({"agent": {"cpus": 2}});
        let issues = unknown_keys(&cfg, Dialect::TaskToml);
        assert_eq!(
            issues[0].problem,
            Problem::Misplaced {
                homes: vec![
                    "environment".into(),
                    "verifier.environment".into(),
                    "steps.verifier.environment".into()
                ]
            }
        );
        let issues = unknown_keys(&json!({"agent": {"cpus": 2}}), Dialect::Frontmatter);
        assert_eq!(
            issues[0].problem,
            Problem::Misplaced {
                homes: vec![
                    "sandbox".into(),
                    "verifier.sandbox".into(),
                    "steps.verifier.sandbox".into()
                ]
            }
        );
    }

    #[test]
    fn lists_tables_for_messages() {
        assert_eq!(list_tables(&["".into()]), "the top level");
        assert_eq!(
            list_tables(&["a".into(), "b".into(), "c".into(), "d".into()]),
            "[a], [b], and [c] (and 1 more)"
        );
    }
}
