//! Whether a TOML file is a RewardKit judge file, as RewardKit's own
//! `JudgeTomlConfig` reads it (task.md's `tools/convert.py`,
//! `rk_judge_toml`). A task whose verifier holds one calls a model while it
//! grades.

use crate::convert::value::{Table, Val};

#[derive(Clone, Copy)]
enum Kind {
    Str,
    Int,
    Float,
    Bool,
    Strs,
    Dict,
    Mcp,
    OneOf(&'static [&'static str]),
}

/// (field, kind, required)
type Fields = &'static [(&'static str, Kind, bool)];

const EFFORTS: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];
const AGGREGATIONS: &[&str] = &[
    "weighted-mean",
    "weighted-sum",
    "all-pass",
    "any-pass",
    "threshold",
    "required-pass",
    "weighted_mean",
    "all_pass",
    "any_pass",
    "required_pass",
];

const JUDGE: Fields = &[
    ("judge", Kind::Str, false),
    ("model", Kind::Str, false),
    ("version", Kind::Str, false),
    ("files", Kind::Strs, false),
    ("mode", Kind::OneOf(&["batched", "individual"]), false),
    ("timeout", Kind::Int, false),
    ("reasoning_effort", Kind::OneOf(EFFORTS), false),
    ("isolated", Kind::Bool, false),
    ("cwd", Kind::Str, false),
    ("reference", Kind::Str, false),
    ("atif_trajectory", Kind::Str, false),
    ("mcp_servers", Kind::Mcp, false),
    ("weight", Kind::Float, false),
    ("prompt_template", Kind::Str, false),
    ("samples", Kind::Int, false),
    ("guard", Kind::OneOf(&["off", "flag", "penalize"]), false),
];
const CRITERION: Fields = &[
    ("description", Kind::Str, true),
    (
        "type",
        Kind::OneOf(&["binary", "likert", "numeric", "rubric"]),
        false,
    ),
    ("name", Kind::Str, false),
    ("id", Kind::Str, false),
    ("points", Kind::Int, false),
    ("levels", Kind::Strs, false),
    ("min", Kind::Float, false),
    ("max", Kind::Float, false),
    ("weight", Kind::Float, false),
    ("files", Kind::Strs, false),
    ("negate", Kind::Bool, false),
    ("optional", Kind::Bool, false),
    ("annotations", Kind::Dict, false),
];
const SCORING: Fields = &[
    ("aggregation", Kind::OneOf(AGGREGATIONS), false),
    ("threshold", Kind::Float, false),
];
const MCP: Fields = &[
    ("name", Kind::Str, true),
    (
        "transport",
        Kind::OneOf(&["stdio", "sse", "streamable-http", "http"]),
        false,
    ),
    ("url", Kind::Str, false),
    ("command", Kind::Str, false),
    ("args", Kind::Strs, false),
    ("allowed_tools", Kind::Strs, false),
];

/// pydantic v2's lax reading of `v` as `kind`.
fn lax(v: &Val, kind: Kind) -> bool {
    match kind {
        Kind::OneOf(allowed) => v.as_str().is_some_and(|s| allowed.contains(&s)),
        Kind::Str => matches!(v, Val::Str(_)),
        Kind::Int => match v {
            Val::Int(_) | Val::Bool(_) => true,
            Val::Float(f) => f.fract() == 0.0 && f.is_finite(),
            Val::Str(s) => {
                let t = s.trim();
                let d = t.strip_prefix(['+', '-']).unwrap_or(t);
                !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())
            }
            _ => false,
        },
        Kind::Float => match v {
            Val::Int(_) | Val::Bool(_) | Val::Float(_) => true,
            Val::Str(s) => crate::convert::value::py_float_str(s).is_some(),
            _ => false,
        },
        Kind::Bool => match v {
            Val::Bool(_) => true,
            Val::Int(i) => *i == 0 || *i == 1,
            Val::Str(s) => matches!(
                s.trim().to_lowercase().as_str(),
                "1" | "on" | "t" | "true" | "y" | "yes" | "0" | "off" | "f" | "false" | "n" | "no"
            ),
            _ => false,
        },
        Kind::Strs => {
            matches!(v, Val::List(items) if items.iter().all(|x| matches!(x, Val::Str(_))))
        }
        Kind::Dict => matches!(v, Val::Table(_)),
        Kind::Mcp => false,
    }
}

/// Check one table, returning the fields it set (valid or not).
fn fields(data: Option<&Val>, spec: Fields, errors: &mut usize) -> Table {
    let empty = Table::new();
    let table = match data {
        None => &empty,
        Some(Val::Table(t)) => t,
        Some(_) => {
            *errors += 1;
            &empty
        }
    };
    let mut ok = Table::new();
    for (k, v) in table {
        let name = if k == "atif-trajectory" && spec.iter().any(|(f, _, _)| *f == "atif_trajectory")
        {
            "atif_trajectory"
        } else {
            k.as_str()
        };
        let Some((_, kind, _)) = spec.iter().find(|(f, _, _)| *f == name) else {
            *errors += 1;
            continue;
        };
        let valid = match kind {
            Kind::Mcp => match v {
                Val::List(servers) => {
                    for s in servers {
                        let m = fields(Some(s), MCP, errors);
                        let transport = match m.get("transport").and_then(Val::as_str) {
                            Some("http") => "streamable-http",
                            Some(t) => t,
                            None => "sse",
                        };
                        let has = |key: &str| m.get(key).is_some_and(Val::truthy);
                        if (transport == "sse" || transport == "streamable-http") && !has("url") {
                            *errors += 1;
                        }
                        if transport == "stdio" && !has("command") {
                            *errors += 1;
                        }
                    }
                    true
                }
                _ => false,
            },
            other => lax(v, *other),
        };
        if valid {
            ok.insert(name.to_string(), v.clone());
        } else {
            *errors += 1;
        }
    }
    for (name, _, required) in spec {
        if *required && !ok.contains_key(*name) {
            *errors += 1;
        }
    }
    ok
}

/// Whether `data` (a parsed TOML file) is a valid RewardKit judge file.
pub(crate) fn is_judge_toml(data: &Table) -> bool {
    let mut errors = 0usize;
    errors += data
        .keys()
        .filter(|k| !matches!(k.as_str(), "judge" | "criterion" | "scoring"))
        .count();
    let crits: &[Val] = match data.get("criterion") {
        Some(Val::List(items)) if !items.is_empty() => items,
        _ => {
            errors += 1;
            &[]
        }
    };
    fields(data.get("judge"), JUDGE, &mut errors);
    for c in crits {
        fields(Some(c), CRITERION, &mut errors);
    }
    fields(data.get("scoring"), SCORING, &mut errors);
    errors == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::value::parse_toml;

    #[test]
    fn recognizes_judge_files() {
        let ok = parse_toml("[judge]\njudge = \"anthropic/claude-sonnet-5\"\n\n[[criterion]]\ndescription = \"Is it right?\"\ntype = \"binary\"\n").unwrap();
        assert!(is_judge_toml(&ok));
        let rubric = parse_toml("[[criterion]]\ntitle = \"Delivery\"\n").unwrap();
        assert!(!is_judge_toml(&rubric));
        let none = parse_toml("[judge]\njudge = \"x\"\n").unwrap();
        assert!(!is_judge_toml(&none));
        let mcp = parse_toml("[judge]\nmcp_servers = [{ name = \"s\", transport = \"stdio\" }]\n[[criterion]]\ndescription = \"d\"\n").unwrap();
        assert!(!is_judge_toml(&mcp), "stdio needs a command");
    }
}
