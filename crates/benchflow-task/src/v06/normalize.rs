//! 0.8's authoring shorthands and profiles, expanded the way
//! `normalize_task_document_frontmatter` does before the config is checked.
//!
//! | Shorthand | Expands to |
//! |---|---|
//! | `name: hello` | `task.name: benchflow/hello` (a `/` keeps your org) |
//! | `image: ubuntu:24.04` | `sandbox.docker_image` |
//! | `verifier: verifier/` | `benchflow.verifier.path`, `.spec`, `.entrypoint` |
//! | `oracle: oracle/` | `benchflow.oracle.path` |
//! | `profile:` / `profiles:` | A named bundle of defaults, under your keys |

use std::sync::LazyLock;

use serde_json::{Map, Value, json};

/// The profile for split-layout compatibility, which 0.8 names after another
/// framework.
const UNLISTED_PROFILE: &str = "harbor-compatible";

/// The profiles 0.8 ships, by name.
static PROFILES: LazyLock<Map<String, Value>> = LazyLock::new(|| {
    let v = json!({
        "code-change": {
            "schema_version": "1.3",
            "metadata": {"category": "code-change"},
            "agent": {"timeout_sec": 7200, "network_mode": "no-network"},
            "verifier": {"timeout_sec": 1200, "user": "root"},
            "sandbox": {
                "docker_image": "ubuntu:24.04",
                "network_mode": "no-network",
                "cpus": 4,
                "memory_mb": 8192,
                "storage_mb": 10240,
                "workdir": "/repo"
            },
            "benchflow": {
                "document_version": "0.3",
                "prompt": {"composition": "append", "order": ["base", "role", "scene", "turn"]}
            }
        },
        "harbor-compatible": {
            "schema_version": "1.3",
            "metadata": {"category": "harbor-compatible"},
            "sandbox": {"cpus": 1, "memory_mb": 2048},
            "benchflow": {"compatibility": {"harbor": {"export": "supported"}}}
        },
        "reward-kit": {
            "benchflow": {
                "verifier": {
                    "spec": "verifier/verifier.md",
                    "rubric": "verifier/rubrics/verifier.md",
                    "entrypoint": "verifier/reward_kit/reward.py",
                    "implementation": {"type": "reward-kit"}
                }
            }
        },
        "acceptance-live": {
            "benchflow": {
                "evidence": {
                    "acceptance_live": {
                        "workspace": {"source": "current-worktree", "target": "/repo"},
                        "calibration": {"from": "calibration.report", "reruns": 1, "flake_rate_max": 0.0},
                        "cases": [
                            {"name": "live-oracle-rerun", "type": "oracle", "reruns": 1,
                             "expect": {"reward_min": 0.99, "flake_rate_max": 0.0}},
                            {"name": "live-reference-verifier", "type": "reference", "reruns": 1,
                             "expect": {"reward_min": 0.99, "flake_rate_max": 0.0}}
                        ]
                    }
                }
            }
        },
        "multi-agent": {
            "agents": {
                "roles": {
                    "architect": {"agent": "codex-acp", "model": "gpt-5.5", "reasoning_effort": "xhigh",
                                  "capabilities": ["code-edit", "tests"]},
                    "implementer": {"agent": "codex-acp", "model": "gpt-5.5", "reasoning_effort": "xhigh",
                                    "capabilities": ["code-edit", "tests"]},
                    "reviewer": {"agent": "claude-agent-acp", "model": "claude-sonnet-5",
                                 "capabilities": ["review"]}
                }
            },
            "scenes": [
                {"name": "design", "turns": [{"role": "architect"}]},
                {"name": "implement", "turns": [{"role": "implementer"}]},
                {"name": "review", "turns": [{"role": "reviewer"}]}
            ]
        },
        "leaderboard-local": {
            "benchflow": {
                "evidence": {"acceptance_live": {"leaderboard": {"required": true, "max_flake_rate": 0.0}}}
            }
        }
    });
    match v {
        Value::Object(m) => m,
        _ => unreachable!("a JSON object literal"),
    }
});

/// What normalizing changed, for diagnostics.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Expanded {
    /// The profiles applied, in order.
    pub profiles: Vec<String>,
}

/// Expand shorthands and profiles. `Err` is a message for the key at fault
/// (as a dotted path, and the message).
pub(crate) fn normalize(
    front: &Map<String, Value>,
) -> Result<(Map<String, Value>, Expanded), (String, String)> {
    let mut raw = front.clone();
    let profiles = parse_profiles(&raw)?;
    let mut defaults = Map::new();
    for p in &profiles {
        if let Some(Value::Object(bundle)) = PROFILES.get(p) {
            defaults = deep_merge(&defaults, bundle);
        }
    }
    let name = raw.remove("name");
    let image = raw.remove("image");
    let verifier_path = pop_path(&mut raw, "verifier")?;
    let oracle_path = pop_path(&mut raw, "oracle")?;
    raw.remove("profile");
    raw.remove("profiles");
    let name_explicit = has_nested(&raw, &["task", "name"]);
    let image_explicit = has_nested(&raw, &["sandbox", "docker_image"]);
    let mut out = deep_merge(&defaults, &raw);
    if let Some(v) = name.filter(|v| !v.is_null())
        && !name_explicit
    {
        let Some(text) = v.as_str().map(str::trim).filter(|t| !t.is_empty()) else {
            return Err(("name".into(), "`name` must be a non-empty string".into()));
        };
        let full = if text.contains('/') {
            text.to_string()
        } else {
            format!("benchflow/{text}")
        };
        ensure_mapping(&mut out, "task")?.insert("name".into(), Value::String(full));
    }
    if let Some(v) = image.filter(|v| !v.is_null())
        && !image_explicit
    {
        let Some(text) = v.as_str().map(str::trim).filter(|t| !t.is_empty()) else {
            return Err(("image".into(), "`image` must be a non-empty string".into()));
        };
        ensure_mapping(&mut out, "sandbox")?
            .insert("docker_image".into(), Value::String(text.to_string()));
    }
    if let Some(path) = verifier_path {
        let base = safe_path(&path, "verifier")?;
        let base = base.trim_end_matches('/').to_string();
        let bf = ensure_mapping(&mut out, "benchflow")?;
        let verifier = ensure_mapping(bf, "verifier")?;
        verifier
            .entry("path")
            .or_insert_with(|| Value::String(format!("{base}/")));
        verifier
            .entry("spec")
            .or_insert_with(|| Value::String(format!("{base}/verifier.md")));
        verifier
            .entry("entrypoint")
            .or_insert_with(|| Value::String(format!("{base}/test.sh")));
    }
    if let Some(path) = oracle_path {
        let base = safe_path(&path, "oracle")?;
        let bf = ensure_mapping(&mut out, "benchflow")?;
        let oracle = ensure_mapping(bf, "oracle")?;
        oracle
            .entry("path")
            .or_insert_with(|| Value::String(format!("{}/", base.trim_end_matches('/'))));
    }
    if !profiles.is_empty() {
        let bf = ensure_mapping(&mut out, "benchflow")?;
        let authoring = ensure_mapping(bf, "authoring")?;
        authoring
            .entry("profiles")
            .or_insert_with(|| Value::Array(profiles.iter().cloned().map(Value::String).collect()));
        authoring.entry("normalized").or_insert(Value::Bool(true));
    }
    Ok((out, Expanded { profiles }))
}

fn parse_profiles(front: &Map<String, Value>) -> Result<Vec<String>, (String, String)> {
    let mut out: Vec<String> = Vec::new();
    for key in ["profile", "profiles"] {
        let Some(raw) = front.get(key).filter(|v| !v.is_null()) else {
            continue;
        };
        let values: Vec<&Value> = match raw {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        for v in values {
            let Some(name) = v.as_str().map(str::trim).filter(|s| !s.is_empty()) else {
                return Err((key.into(), format!("`{key}` entries must be profile names")));
            };
            if !PROFILES.contains_key(name) {
                // A profile named after another framework is still read, but
                // not suggested or listed: BenchFlow's text names none.
                let mut known: Vec<&String> = PROFILES
                    .keys()
                    .filter(|k| k.as_str() != UNLISTED_PROFILE)
                    .collect();
                known.sort();
                let known: Vec<&str> = known.iter().map(|s| s.as_str()).collect();
                // The unlisted profile is suggested only to someone whose
                // typo already starts with its name, so it is their word
                // being repeated, never bench's.
                let names_it = name.to_ascii_lowercase().starts_with("harb");
                let guess = crate::suggest::did_you_mean(name, known.iter().copied())
                    .or_else(|| {
                        names_it
                            .then(|| crate::suggest::did_you_mean(name, [UNLISTED_PROFILE]))
                            .flatten()
                    })
                    .map(|g| format!("did you mean {g:?}? "))
                    .unwrap_or_default();
                return Err((
                    key.into(),
                    format!(
                        "unknown task.md profile {name:?}; {guess}known profiles: {}",
                        known.join(", ")
                    ),
                ));
            }
            if !out.iter().any(|p| p == name) {
                out.push(name.to_string());
            }
        }
    }
    Ok(out)
}

/// A string `verifier:` or `oracle:` is a path shorthand; a mapping is the
/// real table. Returns the path and removes the key.
fn pop_path(front: &mut Map<String, Value>, key: &str) -> Result<Option<String>, (String, String)> {
    match front.get(key) {
        None | Some(Value::Null) | Some(Value::Object(_)) => Ok(None),
        Some(Value::String(s)) if !s.trim().is_empty() => {
            let path = s.trim().to_string();
            front.remove(key);
            Ok(Some(path))
        }
        Some(_) => Err((
            key.into(),
            format!("`{key}` must be a mapping or a non-empty path"),
        )),
    }
}

fn safe_path(path: &str, key: &str) -> Result<String, (String, String)> {
    if path.starts_with('/') || path.split('/').any(|p| p == "..") {
        return Err((
            key.into(),
            format!("the `{key}` path must be relative and stay inside the package"),
        ));
    }
    Ok(path.to_string())
}

fn has_nested(map: &Map<String, Value>, path: &[&str]) -> bool {
    let mut cur = map;
    for (i, key) in path.iter().enumerate() {
        match cur.get(*key) {
            Some(_) if i + 1 == path.len() => return true,
            Some(Value::Object(m)) => cur = m,
            _ => return false,
        }
    }
    false
}

fn ensure_mapping<'m>(
    map: &'m mut Map<String, Value>,
    key: &str,
) -> Result<&'m mut Map<String, Value>, (String, String)> {
    let entry = map
        .entry(key.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if entry.is_null() {
        *entry = Value::Object(Map::new());
    }
    match entry {
        Value::Object(m) => Ok(m),
        _ => Err((
            key.into(),
            format!("`{key}` must be a mapping after normalization"),
        )),
    }
}

/// Merge `over` onto `base`; tables merge key by key, anything else is
/// replaced.
pub(crate) fn deep_merge(
    base: &Map<String, Value>,
    over: &Map<String, Value>,
) -> Map<String, Value> {
    let mut merged = base.clone();
    for (k, v) in over {
        match (merged.get_mut(k), v) {
            (Some(Value::Object(b)), Value::Object(o)) => {
                let m = deep_merge(b, o);
                *b = m;
            }
            _ => {
                merged.insert(k.clone(), v.clone());
            }
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(v: Value) -> Map<String, Value> {
        match v {
            Value::Object(m) => m,
            _ => panic!("not an object"),
        }
    }

    #[test]
    fn expands_shorthands() {
        let (out, _) = normalize(&obj(json!({
            "name": "hello", "image": "ubuntu:24.04", "verifier": "verifier/", "oracle": "oracle"
        })))
        .unwrap();
        assert_eq!(out["task"]["name"], "benchflow/hello");
        assert_eq!(out["sandbox"]["docker_image"], "ubuntu:24.04");
        assert_eq!(
            out["benchflow"]["verifier"]["entrypoint"],
            "verifier/test.sh"
        );
        assert_eq!(out["benchflow"]["oracle"]["path"], "oracle/");
        assert!(out.get("name").is_none() && out.get("verifier").is_none());
    }

    #[test]
    fn explicit_keys_win_over_shorthands_and_profiles() {
        let (out, ex) = normalize(&obj(json!({
            "name": "x", "task": {"name": "acme/y"},
            "profile": "code-change", "sandbox": {"cpus": 2}
        })))
        .unwrap();
        assert_eq!(out["task"]["name"], "acme/y");
        assert_eq!(out["sandbox"]["cpus"], 2);
        assert_eq!(out["sandbox"]["memory_mb"], 8192);
        assert_eq!(ex.profiles, vec!["code-change".to_string()]);
        assert_eq!(out["benchflow"]["authoring"]["normalized"], true);
    }

    #[test]
    fn refuses_bad_shorthands() {
        assert!(normalize(&obj(json!({"profile": "nope"}))).is_err());
        assert!(normalize(&obj(json!({"verifier": "/abs"}))).is_err());
        assert!(normalize(&obj(json!({"oracle": 3}))).is_err());
        assert!(normalize(&obj(json!({"name": ""}))).is_err());
        let (out, _) = normalize(&obj(json!({"verifier": {"timeout_sec": 5}}))).unwrap();
        assert_eq!(out["verifier"]["timeout_sec"], 5);
    }

    /// 0.8's split-layout compatibility profile is still read, but the
    /// unknown-profile message neither lists nor suggests it, since its
    /// name is another framework's.
    #[test]
    fn the_compatibility_profile_is_read_but_not_listed() {
        let (out, ex) = normalize(&obj(json!({"profile": UNLISTED_PROFILE}))).unwrap();
        assert_eq!(ex.profiles, vec![UNLISTED_PROFILE.to_string()]);
        assert_eq!(out["sandbox"]["memory_mb"], 2048);
        for typo in [
            "harbor-compatibl",
            "harbor-compat",
            "Harbour-compatible",
            "nope",
            "compatible",
            "split-compatible",
            "split-layout-compatible",
            "tbench-compatible",
            "compatibility",
        ] {
            let (_, message) = normalize(&obj(json!({ "profile": typo }))).unwrap_err();
            // The message repeats what was typed, then what is known.
            let (_, after) = message.split_once(&format!("{typo:?}")).unwrap();
            let (guess, known) = after.split_once("known profiles: ").unwrap();
            assert!(!known.to_lowercase().contains("harbor"), "{message}");
            // A near miss of the unlisted name is pointed at it.
            let near = typo.to_ascii_lowercase().starts_with("harb");
            if !near {
                assert!(!message.to_lowercase().contains("harbor"), "{message}");
            }
            assert_eq!(
                guess.contains(&format!("did you mean {UNLISTED_PROFILE:?}?")),
                near,
                "{message}"
            );
        }
    }
}
