//! What grading needs, said as the reference's `check` says it: whether a
//! model judges, each judge's settings with the defaults it took marked, and
//! the worst-case grading time (each part's script timeout plus every judge
//! session one after another, before retries).

use std::path::Path;

use serde_json::Value;

use super::TaskDocument;
use super::files::script_models;
use super::judges::{Settings, resolve, roles_in};
use super::judgment;
use super::pyfmt::{criteria, fmt_duration, plural, str_of};
use super::pyval::{duration_s, int, table, truthy};

/// One role's share of one part (the task's verifier, or a stage's).
struct Part {
    stage: Option<String>,
    settings: Settings,
    samples: i128,
}

/// What one model role does across the parts.
#[derive(Default)]
struct Role {
    criteria: usize,
    behaviors: usize,
    sessions: i128,
    seconds: f64,
    parts: Vec<Part>,
}

/// The reference's `grading_plan`.
struct Plan {
    roles: Vec<(String, Role)>,
    scripts: Vec<(String, f64)>,
    unset_scripts: Vec<String>,
    human: usize,
    tests: usize,
    rubric: bool,
    seconds: f64,
    bounded: bool,
    script_models: Vec<String>,
}

/// What one verifier grades with: the task's own, or a stage's.
struct Grader {
    stage: Option<String>,
    rubric: Option<Value>,
    behaviors: Option<Value>,
    timeout: Option<Value>,
}

fn plan(config: &Value, rubric: Option<&Value>, behaviors: Option<&Value>, dir: &Path) -> Plan {
    let mut parts = vec![Grader {
        stage: None,
        rubric: rubric.cloned(),
        behaviors: behaviors.cloned(),
        timeout: table(config, &["verifier"]).get("timeout").cloned(),
    }];
    for (name, st) in table(config, &["stages"]) {
        let folder = dir.join("stages").join(name).join("verifier");
        let stage = Grader {
            stage: Some(name.clone()),
            rubric: judgment::read(&folder.join("rubric.json"), "rubric"),
            behaviors: judgment::read(&folder.join("behaviors.json"), "behaviors"),
            timeout: table(st, &["verifier"]).get("timeout").cloned(),
        };
        if truthy(stage.rubric.as_ref())
            || truthy(stage.behaviors.as_ref())
            || truthy(stage.timeout.as_ref())
        {
            parts.push(stage);
        }
    }
    let mut plan = Plan {
        roles: Vec::new(),
        scripts: Vec::new(),
        unset_scripts: Vec::new(),
        human: 0,
        tests: 0,
        rubric: truthy(rubric),
        seconds: 0.0,
        bounded: true,
        script_models: Vec::new(),
    };
    for Grader {
        stage,
        rubric,
        behaviors,
        timeout,
    } in parts
    {
        let label = stage
            .as_ref()
            .map_or_else(|| "task".to_string(), |s| format!("stage {s}"));
        match duration_s(timeout.as_ref()) {
            Some(s) => {
                plan.scripts.push((label, s));
                plan.seconds += s;
            }
            None if stage.is_none() || truthy(rubric.as_ref()) => plan.unset_scripts.push(label),
            None => {}
        }
        if let Some(Value::Array(crits)) = rubric.as_ref().and_then(|r| r.get("criteria")) {
            for c in crits {
                if let Value::Object(m) = c
                    && !m.contains_key("remove")
                {
                    let judge = m.get("judge").and_then(Value::as_str);
                    plan.human += usize::from(judge == Some("human"));
                    plan.tests +=
                        usize::from(matches!(judge, Some("test" | "rule" | "world" | "replay")));
                }
            }
        }
        for (role, counts) in roles_in(rubric.as_ref(), behaviors.as_ref()) {
            let settings = resolve(config, &role, stage.as_deref(), Some(dir));
            let samples = settings
                .get("samples")
                .and_then(int)
                .filter(|n| *n >= 1)
                .unwrap_or(1);
            let units = if settings.get("per").and_then(Value::as_str) == Some("criterion") {
                (counts.criteria + counts.behaviors) as i128
            } else {
                1
            };
            let seconds = duration_s(settings.get("timeout"));
            let i = match plan.roles.iter().position(|(r, _)| *r == role) {
                Some(i) => i,
                None => {
                    plan.roles.push((role.clone(), Role::default()));
                    plan.roles.len() - 1
                }
            };
            let entry = &mut plan.roles[i].1;
            entry.criteria += counts.criteria;
            entry.behaviors += counts.behaviors;
            entry.sessions += units * samples;
            entry.parts.push(Part {
                stage: stage.clone(),
                settings,
                samples,
            });
            match seconds {
                None => plan.bounded = false,
                Some(s) => {
                    let total = (units * samples) as f64 * s;
                    entry.seconds += total;
                    plan.seconds += total;
                }
            }
        }
    }
    plan.script_models = script_models(config, dir);
    plan.roles.sort_by(|a, b| a.0.cmp(&b.0));
    plan
}

/// A setting as the reference prints it: tables as `key value` pairs, lists
/// joined with commas, everything else as Python's `str()`.
fn shown(v: &Value) -> String {
    match v {
        Value::Object(m) => m
            .iter()
            .map(|(a, b)| format!("{a} {}", str_of(b)))
            .collect::<Vec<_>>()
            .join(" "),
        Value::Array(items) => items.iter().map(str_of).collect::<Vec<_>>().join(", "),
        other => str_of(other),
    }
}

/// The lines `check` prints about grading, then the episode settings a
/// runtime must honor when the task sets them.
pub(crate) fn lines(
    doc: &TaskDocument,
    rubric: Option<&Value>,
    behaviors: Option<&Value>,
    dir: &Path,
) -> Vec<String> {
    let config = doc.config_value();
    let plan = plan(config, rubric, behaviors, dir);
    let needs_model = !plan.roles.is_empty() || !plan.script_models.is_empty();
    let mut out = Vec::new();
    let calls = plan.script_models.join("; ");
    if !plan.roles.is_empty() {
        let uses: Vec<String> = plan
            .roles
            .iter()
            .map(|(role, e)| {
                let mut what = Vec::new();
                if e.criteria > 0 {
                    what.push(criteria(e.criteria));
                }
                if e.behaviors > 0 {
                    what.push(plural(e.behaviors, "behavior"));
                }
                format!("{role} judges {}", what.join(" and "))
            })
            .collect();
        let script = if calls.is_empty() {
            String::new()
        } else {
            format!("; and a script ({calls})")
        };
        out.push(format!(
            "grading needs a model: yes: {}{script}",
            uses.join("; ")
        ));
    } else if !calls.is_empty() {
        out.push(format!("grading needs a model: yes (a script): {calls}"));
    } else if plan.rubric || plan.tests > 0 {
        out.push(format!(
            "grading needs a model: no ({} decided by tests, rules, the world, or replay)",
            criteria(plan.tests)
        ));
    } else {
        out.push("grading needs a model: no (the verifier's scripts write the reward)".into());
    }
    if plan.human > 0
        && let Some(last) = out.last_mut()
    {
        last.push_str(&format!("; {} wait for human raters", criteria(plan.human)));
    }
    for (role, e) in &plan.roles {
        for part in &e.parts {
            let st = &part.settings;
            let mut keys = vec!["model", "harness", "samples"];
            if part.samples > 1 {
                keys.extend(["aggregate", "min_samples"]);
            }
            keys.extend(["per", "timeout", "brief", "budget"]);
            let mut items: Vec<String> = keys
                .iter()
                .filter_map(|k| {
                    st.get(k).map(|v| {
                        format!("{k} {}{}", shown(v), if st.is_filled(k) { "*" } else { "" })
                    })
                })
                .collect();
            if st.get("model").is_none() {
                items.insert(0, "model from the run".into());
            }
            if st.get("budget").is_none() && role == "agent" {
                items.push("no token budget (the timeout bounds it)".into());
            }
            let where_ = part
                .stage
                .as_ref()
                .map_or_else(String::new, |s| format!(" (stage {s})"));
            let defaults = if st.filled.is_empty() {
                ""
            } else {
                "; * marks a default"
            };
            out.push(format!(
                "judge {role}{where_}: {}{defaults}",
                items.join(", ")
            ));
        }
    }
    let sessions = |role: &str, e: &Role| -> String {
        let mut timeouts: Vec<Option<&Value>> = Vec::new();
        for p in &e.parts {
            let t = p.settings.get("timeout");
            if !timeouts.contains(&t) {
                timeouts.push(t);
            }
        }
        if timeouts.len() == 1 {
            let secs = duration_s(timeouts[0]).unwrap_or(0.0);
            format!("{role} {} x {}", e.sessions, fmt_duration(secs))
        } else {
            format!(
                "{role} {}, {}",
                plural(e.sessions as usize, "session"),
                fmt_duration(e.seconds)
            )
        }
    };
    let judged: Vec<String> = plan.roles.iter().map(|(r, e)| sessions(r, e)).collect();
    let scripts: Vec<String> = plan
        .scripts
        .iter()
        .map(|(label, s)| {
            let tail = if label == "task" {
                String::new()
            } else {
                format!(" ({label})")
            };
            format!("scripts {}{tail}", fmt_duration(*s))
        })
        .collect();
    let mut parts = [scripts.join(" + "), judged.join(" + ")]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" + ");
    if rubric.is_some_and(|r| truthy(Some(r)) && truthy(r.get("extends"))) && needs_model {
        parts.push_str(
            "; the shared rubrics it extends may add criteria, which check does not fetch",
        );
    }
    if !plan.bounded {
        out.push(
            "worst-case grading time: unknown, because a judge timeout is not a duration".into(),
        );
    } else if plan.seconds != 0.0 && plan.unset_scripts.is_empty() {
        out.push(format!(
            "worst-case grading time: {} ({parts}; in sequence, before retries)",
            fmt_duration(plan.seconds)
        ));
    } else if plan.seconds != 0.0 {
        out.push(format!(
            "worst-case grading time: {} ({parts}; in sequence, before retries), plus the scripts of the {}, which no [verifier] timeout \
bounds",
            fmt_duration(plan.seconds),
            plan.unset_scripts.join(", ")
        ));
    } else {
        out.push(
            "worst-case grading time: not bounded by the task: no [verifier] timeout is set".into(),
        );
    }
    let training = table(config, &["training"]);
    if training.contains_key("fork") || training.contains_key("max_concurrent") {
        let fork = training
            .get("fork")
            .map_or_else(|| "true".to_string(), |v| str_of(v).to_lowercase());
        let mc = training
            .get("max_concurrent")
            .map_or_else(|| "unlimited".to_string(), str_of);
        out.push(format!("episodes: fork {fork}, max_concurrent {mc}"));
    }
    out
}
