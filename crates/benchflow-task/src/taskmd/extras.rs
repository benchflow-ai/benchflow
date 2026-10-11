//! task.md's runtime extras: stages graded on their own, `[[sandbox.services]]`
//! or a Compose file, `[[verifier.snapshot]]`, `[sandbox] mounts`, a separate
//! verifier's own image, and `[sandbox] gpu_types`.
//!
//! The document checked clean already; this reads the values and refuses,
//! by name, the parts the runtime cannot honor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Map, Value};

use super::TaskDocument;
use super::pyfmt::str_of;
use super::pyval::{duration_s, num, obj, table};
use crate::diag::Diagnostic;
use crate::extras::{
    Combine, Extras, Gate, Hook, Mount, Service, StepSpec, Steps, Unreached, VerifierImage,
};
use crate::package::{ImageRef, Unsupported};

/// The Compose file a task.md package may name in `[sandbox] compose`.
pub(super) const COMPOSE_PATH: &str = "sandbox/docker-compose.yaml";

/// Stage keys that make a stage graded on its own.
const STAGE_RUNTIME_KEYS: &[&str] = &["gate", "outputs", "agent", "verifier", "ready", "mounts"];

fn refuse(refused: &mut Vec<Unsupported>, field: impl Into<String>, reason: impl Into<String>) {
    refused.push(Unsupported {
        field: field.into(),
        reason: reason.into(),
        agents_only: false,
    });
}

fn seconds(v: Option<&Value>) -> Option<Duration> {
    duration_s(v).map(crate::values::duration_from_secs)
}

/// `"<package path>:<sandbox path>"` entries as mounts.
pub(super) fn mounts(
    v: &Value,
    where_: &str,
    root: &Path,
    refused: &mut Vec<Unsupported>,
) -> Vec<Mount> {
    let Value::Array(items) = v else {
        refuse(
            refused,
            where_,
            "must be a list of \"<package path>:<sandbox path>\"",
        );
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items {
        let text = str_of(item);
        let Some((source, target)) = text.split_once(':') else {
            refuse(
                refused,
                format!("{where_} {text}"),
                "must be \"<package path>:<sandbox path>\"",
            );
            continue;
        };
        let rel = Path::new(source);
        if rel.is_absolute()
            || rel
                .components()
                .any(|c| c == std::path::Component::ParentDir)
            || !target.starts_with('/')
        {
            refuse(
                refused,
                format!("{where_} {text}"),
                "the source must be a path inside the package and the target an absolute path",
            );
            continue;
        }
        if !root.join(rel).exists() {
            refuse(
                refused,
                format!("{where_} {text}"),
                format!("{source} is not in the package"),
            );
            continue;
        }
        out.push(Mount {
            source: rel.to_path_buf(),
            target: target.to_string(),
        });
    }
    out
}

/// `[[sandbox.services]]` as services, with a warning (P013) for each
/// command string Compose cuts at a shell operator and an error (P014) for
/// each one it refuses.
fn services(
    v: &Value,
    root: &Path,
    refused: &mut Vec<Unsupported>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<Service> {
    let Value::Array(items) = v else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items {
        let s = obj(Some(item));
        let name = s.get("name").map(str_of).unwrap_or_default();
        let where_ = format!("[sandbox.services.{name}]");
        let mut ok = true;
        for key in s.keys() {
            if !["name", "image", "build", "command", "env", "ready"].contains(&key.as_str())
                && !key.starts_with("x-")
            {
                refuse(refused, format!("{where_} {key}"), "not a service key");
                ok = false;
            }
        }
        let image = match (s.get("image").and_then(Value::as_str), s.get("build")) {
            (Some(reference), None) => ImageRef::Pull {
                reference: reference.to_string(),
            },
            (None, Some(Value::String(folder))) => {
                let rel = Path::new(folder);
                if rel.is_absolute()
                    || rel
                        .components()
                        .any(|c| c == std::path::Component::ParentDir)
                    || !root.join(rel).join("Dockerfile").is_file()
                {
                    refuse(
                        refused,
                        format!("{where_} build"),
                        format!("{folder} must be a folder in the package with a Dockerfile"),
                    );
                    continue;
                }
                ImageRef::Build {
                    context: rel.to_path_buf(),
                    dockerfile: rel.join("Dockerfile"),
                }
            }
            _ => {
                refuse(refused, where_, "needs exactly one of image and build");
                continue;
            }
        };
        let command = match s.get("command") {
            None | Some(Value::Null) => Vec::new(),
            // 0.8 copied the string into a Compose file, and Compose split it.
            Some(Value::String(c)) => match crate::shellwords::parse(c) {
                Ok(split) => {
                    if let Some(dropped) = split.dropped {
                        let field = format!("{where_} command");
                        diagnostics.push(
                            crate::compose::cut_command(&field, &split.words, dropped)
                                .in_file("task.md"),
                        );
                    }
                    split.words
                }
                Err(why) => {
                    let field = format!("{where_} command");
                    diagnostics
                        .push(crate::compose::unsplittable_command(&field, why).in_file("task.md"));
                    refuse(refused, field, crate::compose::refused_command(why));
                    continue;
                }
            },
            Some(Value::Array(a)) => a.iter().map(str_of).collect(),
            Some(_) => {
                refuse(
                    refused,
                    format!("{where_} command"),
                    "must be a string or a list",
                );
                continue;
            }
        };
        let mut env = BTreeMap::new();
        for (k, v) in obj(s.get("env")) {
            let text = str_of(v);
            if text.contains("${") {
                refuse(
                    refused,
                    format!("{where_} env.{k}"),
                    "a value taken from the host's environment; services get only literal values",
                );
                ok = false;
                continue;
            }
            env.insert(k.clone(), text);
        }
        if !ok {
            continue;
        }
        out.push(Service {
            name,
            image,
            command,
            env,
            ready: s.get("ready").and_then(super::map::ready),
        });
    }
    out
}

/// `[[verifier.snapshot]]` as hooks. task.md runs each in a fresh container
/// of the task's image with its `reads` paths on volumes; this runtime runs
/// it in the agent's own container after every agent process is killed,
/// where the paths it reads are already in place.
fn snapshots(v: &Value, refused: &mut Vec<Unsupported>) -> Vec<Hook> {
    let Value::Array(items) = v else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (n, item) in items.iter().enumerate() {
        let s = obj(Some(item));
        let Some(run) = s.get("run").and_then(Value::as_str) else {
            continue;
        };
        for key in s.keys() {
            if !["run", "reads", "service", "timeout", "user"].contains(&key.as_str())
                && !key.starts_with("x-")
            {
                refuse(
                    refused,
                    format!("[verifier] snapshot[{n}] {key}"),
                    "not a snapshot key",
                );
            }
        }
        out.push(Hook {
            run: run.to_string(),
            service: s.get("service").and_then(Value::as_str).map(str::to_string),
            timeout: seconds(s.get("timeout")).unwrap_or(Duration::from_secs(60)),
            user: s.get("user").map(str_of),
        });
    }
    out
}

/// A separate verifier's image, in the spec's order: `[verifier.sandbox]
/// image`, then `verifier/Dockerfile` (built with `verifier/` as the
/// context), then the task's own image (`None` here).
fn verifier_image(cfg: &Value, root: &Path) -> Option<VerifierImage> {
    if let Some(reference) = table(cfg, &["verifier", "sandbox"])
        .get("image")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        return Some(VerifierImage {
            image: ImageRef::Pull {
                reference: reference.to_string(),
            },
            source: "[verifier.sandbox] image".into(),
        });
    }
    root.join("verifier/Dockerfile")
        .is_file()
        .then(|| VerifierImage {
            image: ImageRef::Build {
                context: PathBuf::from("verifier"),
                dockerfile: PathBuf::from("verifier/Dockerfile"),
            },
            source: "verifier/Dockerfile".into(),
        })
}

fn gate(v: &Value, where_: &str, refused: &mut Vec<Unsupported>) -> Option<Gate> {
    match v {
        Value::Object(keys) => {
            let mut out = BTreeMap::new();
            for (k, v) in keys {
                match num(v) {
                    Some(x) => {
                        out.insert(k.clone(), x);
                    }
                    None => refuse(refused, format!("{where_}.{k}"), "must be a number"),
                }
            }
            Some(Gate::Keys(out))
        }
        other => match num(other) {
            Some(x) => Some(Gate::Reward(x)),
            None => {
                refuse(
                    refused,
                    where_,
                    "must be a minimum reward, or a table of minimums by reward key",
                );
                None
            }
        },
    }
}

/// One stage's settings onto `spec`.
fn stage_settings(
    spec: &mut StepSpec,
    stage: &Map<String, Value>,
    root: &Path,
    refused: &mut Vec<Unsupported>,
) {
    let name = spec.name.clone();
    if let Some(g) = stage.get("gate") {
        spec.gate = gate(g, &format!("[stages.{name}] gate"), refused);
    }
    if let Some(o) = stage.get("outputs") {
        let mut r = super::map::Refusals::default();
        spec.outputs = super::map::outputs(o, &mut r);
        refused.extend(r.0);
    }
    for part in ["agent", "verifier"] {
        for (key, value) in obj(stage.get(part)) {
            match key.as_str() {
                "timeout" if part == "agent" => spec.timeout = seconds(Some(value)),
                "timeout" => spec.verifier_timeout = seconds(Some(value)),
                k if k.starts_with("x-") => {}
                k => refuse(
                    refused,
                    format!("[stages.{name}.{part}] {k}"),
                    format!(
                        "a stage's own [{part}] settings other than timeout are not supported yet; the stage uses the task's"
                    ),
                ),
            }
        }
    }
    if let Some(r) = stage.get("ready") {
        spec.ready = super::map::ready(r);
    }
    if let Some(m) = stage.get("mounts") {
        spec.mounts = mounts(m, &format!("[stages.{name}] mounts"), root, refused);
    }
    let own = PathBuf::from("stages").join(&name);
    if root.join(&own).join("verifier/test.sh").is_file() {
        spec.verifier_dirs = vec![own.join("verifier")];
    } else if root.join(&own).join("verifier").is_dir() {
        refuse(
            refused,
            format!("stages/{name}/verifier/"),
            "has no test.sh, the script that grades the stage",
        );
    }
    if root.join(&own).join("oracle/solve.sh").is_file() {
        spec.oracle_dir = Some(own.join("oracle"));
    }
}

/// Stages graded one after another, when any stage is graded on its own
/// (a gate, outputs, its own settings or files). Otherwise `None`: the
/// stages are later prompts of one session, graded once at the end.
fn steps(
    doc: &TaskDocument,
    cfg: &Value,
    root: &Path,
    refused: &mut Vec<Unsupported>,
) -> Option<Steps> {
    let stages = table(cfg, &["stages"]);
    // Each name is a folder in the task (`stages/<name>/`) and in the
    // trial (`steps/<name>/`), so it must be one plain folder name.
    let bad: Vec<&String> = stages
        .keys()
        .filter(|n| !crate::split::config::valid_step_name(n))
        .collect();
    if !bad.is_empty() {
        for name in bad {
            refuse(
                refused,
                format!("[stages.{name:?}]"),
                "a stage name must be a portable folder name: no slashes, no `.` or `..`, no control characters",
            );
        }
        return None;
    }
    let graded = stages.iter().any(|(name, st)| {
        obj(Some(st))
            .keys()
            .any(|k| STAGE_RUNTIME_KEYS.contains(&k.as_str()))
            || root.join("stages").join(name).is_dir()
    });
    if !graded {
        return None;
    }
    let mut out = Vec::new();
    let mut previous: Option<String> = None;
    for (index, (name, st)) in stages.iter().enumerate() {
        let st = obj(Some(st));
        let unlock = st.get("unlock").and_then(Value::as_str);
        let mut spec = StepSpec {
            name: name.clone(),
            graded: true,
            ..StepSpec::default()
        };
        if unlock == Some("at_start") {
            if index != 0 {
                continue; // refused by the stage mapping
            }
            spec.prompt = doc.agent_view();
        } else {
            let chained = unlock == Some("on_submit")
                || (previous.is_some()
                    && unlock == previous.as_ref().map(|p| format!("after:{p}")).as_deref());
            if !chained {
                continue; // refused by the stage mapping
            }
            if out.is_empty() {
                // The instruction is a turn of its own before the first
                // stage unlocks; no stage grades it.
                if stages.contains_key("instruction") {
                    refuse(
                        refused,
                        "[stages.instruction]",
                        "the opening turn is named instruction in the trial's record; rename this stage, or make the first stage unlock at_start",
                    );
                }
                out.push(StepSpec {
                    name: "instruction".into(),
                    prompt: doc.agent_view(),
                    graded: false,
                    ..StepSpec::default()
                });
            }
            spec.prompt = doc.stage_prompt(name).unwrap_or_default();
        }
        stage_settings(&mut spec, st, root, refused);
        out.push(spec);
        previous = Some(name.clone());
    }
    let v = table(cfg, &["verifier"]);
    Some(Steps {
        stages: out,
        combine: match v.get("combine_stages").and_then(Value::as_str) {
            Some("final") => Combine::Final,
            _ => Combine::Mean,
        },
        unreached: match v.get("unreached_stages").and_then(Value::as_str) {
            Some("exclude") => Unreached::Exclude,
            _ => Unreached::Zero,
        },
    })
}

/// Read the extras of a checked document.
pub(super) fn read(
    doc: &TaskDocument,
    cfg: &Value,
    root: &Path,
    separate: bool,
    refused: &mut Vec<Unsupported>,
) -> Extras {
    let sb = table(cfg, &["sandbox"]);
    let mut extras = Extras {
        gpu_types: match sb.get("gpu_types") {
            Some(Value::Array(a)) => a.iter().map(str_of).collect(),
            Some(Value::String(s)) => vec![s.clone()],
            _ => Vec::new(),
        },
        ..Extras::default()
    };
    if let Some(m) = sb.get("mounts") {
        extras.mounts = mounts(m, "[sandbox] mounts", root, refused);
    }
    match (sb.get("services"), sb.get("compose")) {
        (Some(_), Some(_)) => refuse(
            refused,
            "[sandbox] services",
            "[[sandbox.services]] and [sandbox] compose both declare services; declare them in one place",
        ),
        (Some(s), None) => {
            extras.services = services(s, root, refused, &mut extras.diagnostics);
        }
        (None, Some(Value::String(path))) if path == COMPOSE_PATH => {
            let compose = crate::compose::read(root, "sandbox", COMPOSE_PATH, refused);
            extras.services = compose.services;
            extras.main_env = compose.env;
            extras.diagnostics = compose.diagnostics;
        }
        (None, Some(other)) => refuse(
            refused,
            "[sandbox] compose",
            format!(
                "{}: only {COMPOSE_PATH} is read, the Compose file beside the agent's Dockerfile",
                str_of(other)
            ),
        ),
        (None, None) => {}
    }
    if let Some(s) = table(cfg, &["verifier"]).get("snapshot") {
        extras.hooks = snapshots(s, refused);
    }
    if separate {
        extras.verifier_image = verifier_image(cfg, root);
    }
    extras.steps = steps(doc, cfg, root, refused);
    if extras.steps.is_some() && separate {
        refuse(
            refused,
            "[verifier] isolation",
            "\"separate\" with stages graded on their own is not supported yet: the agent's sandbox must stay up between stages",
        );
    }
    extras
}
