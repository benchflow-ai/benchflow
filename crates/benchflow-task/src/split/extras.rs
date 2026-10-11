//! The runtime extras of a split-layout or 0.8 frontmatter package: Harbor's
//! `[[steps]]`, `[[verifier.collect]]` hooks, Compose services in
//! `environment/docker-compose.yaml`, a separate verifier's own image, and
//! GPU types.
//!
//! The config was checked already ([`super::config`]); this reads the
//! values as written and refuses, by name, the parts the runtime cannot
//! honor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Map, Value};

use super::package::{Layout, Names};
use super::schema::Dialect;
use crate::extras::{Combine, Extras, Gate, Hook, StepSpec, Steps, Unreached, VerifierImage};
use crate::package::{ImageRef, Output, Ready, Unsupported};

/// What [`read`] needs besides the raw config.
pub(crate) struct Context<'a> {
    pub root: &'a Path,
    pub raw: &'a Value,
    pub names: &'a Names,
    pub layout: &'a Layout,
    /// The verifier runs in a sandbox of its own.
    pub separate: bool,
    /// The task's prebuilt image, when it names one.
    pub docker_image: Option<&'a str>,
    pub workdir: Option<&'a str>,
    pub gpu_types: &'a [String],
}

fn refuse(refused: &mut Vec<Unsupported>, field: impl Into<String>, reason: impl Into<String>) {
    refused.push(Unsupported {
        field: field.into(),
        reason: reason.into(),
        agents_only: false,
    });
}

fn table<'a>(v: &'a Value, key: &str) -> Option<&'a Map<String, Value>> {
    v.get(key).and_then(Value::as_object)
}

fn secs(v: Option<&Value>) -> Option<Duration> {
    v.and_then(Value::as_f64)
        .map(crate::values::duration_from_secs)
}

/// Read the extras, adding what the runtime cannot honor to `refused`.
pub(crate) fn read(cx: &Context<'_>, refused: &mut Vec<Unsupported>) -> Extras {
    let mut extras = Extras {
        gpu_types: cx.gpu_types.to_vec(),
        ..Extras::default()
    };
    let compose = cx.root.join("environment/docker-compose.yaml");
    if compose.is_file() {
        let compose = crate::compose::read(
            cx.root,
            "environment",
            "environment/docker-compose.yaml",
            refused,
        );
        extras.services = compose.services;
        extras.main_env = compose.env;
        extras.diagnostics = compose.diagnostics;
    }
    extras.hooks = hooks(cx, refused);
    if cx.separate {
        extras.verifier_image = verifier_image(cx, refused);
    }
    extras.steps = steps(cx, refused);
    extras
}

/// `[[verifier.collect]]`: commands run after the agent's turn and before
/// grading.
fn hooks(cx: &Context<'_>, refused: &mut Vec<Unsupported>) -> Vec<Hook> {
    let Some(Value::Array(items)) = cx.raw.get("verifier").and_then(|v| v.get("collect")) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(Value::as_object)
        .enumerate()
        .filter_map(|(i, m)| {
            let run = m.get("command").and_then(Value::as_str)?.to_string();
            let service = m
                .get("service")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| *s != "main")
                .map(str::to_string);
            let user = match m.get("user") {
                Some(Value::String(u)) => Some(u.clone()),
                Some(Value::Number(n)) => Some(n.to_string()),
                Some(_) => {
                    refuse(
                        refused,
                        format!("{}[{i}] user", cx.names.field("verifier", "collect")),
                        "must be a user name or uid",
                    );
                    return None;
                }
                None => None,
            };
            Some(Hook {
                run,
                service,
                timeout: secs(m.get("timeout_sec")).unwrap_or(Duration::from_secs(60)),
                user,
            })
        })
        .collect()
}

/// The separate verifier's image. task.toml follows Harbor's order: the
/// verifier environment's `docker_image`, then `tests/Dockerfile` (built
/// with `tests/` as the context), then the task's image. A 0.8 frontmatter
/// follows 0.8's: the verifier sandbox's `docker_image`, then, when the task
/// declares no verifier sandbox, the task's `docker_image`, then
/// `tests/Dockerfile`, then the task's Dockerfile.
fn verifier_image(cx: &Context<'_>, refused: &mut Vec<Unsupported>) -> Option<VerifierImage> {
    let env_key = match cx.names.dialect {
        Dialect::TaskToml => "environment",
        Dialect::Frontmatter => "sandbox",
    };
    let declared = cx
        .raw
        .get("verifier")
        .and_then(|v| table(v, env_key).or_else(|| table(v, "sandbox")));
    if let Some(env) = declared {
        for key in env.keys() {
            if key != "docker_image" {
                refuse(
                    refused,
                    cx.names.field(&format!("verifier.{env_key}"), key),
                    "a separate verifier's sandbox takes the task's resources and network; only its docker_image is read",
                );
            }
        }
    }
    if let Some(reference) = declared
        .and_then(|e| e.get("docker_image"))
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        return Some(VerifierImage {
            image: ImageRef::Pull {
                reference: reference.to_string(),
            },
            source: cx
                .names
                .field(&format!("verifier.{env_key}"), "docker_image"),
        });
    }
    let dir = cx.layout.verifier_dir;
    if cx.root.join(dir).join("docker-compose.yaml").is_file()
        && !cx.root.join(dir).join("Dockerfile").is_file()
    {
        refuse(
            refused,
            format!("{dir}/docker-compose.yaml"),
            "a verifier image is built from a Dockerfile, not a Compose file",
        );
        return None;
    }
    if cx.names.dialect == Dialect::Frontmatter && declared.is_none() && cx.docker_image.is_some() {
        // 0.8 takes the task's prebuilt image before tests/Dockerfile.
        return None;
    }
    cx.root
        .join(dir)
        .join("Dockerfile")
        .is_file()
        .then(|| VerifierImage {
            image: ImageRef::Build {
                context: PathBuf::from(dir),
                dockerfile: PathBuf::from(dir).join("Dockerfile"),
            },
            source: format!("{dir}/Dockerfile"),
        })
}

fn healthcheck(m: &Map<String, Value>) -> Option<Ready> {
    let get = |k: &str, d: f64| secs(m.get(k)).unwrap_or(crate::values::duration_from_secs(d));
    Some(Ready {
        run: m.get("command").and_then(Value::as_str)?.to_string(),
        interval: get("interval_sec", 5.0),
        timeout: get("timeout_sec", 30.0),
        start_period: get("start_period_sec", 0.0),
        start_interval: get("start_interval_sec", 5.0),
        retries: m
            .get("retries")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(3),
    })
}

/// A step's `artifacts` as outputs, paths resolved as the task's are.
fn step_outputs(
    items: &[Value],
    workdir: Option<&str>,
    field: &str,
    refused: &mut Vec<Unsupported>,
) -> Vec<Output> {
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let (source, destination, exclude, service) = match item {
            Value::String(s) => (s.clone(), None, Vec::new(), None),
            Value::Object(m) => (
                m.get("source")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                m.get("destination")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                m.get("exclude")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default(),
                m.get("service").and_then(Value::as_str).map(str::to_string),
            ),
            _ => continue,
        };
        if service.as_deref().is_some_and(|s| s.trim() != "main") {
            refuse(
                refused,
                format!("{field}[{i}]"),
                "collecting outputs from a Compose service is not supported yet",
            );
            continue;
        }
        let path = if source.starts_with('/') {
            source
        } else if let Some(w) = workdir {
            format!(
                "{}/{}",
                w.trim_end_matches('/'),
                source.trim_start_matches("./")
            )
        } else {
            refuse(
                refused,
                format!("{field}[{i}]"),
                "a relative artifact source depends on the image's working folder; make it absolute or set workdir",
            );
            continue;
        };
        let save_as = destination.unwrap_or_else(|| {
            path.trim_end_matches('/')
                .rsplit('/')
                .next()
                .filter(|n| !n.is_empty())
                .unwrap_or("artifact")
                .to_string()
        });
        out.push(Output {
            path,
            save_as,
            max_bytes: None,
            exclude,
        });
    }
    out
}

/// Harbor's `[[steps]]`: each step's prompt is `steps/<name>/instruction.md`,
/// its verifier the task's `tests/` with `steps/<name>/tests/` on top, its
/// reference solution `steps/<name>/solution/` (else the task's), and its
/// files `steps/<name>/workdir/` (with an optional `setup.sh`).
fn steps(cx: &Context<'_>, refused: &mut Vec<Unsupported>) -> Option<Steps> {
    let Some(Value::Array(items)) = cx.raw.get("steps") else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let steps_field = cx.names.field("", "steps");
    if cx.separate {
        refuse(
            refused,
            steps_field.clone(),
            "a separate verifier for each step is not supported yet: the agent's sandbox must stay up between steps, and the runtime grades a separate verifier after it stops",
        );
    }
    let mut stages = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let Some(m) = item.as_object() else { continue };
        let Some(name) = m.get("name").and_then(Value::as_str) else {
            continue;
        };
        let at = format!("{steps_field}[{i}]");
        let dir = PathBuf::from("steps").join(name);
        let prompt = match std::fs::read_to_string(cx.root.join(&dir).join("instruction.md")) {
            Ok(text) => super::strip_canary(&text),
            Err(e) => {
                refuse(
                    refused,
                    format!("steps/{name}/instruction.md"),
                    format!("the step's prompt cannot be read: {e}"),
                );
                String::new()
            }
        };
        let mut spec = StepSpec {
            name: name.to_string(),
            prompt,
            graded: true,
            ..StepSpec::default()
        };
        for (part, allowed) in [("agent", "timeout_sec"), ("verifier", "timeout_sec")] {
            let Some(t) = table(item, part) else { continue };
            for key in t.keys() {
                if key != allowed {
                    refuse(
                        refused,
                        format!("{at}.{part}.{key}"),
                        format!(
                            "a step's own [{part}] settings other than timeout_sec are not supported yet; the step uses the task's"
                        ),
                    );
                }
            }
            let timeout = secs(t.get("timeout_sec"));
            match part {
                "agent" => spec.timeout = timeout,
                _ => spec.verifier_timeout = timeout,
            }
        }
        spec.gate = match m.get("min_reward") {
            None | Some(Value::Null) => None,
            Some(Value::Number(n)) => n.as_f64().map(Gate::Reward),
            Some(Value::Object(keys)) => Some(Gate::Keys(
                keys.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|v| (k.clone(), v)))
                    .collect::<BTreeMap<_, _>>(),
            )),
            Some(_) => {
                refuse(
                    refused,
                    format!("{at}.min_reward"),
                    "must be a number or a table of numbers",
                );
                None
            }
        };
        spec.ready = table(item, "healthcheck").and_then(healthcheck);
        if let Some(Value::Array(a)) = m.get("artifacts") {
            spec.outputs = step_outputs(a, cx.workdir, &format!("{at}.artifacts"), refused);
        }
        let tests = dir.join("tests");
        if cx.root.join(cx.layout.verifier_dir).is_dir() {
            spec.verifier_dirs
                .push(PathBuf::from(cx.layout.verifier_dir));
        }
        if cx.root.join(&tests).is_dir() {
            spec.verifier_dirs.push(tests.clone());
        }
        let entry = &cx.layout.verifier_entry;
        if !spec
            .verifier_dirs
            .iter()
            .any(|d| cx.root.join(d).join(entry).is_file())
        {
            refuse(
                refused,
                format!("steps/{name}/tests/{entry}"),
                format!(
                    "the step has no verifier: neither steps/{name}/tests/{entry} nor {}/{entry} exists",
                    cx.layout.verifier_dir
                ),
            );
        }
        let solution = dir.join("solution");
        if cx
            .root
            .join(&solution)
            .join(&cx.layout.oracle_entry)
            .is_file()
        {
            spec.oracle_dir = Some(solution);
        }
        let workdir = dir.join("workdir");
        if cx.root.join(&workdir).is_dir() {
            spec.workdir = Some(workdir);
        }
        stages.push(spec);
    }
    let combine = match cx
        .raw
        .get("multi_step_reward_strategy")
        .and_then(Value::as_str)
    {
        Some("final") => Combine::Final,
        _ => Combine::Mean,
    };
    Some(Steps {
        stages,
        combine,
        // Harbor averages only the steps that ran, and "final" is the last
        // step that ran.
        unreached: Unreached::Exclude,
    })
}
