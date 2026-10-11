//! `bench branch`: fork a rollout from a checkpoint into children, each
//! graded; `bench branches`: show a branch job's forks.

use std::path::{Path, PathBuf};

use benchflow_run::trial::branch::{self, ChildPlan, ForkPlan, ForkPoint};
use benchflow_run::{Attempt, source};
use serde_json::{Value, json};

use crate::cli::{BranchArgs, BranchesArgs};
use crate::cmd::engine;
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::ui::{self, bold, dim, plural, ui, warn_mark};
use crate::{note, say};

/// Agents whose subscription logins rotate their tokens on refresh: two
/// children refreshing at once can invalidate each other's.
const ROTATING: &[(&str, &[&str])] = &[
    (
        "claude",
        &[
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_OAUTH_TOKEN",
            "ANTHROPIC_AUTH_TOKEN",
        ],
    ),
    ("codex", &["OPENAI_API_KEY", "CODEX_API_KEY"]),
    ("gemini", &["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
];

pub async fn run(args: BranchArgs) -> Result<Exit> {
    let at = ForkPoint::parse(&args.at).map_err(CliError::usage)?;
    let continue_parent = match args.parent.as_str() {
        "discard" => false,
        "continue" => true,
        other => {
            return Err(CliError::usage(format!(
                "--parent takes discard or continue, not `{other}`"
            )));
        }
    };
    let children = match (args.children, args.child.is_empty()) {
        (Some(k), _) if k < 2 => {
            return Err(CliError::usage(
                "--children takes 2 or more (a group of one has no advantage)",
            ));
        }
        (Some(k), _) => ForkPlan::same_prompt(k),
        (None, false) => args
            .child
            .iter()
            .map(|c| ChildPlan::parse(c))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(CliError::usage)?,
        (None, true) => {
            return Err(CliError::usage("a fork needs children")
                .next("add --children K, or --child label=<name> once per child"));
        }
    };
    let project = context::project()?;
    let mut st = engine::settings(&project, &args.sandbox)?;
    let cache = benchflow_run::config::default_cache_dir();
    let mut roots: Vec<PathBuf> = Vec::new();
    for t in &args.targets {
        let s = source::resolve(t, &cache)
            .await
            .map_err(|e| CliError::usage(format!("{t}: {e}")))?;
        roots.push(s.path.clone());
    }
    let (tasks, bad) = context::load_tasks(&roots)?;
    if !bad.is_empty() || tasks.is_empty() {
        return Err(CliError::usage(format!(
            "{} could not be loaded",
            plural(bad.len().max(1), "task")
        ))
        .next("run `bench check` on it to see its errors"));
    }
    let agent_name = args.agent.clone().unwrap_or_else(|| "oracle".into());
    let env = context::parse_env(&args.env)?;
    // `-e` values reach the agent, over bench.toml's, as `bench run --env`
    // sends them (they were only used to find the login).
    st.defaults = with_flag_env(&st.defaults, &env);
    let attempt = engine::attempt(
        &agent_name,
        args.model.as_deref(),
        args.effort.as_deref(),
        None,
        None,
        &env,
        &[],
    )
    .await?;
    if let Attempt::Model(m) = &attempt {
        m.preflight().map_err(|p| {
            CliError::not_ready(p.message.clone())
                .code(p.code.clone())
                .owner(p.owner)
                .next(p.next.clone())
        })?;
    }
    let plan = ForkPlan {
        at,
        children,
        retain: args.retain_checkpoints,
        resume: args.resume_session,
        concurrency: st.concurrency,
        continue_parent,
    };
    let provider = context::provider(st.sandbox.as_deref()).await?;
    let options = engine::options(&st, provider.as_ref(), args.sandbox.keep)?;
    // Refusals, by name, before any sandbox starts.
    let mut refused: Vec<String> = Vec::new();
    // A parent that goes on after `start` or a stage runs its agent again
    // while the children run theirs; after `prompt:N` it is only graded.
    let parent_runs_on = plan.continue_parent && !matches!(plan.at, ForkPoint::Prompt(_));
    if let Attempt::Model(m) = &attempt
        && ((plan.concurrency > 1 && plan.children.len() > 1) || parent_runs_on)
        && let Some((_, keys)) = ROTATING.iter().find(|(n, _)| m.name().starts_with(n))
        && !keys
            .iter()
            .any(|k| env.contains_key(*k) || std::env::var(k).is_ok_and(|v| !v.trim().is_empty()))
    {
        refused.push(format!(
            "{} would run on a subscription login, whose tokens rotate: {} can invalidate each other's (use {}, or a key: {})",
            m.name(),
            if parent_runs_on { "a continuing parent and its children running at once" } else { "children running at once" },
            match (parent_runs_on, plan.concurrency > 1 && plan.children.len() > 1) {
                (true, true) => "--parent discard and -j 1",
                (true, false) => "--parent discard",
                _ => "-j 1",
            },
            keys.join(", ")
        ));
    }
    for t in &tasks {
        for r in branch::refusals(
            t,
            &attempt,
            &plan,
            &options,
            &provider.capabilities(),
            provider.name(),
        ) {
            refused.push(format!("{}: {r}", t.name));
        }
    }
    refused.dedup();
    if !refused.is_empty() {
        for r in &refused {
            say!("  {} {r}", warn_mark());
        }
        return Err(CliError::usage(format!(
            "the fork was refused ({})",
            plural(refused.len(), "reason")
        ))
        .next("each reason above names what to change"));
    }
    if let Ok(n) = provider.sweep().await
        && n > 0
    {
        note!(
            "  {} removed {} left behind by earlier runs",
            dim("cleanup:"),
            plural(n, "sandbox")
        );
    }
    note!(
        "{} {} at {} into {} · {} · {} at a time",
        bold("Branching"),
        attempt.label(),
        plan.at.as_str(),
        match plan.children.len() {
            1 => "1 child".to_string(),
            n => format!("{n} children"),
        },
        provider.name(),
        plan.concurrency
    );
    let width = tasks
        .iter()
        .map(|t| t.name.chars().count() + 8)
        .max()
        .unwrap_or(10)
        .clamp(10, 60);
    let jobs_dir = st.jobs_dir.clone();
    let job = args.job.clone();
    let outcome =
        engine::with_live_view(provider, options, width, ui::live::View::default(), |rt| {
            branch::run_job(tasks, attempt, plan, jobs_dir, job, rt)
        })
        .await
        .map_err(engine::job_folder_error)?;
    // Every fork completed, and every parent that went on was graded.
    let all_complete = outcome.forks.iter().all(|f| {
        f.status == "completed"
            && (!f.parent_continued || f.parent.status == benchflow_run::outcome::Status::Scored)
    });
    if ui().json {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "branch-finished",
                "dir": outcome.dir,
                "forks": outcome.dir.read_dir().ok().map(|_| views(&outcome.dir)),
            }))
            .unwrap_or_default()
        );
    } else {
        print_views(&views(&outcome.dir));
        say!("  {} {}", dim("job"), ui::path(&outcome.dir));
    }
    Ok(if all_complete { Exit::Ok } else { Exit::Failed })
}

/// One document per forked trial: its tree's forks, flattened.
fn views(job: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    for trial in benchflow_run::read::trial_dirs(job) {
        let Ok(text) = std::fs::read_to_string(trial.join("tree.json")) else {
            continue;
        };
        let Ok(tree) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if tree["kind"] != "benchflow-branch-tree" {
            continue;
        }
        for f in tree["forks"].as_array().into_iter().flatten() {
            let value = f["value"].as_f64();
            out.push(json!({
                "trial": trial.file_name().map(|n| n.to_string_lossy().to_string()),
                "fork": f["id"],
                "fork_point": f["fork_point"],
                "status": f["status"],
                "value": f["value"],
                "checkpoint": {
                    "provider": f["snapshot"]["sandbox"]["provider"],
                    "retention": f["snapshot"]["retention"],
                    "seconds": f["timing_sec"]["checkpoint"],
                },
                "children": f["children"].as_array().into_iter().flatten().map(|c| {
                    let reward = c["reward"].as_f64();
                    json!({
                        "node_id": c["node_id"],
                        "label": c["intervention"]["label"],
                        "status": c["status"],
                        "reward": c["reward"],
                        "advantage": match (reward, value) {
                            (Some(r), Some(v)) => json!(((r - v) * 1e6).round() / 1e6),
                            _ => Value::Null,
                        },
                        "path": c["artifacts"]["path"],
                        "error": child_error(&trial, c),
                    })
                }).collect::<Vec<_>>(),
            }));
        }
    }
    out
}

/// A child's error, as text: the message in its own result.json (tree.json
/// keeps only the code, as 0.8 does), else the code.
fn child_error(trial: &Path, child: &Value) -> Value {
    let from_result = child["artifacts"]["path"]
        .as_str()
        .filter(|p| !p.contains(".."))
        .and_then(|p| std::fs::read_to_string(trial.join(p).join("result.json")).ok())
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|r| r["problem"]["message"].as_str().map(str::to_string));
    match from_result {
        Some(m) => json!(m),
        None => child["error"]["code"].clone(),
    }
}

fn print_views(views: &[Value]) {
    if views.is_empty() {
        say!("  no forks recorded");
        return;
    }
    for v in views {
        say!(
            "  {} {}  fork {}  {}  value {}  checkpoint {} ({}s, {})",
            bold(v["trial"].as_str().unwrap_or("?")),
            dim(v["fork_point"].as_str().unwrap_or("")),
            &v["fork"].as_str().unwrap_or("")[..8.min(v["fork"].as_str().unwrap_or("").len())],
            v["status"].as_str().unwrap_or(""),
            v["value"]
                .as_f64()
                .map_or("—".into(), |x| format!("{x:.3}")),
            v["checkpoint"]["provider"].as_str().unwrap_or("none"),
            v["checkpoint"]["seconds"]
                .as_f64()
                .map_or("?".into(), |s| format!("{s:.1}")),
            v["checkpoint"]["retention"].as_str().unwrap_or("—"),
        );
        for c in v["children"].as_array().into_iter().flatten() {
            say!(
                "    {:<4} {:<16} {:<10} {:>6} {:>7}  {}",
                c["node_id"].as_str().unwrap_or(""),
                c["label"].as_str().unwrap_or(""),
                c["status"].as_str().unwrap_or(""),
                c["reward"]
                    .as_f64()
                    .map_or("—".into(), |r| format!("{r:.3}")),
                c["advantage"]
                    .as_f64()
                    .map_or("—".into(), |a| format!("{a:+.3}")),
                dim(c["error"].as_str().unwrap_or("")),
            );
        }
    }
}

/// `bench branches <job>`.
pub fn show(args: BranchesArgs) -> Result<Exit> {
    if !args.job.is_dir() {
        return Err(CliError::usage(format!(
            "{} is not a folder",
            ui::path(&args.job)
        )));
    }
    let v = views(&args.job);
    if ui().json {
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
    } else {
        print_views(&v);
    }
    Ok(if v.is_empty() { Exit::Failed } else { Exit::Ok })
}

/// `defaults` with the `-e` values laid over its `env`.
fn with_flag_env(
    defaults: &benchflow_run::project::RunDefaults,
    env: &std::collections::BTreeMap<String, String>,
) -> benchflow_run::project::RunDefaults {
    defaults.overlay(&benchflow_run::project::RunDefaults {
        env: env.clone(),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// `-e` values go to the agent's environment over bench.toml's, which
    /// `engine::options` turns into `RunOptions::agent_env`.
    #[test]
    fn flag_env_reaches_the_agent_over_bench_toml() {
        let toml = benchflow_run::project::RunDefaults {
            env: BTreeMap::from([
                ("A".to_string(), "toml".to_string()),
                ("C".to_string(), "kept".to_string()),
            ]),
            ..Default::default()
        };
        let flags = BTreeMap::from([
            ("A".to_string(), "flag".to_string()),
            ("B".to_string(), "x".to_string()),
        ]);
        let env = with_flag_env(&toml, &flags).env;
        assert_eq!(env["A"], "flag");
        assert_eq!(env["B"], "x");
        assert_eq!(env["C"], "kept");
    }
}
