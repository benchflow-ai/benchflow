//! `bench results`: what jobs found. One job: its tasks, its failures by
//! cause, pass@k. Many: a table of jobs. Two or more named: a paired
//! comparison over the tasks they share.

use std::path::PathBuf;

use benchflow_run::job::task_stats;
use benchflow_run::metrics::{Rate, paired, pass_at_k_table};
use benchflow_run::outcome::Status;
use benchflow_run::read::{self, JobView};
use benchflow_run::record::{TrialRecord, fmt_reward};
use serde_json::json;

use super::{engine, review};
use crate::cli::ResultsArgs;
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::say;
use crate::ui::{
    self, bold, dim, fail_mark, ok_mark, plural,
    table::{Align, Table},
    ui, warn_mark,
};

pub fn run(args: ResultsArgs) -> Result<Exit> {
    let project = context::project()?;
    let roots: Vec<PathBuf> = if args.jobs.is_empty() {
        let d = project
            .config
            .run
            .jobs_dir
            .clone()
            .map(|d| project.root.join(d))
            .unwrap_or_else(|| PathBuf::from("jobs"));
        vec![d]
    } else {
        args.jobs.clone()
    };
    let mut jobs: Vec<PathBuf> = Vec::new();
    for r in &roots {
        if !r.exists() {
            return Err(CliError::usage(format!("{} does not exist", r.display()))
                .next("pass a job folder (jobs/<job>) or a folder of jobs"));
        }
        jobs.extend(read::find_jobs(r));
    }
    if jobs.is_empty() {
        return Err(CliError::usage("no jobs found").next("`bench run` writes jobs into ./jobs"));
    }
    if args.analyze {
        return analyze(&args, &jobs);
    }
    let views: Vec<JobView> = jobs.iter().map(|j| read::read_job(j)).collect();
    // Two or more named jobs: compare them.
    if args.jobs.len() >= 2 && views.len() >= 2 {
        return compare(&views);
    }
    if views.len() == 1 {
        return one(&views[0], args.trials);
    }
    many(&views)
}

/// `--analyze`: the job's summary as BenchFlow recorded it, then a model's
/// account of each failed trial. With no job named, the latest one.
fn analyze(args: &ResultsArgs, jobs: &[PathBuf]) -> Result<Exit> {
    let targets: Vec<PathBuf> = if args.jobs.is_empty() {
        jobs.iter()
            .max_by_key(|j| std::fs::metadata(j).and_then(|m| m.modified()).ok())
            .cloned()
            .into_iter()
            .collect()
    } else {
        jobs.to_vec()
    };
    if !ui().json && targets.len() == 1 {
        one(&read::read_job(&targets[0]), args.trials)?;
    }
    let save = !args.no_save;
    let analyses = review::analyze_jobs(&targets, args.model.as_deref(), args.max_trials, save)?;
    if ui().json {
        let v = if analyses.len() == 1 {
            json!(analyses[0])
        } else {
            json!(analyses)
        };
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
    } else {
        for a in &analyses {
            review::print_analysis(a, save);
        }
        review::print_overall(&analyses);
    }
    let incomplete = analyses
        .iter()
        .any(|a| a.trials.iter().any(|t| t.error.is_some()));
    Ok(if incomplete { Exit::Failed } else { Exit::Ok })
}

fn many(views: &[JobView]) -> Result<Exit> {
    if ui().json {
        let v: Vec<_> = views.iter().map(job_json).collect();
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return Ok(Exit::Ok);
    }
    let mut t = Table::new(&["job", "agent", "model", "trials", "passed", "mean", "when"])
        .align(3, Align::Right)
        .align(5, Align::Right);
    // When the table is wider than the terminal, each job gets two lines.
    let mut blocks = Vec::new();
    for v in views {
        let total = v.records.len();
        let passed = v.records.iter().filter(|r| r.passed == Some(true)).count();
        let rate = Rate::new(passed, total);
        let rewards: Vec<f64> = v.records.iter().filter_map(|r| r.reward).collect();
        let agent = v.agent();
        // A `bench test` job mixes the oracle, an empty run, controls, and
        // probes: a pass rate over all of them means nothing.
        let test_plan = is_test_plan(&agent);
        let model = v.model().unwrap_or_else(|| "–".into());
        let mean = if rewards.is_empty() || test_plan {
            "–".into()
        } else {
            fmt_reward(rewards.iter().sum::<f64>() / rewards.len() as f64)
        };
        let when = v.modified().map(crate::ui::ago).unwrap_or_default();
        let score = if test_plan {
            dim("test plan")
        } else {
            format!("{passed} · {}", rate.display())
        };
        let detail = if test_plan {
            format!("{} · {}", plural(total, "trial"), dim("test plan"))
        } else {
            format!(
                "{} · {passed} passed · {} · mean reward {mean}",
                plural(total, "trial"),
                rate.display()
            )
        };
        blocks.push(format!(
            "{}  {} · {model}  {}\n  {detail}",
            ui::path(&v.dir),
            agent_label(&agent),
            dim(&when)
        ));
        t.row(vec![
            ui::path(&v.dir),
            agent_label(&agent),
            model,
            total.to_string(),
            score,
            mean,
            when,
        ]);
    }
    let table = t.render(0);
    if table.lines().any(|l| ui::visible_width(l) > ui::width()) {
        say!("{}\n", blocks.join("\n"));
    } else {
        say!("{table}");
    }
    say!(
        "{}",
        dim("`bench results <job>` for one job; `bench results <a> <b>` to compare.")
    );
    Ok(Exit::Ok)
}

/// Whether a job's agents are a `bench test` plan (the oracle beside
/// probes or controls).
fn is_test_plan(agents: &str) -> bool {
    let names: Vec<&str> = agents.split(',').map(str::trim).collect();
    names.contains(&"oracle")
        && names
            .iter()
            .any(|a| a.starts_with("probe:") || a.starts_with("control:"))
}

/// A job's agents for one table cell: a test plan as `oracle, nop, 8
/// probes`, more than three agents as the first two and a count.
fn agent_label(agents: &str) -> String {
    let names: Vec<&str> = agents
        .split(',')
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .collect();
    let probes = names.iter().filter(|a| a.starts_with("probe:")).count();
    let controls = names.iter().filter(|a| a.starts_with("control:")).count();
    if probes + controls > 0 {
        let mut parts: Vec<String> = names
            .iter()
            .filter(|a| !a.starts_with("probe:") && !a.starts_with("control:"))
            .map(|a| a.to_string())
            .collect();
        if controls > 0 {
            parts.push(plural(controls, "control"));
        }
        if probes > 0 {
            parts.push(plural(probes, "probe"));
        }
        return parts.join(", ");
    }
    if names.len() > 3 {
        return format!("{}, {} +{} more", names[0], names[1], names.len() - 2);
    }
    names.join(", ")
}

fn one(v: &JobView, list_trials: bool) -> Result<Exit> {
    let stats = task_stats(&v.records);
    let total = v.records.len();
    let passed = v.records.iter().filter(|r| r.passed == Some(true)).count();
    let rate = Rate::new(passed, total);
    if ui().json {
        let mut j = job_json(v);
        j["tasks"] = json!(stats);
        j["pass_at_k"] = json!(pass_at_k_table(&stats));
        j["trials"] = json!(
            v.records
                .iter()
                .map(|r| r.result_json())
                .collect::<Vec<_>>()
        );
        println!("{}", serde_json::to_string_pretty(&j).unwrap_or_default());
        return Ok(Exit::Ok);
    }
    say!(
        "{}  {}{}",
        bold(ui::path(&v.dir)),
        v.agent(),
        v.model().map(|m| format!(" · {m}")).unwrap_or_default()
    );
    say!(
        "  {} {}/{} passed · {}",
        bold("Score"),
        passed,
        total,
        rate.display()
    );
    let all: Vec<&TrialRecord> = v.records.iter().collect();
    if let Some(u) = engine::usage(&all) {
        say!("  {}  {u}", dim("Usage"));
    }
    let pak = pass_at_k_table(&stats);
    if pak.len() > 1 {
        let s: Vec<String> = pak
            .iter()
            .map(|(k, v)| format!("pass@{k} {:.1}%", v * 100.0))
            .collect();
        say!("  {}", dim(s.join(" · ")));
    }
    if list_trials {
        let mut t = Table::new(&["trial", "status", "reward", "time", "cost", "reason"])
            .align(2, Align::Right)
            .align(3, Align::Right)
            .align(4, Align::Right);
        for r in &v.records {
            t.row(vec![
                r.rollout_name.clone(),
                status_cell(r.status, r.passed),
                r.reward.map(fmt_reward).unwrap_or_else(|| "–".into()),
                ui::secs(r.timing.total),
                cost_cell(&[r]),
                r.reason().map(|s| ui::truncate(&s, 60)).unwrap_or_default(),
            ]);
        }
        say!("\n{}", t.render(2));
        // Each rubric criterion's verdict, model-judged ones marked.
        let width = crate::ui::width();
        for r in &v.records {
            if let Some(line) = r.rubric.as_ref().and_then(rubric_verdicts) {
                say!(
                    "{}",
                    crate::ui::wrap(&format!("  {} ", dim(&r.rollout_name)), &line, 4, width)
                );
            }
        }
    } else {
        let mut t = Table::new(&["task", "passed", "mean reward", "cost"])
            .align(1, Align::Right)
            .align(2, Align::Right)
            .align(3, Align::Right);
        for (task, s) in &stats {
            let mark = if s.passed == s.trials {
                ok_mark()
            } else if s.passed > 0 {
                warn_mark()
            } else {
                fail_mark()
            };
            t.row(vec![
                format!("{mark} {task}"),
                format!("{}/{}", s.passed, s.trials),
                s.mean_reward()
                    .map(fmt_reward)
                    .unwrap_or_else(|| "–".into()),
                cost_cell(
                    &v.records
                        .iter()
                        .filter(|r| r.task_name == *task)
                        .collect::<Vec<_>>(),
                ),
            ]);
        }
        say!("\n{}", t.render(2));
    }
    // Failures by cause.
    let mut causes: std::collections::BTreeMap<String, Vec<&benchflow_run::record::TrialRecord>> =
        Default::default();
    for r in &v.records {
        if let Some(p) = &r.problem {
            causes.entry(p.code.clone()).or_default().push(r);
        }
    }
    for (code, rs) in &causes {
        let p = rs[0].problem.as_ref().expect("grouped");
        say!(
            "  {} {} [{code}] {}",
            warn_mark(),
            plural(rs.len(), "trial"),
            dim(p.owner.problem())
        );
        let width = crate::ui::width();
        say!("{}", crate::ui::wrap("      ", &p.message, 6, width));
        say!(
            "{}",
            crate::ui::wrap(&format!("      {} ", dim("next:")), &p.next, 12, width)
        );
    }
    Ok(Exit::Ok)
}

fn compare(views: &[JobView]) -> Result<Exit> {
    let a = &views[0];
    let stats: Vec<_> = views.iter().map(|v| task_stats(&v.records)).collect();
    let mut tasks: Vec<String> = stats.iter().flat_map(|s| s.keys().cloned()).collect();
    tasks.sort();
    tasks.dedup();
    let pairs: Vec<_> = views[1..]
        .iter()
        .zip(stats[1..].iter())
        .map(|(v, s)| (v, paired(&stats[0], s)))
        .collect();
    if ui().json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "jobs": views.iter().map(job_json).collect::<Vec<_>>(),
                "tasks": tasks.iter().map(|t| json!({
                    "task": t,
                    "mean_reward": stats.iter().map(|s| s.get(t).and_then(|x| x.mean_reward())).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "paired": pairs.iter().map(|(v, p)| json!({"against": v.dir, "result": p})).collect::<Vec<_>>(),
            }))
            .unwrap_or_default()
        );
        return Ok(Exit::Ok);
    }
    let mut header: Vec<String> = vec!["task".into()];
    for (i, v) in views.iter().enumerate() {
        header.push(format!("{} {}", (b'A' + i as u8) as char, v.agent()));
    }
    let header_refs: Vec<&str> = header.iter().map(String::as_str).collect();
    let mut t = Table::new(&header_refs);
    for i in 1..header.len() {
        t = t.align(i, Align::Right);
    }
    for task in &tasks {
        let mut row = vec![task.clone()];
        for s in &stats {
            row.push(
                s.get(task)
                    .and_then(|x| x.mean_reward())
                    .map(fmt_reward)
                    .unwrap_or_else(|| "–".into()),
            );
        }
        t.row(row);
    }
    for (i, v) in views.iter().enumerate() {
        say!(
            "  {} {}",
            bold((b'A' + i as u8) as char),
            dim(ui::path(&v.dir))
        );
    }
    say!("\n{}", t.render(2));
    for (i, (v, p)) in pairs.iter().enumerate() {
        let b = (b'B' + i as u8) as char;
        match p {
            Some(p) => {
                let parts = [
                    format!(
                        "  A − {b}: {:+.3} mean reward over {} (95% CI {:+.3} to {:+.3})",
                        p.diff,
                        plural(p.tasks, "shared task"),
                        p.low,
                        p.high
                    ),
                    format!(
                        "A better on {}, {b} better on {}, tied on {}",
                        p.a_better, p.b_better, p.ties
                    ),
                ];
                say!("{}", ui::join_fitting(&parts, 4, ui::width()))
            }
            None => say!(
                "  A − {b}: no task was scored in both {} and {}",
                ui::path(&a.dir),
                ui::path(&v.dir)
            ),
        }
    }
    Ok(Exit::Ok)
}

/// A table cell for some trials' model cost: the sum (with how many
/// recorded one, when some did not), `not recorded` when none did, `–` for trials with no model (the oracle).
fn cost_cell(recs: &[&TrialRecord]) -> String {
    let costs: Vec<Option<f64>> = recs.iter().map(|r| r.usage.cost_usd).collect();
    let tokens: u64 = recs
        .iter()
        .map(|r| r.usage.input_tokens + r.usage.output_tokens)
        .sum();
    match costs.iter().flatten().copied().reduce(|a, b| a + b) {
        Some(sum) if costs.iter().all(Option::is_some) => ui::usd(sum),
        Some(sum) => {
            let n = costs.iter().flatten().count();
            format!("{} ({n} of {})", ui::usd(sum), costs.len())
        }
        None if tokens > 0 => "not recorded".into(),
        None => "–".into(),
    }
}

/// A rubric review's verdicts on one line: `written pass · cause fail
/// (llm)`.
fn rubric_verdicts(review: &serde_json::Value) -> Option<String> {
    let verdicts = review.get("verdicts")?.as_array()?;
    let parts: Vec<String> = verdicts
        .iter()
        .map(|v| {
            let id = v.get("id").and_then(|x| x.as_str()).unwrap_or("?");
            let verdict = match v.get("verdict").and_then(|x| x.as_str()) {
                Some("level") => format!("level {}", v["level"].as_str().unwrap_or("?")),
                Some("value") => format!("{}", v["value"]),
                Some(w) => w.to_string(),
                None => "?".into(),
            };
            let role = v["judge"]["role"].as_str().unwrap_or("");
            if role == "test" || role.is_empty() {
                format!("{id} {verdict}")
            } else {
                format!("{id} {verdict} ({role})")
            }
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join(" · "))
}

fn status_cell(s: Status, passed: Option<bool>) -> String {
    match (s, passed) {
        (Status::Scored, Some(true)) => format!("{} passed", ok_mark()),
        (Status::Scored, _) => format!("{} failed", fail_mark()),
        (Status::Unscored, _) => format!("{} unscored", warn_mark()),
        (Status::Errored, _) => format!("{} errored", warn_mark()),
        (Status::Skipped, _) => dim("skipped"),
    }
}

fn job_json(v: &JobView) -> serde_json::Value {
    let total = v.records.len();
    let passed = v.records.iter().filter(|r| r.passed == Some(true)).count();
    let rate = Rate::new(passed, total);
    json!({
        "dir": v.dir,
        "agent": v.agent(),
        "model": v.model(),
        "trials": total,
        "passed": passed,
        "pass_rate": rate,
        "scored": v.records.iter().filter(|r| r.status == Status::Scored).count(),
        // What the trials that recorded a cost cost, and how many did:
        // null, not 0, when none did.
        "cost_usd": v.records.iter().filter_map(|r| r.usage.cost_usd).reduce(|a, b| a + b),
        "cost_recorded": v.records.iter().filter(|r| r.usage.cost_usd.is_some()).count(),
    })
}
