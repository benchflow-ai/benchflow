//! `bench test`: prove each task. The oracle passes, doing nothing fails,
//! the task's controls fail, and exploit probes do not raise the reward.

use std::collections::BTreeMap;
use std::sync::Arc;

use benchflow_run::credentials::{self, Finding, Location};
use benchflow_run::job::JobSpec;
use benchflow_run::outcome::DefenseState;
use benchflow_run::record::{TrialRecord, fmt_reward};
use benchflow_run::testplan::{self, ProbeSet, TaskVerdict};
use serde_json::json;

use crate::cli::TestArgs;
use crate::cmd::engine;
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::ui::{self, bold, dim, fail_mark, ok_mark, plural, table::Table, ui, warn_mark};
use crate::{note, say};

pub async fn run(args: TestArgs) -> Result<Exit> {
    if args.list_probes {
        return list_probes();
    }
    let project = context::project()?;
    let st = engine::settings(&project, &args.sandbox)?;
    let roots = context::task_roots(&project, &args.paths);
    let (tasks, bad) = context::load_tasks(&roots)?;
    for r in &bad {
        note!(
            "  {} {} has errors; run `bench check {}`",
            fail_mark(),
            ui::path(&r.root),
            ui::path(&r.root)
        );
    }
    if tasks.is_empty() {
        return Err(CliError::failed(
            "no task passed `bench check`, so none can be tested",
        ));
    }
    let probes = match args.probes.as_str() {
        "all" => ProbeSet::All,
        "none" => ProbeSet::None,
        list => {
            let ids: Vec<String> = list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            for id in &ids {
                if benchflow_run::probes::find(id).is_none() {
                    return Err(CliError::usage(format!("no probe `{id}`"))
                        .next("`bench test --list-probes` lists them"));
                }
            }
            ProbeSet::Only(ids)
        }
    };
    let units: Vec<_> = tasks
        .iter()
        .map(|t| (t.clone(), testplan::attempts(t, &probes)))
        .collect();
    let n_trials: usize = units.iter().map(|(_, a)| a.len()).sum();
    // What each task set out to run: judged against what ran.
    let planned_units = units.clone();
    let provider = context::provider(st.sandbox.as_deref()).await?;
    let _ = provider.sweep().await;
    let mut options = engine::options(&st, provider.as_ref(), args.sandbox.keep)?;
    options.scan_credentials = true;
    let what = format!(
        "{} {} with {}",
        bold("Testing"),
        plural(tasks.len(), "task"),
        plural(n_trials, "trial"),
    );
    let place = format!("{} · {} at a time", provider.name(), st.concurrency);
    note!(
        "{}",
        ui::fit_or_break(&what, " on ", &place, 2, ui::width())
    );
    let spec = JobSpec {
        carry: Vec::new(),
        units,
        trials: 1,
        jobs_dir: st.jobs_dir.join("tests"),
        job_name: None,
        fresh: true,
        concurrency: st.concurrency,
        source: json!({"targets": roots, "path": roots.first(), "kind": "bench test"}),
    };
    let outcome = engine::run_job(
        spec,
        Arc::clone(&provider),
        options,
        ui::live::View {
            show_agent: true,
            test_plan: true,
            ..Default::default()
        },
    )
    .await?;

    // Verdicts per task.
    let mut by_task: BTreeMap<String, Vec<&TrialRecord>> = BTreeMap::new();
    for r in &outcome.records {
        by_task.entry(r.task_name.clone()).or_default().push(r);
    }
    let verdicts: Vec<TaskVerdict> = planned_units
        .iter()
        .map(|(t, planned)| {
            testplan::judge(
                t,
                by_task.get(&t.name).map(Vec::as_slice).unwrap_or(&[]),
                planned,
            )
        })
        .collect();
    let proven = verdicts.iter().filter(|v| v.ok).count();
    // Credential files in each task's build folder and image.
    let credentials: Vec<Vec<Finding>> = tasks
        .iter()
        .map(|t| {
            let mut found = credentials::scan_task_build(t);
            if let Some(nop) = by_task
                .get(&t.name)
                .and_then(|rs| rs.iter().find(|r| r.agent == "nop"))
            {
                found.extend(nop.custody.credentials.iter().cloned());
            }
            found
        })
        .collect();
    if ui().json {
        let tasks_json: Vec<serde_json::Value> = verdicts
            .iter()
            .zip(&credentials)
            .map(|(v, c)| {
                let mut v = serde_json::to_value(v).unwrap_or_default();
                v["credentials"] = json!(c
                    .iter()
                    .map(|f| json!({"path": f.path, "kind": f.kind, "location": f.location, "agent_can_read": f.agent_can_read, "fix": f.fix()}))
                    .collect::<Vec<_>>());
                v
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "job": outcome.dir,
                "proven": proven,
                "not_started": outcome.not_started,
                "tasks": tasks_json,
            }))
            .unwrap_or_default()
        );
    } else {
        say!("");
        for (v, c) in verdicts.iter().zip(&credentials) {
            print_verdict(v);
            print_credentials(c);
        }
        let line = if proven == verdicts.len() {
            format!(
                "{} {} proved {}",
                ok_mark(),
                plural(verdicts.len(), "task"),
                if verdicts.len() == 1 {
                    "itself"
                } else {
                    "themselves"
                }
            )
        } else {
            format!(
                "{} {} of {} proved {}",
                fail_mark(),
                proven,
                plural(verdicts.len(), "task"),
                if proven == 1 { "itself" } else { "themselves" }
            )
        };
        if outcome.not_started > 0 {
            say!(
                "  {} {} not started (the run was drained or stopped); run `bench test` again to test fully",
                warn_mark(),
                plural(outcome.not_started, "trial")
            );
        }
        let view = dim(format!("bench view {}", ui::path(&outcome.dir))).to_string();
        say!("{}", ui::fit_or_break(&line, "  ", &view, 2, ui::width()));
    }
    Ok(
        if proven == verdicts.len() && bad.is_empty() && outcome.not_started == 0 {
            Exit::Ok
        } else {
            Exit::Failed
        },
    )
}

fn print_verdict(v: &TaskVerdict) {
    say!(
        "  {} {}",
        if v.ok { ok_mark() } else { fail_mark() },
        bold(&v.task)
    );
    let probes: Vec<_> = v
        .checks
        .iter()
        .filter(|c| c.attempt.starts_with("probe:"))
        .collect();
    let probes_ok = probes.iter().all(|c| c.ok);
    let width = v
        .checks
        .iter()
        .filter(|c| !c.attempt.starts_with("probe:") || !probes_ok)
        .map(|c| c.attempt.len())
        .max()
        .unwrap_or(6)
        .max(6);
    for c in &v.checks {
        if c.attempt.starts_with("probe:") && probes_ok {
            continue;
        }
        let mark = if c.ok { ok_mark() } else { fail_mark() };
        let reward = c.reward.map(fmt_reward).unwrap_or_else(|| "–".into());
        say!(
            "{}",
            check_line(&mark, &c.attempt, width, &reward, &c.says, ui::width())
        );
        if let Some(fix) = &c.fix
            && !c.ok
        {
            say!(
                "{}",
                ui::wrap(&format!("        {} ", dim("fix:")), fix, 13, ui::width())
            );
        }
    }
    if !probes.is_empty() && probes_ok {
        say!(
            "      {} {:<width$}  {:>5}  {} refused",
            ok_mark(),
            "probes",
            format!("{}/{}", probes.len(), probes.len()),
            plural(probes.len(), "exploit"),
        );
    }
    if !v.weak_details.is_empty() {
        say!(
            "      {} {}",
            warn_mark(),
            dim("defenses not in force in every trial:")
        );
        let name_width = v
            .weak_details
            .iter()
            .map(|d| d.name.len())
            .max()
            .unwrap_or(0);
        for d in &v.weak_details {
            let state = match d.state {
                DefenseState::Enforced => "enforced",
                DefenseState::NotSupported => "not supported here",
                DefenseState::Off => "off",
                DefenseState::Failed => "failed",
            };
            let text = match &d.note {
                Some(note) => format!("{state}: {note}"),
                None => state.to_string(),
            };
            let first = format!("          {:<name_width$}  ", d.name);
            let indent = ui::visible_width(&first);
            say!("{}", dim(ui::wrap(&first, &text, indent, ui::width())));
        }
    }
}

/// One check of a task's report: mark, attempt, reward, and what it says,
/// wrapped under its own column at `term_width`.
fn check_line(
    mark: &str,
    attempt: &str,
    width: usize,
    reward: &str,
    says: &str,
    term_width: usize,
) -> String {
    let first = format!("      {mark} {attempt:<width$}  {reward:>5}  ");
    let indent = ui::visible_width(&first);
    ui::wrap(&first, says, indent, term_width)
}

/// Credential files a task leaves where its agent could find them: path,
/// kind, and the fix, never the content.
pub fn print_credentials(found: &[Finding]) {
    if found.is_empty() {
        return;
    }
    say!(
        "      {} {}",
        warn_mark(),
        format!(
            "{} where the agent could find {}:",
            plural(found.len(), "credential file"),
            if found.len() == 1 { "it" } else { "them" }
        )
    );
    for location in [Location::Image, Location::BuildFolder] {
        let here: Vec<&Finding> = found.iter().filter(|f| f.location == location).collect();
        let Some(first) = here.first() else {
            continue;
        };
        for f in &here {
            let read = match f.agent_can_read {
                Some(true) => " · the agent user can read it",
                Some(false) => " · root-only",
                None => "",
            };
            say!(
                "          {:<12}  {}  {}",
                match location {
                    Location::Image => "image",
                    Location::BuildFolder => "build folder",
                },
                f.path,
                dim(format!("{}{read}", f.kind))
            );
        }
        say!("        {} {}", dim("fix:"), first.fix());
    }
}

fn list_probes() -> Result<Exit> {
    if ui().json {
        let v: Vec<_> = benchflow_run::probes::ALL
            .iter()
            .map(|p| json!({"id": p.id, "tries": p.tries, "defense": p.defense, "fix": p.fix}))
            .collect();
        println!("{}", serde_json::Value::Array(v));
        return Ok(Exit::Ok);
    }
    let mut t = Table::new(&["probe", "tries", "defense"]);
    for p in benchflow_run::probes::ALL {
        t.row(vec![p.id.to_string(), p.tries.to_string(), dim(p.defense)]);
    }
    say!("{}", t.render(2));
    say!(
        "{}",
        dim(
            "A probe passes when it scores no higher than the empty run and grading still finishes."
        )
    );
    Ok(Exit::Ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long reason wraps under its own column instead of past the terminal.
    #[test]
    fn a_check_line_wraps_under_its_reason() {
        let line = check_line(
            "✗",
            "nop",
            6,
            "1",
            "doing nothing passes, so the verifier cannot tell a solution from the starting state",
            100,
        );
        let lines: Vec<&str> = line.lines().collect();
        assert_eq!(lines.len(), 2, "{line}");
        assert!(lines[0].starts_with("      ✗ nop         1  doing nothing passes"));
        assert!(lines[1].starts_with(&" ".repeat(22)), "{line}");
        assert!(lines.iter().all(|l| ui::visible_width(l) <= 100), "{line}");
    }
}
