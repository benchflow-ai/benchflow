//! `bench regrade`: score a finished job's trials again with each task's
//! current verifier, from the outputs they saved.

use benchflow_run::outcome::Status;
use benchflow_run::record::fmt_reward;
use benchflow_run::regrade;
use serde_json::json;

use crate::cli::{KeepArg, RegradeArgs, SandboxArgs};
use crate::cmd::engine;
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::ui::{self, bold, dim, plural, ui, warn_mark};
use crate::{note, say};

pub async fn run(args: RegradeArgs) -> Result<Exit> {
    let job = args.job.clone();
    if !job.is_dir() || !benchflow_run::read::is_job(&job) {
        return Err(
            CliError::usage(format!("{} is not a job folder", ui::path(&job))).next(
                "pass a folder under jobs/ that `bench run` wrote; `bench results` lists them",
            ),
        );
    }
    let plan = regrade::plan(&job).map_err(|e| CliError::bug("reading the job", e))?;
    if plan.regradable.is_empty() {
        print_not_regradable(&plan.not_regradable);
        return Err(
            CliError::failed(format!("nothing in {} can be regraded", ui::path(&job)))
                .owner(benchflow_run::Owner::You)
                .next("rerun the job with --save-outputs, then regrade it"),
        );
    }

    // Where the fresh sandboxes run: the flag, else where the job ran.
    let ran_on = benchflow_run::read::read_lock(&job)
        .ok()
        .map(|l| l.options.sandbox);
    let project = context::project()?;
    let st = engine::settings(
        &project,
        &SandboxArgs {
            sandbox: args.sandbox.clone().or(ran_on),
            concurrency: args.concurrency,
            keep: KeepArg::Never,
            jobs_dir: None,
            profile: args.profile.clone(),
            runtime: None,
        },
    )?;
    let provider = context::provider(st.sandbox.as_deref()).await?;
    let mut options = engine::options(&st, provider.as_ref(), KeepArg::Never)?;
    if let Some(m) = args.verifier_timeout_multiplier {
        if !(m.is_finite() && m > 0.0) {
            return Err(CliError::usage(format!(
                "--verifier-timeout-multiplier must be a number above 0, not {m}"
            )));
        }
        options.verifier_timeout_multiplier = m;
    }
    let what = format!(
        "{} {} from {}",
        bold("Regrading"),
        plural(plan.regradable.len(), "trial"),
        ui::path(&job),
    );
    let place = format!("{} · {} at a time", provider.name(), st.concurrency);
    note!(
        "{}",
        ui::fit_or_break(&what, " on ", &place, 2, ui::width())
    );
    let not_regradable = plan.not_regradable.clone();
    let width = plan
        .regradable
        .iter()
        .map(|r| r.task.name.chars().count())
        .max()
        .unwrap_or(10)
        .clamp(10, 44);
    let concurrency = st.concurrency;
    let outcome = engine::with_live_view(
        provider,
        options,
        width,
        crate::ui::live::View::default(),
        |rt| regrade::run(plan, rt, concurrency),
    )
    .await
    .map_err(engine::job_folder_error)?;

    if ui().json {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "regrade-finished",
                "dir": outcome.dir,
                "regrade_of": job,
                "changed": outcome.changed(),
                "trials": outcome.changes,
                "not_regradable": not_regradable,
                "not_started": outcome.not_started,
                "summary": outcome.summary,
            }))
            .unwrap_or_default()
        );
    } else {
        say!("");
        say!(
            "  {}  {}  {}",
            bold(format!("{:<w$}", "trial", w = width + 10)),
            bold(format!("{:>8}", "old")),
            bold(format!("{:>8}", "new"))
        );
        for c in &outcome.changes {
            let line = format!(
                "  {:<w$}  {:>8}  {:>8}",
                c.trial,
                reward_text(c.old_status, c.old_reward),
                reward_text(c.new_status, c.new_reward),
                w = width + 10
            );
            if c.changed {
                say!("{line}  {}", warn_mark());
            } else {
                say!("{}", dim(line));
            }
        }
        say!("");
        say!(
            "  {} {} of {}",
            bold("Changed"),
            outcome.changed(),
            plural(outcome.changes.len(), "regraded trial")
        );
        for r in outcome
            .records
            .iter()
            .filter(|r| r.status != Status::Scored)
        {
            if let Some(p) = &r.problem {
                say!(
                    "  {} {} {}: {}",
                    warn_mark(),
                    r.rollout_name,
                    dim(format!("[{}]", p.code)),
                    p.message
                );
            }
        }
        if outcome.not_started > 0 {
            say!(
                "  {} {} not regraded (the regrade was drained or stopped); run `bench regrade {}` again, which writes a new folder",
                warn_mark(),
                plural(outcome.not_started, "trial"),
                ui::path(&job)
            );
        }
        print_not_regradable(&not_regradable);
        say!("  {}  {}", dim("Job  "), ui::path(&outcome.dir));
        say!(
            "  {}  bench results {} {}",
            dim("Next "),
            ui::path(&job),
            ui::path(&outcome.dir)
        );
    }
    let all_scored = outcome.records.iter().all(|r| r.status == Status::Scored)
        && outcome.records.len() == outcome.changes.len();
    Ok(
        if all_scored && not_regradable.is_empty() && outcome.not_started == 0 {
            Exit::Ok
        } else {
            Exit::Failed
        },
    )
}

fn reward_text(status: Status, reward: Option<f64>) -> String {
    match (status, reward) {
        (Status::Scored, Some(r)) => fmt_reward(r),
        (Status::Unscored, _) => "unscored".into(),
        (Status::Skipped, _) => "skipped".into(),
        _ => "errored".into(),
    }
}

fn print_not_regradable(list: &[regrade::NotRegradable]) {
    if list.is_empty() || ui().json {
        return;
    }
    say!(
        "  {} {} cannot be regraded",
        warn_mark(),
        plural(list.len(), "trial")
    );
    let mut by_reason: Vec<(&str, &str, Vec<&str>)> = Vec::new();
    for n in list {
        match by_reason
            .iter_mut()
            .find(|(r, _, _)| *r == n.reason.as_str())
        {
            Some((_, _, trials)) => trials.push(&n.trial),
            None => by_reason.push((&n.reason, &n.next, vec![&n.trial])),
        }
    }
    for (reason, next, trials) in by_reason {
        let more = if trials.len() > 3 {
            format!(" and {} more", trials.len() - 3)
        } else {
            String::new()
        };
        let width = crate::ui::width();
        say!("{}", crate::ui::wrap("      ", reason, 6, width));
        say!(
            "      {} {}{}",
            dim("trials:"),
            trials
                .iter()
                .take(3)
                .copied()
                .collect::<Vec<_>>()
                .join(", "),
            more
        );
        say!(
            "{}",
            crate::ui::wrap(&format!("      {} ", dim("next:")), next, 12, width)
        );
    }
}
