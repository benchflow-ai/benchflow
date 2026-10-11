//! What `run`, `test`, and `shell` share: building the run's options and
//! runtime, resolving the agent, running a job with the live view, and the
//! report at the end.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use benchflow_run::agent::Attempt;
use benchflow_run::config::{Keep, RunOptions};
use benchflow_run::events::Events;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobOutcome, JobSpec};
use benchflow_run::metrics::Rate;
use benchflow_run::outcome::{Owner, Status};
use benchflow_run::project::{Project, RunDefaults};
use benchflow_run::record::{TrialRecord, fmt_reward};
use benchflow_run::trial::Runtime;
use benchflow_sandbox::Provider;

use crate::cli::{KeepArg, SandboxArgs};
use crate::context;
use crate::error::{CliError, Result};
use crate::say;
use crate::ui::{self, bold, dim, fail_mark, ok_mark, plural, ui, warn_mark};

/// The settings a run resolves from flags, the profile, and bench.toml.
pub struct Settings {
    pub defaults: RunDefaults,
    pub jobs_dir: PathBuf,
    pub concurrency: usize,
    pub sandbox: Option<String>,
}

pub fn settings(project: &Project, s: &SandboxArgs) -> Result<Settings> {
    settings_with(project, s, None)
}

/// [`settings`], with job files' settings laid over bench.toml's.
pub fn settings_with(
    project: &Project,
    s: &SandboxArgs,
    job: Option<&RunDefaults>,
) -> Result<Settings> {
    let mut defaults = project.run_defaults(s.profile.as_deref()).map_err(|e| {
        CliError::usage(e).next("profiles live under [profile.<name>] in bench.toml")
    })?;
    if let Some(j) = job {
        defaults = defaults.overlay(j);
    }
    if let Some(r) = &s.runtime {
        defaults.runtime = Some(r.clone());
    }
    let jobs_dir = s
        .jobs_dir
        .clone()
        .or_else(|| defaults.jobs_dir.clone().map(|d| project.root.join(d)))
        .unwrap_or_else(|| PathBuf::from("jobs"));
    Ok(Settings {
        concurrency: s.concurrency.or(defaults.concurrency).unwrap_or(4).max(1),
        sandbox: s.sandbox.clone().or(defaults.sandbox.clone()),
        jobs_dir,
        defaults,
    })
}

pub fn keep(k: KeepArg) -> Keep {
    match k {
        KeepArg::Never => Keep::Never,
        KeepArg::OnFailure => Keep::OnFailure,
        KeepArg::Always => Keep::Always,
    }
}

/// Run options from the resolved settings.
pub fn options(st: &Settings, provider: &dyn Provider, keep_arg: KeepArg) -> Result<RunOptions> {
    options_for(st, provider.name(), keep_arg)
}

/// [`options`] for a backend named `sandbox` ("" when not chosen yet, for
/// plans made before a backend starts).
pub fn options_for(st: &Settings, sandbox: &str, keep_arg: KeepArg) -> Result<RunOptions> {
    let mut o = RunOptions {
        sandbox: sandbox.to_string(),
        keep: keep(keep_arg),
        bench_version: Some(env!("BENCH_LONG_VERSION").to_string()),
        ..RunOptions::default()
    };
    if let Some(u) = &st.defaults.sandbox_user {
        o.sandbox_user = u.clone();
    }
    let d = &st.defaults;
    for (name, x) in [
        ("timeout-multiplier", d.timeout_multiplier),
        ("agent-timeout-multiplier", d.agent_timeout_multiplier),
        ("verifier-timeout-multiplier", d.verifier_timeout_multiplier),
        ("build-timeout-multiplier", d.build_timeout_multiplier),
    ] {
        if let Some(x) = x
            && !(x.is_finite() && x > 0.0)
        {
            return Err(CliError::usage(format!(
                "{name} must be a number above 0, not {x}"
            )));
        }
    }
    if let Some(m) = d.timeout_multiplier {
        o.timeout_multiplier = m;
        o.verifier_timeout_multiplier = m;
        o.build_timeout_multiplier = m;
    }
    if let Some(m) = d.agent_timeout_multiplier {
        o.timeout_multiplier = m;
    }
    if let Some(m) = d.verifier_timeout_multiplier {
        o.verifier_timeout_multiplier = m;
    }
    if let Some(m) = d.build_timeout_multiplier {
        o.build_timeout_multiplier = m;
    }
    if let Some(t) = &st.defaults.idle_timeout {
        o.idle_timeout = context::parse_duration(t)?;
    }
    if let Some(r) = st.defaults.retries {
        o.retries = r;
    }
    o.agent_env = st.defaults.env.clone();
    o.runtime = st.defaults.runtime.clone().filter(|r| !r.is_empty());
    Ok(o)
}

/// Refuse a `--runtime` the backend cannot start sandboxes under, before any
/// sandbox starts.
pub async fn check_runtime(provider: &dyn Provider, options: &RunOptions) -> Result<()> {
    benchflow_run::confinement::check_runtime(provider, options.runtime.as_deref())
        .await
        .map_err(|f| {
            CliError::not_ready(f.message)
                .owner(Owner::You)
                .next(f.next)
        })
}

/// Resolve `--agent` to an attempt. Scripted agents need nothing; model
/// agents come from the agent registry.
pub async fn attempt(
    name: &str,
    model: Option<&str>,
    effort: Option<&str>,
    harness: Option<&str>,
    backend: Option<&str>,
    env: &BTreeMap<String, String>,
    options: &[(String, String)],
) -> Result<Attempt> {
    if backend.is_some() && !benchflow_run::external_agent::is_external(name) {
        return Err(
            CliError::usage(format!("--backend is for the external agent, not `{name}`"))
                .next("drop --backend, or pass -a external"),
        );
    }
    if harness.is_some()
        && (name == "oracle"
            || name == "nop"
            || name.starts_with("probe:")
            || name.starts_with("script:"))
    {
        return Err(CliError::usage(format!(
            "--harness applies to model agents, and `{name}` is a script"
        ))
        .next("drop --harness, or pick a model agent with -a"));
    }
    if let Some((key, _)) = options.first()
        && (name == "oracle"
            || name == "nop"
            || name.starts_with("probe:")
            || name.starts_with("script:"))
    {
        return Err(CliError::usage(format!(
            "agent options apply to model agents, and `{name}` is a script"
        ))
        .next(format!("drop --ak {key}, or pick a model agent with -a")));
    }
    match name {
        "oracle" => Ok(Attempt::Oracle),
        "nop" => Ok(Attempt::Nop),
        n if n.starts_with("script:") => {
            let file = std::path::Path::new(&n["script:".len()..]);
            if file.as_os_str().is_empty() {
                return Err(CliError::usage(
                    "`script:` needs a file: --agent script:wrong.sh",
                ));
            }
            benchflow_run::agent::UserScript::read(file)
                .map(Attempt::Script)
                .map_err(|e| {
                    CliError::usage(format!("cannot read the script {}: {e}", file.display()))
                        .next("give the path of a shell script, relative to this folder: --agent script:path/to/wrong.sh")
                })
        }
        n if n.starts_with("probe:") => benchflow_run::probes::find(n)
            .map(Attempt::Probe)
            .ok_or_else(|| {
                CliError::usage(format!("no probe `{n}`"))
                    .next("`bench test --list-probes` lists them")
            }),
        other => {
            crate::cmd::agents::model_attempt(other, model, effort, harness, backend, env, options)
                .await
        }
    }
}

/// Run a job with the live view, and return its outcome. The first Ctrl-C
/// drains the run (no new trials; running ones finish), the second stops
/// it (running trials are cancelled and their sandboxes torn down), and a
/// third exits at once.
pub async fn run_job(
    spec: JobSpec,
    provider: Arc<dyn Provider>,
    options: RunOptions,
    mut view: ui::live::View,
) -> Result<JobOutcome> {
    check_runtime(provider.as_ref(), &options).await?;
    let show_agent = view.show_agent;
    view.numbered = spec.trials > 1;
    // " #12"
    let number_width = if view.numbered {
        2 + spec.trials.to_string().len()
    } else {
        0
    };
    let label_width = spec
        .units
        .iter()
        .flat_map(|(t, attempts)| {
            attempts.iter().map(move |a| {
                if show_agent {
                    t.name.chars().count() + 1 + a.label().chars().count()
                } else {
                    t.name.chars().count()
                }
            })
        })
        .max()
        .map(|w| w + number_width)
        .unwrap_or(10)
        .clamp(10, 44);
    let outcome = with_live_view(provider, options, label_width, view, |rt| {
        job::run(spec, rt)
    })
    .await;
    outcome.map_err(job_folder_error)
}

/// A job folder that could not be used: another run holds it, or this
/// machine cannot write it (the user's to fix); anything else is a bug.
pub fn job_folder_error(e: std::io::Error) -> CliError {
    use std::io::ErrorKind::*;
    match e.kind() {
        WouldBlock => CliError::not_ready(e.to_string()),
        StorageFull | PermissionDenied | ReadOnlyFilesystem | QuotaExceeded => CliError::failed(
            format!("the job folder could not be written: {e}"),
        )
        .owner(Owner::You)
        .next("free disk space or fix the folder's permissions, then rerun the same command to resume the job"),
        _ => CliError::bug("writing the job folder", e),
    }
}

/// Run `work` on a fresh runtime with the live view. The first Ctrl-C
/// drains the run: no new trial starts, and running ones finish. The second
/// stops it: running trials are cancelled and their sandboxes removed. A
/// third exits at once. A `kill` or a closed terminal stops at once.
pub async fn with_live_view<T, F, Fut>(
    provider: Arc<dyn Provider>,
    options: RunOptions,
    label_width: usize,
    view: ui::live::View,
    work: F,
) -> T
where
    F: FnOnce(Arc<Runtime>) -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let (stop_tx, stop_rx) = benchflow_run::trial::stop_switch();
    let drain = options.drain.clone();
    let rt = Arc::new(Runtime {
        provider,
        images: Images::new().build_timeout_multiplier(options.build_timeout_multiplier),
        options,
        events: Events::new(tx),
        stop: stop_rx,
    });
    let render = tokio::spawn(ui::live::render(rx, label_width, view));
    let interrupt = tokio::spawn(async move {
        let quit = drain_stop_quit(
            move || {
                drain.start();
                if !ui().json {
                    anstream::eprintln!(
                        "\n  {} draining: no new trials start; running ones finish (Ctrl-C again to stop them now)",
                        warn_mark()
                    );
                }
            },
            move || {
                let _ = stop_tx.send(true);
                if !ui().json {
                    anstream::eprintln!(
                        "\n  {} stopping: no new trials; cancelling running ones and removing their sandboxes (Ctrl-C again to quit now)",
                        warn_mark()
                    );
                }
            },
        );
        if quit.await {
            if !ui().json {
                // Not `eprintln!`: it panics when stderr is gone (a closed
                // terminal's SIGHUP), and the run would never exit.
                use std::io::Write;
                let _ = writeln!(
                    anstream::stderr(),
                    "\n  {} quitting now: any sandbox still being removed or built is left; a later `bench run` on this machine removes it once it is idle and done building",
                    warn_mark()
                );
            }
            std::process::exit(130);
        }
    });
    let outcome = work(rt.clone()).await;
    // A sandbox still being made when the run was stopped is removed as
    // soon as its backend hands it over; another Ctrl-C still quits.
    let settled =
        benchflow_run::trial::settle_stopped_starts(std::time::Duration::from_secs(120)).await;
    // A Daytona sandbox still building its image is deleted once it has
    // built (deleting it mid-build cancels the build, which Daytona keeps).
    let cleaned = benchflow_sandbox::settle_cleanups(std::time::Duration::from_secs(60)).await;
    if !ui().json {
        if !settled {
            anstream::eprintln!(
                "  {} a sandbox was still being made after 2 minutes; the next `bench run` here removes it",
                warn_mark()
            );
        }
        if !cleaned {
            anstream::eprintln!(
                "  {} a Daytona sandbox was still building its image after a minute, and is left to finish (deleting it would cancel the build); a later `bench run` on this machine removes it once it is done and idle",
                warn_mark()
            );
        }
        // D4 chaos runs: a delete that outlasted its retries left the
        // sandbox running, and only the trial's log said so.
        let mut left = benchflow_run::trial::left_behind();
        left.extend(benchflow_sandbox::left_behind());
        if let Some(note) = left_behind_note(&left) {
            anstream::eprintln!("  {} {note}", warn_mark());
        }
    }
    interrupt.abort();
    drop(rt);
    let _ = render.await;
    outcome
}

/// The end-of-run warning for sandboxes the run could not remove (each
/// `<backend> <id>`): which, and what removes them.
fn left_behind_note(left: &[String]) -> Option<String> {
    if left.is_empty() {
        return None;
    }
    let mut by_backend: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for entry in left {
        let (backend, id) = entry.split_once(' ').unwrap_or(("", entry.as_str()));
        by_backend.entry(backend).or_default().push(id);
    }
    let it = if left.len() == 1 { "it" } else { "them" };
    let ways: Vec<String> = by_backend
        .iter()
        .map(|(backend, ids)| match *backend {
            "docker" | "podman" => format!("`{backend} rm -f {}`", ids.join(" ")),
            "daytona" => format!("Daytona's dashboard ({})", ids.join(", ")),
            other => format!("the {other} backend ({})", ids.join(", ")),
        })
        .collect();
    Some(format!(
        "could not remove {}: {}; a later `bench run` on this machine removes {it} once idle and done building, or remove {it} now: {}",
        plural(left.len(), "sandbox"),
        left.join(", "),
        ways.join("; ")
    ))
}

/// Wait for signals: a first Ctrl-C calls `on_drain`; the next one (or a
/// first `kill` or hang-up) calls `on_stop`; then one more returns true, to
/// quit now. False when signals cannot be caught.
async fn drain_stop_quit(on_drain: impl FnOnce(), on_stop: impl FnOnce()) -> bool {
    match stop_signal().await {
        None => return false,
        Some(Signal::Interrupt) => {
            on_drain();
            if stop_signal().await.is_none() {
                return false;
            }
        }
        Some(Signal::Other) => {}
    }
    on_stop();
    stop_signal().await.is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signal {
    /// Ctrl-C.
    Interrupt,
    /// A `kill` or a closed terminal (SSH dropped): nobody is there to wait
    /// for a drain, so these stop the run at once.
    Other,
}

/// The next Ctrl-C, `kill`, or hang-up.
async fn stop_signal() -> Option<Signal> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        if let (Ok(mut term), Ok(mut hup)) = (
            signal(SignalKind::terminate()),
            signal(SignalKind::hangup()),
        ) {
            return tokio::select! {
                r = tokio::signal::ctrl_c() => r.ok().map(|_| Signal::Interrupt),
                _ = term.recv() => Some(Signal::Other),
                _ = hup.recv() => Some(Signal::Other),
            };
        }
    }
    tokio::signal::ctrl_c()
        .await
        .ok()
        .map(|_| Signal::Interrupt)
}

/// A failure group's heading: "2 trials errored [R201] · the agent's
/// problem". Trials Ctrl-C stopped say so, without blaming a setup.
fn cause_heading(status: Status, code: &str, owner: Owner, n: usize) -> String {
    let kind = match status {
        Status::Unscored => "unscored",
        Status::Skipped => "not run",
        _ => "errored",
    };
    format!(
        "{} {kind} {}",
        plural(n, "trial"),
        dim(format!("[{code}] · {}", owner.problem()))
    )
}

/// Tokens and model cost over some trials: `549.7K in · 18.4K out ·
/// $0.14`, or None when they used no tokens and recorded no cost (the
/// oracle, a scripted run).
pub fn usage(recs: &[&TrialRecord]) -> Option<String> {
    let input: u64 = recs
        .iter()
        .map(|r| r.usage.input_tokens + r.usage.cache_read_tokens + r.usage.cache_write_tokens)
        .sum();
    let output: u64 = recs.iter().map(|r| r.usage.output_tokens).sum();
    let costs: Vec<Option<f64>> = recs.iter().map(|r| r.usage.cost_usd).collect();
    if input + output == 0 && costs.iter().all(Option::is_none) {
        return None;
    }
    Some(format!(
        "{} in · {} out · {}",
        ui::live::human_count(input),
        ui::live::human_count(output),
        ui::cost(&costs)
    ))
}

/// The summary's Usage line: what the trials run now used, and, after a
/// resume, that the kept trials are not counted.
fn usage_line(now: &[&TrialRecord], kept: &[&TrialRecord]) -> Option<String> {
    let spent = usage(now);
    if kept.is_empty() {
        return spent.map(|u| format!("  {}  {u}", dim("Usage")));
    }
    let mut parts = vec![format!(
        "  {}  {} this run",
        dim("Usage"),
        spent.unwrap_or_else(|| "none".into())
    )];
    let earlier = match usage(kept) {
        Some(u) => format!("{u} earlier"),
        None => "nothing earlier".into(),
    };
    parts.push(format!(
        "not counted: the {} from an earlier run (kept or regraded), {earlier}",
        plural(kept.len(), "trial")
    ));
    Some(ui::join_fitting(&parts, 9, ui::width()))
}

/// The Score line, over the trials that count: those the user stopped
/// (Ctrl-C) are left out, and listed apart by [`stopped_lines`].
pub fn score_line(recs: &[TrialRecord]) -> String {
    let counted: Vec<&TrialRecord> = recs.iter().filter(|r| !r.stopped()).collect();
    let tally = benchflow_run::record::Tally::of(recs);
    let (total, passed, scored) = (tally.counted, tally.passed, tally.scored);
    let rate = Rate::new(passed, total);
    let mut score = format!("  {} {}/{} passed", bold("Score"), passed, total);
    // With nothing scored (a run stopped at once), a pass rate and its
    // interval say nothing.
    // With trials stopped, the rate is partial: the stopped trials are the
    // long ones, which fail more often. No interval then, and the rate as if
    // they had failed beside it.
    if total > 0 && scored > 0 && tally.stopped == 0 {
        score.push_str(&format!(" · {}", rate.display()));
    } else if total > 0 && scored > 0 {
        score.push_str(&format!(" · {:.1}%", 100.0 * passed as f64 / total as f64));
    }
    let rewards: Vec<f64> = counted.iter().filter_map(|r| r.reward).collect();
    if !rewards.is_empty() && rewards.iter().any(|r| *r != 0.0 && *r != 1.0) {
        score.push_str(&format!(
            " · mean reward {}",
            fmt_reward(rewards.iter().sum::<f64>() / rewards.len() as f64)
        ));
    }
    if scored < total {
        score.push_str(&dim(format!(" · {} of {} scored", scored, total)).to_string());
    }
    let stopped = tally.stopped;
    if stopped > 0 {
        score.push_str(
            &dim(format!(
                " · partial: {stopped} stopped ({:.1}% if they had failed)",
                100.0 * passed as f64 / (total + stopped) as f64
            ))
            .to_string(),
        );
    }
    score
}

/// The lines for trials the user stopped (Ctrl-C), in the words the run's
/// summary and `bench results` share; none when there are none.
pub fn stopped_lines(recs: &[TrialRecord]) -> Vec<String> {
    let stopped: Vec<&TrialRecord> = recs.iter().filter(|r| r.stopped()).collect();
    if stopped.is_empty() {
        return Vec::new();
    }
    let names: Vec<&str> = stopped
        .iter()
        .take(3)
        .map(|r| r.rollout_name.as_str())
        .collect();
    let more = if stopped.len() > 3 {
        format!(" and {} more", stopped.len() - 3)
    } else {
        String::new()
    };
    vec![
        format!(
            "  {} {} {} {}",
            warn_mark(),
            plural(stopped.len(), "trial"),
            benchflow_run::record::STOPPED_WORDS,
            dim("[R001]")
        ),
        format!("      {} {}{}", dim("trials:"), names.join(", "), more),
        format!(
            "      {} run the job's `bench run` command again to run them",
            dim("next:")
        ),
    ]
}

/// The end-of-run report: the score, failures grouped by cause with whose
/// problem each is, usage, and where to look next.
pub fn report(outcome: &JobOutcome) {
    if ui().json {
        return;
    }
    let recs = &outcome.records;
    say!("");
    say!("{}", score_line(recs));
    // Failures by cause; trials the user stopped come after, apart.
    let mut causes: BTreeMap<(Owner, String), Vec<&TrialRecord>> = BTreeMap::new();
    for r in recs {
        if let Some(p) = &r.problem {
            if r.status == Status::Skipped && p.code == "R000" || r.stopped() {
                continue;
            }
            causes.entry((p.owner, p.code.clone())).or_default().push(r);
        }
    }
    for ((owner, code), rs) in &causes {
        let p = rs[0].problem.as_ref().expect("grouped by problem");
        say!(
            "  {} {}",
            warn_mark(),
            cause_heading(rs[0].status, code, *owner, rs.len())
        );
        say!("{}", ui::wrap("      ", &p.message, 6, ui::width()));
        let names: Vec<&str> = rs.iter().take(3).map(|r| r.rollout_name.as_str()).collect();
        let more = if rs.len() > 3 {
            format!(" and {} more", rs.len() - 3)
        } else {
            String::new()
        };
        say!("      {} {}{}", dim("trials:"), names.join(", "), more);
        say!(
            "{}",
            ui::wrap(
                &format!("      {} ", dim("next:")),
                &p.next,
                12,
                ui::width()
            )
        );
    }
    for line in stopped_lines(recs) {
        say!("{line}");
    }
    // Zeros that may measure a broken task rather than the agent.
    let suspect: Vec<&TrialRecord> = recs
        .iter()
        .filter(|r| !r.environment_warnings.is_empty())
        .collect();
    if let Some(first) = suspect.first() {
        say!(
            "  {} {} scored, but the logs show the environment failing {}",
            warn_mark(),
            plural(suspect.len(), "trial"),
            dim("· the score stands")
        );
        say!(
            "{}",
            ui::wrap("      ", &first.environment_warnings[0], 6, ui::width())
        );
        let names: Vec<&str> = suspect
            .iter()
            .take(3)
            .map(|r| r.rollout_name.as_str())
            .collect();
        let more = if suspect.len() > 3 {
            format!(" and {} more", suspect.len() - 3)
        } else {
            String::new()
        };
        say!("      {} {}{}", dim("trials:"), names.join(", "), more);
        say!(
            "{}",
            ui::wrap(
                &format!("      {} ", dim("next:")),
                "check the task with `bench test`: if its oracle fails too, the task is broken, \
                 not the agent; environment_warnings in each result.json lists what was seen",
                12,
                ui::width()
            )
        );
    }
    if outcome.not_started > 0 {
        say!(
            "  {} {} not started (a login ran out of quota, or you stopped or drained the run); rerun the same command to run them",
            warn_mark(),
            plural(outcome.not_started, "trial")
        );
    }
    // Usage and time. A resumed run counts only the trials it ran: the
    // kept ones were paid for by the earlier run.
    let now: Vec<&TrialRecord> = recs
        .iter()
        .filter(|r| !outcome.earlier.contains(&r.rollout_name))
        .collect();
    let kept: Vec<&TrialRecord> = recs
        .iter()
        .filter(|r| outcome.earlier.contains(&r.rollout_name))
        .collect();
    let elapsed = outcome
        .summary
        .get("elapsed_sec")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    if let Some(line) = usage_line(&now, &kept) {
        say!("{line}");
    }
    say!("  {}   {}", dim("Time"), ui::secs(elapsed));
    if outcome.carried_reused + outcome.carried_regraded > 0 {
        let resumed = outcome.reused - outcome.carried_reused;
        let mut parts = vec![
            format!("{} reused", outcome.carried_reused),
            format!("{} regraded from earlier jobs", outcome.carried_regraded),
        ];
        if resumed > 0 {
            parts.insert(0, format!("{resumed} kept from this job"));
        }
        say!(
            "  {}  {}: {}, {} ran now",
            dim("Job  "),
            ui::path(&outcome.dir),
            parts.join(", "),
            recs.len() - outcome.reused - outcome.carried_regraded
        );
    } else if outcome.resumed || outcome.reused > 0 {
        say!(
            "  {}  resumed {}: {} reused, {} ran now",
            dim("Job  "),
            ui::path(&outcome.dir),
            outcome.reused,
            recs.len() - outcome.reused
        );
    } else {
        say!("  {}  {}", dim("Job  "), ui::path(&outcome.dir));
    }
    let dir = ui::path(&outcome.dir);
    say!(
        "{}",
        ui::join_fitting(
            &[
                format!("  {}  bench results {dir}", dim("Next ")),
                format!("bench view {dir}")
            ],
            9,
            ui::width()
        )
    );
    let _ = (ok_mark, fail_mark);
}

#[cfg(all(test, unix))]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::drain_stop_quit;

    fn signal_self(name: &str) {
        let status = std::process::Command::new("kill")
            .args([name, &std::process::id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }

    /// Ctrl-C drains, a second stops, a third quits; a `kill` skips the
    /// drain. One test, since the signals go to the whole test process.
    #[tokio::test]
    async fn ctrl_c_drains_then_stops_then_quits_and_a_kill_stops_at_once() {
        use tokio::signal::unix::{SignalKind, signal};
        // Catch both in this process before sending any.
        let _int = signal(SignalKind::interrupt()).unwrap();
        let _term = signal(SignalKind::terminate()).unwrap();
        let pause = || tokio::time::sleep(Duration::from_millis(200));

        let drained = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let (d, s) = (drained.clone(), stopped.clone());
        let quit = tokio::spawn(drain_stop_quit(
            move || d.store(true, Ordering::SeqCst),
            move || s.store(true, Ordering::SeqCst),
        ));
        pause().await;
        signal_self("-INT");
        pause().await;
        assert!(drained.load(Ordering::SeqCst));
        assert!(
            !stopped.load(Ordering::SeqCst),
            "the first Ctrl-C only drains"
        );
        signal_self("-INT");
        pause().await;
        assert!(stopped.load(Ordering::SeqCst));
        assert!(!quit.is_finished());
        signal_self("-INT");
        let quit = tokio::time::timeout(Duration::from_secs(5), quit).await;
        assert!(quit.expect("the third Ctrl-C quits").unwrap());

        let drained = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let (d, s) = (drained.clone(), stopped.clone());
        let quit = tokio::spawn(drain_stop_quit(
            move || d.store(true, Ordering::SeqCst),
            move || s.store(true, Ordering::SeqCst),
        ));
        pause().await;
        signal_self("-TERM");
        pause().await;
        assert!(stopped.load(Ordering::SeqCst));
        assert!(
            !drained.load(Ordering::SeqCst),
            "a kill does not wait for a drain"
        );
        signal_self("-TERM");
        let quit = tokio::time::timeout(Duration::from_secs(5), quit).await;
        assert!(quit.expect("the second kill quits").unwrap());
    }
}

#[cfg(test)]
mod report_tests {
    use super::*;

    /// Reviews of rust/d4-chaos: the warning sent Docker users to "the
    /// backend's dashboard".
    #[test]
    fn sandboxes_left_behind_are_named_with_what_removes_them() {
        assert_eq!(left_behind_note(&[]), None);
        let note = left_behind_note(&["docker bf-hello-1".to_string()]).unwrap();
        assert_eq!(
            note,
            "could not remove 1 sandbox: docker bf-hello-1; a later `bench run` on this machine removes it once idle and done building, or remove it now: `docker rm -f bf-hello-1`"
        );
        let note = left_behind_note(&[
            "daytona 6f1c".to_string(),
            "podman bf-a".to_string(),
            "daytona 9e2d".to_string(),
        ])
        .unwrap();
        assert!(note.starts_with("could not remove 3 sandboxes: "), "{note}");
        assert!(note.contains(", or remove them now: "), "{note}");
        assert!(note.contains("Daytona's dashboard (6f1c, 9e2d)"), "{note}");
        assert!(note.contains("`podman rm -f bf-a`"), "{note}");
        assert!(!note.contains("backend's dashboard"), "{note}");
    }

    fn trial(name: &str, input: u64, cost: Option<f64>) -> TrialRecord {
        let v = serde_json::json!({
            "task_name": "t", "trial_name": name, "status": "scored",
            "agent_result": {"n_input_tokens": input, "n_output_tokens": 10, "cost_usd": cost},
            "started_at": "2026-10-10 13:32:40.000000",
            "finished_at": "2026-10-10 13:33:40.000000",
        });
        benchflow_run::read::record_from_json(&v, std::path::Path::new(name)).expect("a record")
    }

    #[test]
    fn a_resumed_run_counts_only_what_it_spent() {
        let kept = [trial("a", 549_700, Some(0.14))];
        let now = [trial("b", 1_000, Some(0.02))];
        let line = usage_line(
            &now.iter().collect::<Vec<_>>(),
            &kept.iter().collect::<Vec<_>>(),
        )
        .expect("a line");
        assert!(line.contains("1.0K in · 10 out · $0.02 this run"), "{line}");
        assert!(
            line.contains("the 1 trial from an earlier run (kept or regraded), 549.7K in · 10 out · $0.14 earlier"),
            "{line}"
        );
        // Nothing ran now (`--rerun failed` with no failures).
        let line = usage_line(&[], &kept.iter().collect::<Vec<_>>()).expect("a line");
        assert!(line.contains("none this run"), "{line}");
        // A fresh run reads as before.
        let line = usage_line(&now.iter().collect::<Vec<_>>(), &[]).expect("a line");
        assert!(line.ends_with("1.0K in · 10 out · $0.02"), "{line}");
        let unpriced = [trial("c", 1_000, None)];
        let line = usage_line(&unpriced.iter().collect::<Vec<_>>(), &[]).expect("a line");
        assert!(line.ends_with("cost not recorded"), "{line}");
    }

    /// Trials the user stopped are left out of the Score and listed apart,
    /// not blamed on a setup (on 2026-10-10 a run with 3 of 3 scored trials
    /// passed said `Score 3/5 passed · 60.0%`).
    #[test]
    fn stopped_trials_are_left_out_of_the_score_and_listed_apart() {
        let rec = |name: &str, v: serde_json::Value| {
            benchflow_run::read::record_from_json(&v, std::path::Path::new(name)).unwrap()
        };
        let passed =
            serde_json::json!({"status": "scored", "rewards": {"reward": 1.0}, "passed": true});
        let stopped = serde_json::json!({"status": "skipped", "problem": {
            "code": "R001", "owner": "you", "phase": "agent",
            "message": "stopped before it finished", "next": "rerun", "transient": false}});
        let mut recs: Vec<TrialRecord> = (0..3)
            .map(|i| rec(&format!("p{i}"), passed.clone()))
            .collect();
        recs.push(rec("s1", stopped.clone()));
        recs.push(rec("s2", stopped));
        // Without the colors a terminal gets.
        let plain = |text: &str| {
            let mut out = String::new();
            let mut chars = text.chars();
            while let Some(c) = chars.next() {
                if c == '\u{1b}' {
                    chars.by_ref().find(|c| *c == 'm');
                } else {
                    out.push(c);
                }
            }
            out
        };
        let score = plain(&score_line(&recs));
        assert!(
            score.contains("Score 3/3 passed · 100.0% · partial"),
            "{score}"
        );
        assert!(
            score.ends_with("· partial: 2 stopped (60.0% if they had failed)"),
            "{score}"
        );
        assert!(
            !score.contains("43.9"),
            "no interval for a partial rate: {score}"
        );
        let lines: Vec<String> = stopped_lines(&recs).iter().map(|l| plain(l)).collect();
        assert!(
            lines[0]
                .ends_with("2 trials stopped (Ctrl-C, a kill, or a closed terminal), not counted in the pass rate [R001]"),
            "{lines:?}"
        );
        assert!(lines[1].ends_with("s1, s2"), "{lines:?}");
        assert!(stopped_lines(&recs[..3]).is_empty());
        let errored = cause_heading(Status::Errored, "R201", Owner::Agent, 1);
        assert!(errored.starts_with("1 trial errored "), "{errored}");
        assert!(errored.contains("[R201]"), "{errored}");
    }
}

#[cfg(test)]
mod job_folder_tests {
    use super::*;

    #[test]
    fn a_busy_or_unwritable_job_folder_is_not_a_bug() {
        let busy = job_folder_error(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "another bench run is running the job",
        ));
        assert_eq!(busy.owner, Some(Owner::You));
        let full = job_folder_error(std::io::Error::from(std::io::ErrorKind::StorageFull));
        assert_eq!(full.owner, Some(Owner::You));
        assert!(full.next.is_some());
    }
}
