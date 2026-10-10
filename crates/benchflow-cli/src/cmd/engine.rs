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
/// stops new trials and cancels running ones (their sandboxes are still
/// torn down); a second Ctrl-C exits at once.
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
    outcome.map_err(|e| CliError::bug("writing the job folder", e))
}

/// Run `work` on a fresh runtime with the live view. The first Ctrl-C stops
/// new trials and cancels running ones; a second Ctrl-C exits at once.
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
    let rt = Arc::new(Runtime {
        provider,
        images: Images::new().build_timeout_multiplier(options.build_timeout_multiplier),
        options,
        events: Events::new(tx),
        stop: stop_rx,
    });
    let render = tokio::spawn(ui::live::render(rx, label_width, view));
    let interrupt = tokio::spawn(async move {
        let quit = stop_then_quit(move || {
            let _ = stop_tx.send(true);
            if !ui().json {
                anstream::eprintln!(
                    "\n  {} stopping: no new trials; cancelling running ones and removing their sandboxes (Ctrl-C again to quit now)",
                    warn_mark()
                );
            }
        });
        if quit.await {
            std::process::exit(130);
        }
    });
    let outcome = work(rt.clone()).await;
    // A sandbox still being made when the run was stopped is removed as
    // soon as its backend hands it over; a second Ctrl-C still quits.
    if !benchflow_run::trial::settle_stopped_starts(std::time::Duration::from_secs(120)).await
        && !ui().json
    {
        anstream::eprintln!(
            "  {} a sandbox was still being made after 2 minutes; the next run removes it",
            warn_mark()
        );
    }
    interrupt.abort();
    drop(rt);
    let _ = render.await;
    outcome
}

/// Wait for a stop signal and call `on_stop`, then wait for a second one
/// (another Ctrl-C, `kill`, or hang-up): true when it came, to quit now.
async fn stop_then_quit(on_stop: impl FnOnce()) -> bool {
    if !stop_signal().await {
        return false;
    }
    on_stop();
    stop_signal().await
}

/// The first Ctrl-C, a `kill`, or a closed terminal (SSH dropped): each
/// stops the run the same way, so its sandboxes are removed.
async fn stop_signal() -> bool {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        if let (Ok(mut term), Ok(mut hup)) = (
            signal(SignalKind::terminate()),
            signal(SignalKind::hangup()),
        ) {
            return tokio::select! {
                r = tokio::signal::ctrl_c() => r.is_ok(),
                _ = term.recv() => true,
                _ = hup.recv() => true,
            };
        }
    }
    tokio::signal::ctrl_c().await.is_ok()
}

/// A failure group's heading: "2 trials errored [R201] · the agent's
/// problem". Trials Ctrl-C stopped say so, without blaming a setup.
fn cause_heading(status: Status, code: &str, owner: Owner, n: usize) -> String {
    let kind = match status {
        Status::Unscored => "unscored",
        Status::Skipped if code == "R001" => {
            return format!(
                "{} stopped {}",
                plural(n, "trial"),
                dim(format!("[{code}] · you stopped the run"))
            );
        }
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
        "not counted: the {} kept, {earlier}",
        plural(kept.len(), "trial")
    ));
    Some(ui::join_fitting(&parts, 9, ui::width()))
}

/// The end-of-run report: the score, failures grouped by cause with whose
/// problem each is, usage, and where to look next.
pub fn report(outcome: &JobOutcome) {
    if ui().json {
        return;
    }
    let recs = &outcome.records;
    let agent_recs: Vec<&TrialRecord> = recs.iter().collect();
    let total = agent_recs.len();
    let passed = agent_recs.iter().filter(|r| r.passed == Some(true)).count();
    let scored = agent_recs
        .iter()
        .filter(|r| r.status == Status::Scored)
        .count();
    let rate = Rate::new(passed, total);
    say!("");
    let mut score = format!("  {} {}/{} passed", bold("Score"), passed, total);
    // With nothing scored (a run stopped at once), a pass rate and its
    // interval say nothing.
    if total > 0 && scored > 0 {
        score.push_str(&format!(" · {}", rate.display()));
    }
    let rewards: Vec<f64> = agent_recs.iter().filter_map(|r| r.reward).collect();
    if !rewards.is_empty() && rewards.iter().any(|r| *r != 0.0 && *r != 1.0) {
        score.push_str(&format!(
            " · mean reward {}",
            fmt_reward(rewards.iter().sum::<f64>() / rewards.len() as f64)
        ));
    }
    if scored < total {
        score.push_str(&dim(format!(" · {} of {} scored", scored, total)).to_string());
    }
    say!("{score}");
    // Failures by cause.
    let mut causes: BTreeMap<(Owner, String), Vec<&TrialRecord>> = BTreeMap::new();
    for r in recs {
        if let Some(p) = &r.problem {
            if r.status == Status::Skipped && p.code == "R000" {
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
    if outcome.not_started > 0 {
        say!(
            "  {} {} not started (a login ran out of quota, or the run stopped); rerun to resume",
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

    use super::stop_then_quit;

    fn terminate_self() {
        let status = std::process::Command::new("kill")
            .args(["-TERM", &std::process::id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[tokio::test]
    async fn a_second_kill_quits_after_the_first_stops() {
        use tokio::signal::unix::{SignalKind, signal};
        // Catch SIGTERM in this process before sending any.
        let _held = signal(SignalKind::terminate()).unwrap();
        let stopped = Arc::new(AtomicBool::new(false));
        let flag = stopped.clone();
        let quit = tokio::spawn(stop_then_quit(move || flag.store(true, Ordering::SeqCst)));
        tokio::time::sleep(Duration::from_millis(100)).await;
        terminate_self();
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(stopped.load(Ordering::SeqCst));
        assert!(!quit.is_finished());
        terminate_self();
        let quit = tokio::time::timeout(Duration::from_secs(5), quit).await;
        assert!(quit.expect("the second kill quits").unwrap());
    }
}

#[cfg(test)]
mod report_tests {
    use super::*;

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
            line.contains("the 1 trial kept, 549.7K in · 10 out · $0.14 earlier"),
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

    #[test]
    fn stopped_trials_are_not_blamed_on_a_setup() {
        let stopped = cause_heading(Status::Skipped, "R001", Owner::You, 2);
        assert!(stopped.starts_with("2 trials stopped "), "{stopped}");
        assert!(stopped.contains("you stopped the run"), "{stopped}");
        assert!(!stopped.contains("problem"), "{stopped}");
        let errored = cause_heading(Status::Errored, "R201", Owner::Agent, 1);
        assert!(errored.starts_with("1 trial errored "), "{errored}");
        assert!(errored.contains("[R201]"), "{errored}");
    }
}
