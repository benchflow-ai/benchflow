//! `bench run`: run an agent on tasks and score them.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use benchflow_run::config::{DatasetStamp, Rerun};
use benchflow_run::diff;
use benchflow_run::job::JobSpec;
use benchflow_run::jobfile::{self, JobFile};
use benchflow_run::outcome::{Owner, Status};
use benchflow_run::project::RunDefaults;
use benchflow_run::registry::{self, DigestCheck, RegistryError};
use benchflow_run::source;
use serde_json::{Value, json};

use crate::cli::RunArgs;
use crate::cmd::engine;
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::ui::{self, bold, dim, plural, ui, warn_mark};
use crate::{note, say};

pub async fn run(args: RunArgs) -> Result<Exit> {
    let project = context::project()?;
    let (job_file, job_files) = if args.config.is_empty() {
        (JobFile::default(), Vec::new())
    } else {
        let loaded = jobfile::load_sources(&args.config).await.map_err(|e| {
            if e.unreadable {
                CliError::usage(format!("the job file could not be read: {e}"))
            } else {
                CliError::usage(format!("the job file is invalid: {e}"))
                    .next("`bench schema job` lists the keys a job file takes")
            }
        })?;
        (loaded.file, loaded.records)
    };
    let mut st = engine::settings_with(&project, &args.sandbox, Some(&job_file.run))?;
    // Flags win over job files, which win over bench.toml.
    let flags = RunDefaults {
        agent: args.agent.clone(),
        harness: args.harness.clone(),
        model: args.model.clone(),
        reasoning_effort: args.effort.clone(),
        trials: args.trials,
        timeout_multiplier: args.timeout_multiplier,
        agent_timeout_multiplier: args.agent_timeout_multiplier,
        verifier_timeout_multiplier: args.verifier_timeout_multiplier,
        build_timeout_multiplier: args.build_timeout_multiplier,
        idle_timeout: args.idle_timeout.clone(),
        fail_under: args.fail_under,
        env: context::parse_env(&args.env)?,
        agent_options: context::parse_agent_options(&args.agent_options)?
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect(),
        resume_sessions: args.resume_sessions.then_some(true),
        skills: args.skills.clone(),
        task_skills: args.no_task_skills.then_some(false),
        load_trajectory: args.load_trajectory.clone(),
        ..RunDefaults::default()
    };
    st.defaults = st.defaults.overlay(&flags);
    let pick = |flag: &[String], file: &[String]| {
        if flag.is_empty() {
            file.to_vec()
        } else {
            flag.to_vec()
        }
    };
    let agent_name = st.defaults.agent.clone().unwrap_or_else(|| "oracle".into());
    let trials = st.defaults.trials.unwrap_or(1).max(1);
    let resolved = JobFile {
        targets: pick(&args.targets, &job_file.targets),
        dataset: args.dataset.clone().or(job_file.dataset.clone()),
        registry: args.registry.clone().or(job_file.registry.clone()),
        include: pick(&args.include, &job_file.include),
        exclude: pick(&args.exclude, &job_file.exclude),
        run: RunDefaults {
            agent: Some(agent_name.clone()),
            trials: Some(trials),
            concurrency: Some(st.concurrency),
            sandbox: st.sandbox.clone(),
            jobs_dir: Some(st.jobs_dir.clone()),
            ..st.defaults.clone()
        },
    };
    if args.print_config {
        if ui().json {
            println!(
                "{}",
                serde_json::to_string_pretty(&resolved).unwrap_or_default()
            );
        } else {
            print!("{}", resolved.to_toml());
        }
        return Ok(Exit::Ok);
    }
    if resolved.dataset.is_some() && !resolved.targets.is_empty() {
        return Err(CliError::usage(
            "both a dataset and targets were given; a run takes one or the other",
        )
        .next(
            "drop the targets, or drop --dataset (a job file's `targets = []` clears its list)",
        ));
    }
    if resolved.registry.is_some() && resolved.dataset.is_none() {
        return Err(CliError::usage(
            "--registry names where to find --dataset, and no dataset was given",
        )
        .next("add -d <name>@<version>"));
    }
    let rerun = match &args.rerun {
        Some(r) => Rerun::parse(r).map_err(CliError::usage)?,
        None => Rerun::default(),
    };
    let fail_under = resolved.run.fail_under;
    if let Some(f) = fail_under
        && !(0.0..=1.0).contains(&f)
    {
        return Err(CliError::usage(format!(
            "--fail-under takes a rate from 0 to 1, not {f}"
        )));
    }

    // What to run.
    let cache = benchflow_run::config::default_cache_dir();
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut sources = Vec::new();
    let mut targets: Vec<String> = resolved.targets.clone();
    let mut dataset: Option<(registry::Dataset, Vec<registry::FetchedTask>)> = None;
    let mut listed = 0;
    if let Some(spec) = &resolved.dataset {
        let mut d = registry::resolve(spec, resolved.registry.as_deref(), &cache)
            .await
            .map_err(registry_error)?;
        listed = d.tasks.len();
        note!("{}", dataset_line(&d, ui::width()));
        if let Some(range) = &d.bench_version {
            note!(
                "  {} the registry says this dataset was validated with bench {range}",
                dim("note:")
            );
        }
        // Fetch only what will run.
        d.tasks = filter_dataset(d.tasks, &resolved.include, &resolved.exclude, args.limit)?;
        if d.tasks.len() < listed {
            note!(
                "  {} {} of {} after --include, --exclude, and --limit",
                dim("selected"),
                d.tasks.len(),
                listed
            );
        }
        // A dry run fetches nothing unless it must, to tell whether the
        // run would resume a job: that needs the tasks' digests.
        if args.dry_run
            && args.diff.is_empty()
            && !may_resume(&st.jobs_dir, args.job.as_deref(), args.fresh)
        {
            return dataset_plan(&d, listed, &resolved, None);
        }
        let fetched = registry::fetch(&d, &cache, &|m: &str| {
            note!("  {} {}", dim("fetching"), m);
        })
        .await
        .map_err(registry_error)?;
        report_digests(&d, &fetched);
        roots = fetched.iter().map(|f| f.path.clone()).collect();
        targets = vec![spec.clone()];
        dataset = Some((d, fetched));
    } else {
        if targets.is_empty() {
            targets = context::task_roots(&project, &[])
                .iter()
                .map(|p| p.display().to_string())
                .collect();
        }
        for t in &targets {
            let is_remote = !Path::new(t).exists() && source::parse_remote(t).is_some();
            if is_remote {
                note!("  {} {}", dim("fetching"), t);
            }
            let s = source::resolve(t, &cache).await.map_err(source_error)?;
            roots.push(s.path.clone());
            sources.push(s);
        }
    }
    let (tasks, bad) = context::load_tasks(&roots)?;
    for r in &bad {
        let errs = r.diagnostics.iter().filter(|d| d.is_error()).count();
        note!(
            "  {} skipping {}: {} ({})",
            warn_mark(),
            ui::path(&r.root),
            plural(errs, "error"),
            dim(format!("bench check {}", ui::path(&r.root)))
        );
    }
    if tasks.is_empty() && !bad.is_empty() {
        return Err(CliError::failed(format!(
            "none of the {} loaded; each has errors (listed above)",
            plural(bad.len(), "task")
        ))
        .owner(Owner::Task)
        .next("run `bench check` on a task above to see its errors"));
    }
    let tasks = context::select(tasks, &resolved.include, &resolved.exclude, args.limit)?;
    if tasks.is_empty() {
        return Err(CliError::usage(
            "no tasks left to run after --include, --exclude, and --limit",
        ));
    }
    // What the reader dropped from a task (host values in its Compose file).
    for t in &tasks {
        for d in t
            .extras
            .diagnostics
            .iter()
            .filter(|d| d.severity == benchflow_task::Severity::Warning)
        {
            note!("  {} {}: {} [{}]", warn_mark(), t.name, d.message, d.code);
        }
    }

    // Who runs them.
    let model = resolved.run.model.clone();
    let effort = resolved.run.reasoning_effort.clone();
    let env = resolved.run.env.clone();
    let agent_options = context::agent_option_pairs(&resolved.run.agent_options);
    let attempt = engine::attempt(
        &agent_name,
        model.as_deref(),
        effort.as_deref(),
        resolved.run.harness.as_deref(),
        args.backend.as_deref(),
        &env,
        &agent_options,
    )
    .await?;
    let resume_sessions = resolved.run.resume_sessions.unwrap_or(false);
    if resume_sessions
        && let benchflow_run::Attempt::Model(m) = &attempt
        && m.session_dirs().is_none()
    {
        note!(
            "  {} {} cannot resume its own session on this harness; each graded stage starts a fresh one",
            dim("note:"),
            attempt.label()
        );
    }
    let skills = run_skills(&resolved.run, &attempt)?;
    if let benchflow_run::Attempt::Model(m) = &attempt
        && m.skill_paths().is_empty()
        && resolved.run.task_skills != Some(false)
    {
        let with = tasks
            .iter()
            .filter(|t| t.sandbox.skills.is_some() || t.sandbox.skills_dir.is_some())
            .count();
        if with > 0 {
            note!(
                "  {} {} has no skills folder, so {} run without their skills",
                dim("note:"),
                attempt.label(),
                plural(with, "task")
            );
        }
    }
    let load_trajectory = loaded_session(&resolved.run, &attempt)?;
    if let benchflow_run::Attempt::Model(m) = &attempt {
        m.preflight().map_err(|p| {
            CliError::not_ready(p.message.clone())
                .code(p.code.clone())
                .owner(p.owner)
                .next(p.next.clone())
        })?;
    }

    // The plan, before anything starts.
    let units: Vec<_> = tasks
        .iter()
        .map(|t| (t.clone(), vec![attempt.clone()]))
        .collect();
    let refused: Vec<(String, String)> = tasks
        .iter()
        .filter_map(|t| {
            benchflow_run::trial::refusal(t, &attempt).map(|p| (t.name.clone(), p.message))
        })
        .collect();
    let diff_jobs = diff_jobs(&args.diff, &st.jobs_dir)?;
    if args.dry_run && !diff_jobs.is_empty() {
        let mut options =
            engine::options_for(&st, st.sandbox.as_deref().unwrap_or(""), args.sandbox.keep)?;
        apply_flags(&mut options, &args, rerun);
        options.job_files = job_files;
        let (plan, _) = diff_plan(&diff_jobs, &units, trials, args.job.as_deref(), &options)?;
        print_diff(&plan, true);
        return Ok(Exit::Ok);
    }
    if args.dry_run {
        let mut options =
            engine::options_for(&st, st.sandbox.as_deref().unwrap_or(""), args.sandbox.keep)?;
        apply_flags(&mut options, &args, rerun);
        options.skills = skills;
        options.task_skills = resolved.run.task_skills.unwrap_or(true);
        options.load_trajectory = load_trajectory;
        let preview = benchflow_run::job::preview(
            &JobSpec {
                carry: Vec::new(),
                units: units.clone(),
                trials,
                jobs_dir: st.jobs_dir.clone(),
                job_name: args.job.clone(),
                fresh: args.fresh,
                concurrency: st.concurrency,
                source: Value::Null,
            },
            &options,
        );
        if let Some((d, _)) = &dataset {
            return dataset_plan(d, listed, &resolved, Some(&preview));
        }
        let plan = json!({
            "agent": attempt.label(),
            "model": attempt.model(),
            "trials_per_task": trials,
            "concurrency": st.concurrency,
            "sandbox": st.sandbox,
            "jobs_dir": st.jobs_dir,
            "tasks": tasks.iter().map(|t| json!({"name": t.name, "id": t.id, "path": t.root, "digest": t.digest, "format": t.format})).collect::<Vec<_>>(),
            "refused": refused.iter().map(|(t, m)| json!({"task": t, "reason": m})).collect::<Vec<_>>(),
            "sources": sources,
            "job": job_json(Some(&preview)),
        });
        if ui().json {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan).unwrap_or_default()
            );
        } else {
            say!("{}", bold("Plan (nothing runs with --dry-run)"));
            say!(
                "  agent    {}{}",
                attempt.label(),
                model
                    .as_ref()
                    .map(|m| format!(" · {m}"))
                    .unwrap_or_default()
            );
            say!(
                "  tasks    {} × {} = {}",
                plural(tasks.len(), "task"),
                plural(trials, "trial"),
                plural(tasks.len() * trials, "trial")
            );
            say!(
                "  sandbox  {} · {} at a time",
                st.sandbox.clone().unwrap_or_else(|| "auto".into()),
                st.concurrency
            );
            say!("  job      {}", job_text(Some(&preview)));
            for t in tasks.iter().take(20) {
                say!("    {} {}", dim("·"), t.name);
            }
            if tasks.len() > 20 {
                say!("    {}", dim(format!("… and {} more", tasks.len() - 20)));
            }
            for (t, m) in &refused {
                say!("  {} {} would be refused: {}", warn_mark(), t, m);
            }
        }
        return Ok(Exit::Ok);
    }

    let provider = context::provider(st.sandbox.as_deref()).await?;
    if let Ok(n) = provider.sweep().await
        && n > 0
    {
        note!(
            "  {} removed {} left behind by earlier runs",
            dim("cleanup:"),
            plural(n, "sandbox")
        );
    }
    let mut options = engine::options(&st, provider.as_ref(), args.sandbox.keep)?;
    apply_flags(&mut options, &args, rerun);
    let mut source_json = json!({
        "targets": targets,
        "path": roots.first(),
        "remote": sources.iter().filter_map(|s| s.remote.clone()).collect::<Vec<_>>(),
    });
    if let Some((d, fetched)) = &dataset {
        options.dataset = Some(dataset_stamp(d, fetched, &tasks));
        source_json["dataset"] = json!({
            "name": d.name,
            "version": d.version,
            "registry": d.registry,
            "tasks": fetched.iter().map(|f| json!({"name": f.name, "path": f.path, "commit": f.commit, "digest": f.digest})).collect::<Vec<_>>(),
        });
    }
    options.resume_sessions = resume_sessions;
    options.skills = skills;
    options.task_skills = resolved.run.task_skills.unwrap_or(true);
    options.load_trajectory = load_trajectory;
    let carry = if diff_jobs.is_empty() {
        Vec::new()
    } else {
        let (plan, found) = diff_plan(&diff_jobs, &units, trials, args.job.as_deref(), &options)?;
        print_diff(&plan, false);
        plan.carries(&found)
    };

    let what = format!(
        "{} {} on {}",
        bold("Running"),
        match &model {
            Some(m) => format!("{} ({m})", attempt.label()),
            None => attempt.label(),
        },
        plural(tasks.len() * trials, "trial"),
    );
    let place = format!("{} · {} at a time", provider.name(), st.concurrency);
    note!("{}", ui::fit_or_break(&what, " · ", &place, 2, ui::width()));
    let spec = JobSpec {
        carry,
        units,
        trials,
        jobs_dir: st.jobs_dir.clone(),
        job_name: args.job.clone(),
        fresh: args.fresh,
        concurrency: st.concurrency,
        source: source_json,
    };
    let outcome = engine::run_job(
        spec,
        Arc::clone(&provider),
        options,
        ui::live::View::default(),
    )
    .await?;
    engine::report(&outcome);
    if ui().json {
        println!(
            "{}",
            serde_json::to_string(
                &json!({"event": "job-finished", "dir": outcome.dir, "summary": outcome.summary})
            )
            .unwrap_or_default()
        );
    }

    // Exit code: trials not scored, and the gate.
    let recs = &outcome.records;
    let total = recs.len();
    let passed = recs.iter().filter(|r| r.passed == Some(true)).count();
    let statuses: Vec<Status> = recs.iter().map(|r| r.status).collect();
    let mut exit = run_exit(&statuses, outcome.not_started);
    if let Some(f) = fail_under {
        let rate = if total == 0 {
            0.0
        } else {
            passed as f64 / total as f64
        };
        if rate + 1e-12 < f {
            say!(
                "  {} pass rate {:.1}% is below --fail-under {:.1}%",
                crate::ui::fail_mark(),
                rate * 100.0,
                f * 100.0
            );
            exit = Exit::Failed;
        }
    }
    Ok(exit)
}

/// 0 only when every trial was scored: an unscored or errored trial, one
/// stopped by Ctrl-C or a budget (skipped), or one never started is a 1, so
/// `bench run … && next` does not go on after a stopped run.
fn run_exit(statuses: &[Status], not_started: usize) -> Exit {
    let broken = statuses.iter().any(|s| *s != Status::Scored);
    if broken || not_started > 0 {
        Exit::Failed
    } else {
        Exit::Ok
    }
}

/// The `--skill` folders' skills, checked before anything starts: each
/// folder must be a skill or a folder of skills, and a model agent must have
/// somewhere to put them.
fn run_skills(run: &RunDefaults, attempt: &benchflow_run::Attempt) -> Result<Vec<PathBuf>> {
    if run.skills.is_empty() {
        return Ok(Vec::new());
    }
    let skills = benchflow_run::skills::resolve(&run.skills).map_err(|e| {
        CliError::usage(format!("--skill: {e}"))
            .next("pass a folder that holds a SKILL.md, or a folder whose folders each hold one")
    })?;
    match attempt {
        benchflow_run::Attempt::Model(m) if m.skill_paths().is_empty() => {
            return Err(CliError::usage(format!(
                "{} has no skills folder (its manifest names no skill_paths), so it cannot take --skill",
                attempt.label()
            ))
            .next("drop --skill, or pick an agent that reads skills (`bench agents <name>` shows its skill folders)"));
        }
        benchflow_run::Attempt::Model(_) => {}
        _ => note!(
            "  {} {} runs without skills; --skill applies to model agents",
            dim("note:"),
            attempt.label()
        ),
    }
    note!(
        "  {} {}",
        dim("skills"),
        skills
            .iter()
            .map(|p| benchflow_run::skills::name(p))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(skills)
}

/// The `--load-trajectory` file, checked before anything starts: the agent
/// must be able to load a session on this harness, and the file must be
/// one of its sessions.
fn loaded_session(run: &RunDefaults, attempt: &benchflow_run::Attempt) -> Result<Option<PathBuf>> {
    let Some(path) = &run.load_trajectory else {
        return Ok(None);
    };
    let benchflow_run::Attempt::Model(m) = attempt else {
        return Err(CliError::usage(format!(
            "--load-trajectory starts a model agent on a session, and {} is not one",
            attempt.label()
        ))
        .next("drop --load-trajectory, or run -a claude or -a codex"));
    };
    let Some(format) = m.session_file() else {
        return Err(CliError::usage(format!(
            "{} cannot load a session from a file on this harness",
            attempt.label()
        ))
        .next("run -a claude or -a codex, whose session files BenchFlow can place; `bench agents <name>` says whether an agent can"));
    };
    let bytes = std::fs::read(path).map_err(|e| {
        CliError::usage(format!(
            "--load-trajectory: {} cannot be read: {e}",
            ui::path(path)
        ))
    })?;
    let name = benchflow_run::skills::name(path);
    let id = format.check(&name, &bytes).map_err(|why| {
        let next = if name.ends_with(".json") {
            format!(
                "an ATIF trajectory (.json) cannot be loaded yet; pass {}",
                format.expects()
            )
        } else {
            format!("pass {}", format.expects())
        };
        CliError::usage(format!("--load-trajectory: {why}")).next(next)
    })?;
    note!("  {} session {id} from {}", dim("loads"), ui::path(path));
    Ok(Some(
        std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()),
    ))
}

/// The run flags that go into every trial's options.
fn apply_flags(o: &mut benchflow_run::RunOptions, args: &RunArgs, rerun: Rerun) {
    o.extra_instruction = args.extra_instruction.clone();
    o.save_outputs = args.save_outputs;
    o.rerun = rerun;
    o.max_tool_calls = args.max_tool_calls;
    o.max_tokens = args.max_tokens;
}

/// `--diff` jobs: folders as given, else names in the jobs folder.
fn diff_jobs(given: &[PathBuf], jobs_dir: &Path) -> Result<Vec<PathBuf>> {
    given
        .iter()
        .map(|g| {
            let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
            if g.is_dir() {
                return Ok(canon(g));
            }
            let named = jobs_dir.join(g);
            if named.is_dir() {
                return Ok(canon(&named));
            }
            Err(CliError::usage(format!(
                "--diff names no job: neither {} nor {} is a folder",
                ui::path(g),
                ui::path(&named)
            ))
            .next("pass a job folder, or the name of a job in the jobs folder (`bench results` lists them)"))
        })
        .collect()
}

/// Match the planned trials against the `--diff` jobs' trials.
fn diff_plan(
    jobs: &[PathBuf],
    units: &[(
        Arc<benchflow_task::TaskPackage>,
        Vec<benchflow_run::Attempt>,
    )],
    trials: usize,
    job_name: Option<&str>,
    options: &benchflow_run::RunOptions,
) -> Result<(diff::DiffPlan, Vec<diff::SourceTrial>)> {
    let name = job_name.unwrap_or("planned");
    let planned = diff::planned_trials(units, trials, name);
    let target = benchflow_run::lock::build(
        benchflow_run::lock::LockHeader {
            job_name: name,
            fingerprint: "",
            source: &Value::Null,
            trials_per_task: trials,
            concurrency: 1,
            regrade_of: None,
        },
        &planned,
        options,
    );
    let (found, skipped) = diff::load_sources(jobs);
    if skipped.len() == jobs.len() {
        let why: Vec<String> = skipped
            .iter()
            .map(|s| format!("{}: {}", ui::path(&s.job), s.reason))
            .collect();
        return Err(
            CliError::usage(format!("no --diff job could be read ({})", why.join("; "))).next(
                "pass a job this version of bench ran; it writes lock.json in every job folder",
            ),
        );
    }
    let plan = diff::plan(
        &planned,
        &target,
        jobs.to_vec(),
        &found,
        skipped,
        &options.rerun,
    );
    Ok((plan, found))
}

/// Print a reuse plan: one line per planned trial, then the counts.
fn print_diff(plan: &diff::DiffPlan, dry_run: bool) {
    if ui().json {
        let v = json!({"event": "diff", "plan": plan});
        if dry_run {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan).unwrap_or_default()
            );
        } else {
            println!("{}", serde_json::to_string(&v).unwrap_or_default());
        }
        return;
    }
    let heading = format!(
        "Reuse plan against {}",
        plural(plan.sources.len() - plan.skipped.len(), "earlier job")
    );
    if dry_run {
        say!("{} {}", bold(heading), dim("(nothing runs with --dry-run)"));
    } else {
        say!("{}", bold(heading));
    }
    for s in &plan.skipped {
        say!(
            "  {} skipped {}: {}",
            warn_mark(),
            ui::path(&s.job),
            s.reason
        );
    }
    let many = plan.trials.iter().any(|t| t.index > 0);
    let label = |t: &diff::TrialDiff| {
        if many {
            format!("{} #{}", t.task, t.index + 1)
        } else {
            t.task.clone()
        }
    };
    let w = plan
        .trials
        .iter()
        .map(|t| label(t).chars().count())
        .max()
        .unwrap_or(0)
        .min(40);
    for t in &plan.trials {
        let action = match t.action {
            diff::Action::Reuse => ui::paint(ui::GREEN, "reuse  "),
            diff::Action::Regrade => ui::paint(ui::CYAN, "regrade"),
            diff::Action::Rerun => "rerun  ".to_string(),
        };
        let from = t
            .source
            .as_ref()
            .map(|s| {
                let job = s
                    .job
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                format!(" (from {job}/{})", s.trial)
            })
            .unwrap_or_default();
        let first = format!("  {action}  {:<w$}  ", ui::truncate(&label(t), w), w = w);
        say!(
            "{}",
            ui::wrap(&first, &format!("{}{from}", t.reason), 13 + w, ui::width())
        );
    }
    let n = plan.trials.len();
    say!(
        "  {}: {} reuse · {} regrade · {} rerun",
        plural(n, "trial"),
        plan.count(diff::Action::Reuse),
        plan.count(diff::Action::Regrade),
        plan.count(diff::Action::Rerun)
    );
}

fn source_error(e: source::SourceError) -> CliError {
    match e {
        source::SourceError::NotFound(_) => CliError::usage(e.to_string())
            .next("pass a task folder, a folder of tasks, or a GitHub repository such as benchflow-ai/skillsbench/tasks"),
        source::SourceError::NoGit(_) => CliError::not_ready(e.to_string()).next("install git, then run again"),
        source::SourceError::Fetch { .. } => CliError::failed(e.to_string())
            .owner(Owner::You)
            .next("check the network, then the repository name, the path, and the ref; private repositories need git credentials"),
    }
}

fn registry_error(e: RegistryError) -> CliError {
    match e {
        RegistryError::BadRef(_) => CliError::usage(e.to_string()),
        RegistryError::NotFound { .. } => CliError::usage(e.to_string())
            .next("check the name and version; --registry <url-or-file> searches another registry"),
        RegistryError::Unreadable { .. } => CliError::failed(e.to_string())
            .owner(Owner::You)
            .next("check the network, or pass --registry with a local copy of the registry"),
        RegistryError::Invalid { .. } => CliError::failed(e.to_string())
            .owner(Owner::Task)
            .next("the registry is malformed; tell its maintainers, or pass --registry with a fixed copy"),
        RegistryError::Local(_) => CliError::usage(e.to_string())
            .next("fix the dataset.toml; it takes [dataset] name and version, and [[tasks]] with a name and optional digest and path"),
        RegistryError::DigestMismatch { .. } => CliError::failed(e.to_string())
            .owner(Owner::Task)
            .next("the tasks at the pinned commit are not what the registry published; tell the dataset's maintainers"),
        RegistryError::Fetch(inner) => source_error(inner),
    }
}

/// A dataset's tasks narrowed by name, as `context::select` narrows loaded
/// tasks, so only those are fetched.
fn filter_dataset(
    tasks: Vec<registry::DatasetTask>,
    include: &[String],
    exclude: &[String],
    limit: Option<usize>,
) -> Result<Vec<registry::DatasetTask>> {
    let compile = |pats: &[String]| -> Result<Option<globset::GlobSet>> {
        if pats.is_empty() {
            return Ok(None);
        }
        let mut b = globset::GlobSetBuilder::new();
        for p in pats {
            b.add(
                globset::Glob::new(p)
                    .map_err(|e| CliError::usage(format!("bad pattern `{p}`: {e}")))?,
            );
        }
        Ok(Some(b.build().map_err(|e| CliError::usage(e.to_string()))?))
    };
    let inc = compile(include)?;
    let exc = compile(exclude)?;
    let names: Vec<String> = tasks.iter().map(|t| t.name.clone()).collect();
    let mut out: Vec<registry::DatasetTask> = tasks
        .into_iter()
        .filter(|t| inc.as_ref().is_none_or(|g| g.is_match(&t.name)))
        .filter(|t| exc.as_ref().is_none_or(|g| !g.is_match(&t.name)))
        .collect();
    if let Some(n) = limit {
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out.truncate(n);
    }
    if out.is_empty() {
        return Err(no_tasks_left(&names, include));
    }
    Ok(out)
}

/// The dataset line a run starts with; the registry goes on its own line
/// when the whole does not fit the terminal.
fn dataset_line(d: &registry::Dataset, width: usize) -> String {
    ui::fit_or_break(
        &format!(
            "  {} {}@{} · {}",
            dim("dataset"),
            d.name,
            d.version,
            plural(d.tasks.len(), "task")
        ),
        " · ",
        &dim(&d.registry),
        10,
        width,
    )
}

/// Nothing left after --include and the rest: name each --include that
/// matched no task, with the task it most likely meant.
fn no_tasks_left(names: &[String], include: &[String]) -> CliError {
    let unmatched: Vec<String> = include
        .iter()
        .filter(|p| {
            globset::Glob::new(p)
                .map(|g| g.compile_matcher())
                .is_ok_and(|g| !names.iter().any(|n| g.is_match(n)))
        })
        .map(|p| {
            match benchflow_agent::suggest::did_you_mean(p, names.iter().map(String::as_str)) {
                Some(close) => format!("`{p}` (did you mean `{close}`?)"),
                None => format!("`{p}`"),
            }
        })
        .collect();
    let message = if unmatched.is_empty() {
        "no dataset tasks left after --include, --exclude, and --limit".to_string()
    } else {
        format!(
            "no task in the dataset matches --include {}",
            unmatched.join(", ")
        )
    };
    CliError::usage(message).next(format!(
        "check the names (globs such as 'fix-*' work); --dry-run --json without --include lists all {} of the dataset's tasks",
        names.len()
    ))
}

/// Say once how the registry's digests were checked.
fn report_digests(d: &registry::Dataset, fetched: &[registry::FetchedTask]) {
    let pinned = fetched
        .iter()
        .filter(|f| f.check != DigestCheck::NotPinned)
        .count();
    let matched = fetched
        .iter()
        .filter(|f| f.check == DigestCheck::Matched)
        .count();
    if let Some(DigestCheck::Skipped(why)) = fetched
        .iter()
        .map(|f| &f.check)
        .find(|c| matches!(c, DigestCheck::Skipped(_)))
    {
        note!("  {} {why}", dim("note:"));
    } else if pinned > 0 {
        note!(
            "  {} {} of {} match the registry",
            dim("digests"),
            matched,
            plural(pinned, "pinned task")
        );
    }
    let unpinned = d
        .tasks
        .iter()
        .filter(|t| {
            matches!(&t.location, registry::Location::Git { commit, .. } if commit.eq_ignore_ascii_case("head"))
        })
        .count();
    if unpinned > 0 {
        note!(
            "  {} the registry pins {} to no commit, so they ran at the repository's current HEAD (lock.json records the commit)",
            warn_mark(),
            plural(unpinned, "task")
        );
    }
}

/// The dataset's identity and pinned digests, keyed by the loaded tasks'
/// folders.
fn dataset_stamp(
    d: &registry::Dataset,
    fetched: &[registry::FetchedTask],
    tasks: &[Arc<benchflow_task::TaskPackage>],
) -> DatasetStamp {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let mut digests = std::collections::BTreeMap::new();
    for t in tasks {
        let root = canon(&t.root);
        if let Some(digest) = fetched
            .iter()
            .find(|f| canon(&f.path) == root)
            .and_then(|f| f.digest.clone())
        {
            digests.insert(t.root.clone(), digest);
        }
    }
    DatasetStamp {
        name: d.name.clone(),
        version: d.version.clone(),
        registry: d.registry.clone(),
        digests,
    }
}

/// The plan's summary of a dataset's selected tasks: how many of those
/// listed, where they come from, and how many the registry pins by digest.
fn plan_task_lines(d: &registry::Dataset, listed: usize) -> Vec<String> {
    let n = d.tasks.len();
    let mut lines = vec![if n < listed {
        format!(
            "  tasks    {} of {listed} (after --include, --exclude, and --limit)",
            plural(n, "task")
        )
    } else {
        format!("  tasks    {}", plural(n, "task"))
    }];
    let mut from: Vec<(String, usize)> = Vec::new();
    for t in &d.tasks {
        let place = match &t.location {
            registry::Location::Git { url, commit, .. } => {
                format!("{url} at {}", commit.chars().take(12).collect::<String>())
            }
            registry::Location::Local(_) => "local folders".to_string(),
        };
        match from.iter_mut().find(|(p, _)| *p == place) {
            Some((_, k)) => *k += 1,
            None => from.push((place, 1)),
        }
    }
    for (place, k) in &from {
        let count = if from.len() > 1 {
            format!(" ({k})")
        } else {
            String::new()
        };
        lines.push(format!("  from     {place}{count}"));
    }
    let pinned = d.tasks.iter().filter(|t| t.digest.is_some()).count();
    lines.push(if pinned == 0 {
        "  digests  none in the registry; lock.json records each task's digest after fetching"
            .to_string()
    } else {
        format!("  digests  {pinned} of {n} pinned by the registry, checked after fetching")
    });
    lines
}

/// `--dry-run` with a dataset: its tasks as the registry lists them,
/// without fetching anything.
/// Whether a run could resume a job here, judged without the tasks: false
/// when the run would surely start a new job (`--fresh`, a `--job` name not
/// used yet, or no job in the jobs folder).
fn may_resume(jobs_dir: &Path, job: Option<&str>, fresh: bool) -> bool {
    if let Some(name) = job {
        return jobs_dir.join(name).join("evaluation.json").is_file();
    }
    !fresh
        && std::fs::read_dir(jobs_dir).is_ok_and(|mut dirs| {
            dirs.any(|e| e.is_ok_and(|e| e.path().join("evaluation.json").is_file()))
        })
}

/// The plan's job line: whether the run would resume a job, and how much of
/// it, or start a new one. No preview: it would surely start a new one.
fn job_text(p: Option<&benchflow_run::job::Preview>) -> String {
    match p {
        Some(p) if p.resumed => format!(
            "would resume {} ({} of {} kept)",
            ui::path(&p.dir),
            p.kept,
            plural(p.total, "trial")
        ),
        _ => "would start a new job".into(),
    }
}

fn job_json(p: Option<&benchflow_run::job::Preview>) -> Value {
    match p {
        Some(p) if p.resumed => {
            json!({"resume": true, "dir": p.dir, "kept": p.kept, "trials": p.total})
        }
        _ => json!({"resume": false}),
    }
}

fn dataset_plan(
    d: &registry::Dataset,
    listed: usize,
    resolved: &JobFile,
    preview: Option<&benchflow_run::job::Preview>,
) -> Result<Exit> {
    let tasks: Vec<Value> = d
        .tasks
        .iter()
        .map(|t| match &t.location {
            registry::Location::Git { url, commit, path } => {
                json!({"name": t.name, "git_url": url, "commit": commit, "path": path, "digest": t.digest})
            }
            registry::Location::Local(p) => json!({"name": t.name, "path": p, "digest": t.digest}),
        })
        .collect();
    if ui().json {
        let plan = json!({
            "dataset": {"name": d.name, "version": d.version, "registry": d.registry, "description": d.description},
            "agent": resolved.run.agent,
            "model": resolved.run.model,
            "trials_per_task": resolved.run.trials,
            "tasks": tasks,
            "job": job_json(preview),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&plan).unwrap_or_default()
        );
        return Ok(Exit::Ok);
    }
    // With a preview, the tasks were fetched to tell whether it resumes.
    say!(
        "{}",
        bold(if preview.is_some() {
            "Plan (nothing runs with --dry-run)"
        } else {
            "Plan (nothing runs or is fetched with --dry-run)"
        })
    );
    say!("  dataset  {}@{}", d.name, d.version);
    if let Some(desc) = &d.description {
        say!("{}", dim(ui::wrap("           ", desc, 11, ui::width())));
    }
    say!(
        "  agent    {}{}",
        resolved.run.agent.clone().unwrap_or_default(),
        resolved
            .run
            .model
            .as_ref()
            .map(|m| format!(" · {m}"))
            .unwrap_or_default()
    );
    for line in plan_task_lines(d, listed) {
        say!("{line}");
    }
    say!("  job      {}", job_text(preview));
    for t in d.tasks.iter().take(20) {
        say!("    {} {}", dim("·"), t.name);
    }
    if d.tasks.len() > 20 {
        say!(
            "    {}",
            dim(format!(
                "… and {} more (--json lists them all)",
                d.tasks.len() - 20
            ))
        );
    }
    Ok(Exit::Ok)
}

#[cfg(test)]
mod exit_tests {
    use super::*;

    #[test]
    fn the_dataset_line_fits_100_columns() {
        let d = registry::Dataset {
            name: "terminal-bench".into(),
            version: "2.0".into(),
            description: None,
            registry: "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
                .into(),
            bench_version: None,
            tasks: Vec::new(),
        };
        let wide = dataset_line(&d, usize::MAX);
        assert_eq!(wide.lines().count(), 1, "{wide}");
        let narrow = dataset_line(&d, 100);
        assert_eq!(narrow.lines().count(), 2, "{narrow}");
        for line in narrow.lines() {
            assert!(ui::visible_width(line) <= 100, "{line}");
        }
    }

    #[test]
    fn an_include_that_matches_no_task_names_the_closest() {
        let names: Vec<String> = ["fix-git", "regex-log"].map(String::from).to_vec();
        let e = no_tasks_left(&names, &["fix-gti".into(), "regex-*".into()]);
        let text = format!("{e:?}");
        assert!(
            text.contains(
                "no task in the dataset matches --include `fix-gti` (did you mean `fix-git`?)"
            ),
            "{text}"
        );
        assert!(!text.contains("regex-*"), "{text}");
        assert!(
            text.contains("lists all 2 of the dataset's tasks"),
            "{text}"
        );
    }

    #[test]
    fn only_a_run_whose_every_trial_was_scored_exits_0() {
        assert!(run_exit(&[Status::Scored, Status::Scored], 0) == Exit::Ok);
        assert!(run_exit(&[Status::Scored, Status::Skipped], 0) == Exit::Failed);
        assert!(run_exit(&[Status::Unscored], 0) == Exit::Failed);
        assert!(run_exit(&[Status::Errored], 0) == Exit::Failed);
        assert!(run_exit(&[Status::Scored], 1) == Exit::Failed);
    }
}
