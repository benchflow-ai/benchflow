//! `bench import` and `bench export`: between task.md and the split layout
//! (import also reads 0.8's task.md and Terminal-Bench v1), and a job's
//! traces as a Hugging Face dataset or OpenTelemetry.

use std::path::{Path, PathBuf};

use benchflow_task::convert::{ConvertError, ExportOptions, ImportOptions, NoteLevel, Resources};
use serde_json::json;

use benchflow_run::traces;

use crate::cli::{EpisodesArg, ExportArgs, ExportFormat, FilterArg, ImportArgs};
use crate::error::{CliError, Exit, Result};
use crate::say;
use crate::ui::{self, dim, fail_mark, ok_mark, plural, ui, warn_mark};

pub fn import(args: ImportArgs) -> Result<Exit> {
    let mut sources = benchflow_task::discover(&args.source);
    sources.extend(benchflow_task::convert::discover_tbench(&args.source));
    sources.sort();
    if sources.is_empty() {
        return Err(
            CliError::usage(format!("no task in {}", args.source.display())).next(
                "pass a folder with task.toml (split layout), a 0.8 task.md, or a Terminal-Bench v1 task.yaml",
            ),
        );
    }
    let opts = ImportOptions {
        overwrite: args.overwrite,
        tbench_resources: Resources {
            cpus: args.cpus,
            memory_mb: args.memory_mb,
            storage_mb: args.storage_mb,
            gpus: args.gpus,
        },
        ..ImportOptions::default()
    };
    let many = sources.len() > 1;
    let mut failed = 0;
    let mut results = Vec::new();
    for src in &sources {
        if benchflow_task::detect(src) == Some(benchflow_task::Format::TaskMd) {
            say!("  {} {} is already task.md", dim("·"), ui::path(src));
            continue;
        }
        let out = match (&args.out, many) {
            (Some(o), true) => o.join(name_of(src)),
            (Some(o), false) => o.clone(),
            (None, _) => sibling(src, "taskmd"),
        };
        match benchflow_task::convert::import_to_taskmd_with(src, &out, &opts) {
            Ok(rep) => {
                say!(
                    "  {} {} → {}",
                    ok_mark(),
                    ui::path(src),
                    ui::path(&rep.output)
                );
                for n in &rep.notes {
                    match n.level {
                        NoteLevel::Warning => say!("      {} {}", warn_mark(), n.text),
                        NoteLevel::Info => say!("      {}", dim(&n.text)),
                    }
                }
                results.push(json!({"source": src, "out": rep.output, "notes": rep.notes}));
            }
            Err(e) => {
                failed += 1;
                say!("  {} {}: {e}", fail_mark(), ui::path(src));
                results.push(json!({"source": src, "error": e.to_string()}));
            }
        }
    }
    if ui().json {
        println!("{}", serde_json::Value::Array(results));
    } else if failed == 0 && !results.is_empty() {
        say!(
            "{}",
            dim(format!(
                "next: bench check {}",
                ui::path(args.out.as_deref().unwrap_or(Path::new(".")))
            ))
        );
    }
    Ok(if failed > 0 { Exit::Failed } else { Exit::Ok })
}

pub fn export(args: ExportArgs) -> Result<Exit> {
    let is_job = benchflow_task::detect(&args.task).is_none()
        && (benchflow_run::read::is_job(&args.task) || args.task.join("result.json").is_file());
    if is_job {
        return export_traces(args);
    }
    let job_only = [
        (args.format == Some(ExportFormat::Hf), "--format hf"),
        (args.format == Some(ExportFormat::Otel), "--format otel"),
        (args.sharegpt, "--sharegpt"),
        (args.episodes != EpisodesArg::All, "--episodes"),
        (args.filter != FilterArg::All, "--filter"),
        (args.instruction, "--instruction"),
        (args.verifier_output, "--verifier-output"),
    ];
    if let Some((_, flag)) = job_only.iter().find(|(set, _)| *set) {
        return Err(CliError::usage(format!(
            "{flag} exports a job's traces, and {} is not a job folder",
            args.task.display()
        ))
        .next("pass a job folder (one with trial folders holding result.json), or drop the flag"));
    }
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| sibling(&args.task, "split"));
    let opts = ExportOptions {
        overwrite: args.overwrite,
        allow_degraded: args.allow_degraded,
    };
    match benchflow_task::convert::export_split_with(&args.task, &out, &opts) {
        Ok(rep) => {
            if ui().json {
                println!(
                    "{}",
                    json!({"task": args.task, "out": out, "status": rep.status, "losses": rep.losses})
                );
                return Ok(Exit::Ok);
            }
            say!(
                "  {} {} → {}",
                ok_mark(),
                ui::path(&args.task),
                ui::path(&out)
            );
            if rep.losses.is_empty() {
                say!("      {}", dim("nothing lost"));
            } else {
                say!(
                    "      {} {} the split layout cannot hold:",
                    warn_mark(),
                    plural(rep.losses.len(), "thing")
                );
                for l in &rep.losses {
                    say!("        {}", dim(format!("{}: {}", l.path, l.reason)));
                }
            }
            Ok(Exit::Ok)
        }
        Err(e @ ConvertError::Refused(_)) => {
            Err(CliError::failed(e.to_string())
                .next("pass --allow-degraded to export it without them"))
        }
        Err(e) => Err(CliError::failed(e.to_string())),
    }
}

/// `bench export <job>`: its traces as a Hugging Face dataset or OTel.
fn export_traces(args: ExportArgs) -> Result<Exit> {
    if args.allow_degraded {
        return Err(
            CliError::usage("--allow-degraded is for exporting a task, and this is a job")
                .next("drop --allow-degraded"),
        );
    }
    let format = match args.format {
        None | Some(ExportFormat::Hf) => ExportFormat::Hf,
        Some(ExportFormat::Otel) => ExportFormat::Otel,
        Some(ExportFormat::Split) => {
            return Err(CliError::usage(format!(
                "{} is a job folder; the split layout is for tasks",
                args.task.display()
            ))
            .next("use --format hf or --format otel for a job's traces"));
        }
    };
    if format == ExportFormat::Otel {
        let hf_only = [
            (args.sharegpt, "--sharegpt"),
            (args.episodes != EpisodesArg::All, "--episodes"),
            (args.instruction, "--instruction"),
            (args.verifier_output, "--verifier-output"),
        ];
        if let Some((_, flag)) = hf_only.iter().find(|(set, _)| *set) {
            return Err(CliError::usage(format!(
                "{flag} shapes dataset rows; --format otel writes spans"
            ))
            .next("drop the flag, or use --format hf"));
        }
    }
    let opts = traces::ExportOptions {
        episodes: match args.episodes {
            EpisodesArg::All => traces::Episodes::All,
            EpisodesArg::Last => traces::Episodes::Last,
        },
        filter: match args.filter {
            FilterArg::All => traces::Filter::All,
            FilterArg::Success => traces::Filter::Success,
            FilterArg::Failure => traces::Filter::Failure,
        },
        sharegpt: args.sharegpt,
        instruction: args.instruction,
        verifier_output: args.verifier_output,
        overwrite: args.overwrite,
    };
    let result = match format {
        ExportFormat::Otel => {
            let mut out = args
                .out
                .clone()
                .unwrap_or_else(|| sibling(&args.task, "otel.jsonl"));
            if out.extension().is_none_or(|e| e != "jsonl") {
                out.as_mut_os_string().push(".jsonl");
            }
            traces::export_otel(&args.task, &out, &opts)
        }
        _ => {
            let out = args
                .out
                .clone()
                .unwrap_or_else(|| sibling(&args.task, "hf"));
            traces::export_hf(&args.task, &out, &opts)
        }
    };
    let rep = match result {
        Ok(rep) => rep,
        Err(e @ traces::ExportError::OutputExists(_)) => {
            return Err(CliError::usage(e.to_string())
                .next("pass --overwrite to replace it, or name another output"));
        }
        Err(e @ traces::ExportError::Nothing { .. }) => {
            return Err(CliError::failed(e.to_string())
                .next("`bench results <job> --trials` shows what each trial did; trials that never started have no trajectory"));
        }
        Err(e) => return Err(CliError::failed(e.to_string())),
    };
    if ui().json {
        println!("{}", serde_json::to_string(&rep).unwrap_or_default());
        return Ok(Exit::Ok);
    }
    let exported = rep.trials.iter().filter(|t| t.written > 0).count();
    let what = if rep.format == "hf" {
        plural(rep.written, "row")
    } else {
        plural(rep.written, "trace")
    };
    say!(
        "  {} {} → {}  {}",
        ok_mark(),
        ui::path(&rep.job),
        ui::path(&rep.out),
        dim(format!(
            "{what} from {exported} of {}",
            plural(rep.trials.len(), "trial")
        ))
    );
    for t in rep.trials.iter().filter(|t| t.skipped.is_some()) {
        say!(
            "{}",
            ui::wrap(
                &format!("      {} {}: ", warn_mark(), t.trial),
                t.skipped.as_deref().unwrap_or(""),
                8,
                ui::wrap_width()
            )
        );
    }
    if rep.format == "hf" {
        // Relative to here when it can be; Python does not expand `~`.
        let shown = ui::path(&rep.out);
        let shown = if shown.starts_with("~/") {
            rep.out.display().to_string()
        } else {
            shown
        };
        let text = format!(
            "datasets.load_dataset(\"{shown}\") reads it; check the dataset card (README.md) before sharing"
        );
        say!("{}", dim(ui::wrap("next: ", &text, 6, ui::wrap_width())));
    }
    Ok(Exit::Ok)
}

fn name_of(p: &Path) -> String {
    p.canonicalize()
        .ok()
        .and_then(|c| c.file_name().map(|n| n.to_string_lossy().to_string()))
        .unwrap_or_else(|| "task".into())
}

fn sibling(p: &Path, suffix: &str) -> PathBuf {
    let canon = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let parent = canon.parent().map(Path::to_path_buf).unwrap_or_default();
    parent.join(format!("{}-{suffix}", name_of(p)))
}
