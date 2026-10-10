//! Converting between formats: what `bench import` and `bench export` call.
//!
//! - [`import_to_taskmd`] turns a split-layout task (Harbor's `task.toml`
//!   folder) or a BenchFlow 0.8 task.md into a task.md package. It keeps
//!   everything: settings task.md models move to its names (`[environment]`
//!   is `[sandbox]`, `memory_mb = 4096` is `memory = "4 GB"`), the rest stay
//!   as written under `[import.harbor]` or `[import.v06]`, comments follow
//!   their settings, and `tests/`, `solution/`, `environment/` become
//!   `verifier/`, `oracle/`, `sandbox/`.
//! - A Terminal-Bench v1 task (`task.yaml`, `run-tests.sh`,
//!   `docker-compose.yaml`) is first converted to the split layout as
//!   Harbor's `task migrate` does, then imported like any split task.
//! - [`export_split`] writes a task.md package as a split-layout folder and
//!   reports what the split layout cannot hold. It refuses, writing
//!   nothing, when a setting that decides how the task runs would be lost.
//!
//! Both are ports of task.md's reference converter, `tools/convert.py`
//! (`import`, `import-v06`, `export-harbor`), and produce the same files.
//! Neither overwrites an existing output folder unless asked.

mod doc;
mod export;
mod files;
mod harbor;
mod import;
mod rewardkit;
mod tbench;
mod tomltext;
mod v06;
mod value;

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

pub use harbor::{Loss, LossSeverity};
pub use tbench::{Resources, discover_tbench, is_tbench};

use crate::package::Format;

/// How [`import_to_taskmd_with`] runs.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Replace an existing, non-empty output folder.
    pub overwrite: bool,
    /// The task comes from a BenchFlow registry, whose runtime installs a
    /// bundled `environment/skills/` folder by convention: declare it as
    /// `[sandbox] skills = "sandbox/skills"`.
    pub benchflow_skills: bool,
    /// Resources set on Terminal-Bench v1 tasks, which declare none.
    pub tbench_resources: Resources,
}

/// How [`export_split_with`] runs.
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Replace an existing, non-empty output folder.
    pub overwrite: bool,
    /// Export even when the split layout drops a setting that decides how
    /// the task runs.
    pub allow_degraded: bool,
}

/// How much a note matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NoteLevel {
    /// What the conversion did.
    Info,
    /// Something the converted task will not do unless its author acts.
    Warning,
}

/// One thing a conversion did or found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Note {
    pub level: NoteLevel,
    pub text: String,
}

/// What an import did.
#[derive(Debug, Clone, Serialize)]
pub struct ConvertReport {
    /// The format the source was in.
    pub from: Format,
    pub source: PathBuf,
    pub output: PathBuf,
    /// Set when the source was in a format converted to the split layout
    /// first: `terminal-bench-v1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub converted_from: Option<&'static str>,
    pub notes: Vec<Note>,
}

/// Whether an export kept everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportStatus {
    /// The split layout holds everything.
    Lossless,
    /// Exported without some settings (see the losses).
    Degraded,
    /// Not exported: a setting that decides how the task runs would be lost.
    Refused,
}

/// A file export wrote in the converted form instead of the original bytes
/// import kept.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Rewritten {
    pub path: String,
    pub reason: String,
}

/// What a RewardKit conversion recorded at import came to on export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RewardKitExport {
    pub undone: bool,
    pub why: Option<String>,
    pub restored: Vec<String>,
    pub variables: Vec<String>,
}

/// The export report, written as `export-report.json` in the output folder
/// (the same shape as task.md's `tools/convert.py export-harbor` writes).
#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub report: &'static str,
    pub from: &'static str,
    pub to: &'static str,
    pub tool: &'static str,
    pub status: ExportStatus,
    pub allow_degraded: bool,
    pub source_task_dir: String,
    pub output_dir: String,
    pub summary: String,
    /// `sha256:<hex>` of the source task.md.
    pub input_hashes: indexmap::IndexMap<String, String>,
    /// `[import.harbor]` keys merged back into task.toml.
    pub restored_extension_paths: Vec<String>,
    /// Files written with the original bytes import kept.
    pub restored_originals: Vec<String>,
    pub rewritten: Vec<Rewritten>,
    pub losses: Vec<Loss>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewardkit: Option<RewardKitExport>,
    /// Every file written, by path in the output folder.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_files: Option<Vec<String>>,
    /// `sha256:<hex>` of the files export generated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_hashes: Option<indexmap::IndexMap<String, String>>,
}

/// Why a conversion did not happen. Nothing was written unless the error
/// says otherwise.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("{0} is not a task folder: it has no task.md or task.toml")]
    NotATask(PathBuf),
    #[error("{0} is already a task.md package; there is nothing to import")]
    AlreadyTaskMd(PathBuf),
    #[error(
        "{0} is not a task.md package; export reads task.md (convert other formats with import first)"
    )]
    NotTaskMd(PathBuf),
    #[error("{0} already exists and is not empty; choose another folder or allow overwriting")]
    OutputExists(PathBuf),
    #[error("the output folder {output} would overlap the source {input}")]
    OutputOverlaps { input: PathBuf, output: PathBuf },
    #[error("cannot import: {0}")]
    Instruction(String),
    #[error("{0}")]
    Collision(String),
    #[error("{file}: {message}")]
    Invalid { file: PathBuf, message: String },
    #[error("the split layout cannot hold what decides how this task runs: {}", .0.losses.iter().filter(|l| l.severity == LossSeverity::RunAffecting).map(|l| format!("{} ({})", l.path, l.reason)).collect::<Vec<_>>().join("; "))]
    Refused(Box<ExportReport>),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub(crate) fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> ConvertError + '_ {
    move |source| ConvertError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Convert a split-layout task, a BenchFlow 0.8 task.md, or a
/// Terminal-Bench v1 task at `src` into a
/// task.md package at `out`. `out` must not exist or be empty, unless
/// `overwrite` allows replacing what is there.
pub fn import_to_taskmd(
    src: &Path,
    out: &Path,
    overwrite: bool,
) -> Result<ConvertReport, ConvertError> {
    let opts = ImportOptions {
        overwrite,
        ..ImportOptions::default()
    };
    import_to_taskmd_with(src, out, &opts)
}

/// [`import_to_taskmd`] with options.
pub fn import_to_taskmd_with(
    src: &Path,
    out: &Path,
    opts: &ImportOptions,
) -> Result<ConvertReport, ConvertError> {
    if tbench::is_tbench(src) {
        return import_tbench(src, out, opts);
    }
    let from = crate::load::detect(src).ok_or_else(|| ConvertError::NotATask(src.to_path_buf()))?;
    if from == Format::TaskMd {
        return Err(ConvertError::AlreadyTaskMd(src.to_path_buf()));
    }
    let plan = match from {
        Format::Split => import::plan(src, opts.benchflow_skills)?,
        Format::TaskMdV06 => v06::plan(src)?,
        Format::TaskMd => unreachable!("refused above"),
    };
    prepare_output(src, out, opts.overwrite)?;
    write_plan(plan, src, src, out, from, None, Vec::new())
}

/// A Terminal-Bench v1 task: to the split layout in a scratch folder, then
/// imported from there.
fn import_tbench(
    src: &Path,
    out: &Path,
    opts: &ImportOptions,
) -> Result<ConvertReport, ConvertError> {
    let staging = tempfile::tempdir().map_err(io_err(Path::new("a scratch folder")))?;
    let split = staging.path().join("task");
    let notes = tbench::map(src, &split, &opts.tbench_resources)?;
    let plan = import::plan(&split, opts.benchflow_skills)?;
    prepare_output(src, out, opts.overwrite)?;
    write_plan(
        plan,
        &split,
        src,
        out,
        Format::Split,
        Some("terminal-bench-v1"),
        notes,
    )
}

/// Write `plan` (read from `read_from`) into `out` and report it.
fn write_plan(
    plan: ImportPlan,
    read_from: &Path,
    src: &Path,
    out: &Path,
    from: Format,
    converted_from: Option<&'static str>,
    mut notes: Vec<String>,
) -> Result<ConvertReport, ConvertError> {
    let written = match plan.write(read_from, out) {
        Ok(n) => n,
        Err(e) => {
            let _ = cleanup(out);
            return Err(e);
        }
    };
    notes.extend(written);
    Ok(ConvertReport {
        from,
        source: src.to_path_buf(),
        output: out.to_path_buf(),
        converted_from,
        notes: notes
            .into_iter()
            .map(|text| match text.strip_prefix("warning: ") {
                Some(rest) => Note {
                    level: NoteLevel::Warning,
                    text: rest.to_string(),
                },
                None => Note {
                    level: NoteLevel::Info,
                    text,
                },
            })
            .collect(),
    })
}

/// Write the task.md package at `task` as a split-layout folder at `out`
/// (task.toml, instruction.md, environment/, tests/, solution/), with
/// `export-report.json` beside them. Refuses, writing nothing, when the
/// split layout would drop a setting that decides how the task runs
/// ([`export_split_with`] can allow that). `out` must not exist or be
/// empty, unless `overwrite` allows replacing what is there.
pub fn export_split(
    task: &Path,
    out: &Path,
    overwrite: bool,
) -> Result<ExportReport, ConvertError> {
    let opts = ExportOptions {
        overwrite,
        ..ExportOptions::default()
    };
    export_split_with(task, out, &opts)
}

/// [`export_split`] with options.
pub fn export_split_with(
    task: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport, ConvertError> {
    if crate::load::detect(task) != Some(Format::TaskMd) {
        return Err(ConvertError::NotTaskMd(task.to_path_buf()));
    }
    let plan = export::plan(task, out, opts.allow_degraded)?;
    if plan.report.status == ExportStatus::Refused {
        return Err(ConvertError::Refused(Box::new(plan.report)));
    }
    prepare_output(task, out, opts.overwrite)?;
    match export::write(task, out, plan) {
        Ok(report) => Ok(report),
        Err(e) => {
            let _ = cleanup(out);
            Err(e)
        }
    }
}

/// Make sure `out` is a fresh folder that does not overlap `src`.
fn prepare_output(src: &Path, out: &Path, overwrite: bool) -> Result<(), ConvertError> {
    let canon = |p: &Path| fs::canonicalize(p).unwrap_or_else(|_| absolute(p));
    let (s, o) = (canon(src), canon(out));
    if o.starts_with(&s) || s.starts_with(&o) {
        return Err(ConvertError::OutputOverlaps {
            input: src.to_path_buf(),
            output: out.to_path_buf(),
        });
    }
    if out.exists() {
        let empty = fs::read_dir(out).map_err(io_err(out))?.next().is_none();
        if !empty {
            if !overwrite {
                return Err(ConvertError::OutputExists(out.to_path_buf()));
            }
            fs::remove_dir_all(out).map_err(io_err(out))?;
        }
    }
    fs::create_dir_all(out).map_err(io_err(out))
}

fn absolute(p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|d| d.join(p))
            .unwrap_or_else(|_| p.to_path_buf())
    }
}

/// Remove a half-written output folder.
fn cleanup(out: &Path) -> std::io::Result<()> {
    if out.exists() {
        fs::remove_dir_all(out)
    } else {
        Ok(())
    }
}

/// Everything an import writes, decided before anything is written.
pub(crate) struct ImportPlan {
    /// The text of task.md.
    pub task_md: String,
    /// Files copied as they are: (path in the package, path in the source).
    pub copies: Vec<(files::Rel, files::Rel)>,
    /// Files written with these bytes, by path in the package.
    pub writes: Vec<(String, Vec<u8>)>,
    pub notes: Vec<String>,
}

impl ImportPlan {
    /// Write the plan into `out`, which exists and is empty.
    fn write(self, src: &Path, out: &Path) -> Result<Vec<String>, ConvertError> {
        let md = out.join("task.md");
        fs::write(&md, self.task_md.as_bytes()).map_err(io_err(&md))?;
        for (to, from) in &self.copies {
            let (to, from) = (files::under(out, to), files::under(src, from));
            files::copy(&from, &to).map_err(io_err(&from))?;
        }
        for (rel, data) in &self.writes {
            let to = files::under(out, &files::rel(rel));
            files::write(&to, data).map_err(io_err(&to))?;
        }
        Ok(self.notes)
    }
}
