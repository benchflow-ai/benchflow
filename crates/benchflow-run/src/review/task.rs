//! `bench check --review`: a model reads a task and judges it against
//! named criteria.
//!
//! The model gets the task's files (instruction and config, image, verifier,
//! oracle, controls, the rest; numbered, capped, redacted), the facts the
//! checker read from the config, and the checker's own findings, and answers
//! with one verdict per criterion and a list of findings. Each finding names
//! a criterion, a severity, the file and line it is about, the text of that
//! line, the problem, and a fix. The line is checked: it must exist in a file
//! of the task, and when the quoted text is on another line the finding moves
//! there. A finding that cannot be placed is kept without a location, with a
//! note saying why.

use std::path::{Path, PathBuf};

use benchflow_task::diag::{Diagnostic, Severity};
use benchflow_task::package::ImageRef;
use benchflow_task::{CheckReport, TaskPackage};
use serde::Serialize;
use serde_json::Value;

use super::evidence::{Bundle, SentFile, Withheld, read_for_model, squash};
use super::{OPINION, ReviewError, Reviewer, Spend, now_iso, text_field};

/// One thing a task is judged on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Criterion {
    /// The id the model and `--json` use (`verifier-checks-success`).
    pub id: &'static str,
    /// The code findings carry (`M102`); `bench explain` documents it.
    pub code: &'static str,
    /// One line, for people.
    pub title: &'static str,
    /// What the model is told to judge.
    pub question: &'static str,
}

/// The criteria, in the order they are judged.
pub const CRITERIA: &[Criterion] = &[
    Criterion {
        id: "success-stated",
        code: "M101",
        title: "the instruction states what success is",
        question: "Does the instruction say what the agent must produce or change, where, and in what form, so that an agent could tell when it is done? Everything the verifier checks (file names, paths, formats, exact strings, interfaces) must be stated or clearly implied by the instruction, not only by an example.",
    },
    Criterion {
        id: "verifier-checks-success",
        code: "M102",
        title: "the verifier checks what the instruction asks",
        question: "Does the verifier check the outcome the instruction asks for, and not something easier, something already true before the agent starts, or something different? Could the verifier ever give a low reward? Does it check the important parts of what the instruction asks?",
    },
    Criterion {
        id: "oracle-passes",
        code: "M103",
        title: "the oracle would pass and the agent cannot see it",
        question: "Would the oracle (the reference solution) earn full reward from this verifier in this image? Does it work the answer out the way an agent would have to, rather than only writing a final answer? Is it kept out of the agent's view (not copied into the image, not under the agent's working folder)? Say `unclear` with no finding when the task has no oracle.",
    },
    Criterion {
        id: "reward-not-gameable",
        code: "M104",
        title: "the reward cannot be raised without doing the task",
        question: "Can an agent earn reward without doing the task: by writing the reward file, editing or replacing the tests or the programs and libraries the verifier runs (for example through a folder the agent can write that holds an interpreter or tool the verifier calls), reading answers or the verifier from the image, or reaching answers over the network? Remember the agent is an unprivileged user while it works and the verifier usually runs as root afterwards, in the same sandbox.",
    },
    Criterion {
        id: "environment-pinned",
        code: "M105",
        title: "the environment is pinned",
        question: "Is the environment reproducible: a base image with a fixed tag (better, a digest), and language packages the task installs (pip, npm, cargo) pinned to versions? System packages from apt need not be pinned. Do the verifier's own dependencies install reliably?",
    },
    Criterion {
        id: "time-limits-fit",
        code: "M106",
        title: "time limits fit the work",
        question: "Do the agent's and the verifier's time limits fit the work: long enough for a capable agent to do the task and for the verifier to finish (builds, installs, test suites), and not so long that a stuck run wastes hours?",
    },
    Criterion {
        id: "controls-meaningful",
        code: "M107",
        title: "negative controls would fail",
        question: "If the task has negative controls (scripts that run in place of the agent and must not pass), would each one actually earn a low reward from this verifier, and does each test a plausible wrong answer? Say `unclear` with no finding when the task has none.",
    },
];

/// The code for a finding whose criterion is not one of [`CRITERIA`].
pub const OTHER: Criterion = Criterion {
    id: "other",
    code: "M100",
    title: "another problem the model found",
    question: "",
};

pub fn criterion(id: &str) -> Option<&'static Criterion> {
    CRITERIA.iter().find(|c| c.id == id)
}

/// The model's verdict on one criterion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Verdict {
    pub criterion: String,
    /// `pass`, `fail`, or `unclear`.
    pub verdict: String,
    pub reason: String,
}

/// One problem the model found, with its location checked.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Finding {
    pub criterion: String,
    pub code: &'static str,
    pub severity: Severity,
    pub problem: String,
    pub fix: String,
    /// Relative to the task folder, when the model named a file of the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// A line that exists in `file` (1-based).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// The text the model says is on that line.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub quote: String,
    /// What checking the location found, when it was not simply right.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location_note: Option<String>,
}

/// A model's review of one task.
#[derive(Debug, Clone, Serialize)]
pub struct TaskReview {
    pub task: PathBuf,
    pub model: String,
    pub reviewed_at: String,
    pub opinion: &'static str,
    pub verdicts: Vec<Verdict>,
    pub findings: Vec<Finding>,
    pub files_sent: Vec<String>,
    pub files_not_sent: Vec<Withheld>,
    pub spend: Spend,
}

/// The file the review is saved to, with `--save`.
pub const FILE_NAME: &str = "review.json";

/// Folders never sent: version control, caches, and run output.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "jobs",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
];

/// The files of the task at `root`, ordered by what they are for, ready to
/// send.
pub fn collect(
    root: &Path,
    package: Option<&TaskPackage>,
    redactor: &benchflow_agent::trajectory::Redactor,
) -> Bundle {
    let mut files: Vec<(u8, &'static str, PathBuf, String)> = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .min_depth(1)
        .max_depth(8)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| {
            !(e.file_type().is_dir()
                && SKIP_DIRS.contains(&e.file_name().to_string_lossy().as_ref()))
        })
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.into_path();
        let rel = relative(root, &path);
        if rel == FILE_NAME {
            continue;
        }
        let (rank, role) = role_of(root, &path, &rel, package);
        files.push((rank, role, path, rel));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.3.cmp(&b.3)));
    let mut bundle = Bundle::default();
    for (_, role, path, rel) in files {
        match read_for_model(&path, redactor) {
            Ok(lines) => bundle.push(SentFile::new(rel, role, lines)),
            Err(reason) => bundle.withhold(rel, reason),
        }
    }
    bundle
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/")
}

/// The rank (send order) and role of a task file.
fn role_of(
    root: &Path,
    path: &Path,
    rel: &str,
    package: Option<&TaskPackage>,
) -> (u8, &'static str) {
    let under = |dir: &Path| {
        let dir = if dir.is_absolute() {
            dir.to_path_buf()
        } else {
            root.join(dir)
        };
        path.starts_with(&dir)
    };
    let name = rel.rsplit('/').next().unwrap_or(rel);
    let top = rel.split('/').next().unwrap_or(rel);
    if !rel.contains('/')
        && matches!(
            name,
            "task.md" | "task.toml" | "instruction.md" | "TASK.md" | "task.yaml" | "task.yml"
        )
    {
        return (0, "task config and instruction");
    }
    if let Some(p) = package {
        if p.controls.iter().any(|c| under(&c.script)) {
            return (4, "negative control");
        }
        if under(&p.verifier.dir) {
            return (2, "verifier");
        }
        if let Some(o) = &p.oracle
            && under(&o.dir)
        {
            return (3, "oracle");
        }
        if let ImageRef::Build {
            context,
            dockerfile,
        } = &p.sandbox.image
            && (under(dockerfile) || under(context))
        {
            return (1, "sandbox image");
        }
    }
    match top {
        "verifier" | "tests" => (2, "verifier"),
        "oracle" | "solution" => (3, "oracle"),
        "controls" => (4, "negative control"),
        "sandbox" | "environment" => (1, "sandbox image"),
        _ if name == "Dockerfile" || name.ends_with(".dockerfile") => (1, "sandbox image"),
        _ => (5, "other"),
    }
}

/// What the checker read from the config, for the model.
fn facts(package: Option<&TaskPackage>) -> String {
    let Some(p) = package else {
        return "The checker could not read the task's config far enough to list its settings; read them from the files.\n".into();
    };
    let secs = |d: Option<std::time::Duration>| {
        d.map(|d| format!("{}s", d.as_secs()))
            .unwrap_or_else(|| "not set (the runner's default)".into())
    };
    let mut out = String::new();
    out.push_str(&format!("- format: {}\n", p.format.label()));
    out.push_str(&format!("- name: {}\n", p.name));
    match &p.sandbox.image {
        ImageRef::Pull { reference } => out.push_str(&format!("- image: {reference} (pulled)\n")),
        ImageRef::Build { dockerfile, .. } => out.push_str(&format!(
            "- image: built from {}\n",
            relative(&p.root, dockerfile)
        )),
    }
    out.push_str(&format!("- working folder: {}\n", p.workdir()));
    out.push_str(&format!(
        "- agent: runs as user `{}`, time limit {}, network {:?}\n",
        p.agent.user.as_deref().unwrap_or("agent (unprivileged)"),
        secs(p.agent.timeout),
        p.agent_network()
    ));
    out.push_str(&format!(
        "- verifier: {} mounted at {} only after the agent finishes, entry `{}`, runs as {}, time limit {}, network {:?}{}\n",
        relative(&p.root, &p.verifier.dir),
        p.verifier.mount,
        p.verifier.entry,
        p.verifier.user.as_deref().unwrap_or("root"),
        secs(p.verifier.timeout),
        p.verifier_network(),
        if p.verifier.rubric.is_some() {
            ", reward computed from the task's rubric"
        } else {
            ""
        }
    ));
    match &p.oracle {
        Some(o) => out.push_str(&format!(
            "- oracle: {} mounted at {} for reference runs only, entry `{}`\n",
            relative(&p.root, &o.dir),
            o.mount,
            o.entry
        )),
        None => out.push_str("- oracle: none\n"),
    }
    if p.controls.is_empty() {
        out.push_str("- negative controls: none\n");
    }
    for c in &p.controls {
        out.push_str(&format!(
            "- negative control `{}`: {}{}\n",
            c.id,
            relative(&p.root, &c.script),
            c.max_reward
                .map(|m| format!(", may earn at most {m}"))
                .unwrap_or_default()
        ));
    }
    if !p.stages.is_empty() {
        out.push_str(&format!("- later stages: {}\n", p.stages.len()));
    }
    out
}

/// The checker's findings, so the model does not repeat them.
fn checker_findings(report: &CheckReport) -> String {
    if report.diagnostics.is_empty() {
        return "none\n".into();
    }
    let mut out = String::new();
    for d in &report.diagnostics {
        let at = d
            .location
            .as_ref()
            .map(|l| {
                format!(
                    " ({}{})",
                    relative(&report.root, &l.file),
                    l.line.map(|n| format!(":{n}")).unwrap_or_default()
                )
            })
            .unwrap_or_default();
        out.push_str(&format!(
            "- {} {:?}: {}{at}\n",
            d.code, d.severity, d.message
        ));
    }
    out
}

/// The system prompt: how BenchFlow runs a task, the criteria, the answer's
/// shape.
pub fn system_prompt() -> String {
    let mut criteria = String::new();
    for c in CRITERIA {
        criteria.push_str(&format!("- `{}`: {}\n", c.id, c.question));
    }
    format!(
        "You review benchmark tasks for quality. A task gives an AI agent an instruction and a sandbox (a container built from the task's image). After the agent finishes, a verifier program runs in the same sandbox and writes a reward (1 means solved, 0 means not). The oracle is a reference solution that proves the task can be solved. Negative controls are scripts that run in place of the agent and must not earn reward.

How a run goes: the sandbox starts from the task's image. The agent works in it as an unprivileged user unless the task names another. The verifier and oracle folders are not in the sandbox while the agent works; they are copied in when needed, at the paths listed in the facts. The verifier then runs, as root unless the task says otherwise, and writes /logs/verifier/reward.txt or reward.json. Anything the image itself leaves writable stays writable to the agent. Judge the task as written; do not assume a runner will repair it.

Judge the task against each criterion:
{criteria}
Rules:
- Report a finding only for a real problem you can point to in the files shown. Do not report style, wording you would prefer, or a problem the checker already reported.
- Point at the line that shows the problem: `file` is the path shown in the file's header, `line` is the number left of `|`, and `quote` is that line's text copied exactly (without the number). Use null for `line` and \"\" for `quote` only when the problem is something missing from the task.
- `severity`: \"error\" when the task will give wrong rewards (passes without the work, fails correct work, cannot be solved); \"warning\" when it is likely to give wrong rewards or results that cannot be reproduced; \"note\" for a smaller problem.
- `problem` is one sentence: what is wrong and what it does to the reward. `fix` is one sentence: what to change.

Answer with one JSON object and nothing else:
{{\"verdicts\": [{{\"criterion\": \"<id>\", \"verdict\": \"pass\" | \"fail\" | \"unclear\", \"reason\": \"<one sentence>\"}}], \"findings\": [{{\"criterion\": \"<id>\", \"severity\": \"error\" | \"warning\" | \"note\", \"file\": \"<path>\", \"line\": <number or null>, \"quote\": \"<text of that line>\", \"problem\": \"<one sentence>\", \"fix\": \"<one sentence>\"}}]}}
Give one verdict for every criterion, in the order listed. A criterion that fails needs at least one finding."
    )
}

/// The user message: facts, the checker's findings, the files.
pub fn user_prompt(report: &CheckReport, bundle: &Bundle) -> String {
    format!(
        "Facts the checker read from the task's config:\n{}\nWhat the checker already reported:\n{}\nThe task's files, with line numbers:\n\n{}",
        facts(report.package.as_ref()),
        checker_findings(report),
        bundle.render()
    )
}

/// Review the task `report` describes.
pub async fn review(reviewer: &Reviewer, report: &CheckReport) -> Result<TaskReview, ReviewError> {
    let root = &report.root;
    let bundle = collect(root, report.package.as_ref(), reviewer.redactor());
    let (answer, spend) = reviewer
        .ask_json(
            &system_prompt(),
            &user_prompt(report, &bundle),
            16_000,
            &["verdicts", "findings"],
        )
        .await
        .map_err(|(e, _)| e)?;
    let (verdicts, findings) = parse(&answer, root, &bundle);
    Ok(TaskReview {
        task: root.clone(),
        model: reviewer.model.clone(),
        reviewed_at: now_iso(),
        opinion: OPINION,
        verdicts,
        findings,
        files_sent: bundle.sent.iter().map(|f| f.rel.clone()).collect(),
        files_not_sent: bundle.withheld.clone(),
        spend,
    })
}

/// Review every task, a few at a time, in report order. Folders that are not
/// tasks get an error rather than a review.
pub async fn review_all(
    reviewer: &Reviewer,
    reports: &[CheckReport],
) -> Vec<Result<TaskReview, ReviewError>> {
    use futures::StreamExt;
    futures::stream::iter(reports)
        .map(|r| async move {
            if r.format.is_none() {
                return Err(ReviewError {
                    kind: super::ReviewErrorKind::Model,
                    message: format!("{} is not a task", r.root.display()),
                    next: "a task is a folder with task.md (or task.toml)".into(),
                });
            }
            review(reviewer, r).await
        })
        .buffered(4)
        .collect()
        .await
}

/// Read the model's answer into verdicts and checked findings. Missing or
/// odd fields become defaults; nothing here fails.
pub fn parse(answer: &Value, root: &Path, bundle: &Bundle) -> (Vec<Verdict>, Vec<Finding>) {
    let mut verdicts: Vec<Verdict> = Vec::new();
    for v in answer
        .get("verdicts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = text_field(v, "criterion");
        let Some(c) = criterion(&id) else { continue };
        if verdicts.iter().any(|x| x.criterion == c.id) {
            continue;
        }
        let verdict = match text_field(v, "verdict").to_ascii_lowercase().as_str() {
            "pass" => "pass",
            "fail" => "fail",
            _ => "unclear",
        };
        verdicts.push(Verdict {
            criterion: c.id.to_string(),
            verdict: verdict.to_string(),
            reason: text_field(v, "reason"),
        });
    }
    // In criterion order, with the ones the model skipped as unclear.
    let verdicts = CRITERIA
        .iter()
        .map(|c| {
            verdicts
                .iter()
                .find(|v| v.criterion == c.id)
                .cloned()
                .unwrap_or_else(|| Verdict {
                    criterion: c.id.to_string(),
                    verdict: "unclear".into(),
                    reason: "the model gave no verdict".into(),
                })
        })
        .collect();
    let findings = answer
        .get("findings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|f| finding(f, root, bundle))
        .collect();
    (verdicts, findings)
}

fn finding(f: &Value, root: &Path, bundle: &Bundle) -> Option<Finding> {
    let problem = text_field(f, "problem");
    if problem.is_empty() {
        return None;
    }
    let c = criterion(&text_field(f, "criterion")).unwrap_or(&OTHER);
    let severity = match text_field(f, "severity").to_ascii_lowercase().as_str() {
        "error" => Severity::Error,
        "note" | "info" => Severity::Note,
        _ => Severity::Warning,
    };
    let quote = text_field(f, "quote");
    let line = match f.get("line") {
        Some(Value::Number(n)) => n.as_u64().map(|n| n as usize),
        Some(Value::String(s)) => s.trim().parse().ok(),
        _ => None,
    };
    let (file, line, location_note) = place(&text_field(f, "file"), line, &quote, root, bundle);
    Some(Finding {
        criterion: c.id.to_string(),
        code: c.code,
        severity,
        problem,
        fix: text_field(f, "fix"),
        file,
        line,
        quote,
        location_note,
    })
}

/// Check where a finding points: the file must be one of the task's, the
/// line must exist, and the quoted text should be on it. Returns the file,
/// the line, and a note when something did not hold.
fn place(
    file: &str,
    line: Option<usize>,
    quote: &str,
    root: &Path,
    bundle: &Bundle,
) -> (Option<String>, Option<usize>, Option<String>) {
    let file = file.trim();
    if file.is_empty() {
        return (None, None, None);
    }
    let mut rel = file.trim_start_matches("./").to_string();
    if let Ok(stripped) = Path::new(file).strip_prefix(root) {
        rel = relative(Path::new(""), stripped);
    }
    let Some(sent) = bundle.find(&rel) else {
        let note = if bundle.withheld.iter().any(|w| w.file == rel) {
            format!("the model cited {file}, which was not sent to it")
        } else {
            format!("the model cited {file}, which is not a file of this task")
        };
        return (None, None, Some(note));
    };
    let lines = &sent.lines;
    let quoted = squash(quote);
    let holds = |n: usize| {
        !quoted.is_empty()
            && lines.get(n - 1).is_some_and(|l| {
                squash(l).contains(&quoted)
                    || (quoted.len() > 3 && quoted.contains(&squash(l)) && !squash(l).is_empty())
            })
    };
    let elsewhere = || (1..=lines.len()).filter(|n| holds(*n)).collect::<Vec<_>>();
    match line {
        Some(n) if (1..=lines.len()).contains(&n) => {
            if quoted.is_empty() || holds(n) {
                return (Some(rel), Some(n), None);
            }
            let found = elsewhere();
            if let Some(m) = found.iter().min_by_key(|m| m.abs_diff(n)) {
                return (
                    Some(rel),
                    Some(*m),
                    Some(format!(
                        "the model cited line {n}; the text it quoted is on line {m}"
                    )),
                );
            }
            (
                Some(rel),
                Some(n),
                Some(format!(
                    "the text the model quoted is not on line {n}, nor anywhere in the file"
                )),
            )
        }
        Some(n) => {
            let found = elsewhere();
            if let Some(m) = found.first() {
                return (
                    Some(rel),
                    Some(*m),
                    Some(format!(
                        "the model cited line {n}, which does not exist; the text it quoted is on line {m}"
                    )),
                );
            }
            (
                Some(rel.clone()),
                None,
                Some(format!(
                    "the model cited {rel}:{n}, which does not exist (the file has {} line{})",
                    lines.len(),
                    if lines.len() == 1 { "" } else { "s" }
                )),
            )
        }
        None => match elsewhere().first() {
            Some(m) => (Some(rel), Some(*m), None),
            None => (Some(rel), None, None),
        },
    }
}

impl Finding {
    /// The finding in `bench check`'s diagnostic form, marked as the
    /// model's opinion. `model` names who said it.
    pub fn diagnostic(&self, root: &Path, model: &str) -> Diagnostic {
        let c = criterion(&self.criterion).unwrap_or(&OTHER);
        let mut d = Diagnostic::new(self.code, self.severity, self.problem.clone());
        if let Some(file) = &self.file {
            let path = root.join(file);
            let span = self.line.and_then(|n| line_span(&path, n));
            d = d.at(path, self.line, span);
            if span.is_some() {
                d = d.label("the model points here");
            }
        }
        d = d.note_line(format!("criterion: {} ({})", c.title, c.id));
        if let Some(note) = &self.location_note {
            d = d.note_line(note.clone());
        }
        d = d.note_line(format!(
            "this is {model}'s opinion; BenchFlow checked the location, not the claim"
        ));
        if !self.fix.is_empty() {
            d = d.help(format!("the model suggests: {}", self.fix));
        }
        d
    }
}

/// The byte span of line `n` (1-based) in the file at `path`, without its
/// line break and leading indentation.
fn line_span(path: &Path, n: usize) -> Option<(usize, usize)> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut start = 0;
    for (i, line) in text.split_inclusive('\n').enumerate() {
        if i + 1 == n {
            let body = line.trim_end_matches(['\n', '\r']);
            let indent = body.len() - body.trim_start().len();
            let end = start + body.len();
            let from = (start + indent).min(end);
            return Some((from, end.max(from)));
        }
        start += line.len();
    }
    None
}

#[cfg(test)]
mod tests {
    use benchflow_agent::trajectory::Redactor;
    use serde_json::json;

    use super::*;

    fn task() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("task.md"), "Make /app/x say ok.\n").unwrap();
        std::fs::create_dir_all(dir.path().join("verifier")).unwrap();
        std::fs::write(
            dir.path().join("verifier/test.sh"),
            "#!/bin/bash\nif [ -d /app ]; then echo 1; fi > /logs/verifier/reward.txt\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn findings_point_only_at_lines_that_exist() {
        let dir = task();
        let bundle = collect(dir.path(), None, &Redactor::default());
        let answer = json!({"findings": [
            {"criterion": "verifier-checks-success", "severity": "error", "file": "verifier/test.sh",
             "line": 2, "quote": "if [ -d /app ]; then echo 1; fi > /logs/verifier/reward.txt",
             "problem": "passes whenever /app exists", "fix": "check the file's content"},
            {"criterion": "verifier-checks-success", "severity": "error", "file": "verifier/test.sh",
             "line": 99, "quote": "", "problem": "a line that is not there", "fix": ""},
            {"criterion": "success-stated", "severity": "warning", "file": "verifier/test.sh",
             "line": 1, "quote": "if [ -d /app ]", "problem": "quoted text is on another line", "fix": ""},
            {"criterion": "made-up", "severity": "loud", "file": "nope.txt", "line": 1,
             "problem": "a file that does not exist", "fix": ""},
            {"criterion": "success-stated", "problem": ""}
        ]});
        let (verdicts, findings) = parse(&answer, dir.path(), &bundle);
        assert_eq!(verdicts.len(), CRITERIA.len());
        assert!(verdicts.iter().all(|v| v.verdict == "unclear"));
        assert_eq!(findings.len(), 4, "a finding without a problem is dropped");
        assert_eq!(findings[0].line, Some(2));
        assert_eq!(findings[0].location_note, None);
        assert_eq!(findings[1].line, None);
        assert!(
            findings[1]
                .location_note
                .as_ref()
                .unwrap()
                .contains("does not exist")
        );
        assert_eq!(findings[2].line, Some(2), "moved to where the quote is");
        assert!(
            findings[2]
                .location_note
                .as_ref()
                .unwrap()
                .contains("line 2")
        );
        assert_eq!(findings[3].code, "M100");
        assert_eq!(findings[3].severity, Severity::Warning);
        assert_eq!(findings[3].file, None);
        assert!(
            findings[3]
                .location_note
                .as_ref()
                .unwrap()
                .contains("not a file of this task")
        );

        let d = findings[0].diagnostic(dir.path(), "vercel/x/y");
        assert_eq!(d.code, "M102");
        let span = d.location.as_ref().unwrap().span.unwrap();
        let text = std::fs::read_to_string(dir.path().join("verifier/test.sh")).unwrap();
        assert!(text[span.0..span.1].starts_with("if [ -d /app ]"));
        assert!(d.notes.iter().any(|n| n.contains("opinion")));
    }

    #[test]
    fn files_go_in_role_order_with_numbers() {
        let dir = task();
        std::fs::write(
            dir.path().join(".env"),
            "AI_GATEWAY_API_KEY=vck_live_secret_123456\n",
        )
        .unwrap();
        let bundle = collect(dir.path(), None, &Redactor::default());
        let order: Vec<&str> = bundle.sent.iter().map(|f| f.rel.as_str()).collect();
        assert_eq!(order, ["task.md", "verifier/test.sh"]);
        assert!(bundle.withheld.iter().any(|w| w.file == ".env"));
        let text = bundle.render();
        assert!(text.contains("=== verifier/test.sh (verifier, 2 lines) ==="));
        assert!(text.contains("  2| if [ -d /app ]"));
        assert!(!text.contains("vck_live_secret"));
    }

    #[test]
    fn the_system_prompt_names_every_criterion_and_the_answer_shape() {
        let p = system_prompt();
        for c in CRITERIA {
            assert!(p.contains(c.id), "{}", c.id);
        }
        assert!(p.contains("\"verdicts\""));
        assert!(p.contains("\"findings\""));
    }
}
