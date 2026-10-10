//! task.md draft 2: the instruction first, then typed fenced blocks, with
//! the config in a ` ```toml task ` block. BenchFlow's native format.
//!
//! A port of task.md's reference checker (`tools/taskmd.py`): the same
//! document rules, key vocabulary, checks, and messages, with each
//! diagnostic placed at a file, a line, and the byte span of the offending
//! key or value, a stable code, and a fix where one is safe.
//!
//! - [`parse`] splits task.md into the instruction, its blocks, and its
//!   config, reporting structure and syntax errors. It reads no files.
//! - [`check`] checks a package folder: the document, its config, its
//!   rubric, behaviors, and judges, and the files the package must hold. It
//!   also says what grading needs ([`CheckReport::facts`]) and, when nothing
//!   is wrong, builds the [`TaskPackage`](crate::package::TaskPackage) the run
//!   engine reads.
//! - [`agent_view`] is exactly what the agent receives first.
//!
//! [`CheckReport::facts`]: crate::check::CheckReport::facts

pub mod code;
mod config;
mod explain;
mod extras;
mod files;
mod grading;
mod json_spans;
mod judges;
mod judgment;
mod map;
mod pyfmt;
mod pyval;
mod report;
mod scan;
pub mod schema;
mod text;
mod vocab;

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::check::{CheckOptions, CheckReport};
use crate::diag::{Diagnostic, has_errors};
use crate::package::Format;
use crate::source::Source;

pub use explain::EXPLANATIONS;
pub use scan::{Block, BlockKind};

use config::ConfigDoc;
use report::{At, ConfigLoc, Sink};
use text::Text;

/// Where a document's config came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfigSource {
    /// A ` ```toml task ` (or ` ```yaml task `) block.
    Block,
    /// Legacy YAML frontmatter.
    Frontmatter,
    /// No config: the file is all instruction.
    None,
}

impl ConfigSource {
    /// The reference's name for it: `block`, `frontmatter`, or `none`.
    pub fn as_str(self) -> &'static str {
        match self {
            ConfigSource::Block => "block",
            ConfigSource::Frontmatter => "frontmatter",
            ConfigSource::None => "none",
        }
    }
}

/// A parsed task.md.
#[derive(Debug, Clone)]
pub struct TaskDocument {
    /// The file, as given to [`parse`].
    pub path: PathBuf,
    /// What the agent receives first: the text before the first block,
    /// without leading canary comments and blank lines. The agent's view is
    /// this and a final newline ([`TaskDocument::agent_view`]).
    pub instruction: String,
    /// The text before the first block as written, canary comments included.
    pub instruction_source: String,
    /// The leading canary comments, joined with "; ". Never shown to the
    /// agent.
    pub canary: Option<String>,
    /// The task.md blocks, in order.
    pub blocks: Vec<Block>,
    /// The config, with the position of every key and value; `None` when
    /// task.md has none.
    pub config: Option<Source>,
    /// Where the config came from.
    pub config_source: ConfigSource,
    /// Structure and syntax problems found while parsing.
    pub diagnostics: Vec<Diagnostic>,
    text: Text,
    meta: ConfigDoc,
}

static EMPTY_CONFIG: std::sync::LazyLock<Value> =
    std::sync::LazyLock::new(|| Value::Object(Default::default()));

impl TaskDocument {
    /// What the agent receives at the start of a run. Stage and role prompts
    /// come later.
    pub fn agent_view(&self) -> String {
        format!("{}\n", self.instruction)
    }

    /// The config as JSON; an empty table when there is none.
    pub fn config_value(&self) -> &Value {
        self.config.as_ref().map_or(&EMPTY_CONFIG, |s| &s.value)
    }

    /// The task's title for people: `title`, else `name`.
    pub fn title(&self) -> Option<&str> {
        let c = self.config_value();
        [c.get("title"), c.get("name")]
            .into_iter()
            .flatten()
            .find(|v| pyval::truthy(Some(v)))
            .and_then(Value::as_str)
    }

    /// Whether the file has blocks, or is all instruction (`blocks` or
    /// `prompt-only`, as the reference names it).
    pub fn mode(&self) -> &'static str {
        if self.blocks.is_empty() {
            "prompt-only"
        } else {
            "blocks"
        }
    }

    /// What the agent is told, or can learn by asking: the instruction, then
    /// each stage, role, and simulated-user prompt.
    pub fn told(&self) -> Vec<String> {
        let mut out = vec![self.instruction.clone()];
        out.extend(
            self.blocks
                .iter()
                .filter(|b| b.kind.is_prompt())
                .map(Block::prompt),
        );
        out
    }

    /// The stages' prompts, by stage name.
    pub(crate) fn stage_prompt(&self, name: &str) -> Option<String> {
        self.blocks
            .iter()
            .find(|b| b.kind == BlockKind::Stage && b.arg.as_deref() == Some(name))
            .map(Block::prompt)
    }

    /// Where config diagnostics point.
    fn config_loc<'a>(&'a self, empty: &'a Source) -> ConfigLoc<'a> {
        ConfigLoc {
            file: Path::new("task.md"),
            text: &self.text,
            source: self.config.as_ref().unwrap_or(empty),
            block_line: self.meta.block_line,
        }
    }
}

/// True when `text` (a task.md file) is draft 2 rather than 0.8's
/// frontmatter form: it does not open with `---`.
pub fn is_draft2(text: &str) -> bool {
    !text.starts_with("---\n") && !text.starts_with("---\r\n")
}

/// Parse task.md text: the instruction, the blocks, and the config, with
/// structure and syntax diagnostics. `path` labels the document; nothing is
/// read from disk.
pub fn parse(text: &str, path: &Path) -> TaskDocument {
    let normalized = Text::new(text);
    let empty = config::empty_source(0);
    let loc = ConfigLoc {
        file: Path::new("task.md"),
        text: &normalized,
        source: &empty,
        block_line: 1,
    };
    let mut sink = Sink::new(Some(&loc));
    let scanned = scan::scan(&normalized, &mut sink);

    let mut sources: Vec<(ConfigSource, Source, ConfigDoc, usize)> = Vec::new();
    for block in scanned
        .blocks
        .iter()
        .filter(|b| b.kind == BlockKind::Config)
    {
        let (source, meta) = ConfigDoc::parse(block, &mut sink, &normalized);
        sources.push((ConfigSource::Block, source, meta, block.line));
    }
    if let Some((front, offset)) = &scanned.frontmatter {
        let source = match Source::yaml(front, *offset) {
            Ok(s) if s.value.is_object() => s,
            Ok(_) => {
                sink.error(
                    code::CONFIG_SYNTAX,
                    At::Line(2),
                    "frontmatter must be a mapping",
                );
                config::empty_source(*offset)
            }
            Err(e) => {
                sink.error(
                    code::CONFIG_SYNTAX,
                    At::Line(2),
                    format!("frontmatter is not valid YAML: {}", e.message),
                );
                config::empty_source(*offset)
            }
        };
        let mut meta = ConfigDoc::none();
        meta.lang = config::Lang::Yaml;
        meta.block_line = 1;
        sources.push((ConfigSource::Frontmatter, source, meta, 1));
        sink.warning(
            code::FRONTMATTER,
            At::Line(1),
            "legacy YAML frontmatter: `task fmt` moves it into a ```toml task block",
        )
        .help = Some("move its settings into a ```toml task block after the instruction (`task fmt` is task-md's formatter; bench has none)".into());
    }
    if sources.len() > 1 {
        let names: Vec<&str> = sources.iter().map(|(s, ..)| s.as_str()).collect();
        sink.error(
            code::CONFIG_TWICE,
            At::Line(sources[1].3),
            format!("config found in more than one place: {}", names.join(", ")),
        )
        .help = Some(
            "keep one config (a single ```toml task block) and merge the others into it".into(),
        );
    }
    let diagnostics = sink.into_diagnostics();
    let (config_source, config, meta) = match sources.into_iter().next() {
        Some((kind, source, meta, _)) => (kind, Some(source), meta),
        None => (ConfigSource::None, None, ConfigDoc::none()),
    };
    TaskDocument {
        path: path.to_path_buf(),
        instruction: scanned.instruction,
        instruction_source: scanned.instruction_source,
        canary: scanned.canary,
        blocks: scanned.blocks,
        config,
        config_source,
        diagnostics,
        text: normalized,
        meta,
    }
}

/// What the agent receives first in the package at `dir`: the instruction,
/// with canary comments removed, and a final newline. `Err` holds the
/// problems when task.md cannot be read or its structure is broken.
pub fn agent_view(dir: &Path) -> Result<String, Vec<Diagnostic>> {
    let text = read_task_md(dir)?;
    let doc = parse(&text, Path::new("task.md"));
    if has_errors(&doc.diagnostics) {
        return Err(doc.diagnostics);
    }
    Ok(doc.agent_view())
}

/// What the checker finds in a rubric on its own: its `$schema`, keys,
/// scoring, and criteria, without what the agent is told or what the config
/// saves. `text` is the file as written, for positions; `file` names it.
pub(crate) fn rubric_errors(value: &Value, text: Option<&str>, file: &Path) -> Vec<Diagnostic> {
    let text = text.map_or_else(
        || serde_json::to_string_pretty(value).unwrap_or_default(),
        str::to_string,
    );
    if !judgment::has_schema(value, "rubric") {
        return vec![
            Diagnostic::error(
                code::NO_SCHEMA,
                format!(
                    "no task.md $schema ({}rubric-1.json), so this is not a task.md rubric",
                    vocab::SCHEMA_BASE
                ),
            )
            .in_file(file),
        ];
    }
    let rubric = judgment::Judgment {
        value: value.clone(),
        text,
        file: file.to_path_buf(),
    };
    judgment::check_rubric(&rubric, None, Path::new("."), None, false)
}

/// task.md's text, or why it cannot be read.
fn read_task_md(dir: &Path) -> Result<String, Vec<Diagnostic>> {
    let file = dir.join("task.md");
    let unreadable = |message: String| {
        vec![
            Diagnostic::error(code::UNREADABLE, message)
                .in_file("task.md")
                .help("a task.md package is a folder whose task.md holds the instruction and a ```toml task block"),
        ]
    };
    let bytes =
        std::fs::read(&file).map_err(|e| unreadable(format!("cannot read task.md: {e}")))?;
    String::from_utf8(bytes).map_err(|e| {
        let at = e.utf8_error().valid_up_to();
        unreadable(format!(
            "task.md is not UTF-8 text: the bytes at offset {at} are not valid UTF-8"
        ))
    })
}

/// Check a draft 2 package folder. With `opts.deep`, also read the files the
/// config and rubric name (stated quotes in sidecar files, scripts' comments,
/// controls, judge briefs, the Compose file).
pub fn check(dir: &Path, opts: &CheckOptions) -> CheckReport {
    let mut report = CheckReport::new(dir, Some(Format::TaskMd));
    let text = match read_task_md(dir) {
        Ok(t) => t,
        Err(diags) => {
            report.diagnostics = diags;
            return report;
        }
    };
    let doc = parse(&text, Path::new("task.md"));
    let empty = config::empty_source(0);
    let loc = doc.config_loc(&empty);
    let mut diags = doc.diagnostics.clone();
    if dir.join("task.toml").exists() {
        diags.push(
            Diagnostic::error(
                code::HARBOR_TASK_TOML,
                format!(
                    "task.toml is Harbor's config file; task.md keeps config in a ```toml task block ({})",
                    vocab::IMPORT_HINT
                ),
            )
            .in_file("task.toml")
            .help("`bench import <folder>` converts a Harbor task into task.md; otherwise remove task.toml"),
        );
    }
    let block = doc.config_source == ConfigSource::Block;
    let cfg = doc.config_value();
    if block {
        diags.extend(config::check_config(
            &doc.meta,
            &loc,
            &doc.blocks,
            dir,
            opts.deep,
        ));
        diags.extend(files::host_boundary(cfg, dir, &loc, opts.deep));
        diags.extend(files::family_files(cfg, dir, &loc));
        diags.extend(files::imported(cfg, dir, &loc));
    }
    diags.extend(files::canaries(&doc, &loc));

    let told = doc.told();
    let rubric = judgment::load(dir, "rubric", &mut diags);
    if let Some(r) = &rubric {
        diags.extend(judgment::check_rubric(
            r,
            Some(&told),
            dir,
            block.then_some(cfg),
            opts.deep,
        ));
    }
    if block {
        diags.extend(judgment::check_stage_rubrics(cfg, dir, &told, opts.deep));
    }
    let behaviors = judgment::load(dir, "behaviors", &mut diags);
    if let Some(b) = &behaviors {
        diags.extend(judgment::check_behaviors(b));
    }
    if block {
        diags.extend(judges::check(
            cfg,
            rubric.as_ref().map(|r| &r.value),
            behaviors.as_ref().map(|b| &b.value),
            dir,
            &loc,
            opts.deep,
        ));
        if opts.deep {
            diags.extend(files::seat_comments(cfg, dir));
        }
        diags.extend(files::check_script_models(cfg, dir, &loc));
    }
    if matches!(doc.config_source, ConfigSource::Block | ConfigSource::None) {
        diags.extend(files::check_package(
            cfg,
            rubric.is_some(),
            dir,
            &loc,
            opts.deep,
        ));
    }
    if block {
        report.facts = grading::lines(
            &doc,
            rubric.as_ref().map(|r| &r.value),
            behaviors.as_ref().map(|b| &b.value),
            dir,
        );
        if let Some(note) = files::generator_note(&doc, cfg) {
            diags.push(note);
        }
    }
    report.diagnostics = diags;
    if report.ok() {
        report.package = Some(map::package(&doc, rubric.as_ref(), behaviors.as_ref(), dir));
    }
    report
}
