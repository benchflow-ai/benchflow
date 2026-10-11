//! A task.md package to a split-layout folder, ported from task.md's
//! `tools/convert.py export-harbor` (`taskmd_to_harbor`, `export_harbor`).
//!
//! Export writes `task.toml`, `instruction.md` (or one per step), and the
//! package's folders under Harbor's names (`sandbox/` to `environment/`,
//! `verifier/` to `tests/`, `oracle/` to `solution/`). While a file is
//! still what import would have exported, it gets the original bytes import
//! kept in `import/harbor/`, so an unchanged import exports byte for byte.
//! Every setting Harbor cannot hold is a [`Loss`]; losing one that decides
//! how the task runs refuses the export unless degrading is allowed.

use std::collections::BTreeSet;
use std::path::Path;

use indexmap::IndexMap;
use serde_json::Value as Json;

use super::doc;
use super::files::{self, Rel};
use super::harbor::{self, Loss, LossSeverity, harbor_record};
use super::import::originals;
use super::import::originals as originals_dir;
use super::tomltext::{self, Comments};
use super::value::{KPath, Seg, Table, Val, canonical_json, parse_toml};
use super::{ConvertError, ExportReport, ExportStatus, RewardKitExport, Rewritten, io_err};
use crate::taskmd::{split_record, split_record_ignored};

const MODEL_JUDGES: [&str; 4] = ["llm", "agent", "vlm", "panel"];
const TASKMD_SCHEMA: &str = "https://task.md/schema/";
pub(crate) const EXPORT_REPORT: &str = "export-report.json";

/// An edit applied to a copy of the config before it is written.
type Edit<'a> = &'a dyn Fn(Table, Comments) -> (Table, Comments);

/// What a task.md becomes in Harbor's files.
pub(crate) struct HarborFiles {
    /// instruction.md, unless the task is multi-step (then only what import
    /// kept of Harbor's own).
    pub instruction: Option<String>,
    pub toml: String,
    /// Each step's instruction.md, by step name.
    pub steps: IndexMap<String, String>,
    pub losses: Vec<Loss>,
}

/// The name of the split-layout record in `cfg`'s `[import]` table.
fn record_name(cfg: &Table) -> &'static str {
    let import = cfg.get("import").and_then(Val::as_table);
    split_record(|k| import.is_some_and(|i| i.contains_key(k)))
}

/// The other record, when `cfg` has both names: export ignores it.
fn ignored_record(cfg: &Table) -> Option<&'static str> {
    let import = cfg.get("import").and_then(Val::as_table);
    split_record_ignored(|k| import.is_some_and(|i| i.contains_key(k)))
}

/// The split-layout record import kept in `cfg`, if any.
fn record(cfg: &Table) -> Option<&Table> {
    cfg.get("import")?
        .as_table()?
        .get(record_name(cfg))?
        .as_table()
}

fn invalid(message: String) -> ConvertError {
    ConvertError::Invalid {
        file: "task.md".into(),
        message,
    }
}

/// task.md text to Harbor's task.toml, instructions, and losses.
pub(crate) fn taskmd_to_harbor(
    task_md: &str,
    edit: Option<Edit<'_>>,
) -> Result<HarborFiles, ConvertError> {
    let parsed = doc::parse(task_md);
    let block = parsed.blocks.iter().find(|b| b.kind == "config");
    let (mut cfg, mut comments) = match block {
        Some(b) if b.arg.as_deref() == Some("toml") => (
            parse_toml(&b.body)
                .map_err(|e| invalid(format!("the config block is not valid TOML: {e}")))?,
            tomltext::scan_comments(&b.body).map_err(invalid)?,
        ),
        _ => (parsed.config.clone(), Comments::new()),
    };
    if let Some(edit) = edit {
        (cfg, comments) = edit(cfg, comments);
    }
    // A record export ignores goes nowhere, and neither do its comments
    // (remapped, they would land at the top of task.toml).
    if let Some(other) = ignored_record(&cfg) {
        let under = [Seg::Key("import".into()), Seg::Key(other.into())];
        comments.retain(|path, _| !path.starts_with(&under));
    }
    let exported = harbor::config_to_harbor(&cfg, record_name(&cfg), &[]).map_err(invalid)?;
    let comments = harbor::remap_comments(&comments, &exported.moves, &exported.config);
    let toml = tomltext::dumps(&exported.config, &comments)
        .ok_or_else(|| invalid("a value TOML cannot hold".into()))?;
    let mut losses = exported.losses;
    let mut steps = IndexMap::new();
    let has_steps = exported.config.contains_key("steps");
    let instruction = if let Some(Val::List(list)) = exported.config.get("steps") {
        let names: Vec<String> = list
            .iter()
            .filter_map(|s| s.as_table()?.get("name")?.as_str().map(str::to_string))
            .collect();
        let stage_text: IndexMap<String, String> = parsed
            .blocks
            .iter()
            .filter(|b| b.kind == "stage")
            .filter_map(|b| b.arg.clone().map(|a| (a, b.body.clone())))
            .collect();
        for (i, name) in names.iter().enumerate() {
            let text = if i == 0 {
                format!("{}\n", parsed.instruction_source)
            } else {
                format!(
                    "{}\n",
                    stage_text
                        .get(name)
                        .map_or("", String::as_str)
                        .trim_end_matches('\n')
                )
            };
            steps.insert(name.clone(), text);
        }
        record(&cfg)
            .and_then(|h| h.get("instruction"))
            .and_then(Val::as_str)
            .map(str::to_string)
    } else {
        Some(format!("{}\n", parsed.instruction_source))
    };
    for b in &parsed.blocks {
        if b.kind == "role" || b.kind == "user" || (b.kind == "stage" && !has_steps) {
            losses.push(Loss {
                path: format!("the ```{} block", b.info),
                reason: "a split-layout runner shows the agent instruction.md and each step's instruction only"
                    .into(),
                severity: LossSeverity::RunAffecting,
                text: format!("the ```{} block", b.info),
            });
        }
    }
    Ok(HarborFiles {
        instruction,
        toml,
        steps,
        losses,
    })
}

/// A verifier file task.md reads (rubric.json, behaviors.json), known by
/// its `$schema`; `None` for a script's own file of the same name.
fn taskmd_json(path: &Path) -> Option<Json> {
    let text = std::fs::read_to_string(path).ok()?;
    let data: Json = serde_json::from_str(&text).ok()?;
    let schema = data.get("$schema")?.as_str()?;
    (data.is_object() && schema.starts_with(TASKMD_SCHEMA)).then_some(data)
}

fn affects_reward(consequence: Option<&Json>) -> bool {
    match consequence {
        Some(Json::Object(m)) if !m.contains_key("penalty") => {
            m.values().any(|v| affects_reward(Some(v)))
        }
        Some(Json::String(s)) => s != "report",
        _ => true,
    }
}

/// The files Harbor copies into tests/ but never reads: task.md's rubric
/// and behaviors, and the judges' briefs.
fn file_losses(src: &Path, cfg: &Table, rk: Option<&Restore>) -> Vec<Loss> {
    let mut losses = Vec::new();
    let converted: Option<Vec<String>> = rk_record(cfg)
        .and_then(|r| r.get("converted"))
        .and_then(Val::as_list)
        .map(|l| {
            l.iter()
                .map(|x| {
                    x.as_str()
                        .map_or_else(|| harbor::py_repr(x), str::to_string)
                })
                .collect()
        });
    let hint = match (&converted, rk.and_then(|r| r.why.as_ref())) {
        (Some(c), Some(why)) => format!(
            "; its judges came from RewardKit ({}), whose files export writes back only while the conversion is unchanged, and {why}",
            c.join(", ")
        ),
        (Some(c), None) => format!(
            "; its judges came from RewardKit ({}), so export the original split-layout task instead",
            c.join(", ")
        ),
        _ => String::new(),
    };
    let mut folders: Vec<(String, Table)> = vec![("verifier".into(), cfg.clone())];
    if let Some(Val::Table(stages)) = cfg.get("stages") {
        for (n, st) in stages {
            if let Val::Table(t) = st {
                folders.push((format!("stages/{n}/verifier"), t.clone()));
            }
        }
    }
    let mut judged: IndexMap<String, bool> = IndexMap::new();
    for (vdir, _) in &folders {
        let rubric = taskmd_json(&src.join(vdir).join("rubric.json"));
        let uses_models = rubric.as_ref().is_some_and(|r| {
            r.as_object().is_some_and(|o| !o.is_empty())
                && r.get("criteria")
                    .and_then(Json::as_array)
                    .is_some_and(|cs| {
                        cs.iter().any(|c| {
                            c.get("judge")
                                .and_then(Json::as_str)
                                .is_some_and(|j| MODEL_JUDGES.contains(&j))
                        })
                    })
        });
        judged.insert(vdir.clone(), uses_models);
        if rubric.is_some() {
            losses.push(Loss {
                path: format!("{vdir}/rubric.json"),
                reason: format!(
                    "a split-layout runner reads no rubric: its reward comes from test.sh alone{hint}"
                ),
                severity: LossSeverity::RunAffecting,
                text: format!("{vdir}/rubric.json (copied to tests/, which a split-layout runner does not read)"),
            });
        }
        if let Some(behaviors) = taskmd_json(&src.join(vdir).join("behaviors.json")) {
            let reward = behaviors
                .get("watch")
                .and_then(Json::as_array)
                .is_some_and(|w| {
                    w.iter()
                        .filter(|b| b.is_object())
                        .any(|b| affects_reward(b.get("consequence")))
                });
            losses.push(Loss {
                path: format!("{vdir}/behaviors.json"),
                reason: format!(
                    "the split layout watches no behaviors{}",
                    if reward {
                        ", so their consequences never touch the reward"
                    } else {
                        ", and these only report"
                    }
                ),
                severity: if reward {
                    LossSeverity::RunAffecting
                } else {
                    LossSeverity::Metadata
                },
                text: format!(
                    "{vdir}/behaviors.json (copied to tests/, which a split-layout runner does not read)"
                ),
            });
        }
    }
    let any_judged = judged.values().any(|j| *j);
    for (vdir, table) in &folders {
        let judges: Table = table
            .get("verifier")
            .and_then(Val::as_table)
            .and_then(|v| v.get("judges"))
            .and_then(Val::as_table)
            .cloned()
            .unwrap_or_default();
        let uses = judged.get(vdir).copied().unwrap_or(false) || (vdir == "verifier" && any_judged);
        let mut briefs: BTreeSet<String> = BTreeSet::new();
        briefs.insert(format!("{vdir}/judge.md"));
        for v in judges.values() {
            if let Some(b) = v
                .as_table()
                .and_then(|t| t.get("brief"))
                .and_then(Val::as_str)
            {
                briefs.insert(b.to_string());
            }
        }
        for brief in briefs {
            if uses && src.join(&brief).is_file() {
                losses.push(Loss {
                    path: brief.clone(),
                    reason: format!(
                        "the brief task.md's model judges read; a split-layout runner runs no task.md judges{hint}"
                    ),
                    severity: LossSeverity::RunAffecting,
                    text: format!("{brief} (copied to tests/, which a split-layout runner does not read)"),
                });
            }
        }
    }
    losses
}

fn rk_record(cfg: &Table) -> Option<&Table> {
    record(cfg)?.get("rewardkit")?.as_table()
}

/// Every `[verifier.judges.<role>]` table, by dotted name.
fn judge_tables(cfg: &Table) -> IndexMap<String, KPath> {
    let mut out = IndexMap::new();
    let mut places: Vec<(KPath, Option<&Val>)> =
        vec![(super::value::kp(&["verifier"]), cfg.get("verifier"))];
    if let Some(Val::Table(stages)) = cfg.get("stages") {
        for (n, st) in stages {
            if let Val::Table(t) = st {
                places.push((
                    super::value::kp(&["stages", n, "verifier"]),
                    t.get("verifier"),
                ));
            }
        }
    }
    for (base, ver) in places {
        let Some(Val::Table(ver)) = ver else { continue };
        let Some(Val::Table(judges)) = ver.get("judges") else {
            continue;
        };
        for (role, table) in judges {
            if matches!(table, Val::Table(_)) {
                let mut p = base.clone();
                p.push(Seg::Key("judges".into()));
                p.push(Seg::Key(role.clone()));
                out.insert(super::value::dotted(&p), p);
            }
        }
    }
    out
}

fn at<'a>(cfg: &'a Table, path: &[Seg]) -> Option<&'a Val> {
    let mut cur: Option<&Val> = None;
    for (i, p) in path.iter().enumerate() {
        let Seg::Key(k) = p else { return None };
        cur = if i == 0 {
            cfg.get(k)
        } else {
            cur?.as_table()?.get(k)
        };
    }
    cur
}

/// Whether export can undo a RewardKit conversion import recorded.
struct Restore {
    ok: bool,
    why: Option<String>,
    /// Harbor path -> the kept file in the package.
    files: IndexMap<String, Rel>,
    kept: Vec<String>,
    /// Package files the conversion wrote.
    skip: BTreeSet<String>,
    tables: Vec<String>,
    env: Table,
    stage_env: Table,
}

fn rk_restore(src: &Path, cfg: &Table) -> Option<Restore> {
    let dir = originals(record_name(cfg));
    let rk = rk_record(cfg)?;
    let (Some(Val::Table(kept)), Some(Val::Table(wrote))) = (rk.get("kept"), rk.get("wrote"))
    else {
        return None;
    };
    let tables = judge_tables(cfg);
    let mut out = Restore {
        ok: true,
        why: None,
        files: IndexMap::new(),
        kept: {
            let mut k: Vec<String> = kept.keys().cloned().collect();
            k.sort();
            k
        },
        skip: BTreeSet::new(),
        tables: Vec::new(),
        env: rk
            .get("env")
            .and_then(Val::as_table)
            .cloned()
            .unwrap_or_default(),
        stage_env: rk
            .get("stage_env")
            .and_then(Val::as_table)
            .cloned()
            .unwrap_or_default(),
    };
    for (path, h) in wrote {
        let (got, name) = if path.contains('/') {
            let f = src.join(path);
            out.skip.insert(path.clone());
            (
                f.is_file().then(|| files::sha256_file(&f).ok()).flatten(),
                path.clone(),
            )
        } else {
            out.tables.push(path.clone());
            let got = tables
                .get(path)
                .and_then(|p| at(cfg, p))
                .map(|t| files::sha256_bytes(canonical_json(t).as_bytes()));
            (got, format!("[{path}]"))
        };
        if got.as_deref() != h.as_str() && out.ok {
            out.ok = false;
            out.why = Some(format!(
                "{name} {} since the conversion",
                if got.is_some() { "changed" } else { "is gone" }
            ));
        }
    }
    for (rel, h) in kept {
        let r = files::rel(&format!("{dir}/{rel}"));
        let f = files::under(src, &r);
        let same = f.is_file() && files::sha256_file(&f).ok().as_deref() == h.as_str();
        if !same {
            if out.ok {
                out.ok = false;
                out.why = Some(format!(
                    "{dir}/{rel} is missing or no longer matches its recorded hash"
                ));
            }
        } else {
            out.files.insert(rel.clone(), r);
        }
    }
    Some(out)
}

/// The edit that undoes a RewardKit conversion in the config.
fn undo_rewardkit(rk: &Restore) -> impl Fn(Table, Comments) -> (Table, Comments) + '_ {
    move |mut cfg: Table, mut comments: Comments| {
        let tables = judge_tables(&cfg);
        for name in &rk.tables {
            let Some(path) = tables.get(name) else {
                continue;
            };
            remove_at(&mut cfg, path);
            comments.retain(|k, _| k.len() < path.len() || k[..path.len()] != path[..]);
        }
        if !rk.env.is_empty() {
            let v = entry_table(entry_table(&mut cfg, "verifier"), "env");
            v.extend(rk.env.clone());
        }
        for (step, env) in &rk.stage_env {
            let Val::Table(env) = env else { continue };
            if let Some(Val::Table(stage)) = cfg
                .get_mut("stages")
                .and_then(|s| s.as_table_mut())
                .and_then(|s| s.get_mut(step))
            {
                entry_table(entry_table(stage, "verifier"), "env").extend(env.clone());
            }
        }
        (cfg, comments)
    }
}

fn entry_table<'t>(t: &'t mut Table, key: &str) -> &'t mut Table {
    let e = t
        .entry(key.to_string())
        .or_insert_with(|| Val::Table(Table::new()));
    if !matches!(e, Val::Table(_)) {
        *e = Val::Table(Table::new());
    }
    e.as_table_mut().expect("a table")
}

/// Remove the table at `path`, and its parent when that leaves it empty.
fn remove_at(cfg: &mut Table, path: &[Seg]) {
    fn walk<'t>(t: &'t mut Table, path: &[Seg]) -> Option<&'t mut Table> {
        let mut cur = t;
        for p in path {
            let Seg::Key(k) = p else { return None };
            cur = cur.get_mut(k)?.as_table_mut()?;
        }
        Some(cur)
    }
    let (Some(Seg::Key(last)), parent_path) = (path.last(), &path[..path.len() - 1]) else {
        return;
    };
    let Some(parent) = walk(cfg, parent_path) else {
        return;
    };
    parent.shift_remove(last);
    if parent.is_empty()
        && !parent_path.is_empty()
        && let (Some(Seg::Key(pk)), Some(grand)) = (
            parent_path.last(),
            walk(cfg, &parent_path[..parent_path.len() - 1]),
        )
    {
        grand.shift_remove(pk);
    }
}

/// Everything an export writes, decided before anything is written.
pub(crate) struct ExportPlan {
    pub report: ExportReport,
    /// Generated files, by path in the output.
    out_bytes: IndexMap<String, Vec<u8>>,
    /// Package files copied: (source path, output path).
    copies: Vec<(Rel, Rel)>,
    /// RewardKit's own files, written back as Harbor had them.
    rk_files: Vec<(String, Rel)>,
}

fn to_harbor_path(rel: &[String], steps: &BTreeSet<String>) -> Option<Rel> {
    let parts: Vec<&str> = rel.iter().map(String::as_str).collect();
    if parts == ["task.md"] {
        return None;
    }
    let taskmd = |n: &str| -> Option<&'static str> {
        match n {
            "sandbox" => Some("environment"),
            "verifier" => Some("tests"),
            "oracle" => Some("solution"),
            _ => None,
        }
    };
    if let Some(h) = taskmd(parts[0]) {
        let mut out = vec![h.to_string()];
        out.extend(rel[1..].iter().cloned());
        return Some(out);
    }
    if parts[0] == "stages" && parts.len() > 2 && steps.contains(parts[1]) {
        let mut out = vec![
            "steps".to_string(),
            rel[1].clone(),
            taskmd(parts[2]).unwrap_or(parts[2]).to_string(),
        ];
        out.extend(rel[3..].iter().cloned());
        return Some(out);
    }
    Some(rel.to_vec())
}

/// Decide what exporting `src` to `dst` writes, and whether it may.
pub(crate) fn plan(
    src: &Path,
    dst: &Path,
    allow_degraded: bool,
) -> Result<ExportPlan, ConvertError> {
    let md_path = src.join("task.md");
    let task_md = crate::split::read_text(&md_path).map_err(io_err(&md_path))?;
    let generated = taskmd_to_harbor(&task_md, None)?;
    let cfg = doc::parse(&task_md).config;
    let rk = rk_restore(src, &cfg);
    let undo = rk.as_ref().is_some_and(|r| r.ok);
    let mut losses = generated.losses.clone();
    losses.extend(file_losses(src, &cfg, rk.as_ref()));
    if let (true, Some(r)) = (undo, &rk) {
        let tables: Vec<String> = r.tables.iter().map(|t| format!("[{t}]")).collect();
        losses.retain(|l| {
            !r.skip.contains(&l.path) && !tables.iter().any(|t| l.path.starts_with(t.as_str()))
        });
    }
    let kept: Option<&Table> = record(&cfg);
    let dir = originals(record_name(&cfg));
    let originals: Table = kept
        .and_then(|k| {
            harbor_record("originals", k.get("originals"))
                .then(|| k.get("originals"))
                .flatten()
        })
        .and_then(Val::as_table)
        .cloned()
        .unwrap_or_default();
    let mut gen_files: IndexMap<String, String> = IndexMap::new();
    gen_files.insert("task.toml".into(), generated.toml.clone());
    if let Some(i) = &generated.instruction {
        gen_files.insert("instruction.md".into(), i.clone());
    }
    for (name, text) in &generated.steps {
        gen_files.insert(format!("steps/{name}/instruction.md"), text.clone());
    }
    let mut restored: Vec<String> = Vec::new();
    let mut rewritten: Vec<Rewritten> = Vec::new();
    let mut out_bytes: IndexMap<String, Vec<u8>> = IndexMap::new();
    for (rel, text) in &gen_files {
        let mut data = text.clone().into_bytes();
        let mut rec = originals.get(rel).and_then(Val::as_table);
        if rel == "task.toml" && rk.is_some() && !undo {
            rec = None;
        }
        let converted_hash = files::sha256_bytes(&data);
        let field = |r: &Table, k: &str| r.get(k).and_then(Val::as_str).map(str::to_string);
        let same =
            rec.is_some_and(|r| field(r, "converted").as_deref() == Some(converted_hash.as_str()));
        let kept_file = files::under(src, &files::rel(&format!("{dir}/{rel}")));
        if same && rec.is_some_and(|r| field(r, "original") == field(r, "converted")) {
            restored.push(rel.clone());
        } else if same
            && kept_file.is_file()
            && files::sha256_file(&kept_file).ok() == rec.and_then(|r| field(r, "original"))
        {
            data = std::fs::read(&kept_file).map_err(io_err(&kept_file))?;
            restored.push(rel.clone());
        } else {
            if rel == "task.toml" && undo {
                let r = rk.as_ref().expect("undo implies a record");
                let edit = undo_rewardkit(r);
                data = taskmd_to_harbor(&task_md, Some(&edit))?.toml.into_bytes();
            }
            if rec.is_some() || (kept.is_some() && rel == "task.toml") {
                let why = if rel == "task.toml" && rk.is_some() && !undo {
                    format!(
                        "its RewardKit conversion cannot be undone, since {}",
                        rk.as_ref().and_then(|r| r.why.clone()).unwrap_or_default()
                    )
                } else if rec.is_some() && !same {
                    "its settings or text changed since import".to_string()
                } else if rec.is_some() {
                    format!("{dir}/{rel} is missing or no longer matches its recorded hash")
                } else {
                    "import kept no original: the package predates the record".to_string()
                };
                rewritten.push(Rewritten {
                    path: rel.clone(),
                    reason: format!(
                        "{why}, so export wrote the converted form, and the split layout's content digest differs from the original's"
                    ),
                });
            }
        }
        out_bytes.insert(rel.clone(), data);
    }
    let mut skip: BTreeSet<Rel> = originals
        .keys()
        .map(|r| files::rel(&format!("{dir}/{r}")))
        .collect();
    if let Some(r) = &rk {
        skip.extend(r.kept.iter().map(|k| files::rel(&format!("{dir}/{k}"))));
    }
    let mut rk_files = Vec::new();
    if let (true, Some(r)) = (undo, &rk) {
        skip.extend(r.skip.iter().map(|s| files::rel(s)));
        let mut names: Vec<&String> = r.files.keys().collect();
        names.sort();
        restored.extend(names.iter().map(|s| (*s).clone()));
        rk_files = r
            .files
            .iter()
            .map(|(h, f)| (h.clone(), f.clone()))
            .collect();
    }
    let steps: BTreeSet<String> = generated.steps.keys().cloned().collect();
    // With both records, neither folder is the package's own files: the one
    // read gave back its originals, and the other is left out.
    let ignored = ignored_record(&cfg);
    let folders: Vec<String> = match ignored {
        Some(other) => vec![format!("{dir}/"), format!("{}/", originals_dir(other))],
        None => Vec::new(),
    };
    let in_folders = |rel: &Rel| {
        let p = files::posix(rel);
        folders.iter().any(|d| p.starts_with(d.as_str()))
    };
    if let Some(other) = ignored {
        losses.push(Loss {
            path: format!("[import.{other}]"),
            reason: format!(
                "the package also has [import.{}], which export read, restoring its originals from {dir}/; [import.{other}], {}/, and the other files in {dir}/ are left out",
                record_name(&cfg),
                originals_dir(other)
            ),
            severity: LossSeverity::Metadata,
            text: format!("[import.{other}] and {}/", originals_dir(other)),
        });
    }
    let mut copies: Vec<(Rel, Rel)> = Vec::new();
    for rel in files::list(src).map_err(io_err(src))? {
        if skip.contains(&rel) || in_folders(&rel) {
            continue;
        }
        if let Some(to) = to_harbor_path(&rel, &steps) {
            if files::posix(&to) == EXPORT_REPORT {
                losses.push(Loss {
                    path: files::posix(&rel),
                    reason: "the export report takes its place".into(),
                    severity: LossSeverity::Metadata,
                    text: format!(
                        "{} (the package's own file of the export report's name)",
                        files::posix(&rel)
                    ),
                });
                continue;
            }
            copies.push((rel, to));
        }
    }
    let fatal = losses
        .iter()
        .filter(|l| l.severity == LossSeverity::RunAffecting)
        .count();
    let status = if losses.is_empty() {
        ExportStatus::Lossless
    } else if fatal > 0 && !allow_degraded {
        ExportStatus::Refused
    } else {
        ExportStatus::Degraded
    };
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let summary = match status {
        ExportStatus::Lossless => "the split layout holds everything".to_string(),
        ExportStatus::Refused => format!(
            "the split layout would drop {fatal} run-affecting setting{}, so nothing was written (allow a degraded export to write it anyway)",
            plural(fatal)
        ),
        ExportStatus::Degraded => {
            let mut s = format!(
                "exported without {} thing{} the split layout cannot hold",
                losses.len(),
                plural(losses.len())
            );
            if fatal > 0 {
                s.push_str(&format!(", {fatal} of them run-affecting"));
            }
            s
        }
    };
    let mut restored_ext: Vec<String> = kept
        .map(|k| {
            k.iter()
                .filter(|(key, v)| *key != "instruction" && !harbor_record(key, Some(v)))
                .map(|(key, _)| format!("import.{}.{key}", record_name(&cfg)))
                .collect()
        })
        .unwrap_or_default();
    restored_ext.sort();
    let mut input_hashes = IndexMap::new();
    input_hashes.insert(
        "task.md".to_string(),
        files::sha256_file(&md_path).map_err(io_err(&md_path))?,
    );
    let rewardkit = rk.as_ref().map(|r| RewardKitExport {
        undone: undo,
        why: r.why.clone(),
        restored: if undo {
            let mut f: Vec<String> = r.files.keys().cloned().collect();
            f.sort();
            f
        } else {
            Vec::new()
        },
        variables: if undo {
            let mut v: BTreeSet<String> = r.env.keys().cloned().collect();
            for env in r.stage_env.values() {
                if let Val::Table(t) = env {
                    v.extend(t.keys().cloned());
                }
            }
            v.into_iter().collect()
        } else {
            Vec::new()
        },
    });
    let report = ExportReport {
        report: "export",
        from: "task.md",
        // As convert.py writes it: a format identifier other tools read.
        to: "harbor",
        tool: "bench export",
        status,
        allow_degraded,
        source_task_dir: src.display().to_string(),
        output_dir: dst.display().to_string(),
        summary,
        input_hashes,
        restored_extension_paths: restored_ext,
        restored_originals: restored,
        rewritten,
        losses,
        rewardkit,
        emitted_files: None,
        output_hashes: None,
    };
    Ok(ExportPlan {
        report,
        out_bytes,
        copies,
        rk_files,
    })
}

/// Write a planned export into `dst`, which exists and is empty.
pub(crate) fn write(
    src: &Path,
    dst: &Path,
    plan: ExportPlan,
) -> Result<ExportReport, ConvertError> {
    let ExportPlan {
        mut report,
        out_bytes,
        copies,
        rk_files,
    } = plan;
    let mut written: Vec<String> = Vec::new();
    for (rel, data) in &out_bytes {
        let to = files::under(dst, &files::rel(rel));
        files::write(&to, data).map_err(io_err(&to))?;
        written.push(rel.clone());
    }
    let mut emitted = written.clone();
    for (from, to) in &copies {
        let (f, t) = (files::under(src, from), files::under(dst, to));
        files::copy(&f, &t).map_err(io_err(&f))?;
        emitted.push(files::posix(to));
    }
    for (rel, from) in &rk_files {
        let (f, t) = (files::under(src, from), files::under(dst, &files::rel(rel)));
        files::copy(&f, &t).map_err(io_err(&f))?;
        emitted.push(rel.clone());
    }
    emitted.sort();
    let mut hashes = IndexMap::new();
    for w in &written {
        let p = files::under(dst, &files::rel(w));
        hashes.insert(w.clone(), files::sha256_file(&p).map_err(io_err(&p))?);
    }
    report.emitted_files = Some(emitted);
    report.output_hashes = Some(hashes);
    let report_path = dst.join(EXPORT_REPORT);
    let json = serde_json::to_string_pretty(&report).map_err(|e| ConvertError::Invalid {
        file: report_path.clone(),
        message: e.to_string(),
    })?;
    std::fs::write(&report_path, format!("{json}\n")).map_err(io_err(&report_path))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports_a_simple_task() {
        let md = "Do the thing.\n\n```toml task\nname = \"acme/x\"\n\n[agent]\ntimeout = \"5m\"  # five\n\n[verifier]\nmount = \"/tests\"\n```\n";
        let out = taskmd_to_harbor(md, None).unwrap();
        assert_eq!(out.instruction.as_deref(), Some("Do the thing.\n"));
        assert_eq!(
            out.toml,
            "[task]\nname = \"acme/x\"\n\n[agent]\ntimeout_sec = 300.0  # five\n"
        );
        assert!(out.losses.is_empty());
    }

    #[test]
    fn stage_blocks_become_steps() {
        let md = "First.\n\n```stage two\nSecond.\n```\n\n```toml task\n[stages.one]\nunlock = \"at_start\"\n\n[stages.two]\nunlock = \"after:one\"\n```\n";
        let out = taskmd_to_harbor(md, None).unwrap();
        assert_eq!(out.instruction, None);
        assert_eq!(out.steps.get("one").map(String::as_str), Some("First.\n"));
        assert_eq!(out.steps.get("two").map(String::as_str), Some("Second.\n"));
        assert!(out.toml.contains("[[steps]]\nname = \"one\""));
    }

    #[test]
    fn reports_what_harbor_cannot_hold() {
        let md = "Do it.\n\n```role critic\nCriticize.\n```\n\n```toml task\n[sandbox]\nclock = { start = \"2026-01-01T00:00:00Z\", enforce = \"preload\" }\n```\n";
        let out = taskmd_to_harbor(md, None).unwrap();
        let paths: Vec<&str> = out.losses.iter().map(|l| l.path.as_str()).collect();
        assert!(paths.contains(&"[sandbox] clock"));
        assert!(paths.contains(&"the ```role critic block"));
    }
}
