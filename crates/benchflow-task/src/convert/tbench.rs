//! A Terminal-Bench v1 task folder to Harbor's split layout, ported from
//! Harbor's `task migrate` (`harbor/mappers/terminal_bench.py`). Import
//! then turns the split layout into task.md as it does any Harbor task.
//!
//! | Terminal-Bench v1 | Split layout |
//! |---|---|
//! | `task.yaml` `instruction` | `instruction.md` |
//! | `task.yaml` metadata and timeouts | `task.toml` |
//! | `docker-compose.yaml` with one service | Folded into `environment/Dockerfile` (`ENV`, `WORKDIR`, `EXPOSE`, `ENTRYPOINT`, `--platform`) |
//! | `docker-compose.yaml` with more | `environment/docker-compose.yaml` in Harbor's template, main service `main` |
//! | `run-tests.sh` | `tests/test.sh`, writing `reward.txt` from its exit code |
//! | `tests/`, `solution.sh` | `tests/`, `solution/solve.sh` |
//! | Anything else | `environment/` |
//!
//! The output is the same as Harbor's, byte for byte, except that a build
//! context written `./name` is treated as `name` (Harbor copies it twice).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_yaml_ng::{Mapping, Value};

use super::files;
use super::{ConvertError, io_err};

/// What `run-tests.sh` gets appended, so the verifier writes a reward.
pub(crate) const REWARD_LOGGING_SUFFIX: &str = "
_EXIT_CODE=$?
if [ $_EXIT_CODE -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
exit $_EXIT_CODE
";

/// Resources to set on every converted task (`--cpus` and the rest).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    pub cpus: Option<u64>,
    pub memory_mb: Option<u64>,
    pub storage_mb: Option<u64>,
    pub gpus: Option<u64>,
}

/// Whether `dir` is a Terminal-Bench v1 task: it has `task.yaml` and is
/// not already in another format.
pub fn is_tbench(dir: &Path) -> bool {
    dir.join("task.yaml").is_file() && crate::load::detect(dir).is_none()
}

/// The Terminal-Bench v1 tasks under `root` (or `root` itself), sorted:
/// the shallowest folders with `task.yaml` that are not in another format.
/// Skips the folders [`crate::discover`] skips.
pub fn discover_tbench(root: &Path) -> Vec<PathBuf> {
    fn visit(dir: &Path, out: &mut Vec<PathBuf>) {
        if is_tbench(dir) {
            out.push(dir.to_path_buf());
            return;
        }
        if crate::load::detect(dir).is_some() {
            return;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let skip = matches!(
                name.as_str(),
                ".git" | "node_modules" | "target" | "jobs" | ".cache" | "__pycache__" | ".venv"
            );
            if !skip && e.file_type().is_ok_and(|t| t.is_dir()) {
                visit(&e.path(), out);
            }
        }
    }
    let mut out = Vec::new();
    visit(root, &mut out);
    out.sort();
    out
}

fn default_unknown() -> String {
    "unknown".into()
}
fn default_category() -> String {
    "software_engineering".into()
}
fn default_agent_timeout() -> f64 {
    360.0
}
fn default_test_timeout() -> f64 {
    60.0
}

/// The fields of `task.yaml` Harbor reads; the rest are ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct TaskYaml {
    pub instruction: String,
    #[serde(default = "default_unknown")]
    pub author_name: String,
    #[serde(default = "default_unknown")]
    pub author_email: String,
    #[serde(default = "default_unknown")]
    pub difficulty: String,
    #[serde(default = "default_category")]
    pub category: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub estimated_duration_sec: Option<f64>,
    #[serde(default)]
    pub expert_time_estimate_min: Option<f64>,
    #[serde(default)]
    pub junior_time_estimate_min: Option<f64>,
    #[serde(default = "default_agent_timeout")]
    pub max_agent_timeout_sec: f64,
    #[serde(default = "default_test_timeout")]
    pub max_test_timeout_sec: f64,
}

/// `task.yaml` keys Harbor reads.
const KNOWN_KEYS: [&str; 11] = [
    "instruction",
    "author_name",
    "author_email",
    "difficulty",
    "category",
    "tags",
    "estimated_duration_sec",
    "expert_time_estimate_min",
    "junior_time_estimate_min",
    "max_agent_timeout_sec",
    "max_test_timeout_sec",
];

impl TaskYaml {
    pub(crate) fn parse(text: &str) -> Result<(TaskYaml, Vec<String>), String> {
        let raw: Value =
            serde_yaml_ng::from_str(text).map_err(|e| format!("not valid YAML: {e}"))?;
        let Value::Mapping(map) = &raw else {
            return Err("not a mapping of settings".into());
        };
        let ignored = map
            .keys()
            .filter_map(Value::as_str)
            .filter(|k| !KNOWN_KEYS.contains(k))
            .map(str::to_string)
            .collect();
        // A setting left empty takes its default.
        let kept: Mapping = map
            .iter()
            .filter(|(k, v)| !v.is_null() && k.as_str().is_some_and(|k| KNOWN_KEYS.contains(&k)))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let cfg: TaskYaml =
            serde_yaml_ng::from_value(Value::Mapping(kept)).map_err(|e| e.to_string())?;
        if !["easy", "medium", "hard", "unknown"].contains(&cfg.difficulty.as_str()) {
            return Err(format!(
                "difficulty is {:?}; Terminal-Bench v1 uses easy, medium, hard, or unknown",
                cfg.difficulty
            ));
        }
        Ok((cfg, ignored))
    }

    /// `task.toml` as Harbor's `TaskConfig.model_dump_toml` writes it.
    pub(crate) fn to_task_toml(&self, res: &Resources) -> String {
        let mut s = String::from("schema_version = \"1.4\"\nartifacts = []\n\n[metadata]\n");
        let mut kv = |k: &str, v: String| {
            s.push_str(k);
            s.push_str(" = ");
            s.push_str(&v);
            s.push('\n');
        };
        kv("author_name", toml_str(&self.author_name));
        kv("author_email", toml_str(&self.author_email));
        kv("difficulty", toml_str(&self.difficulty));
        kv("category", toml_str(&self.category));
        let tags = if self.tags.is_empty() {
            "[]".to_string()
        } else {
            let items: String = self
                .tags
                .iter()
                .map(|t| format!(" {},", toml_str(t)))
                .collect();
            format!("[{items}]")
        };
        kv("tags", tags);
        for (k, v) in [
            ("estimated_duration_sec", self.estimated_duration_sec),
            ("expert_time_estimate_min", self.expert_time_estimate_min),
            ("junior_time_estimate_min", self.junior_time_estimate_min),
        ] {
            if let Some(v) = v {
                kv(k, py_float(v));
            }
        }
        s.push_str(&format!(
            "\n[verifier]\ntimeout_sec = {}\ncollect = []\n\n[verifier.env]\n\n[agent]\ntimeout_sec = {}\n\n[environment]\nnetwork_mode = \"public\"\nbuild_timeout_sec = 600.0\nos = \"linux\"\n",
            py_float(self.max_test_timeout_sec),
            py_float(self.max_agent_timeout_sec)
        ));
        for (k, v) in [
            ("cpus", res.cpus),
            ("memory_mb", res.memory_mb),
            ("storage_mb", res.storage_mb),
            ("gpus", res.gpus),
        ] {
            if let Some(v) = v {
                s.push_str(&format!("{k} = {v}\n"));
            }
        }
        s.push_str("mcp_servers = []\n\n[environment.env]\n\n[solution.env]\n");
        s
    }
}

/// A TOML basic string, escaped as Python's `toml` package does for the
/// text task.yaml holds.
fn toml_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                out.push_str(&format!("\\u{:04x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A float as Python's `repr` writes it (`180.0`, `0.5`, `1e+16`, `1e-05`).
fn py_float(v: f64) -> String {
    super::value::py_float_repr(v)
}

/// Python's `json.dumps` of a list of strings (ASCII only, `", "` between).
fn py_json_list(items: &[String]) -> String {
    let quoted: Vec<String> = items
        .iter()
        .map(|s| {
            let mut out = String::new();
            for c in serde_json::to_string(s).expect("a string").chars() {
                if c.is_ascii() {
                    out.push(c);
                } else {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{u:04x}"));
                    }
                }
            }
            out
        })
        .collect();
    format!("[{}]", quoted.join(", "))
}

/// A YAML scalar as Python's f-strings write it (`True`, `None`, `8080`).
fn py_scalar(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => py_float(f),
            _ => n.to_string(),
        },
        Value::String(s) => s.clone(),
        other => serde_yaml_ng::to_string(other)
            .unwrap_or_default()
            .trim_end()
            .to_string(),
    }
}

const DOCKERFILE_FIELDS: [&str; 5] = [
    "environment",
    "working_dir",
    "expose",
    "entrypoint",
    "platform",
];
const IGNORED_FIELDS: [&str; 6] = [
    "build",
    "image",
    "container_name",
    "command",
    "volumes",
    "environment",
];
const TBENCH_DEFAULT_VOLUMES: [&str; 2] = [
    "${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}",
    "${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}",
];
const TBENCH_DEFAULT_ENV: &str = "TEST_DIR=${T_BENCH_TEST_DIR}";
/// Main-service keys Harbor's compose template replaces.
const HARBOR_REPLACES: [&str; 7] = [
    "build",
    "image",
    "container_name",
    "command",
    "environment",
    "volumes",
    "deploy",
];

fn services(compose: &Value) -> Option<&Mapping> {
    compose.get("services").and_then(Value::as_mapping)
}

/// Whether the compose file says nothing a Dockerfile cannot.
pub(crate) fn can_collapse(compose: &Value) -> bool {
    let Some(svcs) = services(compose) else {
        return false;
    };
    if svcs.len() != 1 {
        return false;
    }
    let top_ok = compose.as_mapping().is_some_and(|m| {
        m.keys()
            .all(|k| matches!(k.as_str(), Some("services" | "version")))
    });
    if !top_ok {
        return false;
    }
    let Some(svc) = svcs.values().next().and_then(Value::as_mapping) else {
        return svcs.values().next().is_some_and(Value::is_null);
    };
    svc.iter().all(|(k, v)| {
        let Some(k) = k.as_str() else {
            return false;
        };
        if !DOCKERFILE_FIELDS.contains(&k) && !IGNORED_FIELDS.contains(&k) {
            return false;
        }
        if k == "volumes" {
            return v.as_sequence().is_some_and(|vols| {
                vols.iter().all(|x| {
                    x.as_str()
                        .is_some_and(|x| TBENCH_DEFAULT_VOLUMES.contains(&x))
                })
            });
        }
        true
    })
}

/// The main service: `client`, else the first.
pub(crate) fn main_service(compose: &Value) -> Option<(String, Value)> {
    let svcs = services(compose)?;
    let key = if svcs.contains_key("client") {
        Value::String("client".into())
    } else {
        svcs.keys().next()?.clone()
    };
    let name = py_scalar(&key);
    Some((name, svcs.get(&key).cloned().unwrap_or(Value::Null)))
}

/// A service's build context (`.` when it has none).
pub(crate) fn build_context(service: &Value) -> String {
    match service.get("build") {
        Some(Value::Mapping(b)) => b
            .get("context")
            .map(py_scalar)
            .unwrap_or_else(|| ".".into()),
        Some(Value::String(s)) => s.clone(),
        _ => ".".into(),
    }
}

/// The Dockerfile lines that carry a service's settings.
pub(crate) fn dockerfile_additions(service: &Value) -> Vec<String> {
    let mut out = vec!["ENV TEST_DIR=/tests".to_string()];
    match service.get("environment") {
        Some(Value::Sequence(vars)) => {
            for v in vars {
                let v = py_scalar(v);
                if v == TBENCH_DEFAULT_ENV {
                    continue;
                }
                out.push(format!("ENV {v}"));
            }
        }
        Some(Value::Mapping(vars)) => {
            for (k, v) in vars {
                let pair = format!("{}={}", py_scalar(k), py_scalar(v));
                if pair == TBENCH_DEFAULT_ENV {
                    continue;
                }
                out.push(format!("ENV {pair}"));
            }
        }
        _ => {}
    }
    if let Some(wd) = service.get("working_dir").map(py_scalar)
        && !wd.contains("${")
    {
        out.push(format!("WORKDIR {wd}"));
    }
    match service.get("expose") {
        Some(Value::Sequence(ports)) => {
            out.extend(ports.iter().map(|p| format!("EXPOSE {}", py_scalar(p))))
        }
        Some(p) => out.push(format!("EXPOSE {}", py_scalar(p))),
        None => {}
    }
    match service.get("entrypoint") {
        Some(Value::Sequence(parts)) => {
            let parts: Vec<String> = parts.iter().map(py_scalar).collect();
            out.push(format!("ENTRYPOINT {}", py_json_list(&parts)));
        }
        Some(e) => out.push(format!("ENTRYPOINT {}", py_json_list(&[py_scalar(e)]))),
        None => {}
    }
    out
}

/// `FROM` and the spaces after it, and the rest of the line.
fn from_parts(line: &str) -> Option<(&str, &str)> {
    let rest = line.strip_prefix("FROM")?;
    let after = rest.trim_start_matches([' ', '\t']);
    (after.len() < rest.len() && !after.is_empty())
        .then(|| (&line[..line.len() - after.len()], after))
}

/// The image after a `--platform=<p>` flag.
fn after_flag(after: &str) -> Option<&str> {
    let (flag, image) = after.strip_prefix("--platform=")?.split_once([' ', '\t'])?;
    let image = image.trim_start_matches([' ', '\t']);
    (!flag.is_empty() && !image.is_empty()).then_some(image)
}

/// Rewrite the lines `f` returns a new body for; returns how many changed.
fn rewrite_lines(content: &str, f: impl Fn(&str) -> Option<String>) -> (String, usize) {
    let mut changed = 0;
    let mut out = String::with_capacity(content.len() + 32);
    for line in content.split_inclusive('\n') {
        let (body, nl) = line.strip_suffix('\n').map_or((line, ""), |b| (b, "\n"));
        match f(body) {
            Some(new) => {
                changed += 1;
                out.push_str(&new);
                out.push_str(nl);
            }
            None => out.push_str(line),
        }
    }
    (out, changed)
}

/// Set `--platform` on the `FROM` lines as Harbor does: replace every
/// existing flag, or, where there is none, add one to every `FROM`.
/// Returns whether a line changed.
pub(crate) fn set_platform(content: &str, platform: &str) -> (String, bool) {
    let (out, n) = rewrite_lines(content, |body| {
        let (head, after) = from_parts(body)?;
        Some(format!(
            "{head}--platform={platform} {}",
            after_flag(after)?
        ))
    });
    if n > 0 {
        return (out, true);
    }
    let (out, n) = rewrite_lines(content, |body| {
        let (_, after) = from_parts(body)?;
        (!after.starts_with("--platform")).then(|| format!("FROM --platform={platform} {after}"))
    });
    (out, n > 0)
}

/// Python's `read_text`: UTF-8 with universal newlines.
fn read_text(path: &Path) -> Result<String, ConvertError> {
    let bytes = fs::read(path).map_err(io_err(path))?;
    let text = String::from_utf8(bytes).map_err(|_| ConvertError::Invalid {
        file: path.to_path_buf(),
        message: "not UTF-8 text".into(),
    })?;
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}

fn write_text(path: &Path, text: &str) -> Result<(), ConvertError> {
    files::write(path, text.as_bytes()).map_err(io_err(path))
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), ConvertError> {
    use std::os::unix::fs::PermissionsExt;
    let mode = fs::metadata(path)
        .map_err(io_err(path))?
        .permissions()
        .mode();
    fs::set_permissions(path, fs::Permissions::from_mode(mode | 0o111)).map_err(io_err(path))
}

#[cfg(not(unix))]
fn make_executable(_: &Path) -> Result<(), ConvertError> {
    Ok(())
}

/// Refuse a path of the task that resolves outside its folder `root`
/// (canonical): a link out of it, or a Compose path with `..`. A task to
/// import can come from anywhere, and what it names is copied into the new
/// task's image.
fn contained(root: &Path, p: &Path) -> Result<(), ConvertError> {
    match fs::canonicalize(p) {
        Ok(real) if real.starts_with(root) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        _ => Err(ConvertError::Invalid {
            file: p.to_path_buf(),
            message: "is outside the task's folder (a link, `..`, or an absolute path); only the task's own files are converted".into(),
        }),
    }
}

/// A Compose build context, refused when it is not a path inside the task.
fn checked_context(ctx: String, compose: &Path) -> Result<String, ConvertError> {
    let p = Path::new(&ctx);
    if p.is_absolute()
        || ctx.starts_with('~')
        || p.components().any(|c| c == std::path::Component::ParentDir)
    {
        return Err(ConvertError::Invalid {
            file: compose.to_path_buf(),
            message: format!(
                "the build context {ctx:?} is outside the task's folder; only the task's own files are converted"
            ),
        });
    }
    Ok(ctx)
}

/// Copy every file under `from` into `to`. Returns the last file named
/// `Dockerfile` copied, as Harbor picks the build's Dockerfile.
fn copy_tree(from: &Path, to: &Path) -> Result<Option<PathBuf>, ConvertError> {
    let mut dockerfile = None;
    for rel in files::list_all(from) {
        let src = files::under(from, &rel);
        let dst = files::under(to, &rel);
        files::copy(&src, &dst).map_err(io_err(&src))?;
        if rel.last().is_some_and(|n| n == "Dockerfile") {
            dockerfile = Some(dst);
        }
    }
    Ok(dockerfile)
}

fn clean_context(ctx: &str) -> String {
    let c = ctx.trim_start_matches("./").trim_end_matches('/');
    if c.is_empty() {
        ".".into()
    } else {
        c.to_string()
    }
}

/// Convert the Terminal-Bench v1 task at `src` into Harbor's split layout
/// at `dst`, which must be empty. Returns notes on what it did.
pub(crate) fn map(src: &Path, dst: &Path, res: &Resources) -> Result<Vec<String>, ConvertError> {
    let mut notes = vec![
        "converted from Terminal-Bench v1 the way Harbor's task migrate does; review the result before relying on it".to_string(),
    ];
    if src.join("solution.yaml").exists() {
        return Err(ConvertError::Invalid {
            file: src.join("solution.yaml"),
            message: "a scripted solution.yaml cannot be converted; rewrite it as solution.sh"
                .into(),
        });
    }
    let root = fs::canonicalize(src).map_err(io_err(src))?;
    // Every file and folder the conversion reads must be the task's own.
    for p in files::list_all(src)
        .iter()
        .map(|rel| files::under(src, rel))
        .chain(["tests", "environment"].map(|d| src.join(d)))
    {
        contained(&root, &p)?;
    }
    let yaml_path = src.join("task.yaml");
    let (cfg, ignored) =
        TaskYaml::parse(&read_text(&yaml_path)?).map_err(|message| ConvertError::Invalid {
            file: yaml_path.clone(),
            message,
        })?;
    if !ignored.is_empty() {
        notes.push(format!(
            "task.yaml settings with no Harbor or task.md equivalent were dropped: {}",
            ignored.join(", ")
        ));
    }
    let env = dst.join("environment");
    let tests = dst.join("tests");
    fs::create_dir_all(&env).map_err(io_err(&env))?;
    fs::create_dir_all(&tests).map_err(io_err(&tests))?;

    // docker-compose.yaml
    let compose_path = src.join("docker-compose.yaml");
    if !compose_path.is_file() {
        return Err(ConvertError::Invalid {
            file: compose_path,
            message: "missing: a Terminal-Bench v1 task needs docker-compose.yaml".into(),
        });
    }
    let mut handled: BTreeSet<String> = ["docker-compose.yaml".to_string()].into();
    let compose: Value =
        serde_yaml_ng::from_str(&read_text(&compose_path)?).map_err(|e| ConvertError::Invalid {
            file: compose_path.clone(),
            message: format!("not valid YAML: {e}"),
        })?;
    let has_services = services(&compose).is_some_and(|s| !s.is_empty());
    if let Some((main_name, main)) = main_service(&compose).filter(|_| has_services) {
        let ctx = checked_context(clean_context(&build_context(&main)), &compose_path)?;
        let dockerfile = if ctx != "." {
            handled.insert(ctx.clone());
            let from = src.join(&ctx);
            contained(&root, &from)?;
            if from.is_dir() {
                copy_tree(&from, &env)?
            } else {
                None
            }
        } else {
            handled.insert("Dockerfile".into());
            let from = src.join("Dockerfile");
            if from.exists() {
                let to = env.join("Dockerfile");
                files::copy(&from, &to).map_err(io_err(&from))?;
                Some(to)
            } else {
                None
            }
        };
        if let Some(df) = &dockerfile {
            let content = read_text(df)?;
            let refers = |pats: &[&str]| pats.iter().any(|p| content.contains(p));
            if refers(&["COPY tests", "COPY ./tests", "ADD tests", "ADD ./tests"])
                && src.join("tests").is_dir()
                && !env.join("tests").exists()
            {
                copy_tree(&src.join("tests"), &env.join("tests"))?;
            }
            if refers(&[
                "COPY solution.sh",
                "COPY ./solution.sh",
                "ADD solution.sh",
                "ADD ./solution.sh",
            ]) && src.join("solution.sh").exists()
                && !env.join("solution.sh").exists()
            {
                let from = src.join("solution.sh");
                files::copy(&from, &env.join("solution.sh")).map_err(io_err(&from))?;
            }
            let mut content = content;
            if let Some(p) = main.get("platform").map(py_scalar) {
                let (c, changed) = set_platform(&content, &p);
                content = c;
                if changed {
                    notes.push(format!("set --platform={p} on the Dockerfile's FROM lines"));
                }
            }
            let additions = dockerfile_additions(&main);
            if !content.ends_with('\n') {
                content.push('\n');
            }
            content.push_str("\n# Mapped from docker-compose.yaml\n");
            content.push_str(&additions.join("\n"));
            content.push('\n');
            let moved: Vec<&String> = additions
                .iter()
                .filter(|a| *a != "ENV TEST_DIR=/tests")
                .collect();
            if !moved.is_empty() {
                let shown: Vec<&str> = moved.iter().map(|s| s.as_str()).collect();
                notes.push(format!(
                    "docker-compose.yaml settings moved into the Dockerfile: {}",
                    shown.join("; ")
                ));
            }
            write_text(df, &content)?;
        }
        if !can_collapse(&compose) {
            let svcs = services(&compose).expect("services checked above");
            for (k, svc) in svcs {
                if py_scalar(k) == main_name {
                    continue;
                }
                let ctx = checked_context(clean_context(&build_context(svc)), &compose_path)?;
                if ctx == "." {
                    continue;
                }
                let from = src.join(&ctx);
                contained(&root, &from)?;
                if from.is_dir() {
                    let to = env.join(&ctx);
                    if !to.exists() {
                        copy_tree(&from, &to)?;
                    }
                    handled.insert(ctx);
                }
            }
            write_harbor_compose(
                &compose,
                &main_name,
                &main,
                &env.join("docker-compose.yaml"),
            )?;
            notes.push(
                "docker-compose.yaml says more than a Dockerfile can (other services, networks, or volumes), so it was rewritten in Harbor's template with the main service named main".into(),
            );
        }
    }
    for h in ["task.yaml", "solution.sh", "run-tests.sh", "tests"] {
        handled.insert(h.into());
    }

    // solution.sh, tests/, run-tests.sh
    let solution = src.join("solution.sh");
    if solution.exists() {
        let to = dst.join("solution/solve.sh");
        files::copy(&solution, &to).map_err(io_err(&solution))?;
        make_executable(&to)?;
    }
    if src.join("tests").is_dir() {
        copy_tree(&src.join("tests"), &tests)?;
    }
    let run_tests = src.join("run-tests.sh");
    if !run_tests.is_file() {
        return Err(ConvertError::Invalid {
            file: run_tests,
            message: "missing: a Terminal-Bench v1 task needs run-tests.sh".into(),
        });
    }
    let mut test_sh = read_text(&run_tests)?;
    if !test_sh.ends_with('\n') {
        test_sh.push('\n');
    }
    test_sh.push_str(REWARD_LOGGING_SUFFIX);
    let test_path = tests.join("test.sh");
    write_text(&test_path, &test_sh.replace("$TEST_DIR", "/tests"))?;
    make_executable(&test_path)?;

    // Everything else goes into the image's build context.
    for rel in files::list_all(src) {
        if handled.contains(&rel[0]) {
            continue;
        }
        let from = files::under(src, &rel);
        let to = files::under(&env, &rel);
        files::copy(&from, &to).map_err(io_err(&from))?;
    }

    let mut instruction = cfg.instruction.clone();
    if !instruction.ends_with('\n') {
        instruction.push('\n');
    }
    write_text(&dst.join("instruction.md"), &instruction)?;
    write_text(&dst.join("task.toml"), &cfg.to_task_toml(res))?;
    Ok(notes)
}

/// Harbor's main-service template.
fn harbor_main_service() -> Mapping {
    serde_yaml_ng::from_str(
        r#"
build: {context: "${CONTEXT_DIR}"}
image: "${MAIN_IMAGE_NAME}"
command: [sh, -c, sleep infinity]
environment: ["TEST_DIR=${TEST_DIR}"]
deploy: {resources: {limits: {cpus: "${CPUS}", memory: "${MEMORY}"}}}
"#,
    )
    .expect("the template parses")
}

/// `docker-compose.yaml` in Harbor's template: the main service becomes
/// `main` with Harbor's build, image, command, environment, and limits.
fn write_harbor_compose(
    compose: &Value,
    main_name: &str,
    main: &Value,
    to: &Path,
) -> Result<(), ConvertError> {
    let mut out = Mapping::new();
    out.insert("services".into(), Value::Mapping(Mapping::new()));
    if let Some(top) = compose.as_mapping() {
        for (k, v) in top {
            if !matches!(k.as_str(), Some("services" | "version")) {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    let mut svcs = Mapping::new();
    let mut harbor_main = harbor_main_service();
    if let Some(m) = main.as_mapping() {
        for (k, v) in m {
            if !k.as_str().is_some_and(|k| HARBOR_REPLACES.contains(&k)) {
                harbor_main.insert(k.clone(), v.clone());
            }
        }
    }
    svcs.insert("main".into(), Value::Mapping(harbor_main));
    if let Some(all) = services(compose) {
        for (k, v) in all {
            if py_scalar(k) != main_name {
                svcs.insert(k.clone(), v.clone());
            }
        }
    }
    out.insert("services".into(), Value::Mapping(svcs));
    let text =
        serde_yaml_ng::to_string(&Value::Mapping(out)).map_err(|e| ConvertError::Invalid {
            file: to.to_path_buf(),
            message: e.to_string(),
        })?;
    write_text(
        to,
        &text.replace("${T_BENCH_TASK_DOCKER_NAME_PREFIX}", "hb"),
    )
}

#[cfg(test)]
mod tests {
    //! Ported from Harbor's `tests/unit/mappers/test_terminal_bench.py`.

    use super::*;

    fn yaml(s: &str) -> Value {
        serde_yaml_ng::from_str(s).unwrap()
    }

    #[test]
    fn minimal_config() {
        let (c, _) = TaskYaml::parse("instruction: Do something").unwrap();
        assert_eq!(c.instruction, "Do something");
        assert_eq!(c.difficulty, "unknown");
        assert_eq!(c.author_name, "unknown");
        assert_eq!(c.category, "software_engineering");
        assert!(c.tags.is_empty());
        assert_eq!(c.max_agent_timeout_sec, 360.0);
        assert_eq!(c.max_test_timeout_sec, 60.0);
    }

    #[test]
    fn full_config() {
        let (c, _) = TaskYaml::parse(
            "instruction: Do something complex\nauthor_name: John Doe\nauthor_email: john@example.com\ndifficulty: hard\ncategory: devops\ntags: [docker, networking]\nestimated_duration_sec: 120\nmax_agent_timeout_sec: 600\nmax_test_timeout_sec: 120.0\n",
        )
        .unwrap();
        assert_eq!(c.author_name, "John Doe");
        assert_eq!(c.difficulty, "hard");
        assert_eq!(c.category, "devops");
        assert_eq!(c.tags, ["docker", "networking"]);
        assert_eq!(c.estimated_duration_sec, Some(120.0));
        assert_eq!(c.max_agent_timeout_sec, 600.0);
        assert_eq!(c.max_test_timeout_sec, 120.0);
    }

    #[test]
    fn ignores_unknown_fields_and_names_them() {
        let (c, ignored) =
            TaskYaml::parse("instruction: Task\ndifficulty: medium\nsome_random_field: value\nanother_unknown: 123\n").unwrap();
        assert_eq!(c.instruction, "Task");
        assert_eq!(ignored, ["some_random_field", "another_unknown"]);
    }

    #[test]
    fn null_settings_take_defaults_and_bad_difficulty_is_refused() {
        let (c, _) =
            TaskYaml::parse("instruction: T\nexpert_time_estimate_min:\ndifficulty:\n").unwrap();
        assert_eq!(c.expert_time_estimate_min, None);
        assert_eq!(c.difficulty, "unknown");
        assert!(
            TaskYaml::parse("instruction: T\ndifficulty: brutal\n")
                .unwrap_err()
                .contains("brutal")
        );
        assert!(
            TaskYaml::parse("difficulty: easy\n")
                .unwrap_err()
                .contains("instruction")
        );
    }

    #[test]
    fn to_task_toml_carries_timeouts_and_overrides() {
        let (c, _) = TaskYaml::parse(
            "instruction: Test\nmax_agent_timeout_sec: 600\nmax_test_timeout_sec: 120\n",
        )
        .unwrap();
        let t = c.to_task_toml(&Resources::default());
        let v: toml::Table = t.parse().unwrap();
        assert_eq!(v["verifier"]["timeout_sec"].as_float(), Some(120.0));
        assert_eq!(v["agent"]["timeout_sec"].as_float(), Some(600.0));
        let t = c.to_task_toml(&Resources {
            cpus: Some(4),
            memory_mb: Some(8192),
            storage_mb: Some(20480),
            gpus: Some(1),
        });
        for line in [
            "cpus = 4",
            "memory_mb = 8192",
            "storage_mb = 20480",
            "gpus = 1",
        ] {
            assert!(t.contains(line), "{line} in {t}");
        }
    }

    #[test]
    fn python_number_and_string_forms() {
        assert_eq!(py_float(180.0), "180.0");
        assert_eq!(py_float(0.5), "0.5");
        assert_eq!(py_float(1e20), "1e+20");
        assert_eq!(toml_str("a \"b\" \\ c"), r#""a \"b\" \\ c""#);
        let e_acute = char::from_u32(0xe9).unwrap().to_string();
        assert_eq!(
            py_json_list(&["a".into(), e_acute]),
            format!("[\"a\", \"{}u00e9\"]", '\\')
        );
    }

    #[test]
    fn collapse_rules() {
        let yes = [
            "services: {client: {build: {context: .}}}",
            "services: {client: {build: ., environment: [FOO=bar]}}",
            "services: {client: {volumes: ['${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}', '${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}']}}",
            "services: {client: {volumes: ['${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}']}}",
            "services: {client: {build: {dockerfile: Dockerfile}, image: '${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}', container_name: x, command: [sh, -c, sleep infinity], environment: ['TEST_DIR=${T_BENCH_TEST_DIR}'], volumes: ['${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}']}}",
            "services: {client: {build: ., platform: linux/amd64}}",
        ];
        for y in yes {
            assert!(can_collapse(&yaml(y)), "{y}");
        }
        let no = [
            "services: {web: {}, db: {}}",
            "services: {client: {networks: [foo]}}",
            "services: {client: {volumes: ['/data:/data']}}",
            "services: {client: {volumes: ['${CUSTOM_VAR}:/data']}}",
            "services: {client: {}}\nnetworks: {foo: {}}",
        ];
        for y in no {
            assert!(!can_collapse(&yaml(y)), "{y}");
        }
    }

    #[test]
    fn main_service_and_build_context() {
        let (n, s) = main_service(&yaml("services: {web: {}, client: {foo: bar}}")).unwrap();
        assert_eq!((n.as_str(), s), ("client", yaml("{foo: bar}")));
        let (n, s) = main_service(&yaml("services: {web: {x: 1}}")).unwrap();
        assert_eq!((n.as_str(), s), ("web", yaml("{x: 1}")));
        assert_eq!(
            build_context(&yaml("{build: {context: ./subdir}}")),
            "./subdir"
        );
        assert_eq!(build_context(&yaml("{build: ./subdir}")), "./subdir");
        assert_eq!(build_context(&yaml("{}")), ".");
    }

    #[test]
    fn dockerfile_lines_from_a_service() {
        let a = dockerfile_additions(&yaml(
            "{environment: [FOO=bar, BAZ=qux], working_dir: /app, expose: [8080, 9090], entrypoint: [python, -m, app]}",
        ));
        for line in [
            "ENV FOO=bar",
            "ENV BAZ=qux",
            "WORKDIR /app",
            "EXPOSE 8080",
            "EXPOSE 9090",
            r#"ENTRYPOINT ["python", "-m", "app"]"#,
        ] {
            assert!(a.contains(&line.to_string()), "{line} in {a:?}");
        }
        assert_eq!(
            dockerfile_additions(&yaml(
                "{environment: ['TEST_DIR=${T_BENCH_TEST_DIR}'], working_dir: '${X}'}"
            )),
            ["ENV TEST_DIR=/tests"]
        );
        assert_eq!(
            dockerfile_additions(&yaml("{environment: {A: 1, B: true}}")),
            ["ENV TEST_DIR=/tests", "ENV A=1", "ENV B=True"]
        );
    }

    #[test]
    fn platform_on_from_lines() {
        let (c, changed) = set_platform("FROM ubuntu:22.04\nRUN apt-get update\n", "linux/amd64");
        assert!(changed);
        assert_eq!(
            c,
            "FROM --platform=linux/amd64 ubuntu:22.04\nRUN apt-get update\n"
        );
        let (c, _) = set_platform(
            "FROM --platform=linux/arm64 ubuntu:22.04\nRUN apt-get update\n",
            "linux/amd64",
        );
        assert_eq!(
            c,
            "FROM --platform=linux/amd64 ubuntu:22.04\nRUN apt-get update\n"
        );
        assert_eq!(c.matches("--platform").count(), 1);
        let (c, changed) = set_platform("RUN x\n", "linux/amd64");
        assert_eq!((c.as_str(), changed), ("RUN x\n", false));
    }

    fn task(dir: &Path, compose: &str) {
        files::write(
            &dir.join("task.yaml"),
            b"instruction: Test\ndifficulty: easy\n",
        )
        .unwrap();
        files::write(&dir.join("docker-compose.yaml"), compose.as_bytes()).unwrap();
        files::write(&dir.join("Dockerfile"), b"FROM python:3.13\n").unwrap();
        files::write(
            &dir.join("run-tests.sh"),
            b"#!/bin/bash\npytest $TEST_DIR/test_outputs.py\n",
        )
        .unwrap();
    }

    /// Guards f3b25d7a (Terminal-Bench v1 import): Compose build contexts
    /// and links were followed out of the task being converted.
    #[test]
    fn only_the_tasks_own_files_are_converted() {
        let w = tempfile::tempdir().unwrap();
        let outside = w.path().join("outside");
        files::write(&outside.join("Dockerfile"), b"FROM scratch\n").unwrap();
        files::write(&outside.join("private.txt"), b"x").unwrap();
        for (i, ctx) in ["../outside", "/etc", "a/../../outside"].iter().enumerate() {
            let (src, dst) = (
                w.path().join(format!("s{i}")),
                w.path().join(format!("t{i}")),
            );
            task(
                &src,
                &format!(
                    "services:\n  client:\n    build:\n      context: {ctx}\n  db:\n    image: x\n"
                ),
            );
            let e = map(&src, &dst, &Resources::default()).unwrap_err();
            assert!(
                e.to_string().contains("outside the task's folder"),
                "{ctx}: {e}"
            );
            assert!(!dst.join("environment/Dockerfile").exists());
        }
        let (src, dst) = (w.path().join("s-side"), w.path().join("t-side"));
        task(
            &src,
            "services:\n  client:\n    build: .\n  side:\n    build:\n      context: ../outside\n    networks: [x]\nnetworks: {x: {}}\n",
        );
        let e = map(&src, &dst, &Resources::default()).unwrap_err();
        assert!(e.to_string().contains("outside the task's folder"), "{e}");
        assert!(!w.path().join("t-side/outside").exists());
    }

    #[cfg(unix)]
    #[test]
    fn links_out_of_the_task_are_refused() {
        use std::os::unix::fs::symlink;
        let w = tempfile::tempdir().unwrap();
        let outside = w.path().join("outside");
        files::write(&outside.join("private.txt"), b"x").unwrap();
        let cases: [(&str, PathBuf); 3] = [
            ("data.txt", outside.join("private.txt")),
            ("tests", outside.clone()),
            ("ctx", outside.clone()),
        ];
        for (i, (name, target)) in cases.iter().enumerate() {
            let (src, dst) = (
                w.path().join(format!("s{i}")),
                w.path().join(format!("t{i}")),
            );
            let compose = if *name == "ctx" {
                "services:\n  client:\n    build:\n      context: ctx\n"
            } else {
                "services:\n  client:\n    build: .\n"
            };
            task(&src, compose);
            if *name == "tests" {
                std::fs::remove_dir_all(src.join("tests")).ok();
            }
            symlink(target, src.join(name)).unwrap();
            let e = map(&src, &dst, &Resources::default()).unwrap_err();
            assert!(
                e.to_string().contains("outside the task's folder"),
                "{name}: {e}"
            );
        }
        // A link that stays inside the task is still followed.
        let (src, dst) = (w.path().join("s-in"), w.path().join("t-in"));
        task(&src, "services:\n  client:\n    build: .\n");
        files::write(&src.join("real.txt"), b"y").unwrap();
        symlink(src.join("real.txt"), src.join("alias.txt")).unwrap();
        map(&src, &dst, &Resources::default()).unwrap();
        assert!(dst.join("environment/alias.txt").exists());
    }

    #[test]
    fn solution_yaml_is_refused() {
        let w = tempfile::tempdir().unwrap();
        let (src, dst) = (w.path().join("s"), w.path().join("t"));
        task(&src, "services: {client: {build: {context: .}}}");
        files::write(&src.join("solution.yaml"), b"- command: echo hello\n").unwrap();
        let e = map(&src, &dst, &Resources::default())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("solution.yaml") && e.contains("solution.sh"),
            "{e}"
        );
    }

    #[test]
    fn simple_compose_collapses_into_the_dockerfile() {
        let w = tempfile::tempdir().unwrap();
        let (src, dst) = (w.path().join("s"), w.path().join("t"));
        task(
            &src,
            "services: {client: {build: {context: .}, environment: [MY_VAR=hello], working_dir: /app}}",
        );
        files::write(&src.join("solution.sh"), b"#!/bin/bash\necho done\n").unwrap();
        map(&src, &dst, &Resources::default()).unwrap();
        assert!(!dst.join("environment/docker-compose.yaml").exists());
        let df = fs::read_to_string(dst.join("environment/Dockerfile")).unwrap();
        assert!(
            df.contains("ENV MY_VAR=hello") && df.contains("WORKDIR /app"),
            "{df}"
        );
    }

    #[test]
    fn full_task_with_a_compose_that_cannot_collapse() {
        let w = tempfile::tempdir().unwrap();
        let (src, dst) = (w.path().join("s"), w.path().join("t"));
        task(
            &src,
            "services: {client: {build: {context: .}, volumes: ['/data:/data']}}",
        );
        files::write(&src.join("task.yaml"), b"instruction: Write a hello world program\ndifficulty: easy\ncategory: programming\ntags: [python, beginner]\nmax_agent_timeout_sec: 300\nmax_test_timeout_sec: 30\n").unwrap();
        files::write(
            &src.join("solution.sh"),
            b"#!/bin/bash\necho hi > hello.py\n",
        )
        .unwrap();
        files::write(&src.join("tests/test_outputs.py"), b"def test(): pass\n").unwrap();
        files::write(&src.join("data/extra.txt"), b"x").unwrap();
        map(&src, &dst, &Resources::default()).unwrap();
        assert!(
            fs::read_to_string(dst.join("instruction.md"))
                .unwrap()
                .contains("Write a hello world")
        );
        assert!(
            fs::read_to_string(dst.join("task.toml"))
                .unwrap()
                .contains("timeout_sec")
        );
        assert!(dst.join("solution/solve.sh").exists());
        let t = fs::read_to_string(dst.join("tests/test.sh")).unwrap();
        assert!(
            t.contains("_EXIT_CODE=$?") && t.contains("pytest /tests/test_outputs.py"),
            "{t}"
        );
        assert!(dst.join("tests/test_outputs.py").exists());
        assert!(dst.join("environment/data/extra.txt").exists());
        let generated =
            yaml(&fs::read_to_string(dst.join("environment/docker-compose.yaml")).unwrap());
        let main = generated["services"]["main"].as_mapping().unwrap();
        assert!(!main.contains_key("volumes"));
        for k in ["build", "image", "command", "environment", "deploy"] {
            assert!(main.contains_key(k), "{k}");
        }
    }

    #[test]
    fn reward_logging_is_appended() {
        let w = tempfile::tempdir().unwrap();
        let (src, dst) = (w.path().join("s"), w.path().join("t"));
        task(&src, "services: {client: {build: .}}");
        map(&src, &dst, &Resources::default()).unwrap();
        let t = fs::read_to_string(dst.join("tests/test.sh")).unwrap();
        assert!(t.starts_with("#!/bin/bash\n"));
        for s in [
            "_EXIT_CODE=$?",
            "echo 1 > /logs/verifier/reward.txt",
            "echo 0 > /logs/verifier/reward.txt",
            "exit $_EXIT_CODE",
        ] {
            assert!(t.contains(s), "{s}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn scripts_are_executable() {
        use std::os::unix::fs::PermissionsExt;
        let w = tempfile::tempdir().unwrap();
        let (src, dst) = (w.path().join("s"), w.path().join("t"));
        task(&src, "services: {client: {build: {context: .}}}");
        files::write(&src.join("solution.sh"), b"#!/bin/bash\necho done\n").unwrap();
        map(&src, &dst, &Resources::default()).unwrap();
        for f in ["tests/test.sh", "solution/solve.sh"] {
            let mode = fs::metadata(dst.join(f)).unwrap().permissions().mode();
            assert!(mode & 0o111 != 0, "{f} is executable");
        }
    }

    #[test]
    fn discovers_tbench_tasks_but_not_other_formats() {
        let w = tempfile::tempdir().unwrap();
        task(&w.path().join("tasks/a"), "services: {client: {}}");
        task(&w.path().join("tasks/b"), "services: {client: {}}");
        files::write(&w.path().join("tasks/c/task.toml"), b"").unwrap();
        files::write(&w.path().join("tasks/c/task.yaml"), b"instruction: x").unwrap();
        let found = discover_tbench(w.path());
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(discover_tbench(&w.path().join("tasks/a")).len(), 1);
    }
}
