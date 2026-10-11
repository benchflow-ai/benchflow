//! Preparing Daytona images.
//!
//! Daytona builds each sandbox's image itself, from a Dockerfile plus
//! context archives in its object storage, and keeps the result keyed by a
//! hash of both, so sandboxes with the same build start from the cached
//! image. BenchFlow sends:
//!
//! - the task's Dockerfile (or `FROM <image>` for a prebuilt image), then
//!   each agent layer as a `RUN` as root, then `USER root`: Daytona's
//!   in-sandbox daemon must start as root (an image ending in a non-root
//!   `USER` starts but cannot be reached), so the task's own final user is
//!   remembered and commands run as that user by default instead;
//! - the whole build context as one tar, `.dockerignore` honored, uploaded
//!   once under its content hash. (Daytona's SDK uploads only the paths its
//!   own parser finds in `COPY` lines, which misses `ADD`, continued lines,
//!   and `.dockerignore`; one archive of the whole context builds exactly
//!   what `docker build` would.)
//!
//! `prepare` then starts one small "builder" sandbox from the recipe and
//! deletes it once it runs. A broken Dockerfile thus fails once, in
//! `prepare`, with its build log, instead of once per trial, and every
//! trial's sandbox starts from the cached image.

use std::io::{Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::api::{Api, Retry};
use super::dockerignore::DockerIgnore;
use super::s3::Bucket;
use super::types::PushAccess;
use crate::shell::{SYSTEM_PATH, login_free_shell};
use crate::{SandboxError, archive};

/// Where a Daytona image built by BenchFlow keeps the image's own `PATH`
/// and loader variables, one file per name, root's to write. The recipe's
/// last layer records them there and neutralizes them in the container
/// (`PATH` = [`SYSTEM_PATH`], the loader variables empty): Daytona's daemon
/// (root) looks programs up through the container's `PATH`, and its shells
/// read and load what the container's `BASH_ENV`, `LD_PRELOAD`, ... name,
/// for every command, and an image may point any of these at a folder the
/// agent can write. The command wrapper reads these files back and gives
/// each command the image's values (see [`super::toolbox::wrap`]).
pub(crate) const IMAGE_ENV_DIR: &str = "/etc/benchflow/image-env";

/// The names the layer records and neutralizes: `PATH` and every loader
/// variable. A space-separated list for the recipe's `for` loop and the
/// wrapper's.
pub(crate) fn recorded_env_names() -> String {
    let mut names = vec!["PATH"];
    names.extend_from_slice(crate::shell::LOADER_VARS);
    names.join(" ")
}

/// The recipe's last lines: record the image's `PATH` and loader variables
/// in [`IMAGE_ENV_DIR`] (one file each, only those that are set), then
/// neutralize them in the container.
fn system_env_layer() -> String {
    let names = recorded_env_names();
    // `read`-free capture: one file per name, written only when the
    // variable is set, so the wrapper can tell "the image set it" from
    // "the image did not". `mkdir` and `chmod` are named by absolute path:
    // the step runs before `ENV PATH=` below, so the image's own `PATH`
    // (which may omit `/bin`) must not be relied on to find them.
    let record = format!(
        "/bin/mkdir -p {IMAGE_ENV_DIR} && for bf_v in {names}; do \
         eval \"bf_set=\\${{$bf_v+x}}; bf_val=\\${{$bf_v-}}\"; \
         if [ -n \"$bf_set\" ]; then printf '%s' \"$bf_val\" > {IMAGE_ENV_DIR}/$bf_v; fi; \
         done && /bin/chmod -R a-w,a+rX {IMAGE_ENV_DIR}"
    );
    let argv = serde_json::json!(["/bin/sh", "-c", record]);
    // The container's `PATH` is the system folders, and the file-naming
    // loader variables and the shells' start-up files are empty (glibc and
    // the shells treat empty as none). The flags are never set, not even
    // empty, because glibc acts on their presence; they stay as the image
    // left them (normally unset). The command wrapper restores the image's
    // real values from the record.
    let mut env = format!("ENV PATH={SYSTEM_PATH}");
    for v in crate::shell::FILE_LOADER_VARS
        .iter()
        .chain(crate::shell::STARTUP_VARS)
    {
        env.push_str(&format!(" {v}="));
    }
    format!("RUN {argv}\n{env}\n")
}

/// What a sandbox is built from.
#[derive(Debug, Clone)]
pub(crate) struct Recipe {
    /// The Dockerfile Daytona builds.
    pub dockerfile: String,
    pub context_hashes: Vec<String>,
    /// Who commands run as when they name no user: the final `USER` of the
    /// task's Dockerfile. `None` is root.
    pub default_user: Option<String>,
}

impl Recipe {
    /// A recipe from a Dockerfile's text plus agent layers.
    pub(crate) fn new(base: &str, layers: &[String], context_hashes: Vec<String>) -> Self {
        let default_user = final_user(base);
        let mut dockerfile = base.trim_end().to_string();
        dockerfile.push('\n');
        if !layers.is_empty() {
            dockerfile.push_str("USER root\n");
            for layer in layers {
                let argv = serde_json::json!(["/bin/sh", "-c", login_free_shell(layer)]);
                dockerfile.push_str(&format!("RUN {argv}\n"));
            }
        }
        dockerfile.push_str("USER root\n");
        dockerfile.push_str(&system_env_layer());
        Self {
            dockerfile,
            context_hashes,
            default_user,
        }
    }

    /// The recipe under another content key for its `n`th rebuild (`n` 0
    /// is the recipe itself): a comment line, which no build step reads.
    /// Daytona keeps a build that was cancelled and fails every later build
    /// of the same content at once; another key builds afresh.
    pub(crate) fn rebuilt(&self, n: u32) -> Self {
        let mut again = self.clone();
        if n > 0 {
            again.dockerfile.push_str(&format!(
                "# benchflow: rebuild-{n} after a cancelled build\n"
            ));
        }
        again
    }

    /// A content key for this recipe.
    pub(crate) fn key(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.dockerfile.as_bytes());
        for hash in &self.context_hashes {
            h.update(b"\0");
            h.update(hash.as_bytes());
        }
        hex::encode(h.finalize())[..16].to_string()
    }
}

/// The user the final stage of `dockerfile` leaves set, unless it is root
/// or cannot be known here (it uses a build variable).
pub(crate) fn final_user(dockerfile: &str) -> Option<String> {
    let mut user: Option<String> = None;
    for line in logical_lines(dockerfile) {
        let mut words = line.split_whitespace();
        let Some(instruction) = words.next() else {
            continue;
        };
        match instruction.to_ascii_uppercase().as_str() {
            "FROM" => user = None,
            "USER" => {
                let value = words.next().unwrap_or("").trim();
                let name = value.split(':').next().unwrap_or("").to_string();
                user = (!name.is_empty()).then_some(name);
            }
            _ => {}
        }
    }
    match user {
        Some(u) if u == "root" || u == "0" => None,
        Some(u) if u.contains('$') => {
            tracing::warn!(
                "the Dockerfile's final USER {u} uses a build variable; on Daytona, commands that \
                 name no user run as root"
            );
            None
        }
        other => other,
    }
}

/// A Dockerfile's instructions, one per entry: comments dropped and
/// backslash-continued lines joined.
fn logical_lines(dockerfile: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for raw in dockerfile.lines() {
        let trimmed = raw.trim();
        if current.is_empty() && (trimmed.is_empty() || trimmed.starts_with('#')) {
            continue;
        }
        if let Some(head) = trimmed.strip_suffix('\\') {
            current.push_str(head);
            current.push(' ');
        } else {
            current.push_str(trimmed);
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

/// Pack `context` (minus `.dockerignore`d paths), upload it unless Daytona
/// already has it, and return its content hash.
pub(crate) async fn upload_context(
    api: &Api,
    http: reqwest::Client,
    context: &Path,
    dockerfile: &Path,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<String, SandboxError> {
    let ignore = DockerIgnore::load(context, dockerfile);
    let root = context.to_path_buf();
    let packed = tokio::task::spawn_blocking(move || -> std::io::Result<_> {
        let mut file = tempfile::tempfile()?;
        let packed = archive::pack_filtered(&root, &ignore, &mut file)?;
        file.seek(SeekFrom::Start(0))?;
        let len = file.metadata()?.len();
        Ok((file, packed, len))
    })
    .await
    .map_err(|e| SandboxError::io("packing the build context", std::io::Error::other(e)))?;
    let (file, packed, len) = packed.map_err(|e| SandboxError::Image {
        name: String::new(),
        message: format!(
            "the build context {} cannot be read ({e}). This is a problem with the task's files; \
             check that the folder exists and is readable.",
            context.display()
        ),
        log_tail: String::new(),
        task_fault: true,
    })?;
    crate::transfer::report_skipped(
        &format!("build context {}", context.display()),
        &packed.skipped,
    );
    let hash = packed.digest[..32].to_string();

    let access: PushAccess = api
        .json(
            "getting upload access for the build context",
            Retry::Safe,
            |http| {
                Ok(http
                    .get(api.url("/object-storage/push-access"))
                    .timeout(Duration::from_secs(60)))
            },
        )
        .await?;
    let key = format!("{}/{hash}/context.tar", access.organization_id);
    let bucket = Bucket::new(access, http);
    if bucket.exists(&key).await {
        log(&format!("build context {hash} is already on Daytona"));
    } else {
        log(&format!(
            "uploading the build context: {} files, {} bytes",
            packed.files, packed.bytes
        ));
        bucket.put(&key, &file, len).await?;
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_run_as_root_and_the_image_always_ends_as_root() {
        let r = Recipe::new(
            "FROM ubuntu:24.04\nRUN useradd agent\nUSER agent\n",
            &["npm i -g tool".to_string()],
            vec![],
        );
        let lines: Vec<&str> = r.dockerfile.lines().collect();
        assert_eq!(lines[2], "USER agent");
        assert_eq!(lines[3], "USER root");
        assert!(lines[4].starts_with(r#"RUN ["/bin/sh","-c","#));
        assert_eq!(lines[5], "USER root");
        assert_eq!(r.default_user.as_deref(), Some("agent"));
    }

    #[test]
    fn a_prebuilt_image_gets_a_from_line() {
        let r = Recipe::new("FROM ubuntu:24.04\n", &[], vec![]);
        assert_eq!(
            r.dockerfile,
            format!("FROM ubuntu:24.04\nUSER root\n{}", system_env_layer())
        );
        assert_eq!(r.default_user, None);
    }

    /// The image ends with the system folders as its `PATH` and empty
    /// loader variables, so Daytona's daemon never looks a program up in,
    /// loads a library from, or reads a start-up file in a folder of the
    /// task's; the image's own values are recorded first, for the commands.
    #[test]
    fn the_image_records_its_env_then_neutralizes_it() {
        let r = Recipe::new(
            "FROM x\nENV PATH=/app/.venv/bin:$PATH BASH_ENV=/app/.venv/env.sh\n",
            &[],
            vec![],
        );
        let lines: Vec<&str> = r.dockerfile.lines().collect();
        let env = lines[lines.len() - 1];
        assert!(
            env.starts_with(&format!("ENV PATH={SYSTEM_PATH} ")),
            "{env}"
        );
        assert!(env.contains(" LD_PRELOAD= "), "{env}");
        assert!(env.ends_with(" ENV="), "{env}");
        // A flag loader variable is never set, not even empty: glibc acts on
        // its presence (an empty `LD_SHOW_AUXV` prints the auxiliary vector
        // from every program).
        for flag in crate::shell::FLAG_LOADER_VARS {
            assert!(
                !env.split(' ').any(|w| w.starts_with(&format!("{flag}="))),
                "{flag} in the layer's ENV line: {env}"
            );
        }
        let record = lines[lines.len() - 2];
        assert!(record.starts_with(r#"RUN ["/bin/sh","-c","#), "{record}");
        assert!(record.contains(IMAGE_ENV_DIR), "{record}");
        // The recording step is valid sh and writes each set variable's
        // value, and nothing for an unset one.
        let argv: Vec<String> = serde_json::from_str(record.strip_prefix("RUN ").unwrap()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("etc-benchflow");
        let script = argv[2].replace("/etc/benchflow", root.to_str().unwrap());
        let out = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(&script)
            .env("PATH", "/opt/x/bin:/usr/bin:/bin")
            .env("BASH_ENV", "/app/.venv/env.sh")
            .env_remove("LD_PRELOAD")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let got = |n: &str| std::fs::read_to_string(root.join("image-env").join(n)).ok();
        assert_eq!(got("PATH").as_deref(), Some("/opt/x/bin:/usr/bin:/bin"));
        assert_eq!(got("BASH_ENV").as_deref(), Some("/app/.venv/env.sh"));
        assert_eq!(got("LD_PRELOAD"), None, "an unset variable is not recorded");
    }

    #[test]
    fn the_final_stage_decides_the_user() {
        assert_eq!(
            final_user("FROM a AS build\nUSER builder\nFROM b\nRUN x\n"),
            None
        );
        assert_eq!(final_user("FROM b\nUSER app:app\n").as_deref(), Some("app"));
        assert_eq!(final_user("FROM b\nuser 1000\n").as_deref(), Some("1000"));
        assert_eq!(final_user("FROM b\nUSER agent\nUSER root\n"), None);
        assert_eq!(final_user("FROM b\nUSER $NAME\n"), None);
        assert_eq!(
            final_user("FROM b\n# USER commented\nRUN a \\\n  && b\n"),
            None
        );
    }

    #[test]
    fn keys_change_with_dockerfile_and_context() {
        let a = Recipe::new("FROM x\n", &[], vec!["h1".into()]);
        let b = Recipe::new("FROM x\n", &[], vec!["h2".into()]);
        let c = Recipe::new("FROM y\n", &[], vec!["h1".into()]);
        assert_ne!(a.key(), b.key());
        assert_ne!(a.key(), c.key());
        assert_eq!(
            a.key(),
            Recipe::new("FROM x\n", &[], vec!["h1".into()]).key()
        );
    }
}
