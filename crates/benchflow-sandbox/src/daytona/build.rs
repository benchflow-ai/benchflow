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
use crate::shell::login_free_shell;
use crate::{SandboxError, archive};

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
        Self {
            dockerfile,
            context_hashes,
            default_user,
        }
    }

    /// The same recipe under a new content key: a label no build reads.
    /// Daytona keeps a build that was canceled (a run stopped while it
    /// built) and answers every later build of the same content with
    /// `context canceled`; a relabeled recipe builds afresh.
    pub(crate) fn relabeled(&self, salt: &str) -> Self {
        let mut again = self.clone();
        again
            .dockerfile
            .push_str(&format!("LABEL ai.benchflow.rebuild=\"{salt}\"\n"));
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
        assert_eq!(*lines.last().unwrap(), "USER root");
        assert_eq!(r.default_user.as_deref(), Some("agent"));
    }

    #[test]
    fn a_prebuilt_image_gets_a_from_line() {
        let r = Recipe::new("FROM ubuntu:24.04\n", &[], vec![]);
        assert_eq!(r.dockerfile, "FROM ubuntu:24.04\nUSER root\n");
        assert_eq!(r.default_user, None);
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
