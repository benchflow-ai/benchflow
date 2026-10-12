//! The docker/podman CLI backend.
//!
//! BenchFlow shells out to the `docker` CLI (or `podman`, which speaks the
//! same commands) instead of calling the Docker API, so the user's contexts,
//! `DOCKER_HOST`, and registry logins apply with no setup. One sandbox is one
//! container; compose is not used.
//!
//! | Operation | What runs |
//! |---|---|
//! | [`Provider::status`] | `docker version --format '{{json .}}'` |
//! | [`Provider::prepare`] | `docker image inspect`, then `docker build` or `docker pull`; agent layers as a second build `FROM` the first |
//! | [`Provider::create`] | `docker run -d --init ... --entrypoint sh <image> -c '<wait loop>'` |
//! | [`Sandbox::exec`], [`Sandbox::spawn`] | `docker exec -i ... sh -c '<command>'` |
//! | [`Sandbox::upload`], [`Sandbox::write_file`] | a host-built tar into `docker exec -i -u 0 ... tar -xf -` |
//! | [`Sandbox::download`] | `docker exec -u 0 ... tar -cf -`, unpacked defensively on the host |
//! | [`Sandbox::presence`] | `docker inspect`; for a container that stopped, `docker events` since it started (stopped from inside only when it died and was not killed) |
//! | [`Sandbox::destroy`] | `docker rm -f` |
//! | [`Provider::sweep`] | `docker ps -a` by owner label, `docker rm -f` the dead process's containers; `docker rmi -f` stale checkpoint images |
//! | [`Sandbox::checkpoint`] | `docker inspect` (no mounts), `docker commit --change 'LABEL …' --change 'ENV NAME='` |
//! | [`Provider::create_from_checkpoint`] | `docker run --pull never` of the checkpoint image, as `create` |
//!
//! Environment variables never appear on a command line: they travel in a
//! private `--env-file` that is deleted right after the CLI read it.

mod cache;
mod checkpoint;
mod cli;
mod exec;
mod files;
mod image;
mod run;
mod services;
mod status;
mod sweep;

#[cfg(test)]
mod tests;

pub(crate) use image::repo_name;

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;

use crate::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, Provider, ProviderStatus, Sandbox,
    SandboxError, SandboxSpec,
};
use cli::{Cli, Flavor};

impl DockerSandbox {
    /// For a container that exists but does not run: `Stopped` when the
    /// daemon's events since it `started` show that it died (`die`) and
    /// nobody killed or stopped it, which is what code inside stopping it
    /// (`kill -TERM 1`) logs. `docker stop`, `docker kill` and a daemon
    /// shutting down log `kill` first, and a daemon that restarted has no
    /// events from before: those are lost (an error). So is a Podman
    /// container, whose events BenchFlow does not read.
    async fn who_stopped(&self, started: &str) -> Result<crate::Presence, SandboxError> {
        let lost = |why: String| {
            Err(SandboxError::Transport(format!(
                "container {} stopped, {why}",
                self.name
            )))
        };
        // Podman's events are not read: it cannot say who stopped it.
        if self.cli.flavor != Flavor::Docker || started.is_empty() {
            return Ok(crate::Presence::Halted);
        }
        // A little past now, so a daemon whose clock runs ahead still
        // shows the container's last events; the CLI returns then.
        let until = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() + 2)
            .unwrap_or_default();
        let args: Vec<String> = [
            "events",
            "--since",
            started,
            "--until",
            &until.to_string(),
            "--filter",
            "type=container",
            "--filter",
            &format!("container={}", self.name),
            // Only the daemon's own stop events (filters with one key are
            // ORed). Exec events carry the command's text in their action,
            // which code inside (or BenchFlow's own scripts) can make read
            // `kill` or `stop` on a line of its own (convergence review
            // E9-E-B 1).
            "--filter",
            "event=die",
            "--filter",
            "event=kill",
            "--filter",
            "event=stop",
            "--format",
            "{{.Action}}",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let out = self.cli.run(&args, Some(Duration::from_secs(30))).await?;
        if !out.ok() {
            return Err(self.cli.daemon_problem(&out).unwrap_or_else(|| {
                SandboxError::Transport(format!(
                    "container {} stopped, and `{} events` failed: {}",
                    self.name,
                    self.cli.name,
                    out.message()
                ))
            }));
        }
        let actions: Vec<&str> = out.stdout.lines().map(str::trim).collect();
        let killed = actions.iter().any(|a| matches!(*a, "kill" | "stop"));
        // A `kill` is logged just before its `die`, and `stop` after it, so
        // when `die` is still in the daemon's short history, so is any
        // `kill` or `stop`: a `die` alone is a stop from inside, whether or
        // not the container's `start` is still there (convergence review
        // E9-C-B 1).
        if actions.contains(&"die") && !killed {
            return Ok(crate::Presence::Stopped);
        }
        if killed {
            return lost(format!("and the {} daemon stopped it", self.cli.name));
        }
        // No `die` in the history: it dropped out of the daemon's short
        // history (a busy daemon; code inside can make one busy), or a
        // restart emptied it. Nothing says who stopped it, so it is lost
        // once and capped, never a confirmed loss.
        Ok(crate::Presence::Halted)
    }
}

/// Sandboxes as containers of a local (or remote) Docker or Podman daemon.
pub struct DockerProvider {
    cli: Cli,
}

impl DockerProvider {
    /// A provider for the CLI called `cli` (`docker` or `podman`), found on
    /// `PATH`. A name containing `podman` speaks podman's dialect where the
    /// two differ (GPU flags, version output).
    pub fn new(cli: &str) -> Self {
        let flavor = if cli.contains("podman") {
            Flavor::Podman
        } else {
            Flavor::Docker
        };
        let name = match flavor {
            Flavor::Podman => "podman",
            Flavor::Docker => "docker",
        };
        Self {
            cli: Cli {
                name: name.to_string(),
                binary: PathBuf::from(cli),
                flavor,
            },
        }
    }

    /// Run this executable instead of looking the CLI up on `PATH` (for
    /// tests, or a CLI installed elsewhere).
    pub fn with_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.cli.binary = path.into();
        self
    }
}

#[async_trait]
impl Provider for DockerProvider {
    fn name(&self) -> &str {
        &self.cli.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            checkpoints: true,
            network_none: true,
            gpus: true,
            pids_limit: true,
            build_from_dockerfile: true,
            max_disk_mb: None,
            firewall: true,
            network_allow_cidrs: None,
            network_switch: false,
            root_service: false,
            // --network none removes every interface but loopback.
            network_none_leaks_ipv6: false,
        }
    }

    async fn status(&self) -> ProviderStatus {
        status::status(&self.cli).await
    }

    async fn prepare(
        &self,
        spec: &ImageSpec,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        image::prepare(&self.cli, spec, log).await
    }

    async fn release(&self, image: &Image) {
        cache::release(&image.reference);
    }

    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        let name = run::start(&self.cli, spec).await?;
        Ok(Box::new(DockerSandbox {
            cli: self.cli.clone(),
            name,
        }))
    }

    async fn sweep(&self) -> Result<usize, SandboxError> {
        let containers = sweep::sweep(&self.cli).await?;
        let networks = services::sweep_networks(&self.cli).await;
        Ok(containers + networks + checkpoint::sweep(&self.cli).await)
    }

    async fn create_from_checkpoint(
        &self,
        spec: &SandboxSpec,
        checkpoint: &crate::checkpoint::Checkpoint,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        let mut spec = spec.clone();
        spec.image.reference = checkpoint.reference.clone();
        let name = run::start_from(&self.cli, &spec, true).await?;
        Ok(Box::new(DockerSandbox {
            cli: self.cli.clone(),
            name,
        }))
    }

    async fn delete_checkpoint(
        &self,
        checkpoint: &crate::checkpoint::Checkpoint,
    ) -> Result<bool, SandboxError> {
        checkpoint::delete(&self.cli, &checkpoint.reference).await
    }

    async fn list_checkpoints(&self) -> Result<Vec<crate::checkpoint::Listed>, SandboxError> {
        Ok(checkpoint::list(&self.cli)
            .await?
            .into_iter()
            .map(|s| checkpoint::listed(&self.cli, s))
            .collect())
    }

    fn services(&self) -> Option<&dyn crate::services::ServiceHost> {
        Some(self)
    }

    async fn runtimes(&self) -> Result<Option<Vec<String>>, SandboxError> {
        status::runtimes(&self.cli).await
    }
}

/// An image BenchFlow built on this daemon (labeled with this owner scope).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CachedImage {
    pub id: String,
    pub tags: Vec<String>,
    pub bytes: u64,
    /// The last time a run used it, or when it was built if no use is
    /// recorded.
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub use_recorded: bool,
}

impl DockerProvider {
    /// The images BenchFlow built on this daemon under this owner scope
    /// (`BENCHFLOW_OWNER`), with sizes and last use. Images BenchFlow did not
    /// build carry no owner label and are never listed.
    pub async fn cached_images(&self) -> Result<Vec<CachedImage>, SandboxError> {
        Ok(cache::list_checked(&self.cli)
            .await?
            .into_iter()
            .map(|c| CachedImage {
                id: c.id,
                tags: c.tags,
                bytes: c.size,
                last_used: c.last_used,
                use_recorded: c.used_recorded,
            })
            .collect())
    }

    /// Remove one of [`Self::cached_images`] by its tags, never forced: an
    /// image a container uses stays, and the CLI's refusal is returned.
    pub async fn remove_image(&self, image: &CachedImage) -> Result<(), String> {
        let c = cache::Cached {
            id: image.id.clone(),
            tags: image.tags.clone(),
            size: image.bytes,
            last_used: image.last_used,
            used_recorded: image.use_recorded,
        };
        cache::remove(&self.cli, &c).await
    }

    /// Images used this recently may belong to a run in progress.
    pub const RECENT_USE: Duration = cache::RECENT;
}

/// One running container.
pub struct DockerSandbox {
    cli: Cli,
    /// The container's name, which is also its id for the CLI.
    name: String,
}

impl std::fmt::Debug for DockerSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DockerSandbox")
            .field("backend", &self.cli.name)
            .field("name", &self.name)
            .finish()
    }
}

#[async_trait]
impl Sandbox for DockerSandbox {
    fn id(&self) -> &str {
        &self.name
    }

    fn backend(&self) -> &str {
        &self.cli.name
    }

    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        exec::exec(&self.cli, &self.name, exec).await
    }

    async fn spawn(&self, exec: Exec) -> Result<Process, SandboxError> {
        exec::spawn(&self.cli, &self.name, exec).await
    }

    async fn upload(&self, local: &Path, remote: &str) -> Result<(), SandboxError> {
        files::upload(&self.cli, &self.name, local, remote).await
    }

    async fn write_file(
        &self,
        remote: &str,
        contents: &[u8],
        mode: u32,
    ) -> Result<(), SandboxError> {
        files::write_file(&self.cli, &self.name, remote, contents, mode).await
    }

    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        files::download(&self.cli, &self.name, remote, local, None).await
    }

    async fn presence(&self) -> Result<crate::Presence, SandboxError> {
        let args = [
            "inspect".to_string(),
            "--format".to_string(),
            "{{.State.Running}} {{.State.StartedAt}}".to_string(),
            self.name.clone(),
        ];
        let out = self.cli.run(&args, Some(Duration::from_secs(30))).await?;
        if out.ok() {
            let mut words = out.stdout.split_whitespace();
            if words.next() == Some("true") {
                return Ok(crate::Presence::Running);
            }
            return self.who_stopped(words.next().unwrap_or_default()).await;
        }
        let lower = out.message().to_lowercase();
        if [
            "no such container",
            "no such object",
            "no container with name or id",
        ]
        .iter()
        .any(|m| lower.contains(m))
        {
            return Ok(crate::Presence::Gone);
        }
        Err(self.cli.daemon_problem(&out).unwrap_or_else(|| {
            SandboxError::Transport(format!(
                "`{} inspect {}` failed: {}",
                self.cli.name,
                self.name,
                out.message()
            ))
        }))
    }

    async fn download_capped(
        &self,
        remote: &str,
        local: &Path,
        max_bytes: u64,
    ) -> Result<(), SandboxError> {
        files::download(&self.cli, &self.name, remote, local, Some(max_bytes)).await
    }

    async fn checkpoint(
        &self,
        name: &str,
        scrub_env: &[String],
    ) -> Result<crate::checkpoint::Checkpoint, SandboxError> {
        checkpoint::take(&self.cli, &self.name, name, scrub_env).await
    }

    async fn destroy(&self) -> Result<(), SandboxError> {
        let args = ["rm".to_string(), "-f".to_string(), self.name.clone()];
        let out = self.cli.run(&args, Some(Duration::from_secs(120))).await?;
        if out.ok() || out.message().to_lowercase().contains("no such container") {
            return Ok(());
        }
        Err(self.cli.daemon_problem(&out).unwrap_or_else(|| {
            SandboxError::Transport(format!(
                "could not remove container {}: {}. Remove it with `{} rm -f {}`; `bench` also \
                 sweeps leftovers at the start of the next run.",
                self.name,
                out.message(),
                self.cli.name,
                self.name
            ))
        }))
    }
}
