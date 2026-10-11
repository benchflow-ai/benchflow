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
//! | [`Sandbox::destroy`] | `docker rm -f` |
//! | [`Provider::sweep`] | `docker ps -a` by owner label, `docker rm -f` the dead process's containers |
//!
//! Environment variables never appear on a command line: they travel in a
//! private `--env-file` that is deleted right after the CLI read it.

mod cache;
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
        Ok(containers + services::sweep_networks(&self.cli).await)
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
            "{{.State.Running}}".to_string(),
            self.name.clone(),
        ];
        let out = self.cli.run(&args, Some(Duration::from_secs(30))).await?;
        if out.ok() {
            return Ok(if out.stdout.trim() == "true" {
                crate::Presence::Running
            } else {
                crate::Presence::Stopped
            });
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
