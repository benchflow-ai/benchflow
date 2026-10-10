//! The Daytona backend: cloud sandboxes over Daytona's REST API.
//!
//! Configuration comes from `DAYTONA_API_KEY` (required to do anything),
//! `DAYTONA_API_URL` (default `https://app.daytona.io/api`) and
//! `DAYTONA_TARGET` (the region; Daytona's default when unset).
//!
//! | Operation | How |
//! |---|---|
//! | [`Provider::status`] | lists running sandboxes (read-only) |
//! | [`Provider::prepare`] | uploads the build context, then starts and deletes one builder sandbox so Daytona builds and caches the image (see [`build`]) |
//! | [`Provider::create`] | `POST /sandbox` with the recipe, resources, labels and network policy; waits until it runs |
//! | [`Sandbox::exec`] | one toolbox session per command, polled to completion (see [`toolbox`]) |
//! | [`Sandbox::spawn`] | a session command: stdin through the input API, output through the follow websocket |
//! | files | one tar through the toolbox files API, unpacked by a root `tar` |
//! | [`Sandbox::destroy`] | `DELETE /sandbox/{id}` |
//! | [`Provider::sweep`] | deletes our sandboxes of dead processes on this host, and our stale ones |
//!
//! A crashed run never leaks sandboxes for long: each sandbox stops after
//! 15 idle minutes and is deleted when it stops, and BenchFlow keeps the
//! sandboxes it holds active with a ping every 3 minutes.

mod api;
mod build;
pub(crate) mod dockerignore;
mod exec;
mod files;
mod lifecycle;
mod process;
mod s3;
mod toolbox;
mod types;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::owner::{self, Verdict};
use crate::{
    Capabilities, Exec, ExecOutput, Image, ImageSource, ImageSpec, NetworkPolicy, OWNER_LABEL,
    Process, Provider, ProviderStatus, Sandbox, SandboxError, SandboxSpec,
};
use api::{Api, ApiError, Retry};
use build::Recipe;
use lifecycle::{ATTEMPT_LABEL, Failure};
use toolbox::{STAGING, Toolbox};
use types::{BuildInfo, CreateSandbox, NetworkSettings, SandboxDto, SandboxPage};

/// Daytona's API when `DAYTONA_API_URL` is unset.
pub const DEFAULT_API_URL: &str = "https://app.daytona.io/api";

/// Daytona gives one sandbox at most this much disk.
const MAX_DISK_GB: u64 = 10;
/// ... this many CPUs (it refuses more with HTTP 400) ...
const MAX_CPUS: u32 = 4;
/// ... and this much memory.
const MAX_MEMORY_GB: u32 = 16;
/// Minutes without activity before Daytona stops a sandbox (then deletes it).
const AUTO_STOP_MINUTES: u32 = 15;
/// How often a held sandbox is marked active.
const KEEPALIVE_EVERY: Duration = Duration::from_secs(180);
/// How long `create` waits for a sandbox to run. Long enough for Daytona to
/// rebuild an image whose cache it dropped.
const START_TIMEOUT: Duration = Duration::from_secs(1200);
/// Build time allowed when the task sets none.
const DEFAULT_BUILD_TIMEOUT: Duration = Duration::from_secs(1800);
/// Image references of recipes this process prepared.
const BUILD_PREFIX: &str = "bf-build:";
/// The toolbox proxy when a sandbox does not name one.
const DEFAULT_TOOLBOX_URL: &str = "https://proxy.app.daytona.io/toolbox";

/// Where the Daytona API is and how to sign in.
#[derive(Clone)]
pub struct DaytonaConfig {
    pub api_key: Option<String>,
    pub api_url: String,
    pub target: Option<String>,
}

impl fmt::Debug for DaytonaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DaytonaConfig")
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("api_url", &self.api_url)
            .field("target", &self.target)
            .finish()
    }
}

impl DaytonaConfig {
    /// `DAYTONA_API_KEY`, `DAYTONA_API_URL`, `DAYTONA_TARGET`.
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        Self {
            api_key: var("DAYTONA_API_KEY"),
            api_url: var("DAYTONA_API_URL").unwrap_or_else(|| DEFAULT_API_URL.to_string()),
            target: var("DAYTONA_TARGET"),
        }
    }
}

/// Sandboxes in Daytona's cloud.
pub struct DaytonaProvider {
    api: Arc<Api>,
    http: reqwest::Client,
    /// Recipes by the image reference `prepare` returned.
    recipes: Mutex<HashMap<String, Recipe>>,
    /// Per recipe key: whether Daytona has built it in this process.
    built: Mutex<HashMap<String, Arc<tokio::sync::Mutex<bool>>>>,
}

impl fmt::Debug for DaytonaProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DaytonaProvider")
            .field("api", &self.api)
            .finish()
    }
}

impl DaytonaProvider {
    /// A provider configured from the environment. A missing key is not an
    /// error here, so `bench doctor` can report it with its fix; every
    /// operation that needs the key fails with that same fix.
    pub fn from_env() -> Result<Self, SandboxError> {
        Self::new(DaytonaConfig::from_env())
    }

    pub fn new(config: DaytonaConfig) -> Result<Self, SandboxError> {
        let api = Api::new(&config.api_url, config.api_key, config.target)?;
        Ok(Self {
            api: Arc::new(api),
            http: reqwest::Client::new(),
            recipes: Mutex::new(HashMap::new()),
            built: Mutex::new(HashMap::new()),
        })
    }

    /// Delete the sandbox with this id, for a sandbox whose handle was lost.
    /// One that is already gone counts as deleted.
    pub async fn delete_sandbox(&self, id: &str) -> Result<(), SandboxError> {
        lifecycle::delete(&self.api, id).await
    }

    fn recipe_for(&self, reference: &str) -> Result<Recipe, SandboxError> {
        if let Some(recipe) = self.recipes.lock().expect("lock").get(reference) {
            return Ok(recipe.clone());
        }
        if reference.starts_with(BUILD_PREFIX) {
            return Err(SandboxError::Unsupported(format!(
                "image {reference} was prepared by another BenchFlow process, and Daytona needs its \
                 recipe in this one; prepare the image in this process first (a bug in the caller)"
            )));
        }
        Ok(Recipe::new(&format!("FROM {reference}\n"), &[], Vec::new()))
    }

    fn build_gate(&self, key: &str) -> Arc<tokio::sync::Mutex<bool>> {
        self.built
            .lock()
            .expect("lock")
            .entry(key.to_string())
            .or_default()
            .clone()
    }

    /// Start a builder sandbox from `recipe`, follow its build, delete it.
    /// True when Daytona built (or staged) the image now.
    async fn build_now(
        &self,
        spec: &ImageSpec,
        recipe: &Recipe,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<bool, SandboxError> {
        let mut labels = owner::labels();
        labels.insert("ai.benchflow.role".into(), "builder".into());
        labels.insert("ai.benchflow.name".into(), spec.name.clone());
        labels.insert(
            ATTEMPT_LABEL.into(),
            uuid::Uuid::new_v4().simple().to_string(),
        );
        let body = CreateSandbox {
            labels,
            target: self.api.target.as_deref(),
            cpu: Some(1),
            memory: Some(1),
            disk: Some(MAX_DISK_GB as u32),
            network_block_all: false,
            network_allow_list: None,
            auto_stop_interval: AUTO_STOP_MINUTES,
            auto_delete_interval: 0,
            build_info: BuildInfo {
                dockerfile_content: &recipe.dockerfile,
                context_hashes: &recipe.context_hashes,
            },
        };
        log("asking Daytona to build the image");
        let dto = lifecycle::create(&self.api, &body)
            .await
            .map_err(|e| name_image_error(e, &spec.name))?;
        let limit = spec.build_timeout.unwrap_or(DEFAULT_BUILD_TIMEOUT) + Duration::from_secs(120);
        let waited = lifecycle::wait_started(&self.api, &dto.id, limit, log).await;
        if let Err(err) = lifecycle::delete(&self.api, &dto.id).await {
            tracing::warn!("builder sandbox {}: {err}", dto.id);
        }
        match waited {
            Ok(started) => Ok(started.built),
            Err(failure) => Err(build_failure(&spec.name, *failure, limit)),
        }
    }
}

/// Give an image error the image's name.
fn name_image_error(err: SandboxError, name: &str) -> SandboxError {
    match err {
        SandboxError::Image {
            message,
            log_tail,
            task_fault,
            ..
        } => SandboxError::Image {
            name: name.to_string(),
            message,
            log_tail,
            task_fault,
        },
        other => other,
    }
}

/// The error for a builder sandbox that did not start.
fn build_failure(name: &str, failure: Failure, limit: Duration) -> SandboxError {
    if let Some(err) = failure.api {
        return err;
    }
    let log_tail = failure.log_tail.join("\n");
    if failure.timed_out {
        return SandboxError::Image {
            name: name.to_string(),
            message: format!(
                "Daytona did not finish building the image within {}s (it was `{}`). If the build is \
                 just slow, raise the task's build timeout; otherwise run again.",
                limit.as_secs(),
                failure.state
            ),
            log_tail,
            task_fault: false,
        };
    }
    let output = format!("{log_tail}\n{}", failure.reason);
    if pull_refused(&output) {
        return SandboxError::Image {
            name: name.to_string(),
            message: format!(
                "the image the task names cannot be pulled ({}). Fix the image reference in the \
                 task; Daytona pulls without your registry login, so a private image needs a \
                 public copy.",
                if failure.reason.is_empty() {
                    crate::build_log::key_line(&log_tail)
                } else {
                    failure.reason.clone()
                }
            ),
            log_tail,
            task_fault: true,
        };
    }
    let reading = crate::build_log::read(&output);
    if let Some(err) = limit_error(&failure.reason) {
        return err;
    }
    let message = if failure.state == "build_failed" && reading.task_fault {
        format!(
            "the task's Dockerfile failed to build on Daytona ({}). This is a problem in the task: \
             fix the Dockerfile; `docker build` shows the same failure locally.",
            reading.key_line
        )
    } else if failure.state == "build_failed" {
        format!(
            "Daytona could not build the image ({}). The log does not point at the Dockerfile; this \
             is likely on Daytona's side or the network, so run again.",
            if failure.reason.is_empty() {
                reading.key_line.clone()
            } else {
                failure.reason.clone()
            }
        )
    } else {
        format!(
            "Daytona's builder sandbox ended in state `{}` ({}). This is on Daytona's side; run again.",
            failure.state, failure.reason
        )
    };
    SandboxError::Image {
        name: name.to_string(),
        message,
        log_tail,
        task_fault: failure.state == "build_failed" && reading.task_fault,
    }
}

/// A registry's answers for an image that cannot be pulled: a wrong
/// reference, or a private image Daytona has no login for. ghcr.io answers
/// a missing or private image by refusing the anonymous token.
const PULL_REFUSED: &[&str] = &[
    "pull access denied",
    "repository does not exist",
    "manifest unknown",
    "insufficient_scope",
    "failed to fetch anonymous token",
];

/// Whether a failure's output says the image cannot be pulled.
fn pull_refused(output: &str) -> bool {
    let lower = output.to_lowercase();
    PULL_REFUSED.iter().any(|p| lower.contains(p))
}

/// An organization limit named in a failure reason, as a setup error. A
/// registry's `insufficient_scope` is a refused pull, not a limit.
fn limit_error(reason: &str) -> Option<SandboxError> {
    let lower = reason.to_lowercase().replace("insufficient_scope", "");
    ["quota", "limit exceeded", "exceeds", "insufficient"]
        .iter()
        .any(|w| lower.contains(w))
        .then(|| SandboxError::Unavailable {
            message: format!("Daytona refused the sandbox: {reason}"),
            fix:
                "Your Daytona organization is at a resource limit. Lower the run's concurrency or \
                  the task's resources, delete idle sandboxes, or raise the limit in the Daytona \
                  dashboard."
                    .to_string(),
        })
}

/// The error for a trial sandbox that did not start.
fn start_failure(spec: &SandboxSpec, failure: Failure) -> SandboxError {
    if let Some(err) = failure.api {
        return err;
    }
    if let Some(err) = limit_error(&failure.reason) {
        return err;
    }
    if failure.state == "build_failed" {
        return build_failure(&spec.name, failure, START_TIMEOUT);
    }
    SandboxError::Start(format!(
        "Daytona sandbox for {} ended in state `{}`{} ({}). This is on Daytona's side; run again.",
        spec.name,
        failure.state,
        if failure.timed_out {
            " and timed out"
        } else {
            ""
        },
        if failure.reason.is_empty() {
            "no reason given"
        } else {
            failure.reason.as_str()
        }
    ))
}

/// Daytona's whole-number resources for `spec`: cpus and GiB of memory and
/// disk, rounded up, and capped at Daytona's per-sandbox limits (4 CPUs,
/// 16 GiB, 10 GiB) with a warning, as BenchFlow 0.8 clamps them: Daytona
/// refuses to create a sandbox that asks for more.
fn resources(spec: &SandboxSpec) -> (Option<u32>, Option<u32>, Option<u32>) {
    let cap = |what: &str, asked: u32, max: u32, unit: &str| {
        if asked > max {
            tracing::warn!(
                "Daytona caps a sandbox's {what} at {max}{unit}; {} asks for {asked}{unit} and gets {max}{unit}",
                spec.name
            );
        }
        asked.min(max)
    };
    let cpu = spec
        .cpus
        .filter(|c| *c > 0.0)
        .map(|c| cap("CPUs", c.ceil().max(1.0) as u32, MAX_CPUS, ""));
    let memory = spec.memory_mb.filter(|m| *m > 0).map(|m| {
        cap(
            "memory",
            m.div_ceil(1024).max(1) as u32,
            MAX_MEMORY_GB,
            " GB",
        )
    });
    let disk = spec.disk_mb.filter(|d| *d > 0).map(|d| {
        let gb = d.div_ceil(1024).max(1);
        if gb > MAX_DISK_GB {
            tracing::warn!(
                "Daytona caps a sandbox's disk at {MAX_DISK_GB} GB; {} asks for {gb} GB and gets {MAX_DISK_GB} GB",
                spec.name
            );
            MAX_DISK_GB as u32
        } else {
            gb as u32
        }
    });
    (cpu, memory, disk)
}

#[async_trait]
impl Provider for DaytonaProvider {
    fn name(&self) -> &str {
        "daytona"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            network_none: true,
            gpus: false,
            pids_limit: false,
            build_from_dockerfile: true,
            max_disk_mb: Some(MAX_DISK_GB * 1024),
            firewall: true,
            network_allow_cidrs: Some(MAX_ALLOW_CIDRS),
            network_switch: true,
            // Its daemon (process 1, root) answers on port 2280 inside the
            // sandbox without credentials; checked live 2026-10-10.
            root_service: true,
        }
    }

    async fn status(&self) -> ProviderStatus {
        if !self.api.has_key() {
            return ProviderStatus {
                ok: false,
                summary: "Daytona: DAYTONA_API_KEY is not set".to_string(),
                fix: Some(api::key_fix()),
            };
        }
        match count_running(&self.api).await {
            Ok((n, here)) => ProviderStatus {
                ok: true,
                summary: format!(
                    "Daytona: signed in, {}{}",
                    running_summary(n, here, &owner::scope()),
                    if self.api.base() == DEFAULT_API_URL {
                        String::new()
                    } else {
                        format!(" ({})", self.api.base())
                    }
                ),
                fix: None,
            },
            Err(err) => {
                let message = err.message();
                match SandboxError::from(err) {
                    SandboxError::Unavailable { message, fix } => ProviderStatus {
                        ok: false,
                        summary: format!("Daytona: {message}"),
                        fix: Some(fix),
                    },
                    _ => ProviderStatus {
                        ok: false,
                        summary: format!("Daytona: cannot reach {} ({message})", self.api.base()),
                        fix: Some(
                            "Check the network, and DAYTONA_API_URL if you set it, then try again."
                                .to_string(),
                        ),
                    },
                }
            }
        }
    }

    async fn prepare(
        &self,
        spec: &ImageSpec,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        let started = Instant::now();
        if !self.api.has_key() {
            return Err(ApiError::NoKey.into());
        }
        if let Some(platform) = spec.platform.as_deref().filter(|p| *p != "linux/amd64") {
            log(&format!(
                "note: Daytona builds linux/amd64 images; the task asks for {platform}"
            ));
        }
        let (base, hashes) = match &spec.source {
            ImageSource::Build {
                context,
                dockerfile,
            } => {
                let text = tokio::fs::read_to_string(dockerfile).await.map_err(|e| {
                    SandboxError::Image {
                        name: spec.name.clone(),
                        message: format!(
                            "the task's Dockerfile {} cannot be read ({e}); fix the task",
                            dockerfile.display()
                        ),
                        log_tail: String::new(),
                        task_fault: true,
                    }
                })?;
                let hash =
                    build::upload_context(&self.api, self.http.clone(), context, dockerfile, log)
                        .await
                        .map_err(|e| name_image_error(e, &spec.name))?;
                (text, vec![hash])
            }
            ImageSource::Pull { reference } => (format!("FROM {reference}\n"), Vec::new()),
        };
        let recipe = Recipe::new(&base, &spec.layers, hashes);
        let key = recipe.key();
        let reference = match &spec.source {
            ImageSource::Pull { reference } if spec.layers.is_empty() => reference.clone(),
            _ => format!(
                "{BUILD_PREFIX}{}:{key}",
                crate::docker::repo_name(&spec.name)
            ),
        };
        self.recipes
            .lock()
            .expect("lock")
            .insert(reference.clone(), recipe.clone());

        let gate = self.build_gate(&key);
        let mut built = gate.lock().await;
        if *built {
            log(&format!("using the image Daytona built for {reference}"));
            return Ok(Image {
                reference,
                fresh: false,
                duration: started.elapsed(),
            });
        }
        let fresh = self.build_now(spec, &recipe, log).await?;
        *built = true;
        log(&format!("image ready: {reference}"));
        Ok(Image {
            reference,
            fresh,
            duration: started.elapsed(),
        })
    }

    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        if !self.api.has_key() {
            return Err(ApiError::NoKey.into());
        }
        if spec.gpus > 0 {
            return Err(SandboxError::Unsupported(format!(
                "{} asks for {} GPU(s), and BenchFlow's Daytona backend has none; run it with \
                 `--sandbox docker` on a host with GPUs",
                spec.name, spec.gpus
            )));
        }
        if let Some(runtime) = &spec.runtime {
            return Err(SandboxError::Unsupported(format!(
                "{} asks for the `{runtime}` container runtime, and Daytona chooses the runtime of \
                 its sandboxes itself; run it with `--sandbox docker` on a Docker host that has \
                 `{runtime}`, or drop --runtime",
                spec.name
            )));
        }
        if spec.pids_limit.is_some() {
            tracing::debug!(
                "Daytona has no process limit; pids_limit is not enforced for {}",
                spec.name
            );
        }
        let recipe = self.recipe_for(&spec.image.reference)?;
        let (cpu, memory, disk) = resources(spec);
        let mut labels = owner::labels();
        labels.insert("ai.benchflow.name".into(), spec.name.clone());
        labels.insert(
            ATTEMPT_LABEL.into(),
            uuid::Uuid::new_v4().simple().to_string(),
        );
        for (k, v) in &spec.labels {
            labels.entry(k.clone()).or_insert_with(|| v.clone());
        }
        let body = CreateSandbox {
            labels,
            target: self.api.target.as_deref(),
            cpu,
            memory,
            disk,
            network_block_all: spec.network == NetworkPolicy::None,
            network_allow_list: allow_list(&spec.network)?,
            auto_stop_interval: AUTO_STOP_MINUTES,
            auto_delete_interval: 0,
            build_info: BuildInfo {
                dockerfile_content: &recipe.dockerfile,
                context_hashes: &recipe.context_hashes,
            },
        };
        let dto = lifecycle::create(&self.api, &body).await?;
        let name = spec.name.clone();
        let started =
            match lifecycle::wait_started(&self.api, &dto.id, START_TIMEOUT, &|line: &str| {
                tracing::debug!("building {name}: {line}")
            })
            .await
            {
                Ok(started) => started,
                Err(failure) => {
                    if let Err(err) = lifecycle::delete(&self.api, &dto.id).await {
                        tracing::warn!("{err}");
                    }
                    return Err(start_failure(spec, *failure));
                }
            };
        let proxy = started
            .dto
            .toolbox_proxy_url
            .clone()
            .unwrap_or_else(|| DEFAULT_TOOLBOX_URL.to_string());
        let toolbox = Toolbox::new(self.api.clone(), &proxy, &dto.id);
        if let Err(err) = set_up(&toolbox, spec.workdir.as_deref()).await {
            if let Err(cleanup) = lifecycle::delete(&self.api, &dto.id).await {
                tracing::warn!("{cleanup}");
            }
            return Err(err);
        }
        let core = Arc::new(SandboxCore {
            toolbox,
            default_env: spec.env.clone(),
            default_cwd: spec.workdir.clone().filter(|w| !w.is_empty()),
            default_user: recipe.default_user.clone(),
        });
        Ok(Box::new(DaytonaSandbox::new(core)))
    }

    async fn sweep(&self) -> Result<usize, SandboxError> {
        if !self.api.has_key() {
            return Ok(0);
        }
        let ours = lifecycle::list(
            &self.api,
            &serde_json::json!({ OWNER_LABEL: owner::scope() }),
        )
        .await?;
        let now = Utc::now();
        let mut removed = 0;
        for sandbox in ours {
            if !is_stale(&sandbox, now).await {
                continue;
            }
            // The list lags deletes by a few seconds: a sandbox the last run
            // deleted can still be listed as started. Ask for it directly.
            match lifecycle::get(&self.api, &sandbox.id).await {
                Err(e) if e.status() == Some(404) => continue,
                Ok(dto) if matches!(dto.state(), "destroying" | "destroyed") => continue,
                _ => {}
            }
            match lifecycle::delete(&self.api, &sandbox.id).await {
                Ok(()) => {
                    removed += 1;
                    tracing::info!(
                        "deleted leftover Daytona sandbox {} ({})",
                        sandbox.id,
                        sandbox.state()
                    );
                }
                Err(err) => tracing::warn!(
                    "could not delete leftover Daytona sandbox {}: {err}",
                    sandbox.id
                ),
            }
        }
        Ok(removed)
    }
}

/// Whether a sandbox carrying our owner label may be deleted: its process
/// on this host is gone; or it is older than 24 hours and idle for 30
/// minutes; or it failed to start more than 2 hours ago.
async fn is_stale(sandbox: &SandboxDto, now: DateTime<Utc>) -> bool {
    let verdict = owner::judge(|k| sandbox.labels.get(k).map(String::as_str)).await;
    match verdict {
        Verdict::Foreign => return false,
        Verdict::Leftover => return true,
        Verdict::Live | Verdict::OtherHost => {}
    }
    let age = |stamp: &Option<String>| {
        stamp
            .as_deref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|t| now.signed_duration_since(t.with_timezone(&Utc)))
    };
    let Some(created) = age(&sandbox.created_at) else {
        return false;
    };
    let failed = matches!(sandbox.state(), "error" | "build_failed");
    if failed && created >= chrono::Duration::hours(2) {
        return true;
    }
    let idle = age(&sandbox.last_activity_at).unwrap_or(created);
    created >= chrono::Duration::hours(24) && idle >= chrono::Duration::minutes(30)
}

/// Count the organization's running sandboxes (read-only).
/// BenchFlow's running sandboxes on the account: the ones with this owner
/// scope's label, and how many of them this machine started. Others on a
/// shared account (people, other tools) are not counted.
async fn count_running(api: &Api) -> Result<(usize, usize), ApiError> {
    let filter = serde_json::json!({ OWNER_LABEL: owner::scope() }).to_string();
    let mut count = (0, 0);
    let mut cursor: Option<String> = None;
    for _ in 0..20 {
        let page: SandboxPage = api
            .json("listing sandboxes", Retry::Safe, |http| {
                let mut request = http
                    .get(api.url("/sandbox"))
                    .query(&[
                        ("states", "started"),
                        ("labels", filter.as_str()),
                        ("limit", "200"),
                    ])
                    .timeout(Duration::from_secs(30));
                if let Some(c) = &cursor {
                    request = request.query(&[("cursor", c.as_str())]);
                }
                Ok(request)
            })
            .await?;
        let (n, here) = count_ours(&page.items, &owner::scope(), owner::hostname());
        count = (count.0 + n, count.1 + here);
        match page.next_cursor.filter(|c| !c.is_empty()) {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    Ok(count)
}

/// Of `items`, the started sandboxes with owner label `scope`, and of
/// those, the ones started from `host`.
fn count_ours(items: &[SandboxDto], scope: &str, host: &str) -> (usize, usize) {
    let ours: Vec<&SandboxDto> = items
        .iter()
        .filter(|s| s.state() == "started")
        .filter(|s| s.labels.get(OWNER_LABEL).map(String::as_str) == Some(scope))
        .collect();
    let here = ours
        .iter()
        .filter(|s| s.labels.get(crate::HOST_LABEL).map(String::as_str) == Some(host))
        .count();
    (ours.len(), here)
}

/// `bench doctor`'s count: "2 BenchFlow sandboxes running, 1 from this
/// machine", naming the owner scope when `BENCHFLOW_OWNER` set another.
fn running_summary(n: usize, here: usize, scope: &str) -> String {
    let mut s = format!(
        "{n} BenchFlow sandbox{} running",
        if n == 1 { "" } else { "es" }
    );
    if n > 0 {
        s.push_str(&format!(", {here} from this machine"));
    }
    if scope != "bench" {
        s.push_str(&format!(" (owner {scope})"));
    }
    s
}

/// Make the root-only staging folder (and the workdir), retrying while the
/// sandbox's daemon comes up.
async fn set_up(toolbox: &Toolbox, workdir: Option<&str>) -> Result<(), SandboxError> {
    let mut last = None;
    for attempt in 0..20 {
        match toolbox.mkdir(STAGING, "700").await {
            Ok(()) => {
                last = None;
                break;
            }
            Err(err) => {
                last = Some(err);
                tokio::time::sleep(Duration::from_millis(250 * (attempt + 1).min(4))).await;
            }
        }
    }
    if let Some(err) = last {
        return Err(err);
    }
    if let Some(dir) = workdir.filter(|d| !d.is_empty()) {
        toolbox.mkdir(dir, "755").await?;
    }
    Ok(())
}

/// What the operations of one sandbox share.
pub(crate) struct SandboxCore {
    pub toolbox: Toolbox,
    /// Variables every command gets (the task's), under the command's own.
    pub default_env: BTreeMap<String, String>,
    /// The folder commands start in when they name none.
    pub default_cwd: Option<String>,
    /// Who commands run as when they name no user (the image's user).
    pub default_user: Option<String>,
}

/// One Daytona sandbox.
pub struct DaytonaSandbox {
    core: Arc<SandboxCore>,
    keepalive: tokio::task::JoinHandle<()>,
}

impl fmt::Debug for DaytonaSandbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DaytonaSandbox")
            .field("id", &self.core.toolbox.sandbox_id)
            .finish()
    }
}

impl DaytonaSandbox {
    fn new(core: Arc<SandboxCore>) -> Self {
        let api = core.toolbox.api.clone();
        let id = core.toolbox.sandbox_id.clone();
        let keepalive = tokio::spawn(async move {
            loop {
                tokio::time::sleep(KEEPALIVE_EVERY).await;
                let result = api
                    .send("keeping a sandbox active", Retry::Safe, |http| {
                        Ok(http
                            .post(api.url(&format!("/sandbox/{id}/last-activity")))
                            .timeout(Duration::from_secs(30)))
                    })
                    .await;
                match result {
                    Ok(_) => {}
                    Err(err) if err.status() == Some(404) => break,
                    Err(err) => tracing::debug!("keepalive for {id}: {}", err.message()),
                }
            }
        });
        Self { core, keepalive }
    }
}

impl Drop for DaytonaSandbox {
    fn drop(&mut self) {
        // Without the keepalive, Daytona stops the sandbox once it idles and
        // then deletes it.
        self.keepalive.abort();
    }
}

#[async_trait]
impl Sandbox for DaytonaSandbox {
    fn id(&self) -> &str {
        &self.core.toolbox.sandbox_id
    }

    fn backend(&self) -> &str {
        "daytona"
    }

    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        exec::run(&self.core, exec, true).await
    }

    async fn spawn(&self, exec: Exec) -> Result<Process, SandboxError> {
        process::spawn(&self.core, exec).await
    }

    async fn upload(&self, local: &Path, remote: &str) -> Result<(), SandboxError> {
        files::upload(&self.core, local, remote).await
    }

    async fn write_file(
        &self,
        remote: &str,
        contents: &[u8],
        mode: u32,
    ) -> Result<(), SandboxError> {
        files::write_file(&self.core, remote, contents, mode).await
    }

    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        files::download(&self.core, remote, local).await
    }

    async fn destroy(&self) -> Result<(), SandboxError> {
        self.keepalive.abort();
        lifecycle::delete(&self.core.toolbox.api, &self.core.toolbox.sandbox_id).await
    }

    async fn set_network(&self, policy: &NetworkPolicy) -> Result<(), SandboxError> {
        let settings = NetworkSettings {
            network_block_all: *policy == NetworkPolicy::None,
            network_allow_list: allow_list(policy)?.unwrap_or_default(),
        };
        lifecycle::set_network(
            &self.core.toolbox.api,
            &self.core.toolbox.sandbox_id,
            &settings,
        )
        .await
    }
}

/// The most CIDR ranges Daytona takes in a sandbox's allow list.
pub const MAX_ALLOW_CIDRS: usize = 10;

/// Daytona's allow list for `policy`: the ranges of a filtered network, or
/// none. More ranges than Daytona takes are refused here, not by the API.
fn allow_list(policy: &NetworkPolicy) -> Result<Option<String>, SandboxError> {
    match policy {
        NetworkPolicy::Filtered {
            allow: Some(ranges),
        } if ranges.len() > MAX_ALLOW_CIDRS => Err(SandboxError::Unsupported(format!(
            "Daytona limits a sandbox to {MAX_ALLOW_CIDRS} allowed CIDR ranges; this network needs {}",
            ranges.len()
        ))),
        NetworkPolicy::Filtered {
            allow: Some(ranges),
        } => Ok(Some(ranges.join(","))),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> SandboxSpec {
        SandboxSpec {
            image: Image {
                reference: "ubuntu:24.04".into(),
                fresh: false,
                duration: Duration::ZERO,
            },
            name: "t".into(),
            cpus: Some(1.5),
            memory_mb: Some(1500),
            disk_mb: Some(50_000),
            gpus: 0,
            pids_limit: None,
            network: NetworkPolicy::Open,
            env: BTreeMap::new(),
            workdir: None,
            labels: BTreeMap::new(),
            runtime: None,
        }
    }

    /// A registry refusing the pull (`insufficient_scope`) once read as an
    /// organization limit, with a fix about concurrency; it is the task's
    /// image reference.
    #[test]
    fn a_refused_pull_is_the_tasks_image_not_a_limit() {
        let reason = "docker.io/acme/nope:1: failed to resolve source metadata for docker.io/acme/nope:1: \
                      pull access denied, repository does not exist or may require authorization: \
                      server message: insufficient_scope: authorization failed";
        assert!(limit_error(reason).is_none());
        let failure = Failure {
            state: "build_failed".into(),
            reason: reason.into(),
            log_tail: Vec::new(),
            timed_out: false,
            api: None,
        };
        match build_failure("t", failure, Duration::from_secs(60)) {
            SandboxError::Image {
                message,
                task_fault,
                ..
            } => {
                assert!(task_fault);
                assert!(
                    message.starts_with("the image the task names cannot be pulled"),
                    "{message}"
                );
            }
            other => panic!("{other:?}"),
        }
        // A real limit still reads as one.
        assert!(limit_error("insufficient quota: 4 sandboxes running").is_some());
    }

    /// ghcr.io refuses the anonymous token for an image that does not exist
    /// (or is private). Live on Daytona, a task naming such an image read
    /// "the task's Dockerfile failed to build", though it has no Dockerfile.
    #[test]
    fn a_ghcr_image_that_cannot_be_pulled_is_named_as_such() {
        let tail = "ERROR: failed to authorize: failed to fetch anonymous token: unexpected status from \
                    GET request to https://ghcr.io/token?scope=repository%3Aacme%2Fnope%3Apull&service=ghcr.io: \
                    403 Forbidden";
        let failure = Failure {
            state: "build_failed".into(),
            reason: String::new(),
            log_tail: vec![tail.into()],
            timed_out: false,
            api: None,
        };
        match build_failure("t", failure, Duration::from_secs(60)) {
            SandboxError::Image {
                message,
                task_fault,
                ..
            } => {
                assert!(task_fault);
                assert!(
                    message.starts_with("the image the task names cannot be pulled"),
                    "{message}"
                );
                assert!(!message.contains("Dockerfile"), "{message}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn resources_round_up_and_are_capped() {
        assert_eq!(resources(&spec()), (Some(2), Some(2), Some(10)));
        let mut s = spec();
        s.cpus = Some(0.5);
        s.memory_mb = Some(512);
        s.disk_mb = Some(3 * 1024);
        assert_eq!(resources(&s), (Some(1), Some(1), Some(3)));
        s.cpus = None;
        s.memory_mb = None;
        s.disk_mb = None;
        assert_eq!(resources(&s), (None, None, None));
        // SkillsBench's fix-erlang-ssh-cve and gravitational-wave-detection
        // ask for 8 CPUs, which Daytona refused with HTTP 400 (T4.7).
        s.cpus = Some(8.0);
        s.memory_mb = Some(32 * 1024);
        assert_eq!(resources(&s), (Some(4), Some(16), None));
    }

    #[test]
    fn config_debug_hides_the_key() {
        let config = DaytonaConfig {
            api_key: Some("dtn_supersecret".into()),
            api_url: DEFAULT_API_URL.into(),
            target: None,
        };
        assert!(!format!("{config:?}").contains("supersecret"));
    }

    #[tokio::test]
    async fn without_a_key_status_says_where_to_get_one() {
        let provider = DaytonaProvider::new(DaytonaConfig {
            api_key: None,
            api_url: DEFAULT_API_URL.into(),
            target: None,
        })
        .unwrap();
        let status = provider.status().await;
        assert!(!status.ok);
        let fix = status.fix.unwrap();
        assert!(
            fix.contains("https://app.daytona.io/dashboard/keys"),
            "{fix}"
        );
        assert!(fix.contains("export DAYTONA_API_KEY"), "{fix}");
        let log = |_: &str| {};
        let err = provider
            .prepare(
                &ImageSpec {
                    source: ImageSource::Pull {
                        reference: "ubuntu:24.04".into(),
                    },
                    layers: vec![],
                    name: "t".into(),
                    digest: "d".into(),
                    build_timeout: None,
                    platform: None,
                },
                &log,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxError::Unavailable { .. }));
        assert_eq!(provider.sweep().await.unwrap(), 0);
    }

    #[test]
    fn doctor_counts_only_benchflows_own_sandboxes() {
        let sandbox = |state: &str, owner: Option<&str>, host: &str| SandboxDto {
            id: "s".into(),
            state: Some(state.into()),
            error_reason: None,
            labels: owner
                .map(|o| {
                    BTreeMap::from([
                        (OWNER_LABEL.to_string(), o.to_string()),
                        (crate::HOST_LABEL.to_string(), host.to_string()),
                    ])
                })
                .unwrap_or_default(),
            toolbox_proxy_url: None,
            created_at: None,
            last_activity_at: None,
        };
        let items = [
            sandbox("started", Some("bench"), "mac"),
            sandbox("started", Some("bench"), "other"),
            sandbox("stopped", Some("bench"), "mac"),
            sandbox("started", Some("team-b"), "mac"),
            sandbox("started", None, "mac"),
        ];
        assert_eq!(count_ours(&items, "bench", "mac"), (2, 1));
        assert_eq!(
            running_summary(2, 1, "bench"),
            "2 BenchFlow sandboxes running, 1 from this machine"
        );
        assert_eq!(
            running_summary(0, 0, "bench"),
            "0 BenchFlow sandboxes running"
        );
        assert_eq!(
            running_summary(1, 1, "team-b"),
            "1 BenchFlow sandbox running, 1 from this machine (owner team-b)"
        );
    }

    #[tokio::test]
    async fn staleness_follows_owner_age_and_activity() {
        let now = Utc::now();
        let stamp = |ago: chrono::Duration| (now - ago).to_rfc3339();
        let mut labels = owner::labels();
        labels.insert(crate::HOST_LABEL.into(), "another-host".into());
        let sandbox =
            |state: &str, created: chrono::Duration, active: chrono::Duration| SandboxDto {
                id: "s".into(),
                state: Some(state.into()),
                error_reason: None,
                labels: labels.clone(),
                toolbox_proxy_url: None,
                created_at: Some(stamp(created)),
                last_activity_at: Some(stamp(active)),
            };
        let h = chrono::Duration::hours;
        let m = chrono::Duration::minutes;
        assert!(!is_stale(&sandbox("started", h(2), m(1)), now).await);
        assert!(
            !is_stale(&sandbox("started", h(30), m(5)), now).await,
            "recently active"
        );
        assert!(is_stale(&sandbox("started", h(30), h(2)), now).await);
        assert!(is_stale(&sandbox("build_failed", h(3), h(3)), now).await);
        assert!(!is_stale(&sandbox("build_failed", m(30), m(30)), now).await);
        let mut foreign = sandbox("started", h(100), h(100));
        foreign
            .labels
            .insert(OWNER_LABEL.into(), "someone-else".into());
        assert!(!is_stale(&foreign, now).await);
    }

    #[test]
    fn build_failures_are_judged_from_the_log() {
        let failure = |state: &str, tail: &[&str]| Failure {
            state: state.into(),
            reason: "build failed".into(),
            log_tail: tail.iter().map(|s| s.to_string()).collect(),
            timed_out: false,
            api: None,
        };
        let err = build_failure(
            "t",
            failure(
                "build_failed",
                &[
                    "#5 [2/2] RUN exit 3",
                    "ERROR: process \"/bin/sh -c exit 3\" did not complete successfully: exit code: 3",
                ],
            ),
            Duration::from_secs(60),
        );
        assert!(
            matches!(
                err,
                SandboxError::Image {
                    task_fault: true,
                    ..
                }
            ),
            "{err:?}"
        );
        let err = build_failure(
            "t",
            failure("build_failed", &["dial tcp: i/o timeout"]),
            Duration::from_secs(60),
        );
        assert!(
            matches!(
                err,
                SandboxError::Image {
                    task_fault: false,
                    ..
                }
            ),
            "{err:?}"
        );
        let err = build_failure(
            "t",
            Failure {
                reason: "Total CPU quota exceeded".into(),
                ..failure("error", &[])
            },
            Duration::from_secs(60),
        );
        assert!(matches!(err, SandboxError::Unavailable { .. }), "{err:?}");
    }
}
