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
use futures::StreamExt;

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
/// The most rebuilds under a new key one `prepare` tries for an image
/// whose builds Daytona kept as cancelled.
const MAX_REBUILDS: u32 = 5;
/// A builder's label: how long its image may take to build, in seconds.
/// A build cut off at that limit gets as long again to finish, so the
/// sweep leaves a sandbox still building alone for twice its limit.
const BUILD_LIMIT_LABEL: &str = "ai.benchflow.build-limit";
/// How often a builder left to finish its build is checked.
const RETIRE_POLL: Duration = Duration::from_secs(2);
/// How long the sweep at the start of a run spends on one sandbox (its
/// check and its delete): one Daytona keeps failing must not hold up every
/// run.
const SWEEP_PATIENCE: Duration = Duration::from_secs(15);
/// A build running this long is stuck, whatever its limit (unless twice
/// the limit is longer).
const STUCK_FLOOR: Duration = Duration::from_secs(24 * 3600);
/// The longest limit any build or start is given: a week.
const MAX_LIMIT: Duration = Duration::from_secs(7 * 24 * 3600);
/// A sandbox active this recently is never swept (twice the keepalive).
const ACTIVE_WINDOW: Duration = Duration::from_secs(360);
/// Allowance for the gap between Daytona's clock and its stamps.
const CLOCK_MARGIN: chrono::Duration = chrono::Duration::minutes(2);
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
    /// What `prepare` was asked for, by the image reference it returned:
    /// the recipe before any rebuild, and the image's spec. Trials start
    /// from the recipe's current rebuild (see [`DaytonaProvider::recipe_for`]).
    recipes: Mutex<HashMap<String, Prepared>>,
    /// Per recipe key: whether Daytona has built it in this process.
    built: Mutex<HashMap<String, Arc<tokio::sync::Mutex<bool>>>>,
    /// Per recipe key: the last rebuild under a new key (see
    /// [`DaytonaProvider::build_fresh`]).
    rebuilds: Mutex<HashMap<String, u32>>,
}

/// One image `prepare` was asked for.
#[derive(Clone)]
struct Prepared {
    base: Recipe,
    spec: ImageSpec,
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
            rebuilds: Mutex::new(HashMap::new()),
        })
    }

    /// Delete the sandbox with this id, for a sandbox whose handle was lost.
    /// One that is already gone counts as deleted.
    pub async fn delete_sandbox(&self, id: &str) -> Result<(), SandboxError> {
        lifecycle::delete(&self.api, id).await
    }

    /// The recipe a trial of `reference` starts from, and its rebuild
    /// number. Every reference with the same recipe starts from the same
    /// rebuild: two tasks whose build contexts differ only in files the
    /// `.dockerignore` leaves out share one recipe, and the second must not
    /// start from the cancelled original the first rebuilt.
    fn recipe_for(&self, reference: &str) -> Result<(Recipe, u32), SandboxError> {
        if let Some(prepared) = self.recipes.lock().expect("lock").get(reference) {
            let n = self.rebuild_number(&prepared.base.key());
            return Ok((prepared.base.rebuilt(n), n));
        }
        if reference.starts_with(BUILD_PREFIX) {
            return Err(SandboxError::Unsupported(format!(
                "image {reference} was prepared by another BenchFlow process, and Daytona needs its \
                 recipe in this one; prepare the image in this process first (a bug in the caller)"
            )));
        }
        Ok((
            Recipe::new(&format!("FROM {reference}\n"), &[], Vec::new()),
            0,
        ))
    }

    /// The last rebuild of the recipe with `key` (0: none).
    fn rebuild_number(&self, key: &str) -> u32 {
        self.rebuilds
            .lock()
            .expect("lock")
            .get(key)
            .copied()
            .unwrap_or(0)
    }

    /// A trial sandbox of `reference` found its recipe cancelled (Daytona
    /// dropped the built image and its rebuild was cancelled, or another
    /// run's build was): climb the rebuild loop, unless another trial
    /// already did past rebuild `seen`. False when `reference` was not
    /// prepared in this process.
    async fn climb(&self, reference: &str, seen: u32) -> Result<bool, SandboxError> {
        let Some(prepared) = self.recipes.lock().expect("lock").get(reference).cloned() else {
            return Ok(false);
        };
        let key = prepared.base.key();
        let gate = self.build_gate(&key);
        let mut built = gate.lock().await;
        if self.rebuild_number(&key) == seen {
            let name = prepared.spec.name.clone();
            let log = move |line: &str| tracing::info!("rebuilding {name}: {line}");
            self.build_fresh(&prepared.spec, &prepared.base, &log)
                .await?;
        }
        *built = true;
        Ok(true)
    }

    fn build_gate(&self, key: &str) -> Arc<tokio::sync::Mutex<bool>> {
        self.built
            .lock()
            .expect("lock")
            .entry(key.to_string())
            .or_default()
            .clone()
    }

    /// Build `recipe`, under the rebuild number recorded for it.
    ///
    /// Daytona keeps a build that was cancelled (its builder deleted
    /// mid-build: by an older BenchFlow, by hand, or by a sweep of a build
    /// stuck for twice its limit) and fails every later build of the same
    /// recipe at once. Each such failure is answered with a rebuild under a
    /// new key (a comment line, `rebuild-N`), numbered per recipe, so a
    /// recipe whose rebuild is cancelled in turn still gets a fresh build.
    async fn build_fresh(
        &self,
        spec: &ImageSpec,
        recipe: &Recipe,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<bool, SandboxError> {
        let key = recipe.key();
        let mut n = self
            .rebuilds
            .lock()
            .expect("lock")
            .get(&key)
            .copied()
            .unwrap_or(0);
        let mut tries = 0;
        loop {
            let current = recipe.rebuilt(n);
            match self.build_now(spec, &current, log).await {
                Err(err) if cancelled_build(&err) && tries < MAX_REBUILDS => {
                    tries += 1;
                    n += 1;
                    log(&format!(
                        "Daytona kept a cancelled build of this image; building it again under a new key (rebuild-{n})"
                    ));
                    self.rebuilds.lock().expect("lock").insert(key.clone(), n);
                }
                Err(err) => return Err(err),
                Ok(fresh) => return Ok(fresh),
            }
        }
    }

    /// Start a builder sandbox from `recipe`, follow its build, delete it.
    /// True when Daytona built (or staged) the image now.
    async fn build_now(
        &self,
        spec: &ImageSpec,
        recipe: &Recipe,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<bool, SandboxError> {
        let limit = spec
            .build_timeout
            .unwrap_or(DEFAULT_BUILD_TIMEOUT)
            .min(MAX_LIMIT)
            .saturating_add(Duration::from_secs(120));
        self.build_within(spec, recipe, limit, log).await
    }

    /// [`Self::build_now`], allowing the build `limit`.
    async fn build_within(
        &self,
        spec: &ImageSpec,
        recipe: &Recipe,
        limit: Duration,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<bool, SandboxError> {
        let mut labels = owner::labels();
        labels.insert("ai.benchflow.role".into(), "builder".into());
        labels.insert("ai.benchflow.name".into(), spec.name.clone());
        labels.insert(BUILD_LIMIT_LABEL.into(), limit.as_secs().to_string());
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
        // Removes the builder if this future is dropped before it does.
        let mut guard = BuilderGuard {
            api: self.api.clone(),
            attempt: body.labels[ATTEMPT_LABEL].clone(),
            id: None,
            until: deadline(limit),
        };
        let dto = lifecycle::create(&self.api, &body).await;
        let dto = match dto {
            Ok(dto) => dto,
            Err(e) => {
                guard.disarm();
                return Err(name_image_error(e, &spec.name));
            }
        };
        guard.id = Some(dto.id.clone());
        let waited = lifecycle::wait_started(&self.api, &dto.id, limit, log).await;
        guard.disarm();
        match &waited {
            // A build cut off at its limit, or one whose state could not be
            // read, may still be running: deleting its builder now cancels
            // it, and Daytona keeps the cancelled build. It gets as long
            // again to finish, then goes.
            Err(failure) if may_be_building(failure) => {
                retire_later(self.api.clone(), dto.id.clone(), deadline(limit));
            }
            _ => {
                if let Err(err) = lifecycle::delete(&self.api, &dto.id).await {
                    tracing::warn!("builder sandbox {}: {err}", dto.id);
                    crate::note_left_behind("daytona", &dto.id);
                }
            }
        }
        match waited {
            Ok(started) => Ok(started.built),
            Err(failure) => Err(build_failure(&spec.name, *failure, limit)),
        }
    }
}

/// A builder sandbox that is removed in the background when the build is
/// dropped midway (a stopped run cancels its trials): found by its attempt
/// label when its create was still in flight. D4 chaos runs: a Ctrl-C
/// while an image built left the builder behind.
struct BuilderGuard {
    api: Arc<Api>,
    attempt: String,
    id: Option<String>,
    /// The end of the build's time limit.
    until: Instant,
}

impl BuilderGuard {
    fn disarm(&mut self) {
        self.attempt.clear();
    }
}

impl Drop for BuilderGuard {
    fn drop(&mut self) {
        if self.attempt.is_empty() {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let (api, attempt, id, until) = (
            self.api.clone(),
            std::mem::take(&mut self.attempt),
            self.id.take(),
            self.until,
        );
        crate::track_cleanup(runtime.spawn(async move {
            let id = match id {
                Some(id) => Some(id),
                // The create may still land: look for it for a while.
                None => {
                    let mut found = None;
                    for _ in 0..6 {
                        if let Ok(Some(dto)) =
                            lifecycle::find_by_label(&api, ATTEMPT_LABEL, &attempt).await
                        {
                            found = Some(dto.id);
                            break;
                        }
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                    found
                }
            };
            if let Some(id) = id {
                retire(&api, &id, until).await;
            }
        }));
    }
}

/// Delete builder `id` once Daytona no longer builds its image, waiting
/// until `until` for the build and checking again after a failed check.
/// Deleting a builder mid-build cancels the build, and Daytona keeps the
/// cancelled build: every later sandbox of that image fails at once
/// ("context canceled"). A build still running at `until` is left to
/// finish: Daytona deletes the builder once it has built and idled for its
/// auto-stop interval, and a later run's sweep removes it if it never
/// finishes.
async fn retire(api: &Api, id: &str, until: Instant) {
    loop {
        let state = match lifecycle::get(api, id).await {
            Err(e) if e.status() == Some(404) => return,
            Ok(dto) if done_building(dto.state()) => break,
            Ok(dto) => dto.state().to_string(),
            Err(e) => e.message(),
        };
        if Instant::now() >= until {
            tracing::warn!(
                "builder sandbox {id} is still building or cannot be checked ({state}); leaving it to finish"
            );
            crate::note_left_behind("daytona", id);
            return;
        }
        tokio::time::sleep(RETIRE_POLL).await;
    }
    if let Err(err) = lifecycle::delete(api, id).await {
        tracing::warn!("builder sandbox {id}: {err}");
        crate::note_left_behind("daytona", id);
    }
}

/// [`retire`] in the background; the run waits for it before it exits
/// ([`crate::settle_cleanups`]).
fn retire_later(api: Arc<Api>, id: String, until: Instant) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    crate::track_cleanup(runtime.spawn(async move { retire(&api, &id, until).await }));
}

/// A sandbox that did not start but may still be building: it ran out of
/// time, or its state could not be read.
fn may_be_building(failure: &Failure) -> bool {
    failure.timed_out || failure.api.is_some() || !done_building(&failure.state)
}

/// Whether a sandbox still building has been at it for more than a day, or
/// twice its build limit when that is longer (its [`BUILD_LIMIT_LABEL`]; a
/// trial sandbox's start limit when it has none): such a build is stuck.
fn build_overdue(sandbox: &SandboxDto, now: DateTime<Utc>) -> bool {
    let limit = sandbox
        .labels
        .get(BUILD_LIMIT_LABEL)
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(START_TIMEOUT)
        .min(MAX_LIMIT);
    let Some(created) = sandbox
        .created_at
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
    else {
        return false;
    };
    let age = now.signed_duration_since(created.with_timezone(&Utc));
    // Slow is not stuck, and a build left to finish may be one a live run
    // shares: only a day (or twice a longer limit) makes it stuck.
    age.to_std()
        .is_ok_and(|age| age > limit.saturating_mul(2).max(STUCK_FLOOR))
}

/// `limit` from now, capped at [`MAX_LIMIT`].
fn deadline(limit: Duration) -> Instant {
    let now = Instant::now();
    now.checked_add(limit.min(MAX_LIMIT)).unwrap_or(now)
}

/// A state in which Daytona has stopped building the sandbox's image, so
/// deleting it cancels nothing. Any other state, `unknown` and states this
/// code does not know included, may be a build.
fn done_building(state: &str) -> bool {
    matches!(
        state,
        "started" | "stopped" | "error" | "build_failed" | "archived" | "destroyed" | "destroying"
    )
}

/// A build Daytona failed because a build of the same recipe was cancelled
/// (see [`build_failure`]): at once, for a build cancelled earlier, or
/// midway, for one this sandbox shared.
fn cancelled_build(err: &SandboxError) -> bool {
    matches!(err, SandboxError::Image { message, task_fault: false, .. }
        if message.contains("context canceled"))
}

/// BuildKit's words for a cancelled build.
fn cancel_words(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("context canceled") || lower.contains("canceled:")
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
    // A builder deleted while it built (by hand, or by another program) is
    // not a build that failed: neither the task's image nor an agent's
    // layers. D4b live check: such a layered build was blamed on the
    // agent's install (R201) once the task's image built on its own.
    if matches!(failure.state.as_str(), "gone" | "destroyed" | "destroying") {
        return SandboxError::Start(format!(
            "Daytona's builder sandbox for {name} was deleted while it built the image (it was \
             `{}`); something outside this run removed it. Run again.",
            failure.state
        ));
    }
    let log_tail = failure.log_tail.join("\n");
    if failure.timed_out {
        return SandboxError::Image {
            name: name.to_string(),
            message: format!(
                "Daytona did not finish building the image within {}s (it was `{}`). BenchFlow \
                 leaves the build running rather than cancel it, so a run started after it \
                 finishes uses it; if the build is just slow, raise the task's build timeout.",
                limit.as_secs(),
                failure.state
            ),
            log_tail,
            task_fault: false,
        };
    }
    // A build cancelled under it (its builder deleted mid-build, here or by
    // a build it shared) is no step's failure. BuildKit words it
    // `failed to solve: Canceled: context canceled`, which would otherwise
    // read as the task's.
    if cancel_words(&failure.reason) {
        return SandboxError::Image {
            name: name.to_string(),
            message: format!(
                "Daytona could not build the image (context canceled: {}). A build of it was \
                 cancelled; this is on Daytona's side, so run again.",
                failure.reason.trim()
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
            // Its network block let a sandbox reach IPv6 addresses; checked
            // live 2026-10-10.
            network_none_leaks_ipv6: true,
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
        self.recipes.lock().expect("lock").insert(
            reference.clone(),
            Prepared {
                base: recipe.clone(),
                spec: spec.clone(),
            },
        );

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
        let fresh = self.build_fresh(spec, &recipe, log).await?;
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
        // A recipe found cancelled is rebuilt once, then the trial starts
        // from the rebuild.
        let reference = &spec.image.reference;
        let (recipe, seen) = self.recipe_for(reference)?;
        match self.create_from(spec, &recipe).await {
            Err(err) if cancelled_build(&err) && self.climb(reference, seen).await? => {
                let (recipe, _) = self.recipe_for(reference)?;
                self.create_from(spec, &recipe).await
            }
            other => other,
        }
    }

    async fn sweep(&self) -> Result<usize, SandboxError> {
        self.sweep_now().await
    }
}

impl DaytonaProvider {
    /// Start a trial sandbox from `recipe`.
    async fn create_from(
        &self,
        spec: &SandboxSpec,
        recipe: &Recipe,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
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
                // A sandbox whose image Daytona rebuilds (its cache dropped)
                // may still be building: deleting it would cancel the build.
                Err(failure) if may_be_building(&failure) => {
                    retire_later(self.api.clone(), dto.id.clone(), deadline(START_TIMEOUT));
                    return Err(start_failure(spec, *failure));
                }
                Err(failure) => {
                    if let Err(err) = lifecycle::delete(&self.api, &dto.id).await {
                        tracing::warn!("{err}");
                        crate::note_left_behind("daytona", &dto.id);
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
                crate::note_left_behind("daytona", &dto.id);
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

    /// [`Provider::sweep`].
    async fn sweep_now(&self) -> Result<usize, SandboxError> {
        if !self.api.has_key() {
            return Ok(0);
        }
        let ours = lifecycle::list(
            &self.api,
            &serde_json::json!({ OWNER_LABEL: owner::scope() }),
        )
        .await?;
        // Daytona's clock, not this machine's: the ages compare Daytona's
        // own stamps, and a fast clock here made live sandboxes look idle.
        // The margin only makes sandboxes look younger.
        let now = lifecycle::server_now(&self.api)
            .await
            .unwrap_or_else(Utc::now)
            - CLOCK_MARGIN;
        let mut stale = Vec::new();
        for sandbox in ours {
            if !recently_active(&sandbox, now) && is_stale(&sandbox, now).await {
                stale.push(sandbox);
            }
        }
        // One sandbox's check and delete share one limit, so an API that
        // hangs holds up the run's start by that much, not minutes each.
        let api = &self.api;
        let results: Vec<(SandboxDto, Result<Result<bool, SandboxError>, _>)> =
            futures::stream::iter(stale)
                .map(|sandbox| async move {
                    let result =
                        tokio::time::timeout(SWEEP_PATIENCE, sweep_one(api, &sandbox, now)).await;
                    (sandbox, result)
                })
                .buffer_unordered(8)
                .collect()
                .await;
        let mut removed = 0;
        for (sandbox, result) in results {
            match result {
                Ok(Ok(true)) => {
                    removed += 1;
                    tracing::info!(
                        "deleted leftover Daytona sandbox {} ({})",
                        sandbox.id,
                        sandbox.state()
                    );
                }
                Ok(Ok(false)) => {}
                Ok(Err(err)) => tracing::warn!(
                    "could not delete leftover Daytona sandbox {}: {err}",
                    sandbox.id
                ),
                Err(_) => tracing::warn!(
                    "could not delete leftover Daytona sandbox {}: Daytona did not answer within {}s",
                    sandbox.id,
                    SWEEP_PATIENCE.as_secs()
                ),
            }
        }
        Ok(removed)
    }
}

/// Check one stale sandbox again and delete it. True when it was deleted.
async fn sweep_one(
    api: &Api,
    sandbox: &SandboxDto,
    now: DateTime<Utc>,
) -> Result<bool, SandboxError> {
    // The list lags deletes by a few seconds: a sandbox the last run
    // deleted can still be listed as started. Ask for it directly.
    let state = match lifecycle::get(api, &sandbox.id).await {
        Err(e) if e.status() == Some(404) => return Ok(false),
        Ok(dto) if recently_active(&dto, now) => return Ok(false),
        Ok(dto) => dto.state().to_string(),
        Err(_) => sandbox.state().to_string(),
    };
    if matches!(state.as_str(), "destroying" | "destroyed") {
        return Ok(false);
    }
    // A sandbox still building its image is left to finish: deleting it
    // cancels the build, which Daytona keeps (see `retire`). Only one
    // building for longer than any build may take is stuck, and goes.
    if !done_building(&state) && !build_overdue(sandbox, now) {
        return Ok(false);
    }
    lifecycle::delete_within(api, &sandbox.id, SWEEP_PATIENCE).await?;
    Ok(true)
}

/// Whether a sandbox was active within [`ACTIVE_WINDOW`] of `now`: a live
/// run marks its sandboxes active every 3 minutes, so such a sandbox is
/// never deleted, whatever its labels say.
fn recently_active(sandbox: &SandboxDto, now: DateTime<Utc>) -> bool {
    sandbox
        .last_activity_at
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .is_some_and(|t| {
            now.signed_duration_since(t.with_timezone(&Utc))
                < chrono::Duration::from_std(ACTIVE_WINDOW).unwrap_or(chrono::Duration::MAX)
        })
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
        files::download(&self.core, remote, local, None).await
    }

    async fn destroy(&self) -> Result<(), SandboxError> {
        self.keepalive.abort();
        lifecycle::delete(&self.core.toolbox.api, &self.core.toolbox.sandbox_id).await
    }

    async fn presence(&self) -> Result<crate::Presence, SandboxError> {
        let id = &self.core.toolbox.sandbox_id;
        match lifecycle::get(&self.core.toolbox.api, id).await {
            Ok(dto) => Ok(match dto.state() {
                "started" => crate::Presence::Running,
                "destroyed" | "destroying" => crate::Presence::Gone,
                _ => crate::Presence::Stopped,
            }),
            Err(e) if e.status() == Some(404) => Ok(crate::Presence::Gone),
            // Never `Unsupported`: that would send the caller back to
            // asking a program inside the sandbox.
            Err(e) => Err(SandboxError::Transport(format!(
                "asking Daytona about sandbox {id} failed: {}",
                e.message()
            ))),
        }
    }

    async fn download_capped(
        &self,
        remote: &str,
        local: &Path,
        max_bytes: u64,
    ) -> Result<(), SandboxError> {
        files::download(&self.core, remote, local, Some(max_bytes)).await
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

    /// A local Daytona API whose sandbox builds its image for `build`, then
    /// runs; it records each delete and how long after the start it came.
    async fn building_for(build: Duration) -> (String, Arc<Mutex<Vec<(String, Duration)>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let deleted = Arc::new(Mutex::new(Vec::new()));
        let seen = deleted.clone();
        let start = Instant::now();
        tokio::spawn(async move {
            while let Ok((mut conn, _)) = listener.accept().await {
                let mut buf = vec![0u8; 16384];
                let n = conn.read(&mut buf).await.unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).to_string();
                let line = head.lines().next().unwrap_or_default().to_string();
                if line.starts_with("DELETE /api/sandbox/") {
                    let id = line.split_whitespace().nth(1).unwrap_or_default();
                    let id = id.trim_start_matches("/api/sandbox/").to_string();
                    seen.lock().unwrap().push((id, start.elapsed()));
                }
                let state = if start.elapsed() < build {
                    "building_snapshot"
                } else {
                    "started"
                };
                let body = format!(r#"{{"id":"b1","state":"{state}","labels":{{}}}}"#);
                let reply = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = conn.write_all(reply.as_bytes()).await;
            }
        });
        (format!("http://{addr}/api"), deleted)
    }

    /// D4 chaos runs: a Ctrl-C while Daytona built an image dropped the
    /// build and left its builder sandbox behind; deleting the builder
    /// mid-build instead cancels the build, which Daytona then keeps.
    #[tokio::test]
    async fn a_dropped_build_removes_its_builder_once_built() {
        let (url, deleted) = building_for(Duration::from_secs(4)).await;
        let provider = DaytonaProvider::new(DaytonaConfig {
            api_key: Some("k".into()),
            api_url: url,
            target: None,
        })
        .unwrap();
        let spec = ImageSpec {
            source: ImageSource::Pull {
                reference: "ubuntu:24.04".into(),
            },
            layers: vec![],
            name: "t".into(),
            digest: "d".into(),
            build_timeout: None,
            platform: None,
        };
        let recipe = Recipe::new("FROM ubuntu:24.04\n", &[], vec![]);
        let build = provider.build_now(&spec, &recipe, &|_| {});
        assert!(
            tokio::time::timeout(Duration::from_secs(1), build)
                .await
                .is_err()
        );
        // Not `settle_cleanups`: tests in this process share its list, and
        // another test's wait can take this one's removal off it.
        until(|| !deleted.lock().unwrap().is_empty()).await;
        let deleted = deleted.lock().unwrap().clone();
        assert_eq!(deleted.len(), 1, "{deleted:?}");
        assert_eq!(deleted[0].0, "b1");
        assert!(deleted[0].1 >= Duration::from_secs(4), "{deleted:?}");
    }

    /// Wait up to 30 s for `done`.
    async fn until(done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !done() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// One sandbox of [`fake_daytona`].
    struct FakeBox {
        id: String,
        dockerfile: String,
        labels: BTreeMap<String, String>,
        made: Instant,
        created_at: String,
        /// When Daytona last saw it active (its creation when `None`).
        last_activity: Option<String>,
        /// A state that never changes (seeded sandboxes).
        fixed: Option<&'static str>,
        deleted: bool,
    }

    /// What [`fake_daytona`] holds and saw.
    struct FakeWorld {
        /// How long a build takes.
        build: Duration,
        /// Recipes whose build Daytona kept as cancelled.
        poisoned: Vec<String>,
        boxes: Vec<FakeBox>,
        /// Each delete: the sandbox, and whether it was still building.
        deletes: Vec<(String, bool)>,
        /// Builds started, by recipe.
        builds: Vec<String>,
        /// Until then, `GET /sandbox/{id}` answers 400.
        checks_fail_until: Option<Instant>,
        /// `GET /sandbox/{id}` never answers.
        checks_hang: bool,
        /// Where the API listens.
        url: String,
        /// Daytona's clock against this machine's: answers carry a `Date`
        /// header this far off when set.
        skew: Option<chrono::Duration>,
    }

    impl FakeWorld {
        fn state(&self, b: &FakeBox) -> &'static str {
            if let Some(state) = b.fixed {
                state
            } else if self.poisoned.contains(&b.dockerfile) {
                "build_failed"
            } else if b.made.elapsed() < self.build {
                "building_snapshot"
            } else {
                "started"
            }
        }

        fn dto(&self, b: &FakeBox) -> serde_json::Value {
            let state = self.state(b);
            serde_json::json!({
                "id": b.id,
                "state": state,
                "errorReason": if state == "build_failed" { "context canceled" } else { "" },
                "labels": b.labels,
                "createdAt": b.created_at,
                "lastActivityAt": b.last_activity.clone().unwrap_or_else(|| b.created_at.clone()),
                "toolboxProxyUrl": format!("{}/toolbox", self.url),
            })
        }
    }

    /// A local Daytona API that builds as Daytona does: a build takes
    /// `build`; deleting a sandbox while it builds cancels the build, and
    /// every later build of that recipe fails at once with "context
    /// canceled" and no output.
    async fn fake_daytona(build: Duration) -> (DaytonaProvider, Arc<Mutex<FakeWorld>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let world = Arc::new(Mutex::new(FakeWorld {
            build,
            poisoned: Vec::new(),
            boxes: Vec::new(),
            deletes: Vec::new(),
            builds: Vec::new(),
            checks_fail_until: None,
            checks_hang: false,
            url: format!("http://{addr}"),
            skew: None,
        }));
        let w = world.clone();
        tokio::spawn(async move {
            while let Ok((mut conn, _)) = listener.accept().await {
                let w = w.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 8192];
                    let head_end = loop {
                        let Ok(n) = conn.read(&mut chunk).await else {
                            return;
                        };
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(i) = buf.windows(4).position(|x| x == b"\r\n\r\n") {
                            break i + 4;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let length = head
                        .lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse::<usize>().ok())?
                        })
                        .unwrap_or(0);
                    while buf.len() < head_end + length {
                        let Ok(n) = conn.read(&mut chunk).await else {
                            return;
                        };
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
                    let line = head.lines().next().unwrap_or_default().to_string();
                    let mut words = line.split_whitespace();
                    let method = words.next().unwrap_or_default().to_string();
                    let target = words.next().unwrap_or_default().to_string();
                    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
                    let path = path.trim_start_matches("/api");
                    let hang = w.lock().unwrap().checks_hang
                        && method == "GET"
                        && path.starts_with("/sandbox/");
                    if hang {
                        tokio::time::sleep(Duration::from_secs(600)).await;
                    }
                    let (status, reply) = {
                        let mut w = w.lock().unwrap();
                        match (method.as_str(), path) {
                            ("POST", "/sandbox") => {
                                let v: serde_json::Value = serde_json::from_str(&body).unwrap();
                                let dockerfile = v["buildInfo"]["dockerfileContent"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_string();
                                let labels: BTreeMap<String, String> =
                                    serde_json::from_value(v["labels"].clone()).unwrap_or_default();
                                let id = format!("sb{}", w.boxes.len() + 1);
                                w.builds.push(dockerfile.clone());
                                w.boxes.push(FakeBox {
                                    id,
                                    dockerfile,
                                    labels,
                                    made: Instant::now(),
                                    created_at: Utc::now().to_rfc3339(),
                                    last_activity: None,
                                    fixed: None,
                                    deleted: false,
                                });
                                let b = w.boxes.last().unwrap();
                                (200, w.dto(b).to_string())
                            }
                            ("GET", "/sandbox") => {
                                let filter = query
                                    .split('&')
                                    .find_map(|kv| kv.strip_prefix("labels="))
                                    .map(|f| {
                                        let f = f.replace('+', " ");
                                        percent_decode(&f)
                                    })
                                    .unwrap_or_else(|| "{}".into());
                                let filter: BTreeMap<String, String> =
                                    serde_json::from_str(&filter).unwrap_or_default();
                                let items: Vec<serde_json::Value> = w
                                    .boxes
                                    .iter()
                                    .filter(|b| !b.deleted)
                                    .filter(|b| {
                                        filter.iter().all(|(k, v)| b.labels.get(k) == Some(v))
                                    })
                                    .map(|b| w.dto(b))
                                    .collect();
                                (200, serde_json::json!({ "items": items }).to_string())
                            }
                            ("GET", p) if p.ends_with("/build-logs-url") => (404, "{}".into()),
                            ("GET", p) => {
                                let id = p.trim_start_matches("/sandbox/");
                                if w.checks_fail_until.is_some_and(|t| Instant::now() < t) {
                                    (
                                        400,
                                        r#"{"message":"bad gateway in front of the API"}"#.into(),
                                    )
                                } else {
                                    match w.boxes.iter().find(|b| b.id == id && !b.deleted) {
                                        Some(b) => (200, w.dto(b).to_string()),
                                        None => (404, r#"{"message":"not found"}"#.into()),
                                    }
                                }
                            }
                            ("DELETE", p) => {
                                let id = p.trim_start_matches("/sandbox/").to_string();
                                let found = w.boxes.iter().position(|b| b.id == id && !b.deleted);
                                match found {
                                    Some(i) => {
                                        let building = !done_building(w.state(&w.boxes[i]));
                                        if building {
                                            let recipe = w.boxes[i].dockerfile.clone();
                                            w.poisoned.push(recipe);
                                        }
                                        w.boxes[i].deleted = true;
                                        w.deletes.push((id, building));
                                        (200, "{}".into())
                                    }
                                    None => (404, r#"{"message":"not found"}"#.into()),
                                }
                            }
                            _ => (200, "{}".into()),
                        }
                    };
                    let date = match w.lock().unwrap().skew {
                        Some(skew) => format!("date: {}\r\n", (Utc::now() + skew).to_rfc2822()),
                        None => String::new(),
                    };
                    let reply = format!(
                        "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\n{date}connection: close\r\n\r\n{reply}",
                        reply.len()
                    );
                    let _ = conn.write_all(reply.as_bytes()).await;
                });
            }
        });
        let provider = DaytonaProvider::new(DaytonaConfig {
            api_key: Some("k".into()),
            api_url: format!("http://{addr}/api"),
            target: None,
        })
        .unwrap();
        (provider, world)
    }

    fn percent_decode(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn pull_spec(layers: &[&str], build_timeout: Option<Duration>) -> ImageSpec {
        ImageSpec {
            source: ImageSource::Pull {
                reference: "ubuntu:24.04".into(),
            },
            layers: layers.iter().map(|l| l.to_string()).collect(),
            name: "t".into(),
            digest: "d".into(),
            build_timeout,
            platform: None,
        }
    }

    /// Review of D4's rebuild: it rebuilt once under one fixed key, so an
    /// image whose rebuild was cancelled too stayed broken for good. Each
    /// cancelled build is now answered with the next rebuild, and trials
    /// start from the recipe that built.
    #[tokio::test]
    async fn an_image_cancelled_twice_still_gets_a_fresh_build() {
        let (provider, world) = fake_daytona(Duration::ZERO).await;
        let spec = pull_spec(&["echo agent"], None);
        let recipe = Recipe::new("FROM ubuntu:24.04\n", &spec.layers, Vec::new());
        world.lock().unwrap().poisoned = vec![
            recipe.dockerfile.clone(),
            recipe.rebuilt(1).dockerfile.clone(),
        ];
        let image = provider.prepare(&spec, &|_| {}).await.unwrap();
        let builds = world.lock().unwrap().builds.clone();
        assert_eq!(builds.len(), 3, "{builds:?}");
        assert!(builds[2].ends_with("# benchflow: rebuild-2 after a cancelled build\n"));
        // Trial sandboxes start from the recipe that built.
        let (used, _) = provider.recipe_for(&image.reference).unwrap();
        assert_eq!(used.dockerfile, builds[2]);
        // Prepared again in this process: no build, the same recipe.
        let again = provider.prepare(&spec, &|_| {}).await.unwrap();
        assert_eq!(again.reference, image.reference);
        assert!(!again.fresh);
        assert_eq!(world.lock().unwrap().builds.len(), 3);
        assert_eq!(
            provider.recipe_for(&image.reference).unwrap().0.dockerfile,
            builds[2]
        );
        // No builder was deleted mid-build.
        assert!(
            world
                .lock()
                .unwrap()
                .deletes
                .iter()
                .all(|(_, building)| !building)
        );
    }

    /// Reviews of D4's branch: a build that ran past its limit still had
    /// its builder deleted mid-build, which cancels the build and leaves
    /// the image broken for every later run.
    #[tokio::test]
    async fn a_build_past_its_limit_is_left_to_finish() {
        let (provider, world) = fake_daytona(Duration::from_secs(4)).await;
        let spec = pull_spec(&["echo slow"], None);
        let recipe = Recipe::new("FROM ubuntu:24.04\n", &spec.layers, Vec::new());
        let err = provider
            .build_within(&spec, &recipe, Duration::from_secs(1), &|_| {})
            .await
            .unwrap_err();
        assert!(err.to_string().contains("did not finish building"), "{err}");
        assert!(world.lock().unwrap().deletes.is_empty());
        until(|| !world.lock().unwrap().deletes.is_empty()).await;
        let w = world.lock().unwrap();
        assert_eq!(w.deletes, vec![("sb1".to_string(), false)]);
        assert!(w.poisoned.is_empty());
    }

    /// Reviews of D4's branch: a dropped build's guard deleted the builder
    /// after any failed check (an API outage during a Ctrl-C), mid-build.
    #[tokio::test]
    async fn a_dropped_build_is_not_deleted_when_its_check_fails() {
        let (provider, world) = fake_daytona(Duration::from_secs(5)).await;
        let spec = pull_spec(&["echo dropped"], None);
        let recipe = Recipe::new("FROM ubuntu:24.04\n", &spec.layers, Vec::new());
        let build = provider.build_within(&spec, &recipe, Duration::from_secs(60), &|_| {});
        assert!(
            tokio::time::timeout(Duration::from_secs(1), build)
                .await
                .is_err()
        );
        world.lock().unwrap().checks_fail_until = Some(Instant::now() + Duration::from_secs(2));
        until(|| !world.lock().unwrap().deletes.is_empty()).await;
        let w = world.lock().unwrap();
        assert_eq!(w.deletes, vec![("sb1".to_string(), false)]);
        assert!(w.poisoned.is_empty());
    }

    /// Review of D4's branch: the sweep's skip of a building builder came
    /// after the staleness rules, so a stuck builder was never removed,
    /// while a trial sandbox still building its image was deleted mid-build.
    #[tokio::test]
    async fn the_sweep_leaves_young_builds_and_removes_stuck_ones() {
        let (provider, world) = fake_daytona(Duration::ZERO).await;
        let mut child = tokio::process::Command::new("true").spawn().unwrap();
        let dead = child.id().unwrap();
        child.wait().await.unwrap();
        let mut labels = owner::labels();
        labels.insert(crate::PID_LABEL.into(), dead.to_string());
        let ago = |secs: i64| (Utc::now() - chrono::Duration::seconds(secs)).to_rfc3339();
        let seed =
            |id: &str, role: Option<&str>, limit: Option<u64>, age: i64, state: &'static str| {
                let mut labels = labels.clone();
                if let Some(role) = role {
                    labels.insert("ai.benchflow.role".into(), role.into());
                }
                if let Some(limit) = limit {
                    labels.insert(BUILD_LIMIT_LABEL.into(), limit.to_string());
                }
                FakeBox {
                    id: id.into(),
                    dockerfile: format!("FROM {id}\n"),
                    labels,
                    made: Instant::now(),
                    created_at: ago(age),
                    last_activity: None,
                    fixed: Some(state),
                    deleted: false,
                }
            };
        let mut active = seed("trial-active", None, None, 3600, "started");
        active.last_activity = Some(ago(60));
        world.lock().unwrap().boxes = vec![
            // A builder of a dead run, building for well over twice its
            // limit, but less than a day: slow, not stuck (review of
            // rust/d4-chaos, Daytona 3).
            seed(
                "slow-builder",
                Some("builder"),
                Some(600),
                3 * 3600,
                "building_snapshot",
            ),
            // One building for more than a day: stuck.
            seed(
                "stuck-builder",
                Some("builder"),
                Some(600),
                25 * 3600,
                "pending_build",
            ),
            // A state this code does not know may be a build (Daytona 5).
            seed("trial-unknown", None, None, 3600, "unknown"),
            // A trial sandbox whose image Daytona is rebuilding.
            seed("trial-building", None, None, 3600, "building_snapshot"),
            // A trial sandbox of the dead run that runs, idle for an hour.
            seed("trial-started", None, None, 3600, "started"),
            // One its labels call dead that was active a minute ago: a
            // live run's, whatever the labels say (Daytona 2).
            active,
        ];
        let removed = provider.sweep().await.unwrap();
        assert_eq!(removed, 2);
        let mut deleted: Vec<String> = world
            .lock()
            .unwrap()
            .deletes
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        deleted.sort();
        assert_eq!(deleted, ["stuck-builder", "trial-started"]);
    }

    /// Reviews of D4's branch: whether a sandbox still ran was asked of a
    /// command inside it, which an agent's own `sh` could answer. Daytona's
    /// API says it.
    #[tokio::test]
    async fn daytona_says_whether_a_sandbox_runs() {
        let (provider, world) = fake_daytona(Duration::ZERO).await;
        let spec = pull_spec(&[], None);
        let recipe = Recipe::new("FROM ubuntu:24.04\n", &[], Vec::new());
        provider
            .build_within(&spec, &recipe, Duration::from_secs(60), &|_| {})
            .await
            .unwrap();
        world.lock().unwrap().deletes.clear();
        world.lock().unwrap().boxes[0].deleted = false;
        let toolbox = Toolbox::new(provider.api.clone(), "http://127.0.0.1:9/toolbox", "sb1");
        let sandbox = DaytonaSandbox::new(Arc::new(SandboxCore {
            toolbox,
            default_env: BTreeMap::new(),
            default_cwd: None,
            default_user: None,
        }));
        assert_eq!(sandbox.presence().await.unwrap(), crate::Presence::Running);
        world.lock().unwrap().boxes[0].fixed = Some("stopped");
        assert_eq!(sandbox.presence().await.unwrap(), crate::Presence::Stopped);
        world.lock().unwrap().boxes[0].deleted = true;
        assert_eq!(sandbox.presence().await.unwrap(), crate::Presence::Gone);
        // An API error is never "cannot tell" (Outcomes 7), which would send
        // the trial back to asking a program inside the sandbox.
        world.lock().unwrap().checks_fail_until = Some(Instant::now() + Duration::from_secs(60));
        assert!(matches!(
            sandbox.presence().await,
            Err(SandboxError::Transport(_))
        ));
    }

    /// Review of rust/d4-chaos (Daytona 1): a rebuilt recipe was kept only
    /// for the image reference that built it; a second task with the same
    /// recipe started from the cancelled original.
    #[tokio::test]
    async fn every_task_with_a_rebuilt_recipe_starts_from_the_rebuild() {
        let (provider, world) = fake_daytona(Duration::ZERO).await;
        let mut a = pull_spec(&["echo agent"], None);
        a.name = "task-a".into();
        let mut b = a.clone();
        b.name = "task-b".into();
        let recipe = Recipe::new("FROM ubuntu:24.04\n", &a.layers, Vec::new());
        world.lock().unwrap().poisoned = vec![recipe.dockerfile.clone()];
        let image_a = provider.prepare(&a, &|_| {}).await.unwrap();
        let image_b = provider.prepare(&b, &|_| {}).await.unwrap();
        assert_ne!(image_a.reference, image_b.reference);
        let rebuilt = recipe.rebuilt(1).dockerfile;
        assert_eq!(
            provider
                .recipe_for(&image_a.reference)
                .unwrap()
                .0
                .dockerfile,
            rebuilt
        );
        assert_eq!(
            provider
                .recipe_for(&image_b.reference)
                .unwrap()
                .0
                .dockerfile,
            rebuilt
        );
    }

    /// Review of rust/d4-chaos (Daytona 1): a trial sandbox that found its
    /// recipe cancelled after this process built it failed for good; it now
    /// climbs the rebuild loop and starts from the rebuild.
    #[tokio::test]
    async fn a_trial_that_finds_its_recipe_cancelled_starts_from_a_rebuild() {
        let (provider, world) = fake_daytona(Duration::ZERO).await;
        let ispec = pull_spec(&["echo agent"], None);
        let recipe = Recipe::new("FROM ubuntu:24.04\n", &ispec.layers, Vec::new());
        let image = provider.prepare(&ispec, &|_| {}).await.unwrap();
        // Daytona dropped the built image, and its rebuild was cancelled.
        world.lock().unwrap().poisoned = vec![recipe.dockerfile.clone()];
        let mut trial = spec();
        trial.image.reference = image.reference.clone();
        let sandbox = provider.create(&trial).await.unwrap();
        assert!(!sandbox.id().is_empty());
        let builds = world.lock().unwrap().builds.clone();
        assert_eq!(
            builds.last().unwrap(),
            &recipe.rebuilt(1).dockerfile,
            "{builds:?}"
        );
    }

    /// Review of rust/d4-chaos (Daytona 4): the age rules compared
    /// Daytona's stamps with this machine's clock; with it two hours fast,
    /// another machine's sandbox 23 hours old looked 25 hours old and idle.
    #[tokio::test]
    async fn the_sweep_ages_sandboxes_by_daytonas_clock() {
        let (provider, world) = fake_daytona(Duration::ZERO).await;
        let mut labels = owner::labels();
        labels.insert(crate::HOST_LABEL.into(), "another-machine".into());
        let stamp = (Utc::now() - chrono::Duration::hours(25)).to_rfc3339();
        world.lock().unwrap().boxes = vec![FakeBox {
            id: "elsewhere".into(),
            dockerfile: "FROM elsewhere\n".into(),
            labels,
            made: Instant::now(),
            created_at: stamp.clone(),
            last_activity: Some(stamp),
            fixed: Some("started"),
            deleted: false,
        }];
        // Daytona's clock says it is two hours earlier than ours.
        world.lock().unwrap().skew = Some(-chrono::Duration::hours(2));
        assert_eq!(provider.sweep().await.unwrap(), 0);
        world.lock().unwrap().skew = None;
        assert_eq!(provider.sweep().await.unwrap(), 1);
    }

    /// Review of rust/d4-chaos (Daytona 7): the sweep's 15 s limit covered
    /// only the delete; a check that hung held up the run's start for
    /// minutes per stale sandbox.
    #[tokio::test]
    async fn a_hanging_check_holds_up_the_sweep_only_briefly() {
        let (provider, world) = fake_daytona(Duration::ZERO).await;
        let mut child = tokio::process::Command::new("true").spawn().unwrap();
        let dead = child.id().unwrap();
        child.wait().await.unwrap();
        let mut labels = owner::labels();
        labels.insert(crate::PID_LABEL.into(), dead.to_string());
        let stamp = (Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
        world.lock().unwrap().boxes = (0..3)
            .map(|i| FakeBox {
                id: format!("left-{i}"),
                dockerfile: "FROM x\n".into(),
                labels: labels.clone(),
                made: Instant::now(),
                created_at: stamp.clone(),
                last_activity: Some(stamp.clone()),
                fixed: Some("started"),
                deleted: false,
            })
            .collect();
        world.lock().unwrap().checks_hang = true;
        let started = Instant::now();
        let removed = tokio::time::timeout(Duration::from_secs(40), provider.sweep())
            .await
            .expect("the sweep finishes")
            .unwrap();
        assert_eq!(removed, 0);
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "{:?}",
            started.elapsed()
        );
    }

    /// Review of rust/d4-chaos (Daytona 6): BuildKit words a cancelled
    /// build `failed to solve: Canceled: context canceled`, which read as
    /// the task's failure and never started the rebuild.
    #[test]
    fn buildkits_cancel_wording_is_a_cancelled_build() {
        let failure = Failure {
            state: "build_failed".into(),
            reason: "failed to solve: Canceled: context canceled".into(),
            log_tail: vec!["#5 [2/2] RUN sleep 240".into()],
            timed_out: false,
            api: None,
        };
        let err = build_failure("t", failure, Duration::from_secs(60));
        assert!(cancelled_build(&err), "{err:?}");
        assert!(err.is_transient());
    }

    /// Review of rust/d4-chaos (Daytona 9): a huge build-limit label or
    /// build timeout overflowed and panicked.
    #[test]
    fn huge_limits_do_not_overflow() {
        let mut sandbox: SandboxDto = serde_json::from_value(serde_json::json!({
            "id": "x",
            "state": "building_snapshot",
            "createdAt": (Utc::now() - chrono::Duration::days(30)).to_rfc3339(),
            "labels": { "ai.benchflow.build-limit": u64::MAX.to_string() },
        }))
        .unwrap();
        assert!(build_overdue(&sandbox, Utc::now()));
        sandbox
            .labels
            .insert(BUILD_LIMIT_LABEL.into(), "9223372036854775808".into());
        assert!(build_overdue(&sandbox, Utc::now()));
        let _ = deadline(Duration::from_secs(u64::MAX));
    }

    /// D4b live check: a builder deleted by hand mid-build came back as an
    /// image error, so with an agent's layers on top the failure was
    /// blamed on the agent's install once the task's image built alone.
    #[test]
    fn a_builder_deleted_mid_build_is_not_a_failed_build() {
        for state in ["gone", "destroyed", "destroying"] {
            let failure = Failure {
                state: state.into(),
                reason: "not found".into(),
                log_tail: vec!["#5 [2/2] RUN sleep 240".into()],
                timed_out: false,
                api: None,
            };
            let err = build_failure("t", failure, Duration::from_secs(60));
            assert!(
                matches!(&err, SandboxError::Start(m) if m.contains("was deleted while it built")),
                "{err:?}"
            );
            assert!(err.is_transient());
        }
    }

    #[test]
    fn a_cancelled_build_is_recognized() {
        let failure = Failure {
            state: "build_failed".into(),
            reason: "context canceled".into(),
            log_tail: Vec::new(),
            timed_out: false,
            api: None,
        };
        assert!(cancelled_build(&build_failure(
            "t",
            failure,
            Duration::from_secs(60)
        )));
        // A build cancelled midway (one this sandbox shared) streamed some
        // log first: still cancelled (review of rust/d4-chaos, Daytona 6).
        let failure = Failure {
            state: "build_failed".into(),
            reason: "context canceled".into(),
            log_tail: vec!["#5 [2/2] RUN sleep 240".into()],
            timed_out: false,
            api: None,
        };
        assert!(cancelled_build(&build_failure(
            "t",
            failure,
            Duration::from_secs(60)
        )));
        // A step whose own program printed the words is the step's failure.
        let failure = Failure {
            state: "build_failed".into(),
            reason: "process \"/bin/sh -c ./fetch\" did not complete successfully: exit code: 1"
                .into(),
            log_tail: vec![
                "#5 [2/2] RUN ./fetch".into(),
                "fetch: context canceled".into(),
            ],
            timed_out: false,
            api: None,
        };
        assert!(!cancelled_build(&build_failure(
            "t",
            failure,
            Duration::from_secs(60)
        )));
    }

    /// Review of rust/d4-chaos (Daytona 5): `unknown` and states this code
    /// does not know counted as "not building", so a builder in one was
    /// deleted mid-build.
    #[test]
    fn only_known_finished_states_count_as_done_building() {
        for s in [
            "creating",
            "pending_build",
            "building_snapshot",
            "pulling_snapshot",
            "unknown",
            "restoring",
        ] {
            assert!(!done_building(s), "{s}");
        }
        for s in ["started", "error", "build_failed", "stopped", "archived"] {
            assert!(done_building(s), "{s}");
        }
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
