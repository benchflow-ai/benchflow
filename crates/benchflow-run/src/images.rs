//! Preparing images once per run: one build per task, one agent layer per
//! (task, agent) pair, however many trials share them.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use benchflow_sandbox::{Image, ImageSource, ImageSpec, Provider, SandboxError};
use benchflow_task::TaskPackage;
use benchflow_task::package::ImageRef;
use sha2::{Digest, Sha256};
use tokio::sync::OnceCell;

use crate::outcome::{Owner, Phase, Problem};

/// One image's preparation, shared by every trial that needs it.
type Cell = Arc<OnceCell<Result<Image, Problem>>>;

/// A run-wide cache of prepared images, keyed by the task image's digest and
/// the agent layers on top.
#[derive(Default)]
pub struct Images {
    cells: Mutex<HashMap<String, Cell>>,
    /// Trials still to run per image, so the run can release an image when
    /// its last trial is done (see [`Images::expect`]).
    pending: Mutex<HashMap<String, usize>>,
    /// Scales every build timeout (`--build-timeout-multiplier`); 0 means 1.
    build_timeout_multiplier: f64,
}

impl Images {
    pub fn new() -> Self {
        Self::default()
    }

    /// Scale every task's build timeout by `x`.
    pub fn build_timeout_multiplier(mut self, x: f64) -> Self {
        self.build_timeout_multiplier = x;
        self
    }

    /// Count one more trial that will need the image under `key`
    /// ([`key_for`]).
    pub fn expect(&self, key: &str) {
        let mut pending = self.pending.lock().expect("images lock");
        *pending.entry(key.to_string()).or_default() += 1;
    }

    /// One trial that needed the image under `key` is done. After the last
    /// one, the backend may reclaim the image (it keeps images between runs
    /// and trims the least recently used when space runs low); a later trial
    /// that needs it again prepares it again.
    pub async fn finished(&self, key: &str, provider: &dyn Provider) {
        let last = {
            let mut pending = self.pending.lock().expect("images lock");
            match pending.get_mut(key) {
                Some(n) if *n > 1 => {
                    *n -= 1;
                    false
                }
                Some(_) => {
                    pending.remove(key);
                    true
                }
                None => false,
            }
        };
        if !last {
            return;
        }
        let cell = self.cells.lock().expect("images lock").remove(key);
        if let Some(Ok(image)) = cell.as_ref().and_then(|c| c.get()) {
            provider.release(image).await;
        }
    }

    /// Prepare `spec` once; concurrent callers with the same digest wait for
    /// the first and share its result (including its failure). `agent` names
    /// the agent whose install layers `spec` carries, if any.
    ///
    /// When an image with agent layers fails to build, the backend's log
    /// cannot always say whether the task's Dockerfile or the agent's
    /// install failed (Daytona builds both in one go), so the task's image
    /// is prepared alone (once, cached as every other image): if it fails,
    /// the failure is the task's (or the backend's); if it builds, it is the
    /// agent's install.
    pub async fn get(
        &self,
        provider: &dyn Provider,
        spec: &ImageSpec,
        log: &(dyn Fn(&str) + Send + Sync),
        agent: Option<&str>,
    ) -> Result<Image, Problem> {
        let key = cache_key(spec);
        let cell = {
            let mut cells = self.cells.lock().expect("images lock");
            cells.entry(key.clone()).or_default().clone()
        };
        let result = self.prepare_once(&cell, provider, spec, log, agent).await;
        // A failure worth retrying (the network, the backend) is not kept:
        // a retried trial prepares the image again rather than reading the
        // same failure back.
        if result.as_ref().is_err_and(|p| p.transient) {
            let mut cells = self.cells.lock().expect("images lock");
            if cells.get(&key).is_some_and(|c| Arc::ptr_eq(c, &cell)) {
                cells.remove(&key);
            }
        }
        result
    }

    async fn prepare_once(
        &self,
        cell: &Cell,
        provider: &dyn Provider,
        spec: &ImageSpec,
        log: &(dyn Fn(&str) + Send + Sync),
        agent: Option<&str>,
    ) -> Result<Image, Problem> {
        let mut spec = spec.clone();
        if self.build_timeout_multiplier > 0.0 {
            spec.build_timeout = spec.build_timeout.map(|t| {
                benchflow_task::values::scale_duration(t, self.build_timeout_multiplier.max(0.01))
            });
        }
        let spec = &spec;
        cell.get_or_init(|| async {
            match provider.prepare(spec, log).await {
                Ok(image) => Ok(image),
                Err(err @ SandboxError::Image { .. })
                    if !spec.layers.is_empty() && !cancelled_unbuilt(&err) =>
                {
                    log(
                        "the image with the agent's install layers failed to build; checking whether the task's image builds on its own",
                    );
                    let mut task_image = spec.clone();
                    task_image.layers.clear();
                    match Box::pin(self.get(provider, &task_image, log, None)).await {
                        Err(task_problem) => Err(task_problem),
                        Ok(_) => Err(layer_problem(agent.unwrap_or("the agent"), err)),
                    }
                }
                Err(err) => Err(image_problem(&spec.name, err)),
            }
        })
        .await
        .clone()
    }
}

/// A build the backend cancelled: not the task's, and Daytona's "context
/// canceled" (D4 chaos runs: a build an earlier run cancelled, which
/// Daytona keeps; the Daytona backend builds such an image again under new
/// keys, so this is what is left when every rebuild was cancelled too; or
/// a build this one shared, cancelled midway). Neither the task's image
/// nor the agent's layers failed it. Any other build without output (a log
/// that could not be followed, a timeout) is checked against the task's
/// image alone.
fn cancelled_unbuilt(err: &SandboxError) -> bool {
    matches!(err, SandboxError::Image { message, task_fault: false, .. }
        if message.contains("context canceled"))
}

/// An image that failed to build or pull: the task's, the backend's, or
/// this machine's problem.
fn image_problem(name: &str, err: SandboxError) -> Problem {
    match err {
        SandboxError::Image {
            message,
            log_tail,
            task_fault: true,
            ..
        } => Problem::new(
            "R101",
            Owner::Task,
            Phase::Image,
            message,
            "fix the failing step in the task's Dockerfile (or the image it names); image.log in the trial folder has the build output",
        )
        .detail(log_tail),
        SandboxError::Image {
            message, log_tail, ..
        } => Problem::new(
            "R102",
            Owner::Sandbox,
            Phase::Image,
            format!("the image for {name} could not be prepared: {message}"),
            "retry; if it keeps failing, run `bench doctor`",
        )
        .detail(log_tail)
        .transient(),
        SandboxError::Unavailable { message, fix } => {
            Problem::new("R102", Owner::Sandbox, Phase::Image, message, fix)
        }
        err @ (SandboxError::Io { .. } | SandboxError::Unsupported(_)) => {
            crate::trial::sandbox_problem(Phase::Image, err)
        }
        other => {
            let transient = other.is_transient();
            let p = Problem::new(
                "R102",
                Owner::Sandbox,
                Phase::Image,
                format!("the image for {name} could not be prepared: {other}"),
                "retry; if it keeps failing, run `bench doctor`",
            );
            if transient { p.transient() } else { p }
        }
    }
}

/// The agent's install layers failed on a task image that builds on its
/// own. The backend's words about the failure go in the detail: on a
/// one-step build they may blame the Dockerfile.
fn layer_problem(agent: &str, err: SandboxError) -> Problem {
    let transient = err.is_transient();
    let (message, log_tail) = match err {
        SandboxError::Image {
            message, log_tail, ..
        } => (message, log_tail),
        other => (other.to_string(), String::new()),
    };
    let p = Problem::new(
        "R201",
        Owner::Agent,
        Phase::Image,
        format!("{agent} could not be installed on top of the task's image, which builds on its own"),
        format!(
            "image.log in the trial folder has the failing step; the install needs network access and curl, tar, and xz in the image (`bench agents {agent}` shows the install script)"
        ),
    )
    .detail(format!("{message}\n{log_tail}"));
    if transient { p.transient() } else { p }
}

/// The layers a trial of `attempt` on `task` adds to the task's image:
/// iptables, when the network policy or the daemon guard needs it (on a
/// backend with a root service), then the agent's install.
pub fn layers_for(
    task: &TaskPackage,
    attempt: &crate::agent::Attempt,
    caps: &benchflow_sandbox::Capabilities,
) -> Vec<String> {
    let mut layers = match attempt {
        crate::agent::Attempt::Model(m) => m.install_layers(),
        _ => Vec::new(),
    };
    if let Some(layer) = crate::netpolicy::firewall_layer(task, crate::netpolicy::Host::Agent, caps)
    {
        layers.insert(0, layer);
    }
    layers
}

/// The cache key of the image a trial of `attempt` on `task` runs in.
pub fn key_for(
    task: &TaskPackage,
    attempt: &crate::agent::Attempt,
    caps: &benchflow_sandbox::Capabilities,
) -> String {
    cache_key(&spec_for(task, layers_for(task, attempt, caps)))
}

/// A prepared image is the task's image plus its agent layers, if any.
fn cache_key(spec: &ImageSpec) -> String {
    if spec.layers.is_empty() {
        return spec.digest.clone();
    }
    let mut h = Sha256::new();
    for l in &spec.layers {
        h.update(l.as_bytes());
        h.update(b"\0");
    }
    format!("{}+{}", spec.digest, hex::encode(h.finalize()))
}

/// The image spec for a task, with optional agent install layers on top.
pub fn spec_for(task: &TaskPackage, layers: Vec<String>) -> ImageSpec {
    let source = match &task.sandbox.image {
        ImageRef::Pull { reference } => ImageSource::Pull {
            reference: reference.clone(),
        },
        // Task formats give these relative to the task folder.
        ImageRef::Build {
            context,
            dockerfile,
        } => ImageSource::Build {
            context: crate::trial::in_package(&task.root, context),
            dockerfile: crate::trial::in_package(&task.root, dockerfile),
        },
    };
    let mut h = Sha256::new();
    h.update(b"benchflow-image-v1\0");
    match &source {
        ImageSource::Pull { reference } => {
            h.update(b"pull\0");
            h.update(reference.as_bytes());
        }
        ImageSource::Build {
            context,
            dockerfile,
        } => {
            h.update(b"build\0");
            hash_build_context(&mut h, context, dockerfile);
        }
    }
    // The digest names the task's image alone: backends tag agent layers
    // on top of it themselves, so one task image serves every agent.
    ImageSpec {
        source,
        layers,
        name: sanitize_name(&task.name),
        digest: hex::encode(h.finalize()),
        build_timeout: task.sandbox.build_timeout,
        platform: None,
    }
}

/// The image spec for a prebuilt image with agent layers on top, outside
/// any task (`bench agents check`). Its digest is the one a task naming
/// the same image gets, so the agent layer is shared with such tasks.
pub fn pull_spec(name: &str, reference: &str, layers: Vec<String>) -> ImageSpec {
    let mut h = Sha256::new();
    h.update(b"benchflow-image-v1\0");
    h.update(b"pull\0");
    h.update(reference.as_bytes());
    ImageSpec {
        source: ImageSource::Pull {
            reference: reference.to_string(),
        },
        layers,
        name: sanitize_name(name),
        digest: hex::encode(h.finalize()),
        build_timeout: None,
        platform: None,
    }
}

/// Hash the files a build reads: the Dockerfile and the context, by relative
/// path and content, in sorted order. Respects `.dockerignore` loosely: the
/// usual large folders are skipped either way.
fn hash_build_context(h: &mut Sha256, context: &Path, dockerfile: &Path) {
    match std::fs::read(dockerfile) {
        Ok(bytes) => {
            h.update(b"dockerfile\0");
            h.update(&bytes);
        }
        // Two tasks whose Dockerfiles cannot be read must not share one
        // cached result (and one failure).
        Err(_) => {
            h.update(b"unreadable\0");
            h.update(dockerfile.to_string_lossy().as_bytes());
        }
    }
    let mut files: Vec<_> = walkdir::WalkDir::new(context)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let n = e.file_name().to_string_lossy();
            !(e.depth() > 0 && (n == ".git" || n == "node_modules" || n == "__pycache__"))
        })
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    files.sort();
    for f in files {
        let rel = f.strip_prefix(context).unwrap_or(&f);
        h.update(rel.to_string_lossy().as_bytes());
        h.update(b"\0");
        if let Ok(bytes) = std::fs::read(&f) {
            h.update(Sha256::digest(&bytes));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(m) = std::fs::metadata(&f) {
                h.update((m.permissions().mode() & 0o111).to_le_bytes());
            }
        }
    }
}

/// A name safe for image tags and container names.
pub fn sanitize_name(name: &str) -> String {
    let s: String = name
        .to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let s = s.trim_matches(|c| c == '-' || c == '.').to_string();
    let s = if s.is_empty() { "task".to_string() } else { s };
    s.chars().take(48).collect()
}

/// The image spec for another image a task names (a service's, or a
/// separate verifier's), with no agent layers. `name` tags it
/// (`<task>-verifier`, `<task>-svc-api`).
pub fn spec_for_image(task: &TaskPackage, image: &ImageRef, name: &str) -> ImageSpec {
    let mut h = Sha256::new();
    h.update(b"benchflow-image-v1\0");
    let source = match image {
        ImageRef::Pull { reference } => {
            h.update(b"pull\0");
            h.update(reference.as_bytes());
            ImageSource::Pull {
                reference: reference.clone(),
            }
        }
        ImageRef::Build {
            context,
            dockerfile,
        } => {
            let (context, dockerfile) = (task.path(context), task.path(dockerfile));
            h.update(b"build\0");
            hash_build_context(&mut h, &context, &dockerfile);
            ImageSource::Build {
                context,
                dockerfile,
            }
        }
    };
    ImageSpec {
        source,
        layers: Vec::new(),
        name: sanitize_name(name),
        digest: hex::encode(h.finalize()),
        build_timeout: task.sandbox.build_timeout,
        platform: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_task_image_serves_every_agent() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
        let task = benchflow_task::load(&fixture).expect("the hello fixture loads");
        let bare = spec_for(&task, Vec::new());
        let codex = spec_for(&task, vec!["npm i codex".into()]);
        let claude = spec_for(&task, vec!["npm i claude".into()]);
        assert_eq!(
            bare.digest, codex.digest,
            "layers stay out of the task digest"
        );
        assert_eq!(codex.digest, claude.digest);
        let keys = [cache_key(&bare), cache_key(&codex), cache_key(&claude)];
        assert_ne!(keys[0], keys[1]);
        assert_ne!(keys[1], keys[2]);
    }

    #[test]
    fn a_check_image_is_keyed_by_its_reference_and_layers() {
        let a = pull_spec("check goose", "ubuntu:24.04", vec!["x".into()]);
        let b = pull_spec("check-kimi", "ubuntu:24.04", vec!["y".into()]);
        assert_eq!(a.digest, b.digest, "one base image serves every agent");
        assert_ne!(cache_key(&a), cache_key(&b));
        assert_ne!(a.digest, pull_spec("x", "debian:12", Vec::new()).digest);
        assert_eq!(a.name, "check-goose");
    }

    #[test]
    fn the_build_context_is_the_task_folders() {
        // A split-layout task names `environment/Dockerfile` relative to
        // itself: the spec must point into the task folder, and the digest
        // must hash the Dockerfile, not a path that does not exist.
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks");
        let task = benchflow_task::load(&fixtures.join("hello-split")).expect("hello-split loads");
        let spec = spec_for(&task, Vec::new());
        let ImageSource::Build {
            dockerfile,
            context,
        } = &spec.source
        else {
            panic!("hello-split builds its image")
        };
        assert!(dockerfile.is_file(), "{}", dockerfile.display());
        assert!(context.is_dir(), "{}", context.display());
        // A copy with another Dockerfile is another image.
        let dir = tempfile::tempdir().unwrap();
        let copy = dir.path().join("hello-split");
        for d in ["environment", "tests", "solution"] {
            std::fs::create_dir_all(copy.join(d)).unwrap();
        }
        for f in [
            "task.toml",
            "instruction.md",
            "tests/test.sh",
            "solution/solve.sh",
        ] {
            std::fs::copy(fixtures.join("hello-split").join(f), copy.join(f)).unwrap();
        }
        std::fs::write(copy.join("environment/Dockerfile"), "FROM debian:13\n").unwrap();
        let other = benchflow_task::load(&copy).expect("the copy loads");
        assert_ne!(spec.digest, spec_for(&other, Vec::new()).digest);
    }

    /// A provider that counts what it prepared and released, and fails
    /// the builds it is told to (one failure each, in order).
    #[derive(Default)]
    struct Counting {
        prepared: std::sync::Mutex<usize>,
        released: std::sync::Mutex<Vec<String>>,
        failures: std::sync::Mutex<Vec<SandboxError>>,
    }

    #[async_trait::async_trait]
    impl Provider for Counting {
        fn name(&self) -> &str {
            "counting"
        }
        fn capabilities(&self) -> benchflow_sandbox::Capabilities {
            benchflow_sandbox::Capabilities::default()
        }
        async fn status(&self) -> benchflow_sandbox::ProviderStatus {
            benchflow_sandbox::ProviderStatus {
                ok: true,
                summary: String::new(),
                fix: None,
            }
        }
        async fn prepare(
            &self,
            spec: &ImageSpec,
            _log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
        ) -> Result<Image, SandboxError> {
            *self.prepared.lock().unwrap() += 1;
            let mut failures = self.failures.lock().unwrap();
            if !failures.is_empty() {
                return Err(failures.remove(0));
            }
            Ok(Image {
                reference: format!("img/{}", spec.name),
                fresh: true,
                duration: std::time::Duration::ZERO,
            })
        }
        async fn create(
            &self,
            _spec: &benchflow_sandbox::SandboxSpec,
        ) -> Result<Box<dyn benchflow_sandbox::Sandbox>, SandboxError> {
            Err(SandboxError::Unsupported("no sandboxes".into()))
        }
        async fn release(&self, image: &Image) {
            self.released.lock().unwrap().push(image.reference.clone());
        }
    }

    fn image_error(task_fault: bool, message: &str) -> SandboxError {
        SandboxError::Image {
            name: "hello".into(),
            message: message.into(),
            log_tail: "the build's last line".into(),
            task_fault,
        }
    }

    /// A failure worth retrying is prepared again on the retry; a task's
    /// failure is kept, and every trial that needs the image gets it.
    #[tokio::test]
    async fn only_failures_worth_retrying_are_prepared_again() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
        let task = benchflow_task::load(&fixture).expect("the hello fixture loads");
        let spec = spec_for(&task, Vec::new());
        let provider = Counting::default();
        provider.failures.lock().unwrap().push(image_error(
            false,
            "Daytona did not finish building the image within 140s",
        ));
        let images = Images::new();
        let first = images.get(&provider, &spec, &|_| {}, None).await;
        assert!(first.is_err_and(|p| p.code == "R102" && p.transient));
        images.get(&provider, &spec, &|_| {}, None).await.unwrap();
        assert_eq!(*provider.prepared.lock().unwrap(), 2);

        let provider = Counting::default();
        provider.failures.lock().unwrap().push(image_error(
            true,
            "the task's Dockerfile failed to build (RUN exit 3)",
        ));
        let images = Images::new();
        for _ in 0..2 {
            let p = images
                .get(&provider, &spec, &|_| {}, None)
                .await
                .unwrap_err();
            assert_eq!((p.code.as_str(), p.owner), ("R101", Owner::Task));
        }
        assert_eq!(*provider.prepared.lock().unwrap(), 1);
    }

    /// With an agent's install layers on top, a failed build is checked
    /// against the task's image alone: the task's when that fails too, the
    /// agent's when it builds.
    #[tokio::test]
    async fn a_failed_layered_build_is_blamed_on_the_part_that_failed() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
        let task = benchflow_task::load(&fixture).expect("the hello fixture loads");
        let layered = spec_for(&task, vec!["npm i -g @acme/agent".into()]);
        // The layers fail; the task's image alone builds.
        let provider = Counting::default();
        provider.failures.lock().unwrap().push(image_error(
            true,
            "the task's Dockerfile failed to build on Daytona (npm ERR! 404)",
        ));
        let p = Images::new()
            .get(&provider, &layered, &|_| {}, Some("acme"))
            .await
            .unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R201", Owner::Agent));
        assert!(
            p.message.starts_with("acme could not be installed"),
            "{}",
            p.message
        );
        // The layered build fails on the backend's side: not the agent's.
        let provider = Counting::default();
        provider.failures.lock().unwrap().push(SandboxError::Image {
            name: "hello".into(),
            message: "Daytona could not build the image (context canceled)".into(),
            log_tail: String::new(),
            task_fault: false,
        });
        let p = Images::new()
            .get(&provider, &layered, &|_| {}, Some("acme"))
            .await
            .unwrap_err();
        assert_eq!(
            (p.code.as_str(), p.owner),
            ("R102", Owner::Sandbox),
            "{p:?}"
        );
        // Review of rust/d4-chaos: a build without output is not proof the
        // backend failed it. Daytona judged the agent's install from its
        // error reason when the log could not be followed, and Docker's
        // timeout carries no log: both are checked against the task's
        // image, which builds, so they are the agent's.
        for err in [
            SandboxError::Image {
                name: "hello".into(),
                message: "the task's Dockerfile failed to build on Daytona (exit code: 1)".into(),
                log_tail: String::new(),
                task_fault: true,
            },
            SandboxError::Image {
                name: "hello".into(),
                message: "the agent's layers did not finish building within 600s".into(),
                log_tail: String::new(),
                task_fault: false,
            },
        ] {
            let provider = Counting::default();
            provider.failures.lock().unwrap().push(err);
            let p = Images::new()
                .get(&provider, &layered, &|_| {}, Some("acme"))
                .await
                .unwrap_err();
            assert_eq!((p.code.as_str(), p.owner), ("R201", Owner::Agent), "{p:?}");
        }
        // The task's Dockerfile fails with and without the layers.
        let provider = Counting::default();
        provider.failures.lock().unwrap().extend([
            image_error(true, "the task's Dockerfile failed to build (RUN exit 3)"),
            image_error(true, "the task's Dockerfile failed to build (RUN exit 3)"),
        ]);
        let p = Images::new()
            .get(&provider, &layered, &|_| {}, Some("acme"))
            .await
            .unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R101", Owner::Task));
    }

    #[tokio::test]
    async fn an_image_is_released_after_its_last_trial() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
        let task = benchflow_task::load(&fixture).expect("the hello fixture loads");
        let provider = Counting::default();
        let images = Images::new();
        let key = key_for(
            &task,
            &crate::agent::Attempt::Oracle,
            &benchflow_sandbox::Capabilities::default(),
        );
        images.expect(&key);
        images.expect(&key);
        let spec = spec_for(&task, Vec::new());
        images.get(&provider, &spec, &|_| {}, None).await.unwrap();
        images.get(&provider, &spec, &|_| {}, None).await.unwrap();
        assert_eq!(*provider.prepared.lock().unwrap(), 1, "prepared once");
        images.finished(&key, &provider).await;
        assert!(
            provider.released.lock().unwrap().is_empty(),
            "one trial still needs it"
        );
        images.finished(&key, &provider).await;
        assert_eq!(
            provider.released.lock().unwrap().len(),
            1,
            "released after the last"
        );
        // Needed again later: prepared again.
        images.get(&provider, &spec, &|_| {}, None).await.unwrap();
        assert_eq!(*provider.prepared.lock().unwrap(), 2);
    }

    #[test]
    fn names_are_tag_safe() {
        assert_eq!(sanitize_name("Hello World/Task_1"), "hello-world-task-1");
        assert_eq!(sanitize_name("..."), "task");
    }
}
