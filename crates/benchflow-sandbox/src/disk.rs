//! A task's disk limit (`[sandbox] disk`): how it holds on a sandbox, and
//! the watch that stands in for it where the backend cannot cap the disk.
//!
//! A sandbox's writable layer is measured the way its backend sees it, from
//! outside the sandbox: the bytes of its files, and the number of files,
//! folders and links it added ([`LayerUsage::charged`]). Nothing inside the
//! sandbox can change that measure, and nothing the agent runs is needed to
//! stop it.
//!
//! The watch runs from the agent's first turn to the end of the trial
//! (grading included), and stops the sandbox on a measure that succeeded
//! and is over the limit. Each turn of the agent is a window: what the layer
//! grew inside the windows is the agent's. A measure that fails or runs
//! slow is retried within bounds fixed for the trial from the measure taken
//! before the agent's first turn ([`bounds`]), which the agent cannot
//! change. When one kind has gone unmeasured for its whole bound and the
//! backend says the sandbox still runs, the watch stops it as unmeasurable.
//! A sandbox the backend no longer has, or cannot say runs, is never
//! stopped by the watch: the trial's own check of the sandbox decides what
//! happened to it. Nothing is paused while the agent works.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::SandboxError;

/// How a sandbox's disk limit holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiskLimit {
    /// The task sets none, or the backend does not say.
    None,
    /// The backend caps the sandbox's writable layer at this many MB, and
    /// the sandbox's own filesystem shows the cap, smaller than the disk
    /// the backend keeps its containers on.
    Enforced(u64),
    /// The backend gives each sandbox a disk of its own of this many MB
    /// (Daytona), which holds the image too.
    Own(u64),
    /// The task asks for this many MB and the backend does not cap it (the
    /// reason is in the text): BenchFlow measures the writable layer from
    /// the agent's first turn to the end of grading and stops the sandbox
    /// past the limit.
    NotEnforced(u64, String),
}

impl DiskLimit {
    /// The limit in MB, when there is one.
    pub fn mb(&self) -> Option<u64> {
        match self {
            DiskLimit::None => None,
            DiskLimit::Enforced(mb) | DiskLimit::Own(mb) | DiskLimit::NotEnforced(mb, _) => {
                Some(*mb)
            }
        }
    }
}

/// What one added file, folder or link takes at the least on a layer the
/// backend does not cap: one 4 KiB block.
pub const ENTRY_BYTES: u64 = 4096;

/// What one added entry is charged under a cap the backend enforces, where
/// the bytes are the cap's and only an inode flood needs the watch: half a
/// block. Inside an XFS quota every file with data takes at least one 4 KiB
/// block, so a tree charged past the cap this way is mostly empty files,
/// folders or links; a sparse file is never charged its size.
pub const ENTRY_BYTES_CAPPED: u64 = 2048;

const MB: u64 = 1024 * 1024;

/// A sandbox's writable layer, as its backend measures it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayerUsage {
    /// The bytes of the files in it, as their sizes say (a sparse file
    /// counts at its full size); `None` when not measured (never under a
    /// cap the backend enforces).
    pub bytes: Option<u64>,
    /// How many files, folders and links it added; `None` when they were
    /// not counted.
    pub entries: Option<u64>,
}

impl LayerUsage {
    /// The disk it is charged, in bytes.
    /// - [`Watched::Layer`]: its bytes, or a block per entry when that is
    ///   more (an inode flood), not both.
    /// - [`Watched::Entries`]: half a block per entry
    ///   ([`ENTRY_BYTES_CAPPED`]); the bytes are the cap's.
    ///
    /// What was not measured counts as none.
    pub fn charged(&self, watched: Watched) -> u64 {
        let entries = self.entries.unwrap_or(0);
        match watched {
            Watched::Layer => self
                .bytes
                .unwrap_or(0)
                .max(entries.saturating_mul(ENTRY_BYTES)),
            Watched::Entries => entries.saturating_mul(ENTRY_BYTES_CAPPED),
        }
    }

    /// [`Self::charged`] in whole MB, rounded up.
    pub fn mb(&self, watched: Watched) -> u64 {
        self.charged(watched).div_ceil(MB)
    }
}

/// What the watch measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Watched {
    /// The bytes and the entries: the backend does not cap the layer.
    Layer,
    /// The entries alone: the backend caps the layer's blocks, and the
    /// inodes its files take are outside that cap (XFS project quotas
    /// charge the blocks, not the inodes). The bytes are never measured.
    Entries,
}

/// How the watch paces itself. The timeouts and bounds of the live measures
/// are fixed per trial from the base measure ([`bounds`]).
#[derive(Debug, Clone, Copy)]
pub struct Pace {
    /// How often it measures the bytes at the most often.
    pub bytes: Duration,
    /// How often it counts the entries at the most often. Counting can take
    /// seconds on a layer being written (`docker diff` reads what changed),
    /// so it runs on its own.
    pub entries: Duration,
    /// The longest wait between two measures of one kind, however slow they
    /// are: a slow measure slows the next one (three times its time), up to
    /// this, or the kind's own pace when that is longer.
    pub most: Duration,
    /// The shortest timeout a live measure is given.
    pub floor: Duration,
    /// How long each base measure may take (before the agent ever ran, so
    /// nothing of the agent's slows it).
    pub base_timeout: Duration,
    /// Under a cap, a base count of the entries that failed is tried once
    /// more with this timeout: the entries are the only guard there.
    pub entries_retry: Duration,
    /// The bytes' timeout: 2 × the base measure's time, at least this.
    pub bytes_timeout: Duration,
    /// How long the bytes may go unmeasured: 3 × the base, at least this.
    pub bytes_bound: Duration,
    /// The entries' timeout: 2 × the base count's time, at least this.
    pub entries_timeout: Duration,
    /// How long the entries may go uncounted: 3 × the base, at least this,
    /// and at least two timeouts and a pace (see [`bounds`]).
    pub entries_bound: Duration,
    /// How long the end of the watch waits for a stop already under way.
    pub settle: Duration,
}

/// The pace on Docker.
/// - Bytes every second, entries every 15 s; a slow measure backs off to at
///   most 10 s, and a byte measure also waits while a count runs. A writer at disk speed passes the limit by what it
///   writes in one interval and the seconds the stop takes: only a cap the
///   backend enforces holds to the byte.
/// - Before the agent's first turn, each kind is measured with 2 minutes to
///   take (the entries 3 more under a cap). Each kind's timeout is then 2×
///   its base and its bound 3×, never less than 20 s / 30 s for the bytes
///   and 60 s / 120 s for the entries: on a healthy daemon the bytes may go
///   unmeasured 30 s (500 MB/s for 30 s is 15 GB). A daemon slow before the
///   trial keeps the same headroom (review F2-D-r4 M2): a 100 s base count
///   gets 200 s counts and a 415 s bound (the entries' bound is at least two
///   timeouts and a pace: 135 s on a healthy daemon), and a slow base widens
///   the blind window in proportion.
/// - After a count that times out, the next waits for the daemon's walk, at
///   most half the time left, so the second count gets half a timeout: 30 s
///   on a healthy daemon, the base count's time on a slow one. Counts longer
///   than that, with one stall, are stopped as unmeasurable (R110, a rerun),
///   never the agent's.
/// - The two kinds never measure at once (on Docker a `docker diff` holds
///   the container's lock, and `inspect --size` waits behind it), and the
///   time a measure waits for the other kind is not counted as unmeasured.
pub const PACE: Pace = Pace {
    bytes: Duration::from_secs(1),
    entries: Duration::from_secs(15),
    most: Duration::from_secs(10),
    floor: Duration::from_secs(1),
    base_timeout: Duration::from_secs(120),
    entries_retry: Duration::from_secs(180),
    bytes_timeout: Duration::from_secs(20),
    bytes_bound: Duration::from_secs(30),
    entries_timeout: Duration::from_secs(60),
    entries_bound: Duration::from_secs(120),
    settle: Duration::from_secs(120),
};

/// The timeouts and bounds of one trial's watch, and the base measures'
/// times they come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    /// How long the base measure of the bytes took (`None`: not taken or
    /// failed).
    pub base_bytes: Option<Duration>,
    /// How long the base count of the entries took.
    pub base_entries: Option<Duration>,
    /// A live measure of the bytes' timeout.
    pub bytes_timeout: Duration,
    /// How long the bytes may go unmeasured before an unmeasurable stop.
    pub bytes_bound: Duration,
    /// A live count of the entries' timeout.
    pub entries_timeout: Duration,
    /// How long the entries may go uncounted before an unmeasurable stop.
    pub entries_bound: Duration,
}

/// The bounds for base measures that took `base_bytes` and `base_entries`:
/// each kind's timeout is twice its base, its bound three times, each at
/// least the pace's minimum; the entries' bound is at least two timeouts and
/// a pace, so a count that times out leaves a second count half a timeout.
/// A base not taken counts as instant.
pub fn bounds(pace: &Pace, base_bytes: Option<Duration>, base_entries: Option<Duration>) -> Bounds {
    let within = |base: Option<Duration>, times: u32, lo: Duration| -> Duration {
        base.unwrap_or_default().saturating_mul(times).max(lo)
    };
    let entries_timeout = within(base_entries, 2, pace.entries_timeout);
    Bounds {
        base_bytes,
        base_entries,
        bytes_timeout: within(base_bytes, 2, pace.bytes_timeout),
        bytes_bound: within(base_bytes, 3, pace.bytes_bound),
        entries_timeout,
        // Room for a count that times out, then a second count of half a
        // timeout (review F2-A-r6 M1, F2-A-r67 M1).
        entries_bound: within(base_entries, 3, pace.entries_bound)
            .max(entries_timeout * 2 + pace.entries),
    }
}

/// One sandbox's writable layer, as its backend measures and stops it from
/// outside the sandbox.
#[async_trait]
pub trait Layer: Send + Sync {
    /// The bytes of its files, within `timeout`.
    async fn bytes(&self, timeout: Duration) -> Result<u64, SandboxError>;
    /// How many files, folders and links it added, within `timeout`.
    async fn entries(&self, timeout: Duration) -> Result<u64, SandboxError>;
    /// Whether the backend says the sandbox still runs (paused counts as
    /// running): `Ok(false)` when it says it stopped or no longer has it;
    /// an error when it does not answer (the watch asks again).
    async fn status(&self) -> Result<bool, SandboxError>;
    /// Stop the whole sandbox at once.
    async fn stop(&self) -> Result<(), SandboxError>;
    /// Thaw a sandbox whose stop was cut short (the stop freezes first).
    async fn unpause(&self) -> Result<(), SandboxError>;
}

/// How a watch started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartState {
    /// Watched, both kinds as [`Watched`] says.
    Watched,
    /// The base count of the entries failed on a layer whose bytes are
    /// watched: the bytes alone are watched for the trial.
    EntriesUnwatched,
    /// The base measure failed (why): nothing guards the limit, so the
    /// agent must not run. No window is open and nothing is watched.
    Unmeasurable(String),
}

/// A stop of the sandbox on a measure over the limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stop {
    /// What the measure found.
    pub usage: LayerUsage,
    /// The measure began while a window of the agent's was open (a window's
    /// closing measure included).
    pub in_window: bool,
    /// What the agent's windows grew the layer by up to the stop, in
    /// charged bytes: the closed windows', and this one's from its opening
    /// measure to the stop.
    pub growth_at_stop: u64,
}

/// What a watch saw, from the agent's first turn to its end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchReport {
    /// The limit, in MB.
    pub limit_mb: u64,
    /// What was watched.
    pub watched: Watched,
    /// The timeouts and bounds, and the base measures' times.
    pub bounds: Bounds,
    /// The layer before the agent's first turn.
    pub first: Option<LayerUsage>,
    /// The layer as last measured.
    pub last: Option<LayerUsage>,
    /// What the agent's closed windows grew the layer by, in charged bytes.
    pub growth: u64,
    /// The stop on a measure over the limit, when there was one.
    pub stopped: Option<Stop>,
    /// Set when the watch stopped the sandbox because its layer could not
    /// be measured for its bound while the backend said it still ran: how
    /// long it had gone unmeasured then.
    pub unmeasurable: Option<Duration>,
    /// The stop itself failed or did not finish.
    pub stop_error: Option<String>,
    /// The longest stretch without a successful measure of the bytes, when
    /// longer than their timeout.
    pub blind: Option<Duration>,
    /// The same for the entries.
    pub entries_blind: Option<Duration>,
    /// The base count of the entries failed: the bytes alone were watched.
    pub entries_unwatched: bool,
}

impl WatchReport {
    /// [`Self::growth`] in whole MB, rounded down.
    pub fn growth_mb(&self) -> u64 {
        self.growth / MB
    }
}

/// When one kind of measure last succeeded, and its longest gap so far.
#[derive(Debug, Clone, Copy)]
struct Seen {
    last: Instant,
    longest: Duration,
}

impl Seen {
    fn new() -> Self {
        Self {
            last: Instant::now(),
            longest: Duration::ZERO,
        }
    }
    fn now(&mut self) {
        self.longest = self.longest.max(self.last.elapsed());
        self.last = Instant::now();
    }
    /// The longest gap, the current one counted up to `until`.
    fn longest(&self, until: Instant) -> Duration {
        self.longest.max(until.saturating_duration_since(self.last))
    }
}

/// One window of the agent's: the layer charged when it opened (`None`:
/// that measure failed, and the window is charged no growth), and what the
/// windows before it grew.
#[derive(Debug, Clone, Copy)]
struct Window {
    open: Option<u64>,
    prior: u64,
}

#[derive(Debug)]
struct State {
    /// The latest of each measure.
    bytes: Option<u64>,
    entries: Option<u64>,
    seen: [Seen; 2],
    /// The open window, an index into `windows`.
    window: Option<usize>,
    windows: Vec<Window>,
    /// What the closed windows grew, in charged bytes.
    growth: u64,
    stopped: Option<Stop>,
    unmeasurable: Option<Duration>,
    stop_error: Option<String>,
    /// A stop under way.
    stopping: bool,
    /// When the watch decided to stop the sandbox: the gaps end there.
    halted_at: Option<Instant>,
    /// The watch is ending: no stop starts any more.
    ending: bool,
    /// BenchFlow is writing its own archives into the layer: a measure over
    /// the limit then stops nothing (review F2-A-r4 M2).
    held: u32,
    /// Bumped by every hold and release: a measure across a change saw the
    /// archive come or go (review F2-A-r5 M2).
    hold_epoch: u64,
}

impl State {
    fn halted(&self) -> bool {
        self.stopped.is_some() || self.unmeasurable.is_some()
    }
}

/// The two kinds of measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Bytes = 0,
    Entries = 1,
}

/// Why the watch stops the sandbox.
enum Why {
    /// A measure over the limit, begun in this window (if any).
    Over(LayerUsage, Option<usize>),
    /// Unmeasured for this long.
    Unmeasurable(Duration),
}

/// What the watch's tasks share.
struct Shared {
    limit_mb: u64,
    watched: Watched,
    /// The entries are counted (always under [`Watched::Entries`]).
    entries_watched: bool,
    pace: Pace,
    bounds: Bounds,
    layer: Arc<dyn Layer>,
    state: Mutex<State>,
    /// One measure at a time (review F2-B-r4 M2).
    measuring: tokio::sync::Mutex<()>,
}

/// One measure of `kind`, bounded by `timeout`, and how long it took.
async fn timed(
    layer: &dyn Layer,
    kind: Kind,
    timeout: Duration,
) -> (Result<u64, SandboxError>, Duration) {
    let clock = Instant::now();
    let m = match kind {
        Kind::Bytes => layer.bytes(timeout),
        Kind::Entries => layer.entries(timeout),
    };
    let got = match tokio::time::timeout(timeout, m).await {
        Ok(r) => r,
        Err(_) => Err(SandboxError::Transport(format!(
            "the layer's measure did not finish within {}s",
            timeout.as_secs()
        ))),
    };
    (got, clock.elapsed())
}

impl Shared {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The kinds this watch measures.
    fn kinds(&self) -> &'static [Kind] {
        match (self.watched, self.entries_watched) {
            (Watched::Layer, true) => &[Kind::Bytes, Kind::Entries],
            (Watched::Layer, false) => &[Kind::Bytes],
            (Watched::Entries, _) => &[Kind::Entries],
        }
    }

    fn bound(&self, kind: Kind) -> Duration {
        match kind {
            Kind::Bytes => self.bounds.bytes_bound,
            Kind::Entries => self.bounds.entries_bound,
        }
    }

    fn record(&self, kind: Kind, v: u64) -> LayerUsage {
        let mut s = self.state();
        s.seen[kind as usize].now();
        match kind {
            Kind::Bytes => s.bytes = Some(v),
            Kind::Entries => s.entries = Some(v),
        }
        LayerUsage {
            bytes: s.bytes,
            entries: s.entries,
        }
    }

    /// The layer as last measured, when every kind watched has been.
    fn usage(&self) -> Option<LayerUsage> {
        let s = self.state();
        let u = LayerUsage {
            bytes: s.bytes,
            entries: s.entries,
        };
        let known = match self.watched {
            Watched::Layer => u.bytes.is_some(),
            Watched::Entries => u.entries.is_some(),
        };
        known.then_some(u)
    }

    fn over(&self, u: LayerUsage) -> bool {
        u.charged(self.watched) > self.limit_mb.saturating_mul(MB)
    }

    /// How long `kind` has gone without a successful measure.
    fn unmeasured(&self, kind: Kind) -> Duration {
        self.state().seen[kind as usize].last.elapsed()
    }

    /// Every watched kind now, side by side, each within its bound; each
    /// one that succeeds is recorded. The layer when all succeeded.
    async fn measure_all(&self) -> Option<LayerUsage> {
        let _one = self.measuring.lock().await;
        let mut all = true;
        for kind in self.kinds() {
            let (got, _) = timed(self.layer.as_ref(), *kind, self.bound(*kind)).await;
            match got {
                Ok(v) => {
                    self.record(*kind, v);
                }
                Err(_) => all = false,
            }
        }
        if all { self.usage() } else { None }
    }

    /// Record why, then stop the sandbox, once, unless the watch is ending.
    async fn halt(&self, why: Why) {
        {
            let mut s = self.state();
            if s.halted() || s.ending {
                return;
            }
            match why {
                Why::Over(usage, window) => {
                    let charged = usage.charged(self.watched);
                    let growth_at_stop = match window.and_then(|i| s.windows.get(i)) {
                        Some(w) => w.prior + w.open.map_or(0, |o| charged.saturating_sub(o)),
                        None => s.growth,
                    };
                    s.stopped = Some(Stop {
                        usage,
                        in_window: window.is_some(),
                        growth_at_stop,
                    });
                }
                Why::Unmeasurable(blind) => s.unmeasurable = Some(blind),
            }
            s.halted_at = Some(Instant::now());
            s.stopping = true;
        }
        let stopped = self.layer.stop().await;
        let mut s = self.state();
        s.stopping = false;
        if let Err(e) = stopped {
            s.stop_error = Some(e.to_string());
        }
    }

    /// Measure `kind` at its pace until the watch ends or stops the
    /// sandbox.
    async fn watch(self: Arc<Self>, kind: Kind, mut end: tokio::sync::watch::Receiver<bool>) {
        let p = self.pace;
        let (every, timeout, bound) = match kind {
            Kind::Bytes => (p.bytes, self.bounds.bytes_timeout, self.bounds.bytes_bound),
            Kind::Entries => (
                p.entries,
                self.bounds.entries_timeout,
                self.bounds.entries_bound,
            ),
        };
        let most = p.most.max(every);
        let mut wait = every;
        loop {
            tokio::select! {
                _ = tokio::time::sleep(wait) => {}
                _ = end.changed() => return,
            }
            // One measure at a time; the time spent waiting for the other
            // kind's is not this kind's to answer for.
            let asked = Instant::now();
            let _turn = tokio::select! {
                t = self.measuring.lock() => t,
                _ = end.changed() => return,
            };
            let waited = asked.elapsed();
            {
                let mut s = self.state();
                if s.halted() {
                    return;
                }
                let seen = &mut s.seen[kind as usize];
                seen.last = (seen.last + waited).min(Instant::now());
            }
            let unmeasured = self.unmeasured(kind);
            let left = bound.saturating_sub(unmeasured);
            if left.is_zero() {
                drop(_turn);
                // Unmeasured for the whole bound: stopped while the backend
                // says it runs; left to the trial's own check of the sandbox
                // when it says it does not (round 1's F2A-B M2); asked again
                // when it does not answer (review F2-A-r4 M3).
                match self.layer.status().await {
                    Ok(true) => {
                        self.halt(Why::Unmeasurable(unmeasured)).await;
                        return;
                    }
                    Ok(false) => return,
                    Err(_) => {
                        wait = most;
                        continue;
                    }
                }
            }
            // The window the measure begins in, and whether BenchFlow's own
            // archive could be in the layer.
            let (window, held, epoch) = {
                let s = self.state();
                (s.window, s.held, s.hold_epoch)
            };
            let got = tokio::select! {
                got = timed(self.layer.as_ref(), kind, timeout.min(left).max(p.floor)) => got,
                _ = end.changed() => return,
            };
            drop(_turn);
            match got {
                // Begun while BenchFlow's archive was in the layer, or across
                // its coming or going: the daemon answered, so the kind is
                // not unmeasured, but the reading may hold the archive and is
                // never kept (review F2-A-r5 M2, F2-B-r5 M1).
                (Ok(_), took) if held > 0 || self.state().hold_epoch != epoch => {
                    wait = (took * 3).clamp(every, most);
                    self.state().seen[kind as usize].now();
                }
                (Ok(v), took) => {
                    // Slow but answered: back off, within bounds.
                    wait = (took * 3).clamp(every, most);
                    let usage = self.record(kind, v);
                    // The agent's only if its window is still the open one
                    // (review F2-A-r4 M1: a measure that ends after the
                    // close also sees what BenchFlow wrote since).
                    let window = window.filter(|w| self.state().window == Some(*w));
                    if self.over(usage) {
                        self.halt(Why::Over(usage, window)).await;
                        return;
                    }
                }
                // Next at the pace, or at the bound if that comes first. A
                // failed count waits for the daemon's walk to end (review
                // F2-B-r4 M2), but at most half the time left, so a second
                // count always runs within the bound (review F2-A-r5 M1).
                (Err(_), _) => {
                    let left = bound.saturating_sub(self.unmeasured(kind));
                    wait = if kind == Kind::Entries {
                        timeout.min(left / 2)
                    } else {
                        every.min(left)
                    };
                }
            }
        }
    }

    /// After the watch's tasks were cut short: a stop under way is recorded
    /// as not finished, and the sandbox thawed (the stop freezes it first),
    /// with one more try.
    async fn after_cut(&self) {
        let stopping = self.state().stopping;
        if !stopping {
            return;
        }
        self.state().stop_error = Some(format!(
            "the stop did not finish within {}s",
            self.pace.settle.as_secs()
        ));
        for _ in 0..2 {
            if self.layer.unpause().await.is_ok() {
                break;
            }
        }
    }
}

/// A running watch of one sandbox's writable layer, from the agent's first
/// turn to the end of the trial. Dropping it ends it: a trial stopped or
/// ended by a panic leaves nothing measuring, or starting a stop of, its
/// sandbox (one kept for debugging included); a stop already under way is
/// let finish, within [`Pace::settle`].
pub struct LayerWatch {
    shared: Arc<Shared>,
    first: Option<LayerUsage>,
    entries_unwatched: bool,
    end: tokio::sync::watch::Sender<bool>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl std::fmt::Debug for LayerWatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerWatch")
            .field("limit_mb", &self.shared.limit_mb)
            .field("watched", &self.shared.watched)
            .field("first", &self.first)
            .finish_non_exhaustive()
    }
}

impl Drop for LayerWatch {
    fn drop(&mut self) {
        self.shared.state().ending = true;
        let _ = self.end.send(true);
        let tasks = std::mem::take(&mut self.tasks);
        if tasks.is_empty() {
            return;
        }
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                let shared = self.shared.clone();
                rt.spawn(async move { settle(&shared, tasks).await });
            }
            Err(_) => tasks.iter().for_each(|t| t.abort()),
        }
    }
}

/// Wait, within [`Pace::settle`], for tasks told to end: a stop under way
/// finishes; then cut what is left, and thaw.
async fn settle(shared: &Shared, tasks: Vec<tokio::task::JoinHandle<()>>) {
    let deadline = Instant::now() + shared.pace.settle;
    let mut cut = false;
    for mut t in tasks {
        if tokio::time::timeout_at(deadline, &mut t).await.is_err() {
            t.abort();
            cut = true;
        }
    }
    if cut {
        shared.after_cut().await;
    }
}

impl LayerWatch {
    /// Take the base measure (side by side, before the agent's first turn:
    /// [`Pace::base_timeout`] each, and under a cap a second try of the
    /// entries), fix the trial's [`bounds`] from it, open the agent's first
    /// window, and watch the layer in the background. Past `limit_mb` on a
    /// measure that succeeded, the sandbox is stopped. The limit is on the
    /// whole layer, so the turns of a task with stages cannot add up past
    /// it.
    pub async fn start(
        limit_mb: u64,
        pace: Pace,
        watched: Watched,
        layer: Arc<dyn Layer>,
    ) -> (Self, StartState) {
        let base_bytes = async {
            if watched != Watched::Layer {
                return None;
            }
            match timed(layer.as_ref(), Kind::Bytes, pace.base_timeout).await {
                (Ok(v), took) => Some((v, took)),
                (Err(_), _) => None,
            }
        };
        let base_entries = async {
            let mut tries = vec![pace.base_timeout];
            if watched == Watched::Entries {
                tries.push(pace.entries_retry);
            }
            for t in tries {
                if let (Ok(v), took) = timed(layer.as_ref(), Kind::Entries, t).await {
                    return Some((v, took));
                }
            }
            None
        };
        // One after the other: on Docker a count holds the container's lock.
        let b = base_bytes.await;
        let e = base_entries.await;
        let state = match (watched, b, e) {
            (Watched::Layer, None, _) => StartState::Unmeasurable(format!(
                "the writable layer's bytes could not be measured within {} s before the agent's turn",
                pace.base_timeout.as_secs()
            )),
            (Watched::Entries, _, None) => StartState::Unmeasurable(format!(
                "the writable layer's entries could not be counted within {} s, nor {} s, before the agent's turn",
                pace.base_timeout.as_secs(),
                pace.entries_retry.as_secs()
            )),
            (Watched::Layer, _, None) => StartState::EntriesUnwatched,
            _ => StartState::Watched,
        };
        let entries_unwatched = state == StartState::EntriesUnwatched;
        let shared = Arc::new(Shared {
            limit_mb,
            watched,
            entries_watched: watched == Watched::Entries || e.is_some(),
            pace,
            bounds: bounds(&pace, b.map(|x| x.1), e.map(|x| x.1)),
            layer,
            measuring: tokio::sync::Mutex::new(()),
            state: Mutex::new(State {
                bytes: b.map(|x| x.0),
                entries: e.map(|x| x.0),
                // The gaps count from the end of the base measure.
                seen: [Seen::new(), Seen::new()],
                window: None,
                windows: Vec::new(),
                growth: 0,
                stopped: None,
                unmeasurable: None,
                stop_error: None,
                stopping: false,
                halted_at: None,
                ending: false,
                held: 0,
                hold_epoch: 0,
            }),
        });
        let first = shared.usage();
        let (end, ended) = tokio::sync::watch::channel(false);
        let mut watch = Self {
            shared: shared.clone(),
            first,
            entries_unwatched,
            end,
            tasks: Vec::new(),
        };
        if matches!(state, StartState::Unmeasurable(_)) {
            return (watch, state);
        }
        {
            let mut s = shared.state();
            s.windows.push(Window {
                open: first.map(|u| u.charged(watched)),
                prior: 0,
            });
            s.window = Some(0);
        }
        match first.filter(|u| shared.over(*u)) {
            // Already past the limit before the turn (BenchFlow's setup, an
            // image's files): stopped at once, and not inside a window.
            Some(u) => watch.tasks.push(tokio::spawn(async move {
                shared.halt(Why::Over(u, None)).await;
            })),
            None => {
                for kind in shared.kinds() {
                    watch
                        .tasks
                        .push(tokio::spawn(shared.clone().watch(*kind, ended.clone())));
                }
            }
        }
        (watch, state)
    }

    /// Open a window of the agent's (a later turn): measure, then open.
    /// A layer over the limit then is stopped, outside the window. A failed
    /// measure charges the window no growth.
    pub async fn open_window(&self) {
        let shared = &self.shared;
        if shared.state().window.is_some() {
            return;
        }
        let epoch = shared.state().hold_epoch;
        let at_open = shared.measure_all().await;
        let clean = {
            let s = shared.state();
            s.held == 0 && s.hold_epoch == epoch
        };
        if let Some(u) = at_open.filter(|u| clean && shared.over(*u)) {
            shared.halt(Why::Over(u, None)).await;
        }
        let mut s = shared.state();
        if s.window.is_some() {
            return;
        }
        let prior = s.growth;
        s.windows.push(Window {
            open: at_open.map(|u| u.charged(shared.watched)),
            prior,
        });
        s.window = Some(s.windows.len() - 1);
    }

    /// Close the agent's open window (no-op when none is): measure, stop a
    /// layer over the limit (inside the window), and add what the window
    /// grew. A kind whose measure fails keeps its last good value.
    pub async fn close_window(&self) {
        let shared = &self.shared;
        let (window, halted) = {
            let s = shared.state();
            (s.window, s.halted())
        };
        let Some(w) = window else {
            return;
        };
        if !halted {
            shared.measure_all().await;
        }
        let at_close = shared.usage();
        if !halted && let Some(u) = at_close.filter(|u| shared.over(*u)) {
            shared.halt(Why::Over(u, Some(w))).await;
        }
        let mut s = shared.state();
        if s.window != Some(w) {
            return;
        }
        let grew = match (s.windows[w].open, at_close) {
            (Some(open), Some(u)) if !halted => u.charged(shared.watched).saturating_sub(open),
            _ => 0,
        };
        s.growth += grew;
        s.window = None;
    }

    /// The layer before the agent's first turn.
    pub fn first(&self) -> Option<LayerUsage> {
        self.first
    }

    /// BenchFlow writes its own archives into the layer (the session, the
    /// outputs): until [`Self::release`], and for any measure begun before
    /// it, no reading is kept and nothing is stopped for any write, the
    /// agent's included. The archives are removed right after they are
    /// copied out, so they never count against the task's limit (review
    /// F2-A-r4 M2).
    pub fn hold(&self) {
        let mut s = self.shared.state();
        s.held += 1;
        s.hold_epoch += 1;
    }

    /// The end of a [`Self::hold`].
    pub fn release(&self) {
        let mut s = self.shared.state();
        s.held = s.held.saturating_sub(1);
        s.hold_epoch += 1;
    }

    /// Whether the watch stopped the sandbox (or began to).
    pub fn halted(&self) -> bool {
        self.shared.state().halted()
    }

    /// End the watch: no stop starts any more, a stop under way finishes
    /// (within [`Pace::settle`]), and the report is made. It measures
    /// nothing.
    pub async fn finish(mut self) -> WatchReport {
        self.shared.state().ending = true;
        let _ = self.end.send(true);
        let tasks = std::mem::take(&mut self.tasks);
        let shared = self.shared.clone();
        settle(&shared, tasks).await;
        let last = shared.usage();
        let s = shared.state();
        let b = shared.bounds;
        let until = s.halted_at.unwrap_or_else(Instant::now);
        let blind = s.seen[Kind::Bytes as usize].longest(until);
        let entries_blind = s.seen[Kind::Entries as usize].longest(until);
        let watched = shared.kinds();
        WatchReport {
            limit_mb: shared.limit_mb,
            watched: shared.watched,
            bounds: b,
            first: self.first,
            last,
            growth: s.growth,
            stopped: s.stopped,
            unmeasurable: s.unmeasurable,
            stop_error: s.stop_error.clone(),
            blind: (watched.contains(&Kind::Bytes) && blind > b.bytes_timeout).then_some(blind),
            entries_blind: (watched.contains(&Kind::Entries) && entries_blind > b.entries_timeout)
                .then_some(entries_blind),
            entries_unwatched: self.entries_unwatched,
        }
    }
}

/// Free space under this many MB is a full disk: room a verifier's install
/// cannot work in.
pub const FULL_MARGIN_MB: u64 = 64;

/// The inodes of one filesystem, as `df -Pi` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inodes {
    /// Counted.
    Known { total: u64, used: u64, free: u64 },
    /// The filesystem has no fixed number of inodes (btrfs says 0).
    NotApplicable,
    /// `df -Pi` did not run or say.
    Unknown,
}

/// One filesystem, as `df -Pk` and `df -Pi` say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfRow {
    pub total_kb: u64,
    pub used_kb: u64,
    pub avail_kb: u64,
    pub inodes: Inodes,
}

impl DfRow {
    /// Whether it is full: under [`FULL_MARGIN_MB`] free, or its inodes
    /// under `min(16384, total / 50)` free (a failed install's cleanup can
    /// free thousands before anyone reads `df`). Inodes not counted say
    /// only "full" for full blocks, else `None`.
    pub fn full(&self) -> Option<bool> {
        let blocks = self.avail_kb / 1024 < FULL_MARGIN_MB;
        match self.inodes {
            Inodes::Known { total, free, .. } if total > 0 => {
                Some(blocks || free < (total / 50).min(16_384))
            }
            Inodes::Known { .. } | Inodes::NotApplicable => Some(blocks),
            Inodes::Unknown => blocks.then_some(true),
        }
    }
}

/// The line between the two parts of [`df_script`]'s output.
pub const DF_INODES: &str = "BF-I";

/// A script that prints `df -Pk`, then [`DF_INODES`], then `df -Pi`, for
/// each of `paths`, with `df` from the system folders by full path.
pub fn df_script(paths: &[&str]) -> String {
    let paths: Vec<String> = paths.iter().map(|p| crate::shell::quote(p)).collect();
    let paths = paths.join(" ");
    format!(
        "d=/bin/df; [ -x \"$d\" ] || d=/usr/bin/df; \"$d\" -Pk {paths}; echo {DF_INODES}; \"$d\" -Pi {paths}"
    )
}

/// The rows of [`df_script`]'s output (or of `df -Pk` alone), one per
/// filesystem in the order asked; the inodes are matched by position.
pub fn parse_df(out: &str) -> Vec<DfRow> {
    let (blocks, inodes) = match out.split_once(&format!("\n{DF_INODES}\n")) {
        Some((b, i)) => (b, Some(i)),
        None => (out.strip_suffix(DF_INODES).unwrap_or(out), None),
    };
    let rows = |part: &str| -> Vec<Vec<String>> {
        part.lines()
            .filter(|l| !l.trim().is_empty())
            .skip(1)
            .map(|l| l.split_whitespace().map(str::to_string).collect())
            .collect()
    };
    let inode_rows = inodes.map(rows).unwrap_or_default();
    rows(blocks)
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            let n = |k: usize| r.get(k)?.parse::<u64>().ok();
            let inodes = match inode_rows.get(i) {
                Some(ir) => {
                    let n = |k: usize| ir.get(k)?.parse::<u64>().ok();
                    match (n(1), n(2), n(3)) {
                        (Some(0), _, _) => Inodes::NotApplicable,
                        (Some(total), Some(used), Some(free)) => {
                            Inodes::Known { total, used, free }
                        }
                        _ => Inodes::Unknown,
                    }
                }
                None => Inodes::Unknown,
            };
            Some(DfRow {
                total_kb: n(1)?,
                used_kb: n(2)?,
                avail_kb: n(3)?,
                inodes,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    /// One measure of a fake layer: how long it takes, and what it gives
    /// (`Err` fails it).
    type Answer = (Duration, Result<u64, ()>);

    /// A fake layer: each measure is `bytes(i, timeout)` or `entries(i,
    /// timeout)` for the i-th call of its kind. Every call is recorded with
    /// the time it began (from the fake's making) and its timeout.
    #[allow(clippy::type_complexity)]
    struct Fake {
        t0: Instant,
        bytes: Box<dyn Fn(usize, Duration) -> Answer + Send + Sync>,
        entries: Box<dyn Fn(usize, Duration) -> Answer + Send + Sync>,
        status: Box<dyn Fn() -> Result<bool, ()> + Send + Sync>,
        /// How long a stop takes.
        stop_takes: Duration,
        n_bytes: AtomicUsize,
        n_entries: AtomicUsize,
        calls: Mutex<Vec<(Duration, String)>>,
    }

    impl Fake {
        fn new(bytes: impl Fn(usize, Duration) -> Answer + Send + Sync + 'static) -> Self {
            Self {
                t0: Instant::now(),
                bytes: Box::new(bytes),
                entries: Box::new(|_, _| (Duration::ZERO, Ok(0))),
                status: Box::new(|| Ok(true)),
                stop_takes: Duration::ZERO,
                n_bytes: AtomicUsize::new(0),
                n_entries: AtomicUsize::new(0),
                calls: Mutex::new(Vec::new()),
            }
        }
        fn entries(
            mut self,
            f: impl Fn(usize, Duration) -> Answer + Send + Sync + 'static,
        ) -> Self {
            self.entries = Box::new(f);
            self
        }
        fn status(mut self, f: impl Fn() -> Result<bool, ()> + Send + Sync + 'static) -> Self {
            self.status = Box::new(f);
            self
        }
        fn note(&self, what: impl Into<String>) {
            let at = self.t0.elapsed();
            self.calls.lock().unwrap().push((at, what.into()));
        }
        fn names(&self) -> Vec<String> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .map(|c| c.1.clone())
                .collect()
        }
        fn count(&self, what: &str) -> usize {
            self.names().iter().filter(|c| *c == what).count()
        }
        fn starts_with(&self, what: &str) -> Vec<(Duration, String)> {
            let calls = self.calls.lock().unwrap();
            calls
                .iter()
                .filter(|c| c.1.starts_with(what))
                .cloned()
                .collect()
        }
        /// When the first call named `what` began.
        fn at(&self, what: &str) -> Option<Duration> {
            let calls = self.calls.lock().unwrap();
            calls.iter().find(|c| c.1 == what).map(|c| c.0)
        }
    }

    fn failed() -> SandboxError {
        SandboxError::Transport("inspect failed".into())
    }

    #[async_trait]
    impl Layer for Fake {
        async fn bytes(&self, timeout: Duration) -> Result<u64, SandboxError> {
            let i = self.n_bytes.fetch_add(1, Ordering::SeqCst);
            self.note(format!("bytes {}s", timeout.as_secs()));
            let (takes, r) = (self.bytes)(i, timeout);
            tokio::time::sleep(takes).await;
            r.map_err(|_| failed())
        }
        async fn entries(&self, timeout: Duration) -> Result<u64, SandboxError> {
            let i = self.n_entries.fetch_add(1, Ordering::SeqCst);
            self.note(format!("entries {}s", timeout.as_secs()));
            let (takes, r) = (self.entries)(i, timeout);
            tokio::time::sleep(takes).await;
            r.map_err(|_| failed())
        }
        async fn status(&self) -> Result<bool, SandboxError> {
            self.note("status");
            (self.status)().map_err(|_| SandboxError::Transport("no such container".into()))
        }
        async fn stop(&self) -> Result<(), SandboxError> {
            self.note("stop begins");
            tokio::time::sleep(self.stop_takes).await;
            self.note("stop");
            Ok(())
        }
        async fn unpause(&self) -> Result<(), SandboxError> {
            self.note("unpause");
            Ok(())
        }
    }

    const S: Duration = Duration::from_secs(1);

    fn quick(v: u64) -> Answer {
        (Duration::ZERO, Ok(v))
    }

    /// A layer whose bytes are whatever `size` holds, in MB, measured at
    /// once.
    fn sized(size: &Arc<AtomicU64>) -> Fake {
        let size = size.clone();
        Fake::new(move |_, _| quick(size.load(Ordering::SeqCst) * MB))
    }

    async fn start(
        limit: u64,
        watched: Watched,
        fake: Fake,
    ) -> (Arc<Fake>, LayerWatch, StartState) {
        let fake = Arc::new(fake);
        let layer: Arc<dyn Layer> = fake.clone();
        let (w, state) = LayerWatch::start(limit, PACE, watched, layer).await;
        (fake, w, state)
    }

    /// The bounds come from the base measure: 2× and 3× its time, at least
    /// the pace's minimums, with no ceiling (review F2-D-r4 M2).
    #[test]
    fn the_bounds_come_from_the_base_measure() {
        let b = bounds(&PACE, Some(S), Some(S));
        assert_eq!(
            (
                b.bytes_timeout,
                b.bytes_bound,
                b.entries_timeout,
                b.entries_bound
            ),
            (20 * S, 30 * S, 60 * S, 135 * S)
        );
        let b = bounds(&PACE, Some(25 * S), Some(50 * S));
        assert_eq!(
            (
                b.bytes_timeout,
                b.bytes_bound,
                b.entries_timeout,
                b.entries_bound
            ),
            (50 * S, 75 * S, 100 * S, 215 * S)
        );
        let b = bounds(&PACE, Some(119 * S), Some(119 * S));
        assert_eq!(
            (
                b.bytes_timeout,
                b.bytes_bound,
                b.entries_timeout,
                b.entries_bound
            ),
            (238 * S, 357 * S, 238 * S, 491 * S)
        );
    }

    /// Decision 22: past the limit while the agent works, the sandbox is
    /// stopped, inside the agent's window, with what its window grew.
    #[tokio::test(start_paused = true)]
    async fn a_layer_past_its_limit_is_stopped() {
        let fake = Fake::new(|i, _| quick((100 + 50 * i as u64) * MB));
        let (fake, w, state) = start(300, Watched::Layer, fake).await;
        assert_eq!(state, StartState::Watched);
        tokio::time::sleep(20 * S).await;
        let r = w.finish().await;
        assert_eq!(fake.count("stop"), 1);
        let stop = r.stopped.unwrap();
        assert_eq!(stop.usage.mb(Watched::Layer), 350);
        assert!(stop.in_window);
        assert_eq!(stop.growth_at_stop, 250 * MB);
        assert_eq!(r.first.unwrap().mb(Watched::Layer), 100);
    }

    /// U1 (F-7, F-11): bytes measured once (base, 1 s), then hanging. The
    /// sandbox is stopped as unmeasurable within the bound (30 s) and a
    /// second of the last success; no live timeout passes 20 s; nothing
    /// is paused (the trait has no pause; no unpause either).
    #[tokio::test(start_paused = true)]
    async fn bytes_that_hang_are_stopped_within_the_bound() {
        let fake = Fake::new(|i, _| {
            if i == 0 {
                (S, Ok(100 * MB))
            } else {
                (Duration::from_secs(10_000), Ok(0))
            }
        });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(200)).await;
        let r = w.finish().await;
        assert_eq!(r.bounds.bytes_bound, 30 * S);
        assert!(r.unmeasurable.is_some(), "{r:?}");
        let stop = fake.at("stop begins").unwrap();
        assert!(stop <= 32 * S, "stopped at {stop:?}, base ended at 1 s");
        // Live: the report said 139 s against a 90 s bound, counting on
        // after the stop until the watch ended. The gap ends at the stop.
        assert!(r.blind.is_some_and(|b| b <= 31 * S), "{r:?}");
        let live = fake.starts_with("bytes ");
        assert!(live[1..].iter().all(|c| c.1 != "bytes 120s"), "{live:?}");
        assert!(
            live[1..].iter().all(|c| c
                .1
                .trim_start_matches("bytes ")
                .trim_end_matches('s')
                .parse::<u64>()
                .unwrap()
                <= 20),
            "{live:?}"
        );
        assert_eq!(fake.count("unpause"), 0);
    }

    /// U2 (F-11): files planted to slow each measure to 60 s cannot stretch
    /// the bound: stopped within it.
    #[tokio::test(start_paused = true)]
    async fn slow_measures_cannot_stretch_the_bound() {
        let fake = Fake::new(|i, _| {
            if i == 0 {
                (S, Ok(100 * MB))
            } else {
                (60 * S, Ok(100 * MB))
            }
        });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(200)).await;
        let r = w.finish().await;
        assert!(r.unmeasurable.is_some(), "{r:?}");
        assert!(fake.at("stop begins").unwrap() <= 32 * S);
    }

    /// U3 (calibration): a daemon slow before the trial (25 s for the base)
    /// gets 50 s timeouts and a 75 s bound; live measures of 40 s that
    /// succeed never stop an honest trial.
    #[tokio::test(start_paused = true)]
    async fn a_daemon_slow_before_the_trial_gets_longer_bounds() {
        let fake = Fake::new(|i, _| {
            if i == 0 {
                (25 * S, Ok(100 * MB))
            } else {
                (40 * S, Ok(100 * MB))
            }
        });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(600)).await;
        let r = w.finish().await;
        assert_eq!(
            (r.bounds.bytes_timeout, r.bounds.bytes_bound),
            (50 * S, 75 * S)
        );
        assert_eq!((r.stopped, r.unmeasurable), (None, None), "{r:?}");
        assert_eq!(fake.count("stop begins"), 0);
        assert!(fake.count("bytes 50s") >= 5);
    }

    /// U4 (F-8): on a watched layer, a base count of the entries that fails
    /// turns the entries off for the trial: counted once, recorded, and no
    /// stop for it however long the trial runs.
    #[tokio::test(start_paused = true)]
    async fn entries_not_counted_before_the_turn_are_not_watched() {
        let fake = Fake::new(|_, _| quick(100 * MB)).entries(|_, _| (S, Err(())));
        let (fake, w, state) = start(300, Watched::Layer, fake).await;
        assert_eq!(state, StartState::EntriesUnwatched);
        tokio::time::sleep(Duration::from_secs(900)).await;
        let r = w.finish().await;
        assert_eq!(fake.starts_with("entries").len(), 1);
        assert!(r.entries_unwatched);
        assert_eq!((r.stopped, r.unmeasurable), (None, None));
    }

    /// U5: under a cap the entries are the only guard: a failed base count
    /// is tried once more with 180 s; failing twice, nothing watches the
    /// layer, and the agent must not run.
    #[tokio::test(start_paused = true)]
    async fn under_a_cap_the_base_count_gets_a_second_try() {
        let fake = Fake::new(|_, _| panic!("no bytes under a cap"))
            .entries(|i, _| if i == 0 { (S, Err(())) } else { quick(10) });
        let (fake, w, state) = start(300, Watched::Entries, fake).await;
        assert_eq!(state, StartState::Watched);
        assert_eq!(
            fake.names()[..2],
            ["entries 120s".to_string(), "entries 180s".to_string()]
        );
        drop(w);

        let fake = Fake::new(|_, _| panic!("no bytes under a cap")).entries(|_, _| (S, Err(())));
        let (fake, w, state) = start(300, Watched::Entries, fake).await;
        assert!(matches!(state, StartState::Unmeasurable(_)), "{state:?}");
        tokio::time::sleep(Duration::from_secs(600)).await;
        assert_eq!(fake.starts_with("entries").len(), 2, "nothing more");
        let r = w.finish().await;
        assert_eq!(r.stopped, None);

        // On a watched layer, base bytes that fail: the same.
        let fake = Fake::new(|_, _| (S, Err(())));
        let (_, _w, state) = start(300, Watched::Layer, fake).await;
        assert!(matches!(state, StartState::Unmeasurable(_)), "{state:?}");
    }

    /// U6: entries counted once, then failing: an unmeasurable stop within
    /// their bound and a second.
    #[tokio::test(start_paused = true)]
    async fn entries_that_stop_answering_are_stopped_within_their_bound() {
        let fake = Fake::new(|_, _| quick(100 * MB))
            .entries(|i, _| if i == 0 { quick(10) } else { (S, Err(())) });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(600)).await;
        let r = w.finish().await;
        assert_eq!(r.bounds.entries_bound, 135 * S);
        assert!(r.unmeasurable.is_some(), "{r:?}");
        assert!(fake.at("stop begins").unwrap() <= 136 * S);
    }

    /// U7 (F-3): under a cap, the bytes are never measured (a sparse file
    /// cannot stop the trial), and the entries are charged 2 KiB each:
    /// 153,600 of them is 300 MB, not over; one more is.
    #[tokio::test(start_paused = true)]
    async fn under_a_cap_only_the_entries_are_charged() {
        for (n, stops) in [(153_600, false), (153_601, true)] {
            let fake = Fake::new(|_, _| panic!("no bytes under a cap"))
                .entries(move |i, _| quick(if i == 0 { 10 } else { n }));
            let (fake, w, _) = start(300, Watched::Entries, fake).await;
            tokio::time::sleep(Duration::from_secs(60)).await;
            let r = w.finish().await;
            assert_eq!(r.stopped.is_some(), stops, "{n}: {r:?}");
            assert_eq!(fake.count("stop"), stops as usize);
        }
    }

    /// U8: what the agent's windows grew is summed; a gap between them is
    /// not the agent's.
    #[tokio::test(start_paused = true)]
    async fn growth_counts_the_windows_alone() {
        for (gap, growth) in [(400, 300), (500, 200)] {
            let size = Arc::new(AtomicU64::new(100));
            let (_, w, _) = start(500, Watched::Layer, sized(&size)).await;
            size.store(200, Ordering::SeqCst);
            w.close_window().await;
            size.store(gap, Ordering::SeqCst);
            tokio::time::sleep(5 * S).await;
            w.open_window().await;
            size.store(600, Ordering::SeqCst);
            tokio::time::sleep(5 * S).await;
            let r = w.finish().await;
            let stop = r.stopped.unwrap();
            assert!(stop.in_window);
            assert_eq!(stop.growth_at_stop, growth * MB, "gap to {gap}");
        }
    }

    /// U9: a stop after the window closed, none open since, is not inside
    /// a window.
    #[tokio::test(start_paused = true)]
    async fn a_stop_between_windows_is_not_the_agents() {
        let size = Arc::new(AtomicU64::new(100));
        let (_, w, _) = start(500, Watched::Layer, sized(&size)).await;
        size.store(300, Ordering::SeqCst);
        w.close_window().await;
        size.store(600, Ordering::SeqCst);
        tokio::time::sleep(5 * S).await;
        let r = w.finish().await;
        let stop = r.stopped.unwrap();
        assert!(!stop.in_window);
        assert_eq!(stop.growth_at_stop, 200 * MB);
    }

    /// U10: a stop is the window's when the measure that decided it began
    /// in the window and the window was still open when it ended, or was the
    /// closing measure itself. Review F2-A-r4 M1: a live measure that began
    /// before the close and ended after it could see BenchFlow's own writes
    /// (sessions.tar). One measure runs at a time now, so the close waits
    /// for one in flight: its reading is taken before anything BenchFlow
    /// writes after the close.
    #[tokio::test(start_paused = true)]
    async fn a_stop_belongs_to_the_window_its_measure_began_in() {
        // The closing measure finds the layer over.
        let size = Arc::new(AtomicU64::new(100));
        let fake = sized(&size);
        let (_, w, _) = start(500, Watched::Layer, fake).await;
        size.store(600, Ordering::SeqCst);
        w.close_window().await;
        assert!(w.halted());
        let r = w.finish().await;
        assert!(r.stopped.unwrap().in_window);

        // A live measure (20 s timeout) begun in the window, slow, finds
        // the layer over after the close (whose own measure, 30 s timeout,
        // finds it under).
        let late = Arc::new(AtomicBool::new(false));
        let l = late.clone();
        let fake = Fake::new(move |_, t| {
            if l.load(Ordering::SeqCst) && t == 20 * S {
                (5 * S, Ok(600 * MB))
            } else {
                quick(100 * MB)
            }
        });
        let (fake, w, _) = start(500, Watched::Layer, fake).await;
        late.store(true, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(1500)).await;
        w.close_window().await;
        tokio::time::sleep(10 * S).await;
        let r = w.finish().await;
        // The live measure (begun at 1 s, 5 s long) ended before the close
        // measured: its reading is the agent's, and nothing after the close
        // is in it.
        let close = fake.at("bytes 30s").unwrap();
        assert!(
            close >= 6 * S,
            "the close measured at {close:?}, during the live measure"
        );
        assert!(r.stopped.unwrap().in_window, "{r:?}");

        // A measure begun after the close.
        let size = Arc::new(AtomicU64::new(100));
        let (_, w, _) = start(500, Watched::Layer, sized(&size)).await;
        w.close_window().await;
        size.store(600, Ordering::SeqCst);
        tokio::time::sleep(5 * S).await;
        let r = w.finish().await;
        assert!(!r.stopped.unwrap().in_window);
    }

    /// U11 (F-10): a stop that takes longer than the settle is cut, the
    /// sandbox thawed, and the stop recorded as not finished.
    #[tokio::test(start_paused = true)]
    async fn a_stop_cut_short_thaws_the_sandbox() {
        let mut fake = Fake::new(|_, _| quick(600 * MB));
        fake.stop_takes = Duration::from_secs(200);
        let (fake, w, _) = start(500, Watched::Layer, fake).await;
        tokio::time::sleep(S).await;
        let r = w.finish().await;
        assert_eq!(fake.names().last().map(String::as_str), Some("unpause"));
        assert!(r.stop_error.is_some_and(|e| e.contains("did not finish")));
        // Over before the turn: not inside the window.
        assert!(!r.stopped.unwrap().in_window);
    }

    /// U12: once the watch is ending, no stop begins.
    #[tokio::test(start_paused = true)]
    async fn no_stop_begins_once_the_watch_ends() {
        let (fake, w, _) = start(500, Watched::Layer, Fake::new(|_, _| quick(MB))).await;
        let shared = w.shared.clone();
        shared.state().ending = true;
        let over = LayerUsage {
            bytes: Some(600 * MB),
            entries: Some(0),
        };
        shared.halt(Why::Over(over, Some(0))).await;
        shared.halt(Why::Unmeasurable(S)).await;
        assert_eq!(fake.count("stop begins"), 0);
        assert!(!w.halted());
    }

    /// U13: the end of the watch takes no measure.
    #[tokio::test(start_paused = true)]
    async fn finishing_takes_no_measure() {
        let (fake, w, _) = start(500, Watched::Layer, Fake::new(|_, _| quick(MB))).await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        let before = fake.names().len();
        let _ = w.finish().await;
        assert_eq!(fake.names().len(), before);
    }

    /// U14: under a cap the bytes are never measured, and the entries still
    /// are charged and reported.
    #[tokio::test(start_paused = true)]
    async fn under_a_cap_the_entries_are_reported_without_bytes() {
        let n = Arc::new(AtomicU64::new(10));
        let m = n.clone();
        let fake = Fake::new(|_, _| panic!("no bytes under a cap"))
            .entries(move |_, _| quick(m.load(Ordering::SeqCst)));
        let (_, w, _) = start(300, Watched::Entries, fake).await;
        n.store(51_210, Ordering::SeqCst);
        w.close_window().await;
        let r = w.finish().await;
        let last = r.last.unwrap();
        assert_eq!((last.bytes, last.entries), (None, Some(51_210)));
        assert_eq!(r.growth, 51_200 * ENTRY_BYTES_CAPPED);
        assert_eq!(r.growth_mb(), 100);
    }

    /// Round 1's F2A-B M2 stays fixed: a sandbox the daemon no longer has
    /// (or says has stopped) makes every measure fail, and the watch never
    /// stops it; the trial's own check of the sandbox decides.
    #[tokio::test(start_paused = true)]
    async fn a_removed_or_stopped_sandbox_is_never_stopped() {
        for gone in [Err(()), Ok(false)] {
            let fake = Fake::new(|i, _| if i == 0 { quick(MB) } else { (S, Err(())) })
                .status(move || gone);
            let (fake, w, _) = start(300, Watched::Layer, fake).await;
            tokio::time::sleep(Duration::from_secs(300)).await;
            let r = w.finish().await;
            assert_eq!(fake.count("stop begins"), 0);
            assert!(fake.count("status") >= 1);
            assert_eq!((r.stopped, r.unmeasurable), (None, None));
            assert!(r.blind.is_some_and(|b| b >= 30 * S), "{r:?}");
        }
    }

    /// Review F2A-B M1 (round 1): a watch dropped without `finish` (a
    /// stopped trial, a panic) stops measuring and can no longer stop its
    /// sandbox.
    #[tokio::test(start_paused = true)]
    async fn a_dropped_watch_measures_and_stops_nothing() {
        let fake = Fake::new(|i, _| quick((100 + 50 * i as u64) * MB));
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        drop(w);
        let at_drop = fake.n_bytes.load(Ordering::SeqCst);
        tokio::time::sleep(10 * S).await;
        assert!(fake.n_bytes.load(Ordering::SeqCst) <= at_drop + 1);
        assert_eq!(fake.count("stop begins"), 0);
    }

    /// A stop under way when the watch ends is let finish.
    #[tokio::test(start_paused = true)]
    async fn the_end_of_the_watch_waits_for_a_stop_under_way() {
        let mut fake = Fake::new(|_, _| quick(600 * MB));
        fake.stop_takes = 30 * S;
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(S).await;
        assert_eq!(fake.count("stop begins"), 1);
        let r = w.finish().await;
        assert_eq!(fake.count("stop"), 1);
        assert!(r.stopped.is_some() && r.stop_error.is_none(), "{r:?}");
    }

    /// Live on Docker 29, `docker diff` of a layer being written took
    /// seconds; a slow count of the entries does not slow the byte measure
    /// that stops a writer.
    #[tokio::test(start_paused = true)]
    async fn a_slow_count_of_entries_does_not_slow_the_stop() {
        let fake = Fake::new(|i, _| quick((10 + 100 * i as u64) * MB))
            .entries(|i, _| if i == 0 { quick(0) } else { (50 * S, Ok(0)) });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(10 * S).await;
        assert!(fake.at("stop begins").unwrap() <= 5 * S);
        drop(w);
    }

    /// Review F2A-A M2 (round 1): a slow measure slows the next one only up
    /// to the pace's bound.
    #[tokio::test(start_paused = true)]
    async fn a_slow_measure_backs_off_only_to_its_bound() {
        let fake = Fake::new(|_, _| (5 * S, Ok(MB)));
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(150)).await;
        drop(w);
        // Each takes 5 s, then waits 10 s (not 15 s): 10 in 150 s.
        let n = fake.starts_with("bytes ").len() - 1;
        assert!((9..=11).contains(&n), "{n} measures");
    }

    /// An inode flood: the entries alone take a watched layer past the
    /// limit (4 KiB each).
    #[tokio::test(start_paused = true)]
    async fn entries_alone_can_pass_the_limit() {
        let fake = Fake::new(|_, _| quick(MB)).entries(|i, _| quick(80_000 * i as u64));
        let (_, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(60)).await;
        let r = w.finish().await;
        assert!(
            matches!(r.stopped, Some(s) if s.usage.entries == Some(80_000)),
            "{r:?}"
        );
    }

    /// Review F2A-A M3 (round 1): a watched layer is charged its bytes or a
    /// block per entry, whichever is more, not both; a capped one half a
    /// block per entry, never its bytes.
    #[test]
    fn a_layer_is_charged_its_bytes_or_a_block_per_entry() {
        let u = LayerUsage {
            bytes: Some(0),
            entries: Some(1_000_000),
        };
        assert_eq!(u.mb(Watched::Layer), 3907);
        let small = LayerUsage {
            bytes: Some(60_000 * 2048),
            entries: Some(60_000),
        };
        assert_eq!(small.mb(Watched::Layer), 235);
        let sparse = LayerUsage {
            bytes: Some(20 << 30),
            entries: Some(1),
        };
        assert_eq!(sparse.charged(Watched::Entries), 2048);
        assert_eq!(LayerUsage::default().mb(Watched::Layer), 0);
    }

    /// U21 (F-4): fullness reads the inodes too, with a relative threshold;
    /// a filesystem without fixed inodes is judged by its blocks; inodes not
    /// counted never say "not full".
    #[test]
    fn a_disk_is_full_in_blocks_or_in_inodes() {
        let row = |avail_kb: u64, inodes: Inodes| DfRow {
            total_kb: 10_000_000,
            used_kb: 10_000_000 - avail_kb,
            avail_kb,
            inodes,
        };
        let known = |total, free| Inodes::Known {
            total,
            used: total - free,
            free,
        };
        assert_eq!(row(5_000_000, known(1_000_000, 0)).full(), Some(true));
        // 10,000 free of 1,000,000: under 16,384 (a flat 1024 said not full).
        assert_eq!(row(5_000_000, known(1_000_000, 10_000)).full(), Some(true));
        assert_eq!(row(5_000_000, known(1_000_000, 20_000)).full(), Some(false));
        // A small filesystem: 2% of 100,000 is 2,000.
        assert_eq!(row(5_000_000, known(100_000, 1_500)).full(), Some(true));
        assert_eq!(row(5_000_000, Inodes::NotApplicable).full(), Some(false));
        assert_eq!(row(5_000_000, Inodes::Unknown).full(), None);
        assert_eq!(row(10_000, Inodes::Unknown).full(), Some(true));
    }

    #[test]
    fn df_output_is_read_with_its_inodes() {
        let out = "Filesystem 1024-blocks Used Available Capacity Mounted on\noverlay 6291456 1024 6290432 1% /\nBF-I\nFilesystem Inodes IUsed IFree IUse% Mounted on\noverlay 3000000 100 2999900 1% /\n";
        let rows = parse_df(out);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].avail_kb, 6290432);
        assert_eq!(
            rows[0].inodes,
            Inodes::Known {
                total: 3_000_000,
                used: 100,
                free: 2_999_900
            }
        );
        // btrfs: no fixed inodes; a missing inode part: unknown.
        let btrfs = out.replace("3000000 100 2999900", "0 0 0");
        assert_eq!(parse_df(&btrfs)[0].inodes, Inodes::NotApplicable);
        let blocks_only = out.split("BF-I").next().unwrap();
        assert_eq!(parse_df(blocks_only)[0].inodes, Inodes::Unknown);
        assert_eq!(parse_df("df: not found\n"), vec![]);
        assert!(df_script(&["/", "/app"]).contains("/bin/df"));
    }

    /// Review F2-A-r4 M3 / F2-B-r4 M1: a status check that fails (a daemon
    /// stalled for a while) does not end the watch: it asks again, and a
    /// layer that passes the limit afterwards is stopped.
    #[tokio::test(start_paused = true)]
    async fn a_failed_status_check_does_not_end_the_watch() {
        let fake = Fake::new(|i, _| match i {
            0 => quick(100 * MB),
            1..=40 => (S, Err(())),
            _ => quick(600 * MB),
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let fake = fake.status(move || {
            if c.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(())
            } else {
                Ok(true)
            }
        });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(600)).await;
        let r = w.finish().await;
        assert_eq!(fake.count("stop begins"), 1, "{r:?}");
        assert!(r.stopped.is_some() || r.unmeasurable.is_some(), "{r:?}");
    }

    /// Review F2-B-r4 M2: on Docker a `docker diff` holds the container's
    /// lock, so a byte measure waits behind a count. The two kinds never
    /// overlap, and the bytes' clock does not run while they wait: an honest
    /// layer whose count takes 35 s is not stopped.
    #[tokio::test(start_paused = true)]
    async fn a_byte_measure_waiting_behind_a_count_is_not_charged() {
        struct Locked(tokio::sync::Mutex<()>);
        #[async_trait]
        impl Layer for Locked {
            async fn bytes(&self, _t: Duration) -> Result<u64, SandboxError> {
                let _l = self.0.lock().await;
                Ok(100 * MB)
            }
            async fn entries(&self, _t: Duration) -> Result<u64, SandboxError> {
                let _l = self.0.lock().await;
                tokio::time::sleep(35 * S).await;
                Ok(10)
            }
            async fn status(&self) -> Result<bool, SandboxError> {
                Ok(true)
            }
            async fn stop(&self) -> Result<(), SandboxError> {
                panic!("an honest layer was stopped")
            }
            async fn unpause(&self) -> Result<(), SandboxError> {
                Ok(())
            }
        }
        let layer = Arc::new(Locked(tokio::sync::Mutex::new(())));
        let (w, _) = LayerWatch::start(300, PACE, Watched::Layer, layer).await;
        tokio::time::sleep(Duration::from_secs(900)).await;
        let r = w.finish().await;
        assert_eq!((r.stopped, r.unmeasurable), (None, None), "{r:?}");
    }

    /// Review F2-D-r4 M2: the headroom stays relative to the base. A base
    /// count of 100 s gives the live counts 200 s, so counts of 125 s on
    /// the same slow daemon do not stop an honest trial.
    #[tokio::test(start_paused = true)]
    async fn a_slow_base_keeps_its_headroom() {
        let fake = Fake::new(|_, _| quick(100 * MB)).entries(|i, _| {
            if i == 0 {
                (100 * S, Ok(10))
            } else {
                (125 * S, Ok(10))
            }
        });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(1200)).await;
        let r = w.finish().await;
        assert_eq!(fake.count("stop begins"), 0, "{r:?}");
        assert_eq!(
            (r.bounds.entries_timeout, r.bounds.entries_bound),
            (200 * S, 415 * S)
        );
        let b = bounds(&PACE, Some(100 * S), Some(170 * S));
        assert_eq!((b.bytes_timeout, b.bytes_bound), (200 * S, 300 * S));
        assert_eq!((b.entries_timeout, b.entries_bound), (340 * S, 695 * S));
    }

    /// Review F2-A-r5 M1 / F2-B-r5 M2: after one count that timed out, a
    /// second count still runs within the bound: an honest trial whose
    /// count stalled once is not stopped.
    #[tokio::test(start_paused = true)]
    async fn a_count_that_stalls_once_gets_a_second_try() {
        let fake = Fake::new(|_, _| quick(100 * MB))
            .entries(|i, _| if i == 2 { (70 * S, Ok(10)) } else { quick(10) });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(600)).await;
        let r = w.finish().await;
        assert_eq!(fake.count("stop begins"), 0, "{r:?}");
        assert_eq!(r.unmeasurable, None);
    }

    /// Review F2-A-r6 M1: on a daemon whose counts take 30 s, one count
    /// that stalls still leaves the second count half a timeout (30 s),
    /// which is enough for it.
    #[tokio::test(start_paused = true)]
    async fn a_slow_count_that_stalls_once_gets_half_a_timeout_for_its_second_try() {
        let fake = Fake::new(|_, _| quick(100 * MB)).entries(|i, _| {
            if i == 2 {
                (500 * S, Ok(10))
            } else {
                (30 * S, Ok(10))
            }
        });
        let (fake, w, _) = start(300, Watched::Layer, fake).await;
        tokio::time::sleep(Duration::from_secs(900)).await;
        let r = w.finish().await;
        assert_eq!(fake.count("stop begins"), 0, "{r:?}");
    }

    /// Review F2-A-r67 M1 (documented, not changed): after one stalled
    /// count the second gets half a timeout (30 s on a healthy daemon).
    /// Counts of 25 s are graded; counts of 35 s are stopped as unmeasurable,
    /// the sandbox's R110 (a rerun), never the agent's.
    #[tokio::test(start_paused = true)]
    async fn after_a_stall_the_second_count_gets_half_a_timeout() {
        for (takes, stopped) in [(25, false), (35, true)] {
            // A healthy base count (instant): a 60 s timeout, a 135 s bound.
            let fake = Fake::new(|_, _| quick(100 * MB)).entries(move |i, _| match i {
                0 => quick(10),
                2 => (500 * S, Ok(10)),
                _ => (takes * S, Ok(10)),
            });
            let (fake, w, _) = start(300, Watched::Layer, fake).await;
            tokio::time::sleep(Duration::from_secs(900)).await;
            let r = w.finish().await;
            assert_eq!(fake.count("stop begins") == 1, stopped, "{takes} s: {r:?}");
            assert_eq!(r.unmeasurable.is_some(), stopped, "{takes} s");
            assert_eq!(r.stopped, None);
        }
    }
}

#[cfg(test)]
mod hold_tests {
    use super::*;

    struct Sized(Arc<std::sync::atomic::AtomicU64>);
    #[async_trait]
    impl Layer for Sized {
        async fn bytes(&self, _t: Duration) -> Result<u64, SandboxError> {
            Ok(self.0.load(std::sync::atomic::Ordering::SeqCst) << 20)
        }
        async fn entries(&self, _t: Duration) -> Result<u64, SandboxError> {
            Ok(10)
        }
        async fn status(&self) -> Result<bool, SandboxError> {
            Ok(true)
        }
        async fn stop(&self) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn unpause(&self) -> Result<(), SandboxError> {
            Ok(())
        }
    }

    /// Review F2-A-r4 M2: BenchFlow's own archive, written while held and
    /// removed before the release, never stops the sandbox; a layer still
    /// over after the release does.
    #[tokio::test(start_paused = true)]
    async fn benchflows_own_archives_are_held_out_of_the_limit() {
        let size = Arc::new(std::sync::atomic::AtomicU64::new(100));
        let (w, _) =
            LayerWatch::start(300, PACE, Watched::Layer, Arc::new(Sized(size.clone()))).await;
        w.close_window().await;
        w.hold();
        size.store(900, std::sync::atomic::Ordering::SeqCst);
        tokio::time::sleep(Duration::from_secs(20)).await;
        assert!(!w.halted());
        size.store(100, std::sync::atomic::Ordering::SeqCst);
        w.release();
        tokio::time::sleep(Duration::from_secs(20)).await;
        assert!(!w.halted());
        w.hold();
        size.store(900, std::sync::atomic::Ordering::SeqCst);
        w.release();
        tokio::time::sleep(Duration::from_secs(5)).await;
        assert!(w.halted());
    }

    /// A byte measure that reads the size when it starts and answers 3 s
    /// later.
    struct Late(Arc<std::sync::atomic::AtomicU64>);
    #[async_trait]
    impl Layer for Late {
        async fn bytes(&self, _t: Duration) -> Result<u64, SandboxError> {
            let v = self.0.load(std::sync::atomic::Ordering::SeqCst) << 20;
            tokio::time::sleep(Duration::from_secs(3)).await;
            Ok(v)
        }
        async fn entries(&self, _t: Duration) -> Result<u64, SandboxError> {
            Ok(10)
        }
        async fn status(&self) -> Result<bool, SandboxError> {
            Ok(true)
        }
        async fn stop(&self) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn unpause(&self) -> Result<(), SandboxError> {
            Ok(())
        }
    }

    /// Review F2-A-r5 M2 / F2-B-r5 M1: a measure that began while the
    /// archive was in the layer and ends after its removal and the release
    /// stops nothing, and leaves no reading behind that a later measure
    /// could combine with.
    #[tokio::test(start_paused = true)]
    async fn a_measure_straddling_the_release_stops_nothing() {
        let size = Arc::new(std::sync::atomic::AtomicU64::new(100));
        let (w, _) =
            LayerWatch::start(300, PACE, Watched::Layer, Arc::new(Late(size.clone()))).await;
        w.close_window().await;
        w.hold();
        size.store(900, std::sync::atomic::Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(4500)).await;
        // A measure is in flight, begun with the archive in the layer.
        size.store(100, std::sync::atomic::Ordering::SeqCst);
        w.release();
        tokio::time::sleep(Duration::from_secs(60)).await;
        assert!(!w.halted());
        let r = w.finish().await;
        assert_eq!((r.stopped, r.unmeasurable), (None, None), "{r:?}");
    }
}
