//! A sandbox the agent's turn exhausts: the task's disk limit (decision 22),
//! recorded per trial and held from the agent's first turn to the end of
//! grading, and who a resource that ran out belongs to (decision 21).
//!
//! Where the backend caps the writable layer and the sandbox shows the cap,
//! the limit is enforced by the backend. Where it cannot, the trial records
//! "disk limit not enforced", and BenchFlow measures the layer from outside
//! the sandbox ([`benchflow_sandbox::LayerWatch`]), stopping the sandbox
//! once the layer passes the limit. Where the cap holds, it holds the
//! layer's blocks but not the inodes its files take (XFS project quotas
//! charge no inodes), so the watch counts the entries there, alone.
//!
//! Each turn of the agent is a window of the watch. At the end of a turn
//! that stops the agent's work, a rootless agent's processes are killed
//! first, then the window closes, so nothing of the agent's writes after
//! its window is measured.
//!
//! A full disk never moves a failure onto the agent (DECISIONS row 13).
//! Evidence that the disk was full, which moves a failure off the task onto
//! the sandbox (R305), comes in two tiers: the sandbox's own `df` counts
//! only for a rootless agent (a root agent could replace `df`); for a root
//! agent only the daemon's view counts: a fresh measure of the layer under a
//! cap (which can only show room), or a hardened probe of the daemon's own
//! disk.

use benchflow_sandbox::disk::{DfRow, StartState, Watched};
use benchflow_sandbox::{Capabilities, DiskLimit, Sandbox, WatchReport};

use std::time::Duration;

use super::Trial;
use crate::agent::Attempt;
use crate::hardening::{self, Report};
use crate::outcome::{DefenseState, Owner, Phase, Problem, defense};

/// The code of a sandbox the disk limit's watch stopped, or could not
/// watch.
pub(crate) const DISK_STOP: &str = "R110";

const MB: u64 = 1024 * 1024;

/// How a turn of the agent ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TurnEnd {
    /// The agent's work stops (a single turn; a graded, last or failed
    /// stage): a rootless agent's processes are killed, and its window
    /// closes.
    Stop,
    /// An ungraded stage that worked: the agent keeps its processes into
    /// the next stage, and its window stays open until BenchFlow prepares
    /// that stage.
    Carry,
}

impl Trial<'_> {
    /// Record how the task's disk limit holds on the agent's new sandbox,
    /// as the `disk_limit` defense. Nothing when the task sets none.
    pub(super) fn record_disk_limit(&mut self, sb: &dyn Sandbox, caps: &Capabilities) {
        let asked = self.task().sandbox.disk_mb;
        let Some((state, note)) =
            disk_limit_defense(&sb.disk_limit(), asked, caps.max_disk_mb, sb.backend())
        else {
            return;
        };
        if state == DefenseState::NotSupported {
            self.rt.events.log(self.name(), note.clone());
        }
        defense(&mut self.defenses, "disk_limit", state, Some(note.clone()));
        self.disk_seen = DiskSeen {
            note,
            ..DiskSeen::default()
        };
    }

    /// Before a turn of the agent: read the sandbox's own memory counters,
    /// and open the agent's window. Taken right before the turn, so
    /// BenchFlow's own setup is in the measure the turn starts from, and
    /// nothing the turn writes is. A watch that cannot measure the layer
    /// before the first turn (nothing would guard the limit) is the
    /// sandbox's R110, before the agent runs.
    pub(super) async fn turn_begin(&mut self) -> Result<(), Problem> {
        self.memory_before = self.memory_events().await;
        self.agent_window_open().await
    }

    async fn agent_window_open(&mut self) -> Result<(), Problem> {
        match &self.layer_watch {
            Some(w) => w.open_window().await,
            None if !self.watch_ended => {
                let started = match self.sb().disk_limit() {
                    // The daemon holds the cap on the blocks: the entries
                    // alone are watched.
                    DiskLimit::Enforced(mb) => self.sb().watch_layer(mb, Watched::Entries).await,
                    DiskLimit::NotEnforced(mb, _) => {
                        self.sb().watch_layer(mb, Watched::Layer).await
                    }
                    DiskLimit::None | DiskLimit::Own(_) => None,
                };
                if let Some((watch, state)) = started {
                    match state {
                        StartState::Unmeasurable(why) => {
                            drop(watch);
                            return Err(unwatched_problem(&why));
                        }
                        StartState::EntriesUnwatched => {
                            self.disk_seen.entries_unwatched = true;
                            self.rt.events.log(self.name(), ENTRIES_UNWATCHED);
                            self.update_disk_note();
                        }
                        StartState::Watched => {}
                    }
                    self.layer_watch = Some(watch);
                }
            }
            None => {}
        }
        Ok(())
    }

    /// Close the agent's window, if one is open: the watch's closing
    /// measure. No kill.
    pub(super) async fn agent_window_close(&mut self) {
        if let Some(w) = &self.layer_watch {
            w.close_window().await;
        }
    }

    /// After a turn of the agent.
    /// - [`TurnEnd::Stop`]: a rootless agent's processes are killed (a
    ///   failed kill is noted, never the trial's error), then its window
    ///   closes, measured with nothing of the agent's left writing.
    /// - A sandbox the watch stopped ends the turn with that, whatever the
    ///   attempt returned (it only saw its sandbox go): the agent's R316
    ///   when the measure that stopped it began in the agent's window and
    ///   the agent's windows grew the layer by at least half the limit,
    ///   else the sandbox's R110.
    /// - An agent that exited (R204) while its sandbox still runs, with its
    ///   own cgroup's memory limit hit and no more processes killed than
    ///   times it was hit: the agent's R316. Killed otherwise: the
    ///   sandbox's R204.
    pub(super) async fn turn_end(
        &mut self,
        attempted: Result<(), Problem>,
        how: TurnEnd,
    ) -> Result<(), Problem> {
        let before = self.memory_before.take();
        if how == TurnEnd::Stop {
            let killed = self.kill_at_turn_end().await;
            self.agent_window_close().await;
            // The kill could not even start: the sandbox at its own process
            // limit is the agent's R316 (c) now, while the sandbox still runs
            // and before a later stop of the watch could hide it (review
            // F2-A-r4 M4).
            if !killed
                && self.exhaustion_is_the_agents()
                && let Some(why) = self.processes_exhausted().await
            {
                let _ = self.end_watch(None).await;
                return Err(exhausted(Phase::Agent, why)
                    .detail(attempted.err().map(|p| p.to_string()).unwrap_or_default()));
            }
        }
        if self.layer_watch.as_ref().is_some_and(|w| w.halted())
            && let Some(p) = self.end_watch(attempted.as_ref().err()).await
        {
            return Err(p);
        }
        let Err(p) = attempted else {
            return attempted;
        };
        if p.code != "R204" || p.owner != Owner::Agent || !self.exhaustion_is_the_agents() {
            return Err(p);
        }
        let (Some(b), Some(a)) = (before, self.memory_events().await) else {
            return Err(p);
        };
        let (oom, kills) = (a.oom.saturating_sub(b.oom), a.kills.saturating_sub(b.kills));
        let Some(who) = memory_outcome(oom, kills) else {
            return Err(p);
        };
        if self.not_running(false).await.is_some() {
            return Err(p);
        }
        let detail = format!("{}: {}", p.code, p.message);
        if who == Owner::Agent {
            return Err(exhausted(
                Phase::Agent,
                format!(
                    "the sandbox's own memory limit ({} MB) ran out {oom} time{} during the agent's turn, and the out-of-memory killer ended {kills} process{} in it, the agent among them",
                    a.max / MB,
                    if oom == 1 { "" } else { "s" },
                    if kills == 1 { "" } else { "es" },
                ),
            )
            .detail(detail));
        }
        let mut own = Problem::new(
            "R204",
            Owner::Sandbox,
            Phase::Agent,
            format!(
                "the out-of-memory killer ended {kills} process{} in the sandbox during the agent's turn, the agent among them, but the sandbox's own memory limit ran out {oom} time{} (the host's or a parent's memory ran out too), so it is not the agent's doing",
                if kills == 1 { "" } else { "es" },
                if oom == 1 { "" } else { "s" },
            ),
            "check the host's memory and lower --concurrency; the trial is run again",
        )
        .detail(detail);
        own.transient = true;
        Err(own)
    }

    /// A rootless agent's processes, killed as root at the end of its turn
    /// (60 s at most), so nothing of the agent's writes after its window
    /// closes. A failed kill is noted; the kill before grading decides.
    /// Whether it worked (or was not needed).
    async fn kill_at_turn_end(&mut self) -> bool {
        if self.user == "root"
            || self.sandbox.is_none()
            || self.layer_watch.as_ref().is_some_and(|w| w.halted())
        {
            return true;
        }
        let killed = match self
            .root(hardening::kill_user(&self.user), Duration::from_secs(60))
            .await
        {
            Ok(out) if out.success() => Report::parse(&out.stdout)
                .first("killed")
                .and_then(|n| n.parse::<u64>().ok())
                .ok_or_else(|| "the kill did not report back".to_string()),
            Ok(out) => Err(out.tail(300)),
            Err(e) => Err(e.to_string()),
        };
        match killed {
            Ok(n) => {
                self.killed_before_save += n;
                true
            }
            Err(e) => {
                let why = format!("the kill at the end of the agent's turn failed: {e}");
                self.rt.events.log(self.name(), why.clone());
                self.turn_kill_failed = Some(why);
                false
            }
        }
    }

    /// Hold the disk watch while BenchFlow writes one of its own archives
    /// into the layer (review F2-A-r4 M2).
    pub(super) fn hold_watch(&self) {
        if let Some(w) = &self.layer_watch {
            w.hold();
        }
    }

    /// Remove `archive` from the sandbox (it was copied out, or never
    /// finished), then release the watch: BenchFlow's own archives never
    /// count against the task's disk limit.
    pub(super) async fn release_watch(&self, archive: &str) {
        if self.sandbox.is_some() {
            let _ = self
                .root(
                    format!("rm -f {}", benchflow_sandbox::shell::quote(archive)),
                    Duration::from_secs(60),
                )
                .await;
        }
        if let Some(w) = &self.layer_watch {
            w.release();
        }
    }

    /// End the watch (at the end of the trial, or once it stopped the
    /// sandbox): close a window still open (no kill), stop the watch,
    /// record what it could not see, and decide a stop it made. `attempted`
    /// is the agent's own outcome, for the detail.
    pub(super) async fn end_watch(&mut self, attempted: Option<&Problem>) -> Option<Problem> {
        self.agent_window_close().await;
        let watch = self.layer_watch.take()?;
        self.watch_ended = true;
        let report = watch.finish().await;
        self.disk_seen.add(&report);
        self.update_disk_note();
        let halted = report.stopped.is_some() || report.unmeasurable.is_some();
        halted.then(|| self.disk_stopped(&report, attempted))
    }

    /// The `disk_limit` note: the first, and what the watch could not see.
    fn update_disk_note(&mut self) {
        if let Some(d) = self.defenses.get("disk_limit").cloned()
            && !self.disk_seen.note.is_empty()
        {
            defense(
                &mut self.defenses,
                "disk_limit",
                d.state,
                Some(self.disk_seen.to_note()),
            );
        }
    }

    /// The sandbox's own memory counters, where its cgroup is its own and
    /// it has a memory limit of its own.
    async fn memory_events(&self) -> Option<Memory> {
        if !self.sb().own_cgroup() {
            return None;
        }
        let out = self
            .root(hardening::memory_events(), Duration::from_secs(30))
            .await
            .ok()?;
        let r = Report::parse(&out.stdout);
        let n = |k: &str| r.first(k).and_then(|v| v.trim().parse::<u64>().ok());
        Some(Memory {
            max: n("mem-max")?,
            oom: n("mem-oom")?,
            kills: n("mem-oom_kill")?,
        })
    }

    /// Whether an exhausted sandbox can be this attempt's agent's doing:
    /// a model agent or a script. The reference solution and the controls
    /// are the task's; a regrade and BenchFlow's own probes ran no agent of
    /// the user's.
    fn exhaustion_is_the_agents(&self) -> bool {
        matches!(self.spec.attempt, Attempt::Model(_) | Attempt::Script(_))
    }

    /// The outcome of a sandbox the disk limit's watch stopped.
    fn disk_stopped(&mut self, r: &WatchReport, attempted: Option<&Problem>) -> Problem {
        let note = format!(
            "{}; the sandbox was stopped",
            self.defenses
                .get("disk_limit")
                .and_then(|d| d.note.clone())
                .unwrap_or_default()
        );
        let state = self
            .defenses
            .get("disk_limit")
            .map(|d| d.state)
            .unwrap_or(DefenseState::NotSupported);
        defense(&mut self.defenses, "disk_limit", state, Some(note));
        let detail = match attempted {
            Some(a) => format!("the agent's own outcome when its sandbox stopped: {a}"),
            None => String::new(),
        };
        if self.exhaustion_is_the_agents()
            && let Some(why) = watch_stop_is_the_agents(r)
        {
            return exhausted(Phase::Agent, why).detail(detail);
        }
        disk_stop_problem(r, self.phase).detail(detail)
    }

    /// Whether this trial's agent could not act as root: a user of its own
    /// whose separation from root the trial enforced.
    pub(super) fn rootless_agent(&self) -> bool {
        self.user != "root"
            && self
                .defenses
                .get("agent_user")
                .is_some_and(|d| d.state == DefenseState::Enforced)
    }

    /// A failure after the agent's turn (the handoff, the hooks, the
    /// verifier, a later stage's setup) that the agent's exhausting its own
    /// process limit explains, on evidence the agent cannot fake: R316. One
    /// that failed on a full disk with nothing showing the disk had room is
    /// the sandbox's (R305, rerun once, then kept), under a cap too: no
    /// reading taken in the sandbox at one time and of the daemon at another
    /// shows the cap, not the host, was full (round 6; DECISIONS row 13).
    /// Anything else is returned as it is.
    pub(super) async fn exhausted_instead(&self, p: Problem) -> Problem {
        let after_turn = matches!(p.phase, Phase::Handoff | Phase::Verify | Phase::Save)
            || (p.phase == Phase::Setup && self.stage_worked);
        if !after_turn
            || self.sandbox.is_none()
            || matches!(p.owner, Owner::You | Owner::Agent | Owner::Model)
            || p.loss_confirmed == Some(true)
            || matches!(p.code.as_str(), "R000" | "R001" | "R103" | "R107")
        {
            return p;
        }
        let full_disk_words = p.message.contains("disk was full")
            || p.detail
                .as_deref()
                .is_some_and(|d| d.contains("No space left on device"));
        if self.exhaustion_is_the_agents()
            && p.phase == Phase::Handoff
            && matches!(p.code.as_str(), "R305" | "R307")
            && let Some(why) = self.processes_exhausted().await
        {
            return exhausted(Phase::Handoff, why).detail(p.to_string());
        }
        if matches!(p.code.as_str(), "R301" | "R304" | "R104")
            && full_disk_words
            && self.full_evidence().await != Some(false)
        {
            let mut own = Problem::new(
                "R305",
                Owner::Sandbox,
                p.phase,
                format!(
                    "{} on a full disk, and nothing shows the agent filled it (the host's disk, or files from before the agent's turn), so it is the sandbox's",
                    p.message
                ),
                "check the disk the sandbox writes to; the trial is run again once",
            );
            own.detail = p.detail.clone();
            own.loss_confirmed = Some(false);
            return own.transient();
        }
        p
    }

    /// A scored run that did not pass, whose verifier printed "No space
    /// left on device" (review F2-C, F-6): the sandbox's R305 when the disk
    /// is shown full by evidence the agent cannot fake; otherwise `None`,
    /// and the score stands. A score that passed always stands. A full disk
    /// is never the agent's R316 here (round 6: see `exhausted_instead`).
    ///
    /// `agent_was_here`: the verifier ran in the agent's sandbox. A separate
    /// verifier's sandbox, or a regrade's, never held the agent (review
    /// F2-D-r4 M1): its own `df` is read too.
    pub(super) async fn full_disk_instead(
        &self,
        out: &str,
        agent_was_here: bool,
    ) -> Option<Problem> {
        if !out.contains("No space left on device") {
            return None;
        }
        let full = if agent_was_here {
            self.full_evidence().await
        } else {
            let h2 = self.sb().host_df().await.and_then(|r| r.full());
            either(self.df_full_now().await, h2)
        };
        if full != Some(true) {
            return None;
        }
        let mut own = Problem::new(
            "R305",
            Owner::Sandbox,
            Phase::Verify,
            "the verifier failed on a full disk (it printed \"No space left on device\"), and nothing shows the agent filled it, so the score is not kept and it is the sandbox's",
            "check the disk the sandbox writes to; the trial is run again once",
        );
        own.loss_confirmed = Some(false);
        Some(own.transient())
    }

    /// Why the sandbox's own process limit is spent, as its backend sees
    /// it from outside: the count at the sandbox's own limit in one of
    /// three looks a second apart. `None` when it is below (a shortage of
    /// the host's) or the backend cannot say.
    async fn processes_exhausted(&self) -> Option<String> {
        for i in 0..3 {
            if i > 0 {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            if let Ok((now, limit)) = self.sb().processes().await
                && now >= limit
            {
                return Some(format!(
                    "the sandbox held {now} processes, its own limit of {limit}, when BenchFlow went to stop the agent's processes, so the stop could not run"
                ));
            }
        }
        None
    }

    /// Whether the disk the sandbox writes to is full, on the evidence the
    /// tiers allow (review F2-C, F-4 to F-6):
    /// - S, the sandbox's own `df` of `/` and the workdir, counts only for
    ///   a rootless agent;
    /// - H1, under a cap, a fresh measure of the layer from the daemon: at
    ///   the cap (less 64 MB), or below it, which proves the quota not full
    ///   (a file's apparent size is never less than what it takes);
    /// - H2, the hardened probe of the daemon's own disk.
    ///
    /// Under a cap, the quota (S or H1) and the daemon's disk (H2) are
    /// apart: full when either is, not full only when both say so.
    /// Elsewhere S and H2 read the same disk. `None` when nothing tells.
    async fn full_evidence(&self) -> Option<bool> {
        let s = if self.rootless_agent() {
            self.df_full_now().await
        } else {
            None
        };
        let h2 = self.sb().host_df().await.and_then(|r| r.full());
        match self.sb().disk_limit() {
            DiskLimit::Enforced(cap) => {
                // The layer's apparent size can only show the quota is not
                // full: a sparse file inflates it at no block cost (review
                // F2-C-r4 M1). A rootless agent's own `df` decides first.
                let h1 = match s {
                    Some(_) => None,
                    None => self
                        .sb()
                        .layer_bytes()
                        .await
                        .and_then(|b| (b / MB + FULL_MARGIN_MB < cap).then_some(false)),
                };
                tiers(s.or(h1), h2)
            }
            _ => either(s, h2),
        }
    }

    /// Whether `/` or the workdir is full now, as root's `df` in the
    /// sandbox says: full if either is, not full if both are not.
    async fn df_full_now(&self) -> Option<bool> {
        let rows = self.df_reading(&["/", &self.workdir]).await;
        if rows.is_empty() {
            return None;
        }
        let full: Vec<Option<bool>> = rows.iter().map(DfRow::full).collect();
        if full.contains(&Some(true)) {
            Some(true)
        } else if full.iter().all(|f| *f == Some(false)) {
            Some(false)
        } else {
            None
        }
    }

    /// Root's `df` (blocks and inodes) of `paths` in the sandbox, `/bin/df`
    /// by full path, with BenchFlow's plumbing environment.
    async fn df_reading(&self, paths: &[&str]) -> Vec<DfRow> {
        let script = format!("{} 2>/dev/null", benchflow_sandbox::disk::df_script(paths));
        match self.root(script, Duration::from_secs(30)).await {
            Ok(out) => benchflow_sandbox::disk::parse_df(&out.stdout),
            Err(_) => Vec::new(),
        }
    }
}

/// The R110 of a watch that could not measure the layer before the agent's
/// first turn: nothing would guard the limit, so the agent does not run.
/// Filed under setup, so records and `bench results` see an agent that never
/// started (review F2-integrate-A A-1: F1's metadata warning counted it).
fn unwatched_problem(why: &str) -> Problem {
    let mut p = Problem::new(
        DISK_STOP,
        Owner::Sandbox,
        Phase::Setup,
        format!("{why}, so nothing would hold the task's disk limit, and the agent did not run"),
        "check the daemon (`bench doctor`); the trial is run again once",
    );
    p.loss_confirmed = Some(false);
    p.transient()
}

/// Why a stop the watch made is the agent's (R316 (a)): it stopped on a
/// measure over the limit, the stop itself worked, the measure began in
/// one of the agent's windows, and the windows grew the layer by at least
/// half the limit. `None` otherwise.
fn watch_stop_is_the_agents(r: &WatchReport) -> Option<String> {
    let s = r.stopped?;
    let limit = r.limit_mb.saturating_mul(MB);
    if r.unmeasurable.is_some()
        || r.stop_error.is_some()
        || !s.in_window
        || s.growth_at_stop.saturating_mul(2) < limit
    {
        return None;
    }
    Some(format!(
        "{}, and the agent's own turns added {} MB of it, so BenchFlow stopped the sandbox",
        layer_words(r),
        s.growth_at_stop / MB
    ))
}

/// The problem for a stop the watch made that is not shown to be the
/// agent's: the sandbox's (R110), counted as an unexplained failure of the
/// sandbox: rerun once, then kept as it is (`Trial::lost_again`).
fn disk_stop_problem(r: &WatchReport, phase: Phase) -> Problem {
    let stop_failed = r
        .stop_error
        .as_ref()
        .map(|e| format!(" (the stop itself failed: {e})"))
        .unwrap_or_default();
    let whose = match r.stopped {
        // A layer that could not be measured is never the agent's: no
        // measure says who wrote it.
        None => String::new(),
        Some(s) if !s.in_window => {
            "; the measure was taken outside the agent's turns (BenchFlow's own setup or grading, or files from before)".to_string()
        }
        Some(s) => format!(
            "; the agent's own turns added {} MB of it, under half the limit",
            s.growth_at_stop / MB
        ),
    };
    let mut p = Problem::new(
        DISK_STOP,
        Owner::Sandbox,
        phase,
        format!(
            "{}{whose}, so BenchFlow stopped the sandbox{stop_failed}, and the trial was not graded",
            layer_words(r)
        ),
        "if the task's work needs more disk, raise its `[sandbox] disk`; otherwise look at what the agent wrote (the trial's outputs, when saved)",
    );
    // Rerun once (within the run, when it retries); a repeat is capped.
    p.loss_confirmed = Some(false);
    p.transient()
}

/// Whose a memory kill is (R316 (d), review F2-C, F-2): `None` when the
/// out-of-memory killer ended nothing; the agent's when the sandbox's own
/// limit ran out at least as many times as processes were killed; the
/// sandbox's otherwise (the host's or a parent's memory killed more).
pub(super) fn memory_outcome(oom: u64, kills: u64) -> Option<Owner> {
    if kills == 0 {
        return None;
    }
    Some(if oom > 0 && kills <= oom {
        Owner::Agent
    } else {
        Owner::Sandbox
    })
}

/// True wins, then false, else unknown.
fn either(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), _) | (_, Some(false)) => Some(false),
        _ => None,
    }
}

/// Two disks apart (the quota, the daemon's): full when either is, not
/// full only when both say so.
fn tiers(own: Option<bool>, host: Option<bool>) -> Option<bool> {
    match (own, host) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// Free space under this, or a layer within this of its cap, is a full
/// disk.
const FULL_MARGIN_MB: u64 = benchflow_sandbox::disk::FULL_MARGIN_MB;

/// What the disk limit's watch could not see, over all the agent's turns,
/// for the `disk_limit` note.
#[derive(Debug, Clone, Default)]
pub(super) struct DiskSeen {
    /// The note as first recorded.
    note: String,
    /// The bounds the trial's watch had, in words.
    bounds: String,
    blind: Duration,
    entries_blind: Duration,
    entries_unwatched: bool,
}

/// The line logged (and noted) when the entries are not watched.
const ENTRIES_UNWATCHED: &str = "the layer's entries could not be counted before the agent's turn, so only its bytes were watched (an inode flood is bounded only by them)";

impl DiskSeen {
    fn add(&mut self, r: &WatchReport) {
        let b = r.bounds;
        let secs =
            |d: Option<Duration>| d.map_or("-".to_string(), |d| format!("{:.1}", d.as_secs_f64()));
        self.bounds = match r.watched {
            Watched::Layer => format!(
                "the measure before the agent's turn took {} s for the bytes and {} s for the entries, so the bytes could go unmeasured for {} s and the entries for {} s",
                secs(b.base_bytes),
                secs(b.base_entries),
                b.bytes_bound.as_secs(),
                b.entries_bound.as_secs()
            ),
            Watched::Entries => format!(
                "the count before the agent's turn took {} s, so the entries could go uncounted for {} s",
                secs(b.base_entries),
                b.entries_bound.as_secs()
            ),
        };
        self.blind = self.blind.max(r.blind.unwrap_or_default());
        self.entries_blind = self.entries_blind.max(r.entries_blind.unwrap_or_default());
        self.entries_unwatched |= r.entries_unwatched;
    }

    /// The first note, and what the watch could not see, once.
    fn to_note(&self) -> String {
        let mut note = self.note.clone();
        if !self.bounds.is_empty() {
            note += &format!("; {}", self.bounds);
        }
        if self.entries_unwatched {
            note += &format!("; {ENTRIES_UNWATCHED}");
        }
        if !self.blind.is_zero() {
            note += &format!(
                "; the layer's bytes went unmeasured for up to {} s at a time",
                self.blind.as_secs()
            );
        }
        if !self.entries_blind.is_zero() {
            note += &format!(
                "; its entries went uncounted for up to {} s at a time",
                self.entries_blind.as_secs()
            );
        }
        note
    }
}

/// The sandbox's own memory counters ([`hardening::memory_events`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Memory {
    /// Its own limit, in bytes.
    max: u64,
    /// How many times that limit ran out.
    oom: u64,
    /// How many processes in it the out-of-memory killer ended.
    kills: u64,
}

/// R316: the agent exhausted its own sandbox, on the sandbox's own
/// evidence (`why`). Scored 0, kept on resume.
pub(super) fn exhausted(phase: Phase, why: String) -> Problem {
    Problem::new(
        EXHAUSTED,
        Owner::Agent,
        phase,
        format!("the agent exhausted its own sandbox: {why}; the trial scores 0"),
        "look at what the agent did to its sandbox; if its work needs more, raise the task's `[sandbox] disk` or `memory`",
    )
}

/// The code of an agent that exhausted its own sandbox.
pub(crate) const EXHAUSTED: &str = "R316";

/// What the watch saw, in words.
fn layer_words(r: &WatchReport) -> String {
    let what = match (&r.stopped, r.unmeasurable) {
        (Some(s), _) => {
            let m = s.usage;
            let files = match (r.watched, m.bytes) {
                (Watched::Layer, Some(b)) => format!("{} MB of files, ", b / MB),
                _ => String::new(),
            };
            format!(
                "the sandbox's writable layer reached {} MB ({files}{} entries added), past the task's disk limit of {} MB",
                m.mb(r.watched),
                m.entries
                    .map_or_else(|| "uncounted".to_string(), |n| n.to_string()),
                r.limit_mb
            )
        }
        (None, Some(blind)) => format!(
            "the sandbox's writable layer could not be measured for {} s while the daemon said the sandbox still ran, against the task's disk limit of {} MB",
            blind.as_secs(),
            r.limit_mb
        ),
        (None, None) => "the sandbox's writable layer passed the task's disk limit".into(),
    };
    match r.first {
        Some(b) => format!(
            "{what}; it measured {} MB when the agent's first turn began",
            b.mb(r.watched)
        ),
        None => what,
    }
}

/// The `disk_limit` defense for a sandbox whose task asks for `asked` MB
/// of disk: how `limit` holds on it, or for a backend that sizes each
/// sandbox's disk itself (`max_disk_mb`, Daytona), what it gives. `None`
/// when the task asks for no limit.
fn disk_limit_defense(
    limit: &DiskLimit,
    asked: Option<u64>,
    max_disk_mb: Option<u64>,
    backend: &str,
) -> Option<(DefenseState, String)> {
    let asked = asked.filter(|d| *d > 0)?;
    Some(match limit {
        DiskLimit::Enforced(mb) => (
            DefenseState::Enforced,
            format!(
                "{mb} MB, capped by the {backend} daemon (the sandbox's filesystem shows the cap, smaller than the daemon's own disk); BenchFlow counts the files, folders and links the layer adds, 2 KiB each, and stops the sandbox past the cap"
            ),
        ),
        DiskLimit::NotEnforced(mb, why) => (
            DefenseState::NotSupported,
            format!(
                "disk limit not enforced: {why}; BenchFlow measures the writable layer from the agent's first turn to the end of grading, its bytes or a 4 KiB block per added file, whichever is more, and stops the sandbox once it passes {mb} MB"
            ),
        ),
        DiskLimit::Own(mb) => (
            DefenseState::Enforced,
            format!("the {backend} backend gives the sandbox a disk of its own of {mb} MB"),
        ),
        DiskLimit::None => match max_disk_mb {
            // Daytona sizes each sandbox's disk itself, in whole GB, up to
            // its ceiling, and the image takes part of it.
            Some(max) => {
                let got = asked.div_ceil(1024).saturating_mul(1024).min(max);
                let clamp = if got < asked {
                    format!("; the task asks for {asked} MB, and {backend} gives at most {max} MB")
                } else {
                    String::new()
                };
                (
                    DefenseState::Enforced,
                    format!(
                        "the {backend} backend gives the sandbox a disk of its own of {got} MB, which holds the image too{clamp}"
                    ),
                )
            }
            None => (
                DefenseState::NotSupported,
                format!(
                    "disk limit not enforced: the {backend} backend can neither cap nor measure a sandbox's disk; the task asks for {asked} MB"
                ),
            ),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use benchflow_sandbox::LayerUsage;
    use benchflow_sandbox::disk::{Bounds, Stop, bounds};

    fn report(stopped: Option<Stop>) -> WatchReport {
        WatchReport {
            limit_mb: 500,
            watched: Watched::Layer,
            bounds: bounds(&benchflow_sandbox::disk::PACE, None, None),
            first: None,
            last: None,
            growth: 0,
            stopped,
            unmeasurable: None,
            stop_error: None,
            blind: None,
            entries_blind: None,
            entries_unwatched: false,
        }
    }

    fn stop(in_window: bool, growth_mb: u64) -> Option<Stop> {
        Some(Stop {
            usage: LayerUsage {
                bytes: Some(600 * MB),
                entries: Some(0),
            },
            in_window,
            growth_at_stop: growth_mb * MB,
        })
    }

    /// R316 (a): a stop is the agent's only when its measure began in the
    /// agent's window, the windows grew at least half the limit, and the
    /// stop worked.
    #[test]
    fn a_watch_stop_is_the_agents_only_inside_its_windows() {
        assert!(watch_stop_is_the_agents(&report(stop(true, 250))).is_some());
        assert!(watch_stop_is_the_agents(&report(stop(true, 249))).is_none());
        assert!(watch_stop_is_the_agents(&report(stop(false, 500))).is_none());
        let mut failed = report(stop(true, 500));
        failed.stop_error = Some("kill refused".into());
        assert!(watch_stop_is_the_agents(&failed).is_none());
        assert!(watch_stop_is_the_agents(&report(None)).is_none());
        let p = disk_stop_problem(&report(stop(false, 500)), Phase::Verify);
        assert_eq!((p.code.as_str(), p.owner), (DISK_STOP, Owner::Sandbox));
        assert!(p.message.contains("outside the agent's turns"), "{p}");
    }

    /// U20 (F-2): a host kill on top of an own-limit OOM is the sandbox's.
    #[test]
    fn a_memory_kill_is_the_agents_only_when_its_own_limit_explains_it() {
        assert_eq!(memory_outcome(1, 2), Some(Owner::Sandbox));
        assert_eq!(memory_outcome(1, 1), Some(Owner::Agent));
        assert_eq!(memory_outcome(0, 1), Some(Owner::Sandbox));
        assert_eq!(memory_outcome(2, 1), Some(Owner::Agent));
        assert_eq!(memory_outcome(3, 0), None);
    }

    /// U23: the tiers. A root agent's own `df` never clears a disk; under a
    /// cap the quota and the daemon's disk are apart.
    #[test]
    fn the_evidence_tiers_combine() {
        // A root agent: S is not read (None); H2 alone.
        assert_eq!(either(None, Some(false)), Some(false));
        // A rootless agent with S full.
        assert_eq!(either(Some(true), Some(false)), Some(true));
        // Under a cap, root agent: S unread, H1 at the cap: full.
        assert_eq!(tiers(either(None, Some(true)), Some(false)), Some(true));
        // H1 below the cap and H2 not full: not full.
        assert_eq!(tiers(either(None, Some(false)), Some(false)), Some(false));
        // H1 unknown: unknown, never "not full".
        assert_eq!(tiers(either(None, None), Some(false)), None);
    }

    /// Review F2A-A L4 (round 2): what the watch could not see is said
    /// once for all the agent's turns.
    #[test]
    fn the_unmeasured_note_is_said_once_over_the_turns() {
        let turn = |blind: u64| WatchReport {
            blind: Some(Duration::from_secs(blind)),
            bounds: Bounds {
                base_bytes: Some(Duration::from_millis(500)),
                base_entries: Some(Duration::from_secs(2)),
                ..bounds(&benchflow_sandbox::disk::PACE, None, None)
            },
            ..report(None)
        };
        let mut seen = DiskSeen {
            note: "disk limit not enforced: x".into(),
            ..DiskSeen::default()
        };
        seen.add(&turn(40));
        seen.add(&turn(35));
        assert_eq!(
            seen.to_note(),
            "disk limit not enforced: x; the measure before the agent's turn took 0.5 s for the bytes and 2.0 s for the entries, so the bytes could go unmeasured for 30 s and the entries for 135 s; the layer's bytes went unmeasured for up to 40 s at a time"
        );
    }

    /// Decision 22: a task's disk limit is never silently dropped. The
    /// record says how it held: capped, not enforced (and watched), or the
    /// backend's own disk with its clamp.
    #[test]
    fn the_disk_limit_record_says_how_the_limit_held() {
        assert_eq!(
            disk_limit_defense(&DiskLimit::None, None, None, "docker"),
            None
        );
        assert_eq!(
            disk_limit_defense(&DiskLimit::None, Some(0), None, "docker"),
            None
        );
        let (state, note) =
            disk_limit_defense(&DiskLimit::Enforced(300), Some(300), None, "docker").unwrap();
        assert_eq!(state, DefenseState::Enforced);
        assert!(
            note.starts_with("300 MB, capped by the docker daemon"),
            "{note}"
        );
        let (state, note) = disk_limit_defense(
            &DiskLimit::NotEnforced(300, "the docker daemon refused a size cap".into()),
            Some(300),
            None,
            "docker",
        )
        .unwrap();
        assert_eq!(state, DefenseState::NotSupported);
        assert!(
            note.starts_with(
                "disk limit not enforced: the docker daemon refused a size cap; BenchFlow measures"
            ) && note.contains("once it passes 300 MB"),
            "{note}"
        );
        let (state, note) =
            disk_limit_defense(&DiskLimit::None, Some(20_480), Some(10_240), "daytona").unwrap();
        assert_eq!(state, DefenseState::Enforced);
        assert!(
            note.contains("of its own of 10240 MB") && note.contains("gives at most 10240 MB"),
            "{note}"
        );
        let (_, note) =
            disk_limit_defense(&DiskLimit::None, Some(3000), Some(10_240), "daytona").unwrap();
        assert!(
            note.contains("of its own of 3072 MB, which holds the image too"),
            "{note}"
        );
    }
}
