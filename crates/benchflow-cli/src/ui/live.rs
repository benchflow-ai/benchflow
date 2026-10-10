//! The live view of a run.
//!
//! On a terminal: finished trials scroll by as one line each, and a small
//! block at the bottom shows what is running now (its phase, time, and the
//! agent's last tool call) above a progress line. Elsewhere (CI, pipes):
//! one line per finished trial and a heartbeat every minute. With `--json`:
//! one JSON object per event.

use std::collections::BTreeMap;
use std::io::Write;
use std::time::{Duration, Instant};

use benchflow_run::events::{Event, TrialSummary};
use benchflow_run::outcome::{Phase, Status};
use benchflow_run::record::fmt_reward;
use tokio::sync::mpsc;

use super::{
    CYAN, dim, fail_mark, ok_mark, paint, plural, secs, truncate, ui, visible_width, warn_mark,
};

struct Running {
    task: String,
    agent: String,
    /// Which of the task's trials, when it has several.
    number: Option<usize>,
    phase: Phase,
    since: Instant,
    tool_calls: u64,
    last_tool: Option<String>,
    tokens: u64,
}

/// Counts for the progress line and the end-of-run summary.
#[derive(Debug, Default, Clone)]
pub struct Tally {
    pub total: usize,
    pub done: usize,
    pub passed: usize,
    pub failed: usize,
    pub unscored: usize,
    pub errored: usize,
    pub reused: usize,
}

/// How finished trials read in the live view.
#[derive(Debug, Clone, Copy, Default)]
pub struct View {
    /// Show the agent beside the task (several attempts per task).
    pub show_agent: bool,
    /// `bench test`: an empty run, a negative control, or a probe that
    /// scores 0 did what the plan expects, so it gets a check mark.
    pub test_plan: bool,
    /// Several trials per task: number them (`hello #2`).
    pub numbered: bool,
    /// Draw nothing while trials run (no spinner, no heartbeat): the
    /// attempt owns the terminal, as `bench shell`'s does.
    pub still: bool,
}

/// Render events until the channel closes. Returns the tally.
pub async fn render(
    mut rx: mpsc::UnboundedReceiver<Event>,
    label_width: usize,
    view: View,
) -> Tally {
    let show_agent = view.show_agent;
    let u = ui();
    let live = u.interactive && !u.json && u.verbosity > super::Verbosity::Quiet && !view.still;
    let mut running: BTreeMap<String, Running> = BTreeMap::new();
    let mut tally = Tally::default();
    let started = Instant::now();
    let mut drawn = 0usize;
    let mut tick = tokio::time::interval(Duration::from_millis(120));
    let mut last_heartbeat = Instant::now();
    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let mut frame = 0usize;
    let mut stderr = anstream::stderr();
    // Trial numbers per task and agent, in the order trials start.
    let mut started_count: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut numbers: BTreeMap<String, usize> = BTreeMap::new();
    let mut job_started = false;
    loop {
        tokio::select! {
            ev = rx.recv() => {
                let Some(ev) = ev else { break };
                if u.json
                    && let Ok(line) = serde_json::to_string(&ev) {
                        println!("{line}");
                    }
                match ev {
                    Event::JobStarted { dir, total, reused, resumed } => {
                        if resumed && !u.json && u.verbosity > super::Verbosity::Quiet {
                            let _ = writeln!(stderr, "{}", resuming_line(&dir, total, reused));
                        }
                        tally.total = total;
                        tally.reused = reused;
                        tally.done = reused;
                        job_started = true;
                    }
                    Event::TrialStarted { trial, task, agent } => {
                        let number = view.numbered.then(|| {
                            *numbers.entry(trial.clone()).or_insert_with(|| {
                                let n = started_count.entry((task.clone(), agent.clone())).or_default();
                                *n += 1;
                                *n
                            })
                        });
                        running.insert(trial, Running { task, agent, number, phase: Phase::Image, since: Instant::now(), tool_calls: 0, last_tool: None, tokens: 0 });
                    }
                    Event::Phase { trial, phase } => {
                        if let Some(r) = running.get_mut(&trial) { r.phase = phase; }
                    }
                    Event::Activity { trial, tool_calls, last_tool, tokens } => {
                        if let Some(r) = running.get_mut(&trial) {
                            r.tool_calls = tool_calls;
                            if last_tool.is_some() { r.last_tool = last_tool; }
                            r.tokens = tokens;
                        }
                    }
                    Event::Log { trial, line } => {
                        if u.verbosity >= super::Verbosity::Verbose && !u.json {
                            if live { clear(&mut stderr, drawn); drawn = 0; }
                            let _ = writeln!(stderr, "  {} {}", dim(format!("{trial}:")), dim(line));
                        }
                    }
                    Event::Retrying { trial, attempt, reason } => {
                        if !u.json && u.verbosity > super::Verbosity::Quiet {
                            if live { clear(&mut stderr, drawn); drawn = 0; }
                            let _ = writeln!(stderr, "  {} {} {}", warn_mark(), trial, dim(format!("retrying ({attempt}): {reason}")));
                        }
                    }
                    Event::TrialFinished(s) => {
                        running.remove(&s.trial);
                        tally.done += 1;
                        // In a test plan, a check mark is what the plan expects.
                        let flip = expects_failure(&s, view);
                        match (s.status, s.passed) {
                            (Status::Scored, Some(p)) if p != flip => tally.passed += 1,
                            (Status::Scored, None) if flip => tally.passed += 1,
                            (Status::Scored, _) => tally.failed += 1,
                            (Status::Unscored, _) => tally.unscored += 1,
                            _ => tally.errored += 1,
                        }
                        if !u.json && u.verbosity > super::Verbosity::Quiet {
                            if live { clear(&mut stderr, drawn); drawn = 0; }
                            let number = numbers.get(&s.trial).copied();
                            let _ = writeln!(stderr, "{}", finished_line(&s, label_width, view, number));
                        }
                    }
                }
            }
            _ = tick.tick() => {
                frame = (frame + 1) % spinner.len();
                if !live && !view.still && !u.json && u.verbosity > super::Verbosity::Quiet && last_heartbeat.elapsed() > Duration::from_secs(60) && !running.is_empty() {
                    last_heartbeat = Instant::now();
                    let _ = writeln!(stderr, "  {} {}/{} done, {} running, {}", dim("…"), tally.done, tally.total, running.len(), secs(started.elapsed().as_secs_f64()));
                }
            }
        }
        // Nothing to show until the job knows its size.
        if live && job_started {
            clear(&mut stderr, drawn);
            drawn = draw(
                &mut stderr,
                &running,
                &tally,
                started,
                spinner[frame],
                label_width,
                show_agent,
            );
        }
    }
    if live {
        clear(&mut stderr, drawn);
    }
    tally
}

/// Said once when a run picks up a job an earlier run left (the same
/// command run again after Ctrl-C), so the user knows nothing was lost,
/// and how to start over when that is what they meant.
fn resuming_line(dir: &std::path::Path, total: usize, reused: usize) -> String {
    format!(
        "  {} {}: {reused} of {} kept, {} to run {}",
        dim("resuming"),
        super::path(dir),
        super::plural(total, "trial"),
        total - reused,
        dim("· --fresh starts a new job")
    )
}

fn clear(out: &mut impl Write, lines: usize) {
    for _ in 0..lines {
        let _ = write!(out, "\x1b[1A\x1b[2K");
    }
    let _ = write!(out, "\r");
    let _ = out.flush();
}

fn draw(
    out: &mut impl Write,
    running: &BTreeMap<String, Running>,
    tally: &Tally,
    started: Instant,
    spin: &str,
    label_width: usize,
    show_agent: bool,
) -> usize {
    let width = super::width();
    let mut lines = 0;
    let max_rows = 8usize;
    let mut rows: Vec<&Running> = running.values().collect();
    rows.sort_by_key(|r| r.since);
    for r in rows.iter().take(max_rows) {
        let label = row_label(&r.task, &r.agent, show_agent, r.number);
        let phase = match r.phase {
            Phase::Image => "image",
            Phase::Start => "starting",
            Phase::Setup => "setup",
            Phase::Agent => "agent",
            Phase::Handoff => "handoff",
            Phase::Verify => "verifying",
            Phase::Save => "saving",
        };
        let mut detail = String::new();
        if r.phase == Phase::Agent && r.tool_calls > 0 {
            detail = plural(r.tool_calls as usize, "tool call");
            if let Some(t) = &r.last_tool {
                detail.push_str(&format!(" · last: {t}"));
            }
            if r.tokens > 0 {
                detail.push_str(&format!(" · {} tokens", human_count(r.tokens)));
            }
        }
        let head = format!(
            "  {} {:<lw$} {:<9} {:>6}",
            paint(CYAN, spin),
            truncate(&label, label_width),
            phase,
            secs(r.since.elapsed().as_secs_f64()),
            lw = label_width
        );
        let room = width.saturating_sub(visible_width(&head) + 3);
        let line = if detail.is_empty() || room < 10 {
            head
        } else {
            format!("{head}  {}", dim(truncate(&detail, room)))
        };
        let _ = writeln!(out, "{line}");
        lines += 1;
    }
    if running.len() > max_rows {
        let _ = writeln!(
            out,
            "  {}",
            dim(format!("… and {} more running", running.len() - max_rows))
        );
        lines += 1;
    }
    let bar_w = 24usize;
    let filled = (bar_w * tally.done).checked_div(tally.total).unwrap_or(0);
    let bar = format!("{}{}", "█".repeat(filled), dim("░".repeat(bar_w - filled)));
    let mut counts = format!("{}{}", ok_mark(), tally.passed);
    if tally.failed > 0 {
        counts.push_str(&format!(" {}{}", fail_mark(), tally.failed));
    }
    if tally.unscored + tally.errored > 0 {
        counts.push_str(&format!(
            " {}{}",
            warn_mark(),
            tally.unscored + tally.errored
        ));
    }
    if tally.reused > 0 {
        counts.push_str(&dim(format!(" · {} reused", tally.reused)).to_string());
    }
    let _ = writeln!(
        out,
        "  {bar} {}/{}  {counts}  {}",
        tally.done,
        tally.total,
        dim(secs(started.elapsed().as_secs_f64()))
    );
    lines += 1;
    let _ = out.flush();
    lines
}

fn row_label(task: &str, agent: &str, show_agent: bool, number: Option<usize>) -> String {
    let mut label = if show_agent {
        format!("{task} {agent}")
    } else {
        task.to_string()
    };
    if let Some(n) = number {
        label.push_str(&format!(" #{n}"));
    }
    label
}

/// In a test plan, everything but the oracle is meant to fail.
fn expects_failure(s: &TrialSummary, view: View) -> bool {
    view.test_plan
        && (s.agent == "nop" || s.agent.starts_with("control:") || s.agent.starts_with("probe:"))
}

/// One finished trial: mark, task, reward, time, and the reason it failed.
pub fn finished_line(
    s: &TrialSummary,
    label_width: usize,
    view: View,
    number: Option<usize>,
) -> String {
    let label = row_label(&s.task, &s.agent, view.show_agent, number);
    let expect_fail = expects_failure(s, view);
    let (good, bad) = if expect_fail {
        (fail_mark(), ok_mark())
    } else {
        (ok_mark(), fail_mark())
    };
    let (mark, score) = match (s.status, s.passed) {
        (Status::Scored, Some(true)) => (good, s.reward.map(fmt_reward).unwrap_or_default()),
        (Status::Scored, _) => (bad, s.reward.map(fmt_reward).unwrap_or_default()),
        (Status::Skipped, _) => (dim("-"), "–".into()),
        _ => (warn_mark(), "–".into()),
    };
    // Pad before dimming: escapes would count toward the width.
    let mut line = format!(
        "  {mark} {:<lw$} {:>5} {}",
        truncate(&label, label_width),
        score,
        dim(format!("{:>6}", secs(s.seconds))),
        lw = label_width
    );
    // "reward 0" only repeats the score column.
    let reason = s
        .reason
        .as_ref()
        .filter(|r| **r != format!("reward {score}"));
    if let Some(r) = reason {
        let room = super::width().saturating_sub(visible_width(&line) + 2);
        if room > 12 {
            line.push_str("  ");
            line.push_str(&dim(truncate(r, room)));
        }
    } else if s.timed_out {
        line.push_str(&format!("  {}", dim("timed out")));
    }
    line
}

/// `1.2K`, `3.4M`.
pub fn human_count(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}K", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_resumed_job_says_what_it_kept() {
        let line = resuming_line(std::path::Path::new("jobs/2026-10-10__13-37-53"), 2, 0);
        assert!(
            line.contains("jobs/2026-10-10__13-37-53: 0 of 2 trials kept, 2 to run"),
            "{line}"
        );
        assert!(line.contains("--fresh starts a new job"), "{line}");
    }
}
