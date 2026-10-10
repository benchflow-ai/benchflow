//! Spent subscriptions: recognizing "you've hit your limit" in its forms.
//!
//! A login that has used up its subscription reaches BenchFlow three ways:
//!
//! - claude-agent-acp's typed session failure (when asked for it): a turn
//!   that ends with `_meta.jetbrains.air.sessionFailure`, category `limit`
//!   and no action to offer;
//! - the CLI's own words in an error, `You've hit your weekly limit · resets
//!   Oct 2, 4pm (UTC)` (0.8's `usage_limits.py` parses the window and reset);
//! - codex-acp's `-32603` whose `data` says `codexErrorInfo:
//!   "usageLimitExceeded"`, with Codex's `Try again at Oct 3rd, 2026 5:05 PM.`
//!
//! Reset times are understood in UTC (the sandbox's zone); a message naming
//! another zone gives no reset time, since this build carries no time-zone
//! database.

use std::sync::LazyLock;

use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Utc};
use regex::Regex;
use serde_json::Value;

/// The Claude Agent SDK's `USAGE_LIMIT_ERROR_PREFIXES` (0.3.280): how the CLI
/// starts the message it shows instead of an answer when usage is spent.
const USAGE_LIMIT_PREFIXES: [&str; 13] = [
    "You've hit your",
    "You've reached your",
    "You're out of usage credits",
    "Your org is out of usage · add funds to continue",
    "Your org is out of usage · contact your admin",
    "Your seat type doesn't include usage credits",
    "Your seat type doesn't include usage",
    "Your usage allocation has been disabled by your admin",
    "Your group's usage limit is set to $0",
    "Fable 5 requires usage credits",
    "You're out of extra usage",
    "Your seat type doesn't include extra usage",
    // Claude Code before 1.0.60 wrote "Claude AI usage limit reached|<epoch>".
    "Claude AI usage limit reached",
];

/// Claude Code's name for each window, as BenchFlow names it.
const LIMIT_NAME_WINDOWS: [(&str, &str); 4] = [
    ("session limit", "5-hour"),
    ("weekly limit", "7-day"),
    ("opus limit", "7-day Opus"),
    ("sonnet limit", "7-day Sonnet"),
];

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|e| panic!("BenchFlow bug: bad pattern {pattern}: {e}"))
}

static WRAPPERS: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)^(?:ACP error -?\d+:\s*)?(?:Internal error:\s*)?"));
static HIT_YOUR: LazyLock<Regex> =
    LazyLock::new(|| re(r"You've (?:hit|reached) your (?P<name>[^·.\n]+?)(?:\s*[·.]|$)"));
static RESETS: LazyLock<Regex> = LazyLock::new(|| re(r"\bresets (?P<when>[^·\n]+?)\s*(?:·|$)"));
static TRY_AGAIN_AT: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)try again at (?P<when>.+?(?:[ap]m|\([^)]*\)))\s*\.?\s*$"));
static CLOCK: LazyLock<Regex> = LazyLock::new(|| {
    re(concat!(
        r"(?i)^(?:(?P<month>[A-Za-z]{3})[a-z]* (?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s*",
        r"(?:(?P<year>\d{4}),?\s*)?)?",
        r"(?:at )?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>[ap]m)",
        r"(?:\s*\((?P<zone>[^)]+)\))?$"
    ))
});
static LEGACY: LazyLock<Regex> =
    LazyLock::new(|| re(r"Claude AI usage limit reached\|(?P<epoch>\d{9,11})"));
static TRY_AGAIN_IN: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)try again in (?P<span>[^.\n]+)"));
static SPAN_PART: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)(?P<n>\d+)\s*(?P<unit>day|hour|hr|minute|min|second|sec)s?"));

/// What a usage-limit message says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LimitInfo {
    /// The agent's own words, without `ACP error -32603: Internal error:`.
    pub detail: String,
    /// `5-hour`, `7-day`, `7-day Opus`, `7-day Sonnet`.
    pub window: Option<String>,
    pub resets_at: Option<DateTime<Utc>>,
}

/// The agent's words without the JSON-RPC wrappers; Codex's typographic
/// apostrophe made plain.
fn strip_wrappers(text: &str) -> String {
    let text = text.trim().replace('\u{2019}', "'");
    WRAPPERS.replace(&text, "").trim().to_string()
}

/// Whether `text` (wrapped in an ACP error or not) says usage is spent.
pub(crate) fn is_usage_limit_text(text: &str) -> bool {
    let words = strip_wrappers(text);
    USAGE_LIMIT_PREFIXES.iter().any(|p| words.starts_with(p))
}

/// The window and reset a usage-limit message names, or `None` when it is
/// not one (0.8's `parse_limit_text`).
pub(crate) fn parse_limit_text(text: &str, now: DateTime<Utc>) -> Option<LimitInfo> {
    if !is_usage_limit_text(text) {
        return None;
    }
    let mut detail = strip_wrappers(text);
    let window = HIT_YOUR.captures(&detail).and_then(|c| {
        let name = c["name"].trim().to_lowercase();
        LIMIT_NAME_WINDOWS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, w)| w.to_string())
    });
    let mut resets_at = None;
    if let Some(legacy) = LEGACY.captures(&detail) {
        resets_at = legacy["epoch"]
            .parse::<i64>()
            .ok()
            .and_then(|s| Utc.timestamp_opt(s, 0).single());
        detail = "Claude AI usage limit reached".to_string();
    } else if let Some(when) = RESETS
        .captures(&detail)
        .or_else(|| TRY_AGAIN_AT.captures(&detail))
    {
        resets_at = parse_reset_time(&when["when"], now);
    } else if let Some(again) = TRY_AGAIN_IN.captures(&detail) {
        // The numbers are the agent's words: an absurd span gives no reset
        // time rather than an overflow.
        let seconds = SPAN_PART
            .captures_iter(&again["span"])
            .try_fold(0i64, |total, part| {
                let n: i64 = part["n"].parse().ok()?;
                let unit = match part["unit"].to_lowercase().as_str() {
                    "day" => 86_400,
                    "hour" | "hr" => 3600,
                    "minute" | "min" => 60,
                    _ => 1,
                };
                total.checked_add(n.checked_mul(unit)?)
            });
        resets_at = seconds
            .filter(|s| *s > 0)
            .and_then(Duration::try_seconds)
            .and_then(|span| now.checked_add_signed(span));
    }
    Some(LimitInfo {
        detail,
        window,
        resets_at,
    })
}

/// `4pm (UTC)` or `Oct 2, 4pm (UTC)`: the next such time after `now`. A time
/// with no zone is the sandbox's (UTC); other zones are not resolved.
fn parse_reset_time(when: &str, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let m = CLOCK.captures(when.trim())?;
    if let Some(zone) = m.name("zone")
        && !["UTC", "GMT", "Z"].contains(&zone.as_str().trim().to_uppercase().as_str())
    {
        return None;
    }
    let hour: u32 = m["hour"].parse::<u32>().ok()? % 12
        + if m["ampm"].eq_ignore_ascii_case("pm") {
            12
        } else {
            0
        };
    let minute: u32 = m
        .name("minute")
        .map_or(Some(0), |v| v.as_str().parse().ok())?;
    let at = |date: NaiveDate| Some(Utc.from_utc_datetime(&date.and_hms_opt(hour, minute, 0)?));
    if let Some(month) = m.name("month") {
        const MONTHS: [&str; 12] = [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        ];
        let month = MONTHS
            .iter()
            .position(|n| *n == month.as_str().to_lowercase())? as u32
            + 1;
        let day: u32 = m["day"].parse().ok()?;
        let explicit_year = m.name("year").and_then(|y| y.as_str().parse::<i32>().ok());
        let year = explicit_year.unwrap_or(now.year());
        let mut local = at(NaiveDate::from_ymd_opt(year, month, day)?)?;
        if explicit_year.is_none() && local < now - Duration::days(1) {
            local = at(NaiveDate::from_ymd_opt(year + 1, month, day)?)?;
        }
        Some(local)
    } else {
        let mut local = at(now.date_naive())?;
        if local <= now {
            local += Duration::days(1);
        }
        Some(local)
    }
}

/// The `_meta.jetbrains.air.sessionFailure` record in a `_meta`, if any.
pub(crate) fn session_failure(meta: Option<&Value>) -> Option<&serde_json::Map<String, Value>> {
    meta?
        .get("jetbrains")?
        .get("air")?
        .get("sessionFailure")?
        .as_object()
}

/// A failure record's field as text (Python's `str(x or "")`).
pub(crate) fn field(failure: &serde_json::Map<String, Value>, key: &str) -> String {
    match failure.get(key) {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// The login has no quota left: category `limit` with no action to offer
/// (a rate limit offers `retry`), or the CLI's usage-limit words.
pub(crate) fn is_spent_subscription(failure: &serde_json::Map<String, Value>) -> bool {
    let no_actions = failure
        .get("actions")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty);
    (failure.get("category").and_then(Value::as_str) == Some("limit") && no_actions)
        || is_usage_limit_text(&field(failure, "title"))
}

/// Only `warning` is an advisory the turn went on after; any other severity,
/// or none, ends the turn. A spent login never is an advisory.
pub(crate) fn is_advisory(failure: &serde_json::Map<String, Value>) -> bool {
    let warning = failure
        .get("severity")
        .and_then(Value::as_str)
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("warning"));
    warning && !is_spent_subscription(failure)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 30, 12, 0, 0).unwrap()
    }

    #[test]
    fn claude_limit_text_names_window_and_reset() {
        let info = parse_limit_text(
            "ACP error -32603: Internal error: You've hit your weekly limit · resets Oct 2, 4pm (UTC)",
            now(),
        )
        .unwrap();
        assert_eq!(
            info.detail,
            "You've hit your weekly limit · resets Oct 2, 4pm (UTC)"
        );
        assert_eq!(info.window.as_deref(), Some("7-day"));
        assert_eq!(
            info.resets_at,
            Some(Utc.with_ymd_and_hms(2026, 10, 2, 16, 0, 0).unwrap())
        );

        let info =
            parse_limit_text("You've hit your session limit · resets 4pm (UTC)", now()).unwrap();
        assert_eq!(info.window.as_deref(), Some("5-hour"));
        assert_eq!(
            info.resets_at,
            Some(Utc.with_ymd_and_hms(2026, 9, 30, 16, 0, 0).unwrap())
        );
        // 11am today has passed: tomorrow.
        let info =
            parse_limit_text("You've hit your session limit · resets 11am (UTC)", now()).unwrap();
        assert_eq!(
            info.resets_at,
            Some(Utc.with_ymd_and_hms(2026, 10, 1, 11, 0, 0).unwrap())
        );
        // A date more than a day past is next year's.
        let info = parse_limit_text(
            "You've hit your weekly limit · resets Sep 2, 4pm (UTC)",
            now(),
        )
        .unwrap();
        assert_eq!(
            info.resets_at,
            Some(Utc.with_ymd_and_hms(2027, 9, 2, 16, 0, 0).unwrap())
        );
        // Another zone is not resolved.
        let info = parse_limit_text(
            "You've hit your weekly limit · resets 4pm (America/Los_Angeles)",
            now(),
        )
        .unwrap();
        assert_eq!(info.resets_at, None);
    }

    #[test]
    fn codex_and_legacy_forms() {
        // codex-acp's text, typographic apostrophe and all (0.8's test).
        let info = parse_limit_text(
            "You\u{2019}ve hit your usage limit. Upgrade to Plus to continue using Codex \
             (https://chatgpt.com/explore/plus), or try again at Oct 3rd, 2026 5:05 PM.",
            now(),
        )
        .unwrap();
        assert!(info.detail.starts_with("You've hit your usage limit."));
        assert_eq!(
            info.resets_at,
            Some(Utc.with_ymd_and_hms(2026, 10, 3, 17, 5, 0).unwrap())
        );
        assert_eq!(info.window, None);

        let info = parse_limit_text("Claude AI usage limit reached|1790812800", now()).unwrap();
        assert_eq!(info.detail, "Claude AI usage limit reached");
        assert_eq!(
            info.resets_at,
            Some(Utc.with_ymd_and_hms(2026, 10, 1, 0, 0, 0).unwrap())
        );

        let info = parse_limit_text(
            "You've reached your limit, try again in 2 days 3 hours",
            now(),
        )
        .unwrap();
        assert_eq!(
            info.resets_at,
            Some(now() + Duration::seconds(2 * 86_400 + 3 * 3600))
        );
        assert_eq!(parse_limit_text("Rate limited, slow down", now()), None);
    }

    #[test]
    fn absurd_spans_give_no_reset_time_instead_of_a_crash() {
        // The numbers are the agent's words; each of these overflowed a
        // step of the arithmetic before it was checked.
        for text in [
            "You've hit your limit, try again in 200000000000 days",
            "You've hit your limit, try again in 100000000000 days",
            "You've hit your limit, try again in 9223372036854775807 hours",
            "You've hit your limit, try again in 99999999999999999999999 seconds",
        ] {
            assert_eq!(
                parse_limit_text(text, now()).unwrap().resets_at,
                None,
                "{text}"
            );
        }
    }

    #[test]
    fn failure_records_are_spent_or_advisory() {
        let record = |v: Value| v.as_object().unwrap().clone();
        let spent = record(
            serde_json::json!({"category": "limit", "actions": [], "severity": "warning", "title": "x"}),
        );
        assert!(is_spent_subscription(&spent) && !is_advisory(&spent));
        let retry = record(
            serde_json::json!({"category": "limit", "actions": ["retry"], "severity": "Warning"}),
        );
        assert!(!is_spent_subscription(&retry) && is_advisory(&retry));
        let fatal = record(
            serde_json::json!({"category": "connection", "actions": ["new_session"], "severity": "fatal"}),
        );
        assert!(!is_advisory(&fatal));
        let words = record(
            serde_json::json!({"category": "service", "title": "You've hit your weekly limit"}),
        );
        assert!(is_spent_subscription(&words));
    }
}
