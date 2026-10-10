//! How 0.8 names failures (`benchflow._utils.scoring`): the category of an
//! agent or verifier error read from its text when `result.json` stores
//! none, and whether a declared outcome assessment withholds the score.

use crate::py;
use crate::pyjson::{Map, Value, get};

/// Markers of a provider rejecting the login (matched in lowercase text).
const PROVIDER_AUTH: &[&str] = &[
    "permission_denied",
    "leaked",
    "failed to authenticate",
    "invalid bearer token",
    "invalid api key",
    "was rejected as invalid",
    "unauthorized",
    "provider auth failed",
    "http 401",
    "http 403",
];
const PROVIDER_RATE_LIMIT: &[&str] = &[
    "provider rate limited",
    "rate limit",
    "too many requests",
    "http 429",
];
const PROVIDER_UNAVAILABLE: &[&str] = &["provider unavailable", "http 503"];
const PROVIDER_REJECTED: &[&str] = &["provider rejected request", "http 400"];

/// Transport failures that a retry may fix.
const INFRA: &[&str] = &[
    "connection lost",
    "connection reset",
    "connection refused",
    "broken pipe",
    "transient sandbox transport failure",
    "failed to get session command",
    "failed to execute session command",
    "command timed out after",
    "sandbox not found",
    "workspace not found",
    "api connection",
    "api timeout",
    "temporarily unavailable",
];

const VERIFIER_INFRA: &[&str] = &[
    "failed to add tests directory",
    "failed to download verifier directory",
    "failed to download llm-judge input",
    "verifier setup failed",
];

const VERIFIER_DEP_INSTALL: &[&str] = &[
    "dependency install failed",
    "failed to download `",
    "failed to fetch:",
    "error sending request for url",
    "failed to lookup address information",
    "no solution found",
    "could not find a version",
    "resolution impossible",
];

/// How a usage-limit message starts once its ACP wrapper is removed
/// (`benchflow.agents.usage_limits.USAGE_LIMIT_PREFIXES`).
const USAGE_LIMIT_PREFIXES: &[&str] = &[
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
    "Claude AI usage limit reached",
];

fn contains_any(text: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| text.contains(marker))
}

/// The category of an agent error (`classify_error`); `None` for no error.
pub fn classify_error(error: &str) -> Option<&'static str> {
    if error.is_empty() {
        return None;
    }
    let lower = error.to_lowercase();
    // First: the evidence quoted after these may contain any other marker.
    if lower.starts_with("agent integration failure") {
        return Some("agent_integration");
    }
    if lower.starts_with("usage limit reached") || is_usage_limit_text(error) {
        return Some("usage_limit");
    }
    if lower.contains("agent idle for") {
        return Some("idle_timeout");
    }
    if lower.contains("install failed") {
        return Some("install_failure");
    }
    if lower.contains("closed stdout")
        || lower.starts_with("pty ")
        || lower.starts_with("daytonaptyprocess:")
    {
        return Some("pipe_closed");
    }
    if lower.contains("suspected provider api error") {
        return Some("suspected_api_error");
    }
    if lower.contains("provider api error") {
        return Some("api_error");
    }
    if lower.contains("acp error")
        || lower.contains("native harness error")
        || lower.contains("was rejected as invalid")
    {
        return Some(if contains_any(&lower, PROVIDER_AUTH) {
            "provider_auth"
        } else if contains_any(&lower, PROVIDER_RATE_LIMIT) {
            "provider_rate_limit"
        } else if contains_any(&lower, PROVIDER_UNAVAILABLE) {
            "infra_failure"
        } else if contains_any(&lower, PROVIDER_REJECTED) {
            "provider_rejected"
        } else {
            "acp_error"
        });
    }
    if lower.contains("sandbox startup")
        || lower.contains("sandbox creation")
        || is_registry_metadata_error(&lower)
    {
        return Some("sandbox_setup");
    }
    if lower.contains("prompt exceeded wall-clock budget") {
        return Some("timeout");
    }
    if contains_any(&lower, INFRA) {
        return Some("infra_failure");
    }
    if lower.contains("timed out") {
        return Some("timeout");
    }
    Some("other")
}

fn is_registry_metadata_error(lower: &str) -> bool {
    lower.contains("docker compose command failed")
        && (lower.contains("failed to resolve source metadata")
            || lower.contains("failed to do request: head")
            || lower.contains("deadlineexceeded"))
}

/// The category of a verifier error (`classify_verifier_error`).
pub fn classify_verifier_error(error: &str) -> Option<&'static str> {
    if error.is_empty() {
        return None;
    }
    let lower = error.to_lowercase();
    if lower.starts_with("separate verifier ") {
        // The verifier sandbox never received the outputs: nothing assessed.
        return Some("verifier_infra");
    }
    if lower.contains("verifier_wedge:") || lower.contains("verifier recovery ") {
        return Some("verifier_infra");
    }
    // These two checks read the original case, as 0.8 does.
    if error.contains("verifier crashed") {
        if contains_any(&lower, VERIFIER_DEP_INSTALL) {
            return Some("verifier_dep_install");
        }
        if contains_any(&lower, INFRA) || contains_any(&lower, VERIFIER_INFRA) {
            return Some("verifier_infra");
        }
        return Some("verifier_failure");
    }
    if error.contains("verifier timed out") {
        return Some("verifier_timeout");
    }
    Some("verifier_other")
}

/// Whether the text is an agent's usage-limit message, possibly wrapped in
/// `ACP error -32603: Internal error:` (`is_usage_limit_text`).
fn is_usage_limit_text(text: &str) -> bool {
    let plain = py::strip(text).replace('\u{2019}', "'");
    let words = py::strip(strip_acp_wrapper(&plain));
    USAGE_LIMIT_PREFIXES
        .iter()
        .any(|prefix| words.starts_with(prefix))
}

/// Remove a leading `ACP error <code>:` and `Internal error:` (the regex
/// `^(?:ACP error -?\d+:\s*)?(?:Internal error:\s*)?`, case-insensitive).
fn strip_acp_wrapper(text: &str) -> &str {
    let mut rest = text;
    if let Some(after) = strip_prefix_ignore_case(rest, "acp error ") {
        let after = after.strip_prefix('-').unwrap_or(after);
        let code = after.trim_start_matches(|ch: char| ch.is_ascii_digit());
        if code.len() < after.len()
            && let Some(after_colon) = code.strip_prefix(':')
        {
            rest = after_colon.trim_start_matches(py::is_space);
        }
    }
    if let Some(after) = strip_prefix_ignore_case(rest, "internal error:") {
        rest = after.trim_start_matches(py::is_space);
    }
    rest
}

fn strip_prefix_ignore_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &text[prefix.len()..])
}

/// The declared outcome-assessment status (`assessment_status`): `None`
/// when the result declares none; an unrecognized value is `unassessable`.
pub fn assessment_status(result: &Map) -> Option<&'static str> {
    const STATUSES: [&str; 4] = ["pending", "verified", "failed", "unassessable"];
    let assessment = get(result, "assessment");
    if assessment.is_null() {
        // A legacy physical-trial manifest row.
        return (get(result, "status").as_str() == Some("awaiting_assessment"))
            .then_some("pending");
    }
    let status = match assessment {
        Value::Object(map) => get(map, "status"),
        other => other,
    };
    Some(
        status
            .as_str()
            .and_then(|status| STATUSES.into_iter().find(|known| *known == status))
            .unwrap_or("unassessable"),
    )
}

/// Whether a declared assessment is pending or unassessable, which makes
/// the result unscored whatever reward it carries.
pub fn assessment_withholds_score(result: &Map) -> bool {
    matches!(assessment_status(result), Some(status) if status != "verified" && status != "failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_errors_classify_like_0_8() {
        for (error, expected) in [
            (
                "Agent integration failure: login; ACP error http 401",
                "agent_integration",
            ),
            (
                "ACP error -32603: Internal error: You've hit your weekly limit",
                "usage_limit",
            ),
            (
                "ACP error -32603: Internal error: You\u{2019}ve reached your limit",
                "usage_limit",
            ),
            ("agent idle for 600s", "idle_timeout"),
            ("pty closed by the peer", "pipe_closed"),
            (
                "ACP error -32603: provider auth failed (HTTP 401)",
                "provider_auth",
            ),
            (
                "ACP error -32603: provider unavailable (HTTP 503)",
                "infra_failure",
            ),
            ("ACP error -32603: Internal error", "acp_error"),
            ("Prompt exceeded wall-clock budget", "timeout"),
            ("connection reset by peer", "infra_failure"),
            ("agent timed out", "timeout"),
            ("boom", "other"),
        ] {
            assert_eq!(classify_error(error), Some(expected), "{error}");
        }
        assert_eq!(classify_error(""), None);
    }

    #[test]
    fn verifier_errors_classify_like_0_8() {
        for (error, expected) in [
            ("separate verifier never started", "verifier_infra"),
            (
                "verifier crashed: Failed to get session command: ",
                "verifier_infra",
            ),
            (
                "verifier crashed: dependency install failed",
                "verifier_dep_install",
            ),
            ("verifier crashed: boom", "verifier_failure"),
            ("Verifier crashed: boom", "verifier_other"),
            ("verifier timed out after 30.0s", "verifier_timeout"),
        ] {
            assert_eq!(classify_verifier_error(error), Some(expected), "{error}");
        }
    }

    #[test]
    fn assessments_withhold_only_until_assessed() {
        let result = |text: &str| {
            crate::pyjson::parse(text)
                .unwrap()
                .as_object()
                .unwrap()
                .clone()
        };
        assert!(assessment_withholds_score(&result(
            r#"{"assessment": {"status": "pending"}}"#
        )));
        assert!(assessment_withholds_score(&result(
            r#"{"assessment": {"status": ["verified"]}}"#
        )));
        assert!(assessment_withholds_score(&result(
            r#"{"status": "awaiting_assessment"}"#
        )));
        assert!(!assessment_withholds_score(&result(
            r#"{"assessment": "failed"}"#
        )));
        assert!(!assessment_withholds_score(&result(
            r#"{"rewards": {"reward": 1}}"#
        )));
    }
}
