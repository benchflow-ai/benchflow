//! Recognizing why a headless agent failed from its own words.
//!
//! A CLI run headless reports a failure as text: an error event, a final
//! result, or a line on stderr. The patterns here are Harbor's
//! `ERROR_PATTERNS` (`agents/installed/base.py`) and BenchFlow 0.8's
//! usage-limit and provider markers, grouped by what the user does next:
//!
//! - a spent subscription or credit balance (R213): Claude Code's and
//!   Codex's usage-limit messages, parsed for the window and reset time;
//! - a rejected login (R212);
//! - a model the login cannot use (R215);
//! - a provider that limited, failed or refused the request (R214).
//!
//! In a block of text the match nearest the end wins, as in Harbor: a run
//! that retried a rate limit and then hit an auth error failed on the auth
//! error.

use std::sync::LazyLock;

use chrono::{DateTime, Utc};
use regex::Regex;

use crate::acp::limits::{LimitInfo, parse_limit_text};

/// What an agent's error text says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Signal {
    UsageLimit(LimitInfo),
    Login,
    Model,
    RateLimit,
    Unavailable,
    Rejected,
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|e| panic!("BenchFlow bug: bad pattern {pattern}: {e}"))
}

/// Patterns in order of precedence within one line.
static PATTERNS: LazyLock<Vec<(Regex, Signal)>> = LazyLock::new(|| {
    let usage = |p: &str| {
        (
            re(&format!("(?i){p}")),
            Signal::UsageLimit(LimitInfo {
                detail: String::new(),
                window: None,
                resets_at: None,
            }),
        )
    };
    vec![
        // Spent subscriptions and credits.
        usage(r"specified API usage limits"),
        usage(r"You.ve hit your usage limit"),
        usage(r"You have an unpaid invoice"),
        usage(r"Quota exceeded\."),
        usage(r"You have no credits remaining"),
        usage(r"Free tier users do not have access|Upgrade to paid credits"),
        usage(r"Budget has been exceeded!"),
        usage(r"credit balance is too low|insufficient_quota|exceeded your current quota"),
        // Logins.
        (
            re(concat!(
                r"(?i)Not logged in|\binvalid_api_key\b|Invalid API key\b|API key is invalid\b|",
                r"Incorrect API key provided\b|unexpected status 401\b|\b401 Unauthorized\b|",
                r"invalid x-api-key|\bAuthenticationError\b|not able to authenticate you|\bstatus(?:Code)?\W{1,3}401\b|",
                r"Invalid proxy server token passed|\btoken_not_found_in_db\b|",
                r"Please run /login|OAuth token has expired|authentication_error|",
                r"Failed to authenticate|invalid bearer token|provider auth failed|",
                r"unexpected status 403\b|\bHTTP 401\b|\bpermission_denied\b|API key not valid|",
                r"No API key found|missing API key|API key (?:is )?(?:required|not set)|",
                // Moonshot (Kimi Code); the Atlassian CLI (Rovo Dev) and Muse Code.
                r"Invalid Authentication\b|\binvalid_authentication_error\b|\bauthentication failed\b|",
                // Muse Code; GitHub's API (Copilot); Junie's log.
                r"API key from \w+ was rejected|GitHub returned: Bad credentials|reason=AuthorizationFailed",
            )),
            Signal::Login,
        ),
        // Models.
        (
            re(concat!(
                r"(?i)Cannot use this model|\blitellm\.NotFoundError\b|\bmodel_not_found\b|",
                r"model .{1,80} (?:does not exist|is not supported|not found)|",
                r"does not have access to model|Unknown model\b|invalid model\b",
            )),
            Signal::Model,
        ),
        // Providers.
        (
            re(r"(?i)rate.?limit|too many requests|\bHTTP 429\b|unexpected status 429\b"),
            Signal::RateLimit,
        ),
        (
            re(concat!(
                r"(?i)API Error: 500 Internal server error|API Error: Server error mid-response|",
                r"API Error: Overloaded|ServiceUnavailableError|overloaded_error|",
                r"unexpected status 50[0234]\b|\bHTTP 50[0234]\b|provider unavailable|",
                r"API Error: (?:Connection closed|Connection lost) mid-response|",
                r"API Error: Connection to the API was lost|stream closed before completion|",
                r"Could not resolve host|Connection refused|Connection timed out|No route to host|",
                r"SSL_ERROR_SYSCALL|Error when talking to Gemini API",
            )),
            Signal::Unavailable,
        ),
        (
            re(concat!(
                r"(?i)input token count exceeds the maximum|prompt is too long: \d+ tokens > \d+ maximum|",
                r"context_length_exceeded|context window|provider rejected request|\bHTTP 400\b",
            )),
            Signal::Rejected,
        ),
    ]
});

/// What one line says, if anything.
fn classify_line(line: &str, now: DateTime<Utc>) -> Option<Signal> {
    if let Some(info) = parse_limit_text(line, now) {
        return Some(Signal::UsageLimit(info));
    }
    PATTERNS.iter().find_map(|(re, signal)| {
        let m = re.find(line)?;
        Some(match signal {
            Signal::UsageLimit(_) => {
                // Not Claude's or Codex's exact words: keep the line as the detail.
                let detail = line[m.start()..].trim().to_string();
                Signal::UsageLimit(LimitInfo {
                    detail,
                    window: None,
                    resets_at: None,
                })
            }
            other => other.clone(),
        })
    })
}

/// What a block of text says, the last line that says something winning.
pub(crate) fn classify(text: &str, now: DateTime<Utc>) -> Option<(Signal, String)> {
    text.lines()
        .rev()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .find_map(|line| classify_line(line, now).map(|s| (s, line.to_string())))
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 1, 12, 0, 0).unwrap()
    }

    fn kind(text: &str) -> Option<&'static str> {
        classify(text, now()).map(|(s, _)| match s {
            Signal::UsageLimit(_) => "usage",
            Signal::Login => "login",
            Signal::Model => "model",
            Signal::RateLimit => "rate",
            Signal::Unavailable => "down",
            Signal::Rejected => "rejected",
        })
    }

    #[test]
    fn harbors_patterns_by_what_to_do_next() {
        // Harbor's test_error_patterns.py, grouped.
        let cases = [
            (
                "Error: You've hit your usage limit. Try again later.",
                Some("usage"),
            ),
            (
                "You have exceeded your specified API usage limits",
                Some("usage"),
            ),
            (
                "Your credit balance is too low to access the Anthropic API",
                Some("usage"),
            ),
            ("Invalid API key · Please run /login", Some("login")),
            ("Not logged in · Please run /login", Some("login")),
            (
                r#"{"type":"turn.failed","error":{"message":"unexpected status 401 Unauthorized"}}"#,
                Some("login"),
            ),
            ("Invalid proxy server token passed", Some("login")),
            // Seen live: Kimi Code on Moonshot, Rovo Dev's `acli rovodev auth login`.
            (
                "error: failed to run prompt: provider.auth_error: 401 Invalid Authentication",
                Some("login"),
            ),
            ("✗ Error: authentication failed", Some("login")),
            // Seen live with placeholder keys: OpenClaw, MiniMax Code, Devin's log.
            (
                "[openclaw] Reason: ⚠️ anthropic/claude-haiku-4-5 request failed (authentication failed, HTTP 401). Re-authenticate the provider and try again.",
                Some("login"),
            ),
            (
                "BYOK provider custom_provider:benchflow-anthropic upstream error: 401 invalid x-api-key",
                Some("login"),
            ),
            (
                "WARN windsurf_api_client::remote_config: remote config revalidation failed, keeping last-good value: Authentication required: invalid api key",
                Some("login"),
            ),
            (
                "failed to fetch model catalog: authentication failed: your API key from META_API_KEY was rejected — update or unset it",
                Some("login"),
            ),
            // Seen live with placeholder keys: Copilot CLI, Junie's log.
            (
                "  Failed to fetch PAT user login (401): GitHub returned: Bad credentials",
                Some("login"),
            ),
            (
                "WARN  c.i.m.l.m.ArtifactRequestExecutorService - Failed to create artifact: issue.md.junie_standalone, reason=AuthorizationFailed",
                Some("login"),
            ),
            ("Cannot use this model with your plan", Some("model")),
            (
                "litellm.NotFoundError: model gpt-0 not found",
                Some("model"),
            ),
            ("API Error: 429 rate limit exceeded", Some("rate")),
            ("API Error: Overloaded", Some("down")),
            (
                "curl: (6) Could not resolve host: api.anthropic.com",
                Some("down"),
            ),
            (
                "prompt is too long: 210000 tokens > 200000 maximum",
                Some("rejected"),
            ),
            ("everything is fine", None),
        ];
        for (text, want) in cases {
            assert_eq!(kind(text), want, "{text}");
        }
    }

    #[test]
    fn the_last_line_that_says_something_wins() {
        let text = "API Error: 429 rate limit, retrying\nmore output\nInvalid API key · Please run /login\n";
        assert_eq!(kind(text), Some("login"));
        let text = "Invalid API key\nlater: rate limit exceeded";
        assert_eq!(kind(text), Some("rate"));
    }

    #[test]
    fn claude_limits_keep_their_window_and_reset() {
        let (signal, line) =
            classify("You've hit your session limit · resets 4pm (UTC)", now()).unwrap();
        let Signal::UsageLimit(info) = signal else {
            panic!()
        };
        assert_eq!(info.window.as_deref(), Some("5-hour"));
        assert!(info.resets_at.is_some());
        assert!(line.starts_with("You've hit"));
    }
}
