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
//! - a provider that limited, failed or refused the request (R214);
//! - a prompt over the model's context window, which the agent sent.
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
    /// The prompt was over the model's context window: the agent's to
    /// manage, not the provider's failure.
    Overflow,
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
                r"model .{1,80} (?:does not exist|is not supported|not found|has reached its end of life)|",
                r"does not have access to model|Unknown model\b|invalid model\b|",
                // A model the provider retired (NVIDIA's API for NeMo Agent Toolkit).
                r"\[410\] Gone|\bHTTP 410\b|unexpected status 410\b",
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
                r"API Error: 5\d\d\b|API Error: Request timed out|API Error: Connection error|",
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
            // The native loop's phrases, one list for both, and Gemini's
            // wording, which has a number in the middle.
            re(&format!(
                r"(?i){}|input token count(?: \(\d+\))? exceeds the maximum",
                crate::native::error::CONTEXT_PHRASES
                    .iter()
                    .map(|p| regex::escape(p))
                    .collect::<Vec<_>>()
                    .join("|")
            )),
            Signal::Overflow,
        ),
        (
            // A Responses API without a tool the agent sent (MiMo's answer
            // to Codex's hosted `web_search`).
            re(r"(?i)provider rejected request|\bHTTP 400\b|\bresponses_feature_not_supported\b"),
            Signal::Rejected,
        ),
    ]
});

/// A retry delay with a number ("Please retry in 20.5s", "retry after 30",
/// `retryDelay`): a per-minute rate limit that some providers word like a
/// spent quota ("You exceeded your current quota"), not a spent balance. A
/// spent balance that only says to retry later ("retry after topping up")
/// has no number and stays a usage limit, and so does a spent daily quota,
/// whose message gives a delay too (`PerDay` in its quota id).
static RETRY_HINT: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)\bretry (?:in|after) \d|retrydelay"));
static DAILY: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)per.?day|\bdaily\b"));

/// What one line says, if anything. A spent-quota sentence's retry delay
/// and day are looked for with `around`, which says whether the error that
/// holds the line matches a pattern (the delay is often on a line of its
/// own, and the day only in a quota id in the details).
fn classify_line(
    line: &str,
    around: &dyn Fn(&Regex) -> bool,
    now: DateTime<Utc>,
) -> Option<Signal> {
    if let Some(info) = parse_limit_text(line, now) {
        return Some(Signal::UsageLimit(info));
    }
    // Codex's warning for a model outside its catalog, which every provider
    // model is; the run goes on with default settings.
    if line.contains("Defaulting to fallback metadata") {
        return None;
    }
    PATTERNS.iter().find_map(|(re, signal)| {
        let m = re.find(line)?;
        Some(match signal {
            Signal::UsageLimit(_) if around(&RETRY_HINT) && !around(&DAILY) => Signal::RateLimit,
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

/// What a block of text says (a log, stderr), the last line that says
/// something winning. A line's error runs from it to the next blank line:
/// a retry notice earlier in the block does not relabel a later spent
/// balance. Within one such paragraph a spent-quota sentence wins over a
/// rate-limit word on a later line, as in [`classify_error`]: Google's
/// daily 429 printed as indented JSON ends with its help link
/// (`…/docs/rate-limits`) below the quota sentence.
pub(crate) fn classify(text: &str, now: DateTime<Utc>) -> Option<(Signal, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let at = |i: usize| {
        let line = lines[i].trim();
        if line.is_empty() {
            return None;
        }
        // Both patterns match within a line.
        let around = |re: &Regex| {
            lines[i..]
                .iter()
                .take_while(|l| !l.trim().is_empty())
                .any(|l| re.is_match(l))
        };
        classify_line(line, &around, now).map(|s| (s, line.to_string()))
    };
    let (i, found) = (0..lines.len()).rev().find_map(|i| at(i).map(|f| (i, f)))?;
    if found.0 != Signal::RateLimit {
        return Some(found);
    }
    // At most 40 lines back: stderr logs often have no blank lines, and a
    // usage-limit sentence long before must not relabel a later rate limit.
    let floor = i.saturating_sub(40);
    let start = (floor..i)
        .rev()
        .find(|&j| lines[j].trim().is_empty())
        .map_or(floor, |j| j + 1);
    let quota = (start..i)
        .rev()
        .find_map(|j| at(j).filter(|(s, _)| matches!(s, Signal::UsageLimit(_))));
    Some(quota.unwrap_or(found))
}

/// What one error says (an ACP error's message and `data.details`, one
/// failure a headless agent reported): as [`classify`], but the retry delay
/// and day are read over the whole error, and its spent-quota sentence wins
/// over a rate-limit word on a later line (Google's help link,
/// `…/docs/rate-limits`).
pub(crate) fn classify_error(text: &str, now: DateTime<Utc>) -> Option<(Signal, String)> {
    let found = |pick: &dyn Fn(&Signal) -> bool| {
        text.lines()
            .rev()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .find_map(|line| {
                classify_line(line, &|re: &Regex| re.is_match(text), now)
                    .filter(|s| pick(s))
                    .map(|s| (s, line.to_string()))
            })
    };
    match found(&|_| true) {
        Some((Signal::RateLimit, line)) => {
            // A quota sentence is read with the whole error's delay and
            // day, so it is a rate limit itself unless the quota is spent.
            let quota = found(&|s| matches!(s, Signal::UsageLimit(_)));
            Some(quota.unwrap_or((Signal::RateLimit, line)))
        }
        other => other,
    }
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
            Signal::Overflow => "overflow",
        })
    }

    /// A per-minute 429 worded like a spent quota but giving a retry delay
    /// is a rate limit (the provider's, retried), not a usage limit, which
    /// stops the whole job; a spent balance that only says to retry later
    /// stays one. Guards the follow-up of the d7d and orchestrator reviews.
    #[test]
    fn a_quota_message_with_a_retry_delay_is_a_rate_limit() {
        for (text, want) in [
            (
                "You exceeded your current quota. Please retry in 20.5s.",
                "rate",
            ),
            (
                "You exceeded your current quota, please check your plan and billing details.",
                "usage",
            ),
            (
                "Your credit balance is too low, retry after topping up",
                "usage",
            ),
            ("insufficient_quota", "usage"),
            (
                "You exceeded your current quota (GenerateRequestsPerDayPerProjectPerModel-FreeTier). Please retry in 3600s.",
                "usage",
            ),
        ] {
            assert_eq!(kind(text), Some(want), "{text}");
        }
    }

    /// Google's 429 message puts the retry delay on its own line, and the
    /// day only in the quota id: read from the quota sentence to the next
    /// blank line in a block (stderr), and over the whole of one error. A
    /// retry notice before a blank line does not relabel a later spent
    /// balance (REVIEW-C (orch) 1).
    #[test]
    fn a_quota_errors_retry_delay_and_day_may_be_on_other_lines() {
        let google = "You exceeded your current quota, please check your plan and billing details. \
                      For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.\n\
                      * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests\n\
                      Please retry in 20.6s.";
        assert_eq!(kind(google), Some("rate"), "{google}");
        let daily =
            format!("{google}\n* Quota id: GenerateRequestsPerDayPerProjectPerModel-FreeTier");
        assert_eq!(kind(&daily), Some("usage"), "{daily}");
        let apart = "Please retry in 5s.\n\nYou exceeded your current quota";
        assert_eq!(kind(apart), Some("usage"), "{apart}");
        // Google's daily 429 printed to stderr as indented JSON: the help
        // link below the quota sentence no longer makes it a rate limit
        // (D13's low, fixed 2026-10-11); its per-minute twin stays one.
        let pretty = |quota_id: &str| {
            format!(
                "{{\n  \"error\": {{\n    \"code\": 429,\n    \"message\": \"You exceeded your current quota, please check your plan and billing details.\",\n    \"status\": \"RESOURCE_EXHAUSTED\",\n    \"details\": [\n      {{\"quotaId\": \"{quota_id}\"}},\n      {{\"retryDelay\": \"20s\"}},\n      {{\"links\": [{{\"url\": \"https://ai.google.dev/gemini-api/docs/rate-limits\"}}]}}\n    ]\n  }}\n}}"
            )
        };
        let daily = pretty("GenerateRequestsPerDayPerProjectPerModel-FreeTier");
        assert_eq!(kind(&daily), Some("usage"), "{daily}");
        let minute = pretty("GenerateRequestsPerMinutePerProjectPerModel-FreeTier");
        assert_eq!(kind(&minute), Some("rate"), "{minute}");
        // A spent-quota sentence more than 40 lines before a rate limit, in
        // a log with no blank lines, does not relabel it (diff review L9).
        let long = format!(
            "You exceeded your current quota, please check your plan and billing details.\n{}API Error: 429 rate limit exceeded",
            "working on it\n".repeat(60)
        );
        assert_eq!(kind(&long), Some("rate"), "{long}");
        // A rate limit after a blank line is its own error.
        let later = format!("{daily}\n\nAPI Error: 429 rate limit exceeded");
        assert_eq!(kind(&later), Some("rate"), "{later}");
        let one = |text: &str| classify_error(text, now()).map(|(s, _)| s);
        for (text, want) in [
            (
                "Internal error: You exceeded your current quota\n{\"retryDelay\":\"20s\"}",
                Signal::RateLimit,
            ),
            (
                "Internal error: You exceeded your current quota. Please retry in 33s.\n\
                 {\"quotaId\":\"GenerateRequestsPerDayPerProjectPerModel-FreeTier\",\
                 \"links\":[{\"url\":\"https://ai.google.dev/gemini-api/docs/rate-limits\"}]}",
                Signal::UsageLimit(LimitInfo {
                    detail: "exceeded your current quota. Please retry in 33s.".into(),
                    window: None,
                    resets_at: None,
                }),
            ),
            (
                "Internal error: API Error: 429 rate limit exceeded",
                Signal::RateLimit,
            ),
        ] {
            assert_eq!(one(text), Some(want), "{text}");
        }
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
            // Seen live: NeMo Agent Toolkit on a retired NVIDIA model.
            ("Exception: [410] Gone", Some("model")),
            (
                "Exception: [410] Gone\nThe model 'meta/llama-3.3-70b-instruct' has reached its end of life on 2026-08-26T09:00:00Z and is no longer available.",
                Some("model"),
            ),
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
                Some("overflow"),
            ),
            (
                "HTTP 400: Unsupported parameter 'temperature'",
                Some("rejected"),
            ),
            // Both on one line: the window, which the agent sent.
            ("HTTP 400: context_length_exceeded", Some("overflow")),
            // The native loop's wordings too (REVIEW-OUTCOMES L2).
            (
                "HTTP 400: This model's maximum context length is 8192 tokens",
                Some("overflow"),
            ),
            (
                "HTTP 400: Input is too long for requested model.",
                Some("overflow"),
            ),
            // The rest of the native loop's list (REVIEW-OUTCOMES (d7c) L3).
            ("HTTP 400: prompt is too long", Some("overflow")),
            (
                "HTTP 400: too many tokens: 300000 > 200000",
                Some("overflow"),
            ),
            // A rate limit that counts tokens stays a rate limit.
            (
                "Rate limit reached: too many tokens per minute",
                Some("rate"),
            ),
            (
                "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
                Some("overflow"),
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
    fn codexs_fallback_metadata_warning_is_not_a_model_error() {
        // Codex headless on MiMo's Responses API (2026-10-12): a warning,
        // then the provider's refusal of the web search tool.
        let text = "[benchflow] the agent reported an error (not fatal): Model metadata for `mimo-v2.6-flash` not found. Defaulting to fallback metadata; this can degrade performance and cause issues.\n\
             [benchflow] the agent reported an error: data:{\"error\":{\"code\":\"responses_feature_not_supported\",\"message\":\"tool type 'web_search' is not supported by this gateway phase.\",\"param\":\"\",\"type\":\"unsupported_feature\"}}\n\n\n";
        let (signal, line) = classify(text, now()).unwrap();
        assert!(matches!(signal, Signal::Rejected), "{line}");
        assert!(line.contains("web_search"), "{line}");
        assert_eq!(
            kind(
                "Model metadata for `glm-5.3` not found. Defaulting to fallback metadata; this can degrade performance and cause issues."
            ),
            None
        );
        assert_eq!(kind("Error: model `x` not found"), Some("model"));
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
