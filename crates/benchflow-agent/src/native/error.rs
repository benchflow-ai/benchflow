//! Model call failures, classified so a trial can say whose problem each is.

use std::time::Duration;

/// What kind of failure a model call had.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelErrorKind {
    /// The provider rejected the key (401, 403). R212.
    LoginRejected,
    /// The account is out of credit or over its quota (402, or a 429 that
    /// says so). R213.
    Quota,
    /// The provider does not offer the model, or rejects a setting (404,
    /// "model not found"). R215.
    ModelNotOffered,
    /// The conversation no longer fits the model's context window. The
    /// loops handle this themselves (Terminus-2 summarizes).
    ContextLength,
    /// Rate limited, a server error, a timeout, or the network: still failing
    /// after every retry. R214, transient.
    Unavailable,
    /// Any other rejected request, or an answer that could not be read. R214.
    Failed,
}

/// A failed model call.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct ModelError {
    pub kind: ModelErrorKind,
    /// The HTTP status, when there was a response.
    pub status: Option<u16>,
    /// One line: what happened, with the provider's own words.
    pub message: String,
    pub attempts: u32,
}

impl ModelError {
    pub fn new(kind: ModelErrorKind, status: Option<u16>, message: impl Into<String>) -> Self {
        Self {
            kind,
            status,
            message: message.into(),
            attempts: 1,
        }
    }

    /// The trial failure code (`bench explain R21x`).
    pub fn code(&self) -> &'static str {
        match self.kind {
            ModelErrorKind::LoginRejected => "R212",
            ModelErrorKind::Quota => "R213",
            ModelErrorKind::ModelNotOffered => "R215",
            ModelErrorKind::ContextLength
            | ModelErrorKind::Unavailable
            | ModelErrorKind::Failed => "R214",
        }
    }

    /// Whether running the trial again may succeed.
    pub fn transient(&self) -> bool {
        matches!(self.kind, ModelErrorKind::Unavailable)
    }

    /// A 403 that says the account's plan or tier does not include the
    /// model: the key works, so checking it is the wrong advice.
    pub fn plan_lacks_model(&self) -> bool {
        self.status == Some(403) && plan_restricted(&self.message)
    }

    /// What to do next.
    pub fn next_step(&self, provider: &str, key_label: &str) -> String {
        if self.plan_lacks_model() {
            return format!(
                "the {provider} key from {key_label} works, but its plan does not include this model: \
                 upgrade the plan, use a key on a plan that has it, or pick another model with -m"
            );
        }
        match self.kind {
            ModelErrorKind::LoginRejected => format!(
                "the {provider} key from {key_label} was rejected: check that it is current and allowed to use this model"
            ),
            ModelErrorKind::Quota => format!(
                "the {provider} account is out of credit or over its limit: add credit or wait for the limit to reset, then run again"
            ),
            ModelErrorKind::ModelNotOffered => format!(
                "check the model name against the {provider} model list; `bench agents builtin` shows how models are named"
            ),
            ModelErrorKind::ContextLength => {
                "the conversation outgrew the model's context window: raise the model's limit or shorten the task".into()
            }
            ModelErrorKind::Unavailable => format!(
                "{provider} kept failing after {} attempts: run again later (`bench run` resumes the job), or try another model",
                self.attempts
            ),
            ModelErrorKind::Failed => format!(
                "read agent/model_calls.jsonl in the trial folder for the request and the {provider} answer"
            ),
        }
    }
}

/// Phrases providers use for a prompt that no longer fits (Harbor's
/// `_is_context_length_error`, plus Anthropic's own wording).
const CONTEXT_PHRASES: &[&str] = &[
    "context length exceeded",
    "context_length_exceeded",
    "maximum context length",
    "`inputs` tokens + `max_new_tokens`",
    "model's context length",
    "prompt is too long",
    "input is too long for requested model",
    "context window",
    "too many tokens",
];

const QUOTA_PHRASES: &[&str] = &[
    "insufficient",
    "quota",
    "credit",
    "billing",
    "balance",
    "spend limit",
    "spending limit",
    "budget",
    "usage limit",
];

const MODEL_PHRASES: &[&str] = &[
    "model not found",
    "model_not_found",
    "does not exist",
    "unknown model",
    "no such model",
    "not supported",
    "is not available",
    "invalid model",
    "no endpoints found",
];

/// Phrases providers use when a working key's plan or tier lacks the model.
const PLAN_PHRASES: &[&str] = &[
    "free tier",
    "upgrade to",
    "your plan",
    "your current plan",
    "subscription tier",
    "do not have access to this model",
    "does not have access to this model",
    "not included in your",
    "not available on your",
];

/// Whether a provider's words say the plan, not the key, is the problem.
pub fn plan_restricted(text: &str) -> bool {
    let lower = text.to_lowercase();
    PLAN_PHRASES.iter().any(|p| lower.contains(p))
}

/// Classify a non-success HTTP answer. `retryable` says whether the caller
/// should try again (rate limits and server errors).
pub fn classify(status: u16, body: &str) -> (ModelErrorKind, bool) {
    let lower = body.to_lowercase();
    let has = |phrases: &[&str]| phrases.iter().any(|p| lower.contains(p));
    match status {
        401 | 403 => (ModelErrorKind::LoginRejected, false),
        402 => (ModelErrorKind::Quota, false),
        429 if has(QUOTA_PHRASES) && !lower.contains("rate limit") => {
            (ModelErrorKind::Quota, false)
        }
        408 | 409 | 425 | 429 => (ModelErrorKind::Unavailable, true),
        500..=599 => {
            if has(CONTEXT_PHRASES) {
                (ModelErrorKind::ContextLength, false)
            } else {
                (ModelErrorKind::Unavailable, true)
            }
        }
        _ if has(CONTEXT_PHRASES) => (ModelErrorKind::ContextLength, false),
        404 => (ModelErrorKind::ModelNotOffered, false),
        400 | 422 if has(MODEL_PHRASES) => (ModelErrorKind::ModelNotOffered, false),
        _ => (ModelErrorKind::Failed, false),
    }
}

/// The one-line message for a refused model call: the provider and its
/// words, led by what a 403 plan restriction means.
pub fn plan_message(provider: &str, status: u16, words: &str) -> String {
    let message = format!("{provider} answered {status}: {words}");
    if status == 403 && plan_restricted(words) {
        format!("the login works, but its plan does not include this model: {message}")
    } else {
        message
    }
}

/// The provider's error message from a JSON error body (`{"error":
/// {"message": ...}}` and its variants), else the body's start.
pub fn provider_message(body: &str) -> String {
    let parsed: Option<serde_json::Value> = serde_json::from_str(body).ok();
    let from_json = parsed.as_ref().and_then(|v| {
        let e = v.get("error").unwrap_or(v);
        e.get("message")
            .and_then(|m| m.as_str())
            .or_else(|| e.as_str())
            .or_else(|| v.get("message").and_then(|m| m.as_str()))
            .map(str::to_string)
    });
    let text = from_json.unwrap_or_else(|| body.trim().to_string());
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() > 300 {
        format!("{}…", one_line.chars().take(300).collect::<String>())
    } else {
        one_line
    }
}

/// How long to wait before attempt `attempt` (1-based retries): exponential
/// from `base`, capped at `cap`, with up to a quarter of jitter; a provider's
/// `Retry-After` wins when it is longer (also capped).
pub fn backoff(
    base: Duration,
    cap: Duration,
    attempt: u32,
    retry_after: Option<Duration>,
) -> Duration {
    let exp = base.saturating_mul(2u32.saturating_pow(attempt.saturating_sub(1)));
    let wait = exp.min(cap);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let jitter = wait.mul_f64(f64::from(nanos % 1000) / 4000.0);
    let wait = wait + jitter;
    match retry_after {
        Some(r) if r > wait => r.min(cap.max(Duration::from_secs(60))),
        _ => wait,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_map_to_codes() {
        let code = |s, b: &str| {
            let (kind, _) = classify(s, b);
            ModelError::new(kind, Some(s), "x").code()
        };
        assert_eq!(code(401, "invalid api key"), "R212");
        assert_eq!(code(403, ""), "R212");
        assert_eq!(code(402, ""), "R213");
        assert_eq!(
            code(429, r#"{"error":{"message":"insufficient credits"}}"#),
            "R213"
        );
        assert_eq!(code(429, "rate limit exceeded"), "R214");
        assert_eq!(code(404, "model_not_found"), "R215");
        assert_eq!(code(400, "The model `x` does not exist"), "R215");
        assert_eq!(code(500, "boom"), "R214");
        assert_eq!(code(400, "something else"), "R214");
    }

    #[test]
    fn a_plan_that_lacks_the_model_is_not_a_bad_key() {
        // computer-1 on the Vercel gateway's free tier, today.
        let body = r#"{"error":{"message":"Free tier users do not have access to this model. Upgrade to paid credits at https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%3Fmodal%3Dtop-up for unrestricted access.","type":"forbidden"}}"#;
        let (kind, retry) = classify(403, body);
        assert_eq!((kind, retry), (ModelErrorKind::LoginRejected, false));
        let e = ModelError::new(
            kind,
            Some(403),
            plan_message("vercel", 403, &provider_message(body)),
        );
        assert_eq!(e.code(), "R212");
        assert!(e.plan_lacks_model());
        assert!(
            e.message.starts_with(
                "the login works, but its plan does not include this model: vercel answered 403: \
                 Free tier users"
            ),
            "{}",
            e.message
        );
        let next = e.next_step("vercel", "AI_GATEWAY_API_KEY");
        assert!(
            next.contains("works, but its plan does not include this model"),
            "{next}"
        );
        assert!(!next.contains("check that it is current"), "{next}");
        // A plain bad key keeps the old advice.
        let bad = ModelError::new(
            ModelErrorKind::LoginRejected,
            Some(403),
            plan_message("vercel", 403, "Invalid API key"),
        );
        assert!(!bad.plan_lacks_model());
        assert!(
            bad.next_step("vercel", "K")
                .contains("check that it is current")
        );
    }

    #[test]
    fn rate_limits_and_server_errors_retry() {
        assert!(classify(429, "slow down").1);
        assert!(classify(503, "").1);
        assert!(!classify(400, "bad").1);
        assert!(!classify(401, "").1);
    }

    #[test]
    fn context_overflow_is_recognized() {
        let (kind, retry) = classify(
            400,
            r#"{"error":{"message":"This model's maximum context length is 8192 tokens"}}"#,
        );
        assert_eq!(kind, ModelErrorKind::ContextLength);
        assert!(!retry);
        assert_eq!(
            classify(400, "prompt is too long: 300000 tokens").0,
            ModelErrorKind::ContextLength
        );
    }

    #[test]
    fn provider_messages_are_one_line() {
        assert_eq!(
            provider_message(r#"{"error": {"message": "bad\nkey", "type": "auth"}}"#),
            "bad key"
        );
        assert_eq!(provider_message("plain text"), "plain text");
    }

    #[test]
    fn backoff_grows_and_honors_retry_after() {
        let base = Duration::from_millis(100);
        let cap = Duration::from_secs(10);
        assert!(backoff(base, cap, 1, None) < Duration::from_millis(130));
        assert!(backoff(base, cap, 4, None) >= Duration::from_millis(800));
        assert!(backoff(base, cap, 20, None) <= Duration::from_millis(12_500));
        assert_eq!(
            backoff(base, cap, 1, Some(Duration::from_secs(5))),
            Duration::from_secs(5)
        );
    }
}
