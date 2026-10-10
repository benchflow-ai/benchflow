//! The plain-text formats: `text`, `code-puppy-text`, `gemini-text` and
//! `aider-text`.
//!
//! - `text`: every line is the agent's message (a line that reads like a
//!   provider error is also a notice that explains a failed run);
//! - `code-puppy-text`: `text`, then Code Puppy's usage file;
//! - `gemini-text`: `gemini --prompt` prints only its final answer on
//!   stdout, so every line is the message; Gemini CLI's own notices
//!   (`Loaded cached credentials.`, YOLO-mode banners) go to the log. Its
//!   session file has the tool calls, but reading it means another trip to
//!   the sandbox after the run, so this build leaves it there.
//! - `aider-text`: aider's console. `Tokens: 4.2k sent, 1.1k cache hit, 512
//!   received. Cost: $0.01 message, $0.03 session.` is usage, `Applied edit
//!   to <file>` is an edit, `Running <command>` (after a `Run shell
//!   command?` that `--yes` answers) is a command, the startup banner goes to
//!   the log, a LiteLLM error (`litellm.AuthenticationError: ...` and the
//!   lines after it, up to a blank line) is a failure (aider prints it and
//!   exits 0), and everything else is the agent's message.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::json;

use crate::headless::classify::classify;
use crate::headless::parser::{Failure, Out, Parser, ToolDone, ToolStart, message, one_line};
use crate::trajectory::Usage;

/// Every line is the agent's message. A line that reads like a provider
/// error (a rejected key, a usage limit) is also kept as a notice, so a run
/// that fails is explained by it even though the line was the message.
#[derive(Default)]
pub struct PlainText;

impl Parser for PlainText {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        out.push(message(&format!("{line}\n")));
        explain(line, out);
        true
    }
}

/// A notice for `line` when it reads like a provider error.
fn explain(line: &str, out: &mut Vec<Out>) {
    if classify(line, chrono::Utc::now()).is_some() {
        out.push(Out::Failure(Failure::notice(line.trim())));
    }
}

/// Code Puppy's console, then the usage file it writes when it returns.
///
/// Every console line is the agent's message (as `text`). After Code Puppy
/// exits, the command prints its `--usage-file` as `{"@bf": "usage",
/// "item": {"input_tokens", "output_tokens", "cache_read_tokens",
/// "cost_usd"}}`. Code Puppy exits 0 when it swallowed a model or login
/// error, without writing the file, so the command exits 1 when the file is
/// missing (Harbor's guard), and the console's error lines explain it.
#[derive(Default)]
pub struct CodePuppyText;

impl Parser for CodePuppyText {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if let Some((name, item)) = super::follow::followed(line) {
            if name == "usage" {
                let usage = Usage {
                    input_tokens: crate::headless::parser::count(item.get("input_tokens")),
                    output_tokens: crate::headless::parser::count(item.get("output_tokens")),
                    cached_read_tokens: crate::headless::parser::count(
                        item.get("cache_read_tokens"),
                    ),
                    ..Usage::default()
                };
                if usage != Usage::default() {
                    out.push(Out::Usage(usage));
                }
                if let Some(cost) = item.get("cost_usd").and_then(serde_json::Value::as_f64) {
                    out.push(Out::Cost(cost));
                }
            }
            return true;
        }
        PlainText.line(line, out)
    }
}

/// Gemini CLI's non-interactive text output.
#[derive(Default)]
pub struct GeminiText;

const GEMINI_NOISE: [&str; 6] = [
    "Loaded cached credentials.",
    "YOLO mode is enabled.",
    "Data collection is disabled.",
    "[STARTUP]",
    "[DEBUG]",
    "Flushing log events",
];

impl Parser for GeminiText {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if GEMINI_NOISE
            .iter()
            .any(|n| line.trim_start().starts_with(n))
        {
            return false;
        }
        out.push(message(&format!("{line}\n")));
        true
    }
}

static TOKENS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^Tokens: (?P<rest>.+?)\.(?: Cost: \$(?P<msg>[\d.,]+) message, \$(?P<session>[\d.,]+) session\.?)?\s*$")
        .expect("valid regex")
});
static TOKEN_PART: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<n>[\d.,]+)(?P<unit>[kKmM]?) (?P<what>sent|received|cache write|cache hit)")
        .expect("valid regex")
});
static APPLIED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^Applied edit to (?P<file>.+?)\s*$").expect("valid regex"));
static RUNNING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^Running (?P<cmd>.+?)\s*$").expect("valid regex"));

/// aider's startup banner and prompts, kept for the log.
const AIDER_NOISE: [&str; 18] = [
    "Warning: Input is not a terminal",
    "Analytics have been",
    "Model: ",
    "Cost estimates may be inaccurate",
    "Aider v",
    "Main model:",
    "Weak model:",
    "Editor model:",
    "Git repo:",
    "Repo-map:",
    "Added ",
    "Use /help",
    "Run shell command",
    "Add command output",
    "Create new file",
    "Allow edits to",
    "Commit ",
    "https://aider.chat/",
];

/// aider's console.
#[derive(Default)]
pub struct AiderText {
    usage: Usage,
    calls: usize,
    /// Inside a LiteLLM error block, which runs to the next blank line.
    in_error: bool,
    /// Blank lines held back until text follows them.
    blank: usize,
}

impl Parser for AiderText {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let trimmed = line.trim();
        if trimmed.starts_with("litellm.") && trimmed.contains("Error") {
            self.in_error = true;
            out.push(Out::Failure(Failure::fatal(trimmed)));
            return true;
        }
        if self.in_error {
            if trimmed.is_empty() {
                self.in_error = false;
            } else {
                out.push(Out::Failure(Failure::notice(trimmed)));
            }
            return true;
        }
        if let Some(c) = TOKENS.captures(line) {
            for part in TOKEN_PART.captures_iter(&c["rest"]) {
                let n = parse_count(&part["n"], &part["unit"]);
                let u = &mut self.usage;
                let slot = match &part["what"] {
                    "sent" => &mut u.input_tokens,
                    "received" => &mut u.output_tokens,
                    "cache write" => &mut u.cached_write_tokens,
                    _ => &mut u.cached_read_tokens,
                };
                *slot = Some(slot.unwrap_or(0) + n);
            }
            out.push(Out::Usage(self.usage));
            if let Some(session) = c.name("session")
                && let Ok(cost) = session.as_str().replace(',', "").parse::<f64>()
            {
                out.push(Out::Cost(cost));
            }
            return true;
        }
        if let Some(c) = APPLIED.captures(line) {
            self.call(
                out,
                "edit",
                format!("edit {}", &c["file"]),
                json!({"path": &c["file"]}),
            );
            return true;
        }
        if let Some(c) = RUNNING.captures(line) {
            self.call(
                out,
                "execute",
                one_line(&c["cmd"]),
                json!({"command": &c["cmd"]}),
            );
            return true;
        }
        if AIDER_NOISE.iter().any(|n| line.starts_with(n)) || line.starts_with('─') {
            return false;
        }
        if trimmed.is_empty() {
            self.blank += 1;
            return true;
        }
        let blank = "\n".repeat(std::mem::take(&mut self.blank));
        out.push(message(&format!("{blank}{line}\n")));
        true
    }
}

impl AiderText {
    fn call(&mut self, out: &mut Vec<Out>, kind: &str, title: String, input: serde_json::Value) {
        self.calls += 1;
        let id = format!("aider-{}", self.calls);
        out.push(
            ToolStart {
                id: &id,
                title,
                kind,
                status: "in_progress",
                input: Some(input),
                ..ToolStart::default()
            }
            .out(),
        );
        out.push(
            ToolDone {
                id: &id,
                status: "completed",
                ..ToolDone::default()
            }
            .out(),
        );
    }
}

/// `4.2k` is 4200, `1,234` is 1234, `1.1M` is 1100000.
fn parse_count(number: &str, unit: &str) -> u64 {
    let n: f64 = number.replace(',', "").parse().unwrap_or(0.0);
    let scale = match unit {
        "k" | "K" => 1e3,
        "m" | "M" => 1e6,
        _ => 1.0,
    };
    (n * scale).round() as u64
}

#[cfg(test)]
mod tests {
    use crate::headless::tests::{events_of, run_lines};
    use crate::trajectory::Event;

    fn lines(text: &str) -> Vec<String> {
        text.lines().map(str::to_string).collect()
    }

    #[tokio::test]
    async fn aider_console_becomes_edits_usage_and_text() {
        let run = run_lines(
            "aider-text",
            &lines(
                "Aider v0.86.2\nMain model: gpt-4o with diff edit format\nGit repo: .git with 3 files\n\
                 I'll fix the cipher.\n\nsrc/cipher.py\nApplied edit to src/cipher.py\n\
                 Commit 1a2b3c4 fix: cipher\nRunning python -m pytest -q\n\
                 Tokens: 4.2k sent, 1,100 cache hit, 512 received. Cost: $0.01 message, $0.03 session.\n\
                 Tokens: 1.0k sent, 88 received. Cost: $0.01 message, $0.04 session.",
            ),
        )
        .await;
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            ["user_message", "agent_message", "tool_call", "tool_call"],
            "{events:?}"
        );
        assert!(
            matches!(&events[1], Event::AgentMessage(t) if t.text == "I'll fix the cipher.\n\nsrc/cipher.py\n")
        );
        assert!(
            matches!(&events[2], Event::ToolCall(c) if c.kind == "edit" && c.title == "edit src/cipher.py")
        );
        assert!(
            matches!(&events[3], Event::ToolCall(c) if c.kind == "execute" && c.title == "python -m pytest -q")
        );
        let usage = run.usage.unwrap();
        assert_eq!(
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_read_tokens
            ),
            (Some(5200), Some(600), Some(1100))
        );
        assert_eq!(run.cost_usd, Some(0.04));
        assert!(run.log.iter().any(|l| l.starts_with("Aider v0.86.2")));
    }

    #[tokio::test]
    async fn a_litellm_error_fails_the_run_even_though_aider_exits_0() {
        // Seen live with a bad Anthropic key: aider prints this and exits 0.
        let run = run_lines(
            "aider-text",
            &lines(
                "Analytics have been permanently disabled.\n\nModel: anthropic/claude-sonnet-4-6 with whole edit format, infinite output\n\n\
                 litellm.AuthenticationError: AnthropicException - \n\
                 {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid \n\
                 x-api-key\"},\"request_id\":\"req_1\"}\n\
                 The API provider is not able to authenticate you. Check your API key.\n",
            ),
        )
        .await;
        let err = run.result.as_ref().unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
        let messages = events_of(&run)
            .iter()
            .filter(|e| matches!(e, Event::AgentMessage(_)))
            .count();
        assert_eq!(messages, 0, "{:?}", events_of(&run));
    }

    #[tokio::test]
    async fn gemini_text_is_the_answer_and_its_notices_are_log() {
        let run = run_lines(
            "gemini-text",
            &lines("Loaded cached credentials.\nThe file now says hello.\nDone."),
        )
        .await;
        let events = events_of(&run);
        assert!(
            matches!(&events[1], Event::AgentMessage(t) if t.text == "The file now says hello.\nDone.\n")
        );
        assert_eq!(run.log, ["Loaded cached credentials."]);
    }
}
