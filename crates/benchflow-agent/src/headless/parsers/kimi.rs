//! `kimi-stream-json`: `kimi --prompt ... --output-format stream-json`
//! (Kimi Code, the npm `@moonshot-ai/kimi-code` CLI), and `kimi --print
//! --output-format stream-json` (the Python kimi-cli, which prints the same
//! messages).
//!
//! One message per line, in chat format:
//!
//! - `{"role": "assistant", "content": "...", "reasoning_content": "...",
//!   "tool_calls": [{"id", "function": {"name", "arguments"}}]}` (the
//!   arguments a JSON string);
//! - `{"role": "tool", "tool_call_id", "content"}`;
//! - `{"role": "meta", "type": "system.version" | "session.resume_hint" |
//!   ...}`; a meta line whose type names an error, or that carries an
//!   `error`, is the CLI's failure.
//!
//! The stream has no token counts; the session journal does. After kimi
//! exits, the command prints the journal's (`$KIMI_CODE_HOME/sessions/*/
//! session_*/agents/main/wire.jsonl`) lines that mention usage as
//! `{"@bf": "wire", "item": <line>}`: `usage.record` events, `step.end`
//! loop events, or `agent.message.appended` events with `meta.usage`
//! (`inputOther`, `inputCacheRead`, `inputCacheCreation`, `output`),
//! depending on the version. kimi-cli's session journal
//! (`~/.kimi/sessions/*/*/wire.jsonl`) has `{"message": {"type":
//! "StatusUpdate", "payload": {"token_usage": {"input_other", "output",
//! "input_cache_read", "input_cache_creation"}}}}` records, one per step.
//! One kind is counted, so a step is not counted twice.

use serde_json::Value;

use super::follow::{ChatMessages, followed};
use crate::headless::parser::{Failure, Out, Parser, count, json_object, str_of};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct KimiStreamJson {
    chat: ChatMessages,
    /// Usage from `usage.record`, `step.end` and message events, kept apart.
    records: Option<Usage>,
    steps: Option<Usage>,
    messages: Option<Usage>,
    statuses: Option<Usage>,
}

impl Parser for KimiStreamJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if let Some((name, item)) = followed(line) {
            if name == "wire" {
                self.wire(&item);
            }
            return true;
        }
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "role") {
            "assistant" | "tool" => self.chat.message(&event, None, out),
            "meta" => {
                let kind = str_of(&event, "type");
                let error = event.get("error");
                if kind.contains("error") || kind.contains("fail") || error.is_some() {
                    let text = error
                        .map(|e| match e {
                            Value::String(s) => s.clone(),
                            other => match str_of(other, "message") {
                                "" => other.to_string(),
                                m => m.to_string(),
                            },
                        })
                        .or_else(|| {
                            ["message", "content"]
                                .iter()
                                .map(|k| str_of(&event, k))
                                .find(|s| !s.is_empty())
                                .map(str::to_string)
                        })
                        .unwrap_or_else(|| format!("kimi reported {kind}"));
                    out.push(Out::Failure(Failure::fatal(text)));
                }
            }
            "user" | "system" => {}
            _ => return false,
        }
        true
    }

    fn finish(&mut self, out: &mut Vec<Out>) {
        if let Some(usage) = self
            .records
            .or(self.steps)
            .or(self.messages)
            .or(self.statuses)
        {
            out.push(Out::Usage(usage));
        }
    }
}

impl KimiStreamJson {
    fn wire(&mut self, event: &Value) {
        if let Some(msg) = event.get("message")
            && str_of(msg, "type") == "StatusUpdate"
        {
            let usage = msg.get("payload").and_then(|p| p.get("token_usage"));
            if let Some(usage) = usage.filter(|u| u.is_object()) {
                let total = self.statuses.get_or_insert_with(Usage::default);
                for (slot, key) in [
                    (&mut total.input_tokens, "input_other"),
                    (&mut total.cached_read_tokens, "input_cache_read"),
                    (&mut total.cached_write_tokens, "input_cache_creation"),
                    (&mut total.output_tokens, "output"),
                ] {
                    if let Some(n) = count(usage.get(key)) {
                        *slot = Some(slot.unwrap_or(0) + n);
                    }
                }
            }
            return;
        }
        let (slot, usage) = match str_of(event, "type") {
            "usage.record" => (&mut self.records, event.get("usage")),
            "context.append_loop_event" => {
                let inner = event.get("event").unwrap_or(&Value::Null);
                if str_of(inner, "type") != "step.end" {
                    return;
                }
                (&mut self.steps, inner.get("usage"))
            }
            "agent.message.appended" => {
                let meta = event.get("message").and_then(|m| m.get("meta"));
                (&mut self.messages, meta.and_then(|m| m.get("usage")))
            }
            _ => return,
        };
        let Some(usage) = usage.filter(|u| u.is_object()) else {
            return;
        };
        let total = slot.get_or_insert_with(Usage::default);
        let add = |slot: &mut Option<u64>, key: &str| {
            if let Some(n) = count(usage.get(key)) {
                *slot = Some(slot.unwrap_or(0) + n);
            }
        };
        add(&mut total.input_tokens, "inputOther");
        add(&mut total.cached_read_tokens, "inputCacheRead");
        add(&mut total.cached_write_tokens, "inputCacheCreation");
        add(&mut total.output_tokens, "output");
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    /// Harbor's `_STREAM_JSON` (test_kimi_code_trajectory.py): kimi-code
    /// 2.1.1 creating hello.txt.
    const STREAM: &str = r#"{"role":"meta","type":"system.version","version":"2.1.1"}
{"role":"assistant","tool_calls":[{"type":"function","id":"call_419963a4c28a47d2bb9d7315","function":{"name":"Write","arguments":"{\"path\":\"/app/hello.txt\",\"content\":\"Hello, world!\\n\"}"}}]}
{"role":"tool","tool_call_id":"call_419963a4c28a47d2bb9d7315","content":"Wrote 14 bytes to /app/hello.txt"}
{"role":"assistant","content":"Created `/app/hello.txt` with the content:\n\n```\nHello, world!\n```"}
{"role":"meta","type":"session.resume_hint","session_id":"session_8192ee32-068a-47f4-a145-829f33b1b234","command":"kimi -r session_8192ee32-068a-47f4-a145-829f33b1b234","content":"To resume this session: kimi -r session_8192ee32-068a-47f4-a145-829f33b1b234"}"#;

    fn lines(text: &str) -> Vec<String> {
        text.lines().map(str::to_string).collect()
    }

    #[tokio::test]
    async fn the_stream_becomes_calls_results_and_the_answer() {
        let mut l = lines(STREAM);
        // The journal's usage, printed after kimi exits (two of each kind
        // of record for the same two steps; one kind is counted).
        for (input, cached, output) in [(19745, 0, 86), (126, 19744, 29)] {
            let usage = json!({"inputOther": input, "output": output, "inputCacheRead": cached, "inputCacheCreation": 0});
            l.push(
                json!({"@bf": "wire", "item": {"type": "usage.record", "usage": usage}})
                    .to_string(),
            );
            l.push(json!({"@bf": "wire", "item": {"type": "context.append_loop_event", "event": {"type": "step.end", "usage": usage}}}).to_string());
        }
        let run = run_lines("kimi-stream-json", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, ["user_message", "tool_call", "agent_message"]);
        assert!(
            matches!(&events[1], Event::ToolCall(c) if c.title == "Write /app/hello.txt" && c.status.as_str() == "completed")
        );
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.cached_read_tokens, u.output_tokens),
            (Some(19871), Some(19744), Some(115))
        );
    }

    #[tokio::test]
    async fn without_a_journal_there_is_no_usage_rather_than_zero() {
        let run = run_lines("kimi-stream-json", &lines(STREAM)).await;
        assert_eq!(run.usage, None);
    }

    #[tokio::test]
    async fn message_events_count_when_nothing_else_does() {
        let mut l = lines(STREAM);
        l.push(json!({"@bf": "wire", "item": {"type": "agent.message.appended", "message": {"meta": {"usage": {"inputOther": 5, "output": 2}}}}}).to_string());
        let run = run_lines("kimi-stream-json", &l).await;
        assert_eq!(run.usage.unwrap().output_tokens, Some(2));
    }

    /// kimi-cli 1.52's `--print --output-format stream-json`: content as
    /// parts, then its journal's StatusUpdate records.
    #[tokio::test]
    async fn kimi_cli_print_mode_and_its_journal() {
        let l = vec![
            json!({"role": "assistant", "content": [{"type": "think", "think": "Write it."}, {"type": "text", "text": "Writing."}],
                   "tool_calls": [{"type": "function", "id": "tc_1", "function": {"name": "WriteFile", "arguments": "{\"path\": \"hello.txt\"}"}}]}).to_string(),
            json!({"role": "tool", "tool_call_id": "tc_1", "content": [{"type": "text", "text": "Wrote 2 bytes"}]}).to_string(),
            json!({"role": "assistant", "content": [{"type": "text", "text": "Done."}]}).to_string(),
            json!({"@bf": "wire", "item": {"timestamp": 1.0, "message": {"type": "StatusUpdate", "payload": {"token_usage": {"input_other": 100, "output": 20, "input_cache_read": 50, "input_cache_creation": 0}}}}}).to_string(),
            json!({"@bf": "wire", "item": {"timestamp": 2.0, "message": {"type": "StatusUpdate", "payload": {"token_usage": {"input_other": 10, "output": 5, "input_cache_read": 150}}}}}).to_string(),
        ];
        let run = run_lines("kimi-stream-json", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let kinds: Vec<&str> = events_of(&run).iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_message",
                "tool_call",
                "agent_message"
            ]
        );
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.cached_read_tokens, u.output_tokens),
            (Some(110), Some(200), Some(25))
        );
    }

    #[tokio::test]
    async fn an_error_meta_line_fails_the_run() {
        let l = vec![
            r#"{"role":"meta","type":"system.version","version":"2.1.1"}"#.to_string(),
            r#"{"role":"meta","type":"error","error":{"message":"401 Invalid Authentication"}}"#
                .to_string(),
        ];
        let run = run_lines_exit("kimi-stream-json", &l, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }

    #[tokio::test]
    async fn noise_unknown_lines_and_a_truncated_tail() {
        let mut l = vec!["not json".to_string()];
        l.extend(lines(STREAM));
        l.push(r#"{"role":"future","x":1}"#.to_string());
        l.push(r#"{"role":"assistant","content":"cut"#.to_string());
        let run = run_lines("kimi-stream-json", &l).await;
        assert!(run.result.is_ok());
        assert_eq!(run.log.len(), 3, "{:?}", run.log);
    }
}
