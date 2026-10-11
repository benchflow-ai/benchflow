//! The native loops' agent options (`bench run --ak <name>=<value>`), named
//! as Harbor's agents name them, with the same types and checks as a
//! manifest's `[options.<name>]` tables.

use std::time::Duration;

use crate::manifest::{AgentOption, OptionKind, OptionRoute, OptionValue, OptionValues};

use super::{BUILTIN, COMPUTER_1, NativeLoop, TERMINUS_2};

fn option(
    name: &str,
    kind: OptionKind,
    default: Option<OptionValue>,
    (min, max): (Option<OptionValue>, Option<OptionValue>),
    description: &str,
) -> AgentOption {
    AgentOption {
        name: name.into(),
        kind,
        choices: Vec::new(),
        default,
        description: description.into(),
        min,
        max,
        route: OptionRoute::Native,
        harnesses: Vec::new(),
    }
}

fn int(n: i64) -> Option<OptionValue> {
    Some(OptionValue::Int(n))
}

fn float(x: f64) -> Option<OptionValue> {
    Some(OptionValue::Float(x))
}

fn temperature() -> AgentOption {
    option(
        "temperature",
        OptionKind::Float,
        None,
        (float(0.0), float(2.0)),
        "Sampling temperature; unset leaves the provider's default.",
    )
}

fn max_output_tokens() -> AgentOption {
    option(
        "max_output_tokens",
        OptionKind::Int,
        None,
        (int(1), None),
        "Most tokens per model response.",
    )
}

/// The options the native agent `name` (from [`super::find`]) takes.
pub fn declared(name: &str) -> Vec<AgentOption> {
    match name {
        TERMINUS_2 => vec![
            option(
                "max_turns",
                OptionKind::Int,
                int(1_000_000),
                (int(1), None),
                "Most model turns.",
            ),
            AgentOption {
                choices: vec!["json".into(), "xml".into()],
                ..option(
                    "parser_name",
                    OptionKind::Enum,
                    Some(OptionValue::String("json".into())),
                    (None, None),
                    "The response format the model writes: json or xml.",
                )
            },
            temperature(),
            max_output_tokens(),
            option(
                "enable_summarize",
                OptionKind::Bool,
                Some(OptionValue::Bool(true)),
                (None, None),
                "Summarize the conversation when the context fills up.",
            ),
            option(
                "proactive_summarization_threshold",
                OptionKind::Int,
                int(8000),
                (int(0), None),
                "Summarize before a turn when fewer tokens than this are free; 0 turns it off.",
            ),
            option(
                "tmux_pane_width",
                OptionKind::Int,
                int(160),
                (int(20), int(1000)),
                "Starting tmux pane width, in columns.",
            ),
            option(
                "tmux_pane_height",
                OptionKind::Int,
                int(40),
                (int(10), int(1000)),
                "Starting tmux pane height, in rows.",
            ),
        ],
        BUILTIN => vec![
            temperature(),
            max_output_tokens(),
            option(
                "bash_timeout_sec",
                OptionKind::Int,
                int(600),
                (int(1), None),
                "Default and longest time one bash call may take, in seconds.",
            ),
        ],
        COMPUTER_1 => vec![
            option(
                "max_turns",
                OptionKind::Int,
                int(50),
                (int(1), None),
                "Most model turns per prompt.",
            ),
            temperature(),
            max_output_tokens(),
        ],
        _ => Vec::new(),
    }
}

impl NativeLoop {
    /// Apply options checked against [`declared`].
    pub fn apply_options(&mut self, values: &OptionValues) {
        let int = |v: &OptionValue| match v {
            OptionValue::Int(i) => (*i).max(0) as u64,
            _ => 0,
        };
        let float = |v: &OptionValue| match v {
            OptionValue::Float(x) => Some(*x),
            OptionValue::Int(i) => Some(*i as f64),
            _ => None,
        };
        let tokens = |v: &OptionValue| Some(int(v).min(u32::MAX as u64) as u32);
        for (name, v) in values {
            match (&mut *self, name.as_str()) {
                (Self::Terminus(c), "max_turns") => c.max_episodes = int(v),
                (Self::Terminus(c), "parser_name") => {
                    if let OptionValue::String(s) = v {
                        c.parser = if s == "xml" {
                            super::terminus::ParserKind::Xml
                        } else {
                            super::terminus::ParserKind::Json
                        };
                    }
                }
                (Self::Terminus(c), "temperature") => c.temperature = float(v),
                (Self::Terminus(c), "max_output_tokens") => c.max_output_tokens = tokens(v),
                (Self::Terminus(c), "enable_summarize") => {
                    c.enable_summarize = v == &OptionValue::Bool(true)
                }
                (Self::Terminus(c), "proactive_summarization_threshold") => {
                    c.proactive_summarization_threshold = int(v)
                }
                (Self::Terminus(c), "tmux_pane_width") => c.pane_width = int(v) as u32,
                (Self::Terminus(c), "tmux_pane_height") => c.pane_height = int(v) as u32,
                (Self::Builtin(c), "temperature") => c.temperature = float(v),
                (Self::Builtin(c), "max_output_tokens") => c.max_output_tokens = tokens(v),
                (Self::Builtin(c), "bash_timeout_sec") => {
                    c.bash_timeout = Duration::from_secs(int(v))
                }
                (Self::Computer(c), "max_turns") => {
                    c.max_turns = int(v).min(u32::MAX as u64) as u32
                }
                (Self::Computer(c), "temperature") => c.temperature = float(v),
                (Self::Computer(c), "max_output_tokens") => c.max_output_tokens = tokens(v),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::options::resolve;

    #[test]
    fn terminus_options_set_its_loop() {
        let given = [
            ("max_turns".to_string(), "12".to_string()),
            ("parser_name".to_string(), "xml".to_string()),
            ("temperature".to_string(), "0.5".to_string()),
        ];
        let values = resolve("terminus-2", &declared(TERMINUS_2), &given, None).unwrap();
        let mut native = NativeLoop::for_agent(TERMINUS_2).unwrap();
        native.apply_options(&values);
        let NativeLoop::Terminus(c) = native else {
            unreachable!()
        };
        assert_eq!(c.max_episodes, 12);
        assert_eq!(c.parser, super::super::terminus::ParserKind::Xml);
        assert_eq!(c.temperature, Some(0.5));
        let e = resolve(
            "terminus-2",
            &declared(TERMINUS_2),
            &[("temperature".into(), "3".into())],
            None,
        )
        .unwrap_err();
        assert!(e.message.contains("a number from 0 to 2"), "{}", e.message);
        let e = resolve(
            "terminus-2",
            &declared(TERMINUS_2),
            &[("parser".into(), "xml".into())],
            None,
        )
        .unwrap_err();
        assert!(e.next.contains("did you mean `parser_name`?"), "{}", e.next);
    }
}
