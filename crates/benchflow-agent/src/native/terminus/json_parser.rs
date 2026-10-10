//! Terminus-2's JSON reply parser.
//!
//! Ported from Harbor's `terminus_json_plain_parser.py`
//! (https://github.com/laude-institute/harbor, Apache License 2.0). The
//! checks, the warnings and their wording, and the two auto-fixes are
//! Harbor's, since the model reads them as feedback. JSON syntax errors
//! come from `serde_json` rather than Python's `json`, so their wording
//! differs.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use super::{ParseResult, ParsedCommand, combine_warnings, join_warnings, py_repr};

/// An auto-fix: the corrected reply, when it applies to the error.
type Fix = fn(&str, &str) -> Option<String>;

const REQUIRED_FIELDS: [&str; 3] = ["analysis", "plan", "commands"];

/// Parse a reply, trying Harbor's auto-fixes when it does not parse.
pub fn parse_response(response: &str) -> ParseResult {
    let result = try_parse(response);
    if result.error.is_empty() {
        return result;
    }
    let fixes: [(&str, Fix); 2] = [
        (
            "Fixed incomplete JSON by adding missing closing brace",
            fix_incomplete_json,
        ),
        ("Extracted JSON from mixed content", fix_mixed_content),
    ];
    for (name, fix) in fixes {
        if let Some(corrected) = fix(response, &result.error) {
            let mut corrected_result = try_parse(&corrected);
            if corrected_result.error.is_empty() {
                let auto = format!("AUTO-CORRECTED: {name} - please fix this in future responses");
                corrected_result.warning = combine_warnings(&auto, &corrected_result.warning);
                return corrected_result;
            }
        }
    }
    result
}

fn failure(error: String, warnings: &[String]) -> ParseResult {
    ParseResult {
        error,
        warning: join_warnings(warnings),
        ..ParseResult::default()
    }
}

fn try_parse(response: &str) -> ParseResult {
    let mut warnings: Vec<String> = Vec::new();
    let (json_content, extra) = extract_json_content(response);
    warnings.extend(extra);
    if json_content.is_empty() {
        return failure("No valid JSON found in response".into(), &warnings);
    }
    let parsed: Value = match serde_json::from_str(json_content) {
        Ok(v) => v,
        Err(e) => {
            let mut msg = format!("Invalid JSON: {e}");
            if json_content.chars().count() < 200 {
                msg.push_str(&format!(" | Content: {}", py_repr(json_content)));
            } else {
                let preview: String = json_content.chars().take(100).collect();
                msg.push_str(&format!(" | Content preview: {}...", py_repr(&preview)));
            }
            return failure(msg, &warnings);
        }
    };
    let Some(data) = parsed.as_object() else {
        return failure("Response must be a JSON object".into(), &warnings);
    };
    if let Some(error) = validate_structure(data, json_content, &mut warnings) {
        return failure(error, &warnings);
    }
    let is_complete = match data.get("task_complete") {
        Some(Value::String(s)) => matches!(s.to_lowercase().as_str(), "true" | "1" | "yes"),
        Some(v) => truthy(v),
        None => false,
    };
    let text = |k: &str| match data.get(k) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    let (analysis, plan) = (text("analysis"), text("plan"));
    let commands_data = data
        .get("commands")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    match parse_commands(&commands_data, &mut warnings) {
        Ok(commands) => ParseResult {
            commands,
            is_task_complete: is_complete,
            error: String::new(),
            warning: join_warnings(&warnings),
            analysis,
            plan,
        },
        Err(parse_error) => {
            if is_complete {
                // A finished task's bad commands are only a warning.
                warnings.push(parse_error);
                ParseResult {
                    commands: Vec::new(),
                    is_task_complete: true,
                    error: String::new(),
                    warning: join_warnings(&warnings),
                    analysis,
                    plan,
                }
            } else {
                ParseResult {
                    error: parse_error,
                    warning: join_warnings(&warnings),
                    analysis,
                    plan,
                    ..ParseResult::default()
                }
            }
        }
    }
}

/// Python truthiness, which `if is_task_complete:` applies.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The first balanced `{...}` object, and warnings about text around it.
fn extract_json_content(response: &str) -> (&str, Vec<String>) {
    let mut start: Option<usize> = None;
    let mut end: Option<usize> = None;
    let mut depth: i64 = 0;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, c) in response.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if c == '\\' {
            escape_next = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if c == '{' {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 && start.is_some() {
                    end = Some(i + 1);
                    break;
                }
            }
        }
    }
    let (Some(start), Some(end)) = (start, end) else {
        return ("", vec!["No valid JSON object found".into()]);
    };
    let mut warnings = Vec::new();
    if !response[..start].trim().is_empty() {
        warnings.push("Extra text detected before JSON object".into());
    }
    if !response[end..].trim().is_empty() {
        warnings.push("Extra text detected after JSON object".into());
    }
    (&response[start..end], warnings)
}

fn validate_structure(
    data: &Map<String, Value>,
    json_content: &str,
    warnings: &mut Vec<String>,
) -> Option<String> {
    let missing: Vec<&str> = REQUIRED_FIELDS
        .iter()
        .copied()
        .filter(|f| !data.contains_key(*f))
        .collect();
    if !missing.is_empty() {
        return Some(format!("Missing required fields: {}", missing.join(", ")));
    }
    if !matches!(data.get("analysis"), Some(Value::String(_))) {
        warnings.push("Field 'analysis' should be a string".into());
    }
    if !matches!(data.get("plan"), Some(Value::String(_))) {
        warnings.push("Field 'plan' should be a string".into());
    }
    if !matches!(data.get("commands"), Some(Value::Array(_))) {
        return Some("Field 'commands' must be an array".into());
    }
    check_field_order(json_content, warnings);
    match data.get("task_complete") {
        None | Some(Value::Null) | Some(Value::Bool(_)) | Some(Value::String(_)) => {}
        Some(_) => warnings.push("Field 'task_complete' should be a boolean or string".into()),
    }
    None
}

fn parse_commands(
    commands: &[Value],
    warnings: &mut Vec<String>,
) -> Result<Vec<ParsedCommand>, String> {
    let mut out = Vec::new();
    for (i, cmd) in commands.iter().enumerate() {
        let n = i + 1;
        let Some(cmd) = cmd.as_object() else {
            return Err(format!("Command {n} must be an object"));
        };
        let Some(keystrokes) = cmd.get("keystrokes") else {
            return Err(format!("Command {n} missing required 'keystrokes' field"));
        };
        let Some(keystrokes) = keystrokes.as_str() else {
            return Err(format!("Command {n} 'keystrokes' must be a string"));
        };
        let duration = match cmd.get("duration") {
            // Python's isinstance(x, (int, float)) also takes booleans.
            Some(Value::Number(d)) => d.as_f64().unwrap_or(1.0),
            Some(Value::Bool(b)) => f64::from(u8::from(*b)),
            Some(_) => {
                warnings.push(format!(
                    "Command {n}: Invalid duration value, using default 1.0"
                ));
                1.0
            }
            None => {
                warnings.push(format!(
                    "Command {n}: Missing duration field, using default 1.0"
                ));
                1.0
            }
        };
        let unknown: Vec<&str> = cmd
            .keys()
            .map(String::as_str)
            .filter(|k| *k != "keystrokes" && *k != "duration")
            .collect();
        if !unknown.is_empty() {
            warnings.push(format!(
                "Command {n}: Unknown fields: {}",
                unknown.join(", ")
            ));
        }
        if i < commands.len() - 1 && !keystrokes.is_empty() && !keystrokes.ends_with('\n') {
            warnings.push(format!(
                "Command {n} should end with newline when followed by another command. Otherwise the two commands will be concatenated together on the same line."
            ));
        }
        out.push(ParsedCommand {
            keystrokes: keystrokes.to_string(),
            duration,
        });
    }
    Ok(out)
}

fn fix_incomplete_json(response: &str, error: &str) -> Option<String> {
    let applies = [
        "Invalid JSON",
        "Expecting",
        "Unterminated",
        "No valid JSON found",
    ]
    .iter()
    .any(|p| error.contains(p));
    if !applies {
        return None;
    }
    let missing = response.matches('{').count() as i64 - response.matches('}').count() as i64;
    (missing > 0).then(|| format!("{response}{}", "}".repeat(missing as usize)))
}

static MIXED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}").expect("valid regex"));

fn fix_mixed_content(response: &str, _error: &str) -> Option<String> {
    MIXED
        .find_iter(response)
        .map(|m| m.as_str())
        .find(|m| serde_json::from_str::<Value>(m).is_ok())
        .map(str::to_string)
}

fn check_field_order(response: &str, warnings: &mut Vec<String>) {
    let mut positions: Vec<(&str, usize)> = Vec::new();
    for field in REQUIRED_FIELDS {
        let re = Regex::new(&format!(r#""({field})"\s*:"#)).expect("valid regex");
        if let Some(m) = re.find(response) {
            positions.push((field, m.start()));
        }
    }
    if positions.len() < 2 {
        return;
    }
    let expected: Vec<&str> = positions.iter().map(|(f, _)| *f).collect();
    let mut sorted = positions.clone();
    sorted.sort_by_key(|(_, p)| *p);
    let actual: Vec<&str> = sorted.iter().map(|(f, _)| *f).collect();
    if actual != expected {
        warnings.push(format!(
            "Fields appear in wrong order. Found: {}, expected: {}",
            actual.join(" → "),
            expected.join(" → ")
        ));
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn response(commands: Value) -> String {
        json!({"analysis": "test", "plan": "test", "commands": commands}).to_string()
    }

    // Harbor's tests/unit/agents/terminus_2/test_terminus_json_plain_parser.py

    #[test]
    fn empty_keystrokes_no_warning() {
        let r = parse_response(&response(json!([
            {"keystrokes": "", "duration": 10.0},
            {"keystrokes": "ls\n", "duration": 1.0}
        ])));
        assert_eq!(r.error, "");
        assert!(!r.warning.contains("should end with newline"));
    }

    #[test]
    fn non_empty_keystrokes_without_newline_warns() {
        let r = parse_response(&response(json!([
            {"keystrokes": "ls", "duration": 1.0},
            {"keystrokes": "pwd\n", "duration": 1.0}
        ])));
        assert!(r.warning.contains("should end with newline"));
    }

    #[test]
    fn non_empty_keystrokes_with_newline_no_warning() {
        let r = parse_response(&response(json!([
            {"keystrokes": "ls\n", "duration": 1.0},
            {"keystrokes": "pwd\n", "duration": 1.0}
        ])));
        assert!(!r.warning.contains("should end with newline"));
    }

    #[test]
    fn last_command_empty_keystrokes_no_warning() {
        let r = parse_response(&response(json!([
            {"keystrokes": "ls\n", "duration": 1.0},
            {"keystrokes": "", "duration": 5.0}
        ])));
        assert!(!r.warning.contains("should end with newline"));
    }

    #[test]
    fn multiple_empty_keystrokes_between_commands() {
        let r = parse_response(&response(json!([
            {"keystrokes": "ls\n", "duration": 1.0},
            {"keystrokes": "", "duration": 2.0},
            {"keystrokes": "", "duration": 3.0},
            {"keystrokes": "pwd\n", "duration": 1.0}
        ])));
        assert!(!r.warning.contains("should end with newline"));
    }

    // Further cases from the parser's own branches.

    #[test]
    fn a_complete_reply_parses() {
        let r = parse_response(
            r#"{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}], "task_complete": true}"#,
        );
        assert_eq!(r.error, "");
        assert_eq!(r.warning, "");
        assert!(r.is_task_complete);
        assert_eq!(
            r.commands,
            vec![ParsedCommand {
                keystrokes: "ls -la\n".into(),
                duration: 0.1
            }]
        );
        assert_eq!((r.analysis.as_str(), r.plan.as_str()), ("a", "p"));
    }

    #[test]
    fn text_around_the_object_is_a_warning() {
        let r = parse_response(r#"Sure! {"analysis": "a", "plan": "p", "commands": []} done"#);
        assert_eq!(r.error, "");
        assert_eq!(
            r.warning,
            "- Extra text detected before JSON object\n- Extra text detected after JSON object"
        );
    }

    #[test]
    fn missing_fields_are_an_error() {
        let r = parse_response(r#"{"analysis": "a"}"#);
        assert_eq!(r.error, "Missing required fields: plan, commands");
    }

    #[test]
    fn no_json_is_an_error() {
        let r = parse_response("I will run ls");
        assert_eq!(r.error, "No valid JSON found in response");
        assert_eq!(r.warning, "- No valid JSON object found");
    }

    #[test]
    fn a_missing_closing_brace_is_auto_corrected() {
        let r = parse_response(
            r#"{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\n", "duration": 1}]"#,
        );
        assert_eq!(r.error, "", "{r:?}");
        assert!(r.warning.starts_with(
            "- AUTO-CORRECTED: Fixed incomplete JSON by adding missing closing brace - please fix this in future responses"
        ), "{}", r.warning);
        assert_eq!(r.commands.len(), 1);
    }

    #[test]
    fn task_complete_strings_and_bad_commands() {
        let r = parse_response(
            r#"{"analysis": "a", "plan": "p", "commands": [], "task_complete": "yes"}"#,
        );
        assert!(r.is_task_complete);
        let r = parse_response(
            r#"{"analysis": "a", "plan": "p", "commands": ["ls"], "task_complete": true}"#,
        );
        assert_eq!(r.error, "");
        assert!(r.is_task_complete);
        assert!(r.warning.contains("Command 1 must be an object"));
        let r = parse_response(r#"{"analysis": "a", "plan": "p", "commands": ["ls"]}"#);
        assert_eq!(r.error, "Command 1 must be an object");
    }

    #[test]
    fn durations_default_with_a_warning_and_order_is_checked() {
        let r = parse_response(
            r#"{"plan": "p", "analysis": "a", "commands": [{"keystrokes": "ls\n", "extra": 1}]}"#,
        );
        assert_eq!(r.commands[0].duration, 1.0);
        assert!(
            r.warning
                .contains("Command 1: Missing duration field, using default 1.0")
        );
        assert!(r.warning.contains("Command 1: Unknown fields: extra"));
        assert!(r.warning.contains(
            "Fields appear in wrong order. Found: plan → analysis → commands, expected: analysis → plan → commands"
        ), "{}", r.warning);
    }

    #[test]
    fn invalid_json_reports_the_content() {
        let r = parse_response(r#"{"analysis": "a", "plan": "p", "commands": [}"#);
        assert!(r.error.starts_with("Invalid JSON: "), "{}", r.error);
        assert!(r.error.contains("| Content: '{"), "{}", r.error);
    }
}
