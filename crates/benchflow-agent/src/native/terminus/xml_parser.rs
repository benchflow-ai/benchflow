//! Terminus-2's XML reply parser.
//!
//! Ported from Harbor's `terminus_xml_plain_parser.py`
//! (https://github.com/laude-institute/harbor, Apache License 2.0): the
//! same sections, checks, warnings and wording, and the salvage of a reply
//! cut off at the output limit. Where Harbor iterates a Python set (missing
//! sections, unknown tags), this port uses the sections' and tags' order.

use std::sync::LazyLock;

use regex::Regex;

use super::{ParseResult, ParsedCommand, combine_warnings, join_warnings};

const REQUIRED_SECTIONS: [&str; 3] = ["analysis", "plan", "commands"];
const ALL_SECTIONS: [&str; 4] = ["analysis", "plan", "commands", "task_complete"];

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("valid regex")
}

static KEYSTROKES: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?s)<keystrokes([^>]*)>(.*?)</keystrokes>"));
static DURATION: LazyLock<Regex> = LazyLock::new(|| re(r#"duration\s*=\s*["']([^"']*)["']"#));
static UNQUOTED: LazyLock<Regex> = LazyLock::new(|| re(r#"(\w+)\s*=\s*([^"'\s>]+)"#));
static SINGLE_QUOTED: LazyLock<Regex> = LazyLock::new(|| re(r"(\w+)\s*=\s*'([^']*)'"));
static ATTRIBUTE: LazyLock<Regex> = LazyLock::new(|| re(r"(\w+)\s*="));
static TASK_COMPLETE_TRUE: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)<task_complete>\s*true\s*</task_complete>"));

/// Parse a reply, trying Harbor's auto-fix when it does not parse.
pub fn parse_response(response: &str) -> ParseResult {
    let result = try_parse(response);
    if result.error.is_empty() {
        return result;
    }
    // Harbor's one fix; its trigger ("Missing </response> closing tag") is
    // an error the parser never reports, so it never applies there either.
    if result.error.contains("Missing </response> closing tag") {
        let corrected = format!("{}\n</response>", response.trim_end());
        let mut r = try_parse(&corrected);
        if r.error.is_empty() {
            let auto = "AUTO-CORRECTED: Missing </response> tag was automatically inserted - please fix this in future responses";
            r.warning = combine_warnings(auto, &r.warning);
            return r;
        }
    }
    result
}

fn try_parse(response: &str) -> ParseResult {
    let mut warnings: Vec<String> = Vec::new();
    check_extra_text(response, &mut warnings);
    let content = extract_response_content(response);
    if content.is_empty() {
        return ParseResult {
            error: "No <response> tag found".into(),
            warning: join_warnings(&warnings),
            ..ParseResult::default()
        };
    }
    let is_complete = TASK_COMPLETE_TRUE.is_match(content);
    let sections = extract_sections(content, &mut warnings);
    let get = |k: &str| {
        sections
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.clone())
    };
    let analysis = get("analysis").unwrap_or_default();
    let plan = get("plan").unwrap_or_default();
    let ok = |commands, warnings: &[String], is_complete| ParseResult {
        commands,
        is_task_complete: is_complete,
        error: String::new(),
        warning: join_warnings(warnings),
        analysis: analysis.clone(),
        plan: plan.clone(),
    };
    let commands_content = get("commands");
    match commands_content.as_deref() {
        Some("") => {
            if !is_complete {
                warnings.push("Commands section is empty; not taking any action. If you want to wait a specific amount of time please use `sleep`, but if you're waiting for a command to finish then continue to wait.".into());
            }
            ok(Vec::new(), &warnings, is_complete)
        }
        None => {
            if is_complete {
                ok(Vec::new(), &warnings, true)
            } else {
                ParseResult {
                    error: "Missing <commands> section".into(),
                    warning: join_warnings(&warnings),
                    analysis: analysis.clone(),
                    plan: plan.clone(),
                    ..ParseResult::default()
                }
            }
        }
        Some(commands) => {
            let parsed = parse_xml_commands(commands, &mut warnings);
            ok(parsed, &warnings, is_complete)
        }
    }
}

fn check_extra_text(response: &str, warnings: &mut Vec<String>) {
    let Some(start) = response.find("<response>") else {
        return;
    };
    if !response[..start].trim().is_empty() {
        warnings.push("Extra text detected before <response> tag".into());
    }
    if let Some(end) = response.find("</response>") {
        let after = &response[end + "</response>".len()..];
        if !after.trim().is_empty() {
            warnings.push("Extra text detected after </response> tag".into());
            let total = response.matches("<response>").count();
            if total > 1 {
                warnings.push(format!(
                    "IMPORTANT: Only issue one <response> block at a time. You issued {total} and only the first was executed."
                ));
            }
        }
    }
}

fn extract_response_content(response: &str) -> &str {
    let Some(start) = response.find("<response>") else {
        return "";
    };
    let inner = start + "<response>".len();
    match response[start..].find("</response>") {
        Some(end) => response[inner..start + end].trim(),
        None => response[inner..].trim(),
    }
}

/// The sections found, in the order of [`ALL_SECTIONS`], with their text.
fn extract_sections(content: &str, warnings: &mut Vec<String>) -> Vec<(&'static str, String)> {
    let mut sections = Vec::new();
    for name in ALL_SECTIONS {
        let full = re(&format!(r"(?s)<{name}>(.*?)</{name}>"));
        if let Some(c) = full.captures(content) {
            sections.push((name, c[1].trim().to_string()));
            continue;
        }
        let self_closing = re(&format!(r"<{name}\s*/>"));
        if self_closing.is_match(content) || content.contains(&format!("<{name}></{name}>")) {
            sections.push((name, String::new()));
        }
    }
    for name in REQUIRED_SECTIONS {
        if !sections.iter().any(|(n, _)| *n == name) {
            warnings.push(format!("Missing <{name}> section"));
        }
    }
    let mut unexpected: Vec<String> = Vec::new();
    for tag in find_top_level_tags(content) {
        if !ALL_SECTIONS.contains(&tag.as_str()) && !unexpected.contains(&tag) {
            unexpected.push(tag);
        }
    }
    for tag in unexpected {
        warnings.push(format!(
            "Unknown tag found: <{tag}>, expected analysis/plan/commands/task_complete"
        ));
    }
    for name in ALL_SECTIONS {
        let count = re(&format!(r"<{name}(?:\s|>|/>)"))
            .find_iter(content)
            .count();
        if count > 1 {
            if name == "commands" {
                warnings.push(format!(
                    "IMPORTANT: Only issue one <commands> block at a time. You issued {count} and only the first was executed."
                ));
            } else {
                warnings.push(format!("Multiple <{name}> sections found"));
            }
        }
    }
    check_section_order(content, warnings);
    sections
}

fn parse_xml_commands(xml: &str, warnings: &mut Vec<String>) -> Vec<ParsedCommand> {
    let matches: Vec<(String, String)> = KEYSTROKES
        .captures_iter(xml)
        .map(|c| (c[1].to_string(), c[2].to_string()))
        .collect();
    let mut commands = Vec::new();
    for (i, (attributes, keystrokes)) in matches.iter().enumerate() {
        let n = i + 1;
        check_attribute_issues(attributes, n, warnings);
        let mut duration = 1.0;
        match DURATION.captures(attributes) {
            Some(c) => match c[1].trim().parse::<f64>() {
                Ok(d) => duration = d,
                Err(_) => warnings.push(format!(
                    "Command {n}: Invalid duration value '{}', using default 1.0",
                    &c[1]
                )),
            },
            None => warnings.push(format!(
                "Command {n}: Missing duration attribute, using default 1.0"
            )),
        }
        if i < matches.len() - 1 && !keystrokes.ends_with('\n') {
            warnings.push(format!(
                "Command {n} should end with newline when followed by another command. Otherwise the two commands will be concatenated together on the same line."
            ));
        }
        commands.push(ParsedCommand {
            keystrokes: keystrokes.clone(),
            duration,
        });
    }
    for (entity, c) in [
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&amp;", "&"),
        ("&quot;", "\""),
        ("&apos;", "'"),
    ] {
        if xml.contains(entity) {
            warnings.push(format!(
                "Warning: {entity} is read verbatim and not converted to {c}. NEVER USE {entity}, unless you want these exact characters to appear directly in the output."
            ));
        }
    }
    if xml.contains("\\r\\n") {
        warnings.push(
            "Warning: \\r\\n line endings are not necessary - use \\n instead for simpler output"
                .into(),
        );
    }
    commands
}

fn find_top_level_tags(content: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut depth: i64 = 0;
    let mut i = 0;
    let bytes = content.as_bytes();
    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        let Some(rel) = content[i..].find('>') else {
            break;
        };
        let end = i + rel;
        let tag = &content[i + 1..end];
        if tag.starts_with('!') || tag.starts_with('?') {
            i = end + 1;
            continue;
        }
        if tag.starts_with('/') {
            depth -= 1;
            i = end + 1;
            continue;
        }
        let self_closing = tag.ends_with('/');
        let mut name = if tag.contains(' ') {
            tag.split_whitespace().next().unwrap_or_default()
        } else {
            tag
        };
        if let Some(stripped) = name.strip_suffix('/') {
            name = stripped;
        }
        if depth == 0 {
            tags.push(name.to_string());
        }
        if !self_closing {
            depth += 1;
        }
        i = end + 1;
    }
    tags
}

fn check_section_order(content: &str, warnings: &mut Vec<String>) {
    let mut positions: Vec<(&str, usize)> = Vec::new();
    for name in REQUIRED_SECTIONS {
        if let Some(m) = re(&format!(r"<{name}(?:\s|>|/>)")).find(content) {
            positions.push((name, m.start()));
        }
    }
    if positions.len() < 2 {
        return;
    }
    let expected: Vec<&str> = positions.iter().map(|(n, _)| *n).collect();
    let mut sorted = positions.clone();
    sorted.sort_by_key(|(_, p)| *p);
    let actual: Vec<&str> = sorted.iter().map(|(n, _)| *n).collect();
    if actual != expected {
        warnings.push(format!(
            "Sections appear in wrong order. Found: {}, expected: {}",
            actual.join(" → "),
            expected.join(" → ")
        ));
    }
}

fn check_attribute_issues(attributes: &str, n: usize, warnings: &mut Vec<String>) {
    for c in UNQUOTED.captures_iter(attributes) {
        warnings.push(format!(
            "Command {n}: Attribute '{}' value should be quoted: {}=\"{}\"",
            &c[1], &c[1], &c[2]
        ));
    }
    for c in SINGLE_QUOTED.captures_iter(attributes) {
        warnings.push(format!(
            "Command {n}: Use double quotes for attribute '{}': {}=\"{}\"",
            &c[1], &c[1], &c[2]
        ));
    }
    for c in ATTRIBUTE.captures_iter(attributes) {
        if &c[1] != "duration" {
            warnings.push(format!(
                "Command {n}: Unknown attribute '{}' - known attributes are: duration",
                &c[1]
            ));
        }
    }
}

/// A reply cut off at the output limit, trimmed to its first complete
/// `<response>` when that parses without errors or a second block.
/// Returns the salvaged reply and whether several blocks were seen.
pub fn salvage_truncated_response(truncated: &str) -> (Option<String>, bool) {
    let Some(commands_end) = truncated.find("</commands>") else {
        return (None, false);
    };
    let Some(rel) = truncated[commands_end..].find("</response>") else {
        return (None, false);
    };
    let clean = &truncated[..commands_end + rel + "</response>".len()];
    let parsed = parse_response(clean);
    let lower = parsed.warning.to_lowercase();
    let multiple = lower.contains("only issue one") && lower.contains("block at a time");
    if parsed.error.is_empty() && !multiple {
        (Some(clean.to_string()), false)
    } else {
        (None, multiple)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_complete_reply_parses_verbatim_keystrokes() {
        let r = parse_response(
            "<response>\n<analysis>a</analysis>\n<plan>p</plan>\n<commands>\n<keystrokes duration=\"0.1\">ls -la && echo '<x>'\n</keystrokes>\n<keystrokes duration=\"1.0\">cd project\n</keystrokes>\n</commands>\n<task_complete>true</task_complete>\n</response>",
        );
        assert_eq!(r.error, "");
        assert_eq!(r.warning, "");
        assert!(r.is_task_complete);
        assert_eq!(r.commands[0].keystrokes, "ls -la && echo '<x>'\n");
        assert_eq!(r.commands[0].duration, 0.1);
        assert_eq!(r.commands.len(), 2);
    }

    #[test]
    fn a_missing_response_or_commands_is_an_error() {
        assert_eq!(parse_response("just text").error, "No <response> tag found");
        let r = parse_response("<response><analysis>a</analysis><plan>p</plan></response>");
        assert_eq!(r.error, "Missing <commands> section");
        assert!(r.warning.contains("Missing <commands> section"));
        let r = parse_response(
            "<response><analysis>a</analysis><plan>p</plan><task_complete>true</task_complete></response>",
        );
        assert_eq!(r.error, "");
        assert!(r.is_task_complete);
    }

    #[test]
    fn empty_commands_warn_unless_complete() {
        let r = parse_response(
            "<response><analysis>a</analysis><plan>p</plan><commands></commands></response>",
        );
        assert!(r.warning.contains("Commands section is empty"));
        let r = parse_response(
            "<response><analysis>a</analysis><plan>p</plan><commands/><task_complete>TRUE</task_complete></response>",
        );
        assert!(!r.warning.contains("Commands section is empty"));
        assert!(r.is_task_complete);
    }

    #[test]
    fn attributes_entities_and_extra_blocks_warn() {
        let r = parse_response(
            "x<response><analysis>a</analysis><plan>p</plan><commands><keystrokes duration='2' speed=fast>a &amp; b</keystrokes><keystrokes>c\n</keystrokes></commands><extra/></response><response></response>",
        );
        let w = &r.warning;
        assert!(
            w.contains("Extra text detected before <response> tag"),
            "{w}"
        );
        assert!(
            w.contains("IMPORTANT: Only issue one <response> block at a time. You issued 2"),
            "{w}"
        );
        assert!(
            w.contains("Command 1: Use double quotes for attribute 'duration': duration=\"2\""),
            "{w}"
        );
        assert!(
            w.contains("Command 1: Attribute 'speed' value should be quoted: speed=\"fast\""),
            "{w}"
        );
        assert!(w.contains("Command 1: Unknown attribute 'speed'"), "{w}");
        assert!(w.contains("Command 1 should end with newline"), "{w}");
        assert!(
            w.contains("Command 2: Missing duration attribute, using default 1.0"),
            "{w}"
        );
        assert!(w.contains("Warning: &amp; is read verbatim"), "{w}");
        assert!(w.contains("Unknown tag found: <extra>"), "{w}");
        assert_eq!(r.commands[0].duration, 2.0);
        assert_eq!(r.commands[0].keystrokes, "a &amp; b");
    }

    #[test]
    fn sections_out_of_order_warn() {
        let r = parse_response(
            "<response><plan>p</plan><analysis>a</analysis><commands></commands></response>",
        );
        assert!(r.warning.contains(
            "Sections appear in wrong order. Found: plan → analysis → commands, expected: analysis → plan → commands"
        ), "{}", r.warning);
    }

    #[test]
    fn a_cut_off_reply_is_salvaged_up_to_its_response() {
        let text = "<response><analysis>a</analysis><plan>p</plan><commands><keystrokes duration=\"1\">ls\n</keystrokes></commands></response>\n<response><analysis>more";
        let (salvaged, multiple) = salvage_truncated_response(text);
        assert!(!multiple);
        assert!(salvaged.unwrap().ends_with("</commands></response>"));
        assert_eq!(salvage_truncated_response("<response><analysis>a").0, None);
    }
}
