//! Private keys an agent reads in parts, through the real trainer path:
//! ACP `tool_call` events, `write_rollout` (`to_verifiers_record`,
//! `to_adp_record`), then `write_job`'s job-level `verifiers.jsonl` and
//! `adp.jsonl` (decision G-1).
//!
//! A key read with `head`, `sed -n` and `tail` leaves its middle part in a
//! string with no marker. Every key here is made up at test time in the
//! armoured shape of a real one (its label, its body's length and its line
//! width as `openssl` and `ssh-keygen` write them), so no key is in the
//! repository. "Body lines" are the key's lines of 16 characters or more;
//! shorter remainders are a known limit (decision G-3).

use std::path::{Path, PathBuf};

use benchflow_agent::trajectory::{
    AdpMeta, Redactor, VerifiersMeta, adp_line, parse_jsonl, to_adp_record, to_jsonl,
    to_python_json, to_verifiers_record, verifiers_line,
};
use benchflow_run::trainer::{
    ADP, ATIF, JOB_ADP, JOB_VERIFIERS, RolloutInfo, VERIFIERS, read_events, write_job,
    write_rollout,
};
use serde_json::{Map, Value, json};

/// A seeded xorshift generator: every run builds the same keys and records.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(rng: &mut Rng, n: usize) -> String {
    (0..n).map(|_| B64[rng.below(64)] as char).collect()
}

/// A key's armoured shape.
struct Shape {
    name: &'static str,
    label: &'static str,
    /// The body's base64 length.
    chars: usize,
    /// Characters per body line (64 for PEM, 70 for OpenSSH).
    width: usize,
}

const fn shape(name: &'static str, label: &'static str, chars: usize, width: usize) -> Shape {
    Shape {
        name,
        label,
        chars,
        width,
    }
}

/// RSA 1024 to 4096, P-256, P-521 and Ed25519, each as PKCS#1 or SEC1
/// (where it has one), PKCS#8 and OpenSSH, sized like keys from `openssl
/// genpkey` and `ssh-keygen`.
const SHAPES: &[Shape] = &[
    shape("rsa-1024 pkcs1", "RSA PRIVATE KEY", 812, 64),
    shape("rsa-1024 pkcs8", "PRIVATE KEY", 848, 64),
    shape("rsa-1024 openssh", "OPENSSH PRIVATE KEY", 948, 70),
    shape("rsa-2048 pkcs1", "RSA PRIVATE KEY", 1588, 64),
    shape("rsa-2048 pkcs8", "PRIVATE KEY", 1624, 64),
    shape("rsa-2048 openssh", "OPENSSH PRIVATE KEY", 1716, 70),
    shape("rsa-3072 pkcs1", "RSA PRIVATE KEY", 2356, 64),
    shape("rsa-3072 pkcs8", "PRIVATE KEY", 2392, 64),
    shape("rsa-3072 openssh", "OPENSSH PRIVATE KEY", 2484, 70),
    shape("rsa-4096 pkcs1", "RSA PRIVATE KEY", 3132, 64),
    shape("rsa-4096 pkcs8", "PRIVATE KEY", 3164, 64),
    shape("rsa-4096 openssh", "OPENSSH PRIVATE KEY", 3252, 70),
    shape("p-256 sec1", "EC PRIVATE KEY", 164, 64),
    shape("p-256 pkcs8", "PRIVATE KEY", 184, 64),
    shape("p-256 openssh", "OPENSSH PRIVATE KEY", 416, 70),
    shape("p-521 sec1", "EC PRIVATE KEY", 300, 64),
    shape("p-521 pkcs8", "PRIVATE KEY", 324, 64),
    shape("p-521 openssh", "OPENSSH PRIVATE KEY", 636, 70),
    shape("ed25519 pkcs8", "PRIVATE KEY", 64, 64),
    shape("ed25519 openssh", "OPENSSH PRIVATE KEY", 356, 70),
];

/// A made-up key of `shape`: its lines, armour included.
fn key(shape: &Shape, rng: &mut Rng) -> Vec<String> {
    let mut body = base64(rng, shape.chars);
    // DER fills whole base64 quads; many keys end in padding.
    if rng.below(2) == 0 {
        body.replace_range(shape.chars - 1.., "=");
    }
    let mut lines = vec![format!("-----BEGIN {}-----", shape.label)];
    lines.extend(
        body.as_bytes()
            .chunks(shape.width)
            .map(|c| String::from_utf8(c.to_vec()).unwrap()),
    );
    lines.push(format!("-----END {}-----", shape.label));
    lines
}

/// The key's body lines a leak check looks for (not an armour header such
/// as PGP's `Comment:`).
fn body_lines(key: &[String]) -> Vec<&str> {
    key[1..key.len() - 1]
        .iter()
        .map(String::as_str)
        .filter(|l| l.len() >= 16 && l.bytes().all(|b| B64.contains(&b) || b == b'='))
        .collect()
}

/// How many of `body` are in `text`.
fn leaked(text: &str, body: &[&str]) -> usize {
    body.iter().filter(|l| text.contains(**l)).count()
}

/// How a tool's output reaches the trajectory.
#[derive(Clone, Copy, Debug)]
enum View {
    /// A flat text block: the output as it is in ADP, after the
    /// `[tool_call: …]` title in verifiers.
    Bare,
    /// [`View::Bare`] with no newline after the output's last line.
    BareNoNewline,
    /// An ACP content block (`{"type": "content", "content": {…}}`), which
    /// verifiers writes as its Python `repr`.
    Repr,
    /// [`View::Repr`] with no newline after the output's last line: the
    /// `repr` then ends `…-----'}` (review G1-R2-A-1 at cd992801; about half
    /// of real nested blocks end without a newline).
    ReprNoNewline,
    /// Claude Code's Bash: the command's description, then a `console`
    /// fence.
    Console,
    /// Claude Code's Bash with no description.
    ConsoleOnly,
    /// Claude Code's Read with an offset: a fence and `N\t` gutters.
    Read,
    /// A numbered view with `N→` gutters (the Read tool's own text).
    Arrow,
    /// Hermes headless: its own JSON, the output's last newline stripped,
    /// so an END shown alone ends `…-----", "exit_code": 0, …` (review
    /// G1-R3-A-1 at 47a1a11a).
    HermesJson,
    /// An ACP content block written text-first (the Python ACP SDK's key
    /// order: Hermes, OpenHands, Mistral Vibe), with no newline after the
    /// output: its `repr` ends `…-----', 'type': 'text'}` (G1-R3-A-1).
    TextFirstNoNewline,
    /// OpenHands: a terminal escape before the output's first line.
    OpenHands,
    /// Qwen Code: `Output: ` before the output's first line.
    QwenCode,
    /// Hermes over ACP: `- **output:** ` before the output's first line.
    HermesAcp,
    /// Another prefix before the output's first line (review G1-R7-A-1 at
    /// 8290c756): after a `Result:` line in a flat text block, or at the
    /// start of a content block (its `repr` in verifiers) with no newline
    /// after the output.
    Prefixed {
        prefix: &'static str,
        in_block: bool,
    },
}

/// The prefixes of review G1-R7-A-1 that one CSI with a letter for its
/// final byte did not cover: stacked escapes, bracketed paste, a window
/// title (OSC ended by BEL or ST), a character-set choice, keypad mode, and
/// Hermes's label without its list dash.
const OTHER_PREFIXES: [&str; 7] = [
    "\x1b[?2004l\x1b[0m",
    "\x1b[200~",
    "\x1b]0;root@box: /app\x07",
    "\x1b]2;title\x1b\\",
    "\x1b(B\x1b[m",
    "\x1b[?1l\x1b>",
    "**output:** ",
];

const VIEWS: [View; 7] = [
    View::Bare,
    View::BareNoNewline,
    View::Repr,
    View::ReprNoNewline,
    View::Console,
    View::ConsoleOnly,
    View::Read,
];

fn block(text: String) -> Value {
    json!({"type": "content", "content": {"type": "text", "text": text}})
}

/// A completed tool call that showed `lines` (numbered from `first`).
fn tool_call(id: &str, title: &str, view: View, lines: &[String], first: usize) -> Value {
    let out = lines.join("\n");
    let content = match view {
        View::Bare => vec![json!({"type": "text", "text": format!("{out}\n")})],
        View::BareNoNewline => vec![json!({"type": "text", "text": out})],
        View::Repr => vec![block(format!("{out}\n"))],
        View::ReprNoNewline => vec![block(out)],
        View::HermesJson => vec![json!({"type": "text", "text": format!(
            r#"{{"output": {}, "exit_code": 0, "error": null}}"#,
            Value::String(out)
        )})],
        View::TextFirstNoNewline => {
            vec![json!({"type": "content", "content": {"text": out, "type": "text"}})]
        }
        View::OpenHands => vec![json!({"type": "text", "text": format!(
            "Tool: terminal\nResult:\n\x1b[?2004l{out}\n\x1b[?2004h"
        )})],
        View::QwenCode => vec![json!({"type": "text", "text": format!(
            "Command: {title}\nDirectory: (root)\nOutput: {out}\nError: (none)\nExit Code: 0"
        )})],
        View::HermesAcp => vec![block(format!("- **output:** {out}\n- **exit_code:** 0"))],
        View::Prefixed {
            prefix,
            in_block: false,
        } => vec![json!({"type": "text", "text": format!("Result:\n{prefix}{out}\n")})],
        View::Prefixed {
            prefix,
            in_block: true,
        } => vec![block(format!("{prefix}{out}"))],
        View::Console => vec![
            block(format!("Print part of the key ({title})")),
            block(format!("```console\n{out}\n```")),
        ],
        View::ConsoleOnly => vec![block(format!("```console\n{out}\n```"))],
        View::Read | View::Arrow => {
            let sep = if matches!(view, View::Read) {
                "\t"
            } else {
                "\u{2192}"
            };
            let numbered: Vec<String> = lines
                .iter()
                .enumerate()
                .map(|(i, l)| format!("{:>6}{sep}{l}", first + i))
                .collect();
            vec![block(format!("```\n{}\n```", numbered.join("\n")))]
        }
    };
    let kind = if matches!(view, View::Read | View::Arrow) {
        "read"
    } else {
        "execute"
    };
    json!({"type": "tool_call", "tool_call_id": id, "kind": kind, "title": title,
           "status": "completed", "content": content})
}

/// The events of an agent that reads `key` in three parts (`head`, `sed
/// -n`, `tail`) with messages between them, `filler` characters of other
/// work before the middle read, and a closing message unless it stops on
/// the `tail` (then ADP's reward rides on that call, between the markers).
fn three_part_read(key: &[String], view: View, filler: usize, closing: bool) -> Vec<Value> {
    let n = key.len();
    let (a, c) = (n / 3, 2 * n / 3);
    let mut events = vec![
        json!({"type": "user_message", "text": "What kind of key is in id? Read it in parts."}),
        tool_call("c1", &format!("head -n {a} id"), view, &key[..a], 1),
        json!({"type": "agent_message", "text": "Reading the next part."}),
    ];
    if filler > 0 {
        let words = "the agent keeps working on something else ".repeat(filler / 40 + 1);
        events.push(json!({"type": "agent_message", "text": &words[..filler]}));
    }
    events.push(tool_call(
        "c2",
        &format!("sed -n {},{}p id", a + 1, c),
        view,
        &key[a..c],
        a + 1,
    ));
    events.push(json!({"type": "agent_message", "text": "And the last part."}));
    events.push(tool_call(
        "c3",
        &format!("tail -n +{} id", c + 1),
        view,
        &key[c..],
        c + 1,
    ));
    if closing {
        events.push(json!({"type": "agent_message", "text": "It is a private key."}));
    }
    events
}

fn info(rewards: Value) -> RolloutInfo {
    RolloutInfo {
        task_name: "keys".into(),
        rollout_name: "keys__1".into(),
        agent_name: "claude".into(),
        rewards: rewards.as_object().cloned(),
        ..RolloutInfo::default()
    }
}

/// A job folder with one trial written from `events` the way a run writes
/// it: each event redacted on its own into `trajectory/acp_trajectory.jsonl`
/// (`to_jsonl`, as the trajectory writer does), the trainer files built
/// from that file (`write_trial`'s path), then the job pass. Returns the job
/// folder's guard and the trial folder. Review G1-R2-A-1: built from the
/// raw events, the walk saw markers that the per-event pass had already
/// replaced in a real run.
fn job_of(events: &[Value], info: &RolloutInfo) -> (tempfile::TempDir, PathBuf) {
    let job = tempfile::tempdir().unwrap();
    let trial = job.path().join("keys__1");
    let text: Vec<String> = events.iter().map(Value::to_string).collect();
    let events = parse_jsonl(&text.join("\n")).unwrap();
    std::fs::create_dir_all(trial.join("trajectory")).unwrap();
    std::fs::write(
        trial.join("trajectory/acp_trajectory.jsonl"),
        to_jsonl(&events, &Redactor::default()),
    )
    .unwrap();
    let events = read_events(&trial).unwrap();
    assert!(write_rollout(&trial, &events, info, &Redactor::default()).is_empty());
    assert!(write_job(job.path()).is_empty());
    (job, trial)
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn parse(line: &str) -> Value {
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"))
}

/// `value` with every string emptied: its keys, array lengths, numbers,
/// booleans and nulls, which redaction must never change.
fn skeleton(value: &Value) -> Value {
    match value {
        Value::String(_) => Value::String(String::new()),
        Value::Array(items) => Value::Array(items.iter().map(skeleton).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), skeleton(v)))
                .collect::<Map<String, Value>>(),
        ),
        other => other.clone(),
    }
}

/// The records the exporters build from `events`, unredacted.
fn records(events: &[Value], info: &RolloutInfo) -> [Value; 2] {
    let verifiers = to_verifiers_record(
        events,
        &VerifiersMeta {
            task_id: &info.task_name,
            prompts: &info.prompts,
            rewards: info.rewards.as_ref(),
            error: None,
            model: None,
        },
    );
    let adp = to_adp_record(
        events,
        &AdpMeta {
            trajectory_id: "keys__1",
            task_id: &info.task_name,
            prompts: &info.prompts,
            model: None,
            reward: info.rewards.as_ref().and_then(|r| r.get("reward")),
        },
    )
    .unwrap();
    [verifiers, adp]
}

/// Review E1-BACKSTOP-A 1, E1-BACKSTOP-B 1 (and FIX9-A 1): a key read in
/// three parts keeps no body line in any trainer file, in every shape the
/// exporters write tool output in, for every key type and size. The
/// reverted walk matched only strings made of key lines, which no exporter
/// writes, so every shape here leaked its middle part. LAST-A 1: no line
/// changes structure, and the messages between the parts stay.
#[test]
fn a_key_read_in_three_parts_leaves_no_body_line_in_any_trainer_file() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut failures = Vec::new();
    for shape in SHAPES {
        let key = key(shape, &mut rng);
        let body = body_lines(&key);
        for view in VIEWS {
            for closing in [true, false] {
                let events = three_part_read(&key, view, 0, closing);
                let info = info(json!({"reward": 1.0}));
                let (job, trial) = job_of(&events, &info);
                let raw = records(&events, &info);
                for ((rel, out), raw) in [(VERIFIERS, JOB_VERIFIERS), (ADP, JOB_ADP)]
                    .iter()
                    .zip(&raw)
                {
                    let trial_text = read(&trial.join(rel));
                    let job_text = read(&job.path().join(out));
                    let case = format!("{} {view:?} closing={closing} {out}", shape.name);
                    let left = leaked(&trial_text, &body) + leaked(&job_text, &body);
                    if left > 0 {
                        failures.push(format!("{case}: {left} body lines left"));
                    }
                    if job_text != trial_text {
                        failures.push(format!("{case}: the job line is not the trial line"));
                    }
                    let line = parse(job_text.trim_end());
                    if skeleton(&line) != skeleton(raw) {
                        failures.push(format!("{case}: the structure changed"));
                    }
                    let markers = [key[0].as_str(), key[key.len() - 1].as_str()];
                    for kept in ["Reading the next part.", "And the last part.", "sed -n "]
                        .into_iter()
                        .chain(markers)
                    {
                        if !job_text.contains(kept) {
                            failures.push(format!("{case}: {kept:?} is gone"));
                        }
                    }
                }
            }
        }
    }
    let shown: Vec<&String> = failures.iter().take(12).collect();
    assert!(
        failures.is_empty(),
        "{} failures, first: {shown:#?}",
        failures.len()
    );
}

/// The characters, in `line`, from the end of the key's BEGIN marker to the
/// first body line of the middle part.
fn distance(line: &str, key: &[String]) -> usize {
    let begin = line.find(&key[0]).unwrap() + key[0].len();
    let n = key.len();
    let middle = &key[n / 3];
    line[begin..].find(middle.as_str()).unwrap()
}

/// Decision G-1's span: the middle part goes when it starts 8,000
/// characters after the BEGIN marker in the job line, and stays at 8,400,
/// past the 8,192 the walk reaches (a known limit, in `KEY_SPAN`'s comment
/// and the redaction docs). The middle part is a whole RSA-4096 third, so
/// the lines that run past 8,192 go with the first.
#[test]
fn a_key_parts_span_reaches_8192_characters() {
    let mut rng = Rng(7);
    let key = key(&SHAPES[9], &mut rng);
    let n = key.len();
    let middle: Vec<&str> = key[n / 3..2 * n / 3].iter().map(String::as_str).collect();
    for view in VIEWS {
        for (index, (rel, out)) in [(VERIFIERS, JOB_VERIFIERS), (ADP, JOB_ADP)]
            .into_iter()
            .enumerate()
        {
            // The line as the value pass alone writes it, to measure where
            // the middle part starts.
            let line_with = |filler: usize| {
                let events = three_part_read(&key, view, filler, true);
                let record = records(&events, &info(json!({"reward": 1.0})))[index].clone();
                let r = Redactor::default();
                if index == 0 {
                    verifiers_line(&record, &r)
                } else {
                    adp_line(&record, &r)
                }
            };
            let base = distance(&line_with(100), &key);
            for (target, gone) in [(8_000, true), (8_400, false)] {
                let filler = 100 + target - base;
                assert_eq!(distance(&line_with(filler), &key), target);
                let events = three_part_read(&key, view, filler, true);
                let (job, trial) = job_of(&events, &info(json!({"reward": 1.0})));
                let job_text = read(&job.path().join(out));
                assert_eq!(job_text, read(&trial.join(rel)));
                let left = leaked(&job_text, &middle);
                if gone {
                    assert_eq!(left, 0, "{view:?} {out} at {target}: {job_text}");
                } else {
                    assert_eq!(left, middle.len(), "{view:?} {out} at {target}");
                }
                // The other parts are next to their markers, so they go
                // either way.
                assert_eq!(leaked(&job_text, &body_lines(&key)), left, "{view:?} {out}");
            }
        }
    }
}

/// Review G1-R1-A-1: `head -1 id; wc -l id` shows the BEGIN marker and then
/// other output (`27 id`), so the marker opened no span and the middle part
/// read later kept every body line in the trainer files. A marker alone on
/// its line opens one now.
#[test]
fn a_marker_line_followed_by_other_output_opens_a_span() {
    let mut rng = Rng(41);
    let mut failures = Vec::new();
    for shape in [&SHAPES[3], &SHAPES[5], &SHAPES[14], &SHAPES[19]] {
        let key = key(shape, &mut rng);
        let body = body_lines(&key);
        let n = key.len();
        let c = n / 2;
        for view in [
            View::Bare,
            View::BareNoNewline,
            View::Repr,
            View::ReprNoNewline,
            View::Console,
            View::ConsoleOnly,
        ] {
            let first = [key[0].clone(), format!("{n} id")];
            let events = vec![
                json!({"type": "user_message", "text": "What kind of key is in id?"}),
                tool_call("c1", "head -1 id; wc -l id", view, &first, 1),
                json!({"type": "agent_message", "text": "Reading the next part."}),
                tool_call("c2", &format!("sed -n 2,{c}p id"), view, &key[1..c], 2),
                json!({"type": "agent_message", "text": "And the last part."}),
                tool_call(
                    "c3",
                    &format!("tail -n +{} id", c + 1),
                    view,
                    &key[c..],
                    c + 1,
                ),
            ];
            let (job, trial) = job_of(&events, &info(json!({"reward": 1.0})));
            for file in [
                trial.join(VERIFIERS),
                trial.join(ADP),
                job.path().join(JOB_VERIFIERS),
                job.path().join(JOB_ADP),
            ] {
                let left = leaked(&read(&file), &body);
                if left > 0 {
                    failures.push(format!(
                        "{} {view:?} {}: {left}",
                        shape.name,
                        file.display()
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "body lines left: {failures:#?}");
}

/// Review G1-R1-A-2 and G1-R1-B-1: an END marker shown before the middle
/// part (a `grep -n` listing of both markers, a `grep` for the END line,
/// prose quoting it) closed the span, so the parts read after it kept every
/// body line in the trainer files. An END closes the span only after a body
/// line now. Review G1-R2-A-1: the trial is written the way a run writes it
/// ([`job_of`]), where each event was redacted before the walk: the listing
/// lost both markers and the other two gained a redaction before the END,
/// so all three still leaked. Review G1-R3-B-1: in a `grep -rn` listing
/// whose first BEGIN is a source file's quote of the marker, the event
/// pass paired that quote with the key's END and replaced the key's BEGIN
/// too.
#[test]
fn an_end_shown_before_the_middle_part_leaves_the_span_open() {
    let mut rng = Rng(43);
    let mut failures = Vec::new();
    for shape in [&SHAPES[3], &SHAPES[5], &SHAPES[14], &SHAPES[19]] {
        let key = key(shape, &mut rng);
        let body = body_lines(&key);
        let n = key.len();
        let (a, c) = (n / 3, 2 * n / 3);
        let (begin, end) = (&key[0], &key[n - 1]);
        for view in [
            View::Bare,
            View::BareNoNewline,
            View::Repr,
            View::ReprNoNewline,
            View::Console,
            View::ConsoleOnly,
        ] {
            let listing = [format!("1:{begin}"), format!("{n}:{end}")];
            let recursive = [
                format!("./src/pem.py:12:    if line.startswith(\"{begin}\"):"),
                format!("./keys/id:1:{begin}"),
                format!("./keys/id:{n}:{end}"),
            ];
            let head = tool_call("c1", &format!("head -n {a} id"), view, &key[..a], 1);
            let middle = tool_call(
                "c3",
                &format!("sed -n {},{c}p id", a + 1),
                view,
                &key[a..c],
                a + 1,
            );
            let tail = tool_call(
                "c4",
                &format!("tail -n +{} id", c + 1),
                view,
                &key[c..],
                c + 1,
            );
            let cases = [
                (
                    "grep listing",
                    vec![
                        tool_call("c1", "grep -n 'PRIVATE KEY' id", view, &listing, 1),
                        tool_call("c2", &format!("sed -n 2,{c}p id"), view, &key[1..c], 2),
                        tool_call(
                            "c3",
                            &format!("sed -n {},{}p id", c + 1, n - 1),
                            view,
                            &key[c..n - 1],
                            c + 1,
                        ),
                    ],
                ),
                (
                    "grep -rn listing, a source file first",
                    vec![
                        tool_call("c1", "grep -rn 'PRIVATE KEY' .", view, &recursive, 1),
                        tool_call(
                            "c2",
                            &format!("sed -n 2,{}p keys/id", n - 1),
                            view,
                            &key[1..n - 1],
                            2,
                        ),
                    ],
                ),
                (
                    "grep for the END",
                    vec![
                        head.clone(),
                        tool_call(
                            "c2",
                            &format!("grep -n -- '{end}' id"),
                            view,
                            &listing[1..],
                            n,
                        ),
                        middle.clone(),
                        tail.clone(),
                    ],
                ),
                (
                    "quoted END",
                    vec![
                        head.clone(),
                        json!({"type": "agent_message", "text": format!("The key ends at {end}, so I read the middle next.")}),
                        middle.clone(),
                        tail.clone(),
                    ],
                ),
            ];
            for (name, events) in cases {
                let (job, trial) = job_of(&events, &info(json!({"reward": 1.0})));
                for file in [
                    trial.join(VERIFIERS),
                    trial.join(ADP),
                    job.path().join(JOB_VERIFIERS),
                    job.path().join(JOB_ADP),
                ] {
                    let text = read(&file);
                    let left = leaked(&text, &body);
                    if left > 0 {
                        failures.push(format!(
                            "{} {view:?} {name} {}: {left}",
                            shape.name,
                            file.display()
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "body lines left: {failures:#?}");
}

/// Review G1-R2-B-1: a key's last lines shown in their own output before
/// its middle part (`head`, then `tail`, then `sed -n`) closed the span at
/// the `tail` output's END, so the middle part read afterwards kept every
/// body line in the trainer files. An END closes no span now. The order
/// head, middle, tail is the control. Both ends in one output (`head -3 id;
/// tail -3 id`), then the middle, keeps the middle: a known limit, pinned
/// here.
#[test]
fn a_key_read_head_then_tail_then_middle_keeps_no_body_line() {
    let mut rng = Rng(47);
    let mut failures = Vec::new();
    for shape in [&SHAPES[3], &SHAPES[5], &SHAPES[14], &SHAPES[19]] {
        let key = key(shape, &mut rng);
        let body = body_lines(&key);
        let n = key.len();
        let h = n / 2;
        for view in VIEWS {
            let call = |id: &str, title: String, lines: &[String], first: usize| {
                tool_call(id, &title, view, lines, first)
            };
            let numbered = |from: usize, to: usize| -> Vec<String> {
                (from..to)
                    .map(|i| {
                        let sep = if i == n - 1 { ':' } else { '-' };
                        format!("{}{sep}{}", i + 1, key[i])
                    })
                    .collect()
            };
            let cases = [
                (
                    "head -3, tail -3, sed",
                    vec![
                        call("c1", "head -3 id".into(), &key[..3], 1),
                        call("c2", "tail -3 id".into(), &key[n - 3..], n - 2),
                        call("c3", format!("sed -n 4,{}p id", n - 3), &key[3..n - 3], 4),
                    ],
                ),
                (
                    "head -1, tail -n +h, sed",
                    vec![
                        call("c1", "head -1 id".into(), &key[..1], 1),
                        call("c2", format!("tail -n +{} id", h + 1), &key[h..], h + 1),
                        call("c3", format!("sed -n 2,{h}p id"), &key[1..h], 2),
                    ],
                ),
                (
                    "head -3, grep -n -B1 END, sed",
                    vec![
                        call("c1", "head -3 id".into(), &key[..3], 1),
                        call(
                            "c2",
                            "grep -n -B1 END id".into(),
                            &numbered(n - 2, n),
                            n - 1,
                        ),
                        call("c3", format!("sed -n 4,{}p id", n - 2), &key[3..n - 2], 4),
                    ],
                ),
                (
                    "control: head, sed, tail",
                    vec![
                        call("c1", "head -3 id".into(), &key[..3], 1),
                        call("c2", format!("sed -n 4,{}p id", n - 3), &key[3..n - 3], 4),
                        call("c3", "tail -3 id".into(), &key[n - 3..], n - 2),
                    ],
                ),
            ];
            for (name, events) in cases {
                let (job, trial) = job_of(&events, &info(json!({"reward": 1.0})));
                for file in [
                    trial.join(VERIFIERS),
                    trial.join(ADP),
                    job.path().join(JOB_VERIFIERS),
                    job.path().join(JOB_ADP),
                ] {
                    let left = leaked(&read(&file), &body);
                    if left > 0 {
                        failures.push(format!(
                            "{} {view:?} {name} {}: {left}",
                            shape.name,
                            file.display()
                        ));
                    }
                }
            }
            // The known limit: both ends in one output, then the middle. The
            // event's redaction replaces the key that output shows whole,
            // markers and all (0.8's rule), so the walk sees no BEGIN.
            let ends: Vec<String> = key[..3].iter().chain(&key[n - 3..]).cloned().collect();
            let events = vec![
                call("c1", "head -3 id; tail -3 id".into(), &ends, 1),
                call("c2", format!("sed -n 4,{}p id", n - 3), &key[3..n - 3], 4),
            ];
            let (job, _) = job_of(&events, &info(json!({"reward": 1.0})));
            let middle: Vec<&str> = key[3..n - 3]
                .iter()
                .map(String::as_str)
                .filter(|l| body.contains(l))
                .collect();
            let text = read(&job.path().join(JOB_VERIFIERS));
            assert_eq!(
                leaked(&text, &middle),
                middle.len(),
                "{} {view:?}",
                shape.name
            );
        }
    }
    assert!(failures.is_empty(), "body lines left: {failures:#?}");
}

/// Review G1-R3-B-2: a `tail` part's body is redacted in its own event, so
/// its END did not carry the span on: the middle read soon after the `tail`
/// but more than 8,192 characters after the BEGIN kept its body lines, and
/// with the `tail` read first no span opened. An END with a redacted body
/// before it carries the span on now. Review G1-R1-B-1 (round 1 at
/// 6d38ceed): a `tail` part with only the END line (`tail -1 id`, `grep -n
/// END id`) opened no span, so the middle read next kept every body line;
/// an END alone on its line opens one now. Review G1-R2-A-1 (at cd992801):
/// with no newline after the END, its `repr` block ends `…-----'}`, which
/// did not count as alone, so the same orders kept 25 of 25 body lines in
/// the verifiers files. Review G1-R3-A-1 (at 47a1a11a): the same with
/// the wrapper's next field after the quote (Hermes's headless JSON, a
/// text-first `repr` block with no newline). A middle part read before any
/// part with a marker keeps its body lines: a known limit, pinned here.
#[test]
fn a_key_part_read_after_its_tail_keeps_no_body_line() {
    let mut rng = Rng(53);
    let mut failures = Vec::new();
    for shape in [&SHAPES[3], &SHAPES[5], &SHAPES[14], &SHAPES[19]] {
        let key = key(shape, &mut rng);
        let body = body_lines(&key);
        let n = key.len();
        let middle: Vec<&str> = key[3..n - 3]
            .iter()
            .map(String::as_str)
            .filter(|l| body.contains(l))
            .collect();
        // Review G1-R3-A-1: views whose wrapper goes on after the quote.
        for view in VIEWS
            .into_iter()
            .chain([View::HermesJson, View::TextFirstNoNewline])
        {
            let call = |id: &str, title: &str, lines: &[String], first: usize| {
                tool_call(id, title, view, lines, first)
            };
            let other = |chars: usize| json!({"type": "agent_message", "text": "other work ".repeat(chars / 11)});
            let head = call("c1", "head -3 id", &key[..3], 1);
            let tail = call("c2", "tail -3 id", &key[n - 3..], n - 2);
            let sed = call("c3", "sed -n 4,-3p id", &key[3..n - 3], 4);
            // Review G1-R1-B-1: a `tail` part with only the END line.
            let end_only = call("c2", "tail -1 id", &key[n - 1..], n);
            let end_line = [format!("{n}:{}", key[n - 1])];
            // A shell command's output: the Read tool never shows a grep.
            let shell = if matches!(view, View::Read) {
                View::Console
            } else {
                view
            };
            let grep_end = tool_call("c2", "grep -n END id", shell, &end_line, n);
            let sed_all = call("c3", &format!("sed -n 2,{}p id", n - 1), &key[1..n - 1], 2);
            let head_1 = call("c1", "head -1 id", &key[..1], 1);
            let tail_1 = call("c3", "tail -1 id", &key[n - 1..], n);
            let cases = [
                (
                    "tail, sed, head",
                    vec![tail.clone(), sed.clone(), head.clone()],
                ),
                (
                    "tail -1, sed, head -1",
                    vec![end_only.clone(), sed_all.clone(), head_1.clone()],
                ),
                (
                    "head -1, sed, tail -1",
                    vec![head_1, sed_all.clone(), tail_1],
                ),
                ("tail -1, sed", vec![end_only, sed_all.clone()]),
                ("grep -n END, sed", vec![grep_end, sed_all]),
                (
                    "head, 6 KB, tail, 3 KB, sed",
                    vec![
                        head.clone(),
                        other(6_000),
                        tail.clone(),
                        other(3_000),
                        sed.clone(),
                    ],
                ),
            ];
            for (name, events) in cases {
                let (job, trial) = job_of(&events, &info(json!({"reward": 1.0})));
                for file in [
                    trial.join(VERIFIERS),
                    trial.join(ADP),
                    job.path().join(JOB_VERIFIERS),
                    job.path().join(JOB_ADP),
                ] {
                    let left = leaked(&read(&file), &body);
                    if left > 0 {
                        failures.push(format!(
                            "{} {view:?} {name} {}: {left}",
                            shape.name,
                            file.display()
                        ));
                    }
                }
            }
            // The known limit: the middle part first.
            let (job, _) = job_of(&[sed, head, tail], &info(json!({"reward": 1.0})));
            let text = read(&job.path().join(JOB_VERIFIERS));
            assert_eq!(
                leaked(&text, &middle),
                middle.len(),
                "{} {view:?}",
                shape.name
            );
            assert_eq!(
                leaked(&text, &body),
                middle.len(),
                "{} {view:?}",
                shape.name
            );
        }
    }
    assert!(failures.is_empty(), "body lines left: {failures:#?}");
}

/// Review G1F-A-1 and G1F-A-2 (at fb5d1c52; G1-R3-A-2 at 47a1a11a had
/// pinned this as a known limit): an agent that puts a terminal escape
/// (OpenHands) or a label (`Output: ` in Qwen Code, `- **output:** ` in
/// Hermes over ACP) before the first line of a command's output hid a
/// marker that `head -1`, `tail -1` or `grep -n END` showed, so it opened
/// no span, and the middle part's first body line carried the same prefix,
/// so it was not a key line. `head -1`, `sed -n`, `tail -1` kept every body
/// line in the job files, which the job pass's text pass (before the walk
/// replaced it) had removed;
/// `head`, `sed -n`, `tail` kept the middle part's first line. The walk now
/// steps over the prefix, and no order keeps a body line. Review G1-R7-A-1
/// (at 8290c756): the same with the other prefixes in [`OTHER_PREFIXES`],
/// in a flat text block and in a content block.
#[test]
fn a_prefix_before_the_output_keeps_no_body_line() {
    let mut rng = Rng(61);
    let mut failures = Vec::new();
    for shape in [&SHAPES[3], &SHAPES[5], &SHAPES[14], &SHAPES[19]] {
        let key = key(shape, &mut rng);
        let body = body_lines(&key);
        let n = key.len();
        let (a, c) = (n / 3, 2 * n / 3);
        let q = n / 4;
        let others = OTHER_PREFIXES
            .into_iter()
            .flat_map(|prefix| [false, true].map(|in_block| View::Prefixed { prefix, in_block }));
        for view in [View::OpenHands, View::QwenCode, View::HermesAcp]
            .into_iter()
            .chain(others)
        {
            let call = |id: &str, title: &str, lines: &[String], first: usize| {
                tool_call(id, title, view, lines, first)
            };
            let other = |chars: usize| json!({"type": "agent_message", "text": "other work ".repeat(chars / 11)});
            let sed_all = call("c2", &format!("sed -n 2,{}p id", n - 1), &key[1..n - 1], 2);
            let head_1 = call("c1", "head -1 id", &key[..1], 1);
            let tail_1 = call("c3", "tail -1 id", &key[n - 1..], n);
            let grep_end = call("c1", "grep -n END id", &[format!("{n}:{}", key[n - 1])], n);
            let head = call("c1", &format!("head -n {a} id"), &key[..a], 1);
            let sed = call(
                "c2",
                &format!("sed -n {},{c}p id", a + 1),
                &key[a..c],
                a + 1,
            );
            let tail = call("c3", &format!("tail -n +{} id", c + 1), &key[c..], c + 1);
            let head_3 = call("c1", "head -3 id", &key[..3], 1);
            let sed_3 = call("c2", &format!("sed -n 4,{}p id", n - 3), &key[3..n - 3], 4);
            let tail_3 = call("c3", "tail -3 id", &key[n - 3..], n - 2);
            let parts: Vec<Value> = (0..4)
                .map(|i| {
                    let end = if i == 3 { n } else { (i + 1) * q };
                    call(
                        &format!("p{i}"),
                        &format!("part {i}"),
                        &key[i * q..end],
                        i * q + 1,
                    )
                })
                .collect();
            let cases = [
                (
                    "head -1, sed, tail -1",
                    vec![head_1.clone(), sed_all.clone(), tail_1.clone()],
                ),
                ("head -1, sed", vec![head_1.clone(), sed_all.clone()]),
                ("tail -1, sed", vec![tail_1.clone(), sed_all.clone()]),
                (
                    "tail -1, sed, head -1",
                    vec![tail_1, sed_all.clone(), head_1],
                ),
                ("grep -n END, sed", vec![grep_end, sed_all]),
                (
                    "head, sed, tail",
                    vec![head.clone(), sed.clone(), tail.clone()],
                ),
                (
                    "tail, sed, head",
                    vec![tail.clone(), sed.clone(), head.clone()],
                ),
                (
                    "head, other work, sed, tail",
                    vec![head, other(6000), sed, tail],
                ),
                ("head -3, sed, tail -3", vec![head_3, sed_3, tail_3]),
                (
                    "four parts, other work between",
                    vec![
                        parts[0].clone(),
                        other(6000),
                        parts[1].clone(),
                        other(6000),
                        parts[2].clone(),
                        other(6000),
                        parts[3].clone(),
                    ],
                ),
            ];
            for (order, events) in cases {
                let (job, trial) = job_of(&events, &info(json!({"reward": 1.0})));
                for file in [
                    trial.join(VERIFIERS),
                    trial.join(ADP),
                    job.path().join(JOB_VERIFIERS),
                    job.path().join(JOB_ADP),
                ] {
                    let text = read(&file);
                    let left = leaked(&text, &body);
                    if left > 0 {
                        let name = file.file_name().unwrap().to_string_lossy();
                        let scope = if file.starts_with(&trial) {
                            "trial"
                        } else {
                            "job"
                        };
                        failures.push(format!(
                            "{} {view:?} {order}: {scope} {name} keeps {left} of {} body lines",
                            shape.name,
                            body.len()
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "body lines left: {failures:#?}");
}

/// Review G1-R1-A-2 (round 1 at 6d38ceed): the walk rewrote every string of
/// the record, so after a key's head part a task id or model name made
/// only of base64 characters (`sshkeyrotationaudit`) became
/// `***REDACTED***` in `info` and `details`, and a trainer that groups
/// rollouts by task or environment lost the rollout. Only the messages'
/// text is walked now; the key's body still goes.
#[test]
fn a_key_part_leaves_the_record_metadata_alone() {
    let mut rng = Rng(59);
    let (task, model) = ("sshkeyrotationaudit", "org/modelname1234567");
    for shape in [&SHAPES[5], &SHAPES[11], &SHAPES[19]] {
        let key = key(shape, &mut rng);
        let body = body_lines(&key);
        let n = key.len();
        let events = vec![
            json!({"type": "user_message", "text": "Audit the key rotation."}),
            tool_call("c1", "head -2 id", View::Console, &key[..2], 1),
            tool_call(
                "c2",
                &format!("sed -n 3,{}p id", n - 1),
                View::Console,
                &key[2..n - 1],
                3,
            ),
        ];
        let info = RolloutInfo {
            task_name: task.into(),
            model: Some(model.into()),
            ..info(json!({"reward": 1.0}))
        };
        let (job, trial) = job_of(&events, &info);
        for file in [trial.join(VERIFIERS), job.path().join(JOB_VERIFIERS)] {
            let text = read(&file);
            let record = parse(text.lines().next().unwrap());
            let meta = &record["info"];
            assert_eq!(
                (&meta["task_id"], &meta["environment"], &meta["model"]),
                (&json!(task), &json!(task), &json!(model)),
                "{} {}",
                shape.name,
                file.display()
            );
            assert_eq!(leaked(&text, &body), 0, "{} {}", shape.name, file.display());
        }
        for file in [trial.join(ADP), job.path().join(JOB_ADP)] {
            let text = read(&file);
            let record = parse(text.lines().next().unwrap());
            let meta = &record["details"];
            assert_eq!(
                (&meta["task_id"], &meta["environment"], &meta["model"]),
                (&json!(task), &json!(task), &json!(model)),
                "{} {}",
                shape.name,
                file.display()
            );
            assert_eq!(leaked(&text, &body), 0, "{} {}", shape.name, file.display());
        }
    }
}

/// A change to an exporter's record, given the grep command to quote.
type AddField = fn(&mut Value, &str);

/// Write `lines` as a trial's `rel` file, run the job pass, and return the
/// job file's text.
fn job_text_of(rel: &str, out: &str, lines: &[String]) -> String {
    let job = tempfile::tempdir().unwrap();
    let path = job.path().join("keys__1").join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, lines.concat()).unwrap();
    assert!(write_job(job.path()).is_empty());
    read(&job.path().join(out))
}

/// Review LAST-B 1: the old job pass ran the text rules over the whole
/// line and kept the result only if it still parsed, so one unrelated match
/// that broke the JSON (a reward item `found_password: 1.0`, `"api_key":
/// null`, `"use_credentials": true`, a BEGIN/END pair across structures)
/// cancelled the cut and the middle part of the key stayed. Each case keeps
/// 0 body lines now, and the job line has the trial line's structure.
#[test]
fn unrelated_matches_on_the_line_do_not_bring_the_middle_part_back() {
    let mut rng = Rng(11);
    for shape in [&SHAPES[3], &SHAPES[14], &SHAPES[19]] {
        let key = key(shape, &mut rng);
        let body = body_lines(&key);
        for view in [
            View::Bare,
            View::BareNoNewline,
            View::Repr,
            View::ReprNoNewline,
            View::Console,
        ] {
            let events = three_part_read(&key, view, 0, true);
            // A security task's reward item, through the real path.
            let info = info(json!({"reward": 1.0, "found_password": 1.0}));
            let (job, _) = job_of(&events, &info);
            for out in [JOB_VERIFIERS, JOB_ADP] {
                let text = read(&job.path().join(out));
                assert_eq!(leaked(&text, &body), 0, "{} {view:?} {out}", shape.name);
            }
            let text = read(&job.path().join(JOB_VERIFIERS));
            assert_eq!(
                parse(text.trim_end())["metrics"],
                json!({"found_password": 1.0})
            );

            // Fields other producers add, and a pair across structures.
            let info = self::info(json!({"reward": 1.0}));
            let [verifiers, adp] = records(&events, &info);
            let grep = "{\"cmd\": \"grep -c '-----BEGIN RSA PRIVATE KEY-----' id\"}";
            let cases: [(&str, AddField); 3] = [
                ("api_key null", |r, _| {
                    let at = if r.get("info").is_some() {
                        "info"
                    } else {
                        "details"
                    };
                    r[at]["api_key"] = Value::Null;
                }),
                ("use_credentials true", |r, _| {
                    let at = if r.get("info").is_some() {
                        "info"
                    } else {
                        "details"
                    };
                    r[at]["use_credentials"] = Value::Bool(true);
                }),
                ("pair across structures", |r, grep| {
                    let pair = [
                        json!({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": grep}}]}),
                        json!({"role": "tool", "content": "-----END RSA PRIVATE KEY-----"}),
                    ];
                    let list = if r.get("prompt").is_some() {
                        "prompt"
                    } else {
                        "content"
                    };
                    let items = r[list].as_array_mut().unwrap();
                    items.splice(0..0, pair);
                }),
            ];
            for (name, add) in cases {
                for (rel, out, record) in
                    [(VERIFIERS, JOB_VERIFIERS, &verifiers), (ADP, JOB_ADP, &adp)]
                {
                    let mut record = record.clone();
                    add(&mut record, grep);
                    let r = Redactor::default();
                    let trial = if rel == VERIFIERS {
                        verifiers_line(&record, &r)
                    } else {
                        adp_line(&record, &r)
                    };
                    let text = job_text_of(rel, out, std::slice::from_ref(&trial));
                    let case = format!("{} {view:?} {name} {out}", shape.name);
                    assert_eq!(leaked(&text, &body), 0, "{case}: {text}");
                    assert_eq!(
                        skeleton(&parse(text.trim_end())),
                        skeleton(&parse(trial.trim_end())),
                        "{case}"
                    );
                    assert!(text.contains("Reading the next part."), "{case}");
                }
            }
        }
    }
}

/// `grep -rn -A5 BEGIN` (and `-B` before an END) under folders whose names
/// hold `+`, `@`, `,`, `%`, `=` or non-ASCII characters (review FINAL-B 1):
/// each context line starts with the file name, which the half rules now
/// take as a gutter. Prose before an END marker stays as it is.
#[test]
fn grep_context_under_odd_file_names_keeps_no_body_line() {
    let mut rng = Rng(23);
    let key = key(&SHAPES[3], &mut rng);
    let body = body_lines(&key);
    let n = key.len();
    for path in [
        "./k/key+old.pem",
        "/home/a@b/.ssh/id_rsa",
        "./a,b%c=d/id_rsa",
        "./密钥/id_rsa",
    ] {
        let after: Vec<String> = std::iter::once(format!("{path}:1:{}", key[0]))
            .chain((1..6).map(|i| format!("{path}-{}-{}", i + 1, key[i])))
            .collect();
        let before: Vec<String> = (n - 6..n - 1)
            .map(|i| format!("{path}-{}-{}", i + 1, key[i]))
            .chain(std::iter::once(format!("{path}:{n}:{}", key[n - 1])))
            .collect();
        for view in [
            View::Bare,
            View::BareNoNewline,
            View::Repr,
            View::ReprNoNewline,
            View::Console,
        ] {
            let events = vec![
                json!({"type": "user_message", "text": "Note e.g. the key is:\n-----END PRIVATE KEY-----\nFind it."}),
                tool_call("c1", "grep -rn -A5 BEGIN .", view, &after, 1),
                json!({"type": "agent_message", "text": "Now its end."}),
                tool_call("c2", "grep -rn -B5 END .", view, &before, 1),
            ];
            let (job, trial) = job_of(&events, &info(json!({"reward": 1.0})));
            for file in [
                trial.join(VERIFIERS),
                trial.join(ADP),
                job.path().join(JOB_VERIFIERS),
                job.path().join(JOB_ADP),
            ] {
                let text = read(&file);
                let case = format!("{path} {view:?} {}", file.display());
                assert_eq!(leaked(&text, &body), 0, "{case}: {text}");
                assert!(
                    text.contains(r"Note e.g. the key is:\n-----END PRIVATE KEY-----\nFind it."),
                    "{case}: {text}"
                );
            }
        }
    }
}

/// Decision G-3: an OpenPGP private key block matched no rule. Whole, as
/// halves (a block longer than 8,192 characters), and read in three parts,
/// it keeps no body line in the trial files or the job files. Review
/// G1-R1-A-3: a block over 8,192 characters with a `Comment:` header kept
/// its head half's body in `trainer/atif.json`, which is redacted value by
/// value with no walk, when read in two halves. Review G1-R2-A-2: so did a
/// head half in a numbered view (`cat -n`, the Read tool's `N→`), with a
/// header or with only the blank line after the marker.
#[test]
fn pgp_private_key_blocks_are_redacted() {
    let mut rng = Rng(31);
    // 2,380 and 9,484 leave a 12-character last line before the checksum,
    // as GnuPG's default key does (review G1-R1-A-1 at 6d38ceed: the END
    // half kept every body line); 2,400 and 9,500 leave 32 and 28.
    for chars in [2_380, 2_400, 9_484, 9_500] {
        for header in [
            None,
            Some("Comment: made up for a test"),
            Some("Version: GnuPG v1"),
        ] {
            let body = base64(&mut rng, chars);
            let mut key = vec!["-----BEGIN PGP PRIVATE KEY BLOCK-----".to_string()];
            key.extend(header.map(str::to_string));
            key.push(String::new());
            key.extend(
                body.as_bytes()
                    .chunks(64)
                    .map(|c| String::from_utf8(c.to_vec()).unwrap()),
            );
            key.push(format!("={}", base64(&mut rng, 4)));
            key.push("-----END PGP PRIVATE KEY BLOCK-----".into());
            let lines = body_lines(&key);
            let mut cases = vec![(
                "whole",
                vec![
                    json!({"type": "user_message", "text": "Export the signing key."}),
                    tool_call(
                        "c1",
                        "gpg --export-secret-keys --armor",
                        View::Console,
                        &key,
                        1,
                    ),
                ],
                true,
            )];
            // Two halves, each with its marker (`head`, then `tail`).
            let half = key.len() / 2;
            for view in [View::Console, View::Read, View::Arrow] {
                cases.push((
                    "halves",
                    vec![
                        json!({"type": "user_message", "text": "Export the signing key."}),
                        tool_call(
                            "c1",
                            &format!("head -n {half} k.asc"),
                            view,
                            &key[..half],
                            1,
                        ),
                        tool_call(
                            "c2",
                            &format!("tail -n +{} k.asc", half + 1),
                            view,
                            &key[half..],
                            half + 1,
                        ),
                    ],
                    true,
                ));
            }
            // The END half alone (`tail`, no head part before it).
            cases.push((
                "tail only",
                vec![
                    json!({"type": "user_message", "text": "Export the signing key."}),
                    tool_call(
                        "c1",
                        &format!("tail -n +{} k.asc", half + 1),
                        View::Console,
                        &key[half..],
                        half + 1,
                    ),
                ],
                true,
            ));
            cases.push((
                "three parts",
                three_part_read(&key, View::Repr, 0, true),
                false,
            ));
            for (name, events, is_whole) in cases {
                let (job, trial) = job_of(&events, &info(json!({"reward": 1.0})));
                let mut files = vec![
                    trial.join(VERIFIERS),
                    trial.join(ADP),
                    job.path().join(JOB_VERIFIERS),
                    job.path().join(JOB_ADP),
                ];
                // A middle part read on its own stays in `atif.json` (a known
                // limit); a block shown whole or in two halves does not.
                if is_whole {
                    files.push(trial.join(ATIF));
                    files.push(trial.join("trajectory/acp_trajectory.jsonl"));
                }
                for file in files {
                    let text = read(&file);
                    let left = leaked(&text, &lines);
                    assert_eq!(left, 0, "{chars} {header:?} {name} {}", file.display());
                }
            }
        }
    }
}

/// A random string from fragments that make the walk and the rules work:
/// markers, body lines, gutters, separators and their escapes, quotes,
/// non-ASCII and control characters, `name: value` carriers, and words.
fn fuzz_string(rng: &mut Rng) -> String {
    const ATOMS: &[&str] = &[
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----END OPENSSH PRIVATE KEY-----",
        "-----BEGIN PGP PRIVATE KEY BLOCK-----",
        "-----END PGP PRIVATE KEY BLOCK-----",
        "-----BEGIN PRIVATE KEY-----",
        "\n",
        "\n",
        "\\n",
        "\\\\n",
        "\r\n",
        "\t",
        "\\t",
        " ",
        "'",
        "\"",
        "\\\"",
        "\\",
        "```console\n",
        "\n```",
        "    12\t",
        "12-",
        "12:",
        "./k/id_rsa-3-",
        "./k/key+old.pem-",
        "/home/a@b/.ssh/id_rsa:4:",
        "./a,b%c=d/x-",
        "./密钥/id_rsa-2-",
        "My Keys/id-",
        "→",
        "é",
        "😀",
        "\u{7f}",
        "\u{1}",
        "Note e.g. the key is:",
        "[tool_call: sed -n 2,4p id]\n",
        "{'type': 'text', 'text': '",
        "'}",
        "api_key=",
        "token: ",
        "\"password\": ",
        "***REDACTED***",
        "=AbCd",
        "done",
        "the hash is",
    ];
    let mut s = String::new();
    for _ in 0..rng.below(14) {
        match rng.below(5) {
            0 => {
                let n = 16 + rng.below(56);
                s.push_str(&base64(rng, n));
            }
            1 => {
                let n = 1 + rng.below(15);
                s.push_str(&base64(rng, n));
            }
            _ => s.push_str(rng.pick(ATOMS)),
        }
    }
    s
}

fn fuzz_value(rng: &mut Rng, depth: usize) -> Value {
    const KEYS: &[&str] = &[
        "role",
        "content",
        "api_key",
        "use_credentials",
        "found_password",
        "token",
        "reward",
        "completion",
        "description",
        "x",
    ];
    match rng.below(if depth == 0 { 4 } else { 8 }) {
        0..=2 => Value::String(fuzz_string(rng)),
        3 => match rng.below(8) {
            0 => Value::Null,
            1 => Value::Bool(rng.below(2) == 0),
            2 => json!(rng.next() as i64),
            3 => json!(-0.0),
            4 => json!(1e16),
            5 => json!(1e-5),
            6 => json!(1.0),
            _ => json!((rng.below(1000) as f64) / 7.0),
        },
        4 | 5 => Value::Array(
            (0..rng.below(6))
                .map(|_| fuzz_value(rng, depth - 1))
                .collect(),
        ),
        _ => {
            let mut map = Map::new();
            for _ in 0..rng.below(6) {
                let key = rng.pick(KEYS).to_string();
                map.insert(key, fuzz_value(rng, depth - 1));
            }
            Value::Object(map)
        }
    }
}

/// A seeded fuzz of 10,000 records through the job pass: every job line is
/// JSON, has its trial line's structure (keys, array lengths, numbers), is
/// byte for byte the trial line when there is nothing to redact, and a
/// second pass changes nothing.
#[test]
fn fuzzed_records_keep_their_structure_through_the_job_pass() {
    let mut rng = Rng(0x5eed_1012);
    let redactor = Redactor::default();
    let trial_lines: Vec<String> = (0..10_000)
        .map(|_| {
            let mut record = json!({
                "prompt": fuzz_value(&mut rng, 3),
                "completion": fuzz_value(&mut rng, 3),
                "reward": 1.0,
                "info": fuzz_value(&mut rng, 2),
            });
            // The trial line, as the trainer writer's value pass leaves it.
            redactor.redact_json(&mut record);
            to_python_json(&record) + "\n"
        })
        .collect();
    let once = job_text_of(VERIFIERS, JOB_VERIFIERS, &trial_lines);
    let twice = job_text_of(
        VERIFIERS,
        JOB_VERIFIERS,
        &once.lines().map(|l| format!("{l}\n")).collect::<Vec<_>>(),
    );
    assert_eq!(once.lines().count(), trial_lines.len());
    let (mut invalid, mut changed_shape, mut not_idempotent, mut cut) = (0, 0, 0, 0);
    for ((trial, job), again) in trial_lines.iter().zip(once.lines()).zip(twice.lines()) {
        let Ok(value) = serde_json::from_str::<Value>(job) else {
            invalid += 1;
            continue;
        };
        if skeleton(&value) != skeleton(&parse(trial.trim_end())) {
            changed_shape += 1;
        }
        if job != again {
            not_idempotent += 1;
        }
        if job != trial.trim_end() {
            cut += 1;
            // Only key body lines go, each run for one marker.
            assert!(job.contains("***REDACTED***"), "{trial}\n{job}");
        }
    }
    eprintln!("10000 records: {cut} with key lines removed");
    assert_eq!(
        (invalid, changed_shape, not_idempotent),
        (0, 0, 0),
        "invalid, structure changed, changed on a second pass"
    );
    assert!(cut > 0, "the fuzz never opened a key span");
}
