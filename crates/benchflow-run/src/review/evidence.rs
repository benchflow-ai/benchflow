//! What a review sends: files with line numbers, size-capped and redacted.
//!
//! A file is sent with its line numbers (`  12| text`) so the model can
//! point at a line, and only as a whole number of lines: a long file keeps
//! its first and last lines with a marker for the lines left out, so every
//! number shown is the file's real line number. Redaction never changes how
//! many lines a file has; a file whose redaction would (a private key across
//! several lines) is not sent. Files that usually hold credentials are never
//! read.

use std::path::Path;

use benchflow_agent::trajectory::Redactor;
use serde::Serialize;

/// Most bytes of one file sent in full; a longer file is cut to its first
/// [`HEAD_LINES`] and last [`TAIL_LINES`] lines.
pub const FILE_CAP: usize = 24 * 1024;
pub const HEAD_LINES: usize = 240;
pub const TAIL_LINES: usize = 60;
/// Most bytes of files in one request; files past it are listed, not sent.
pub const TOTAL_CAP: usize = 120 * 1024;

/// A file sent to the model, as the model saw it.
#[derive(Debug, Clone)]
pub struct SentFile {
    /// Relative to the folder reviewed, with `/`.
    pub rel: String,
    /// What the file is for (`verifier`, `oracle`, ...).
    pub role: &'static str,
    /// The file's lines after redaction (all of them, sent or not).
    pub lines: Vec<String>,
    /// Lines left out of the request, as a 1-based inclusive range.
    pub omitted: Option<(usize, usize)>,
}

impl SentFile {
    /// The block the request holds: a header, then numbered lines.
    pub fn render(&self) -> String {
        let width = self.lines.len().max(1).to_string().len().max(3);
        let mut out = format!(
            "=== {} ({}, {} line{}) ===\n",
            self.rel,
            self.role,
            self.lines.len(),
            if self.lines.len() == 1 { "" } else { "s" }
        );
        for (i, line) in self.lines.iter().enumerate() {
            let n = i + 1;
            if let Some((from, to)) = self.omitted {
                if n == from {
                    out.push_str(&format!("{:>width$}| … lines {from}–{to} not sent …\n", ""));
                }
                if (from..=to).contains(&n) {
                    continue;
                }
            }
            out.push_str(&format!("{n:>width$}| {line}\n"));
        }
        out
    }

    fn sent_bytes(&self) -> usize {
        self.lines
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                self.omitted
                    .is_none_or(|(from, to)| !(from..=to).contains(&(i + 1)))
            })
            .map(|(_, l)| l.len() + 8)
            .sum()
    }
}

/// A file that was not sent, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Withheld {
    pub file: String,
    pub reason: String,
}

/// Why a file is never read: its name says it may hold credentials.
pub fn credential_file(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n == ".env"
        || n.starts_with(".env.")
        || n.ends_with(".env")
        || n == ".netrc"
        || n == ".npmrc"
        || n == ".pypirc"
        || n == ".git-credentials"
        || n.starts_with("credentials")
        || n.starts_with("id_rsa")
        || n.starts_with("id_ed25519")
        || n.starts_with("id_ecdsa")
        || [".pem", ".key", ".p12", ".pfx", ".keystore", ".jks"]
            .iter()
            .any(|ext| n.ends_with(ext))
}

/// Read `path` as text for the model: `Err` says why it is not sent.
pub fn read_for_model(path: &Path, redactor: &Redactor) -> Result<Vec<String>, String> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if credential_file(&name) {
        return Err("not sent: files like this usually hold credentials".into());
    }
    let bytes = std::fs::read(path).map_err(|e| format!("not sent: cannot read it ({e})"))?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Err(format!("not sent: {} bytes", bytes.len()));
    }
    if bytes.contains(&0) {
        return Err(format!("not sent: binary, {} bytes", bytes.len()));
    }
    let text = String::from_utf8(bytes).map_err(|_| "not sent: not UTF-8 text".to_string())?;
    redact_lines(&text, redactor)
}

/// `text`'s lines with secrets replaced, or why it cannot be sent.
pub fn redact_lines(text: &str, redactor: &Redactor) -> Result<Vec<String>, String> {
    let redacted = redactor.redact(text);
    if redacted.lines().count() != text.lines().count() {
        return Err("not sent: it holds a private key or another multi-line secret".into());
    }
    Ok(redacted.lines().map(str::to_string).collect())
}

impl SentFile {
    /// A file's lines, cut to the per-file cap.
    pub fn new(rel: String, role: &'static str, lines: Vec<String>) -> Self {
        let bytes: usize = lines.iter().map(|l| l.len() + 1).sum();
        let omitted = (bytes > FILE_CAP && lines.len() > HEAD_LINES + TAIL_LINES)
            .then(|| (HEAD_LINES + 1, lines.len() - TAIL_LINES));
        let mut file = Self {
            rel,
            role,
            lines,
            omitted,
        };
        // Very long lines: keep each one, shortened, so numbering holds.
        if file.sent_bytes() > FILE_CAP {
            let per_line = (FILE_CAP / file.lines.len().max(1)).max(80);
            for line in &mut file.lines {
                if line.len() > per_line {
                    let mut cut = per_line;
                    while !line.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    line.truncate(cut);
                    line.push_str(" …(line cut)");
                }
            }
        }
        file
    }
}

/// Files packed into one request under [`TOTAL_CAP`].
#[derive(Debug, Default)]
pub struct Bundle {
    pub sent: Vec<SentFile>,
    pub withheld: Vec<Withheld>,
    bytes: usize,
}

impl Bundle {
    /// Add a file, or list it as not sent when the request is full.
    pub fn push(&mut self, file: SentFile) {
        let size = file.sent_bytes();
        if self.bytes + size > TOTAL_CAP {
            self.withheld.push(Withheld {
                file: file.rel,
                reason: "not sent: the request was already at its size cap".into(),
            });
            return;
        }
        self.bytes += size;
        self.sent.push(file);
    }

    pub fn withhold(&mut self, file: String, reason: String) {
        self.withheld.push(Withheld { file, reason });
    }

    pub fn find(&self, rel: &str) -> Option<&SentFile> {
        self.sent.iter().find(|f| f.rel == rel)
    }

    /// Every sent file's block, then the list of files not sent.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for f in &self.sent {
            out.push_str(&f.render());
            out.push('\n');
        }
        if !self.withheld.is_empty() {
            out.push_str("=== files not sent ===\n");
            for w in &self.withheld {
                out.push_str(&format!("- {} ({})\n", w.file, w.reason));
            }
        }
        out
    }
}

/// The last `max` bytes of `text`, starting at a line, with a marker when
/// cut.
pub fn tail(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    let rest = &text[start..];
    let rest = rest.split_once('\n').map_or(rest, |(_, r)| r);
    format!("…(earlier output not sent)\n{rest}")
}

/// The first `head` and last `tail_bytes` bytes of `text`, with a marker
/// between when cut.
pub fn head_tail(text: &str, head: usize, tail_bytes: usize) -> String {
    if text.len() <= head + tail_bytes {
        return text.to_string();
    }
    let mut end = head;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n…(middle not sent)…\n{}",
        &text[..end],
        tail(&text[end..], tail_bytes).trim_start_matches("…(earlier output not sent)\n")
    )
}

/// `text` collapsed to single spaces, for comparing quotes.
pub fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_files_keep_their_line_numbers() {
        let lines: Vec<String> = (1..=1000)
            .map(|i| format!("line {i} {}", "x".repeat(40)))
            .collect();
        let f = SentFile::new("big.txt".into(), "other", lines);
        assert_eq!(f.omitted, Some((HEAD_LINES + 1, 1000 - TAIL_LINES)));
        let text = f.render();
        assert!(text.contains(" 240| line 240 "));
        assert!(text.contains("lines 241–940 not sent"));
        assert!(text.contains(" 941| line 941 "));
        assert!(!text.contains("| line 500 "));
    }

    #[test]
    fn secrets_are_replaced_and_key_files_are_never_read() {
        let r = Redactor::new(["host-secret-value-123"]);
        let lines = redact_lines(
            "a\nOPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwx\nuse host-secret-value-123\n",
            &r,
        )
        .unwrap();
        assert_eq!(lines.len(), 3);
        assert!(!lines.join("\n").contains("abcdefghijklmnop"));
        assert!(!lines.join("\n").contains("host-secret-value-123"));
        for name in [
            ".env",
            "prod.env",
            ".env.local",
            "id_rsa",
            "server.pem",
            ".npmrc",
        ] {
            assert!(credential_file(name), "{name}");
        }
        assert!(!credential_file("environment.md"));
        assert!(!credential_file("Dockerfile"));
    }

    #[test]
    fn a_multi_line_key_withholds_the_file() {
        let pem = "x\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----\ny\n";
        let r = Redactor::default();
        match redact_lines(pem, &r) {
            Err(why) => assert!(why.contains("private key")),
            Ok(lines) => assert!(!lines.join("\n").contains("MIIEvQIBADAN")),
        }
    }

    #[test]
    fn the_total_cap_lists_what_did_not_fit() {
        let mut b = Bundle::default();
        for i in 0..20 {
            let lines = vec!["y".repeat(200); 100];
            b.push(SentFile::new(format!("f{i}"), "other", lines));
        }
        assert!(!b.withheld.is_empty());
        assert!(
            b.render()
                .contains("the request was already at its size cap")
        );
    }
}
