//! `acp_trajectory.jsonl`: one event per line, written exactly as 0.8
//! writes it.
//!
//! 0.8 serializes with Python's `json.dumps` defaults: `", "` and `": "`
//! separators, every non-ASCII character (and DEL) escaped as `\uXXXX`,
//! floats as Python's `repr`, lines joined by `\n` with no newline after the
//! last. [`PyFormatter`] reproduces that, so a trajectory read from a 0.8
//! file and written again is byte for byte the same file.

use std::borrow::Borrow;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use serde_json::ser::{Formatter, Serializer};

use super::event::Event;
use super::redact::Redactor;

/// A `serde_json` formatter that writes like Python's `json.dumps(obj)`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PyFormatter;

impl Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        // serde_json escapes quotes, backslashes and control characters the
        // way Python does; this escapes the rest of what `ensure_ascii` does.
        let mut start = 0;
        for (i, c) in fragment.char_indices() {
            if (c as u32) < 0x7f {
                continue;
            }
            writer.write_all(&fragment.as_bytes()[start..i])?;
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) {
                write!(writer, "\\u{unit:04x}")?;
            }
            start = i + c.len_utf8();
        }
        writer.write_all(&fragment.as_bytes()[start..])
    }

    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(python_float_repr(value).as_bytes())
    }

    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        self.write_f64(writer, f64::from(value))
    }
}

/// Python's `repr(float)`: the shortest digits that round-trip, plain for
/// exponents from -4 to 15, scientific (`1e-05`, `1e+16`) otherwise.
pub fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` prints the shortest round-trip digits: `-1.2345e3`.
    let sci = format!("{value:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let point = exp + 1; // digits before the decimal point
    if -4 < point && point <= 16 {
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let tail = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!(
            "{sign}{}{tail}e{exp_sign}{:02}",
            &digits[..1],
            exp.unsigned_abs()
        )
    }
}

/// `value` as Python's `json.dumps(value)` writes it.
pub fn to_python_json(value: &Value) -> String {
    let mut out = Vec::new();
    let mut serializer = Serializer::with_formatter(&mut out, PyFormatter);
    value
        .serialize(&mut serializer)
        .expect("serializing a JSON value into memory cannot fail");
    String::from_utf8(out).expect("the formatter writes ASCII")
}

/// Events as `acp_trajectory.jsonl` text: redacted, one per line, no newline
/// after the last (0.8's `TrajectoryWriter`).
pub fn to_jsonl<'a, I>(events: I, redactor: &Redactor) -> String
where
    I: IntoIterator<Item = &'a Event>,
{
    let mut lines = Vec::new();
    for event in events {
        let mut json = event.to_json();
        redactor.redact_json(&mut json);
        lines.push(to_python_json(&json));
    }
    lines.join("\n")
}

/// A line of `acp_trajectory.jsonl` that is not a JSON object.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("line {line} of the trajectory is not a JSON object: {message}")]
pub struct JsonlError {
    pub line: usize,
    pub message: String,
}

/// Read `acp_trajectory.jsonl` text. Blank lines are skipped.
pub fn parse_jsonl(text: &str) -> Result<Vec<Event>, JsonlError> {
    let mut events = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let error = |message: String| JsonlError {
            line: i + 1,
            message,
        };
        match serde_json::from_str::<Value>(line) {
            Ok(Value::Object(map)) => events.push(Event::from_json(map)),
            Ok(_) => return Err(error("expected an object".into())),
            Err(e) => return Err(error(e.to_string())),
        }
    }
    Ok(events)
}

/// Rewrites `acp_trajectory.jsonl` as a run goes, like 0.8's writer: each
/// write replaces the file atomically (a temporary file, then a rename), so
/// a reader never sees half a line, and an unchanged trajectory is not
/// rewritten. Each event's redacted line is kept and reused while the event
/// is unchanged, so a long run is not re-redacted on every write.
#[derive(Debug)]
pub struct TrajectoryWriter {
    path: PathBuf,
    tmp: PathBuf,
    redactor: Redactor,
    /// Per event: a hash of its unredacted JSON and the line written for it.
    lines: Vec<(u64, String)>,
    last: Option<u64>,
}

fn hash_of(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

impl TrajectoryWriter {
    /// A writer for `path` that redacts with `redactor`. Creates the folder
    /// and removes a temporary file a crashed writer left.
    pub async fn new(path: impl Into<PathBuf>, redactor: Redactor) -> io::Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        match tokio::fs::remove_file(&tmp).await {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        Ok(Self {
            path,
            tmp,
            redactor,
            lines: Vec::new(),
            last: None,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write the events (all of them, as [`to_jsonl`] would) unless the
    /// result equals the last write. Returns whether the file changed.
    pub async fn write<I>(&mut self, events: I) -> io::Result<bool>
    where
        I: IntoIterator,
        I::Item: Borrow<Event>,
    {
        let mut count = 0;
        for (i, event) in events.into_iter().enumerate() {
            let mut json = event.borrow().to_json();
            let raw = hash_of(&json.to_string());
            if self.lines.get(i).is_none_or(|(hash, _)| *hash != raw) {
                self.redactor.redact_json(&mut json);
                let line = to_python_json(&json);
                if i < self.lines.len() {
                    self.lines[i] = (raw, line);
                } else {
                    self.lines.push((raw, line));
                }
            }
            count = i + 1;
        }
        self.lines.truncate(count);
        let payload = self
            .lines
            .iter()
            .map(|(_, line)| line.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let digest = hash_of(&payload);
        if self.last == Some(digest) {
            return Ok(false);
        }
        tokio::fs::write(&self.tmp, &payload).await?;
        tokio::fs::rename(&self.tmp, &self.path).await?;
        self.last = Some(digest);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn floats_read_back_exactly() {
        // serde_json's default parser can land one bit off; with
        // `float_roundtrip` this number reads as Python reads it.
        let v: serde_json::Value = serde_json::from_str("0.0009093300000000001").unwrap();
        assert_eq!(
            python_float_repr(v.as_f64().unwrap()),
            "0.0009093300000000001"
        );
    }

    #[test]
    fn floats_print_like_python_repr() {
        // Expected values from CPython 3.12's repr().
        for (value, expected) in [
            (600.0, "600.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e-7, "1.5e-07"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (-2.5, "-2.5"),
            (123456789.125, "123456789.125"),
            (1e100, "1e+100"),
            (-0.0, "-0.0"),
            (45.3, "45.3"),
        ] {
            assert_eq!(python_float_repr(value), expected, "{value}");
        }
    }

    #[test]
    fn json_matches_python_dumps() {
        let value = json!({"a": [1, 2.5, "\u{e9}\u{1f600}\u{7f}"], "b": {"c": null, "d": true}, "e": "q\"\\\n\t\u{1}"});
        // Python: json.dumps({'a': [1, 2.5, 'é😀<DEL>'], 'b': {'c': None, 'd': True}, 'e': 'q"\\\n\t<0x01>'})
        // escapes non-ASCII as UTF-16 units; the backslash is built from its
        // code point so no escape sequence in this file can be rewritten.
        let u = |hex: &str| format!("{}u{hex}", char::from(92));
        let expected = format!(
            r#"{{"a": [1, 2.5, "{}{}{}{}"], "b": {{"c": null, "d": true}}, "e": "q\"\\\n\t{}"}}"#,
            u("00e9"),
            u("d83d"),
            u("de00"),
            u("007f"),
            u("0001")
        );
        assert_eq!(to_python_json(&value), expected);
        assert_eq!(to_python_json(&json!([])), "[]");
        assert_eq!(to_python_json(&json!({})), "{}");
    }

    #[test]
    fn bad_lines_name_their_number() {
        let err = parse_jsonl("{\"type\": \"nop\", \"note\": \"x\"}\n\n[1]\n").unwrap_err();
        assert_eq!(err.line, 3);
        assert!(parse_jsonl("{oops").is_err());
    }

    #[tokio::test]
    async fn the_writer_replaces_the_file_and_skips_unchanged_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trajectory/acp_trajectory.jsonl");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path.with_extension("jsonl.tmp"), "stale").unwrap();
        let mut writer = TrajectoryWriter::new(&path, Redactor::new(["secret-login-value"]))
            .await
            .unwrap();
        assert!(!path.with_extension("jsonl.tmp").exists());
        let note = |text: &str| Event::Nop {
            note: text.to_string(),
        };
        assert!(
            writer
                .write([note("a secret-login-value"), note("b")])
                .await
                .unwrap()
        );
        assert!(
            !writer
                .write([note("a secret-login-value"), note("b")])
                .await
                .unwrap()
        );
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            written,
            "{\"type\": \"nop\", \"note\": \"a ***REDACTED***\"}\n{\"type\": \"nop\", \"note\": \"b\"}"
        );
        // Changed and dropped events are rewritten.
        assert!(writer.write([note("c")]).await.unwrap());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"type\": \"nop\", \"note\": \"c\"}"
        );
    }
}
