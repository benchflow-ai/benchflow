//! JSON as Python's `json` module reads and writes it.
//!
//! The 0.8 viewer reads trajectories with `json.loads` and writes payloads
//! with `json.dumps`, and some payload fields embed `json.dumps` text
//! (unknown events, diagnostics, structured tool output). To produce the
//! same bytes, this module follows Python where it differs from strict JSON:
//!
//! - `NaN`, `Infinity` and `-Infinity` are accepted on input.
//! - Integers keep every digit ([`Int::Big`]).
//! - A lone UTF-16 surrogate escape such as `"\ud800"` becomes `?`, which is
//!   what the 0.8 server sends for it after its final UTF-8 encoding.
//! - Output uses `", "` and `": "` separators, leaves non-ASCII characters
//!   as they are, formats floats like Python's `repr`, and writes
//!   non-finite floats as `null` (0.8's `dumps_finite`).
//! - Containers nested [`MAX_DEPTH`] or more levels deep are refused, like
//!   0.8's `_parse_json`.

use std::fmt::Write as _;

use indexmap::IndexMap;

/// Containers at this depth or deeper make a document unreadable. The
/// outermost container is at depth 0 (0.8's `_MAX_JSON_DEPTH`).
pub const MAX_DEPTH: usize = 64;

/// A JSON object with its keys in document order.
pub type Map = IndexMap<String, Value>;

/// A JSON value as Python holds it after `json.loads`.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(Int),
    /// A float; non-finite after `NaN`, `Infinity`, or an overflowing literal.
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Map),
}

/// A Python `int`: exact at any size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Int {
    Small(i64),
    /// Decimal digits, with a leading `-` when negative, for values outside `i64`.
    Big(String),
}

static NULL: Value = Value::Null;

impl Value {
    /// The value under `key` when this is an object, else `null`: Python's
    /// `mapping.get(key)` (a missing key and a `null` value read the same).
    pub fn get(&self, key: &str) -> &Value {
        match self {
            Value::Object(map) => map.get(key).unwrap_or(&NULL),
            _ => &NULL,
        }
    }

    pub fn as_object(&self) -> Option<&Map> {
        match self {
            Value::Object(map) => Some(map),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(text) => Some(text),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

/// `mapping.get(key)` for an optional mapping: `null` when there is none.
pub fn get<'a>(map: &'a Map, key: &str) -> &'a Value {
    map.get(key).unwrap_or(&NULL)
}

/// An object from `(key, value)` pairs, keys in the order given.
pub fn object<const N: usize>(fields: [(&str, Value); N]) -> Value {
    Value::Object(
        fields
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

impl From<bool> for Value {
    fn from(flag: bool) -> Self {
        Value::Bool(flag)
    }
}

impl From<String> for Value {
    fn from(text: String) -> Self {
        Value::Str(text)
    }
}

impl From<&str> for Value {
    fn from(text: &str) -> Self {
        Value::Str(text.to_string())
    }
}

impl From<f64> for Value {
    fn from(number: f64) -> Self {
        Value::Float(number)
    }
}

impl From<i64> for Value {
    fn from(number: i64) -> Self {
        Value::Int(Int::Small(number))
    }
}

impl From<usize> for Value {
    fn from(number: usize) -> Self {
        Value::Int(Int::Small(i64::try_from(number).unwrap_or(i64::MAX)))
    }
}

impl From<Int> for Value {
    fn from(int: Int) -> Self {
        Value::Int(int)
    }
}

impl From<Vec<Value>> for Value {
    fn from(items: Vec<Value>) -> Self {
        Value::Array(items)
    }
}

impl From<Vec<String>> for Value {
    fn from(items: Vec<String>) -> Self {
        Value::Array(items.into_iter().map(Value::Str).collect())
    }
}

impl From<Map> for Value {
    fn from(map: Map) -> Self {
        Value::Object(map)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(value: Option<T>) -> Self {
        value.map_or(Value::Null, Into::into)
    }
}

impl Int {
    /// Read JSON integer text (`-?(0|[1-9][0-9]*)`).
    fn from_json(text: &str) -> Int {
        match text.parse::<i64>() {
            Ok(value) => Int::Small(value),
            Err(_) => Int::Big(text.to_string()),
        }
    }

    /// The nearest float, as Python's `float(int)` computes it; `None` when
    /// it overflows (Python raises `OverflowError`).
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            // `as` rounds to the nearest float, ties to even, like Python.
            Int::Small(value) => Some(*value as f64),
            Int::Big(digits) => digits.parse::<f64>().ok().filter(|value| value.is_finite()),
        }
    }

    /// The exact integer value of a finite, integral float: Python's `int(x)`.
    pub fn from_integral_f64(value: f64) -> Int {
        debug_assert!(value.is_finite() && value.fract() == 0.0);
        if value.abs() < 9.0e18 {
            return Int::Small(value as i64);
        }
        // Every float this large is normal: value = mantissa * 2^exponent.
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as u32 - 1075;
        let mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
        let mut digits = decimal_shifted(mantissa, exponent);
        if value < 0.0 {
            digits.insert(0, '-');
        }
        Int::Big(digits)
    }

    /// Whether this is the integer `n` (Python's `value == n`).
    pub fn is(&self, n: i64) -> bool {
        matches!(self, Int::Small(value) if *value == n)
    }
}

impl std::fmt::Display for Int {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Int::Small(value) => write!(f, "{value}"),
            Int::Big(digits) => f.write_str(digits),
        }
    }
}

/// `mantissa * 2^shift` in decimal.
fn decimal_shifted(mantissa: u64, shift: u32) -> String {
    const BASE: u64 = 1_000_000_000;
    // Little-endian limbs in base 10^9.
    let mut limbs = Vec::new();
    let mut rest = mantissa;
    while rest > 0 {
        limbs.push(rest % BASE);
        rest /= BASE;
    }
    for _ in 0..shift {
        let mut carry = 0;
        for limb in &mut limbs {
            let doubled = *limb * 2 + carry;
            *limb = doubled % BASE;
            carry = doubled / BASE;
        }
        if carry > 0 {
            limbs.push(carry);
        }
    }
    let mut out = String::new();
    for (index, limb) in limbs.iter().rev().enumerate() {
        if index == 0 {
            let _ = write!(out, "{limb}");
        } else {
            let _ = write!(out, "{limb:09}");
        }
    }
    out
}

/// The decimal digits and exponent Python's `repr` picks for a finite,
/// non-negative float: the fewest digits that read back to the same float,
/// and of those the closest, with an exact tie going to the even digit.
///
/// Rust's `{:e}` finds the fewest digits but settles a tie by rounding up,
/// so when the correctly rounded string of that length also reads back,
/// that one is Python's.
fn shortest_digits(magnitude: f64) -> (String, i32) {
    let split = |text: &str| -> (String, i32) {
        let (mantissa, exponent) = text
            .split_once('e')
            .expect("`{:e}` always writes an exponent");
        let digits = mantissa.chars().filter(|ch| *ch != '.').collect();
        (
            digits,
            exponent.parse().expect("`{:e}` writes an integer exponent"),
        )
    };
    let shortest = split(&format!("{magnitude:e}"));
    let rounded = format!("{magnitude:.*e}", shortest.0.len() - 1);
    if rounded.parse::<f64>() == Ok(magnitude) {
        split(&rounded)
    } else {
        shortest
    }
}

/// A float as Python's `repr` writes it: the shortest digits that read back
/// to the same float, in fixed notation from 1e-4 up to 1e16 and in
/// exponent notation (`1e+16`, `1e-05`) outside that range.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let (digits, exponent) = shortest_digits(value.abs());
    // The decimal point sits after `point` digits.
    let point = exponent + 1;
    let mut out = String::new();
    if value.is_sign_negative() {
        out.push('-');
    }
    if !(-4 < point && point <= 16) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let shown = point - 1;
        let _ = write!(
            out,
            "e{}{:02}",
            if shown < 0 { '-' } else { '+' },
            shown.abs()
        );
    } else if point <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', point.unsigned_abs() as usize));
        out.push_str(&digits);
    } else {
        let point = point as usize;
        if digits.len() <= point {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', point - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point]);
            out.push('.');
            out.push_str(&digits[point..]);
        }
    }
    out
}

// ── reading ─────────────────────────────────────────────────────────────

/// Read one JSON document as `json.loads` would; `None` when it is
/// malformed or nested [`MAX_DEPTH`] or more containers deep.
pub fn parse(text: &str) -> Option<Value> {
    let mut parser = Parser {
        text,
        bytes: text.as_bytes(),
        pos: 0,
    };
    parser.skip_whitespace();
    let value = parser.value(0)?;
    parser.skip_whitespace();
    (parser.pos == parser.bytes.len()).then_some(value)
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn literal(&mut self, word: &str) -> Option<()> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Some(())
        } else {
            None
        }
    }

    fn value(&mut self, depth: usize) -> Option<Value> {
        match self.peek()? {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b'"' => self.string().map(Value::Str),
            b'n' => self.literal("null").map(|()| Value::Null),
            b't' => self.literal("true").map(|()| Value::Bool(true)),
            b'f' => self.literal("false").map(|()| Value::Bool(false)),
            b'N' => self.literal("NaN").map(|()| Value::Float(f64::NAN)),
            b'I' => self
                .literal("Infinity")
                .map(|()| Value::Float(f64::INFINITY)),
            b'-' if self.peek_at(1) == Some(b'I') => self
                .literal("-Infinity")
                .map(|()| Value::Float(f64::NEG_INFINITY)),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn object(&mut self, depth: usize) -> Option<Value> {
        if depth >= MAX_DEPTH {
            return None;
        }
        self.pos += 1;
        let mut map = Map::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Some(Value::Object(map));
        }
        loop {
            if self.peek() != Some(b'"') {
                return None;
            }
            let key = self.string()?;
            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return None;
            }
            self.pos += 1;
            self.skip_whitespace();
            let value = self.value(depth + 1)?;
            // A repeated key keeps its first position and takes the last
            // value, like a Python dict.
            map.insert(key, value);
            self.skip_whitespace();
            match self.peek()? {
                b'}' => {
                    self.pos += 1;
                    return Some(Value::Object(map));
                }
                b',' => {
                    self.pos += 1;
                    self.skip_whitespace();
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self, depth: usize) -> Option<Value> {
        if depth >= MAX_DEPTH {
            return None;
        }
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Some(Value::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.peek()? {
                b']' => {
                    self.pos += 1;
                    return Some(Value::Array(items));
                }
                b',' => {
                    self.pos += 1;
                    self.skip_whitespace();
                }
                _ => return None,
            }
        }
    }

    /// A number, matched exactly as CPython's scanner matches one: a `.`
    /// must be followed by a digit, and an exponent without digits is not
    /// part of the number.
    fn number(&mut self) -> Option<Value> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek()? {
            b'0' => self.pos += 1,
            b'1'..=b'9' => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return None,
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') && matches!(self.peek_at(1), Some(b'0'..=b'9')) {
            is_float = true;
            self.pos += 2;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let exponent_start = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let digits_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos > digits_start {
                is_float = true;
            } else {
                self.pos = exponent_start;
            }
        }
        let text = &self.text[start..self.pos];
        Some(if is_float {
            // Overflow reads as an infinity, as Python's `float()` does.
            Value::Float(text.parse().ok()?)
        } else {
            Value::Int(Int::from_json(text))
        })
    }

    /// A string; the cursor is on the opening quote.
    fn string(&mut self) -> Option<String> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let run = self.pos;
            while let Some(byte) = self.peek() {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            // Runs end at ASCII bytes, so both ends are character boundaries.
            out.push_str(&self.text[run..self.pos]);
            match self.peek()? {
                b'"' => {
                    self.pos += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.pos += 1;
                    self.escape(&mut out)?;
                }
                // A raw control character (Python's strict mode).
                _ => return None,
            }
        }
    }

    /// One escape after its backslash.
    fn escape(&mut self, out: &mut String) -> Option<()> {
        let byte = self.peek()?;
        self.pos += 1;
        let ch = match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => return self.unicode_escape(out),
            _ => return None,
        };
        out.push(ch);
        Some(())
    }

    fn unicode_escape(&mut self, out: &mut String) -> Option<()> {
        let unit = self.hex4()?;
        let ch = match unit {
            0xD800..=0xDBFF => {
                // A high surrogate pairs with an immediately following
                // `\uDC00`-`\uDFFF` escape. Python reads the next escape's
                // four characters as hex whatever they are (an error when
                // they are not), and reads them again as their own escape
                // when they are not a low surrogate.
                if self.peek() == Some(b'\\') && self.peek_at(1) == Some(b'u') {
                    let save = self.pos;
                    self.pos += 2;
                    let low = self.hex4()?;
                    if (0xDC00..=0xDFFF).contains(&low) {
                        char::from_u32(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00))
                    } else {
                        self.pos = save;
                        None
                    }
                } else {
                    None
                }
            }
            0xDC00..=0xDFFF => None,
            _ => char::from_u32(unit),
        };
        // A lone surrogate cannot be UTF-8; 0.8 writes it as `?`.
        out.push(ch.unwrap_or('?'));
        Some(())
    }

    fn hex4(&mut self) -> Option<u32> {
        let digits = self.bytes.get(self.pos..self.pos + 4)?;
        let mut unit = 0;
        for digit in digits {
            unit = unit * 16 + (*digit as char).to_digit(16)?;
        }
        self.pos += 4;
        Some(unit)
    }
}

// ── writing ─────────────────────────────────────────────────────────────

/// `json.dumps(value, ensure_ascii=False)` with non-finite floats as `null`.
pub fn dumps(value: &Value) -> String {
    let mut writer = Writer::new();
    writer.value(value);
    writer.into_string()
}

/// Sends finished output somewhere; returns `false` to stop writing (the
/// reader went away).
pub type Flush<'a> = dyn FnMut(String) -> bool + 'a;

/// Writes JSON text the way `json.dumps(..., ensure_ascii=False)` does.
///
/// A writer either collects everything in memory ([`Writer::new`]) or hands
/// finished chunks to a callback as it goes ([`Writer::streaming`]), so a
/// large payload never has to sit in memory twice.
pub struct Writer<'a> {
    buf: String,
    script_safe: bool,
    flush: Option<&'a mut Flush<'a>>,
    open: bool,
}

/// A streaming writer flushes once its buffer holds this many bytes.
const CHUNK: usize = 64 * 1024;

impl Default for Writer<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Writer<'a> {
    pub fn new() -> Self {
        Writer {
            buf: String::new(),
            script_safe: false,
            flush: None,
            open: true,
        }
    }

    pub fn streaming(flush: &'a mut Flush<'a>) -> Self {
        Writer {
            buf: String::new(),
            script_safe: false,
            flush: Some(flush),
            open: true,
        }
    }

    /// Also escape `&`, `<`, `>`, U+2028 and U+2029 inside strings, so the
    /// text can sit in an HTML `<script>` element (0.8's `_json_for_script`).
    pub fn set_script_safe(&mut self, on: bool) {
        self.script_safe = on;
    }

    /// Append text as it is.
    pub fn raw(&mut self, text: &str) {
        if self.open {
            self.buf.push_str(text);
            self.maybe_flush();
        }
    }

    pub fn value(&mut self, value: &Value) {
        if !self.open {
            return;
        }
        match value {
            Value::Null => self.buf.push_str("null"),
            Value::Bool(true) => self.buf.push_str("true"),
            Value::Bool(false) => self.buf.push_str("false"),
            Value::Int(int) => {
                let _ = write!(self.buf, "{int}");
            }
            Value::Float(number) if number.is_finite() => self.buf.push_str(&float_repr(*number)),
            Value::Float(_) => self.buf.push_str("null"),
            Value::Str(text) => self.string(text),
            Value::Array(items) => {
                self.buf.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        self.buf.push_str(", ");
                    }
                    self.value(item);
                    self.maybe_flush();
                }
                self.buf.push(']');
            }
            Value::Object(map) => {
                self.buf.push('{');
                for (index, (key, item)) in map.iter().enumerate() {
                    if index > 0 {
                        self.buf.push_str(", ");
                    }
                    self.string(key);
                    self.buf.push_str(": ");
                    self.value(item);
                    self.maybe_flush();
                }
                self.buf.push('}');
            }
        }
    }

    fn string(&mut self, text: &str) {
        self.buf.push('"');
        let mut start = 0;
        for (index, ch) in text.char_indices() {
            let escaped = match ch {
                '"' => "\\\"",
                '\\' => "\\\\",
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                '\u{8}' => "\\b",
                '\u{c}' => "\\f",
                '\u{0}'..='\u{1f}' => "",
                '&' if self.script_safe => "\\u0026",
                '<' if self.script_safe => "\\u003c",
                '>' if self.script_safe => "\\u003e",
                '\u{2028}' if self.script_safe => "\\u2028",
                '\u{2029}' if self.script_safe => "\\u2029",
                _ => continue,
            };
            self.buf.push_str(&text[start..index]);
            if escaped.is_empty() {
                let _ = write!(self.buf, "\\u{:04x}", ch as u32);
            } else {
                self.buf.push_str(escaped);
            }
            start = index + ch.len_utf8();
        }
        self.buf.push_str(&text[start..]);
        self.buf.push('"');
    }

    fn maybe_flush(&mut self) {
        if self.buf.len() < CHUNK {
            return;
        }
        if let Some(flush) = self.flush.as_mut() {
            let chunk = std::mem::take(&mut self.buf);
            self.open = flush(chunk);
        }
    }

    /// Hand over what is left; returns whether the reader is still there.
    pub fn finish(mut self) -> bool {
        if let Some(flush) = self.flush.as_mut()
            && self.open
            && !self.buf.is_empty()
        {
            let chunk = std::mem::take(&mut self.buf);
            self.open = flush(chunk);
        }
        self.open
    }

    /// The text written so far (for an in-memory writer).
    pub fn into_string(self) -> String {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(text: &str) -> Option<String> {
        parse(text).map(|value| dumps(&value))
    }

    #[test]
    fn writes_python_separators_and_keeps_key_order() {
        assert_eq!(
            round_trip(r#"{"b":1,"a":[1,2.5,{}],"c":{}}"#).unwrap(),
            r#"{"b": 1, "a": [1, 2.5, {}], "c": {}}"#
        );
    }

    #[test]
    fn repeated_keys_keep_the_first_position_and_last_value() {
        assert_eq!(
            round_trip(r#"{"a":1,"b":2,"a":3}"#).unwrap(),
            r#"{"a": 3, "b": 2}"#
        );
    }

    #[test]
    fn non_finite_numbers_read_and_write_as_null() {
        assert_eq!(
            round_trip("[NaN, Infinity, -Infinity, 1e400]").unwrap(),
            "[null, null, null, null]"
        );
        assert_eq!(parse("-NaN"), None);
    }

    #[test]
    fn big_integers_keep_every_digit() {
        assert_eq!(
            round_trip("[123456789012345678901234567890, -0]").unwrap(),
            "[123456789012345678901234567890, 0]"
        );
    }

    #[test]
    fn numbers_follow_the_cpython_scanner() {
        for bad in [
            "01",
            "1.",
            "1e",
            "1.5e+",
            "-",
            "+1",
            ".5",
            "[1,]",
            "{\"a\":1,}",
        ] {
            assert_eq!(parse(bad), None, "{bad}");
        }
        assert_eq!(round_trip("1E5").unwrap(), "100000.0");
        assert_eq!(round_trip("-1e-400").unwrap(), "-0.0");
    }

    #[test]
    fn lone_surrogates_become_question_marks() {
        assert_eq!(parse("\"a\\ud800b\""), Some(Value::Str("a?b".into())));
        assert_eq!(
            parse("\"\\ud83d\\ude00\""),
            Some(Value::Str("\u{1f600}".into()))
        );
        assert_eq!(parse("\"\\ud800\\u0041\""), Some(Value::Str("?A".into())));
        assert_eq!(parse(r#""\ud800\uZZZZ""#), None);
    }

    #[test]
    fn strings_escape_like_python() {
        let value = Value::Str("q\" b\\ n\n t\t \u{1} \u{7f} é \u{2028} <&>".into());
        assert_eq!(
            dumps(&value),
            "\"q\\\" b\\\\ n\\n t\\t \\u0001 \u{7f} é \u{2028} <&>\""
        );
        let mut writer = Writer::new();
        writer.set_script_safe(true);
        writer.value(&value);
        assert_eq!(
            writer.into_string(),
            "\"q\\\" b\\\\ n\\n t\\t \\u0001 \u{7f} é \\u2028 \\u003c\\u0026\\u003e\""
        );
    }

    #[test]
    fn control_characters_and_a_byte_order_mark_are_errors() {
        assert_eq!(parse("\"\u{1}\""), None);
        assert_eq!(parse("\u{feff}{}"), None);
        assert!(parse("\"\u{7f}\"").is_some());
    }

    #[test]
    fn depth_is_bounded_like_0_8() {
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert!(parse(&ok).is_some());
        assert_eq!(parse(&deep), None);
        let very_deep = "[".repeat(100_000);
        assert_eq!(parse(&very_deep), None);
    }

    #[test]
    fn floats_print_like_python_repr() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
            (123.456, "123.456"),
        ];
        for (value, expected) in cases {
            assert_eq!(float_repr(value), expected);
        }
    }

    #[test]
    fn integral_floats_convert_exactly() {
        assert_eq!(Int::from_integral_f64(3.0), Int::Small(3));
        assert_eq!(
            Int::from_integral_f64(1e20).to_string(),
            "100000000000000000000"
        );
        assert_eq!(
            Int::from_integral_f64(1e25).to_string(),
            "10000000000000000905969664"
        );
        assert_eq!(
            Int::from_integral_f64(-1e20).to_string(),
            "-100000000000000000000"
        );
    }

    #[test]
    fn streaming_writer_hands_over_chunks_and_stops_when_told() {
        let big = Value::Array(vec![Value::Str("x".repeat(1000)); 200]);
        let mut chunks = Vec::new();
        let mut flush = |chunk: String| {
            chunks.push(chunk);
            true
        };
        let mut writer = Writer::streaming(&mut flush);
        writer.value(&big);
        assert!(writer.finish());
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), dumps(&big));

        let mut calls = 0;
        let mut refuse = |_chunk: String| {
            calls += 1;
            false
        };
        let mut writer = Writer::streaming(&mut refuse);
        writer.value(&big);
        assert!(!writer.finish());
        assert_eq!(calls, 1);
    }
}
