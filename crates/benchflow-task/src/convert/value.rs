//! Config values as task.md's reference converter sees them: Python's
//! `tomllib` (or PyYAML) output, with tables that keep insertion order.
//!
//! The converter's output must match `tools/convert.py` byte for byte, so
//! the few places where Python's formatting shows through are reproduced
//! here: `repr()` of floats, `isoformat()` of dates, and `float()` and
//! `int()` of strings.

use indexmap::IndexMap;
use toml_edit::{Item, Value as TomlValue};

/// One config value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Val {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    /// A date, time, or date-time, kept as Python's `isoformat()` writes it.
    Date(String),
    List(Vec<Val>),
    Table(Table),
    /// YAML's null; never in TOML.
    Null,
}

/// A table that keeps its keys in insertion order.
pub(crate) type Table = IndexMap<String, Val>;

/// One step of a path into a config: a key, or a list index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Seg {
    Key(String),
    Index(usize),
}

/// A path into a config, like Python's key tuples: `("steps", 0, "agent")`.
pub(crate) type KPath = Vec<Seg>;

/// Build a path from keys.
pub(crate) fn kp(keys: &[&str]) -> KPath {
    keys.iter().map(|k| Seg::Key((*k).to_string())).collect()
}

/// `path + more`.
pub(crate) fn kjoin(path: &[Seg], more: &[Seg]) -> KPath {
    let mut p = path.to_vec();
    p.extend_from_slice(more);
    p
}

/// `path + (key,)`.
pub(crate) fn kkey(path: &[Seg], key: &str) -> KPath {
    kjoin(path, &[Seg::Key(key.to_string())])
}

/// `path + (index,)`.
pub(crate) fn kidx(path: &[Seg], i: usize) -> KPath {
    kjoin(path, &[Seg::Index(i)])
}

/// Python's `'.'.join(map(str, path))`.
pub(crate) fn dotted(path: &[Seg]) -> String {
    path.iter()
        .map(|s| match s {
            Seg::Key(k) => k.clone(),
            Seg::Index(i) => i.to_string(),
        })
        .collect::<Vec<_>>()
        .join(".")
}

impl Val {
    pub(crate) fn as_table(&self) -> Option<&Table> {
        match self {
            Val::Table(t) => Some(t),
            _ => None,
        }
    }

    pub(crate) fn as_table_mut(&mut self) -> Option<&mut Table> {
        match self {
            Val::Table(t) => Some(t),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }

    pub(crate) fn as_list(&self) -> Option<&Vec<Val>> {
        match self {
            Val::List(l) => Some(l),
            _ => None,
        }
    }

    /// Python truthiness.
    pub(crate) fn truthy(&self) -> bool {
        match self {
            Val::Str(s) | Val::Date(s) => !s.is_empty(),
            Val::Int(i) => *i != 0,
            Val::Float(f) => *f != 0.0,
            Val::Bool(b) => *b,
            Val::List(l) => !l.is_empty(),
            Val::Table(t) => !t.is_empty(),
            Val::Null => false,
        }
    }

    /// Python's `float(v)`; `None` where Python raises.
    pub(crate) fn py_float(&self) -> Option<f64> {
        match self {
            Val::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Val::Int(i) => Some(*i as f64),
            Val::Float(f) => Some(*f),
            Val::Str(s) => py_float_str(s),
            _ => None,
        }
    }

    /// Python's `int(v)`; `None` where Python raises.
    pub(crate) fn py_int(&self) -> Option<i64> {
        match self {
            Val::Bool(b) => Some(i64::from(*b)),
            Val::Int(i) => Some(*i),
            Val::Float(f) if f.is_finite() => Some(f.trunc() as i64),
            Val::Str(s) => {
                let t = s.trim().replace('_', "");
                let digits = t.strip_prefix(['+', '-']).unwrap_or(&t);
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                t.parse().ok()
            }
            _ => None,
        }
    }
}

/// Python's `float(str)`: surrounding whitespace, `inf`, `nan`, and
/// underscores between digits are allowed.
pub(crate) fn py_float_str(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() || t.starts_with('_') || t.ends_with('_') || t.contains("__") {
        return None;
    }
    let t = t.replace('_', "");
    let lower = t.to_ascii_lowercase();
    let body = lower.trim_start_matches(['+', '-']);
    if matches!(body, "inf" | "infinity" | "nan") {
        return t.parse().ok();
    }
    if !body
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '+' | '-'))
    {
        return None;
    }
    t.parse().ok()
}

/// Python's `repr(float)`: the shortest digits that round-trip, written
/// in fixed notation for exponents from -4 to 15 and in scientific
/// notation otherwise (`1e+16`, `1.5e-05`).
pub(crate) fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if (point as usize) >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let mant = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mant}e{esign}{:02}", exp.abs())
    }
}

/// Python's `isoformat()` for what `tomllib` makes of a TOML date-time:
/// `Z` becomes `+00:00`, and fractions are microseconds, truncated.
pub(crate) fn isoformat(d: &toml_edit::Datetime) -> String {
    let mut out = String::new();
    if let Some(date) = d.date {
        out.push_str(&format!(
            "{:04}-{:02}-{:02}",
            date.year, date.month, date.day
        ));
    }
    if let Some(time) = d.time {
        if d.date.is_some() {
            out.push('T');
        }
        out.push_str(&format!(
            "{:02}:{:02}:{:02}",
            time.hour,
            time.minute,
            time.second.unwrap_or(0)
        ));
        let micros = time.nanosecond.unwrap_or(0) / 1000;
        if micros > 0 {
            out.push_str(&format!(".{micros:06}"));
        }
    }
    match d.offset {
        Some(toml_edit::Offset::Z) => out.push_str("+00:00"),
        Some(toml_edit::Offset::Custom { minutes }) => {
            let sign = if minutes < 0 { '-' } else { '+' };
            let m = minutes.unsigned_abs();
            out.push_str(&format!("{sign}{:02}:{:02}", m / 60, m % 60));
        }
        None => {}
    }
    out
}

/// Parse TOML text into a table, as `tomllib.loads` would.
pub(crate) fn parse_toml(text: &str) -> Result<Table, String> {
    let doc = toml_edit::Document::parse(text).map_err(|e| e.message().trim().to_string())?;
    Ok(table_of(doc.as_table()))
}

fn table_of(t: &toml_edit::Table) -> Table {
    let mut out = Table::new();
    for (k, item) in t.iter() {
        if let Some(v) = item_val(item) {
            out.insert(k.to_string(), v);
        }
    }
    out
}

fn item_val(item: &Item) -> Option<Val> {
    match item {
        Item::None => None,
        Item::Value(v) => Some(value_val(v)),
        Item::Table(t) => Some(Val::Table(table_of(t))),
        Item::ArrayOfTables(a) => Some(Val::List(
            a.iter().map(|t| Val::Table(table_of(t))).collect(),
        )),
    }
}

fn value_val(v: &TomlValue) -> Val {
    match v {
        TomlValue::String(s) => Val::Str(s.value().clone()),
        TomlValue::Integer(i) => Val::Int(*i.value()),
        TomlValue::Float(f) => Val::Float(*f.value()),
        TomlValue::Boolean(b) => Val::Bool(*b.value()),
        TomlValue::Datetime(d) => Val::Date(isoformat(d.value())),
        TomlValue::Array(a) => Val::List(a.iter().map(value_val).collect()),
        TomlValue::InlineTable(t) => {
            let mut out = Table::new();
            for (k, x) in t.iter() {
                out.insert(k.to_string(), value_val(x));
            }
            Val::Table(out)
        }
    }
}

/// A YAML value, as PyYAML's `safe_load` gives it, where the two agree.
pub(crate) fn from_yaml(v: &serde_yaml_ng::Value) -> Val {
    use serde_yaml_ng::Value as Y;
    match v {
        Y::Null => Val::Null,
        Y::Bool(b) => Val::Bool(*b),
        Y::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => Val::Int(i),
            (None, Some(f)) if n.is_f64() => Val::Float(f),
            _ => Val::Str(n.to_string()),
        },
        Y::String(s) => Val::Str(s.clone()),
        Y::Sequence(items) => Val::List(items.iter().map(from_yaml).collect()),
        Y::Mapping(m) => Val::Table(m.iter().map(|(k, x)| (yaml_key(k), from_yaml(x))).collect()),
        Y::Tagged(t) => from_yaml(&t.value),
    }
}

/// Python's `str()` of a YAML mapping key.
pub(crate) fn yaml_key(k: &serde_yaml_ng::Value) -> String {
    use serde_yaml_ng::Value as Y;
    match k {
        Y::String(s) => s.clone(),
        Y::Bool(true) => "True".into(),
        Y::Bool(false) => "False".into(),
        Y::Null => "None".into(),
        Y::Number(n) => match n.as_f64() {
            Some(f) if !n.is_i64() && !n.is_u64() => py_float_repr(f),
            _ => n.to_string(),
        },
        other => serde_yaml_ng::to_string(other)
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

/// Parse YAML text into a value; an empty document is `Null`.
pub(crate) fn parse_yaml(text: &str) -> Result<Val, String> {
    if text.trim().is_empty() {
        return Ok(Val::Null);
    }
    let v: serde_yaml_ng::Value = serde_yaml_ng::from_str(text).map_err(|e| e.to_string())?;
    Ok(from_yaml(&v))
}

/// Canonical JSON of a value, as `json.dumps(v, sort_keys=True,
/// separators=(",", ":"), ensure_ascii=False, default=str)`.
pub(crate) fn canonical_json(v: &Val) -> String {
    match v {
        Val::Str(s) => json_str(s),
        Val::Date(s) => json_str(&s.replacen('T', " ", 1)),
        Val::Int(i) => i.to_string(),
        Val::Float(f) => {
            if f.is_nan() {
                "NaN".into()
            } else if f.is_infinite() {
                if *f > 0.0 {
                    "Infinity".into()
                } else {
                    "-Infinity".into()
                }
            } else {
                py_float_repr(*f)
            }
        }
        Val::Bool(b) => b.to_string(),
        Val::Null => "null".into(),
        Val::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Val::Table(t) => {
            let mut keys: Vec<&String> = t.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", json_str(k), canonical_json(&t[k])))
                .collect();
            format!("{{{}}}", body.join(","))
        }
    }
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        for (f, s) in [
            (600.0, "600.0"),
            (0.5, "0.5"),
            (1e16, "1e+16"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (123456789012345.6, "123456789012345.6"),
            (1234567890123456.0, "1234567890123456.0"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (-2.5, "-2.5"),
            (0.1 + 0.2, "0.30000000000000004"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(py_float_repr(f), s, "{f}");
        }
    }

    #[test]
    fn python_number_parsing() {
        assert_eq!(Val::Str(" 600 ".into()).py_float(), Some(600.0));
        assert_eq!(Val::Str("1_000".into()).py_int(), Some(1000));
        assert_eq!(Val::Str("4.5".into()).py_int(), None);
        assert_eq!(Val::Float(4096.7).py_int(), Some(4096));
        assert_eq!(Val::Bool(true).py_float(), Some(1.0));
        assert_eq!(Val::Str("abc".into()).py_float(), None);
        assert_eq!(Val::List(vec![]).py_float(), None);
    }

    #[test]
    fn dates_print_like_isoformat() {
        let t = parse_toml("a = 1979-05-27T07:32:00Z\nb = 1979-05-27T00:32:00.999999999-07:00\nc = 1979-05-27\nd = 07:32:00\n").unwrap();
        assert_eq!(t["a"], Val::Date("1979-05-27T07:32:00+00:00".into()));
        assert_eq!(t["b"], Val::Date("1979-05-27T00:32:00.999999-07:00".into()));
        assert_eq!(t["c"], Val::Date("1979-05-27".into()));
        assert_eq!(t["d"], Val::Date("07:32:00".into()));
    }

    #[test]
    fn toml_keeps_document_order() {
        let t = parse_toml("z = 1\n[b]\nx = 1\n[a]\ny = 2\n[b.c]\nq = 3\n").unwrap();
        assert_eq!(t.keys().collect::<Vec<_>>(), vec!["z", "b", "a"]);
        assert_eq!(
            t["b"].as_table().unwrap().keys().collect::<Vec<_>>(),
            vec!["x", "c"]
        );
    }

    #[test]
    fn yaml_values() {
        let v = parse_yaml("a: ~\nb: [1, null, 2.5]\n3: x\n").unwrap();
        let t = v.as_table().unwrap();
        assert_eq!(t["a"], Val::Null);
        assert_eq!(
            t["b"],
            Val::List(vec![Val::Int(1), Val::Null, Val::Float(2.5)])
        );
        assert_eq!(t["3"], Val::Str("x".into()));
        assert_eq!(parse_yaml("  \n").unwrap(), Val::Null);
    }

    #[test]
    fn canonical_json_sorts_keys() {
        let t = parse_toml("b = 1.0\na = \"é\"\n").unwrap();
        assert_eq!(canonical_json(&Val::Table(t)), "{\"a\":\"é\",\"b\":1.0}");
    }
}
