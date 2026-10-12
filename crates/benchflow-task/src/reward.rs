//! Reward files: what a verifier writes to `/logs/verifier/` when no rubric
//! computes the reward (as in Harbor, and BenchFlow 0.8).
//!
//! `reward.json` is preferred: a flat object of finite numbers, or an
//! envelope with a numeric `reward` and other keys. `reward.txt` holds one
//! number. When both exist, the JSON's `reward` must equal the text's.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Value};

/// Which file the reward came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RewardSource {
    /// `reward.json`.
    Json,
    /// `reward.txt`.
    Text,
    /// `reward.json` and `reward.txt`, which agree.
    Both,
}

/// A trial's rewards.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rewards {
    /// The scalar reward.
    pub reward: f64,
    /// Every number the verifier wrote, `reward` included.
    pub values: BTreeMap<String, f64>,
    /// Which file the reward came from.
    pub source: RewardSource,
    /// Other keys of an envelope (`{"reward": 1, "details": {...}}`), as
    /// written.
    pub extra: Map<String, Value>,
}

/// Why no reward could be read.
#[derive(Debug, Clone, PartialEq)]
pub enum RewardError {
    /// Neither `reward.json` nor `reward.txt` exists.
    NotFound { dir: PathBuf },
    /// The file exists and is empty.
    Empty { file: PathBuf },
    /// The file is not a reward: not a number, not finite, or not the JSON
    /// a reward file holds.
    Parse { file: PathBuf, detail: String },
    /// `reward.json`'s `reward` differs from `reward.txt`.
    Disagree { json: f64, text: f64 },
}

impl RewardError {
    /// What the task's author can do about it.
    pub fn hint(&self) -> &'static str {
        match self {
            RewardError::NotFound { .. } => {
                "make the verifier write /logs/verifier/reward.txt (one number) or reward.json (an object of numbers), or give the \
task a rubric so the reward is computed"
            }
            RewardError::Empty { .. } => {
                "the verifier created the file but wrote nothing: write the reward, such as 1 or 0.5"
            }
            RewardError::Parse { .. } => {
                "reward.txt holds one number from 0 to 1; reward.json holds an object of finite numbers, or a numeric \"reward\" \
(from 0 to 1) beside other keys"
            }
            RewardError::Disagree { .. } => {
                "write the same reward to both files, or only one of them"
            }
        }
    }
}

impl fmt::Display for RewardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RewardError::NotFound { dir } => write!(
                f,
                "no reward file in {}: neither reward.json nor reward.txt",
                dir.display()
            ),
            RewardError::Empty { file } => write!(f, "{} is empty", file.display()),
            RewardError::Parse { file, detail } => {
                write!(f, "{} is not a reward: {detail}", file.display())
            }
            RewardError::Disagree { json, text } => {
                write!(
                    f,
                    "reward.json says reward = {json} and reward.txt says {text}"
                )
            }
        }
    }
}

impl std::error::Error for RewardError {}

/// Whether two rewards are the same (within 1e-9, as BenchFlow 0.8 checks).
fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9
}

fn read_nonempty(file: &Path) -> Result<Option<String>, RewardError> {
    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(RewardError::Parse {
                file: file.to_path_buf(),
                detail: format!("it cannot be read ({e})"),
            });
        }
    };
    let text = String::from_utf8(bytes).map_err(|_| RewardError::Parse {
        file: file.to_path_buf(),
        detail: "it is not UTF-8 text".into(),
    })?;
    if text.trim().is_empty() {
        return Err(RewardError::Empty {
            file: file.to_path_buf(),
        });
    }
    Ok(Some(text))
}

/// `reward.txt`: one finite number.
fn parse_text(file: &Path, text: &str) -> Result<f64, RewardError> {
    let t = text.trim();
    match t.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        Ok(v) => Err(RewardError::Parse {
            file: file.to_path_buf(),
            detail: format!("{v} is not a finite number"),
        }),
        Err(_) => Err(RewardError::Parse {
            file: file.to_path_buf(),
            detail: format!(
                "{:?} is not a number",
                t.chars().take(40).collect::<String>()
            ),
        }),
    }
}

/// What `reward.json` holds.
struct JsonRewards {
    /// Its `reward`, when it has one.
    reward: Option<f64>,
    /// Every number in it.
    values: BTreeMap<String, f64>,
    /// The other keys of an envelope.
    extra: Map<String, Value>,
}

/// What a Python `NaN`, `Infinity` or `-Infinity` is read as at first: a
/// string no verifier writes, so where it stood can be told afterwards.
const NON_FINITE: &str = "\u{0}BenchFlow non-finite ";

/// `text` with Python's `NaN`, `Infinity` and `-Infinity` outside strings
/// (which 0.8's `json.loads` reads) written as [`NON_FINITE`] strings;
/// `None` when it holds none. [`place_non_finite`] then refuses each one
/// where 0.8 range-checks a number and nulls the rest.
fn python_constants_marked(text: &str) -> Option<String> {
    const WORDS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let (mut in_string, mut escaped, mut last, mut i) = (false, false, 0, 0);
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
        } else if b == b'"' {
            in_string = true;
        } else if let Some(word) = WORDS.iter().find(|w| bytes[i..].starts_with(w.as_bytes())) {
            out.push_str(&text[last..i]);
            out.push_str(&Value::String(format!("{NON_FINITE}{word}")).to_string());
            i += word.len();
            last = i;
            continue;
        }
        i += 1;
    }
    (last > 0).then(|| {
        out.push_str(&text[last..]);
        out
    })
}

/// The constant a [`python_constants_marked`] string stands for.
fn non_finite(v: &Value) -> Option<&str> {
    v.as_str()?.strip_prefix(NON_FINITE)
}

/// Refuses a Python constant where 0.8's `validate_reward_map` range-checks
/// a number (strict 0.8 refuses it there: `reward`, a top-level key outside
/// [`UNCHECKED_KEYS`], a `metrics` entry, a `rubric` item's `score`), and
/// writes the others as `null`, as 0.8's writers scrub them; 0.8 passes
/// those through (`{"details": {"x": NaN}}`).
fn place_non_finite(map: &mut Map<String, Value>) -> Result<(), String> {
    fn nulled(v: &mut Value) {
        match v {
            Value::String(_) if non_finite(v).is_some() => *v = Value::Null,
            Value::Array(items) => items.iter_mut().for_each(nulled),
            Value::Object(map) => map.values_mut().for_each(nulled),
            _ => {}
        }
    }
    let refused = |key: String, c: &str| {
        Err(format!(
            "{key} is {c}, not a number in [0, 1] (BenchFlow 0.8 refuses it too)"
        ))
    };
    for (key, value) in map.iter_mut() {
        if key == "reward" || !UNCHECKED_KEYS.contains(&key.as_str()) {
            if let Some(c) = non_finite(value) {
                return refused(key.clone(), c);
            }
        } else if key == "metrics"
            && let Value::Object(metrics) = &*value
            && let Some((k, c)) = metrics.iter().find_map(|(k, v)| Some((k, non_finite(v)?)))
        {
            return refused(format!("metrics.{k}"), c);
        } else if key == "rubric"
            && let Value::Array(items) = &*value
            && let Some((i, c)) = items
                .iter()
                .enumerate()
                .find_map(|(i, item)| Some((i, non_finite(item.get("score")?)?)))
        {
            return refused(format!("rubric[{i}].score"), c);
        }
        nulled(value);
    }
    Ok(())
}

/// `reward.json`: its numbers, its `reward` if it has one, and the other
/// keys of an envelope.
fn parse_json(file: &Path, text: &str) -> Result<JsonRewards, RewardError> {
    let err = |detail: String| RewardError::Parse {
        file: file.to_path_buf(),
        detail,
    };
    let value: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(e) => {
            let mut value: Value = python_constants_marked(text)
                .and_then(|t| serde_json::from_str(&t).ok())
                .ok_or_else(|| err(format!("it is not JSON ({e})")))?;
            if let Value::Object(map) = &mut value {
                place_non_finite(map).map_err(err)?;
            }
            value
        }
    };
    let Value::Object(map) = value else {
        return Err(err("it is not an object of numbers".into()));
    };
    let reward = match map.get("reward") {
        None => None,
        Some(Value::Number(n)) => Some(
            n.as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| err("reward is not finite".into()))?,
        ),
        Some(other) => return Err(err(format!("reward is {other}, not a number"))),
    };
    let mut values = BTreeMap::new();
    let mut extra = Map::new();
    for (k, v) in &map {
        match v {
            Value::Number(n) => match n.as_f64().filter(|x| x.is_finite()) {
                Some(x) => {
                    values.insert(k.clone(), x);
                }
                None => return Err(err(format!("{k} is not a finite number"))),
            },
            // A flat reward map holds only numbers; an envelope, with a
            // numeric reward, may hold anything beside it.
            other if reward.is_some() => {
                extra.insert(k.clone(), other.clone());
            }
            _ if map.contains_key("metrics")
                && map.get("aggregate").is_some_and(Value::is_object) =>
            {
                return Err(err(
                    "it has metrics and an aggregate policy but no reward key: BenchFlow 0.8 computes the reward from them, which bench does not do yet; have the verifier write the reward too".into(),
                ));
            }
            other => {
                return Err(err(format!(
                    "{k} is {other}, not a number, and there is no numeric reward key"
                )));
            }
        }
    }
    Ok(JsonRewards {
        reward,
        values,
        extra,
    })
}

/// Read the rewards in a downloaded `/logs/verifier` folder.
///
/// `reward.json` is preferred; its scalar reward is `reward`, else
/// `reward.txt`'s number. Without either, the file is refused, as BenchFlow
/// 0.8's default (strict) mode refuses it; [`read_rewards_with`] can take
/// `score` or `rewards` in its place, as 0.8's lenient mode does. When both files exist, the
/// JSON's `reward` must equal the text's. The reward must be in [0, 1], and
/// so must the other numbers 0.8's `validate_reward_map` range-checks: a
/// top-level number under any key but its structured ones
/// ([`UNCHECKED_KEYS`]), each number in `metrics`, and each `rubric` item's
/// `score`. A task that declares a wider `[verifier] reward_range` is read
/// with [`read_rewards_in`], which checks those numbers against its range
/// instead (a rubric item's `score` stays in [0, 1], as in 0.8).
///
/// Only that numeric rule is 0.8's. Its default (strict) mode also refuses
/// non-numbers that bench accepts beside a numeric reward (`"done": true`, a
/// string in `metrics`, a `space` it does not know); its lenient mode
/// (`BENCHFLOW_REWARD_LENIENT=1`) drops an out-of-range number and keeps the
/// trial, where bench refuses the file. Python's `NaN` and `Infinity`, which
/// 0.8's `json.loads` reads, are refused where 0.8 range-checks a number and
/// read as `null` where it passes values through.
pub fn read_rewards(dir: &Path) -> Result<Rewards, RewardError> {
    read_rewards_with(dir, false)
}

/// [`read_rewards`], taking a reward from `score`, else `rewards`, when
/// `reward.json` has no `reward` and there is no `reward.txt`, if
/// `aliases` is set: BenchFlow 0.8's lenient mode
/// (`BENCHFLOW_REWARD_LENIENT=1`, read by [`lenient_from_env`]).
pub fn read_rewards_with(dir: &Path, aliases: bool) -> Result<Rewards, RewardError> {
    read_rewards_in(dir, aliases, None)
}

/// [`read_rewards_with`], checking the reward and the numbers 0.8 checks
/// against the task's declared `reward_range` (`[lo, hi]`, which only
/// widens [0, 1]; `None` is [0, 1]). A rubric item's `score` stays in
/// [0, 1], as in 0.8.
pub fn read_rewards_in(
    dir: &Path,
    aliases: bool,
    range: Option<(f64, f64)>,
) -> Result<Rewards, RewardError> {
    let (lo, hi) = range.unwrap_or((0.0, 1.0));
    let whose = if range.is_some() {
        " (the task's reward_range)"
    } else {
        ""
    };
    let r = read_any_rewards(dir, aliases)?;
    let file = || {
        dir.join(match r.source {
            RewardSource::Text => "reward.txt",
            RewardSource::Json | RewardSource::Both => "reward.json",
        })
    };
    if !(lo..=hi).contains(&r.reward) {
        return Err(RewardError::Parse {
            file: file(),
            detail: format!("the reward is {}, outside [{lo}, {hi}]{whose}", r.reward),
        });
    }
    let outside = |key: &str, v: f64| {
        if key.starts_with("rubric[") {
            !(0.0..=1.0).contains(&v)
        } else {
            !(lo..=hi).contains(&v)
        }
    };
    if let Some((key, value)) = scored_numbers(&r).find(|(k, v)| outside(k, *v)) {
        let (klo, khi, kwhose) = if key.starts_with("rubric[") {
            (0.0, 1.0, "")
        } else {
            (lo, hi, whose)
        };
        return Err(RewardError::Parse {
            file: dir.join("reward.json"),
            detail: format!(
                "{key} is {value}, outside [{klo}, {khi}]{kwhose} (BenchFlow 0.8 refuses this number too; numbers under its reserved keys such as raw, errors or details are not checked)"
            ),
        });
    }
    Ok(r)
}

/// Keys a reward.json without `reward` takes its reward from in lenient
/// mode, in this order: 0.8's `_LENIENT_REWARD_ALIASES`. 0.8 takes a reward
/// from no other key (but for an aggregate policy over `metrics`, which
/// bench does not apply), so a lone `{"errors": 1}` is not a reward of 1.
const REWARD_ALIASES: [&str; 2] = ["score", "rewards"];

/// The variable that turns on lenient reward files, as in BenchFlow 0.8.
pub const LENIENT_ENV: &str = "BENCHFLOW_REWARD_LENIENT";

/// Whether [`LENIENT_ENV`] is set to a true value (`1`, `true`, `yes`,
/// `on`, in any case), as 0.8 reads it.
pub fn lenient_from_env() -> bool {
    std::env::var(LENIENT_ENV).is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Top-level keys whose numbers 0.8's `validate_reward_map` does not
/// range-check: its structured keys (`_STRUCTURED_REWARD_KEYS`), passed
/// through as they are, and the keys it validates in their own way
/// (`reward` itself, `rubric`, `metrics`, `space`, `granularity`,
/// `aggregate`).
pub const UNCHECKED_KEYS: [&str; 20] = [
    "aggregate",
    "artifacts",
    "debug",
    "details",
    "details_path",
    "errors",
    "evidence",
    "items",
    "metadata",
    "participants",
    "raw",
    "reason",
    "reasons",
    "regressions",
    "winner",
    "reward",
    "rubric",
    "metrics",
    "space",
    "granularity",
];

/// The numbers 0.8's `validate_reward_map` range-checks besides `reward`:
/// top-level numbers outside [`UNCHECKED_KEYS`], the numbers in `metrics`,
/// and the `score` of each `rubric` item.
pub fn scored_numbers(r: &Rewards) -> impl Iterator<Item = (String, f64)> + '_ {
    let top = r
        .values
        .iter()
        .filter(|(k, _)| !UNCHECKED_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), *v));
    let metrics = r
        .extra
        .get("metrics")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(k, v)| Some((format!("metrics.{k}"), v.as_f64()?)));
    let rubric = r
        .extra
        .get("rubric")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, item)| Some((format!("rubric[{i}].score"), item.get("score")?.as_f64()?)));
    top.chain(metrics).chain(rubric)
}

/// [`read_rewards_with`] without the range check.
fn read_any_rewards(dir: &Path, aliases: bool) -> Result<Rewards, RewardError> {
    let json_file = dir.join("reward.json");
    let text_file = dir.join("reward.txt");
    let json = read_nonempty(&json_file)?;
    let text = read_nonempty(&text_file)?;
    match (json, text) {
        (None, None) => Err(RewardError::NotFound {
            dir: dir.to_path_buf(),
        }),
        (None, Some(t)) => {
            let reward = parse_text(&text_file, &t)?;
            Ok(Rewards {
                reward,
                values: BTreeMap::from([("reward".to_string(), reward)]),
                source: RewardSource::Text,
                extra: Map::new(),
            })
        }
        (Some(j), t) => {
            let JsonRewards {
                reward: json_reward,
                mut values,
                extra,
            } = parse_json(&json_file, &j)?;
            let text_reward = t.map(|t| parse_text(&text_file, &t)).transpose()?;
            let (reward, source) = match (json_reward, text_reward) {
                (Some(a), Some(b)) if !same(a, b) => {
                    return Err(RewardError::Disagree { json: a, text: b });
                }
                (Some(a), Some(_)) => (a, RewardSource::Both),
                (Some(a), None) => (a, RewardSource::Json),
                (None, Some(b)) => {
                    values.insert("reward".into(), b);
                    (b, RewardSource::Both)
                }
                (None, None) => match REWARD_ALIASES.iter().find_map(|k| values.get(*k)) {
                    Some(&r) if aliases => (r, RewardSource::Json),
                    Some(_) => {
                        return Err(RewardError::Parse {
                            file: json_file,
                            detail: format!(
                                "it has no reward key; BenchFlow 0.8 takes score or rewards in its place only in lenient mode, \
which {LENIENT_ENV}=1 turns on for bench too"
                            ),
                        });
                    }
                    None => {
                        return Err(RewardError::Parse {
                            file: json_file,
                            detail: "it has no reward key, nor score or rewards (the keys BenchFlow 0.8 takes a reward from in their place), so it does not say which number is the reward".into(),
                        });
                    }
                },
            };
            Ok(Rewards {
                reward,
                values,
                source,
                extra,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(files: &[(&str, &str)]) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for (name, text) in files {
            std::fs::write(d.path().join(name), text).unwrap();
        }
        d
    }

    /// Guards the D5 fix: a reward outside [0, 1] was scored (reward 2 or
    /// -0.2 raised or lowered the job's mean); 0.8 makes it a verifier error.
    #[test]
    fn a_reward_outside_zero_to_one_is_refused() {
        for (name, text) in [
            ("reward.txt", "2"),
            ("reward.txt", "-0.2"),
            ("reward.txt", "1e3"),
            ("reward.json", r#"{"reward": 1.5}"#),
            ("reward.json", r#"{"score": 7}"#),
        ] {
            let d = dir(&[(name, text)]);
            // Lenient, so `score` is read as the reward and its range checked.
            let e = read_rewards_with(d.path(), true).unwrap_err();
            assert!(
                matches!(&e, RewardError::Parse { detail, .. } if detail.contains("outside [0, 1]")),
                "{name} {text}: {e:?}"
            );
        }
        for text in ["0", "1", "0.25", "-0", "1.0"] {
            let d = dir(&[("reward.txt", text)]);
            assert!(read_rewards(d.path()).is_ok(), "{text}");
        }
    }

    /// The numbers 0.8's `validate_reward_map` range-checks must be in
    /// [0, 1]: `{"reward": 0.5, "tests": 12}` is a verifier error there.
    #[test]
    fn every_scored_number_must_be_in_zero_to_one() {
        for (text, key) in [
            (r#"{"reward": 0.5, "tests": 12}"#, "tests is 12"),
            (r#"{"reward": 0.5, "penalty": -0.1}"#, "penalty is -0.1"),
            (
                r#"{"reward": 0.5, "metrics": {"acc": 3}}"#,
                "metrics.acc is 3",
            ),
            (
                r#"{"reward": 0.5, "rubric": [{"score": 1}, {"score": 2.5}]}"#,
                "rubric[1].score is 2.5",
            ),
        ] {
            let d = dir(&[("reward.json", text)]);
            let e = read_rewards(d.path()).unwrap_err();
            assert!(
                matches!(&e, RewardError::Parse { detail, .. } if detail.starts_with(key) && detail.contains("outside [0, 1]")),
                "{text}: {e:?}"
            );
        }
    }

    /// 0.8 passes numbers under its structured keys through unchecked, so
    /// it scores these (they gave R302 on the first cut of the range check).
    #[test]
    fn numbers_under_0_8s_reserved_keys_are_not_range_checked() {
        for text in [
            r#"{"reward": 0.42, "raw": 42, "errors": 3, "winner": 2, "items": 10, "details_path": 3, "regressions": 2, "participants": 5, "debug": 7}"#,
            r#"{"reward": 0.875, "artifacts": 9, "evidence": -1, "metadata": 11, "reason": 12, "reasons": 13, "details": 14}"#,
            r#"{"reward": 0, "errors": 3}"#,
        ] {
            let d = dir(&[("reward.json", text)]);
            let r = read_rewards(d.path()).unwrap_or_else(|e| panic!("{text}: {e:?}"));
            assert!(r.reward <= 1.0);
        }
    }

    /// Where bench is more lenient than 0.8's default (strict) mode, listed
    /// so that a change is deliberate: strict 0.8 refuses each of these
    /// (an unknown non-numeric key, a non-numeric metric, a `space`,
    /// `granularity` or `aggregate` it does not know); its lenient mode
    /// drops those keys and keeps the trial.
    #[test]
    fn non_numbers_beside_a_numeric_reward_are_accepted_unlike_strict_0_8() {
        for text in [
            r#"{"reward": 1, "done": true}"#,
            r#"{"reward": 0.5, "metrics": {"acc": 1, "note": "x"}}"#,
            r#"{"reward": 0.5, "space": 0.7, "granularity": 2}"#,
            r#"{"reward": 0.875, "aggregate": 15}"#,
        ] {
            let d = dir(&[("reward.json", text)]);
            assert!(read_rewards(d.path()).is_ok(), "{text}");
        }
    }

    /// 0.8's `json.loads` reads Python's `NaN` and `Infinity`; under a key
    /// 0.8 passes through, the file loads.
    #[test]
    fn python_nan_and_infinity_load_where_0_8_passes_them_through() {
        for text in [
            r#"{"reward": 0.5, "details": {"x": NaN}}"#,
            r#"{"reward": 0.5, "raw": Infinity, "debug": -Infinity}"#,
        ] {
            let d = dir(&[("reward.json", text)]);
            let r = read_rewards(d.path()).unwrap_or_else(|e| panic!("{text}: {e:?}"));
            assert_eq!(r.reward, 0.5);
        }
        let d = dir(&[(
            "reward.json",
            r#"{"reward": 0.5, "details": "NaN \"Infinity\""}"#,
        )]);
        let r = read_rewards(d.path()).unwrap();
        assert_eq!(r.extra["details"], r#"NaN "Infinity""#);
        for text in [r#"{"reward": NaN}"#, r#"{"reward": Infinity}"#] {
            let d = dir(&[("reward.json", text)]);
            assert!(read_rewards(d.path()).is_err(), "{text}");
        }
    }

    /// Strict 0.8 refuses a NaN or Infinity wherever it range-checks a
    /// number; Python's json.dump writes NaN for a 0/0 metric. Reading them
    /// as null let these through.
    #[test]
    fn python_nan_and_infinity_are_refused_where_0_8_checks_numbers() {
        for (text, what) in [
            (r#"{"reward": 0.0, "accuracy": NaN}"#, "accuracy is NaN"),
            (
                r#"{"reward": 1, "metrics": {"loss": Infinity}}"#,
                "metrics.loss is Infinity",
            ),
            (
                r#"{"reward": 0.5, "metrics": {"acc": NaN}}"#,
                "metrics.acc is NaN",
            ),
            (
                r#"{"reward": 0.5, "rubric": [{"score": 1}, {"score": NaN}]}"#,
                "rubric[1].score is NaN",
            ),
            (r#"{"reward": -Infinity}"#, "reward is -Infinity"),
        ] {
            let d = dir(&[("reward.json", text)]);
            let e = read_rewards(d.path()).unwrap_err();
            assert!(
                matches!(&e, RewardError::Parse { detail, .. } if detail.starts_with(what) && detail.contains("not a number in [0, 1]")),
                "{text}: {e:?}"
            );
        }
        // Where 0.8 passes values through, they are read as null.
        let d = dir(&[(
            "reward.json",
            r#"{"reward": 0.5, "details": {"x": NaN}, "rubric": [{"score": 1, "raw": NaN}]}"#,
        )]);
        let r = read_rewards(d.path()).unwrap();
        assert_eq!(r.extra["details"]["x"], Value::Null);
        assert_eq!(r.extra["rubric"][0]["raw"], Value::Null);
    }

    /// Without `reward`, 0.8 takes a reward only from `score` or `rewards`
    /// (its lenient aliases), and only in lenient mode: `{"errors": 1}`
    /// scored 1.0 and passed, and `{"score": 0.4}` scored where strict 0.8
    /// refuses it (D5d's open finding, fixed 2026-10-11).
    #[test]
    fn a_reward_comes_only_from_keys_0_8_takes_one_from() {
        for text in [
            r#"{"errors": 1}"#,
            r#"{"errors": 0}"#,
            r#"{"aggregate": 0.25}"#,
            r#"{"accuracy": 0.7}"#,
        ] {
            let d = dir(&[("reward.json", text)]);
            let e = read_rewards(d.path()).unwrap_err();
            assert!(
                matches!(&e, RewardError::Parse { detail, .. } if detail.contains("no reward key, nor score or rewards")),
                "{text}: {e:?}"
            );
        }
        for (text, want) in [
            (r#"{"score": 0.4}"#, 0.4),
            (r#"{"rewards": 0.3}"#, 0.3),
            (r#"{"score": 0.4, "rewards": 1, "accuracy": 1}"#, 0.4),
        ] {
            let d = dir(&[("reward.json", text)]);
            // Strict, 0.8's default: refused, naming the way to allow it.
            let e = read_rewards(d.path()).unwrap_err();
            assert!(
                matches!(&e, RewardError::Parse { detail, .. } if detail.contains("only in lenient mode") && detail.contains("BENCHFLOW_REWARD_LENIENT=1")),
                "{text}: {e:?}"
            );
            assert_eq!(
                read_rewards_with(d.path(), true).unwrap().reward,
                want,
                "{text}"
            );
        }
        // Lenient mode takes no other key.
        let d = dir(&[("reward.json", r#"{"errors": 1}"#)]);
        assert!(read_rewards_with(d.path(), true).is_err());
        // A reward.txt still gives the reward in either mode.
        let d = dir(&[("reward.json", r#"{"score": 0.4}"#), ("reward.txt", "1")]);
        assert_eq!(read_rewards(d.path()).unwrap().reward, 1.0);
    }

    /// A task that declares `[verifier] reward_range` (BenchFlow 0.8's, which
    /// only widens [0, 1]) has rewards inside it scored, and ones outside it
    /// refused; a rubric item's score stays in [0, 1]. Before 2026-10-11
    /// (break campaign E9) bench refused such a task at load.
    #[test]
    fn a_declared_reward_range_widens_what_is_scored() {
        let wide = Some((-1.0, 1.0));
        for (name, text, want) in [
            ("reward.txt", "-1", -1.0),
            ("reward.txt", "-0.5", -0.5),
            ("reward.json", r#"{"reward": -0.25, "accuracy": -1}"#, -0.25),
            (
                "reward.json",
                r#"{"reward": 1, "metrics": {"safety": -0.5}}"#,
                1.0,
            ),
        ] {
            let d = dir(&[(name, text)]);
            assert_eq!(
                read_rewards_in(d.path(), false, wide).unwrap().reward,
                want,
                "{name} {text}"
            );
            // Undeclared, the same file is refused, as before.
            assert!(read_rewards(d.path()).is_err(), "{name} {text}");
        }
        for (name, text) in [
            ("reward.txt", "-1.5"),
            ("reward.txt", "2"),
            ("reward.json", r#"{"reward": 0.5, "accuracy": -2}"#),
            (
                "reward.json",
                r#"{"reward": 0.5, "rubric": [{"score": -0.5}]}"#,
            ),
        ] {
            let d = dir(&[(name, text)]);
            let e = read_rewards_in(d.path(), false, wide).unwrap_err();
            assert!(
                matches!(&e, RewardError::Parse { detail, .. } if detail.contains("outside [")),
                "{name} {text}: {e:?}"
            );
        }
        let d = dir(&[("reward.txt", "3")]);
        let e = read_rewards_in(d.path(), false, wide).unwrap_err();
        assert!(
            e.to_string()
                .contains("outside [-1, 1] (the task's reward_range)"),
            "{e}"
        );
    }

    /// 0.8 computes a reward from `metrics` and an `aggregate` policy;
    /// bench does not, and says so instead of "metrics is ..., not a number".
    #[test]
    fn metrics_with_an_aggregate_policy_and_no_reward_say_what_is_missing() {
        let d = dir(&[(
            "reward.json",
            r#"{"metrics": {"a": 1, "b": 0}, "aggregate": {"method": "mean"}}"#,
        )]);
        let e = read_rewards(d.path()).unwrap_err();
        assert!(
            matches!(&e, RewardError::Parse { detail, .. } if detail.contains("aggregate policy but no reward key")),
            "{e:?}"
        );
    }

    #[test]
    fn text_alone() {
        let d = dir(&[("reward.txt", "0.5\n")]);
        let r = read_rewards(d.path()).unwrap();
        assert_eq!((r.reward, r.source), (0.5, RewardSource::Text));
    }

    #[test]
    fn a_flat_json_map_is_preferred() {
        let d = dir(&[("reward.json", r#"{"reward": 1, "accuracy": 0.75}"#)]);
        let r = read_rewards(d.path()).unwrap();
        assert_eq!(r.reward, 1.0);
        assert_eq!(r.values["accuracy"], 0.75);
        assert_eq!(r.source, RewardSource::Json);
    }

    #[test]
    fn an_envelope_keeps_its_other_keys() {
        let d = dir(&[(
            "reward.json",
            r#"{"reward": 0.25, "details": {"a": [1]}, "done": true}"#,
        )]);
        let r = read_rewards(d.path()).unwrap();
        assert_eq!(r.reward, 0.25);
        assert_eq!(r.extra["done"], Value::Bool(true));
        assert!(r.extra.contains_key("details"));
    }

    #[test]
    fn a_single_number_without_a_reward_key_is_the_reward() {
        let d = dir(&[("reward.json", r#"{"score": 0.4}"#)]);
        assert_eq!(read_rewards_with(d.path(), true).unwrap().reward, 0.4);
        let d = dir(&[("reward.json", r#"{"a": 0.4, "b": 1}"#)]);
        assert!(matches!(
            read_rewards(d.path()),
            Err(RewardError::Parse { .. })
        ));
        let d = dir(&[
            ("reward.json", r#"{"a": 0.4, "b": 1}"#),
            ("reward.txt", "1"),
        ]);
        let r = read_rewards(d.path()).unwrap();
        assert_eq!((r.reward, r.source), (1.0, RewardSource::Both));
    }

    #[test]
    fn errors_say_what_and_what_next() {
        let d = dir(&[]);
        let e = read_rewards(d.path()).unwrap_err();
        assert!(matches!(e, RewardError::NotFound { .. }));
        assert!(e.to_string().contains("neither reward.json nor reward.txt"));
        assert!(!e.hint().is_empty());

        let d = dir(&[("reward.txt", "")]);
        assert!(matches!(
            read_rewards(d.path()),
            Err(RewardError::Empty { .. })
        ));
        let d = dir(&[("reward.json", "  \n")]);
        assert!(matches!(
            read_rewards(d.path()),
            Err(RewardError::Empty { .. })
        ));

        for bad in ["abc", "nan", "inf", "1 2"] {
            let d = dir(&[("reward.txt", bad)]);
            assert!(
                matches!(read_rewards(d.path()), Err(RewardError::Parse { .. })),
                "{bad}"
            );
        }
        for bad in ["[1]", "{", r#"{"reward": "1"}"#, r#"{"a": "x"}"#, "1e999"] {
            let d = dir(&[("reward.json", bad)]);
            assert!(
                matches!(read_rewards(d.path()), Err(RewardError::Parse { .. })),
                "{bad}"
            );
        }
    }

    #[test]
    fn both_files_must_agree() {
        let d = dir(&[("reward.json", r#"{"reward": 1}"#), ("reward.txt", "0")]);
        let e = read_rewards(d.path()).unwrap_err();
        assert_eq!(
            e,
            RewardError::Disagree {
                json: 1.0,
                text: 0.0
            }
        );
        assert_eq!(
            e.to_string(),
            "reward.json says reward = 1 and reward.txt says 0"
        );
        let d = dir(&[("reward.json", r#"{"reward": 1}"#), ("reward.txt", "1.0\n")]);
        assert_eq!(read_rewards(d.path()).unwrap().source, RewardSource::Both);
    }
}
