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
                "reward.txt holds one finite number; reward.json holds an object of finite numbers, or a numeric \"reward\" beside \
other keys"
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

/// `reward.json`: its numbers, its `reward` if it has one, and the other
/// keys of an envelope.
fn parse_json(file: &Path, text: &str) -> Result<JsonRewards, RewardError> {
    let err = |detail: String| RewardError::Parse {
        file: file.to_path_buf(),
        detail,
    };
    let value: Value =
        serde_json::from_str(text).map_err(|e| err(format!("it is not JSON ({e})")))?;
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
/// `reward.txt`'s number, else its only number. When both files exist, the
/// JSON's `reward` must equal the text's.
pub fn read_rewards(dir: &Path) -> Result<Rewards, RewardError> {
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
                (None, None) if values.len() == 1 => (
                    *values.values().next().expect("one value"),
                    RewardSource::Json,
                ),
                (None, None) => {
                    return Err(RewardError::Parse {
                        file: json_file,
                        detail: format!(
                            "it holds {} numbers and no reward key, so it does not say which one is the reward",
                            values.len()
                        ),
                    });
                }
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
        assert_eq!(read_rewards(d.path()).unwrap().reward, 0.4);
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
