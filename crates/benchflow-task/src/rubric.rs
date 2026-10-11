//! task.md rubrics (`verifier/rubric.json`, schema rubric-1) and how they
//! score (task.md's docs/rubrics.md, "Scoring").
//!
//! A rubric lists criteria. A gate must pass or the reward is 0; a criterion
//! with points earns positive points on pass and is charged negative points
//! on fail; a continuous criterion follows the judge's number in proportion;
//! a criterion with levels earns the level the judge picked. Test-judged
//! criteria are decided from the verifier's CTRF report by the spec's
//! matching rule ([`crate::ctrf`]).
//!
//! Scoring is a port of the reference's `tools/rubrics.py` `score()`, in
//! exact rational arithmetic as Python's `Fraction` does it, so a reward here
//! is the reward there. (Numbers too large for 128-bit fractions fall back to
//! floating point.)

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

use crate::ctrf::{Ctrf, CtrfTest};
use crate::diag::Diagnostic;

pub use ratio::Ratio;

/// The rubric-1 schema URL.
pub const RUBRIC_SCHEMA: &str = "https://task.md/schema/rubric-1.json";
/// The review-1 schema URL.
pub const REVIEW_SCHEMA: &str = "https://task.md/schema/review-1.json";

/// Who decides a criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Judge {
    /// A test in the verifier's CTRF report.
    Test,
    /// A rule expression.
    Rule,
    /// A predicate over trusted world state.
    World,
    /// A re-run of the deliverable at hidden parameters.
    Replay,
    /// One model call with the evidence inline.
    Llm,
    /// A model that works in a sandbox.
    Agent,
    /// A vision-language model.
    Vlm,
    /// A person.
    Human,
    /// Several models (reserved).
    Panel,
}

impl Judge {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "test" => Judge::Test,
            "rule" => Judge::Rule,
            "world" => Judge::World,
            "replay" => Judge::Replay,
            "llm" => Judge::Llm,
            "agent" => Judge::Agent,
            "vlm" => Judge::Vlm,
            "human" => Judge::Human,
            "panel" => Judge::Panel,
            _ => return None,
        })
    }

    /// The role's name in task.md.
    pub fn as_str(self) -> &'static str {
        match self {
            Judge::Test => "test",
            Judge::Rule => "rule",
            Judge::World => "world",
            Judge::Replay => "replay",
            Judge::Llm => "llm",
            Judge::Agent => "agent",
            Judge::Vlm => "vlm",
            Judge::Human => "human",
            Judge::Panel => "panel",
        }
    }

    /// Whether a model fills this role.
    pub fn is_model(self) -> bool {
        matches!(self, Judge::Llm | Judge::Agent | Judge::Vlm | Judge::Panel)
    }
}

/// Which way a criterion's text points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    /// The text states what should hold (the default).
    #[default]
    Good,
    /// The text states what should not happen; the criterion passes when it
    /// did not.
    Bad,
}

/// How the earned points become the partial score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// Earned points over the maximum positive points, clipped to [0, 1].
    #[default]
    Points,
    /// The raw total, neither divided nor clipped.
    Sum,
}

/// Which score is the reward written to `reward.txt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Headline {
    /// The partial score (the default).
    #[default]
    Partial,
    /// 1 when every gate passed and the partial score reached the threshold.
    Strict,
}

/// A rubric's scoring settings.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Scoring {
    /// How earned points become the partial score.
    pub method: Method,
    /// Which score `reward.txt` gets.
    pub headline: Headline,
    /// What `strict` needs: a fraction under `points` (default 1), points
    /// under `sum` (default the maximum positive points).
    pub pass_threshold: Option<f64>,
}

/// Where a criterion is asked for: a quote from what the agent is told, or
/// from a file it is told to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stated {
    /// A quote from the instruction, or a stage, role, or user prompt.
    Quote(String),
    /// A quote from a file in the package the agent is told to read.
    File { file: String, quote: String },
}

/// One criterion of a rubric.
#[derive(Debug, Clone, PartialEq)]
pub struct Criterion {
    /// Lowercase letters, digits, and hyphens; unique in the rubric.
    pub id: String,
    /// The claim a judge checks.
    pub text: String,
    /// Which way `text` points.
    pub outcome: Outcome,
    /// A gate: the reward is 0 unless it passes.
    pub gate: bool,
    /// The points it carries (none for a gate).
    pub points: Option<f64>,
    /// The judge's scale for a continuous criterion: `(min, max)`.
    pub range: Option<(f64, f64)>,
    /// Who decides it.
    pub judge: Judge,
    /// For a test-judged criterion, the tests it names.
    pub check: Option<String>,
    /// What a model judge reads.
    pub evidence: Vec<String>,
    /// `cite = "independent"`: a pass needs a citation the solver did not
    /// produce.
    pub cite_independent: bool,
    /// `missing = "no-penalty"`: a bad-outcome criterion whose evidence the
    /// solver never produced is skipped, not judged.
    pub missing_no_penalty: bool,
    /// Where the agent is asked for it.
    pub stated: Vec<Stated>,
    /// Why the agent should meet it without being told.
    pub implicit: Option<String>,
    /// Level keys and their anchors, in order.
    pub levels: Vec<(String, String)>,
    /// A hidden reference value, such as `verifier/answers.toml#t_c`.
    pub reference: Option<String>,
    /// Notes for the judge.
    pub guidance: Option<String>,
    /// A label for reporting.
    pub group: Option<String>,
    /// `x-` keys.
    pub extensions: Map<String, Value>,
    /// The criterion as written; scoring reads it as the reference does.
    raw: Map<String, Value>,
}

/// A parsed `verifier/rubric.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rubric {
    /// The rubric's own version, `x.y.z`.
    pub version: Option<String>,
    /// Shared rubrics this one extends, by URL.
    pub extends: Vec<String>,
    /// How it scores.
    pub scoring: Scoring,
    /// Its criteria, in order (removal entries left out).
    pub criteria: Vec<Criterion>,
    /// Agreement and calibration records, as written.
    pub validation: Option<Value>,
    /// The file as written.
    pub raw: Value,
}

impl Serialize for Rubric {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.raw.serialize(s)
    }
}

impl<'de> Deserialize<'de> for Rubric {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = Value::deserialize(d)?;
        Rubric::from_value(&raw).map_err(|diags| {
            let first = diags
                .first()
                .map_or_else(|| "not a rubric".to_string(), |d| d.message.clone());
            serde::de::Error::custom(first)
        })
    }
}

fn opt_str(m: &Map<String, Value>, key: &str) -> Option<String> {
    m.get(key).and_then(Value::as_str).map(str::to_string)
}

impl Rubric {
    /// Read and check `verifier/rubric.json`. `Err` holds every error the
    /// task.md checker finds in it.
    pub fn from_path(path: &Path) -> Result<Rubric, Vec<Diagnostic>> {
        let file = path
            .file_name()
            .map_or_else(|| path.to_path_buf(), std::path::PathBuf::from);
        let text = std::fs::read_to_string(path).map_err(|e| {
            vec![
                Diagnostic::error(
                    crate::taskmd::code::JUDGMENT_FILE,
                    format!("cannot read the rubric: {e}"),
                )
                .in_file(&file),
            ]
        })?;
        let value: Value = serde_json::from_str(&text).map_err(|e| {
            vec![
                Diagnostic::error(
                    crate::taskmd::code::JUDGMENT_FILE,
                    format!("not valid JSON: {e}"),
                )
                .at(&file, Some(e.line().max(1)), None),
            ]
        })?;
        Self::check_and_build(&value, Some(&text), path)
    }

    /// Check a rubric given as JSON. `Err` holds every error the task.md
    /// checker finds in it.
    pub fn from_value(value: &Value) -> Result<Rubric, Vec<Diagnostic>> {
        Self::check_and_build(value, None, Path::new("verifier/rubric.json"))
    }

    fn check_and_build(
        value: &Value,
        text: Option<&str>,
        path: &Path,
    ) -> Result<Rubric, Vec<Diagnostic>> {
        let errors: Vec<Diagnostic> = crate::taskmd::rubric_errors(value, text, path)
            .into_iter()
            .filter(Diagnostic::is_error)
            .collect();
        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(Self::build(value))
    }

    /// The typed view of a rubric that passed the checker.
    fn build(value: &Value) -> Rubric {
        let empty = Map::new();
        let top = value.as_object().unwrap_or(&empty);
        let scoring = top
            .get("scoring")
            .and_then(Value::as_object)
            .unwrap_or(&empty);
        let scoring = Scoring {
            method: if scoring.get("method").and_then(Value::as_str) == Some("sum") {
                Method::Sum
            } else {
                Method::Points
            },
            headline: if scoring.get("headline").and_then(Value::as_str) == Some("strict") {
                Headline::Strict
            } else {
                Headline::Partial
            },
            pass_threshold: scoring.get("pass_threshold").and_then(Value::as_f64),
        };
        let criteria = top
            .get("criteria")
            .and_then(Value::as_array)
            .map(|c| {
                c.iter()
                    .filter_map(Value::as_object)
                    .filter(|m| !m.contains_key("remove"))
                    .map(criterion)
                    .collect()
            })
            .unwrap_or_default();
        Rubric {
            version: opt_str(top, "version"),
            extends: top
                .get("extends")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            scoring,
            criteria,
            validation: top.get("validation").cloned(),
            raw: value.clone(),
        }
    }

    /// Whether grading needs a model: some criterion is judged by `llm`,
    /// `agent`, `vlm`, or `panel`.
    pub fn needs_model(&self) -> bool {
        self.criteria.iter().any(|c| c.judge.is_model())
    }

    /// The verdict of every test-judged criterion, from the verifier's CTRF
    /// report: pass when a test matches its check and every matching test
    /// passed. A missing or malformed report fails them all.
    pub fn test_verdicts(&self, ctrf: &Ctrf) -> BTreeMap<String, Verdict> {
        self.criteria
            .iter()
            .filter(|c| c.judge == Judge::Test)
            .map(|c| {
                let pass = crate::ctrf::test_verdict(c.check.as_deref().unwrap_or(""), &ctrf.tests);
                (
                    c.id.clone(),
                    if pass { Verdict::Pass } else { Verdict::Fail },
                )
            })
            .collect()
    }

    /// Score one trial. A criterion with no verdict, or a level that is not
    /// one of its levels, scores as `skip`: nothing earned or charged, and
    /// never a pass.
    pub fn score(&self, verdicts: &BTreeMap<String, Verdict>) -> RubricScore {
        self.score_with(verdicts, &[])
    }

    /// [`Rubric::score`] with the consequences of the behaviors detected in
    /// the trial.
    pub fn score_with(
        &self,
        verdicts: &BTreeMap<String, Verdict>,
        behaviors: &[Consequence],
    ) -> RubricScore {
        let mut fixed: BTreeMap<String, Verdict> = BTreeMap::new();
        let mut problems: BTreeMap<String, String> = BTreeMap::new();
        for c in &self.criteria {
            let v = verdicts.get(&c.id).cloned().unwrap_or(Verdict::Skip);
            if let Verdict::Level(level) = &v
                && !c.levels.is_empty()
                && !c.levels.iter().any(|(k, _)| k == level)
            {
                problems.insert(c.id.clone(), format!("{level:?} is not one of its levels"));
                fixed.insert(c.id.clone(), Verdict::Skip);
                continue;
            }
            fixed.insert(c.id.clone(), v);
        }
        let mut score = match self.try_score_with(&fixed, behaviors) {
            Ok(s) => s,
            Err(_) => self.float_score(&fixed, behaviors),
        };
        for d in &mut score.criteria {
            d.problem = problems.remove(&d.id);
        }
        score
    }

    /// Score exactly as the reference's `score()` does, failing where it
    /// raises: a level that is not one of the criterion's levels, or a value
    /// that is not a number.
    pub fn try_score(
        &self,
        verdicts: &BTreeMap<String, Verdict>,
    ) -> Result<RubricScore, ScoreError> {
        self.try_score_with(verdicts, &[])
    }

    /// [`Rubric::try_score`] with behavior consequences.
    pub fn try_score_with(
        &self,
        verdicts: &BTreeMap<String, Verdict>,
        behaviors: &[Consequence],
    ) -> Result<RubricScore, ScoreError> {
        compute::<Ratio>(self, verdicts, behaviors).map(|p| p.finish(self, verdicts, true))
    }

    fn float_score(
        &self,
        verdicts: &BTreeMap<String, Verdict>,
        behaviors: &[Consequence],
    ) -> RubricScore {
        match compute::<f64>(self, verdicts, behaviors) {
            Ok(p) => p.finish(self, verdicts, false),
            Err(_) => Parts::<f64>::zero(self).finish(self, verdicts, false),
        }
    }

    /// Grade a trial: decide the test-judged criteria from `ctrf`, take the
    /// model and human judges' verdicts from `judged`, and score, keeping
    /// each test criterion's matches for the review.
    pub fn grade(&self, ctrf: &Ctrf, judged: &BTreeMap<String, Verdict>) -> RubricScore {
        let mut verdicts = judged.clone();
        verdicts.extend(self.test_verdicts(ctrf));
        let mut score = self.score(&verdicts);
        score.tool = ctrf.tool.clone();
        for d in &mut score.criteria {
            let Some(c) = self.criteria.iter().find(|c| c.id == d.id) else {
                continue;
            };
            if c.judge != Judge::Test {
                continue;
            }
            let matched =
                crate::ctrf::matching_tests(c.check.as_deref().unwrap_or(""), &ctrf.tests);
            d.rationale = Some(test_rationale(&matched));
        }
        score
    }
}

fn criterion(m: &Map<String, Value>) -> Criterion {
    let truthy = |v: Option<&Value>| match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    };
    let points = m.get("points").and_then(|p| match p {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => Ratio::parse(s)
            .map(Ratio::to_f64)
            .or_else(|| s.trim().parse().ok()),
        _ => None,
    });
    let range = m
        .get("score")
        .and_then(Value::as_object)
        .and_then(|s| Some((s.get("min")?.as_f64()?, s.get("max")?.as_f64()?)));
    let stated = match m.get("stated") {
        Some(Value::String(q)) => vec![Stated::Quote(q.clone())],
        Some(Value::Object(f)) => vec![file_quote(f)],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|q| match q {
                Value::String(s) => Some(Stated::Quote(s.clone())),
                Value::Object(f) => Some(file_quote(f)),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    Criterion {
        id: opt_str(m, "id").unwrap_or_default(),
        text: m
            .get("text")
            .map(|t| t.as_str().map_or_else(|| t.to_string(), str::to_string))
            .unwrap_or_default(),
        outcome: if m.get("outcome").and_then(Value::as_str) == Some("bad") {
            Outcome::Bad
        } else {
            Outcome::Good
        },
        gate: truthy(m.get("gate")),
        points,
        range,
        judge: m
            .get("judge")
            .and_then(Value::as_str)
            .and_then(Judge::parse)
            .unwrap_or(Judge::Test),
        check: m.get("check").and_then(Value::as_str).map(str::to_string),
        evidence: m
            .get("evidence")
            .and_then(Value::as_array)
            .map(|e| {
                e.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        cite_independent: m.get("cite").and_then(Value::as_str) == Some("independent"),
        missing_no_penalty: m.get("missing").and_then(Value::as_str) == Some("no-penalty"),
        stated,
        implicit: opt_str(m, "implicit"),
        levels: m
            .get("levels")
            .and_then(Value::as_object)
            .map(|l| {
                l.iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            v.as_str().map_or_else(|| v.to_string(), str::to_string),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default(),
        reference: opt_str(m, "reference"),
        guidance: opt_str(m, "guidance"),
        group: opt_str(m, "group"),
        extensions: m
            .iter()
            .filter(|(k, _)| k.starts_with("x-"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        raw: m.clone(),
    }
}

fn file_quote(f: &Map<String, Value>) -> Stated {
    Stated::File {
        file: opt_str(f, "file").unwrap_or_default(),
        quote: opt_str(f, "quote").unwrap_or_default(),
    }
}

fn test_rationale(matched: &[&CtrfTest]) -> String {
    let text = if matched.is_empty() {
        "no test in ctrf.json matches this check".to_string()
    } else if matched.iter().all(|t| t.status == "passed") {
        format!("all {} matching tests passed", matched.len())
    } else {
        let failed: Vec<String> = matched
            .iter()
            .filter(|t| t.status != "passed")
            .map(|t| format!("{} ({})", t.name, t.status))
            .collect();
        format!("not passed: {}", failed.join(", "))
    };
    text.chars().take(300).collect()
}

/// What a judge decided for one criterion. Pass means the task went as it
/// should, whatever the criterion's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "verdict", content = "value")]
pub enum Verdict {
    /// The task went as it should on this criterion.
    Pass,
    /// It did not.
    Fail,
    /// The level the judge picked, by key (`"2"`). For a criterion without
    /// levels, a number of points written as text.
    Level(String),
    /// A number: the judge's value on a continuous criterion, or the points
    /// a criterion with levels earned.
    Value(serde_json::Number),
    /// The check did not run: nothing is earned or charged, and it is never
    /// a pass.
    Skip,
}

impl Verdict {
    /// The review's word for it: `pass`, `fail`, `level`, `value`, `skip`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
            Verdict::Level(_) => "level",
            Verdict::Value(_) => "value",
            Verdict::Skip => "skip",
        }
    }
}

/// The consequence of a behavior detected in a trial.
#[derive(Debug, Clone, PartialEq)]
pub enum Consequence {
    /// The trial fails: the partial score is 0.
    Fail,
    /// Reported only.
    Report,
    /// A penalty in reward units (negative).
    Penalty(serde_json::Number),
}

/// Why a score could not be computed exactly as the reference computes it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScoreError {
    /// A judge picked a level the criterion does not have.
    #[error("{id}: {level:?} is not one of its levels")]
    NotALevel { id: String, level: String },
    /// A value or a setting is not a number.
    #[error("{id}: {what} is not a number")]
    NotANumber { id: String, what: String },
    /// A continuous criterion's min equals its max.
    #[error("{id}: its score range is empty")]
    EmptyRange { id: String },
    /// A number is too large for exact arithmetic.
    #[error("a number is too large to score exactly")]
    Overflow,
}

/// One criterion's part in a score.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CriterionScore {
    pub id: String,
    pub judge: Judge,
    pub gate: bool,
    pub verdict: Verdict,
    /// What the verdict adds to the total: positive when it earns points,
    /// negative when it is charged them; `None` for a gate or a skip.
    pub score: Option<f64>,
    /// For a test-judged criterion, its check.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub check: Option<String>,
    /// Why, briefly (for a test criterion: which tests matched).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    /// Set when the verdict could not be used as given (a level the
    /// criterion does not have), which scored it as a skip.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
}

/// A trial's score.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RubricScore {
    /// The reward written to `reward.txt`: the partial score, or the strict
    /// one (1 or 0) when the rubric's headline is `strict`.
    pub reward: f64,
    /// The points earned less the points charged, over the maximum positive
    /// points, clipped to [0, 1] (under `sum`, the raw total); 0 when a gate
    /// failed or a behavior failed the trial.
    pub partial: f64,
    /// Every gate passed, no behavior failed the trial, and the partial
    /// score reached the pass threshold.
    pub strict: bool,
    /// The partial score before clipping and gating, plus behavior
    /// penalties.
    pub raw: f64,
    pub earned: f64,
    /// The maximum positive points.
    pub maximum: f64,
    /// Behavior penalties, in reward units.
    pub penalty: f64,
    /// The gates that failed.
    pub failed_gates: Vec<String>,
    pub criteria: Vec<CriterionScore>,
    /// The rubric's version, for the review.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rubric_version: Option<String>,
    /// The test runner the CTRF report names.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(skip)]
    exact: Option<ExactScore>,
}

/// A score's parts as exact fractions, written as Python's `str(Fraction)`
/// writes them (`3/4`, `1`, `-2`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExactScore {
    pub raw: String,
    pub partial: String,
    pub earned: String,
    pub maximum: String,
    pub penalty: String,
}

impl RubricScore {
    /// The score's parts as exact fractions, when the numbers were small
    /// enough to compute exactly.
    pub fn exact(&self) -> Option<&ExactScore> {
        self.exact.as_ref()
    }

    /// `review.json` in the review-1 shape: one verdict per criterion, in
    /// the rubric's order, with a summary of the verdicts.
    pub fn review_json(&self) -> Value {
        let mut summary = Map::new();
        let mut verdicts = Vec::new();
        for c in &self.criteria {
            let mut v = Map::new();
            v.insert("id".into(), json!(c.id));
            v.insert("verdict".into(), json!(c.verdict.as_str()));
            match &c.verdict {
                Verdict::Level(l) => {
                    v.insert("level".into(), json!(l));
                }
                Verdict::Value(n) => {
                    v.insert("value".into(), Value::Number(n.clone()));
                }
                _ => {}
            }
            v.insert("score".into(), c.score.map_or(Value::Null, |s| json!(s)));
            if let Some(check) = &c.check {
                v.insert("check".into(), json!(check));
            }
            if c.gate {
                v.insert("gate".into(), Value::Bool(true));
            }
            let mut judge = Map::new();
            judge.insert("role".into(), json!(c.judge.as_str()));
            if c.judge == Judge::Test {
                judge.insert(
                    "tool".into(),
                    self.tool.as_ref().map_or(Value::Null, |t| json!(t)),
                );
            }
            if let Some(version) = &self.rubric_version {
                judge.insert("rubric_version".into(), json!(version));
            }
            v.insert("judge".into(), Value::Object(judge));
            if let Some(r) = &c.rationale {
                v.insert("rationale".into(), json!(r));
            }
            let count = summary
                .entry(c.verdict.as_str().to_string())
                .or_insert(json!(0));
            *count = json!(count.as_u64().unwrap_or(0) + 1);
            verdicts.push(Value::Object(v));
        }
        let mut review = Map::new();
        review.insert("$schema".into(), json!(REVIEW_SCHEMA));
        review.insert("record".into(), json!("verifier"));
        if let Some(version) = &self.rubric_version {
            review.insert("rubric_version".into(), json!(version));
        }
        review.insert("summary".into(), Value::Object(summary));
        review.insert("verdicts".into(), Value::Array(verdicts));
        review.insert(
            "x-benchflow".into(),
            json!({
                "raw": self.raw,
                "earned": self.earned,
                "maximum": self.maximum,
                "failed_gates": self.failed_gates,
                "partial": self.partial,
                "strict": self.strict,
            }),
        );
        Value::Object(review)
    }
}

/// Arithmetic the score needs: exact fractions, or floating point when the
/// fractions would overflow.
trait Num: Copy + PartialOrd + std::fmt::Debug {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(self, o: Self) -> Option<Self>;
    fn sub(self, o: Self) -> Option<Self>;
    fn mul(self, o: Self) -> Option<Self>;
    fn div(self, o: Self) -> Option<Self>;
    /// Python's `Fraction(str(v))` for a JSON value.
    fn from_json(v: &Value) -> Option<Self>;
    fn to_f64(self) -> f64;
    fn text(self) -> Option<String>;
}

impl Num for Ratio {
    fn zero() -> Self {
        Ratio::from_int(0)
    }
    fn one() -> Self {
        Ratio::from_int(1)
    }
    fn add(self, o: Self) -> Option<Self> {
        self.checked_add(o)
    }
    fn sub(self, o: Self) -> Option<Self> {
        self.checked_sub(o)
    }
    fn mul(self, o: Self) -> Option<Self> {
        self.checked_mul(o)
    }
    fn div(self, o: Self) -> Option<Self> {
        self.checked_div(o)
    }
    fn from_json(v: &Value) -> Option<Self> {
        Ratio::from_json(v)
    }
    fn to_f64(self) -> f64 {
        Ratio::to_f64(self)
    }
    fn text(self) -> Option<String> {
        Some(self.to_string())
    }
}

impl Num for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn add(self, o: Self) -> Option<Self> {
        Some(self + o)
    }
    fn sub(self, o: Self) -> Option<Self> {
        Some(self - o)
    }
    fn mul(self, o: Self) -> Option<Self> {
        Some(self * o)
    }
    fn div(self, o: Self) -> Option<Self> {
        (o != 0.0).then(|| self / o)
    }
    fn from_json(v: &Value) -> Option<Self> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => Ratio::parse(s)
                .map(Ratio::to_f64)
                .or_else(|| s.trim().parse().ok()),
            _ => None,
        }
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn text(self) -> Option<String> {
        None
    }
}

fn max<N: Num>(a: N, b: N) -> N {
    if b > a { b } else { a }
}

fn min<N: Num>(a: N, b: N) -> N {
    if b < a { b } else { a }
}

/// A score's parts while computing it.
struct Parts<N> {
    base: N,
    partial: N,
    strict: bool,
    earned: N,
    maximum: N,
    penalty: N,
    failed: Vec<String>,
    adds: Vec<Option<N>>,
}

impl<N: Num> Parts<N> {
    fn zero(rubric: &Rubric) -> Self {
        Parts {
            base: N::zero(),
            partial: N::zero(),
            strict: false,
            earned: N::zero(),
            maximum: N::zero(),
            penalty: N::zero(),
            failed: Vec::new(),
            adds: vec![None; rubric.criteria.len()],
        }
    }

    fn finish(
        self,
        rubric: &Rubric,
        verdicts: &BTreeMap<String, Verdict>,
        exact: bool,
    ) -> RubricScore {
        let raw = self.base.add(self.penalty).unwrap_or(self.base);
        let partial = self.partial.to_f64();
        let reward = match rubric.scoring.headline {
            Headline::Partial => partial,
            Headline::Strict => {
                if self.strict {
                    1.0
                } else {
                    0.0
                }
            }
        };
        let criteria = rubric
            .criteria
            .iter()
            .zip(&self.adds)
            .map(|(c, add)| CriterionScore {
                id: c.id.clone(),
                judge: c.judge,
                gate: c.gate,
                verdict: verdicts.get(&c.id).cloned().unwrap_or(Verdict::Skip),
                score: add.map(Num::to_f64),
                check: (c.judge == Judge::Test).then(|| c.check.clone().unwrap_or_default()),
                rationale: None,
                problem: None,
            })
            .collect();
        let exact = if exact {
            (|| {
                Some(ExactScore {
                    raw: raw.text()?,
                    partial: self.partial.text()?,
                    earned: self.earned.text()?,
                    maximum: self.maximum.text()?,
                    penalty: self.penalty.text()?,
                })
            })()
        } else {
            None
        };
        RubricScore {
            reward,
            partial,
            strict: self.strict,
            raw: raw.to_f64(),
            earned: self.earned.to_f64(),
            maximum: self.maximum.to_f64(),
            penalty: self.penalty.to_f64(),
            failed_gates: self.failed,
            criteria,
            rubric_version: rubric.version.clone(),
            tool: None,
            exact,
        }
    }
}

/// Python's truth value of a JSON value.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(m)) => !m.is_empty(),
    }
}

/// The reference's `score()`, over `N`.
fn compute<N: Num>(
    rubric: &Rubric,
    verdicts: &BTreeMap<String, Verdict>,
    behaviors: &[Consequence],
) -> Result<Parts<N>, ScoreError> {
    let ov = |o: Option<N>| o.ok_or(ScoreError::Overflow);
    let scoring = rubric.raw.get("scoring");
    let method_sum = scoring
        .and_then(|s| s.get("method"))
        .and_then(Value::as_str)
        == Some("sum");
    let mut parts = Parts::<N>::zero(rubric);
    for (i, c) in rubric.criteria.iter().enumerate() {
        let raw = &c.raw;
        let id = &c.id;
        let v = verdicts.get(id).cloned().unwrap_or(Verdict::Skip);
        if truthy(raw.get("gate")) {
            if v != Verdict::Pass {
                parts.failed.push(id.clone());
            }
            continue;
        }
        let points = raw.get("points").cloned().unwrap_or(json!(0));
        let pts = N::from_json(&points).ok_or_else(|| ScoreError::NotANumber {
            id: id.clone(),
            what: "points".into(),
        })?;
        if pts > N::zero() {
            parts.maximum = ov(parts.maximum.add(pts))?;
        }
        // A skip scores as the reference's skip value: a pass of a penalty
        // (nothing charged), a fail of anything else (nothing earned).
        let as_bool = match &v {
            Verdict::Pass => Some(true),
            Verdict::Fail => Some(false),
            Verdict::Skip => Some(pts < N::zero()),
            _ => None,
        };
        let add = if let Some(pass) = as_bool {
            if pts > N::zero() {
                if pass { pts } else { N::zero() }
            } else if pass {
                N::zero()
            } else {
                pts
            }
        } else if truthy(raw.get("score")) {
            let range = raw.get("score");
            let bound = |k: &str| {
                range
                    .and_then(|r| r.get(k))
                    .and_then(N::from_json)
                    .ok_or_else(|| ScoreError::NotANumber {
                        id: id.clone(),
                        what: format!("score.{k}"),
                    })
            };
            let (lo, hi) = (bound("min")?, bound("max")?);
            let value = verdict_number::<N>(&v).ok_or_else(|| ScoreError::NotANumber {
                id: id.clone(),
                what: "the verdict".into(),
            })?;
            let span = ov(hi.sub(lo))?;
            let ratio = ov(value.sub(lo))?
                .div(span)
                .ok_or_else(|| ScoreError::EmptyRange { id: id.clone() })?;
            let f = min(max(ratio, N::zero()), N::one());
            let well = if raw.get("outcome").and_then(Value::as_str) == Some("bad") {
                ov(N::one().sub(f))?
            } else {
                f
            };
            if pts > N::zero() {
                ov(pts.mul(well))?
            } else {
                ov(pts.mul(ov(N::one().sub(well))?))?
            }
        } else {
            if let Verdict::Level(level) = &v
                && truthy(raw.get("levels"))
                && !raw
                    .get("levels")
                    .and_then(Value::as_object)
                    .is_some_and(|l| l.contains_key(level))
            {
                return Err(ScoreError::NotALevel {
                    id: id.clone(),
                    level: level.clone(),
                });
            }
            verdict_number::<N>(&v).ok_or_else(|| ScoreError::NotANumber {
                id: id.clone(),
                what: "the verdict".into(),
            })?
        };
        parts.earned = ov(parts.earned.add(add))?;
        // A skip scores nothing: its review verdict carries no score.
        parts.adds[i] = (v != Verdict::Skip).then_some(add);
    }
    parts.base = if method_sum {
        parts.earned
    } else if parts.maximum != N::zero() {
        ov(parts.earned.div(parts.maximum))?
    } else {
        ov(N::one().add(parts.earned))?
    };
    for b in behaviors {
        if let Consequence::Penalty(p) = b {
            let p = N::from_json(&Value::Number(p.clone())).ok_or(ScoreError::NotANumber {
                id: "behavior".into(),
                what: "penalty".into(),
            })?;
            parts.penalty = ov(parts.penalty.add(p))?;
        }
    }
    let fail = behaviors.contains(&Consequence::Fail);
    let mut partial = if method_sum {
        parts.base
    } else {
        min(max(parts.base, N::zero()), N::one())
    };
    if !parts.failed.is_empty() || fail {
        partial = N::zero();
    } else if parts.penalty != N::zero() {
        partial = max(ov(partial.add(parts.penalty))?, min(partial, N::zero()));
    }
    let threshold = match scoring.and_then(|s| s.get("pass_threshold")) {
        Some(t) => N::from_json(t).ok_or(ScoreError::NotANumber {
            id: "scoring".into(),
            what: "pass_threshold".into(),
        })?,
        None if method_sum => parts.maximum,
        None => N::one(),
    };
    parts.partial = partial;
    parts.strict = parts.failed.is_empty() && !fail && partial >= threshold;
    Ok(parts)
}

/// `Fraction(str(v))` for a non-boolean verdict.
fn verdict_number<N: Num>(v: &Verdict) -> Option<N> {
    match v {
        Verdict::Level(s) => N::from_json(&Value::String(s.clone())),
        Verdict::Value(n) => N::from_json(&Value::Number(n.clone())),
        _ => None,
    }
}

/// Exact fractions, as Python's `fractions.Fraction`.
mod ratio {
    use std::cmp::Ordering;
    use std::fmt;

    use serde_json::Value;

    /// A fraction in lowest terms with a positive denominator.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Ratio {
        num: i128,
        den: i128,
    }

    fn gcd(mut a: i128, mut b: i128) -> i128 {
        a = a.abs();
        b = b.abs();
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    impl Ratio {
        /// `n / d` in lowest terms; `None` when `d` is 0.
        pub fn new(num: i128, den: i128) -> Option<Ratio> {
            if den == 0 {
                return None;
            }
            let g = gcd(num, den).max(1);
            let (mut n, mut d) = (num / g, den / g);
            if d < 0 {
                n = n.checked_neg()?;
                d = d.checked_neg()?;
            }
            Some(Ratio { num: n, den: d })
        }

        /// A whole number.
        pub fn from_int(n: i128) -> Ratio {
            Ratio { num: n, den: 1 }
        }

        pub fn numer(self) -> i128 {
            self.num
        }

        pub fn denom(self) -> i128 {
            self.den
        }

        pub fn checked_add(self, o: Ratio) -> Option<Ratio> {
            let n = self
                .num
                .checked_mul(o.den)?
                .checked_add(o.num.checked_mul(self.den)?)?;
            Ratio::new(n, self.den.checked_mul(o.den)?)
        }

        pub fn checked_sub(self, o: Ratio) -> Option<Ratio> {
            self.checked_add(Ratio {
                num: o.num.checked_neg()?,
                den: o.den,
            })
        }

        pub fn checked_mul(self, o: Ratio) -> Option<Ratio> {
            let g1 = gcd(self.num, o.den).max(1);
            let g2 = gcd(o.num, self.den).max(1);
            Ratio::new(
                (self.num / g1).checked_mul(o.num / g2)?,
                (self.den / g2).checked_mul(o.den / g1)?,
            )
        }

        pub fn checked_div(self, o: Ratio) -> Option<Ratio> {
            if o.num == 0 {
                return None;
            }
            self.checked_mul(Ratio::new(o.den, o.num)?)
        }

        pub fn to_f64(self) -> f64 {
            self.num as f64 / self.den as f64
        }

        /// Python's `Fraction(text)`: an optional sign, then `n/d`, or a
        /// decimal with an optional exponent; spaces around it, and
        /// underscores between digits, are allowed.
        pub fn parse(text: &str) -> Option<Ratio> {
            let t = text.trim();
            let (neg, t) = match t.as_bytes().first()? {
                b'-' => (true, &t[1..]),
                b'+' => (false, &t[1..]),
                _ => (false, t),
            };
            let digits = |s: &str| -> Option<i128> {
                if s.is_empty() || s.starts_with('_') || s.ends_with('_') || s.contains("__") {
                    return None;
                }
                let clean: String = s.chars().filter(|c| *c != '_').collect();
                if !clean.chars().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                clean.parse().ok()
            };
            let r = if let Some((n, d)) = t.split_once('/') {
                Ratio::new(digits(n.trim_end())?, digits(d.trim_start())?)?
            } else {
                let lower = t.to_ascii_lowercase();
                let (mant, exp) = match lower.split_once('e') {
                    Some((m, e)) => {
                        let (eneg, e) = match e.as_bytes().first()? {
                            b'-' => (true, &e[1..]),
                            b'+' => (false, &e[1..]),
                            _ => (false, e),
                        };
                        let x = i32::try_from(digits(e)?).ok()?;
                        (m.to_string(), if eneg { -x } else { x })
                    }
                    None => (lower.clone(), 0),
                };
                let (int_part, frac_part) = match mant.split_once('.') {
                    Some((i, f)) => (i, f),
                    None => (mant.as_str(), ""),
                };
                if int_part.is_empty() && frac_part.is_empty() {
                    return None;
                }
                let whole = if int_part.is_empty() {
                    0
                } else {
                    digits(int_part)?
                };
                let frac = if frac_part.is_empty() {
                    0
                } else {
                    digits(frac_part)?
                };
                let scale = 10i128.checked_pow(
                    u32::try_from(frac_part.chars().filter(|c| *c != '_').count()).ok()?,
                )?;
                let mut num = whole.checked_mul(scale)?.checked_add(frac)?;
                let mut den = scale;
                if exp >= 0 {
                    num = num.checked_mul(10i128.checked_pow(exp as u32)?)?;
                } else {
                    den = den.checked_mul(10i128.checked_pow((-exp) as u32)?)?;
                }
                Ratio::new(num, den)?
            };
            if neg {
                Some(Ratio {
                    num: -r.num,
                    den: r.den,
                })
            } else {
                Some(r)
            }
        }

        /// `Fraction(str(v))` for a JSON number or string.
        pub fn from_json(v: &Value) -> Option<Ratio> {
            match v {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Some(Ratio::from_int(i128::from(i)))
                    } else if let Some(u) = n.as_u64() {
                        Some(Ratio::from_int(i128::from(u)))
                    } else {
                        // Python's str() of a float: its shortest round-trip digits.
                        let f = n.as_f64()?;
                        if !f.is_finite() {
                            return None;
                        }
                        Ratio::parse(&crate::taskmd::pyfmt::shortest_e(f))
                    }
                }
                Value::String(s) => Ratio::parse(s),
                _ => None,
            }
        }
    }

    impl PartialOrd for Ratio {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            match (self.num.checked_mul(o.den), o.num.checked_mul(self.den)) {
                (Some(a), Some(b)) => Some(a.cmp(&b)),
                _ => self.to_f64().partial_cmp(&o.to_f64()),
            }
        }
    }

    impl fmt::Display for Ratio {
        /// As Python's `str(Fraction)`: `3/4`, or `2` for a whole number.
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.den == 1 {
                write!(f, "{}", self.num)
            } else {
                write!(f, "{}/{}", self.num, self.den)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn parses_like_python() {
            assert_eq!(Ratio::parse("2.5").unwrap().to_string(), "5/2");
            assert_eq!(Ratio::parse(" -3/6 ").unwrap().to_string(), "-1/2");
            assert_eq!(Ratio::parse("1e-05").unwrap().to_string(), "1/100000");
            assert_eq!(Ratio::parse("1_000").unwrap().to_string(), "1000");
            assert_eq!(Ratio::parse(".5").unwrap().to_string(), "1/2");
            assert_eq!(Ratio::parse("5.").unwrap().to_string(), "5");
            assert_eq!(Ratio::parse("1E+2").unwrap().to_string(), "100");
            for bad in [
                "", ".", "abc", "1/0", "1__0", "_1", "1.2.3", "inf", "1 / 2 3",
            ] {
                assert!(Ratio::parse(bad).is_none(), "{bad}");
            }
        }

        #[test]
        fn floats_read_as_their_shortest_digits() {
            assert_eq!(Ratio::from_json(&json!(0.1)).unwrap().to_string(), "1/10");
            assert_eq!(
                Ratio::from_json(&json!(0.30000000000000004))
                    .unwrap()
                    .to_string(),
                "7500000000000001/25000000000000000"
            );
            assert_eq!(Ratio::from_json(&json!(3)).unwrap().to_string(), "3");
        }

        #[test]
        fn arithmetic() {
            let a = Ratio::new(1, 3).unwrap();
            let b = Ratio::new(1, 6).unwrap();
            assert_eq!(a.checked_add(b).unwrap().to_string(), "1/2");
            assert_eq!(a.checked_sub(b).unwrap().to_string(), "1/6");
            assert_eq!(a.checked_mul(b).unwrap().to_string(), "1/18");
            assert_eq!(a.checked_div(b).unwrap().to_string(), "2");
            assert!(a > b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rubric(v: Value) -> Rubric {
        Rubric::build(&v)
    }

    fn verdicts(items: &[(&str, Verdict)]) -> BTreeMap<String, Verdict> {
        items
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn criteria() -> Value {
        json!([
            {"id": "gate", "gate": true, "judge": "test", "check": "x"},
            {"id": "earn", "points": 3, "judge": "test", "check": "y"},
            {"id": "charge", "points": -2, "judge": "test", "check": "z"},
            {"id": "lvl", "points": 4, "judge": "agent", "levels": {"0": "a", "2": "b", "4": "c"}},
            {"id": "cont", "points": 2, "judge": "llm", "score": {"min": 0, "max": 10}},
            {"id": "bad", "points": 2, "outcome": "bad", "judge": "llm", "score": {"min": 0, "max": 10}}
        ])
    }

    fn full() -> Vec<(&'static str, Verdict)> {
        vec![
            ("gate", Verdict::Pass),
            ("earn", Verdict::Pass),
            ("charge", Verdict::Pass),
            ("lvl", Verdict::Level("4".into())),
            ("cont", Verdict::Value(10.into())),
            ("bad", Verdict::Value(0.into())),
        ]
    }

    #[test]
    fn scores_by_the_rules_of_docs_rubrics() {
        let r = rubric(json!({"scoring": {"method": "points"}, "criteria": criteria()}));
        let s = r.score(&verdicts(&full()));
        assert_eq!((s.partial, s.strict, s.maximum), (1.0, true, 11.0));
        let mut charged = full();
        charged[2].1 = Verdict::Fail;
        charged[4].1 = Verdict::Value(5.into());
        let s = r.score(&verdicts(&charged));
        assert_eq!(s.exact().unwrap().earned, "8");
        assert_eq!(s.exact().unwrap().partial, "8/11");
        assert!(!s.strict);
        let mut gated = full();
        gated[0].1 = Verdict::Fail;
        let s = r.score(&verdicts(&gated));
        assert_eq!((s.partial, s.strict), (0.0, false));
        assert_eq!(s.failed_gates, ["gate"]);
    }

    #[test]
    fn gates_alone_score_one_and_sum_is_unclipped() {
        let r =
            rubric(json!({"criteria": [{"id": "g", "gate": true, "judge": "test", "check": "t"}]}));
        let s = r.score(&verdicts(&[("g", Verdict::Pass)]));
        assert_eq!((s.partial, s.strict), (1.0, true));
        let r = rubric(json!({"scoring": {"method": "sum"}, "criteria": criteria()}));
        let mut v = full();
        v[2].1 = Verdict::Fail;
        let s = r.score(&verdicts(&v));
        assert_eq!(s.partial, 9.0);
        assert!(!s.strict);
    }

    #[test]
    fn headline_strict_writes_one_or_zero() {
        let r = rubric(
            json!({"scoring": {"headline": "strict", "pass_threshold": 0.7}, "criteria": criteria()}),
        );
        let mut v = full();
        v[2].1 = Verdict::Fail;
        v[4].1 = Verdict::Value(5.into());
        let s = r.score(&verdicts(&v));
        assert!(s.strict);
        assert_eq!(s.reward, 1.0);
    }

    #[test]
    fn levels_must_be_levels() {
        let r = rubric(json!({"criteria": criteria()}));
        let mut v = full();
        v[3].1 = Verdict::Level("3".into());
        assert_eq!(
            r.try_score(&verdicts(&v)).unwrap_err(),
            ScoreError::NotALevel {
                id: "lvl".into(),
                level: "3".into()
            }
        );
        let s = r.score(&verdicts(&v));
        assert!(s.criteria[3].problem.is_some());
        assert_eq!(s.criteria[3].verdict, Verdict::Skip);
        assert_eq!(s.criteria[3].score, None);
    }

    #[test]
    fn a_skip_earns_and_charges_nothing() {
        let r = rubric(json!({"criteria": criteria()}));
        let s = r.score(&verdicts(&[("gate", Verdict::Pass)]));
        assert_eq!(s.earned, 0.0);
        let charge = s.criteria.iter().find(|c| c.id == "charge").unwrap();
        assert_eq!((charge.score, &charge.verdict), (None, &Verdict::Skip));
    }

    #[test]
    fn penalties_floor_at_zero() {
        let r =
            rubric(json!({"criteria": [{"id": "a", "points": 1, "judge": "test", "check": "t"}]}));
        let v = verdicts(&[("a", Verdict::Pass)]);
        let s = r.score_with(
            &v,
            &[Consequence::Penalty(
                serde_json::Number::from_f64(-0.25).unwrap(),
            )],
        );
        assert_eq!(s.exact().unwrap().partial, "3/4");
        let s = r.score_with(
            &v,
            &[Consequence::Penalty(
                serde_json::Number::from_f64(-2.0).unwrap(),
            )],
        );
        assert_eq!(s.partial, 0.0);
        assert_eq!(s.exact().unwrap().raw, "-1");
        let s = r.score_with(&v, &[Consequence::Fail]);
        assert_eq!((s.partial, s.strict), (0.0, false));
    }

    #[test]
    fn review_json_has_one_verdict_per_criterion() {
        let r = rubric(json!({"version": "1.0.0", "criteria": criteria()}));
        let ctrf = Ctrf::from_slice(br#"{"results": {"tool": {"name": "pytest"}, "tests": [{"name": "x", "status": "passed"}, {"name": "y", "status": "failed"}]}}"#);
        let judged = verdicts(&[
            ("lvl", Verdict::Level("2".into())),
            ("cont", Verdict::Value(5.into())),
            ("bad", Verdict::Value(5.into())),
        ]);
        let s = r.grade(&ctrf, &judged);
        let review = s.review_json();
        assert_eq!(review["$schema"], REVIEW_SCHEMA);
        assert_eq!(review["rubric_version"], "1.0.0");
        let v = review["verdicts"].as_array().unwrap();
        assert_eq!(v.len(), 6);
        assert_eq!(v[0]["gate"], true);
        assert_eq!(v[0]["judge"]["tool"], "pytest");
        assert_eq!(v[1]["verdict"], "fail");
        assert_eq!(v[1]["rationale"], "not passed: y (failed)");
        assert_eq!(v[2]["rationale"], "no test in ctrf.json matches this check");
        assert_eq!(v[3]["level"], "2");
        assert_eq!(review["summary"]["fail"], 2);
    }

    #[test]
    fn needs_model() {
        assert!(rubric(json!({"criteria": criteria()})).needs_model());
        assert!(
            !rubric(
                json!({"criteria": [{"id": "g", "gate": true, "judge": "test", "check": "t"}]})
            )
            .needs_model()
        );
    }
}
