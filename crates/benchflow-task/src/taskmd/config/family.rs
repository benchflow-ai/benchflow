//! Task families: `[family.splits]` and the rules that pick each split's
//! seeds (family@1's grammar, as the reference's `parse_rule` reads it).
//!
//! These checks are static. A family's generator is never run here: `bench
//! check` reads files and runs no package code.

use serde_json::Value;

use crate::source::join;

use super::super::code;
use super::super::pyfmt::{json_dumps, prefix, repr_str};
use super::super::pyval::table;
use super::super::report::At;
use super::super::vocab::SPLIT_ROLES;
use super::Checker;

/// How many seeds the overlap check covers.
const SEEDS_CHECKED: i128 = 100_000;

/// Why a rule is not valid, in the reference's words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuleError(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arith {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone)]
enum Node {
    Int(i128),
    Seed,
    Neg(Box<Node>),
    Arith(Arith, Box<Node>, Box<Node>),
    Chain(Vec<Node>, Vec<Cmp>),
    Not(Box<Node>),
    And(Box<Node>, Box<Node>),
    Or(Box<Node>, Box<Node>),
}

impl Node {
    fn is_bool(&self) -> bool {
        matches!(
            self,
            Node::Chain(..) | Node::Not(_) | Node::And(..) | Node::Or(..)
        )
    }

    fn int(&self, seed: i128) -> Result<i128, RuleError> {
        let too_big = || RuleError("a number in the rule is too large".into());
        match self {
            Node::Int(n) => Ok(*n),
            Node::Seed => Ok(seed),
            Node::Neg(x) => x.int(seed)?.checked_neg().ok_or_else(too_big),
            Node::Arith(op, a, b) => {
                let (a, b) = (a.int(seed)?, b.int(seed)?);
                match op {
                    Arith::Add => a.checked_add(b).ok_or_else(too_big),
                    Arith::Sub => a.checked_sub(b).ok_or_else(too_big),
                    Arith::Mul => a.checked_mul(b).ok_or_else(too_big),
                    Arith::FloorDiv | Arith::Mod => {
                        if b == 0 {
                            return Err(RuleError("division by zero".into()));
                        }
                        // Python's // floors, and % takes the divisor's sign.
                        let mut r = a % b;
                        if r != 0 && ((r < 0) != (b < 0)) {
                            r += b;
                        }
                        Ok(if *op == Arith::Mod { r } else { (a - r) / b })
                    }
                }
            }
            _ => Ok(i128::from(self.truth(seed)?)),
        }
    }

    fn truth(&self, seed: i128) -> Result<bool, RuleError> {
        match self {
            Node::Chain(terms, ops) => {
                let values: Vec<i128> = terms
                    .iter()
                    .map(|t| t.int(seed))
                    .collect::<Result<_, _>>()?;
                Ok(ops.iter().enumerate().all(|(k, op)| {
                    let (a, b) = (values[k], values[k + 1]);
                    match op {
                        Cmp::Eq => a == b,
                        Cmp::Ne => a != b,
                        Cmp::Lt => a < b,
                        Cmp::Le => a <= b,
                        Cmp::Gt => a > b,
                        Cmp::Ge => a >= b,
                    }
                }))
            }
            Node::Not(x) => Ok(!x.truth(seed)?),
            Node::And(a, b) => Ok(a.truth(seed)? && b.truth(seed)?),
            Node::Or(a, b) => Ok(a.truth(seed)? || b.truth(seed)?),
            other => Ok(other.int(seed)? != 0),
        }
    }
}

/// A parsed split rule: which seeds it selects.
#[derive(Debug, Clone)]
pub(crate) struct Rule(Node);

impl Rule {
    /// Whether the rule selects `seed`.
    pub(crate) fn selects(&self, seed: i128) -> Result<bool, RuleError> {
        self.0.truth(seed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tok {
    Int,
    Name,
    Op,
    End,
}

struct Parser<'a> {
    tokens: Vec<(Tok, String)>,
    at: usize,
    variable: &'a str,
}

/// Parse a split rule by family@1's grammar: comparisons of integer
/// arithmetic over one variable, combined with `and`, `or`, and `not`.
pub(crate) fn parse_rule(text: &str, variable: &str) -> Result<Rule, RuleError> {
    let mut p = Parser {
        tokens: tokenize(text)?,
        at: 0,
        variable,
    };
    let node = p.rule()?;
    if p.peek().0 != Tok::End {
        return Err(RuleError(format!("unexpected {}", repr_str(&p.peek().1))));
    }
    if !node.is_bool() {
        return Err(RuleError(
            "a rule is a comparison, or comparisons combined with and, or, and not".into(),
        ));
    }
    Ok(Rule(node))
}

fn tokenize(text: &str) -> Result<Vec<(Tok, String)>, RuleError> {
    let mut tokens = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let rest = &text[pos..];
        if rest.trim().is_empty() {
            break;
        }
        let skipped = rest.len() - rest.trim_start().len();
        let start = pos + skipped;
        let s = &text[start..];
        let c = s.chars().next().unwrap_or(' ');
        let (kind, len) = if c.is_ascii_digit() {
            (Tok::Int, s.chars().take_while(char::is_ascii_digit).count())
        } else if c.is_ascii_alphabetic() || c == '_' {
            (
                Tok::Name,
                s.chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .count(),
            )
        } else if ["==", "!=", "<=", ">=", "//"]
            .iter()
            .any(|op| s.starts_with(op))
        {
            (Tok::Op, 2)
        } else if "<>+-*%()".contains(c) {
            (Tok::Op, 1)
        } else {
            return Err(RuleError(format!(
                "unexpected {}",
                repr_str(prefix(rest.trim(), 12))
            )));
        };
        tokens.push((kind, s[..len].to_string()));
        pos = start + len;
    }
    tokens.push((Tok::End, String::new()));
    Ok(tokens)
}

impl Parser<'_> {
    fn peek(&self) -> &(Tok, String) {
        &self.tokens[self.at.min(self.tokens.len() - 1)]
    }

    fn take(&mut self, value: Option<&str>) -> Result<(Tok, String), RuleError> {
        let tok = self.peek().clone();
        if let Some(v) = value
            && tok.1 != v
        {
            let found = if tok.1.is_empty() {
                "the end"
            } else {
                tok.1.as_str()
            };
            return Err(RuleError(format!(
                "expected {}, found {}",
                repr_str(v),
                repr_str(found)
            )));
        }
        self.at += 1;
        Ok(tok)
    }

    fn is_name(&self, word: &str) -> bool {
        let t = self.peek();
        t.0 == Tok::Name && t.1 == word
    }

    fn rule(&mut self) -> Result<Node, RuleError> {
        self.or()
    }

    fn or(&mut self) -> Result<Node, RuleError> {
        let mut left = self.and()?;
        while self.is_name("or") {
            self.take(None)?;
            let right = self.and()?;
            let (a, b) = (as_bool(left)?, as_bool(right)?);
            left = Node::Or(Box::new(a), Box::new(b));
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Node, RuleError> {
        let mut left = self.not()?;
        while self.is_name("and") {
            self.take(None)?;
            let right = self.not()?;
            let (a, b) = (as_bool(left)?, as_bool(right)?);
            left = Node::And(Box::new(a), Box::new(b));
        }
        Ok(left)
    }

    fn not(&mut self) -> Result<Node, RuleError> {
        if self.is_name("not") {
            self.take(None)?;
            let inner = self.not()?;
            return Ok(Node::Not(Box::new(as_bool(inner)?)));
        }
        self.cmp()
    }

    fn cmp(&mut self) -> Result<Node, RuleError> {
        let left = self.sum()?;
        let mut ops = Vec::new();
        let mut terms = vec![left];
        while let Some(op) = cmp_op(&self.peek().1) {
            self.take(None)?;
            ops.push(op);
            terms.push(self.sum()?);
        }
        if ops.is_empty() {
            return Ok(terms.pop().expect("one term"));
        }
        let terms = terms
            .into_iter()
            .map(as_int)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Node::Chain(terms, ops))
    }

    fn sum(&mut self) -> Result<Node, RuleError> {
        let mut left = self.term()?;
        while matches!(self.peek().1.as_str(), "+" | "-") {
            let op = if self.take(None)?.1 == "+" {
                Arith::Add
            } else {
                Arith::Sub
            };
            let a = as_int(left)?;
            let b = as_int(self.term()?)?;
            left = Node::Arith(op, Box::new(a), Box::new(b));
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<Node, RuleError> {
        let mut left = self.factor()?;
        while matches!(self.peek().1.as_str(), "*" | "//" | "%") {
            let op = match self.take(None)?.1.as_str() {
                "*" => Arith::Mul,
                "//" => Arith::FloorDiv,
                _ => Arith::Mod,
            };
            let a = as_int(left)?;
            let b = as_int(self.factor()?)?;
            left = Node::Arith(op, Box::new(a), Box::new(b));
        }
        Ok(left)
    }

    fn factor(&mut self) -> Result<Node, RuleError> {
        let (kind, value) = self.peek().clone();
        if value == "-" {
            self.take(None)?;
            let inner = as_int(self.factor()?)?;
            return Ok(Node::Neg(Box::new(inner)));
        }
        if kind == Tok::Int {
            self.take(None)?;
            let n = value
                .parse::<i128>()
                .map_err(|_| RuleError("a number in the rule is too large".into()))?;
            return Ok(Node::Int(n));
        }
        if kind == Tok::Name && !matches!(value.as_str(), "and" | "or" | "not") {
            self.take(None)?;
            if value != self.variable {
                return Err(RuleError(format!(
                    "a rule reads one variable, {}, not {}",
                    self.variable,
                    repr_str(&value)
                )));
            }
            return Ok(Node::Seed);
        }
        if value == "(" {
            self.take(None)?;
            let inner = self.rule()?;
            self.take(Some(")"))?;
            return Ok(inner);
        }
        let shown = if value.is_empty() {
            "end of rule"
        } else {
            value.as_str()
        };
        Err(RuleError(format!("unexpected {}", repr_str(shown))))
    }
}

fn cmp_op(s: &str) -> Option<Cmp> {
    Some(match s {
        "==" => Cmp::Eq,
        "!=" => Cmp::Ne,
        "<" => Cmp::Lt,
        "<=" => Cmp::Le,
        ">" => Cmp::Gt,
        ">=" => Cmp::Ge,
        _ => return None,
    })
}

fn as_bool(node: Node) -> Result<Node, RuleError> {
    if node.is_bool() {
        Ok(node)
    } else {
        Err(RuleError(
            "and, or, and not combine comparisons, not numbers".into(),
        ))
    }
}

fn as_int(node: Node) -> Result<Node, RuleError> {
    if node.is_bool() {
        Err(RuleError(
            "arithmetic and comparisons take numbers, not comparisons".into(),
        ))
    } else {
        Ok(node)
    }
}

/// Seeds that two splits both select, over the first [`SEEDS_CHECKED`]
/// seeds: at most three messages, one per pair of splits (the reference's
/// `_family_overlaps`). `rules` must be sorted by name.
fn overlaps(rules: &[(String, Rule)]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();
    for seed in 0..SEEDS_CHECKED {
        let mut hits = Vec::new();
        for (name, rule) in rules {
            match rule.selects(seed) {
                Ok(true) => hits.push(name.as_str()),
                Ok(false) => {}
                Err(e) => {
                    out.push(format!("[family.splits]: a rule fails on a seed: {}", e.0));
                    return out;
                }
            }
        }
        if hits.len() > 1 {
            let pair = (hits[0].to_string(), hits[1].to_string());
            if !seen.contains(&pair) {
                out.push(format!(
                    "[family.splits] {} and {} both select seed {seed}: no seed may fall in two splits",
                    hits[0], hits[1]
                ));
                seen.push(pair);
                if out.len() >= 3 {
                    break;
                }
            }
        }
    }
    out
}

/// `[family]`: a warning for `database`, and `[family.splits]`: each a rule
/// named by its role, or a table with a rule and a role, with no seed in two
/// splits (the reference's `check_family`).
pub(super) fn check(c: &mut Checker<'_>) {
    let family = table(c.config, &["family"]);
    if family.contains_key("database") {
        c.warn(
            code::FAMILY,
            At::Key("family.database"),
            "[family] database is not read under family@1: the generator writes an instance's files into agent/ and verifier/ \
(docs/runtime/episodes.md)",
        );
    }
    let Some(splits) = family.get("splits").filter(|v| !v.is_null()) else {
        return;
    };
    let Value::Object(splits) = splits else {
        c.err(
            code::FAMILY,
            At::Value("family.splits"),
            "[family.splits] is a table of splits",
        );
        return;
    };
    let variable = match family.get("seed_param") {
        None => "seed",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => "seed",
    };
    let mut rules: Vec<(String, Rule)> = Vec::new();
    for (name, entry) in splits {
        if name.starts_with("x-") {
            continue;
        }
        let path = join("family.splits", name);
        let rule = match entry {
            Value::String(rule) => {
                if !SPLIT_ROLES.contains(&name.as_str()) {
                    c.err(
                        code::FAMILY,
                        At::Key(&path),
                        format!(
                            "[family.splits] {name} needs a role: write [family.splits.{name}] with rule = {} and role = \"train\", \
\"validation\", or \"test\"",
                            json_dumps(entry)
                        ),
                    );
                }
                rule.as_str()
            }
            Value::Object(t) => {
                if !t.contains_key("role") && !SPLIT_ROLES.contains(&name.as_str()) {
                    c.err(
                        code::FAMILY,
                        At::Key(&path),
                        format!("[family.splits.{name}] needs a role: \"train\", \"validation\", or \"test\""),
                    );
                }
                match t.get("rule") {
                    Some(Value::String(r)) if !r.trim().is_empty() => r.as_str(),
                    _ => {
                        c.err(
                            code::FAMILY,
                            At::Key(&path),
                            format!("[family.splits.{name}] needs a rule over the seed, such as \"seed % 10 == 9\""),
                        );
                        continue;
                    }
                }
            }
            _ => {
                c.err(
                    code::FAMILY,
                    At::Value(&path),
                    format!("[family.splits] {name} is a rule, or a table with rule and role"),
                );
                continue;
            }
        };
        match parse_rule(rule, variable) {
            Ok(r) => rules.push((name.clone(), r)),
            Err(e) => {
                let at = if entry.is_object() {
                    join(&path, "rule")
                } else {
                    path.clone()
                };
                c.err(
                    code::FAMILY,
                    At::Value(&at),
                    format!(
                        "[family.splits] {name}: the rule {} is not valid: {}",
                        repr_str(rule),
                        e.0
                    ),
                );
            }
        }
    }
    // The scan is the slow part of a family's check; testing a fix skips it.
    if rules.len() < 2 || c.doc.is_none() {
        return;
    }
    rules.sort_by(|a, b| a.0.cmp(&b.0));
    for message in overlaps(&rules) {
        c.err(code::FAMILY, At::Key("family.splits"), message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picks(rule: &str, seed: i128) -> bool {
        parse_rule(rule, "seed").unwrap().selects(seed).unwrap()
    }

    fn error(rule: &str) -> String {
        parse_rule(rule, "seed").unwrap_err().0
    }

    #[test]
    fn rules_evaluate_like_python() {
        assert!(picks("seed % 10 < 8", 7));
        assert!(!picks("seed % 10 < 8", 9));
        assert!(picks("seed % 10 == 8 or seed % 10 == 9", 19));
        assert!(picks("not seed < 5", 5));
        assert!(picks("1 < seed < 3", 2) && !picks("1 < seed < 3", 3));
        assert!(picks("-7 // 2 == -4", 0));
        assert!(picks("-7 % 3 == 2", 0));
        assert!(picks("7 % -3 == -2", 0));
        assert!(picks("(seed + 1) * 2 == 6", 2));
    }

    #[test]
    fn errors_read_like_the_reference() {
        assert_eq!(error("seed $ 3"), "unexpected '$ 3'");
        assert_eq!(
            error("seed + 1"),
            "a rule is a comparison, or comparisons combined with and, or, and not"
        );
        assert_eq!(error("x < 3"), "a rule reads one variable, seed, not 'x'");
        assert_eq!(
            error("seed < 3 and 4"),
            "and, or, and not combine comparisons, not numbers"
        );
        assert_eq!(
            error("(seed < 3) + 1 < 2"),
            "arithmetic and comparisons take numbers, not comparisons"
        );
        assert_eq!(error("(seed < 3"), "expected ')', found 'the end'");
        assert_eq!(error("seed <"), "unexpected 'end of rule'");
        assert_eq!(error("seed < 3 3"), "unexpected '3'");
        let r = parse_rule("seed // 0 == 1", "seed").unwrap();
        assert_eq!(r.selects(1).unwrap_err().0, "division by zero");
    }

    #[test]
    fn overlaps_name_the_first_seed() {
        let rules = vec![
            (
                "a".to_string(),
                parse_rule("seed % 10 < 5", "seed").unwrap(),
            ),
            (
                "b".to_string(),
                parse_rule("seed % 10 > 3", "seed").unwrap(),
            ),
        ];
        assert_eq!(
            overlaps(&rules),
            ["[family.splits] a and b both select seed 4: no seed may fall in two splits"]
        );
    }
}
