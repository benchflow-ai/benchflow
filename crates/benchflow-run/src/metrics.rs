//! Numbers about trials: pass rates with intervals, pass@k, and paired
//! comparisons between runs.

use std::collections::BTreeMap;

use serde::Serialize;

/// A proportion with a 95% Wilson score interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Rate {
    pub successes: usize,
    pub n: usize,
    pub value: f64,
    pub low: f64,
    pub high: f64,
}

impl Rate {
    pub fn new(successes: usize, n: usize) -> Self {
        if n == 0 {
            return Self {
                successes,
                n,
                value: 0.0,
                low: 0.0,
                high: 0.0,
            };
        }
        let (low, high) = wilson(successes, n);
        Self {
            successes,
            n,
            value: successes as f64 / n as f64,
            low,
            high,
        }
    }

    /// `62.5% (40.1–80.2)`.
    pub fn display(&self) -> String {
        if self.n == 0 {
            return "–".into();
        }
        format!(
            "{:.1}% ({:.1}–{:.1})",
            self.value * 100.0,
            self.low * 100.0,
            self.high * 100.0
        )
    }
}

/// The 95% Wilson score interval for `k` successes in `n` trials.
pub fn wilson(k: usize, n: usize) -> (f64, f64) {
    let z = 1.959_963_984_540_054_f64;
    let n_f = n as f64;
    let p = k as f64 / n_f;
    let denom = 1.0 + z * z / n_f;
    let center = (p + z * z / (2.0 * n_f)) / denom;
    let half = z * ((p * (1.0 - p) / n_f + z * z / (4.0 * n_f * n_f)).sqrt()) / denom;
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// The unbiased pass@k estimator: the chance that at least one of k samples
/// drawn from n trials (c of them passing) passes.
pub fn pass_at_k(n: usize, c: usize, k: usize) -> Option<f64> {
    if k == 0 || n < k {
        return None;
    }
    if n - c < k {
        return Some(1.0);
    }
    // 1 - C(n-c, k) / C(n, k), as a running product for stability.
    let mut prod = 1.0f64;
    for i in (n - c + 1)..=n {
        prod *= 1.0 - k as f64 / i as f64;
    }
    Some(1.0 - prod)
}

/// Per-task results across repeated trials.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TaskStats {
    pub trials: usize,
    pub passed: usize,
    pub scored: usize,
    pub reward_sum: f64,
}

impl TaskStats {
    pub fn mean_reward(&self) -> Option<f64> {
        (self.scored > 0).then(|| self.reward_sum / self.scored as f64)
    }
}

/// pass@k averaged over tasks, for every k up to the smallest scored-trial
/// count. A task with no scored trial is left out, as 0.8 leaves it out.
pub fn pass_at_k_table(tasks: &BTreeMap<String, TaskStats>) -> BTreeMap<usize, f64> {
    let mut out = BTreeMap::new();
    let min_n = tasks
        .values()
        .map(|t| t.scored)
        .filter(|&n| n > 0)
        .min()
        .unwrap_or(0);
    for k in 1..=min_n {
        let vals: Vec<f64> = tasks
            .values()
            .filter_map(|t| pass_at_k(t.scored, t.passed, k))
            .collect();
        if !vals.is_empty() {
            out.insert(k, vals.iter().sum::<f64>() / vals.len() as f64);
        }
    }
    out
}

/// A paired comparison of two runs over the tasks both scored.
#[derive(Debug, Clone, Serialize)]
pub struct Paired {
    pub tasks: usize,
    /// Mean per-task reward of A minus B.
    pub diff: f64,
    pub low: f64,
    pub high: f64,
    /// Tasks where A did better, worse, and the same.
    pub a_better: usize,
    pub b_better: usize,
    pub ties: usize,
}

/// Compare per-task mean rewards of two runs (A minus B), with a 95%
/// interval from the t distribution over task differences.
pub fn paired(a: &BTreeMap<String, TaskStats>, b: &BTreeMap<String, TaskStats>) -> Option<Paired> {
    let diffs: Vec<f64> = a
        .iter()
        .filter_map(|(task, sa)| {
            let sb = b.get(task)?;
            Some(sa.mean_reward()? - sb.mean_reward()?)
        })
        .collect();
    let n = diffs.len();
    if n == 0 {
        return None;
    }
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let (low, high) = if n > 1 {
        let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let se = (var / n as f64).sqrt();
        let t = t_975(n - 1);
        (mean - t * se, mean + t * se)
    } else {
        (mean, mean)
    };
    Some(Paired {
        tasks: n,
        diff: mean,
        low,
        high,
        a_better: diffs.iter().filter(|d| **d > 1e-12).count(),
        b_better: diffs.iter().filter(|d| **d < -1e-12).count(),
        ties: diffs.iter().filter(|d| d.abs() <= 1e-12).count(),
    })
}

/// The 97.5th percentile of Student's t with `df` degrees of freedom.
fn t_975(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    match df {
        0 => f64::INFINITY,
        1..=30 => TABLE[df - 1],
        31..=60 => 2.000,
        61..=120 => 1.980,
        _ => 1.960,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pass@k and the Wilson interval against 0.8's `pass_at_k.py` (Python
    /// 3.12) on every n up to 40, every c and k: the fixture holds 0.8's
    /// value (`null` when n < k, `"error"` for k = 0, which 0.8 refuses).
    /// The Wilson z differs in the seventh digit (0.8's 1.959964), hence the
    /// tolerance.
    #[test]
    fn pass_at_k_and_wilson_match_08() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/python-pass-at-k.json");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut n_cases = 0;
        for case in v["pass_at_k"].as_array().unwrap() {
            let get = |k: &str| case[k].as_u64().unwrap() as usize;
            let got = pass_at_k(get("n"), get("c"), get("k"));
            match &case["pass_at_k"] {
                serde_json::Value::Number(want) => {
                    let (want, got) = (want.as_f64().unwrap(), got.unwrap());
                    assert!((want - got).abs() < 1e-12, "{case}: {got}");
                }
                _ => assert_eq!(got, None, "{case}"),
            }
            n_cases += 1;
        }
        assert!(n_cases > 20_000);
        for case in v["wilson"].as_array().unwrap() {
            let (Some(lo), Some(hi)) = (case["lo"].as_f64(), case["hi"].as_f64()) else {
                continue;
            };
            let (l, h) = wilson(
                case["k"].as_u64().unwrap() as usize,
                case["n"].as_u64().unwrap() as usize,
            );
            assert!(
                (l - lo).abs() < 1e-6 && (h - hi).abs() < 1e-6,
                "{case}: {l} {h}"
            );
        }
    }

    /// Guards the D5 fix: one task whose trials all errored made the
    /// smallest trial count 0, so summary.json's pass@k was `{}`. 0.8 leaves
    /// unscored tasks out (`pass_at_k.py`) and gives {1: 0.5, 2: 1.0} here.
    #[test]
    fn an_unscored_task_does_not_empty_pass_at_k() {
        let tasks = BTreeMap::from([
            (
                "a".to_string(),
                TaskStats {
                    trials: 2,
                    passed: 1,
                    scored: 2,
                    reward_sum: 1.0,
                },
            ),
            (
                "b".to_string(),
                TaskStats {
                    trials: 2,
                    passed: 0,
                    scored: 0,
                    reward_sum: 0.0,
                },
            ),
        ]);
        assert_eq!(
            pass_at_k_table(&tasks),
            BTreeMap::from([(1, 0.5), (2, 1.0)])
        );
    }

    #[test]
    fn wilson_known_values() {
        let (l, h) = wilson(5, 10);
        assert!((l - 0.2366).abs() < 1e-3 && (h - 0.7634).abs() < 1e-3);
        let (l, h) = wilson(0, 10);
        assert_eq!(l, 0.0);
        assert!((h - 0.2775).abs() < 1e-3);
    }

    #[test]
    fn pass_at_k_matches_formula() {
        assert_eq!(pass_at_k(5, 0, 1), Some(0.0));
        assert_eq!(pass_at_k(5, 5, 3), Some(1.0));
        assert!((pass_at_k(5, 2, 1).unwrap() - 0.4).abs() < 1e-12);
        // 1 - C(3,2)/C(5,2) = 1 - 3/10
        assert!((pass_at_k(5, 2, 2).unwrap() - 0.7).abs() < 1e-12);
        assert_eq!(pass_at_k(2, 1, 3), None);
    }

    #[test]
    fn paired_diff() {
        let mk = |r: &[(&str, f64)]| {
            r.iter()
                .map(|(t, v)| {
                    (
                        t.to_string(),
                        TaskStats {
                            trials: 1,
                            passed: (*v == 1.0) as usize,
                            scored: 1,
                            reward_sum: *v,
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        let a = mk(&[("x", 1.0), ("y", 1.0), ("z", 0.0)]);
        let b = mk(&[("x", 0.0), ("y", 1.0), ("z", 0.0), ("w", 1.0)]);
        let p = paired(&a, &b).unwrap();
        assert_eq!(p.tasks, 3);
        assert!((p.diff - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!((p.a_better, p.b_better, p.ties), (1, 0, 2));
    }
}
