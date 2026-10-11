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

/// The 95% Wilson score interval for `k` successes in `n` trials, with z
/// the exact 97.5% normal quantile (as the viewer's). It is 0.8's
/// `pass_at_k.wilson_interval` formula, but 0.8 uses z = 1.96 there, so
/// 0.8's endpoints differ by up to 1e-5 for n <= 40. The two do not feed
/// the same number anyway: summary.json's `score_interval` (a Rust field)
/// counts every trial, while 0.8's `solve_rates.solve_rate_interval`
/// leaves out unscored trials and widens for repeated trials of a task
/// (`wilson-clustered`).
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
    /// The 95% interval's bounds; `None` when there is none (see `no_interval`).
    pub low: Option<f64>,
    pub high: Option<f64>,
    /// Why there is no interval, when there is none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_interval: Option<&'static str>,
    /// Tasks where A did better, worse, and the same.
    pub a_better: usize,
    pub b_better: usize,
    pub ties: usize,
}

/// The fewest shared tasks a paired interval is given for. With two, the t
/// critical value is 12.7 and the interval covers the whole range anyway.
pub const MIN_PAIRED_TASKS: usize = 3;

/// Why a paired interval is missing: too few tasks.
pub const TOO_FEW_TASKS: &str = "too few tasks";
/// Why a paired interval is missing: no spread to make one from.
pub const SAME_DIFFERENCE: &str = "every task differs by the same amount";

/// The 95% t-interval of the mean of `diffs`, clipped to `[-bound, bound]`
/// (the furthest apart two rewards can be), or why there is none: fewer
/// than [`MIN_PAIRED_TASKS`] differences, or all of them equal (no spread,
/// so the interval would have zero width).
pub fn paired_interval(diffs: &[f64], bound: f64) -> Result<(f64, f64), &'static str> {
    let n = diffs.len();
    if n < MIN_PAIRED_TASKS {
        return Err(TOO_FEW_TASKS);
    }
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    if var <= 1e-24 {
        return Err(SAME_DIFFERENCE);
    }
    let half = t_975(n - 1) * (var / n as f64).sqrt();
    Ok(((mean - half).max(-bound), (mean + half).min(bound)))
}

/// Compare per-task mean rewards of two runs (A minus B), with a 95%
/// interval from the t distribution over task differences.
pub fn paired(a: &BTreeMap<String, TaskStats>, b: &BTreeMap<String, TaskStats>) -> Option<Paired> {
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .filter_map(|(task, sa)| {
            let sb = b.get(task)?;
            Some((sa.mean_reward()?, sb.mean_reward()?))
        })
        .collect();
    let diffs: Vec<f64> = pairs.iter().map(|(x, y)| x - y).collect();
    let n = diffs.len();
    if n == 0 {
        return None;
    }
    let mean = diffs.iter().sum::<f64>() / n as f64;
    // Rewards are usually 0 to 1; wider ones widen the possible range.
    let (lo, hi) = pairs
        .iter()
        .flat_map(|(x, y)| [*x, *y])
        .fold((0.0_f64, 1.0_f64), |(lo, hi), r| (lo.min(r), hi.max(r)));
    let interval = paired_interval(&diffs, hi - lo);
    Some(Paired {
        tasks: n,
        diff: mean,
        low: interval.ok().map(|i| i.0),
        high: interval.ok().map(|i| i.1),
        no_interval: interval.err(),
        a_better: diffs.iter().filter(|d| **d > 1e-12).count(),
        b_better: diffs.iter().filter(|d| **d < -1e-12).count(),
        ties: diffs.iter().filter(|d| d.abs() <= 1e-12).count(),
    })
}

/// The 97.5th percentile of Student's t with `df` degrees of freedom: the
/// exact quantile, as the viewer computes it, so `bench results` and
/// `/compare` give the same interval at any number of tasks.
fn t_975(df: usize) -> f64 {
    if df == 0 {
        return f64::INFINITY;
    }
    t_quantile_975(df as f64)
}

/// Found by bisection on the distribution function.
fn t_quantile_975(df: f64) -> f64 {
    let (mut low, mut high) = (0.0_f64, 1000.0_f64);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if t_cdf(middle, df) < 0.975 {
            low = middle;
        } else {
            high = middle;
        }
    }
    0.5 * (low + high)
}

/// P(T <= t) for t >= 0.
fn t_cdf(t: f64, df: f64) -> f64 {
    1.0 - 0.5 * regularized_beta(df / (df + t * t), df / 2.0, 0.5)
}

/// The regularized incomplete beta function I_x(a, b), by its continued
/// fraction (Numerical Recipes' `betai`).
fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(x, a, b) / a
    } else {
        1.0 - front * beta_fraction(1.0 - x, b, a) / b
    }
}

fn beta_fraction(x: f64, a: f64, b: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = f64::from(m);
        let even = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        for coefficient in [
            even,
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0)),
        ] {
            d = 1.0 + coefficient * d;
            if d.abs() < TINY {
                d = TINY;
            }
            c = 1.0 + coefficient / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            h *= d * c;
        }
        if (d * c - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// ln Γ(x) for x > 0 (Lanczos, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let series = COEFFICIENTS[1..]
        .iter()
        .enumerate()
        .fold(COEFFICIENTS[0], |sum, (index, coefficient)| {
            sum + coefficient / (x + index as f64 + 1.0)
        });
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pass@k and the Wilson interval against 0.8's `pass_at_k.py` (Python
    /// 3.12) on every n up to 40, every c and k: the fixture holds 0.8's
    /// pass@k (`null` when n < k, `"error"` for k = 0, which 0.8 refuses)
    /// and 0.8's `wilson_interval(k, n, z)` called with this crate's z (the
    /// fixture's `wilson_z`; see [`wilson`] for 0.8's own 1.96), so the
    /// endpoints must agree to the last bit or two.
    /// `tests/fixtures/make_pass_at_k.py` writes the fixture.
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
        assert_eq!(v["wilson_z"].as_f64(), Some(1.959_963_984_540_054));
        let mut n_wilson = 0;
        for case in v["wilson"].as_array().unwrap() {
            let (Some(lo), Some(hi)) = (case["lo"].as_f64(), case["hi"].as_f64()) else {
                continue;
            };
            let (l, h) = wilson(
                case["k"].as_u64().unwrap() as usize,
                case["n"].as_u64().unwrap() as usize,
            );
            assert!(
                (l - lo).abs() <= 4.0 * f64::EPSILON && (h - hi).abs() <= 4.0 * f64::EPSILON,
                "{case}: {l} {h}"
            );
            n_wilson += 1;
        }
        assert_eq!(n_wilson, 860);
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
        // Differences 1, 0, 0: the t-interval (-1.101 to +1.768) is clipped
        // to the most a 0-to-1 reward can differ by.
        assert_eq!((p.low, p.high), (Some(-1.0), Some(1.0)));
    }

    /// Cases `bench results` printed as intervals on 2026-10-10: two tied
    /// tasks (+0.000 to +0.000), A passing both and B failing both (+1.000
    /// to +1.000), and differences 1 and 0 (-5.853 to +6.853).
    #[test]
    fn no_paired_interval_without_spread_or_with_too_few_tasks() {
        assert_eq!(paired_interval(&[0.0], 1.0), Err(TOO_FEW_TASKS));
        assert_eq!(paired_interval(&[1.0, 0.0], 1.0), Err(TOO_FEW_TASKS));
        assert_eq!(paired_interval(&[0.0, 0.0, 0.0], 1.0), Err(SAME_DIFFERENCE));
        assert_eq!(paired_interval(&[1.0; 4], 1.0), Err(SAME_DIFFERENCE));
        // 0.8 +- 2.776 * 0.2: the upper bound is clipped at 1.
        let (low, high) = paired_interval(&[1.0, 1.0, 0.0, 1.0, 1.0], 1.0).unwrap();
        assert!((low - 0.2448).abs() < 1e-3 && high == 1.0, "{low} {high}");
        // Behavior review 10: 40 tasks (+1 x14, 0 x20, -1 x6). The step
        // table used 2.000 for df 39 and printed -0.017 to +0.417; the exact
        // quantile (2.0227) gives the viewer's -0.020 to +0.420.
        let mut diffs = vec![1.0; 14];
        diffs.extend([0.0; 20]);
        diffs.extend([-1.0; 6]);
        let (low, high) = paired_interval(&diffs, 1.0).unwrap();
        assert!(
            (low + 0.020).abs() < 1e-3 && (high - 0.420).abs() < 1e-3,
            "{low} {high}"
        );
        let (low, high) = paired_interval(&[-1.0, 1.0, 0.5], 1.0).unwrap();
        assert_eq!((low, high), (-1.0, 1.0));
        // Rewards on a wider scale widen the bound.
        let (_, high) = paired_interval(&[3.0, 0.0, 1.0], 3.0).unwrap();
        assert!(high > 1.0 && high <= 3.0, "{high}");
    }
}
