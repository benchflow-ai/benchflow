//! The few statistics the compare, outcomes, and Pareto views show: a pass rate's
//! Wilson interval, a paired mean difference's t-interval, and percentiles.

/// The two-sided 95% normal quantile.
const Z95: f64 = 1.959_963_984_540_054;

/// The 95% Wilson score interval for `passed` successes in `total` trials
/// (the interval `bench` writes to `summary.json`); `None` with no trials.
pub fn wilson(passed: usize, total: usize) -> Option<(f64, f64)> {
    if total == 0 {
        return None;
    }
    let n = total as f64;
    let p = passed as f64 / n;
    let z2 = Z95 * Z95;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    Some(((center - half).max(0.0), (center + half).min(1.0)))
}

/// A paired comparison: the mean of `differences` with its 95% t-interval,
/// or why there is none. The rules are `bench results`': no interval from
/// fewer than three pairs or from differences that are all the same, and
/// none reaching past the furthest two rewards can be apart.
#[derive(Debug, Clone, PartialEq)]
pub struct MeanInterval {
    pub mean: f64,
    pub interval: Option<(f64, f64)>,
    pub no_interval: Option<&'static str>,
}

/// The fewest pairs an interval is given for.
const MIN_PAIRS: usize = 3;

/// `bound` is the most two rewards can differ by (1 for rewards 0 to 1).
pub fn mean_with_t_interval(differences: &[f64], bound: f64) -> Option<MeanInterval> {
    let n = differences.len();
    if n == 0 {
        return None;
    }
    let mean = differences.iter().sum::<f64>() / n as f64;
    let none = |why| {
        Some(MeanInterval {
            mean,
            interval: None,
            no_interval: Some(why),
        })
    };
    if n < MIN_PAIRS {
        return none("too few tasks");
    }
    let variance = differences
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;
    if variance <= 1e-24 {
        return none("every task differs by the same amount");
    }
    let half = t_quantile_975((n - 1) as f64) * (variance / n as f64).sqrt();
    Some(MeanInterval {
        mean,
        interval: Some(((mean - half).max(-bound), (mean + half).min(bound))),
        no_interval: None,
    })
}

/// The 97.5th percentile of Student's t with `df` degrees of freedom (the
/// critical value of a two-sided 95% interval), found by bisection on the
/// distribution function.
pub fn t_quantile_975(df: f64) -> f64 {
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

/// The `q` quantile (0 to 1) of `values` by linear interpolation between
/// order statistics (NumPy's default); `None` when there are none.
pub fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let below = position.floor() as usize;
    let above = position.ceil() as usize;
    let weight = position - below as f64;
    Some(sorted[below] + (sorted[above] - sorted[below]) * weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} is not {expected}"
        );
    }

    #[test]
    fn wilson_matches_the_engine() {
        // The fake job's summary.json: 2 of 4 passed.
        let (low, high) = wilson(2, 4).unwrap();
        close(low, 0.150_038_989_152_149_47, 1e-12);
        close(high, 0.849_961_010_847_850_6, 1e-12);
        assert_eq!(wilson(0, 0), None);
        let (low, high) = wilson(10, 10).unwrap();
        close(low, 0.722_467, 1e-6);
        close(high, 1.0, 1e-12);
    }

    #[test]
    fn t_critical_values_match_tables() {
        for (df, expected) in [
            (1.0, 12.706_204_736),
            (2.0, 4.302_652_730),
            (5.0, 2.570_581_836),
            (10.0, 2.228_138_852),
            (30.0, 2.042_272_456),
            (1000.0, 1.962_339_081),
        ] {
            close(t_quantile_975(df), expected, 1e-6);
        }
    }

    #[test]
    fn paired_means_carry_a_t_interval() {
        assert_eq!(mean_with_t_interval(&[], 1.0), None);
        let none = |mean, why| MeanInterval {
            mean,
            interval: None,
            no_interval: Some(why),
        };
        assert_eq!(
            mean_with_t_interval(&[0.5], 1.0),
            Some(none(0.5, "too few tasks"))
        );
        assert_eq!(
            mean_with_t_interval(&[1.0, 0.0], 1.0),
            Some(none(0.5, "too few tasks"))
        );
        assert_eq!(
            mean_with_t_interval(&[1.0, 1.0, 1.0], 1.0),
            Some(none(1.0, "every task differs by the same amount"))
        );
        // mean 0.5, sd 0.5, n 3: half-width 4.3027 * 0.5 / sqrt(3).
        let paired = mean_with_t_interval(&[1.0, 0.0, 0.5], 2.0).unwrap();
        close(paired.mean, 0.5, 1e-12);
        let (low, high) = paired.interval.unwrap();
        close(high - 0.5, 1.242_068_856, 1e-6);
        close(0.5 - low, 1.242_068_856, 1e-6);
        // ... clipped to what rewards from 0 to 1 allow.
        let paired = mean_with_t_interval(&[1.0, 0.0, 0.5], 1.0).unwrap();
        assert_eq!(paired.interval.unwrap().1, 1.0);
    }

    #[test]
    fn quantiles_interpolate_like_numpy() {
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[3.0], 0.9), Some(3.0));
        assert_eq!(quantile(&[4.0, 1.0, 3.0, 2.0], 0.5), Some(2.5));
        close(
            quantile(&[1.0, 2.0, 3.0, 4.0, 10.0], 0.9).unwrap(),
            7.6,
            1e-12,
        );
    }
}
