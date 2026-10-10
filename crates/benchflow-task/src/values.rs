//! The value kinds task.md config uses: durations, sizes, counts, networks.
//!
//! The grammars match task.md draft 2 (`docs/document.md`, "Values"):
//!
//! | Kind | Written as |
//! |---|---|
//! | Duration | `"90s"`, `"15m"`, `"1h30m"`, `"0.5s"` |
//! | Size | `"512 MB"`, `"4 GB"` (1 GB is 1024 MB, as in Docker) |
//! | Count | `400000`, `"400K"`, `"2M"` |
//! | Network | `"none"`, `"open"`, `["pypi.org", "*.github.com"]`, `{ block = [...] }` |

use std::fmt;
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};

static DURATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$").unwrap()
});
static SIZE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d+(?:\.\d+)?) ?(MB|GB|TB)$").unwrap());
static COUNT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\d+(?:\.\d+)?)([KM])$").unwrap());

/// Parse a task.md duration such as `"1h30m"`. `None` when it is not one.
pub fn parse_duration(text: &str) -> Option<Duration> {
    if !text.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    let caps = DURATION.captures(text)?;
    let mut secs = 0.0f64;
    for (i, mult) in [(1, 3600.0), (2, 60.0), (3, 1.0)] {
        if let Some(m) = caps.get(i) {
            secs += m.as_str().parse::<f64>().ok()? * mult;
        }
    }
    Some(Duration::from_secs_f64(secs))
}

/// Format seconds the way task.md writes them: `"1h30m"`, `"90s"`, `"2m"`.
pub fn format_duration(d: Duration) -> String {
    let total = d.as_secs_f64();
    if total < 60.0 || (total.fract() != 0.0 && total < 3600.0) {
        let s = if total.fract() == 0.0 {
            format!("{}", total as u64)
        } else {
            format!("{total}")
        };
        return format!("{s}s");
    }
    let secs = total.round() as u64;
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    let mut out = String::new();
    if h > 0 {
        out.push_str(&format!("{h}h"));
    }
    if m > 0 {
        out.push_str(&format!("{m}m"));
    }
    if s > 0 {
        out.push_str(&format!("{s}s"));
    }
    out
}

/// Parse a size such as `"4 GB"` into megabytes (1 GB = 1024 MB).
pub fn parse_size_mb(text: &str) -> Option<u64> {
    let caps = SIZE.captures(text)?;
    let n: f64 = caps[1].parse().ok()?;
    let mult = match &caps[2] {
        "MB" => 1.0,
        "GB" => 1024.0,
        "TB" => 1024.0 * 1024.0,
        _ => return None,
    };
    Some((n * mult).round() as u64)
}

/// Parse a size into bytes, for output caps (`max_bytes`): a whole number of
/// bytes, or a size string.
pub fn parse_bytes(value: &serde_json::Value) -> Option<u64> {
    match value {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => parse_size_mb(s).map(|mb| mb * 1024 * 1024),
        _ => None,
    }
}

/// Parse a count: a whole number, or `"400K"` / `"2M"` that comes to one.
pub fn parse_count(value: &serde_json::Value) -> Option<u64> {
    match value {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => {
            let caps = COUNT.captures(s)?;
            let n: f64 = caps[1].parse().ok()?;
            let total = n * if &caps[2] == "K" {
                1_000.0
            } else {
                1_000_000.0
            };
            (total.fract() == 0.0).then_some(total as u64)
        }
        _ => None,
    }
}

/// A network setting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case", tag = "mode", content = "hosts")]
pub enum Network {
    /// Everything reachable (task.md's default, as in Harbor).
    #[default]
    Open,
    /// Nothing reachable.
    None,
    /// Only these hosts: exact names, `*.example.com`, IPs, or CIDR ranges.
    Allow(Vec<String>),
    /// Everything except these hosts.
    Block(Vec<String>),
}

impl Network {
    /// Read a network value from config. `Err` names what was wrong.
    pub fn from_value(value: &serde_json::Value) -> Result<Self, String> {
        match value {
            serde_json::Value::String(s) if s == "none" => Ok(Self::None),
            serde_json::Value::String(s) if s == "open" => Ok(Self::Open),
            serde_json::Value::String(s) => Err(format!(
                "`{s}` is not a network: use \"none\", \"open\", a list of hosts, or {{ block = [...] }}"
            )),
            serde_json::Value::Array(items) => {
                let hosts: Option<Vec<String>> = items
                    .iter()
                    .map(|v| v.as_str().map(str::to_string))
                    .collect();
                match hosts {
                    Some(h) if !h.is_empty() => Ok(Self::Allow(h)),
                    Some(_) => Err(
                        "an allowed-hosts list must not be empty; use \"none\" for no network"
                            .into(),
                    ),
                    None => Err("an allowed-hosts list holds host names as strings".into()),
                }
            }
            serde_json::Value::Object(map) => {
                let Some(block) = map.get("block") else {
                    return Err("a network table holds one key, block".into());
                };
                if map.len() > 1 {
                    return Err("a network table holds one key, block".into());
                }
                let hosts: Option<Vec<String>> = block
                    .as_array()
                    .map(|a| a.iter().map(|v| v.as_str().map(str::to_string)).collect())
                    .unwrap_or(None);
                match hosts {
                    Some(h) if !h.is_empty() => Ok(Self::Block(h)),
                    _ => Err("block lists the hosts to block, as strings".into()),
                }
            }
            _ => {
                Err("a network is \"none\", \"open\", a list of hosts, or { block = [...] }".into())
            }
        }
    }

    /// Whether `self` reaches no more than `outer` does.
    pub fn within(&self, outer: &Network) -> bool {
        match (self, outer) {
            (_, Network::Open) => true,
            (Network::None, _) => true,
            (Network::Allow(a), Network::Allow(b)) => a.iter().all(|h| b.contains(h)),
            (Network::Allow(_), Network::Block(_)) => true,
            (Network::Block(a), Network::Block(b)) => b.iter().all(|h| a.contains(h)),
            _ => false,
        }
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Network::Open => write!(f, "open"),
            Network::None => write!(f, "none"),
            Network::Allow(h) => write!(f, "only {}", h.join(", ")),
            Network::Block(h) => write!(f, "open except {}", h.join(", ")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn durations() {
        assert_eq!(parse_duration("90s"), Some(Duration::from_secs(90)));
        assert_eq!(parse_duration("15m"), Some(Duration::from_secs(900)));
        assert_eq!(parse_duration("1h30m"), Some(Duration::from_secs(5400)));
        assert_eq!(parse_duration("0.5s"), Some(Duration::from_millis(500)));
        assert_eq!(parse_duration("1.5h"), Some(Duration::from_secs(5400)));
        assert_eq!(parse_duration(""), None);
        assert_eq!(parse_duration("m"), None);
        assert_eq!(parse_duration("10"), None);
        assert_eq!(parse_duration("10 m"), None);
        assert_eq!(parse_duration("5min"), None);
    }

    #[test]
    fn formats_durations() {
        assert_eq!(format_duration(Duration::from_secs(90)), "1m30s");
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(5400)), "1h30m");
        assert_eq!(format_duration(Duration::from_secs(120)), "2m");
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size_mb("512 MB"), Some(512));
        assert_eq!(parse_size_mb("4 GB"), Some(4096));
        assert_eq!(parse_size_mb("4GB"), Some(4096));
        assert_eq!(parse_size_mb("0.5 GB"), Some(512));
        assert_eq!(parse_size_mb("4 gb"), None);
        assert_eq!(parse_size_mb("4096"), None);
    }

    #[test]
    fn counts() {
        assert_eq!(parse_count(&json!(400000)), Some(400_000));
        assert_eq!(parse_count(&json!("400K")), Some(400_000));
        assert_eq!(parse_count(&json!("2M")), Some(2_000_000));
        assert_eq!(parse_count(&json!("1.5M")), Some(1_500_000));
        assert_eq!(parse_count(&json!("1.0005K")), None);
        assert_eq!(parse_count(&json!("2G")), None);
    }

    #[test]
    fn networks() {
        assert_eq!(Network::from_value(&json!("none")), Ok(Network::None));
        assert_eq!(Network::from_value(&json!("open")), Ok(Network::Open));
        assert_eq!(
            Network::from_value(&json!(["pypi.org"])),
            Ok(Network::Allow(vec!["pypi.org".into()]))
        );
        assert_eq!(
            Network::from_value(&json!({"block": ["x.com"]})),
            Ok(Network::Block(vec!["x.com".into()]))
        );
        assert!(Network::from_value(&json!("offline")).is_err());
        assert!(Network::from_value(&json!([])).is_err());
        assert!(Network::None.within(&Network::Open));
        assert!(!Network::Open.within(&Network::None));
    }
}
