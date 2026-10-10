//! Host lists in network settings: `allowed_hosts` (Harbor's rules, which
//! BenchFlow 0.8 shares), and 0.8's `blocked_hosts` and `blocked_urls`.

use std::net::IpAddr;
use std::sync::LazyLock;

use regex::Regex;

static LABEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$").unwrap());

const NOT_A_HOST: &str = "allowed_hosts entries must be hostnames, IP addresses, or IP CIDR ranges, not URLs, ports, or paths";

/// Check and normalize `allowed_hosts` as Harbor does: an exact hostname, a
/// leading-wildcard hostname (`*.example.com`), an IP address, or a strict
/// CIDR range (`10.0.0.0/8`, not `10.0.0.1/8`). Entries are trimmed and
/// lowercased, and a trailing dot is dropped.
pub(crate) fn allowed_hosts(hosts: &[String]) -> Result<Vec<String>, String> {
    hosts.iter().map(|h| allowed_host(h)).collect()
}

fn allowed_host(raw: &str) -> Result<String, String> {
    let host = raw.trim().to_lowercase();
    let host = host.trim_end_matches('.').to_string();
    if host.is_empty() {
        return Err("allowed_hosts entries must be non-empty hostnames".into());
    }
    if host.contains('/') {
        return cidr(&host).ok_or_else(|| NOT_A_HOST.to_string());
    }
    if host.contains(':') {
        return match host.parse::<IpAddr>() {
            Ok(IpAddr::V6(a)) if !host.contains('%') => Ok(a.to_string()),
            _ => Err(NOT_A_HOST.into()),
        };
    }
    if host.contains('[') || host.contains(']') {
        return Err(NOT_A_HOST.into());
    }
    let name = if let Some(rest) = host.strip_prefix("*.") {
        if rest.is_empty() {
            return Err("allowed_hosts wildcard entries must include a hostname suffix".into());
        }
        if rest.parse::<IpAddr>().is_ok() {
            return Err(
                "allowed_hosts wildcard entries must target hostnames, not IP address literals"
                    .into(),
            );
        }
        rest
    } else if host.contains('*') {
        return Err("allowed_hosts wildcard entries must use a leading '*.' prefix".into());
    } else {
        &host
    };
    if !name.split('.').all(|l| LABEL.is_match(l)) {
        return Err(
            "allowed_hosts entries must be IP addresses, IP CIDR ranges, or valid hostnames or leading wildcard host patterns containing only letters, digits, hyphens, and dots"
                .into(),
        );
    }
    Ok(host)
}

/// A strict CIDR range (no host bits set), written back in canonical form.
fn cidr(text: &str) -> Option<String> {
    let (addr, prefix) = text.split_once('/')?;
    let addr: IpAddr = addr.parse().ok()?;
    if prefix.is_empty() || !prefix.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let prefix: u32 = prefix.parse().ok()?;
    let ok = match addr {
        IpAddr::V4(a) => prefix <= 32 && (prefix == 32 || u32::from(a) & (u32::MAX >> prefix) == 0),
        IpAddr::V6(a) => {
            prefix <= 128 && (prefix == 128 || u128::from(a) & (u128::MAX >> prefix) == 0)
        }
    };
    ok.then(|| format!("{addr}/{prefix}"))
}

/// BenchFlow 0.8's hostname rule for `blocked_hosts` and URL hosts: no
/// scheme, port, or path, and valid labels.
fn hostname(raw: &str, field: &str) -> Result<String, String> {
    let host = raw.trim().to_lowercase();
    let host = host.trim_end_matches('.').to_string();
    if host.is_empty() {
        return Err(format!("{field} entries must be non-empty hostnames"));
    }
    if host.contains("://") || host.contains('/') || host.contains(':') {
        return Err(format!(
            "{field} entries must be hostnames, not URLs, ports, or paths"
        ));
    }
    if !host.split('.').all(|l| LABEL.is_match(l)) {
        return Err(format!(
            "{field} entries must be valid hostnames containing only letters, digits, hyphens, and dots"
        ));
    }
    Ok(host)
}

/// Check `blocked_hosts` (BenchFlow 0.8): hostnames, never IP addresses.
pub(crate) fn blocked_hosts(hosts: &[String]) -> Result<Vec<String>, String> {
    hosts
        .iter()
        .map(|h| {
            let host = hostname(h, "blocked_hosts")?;
            if host.parse::<IpAddr>().is_ok() {
                return Err("blocked_hosts entries must name a hostname, not an IP".into());
            }
            Ok(host)
        })
        .collect()
}

/// Check `blocked_urls` (BenchFlow 0.8): http(s) URL prefixes whose host is
/// a hostname; the scheme defaults to https.
pub(crate) fn blocked_urls(urls: &[String]) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for raw in urls {
        let url = raw.trim();
        if url.is_empty() {
            return Err("blocked_urls entries must be non-empty URLs".into());
        }
        let url = if url.contains("://") {
            url.to_string()
        } else {
            format!("https://{url}")
        };
        let (scheme, rest) = url.split_once("://").unwrap_or(("", &url));
        let scheme = scheme.to_lowercase();
        if scheme != "http" && scheme != "https" {
            return Err("blocked_urls entries must use the http or https scheme".into());
        }
        let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let (netloc, tail) = rest.split_at(end);
        if netloc.contains('@') {
            return Err("blocked_urls entries must not contain userinfo".into());
        }
        if netloc.contains(':') {
            return Err("blocked_urls entries must not contain a port".into());
        }
        let host = hostname(netloc, "blocked_urls")?;
        if host.parse::<IpAddr>().is_ok() {
            return Err("blocked_urls entries must name a hostname, not an IP".into());
        }
        let path = tail.split(['?', '#']).next().unwrap_or("");
        let entry = format!("{scheme}://{host}{}", path.trim_end_matches('/'));
        if !out.contains(&entry) {
            out.push(entry);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn allowed_hosts_follow_harbor() {
        assert_eq!(
            allowed_hosts(&v(&["PyPI.org.", "*.github.com", "10.0.0.0/8", "::1"])),
            Ok(v(&["pypi.org", "*.github.com", "10.0.0.0/8", "::1"]))
        );
        assert!(allowed_hosts(&v(&["10.0.0.1/8"])).is_err(), "host bits set");
        assert!(allowed_hosts(&v(&["https://pypi.org"])).is_err());
        assert!(allowed_hosts(&v(&["pypi.org:443"])).is_err());
        assert!(allowed_hosts(&v(&["a*.b.com"])).is_err());
        assert!(allowed_hosts(&v(&["*.1.2.3.4"])).is_err());
        assert!(allowed_hosts(&v(&[""])).is_err());
    }

    #[test]
    fn blocked_lists_follow_benchflow() {
        assert_eq!(
            blocked_hosts(&v(&["Mirror.example.net"])),
            Ok(v(&["mirror.example.net"]))
        );
        assert!(blocked_hosts(&v(&["1.2.3.4"])).is_err());
        assert_eq!(
            blocked_urls(&v(&["github.com/org/repo/", "https://example.org/a?q=1"])),
            Ok(v(&["https://github.com/org/repo", "https://example.org/a"]))
        );
        assert!(blocked_urls(&v(&["ftp://x.org"])).is_err());
        assert!(blocked_urls(&v(&["https://x.org:8080/a"])).is_err());
    }
}
