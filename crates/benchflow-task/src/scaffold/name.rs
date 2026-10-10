//! Task names for new tasks.
//!
//! A name is lowercase letters, digits, and hyphens, optionally after an
//! organization and a slash: `parse-logs` or `acme/parse-logs`. Each part
//! starts and ends with a letter or a digit. A name without an organization
//! is filed under [`LOCAL_ORG`], so every new task has the `org/name` form
//! that Harbor's `[task] name` and task registries require, and the task
//! exports to the split layout without a rename.

/// The organization a name without one is filed under.
pub const LOCAL_ORG: &str = "local";

/// The full `org/name` for `name`: `name` itself when it has an
/// organization, else `local/<name>`. `None` when `name` is not a valid task
/// name.
pub(crate) fn qualify(name: &str) -> Option<String> {
    match name.split_once('/') {
        Some((org, short)) => (is_part(org) && is_part(short)).then(|| name.to_string()),
        None => is_part(name).then(|| format!("{LOCAL_ORG}/{name}")),
    }
}

/// One part of a name: lowercase letters, digits, and hyphens, starting and
/// ending with a letter or a digit.
fn is_part(s: &str) -> bool {
    let edge = |b: Option<u8>| b.is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
    edge(s.bytes().next())
        && edge(s.bytes().last())
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// A valid name close to an invalid one: lowercased, with every other
/// character turned into a hyphen and stray hyphens dropped
/// (`My_Task` becomes `my-task`). `None` when nothing usable is left.
pub(crate) fn suggest(name: &str) -> Option<String> {
    let lowered = name.trim().to_lowercase();
    let parts: Vec<String> = lowered
        .split('/')
        .map(|part| {
            let mapped: String = part
                .chars()
                .map(|c| {
                    if c.is_ascii_lowercase() || c.is_ascii_digit() {
                        c
                    } else {
                        '-'
                    }
                })
                .collect();
            mapped
                .split('-')
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join("-")
        })
        .filter(|part| !part.is_empty())
        .collect();
    let fixed = match parts.as_slice() {
        [] => return None,
        [only] => only.clone(),
        [org, rest @ ..] => format!("{org}/{}", rest.join("-")),
    };
    (fixed != name && qualify(&fixed).is_some()).then_some(fixed)
}

/// A title for people, made from the name after its organization: hyphens
/// become spaces and the first letter is capitalized (`acme/parse-logs`
/// becomes `Parse logs`).
pub(crate) fn title(name: &str) -> String {
    let short = name.rsplit('/').next().unwrap_or(name);
    let words: Vec<&str> = short.split('-').filter(|w| !w.is_empty()).collect();
    let mut out = words.join(" ");
    if let Some(first) = out.get(..1) {
        let upper = first.to_ascii_uppercase();
        out.replace_range(..1, &upper);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_names_go_under_local() {
        assert_eq!(qualify("parse-logs").as_deref(), Some("local/parse-logs"));
        assert_eq!(qualify("2048").as_deref(), Some("local/2048"));
        assert_eq!(qualify("a").as_deref(), Some("local/a"));
    }

    #[test]
    fn org_names_are_kept() {
        assert_eq!(
            qualify("acme/parse-logs").as_deref(),
            Some("acme/parse-logs")
        );
        assert_eq!(qualify("a-1/b--2").as_deref(), Some("a-1/b--2"));
    }

    #[test]
    fn invalid_names_are_refused() {
        for bad in [
            "",
            "Parse-Logs",
            "parse_logs",
            "parse logs",
            "-parse",
            "parse-",
            "acme/",
            "/parse",
            "a/b/c",
            "parse.logs",
            "café",
            " parse",
        ] {
            assert_eq!(qualify(bad), None, "{bad:?} should be refused");
        }
    }

    #[test]
    fn suggestions_are_valid_and_close() {
        assert_eq!(suggest("My_Task").as_deref(), Some("my-task"));
        assert_eq!(
            suggest("Acme/Parse Logs").as_deref(),
            Some("acme/parse-logs")
        );
        assert_eq!(suggest("-parse-").as_deref(), Some("parse"));
        assert_eq!(suggest("a/b/c").as_deref(), Some("a/b-c"));
        assert_eq!(suggest("acme/").as_deref(), Some("acme"));
        assert_eq!(suggest("___"), None);
        assert_eq!(suggest(""), None);
        assert_eq!(
            suggest("parse-logs"),
            None,
            "a valid name needs no suggestion"
        );
    }

    #[test]
    fn titles_come_from_the_short_name() {
        assert_eq!(title("local/parse-logs"), "Parse logs");
        assert_eq!(title("acme/hello"), "Hello");
        assert_eq!(title("acme/2048-game"), "2048 game");
        assert_eq!(title("acme/a--b"), "A b");
    }
}
