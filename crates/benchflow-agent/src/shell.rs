//! Shell quoting that matches Python's `shlex.quote`.
//!
//! BenchFlow 0.8 built its install scripts with `shlex.quote`, and the
//! built-in agents' scripts are checked byte for byte against 0.8's. The
//! `shlex` crate quotes differently (it prefers other escapes), so this module
//! reimplements Python's rule exactly: a string made only of
//! `[A-Za-z0-9_@%+=:,./-]` is left alone, the empty string becomes `''`, and
//! anything else is wrapped in single quotes with each `'` written as
//! `'"'"'`.

/// Quote `s` for a POSIX shell exactly as Python's `shlex.quote` does.
pub fn quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    if s.chars().all(is_safe) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// Python's `shlex._find_unsafe` complement: `[\w@%+=:,./-]` with ASCII `\w`.
fn is_safe(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
}

#[cfg(test)]
mod tests {
    use super::quote;

    // Expected values from CPython 3.12's shlex.quote.
    #[test]
    fn matches_python_shlex_quote() {
        assert_eq!(quote(""), "''");
        assert_eq!(
            quote("/opt/benchflow/bin/claude"),
            "/opt/benchflow/bin/claude"
        );
        assert_eq!(quote("#!/bin/sh"), "'#!/bin/sh'");
        assert_eq!(quote("2.1.280 (Claude Code)"), "'2.1.280 (Claude Code)'");
        assert_eq!(quote("it's"), "'it'\"'\"'s'");
        assert_eq!(quote("a b"), "'a b'");
        assert_eq!(quote("x=1,y:2@h%+"), "x=1,y:2@h%+");
        assert_eq!(quote("café"), "'café'");
        assert_eq!(quote("$HOME"), "'$HOME'");
    }
}
