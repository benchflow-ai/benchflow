//! Shell helpers shared by the backends and the run engine.

use std::collections::BTreeMap;

/// The `PATH` BenchFlow's own root commands run with: the standard system
/// folders only. The image's `PATH` may name folders the agent user can
/// write (a venv in the workdir, say), and a root command that looked a
/// program up there after the agent's turn would run the agent's program.
pub const SYSTEM_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

/// Quote one word for a POSIX shell.
pub fn quote(word: &str) -> String {
    if !word.is_empty()
        && word
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"@%+=:,./-_".contains(&b))
    {
        return word.to_string();
    }
    format!("'{}'", word.replace('\'', r"'\''"))
}

/// Wrap `command` so it runs under bash when the image has it, else sh.
///
/// The command travels as one quoted word, so nothing in it is interpreted by
/// the outer shell.
pub fn login_free_shell(command: &str) -> String {
    let q = quote(command);
    format!("if command -v bash >/dev/null 2>&1; then exec bash -c {q}; else exec sh -c {q}; fi")
}

/// Whether `key` can be set with `export KEY=...`: a letter or `_`, then
/// letters, digits and `_`.
pub fn is_env_name(key: &str) -> bool {
    let mut bytes = key.bytes();
    matches!(bytes.next(), Some(b) if b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// `export K=V; ...` prefix for a set of variables.
///
/// Names a shell cannot export (see [`is_env_name`]) are left out with a
/// warning: one such line would make the whole prefix fail, and the command
/// after it would never run.
pub fn exports(env: &BTreeMap<String, String>) -> String {
    let mut s = String::new();
    for (k, v) in env {
        if !is_env_name(k) {
            tracing::warn!(
                "leaving out environment variable {k:?}: a shell cannot export that name"
            );
            continue;
        }
        s.push_str("export ");
        s.push_str(k);
        s.push('=');
        s.push_str(&quote(v));
        s.push_str("; ");
    }
    s
}

/// Drop from root to `user` and run `command` with bash (sh when the image
/// has no bash), keeping the caller's environment except `HOME`, `USER` and
/// `LOGNAME`, which name the target user as `docker exec -u` does.
///
/// `user` is a name or a numeric uid. Uses `setpriv` (util-linux) when
/// present, else `su`. A name that does not exist fails the command with
/// exit code 126 and a message on stderr; it never runs as root.
pub fn as_user(user: &str, command: &str) -> String {
    let inner = quote(&login_free_shell(command));
    let u = quote(user);
    format!(
        "bf_user={u}; \
         if ! id \"$bf_user\" >/dev/null 2>&1; then \
         case \"$bf_user\" in *[!0-9]*) echo \"benchflow: no user named $bf_user in this sandbox\" >&2; exit 126;; esac; fi; \
         bf_entry=$(awk -F: -v u=\"$bf_user\" '$1 == u || $3 == u {{ print $1 \":\" $6; exit }}' /etc/passwd 2>/dev/null); \
         if [ -n \"$bf_entry\" ]; then export USER=\"${{bf_entry%%:*}}\" LOGNAME=\"${{bf_entry%%:*}}\" HOME=\"${{bf_entry#*:}}\"; \
         else export USER=\"$bf_user\" LOGNAME=\"$bf_user\" HOME=/; fi; \
         if command -v setpriv >/dev/null 2>&1 && setpriv --help 2>&1 | grep -q reuid; then \
         if [ -n \"$bf_entry\" ]; then exec setpriv --reuid=\"$(id -u \"$bf_user\")\" --regid=\"$(id -g \"$bf_user\")\" --init-groups -- sh -c {inner}; \
         else exec setpriv --reuid=\"$bf_user\" --regid=\"$bf_user\" --clear-groups -- sh -c {inner}; fi; \
         else exec su -s /bin/sh \"$USER\" -c {inner}; fi"
    )
}

/// A script that kills every process whose environment carries
/// [`EXEC_ID_ENV`](crate::EXEC_ID_ENV)`=<exec_id>`: a command and everything
/// it started, unless a process cleared its own environment. Run it as root.
///
/// It makes up to five passes over `/proc`, so children forked during one
/// pass are caught by the next, and it never kills itself.
pub fn kill_tagged(exec_id: &str) -> String {
    format!(
        "id={id}; me=$$; pass=0; \
         while [ $pass -lt 5 ]; do pass=$((pass + 1)); found=0; \
         for d in /proc/[0-9]*; do p=${{d#/proc/}}; [ \"$p\" = \"$me\" ] && continue; \
         if grep -qsF \"{env}=$id\" \"$d/environ\" 2>/dev/null; then kill -9 \"$p\" 2>/dev/null && found=1; fi; \
         done; [ $found = 0 ] && break; done; exit 0",
        id = quote(exec_id),
        env = crate::EXEC_ID_ENV
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_only_when_needed() {
        assert_eq!(quote("abc"), "abc");
        assert_eq!(quote("/app/x.txt"), "/app/x.txt");
        assert_eq!(quote("a b"), "'a b'");
        assert_eq!(quote("it's"), r"'it'\''s'");
        assert_eq!(quote(""), "''");
    }

    #[test]
    fn exports_are_quoted() {
        let mut env = BTreeMap::new();
        env.insert("A".into(), "x y".into());
        assert_eq!(exports(&env), "export A='x y'; ");
    }

    #[test]
    fn exports_leave_out_names_a_shell_cannot_set() {
        let mut env = BTreeMap::new();
        env.insert("OK_1".into(), "v".into());
        env.insert("a.b".into(), "v".into());
        env.insert("X;rm -rf /".into(), "v".into());
        env.insert("1BAD".into(), "v".into());
        assert_eq!(exports(&env), "export OK_1=v; ");
    }

    #[test]
    fn env_names() {
        assert!(is_env_name("_"));
        assert!(is_env_name("PATH"));
        assert!(is_env_name("a1_b"));
        assert!(!is_env_name(""));
        assert!(!is_env_name("1A"));
        assert!(!is_env_name("A-B"));
    }

    /// Run a script with the host's sh and return (exit code, stdout).
    fn sh(script: &str) -> (i32, String) {
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .output()
            .unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    }

    #[test]
    fn login_free_shell_runs_the_command_once_with_its_quoting_intact() {
        let (code, out) = sh(&login_free_shell(r#"printf '%s|' "a b" 'c'"d"; exit 3"#));
        assert_eq!(code, 3);
        assert_eq!(out, "a b|cd|");
    }

    #[test]
    fn exports_survive_a_round_trip_through_sh() {
        let mut env = BTreeMap::new();
        env.insert("V".into(), "it's $HOME `x` \"q\"\nline2".into());
        let (code, out) = sh(&format!("{}printf %s \"$V\"", exports(&env)));
        assert_eq!(code, 0);
        assert_eq!(out, "it's $HOME `x` \"q\"\nline2");
    }

    #[test]
    fn as_user_refuses_an_unknown_name() {
        // Runs on the host as an unprivileged user: the unknown-name check
        // comes before any privilege change, so this is safe to run anywhere.
        let (code, _) = sh(&as_user("no-such-user-bf", "echo should-not-run"));
        assert_eq!(code, 126);
    }
}
