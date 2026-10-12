//! Shell helpers shared by the backends and the run engine.

use std::collections::BTreeMap;

/// The `PATH` BenchFlow's own root commands run with: the standard system
/// folders only. The image's `PATH` may name folders the agent user can
/// write (a venv in the workdir, say), and a root command that looked a
/// program up there after the agent's turn would run the agent's program.
pub const SYSTEM_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

/// Loader variables that name a file or folder. glibc and the shells treat
/// an empty value as "none", so setting one empty neutralizes the image's
/// value without the loader acting on it. These are the only loader
/// variables BenchFlow ever passes as empty (in `docker exec -e` and the
/// Daytona image's `ENV`); the flags below it never sets, because glibc
/// acts on their mere presence (`LD_SHOW_AUXV` prints the auxiliary vector
/// from every program, empty value and all).
pub const FILE_LOADER_VARS: &[&str] = &[
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_DEBUG_OUTPUT",
    "LD_PROFILE",
    "LD_PROFILE_OUTPUT",
    "GCONV_PATH",
    "LOCPATH",
    "NLSPATH",
    "HOSTALIASES",
    "MALLOC_TRACE",
];

/// The shells' start-up files. Like [`FILE_LOADER_VARS`] (empty means
/// none, so safe to pass empty), but a command's own shell should read
/// them — conda and nvm set tools up through `BASH_ENV` — so they are put
/// back just before the command's shell starts, not after it
/// ([`command_shell`]). BenchFlow's own root programs still run without
/// them.
pub const STARTUP_VARS: &[&str] = &["BASH_ENV", "ENV"];

/// Loader flags: glibc acts on each of these when it is merely present,
/// whatever the value (`LD_SHOW_AUXV` prints the auxiliary vector,
/// `LD_DYNAMIC_WEAK` changes symbol binding), and none names a file an
/// image's value could point at the agent's code. BenchFlow never sets
/// them — not even empty — and drops them only with `unset` inside the
/// command's shell.
pub const FLAG_LOADER_VARS: &[&str] = &[
    "LD_DEBUG",
    "LD_ORIGIN_PATH",
    "LD_DYNAMIC_WEAK",
    "LD_BIND_NOT",
    "LD_SHOW_AUXV",
    "LD_USE_LOAD_BIAS",
    "LD_HWCAP_MASK",
    "LD_ASSUME_KERNEL",
    "GLIBC_TUNABLES",
    "RES_OPTIONS",
];

/// Every variable that changes how programs load or start. BenchFlow's own
/// root programs never run with any of them ([`crate::Exec::system`]), and
/// the backends' plumbing moves them aside around the command
/// ([`stash_loader_vars`]).
pub const LOADER_VARS: &[&str] = &[
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_DEBUG",
    "LD_DEBUG_OUTPUT",
    "LD_PROFILE",
    "LD_PROFILE_OUTPUT",
    "LD_ORIGIN_PATH",
    "LD_DYNAMIC_WEAK",
    "LD_BIND_NOT",
    "LD_SHOW_AUXV",
    "LD_USE_LOAD_BIAS",
    "LD_HWCAP_MASK",
    "LD_ASSUME_KERNEL",
    "GCONV_PATH",
    "GLIBC_TUNABLES",
    "MALLOC_TRACE",
    "LOCPATH",
    "NLSPATH",
    "HOSTALIASES",
    "RES_OPTIONS",
    "BASH_ENV",
    "ENV",
];

/// The variables BenchFlow may pass as an empty string (a file is named, so
/// empty means none): [`FILE_LOADER_VARS`] and [`STARTUP_VARS`]. The flags
/// are never passed.
pub fn empty_safe_vars() -> impl Iterator<Item = &'static str> {
    FILE_LOADER_VARS
        .iter()
        .chain(STARTUP_VARS)
        .chain(SYSTEM_EXTRA_VARS)
        .copied()
}

/// Other variables that tell a program BenchFlow runs as root to load code
/// of the environment's choosing: node's `NODE_OPTIONS`/`NODE_PATH`, GNU
/// tar's `TAR_OPTIONS`, apt's `APT_CONFIG`, Perl's and Python's. They name
/// files or options, so an empty value is "none". [`crate::Exec::system`]
/// drops them too, so the forwarder's `node`, every root `tar`, and the
/// tmux install do not run the image's or the task's code.
pub const SYSTEM_EXTRA_VARS: &[&str] = &[
    "NODE_OPTIONS",
    "NODE_PATH",
    "TAR_OPTIONS",
    "APT_CONFIG",
    "PERL5OPT",
    "PERL5LIB",
    "PERLLIB",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
];

/// Shell source that moves each of `names` that is set to a **non-empty**
/// value into `BF_KEEP_<name>` (exported, so it crosses `exec` and the
/// switch to another user) and unsets it, so the programs started after it
/// run without them. An empty value is dropped, not saved: the Daytona
/// image sets [`FILE_LOADER_VARS`] and [`STARTUP_VARS`] empty, and those
/// empties must not come back to the command (and `LD_SHOW_AUXV` must never
/// be present at all). The real values come from the image record and the
/// env file instead.
fn stash_vars(names: &[&str]) -> String {
    // A variable that is set at all (empty included) replaces any older
    // copy: a task's `NAME=''` in the env file must not let the container's
    // value (kept by the session line's stash) come back.
    format!(
        "for bf_v in {}; do eval \"if [ -n \\\"\\${{$bf_v+x}}\\\" ]; then unset BF_KEEP_$bf_v; if [ -n \\\"\\${{$bf_v}}\\\" ]; then BF_KEEP_$bf_v=\\\"\\${{$bf_v}}\\\"; export BF_KEEP_$bf_v; fi; fi; unset $bf_v\"; done; ",
        names.join(" ")
    )
}

/// Stash every loader and start-up variable (see [`stash_vars`]).
pub(crate) fn stash_loader_vars() -> String {
    stash_vars(LOADER_VARS)
}

/// Shell source that puts `names` back from `BF_KEEP_<name>`.
fn restore_vars(names: &[&str]) -> String {
    format!(
        "for bf_v in {}; do eval \"if [ -n \\\"\\${{BF_KEEP_$bf_v+x}}\\\" ]; then $bf_v=\\\"\\${{BF_KEEP_$bf_v}}\\\"; export $bf_v; unset BF_KEEP_$bf_v; fi\"; done; ",
        names.join(" ")
    )
}

/// Put the loader variables (not the start-up files) back, those not in
/// `unset`, inside the command's own shell: they take effect when the
/// command runs a program, so restoring them after the shell has started is
/// enough, and keeps them off the shells and programs on the way.
fn restore_loader_vars(unset: &[String]) -> String {
    let names: Vec<&str> = FILE_LOADER_VARS
        .iter()
        .chain(FLAG_LOADER_VARS)
        .copied()
        .filter(|v| !unset.iter().any(|u| u == v))
        .collect();
    if names.is_empty() {
        String::new()
    } else {
        restore_vars(&names)
    }
}

/// Put the start-up files ([`STARTUP_VARS`]) back, those not in `unset`,
/// and do what a non-interactive `bash` does with `BASH_ENV` as it starts:
/// source it. This runs inside the command's shell after
/// [`restore_loader_vars`], so the start-up file sees the image's loader
/// variables and its own changes to them (a CUDA image's `LD_LIBRARY_PATH`
/// that a conda or ROS setup script prepends to) win, as on Docker and with
/// a plain `bash -c`. It sources only a value that came back from a stash —
/// on Docker the shell already sourced it at start — and only in a bash
/// that is not in POSIX mode, as bash itself does; a leading `~/` is
/// expanded as bash does. A non-interactive shell never reads `ENV`, so
/// `ENV` is only put back.
fn restore_startup_vars(unset: &[String]) -> String {
    let keep = |v: &str| !unset.iter().any(|u| u == v);
    let mut out = String::new();
    if keep("ENV") {
        out.push_str(&restore_vars(&["ENV"]));
    }
    if keep("BASH_ENV") {
        out.push_str(
            "if [ -n \"${BF_KEEP_BASH_ENV+x}\" ]; then BASH_ENV=\"$BF_KEEP_BASH_ENV\"; export BASH_ENV; unset BF_KEEP_BASH_ENV; \
             if [ -n \"${BASH_VERSION-}\" ] && ! shopt -oq posix 2>/dev/null && [ -n \"$BASH_ENV\" ]; then \
             bf_be=$BASH_ENV; case $bf_be in \"~/\"*) bf_be=\"$HOME/${bf_be#\\~/}\";; esac; \
             if [ -r \"$bf_be\" ]; then . \"$bf_be\"; fi; unset bf_be; fi; fi; ",
        );
    }
    out
}

/// Drop every `BF_KEEP_*` copy left over (of variables the command runs
/// without), so the command's environment carries none of them.
fn clear_kept() -> String {
    let names: Vec<String> = LOADER_VARS.iter().map(|v| format!("BF_KEEP_{v}")).collect();
    format!("unset {}; ", names.join(" "))
}

/// `unset <names>; ` for the names a shell can unset (see [`is_env_name`]);
/// empty when there are none.
pub(crate) fn unset_vars(names: &[String]) -> String {
    let names: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| is_env_name(n))
        .collect();
    if names.is_empty() {
        String::new()
    } else {
        format!("unset {}; ", names.join(" "))
    }
}

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
/// the outer shell. For image builds, where bash is looked up through the
/// build's `PATH`; commands in a running sandbox use [`command_shell`].
pub fn login_free_shell(command: &str) -> String {
    let q = quote(command);
    format!("if command -v bash >/dev/null 2>&1; then exec bash -c {q}; else exec sh -c {q}; fi")
}

/// The command that runs a script file a task or user supplies (a reference
/// solution, a stage's `setup.sh`, a control, a `script:` agent): with bash,
/// as BenchFlow 0.8 runs it, or with sh on an image that has no bash (Alpine,
/// busybox), where `bash <file>` would exit 127 before the script ran.
pub fn script_file(path: &str) -> String {
    let q = quote(path);
    format!("if command -v bash >/dev/null 2>&1; then bash {q}; else sh {q}; fi")
}

/// Shell source that starts a script of BenchFlow's own plumbing: keep the
/// `PATH` the shell was given in `bf_path` (an unset one becomes
/// [`SYSTEM_PATH`], the shells' own default), and from here on look
/// programs up in `sys` only. The script hands `bf_path` back with
/// `PATH=$bf_path` just before it `exec`s, by absolute path, the program it
/// runs the command with.
///
/// Plumbing scripts often run as root before the command drops to the
/// agent's user, and the image's `PATH` may start with a folder the agent
/// can write (a venv in the workdir, say): a lookup there would run the
/// agent's program as root.
pub(crate) fn keep_path(sys: &str) -> String {
    let q = quote(sys);
    format!("bf_path=${{PATH-{q}}}; PATH={q}; export PATH; ")
}

/// `script` with `PATH` set to [`SYSTEM_PATH`] first, for a script of
/// BenchFlow's own that runs as root and needs no `PATH` of the image's.
pub fn with_system_path(script: &str) -> String {
    format!("PATH={SYSTEM_PATH}; export PATH; {script}")
}

/// Run `command` under bash when the image has it, else sh, for a command
/// in a running sandbox (as any user), without the variables in `unset`.
///
/// Bash is looked up in the system folders only and `/bin/sh` is named by
/// its path, so a program the agent placed on the image's `PATH` never
/// stands in for the shell, even in a root command. The command itself runs
/// with the `PATH` this shell was given, the loader variables a backend's
/// plumbing moved aside ([`stash_loader_vars`]) back in place, and with
/// `$0` set to `bash` or `sh` as a plain `bash -c` would have it.
pub fn command_shell(command: &str, unset: &[String]) -> String {
    command_shell_in(SYSTEM_PATH, command, unset)
}

/// [`command_shell`], looking bash up in `sys`.
pub(crate) fn command_shell_in(sys: &str, command: &str, unset: &[String]) -> String {
    // The command is held in one variable and named once, so nothing that
    // wraps this script (Daytona nests it in a session line) copies the
    // command text: a long command used to be pasted a dozen times, three
    // quoting levels deep, and overflowed the argument.
    let c = quote(&command_body(command, unset));
    format!(
        "bf_cmd={c}; {keep}\
         bf_sh=$(command -v bash 2>/dev/null); case \"$bf_sh\" in /*) ;; *) bf_sh=/bin/sh;; esac; \
         PATH=$bf_path; \
         exec \"$bf_sh\" -c \"$bf_cmd\" \"${{bf_sh##*/}}\"",
        keep = keep_path(sys),
    )
}

/// The `-c` body for the command's shell: put back the loader variables a
/// backend stashed ([`stash_loader_vars`]), drop the ones in `unset`, put
/// back the start-up files and source `BASH_ENV` as bash would
/// ([`restore_startup_vars`]), then the command. This runs in the command's
/// own shell, after any switch to a user, so the programs the switch runs
/// (`setpriv`, `su`, ...) see none of them; and the command text appears
/// once.
fn command_body(command: &str, unset: &[String]) -> String {
    format!(
        "{loader}{unset}{startup}{clear}{command}",
        loader = restore_loader_vars(unset),
        unset = unset_vars(unset),
        startup = restore_startup_vars(unset),
        clear = clear_kept(),
    )
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
///
/// Everything this script runs as root (`id`, `awk`, `grep`, `setpriv` or
/// `su`, and the shell after them) is found in the system folders or named
/// by its path; the command gets the `PATH` this script was given (see
/// [`keep_path`]), less the variables in `unset`. Run it with the loader
/// variables stashed ([`stash_loader_vars`]): the command gets them back.
pub fn as_user(user: &str, command: &str, unset: &[String]) -> String {
    as_user_in(SYSTEM_PATH, user, command, unset)
}

/// [`as_user`], looking programs up in `sys`.
pub(crate) fn as_user_in(sys: &str, user: &str, command: &str, unset: &[String]) -> String {
    // As in [`command_shell_in`], the command is named once (`bf_cmd`): the
    // privilege-drop branches reference it, so neither they nor the session
    // line that wraps this copy the command text.
    let c = quote(&command_body(command, unset));
    let u = quote(user);
    format!(
        "bf_cmd={c}; {keep}bf_user={u}; \
         if ! id \"$bf_user\" >/dev/null 2>&1; then \
         case \"$bf_user\" in *[!0-9]*) echo \"benchflow: no user named $bf_user in this sandbox\" >&2; exit 126;; esac; fi; \
         bf_entry=$(awk -F: -v u=\"$bf_user\" '$1 == u || $3 == u {{ print $1 \":\" $6; exit }}' /etc/passwd 2>/dev/null); \
         if [ -n \"$bf_entry\" ]; then export USER=\"${{bf_entry%%:*}}\" LOGNAME=\"${{bf_entry%%:*}}\" HOME=\"${{bf_entry#*:}}\"; \
         else export USER=\"$bf_user\" LOGNAME=\"$bf_user\" HOME=/; fi; \
         bf_sh=$(command -v bash 2>/dev/null); case \"$bf_sh\" in /*) ;; *) bf_sh=/bin/sh;; esac; \
         bf_setpriv=$(command -v setpriv 2>/dev/null); \
         case \"$bf_setpriv\" in /*) \"$bf_setpriv\" --help 2>&1 | grep -q reuid || bf_setpriv=;; *) bf_setpriv=;; esac; \
         if [ -n \"$bf_setpriv\" ]; then \
         if [ -n \"$bf_entry\" ]; then bf_uid=$(id -u \"$bf_user\"); bf_gid=$(id -g \"$bf_user\"); PATH=$bf_path; \
         exec \"$bf_setpriv\" --reuid=\"$bf_uid\" --regid=\"$bf_gid\" --init-groups -- \"$bf_sh\" -c \"$bf_cmd\" \"${{bf_sh##*/}}\"; \
         else PATH=$bf_path; \
         exec \"$bf_setpriv\" --reuid=\"$bf_user\" --regid=\"$bf_user\" --clear-groups -- \"$bf_sh\" -c \"$bf_cmd\" \"${{bf_sh##*/}}\"; fi; fi; \
         bf_su=$(command -v su 2>/dev/null); \
         case \"$bf_su\" in /*) ;; *) echo \"benchflow: this sandbox has neither setpriv nor su to run a command as $bf_user\" >&2; exit 126;; esac; \
         PATH=$bf_path; exec \"$bf_su\" -s \"$bf_sh\" \"$USER\" -c \"$bf_cmd\"",
        keep = keep_path(sys),
    )
}

/// A script that kills every process whose environment carries
/// [`EXEC_ID_ENV`](crate::EXEC_ID_ENV)`=<exec_id>`: a command and everything
/// it started, unless a process cleared its own environment. Run it as root;
/// it finds `grep` in the system folders only, and runs it without the
/// loader variables (start the shell itself without them too).
///
/// It makes up to five passes over `/proc`, so children forked during one
/// pass are caught by the next, and it never kills itself.
pub fn kill_tagged(exec_id: &str) -> String {
    format!(
        "PATH={SYSTEM_PATH}; export PATH; unset {loader}; id={id}; me=$$; pass=0; \
         while [ $pass -lt 5 ]; do pass=$((pass + 1)); found=0; \
         for d in /proc/[0-9]*; do p=${{d#/proc/}}; [ \"$p\" = \"$me\" ] && continue; \
         if grep -qsF \"{env}=$id\" \"$d/environ\" 2>/dev/null; then kill -9 \"$p\" 2>/dev/null && found=1; fi; \
         done; [ $found = 0 ] && break; done; exit 0",
        id = quote(exec_id),
        env = crate::EXEC_ID_ENV,
        loader = LOADER_VARS.join(" ")
    )
}

/// Marker programs for tests: a folder of programs named like the ones
/// BenchFlow's plumbing runs, each of which only appends its name to a log
/// and then runs the real program from the system folders. Put the folder
/// first on a command's `PATH`; any name in [`Markers::ran`] was looked up
/// through that `PATH`.
#[cfg(test)]
pub(crate) mod markers {
    use std::path::{Path, PathBuf};

    /// The names BenchFlow's plumbing runs, and `bash` and `sleep`.
    pub(crate) const NAMES: &[&str] = &[
        "sh", "bash", "setsid", "mkdir", "cat", "rm", "id", "awk", "grep", "setpriv", "su", "tar",
        "sleep",
    ];

    pub(crate) struct Markers {
        dir: tempfile::TempDir,
    }

    /// The first `name` in the system folders.
    pub(crate) fn real(name: &str) -> Option<PathBuf> {
        super::SYSTEM_PATH
            .split(':')
            .map(|d| Path::new(d).join(name))
            .find(|p| p.is_file())
    }

    impl Markers {
        pub(crate) fn new() -> Self {
            use std::os::unix::fs::PermissionsExt as _;
            let dir = tempfile::tempdir().unwrap();
            // Writable by anyone, so a marker that runs as another user
            // (a test run as root that drops privileges) still records.
            std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
            let bin = dir.path().join("bin");
            std::fs::create_dir(&bin).unwrap();
            let log = dir.path().join("ran");
            for name in NAMES {
                let then = match real(name) {
                    Some(p) => format!("exec {} \"$@\"", super::quote(&p.to_string_lossy())),
                    None => "exit 127".to_string(),
                };
                let script = format!(
                    "#!/bin/sh\necho {name} >> {log}\n{then}\n",
                    log = super::quote(&log.to_string_lossy())
                );
                let path = bin.join(name);
                std::fs::write(&path, script).unwrap();
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            Self { dir }
        }

        /// The marker folder.
        pub(crate) fn bin(&self) -> String {
            self.dir.path().join("bin").to_string_lossy().into_owned()
        }

        /// A `PATH` with the marker folder first, then the system folders.
        pub(crate) fn path(&self) -> String {
            format!("{}:{}", self.bin(), super::SYSTEM_PATH)
        }

        /// The markers that ran, in order.
        pub(crate) fn ran(&self) -> Vec<String> {
            std::fs::read_to_string(self.dir.path().join("ran"))
                .unwrap_or_default()
                .lines()
                .map(str::to_string)
                .collect()
        }

        /// Run `script` with `/bin/sh` and the markers first on `PATH`:
        /// (exit code, stdout, stderr).
        pub(crate) fn sh(&self, script: &str) -> (i32, String, String) {
            self.sh_env(script, &[])
        }

        /// [`Markers::sh`], with `env` set too.
        pub(crate) fn sh_env(&self, script: &str, env: &[(&str, &str)]) -> (i32, String, String) {
            let out = std::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(script)
                .env("PATH", self.path())
                .envs(env.iter().copied())
                .stdin(std::process::Stdio::null())
                .output()
                .unwrap();
            (
                out.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&out.stdout).into_owned(),
                String::from_utf8_lossy(&out.stderr).into_owned(),
            )
        }
    }
}

/// A stand-in for the system folders, for tests: each program records its
/// name and which loader variables it started with, then runs the real one.
/// `setpriv` and `setsid` are fakes that run what follows without switching
/// users or sessions (`setpriv` also records its options in `args`).
#[cfg(test)]
pub(crate) mod recorders {

    /// The loader variables the tests set, and their values: harmless ones
    /// (a blank `LD_PRELOAD` preloads nothing) that a command can print.
    pub(crate) fn loader_env(read_log: &str) -> Vec<(&'static str, String)> {
        vec![
            ("LD_PRELOAD", " ".to_string()),
            ("LD_LIBRARY_PATH", "/nonexistent-bf-lib".to_string()),
            ("BASH_ENV", read_log.to_string()),
            ("ENV", read_log.to_string()),
        ]
    }

    /// A command that prints the loader variables the tests set.
    pub(crate) const PRINT_LOADER_VARS: &str = r#"printf '[%s]' "${LD_PRELOAD-unset}" "${LD_LIBRARY_PATH-unset}" "${BASH_ENV-unset}" "${ENV-unset}""#;

    pub(crate) struct Recorders {
        dir: tempfile::TempDir,
    }

    const NAMES: &[&str] = &[
        "id", "awk", "grep", "mkdir", "cat", "rm", "bash", "tar", "setsid", "setpriv",
    ];

    impl Recorders {
        pub(crate) fn new() -> Self {
            use std::os::unix::fs::PermissionsExt as _;
            let dir = tempfile::tempdir().unwrap();
            std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
            let bin = dir.path().join("sys");
            std::fs::create_dir(&bin).unwrap();
            let log = super::quote(&dir.path().join("env").to_string_lossy());
            let args = super::quote(&dir.path().join("args").to_string_lossy());
            // The file BASH_ENV and ENV name: it records each read.
            let read = dir.path().join("read.sh");
            std::fs::write(
                &read,
                format!(
                    "echo read >> {}\n",
                    super::quote(&dir.path().join("reads").to_string_lossy())
                ),
            )
            .unwrap();
            for name in NAMES {
                let then = match *name {
                    "setsid" => "exec \"$@\"".to_string(),
                    "setpriv" => format!(
                        "case \"$1\" in --help) echo --reuid; exit 0;; esac\n\
                         for a; do shift; [ \"$a\" = -- ] && break; printf '%s\\n' \"$a\" >> {args}; done\n\
                         exec \"$@\""
                    ),
                    _ => format!(
                        "exec {} \"$@\"",
                        super::quote(&super::markers::real(name).unwrap().to_string_lossy())
                    ),
                };
                let script = format!(
                    "#!/bin/sh\nbf_r={name}; for bf_v in LD_PRELOAD LD_LIBRARY_PATH BASH_ENV ENV; do \
                     eval \"[ -n \\\"\\${{$bf_v+x}}\\\" ]\" && bf_r=\"$bf_r $bf_v\"; done; echo \"$bf_r\" >> {log}\n{then}\n"
                );
                let path = bin.join(name);
                std::fs::write(&path, script).unwrap();
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            Self { dir }
        }

        /// The stand-in folder, as a `PATH`.
        pub(crate) fn sys(&self) -> String {
            self.dir.path().join("sys").to_string_lossy().into_owned()
        }

        /// The file `BASH_ENV` and `ENV` name in [`loader_env`].
        pub(crate) fn read_file(&self) -> String {
            self.dir
                .path()
                .join("read.sh")
                .to_string_lossy()
                .into_owned()
        }

        fn lines(&self, file: &str) -> Vec<String> {
            std::fs::read_to_string(self.dir.path().join(file))
                .unwrap_or_default()
                .lines()
                .map(str::to_string)
                .collect()
        }

        /// Each program that ran, with the loader variables it saw:
        /// `"id"`, `"bash LD_PRELOAD LD_LIBRARY_PATH BASH_ENV ENV"`.
        pub(crate) fn ran(&self) -> Vec<String> {
            self.lines("env")
        }

        /// How often the `BASH_ENV`/`ENV` file was read.
        pub(crate) fn reads(&self) -> usize {
            self.lines("reads").len()
        }

        /// The options the fake `setpriv` got.
        pub(crate) fn setpriv_args(&self) -> Vec<String> {
            self.lines("args")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shell whose PATH has `sh` and nothing else, as on an image without bash.
    #[cfg(unix)]
    fn run_without_bash(command: &str) -> std::process::Output {
        let bin = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink("/bin/sh", bin.path().join("sh")).unwrap();
        std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(command)
            .env_clear()
            .env("PATH", bin.path())
            .output()
            .unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn a_script_file_runs_on_an_image_without_bash() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("solve it.sh");
        std::fs::write(&script, "#!/bin/sh\necho ran \"$0\"\nexit 3\n").unwrap();
        let path = script.to_str().unwrap();
        let old = run_without_bash(&format!("bash {}", quote(path)));
        assert_eq!(old.status.code(), Some(127));
        let out = run_without_bash(&script_file(path));
        assert_eq!(out.status.code(), Some(3), "{out:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("ran {path}\n")
        );
        assert_eq!(
            script_file("/oracle/solve.sh"),
            "if command -v bash >/dev/null 2>&1; then bash /oracle/solve.sh; else sh /oracle/solve.sh; fi"
        );
    }

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
        let (code, _) = sh(&as_user("no-such-user-bf", "echo should-not-run", &[]));
        assert_eq!(code, 126);
    }

    /// The control for the tests below: a plain lookup through a `PATH`
    /// with the markers first runs the markers.
    #[test]
    fn markers_record_a_lookup_through_the_path() {
        let m = markers::Markers::new();
        let dir = tempfile::tempdir().unwrap();
        let (code, _, err) = m.sh(&format!(
            "mkdir -p {d}/x && id >/dev/null && rm -r {d}/x",
            d = quote(&dir.path().to_string_lossy())
        ));
        assert_eq!(code, 0, "{err}");
        assert_eq!(m.ran(), ["mkdir", "id", "rm"]);
    }

    /// The shell for a command is found in the system folders, never
    /// through the `PATH` it was given, and the command gets that `PATH`.
    #[test]
    fn command_shell_finds_no_program_through_the_given_path() {
        let m = markers::Markers::new();
        let (code, out, err) = m.sh(&command_shell(
            r#"printf '%s|%s' "$PATH" "$0"; exit 4"#,
            &[],
        ));
        assert_eq!(code, 4, "{err}");
        let shell = if markers::real("bash").is_some() {
            "bash"
        } else {
            "sh"
        };
        assert_eq!(out, format!("{}|{shell}", m.path()));
        assert!(m.ran().is_empty(), "{:?}", m.ran());
        // A shell given no PATH at all hands on the system folders.
        let out = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(format!(
                "unset PATH; {}",
                command_shell(r#"printf %s "$PATH""#, &[])
            ))
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout), SYSTEM_PATH);
    }

    /// `as_user` runs `id`, `awk`, `grep` and `setpriv` (or `su`) as root;
    /// none of them may come from the `PATH` it was given. Uses a uid no
    /// one has, so on this host the switch fails (or, as root, runs
    /// `true` as that uid); what matters is which programs were looked up.
    #[test]
    fn as_user_finds_no_program_through_the_given_path() {
        let m = markers::Markers::new();
        let _ = m.sh(&as_user("4242424", "true", &[]));
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    /// A `setpriv` stand-in for [`as_user_in`]: answers `--help` with the
    /// option `as_user` checks for, records its other arguments, and runs
    /// what follows `--` without switching users.
    const FAKE_SETPRIV: &str = r#"#!/bin/sh
case "$1" in --help) echo "--reuid"; exit 0;; esac
for a; do shift; [ "$a" = -- ] && break; printf '%s\n' "$a" >> "${0%/*}/args"; done
exec "$@"
"#;

    /// The whole switch to a user, with a fake `setpriv` in a stand-in for
    /// the system folders: nothing is looked up through the given `PATH`,
    /// and the command runs with that `PATH` and the user's name and home.
    #[test]
    fn as_user_hands_the_command_the_path_it_was_given() {
        use std::os::unix::fs::PermissionsExt as _;
        let sys = tempfile::tempdir().unwrap();
        for name in ["id", "awk", "grep"] {
            std::os::unix::fs::symlink(markers::real(name).unwrap(), sys.path().join(name))
                .unwrap();
        }
        let setpriv = sys.path().join("setpriv");
        std::fs::write(&setpriv, FAKE_SETPRIV).unwrap();
        std::fs::set_permissions(&setpriv, std::fs::Permissions::from_mode(0o755)).unwrap();
        let sys_path = format!("{}:/usr/bin:/bin", sys.path().display());
        let m = markers::Markers::new();
        let print = r#"printf '%s|%s|%s' "$PATH" "$USER" "$HOME""#;

        // A user in /etc/passwd: `nobody` is there on Linux and macOS.
        let home = std::fs::read_to_string("/etc/passwd")
            .unwrap()
            .lines()
            .find_map(|l| {
                let f: Vec<_> = l.split(':').collect();
                (f.first() == Some(&"nobody")).then(|| f[5].to_string())
            })
            .unwrap();
        let (code, out, err) = m.sh(&as_user_in(&sys_path, "nobody", print, &[]));
        assert_eq!(code, 0, "{err}");
        assert_eq!(out, format!("{}|nobody|{home}", m.path()));
        let args = std::fs::read_to_string(sys.path().join("args")).unwrap();
        assert!(args.contains("--init-groups"), "{args}");
        std::fs::remove_file(sys.path().join("args")).unwrap();

        // A uid with no entry.
        let (code, out, err) = m.sh(&as_user_in(&sys_path, "4242424", print, &[]));
        assert_eq!(code, 0, "{err}");
        assert_eq!(out, format!("{}|4242424|/", m.path()));
        let args = std::fs::read_to_string(sys.path().join("args")).unwrap();
        assert!(args.contains("--reuid=4242424\n"), "{args}");
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    /// The kill runs as root after the agent's turn; `grep` is BenchFlow's.
    #[test]
    fn the_kill_finds_no_program_through_the_given_path() {
        let m = markers::Markers::new();
        let (code, _, err) = m.sh(&kill_tagged("no-such-exec"));
        assert_eq!(code, 0, "{err}");
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    #[test]
    fn with_system_path_pins_path_for_the_script() {
        let m = markers::Markers::new();
        let (code, out, _) = m.sh(&with_system_path(r#"cat </dev/null; printf %s "$PATH""#));
        assert_eq!((code, out.as_str()), (0, SYSTEM_PATH));
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    /// The whole switch to a user with the loader variables set (and
    /// stashed, as the backends do first): `id`, `awk`, `grep` and `setpriv`
    /// start without them, the user's shell gets them back (and reads
    /// `BASH_ENV`, as a plain `bash -c` would), and with `unset` the
    /// command runs without them.
    #[test]
    fn as_user_starts_root_programs_without_loader_variables() {
        let r = recorders::Recorders::new();
        let m = markers::Markers::new();
        let env = recorders::loader_env(&r.read_file());
        let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let script = |unset: &[String]| {
            format!(
                "{}{}",
                stash_loader_vars(),
                as_user_in(&r.sys(), "nobody", recorders::PRINT_LOADER_VARS, unset)
            )
        };
        let (code, out, err) = m.sh_env(&script(&[]), &env);
        assert_eq!(code, 0, "{err}");
        let read = r.read_file();
        assert_eq!(out, format!("[ ][/nonexistent-bf-lib][{read}][{read}]"));
        let ran = r.ran();
        // No loader or start-up variable reaches any program on the way, the
        // command's own shell included: they are put back inside it.
        assert!(
            ran.iter().all(|l| !l.contains(' ')),
            "a program on the way saw a loader variable: {ran:?}"
        );
        for step in ["id", "awk", "grep"] {
            assert!(ran.iter().any(|l| l == step), "{step} did not run: {ran:?}");
        }
        assert!(
            ran.iter().any(|l| l.starts_with("setpriv")),
            "setpriv did not run: {ran:?}"
        );
        // The command's shell starts with none of them, then puts back the
        // loader variables and sources `BASH_ENV` itself, once (conda/nvm
        // recipes), as a plain `bash -c` would at start-up.
        assert_eq!(ran.last().unwrap(), "bash", "{ran:?}");
        assert_eq!(r.reads(), 1, "the command's shell sources BASH_ENV once");
        assert!(r.setpriv_args().contains(&"--init-groups".to_string()));

        let r2 = recorders::Recorders::new();
        let env2 = recorders::loader_env(&r2.read_file());
        let env2: Vec<(&str, &str)> = env2.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let unset: Vec<String> = LOADER_VARS.iter().map(|v| v.to_string()).collect();
        let script = format!(
            "{}{}",
            stash_loader_vars(),
            as_user_in(&r2.sys(), "nobody", recorders::PRINT_LOADER_VARS, &unset)
        );
        let (code, out, err) = m.sh_env(&script, &env2);
        assert_eq!(code, 0, "{err}");
        assert_eq!(out, "[unset][unset][unset][unset]");
        assert!(r2.ran().iter().all(|l| !l.contains(' ')), "{:?}", r2.ran());
        assert_eq!(r2.reads(), 0);
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    /// `command_shell` hands back what a stash moved aside, and drops the
    /// variables in `unset` whatever their source.
    #[test]
    fn command_shell_restores_stashed_variables_and_drops_unset_ones() {
        let print = r#"printf '[%s][%s][%s]' "${LD_PRELOAD-unset}" "${BF_KEEP_LD_PRELOAD-unset}" "${GCONV_PATH-unset}""#;
        let script = format!(
            "LD_PRELOAD=/x.so; export LD_PRELOAD; {}/bin/sh -c {}",
            stash_loader_vars(),
            quote(&command_shell(print, &[]))
        );
        let (code, out) = sh(&script);
        assert_eq!((code, out.as_str()), (0, "[/x.so][unset][unset]"));
        let script = format!(
            "LD_PRELOAD=/x.so; GCONV_PATH=; export LD_PRELOAD GCONV_PATH; {}/bin/sh -c {}",
            stash_loader_vars(),
            quote(&command_shell(
                print,
                &["LD_PRELOAD".to_string(), "GCONV_PATH".to_string()]
            ))
        );
        let (code, out) = sh(&script);
        assert_eq!((code, out.as_str()), (0, "[unset][unset][unset]"));
    }

    /// A long command is named once, not copied, so it fits in one
    /// argument (it used to be pasted about a dozen times, three quoting
    /// levels deep). 100 KB of realistic command text runs through both the
    /// plain shell and the switch to a user. The binding limit is the
    /// kernel's cap on one argument (`MAX_ARG_STRLEN`, 128 KB on Linux):
    /// since the command is now named once, a command of about that size
    /// fits, less the overhead of quoting single quotes (each becomes four
    /// bytes), so a single-quote-dense command reaches the cap sooner. A
    /// larger command would need staging through a file.
    #[test]
    fn a_long_command_runs_through_the_wrapper() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        // 100 KB of realistic command text: double quotes, spaces, `$`, `;`
        // (the bytes a pasted file holds), without single quotes, which is
        // the common case (an agent pasting a file through tmux send-keys).
        let chunk = "x=\"a b c\"; echo $x > /dev/null; ";
        let big = chunk.repeat(100 * 1024 / chunk.len() + 1);
        let command = format!(
            "printf %s {} > {}; printf done >> {}",
            quote(&big),
            quote(&out.to_string_lossy()),
            quote(&out.to_string_lossy()),
        );
        let line = command_shell(&command, &[]);
        assert!(
            line.matches("printf %s").count() == 1,
            "the command is named once"
        );
        let (code, _, err) = markers::Markers::new().sh(&line);
        assert_eq!(code, 0, "{err}");
        let got = std::fs::read_to_string(&out).unwrap();
        assert_eq!(got, format!("{big}done"));
        assert!(
            big.len() >= 100 * 1024,
            "the command body is {} bytes",
            big.len()
        );

        // The same through the switch to a user: `as_user` names the command
        // once, and it runs (through the fake `setpriv` of a stand-in system
        // folder, so no real privilege is needed on the host).
        let out2 = dir.path().join("out2");
        let me = nix_uid();
        let command = format!(
            "printf %s {} > {}",
            quote(&big),
            quote(&out2.to_string_lossy())
        );
        let line = as_user(&me, &command, &[]);
        assert_eq!(line.matches("printf %s").count(), 1, "named once");
        let r = recorders::Recorders::new();
        let (code, _, err) = markers::Markers::new().sh(&as_user_in(&r.sys(), &me, &command, &[]));
        assert_eq!(code, 0, "{err}");
        assert_eq!(std::fs::read_to_string(&out2).unwrap(), big);
    }

    /// This process's own uid as a string, so `as_user` switches to the
    /// user it is already (no privilege needed on the host).
    fn nix_uid() -> String {
        let out = std::process::Command::new("id").arg("-u").output().unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    #[test]
    fn unset_vars_leaves_out_names_a_shell_cannot_unset() {
        assert_eq!(unset_vars(&[]), "");
        assert_eq!(
            unset_vars(&["A".into(), "B;rm".into(), "C_1".into()]),
            "unset A C_1; "
        );
    }
}
