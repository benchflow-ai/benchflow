//! The shell programs that set a sandbox up and hand it from the agent to
//! the verifier.
//!
//! Every program here runs as root inside the sandbox through `bash -c` (or
//! `sh -c`), and uses only POSIX shell and tools that busybox and coreutils
//! both have. Each prints machine-readable lines (`BF <tag> ...`) that the
//! runner parses into the trial's defense record and custody report.
//!
//! The order of a trial:
//!
//! 1. [`setup_user`]: create the non-root agent user and hand it the workdir.
//! 2. [`lock_hidden`]: lock any verifier or oracle folders the image ships,
//!    [`lock_system_path`]: lock folders on `PATH` the agent could write,
//!    and [`lock_system_dirs`]: remove group and other write permission
//!    from system folders (`/usr`, `/etc`, `/opt`, ...).
//! 3. [`baseline`]: snapshot the files that steer test runners
//!    (`conftest.py`, `pytest.ini`, `sitecustomize.py`, `.pth`, pytest
//!    plugin entry points, ...), and
//!    [`preload_snapshot`]: snapshot `/etc/ld.so.preload`.
//! 4. The agent works, every command it runs prefixed by [`oom_prefix`] so
//!    the kernel stops its processes first when memory runs out.
//! 5. [`kill_user`]: kill every process of the agent user and confirm none is
//!    left.
//! 6. [`handoff`]: clear `/logs/verifier`, remove special files and escaping
//!    symlinks from the workdir, and remove or restore planted and edited
//!    test-runner files against the baseline; [`preload_check`]: put
//!    `/etc/ld.so.preload` back as it was.
//! 7. [`trusted_path`]: the verifier's `PATH`, root-owned folders only;
//!    [`verifier_env_check`]: confirm the verifier's environment holds no
//!    loader variables before it runs, under [`verifier_env_prefix`].

use crate::shell_quote as q;

/// The variable that carries the image's own `PATH` into the root programs
/// that judge it ([`lock_system_path`], [`trusted_path`]): those run, like
/// every root command of the runner, with `PATH` set to
/// [`benchflow_sandbox::shell::SYSTEM_PATH`].
pub const IMAGE_PATH_VAR: &str = "BENCHFLOW_IMAGE_PATH";

/// Where BenchFlow keeps root-only state inside a sandbox: the baseline the
/// handoff restores from, the `/etc/ld.so.preload` snapshot, the packed
/// outputs.
///
/// Not under `/root`: a task with no declared workdir works in its image's
/// WORKDIR, which is often `/root`, and the agent user owns its workdir; the
/// owner of a folder can rename a root-owned folder inside it and put its
/// own in place. [`setup_user`] keeps the folders above it root's, and the
/// runner refuses a workdir that holds it ([`holds_state`]).
pub const STATE_DIR: &str = "/var/lib/benchflow";

/// The first line of every root program that writes into [`STATE_DIR`]
/// (the handoff, which also restores files elsewhere, and the network
/// policy's hosts pinning set it only around their own writes there): what
/// it makes is root's alone whatever umask the backend's exec starts with (a
/// Docker daemon's `docker exec` can start at `0000`, and the handoff then
/// found the baseline writable by others and stopped every trial of a
/// rootless agent with R314).
const STATE_UMASK: &str = "umask 077";

/// Every root command BenchFlow sends starts with this, so a folder root
/// creates (a missing mount or workdir parent, `/logs/verifier` between
/// stages, the parents a handoff restore recreates) is `0755`, not the
/// `0777` a `docker exec` umask of `0000` — which this campaign's
/// Docker-in-Docker daemon has — would give. The state writers set their
/// own [`STATE_UMASK`] afterwards, so their folders stay `0700`.
pub const PLUMBING_UMASK: &str = "umask 022";

/// `script` run with the plumbing umask ([`PLUMBING_UMASK`]) set first.
pub fn with_plumbing_umask(script: &str) -> String {
    format!("{PLUMBING_UMASK}\n{script}")
}

/// Whether giving the agent user `workdir` would give it the folder that
/// holds [`STATE_DIR`]: the workdir is `/`, the state folder, or one of the
/// folders above it.
pub fn holds_state(workdir: &str) -> bool {
    let w = workdir.trim_end_matches('/');
    w.is_empty() || STATE_DIR == w || STATE_DIR.starts_with(&format!("{w}/"))
}

/// Make [`STATE_DIR`] empty and root's alone, and the folders above it
/// root's and not writable by others, so no other user can move it aside.
/// Whatever an image ships there is removed: BenchFlow fills it during the
/// trial, and the handoff refuses anything in it that is not root's.
fn make_state_dir() -> String {
    format!(
        r#"rm -rf {state} 2>/dev/null || true; mkdir -p {state}; chown root:root {state}; chmod 700 {state}
d={state}; while d=$(dirname "$d"); [ "$d" != / ]; do chown root:root "$d" 2>/dev/null; chmod go-w "$d" 2>/dev/null; done"#,
        state = q(STATE_DIR),
    )
}

/// The root programs that prepare a sandbox no agent user works in (an
/// agent running as root, or a separate verifier's fresh sandbox): the log
/// folders, `extra` (quoted folders) made too, and [`STATE_DIR`].
pub fn prepare_root(extra: &[&str]) -> String {
    let extra: Vec<String> = extra.iter().map(|d| q(d)).collect();
    format!(
        "mkdir -p {extra} /logs/agent /logs/verifier /logs/artifacts\n{state}",
        extra = extra.join(" "),
        state = make_state_dir(),
    )
}

/// The sandbox user agents run as unless a task names another.
pub const DEFAULT_USER: &str = "agent";

/// Files that change how pytest and Python behave at startup. An agent that
/// plants or edits one can make tests pass without doing the task.
///
/// Project files (`Makefile`, `package.json`, `pyproject.toml`, `setup.cfg`,
/// `tox.ini`, JS test configs) are not here: they are often the answer
/// itself ("write a Makefile", "fix the build"), and removing or restoring
/// them undid correct work. BenchFlow 0.8 never touched them either.
///
/// `entry_points.txt` counts only when it mentions `pytest11` anywhere, the
/// group pytest loads plugins from on its own (Python's own reader also takes
/// `[[pytest11]]` and other spellings a line match would miss): [`handoff`]
/// keeps every other one the agent's installs wrote. BenchFlow 0.8 refuses
/// such plugins with its pytest plugin guard.
///
/// A link with one of these names counts as the file would: pytest and
/// Python follow it.
const STEERING_FILES: &[&str] = &[
    "conftest.py",
    "pytest.ini",
    "sitecustomize.py",
    "usercustomize.py",
    "*.pth",
    "entry_points.txt",
];

/// Whether a file name is one of [`STEERING_FILES`] (an `entry_points.txt`
/// then counts only when it names `pytest11`, as [`SKIP_UNLESS_STEERING`]
/// says).
pub(crate) fn is_steering_name(name: &str) -> bool {
    // `find -name '*.pth'` matches `.pth` itself: `*` may match nothing.
    STEERING_FILES.iter().any(|n| match n.strip_prefix('*') {
        Some(ext) => name.ends_with(ext),
        None => name == *n,
    })
}

/// Shell: `continue` the enclosing loop over `$f` unless `$f` steers
/// pytest. Every steering file does, except an `entry_points.txt` that never
/// mentions `pytest11`.
const SKIP_UNLESS_STEERING: &str =
    r#"case "$f" in */entry_points.txt) grep -Fq pytest11 "$f" 2>/dev/null || continue;; esac"#;

/// `find` tests for a steering file: one of [`STEERING_FILES`]' names, a
/// regular file or a link.
fn steering_find_expr() -> String {
    let names: Vec<String> = STEERING_FILES
        .iter()
        .map(|n| format!("-name {}", q(n)))
        .collect();
    format!("\\( -type f -o -type l \\) \\( {} \\)", names.join(" -o "))
}

/// The folders the steering-file scan skips: kernel filesystems, the package
/// caches agents fill, BenchFlow's own state, and the root-owned system
/// folders where package installs put `.pth` files and test suites
/// (`/usr/lib`, `/usr/local/lib`, `/etc`), as 0.8 skips them. A non-root
/// agent cannot write there; a solution's `pip install -e .` as root writes a
/// `.pth` there that the verifier needs.
fn prune_expr() -> String {
    let dirs = [
        "/proc",
        "/sys",
        "/dev",
        "/run",
        STATE_DIR,
        "/opt/benchflow",
        "/var/lib/apt",
        "/var/cache",
        "/usr/share",
        "/usr/lib",
        "/usr/local/lib",
        "/etc",
    ];
    let mut parts: Vec<String> = dirs.iter().map(|d| format!("-path {}", q(d))).collect();
    parts.push("-name node_modules".to_string());
    parts.push("-name .git".to_string());
    format!("\\( {} \\) -prune", parts.join(" -o "))
}

/// Folders outside the workdir that tasks tell the agent to write its answer
/// to; BenchFlow 0.8 hands each to the agent user when the image has it
/// (SkillsBench's `civ6-adjacency-optimizer` builds `/output` root-owned).
pub const OUTPUT_DIRS: [&str; 2] = ["/output", "/outputs"];

/// Hand `dirs` to `owner` (`user:group`) when each is a real folder; a
/// symlink is left alone, so the image cannot point the chown elsewhere.
fn own_dirs(owner: &str, dirs: &[&str]) -> String {
    let list: Vec<String> = dirs.iter().map(|d| q(d)).collect();
    format!(
        r#"for d in {list}; do if [ -d "$d" ] && [ ! -L "$d" ]; then chown -R {owner} "$d"; fi; done"#,
        list = list.join(" "),
    )
}

/// Create the agent user (if missing), give it a home, the workdir, the
/// [`OUTPUT_DIRS`] the image has, and `/logs/agent`, and keep
/// `/logs/verifier` root-only during the agent's run.
///
/// Prints `BF user <name> <uid>`; exits 3 when the image can create no user.
pub fn setup_user(user: &str, workdir: &str) -> String {
    format!(
        r#"set -e
u={user}; w={workdir}
if ! id -u "$u" >/dev/null 2>&1; then
  if command -v useradd >/dev/null 2>&1; then
    useradd -m -s /bin/bash "$u" 2>/dev/null || useradd -m "$u"
  elif command -v adduser >/dev/null 2>&1; then
    adduser -D "$u" 2>/dev/null || adduser --disabled-password --gecos '' "$u"
  else
    echo "BF error no-useradd" >&2; exit 3
  fi
fi
g=$(id -gn "$u")
home=$(sh -c "echo ~$u")
[ -d "$home" ] || mkdir -p "$home"
chown "$u:$g" "$home"
mkdir -p "$w" /logs/agent /logs/artifacts /logs/verifier
chown -R "$u:$g" "$w" /logs/agent /logs/artifacts
{outputs}
chown root:root /logs/verifier
chmod 755 /logs
chmod 700 /logs/verifier
{state}
echo "BF user $u $(id -u "$u") $home""#,
        user = q(user),
        workdir = q(workdir),
        state = make_state_dir(),
        outputs = own_dirs(r#""$u:$g""#, &OUTPUT_DIRS),
    )
}

/// Let root's git work in the workdir, which `setup_user` gave the agent
/// user: git refuses a repository another user owns ("dubious ownership").
/// Run as root before a reference solution that runs as root, as BenchFlow
/// 0.8 does; best effort, since not every image has git. The solution runs
/// with `HOME=/root`, so the entry goes there too: an exec inherits an
/// image's `ENV HOME`, which may name the workdir or another user's home.
pub fn trust_workdir_for_git(workdir: &str) -> String {
    format!(
        "command -v git >/dev/null 2>&1 && HOME=/root git config --global --add safe.directory {} 2>/dev/null || true",
        q(workdir)
    )
}

/// Give the workdir and everything in it to `owner` and its group, staying
/// on the workdir's own filesystem and changing links themselves, never
/// what they point at. After the handoff the owner is root (or the task's
/// verifier user), as BenchFlow 0.8 gives the workspace to root before
/// grading: programs that check who owns a folder then work for the
/// verifier, such as git, which refuses a repository another user owns
/// ("detected dubious ownership"). Before a later stage the owner is the
/// agent user again; only after a graded stage, whose handoff stopped every
/// agent process, so nothing can swap a path while this runs.
///
/// A file with more than one hard link keeps its owner unless the agent
/// user `agent` owns it: the other link may be a file outside the workdir
/// (an agent can link `/etc/passwd` into it), and changing the owner of one
/// link changes the file's; the agent user's own file it may give away.
///
/// The workdir itself changes first: when that fails (a read-only mount),
/// or the image has no `find` that takes the options used, it exits 3,
/// and with no `owner` user, or a workdir of `/`, it exits 2 (`BF error
/// ...` on stderr). Then `BF root-owned`. A workdir that is a link (the
/// image's `/app -> /srv/app`) is walked where it leads when the folder
/// holding the link is root's alone, so no other user could have made it;
/// otherwise only the link changes, with a `BF unchanged` note. The rest is walked
/// from inside each folder where `find` can (`-execdir`), so no path grows
/// past the system's limit however deep the agent nested its folders. A
/// path that could not be changed is not fatal: the first 5 errors are
/// printed as `BF unchanged <error>` (one line each); a file that went away
/// during the walk is no error. Ends with `BF owned <owner>`.
pub fn own_workdir(workdir: &str, owner: &str, agent: &str) -> String {
    format!(
        r#"w={workdir}; o={owner}; a={agent}
r=$(readlink -f "$w" 2>/dev/null); [ -n "$r" ] || r=$w
for p in "$w" "$r"; do case "$p" in /|'') echo "BF error workdir $w" >&2; exit 2;; esac; done
g=$(id -gn "$o" 2>/dev/null) || {{ echo "BF error no-user $o" >&2; exit 2; }}
mine=; au=$(id -u "$a" 2>/dev/null) && case "$au" in ''|*[!0-9]*) ;; *) mine="-o -user $au";; esac
if [ -e "$w" ]; then
  command -v find >/dev/null 2>&1 || {{ echo "BF error no-find" >&2; exit 3; }}
  find "$w" -xdev -mindepth 1 -maxdepth 0 \( -type d -o -links 1 \) -print >/dev/null 2>&1 || {{ echo "BF error find" >&2; exit 3; }}
fi
t=$w
if [ -L "$w" ]; then
  p=$(dirname "$w"); pm=$(stat -Lc %A "$p" 2>/dev/null)
  case "$pm" in ?????w????|????????w?) pm=open;; esac
  if [ "$(stat -Lc %u "$p" 2>/dev/null)" = 0 ] && [ "$pm" != open ]; then t=$r
  else echo "BF unchanged $w is a link in a folder others may write, so it was not followed"; fi
fi
if [ -d "$t" ]; then
  chown -h "$o:$g" "$t" || {{ echo "BF error chown $w" >&2; exit 3; }}
  echo "BF root-owned"
  if find "$t" -maxdepth 0 -execdir true \; 2>/dev/null; then x=-execdir; else x=-exec; fi
  # The errors are streamed, never held: a deep tree can give millions.
  # Each filter reads to the end, so find never meets a closed pipe.
  find "$t" -xdev -mindepth 1 \( -type d -o -links 1 $mine \) $x chown -h "$o:$g" {{}} + 2>&1 >/dev/null \
    | grep -v 'No such file or directory' | grep . | sed -n '1,5p' \
    | while IFS= read -r l; do printf 'BF unchanged %s\n' "$(printf %s "$l" | tr -d '\r')"; done
fi
echo "BF owned $o""#,
        workdir = q(workdir),
        owner = q(owner),
        agent = q(agent),
    )
}

/// Read-only checks, run as the agent user before the agent starts, for a
/// way it can become root: `sudo` without a password, a `doas` rule
/// without one, system files it can write, a group or socket that is
/// root's equal, setuid-root shells and interpreters, file capabilities
/// that grant root's powers, and capabilities of its own. Nothing is run
/// as root: `sudo -n -l` only lists. Prints `BF root-path <how>` for each
/// way found, and `BF root-path-unchecked <what>` for a check the image
/// has no tool for.
pub const ROOT_PATHS_AS_USER: &str = r#"bf_t() { s=$1; shift; if command -v timeout >/dev/null 2>&1; then timeout "$s" "$@"; else "$@"; fi; }
if command -v sudo >/dev/null 2>&1; then
  if out=$(bf_t 20 sudo -n -l </dev/null 2>/dev/null); then
    rules=$(printf '%s\n' "$out" | grep -E '^[[:space:]]+\(' | grep NOPASSWD | head -n 3)
    [ -n "$rules" ] || rules=$(printf '%s\n' "$out" | grep -E '^[[:space:]]+\(' | head -n 3)
    rules=$(printf '%s\n' "$rules" | sed 's/^[[:space:]]*//' | tr '\n' ';' | sed 's/;$//; s/;/; /g')
    [ -n "$rules" ] && echo "BF root-path sudo runs commands for it without a password ($rules)"
  fi
fi
if [ -r /etc/doas.conf ]; then
  me=$(id -un); gs=" $(id -Gn) "
  awk -v me="$me" -v gs="$gs" '$1 == "permit" {
    nopass = 0; who = ""
    for (i = 2; i <= NF; i++) { if ($i == "nopass") nopass = 1; else if ($i ~ /^(persist|keepenv|nolog|setenv)$/ || $i ~ /[{}]/) continue; else { who = $i; break } }
    if (nopass && (who == me || (substr(who, 1, 1) == ":" && index(gs, " " substr(who, 2) " ")))) { print "BF root-path doas runs commands for it without a password (" $0 ")"; exit }
  }' /etc/doas.conf
fi
for f in /etc/passwd /etc/shadow /etc/group /etc/sudoers /etc/sudoers.d /etc/doas.conf; do
  [ -e "$f" ] && [ -w "$f" ] && echo "BF root-path it can write $f"
done
for g in $(id -Gn 2>/dev/null); do [ "$g" = disk ] && echo "BF root-path it is in the disk group (raw access to the disks)"; done
for f in /var/run/docker.sock /run/docker.sock /run/podman/podman.sock /var/lib/lxd/unix.socket; do
  [ -S "$f" ] && [ -w "$f" ] && echo "BF root-path it can use the container daemon at $f"
done
cap=$(awk '/^CapEff:/ { print $2 }' /proc/self/status 2>/dev/null)
case $cap in ''|0000000000000000) ;; *) echo "BF root-path its processes hold capabilities (CapEff $cap)";; esac
if command -v find >/dev/null 2>&1; then
  bf_t 60 find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o -type f -perm -4000 -user root -print 2>/dev/null | while IFS= read -r f; do
    case ${f##*/} in
      sh|bash|dash|zsh|ksh|mksh|csh|tcsh|fish|ash|busybox|python|python[0-9]*|perl|perl[0-9]*|ruby*|lua*|node|php*|env|find|vi|vim|vim.*|nano|less|more|awk|gawk|mawk|nawk|cp|mv|tee|dd|chmod|chown|install|rsync|tar|cat|sed|xargs|nice|timeout|stdbuf|strace|gdb|nsenter|unshare|chroot|setpriv|capsh|docker)
        echo "BF root-path $f is setuid root";;
    esac
  done
else
  echo "BF root-path-unchecked setuid files (no find)"
fi
if command -v getcap >/dev/null 2>&1; then
  bf_t 60 getcap -r /usr /bin /sbin /opt /home 2>/dev/null | grep -Ei 'cap_(setuid|setgid|sys_admin|dac_override|dac_read_search|chown|fowner|sys_ptrace|sys_module|net_admin)|[[:space:]]=ep' | head -n 5 | while IFS= read -r l; do echo "BF root-path $l (a file capability)"; done
else
  echo "BF root-path-unchecked file capabilities (no getcap)"
fi
true
"#;

/// The read-only checks that need root to read `/etc/shadow`: a uid-0
/// account with no password (so `su` asks for none), and an agent user
/// with no password that `sudo` lists rules for. Prints `BF root-path`.
pub fn root_paths_as_root(user: &str) -> String {
    format!(
        r#"u={user}
awk -F: '$3 == 0 && $2 == "" {{ print $1 }}' /etc/passwd 2>/dev/null | while read -r a; do echo "BF root-path $a (uid 0) has no password at all"; done
if [ -r /etc/shadow ]; then
  if command -v su >/dev/null 2>&1; then
    awk -F: 'NR == FNR {{ if ($3 == 0) z[$1] = 1; next }} ($1 in z) && $2 == "" {{ print $1 }}' /etc/passwd /etc/shadow | while read -r a; do echo "BF root-path $a (uid 0) has an empty password, so su asks for none"; done
  fi
  if command -v sudo >/dev/null 2>&1 && awk -F: -v u="$u" '$1 == u && $2 == "" {{ f = 1 }} END {{ exit !f }}' /etc/shadow; then
    sudo -l -U "$u" 2>/dev/null | grep -qE '^[[:space:]]+\(' && echo "BF root-path it has an empty password, and sudo has rules for it"
  fi
fi
true
"#,
        user = q(user),
    )
}

/// Lock folders the image already ships at the verifier and oracle mounts, so
/// the agent cannot read hidden tests or the reference solution. Prints
/// `BF locked <path>` for each one found.
pub fn lock_hidden(paths: &[String]) -> String {
    let list: Vec<String> = paths.iter().map(|p| q(p)).collect();
    format!(
        r#"for d in {list}; do
  if [ -L "$d" ]; then echo "BF locked-symlink $d"; rm -f "$d"; continue; fi
  [ -e "$d" ] || continue
  chown -R root:root "$d" && chmod 700 "$d" && echo "BF locked $d"
done
true"#,
        list = list.join(" ")
    )
}

/// Lock the folders on the sandbox's `PATH` (and the folders above them)
/// that someone other than root could write, before the agent starts.
///
/// Every command the runner sends, as root, resolves `sh` and `bash` through
/// this `PATH`, and so does the sandbox's own exec service (Daytona's daemon
/// starts each command session that way). An agent that can write one of
/// these folders could replace `sh` and have the runner, or the verifier, run
/// its program as root. Folders in the workdir, homes, and temp folders are
/// left alone, as [`trusted_path`] leaves them out. Prints
/// `BF unlocked-path <folder>` for each folder it locked.
pub fn lock_system_path(workdir: &str) -> String {
    format!(
        r#"w={workdir}
IFS=:
for d in ${{{IMAGE_PATH_VAR}:-${{PATH:-}}}}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; do
  case "$d" in /*) ;; *) continue;; esac
  case "$d" in "$w"|"$w"/*|/tmp|/tmp/*|/var/tmp|/var/tmp/*|/home/*|/dev/shm*) continue;; esac
  [ -d "$d" ] || continue
  d=$(readlink -f "$d" 2>/dev/null || echo "$d")
  while :; do
    m=$(stat -c %A "$d" 2>/dev/null); u=$(stat -c %u "$d" 2>/dev/null)
    case "$m" in ?????w????|????????w?) bad=1;; *) bad=;; esac
    [ -n "$u" ] && [ "$u" != 0 ] && bad=1
    if [ -n "$bad" ]; then chown root:root "$d" 2>/dev/null; chmod go-w "$d" && echo "BF unlocked-path $d"; fi
    [ "$d" = / ] && break
    d=$(dirname "$d")
  done
done
true"#,
        workdir = q(workdir)
    )
}

/// A shell function, `foreign_links <path>`, that prints every symlink on
/// the way to `<path>` (itself included) that someone other than root owns.
/// A symlink belongs to whoever made it, so these are the ones the agent
/// (or another user) made: a root program that writes or reads through one
/// goes where that user chose.
const FOREIGN_LINKS: &str = r#"foreign_links() (
  p=; IFS=/
  for c in ${1#/}; do
    [ -n "$c" ] || continue
    p="$p/$c"
    if [ -L "$p" ] && [ "$(stat -c %u "$p" 2>/dev/null)" != 0 ]; then printf '%s\n' "$p"; fi
  done
)"#;

/// Make `path` safe for root to write into after the agent's turn: remove
/// every symlink on the way to it that root does not own (`BF symlink
/// <link>`), and, with `foreign`, remove `path` itself when anything in it
/// belongs to another user (`BF planted <path>`): files root uploads there
/// next (the verifier, a stage's mounts) must not mix with files the agent
/// left. Ends with `BF claimed <path>`.
pub fn claim(path: &str, foreign: bool) -> String {
    let drop_foreign = if foreign {
        r#"if [ -e "$t" ] && [ -n "$(find "$t" ! -user 0 -print 2>/dev/null | head -n 1)" ]; then
  echo "BF planted $t"; rm -rf -- "$t"
fi"#
    } else {
        ""
    };
    format!(
        r#"{FOREIGN_LINKS}
t={path}
foreign_links "$t" | while IFS= read -r l; do echo "BF symlink $l"; rm -f -- "$l"; done
{drop_foreign}
echo "BF claimed $t""#,
        path = q(path),
    )
}

/// Print `BF open <folder>` for each folder above `path` that someone other
/// than root owns, or that group or others can write without the sticky
/// bit: whoever can write it can move `path` aside and put other files in
/// its place.
pub fn open_parents(path: &str) -> String {
    format!(
        r#"d={path}
while :; do
  d=$(dirname "$d")
  if [ -d "$d" ]; then
    u=$(stat -Lc %u "$d" 2>/dev/null); m=$(stat -Lc %A "$d" 2>/dev/null); o=
    [ "$u" = 0 ] || o=1
    case "$m" in ?????w????|????????w?) [ -k "$d" ] || o=1;; esac
    [ -n "$o" ] && echo "BF open $d"
  fi
  [ "$d" = / ] && break
done
true"#,
        path = q(path),
    )
}

/// Run as root: print `BF open <folder>` for each folder above `path` the
/// user `agent` can make an entry in: its own, one others may write (the
/// sticky bit does not stop a new entry, such as a link where root is about
/// to make a folder), or one its group may write. The check is root's, not
/// the agent user's: a process the agent left running could change what a
/// check of its own user prints. Prints `BF open ?` when the user is unknown,
/// and `BF target-closed` when `path` is already a folder the user cannot
/// make entries in (an earlier mount, or the image, placed it root's).
pub fn agent_writable_parents(path: &str, agent: &str) -> String {
    format!(
        r#"d={path}; a={agent}; t=$d
au=$(id -u "$a" 2>/dev/null) || {{ echo "BF open ?"; exit 0; }}
ag=" $(id -G "$a" 2>/dev/null) "
may() {{
  u=$(stat -Lc %u "$1" 2>/dev/null); g=$(stat -Lc %g "$1" 2>/dev/null); m=$(stat -Lc %A "$1" 2>/dev/null)
  [ "$u" = "$au" ] && return 0
  case "$m" in ????????w?) return 0;; esac
  case "$m" in ?????w????) case "$ag" in *" $g "*) return 0;; esac;; esac
  return 1
}}
while :; do
  d=$(dirname "$d")
  if [ -d "$d" ] && may "$d"; then echo "BF open $d"; fi
  [ "$d" = / ] && break
done
if [ -d "$t" ] && [ ! -L "$t" ] && ! may "$t"; then echo "BF target-closed"; fi
true"#,
        path = q(path),
        agent = q(agent),
    )
}

/// Snapshot the steering files (path and SHA-256) and keep a copy of each, in
/// root-only state, before the agent starts; and note the `*.py` files
/// already in the temp folders.
pub fn baseline() -> String {
    format!(
        r#"{umask}
b={state}/baseline
rm -rf "$b"; mkdir -p "$b"; chmod 700 {state} "$b"
find / -xdev {prune} -o {names} -print 2>/dev/null | LC_ALL=C sort > "$b/list"
: > "$b/sums"
while IFS= read -r f; do
  s=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1) && printf '%s  %s\n' "$s" "$f" >> "$b/sums"
done < "$b/list"
if [ -s "$b/list" ]; then tar -cf "$b/files.tar" -T "$b/list" 2>/dev/null || true; fi
find /tmp /var/tmp -xdev -name '*.py' -type f -print 2>/dev/null | while IFS= read -r f; do
  s=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1) && printf '%s  %s\n' "$s" "$f"
done > "$b/tmp-sums"
echo "BF baseline $(wc -l < "$b/list" | tr -d ' ')""#,
        umask = STATE_UMASK,
        state = q(STATE_DIR),
        prune = prune_expr(),
        names = steering_find_expr(),
    )
}

/// Add a stage's own test-runner files to [`baseline`] while an earlier
/// stage's agent may still run, rather than take the baseline again: a file
/// that agent planted or edited would then count as the task's. Each entry
/// is a path and the SHA-256 of the file the stage ships, computed on this
/// machine; BenchFlow first writes the file's copy to
/// `<baseline>/added<path>`, where the handoff restores an edited one from.
/// Ends with `BF baseline-added <n>`; prints only `BF baseline-missing` when
/// there is no baseline to add to (its own defense already says so).
pub fn baseline_add(entries: &[(String, String)]) -> String {
    let adds: String = entries
        .iter()
        .map(|(path, sum)| format!("add {} {}\n", q(path), q(sum)))
        .collect();
    format!(
        r#"{umask}
b={state}/baseline
if [ ! -f "$b/list" ] || [ ! -f "$b/sums" ]; then echo "BF baseline-missing"; exit 0; fi
# The copies' folders were made by an upload, with whatever umask it had:
# the handoff refuses a state folder anyone but root may write.
[ -d "$b/added" ] && chmod -R go-w "$b/added"
add() {{
  {{ grep -Fxv -- "$1" "$b/list"; printf '%s\n' "$1"; }} | LC_ALL=C sort > "$b/list.new" && mv "$b/list.new" "$b/list" || exit 1
  {{ while IFS= read -r line; do [ "${{line#*  }}" = "$1" ] || printf '%s\n' "$line"; done < "$b/sums"; printf '%s  %s\n' "$2" "$1"; }} > "$b/sums.new" && mv "$b/sums.new" "$b/sums" || exit 1
}}
{adds}echo "BF baseline-added {n}""#,
        umask = STATE_UMASK,
        state = q(STATE_DIR),
        n = entries.len(),
    )
}

/// Kill every process of `user` and confirm none survives (zombies excepted).
///
/// Kills from inside the user's own identity (`kill -9 -1` as that uid
/// signals everything it owns, in any PID namespace), then sweeps `/proc`,
/// several rounds. Prints `BF killed <n>`; exits 1 and prints
/// `BF survivors <pids>` when processes remain. Reads `/proc` with shell
/// builtins, so it needs neither procps nor python.
pub fn kill_user(user: &str) -> String {
    format!(
        r#"u={user}
uid=$(id -u "$u" 2>/dev/null) || {{ echo "BF error no-user" >&2; exit 2; }}
case "$uid" in ''|*[!0-9]*|0) echo "BF error bad-uid $uid" >&2; exit 2;; esac
live() {{
  for e in /proc/[0-9]*; do
    p=${{e##*/}}
    st=; eu=
    if {{ while read -r k a b rest || [ -n "$k" ]; do
            case "$k" in State:) st=$a;; Uid:) eu=$b;; esac; k=
          done < "$e/status"; }} 2>/dev/null && [ -n "$st" ]; then
      [ "$eu" = "$uid" ] && [ "$st" != Z ] && echo "$p"
    fi
  done
  return 0
}}
n=$(live | wc -l | tr -d ' ')
for r in 1 2 3 4 5 6 7 8; do
  if command -v setpriv >/dev/null 2>&1; then
    setpriv --reuid="$uid" --regid="$(id -g "$u")" --clear-groups sh -c 'kill -9 -1' 2>/dev/null
  else
    su -s /bin/sh "$u" -c 'kill -9 -1' 2>/dev/null
  fi
  for p in $(live); do kill -9 "$p" 2>/dev/null; done
  s=$(live)
  [ -z "$s" ] && {{ echo "BF killed $n"; exit 0; }}
  sleep 0.5 2>/dev/null || sleep 1
done
echo "BF survivors" $(live)
exit 1"#,
        user = q(user)
    )
}

/// What in the state folder `$sd` is not root's alone: anything not
/// root's, a link (the copy of a linked `/etc/ld.so.preload` excepted), or
/// anything others, or a group other than root's, may write. One path per
/// line; the scan crosses mounts.
const STATE_NOT_ROOTS: &str = r#"find "$sd" \( ! -user 0 -o \( \( -type l -o -perm -0002 -o \( -perm -0020 ! -group 0 \) \) ! -path "$sd/preload" \) \) -print 2>/dev/null"#;

/// The handoff's check of [`STATE_DIR`] on its own, for a sandbox that
/// carries the folder over from a checkpoint (a branch's parent before the
/// checkpoint, and each child before its agent): `BF state-owner <path>`
/// for each path not root's alone (at most 5; the folder itself when it is
/// missing or a link), `BF state-unchecked` when `find` failed (`no-find`
/// when the image has none), `BF state-baseline` when the test-runner
/// baseline is there, and `BF state-checked` last.
pub fn state_check() -> String {
    format!(
        r#"sd={state}
note() {{ printf 'BF state-owner %s\n' "$(printf %s "$1" | tr '\n' '?')"; }}
if ! command -v find >/dev/null 2>&1; then echo "BF state-unchecked no-find"
elif [ -L "$sd" ] || [ ! -d "$sd" ]; then note "$sd"
else
  hits=$({not_roots}); rc=$?
  if [ -n "$hits" ]; then
    printf '%s\n' "$hits" | head -n 5 | while IFS= read -r f; do note "$f"; done
  elif [ "$rc" != 0 ]; then echo "BF state-unchecked"; fi
  [ -f "$sd/baseline/list" ] && [ ! -L "$sd/baseline/list" ] && echo "BF state-baseline"
fi
echo "BF state-checked""#,
        state = q(STATE_DIR),
        not_roots = STATE_NOT_ROOTS,
    )
}

/// Hand the sandbox from the agent to the verifier.
///
/// - Clears `/logs/verifier` (anything the agent wrote there is discarded).
/// - Removes every file or folder whose name has a line break in it, in the
///   workdir, the temp folders, and the folders the steering scan reads
///   (`BF special ...`, the break shown as `?`): the scans below read one
///   path per line, and such a name would read as two paths, one of which
///   the agent chose (the baseline, say, which would end the cleanup).
/// - Removes FIFOs, sockets, and device files from the workdir and the
///   [`OUTPUT_DIRS`] the agent was given, and symlinks there that point
///   outside both (`BF special ...`, `BF symlink ...`): a verifier that
///   opens a FIFO hangs, and a symlink can point it at files the agent
///   should not choose.
/// - Removes steering files the agent planted and restores ones it edited
///   (`BF planted ...`, `BF restored ...`), against [`baseline`] (and the
///   copies [`baseline_add`] keeps). A file whose folder is reached through
///   a link someone other than root made, leading out of the workdir (the
///   agent can replace a folder of its own outside the workdir, such as a
///   later stage's mount it copied in, with a link to `/etc`), is neither
///   removed nor restored: `BF lost ...`.
/// - Removes `*.py` files left in `/tmp` and `/var/tmp`, which shadow
///   modules for any Python started there; files that were there unchanged
///   before the agent started (the sandbox's own, such as Daytona's
///   `daytona_repl_worker.py`) stay.
/// - Reports files over 1 GB in the workdir (`BF large ...`).
///
/// First of all, it checks that [`STATE_DIR`] is still root's alone: the
/// baseline it compares against and restores from lives there, and a
/// baseline the agent could write would let it choose what root restores.
/// When the folder is missing or a link, or anything in it is not root's,
/// is a link (the copy of a linked `/etc/ld.so.preload` excepted), or is
/// writable by others or by a group other than root's, it prints
/// `BF state-owner <path>` for each (at most 5); otherwise, when `find`
/// failed, `BF state-unchecked` (`BF state-unchecked no-find` when the image
/// has none). Either way it then prints `BF handoff stopped` and does
/// nothing else. The scan crosses mounts: a folder mounted inside the state
/// folder is checked too.
///
/// With `strict_state` false (whoever worked in the sandbox was root or
/// could become root, and so could change the folder without a trace), it
/// prints `BF state-noted <path>` instead and goes on.
///
/// Its last line is `BF handoff done` when it ran to the end; the runner
/// grades only then. A path it prints (a link's target) has any line break
/// shown as `?`, so no path the agent chose can print a line of its own.
pub fn handoff(workdir: &str, verifier_user: Option<&str>, strict_state: bool) -> String {
    let chown = match verifier_user {
        Some(u) if u != "root" && u != "0" => {
            format!("chown {u} /logs/verifier 2>/dev/null || true", u = q(u))
        }
        _ => String::new(),
    };
    format!(
        r#"w={workdir}; sd={state}; b={state}/baseline
{foreign}
out_of_reach() {{
  foreign_links "$(dirname "$1")" | while IFS= read -r l; do
    t=$(readlink -f "$l" 2>/dev/null)
    case "$t" in "$rw"|"$rw"/*) ;; *) echo out; break;; esac
  done | grep -q out
}}
{state_mode}
if ! command -v find >/dev/null 2>&1; then echo "BF state-unchecked no-find"; stop
elif [ -L "$sd" ] || [ ! -d "$sd" ]; then note "$sd"; stop
else
  hits=$({not_roots}); rc=$?
  if [ -n "$hits" ]; then
    printf '%s\n' "$hits" | head -n 5 | while IFS= read -r f; do note "$f"; done
    stop
  elif [ "$rc" != 0 ]; then echo "BF state-unchecked"; stop; fi
fi
rm -rf /logs/verifier; mkdir -p /logs/verifier; chown root:root /logs/verifier; chmod 755 /logs/verifier
{chown}
nl='
'
exec 3>&1
drop='for f; do printf "BF special %s\n" "$(printf %s "$f" | tr "\n" "?")" >&3; rm -rf -- "$f"; done; true'
for d in "$w" /tmp /var/tmp {outputs}; do
  [ -d "$d" ] && {{ [ "$d" = "$w" ] || [ ! -L "$d" ]; }} && find "$d" -xdev -name "*$nl*" -prune -exec sh -c "$drop" sh {{}} \; 2>/dev/null
done
rw=$(readlink -f "$w" 2>/dev/null || echo "$w")
for d in "$w" {outputs}; do
  [ -d "$d" ] || continue
  [ "$d" != "$w" ] && [ -L "$d" ] && continue
  find "$d" -xdev \( -type p -o -type s -o -type b -o -type c \) -print 2>/dev/null | while IFS= read -r f; do
    echo "BF special $f"; rm -f "$f"
  done
  rd=$(readlink -f "$d" 2>/dev/null || echo "$d")
  find "$d" -xdev -type l -print 2>/dev/null | while IFS= read -r l; do
    t=$(readlink -f "$l" 2>/dev/null || true)
    case "$t" in "$rw"|"$rw"/*|"$rd"|"$rd"/*) ;; *) printf 'BF symlink %s -> %s\n' "$l" "$(printf %s "$t" | tr '\n' '?')"; rm -f "$l";; esac
  done
  find "$d" -xdev -type f -size +1048576k -print 2>/dev/null | while IFS= read -r f; do echo "BF large $f"; done
done
if [ -f "$b/list" ]; then
  find / -xdev {prune} -o -name "*$nl*" -prune -exec sh -c "$drop" sh {{}} \; -o {names} -print 2>/dev/null | ({umask}; LC_ALL=C sort > "$b/after")
  while IFS= read -r f; do
    if ! grep -Fxq -- "$f" "$b/list"; then {skip}; echo "BF planted $f"; rm -f "$f"; fi
  done < "$b/after"
  while IFS= read -r line; do
    s=${{line%%  *}}; f=${{line#*  }}
    now=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
    if [ "$now" != "$s" ]; then
      {skip}
      if out_of_reach "$f"; then echo "BF lost $f"; continue; fi
      rm -rf "$f"
      if [ -f "$b/added$f" ]; then cp -p "$b/added$f" "$f" 2>/dev/null && echo "BF restored $f" || echo "BF lost $f"
      else tar -xf "$b/files.tar" -C / "${{f#/}}" 2>/dev/null && echo "BF restored $f" || echo "BF lost $f"; fi
    fi
  done < "$b/sums"
fi
find /tmp /var/tmp -xdev -name '*.py' -type f -print 2>/dev/null | while IFS= read -r f; do
  s=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
  [ -f "$b/tmp-sums" ] && grep -Fxq -- "$s  $f" "$b/tmp-sums" && continue
  echo "BF tmp-module $f"; rm -f "$f"
done
echo "BF handoff done""#,
        workdir = q(workdir),
        state = q(STATE_DIR),
        not_roots = STATE_NOT_ROOTS,
        umask = STATE_UMASK,
        state_mode = if strict_state {
            r#"note() { printf 'BF state-owner %s\n' "$1"; }
stop() { echo "BF handoff stopped"; exit 0; }"#
        } else {
            r#"note() { printf 'BF state-noted %s\n' "$1"; }
stop() { :; }"#
        },
        chown = chown,
        foreign = FOREIGN_LINKS,
        prune = prune_expr(),
        names = steering_find_expr(),
        skip = SKIP_UNLESS_STEERING,
        outputs = OUTPUT_DIRS.join(" "),
    )
}

/// Print the verifier's `PATH`: the image's `PATH` entries that are absolute,
/// root-owned, and not writable by group or others, outside the workdir and
/// temp folders. Falls back to the standard system folders.
pub fn trusted_path(workdir: &str) -> String {
    format!(
        r#"w={workdir}
out=
IFS=:
for d in ${{{IMAGE_PATH_VAR}:-${{PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}}}}; do
  case "$d" in /*) ;; *) continue;; esac
  case "$d" in "$w"|"$w"/*|/tmp|/tmp/*|/var/tmp|/var/tmp/*|/home/*|/dev/shm*) continue;; esac
  [ -d "$d" ] || continue
  [ "$(stat -c %u "$d" 2>/dev/null)" = 0 ] || continue
  m=$(stat -c %A "$d" 2>/dev/null)
  case "$m" in ?????w????|????????w?) continue;; esac
  case ":$out:" in *":$d:"*) continue;; esac
  out=${{out:+$out:}}$d
done
echo "BF path ${{out:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}}""#,
        workdir = q(workdir)
    )
}

/// The system folders [`lock_system_dirs`] takes group and other write
/// permission from. Missing ones are skipped, and so are symlinks (`/bin` is
/// often a link to `/usr/bin`, which is covered through `/usr`).
pub const SYSTEM_DIRS: &[&str] = &[
    "/usr", "/etc", "/opt", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/boot",
    "/srv", "/root",
];

/// How many changed paths [`lock_system_dirs`] names one by one; the rest
/// are only counted.
pub const SYSTEM_DIRS_NAMED: usize = 20;

/// Remove group and other write permission from everything under
/// [`SYSTEM_DIRS`] except the workdir and the agent user's home, before the
/// agent starts.
///
/// An agent that can write a library, an interpreter, or a file in `/etc`
/// can change what the verifier runs, wherever it looks programs up. Prints
/// `BF syswrite <old mode> <path>` for the first [`SYSTEM_DIRS_NAMED`]
/// paths it changed, `BF syswrite-count <n>` for all of them, and
/// `BF syswrite-left <n>` for paths still writable afterwards (0 when it
/// held). Symlinks are left alone: their own mode means nothing.
///
/// Permissions do not bind a file's owner, so it also prints
/// `BF sysowned <path>` for the first [`SYSTEM_DIRS_NAMED`] paths there
/// that `user` owns (an image's own user that a task names as its agent
/// may own a venv in `/opt`), and `BF sysowned-count <n>`. Those are left
/// as they are: the image gave them to that user on purpose.
pub fn lock_system_dirs(workdir: &str, home: &str, user: &str) -> String {
    let dirs: Vec<String> = SYSTEM_DIRS.iter().map(|d| q(d)).collect();
    format!(
        r#"{umask}
w={workdir}; h={home}; l={state}/syswrite
mkdir -p {state}; chmod 700 {state}
scan() {{
  for r in {dirs}; do
    [ -d "$r" ] && [ ! -L "$r" ] || continue
    find "$r" -xdev \( -path "$w" -o -path "$h" \) -prune -o ! -type l \( -perm -020 -o -perm -002 \) -print 2>/dev/null
  done
}}
scan > "$l"
n=0
while IFS= read -r f; do
  n=$((n + 1))
  [ "$n" -le {named} ] && echo "BF syswrite $(stat -c %a "$f" 2>/dev/null) $f"
  chmod go-w "$f" 2>/dev/null
done < "$l"
echo "BF syswrite-count $n"
echo "BF syswrite-left $(scan | wc -l | tr -d ' ')"
rm -f "$l"
u={user}
if id -u "$u" >/dev/null 2>&1; then
  n=0
  for r in {dirs}; do
    [ -d "$r" ] && [ ! -L "$r" ] || continue
    find "$r" -xdev \( -path "$w" -o -path "$h" \) -prune -o -user "$u" -print 2>/dev/null
  done > "$l"
  while IFS= read -r f; do
    n=$((n + 1))
    [ "$n" -le {named} ] && echo "BF sysowned $f"
  done < "$l"
  echo "BF sysowned-count $n"
  rm -f "$l"
fi"#,
        umask = STATE_UMASK,
        workdir = q(workdir),
        home = q(home),
        user = q(user),
        state = q(STATE_DIR),
        dirs = dirs.join(" "),
        named = SYSTEM_DIRS_NAMED,
    )
}

/// Keep a copy of `/etc/ld.so.preload` (or note that there is none) in
/// root-only state, before the agent starts. Every dynamically linked
/// program, the verifier's included, loads the libraries that file names.
/// Prints `BF preload absent` or `BF preload present <lines>`.
pub fn preload_snapshot() -> String {
    format!(
        r#"{umask}
b={state}/preload; p=/etc/ld.so.preload
mkdir -p {state}; chmod 700 {state}; rm -rf "$b" "$b.sum"
if [ -e "$p" ] || [ -L "$p" ]; then
  cp -P "$p" "$b" && {{ sha256sum < "$p" 2>/dev/null | cut -d' ' -f1; }} > "$b.sum"
  echo "BF preload present $(wc -l < "$p" | tr -d ' ')"
else
  echo absent > "$b.sum"; echo "BF preload absent"
fi"#,
        umask = STATE_UMASK,
        state = q(STATE_DIR),
    )
}

/// Compare `/etc/ld.so.preload` with [`preload_snapshot`]'s copy and put it
/// back when it changed. Prints `BF preload absent`, `BF preload unchanged`,
/// `BF preload restored`, or `BF preload no-baseline`; prints
/// `BF preload stuck` and exits 1 when it could not be put back.
pub fn preload_check() -> String {
    format!(
        r#"b={state}/preload; p=/etc/ld.so.preload
sum() {{ if [ -e "$p" ] || [ -L "$p" ]; then s=$(sha256sum < "$p" 2>/dev/null | cut -d' ' -f1); echo "${{s:-unreadable}}"; else echo absent; fi; }}
[ -f "$b.sum" ] || {{ echo "BF preload no-baseline"; exit 0; }}
before=$(cat "$b.sum"); now=$(sum)
if [ "$now" = "$before" ]; then
  if [ "$now" = absent ]; then echo "BF preload absent"; else echo "BF preload unchanged"; fi
  exit 0
fi
rm -rf "$p"
[ "$before" = absent ] || {{ cp -P "$b" "$p" && {{ [ -L "$p" ] || chmod 644 "$p"; }}; }}
if [ "$(sum)" = "$before" ]; then echo "BF preload restored"; else echo "BF preload stuck"; exit 1; fi"#,
        state = q(STATE_DIR),
    )
}

/// The value [`oom_prefix`] gives every agent process: the kernel's
/// highest, so the agent's processes are the first it stops when the
/// sandbox runs out of memory.
pub const AGENT_OOM_SCORE_ADJ: i32 = 1000;

/// A line put before every command the agent (or a scripted attempt) runs:
/// it raises the shell's `oom_score_adj`, which every process it starts
/// inherits. Raising it needs no privilege; lowering it below the value the
/// sandbox started with does, so an agent can at most undo it, never go
/// below the runner's and the verifier's.
pub fn oom_prefix() -> String {
    format!("{{ echo {AGENT_OOM_SCORE_ADJ} > /proc/self/oom_score_adj; }} 2>/dev/null\n")
}

/// Print the sandbox's memory limit as the kernel sees it: `BF mem v2 <max>
/// <swap max>` (cgroup v2), `BF mem v1 <limit> -` (cgroup v1), or
/// `BF mem none`; and `BF memtotal <kB>` from `/proc/meminfo`.
pub fn memory_limit() -> String {
    r#"if [ -r /sys/fs/cgroup/memory.max ]; then
  echo "BF mem v2 $(cat /sys/fs/cgroup/memory.max) $(cat /sys/fs/cgroup/memory.swap.max 2>/dev/null || echo -)"
elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  echo "BF mem v1 $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) -"
else
  echo "BF mem none"
fi
while read -r k v rest; do [ "$k" = MemTotal: ] && echo "BF memtotal $v"; done < /proc/meminfo 2>/dev/null
true"#
        .to_string()
}

/// Print how many processes the sandbox's out-of-memory killer ended:
/// `BF oom-kill <n>` from the memory cgroup (v2 `memory.events`, or v1
/// `memory.oom_control` on kernels that count), nothing when neither says.
pub fn oom_kills() -> String {
    r#"for f in /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory/memory.oom_control; do
  [ -r "$f" ] && while read -r k v; do [ "$k" = oom_kill ] && echo "BF oom-kill $v"; done < "$f"
done
true"#
        .to_string()
}

/// Environment variables that change how programs load or start: the
/// dynamic loader's (`LD_PRELOAD`, `LD_LIBRARY_PATH`, ...), glibc's
/// (`GCONV_PATH`, ...), and the shells' start-up files (`BASH_ENV`, `ENV`).
/// The verifier runs with every one unset, except ones the task's own
/// `[verifier] env` sets.
pub use benchflow_sandbox::shell::LOADER_VARS;

/// The loader variables the verifier runs without: [`LOADER_VARS`] minus
/// the ones the task's `[verifier] env` sets itself.
pub fn unset_vars(task_env: &std::collections::BTreeMap<String, String>) -> Vec<&'static str> {
    LOADER_VARS
        .iter()
        .copied()
        .filter(|v| !task_env.contains_key(*v))
        .collect()
}

/// The line the verifier's command starts with: unset `vars`, so neither the
/// verifier nor anything it starts inherits them.
pub fn verifier_env_prefix(vars: &[&str]) -> String {
    if vars.is_empty() {
        return String::new();
    }
    format!("unset {}\n", vars.join(" "))
}

/// Check the environment the verifier would get, run with the verifier's
/// own user, folder, and variables. Prints `BF venv-dropped <name>` for each
/// of `vars` the sandbox's environment held (they are unset next, as
/// [`verifier_env_prefix`] does), then, after the unset: `BF venv-set
/// <name>` for any still set, `BF venv-safepath <value>`,
/// `BF venv-path-bad <folder> <why>` for each `PATH` folder that is relative
/// or that someone other than root could write, `BF venv-path <PATH>`, and
/// `BF venv-done`.
pub fn verifier_env_check(vars: &[&str]) -> String {
    let list = vars.join(" ");
    format!(
        r#"for v in {list}; do eval "[ -n \"\${{$v+x}}\" ]" && echo "BF venv-dropped $v"; done
{prefix}for v in {list}; do eval "[ -n \"\${{$v+x}}\" ]" && echo "BF venv-set $v"; done
echo "BF venv-safepath ${{PYTHONSAFEPATH:-}}"
p=${{PATH:-}}
IFS=:
for d in $p; do
  case "$d" in /*) ;; *) echo "BF venv-path-bad '$d' relative"; continue;; esac
  [ -d "$d" ] || continue
  u=$(stat -Lc %u "$d" 2>/dev/null); m=$(stat -Lc %A "$d" 2>/dev/null)
  [ "$u" = 0 ] || echo "BF venv-path-bad $d owner-uid-$u"
  case "$m" in ?????w????|????????w?) echo "BF venv-path-bad $d $m";; esac
done
unset IFS
echo "BF venv-path $p"
echo "BF venv-done""#,
        prefix = verifier_env_prefix(vars),
    )
}

/// Caps on what output collection copies, from task.md's defaults: 100 MB
/// per file, 10,000 files, 32 folders deep, and 1 GiB in all (0.8's cap on
/// collected artifacts).
pub const MAX_OUTPUT_FILE_BYTES: u64 = 100 * 1024 * 1024;
pub const MAX_OUTPUT_FILES: u64 = 10_000;
pub const MAX_OUTPUT_DEPTH: u32 = 32;
pub const MAX_OUTPUT_TOTAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Where [`pack_outputs`] leaves its archive.
pub const OUTPUTS_TAR: &str = "/var/lib/benchflow/outputs.tar";

/// Pack the outputs at `paths` into [`OUTPUTS_TAR`] (paths relative to `/`),
/// taking custody of only what is safe to copy (an output reached through a
/// symlink another user made is skipped, as `BF skipped-special`): regular files up to
/// [`MAX_OUTPUT_FILE_BYTES`], at most [`MAX_OUTPUT_FILES`], at most
/// [`MAX_OUTPUT_DEPTH`] deep, and at most [`MAX_OUTPUT_TOTAL_BYTES`] in all.
/// Symlinks and special files are never copied (`BF skipped-special`), nor
/// are files over a size cap (`BF skipped-large`),
/// so a sparse 100 GB file or a FIFO cannot stall the copy. Paths with a
/// line break are left out: the list `tar` reads has one path per line. Prints
/// `BF packed <files> <bytes>`.
pub fn pack_outputs(paths: &[String]) -> String {
    pack_into(paths, OUTPUTS_TAR)
}

/// The agent's session folders, packed by [`pack_into`] at the end of its
/// turn.
pub const SESSIONS_TAR: &str = "/var/lib/benchflow/sessions.tar";

/// [`pack_outputs`], into the archive `tar`.
pub fn pack_into(paths: &[String], tar: &str) -> String {
    pack_capped(paths, tar, MAX_OUTPUT_TOTAL_BYTES)
}

/// [`pack_into`], with `total` bytes in all.
fn pack_capped(paths: &[String], tar: &str, total: u64) -> String {
    let list: Vec<String> = paths.iter().map(|p| q(p)).collect();
    format!(
        r#"{umask}
{links}
o={state}; mkdir -p "$o"; chmod 700 "$o"; l="$o/outputs.list"; : > "$l"; rm -f {tar}
nl='
'
for p in {list}; do
  [ -e "$p" ] || [ -L "$p" ] || {{ echo "BF missing $p"; continue; }}
  if [ -L "$p" ] || [ -n "$(foreign_links "$p")" ]; then echo "BF skipped-special $p"; continue; fi
  find "$p" -xdev -maxdepth {depth} \( -type l -o -type p -o -type s -o -type b -o -type c \) -print 2>/dev/null | while IFS= read -r f; do echo "BF skipped-special $f"; done
  find "$p" -xdev -maxdepth {depth} -type f -size +{kb}k -print 2>/dev/null | while IFS= read -r f; do echo "BF skipped-large $f"; done
  find "$p" -xdev -maxdepth {depth} -path "*$nl*" -prune -o -type f -size -{kb1}k -print 2>/dev/null >> "$l"
done
n=$(wc -l < "$l" | tr -d ' ')
if [ "$n" -gt {max_files} ]; then echo "BF skipped-count $((n - {max_files}))"; head -n {max_files} "$l" > "$l.cut"; mv "$l.cut" "$l"; fi
t=0; : > "$l.fit"
while IFS= read -r f; do
  s=$(stat -c %s "$f" 2>/dev/null || wc -c < "$f" 2>/dev/null | tr -d ' '); s=${{s:-0}}
  if [ $((t + s)) -gt {total} ]; then echo "BF skipped-large $f"; else t=$((t + s)); printf '%s\n' "$f" >> "$l.fit"; fi
done < "$l"
mv "$l.fit" "$l"
sed 's|^/||' "$l" > "$l.rel"
if [ -s "$l.rel" ]; then (cd / && tar -cf {tar} -T "$l.rel") 2>/dev/null || {{ echo "BF error tar"; exit 1; }}; else tar -cf {tar} -T /dev/null; fi
echo "BF packed $(wc -l < "$l.rel" | tr -d ' ') $(wc -c < {tar} | tr -d ' ')""#,
        umask = STATE_UMASK,
        links = FOREIGN_LINKS,
        state = q(STATE_DIR),
        tar = q(tar),
        list = list.join(" "),
        depth = MAX_OUTPUT_DEPTH,
        kb = MAX_OUTPUT_FILE_BYTES / 1024,
        kb1 = MAX_OUTPUT_FILE_BYTES / 1024 + 1,
        max_files = MAX_OUTPUT_FILES,
    )
}

/// Parsed `BF ...` lines from a hardening program.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    pub lines: Vec<(String, String)>,
}

impl Report {
    pub fn parse(output: &str) -> Self {
        let lines = output
            .lines()
            .filter_map(|l| l.strip_prefix("BF "))
            .map(|l| match l.split_once(' ') {
                Some((tag, rest)) => (tag.to_string(), rest.to_string()),
                None => (l.to_string(), String::new()),
            })
            .collect();
        Self { lines }
    }

    pub fn all(&self, tag: &str) -> Vec<&str> {
        self.lines
            .iter()
            .filter(|(t, _)| t == tag)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    pub fn first(&self, tag: &str) -> Option<&str> {
        self.all(tag).into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_path_checks_are_valid_sh_and_quiet_for_a_plain_user() {
        for script in [ROOT_PATHS_AS_USER.to_string(), root_paths_as_root("agent")] {
            let out = std::process::Command::new("sh")
                .arg("-n")
                .arg("-c")
                .arg(&script)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        // Nothing in them runs a command as root: sudo only lists.
        assert!(ROOT_PATHS_AS_USER.contains("sudo -n -l"));
        assert_eq!(ROOT_PATHS_AS_USER.matches("sudo -").count(), 1);
        assert!(root_paths_as_root("agent").contains("sudo -l -U"));
    }

    /// A stand-in `sudo` that lists a rule without a password, and a
    /// doas.conf, on a fake PATH: each way is reported.
    #[cfg(unix)]
    #[test]
    fn passwordless_sudo_is_reported_as_a_way_to_root() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let sudo = d.path().join("sudo");
        std::fs::write(
            &sudo,
            "#!/bin/sh\n[ \"$1 $2\" = \"-n -l\" ] || exit 9\nprintf 'Matching Defaults entries:\\n    env_reset\\n\\nUser agent may run the following commands:\\n    (ALL) NOPASSWD: ALL\\n'\n",
        )
        .unwrap();
        std::fs::set_permissions(&sudo, std::fs::Permissions::from_mode(0o755)).unwrap();
        // A `find` that finds nothing keeps the test off this machine's disk.
        let find = d.path().join("find");
        std::fs::write(&find, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&find, std::fs::Permissions::from_mode(0o755)).unwrap();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(ROOT_PATHS_AS_USER)
            .env("PATH", format!("{}:/usr/bin:/bin", d.path().display()))
            .output()
            .unwrap();
        let r = Report::parse(&String::from_utf8_lossy(&out.stdout));
        assert!(
            r.all("root-path")
                .contains(&"sudo runs commands for it without a password ((ALL) NOPASSWD: ALL)"),
            "{r:?}"
        );
        // A sudo that needs a password is not a way.
        std::fs::write(
            &sudo,
            "#!/bin/sh\necho 'sudo: a password is required' >&2\nexit 1\n",
        )
        .unwrap();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(ROOT_PATHS_AS_USER)
            .env("PATH", format!("{}:/usr/bin:/bin", d.path().display()))
            .output()
            .unwrap();
        let r = Report::parse(&String::from_utf8_lossy(&out.stdout));
        assert!(
            !r.all("root-path").iter().any(|l| l.starts_with("sudo")),
            "{r:?}"
        );
    }

    #[test]
    fn report_parses_tagged_lines() {
        let r =
            Report::parse("noise\nBF planted /app/conftest.py\nBF killed 3\nBF special /app/x\n");
        assert_eq!(r.all("planted"), vec!["/app/conftest.py"]);
        assert_eq!(r.first("killed"), Some("3"));
        assert_eq!(r.all("special"), vec!["/app/x"]);
        assert!(r.all("nothing").is_empty());
    }

    #[test]
    fn programs_quote_their_inputs() {
        let s = setup_user("agent", "/app dir");
        assert!(s.contains("w='/app dir'"));
        let h = handoff("/app", Some("tester"), true);
        assert!(h.contains("chown tester /logs/verifier"));
        assert!(!handoff("/app", Some("root"), true).contains("chown root /logs/verifier 2>"));
    }

    /// Only pytest and Python startup files count as planted or edited;
    /// project files are the agent's work. Guards the e2e-docker fix: on
    /// Terminal-Bench 2, build-cython-ext's own `pip install -e .` left
    /// `__editable__.pyknotid-0.5.3.pth` in /usr/local/lib, the handoff
    /// removed it, and the oracle failed; cloned Makefiles were removed too.
    #[test]
    fn only_test_runner_files_are_steering_files() {
        let dir = tempfile::tempdir().unwrap();
        let names = [
            "conftest.py",
            "pytest.ini",
            "sitecustomize.py",
            "usercustomize.py",
            "zz.pth",
            "Makefile",
            "package.json",
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            "jest.config.js",
        ];
        for n in names {
            std::fs::write(dir.path().join(n), "").unwrap();
        }
        let script = format!(
            "find {} {} -print | sed 's|.*/||' | LC_ALL=C sort",
            q(&dir.path().display().to_string()),
            steering_find_expr()
        );
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "conftest.py\npytest.ini\nsitecustomize.py\nusercustomize.py\nzz.pth\n"
        );
        let prune = prune_expr();
        for d in ["/usr/lib", "/usr/local/lib", "/etc", "/proc"] {
            assert!(
                prune.contains(&format!("-path {d} ")),
                "{d} not pruned: {prune}"
            );
        }
    }

    /// Run [`handoff`] on the host with every absolute path it touches moved
    /// under `root`, so nothing outside `root` is read or changed.
    fn handoff_under(root: &std::path::Path, workdir: &std::path::Path) -> String {
        handoff_with(root, workdir, true, true)
    }

    fn id(flag: &str) -> String {
        let out = std::process::Command::new("id").arg(flag).output().unwrap();
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    /// [`handoff_under`], strict or not; with `as_root`, the test's own user
    /// and group stand in for root's as the state folder's owners.
    fn handoff_with(
        root: &std::path::Path,
        workdir: &std::path::Path,
        strict: bool,
        as_root: bool,
    ) -> String {
        let r = root.to_str().unwrap();
        // Setup made the folder.
        let state = root.join("state");
        if std::fs::symlink_metadata(&state).is_err() {
            std::fs::create_dir_all(&state).unwrap();
        }
        let mut script = handoff(workdir.to_str().unwrap(), None, strict);
        if as_root {
            script = script
                .replace("! -user 0", &format!("! -user {}", id("-u")))
                .replace("! -group 0", &format!("! -group {}", id("-g")));
        }
        let script = script
            .replace("/logs/verifier", &format!("{r}/logs/verifier"))
            .replace(STATE_DIR, &format!("{r}/state"))
            .replace("find / -xdev", &format!("find {r} -xdev"))
            .replace("/tmp /var/tmp", &format!("{r}/tmp {r}/var/tmp"))
            .replace(
                &OUTPUT_DIRS.join(" "),
                &OUTPUT_DIRS.map(|d| format!("{r}{d}")).join(" "),
            );
        for left in [
            "find / ",
            " /tmp ",
            " /var/tmp ",
            " /output",
            " /logs/",
            STATE_DIR,
        ] {
            assert!(
                !script.contains(left),
                "the test would touch the host ({left:?})"
            );
        }
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .expect("sh");
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    /// Run [`baseline`] on the host under `root`, as [`handoff_under`] does.
    fn baseline_under(root: &std::path::Path) -> String {
        let r = root.to_str().unwrap();
        let script = baseline()
            .replace(STATE_DIR, &format!("{r}/state"))
            .replace("find / -xdev", &format!("find {r} -xdev"))
            .replace("/tmp /var/tmp", &format!("{r}/tmp {r}/var/tmp"));
        assert!(!script.contains("find / "), "the test would touch the host");
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .expect("sh");
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    /// pytest loads every `[pytest11]` entry point it finds, so a plugin the
    /// agent registers in a package folder the verifier's pytest reads
    /// steers grading as a `conftest.py` would (BenchFlow 0.8's plugin guard
    /// refuses them). Found comparing the engines on Terminal-Bench 2's
    /// `mailman`. Other entry points are the agent's installs and stay.
    #[cfg(unix)]
    #[test]
    fn handoff_drops_planted_pytest_plugins_and_keeps_other_entry_points() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        for d in ["tmp", "var/tmp", "logs", "state"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let app = root.join("app");
        let site = app.join(".venv/lib/python3.12/site-packages");
        let shipped = site.join("tool-1.0.dist-info");
        std::fs::create_dir_all(&shipped).unwrap();
        let shipped_eps = "[console_scripts]\ntool = tool:main\n";
        std::fs::write(shipped.join("entry_points.txt"), shipped_eps).unwrap();
        let out = baseline_under(&root);
        assert!(out.contains("BF baseline 1"), "{out}");

        // The agent's turn: a new pytest plugin, a new package with a
        // console script, and a pytest plugin added to the shipped package.
        let plugin = site.join("helper-0.1.dist-info");
        std::fs::create_dir_all(&plugin).unwrap();
        std::fs::write(
            plugin.join("entry_points.txt"),
            "[pytest11]\nhelper = helper\n",
        )
        .unwrap();
        let cli = site.join("cli-2.0.dist-info");
        std::fs::create_dir_all(&cli).unwrap();
        std::fs::write(
            cli.join("entry_points.txt"),
            "[console_scripts]\ncli = cli:main\n",
        )
        .unwrap();
        std::fs::write(
            shipped.join("entry_points.txt"),
            format!("{shipped_eps}\n[pytest11]\ntool = tool.plugin\n"),
        )
        .unwrap();

        let out = handoff_under(&root, &app);
        assert!(out.contains("BF handoff done"), "{out}");
        assert!(
            !plugin.join("entry_points.txt").exists(),
            "the planted pytest plugin survived: {out}"
        );
        assert!(
            cli.join("entry_points.txt").exists(),
            "a console-script entry point was removed: {out}"
        );
        assert_eq!(
            std::fs::read_to_string(shipped.join("entry_points.txt")).unwrap(),
            shipped_eps,
            "the edited entry points were not restored: {out}"
        );
    }

    /// Python's entry-point reader takes more than a `[pytest11]` line
    /// (`[[pytest11]]` loads the same plugin), and pytest and Python follow a
    /// link named like a steering file: each of these is planted too.
    #[cfg(unix)]
    #[test]
    fn handoff_drops_plugins_however_spelled_and_linked_steering_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        for d in ["tmp", "var/tmp", "logs", "state"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let app = root.join("app");
        let site = app.join(".venv/lib/python3.12/site-packages");
        std::fs::create_dir_all(&site).unwrap();
        let out = baseline_under(&root);
        assert!(out.contains("BF baseline 0"), "{out}");

        // The agent's turn: a plugin under a doubled header, a linked
        // entry_points.txt that registers one, and a conftest.py that is a
        // link to a module of its own.
        let doubled = site.join("a-0.1.dist-info");
        std::fs::create_dir_all(&doubled).unwrap();
        std::fs::write(doubled.join("entry_points.txt"), "[[pytest11]]\na = a\n").unwrap();
        let linked = site.join("b-0.1.dist-info");
        std::fs::create_dir_all(&linked).unwrap();
        std::fs::write(site.join("b.txt"), "[pytest11]\nb = b\n").unwrap();
        std::os::unix::fs::symlink(site.join("b.txt"), linked.join("entry_points.txt")).unwrap();
        std::fs::write(app.join("helper.py"), "x = 1\n").unwrap();
        std::os::unix::fs::symlink("helper.py", app.join("conftest.py")).unwrap();

        let out = handoff_under(&root, &app);
        assert!(out.contains("BF handoff done"), "{out}");
        for planted in [
            doubled.join("entry_points.txt"),
            linked.join("entry_points.txt"),
            app.join("conftest.py"),
        ] {
            assert!(
                std::fs::symlink_metadata(&planted).is_err(),
                "{} survived: {out}",
                planted.display()
            );
        }
        assert!(
            app.join("helper.py").exists(),
            "the link's target, the agent's own module, was removed: {out}"
        );
    }

    /// Under an exec umask of 0 (`docker exec` on the campaign's
    /// Docker-in-Docker daemon starts there), what setup's baseline and
    /// preload snapshot, the packed outputs, and a first handoff write into
    /// the state folder is still root's alone: a strict handoff after them
    /// (the next stage's, or a rootless agent's) runs to its end instead of
    /// stopping with R314.
    #[cfg(unix)]
    #[test]
    fn state_files_are_root_only_whatever_the_umask() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let r = root.to_str().unwrap();
        let app = root.join("app");
        for d in ["app", "tmp", "var/tmp", "logs"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::fs::write(app.join("conftest.py"), "x").unwrap();
        std::fs::write(app.join("out.txt"), "y").unwrap();
        // Each writer is an exec of its own in a trial, and each starts at
        // the backend's umask: one shell for all four would let the first
        // writer's umask cover the others.
        let mut out = String::new();
        for writer in [
            baseline(),
            preload_snapshot(),
            pack_outputs(&[format!("{r}/app/out.txt")]),
            handoff(app.to_str().unwrap(), None, false),
        ] {
            let writer = writer
                .replace("/logs/verifier", &format!("{r}/logs/verifier"))
                .replace(STATE_DIR, &format!("{r}/state"))
                .replace("find / -xdev", &format!("find {r}/app -xdev"))
                .replace("/tmp /var/tmp", &format!("{r}/tmp {r}/var/tmp"))
                .replace("p=/etc/ld.so.preload", &format!("p={r}/no-preload"))
                .replace(
                    &OUTPUT_DIRS.join(" "),
                    &OUTPUT_DIRS.map(|d| format!("{r}{d}")).join(" "),
                );
            assert!(!writer.contains(STATE_DIR));
            let o = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!("umask 000\n{writer}"))
                .output()
                .unwrap();
            out.push_str(&String::from_utf8_lossy(&o.stdout));
        }
        assert!(out.contains("BF baseline 1"), "{out}");
        assert!(out.contains("BF packed 1"), "{out}");
        assert!(out.trim_end().ends_with("BF handoff done"), "{out}");
        for f in [
            "baseline/list",
            "baseline/after",
            "preload.sum",
            "outputs.tar",
        ] {
            assert!(root.join("state").join(f).is_file(), "{f}: {out}");
        }
        // The strict check, with the test's user standing in for root.
        let out = handoff_with(&root, &app, true, true);
        assert!(!out.contains("BF state-owner"), "{out}");
        assert!(out.trim_end().ends_with("BF handoff done"), "{out}");
        // The system-folder lock keeps its scan in the state folder while it
        // runs: stopped there by its timeout, what it leaves is root's alone.
        assert!(lock_system_dirs("/w", "/h", "agent").starts_with(STATE_UMASK));
        // A later stage's additions to the baseline are written there too.
        assert!(baseline_add(&[]).starts_with(STATE_UMASK));
    }

    /// A restored `/etc/ld.so.preload` stays readable by every user, as the
    /// loader needs it to be, whatever the umask of the snapshot's exec.
    #[cfg(unix)]
    #[test]
    fn a_restored_preload_file_stays_readable() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let r = root.to_str().unwrap();
        let preload = root.join("ld.so.preload");
        std::fs::write(&preload, "/usr/lib/libshim.so\n").unwrap();
        std::fs::set_permissions(&preload, std::fs::Permissions::from_mode(0o644)).unwrap();
        let run = |script: String| {
            let script = script
                .replace(STATE_DIR, &format!("{r}/state"))
                .replace("p=/etc/ld.so.preload", &format!("p={r}/ld.so.preload"));
            let o = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!("umask 000\n{script}"))
                .output()
                .unwrap();
            String::from_utf8_lossy(&o.stdout).into_owned()
        };
        assert!(run(preload_snapshot()).contains("BF preload present"));
        std::fs::write(&preload, "/tmp/evil.so\n").unwrap();
        let out = run(preload_check());
        assert!(out.contains("BF preload restored"), "{out}");
        assert_eq!(
            std::fs::read_to_string(&preload).unwrap(),
            "/usr/lib/libshim.so\n"
        );
        let mode = std::fs::metadata(&preload).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o644, "{out}");
    }

    /// The state check on its own: a folder of the test user's (standing in
    /// for root's) with a baseline passes and says so; a link in place of
    /// the folder, or a file others may write, is named.
    #[cfg(unix)]
    #[test]
    fn the_state_check_names_what_is_not_roots_alone() {
        use std::os::unix::fs::PermissionsExt;
        let run = |root: &std::path::Path| {
            let script = state_check()
                .replace("! -user 0", &format!("! -user {}", id("-u")))
                .replace("! -group 0", &format!("! -group {}", id("-g")))
                .replace(STATE_DIR, root.join("state").to_str().unwrap());
            assert!(!script.contains(STATE_DIR));
            let out = std::process::Command::new("sh")
                .arg("-c")
                .arg(&script)
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("state/baseline")).unwrap();
        std::fs::write(root.join("state/baseline/list"), "").unwrap();
        let out = run(&root);
        assert_eq!(out, "BF state-baseline\nBF state-checked\n");
        let list = root.join("state/baseline/list");
        std::fs::set_permissions(&list, std::fs::Permissions::from_mode(0o666)).unwrap();
        let out = run(&root);
        assert!(
            out.contains(&format!("BF state-owner {}\n", list.display())),
            "{out}"
        );
        assert!(out.ends_with("BF state-checked\n"), "{out}");
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("elsewhere/baseline")).unwrap();
        std::fs::write(root.join("elsewhere/baseline/list"), "").unwrap();
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("state")).unwrap();
        let out = run(&root);
        assert!(out.starts_with("BF state-owner "), "{out}");
        assert!(!out.contains("BF state-baseline"), "{out}");
    }

    /// A path with a line break in it must not split into two paths for the
    /// root programs: one of the halves could name any file, such as the
    /// baseline that planted-file cleanup reads.
    #[cfg(unix)]
    #[test]
    fn handoff_is_not_steered_by_line_breaks_in_names() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let baseline = root.join("state/baseline");
        for d in ["app/site", "tmp", "var/tmp", "logs"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::fs::create_dir_all(&baseline).unwrap();
        std::fs::write(baseline.join("list"), "").unwrap();
        std::fs::write(baseline.join("sums"), "").unwrap();
        std::fs::write(baseline.join("tmp-sums"), "").unwrap();
        let app = root.join("app");
        // The agent plants a conftest.py, then tries to make the handoff
        // delete the baseline so the cleanup is skipped: a symlink at
        // `<app>/x\n<baseline>/list` reads as two lines, the second the
        // baseline's own list.
        std::fs::write(app.join("conftest.py"), "forced pass").unwrap();
        let trick = app.join(format!("x\n{}", baseline.display()));
        std::fs::create_dir_all(&trick).unwrap();
        std::os::unix::fs::symlink("/nonexistent", trick.join("list")).unwrap();
        // And a .pth file whose name splits into two paths that do not exist.
        std::fs::write(app.join("site/evil\nx.pth"), "import os").unwrap();

        let out = handoff_under(&root, &app);
        assert!(
            baseline.join("list").is_file(),
            "the baseline was deleted: {out}"
        );
        assert!(
            !app.join("conftest.py").exists(),
            "the planted conftest.py survived: {out}"
        );
        assert!(
            !app.join("site/evil\nx.pth").exists(),
            "the planted .pth survived: {out}"
        );
        assert!(!app.join("x\n").exists(), "{out}");
    }

    /// A state folder the agent got hold of (here: replaced by a link) stops
    /// the handoff before it reads the baseline, so nothing is restored from
    /// files the agent could have chosen, and the runner refuses to grade.
    #[cfg(unix)]
    #[test]
    fn handoff_stops_when_the_state_folder_is_not_roots_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let app = root.join("app");
        for d in ["app", "tmp", "var/tmp", "logs", "elsewhere/baseline"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        // A baseline that lists no conftest.py: had the handoff gone on, it
        // would have removed the planted one.
        for f in ["list", "sums", "tmp-sums"] {
            std::fs::write(root.join("elsewhere/baseline").join(f), "").unwrap();
        }
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("state")).unwrap();
        std::fs::write(app.join("conftest.py"), "planted").unwrap();
        let out = handoff_under(&root, &app);
        assert!(out.contains("BF state-owner"), "{out}");
        assert!(out.contains("BF handoff stopped"), "{out}");
        assert!(!out.contains("BF handoff done"), "{out}");
        assert!(
            app.join("conftest.py").exists(),
            "the handoff went on: {out}"
        );
    }

    /// A file in the state folder anyone may write, or a link there, stops
    /// the handoff as well; the copy of a linked `/etc/ld.so.preload` is the
    /// one link BenchFlow makes there itself.
    #[cfg(unix)]
    #[test]
    fn handoff_stops_on_a_state_file_others_may_write_or_a_link() {
        use std::os::unix::fs::PermissionsExt;
        let setup = |name: &str| {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path().canonicalize().unwrap();
            for d in ["app", "tmp", "var/tmp", "logs", "state/baseline"] {
                std::fs::create_dir_all(root.join(d)).unwrap();
            }
            std::fs::write(root.join("app/conftest.py"), name).unwrap();
            (tmp, root)
        };
        // Writable by others.
        let (_t, root) = setup("open");
        let list = root.join("state/baseline/list");
        std::fs::write(&list, "").unwrap();
        std::fs::set_permissions(&list, std::fs::Permissions::from_mode(0o666)).unwrap();
        let out = handoff_under(&root, &root.join("app"));
        assert!(
            out.contains(&format!("BF state-owner {}\n", list.display())),
            "{out}"
        );
        assert!(out.trim_end().ends_with("BF handoff stopped"), "{out}");
        // A link.
        let (_t, root) = setup("link");
        std::os::unix::fs::symlink("/etc/hosts", root.join("state/baseline/list")).unwrap();
        let out = handoff_under(&root, &root.join("app"));
        assert!(out.trim_end().ends_with("BF handoff stopped"), "{out}");
        // The preload copy may be a link.
        let (_t, root) = setup("preload");
        std::os::unix::fs::symlink("/etc/hosts", root.join("state/preload")).unwrap();
        let out = handoff_under(&root, &root.join("app"));
        assert!(!out.contains("BF state-owner"), "{out}");
        assert!(out.trim_end().ends_with("BF handoff done"), "{out}");
    }

    /// A file in the state folder that is not root's stops the handoff
    /// (here every file is the test user's, and the check is root's).
    #[cfg(unix)]
    #[test]
    fn handoff_stops_on_a_state_file_that_is_not_roots() {
        if id("-u") == "0" {
            return; // root's own files are root's
        }
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        for d in ["app", "tmp", "var/tmp", "logs", "state/baseline"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::fs::write(root.join("state/baseline/list"), "").unwrap();
        let out = handoff_with(&root, &root.join("app"), true, false);
        assert!(out.contains("BF state-owner "), "{out}");
        assert!(out.trim_end().ends_with("BF handoff stopped"), "{out}");
    }

    /// When whoever worked could act as root, a change in the state folder
    /// is noted and the handoff goes on.
    #[cfg(unix)]
    #[test]
    fn a_lenient_handoff_notes_the_state_folder_and_goes_on() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        for d in ["app", "tmp", "var/tmp", "logs", "state/baseline"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let list = root.join("state/baseline/list");
        std::fs::write(&list, "").unwrap();
        std::fs::set_permissions(&list, std::fs::Permissions::from_mode(0o666)).unwrap();
        let out = handoff_with(&root, &root.join("app"), false, true);
        assert!(
            out.contains(&format!("BF state-noted {}\n", list.display())),
            "{out}"
        );
        assert!(!out.contains("BF state-owner"), "{out}");
        assert!(out.trim_end().ends_with("BF handoff done"), "{out}");
    }

    /// A scan of the state folder that fails stops the handoff as
    /// unchecked; what it found before failing still counts as a change.
    #[cfg(unix)]
    #[test]
    fn a_failed_state_scan_stops_the_handoff() {
        use std::os::unix::fs::PermissionsExt;
        let uid = std::process::Command::new("id").arg("-u").output().unwrap();
        if String::from_utf8_lossy(&uid.stdout).trim() == "0" {
            return; // root reads a folder whatever its mode
        }
        for found in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path().canonicalize().unwrap();
            for d in [
                "app",
                "tmp",
                "var/tmp",
                "logs",
                "state/baseline",
                "state/sealed",
            ] {
                std::fs::create_dir_all(root.join(d)).unwrap();
            }
            let open = root.join("state/baseline/list");
            std::fs::write(&open, "").unwrap();
            if found {
                std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o666)).unwrap();
            }
            let sealed = root.join("state/sealed");
            std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o000)).unwrap();
            let out = handoff_under(&root, &root.join("app"));
            std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o700)).unwrap();
            assert!(out.trim_end().ends_with("BF handoff stopped"), "{out}");
            assert_eq!(out.contains("BF state-owner"), found, "{out}");
            assert_eq!(out.contains("BF state-unchecked"), !found, "{out}");
        }
    }

    /// A stage's own files join the baseline from the sums given (computed
    /// on the host), replacing what the baseline had for the same path; the
    /// handoff then keeps them, and puts back an edited one from BenchFlow's
    /// own copy rather than from the first baseline's archive. A file the
    /// opening stage planted is not added, so the handoff removes it. Guards
    /// the D7b fix of the stage boundary (part 2 of its brief), where a
    /// later stage took the baseline again while the agent still ran.
    #[cfg(unix)]
    #[test]
    fn a_stages_own_files_join_the_baseline_and_are_restored_from_its_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let app = root.join("app");
        for d in ["app", "tmp", "var/tmp", "logs", "state/baseline"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let sum = |b: &[u8]| {
            use sha2::Digest;
            hex::encode(sha2::Sha256::digest(b))
        };
        let shipped = app.join("conftest.py");
        let other = app.join("pytest.ini");
        let b = root.join("state/baseline");
        // The first baseline knew an older conftest.py and a pytest.ini.
        std::fs::write(
            b.join("list"),
            format!("{}\n{}\n", shipped.display(), other.display()),
        )
        .unwrap();
        std::fs::write(&other, "[pytest]\n").unwrap();
        std::fs::write(
            b.join("sums"),
            format!(
                "{}  {}\n{}  {}\n",
                sum(b"OLD = 1\n"),
                shipped.display(),
                sum(b"[pytest]\n"),
                other.display()
            ),
        )
        .unwrap();
        // The stage ships a new one; BenchFlow keeps its copy.
        let copy = root.join(format!("state/baseline/added{}", shipped.display()));
        std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
        std::fs::write(&copy, "SHIPPED = 2\n").unwrap();
        let r = root.to_str().unwrap();
        let add = baseline_add(&[(shipped.to_str().unwrap().to_string(), sum(b"SHIPPED = 2\n"))])
            .replace(STATE_DIR, &format!("{r}/state"));
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&add)
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "BF baseline-added 1\n"
        );
        let list = std::fs::read_to_string(b.join("list")).unwrap();
        assert_eq!(list.lines().count(), 2, "{list}");
        let sums = std::fs::read_to_string(b.join("sums")).unwrap();
        assert!(
            sums.contains(&format!(
                "{}  {}\n",
                sum(b"SHIPPED = 2\n"),
                shipped.display()
            )),
            "{sums}"
        );
        assert!(!sums.contains(&sum(b"OLD = 1\n")), "{sums}");
        assert!(sums.contains(&sum(b"[pytest]\n")), "{sums}");
        // The agent edited the stage's file and planted another.
        std::fs::write(&shipped, "EDITED = 3\n").unwrap();
        let planted = app.join("sub/conftest.py");
        std::fs::create_dir_all(planted.parent().unwrap()).unwrap();
        std::fs::write(&planted, "PLANTED\n").unwrap();
        let out = handoff_under(&root, &app);
        assert!(
            out.contains(&format!("BF restored {}\n", shipped.display())),
            "{out}"
        );
        assert!(
            out.contains(&format!("BF planted {}\n", planted.display())),
            "{out}"
        );
        assert_eq!(std::fs::read_to_string(&shipped).unwrap(), "SHIPPED = 2\n");
        assert!(!planted.exists());
        assert_eq!(std::fs::read_to_string(&other).unwrap(), "[pytest]\n");
        // Nothing to add to: said so, and nothing made.
        let none = baseline_add(&[]).replace(STATE_DIR, &format!("{r}/nowhere"));
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&none)
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "BF baseline-missing\n"
        );
    }

    /// The handoff never removes or restores a test-runner file through a
    /// link that leads out of the workdir: the agent can replace a folder
    /// of its own outside the workdir (a later stage's mount it copied in)
    /// with a link to `/etc`, and root would then write the restored file
    /// there. A link that stays inside the workdir is fine. Guards the D7b
    /// fix of REVIEW-SEC H1 on b875d9d4.
    #[cfg(unix)]
    #[test]
    fn a_restore_never_goes_through_a_link_out_of_the_workdir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let app = root.join("app");
        for d in [
            "app/real",
            "tmp",
            "var/tmp",
            "logs",
            "state/baseline",
            "etcx",
            "srv",
        ] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::os::unix::fs::symlink(root.join("etcx"), root.join("srv/data")).unwrap();
        std::os::unix::fs::symlink(app.join("real"), app.join("sub")).unwrap();
        let sum = |b: &[u8]| {
            use sha2::Digest;
            hex::encode(sha2::Sha256::digest(b))
        };
        let outside = root.join("srv/data/conftest.py");
        let inside = app.join("sub/conftest.py");
        let b = root.join("state/baseline");
        std::fs::write(
            b.join("list"),
            format!("{}\n{}\n", inside.display(), outside.display()),
        )
        .unwrap();
        std::fs::write(
            b.join("sums"),
            format!(
                "{s}  {}\n{s}  {}\n",
                inside.display(),
                outside.display(),
                s = sum(b"SHIPPED\n")
            ),
        )
        .unwrap();
        for p in [&inside, &outside] {
            let copy = root.join(format!("state/baseline/added{}", p.display()));
            std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
            std::fs::write(&copy, "SHIPPED\n").unwrap();
        }
        std::fs::write(app.join("real/conftest.py"), "EDITED\n").unwrap();
        let out = handoff_under(&root, &app);
        assert!(
            out.contains(&format!("BF lost {}\n", outside.display())),
            "{out}"
        );
        assert!(!root.join("etcx/conftest.py").exists(), "{out}");
        assert!(
            out.contains(&format!("BF restored {}\n", inside.display())),
            "{out}"
        );
        assert_eq!(
            std::fs::read_to_string(app.join("real/conftest.py")).unwrap(),
            "SHIPPED\n"
        );
    }

    /// The folders above a mount a running agent could make an entry in:
    /// its own, ones others may write (sticky or not), ones its group may
    /// write; not another user's it cannot write (a web server's), as
    /// `open_parents` would have it. Run on this machine with a stand-in
    /// `stat -Lc` (the host's may not take `-c`). Guards the D7b fix of
    /// REVIEW-CORRECT M2.
    #[cfg(unix)]
    #[test]
    fn the_folders_a_running_agent_can_write_are_its_own_and_open_ones() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(
            bin.join("stat"),
            "#!/bin/sh\nfmt=; for a; do case \"$a\" in -Lc) ;; %*) fmt=$a;; *) f=$a;; esac; done\n\
             l=$(ls -lnd \"$f\")\n\
             case \"$fmt\" in %u) echo \"$l\" | awk '{print $3}';; %g) echo \"$l\" | awk '{print $4}';; \
             %A) echo \"$l\" | awk '{print substr($1,1,10)}';; esac\n",
        )
        .unwrap();
        std::fs::set_permissions(
            bin.join("stat"),
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();
        let me = whoami();
        let run = |path: &str| {
            let out = std::process::Command::new("sh")
                .arg("-c")
                .arg(agent_writable_parents(path, &me))
                .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
                .output()
                .unwrap();
            String::from_utf8(out.stdout).unwrap()
        };
        // A folder of its own.
        let mine = root.join("mine");
        std::fs::create_dir_all(&mine).unwrap();
        let out = run(&format!("{}/data", mine.display()));
        assert!(
            out.contains(&format!("BF open {}\n", mine.display())),
            "{out}"
        );
        // Root's folders it cannot write (here: the system's) are not open.
        // Some CI runners give the test user /usr/share, or run as root;
        // there these two checks say nothing, so they are skipped.
        let system_closed = {
            use std::os::unix::fs::MetadataExt;
            let mine_uid = std::fs::metadata(&root).unwrap().uid();
            std::fs::metadata("/usr/share")
                .is_ok_and(|m| m.uid() == 0 && m.mode() & 0o022 == 0 && mine_uid != 0)
        };
        if system_closed {
            let out = run("/usr/share/benchflow-none/data");
            assert!(!out.contains("BF open /usr"), "{out}");
        }
        // A sticky folder others may write is.
        if std::path::Path::new("/private/tmp").is_dir() || std::path::Path::new("/tmp").is_dir() {
            let t = std::fs::canonicalize("/tmp").unwrap();
            let out = run(&format!("{}/benchflow-none/data", t.display()));
            assert!(out.contains(&format!("BF open {}\n", t.display())), "{out}");
        }
        // A target that is already a folder it cannot write (root's): said
        // so (REVIEW-OUTCOMES M2); one of its own is not.
        if system_closed {
            let out = run("/usr/share");
            assert!(out.contains("BF target-closed"), "{out}");
        }
        let out = run(mine.to_str().unwrap());
        assert!(!out.contains("BF target-closed"), "{out}");
        // An unknown user: said so.
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(agent_writable_parents("/x/y", "no-such-user-bf"))
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout), "BF open ?\n");
    }

    /// A workdir that is a link in a folder others may write is not
    /// followed (an agent could have made it), and is noted; an image
    /// without `find` stops the change rather than letting grading run
    /// against the agent's files. Guards the D7b fixes of REVIEW-CORRECT
    /// L3 and L4.
    #[cfg(unix)]
    #[test]
    fn own_workdir_follows_no_link_another_user_could_make_and_needs_find() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("real/inside")).unwrap();
        std::os::unix::fs::symlink(root.join("real"), root.join("app")).unwrap();
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let log = root.join("log");
        std::fs::write(
            bin.join("chown"),
            format!(
                "#!/bin/sh\nfor a; do echo \"$a\"; done >> {}\n",
                q(log.to_str().unwrap())
            ),
        )
        .unwrap();
        std::fs::set_permissions(
            bin.join("chown"),
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();
        let me = whoami();
        let app = root.join("app");
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(own_workdir(app.to_str().unwrap(), &me, &me))
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("so it was not followed"), "{stdout}");
        let changed = std::fs::read_to_string(&log).unwrap_or_default();
        assert!(!changed.contains("inside"), "{changed}");
        // No find on PATH: refused before anything changes.
        let nofind = root.join("nofind");
        std::fs::create_dir_all(&nofind).unwrap();
        for tool in [
            "id", "readlink", "dirname", "chown", "stat", "grep", "head", "tr",
        ] {
            for d in ["/usr/bin", "/bin", "/usr/sbin", "/sbin"] {
                let p = std::path::Path::new(d).join(tool);
                if p.exists() {
                    let _ = std::os::unix::fs::symlink(&p, nofind.join(tool));
                    break;
                }
            }
        }
        let out = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(own_workdir(root.join("real").to_str().unwrap(), &me, &me))
            .env("PATH", &nofind)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(3), "{out:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("BF error no-find"));
    }

    /// The names the baseline's scan looks for, matched on this machine.
    #[test]
    fn steering_names_match_as_the_scan_does() {
        for n in [
            "conftest.py",
            "pytest.ini",
            "sitecustomize.py",
            "usercustomize.py",
            "a.pth",
            ".pth",
        ] {
            assert!(is_steering_name(n), "{n}");
        }
        for n in ["conftest.pyc", "my_conftest.py", "pytest.ini.bak"] {
            assert!(!is_steering_name(n), "{n}");
        }
    }

    /// A link's target, which the agent chose, is printed on one line with
    /// any line break shown as `?`: a target name cannot print a `BF` line
    /// of its own, such as a forged end of the handoff.
    #[cfg(unix)]
    #[test]
    fn a_link_target_cannot_print_a_line_of_its_own() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        for d in ["app", "tmp", "var/tmp", "logs", "elsewhere"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let target = root.join("elsewhere/x\nBF handoff done");
        std::fs::write(&target, "").unwrap();
        std::os::unix::fs::symlink(&target, root.join("app/l")).unwrap();
        let out = handoff_under(&root, &root.join("app"));
        let forged = format!("{}?BF handoff done", root.join("elsewhere/x").display());
        assert!(out.contains(&forged), "{out}");
        assert_eq!(
            out.lines().filter(|l| *l == "BF handoff done").count(),
            1,
            "{out}"
        );
        assert!(!root.join("app/l").exists());
    }

    /// The agent user owns `/output` and `/outputs` as it owns the workdir,
    /// so the handoff takes the same custody of special files and escaping
    /// symlinks there.
    #[cfg(unix)]
    #[test]
    fn handoff_takes_custody_of_the_output_folders_too() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let baseline = root.join("state/baseline");
        for d in ["app", "tmp", "var/tmp", "logs", "output"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::fs::create_dir_all(&baseline).unwrap();
        std::fs::write(root.join("output/answer.json"), "{}").unwrap();
        std::os::unix::fs::symlink(root.join("state"), root.join("output/escape")).unwrap();
        std::os::unix::fs::symlink(root.join("output/answer.json"), root.join("output/same"))
            .unwrap();
        let fifo = std::process::Command::new("mkfifo")
            .arg(root.join("output/pipe"))
            .status()
            .unwrap();
        assert!(fifo.success());
        let out = handoff_under(&root, &root.join("app"));
        assert!(out.contains("BF handoff done"), "{out}");
        let gone = |n: &str| std::fs::symlink_metadata(root.join("output").join(n)).is_err();
        assert!(gone("escape"), "{out}");
        assert!(gone("pipe"), "{out}");
        assert!(!gone("same"), "a link inside the folder stays: {out}");
        assert!(!gone("answer.json"), "{out}");
    }

    fn whoami() -> String {
        let me = std::process::Command::new("id")
            .arg("-un")
            .output()
            .unwrap();
        String::from_utf8(me.stdout).unwrap().trim().to_string()
    }

    /// The workdir goes to the owner named, on its own filesystem, links
    /// changed themselves and never followed; `/` and unknown users are
    /// refused before anything changes.
    #[cfg(unix)]
    #[test]
    fn own_workdir_gives_the_folder_to_its_owner_and_refuses_the_root_folder() {
        let s = own_workdir("/root", "root", "agent");
        assert!(s.contains("-xdev"), "{s}");
        assert!(s.contains(r#"chown -h "$o:$g""#), "{s}");
        let tmp = tempfile::tempdir().unwrap();
        let w = tmp.path().join("work");
        std::fs::create_dir_all(w.join("repo/.git")).unwrap();
        std::fs::write(w.join("repo/file"), "x").unwrap();
        std::os::unix::fs::symlink(tmp.path().join("outside"), w.join("link")).unwrap();
        let me = whoami();
        let sh = |script: String| {
            std::process::Command::new("sh")
                .arg("-c")
                .arg(script)
                .output()
                .unwrap()
        };
        // Owning what it already owns is all a test can do without root.
        let out = sh(own_workdir(w.to_str().unwrap(), &me, &me));
        assert!(out.status.success());
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("BF root-owned\nBF owned {me}\n")
        );
        assert!(!tmp.path().join("outside").exists(), "a link was followed");
        let to_root = tmp.path().join("to-root");
        std::os::unix::fs::symlink("/", &to_root).unwrap();
        for (workdir, owner) in [
            ("/", me.as_str()),
            ("", me.as_str()),
            (to_root.to_str().unwrap(), me.as_str()),
            (w.to_str().unwrap(), "no-such-user-bf"),
        ] {
            let out = sh(own_workdir(workdir, owner, &me));
            assert_eq!(out.status.code(), Some(2), "{workdir:?} {owner}");
            assert!(!String::from_utf8_lossy(&out.stdout).contains("BF owned"));
        }
    }

    /// A folder chain nested past the system's path limit (an agent can
    /// make one with `mkdir x && cd x` in a loop) changes owner without an
    /// error: the walk runs `chown` from inside each folder. Run with the
    /// host's `find` and `chown` on the test user's own files. The same tree
    /// with full paths (`-exec`, as b7d315aa had it) fails, which shows the
    /// tree is deep enough. Guards the D7b fix of FIX-REVIEW-A 1, where such
    /// a tree turned the agent's trial into an ungraded sandbox error.
    #[cfg(unix)]
    #[test]
    fn own_workdir_walks_a_tree_nested_past_the_path_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let w = tmp.path().canonicalize().unwrap().join("work");
        std::fs::create_dir_all(&w).unwrap();
        let name = "d".repeat(200);
        let made = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!(
                // `cd -P`: dash resolves a logical `cd` to the whole path,
                // which fails past PATH_MAX (4096 on Linux).
                "cd {} && i=0; while [ $i -lt 30 ]; do mkdir {name} && cd -P {name} || exit 1; i=$((i+1)); done; echo x > f",
                q(w.to_str().unwrap())
            ))
            .output()
            .unwrap();
        assert!(made.status.success(), "{made:?}");
        let me = whoami();
        let run = |script: String| {
            let out = std::process::Command::new("sh")
                .arg("-c")
                .arg(script)
                .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        let script = own_workdir(w.to_str().unwrap(), &me, &me);
        assert_eq!(
            run(script.clone()),
            format!("BF root-owned\nBF owned {me}\n")
        );
        let full_paths = run(script.replace("x=-execdir", "x=-exec"));
        assert!(full_paths.contains("BF unchanged "), "{full_paths}");
        assert!(
            full_paths.ends_with(&format!("BF owned {me}\n")),
            "{full_paths}"
        );
    }

    /// Hard links keep their owner unless the agent user owns them; links
    /// change themselves; only a failure on the workdir itself is fatal (exit
    /// 3), one below it is a `BF unchanged` note, and a file that went away while
    /// the walk ran is no error. Guards the D7b fixes of FIX-REVIEW-A 1 and 2
    /// and FIX-REVIEW-B 3 on b7d315aa.
    #[test]
    fn own_workdir_leaves_others_hard_links_alone_and_notes_what_kept_its_owner() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let w = root.join("work");
        std::fs::create_dir_all(w.join("sub")).unwrap();
        std::fs::write(w.join("sub/mine"), "x").unwrap();
        // A file outside the workdir linked into it.
        std::fs::write(root.join("passwd"), "root:x:0:0").unwrap();
        std::fs::hard_link(root.join("passwd"), w.join("sub/linked")).unwrap();
        std::fs::write(root.join("target"), "t").unwrap();
        std::os::unix::fs::symlink(root.join("target"), w.join("sym")).unwrap();
        let bin = root.join("bin");
        std::fs::create_dir(&bin).unwrap();
        let log = root.join("chown.log");
        // A stand-in chown that logs each path it is given (made absolute
        // when `find` runs it from inside a folder), and fails as `mode`
        // says: on the workdir itself, below it, or as a file gone.
        let stub = |mode: &str| {
            let _ = std::fs::remove_file(&log);
            std::fs::write(
                bin.join("chown"),
                format!(
                    "#!/bin/sh\nw={w}\nfor a; do case \"$a\" in -*|*:*) ;; /*) echo \"$a\";; *) echo \"$(pwd -P)/${{a#./}}\";; esac; done >> {log}\n\
                     for a; do case \"$a\" in -*|*:*) continue;; esac\n\
                       case {mode} in\n\
                         root) [ \"$a\" = \"$w\" ] && {{ echo \"chown: $a: Read-only file system\" >&2; exit 1; }};;\n\
                         below) [ \"$a\" = \"$w\" ] || {{ echo \"chown: $a: Read-only file system\" >&2; exit 1; }};;\n\
                         gone) [ \"$a\" = \"$w\" ] || {{ echo \"chown: $a: No such file or directory\" >&2; exit 1; }};;\n\
                       esac\n\
                     done\nexit 0\n",
                    w = q(w.to_str().unwrap()),
                    log = q(&log.display().to_string()),
                    mode = mode,
                ),
            )
            .unwrap();
            std::fs::set_permissions(
                bin.join("chown"),
                std::os::unix::fs::PermissionsExt::from_mode(0o755),
            )
            .unwrap();
        };
        let me = whoami();
        let run = |agent: &str| {
            std::process::Command::new("sh")
                .arg("-c")
                .arg(own_workdir(w.to_str().unwrap(), &me, agent))
                .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
                .output()
                .unwrap()
        };
        let changed = || std::fs::read_to_string(&log).unwrap_or_default();
        let has = |log: &str, p: &std::path::Path| log.lines().any(|l| l == p.to_str().unwrap());

        // Someone else's hard link (the agent is another user) keeps its owner.
        stub("ok");
        let out = run("no-such-user-bf");
        assert!(out.status.success(), "{out:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("BF root-owned\nBF owned {me}\n")
        );
        let log1 = changed();
        assert!(has(&log1, &w), "{log1}");
        assert!(has(&log1, &w.join("sub/mine")), "{log1}");
        assert!(has(&log1, &w.join("sub")), "{log1}");
        assert!(
            has(&log1, &w.join("sym")),
            "the link itself changes: {log1}"
        );
        assert!(!has(&log1, &root.join("target")), "{log1}");
        assert!(
            !has(&log1, &w.join("sub/linked")),
            "a hard link's file may be outside the workdir: {log1}"
        );
        // The agent user's own hard link is given away (FIX-REVIEW-A 2).
        stub("ok");
        assert!(run(&me).status.success());
        assert!(has(&changed(), &w.join("sub/linked")), "{}", changed());

        // The workdir itself refused: fatal.
        stub("root");
        let out = run(&me);
        assert_eq!(out.status.code(), Some(3), "{out:?}");
        assert!(!String::from_utf8_lossy(&out.stdout).contains("BF owned"));
        assert!(String::from_utf8_lossy(&out.stderr).contains("BF error chown"));
        // A path below it refused: noted, and the workdir is still owned.
        stub("below");
        let out = run(&me);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{out:?}");
        assert!(stdout.contains("BF unchanged chown: "), "{stdout}");
        assert!(stdout.contains("Read-only file system"), "{stdout}");
        assert!(stdout.ends_with(&format!("BF owned {me}\n")), "{stdout}");
        // A file gone during the walk: no note.
        stub("gone");
        let out = run(&me);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("BF root-owned\nBF owned {me}\n")
        );
    }

    /// Run `script` under host `sh` with a stand-in `stat` that reads the
    /// owner's uid from `ls -ln` (the host's `stat` may not take `-c`).
    fn with_stand_in_stat(root: &std::path::Path, script: &str) -> String {
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(
            bin.join("stat"),
            "#!/bin/sh\nfor a; do f=$a; done\nls -lnd \"$f\" | awk '{print $3}'\n",
        )
        .unwrap();
        std::fs::set_permissions(
            bin.join("stat"),
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    /// Before root writes into a path after the agent's turn, symlinks on
    /// the way that another user made are removed, and with `foreign` so is
    /// anything another user left there. (The test runs as a user other
    /// than root, so every file here counts as another user's.)
    #[cfg(unix)]
    #[test]
    fn claim_removes_other_users_links_and_files_on_the_way() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("app")).unwrap();
        std::fs::create_dir_all(root.join("elsewhere")).unwrap();
        std::fs::write(root.join("elsewhere/keep.txt"), "x").unwrap();
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("app/tests")).unwrap();
        let target = root.join("app/tests/inner");
        let out = with_stand_in_stat(&root, &claim(target.to_str().unwrap(), false));
        assert!(out.contains("BF symlink "), "{out}");
        assert!(
            std::fs::symlink_metadata(root.join("app/tests")).is_err(),
            "{out}"
        );
        assert!(
            root.join("elsewhere/keep.txt").is_file(),
            "the link's target stays"
        );

        std::fs::create_dir_all(root.join("app/v")).unwrap();
        std::fs::write(root.join("app/v/test_extra.py"), "x").unwrap();
        let v = root.join("app/v");
        let out = with_stand_in_stat(&root, &claim(v.to_str().unwrap(), false));
        assert!(v.join("test_extra.py").is_file(), "{out}");
        let out = with_stand_in_stat(&root, &claim(v.to_str().unwrap(), true));
        assert!(out.contains("BF planted "), "{out}");
        assert!(!v.exists(), "{out}");
    }

    /// A line break in an output's path must not let the agent add a file
    /// from elsewhere (a root-only one, say) to the saved outputs.
    #[cfg(unix)]
    #[test]
    fn pack_outputs_takes_no_paths_from_line_breaks_in_names() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let r = root.to_str().unwrap();
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(app.join("ok.txt"), "mine").unwrap();
        std::fs::write(root.join("secret.txt"), "root only").unwrap();
        let trick = app.join(format!("x\n{r}"));
        std::fs::create_dir_all(&trick).unwrap();
        std::fs::write(trick.join("secret.txt"), "decoy").unwrap();
        let script = pack_outputs(&[app.to_str().unwrap().to_string()])
            .replace(STATE_DIR, &format!("{r}/state"));
        assert!(!script.contains(STATE_DIR));
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let mut ar =
            tar::Archive::new(std::fs::File::open(root.join("state/outputs.tar")).unwrap());
        let names: Vec<String> = ar
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        let secret = format!("{}/secret.txt", r.trim_start_matches('/'));
        assert!(!names.contains(&secret), "{names:?}");
        assert!(names.iter().any(|n| n.ends_with("app/ok.txt")), "{names:?}");
    }

    /// Outputs stop at the total cap: files past it are left out as large.
    #[cfg(unix)]
    #[test]
    fn pack_outputs_stops_at_the_total_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let r = root.to_str().unwrap();
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        for name in ["a.txt", "b.txt", "c.txt"] {
            std::fs::write(app.join(name), "0123456789").unwrap();
        }
        let script = pack_capped(&[app.to_str().unwrap().to_string()], OUTPUTS_TAR, 25)
            .replace(STATE_DIR, &format!("{r}/state"));
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let report = Report::parse(&String::from_utf8_lossy(&out.stdout));
        assert_eq!(report.all("skipped-large").len(), 1, "{report:?}");
        assert!(
            report.first("packed").unwrap().starts_with("2 "),
            "{report:?}"
        );
        let mut ar =
            tar::Archive::new(std::fs::File::open(root.join("state/outputs.tar")).unwrap());
        // macOS's tar adds `._` entries for extended attributes.
        let files: Vec<String> = ar
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".txt") && !n.rsplit('/').next().unwrap().starts_with("._"))
            .collect();
        assert_eq!(files.len(), 2, "{files:?}");
    }

    /// The agent user gets a real `/output` folder, as in 0.8, but never the
    /// target of a symlink planted there.
    #[test]
    fn output_folders_go_to_the_agent_unless_they_are_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join("output")).unwrap();
        std::fs::create_dir(root.join("elsewhere")).unwrap();
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("outputs")).unwrap();
        // A stand-in chown that records what it was asked to change.
        let bin = root.join("bin");
        std::fs::create_dir(&bin).unwrap();
        let log = root.join("chown.log");
        std::fs::write(
            bin.join("chown"),
            format!(
                "#!/bin/sh\necho \"$@\" >> {}\n",
                q(&log.display().to_string())
            ),
        )
        .unwrap();
        std::fs::set_permissions(
            bin.join("chown"),
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();
        let output = root.join("output").display().to_string();
        let outputs = root.join("outputs").display().to_string();
        let missing = root.join("missing").display().to_string();
        let script = own_dirs("agent:agent", &[&output, &outputs, &missing]);
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .status()
            .unwrap();
        assert!(status.success());
        assert_eq!(
            std::fs::read_to_string(&log).unwrap(),
            format!("-R agent:agent {output}\n")
        );
        assert!(setup_user("agent", "/app").contains(r#"chown -R "$u:$g" "$d""#));
        assert!(setup_user("agent", "/app").contains("for d in /output /outputs;"));
    }

    /// The programs must at least parse as POSIX shell.
    #[test]
    fn programs_are_valid_shell() {
        for (name, prog) in [
            ("setup_user", setup_user("agent", "/app")),
            (
                "lock_hidden",
                lock_hidden(&["/verifier".into(), "/oracle".into()]),
            ),
            ("lock_system_path", lock_system_path("/app")),
            ("baseline", baseline()),
            ("kill_user", kill_user("agent")),
            ("handoff", handoff("/app", None, true)),
            ("handoff-lenient", handoff("/app", None, false)),
            ("claim", claim("/tests", true)),
            ("claim_links_only", claim("/app/data", false)),
            ("open_parents", open_parents("/app/data")),
            ("trusted_path", trusted_path("/app")),
            (
                "lock_system_dirs",
                lock_system_dirs("/app", "/home/agent", "agent"),
            ),
            ("preload_snapshot", preload_snapshot()),
            ("preload_check", preload_check()),
            ("memory_limit", memory_limit()),
            ("oom_prefix", format!("{}true", oom_prefix())),
            ("verifier_env_check", verifier_env_check(LOADER_VARS)),
            ("verifier_env_check_empty", verifier_env_check(&[])),
            (
                "pack_outputs",
                pack_outputs(&["/app".into(), "/var/log/x".into()]),
            ),
        ] {
            let out = std::process::Command::new("sh")
                .arg("-n")
                .arg("-c")
                .arg(&prog)
                .output()
                .expect("sh");
            assert!(
                out.status.success(),
                "{name} is not valid sh: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
