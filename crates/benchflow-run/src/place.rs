//! Files BenchFlow writes into the agent user's home: login and settings
//! files, a loaded session.
//!
//! Root never writes there. The agent owns its home and may already have run
//! in this sandbox (a later stage; an ungraded stage's processes even stay
//! alive into the next), so any path under the home may be a link it
//! planted, swapped at any moment: root writing through it, or changing an
//! owner after, would act on whatever the link points at. Instead root
//! stages each file in a fresh folder of its own under [`STAGE_ROOT`], gives
//! the staged copy to the agent user, and the agent user copies it into
//! place. The agent user can only write where it may, so a link it planted
//! gains it nothing.
//!
//! The folders on the way to a file become the agent user's, as when root
//! wrote the files itself (an image or a task mount may have made
//! `~/.claude` root's). Root changes each one from inside it: it enters the
//! folder and checks that the folder it is in is the one it looked at
//! (device and inode), so a folder swapped for a link at any moment is never
//! followed; it stops at a link, a missing folder, a mount, or a `.` or `..`
//! in the path. Every command runs with `PATH` set to the system folders,
//! never the image's (a folder on the image's `PATH` may be the agent's).

use std::time::Duration;

use benchflow_sandbox::shell::SYSTEM_PATH;
use benchflow_sandbox::{Exec, ExecOutput, Sandbox};

use crate::hardening::Report;
use crate::outcome::{Owner, Phase, Problem};
use crate::shell_quote as q;
use crate::trial::{sandbox_problem, unreported_in};

/// Where root makes each stage: a folder only root may change, which the
/// agent user may pass through but not list, in `/var/lib`, whose parents
/// setup made root's (as for [`crate::hardening::STATE_DIR`]); root checks
/// that they are before each use. Not `/tmp`: an image may leave `/tmp`
/// without its sticky bit, or make it the agent's workdir, and the agent
/// could then swap a stage for a link.
pub(crate) const STAGE_ROOT: &str = "/var/lib/benchflow-place";

/// One file for the agent user: its absolute path, contents, and mode.
pub(crate) struct ForAgent<'a> {
    pub path: String,
    pub contents: &'a [u8],
    pub mode: u32,
}

/// The agent user the files are for.
pub(crate) struct AgentUser<'a> {
    pub name: &'a str,
    pub home: &'a str,
    /// An agent already worked in this sandbox (an earlier stage), so a
    /// change to its home that stops the copy is the agent's own.
    pub earlier_turns: bool,
}

/// How long each step may take.
const STEP: Duration = Duration::from_secs(120);

/// The last line the agent user's copy prints when it ran to the end.
const DONE: &str = "BF placement done";

/// What a failed placement asks of the task.
const NEXT: &str = "make the agent's home and the folders in it the agent user's in the image, or set [agent] user = \"root\" in the task (the agent-user defense is then off)";

/// Root's commands here: the system folders' programs only, and none of
/// the image's shell start-up files or `cd` search folders.
fn as_root(script: String) -> Exec {
    Exec::new(script)
        .root()
        .env("PATH", SYSTEM_PATH)
        .env("CDPATH", "")
        .env("BASH_ENV", "")
        .env("ENV", "")
        .timeout(STEP)
}

/// The folders between `home` and each file, top-down, for files in the
/// home; one list per file.
fn folders_on_the_way(home: &str, files: &[ForAgent]) -> Vec<Vec<String>> {
    let home = home.trim_end_matches('/');
    files
        .iter()
        .filter_map(|f| {
            let rest = f.path.strip_prefix(&format!("{home}/"))?;
            let parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty()).collect();
            // A `.` or `..` would lead out of the home: no folder of such a
            // path is given.
            if parts.iter().any(|p| *p == "." || *p == "..") {
                return None;
            }
            let folders: Vec<String> = parts[..parts.len().saturating_sub(1)]
                .iter()
                .map(|p| p.to_string())
                .collect();
            (!folders.is_empty()).then_some(folders)
        })
        .collect()
}

/// Root's first step: check that only root may change the folders above
/// [`STAGE_ROOT`] (`BF stage-unsafe <folder>` and exit 4 when not; a group
/// the agent user is in counts as someone else), make the stage, and give
/// the agent user the folders on the way to its files, as the module docs
/// describe (`BF mount <folder>` where a mount stopped it). It gives none
/// when the home is a link, or a folder above it is not root's alone (an
/// image that puts the home in the workdir, say). Ends with `BF
/// stage-ready`.
fn prepare_script(user: &AgentUser, stage: &str, files: &[ForAgent]) -> String {
    let mut give = String::new();
    for folders in folders_on_the_way(user.home, files) {
        let list: Vec<String> = folders.iter().map(|d| q(d)).collect();
        give.push_str(&format!("  give {}\n", list.join(" ")));
    }
    format!(
        r#"r={root}; s={stage}; u={user}; h={home}
uid=$(id -u "$u") || exit 3; g=$(id -gn "$u") || exit 3
case "$uid" in ''|*[!0-9]*|0) exit 3;; esac
ug=" $(id -G "$u" 2>/dev/null) "
safe() {{
  [ -L "$1" ] && return 1
  [ -d "$1" ] || return 1
  [ -z "$(find "$1" -maxdepth 0 \( ! -user 0 -o -perm -0002 \) -print 2>/dev/null)" ] || return 1
  if [ -n "$(find "$1" -maxdepth 0 -perm -0020 -print 2>/dev/null)" ]; then
    [ "$(stat -c %g "$1" 2>/dev/null)" = 0 ] || return 1
    case "$ug" in *" 0 "*) return 1;; esac
  fi
  return 0
}}
for d in / /var /var/lib; do safe "$d" || {{ echo "BF stage-unsafe $d"; exit 4; }}; done
if [ -L "$r" ] || {{ [ -e "$r" ] && [ ! -d "$r" ]; }}; then rm -f "$r"; fi
mkdir -p "$r" && chown -h root:root "$r" && chmod 0711 "$r" && [ -d "$r" ] && [ ! -L "$r" ] && mkdir -m 0711 "$s" || exit 3
home_ok() {{
  case "$h" in /?*) ;; *) return 1;; esac
  [ -L "$h" ] && return 1
  a=$h
  while a=$(dirname "$a"); [ "$a" != / ]; do safe "$a" || return 1; done
  return 0
}}
give() (
  cd -P "$h" 2>/dev/null || exit 0
  [ "$(pwd -P)" = "$h" ] || exit 0
  hd=$(stat -c %d . 2>/dev/null) || exit 0
  for p in "$@"; do
    if [ -L "./$p" ] || [ ! -d "./$p" ]; then exit 0; fi
    want=$(stat -c %d:%i "./$p" 2>/dev/null) || exit 0
    [ "${{want%%:*}}" = "$hd" ] || {{ echo "BF mount $p"; exit 0; }}
    cd -P "./$p" 2>/dev/null || exit 0
    [ "$(stat -c %d:%i . 2>/dev/null)" = "$want" ] || exit 0
    chown -h "$u:$g" . 2>/dev/null
  done
)
if home_ok; then
{give}fi
echo "BF stage-ready""#,
        root = q(STAGE_ROOT),
        stage = q(stage),
        user = q(user.name),
        home = q(user.home.trim_end_matches('/')),
    )
}

/// The agent user's copy of each staged file into place: `BF unplaced
/// <path>` for each it could not write, then [`DONE`]. An existing file is
/// removed first, so the copy never writes through a link or opens a pipe
/// (one it cannot remove is reported, not opened), and the new one is made
/// private before its mode is set.
fn copy_script(stage: &str, files: &[ForAgent]) -> String {
    let mut s = format!(
        r#"umask 077; s={stage}
place() {{
  d=$2
  mkdir -p "$(dirname "$d")" 2>/dev/null
  if [ -d "$d" ] && [ ! -L "$d" ]; then printf 'BF unplaced %s\n' "$d"; return; fi
  rm -f "$d" 2>/dev/null
  if [ -e "$d" ] || [ -L "$d" ]; then printf 'BF unplaced %s\n' "$d"; return; fi
  if cp "$s/$1" "$d" 2>/dev/null && chmod "$3" "$d" 2>/dev/null; then :; else printf 'BF unplaced %s\n' "$d"; fi
}}
"#,
        stage = q(stage)
    );
    for (i, f) in files.iter().enumerate() {
        s.push_str(&format!("place {i} {} {:o}\n", q(&f.path), f.mode));
    }
    s.push_str(&format!("echo \"{DONE}\""));
    s
}

/// Whether a step that ends by printing `marker` ran to its end.
fn reported(out: &ExecOutput, marker: &str) -> bool {
    out.stdout.lines().rev().find(|l| !l.trim().is_empty()) == Some(marker)
}

/// A step that did not end with its marker: the sandbox's when it ran out
/// of time (retried on resume), R307 when it exited 0 but said nothing
/// (something on the image's `PATH` answered in its place), else R104.
fn failed(out: &ExecOutput, what: &str, message: String) -> Problem {
    if out.timed_out {
        return Problem::new(
            "R103",
            Owner::Sandbox,
            Phase::Setup,
            format!("{what} did not finish in {} s", STEP.as_secs()),
            "retry; if it repeats, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
        )
        .detail(out.tail(2000))
        .transient();
    }
    if out.success() {
        return unreported_in(Phase::Setup, what, out);
    }
    Problem::new("R104", Owner::Task, Phase::Setup, message, NEXT).detail(out.tail(2000))
}

/// Write `files` for `user`, who is not root, as the module docs describe.
pub(crate) async fn for_agent(
    sb: &dyn Sandbox,
    user: &AgentUser<'_>,
    files: &[ForAgent<'_>],
) -> Result<(), Problem> {
    if files.is_empty() {
        return Ok(());
    }
    let stage = format!("{STAGE_ROOT}/{}", uuid::Uuid::new_v4().simple());
    let out = sb
        .exec(as_root(prepare_script(user, &stage, files)))
        .await
        .map_err(|e| sandbox_problem(Phase::Setup, e))?;
    if !reported(&out, "BF stage-ready") {
        // Best effort, as below: the step may have made the stage.
        let _ = sb.exec(as_root(format!("rm -rf {}", q(&stage)))).await;
    }
    if let Some(folder) = Report::parse(&out.stdout).first("stage-unsafe") {
        return Err(Problem::new(
            "R104",
            Owner::Task,
            Phase::Setup,
            format!(
                "{folder} can be changed by users other than root, so BenchFlow cannot stage the agent's files in {STAGE_ROOT} safely"
            ),
            "make /, /var, and /var/lib root's and not writable by group or others in the image",
        )
        .detail(out.tail(2000)));
    }
    if !reported(&out, "BF stage-ready") {
        return Err(failed(
            &out,
            "making the folder for the agent's files",
            format!(
                "could not make {stage} for the agent's files, or find the agent user `{}` (the image needs `id`, `chown`, and a writable /var/lib)",
                user.name
            ),
        ));
    }
    // A mount on the way is the task's (the agent cannot mount), so what
    // it blocks is too.
    let mounted = !Report::parse(&out.stdout).all("mount").is_empty();
    let result = stage_and_copy(sb, user, &stage, files, mounted).await;
    // Best effort: what is left there is the agent user's own, in a folder
    // only root may change.
    let _ = sb.exec(as_root(format!("rm -rf {}", q(&stage)))).await;
    result
}

async fn stage_and_copy(
    sb: &dyn Sandbox,
    user: &AgentUser<'_>,
    stage: &str,
    files: &[ForAgent<'_>],
    mounted: bool,
) -> Result<(), Problem> {
    let mut staged = Vec::new();
    for (i, f) in files.iter().enumerate() {
        let path = format!("{stage}/{i}");
        sb.write_file(&path, f.contents, 0o600)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        staged.push(q(&path));
    }
    let out = sb
        .exec(as_root(format!(
            "g=$(id -gn {u}) || exit 3; chown -h {u}:\"$g\" {files} && echo \"BF staged\"",
            u = q(user.name),
            files = staged.join(" ")
        )))
        .await
        .map_err(|e| sandbox_problem(Phase::Setup, e))?;
    if !reported(&out, "BF staged") {
        return Err(failed(
            &out,
            "handing the agent's files to the agent user",
            format!(
                "could not give the agent user `{}` its files (the image needs `id` and `chown`)",
                user.name
            ),
        ));
    }
    let out = sb
        .exec(
            Exec::new(copy_script(stage, files))
                .user(user.name.to_string())
                .env("PATH", SYSTEM_PATH)
                .timeout(STEP),
        )
        .await
        .map_err(|e| sandbox_problem(Phase::Setup, e))?;
    if !reported(&out, DONE) {
        // After an earlier stage, what stalls the agent user's own copy in
        // its own home (a pipe it put where a file goes) is the agent's.
        if out.timed_out && user.earlier_turns {
            return Err(Problem::new(
                "R205",
                Owner::Agent,
                Phase::Setup,
                format!(
                    "the agent user `{}`'s copy of its files into its home did not finish in {} s; the agent already worked in this sandbox, and what it left in its home is the likely cause",
                    user.name,
                    STEP.as_secs()
                ),
                "read the earlier stages' trajectories for what the agent did to its home",
            )
            .detail(out.tail(2000)));
        }
        return Err(failed(
            &out,
            "the agent user's copy of its files",
            format!(
                "the agent user `{}` could not copy its files into its home",
                user.name
            ),
        ));
    }
    let unplaced = Report::parse(&out.stdout);
    let unplaced = unplaced.all("unplaced");
    if unplaced.is_empty() {
        return Ok(());
    }
    let list = unplaced.join(", ");
    Err(if user.earlier_turns && !mounted {
        Problem::new(
            "R205",
            Owner::Agent,
            Phase::Setup,
            format!(
                "the agent user `{}` could not write {list}, where BenchFlow puts the agent's files; the agent already worked in this sandbox, and what it did to its home (a file where a folder belongs, a folder it made read-only) is the likely cause",
                user.name
            ),
            "read the earlier stages' trajectories for what the agent did to its home",
        )
    } else {
        Problem::new(
            "R104",
            Owner::Task,
            Phase::Setup,
            format!(
                "the agent user `{}` could not write {list}: a folder on the way is not its to write (root's, or read-only), or is a link to one",
                user.name
            ),
            NEXT,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file<'a>(path: &str, contents: &'a [u8], mode: u32) -> ForAgent<'a> {
        ForAgent {
            path: path.into(),
            contents,
            mode,
        }
    }

    /// The copy runs one `place` line per file, in order, and ends with the
    /// marker; paths are quoted for the shell.
    #[test]
    fn the_copy_script_places_every_file_and_reports_back() {
        let files = [
            file("/home/agent/.codex/auth.json", b"{}", 0o600),
            file("/home/agent/my dir/it's.toml", b"", 0o644),
        ];
        let s = copy_script("/tmp/.benchflow-place-x", &files);
        assert!(
            s.contains("place 0 /home/agent/.codex/auth.json 600\n"),
            "{s}"
        );
        assert!(
            s.contains("place 1 '/home/agent/my dir/it'\\''s.toml' 644\n"),
            "{s}"
        );
        assert!(s.ends_with(&format!("echo \"{DONE}\"")), "{s}");
    }

    /// Run as a user on this machine, the copy writes where it may (never
    /// through a link that was there), sets each mode, and names what it
    /// could not write.
    #[cfg(unix)]
    #[test]
    fn the_copy_writes_where_it_may_and_names_the_rest() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let stage = tmp.path().join("stage");
        std::fs::create_dir_all(&stage).unwrap();
        for (i, body) in ["one", "two", "three", "four"].iter().enumerate() {
            std::fs::write(stage.join(i.to_string()), body).unwrap();
        }
        let blocked = tmp.path().join("blocked");
        std::fs::create_dir_all(&blocked).unwrap();
        std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o555)).unwrap();
        let home = tmp.path().join("home");
        let ok = home.join(".cfg/a.json");
        let no = blocked.join("b.json");
        // A link where a file goes: replaced, its target left alone.
        let target = tmp.path().join("target");
        std::fs::write(&target, "kept").unwrap();
        let link = home.join("link.json");
        std::fs::create_dir_all(&home).unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let dir = home.join("a-folder");
        std::fs::create_dir_all(&dir).unwrap();
        let path = |p: &std::path::Path| p.to_str().unwrap().to_string();
        let files = [
            file(&path(&ok), b"", 0o640),
            file(&path(&no), b"", 0o600),
            file(&path(&link), b"", 0o600),
            file(&path(&dir), b"", 0o600),
        ];
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(copy_script(&path(&stage), &files))
            .output()
            .unwrap();
        std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o755)).unwrap();
        let out = String::from_utf8(out.stdout).unwrap();
        assert_eq!(std::fs::read_to_string(&ok).unwrap(), "one");
        let mode = std::fs::metadata(&ok).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "kept");
        assert!(!std::fs::symlink_metadata(&link).unwrap().is_symlink());
        assert_eq!(std::fs::read_to_string(&link).unwrap(), "three");
        assert!(dir.is_dir());
        assert!(
            out.contains(&format!("BF unplaced {}\n", dir.display())),
            "{out}"
        );
        // Root may write in a folder it cannot write to by its mode.
        let uid = std::process::Command::new("id").arg("-u").output().unwrap();
        if String::from_utf8_lossy(&uid.stdout).trim() != "0" {
            assert!(
                out.contains(&format!("BF unplaced {}\n", no.display())),
                "{out}"
            );
        }
        assert!(out.ends_with(&format!("{DONE}\n")), "{out}");
    }

    /// The folders on the way are those between the home and each file,
    /// top-down, named from the home; a file outside the home has none, and
    /// so has a path with `.` or `..` in it.
    #[test]
    fn the_folders_on_the_way_are_below_the_home() {
        let files = [
            file("/home/agent/.claude/projects/-app/s.jsonl", b"", 0o600),
            file("/home/agent/top.json", b"", 0o600),
            file("/etc/elsewhere.conf", b"", 0o600),
            file("/home/agent/.codex/auth.json", b"", 0o600),
            file("/home/agent/../../etc/x/y.conf", b"", 0o600),
            file("/home/agent/./a/b.json", b"", 0o600),
        ];
        assert_eq!(
            folders_on_the_way("/home/agent/", &files),
            vec![
                vec![
                    ".claude".to_string(),
                    "projects".to_string(),
                    "-app".to_string(),
                ],
                vec![".codex".to_string()],
            ]
        );
    }

    /// Run on this machine as its user, root's first step makes the stage
    /// and gives each real folder on the way, from inside it, top-down,
    /// stopping at a link or a missing folder.
    #[cfg(unix)]
    #[test]
    fn the_first_step_gives_real_folders_and_stops_at_a_link() {
        let tmp = tempfile::tempdir().unwrap();
        let t = tmp.path().canonicalize().unwrap();
        let home = t.join("home");
        // `-app` as Claude Code names a session folder: a leading dash.
        std::fs::create_dir_all(home.join(".claude/projects/-app")).unwrap();
        std::fs::create_dir_all(t.join("elsewhere/inner")).unwrap();
        std::os::unix::fs::symlink(t.join("elsewhere"), home.join(".codex")).unwrap();
        let me = std::process::Command::new("id")
            .arg("-un")
            .output()
            .unwrap();
        let me = String::from_utf8(me.stdout).unwrap().trim().to_string();
        let h = home.to_str().unwrap().to_string();
        let user = AgentUser {
            name: &me,
            home: &h,
            earlier_turns: false,
        };
        let files = [
            file(&format!("{h}/.claude/projects/-app/s.jsonl"), b"", 0o600),
            file(&format!("{h}/.codex/inner/auth.json"), b"", 0o600),
        ];
        let root = t.join("place");
        let stage = root.join("abc");
        // BSD `stat` spells GNU's `-c` as `-f`; `/var` is a link here.
        let stat = if cfg!(target_os = "linux") {
            "-c"
        } else {
            "-f"
        };
        let script = prepare_script(&user, stage.to_str().unwrap(), &files)
            .replace(&q(STAGE_ROOT), &q(root.to_str().unwrap()))
            .replace("for d in / /var /var/lib; do", "for d in /; do")
            .replace("if home_ok; then", "if true; then")
            .replace("chown -h root:root \"$r\" && ", "")
            .replace("stat -c", &format!("stat {stat}"))
            .replace(
                "chown -h \"$u:$g\" . 2>/dev/null",
                "echo \"GAVE $(pwd -P)\"",
            );
        let uid = std::process::Command::new("id").arg("-u").output().unwrap();
        if String::from_utf8_lossy(&uid.stdout).trim() == "0" {
            return; // the script refuses uid 0 as the agent user
        }
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .unwrap();
        let out = String::from_utf8(out.stdout).unwrap();
        let gave: Vec<&str> = out
            .lines()
            .filter_map(|l| l.strip_prefix("GAVE "))
            .collect();
        assert_eq!(
            gave,
            vec![
                format!("{h}/.claude"),
                format!("{h}/.claude/projects"),
                format!("{h}/.claude/projects/-app"),
            ],
            "{out}"
        );
        assert!(stage.is_dir(), "{out}");
        assert!(out.ends_with("BF stage-ready\n"), "{out}");
    }

    /// A pipe where a file goes, in a folder the agent user cannot change,
    /// is reported, never opened (opening it would stall the copy).
    #[cfg(unix)]
    #[test]
    fn the_copy_never_opens_a_pipe_it_cannot_remove() {
        use std::os::unix::fs::PermissionsExt;
        let uid = std::process::Command::new("id").arg("-u").output().unwrap();
        if String::from_utf8_lossy(&uid.stdout).trim() == "0" {
            return; // root removes the pipe whatever the folder's mode
        }
        let tmp = tempfile::tempdir().unwrap();
        let stage = tmp.path().join("stage");
        std::fs::create_dir_all(&stage).unwrap();
        std::fs::write(stage.join("0"), "one").unwrap();
        let dir = tmp.path().join("home/.codex");
        std::fs::create_dir_all(&dir).unwrap();
        let pipe = dir.join("auth.json");
        let made = std::process::Command::new("mkfifo")
            .arg(&pipe)
            .status()
            .unwrap();
        assert!(made.success());
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let files = [file(pipe.to_str().unwrap(), b"", 0o600)];
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg(copy_script(stage.to_str().unwrap(), &files))
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let start = std::time::Instant::now();
        let status = loop {
            if let Some(s) = child.try_wait().unwrap() {
                break Some(s);
            }
            if start.elapsed() > Duration::from_secs(10) {
                let _ = child.kill();
                break None;
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(status.is_some(), "the copy stalled on the pipe");
        let out = child.wait_with_output().unwrap();
        let out = String::from_utf8(out.stdout).unwrap();
        assert!(
            out.contains(&format!("BF unplaced {}\n", pipe.display())),
            "{out}"
        );
        assert!(out.ends_with(&format!("{DONE}\n")), "{out}");
    }

    /// A step ran to its end only when its marker is its last line.
    #[test]
    fn a_step_reported_only_with_its_marker_last() {
        let out = |stdout: &str| ExecOutput {
            stdout: stdout.into(),
            ..Default::default()
        };
        assert!(reported(&out("BF staged\n"), "BF staged"));
        assert!(reported(&out("noise\nBF staged\n\n"), "BF staged"));
        assert!(!reported(&out("BF staged\nmore\n"), "BF staged"));
        assert!(!reported(&out("BF staged later\n"), "BF staged"));
        assert!(!reported(&out(""), "BF staged"));
    }
}
