//! Copying a path out of the sandbox that code inside may control: under a
//! byte cap and a deadline, with any partial copy removed when it fails
//! (reviews F2-A-r67 P1, F2-A-r8 H1 and M2). A root agent's leftover
//! processes are still alive at every such copy: one can swap the path for
//! a huge sparse file or a folder, or replace `tar` with a program that
//! never ends, or one that streams the path and then says it is missing.

use std::fmt;
use std::path::Path;
use std::time::Duration;

use benchflow_sandbox::{Sandbox, SandboxError};

use crate::hardening;

/// The longest a copy out of the sandbox may take.
pub(crate) const COPY_OUT_TIMEOUT: Duration = Duration::from_secs(180);

/// What a packed archive may add to its files' content: its own tar
/// headers and padding (at most [`hardening::MAX_OUTPUT_FILES`] files).
pub(crate) const TAR_SLACK: u64 = 64 * 1024 * 1024;

/// The cap for a copy of an archive packed under the output caps.
pub(crate) const ARCHIVE_CAP: u64 = hardening::MAX_OUTPUT_TOTAL_BYTES + TAR_SLACK;

/// What a copy out must arrive as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    /// One file (an archive, a log): anything else is refused.
    File,
    /// A file or a folder.
    Any,
    /// A file, or a folder of its name holding it (as a backend may
    /// unpack one): judge evidence. Takes as few entries as a file.
    FileOrFolder,
}

impl Shape {
    /// The entries a copy of this shape may bring, folders created on the
    /// way included (review F2-A-r12-3): one file needs a handful; a folder
    /// (the verifier's) fifty times the files the outputs may hold: a file
    /// costs 1 or 2 entries, a link up to 3, a folder 2 or 3 (its header, a
    /// long-name or PAX header, and its creation), so 100,000 files, links
    /// and folders in all always fit (review F2-B-r13-1, and the review of
    /// 661c655e).
    pub(crate) fn max_entries(self) -> u64 {
        match self {
            Self::File | Self::FileOrFolder => benchflow_sandbox::archive::FILE_ENTRIES,
            Self::Any => FOLDER_ENTRIES,
        }
    }
}

/// The entries a copy of a folder may bring.
pub(crate) const FOLDER_ENTRIES: u64 = 50 * hardening::MAX_OUTPUT_FILES;

/// Why a copy out failed.
#[derive(Debug)]
pub(crate) enum CopyOutError {
    /// The copy did not finish within its deadline.
    TimedOut(Duration),
    /// A copy that had to be one file arrived as something else, or as
    /// nothing.
    NotAFile,
    /// The copy itself failed, past its cap among other things.
    Failed(SandboxError),
}

impl CopyOutError {
    /// True when more than the cap would have arrived.
    pub(crate) fn over_cap(&self) -> bool {
        matches!(self, Self::Failed(SandboxError::Io { source, .. })
            if source.kind() == std::io::ErrorKind::FileTooLarge)
    }

    /// As a sandbox error. A copy that never ended is the connection's
    /// (`Transport`), as a stalled one always was (review F2-A-r9 M3): a
    /// slow backend looks the same as code inside that holds the copy
    /// open, so the loss rules decide (run again once, a repeat kept as an
    /// error), never a grade. One that arrived as the wrong shape is the
    /// sandbox's content, as one past its cap is.
    pub(crate) fn into_sandbox(self) -> SandboxError {
        match self {
            Self::Failed(e) => e,
            t @ Self::TimedOut(_) => SandboxError::Transport(t.to_string()),
            n @ Self::NotAFile => SandboxError::Content(n.to_string()),
        }
    }
}

impl fmt::Display for CopyOutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimedOut(d) => write!(f, "the copy did not finish within {}s", d.as_secs()),
            Self::NotAFile => f.write_str("the copy did not arrive as a file"),
            Self::Failed(e) => e.fmt(f),
        }
    }
}

/// Copy `remote` out of the sandbox to `local`, failing before more than
/// `max_bytes` of file content arrive or after `deadline`, or when `shape`
/// is [`Shape::File`] and something else arrived. On every failure,
/// whatever is at `local` (a file or a folder) is removed, a path missing
/// in the sandbox included: that verdict can rest on what code inside
/// prints after it streamed the path (review F2-A-r9 M1). So `local` must
/// be a path the caller has nothing of its own at.
pub(crate) async fn copy_out(
    sb: &dyn Sandbox,
    remote: &str,
    local: &Path,
    max_bytes: u64,
    deadline: Duration,
    shape: Shape,
) -> Result<(), CopyOutError> {
    let limits = benchflow_sandbox::archive::Limits {
        bytes: Some(max_bytes),
        entries: Some(shape.max_entries()),
    };
    let r = match tokio::time::timeout(deadline, sb.download_limited(remote, local, limits)).await {
        Ok(Ok(())) if shape != Shape::File => return Ok(()),
        // Anything but a regular file: a folder, or nothing at all, which
        // is how a link, a FIFO or a device in the archive's place arrives
        // (the unpacker skips it and reports the copy done; review
        // F2-A-r10 M1).
        Ok(Ok(())) => match std::fs::symlink_metadata(local) {
            // Readable and writable by this machine whatever mode it came
            // with: a mode-000 archive must not fail this machine's own
            // read (review F2-A-r11 M1).
            Ok(m) if m.is_file() => {
                use std::os::unix::fs::PermissionsExt;
                match std::fs::set_permissions(local, std::fs::Permissions::from_mode(0o600)) {
                    Ok(()) => return Ok(()),
                    Err(e) => CopyOutError::Failed(SandboxError::io(
                        format!("making {} readable", local.display()),
                        e,
                    )),
                }
            }
            _ => CopyOutError::NotAFile,
        },
        Ok(Err(e)) => CopyOutError::Failed(e),
        Err(_) => CopyOutError::TimedOut(deadline),
    };
    remove(local);
    Err(r)
}

/// Remove a copy, a file or a folder (a folder in place of an archive is
/// how code inside can make one arrive).
pub(crate) fn remove(path: &Path) {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.is_dir() => {
            let _ = std::fs::remove_dir_all(path);
        }
        Ok(_) => {
            let _ = std::fs::remove_file(path);
        }
        Err(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use benchflow_sandbox::{Exec, ExecOutput, Process};

    /// What the fake's copy does at `local`.
    #[derive(Clone, Copy)]
    enum Copy {
        File,
        /// A file with mode 000, as code inside can leave the archive.
        Unreadable,
        Folder,
        /// Reports the copy done with nothing written, as a link or a FIFO
        /// in the archive's place does.
        Nothing,
        /// Streams a file, then reports the path missing, as a replaced
        /// `tar` on Docker can.
        FileThenMissing,
        FolderThenMissing,
        OverCap,
        /// Writes a file, then never ends.
        Endless,
    }

    struct Fake(
        Copy,
        std::sync::Mutex<Option<u64>>,
        std::sync::Mutex<Option<u64>>,
    );

    fn missing() -> SandboxError {
        SandboxError::io(
            "downloading /x",
            std::io::Error::new(std::io::ErrorKind::NotFound, "gone"),
        )
    }

    #[async_trait::async_trait]
    impl Sandbox for Fake {
        fn id(&self) -> &str {
            "fake"
        }
        fn backend(&self) -> &str {
            "fake"
        }
        async fn exec(&self, _: Exec) -> Result<ExecOutput, SandboxError> {
            unimplemented!()
        }
        async fn spawn(&self, _: Exec) -> Result<Process, SandboxError> {
            unimplemented!()
        }
        async fn upload(&self, _: &Path, _: &str) -> Result<(), SandboxError> {
            unimplemented!()
        }
        async fn write_file(&self, _: &str, _: &[u8], _: u32) -> Result<(), SandboxError> {
            unimplemented!()
        }
        async fn download(&self, _: &str, _: &Path) -> Result<(), SandboxError> {
            panic!("an uncapped copy")
        }
        async fn destroy(&self) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn download_limited(
            &self,
            remote: &str,
            local: &Path,
            limits: benchflow_sandbox::archive::Limits,
        ) -> Result<(), SandboxError> {
            *self.2.lock().unwrap() = limits.entries;
            self.download_capped(remote, local, limits.bytes.unwrap())
                .await
        }
        async fn download_capped(
            &self,
            _: &str,
            local: &Path,
            max_bytes: u64,
        ) -> Result<(), SandboxError> {
            *self.1.lock().unwrap() = Some(max_bytes);
            let file = || std::fs::write(local, vec![0u8; 4096]).unwrap();
            let folder = || {
                std::fs::create_dir_all(local.join("junk")).unwrap();
                std::fs::write(local.join("junk/f"), "x").unwrap();
            };
            match self.0 {
                Copy::Nothing => {}
                Copy::File => file(),
                Copy::Unreadable => {
                    use std::os::unix::fs::PermissionsExt;
                    file();
                    std::fs::set_permissions(local, std::fs::Permissions::from_mode(0o000))
                        .unwrap();
                }
                Copy::Folder => folder(),
                Copy::FileThenMissing => {
                    file();
                    return Err(missing());
                }
                Copy::FolderThenMissing => {
                    folder();
                    return Err(missing());
                }
                Copy::OverCap => {
                    file();
                    return Err(SandboxError::io(
                        "unpacking",
                        std::io::Error::new(std::io::ErrorKind::FileTooLarge, "cap"),
                    ));
                }
                Copy::Endless => {
                    file();
                    std::future::pending::<()>().await;
                }
            }
            Ok(())
        }
    }

    async fn run(copy: Copy, shape: Shape) -> (Result<(), CopyOutError>, bool, Option<u64>) {
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("a.tar");
        let sb = Fake(copy, Default::default(), Default::default());
        let r = copy_out(&sb, "/x", &local, 1234, COPY_OUT_TIMEOUT, shape).await;
        // The entry budget each shape gives (review F2-A-r12-3).
        let want = match shape {
            Shape::File | Shape::FileOrFolder => 8,
            Shape::Any => 500_000,
        };
        assert_eq!(*sb.2.lock().unwrap(), Some(want));
        let max = *sb.1.lock().unwrap();
        (r, local.exists(), max)
    }

    /// Each of `copy_out`'s branches (review F2-B-r9 L4): what it returns,
    /// whether anything is left at `local`, and the cap it passed on.
    #[tokio::test(start_paused = true)]
    async fn copy_out_keeps_only_a_whole_copy_of_the_right_shape() {
        let (r, left, max) = run(Copy::File, Shape::File).await;
        assert!(r.is_ok() && left, "{r:?}");
        assert_eq!(max, Some(1234));
        // A file this machine could not read arrives readable by it
        // (review F2-A-r11 M1).
        {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let local = dir.path().join("a.tar");
            let sb = Fake(Copy::Unreadable, Default::default(), Default::default());
            let r = copy_out(&sb, "/x", &local, 1234, COPY_OUT_TIMEOUT, Shape::File).await;
            assert!(r.is_ok(), "{r:?}");
            let mode = std::fs::metadata(&local).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
            assert_eq!(std::fs::read(&local).unwrap().len(), 4096);
        }
        let (r, left, _) = run(Copy::Folder, Shape::Any).await;
        assert!(r.is_ok() && left, "{r:?}");
        let (r, left, _) = run(Copy::Folder, Shape::FileOrFolder).await;
        assert!(r.is_ok() && left, "{r:?}");
        // A folder in place of a file (review F2-A-r9 M2).
        let (r, left, _) = run(Copy::Folder, Shape::File).await;
        assert!(matches!(r, Err(CopyOutError::NotAFile)) && !left, "{r:?}");
        // Nothing at all, for a link or a FIFO (review F2-A-r10 M1).
        let (r, left, _) = run(Copy::Nothing, Shape::File).await;
        assert!(matches!(r, Err(CopyOutError::NotAFile)) && !left, "{r:?}");
        let (r, _, _) = run(Copy::Nothing, Shape::Any).await;
        assert!(r.is_ok(), "{r:?}");
        assert!(matches!(
            CopyOutError::NotAFile.into_sandbox(),
            SandboxError::Content(_)
        ));
        // Streamed, then "missing" (review F2-A-r9 M1): nothing stays.
        for copy in [Copy::FileThenMissing, Copy::FolderThenMissing] {
            let (r, left, _) = run(copy, Shape::Any).await;
            assert!(
                matches!(&r, Err(CopyOutError::Failed(e)) if e.is_missing_in_sandbox()),
                "{r:?}"
            );
            assert!(!left, "a copy said to be missing left what arrived");
        }
        let (r, left, _) = run(Copy::OverCap, Shape::File).await;
        assert!(
            r.as_ref().is_err_and(CopyOutError::over_cap) && !left,
            "{r:?}"
        );
        // Past the deadline: the connection's, as a stall always was
        // (review F2-A-r9 M3), so the loss rules decide.
        let start = tokio::time::Instant::now();
        let (r, left, _) = run(Copy::Endless, Shape::File).await;
        assert!(start.elapsed() >= COPY_OUT_TIMEOUT);
        assert!(start.elapsed() < COPY_OUT_TIMEOUT + Duration::from_secs(1));
        assert!(
            matches!(r, Err(CopyOutError::TimedOut(_))) && !left,
            "{r:?}"
        );
        assert!(matches!(
            r.unwrap_err().into_sandbox(),
            SandboxError::Transport(_)
        ));
    }

    #[test]
    fn remove_takes_a_file_or_a_folder() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.tar");
        std::fs::write(&file, "x").unwrap();
        remove(&file);
        assert!(!file.exists());
        let folder = dir.path().join("b.tar");
        std::fs::create_dir_all(folder.join("junk")).unwrap();
        std::fs::write(folder.join("junk/f"), "x").unwrap();
        remove(&folder);
        assert!(!folder.exists());
        remove(&dir.path().join("none"));
    }
}
