//! Reading a failed image build's log: was it the Dockerfile, or the
//! network and the build host? Both backends build with BuildKit, so they
//! share one reading.

/// Output that means the failure was the network, a mirror, or the build
/// host, not the Dockerfile; such failures are worth retrying. The first six
/// are BenchFlow 0.8's `_DOCKER_BUILD_RETRYABLE_ERRORS`.
const TRANSIENT: &[&str] = &[
    "at least one invalid signature was encountered",
    "is not signed",
    "no space left on device",
    "readtimeouterror",
    "read timed out",
    "connection timed out",
    "connection reset by peer",
    "connection refused",
    "i/o timeout",
    "tls handshake timeout",
    "temporary failure in name resolution",
    "temporary failure resolving",
    "could not resolve host",
    "unexpected eof",
    "toomanyrequests",
    "429 too many requests",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway time",
    "hash sum mismatch",
    "context deadline exceeded",
];

/// Output that means the Dockerfile (or the image it names) is at fault.
const TASK_FAULT: &[&str] = &[
    "did not complete successfully",
    "returned a non-zero code",
    "dockerfile parse error",
    "unknown instruction",
    "failed to compute cache key",
    "copy failed",
    "add failed",
    "building at step",
    "failed to read dockerfile",
    "cannot locate specified dockerfile",
    "failed to resolve source metadata",
    "pull access denied",
    "manifest unknown",
    "repository does not exist",
    "invalid reference format",
    "no matching manifest",
    "failed to solve",
];

/// What a failed build's output says about the failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reading {
    /// The transient marker found, if any.
    pub transient: Option<&'static str>,
    /// The Dockerfile is at fault (and the failure was not transient).
    pub task_fault: bool,
    /// The line that best says what went wrong.
    pub key_line: String,
}

/// Read a failed build's output.
pub(crate) fn read(text: &str) -> Reading {
    let lower = text.to_lowercase();
    let transient = TRANSIENT.iter().copied().find(|p| lower.contains(p));
    let task_fault = transient.is_none() && TASK_FAULT.iter().any(|p| lower.contains(p));
    Reading {
        transient,
        task_fault,
        key_line: key_line(text),
    }
}

/// The last line that mentions an error, else the last non-empty line.
pub(crate) fn key_line(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let found = lines
        .iter()
        .rev()
        .find(|l| l.to_lowercase().contains("error"))
        .or(lines.last())
        .copied()
        .unwrap_or("no output");
    let count = found.chars().count();
    if count <= 300 {
        found.to_string()
    } else {
        format!("…{}", found.chars().skip(count - 300).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failing_run_step_is_the_tasks_fault() {
        let r = read(
            "#5 [2/2] RUN exit 3\nERROR: failed to solve: process \"/bin/sh -c exit 3\" did not complete successfully: exit code: 3\n",
        );
        assert!(r.task_fault);
        assert!(r.transient.is_none());
        assert!(r.key_line.starts_with("ERROR: failed to solve"));
    }

    #[test]
    fn a_network_failure_is_transient_even_inside_a_run_step() {
        let r = read(
            "E: Failed to fetch http://archive.ubuntu.com/... Connection timed out\nERROR: process did not complete successfully: exit code: 100",
        );
        assert_eq!(r.transient, Some("connection timed out"));
        assert!(!r.task_fault);
    }

    #[test]
    fn an_unknown_failure_is_neither() {
        let r = read("something odd happened");
        assert!(!r.task_fault);
        assert!(r.transient.is_none());
        assert_eq!(r.key_line, "something odd happened");
    }
}
