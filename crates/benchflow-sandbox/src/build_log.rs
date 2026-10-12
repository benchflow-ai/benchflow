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
    // A registry or mirror that cannot be reached (BuildKit's words for a
    // `FROM` it could not look up, which also say "failed to resolve source
    // metadata").
    "no such host",
    "network is unreachable",
    "no route to host",
    "server misbehaving",
    "tls: handshake failure",
];

/// The build lost its daemon (E5: the Docker daemon killed mid-build was the
/// task's R101, by BuildKit's generic "failed to solve"). Read only in the
/// build's own last error line ([`key_line`]): a `RUN docker version` in a
/// Dockerfile prints "Cannot connect to the Docker daemon" too, and a later
/// failing step is still the task's (E5 review A3).
const DAEMON_LOST: &[&str] = &[
    "error reading from server: eof",
    "failed to solve: unavailable",
    "code = unavailable",
    "transport is closing",
    "cannot connect to the docker daemon",
    "error during connect",
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
    let key = key_line(text);
    let key_lower = key.to_lowercase();
    let transient = TRANSIENT
        .iter()
        .copied()
        .find(|p| lower.contains(p))
        .or_else(|| DAEMON_LOST.iter().copied().find(|p| key_lower.contains(p)));
    let task_fault = transient.is_none() && TASK_FAULT.iter().any(|p| lower.contains(p));
    Reading {
        transient,
        task_fault,
        key_line: key,
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

    /// E5 (the Docker daemon killed mid-build on the VM): the build's
    /// lost connection was read as the Dockerfile's failure (R101).
    #[test]
    fn a_build_that_lost_its_daemon_is_not_the_tasks_fault() {
        for text in [
            "ERROR: failed to build: failed to solve: Unavailable: error reading from server: EOF",
            "ERROR: failed to solve: rpc error: code = Unavailable desc = transport is closing",
            "Cannot connect to the Docker daemon at tcp://127.0.0.1:23751. Is the docker daemon running?",
            "error during connect: Post \"http://127.0.0.1:23751/v1.52/build\": EOF",
        ] {
            let r = read(text);
            assert!(r.transient.is_some(), "{text}");
            assert!(!r.task_fault, "{text}");
        }
    }

    /// E5 review (A3): a step that prints the daemon's words (`RUN docker
    /// version || true` with no daemon) before a later step fails is the
    /// task's failure, not a lost daemon.
    #[test]
    fn a_step_that_prints_the_daemons_words_is_still_the_tasks() {
        let r = read(
            "#6 [2/3] RUN docker version || true\n#6 0.21 Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?\n#6 0.22 rpc error: code = Unavailable desc = transport is closing\n#6 DONE 0.3s\n#7 [3/3] RUN exit 3\nERROR: failed to solve: process \"/bin/sh -c exit 3\" did not complete successfully: exit code: 3\n",
        );
        assert!(r.transient.is_none(), "{r:?}");
        assert!(r.task_fault);
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
    fn an_unreachable_registry_is_not_the_tasks_fault() {
        for text in [
            "ERROR: failed to solve: ubuntu:24.04: failed to resolve source metadata for docker.io/library/ubuntu:24.04: failed to do request: Head \"https://registry-1.docker.io/v2/library/ubuntu/manifests/24.04\": dial tcp: lookup registry-1.docker.io on 127.0.0.11:53: no such host",
            "ERROR: failed to solve: failed to resolve source metadata for ghcr.io/x/y:1: dial tcp 140.82.112.34:443: connect: network is unreachable",
        ] {
            let r = read(text);
            assert!(r.transient.is_some(), "{text}");
            assert!(!r.task_fault, "{text}");
        }
        // An image that does not exist still is.
        let r = read(
            "ERROR: failed to solve: ubuntu:99.04: failed to resolve source metadata for docker.io/library/ubuntu:99.04: docker.io/library/ubuntu:99.04: not found: manifest unknown",
        );
        assert!(r.task_fault);
    }

    #[test]
    fn an_unknown_failure_is_neither() {
        let r = read("something odd happened");
        assert!(!r.task_fault);
        assert!(r.transient.is_none());
        assert_eq!(r.key_line, "something odd happened");
    }
}
