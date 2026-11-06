//! One error type for every command: a stable kind, a message, the errno
//! when the kernel gave one, and optional detail for the host.

use rustix::io::Errno;
use serde_json::{json, Map, Value};

#[derive(Debug)]
pub struct Error {
    pub kind: &'static str,
    pub msg: String,
    pub errno: Option<i32>,
    pub detail: Option<Value>,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(kind: &'static str, msg: impl Into<String>) -> Self {
        Error { kind, msg: msg.into(), errno: None, detail: None }
    }

    /// A request the protocol does not accept.
    pub fn bad(msg: impl Into<String>) -> Self {
        Error::new("bad-request", msg)
    }

    /// A failed system call on *what*.
    pub fn sys(e: Errno, what: impl AsRef<str>) -> Self {
        Error {
            kind: errno_kind(e),
            msg: format!("{}: {}", what.as_ref(), errno_text(e)),
            errno: Some(e.raw_os_error()),
            detail: None,
        }
    }

    pub fn io(e: std::io::Error, what: impl AsRef<str>) -> Self {
        match e.raw_os_error() {
            Some(n) => Error::sys(Errno::from_raw_os_error(n), what),
            // A bound of ours (tar-out's `max_bytes`), not the kernel's.
            None if e.kind() == std::io::ErrorKind::FileTooLarge => Error::new("too-large", format!("{}: {}", what.as_ref(), e)),
            None => Error::new("os", format!("{}: {}", what.as_ref(), e)),
        }
    }

    pub fn with_detail(mut self, detail: Value) -> Self {
        self.detail = Some(detail);
        self
    }

    pub fn is(&self, e: Errno) -> bool {
        self.errno == Some(e.raw_os_error())
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("kind".into(), json!(self.kind));
        m.insert("message".into(), json!(self.msg));
        if let Some(n) = self.errno {
            m.insert("errno".into(), json!(n));
        }
        if let Some(d) = &self.detail {
            m.insert("detail".into(), d.clone());
        }
        Value::Object(m)
    }
}

/// The kinds the host matches on. Links and mounts have kinds of their own,
/// because refusing them is the point of the file primitives.
pub fn errno_kind(e: Errno) -> &'static str {
    match e {
        Errno::LOOP => "link",
        Errno::XDEV => "mount",
        Errno::NOENT => "missing",
        Errno::EXIST => "exists",
        Errno::NOTDIR => "not-dir",
        Errno::ISDIR => "is-dir",
        Errno::NOTEMPTY => "not-empty",
        Errno::BUSY => "busy",
        Errno::ACCESS | Errno::PERM => "denied",
        Errno::SRCH => "gone",
        _ => "os",
    }
}

fn errno_text(e: Errno) -> String {
    let name = match e {
        Errno::LOOP => "ELOOP (a link, not followed)",
        Errno::XDEV => "EXDEV (another mount, not crossed)",
        Errno::NOENT => "ENOENT",
        Errno::EXIST => "EEXIST",
        Errno::NOTDIR => "ENOTDIR",
        Errno::ISDIR => "EISDIR",
        Errno::NOTEMPTY => "ENOTEMPTY",
        Errno::BUSY => "EBUSY",
        Errno::ACCESS => "EACCES",
        Errno::PERM => "EPERM",
        Errno::SRCH => "ESRCH",
        Errno::INVAL => "EINVAL",
        Errno::NOSYS => "ENOSYS",
        _ => "",
    };
    if name.is_empty() {
        format!("errno {}", e.raw_os_error())
    } else {
        name.to_string()
    }
}

impl From<Errno> for Error {
    fn from(e: Errno) -> Self {
        Error::sys(e, "system call")
    }
}
