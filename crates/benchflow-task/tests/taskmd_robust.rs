//! The checker never panics: every fixture's task.md and rubric.json,
//! mutated at random (deterministically), still checks to a report.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use benchflow_task::check::CheckOptions;
use benchflow_task::taskmd;

/// xorshift64*: small, fast, and the same everywhere.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

const SNIPPETS: &[&str] = &[
    "[",
    "]",
    "\"",
    "'",
    "=",
    "```",
    "~~~",
    "\n",
    "\r\n",
    "\r",
    "{",
    "}",
    ",",
    ".",
    "x-",
    "\t",
    "é",
    "🙂",
    "${HOME}",
    "[[a]]\n",
    "\n```toml task\n",
    "\n```stage s\n",
    "<!-- canary -->\n",
    "\\",
    "1e999",
    "-0",
    "null",
    "true",
    "[sandbox]\n",
    "\u{feff}",
];

/// One random edit of `text`, on character boundaries.
fn mutate(text: &str, rng: &mut Rng) -> String {
    let chars: Vec<char> = text.chars().collect();
    let at = rng.below(chars.len() + 1);
    let mut out: Vec<char> = chars.clone();
    match rng.below(4) {
        0 => {
            let end = (at + 1 + rng.below(12)).min(chars.len());
            out.drain(at.min(end)..end);
        }
        1 => {
            let snippet = SNIPPETS[rng.below(SNIPPETS.len())];
            out.splice(at..at, snippet.chars());
        }
        2 => {
            let lines: Vec<&str> = text.split('\n').collect();
            let i = rng.below(lines.len());
            let mut l: Vec<&str> = lines.clone();
            l.insert(i, lines[i]);
            return l.join("\n");
        }
        _ => {
            let lines: Vec<&str> = text.split('\n').collect();
            let (i, j) = (rng.below(lines.len()), rng.below(lines.len()));
            let mut l = lines.clone();
            l.swap(i, j);
            return l.join("\n");
        }
    }
    out.into_iter().collect()
}

fn copy_dir(from: &Path, to: &Path) {
    for entry in walkdir::WalkDir::new(from).into_iter().flatten() {
        let rel = entry.path().strip_prefix(from).unwrap();
        let dest = to.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&dest).unwrap();
        } else if entry.file_type().is_file() {
            std::fs::copy(entry.path(), &dest).unwrap();
        }
    }
}

fn packages() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/taskmd");
    let mut out = Vec::new();
    for group in ["examples", "imported", "cases"] {
        for entry in std::fs::read_dir(root.join(group)).unwrap().flatten() {
            out.push(entry.path());
        }
    }
    out.sort();
    out
}

/// Mutants per package; `TASKMD_FUZZ_ROUNDS` searches harder.
fn rounds() -> usize {
    std::env::var("TASKMD_FUZZ_ROUNDS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(6)
}

#[test]
fn checking_never_panics() {
    let mut rng = Rng(0x5eed_1234_abcd_0042);
    let mut checked = 0;
    for pkg in packages() {
        let dir = tempfile::tempdir().unwrap();
        copy_dir(&pkg, dir.path());
        let task = std::fs::read_to_string(pkg.join("task.md")).unwrap_or_default();
        let rubric = std::fs::read_to_string(pkg.join("verifier/rubric.json")).ok();
        for round in 0..rounds() {
            let mut text = task.clone();
            for _ in 0..1 + rng.below(3) {
                text = mutate(&text, &mut rng);
            }
            std::fs::write(dir.path().join("task.md"), &text).unwrap();
            if let Some(r) = &rubric
                && round % 2 == 1
            {
                std::fs::write(dir.path().join("verifier/rubric.json"), mutate(r, &mut rng))
                    .unwrap();
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                let report = taskmd::check(dir.path(), &CheckOptions { deep: true });
                let _ = taskmd::parse(&text, Path::new("task.md"));
                let _ = taskmd::agent_view(dir.path());
                report
            }));
            assert!(
                result.is_ok(),
                "{} panicked on task.md:\n{text}",
                pkg.display()
            );
            checked += 1;
        }
    }
    assert!(checked > 1000);
}
