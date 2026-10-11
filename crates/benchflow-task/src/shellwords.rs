//! A service's `command` string split into words as Docker Compose splits
//! it, so a service runs the argv it ran under BenchFlow 0.8.
//!
//! 0.8 copied a service's `command` into a Compose file, and Compose
//! (compose-go v2, `types/command.go`, `ShellCommand.DecodeMapstructure`)
//! splits a string command with `shellwords.Parse` of
//! `github.com/mattn/go-shellwords`: the package's default parser, which
//! expands no variables, runs no backticks or `$(...)`, and reads no
//! comments. [`parse`] is a line-by-line port of that `Parser.Parse` from
//! go-shellwords v1.0.15, the version Docker Compose v5.6.0 builds in;
//! v1.0.14 (Compose v5.4.0 to v5.5.1) splits every string the same with
//! these settings. Other versions split some strings differently: of the
//! fixture's 7,836, v1.0.13 (Compose v5.3.1) gives another answer for 90,
//! v1.0.12 (likely older Compose v2 releases) for 280, and v1.0.16 (in no Compose
//! release yet) for 10, all around backslashes, backticks, `$(`, `\t`/`\n`,
//! and digits before `>`. It is not a POSIX shell:
//!
//! - A backslash outside single quotes takes the next character as is, inside
//!   double quotes too (`"a\.b"` is `a.b`), except that `\t` and `\n` are a
//!   tab and a newline. Inside single quotes a backslash is literal.
//! - `#` is an ordinary character: `--requirepass #x` keeps `#x`.
//! - A backtick or `$(...)` span stays in its word as written, with its
//!   quotes, backslashes, and spaces.
//! - Reading stops at a `;`, `&`, `|`, `<`, or `>` outside quotes: Compose
//!   runs the words before it and drops the rest, and before `>` it drops a
//!   word of digits too (the `2` of `2>&1`).
//! - Compose refuses the file for an unclosed quote, backtick, or `$(`, a `(`
//!   outside quotes that does not open `$(` (in v1.0.14 and v1.0.15 that
//!   includes a `$(` whose `$` follows a backslash in the word, escaped or
//!   not: `echo a\\$(date)` and `echo a\\\\$(date)` are refused), a `)`
//!   that closes none, and a lone backslash at the end.
//!
//! Compose replaces `$$` and `${VAR:-default}` in the string before it
//! splits it; [`parse`] splits the string as written. So the two can judge
//! a string with those next to a `(` differently: Compose v5.6.0 refuses
//! `'\'$$(x)` (it splits `'\'$(x)`), which `parse` accepts, and runs
//! `echo ${X:-$}(1)` as `echo $(1)`, which `parse` refuses. The fixture
//! holds go-shellwords' answers on the raw strings.
//!
//! `tests/fixtures/shellwords/go-shellwords-parse.jsonl` holds go-shellwords'
//! own answers for 7,836 strings; its generator is beside it.

use std::fmt;

/// What Compose makes of a command string it accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Split<'a> {
    /// The argv Compose runs; empty runs the image's own command.
    pub words: Vec<String>,
    /// The text from the shell operator Compose stopped at (from the word of
    /// digits before a `>`), which it drops: `> /dev/null`, `2>&1`.
    pub dropped: Option<&'a str>,
}

/// Why Compose refuses a command string. go-shellwords says only "invalid
/// command line string"; this names the cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    SingleQuote,
    DoubleQuote,
    Backtick,
    Subshell,
    OpenParen,
    CloseParen,
    TrailingBackslash,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Refusal::SingleQuote => "a single quote is not closed",
            Refusal::DoubleQuote => "a double quote is not closed",
            Refusal::Backtick => "a backtick is not closed",
            Refusal::Subshell => "a `$(` is not closed",
            Refusal::OpenParen => "a `(` outside quotes does not open a `$(`",
            Refusal::CloseParen => "a `)` outside quotes closes no `$(`",
            Refusal::TrailingBackslash => "it ends in a lone backslash",
        })
    }
}

/// go-shellwords' `argType`: no word yet, a word, or a word that ended a
/// quote (kept even when empty, as `''` is).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Got {
    No,
    Single,
    Quoted,
}

/// go-shellwords' `isSpace`.
fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

fn flush(args: &mut Vec<String>, buf: &mut String, got: &mut Got) {
    if *got == Got::Quoted || (*got != Got::No && !buf.is_empty()) {
        args.push(std::mem::take(buf));
    } else {
        buf.clear();
    }
    *got = Got::No;
}

/// Split `line` as Compose does: go-shellwords v1.0.15's `Parse` with
/// `ParseEnv`, `ParseBacktick`, and `ParseComment` off. The branches those
/// settings switch on, and the backtick buffer only they read, are left out.
pub(crate) fn parse(line: &str) -> Result<Split<'_>, Refusal> {
    let mut args = Vec::new();
    let mut buf = String::new();
    let mut escaped = false;
    let mut double_quoted = false;
    let mut single_quoted = false;
    let mut back_quote = false;
    let mut dollar_quote = false;
    let mut got = Got::No;
    // Where the word in `buf` began: after the last space that ended one.
    let mut word_start = 0;
    let mut dropped = None;

    for (i, r) in line.char_indices() {
        if escaped {
            escaped = false;
            if back_quote || dollar_quote {
                buf.push('\\');
                buf.push(r);
                got = Got::Single;
                continue;
            }
            buf.push(match r {
                't' => '\t',
                'n' => '\n',
                r => r,
            });
            got = Got::Single;
            continue;
        }

        if r == '\\' {
            if single_quoted {
                buf.push('\\');
            } else {
                escaped = true;
            }
            continue;
        }

        if is_space(r) {
            if single_quoted || double_quoted || back_quote || dollar_quote {
                buf.push(r);
            } else {
                flush(&mut args, &mut buf, &mut got);
                word_start = i + 1;
            }
            continue;
        }

        match r {
            '`' => {
                if !single_quoted && !double_quoted && !dollar_quote {
                    back_quote = !back_quote;
                }
            }
            ')' => {
                if !single_quoted && !double_quoted && !back_quote {
                    if !dollar_quote {
                        return Err(Refusal::CloseParen);
                    }
                    buf.push(')');
                    dollar_quote = false;
                    got = Got::Single;
                    continue;
                }
            }
            '(' => {
                if !single_quoted && !double_quoted && !back_quote {
                    let b = buf.as_bytes();
                    let n = b.len();
                    if !dollar_quote && n > 0 && b[n - 1] == b'$' && !(n > 1 && b[n - 2] == b'\\') {
                        dollar_quote = true;
                        buf.push('(');
                        continue;
                    }
                    return Err(Refusal::OpenParen);
                }
            }
            '"' => {
                if !single_quoted && !dollar_quote && !back_quote {
                    if double_quoted {
                        got = Got::Quoted;
                    }
                    double_quoted = !double_quoted;
                    continue;
                }
            }
            '\'' => {
                if !double_quoted && !dollar_quote && !back_quote {
                    if single_quoted {
                        got = Got::Quoted;
                    }
                    single_quoted = !single_quoted;
                    continue;
                }
            }
            ';' | '&' | '|' | '<' | '>'
                if !(single_quoted || double_quoted || back_quote || dollar_quote) =>
            {
                let mut from = i;
                if r == '>' && !buf.is_empty() && buf.bytes().all(|c| c.is_ascii_digit()) {
                    got = Got::No;
                    from = word_start;
                }
                dropped = Some(&line[from..]);
                break;
            }
            _ => {}
        }

        got = Got::Single;
        buf.push(r);
    }

    flush(&mut args, &mut buf, &mut got);

    if single_quoted {
        Err(Refusal::SingleQuote)
    } else if double_quoted {
        Err(Refusal::DoubleQuote)
    } else if back_quote {
        Err(Refusal::Backtick)
    } else if dollar_quote {
        Err(Refusal::Subshell)
    } else if escaped {
        Err(Refusal::TrailingBackslash)
    } else {
        Ok(Split {
            words: args,
            dropped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(line: &str) -> Vec<String> {
        parse(line).unwrap().words
    }

    /// Guards the fix of D5's split (commit 9fd29af9), which was checked
    /// against Python's `shlex.split`, a program 0.8 never ran on a service
    /// command. Every case in the fixture must split, or be refused, as
    /// go-shellwords v1.0.15 itself does (the generator is
    /// `tests/fixtures/shellwords/generator`).
    #[test]
    fn parse_matches_go_shellwords() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/shellwords/go-shellwords-parse.jsonl");
        let text = std::fs::read_to_string(path).unwrap();
        let (mut n, mut refused) = (0, 0);
        let mut wrong = Vec::new();
        for line in text.lines() {
            let case: serde_json::Value = serde_json::from_str(line).unwrap();
            let input = case["text"].as_str().unwrap();
            let want: Option<Vec<String>> = match case.get("error") {
                Some(e) => {
                    assert_eq!(e, "invalid command line string");
                    refused += 1;
                    None
                }
                None => Some(serde_json::from_value(case["words"].clone()).unwrap()),
            };
            let got = parse(input).ok().map(|s| s.words);
            if got != want {
                wrong.push(format!("{input:?}: got {got:?}, Compose gives {want:?}"));
            }
            n += 1;
        }
        assert!(
            wrong.is_empty(),
            "{} of {n} differ:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
        assert!(n >= 7836, "{n} cases");
        assert!(
            refused > 1000 && n - refused > 1000,
            "{refused} of {n} refused"
        );
    }

    /// The review of D5's split: what Compose ran, where `shlex.split`
    /// differs.
    #[test]
    fn the_reviewed_cases_split_as_compose_splits_them() {
        assert_eq!(
            words("redis-server --save '' --requirepass #hunter2"),
            ["redis-server", "--save", "", "--requirepass", "#hunter2"]
        );
        assert_eq!(
            words(r#"sh -c "echo \`date\`""#),
            ["sh", "-c", "echo `date`"]
        );
        assert_eq!(
            words(r#"sh -c "grep -E 'a\.b' f""#),
            ["sh", "-c", "grep -E 'a.b' f"]
        );
        assert_eq!(words(r#"printf "a\tb\n""#), ["printf", "a\tb\n"]);
        assert_eq!(words(r"echo 'a\tb'"), ["echo", r"a\tb"]);
        let cut = parse("redis-server --save 60 1 > /dev/null").unwrap();
        assert_eq!(cut.words, ["redis-server", "--save", "60", "1"]);
        assert_eq!(cut.dropped, Some("> /dev/null"));
        let cut = parse("cmd --flag 2>&1").unwrap();
        assert_eq!(cut.words, ["cmd", "--flag"]);
        assert_eq!(cut.dropped, Some("2>&1"));
        assert_eq!(parse("cmd a2>x").unwrap().words, ["cmd", "a2"]);
        assert_eq!(parse("sleep infinity").unwrap().dropped, None);
        assert_eq!(words("echo $(date) `id`"), ["echo", "$(date)", "`id`"]);
    }

    #[test]
    fn refusals_name_their_cause() {
        for (line, why) in [
            ("python -c print(1)", Refusal::OpenParen),
            ("echo $(echo (x))", Refusal::OpenParen),
            (r"echo a\\$(date)", Refusal::OpenParen),
            // Two literal backslashes before `$(` (the D5b review of
            // cbf42071, finding 7): v1.0.14 and v1.0.15 refuse it too.
            (r"echo a\\\\$(date)", Refusal::OpenParen),
            (r"'\\'$(x)", Refusal::OpenParen),
            ("echo )", Refusal::CloseParen),
            ("echo 'a", Refusal::SingleQuote),
            ("echo \"a\\", Refusal::DoubleQuote),
            ("echo `a", Refusal::Backtick),
            ("echo $(a", Refusal::Subshell),
            ("echo a\\", Refusal::TrailingBackslash),
        ] {
            assert_eq!(parse(line), Err(why), "{line:?}");
        }
        // Compose stops at the operator before it reads the rest.
        assert_eq!(words("echo a; echo ("), ["echo", "a"]);
    }
}
