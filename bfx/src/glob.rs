//! Path patterns for `walk`: `*` and `?` stay within one component, `**`
//! matches any number of whole components. No classes, no escapes.

pub fn matches(pat: &[u8], path: &[u8]) -> bool {
    let p: Vec<&[u8]> = pat.split(|b| *b == b'/').collect();
    let s: Vec<&[u8]> = path.split(|b| *b == b'/').collect();
    comps(&p, &s)
}

fn comps(p: &[&[u8]], s: &[&[u8]]) -> bool {
    match p.split_first() {
        None => s.is_empty(),
        Some((first, rest)) if *first == b"**" => (0..=s.len()).any(|k| comps(rest, &s[k..])),
        Some((first, rest)) => match s.split_first() {
            Some((name, srest)) => one(first, name) && comps(rest, srest),
            None => false,
        },
    }
}

fn one(p: &[u8], s: &[u8]) -> bool {
    match p.split_first() {
        None => s.is_empty(),
        Some((b'*', rest)) => (0..=s.len()).any(|k| one(rest, &s[k..])),
        Some((b'?', rest)) => !s.is_empty() && one(rest, &s[1..]),
        Some((c, rest)) => s.first() == Some(c) && one(rest, &s[1..]),
    }
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn patterns() {
        assert!(matches(b"**/conftest.py", b"conftest.py"));
        assert!(matches(b"**/conftest.py", b"a/b/conftest.py"));
        assert!(!matches(b"*/conftest.py", b"a/b/conftest.py"));
        assert!(matches(b"*.pth", b"x.pth"));
        assert!(!matches(b"*.pth", b"d/x.pth"));
        assert!(matches(b"a/**", b"a/b/c"));
        assert!(matches(b"a?c", b"abc"));
        assert!(!matches(b"a?c", b"ac"));
    }
}
