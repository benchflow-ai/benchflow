//! "Did you mean" suggestions that match task.md's reference checker.
//!
//! The reference tools use Python's `difflib.get_close_matches(word,
//! possibilities, n=1, cutoff=0.6)`, which ranks by
//! `SequenceMatcher(None, candidate, word).ratio()` (it sets the word as the
//! second sequence; the ratio is not symmetric). This module implements
//! the same ratio (Ratcliff/Obershelp longest matching blocks, with difflib's
//! "autojunk" heuristic, which only matters for strings of 200 or more
//! characters) so a suggestion here is the suggestion there.

use std::collections::HashMap;

/// `difflib.SequenceMatcher(None, a, b).ratio()`.
pub fn ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    let matches = Matcher::new(&a, &b).matching_chars();
    2.0 * matches as f64 / total as f64
}

/// `difflib.get_close_matches(word, possibilities, n, cutoff)`: the best `n`
/// candidates with a ratio of at least `cutoff`, best first. Ties keep
/// difflib's order (higher ratio first, then the later candidate in sorted
/// input order, since difflib uses a max-heap of (score, word)).
pub fn close_matches<'a, I>(word: &str, possibilities: I, n: usize, cutoff: f64) -> Vec<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut scored: Vec<(f64, &str)> = Vec::new();
    for cand in possibilities {
        // difflib checks real_quick_ratio and quick_ratio first; they are
        // upper bounds, so skipping them changes nothing but speed.
        let r = ratio(cand, word);
        if r >= cutoff {
            scored.push((r, cand));
        }
    }
    // heapq.nlargest(n, result) over (score, word) tuples: by score, then by
    // word, both descending.
    scored.sort_by(|x, y| {
        y.0.partial_cmp(&x.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| y.1.cmp(x.1))
    });
    scored
        .into_iter()
        .take(n)
        .map(|(_, w)| w.to_string())
        .collect()
}

/// The best close match with difflib's default cutoff, as the reference
/// checker asks for it.
pub fn did_you_mean<'a, I>(word: &str, possibilities: I) -> Option<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut sorted: Vec<&str> = possibilities.into_iter().collect();
    sorted.sort_unstable();
    close_matches(word, sorted, 1, 0.6).into_iter().next()
}

struct Matcher<'s> {
    a: &'s [char],
    b: &'s [char],
    b2j: HashMap<char, Vec<usize>>,
}

impl<'s> Matcher<'s> {
    fn new(a: &'s [char], b: &'s [char]) -> Self {
        let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
        for (i, &c) in b.iter().enumerate() {
            b2j.entry(c).or_default().push(i);
        }
        // difflib's autojunk: in sequences of 200+ items, an item occurring in
        // more than 1% of positions (+1) is junk.
        let n = b.len();
        if n >= 200 {
            let ntest = n / 100 + 1;
            b2j.retain(|_, idxs| idxs.len() <= ntest);
        }
        Self { a, b, b2j }
    }

    /// difflib's find_longest_match on a[alo..ahi], b[blo..bhi], with no junk
    /// function (so only popular-element pruning applies).
    fn longest_match(
        &self,
        alo: usize,
        ahi: usize,
        blo: usize,
        bhi: usize,
    ) -> (usize, usize, usize) {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = self.b2j.get(&self.a[i]) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j2len.get(&(j.wrapping_sub(1))).copied().unwrap_or(0) + 1;
                    let k = if j == 0 { 1 } else { k };
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }
        // Extend the match with equal elements that were pruned as popular
        // (difflib extends over non-junk, then junk; with no isjunk both
        // loops compare plain equality of elements outside b2j).
        while besti > alo
            && bestj > blo
            && !self.b2j.contains_key(&self.b[bestj - 1])
            && self.a[besti - 1] == self.b[bestj - 1]
        {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && !self.b2j.contains_key(&self.b[bestj + bestsize])
            && self.a[besti + bestsize] == self.b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    }

    fn matching_chars(&self) -> usize {
        let mut total = 0;
        let mut queue = vec![(0, self.a.len(), 0, self.b.len())];
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let (i, j, k) = self.longest_match(alo, ahi, blo, bhi);
            if k > 0 {
                total += k;
                if alo < i && blo < j {
                    queue.push((alo, i, blo, j));
                }
                if i + k < ahi && j + k < bhi {
                    queue.push((i + k, ahi, j + k, bhi));
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values from CPython 3.12 difflib.
    #[test]
    fn ratios_match_difflib() {
        assert!((ratio("timout", "timeout") - 0.9230769230769231).abs() < 1e-12);
        assert!((ratio("budgett", "budget") - 0.9230769230769231).abs() < 1e-12);
        assert!((ratio("abcd", "bcde") - 0.75).abs() < 1e-12);
        assert!((ratio("", "") - 1.0).abs() < 1e-12);
        assert!((ratio("network_mode", "network") - 0.7368421052631579).abs() < 1e-12);
    }

    #[test]
    fn suggests_like_get_close_matches() {
        let keys = ["timeout", "on_timeout", "budget", "user", "network"];
        assert_eq!(did_you_mean("timout", keys), Some("timeout".into()));
        assert_eq!(did_you_mean("netwrk", keys), Some("network".into()));
        assert_eq!(did_you_mean("zzz", keys), None);
    }

    /// difflib scores the candidate as the first sequence:
    /// `SequenceMatcher(None, "harbor-compatible", word)` is 0.571 for this
    /// word, under the cutoff, while the other order gives 0.619.
    #[test]
    fn the_candidate_is_the_first_sequence_as_in_difflib() {
        let word = "terminal-bench-compatible";
        assert!((ratio("harbor-compatible", word) - 0.5714285714285714).abs() < 1e-12);
        assert_eq!(did_you_mean(word, ["harbor-compatible"]), None);
    }
}
