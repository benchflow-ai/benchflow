//! Every diagnostic code, gathered from the modules that emit them.

use crate::diag::Explanation;

/// All explanations, in code order.
pub fn all() -> Vec<&'static Explanation> {
    let mut v: Vec<&'static Explanation> = [
        crate::taskmd::EXPLANATIONS,
        crate::split::EXPLANATIONS,
        crate::v06::EXPLANATIONS,
        crate::load::EXPLANATIONS,
        crate::compose::EXPLANATIONS,
    ]
    .iter()
    .flat_map(|s| s.iter())
    .collect();
    v.sort_by_key(|e| e.code);
    v
}
