# Lightweight document review

I want a small product where someone can invite a trusted person to review one document and leave useful comments. It should feel safe and focused rather than like giving away access to a whole workspace. Please shape the product and build the next coherent version.

# Task brief

I want a small product where someone can invite a trusted person to review one document and leave useful comments. It should feel safe and focused rather than like giving away access to a whole workspace. Please shape the product and build the next coherent version.

# Behavioral reference (concepts, not phrase hunt)

- Invite to a single document is a first-class entity; accepting grants focused review access (not whole-workspace admin).
- Accepted reviewers can leave comments that remain visible with their content.
- Expiry ends access: after expire, further comments must not appear; access/status reflects denied/expired.
- Revocation ends access: after revoke, further comments must not appear; access/status reflects denied/revoked.
- Private / blocked comment text must not leak into projections after access ends.
- Preferred product surface: a small reducer/state module (`createInitialState` / `reduce` / `project`) that can drive a simple UI.
