# User Lookup Response Bug

Fix `handleUserLookup` in `src/users.js`.

Requirements:

- Distinguish missing input, missing users, and successful lookups with
  appropriate HTTP-style statuses.
- Keep successful responses and error responses predictable for callers.
- Error responses should have stable machine-readable and human-readable
  fields.

Do not add external dependencies.

Work in the provided repository. Do not add external dependencies.
