"""benchflow.experimental — unstable features.

Anything under this package is NOT part of the semver-stable surface. APIs
may change or be removed in any minor release. Users must opt in explicitly
via ``from benchflow.experimental.<feature> import ...`` — there are no
re-exports at package level and no names flow into ``benchflow.__init__``.

Graduation criteria (move out of experimental/ into a top-level module or
contracts/): at least one month of soak, plus three external callers OR an
explicit keep decision, plus a Quint spec + bridge test under
``sandbox/v2/spec/`` and ``sandbox/v2/tests/``.

Removal criteria (delete the feature): no caller growth after one month, or
an unresolved design fork after two spec attempts, or a spec surfaces an
invariant nobody wants to commit to. Single-commit delete — the one-file +
no-re-export discipline keeps the blast radius small.
"""
