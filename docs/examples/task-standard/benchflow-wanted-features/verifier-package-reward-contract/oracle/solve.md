# Oracle Implementation Sketch

Add a `VerifierDocument` parser beside `TaskDocument`, then teach verifier
resolution to select `verifier/verifier.md` when present. Keep root
`VerifierConfig` as runtime/compatibility configuration, not as the full native
verifier package.

The reward parser should prefer `reward.json`, preserve multi-metric maps,
require agreement with `reward.txt` when a scalar aggregate exists, and copy
`reward-details.json` into result artifacts.

