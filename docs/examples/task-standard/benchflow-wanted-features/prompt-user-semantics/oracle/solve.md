# Oracle Implementation Sketch

Add prompt composition support to the task document/runtime layer. The default
for current behavior can remain fallback for compatibility, but native tasks
that declare `benchflow.prompt.composition: append` should materialize base,
role, scene, and turn prompts in the declared order.

Document-declared `user` and `benchflow.nudges` should either compile into
`RolloutConfig.user` and the existing user loop, or be rejected by
sandbox-aware validation as metadata-only.

