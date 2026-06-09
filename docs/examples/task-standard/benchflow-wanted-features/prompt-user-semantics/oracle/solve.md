# Oracle Implementation Sketch

Add prompt composition support to the task document/runtime layer. The default
for current behavior can remain fallback for compatibility, but native tasks
that declare `benchflow.prompt.composition: append` should materialize base,
role, scene, and turn prompts in the declared order.

Document-declared `user` and `benchflow.nudges` should compile into
`RolloutConfig.user` for supported linear single- or multi-scene loops.
The first supported `benchflow.teams` runtime should be a narrow sequential
shared-workspace handoff subset: explicit multi-role scene turns execute in
order through the normal role switching path, and round logs include scene,
role, and handoff metadata.
Human confirmation policy should install a fail-closed ACP permission handler
unless the caller supplied an explicit `on_ask_user` handler. Branchable
`ask_user` requests should preserve both option IDs and option kinds so future
branch scoring can distinguish reject/allow semantics without parsing raw ACP
payloads. Full branching, interactive approval UI, parallel team execution,
full trajectory sharing, handoff artifacts, unsupported models, or malformed
metadata should be rejected by sandbox-aware validation rather than ignored.
