# Oracle Implementation Sketch

The reference solution should validate and extend the landed runtime modules:
`src/benchflow/task/runtime_view.py`, `src/benchflow/task/package.py`, and
`src/benchflow/task/runtime_capabilities.py`.

`TaskPackage` should be the package-level place that answers which native or
compatibility files are authoritative and what runtime issues are known for a
selected sandbox. Capability validation should remain a pure function, then be
wired into rollout setup and `bench tasks check --sandbox`.
