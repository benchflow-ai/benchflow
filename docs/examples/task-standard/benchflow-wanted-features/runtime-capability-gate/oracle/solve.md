# Oracle Implementation Sketch

The reference solution should introduce a small task runtime module, likely
`src/benchflow/task/runtime_view.py` plus
`src/benchflow/task/runtime_capabilities.py`.

The runtime view must be the only place that answers which native or
compatibility files are authoritative. Capability validation should be a pure
function first, then wired into rollout setup and `bench tasks check --sandbox`.

