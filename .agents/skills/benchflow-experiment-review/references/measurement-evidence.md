# Timing, usage and cost evidence

Start from raw events and the exact parser/source revision, not dashboard totals.

## Timing

- Specify each metric's boundaries. CLI launch to ready, provider creation,
  agent install, execution, verification, teardown and final resource confirmation
  differ. Match boundaries across arms; report work outside the interval.
- Distinguish elapsed monotonic clocks from epoch clocks. Epoch timestamps can
  jump; do not call them monotonic without source evidence. Record clock domains,
  synchronization and uncertainty for events from separate hosts/sandboxes.
- Account for overlap. Parallel model calls/tool intervals cannot simply be
  summed and subtracted from wall time. Use interval unions and a declared rule
  for model/tool overlap. Missing or opaque events remain missing evidence.
- Verify phase marks independently. A `timing_complete` flag may cover model/tool
  capture while some lifecycle phases are unavailable. Give each phase its
  actual admitted denominator.
- Preserve intended populations and excluded identities/reasons, including
  failed starts, timeouts and missing terminal events. Conditional timings among
  admitted trials do not establish full-population performance.
- Compute paired differences on matched admitted identities. Difference of
  marginal medians is a different statistic. Phase medians are not additive,
  especially when their populations differ.
- Match resampling to the design: retain repeated trials within a task when
  bootstrapping tasks. Record order, seed, resamples, estimator, sample size and
  interval method. A confidence interval does not remove missing-data bias or
  prove equivalent solve rates.
- Separate schedule length, worker limits, overlapping process intervals and
  observed live sandboxes. Record load/resources, warm/cold setup, context size,
  dependency installs, cache state and placement. Do not pool different loads.
- Source inspection or a profile localizes work; a matched ablation or fresh
  exact-build comparison tests an optimization. Separate predicted from measured
  savings and preserve security and cleanup requirements.

## Usage and cost

Trace provider response/native OTel event → capture/flush → stored request and
attempt identity → normalization → pricing → export → report. Check joins
against raw records. An empty report field does not prove capture was absent;
a captured event does not prove aggregation or pricing was correct.

Distinguish:

- **Provider-billed spend:** invoice, billing meter or scoped balance change.
  Balance changes can include probes, helpers and unrelated requests. State scope.
- **Native reported cost:** preserve emitter/model/event fields. A client may
  estimate list price while using a subscription or another provider; it is not
  automatically a bill.
- **Estimated cost:** usage times dated, sourced rates. Record currency, units,
  model alias resolution and applicable service/cache tiers.
- **Unknown/partial cost:** retain known subtotal and unpriced request/model
  counts separately; never label a subtotal a complete total. Missing pricing,
  missing usage and missing capture are distinct. Missing is not zero.

Read provider schemas before normalizing tokens: does input include cache reads,
does output include reasoning, and how are cache writes, tools, batching and
service tiers billed? Do not count cache tokens twice, impose one provider's
convention universally or silently clamp inconsistent counts. A model-family
name is not sufficient price provenance.

For OTel and other event streams, check:

- run/trial/attempt/request/model joins, including subagents and helper calls;
- coverage, exporter drop/error diagnostics, final flush, shutdown, timeout and
  cancellation;
- retransmission, replay and deduplication before aggregation;
- counter versus delta semantics and time period;
- retries/provider errors, preserving returned usage and billing uncertainty;
- credential/content redaction without losing accounting identities;
- consumers reading authoritative records and surfacing partial/unknown data.

OpenTelemetry is instrumentation and transport, not a universal billing oracle.
Usage/cost events do not establish full request/response capture. Product
analytics opt-out is separate from authorized experiment trace collection;
verify settings and propagate them explicitly across remote/container boundaries.

Use bounded real-model controls for changed routes. Join a raw provider/native
event to its stored record, normalized usage, selected rate, rollout export and
report. Add a focused negative control for the observed failure, such as an
unpriced helper, duplicate retry, missing flush or inclusive cached input.
Keep unsupported routes and untested cases explicitly uncovered.
