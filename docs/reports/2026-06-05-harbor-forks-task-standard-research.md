# Harbor Forks And Task Standard Research

Date: 2026-06-05

## Summary

The research supports a clear direction:

- Keep `task.md` as BenchFlow's native authoring document.
- Keep Harbor/Pier split layouts as import/export compatibility targets.
- Use `oracle/` and `verifier/` natively, exporting them as `solution/` and
  `tests/` when needed.
- Separate authoring, runtime, and adapter views before declaring `task.md` an
  interchange standard.
- Add fail-closed runtime capability checks for parsed-but-unsupported fields.

The requested pool of ten simultaneous subagents hit a platform limit below ten
active threads. I maintained the maximum available pool, replaced completed
lanes, and accepted more than ten total research agents across the run.

## Harbor Upstream

Current Harbor keeps a split package:

```text
task.toml
instruction.md
environment/
solution/
tests/
```

Important Harbor concepts for BenchFlow:

| Concept | Harbor signal | BenchFlow response |
|---|---|---|
| schema version | current schema shape is `1.3` | Keep `schema_version` rooted in `TaskConfig` |
| network policy | `no-network`, `public`, `allowlist` plus phase overrides | Current: parse now. Needed: fail closed until each sandbox proves it honors the policy |
| verifier env | shared or separate verifier environment | Do not hide under `tests/`; model verifier runtime |
| multi-step tasks | `steps[]` with per-step verifier/reward/artifacts | Keep orthogonal from `scenes` |
| reward files | `/logs/verifier/reward.txt` and optional JSON | Preserve scalar plus structured evidence |
| artifacts | root and step artifact declarations | Proposed: materialize artifact declarations or reject them per runtime; current runtime enforcement is incomplete |

Harbor's split format is still the safest interchange target. BenchFlow should
export to it, not force other tools to ingest `task.md`.

## Pier / DeepSWE

Pier is Harbor-compatible by design, but it stresses different needs:

- smaller, opinionated agent set
- installed agents for air-gapped tasks
- augmented ATIF trajectories and critique viewer
- environments such as Docker and Modal
- network allowlists derived from agent configs
- `solution/` and `tests/` vocabulary remains externally visible

Takeaway: `oracle/` and `verifier/` are better native names. Future exporters
must emit Pier-compatible `solution/` and `tests/`; preserving unknown Pier
fields in compatibility metadata is proposed, not current adapter behavior.

## FrontierSWE `harbor_ext`

FrontierSWE does not replace Harbor's task abstraction. It keeps Harbor-shaped
tasks and adds execution plumbing:

- Modal-managed environments
- private images and GHCR credentials
- GPU and large-resource task support
- domain/firewall allowlists
- persistent trial volumes
- long timeout management
- no-search variants of CLI agents
- richer task-specific `reward.json` payloads

Takeaway: the task standard must not flatten evidence to a single float. The
runtime view should preserve `/logs/agent`, CLI-native trajectories, scalar
reward, structured reward, and task-specific artifacts.

## Harbor Fork Inventory

GitHub shows roughly 1.1k public forks. The inspected high-signal forks mostly
show integration/runtime divergence rather than task-format invention.

| Fork | Signal | Task-standard implication |
|---|---|---|
| `alibaba/harbor` | image builder and registry utilities | package assets need image provenance |
| `LLM360/harbor` | Kubernetes/SQS/sglang/runtime changes | environment backends vary faster than task docs |
| `JinnanDuan/bitfun-harbor` | BitFun agent integration and viewer/API changes | agents and trajectory metadata need extension points |
| `Chakra-Network/marina` | active renamed fork with targeted OpenHands SDK runner changes | native standard should not assume one agent harness |
| `invisible-tools/harbor-TS` | TypeScript port plan | format needs language-neutral spec, not Python classes |
| `JaneliaScientificComputingSystems/harbor` | Podman and HPC/LSF scripts | resource/backend fields need capability negotiation |
| `formula-code/harbor` | benchmark adapter fork | imports/exports must carry source provenance |
| `sentient-agi/harbor` | OfficeQA adapter/publish workflows | task packages need benchmark lineage metadata |
| `KooshaPari/portage` | large runtime/provider divergence | adapter view should isolate provider concerns |
| `refreshdotdev/harbor-mm` | multimodal trajectory viewer changes | evidence/artifact plane must handle screenshots/video |

Watch-only org forks such as `codesandbox`, `tensorzero`, `letta-ai`, and
`NousResearch` were visible but did not show substantive public `main`
divergence in the inspected comparison.

## Fork Needs To Standard Traceability

The abstraction only matters if it makes fork maintainers stop carrying local
patches for the same unmet needs. This is the user-need map behind the draft
standard.

| Fork / community | User need | Current support | Target response | Proof target |
|---|---|---|---|---|
| Harbor upstream | Keep the existing task authoring surface stable | Harbor-shaped config parses; split layout loads | Keep root config compatible and add fail-closed runtime/export gates | native-to-Harbor round trip has canonical config/prompt/oracle/verifier equivalence |
| Pier / DeepSWE | Run Harbor-format tasks with installed agents, no-internet defaults, allowlisted egress, and critique traces | Known Harbor-compatible fields load | Future exporters map native `oracle/` / `verifier/` to `solution/` / `tests/`; `benchflow:` carries critique/evidence/runtime metadata | no-network and no-skill trajectories prove isolation; Pier export lists no Harbor-compatible losses |
| FrontierSWE | Run heavy no-search frontier tasks with private images, Modal, GPU resources, persistent state, and rich scoring | Resource fields parse; runtime policy is metadata-only | Runtime policy, assets, backend capability gates, and structured reward/evidence become target standard concepts | unsupported backend features fail closed; supported runs preserve private asset metadata and `reward.json` evidence |
| Alibaba / image-provider forks | Build and publish task images across registries/providers | Image fields fit config/metadata | Asset/provenance plane tracks images, hashes, registries, and provider metadata | result metadata records image hash, registry, provider, and build/source revision |
| LLM360 / Kubernetes providers | Queue work across Kubernetes/SQS/model-serving backends | Provider fields are not native runtime support | Runtime view separates backend capabilities from static authoring and fails closed | Kubernetes/SQS fields are backend capabilities, not accepted silently by non-supporting sandboxes |
| BitFun / agent-wrapper forks | Add new agent launchers and viewer/API affordances | `agents.roles` handles harness selection | `benchflow:` carries viewer/trajectory metadata | viewer metadata and role-to-agent mapping survive parser, rollout metadata, and export loss reports |
| Marina / OpenHands forks | Wire OpenHands SDK and distribution-specific runners | Adapters/runners coexist with task loading | Keep foreign adapter view separate so runner behavior is not baked into `task.md` | OpenHands-specific launch data lives in adapter/runtime metadata, not root schema keys |
| TypeScript port | Implement task tooling outside Python | `task.md` is static markdown/YAML | Keep schema language-neutral; Python models are implementation details | schema fixtures validate with a non-Python parser without Python class-name assumptions |
| Janelia HPC / Podman | Run on Podman/HPC/LSF instead of standard Docker | Resources parse but backend support varies | Runtime capability matrix and backend checks make provider support explicit | Podman/HPC fields produce explicit support/unsupported diagnostics per backend |
| FormulaCode / Sentient adapters | Publish new benchmark datasets and adapters | Source metadata exists | Source provenance, degraded exports, and compatibility reports become package metadata | imports/exports carry dataset source, source hash, and degraded-export report |
| Refresh multimodal/CUA | Preserve screenshots, video, and UI trajectories | Artifacts/log paths exist | Evidence/artifact plane supports multimodal outputs and trajectory references | screenshot/HAR/video/trajectory artifacts are declared and emitted in result metadata |
| Portage / provider-divergent forks | Large runtime/provider divergence without new task semantics | Provider concerns live outside root task config | Keep provider concerns in runtime/adapters rather than root task syntax | provider-specific divergence does not add root keys unless tied to shared F1-F6 needs |
| Watch-only forks | No substantive public task-format divergence observed | No standard fields added for them | No new fields without a concrete user need | field proposals cite a real need or are rejected |
| METR-style task developers | Need assets, secrets, protected scoring, and QA evidence | `benchflow:` can carry raw metadata | Add typed asset, secret/resource, and validation evidence extensions | oracle/no-op/flake/anti-cheat validation evidence exists before publication |

## Adjacent Standards

| Standard | Shape | What BenchFlow should use |
|---|---|---|
| Terminal-Bench | `task.yaml`, Docker/Compose, `solution.*`, `run-tests.sh`, tests copied after agent run | verifier isolation and simple export target |
| SWE-bench | dataset row: repo, base commit, issue, gold patch, test patch, pass/fail test sets | provenance, hidden test patch, repo snapshot identity |
| SWE-agent | env plus problem statement, `.traj` with actions/observations/stats | trajectory as evidence, not just logs |
| OpenHands evals | eval instance, metadata, output/history/metrics/errors | run metadata and error taxonomy |
| Inspect | task = dataset + solver + scorer + sandbox | clean separation of task, solver, scorer, sandbox |
| OSWorld | desktop snapshot, setup config, apps, evaluator | environment state/reset and action/observation policy |
| WebArena / Verified | site fleet, auth state, start URLs, deterministic structural checks | browser auth/assets, HAR/screenshot traces, deterministic verifiers |
| MiniWoB / BrowserGym | Gym-style setup/validate/teardown, terminal reward | step rewards and simulator action spaces |
| METR task standard | task validity, assets, scoring/evidence discipline | stronger evidence and asset manifest requirements |

The common pattern is not a single file format. It is a package with:

- prompt or task intent
- reproducible environment state
- hidden or held-out verifier
- oracle/reference when available
- provenance and assets
- trajectory/evidence output
- adapter-specific metadata

## Source Reference Table

| Area | Source | Source fact | BenchFlow inference |
|---|---|---|---|
| Harbor upstream | [Harbor tasks](https://www.harborframework.com/docs/tasks), [multi-step](https://www.harborframework.com/docs/tasks/multi-step) | Harbor documents split task layout, network policy, verifier environments, multi-step tasks, rewards, and artifacts. | Keep root config Harbor-compatible and treat split layout as the primary export target. |
| BenchFlow Harbor adapter | [harbor.py](/Users/lixiangyi/.codex/worktrees/50ff/benchflow/src/benchflow/adapters/harbor.py:1), [inbound.py](/Users/lixiangyi/.codex/worktrees/50ff/benchflow/src/benchflow/adapters/inbound.py:1) | Current code still treats Harbor-shaped directories as adapter/native-shaped input. | The proposed standard should rename this as foreign-layout compatibility, not native authoring vocabulary. |
| Pier / DeepSWE | [Pier](https://github.com/datacurve-ai/pier), [DeepSWE](https://github.com/datacurve-ai/deep-swe) | Pier is Harbor-compatible and DeepSWE publishes long-horizon Harbor-format tasks. | Preserve Harbor/Pier export and avoid `task.md`-only interchange. |
| FrontierSWE | [frontier-swe](https://github.com/Proximal-Labs/frontier-swe), [harbor_ext](https://github.com/Proximal-Labs/frontier-swe/tree/main/harbor_ext) | FrontierSWE depends on Harbor and extends runtime/network/agent behavior through `harbor_ext`. | Model backend capabilities, private assets, no-search policy, and rich evidence without replacing Harbor task shape. |
| Fork inventory | [GitHub forks API](https://api.github.com/repos/harbor-framework/harbor/forks?sort=stargazers&per_page=10) | Inspected high-signal forks show image, provider, agent, HPC, adapter, and viewer deltas. | Standard fields need fork-user traceability; provider-specific concerns should stay in runtime/adapters. |
| Terminal-Bench | [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | Terminal-Bench uses flat task metadata plus verifier/oracle scripts; BenchFlow has an inbound adapter. | Treat Terminal-Bench as an adapter/export target, not the native schema. |
| SWE-bench | [SWE-bench](https://github.com/SWE-bench/SWE-bench) | SWE-bench evaluates repo patches against issue/test metadata in Docker. | Model repo provenance, base commits, hidden test patches, and patch artifacts explicitly. |
| SWE-agent / OpenHands | [SWE-agent](https://github.com/SWE-agent/SWE-agent), [OpenHands](https://github.com/OpenHands/OpenHands) | These are agent/runtime harnesses with trajectories and SWE-bench conversion. | Use their trajectories/eval outputs as evidence inputs, not as task package syntax. |
| Inspect | [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) | Inspect separates dataset, solver, scorer, and sandbox in Python task definitions. | Borrow the separation of planes while keeping BenchFlow document-first. |
| OSWorld | [OSWorld](https://github.com/xlang-ai/OSWorld) | OSWorld tasks combine desktop state/setup, instruction, observations, actions, and evaluator getters. | Put desktop state and verifier getters in runtime/verifier planes rather than scenes. |
| WebArena / BrowserGym | [WebArena](https://github.com/web-arena-x/webarena), [BrowserGym](https://github.com/ServiceNow/BrowserGym) | Browser tasks need sites, auth/start URLs, action/observation spaces, traces, and validation. | Add browser/desktop environment metadata and deterministic verifier concepts under `benchflow:`. |
| Harbor Reward Kit | [Reward Kit](https://www.harborframework.com/docs/rewardkit), [judge criteria](https://www.harborframework.com/docs/rewardkit/judge-criteria) | Reward Kit packages programmatic criteria, judge criteria, multi-reward JSON, and reward details. | Treat `verifier/` as a package with `verifier.md`, reward directories, rubrics, judges, `reward.json`, and `reward-details.json`. |
| AgentBeats | [AgentBeats](https://agentbeats.dev/), [docs](https://docs.agentbeats.dev/) | AgentBeats models assessment as an agent that sets up environments, interacts with participant agents, and emits structured result JSON. | Model assessor agents as verifier strategies, not interaction scenes or root task syntax. |
| OpenReward Standard | [OpenReward Standard](https://openrewardstandard.io/), [rewards](https://openrewardstandard.io/concepts/rewards) | ORS returns rewards through tool outputs over episodes and preserves metadata. | Treat ORS as adapter/runtime compatibility; preserve reward events and aggregate explicitly. |
| METR | [METR task-standard](https://github.com/METR/task-standard), [METR QA](https://taskdev.metr.org/quality-assurance/) | METR is code-first but has strong asset, secret, scoring, and QA discipline. | Copy evidence discipline, not the Python `TaskFamily` package shape. |

## Proposed Standard

See [../task-standard.md](../task-standard.md). The essential choices are:

1. Root config stays Harbor-compatible and strict.
2. BenchFlow-only extensions live under `benchflow:`.
3. `schema_version` and `benchflow.document_version` are separate.
4. `scenes` are agent interaction; `steps` are verifier/runtime checkpoints.
5. `oracle/` and `verifier/` are native; `solution/` and `tests/` are
   compatibility/export names.
6. Compatibility exports report loss instead of pretending every target can
   express teams, scenes, nudges, assets, or evidence.
7. `verifier/` is a first-class verifier package. `verifier/test.sh` is the
   minimum script strategy, while `verifier/verifier.md` can declare Reward
   Kit criteria, ORS reward-event aggregation, LLM judges, agent judges,
   hidden inputs, output files, and calibration evidence.

## Technical Debt Surfaced

Priority backlog:

1. Create `TaskPackage` / `TaskRuntimeView` as the single module for task
   entrypoint, prompt materialization, verifier/oracle path resolution, source
   hashes, scenes, and runtime capability checks.
2. Split native authoring validation from foreign adapter validation so
   Harbor/Pier extension keys can be preserved or reported instead of silently
   dropped or rejected by the wrong mode.
3. Fix legacy migration for reserved markdown headings inside `instruction.md`;
   broad migration should not reinterpret prompt text as `task.md` role/scene
   sections.
4. Add fail-closed runtime capability validation for parsed network policy,
   `steps`, artifacts, separate verifier environments, Windows, TPU,
   healthchecks, workdir, and non-`main` verifier services.
5. Enforce mixed native/legacy equivalence instead of only preferring native
   `task.md`, `verifier/`, and `oracle/`.
6. Decide direct `Runtime` / `bf.run("agent", task_path=...)` semantics for
   document-declared scenes.
7. Compile `user` / `## user-persona` into a simulated-user loop or reject it
   as runnable configuration.
8. Make prompt composition explicit so scene prompts do not drop role guardrails.
9. Fix `reward.json` precedence: prefer rich JSON when present, require
   agreement with `reward.txt` when both exist and an aggregate is present,
   preserve Harbor Reward Kit multi-metric maps, and keep `reward-details.json`
   plus structured evidence fields.
10. Split native vocabulary from foreign adapter vocabulary in code and docs.
11. Add `VerifierDocument` for `verifier/verifier.md`, with
    `verifier/task.toml` as an optional compatibility projection and
    `test.sh` as one strategy rather than the whole verifier model.
12. Add verifier strategy support for deterministic script, Reward Kit
    criteria, LLM judge, agent judge, and ORS episode aggregation; unsupported
    strategies fail closed before scoring starts.
13. Type and enforce the `benchflow:` namespace once the draft stabilizes.
14. Add Harbor/Pier/Terminal-Bench/SWE-bench/ORS/AgentBeats export loss reports and
    conformance tests.

Known code risks from the draft slice:

- `render_task_md_from_legacy()` inserts raw legacy prompts under `## prompt`;
  if a legacy `instruction.md` already contains reserved headings such as
  `## role:name` or `## scene:name`, migration may reinterpret the prompt
  instead of round-tripping it literally.
- Native path helpers prefer `verifier/` over `tests/` and `oracle/` over
  `solution/`; structural validation does not yet compare duplicate alias
  trees for equivalence.
- Strict unknown-key validation currently applies to Harbor/Pier imports too.
  Adapter import mode still needs a compatibility-preserving path for foreign
  extension keys.
- `Verifier` execution is still script-first: `verifier/verifier.md`,
  Reward Kit criteria, agent-judge roles, and `reward-details.json` are target
  package concepts, not current runtime behavior.

## Validation Matrix To Run Next

Use `/tmp/benchflow-taskmd-parity` as the output root and pin remote sources
for reproducibility.

| Lane | Tasks | Pass criteria |
|---|---|---|
| Local parser/schema | all `docs/examples/task-md/**/task.md` | `tests/test_task_document.py`, `tests/test_task_config.py`, and `tests/test_tasks.py` pass; unknown fields, config-key alias collisions, and bad roles fail; filesystem alias collisions are a target check |
| Real SkillsBench parity | `weighted-gdp-calc`, `3d-scan-calc`, `citation-check` | legacy vs migrated `task.md` oracle Docker rewards/errors match; `3d-scan-calc` also checks with-skill behavior |
| Generated skill-eval | `regex-email-parser`, `topo-sort-with-cycle-detection`, `optimize-quadratic-to-nlogn` | generated tasks are native `task.md` + `verifier/`; rewards and verifier artifacts are present |
| Harbor registry tasks | `binary-audit/caddy-backdoor-detect`, `terminal-bench/adaptive-rejection-sampler` | registry tasks check, migration preserves config/prompt, no silent skip |
| Terminal-Bench-2 | `regex-log`, `cancel-async-tasks`, `adaptive-rejection-sampler` | legacy adapter output and migrated `task.md` output produce matching reward/error state |
| Pier / DeepSWE | `abs-module-cache-flags`, `actionlint-action-pinning-lint` | Harbor migration preserves docker image, no-internet/resource fields, oracle env, verifier, prompt |
| Advanced schema fields | `docs/examples/task-md/harbor-parity/task.md` | parse succeeds for allowlist, workdir, healthcheck, TPU, GPUs, separate verifier env, root/step artifacts, steps; runtime must fail closed until implemented |
| Scenes/user semantics | multi-scene and NudgeBench parser fixtures | scenes compile in order, role/model overrides survive, scene boundaries reconnect; user metadata remains explicit metadata-only |
| Negative contracts | unknown keys, oracle+solution collision, dangling roles, no-skill isolation, unsupported runtime fields | all are red with clear errors rather than silently running |

Every live eval should produce `result.json`, `prompts.json`, `timing.json`,
`trajectory/acp_trajectory.jsonl`, `rewards.jsonl`, and verifier reward
artifacts. Remote/source-backed tasks should include source file hashes in the
result metadata.

## Decision

Make `task.md` the best BenchFlow-native authoring standard, but do not call it
the universal interchange format. The next product-quality milestone is a
runtime capability gate that can say, precisely, "this task parses, but this
sandbox cannot honor these semantics."
