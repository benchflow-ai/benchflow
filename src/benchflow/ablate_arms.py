"""Ablation pre-flight — everything decidable from the request alone.

The parse/validate half of ``bench eval ablate`` (rollout-branching RFC §5),
split out of :mod:`benchflow.ablate` so that module stays the *orchestration*
of an experiment and this one stays its *admission gate*. An ablation costs a
full task run before the branch, so a request the branch engine would reject
at fork time must not cost that run first: arm specs parse into executable
:class:`~benchflow.branch_delta.BranchDelta` values here (:func:`parse_arm`,
:func:`parse_arms`), stage/arm combinations are rejected before anything runs
(:func:`validate_arms_for_stage`), and the one task and its environment
binding resolve fail-closed (:func:`resolve_ablation_task`,
:func:`resolve_ablation_environment_binding`). The engine keeps its own gates
— everything here is a pre-flight mirror, never a replacement.

The ablation error hierarchy lives here too, beside its earliest raisers;
:mod:`benchflow.ablate` re-exports every public name, so callers keep
importing the one ablation surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchflow.branch_delta import BranchDelta
from benchflow.branch_skill import (
    FRESH_CHILD_LAYER,
    FRESH_CHILD_STAGE,
    SKILL_DELTA_STAGE,
)
from benchflow.branch_stage import (
    BRANCH_STAGES,
    MARKED_STAGES,
    STAGE_POST_RESEARCH,
    validate_stage,
)
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL

if TYPE_CHECKING:
    from benchflow.environment.manifest import ManifestBinding

#: Arm spec prefix for a plan-injection arm: ``inject:<path-to-file>``.
INJECT_PREFIX = "inject:"

#: Arm spec prefix for a config-override arm: ``config:<inline-or-@file>`` —
#: the same dual "value or ref" form as the run-level ``--config-override``.
CONFIG_PREFIX = "config:"

#: Arm spec prefix for an environment arm: ``env:<registry-ref-or-path>``
#: (the ``env0@prod`` vs ``env0@outage`` tool-outage pattern).
ENV_PREFIX = "env:"

ARM_KIND_SKILL_MODE = "skill-mode"
ARM_KIND_INJECT = "inject"
ARM_KIND_CONFIG = "config-override"
ARM_KIND_ENV = "environment-ref"

#: The stages ``bench eval ablate`` can capture on its own — everything the
#: lifecycle reaches without an explicit ``Rollout.mark_stage()`` call.
CAPTURABLE_STAGES: tuple[str, ...] = tuple(
    stage for stage in BRANCH_STAGES if stage not in MARKED_STAGES
)

_SKILL_ARM_NAMES = (SKILL_MODE_WITH_SKILL, SKILL_MODE_NO_SKILL)
_ARM_SPEC_HELP = (
    f"supported arms are {SKILL_MODE_WITH_SKILL!r}, {SKILL_MODE_NO_SKILL!r}, "
    f"'{INJECT_PREFIX}<path-to-file>', "
    f"'{CONFIG_PREFIX}<inline-json-or-@file>', and "
    f"'{ENV_PREFIX}<registry-ref>'"
)


class AblationError(Exception):
    """Base class for ablation failures the CLI reports without a traceback."""


class AblationSpecError(AblationError, ValueError):
    """The requested ablation cannot run as asked (fail closed, before any run).

    Raised for a malformed arm spec, an arm the branch engine could not execute
    at the requested stage, a stage this command cannot capture, or a
    ``--tasks-dir`` that does not name exactly one task.
    """


class AblationRunError(AblationError, RuntimeError):
    """The parent rollout never reached the stage boundary to branch from."""


# Arms


@dataclass(frozen=True)
class AblationArm:
    """One arm of the ablation: a recognizable name and the delta it runs under.

    ``name`` is the spec exactly as the caller wrote it — it is the row label
    in the table, the key in ``ablation.json``, and the reference other arms
    are compared against, so it survives the round trip verbatim. ``source``
    records where an injection arm read its text from; the text itself is
    never recorded (provenance hashes it, per #908).
    """

    name: str
    kind: str
    delta: BranchDelta
    source: str | None = None


def parse_arm(spec: str) -> AblationArm:
    """Parse one arm spec into an :class:`AblationArm` (fail closed).

    Five kinds, each mapping onto exactly one executable
    :class:`BranchDelta` field: ``with-skill`` / ``no-skill`` become
    ``skill_mode`` (the child re-runs installation as a fresh rollout from the
    ``env-ready`` snapshot), ``inject:<path>`` reads the file and becomes
    ``injected_prompt`` (the child's user-visible continuation prompt) — which
    at ``--at-stage env-ready`` is *also* delivered by a fresh rollout, since
    every child of that boundary installs the agent for itself —
    ``config:<inline-or-@file>`` parses through the run-level overlay loader
    and becomes ``config_override`` (the child runs fresh under the parent's
    config with the allowlisted patch deep-merged on top, #790), and
    ``env:<registry-ref>`` becomes ``environment_ref`` (the service-level
    environment swap; resolution against the parent's manifest happens in
    :func:`~benchflow.ablate.run_ablation`, where the parent's manifest is
    known). An unknown kind, an empty spec, an injection file that is missing
    or blank, or a config patch that is unparsable or touches a
    non-allowlisted section raises :class:`AblationSpecError` — a silently
    dropped arm would publish an ablation table with a missing comparison, and
    a scorer-touching patch must die here, before the parent run costs
    anything.
    """
    name = spec.strip()
    if not name:
        raise AblationSpecError(f"empty arm in --arms — {_ARM_SPEC_HELP}")
    if name in _SKILL_ARM_NAMES:
        return AblationArm(
            name=name, kind=ARM_KIND_SKILL_MODE, delta=BranchDelta(skill_mode=name)
        )
    if name.startswith(CONFIG_PREFIX):
        return _parse_config_arm(name)
    if name.startswith(ENV_PREFIX):
        ref = name[len(ENV_PREFIX) :].strip()
        if not ref:
            raise AblationSpecError(
                f"arm {name!r} names no environment — the environment arm is "
                f"'{ENV_PREFIX}<registry-ref>' (e.g. '{ENV_PREFIX}env0@outage') "
                "or a manifest file path"
            )
        return AblationArm(
            name=name, kind=ARM_KIND_ENV, delta=BranchDelta(environment_ref=ref)
        )
    if name.startswith(INJECT_PREFIX):
        raw = name[len(INJECT_PREFIX) :].strip()
        if not raw:
            raise AblationSpecError(
                f"arm {name!r} names no file — the injection arm is "
                f"'{INJECT_PREFIX}<path-to-file>' (e.g. "
                f"'{INJECT_PREFIX}oracle-plan.md')"
            )
        path = Path(raw)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AblationSpecError(
                f"arm {name!r} cannot read its injection file {raw!r}: {exc}"
            ) from None
        if not text.strip():
            raise AblationSpecError(
                f"arm {name!r} reads an empty injection file {raw!r} — an "
                "injection arm must carry the text it injects"
            )
        return AblationArm(
            name=name,
            kind=ARM_KIND_INJECT,
            delta=BranchDelta(injected_prompt=text),
            source=str(path),
        )
    raise AblationSpecError(f"unknown arm {name!r} — {_ARM_SPEC_HELP}")


def _parse_config_arm(name: str) -> AblationArm:
    """Parse a ``config:<inline-or-@file>`` arm through the #790 loader.

    The value after the prefix is exactly what ``--config-override`` accepts —
    inline JSON/YAML/TOML or an ``@file`` ref — parsed by the same
    :func:`~benchflow._utils.config_override.load_config_override` so the two
    surfaces cannot drift. The allowlist runs *here*, at parse time: an arm
    that patches the scorer must fail before the parent run costs anything,
    exactly as the branch engine would fail it at fork time.
    """
    from benchflow._utils.config_override import load_config_override, validate_overlay

    raw = name[len(CONFIG_PREFIX) :].strip()
    if not raw:
        raise AblationSpecError(
            f"arm {name!r} carries no patch — the config arm is "
            f"'{CONFIG_PREFIX}<inline-json-or-@file>' (e.g. "
            f"""'{CONFIG_PREFIX}{{"agent": {{"timeout_sec": 60}}}}' or """
            f"'{CONFIG_PREFIX}@overlay.yaml')"
        )
    try:
        overlay = load_config_override(raw)
    except (ValueError, OSError) as exc:
        raise AblationSpecError(
            f"arm {name!r} cannot load its config patch: {exc}"
        ) from None
    if not overlay:
        raise AblationSpecError(
            f"arm {name!r} parses to an empty patch — a config arm must "
            "change at least one allowlisted section"
        )
    try:
        validate_overlay(overlay)
    except ValueError as exc:
        raise AblationSpecError(f"arm {name!r} is not executable: {exc}") from None
    return AblationArm(
        name=name,
        kind=ARM_KIND_CONFIG,
        delta=BranchDelta(config_override=overlay),
        source=raw[1:] if raw.startswith("@") else None,
    )


def _split_arm_specs(spec: str) -> list[str]:
    """Split ``--arms`` on commas, ignoring commas nested in JSON braces.

    A ``config:`` arm may carry inline JSON (``config:{"agent": {"a": 1,
    "b": 2}}``) whose commas are content, not separators. Depth counting over
    ``{}``/``[]`` keeps every historical spec splitting exactly as before —
    no other arm kind can contain a brace. Inside a JSON string literal,
    braces, brackets and commas are content too, so the walk tracks quote
    state (with ``\\``-escape handling) and ignores structure until the
    string closes. Commas in a ``config:@<path>`` file path remain
    unrepresentable — that grammar limit is documented on ``--arms``.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    for char in spec:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            current.append(char)
            continue
        if char == '"' and depth > 0:
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def parse_arms(spec: str) -> list[AblationArm]:
    """Parse a comma-separated ``--arms`` value into ordered arms.

    Two whole-request rules beyond the per-arm ones. A fork needs at least two
    children (the branch engine's own ``n >= 2``), so a single arm is rejected
    here rather than after a full parent run; and duplicate arm names are
    rejected because the report keys arms by name, which would make the
    attribution ambiguous.
    """
    if not spec.strip():
        raise AblationSpecError(
            "--arms is empty — pass at least two arms, e.g. "
            f"'{SKILL_MODE_WITH_SKILL},{SKILL_MODE_NO_SKILL}'"
        )
    arms = [parse_arm(entry) for entry in _split_arm_specs(spec)]
    seen: set[str] = set()
    for arm in arms:
        if arm.name in seen:
            raise AblationSpecError(
                f"duplicate arm {arm.name!r} in --arms — each arm is one "
                "controlled change, and the report keys arms by name"
            )
        seen.add(arm.name)
    if len(arms) < 2:
        raise AblationSpecError(
            f"--arms needs at least two arms to compare, got one "
            f"({arms[0].name!r}) — a branch forks into >= 2 children. Pair it "
            f"with a counterpart arm, e.g. "
            f"'{SKILL_MODE_WITH_SKILL},{SKILL_MODE_NO_SKILL}'"
        )
    return arms


def validate_arms_for_stage(
    arms: Sequence[AblationArm],
    stage: str,
    *,
    snapshot_layers: frozenset[str] | set[str] | None = None,
    research_end_marker: str | None = None,
) -> str:
    """Reject a stage/arm combination before the parent rollout runs.

    Pre-flight gates, each mirroring a rule that lives elsewhere:

    * ``post-research`` is a mid-``execute()`` cut point only an explicit
      ``Rollout.mark_stage()`` can record (RFC §3.2). This command drives the
      lifecycle, not the agent's planning, so on its own it cannot mark it —
      ``--mark-research-end-on <workspace-path>`` supplies the concrete
      trigger (the engine marks the stage the first time that file exists in
      the workspace; see :func:`~benchflow.ablate.watch_research_end`), and
      without it the rejection says so up front, which beats running the whole
      task and dying at the fork. The marker is meaningless for any other
      stage and fails closed there.
    * a ``skill_mode`` arm executes only at ``env-ready``, because skills are
      deployed by ``install_agent()`` (RFC §3.3). The branch engine fails
      closed on this too; here it costs nothing instead of a full run.
    * a ``config:`` arm executes only at ``env-ready`` for the same shape of
      reason: the overlay is applied by the child's own ``setup()``, which
      only a fresh child of that boundary re-runs.
    * an ``env:`` arm executes only at ``env-ready`` likewise: the child
      manifest's services are provisioned over the restored snapshot by the
      fresh child. (Its *content* gates — resolvable ref, same image,
      framework-started services — need the parent's manifest and run in
      :func:`~benchflow.ablate.run_ablation`, still before the parent run.)
    * an ``env-ready`` ablation runs *every* arm as a fresh rollout — that
      boundary precedes ``install_agent()``, so each child installs the agent
      for itself — which needs the container layer in the stage snapshot. The
      engine raises :class:`~benchflow.rollout_branch.BranchChildExecutionNotSupported`
      for this; the mirror here is checked only when the caller passes the
      layers it will request.

    Returns the validated stage, so callers can use this as their one gate.
    """
    validate_stage(stage, field="--at-stage")
    if research_end_marker is not None and stage != STAGE_POST_RESEARCH:
        raise AblationSpecError(
            f"--mark-research-end-on only applies to --at-stage "
            f"{STAGE_POST_RESEARCH!r}, not {stage!r}: the marker file's first "
            "appearance is what defines the research-end boundary, and no "
            "other stage is captured from it"
        )
    if stage in MARKED_STAGES and research_end_marker is None:
        raise AblationSpecError(
            f"--at-stage {stage!r} cannot be captured by this command without "
            "a research-end trigger: it is a mid-execute() cut point only an "
            "explicit Rollout.mark_stage() call can record. Pass "
            "--mark-research-end-on <workspace-path> (e.g. /app/PLAN.md) so "
            "the engine marks the stage the first time that file appears "
            "during the agent's run; or drive the run through the SDK — "
            f"await rollout.mark_stage({stage!r}) at the cut point during the "
            f"agent's run, then await rollout.branch_at_stage({stage!r}, n, "
            "deltas=[...]) — or ablate one of "
            f"{list(CAPTURABLE_STAGES)!r}"
        )
    for arm in arms:
        if arm.kind == ARM_KIND_SKILL_MODE and stage != SKILL_DELTA_STAGE:
            raise AblationSpecError(
                f"arm {arm.name!r} needs --at-stage {SKILL_DELTA_STAGE!r}, not "
                f"{stage!r}: skills are deployed by install_agent(), which has "
                f"already run by {stage!r}, so the arm would measure nothing"
            )
        if arm.kind == ARM_KIND_CONFIG and stage != FRESH_CHILD_STAGE:
            raise AblationSpecError(
                f"arm {arm.name!r} needs --at-stage {FRESH_CHILD_STAGE!r}, not "
                f"{stage!r}: the config patch is applied by the child's own "
                f"setup(), and by {stage!r} the parent's config has already "
                "been consumed — the arm would record the override and run "
                "without it"
            )
        if arm.kind == ARM_KIND_ENV and stage != FRESH_CHILD_STAGE:
            raise AblationSpecError(
                f"arm {arm.name!r} needs --at-stage {FRESH_CHILD_STAGE!r}, not "
                f"{stage!r}: the child manifest's services are provisioned "
                "over the restored snapshot by the fresh child rollout, and "
                f"by {stage!r} the parent's provisioned services survive the "
                "fork — the arm would record the swap and run without it"
            )
    if (
        stage == FRESH_CHILD_STAGE
        and snapshot_layers is not None
        and FRESH_CHILD_LAYER not in snapshot_layers
    ):
        raise AblationSpecError(
            f"--at-stage {FRESH_CHILD_STAGE!r} needs the {FRESH_CHILD_LAYER!r} "
            f"snapshot layer, got {sorted(frozenset(snapshot_layers))!r}: that "
            "boundary precedes install_agent(), so every arm re-installs the "
            "agent for itself and the container layer is what rolls one arm's "
            "installation back before the next one runs"
        )
    return stage


def resolve_ablation_task(tasks_dir: Path) -> Path:
    """The one task an ablation runs, resolved from ``--tasks-dir``.

    An ablation compares arms *within* one task — the arms are the axis, the
    task is fixed — so a directory holding several tasks is a request this
    command cannot answer, not a batch to expand. It fails closed naming the
    tasks it found.
    """
    from benchflow.task.discovery import is_task_dir, resolve_task_collection_root

    path = Path(tasks_dir)
    if not path.is_dir():
        raise AblationSpecError(f"--tasks-dir {str(path)!r} is not a directory")
    root = resolve_task_collection_root(path)
    if is_task_dir(root):
        return root
    tasks = sorted(
        child for child in root.iterdir() if child.is_dir() and is_task_dir(child)
    )
    if not tasks:
        raise AblationSpecError(
            f"no task found under --tasks-dir {str(path)!r} — a task directory "
            "carries a task.md or task.toml"
        )
    if len(tasks) > 1:
        names = [task.name for task in tasks]
        raise AblationSpecError(
            f"--tasks-dir {str(path)!r} holds {len(names)} tasks ({names!r}) — "
            "an ablation compares arms within one task; point --tasks-dir at "
            "the task directory itself"
        )
    return tasks[0]


def resolve_ablation_environment_binding(
    task_path: Path, *, explicit: Path | str | None = None
) -> ManifestBinding | None:
    """The environment the ablation binds — flag first, then the task's own.

    ``explicit`` is the ``--environment-manifest`` value (a manifest path or a
    ``name@version`` registry spec): when given it wins outright and the
    task-declared manifest is not even resolved — the same precedence ``bench
    eval run`` applies (an explicit run-level manifest suppresses
    ``manifest_from_task_document``). Otherwise the task's own ``task.md``
    declaration is resolved
    (:func:`~benchflow.environment.manifest.manifest_binding_from_task_document`),
    exactly as a normal evaluation resolves it; a stateful task ablated
    without it would run the parent — and therefore every arm forked from it —
    in a *different* environment than the run it is meant to explain.

    The returned :class:`~benchflow.environment.manifest.ManifestBinding`
    keeps the ref verbatim and the manifest's content address, which is what
    :func:`~benchflow.branch_report.environment_stamp` writes into
    ``ablation.json``. Resolution failures are fatal rather than degrading to
    ``None``: an ablation whose declared environment could not be built is not
    an ablation that ran without services, and it fails before the parent run
    costs anything.
    """
    from benchflow._utils.text import describe_exception
    from benchflow.environment.manifest import (
        load_manifest_binding,
        manifest_binding_from_task_document,
    )

    if explicit is not None:
        try:
            return load_manifest_binding(explicit)
        except Exception as exc:
            raise AblationSpecError(
                f"--environment-manifest {str(explicit)!r} does not resolve to "
                f"an environment manifest: {describe_exception(exc)}"
            ) from exc
    try:
        return manifest_binding_from_task_document(task_path)
    except Exception as exc:
        raise AblationSpecError(
            f"{task_path.name} declares an environment manifest in its task.md "
            f"that could not be resolved: {describe_exception(exc)}. Every arm "
            "forks the parent's environment, so this ablation would compare "
            "arms in a world the task says is the wrong one"
        ) from exc
