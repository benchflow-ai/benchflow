"""The embodiment spec: what a simulated robot senses, how it is commanded, and what skills it has.

One schema covers single arms, bimanual rigs, dexterous hands, mobile manipulators, quadrupeds, humanoids,
drones and skill-only robots. A backend keeps its native flat action vector; action groups name slices of it
in declaration order. See docs/embodied.md.

Standard library only: this module is vendored into simulator images.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

SPEC_VERSION = "1"

KINDS = (
    "arm",
    "bimanual",
    "hand",
    "mobile_manipulator",
    "quadruped",
    "humanoid",
    "drone",
    "skill_only",
    "other",
)
HOLD_POLICIES = ("zero", "last")
ARG_TYPES = ("float", "int", "str", "bool", "object", "enum")
SKILL_IMPLS = ("builtin", "backend")


class SpecError(ValueError):
    """An embodiment spec, or an action checked against one, is invalid."""


@dataclass
class ActionGroup:
    """A named slice of the flat action vector."""

    name: str
    components: list[str]
    low: list[float]
    high: list[float]
    mode: str
    units: str = ""
    frame: str = ""
    hold: str = "zero"
    initial: list[float] | None = None
    doc: str = ""

    @property
    def dim(self) -> int:
        return len(self.components)

    def hold_value(self, last: list[float] | None) -> list[float]:
        if self.hold == "last" and last is not None:
            return list(last)
        if self.hold == "last" and self.initial is not None:
            return list(self.initial)
        return [0.0] * self.dim


@dataclass
class Camera:
    name: str
    mount: str = "world"
    width: int | None = None
    height: int | None = None
    calibrated: bool = True
    doc: str = ""


@dataclass
class Field:
    """One observation field."""

    name: str
    shape: list[int] = field(default_factory=list)
    units: str = ""
    frame: str = ""
    privileged: bool = False
    doc: str = ""


@dataclass
class Sensors:
    cameras: list[Camera] = field(default_factory=list)
    proprioception: list[Field] = field(default_factory=list)
    state: list[Field] = field(default_factory=list)


@dataclass
class SkillArg:
    name: str
    type: str = "float"
    units: str = ""
    optional: bool = False
    default: Any = None
    choices: list[Any] | None = None
    doc: str = ""


@dataclass
class Skill:
    name: str
    args: list[SkillArg] = field(default_factory=list)
    doc: str = ""
    preconditions: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    impl: str = "backend"
    # builtin skills: which action groups the controller drives (e.g. {"arm": "right.ee_delta"})
    binds: dict[str, str] = field(default_factory=dict)


@dataclass
class Budgets:
    max_steps: int = 500
    max_wall_s: float | None = None
    max_repeat: int = 50
    max_skill_steps: int = 150
    settle_steps: int = 10


@dataclass
class RewardSpec:
    dense: str = "sparse"  # "shaped": the backend's own per-step reward; "sparse": the success indicator
    success_mode: str = "final"


@dataclass
class Embodiment:
    name: str
    kind: str
    action_groups: list[ActionGroup]
    sensors: Sensors = field(default_factory=Sensors)
    skills: list[Skill] = field(default_factory=list)
    budgets: Budgets = field(default_factory=Budgets)
    reward: RewardSpec = field(default_factory=RewardSpec)
    step_s: float | None = None
    doc: str = ""
    spec_version: str = SPEC_VERSION

    # ---- structure ----------------------------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return sum(g.dim for g in self.action_groups)

    def group(self, name: str) -> ActionGroup:
        for g in self.action_groups:
            if g.name == name:
                return g
        raise SpecError(
            f"unknown action group {name!r}; groups: {', '.join(g.name for g in self.action_groups)}"
        )

    def slices(self) -> dict[str, slice]:
        out, i = {}, 0
        for g in self.action_groups:
            out[g.name] = slice(i, i + g.dim)
            i += g.dim
        return out

    def flat_names(self) -> list[str]:
        return [c for g in self.action_groups for c in g.components]

    def flat_low(self) -> list[float]:
        return [float(x) for g in self.action_groups for x in g.low]

    def flat_high(self) -> list[float]:
        return [float(x) for g in self.action_groups for x in g.high]

    def find_skill(self, name: str) -> Skill | None:
        for s in self.skills:
            if s.name == name or name in s.aliases:
                return s
        return None

    # ---- actions ------------------------------------------------------------------------------------------
    def check_group_values(self, name: str, values: Any) -> list[float]:
        g = self.group(name)
        if not isinstance(values, (list, tuple)) or len(values) != g.dim:
            raise SpecError(
                f"{name} takes {g.dim} number(s) {g.components}, got {values!r}"
            )
        out = []
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise SpecError(f"{name}: {v!r} is not a number")
            if not math.isfinite(float(v)):
                raise SpecError(f"{name}: {v!r} is not finite")
            out.append(float(v))
        return out

    def pack(
        self,
        groups: dict[str, Any],
        last: dict[str, list[float]] | None = None,
    ) -> list[float]:
        """Flat action vector from commanded groups; other groups follow their hold policy. Values are clipped."""
        unknown = [n for n in groups if n not in {g.name for g in self.action_groups}]
        if unknown:
            raise SpecError(
                f"unknown action group(s) {unknown}; groups: {', '.join(g.name for g in self.action_groups)}"
            )
        flat: list[float] = []
        for g in self.action_groups:
            if g.name in groups:
                vals = self.check_group_values(g.name, groups[g.name])
            else:
                vals = g.hold_value((last or {}).get(g.name))
            flat += [
                min(max(v, lo), hi)
                for v, lo, hi in zip(vals, g.low, g.high, strict=True)
            ]
        return flat

    def unpack(self, flat: list[float]) -> dict[str, list[float]]:
        if len(flat) != self.dim:
            raise SpecError(
                f"action must be a list of {self.dim} numbers {self.flat_names()}"
            )
        return {n: [float(x) for x in flat[s]] for n, s in self.slices().items()}

    def hold_action(self, last: dict[str, list[float]] | None = None) -> list[float]:
        return self.pack({}, last)

    # ---- (de)serialisation --------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for g in d["action_groups"]:
            g["dim"] = len(g["components"])
        return _drop_empty(d)

    @classmethod
    def from_dict(cls, d: dict) -> Embodiment:
        groups = [
            ActionGroup(**{k: v for k, v in g.items() if k != "dim"})
            for g in d.get("action_groups", [])
        ]
        sensors_d = d.get("sensors") or {}
        sensors = Sensors(
            cameras=[Camera(**c) for c in sensors_d.get("cameras", [])],
            proprioception=[Field(**f) for f in sensors_d.get("proprioception", [])],
            state=[Field(**f) for f in sensors_d.get("state", [])],
        )
        skills = []
        for s in d.get("skills", []):
            kw: dict[str, Any] = dict(s)
            kw["args"] = [SkillArg(**a) for a in s.get("args", [])]
            skills.append(Skill(**kw))
        return cls(
            name=d["name"],
            kind=d["kind"],
            action_groups=groups,
            sensors=sensors,
            skills=skills,
            budgets=Budgets(**(d.get("budgets") or {})),
            reward=RewardSpec(**(d.get("reward") or {})),
            step_s=d.get("step_s"),
            doc=d.get("doc", ""),
            spec_version=str(d.get("spec_version", SPEC_VERSION)),
        )

    # ---- validation ---------------------------------------------------------------------------------------
    def validate(self) -> Embodiment:
        """Raise SpecError on the first structural problem; return self."""
        if not self.name:
            raise SpecError("embodiment needs a name")
        if self.kind not in KINDS:
            raise SpecError(f"kind {self.kind!r} is not one of {KINDS}")
        names: set[str] = set()
        for g in self.action_groups:
            if not g.name or g.name in names:
                raise SpecError(
                    f"action group names must be unique and non-empty: {g.name!r}"
                )
            names.add(g.name)
            if g.dim == 0:
                raise SpecError(f"action group {g.name!r} has no components")
            if len(g.low) != g.dim or len(g.high) != g.dim:
                raise SpecError(
                    f"action group {g.name!r}: low/high must have {g.dim} entries"
                )
            if any(lo > hi for lo, hi in zip(g.low, g.high, strict=True)):
                raise SpecError(f"action group {g.name!r}: low > high")
            if g.hold not in HOLD_POLICIES:
                raise SpecError(
                    f"action group {g.name!r}: hold must be one of {HOLD_POLICIES}"
                )
            if g.initial is not None and len(g.initial) != g.dim:
                raise SpecError(
                    f"action group {g.name!r}: initial must have {g.dim} entries"
                )
            if not g.mode:
                raise SpecError(f"action group {g.name!r} needs a control mode")
        if not self.action_groups and not self.skills:
            raise SpecError("an embodiment needs at least one action group or skill")
        skill_names: set[str] = set()
        for s in self.skills:
            for n in [s.name, *s.aliases]:
                if n in skill_names:
                    raise SpecError(f"skill name or alias {n!r} is used twice")
                skill_names.add(n)
            if s.impl not in SKILL_IMPLS:
                raise SpecError(f"skill {s.name!r}: impl must be one of {SKILL_IMPLS}")
            for a in s.args:
                if a.type not in ARG_TYPES:
                    raise SpecError(
                        f"skill {s.name!r} arg {a.name!r}: type must be one of {ARG_TYPES}"
                    )
                if a.type == "enum" and not a.choices:
                    raise SpecError(
                        f"skill {s.name!r} arg {a.name!r}: enum needs choices"
                    )
            for role, group in s.binds.items():
                if group not in names:
                    raise SpecError(
                        f"skill {s.name!r} binds {role} to unknown group {group!r}"
                    )
        cams = [c.name for c in self.sensors.cameras]
        if len(cams) != len(set(cams)):
            raise SpecError("camera names must be unique")
        if self.budgets.max_steps < 1 or self.budgets.max_repeat < 1:
            raise SpecError("budgets must be positive")
        if self.reward.dense not in ("shaped", "sparse"):
            raise SpecError("reward.dense must be 'shaped' or 'sparse'")
        if self.reward.success_mode not in ("final", "first"):
            raise SpecError("reward.success_mode must be 'final' or 'first'")
        return self


def _drop_empty(x: Any) -> Any:
    """Drop None values and empty containers/strings from nested dicts (keeps lists' shape)."""
    if isinstance(x, dict):
        return {
            k: _drop_empty(v)
            for k, v in x.items()
            if v is not None and v != "" and v != [] and v != {}
        }
    if isinstance(x, list):
        return [_drop_empty(v) for v in x]
    return x


# ---- skill argument binding -------------------------------------------------------------------------------


def bind_skill_args(skill: Skill, args: Any) -> dict[str, Any]:
    """Map positional (list) or keyword (dict) arguments onto the skill's typed args.

    Positional strings of the form ``key=value`` are keyword arguments. Raises SpecError on a missing, unknown
    or mistyped argument.
    """
    pos: list[Any] = []
    kw: dict[str, Any] = {}
    if isinstance(args, dict):
        kw = dict(args)
    elif isinstance(args, (list, tuple)):
        for a in args:
            if (
                isinstance(a, str)
                and "=" in a
                and a.split("=", 1)[0] in {x.name for x in skill.args}
            ):
                k, v = a.split("=", 1)
                kw[k] = v
            else:
                pos.append(a)
    else:
        raise SpecError("skill arguments must be a list or an object")
    if len(pos) > len(skill.args):
        raise SpecError(
            f"{skill.name} takes at most {len(skill.args)} argument(s) "
            f"({', '.join(a.name for a in skill.args)}), got {len(pos)}"
        )
    out: dict[str, Any] = {}
    for a, v in zip(
        skill.args, pos, strict=False
    ):  # fewer positionals than args is fine
        out[a.name] = v
    for k, v in kw.items():
        if k not in {a.name for a in skill.args}:
            raise SpecError(f"{skill.name} has no argument {k!r}")
        if k in out:
            raise SpecError(f"{skill.name}: argument {k!r} given twice")
        out[k] = v
    for a in skill.args:
        if a.name not in out:
            if not a.optional:
                raise SpecError(f"{skill.name} needs argument {a.name!r}")
            if a.default is not None:
                out[a.name] = a.default
            continue
        out[a.name] = _coerce(skill.name, a, out[a.name])
    return out


def _coerce(skill: str, a: SkillArg, v: Any) -> Any:
    try:
        if a.type == "float":
            f = float(v)
            if not math.isfinite(f):
                raise ValueError
            return f
        if a.type == "int":
            return int(v)
        if a.type == "bool":
            if isinstance(v, str):
                if v.lower() in ("1", "true", "yes", "on"):
                    return True
                if v.lower() in ("0", "false", "no", "off"):
                    return False
                raise ValueError
            return bool(v)
        if a.type == "enum":
            if v not in (a.choices or []):
                raise ValueError
            return v
        return str(v)
    except (TypeError, ValueError):
        extra = f" (one of {a.choices})" if a.choices else ""
        raise SpecError(
            f"{skill}: argument {a.name!r} must be {a.type}{extra}, got {v!r}"
        ) from None


# ---- builders for common embodiments ----------------------------------------------------------------------


def ee_delta_group(
    prefix: str = "arm",
    scale: str = "",
    limit: float = 1.0,
    rotation: bool = False,
    rot_limit: float | None = None,
    rot_units: str = "",
) -> list[ActionGroup]:
    groups = [
        ActionGroup(
            name=f"{prefix}.ee_delta",
            components=["dx", "dy", "dz"],
            low=[-limit] * 3,
            high=[limit] * 3,
            mode="ee_delta_pos",
            units=scale,
            frame="world",
        )
    ]
    if rotation:
        rl = limit if rot_limit is None else rot_limit
        groups.append(
            ActionGroup(
                name=f"{prefix}.ee_rot_delta",
                components=["rx", "ry", "rz"],
                low=[-rl] * 3,
                high=[rl] * 3,
                mode="ee_delta_rot",
                units=rot_units,
                frame="world",
            )
        )
    return groups


def gripper_group(
    name: str = "gripper",
    hold: str = "last",
    initial: float | None = -1.0,
    doc: str = "+1 closes, -1 opens",
) -> ActionGroup:
    return ActionGroup(
        name=name,
        components=["grip"],
        low=[-1.0],
        high=[1.0],
        mode="gripper",
        units="normalized",
        hold=hold,
        initial=None if initial is None else [initial],
        doc=doc,
    )


def move_to_skill(
    arm: str = "arm", gripper: str | None = "gripper", aliases: list[str] | None = None
) -> Skill:
    binds = {"arm": f"{arm}.ee_delta"}
    if gripper:
        binds["gripper"] = gripper
    args = [
        SkillArg("x", "float", "m"),
        SkillArg("y", "float", "m"),
        SkillArg("z", "float", "m"),
    ]
    if gripper:
        args.append(
            SkillArg(
                "grip", "float", optional=True, doc="gripper command held while moving"
            )
        )
    args += [
        SkillArg("max_steps", "int", optional=True, default=100),
        SkillArg("tol", "float", "m", optional=True, default=0.01),
    ]
    return Skill(
        name=f"{arm}.move_to",
        args=args,
        aliases=list(aliases or []),
        impl="builtin",
        binds=binds,
        preconditions=["skills are enabled for the task"],
        doc="Servo the end effector toward a world point; stops when within tol, when it stalls, or after max_steps.",
    )


def grip_skill(gripper: str = "gripper", aliases: list[str] | None = None) -> Skill:
    return Skill(
        name=f"{gripper}.set",
        args=[
            SkillArg("value", "float", doc="+1 close, -1 open"),
            SkillArg("steps", "int", optional=True, default=15),
        ],
        aliases=list(aliases or []),
        impl="builtin",
        binds={"gripper": gripper},
        preconditions=["skills are enabled for the task"],
        doc="Hold the arm still and command the gripper for a number of steps.",
    )
