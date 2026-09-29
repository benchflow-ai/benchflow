"""The trusted episode server.

One `EpisodeServer` owns one simulator (a `SimBackend`), answers one JSON request per Unix-socket connection (the
`robo` command is the client), enforces the step and wall-clock budgets, records the video and a per-step
state/action/reward trace, and decides success itself when the episode ends. The agent only sees what `observe`
returns.

Files written to the run directory (see docs/embodied.md):
  episode.json      header: task, seed, embodiment, observation mode, initial state
  steps.jsonl       one line per simulator step: action by group, reward, success, state after the step, frame
  trace.jsonl       every request and response (images referenced by path; never the oracle token)
  result.json       success, outcome, steps used, return, the agent's final message
  frames.jsonl      per video frame: wall time, budget steps and simulator steps completed when it was rendered
  video_index.json  the video, its fps and the step <-> frame map
  recording.mp4     the episode video

Needs numpy; Pillow and imageio for images and video. Nothing else from BenchFlow.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend import SimBackend, StepResult
from .protocol import PROTOCOL_VERSION, legacy_skill_request
from .skills import builtin_for
from .spec import Embodiment, SpecError, bind_skill_args

FPS = 30
MAX_REPEAT = 50
LEGACY_VISIBLE = ["hand_pos", "gripper_open"]
ENDING_MESSAGES = {
    "budget_exhausted": "step budget exhausted",
    "success_reached": "goal reached, episode scored as solved",
    "wall_time_exhausted": "wall-clock budget exhausted",
}


@dataclass
class EpisodeConfig:
    """Per-task episode settings. `from_task` reads them from a task's format block (e.g. `robouse:`)."""

    task_id: str = ""
    seed: int = 0
    max_steps: int = 500
    max_wall_s: float = 1800.0
    max_repeat: int = MAX_REPEAT
    max_skill_steps: int = 150
    settle_steps: int = 10
    frame_every: int = 2
    skills_enabled: bool = True
    success_mode: str = "final"  # "final": judged after `robo done` and the settle period; "first": first success
    obs_mode: str = "state"  # "vision": only visible fields + images (the oracle token still sees everything)
    visible_fields: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    uncalibrated_cameras: list[str] = field(default_factory=list)
    roles: dict[str, dict] = field(default_factory=dict)
    trace_state: bool = True
    linger_s: float = 5.0
    show_max_repeat: bool = False

    @classmethod
    def from_task(
        cls,
        block: dict,
        embodiment: Embodiment,
        *,
        seed: int | None = None,
        default_camera: str | None = None,
        max_wall_s: float | None = None,
    ) -> EpisodeConfig:
        b = embodiment.budgets
        cams = list(block.get("cameras") or [])
        if not cams:
            if default_camera:
                cams = [default_camera]
            elif embodiment.sensors.cameras:
                cams = [embodiment.sensors.cameras[0].name]
        visible = block.get("visible_fields")
        if visible is None:
            visible = [f.name for f in embodiment.sensors.proprioception] or list(
                LEGACY_VISIBLE
            )
        return cls(
            task_id=str(block.get("id", "")),
            seed=int(block.get("seed", 0) if seed is None else seed),
            max_steps=int(block.get("max_steps", b.max_steps)),
            max_wall_s=float(
                max_wall_s if max_wall_s is not None else (b.max_wall_s or 1800.0)
            ),
            max_repeat=max(
                1, min(MAX_REPEAT, int(block.get("max_repeat", b.max_repeat)))
            ),
            max_skill_steps=int(block.get("max_skill_steps", b.max_skill_steps)),
            settle_steps=int(block.get("settle_steps", b.settle_steps)),
            frame_every=max(1, int(block.get("frame_every", 2))),
            skills_enabled=bool(block.get("skills", True)),
            success_mode=str(block.get("success_mode", embodiment.reward.success_mode)),
            obs_mode=str(block.get("obs_mode", "state")),
            visible_fields=list(visible),
            cameras=cams,
            uncalibrated_cameras=list(block.get("uncalibrated_cameras", [])),
            roles=dict(block.get("roles") or {}),
            trace_state=bool(block.get("trace_state", True)),
            linger_s=float(block.get("linger_s", 5)),
            show_max_repeat="max_repeat" in block,
        )


class EpisodeServer:
    def __init__(
        self,
        backend: SimBackend,
        config: EpisodeConfig,
        run_dir: str | Path,
        workspace: str | Path | None = None,
        oracle_token: str | None = None,
    ):
        self.backend = backend
        self.config = config
        self.embodiment: Embodiment = backend.embodiment().validate()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(workspace) if workspace else None
        if oracle_token is None:
            oracle_token = os.environ.get("ROBO_ORACLE_TOKEN") or os.environ.get(
                "ROBOUSE_ORACLE_TOKEN", ""
            )
        self._oracle_token = oracle_token
        self.backend.reset(config.seed)
        self._save_workspace_images()
        self.steps = 0  # budget steps (skills included)
        self.sim_steps = 0  # every simulator step, including the settle period
        self.requests = 0
        self.success_ever = False
        self.finished = False
        self.outcome: str | None = None
        self.agent_text = ""
        self.result: dict = {}
        self.total_reward = 0.0
        self.t0 = time.time()
        self._last: dict[str, list[float]] = {}
        self._role: str | None = None
        self._frames: list[Any] = []
        self._frame_meta: list[dict] = []
        self._obs_count = 0
        self._trace = open(self.run_dir / "trace.jsonl", "w")  # noqa: SIM115 - open for the episode
        self._steps_f = open(self.run_dir / "steps.jsonl", "w")  # noqa: SIM115 - open for the episode
        initial = self.backend.observe()
        self.initial_state_sha256 = _state_hash(initial)
        (self.run_dir / "episode.json").write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "task": config.task_id,
                    "seed": config.seed,
                    "backend": getattr(backend, "name", type(backend).__name__),
                    "embodiment": self.embodiment.to_dict(),
                    "observation_mode": config.obs_mode,
                    "visible_fields": config.visible_fields,
                    "success_mode": config.success_mode,
                    "recording": {
                        "camera": config.cameras[0] if config.cameras else None,
                        "fps": FPS,
                        "frame_every": config.frame_every,
                    },
                    "initial_state": _jsonable(initial),
                    "initial_state_sha256": self.initial_state_sha256,
                },
                indent=2,
            )
        )
        self._record_frame(force=True)

    # ---- helpers used by skills --------------------------------------------------------------------------
    def pack(self, groups: dict[str, Any]) -> list[float]:
        return self.embodiment.pack(groups, self._last)

    def state(self, privileged: bool = False) -> dict:
        st = self.backend.observe()
        if privileged:
            extra = getattr(self.backend, "privileged_state", None)
            if callable(extra):
                st = {**st, **extra()}
        if self.config.obs_mode == "vision" and not privileged:
            st = {k: v for k, v in st.items() if k in self.config.visible_fields}
        return st

    def env_step(self, action: list[float], op: str = "act") -> StepResult:
        info = _as_step_result(self.backend.step(action))
        self.steps += 1
        self.sim_steps += 1
        success = self._success_after_step(info)
        self.success_ever = self.success_ever or success
        frame = self._record_frame()
        self._log_step(op, action, info, frame, success)
        if success and self.config.success_mode == "first" and not self.finished:
            self.finish("success_reached", "", settle=False)
        return info

    def _success_after_step(self, info: StepResult) -> bool:
        """The step's success signal; when the backend only judges on demand (sparse reward), probe it."""
        if info.success:
            return True
        if self.embodiment.reward.dense == "sparse" and getattr(
            self.backend, "probe_success", True
        ):
            try:
                return bool(self.backend.success())
            except (
                Exception
            ):  # a backend that cannot judge mid-episode keeps the step's value
                return False
        return False

    # ---- recording ---------------------------------------------------------------------------------------
    def _save_workspace_images(self) -> None:
        hook = getattr(self.backend, "workspace_images", None)
        imgs = hook() if callable(hook) else {}
        if not imgs:
            return
        for d in [self.run_dir] + ([self.workspace] if self.workspace else []):
            (d / "observations").mkdir(parents=True, exist_ok=True)
            for name, img in imgs.items():
                save_png(d / "observations" / name, img)

    def _append_frame(self, img: Any) -> int:
        import numpy as np

        self._frames.append(np.asarray(img, dtype=np.uint8))
        i = len(self._frames) - 1
        self._frame_meta.append(
            {
                "i": i,
                "t": round(time.time() - self.t0, 3),
                "step": self.steps,
                "sim_step": self.sim_steps,
            }
        )
        return i

    def _record_frame(self, force: bool = False) -> int | None:
        if force or self.steps % self.config.frame_every == 0:
            return self._append_frame(self.backend.render(None))
        return None

    def _log_step(
        self,
        op: str,
        action: list[float],
        info: StepResult,
        frame: int | None,
        success: bool,
        groups: dict[str, Any] | None = None,
    ) -> None:
        """Append the steps.jsonl line of the simulator step just taken (``sim_steps`` already counts it)."""
        reward = (
            float(info.reward)
            if self.embodiment.reward.dense == "shaped"
            else float(success)
        )
        self.total_reward += reward
        if groups is None:
            try:
                groups = self.embodiment.unpack([float(x) for x in action])
            except (
                SpecError
            ):  # a backend-specific skill action (a 4-number end-effector command)
                groups = {"_raw": [float(x) for x in action]}
            for gname, vals in groups.items():
                if gname != "_raw":
                    self._last[gname] = vals
        rec: dict[str, Any] = {
            "i": self.sim_steps - 1,
            "step": self.steps,
            "t": round(time.time() - self.t0, 3),
            "op": op,
            "action": groups,
            "reward": reward,
            "success": success,
        }
        if self._role:
            rec["role"] = self._role
        if info.info:
            rec["info"] = _jsonable(info.info)
        if self.config.trace_state:
            try:
                rec["state"] = _jsonable(self.backend.observe())
            except Exception as e:  # never let tracing break the episode
                rec["state_error"] = f"{type(e).__name__}: {e}"[:200]
        if frame is not None:
            rec["frame"] = frame
        self._steps_f.write(json.dumps(rec) + "\n")
        self._steps_f.flush()

    # ---- request handling --------------------------------------------------------------------------------
    def _role_allows(self, role: str | None, op: str) -> bool:
        if not role or role not in self.config.roles:
            return True
        allow = self.config.roles[role].get("allow", "all")
        if allow == "all":
            return True
        if isinstance(allow, str):  # a single op, not a substring match
            allow = [allow]
        eff = "skill" if op in ("move_to", "grip") else op
        return op in allow or eff in allow

    def handle(self, req: dict) -> dict:
        self.requests += 1
        privileged = bool(self._oracle_token) and req.get("token") == self._oracle_token
        req = {
            k: v for k, v in req.items() if k != "token"
        }  # never write the token to the trace
        op = req.get("op")
        role = req.get("role") if isinstance(req.get("role"), str) else None
        self._role = role
        t = round(time.time() - self.t0, 3)
        try:
            if not self._role_allows(role, str(op)):
                resp = {
                    "ok": False,
                    "error": f"role {role!r} may not use {op!r} in this task",
                }
            elif self.finished and op not in ("info", "status"):
                resp = {"ok": False, "error": f"episode finished ({self.outcome})"}
            elif op == "info":
                resp = {"ok": True, "result": self.info()}
            elif op == "status":
                resp = {
                    "ok": True,
                    "result": {
                        "finished": self.finished,
                        "outcome": self.outcome,
                        "steps": self.steps,
                    },
                }
            elif op == "observe":
                resp = {
                    "ok": True,
                    "result": self.observe(
                        bool(req.get("image", False)), privileged, req.get("camera")
                    ),
                }
            elif op == "act":
                resp = self.act(req, int(req.get("repeat", 1)))
            elif op in ("move_to", "grip"):
                resp = self.legacy_skill(req)
            elif op == "skill":
                resp = self.skill(
                    req.get("name"),
                    req.get("args") if req.get("args") is not None else [],
                )
            elif op in ("done", "give_up"):
                resp = {
                    "ok": True,
                    "result": self.finish(
                        "done" if op == "done" else "gave_up",
                        str(req.get("text", ""))[:4000],
                    ),
                }
            else:
                resp = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:  # bounded error back to the agent
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
        if not self.finished and self.steps >= self.config.max_steps:
            self.finish("budget_exhausted", "")
        result = resp.get("result")
        if (
            self.finished
            and op in ("act", "move_to", "grip", "skill")
            and resp.get("ok")
            and isinstance(result, dict)
        ):
            result["episode"] = (
                f"finished: {ENDING_MESSAGES.get(self.outcome or '', self.outcome)}"
            )
        line: dict[str, Any] = {
            "t": t,
            "steps": self.steps,
            "req": req,
            "resp": _strip(resp),
        }
        if role:
            line["role"] = role
        self._trace.write(json.dumps(line) + "\n")
        self._trace.flush()
        self._role = None
        return resp

    def info(self) -> dict:
        e = self.embodiment
        spec = e.to_dict()
        spec["budgets"] = {
            **spec.get("budgets", {}),
            "max_steps": self.config.max_steps,
            "max_repeat": self.config.max_repeat,
            "settle_steps": self.config.settle_steps,
        }
        spec["reward"] = {
            **spec.get("reward", {}),
            "success_mode": self.config.success_mode,
        }
        if not self.config.skills_enabled:
            spec.pop("skills", None)
        out: dict[str, Any] = {
            "task": self.config.task_id,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "protocol": PROTOCOL_VERSION,
            "action": {
                "names": e.flat_names(),
                "low": e.flat_low(),
                "high": e.flat_high(),
                "doc": e.doc,
            },
            "skills": self._legacy_skill_names(),
            "steps_used": self.steps,
            "max_steps": self.config.max_steps,
            "observation_mode": self.config.obs_mode,
        }
        if self.config.show_max_repeat:
            out["max_repeat"] = self.config.max_repeat
        if self.config.obs_mode == "vision":
            out["visible_fields"] = self.config.visible_fields
            out["cameras"] = self.camera_info()
        out["embodiment"] = spec
        return out

    def _legacy_skill_names(self) -> list[str]:
        if not self.config.skills_enabled:
            return []
        legacy = getattr(self.backend, "legacy_skill_names", None)
        if callable(legacy):
            return list(legacy())
        return [s.name for s in self.embodiment.skills]

    def camera_info(self) -> list[dict]:
        """Calibration of each configured camera for the images `observe` saves."""
        calib = getattr(self.backend, "camera_calibration", None)
        if not callable(calib):
            return []
        out = []
        for name in self.config.cameras:
            img = self.backend.render(name)
            h, w = img.shape[:2]
            c = calib(name, w, h)
            if c is None:
                continue
            if name in self.config.uncalibrated_cameras:
                c = {
                    "name": name,
                    "width": w,
                    "height": h,
                    "fovy_deg": c.get("fovy_deg"),
                    "projection": None,
                    "calibrated": False,
                }
            out.append(c)
        return out

    def observe(
        self, image: bool, privileged: bool = False, camera: str | None = None
    ) -> dict:
        out: dict[str, Any] = {
            "state": self.state(privileged),
            "steps_used": self.steps,
            "max_steps": self.config.max_steps,
        }
        if image or self.config.obs_mode == "vision":
            if camera == "all":
                cams: list[str | None] = list(self.config.cameras)
            elif camera in self.config.cameras:
                cams = [camera]
            else:
                cams = (
                    list(self.config.cameras)
                    if self.config.obs_mode == "vision"
                    else [None]
                )
            paths = []
            for cam in cams:
                self._obs_count += 1
                img = self.backend.render(cam)
                safe = str(cam).replace("/", "_").replace(os.sep, "_") if cam else ""
                name = (
                    f"obs_{self._obs_count:03d}" + (f"_{safe}" if cam else "") + ".png"
                )
                (self.run_dir / "observations").mkdir(exist_ok=True)
                save_png(self.run_dir / "observations" / name, img)
                if self.workspace:
                    (self.workspace / "observations").mkdir(parents=True, exist_ok=True)
                    save_png(self.workspace / "observations" / name, img)
                    paths.append(str(self.workspace / "observations" / name))
                else:
                    paths.append(str(self.run_dir / "observations" / name))
            out["image_path"] = paths[0] if len(paths) == 1 else paths
        return out

    def act(self, req: dict, repeat: int) -> dict:
        e = self.embodiment
        if req.get("groups") is not None:
            groups = req["groups"]
            if not isinstance(groups, dict) or not groups:
                return {
                    "ok": False,
                    "error": "groups must be an object {group: [numbers]}",
                }
            try:
                action = e.pack(groups, self._last)
            except SpecError as err:
                return {"ok": False, "error": str(err)}
        else:
            action = req.get("action")
            if (
                not isinstance(action, list)
                or len(action) != e.dim
                or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in action
                )
            ):
                return {
                    "ok": False,
                    "error": f"action must be a list of {e.dim} numbers {e.flat_names()}",
                }
        if not all(math.isfinite(float(x)) for x in action):
            return {
                "ok": False,
                "error": "action values must be finite numbers (no nan or inf)",
            }
        repeat = max(1, min(self.config.max_repeat, repeat))
        n = 0
        for _ in range(repeat):
            if self.steps >= self.config.max_steps or self.finished:
                break
            self.env_step(action, op="act")
            n += 1
        return {
            "ok": True,
            "result": {
                "executed_steps": n,
                "state": self.state(),
                "steps_used": self.steps,
            },
        }

    def legacy_skill(self, req: dict) -> dict:
        op = req.get("op")
        try:
            alias, kw = legacy_skill_request(req)
        except ValueError as err:
            return {"ok": False, "error": str(err)}
        skill = (
            self.embodiment.find_skill(alias) if self.config.skills_enabled else None
        )
        if skill is None:
            return {
                "ok": False,
                "error": f"{op} is not available for this task; use act",
            }
        declared = {
            a.name for a in skill.args
        }  # protocol-1 clients always send max_steps / tol
        kw = {k: v for k, v in kw.items() if k in declared}
        return self._run_skill(skill, kw)

    def skill(self, name: Any, args: Any) -> dict:
        if not isinstance(name, str) or not isinstance(args, (list, dict)):
            return {"ok": False, "error": "skill needs a name and a list of arguments"}
        run = getattr(self.backend, "run_skill", None)
        if not self.config.skills_enabled:
            return {
                "ok": False,
                "error": "no named skills in this task; see `robo info`",
            }
        skill = self.embodiment.find_skill(name)
        if skill is None:
            # a backend whose skills are only known at run time (e.g. a remote worker)
            if callable(run) and (
                not self.embodiment.skills
                or getattr(self.backend, "open_skill_set", False)
            ):
                return self._backend_skill(name, args, None)
            if not self.embodiment.skills and not callable(run):
                return {
                    "ok": False,
                    "error": "no named skills in this task; see `robo info`",
                }
            names = ", ".join(s.name for s in self.embodiment.skills)
            return {"ok": False, "error": f"unknown skill {name!r}; skills: {names}"}
        return self._run_skill(skill, args)

    def _run_skill(self, skill, args: Any) -> dict:
        try:
            kw = bind_skill_args(skill, args)
        except SpecError as err:
            return {"ok": False, "error": str(err)}
        if skill.impl == "builtin":
            ctrl = builtin_for(skill)
            if ctrl is None:
                return {
                    "ok": False,
                    "error": f"no built-in controller for {skill.name}",
                }
            return ctrl(self, skill, kw)
        return self._backend_skill(skill.name, args, skill)

    def _backend_skill(self, name: str, args: Any, skill) -> dict:
        """A backend skill counts as one step of the budget, however long it runs in the simulator; a closed-loop
        backend skill (`start_skill`) counts every control step it runs."""
        if isinstance(args, dict):
            order = [a.name for a in skill.args] if skill else list(args)
            args = [args[k] for k in order if k in args]
        start = getattr(self.backend, "start_skill", None)
        if callable(start):
            return self._closed_loop_skill(start, skill.name if skill else name, args)
        run = getattr(self.backend, "run_skill", None)
        if not callable(run):
            return {
                "ok": False,
                "error": f"skill {name!r} is not implemented by this simulator",
            }
        if self.steps >= self.config.max_steps:
            return {"ok": False, "error": "step budget exhausted"}
        out = run(skill.name if skill else name, [str(a) for a in args])
        self.steps += 1
        self.sim_steps += 1
        pop = getattr(self.backend, "pop_frames", None)
        for fr in pop() if callable(pop) else []:
            self._append_frame(fr)
        frame = self._record_frame(force=True)
        # a backend skill is one step of the trace too: same success probe, reward and `first` rule
        success = self._success_after_step(
            StepResult(success=bool(out.get("success", False)))
        )
        self.success_ever = self.success_ever or success
        self._log_step(
            f"skill:{skill.name if skill else name}",
            [],
            StepResult(success=success, info=dict(out)),
            frame,
            success,
            groups={"skill": [str(a) for a in args]},
        )
        if success and self.config.success_mode == "first" and not self.finished:
            self.finish("success_reached", "", settle=False)
        return {
            "ok": True,
            "result": {
                **out,
                "steps_used": self.steps,
                "max_steps": self.config.max_steps,
            },
        }

    def _closed_loop_skill(self, start, name: str, args: list) -> dict:
        """A backend skill written as a generator (`start_skill(name, args)`) that yields one full action per control
        step and returns a result dict. Every yielded action runs as an ordinary step (budget, trace, video frames,
        success checks), so a skill never moves the robot in a way `act` could not."""
        try:
            gen = start(name, [str(a) for a in args])
        except ValueError as err:
            return {"ok": False, "error": str(err)}
        n, out = 0, {}
        while True:
            if self.steps >= self.config.max_steps or self.finished:
                gen.close()
                out = {
                    "stopped": "step budget exhausted"
                    if self.steps >= self.config.max_steps
                    else "episode finished"
                }
                break
            try:
                action = next(gen)
            except StopIteration as stop:
                out = stop.value or {}
                break
            except (
                Exception
            ) as err:  # a skill that fails part-way reports it; its steps still count
                gen.close()
                out = {"error": f"{type(err).__name__}: {err}"[:300]}
                break
            self.env_step([float(x) for x in action], op=f"skill:{name}")
            n += 1
        return {
            "ok": True,
            "result": {
                "skill": name,
                **out,
                "executed_steps": n,
                "state": self.state(),
                "steps_used": self.steps,
                "max_steps": self.config.max_steps,
            },
        }

    # ---- ending ------------------------------------------------------------------------------------------
    def finish(self, outcome: str, text: str, settle: bool = True) -> dict:
        if self.finished:
            return {"outcome": self.outcome}
        hold_fn = getattr(self.backend, "hold_action", None)
        hold = (
            list(hold_fn())
            if callable(hold_fn)
            else self.embodiment.hold_action(self._last)
        )
        for _ in range(
            self.config.settle_steps if settle else 0
        ):  # success must still hold afterwards
            try:
                info = self.backend.step(hold)
            except (
                Exception
            ):  # a backend that cannot step further is judged on its current state
                break
            self.sim_steps += 1
            frame = self._append_frame(self.backend.render(None))
            info = _as_step_result(info)
            self._log_step("settle", hold, info, frame, self._success_after_step(info))
        judge = getattr(
            self.backend, "judge", None
        )  # e.g. safety tasks, where refusing is the rewarded ending
        judge_error = None
        try:
            if callable(judge):
                success = bool(judge(outcome, text))
            elif self.config.success_mode == "first":
                success = bool(self.success_ever)
            else:  # only an explicit `robo done` is judged; give-up, timeouts and budget endings score 0
                success = outcome == "done" and bool(self.backend.success())
        except (
            Exception
        ) as err:  # a simulator that died cannot be judged: score 0 and record why
            success, judge_error = False, f"{type(err).__name__}: {err}"[:300]
        detail = getattr(self.backend, "last_judge", None)
        self.finished, self.outcome, self.agent_text = True, outcome, text
        self.result = {
            "task": self.config.task_id,
            "seed": self.config.seed,
            "success": success,
            "success_ever": self.success_ever,
            "success_mode": self.config.success_mode,
            "outcome": outcome,
            "steps_used": self.steps,
            "max_steps": self.config.max_steps,
            "requests": self.requests,
            "agent_text": text,
            "wall_time_s": round(time.time() - self.t0, 2),
            "protocol": PROTOCOL_VERSION,
            "embodiment": self.embodiment.name,
            "embodiment_mode": self.embodiment.mode,
            "sim_steps": self.sim_steps,
            "return": round(self.total_reward, 6),
            "initial_state_sha256": self.initial_state_sha256,
            **({"judge_error": judge_error} if judge_error else {}),
            **({"judge_detail": _jsonable(detail)} if detail else {}),
        }
        (self.run_dir / "result.json").write_text(json.dumps(self.result, indent=2))
        self._steps_f.flush()
        self._write_video()
        return {"outcome": outcome, "episode": "finished"}

    def _write_video(self) -> None:
        with open(self.run_dir / "frames.jsonl", "w") as f:
            for m in self._frame_meta:
                f.write(json.dumps(m) + "\n")
        try:
            import imageio.v2 as imageio
        except (
            ImportError
        ):  # no video without imageio; the frame index is still written
            imageio = None
        index = {
            "video": "recording.mp4" if self._frames and imageio is not None else None,
            "fps": FPS,
            "camera": self.config.cameras[0] if self.config.cameras else None,
            "n_frames": len(self._frame_meta),
            "frames": [
                [m["i"], m["t"], m["step"], m["sim_step"]] for m in self._frame_meta
            ],
            "columns": ["frame", "t", "step", "sim_step"],
        }
        (self.run_dir / "video_index.json").write_text(json.dumps(index))
        if not self._frames or imageio is None:
            self._frames = []
            return
        with imageio.get_writer(
            self.run_dir / "recording.mp4",
            fps=FPS,
            codec="libx264",
            quality=7,
            macro_block_size=16,
            ffmpeg_log_level="error",
        ) as w:
            for fr in self._frames:
                w.append_data(fr)
        self._frames = []

    def close(self) -> None:
        if not self.finished:
            self.finish("agent_exited", "")
        self._trace.close()
        self._steps_f.close()
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


def _as_step_result(info: Any) -> StepResult:
    """Tolerate a backend's own result type with the same fields (success, reward, info or extra)."""
    if isinstance(info, StepResult):
        return info
    return StepResult(
        bool(info.success),
        float(getattr(info, "reward", 0.0)),
        dict(getattr(info, "info", None) or getattr(info, "extra", None) or {}),
    )


def save_png(path: str | Path, img: Any) -> None:
    """Save an HxWx3 (or HxW) uint8 image as PNG: with Pillow when installed, else a small zlib encoder."""
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        Image.fromarray(arr).save(path)
        return
    import struct
    import zlib

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y, :, :3].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def _strip(resp: dict) -> dict:
    """Drop the raw obs vector from the trace to keep it readable."""
    r = json.loads(json.dumps(resp, default=str))
    st = (
        (r.get("result") or {}).get("state")
        if isinstance(r.get("result"), dict)
        else None
    )
    if isinstance(st, dict):
        st.pop("obs_vector", None)
    return r


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=_default))


def _default(o: Any) -> Any:
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(o, "item", None)
    if callable(item):
        return item()
    return str(o)


def _state_hash(state: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(state), sort_keys=True).encode()
    ).hexdigest()


def serve(
    server: EpisodeServer,
    sock_path: str,
    *,
    ready_file: str | Path | None = None,
) -> dict:
    """Serve one episode on a Unix socket until it ends (plus `linger_s`) or a `shutdown` request arrives."""
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(8)
    srv.settimeout(1.0)
    if ready_file:
        Path(ready_file).write_text(sock_path)
    deadline = time.time() + server.config.max_wall_s
    stop_after_finish = None
    try:
        while True:
            if time.time() > deadline and not server.finished:
                server.finish("wall_time_exhausted", "")
            if server.finished and stop_after_finish is None:
                stop_after_finish = time.time() + server.config.linger_s
            if stop_after_finish and time.time() > stop_after_finish:
                break
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            with conn:
                conn.settimeout(10)
                buf = b""
                try:
                    while not buf.endswith(b"\n") and len(buf) < 1_000_000:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                    req = json.loads(buf.decode() or "{}")
                    if req.get("op") == "shutdown":
                        conn.sendall(b'{"ok": true}\n')
                        break
                    resp = server.handle(req)
                except Exception as e:
                    resp = {"ok": False, "error": f"bad request: {e}"[:300]}
                # the client may have gone away (e.g. its shell command timed out); keep serving
                with contextlib.suppress(OSError):
                    conn.sendall((json.dumps(resp, default=_default) + "\n").encode())
    finally:
        server.close()
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
    return server.result
