"""`robo`: the only way an agent touches the robot. Standard library only; this file is copied as-is into agent images.

Finds the episode socket from $ROBO_SOCKET (or $ROBOUSE_SOCKET, or ./.robouse/socket) and sends one JSON request.

  robo info                                     the robot: action groups, skills, sensors, budgets
  robo observe [--image] [--camera C|all]       state; --image also saves camera PNGs and prints their paths
  robo act GROUP=V1,V2,... [GROUP=...] [--repeat N]
                                                one step (N steps, N <= the task's max_repeat) commanding the named
                                                action groups; groups you do not name hold (see `robo info`)
  robo act V1 V2 ... [--repeat N]               the whole flat action vector, in `robo info` order
  robo skill NAME [ARG ...] [key=value ...]     run a named skill listed by `robo info`
  robo done ["message"]                         end the episode and ask for scoring
  robo give-up ["message"]                      end the episode without claiming success
  robo move-to X Y Z [--grip G]                 shorthand for the arm's move_to skill
  robo grip G [--steps N]                       shorthand for the gripper's set skill (+1 close, -1 open)
Add --json to any command for raw JSON output.

Examples:
  robo act arm.ee_delta=0.2,0,-0.1 gripper=1 --repeat 5
  robo act left.ee_delta=0,0.3,0 right.gripper=-1
  robo skill arm.move_to 0.1 0.6 0.2 grip=1
"""

from __future__ import annotations

import json
import os
import socket
import sys

PROTOCOL = 2


def _socket_path() -> str | None:
    for var in ("ROBO_SOCKET", "ROBOUSE_SOCKET"):
        if os.environ.get(var):
            return os.environ[var]
    p = os.path.join(os.getcwd(), ".robouse", "socket")
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    return None


def _send(req: dict) -> dict:
    path = _socket_path()
    if not path:
        return {"ok": False, "error": "no episode: ROBO_SOCKET is not set"}
    token = os.environ.get("ROBO_ORACLE_TOKEN") or os.environ.get(
        "ROBOUSE_ORACLE_TOKEN"
    )
    if token:  # set only for the reference solution
        req = {**req, "token": token}
    role = os.environ.get("ROBO_ROLE")
    if role and "role" not in req:
        req = {**req, "role": role}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(float(os.environ.get("ROBO_TIMEOUT_S", "120")))
    try:
        s.connect(path)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode() or '{"ok": false, "error": "empty response"}')
    except (FileNotFoundError, ConnectionRefusedError):
        return {
            "ok": False,
            "error": "episode server is not running (episode may have finished)",
        }
    except OSError as e:  # timeout, permission, ...
        return {"ok": False, "error": f"episode server unreachable: {e}"}
    except ValueError:
        return {"ok": False, "error": "bad response from the episode server"}
    finally:
        s.close()


# ---- output -------------------------------------------------------------------------------------------------


def _fmt_state(st: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in st.items() if k != "obs_vector")


def _fmt_embodiment(e: dict) -> list[str]:
    out = [
        f"embodiment: {e.get('name')} ({e.get('kind')})"
        + (f", one step = {e['step_s']} s" if e.get("step_s") else "")
    ]
    if e.get("doc"):
        out.append(f"  {e['doc']}")
    groups = e.get("action_groups") or []
    if groups:
        out.append(
            "action groups (robo act GROUP=V1,V2,...; the flat vector is these in order):"
        )
        for g in groups:
            rng = f"[{', '.join(_n(x) for x in g['low'])}] .. [{', '.join(_n(x) for x in g['high'])}]"
            line = f"  {g['name']}[{g.get('dim', len(g['components']))}] {','.join(g['components'])}  {rng}  mode={g.get('mode')}"
            if g.get("units"):
                line += f"  units: {g['units']}"
            line += f"  hold={g.get('hold', 'zero')}"
            out.append(line)
            if g.get("doc"):
                out.append(f"      {g['doc']}")
    skills = e.get("skills") or []
    if skills:
        out.append("skills (robo skill NAME ARGS):")
        for s in skills:
            args = " ".join(
                (f"[{a['name']}]" if a.get("optional") else a["name"])
                + (f":{a['type']}" if a.get("type") else "")
                for a in s.get("args", [])
            )
            alias = f"  (alias: {', '.join(s['aliases'])})" if s.get("aliases") else ""
            out.append(f"  {s['name']} {args}{alias}")
            if s.get("doc"):
                out.append(f"      {s['doc']}")
    sensors = e.get("sensors") or {}
    if sensors.get("cameras"):
        out.append(
            "cameras: "
            + ", ".join(
                f"{c['name']} ({c.get('mount', 'world')})" for c in sensors["cameras"]
            )
        )
    if sensors.get("proprioception"):
        out.append(
            "proprioception: " + ", ".join(f["name"] for f in sensors["proprioception"])
        )
    b = e.get("budgets") or {}
    if b:
        out.append("budgets: " + ", ".join(f"{k}={v}" for k, v in b.items()))
    return out


def _n(x) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def _print(resp: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(resp))
        return 0 if resp.get("ok") else 1
    if not resp.get("ok"):
        print(f"error: {resp.get('error')}")
        return 1
    r = resp.get("result", {})
    if not isinstance(r, dict):
        print(r)
        return 0
    for k, v in r.items():
        if k == "state" and isinstance(v, dict):
            print("state:")
            print(_fmt_state(v))
        elif k == "embodiment" and isinstance(v, dict):
            print("\n".join(_fmt_embodiment(v)))
        elif k == "action" and isinstance(v, dict):
            print(f"action: {v['names']} low={v['low']} high={v['high']}")
            if v.get("doc"):
                print(f"  {v['doc']}")
        elif k == "cameras" and isinstance(v, list):
            print("camera calibration:")
            for c in v:
                print(f"  {json.dumps(c)}")
        else:
            print(f"{k}: {v}")
    return 0


# ---- argument parsing ---------------------------------------------------------------------------------------


def _parse_groups(tokens: list[str]) -> dict:
    groups: dict[str, list[float]] = {}
    for t in tokens:
        name, _, vals = t.partition("=")
        if not name or not vals:
            raise ValueError(f"expected GROUP=V1,V2,..., got {t!r}")
        if name in groups:
            raise ValueError(f"group {name!r} given twice")
        groups[name] = [float(x) for x in vals.replace(" ", "").split(",") if x != ""]
    return groups


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    def opt(name: str, default=None, cast=float):
        if name in args:
            i = args.index(name)
            v = cast(args[i + 1])
            del args[i : i + 2]
            return v
        return default

    cmd, rest = args[0], args[1:]
    try:
        if cmd == "info":
            return _print(_send({"op": "info"}), as_json)
        if cmd == "status":
            return _print(_send({"op": "status"}), as_json)
        if cmd == "observe":
            args = rest
            cam = opt("--camera", None, str)
            if os.environ.get("ROBOUSE_TEXT_ONLY") and ("--image" in args or cam):
                print(
                    "note: this harness's model cannot read images, so --image and --camera are ignored",
                    file=sys.stderr,
                )
                return _print(
                    _send({"op": "observe", "image": False, "camera": None}), as_json
                )
            return _print(
                _send(
                    {
                        "op": "observe",
                        "image": "--image" in args or bool(cam),
                        "camera": cam,
                    }
                ),
                as_json,
            )
        if cmd == "act":
            args = rest
            rep = opt("--repeat", 1, int)
            if args and all("=" in a for a in args):
                return _print(
                    _send({"op": "act", "groups": _parse_groups(args), "repeat": rep}),
                    as_json,
                )
            if any("=" in a for a in args):
                raise ValueError(
                    "use either GROUP=VALUES arguments or a flat list of numbers, not both"
                )
            return _print(
                _send({"op": "act", "action": [float(x) for x in args], "repeat": rep}),
                as_json,
            )
        if cmd in ("move-to", "move_to"):
            args = rest
            grip = opt("--grip", None, float)
            ms = opt("--max-steps", 100, int)
            tol = opt("--tol", 0.01, float)
            return _print(
                _send(
                    {
                        "op": "move_to",
                        "pos": [float(x) for x in args[:3]],
                        "grip": grip,
                        "max_steps": ms,
                        "tol": tol,
                    }
                ),
                as_json,
            )
        if cmd == "grip":
            args = rest
            steps = opt("--steps", 15, int)
            return _print(
                _send({"op": "grip", "value": float(args[0]), "steps": steps}), as_json
            )
        if cmd == "skill":
            if not rest:
                raise IndexError("skill needs a name")
            return _print(
                _send({"op": "skill", "name": rest[0], "args": rest[1:]}), as_json
            )
        if cmd == "done":
            return _print(_send({"op": "done", "text": " ".join(rest)}), as_json)
        if cmd in ("give-up", "give_up"):
            return _print(_send({"op": "give_up", "text": " ".join(rest)}), as_json)
    except (ValueError, IndexError) as e:
        print(f"error: bad arguments ({e}); run `robo help`")
        return 2
    print(f"error: unknown command {cmd!r}; run `robo help`")
    return 2


if __name__ == "__main__":
    sys.exit(main())
