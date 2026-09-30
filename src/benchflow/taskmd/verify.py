"""The ``taskmd`` verifier strategy: how a materialized draft 2 package is graded.

A materialized package's ``verifier/verifier.md`` selects this strategy
(``benchflow.taskmd.materialize``). ``Verifier.verify()`` calls
:func:`verify_taskmd`, which follows docs/runtime/judging.md's order:

1. **Offline.** When the verifier's network is ``"none"``, it runs offline: a
   separate verifier sandbox is started without a network, and a shared
   verifier's container is taken offline (``iptables``) before ``test.sh`` runs
   when the agent's run left it online, or the trial is not scored.
2. **Kept copy.** With model judges, the saved outputs (``[sandbox] outputs``,
   or the working folder) are copied to the host before any script runs:
   what the judges read, and ``submission_tree``.
3. **Script verifier.** ``test.sh`` runs in the task's working folder (not in
   ``/verifier``). Without a rubric its ``reward.txt``/``reward.json`` is the
   reward, as in Harbor. With one, any reward file it writes is set aside
   (``script-reward.*``) and ``/logs/verifier/ctrf.json`` is read.
4. **Model judges** (``benchflow.taskmd.judging``), after the tests, skipping
   model-judged criteria once a test gate failed (``lazy``).
5. **Review.** ``review.json`` (review-1), and ``reward.json``/``reward.txt``
   with the headline reward, ``strict``, and ``partial``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.taskmd import grading, judging
from benchflow.taskmd import trajectory as traj
from benchflow.taskmd._util import listed, table
from benchflow.taskmd._vendor import judgeprompt as jp
from benchflow.taskmd._vendor import taskmd as ref
from benchflow.taskmd.materialize import (
    judge_package_dir,
    shared_rubrics,
    taskmd_metadata,
)

logger = logging.getLogger(__name__)

JUDGE_DIR = "taskmd-judge"
RUNNER_UID = 65534
KEPT_CAPS = {"bytes": 104_857_600, "files": 10_000, "depth": 32}


class TaskMdVerifierError(RuntimeError):
    """Grading could not finish; the trial is left unscored (a verifier error)."""


def _offline_script() -> str:
    """Take this container offline unless it already is. Prints the outcome; exits nonzero when it cannot."""
    return r"""
if [ -z "$(awk 'NR > 2 { sub(/:.*/, "", $1); if ($1 != "lo") print $1 }' /proc/net/dev 2>/dev/null)" ]; then
  echo "offline: no network interface but lo"; exit 0
fi
if command -v iptables >/dev/null 2>&1 && iptables -I OUTPUT 1 ! -o lo -j REJECT 2>/dev/null; then
  command -v ip6tables >/dev/null 2>&1 && ip6tables -I OUTPUT 1 ! -o lo -j REJECT 2>/dev/null
  echo "offline: iptables rejects every packet leaving by an interface but lo"; exit 0
fi
if command -v bash >/dev/null 2>&1; then
  if bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null || bash -c 'exec 3<>/dev/tcp/8.8.8.8/53' 2>/dev/null; then
    echo "online: the container reaches the internet and iptables is not available"; exit 1
  fi
  echo "offline: the platform blocks egress"; exit 0
fi
echo "unknown: no iptables and no bash to check the network"; exit 1
"""


async def take_offline(sandbox: Any) -> str:
    """Make sure the verifier runs without a network; return how, or raise."""
    from benchflow.sandbox.lockdown import _exec_return_code

    result = await sandbox.exec(_offline_script(), user="root", timeout_sec=60)
    output = (getattr(result, "stdout", "") or "").strip()
    if _exec_return_code(result) != 0:
        raise TaskMdVerifierError(
            f'[verifier] network is "none", and BenchFlow could not take the verifier offline: {output}'
        )
    return output


def kept_paths(config: dict[str, Any]) -> list[str]:
    """What the runtime saves: each declared output's path, or the working folder."""
    sandbox = table(config.get("sandbox"))
    paths = [
        p for p in (jp.restored_path(o) for o in listed(sandbox.get("outputs"))) if p
    ]
    if paths:
        return paths
    workdir = sandbox.get("workdir")
    return [jp.normalize_path(workdir)] if isinstance(workdir, str) else []


def _within_caps(root: Path) -> str | None:
    total, files, depth = 0, 0, 0
    if root.is_file():
        return None if root.stat().st_size <= KEPT_CAPS["bytes"] else "over 100 MB"
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            total += path.stat().st_size
            depth = max(depth, len(path.relative_to(root).parts))
    if total > KEPT_CAPS["bytes"]:
        return f"{total} bytes, over the 100 MB cap"
    if files > KEPT_CAPS["files"]:
        return f"{files} files, over the 10,000 cap"
    if depth > KEPT_CAPS["depth"]:
        return f"a depth of {depth}, over the cap of 32"
    return None


async def copy_kept(sandbox: Any, paths: list[str], fs_root: Path) -> dict[str, str]:
    """Copy the saved outputs to ``fs_root`` at their absolute paths. Returns refused path -> why."""
    refused: dict[str, str] = {}
    for path in paths:
        probe = await sandbox.exec(
            f"if [ -L {shlex.quote(path)} ]; then echo link; elif [ -d {shlex.quote(path)} ]; then echo dir; "
            f"elif [ -f {shlex.quote(path)} ]; then echo file; elif [ -e {shlex.quote(path)} ]; then echo special; else echo none; fi",
            user="root",
            timeout_sec=30,
        )
        kind = (getattr(probe, "stdout", "") or "").strip()
        target = fs_root / path.lstrip("/")
        if kind == "none":
            continue
        if kind in ("link", "special"):
            refused[path] = (
                f"a {'symbolic link' if kind == 'link' else 'special file'} is never saved"
            )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "dir":
            await sandbox.download_dir(source_dir=path, target_dir=target)
        else:
            await sandbox.download_file(path, target)
        # Only regular files and folders are kept; a link or special file anywhere is refused.
        for found in sorted(target.rglob("*"), reverse=True) if target.is_dir() else []:
            if found.is_symlink() or not (found.is_file() or found.is_dir()):
                found.unlink(missing_ok=True)
                refused[
                    str(PurePosixPath(path) / found.relative_to(target).as_posix())
                ] = "a symbolic link or special file is never saved"
        why = _within_caps(target)
        if why:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            refused[path] = why
    return refused


# The verifier's files and any oracle, removed from the judge's sandbox before its first call.
REMOVED = (
    "/verifier",
    "/tests",
    "/oracle",
    "/solution",
    "/oracle_backup",
    "/solution_oracle_backup",
)
# The verifier's and the solver's logs. BenchFlow bind-mounts them from the
# host, so they are hidden from the runner's uid, never deleted.
HIDDEN = ("/logs/verifier", "/logs/agent")


def _hide_script(
    root: str = "",
    mountinfo: str = "/proc/self/mountinfo",
    hidden: tuple[str, ...] = HIDDEN,
) -> str:
    """Shell that takes the verifier's files out of the judge's sandbox.

    Each path of :data:`REMOVED` is deleted, unless it is or holds a mount
    point: deleting through a bind mount deletes the host's files, so such a
    path is hidden instead (mode 700; the runner's uid cannot enter it), as is
    each path of ``hidden``. The hidden paths' modes are saved in
    ``/run/taskmd-judge/modes`` for :func:`restore_hidden`. ``root``
    prefixes every path, so tests can run the script against a folder.
    """
    state = shlex.quote(f"{root}/run/taskmd-judge")
    removed = " ".join(shlex.quote(root + p) for p in REMOVED)
    hide_paths = " ".join(shlex.quote(root + p) for p in hidden)
    return f"""
set -e
mkdir -p {state} && chmod 700 {state}
: > {state}/modes
mounts=$(awk '{{print $5}}' {shlex.quote(mountinfo)} 2>/dev/null || true)
held() {{ printf '%s\\n' "$mounts" | awk -v p="$1" '$0 == p || index($0, p "/") == 1 {{ f = 1 }} END {{ exit !f }}'; }}
hide() {{
  m=$(stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1")
  echo "$m $1" >> {state}/modes
  chmod 700 "$1"
}}
for p in {removed}; do
  if [ -L "$p" ]; then rm -f "$p"
  elif [ -e "$p" ]; then
    if held "$p"; then hide "$p"; else rm -rf "$p"; fi
  fi
done
for p in {hide_paths}; do
  if [ -e "$p" ]; then hide "$p"; fi
done
"""


def _restore_script(root: str = "") -> str:
    """Shell that gives the hidden paths back their modes."""
    modes = shlex.quote(f"{root}/run/taskmd-judge/modes")
    return f"""
if [ -f {modes} ]; then
  while read -r m p; do chmod "$m" "$p" 2>/dev/null || true; done < {modes}
  : > {modes}
fi
"""


async def restore_hidden(sandbox: Any) -> None:
    """Give the paths hidden from the judge's runner their modes back (best effort; a no-op when none were)."""
    try:
        await sandbox.exec(_restore_script(), user="root", timeout_sec=60)
    except Exception as exc:  # the sandbox is discarded next; no score depends on this
        logger.warning("Could not restore the paths hidden from the judge: %s", exc)


@dataclass
class SandboxRunner:
    """judge-tools@1's ``run``: each call in a fresh copy of the submission's environment.

    The runner is the separate verifier sandbox: a fresh container of the
    task's image holding only the kept copy. Before the first call BenchFlow
    removes ``/verifier``, ``/tests``, and any oracle from it, hides
    ``/logs/verifier`` and ``/logs/agent`` (the verifier's and the solver's
    logs, mounted from the host) from the runner's uid unless they hold a
    kept path or a view, and stashes the kept copy; before every call it
    restores the kept copy, a fresh working folder and HOME, then runs
    ``bash --noprofile --norc -c`` as an unprivileged uid with no network,
    stdin at /dev/null, and stdout and stderr merged, under the call's
    timeout, and kills whatever the command left. What persists between calls
    outside the kept copy is the container's other writable paths: a fresh
    container per call is not provided. :func:`restore_hidden` gives the
    hidden paths their modes back.
    """

    sandbox: Any
    kept: list[str]
    views: dict[str, bytes]
    path: str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    drop: str = (
        "root"  # how commands lose root: setpriv, runuser, or root when neither exists
    )
    ready: bool = False

    async def setup(self) -> None:
        kept = " ".join(shlex.quote(p.lstrip("/")) for p in self.kept)
        # A log folder holding a kept path or a view stays visible: the judge reads those.
        hidden = tuple(
            h
            for h in HIDDEN
            if not any(
                k == h or k.startswith(h + "/") for k in [*self.kept, *self.views]
            )
        )
        script = f"""{_hide_script(hidden=hidden)}
cd /
present=""
for p in {kept}; do [ -e "/$p" ] && present="$present $p"; done
tar -cf /run/taskmd-judge/kept.tar --files-from /dev/null
[ -n "$present" ] && tar -cf /run/taskmd-judge/kept.tar $present
chmod 600 /run/taskmd-judge/kept.tar
tr '\\0' '\\n' < /proc/1/environ 2>/dev/null | sed -n 's/^PATH=//p' | head -1 > /run/taskmd-judge/path || true
if command -v setpriv >/dev/null 2>&1; then echo setpriv; elif command -v runuser >/dev/null 2>&1 && getent passwd {RUNNER_UID} >/dev/null 2>&1; then echo runuser; else echo root; fi
"""
        result = await self.sandbox.exec(script, user="root", timeout_sec=300)
        from benchflow.sandbox.lockdown import _exec_return_code

        if _exec_return_code(result) != 0:
            raise TaskMdVerifierError(
                f"the judge's runner could not be set up: {(result.stderr or result.stdout or '')[-300:]}"
            )
        self.drop = ((result.stdout or "").strip().splitlines() or ["root"])[-1]
        path = await self.sandbox.exec(
            "cat /run/taskmd-judge/path", user="root", timeout_sec=30
        )
        image_path = (path.stdout or "").strip()
        if image_path:
            keep = [
                entry
                for entry in image_path.split(":")
                if entry
                and not any(
                    entry == k or entry.startswith(k.rstrip("/") + "/")
                    for k in [*self.kept, "/tmp/taskmd-judge"]
                )
            ]
            self.path = ":".join(keep) or self.path
        for view_path, data in self.views.items():
            await self.sandbox.exec(
                f"mkdir -p {shlex.quote(str(PurePosixPath(view_path).parent))} && printf %s {shlex.quote(base64.b64encode(data).decode())} | base64 -d > {shlex.quote(view_path)} && chmod 644 {shlex.quote(view_path)}",
                user="root",
                timeout_sec=60,
            )
        self.ready = True

    async def __call__(
        self, command: str, timeout: int
    ) -> tuple[int | None, bool, bytes]:
        if not self.ready:
            await self.setup()
        kept = " ".join(shlex.quote(p) for p in self.kept)
        if self.drop == "setpriv":
            drop = f"setpriv --reuid={RUNNER_UID} --regid={RUNNER_UID} --clear-groups --no-new-privs"
        elif self.drop == "runuser":
            drop = f"runuser -u $(getent passwd {RUNNER_UID} | cut -d: -f1) --"
        else:
            drop = ""
        encoded = base64.b64encode(command.encode("utf-8")).decode()
        script = f"""
for p in {kept}; do rm -rf "$p"; done
tar -C / -xpf /run/taskmd-judge/kept.tar 2>/dev/null
for p in {kept}; do [ -e "$p" ] && chown -R {RUNNER_UID}:{RUNNER_UID} "$p"; done
rm -rf /tmp/taskmd-judge && mkdir -p /tmp/taskmd-judge/work /tmp/taskmd-judge/home && chown -R {RUNNER_UID}:{RUNNER_UID} /tmp/taskmd-judge
cmd=$(printf %s {shlex.quote(encoded)} | base64 -d)
cd /tmp/taskmd-judge/work
env -i PATH={shlex.quote(self.path)} HOME=/tmp/taskmd-judge/home LANG=C.UTF-8 PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 \
  {drop} timeout -k 5 {int(timeout)} bash --noprofile --norc -c "$cmd" </dev/null >/run/taskmd-judge/out 2>&1
code=$?
for d in /proc/[0-9]*; do
  u=$(awk '/^Uid:/ {{print $2}}' "$d/status" 2>/dev/null)
  [ "$u" = "{RUNNER_UID}" ] && kill -9 "${{d#/proc/}}" 2>/dev/null
done
size=$(wc -c < /run/taskmd-judge/out)
echo "$code $size"
if [ "$size" -le 16384 ]; then base64 -w0 /run/taskmd-judge/out; echo; else head -c 8192 /run/taskmd-judge/out | base64 -w0; echo; tail -c 8192 /run/taskmd-judge/out | base64 -w0; echo; fi
"""
        result = await self.sandbox.exec(
            script, user="root", timeout_sec=int(timeout) + 60
        )
        lines = (result.stdout or "").splitlines()
        try:
            code_text, size_text = lines[0].split()
            code, size = int(code_text), int(size_text)
        except (IndexError, ValueError) as exc:
            raise TaskMdVerifierError(
                f"the judge's runner failed: {(result.stderr or result.stdout or '')[-300:]}"
            ) from exc
        if size <= traj.SHELL_CAP:
            output = base64.b64decode(lines[1] if len(lines) > 1 else "")
        else:
            head = base64.b64decode(lines[1])
            tail = base64.b64decode(lines[2])
            output = head + b"\0" * (size - len(head) - len(tail)) + tail
        return code, code == 124, output


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


async def verify_taskmd(verifier: Any, strategy: Any) -> Any:
    """Grade a materialized draft 2 package (``Verifier.verify()``'s ``taskmd`` strategy)."""
    from benchflow.task.verifier_errors import VerifierResult

    cfg = dict(strategy.config)
    task_dir = Path(verifier._task.paths.task_dir)
    meta = taskmd_metadata(task_dir)
    if meta is None:
        raise TaskMdVerifierError(
            f"{task_dir} is not a materialized task.md package (no metadata.taskmd)"
        )
    config = table(meta.get("config"))
    paths = verifier._rollout_paths
    judge_dir = paths.verifier_dir / JUDGE_DIR
    sandbox = verifier._sandbox
    offline_note = None
    if cfg.get("offline"):
        offline_note = await take_offline(sandbox)
    workdir = cfg.get("workdir") if isinstance(cfg.get("workdir"), str) else None
    grading_kind = cfg.get("grading")
    has_script = isinstance(cfg.get("command"), str)

    script_timeout = cfg.get("script_timeout")
    script_timeout = (
        float(script_timeout) if isinstance(script_timeout, (int, float)) else None
    )
    if grading_kind != "rubric":
        if not has_script:
            raise TaskMdVerifierError("the package has no test.sh and no rubric")
        return await verifier._verify_test_script(
            strategy=None,
            cwd=workdir,
            script_timeout_sec=script_timeout,
            parse_rewards=True,
        )

    pkg = judge_package_dir(task_dir)
    shared = shared_rubrics(task_dir, meta)
    document = ref.parse(pkg)
    rubric = document.rubric or {}
    criteria = jp.merged_criteria(rubric, shared)
    model_roles = {c.get("judge") for c in criteria} & {"llm", "agent"}
    fs_root = judge_dir / "fs"
    if judge_dir.exists():
        shutil.rmtree(judge_dir)
    refused: dict[str, str] = {}
    kept = kept_paths(config)
    if model_roles:
        fs_root.mkdir(parents=True)
        refused = await copy_kept(sandbox, kept, fs_root)

    return_code = None
    if has_script:
        await verifier._verify_test_script(
            strategy=None,
            cwd=workdir,
            script_timeout_sec=script_timeout,
            parse_rewards=False,
        )
        return_code = getattr(verifier, "test_return_code", None)
    for name in ("reward.txt", "reward.json"):
        path = paths.verifier_dir / name
        if path.exists():
            path.rename(paths.verifier_dir / f"script-{name}")
    ctrf = paths.verifier_dir / "ctrf.json"
    report = grading.read_ctrf(ctrf.read_bytes() if ctrf.is_file() else None)

    values: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    gate_failed = False
    for c in criteria:
        if c.get("judge") != "test":
            continue
        matched = grading.matching_tests(str(c.get("check", "")), report.tests)
        passed = bool(matched) and all(t.get("status") == "passed" for t in matched)
        values[c["id"]] = passed
        if c.get("gate") and not passed:
            gate_failed = True
        rationale = (
            "no test in ctrf.json matches this check"
            if not matched
            else f"all {len(matched)} matching tests passed"
            if passed
            else "not passed: "
            + ", ".join(
                f"{t['name']} ({t['status']})"
                for t in matched
                if t.get("status") != "passed"
            )
        )
        records[c["id"]] = {
            "check": str(c.get("check")),
            "verdict": "pass" if passed else "fail",
            "judge": {"role": "test", "tool": report.tool}
            | (
                {"rubric_version": rubric["version"]}
                if isinstance(rubric.get("version"), str)
                else {}
            ),
            "rationale": rationale[:300],
        }

    lazy = True
    judges_table = table(table(config.get("verifier")).get("judges"))
    if judges_table.get("lazy") is False:
        lazy = False
    usage_total: dict[str, Any] = {}
    setup_hash = None
    if model_roles:
        if (paths.verifier_dir / "judge").is_dir():
            raise TaskMdVerifierError(
                "the verifier wrote files for the judges (/logs/verifier/judge/), which BenchFlow does not serve them yet"
            )
        skip = set()
        if lazy and gate_failed:
            skip = {
                c["id"]
                for c in criteria
                if c.get("judge") in ("llm", "agent")
                and not float(c.get("points", 0) or 0) < 0
            }
        try:
            judged, setup_hash, usage_total = await _run_judges(
                verifier=verifier,
                meta=meta,
                config=config,
                document=document,
                pkg=pkg,
                shared=shared,
                criteria=criteria,
                report=report,
                fs_root=fs_root,
                kept=kept,
                refused=refused,
                skip=skip,
                judge_dir=judge_dir,
                separate=cfg.get("isolation") == "separate",
                sandbox=sandbox,
            )
        finally:
            if "agent" in model_roles:
                await restore_hidden(sandbox)
        for ident, record in judged.items():
            records[ident] = record
            values[ident] = record.pop("_value")

    missing_ids = [c["id"] for c in criteria if c["id"] not in values]
    if missing_ids:
        raise TaskMdVerifierError(f"no verdict for {', '.join(missing_ids)}")
    result = grading.score(rubric, criteria, values)
    reward = grading.headline(rubric, result)
    verdicts = []
    summary: dict[str, int] = {}
    for c in criteria:
        record = records[c["id"]]
        entry = {"id": c["id"], **record}
        if c.get("gate"):
            entry["gate"] = True
        entry["score"] = grading.verdict_points(
            c, values[c["id"]] if record["verdict"] != "skip" else None
        )
        verdicts.append(entry)
        summary[entry["verdict"]] = summary.get(entry["verdict"], 0) + 1
    review: dict[str, Any] = {"$schema": grading.REVIEW_SCHEMA, "record": "verifier"}
    if isinstance(rubric.get("version"), str):
        review["rubric_version"] = rubric["version"]
    if model_roles:
        review["submission_tree"] = jp.submission_tree(fs_root)
    review["summary"] = summary
    review["verdicts"] = verdicts
    review["x-benchflow"] = {
        "format": "task.md draft 2",
        "reference_commit": (meta.get("reference") or {}).get("commit"),
        "script_exit": return_code,
        "tests": {"count": len(report.tests), "problem": report.problem},
        "raw": float(result.raw),
        "earned": float(result.earned),
        "maximum": float(result.maximum),
        "failed_gates": result.failed,
        "offline": offline_note,
        "refused_outputs": refused,
        "judge_setup_sha256": setup_hash,
        "judge_usage": usage_total,
        "not_checked": ["[runs]", "[integrity.controls]", "rubric validation"],
    }
    _write_json(paths.verifier_dir / "review.json", review)
    rewards = {
        "reward": reward,
        "strict": 1.0 if result.strict else 0.0,
        "partial": float(result.partial),
    }
    _write_json(paths.reward_json_path, rewards)
    paths.reward_text_path.write_text(f"{reward}\n")
    return VerifierResult(rewards=rewards)


async def _run_judges(
    *,
    verifier: Any,
    meta: dict[str, Any],
    config: dict[str, Any],
    document: Any,
    pkg: Path,
    shared: dict[str, str],
    criteria: list[dict[str, Any]],
    report: grading.TestReport,
    fs_root: Path,
    kept: list[str],
    refused: dict[str, str],
    skip: set[str],
    judge_dir: Path,
    separate: bool,
    sandbox: Any,
) -> tuple[dict[str, dict[str, Any]], str | None, dict[str, Any]]:
    """Run every model-judged unit's samples and combine them per criterion."""
    credentials = judging.credentials_from_env()
    if credentials is None:
        raise TaskMdVerifierError(
            "the rubric has model-judged criteria, and no judge credentials are set "
            "(ANTHROPIC_API_KEY, or CLAUDE_CODE_OAUTH_TOKEN)"
        )
    paths = verifier._rollout_paths
    # /judge/ files: the instruction, the trajectory view, the tests.
    judge_root = fs_root / "judge"
    judge_root.mkdir(parents=True, exist_ok=True)
    instruction = verifier._task.instruction or ""
    for turn in meta.get("turns") or []:
        instruction = instruction.rstrip("\n") + "\n\n" + str(turn.get("prompt", ""))
    instruction_bytes = (instruction.rstrip("\n") + "\n").encode("utf-8")
    (judge_root / "instruction.md").write_bytes(instruction_bytes)
    events = _trial_events(paths)
    oracle_output = _oracle_output(paths)
    record = traj.solver_record(events, oracle_output)
    (judge_root / "trajectory.jsonl").write_text(jp.trajectory_view(record))
    (judge_root / "trajectory-reasoning.jsonl").write_text(
        jp.trajectory_view(record, reasoning=True)
    )
    (judge_root / "tests.json").write_text(grading.tests_view(report))
    _write_json(judge_dir / "trajectory-1.json", record)
    scripted = traj.is_scripted(events)
    mounts = (str(meta.get("oracle_mount") or "/oracle"),) if scripted else ()

    workdir = (
        (config.get("sandbox") or {}).get("workdir")
        if isinstance(config.get("sandbox"), dict)
        else None
    )
    files = judging.JudgeFiles(
        root=fs_root, kept=kept, separate_verifier=separate, scripted_mounts=mounts
    )
    tests_by_id = {
        t["id"]: t for t in json.loads((judge_root / "tests.json").read_text())["tests"]
    }
    by_id = {c["id"]: c for c in criteria}
    sessions = judging.compile_sessions(pkg, shared, fs_root)
    seat_visible = bool(mounts) and _seat_visible(files, mounts)
    # Criteria decided without a session (docs/runtime/judging.md, "Missing evidence").
    decided = {
        c["id"]: found
        for c in criteria
        if c.get("judge") in ("llm", "agent")
        and c["id"] not in skip
        and (found := _evidence_missing(c, fs_root, workdir, refused)) is not None
        and found[1] is not None
    }
    used_roles = sorted({c.get("judge") for c in criteria} & {"llm", "agent"})
    setup: dict[str, Any] | None = None
    samples: dict[str, list[dict[str, Any]]] = {}
    usage_total = {
        "sessions": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_calls": 0,
        "seconds": 0.0,
    }
    runner: SandboxRunner | None = None
    submission = jp.submission_tree(fs_root)
    judged_meta: dict[str, dict[str, Any]] = {}
    for session in sessions:
        role = session["role"]
        ids = [i for i in session["criteria"] if i not in skip and i not in decided]
        if not ids:
            continue
        settings, _ = ref.resolve_judge(document.config, role, None, pkg)
        model = judging.model_for(role, settings)
        if model is None:
            raise TaskMdVerifierError(
                f"[verifier.judges.{role}] names no model: set one, or {judging.MODEL_ENV}"
            )
        if not model.startswith(("claude-", "anthropic/")):
            raise TaskMdVerifierError(
                f"judge model {model!r}: BenchFlow runs judge-loop@1 over the Anthropic Messages API only"
            )
        if setup is None:
            models = {}
            for used in used_roles:
                chosen_model = judging.model_for(
                    used, ref.resolve_judge(document.config, used, None, pkg)[0]
                )
                if chosen_model is not None:
                    models[used] = chosen_model
            setup = jp.judge_setup(
                pkg, shared=shared, models=models, config=document.config
            )
        seconds = int(-(-(ref.duration_s(settings.get("timeout")) or 120) // 1))
        budget = table(settings.get("budget"))
        tokens = (
            ref.count_value(budget.get("tokens"))
            if budget.get("tokens") is not None
            else None
        )
        budget_text = ref.session_budget(document, pkg, session) or f"{seconds} seconds"
        calls = None
        if role == "agent":
            for part in budget_text.split(", "):
                if part.endswith(" tool calls"):
                    calls = int(part.split()[0])
        brief = (
            jp.normalize_brief((pkg / session["brief"]).read_bytes())
            if session.get("brief")
            else None
        )
        tools = (
            ("read", "run", "submit_review") if role == "agent" else ("submit_review",)
        )
        spec = judging.SessionSpec(
            role=role,
            unit=session["unit"],
            model=model,
            assignment=session["assignment"],
            prompt_masked=session["prompt"],
            brief=brief,
            budget_text=budget_text,
            seconds=seconds,
            tokens=tokens,
            tool_calls=calls,
            tools=tools,
        )
        if jp.prompt_text(brief, session["assignment"]) != session["prompt"]:
            raise TaskMdVerifierError(
                "judge-prompt@1 did not compile to the reference compiler's bytes"
            )
        if role == "agent" and runner is None:
            if not separate:
                raise TaskMdVerifierError(
                    'an agent judge needs [verifier] isolation = "separate"'
                )
            views = _views(settings, judge_root, record)
            files.views = views
            runner = SandboxRunner(sandbox=sandbox, kept=kept, views=views)
            await runner.setup()
        evidence = (
            judging.evidence_with_fence(session["assignment"], fs_root)
            if role == "llm"
            else None
        )
        n_samples = int(settings.get("samples", 1))
        client = judging.MessagesClient(credentials)
        unit_meta = {
            "role": role,
            "unit": session["unit"],
            "model": model,
            "prompt_sha256": session["prompt_sha256"],
            "evidence_sha256": session.get("evidence_sha256"),
            "settings": settings,
        }
        for sample in range(1, n_samples + 1):
            accepted = None
            for attempt in (1, 2):
                seed = jp.judge_seed(submission, role, session["unit"], sample, attempt)
                judging.SESSIONS.acquire()
                result = await judging.run_session(
                    spec,
                    call=client,
                    credentials=credentials,
                    files=files,
                    evidence_text=evidence,
                    runner=runner,
                )
                name = (
                    f"{role}-{session['unit'].replace(':', '-')}-s{sample}-a{attempt}"
                )
                _write_json(
                    judge_dir / f"{name}.json",
                    {
                        "role": role,
                        "unit": session["unit"],
                        "sample": sample,
                        "attempt": attempt,
                        "seed": seed,
                        "end": result.end,
                        "usage": result.usage,
                        "model": model,
                        "harness": f"{judging.HARNESS}@{judging.HARNESS_VERSION}",
                        "system_note": credentials.system_note,
                        "accepted": result.accepted,
                        "transcript": result.transcript,
                    },
                )
                usage_total["sessions"] += 1
                for key in ("prompt_tokens", "completion_tokens", "tool_calls"):
                    usage_total[key] += int(result.usage.get(key, 0))
                usage_total["seconds"] += float(result.usage.get("seconds", 0))
                if result.accepted is not None:
                    accepted = (result, seed, name)
                    break
            for ident in ids:
                crit = by_id[ident]
                if accepted is None:
                    samples.setdefault(ident, []).append(
                        {"verdict": "error", "score": None, "trajectory": name}
                    )
                    continue
                res, seed, name = accepted
                submitted = table(res.accepted)
                raw = next(
                    v for v in listed(submitted.get("verdicts")) if v["id"] == ident
                )
                ctx = judging.CitationContext(
                    files=files,
                    record=record,
                    tests=tests_by_id,
                    instruction=instruction_bytes,
                    workdir=workdir if isinstance(workdir, str) else None,
                    reasoning_served=any(
                        str(e).startswith(jp.RUNTIME_KINDS["trajectory:reasoning"])
                        for criterion in session["assignment"]["criteria"]
                        for e in criterion["evidence"]
                    ),
                )
                cites = [
                    judging.check_citation(c, ctx, res.judge_steps)
                    for c in raw.get("citations", [])
                ]
                verdict = {**raw, "citations": cites}
                verdict, flags = judging.apply_citation_rules(crit, verdict)
                if seat_visible:
                    flags.append("seat-visible")
                entry: dict[str, Any] = {
                    "verdict": verdict["verdict"],
                    "score": grading.verdict_points(
                        crit, judging.verdict_value(crit, verdict)
                    ),
                    "citations": cites,
                    "rationale": str(verdict.get("rationale", ""))[
                        : judging.MAX_RATIONALE
                    ],
                    "trajectory": name,
                }
                if verdict["verdict"] == "level":
                    entry["level"] = str(verdict["level"])
                if verdict["verdict"] == "value":
                    entry["value"] = verdict["value"]
                if flags:
                    entry["flags"] = flags
                entry["_raw"] = verdict
                samples.setdefault(ident, []).append(entry)
        for ident in ids:
            judged_meta[ident] = unit_meta
    setup_hash = jp.sha256_hex(jp.jcs(setup)) if setup is not None else None
    out: dict[str, dict[str, Any]] = {}
    for c in criteria:
        if c.get("judge") not in ("llm", "agent"):
            continue
        ident = c["id"]
        role = c["judge"]
        judge_record: dict[str, Any] = {"role": role}
        if ident in skip:
            out[ident] = {
                "verdict": "skip",
                "judge": judge_record,
                "flags": ["lazy"],
                "rationale": "a gate decided by a test failed, so no judge session ran for this criterion (lazy)",
                "_value": grading.skip_value(c),
            }
            continue
        if ident in decided:
            flag, label, value = decided[ident]
            out[ident] = {
                "verdict": label,
                "judge": judge_record,
                "flags": [flag],
                "rationale": (
                    "an output this criterion names was refused, so it fails without a judge session"
                    if flag == "output-refused"
                    else "the solver saved none of the files this criterion names, so it is decided without a judge session"
                ),
                "_value": value,
            }
            continue
        unit = judged_meta.get(ident, {})
        judge_record |= {
            "model": unit.get("model"),
            "harness": judging.HARNESS,
            "harness_version": judging.HARNESS_VERSION,
            "prompt_format": jp.FORMAT,
            "prompt_sha256": unit.get("prompt_sha256"),
            "provider": "anthropic",
        }
        if setup_hash:
            judge_record["judge_setup_sha256"] = setup_hash
        if unit.get("evidence_sha256"):
            judge_record["evidence_sha256"] = unit["evidence_sha256"]
        if isinstance((document.rubric or {}).get("version"), str):
            judge_record["rubric_version"] = document.rubric["version"]
        entries = samples.get(ident, [])
        valid = [e for e in entries if e["verdict"] != "error"]
        settings = unit.get("settings") or {}
        min_samples = int(settings.get("min_samples", settings.get("samples", 1)))
        public = [{k: v for k, v in e.items() if k != "_raw"} for e in entries]
        if len(valid) < min_samples:
            raise TaskMdVerifierError(
                f"criterion {ident}: {len(valid)} valid samples of {len(entries)}, and min_samples is {min_samples}: "
                "the judge failed, so the trial is not scored (BenchFlow does not judge a reference to attribute it)"
            )
        missing_file = _evidence_missing(c, fs_root, workdir, refused)
        numbers = [judging.sample_number(c, e["_raw"]) for e in valid]
        aggregate = str(settings.get("aggregate", "median"))
        combined = judging.combine(numbers, aggregate)
        chosen = next(e for e, n in zip(valid, numbers, strict=True) if n == combined)
        verdict = chosen["_raw"]
        value = judging.verdict_value(c, verdict)
        record_out: dict[str, Any] = {
            "verdict": verdict["verdict"],
            "judge": judge_record,
            "samples": public,
            "spread": float(max(numbers) - min(numbers)) if numbers else 0.0,
            "citations": chosen["citations"],
            "rationale": chosen["rationale"],
        }
        if verdict["verdict"] == "level":
            record_out["level"] = str(verdict["level"])
        if verdict["verdict"] == "value":
            record_out["value"] = verdict["value"]
        flags = list(chosen.get("flags", []))
        if missing_file is not None:
            flags.append(missing_file[0])  # judged on what remains (a bad outcome)
        if flags:
            record_out["flags"] = flags
        if aggregate == "mean" and numbers:
            record_out["x-mean"] = float(sum(numbers) / len(numbers))
        record_out["_value"] = value
        out[ident] = record_out
    return out, setup_hash, usage_total


def _evidence_missing(
    criterion: dict[str, Any], fs_root: Path, workdir: Any, refused: dict[str, str]
) -> tuple[str, str | None, Any] | None:
    """docs/runtime/judging.md, "Missing evidence": (flag, verdict, the score() input), or None.

    A criterion that names a refused output fails without a session, whatever
    its outcome. One whose every item is a file the solver never saved fails
    without a session when its outcome is good; a bad one is judged on what
    remains (None as its input), or skipped with ``missing = "no-penalty"``.
    """
    items = [str(i).split("#", 1)[0] for i in criterion.get("evidence") or []]
    files = [
        i for i in items if i not in jp.RUNTIME_KINDS and not i.startswith("diff:")
    ]
    paths = []
    for item in files:
        try:
            paths.append(
                jp.normalize_path(item, workdir if isinstance(workdir, str) else None)
            )
        except jp.JudgePromptError:
            return None
    if any(p == r or p.startswith(r.rstrip("/") + "/") for p in paths for r in refused):
        return ("output-refused", "fail", False)
    if not files or len(files) != len(items):
        return None
    if all(not (fs_root / p.lstrip("/")).exists() for p in paths):
        if criterion.get("outcome", "good") == "good":
            return ("evidence-missing", "fail", False)
        if criterion.get("missing") == "no-penalty":
            return ("evidence-missing", "skip", grading.skip_value(criterion))
        return ("evidence-missing", None, None)
    return None


def _seat_visible(files: judging.JudgeFiles, mounts: tuple[str, ...]) -> bool:
    """Whether anything a session could read still names the scripted seat's mount."""
    for path in (files.root / "judge").rglob("*"):
        if path.is_file():
            with contextlib.suppress(OSError):
                data = path.read_text(encoding="utf-8", errors="replace")
                if any(m in data for m in mounts) or "oracle/solve.sh" in data:
                    return True
    for kept in files.kept:
        base = files.host(kept)
        for path in (
            [base] if base.is_file() else base.rglob("*") if base.is_dir() else []
        ):
            if path.is_file() and path.stat().st_size < 4_000_000:
                with contextlib.suppress(OSError):
                    data = path.read_text(encoding="utf-8", errors="replace")
                    if any(m in data for m in mounts):
                        return True
    return False


def _views(
    settings: dict[str, Any], judge_root: Path, record: dict[str, Any]
) -> dict[str, bytes]:
    """The agent role's views: the trajectory view at each named path."""
    views: dict[str, bytes] = {}
    for view in settings.get("views") or []:
        if (
            isinstance(view, dict)
            and view.get("format") == "trajectory-1"
            and isinstance(view.get("path"), str)
        ):
            views[jp.normalize_path(view["path"])] = jp.trajectory_view(record).encode(
                "utf-8"
            )
    return views


def _trial_events(paths: Any) -> list[dict[str, Any]]:
    for folder in (paths.agent_dir, paths.rollout_dir.parent / "agent"):
        path = folder / "acp_trajectory.jsonl"
        if path.is_file():
            return traj.load_events(path.read_bytes())
    return []


def _oracle_output(paths: Any) -> bytes | None:
    for folder in (paths.agent_dir, paths.rollout_dir.parent / "agent"):
        path = folder / "oracle.txt"
        if path.is_file():
            return path.read_bytes()
    return None


__all__ = ["TaskMdVerifierError", "verify_taskmd"]
