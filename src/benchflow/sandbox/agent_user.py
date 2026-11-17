"""The user a task's own harness runs the agent as, beside BenchFlow's.

A task's own harness runs the agent as the task's ``[agent].user``, else as
main's image ``USER`` (for Compose, main's ``user:`` first): root on nearly
every public task. BenchFlow runs an agent as ``--sandbox-user`` (``agent``
unless the run says otherwise) and the oracle as ``[agent].user`` or root.
Where the two differ, a task that needs root scores lower under BenchFlow
(F2b-USERQ, 56 tasks: 43% of the reference solutions pass as ``agent``, 86%
under the tasks' own harness).

Nothing here changes who runs the agent. The run says so in one line before
the agent starts, once per task, and every trial's result.json records both
users and the ``--sandbox-user`` that would match (:data:`RECORD_KEY`), as the
run card does per task. The task's user is read on the host only: task.toml,
the Compose file, the Dockerfile's ``USER`` and ``docker image inspect``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from benchflow._utils.config import normalize_sandbox_user
from benchflow.sandbox.pid1_ptrace import _local_image_user, dockerfile_user

#: result.json's and the run card's key.
RECORD_KEY = "agent_user"
ROOT = "root"
#: Where the task's user came from, as the record names it.
FROM_TASK = "task.toml [agent].user"
FROM_COMPOSE = "compose main user"
FROM_IMAGE = "image USER"
_NAME = re.compile(r"^[a-z_][a-z0-9_-]*$")
#: Agents that refuse a root sandbox user, and why: no --sandbox-user then
#: matches a root task. agents/scripts.py refuses every script: agent;
#: agents/muse_config.py, agents/codex_connector_policy.py (its default
#: policy for a task run) and agents/claude_connector_policy.py.
_ROOT_REFUSED = {
    "muse-code": "muse-code needs a non-root sandbox user",
    "codex-acp": (
        "codex-acp's default Codex Apps policy needs a non-root sandbox user"
    ),
    "claude-agent-acp": (
        "claude-agent-acp's connector restrictions need a non-root sandbox user"
    ),
}

#: Tasks whose line this process has printed or covered, by task folder name,
#: with the user it named: once per task (the run start's, else the trial's).
_reported: dict[str, str | None] = {}
#: Tasks whose no-working-tree line this process has printed.
_tree_reported: set[str] = set()

#: agent_user's ``working_tree``: whether the agent may write the task's
#: working tree. BenchFlow hands the agent user its workspace: the task's
#: ``[environment].workdir``, else the image's working folder; an image that
#: works in ``/`` gets ``/root`` instead (rollout/_setup.py
#: ``_resolve_agent_cwd``), so the task's own files (FeatureBench keeps its
#: repository in /testbed) are never the agent's to write.
TREE_ROOT = "root"
TREE_HANDED = "handed"
TREE_NOT_HANDED = "not handed"


def _user(value: Any) -> str | None:
    """A ``USER``/``user:`` value as a user name: an empty one, ``root`` and
    ``0`` (with or without a group) are root."""
    if value is None:
        return None
    name = str(value).strip().split(":", 1)[0].strip()
    return ROOT if name in ("", ROOT, "0") else name


def _declared(config: Any) -> Any:
    return getattr(getattr(config, "agent", None), "user", None)


def _compose_main(environment_dir: Path) -> dict[str, Any]:
    compose = environment_dir / "docker-compose.yaml"
    if not compose.is_file():
        return {}
    try:
        spec = yaml.safe_load(compose.read_text()) or {}
        main = (spec.get("services") or {}).get("main") or {}
    except (OSError, ValueError, yaml.YAMLError, AttributeError):
        return {}
    return main if isinstance(main, dict) else {}


def _static_image_user(environment_dir: Path, config: Any) -> str | None:
    """Main's image ``USER`` from the task's files: the Dockerfile's (its
    stages and base image), else a prebuilt image's in this host's image
    store; None when that needs an image this host does not hold."""
    dockerfile = environment_dir / "Dockerfile"
    build = _compose_main(environment_dir).get("build")
    if isinstance(build, dict):
        context = str(build.get("context", "."))
        folder = environment_dir if "${" in context else environment_dir / context
        dockerfile = folder / str(build.get("dockerfile", "Dockerfile"))
    if dockerfile.is_file():
        try:
            return dockerfile_user(dockerfile.read_text())[0]
        except OSError:
            return None
    image = getattr(getattr(config, "sandbox", None), "docker_image", None)
    return _local_image_user(str(image)) if image else None


def task_user(
    task_dir: Path, config: Any, image_user: str | None = None
) -> tuple[str | None, str | None]:
    """(The user the task's own harness runs the agent as, where that came
    from): ``[agent].user``, else main's Compose ``user:``, else main's image
    ``USER`` (*image_user* when the started sandbox's image was read, else
    from the task's files); (None, None) when it cannot be told here."""
    declared = _declared(config)
    if declared is not None and str(declared).strip():
        return _user(declared), FROM_TASK
    environment_dir = Path(task_dir) / "environment"
    compose_user = _compose_main(environment_dir).get("user")
    if compose_user is not None:
        return _user(compose_user), FROM_COMPOSE
    if image_user is None:
        image_user = _static_image_user(environment_dir, config)
    if image_user is None:
        return None, None
    return _user(image_user), FROM_IMAGE


def bench_user(agent: str | None, sandbox_user: str | None, config: Any) -> str:
    """The user BenchFlow runs *agent* as: the oracle as a named
    ``[agent].user`` or root (it takes no ``--sandbox-user``), any other agent
    as the sandbox user, root when there is none."""
    if agent == "oracle":
        declared = _declared(config)
        named = isinstance(declared, str) and declared.strip()
        return (_user(declared) if named else None) or ROOT
    return _user(normalize_sandbox_user(sandbox_user)) or ROOT


def root_refusal(agent: str | None) -> str | None:
    """Why *agent* refuses a root sandbox user (:data:`_ROOT_REFUSED`), None
    when it runs as root."""
    from benchflow.agents.registry import is_operator_script

    if is_operator_script(agent):
        return "a script: agent needs a non-root sandbox user"
    return _ROOT_REFUSED.get(agent or "")


def matching_flag(agent: str | None, user: str | None) -> str | None:
    """The ``--sandbox-user`` that runs *agent* as *user*; None when none
    does: the oracle takes none, an agent that refuses root
    (:func:`root_refusal`) has none for a root task, and the flag takes a
    name, not a uid."""
    if user is None or agent == "oracle":
        return None
    if user == ROOT and root_refusal(agent):
        return None
    return _flag_for(user)


def _flag_for(user: str | None) -> str | None:
    """``--sandbox-user`` naming *user* (root is ``null``); None for a uid."""
    if user is None:
        return None
    if user == ROOT:
        return "--sandbox-user null"
    return f"--sandbox-user {user}" if _NAME.match(user) else None


def record(
    task_dir: Path,
    config: Any,
    agent: str | None,
    sandbox_user: str | None,
    image_user: str | None = None,
    configured_workdir: str | None = None,
    default_pwd: str | None = None,
) -> dict[str, Any]:
    """result.json's ``agent_user``: the task's harness user and its source,
    the user BenchFlow runs the agent as, whether they are the same (None
    when the task's user is unknown, or a uid beside a name) and the flag
    that matches."""
    user, source = task_user(task_dir, config, image_user)
    ran = bench_user(agent, sandbox_user, config)
    if user is None:
        same = None
    elif user == ran:
        same = True
    elif user.isdigit() != ran.isdigit() and ROOT not in (user, ran):
        same = None
    else:
        same = False
    return {
        "task_harness": user,
        "task_harness_source": source,
        "ran_as": ran,
        # The image's working folder as the start probe saw it (None when it
        # was not read), and whether the agent may write the task's tree.
        "image_workdir": default_pwd,
        "working_tree": working_tree(agent, ran, configured_workdir, default_pwd),
        "same": same,
        # A run that already matches names the flag it ran with, even for an
        # agent whose own defaults refuse root (codex-acp under
        # --codex-apps-policy inherit).
        "matching_flag": (
            _flag_for(user)
            if same and agent != "oracle"
            else matching_flag(agent, user)
        ),
    }


def _shown(user: str) -> str:
    return user if user == ROOT else f"`{user}`"


def line(
    entry: dict[str, Any], agent: str | None, task: str | None = None
) -> str | None:
    """The one line for a task whose harness user differs from BenchFlow's;
    None when they are the same or the task's user is unknown."""
    if entry.get("same") is not False:
        return None
    user, ran, flag = entry["task_harness"], entry["ran_as"], entry["matching_flag"]
    who = f"{task}: its own harness" if task else "This task's own harness"
    runs = (
        "BenchFlow's oracle runs solve.sh as"
        if agent == "oracle"
        else "BenchFlow runs it as"
    )
    text = f"{who} runs the agent as {_shown(user)}; {runs} {_shown(ran)}."
    if flag:
        return f"{text} To match it, pass `{flag}`."
    if agent == "oracle":
        return f"{text} No --sandbox-user changes the oracle's user."
    why = root_refusal(agent) if user == ROOT else None
    if why:
        return f"{text} No --sandbox-user matches it: {why}."
    return f"{text} No --sandbox-user names that user."


def working_tree(
    agent: str | None,
    ran: str | None,
    configured_workdir: str | None,
    default_pwd: str | None,
) -> str | None:
    """:data:`TREE_ROOT` when the agent runs as root, :data:`TREE_HANDED`
    when BenchFlow hands its user a workspace of the task's
    (``[environment].workdir`` or a working folder other than ``/``),
    :data:`TREE_NOT_HANDED` when the image works in ``/`` and the task names
    no workdir; None when that is not known here (the oracle is handed none:
    it runs as the task's own user)."""
    if ran is None:
        return None
    if ran == ROOT:
        return TREE_ROOT
    if agent == "oracle":
        return None
    if configured_workdir:
        return TREE_HANDED
    if default_pwd is None:
        return None
    return TREE_NOT_HANDED if default_pwd.strip() == "/" else TREE_HANDED


def tree_line(task: str, entry: dict[str, Any], agent: str | None) -> str | None:
    """The trial's line when the agent user gets no working tree, once per
    task, before the agent starts: its 0s there are not the model's."""
    if entry.get("working_tree") != TREE_NOT_HANDED or task in _tree_reported:
        return None
    _tree_reported.add(task)
    user = _shown(entry["ran_as"])
    text = (
        f"{task}: BenchFlow hands {user} no working tree for this task. Its "
        "image works in / and the task names no [environment].workdir, so "
        f"{user} starts in /root and cannot write the task's own files (such "
        "as a repository under /testbed) unless the image lets every user "
        f"write them: read its 0s under {user} as that, not as the model's."
    )
    if root_refusal(agent) is None:
        return (
            f"{text} To let the agent write them, pass `--sandbox-user null`, "
            "or set [environment].workdir to the task's working folder."
        )
    return f"{text} Set [environment].workdir to the task's working folder to hand it over."


def start_lines(entries: dict[str, dict[str, Any]], agent: str | None) -> list[str]:
    """The run start's line for *entries* (task folder name to record): one
    task's own, or for a batch one line with the count (the records name the
    tasks). A task whose user is not known yet is reported by its trial."""
    known = {
        name: entry
        for name, entry in entries.items()
        if entry.get("task_harness") is not None and name not in _reported
    }
    for name, entry in known.items():
        _reported[name] = entry["task_harness"]
    if len(entries) == 1:
        name, entry = next(iter(known.items()), (None, {}))
        text = line(entry, agent) if name is not None else None
        return [text] if text else []
    differ = [entry for entry in known.values() if entry.get("same") is False]
    if not differ:
        return []
    ran = sorted({entry["ran_as"] for entry in differ})
    flags = {entry["matching_flag"] for entry in differ}
    text = (
        f"{len(differ)} of {len(entries)} tasks' own harnesses run the agent as "
        f"another user than BenchFlow does ({', '.join(_shown(u) for u in ran)}); "
        f"each trial's result.json names both under {RECORD_KEY}."
    )
    if len(flags) == 1 and None not in flags:
        text += f" To match them, pass `{next(iter(flags))}`."
    else:
        text += " Each record names the flag that matches, where one does."
    return [text]


def trial_line(task: str, entry: dict[str, Any], agent: str | None) -> str | None:
    """The trial's line, once the task's user is known and before the agent
    starts: only when the run start did not report this task with that user."""
    user = entry.get("task_harness")
    if user is None or _reported.get(task, "") == user:
        return None
    _reported[task] = user
    return line(entry, agent, task)
