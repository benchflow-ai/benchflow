"""Read task.md draft-1 files into the runtime's v0.6 document model.

task.md draft 1 has no YAML frontmatter. The file opens with the instruction,
which is exactly what the agent receives, followed by typed fenced blocks:
```` ```toml task ```` (the config; ```` ```yaml task ```` is accepted),
```` ```stage <name> ````, ```` ```role <name> ````, ```` ```user ````, and
```` ```notes ````. A file with no blocks is all instruction. The package builds
the agent's sandbox from ``sandbox/`` rather than ``environment/``.

This module is an adapter, not a second loader. It parses the draft-1 file,
maps the config block to v0.6 frontmatter data, and hands that data to the
unchanged v0.6 pipeline (normalization, ``TaskConfig`` validation, roles,
scenes). Stage, role, and user blocks become v0.6 scene prompts, role prompts,
and the user persona. ``[verifier] mount`` and ``[oracle] mount`` say where
``verifier/`` and ``oracle/`` appear in the sandbox, and a task.md rubric in
``verifier/rubric.json`` is kept for the verifier, which grades it from the
test script's CTRF report (``benchflow.task.verifier_rubric``).

Credit: the block rules are ported from the task.md draft-1 specification
(``docs/document.md``, ``docs/package.md``) and its reference parser
(``tools/taskmd.py``). The config mapping is ported from ``config_to_v06`` in
the spec's ``tools/convert.py``, including its table renames
(``V06_TABLE_NAMES``, ``V06_VERIFIER_NAMES``), as of task-md commit a497e86.

The adapter fails closed. Every draft-1 setting is handled in one of three ways:

- mapped: it has a v0.6 key with the same meaning. The runtime's launch gate
  (``runtime_capabilities``) still refuses v0.6 keys it cannot execute, such as
  ``steps`` or a separate verifier sandbox.
- ignored: it changes nothing the runtime does. This covers descriptive fields
  (title, credits, provenance, a canary, notes blocks) and values that restate
  the runtime's own behavior.
- unsupported: everything else. The document still loads so it can be
  inspected, but ``validate_task_runtime_support`` reports each setting and the
  runtime refuses to launch the task.

A malformed file (text after the first block, an unknown key, a bad duration,
and so on) raises ``TaskDocumentParseError``, as a malformed v0.6 file does.
"""

from __future__ import annotations

import copy
import json
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from benchflow.task._document_normalize import TaskDocumentParseError
from benchflow.task.verifier_rubric import rubric_gaps

DRAFT1_SANDBOX_DIRNAME = "sandbox"
# Where verifier/ and oracle/ appear in the sandbox. The defaults are draft 1's;
# a Harbor import records /tests and /solution. These four are the paths the
# runtime locks away from the agent before it starts (sandbox.lockdown), and
# pytest's conftest cutoff follows the verifier path, so no other path is safe.
DEFAULT_VERIFIER_MOUNT = "/verifier"
DEFAULT_ORACLE_MOUNT = "/oracle"
SUPPORTED_VERIFIER_MOUNTS = (DEFAULT_VERIFIER_MOUNT, "/tests")
SUPPORTED_ORACLE_MOUNTS = (DEFAULT_ORACLE_MOUNT, "/solution")

# File syntax (tools/taskmd.py) -----------------------------------------------

_FENCE = re.compile(
    r"^(?P<indent>[ ]{0,3})(?P<marker>`{3,}|~{3,})[ \t]*(?P<info>[^`]*?)[ \t]*$"
)
_CANARY_LINE = re.compile(r"^<!--(?P<text>[^>]*canary[^>]*)-->[ \t]*$", re.I)
_PLAIN_BLOCKS = {"notes", "user"}
_NAMED_BLOCKS = {"stage", "role"}
_CONFIG_LANGS = {"toml", "yaml", "yml"}
# Fences that were blocks in earlier drafts and now live in verifier/.
_MOVED_BLOCKS = {
    "rubric": "verifier/rubric.json",
    "behaviors": "verifier/behaviors.json",
}

# Config vocabulary (tools/taskmd.py) -----------------------------------------

_IDENTITY = {"name", "title", "version", "description", "authors", "keywords"}
_TABLES = {
    "about", "sandbox", "agent", "verifier", "oracle", "world", "tiers",
    "conventions", "stages", "roles", "user", "interaction", "variants", "matrix",
    "family", "training", "trajectory", "preference", "integrity", "runs",
    "credits", "provenance", "import",
}  # fmt: skip
_ARRAYS = {"credits"}
_CLOSED_KEYS = {
    "sandbox": {
        "image", "os", "cpus", "memory", "disk", "gpus", "gpu_types", "tpu",
        "network", "workdir", "env", "skills", "mcp", "ready", "build_timeout",
        "outputs", "mounts", "boundary",
    },
    "agent": {"timeout", "on_timeout", "budget", "user", "network", "system_prompt_append"},
    "verifier": {
        "timeout", "user", "env", "network", "isolation", "sandbox", "snapshot",
        "combine_stages", "unreached_stages", "judges", "human", "mount",
    },
    "oracle": {"env", "mount"},
    "runs": {"trials", "pinned", "errored", "retries", "disclose", "trajectory"},
}  # fmt: skip
_ENUMS = (
    ("verifier", "isolation", ("shared", "separate")),
    ("verifier", "combine_stages", ("mean", "final")),
    ("verifier", "unreached_stages", ("zero", "exclude")),
    ("agent", "on_timeout", ("grade", "grade-flagged", "fail")),
    ("sandbox", "boundary", ("container", "gvisor", "microvm", "vm")),
    ("runs", "errored", ("zero", "retry")),
    ("runs", "trajectory", ("required", "optional")),
)
_HARBOR_NAMES = {  # Harbor task.toml spellings -> task.md's
    "schema_version": "nothing (task.md has no schema version key)",
    "task": "identity keys at the top of the block (name, version, ...)",
    "metadata": "[about]",
    "environment": "[sandbox]",
    "solution": "[oracle]",
    "steps": "[stages.<name>]",
    "artifacts": "[sandbox] outputs",
    "source": "[provenance] source",
    "multi_step_reward_strategy": "[verifier] combine_stages",
    "docker_image": "image",
    "memory_mb": "memory",
    "storage_mb": "disk",
    "build_timeout_sec": "build_timeout",
    "timeout_sec": "timeout",
    "network_mode": "network",
    "allowed_hosts": "network (a list of hosts)",
    "allow_internet": "network",
    "skills_dir": "skills",
    "healthcheck": "ready",
    "mcp_servers": "mcp",
    "environment_mode": "isolation",
    "collect": "snapshot",
}
_WORLD_KINDS = {"virtual", "simulated", "physical"}
_CREDIT_ROLES = (None, "author", "advisor", "domain-reviewer", "technical-reviewer")
_DURATION_SHAPE = re.compile(
    r"^(?=\d)(?:\d+(?:\.\d+)?h)?(?:\d+(?:\.\d+)?m)?(?:\d+(?:\.\d+)?s)?$"
)
_SIZE_SHAPE = re.compile(r"^\d+(?:\.\d+)? ?(?:MB|GB|TB)$")

# Values (tools/convert.py) ----------------------------------------------------

_DURATION = re.compile(
    r"^(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?$"
)
_SIZE = re.compile(r"^(?P<n>\d+(?:\.\d+)?) ?(?P<u>MB|GB|TB)$")
_AUTHOR = re.compile(r"^(?P<name>[^<>]*?)(?: <(?P<email>[^<>@\s]+@[^<>\s]+)>)?$")
_UNITS = {"MB": 1, "GB": 1024, "TB": 1024 * 1024}  # 1 GB is 1024 MB, as in Docker

# Keys the v0.6 runtime reads from a role and from the simulated user. The
# document model passes both through as mappings, so anything else would be
# dropped without a word.
_V06_ROLE_KEYS = {
    "agent", "model", "reasoning_effort", "env", "timeout_sec",
    "idle_timeout_sec", "skills_dir", "capabilities",
}  # fmt: skip
_V06_USER_KEYS = {"model", "stop_rule", "private_facts"}

_SCHEMA_BASE = "https://task.md/schema/"


@dataclass(frozen=True)
class Draft1Finding:
    """A draft-1 setting the adapter did not map to v0.6, and why."""

    path: str
    reason: str


@dataclass(frozen=True)
class Draft1Document:
    """A draft-1 ``task.md``, parsed and mapped to v0.6 data.

    ``frontmatter`` is the v0.6 frontmatter the config block maps to, before
    v0.6 normalization. ``unsupported`` lists settings the runtime cannot
    honor; the launch gate refuses the task while any remain. ``ignored``
    lists settings that change nothing the runtime does. ``verifier_mount`` and
    ``oracle_mount`` are the declared sandbox paths, or the defaults.
    ``rubric`` is ``verifier/rubric.json`` when it is task.md's (by
    ``$schema``) and the runtime can grade it; it is read only when the
    package is, so the verifier grades the copy that was checked.
    """

    instruction: str
    frontmatter: dict[str, Any]
    role_prompts: dict[str, str]
    scene_prompts: dict[str, str]
    user_persona: str | None
    config: dict[str, Any]
    unsupported: tuple[Draft1Finding, ...] = ()
    ignored: tuple[Draft1Finding, ...] = ()
    canary: str | None = None
    verifier_mount: str = DEFAULT_VERIFIER_MOUNT
    oracle_mount: str = DEFAULT_ORACLE_MOUNT
    rubric: dict[str, Any] | None = None


@dataclass
class _Findings:
    unsupported: list[Draft1Finding] = field(default_factory=list)
    ignored: list[Draft1Finding] = field(default_factory=list)

    def refuse(self, path: str, reason: str) -> None:
        self.unsupported.append(Draft1Finding(path, reason))

    def ignore(self, path: str, reason: str) -> None:
        self.ignored.append(Draft1Finding(path, reason))


@dataclass(frozen=True)
class _Block:
    kind: str  # "config", "stage", "role", "user", "notes"
    arg: str | None
    info: str
    body: str
    line: int  # line of the opening fence


def is_draft1_task_md(text: str) -> bool:
    """Whether ``task.md`` text is a draft-1 document rather than a v0.6 one.

    v0.6 requires YAML frontmatter on the first line, so a file whose first
    non-blank line is not ``---`` was never a loadable v0.6 document. A file
    that reaches ``---`` after blank lines or a byte-order mark, and a blank
    file, keep the v0.6 path and its existing error.
    """

    for line in text.splitlines():
        stripped = line.lstrip("\ufeff").strip()
        if stripped:
            return stripped != "---"
    return False


def is_draft1_task_dir(task_dir: str | Path) -> bool:
    """Whether ``task_dir/task.md`` is a draft-1 document.

    Reads only up to the first non-blank line, so path lookups stay cheap.
    """

    try:
        with (Path(task_dir) / "task.md").open(
            encoding="utf-8", errors="replace"
        ) as handle:
            for line in handle:
                stripped = line.lstrip("\ufeff").strip()
                if stripped:
                    return stripped != "---"
    except OSError:
        return False
    return False


def draft1_mounts(task_dir: str | Path) -> tuple[str, str] | None:
    """The (verifier, oracle) sandbox paths of a draft-1 package, or ``None``.

    ``None`` means ``task_dir`` is not a draft-1 package. A malformed draft-1
    ``task.md`` raises ``TaskDocumentParseError`` rather than falling back to a
    default path.
    """

    if not is_draft1_task_dir(task_dir):
        return None
    text = (Path(task_dir) / "task.md").read_text(encoding="utf-8")
    document = read_draft1_task_md(text)
    return document.verifier_mount, document.oracle_mount


def read_draft1_task_md(
    text: str, *, task_dir: str | Path | None = None
) -> Draft1Document:
    """Parse a draft-1 ``task.md`` and map it to v0.6 data.

    ``task_dir`` enables the package checks: a Harbor ``task.toml`` beside
    ``task.md`` is an error, ``verifier/rubric.json`` and
    ``verifier/behaviors.json`` in task.md's schema are classified, and a
    ``verifier/verifier.md`` strategy is checked against them.
    """

    root = Path(task_dir) if task_dir is not None else None
    text = text.replace("\r\n", "\n").removeprefix("\ufeff")
    errors: list[str] = []
    instruction_lines, blocks = _scan(text.split("\n"), errors)
    # The instruction keeps its bytes; the agent's view drops leading canary
    # comments and blank lines, and blank lines before the first block belong
    # to neither.
    canaries, instruction = _strip_canary("\n".join(instruction_lines).rstrip("\n"))
    if not instruction.strip():
        errors.append(
            "the instruction is empty: task.md must start with the task itself"
        )

    seen: set[tuple[str, str | None]] = set()
    for block in blocks:
        key = (block.kind, block.arg if block.kind in _NAMED_BLOCKS else None)
        if key in seen:
            errors.append(f"line {block.line}: duplicate ```{block.info} block")
        seen.add(key)

    config: dict[str, Any] = {}
    for block in blocks:
        if block.kind == "config":
            config = _parse_config(block, errors)
    if root is not None and (root / "task.toml").exists():
        errors.append(
            "task.toml is Harbor's config file and means something else here; "
            "a draft-1 task.md keeps its config in a ```toml task block"
        )
    if not errors:
        _check_config(config, blocks, errors)
    if errors:
        raise TaskDocumentParseError("task.md draft 1: " + "; ".join(errors))

    findings = _Findings()
    mounts = _mounts(config, findings)
    try:
        frontmatter = _config_to_v06(config, findings)
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        raise TaskDocumentParseError(
            f"task.md draft 1: the config block cannot be mapped: {e}"
        ) from e
    _check_prompt_settings(config, findings)
    if canaries:
        findings.ignore("canary comment", "stripped from the agent's view")
    for block in blocks:
        if block.kind == "notes":
            findings.ignore(
                "```notes", "author and reviewer notes, never shown to an agent"
            )
    rubric = None
    if root is not None:
        rubric = _check_judgment_files(root, findings)
        _check_verifier_strategy(root, rubric, mounts[0], findings)

    return Draft1Document(
        instruction=instruction,
        frontmatter=frontmatter,
        role_prompts={b.arg: _prompt(b) for b in blocks if b.kind == "role" and b.arg},
        scene_prompts={
            b.arg: _prompt(b) for b in blocks if b.kind == "stage" and b.arg
        },
        user_persona=next((_prompt(b) for b in blocks if b.kind == "user"), None),
        config=config,
        unsupported=tuple(findings.unsupported),
        ignored=tuple(findings.ignored),
        canary="; ".join(canaries) or None,
        verifier_mount=mounts[0],
        oracle_mount=mounts[1],
        rubric=rubric,
    )


def _mounts(config: dict[str, Any], findings: _Findings) -> tuple[str, str]:
    """``[verifier] mount`` and ``[oracle] mount``, or the defaults.

    Declared paths outside the supported set are recorded as unsupported, so
    the launch gate refuses the task; the declared path is still returned, and
    ``TaskPaths`` refuses to use it.
    """

    resolved: list[str] = []
    for table, default, supported in (
        ("verifier", DEFAULT_VERIFIER_MOUNT, SUPPORTED_VERIFIER_MOUNTS),
        ("oracle", DEFAULT_ORACLE_MOUNT, SUPPORTED_ORACLE_MOUNTS),
    ):
        mount = _table(config, table).get("mount", default)
        if not isinstance(mount, str):
            raise TaskDocumentParseError(
                f"task.md draft 1: [{table}] mount must be a sandbox path"
            )
        if mount not in supported:
            findings.refuse(
                f"[{table}] mount",
                f"{mount!r} is not supported; this runtime places {table}/ only "
                f"at {' or '.join(supported)}, the paths it locks away from the "
                "agent before the run",
            )
        resolved.append(mount)
    return resolved[0], resolved[1]


def undelivered_prompt_findings(
    document: Draft1Document,
    *,
    scene_names: set[str],
    scene_role_names: set[str],
) -> tuple[Draft1Finding, ...]:
    """Stage and role blocks no v0.6 scene delivers.

    v0.6 shows a scene prompt only in the scene of the same name, and a role
    prompt only to a role that takes part in a scene. Draft-1 stages unlock on
    their own rules, which v0.6 has no way to run.
    """

    findings: list[Draft1Finding] = []
    for name in document.scene_prompts:
        if name not in scene_names:
            findings.append(
                Draft1Finding(
                    f"```stage {name}",
                    "staged prompts are not revealed by this runtime; v0.6 "
                    "shows a scene prompt only in a scene of the same name "
                    "([interaction] scenes)",
                )
            )
    for name in document.role_prompts:
        if name not in scene_role_names:
            findings.append(
                Draft1Finding(
                    f"```role {name}",
                    "role prompts reach an agent only through v0.6 scenes "
                    "([interaction] scenes naming the role)",
                )
            )
    return tuple(findings)


# Parsing ----------------------------------------------------------------------


def _block_kind(info: str) -> tuple[str | None, str | None]:
    """Map a fence info string to (kind, arg); (None, None) is instruction content."""

    words = info.split()
    if not words:
        return None, None
    head = words[0].lower()
    if len(words) == 2 and head in _CONFIG_LANGS and words[1] == "task":
        return "config", head
    if len(words) == 1 and head in _PLAIN_BLOCKS:
        return head, None
    if len(words) == 2 and head in _NAMED_BLOCKS:
        return head, words[1]
    if len(words) == 1 and head in _MOVED_BLOCKS:
        return "moved", head
    return None, None


def _scan(lines: list[str], errors: list[str]) -> tuple[list[str], list[_Block]]:
    """Split the file into instruction lines and task.md blocks."""

    instruction: list[str] = []
    blocks: list[_Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _FENCE.match(line)
        if not match:
            if blocks and line.strip():
                errors.append(
                    f"line {i + 1}: text after the first task.md block; move it "
                    "into the instruction or a notes block"
                )
            elif not blocks:
                instruction.append(line)
            i += 1
            continue
        marker, info = match.group("marker"), match.group("info")
        closing = re.compile(
            rf"^[ ]{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$"
        )
        close = next(
            (j for j in range(i + 1, len(lines)) if closing.match(lines[j])), None
        )
        if close is None:
            errors.append(f"line {i + 1}: unterminated fenced block")
            close = len(lines)
        kind, arg = _block_kind(info)
        if kind == "moved":
            errors.append(
                f"line {i + 1}: ```{arg} blocks are not part of task.md; move "
                f"them to {_MOVED_BLOCKS[str(arg)]}"
            )
        elif kind is None:
            if blocks:
                errors.append(
                    f"line {i + 1}: fenced block ```{info} after the first "
                    "task.md block; move it into the instruction"
                )
            else:
                instruction.extend(lines[i : close + 1])
        else:
            body = "\n".join(lines[i + 1 : close])
            blocks.append(_Block(kind, arg, info, body, i + 1))
        i = close + 1
    return instruction, blocks


def _strip_canary(text: str) -> tuple[list[str], str]:
    """Drop leading HTML comments that contain "canary", then leading blank lines."""

    lines = text.split("\n")
    found: list[str] = []
    while lines and (match := _CANARY_LINE.match(lines[0].strip())):
        found.append(match.group("text").strip())
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return found, "\n".join(lines)


def _prompt(block: _Block) -> str:
    """What an agent sees when a stage, role, or user block is used."""

    return _strip_canary(block.body)[1]


def _parse_config(block: _Block, errors: list[str]) -> dict[str, Any]:
    if block.arg == "toml":
        try:
            return tomllib.loads(block.body)
        except tomllib.TOMLDecodeError as e:
            errors.append(f"line {block.line}: config block is not valid TOML: {e}")
            return {}
    try:
        data = yaml.safe_load(block.body) or {}
    except yaml.YAMLError as e:
        errors.append(f"line {block.line}: config block is not valid YAML: {e}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"line {block.line}: config block must be a mapping")
        return {}
    return data


def _unknown_key(key: str, where: str) -> str:
    hint = _HARBOR_NAMES.get(key)
    if hint:
        return f"{where}{key} is Harbor's name; task.md uses {hint}"
    return f"unknown key {where}{key}"


def _network_ok(value: Any) -> bool:
    if value in ("none", "open"):
        return True
    if isinstance(value, list):
        return all(isinstance(host, str) for host in value)
    return (
        isinstance(value, dict)
        and set(value) == {"block"}
        and isinstance(value["block"], list)
    )


def _check_phase(table: dict[str, Any], where: str, errors: list[str]) -> None:
    for key in ("timeout", "build_timeout"):
        value = table.get(key)
        if key in table and not (
            isinstance(value, str) and _DURATION_SHAPE.match(value)
        ):
            errors.append(
                f'{where} {key} must be a duration such as "90s", "15m", or "1h30m"'
            )
    for key in ("memory", "disk"):
        value = table.get(key)
        if key in table and not (isinstance(value, str) and _SIZE_SHAPE.match(value)):
            errors.append(f'{where} {key} must be a size such as "512 MB" or "4 GB"')
    if "network" in table and not _network_ok(table["network"]):
        errors.append(
            f'{where} network must be "none", "open", a list of hosts, or '
            "{ block = [hosts] }"
        )


def _table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _check_config(
    config: dict[str, Any], blocks: list[_Block], errors: list[str]
) -> None:
    """Draft 1's vocabulary rules: unknown tables and keys are errors unless x-."""

    for key, value in config.items():
        if key in _IDENTITY:
            if isinstance(value, dict):
                errors.append(f"{key} is an identity key, not a table")
        elif key not in _TABLES and not key.startswith("x-"):
            errors.append(_unknown_key(key, ""))
        elif key in _ARRAYS:
            if not (
                isinstance(value, list) and all(isinstance(x, dict) for x in value)
            ):
                errors.append(f"{key} must be written as [[{key}]] tables")
        elif (
            key != "import" and not key.startswith("x-") and not isinstance(value, dict)
        ):
            errors.append(f"[{key}] must be a table")
    for name, allowed in _CLOSED_KEYS.items():
        table = config.get(name)
        if not isinstance(table, dict):
            continue
        for key in table:
            if key not in allowed and not key.startswith("x-"):
                errors.append(_unknown_key(key, f"[{name}] "))
        _check_phase(table, f"[{name}]", errors)
    for name, key, allowed in _ENUMS:
        value = _table(config, name).get(key)
        if value is not None and value not in allowed:
            choices = ", ".join(repr(a) for a in allowed)
            errors.append(f"[{name}] {key} must be one of {choices}")
    verifier_sandbox = _table(config, "verifier").get("sandbox")
    if isinstance(verifier_sandbox, dict):
        _check_phase(verifier_sandbox, "[verifier.sandbox]", errors)
    budget = _table(config, "agent").get("budget")
    if budget is not None and not (
        isinstance(budget, dict) and set(budget) <= {"tool_calls", "tokens"}
    ):
        errors.append("[agent] budget is a table of tool_calls and tokens")
    for credit in config.get("credits") or []:
        if isinstance(credit, dict) and (
            not isinstance(credit.get("name"), str)
            or credit.get("role") not in _CREDIT_ROLES
        ):
            errors.append(
                "each [[credits]] entry has a name and a role: author, advisor, "
                "domain-reviewer, or technical-reviewer"
            )
    if not isinstance(config.get("authors", []), list):
        errors.append("authors must be a list")
    for author in config.get("authors") or []:
        if not (
            isinstance(author, str)
            or (isinstance(author, dict) and isinstance(author.get("name"), str))
        ):
            errors.append(
                'authors entries are "Name <email>" strings or { name, email } tables'
            )
    world = config.get("world")
    if isinstance(world, dict) and world.get("kind") not in _WORLD_KINDS:
        errors.append(f"[world] kind must be one of {sorted(_WORLD_KINDS)}")
    stages = _table(config, "stages")
    staged = {block.arg for block in blocks if block.kind == "stage"}
    at_start = [
        name
        for name, st in stages.items()
        if isinstance(st, dict) and st.get("unlock") == "at_start"
    ]
    if len(at_start) > 1:
        errors.append(
            "only one stage can unlock at_start: its prompt is the instruction"
        )
    for name in at_start:
        if name in staged:
            errors.append(
                f"stage {name} unlocks at_start, so its prompt is the instruction; "
                f"remove its ```stage {name} block"
            )
    for name, st in stages.items():
        if not isinstance(st, dict):
            continue
        for part in ("agent", "verifier"):
            if isinstance(st.get(part), dict):
                _check_phase(st[part], f"[stages.{name}.{part}]", errors)
        if name not in at_start and name not in staged:
            errors.append(f"[stages.{name}] has no ```stage {name} block")


# Config mapping: config_to_v06 (tools/convert.py) ------------------------------


def _seconds(text: Any) -> float:
    match = _DURATION.match(text) if isinstance(text, str) else None
    if not match or not any(match.groupdict().values()):
        raise ValueError(f"not a duration: {text!r} (write 90s, 15m, 2h, or 1h30m)")
    hours, minutes, secs = (float(match[g] or 0) for g in ("h", "m", "s"))
    return hours * 3600 + minutes * 60 + secs


def _megabytes(text: Any) -> int:
    match = _SIZE.match(text) if isinstance(text, str) else None
    if not match:
        raise ValueError(f"not a size: {text!r} (write 512 MB or 4 GB)")
    return round(float(match["n"]) * _UNITS[match["u"]])


def _author(value: Any) -> Any:
    if isinstance(value, str):
        match = _AUTHOR.match(value)
        if match and match["email"]:
            return {"name": match["name"], "email": match["email"]}
        return {"name": value}
    return value


_Convert = Callable[[Any], Any] | str | None
# (v0.6 key, task.md key, conversion). Harbor and v0.6 share these names, except
# the table renames in _V06_TABLE_NAMES and _V06_VERIFIER_NAMES. The spec's
# tool also maps [[verifier.snapshot]] to Harbor's collect and an output's
# service; the v0.6 runtime has neither field, so both are refused here.
_SANDBOX: tuple[tuple[str, str, _Convert], ...] = (
    ("docker_image", "image", None),
    ("os", "os", None),
    ("cpus", "cpus", None),
    ("memory_mb", "memory", _megabytes),
    ("storage_mb", "disk", _megabytes),
    ("gpus", "gpus", None),
    ("gpu_types", "gpu_types", None),
    ("tpu", "tpu", None),
    ("workdir", "workdir", None),
    ("env", "env", None),
    ("skills_dir", "skills", None),
    ("mcp_servers", "mcp", None),
    ("healthcheck", "ready", "ready"),
    ("build_timeout_sec", "build_timeout", _seconds),
)
_READY: tuple[tuple[str, str, _Convert], ...] = (
    ("command", "run", None),
    ("interval_sec", "interval", _seconds),
    ("timeout_sec", "timeout", _seconds),
    ("start_period_sec", "start_period", _seconds),
    ("start_interval_sec", "start_interval", _seconds),
    ("retries", "retries", None),
)
_AGENT: tuple[tuple[str, str, _Convert], ...] = (
    ("timeout_sec", "timeout", _seconds),
    ("user", "user", None),
)
_VERIFIER: tuple[tuple[str, str, _Convert], ...] = (
    ("timeout_sec", "timeout", _seconds),
    ("user", "user", None),
    ("env", "env", None),
    ("environment_mode", "isolation", None),
    ("environment", "sandbox", "sandbox"),
)
_OUTPUT: tuple[tuple[str, str, _Convert], ...] = (
    ("source", "path", None),
    ("destination", "save_as", None),
    ("exclude", "exclude", None),
)
_PACKAGE: tuple[tuple[str, str, _Convert], ...] = (
    ("name", "name", None),
    ("version", "version", None),
    ("description", "description", None),
    ("authors", "authors", "authors"),
    ("keywords", "keywords", None),
)
_STEP: tuple[tuple[str, str, _Convert], ...] = (
    ("min_reward", "gate", None),
    ("healthcheck", "ready", "ready"),
    ("artifacts", "outputs", "outputs"),
)
_NATIVE = {  # the task.md keys that have a v0.6 equivalent
    "sandbox": {n for _, n, _ in _SANDBOX} | {"network", "outputs"},
    "agent": {n for _, n, _ in _AGENT} | {"network"},
    "verifier": {n for _, n, _ in _VERIFIER}
    | {"network", "combine_stages", "unreached_stages"},
    "oracle": {"env"},
    "ready": {n for _, n, _ in _READY},
    "output": {n for _, n, _ in _OUTPUT},
}
_TOP = {
    "name", "title", "version", "description", "authors", "keywords", "about",
    "sandbox", "agent", "verifier", "oracle", "stages", "provenance", "import",
}  # fmt: skip
# v0.6 frontmatter takes Harbor's config model with its own spellings.
_V06_TABLE_NAMES = (("sandbox", "environment"), ("oracle", "solution"))
_V06_VERIFIER_NAMES = (("sandbox_mode", "environment_mode"), ("sandbox", "environment"))
# Values that restate what this runtime already does, so declaring them changes nothing.
_RESTATED_DEFAULTS = {
    (
        "agent",
        "on_timeout",
        "grade",
    ): "the runtime grades what the agent left at the time limit",
    (
        "sandbox",
        "boundary",
        "container",
    ): "every sandbox backend isolates at least a container",
}
_UNSUPPORTED_TABLES = {
    "world": "worlds (desktop, browser, simulator, robot, lab) are not provided",
    "tiers": "sim-to-real tiers are not provided",
    "conventions": "world conventions are not provided",
    "variants": "variants are not run; the runtime would run only the base task",
    "matrix": "condition matrices are not run",
    "family": "seeded task families are not generated",
    "training": "training splits and reward properties are not applied",
    "trajectory": "trajectory requirements are not checked",
    "preference": "the human-preference protocol is not run",
    "runs": (
        "scored-run rules (minimum trials, pinned settings, errored-trial "
        "counting, disclosure) are not enforced"
    ),
    "integrity": "integrity profiles, threat models, and resource classes are not enforced",
}
_UNSUPPORTED_KEYS = {
    ("agent", "on_timeout"): "only grade, the runtime's behavior, is supported",
    ("agent", "budget"): "tool-call and token budgets are not enforced",
    ("agent", "system_prompt_append"): "system prompt additions are not applied",
    ("sandbox", "mounts"): "task files are not mounted at start",
    ("sandbox", "boundary"): "only container isolation is guaranteed on every backend",
    ("verifier", "snapshot"): "snapshot commands are not run before grading",
    ("verifier", "judges"): "rubric judge models are not configured by this runtime",
    ("verifier", "human"): "human judging is not supported",
}


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def _rename_tables(data: dict[str, Any], *, to_v06: bool) -> None:
    """Switch Harbor-shaped data between Harbor's names and v0.6's, in steps too.

    Steps are a list in config and a table keyed by step name in the settings
    ``[import.*]`` keeps (``_rename_v06`` in the spec's tools).
    """

    for v06, harbor in _V06_TABLE_NAMES:
        old, new = (harbor, v06) if to_v06 else (v06, harbor)
        if old in data:
            data[new] = data.pop(old)
    raw_steps: Any = data.get("steps")
    steps: list[Any] = (
        list(raw_steps.values())
        if isinstance(raw_steps, dict)
        else raw_steps
        if isinstance(raw_steps, list)
        else []
    )
    verifiers: list[Any] = [data.get("verifier")]
    verifiers += [step.get("verifier") for step in steps if isinstance(step, dict)]
    for verifier in verifiers:
        if isinstance(verifier, dict):
            for v06, harbor in _V06_VERIFIER_NAMES:
                old, new = (harbor, v06) if to_v06 else (v06, harbor)
                if old in verifier:
                    verifier[new] = verifier.pop(old)


def _is_step_chain(stages: dict[str, Any]) -> bool:
    names = list(stages)
    return all(
        isinstance(st, dict)
        and st.get("unlock") == ("at_start" if i == 0 else f"after:{names[i - 1]}")
        for i, st in enumerate(stages.values())
    )


class _Mapper:
    """task.md config tables -> Harbor-shaped tables, recording what has no equivalent."""

    def __init__(self, findings: _Findings) -> None:
        self.findings = findings

    def table(
        self,
        src: dict[str, Any],
        spec: tuple[tuple[str, str, _Convert], ...],
        where: str,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, name, convert in spec:
            if name not in src:
                continue
            value = src[name]
            inner = f"{where}.{name}" if where else name
            if convert == "ready":
                ready = _as_table(value, inner)
                value = self.table(ready, _READY, inner)
                self.unknown(ready, _NATIVE["ready"], inner)
            elif convert == "sandbox":
                sandbox = _as_table(value, inner)
                value = self.sandbox(sandbox, inner)
                self.unknown(sandbox, _NATIVE["sandbox"] - {"outputs"}, inner)
            elif convert == "outputs":
                items = []
                for item in value:
                    if isinstance(item, dict):
                        self.unknown(item, _NATIVE["output"], inner)
                        item = self.table(item, _OUTPUT, inner)
                    items.append(item)
                value = items
            elif convert == "authors":
                value = [_author(author) for author in value]
            elif convert is not None and not isinstance(convert, str):
                value = convert(value)
            out[key] = value
        return out

    def network(self, src: dict[str, Any], where: str, out: dict[str, Any]) -> None:
        if "network" not in src:
            return
        value = src["network"]
        if value == "none":
            out["network_mode"] = "no-network"
        elif value == "open":
            out["network_mode"] = "public"
        elif isinstance(value, list):
            out["network_mode"] = "allowlist"
            out["allowed_hosts"] = value
        elif isinstance(value, dict) and "block" in value:
            self.findings.refuse(
                f"[{where}] network",
                "block lists (open except the listed hosts) have no v0.6 "
                "equivalent; v0.6 denylists (sandbox.network_mode = "
                '"denylist") also block subdomains and need a sandbox user',
            )
        else:
            raise ValueError(f"not a network setting: {value!r}")

    def sandbox(self, src: dict[str, Any], where: str) -> dict[str, Any]:
        out = self.table(src, _SANDBOX, where)
        self.network(src, where, out)
        return out

    def phase(
        self,
        src: dict[str, Any],
        spec: tuple[tuple[str, str, _Convert], ...],
        where: str,
    ) -> dict[str, Any]:
        out = self.table(src, spec, where)
        self.network(src, where, out)
        return out

    def unknown(self, src: dict[str, Any], known: set[str], where: str) -> None:
        for key, value in src.items():
            if key in known:
                continue
            restated = (
                _RESTATED_DEFAULTS.get((where, key, value))
                if isinstance(value, str)
                else None
            )
            if restated:
                self.findings.ignore(f"[{where}] {key}", restated)
                continue
            reason = _UNSUPPORTED_KEYS.get((where, key), "no v0.6 equivalent")
            self.findings.refuse(f"[{where}] {key}", reason)


def _as_table(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"[{where}] must be a table")
    return value


def _config_to_v06(cfg: dict[str, Any], findings: _Findings) -> dict[str, Any]:
    """task.md config -> v0.6 frontmatter data (``config_to_v06``)."""

    # Mounts are the runtime's to honor (``_mounts``), not v0.6 config keys.
    cfg = {
        key: (
            {k: v for k, v in value.items() if k != "mount"}
            if key in ("verifier", "oracle") and isinstance(value, dict)
            else value
        )
        for key, value in cfg.items()
    }
    mapper = _Mapper(findings)
    out: dict[str, Any] = {}

    # Identity keys become [task]; v0.6, like Harbor, needs a name for it.
    package = mapper.table(cfg, _PACKAGE, "")
    if package and "name" not in package:
        findings.ignore(
            ", ".join(k for _, k, _ in _PACKAGE if k in cfg),
            "descriptive; v0.6 keeps these in [task], which needs a name",
        )
    elif package:
        out["task"] = package
    if "title" in cfg:
        findings.ignore("title", "descriptive")
    if isinstance(cfg.get("about"), dict):
        out["metadata"] = cfg["about"]

    sandbox = cfg.get("sandbox")
    if isinstance(sandbox, dict):
        out["environment"] = mapper.sandbox(sandbox, "sandbox")
        if "outputs" in sandbox:
            out["artifacts"] = mapper.table(sandbox, (_STEP[2],), "sandbox")[
                "artifacts"
            ]
        mapper.unknown(sandbox, _NATIVE["sandbox"], "sandbox")

    verifier = _table(cfg, "verifier")
    for part, spec in (("agent", _AGENT), ("verifier", _VERIFIER)):
        src = cfg.get(part)
        if part == "verifier" and isinstance(src, dict) and "unreached_stages" in src:
            src = {k: v for k, v in src.items() if k != "unreached_stages"}
            if not src:
                continue
        if isinstance(src, dict):
            out[part] = mapper.phase(src, spec, part)
            mapper.unknown(src, _NATIVE[part], part)
    if "combine_stages" in verifier:
        out["multi_step_reward_strategy"] = verifier["combine_stages"]
    oracle = _table(cfg, "oracle")
    if oracle:
        out["solution"] = mapper.table(oracle, (("env", "env", None),), "oracle")
        mapper.unknown(oracle, _NATIVE["oracle"], "oracle")

    stages = _table(cfg, "stages")
    gated = any("gate" in st for st in stages.values() if isinstance(st, dict))
    if (
        stages
        and gated
        and verifier.get("unreached_stages", "zero") == "zero"
        and verifier.get("combine_stages", "mean") == "mean"
    ):
        findings.refuse(
            "[verifier] unreached_stages",
            "draft 1 scores a gated stage the run never reached as 0; v0.6, "
            "like Harbor, averages only the steps that ran",
        )
    if stages and _is_step_chain(stages):
        steps = []
        for name, st in stages.items():
            step: dict[str, Any] = {"name": name}
            for part, spec in (("agent", _AGENT), ("verifier", _VERIFIER)):
                if isinstance(st.get(part), dict):
                    step[part] = mapper.phase(st[part], spec, f"stages.{name}.{part}")
                    known = _NATIVE[part] - {"combine_stages", "unreached_stages"}
                    mapper.unknown(st[part], known, f"stages.{name}.{part}")
            step.update(mapper.table(st, _STEP, f"stages.{name}"))
            mapper.unknown(
                st,
                {"unlock", "agent", "verifier"} | {n for _, n, _ in _STEP},
                f"stages.{name}",
            )
            steps.append(step)
        out["steps"] = steps
    elif stages:
        findings.refuse(
            "[stages]",
            "stages that do not unlock one after another from the start "
            "(on_submit, on_request, at_turn) have no v0.6 equivalent",
        )

    provenance = _table(cfg, "provenance")
    if "source" in provenance:
        out["source"] = provenance["source"]
    for key in provenance:
        if key != "source":
            findings.ignore(f"[provenance] {key}", "descriptive")

    _merge_imports(cfg, out, findings)

    for key, value in cfg.items():
        if (
            key in _TOP
            or key in {"roles", "user", "interaction"}
            or key.startswith("x-")
        ):
            continue
        if key == "credits":
            findings.ignore("[[credits]]", "descriptive")
        elif (
            key == "integrity" and isinstance(value, dict) and set(value) <= {"canary"}
        ):
            findings.ignore(
                "[integrity] canary",
                "contamination marker; the runtime does not use it",
            )
        else:
            findings.refuse(
                f"[{key}]", _UNSUPPORTED_TABLES.get(key, "no v0.6 equivalent")
            )

    _rename_tables(out, to_v06=True)

    interaction = dict(_table(cfg, "interaction"))
    if "roles" in cfg or "agents" in interaction:
        agents = interaction.pop("agents", {})
        roles = {"roles": cfg["roles"]} if "roles" in cfg else {}
        out["agents"] = roles | (agents if isinstance(agents, dict) else {})
    if "scenes" in interaction:
        out["scenes"] = interaction.pop("scenes")
    for key in interaction:
        findings.refuse(f"[interaction] {key}", "only scenes has a v0.6 equivalent")
    if "user" in cfg:
        out["user"] = cfg["user"]
    for key, value in cfg.items():
        if key == "x-benchflow":
            out["benchflow"] = value
        elif key.startswith("x-"):
            findings.refuse(
                f"[{key}]",
                "an extension for another tool; this runtime reads only [x-benchflow]",
            )
    return out


def _merge_imports(
    cfg: dict[str, Any], out: dict[str, Any], findings: _Findings
) -> None:
    """Restore settings an importer kept under [import.harbor] and [import.v06].

    As ``config_to_v06`` does: v0.6 shares Harbor's config model, so both are
    merged back (v0.6's over Harbor's) and validated like any v0.6 setting; a
    key the runtime does not model fails ``TaskConfig`` validation. The spec's
    tool drops other formats and unknown step names silently; here they are
    refused. Harbor's multi-step instruction.md is recorded as ignored.
    """

    imports = cfg.get("import")
    if imports is None:
        return
    if not isinstance(imports, dict):
        raise ValueError("[import] must be a table of formats")
    for stash in imports:
        if stash not in ("harbor", "v06"):
            findings.refuse(
                f"[import.{stash}]",
                "settings kept from a task in another format are not applied",
            )
    kept: dict[str, Any] = {}
    for stash in ("harbor", "v06"):
        table = imports.get(stash)
        if table is None:
            continue
        if not isinstance(table, dict):
            raise ValueError(f"[import.{stash}] must be a table")
        table = copy.deepcopy(table)
        if stash == "v06":
            _rename_tables(table, to_v06=False)
        _merge(kept, table)
    if "instruction" in kept:
        kept.pop("instruction")
        findings.ignore(
            "[import.harbor] instruction",
            "Harbor does not show instruction.md in multi-step tasks",
        )
    step_extra = kept.pop("steps", None)
    if isinstance(step_extra, dict) and "steps" in out:
        index = {step["name"]: i for i, step in enumerate(out["steps"])}
        for name, extra in step_extra.items():
            if name in index and isinstance(extra, dict):
                _merge(out["steps"][index[name]], extra)
            else:
                findings.refuse(f"[import] steps.{name}", "no stage of that name")
    elif step_extra is not None:
        kept["steps"] = step_extra
    _merge(out, kept)


# Prompts and package files ----------------------------------------------------


def _check_prompt_settings(config: dict[str, Any], findings: _Findings) -> None:
    """Role and user settings the v0.6 runtime would pass over without a word."""

    for name, role in _table(config, "roles").items():
        if not isinstance(role, dict):
            continue
        if not role.get("agent"):
            raise TaskDocumentParseError(
                f"task.md draft 1: [roles.{name}] needs agent, the harness that "
                "plays the role; this runtime runs each role as a v0.6 agent role"
            )
        for key in role:
            if key not in _V06_ROLE_KEYS:
                findings.refuse(
                    f"[roles.{name}] {key}", "v0.6 roles do not read this setting"
                )
    for key in _table(config, "user"):
        if key not in _V06_USER_KEYS:
            findings.refuse(
                f"[user] {key}",
                "the v0.6 simulated user reads only model, stop_rule, and private_facts",
            )


def _check_judgment_files(task_dir: Path, findings: _Findings) -> dict[str, Any] | None:
    """``verifier/rubric.json`` and ``verifier/behaviors.json`` in task.md's schema.

    Returns the rubric when the runtime can grade it from the test script's
    CTRF report. Files without task.md's ``$schema`` belong to the verifier's
    own scripts and are left alone, as the reference parser does; a draft-1
    task without a task.md rubric keeps today's contract, where the script
    writes reward.txt or reward.json itself.
    """

    rubric: dict[str, Any] | None = None
    for kind in ("rubric", "behaviors"):
        rel = f"verifier/{kind}.json"
        path = task_dir / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            findings.refuse(
                rel,
                f"cannot be read as JSON, so the runtime cannot tell whether it is the task's {kind}: {e}",
            )
            continue
        if not isinstance(data, dict) or not str(data.get("$schema", "")).startswith(
            f"{_SCHEMA_BASE}{kind}-"
        ):
            continue
        if kind == "behaviors":
            findings.refuse(
                rel,
                "watched behaviors are not detected and their consequences "
                "(fail, invalid, penalties, behavior tags) are not applied",
            )
            continue
        gaps = rubric_gaps(data)
        if gaps:
            findings.refuse(rel, "; ".join(gaps))
            continue
        rubric = data
        if any("stated" in c or "implicit" in c for c in data["criteria"]) or (
            "validation" in data
        ):
            findings.ignore(
                f"{rel} stated, implicit, validation",
                "authoring checks and evidence; not used to grade a run",
            )
    return rubric


def _check_verifier_strategy(
    task_dir: Path,
    rubric: dict[str, Any] | None,
    verifier_mount: str,
    findings: _Findings,
) -> None:
    """A BenchFlow ``verifier/verifier.md`` strategy in a draft-1 package.

    Draft 1 does not define verifier.md. The runtime would run its selected
    strategy instead of test.sh, so it cannot also grade a task.md rubric, and
    its non-script strategies read the verifier folder only at /verifier.
    """

    if not (task_dir / "verifier" / "verifier.md").is_file():
        return
    if rubric is not None:
        findings.refuse(
            "verifier/verifier.md",
            "a verifier.md strategy and a task.md rubric (verifier/rubric.json) "
            "both define grading",
        )
    if verifier_mount != DEFAULT_VERIFIER_MOUNT:
        findings.refuse(
            "verifier/verifier.md",
            f"verifier.md strategies read the verifier folder at "
            f"{DEFAULT_VERIFIER_MOUNT}, not at [verifier] mount = {verifier_mount!r}",
        )
