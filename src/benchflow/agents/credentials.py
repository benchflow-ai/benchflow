"""Credential file writing into the agent sandbox.

Single home for "writes a file under the agent's credential dir":
    - upload_credential       core helper: stage tmpfile, upload to container
    - write_credential_files  agent + provider credential files (cf. AgentConfig)
    - write_gemini_vertex_settings  ~/.gemini/settings.json for Vertex backend
    - upload_subscription_auth  host login files (e.g. ~/.claude/.credentials.json)
    - isolate_agent_for_proxy_capture  prove process/auth isolation in API mode

The Gemini Vertex settings helper lives here (not in _agent_env.py) so the
module has a single coherent role and zero horizontal imports between phase
modules. Putting it elsewhere creates a two-way cycle with upload_credential.

Does not own:
    - Resolving which env vars become credentials — see _agent_env.py
    - Detecting whether host subscription auth is available — see
      _agent_env.check_subscription_auth (read-only filesystem probe)
"""

import json
import logging
import os
import shlex
import tempfile
from importlib.resources import files as resource_files
from pathlib import Path, PurePosixPath

from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)


_BENCHFLOW_PREFIX = "/opt/benchflow"
_BENCHFLOW_NODE_BIN = f"{_BENCHFLOW_PREFIX}/node/bin/node"
_PROXY_AUTH_CLEANUP_JS = (
    resource_files("benchflow.agents")
    .joinpath("resources", "proxy_auth_cleanup.js")
    .read_text(encoding="utf-8")
    .strip()
)


def _owner_from_home(cred_home: str) -> str | None:
    """Return sandbox username for /home/<user> credential homes."""
    parts = Path(cred_home).parts
    if len(parts) == 3 and parts[0] == "/" and parts[1] == "home":
        return parts[2]
    return None


def _proxy_process_isolation_guard(cred_home: str) -> tuple[str, bool]:
    """Return a root shell guard proving the agent UID has no live process."""

    sandbox_owner = _owner_from_home(cred_home)
    if sandbox_owner is None:
        return "false", False
    quoted_owner = shlex.quote(sandbox_owner)
    return (
        "command -v pgrep >/dev/null 2>&1 && "
        f"bf_agent_uid=$(id -u -- {quoted_owner}) && "
        '[ "$bf_agent_uid" -ne 0 ] && '
        '{ pgrep -u "$bf_agent_uid" >/dev/null 2>&1; '
        'bf_pgrep_rc=$?; [ "$bf_pgrep_rc" -eq 1 ]; }',
        True,
    )


def _proxy_auth_cleanup_command(
    process_guard: str,
    paths: list[str],
    settings_targets: list[dict[str, object]],
) -> str:
    """Build the no-follow cleanup command gated by process isolation.

    JavaScript agents install BenchFlow's private Node runtime before this
    command executes. Python-only agents need not pay that installation cost:
    when the runtime is absent they remain trusted only if every possible
    credential and settings target is already absent. A present target fails
    closed instead of attempting a weaker shell rewrite.
    """

    cleanup = " ".join(
        (
            f"{_BENCHFLOW_NODE_BIN} -e",
            shlex.quote(_PROXY_AUTH_CLEANUP_JS),
            "--",
            shlex.quote(json.dumps(paths, separators=(",", ":"))),
            shlex.quote(json.dumps(settings_targets, separators=(",", ":"))),
        )
    )
    target_paths = [*paths, *(str(target["path"]) for target in settings_targets)]
    absence_checks = " && ".join(
        f"[ ! -e {shlex.quote(path)} ] && [ ! -L {shlex.quote(path)} ]"
        for path in dict.fromkeys(target_paths)
    )
    if not absence_checks:
        absence_checks = "true"
    return (
        f"{process_guard} && if [ -x {_BENCHFLOW_NODE_BIN} ]; then "
        f"{cleanup}; else {absence_checks}; fi"
    )


async def upload_credential(
    env,
    path: str,
    content: str,
    *,
    owner: str | None = None,
) -> None:
    """Write a credential file into the container via upload_file."""
    parent = path.rsplit("/", 1)[0]
    await env.exec(f"mkdir -p {shlex.quote(parent)}", timeout_sec=10)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        await env.upload_file(tmp_path, path)
    finally:
        os.unlink(tmp_path)
    if owner:
        q_owner = shlex.quote(owner)
        q_parent = shlex.quote(parent)
        q_path = shlex.quote(path)
        await env.exec(
            f"chown -R {q_owner}:{q_owner} {q_parent} && chmod 600 {q_path}",
            timeout_sec=10,
        )


async def write_credential_files(
    env,
    agent: str,
    agent_env: dict,
    agent_cfg,
    model: str | None,
    cred_home: str,
) -> None:
    """Write credential files into container from agent + provider configs.

    Two schemas live side by side intentionally — do not unify until a 3rd
    pattern appears. Today: 1 agent (codex `template`-wraps a raw key) + 2
    providers (vertex `post_env`-points GOOGLE_APPLICATION_CREDENTIALS at the
    written file). Same op shape, different intent (compile-time value
    transform vs. runtime env side effect). Provider list is `list[dict]`,
    agent list is `list[CredentialFile]` dataclass — keep dict access vs.
    attribute access straight when editing the loops below.
    """
    # Provider credential files (e.g. GCP ADC for Vertex)
    owner = _owner_from_home(cred_home)
    if model:
        from benchflow.agents.providers import find_provider

        _prov = find_provider(model)
        if _prov:
            _, _prov_cfg = _prov
            for cf in _prov_cfg.credential_files:
                value = agent_env.get(cf["env_source"])
                if value:
                    path = cf["path"].format(home=cred_home)
                    await upload_credential(env, path, value, owner=owner)
                    for k, v in cf.get("post_env", {}).items():
                        agent_env.setdefault(k, v.format(home=cred_home))
                    logger.info("Provider credential file written: %s", path)

    # Gemini CLI needs settings.json to use Vertex AI backend
    await write_gemini_vertex_settings(env, agent, model, cred_home)

    # Agent credential files (e.g. codex auth.json)
    if (
        agent == "codex-acp"
        and "OPENAI_API_KEY" not in agent_env
        and agent_env.get("CODEX_AUTH_JSON")
    ):
        path = f"{cred_home}/.codex/auth.json"
        await upload_credential(env, path, agent_env["CODEX_AUTH_JSON"], owner=owner)
        logger.info("Agent credential file written: %s", path)
        return

    if agent_cfg and agent_cfg.credential_files:
        for cf in agent_cfg.credential_files:
            value = agent_env.get(cf.env_source)
            if value:
                content = cf.template.format(value=value) if cf.template else value
                path = cf.path.format(home=cred_home)
                await upload_credential(env, path, content, owner=owner)
                logger.info("Agent credential file written: %s", path)


async def write_gemini_vertex_settings(
    env,
    agent: str,
    model: str | None,
    cred_home: str,
) -> None:
    """Write ~/.gemini/settings.json to select Vertex AI backend.

    Gemini CLI defaults to API key auth. When a google-vertex/ model is
    used, we must write settings.json with selectedType=vertex-ai so the
    CLI uses ADC instead of looking for GEMINI_API_KEY.

    No conflict with upload_subscription_auth: Vertex models have
    infer_env_key_for_model() return None, so subscription auth is
    never triggered for Vertex — the two paths are mutually exclusive.
    """
    if not model or agent != "gemini":
        return
    from benchflow.agents.registry import is_vertex_model

    if not is_vertex_model(model):
        return
    settings = json.dumps(
        {"security": {"auth": {"selectedType": "vertex-ai"}}},
    )
    path = f"{cred_home}/.gemini/settings.json"
    await upload_credential(env, path, settings, owner=_owner_from_home(cred_home))
    logger.info("Gemini Vertex settings written: %s", path)


async def upload_subscription_auth(
    env,
    agent: str,
    cred_home: str,
) -> None:
    """Upload host subscription auth files into the container.

    Called when _BENCHFLOW_SUBSCRIPTION_AUTH is set, meaning no API key
    was provided but a host auth file was detected.
    """
    agent_cfg = AGENTS.get(agent)
    if not agent_cfg or not agent_cfg.subscription_auth:
        return
    owner = _owner_from_home(cred_home)
    for f in agent_cfg.subscription_auth.files:
        host_path = Path(f.host_path).expanduser()
        if not host_path.is_file():
            continue
        container_path = f.container_path.format(home=cred_home)
        content = host_path.read_text()
        await upload_credential(env, container_path, content, owner=owner)
        logger.info(
            "Subscription auth uploaded: %s -> %s",
            host_path,
            container_path,
        )


async def isolate_agent_for_proxy_capture(
    env,
    *,
    agent: str,
    agent_env: dict[str, str],
    cred_home: str,
) -> bool:
    """Prove process isolation and remove every native route before API proxying.

    A shared sandbox can retain credentials from an earlier role. The current
    agent's login file is therefore not a sufficient cleanup boundary: all
    registry-declared subscription files, credential-bearing settings, and Pi's
    generated provider map are removed through the same no-follow traversal.
    Unsafe effective/config homes and any live sandbox-user process fail capture
    trust closed. Root-agent runs remain audit-only because their processes
    cannot be distinguished safely from orchestration.
    """

    if env is None:
        logger.warning("Cannot prove proxy process isolation for %s", agent)
        return False

    process_guard, processes_safe = _proxy_process_isolation_guard(cred_home)
    home = PurePosixPath(cred_home)
    roots = [home]
    paths_safe = True
    for home_env in ("HOME", "BENCHFLOW_AGENT_HOME"):
        override_value = agent_env.get(home_env, "")
        if override_value and override_value != cred_home:
            override_home = _safe_proxy_auth_root(
                override_value,
                cred_home=cred_home,
                agent=agent,
            )
            if override_home is None:
                paths_safe = False
            else:
                roots.append(override_home)
        agent_env[home_env] = cred_home

    roots = list(dict.fromkeys(roots))
    paths = [str(root / ".pi/agent/models.json") for root in roots]
    settings_targets: list[dict[str, object]] = []
    seen_auth: set[int] = set()
    for registered_agent in AGENTS.values():
        subscription_auth = registered_agent.subscription_auth
        if subscription_auth is None or id(subscription_auth) in seen_auth:
            continue
        seen_auth.add(id(subscription_auth))
        primary_files = [
            auth_file
            for auth_file in subscription_auth.files
            if auth_file.host_path == subscription_auth.detect_file
        ]
        if len(primary_files) != 1:
            logger.warning(
                "Cannot prove proxy isolation for registered %s credentials",
                registered_agent.name,
            )
            paths_safe = False
            continue

        relative_files: list[PurePosixPath] = []
        for auth_file in subscription_auth.files:
            try:
                rendered = PurePosixPath(
                    auth_file.container_path.format(home=cred_home)
                )
                relative_files.append(rendered.relative_to(home))
            except (KeyError, ValueError):
                logger.warning(
                    "Unsafe registered credential path for %s",
                    registered_agent.name,
                )
                paths_safe = False
        for root in roots:
            paths.extend(str(root / relative_path) for relative_path in relative_files)

        try:
            primary_rendered = PurePosixPath(
                primary_files[0].container_path.format(home=cred_home)
            )
            primary_relative = primary_rendered.relative_to(home)
        except (KeyError, ValueError):
            paths_safe = False
            continue
        settings_parents = [root / primary_relative.parent for root in roots]

        if subscription_auth.config_dir_env:
            override_value = agent_env.pop(subscription_auth.config_dir_env, "")
            if override_value:
                override_dir = _safe_proxy_auth_root(
                    override_value,
                    cred_home=cred_home,
                    agent=agent,
                )
                if override_dir is None:
                    paths_safe = False
                else:
                    paths.extend(
                        str(override_dir / relative_path.name)
                        for relative_path in relative_files
                    )
                    settings_parents.append(override_dir)

        if subscription_auth.proxy_settings_file:
            settings_file = PurePosixPath(subscription_auth.proxy_settings_file)
            if (
                not subscription_auth.proxy_settings_drop_keys
                or settings_file.is_absolute()
                or len(settings_file.parts) != 1
                or settings_file.name != subscription_auth.proxy_settings_file
            ):
                logger.warning(
                    "Cannot prove proxy settings isolation for %s",
                    registered_agent.name,
                )
                paths_safe = False
                continue
            for parent in settings_parents:
                target: dict[str, object] = {
                    "path": str(parent / settings_file),
                    "drop_keys": list(subscription_auth.proxy_settings_drop_keys),
                }
                settings_targets.append(target)

    paths = list(dict.fromkeys(paths))
    settings_targets = list(
        {str(target["path"]): target for target in settings_targets}.values()
    )
    cleanup_command = _proxy_auth_cleanup_command(
        process_guard,
        paths,
        settings_targets,
    )
    try:
        result = await env.exec(
            cleanup_command,
            user="root",
            timeout_sec=10,
        )
    except Exception as exc:
        logger.warning("Failed to isolate %s proxy credentials: %s", agent, exc)
        return False
    if result.return_code != 0 or not paths_safe or not processes_safe:
        detail = (getattr(result, "stderr", "") or "").strip()[:500]
        logger.warning(
            "Proxy credential/process isolation failed for %s "
            "(rc=%s, paths_safe=%s, processes_safe=%s)%s",
            agent,
            result.return_code,
            paths_safe,
            processes_safe,
            f": {detail}" if detail else "",
        )
        return False
    return True


def _safe_proxy_auth_root(
    value: str,
    *,
    cred_home: str,
    agent: str,
) -> PurePosixPath | None:
    """Return an in-home auth root, refusing unsafe root-owned deletion paths."""

    candidate = PurePosixPath(value)
    home = PurePosixPath(cred_home)
    try:
        candidate.relative_to(home)
    except ValueError:
        candidate_safe = False
    else:
        candidate_safe = bool(
            candidate.is_absolute()
            and ".." not in candidate.parts
            and "\x00" not in value
        )
    if not candidate_safe:
        logger.warning("Refusing root cleanup outside the sandbox home for %s", agent)
        return None
    return candidate
