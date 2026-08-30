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
from pathlib import Path, PurePosixPath

from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)


_BENCHFLOW_PREFIX = "/opt/benchflow"
_BENCHFLOW_NODE_BIN = f"{_BENCHFLOW_PREFIX}/node/bin/node"
_PROXY_AUTH_CLEANUP_JS = r"""
const fs = require("fs");

const targets = JSON.parse(process.argv[1]);
const constants = fs.constants;
const directoryFlags =
  constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW;

function openDirectoryNoFollow(parts) {
  let descriptor = fs.openSync("/", directoryFlags);
  try {
    for (const part of parts) {
      const next = fs.openSync(
        `/proc/self/fd/${descriptor}/${part}`,
        directoryFlags,
      );
      fs.closeSync(descriptor);
      descriptor = next;
    }
    return descriptor;
  } catch (error) {
    fs.closeSync(descriptor);
    throw error;
  }
}

function removeCredentialNoFollow(target) {
  const parts = target.split("/").filter(Boolean);
  const basename = parts.pop();
  let parent;
  try {
    parent = openDirectoryNoFollow(parts);
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
  try {
    const descriptorPath = `/proc/self/fd/${parent}/${basename}`;
    try {
      fs.unlinkSync(descriptorPath);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    try {
      fs.lstatSync(descriptorPath);
    } catch (error) {
      if (error.code === "ENOENT") return;
      throw error;
    }
    throw new Error(`credential remained after deletion: ${target}`);
  } finally {
    fs.closeSync(parent);
  }
}

for (const target of targets) removeCredentialNoFollow(target);
""".strip()


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
    """Prove process isolation and remove native auth before an API proxy run.

    A reused image may already contain the CLI login detected by
    ``SubscriptionAuth.detect_file``. API-key mode must not leave that alternate
    provider route available to the agent. A safe CLI config-home override is
    scrubbed as well as removed before launch; an override outside the sandbox
    user's home fails capture trust without allowing a root deletion there.
    Every API-proxied agent first requires a non-root sandbox user with no live
    processes; an existing agent or task process is preserved and makes capture
    audit-only. Subscription-capable agents then use their already-required
    JavaScript runtime to traverse with no-follow directory descriptors.
    Root-agent runs remain audit-only because stale root processes cannot be
    safely distinguished. Return false unless process isolation is proven and
    every eligible credential can be identified, removed, and verified absent.
    """

    agent_cfg = AGENTS.get(agent)
    subscription_auth = agent_cfg.subscription_auth if agent_cfg else None
    if env is None:
        logger.warning("Cannot prove proxy process isolation for %s", agent)
        return False

    process_guard, processes_safe = _proxy_process_isolation_guard(cred_home)

    if subscription_auth is None:
        try:
            result = await env.exec(
                process_guard,
                user="root",
                timeout_sec=10,
            )
        except Exception as exc:
            logger.warning("Failed to isolate %s agent processes: %s", agent, exc)
            return False
        return bool(result.return_code == 0 and processes_safe)

    primary_files = [
        auth_file
        for auth_file in subscription_auth.files
        if auth_file.host_path == subscription_auth.detect_file
    ]
    if len(primary_files) != 1:
        logger.warning(
            "Cannot prove proxy isolation for %s subscription credentials", agent
        )
        return False

    home = PurePosixPath(cred_home)
    primary_path = primary_files[0].container_path.format(home=cred_home)
    primary_relative_path = PurePosixPath(primary_path).relative_to(home)
    paths = [primary_path]
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
                paths.append(str(override_home / primary_relative_path))
        agent_env[home_env] = cred_home

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
                paths.append(str(override_dir / PurePosixPath(primary_path).name))

    paths = list(dict.fromkeys(paths))
    cleanup_command = f"{process_guard} && " + " ".join(
        (
            f"{_BENCHFLOW_NODE_BIN} -e",
            shlex.quote(_PROXY_AUTH_CLEANUP_JS),
            "--",
            shlex.quote(json.dumps(paths, separators=(",", ":"))),
        )
    )
    try:
        result = await env.exec(
            cleanup_command,
            user="root",
            timeout_sec=10,
        )
    except Exception as exc:
        logger.warning("Failed to isolate %s subscription credential: %s", agent, exc)
        return False
    if result.return_code != 0 or not paths_safe or not processes_safe:
        detail = (getattr(result, "stderr", "") or "").strip()[:500]
        logger.warning(
            "Subscription credential remained accessible in proxy mode for %s%s",
            agent,
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
