"""Credential file writing into the agent sandbox.

Single home for "writes a file under the agent's credential dir":
    - upload_credential       core helper: stage tmpfile, upload to container
    - write_credential_files  agent + provider credential files (cf. AgentConfig)
    - write_gemini_vertex_settings  ~/.gemini/settings.json for Vertex backend
    - upload_subscription_auth  host login files (e.g. ~/.claude/.credentials.json)

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
import tempfile
from pathlib import Path

from benchflow.agents.registry import AGENTS

logger = logging.getLogger(__name__)


async def upload_credential(env, path: str, content: str) -> None:
    """Write a credential file into the container via upload_file."""
    parent = path.rsplit("/", 1)[0]
    await env.exec(f"mkdir -p {parent}", timeout_sec=10)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        await env.upload_file(tmp_path, path)
    finally:
        os.unlink(tmp_path)


async def write_credential_files(
    env,
    agent: str,
    agent_env: dict,
    agent_cfg,
    model: str | None,
    cred_home: str,
) -> None:
    """Write credential files into container from agent + provider configs."""
    # Provider credential files (e.g. GCP ADC for Vertex)
    if model:
        from benchflow.agents.providers import find_provider

        _prov = find_provider(model)
        if _prov:
            _, _prov_cfg = _prov
            for cf in _prov_cfg.credential_files:
                value = agent_env.get(cf["env_source"])
                if value:
                    path = cf["path"].format(home=cred_home)
                    await upload_credential(env, path, value)
                    for k, v in cf.get("post_env", {}).items():
                        agent_env.setdefault(k, v.format(home=cred_home))
                    logger.info("Provider credential file written: %s", path)

    # Gemini CLI needs settings.json to use Vertex AI backend
    await write_gemini_vertex_settings(env, agent, model, cred_home)

    # Agent credential files (e.g. codex auth.json)
    if agent_cfg and agent_cfg.credential_files:
        for cf in agent_cfg.credential_files:
            value = agent_env.get(cf.env_source)
            if value:
                content = cf.template.format(value=value) if cf.template else value
                path = cf.path.format(home=cred_home)
                await upload_credential(env, path, content)
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
    await upload_credential(env, path, settings)
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
    for f in agent_cfg.subscription_auth.files:
        host_path = Path(f.host_path).expanduser()
        if not host_path.is_file():
            continue
        container_path = f.container_path.format(home=cred_home)
        content = host_path.read_text()
        await upload_credential(env, container_path, content)
        logger.info(
            "Subscription auth uploaded: %s -> %s",
            host_path,
            container_path,
        )
