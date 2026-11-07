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
import posixpath
import shlex
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from string import Formatter
from uuid import uuid4

from benchflow.agents.login_dirs import login_path
from benchflow.agents.muse_config import validate_muse_model
from benchflow.agents.registry import AGENTS, is_vertex_model
from benchflow.sandbox._exec_output import exec_raw
from benchflow.trust.state import TRUSTED_TRANSFER, admit

logger = logging.getLogger(__name__)

# These runtime files can contain API keys or OAuth tokens. Keep this inventory
# beside the credential writers, including files written by harness launchers
# rather than upload_credential. Evidence capture records every exclusion.
CREDENTIAL_EVIDENCE_PATHS = (
    ".codex/auth.json",
    ".codex/config.toml",
    ".claude/.credentials.json",
    ".config/opencode/auth.json",
    ".config/opencode/opencode.json",
    ".local/share/opencode/auth.json",
    ".config/mimocode/mimocode.json",
    "mimocode.json",
    ".openhands/agent_settings.json",
    ".pi/agent/models.json",
    ".pi/agent/auth.json",
    ".openclaw/openclaw.json",
    ".openclaw/agents/main/agent/auth-profiles.json",
    ".gemini/oauth_creds.json",
    ".config/muse/auth.json",
    ".aws/credentials",
    ".config/gcloud/application_default_credentials.json",
    ".ssh",
)


def credential_evidence_overrides(
    agent_env: Mapping[str, str], *, workspace: str, cred_home: str
) -> tuple[str, ...]:
    """Resolve supported credential/config file overrides in sandbox paths.

    Values are filenames, not credential JSON. In particular, CODEX_CONFIG is
    inline JSON forwarded to the adapter and must never be treated as a path.
    """
    paths: set[str] = set()
    for variable in (
        "MIMOCODE_CONFIG",
        "OPENCODE_CONFIG",
        "OPENCLAW_CONFIG_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_SHARED_CREDENTIALS_FILE",
    ):
        value = agent_env.get(variable)
        if not value:
            continue
        if value.startswith("~/"):
            value = str(PurePosixPath(cred_home) / value[2:])
        paths.add(posixpath.normpath(str(PurePosixPath(workspace) / value)))
    return tuple(sorted(paths))


def _owner_from_home(cred_home: str) -> str | None:
    """Return sandbox username for /home/<user> credential homes."""
    parts = Path(cred_home).parts
    if len(parts) == 3 and parts[0] == "/" and parts[1] == "home":
        return parts[2]
    return None


def _credential_parent_steps(path: str) -> str:
    """Create a literal parent chain, never traversing a symbolic link.

    The caller has already stopped the previous role's writers. Do not
    canonicalize through a link: the ledger owns this name, not its target.
    """
    if not path.startswith("/") or path == "/" or ".." in path.split("/"):
        raise RuntimeError("Agent credential path must be an absolute file path")
    parts = PurePosixPath(path).parts[1:-1]
    steps = ["export PATH=/usr/bin:/bin", "umask 077"]
    for i in range(len(parts)):
        parent = shlex.quote("/" + "/".join(parts[: i + 1]))
        steps.append(
            f"[ ! -L {parent} ] && "
            f"{{ [ -d {parent} ] || mkdir -- {parent}; }} && [ -x {parent} ]"
        )
    return " && ".join(steps)


async def prepare_launcher_credential(
    env,
    path: str,
    *,
    owner: str | None,
    record: Callable[[str], None],
) -> None:
    """Ledger a launcher's destination and give it a private, unaliased inode.

    Preserve regular config contents for launchers that merge settings. A
    symlink is unlinked without reading its target; hard links are detached
    by copying into a fresh inode. An absent file stays absent (native login
    selection can depend on absence). Launchers create new files with 077.
    A failed preparation remains a cleanup obligation, like partial uploads.
    """
    record(path)
    dest = shlex.quote(path)
    command = (
        _credential_parent_steps(path)
        + f" && if [ -L {dest} ]; then rm -f -- {dest}; "
        + f"elif [ -e {dest} ]; then "
        + f"[ -f {dest} ] || exit 1; "
        # Hold the old bytes in an FD, not an unledgered temporary file.
        + f"(exec 9< {dest} && rm -f -- {dest} && "
        + f"set -C && cat <&9 > {dest}); fi"
    )
    result = await exec_raw(env.exec, command, user=owner or "root", timeout_sec=10)
    if _integer_status(result) != 0:
        raise RuntimeError("Launcher credential preparation failed")


async def upload_credential(
    env,
    path: str,
    content: str,
    *,
    owner: str | None = None,
) -> None:
    """Upload privately, then write agent paths only with their owner's rights.

    Transfer APIs run as root. Their destination must therefore never be in
    an agent-controlled tree, including on role changes and snapshot restores.
    A fresh directory directly below the protected filesystem root gives the
    upload, ownership transfer and cleanup a parent the agent cannot replace.
    """
    parent_steps = _credential_parent_steps(path)
    stage = f"/.runtime-credential-{uuid4().hex}"
    prepare = (
        "case $(stat -c '%u:%a' /) in 0:755|0:555|0:711|0:700) ;; *) exit 1;; esac; "
        f"umask 077; mkdir -m 700 {stage}"
    )
    result = await exec_raw(env.exec, prepare, user="root", timeout_sec=10)
    if _integer_status(result) != 0:
        raise RuntimeError("Private credential staging failed")
    staged_path = f"{stage}/credential"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        await env.upload_file(
            tmp_path,
            staged_path,
            mode="600",
            admission=admit(
                env,
                "main",
                TRUSTED_TRANSFER,
                reason="the operator's credentials, staged by BenchFlow",
            ),
        )
        if owner:
            # Only this newly uploaded inode is handed over. Its parent stays
            # root-owned, so the agent cannot redirect any privileged operation.
            result = await exec_raw(
                env.exec,
                f"chown {shlex.quote(owner)} {staged_path} && chmod 711 {stage}",
                user="root",
                timeout_sec=10,
            )
            if _integer_status(result) != 0:
                raise RuntimeError("Private credential handoff failed")
        result = await exec_raw(
            env.exec,
            f"{parent_steps} && "
            f"rm -f -- {shlex.quote(path)} && "
            f"(set -C; cat {staged_path} > {shlex.quote(path)})",
            user=owner or "root",
            timeout_sec=10,
        )
        if _integer_status(result) != 0:
            raise RuntimeError("Agent credential installation failed")
    finally:
        os.unlink(tmp_path)
        result = await exec_raw(
            env.exec,
            f"rm -f {staged_path} && rmdir {stage}",
            user="root",
            timeout_sec=10,
        )
        if _integer_status(result) != 0:
            raise RuntimeError("Private credential staging cleanup failed")


async def place_credential(
    env,
    path: str,
    content: str,
    *,
    owner: str | None,
    record: Callable[[str], None] | None,
) -> None:
    """The one staging write: record the destination, then upload.

    Recording first puts a partially completed upload, or one followed by a
    failing sibling, on the caller's cleanup ledger (C27).
    """
    if record is not None:
        record(path)
    await upload_credential(env, path, content, owner=owner)


async def remove_muse_credentials(
    env, sandbox_user: str, *, timeout_sec: int = 10
) -> None:
    """Remove the managed login only after the caller has stopped its writers.

    Run as the credential owner, never root on an agent-controlled parent.
    Refuse symlinked parents and verify absence without reading login bytes.
    A folder is looked into only once the owner can search it: in a folder
    it cannot search, a login looks absent to it (AG-R2-1), so that is a
    refusal, never "nothing to remove". Root then confirms the login is
    gone, by name only. A failure prevents a new role or shared verifier
    from being admitted.
    """
    home = f"/home/{sandbox_user}"
    parents = " ".join(
        shlex.quote(path) for path in (home, f"{home}/.config", f"{home}/.config/muse")
    )
    path = shlex.quote(f"{home}/.config/muse/auth.json")
    command = (
        f"for d in {parents}; do "
        'if [ -L "$d" ]; then echo "Muse credential cleanup refused: linked parent" >&2; exit 1; fi; '
        'if [ ! -e "$d" ]; then exit 0; fi; '
        '[ -d "$d" ] || exit 1; '
        'if [ ! -x "$d" ]; then echo "Muse credential cleanup refused: unsearchable parent" >&2; exit 1; fi; '
        "done; "
        f"rm -f -- {path} && [ ! -e {path} ] && [ ! -L {path} ]"
    )
    result = await exec_raw(
        env.exec, command, user=sandbox_user, timeout_sec=timeout_sec
    )
    absent = await exec_raw(
        env.exec,
        f"[ ! -e {path} ] && [ ! -L {path} ]",
        user="root",
        timeout_sec=timeout_sec,
    )
    if _integer_status(result) != 0 or _integer_status(absent) != 0:
        # Sandbox output may be agent-controlled: do not include it or content.
        raise RuntimeError(
            "Muse credential cleanup failed; refusing another role or verifier"
        )


async def remove_staged_credentials(
    env,
    sandbox_user: str | None,
    paths: Sequence[str],
    *,
    timeout_sec: int = 10,
) -> None:
    """Remove credential files a connection staged, wherever they were declared.

    Run as the credential owner (root only when the agent runs as root): it
    can remove nothing the agent itself could not. Every parent directory is
    checked: a symlink or an
    unsearchable parent is a refusal; a missing one means the file is
    already gone. Root then confirms every path is absent, by name only. A
    failure raises, so no next role is admitted while a previous login may
    still be readable.
    """
    lines: list[str] = []
    names: list[str] = []
    for path in dict.fromkeys(paths):
        path = posixpath.normpath(path)
        if not path.startswith("/") or path == "/":
            raise RuntimeError("Credential cleanup refused: not an absolute file path")
        parts = path.split("/")[1:-1]
        parents = ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]
        quoted = shlex.quote(path)
        names.append(quoted)
        checks = " ".join(shlex.quote(d) for d in parents)
        lines.append(
            f"present=1; for d in {checks}; do "
            'if [ -L "$d" ]; then exit 1; fi; '
            'if [ ! -e "$d" ]; then present=0; break; fi; '
            '[ -d "$d" ] && [ -x "$d" ] || exit 1; '
            "done; "
            f'if [ "$present" = 1 ]; then rm -f -- {quoted} && '
            f"[ ! -e {quoted} ] && [ ! -L {quoted} ] || exit 1; fi"
        )
    if not names:
        return
    result = await exec_raw(
        env.exec,
        "; ".join(lines),
        user=sandbox_user or "root",
        timeout_sec=timeout_sec,
    )
    absent = await exec_raw(
        env.exec,
        " && ".join(f"[ ! -e {n} ] && [ ! -L {n} ]" for n in names),
        user="root",
        timeout_sec=timeout_sec,
    )
    if _integer_status(result) != 0 or _integer_status(absent) != 0:
        # Sandbox output may be agent-controlled: do not include it or content.
        raise RuntimeError(
            "Credential cleanup failed; refusing another role or verifier"
        )


async def muse_login_present(env, sandbox_user: str, *, timeout_sec: int = 10) -> bool:
    """Whether a managed Muse login may be in *sandbox_user*'s home.

    Asked as root, by name only (no folder mode hides it from root). Anything
    but a clear "absent" counts as present, so the caller removes it.
    """
    path = shlex.quote(f"/home/{sandbox_user}/.config/muse/auth.json")
    result = await exec_raw(
        env.exec,
        f"if [ -e {path} ] || [ -L {path} ]; then exit 0; else exit 1; fi",
        user="root",
        timeout_sec=timeout_sec,
    )
    return _integer_status(result) != 1


def _integer_status(result) -> int | None:
    """The exit status of *result*, when it is an actual integer.

    Existing adapters also use exit_code. Accept either spelling, but require
    an actual integer status: absent/untyped success cannot prove removal.
    """
    for name in ("return_code", "exit_code"):
        value = getattr(result, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


async def write_credential_files(
    env,
    agent: str,
    agent_env: dict,
    agent_cfg,
    model: str | None,
    cred_home: str,
    *,
    record: Callable[[str], None] | None = None,
) -> list[str]:
    """Write credential files into container from agent + provider configs.

    Every destination is passed to *record* before it is written (the
    connection's cleanup ledger), and the paths written are returned.

    Two schemas live side by side intentionally — do not unify until a 3rd
    pattern appears. Today: 1 agent (codex `template`-wraps a raw key) + 2
    providers (vertex `post_env`-points GOOGLE_APPLICATION_CREDENTIALS at the
    written file). Same op shape, different intent (compile-time value
    transform vs. runtime env side effect). Provider list is `list[dict]`,
    agent list is `list[CredentialFile]` dataclass — keep dict access vs.
    attribute access straight when editing the loops below.
    """
    if agent == "muse-code":
        # Defense in depth for callers that bypass environment resolution.
        validate_muse_model(model)
    written: list[str] = []
    # Every file below is placed from the run's credential table
    # (launch_env.CredentialPolicy), the same one the launch variables use:
    # a file holding a credential is written into the agent's home only when
    # the table admits that credential for this agent. On a proxy route that
    # is never an upstream credential: Vertex ADC stays in the proxy's own
    # storage (the host, or its mode-600 sandbox runtime dir). The Gemini
    # CLI's native Vertex route admits the ADC file as its login.
    from benchflow.agents.launch_env import credential_policy

    policy = credential_policy(agent, model, agent_env)

    def _value(source: str) -> str | None:
        return policy.admitted_value(source, agent_env)

    # Provider credential files (e.g. GCP ADC for Vertex). Muse's subscription
    # is agent-owned and must never upload an unrelated provider credential.
    # On a proxy route the table admits no ADC, so none is written here.
    owner = _owner_from_home(cred_home)
    if model and agent != "muse-code":
        from benchflow.agents.providers import find_provider

        _prov = find_provider(model)
        if _prov:
            _, _prov_cfg = _prov
            for cf in _prov_cfg.credential_files:
                value = _value(cf["env_source"])
                if value:
                    path = cf["path"].format(home=cred_home)
                    await place_credential(env, path, value, owner=owner, record=record)
                    written.append(path)
                    for k, v in cf.get("post_env", {}).items():
                        agent_env.setdefault(k, v.format(home=cred_home))
                    logger.info("Provider credential file written: %s", path)

    # Gemini CLI's Vertex backend calls Vertex itself with the ADC: only its
    # native route selects it, never a proxy route.
    if policy.vertex_settings:
        settings = await write_gemini_vertex_settings(
            env, agent, model, cred_home, record=record
        )
        if settings:
            written.append(settings)

    # Agent credential files (e.g. codex auth.json): only the login the
    # route selected, never an inherited auth.json of another account.
    codex_auth = _value("CODEX_AUTH_JSON") if agent == "codex-acp" else None
    # A reconnect retains the launcher's explicit empty API-key guard. Only
    # an admitted key can outrank this login, not the presence of that slot
    # (or a value the credential table withheld from this connection).
    if codex_auth and not _value("OPENAI_API_KEY"):
        path = f"{cred_home}/.codex/auth.json"
        await place_credential(env, path, codex_auth, owner=owner, record=record)
        logger.info("Agent credential file written: %s", path)
        return [*written, path]

    if agent_cfg and agent_cfg.credential_files:
        for cf in agent_cfg.credential_files:
            if not agent_env.get(cf.env_source):
                continue
            if _interpolates(cf.template):
                # Rendered only from an admitted value, whatever the
                # template does with it (repr, format spec, encoding).
                value = _value(cf.env_source)
                if value is None:
                    continue
                content = cf.template.format(value=value) if cf.template else value
            else:
                # Structurally no replacement field (mimo's escaped
                # {env:XIAOMI_API_KEY}): the file holds no credential.
                content = cf.template.format()
            path = cf.path.format(home=cred_home)
            await place_credential(env, path, content, owner=owner, record=record)
            written.append(path)
            logger.info("Agent credential file written: %s", path)
    return written


def _interpolates(template: str) -> bool:
    """Whether a credential-file template has any replacement field.

    An empty template is the raw value. Any field at all counts, so a
    template can only skip admission by containing no field structurally.
    """
    if not template:
        return True
    return any(field is not None for _, field, _, _ in Formatter().parse(template))


async def write_gemini_vertex_settings(
    env,
    agent: str,
    model: str | None,
    cred_home: str,
    *,
    record: Callable[[str], None] | None = None,
) -> str | None:
    """Write ~/.gemini/settings.json to select Vertex AI backend; its path.

    Gemini CLI defaults to API key auth. When a google-vertex/ model is
    used, we must write settings.json with selectedType=vertex-ai so the
    CLI uses ADC instead of looking for GEMINI_API_KEY.

    No conflict with upload_subscription_auth: Vertex models have
    infer_env_key_for_model() return None, so subscription auth is
    never triggered for Vertex — the two paths are mutually exclusive.
    """
    if not model or agent != "gemini":
        return None
    if not is_vertex_model(model):
        return None
    settings = json.dumps(
        {"security": {"auth": {"selectedType": "vertex-ai"}}},
    )
    path = f"{cred_home}/.gemini/settings.json"
    await place_credential(
        env, path, settings, owner=_owner_from_home(cred_home), record=record
    )
    logger.info("Gemini Vertex settings written: %s", path)
    return path


async def upload_subscription_auth(
    env,
    agent: str,
    cred_home: str,
    *,
    record: Callable[[str], None] | None = None,
) -> list[str]:
    """Upload host subscription auth files into the container; their paths.

    Called when _BENCHFLOW_SUBSCRIPTION_AUTH is set, meaning no API key
    was provided but a host auth file was detected.
    """
    agent_cfg = AGENTS.get(agent)
    if not agent_cfg or not agent_cfg.subscription_auth:
        return []
    owner = _owner_from_home(cred_home)
    written: list[str] = []
    for f in agent_cfg.subscription_auth.files:
        host_path = login_path(f.host_path)
        if not host_path.is_file():
            continue
        container_path = f.container_path.format(home=cred_home)
        content = host_path.read_text()
        await place_credential(env, container_path, content, owner=owner, record=record)
        written.append(container_path)
        logger.info(
            "Subscription auth uploaded: %s -> %s",
            host_path,
            container_path,
        )
    return written
