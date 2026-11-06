"""Agent provisioning inside the sandbox: install + skill deployment.

Owns the "prepare the sandbox for the agent" phase that runs once,
sequentially, before ACP connection:

    install_agent  → registry-driven install_cmd, captures stdout, raises
                     AgentInstallError on non-zero return code
    install_from_layer → the same install, already in the sandbox's image:
                     the cached agent layer (sandbox.env_layers) ran it once
    deploy_skills  → runtime skill upload (Dockerfile-mount fallback) and
                     distribution into agent-specific discovery paths

Together they form the install → distribute lifecycle that the SDK loop
runs as: install_agent → deploy_skills → connect_acp → execute_prompts.

Does not own:
    - Agent / provider env vars — see _agent_env.py
    - Credential file writing — see _credentials.py
"""

import hashlib
import logging
import re
import shlex
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from benchflow._root_only import root_only_path_check, root_only_tree_check
from benchflow._utils.reachable_write import ReachablePath
from benchflow._uv_pin import BENCHFLOW_ROOT
from benchflow.agents.registry import (
    _AGENT_UV_ROOT,
    AGENT_INSTALLERS,
    AGENTS,
    AgentConfig,
    cacheable_install,
)
from benchflow.models import AgentInstallError
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox.env_layers import AgentLayerRecipe
from benchflow.skill_policy import validate_container_mount_path
from benchflow.trust.state import TRUSTED_TRANSFER, IngressKind, admit

logger = logging.getLogger(__name__)

# Root's private staging for install scripts, in the protected runtime tree.
# Every root command here goes to ``env.exec`` with no user: the plumbing
# route (a root-only PATH, no PYTHON* or loader values; FF-install-C14, C15).
_INSTALL_STAGING_ROOT = f"{BENCHFLOW_ROOT}/agent-install"


_ORACLE_SKILL_PATHS = [
    "$HOME/.claude/skills",
    "$HOME/.codex/skills",
    "$HOME/.opencode/skills",
    "$HOME/.agents/skills",
    "$WORKSPACE/skills",
]


def _skill_link_cmd(source: str, dest: str) -> str:
    """Link skills as the destination owner; never chown agent-controlled paths."""
    q_dest = shlex.quote(dest)
    if source == dest:
        return f"mkdir -p {q_dest}"
    parent = shlex.quote(str(Path(dest).parent))
    q_source = shlex.quote(source)
    return f"mkdir -p {parent} && rm -rf {q_dest} && ln -sfn {q_source} {q_dest}"


def _owner_from_home(home: str) -> str | None:
    """Infer the sandbox user for normal /home/<user> agent homes."""
    parts = Path(home).parts
    if len(parts) == 3 and parts[0] == "/" and parts[1] == "home":
        return parts[2]
    return None


def _policy_home_dirs(agent: str, agent_cfg: AgentConfig) -> list[str]:
    """Agent home dirs a web-policy setup command may create."""
    dirs = set(agent_cfg.home_dirs)
    for owned_path in agent_cfg.disallow_web_tools_owned_paths:
        if not owned_path.startswith("$HOME/"):
            raise ValueError(
                f"disallow_web_tools_owned_paths entry {owned_path!r} must start with $HOME/"
            )
        rel = owned_path.removeprefix("$HOME/").rstrip("/")
        if rel:
            dirs.add(rel)
    for skill_path in agent_cfg.skill_paths:
        if skill_path.startswith("$HOME/"):
            rel = skill_path.removeprefix("$HOME/")
            top = rel.split("/", 1)[0]
            if top:
                dirs.add(top)
    if not dirs:
        dirs.add(f".{agent}")
    return sorted(dirs)


async def _link_skill_paths(
    env,
    source: str,
    skill_paths: list[str],
    home: str,
    cwd: str,
    sandbox_user: str | None,
    expected_skill_names: Sequence[str] = (),
) -> int:
    """Link one shared skills tree into each configured discovery path."""
    _VALID_PREFIXES = ("$HOME/", "$WORKSPACE/")
    for sp in skill_paths:
        if not any(sp.startswith(p) for p in _VALID_PREFIXES):
            raise ValueError(f"skill_path {sp!r} must start with $HOME/ or $WORKSPACE/")

    parts = []
    expected = tuple(sorted(set(expected_skill_names)))
    expected_text = "\n".join(expected)
    for sp in skill_paths:
        expanded = sp.replace("$HOME", home).replace("$WORKSPACE", cwd)
        parts.append(_skill_link_cmd(source, expanded))
        q_source = shlex.quote(source)
        q_expanded = shlex.quote(expanded)
        parts.append(
            f"test -d {q_source} && test -d {q_expanded}"
            ' && source_catalog="$(find -L '
            f"{q_source} -mindepth 2 -maxdepth 2 -type f -name SKILL.md "
            "-printf '%h\\n' | sed 's#.*/##' | LC_ALL=C sort -u)" + '"'
            ' && actual="$(find -L '
            f"{q_expanded} -mindepth 2 -maxdepth 2 -type f -name SKILL.md "
            "-printf '%h\\n' | sed 's#.*/##' | LC_ALL=C sort -u)" + '"'
            ' && test "$actual" = "$source_catalog"'
        )
        if expected:
            q_expected = shlex.quote(expected_text)
            parts.append(f'test "$source_catalog" = {q_expected}')
    if parts:
        cmd = " && ".join(parts)
        result = await exec_raw(env.exec, cmd, user=sandbox_user, timeout_sec=15)
        if result.return_code != 0:
            stdout = (diagnostic_result(result).stdout.render() or "").strip()
            stderr = (diagnostic_result(result).stderr.render() or "").strip()
            details = [
                f"exit code {result.return_code}",
                f"command: {cmd}",
            ]
            if stdout:
                details.append(f"stdout: {stdout}")
            if stderr:
                details.append(f"stderr: {stderr}")
            if expected:
                raise RuntimeError(
                    "experiment_fidelity/skill_deployment_missing: "
                    f"expected={','.join(expected)}; {'; '.join(details)}"
                )
            raise RuntimeError(
                f"Failed to link skills from {source}: {'; '.join(details)}"
            )
    return len(skill_paths) if parts else 0


def effective_install_timeout(
    agent: str, sandbox_setup_timeout: int = 120
) -> int | None:
    """Timeout enforced for the agent install step, or None when no install runs.

    Single source of truth shared by :func:`install_agent` and the
    ``config.json`` recorder, so the recorded value always equals the enforced
    one: a per-agent registry ``install_timeout`` overrides the configured
    ``sandbox_setup_timeout``.
    """
    agent_base = agent.split()[0]
    if agent_base not in AGENT_INSTALLERS:
        return None
    agent_cfg = AGENTS.get(agent_base)
    if agent_cfg is None:
        # Defensive fallback, not dead code: an installer can live in
        # AGENT_INSTALLERS without a matching AGENTS entry (e.g. one injected
        # directly into the derived install table by external tooling). The
        # built-in registry keeps the two maps in lockstep, but when they
        # diverge we still bound the install by the configured sandbox setup
        # timeout rather than leaving it unbounded. Covered by
        # test_effective_install_timeout_branches / the fake-agent fallback test.
        return sandbox_setup_timeout
    return agent_cfg.install_timeout


#: Where an agent layer leaves the digest of the install it ran. The trial
#: reads it before it skips that install, so a sandbox whose image does not
#: hold this exact install (another recipe, a task that removed it) installs
#: in the trial as before.
LAYER_STAMP_DIR = "/opt/benchflow/layers"


@dataclass(frozen=True)
class AgentLayerPlan:
    """The cacheable part of one agent's install, as a layer recipe.

    ``install`` is what the trial would run (the registry's install, and a
    native harness's CLI after it); ``recipe`` runs it once in the layer and
    then writes ``stamp`` (the sha256 of ``install``) to ``stamp_path``.
    """

    agent: str
    install: str
    stamp: str
    stamp_path: str
    recipe: AgentLayerRecipe


def agent_layer_plan(
    agent: str, *, harness: str = "acp", sandbox_setup_timeout: int = 120
) -> AgentLayerPlan | None:
    """The layer recipe for ``agent``'s install, or None when it is not cacheable.

    Only an install the registry marks as pinned (``CACHEABLE_INSTALLS``) and
    that still runs the built-in command is cached. Nothing per trial is in
    it: the sandbox user, credentials, skills and lockdown run in the trial
    after it (``Rollout.install_agent``).
    """
    agent_base = agent.split()[0] if agent.split() else ""
    found = cacheable_install(agent_base)
    if found is None:
        return None
    version, install = found
    if harness == "native":
        from benchflow.native_harness.harnesses import NATIVE_HARNESSES

        native = NATIVE_HARNESSES.get(agent_base)
        if native is None:
            return None
        if native.install_cmd:
            # In the trial the CLI installs after the agent, when the agent's
            # install exits 0: the same order and condition.
            install = f"( {install} ) && ( {native.install_cmd} )"
            version = f"{version}+native-{native.package}"
    stamp = "sha256:" + hashlib.sha256(install.encode()).hexdigest()
    stamp_path = f"{LAYER_STAMP_DIR}/agent-{agent_base}.recipe"
    recipe = (
        f"( {install} ) && mkdir -p {LAYER_STAMP_DIR} && "
        f"printf '%s\\n' {stamp} > {stamp_path} && "
        f"chmod 0755 {LAYER_STAMP_DIR} && chmod 0644 {stamp_path}"
    )
    timeout = effective_install_timeout(agent_base, sandbox_setup_timeout)
    return AgentLayerPlan(
        agent=agent_base,
        install=install,
        stamp=stamp,
        stamp_path=stamp_path,
        recipe=AgentLayerRecipe(
            agent=agent_base,
            version=version,
            install_cmd=recipe,
            timeout_sec=timeout or 900,
        ),
    )


async def install_from_layer(
    env, plan: AgentLayerPlan, rollout_dir: Path, *, layer_key: str
) -> str | None:
    """Use the install the sandbox's agent layer holds: None, or why it cannot.

    Reads the layer's stamp in the live sandbox (one command) and, when it
    names ``plan``'s install, writes the install log, runs nothing more and
    returns None. Otherwise it returns the reason, and the caller installs
    in the trial.
    """
    try:
        result = await exec_raw(
            env.exec, f"cat {shlex.quote(plan.stamp_path)}", timeout_sec=30
        )
    except Exception as e:
        return f"the layer's stamp could not be read: {e}"
    found = (getattr(result, "stdout", "") or "").strip()
    if getattr(result, "return_code", 1) != 0 or found != plan.stamp:
        return f"the sandbox holds no {plan.agent} install from layer {layer_key}"
    install_log = ReachablePath.rollout(rollout_dir, "agent") / "install-stdout.txt"
    install_log.parent.mkdir(parents=True, exist_ok=True)
    install_log.write(
        f"$ {plan.install}\n"
        f"=== ran once in the cached agent layer {layer_key} ({plan.stamp}), "
        "not in this trial ===\n"
    )
    logger.info(f"{plan.agent} installed from the cached agent layer")
    return None


async def _run_installer(env, command: str, timeout_sec: int | None) -> ExecResult:
    """Upload large scripts instead of putting their bodies in one exec.

    AgentCore caps the final command at 65,536 characters, including shell
    quoting. Leave room for its user/shell wrappers; uploads use the backend's
    bounded file transport (sealed chunks on AgentCore). Docker layer builds
    still consume the registry's original, self-contained recipe.
    """
    if len(shlex.quote(shlex.quote(command)).encode()) <= 48_000:
        return await exec_raw(env.exec, command, timeout_sec=timeout_sec)

    prepared = await exec_raw(
        env.exec,
        # A 0700 child of /tmp can still be renamed by /tmp's owner (a task
        # may give that workspace to an earlier role). Stage in the protected
        # runtime tree instead, checked root's alone before anything is
        # uploaded. The shared root must remain traversable for the agent's
        # managed Python and launchers; only staging itself is private to root.
        f"mkdir -p -m 755 {BENCHFLOW_ROOT} && "
        f"( umask 077; mkdir -p {_INSTALL_STAGING_ROOT} ) && "
        # The tree's top too, not only as a folder on the way: a sticky one
        # that others may write is refused here (FF-install-C12).
        f"{root_only_path_check(BENCHFLOW_ROOT, 'the agent installer staging')} && "
        f"{root_only_tree_check(_INSTALL_STAGING_ROOT, 'the agent installer staging')}"
        f" && ( umask 077; mktemp -d {_INSTALL_STAGING_ROOT}/install.XXXXXXXXXX )",
        timeout_sec=30,
    )
    if prepared.return_code != 0:
        return prepared
    directory = (prepared.stdout or "").strip()
    if not re.fullmatch(
        re.escape(_INSTALL_STAGING_ROOT) + r"/install\.[A-Za-z0-9]+", directory
    ):
        raise RuntimeError("Agent installer staging returned an invalid directory")
    script = f"{directory}/install.sh"
    try:
        with tempfile.TemporaryDirectory(prefix="benchflow-agent-install-") as local:
            source = Path(local) / "install.sh"
            source.write_text(command)
            await env.upload_file(
                source,
                script,
                admission=admit(
                    env, "main", TRUSTED_TRANSFER, reason="BenchFlow's installer"
                ),
            )
        return await exec_raw(
            env.exec, f"/bin/bash {shlex.quote(script)}", timeout_sec=timeout_sec
        )
    finally:
        # Only the two paths we just created: never recursively delete a
        # backend-provided path. Cleanup must not hide the install's exit code.
        try:
            cleaned = await exec_raw(
                env.exec,
                f"rm -f -- {shlex.quote(script)} && rmdir -- {shlex.quote(directory)}",
                timeout_sec=30,
            )
            if cleaned.return_code != 0:
                logger.warning(
                    "Could not remove agent installer staging: %s", directory
                )
        except Exception:
            logger.warning("Could not remove agent installer staging: %s", directory)


async def install_agent(
    env,
    agent: str,
    rollout_dir: Path,
    sandbox_setup_timeout: int = 120,
    *,
    install_layer_bypass_reason: str | None = None,
) -> AgentConfig | None:
    """Install agent in sandbox and return its config."""
    agent_base = agent.split()[0]
    agent_cfg = AGENTS.get(agent_base)
    if agent_base not in AGENT_INSTALLERS:
        return agent_cfg
    install_cmd = AGENT_INSTALLERS[agent_base]
    install_log = ReachablePath.rollout(rollout_dir, "agent") / "install-stdout.txt"
    # A Daytona image built with this agent's cached layer already holds the
    # install (sandbox.daytona_layers); the user, credentials, skills and
    # lockdown still run per trial after this.
    from benchflow.sandbox.daytona_layers import cached_layer_in_sandbox
    from benchflow.sandbox.env_layers import fresh_install_requested

    bypass_reason = install_layer_bypass_reason
    if fresh_install_requested():
        bypass_reason = "fresh install asked for"
    layer = None
    if bypass_reason is None:
        layer = await cached_layer_in_sandbox(env, "agent", agent_base)
    else:
        note = getattr(type(env), "note_install_layer", None)
        if callable(note):
            note(env, "agent", agent_base, f"not used: {bypass_reason}")
    if layer is not None and layer.script == install_cmd:
        logger.info(f"{agent_base} installed from the cached image layer")
        install_log.parent.mkdir(parents=True, exist_ok=True)
        install_log.write(
            f"$ {install_cmd}\n=== ran in the cached image layer, not here ===\n"
        )
        return agent_cfg
    install_timeout = effective_install_timeout(agent, sandbox_setup_timeout)
    logger.info(f"Installing {agent_base} in sandbox (timeout={install_timeout}s)...")
    # The registry describes both the host-authored lock and its image-build
    # recipe. In a live sandbox upload the lock: an earlier role may have
    # sealed a Python-free kit, which must continue refusing Python helpers.
    # Do not reinterpret a custom install override or parse shell text.
    if agent_cfg is not None and install_cmd == agent_cfg.install_cmd:
        for lock in agent_cfg.requirements_files:
            if (
                Path(lock.path).parent != Path(_AGENT_UV_ROOT)
                or install_cmd.count(lock.command) != 1
            ):
                raise RuntimeError("Invalid agent requirements-file recipe")
            prepared = await exec_raw(
                env.exec,
                f"mkdir -p {_AGENT_UV_ROOT} && "
                + root_only_tree_check(BENCHFLOW_ROOT, "agent requirements upload"),
                timeout_sec=30,
            )
            if prepared.return_code != 0:
                raise RuntimeError(
                    "Agent requirements upload needs a protected /opt/benchflow"
                )
            with tempfile.TemporaryDirectory(
                prefix="benchflow-agent-requirements-"
            ) as local:
                source = Path(local) / "requirements.txt"
                source.write_text(lock.content)
                await env.upload_file(
                    source,
                    lock.path,
                    admission=admit(
                        env,
                        "main",
                        TRUSTED_TRANSFER,
                        reason="BenchFlow's installer lock",
                    ),
                )
            # Docker's upload keeps the host user's uid (FF-INSTALL-ON-PT-R1),
            # and the recipe checks the agents folder root's alone before uv
            # reads the lock. Only root writes in that folder, checked above,
            # so the file there is this upload: make it root's (the link
            # itself, never a target), then check the folder again.
            qlock = shlex.quote(lock.path)
            owned = await exec_raw(
                env.exec,
                f"chown -h 0:0 -- {qlock} && chmod go-w -- {qlock} && "
                + root_only_tree_check(_AGENT_UV_ROOT, "agent requirements upload"),
                timeout_sec=30,
            )
            if owned.return_code != 0:
                raise RuntimeError(
                    "Agent requirements upload could not be made root's alone"
                )
            install_cmd = install_cmd.replace(
                lock.command, f"test -f {shlex.quote(lock.path)}", 1
            )
    install_result = await _run_installer(
        env,
        install_cmd,
        install_timeout,
    )
    install_log.parent.mkdir(parents=True, exist_ok=True)
    stdout = diagnostic_result(install_result).stdout.render() or ""
    stderr = diagnostic_result(install_result).stderr.render() or ""
    parts = [f"$ {install_cmd}\n"]
    if stdout:
        parts.append("=== stdout ===\n")
        parts.append(stdout)
        if not stdout.endswith("\n"):
            parts.append("\n")
    if stderr:
        parts.append("=== stderr ===\n")
        parts.append(stderr)
        if not stderr.endswith("\n"):
            parts.append("\n")
    install_log.write_text("".join(parts))
    if install_result.return_code != 0:
        diag = await exec_raw(
            env.exec,
            "echo 'OS:' && cat /etc/os-release 2>/dev/null | head -2; "
            "echo 'Node:' && node --version 2>&1; "
            f"echo 'Agent:' && which {agent_base} 2>&1",
            timeout_sec=10,
        )
        raise AgentInstallError(
            agent=agent_base,
            return_code=install_result.return_code,
            stdout="".join(parts),
            diagnostics=diagnostic_result(diag).stdout.render() or "",
            log_path=install_log,
        )
    return agent_cfg


async def apply_web_tool_policy(
    env,
    agent: str,
    agent_cfg: AgentConfig | None,
    home: str,
    *,
    disallow: bool,
    disallow_hosted_search: bool = False,
) -> None:
    """Apply the agent's no-web or hosted-search-off setup command in the agent home."""
    if not agent_cfg:
        return
    if disallow:
        policy, setup_cmd = "no-web", agent_cfg.disallow_web_tools_setup_cmd
    elif disallow_hosted_search:
        policy, setup_cmd = "hosted-search", agent_cfg.disallow_hosted_search_setup_cmd
    else:
        return
    if not setup_cmd:
        return

    cmd = f"export BENCHFLOW_AGENT_HOME={shlex.quote(home)}; {setup_cmd}"
    owner = _owner_from_home(home)
    # The home may come from an earlier role or snapshot, including symlinks.
    # Create settings as their consumer so no destination grants root authority.
    _policy_home_dirs(agent, agent_cfg)  # Validate declared home-relative paths.
    result = await exec_raw(env.exec, cmd, user=owner, timeout_sec=15)
    if result.return_code != 0:
        stdout = (diagnostic_result(result).stdout.render() or "").strip()
        stderr = (diagnostic_result(result).stderr.render() or "").strip()
        details = [f"exit code {result.return_code}", f"command: {cmd}"]
        if stdout:
            details.append(f"stdout: {stdout}")
        if stderr:
            details.append(f"stderr: {stderr}")
        raise RuntimeError(
            f"Failed to apply {policy} policy for {agent}: {'; '.join(details)}"
        )


def _prepare_skill_destination_cmd(destination: str, owner: str) -> str:
    """Provision missing directories only below an immutable root-owned chain.

    Stop before a link or a directory another user can modify. The owner will
    make the remaining directories itself. Existing objects never change owner;
    new directories stay root-owned until the recursive walk returns, so their
    parents cannot be swapped before the non-recursive ownership handoff.
    """
    components = " ".join(shlex.quote(part) for part in Path(destination).parts[1:])
    return (
        "unset CDPATH; "
        "bf_skill_dirs() ( "
        '[ "$#" -gt 0 ] || exit 0; '
        "case $(stat -c '%u:%a' .) in 0:755|0:555|0:711|0:700) ;; *) exit 0;; esac; "
        'name=$1; shift; [ ! -L "$name" ] || exit 0; '
        'if [ -e "$name" ]; then '
        '[ -d "$name" ] || exit 0; cd -- "$name" || exit 1; bf_skill_dirs "$@"; '
        'else mkdir -m 755 -- "$name" || exit 1; '
        '(cd -- "$name" && bf_skill_dirs "$@") || exit 1; '
        f'chown -h {shlex.quote(owner)} "$name" || exit 1; fi ); '
        f"cd / && bf_skill_dirs {components}"
    )


async def _upload_skills(
    env, source: Path, destination: str, owner: str, *, agent_derived: bool = False
) -> None:
    """Root transports touch only a fresh private tree; its consumer copies it.

    *agent_derived* skills (a learner store's: earlier agents wrote them) are
    an ingress of agent-derived bytes; the task's and the operator's are
    BenchFlow's own transfer."""
    stage = f"/.benchflow-skills-{uuid4().hex}"

    async def checked(command: str, user: str, phase: str) -> None:
        result = await exec_raw(env.exec, command, user=user, timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(f"Runtime skill {phase} failed")

    await checked(
        "case $(stat -c '%u:%a' /) in 0:755|0:555|0:711|0:700) ;; *) exit 1;; esac; "
        f"umask 077; mkdir -m 700 {stage}",
        "root",
        "staging",
    )
    try:
        await env.upload_dir(
            source,
            f"{stage}/tree",
            admission=(
                admit(env, "main", IngressKind.AGENT_DERIVED_UPLOAD)
                if agent_derived
                else admit(env, "main", TRUSTED_TRANSFER, reason="configured skills")
            ),
        )
        # The private parent prevents agent access until all uploaded objects
        # have root ownership and read-only public permissions. Neither walk
        # follows links from the trusted local skill bundle.
        await checked(
            f"chown -hR root:root {stage} && "
            f"find -P {stage}/tree -type d -exec chmod 755 {{}} + && "
            f"find -P {stage}/tree -type f -exec chmod a+rX,u+w,go-w,ug-s,o-t {{}} + && "
            f"chmod 755 {stage}",
            "root",
            "publication",
        )
        await checked(
            _prepare_skill_destination_cmd(destination, owner),
            "root",
            "directory preparation",
        )
        target = shlex.quote(destination)
        await checked(
            f"mkdir -p -- {target} && cp -R -- {stage}/tree/. {target}/",
            owner,
            "copy",
        )
    finally:
        # Only the generated private tree is removed, never a destination or
        # a link selected by the agent. No agent can replace its root parent.
        await checked(f"rm -rf -- {stage}", "root", "staging cleanup")


async def deploy_skills(
    env,
    task_path: Path,
    skills_dir: str | Path | None,
    agent_cfg,
    sandbox_user: str | None,
    agent_cwd: str,
    skills_sandbox_dir: str | None = None,
    *,
    agent_derived: bool = False,
) -> None:
    """Deploy skills as the consumer (None means the trusted root oracle).

    *agent_derived*: the skills came from earlier agents (a learner store),
    so a runtime upload is admitted as agent-derived bytes.

    A missing agent config also occurs with skip_agent_install; it selects
    generic discovery paths, never a different execution identity.
    """
    expected_skill_names = (
        tuple(sorted(path.parent.name for path in Path(skills_dir).glob("*/SKILL.md")))
        if skills_dir
        else ()
    )
    target_skills_dir = (
        validate_container_mount_path(skills_sandbox_dir or "/skills")
        if skills_dir or skills_sandbox_dir
        else None
    )
    effective_skills = target_skills_dir if skills_sandbox_dir else None

    # Runtime upload (fallback if not baked into Dockerfile)
    if skills_dir:
        assert target_skills_dir is not None
        dockerfile = task_path / "environment" / "Dockerfile"
        injected_copy = f"COPY _deps/skills {target_skills_dir.rstrip('/')}/"
        already_injected = (
            dockerfile.exists() and injected_copy in dockerfile.read_text()
        )
        if not already_injected:
            skills_path = Path(skills_dir)
            if skills_path.is_dir():
                logger.info(f"Deploying skills via runtime upload from {skills_path}")
                await _upload_skills(
                    env,
                    skills_path,
                    target_skills_dir,
                    sandbox_user or "root",
                    agent_derived=agent_derived,
                )
                logger.info("Skills deployed to %s", target_skills_dir)
                effective_skills = target_skills_dir
            else:
                logger.warning(f"Skills dir not found: {skills_path}")
        else:
            logger.info("Skills already injected via Dockerfile")
            effective_skills = target_skills_dir

    # Distribute to agent-specific discovery paths
    if agent_cfg is not None:
        skill_paths = agent_cfg.skill_paths
    else:
        skill_paths = _ORACLE_SKILL_PATHS
    if effective_skills and skill_paths:
        home = f"/home/{sandbox_user}" if sandbox_user else "/root"
        count = await _link_skill_paths(
            env,
            effective_skills,
            skill_paths,
            home,
            agent_cwd,
            sandbox_user,
            expected_skill_names,
        )
        label = agent_cfg.name if agent_cfg else "a scripted agent"
        if count:
            logger.info(f"Skills distributed to {count} paths for {label}")
