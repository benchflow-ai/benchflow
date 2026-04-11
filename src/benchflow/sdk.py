"""benchflow SDK — unified run() that uses ACP inside Harbor environments.

One execution path:
1. Start Harbor environment (Docker or Daytona)
2. Install ACP agent in sandbox
3. Connect via live stdio pipe (ContainerTransport)
4. ACP: initialize → session/new → session/prompt (multi-turn)
5. Capture trajectory from session/update notifications
6. Run Harbor verifier
7. Stop environment
"""

import asyncio
import json
import logging
import os
import re
import shlex
import tempfile
from datetime import datetime
from pathlib import Path

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

from benchflow._env_setup import (
    _create_environment,
    _inject_skills_into_dockerfile,
    _patch_harbor_dind,
    stage_dockerfile_deps,
)
from benchflow._models import AgentInstallError, RunResult
from benchflow._trajectory import (
    _capture_session_trajectory,
    _scrape_agent_trajectory,
)
from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.agents.registry import (
    AGENTS,
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AgentConfig,
    get_sandbox_home_dirs,
)
from benchflow.process import DockerProcess, DaytonaProcess

logger = logging.getLogger(__name__)

_DIAG_TRUNCATE = 2000  # max chars for diagnostic stdout/stderr in logs

# Path lockdown defaults and validation
_DEFAULT_LOCKED = ["/solution", "/tests"]
_SAFE_PATH_RE = re.compile(r"^/[a-zA-Z0-9_./*?\-]+(/[a-zA-Z0-9_./*?\-]+)*$")


def _validate_locked_path(p: str) -> None:
    """Validate a locked path — reject injection and traversal."""
    p_norm = os.path.normpath(p)
    if p_norm != p:
        raise ValueError(
            f"Invalid locked path {p!r}: normalizes to {p_norm!r} — "
            f"use the normalized form directly"
        )
    if any(c == ".." for c in p.split("/")):
        raise ValueError(f"Invalid locked path {p!r}: '..' component not allowed")
    if not _SAFE_PATH_RE.match(p):
        raise ValueError(
            f"Invalid locked path {p!r}: must be absolute, "
            f"alphanumeric with /-_.*? only"
        )
    if p.endswith("/") and p != "/":
        raise ValueError(
            f"Invalid locked path {p!r}: trailing slash not allowed "
            f"(chown on '/dir/' may have unintended scope)"
        )


def _resolve_locked_paths(
    sandbox_user: str | None,
    sandbox_locked_paths: list[str] | None,
) -> list[str]:
    """Resolve effective locked paths.

    - sandbox_user=None → [] (no lockdown)
    - sandbox_user set, paths=None → defaults (/solution, /tests)
    - sandbox_user set, paths=[] → [] (explicit opt-out)
    - sandbox_user set, paths=[...] → union of defaults + caller paths
    """
    if not sandbox_user:
        if sandbox_locked_paths:
            raise ValueError("sandbox_locked_paths requires sandbox_user")
        return []
    if sandbox_locked_paths is None:
        return list(_DEFAULT_LOCKED)
    if not sandbox_locked_paths:
        return []  # explicit opt-out
    return list(dict.fromkeys(_DEFAULT_LOCKED + sandbox_locked_paths))


# Apply DinD patch once at import time
_patch_harbor_dind()

# Re-exported from registry for backwards compat (AGENT_INSTALLERS, AGENT_LAUNCH
# are imported above alongside AGENTS)


class SDK:
    """benchflow SDK.

    Usage:
        sdk = SDK()
        result = await sdk.run(
            task_path="path/to/task",
            agent="claude-agent-acp",
            prompts=["solve the task", "now test your solution"],
            agent_env={"ANTHROPIC_API_KEY": "..."},
        )
        print(result.rewards)
        print(result.trajectory)
    """

    @staticmethod
    async def _upload_credential(env, path: str, content: str) -> None:
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

    async def _write_credential_files(
        self, env, agent: str, agent_env: dict, agent_cfg, model: str | None,
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
                        await self._upload_credential(env, path, value)
                        for k, v in cf.get("post_env", {}).items():
                            agent_env.setdefault(k, v.format(home=cred_home))
                        logger.info("Provider credential file written: %s", path)

        # Gemini CLI needs settings.json to use Vertex AI backend
        await self._write_gemini_vertex_settings(env, agent, model, cred_home)

        # Agent credential files (e.g. codex auth.json)
        if agent_cfg and agent_cfg.credential_files:
            for cf in agent_cfg.credential_files:
                value = agent_env.get(cf.env_source)
                if value:
                    content = cf.template.format(value=value) if cf.template else value
                    path = cf.path.format(home=cred_home)
                    await self._upload_credential(env, path, content)
                    logger.info("Agent credential file written: %s", path)

    async def _write_gemini_vertex_settings(
        self, env, agent: str, model: str | None, cred_home: str,
    ) -> None:
        """Write ~/.gemini/settings.json to select Vertex AI backend.

        Gemini CLI defaults to API key auth. When a google-vertex/ model is
        used, we must write settings.json with selectedType=vertex-ai so the
        CLI uses ADC instead of looking for GEMINI_API_KEY.

        No conflict with _upload_subscription_auth: Vertex models have
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
        await self._upload_credential(env, path, settings)
        logger.info("Gemini Vertex settings written: %s", path)

    async def _upload_subscription_auth(
        self, env, agent: str, cred_home: str,
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
            await self._upload_credential(env, container_path, content)
            logger.info(
                "Subscription auth uploaded: %s -> %s", host_path, container_path,
            )

    @staticmethod
    def _init_trial(
        task_path: Path,
        job_name: str | None,
        trial_name: str | None,
        jobs_dir: str | Path,
    ) -> tuple["Task", Path, "TrialPaths", datetime, str, str]:
        """Set up trial directory tree and return core trial objects."""
        from uuid import uuid4
        task = Task(task_path)
        job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        trial_name = trial_name or f"{task_path.name}__{uuid4().hex[:8]}"
        trial_dir = Path(jobs_dir) / job_name / trial_name
        trial_paths = TrialPaths(trial_dir)
        started_at = datetime.now()
        # Pre-create trial directory tree so Docker doesn't create them as root.
        trial_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("agent", "verifier", "artifacts", "trajectory"):
            (trial_dir / subdir).mkdir(exist_ok=True)
        return task, trial_dir, trial_paths, started_at, job_name, trial_name

    @staticmethod
    def _auto_inherit_env(agent_env: dict[str, str]) -> None:
        """Copy well-known API keys from host os.environ into agent_env."""
        from benchflow.agents.providers import PROVIDERS
        keys = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"}
        for cfg in PROVIDERS.values():
            if cfg.auth_env:
                keys.add(cfg.auth_env)
            for env_var in cfg.url_params.values():
                keys.add(env_var)
        for key in keys:
            if key in os.environ:
                agent_env.setdefault(key, os.environ[key])
        # Mirror GEMINI_API_KEY as GOOGLE_API_KEY (some agents expect one or the other)
        if "GEMINI_API_KEY" in agent_env and "GOOGLE_API_KEY" not in agent_env:
            agent_env["GOOGLE_API_KEY"] = agent_env["GEMINI_API_KEY"]

    @staticmethod
    def _inject_vertex_credentials(agent_env: dict[str, str], model: str) -> None:
        """Inject ADC credentials and defaults for Vertex AI models."""
        from benchflow.agents.registry import is_vertex_model
        if not is_vertex_model(model):
            return
        adc_path = Path.home() / ".config/gcloud/application_default_credentials.json"
        if not adc_path.exists():
            raise ValueError(
                f"Vertex AI model {model!r} requires ADC credentials. "
                f"Run: gcloud auth application-default login"
            )
        agent_env.setdefault("GOOGLE_APPLICATION_CREDENTIALS_JSON", adc_path.read_text())
        agent_env.setdefault("GOOGLE_CLOUD_LOCATION", "global")
        if "GOOGLE_CLOUD_PROJECT" not in agent_env:
            raise ValueError(
                f"GOOGLE_CLOUD_PROJECT required for Vertex AI model {model!r}. "
                f"Export it or pass via --ae GOOGLE_CLOUD_PROJECT=<project>"
            )

    @staticmethod
    def _resolve_provider_env(
        agent_env: dict[str, str], model: str, agent: str,
    ) -> None:
        """Detect provider for model, inject BENCHFLOW_PROVIDER_* and env_mapping."""
        from benchflow.agents.providers import find_provider, resolve_base_url, strip_provider_prefix
        agent_env.setdefault("BENCHFLOW_PROVIDER_MODEL", strip_provider_prefix(model))
        agent_cfg = AGENTS.get(agent)
        # Agent-declared protocol takes precedence over provider's primary so
        # multi-endpoint providers (e.g. zai) route to the right URL.
        agent_protocol = agent_cfg.api_protocol if agent_cfg else ""
        _prov = find_provider(model)
        if _prov:
            _prov_name, _prov_cfg = _prov
            agent_env.setdefault("BENCHFLOW_PROVIDER_NAME", _prov_name)
            try:
                agent_env.setdefault(
                    "BENCHFLOW_PROVIDER_BASE_URL",
                    resolve_base_url(_prov_cfg, agent_env, protocol=agent_protocol or None),
                )
            except KeyError:
                pass  # URL params missing — will fail later with clear error
            agent_env.setdefault(
                "BENCHFLOW_PROVIDER_PROTOCOL",
                agent_protocol or _prov_cfg.api_protocol,
            )
            if _prov_cfg.models:
                agent_env.setdefault("BENCHFLOW_PROVIDER_MODELS",
                                     json.dumps(_prov_cfg.models))
            if _prov_cfg.auth_type == "api_key" and _prov_cfg.auth_env:
                _key = agent_env.get(_prov_cfg.auth_env, "")
                if _key:
                    agent_env.setdefault("BENCHFLOW_PROVIDER_API_KEY", _key)
        # Apply agent env_mapping: translate BENCHFLOW_PROVIDER_* → agent-native vars
        if agent_cfg and agent_cfg.env_mapping:
            for src, dst in agent_cfg.env_mapping.items():
                if src in agent_env:
                    agent_env.setdefault(dst, agent_env[src])

    @staticmethod
    def _check_subscription_auth(agent: str, required_key: str) -> bool:
        """Return True if host subscription auth can substitute for required_key."""
        agent_cfg = AGENTS.get(agent)
        if not agent_cfg or not agent_cfg.subscription_auth:
            return False
        sa = agent_cfg.subscription_auth
        if sa.replaces_env != required_key:
            return False
        return Path(sa.detect_file).expanduser().is_file()

    @staticmethod
    def _resolve_agent_env(
        agent: str,
        model: str | None,
        agent_env: dict[str, str] | None,
    ) -> dict[str, str]:
        """Resolve agent environment: auto-inherit keys, provider vars, env_mapping."""
        agent_env = dict(agent_env or {})
        SDK._auto_inherit_env(agent_env)
        if model:
            SDK._inject_vertex_credentials(agent_env, model)
            SDK._resolve_provider_env(agent_env, model, agent)
            # Validate required API key for the chosen model
            from benchflow.agents.registry import infer_env_key_for_model
            required_key = infer_env_key_for_model(model)
            if required_key and required_key not in agent_env:
                if SDK._check_subscription_auth(agent, required_key):
                    agent_env["_BENCHFLOW_SUBSCRIPTION_AUTH"] = "1"
                    logger.info(
                        "Using host subscription auth (no %s set)", required_key,
                    )
                else:
                    raise ValueError(
                        f"{required_key} required for model {model!r} but not set. "
                        f"Export it, pass via agent_env, or log in with the "
                        f"agent CLI (e.g. claude login, codex --login)."
                    )
        else:
            # No model specified — still check subscription auth for required env vars
            agent_cfg = AGENTS.get(agent)
            if agent_cfg:
                for req_key in agent_cfg.requires_env:
                    if req_key not in agent_env:
                        if SDK._check_subscription_auth(agent, req_key):
                            agent_env["_BENCHFLOW_SUBSCRIPTION_AUTH"] = "1"
                            logger.info(
                                "Using host subscription auth (no %s set)", req_key,
                            )
        # Increase output token limit to avoid truncation errors
        agent_env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "128000")
        # Disable telemetry/non-essential traffic in container
        agent_env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        return agent_env

    @staticmethod
    def _write_config(
        trial_dir: Path,
        *,
        task_path: Path,
        agent: str,
        model: str | None,
        environment: str,
        skills_dir: str | Path | None,
        sandbox_user: str | None,
        context_root: str | Path | None,
        sandbox_locked_paths: list[str] | None = None,
        timeout: int,
        started_at: datetime,
        agent_env: dict[str, str],
    ) -> None:
        """Write config.json to trial_dir with secrets filtered out."""
        _secret_substrings = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIALS")
        recorded_env = {
            k: v for k, v in agent_env.items()
            if not any(s in k.upper() for s in _secret_substrings)
        }
        config_data = {
            "task_path": str(task_path),
            "agent": agent,
            "model": model,
            "environment": environment,
            "skills_dir": str(skills_dir) if skills_dir else None,
            "sandbox_user": sandbox_user,
            "sandbox_locked_paths": sandbox_locked_paths,
            "context_root": str(context_root) if context_root else None,
            "timeout_sec": timeout,
            "started_at": str(started_at),
            "agent_env": recorded_env,
        }
        (trial_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    @staticmethod
    def _build_result(
        trial_dir: Path,
        *,
        task_name: str,
        trial_name: str,
        agent: str,
        agent_name: str,
        model: str,
        n_tool_calls: int,
        prompts: list[str],
        error: str | None,
        verifier_error: str | None,
        trajectory: list[dict],
        partial_trajectory: bool,
        trajectory_source: str | None = None,
        rewards: dict | None,
        started_at: datetime,
        timing: dict[str, float],
    ) -> RunResult:
        """Build RunResult and write result.json, timing.json, prompts.json, trajectory."""
        result = RunResult(
            task_name=task_name,
            trial_name=trial_name,
            rewards=rewards,
            trajectory=trajectory,
            agent=agent,
            agent_name=agent_name,
            model=model,
            n_tool_calls=n_tool_calls,
            n_prompts=len(prompts),
            error=error,
            verifier_error=verifier_error,
            partial_trajectory=partial_trajectory,
            trajectory_source=trajectory_source,
            started_at=started_at,
            finished_at=datetime.now(),
        )
        # Finalize timing
        timing["total"] = (result.finished_at - result.started_at).total_seconds()
        timing = {k: round(v, 1) for k, v in timing.items()}
        # Save trajectory
        traj_dir = trial_dir / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / "acp_trajectory.jsonl").write_text(
            "\n".join(json.dumps(e, default=str) for e in trajectory)
        )
        # Save result.json, prompts.json, timing.json
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": result.task_name,
                    "trial_name": result.trial_name,
                    "rewards": result.rewards,
                    "agent": result.agent,
                    "agent_name": result.agent_name,
                    "model": result.model,
                    "n_tool_calls": result.n_tool_calls,
                    "n_prompts": result.n_prompts,
                    "error": result.error,
                    "verifier_error": result.verifier_error,
                    "partial_trajectory": result.partial_trajectory,
                    "trajectory_source": result.trajectory_source,
                    "started_at": str(result.started_at),
                    "finished_at": str(result.finished_at),
                    "timing": timing,
                },
                indent=2,
            )
        )
        (trial_dir / "timing.json").write_text(json.dumps(timing, indent=2))
        (trial_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
        return result

    @staticmethod
    def _resolve_prompts(task_path: Path, prompts: list[str | None] | None) -> list[str]:
        """Read instruction.md and resolve prompt list."""
        instruction_path = task_path / "instruction.md"
        if not instruction_path.exists():
            raise FileNotFoundError(f"Task missing instruction.md: {task_path}")
        instruction = instruction_path.read_text().strip()
        if prompts is None:
            return [instruction]
        return [p if p is not None else instruction for p in prompts]

    async def _start_env_and_upload(self, env, task_path: Path, timing: dict) -> None:
        """Start environment and upload task files."""
        logger.info(f"Starting environment: {task_path.name}")
        t0 = datetime.now()
        await env.start(force_build=False)
        timing["environment_setup"] = (datetime.now() - t0).total_seconds()
        if (task_path / "instruction.md").exists():
            await env.upload_file(task_path / "instruction.md", "/instruction.md")
        if (task_path / "solution").is_dir():
            await env.upload_dir(task_path / "solution", "/solution")

    async def _run_oracle(self, env, task_path: Path, timeout: int) -> tuple[list[dict], str]:
        """Run oracle mode (solution/solve.sh), return (trajectory, agent_name)."""
        logger.info("Oracle mode: running solution/solve.sh")
        if not (task_path / "solution" / "solve.sh").exists():
            raise FileNotFoundError(f"Oracle requires solution/solve.sh: {task_path}")
        result = await env.exec(
            "chmod +x /solution/solve.sh && "
            "/solution/solve.sh 2>&1 | tee /logs/agent/oracle.txt",
            timeout_sec=timeout,
        )
        if result.return_code != 0:
            logger.warning(f"Oracle solve.sh exited with rc={result.return_code}")
        trajectory = [{
            "type": "oracle",
            "command": "solution/solve.sh",
            "return_code": result.return_code,
            "stdout": (result.stdout or "")[:_DIAG_TRUNCATE],
        }]
        return trajectory, "oracle"

    async def _install_agent(self, env, agent: str, trial_dir: Path) -> AgentConfig | None:
        """Install agent in sandbox and return its config."""
        agent_base = agent.split()[0]
        agent_cfg = AGENTS.get(agent_base)
        if agent_base not in AGENT_INSTALLERS:
            return agent_cfg
        install_timeout = agent_cfg.install_timeout if agent_cfg else 900
        logger.info(f"Installing {agent_base} in sandbox (timeout={install_timeout}s)...")
        install_result = await env.exec(
            AGENT_INSTALLERS[agent_base], timeout_sec=install_timeout,
        )
        install_log = trial_dir / "agent" / "install-stdout.txt"
        install_log.parent.mkdir(parents=True, exist_ok=True)
        install_log.write_text(install_result.stdout or "")
        if install_result.return_code != 0:
            diag = await env.exec(
                "echo 'OS:' && cat /etc/os-release 2>/dev/null | head -2; "
                "echo 'Node:' && node --version 2>&1; "
                f"echo 'Agent:' && which {agent_base} 2>&1",
                timeout_sec=10,
            )
            raise AgentInstallError(
                agent=agent_base,
                return_code=install_result.return_code,
                stdout=install_result.stdout or "",
                diagnostics=diag.stdout or "",
                log_path=str(install_log),
            )
        return agent_cfg

    async def _deploy_skills(
        self, env, task_path: Path, skills_dir: str | Path | None,
        agent_cfg, sandbox_user: str | None, agent_cwd: str, task: "Task",
    ) -> None:
        """Deploy and distribute skills into sandbox."""
        # Runtime upload (fallback if not baked into Dockerfile)
        if skills_dir:
            dockerfile = task_path / "environment" / "Dockerfile"
            already_injected = (
                dockerfile.exists()
                and "COPY _deps/skills /skills/" in dockerfile.read_text()
            )
            if not already_injected:
                skills_path = Path(skills_dir)
                if skills_path.is_dir():
                    logger.info(f"Deploying skills via runtime upload from {skills_path}")
                    await env.upload_dir(skills_path, "/skills")
                    if agent_cfg and agent_cfg.skill_paths:
                        parts = []
                        for sp in agent_cfg.skill_paths:
                            expanded = sp.replace("$HOME", "/root").replace("$WORKSPACE", "/app")
                            parent = str(Path(expanded).parent)
                            parts.append(f"mkdir -p '{parent}' && ln -sf /skills '{expanded}'")
                        await env.exec(" && ".join(parts), timeout_sec=10)
                    logger.info("Skills deployed to /skills and symlinked")
                else:
                    logger.warning(f"Skills dir not found: {skills_path}")
            else:
                logger.info("Skills already injected via Dockerfile")

        # Distribute to agent-specific discovery paths
        task_skills_dir = task.config.environment.skills_dir
        effective_skills = "/skills" if skills_dir else task_skills_dir
        if effective_skills and agent_cfg and agent_cfg.skill_paths:
            home = f"/home/{sandbox_user}" if sandbox_user else "/root"
            parts = []
            for sp in agent_cfg.skill_paths:
                expanded = sp.replace("$HOME", home).replace("$WORKSPACE", agent_cwd)
                q_expanded = shlex.quote(expanded)
                q_skills = shlex.quote(effective_skills)
                parts.append(f"mkdir -p {q_expanded} && cp -r {q_skills}/. {q_expanded}/ 2>/dev/null")
            if parts:
                await env.exec("; ".join(parts), timeout_sec=15)
                logger.info(f"Skills distributed to {len(parts)} paths for {agent_cfg.name}")

    @staticmethod
    def _build_priv_drop_cmd(agent_launch: str, sandbox_user: str) -> str:
        """Build a shell command that drops to sandbox_user via setpriv or su.

        setpriv (util-linux, Debian/Ubuntu) execs directly with no parent process.
        su -l is the universal fallback (works on Alpine/BusyBox too).
        No outer sh -c wrapper — DockerProcess wraps in bash -c already.
        """
        inner = f"export HOME=/home/{sandbox_user} && cd /home/{sandbox_user} && {agent_launch}"
        quoted = shlex.quote(inner)
        return (
            f"if setpriv --help 2>&1 | grep -q reuid; then"
            f" exec setpriv --reuid={sandbox_user} --regid={sandbox_user}"
            f" --init-groups -- bash -c {quoted};"
            f" else exec su -l {sandbox_user} -c {quoted};"
            f" fi"
        )

    @staticmethod
    async def _lockdown_paths(env, paths: list[str]) -> None:
        """Lock directories so the sandbox user cannot access them.

        Runs after all root-level setup but before agent launch.
        Uses chown-then-chmod ordering to prevent TOCTOU window.
        Rejects symlinks and validates path patterns against injection.
        """
        if not paths:
            return

        for p in paths:
            _validate_locked_path(p)

        # Build shell command: reject symlinks, chown before chmod
        parts = []
        for p in paths:
            parts.append(
                f'for d in {p}; do '
                f'  [ -L "$d" ] && echo "WARN: skipping symlink $d" >&2 && continue; '
                f'  [ -e "$d" ] || continue; '
                f'  chown root:root "$d" && chmod 700 "$d"; '
                f'done'
            )
        cmd = " && ".join(parts)
        await env.exec(cmd, timeout_sec=30)

    async def _setup_sandbox_user(self, env, sandbox_user: str, workspace: str) -> str:
        """Create non-root sandbox user, grant workspace access. Return agent_cwd."""
        if not re.match(r'^[a-z_][a-z0-9_-]*$', sandbox_user):
            raise ValueError(f"Invalid sandbox_user: {sandbox_user!r} (must be alphanumeric)")
        logger.info(f"Setting up sandbox user: {sandbox_user}")
        await env.exec(
            f"id -u {sandbox_user} >/dev/null 2>&1 || "
            f"useradd -m -s /bin/bash {sandbox_user} && "
            f"mkdir -p /home/{sandbox_user}/.local/bin && "
            "if [ -d /root/.local/bin ]; then "
            f"cp -aL /root/.local/bin/. /home/{sandbox_user}/.local/bin/ 2>/dev/null || true; fi && "
            "if [ -d /root/.nvm ]; then "
            f"cp -a /root/.nvm/. /home/{sandbox_user}/.nvm/ 2>/dev/null || true; fi && "
            f"for d in {' '.join(sorted(get_sandbox_home_dirs()))}; do "
            f"if [ -d /root/$d ]; then mkdir -p /home/{sandbox_user}/$d && "
            f"cp -a /root/$d/. /home/{sandbox_user}/$d/ 2>/dev/null || true; fi; done && "
            f"chown -R {sandbox_user}:{sandbox_user} /home/{sandbox_user} && "
            f"chown -R {sandbox_user}:{sandbox_user} {shlex.quote(workspace)}",
            timeout_sec=30,
        )
        logger.info(f"Sandbox user {sandbox_user} ready (workspace={workspace})")
        return workspace

    async def _connect_acp(
        self, env, agent: str, agent_launch: str, agent_env: dict,
        sandbox_user: str | None, model: str | None,
        trial_dir: Path, environment: str, agent_cwd: str,
    ) -> tuple[ACPClient, object, str]:
        """Create ACP transport, connect, init session, set model. Return (client, session, agent_name)."""
        # Resolve agent binary path for non-docker environments
        if environment != "docker":
            which_result = await env.exec(f"which {agent_launch.split()[0]}", timeout_sec=10)
            if which_result.return_code == 0 and (which_result.stdout or "").strip():
                full_path = which_result.stdout.strip()
                parts = agent_launch.split()
                parts[0] = full_path
                agent_launch = " ".join(parts)
                logger.info(f"Resolved agent path: {agent_launch}")

        if sandbox_user:
            agent_launch = self._build_priv_drop_cmd(agent_launch, sandbox_user)
            logger.info(f"Agent sandboxed as: {sandbox_user}")

        if environment == "docker":
            live_proc = DockerProcess.from_harbor_env(env)
        else:
            live_proc = await DaytonaProcess.from_harbor_env(env)

        agent_log = trial_dir / "agent" / f"{agent.replace('-', '_')}.txt"
        transport = ContainerTransport(
            container_process=live_proc, command=agent_launch,
            env=agent_env, cwd=agent_cwd, agent_log_path=agent_log,
        )
        acp_client = ACPClient(transport)
        await acp_client.connect()

        init_result = await asyncio.wait_for(acp_client.initialize(), timeout=60)
        agent_name = init_result.agent_info.name if init_result.agent_info else agent
        logger.info(f"ACP agent: {agent_name}")

        session = await asyncio.wait_for(acp_client.session_new(cwd=agent_cwd), timeout=60)
        logger.info(f"Session: {session.session_id}")

        if model:
            from benchflow.agents.providers import strip_provider_prefix
            acp_model_id = strip_provider_prefix(model)
            try:
                await asyncio.wait_for(acp_client.set_model(acp_model_id), timeout=60)
                logger.info(f"Model set to: {acp_model_id} (from {model})")
            except Exception as e:
                logger.warning(f"Failed to set model via ACP: {e}")

        return acp_client, session, agent_name

    async def _execute_prompts(
        self, acp_client: ACPClient, session, prompts: list[str], timeout: int,
    ) -> tuple[list[dict], int]:
        """Send prompts via ACP and capture trajectory. Return (trajectory, n_tool_calls)."""
        for i, prompt in enumerate(prompts):
            logger.info(f"Prompt {i + 1}/{len(prompts)}: {prompt[:80]}...")
            prompt_result = await asyncio.wait_for(
                acp_client.prompt(prompt), timeout=timeout,
            )
            logger.info(
                f"  → {prompt_result.stop_reason.value}, "
                f"{len(session.tool_calls)} total tool calls"
            )
        trajectory = _capture_session_trajectory(session)
        return trajectory, len(session.tool_calls)

    # Trusted env vars for verifier execution — override any agent pollution.
    #
    # PYTEST_DISABLE_PLUGIN_AUTOLOAD intentionally omitted: would break ~94
    # SkillsBench tasks that rely on pytest-json-ctrf's --ctrf flag. Entry-point
    # plugin injection is already blocked by verifier-runs-as-root + system
    # site-packages permissions + the .pth cleanup in _CLEANUP_CMD.
    #
    # PYTHONNOUSERSITE intentionally omitted: verifier runs as root, so the
    # only user-site dir on sys.path is /root/.local which sandbox_user cannot
    # touch, and _CLEANUP_CMD already wipes .pth files there as belt-and-braces.
    _VERIFIER_ENV = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTEST_ADDOPTS": (
            "-c /dev/null "          # block pyproject.toml/pytest.ini/tox.ini/setup.cfg discovery
            "--confcutdir=/tests "   # block conftest.py walk-up beyond /tests
            "--rootdir=/tests "
            "-p no:cacheprovider"
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "",
        "PYTHONHOME": "",
        "PYTHONSTARTUP": "",
        "PYTHONSAFEPATH": "1",       # drop implicit '' (cwd) from sys.path
        "LD_PRELOAD": "",
        "LD_LIBRARY_PATH": "",
    }

    # Cleanup command for pytest hook / Python startup injection.
    # Removes conftest.py outside /tests, sitecustomize.py/usercustomize.py
    # and .pth files from writable sys.path entries (preserves /usr/lib,
    # /usr/local/lib).
    _CLEANUP_CMD = (
        "find / -maxdepth 5 -name conftest.py -not -path '/tests/*' -delete 2>/dev/null; "
        'python3 -c "'
        "import sys,os;"
        "[os.remove(os.path.join(d,f)) "
        " for d in sys.path "
        " for f in ('sitecustomize.py','usercustomize.py') "
        " if d and not d.startswith('/usr/lib') and not d.startswith('/usr/local/lib') "
        " and os.path.isfile(os.path.join(d,f))];"
        "[os.remove(os.path.join(d,f)) "
        " for d in sys.path if d and os.path.isdir(d) "
        " for f in os.listdir(d) if f.endswith('.pth') "
        " and not d.startswith('/usr/lib') and not d.startswith('/usr/local/lib') "
        " and os.path.isfile(os.path.join(d,f))]"
        '" 2>/dev/null || true'
    )

    async def _harden_before_verify(self, env, task: "Task", sandbox_user: str | None) -> None:
        """Neutralize agent tampering before running the verifier.

        1. Kill sandbox-user processes (prevent concurrent writes).
        2. Remove injected conftest.py, sitecustomize.py, .pth files.
        3. Merge trusted env vars into task.config.verifier.env.
        """
        if sandbox_user:
            await env.exec(
                f"pkill -u {sandbox_user} 2>/dev/null; "
                f"sleep 1; pkill -9 -u {sandbox_user} 2>/dev/null || true",
                timeout_sec=10,
            )
        await env.exec(self._CLEANUP_CMD, timeout_sec=10)

        verifier_env = dict(self._VERIFIER_ENV)
        if task.config.verifier.env:
            verifier_env.update(task.config.verifier.env)
        task.config.verifier.env = verifier_env

    async def _verify(self, env, task: "Task", trial_paths: "TrialPaths", timing: dict, sandbox_user: str | None = None) -> tuple[dict | None, str | None]:
        """Run verifier with pre-verification hardening."""
        trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        await self._harden_before_verify(env, task, sandbox_user)
        logger.info("Running verifier...")
        t0 = datetime.now()
        verifier_error = None
        try:
            verifier = Verifier(task=task, trial_paths=trial_paths, environment=env)
            verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=task.config.verifier.timeout_sec,
            )
            timing["verifier"] = (datetime.now() - t0).total_seconds()
            rewards = verifier_result.rewards
            logger.info(f"Rewards: {rewards}")
        except asyncio.TimeoutError:
            timing["verifier"] = (datetime.now() - t0).total_seconds()
            # NOTE: these prefixes must stay in sync with classify_verifier_error() in _scoring.py
            verifier_error = f"verifier timed out after {task.config.verifier.timeout_sec}s"
            rewards = None
            logger.error(verifier_error)
        except Exception as e:
            timing["verifier"] = (datetime.now() - t0).total_seconds()
            # NOTE: these prefixes must stay in sync with classify_verifier_error() in _scoring.py
            verifier_error = f"verifier crashed: {e}"
            rewards = None
            logger.error(verifier_error)
        return rewards, verifier_error

    async def run(
        self,
        task_path: str | Path,
        agent: str = "claude-agent-acp",
        prompts: list[str | None] | None = None,
        *,
        model: str | None = None,
        agent_env: dict[str, str] | None = None,
        job_name: str | None = None,
        trial_name: str | None = None,
        jobs_dir: str | Path = "jobs",
        environment: str = "docker",
        skills_dir: str | Path | None = None,
        sandbox_user: str | None = "agent",
        sandbox_locked_paths: list[str] | None = None,
        pre_agent_hooks: list | None = None,
        context_root: str | Path | None = None,
    ) -> RunResult:
        """Run a task with an ACP agent inside a sandbox.

        Args:
            task_path: Path to Harbor-format task directory
            agent: ACP agent name or command (e.g. "claude-agent-acp", "openclaw")
            prompts: List of prompts to send. Default: [instruction.md content]
            model: Model to use (e.g. "claude-haiku-4-5-20251001"). Set via ACP session/set_model.
            agent_env: Environment variables for the agent (API keys etc.)
            job_name: Job name. Auto-generated if not provided.
            trial_name: Custom trial name. Auto-generated if not provided.
            jobs_dir: Directory for job output (Harbor convention).
            environment: Environment type — "docker" or "daytona".
            skills_dir: Path to skills directory. Copied into sandbox and symlinked
                to agent-specific discovery paths (e.g. ~/.claude/skills/).
            sandbox_user: Run agent as this non-root user (e.g. "agent"). Uses
                setpriv (Debian/Ubuntu) or su (Alpine/others) — no external
                dependencies. Setup (install) and verification run as root.
            pre_agent_hooks: List of async callables(env) to run after setup but
                before agent launch. Use for starting background services, etc.
            context_root: Repo root for resolving Dockerfile COPY paths. When set,
                scans environment/Dockerfile for COPY sources relative to this root,
                copies them into environment/_deps/, and rewrites the Dockerfile.

        Returns:
            RunResult with rewards, trajectory, and metadata.
        """
        if sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown. "
                "Root can read solution/test files. "
                "Set sandbox_user='agent' for answer integrity."
            )

        # Resolve effective locked paths
        effective_locked = _resolve_locked_paths(sandbox_user, sandbox_locked_paths)

        task_path = Path(task_path)
        task, trial_dir, trial_paths, started_at, job_name, trial_name = self._init_trial(
            task_path, job_name, trial_name, jobs_dir,
        )
        agent_env = self._resolve_agent_env(agent, model, agent_env)
        prompts = self._resolve_prompts(task_path, prompts)
        agent_launch = AGENT_LAUNCH.get(agent, agent)

        if context_root:
            stage_dockerfile_deps(task_path, Path(context_root))
        if skills_dir:
            _inject_skills_into_dockerfile(task_path, Path(skills_dir))

        env = _create_environment(environment, task, task_path, trial_name, trial_paths)
        timeout = task.config.agent.timeout_sec
        timing: dict[str, float] = {}

        self._write_config(
            trial_dir,
            task_path=task_path, agent=agent, model=model, environment=environment,
            skills_dir=skills_dir, sandbox_user=sandbox_user, context_root=context_root,
            sandbox_locked_paths=effective_locked,
            timeout=timeout, started_at=started_at, agent_env=agent_env,
        )

        acp_client: ACPClient | None = None
        trajectory: list[dict] = []
        partial_trajectory = False
        trajectory_source: str | None = None
        agent_name = ""
        n_tool_calls = 0
        error = None
        verifier_error = None
        rewards = None

        try:
            await self._start_env_and_upload(env, task_path, timing)
            t_agent_setup = datetime.now()
            t_agent_exec = t_agent_setup

            for hook in (pre_agent_hooks or []):
                await hook(env)

            if agent == "oracle":
                trajectory, agent_name = await self._run_oracle(env, task_path, timeout)
            else:
                agent_cfg = await self._install_agent(env, agent, trial_dir)
                cred_home = f"/home/{sandbox_user}" if sandbox_user else "/root"
                await self._write_credential_files(
                    env, agent, agent_env, agent_cfg, model, cred_home,
                )
                if agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
                    await self._upload_subscription_auth(env, agent, cred_home)

                # Detect working directory (preserved when sandbox user is set)
                cwd_result = await env.exec("pwd", timeout_sec=10)
                agent_cwd = (cwd_result.stdout or "").strip() or "/app"
                if sandbox_user:
                    agent_cwd = await self._setup_sandbox_user(env, sandbox_user, workspace=agent_cwd)

                await self._deploy_skills(
                    env, task_path, skills_dir, agent_cfg, sandbox_user, agent_cwd, task,
                )

                await self._lockdown_paths(env, effective_locked)

                acp_client, session, agent_name = await self._connect_acp(
                    env, agent, agent_launch, agent_env, sandbox_user,
                    model, trial_dir, environment, agent_cwd,
                )
                timing["agent_setup"] = (datetime.now() - t_agent_setup).total_seconds()
                t_agent_exec = datetime.now()

                trajectory, n_tool_calls = await self._execute_prompts(
                    acp_client, session, prompts, timeout,
                )
                trajectory_source = "acp"

            if agent != "oracle" and "agent_setup" not in timing:
                timing["agent_setup"] = (datetime.now() - t_agent_setup).total_seconds()
            if agent == "oracle":
                timing["agent_execution"] = (datetime.now() - t_agent_setup).total_seconds()
            elif "agent_execution" not in timing:
                timing["agent_execution"] = (datetime.now() - t_agent_exec).total_seconds()

            # Fallback: scrape agent-native trajectory if ACP captured nothing
            if not trajectory and agent != "oracle":
                scraped = await _scrape_agent_trajectory(env, agent, sandbox_user)
                if scraped:
                    trajectory = scraped
                    trajectory_source = "scraped"
                    # Do NOT overwrite n_tool_calls — keep ACP-sourced value (trusted).
                    # Scraped trajectory is agent-writable and forgeable.
                    logger.warning(
                        f"Using scraped trajectory ({len(scraped)} events) from "
                        f"agent-writable directory — data is UNTRUSTED"
                    )

            rewards, verifier_error = await self._verify(env, task, trial_paths, timing, sandbox_user=sandbox_user)

        except asyncio.TimeoutError:
            error = f"Agent timed out after {timeout}s"
            logger.error(error)
        except ConnectionError as e:
            error = str(e)
            logger.error(f"Agent connection lost: {error}")
        except Exception as e:
            error = str(e)
            logger.error("Run failed", exc_info=True)

        finally:
            if not trajectory and acp_client:
                try:
                    trajectory = _capture_session_trajectory(acp_client.session)
                    if trajectory:
                        partial_trajectory = True
                        trajectory_source = "partial_acp"
                        n_tool_calls = len(acp_client.session.tool_calls)
                        logger.info(f"Captured {len(trajectory)} partial trajectory events")
                except Exception as e:
                    logger.warning(f"Partial trajectory capture failed: {e}")

            if acp_client:
                try:
                    await acp_client.close()
                except Exception as e:
                    logger.warning(f"ACP client close failed: {e}")
            try:
                await env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        return self._build_result(
            trial_dir,
            task_name=task_path.name,
            trial_name=trial_name,
            agent=agent,
            agent_name=agent_name,
            model=model or "",
            n_tool_calls=n_tool_calls,
            prompts=prompts,
            error=error,
            verifier_error=verifier_error,
            trajectory=trajectory,
            partial_trajectory=partial_trajectory,
            trajectory_source=trajectory_source,
            rewards=rewards,
            started_at=started_at,
            timing=timing,
        )
