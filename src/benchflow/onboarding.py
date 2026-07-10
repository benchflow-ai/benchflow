"""Onboarding logic behind ``bench init`` / ``bench doctor``.

The logic the CLI wizard is thin glue over: a private env file for
credentials (secrets live in env vars, per the provider registry's
``auth_env`` contract), TOML preferences, provider/agent compatibility
filtering, and the health checks the post-init smoke test and ``bench
doctor`` share.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchflow.agents.providers import ProviderConfig


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """One dotenv line -> (key, value), or None for comments/malformed."""
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, _, raw = line.partition("=")
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        return None
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return key, value


def write_env_file(path: str | Path, updates: dict[str, str]) -> None:
    """Merge *updates* into the env file at *path* (created, mode forced 0600).

    The merge is line-preserving: keys already present are replaced in place,
    new keys are appended, and everything the parser does not understand —
    comments, malformed lines — is kept verbatim, so a hand-edited file
    survives the next init. An undecodable file raises OSError instead of
    being clobbered.
    """
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    except UnicodeDecodeError as exc:
        raise OSError(
            f"refusing to rewrite {path}: not valid UTF-8 ({exc}); fix or"
            " delete it first"
        ) from exc
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        parsed = _parse_env_line(line)
        if parsed and parsed[0] in remaining:
            out.append(f'{parsed[0]}="{remaining.pop(parsed[0])}"')
        else:
            out.append(line)
    out.extend(f'{k}="{v}"' for k, v in remaining.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text("\n".join(out) + "\n")


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse a dotenv-style file; missing/unreadable file reads as empty.

    Tolerates the dialects people hand-paste: ``export KEY=...`` prefixes and
    single- or double-quoted values. Malformed lines (empty or whitespace
    keys) and undecodable bytes are skipped/degraded — never fatal, because
    the CLI auto-loads this file on every invocation, including the ones
    needed to repair it.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        import sys

        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        parsed = _parse_env_line(line)
        if parsed:
            values[parsed[0]] = parsed[1]
    return values


def load_env_file(path: str | Path) -> list[str]:
    """setdefault every env-file entry into os.environ; report what was set.

    The real environment always wins — the file only fills gaps, so an
    exported key or a CI secret is never clobbered by a stale init.
    """
    import os

    applied = []
    for key, value in read_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied


def save_prefs(path: str | Path, prefs: dict) -> None:
    """Write run preferences (agent/model/dataset/sandbox...) as TOML."""
    import tomli_w

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(prefs).encode())


def load_prefs(path: str | Path) -> dict:
    """Read preferences TOML; missing file reads as empty."""
    import tomllib

    path = Path(path)
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def resolve_provider(model: str) -> tuple[str, ProviderConfig] | None:
    """Map a model id ("provider/model" or bare) to (provider_name, config).

    Returns None when no registered provider claims the model — the wizard
    then falls back to the model's inferred well-known key
    (claude-*/gpt-*/gemini-* via infer_env_key_for_model), or errors if none
    can be inferred.
    """
    from benchflow.agents.providers import PROVIDERS, find_provider_for_bare_model

    if "/" in model:
        prefix = model.split("/", 1)[0]
        cfg = PROVIDERS.get(prefix)
        return (prefix, cfg) if cfg else None
    return find_provider_for_bare_model(model)


def compatible_agents(model: str | None = None) -> list[str]:
    """Registered agent names that can actually route *model*.

    With no model (the agent-first wizard flow) the full registered list is
    returned, minus the never-routed agents; the protocol filter then applies
    in the other direction (compatible_providers).

    Reuses the same provider-protocol gate the run path enforces
    (env._provider_supports_agent_protocol), so the wizard never offers an
    agent the run would immediately reject — e.g. anthropic-messages or
    openai-responses agents on an openai-completions-only provider. Agents
    that bypass provider routing entirely are never offered (gemini speaks
    its provider's native wire; oracle is not a registered agent — special-
    cased in rollout — and is listed defensively).
    """
    from benchflow.agents.env import _provider_supports_agent_protocol
    from benchflow.agents.registry import AGENTS

    non_routed = {"oracle", "gemini"}
    resolved = resolve_provider(model) if model else None
    names = []
    for name, cfg in sorted(AGENTS.items()):
        if name in non_routed:
            continue
        if resolved and not _provider_supports_agent_protocol(
            resolved[1], cfg.api_protocol or ""
        ):
            continue
        names.append(name)
    return names


from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class CheckResult:
    """One doctor/smoke row: what was checked, verdict, human detail.

    ``skipped=True`` marks a check that could not be exercised (e.g. run-time
    auth) — it never fails a doctor run (``ok`` stays True) but is rendered
    distinctly and counted in the summary so green does not over-certify.
    """

    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


def _sanitize(text: str) -> str:
    """Strip control characters so server-supplied text cannot inject
    terminal escape sequences through doctor/smoke output."""
    return "".join(ch for ch in text if ch.isprintable() or ch in " \t")


def shell_join(argv: list[str]) -> str:
    """Render argv as a copy-pasteable shell command."""
    import shlex

    return shlex.join(argv)


def normalize_dataset_input(
    dataset: str,
    *,
    local_tasks_dir: bool = False,
    cwd: str | Path | None = None,
) -> str:
    """Normalize task-dir values without changing valid registry specs."""
    if "/" in dataset or dataset.startswith("."):
        return dataset
    if local_tasks_dir or (Path(cwd or ".") / dataset).is_dir():
        return f"./{dataset}"
    return dataset


def _ping_headers(prov_name: str, protocol: str, key: str) -> dict[str, str]:
    if protocol == "openai-completions":
        if prov_name.startswith("azure-foundry-"):
            return {"api-key": key}
        return {"Authorization": f"Bearer {key}"}
    if protocol == "anthropic-messages":
        # x-api-key is the Anthropic wire header (anthropic-version is
        # required by that wire contract); Azure's Anthropic surface accepts
        # api-key too, so send both key headers.
        return {
            "x-api-key": key,
            "api-key": key,
            "anthropic-version": "2023-06-01",
        }
    return {}


def _openai_completion_token_limit(bare_model: str) -> dict[str, int]:
    """Return the supported minimal token limit for chat-completions pings."""
    if bare_model.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": 8}
    return {"max_tokens": 1}


def _run_path_env(model: str, env: dict[str, str], agent: str | None) -> dict[str, str]:
    """Normalize a check env with the same resolver used before real runs."""
    if not agent:
        return dict(env)
    from benchflow.agents.env import resolve_agent_env

    try:
        return resolve_agent_env(agent, model, env)
    except Exception:
        return dict(env)


def saved_setup_env_updates(
    *,
    agent: str,
    model: str,
    env: dict[str, str],
    updates: dict[str, str],
) -> dict[str, str]:
    """Return env-file updates, including derived provider setup values."""
    saved = dict(updates)
    resolved = resolve_provider(model)
    if not resolved or not resolved[0].startswith("azure-foundry-"):
        return saved
    normalized = _run_path_env(model, {**env, **updates}, agent)
    resource = normalized.get("AZURE_RESOURCE")
    if resource:
        saved.setdefault("AZURE_RESOURCE", resource)
    return saved


def model_ping(model: str, env: dict[str, str], transport=None) -> CheckResult:
    """Verify key + model id + endpoint with one minimal completion.

    GET /models is not used: it can 200 while the actual route is broken
    (wrong model id, upstream 5xx) — a minimal completion is the cheapest
    request that exercises the full path. The endpoint uses the same
    resolve_base_url + path-segment join as the run path, but prefers the
    openai-completions endpoint when a provider has several — it validates
    the key/route, not necessarily the endpoint the chosen agent will use.
    Provider classes the ping cannot exercise (ADC, Bedrock bearer auth) are
    skipped honestly instead of reported as failures.
    """
    import httpx

    from benchflow.agents.providers import resolve_base_url, strip_provider_prefix

    resolved = resolve_provider(model)
    if not resolved:
        from benchflow.agents.registry import infer_env_key_for_model

        if infer_env_key_for_model(model):
            # Well-known family (claude-*/gpt-*/gemini-*) with no registered
            # endpoint: nothing to ping — the run wires it natively.
            return CheckResult(
                "model ping",
                True,
                "skipped — no registered endpoint to ping for this model"
                " family; auth is exercised at run time",
                skipped=True,
            )
        return CheckResult("model ping", False, f"no registered provider for {model!r}")
    prov_name, cfg = resolved
    name = f"model ping ({prov_name})"
    if cfg.auth_type != "api_key":
        return CheckResult(
            name,
            True,
            f"skipped — {cfg.auth_type} auth is exercised at run time",
            skipped=True,
        )
    key = env.get(cfg.auth_env or "", "")
    if not key:
        return CheckResult(name, False, f"{cfg.auth_env} is not set")

    endpoints = cfg.all_endpoints
    bare_model = strip_provider_prefix(model)
    if "openai-completions" in endpoints:
        protocol = "openai-completions"
        path, ok_field = "/chat/completions", "choices"
        headers = _ping_headers(prov_name, protocol, key)
        payload = {
            "model": bare_model,
            "messages": [{"role": "user", "content": "ping"}],
            **_openai_completion_token_limit(bare_model),
        }
    elif "anthropic-messages" in endpoints:
        protocol = "anthropic-messages"
        path, ok_field = "/v1/messages", "content"
        headers = _ping_headers(prov_name, protocol, key)
        payload = {
            "model": bare_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    else:
        return CheckResult(
            name,
            True,
            f"skipped — no pingable endpoint ({sorted(endpoints)})",
            skipped=True,
        )
    try:
        base = resolve_base_url(cfg, env, protocol=protocol).rstrip("/")
    except KeyError as exc:
        return CheckResult(name, False, _sanitize(str(exc)))
    url = f"{base}{path}"
    import logging

    _httpx_logger = logging.getLogger("httpx")
    _prev_level = _httpx_logger.level
    _httpx_logger.setLevel(logging.WARNING)  # keep request logs out of the rows
    try:
        with httpx.Client(transport=transport, timeout=30) as client:
            resp = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        return CheckResult(name, False, _sanitize(f"request failed: {exc}"))
    finally:
        _httpx_logger.setLevel(_prev_level)
    if resp.status_code == 200:
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and ok_field in body:
            return CheckResult(name, True, f"1-token completion OK ({url})")
        return CheckResult(
            name,
            False,
            _sanitize(f"200 but not a completion response: {resp.text[:200]}"),
        )
    return CheckResult(
        name, False, _sanitize(f"HTTP {resp.status_code}: {resp.text[:200]}")
    )


def run_doctor(
    model: str,
    sandbox: str,
    env: dict[str, str],
    ping_transport=None,
    skip_ping: bool = False,
    agent: str | None = None,
) -> list[CheckResult]:
    """The shared check set behind `bench doctor` and the init smoke stage.

    When *agent* is given and a host subscription login covers the model's
    key (check_subscription_auth), the key/route/ping rows are skipped — a
    subscription setup must not be failed by checks that only understand API
    keys.
    """
    import shutil

    env = _run_path_env(model, env, agent)
    results: list[CheckResult] = []
    if sandbox == "docker":
        found = shutil.which("docker")
        results.append(
            CheckResult(
                "docker",
                bool(found),
                found or "docker binary not found on PATH",
            )
        )
    elif sandbox == "daytona":
        has = bool(env.get("DAYTONA_API_KEY"))
        results.append(
            CheckResult(
                "daytona (DAYTONA_API_KEY)",
                has,
                "set" if has else "DAYTONA_API_KEY is not set",
            )
        )
    else:
        from benchflow.sandbox.providers import SANDBOX_PROVIDERS

        known = sandbox in SANDBOX_PROVIDERS
        results.append(
            CheckResult(
                "sandbox",
                known,
                f"{sandbox} (no init-time checks for this provider)"
                if known
                else f"unknown sandbox {sandbox!r} (expected one of"
                f" {', '.join(SANDBOX_PROVIDERS)})",
            )
        )
    resolved = resolve_provider(model)
    auth_env: str | None = None
    if resolved:
        _, cfg = resolved
        if cfg.auth_type == "api_key":
            auth_env = cfg.auth_env
    else:
        # No registered provider endpoint, but well-known model families
        # (claude-*, gpt-*, gemini-*) still run via their inferred key.
        from benchflow.agents.registry import infer_env_key_for_model

        auth_env = infer_env_key_for_model(model)
        if not auth_env:
            results.append(
                CheckResult("provider", False, f"no registered provider for {model!r}")
            )
            return results

    if agent and auth_env and not env.get(auth_env):
        from benchflow.agents.env import check_subscription_auth

        if check_subscription_auth(agent, auth_env):
            results.append(
                CheckResult(
                    f"provider auth ({agent})",
                    True,
                    "skipped — host subscription login covers this model;"
                    " key/route/ping checks do not apply",
                    skipped=True,
                )
            )
            return results

    if auth_env:
        has = bool(env.get(auth_env))
        results.append(
            CheckResult(
                f"provider key ({auth_env})",
                has,
                "set" if has else f"{auth_env} is not set",
            )
        )
    # LiteLLM route resolution is pure (no network) and catches the
    # env/route errors the proxy would hit at run time — a doctor-green ping
    # alone does not prove the proxy lane resolves.
    try:
        from benchflow.providers.litellm_config import resolve_litellm_route

        route = resolve_litellm_route(model, env)
        results.append(
            CheckResult("litellm route", True, f"resolves (alias {route.model_alias})")
        )
    except Exception as exc:
        results.append(
            CheckResult(
                "litellm route", False, _sanitize(f"{type(exc).__name__}: {exc}")
            )
        )
    if not skip_ping:
        results.append(model_ping(model, env, transport=ping_transport))
    return results


def _run_args(prefs: dict, *, agent: str, model: str | None) -> list[str]:
    args = ["bench", "eval", "run", "--agent", agent]
    if model:
        args += ["--model", model]
    dataset = prefs["dataset"]
    if "/" in dataset or dataset.startswith("."):
        args += ["--tasks-dir", dataset]
    else:
        args += ["-d", dataset]
    args += ["--sandbox", prefs["sandbox"]]
    if prefs.get("skill_mode"):
        args += ["--skill-mode", prefs["skill_mode"]]
    return args


def final_command(prefs: dict) -> str:
    """The ready-to-run command the wizard prints (and copies) at the end."""
    return shell_join(_run_args(prefs, agent=prefs["agent"], model=prefs["model"]))


def smoke_argv(prefs: dict, task: str) -> list[str]:
    """Stage-1 smoke: the credential-free oracle agent on ONE task — proves
    install + sandbox plumbing before any API key is involved."""
    return [*_run_args(prefs, agent="oracle", model=None), "--include", task]


def detect_key_sources(
    auth_env: str, agent: str | None = None, cwd: str | Path | None = None
) -> list[tuple[str, str | None]]:
    """Every credential source found for *auth_env*, in run-path order.

    Order matches what the RUN would actually use (resolve_agent_env
    inherits a local .env first, then the process environment, and only uses
    host subscription auth when no key source is present): ``./.env`` in the
    working folder, then the process environment (which already includes the
    saved ~/.benchflow/.env via startup autoload), then host subscription
    login (needs *agent*; value None — nothing to store). The wizard's auth
    menu shows all of them; the non-interactive path takes the first.
    """
    import os

    sources: list[tuple[str, str | None]] = []
    value = read_env_file(Path(cwd or ".") / ".env").get(auth_env)
    if value:
        sources.append(("./.env", value))
    if os.environ.get(auth_env):
        sources.append(("environment", os.environ[auth_env]))
    if agent:
        from benchflow.agents.env import check_subscription_auth

        if check_subscription_auth(agent, auth_env):
            sources.append(("subscription", None))
    return sources


def detect_key(
    auth_env: str, agent: str | None = None, cwd: str | Path | None = None
) -> tuple[str | None, str | None]:
    """First (run-path-preferred) credential source, or ``(None, None)``."""
    sources = detect_key_sources(auth_env, agent=agent, cwd=cwd)
    return sources[0] if sources else (None, None)


def dataset_choices() -> list[tuple[str, str]]:
    """(spec, description) picker entries from the dataset registry, newest
    version first within each name; empty when the registry is unreachable
    (the wizard then falls back to free-form entry)."""
    from benchflow._utils import dataset_registry as dr

    try:
        entries = [
            e
            for e in dr.load_registry(dr.DEFAULT_REGISTRY_SOURCE)
            if isinstance(e, dict)
        ]
    except Exception:
        return []

    def _vkey(e):
        try:
            return tuple(int(p) for p in str(e.get("version", "")).split("."))
        except ValueError:
            return ()

    ordered = sorted(entries, key=_vkey, reverse=True)
    ordered.sort(key=lambda e: str(e.get("name", "")))
    import textwrap

    return [
        (
            f"{e['name']}@{e['version']}",
            textwrap.shorten(str(e.get("description", "")), width=80, placeholder="…"),
        )
        for e in ordered
        if e.get("name") and e.get("version")
    ]


def compatible_providers(agent: str) -> list[str]:
    """Registered provider names the chosen agent can route — the same
    protocol gate as compatible_agents, applied in the other direction for
    the agent-first wizard flow."""
    from benchflow.agents.env import _provider_supports_agent_protocol
    from benchflow.agents.providers import PROVIDERS
    from benchflow.agents.registry import AGENTS

    cfg = AGENTS.get(agent)
    protocol = (cfg.api_protocol or "") if cfg else ""
    return [
        name
        for name in sorted(PROVIDERS)
        if _provider_supports_agent_protocol(PROVIDERS[name], protocol)
    ]


# The ACP catalog names in benchflow-ai/agents (acp/<name>/manifest.toml).
# STATIC on purpose: browsing must cost zero network; selecting an entry
# lazily fetches only that agent's manifest (remote_manifests.fetch_one). A
# stale entry fails its fetch with a clear message — rot is loud, not silent.
CATALOG_AGENTS: tuple[str, ...] = (
    "amp-acp",
    "auggie",
    "autohand",
    "cline",
    "codebuddy-code",
    "cortex-code",
    "corust-agent",
    "crow-cli",
    "dimcode",
    "dirac",
    "factory-droid",
    "fast-agent",
    "github-copilot-cli",
    "glm-acp-agent",
    "goose",
    "grok-build",
    "junie",
    "kilo",
    "mimo-acp",
    "minion-code",
    "mistral-vibe",
    "nova",
    "poolside",
    "qoder",
    "qwen-code",
    "sigit",
    "stakpak",
    "vtcode",
)


# Package-path agents (pip-installed, not manifest-fetchable). Static like
# CATALOG_AGENTS; selecting one that is not installed yields the exact
# install command (install_hint) instead of a dead end.
AI_SDK_AGENTS: tuple[str, ...] = (
    "ai-sdk",
    "ai-sdk-pi",
    "ai-sdk-codex",
    "ai-sdk-claude-code",
    "ai-sdk-deepagents",
    "ai-sdk-opencode",
)
OMNIGENT_AGENTS: tuple[str, ...] = (
    "omnigent-pi",
    "omnigent-claude",
    "omnigent-openai-agents",
    "omnigent-codex",
    "omnigent-goose",
    "omnigent-qwen",
    "omnigent-kimi",
    "omnigent-hermes",
    "omnigent-copilot",
    "omnigent-cursor",
    "omnigent-antigravity",
    "omnigent-claude-native",
    "omnigent-codex-native",
    "omnigent-goose-native",
    "omnigent-qwen-native",
    "omnigent-kimi-native",
    "omnigent-hermes-native",
    "omnigent-cursor-native",
    "omnigent-antigravity-native",
    "omnigent-kiro-native",
    "omnigent-opencode-native",
    "omnigent-pi-native",
)
_AI_SDK_SUBDIRS = {
    "ai-sdk": "ai-sdk/acp",
    "ai-sdk-pi": "ai-sdk/harness-pi",
    "ai-sdk-codex": "ai-sdk/harness-codex",
    "ai-sdk-claude-code": "ai-sdk/harness-claude-code",
    "ai-sdk-deepagents": "ai-sdk/harness-deepagents",
    "ai-sdk-opencode": "ai-sdk/harness-opencode",
}


def path_choices(path: str) -> list[tuple[str, str]]:
    """Static (name, desc) entries for one adaptation path's browse menu."""
    from benchflow.agents.registry import AGENTS

    names = {
        "acp": CATALOG_AGENTS,
        "ai-sdk": AI_SDK_AGENTS,
        "omnigent": OMNIGENT_AGENTS,
    }[path]
    return [(n, "" if n not in AGENTS else "installed") for n in names]


def install_hint(name: str) -> str | None:
    """The exact install command for a package-path agent, or None for
    manifest-path (acp) agents."""
    base = "https://github.com/benchflow-ai/agents@main"
    if name.startswith("omnigent-"):
        return f'uv pip install "omnigent-benchflow @ git+{base}#subdirectory=omnigent"'
    sub = _AI_SDK_SUBDIRS.get(name)
    if sub:
        return f'uv pip install "{name} @ git+{base}#subdirectory={sub}"'
    return None


def catalog_choices() -> list[tuple[str, str]]:
    """Static (name, description) entries for the wizard's catalog browse —
    ACP catalog agents not already registered locally. Zero network."""
    from benchflow.agents.registry import AGENTS

    return [(n, "") for n in CATALOG_AGENTS if n not in AGENTS]


def acp_agents() -> list[str]:
    """Locally-registered ACP-native agents for the wizard menu — NO network.

    Only what is already in the registry (core built-ins + any installed
    plugin packages), minus the other adaptation paths (ai-sdk-* / omnigent-*)
    which the wizard does not list. The full benchflow-ai/agents manifest
    catalog is fetched lazily — only when an agent is actually resolved for a
    run (resolve_agent's miss path), not to populate a menu — so a bare
    ``bench init`` never clones the repo. Names not shown here stay reachable
    via ``--agent <name>`` and the menu's "other" escape, both of which
    resolve (and lazily fetch) on demand.
    """
    return [
        name
        for name in compatible_agents()
        if not (
            name == "ai-sdk"
            or name.startswith("ai-sdk-")
            or name.startswith("omnigent-")
        )
    ]
