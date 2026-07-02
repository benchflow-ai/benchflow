"""Onboarding logic behind ``bench init`` / ``bench doctor``.

Pure functions the CLI wizard is thin glue over: a private env file for
credentials (secrets live in env vars, per the provider registry's
``auth_env`` contract), TOML preferences, provider/agent compatibility
filtering, and the health checks the post-init smoke test and ``bench
doctor`` share.
"""

from __future__ import annotations

from pathlib import Path


def write_env_file(path: str | Path, updates: dict[str, str]) -> None:
    """Merge *updates* into the env file at *path* (created 0600).

    Existing keys not in *updates* are preserved; the file never widens its
    permissions once created.
    """
    path = Path(path)
    merged = read_env_file(path)
    merged.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f'{k}="{v}"\n' for k, v in merged.items())
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(body)


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse a KEY="value" env file; missing file reads as empty."""
    path = Path(path)
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        values[key.strip()] = raw.strip().strip('"')
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


def resolve_provider(model: str):
    """Map a model id ("provider/model" or bare) to (provider_name, config).

    Returns None when no registered provider claims the model — the wizard
    treats that as "custom endpoint" territory rather than an error.
    """
    from benchflow.agents.providers import PROVIDERS, find_provider_for_bare_model

    if "/" in model:
        prefix = model.split("/", 1)[0]
        cfg = PROVIDERS.get(prefix)
        return (prefix, cfg) if cfg else None
    return find_provider_for_bare_model(model)


def compatible_agents(model: str) -> list[str]:
    """Registered agent names that can actually route *model*.

    Reuses the same provider-protocol gate the run path enforces
    (env._provider_supports_agent_protocol), so the wizard never offers an
    agent the run would immediately reject — e.g. anthropic-messages or
    openai-responses agents on an openai-completions-only provider. Agents
    that bypass provider routing entirely (oracle has no model; gemini speaks
    its provider's native wire) are never offered.
    """
    from benchflow.agents.env import _provider_supports_agent_protocol
    from benchflow.agents.registry import AGENTS

    non_routed = {"oracle", "gemini"}
    resolved = resolve_provider(model)
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
    """One doctor/smoke row: what was checked, verdict, human detail."""

    name: str
    ok: bool
    detail: str = ""


def model_ping(model: str, env: dict[str, str], transport=None) -> CheckResult:
    """Verify key + model id + endpoint with ONE max_tokens=1 completion.

    GET /models is not used: it can 200 while the actual route is broken
    (wrong model id, upstream 5xx) — a 1-token completion is the cheapest
    request that exercises the full path.
    """
    import httpx

    from benchflow.agents.providers import resolve_base_url, strip_provider_prefix

    resolved = resolve_provider(model)
    if not resolved:
        return CheckResult("model ping", False, f"no registered provider for {model!r}")
    prov_name, cfg = resolved
    name = f"model ping ({prov_name})"
    key = env.get(cfg.auth_env or "", "")
    if cfg.auth_type == "api_key" and not key:
        return CheckResult(name, False, f"{cfg.auth_env} is not set")
    try:
        base = resolve_base_url(cfg, env).rstrip("/")
    except KeyError as exc:
        return CheckResult(name, False, str(exc))
    url = f"{base}/chat/completions"
    if not base.endswith("/v1") and "/v1/" not in base:
        url = f"{base}/v1/chat/completions"
    payload = {
        "model": strip_provider_prefix(model),
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    try:
        with httpx.Client(transport=transport, timeout=30) as client:
            resp = client.post(
                url, json=payload, headers={"Authorization": f"Bearer {key}"}
            )
    except httpx.HTTPError as exc:
        return CheckResult(name, False, f"request failed: {exc}")
    if resp.status_code == 200:
        return CheckResult(name, True, f"1-token completion OK ({url})")
    return CheckResult(name, False, f"HTTP {resp.status_code}: {resp.text[:200]}")


def run_doctor(
    model: str,
    sandbox: str,
    env: dict[str, str],
    ping_transport=None,
    skip_ping: bool = False,
) -> list[CheckResult]:
    """The shared check set behind `bench doctor` and the init smoke stage."""
    import shutil

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
    resolved = resolve_provider(model)
    if resolved:
        _, cfg = resolved
        if cfg.auth_type == "api_key" and cfg.auth_env:
            has = bool(env.get(cfg.auth_env))
            results.append(
                CheckResult(
                    f"provider key ({cfg.auth_env})",
                    has,
                    "set" if has else f"{cfg.auth_env} is not set",
                )
            )
    else:
        results.append(
            CheckResult("provider", False, f"no registered provider for {model!r}")
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
    return " ".join(_run_args(prefs, agent=prefs["agent"], model=prefs["model"]))


def smoke_argv(prefs: dict, task: str) -> list[str]:
    """Stage-1 smoke: the credential-free oracle agent on ONE task — proves
    install + sandbox plumbing before any API key is involved."""
    return [*_run_args(prefs, agent="oracle", model=None), "--include", task]
