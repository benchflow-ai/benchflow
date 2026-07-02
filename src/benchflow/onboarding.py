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
    """Parse a dotenv-style file; missing/unreadable file reads as empty.

    Tolerates the dialects people hand-paste: ``export KEY=...`` prefixes and
    single- or double-quoted values. Malformed lines (empty or whitespace
    keys) are skipped — never fatal, because the CLI auto-loads this file on
    every invocation, including the ones needed to repair it.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        import sys

        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, raw = line.partition("=")
        key = key.strip()
        if not key or any(c.isspace() for c in key):
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
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


def _sanitize(text: str) -> str:
    """Strip control characters so server-supplied text cannot inject
    terminal escape sequences through doctor/smoke output."""
    return "".join(ch for ch in text if ch.isprintable() or ch in " \t")


def model_ping(model: str, env: dict[str, str], transport=None) -> CheckResult:
    """Verify key + model id + endpoint with ONE max_tokens=1 completion.

    GET /models is not used: it can 200 while the actual route is broken
    (wrong model id, upstream 5xx) — a 1-token completion is the cheapest
    request that exercises the full path. The endpoint is joined exactly the
    way the run path does (resolve_base_url per protocol + one path segment);
    no URL guessing. Provider classes the ping cannot exercise (ADC, AWS
    SigV4) are skipped honestly instead of reported as failures.
    """
    import httpx

    from benchflow.agents.providers import resolve_base_url, strip_provider_prefix

    resolved = resolve_provider(model)
    if not resolved:
        return CheckResult("model ping", False, f"no registered provider for {model!r}")
    prov_name, cfg = resolved
    name = f"model ping ({prov_name})"
    if cfg.auth_type != "api_key":
        return CheckResult(
            name,
            True,
            f"skipped — {cfg.auth_type} auth is exercised at run time",
        )
    key = env.get(cfg.auth_env or "", "")
    if not key:
        return CheckResult(name, False, f"{cfg.auth_env} is not set")

    endpoints = cfg.all_endpoints
    bare_model = strip_provider_prefix(model)
    if "openai-completions" in endpoints:
        protocol = "openai-completions"
        path, ok_field = "/chat/completions", "choices"
        headers = {"Authorization": f"Bearer {key}"}
        payload = {
            "model": bare_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    elif "anthropic-messages" in endpoints:
        protocol = "anthropic-messages"
        path, ok_field = "/v1/messages", "content"
        # x-api-key is the Anthropic wire header; Azure's Anthropic surface
        # accepts api-key too — send both.
        headers = {"x-api-key": key, "api-key": key}
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
        )
    try:
        base = resolve_base_url(cfg, env, protocol=protocol).rstrip("/")
    except KeyError as exc:
        return CheckResult(name, False, _sanitize(str(exc)))
    url = f"{base}{path}"
    try:
        with httpx.Client(transport=transport, timeout=30) as client:
            resp = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        return CheckResult(name, False, _sanitize(f"request failed: {exc}"))
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
    else:
        from benchflow.sandbox.providers import SANDBOX_PROVIDERS

        results.append(
            CheckResult(
                "sandbox",
                sandbox in SANDBOX_PROVIDERS,
                f"unknown sandbox {sandbox!r} (expected one of"
                f" {', '.join(SANDBOX_PROVIDERS)})",
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
        # No registered provider endpoint, but well-known model families
        # (claude-*, gpt-*, gemini-*) still run via their inferred key — check
        # its presence instead of failing a setup the run path supports.
        from benchflow.agents.registry import infer_env_key_for_model

        inferred = infer_env_key_for_model(model)
        if inferred:
            has = bool(env.get(inferred))
            results.append(
                CheckResult(
                    f"provider key ({inferred})",
                    has,
                    "set" if has else f"{inferred} is not set",
                )
            )
        else:
            results.append(
                CheckResult("provider", False, f"no registered provider for {model!r}")
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
        results.append(CheckResult("litellm route", False, _sanitize(str(exc))))
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
