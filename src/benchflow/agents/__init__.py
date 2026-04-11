"""benchflow agents — registry, providers, and ACP shims.

This package owns *what agents and providers exist* and how the SDK
talks to them. The SDK reads everything it needs from the two
registries below — adding a new agent or provider is a registry-only
change. ``tests/test_registry_invariants.py`` runs contract checks
against every entry; read it for the executable schema.

Files
-----
- ``registry.py``    ``AGENTS``, ``AgentConfig``, ``CredentialFile``,
                     ``SubscriptionAuth``. The "add a new agent" recipe
                     and the per-field rules live in the module
                     docstring.
- ``providers.py``   ``PROVIDERS``, ``ProviderConfig``. Custom +
                     native LLM providers, ``base_url`` / ``url_params``
                     resolution, ADC handling. The "add a new provider"
                     recipe lives in the module docstring.
- ``openclaw_acp_shim.py``  Standalone script (read at import time by
                     ``registry.py``) that wraps ``openclaw agent
                     --local`` as an ACP server over stdio. Needed
                     because openclaw's native ACP bridge requires a
                     gateway.
- ``user_agent.py``  Interactive stdin/stdout ACP agent for manual
                     debugging. **Backlog — not wired into SDK or CLI.**
                     Greppable but inert.

Nothing is re-exported from this ``__init__``: importers go through
``benchflow.agents.registry`` / ``benchflow.agents.providers`` directly,
which is what the registry-only-change rule depends on.
"""
