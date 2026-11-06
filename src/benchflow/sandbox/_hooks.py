"""Optional sandbox methods the rollout and hardening call when a backend has them."""

from __future__ import annotations

from typing import Any

__all__ = ["sandbox_hook"]


def sandbox_hook(env: Any, name: str) -> Any:
    """*env*'s method *name* when *env*'s class defines it, else None.

    Looked up on the class, so a stand-in or a mock sandbox, whose attributes
    answer any name, has none: the caller then does without it.
    """
    if not callable(getattr(type(env), name, None)):
        return None
    return getattr(env, name)
