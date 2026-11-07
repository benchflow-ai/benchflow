"""Environment values as the agent receives them: strings.

A YAML or JSON config can write ``MAX_THINKING_TOKENS: 32000`` or
``MY_FLAG: true``; the loader gets an int or a bool. Nothing past the config
handles those: the sandbox's env file ``shlex.quote``s each value (a TypeError
on a number, and ``''`` for 0 or false), and the redaction and resume helpers
match strings. So every config loader turns them into the string a shell user
would write, once, and the rest of BenchFlow only ever sees strings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def env_value_text(value: object) -> str:
    """*value* as the string an env var holds: ``true``/``false`` for a bool
    (as YAML spells it), ``str()`` for a number, a string as it is, and
    ``str()`` for anything else (never raises; callers that must refuse
    those use :func:`normalize_env_mapping`)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else str(value)


def normalize_env_mapping(env: Any, what: str = "agent_env") -> dict[str, str]:
    """*env* with every value a string (:func:`env_value_text`).

    A number or a bool is written as the string; a null, a list or a mapping
    has no env-var spelling and is refused with the name, rather than
    reaching the sandbox as ``''`` or a crash.
    """
    if env is None:
        return {}
    if not isinstance(env, Mapping):
        raise ValueError(
            f"{what} must be a mapping of names to values, got {type(env).__name__}"
        )
    out: dict[str, str] = {}
    for name, value in env.items():
        if value is None or not isinstance(value, (str, int, float)):
            shown = "null" if value is None else type(value).__name__
            raise ValueError(
                f"{what} {name} is {shown}: an env value must be a string, a "
                f'number or true/false (write {name}: "" for an empty value)'
            )
        out[str(name)] = env_value_text(value)
    return out
