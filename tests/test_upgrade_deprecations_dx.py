"""Deprecated names warn when used (DF-p4-upgrader-6).

``RuntimeResult`` never warned, so ``isinstance(result, bf.RuntimeResult)``
turned False silently, and ``RuntimeConfig(snapshot_policy="none")`` did not
warn although the field does nothing.
"""

from __future__ import annotations

import warnings

# --- DF-6: deprecations warn when used ----------------------------------------


def _warned(fn) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if w.category is DeprecationWarning]


def test_runtime_result_warns_on_every_lookup() -> None:
    import benchflow as bf

    for _ in range(2):
        messages = _warned(lambda: bf.RuntimeResult)
        assert any("isinstance(result, RuntimeResult) is False" in m for m in messages)

    def from_import():
        from benchflow import RuntimeResult  # noqa: F401

    def from_module():
        from benchflow.runtime import RuntimeResult  # noqa: F401

    assert _warned(from_import) and _warned(from_module)
    assert "RuntimeResult" not in bf.__all__


def test_every_unused_runtime_config_field_warns_when_passed() -> None:
    import benchflow as bf

    for field, value in (("snapshot_policy", "none"), ("max_rounds", 10)):
        messages = _warned(lambda f=field, v=value: bf.RuntimeConfig(**{f: v}))
        assert any(f"RuntimeConfig.{field} is not used" in m for m in messages)
    assert not _warned(lambda: bf.RuntimeConfig())
    assert not _warned(lambda: bf.RuntimeConfig(timeout=900))
