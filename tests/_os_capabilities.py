"""OS wrapper setup for regressions at frozen 7acea6f7d."""

import functools
import os


def patch_os(monkeypatch, name, wrapper):
    """Preserve only the capabilities advertised by the original callable."""
    original = getattr(os, name)
    functools.update_wrapper(wrapper, original)
    for attribute in (
        "supports_dir_fd",
        "supports_fd",
        "supports_follow_symlinks",
        "supports_effective_ids",
    ):
        supported = getattr(os, attribute, ())
        if original in supported:
            monkeypatch.setattr(os, attribute, {*supported, wrapper})
    monkeypatch.setattr(os, name, wrapper)
