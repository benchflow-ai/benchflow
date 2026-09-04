"""Trajectory viewer package.

Import surface preserved from the former single-module viewer: everything
tests and the CLI reached via ``benchflow.trajectories.viewer`` re-exports
here. See the sibling modules for the actual responsibilities.
"""

from .catalog import (
    _discover_rollouts,
    _resolve_browse_rollout,
    _rollout_summary,
    _runs_cap,
)
from .legacy import (
    _NO_TRAJECTORIES_HTML,
    _VIEWER_CSS,
    TRACE_IR_ENV,
    _confirm_bar_html,
    _inject_confirm_bar,
    _message_block,
    _page,
    _prompt_block,
    _render_acp_events,
    _result_block,
    _thought_block,
    _tool_accent_class,
    render_jsonl_file,
    render_rollout,
    render_turn,
)
from .payload import (
    _DIAGNOSTIC_KEYS_FALLBACK,
    _build_acp_payload,
    _diagnostic_keys,
    _is_acp_rollout_dir,
    _load_prompts,
    _load_result_json,
    _parse_jsonl,
    _safe_json,
    _tool_content_texts,
)
from .render import _render_acp_trajectory, _render_shell
from .server import serve
from .sources import (
    _HF_VIEWER_FILES,
    HfDatasetSource,
    LocalPathSource,
    ViewerSourceError,
    parse_source,
    resolve_hf_dataset,
)

__all__ = [
    # public surface
    "render_rollout",
    "render_turn",
    "render_jsonl_file",
    "serve",
    "parse_source",
    "resolve_hf_dataset",
    "LocalPathSource",
    "HfDatasetSource",
    "ViewerSourceError",
    "TRACE_IR_ENV",
    # internals reached by tests and siblings through the historical module
    "_DIAGNOSTIC_KEYS_FALLBACK",
    "_HF_VIEWER_FILES",
    "_NO_TRAJECTORIES_HTML",
    "_VIEWER_CSS",
    "_build_acp_payload",
    "_confirm_bar_html",
    "_diagnostic_keys",
    "_discover_rollouts",
    "_inject_confirm_bar",
    "_is_acp_rollout_dir",
    "_load_prompts",
    "_load_result_json",
    "_message_block",
    "_page",
    "_parse_jsonl",
    "_prompt_block",
    "_render_acp_events",
    "_render_acp_trajectory",
    "_render_shell",
    "_resolve_browse_rollout",
    "_result_block",
    "_rollout_summary",
    "_runs_cap",
    "_safe_json",
    "_thought_block",
    "_tool_accent_class",
    "_tool_content_texts",
]
