"""Unified ``task.md`` authoring document support.

The runtime still consumes the stable ``TaskConfig`` and instruction string.
This module owns the document-shaped authoring layer so ``task/config.py`` does
not become the home for prompt, role, scene, and simulated-user syntax.

The implementation is split across three state-free layers — a profile-preset
data table, a filesystem/JSON evidence-discovery subsystem, and the
markdown/frontmatter parse+normalize core — under ``_document_*`` submodules.
This module is a thin façade that re-exports every public and underscore symbol
those layers define, so ``benchflow.task.document`` stays import-compatible. The
unused names below are intentional façade re-exports, not dead imports.
"""

from __future__ import annotations

from benchflow.task._document_evidence import (
    _acceptance_live_evidence_from_report,  # noqa: F401
    _apply_conventional_evidence,  # noqa: F401
    _calibration_evidence_from_report,  # noqa: F401
    _discover_conventional_evidence,  # noqa: F401
    _first_number,  # noqa: F401
    _max_case_reward,  # noqa: F401
    _pin_existing_files,  # noqa: F401
    _read_json,  # noqa: F401
)
from benchflow.task._document_normalize import (
    TaskDocumentParseError,
    _apply_image_shorthand,  # noqa: F401
    _apply_name_shorthand,  # noqa: F401
    _apply_path_shorthand,  # noqa: F401
    _deep_merge,  # noqa: F401
    _ensure_mapping,  # noqa: F401
    _has_nested,  # noqa: F401
    _mapping,  # noqa: F401
    _merge_missing,  # noqa: F401
    _parse_authoring_profiles,  # noqa: F401
    _pop_path_shorthand,  # noqa: F401
    _record_applied_profiles,  # noqa: F401
    _safe_relative_posix_path,  # noqa: F401
    normalize_task_document_frontmatter,
)
from benchflow.task._document_parse import (
    _DOCUMENT_ONLY_FRONTMATTER_KEYS,  # noqa: F401
    _SECTION_RE,  # noqa: F401
    TASK_DOCUMENT_FILENAME,
    TaskDocument,
    _config_from_frontmatter,  # noqa: F401
    _escape_reserved_section_headings,  # noqa: F401
    _extract_prompt_sections,  # noqa: F401
    _load_prompt_sidecars,  # noqa: F401
    _lookup_role,  # noqa: F401
    _normalize_section_key,  # noqa: F401
    _optional_int,  # noqa: F401
    _optional_str,  # noqa: F401
    _parse_roles,  # noqa: F401
    _parse_scenes,  # noqa: F401
    _parse_turns,  # noqa: F401
    _PromptSections,  # noqa: F401
    _scene_role_names,  # noqa: F401
    _split_frontmatter,  # noqa: F401
    _string_dict,  # noqa: F401
    _string_list,  # noqa: F401
    _unescape_reserved_section_headings,  # noqa: F401
    render_normalized_task_md,
    render_task_md,  # noqa: F401
    render_task_md_from_legacy,
)
from benchflow.task._document_profiles import (
    _AUTHORING_ONLY_FRONTMATTER_KEYS,  # noqa: F401
    _PROFILE_KEYS,  # noqa: F401
    _TASK_AUTHORING_PROFILES,  # noqa: F401
)

__all__ = [
    "TASK_DOCUMENT_FILENAME",
    "TaskDocument",
    "TaskDocumentParseError",
    "normalize_task_document_frontmatter",
    "render_normalized_task_md",
    "render_task_md_from_legacy",
]
