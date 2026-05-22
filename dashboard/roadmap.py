"""Linear-backed roadmap data for the dashboard."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

LINEAR_PROJECT = "BenchFlow v0.5 — architecture migration"
LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

LINEAR_PROJECT_FIELDS = """
id
name
url
progress
targetDate
startedAt
completedAt
canceledAt
projectMilestones(first: 250) {
  nodes {
    id
    name
    sortOrder
    status
    progress
    targetDate
    createdAt
    updatedAt
  }
}
"""

LINEAR_PROJECT_BY_NAME_QUERY = (
    """
query DashboardRoadmapProjectByName($project: String!) {
  projects(filter: { name: { eq: $project } }, first: 1) {
    nodes {
"""
    + LINEAR_PROJECT_FIELDS
    + """
    }
  }
}
"""
)

LINEAR_PROJECT_BY_ID_QUERY = (
    """
query DashboardRoadmapProjectById($projectId: String!) {
  project(id: $projectId) {
"""
    + LINEAR_PROJECT_FIELDS
    + """
  }
}
"""
)

LINEAR_PROJECT_BY_SLUG_QUERY = (
    """
query DashboardRoadmapProjectBySlug($projectSlug: String!) {
  projects(filter: { slugId: { eq: $projectSlug } }, first: 1) {
    nodes {
"""
    + LINEAR_PROJECT_FIELDS
    + """
    }
  }
}
"""
)

LINEAR_ISSUE_FIELDS = """
identifier
title
description
url
priority
priorityLabel
createdAt
updatedAt
completedAt
startedAt
canceledAt
state { name type }
projectMilestone {
  id
  name
  sortOrder
  status
  progress
  targetDate
}
parent { identifier title }
team { key name }
creator { name }
assignee { name }
estimate
dueDate
branchName
labels(first: 20) { nodes { name color } pageInfo { hasNextPage } }
comments(first: 10) {
  nodes {
    id
    body
    createdAt
    updatedAt
    user { name }
  }
  pageInfo { hasNextPage }
}
"""

LINEAR_ROADMAP_BY_ID_QUERY = (
    """
query DashboardRoadmapById($projectId: ID!, $first: Int!, $after: String) {
  issues(
    filter: { project: { id: { eq: $projectId } } }
    first: $first
    after: $after
  ) {
    nodes {
"""
    + LINEAR_ISSUE_FIELDS
    + """
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""
)


@dataclass(frozen=True)
class LinearProjectSelector:
    kind: str
    value: str

    @property
    def variable_name(self) -> str:
        return {
            "id": "projectId",
            "slug": "projectSlug",
            "name": "project",
        }[self.kind]

    def variables(self) -> dict[str, str]:
        return {self.variable_name: self.value}


def _project_selector(
    *,
    project: str | None = None,
    project_id: str | None = None,
    project_slug: str | None = None,
) -> LinearProjectSelector:
    if project_id:
        return LinearProjectSelector(kind="id", value=project_id)
    if project_slug:
        return LinearProjectSelector(kind="slug", value=project_slug)
    return LinearProjectSelector(kind="name", value=project or LINEAR_PROJECT)


def _linear_project_query(selector: LinearProjectSelector) -> str:
    if selector.kind == "id":
        return LINEAR_PROJECT_BY_ID_QUERY
    if selector.kind == "slug":
        return LINEAR_PROJECT_BY_SLUG_QUERY
    return LINEAR_PROJECT_BY_NAME_QUERY


def _linear_project_from_payload(
    selector: LinearProjectSelector,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    if selector.kind == "id":
        return (payload.get("data") or {}).get("project")
    return next(
        iter(((payload.get("data") or {}).get("projects") or {}).get("nodes") or []),
        None,
    )


def _linear_project_slug_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "linear.app"
        or len(parts) != 3
        or parts[1] != "project"
        or not parts[0]
        or not parts[2]
    ):
        raise ValueError(
            "LINEAR_PROJECT_URL must look like https://linear.app/<workspace>/project/<slug>"
        )
    return parts[2]


def _project_kwargs_from_env(
    environ: dict[str, str] | os._Environ[str],
) -> dict[str, str]:
    project_id = (environ.get("LINEAR_PROJECT_ID") or "").strip()
    if project_id:
        return {"project_id": project_id}
    project_url = (environ.get("LINEAR_PROJECT_URL") or "").strip()
    if project_url:
        return {"project_slug": _linear_project_slug_from_url(project_url)}
    project_slug = (environ.get("LINEAR_PROJECT_SLUG") or "").strip()
    if project_slug:
        return {"project_slug": project_slug}
    project_name = (
        environ.get("LINEAR_PROJECT_NAME") or environ.get("LINEAR_PROJECT") or ""
    ).strip()
    if project_name and project_name != LINEAR_PROJECT:
        return {"project": project_name}
    return {}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _issue_number(identifier: str) -> int:
    with suppress(Exception):
        return int(identifier.rsplit("-", 1)[1])
    return 999999


def _priority_name(value: Any, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return {
        0: "No priority",
        1: "Urgent",
        2: "High",
        3: "Medium",
        4: "Low",
    }.get(value)


def _milestone_parts(raw_name: str | None) -> tuple[str, str, int]:
    if not raw_name:
        return ("Queue", "Queue / unmilestoned", 1_000_000)
    match = re.match(r"^(M\d+)\s*[·:-]\s*(.+)$", raw_name)
    if not match:
        return (raw_name, raw_name, 800)
    ident, name = match.groups()
    return (ident, name, int(ident[1:]))


def _drop_empty_values(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v not in (None, "", {}, [])}


def _linear_graphql_error(payload: dict[str, Any], fallback: str) -> str | None:
    errors = payload.get("errors")
    if not errors:
        return None
    first = errors[0] if isinstance(errors, list) and errors else {}
    if isinstance(first, dict) and first.get("message"):
        return str(first["message"])
    return fallback


def _linear_issue_page(payload: dict[str, Any]) -> dict[str, Any]:
    page = (payload.get("data") or {}).get("issues")
    if not isinstance(page, dict):
        raise RuntimeError("Linear issue response missing issues data")
    if not isinstance(page.get("nodes"), list):
        raise RuntimeError("Linear issue response missing issue nodes")
    if not isinstance(page.get("pageInfo"), dict):
        raise RuntimeError("Linear issue response missing pagination data")
    return page


def _linear_labels(raw: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(raw, dict):
        nodes = raw.get("nodes") or []
        page_info = raw.get("pageInfo") or {}
        truncated = (
            bool(page_info.get("hasNextPage")) if isinstance(page_info, dict) else False
        )
    elif isinstance(raw, list):
        nodes = raw
        truncated = False
    else:
        nodes = []
        truncated = False
    labels = [
        _drop_empty_values({"name": label.get("name"), "color": label.get("color")})
        for label in nodes
        if isinstance(label, dict)
    ]
    return labels, truncated


def _linear_comments(raw: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(raw, dict):
        nodes = raw.get("nodes") or []
        page_info = raw.get("pageInfo") or {}
        truncated = (
            bool(page_info.get("hasNextPage")) if isinstance(page_info, dict) else False
        )
    elif isinstance(raw, list):
        nodes = raw
        truncated = False
    else:
        nodes = []
        truncated = False

    comments = []
    for comment in nodes:
        if not isinstance(comment, dict) or not comment.get("body"):
            continue
        user = comment.get("user") or {}
        comments.append(
            _drop_empty_values(
                {
                    "id": comment.get("id"),
                    "body": comment.get("body"),
                    "created_at": comment.get("createdAt") or comment.get("created_at"),
                    "updated_at": comment.get("updatedAt") or comment.get("updated_at"),
                    "user": user.get("name") if isinstance(user, dict) else None,
                }
            )
        )
    return comments, truncated


def _linear_issue_to_dashboard(issue: dict[str, Any]) -> dict[str, Any]:
    state = issue.get("state") or {}
    milestone = issue.get("projectMilestone") or {}
    parent = issue.get("parent") or {}
    assignee = issue.get("assignee") or {}
    team = issue.get("team") or {}
    creator = issue.get("creator") or {}
    priority_value = issue.get("priority")
    labels, labels_truncated = _linear_labels(issue.get("labels"))
    comments, comments_truncated = _linear_comments(issue.get("comments"))
    identifier = issue.get("identifier") or issue.get("id") or "UNKNOWN"
    out = {
        "id": identifier,
        "title": issue.get("title") or "",
        "description": issue.get("description"),
        "url": issue.get("url"),
        "status": state.get("name") or issue.get("status") or "Unknown",
        "status_type": state.get("type") or issue.get("status_type") or "unknown",
        "updated_at": issue.get("updatedAt") or issue.get("updated_at"),
        "created_at": issue.get("createdAt") or issue.get("created_at"),
        "started_at": issue.get("startedAt") or issue.get("started_at"),
        "completed_at": issue.get("completedAt") or issue.get("completed_at"),
        "canceled_at": issue.get("canceledAt") or issue.get("canceled_at"),
        "parent": parent.get("identifier") or issue.get("parent"),
        "parent_title": parent.get("title"),
        "assignee": assignee.get("name"),
        "team": team.get("key") or team.get("name"),
        "team_name": team.get("name"),
        "creator": creator.get("name"),
        "estimate": issue.get("estimate"),
        "due_date": issue.get("dueDate") or issue.get("due_date"),
        "branch_name": issue.get("branchName") or issue.get("branch_name"),
        "labels": labels,
        "labels_truncated": labels_truncated or None,
        "comments": comments,
        "comments_truncated": comments_truncated or None,
        "milestone": _drop_empty_values(
            {
                "linear_id": milestone.get("id"),
                "name": milestone.get("name") or issue.get("milestone"),
                "sort_order": milestone.get("sortOrder"),
                "status": milestone.get("status"),
                "progress": milestone.get("progress"),
                "target_date": milestone.get("targetDate"),
            }
        ),
    }
    if priority_value is not None:
        out["priority"] = {
            "value": priority_value,
            "name": _priority_name(priority_value, issue.get("priorityLabel")),
        }
    return _drop_empty_values(out)


def _linear_milestone_to_dashboard(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    display_id, display_name, fallback_sort = _milestone_parts(raw.get("name"))
    sort_order = raw.get("sortOrder", raw.get("sort_order"))
    milestone = _drop_empty_values(
        {
            "id": display_id,
            "name": display_name,
            "linear_id": raw.get("id") or raw.get("linear_id"),
            "status": raw.get("status"),
            "progress": raw.get("progress"),
            "target_date": raw.get("targetDate") or raw.get("target_date"),
            "created_at": raw.get("createdAt") or raw.get("created_at"),
            "updated_at": raw.get("updatedAt") or raw.get("updated_at"),
            "_sort": sort_order if sort_order is not None else fallback_sort,
        }
    )
    milestone["issues"] = []
    return milestone


def _milestone_group_key(milestone: dict[str, Any]) -> str:
    return milestone.get("linear_id") or milestone.get("name") or "Queue"


def _normalize_linear_project(project: dict[str, Any] | None) -> dict[str, Any]:
    project = project or {"name": LINEAR_PROJECT}
    milestones = (project.get("projectMilestones") or {}).get("nodes") or []
    return _drop_empty_values(
        {
            "id": project.get("id"),
            "name": project.get("name") or LINEAR_PROJECT,
            "url": project.get("url"),
            "progress": project.get("progress"),
            "target_date": project.get("targetDate"),
            "started_at": project.get("startedAt"),
            "completed_at": project.get("completedAt"),
            "canceled_at": project.get("canceledAt"),
            "milestones": [_linear_milestone_to_dashboard(m) for m in milestones],
        }
    )


def linear_issues_to_roadmap(
    issues: list[dict[str, Any]],
    *,
    source_kind: str,
    project_metadata: dict[str, Any] | None = None,
    generated_at: str | None = None,
    source_error: str | None = None,
) -> dict[str, Any]:
    """Convert Linear issue nodes into the dashboard roadmap contract."""
    project_info = _normalize_linear_project(project_metadata)
    grouped: dict[str, dict[str, Any]] = {}
    for milestone in project_info.get("milestones", []):
        grouped[_milestone_group_key(milestone)] = milestone

    for raw in issues:
        issue = _linear_issue_to_dashboard(raw)
        raw_milestone = issue.pop("milestone", {})
        milestone = _linear_milestone_to_dashboard(raw_milestone)
        group = grouped.setdefault(_milestone_group_key(milestone), milestone)
        group["issues"].append(issue)

    milestones = sorted(grouped.values(), key=lambda item: (item["_sort"], item["id"]))
    for milestone in milestones:
        milestone["issues"].sort(key=lambda issue: _issue_number(issue["id"]))
        milestone.pop("linear_id", None)
        del milestone["_sort"]

    source = {
        "kind": source_kind,
        "generated_at": generated_at or _now(),
        "issue_count": sum(len(m["issues"]) for m in milestones),
    }
    if source_error:
        source["error"] = source_error
    return {
        "available": True,
        "project": project_info.get("name") or LINEAR_PROJECT,
        "project_url": project_info.get("url"),
        "project_progress": project_info.get("progress"),
        "project_target_date": project_info.get("target_date"),
        "source": source,
        "milestones": milestones,
    }


def _post_linear_graphql(
    query: str,
    variables: dict[str, Any],
    api_key: str,
    timeout: float = 20.0,
) -> dict[str, Any]:
    req = urllib.request.Request(
        LINEAR_GRAPHQL_URL,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_linear_roadmap(
    api_key: str,
    *,
    project: str = LINEAR_PROJECT,
    project_id: str | None = None,
    project_slug: str | None = None,
    request_fn: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch the roadmap directly from Linear's GraphQL API."""
    request = request_fn or _post_linear_graphql
    selector = _project_selector(
        project=project, project_id=project_id, project_slug=project_slug
    )
    project_payload = request(
        _linear_project_query(selector), selector.variables(), api_key
    )
    if error := _linear_graphql_error(project_payload, "Linear project query failed"):
        raise RuntimeError(error)
    project_metadata = _linear_project_from_payload(selector, project_payload)
    if not project_metadata:
        raise RuntimeError(
            f"Linear project not found: {selector.kind} {selector.value}"
        )
    resolved_project_id = project_metadata.get("id")
    if not resolved_project_id:
        raise RuntimeError(
            f"Linear project missing id: {selector.kind} {selector.value}"
        )

    issues: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        payload = request(
            LINEAR_ROADMAP_BY_ID_QUERY,
            {"projectId": resolved_project_id, "first": 250, "after": cursor},
            api_key,
        )
        if error := _linear_graphql_error(payload, "Linear query failed"):
            raise RuntimeError(error)
        page = _linear_issue_page(payload)
        issues.extend(page["nodes"])
        info = page.get("pageInfo") or {}
        if not info.get("hasNextPage"):
            break
        next_cursor = info.get("endCursor")
        if not next_cursor:
            raise RuntimeError("Linear issue pagination missing cursor")
        if next_cursor in seen_cursors:
            raise RuntimeError("Linear issue pagination repeated cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return linear_issues_to_roadmap(
        issues,
        source_kind="linear-live",
        project_metadata=project_metadata,
    )


def _unavailable_roadmap(error: str) -> dict[str, Any]:
    return {
        "available": False,
        "project": LINEAR_PROJECT,
        "source": {
            "kind": "unavailable",
            "generated_at": _now(),
            "issue_count": 0,
            "error": error,
        },
        "milestones": [],
    }


def collect_roadmap(
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return live Linear roadmap data, or a marked unavailable state."""
    environ = os.environ if env is None else env
    api_key = environ.get("LINEAR_API_KEY")
    if not api_key:
        return _unavailable_roadmap("set LINEAR_API_KEY to mirror Linear")
    try:
        return fetch_linear_roadmap(api_key, **_project_kwargs_from_env(environ))
    except Exception as exc:
        detail = str(exc)
        allowed_prefixes = (
            "LINEAR_PROJECT_URL",
            "Linear project not found",
            "Linear project missing id",
            "Linear issue response",
            "Linear issue pagination",
        )
        if detail.startswith(allowed_prefixes):
            return _unavailable_roadmap(f"live Linear refresh failed: {detail}")
        return _unavailable_roadmap("live Linear refresh failed")
