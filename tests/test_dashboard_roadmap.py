"""Tests for the dashboard's Linear-backed roadmap feed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dashboard import generate
from dashboard.roadmap import (
    collect_roadmap,
    fetch_linear_roadmap,
    linear_issues_to_roadmap,
)


def test_linear_issues_group_under_linear_milestones_and_preserve_fields():
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    roadmap = linear_issues_to_roadmap(
        [
            {
                "identifier": "ENG-120",
                "title": "Contracts",
                "description": "Mirror the full Linear ticket detail into the dashboard.",
                "url": "https://linear.app/benchflow/issue/ENG-120/contracts",
                "priority": 2,
                "priorityLabel": "High",
                "createdAt": "2026-05-20T20:00:00.000Z",
                "updatedAt": "2026-05-21T20:00:00.000Z",
                "startedAt": "2026-05-21T12:00:00.000Z",
                "completedAt": None,
                "canceledAt": None,
                "state": {"name": "In Progress", "type": "started"},
                "projectMilestone": {
                    "id": "milestone-m2",
                    "name": "M2 · Four-plane contracts",
                    "sortOrder": 20.0,
                    "status": "next",
                    "progress": 0.25,
                    "targetDate": "2026-06-01",
                },
                "parent": {
                    "identifier": "ENG-117",
                    "title": "RFC",
                },
                "team": {"key": "ENG", "name": "Engineering"},
                "creator": {"name": "Grace Hopper"},
                "assignee": {"name": "Ada Lovelace"},
                "estimate": 3,
                "dueDate": "2026-06-02",
                "branchName": "codex/eng-120-contracts",
                "labels": {"nodes": [{"name": "architecture", "color": "#2b6cb0"}]},
                "comments": {
                    "nodes": [
                        {
                            "id": "comment-1",
                            "body": "Implementation note from Linear.",
                            "createdAt": "2026-05-21T21:00:00.000Z",
                            "updatedAt": "2026-05-21T21:30:00.000Z",
                            "user": {"name": "Yimin"},
                        }
                    ],
                    "pageInfo": {"hasNextPage": True},
                },
            },
            {
                "identifier": "ENG-117",
                "title": "RFC",
                "url": "https://linear.app/benchflow/issue/ENG-117/rfc",
                "priority": 1,
                "priorityLabel": "Urgent",
                "state": {"name": "In Review", "type": "started"},
                "projectMilestone": {
                    "id": "milestone-m1",
                    "name": "M1 · Cut dead architecture",
                    "sortOrder": 10.0,
                    "status": "current",
                    "progress": 0.5,
                },
            },
        ],
        source_kind="linear-live",
        project_metadata={
            "id": "project-1",
            "name": "BenchFlow v0.5 — architecture migration",
            "url": "https://linear.app/benchflow/project/v05",
            "progress": 0.4,
            "targetDate": "2026-06-15",
            "projectMilestones": {
                "nodes": [
                    {
                        "id": "milestone-m2",
                        "name": "M2 · Four-plane contracts",
                        "sortOrder": 20.0,
                        "status": "next",
                        "progress": 0.25,
                        "targetDate": "2026-06-01",
                    },
                    {
                        "id": "milestone-m1",
                        "name": "M1 · Cut dead architecture",
                        "sortOrder": 10.0,
                        "status": "current",
                        "progress": 0.5,
                    },
                ]
            },
        },
        generated_at="2026-05-21 20:00:00",
    )

    assert roadmap["available"] is True
    assert roadmap["source"]["kind"] == "linear-live"
    assert roadmap["project_url"] == "https://linear.app/benchflow/project/v05"
    assert roadmap["project_progress"] == 0.4
    assert roadmap["project_target_date"] == "2026-06-15"
    assert [m["id"] for m in roadmap["milestones"]] == ["M1", "M2"]
    assert "linear_id" not in roadmap["milestones"][0]
    assert roadmap["milestones"][0]["progress"] == 0.5
    issue = roadmap["milestones"][1]["issues"][0]
    assert issue["id"] == "ENG-120"
    assert issue["description"] == "Mirror the full Linear ticket detail into the dashboard."
    assert issue["url"] == "https://linear.app/benchflow/issue/ENG-120/contracts"
    assert issue["status_type"] == "started"
    assert issue["priority"] == {"value": 2, "name": "High"}
    assert issue["parent"] == "ENG-117"
    assert issue["assignee"] == "Ada Lovelace"
    assert issue["team"] == "ENG"
    assert issue["team_name"] == "Engineering"
    assert issue["creator"] == "Grace Hopper"
    assert issue["estimate"] == 3
    assert issue["due_date"] == "2026-06-02"
    assert issue["branch_name"] == "codex/eng-120-contracts"
    assert issue["labels"] == [{"name": "architecture", "color": "#2b6cb0"}]
    assert issue["comments"] == [
        {
            "id": "comment-1",
            "body": "Implementation note from Linear.",
            "created_at": "2026-05-21T21:00:00.000Z",
            "updated_at": "2026-05-21T21:30:00.000Z",
            "user": "Yimin",
        }
    ]
    assert issue["comments_truncated"] is True
    assert issue["started_at"] == "2026-05-21T12:00:00.000Z"
    assert "assignee_email" not in issue
    assert "parent_url" not in issue


def test_linear_issue_query_requests_ticket_body_and_comments():
    """Guards Tickets against title-only Linear mirrors."""
    from dashboard.roadmap import LINEAR_ISSUE_FIELDS

    assert "\ndescription\n" in LINEAR_ISSUE_FIELDS
    assert "comments(first: 10)" in LINEAR_ISSUE_FIELDS
    assert "body" in LINEAR_ISSUE_FIELDS
    assert "user { name }" in LINEAR_ISSUE_FIELDS


def test_collect_roadmap_requires_live_linear_key():
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    roadmap = collect_roadmap(env={})

    assert roadmap["available"] is False
    assert roadmap["source"]["kind"] == "unavailable"
    assert "LINEAR_API_KEY" in roadmap["source"]["error"]
    assert roadmap["milestones"] == []


def test_collect_roadmap_does_not_mask_live_linear_failure(monkeypatch):
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    def fail_fetch(_api_key):
        raise RuntimeError("network down")

    monkeypatch.setattr("dashboard.roadmap.fetch_linear_roadmap", fail_fetch)

    roadmap = collect_roadmap(env={"LINEAR_API_KEY": "lin_api_test"})
    assert roadmap["available"] is False
    assert roadmap["source"]["kind"] == "unavailable"
    assert roadmap["source"]["error"] == "live Linear refresh failed"
    assert roadmap["milestones"] == []


def test_collect_roadmap_sanitizes_unknown_linear_failures(monkeypatch):
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    def fail_fetch(_api_key):
        raise RuntimeError("token lin_api_secret failed against https://api.linear.app/graphql")

    monkeypatch.setattr("dashboard.roadmap.fetch_linear_roadmap", fail_fetch)

    roadmap = collect_roadmap(env={"LINEAR_API_KEY": "lin_api_test"})

    assert roadmap["available"] is False
    assert roadmap["source"]["error"] == "live Linear refresh failed"
    assert "lin_api" not in roadmap["source"]["error"]
    assert "api.linear.app" not in roadmap["source"]["error"]


def test_collect_roadmap_prefers_linear_project_id_env_over_name(monkeypatch):
    """Guards Linear sync against project renames by preferring LINEAR_PROJECT_ID."""
    calls = []

    def fake_fetch(api_key, **kwargs):
        calls.append((api_key, kwargs))
        return linear_issues_to_roadmap(
            [],
            source_kind="linear-live",
            project_metadata={"id": kwargs["project_id"], "name": "Renamed BenchFlow v0.5"},
        )

    monkeypatch.setattr("dashboard.roadmap.fetch_linear_roadmap", fake_fetch)

    roadmap = collect_roadmap(
        env={
            "LINEAR_API_KEY": "lin_api_test",
            "LINEAR_PROJECT_ID": "project-uuid",
            "LINEAR_PROJECT_URL": "https://linear.app/benchflow/project/stale-url",
            "LINEAR_PROJECT_NAME": "stale old name",
        }
    )

    assert roadmap["available"] is True
    assert calls == [("lin_api_test", {"project_id": "project-uuid"})]
    assert roadmap["project"] == "Renamed BenchFlow v0.5"


def test_collect_roadmap_uses_safe_linear_project_url_or_slug(monkeypatch):
    """Guards convenient Linear project config while keeping ID as the preferred selector."""
    calls = []

    def fake_fetch(api_key, **kwargs):
        calls.append((api_key, kwargs))
        return linear_issues_to_roadmap(
            [],
            source_kind="linear-live",
            project_metadata={"id": "project-from-selector", "name": "BenchFlow v0.5"},
        )

    monkeypatch.setattr("dashboard.roadmap.fetch_linear_roadmap", fake_fetch)

    collect_roadmap(
        env={
            "LINEAR_API_KEY": "lin_api_test",
            "LINEAR_PROJECT_URL": "https://linear.app/benchflow/project/benchflow-v05-architecture-migration/?utm=1#overview",
        }
    )
    collect_roadmap(
        env={
            "LINEAR_API_KEY": "lin_api_test",
            "LINEAR_PROJECT_SLUG": "benchflow-v05-from-slug",
        }
    )

    assert calls == [
        ("lin_api_test", {"project_slug": "benchflow-v05-architecture-migration"}),
        ("lin_api_test", {"project_slug": "benchflow-v05-from-slug"}),
    ]


def test_collect_roadmap_rejects_unsafe_linear_project_url(monkeypatch):
    """Guards the Linear mirror against unsafe or ambiguous project URLs."""
    calls = []

    def fake_fetch(api_key, **kwargs):
        calls.append((api_key, kwargs))
        raise AssertionError("unsafe URL should fail before fetch")

    monkeypatch.setattr("dashboard.roadmap.fetch_linear_roadmap", fake_fetch)

    roadmap = collect_roadmap(
        env={
            "LINEAR_API_KEY": "lin_api_test",
            "LINEAR_PROJECT_URL": "https://example.com/benchflow/project/not-linear",
        }
    )

    assert calls == []
    assert roadmap["available"] is False
    assert "LINEAR_PROJECT_URL" in roadmap["source"]["error"]


def test_fetch_linear_roadmap_pages_through_linear_graphql():
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    calls = []

    def fake_request(query, variables, api_key):
        calls.append((query, variables, api_key))
        if "DashboardRoadmapProject" in query:
            return {
                "data": {
                    "projects": {
                        "nodes": [
                            {
                                "id": "project-1",
                                "name": "BenchFlow v0.5 — architecture migration",
                                "url": "https://linear.app/project",
                                "progress": 0.75,
                                "projectMilestones": {
                                    "nodes": [
                                        {
                                            "id": "milestone-m1",
                                            "name": "M1 · Cut dead architecture",
                                            "sortOrder": 10.0,
                                            "status": "current",
                                            "progress": 0.5,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                }
            }
        if variables["after"] is None:
            return {
                "data": {
                    "issues": {
                        "nodes": [
                            {
                                "identifier": "ENG-117",
                                "title": "RFC",
                                "state": {"name": "In Review", "type": "started"},
                                "projectMilestone": {
                                    "id": "milestone-m1",
                                    "name": "M1 · Cut dead architecture",
                                },
                            }
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": "next"},
                    }
                }
            }
        return {
            "data": {
                "issues": {
                    "nodes": [
                        {
                            "identifier": "ENG-118",
                            "title": "Cut dead architecture",
                            "state": {"name": "Backlog", "type": "backlog"},
                            "projectMilestone": {
                                "id": "milestone-m1",
                                "name": "M1 · Cut dead architecture",
                            },
                        }
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }

    roadmap = fetch_linear_roadmap("lin_api_key", request_fn=fake_request)
    assert [call[1].get("after") for call in calls[1:]] == [None, "next"]
    assert {call[2] for call in calls} == {"lin_api_key"}
    assert roadmap["project_url"] == "https://linear.app/project"
    assert [i["id"] for i in roadmap["milestones"][0]["issues"]] == ["ENG-117", "ENG-118"]


def test_fetch_linear_roadmap_can_target_stable_project_id():
    """Guards the Linear project selection follow-up on v0.5-integration@ffef85d."""
    calls = []

    def fake_request(query, variables, api_key):
        calls.append((query, variables, api_key))
        assert "projectId" in variables
        assert "project" not in variables
        if "DashboardRoadmapProjectById" in query:
            return {
                "data": {
                    "project": {
                        "id": "project-uuid",
                        "name": "Renamed BenchFlow v0.5",
                        "url": "https://linear.app/project",
                        "progress": 0.5,
                        "projectMilestones": {"nodes": []},
                    }
                }
            }
        return {
            "data": {
                "issues": {
                    "nodes": [
                        {
                            "identifier": "ENG-125",
                            "title": "Keep Linear mirror stable after project rename",
                            "state": {"name": "Todo", "type": "unstarted"},
                        }
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }

    roadmap = fetch_linear_roadmap("lin_api_key", project_id="project-uuid", request_fn=fake_request)

    assert "DashboardRoadmapProjectById($projectId: String!" in calls[0][0]
    assert "DashboardRoadmapById($projectId: ID!" in calls[1][0]
    assert [call[1]["projectId"] for call in calls] == ["project-uuid", "project-uuid"]
    assert {call[2] for call in calls} == {"lin_api_key"}
    assert roadmap["project"] == "Renamed BenchFlow v0.5"
    assert roadmap["milestones"][0]["issues"][0]["id"] == "ENG-125"


def test_collect_roadmap_bad_project_id_fails_closed_without_name_fallback(monkeypatch):
    """Guards explicit Linear selectors against silent fallback to stale names."""
    calls = []

    def fake_request(query, variables, api_key):
        calls.append((query, variables, api_key))
        assert "DashboardRoadmapProjectById" in query
        assert variables == {"projectId": "missing-project"}
        return {"data": {"project": None}}

    monkeypatch.setattr("dashboard.roadmap._post_linear_graphql", fake_request)

    roadmap = collect_roadmap(
        env={
            "LINEAR_API_KEY": "lin_api_test",
            "LINEAR_PROJECT_ID": "missing-project",
            "LINEAR_PROJECT_NAME": "stale-but-valid-name",
        }
    )

    assert roadmap["available"] is False
    assert roadmap["source"]["kind"] == "unavailable"
    assert "Linear project not found: id missing-project" in roadmap["source"]["error"]
    assert [call[1] for call in calls] == [{"projectId": "missing-project"}]
    assert all("DashboardRoadmapProjectByName" not in call[0] for call in calls)


def test_linear_issue_labels_accept_connection_or_list_shapes():
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    roadmap = linear_issues_to_roadmap(
        [
            {
                "identifier": "ENG-126",
                "title": "Label shape",
                "state": {"name": "Todo", "type": "unstarted"},
                "labels": [{"name": "bug", "color": "#cc0000"}, {"name": "ui"}, "bad"],
            }
        ],
        source_kind="linear-live",
        generated_at="2026-05-21 20:00:00",
    )

    issue = roadmap["milestones"][0]["issues"][0]
    assert issue["labels"] == [{"name": "bug", "color": "#cc0000"}, {"name": "ui"}]


def test_linear_issue_labels_mark_truncated_connection_shape():
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    roadmap = linear_issues_to_roadmap(
        [
            {
                "identifier": "ENG-127",
                "title": "Many labels",
                "state": {"name": "Todo", "type": "unstarted"},
                "labels": {
                    "nodes": [{"name": "architecture", "color": "#2b6cb0"}],
                    "pageInfo": {"hasNextPage": True},
                },
            }
        ],
        source_kind="linear-live",
        generated_at="2026-05-21 20:00:00",
    )

    issue = roadmap["milestones"][0]["issues"][0]
    assert issue["labels"] == [{"name": "architecture", "color": "#2b6cb0"}]
    assert issue["labels_truncated"] is True


def test_fetch_linear_roadmap_filters_issues_by_resolved_project_id_after_name_lookup():
    """Guards Linear sync against project renames after resolving project metadata."""
    calls = []

    def fake_request(query, variables, api_key):
        calls.append((query, variables, api_key))
        if "DashboardRoadmapProjectByName" in query:
            return {
                "data": {
                    "projects": {
                        "nodes": [
                            {
                                "id": "project-uuid",
                                "name": "Canonical Linear Project",
                                "url": "https://linear.app/benchflow/project/canonical",
                                "projectMilestones": {"nodes": []},
                            }
                        ]
                    }
                }
            }
        assert "DashboardRoadmapById" in query
        return {
            "data": {
                "issues": {
                    "nodes": [],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }

    roadmap = fetch_linear_roadmap(
        "lin_api_test",
        project="stale configured name",
        request_fn=fake_request,
    )

    assert roadmap["project"] == "Canonical Linear Project"
    assert calls[0][1] == {"project": "stale configured name"}
    assert calls[1][1] == {"projectId": "project-uuid", "first": 250, "after": None}


def test_fetch_linear_roadmap_resolves_slug_before_fetching_issues_by_id():
    """Guards Linear URL/slug project selection without name fallback."""
    calls = []

    def fake_request(query, variables, api_key):
        calls.append((query, variables, api_key))
        if "DashboardRoadmapProjectBySlug" in query:
            return {
                "data": {
                    "projects": {
                        "nodes": [
                            {
                                "id": "project-from-slug",
                                "name": "BenchFlow v0.5",
                                "projectMilestones": {"nodes": []},
                            }
                        ]
                    }
                }
            }
        assert "DashboardRoadmapById" in query
        return {
            "data": {
                "issues": {
                    "nodes": [],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }

    fetch_linear_roadmap(
        "lin_api_test",
        project_slug="benchflow-v05",
        request_fn=fake_request,
    )

    assert calls[0][1] == {"projectSlug": "benchflow-v05"}
    assert calls[1][1] == {"projectId": "project-from-slug", "first": 250, "after": None}


def test_fetch_linear_roadmap_fails_cleanly_on_malformed_issue_payload():
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    def fake_request(query, variables, api_key):
        if "DashboardRoadmapProjectById" in query:
            return {
                "data": {
                    "project": {
                        "id": "project-uuid",
                        "name": "BenchFlow v0.5",
                        "projectMilestones": {"nodes": []},
                    }
                }
            }
        return {"data": {}}

    try:
        fetch_linear_roadmap("lin_api_test", project_id="project-uuid", request_fn=fake_request)
    except RuntimeError as exc:
        assert str(exc) == "Linear issue response missing issues data"
    else:
        raise AssertionError("fetch_linear_roadmap should reject malformed issue payloads")


def test_fetch_linear_roadmap_rejects_repeated_pagination_cursor():
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    def fake_request(query, variables, api_key):
        if "DashboardRoadmapProjectById" in query:
            return {
                "data": {
                    "project": {
                        "id": "project-uuid",
                        "name": "BenchFlow v0.5",
                        "projectMilestones": {"nodes": []},
                    }
                }
            }
        return {
            "data": {
                "issues": {
                    "nodes": [],
                    "pageInfo": {"hasNextPage": True, "endCursor": "same"},
                }
            }
        }

    try:
        fetch_linear_roadmap("lin_api_test", project_id="project-uuid", request_fn=fake_request)
    except RuntimeError as exc:
        assert str(exc) == "Linear issue pagination repeated cursor"
    else:
        raise AssertionError("fetch_linear_roadmap should reject repeated pagination cursors")


def test_fetch_linear_roadmap_rejects_next_page_without_cursor():
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    calls = []

    def fake_request(query, variables, api_key):
        calls.append((query, variables, api_key))
        if "DashboardRoadmapProjectById" in query:
            return {
                "data": {
                    "project": {
                        "id": "project-uuid",
                        "name": "BenchFlow v0.5",
                        "projectMilestones": {"nodes": []},
                    }
                }
            }
        return {
            "data": {
                "issues": {
                    "nodes": [],
                    "pageInfo": {"hasNextPage": True, "endCursor": None},
                }
            }
        }

    try:
        fetch_linear_roadmap("lin_api_test", project_id="project-uuid", request_fn=fake_request)
    except RuntimeError as exc:
        assert str(exc) == "Linear issue pagination missing cursor"
    else:
        raise AssertionError("fetch_linear_roadmap should reject missing pagination cursors")

    assert len(calls) == 2


def test_fetch_linear_roadmap_requires_resolved_project_id_before_issue_query():
    """Guards Linear URL/slug project selection against malformed project payloads."""
    calls = []

    def fake_request(query, variables, api_key):
        calls.append((query, variables, api_key))
        assert "DashboardRoadmapProjectBySlug" in query
        return {
            "data": {
                "projects": {
                    "nodes": [
                        {
                            "name": "BenchFlow v0.5",
                            "projectMilestones": {"nodes": []},
                        }
                    ]
                }
            }
        }

    try:
        fetch_linear_roadmap(
            "lin_api_test",
            project_slug="benchflow-v05",
            request_fn=fake_request,
        )
    except RuntimeError as exc:
        assert "Linear project missing id: slug benchflow-v05" in str(exc)
    else:
        raise AssertionError("fetch_linear_roadmap should require a resolved project id")

    assert len(calls) == 1


def test_build_data_has_no_hardcoded_roadmap_fallback(monkeypatch):
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    monkeypatch.setattr(
        generate,
        "collect_tests",
        lambda: {"summary": {"passed": 0, "failed": 0, "skipped": 0, "total": 0}},
    )
    monkeypatch.setattr(generate, "collect_jobs", lambda: {"total_tasks": 0, "groups": []})
    monkeypatch.setattr(generate, "collect_experiments", lambda: [])
    monkeypatch.setattr(
        generate,
        "collect_roadmap",
        lambda: {
            "available": False,
            "project": "BenchFlow v0.5 — architecture migration",
            "source": {"kind": "unavailable", "generated_at": "now", "issue_count": 0},
            "milestones": [],
        },
    )

    data = generate.build_data()
    assert data["summary"]["issues_total"] == 0
    assert data["summary"]["issues_active"] == 0
    assert data["roadmap"]["milestones"] == []


def test_main_refuses_non_live_roadmap_without_local_dev_flag(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    out = tmp_path / "data.json"
    monkeypatch.setattr(generate, "ROOT", tmp_path)
    monkeypatch.setattr(generate, "OUT", out)
    monkeypatch.setattr(
        generate,
        "build_data",
        lambda: {
            "summary": {
                "tests": {"passed": 0, "failed": 0, "skipped": 0, "total": 0},
                "capabilities_shipped": 0,
                "capabilities_total": 0,
            },
            "jobs": {"total_tasks": 0, "groups": []},
            "experiments": [],
            "roadmap": {
                "available": False,
                "source": {
                    "kind": "unavailable",
                    "generated_at": "now",
                    "issue_count": 0,
                    "error": "set LINEAR_API_KEY to mirror Linear",
                },
                "milestones": [],
            },
        },
    )
    monkeypatch.setattr(sys, "argv", ["generate.py"])

    assert generate.main() == 1
    assert not out.exists()
    assert "Roadmap must mirror live Linear" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["generate.py", "--allow-missing-linear"])
    assert generate.main() == 0
    assert json.loads(out.read_text())["roadmap"]["source"]["kind"] == "unavailable"


def test_main_requires_real_linear_key_without_static_fallback(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    """Guards the Linear roadmap sync follow-up on v0.5-integration@ffef85d."""
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    monkeypatch.setattr(generate, "ROOT", tmp_path)
    monkeypatch.setattr(generate, "OUT", tmp_path / "data.json")
    monkeypatch.setattr(
        generate,
        "collect_tests",
        lambda: {"summary": {"passed": 0, "failed": 0, "skipped": 0, "total": 0}},
    )
    monkeypatch.setattr(generate, "collect_jobs", lambda: {"total_tasks": 0, "groups": []})
    monkeypatch.setattr(generate, "collect_experiments", lambda: [])
    monkeypatch.setattr(generate, "collect_repo_status", lambda: {"available": True, "dirty": False})
    monkeypatch.setattr(sys, "argv", ["generate.py"])

    assert generate.main() == 1
    assert not generate.OUT.exists()
    err = capsys.readouterr().err
    assert "Roadmap must mirror live Linear" in err
    assert "LINEAR_API_KEY" in err
