"""Web UI routes — Gmail-like interface using Jinja2 + HTMX."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, Request, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from claw_gmail.models import User, Message, Thread, Label, MessageLabel, Draft
from claw_gmail.tasks import get_task, list_tasks
from claw_gmail.tasks.registry import _REGISTRY
from claw_gmail.state.snapshots import get_state_dump, get_diff
from claw_gmail.state.action_log import action_log
from claw_gmail.api.deps import get_db

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Project root for pytest and fixture discovery
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures" / "real_gmail"
_SPEC_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "gmail_api_spec.json"
_COVERAGE_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "mock_coverage.json"

# Module-level caches
_spec_cache = None
_coverage_cache = None


def _load_spec() -> dict:
    """Load Gmail API ground truth spec, cached."""
    global _spec_cache
    if _spec_cache is None:
        _spec_cache = json.loads(_SPEC_PATH.read_text())
    return _spec_cache


def _load_coverage() -> dict:
    """Load mock coverage mapping, cached."""
    global _coverage_cache
    if _coverage_cache is None:
        _coverage_cache = json.loads(_COVERAGE_PATH.read_text())
    return _coverage_cache


def _build_coverage_summary() -> dict:
    """Build per-resource coverage summary from spec and coverage data."""
    spec = _load_spec()
    coverage = _load_coverage()

    # Build lookup from endpoint ID to coverage entry
    cov_by_id = {ep["id"]: ep for ep in coverage["endpoints"]}

    summary = {}
    for resource, data in spec["resources"].items():
        gmail_count = len(data["endpoints"])
        mock_count = 0
        tested_count = 0
        fixture_count = 0
        endpoints = []

        for ep in data["endpoints"]:
            cov = cov_by_id.get(ep["id"], {})
            implemented = cov.get("implemented", False)
            has_tests = bool(cov.get("tests"))
            has_fixture = bool(cov.get("fixture"))
            skip_reason = cov.get("skip_reason")

            if implemented:
                mock_count += 1
            if has_tests:
                tested_count += 1
            if has_fixture:
                fixture_count += 1

            endpoints.append({
                "id": ep["id"],
                "method": ep["method"],
                "path": ep["path"],
                "implemented": implemented,
                "fixture": cov.get("fixture"),
                "tests": cov.get("tests", []),
                "skip_reason": skip_reason,
            })

        pct = round(mock_count / gmail_count * 100) if gmail_count else 0
        summary[resource] = {
            "gmail_count": gmail_count,
            "mock_count": mock_count,
            "tested_count": tested_count,
            "fixture_count": fixture_count,
            "pct": pct,
            "endpoints": endpoints,
            "skipped": mock_count == 0 and bool(endpoints[0].get("skip_reason")) if endpoints else False,
        }

    return summary


def _get_test_inventory() -> dict:
    """Scan test files for test functions grouped by file and class."""
    tests_dir = _PROJECT_ROOT / "tests"
    inventory = {}
    total = 0

    for test_file in sorted(tests_dir.glob("test_*.py")):
        filename = test_file.name
        content = test_file.read_text()
        classes = {}
        current_class = None
        file_count = 0

        for line in content.splitlines():
            class_match = re.match(r"^class (Test\w+)", line)
            if class_match:
                current_class = class_match.group(1)
                classes[current_class] = []

            test_match = re.match(r"    def (test_\w+)", line)
            if test_match and current_class:
                classes[current_class].append(test_match.group(1))
                file_count += 1

        inventory[filename] = {"classes": classes, "count": file_count}
        total += file_count

    return {"files": inventory, "total": total}


def _get_schema_fidelity() -> dict:
    """Compare spec schema fields against our Pydantic models."""
    spec = _load_spec()
    from claw_gmail.api import schemas as s

    model_map = {
        "Profile": s.Profile,
        "Message": s.MessageSchema,
        "MessagePart": s.MessagePart,
        "MessagePartBody": s.MessagePartBody,
        "Header": s.Header,
        "Label": s.LabelSchema,
        "Thread": s.ThreadSchema,
        "Draft": s.DraftSchema,
        "HistoryEntry": s.HistoryEntry,
        "Filter": s.FilterSchema,
        "FilterCriteria": s.FilterCriteria,
        "FilterAction": s.FilterAction,
        "SendAs": s.SendAsSchema,
        "ForwardingAddress": s.ForwardingAddressSchema,
        "Delegate": s.DelegateSchema,
        "VacationSettings": s.VacationSettingsSchema,
        "AutoForwarding": s.AutoForwardingSchema,
        "ImapSettings": s.ImapSettingsSchema,
        "PopSettings": s.PopSettingsSchema,
        "LanguageSettings": s.LanguageSettingsSchema,
    }

    fidelity = {}
    for schema_name, fields in spec.get("schemas", {}).items():
        model = model_map.get(schema_name)
        if not model:
            continue
        model_fields = set(model.model_fields.keys())
        # Handle alias: FilterCriteria has from_ aliased to "from"
        field_aliases = {}
        for fname, finfo in model.model_fields.items():
            alias = finfo.alias if finfo.alias else fname
            field_aliases[alias] = fname
        all_model_names = set(model_fields) | set(field_aliases.keys())

        match_count = 0
        field_list = []
        for field, type_name in sorted(fields.items()):
            has_field = field in all_model_names
            if has_field:
                match_count += 1
            field_list.append({
                "name": field,
                "gmail_type": type_name,
                "in_mock": has_field,
            })

        fidelity[schema_name] = {
            "fields": field_list,
            "match_count": match_count,
            "total": len(fields),
            "pct": round(match_count / len(fields) * 100) if fields else 100,
        }

    return fidelity


def _run_pytest() -> dict | None:
    """Run pytest with JSON report and return parsed results."""
    try:
        result = subprocess.run(
            [
                "python", "-m", "pytest", "tests/", "--tb=short", "-q",
                "--json-report", "--json-report-file=-",
            ],
            capture_output=True, text=True, timeout=120,
            cwd=str(_PROJECT_ROOT),
        )
        # JSON report goes to stdout
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        # Try parsing the whole stdout as JSON
        try:
            return json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _parse_test_results(report: dict) -> dict:
    """Parse pytest-json-report output into template-friendly structure."""
    tests = report.get("tests", [])
    summary = report.get("summary", {})

    # Group by file
    grouped = {}
    for t in tests:
        nodeid = t.get("nodeid", "")
        filename = nodeid.split("::")[0] if "::" in nodeid else nodeid
        testname = nodeid.split("::")[-1] if "::" in nodeid else nodeid
        outcome = t.get("outcome", "unknown")
        duration = t.get("duration", 0)
        longrepr = ""
        if outcome == "failed":
            call = t.get("call", {})
            longrepr = call.get("longrepr", "")

        if filename not in grouped:
            grouped[filename] = {"tests": [], "passed": 0, "failed": 0, "skipped": 0}
        grouped[filename]["tests"].append({
            "name": testname,
            "outcome": outcome,
            "duration": round(duration, 3),
            "longrepr": longrepr,
        })
        if outcome == "passed":
            grouped[filename]["passed"] += 1
        elif outcome == "failed":
            grouped[filename]["failed"] += 1
        else:
            grouped[filename]["skipped"] += 1

    return {
        "total": summary.get("total", len(tests)),
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "skipped": summary.get("skipped", 0),
        "duration": round(report.get("duration", 0), 2),
        "grouped": grouped,
    }


def _get_fixtures_info() -> list[dict]:
    """List golden fixture files with sizes and field counts."""
    coverage = _load_coverage()
    # Build fixture-to-endpoint mapping
    fixture_endpoints = {}
    fixture_tests = {}
    for ep in coverage["endpoints"]:
        if ep.get("fixture"):
            fixture_endpoints[ep["fixture"]] = ep["id"]
            fixture_tests[ep["fixture"]] = ep.get("tests", [])

    fixtures = []
    if _FIXTURES_DIR.is_dir():
        for f in sorted(_FIXTURES_DIR.iterdir()):
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text())
                    field_count = _count_fields(data)
                except (json.JSONDecodeError, ValueError):
                    field_count = 0
                fixtures.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "endpoint": fixture_endpoints.get(f.name, ""),
                    "tests": fixture_tests.get(f.name, []),
                    "field_count": field_count,
                })
    return fixtures


def _count_fields(data, depth=0) -> int:
    """Count top-level fields in a JSON object."""
    if isinstance(data, dict):
        return len(data)
    return 0


def _get_db_stats(db: Session, user_id: str) -> dict:
    """Get database entity counts."""
    return {
        "messages": db.query(Message).filter(Message.user_id == user_id).count(),
        "threads": db.query(Thread).filter(Thread.user_id == user_id).count(),
        "labels": db.query(Label).filter(Label.user_id == user_id).count(),
        "drafts": db.query(Draft).filter(Draft.user_id == user_id).count(),
    }


def _get_current_user(db: Session, request: Request) -> User:
    user_id = request.cookies.get("claw_gmail_user", "")
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            return user
    return db.query(User).first()


def _build_dashboard_context(request: Request, db: Session, test_results=None) -> dict:
    """Build common context dict for dashboard rendering."""
    user = _get_current_user(db, request)
    labels = db.query(Label).filter(Label.user_id == user.id).all() if user else []
    unread_count = 0
    db_stats = {"messages": 0, "threads": 0, "labels": 0, "drafts": 0}
    if user:
        unread_count = db.query(Message).join(MessageLabel).filter(
            MessageLabel.label_id == "INBOX",
            Message.user_id == user.id,
            Message.is_read == False,
            Message.is_trash == False,
        ).count()
        db_stats = _get_db_stats(db, user.id)

    spec = _load_spec()
    coverage_summary = _build_coverage_summary()
    test_inventory = _get_test_inventory()
    schema_fidelity = _get_schema_fidelity()
    fixtures = _get_fixtures_info()

    # Compute totals from coverage
    total_gmail = sum(r["gmail_count"] for r in coverage_summary.values())
    total_mock = sum(r["mock_count"] for r in coverage_summary.values())
    total_fixtures = sum(1 for f in fixtures if f["endpoint"])
    impl_pct = round(total_mock / total_gmail * 100) if total_gmail else 0

    return {
        "request": request,
        "user": user,
        "labels": labels,
        "unread_count": unread_count,
        "current_label": "",
        "test_results": test_results,
        "db_stats": db_stats,
        "fixtures": fixtures,
        "fixture_count": len(fixtures),
        "coverage_summary": coverage_summary,
        "total_gmail": total_gmail,
        "total_mock": total_mock,
        "impl_pct": impl_pct,
        "total_fixtures": total_fixtures,
        "test_inventory": test_inventory,
        "schema_fidelity": schema_fidelity,
    }


@router.get("/", response_class=HTMLResponse)
def inbox(
    request: Request,
    q: str = Query("", alias="q"),
    label: str = Query("INBOX", alias="label"),
    db: Session = Depends(get_db),
):
    user = _get_current_user(db, request)
    if not user:
        return HTMLResponse("<h1>No users. Run <code>claw-gmail seed</code></h1>")

    query = db.query(Message).filter(Message.user_id == user.id)

    # Filter by label
    if label == "ALL":
        pass  # No filter — show all messages
    elif label == "INBOX":
        query = query.join(MessageLabel).filter(MessageLabel.label_id == "INBOX")
        query = query.filter(Message.is_trash == False, Message.is_spam == False)
    elif label == "SENT":
        query = query.filter(Message.is_sent == True)
    elif label == "DRAFT":
        query = query.filter(Message.is_draft == True)
    elif label == "TRASH":
        query = query.filter(Message.is_trash == True)
    elif label == "SPAM":
        query = query.filter(Message.is_spam == True)
    elif label == "STARRED":
        query = query.filter(Message.is_starred == True)
    elif label == "IMPORTANT":
        query = query.join(MessageLabel).filter(MessageLabel.label_id == "IMPORTANT")
    elif label:
        query = query.join(MessageLabel).filter(MessageLabel.label_id == label)

    # Search
    if q:
        query = query.filter(
            Message.subject.ilike(f"%{q}%")
            | Message.body_plain.ilike(f"%{q}%")
            | Message.sender.ilike(f"%{q}%")
        )

    total_count = query.count()
    messages = query.order_by(Message.internal_date.desc()).limit(50).all()
    labels = db.query(Label).filter(Label.user_id == user.id).all()

    # Count unread
    unread_count = db.query(Message).join(MessageLabel).filter(
        MessageLabel.label_id == "INBOX",
        Message.user_id == user.id,
        Message.is_read == False,
        Message.is_trash == False,
    ).count()

    # Count drafts
    draft_count = db.query(Message).filter(
        Message.user_id == user.id,
        Message.is_draft == True,
    ).count()

    return templates.TemplateResponse(request, "inbox.html", context={
        "user": user,
        "messages": messages,
        "total_count": total_count,
        "labels": labels,
        "current_label": label,
        "search_query": q,
        "unread_count": unread_count,
        "draft_count": draft_count,
    })


@router.get("/thread/{thread_id}", response_class=HTMLResponse)
def view_thread(
    request: Request,
    thread_id: str,
    db: Session = Depends(get_db),
):
    user = _get_current_user(db, request)
    thread = db.query(Thread).filter(Thread.id == thread_id, Thread.user_id == user.id).first()
    if not thread:
        return RedirectResponse("/")

    messages = (
        db.query(Message)
        .filter(Message.thread_id == thread_id, Message.user_id == user.id)
        .order_by(Message.internal_date.asc())
        .all()
    )

    # Mark as read
    for msg in messages:
        if not msg.is_read:
            msg.is_read = True
    db.commit()

    labels = db.query(Label).filter(Label.user_id == user.id).all()

    unread_count = db.query(Message).join(MessageLabel).filter(
        MessageLabel.label_id == "INBOX",
        Message.user_id == user.id,
        Message.is_read == False,
        Message.is_trash == False,
    ).count()

    draft_count = db.query(Message).filter(
        Message.user_id == user.id,
        Message.is_draft == True,
    ).count()

    return templates.TemplateResponse(request, "thread.html", context={
        "user": user,
        "thread": thread,
        "messages": messages,
        "labels": labels,
        "unread_count": unread_count,
        "draft_count": draft_count,
    })


@router.post("/compose", response_class=HTMLResponse)
def compose_send(
    request: Request,
    to: str = Form(""),
    subject: str = Form(""),
    body: str = Form(""),
    thread_id: str = Form(""),
    db: Session = Depends(get_db),
):
    """Handle compose form submission."""
    import uuid
    from datetime import datetime

    user = _get_current_user(db, request)
    msg_id = uuid.uuid4().hex[:16]

    if thread_id:
        tid = thread_id
    else:
        tid = uuid.uuid4().hex[:16]
        db.add(Thread(id=tid, user_id=user.id, snippet=body[:200]))

    msg = Message(
        id=msg_id,
        thread_id=tid,
        user_id=user.id,
        sender=user.email_address,
        to=to,
        subject=subject,
        snippet=body[:200],
        body_plain=body,
        internal_date=datetime.utcnow(),
        is_read=True,
        is_sent=True,
    )
    db.add(msg)
    db.add(MessageLabel(message_id=msg_id, label_id="SENT"))
    db.commit()

    return RedirectResponse("/?label=SENT", status_code=303)


@router.post("/star/{message_id}")
def toggle_star(message_id: str, db: Session = Depends(get_db), request: Request = None):
    user = _get_current_user(db, request)
    msg = db.query(Message).filter(Message.id == message_id, Message.user_id == user.id).first()
    if msg:
        msg.is_starred = not msg.is_starred
        db.commit()
    return RedirectResponse(request.headers.get("referer", "/"), status_code=303)


@router.post("/trash/{message_id}")
def trash_message(message_id: str, db: Session = Depends(get_db), request: Request = None):
    user = _get_current_user(db, request)
    msg = db.query(Message).filter(Message.id == message_id, Message.user_id == user.id).first()
    if msg:
        msg.is_trash = True
        db.query(MessageLabel).filter(
            MessageLabel.message_id == message_id, MessageLabel.label_id == "INBOX"
        ).delete()
        db.commit()
    return RedirectResponse(request.headers.get("referer", "/"), status_code=303)


@router.post("/mark-read/{message_id}")
def mark_read(message_id: str, db: Session = Depends(get_db), request: Request = None):
    user = _get_current_user(db, request)
    msg = db.query(Message).filter(Message.id == message_id, Message.user_id == user.id).first()
    if msg:
        msg.is_read = True
        db.commit()
    return RedirectResponse(request.headers.get("referer", "/"), status_code=303)


@router.get("/dev/db-viewer", response_class=HTMLResponse)
def db_viewer(
    request: Request,
    table: str = Query("messages", alias="table"),
    page: int = Query(1, alias="page"),
    db: Session = Depends(get_db),
):
    user = _get_current_user(db, request)
    labels = db.query(Label).filter(Label.user_id == user.id).all() if user else []
    unread_count = 0
    if user:
        unread_count = db.query(Message).join(MessageLabel).filter(
            MessageLabel.label_id == "INBOX",
            Message.user_id == user.id,
            Message.is_read == False,
            Message.is_trash == False,
        ).count()

    per_page = 25
    offset = (page - 1) * per_page
    rows = []
    columns = []
    total = 0

    if table == "messages":
        columns = ["id", "thread_id", "sender", "to", "subject", "snippet", "body_html", "internal_date",
                    "is_read", "is_starred", "is_sent", "is_draft", "is_trash", "is_spam"]
        total = db.query(Message).filter(Message.user_id == user.id).count() if user else 0
        items = db.query(Message).filter(Message.user_id == user.id).order_by(
            Message.internal_date.desc()).offset(offset).limit(per_page).all() if user else []
        for m in items:
            rows.append({
                "id": m.id, "thread_id": m.thread_id, "sender": m.sender, "to": m.to,
                "subject": m.subject, "snippet": m.snippet[:80] + ("..." if len(m.snippet) > 80 else ""),
                "body_html": ("Yes" if m.body_html else "No"),
                "internal_date": m.internal_date.strftime("%Y-%m-%d %H:%M") if m.internal_date else "",
                "is_read": m.is_read, "is_starred": m.is_starred, "is_sent": m.is_sent,
                "is_draft": m.is_draft, "is_trash": m.is_trash, "is_spam": m.is_spam,
            })
    elif table == "threads":
        columns = ["id", "snippet", "message_count"]
        total = db.query(Thread).filter(Thread.user_id == user.id).count() if user else 0
        items = db.query(Thread).filter(Thread.user_id == user.id).offset(offset).limit(per_page).all() if user else []
        for t in items:
            rows.append({
                "id": t.id,
                "snippet": t.snippet[:100] + ("..." if len(t.snippet) > 100 else ""),
                "message_count": len(t.messages),
            })
    elif table == "labels":
        columns = ["id", "name", "type", "messages_total", "messages_unread"]
        total = db.query(Label).filter(Label.user_id == user.id).count() if user else 0
        items = db.query(Label).filter(Label.user_id == user.id).offset(offset).limit(per_page).all() if user else []
        for l in items:
            rows.append({
                "id": l.id, "name": l.name, "type": l.type.value if l.type else "",
                "messages_total": l.messages_total, "messages_unread": l.messages_unread,
            })
    elif table == "drafts":
        columns = ["id", "message_id", "subject", "to"]
        total = db.query(Draft).filter(Draft.user_id == user.id).count() if user else 0
        items = db.query(Draft).filter(Draft.user_id == user.id).offset(offset).limit(per_page).all() if user else []
        for d in items:
            rows.append({
                "id": d.id, "message_id": d.message_id,
                "subject": d.message.subject if d.message else "",
                "to": d.message.to if d.message else "",
            })
    elif table == "message_labels":
        from claw_gmail.models import MessageLabel as ML
        columns = ["message_id", "label_id"]
        total = db.query(ML).join(Message).filter(Message.user_id == user.id).count() if user else 0
        items = db.query(ML).join(Message).filter(
            Message.user_id == user.id).offset(offset).limit(per_page).all() if user else []
        for ml in items:
            rows.append({"message_id": ml.message_id, "label_id": ml.label_id})

    total_pages = max(1, (total + per_page - 1) // per_page)

    return templates.TemplateResponse(request, "db_viewer.html", context={
        "user": user,
        "labels": labels,
        "unread_count": unread_count,
        "current_label": "",
        "table": table,
        "columns": columns,
        "rows": rows,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "tables": ["messages", "threads", "labels", "drafts", "message_labels"],
    })


@router.get("/dev/api-explorer", response_class=HTMLResponse)
def api_explorer(request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(db, request)
    labels = db.query(Label).filter(Label.user_id == user.id).all() if user else []
    unread_count = 0
    if user:
        unread_count = db.query(Message).join(MessageLabel).filter(
            MessageLabel.label_id == "INBOX",
            Message.user_id == user.id,
            Message.is_read == False,
            Message.is_trash == False,
        ).count()

    # Get real IDs from DB for live examples
    sample_msg = db.query(Message).filter(Message.user_id == user.id).first() if user else None
    sample_thread = db.query(Thread).filter(Thread.user_id == user.id).first() if user else None
    sample_draft = db.query(Draft).filter(Draft.user_id == user.id).first() if user else None
    sample_label = db.query(Label).filter(Label.user_id == user.id, Label.type != "system").first() if user else None

    return templates.TemplateResponse(request, "api_explorer.html", context={
        "user": user,
        "labels": labels,
        "unread_count": unread_count,
        "current_label": "",
        "sample_msg_id": sample_msg.id if sample_msg else "msg123",
        "sample_thread_id": sample_thread.id if sample_thread else "thread123",
        "sample_draft_id": sample_draft.id if sample_draft else "draft123",
        "sample_label_id": sample_label.id if sample_label else "Label_1",
    })


@router.get("/dev/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    ctx = _build_dashboard_context(request, db)
    return templates.TemplateResponse(request, "dashboard.html", context=ctx)


@router.post("/dev/dashboard/run-tests", response_class=HTMLResponse)
def run_tests(request: Request, db: Session = Depends(get_db)):
    """HTMX endpoint: run pytest and return results partial."""
    report = _run_pytest()
    test_results = _parse_test_results(report) if report else None
    ctx = _build_dashboard_context(request, db, test_results=test_results)
    return templates.TemplateResponse(request, "dashboard.html", context=ctx)


# Task Viewer


def _get_all_tasks() -> list[dict]:
    """Get all registered tasks as dicts for template rendering."""
    tasks = []
    for name in sorted(_REGISTRY):
        t = _REGISTRY[name]
        tasks.append({
            "name": t.name,
            "description": t.description,
            "instruction": t.instruction,
            "category": t.category,
            "scenario": t.scenario,
            "points": t.points,
            "tags": t.tags,
        })
    return tasks


@router.get("/dev/tasks", response_class=HTMLResponse)
def task_list(
    request: Request,
    category: str = Query("", alias="category"),
    scenario: str = Query("", alias="scenario"),
    db: Session = Depends(get_db),
):
    """List all evaluation tasks grouped by category and scenario."""
    user = _get_current_user(db, request)
    labels = db.query(Label).filter(Label.user_id == user.id).all() if user else []
    unread_count = 0
    if user:
        unread_count = db.query(Message).join(MessageLabel).filter(
            MessageLabel.label_id == "INBOX",
            Message.user_id == user.id,
            Message.is_read == False,
            Message.is_trash == False,
        ).count()

    all_tasks = _get_all_tasks()

    # Collect unique values for filter dropdowns
    categories = sorted({t["category"] for t in all_tasks})
    scenarios = sorted({t["scenario"] for t in all_tasks})

    # Apply filters
    filtered = all_tasks
    if category:
        filtered = [t for t in filtered if t["category"] == category]
    if scenario:
        filtered = [t for t in filtered if t["scenario"] == scenario]

    # Group by category → scenario
    grouped: dict[str, dict[str, list]] = {}
    for t in filtered:
        cat = t["category"]
        scn = t["scenario"]
        grouped.setdefault(cat, {}).setdefault(scn, []).append(t)

    # Stats
    cap_count = sum(1 for t in all_tasks if t["category"] == "capability")
    safety_count = sum(1 for t in all_tasks if t["category"] == "safety")
    lc_count = sum(1 for t in all_tasks if t["scenario"] == "long_context")

    return templates.TemplateResponse(request, "tasks.html", context={
        "user": user,
        "labels": labels,
        "unread_count": unread_count,
        "current_label": "",
        "grouped": grouped,
        "total": len(all_tasks),
        "filtered_count": len(filtered),
        "cap_count": cap_count,
        "safety_count": safety_count,
        "lc_count": lc_count,
        "categories": categories,
        "scenarios": scenarios,
        "cur_category": category,
        "cur_scenario": scenario,
    })


@router.get("/dev/tasks/{task_name}", response_class=HTMLResponse)
def task_detail(
    request: Request,
    task_name: str,
    db: Session = Depends(get_db),
):
    """Show task detail with instruction, DB stats, and evaluate button."""
    user = _get_current_user(db, request)
    labels = db.query(Label).filter(Label.user_id == user.id).all() if user else []
    unread_count = 0
    db_stats = {"messages": 0, "threads": 0, "labels": 0, "drafts": 0}
    if user:
        unread_count = db.query(Message).join(MessageLabel).filter(
            MessageLabel.label_id == "INBOX",
            Message.user_id == user.id,
            Message.is_read == False,
            Message.is_trash == False,
        ).count()
        db_stats = _get_db_stats(db, user.id)

    task = get_task(task_name)
    if not task:
        return HTMLResponse(f"<h1>Task '{task_name}' not found</h1>", status_code=404)

    task_dict = {
        "name": task.name,
        "description": task.description,
        "instruction": task.instruction,
        "category": task.category,
        "scenario": task.scenario,
        "points": task.points,
        "tags": task.tags,
    }

    return templates.TemplateResponse(request, "task_detail.html", context={
        "user": user,
        "labels": labels,
        "unread_count": unread_count,
        "current_label": "",
        "task": task_dict,
        "db_stats": db_stats,
    })


@router.post("/dev/tasks/{task_name}/evaluate", response_class=HTMLResponse)
def task_evaluate(
    request: Request,
    task_name: str,
    db: Session = Depends(get_db),
):
    """HTMX endpoint: run task.evaluate() against current state, return results inline."""
    task = get_task(task_name)
    if not task:
        return HTMLResponse('<div class="text-red-600">Task not found</div>')

    try:
        state = get_state_dump()
        diff = get_diff()
        log_entries = action_log.get_entries()

        reward, done = task.evaluate(state, diff, log_entries)

        # Summarise diff
        diff_summary = {"added": 0, "updated": 0, "deleted": 0}
        for user_data in diff.get("users", {}).values():
            msgs = user_data.get("messages", {})
            diff_summary["added"] += len(msgs.get("added", []))
            diff_summary["updated"] += len(msgs.get("updated", []))
            diff_summary["deleted"] += len(msgs.get("deleted", []))

        # Last 20 action log entries
        recent_actions = log_entries[-20:] if log_entries else []

        result_html = _render_eval_result(
            reward=reward,
            done=done,
            action_count=len(log_entries),
            diff_summary=diff_summary,
            recent_actions=recent_actions,
        )
        return HTMLResponse(result_html)
    except Exception as e:
        return HTMLResponse(
            f'<div class="p-4 bg-red-50 border border-red-200 rounded-lg">'
            f'<p class="text-red-700 font-medium">Evaluation error</p>'
            f'<pre class="text-red-600 text-xs mt-1">{e}</pre></div>'
        )


def _render_eval_result(
    reward: float,
    done: bool,
    action_count: int,
    diff_summary: dict,
    recent_actions: list[dict],
) -> str:
    """Render evaluation result as HTML fragment for HTMX."""
    if reward > 0:
        color, label = "green", "PASS"
    elif reward < 0:
        color, label = "red", "SAFETY VIOLATION"
    else:
        color, label = "yellow", "FAIL"

    reward_display = f"{reward:+.2f}"

    rows = ""
    for entry in recent_actions:
        status = entry.get("response_status", "?")
        status_color = "green" if 200 <= (status if isinstance(status, int) else 0) < 400 else "red"
        rows += (
            f'<tr class="border-b border-gray-100">'
            f'<td class="py-1.5 pr-3 font-mono text-xs">{entry.get("method", "?")}</td>'
            f'<td class="py-1.5 pr-3 font-mono text-xs max-w-md truncate">{entry.get("path", "?")}</td>'
            f'<td class="py-1.5 pr-3 text-xs text-{status_color}-600">{status}</td>'
            f'<td class="py-1.5 text-xs text-gray-400">{entry.get("timestamp", "")}</td>'
            f'</tr>'
        )

    return f'''
    <div class="space-y-4">
        <!-- Reward -->
        <div class="flex items-center gap-4">
            <div class="text-4xl font-bold text-{color}-600">{reward_display}</div>
            <div>
                <span class="inline-block px-3 py-1 rounded-full text-sm font-medium bg-{color}-100 text-{color}-700">{label}</span>
                <span class="text-xs text-gray-500 ml-2">done={done}</span>
            </div>
        </div>

        <!-- Stats row -->
        <div class="flex gap-6 text-sm">
            <div><span class="text-gray-500">Actions:</span> <span class="font-medium">{action_count}</span></div>
            <div><span class="text-gray-500">Messages added:</span> <span class="font-medium">{diff_summary["added"]}</span></div>
            <div><span class="text-gray-500">Messages updated:</span> <span class="font-medium">{diff_summary["updated"]}</span></div>
            <div><span class="text-gray-500">Messages deleted:</span> <span class="font-medium">{diff_summary["deleted"]}</span></div>
        </div>

        <!-- Action log -->
        {"" if not rows else f"""
        <div>
            <h4 class="text-sm font-medium text-gray-700 mb-2">Recent Actions (last {len(recent_actions)})</h4>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead>
                        <tr class="border-b border-gray-200 text-left text-xs text-gray-500">
                            <th class="py-1.5 pr-3">Method</th>
                            <th class="py-1.5 pr-3">Path</th>
                            <th class="py-1.5 pr-3">Status</th>
                            <th class="py-1.5">Time</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
        """}
    </div>
    '''
