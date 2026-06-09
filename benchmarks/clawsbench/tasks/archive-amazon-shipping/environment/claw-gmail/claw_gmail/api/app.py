"""Main FastAPI application."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from claw_gmail.state.action_log import action_log
from claw_gmail.state.snapshots import get_state_dump, get_diff, take_snapshot, restore_snapshot
from . import messages, threads, labels, drafts, history, settings
from claw_gmail.web.routes import router as web_router

app = FastAPI(
    title="Mock Gmail API",
    description="Gmail-compatible REST API for AI agent safety evaluation and RL training",
    version="0.1.0",
)


# Gmail-style error responses
_HTTP_STATUS_MAP = {
    400: ("INVALID_ARGUMENT", "badRequest"),
    401: ("UNAUTHENTICATED", "unauthorized"),
    403: ("PERMISSION_DENIED", "forbidden"),
    404: ("NOT_FOUND", "notFound"),
    409: ("ALREADY_EXISTS", "conflict"),
    429: ("RESOURCE_EXHAUSTED", "rateLimitExceeded"),
    500: ("INTERNAL", "backendError"),
}


@app.exception_handler(HTTPException)
async def gmail_error_handler(request: Request, exc: HTTPException):
    """Return errors in Gmail API format: {"error": {"code": ..., "message": ..., ...}}."""
    status, reason = _HTTP_STATUS_MAP.get(exc.status_code, ("UNKNOWN", "unknown"))
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": message,
                "status": status,
                "errors": [
                    {
                        "message": message,
                        "domain": "global",
                        "reason": reason,
                    }
                ],
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def gmail_validation_error_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic 422 errors to Google API-style 400 errors.

    Real Gmail API returns 400 with {"error": {"code": 400, ...}}
    for invalid request bodies. FastAPI/Pydantic returns 422 by default.
    """
    errors = exc.errors()
    message = "; ".join(
        f"{'.'.join(str(loc) for loc in e.get('loc', []))}: {e.get('msg', '')}"
        for e in errors
    ) if errors else "Invalid request body."
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": 400,
                "message": message,
                "reason": "required",
                "status": "INVALID_ARGUMENT",
                "errors": [
                    {
                        "message": message,
                        "domain": "global",
                        "reason": "required",
                    }
                ],
            }
        },
    )

# CORS — allow everything for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Action logging middleware
class ActionLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip logging for admin/static/docs endpoints
        path = str(request.url.path)
        query = str(request.url.query)
        full_path = f"{path}?{query}" if query else path
        if path.startswith(("/_admin", "/docs", "/openapi", "/static", "/mcp")):
            return await call_next(request)

        body_bytes = await request.body()
        body_dict = None
        if body_bytes:
            try:
                body_dict = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        response = await call_next(request)

        # Extract user from header or path
        user_id = request.headers.get("X-Mock-Gmail-User", "")
        if not user_id and "/users/" in path:
            parts = path.split("/users/")
            if len(parts) > 1:
                user_id = parts[1].split("/")[0]

        action_log.record(
            method=request.method,
            path=full_path,
            user_id=user_id,
            request_body=body_dict,
            response_status=response.status_code,
        )

        return response


app.add_middleware(ActionLogMiddleware)


# Gmail API routes
GMAIL_PREFIX = "/gmail/v1"

app.include_router(web_router, tags=["web"])
app.include_router(messages.router, prefix=GMAIL_PREFIX, tags=["messages"])
app.include_router(threads.router, prefix=GMAIL_PREFIX, tags=["threads"])
app.include_router(labels.router, prefix=GMAIL_PREFIX, tags=["labels"])
app.include_router(drafts.router, prefix=GMAIL_PREFIX, tags=["drafts"])
app.include_router(history.router, prefix=GMAIL_PREFIX, tags=["history"])
app.include_router(settings.router, prefix=GMAIL_PREFIX, tags=["settings"])


# Profile
from claw_gmail.models import get_session_factory, User, Message, Thread
from .deps import get_db, resolve_user_id
from .schemas import Profile
from fastapi import Depends
from sqlalchemy.orm import Session


@app.get(f"{GMAIL_PREFIX}/users/{{userId}}/profile", response_model=Profile, tags=["profile"])
def get_profile(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    user = db.query(User).filter(User.id == _user_id).first()
    msg_count = db.query(Message).filter(Message.user_id == _user_id).count()
    thread_count = db.query(Thread).filter(Thread.user_id == _user_id).count()
    return Profile(
        emailAddress=user.email_address,
        messagesTotal=msg_count,
        threadsTotal=thread_count,
        historyId=str(user.history_id),
    )


# Attachments
from claw_gmail.models import Attachment
from .schemas import AttachmentSchema


@app.get(
    f"{GMAIL_PREFIX}/users/{{userId}}/messages/{{messageId}}/attachments/{{attachmentId}}",
    response_model=AttachmentSchema,
    tags=["attachments"],
)
def get_attachment(
    userId: str,
    messageId: str,
    attachmentId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    from fastapi import HTTPException
    att = db.query(Attachment).filter(
        Attachment.id == attachmentId,
        Attachment.message_id == messageId,
    ).first()
    if not att:
        raise HTTPException(404, "Attachment not found")
    return AttachmentSchema(attachmentId=att.id, size=att.size, data=att.data)


# Admin endpoints
@app.post("/_admin/reset", tags=["admin"])
def admin_reset():
    """Reset to initial seed state."""
    success = restore_snapshot("initial")
    action_log.clear()
    if success:
        return {"status": "ok", "message": "Reset to initial state"}
    return {"status": "error", "message": "No initial snapshot found. Run `claw-gmail seed` first."}


@app.post("/_admin/seed", tags=["admin"])
def admin_seed(scenario: str = "default", seed: int = 42):
    """Re-seed database with a specific scenario (drops and recreates all data)."""
    from claw_gmail.models import Base, get_engine
    from claw_gmail.seed.generator import seed_database
    # Extract current DB path before seed_database resets the engine
    engine = get_engine()
    db_url = str(engine.url)
    db_path = db_url.replace("sqlite:///", "") if db_url.startswith("sqlite:///") else None
    Base.metadata.drop_all(engine)
    try:
        result = seed_database(scenario=scenario, seed=seed, db_path=db_path)
        action_log.clear()
        return {"status": "ok", "scenario": scenario, **result}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/_admin/state", tags=["admin"])
def admin_state():
    """Full state dump for evaluation."""
    return get_state_dump()


@app.get("/_admin/diff", tags=["admin"])
def admin_diff():
    """Diff vs initial state."""
    return get_diff()


@app.get("/_admin/action_log", tags=["admin"])
def admin_action_log():
    """Audit trail of all API calls."""
    return {"entries": action_log.get_entries(), "count": len(action_log)}


@app.post("/_admin/snapshot/{name}", tags=["admin"])
def admin_snapshot(name: str):
    """Save current state as a named snapshot."""
    path = take_snapshot(name)
    return {"status": "ok", "path": str(path)}


@app.post("/_admin/restore/{name}", tags=["admin"])
def admin_restore(name: str):
    """Restore from a named snapshot."""
    success = restore_snapshot(name)
    if success:
        return {"status": "ok", "message": f"Restored from snapshot '{name}'"}
    return {"status": "error", "message": f"Snapshot '{name}' not found"}


@app.get("/_admin/tasks", tags=["admin"])
def admin_tasks():
    """JSON list of all registered task metadata."""
    from claw_gmail.tasks import list_tasks as _list_tasks, get_task as _get_task

    tasks = []
    for name in _list_tasks():
        t = _get_task(name)
        tasks.append({
            "name": t.name,
            "description": t.description,
            "instruction": t.instruction,
            "category": t.category,
            "scenario": t.scenario,
            "points": t.points,
            "tags": t.tags,
        })
    return {"tasks": tasks, "count": len(tasks)}


@app.post("/_admin/tasks/{task_name}/evaluate", tags=["admin"])
def admin_task_evaluate(task_name: str):
    """Run task.evaluate() against current state, return JSON results."""
    from claw_gmail.tasks import get_task as _get_task

    task = _get_task(task_name)
    if not task:
        raise HTTPException(404, f"Task '{task_name}' not found")

    state = get_state_dump()
    diff = get_diff()
    log_entries = action_log.get_entries()

    reward, done = task.evaluate(state, diff, log_entries)

    diff_summary = {"added": 0, "updated": 0, "deleted": 0}
    for user_data in diff.get("users", {}).values():
        msgs = user_data.get("messages", {})
        diff_summary["added"] += len(msgs.get("added", []))
        diff_summary["updated"] += len(msgs.get("updated", []))
        diff_summary["deleted"] += len(msgs.get("deleted", []))

    recent_actions = log_entries[-20:] if log_entries else []

    return {
        "task_name": task_name,
        "reward": reward,
        "done": done,
        "diff_summary": diff_summary,
        "action_count": len(log_entries),
        "recent_actions": recent_actions,
    }


@app.get("/_admin/tasks/{task_name}/files", tags=["admin"])
def admin_task_files(task_name: str):
    """Serve all Harbor task file contents dynamically."""
    import pathlib, tomllib

    harbor_dir = pathlib.Path(__file__).resolve().parents[2] / "tasks" / "harbor"
    # Map registered task name to directory name (strip 'harbor-' prefix)
    dir_name = task_name.removeprefix("harbor-")
    task_dir = harbor_dir / dir_name

    if not task_dir.exists():
        raise HTTPException(404, f"Harbor task directory not found: {dir_name}")

    TEXT_EXTENSIONS = {
        ".md", ".py", ".sh", ".toml", ".yaml", ".yml", ".json", ".txt",
        ".cfg", ".ini", ".env", ".j2", ".jinja", ".dockerfile",
    }

    files: dict = {}
    all_files: list[dict] = []

    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(task_dir))
        # Skip __pycache__ and hidden files
        if "__pycache__" in rel or rel.startswith("."):
            continue
        is_text = path.suffix.lower() in TEXT_EXTENSIONS or path.name in {
            "Dockerfile", "Makefile", "entrypoint.sh",
        }
        entry = {"path": rel, "size": path.stat().st_size, "is_text": is_text}
        if is_text:
            try:
                entry["content"] = path.read_text()
            except Exception:
                entry["content"] = "(failed to read)"
        all_files.append(entry)

    files["all_files"] = all_files

    # Legacy keys for backward compat
    for f in all_files:
        if f["path"] == "tests/evaluate.py":
            files["evaluate_py"] = f.get("content", "")
        elif f["path"] == "task.toml":
            files["task_toml"] = f.get("content", "")
            try:
                files["task_meta"] = tomllib.loads(files["task_toml"])
            except Exception:
                files["task_meta"] = {}
        elif f["path"] == "instruction.md":
            files["instruction_md"] = f.get("content", "")
        elif f["path"] == "solution/solve.sh":
            files["solve_sh"] = f.get("content", "")
        elif f["path"] == "environment/Dockerfile":
            files["dockerfile"] = f.get("content", "")

    return files


def _scan_skill_files(skill_dir: pathlib.Path) -> list[dict]:
    """Recursively collect all files in a skill directory."""
    files = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(skill_dir))
        # Skip hidden files and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        try:
            content = path.read_text()
        except (UnicodeDecodeError, ValueError):
            content = f"(binary file, {path.stat().st_size} bytes)"
        files.append({"path": rel, "content": content, "size": path.stat().st_size})
    return files


@app.get("/_admin/skills", tags=["admin"])
def admin_skills():
    """List all agent skills from the skills/ directory."""
    import pathlib

    skills_dir = pathlib.Path(__file__).resolve().parents[5] / "skills"
    skills = []
    if skills_dir.is_dir():
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                skill_files = _scan_skill_files(skill_dir)
                skills.append({
                    "name": skill_dir.name,
                    "content": skill_md.read_text(),
                    "files": skill_files,
                })
    return {"skills": skills, "count": len(skills)}


@app.get("/_admin/skills/{skill_name}", tags=["admin"])
def admin_skill_detail(skill_name: str):
    """Get a single skill with all files in its directory."""
    import pathlib

    skills_dir = pathlib.Path(__file__).resolve().parents[5] / "skills"
    skill_dir = skills_dir / skill_name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise HTTPException(404, f"Skill '{skill_name}' not found")
    return {
        "name": skill_name,
        "content": skill_md.read_text(),
        "files": _scan_skill_files(skill_dir),
    }


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}
