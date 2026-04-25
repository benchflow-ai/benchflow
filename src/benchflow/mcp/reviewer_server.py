"""MCP reviewer server — exposes code review as a tool for other agents.

Runs as a sidecar in the same sandbox. The coder agent calls the
`review_code` tool via MCP; the reviewer LLM analyzes the code and
returns structured feedback.

This replaces the filesystem-based outbox pattern from followup-bench
with a clean tool-call interface that prevents reward hacking (reviewer
never has write access to /app/).

Usage in task.toml:
    [[environment.mcp_servers]]
    name = "reviewer"
    transport = "streamable-http"
    url = "http://localhost:8100/mcp"

Or start manually:
    python -m benchflow.mcp.reviewer_server --port 8100
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default review prompt — can be overridden via REVIEWER_PROMPT env var
DEFAULT_REVIEW_PROMPT = """You are an expert code reviewer. Analyze the provided code for:
1. Correctness — does it produce the right output for all inputs?
2. Completeness — does it address every requirement?
3. Bugs — trace through with concrete inputs, flag any wrong paths.

Be specific: reference file names and concrete failing inputs.
If uncertain, say so. Only report evidence-backed issues."""


def create_reviewer_server(
    model: str = "gemini-3.1-flash-lite",
    port: int = 8100,
    review_prompt: str | None = None,
):
    """Create a FastMCP reviewer server.

    Returns the server app (for use with uvicorn or similar).
    Requires: pip install fastmcp
    """
    try:
        from fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "fastmcp required for MCP reviewer server. "
            "Install with: pip install fastmcp"
        ) from e

    mcp = FastMCP("benchflow-reviewer")
    prompt = review_prompt or os.environ.get("REVIEWER_PROMPT", DEFAULT_REVIEW_PROMPT)

    @mcp.tool()
    async def review_code(
        files: list[str] | None = None,
        task_instruction: str | None = None,
    ) -> str:
        """Review code files for correctness, completeness, and bugs.

        Args:
            files: List of file paths to review (relative to /app/).
                   If None, reviews all modified files in /app/.
            task_instruction: Optional task description for context.

        Returns:
            Structured review feedback as a string.
        """
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Error: No GOOGLE_API_KEY or GEMINI_API_KEY set for reviewer."

        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(model)

        # Read files — restrict to /app/ to prevent path traversal
        file_contents = []
        target_files = files or _find_modified_files()
        app_root = Path("/app").resolve()
        for f in target_files:
            path = (Path("/app") / f).resolve()
            if not str(path).startswith(str(app_root)):
                continue
            if path.exists():
                content = path.read_text()[:50000]
                file_contents.append(f"=== {f} ===\n{content}")

        if not file_contents:
            return "No files found to review."

        review_input = "\n\n".join(file_contents)
        if task_instruction:
            review_input = f"TASK:\n{task_instruction}\n\nCODE:\n{review_input}"

        full_prompt = f"{prompt}\n\nReview the following:\n{review_input}"

        response = await asyncio.to_thread(llm.generate_content, full_prompt)
        return response.text

    @mcp.tool()
    async def get_review_status() -> str:
        """Check if the reviewer is ready."""
        return json.dumps({"status": "ready", "model": model})

    return mcp


def _find_modified_files() -> list[str]:
    """Find files in /app/ that look like code (not config/data)."""
    app = Path("/app")
    if not app.exists():
        return []
    code_exts = {".py", ".js", ".ts", ".sh", ".c", ".cpp", ".go", ".rs", ".java", ".rb"}
    return [
        str(f.relative_to(app))
        for f in app.rglob("*")
        if f.is_file() and f.suffix in code_exts and ".git" not in str(f)
    ][:20]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BenchFlow MCP Reviewer Server")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = create_reviewer_server(model=args.model, port=args.port)
    server.run(transport="streamable-http", host=args.host, port=args.port)
