"""Personal benchmark curation from agent traces.

Import Claude Code sessions, opentraces datasets, or HuggingFace trace
datasets and convert them into runnable BenchFlow tasks.

Submodules:

* ``parsers`` — read Claude Code JSONL and opentraces formats
* ``task_gen`` — convert parsed traces into task.toml + instruction.md
* ``huggingface`` — download trace datasets from HuggingFace Hub
"""

from benchflow.traces.models import ParsedTrace, TraceStep
from benchflow.traces.parsers import (
    parse_claude_code_file,
    parse_claude_code_session,
    parse_opentraces_record,
)
from benchflow.traces.task_gen import generate_task

__all__ = [
    "ParsedTrace",
    "TraceStep",
    "generate_task",
    "parse_claude_code_file",
    "parse_claude_code_session",
    "parse_opentraces_record",
]
