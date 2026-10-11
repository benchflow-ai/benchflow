"""Runs DSPy's recursive language model (dspy.RLM) once on the task, for BenchFlow.

Written for BenchFlow, after Harbor's `dspy-rlm` agent (Apache-2.0,
harbor/agents/dspy_rlm.py): the same signature (`instruction, file_tree ->
solution`), limits (20 iterations, 50 sub-LM calls, 10,000 output
characters, 30 s per tool call, 16,000 output tokens per model call), and
seven tools with Harbor's names and docstrings. The difference is where it
runs: Harbor runs DSPy on the host and bridges each tool to the sandbox;
here DSPy runs inside the sandbox as the agent user, so the tools act on
the task's files directly. The model's Python still runs in DSPy's
Deno/Pyodide REPL, where the tools are plain functions.

The file tree is listed from the task's working folder (Harbor lists its
`working_dir`, `/` by default).

Model: BenchFlow's provider forwarder serves the run's model at
`$OPENAI_BASE_URL` with a placeholder key; `$BENCHFLOW_DSPY_LM` is the
LiteLLM name (`openai/<model>`). The forwarder adds the real key and, on a
gateway, the allowed upstreams to every request.

Options (`bench run --env NAME=VALUE`): `DSPY_RLM_SIGNATURE`,
`DSPY_RLM_MAX_ITERATIONS`, `DSPY_RLM_MAX_LLM_CALLS`,
`DSPY_RLM_MAX_OUTPUT_CHARS`, `DSPY_RLM_TOOL_TIMEOUT` (seconds).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import bf_events

# Deno's cache, filled with Pyodide when the agent was installed.
DENO_CACHE = Path("/opt/benchflow/agents/dspy-rlm/deno")

FIX = "run dspy-rlm with a provider model, such as -m vercel/deepseek/deepseek-v4-flash-0731"


def int_option(name: str, default: int) -> int:
    value = bf_events.option(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise bf_events.SetupError(f"{name} must be a whole number, not `{value}`", f"pass --env {name}=<number>")


class Tools:
    """The seven tools of the dspy-rlm agent, acting on this machine in the working folder."""

    def __init__(self, events: bf_events.Events, cwd: Path, timeout: int) -> None:
        self.events = events
        self.cwd = cwd
        self.timeout = timeout

    def _shell(self, command: str, cwd: str | None = None) -> tuple[int, str, str]:
        try:
            done = subprocess.run(
                ["bash", "-c", command],
                cwd=cwd or self.cwd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout,
                stdin=subprocess.DEVNULL,
            )
            return done.returncode, done.stdout, done.stderr
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return 124, out, f"timed out after {self.timeout}s"
        except OSError as exc:
            return 127, "", str(exc)

    @staticmethod
    def _format(code: int, stdout: str, stderr: str, empty: str = "(no output)") -> str:
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr] {stderr}")
        if code != 0:
            parts.append(f"[exit code {code}]")
        return "\n".join(parts) if parts else empty

    def _path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.cwd / p

    def exec_command(self, command: str, cwd: str | None = None) -> str:
        """Execute a shell command in the environment. Returns stdout+stderr."""
        return self._format(*self._shell(command, cwd))

    def read_file(self, path: str) -> str:
        """Read a file from the environment. Returns file contents."""
        try:
            return self._path(path).read_text(errors="replace")
        except OSError as exc:
            return f"[error] {exc.strerror or 'file not found'}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the environment."""
        try:
            p = self._path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return "ok"
        except OSError as exc:
            return f"[error] {exc.strerror or 'write failed'}"

    def list_directory(self, path: str = ".") -> str:
        """List files and directories. Returns ls -la output."""
        code, out, err = self._shell(f"ls -la -- {quote(path)}")
        return out if code == 0 else f"[error] {err or 'directory not found'}"

    def find_files(self, pattern: str, path: str = ".") -> str:
        """Find files matching a glob pattern."""
        code, out, err = self._shell(f"find {quote(path)} -name {quote(pattern)} -type f 2>/dev/null | head -50")
        if code != 0:
            return f"[error] {err or 'find failed'}"
        return out or "(no matches)"

    def search_content(self, pattern: str, path: str = ".", file_glob: str = "") -> str:
        """Search file contents with grep. Returns matching lines."""
        glob_flag = f"--include={quote(file_glob)}" if file_glob else ""
        code, out, _ = self._shell(f"grep -rn {glob_flag} -- {quote(pattern)} {quote(path)} 2>/dev/null | head -100")
        return out if code == 0 and out else "(no matches)"

    def apply_patch(self, patch: str) -> str:
        """Apply a unified diff patch. The patch should be in unified diff format."""
        try:
            done = subprocess.run(
                ["patch", "-p1", "--no-backup-if-mismatch"],
                cwd=self.cwd,
                input=patch,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return self._format(done.returncode, done.stdout, done.stderr, empty="patch applied successfully")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"[error] {exc}"

    def recorded(self) -> list[Any]:
        """The tools, each reporting its call and result as events."""
        import functools

        def wrap(fn: Any) -> Any:
            @functools.wraps(fn)
            def tool(*args: Any, **kwargs: Any) -> str:
                names = list(fn.__code__.co_varnames[1 : fn.__code__.co_argcount])
                arguments = {**dict(zip(names, args)), **kwargs}
                call = self.events.tool_call(None, fn.__name__, arguments)
                result = fn(*args, **kwargs)
                failed = result.startswith("[error]") or "[exit code " in result
                self.events.tool_result(call, result, failed=failed)
                return result

            return tool

        return [
            wrap(getattr(self, name))
            for name in (
                "exec_command",
                "read_file",
                "write_file",
                "list_directory",
                "find_files",
                "search_content",
                "apply_patch",
            )
        ]


def quote(word: str) -> str:
    import shlex

    return shlex.quote(word)


def use_deno_cache() -> None:
    """Give Deno a writable copy of the Pyodide cache made at install time."""
    if "DENO_DIR" in os.environ:
        return
    target = Path.home() / ".cache" / "benchflow-deno"
    if DENO_CACHE.is_dir() and not target.exists():
        shutil.copytree(DENO_CACHE, target, symlinks=True)
    os.environ["DENO_DIR"] = str(target)


def main(events: bf_events.Events) -> str | None:
    import dspy
    from dspy.utils.callback import BaseCallback

    lm_name = bf_events.option("BENCHFLOW_DSPY_LM")
    base_url = bf_events.option("OPENAI_BASE_URL")
    if not lm_name or not base_url:
        raise bf_events.SetupError("dspy-rlm reaches its model through BenchFlow's provider forwarder, which was not set up", FIX)
    signature = bf_events.option("DSPY_RLM_SIGNATURE") or "instruction, file_tree -> solution"
    inputs = [f.strip().split(":", 1)[0].strip() for f in signature.split("->", 1)[0].split(",") if f.strip()]
    if len(inputs) != 2:
        raise bf_events.SetupError(
            f"the dspy-rlm signature must have exactly two inputs (the instruction and the file tree), not {signature!r}",
            "pass --env DSPY_RLM_SIGNATURE='instruction, file_tree -> solution' or another two-input signature",
        )
    workdir = bf_events.workdir()
    tools = Tools(events, workdir, int_option("DSPY_RLM_TOOL_TIMEOUT", 30))
    use_deno_cache()

    events.start("dspy-rlm", "dspy", bf_events.package_version("dspy"), bf_events.model())

    tree = subprocess.run(
        "find . -maxdepth 3 -type f | head -200",
        shell=True,
        cwd=workdir,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
    ).stdout.strip() or "(empty)"

    lm_kwargs: dict[str, Any] = {}
    if effort := bf_events.effort():
        lm_kwargs["reasoning_effort"] = effort
    lm = dspy.LM(
        lm_name,
        api_base=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        max_tokens=16_000,
        cache=False,
        **lm_kwargs,
    )
    totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    class Report(BaseCallback):
        """Reasoning as thoughts, and usage after every model call."""

        def __init__(self) -> None:
            self.actions: set[str] = set()

        def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
            if instance is rlm.generate_action:
                self.actions.add(call_id)

        def on_module_end(self, call_id: str, outputs: Any, exception: Exception | None = None) -> None:
            if call_id in self.actions and outputs is not None:
                self.actions.discard(call_id)
                events.thought(getattr(outputs, "reasoning", None))

        def on_lm_end(self, call_id: str, outputs: Any, exception: Exception | None = None) -> None:
            if not lm.history:
                return
            entry = lm.history[-1]
            usage = entry.get("usage") or {}
            totals["input_tokens"] += int(usage.get("prompt_tokens") or 0)
            totals["output_tokens"] += int(usage.get("completion_tokens") or 0)
            # The gateway's own price comes in usage.cost; LiteLLM's estimate
            # in `cost`, when it knows the model.
            cost = usage.get("cost") if usage.get("cost") is not None else entry.get("cost")
            try:
                totals["cost_usd"] += float(cost or 0)
            except (TypeError, ValueError):
                pass
            events.set_usage(**totals)

    class RecordedRLM(dspy.RLM):
        """dspy.RLM that reports each REPL step as a `python` tool call."""

        def _execute_code(self, repl: Any, code: str, input_args: dict[str, Any]) -> Any:
            call = events.tool_call(None, "python", {"code": code}, title="python")
            result = super()._execute_code(repl, code, input_args)
            output = "\n".join(map(str, result)) if isinstance(result, list) else str(result)
            events.tool_result(call, output, failed=output.startswith("[Error]"))
            return result

    dspy.configure(lm=lm, track_usage=True, callbacks=[Report()])
    rlm = RecordedRLM(
        signature=signature,
        max_iters=int_option("DSPY_RLM_MAX_ITERATIONS", 20),
        max_llm_calls=int_option("DSPY_RLM_MAX_LLM_CALLS", 50),
        max_output_chars=int_option("DSPY_RLM_MAX_OUTPUT_CHARS", 10_000),
        tools=tools.recorded(),
    )
    prediction = rlm(**{inputs[0]: bf_events.instruction(), inputs[1]: tree})
    events.set_usage(**totals)
    fields = list(prediction.keys())
    solution = str(prediction[fields[0]]) if fields else str(prediction)
    events.message(solution)
    return solution


if __name__ == "__main__":
    bf_events.run(main)
