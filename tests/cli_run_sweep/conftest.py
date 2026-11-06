"""Offline CLI regression fixtures."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
import yaml
from typer.testing import CliRunner

from benchflow.cli import main
from benchflow.evaluation import Evaluation


@pytest.fixture
def tasks(tmp_path):
    root = tmp_path / "tasks"
    source = Path(__file__).parents[1] / "examples/hello-world-task"
    for name in ("task-a", "task-b", "task-c"):
        shutil.copytree(source, root / name)
    return root


@pytest.fixture
def yaml_run(tasks, tmp_path, monkeypatch, capsys):
    def invoke(
        raw,
        args,
        command=("eval", "run"),
        option="--run-config",
        *,
        layout="native",
        on_run=None,
        model_agent=False,
    ):
        path = tmp_path / "run.yaml"
        source = (
            {"tasks_dir": str(tasks), "agent": "nop"}
            if layout == "native"
            else {"datasets": [{"path": str(tasks)}], "agents": [{"name": "nop"}]}
        )
        if model_agent:
            if layout == "native":
                source.update(agent="codex", model="openai/gpt-test")
            else:
                source["agents"] = [{"name": "codex", "model_name": "openai/gpt-test"}]
        path.write_text(
            yaml.safe_dump({**source, "jobs_dir": str(tmp_path / "jobs"), **raw})
        )
        seen = []

        async def capture(self):
            seen.append(self._config)
            if on_run is not None:
                on_run(self)
            raise ValueError("fixture captured effective config")

        monkeypatch.setattr(Evaluation, "run", capture)
        if isinstance(command, str):
            # Exercise the exported Python callbacks without manufacturing a
            # Click context: supplied None and an omitted keyword differ here.
            kwargs = {"sandbox_user": args[1]} if args else {}
            capsys.readouterr()
            try:
                getattr(main, command)(config_file=path, **kwargs)
            except typer.Exit as exc:
                exception = SystemExit(exc.exit_code)
            else:
                exception = None
            output = capsys.readouterr()
            result = SimpleNamespace(
                exception=exception, output=output.out + output.err
            )
        else:
            result = CliRunner().invoke(main.app, [*command, option, str(path), *args])
        return result, seen

    return invoke
