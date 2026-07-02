import csv
import json
from pathlib import Path

from benchflow._utils.benchmark_repos import ResolvedSource
from benchflow.adapters.source import adapt_resolved_source_if_needed
from benchflow.task import Task


def _resolved(root: Path, *, repo: str, path: str, sha: str = "abc123") -> ResolvedSource:
    source_root = root / path if path else root
    return ResolvedSource(
        path=source_root,
        provenance={
            "type": "github",
            "repo": repo,
            "requested_ref": None,
            "resolved_sha": sha,
            "path": path,
            "local_path": str(source_root),
            "dirty": False,
            "file_hashes": {},
        },
    )


def test_mcp_atlas_source_adapter_materializes_native_tasks(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards cd8e250b MCP Atlas adapter work against zero-task sources."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "mcp-atlas"
    (repo / ".git").mkdir(parents=True)
    source = repo / "services" / "mcp_eval"
    source.mkdir(parents=True)
    with (source / "sample_tasks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["TASK", "ENABLED_TOOLS", "PROMPT", "TRAJECTORY", "GTFA_CLAIMS"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "TASK": "atlas-task-1",
                "ENABLED_TOOLS": json.dumps(["search_query", "fetch_read"]),
                "PROMPT": "Answer using tools.",
                "TRAJECTORY": "[]",
                "GTFA_CLAIMS": json.dumps(["The answer is correct."]),
            }
        )

    adapted = adapt_resolved_source_if_needed(
        _resolved(repo, repo="scaleapi/mcp-atlas", path="services/mcp_eval")
    )

    task_dir = adapted.path / "atlas-task-1"
    task = Task(task_dir)
    assert adapted.path != source
    assert task.config.task is not None
    assert task.config.task.name == "mcp-atlas/atlas-task-1"
    assert task.config.environment.mcp_servers[0].tools == [
        "search_query",
        "fetch_read",
    ]
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    compose = (task_dir / "environment" / "docker-compose.yaml").read_text()
    bridge = (task_dir / "environment" / "mcp_bridge.py").read_text()
    assert "fastmcp==3.4.2" in dockerfile
    assert "COPY mcp_bridge.py /mcp_bridge.py" in dockerfile
    assert "ENABLED_SERVERS: \"fetch,search\"" in compose
    assert "python /mcp_bridge.py" in compose
    assert "MCP Atlas bridge registered" in bridge
    assert adapted.provenance["adapter"]["name"] == "mcp-atlas"


def test_toolathlon_source_adapter_materializes_mcp_and_setup(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards cd8e250b Toolathlon adapter work against raw finalpool task dirs."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Toolathlon"
    (repo / ".git").mkdir(parents=True)
    (repo / "global_preparation").mkdir()
    (repo / "configs").mkdir()
    (repo / "configs" / "users_data.json").write_text("{}")
    mcp_dir = repo / "configs" / "mcp_servers"
    mcp_dir.mkdir()
    (mcp_dir / "filesystem.yaml").write_text(
        """
type: stdio
name: filesystem
params:
  command: npx
  args:
    - "-y"
    - "@modelcontextprotocol/server-filesystem"
    - "${agent_workspace}"
  cwd: "${agent_workspace}"
"""
    )
    (mcp_dir / "word.yaml").write_text(
        """
type: stdio
name: word
params:
  command: uvx
  args:
    - "--from"
    - "office-word-mcp-server"
    - "word_mcp_server"
  cwd: "${agent_workspace}"
"""
    )
    task_dir = repo / "tasks" / "finalpool" / "arrange-workspace"
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "docs" / "agent_system_prompt.md").write_text(
        "Accessible workspace directory: !!<<<<||||workspace_dir||||>>>>!!\n"
    )
    (task_dir / "docs" / "task.md").write_text("Arrange the workspace.\n")
    (task_dir / "task_config.json").write_text(
        json.dumps(
            {"needed_mcp_servers": ["filesystem", "word"], "needed_local_tools": []}
        )
    )

    adapted = adapt_resolved_source_if_needed(
        _resolved(repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool", sha="d" * 40)
    )

    generated = adapted.path / "arrange-workspace"
    task = Task(generated)
    assert task.config.task is not None
    assert task.config.task.name == "toolathlon/arrange-workspace"
    assert task.config.environment.workdir == "/workspace/agent_workspace"
    assert task.config.environment.setup_commands
    assert task.config.environment.mcp_servers[0].exclude_tags == [
        "__benchflow_exclude_no_tools__"
    ]
    word = task.config.environment.mcp_servers[1]
    assert word.command == "/usr/local/bin/uvx"
    assert word.args == [
        "--from",
        "office-word-mcp-server==1.1.11",
        "word_mcp_server",
    ]
    assert word.cwd == "/workspace/agent_workspace"
    setup_command = task.config.environment.setup_commands[0].command
    dockerfile = (generated / "environment" / "Dockerfile").read_text()
    task_toml = (generated / "task.toml").read_text()
    test_sh = (generated / "tests" / "test.sh").read_text()
    assert "lockon0927/toolathlon-task-image" in dockerfile
    assert "rsync -a --delete" not in dockerfile
    assert "rsync -a --exclude .git /tmp/toolathlon-src/ /workspace/" in dockerfile
    assert "global_configs_example.py" in dockerfile
    assert "global_configs.py" in dockerfile
    assert "cp \"$(command -v uv)\" /usr/local/bin/uv" in dockerfile
    assert "chmod -R a+rwX /workspace/utils/local_servers" in dockerfile
    assert "chmod -R a+rwX /workspace/agent_workspace" in task_toml
    assert 'chmod -R go-rwx "$private"' in setup_command
    assert "BenchFlow Toolathlon verifier setup error" in test_sh
    assert "toolathlon_evaluator.log" in test_sh
    assert "evaluator crashed" in test_sh
    assert "Traceback (most recent call last):" in test_sh
    assert "/usr/local/bin/uv run python" in test_sh


def test_toolathlon_gym_adapter_normalizes_postgres_env(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards cd8e250b Toolathlon-GYM adapter work against stale PG service names."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "toolathlon_gym"
    (repo / ".git").mkdir(parents=True)
    (repo / "db").mkdir()
    (repo / "db" / "init.sql.gz").write_bytes(b"")
    mcp_dir = repo / "configs" / "mcp_servers"
    mcp_dir.mkdir(parents=True)
    (mcp_dir / "snowflake.yaml").write_text(
        """
type: stdio
name: snowflake
params:
  command: uv
  args:
    - run
    - mcp_snowflake_server
  env:
    PG_HOST: toolathlon_pg
    PG_USER: postgres
    PG_PASSWORD: postgres
    PG_DATABASE: old_db
"""
    )
    task_dir = repo / "tasks" / "finalpool" / "salary-report"
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "docs" / "agent_system_prompt.md").write_text(
        "Workspace: !!<<<<||||workspace_dir||||>>>>!!\n"
    )
    (task_dir / "docs" / "task.md").write_text("Create the report.\n")
    (task_dir / "task_config.json").write_text(
        json.dumps({"needed_mcp_servers": ["snowflake"], "needed_local_tools": []})
    )

    adapted = adapt_resolved_source_if_needed(
        _resolved(
            repo,
            repo="eigent-ai/toolathlon_gym",
            path="tasks/finalpool",
            sha="e" * 40,
        )
    )

    generated = adapted.path / "salary-report"
    task = Task(generated)
    env = task.config.environment.mcp_servers[0].env
    assert env["PG_HOST"] == "postgres"
    assert env["PG_USER"] == "eigent"
    assert env["PG_PASSWORD"] == "camel"
    assert env["PG_DATABASE"] == "toolathlon_gym"
    assert "chmod -R a+rwX /opt/local_servers" in (
        generated / "environment" / "Dockerfile"
    ).read_text()
    assert "chmod -R a+rwX /workspace/agent_workspace" in (
        generated / "task.toml"
    ).read_text()
    setup_command = task.config.environment.setup_commands[0].command
    assert 'chmod -R go-rwx "$private"' in setup_command
    assert "postgres:" in (generated / "environment" / "docker-compose.yaml").read_text()


def test_toolathlon_adapter_skips_tasks_with_missing_repo_configs(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards PR #878 against materializing Toolathlon tasks missing credentials."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Toolathlon"
    (repo / ".git").mkdir(parents=True)
    (repo / "global_preparation").mkdir()
    (repo / "configs" / "mcp_servers").mkdir(parents=True)
    (repo / "configs" / "users_data.json").write_text("{}")
    (repo / "configs" / "mcp_servers" / "filesystem.yaml").write_text(
        """
type: stdio
name: filesystem
params:
  command: npx
  args:
    - "-y"
    - "@modelcontextprotocol/server-filesystem"
    - "${agent_workspace}"
"""
    )
    task_dir = repo / "tasks" / "finalpool" / "ab-testing"
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "preprocess").mkdir()
    (task_dir / "docs" / "agent_system_prompt.md").write_text(
        "Workspace: !!<<<<||||workspace_dir||||>>>>!!\n"
    )
    (task_dir / "docs" / "task.md").write_text("Do the task.\n")
    (task_dir / "task_config.json").write_text(
        json.dumps({"needed_mcp_servers": ["filesystem"], "needed_local_tools": []})
    )
    (task_dir / "preprocess" / "main.py").write_text(
        "open('configs/gcp-service_account.keys.json').read()\n"
    )

    adapted = adapt_resolved_source_if_needed(
        _resolved(repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool")
    )

    assert not (adapted.path / "ab-testing").exists()
    skipped = json.loads(
        (adapted.path / ".benchflow-source-adapter-skipped.json").read_text()
    )
    assert skipped == {
        "skipped": [
            {
                "reason": "references missing repo config: "
                "configs/gcp-service_account.keys.json",
                "task_id": "ab-testing",
            }
        ]
    }
