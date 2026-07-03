import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from benchflow._utils.benchmark_repos import ResolvedSource
from benchflow.adapters.source import adapt_resolved_source_if_needed
from benchflow.task import Task


def _resolved(
    root: Path, *, repo: str, path: str, sha: str = "abc123"
) -> ResolvedSource:
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
    assert 'ENABLED_SERVERS: "fetch,search"' in compose
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
        _resolved(
            repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool", sha="d" * 40
        )
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
    # The real server is wrapped in the container launcher so ${token.X} in
    # argv/env resolves at spawn time against the per-task token file.
    word = task.config.environment.mcp_servers[1]
    assert word.command == "/usr/bin/python3"
    assert word.args == [
        "/workspace/.toolathlon/toolathlon_container.py",
        "launch",
        "/usr/local/bin/uvx",
        "--from",
        "office-word-mcp-server==1.1.11",
        "word_mcp_server",
    ]
    assert word.cwd == "/workspace/agent_workspace"
    assert (
        word.env["TOOLATHLON_TASK_DIR"]
        == "/workspace/tasks/finalpool/arrange-workspace"
    )
    # First setup command stages the container helper and writes the global
    # token_key_session.py; the preprocess command runs last.
    token_setup = task.config.environment.setup_commands[0].command
    assert "toolathlon_container.py write-config" in token_setup
    assert "TOOLATHLON_CONTAINER_MODULE_B64" in token_setup
    setup_command = task.config.environment.setup_commands[-1].command
    dockerfile = (generated / "environment" / "Dockerfile").read_text()
    task_toml = (generated / "task.toml").read_text()
    test_sh = (generated / "tests" / "test.sh").read_text()
    assert "lockon0927/toolathlon-task-image" in dockerfile
    assert "rsync -a --delete" not in dockerfile
    assert "rsync -a --exclude .git /tmp/toolathlon-src/ /workspace/" in dockerfile
    assert "global_configs_example.py" in dockerfile
    assert "global_configs.py" in dockerfile
    assert 'cp "$(command -v uv)" /usr/local/bin/uv' in dockerfile
    assert "chmod -R a+rwX /workspace/utils/local_servers" in dockerfile
    assert "chmod -R a+rwX /workspace/agent_workspace" in task_toml
    assert 'chmod -R go-rwx "$private"' in setup_command
    assert "BenchFlow Toolathlon verifier setup error" in test_sh
    assert "toolathlon_evaluator.log" in test_sh
    # A non-zero evaluator exit is a task fail (reward 0), matching upstream's
    # exit-code scoring; only a broken evaluator ENVIRONMENT escalates.
    assert "evaluator environment failure" in test_sh
    assert "ModuleNotFoundError|ImportError|PermissionError" in test_sh
    assert "Traceback (most recent call last):" not in test_sh
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
    assert (
        "chmod -R a+rwX /opt/local_servers"
        in (generated / "environment" / "Dockerfile").read_text()
    )
    assert (
        "chmod -R a+rwX /workspace/agent_workspace"
        in (generated / "task.toml").read_text()
    )
    setup_command = task.config.environment.setup_commands[-1].command
    assert 'chmod -R go-rwx "$private"' in setup_command
    assert (
        "postgres:" in (generated / "environment" / "docker-compose.yaml").read_text()
    )


def test_toolathlon_adapter_materializes_tasks_with_missing_repo_configs(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards PR #878 against dropping Toolathlon credential-backed tasks."""
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
    (repo / "configs" / "mcp_servers" / "google_sheet.yaml").write_text(
        """
type: stdio
name: google_sheet
params:
  command: uvx
  args:
    - "mcp-google-sheets"
  env:
    CREDENTIALS_PATH: "${token.google_oauth2_credentials_path}"
    TOKEN_PATH: "${token.google_oauth2_token_path}"
"""
    )
    (repo / "configs" / "mcp_servers" / "google_calendar.yaml").write_text(
        """
type: stdio
name: google_calendar
params:
  command: npx
  args:
    - "-y"
    - "@gongrzhe/server-calendar-autoauth-mcp"
  cwd: "${agent_workspace}"
"""
    )
    (repo / "configs" / "mcp_servers" / "google_forms.yaml").write_text(
        """
type: stdio
name: google_forms
params:
  command: node
  args:
    - "${local_servers_paths}/google-forms-mcp/build/index.js"
  env:
    GOOGLE_CLIENT_ID: "${token.google_client_id}"
    GOOGLE_CLIENT_SECRET: "${token.google_client_secret}"
    GOOGLE_REFRESH_TOKEN: "${token.google_refresh_token}"
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
    sheet_task = repo / "tasks" / "finalpool" / "sheet-only"
    (sheet_task / "docs").mkdir(parents=True)
    (sheet_task / "docs" / "agent_system_prompt.md").write_text(
        "Workspace: !!<<<<||||workspace_dir||||>>>>!!\n"
    )
    (sheet_task / "docs" / "task.md").write_text("Update the sheet.\n")
    (sheet_task / "task_config.json").write_text(
        json.dumps({"needed_mcp_servers": ["google_sheet"], "needed_local_tools": []})
    )
    calendar_task = repo / "tasks" / "finalpool" / "calendar-only"
    (calendar_task / "docs").mkdir(parents=True)
    (calendar_task / "docs" / "agent_system_prompt.md").write_text(
        "Workspace: !!<<<<||||workspace_dir||||>>>>!!\n"
    )
    (calendar_task / "docs" / "task.md").write_text("Update the calendar.\n")
    (calendar_task / "task_config.json").write_text(
        json.dumps(
            {"needed_mcp_servers": ["google_calendar"], "needed_local_tools": []}
        )
    )
    forms_task = repo / "tasks" / "finalpool" / "forms-only"
    (forms_task / "docs").mkdir(parents=True)
    (forms_task / "docs" / "agent_system_prompt.md").write_text(
        "Workspace: !!<<<<||||workspace_dir||||>>>>!!\n"
    )
    (forms_task / "docs" / "task.md").write_text("Update the form.\n")
    (forms_task / "task_config.json").write_text(
        json.dumps({"needed_mcp_servers": ["google_forms"], "needed_local_tools": []})
    )

    adapted = adapt_resolved_source_if_needed(
        _resolved(repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool")
    )

    generated = adapted.path / "ab-testing"
    task = Task(generated)
    assert task.config.metadata["required_credential_files"] == [
        "configs/gcp-service_account.keys.json"
    ]
    assert task.config.metadata["credential_env_options"] == [
        {
            "file": "configs/gcp-service_account.keys.json",
            "env": "TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON",
            "base64_env": "TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON_B64",
        }
    ]
    assert not (adapted.path / ".benchflow-source-adapter-skipped.json").exists()
    # setup_commands: [token-setup, credential-inject, preprocess].
    assert len(task.config.environment.setup_commands) == 3
    assert (
        "toolathlon_container.py write-config"
        in task.config.environment.setup_commands[0].command
    )
    credential_setup = task.config.environment.setup_commands[1]
    assert credential_setup.env == {
        "TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON": "${TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON:-}",
        "TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON_B64": "${TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON_B64:-}",
    }
    assert "BenchFlow Toolathlon credential setup error" in credential_setup.command
    assert "configs/gcp-service_account.keys.json" in credential_setup.command
    assert "target.chmod(0o644)" in credential_setup.command
    assert "/home/agent" not in credential_setup.command
    assert ".gmail-mcp" not in credential_setup.command
    # ${token.X} env stays literal at materialization; the launcher resolves it
    # at spawn time against the container's token_key_session files.
    sheet = Task(adapted.path / "sheet-only")
    assert sheet.config.metadata["required_credential_files"] == [
        "configs/google_credentials.json"
    ]
    sheet_server = sheet.config.environment.mcp_servers[0]
    assert sheet_server.command == "/usr/bin/python3"
    assert sheet_server.args[:3] == [
        "/workspace/.toolathlon/toolathlon_container.py",
        "launch",
        "/usr/local/bin/uvx",
    ]
    assert sheet_server.env["CREDENTIALS_PATH"] == (
        "${token.google_oauth2_credentials_path}"
    )
    assert sheet_server.env["TOKEN_PATH"] == "${token.google_oauth2_token_path}"
    assert (
        sheet_server.env["TOOLATHLON_TASK_DIR"]
        == "/workspace/tasks/finalpool/sheet-only"
    )
    calendar = Task(adapted.path / "calendar-only")
    assert calendar.config.metadata["required_credential_files"] == [
        "configs/gcp-oauth.keys.json",
        "configs/google_credentials.json",
    ]
    calendar_server = calendar.config.environment.mcp_servers[0]
    assert "HOME" not in calendar_server.env
    assert (
        calendar_server.env["CALENDAR_OAUTH_PATH"]
        == "/workspace/configs/gcp-oauth.keys.json"
    )
    assert calendar_server.env["CALENDAR_CREDENTIALS_PATH"] == (
        "/workspace/agent_workspace/.toolathlon/calendar_credentials.json"
    )
    workspace = tmp_path / "workspace"
    (workspace / "configs").mkdir(parents=True)
    google_payload = {
        "client_id": "client",
        "client_secret": "secret",
        "refresh_token": "refresh",
    }
    oauth_payload = {"installed": {"client_id": "oauth-client"}}
    (workspace / "configs" / "google_credentials.json").write_text(
        json.dumps(google_payload)
    )
    (workspace / "configs" / "gcp-oauth.keys.json").write_text(
        json.dumps(oauth_payload)
    )
    calendar_command = calendar.config.environment.setup_commands[1].command.replace(
        "/usr/local/bin/uv run python", sys.executable
    )
    result = subprocess.run(
        calendar_command,
        cwd=workspace,
        env={"PATH": os.environ["PATH"]},
        shell=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(
            (
                workspace
                / "agent_workspace"
                / ".toolathlon"
                / "google_credentials.json"
            ).read_text()
        )
        == google_payload
    )
    assert (
        json.loads(
            (
                workspace
                / "agent_workspace"
                / ".toolathlon"
                / "calendar_credentials.json"
            ).read_text()
        )
        == google_payload
    )
    assert not (workspace / ".mcp-home").exists()
    forms = Task(adapted.path / "forms-only")
    assert "required_credential_files" not in forms.config.metadata

def _run_container_helper(
    workspace: Path, argv: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    module = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "benchflow"
        / "adapters"
        / "_toolathlon_container.py"
    )
    full_env = {"PATH": os.environ["PATH"], "TOOLATHLON_WORKSPACE": str(workspace)}
    full_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(module), *argv],
        env=full_env,
        text=True,
        capture_output=True,
        timeout=30,
    )

def test_toolathlon_container_write_config_bakes_secrets(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    result = _run_container_helper(
        tmp_path,
        ["write-config"],
        {
            "TOOLATHLON_GCP_PROJECT_ID": "proj-123",
            "TOOLATHLON_MAPS_API_KEY": "maps-abc",
            "TOOLATHLON_GITHUB_TOKEN": "gho_xyz",
        },
    )
    assert result.returncode == 0, result.stderr
    generated = (tmp_path / "configs" / "token_key_session.py").read_text()
    ns: dict[str, object] = {}
    exec(generated, ns)
    tokens = ns["all_token_key_session"]
    assert tokens["gcp_project_id"] == "proj-123"
    assert tokens["google_cloud_console_api_key"] == "maps-abc"
    assert tokens["github_token"] == "gho_xyz"
    # Unset secrets fall back to their example defaults, not the literal env ref.
    assert tokens["huggingface_token"] == "XX"
    assert tokens["github_read_only"] == "1"

def test_toolathlon_container_launch_resolves_tokens(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "token_key_session.py").write_text(
        "all_token_key_session = {"
        "'gcp_project_id': 'global-proj',"
        "'google_cloud_allowed_bigquery_datasets': 'null'}\n"
    )
    task_dir = tmp_path / "tasks" / "finalpool" / "demo"
    task_dir.mkdir(parents=True)
    # Per-task file overrides the global dataset and derives a value from a file
    # written by "preprocess" — exactly the runtime-token case.
    (task_dir / "files").mkdir()
    (task_dir / "files" / "folder_id.txt").write_text("FID-999")
    (task_dir / "token_key_session.py").write_text(
        "import os\n"
        "from addict import Dict\n"
        "_here = os.path.dirname(__file__)\n"
        "with open(os.path.join(_here, 'files', 'folder_id.txt')) as f:\n"
        "    _fid = f.read().strip()\n"
        "all_token_key_session = Dict("
        "google_cloud_allowed_bigquery_datasets='demo_ds',"
        "google_sheets_folder_id=_fid)\n"
    )
    result = _run_container_helper(
        tmp_path,
        [
            "launch",
            "/bin/echo",
            "--project=${token.gcp_project_id}",
            "--dataset=${token.google_cloud_allowed_bigquery_datasets}",
            "--folder=${token.google_sheets_folder_id}",
        ],
        {"TOOLATHLON_TASK_DIR": str(task_dir)},
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert "--project=global-proj" in out  # from global
    assert "--dataset=demo_ds" in out  # per-task override wins
    assert "--folder=FID-999" in out  # runtime file value

def test_toolathlon_container_launch_resolves_env(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "token_key_session.py").write_text(
        "all_token_key_session = {'github_token': 'gho_secret'}\n"
    )
    result = _run_container_helper(
        tmp_path,
        ["launch", "/usr/bin/env"],
        {
            "TOOLATHLON_TASK_DIR": str(tmp_path),
            "GITHUB_TOKEN": "${token.github_token}",
        },
    )
    assert result.returncode == 0, result.stderr
    assert "GITHUB_TOKEN=gho_secret" in result.stdout

def test_toolathlon_container_launch_ensures_dirs(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "token_key_session.py").write_text(
        "all_token_key_session = {}\n"
    )
    storage = tmp_path / "agent_workspace" / "arxiv_local_storage"
    assert not storage.exists()
    result = _run_container_helper(
        tmp_path,
        ["launch", "/bin/echo", "ok"],
        {
            "TOOLATHLON_TASK_DIR": str(tmp_path),
            "TOOLATHLON_ENSURE_DIRS": str(storage),
        },
    )
    assert result.returncode == 0, result.stderr
    assert storage.is_dir()

def test_toolathlon_arxiv_server_declares_ensure_dirs(tmp_path: Path, monkeypatch) -> None:
    """The arxiv_local server's --storage-path is passed to the launcher as a
    directory to pre-create, so the evaluator's listdir cannot crash."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Toolathlon"
    (repo / ".git").mkdir(parents=True)
    (repo / "global_preparation").mkdir()
    (repo / "configs" / "mcp_servers").mkdir(parents=True)
    (repo / "configs" / "users_data.json").write_text("{}")
    (repo / "configs" / "mcp_servers" / "arxiv_local.yaml").write_text(
        """
type: stdio
name: arxiv_local
params:
  command: uv
  args:
    - "run"
    - "python"
    - "-m"
    - "utils.local_servers.arxiv_local_wrapper"
    - "--storage-path"
    - "${agent_workspace}/arxiv_local_storage"
  cwd: "${agent_workspace}"
"""
    )
    task_dir = repo / "tasks" / "finalpool" / "find-alita-paper"
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "docs" / "agent_system_prompt.md").write_text("W: !!<<<<||||workspace_dir||||>>>>!!\n")
    (task_dir / "docs" / "task.md").write_text("Find the paper.\n")
    (task_dir / "task_config.json").write_text(
        json.dumps({"needed_mcp_servers": ["arxiv_local"], "needed_local_tools": []})
    )
    adapted = adapt_resolved_source_if_needed(
        _resolved(repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool")
    )
    task = Task(adapted.path / "find-alita-paper")
    arxiv = task.config.environment.mcp_servers[0]
    assert (
        arxiv.env["TOOLATHLON_ENSURE_DIRS"]
        == "/workspace/agent_workspace/arxiv_local_storage"
    )
