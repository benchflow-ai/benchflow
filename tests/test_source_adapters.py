import csv
import imaplib
import json
import os
import smtplib
import subprocess
import sys
import time
from email.mime.text import MIMEText
from pathlib import Path

import yaml

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
    assert (
        "git fetch --depth 1 origin dddddddddddddddddddddddddddddddddddddddd"
        in dockerfile
    )
    assert "git checkout FETCH_HEAD" in dockerfile
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
    assert "ast.walk" in setup_command
    assert "ast.walk" in test_sh
    assert "/usr/local/bin/uv run python" in test_sh
    assert "toolathlon_container.py write-task-tokens" in setup_command
    assert setup_command.index("write-task-tokens") < setup_command.index(
        'chmod -R go-rwx "$private"'
    )


def _run_toolathlon_setup_script(
    tmp_path: Path, preprocess_source: str
) -> subprocess.CompletedProcess:
    from benchflow.adapters._toolathlon import (
        _toolathlon_container_module_source,
        _toolathlon_setup_command,
    )

    workspace = tmp_path / "workspace"
    task_dir = workspace / "tasks" / "finalpool" / "demo"
    preprocess_dir = task_dir / "preprocess"
    preprocess_dir.mkdir(parents=True)
    (workspace / "agent_workspace").mkdir(parents=True)
    for package in (
        workspace / "tasks",
        workspace / "tasks" / "finalpool",
        task_dir,
        preprocess_dir,
    ):
        (package / "__init__.py").write_text("")
    (preprocess_dir / "main.py").write_text(preprocess_source)
    helper = workspace / ".toolathlon" / "toolathlon_container.py"
    helper.parent.mkdir()
    helper.write_text(_toolathlon_container_module_source())
    helper.chmod(0o755)

    script = _toolathlon_setup_command(task_name="demo", variant="gym")
    script = script.replace("/workspace", str(workspace))
    script = script.replace("/opt/venv/bin/python3", sys.executable)
    script_path = tmp_path / "setup.sh"
    script_path.write_text(script)

    env = os.environ.copy()
    env["TOOLATHLON_WORKSPACE"] = str(workspace)

    return subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )


def test_toolathlon_preprocess_ignores_launch_time_in_comments(
    tmp_path: Path,
) -> None:
    """Guards PR #887: launch_time comments do not receive the CLI flag."""
    result = _run_toolathlon_setup_script(
        tmp_path,
        "import argparse\n"
        "# launch_time is documented here but not accepted by this script.\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--agent_workspace', required=True)\n"
        "parser.parse_args()\n",
    )
    assert result.returncode == 0, result.stderr
    assert "unrecognized arguments: --launch_time" not in result.stderr


def _run_toolathlon_verifier_script(
    tmp_path: Path, evaluator_source: str
) -> subprocess.CompletedProcess:
    from benchflow.adapters._toolathlon import _toolathlon_test_sh

    workspace = tmp_path / "workspace"
    logs = tmp_path / "logs" / "verifier"
    eval_dir = workspace / "tasks" / "finalpool" / "demo" / "evaluation"
    eval_dir.mkdir(parents=True)
    (workspace / "agent_workspace").mkdir(parents=True)
    (workspace / "tasks" / "finalpool" / "demo" / "groundtruth_workspace").mkdir()
    for package in (
        workspace / "tasks",
        workspace / "tasks" / "finalpool",
        workspace / "tasks" / "finalpool" / "demo",
        eval_dir,
    ):
        (package / "__init__.py").write_text("")
    (eval_dir / "main.py").write_text(evaluator_source)

    script = _toolathlon_test_sh(task_name="demo", variant="gym")
    script = script.replace("/workspace", str(workspace))
    script = script.replace("/logs/verifier", str(logs))
    script = script.replace("/opt/venv/bin/python3", sys.executable)
    script_path = tmp_path / "test.sh"
    script_path.write_text(script)

    return subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_toolathlon_verifier_scores_task_failure_exit_zero(tmp_path: Path) -> None:
    """Guards PR #887: evaluator task failures score 0 without infra failure."""
    result = _run_toolathlon_verifier_script(
        tmp_path,
        "raise ValueError('Some tests FAILED')\n",
    )

    assert result.returncode == 0, result.stderr
    reward_dir = tmp_path / "logs" / "verifier"
    assert (reward_dir / "reward.txt").read_text() == "0.0\n"
    assert json.loads((reward_dir / "reward.json").read_text()) == {"reward": 0.0}
    assert "ValueError: Some tests FAILED" in result.stdout


def test_toolathlon_verifier_escalates_import_failure(tmp_path: Path) -> None:
    """Guards PR #887: broken evaluator environments fail the verifier."""
    result = _run_toolathlon_verifier_script(
        tmp_path,
        "import definitely_missing_toolathlon_dependency\n",
    )

    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stdout
    assert "evaluator environment failure" in result.stderr
    assert not (tmp_path / "logs" / "verifier" / "reward.txt").exists()


def test_toolathlon_verifier_ignores_launch_time_in_comments(
    tmp_path: Path,
) -> None:
    """Guards PR #887: launch_time comments do not downgrade verifier reward."""
    result = _run_toolathlon_verifier_script(
        tmp_path,
        "import argparse\n"
        "# launch_time is documented here but not accepted by this evaluator.\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--agent_workspace', required=True)\n"
        "parser.add_argument('--groundtruth_workspace', required=True)\n"
        "parser.add_argument('--res_log_file', required=True)\n"
        "parser.parse_args()\n",
    )
    assert result.returncode == 0, result.stderr
    logs = tmp_path / "logs" / "verifier"
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert (
        "unrecognized arguments: --launch_time"
        not in (logs / "toolathlon_evaluator.log").read_text()
    )


def _write_toolathlon_repo_skeleton(repo: Path) -> Path:
    """Minimal upstream Toolathlon layout; returns tasks/finalpool."""
    (repo / ".git").mkdir(parents=True)
    (repo / "global_preparation").mkdir()
    (repo / "configs").mkdir()
    (repo / "configs" / "users_data.json").write_text("{}")
    mcp_dir = repo / "configs" / "mcp_servers"
    mcp_dir.mkdir()
    (mcp_dir / "emails.yaml").write_text(
        "type: stdio\nname: emails\nparams:\n  command: uvx\n"
        '  args: ["emails-mcp", "${token.emails_config_file}"]\n'
        '  cwd: "${agent_workspace}"\n'
    )
    return repo / "tasks" / "finalpool"


def _write_toolathlon_task(finalpool: Path, name: str, servers: list[str]) -> Path:
    task_dir = finalpool / name
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "docs" / "agent_system_prompt.md").write_text("Workspace.\n")
    (task_dir / "docs" / "task.md").write_text(f"Do {name}.\n")
    (task_dir / "task_config.json").write_text(
        json.dumps({"needed_mcp_servers": servers, "needed_local_tools": []})
    )
    return task_dir


def test_toolathlon_email_task_gets_poste_sidecar(tmp_path: Path, monkeypatch) -> None:
    """An email task materializes a poste sidecar compose + a pre-preprocess
    config rewrite; a non-email task stays single-container."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Toolathlon"
    finalpool = _write_toolathlon_repo_skeleton(repo)
    email_task = _write_toolathlon_task(finalpool, "apply-phd-email", ["emails"])
    (email_task / "email_config.json").write_text(
        json.dumps({"imap_server": "localhost", "imap_port": 1143, "smtp_port": 1587})
    )
    _write_toolathlon_task(finalpool, "find-alita-paper", ["filesystem"])

    adapted = adapt_resolved_source_if_needed(
        _resolved(
            repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool", sha="e" * 40
        )
    )

    # Email task: compose with a lightweight mail sidecar + gated main.
    email = adapted.path / "apply-phd-email"
    compose = yaml.safe_load(
        (email / "environment" / "docker-compose.yaml").read_text()
    )
    assert compose["services"]["poste"]["image"] == "${MAIN_IMAGE_NAME}"
    assert compose["services"]["poste"]["hostname"] == "poste"
    assert (
        compose["services"]["main"]["depends_on"]["poste"]["condition"]
        == "service_healthy"
    )
    assert "ports" not in compose["services"]["poste"]
    poste_dir = email / "environment" / "poste"
    assert (poste_dir / "fake_mail.py").exists()
    assert "/toolathlon-poste/fake_mail.py" in compose["services"]["poste"]["command"]

    task = Task(email)
    # Extra headroom for the DinD compose + sidecar image.
    assert task.config.environment.memory_mb == 12288
    commands = [c.command for c in task.config.environment.setup_commands]
    rewrite_idx = next(
        i for i, c in enumerate(commands) if "poste" in c and "imap_server" in c
    )
    preprocess_idx = next(i for i, c in enumerate(commands) if "preprocess.main" in c)
    assert rewrite_idx < preprocess_idx  # rewrite runs before preprocess seeds

    # Non-email task: single container, no compose.
    other = adapted.path / "find-alita-paper"
    assert not (other / "environment" / "docker-compose.yaml").exists()


def test_toolathlon_fake_mail_records_sender_sent(tmp_path: Path) -> None:
    """The lightweight mail sidecar preserves the sender Sent-folder contract
    Toolathlon verifiers use."""
    root = tmp_path / "mail"
    ready = tmp_path / "ready"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchflow.adapters._toolathlon_fake_mail",
            "--root",
            str(root),
            "--smtp-port",
            "11587",
            "--submission-port",
            "12587",
            "--imap-port",
            "11143",
            "--http-port",
            "11080",
            "--ready-file",
            str(ready),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(50):
            if ready.exists():
                break
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=1)
                raise AssertionError(f"fake mail exited: {stdout} {stderr}")
            time.sleep(0.1)
        assert ready.exists()

        msg = MIMEText(
            "Survey: https://docs.google.com/forms/d/abcdefghijk1234567890/edit",
            "plain",
        )
        msg["From"] = "Customer Support Team"
        msg["To"] = "tyler_perez28@mcp.com"
        msg["Subject"] = "Survey"

        smtp = smtplib.SMTP("127.0.0.1", 12587, timeout=10)
        smtp.send_message(msg, from_addr="jason_cruz@mcp.com")
        smtp.quit()

        imap = imaplib.IMAP4("127.0.0.1", 11143, timeout=10)
        imap.login("jason_cruz@mcp.com", "anything")
        assert imap.select("Sent")[0] == "OK"
        status, nums = imap.search(None, "ALL")
        assert status == "OK"
        assert nums and nums[0] == b"1"
        status, data = imap.fetch(b"1", "(RFC822)")
        assert status == "OK"
        assert b"tyler_perez28@mcp.com" in data[0][1]
        assert b"docs.google.com/forms" in data[0][1]
        status, _ = imap.append(
            "Sent",
            None,
            None,
            (
                b"From: jason_cruz@mcp.com\r\n"
                b"To: followup@mcp.com\r\n"
                b"Subject: Follow-up\r\n\r\n"
                b"Thanks"
            ),
        )
        assert status == "OK"
        status, nums = imap.search(None, "ALL")
        assert status == "OK"
        assert nums and nums[0] == b"1 2"
        status, data = imap.fetch(b"2", "(RFC822)")
        assert status == "OK"
        assert b"followup@mcp.com" in data[0][1]
        imap.logout()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_compose_build_tags_main_image_for_sidecar_reuse() -> None:
    """Compose sidecars can reference ${MAIN_IMAGE_NAME} without registry pulls."""
    compose_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "benchflow"
        / "sandbox"
        / "_compose_files"
        / "docker-compose-build.yaml"
    )
    compose = yaml.safe_load(compose_path.read_text())
    assert compose["services"]["main"]["image"] == "${MAIN_IMAGE_NAME}"


def test_toolathlon_k8s_task_gets_host_network_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    """k8s tasks need Docker-socket + host networking, even with Poste present."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Toolathlon"
    finalpool = _write_toolathlon_repo_skeleton(repo)
    (repo / "configs" / "mcp_servers" / "k8s.yaml").write_text(
        "type: stdio\nname: k8s\nparams:\n  command: npx\n"
        '  args: ["-y", "mcp-server-kubernetes"]\n'
        "  env:\n"
        '    KUBECONFIG_PATH: "${token.kubeconfig_path}"\n'
    )
    k8s_task = _write_toolathlon_task(
        finalpool, "k8s-deployment-cleanup", ["k8s", "emails"]
    )
    (k8s_task / "email_config.json").write_text(
        json.dumps({"imap_server": "localhost", "imap_port": 1143, "smtp_port": 1587})
    )

    adapted = adapt_resolved_source_if_needed(
        _resolved(
            repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool", sha="e" * 40
        )
    )

    generated = adapted.path / "k8s-deployment-cleanup"
    compose = yaml.safe_load(
        (generated / "environment" / "docker-compose.yaml").read_text()
    )
    main = compose["services"]["main"]
    assert main["network_mode"] == "host"
    assert "/var/run/docker.sock:/var/run/docker.sock" in main["volumes"]
    assert main["environment"]["DOCKER_HOST"] == "unix:///var/run/docker.sock"
    assert main["depends_on"]["poste"]["condition"] == "service_healthy"
    assert "1143:143" in compose["services"]["poste"]["ports"]
    assert "1587:587" in compose["services"]["poste"]["ports"]

    dockerfile = (generated / "environment" / "Dockerfile").read_text()
    assert "docker-28.3.3.tgz" in dockerfile
    assert "docker.real" in dockerfile
    assert "Backing Filesystem" in dockerfile
    assert "kind.sigs.k8s.io/dl/v0.32.0" in dockerfile
    assert "release/v1.34.1/bin/linux" in dockerfile
    assert "helm-v3.15.4-linux" in dockerfile

    task = Task(generated)
    assert task.config.environment.cpus == 6
    assert task.config.environment.memory_mb == 16384
    assert task.config.environment.storage_mb == 49152
    commands = [c.command for c in task.config.environment.setup_commands]
    assert any("docker info" in c and "helm version" in c for c in commands)
    preprocess_idx = next(i for i, c in enumerate(commands) if "preprocess.main" in c)
    kubeconfig_idx = next(i for i, c in enumerate(commands) if "k8s_configs" in c)
    assert preprocess_idx < kubeconfig_idx
    assert not any("imap_server" in c and "poste-rewrite" in c for c in commands)


def test_toolathlon_woocommerce_task_gets_first_boot_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    """WooCommerce tasks initialize stock WP/MySQL sidecars instead of needing
    pre-existing local seed images in the sandbox Docker daemon."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Toolathlon"
    finalpool = _write_toolathlon_repo_skeleton(repo)
    (repo / "configs" / "mcp_servers" / "woocommerce.yaml").write_text(
        "type: stdio\nname: woocommerce\nparams:\n  command: uvx\n"
        '  args: ["woocommerce-mcp"]\n'
    )
    woo_task = _write_toolathlon_task(
        finalpool, "woocommerce-new-product", ["woocommerce"]
    )
    (woo_task / "token_key_session.py").write_text(
        'woocommerce_site_url = "http://localhost:10003/store97"\n'
    )

    adapted = adapt_resolved_source_if_needed(
        _resolved(
            repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool", sha="e" * 40
        )
    )

    generated = adapted.path / "woocommerce-new-product"
    compose = yaml.safe_load(
        (generated / "environment" / "docker-compose.yaml").read_text()
    )
    assert compose["services"]["woo-db"]["image"] == "mysql:8.0"
    assert compose["services"]["woo"]["image"] == "wordpress:6.8.2-php8.2-apache"
    assert "pull_policy" not in compose["services"]["woo-db"]
    assert "pull_policy" not in compose["services"]["woo"]
    assert compose["services"]["woo"]["entrypoint"] == [
        "/bin/bash",
        "/toolathlon-woo/entry.sh",
    ]
    assert "./woo:/toolathlon-woo:ro" in compose["services"]["woo"]["volumes"]
    assert (generated / "environment" / "woo" / "entry.sh").exists()
    assert (generated / "environment" / "woo" / "users_data.json").exists()
    entry = (generated / "environment" / "woo" / "entry.sh").read_text()
    assert "wp-content/uploads/sites/${site_id}" in entry
    assert "$site_uploads/$month_path" in entry
    assert "$site_uploads/wc-logs" in entry
    assert "$site_uploads/woocommerce_uploads" in entry
    task = Task(generated)
    assert task.config.environment.build_timeout_sec == 2400
    assert any(
        "woo-rewrite" in c.command for c in task.config.environment.setup_commands
    )


def test_toolathlon_canvas_task_reseeds_tokens_on_first_boot(
    tmp_path: Path, monkeypatch
) -> None:
    """Canvas sidecars repair predefined API tokens after every fresh boot."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Toolathlon"
    finalpool = _write_toolathlon_repo_skeleton(repo)
    (repo / "configs" / "mcp_servers" / "canvas.yaml").write_text(
        'type: stdio\nname: canvas\nparams:\n  command: uvx\n  args: ["canvas-mcp"]\n'
    )
    _write_toolathlon_task(finalpool, "canvas-list-test", ["canvas"])

    adapted = adapt_resolved_source_if_needed(
        _resolved(
            repo, repo="hkust-nlp/Toolathlon", path="tasks/finalpool", sha="e" * 40
        )
    )

    compose = yaml.safe_load(
        (
            adapted.path / "canvas-list-test" / "environment" / "docker-compose.yaml"
        ).read_text()
    )
    canvas = compose["services"]["canvas"]
    assert canvas["image"] == "lbjay/canvas-docker"
    assert "pull_policy" not in canvas
    assert canvas["entrypoint"] == ["/bin/bash", "/toolathlon-canvas/entry.sh"]
    assert "./canvas:/toolathlon-canvas:ro" in canvas["volumes"]
    canvas_dir = adapted.path / "canvas-list-test" / "environment" / "canvas"
    assert "seed_canvas.rb" in {p.name for p in canvas_dir.iterdir()}
    entry = (canvas_dir / "entry.sh").read_text()
    seed = (canvas_dir / "seed_canvas.rb").read_text()
    assert 'rm -f "$CANVAS_DIR"/tmp/pids/server.pid' in entry
    assert "repairing predefined API tokens" in entry
    assert "http://localhost:3000/api/v1/accounts/1/courses" in entry
    assert "mcpcanvasadmintoken2" in entry
    assert "reset_token(user, 'Predefined API Token'" in seed
    assert "reset_token(user, 'Admin API Token'" in seed
    assert "Role.get_built_in_role('AccountAdmin')" in seed
    assert "account_user.workflow_state = 'active'" in seed
    task = Task(adapted.path / "canvas-list-test")
    assert task.config.environment.build_timeout_sec == 2400


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
    # JSON creds keep 0644; the mode is data-driven off the spec now.
    assert "target.chmod(mode)" in credential_setup.command
    assert '"file_mode": 420' in credential_setup.command  # 0o644
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
    """Guards PR #887: container config bakes runtime secrets and attributes."""
    (tmp_path / "configs").mkdir()
    result = _run_container_helper(
        tmp_path,
        ["write-config"],
        {
            "TOOLATHLON_GCP_PROJECT_ID": "proj-123",
            "TOOLATHLON_MAPS_API_KEY": "maps-abc",
            "TOOLATHLON_GITHUB_TOKEN": "gho_xyz",
            "TOOLATHLON_NOTION_KEY": "ntn_main",
            "TOOLATHLON_NOTION_SOURCE_PAGE_URL": "https://www.notion.so/src111",
            "TOOLATHLON_NOTION_EVAL_PAGE_URL": "https://www.notion.so/eval222",
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
    # Notion preprocess reads the page URLs off the global config; without them
    # baked in, ``notion_remove_and_duplicate`` gets ``None`` and every notion
    # task's setup crashes.
    assert tokens["notion_integration_key"] == "ntn_main"
    assert tokens["source_notion_page_url"] == "https://www.notion.so/src111"
    assert tokens["eval_notion_page_url"] == "https://www.notion.so/eval222"
    # Unset secrets fall back to their example defaults, not the literal env ref.
    assert tokens["huggingface_token"] == "XX"
    assert tokens["github_read_only"] == "1"
    assert (
        tokens["kubeconfig_path"]
        == "/workspace/deployment/k8s/configs/cluster1-config.yaml"
    )
    assert (
        tokens["emails_config_file"] == "/workspace/configs/example_email_config.json"
    )
    # Upstream preprocess/eval read tokens via ATTRIBUTE access (addict.Dict);
    # the generated dict must support it (a plain dict would AttributeError).
    assert tokens.github_token == "gho_xyz"
    assert tokens.gcp_project_id == "proj-123"


def test_toolathlon_credential_setup_pem_key_skips_json_and_locks_mode() -> None:
    """Guards PR #887: PEM credentials bypass JSON parsing and lock mode."""
    from benchflow.adapters._toolathlon import _toolathlon_credential_setup_command

    cmd = _toolathlon_credential_setup_command({"configs/snowflake_rsa_key.p8"})
    assert cmd is not None
    # The Snowflake private key is PEM, not JSON: written verbatim (no json.loads
    # gate, which would reject it) and locked to 0600 (== 384) as a private key.
    assert '"content_format": "pem"' in cmd
    assert '"file_mode": 384' in cmd
    assert "TOOLATHLON_SNOWFLAKE_RSA_KEY_B64" in cmd
    assert "if spec['content_format'] == 'json':" in cmd


def test_toolathlon_container_launch_resolves_tokens(tmp_path: Path) -> None:
    """Guards PR #887: launcher resolves global and per-task token refs."""
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
    out = f"{result.stdout}\n{result.stderr}"
    assert "--project=global-proj" in out  # from global
    assert "--dataset=demo_ds" in out  # per-task override wins
    assert "--folder=FID-999" in out  # runtime file value


def test_toolathlon_container_launch_uses_task_token_snapshot(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "token_key_session.py").write_text(
        "all_token_key_session = {'gcp_project_id': 'global-proj'}\n"
    )
    task_dir = tmp_path / "tasks" / "finalpool" / "ab-testing"
    groundtruth = task_dir / "groundtruth_workspace"
    groundtruth.mkdir(parents=True)
    (groundtruth / "log_bucket_name.txt").write_text("abtesting_logging_123")
    (task_dir / "token_key_session.py").write_text(
        "import os\n"
        "from addict import Dict\n"
        "_here = os.path.dirname(__file__)\n"
        "with open(os.path.join(_here, 'groundtruth_workspace', "
        "'log_bucket_name.txt')) as f:\n"
        "    _bucket = f.read().strip()\n"
        "all_token_key_session = Dict("
        "google_cloud_allowed_log_buckets=_bucket)\n"
    )

    result = _run_container_helper(
        tmp_path,
        ["write-task-tokens", str(task_dir)],
    )
    assert result.returncode == 0, result.stderr
    (groundtruth / "log_bucket_name.txt").unlink()

    result = _run_container_helper(
        tmp_path,
        [
            "launch",
            "/bin/echo",
            "--bucket=${token.google_cloud_allowed_log_buckets}",
        ],
        {"TOOLATHLON_TASK_DIR": str(task_dir)},
    )
    assert result.returncode == 0, result.stderr
    assert "--bucket=abtesting_logging_123" in f"{result.stdout}\n{result.stderr}"


def test_toolathlon_container_launch_wraps_google_cloud_mcp(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "token_key_session.py").write_text(
        "all_token_key_session = {}\n"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uvx = fake_bin / "uvx"
    fake_uvx.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    fake_uvx.chmod(0o755)
    runtime_dir = tmp_path / "runtime"

    result = _run_container_helper(
        tmp_path,
        [
            "launch",
            str(fake_uvx),
            "google-cloud-mcp",
            "--project-id",
            "toolathlon-bench",
            "--allowed-buckets",
            "promo-assets-for-b*",
        ],
        {
            "TOOLATHLON_TASK_DIR": str(tmp_path),
            "TOOLATHLON_RUNTIME_DIR": str(runtime_dir),
        },
    )
    assert result.returncode == 0, result.stderr
    out = f"{result.stdout}\n{result.stderr}"
    assert "--from" in out
    assert "google-cloud-mcp" in out
    assert "python" in out
    assert "google_cloud_mcp_pattern_wrapper.py" in out
    assert "--allowed-buckets" in out
    assert "promo-assets-for-b*" in out
    wrapper = runtime_dir / "google_cloud_mcp_pattern_wrapper.py"
    assert "_PatternSet" in wrapper.read_text()


def test_toolathlon_container_launch_resolves_env(tmp_path: Path) -> None:
    """Guards PR #887: launcher resolves token refs inside child env."""
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
    assert "GITHUB_TOKEN=gho_secret" in f"{result.stdout}\n{result.stderr}"


def test_toolathlon_container_launch_prefers_generated_kubeconfig(
    tmp_path: Path,
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "token_key_session.py").write_text(
        "all_token_key_session = {}\n"
    )
    task_dir = tmp_path / "tasks" / "finalpool" / "k8s-mysql"
    (task_dir / "k8s_configs").mkdir(parents=True)
    generated = task_dir / "k8s_configs" / "cluster-mysql-inst-alpha-config.yaml"
    generated.write_text("apiVersion: v1\n")
    (task_dir / "token_key_session.py").write_text(
        "from addict import Dict\n"
        "all_token_key_session = Dict("
        "kubeconfig_path='k8s_configs/cluster-mysql-config.yaml')\n"
    )
    result = _run_container_helper(
        tmp_path,
        ["launch", "/bin/echo", "KUBECONFIG=${token.kubeconfig_path}"],
        {"TOOLATHLON_TASK_DIR": str(task_dir)},
    )
    assert result.returncode == 0, result.stderr
    assert f"KUBECONFIG={generated}" in f"{result.stdout}\n{result.stderr}"


def test_toolathlon_container_launch_ensures_dirs(tmp_path: Path) -> None:
    """Guards PR #887: launcher creates server storage dirs before spawn."""
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


def test_toolathlon_container_launch_filters_non_json_stdout(
    tmp_path: Path,
) -> None:
    """Noisy MCP startup logs are moved off stdout before they reach clients."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "token_key_session.py").write_text(
        "all_token_key_session = {}\n"
    )
    result = _run_container_helper(
        tmp_path,
        [
            "launch",
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('server started'); "
                'print(\'{"jsonrpc":"2.0","id":1,"result":{}}\')'
            ),
        ],
        {"TOOLATHLON_TASK_DIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '{"jsonrpc":"2.0","id":1,"result":{}}'
    assert "server started" in result.stderr


def test_toolathlon_arxiv_server_declares_ensure_dirs(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards PR #887: arxiv_local storage is pre-created for evaluators."""
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
    (task_dir / "docs" / "agent_system_prompt.md").write_text(
        "W: !!<<<<||||workspace_dir||||>>>>!!\n"
    )
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
