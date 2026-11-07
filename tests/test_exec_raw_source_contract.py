"""Guards FF-sw-docker-C13: new harness readers cannot bypass raw exec.

The rule deliberately covers every exec use, not a list of known parsers or
command spellings. Even a future reader, a callback, or a fallback must opt in.
Only provider SDK entry points and transparent protocol forwarding are exempt.
"""

import ast
from pathlib import Path

import pytest

SDK_FILES = {
    "sandbox/daytona.py",
    "sandbox/process/daytona.py",
}
FORWARDERS = {"runtime.py", "sandbox/daytona.py", "rollout/session_factory_runtime.py"}


def exec_violations(source, filename="reader.py"):
    tree = ast.parse(source)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    violations = []

    def raw_argument(node):
        parent = parents.get(node)
        return (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "exec_raw"
            and parent.args
            and parent.args[0] is node
        )

    for node in ast.walk(tree):
        methods = {
            "exec",
            "exec_transient",
            "exec_in_service",
        }
        dynamic = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in methods
        )
        if not dynamic and (
            not isinstance(node, ast.Attribute) or node.attr not in methods
        ):
            continue
        if raw_argument(node):
            continue
        receiver = ast.unparse(node.args[0] if dynamic else node.value)
        parent = parents.get(node)
        if filename in SDK_FILES and receiver in {
            "sandbox.process",
            "self._require_sandbox().process",
        }:
            continue  # External SDK: no BenchFlow result redactor.
        if (
            filename == "sandbox/daytona_channel.py"
            and receiver == "self.provider.process"
        ):
            continue  # the captured provider's raw SDK, not sandbox.exec.
        if (
            filename in {"sandbox/modal_impl.py", "sandbox/process/modal.py"}
            and receiver == "self._sandbox"
            and isinstance(parent, ast.Attribute)
            and parent.attr == "aio"
        ):
            continue  # Modal's SDK coroutine, not a BenchFlow exec result.
        # The session-factory agent's sandbox forwards to the sandbox with
        # only the no_new_privs prefix added to the command (K1-R11-1).
        if filename in FORWARDERS and receiver in {
            "self._inner",
            "self._strategy",
            "self.sandbox",
        }:
            # Forward the very same result; no extraction, filtering or parsing.
            await_node = parents.get(parent)
            return_node = parents.get(await_node)
            if (
                isinstance(parent, ast.Call)
                and parent.func is node
                and isinstance(await_node, ast.Await)
                and isinstance(return_node, ast.Return)
            ):
                continue
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "partial"
            and len(parent.args) > 1
            and isinstance(parent.args[0], ast.Name)
            and parent.args[0].id == "exec_raw"
            and parent.args[1] is node
        ):
            continue
        # Transient polling chooses one callable, then must pass it to exec_raw.
        alias_node = parent
        if isinstance(parent, ast.BoolOp):
            alias_node = parents.get(parent)
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "getattr"
        ):
            alias_node = parents.get(parent)
        if (
            isinstance(alias_node, ast.Assign)
            and len(alias_node.targets) == 1
            and isinstance(alias_node.targets[0], ast.Name)
        ):
            name = alias_node.targets[0].id
            scope = node
            while scope in parents and not isinstance(
                scope, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                scope = parents[scope]
            uses = [
                use
                for use in ast.walk(scope)
                if isinstance(use, ast.Name)
                and isinstance(use.ctx, ast.Load)
                and use.id == name
            ]
            if uses and all(raw_argument(use) for use in uses):
                continue
        violations.append(node.lineno)
    return violations


def test_every_harness_exec_uses_the_raw_api():
    root = Path(__file__).parents[1] / "src/benchflow"
    violations = []
    for path in root.rglob("*.py"):
        name = path.relative_to(root).as_posix()
        violations.extend(
            f"{name}:{line}" for line in exec_violations(path.read_text(), name)
        )
    assert not violations, "Harness exec bypasses exec_raw:\n" + "\n".join(violations)


def test_channel_sdk_inventory_does_not_exempt_harness_readers():
    """The new transport's SDK exception cannot hide a harness reader."""
    name = "sandbox/daytona_channel.py"
    assert not exec_violations(
        'async def read(self):\n    return await self.provider.process.exec("cat journal")',
        name,
    )
    assert exec_violations(
        'async def read(self):\n    return await self.sandbox.exec("cat journal")',
        name,
    )


@pytest.mark.parametrize(
    "reader",
    [
        'result = await env.exec("cat manifest")\njson.loads(result.stdout)',
        'json.loads((await env.exec("cat manifest")).stdout)',
        'result = await env.exec("batch")\nparse_merged(getattr(result, "stdout", None), tag, 2)',
        'try:\n result = await exec_raw(env.exec, "batch")\nexcept RuntimeError:\n result = await env.exec("fallback")',
        'run = env.exec\nresult = await run("cat manifest")',
        'read = getattr(env, "exec_transient", env.exec)\nresult = await read("cat manifest")',
        'result = await env.exec_in_service("worker", "cat manifest")',
    ],
)
def test_source_rule_rejects_new_readers_aliases_batches_and_fallbacks(reader):
    assert exec_violations(
        "async def reader():\n"
        + "\n".join("    " + line for line in reader.splitlines())
    )


def test_source_rule_does_not_exempt_a_parser_in_a_forwarder_file():
    assert exec_violations(
        "async def exec(self, cmd):\n    result = await self._inner.exec(cmd)\n    return json.loads(result.stdout)",
        "runtime.py",
    )


# Presentation helpers may return rendered text to their callers. Every call
# to one is itself a diagnostic origin, so moving the conversion into a helper
# cannot launder it into a parser or accounting function.
PRESENTATION_HELPERS = {
    ("sandbox/_exec_output.py", "diagnostic_result"),
    ("sandbox/_exec_output.py", "render"),
    ("sandbox/lockdown.py", "_exec_failure_detail"),
    ("providers/litellm_runtime.py", "log_tail"),
    ("providers/litellm_runtime.py", "_exec_details"),
    ("providers/litellm_bedrock_preflight.py", "_exec_details"),
    ("providers/litellm_bedrock_preflight.py", "_failure_message"),
}

# These return published trace metadata or the existing transport API. Readers
# of the latter must independently opt into raw_output/exec_raw above.
PRESENTATION_RETURNS = {
    ("rollout/_setup.py", "_run_oracle"),
    ("sandbox/docker.py", "exec"),
    *(
        ("sandbox/remote_docker.py", name)
        for name in (
            "_docker_cli",
            "_run_docker_compose_command",
            "_run_docker_container_command",
        )
    ),
}
PRESENTATION_SINKS = {
    "RuntimeError",
    "ValueError",
    "AgentInstallError",
    "SandboxStartupError",
    "SeparateVerifierError",
    "JudgeInfrastructureError",
    "SandboxDaemonGuardError",
    "KeptServiceFreezerUnavailable",
    "EnvironmentSnapshotError",
    "EnvironmentRestoreError",
    "EvidenceError",
    "BedrockPatchPreflightError",
    "cloud_metadata_record",
    "TaskMdVerifierError",
}
PRESENTATION_PARAMETERS = {
    ("review/evidence.py", "_capture_workspace_with_tar"): {"python_error"},
}


def diagnostic_parser_violations(source, filename="reader.py"):
    """Allow presentation sinks, never a list of known parser spellings.

    Scan all backends and all consumers. Follow aliases, containers, keyword
    arguments and explicit rendering; unknown calls/returns fail closed. The
    runtime non-string type rejects accidental parser inputs independently.
    """
    tree = ast.parse(source)
    violations = set()
    transports = {
        "_docker_cli",
        "_run_docker_compose_command",
        "_run_docker_container_command",
    }
    helper_names = {name for _, name in PRESENTATION_HELPERS} - {"render"}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "benchflow.sandbox._exec_output"
        ):
            helper_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name
                in {"diagnostic_result", "DiagnosticText", "DiagnosticResult"}
            )
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tainted = set(PRESENTATION_PARAMETERS.get((filename, scope.name), ()))

        def diagnostic(node, tainted=tainted):
            if node is None:
                return False
            if isinstance(node, ast.Name):
                return node.id in tainted or node.id in helper_names
            if isinstance(node, ast.Attribute):
                if node.attr in {"return_code", "exit_code"}:
                    return False
                if node.attr == "_diagnostic_result":
                    return True
                if ast.unparse(node) in tainted:
                    return True
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func).rsplit(".", 1)[-1]
                if name in PRESENTATION_SINKS:
                    return False
                parameters = PRESENTATION_PARAMETERS.get((filename, name), set())
                if (
                    parameters
                    and not any(diagnostic(arg) for arg in node.args)
                    and all(
                        not diagnostic(keyword.value) or keyword.arg in parameters
                        for keyword in node.keywords
                    )
                ):
                    return False
                if name in helper_names | {
                    "DiagnosticText",
                    "DiagnosticResult",
                    "_redact_result",
                }:
                    return True
                if name == "exec_raw":
                    return False
                if isinstance(node.func, ast.Attribute) and name in transports:
                    if ast.unparse(node.func.value) == "super()":
                        return False
                    return not any(
                        keyword.arg == "raw_output"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                        for keyword in node.keywords
                    )
            return any(diagnostic(child) for child in ast.iter_child_nodes(node))

        nodes = list(ast.walk(scope))
        while True:
            previous = set(tainted)
            for node in nodes:
                if isinstance(
                    node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)
                ) and diagnostic(node.value):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        tainted.add(ast.unparse(target))
                        if isinstance(target, (ast.Tuple, ast.List)):
                            tainted.update(ast.unparse(child) for child in target.elts)
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"append", "extend", "update"}
                    and any(diagnostic(arg) for arg in node.args)
                ):
                    tainted.add(ast.unparse(node.func.value))
            if tainted == previous:
                break
        for node in nodes:
            if (
                isinstance(node, ast.Return)
                and node.value is not None
                and diagnostic(node.value)
                and (filename, scope.name)
                not in PRESENTATION_HELPERS | PRESENTATION_RETURNS
            ):
                violations.add(node.lineno)
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            method = name.rsplit(".", 1)[-1]
            args = [*node.args, *(keyword.value for keyword in node.keywords)]
            receiver_tainted = isinstance(node.func, ast.Attribute) and diagnostic(
                node.func.value
            )
            if not receiver_tainted and not any(diagnostic(arg) for arg in args):
                continue
            # Formatting retains diagnostic provenance. Publication is a sink.
            if method in {
                "render",
                "strip",
                "lstrip",
                "rstrip",
                "join",
                "endswith",
                "append",
                "extend",
                "update",
            }:
                continue
            if name in {
                "str",
                "len",
                "getattr",
                "diagnostic_result",
                "DiagnosticResult",
                "DiagnosticText",
            }:
                continue
            if (
                name.startswith(("logger.", "self.logger."))
                or name == "sys.stderr.write"
                or method == "write_text"
            ):
                continue
            if name in PRESENTATION_SINKS or (filename, method) in PRESENTATION_HELPERS:
                continue
            if name in {"asyncio.wait_for", "self._up_on_own_subnets"}:
                # These forward/await a command; they do not read its streams.
                continue
            parameters = PRESENTATION_PARAMETERS.get((filename, method), set())
            if (
                parameters
                and not any(diagnostic(arg) for arg in node.args)
                and all(
                    not diagnostic(keyword.value) or keyword.arg in parameters
                    for keyword in node.keywords
                )
            ):
                continue
            violations.add(node.lineno)
    return sorted(violations)


def test_no_structured_parser_consumes_a_diagnostic_copy():
    root = Path(__file__).parents[1] / "src/benchflow"
    violations = []
    for path in root.rglob("*.py"):
        source = path.read_text()
        violations.extend(
            f"{path.relative_to(root)}:{line}"
            for line in diagnostic_parser_violations(
                source, path.relative_to(root).as_posix()
            )
        )
    assert not violations, "Parser consumes diagnostic output:\n" + "\n".join(
        violations
    )


@pytest.mark.parametrize(
    "reader",
    [
        "return json.loads(diagnostic_result(result).stdout)",
        'display = diagnostic_result(result)\ndata = getattr(display, "stdout", "")\nreturn json.loads(data)',
        "display = diagnostic_result(result)\nreturn parse_merged(display.stdout, tag, 2)",
        "display = diagnostic_result(result).stdout.strip()\nreturn Path(display)",
        "text = diagnostic_result(result).stdout\ntrajectory_from_litellm_callback_log(text, session_id='fixture', agent_name='fixture')",
        "text = diagnostic_result(result).stdout.render()\nnew_importer(text)",
        "text = diagnostic_result(result).stderr.render()\naccount_usage(text=text)",
        "text = diagnostic_result(result).stdout\ncopy = text\nreturn copy",
        "text = diagnostic_result(result).stdout.render()\nparts = []\nparts.append(text)\nscore(''.join(parts))",
        "text = diagnostic_result(result).stdout.render()\nrecord = {'stdout': text}\nfuture_reader(record)",
        "text: str = diagnostic_result(result).stdout.render()\nfuture_reader(text)",
        "text = diagnostic_result(result).stdout.render()\nfuture_reader(f'{text}')",
        "future_reader(result._diagnostic_result.stdout)",
    ],
)
def test_source_rule_rejects_parsing_diagnostic_copy(reader):
    source = "def reader(result):\n" + "\n".join(
        "    " + line for line in reader.splitlines()
    )
    assert diagnostic_parser_violations(source)


@pytest.mark.parametrize(
    "method",
    ["_docker_cli", "_run_docker_compose_command", "_run_docker_container_command"],
)
def test_source_rule_covers_backend_parser_streams(method):
    source = f'async def reader(env):\n    result = await env.{method}(["new-command"])\n    return json.loads(result.stdout)'
    assert diagnostic_parser_violations(source)
    assert not diagnostic_parser_violations(
        source.replace('(["new-command"])', '(["new-command"], raw_output=True)')
    )
    assert not diagnostic_parser_violations(
        source.replace(
            f'env.{method}(["new-command"])', f'exec_raw(env.{method}, ["new-command"])'
        )
    )


@pytest.mark.parametrize(
    "filename",
    [
        "sandbox/docker.py",
        "sandbox/remote_docker.py",
        "sandbox/daytona.py",
        "sandbox/daytona_dind.py",
        "sandbox/daytona_strategies.py",
        "sandbox/modal_impl.py",
        "sandbox/e2b.py",
        "sandbox/runloop.py",
        "runtime.py",
        "providers/litellm_runtime.py",
    ],
)
@pytest.mark.parametrize(
    "reader",
    [
        "copy = diagnostic_result\ntext = copy(result).stdout.render()\nunknown_importer(text)",
        "from benchflow.sandbox._exec_output import diagnostic_result as display\ntext = display(result).stdout.render()\naccounting(text=text)",
        "text, other = (diagnostic_result(result).stdout.render(), 0)\nnew_scorer(text)",
        "text = diagnostic_result(result).stderr.render()\ntext.splitlines()",
        "text = diagnostic_result(result).stdout.render()\nbox.value = text\nnew_reader(box.value)",
    ],
)
def test_diagnostic_rule_has_no_backend_or_consumer_name_exemption(filename, reader):
    """Guards C14/79b9a530 across aliases, unpacking, fields and every backend."""
    source = "def reader(result):\n" + "\n".join(
        "    " + line for line in reader.splitlines()
    )
    assert diagnostic_parser_violations(source, filename)
