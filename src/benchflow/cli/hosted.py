"""``bench hosted``: run a benchmark on BenchFlow's hosted service instead of this machine.

The client is :mod:`benchflow.hosted`. The address is ``--hub`` or ``BENCHFLOW_HOSTED_URL``; the key comes from
``BENCHFLOW_HUB_KEY`` or a key file, never from the command line. Every command checks the key offline before it
calls the service and never prints more of it than its kind (``bfk_…``, ``bh_…``). There is no default address:
with neither ``--hub`` nor ``BENCHFLOW_HOSTED_URL``, every command stops before sending anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from benchflow.cli._shared import print_error, print_hint

if TYPE_CHECKING:
    from benchflow.hosted import HostedClient

# A run that ended in one of these exits 1; any other final status exits 0.
_FAILED = frozenset({"failed", "cancelled", "canceled", "error", "errored"})

_RunId = Annotated[str, typer.Argument(help="The run id `bench hosted run` printed")]
_Version = Annotated[
    str, typer.Argument(help="The hub version to run, e.g. PKG-TERMINALBENCH-2-0")
]
_Json = Annotated[bool, typer.Option("--json", help="Print the service's JSON answer")]
_Hub = Annotated[
    str | None,
    typer.Option(
        "--hub", help="The hosted service's address (default: $BENCHFLOW_HOSTED_URL)"
    ),
]
_KeyFile = Annotated[
    Path | None,
    typer.Option(
        "--key-file",
        help="A file holding the key, mode 600 (default: $BENCHFLOW_HUB_KEY, then ~/.config/benchflow/hub-key)",
    ),
]
_Agent = Annotated[
    str,
    typer.Option(
        "--agent", "-a", help="The agent (default: oracle, the reference solution)"
    ),
]
_Model = Annotated[str | None, typer.Option("--model", "-m", help="The agent's model")]
_NTasks = Annotated[
    int | None,
    typer.Option("--n-tasks", "-n", help="Run at most N of the version's tasks", min=1),
]
_Include = Annotated[
    list[str] | None,
    typer.Option(
        "--include",
        "-i",
        help="Only these tasks, by name (comma-separated or repeated)",
    ),
]
_Trials = Annotated[int, typer.Option("--trials", help="Trials per task", min=1)]
_Concurrency = Annotated[
    int,
    typer.Option(
        "--concurrency",
        "-j",
        help="Trials at a time (the service may lower it to fit its sandbox cap)",
        min=1,
    ),
]
_Name = Annotated[str | None, typer.Option("--name", help="A name for the run")]
_MaxSeconds = Annotated[
    int | None,
    typer.Option(
        "--max-sandbox-seconds", help="Stop the run past this much sandbox time", min=1
    ),
]
_Secret = Annotated[
    list[str] | None,
    typer.Option(
        "--secret",
        help="A secret stored on the service, by name, for the run's agent (repeatable)",
    ),
]
_ModelKey = Annotated[
    list[str] | None,
    typer.Option(
        "--model-key",
        help=(
            "Your own provider key for this run only, as KIND=SOURCE: KIND is anthropic, deepseek, kimi, zai or mimo; "
            "SOURCE is file:PATH, env:VAR or - (stdin), never the key itself. The service keeps it until the run ends "
            "and the sandbox never sees it (repeatable: keys of one kind are tried in turn)"
        ),
    ),
]
_Endpoint = Annotated[
    str | None,
    typer.Option(
        "--endpoint",
        help="Your own model endpoint for this run only: a CLIProxyAPI you run, or any Anthropic-compatible https URL",
    ),
]
_EndpointKey = Annotated[
    str | None,
    typer.Option(
        "--endpoint-key",
        help="Where --endpoint's key is: file:PATH, env:VAR or - (stdin)",
    ),
]
_MaxCost = Annotated[
    float | None,
    typer.Option(
        "--max-cost-usd",
        help="The run's model budget: the service refuses its model calls past it (its own count)",
        min=0.0,
    ),
]
_Source = Annotated[
    str,
    typer.Option(
        "--from",
        help="Where the value is: file:PATH, env:VAR or - (stdin, the default)",
    ),
]
_Wait = Annotated[bool, typer.Option("--wait", help="Wait for the run to finish")]
_Timeout = Annotated[
    float | None,
    typer.Option(
        "--timeout",
        help="Stop waiting after this many seconds (the run goes on)",
        min=1,
    ),
]
_Output = Annotated[
    Path,
    typer.Option("--output", "-o", help="The file to write the trajectories to"),
]
_Format = Annotated[
    str, typer.Option("--format", help="atif (one ATIF document per line) or messages")
]


@contextmanager
def _service(hub: str | None, key_file: Path | None) -> Iterator[HostedClient]:
    """The client, with every :class:`~benchflow.hosted.HostedError` printed as one error line."""
    from benchflow.hosted import HostedClient, HostedError

    try:
        yield HostedClient.from_env(hub, key_file=key_file)
    except HostedError as e:
        print_error(str(e), next_step=e.hint)
        raise typer.Exit(e.exit_code) from None


def _split(values: list[str] | None) -> list[str]:
    return [v.strip() for value in values or [] for v in value.split(",") if v.strip()]


def _show_page(client: HostedClient, run_id: str) -> None:
    print_hint(f"page: {client.run_page(run_id)} (sign in to read it)")


def _wait(
    client: HostedClient, run_id: str, timeout: float | None, output_json: bool
) -> None:
    """Wait for the run, printing each status it reaches; exits 1 when it failed or was cancelled."""
    from benchflow.hosted import summary_line

    def show(record: dict[str, Any]) -> None:
        if not output_json:
            print_hint(f"{run_id}: {summary_line(record)}", stderr=True)

    record = client.wait(run_id, timeout=timeout, on_status=show)
    if output_json:
        typer.echo(json.dumps(record))
    else:
        print_hint(summary_line(record))
        _show_page(client, run_id)
    status = str(record.get("status") or "").lower()
    if status in _FAILED:
        print_error(
            f"run {run_id} ended {status}",
            next_step="read what happened with `bench hosted events` and the run id",
        )
        raise typer.Exit(1)


def _submit(
    client: HostedClient,
    version: str,
    options: dict[str, Any],
    *,
    wait: bool,
    timeout: float | None,
    output_json: bool,
) -> None:
    answer = client.submit(version, **options)
    run_id = str(answer["run_id"])
    if output_json and not wait:
        typer.echo(json.dumps({**answer, "page": client.run_page(run_id)}))
        return
    if not output_json:
        print_hint(f"queued run {run_id}", style="green")
        for warning in answer.get("warnings") or []:
            print_hint(f"warning: {warning}", style="yellow", stderr=True)
        _show_page(client, run_id)
    if wait:
        _wait(client, run_id, timeout, output_json)
    else:
        print_hint(f"follow it with: bench hosted wait {run_id}")


def _run_options(
    agent: str,
    model: str | None,
    n_tasks: int | None,
    include: list[str] | None,
    trials: int,
    concurrency: int,
    name: str | None,
    max_sandbox_seconds: int | None,
    secrets: list[str] | None,
    model_keys: list[str] | None = None,
    endpoint: str | None = None,
    endpoint_key: str | None = None,
    max_cost_usd: float | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "agent": agent,
        "model": model,
        "tasks": _split(include),
        "max_tasks": n_tasks,
        "n_trials": trials,
        "concurrency": concurrency,
        "name": name,
        "max_sandbox_seconds": max_sandbox_seconds,
        "secrets": _split(secrets),
    }
    if max_cost_usd is not None:
        options["max_cost_usd"] = max_cost_usd
    once = _once(model_keys, endpoint, endpoint_key)
    if once:
        options["secrets_once"] = once
    return options


def _once(
    model_keys: list[str] | None, endpoint: str | None, endpoint_key: str | None
) -> list[dict[str, str]]:
    """The run's own keys (--model-key, --endpoint): read here from where they are kept, sent with the run, never
    printed. A bad flag stops the command before anything is sent."""
    from benchflow.hosted import KEY_KINDS, HostedError, normalize_endpoint, read_secret

    out: list[dict[str, str]] = []
    try:
        counts: dict[str, int] = {}
        for item in model_keys or []:
            kind, sep, source = item.partition("=")
            kind = kind.strip().lower()
            if not sep or kind not in KEY_KINDS:
                raise HostedError(
                    f"--model-key takes KIND=SOURCE with KIND one of {', '.join(KEY_KINDS)}",
                    hint="e.g. --model-key deepseek=file:~/.config/benchflow/deepseek.key",
                    exit_code=2,
                )
            counts[kind] = counts.get(kind, 0) + 1
            name = kind.upper() + (f"_{counts[kind]}" if counts[kind] > 1 else "")
            out.append({"name": name, "kind": kind, "value": read_secret(source)})
        if bool(endpoint) != bool(endpoint_key):
            raise HostedError(
                "--endpoint and --endpoint-key go together",
                exit_code=2,
                hint="e.g. --endpoint https://cpa.example.com --endpoint-key file:~/cpa_client.key",
            )
        if endpoint:
            out.append(
                {
                    "name": "ENDPOINT_URL",
                    "kind": "provider_base_url",
                    "value": normalize_endpoint(endpoint),
                }
            )
            out.append(
                {
                    "name": "ENDPOINT_KEY",
                    "kind": "provider_api_key",
                    "value": read_secret(endpoint_key or ""),
                }
            )
    except HostedError as e:
        print_error(str(e), next_step=getattr(e, "hint", None))
        raise typer.Exit(e.exit_code) from None
    return out


def _print_results(client: HostedClient, run_id: str, results: dict[str, Any]) -> None:
    from benchflow.hosted import summary_line

    print_hint(summary_line(results["run"]))
    metrics = results.get("metrics") or {}
    for key in metrics.get("keys") or []:
        if isinstance(key, dict) and isinstance(key.get("mean"), int | float):
            spread = (
                f" ± {key['sd']:.3g}" if isinstance(key.get("sd"), int | float) else ""
            )
            print_hint(
                f"{key.get('key')}: mean {key['mean']:.3g}{spread} over {key.get('n')}"
            )
    for error in metrics.get("errors") or []:
        if isinstance(error, dict):
            print_hint(
                f"{error.get('type')}: {error.get('n')} trial(s), e.g. {error.get('sample')}",
                style="yellow",
            )
    for trace in results.get("traces") or []:
        print_hint(
            f"  {trace.get('task_name')}  reward {trace.get('reward')}  {trace.get('outcome') or ''}".rstrip()
        )
    _show_page(client, run_id)


def register_hosted(app: typer.Typer) -> None:
    """Attach the ``hosted`` command group to the top-level benchflow app."""
    hosted_app = typer.Typer(
        help=(
            "Hosted evals: run a benchmark on BenchFlow's servers instead of this "
            "machine (BENCHFLOW_HOSTED_URL, BENCHFLOW_HUB_KEY)."
        ),
        no_args_is_help=True,
    )
    app.add_typer(hosted_app, name="hosted")

    @hosted_app.command("run")
    def hosted_run(
        version: _Version,
        agent: _Agent = "oracle",
        model: _Model = None,
        n_tasks: _NTasks = None,
        include: _Include = None,
        trials: _Trials = 1,
        concurrency: _Concurrency = 1,
        name: _Name = None,
        max_sandbox_seconds: _MaxSeconds = None,
        secret: _Secret = None,
        model_key: _ModelKey = None,
        endpoint: _Endpoint = None,
        endpoint_key: _EndpointKey = None,
        max_cost_usd: _MaxCost = None,
        wait: _Wait = False,
        timeout: _Timeout = None,
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Queue a run of a hub version on the hosted service; print its id and page."""
        options = _run_options(
            agent, model, n_tasks, include, trials, concurrency, name,
            max_sandbox_seconds, secret, model_key, endpoint, endpoint_key,
            max_cost_usd,
        )  # fmt: skip
        with _service(hub, key_file) as client:
            _submit(
                client,
                version,
                options,
                wait=wait,
                timeout=timeout,
                output_json=output_json,
            )

    @hosted_app.command("submit")
    def hosted_submit(
        version: Annotated[
            str,
            typer.Option(
                "--version", help="The hub version to run, e.g. PKG-TERMINALBENCH-2-0"
            ),
        ],
        agent: _Agent = "oracle",
        model: _Model = None,
        n_tasks: _NTasks = None,
        include: _Include = None,
        trials: _Trials = 1,
        concurrency: _Concurrency = 1,
        name: _Name = None,
        max_sandbox_seconds: _MaxSeconds = None,
        secret: _Secret = None,
        model_key: _ModelKey = None,
        endpoint: _Endpoint = None,
        endpoint_key: _EndpointKey = None,
        max_cost_usd: _MaxCost = None,
        wait: _Wait = False,
        timeout: _Timeout = None,
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Queue a run: `bench hosted run` with the version as --version."""
        options = _run_options(
            agent, model, n_tasks, include, trials, concurrency, name,
            max_sandbox_seconds, secret, model_key, endpoint, endpoint_key,
            max_cost_usd,
        )  # fmt: skip
        with _service(hub, key_file) as client:
            _submit(
                client,
                version,
                options,
                wait=wait,
                timeout=timeout,
                output_json=output_json,
            )

    @hosted_app.command("estimate")
    def hosted_estimate(
        version: _Version,
        agent: _Agent = "oracle",
        model: _Model = None,
        n_tasks: _NTasks = None,
        include: _Include = None,
        trials: _Trials = 1,
        concurrency: _Concurrency = 1,
        max_sandbox_seconds: _MaxSeconds = None,
        max_cost_usd: _MaxCost = None,
        output_json: Annotated[
            bool,
            typer.Option(
                "--json", help="Print the service's JSON answer (the only output)"
            ),
        ] = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Ask what a run would likely take and whether it fits, without starting it (JSON).

        It takes the run's selection and budget options, never its keys: an estimate sends none."""
        options = _run_options(
            agent, model, n_tasks, include, trials, concurrency, None,
            max_sandbox_seconds, None, max_cost_usd=max_cost_usd,
        )  # fmt: skip
        with _service(hub, key_file) as client:
            typer.echo(json.dumps(client.estimate(version, **options), indent=2))

    secret_app = typer.Typer(
        help=(
            "Your workspace's own model keys on the service: a provider key, or an endpoint (a CLIProxyAPI you run) "
            "with its key. Runs name them with --secret; the service's key proxy uses them and the sandbox never "
            "sees them. Values are read from a file, a variable or stdin, never the command line."
        ),
        no_args_is_help=True,
    )
    hosted_app.add_typer(secret_app, name="secret")

    @secret_app.command("set")
    def secret_set(
        name: Annotated[str, typer.Argument(help="The key's name, e.g. DEEPSEEK")],
        kind: Annotated[
            str, typer.Option("--kind", help="anthropic, deepseek, kimi, zai or mimo")
        ],
        source: _Source = "-",
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Store a provider key for your runs (name it in a run with --secret NAME)."""
        from benchflow.hosted import check_secret_kind, read_secret

        with _service(hub, key_file) as client:
            kind = check_secret_kind(kind, client.key)
            info = client.save_secret(name, kind, read_secret(source))
            if output_json:
                typer.echo(
                    json.dumps({k: info.get(k) for k in ("name", "kind", "hint")})
                )
                return
            print_hint(
                f"stored {info.get('name')} ({info.get('kind')}) {info.get('hint') or ''}".rstrip(),
                style="green",
            )
            print_hint(
                f"use it: bench hosted run <version> --secret {info.get('name')} ..."
            )

    @secret_app.command("endpoint")
    def secret_endpoint(
        name: Annotated[
            str, typer.Argument(help="A name; the service stores NAME_URL and NAME_KEY")
        ],
        url: Annotated[
            str,
            typer.Option(
                "--url",
                help="The endpoint, https (a CLIProxyAPI or any Anthropic-compatible URL)",
            ),
        ],
        source: _Source = "-",
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Store your own model endpoint with its key; the service sends a stored endpoint only the key stored with it."""
        from benchflow.hosted import normalize_endpoint, read_secret

        with _service(hub, key_file) as client:
            answer = client.save_endpoint(
                name, normalize_endpoint(url), read_secret(source)
            )
            items = answer.get("items") or []
            if output_json:
                typer.echo(
                    json.dumps(
                        {
                            "items": [
                                {k: i.get(k) for k in ("name", "kind", "hint")}
                                for i in items
                            ]
                        }
                    )
                )
                return
            names = [str(i.get("name")) for i in items]
            print_hint(f"stored {', '.join(names)}", style="green")
            print_hint(
                "use it: bench hosted run <version> "
                + " ".join(f"--secret {n}" for n in names)
                + " ..."
            )

    @secret_app.command("list")
    def secret_list(
        output_json: _Json = False, hub: _Hub = None, key_file: _KeyFile = None
    ) -> None:
        """Your workspace's stored keys: names, kinds and hints, never values."""
        with _service(hub, key_file) as client:
            items = client.list_secrets()
            if output_json:
                typer.echo(json.dumps(items))
                return
            for item in items:
                print_hint(
                    f"{item.get('name')}  {item.get('kind')}  {item.get('hint') or ''}".rstrip()
                )
            if not items:
                print_hint("no stored keys")

    @secret_app.command("rm")
    def secret_rm(
        name: Annotated[str, typer.Argument(help="The stored key's name")],
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Delete a stored key (an account key deletes what account keys stored)."""
        with _service(hub, key_file) as client:
            if not client.delete_secret(name):
                print_error(
                    f"no stored key named {name}",
                    next_step="run `bench hosted secret list` to see the names this "
                    "workspace keeps; nothing was deleted",
                )
                raise typer.Exit(1)
            if output_json:
                typer.echo(json.dumps({"deleted": name}))
                return
            print_hint(f"deleted {name}", style="green")

    @hosted_app.command("status")
    def hosted_status(
        run_id: _RunId,
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Show a run's status, its trials so far and its mean reward."""
        from benchflow.hosted import summary_line

        with _service(hub, key_file) as client:
            record = client.status(run_id)
            if output_json:
                typer.echo(json.dumps(record))
            else:
                print_hint(summary_line(record))
                _show_page(client, run_id)

    @hosted_app.command("wait")
    def hosted_wait(
        run_id: _RunId,
        timeout: _Timeout = None,
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Wait for a run to finish; exits 1 when it failed or was cancelled."""
        with _service(hub, key_file) as client:
            _wait(client, run_id, timeout, output_json)

    @hosted_app.command("results")
    def hosted_results(
        run_id: _RunId,
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Show a finished run's rewards: the summary, each reward key, and each trial."""
        with _service(hub, key_file) as client:
            results = client.results(run_id)
            if output_json:
                typer.echo(json.dumps(results))
            else:
                _print_results(client, run_id, results)

    @hosted_app.command("trajectories")
    def hosted_trajectories(
        run_id: _RunId,
        output: _Output,
        format: _Format = "atif",
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Save a run's trajectories, one JSON document per line, to a file."""
        with _service(hub, key_file) as client:
            path = client.trajectories(run_id, output, format=format)
            print_hint(f"wrote {path} ({path.stat().st_size} bytes)")

    @hosted_app.command("download")
    def hosted_download(
        run_id: _RunId,
        output: _Output,
        format: _Format = "atif",
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Save a run's trajectories to a file: `bench hosted trajectories`."""
        with _service(hub, key_file) as client:
            path = client.trajectories(run_id, output, format=format)
            print_hint(f"wrote {path} ({path.stat().st_size} bytes)")

    @hosted_app.command("events")
    def hosted_events(
        run_id: _RunId,
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Show what happened to a run, event by event."""
        with _service(hub, key_file) as client:
            events = client.events(run_id)
            if output_json:
                typer.echo(json.dumps(events))
                return
            for event in events:
                if isinstance(event, dict):
                    print_hint(
                        f"{event.get('ts') or ''} {event.get('kind')}: {event.get('message')}".strip()
                    )

    @hosted_app.command("logs")
    def hosted_logs(
        run_id: _RunId,
        tail: Annotated[
            int | None, typer.Option("--tail", help="Only the last N characters", min=1)
        ] = None,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Print a run's log."""
        with _service(hub, key_file) as client:
            typer.echo(client.logs(run_id, tail=tail), nl=False)

    @hosted_app.command("cancel")
    def hosted_cancel(
        run_id: _RunId,
        output_json: _Json = False,
        hub: _Hub = None,
        key_file: _KeyFile = None,
    ) -> None:
        """Ask the service to stop a run."""
        with _service(hub, key_file) as client:
            answer = client.cancel(run_id)
            if output_json:
                typer.echo(json.dumps(answer))
            elif answer.get("ok") is False:
                print_hint(
                    f"run {run_id} was not running; nothing to stop", style="yellow"
                )
            else:
                print_hint(f"asked the service to stop run {run_id}")
