"""User-private local runtime management."""

import typer


def register_daemon(app: typer.Typer) -> None:
    daemon = typer.Typer(help="Manage the opt-in warm local runtime.")
    app.add_typer(daemon, name="daemon")

    @daemon.command()
    def start(
        prepare: int = typer.Option(
            0,
            "--prepare",
            min=0,
            max=4,
            help=(
                "Prepare up to N sandboxes ahead: after a one-task Docker run "
                "ends, the same command is made ready for its next identical "
                "call (0: off)."
            ),
        ),
        prepare_ttl: float = typer.Option(
            300.0,
            "--prepare-ttl",
            min=1.0,
            help="Seconds a prepared sandbox waits for its call before it is removed.",
        ),
    ) -> None:
        """Start a private runtime; attach runs with BENCHFLOW_DAEMON=auto."""
        from benchflow._daemon import RuntimeMismatch
        from benchflow._daemon import start as start_runtime

        try:
            start_runtime(prepare, prepare_ttl)
        except RuntimeMismatch as exc:
            from benchflow.cli._shared import print_error

            print_error(
                str(exc),
                next_step="`bench daemon stop` (with the installation that started it) or wait for it to stop, then start again",
            )
            raise typer.Exit(1) from None
        typer.echo("Local runtime ready. Set BENCHFLOW_DAEMON=auto for runs.")

    @daemon.command()
    def status() -> None:
        """Show whether this user's local runtime is running."""
        from benchflow._daemon import request, stopping_runtime

        try:
            state = request({"action": "status"})
        except (OSError, EOFError):
            draining = stopping_runtime()
            if draining is not None:
                typer.echo(
                    "Local runtime is stopping; it waits for its "
                    f"{draining.get('running', 'running')} running calls."
                )
                return
            from benchflow.cli._shared import print_error

            print_error(
                "Local runtime is not running.",
                next_step="`bench daemon start` starts it",
            )
            raise typer.Exit(1) from None
        typer.echo(
            f"Local runtime running; {state['workers']} active calls, "
            f"{state['parked']} prepared and {state['preparing']} preparing "
            f"(up to {state['prepare']})."
        )

    @daemon.command()
    def stop() -> None:
        """Stop an idle local runtime; never interrupt an active call."""
        from benchflow._daemon import request

        try:
            state = request({"action": "stop"})
        except (OSError, EOFError):
            from benchflow.cli._shared import print_error

            print_error(
                "Local runtime is not running.",
                next_step="`bench daemon start` starts it",
            )
            raise typer.Exit(1) from None
        if not state["stopping"]:
            from benchflow.cli._shared import print_error

            print_error(
                f"Local runtime is busy: {state.get('workers', 'some')} active calls.",
                next_step="wait for them to finish, then `bench daemon stop` again",
            )
            raise typer.Exit(1)
        typer.echo("Local runtime stopping.")
