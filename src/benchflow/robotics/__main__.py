"""python -m benchflow.robotics — physical benchmark operator commands."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import random
import shutil
import time
from pathlib import Path

from .recording import RecordingSidecar
from .runner import (
    load_setup,
    provider_environment,
    report_trials,
    run_trial,
    score_trial,
)
from .tasks import build_tasks


def main():
    parser = argparse.ArgumentParser(description="BenchFlow physical robot experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Build native task packages")
    init.add_argument("output", type=Path)
    for name in ("run", "smoke", "probe"):
        command = sub.add_parser(name)
        command.add_argument("--setup", type=Path, required=True)
        command.add_argument("--task", dest="task_path", type=Path, required=True)
        command.add_argument("--output", dest="output_root", type=Path, required=True)
        command.add_argument("--reset-id", required=True)
        command.add_argument("--operator", required=True)
        command.add_argument("--agent", choices=["codex", "claude"], default="codex")
        command.add_argument("--backend", choices=["docker", "host"], default="docker")
        command.add_argument(
            "--model", default="" if name == "smoke" else None, required=name != "smoke"
        )
        command.add_argument("--reasoning-effort", default="max")
        command.add_argument("--env-file", type=Path)
        command.add_argument("--timeout", type=int, default=1800)
        command.add_argument("--bind", default="127.0.0.1")
        command.add_argument("--advertised-host", default="host.docker.internal")
        if name == "run":
            command.add_argument(
                "--execute",
                action="store_true",
                required=True,
                help="Operator attests the physical reset is ready and authorizes trial motion",
            )
    score = sub.add_parser("score")
    score.add_argument("trial", type=Path)
    score.add_argument(
        "--placement", action="append", required=True, help="BLOCK=observed_location"
    )
    score.add_argument("--reviewer", required=True)
    score.add_argument("--interventions", type=int, required=True)
    score.add_argument(
        "--external-interruption",
        help="Observed external event that invalidates comparison scoring",
    )
    score.add_argument("--cups-upright", choices=["yes", "no"], required=True)
    score.add_argument(
        "--evidence", required=True, help="Reviewed camera files and time ranges"
    )
    report = sub.add_parser("report")
    report.add_argument("root", type=Path)
    report.add_argument("--csv", action="store_true")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--env-file", type=Path)
    record = sub.add_parser(
        "record", help="Record both cameras independently of any agent or BenchFlow run"
    )
    record.add_argument("--setup", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    schedule = sub.add_parser("plan")
    schedule.add_argument("--models", type=Path, required=True)
    schedule.add_argument("--tasks", nargs="+", required=True)
    schedule.add_argument("--setups", nargs="+", required=True)
    schedule.add_argument("--repetitions", type=int, default=3)
    schedule.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps([str(p) for p in build_tasks(args.output)], indent=2))
    elif args.command in {"run", "smoke", "probe"}:
        options = vars(args).copy()
        mode = options.pop("command")
        options["smoke"] = mode == "smoke"
        options["probe"] = mode == "probe"
        options["setup_path"] = options.pop("setup")
        options["allow_motion"] = options.pop("execute", False)
        print(asyncio.run(run_trial(**options)))
    elif args.command == "score":
        placements = {}
        for item in args.placement:
            key, separator, value = item.partition("=")
            if not separator or not value or key in placements:
                parser.error("Placements must be unique BLOCK=LOCATION entries")
            placements[key] = value
        print(
            json.dumps(
                score_trial(
                    args.trial,
                    placements=placements,
                    reviewer=args.reviewer,
                    interventions=args.interventions,
                    cups_upright=args.cups_upright == "yes",
                    evidence=args.evidence,
                    external_interruption=args.external_interruption,
                ),
                indent=2,
            )
        )
    elif args.command == "report":
        rows = report_trials(args.root)
        if args.csv and rows:
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer, fieldnames=sorted({key for row in rows for key in row})
            )
            writer.writeheader()
            writer.writerows(rows)
            print(buffer.getvalue(), end="")
        else:
            print(json.dumps(rows, indent=2))
    elif args.command == "record":
        sidecar = RecordingSidecar(
            args.output.resolve(), load_setup(args.setup)["cameras"]
        )
        try:
            sidecar.start()
            print(
                f"Recording both cameras to {args.output.resolve()}. Create its STOP file or press Ctrl-C to finish.",
                flush=True,
            )
            while (
                sidecar.process is not None
                and sidecar.process.poll() is None
                and not (sidecar.output / "STOP").exists()
            ):
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            summary = sidecar.stop()
            print(json.dumps(summary, indent=2))
        if not summary.get("complete") or not all(
            summary.get("exports", {}).get(name, {}).get("ok", False)
            for name in ("wrist", "side")
        ):
            raise SystemExit("Recording incomplete; preserve the output for inspection")
    elif args.command == "doctor":
        variables = provider_environment(args.env_file)
        print(
            json.dumps(
                {
                    "executables": {
                        name: shutil.which(name)
                        for name in ("docker", "ffmpeg", "bench", "codex", "claude")
                    },
                    "credential_variable_names": sorted(variables),
                    "codex_login_file_present": (
                        Path.home() / ".codex/auth.json"
                    ).exists(),
                    "note": "Presence is not authentication validation. Exact-model API probes and Docker smoke trials are required before a campaign.",
                },
                indent=2,
            )
        )
    elif args.command == "plan":
        models = json.loads(args.models.read_text())
        if (
            args.repetitions < 1
            or not models
            or any(not model.get("model") for model in models)
        ):
            parser.error(
                "Positive repetitions and exact model IDs are required; no model substitutions"
            )
        rng = random.Random(args.seed)
        rows = []
        for repetition in range(args.repetitions):
            blocks = [(setup, task) for setup in args.setups for task in args.tasks]
            rng.shuffle(blocks)
            for setup, task in blocks:
                order = list(models)
                rng.shuffle(order)
                for model in order:
                    rows.append(
                        {
                            "slot": len(rows) + 1,
                            "setup_id": setup,
                            "task": task,
                            "repetition": repetition + 1,
                            **model,
                            "status": "needs_physical_reset",
                        }
                    )
        print(
            json.dumps(
                {"seed": args.seed, "concurrency_per_arm": 1, "trials": rows}, indent=2
            )
        )


if __name__ == "__main__":
    main()
