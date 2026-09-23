"""Build self-contained native BenchFlow task packages for physical control."""

from __future__ import annotations

import json
from pathlib import Path

from . import client

SCENARIOS = {
    "sort-blue-left": {
        "instruction": "Put the blue block inside the left red cup. Put the green and yellow blocks inside the right red cup. Both cups must remain upright. Leave the gripper empty and clear of the cups.",
        "expected": {"blue": "left", "green": "right", "yellow": "right"},
    },
    "sort-green-left": {
        "instruction": "Put the green block inside the left red cup. Put the blue and yellow blocks inside the right red cup. Both cups must remain upright. Leave the gripper empty and clear of the cups.",
        "expected": {"green": "left", "blue": "right", "yellow": "right"},
    },
    "pick-yellow": {
        "instruction": "Pick up the yellow block and hold it with its bottom at least 5 cm above the table for 3 seconds. Finish while holding the block. Do not move the other objects.",
        "expected": {"yellow": "lifted_5cm_for_3s"},
    },
}

CONTROL_INSTRUCTIONS = """
You directly control a real, powered MakerMods Metal arm. Use the images to
locate the objects in this trial; no previous conversation or task solutions
are available. Left/right are from the arm looking forward from its resting
position. Read /app/setup.json for this setup's public calibration facts.

Use `python /app/robot.py observe` before any motion. Each response provides
the measured end-effector pose, joints, and two image paths. Open BOTH images
using your image-viewing tool after every motion, before commanding another.
Wrist view establishes alignment; side view checks height and clearance.
These are camera images, not simulator coordinates.

Commands (metres and degrees):
```
python /app/robot.py observe
python /app/robot.py status
python /app/robot.py tip X Y Z PITCH
python /app/robot.py roll DEGREES
python /app/robot.py gripper DEGREES
python /app/robot.py monitor SECONDS
python /app/robot.py finish
```
`tip` specifies an end-effector pose; the host harness's IK solver computes
the joints and executes the checked trajectory. Do not implement alternative
joint control. X is forward, Y left, Z up. Pitch 0 is horizontal and -80 is
nearly downward. Gripper 0 is closed and 112 fully open; approach with jaws
wider than the block, align flat faces, and avoid squeezing it out. `roll`
only rotates the wrist. `monitor` records encoders without commanding motion.

Map the destination with empty jaws before grasping. Approach above the
block, align from both cameras, then descend within the commissioned floor.
Close gently, check contact and make a short vertical lift before transport.
Raise above cup rims with allowance for the held object's full extent before
lateral travel. Check placement from both cameras before release. Lower
poses must respect the table floor in /app/setup.json; never widen limits.
An IK or per-move-limit rejection calls for a smaller, checked step.

Do not batch motions or run background control scripts. Stop if hands enter
the motion area, a camera is lost, vibration becomes concerning, or contact
is uncertain. A transport failure can mean the motion already executed:
never repeat it blindly. Tell the operator and stop. The human emergency
stop/power switch is independent of this software. `finish` captures a final
observation and revokes further control; it does not move or depower the arm.
Recording and scoring are host-owned. Report uncertainty honestly and call
`finish` once the task is complete or cannot safely proceed.
"""


def build_tasks(root: Path) -> list[Path]:
    paths = []
    for name, scenario in SCENARIOS.items():
        task = root / name
        for directory in (task / "environment", task / "verifier"):
            directory.mkdir(parents=True, exist_ok=True)
        (task / "task.md").write_text(
            '---\nschema_version: "1.3"\ntask:\n'
            f"  name: metal-real/{name}\n  description: Supervised real Metal arm manipulation\n"
            "metadata:\n  category: robotics\n  tags: [robotics, physical, llm-as-policy]\n"
            "agent:\n  timeout_sec: 1800\nverifier:\n  timeout_sec: 30\n"
            "sandbox:\n  cpus: 2\n  memory_mb: 4096\n---\n"
            f"# Physical Metal arm: {name}\n\n## prompt\n\n{scenario['instruction']}\n"
            + CONTROL_INSTRUCTIONS
        )
        (task / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
        (task / "environment" / "robot.py").write_text(
            Path(client.__file__).read_text()
        )
        (task / "environment" / "Dockerfile").write_text(
            "FROM python:3.12-slim-bookworm\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends "
            "ca-certificates curl git bash && rm -rf /var/lib/apt/lists/*\n"
            "WORKDIR /app\nCOPY robot.py /app/robot.py\n"
        )
        (task / "environment" / "docker-compose.yaml").write_text(
            'services:\n  main:\n    extra_hosts:\n      - "host.docker.internal:host-gateway"\n'
        )
        (task / "verifier" / "test.sh").write_text(
            "#!/bin/sh\nset -eu\n"
            'echo "Physical score requires host adjudication: use python -m benchflow.robotics score." >&2\n'
            'echo "No physical reward can be inferred from agent-written files." >&2\nexit 2\n'
        )
        (task / "verifier" / "verifier.md").write_text(
            '---\ndocument_version: "0.3"\nverifier:\n  name: physical-review\n'
            "  default_strategy: host-review\n  strategies:\n    host-review:\n"
            "      type: script\n      command: ./test.sh\n"
            "  outputs:\n    reward_text: /logs/verifier/reward.txt\n"
            "    reward_json: /logs/verifier/reward.json\n---\n# Physical evidence verifier\n\n"
            "Scoring is deferred to the host reviewer after the agent loses access. "
            "The robotics SDK intentionally uses skip_verify=True, preserves the native "
            "BenchFlow result, and writes assessment.json plus reward.txt in the host trial directory. "
            "Generic bench eval runs must not invent a score from agent-authored claims.\n"
        )
        paths.append(task)
    return paths
