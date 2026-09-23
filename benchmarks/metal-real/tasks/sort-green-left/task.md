---
schema_version: "1.3"
task:
  name: metal-real/sort-green-left
  description: Supervised real Metal arm manipulation
metadata:
  category: robotics
  tags: [robotics, physical, llm-as-policy]
agent:
  timeout_sec: 1800
verifier:
  timeout_sec: 30
sandbox:
  cpus: 2
  memory_mb: 4096
---
# Physical Metal arm: sort-green-left

## prompt

Put the green block inside the left red cup. Put the blue and yellow blocks inside the right red cup. Both cups must remain upright. Leave the gripper empty and clear of the cups.

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
