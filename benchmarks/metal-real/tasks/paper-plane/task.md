---
schema_version: "1.3"
task:
  name: metal-real/paper-plane
  description: Supervised real Metal arm manipulation
metadata:
  category: robotics
  tags: [robotics, physical, llm-as-policy]
agent:
  timeout_sec: 3600
verifier:
  timeout_sec: 30
sandbox:
  cpus: 2
  memory_mb: 4096
---
# Physical Metal arm: paper-plane

## prompt

Fold the pink sheet of paper on the table into a paper plane. Keep going until it is a plane or your time is up; if a full plane proves impossible, make as many real folds as you can and say honestly what the cameras show.

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
nearly downward. Gripper 0 is closed and 112 fully open; flat paper cannot be pinched:
raise an edge or corner first (drag the sheet with closed tips so it buckles
against something, or slide a closed tip under a curled corner), then pinch
the raised part and prove the grasp with a small lift seen on camera. `roll`
only rotates the wrist. `monitor` records encoders without commanding motion.

Do not batch motions or run background control scripts. Stop if hands enter
the motion area, a camera is lost, vibration becomes concerning, or the arm
touches anything other than the paper. Touching and dragging the paper is
expected; a reply that says 'stopped early' with a wrist_yaw residual means the
tips are in contact with the sheet, which is normal: lift 3 cm and continue. A transport failure can mean the motion already executed:
never repeat it blindly. Tell the operator and stop. The human emergency
stop/power switch is independent of this software. `finish` captures a final
observation and revokes further control; it does not move or depower the arm.
Recording and scoring are host-owned. Report uncertainty honestly and call
`finish` once the task is complete or cannot safely proceed.
