---
schema_version: "1.3"
task:
  name: metal-real/untie-knot
  description: Supervised real Metal arm manipulation with two arms
metadata:
  category: robotics
  tags: [robotics, physical, llm-as-policy, two-arm]
agent:
  timeout_sec: 3600
verifier:
  timeout_sec: 30
sandbox:
  cpus: 2
  memory_mb: 4096
---
# Physical Metal arms: untie-knot

## prompt

Untie the knot in the white paper tape measure lying on the table, using both
arms, so that the tape is a single open strip with no loop passing through it.
Leave the tape on the table and end with both grippers empty and both arms
raised. Keep going until it is untied or your time is up, and say honestly
what the cameras show. The knot is loose: one end of the tape passes through a
ring formed by the tape. The way that works with two hands: one arm pinches
the ring (a loop standing on edge) and lifts it a little, the other arm pinches
the strand that hangs or stands from it and pulls that strand out through the
ring; or one arm holds the ring still while the other drags the free end back
through it. The pink
sheet nearby is not part of the task, and neither is the doll in a basket at the
far right corner of the table: treat it as an obstacle and keep the right arm
away from that corner (see scene in setup.json).

You directly control two real, powered MakerMods Metal arms named `right`
and `left`. Read /app/setup.json for the public facts: where the arms stand,
how their coordinate frames relate, reach, contact behaviour and the two-arm
safety rules. Left/right there means the arm's name, not a direction. Use the
images to locate the tape; no previous conversation or solutions are
available.

Every command names an arm with `--arm right` or `--arm left`. Use
`python /app/robot.py --arm right observe` and `python /app/robot.py --arm left observe`
before any motion. Each response provides that arm's measured end-effector
pose, joints, and two image paths: its own wrist camera and the shared side
camera. Open the images using your image-viewing tool after every motion,
before commanding another. Wrist views establish alignment; the side view
checks height, clearance and how close the two grippers are.

Commands (metres and degrees; ARM is right or left):
```
python /app/robot.py --arm ARM observe
python /app/robot.py --arm ARM status
python /app/robot.py --arm ARM tip X Y Z PITCH
python /app/robot.py --arm ARM roll DEGREES
python /app/robot.py --arm ARM gripper DEGREES
python /app/robot.py --arm ARM monitor SECONDS
python /app/robot.py finish
```
`tip` specifies an end-effector pose in THAT arm's own base frame; the host
harness's IK solver computes the joints and executes the checked trajectory.
X is forward, Y left, Z up. Pitch 0 is horizontal and -80 is nearly downward.
Gripper 0 is closed and 112 fully open. `roll` rotates the jaws about the
tool axis: at roll 0 the jaws close along the arm's sideways direction, at
roll 90 along its forward direction. `monitor` records encoders without
commanding motion. Do not implement alternative joint control.

Two-arm rules: move one arm at a time and look at both wrist images after
each move. The bridge refuses a `tip` target within 20 cm horizontally of the
other arm's gripper, and the harness refuses anything within 10 mm of the
table; a rejection calls for a different, checked step, not a retry of the
same one. Never place one arm's tips under the other arm's gripper: each wrist
housing extends about 16 cm back from its tips. When one arm holds the tape in
the air and the other must grasp a hanging strand, keep the holding arm high
(tips 30 cm or more above the table) and approach the strand with the other
arm at a shallow pitch of about -50 from its own side, so its housing leans
away. A strand hanging in the air can be pinched when it is edge-on to the
jaws: adjust `roll` until the strand appears as a thin line perpendicular to
the jaw line in the wrist image, then close and test with a small pull.

Lessons from earlier attempts on this bench: (1) pinch the ring with the arm
whose base is nearest to it, approaching nearly vertically (pitch about -80)
from 6 cm with the jaws narrowed to about 45 so they straddle one wall of the
ring, and prove it with a 3 cm lift; a far, shallow reach across the table
rarely pinches. (2) Once the ring is held and lifted 25-30 cm, the strand's
lower part hangs to the table; with the other arm, pinch it where it hangs
vertically 3-6 cm above the table, not the loose end lying flat. Aim so the
strand crosses the jaw gap in the wrist image, set roll so it is edge-on to
the jaws, then close and test with a 3 cm pull. (3) A hanging strand's depth
is hard to judge from one camera: check it in the side view before closing.

Do not batch motions or run background control scripts. Stop if hands enter
the motion area, a camera is lost, vibration becomes concerning, or an arm
touches anything other than the tape. Touching and dragging the tape is
expected; a reply that says 'stopped early' with a wrist_yaw residual means the
tips are in contact with the table, which is normal: lift 3 cm and continue. A
transport failure can mean the motion already executed: never repeat it
blindly. Tell the operator and stop. The human emergency stop/power switch is
independent of this software. `finish` captures a final observation from both
arms and revokes further control; it does not move or depower the arms.
Recording and scoring are host-owned. Report uncertainty honestly and call
`finish` once the task is complete or cannot safely proceed.
