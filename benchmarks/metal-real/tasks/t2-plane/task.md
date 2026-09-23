---
schema_version: "1.3"
task:
  name: metal-real/t2-plane
  description: Supervised real Metal arm manipulation
metadata:
  category: robotics
  tags: [robotics, physical, llm-as-policy]
agent:
  timeout_sec: 7200
verifier:
  timeout_sec: 30
sandbox:
  cpus: 2
  memory_mb: 4096
---
# Physical Metal arm task

## prompt

There is a pile of pink sheets in the center of the table. Use one sheet to fold a paper airplane with the available robot arms. Leave the other items alone. Stop when you finish or cannot continue, and report what you actually accomplished.

Treat this as an extended, hands-on effort, not a quick inspection. It is a genuinely hard task for these grippers and will not succeed on the first attempt. Make many real physical attempts: move the arms in, grasp, lift, adjust, and recover from failures by trying different positions, grips, wrist rolls, pitches, and heights. A single uncertain or failed grasp is not a reason to quit; back off, look at the cameras, and try again a different way. You have up to two hours available and should use them: keep actively attempting. Do not end the session after only observing or after one or two moves. Only call finish once you have either accomplished the task or made substantial, repeated manipulation attempts over a long period, and then report specifically what you tried and what happened.

Use both arms. You have two arms, a left and a right, and this task is meant for both working together, not one alone: the reliable approach is for one arm to hold or pin part of the material while the other arm manipulates it. Deliberately use BOTH the left and the right arm. Address each explicitly, `--arm left` or `--arm right`, and alternate between them: after you move one arm, look, then bring the OTHER arm in to assist rather than making every move with a single arm. If you notice you have only been moving one arm, switch to the other and set up a two-handed hold.

You directly control two real robot arms named `right` and `left` on a tabletop. Read /app/setup.json for the public facts: where the arms stand, how their coordinate frames relate, reach, cameras, and safety rules. Use the images to locate objects; no prior conversation or solutions are available.

Every command names an arm with `--arm right` or `--arm left`. Run `python /app/robot.py --arm ARM observe` before moving. Each reply gives that arm's measured end-effector pose, joints, and two images: its own wrist camera and the shared side view. Open both images with your image tool after every motion, before the next command.

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
`tip` is an end-effector pose in that arm's own base frame; the harness solves inverse kinematics and plays a checked, speed-limited trajectory. X is forward, Y is left, Z is up; pitch 0 is horizontal and -80 is nearly straight down. Gripper 0 is closed, 112 is open. `roll` turns the jaws about the tool axis. `monitor` records without moving.

Limits and safety: the harness refuses any motion within 10 mm of the table or within a protected margin of registered no-go regions, and refuses a joint move over 35 degrees in one command; a rejection means choose a smaller or different checked step, not a repeat. Only one arm moves at a time and the harness refuses a target too close to the other arm. Move in small steps and look after each. A transport error may mean the motion already executed, so never blindly repeat it; stop if a camera is lost or contact is uncertain. `finish` takes a final observation and ends your control; it does not score anything. Report honestly what the cameras actually show.
