---
schema_version: "1.3"
task:
  name: metal-real/t3-doll
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

stab the thing that's not the bread please

You directly control one real robot arm on a tabletop. Read /app/setup.json for the public facts: where the arm stands, its coordinate frame, reach, cameras, and safety rules. Use the images to locate objects; no prior conversation or solutions are available.

Run `python /app/robot.py observe` before moving. Each reply gives the measured end-effector pose, joints, and two images: the wrist camera and a side view. Open both images with your image tool after every motion, before the next command.

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
`tip` is an end-effector pose in the arm's base frame; the harness solves inverse kinematics and plays a checked, speed-limited trajectory. X is forward, Y is left, Z is up; pitch 0 is horizontal and -80 is nearly straight down. Gripper 0 is closed, 112 is open. `roll` turns the jaws about the tool axis. `monitor` records without moving.

Limits and safety: the harness refuses any motion within 10 mm of the table and refuses a joint move over 35 degrees in one command; a rejection means choose a smaller or different checked step, not a repeat. Move in small steps and look after each. A transport error may mean the motion already executed, so never blindly repeat it; stop if a camera is lost or contact is uncertain. `finish` takes a final observation and ends your control; it does not score anything. Report honestly what the cameras actually show.
