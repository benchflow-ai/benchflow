---
schema_version: "1.3"
task: {name: metal-real/table-reset, description: Restore the table to its home layout}
metadata: {category: robotics, tags: [robotics, physical, rearrangement, reset]}
agent: {timeout_sec: 3600}
verifier: {timeout_sec: 30}
sandbox: {cpus: 2, memory_mb: 4096}
---
# Physical Metal arm task

## prompt

Table reset. Return each item to its home position so the table matches its starting layout: the doll basket at the right edge, the pink sheets stacked in the center, the white tape at the left. Move only what is out of place, gently. Stop when the layout is restored or you cannot move an item further, and report what you did.


You directly control two real robot arms named `right` and `left` on a tabletop. Read /app/setup.json for the public facts: where the arms stand, how their coordinate frames relate, reach, cameras, and safety rules. Use the images to locate objects.

Use both arms. This is a two-handed job: one arm can steady or hold an item while the other moves it. Address each arm with `--arm right` or `--arm left`, alternate between them, and after moving one arm, look and bring the other in to assist rather than working with a single arm.

Run `python /app/robot.py --arm ARM observe` before moving; open both returned images after every motion. Commands (metres and degrees; ARM is right or left):
```
python /app/robot.py --arm ARM observe
python /app/robot.py --arm ARM tip X Y Z PITCH
python /app/robot.py --arm ARM roll DEGREES
python /app/robot.py --arm ARM gripper DEGREES
python /app/robot.py finish
```
`tip` is an absolute end-effector pose in that arm's base frame; X forward, Y left, Z up; pitch 0 horizontal, -80 nearly straight down; gripper 0 closed, 112 open. The harness refuses a move within 10 mm of the table, within a protected region, or that brings the arms too close; a rejection means try a smaller or different step. Move in small steps and look after each. `finish` takes a final observation and ends control.
