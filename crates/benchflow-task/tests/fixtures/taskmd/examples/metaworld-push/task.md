A robot arm stands at a table with a puck and a green target marker. Push the puck onto the target.

You control the arm with the `robo` command: `robo observe` prints the gripper and object positions (and a camera image with `--image`), `robo act dx dy dz grip` moves the gripper, and `robo done` ends the episode. You have 300 steps; finishing in 150 or fewer earns extra credit.

```toml task
name = "examples/metaworld-push"
title = "Push a puck to a target (simulated arm)"
version = "3.1.0"
keywords = ["robotics", "simulation", "manipulation", "agent-as-policy"]

[agent]
timeout = "30m"

[sandbox]
image = "ghcr.io/example/robouse-agent:2026.09"

[world]
kind = "simulated"
profile = "metaworld@3.0.0"
scene = "push-v3"
interface = "robo-cli@1"
time = "decision-synchronous"

[world.episode]
success = "first"
public_seeds = [7]
hidden_seeds = 5

[world.budget]
sim_steps = 300

[[world.safety.limit]]
quantity = "contact_force"
max = "40 N"
enforced_by = "episode-server"
measured_by = "simulator"
proxy = true
on_violation = "end-episode"

[world.record]
streams = ["video", "state-trace"]
required = ["state-trace"]
```
