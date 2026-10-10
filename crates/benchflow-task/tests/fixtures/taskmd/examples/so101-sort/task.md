Three wooden blocks (red, green, blue) are on the table in front of the arm, and three bins with matching colored tape are to its right. Put each block into the bin of its color within 5 minutes.

Use the `robo` command to control the arm and cameras. Move slowly near the bins; they are light and tip over easily.

```notes
Operator reset: return blocks to the three taped start squares in random color order (use the dice), set the bins upright on their outlines, then press the reset button. The harness refuses to start until the reset photo matches the start layout. No force limit is declared: the SO-101's servos report load, not force, so a force limit could be neither enforced nor measured.
```

```toml task
name = "examples/so101-sort"
title = "Sort colored blocks with a real SO-101 arm"
version = "3.2.0"
keywords = ["robotics", "real-hardware", "manipulation"]

[agent]
timeout = "15m"

[sandbox]
image = "ghcr.io/example/robouse-agent:2026.09"

[world]
kind = "physical"
profile = "so-101-bench@2"
interface = "robot-cli@2"
time = "real-time"

[world.embodiment]
cameras = ["overhead", "wrist"]
frames = "base_link; metres and degrees"
calibration = "required-before-run"

[world.setup]
objects = ["red block", "green block", "blue block", "three bins with colored tape"]
reset = "operator"
reset_check = "photo-match"
randomize = "roll a die to assign block colors to the three start squares"

[world.budget]
wall_s = 900
commands = 400

[world.safety]
guard = "hardware-server"
supervision = "operator-present"
estop = "required"
stop_if = ["person-in-workspace"]

[[world.safety.limit]]
quantity = "tcp_position"
frame = "base_link"
min = ["0.10 m", "-0.25 m", "0.02 m"]
max = ["0.45 m", "0.25 m", "0.30 m"]
enforced_by = "hardware-server"
measured_by = "forward-kinematics"
on_violation = "reject-command"

[[world.safety.limit]]
quantity = "tcp_speed"
max = "0.15 m/s"
enforced_by = "hardware-server"
measured_by = "joint-encoders"
on_violation = "clamp"

[world.record]
streams = ["overhead-video", "wrist-video", "joint-trace", "guard-ledger"]
required = ["overhead-video", "guard-ledger"]

[verifier.human]
assessors = 2
evidence = ["overhead-video#final", "overhead-video"]
pending_until_assessed = true
```
