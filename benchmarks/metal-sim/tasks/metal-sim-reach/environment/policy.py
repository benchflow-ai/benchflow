"""Your controller for the simulated MakerMods Metal arm.

BenchFlow runs this file through the simulator: ``act`` is called once per control step
(10 Hz) and must return the seven absolute joint targets in DEGREES, in this order:

    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper

``observation`` is a dict:
    joint_pos   [7]  current joint angles (deg); zero = the vendor's parked pose (upper arm
                     horizontal pointing backward, forearm folded forward, gripper ~19 cm up)
    eef_pos     [3]  fingertip point in metres (base at origin, +x forward, +y left, +z up)
    target_pos  [3]  centre of the red cube (m); it rests on the table (z = cube_size / 2)
    goal             "reach" or "pick"
    grasped          True once the cube is held (pick tasks only)
    instruction      the natural-language task text
    step             steps taken so far in this episode
    labels           the seven joint names above

The controller clamps every command to the soft limits and to 3 deg of travel per step
(9 deg for the gripper), so returning a far-away target moves the arm toward it gradually.
Optional: ``def reset(scene: dict)`` runs at the start of every episode.

Try it locally inside the environment:
    metal-sim-eval --task metal-sim-reach --policy-file /app/policy.py -T num_scenes=4 --no-frames
"""


def reset(scene: dict) -> None:
    pass


def act(observation: dict) -> list[float]:
    # Placeholder: hold the current pose. Replace with your controller.
    return list(observation["joint_pos"])
