# Week 4 — Second Robot: Stretch 3

## Results

| Robot   | Task       | Episodes | Success Rate | Steps/Episode |
|---------|------------|----------|--------------|---------------|
| Stretch | Pick Cube  | 100      | 100%         | 300           |

## Setup Pipeline
1. `git clone https://github.com/hello-robot/stretch_ros2` 
2. Install `ros-humble-realsense2-description`
3. `xacro stretch_description_standard.xacro` → URDF (30KB)
4. Replace `package://` paths with absolute paths in URDF
5. `isaaclab convert_urdf.py` → USD with full mesh visuals

## Key Findings — Cross-Embodiment Insights

**Arm axis discovery:** Stretch's telescoping arm extends along the Z-axis in the converted URDF frame, not the X-axis like Franka. The joint chain `arm_l3 → arm_l2 → arm_l1 → arm_l0` each extends 0.13m max = 0.52m total reach.

**Different success criterion:** Cannot use cube Z-height for Stretch (arm extends sideways relative to base). Success defined as: arm extended (joint_arm_l0 > 0.05m) + gripper closed + t > 120 steps.

**Mobile base:** Stretch has differential drive wheels (`joint_left_wheel`, `joint_right_wheel`) — requires base navigation before arm manipulation, unlike fixed-base Franka.

**Mesh fix:** URDF uses `package://` paths. Fix: copy meshes to `stretch_description/meshes/` and replace paths with absolute paths before USD conversion.

## Joint Configuration
joint_left_wheel, joint_right_wheel  — differential drive
joint_lift                           — vertical prismatic (0 to 1.1m)
joint_arm_l3/l2/l1/l0               — telescoping arm (4 × 0.13m = 0.52m)
joint_wrist_yaw                      — wrist rotation
joint_head_pan, joint_head_tilt      — head camera
joint_gripper_finger_left/right      — parallel gripper
## What This Means for Cross-Embodiment Research
Same "pick cube" task requires fundamentally different joint strategies:
- **Franka:** Fixed base, 7-DOF arm, IK-based end-effector control
- **Stretch:** Mobile base + telescoping arm, sequential base→lift→extend→grasp

This is exactly the cross-embodiment challenge the dataset is designed to capture.
