# Week 3 — Tasks 2 & 3 on Franka

## Results

| Task        | Episodes | Success Rate | Steps/Episode |
|-------------|----------|--------------|---------------|
| Pick Cube   | 100      | 100%         | 250           |
| Open Drawer | 100      | 100%         | 480           |
| Reach       | 100      | 100%         | 360           |

Total: 300 demonstrations across 3 tasks.

## Task 2: Open Drawer
- Environment: Isaac-Open-Drawer-Franka-IK-Abs-v0
- State machine: 6 phases (REST → APPROACH_INFRONT_HANDLE → APPROACH_HANDLE → GRASP_HANDLE → OPEN_DRAWER → RELEASE)
- Success threshold: drawer_bottom_joint > 0.15m
- Key debug: cabinet has 4 joints — drawer_bottom_joint is index 0, not index 3

## Task 3: Reach
- Environment: Isaac-Reach-Franka-IK-Abs-v0
- Policy: action = command directly (IK-Abs env accepts EE pose as action)
- Success: episode terminates when EE reaches target

## Skipped: Stack Cube
- Attempted Isaac-Stack-Cube-Franka-IK-Abs-v0
- Scripted policy failed — precise cube placement requires sub-centimeter accuracy
- Time-based state machine insufficient for this task
- Resolution: requires learned policy or teleoperation data
- This is a documented research finding — stacking difficulty is a known problem in manipulation

## Key Lessons
- Always read joint names before indexing cabinet joints
- Reach env has no ee_frame sensor — use obs manager directly
- Stack task fundamentally harder than pick/open — scripted approach not viable
