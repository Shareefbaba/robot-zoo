# Week 5 — Third Robot: Fetch

## Goal
Load Fetch robot, port pick-cube task, collect 100+ demos.

## What We Did

### Robot Setup
- Downloaded Fetch URDF from `fetchrobotics/fetch_ros` GitHub
- Fixed `package://` paths to absolute paths
- Converted URDF to USD using Isaac Lab URDF importer
- Created `envs/fetch_cfg.py` with proper actuator groups

### Joint Configuration (15 joints)
| Index | Joint | Role |
|-------|-------|------|
| 2 | torso_lift | Arm height |
| 5 | shoulder_pan | Base Y alignment |
| 7 | shoulder_lift | Arm forward/down |
| 9 | elbow_flex | Reach distance |
| 11 | wrist_flex | Gripper angle |
| 13,14 | gripper fingers | Grasp |

### Grasp Config Discovery (Y3)
Through systematic body position diagnostics:
- `slift=1.20, eflex=-0.9, wflex=1.15, torso=0.1`
- Gripper reaches x=0.603, z=0.055 — 3mm from cube center

### Navigation Strategy
Same P-controller approach as Stretch:
- Spawn robot 1m behind target: `target_rx = cx - 0.603`
- P-controller (Kp=1.5) drives base to target
- Then execute 4-phase arm sequence

### Pick Sequence
1. **Navigate** — P-controller to cube position
2. **Pre-grasp** — arm above cube (slift=1.0, eflex=-0.7)
3. **Lower** — to grasp height (slift=1.20, eflex=-0.9)
4. **Grasp** — close gripper
5. **Lift** — raise arm (slift=0.5, eflex=-0.3)

## Results
- **113 successful demonstrations**
- **Max cube height: ~0.20m**
- Cube randomization: cx±0.05, cy±0.05

## Cross-Embodiment Insights
| Robot | Base | Arm | Grasp Strategy |
|-------|------|-----|----------------|
| Franka | Fixed | 7-DOF revolute | IK to target pose |
| Stretch | Mobile | Telescoping prismatic | Extend in -Y direction |
| Fetch | Mobile | 7-DOF revolute | Forward-down shoulder config |

## Dataset Status
| Robot | Task | Episodes |
|-------|------|----------|
| Franka | Pick Cube | 100 |
| Franka | Open Drawer | 100 |
| Franka | Reach | 100 |
| Stretch | Pick Cube | 109 |
| Fetch | Pick Cube | 113 |
| **TOTAL** | | **522** |
