# Building Robot Zoo: An Open-Source Multi-Embodiment Robotics Benchmark

*How I built a 13,058-demonstration dataset across three structurally different robots in 8 weeks using Isaac Lab*

---

## The Problem That Started This

If you train a robot policy on a Franka Panda arm and deploy it on a Fetch robot, it fails completely — even for the exact same task. The joint configurations are different, the workspace is different, the base is different. This is the cross-embodiment problem, and it's one of the hardest open challenges in robot learning today.

Physical Intelligence, Google DeepMind, and NVIDIA are all working on it. But almost all the research uses proprietary data. Open-source benchmarks that span multiple robot embodiments are rare.

That's the gap Robot Zoo tries to fill.

---

## What is Robot Zoo?

Robot Zoo is an open-source multi-embodiment manipulation dataset built with Isaac Lab 0.54.3 and Isaac Sim 5.1. It contains **13,058 successful demonstrations** across **3 robots** and **5 tasks**.

| Task | Franka | Stretch | Fetch | Total |
|------|--------|---------|-------|-------|
| Pick Cube | 1000 | 1058 | 1000 | 3058 |
| Open Drawer | 1000 | — | — | 1000 |
| Reach | 1000 | 1000 | 1000 | 3000 |
| Place Cube | 1000 | 1000 | 1000 | 3000 |
| Push | 1000 | — | 1000 | 2000 |
| **Total** | **5000** | **3058** | **4000** | **13,058** |

The "—" entries are not oversights — they are documented failures. More on that later.

---

## The Three Robots

### Franka Panda — The Baseline

Franka is the cleanest starting point. Fixed base, 7-DOF revolute arm, and Isaac Lab has a built-in IK environment for it. I used a Warp GPU state machine for the scripted policy:

```python
class PickSmState:
    REST                  = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT       = wp.constant(2)
    GRASP_OBJECT          = wp.constant(3)
    LIFT_OBJECT           = wp.constant(4)
```

Success rate: ~100%. The IK environment handles all the hard geometry.

### Stretch 3 — The Hard One

Stretch 3 is Hello Robot's mobile manipulator. It has a telescoping prismatic arm that extends in the **-Y direction** — completely different from Franka's revolute joints. There's no Isaac Lab IK environment for it.

To make it work I had to:
1. Convert the URDF to USD manually
2. Discover through systematic body position diagnostics that the correct base offset is `cx - 0.227`
3. Write a P-controller for base navigation
4. Build a lookup table of joint configurations indexed by target height

The arm extends with four prismatic joints (`ARM_L0` through `ARM_L3`), each contributing 0.13m of reach. The gripper closes by setting the finger joints to -0.6.

Success rate: ~63%. Mobile manipulation is harder.

### Fetch — The Middle Ground

Fetch has a 7-DOF revolute arm like Franka but mounted on a mobile base with a torso lift joint. Similar to Franka in arm kinematics but completely different in how you position the robot.

The key discovery was the **Y3 grasp configuration** — found through 8 iterations of systematic joint diagnostics:
Base navigation uses the same P-controller pattern as Stretch.

Success rate: ~61%.

---

## What Cross-Embodiment Actually Means

Porting the pick task to three robots forced a concrete answer to an abstract question. Here's what the same task looks like across embodiments:

**Franka:** IK environment → Warp state machine → joint targets

**Stretch:** P-controller navigate base → extend four prismatic arm segments → close finger joints

**Fetch:** P-controller navigate base → slift=1.20, eflex=-0.9, wflex=1.15 → close gripper fingers

These are not variations of the same solution. They are fundamentally different control strategies for the same physical goal. A policy that works on one will not trivially transfer to another. That's the research problem, made concrete.

---

## Software Design

All three robots share a common interface:

```python
class BaseRobot(ABC):
    def get_base_target(self, cx, cy) -> tuple: ...
    def get_pregrasp_joint_target(self, cx, cy) -> torch.Tensor: ...
    def get_grasp_joint_target(self, cx, cy) -> torch.Tensor: ...
    def get_lift_joint_target(self, cx, cy) -> torch.Tensor: ...
    def get_num_joints(self) -> int: ...
```

`FrankaRobot`, `StretchRobot`, and `FetchRobot` each implement this interface. The demo generation scripts share their high-level structure — adding a fourth robot is roughly 50 lines of code.

12 unit tests verify the interface contract:
---

## Documented Failures

Several robot-task combinations were attempted and dropped:

**Stretch Open Drawer** — The telescoping arm can't reach far enough into the cabinet workspace. The prismatic joints run out of extension before the gripper contacts the handle.

**Stretch/Fetch Push** — Moving the base during contact with the cube causes instability. The robot pushes the cube but also pushes itself, making success unreliable.

**Stack Cube (all robots)** — Stacking requires precise placement that scripted policies can't reliably achieve. This needs a learned policy.

These are not embarrassing. They are the boundary conditions of scripted manipulation — exactly the kind of information that helps researchers decide where to invest effort in learning-based approaches.

---

## Technical Stack

- Isaac Sim 5.1.0 + Isaac Lab 0.54.3
- PyTorch 2.7.0 + Warp (GPU state machines)
- Python 3.11
- RTX 5060 Ti 16GB
- Ubuntu 22.04

Total compute: approximately 15 hours of headless simulation across all tasks.

---

## How to Use the Dataset

```python
from huggingface_hub import snapshot_download
import numpy as np
from pathlib import Path

path = snapshot_download(
    repo_id="shareef14/robot-zoo",
    repo_type="dataset"
)

# Load a Fetch pick episode
ep = np.load(Path(path) / "fetch_pick_cube" / "episode_0000.npz")

states  = ep["observation_state"]    # (T, 15) — joint pos + vel
obj_pos = ep["observation_cube_pos"] # (T, 3)  — cube world position
actions = ep["action"]               # (T, 15) — joint targets
rewards = ep["reward"]               # (T,)    — success signal
```

---

## What's Next

Robot Zoo is the first of five planned mini-projects toward a publishable contribution on cross-embodiment policy transfer:

1. **Robot Zoo** ← this post
2. Baseline policy training (BC, ACT, Diffusion Policy) on the dataset
3. Cross-embodiment transfer experiments — train on Franka, evaluate on Fetch
4. Video foundation model integration
5. Research paper

---

## Links

- **Dataset**: https://huggingface.co/datasets/shareef14/robot-zoo
- **GitHub**: https://github.com/Shareefbaba/robot-zoo
- **Demo Video**: https://youtu.be/tAKlP3oEv3Y
- **Portfolio**: https://shareefbaba.github.io

---

*Shareef Baba Shaik is a Robotics Engineer based in Hyderabad, India, specializing in ROS2, autonomous navigation, and robot learning.*

**Connect with me:**
- 🔗 LinkedIn: https://linkedin.com/in/shareefbaba
- 💻 GitHub: https://github.com/Shareefbaba
- 🌐 Portfolio: https://shareefbaba.github.io
- 🤗 HuggingFace: https://huggingface.co/shareef14
