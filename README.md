# Robot Zoo

A multi-embodiment robotics dataset built in Isaac Lab. 13,058 scripted demonstrations across 3 robots and 5 manipulation tasks, released openly so cross-embodiment research has something concrete to work with.

[![Dataset on HuggingFace](https://img.shields.io/badge/🤗%20Dataset-shareef14%2Frobot--zoo-yellow)](https://huggingface.co/datasets/shareef14/robot-zoo)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-0.54.3-green)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Why this exists

Cross-embodiment robotics — taking a policy trained on one robot and deploying it on another — is a hard, open problem. Most public robot learning datasets stick to a single embodiment. That makes it difficult to study what actually transfers across robots and what doesn't.

Robot Zoo is a small step toward fixing that. It contains the same five manipulation tasks performed on three structurally different robots: a fixed-base 7-DOF arm (Franka Panda), a mobile manipulator with a telescoping arm (Stretch 3), and a wheeled robot with a revolute arm and torso lift (Fetch). Every demonstration is recorded in the same observation/action format so direct cross-robot comparisons are easy to set up.

## Dataset at a glance

| Task         | Franka | Stretch | Fetch | Total |
|--------------|--------|---------|-------|-------|
| Pick Cube    | 1000   | 1058    | 1000  | 3058  |
| Open Drawer  | 1000   | —       | —     | 1000  |
| Reach        | 1000   | 1000    | 1000  | 3000  |
| Place Cube   | 1000   | 1000    | 1000  | 3000  |
| Push         | 1000   | —       | 1000  | 2000  |
| **Total**    | **5000** | **3058** | **4000** | **13,058** |

Combinations marked with "—" were attempted and dropped because the embodiment couldn't reasonably perform the task with a scripted policy. Those failures are documented in the per-week notes — they are useful research data in their own right.

## Robots

| Robot         | Base       | Arm                       | DOF | Why it's interesting                            |
|---------------|------------|---------------------------|-----|-------------------------------------------------|
| Franka Panda  | Fixed      | 7-DOF revolute            | 7   | Clean baseline, IK environment available        |
| Stretch 3     | Mobile     | 4-link telescoping prismatic | 8 | Completely different kinematics from a typical arm |
| Fetch         | Mobile     | 7-DOF revolute + torso lift | 7  | Mobile base + revolute arm — the hardest combo  |

## Tasks

- **Pick Cube** — grasp a 5cm cube and lift it above a height threshold.
- **Open Drawer** — pull a cabinet drawer open until the prismatic joint clears 0.15 m.
- **Reach** — drive the end-effector to a randomly placed 3D target sphere.
- **Place Cube** — pick the cube and release it at a different position on the surface.
- **Push** — push the cube horizontally past a displacement threshold.

Each task has randomized initial conditions (cube position, target pose, etc.) so the demonstrations cover a real distribution rather than a single scripted trajectory.

## Data format

Every episode is a single `.npz` file containing:

| Key                     | Shape                         | Description                                |
|-------------------------|-------------------------------|--------------------------------------------|
| `observation_state`     | `(T, joint_dim)`              | Joint positions and velocities per step    |
| `observation_cube_pos`  | `(T, 3)`                      | World-frame position of the manipulated object (or target sphere for reach) |
| `action`                | `(T, action_dim)`             | Joint targets sent to the robot            |
| `reward`                | `(T,)`                        | Per-step success signal                    |

Per-task metadata (robot name, task name, total episodes, simulator version) is stored in a `metadata.json` next to the episode files.

## Quickstart — load the dataset

```python
from huggingface_hub import snapshot_download
import numpy as np
from pathlib import Path

path = snapshot_download(repo_id="shareef14/robot-zoo", repo_type="dataset")

ep = np.load(Path(path) / "fetch_pick_cube" / "episode_0000.npz")
print(ep["observation_state"].shape)    # (T, 15)  — 15 joints for Fetch
print(ep["observation_cube_pos"].shape) # (T, 3)
print(ep["action"].shape)               # (T, 15)
```

## Reproducing the dataset

The whole pipeline runs locally on a single GPU. The scripted policies are deterministic given a seed, but cube and target positions are randomized per episode.

### Requirements

- Ubuntu 22.04
- NVIDIA GPU with at least 8 GB of VRAM (developed on RTX 5060 Ti)
- CUDA 12.x driver
- Conda

### Setup

```bash
git clone https://github.com/Shareefbaba/robot-zoo.git
cd robot-zoo

conda create -n isaaclab python=3.11 -y
conda activate isaaclab

pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh --install
```

### Generate demonstrations

```bash
cd ~/IsaacLab

# Franka pick cube — 100 successful demos
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos.py --num_demos 100 --headless

# Stretch pick cube
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_stretch_pick.py --num_demos 100 --headless

# Fetch pick cube
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_fetch_pick.py --num_demos 100 --headless
```

Episodes are saved into `data/demos/<robot>_<task>/episode_XXXX.npz`.

## Repository layout
robot_zoo/
├── assets/         # Stretch and Fetch USD files
├── configs/        # Per-task YAML configs (Franka, Stretch, Fetch)
├── data/demos/     # Generated demonstrations (one folder per task-robot pair)
├── docs/           # Per-week development notes (week03–week06)
├── envs/           # BaseRobot abstraction + per-robot configs
├── notebooks/      # 12 visualization notebooks (one per task-robot pair)
├── scripts/        # Demo generation scripts
└── tests/          # Unit tests for the BaseRobot interface
## Design notes

**A common interface for all three robots.** All robots inherit from a `BaseRobot` abstract class with five methods: `get_num_joints`, `get_base_target`, `get_pregrasp_joint_target`, `get_grasp_joint_target`, and `get_lift_joint_target`. Generation scripts share their high-level structure regardless of which robot they're driving — adding a fourth robot to the codebase is roughly 50 lines.

**Two scripting patterns.** Franka tasks use Isaac Lab's existing IK environment plus a Warp state machine running on the GPU. Stretch and Fetch tasks bypass IK entirely: the base is teleported into position via a P-controller, then the arm runs a small lookup table of joint configurations tuned for different target heights. This was the practical way to deal with the very different arm kinematics.

**Honest about failures.** Several robot–task pairs were dropped after a few hours of attempts: Stretch can't reach far enough into the cabinet to open the drawer, mobile-base push is unstable because the base moves during contact, and stacking needs a learned policy. Those decisions are documented rather than hidden.

## Limitations

- Policies are scripted, not learned. This is a dataset paper, not a method paper.
- Single environment per task. No domain randomization on lighting, textures, or friction yet.
- No camera observations in the released format — only proprioception and object pose. Adding RGB rollouts is on the roadmap.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use Robot Zoo in your research:

```bibtex
@misc{shaik2026robotzoo,
  author = {Shaik, Shareef Baba},
  title  = {Robot Zoo: A Multi-Embodiment Manipulation Dataset},
  year   = {2026},
  url    = {https://huggingface.co/datasets/shareef14/robot-zoo}
}
```

## Author

**Shareef Baba Shaik** — Robotics Engineer, Hyderabad, India
[GitHub](https://github.com/Shareefbaba) · [LinkedIn](https://linkedin.com/in/shareefbaba)
