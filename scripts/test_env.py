# robot_zoo/scripts/test_env.py
# Launches the environment and steps it with random actions

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Franka Pick env")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- imports must come AFTER sim starts --
import sys
import torch
sys.path.insert(0, "/home/shareef")

from isaaclab.envs import ManagerBasedRLEnv
from robot_zoo.envs.franka_pick import FrankaPickEnvCfg


def main():
    cfg = FrankaPickEnvCfg()
    env = ManagerBasedRLEnv(cfg=cfg)

    print("\n✅ Environment created!")
    print(f"   Obs space  : {env.observation_space}")
    print(f"   Act space  : {env.action_space}")

    obs, _ = env.reset()
    print(f"   Obs shape  : {obs['policy'].shape}")

    print("\n⏳ Running 100 steps with random actions...")
    for step in range(100):
        action = torch.randn(env.num_envs, env.action_space.shape[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 20 == 0:
            print(f"   Step {step:3d} | reward: {reward.item():.4f} | done: {terminated.item()}")

    print("\n✅ Week 1 environment works!")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
